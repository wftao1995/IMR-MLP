// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtp_adjustment.h"
#include "mtp_model_internal.h"
#include "utils/temporary_directory.h"
#include "model_bundle.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_descriptor_linear.h"
#include "potentials/potential_eam_poly.h"
#include "potentials/potential_zbl.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_hybrid.h"
#include "potentials/potential_hybrid_scale.h"
#include "tools/chem.h"
#include <algorithm>
#include <map>
#include <random>

namespace Potentials::MTP
{
using namespace detail;
using GPM::require;

void adjust_legacy(PotMTP &pot, const Adjustment &operation)
{
    if (operation.order.size() != 0)
    {
        if (operation.order.size() > static_cast<size_t>(pot.species_count))
        {
            std::cout
                << "Add new species into potential, this part of coefs will be initialized randomly"
                << std::endl;
        }
        const auto old_names = pot.potential_tag.elements();
        if (operation.order.size() > static_cast<size_t>(pot.species_count) && !old_names.empty())
        {
            if (operation.elements.size() != operation.order.size())
                GPERROR("Tagged species expansion requires --elements for every output species");
            for (size_t i = 0; i < operation.order.size(); ++i)
            {
                const int source = operation.order[i];
                if (source >= 0 && source < pot.species_count &&
                    operation.elements[i] != old_names[source])
                    GPERROR("Output element order conflicts with existing species identities");
            }
            MTPMetadata::Tag checked;
            checked.set_elements(operation.elements, static_cast<int>(operation.order.size()));
            pot.potential_tag.fields.erase("elements");
        }
        pot.adjust_species(operation.order);
    }
    if (!operation.elements.empty())
    {
        std::map<int, std::string> mapping;
        for (size_t i = 0; i < operation.elements.size(); ++i)
            mapping[static_cast<int>(i)] = operation.elements[i];
        pot.set_element_map(mapping);
    }
    Potentials::MTP::validate_graph(pot);
    pot.ReorderByLevel();
}

void save_legacy_adjusted(PotMTP &pot, const std::string &output)
{
    const auto target = std::filesystem::absolute(output).lexically_normal();
    Utils::TemporaryDirectory stage(target.parent_path());
    const auto temporary = (stage.path / target.filename()).string();
    pot.SaveOrdered(temporary);
    Potentials::PotMTP check(temporary);
    if (check.inited() != pot.inited())
        GPERROR("Adjusted model failed native reload validation");
    std::filesystem::rename(temporary, target);
}

void validate_graph(const Potentials::PotMTP &model)
{
    if (model.alpha_moments_count < model.alpha_index_basic_count ||
        model.alpha_index_basic_count < 0)
        GPERROR("Invalid moment graph dimensions");
    std::vector<bool> defined(model.alpha_moments_count, false);
    for (int i = 0; i < model.alpha_index_basic_count; ++i)
        defined[i] = true;
    for (int i = 0; i < model.alpha_index_times_count; ++i)
    {
        const int a = model.alpha_index_times(i, 0), b = model.alpha_index_times(i, 1),
                  target = model.alpha_index_times(i, 3);
        if (a < 0 || b < 0 || target < model.alpha_index_basic_count ||
            target >= model.alpha_moments_count || a >= target || b >= target || !defined[a] ||
            !defined[b])
            GPERROR("Invalid or cyclic moment calculation order");
        defined[target] = true;
    }
    for (int i = 0; i < model.alpha_scalar_moments; ++i)
    {
        int id = model.alpha_moment_mapping[i];
        if (id < 0 || id >= model.alpha_moments_count || !defined[id])
            GPERROR("Invalid scalar moment mapping");
    }
}

namespace
{
std::vector<int> identity(int n)
{
    std::vector<int> result(n);
    std::iota(result.begin(), result.end(), 0);
    return result;
}

struct Species
{
    int old_count;
    std::vector<int> source;
    std::vector<std::string> names;
    std::vector<bool> fresh;
};

Species species_plan(AnyPotential &root, const Adjustment &op)
{
    Species plan;
    plan.old_count = root.species_count;
    plan.source = identity(plan.old_count);
    plan.names.resize(plan.old_count);
    for (const auto &[i, name] : model_element_map(root))
        plan.names.at(i) = name;
    if (!op.appended.empty())
    {
        require(op.initialization == "clone" || op.initialization == "random",
                "Specify clone or random species initialization");
        auto sources = op.sources;
        if (sources.empty() && op.initialization == "random")
            sources.assign(op.appended.size(), 0);
        require(sources.size() == op.appended.size(),
                "Provide one original clone source per new element");
        for (size_t i = 0; i < sources.size(); ++i)
        {
            require(sources[i] >= 0 && sources[i] < plan.old_count,
                    "Clone source must be an original species ID");
            plan.source.push_back(sources[i]);
            plan.names.push_back(op.appended[i]);
        }
        require(std::none_of(plan.names.begin(), plan.names.end(),
                             [](auto &s)
                             {
                                 return s.empty();
                             }),
                "Declare the existing element map before appending named elements");
    }
    else
        require(op.sources.empty() && op.initialization.empty(),
                "Species initialization requires appended elements");
    plan.fresh.resize(plan.source.size());
    for (size_t i = plan.old_count; i < plan.fresh.size(); ++i)
        plan.fresh[i] = true;
    if (!op.order.empty())
    {
        require(op.order.size() == plan.source.size(),
                "Element permutation must include every output species");
        std::set<int> seen;
        auto sources = plan.source;
        auto names = plan.names;
        auto fresh = plan.fresh;
        for (size_t i = 0; i < op.order.size(); ++i)
        {
            const int id = op.order[i];
            require(id >= 0 && id < int(sources.size()) && seen.insert(id).second,
                    "Invalid element permutation");
            plan.source[i] = sources[id];
            plan.names[i] = names[id];
            plan.fresh[i] = fresh[id];
        }
    }
    if (!op.elements.empty())
    {
        require(op.elements.size() == plan.names.size(), "Output element count mismatch");
        for (size_t i = 0; i < plan.names.size(); ++i)
            require(plan.names[i].empty() || plan.names[i] == op.elements[i],
                    "Output elements cannot rename existing species");
        plan.names = op.elements;
    }
    if (std::none_of(plan.names.begin(), plan.names.end(),
                     [](auto &s)
                     {
                         return s.empty();
                     }))
    {
        MTPMetadata::Tag checked;
        checked.set_elements(plan.names, plan.names.size());
    }
    return plan;
}

std::vector<double> expand(const std::vector<double> &old, const Species &plan, bool pair,
                           bool random, std::mt19937 &rng)
{
    const int s = plan.old_count, n = plan.source.size(), groups = pair ? s * s : s;
    require(groups > 0 && old.size() % groups == 0,
            "Invalid species-dependent parameter dimensions");
    const int stride = old.size() / groups;
    std::vector<double> result((pair ? n * n : n) * stride);
    std::uniform_real_distribution<double> uniform(-.1, .1);
    if (pair)
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < stride; ++k)
                    result[(i * n + j) * stride + k] =
                        random && (plan.fresh[i] || plan.fresh[j])
                            ? uniform(rng)
                            : old[(plan.source[i] * s + plan.source[j]) * stride + k];
    else
        for (int c = 0; c < stride; ++c)
            for (int i = 0; i < n; ++i)
                result[c * n + i] =
                    random && plan.fresh[i] ? uniform(rng) : old[c * s + plan.source[i]];
    return result;
}

using BasisPermutations = std::map<std::string, std::vector<int>>;

std::vector<double> permute_scalar(const double *old, int count,
                                   const std::vector<int> &permutation)
{
    if (count == 0)
        return {};
    const int scalars = permutation.size();
    require(scalars > 0 && count % scalars == 0, "Invalid scalar-channel layout");
    std::vector<double> result(count);
    for (int offset = 0; offset < count; offset += scalars)
        for (int i = 0; i < scalars; ++i)
            result[offset + i] = old[offset + permutation[i]];
    return result;
}

void copy_common(AnyPotential &next, AnyPotential &old, const Species &plan)
{
    const int n = plan.source.size();
    auto matrix = [&](const Eigen::MatrixXd &source)
    {
        if (source.size() == 0)
            return Eigen::MatrixXd();
        require(source.rows() == plan.old_count && source.cols() == plan.old_count,
                "Invalid species matrix dimensions");
        Eigen::MatrixXd result(n, n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                result(i, j) = source(plan.source[i], plan.source[j]);
        return result;
    };
    next.species_count = n;
    next.cutoff = old.cutoff;
    next.cutoffsq = old.cutoffsq;
    next.cutoffs = matrix(old.cutoffs);
    next.cutoffsqs = next.cutoffs.array().square().matrix();
    next.r_mins = matrix(old.r_mins);
    next.add_efs = old.add_efs;
    next.reg_L1_vector.clear();
    next.reg_L2_vector.clear();
    next.gpm_source.reset();
    next.gpm_elements.clear();
    for (int i = 0; i < n; ++i)
        if (!plan.names[i].empty())
            next.gpm_elements[i] = plan.names[i];
    if (auto *mtp = dynamic_cast<PotMTP *>(&next))
        mtp->potential_tag.fields.erase("elements");
    if (model_children(next).empty())
        next.set_element_map(next.gpm_elements);
}

std::unique_ptr<AnyPotential> morph(AnyPotential &old, const std::string &id, const Adjustment &op,
                                    const Species &plan, std::mt19937 &rng,
                                    BasisPermutations &permutations)
{
    const auto type = ModelBundle::type(old);
    const int s = plan.old_count, n = plan.source.size();
    const bool random = op.initialization == "random";
    std::map<std::string, std::unique_ptr<AnyPotential>> children;
    for (const auto &[role, child] : model_children(old))
        children[role] = morph(*child, id + "/" + role, op, plan, rng, permutations);
    std::unique_ptr<AnyPotential> result;
    std::vector<int> permutation;
    auto expand_block = [&](const double *data, int count, bool pair = false)
    {
        return expand(std::vector<double>(data, data + count), plan, pair, random, rng);
    };
    if (auto *source = dynamic_cast<PotMTP *>(&old))
    {
        validate_graph(*source);
        require(type != "mtp_m" || source->supports_species_append(),
                "MTP-M correction parameters have no defined migration adapter");
        auto model = copy_mtp(*source);
        auto &mtp = *model;
        // Reuse the established pair-basis append/permutation rules. This owned
        // ordinary copy supplies only the radial bases, never a replacement head.
        PotMTP bases(*source);
        bases.reg_L1_vector.clear();
        bases.reg_L2_vector.clear();
        bases.potential_tag.fields.erase("elements");
        auto sources = op.sources;
        if (sources.empty())
            sources.assign(op.appended.size(), 0);
        if (!sources.empty())
            bases.append_species(sources);
        if (!op.order.empty())
            bases.adjust_species(op.order);
        std::swap(mtp.p_radical_basis, bases.p_radical_basis);
        mtp.p_radical_basis_pair.swap(bases.p_radical_basis_pair);
        auto radial = expand_block(source->radial_coeffs, source->radial_coeff_size, true);
        auto species = expand_block(source->species_coeffs, source->species_coeff_size);
        int model_count = 0;
        if (type == "mtp_m")
        {
            int r, sp, m;
            source->get_n_coeffs(r, sp, m, model_count);
        }
        const int old_tail = source->radial_coeff_size + source->species_coeff_size +
                             source->moment_coeff_size + model_count;
        const int tail = source->reg_coeff_size - old_tail;
        const int channels = tail / (s + source->alpha_scalar_moments);
        std::vector<double> latent_species;
        if (tail)
            latent_species = expand_block(source->regression_coeffs + old_tail, channels * s);
        auto *les = dynamic_cast<PotMTP_LES *>(&mtp);
        auto *old_les = dynamic_cast<PotMTP_LES *>(source);
        std::vector<double> separate_species;
        if (les && les->lr_les)
            separate_species =
                expand_block(old_les->les_species_coeffs, old_les->latent_species_count());
        permutation = identity(mtp.alpha_scalar_moments);
        if (op.basis == "level")
            mtp.ReorderByLevel(permutation);
        else if (op.basis == "level-group")
            mtp.ReorderByLevelGroup(permutation);
        else
            require(op.basis == "keep", "Unknown basis order: " + op.basis);
        auto moments =
            permute_scalar(source->moment_coeffs, source->moment_coeff_size, permutation);
        std::vector<double> coefficients = radial;
        coefficients.insert(coefficients.end(), species.begin(), species.end());
        coefficients.insert(coefficients.end(), moments.begin(), moments.end());
        if (model_count)
            coefficients.insert(coefficients.end(),
                                source->regression_coeffs + old_tail - model_count,
                                source->regression_coeffs + old_tail);
        if (tail)
        {
            coefficients.insert(coefficients.end(), latent_species.begin(), latent_species.end());
            auto latent_moments =
                permute_scalar(source->regression_coeffs + old_tail + channels * s,
                               tail - channels * s, permutation);
            coefficients.insert(coefficients.end(), latent_moments.begin(), latent_moments.end());
        }
        auto storage = std::make_unique<double[]>(coefficients.size());
        std::copy(coefficients.begin(), coefficients.end(), storage.get());
        delete[] mtp.regression_coeffs;
        mtp.regression_coeffs = storage.release();
        mtp.species_count = n;
        mtp.radial_coeff_size = radial.size();
        mtp.species_coeff_size = species.size();
        mtp.reg_coeff_size = coefficients.size();
        mtp.lin_coeff_size = species.size() + moments.size();
        mtp.radial_coeffs = mtp.regression_coeffs;
        mtp.linear_coeffs = mtp.species_coeffs = mtp.radial_coeffs + radial.size();
        mtp.moment_coeffs = mtp.species_coeffs + species.size();
        if (type == "mtp_m")
            mtp.init_coeffs();
        if (les && les->lr_les)
        {
            auto storage = std::make_unique<double[]>(separate_species.size());
            std::copy(separate_species.begin(), separate_species.end(), storage.get());
            delete[] les->les_species_coeffs;
            les->les_species_coeffs = storage.release();
            auto moments = permute_scalar(old_les->les_moment_coeffs,
                                          old_les->latent_moment_count(), permutation);
            std::copy(moments.begin(), moments.end(), les->les_moment_coeffs);
        }
        result = std::move(model);
    }
    else if (auto *nn = dynamic_cast<PotMTPNN1 *>(&old))
    {
        auto settings = nn->CurrentSettings();
        settings.Modify("nnlay", std::to_string(nn->nnlay));
        settings.Modify("nnhidden",
                        std::to_string(nn->nnlay == 1 ? nn->p_nn->n_neuron : nn->p_nn2->n_neuron1));
        if (nn->nnlay == 2)
            settings.Modify("nnhidden2", std::to_string(nn->p_nn2->n_neuron2));
        auto descriptor =
            std::unique_ptr<PotMTP>(static_cast<PotMTP *>(children.at("descriptor").release()));
        auto model = std::make_unique<PotMTPNN1>(settings, std::move(descriptor));
        const auto &p = permutations.at(id + "/descriptor");
        auto copy_nn = [&](auto &next, const auto &source)
        {
            next = source;
            for (size_t i = 0; i < p.size(); ++i)
            {
                next.w_input.row(i + 1) = source.w_input.row(p[i] + 1);
                next.norm_mean[i + 1] = source.norm_mean[p[i] + 1];
                next.norm_irstd[i + 1] = source.norm_irstd[p[i] + 1];
            }
        };
        if (nn->nnlay == 1)
            copy_nn(*model->p_nn, *nn->p_nn);
        else
            copy_nn(*model->p_nn2, *nn->p_nn2);
        model->species_coeffs = expand(nn->species_coeffs, plan, false, random, rng);
        model->train_nn_only = nn->train_nn_only;
        model->set_inited();
        result = std::move(model);
    }
    else if (auto *linear = dynamic_cast<PotentialDescriptorLinear *>(&old))
    {
        require(children.count("descriptor"),
                "Only MTM-backed descriptor_linear supports MTP adjustments");
        auto descriptor = std::shared_ptr<AnyPotential>(std::move(children.at("descriptor")));
        GPSettings settings;
        settings.Modify("descriptor:dtype", "mtm");
        settings.Modify("species_count", std::to_string(n));
        auto model = std::make_unique<PotentialDescriptorLinear>(
            settings, std::make_unique<DescriptorMTM>(dynamic_cast<PotMTP &>(*descriptor)));
        model->gpm_resources = descriptor;
        std::vector<double> before(linear->get_NL_coeffs_count());
        linear->nl_get_coeffs(before.data());
        auto coefficients =
            permute_scalar(before.data(), before.size(), permutations.at(id + "/descriptor"));
        model->nl_set_coeffs(coefficients.data());
        result = std::move(model);
    }
    else if (auto *zbl = dynamic_cast<PotentialZBL *>(&old))
    {
        auto model = std::make_unique<PotentialZBL>();
        model->r0 = zbl->r0;
        model->r1 = zbl->r1;
        model->z.resize(n);
        for (int i = 0; i < n; ++i)
        {
            model->z[i] = zbl->z.at(plan.source[i]);
            if (plan.fresh[i])
            {
                require(Chemicals::atomic_numbers.count(plan.names[i]),
                        "ZBL requires a valid new element symbol");
                model->z[i] = Chemicals::atomic_numbers.at(plan.names[i]);
            }
        }
        result = std::move(model);
    }
    else if (type == "hybrid" || type == "hybrid_scale" || type == "hmtp")
    {
        std::vector<std::unique_ptr<AnyPotential>> parts;
        for (size_t i = 0; i < children.size(); ++i)
            parts.push_back(std::move(children.at("pot" + std::to_string(i + 1))));
        result = make_composite(type, composite_settings(old), std::move(parts));
    }
    else if (auto *eam = dynamic_cast<PotentialEAMPoly *>(&old))
    {
        require(op.appended.empty() && op.order.empty(),
                "EAMPoly species migration is not defined for MTP workflows");
        result = std::make_unique<PotentialEAMPoly>(*eam);
    }
    else
        require(false, "MTP workflow does not support model type: " + type);
    copy_common(*result, old, plan);
    permutations[id] = std::move(permutation);
    return result;
}

// Append a block's new-to-old indices in the enclosing parameter coordinates.
void append_indices(std::vector<int> &target, const std::vector<int> &block, int offset)
{
    for (int source : block)
        target.push_back(source < 0 ? -1 : offset + source);
}

std::vector<int> species_indices(int count, const Species &plan, bool pair, bool random)
{
    const int s = plan.old_count, n = plan.source.size();
    const int stride = count / (pair ? s * s : s);
    std::vector<int> result((pair ? n * n : n) * stride, -1);
    if (pair)
    {
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                if (!(random && (plan.fresh[i] || plan.fresh[j])))
                    for (int k = 0; k < stride; ++k)
                        result[(i * n + j) * stride + k] =
                            (plan.source[i] * s + plan.source[j]) * stride + k;
    }
    else
    {
        for (int c = 0; c < stride; ++c)
            for (int i = 0; i < n; ++i)
                if (!(random && plan.fresh[i]))
                    result[c * n + i] = c * s + plan.source[i];
    }
    return result;
}

std::vector<int> scalar_indices(int count, const std::vector<int> &permutation)
{
    std::vector<int> result;
    const int scalars = permutation.size();
    for (int offset = 0; offset < count; offset += scalars)
        append_indices(result, permutation, offset);
    return result;
}

enum class ParameterLayout
{
    Nonlinear,
    Regression,
    Linear
};

// Layout rules follow each model's nl_get_coeffs order. No parameter values are
// read or modified to discover coordinates. -1 denotes no transferable penalty
// (a new random parameter, a constant, or a coupled gauge coordinate).
std::vector<int> parameter_indices(AnyPotential &old_model, AnyPotential &new_model,
                                   const std::string &id, const Adjustment &op, const Species &plan,
                                   const BasisPermutations &permutations,
                                   ParameterLayout layout = ParameterLayout::Nonlinear)
{
    const auto type = ModelBundle::type(old_model);
    const auto children = model_children(old_model), next_children = model_children(new_model);
    auto *old = &old_model;
    auto *next = &new_model;
    const bool random = op.initialization == "random";
    std::vector<int> result;
    auto species = [&](int count, bool pair = false)
    {
        return species_indices(count, plan, pair, random);
    };
    auto child_id = [&](const std::string &role)
    {
        return id + "/" + role;
    };
    auto child_map = [&](const std::string &role)
    {
        return parameter_indices(*children.at(role), *next_children.at(role), child_id(role), op,
                                 plan, permutations);
    };
    auto child_count = [&](const std::string &role)
    {
        return children.at(role)->get_NL_coeffs_count();
    };

    if (type == "mtp" || type == "mtp_m" || type == "mtp_les")
    {
        auto &a = dynamic_cast<PotMTP &>(*old);
        auto &b = dynamic_cast<PotMTP &>(*next);
        const auto &p = permutations.at(id);
        auto moments = [&](int count)
        {
            return scalar_indices(count, p);
        };
        // Physical regression layout: radial, species channels, moment channels,
        // head parameters, then optional LES species and moment channels.
        auto regression = [&]
        {
            std::vector<int> map;
            append_indices(map, species(a.radial_coeff_size, true), 0);
            append_indices(map, species(a.species_coeff_size), a.radial_coeff_size);
            int offset = a.radial_coeff_size + a.species_coeff_size;
            append_indices(map, moments(a.moment_coeff_size), offset);
            offset += a.moment_coeff_size;
            int model_count = 0;
            if (type == "mtp_m")
            {
                int radial, sp, moment;
                a.get_n_coeffs(radial, sp, moment, model_count);
            }
            append_indices(map, identity(model_count), offset);
            offset += model_count;
            const int tail = a.reg_coeff_size - offset;
            if (tail)
            {
                const int channels = tail / (a.species_count + a.alpha_scalar_moments);
                const int count = channels * a.species_count;
                append_indices(map, species(count), offset);
                append_indices(map, moments(tail - count), offset + count);
            }
            return map;
        };
        if (layout != ParameterLayout::Nonlinear || type == "mtp")
        {
            result = regression();
            if (layout == ParameterLayout::Linear)
            {
                result = std::vector<int>(result.begin() + b.radial_coeff_size,
                                          result.begin() + b.radial_coeff_size + b.lin_coeff_size);
                for (auto &index : result)
                    if (index >= 0)
                        index -= a.radial_coeff_size;
            }
        }
        else
        {
            // MTP-M request masks select groups; LES has a separate latent tail.
            const auto new_groups = next->nl_parameter_groups();
            for (const auto &group : old->nl_parameter_groups())
            {
                const int count = group.end - group.begin;
                std::vector<int> map;
                if (group.name == "sr_radial")
                    map = species(count, true);
                else if (group.name == "sr_species" || group.name == "les_species")
                    map = species(count);
                else if (group.name == "les_moment")
                    map = moments(count);
                else if (group.name == "sr_linear")
                {
                    map = species(a.species_count);
                    append_indices(map, moments(count - a.species_count), a.species_count);
                }
                else if (group.name == "sr_moment")
                {
                    const int extra_species = a.species_coeff_size - a.species_count;
                    map = species(extra_species);
                    append_indices(map, moments(a.moment_coeff_size), extra_species);
                }
                else
                    map = identity(count); // Head and SOG optimizer coordinates stay fixed.
                const auto target = std::find_if(new_groups.begin(), new_groups.end(),
                                                 [&](const auto &g)
                                                 {
                                                     return g.name == group.name;
                                                 });
                require(target != new_groups.end() &&
                            int(map.size()) == target->end - target->begin,
                        "MTP parameter group changed unexpectedly: " + group.name);
                append_indices(result, map, group.begin);
            }
        }
    }
    else if (type == "mtpnn1")
    {
        const auto &nn = dynamic_cast<const PotMTPNN1 &>(*old);
        const auto descriptor = child_id("descriptor");
        const auto &p = permutations.at(descriptor);
        if (!nn.train_nn_only)
        {
            append_indices(result, species(nn.p_mtp->radial_coeff_size, true), 0);
            append_indices(result, species(nn.p_mtp->species_count), nn.p_mtp->radial_coeff_size);
        }
        const int offset =
            nn.train_nn_only ? 0 : nn.p_mtp->radial_coeff_size + nn.p_mtp->species_count;
        auto weights = identity(nn.nn_coeff_size);
        const int rows = nn.p_mtp->alpha_scalar_moments + 1;
        const int cols = nn.nnlay == 1 ? nn.p_nn->n_neuron : nn.p_nn2->n_neuron1;
        for (int c = 0; c < cols; ++c)
            for (size_t i = 0; i < p.size(); ++i)
                weights[c * rows + i + 1] = c * rows + p[i] + 1;
        append_indices(result, weights, offset);
        append_indices(result, species(nn.species_coeffs.size()), offset + nn.nn_coeff_size);
    }
    else if (type == "descriptor_linear")
        result = permutations.at(child_id("descriptor"));
    else if (type == "hybrid")
    {
        const auto &hybrid = dynamic_cast<const PotentialHybrid &>(*old);
        int radial_offset = 0, moment_offset = hybrid.nrad_coeffs + hybrid.species_count;
        for (int i : hybrid.train_ids)
        {
            auto &mtp = dynamic_cast<PotMTP &>(*hybrid.ppots[i]);
            append_indices(result, species(mtp.radial_coeff_size, true), radial_offset);
            radial_offset += mtp.radial_coeff_size;
        }
        if (!hybrid.train_ids.empty())
            append_indices(result, species(hybrid.species_count), hybrid.nrad_coeffs);
        for (int i : hybrid.train_ids)
        {
            const auto &p = permutations.at(child_id("pot" + std::to_string(i + 1)));
            append_indices(result, p, moment_offset);
            moment_offset += p.size();
        }
    }
    else if (type == "hybrid_scale" || type == "hmtp")
    {
        int offset = 0;
        for (size_t i = 0; i < children.size(); ++i)
        {
            const auto role = "pot" + std::to_string(i + 1);
            append_indices(result, child_map(role), offset);
            offset += child_count(role);
        }
    }
    else // ZBL has no optimizer parameters; EAMPoly is unchanged by supported operations.
        result = identity(old->get_NL_coeffs_count());

    const auto *mtp = dynamic_cast<const PotMTP *>(next);
    const int expected = layout == ParameterLayout::Nonlinear    ? next->get_NL_coeffs_count()
                         : layout == ParameterLayout::Regression ? mtp->reg_coeff_size
                                                                 : mtp->lin_coeff_size;
    require(int(result.size()) == expected, "Parameter migration layout mismatch at " + id);
    return result;
}

}

std::unique_ptr<AnyPotential> adjust_tree(AnyPotential &model, const Adjustment &op)
{
    const auto plan = species_plan(model, op);
    std::mt19937 rng = RNG.mtd;
    BasisPermutations permutations;
    auto result = morph(model, "root", op, plan, rng, permutations);
    auto migrate = [&](auto &&self, AnyPotential &old, AnyPotential &next,
                       const std::string &id) -> void
    {
        for (bool l1 : {true, false})
        {
            const auto &weights = l1 ? old.reg_L1_vector : old.reg_L2_vector;
            if (weights.empty())
                continue;
            auto *mtp = dynamic_cast<PotMTP *>(&old);
            auto layout = ParameterLayout::Nonlinear;
            if (weights.size() != size_t(old.get_NL_coeffs_count()))
            {
                if (mtp && weights.size() == size_t(mtp->reg_coeff_size))
                    layout = ParameterLayout::Regression;
                else if (mtp && weights.size() == size_t(mtp->lin_coeff_size))
                    layout = ParameterLayout::Linear;
                else
                    require(false, "Unknown regularization layout at " + id);
            }
            const auto map = parameter_indices(old, next, id, op, plan, permutations, layout);
            auto &updated = l1 ? next.reg_L1_vector : next.reg_L2_vector;
            updated.assign(map.size(), 0.);
            std::vector<bool> seen(weights.size());
            for (size_t i = 0; i < map.size(); ++i)
                if (map[i] >= 0)
                {
                    updated[i] = weights.at(map[i]);
                    seen[map[i]] = true;
                }
            for (size_t i = 0; i < weights.size(); ++i)
                require(seen[i] || weights[i] == 0.,
                        "Cannot preserve coupled/gauge regularization at " + id +
                            "; remove that penalty before changing its reference gauge");
        }
        const auto next_children = model_children(next);
        for (const auto &[role, child] : model_children(old))
            self(self, *child, *next_children.at(role), id + "/" + role);
    };
    migrate(migrate, model, *result, "root");
    return result;
}
}
