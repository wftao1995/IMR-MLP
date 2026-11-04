// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtp_model_internal.h"
#include "model_bundle.h"

#include "potential_mtp_m.h"
#include "potential_mtp_LES.h"
#include "potential_mtpnn1.h"
#include "potential_descriptor_linear.h"
#include "potential_eam_poly.h"
#include "potential_hybrid.h"
#include "potential_hybrid_scale.h"
#include "potential_hybmtp.h"

namespace Potentials::MTP::detail
{
using GPM::require;

std::map<std::string, AnyPotential *> model_children(AnyPotential &model)
{
    if (auto *nn = dynamic_cast<PotMTPNN1 *>(&model))
        return {{"descriptor", nn->p_mtp}};
    if (auto *linear = dynamic_cast<PotentialDescriptorLinear *>(&model))
        if (auto *mtm = dynamic_cast<const DescriptorMTM *>(&linear->descriptor()))
            return {{"descriptor", &mtm->pot()}};
    std::vector<const AnyPotential *> children;
    if (auto *hybrid = dynamic_cast<PotentialHybrid *>(&model))
        children = hybrid->children();
    else if (auto *scaled = dynamic_cast<PotentialHybridScale *>(&model))
        children = scaled->children();
    else if (auto *hmtp = dynamic_cast<PotMTPHyb3 *>(&model))
        for (auto *child : hmtp->children())
            children.push_back(child);
    std::map<std::string, AnyPotential *> result;
    for (size_t i = 0; i < children.size(); ++i)
        result.emplace("pot" + std::to_string(i + 1), const_cast<AnyPotential *>(children[i]));
    return result;
}

std::map<int, std::string> model_element_map(AnyPotential &model)
{
    auto elements = model.gpm_elements;
    auto named = [&](const std::vector<std::string> &names)
    {
        for (size_t i = 0; i < names.size(); ++i)
            if (!names[i].empty())
                elements[i] = names[i];
    };
    if (auto *mtp = dynamic_cast<PotMTP *>(&model))
        named(mtp->potential_tag.elements());
    else if (auto *eam = dynamic_cast<PotentialEAMPoly *>(&model))
        named(eam->element_names());
    for (const auto &[role, child] : model_children(model))
        for (const auto &[i, name] : model_element_map(*child))
        {
            require(!elements.count(i) || elements.at(i) == name,
                    "Child and parent element maps differ");
            elements[i] = name;
        }
    return elements;
}

std::vector<std::string> model_elements(AnyPotential &model)
{
    const auto elements = model_element_map(model);
    std::vector<std::string> names;
    for (int i = 0; i < model.species_count; ++i)
    {
        if (!elements.count(i))
            return {};
        names.push_back(elements.at(i));
    }
    return names;
}

GPSettings composite_settings(AnyPotential &model)
{
    GPSettings settings;
    auto children = model_children(model);
    settings.Modify("npot", std::to_string(children.size()));
    for (const auto &[role, child] : children)
        settings.Modify(role + ":pottype", ModelBundle::type(*child));
    if (auto *hybrid = dynamic_cast<PotentialHybrid *>(&model))
        for (size_t i = 0; i < children.size(); ++i)
            settings.Modify("pot" + std::to_string(i + 1) + ":train",
                            std::find(hybrid->train_ids.begin(), hybrid->train_ids.end(), int(i)) !=
                                    hybrid->train_ids.end()
                                ? "true"
                                : "false");
    if (auto *scaled = dynamic_cast<PotentialHybridScale *>(&model))
    {
        settings.Modify("feature", scaled->feature);
        settings.Modify("transition", scaled->transition);
        auto text = [](double value)
        {
            std::ostringstream out;
            out << std::setprecision(17) << value;
            return out.str();
        };
        settings.Modify("feature_r0", text(scaled->feature_r0));
        settings.Modify("feature_r1", text(scaled->feature_r1));
        for (size_t i = 0; i < scaled->boundaries().size(); ++i)
        {
            const auto &b = scaled->boundaries()[i];
            settings.Modify("boundary" + std::to_string(i + 1), text(b.left) + " " + text(b.right));
        }
    }
    return settings;
}

std::unique_ptr<AnyPotential> make_composite(const std::string &type, GPSettings settings,
                                             std::vector<std::unique_ptr<AnyPotential>> children)
{
    if (type == "hybrid")
        return std::make_unique<PotentialHybrid>(settings, std::move(children));
    if (type == "hybrid_scale")
        return std::make_unique<PotentialHybridScale>(settings, std::move(children));
    require(children.size() == 2 || children.size() == 3,
            "HMTP requires two or three MTP children");
    for (const auto &child : children)
        require(typeid(*child) == typeid(PotMTP), "HMTP children must be ordinary MTP");
    auto take = [&](size_t i)
    {
        return std::unique_ptr<PotMTP>(static_cast<PotMTP *>(children[i].release()));
    };
    return std::make_unique<PotMTPHyb3>(take(0), take(1), children.size() == 3 ? take(2) : nullptr);
}

namespace
{
template <class T> std::unique_ptr<PotMTP> copy_head(PotMTP_M<T> &model, bool short_range_only)
{
    auto result = std::make_unique<PotMTP_M<T>>(model);
    if (short_range_only && result->lr_les)
    {
        result->reg_coeff_size -= result->les_head_count();
        result->lr_les = result->req_les = result->req_sog = false;
        result->les_coeffs = result->les_species = result->les_moment = nullptr;
        result->potential_tag.fields.erase("long_range");
    }
    return result;
}
}

std::unique_ptr<PotMTP> copy_mtp(PotMTP &model, bool short_range_only)
{
    std::unique_ptr<PotMTP> result;
    if (typeid(model) == typeid(PotMTP))
        result = std::make_unique<PotMTP>(model);
    else if (auto *les = dynamic_cast<PotMTP_LES *>(&model))
    {
        if (short_range_only)
        {
            result = std::make_unique<PotMTP>(model);
            result->potential_tag.fields.erase("long_range");
            result->potential_tag.fields["type"] = MTPMetadata::Value("MTP");
        }
        else
            result = std::make_unique<PotMTP_LES>(*les);
    }
#define COPY_HEAD(T)                                                                               \
    else if (auto *mtp = dynamic_cast<PotMTP_M<T> *>(&model)) result =                             \
        copy_head(*mtp, short_range_only);
    COPY_HEAD(m_model_mtpfs)
    COPY_HEAD(m_model_mtpq2)
    COPY_HEAD(m_model_mtpli)
    COPY_HEAD(m_model_nn1)
    COPY_HEAD(m_model_nn2)
#undef COPY_HEAD
    else GPERROR("Unsupported MTP head for model copying");
    if (short_range_only)
    {
        result->reg_L1_vector.clear();
        result->reg_L2_vector.clear();
    }
    return result;
}
}
