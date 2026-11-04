#include "mtp_tree_workflow.h"
#include "mtp_workflow.h"
#include "potentials/gpm_codec.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_hybrid.h"
#include "potentials/potential_hybrid_scale.h"
#include "potentials/potential_hybmtp.h"
#include "potentials/potential_descriptor_linear.h"
#include "descriptors/descriptor_MTM.h"
#include "tools/chem.h"
#include <bit>
#include <unordered_map>

namespace MTPWorkflow
{
using namespace Potentials;
using namespace GPM;

namespace
{
struct Snapshot
{
    Stage stage;
    std::shared_ptr<Reader> reader;

    explicit Snapshot(AnyPotential &model)
    {
        const auto file = (stage.path / "source.gpm").string();
        ModelBundle::save_potential(model, file, ModelBundle::Profile::Inference);
        reader = std::make_shared<Reader>(file);
    }
};

struct Graph
{
    std::shared_ptr<Reader> reader;
    Writer writer;
    Stage stage;
    size_t serial = 0;

    explicit Graph(std::shared_ptr<Reader> r) : reader(std::move(r))
    {
        writer.preserve(reader);
        writer.manifest.erase("active_learning");
    }

    Object &node(const std::string &id)
    {
        return writer.manifest.at("models").get<Object>().at(id).get<Object>();
    }

    Value numeric(const std::vector<double> &values, std::vector<uint64_t> shape = {})
    {
        return vector_block(writer, "workflow/" + std::to_string(serial++), values,
                            std::move(shape));
    }

    template <class T> void field(Object &object, const std::string &name, T value)
    {
        State state(writer, "workflow/" + std::to_string(serial++));
        state.field(name, value);
        object[name] = state.fields().at(name);
    }

    std::unique_ptr<AnyPotential> load(const std::string &root = "")
    {
        if (!root.empty())
        {
            writer.manifest["root_model"] = Value(root);
            writer.manifest["species"] = node(root).at("species");
        }
        const auto file = (stage.path / ("graph-" + std::to_string(serial++) + ".gpm")).string();
        writer.save(file);
        return ModelBundle::load(file)->release_potential();
    }
};

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

Species species_plan(const Object &root, const Adjustment &op)
{
    Species plan;
    const auto &species = array(at(root, "species"));
    plan.old_count = species.size();
    plan.source = identity(plan.old_count);
    for (const auto &entry : species)
    {
        const auto &s = object(entry);
        plan.names.push_back(s.count("symbol") ? string(s.at("symbol")) : "");
    }
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

void declare_species(Graph &g, Object &node, const Species &plan)
{
    Array entries;
    for (size_t i = 0; i < plan.source.size(); ++i)
    {
        Object s{{"id", Value(double(i))}};
        if (!plan.names[i].empty())
            s["symbol"] = Value(plan.names[i]);
        entries.emplace_back(s);
    }
    node["species"] = Value(entries);
    auto &common = node.at("common").get<Object>();
    common["species_count"] = Value(double(entries.size()));
    for (const auto *key : {"cutoffs", "r_mins"})
    {
        const auto old = values(*g.reader, common.at(key));
        if (old.empty())
            continue;
        require(old.size() == size_t(plan.old_count * plan.old_count),
                "Invalid species matrix dimensions");
        const int n = plan.source.size();
        std::vector<double> result(n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                result[i * n + j] = old[plan.source[i] * plan.old_count + plan.source[j]];
        common[key] = g.numeric(result, {uint64_t(n), uint64_t(n)});
    }
    common["reg_L1"] = g.numeric({});
    common["reg_L2"] = g.numeric({});
}

void permute_scalar(Graph &g, Value &field, const std::vector<int> &permutation)
{
    auto old = values(*g.reader, field), result = old;
    const int b = permutation.size();
    require(b && old.size() % b == 0, "Invalid scalar-channel layout");
    for (int c = 0; c < int(old.size()) / b; ++c)
        for (int i = 0; i < b; ++i)
            result[c * b + i] = old[c * b + permutation[i]];
    field = g.numeric(result);
}

std::vector<int> morph(Graph &g, const std::string &id, const Adjustment &op, const Species &plan,
                       std::mt19937 &rng)
{
    auto &node = g.node(id);
    const auto type = string(node.at("type"));
    const int s = plan.old_count, n = plan.source.size();
    std::map<std::string, std::vector<int>> child_permutations;
    for (const auto &[role, ref] : object(node.at("children")))
        child_permutations[role] = morph(g, string(ref), op, plan, rng);
    const bool random = op.initialization == "random";
    std::vector<int> permutation;
    if (type == "mtp" || type == "mtp_m" || type == "mtp_les")
    {
        auto model = GPMCodecAccess::decode(*g.reader, id);
        auto &mtp = dynamic_cast<PotMTP &>(*model);
        validate_graph(mtp);
        require(!mtp.coeffs_handled_external, "Borrowed MTP storage cannot be migrated");
        require(type != "mtp_m" || mtp.supports_species_append(),
                "MTP-M correction parameters have no defined migration adapter");
        auto &config = node.at("configuration").get<Object>();
        auto &parameters = node.at("parameters").get<Object>();
        parameters["radial"] =
            g.numeric(expand(values(*g.reader, parameters.at("radial")), plan, true, random, rng));
        parameters["species"] = g.numeric(
            expand(values(*g.reader, parameters.at("species")), plan, false, random, rng));
        auto &bases = config.at("radial_bases").get<Array>();
        if (bases.size() > 1)
        {
            require(bases.size() == size_t(s * s), "Invalid split radial basis layout");
            auto old = bases;
            bases.resize(n * n);
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    bases[i * n + j] = old[plan.source[i] * s + plan.source[j]];
        }
        config["species_count"] = Value(double(n));
        auto &meta = config.at("metadata").get<Object>();
        if (std::none_of(plan.names.begin(), plan.names.end(),
                         [](auto &name)
                         {
                             return name.empty();
                         }))
        {
            Array names;
            for (auto &name : plan.names)
                names.emplace_back(name);
            meta["elements"] = Value(names);
        }
        for (auto &value : config.at("head_fields").get<Array>())
        {
            auto &field = value.get<Object>();
            const auto name = string(field.at("name"));
            if (field.count("array") && name.find("les") != std::string::npos &&
                name.find("species") != std::string::npos)
                field["array"] = g.numeric(
                    expand(values(*g.reader, field.at("array")), plan, false, random, rng));
        }
        permutation = identity(mtp.alpha_scalar_moments);
        if (op.basis != "keep")
        {
            require(op.basis == "level" || op.basis == "level-group",
                    "Unknown basis order: " + op.basis);
            for (int i = 0; i < mtp.alpha_scalar_moments; ++i)
                mtp.moment_coeffs[i] = i;
            if (op.basis == "level")
                mtp.ReorderByLevel();
            else
                mtp.ReorderByLevelGroup();
            for (size_t i = 0; i < permutation.size(); ++i)
                permutation[i] = int(mtp.moment_coeffs[i]);
            std::vector<int32_t> basic, times, mapping;
            for (int i = 0; i < mtp.alpha_index_basic.rows(); ++i)
                for (int j = 0; j < 4; ++j)
                    basic.push_back(mtp.alpha_index_basic(i, j));
            for (int i = 0; i < mtp.alpha_index_times.rows(); ++i)
                for (int j = 0; j < 4; ++j)
                    times.push_back(mtp.alpha_index_times(i, j));
            for (int i = 0; i < mtp.alpha_scalar_moments; ++i)
                mapping.push_back(mtp.alpha_moment_mapping[i]);
            parameters["basic_indices"] =
                index_block(g.writer, "workflow/basic/" + id, basic, {basic.size() / 4, 4});
            parameters["product_indices"] =
                index_block(g.writer, "workflow/times/" + id, times, {times.size() / 4, 4});
            parameters["scalar_mapping"] =
                index_block(g.writer, "workflow/scalars/" + id, mapping, {mapping.size()});
            permute_scalar(g, parameters.at("moments"), permutation);
            for (auto &value : config.at("head_fields").get<Array>())
            {
                auto &field = value.get<Object>();
                const auto name = string(field.at("name"));
                if (field.count("array") && name.find("les") != std::string::npos &&
                    name.find("moment") != std::string::npos)
                    permute_scalar(g, field.at("array"), permutation);
            }
        }
    }
    else if (type == "mtpnn1")
    {
        auto &state = node.at("state").get<Object>();
        state["species_coefficients"] = g.numeric(
            expand(values(*g.reader, state.at("species_coefficients")), plan, false, random, rng));
        const auto &p = child_permutations.at("descriptor");
        if (op.basis != "keep")
        {
            const int rows = integer(state.at("input")), cols = integer(state.at("hidden"));
            require(rows == int(p.size()) + 1, "NN descriptor input dimensions mismatch");
            auto weights = values(*g.reader, state.at("input_weights")), next = weights;
            for (size_t i = 0; i < p.size(); ++i)
                for (int j = 0; j < cols; ++j)
                    next[(i + 1) * cols + j] = weights[(p[i] + 1) * cols + j];
            state["input_weights"] = g.numeric(next, {uint64_t(rows), uint64_t(cols)});
            for (const auto *key : {"mean", "inverse_std"})
            {
                auto old = values(*g.reader, state.at(key)), v = old;
                for (size_t i = 0; i < p.size(); ++i)
                    v[i + 1] = old[p[i] + 1];
                state[key] = g.numeric(v, {uint64_t(rows), 1});
            }
        }
    }
    else if (type == "descriptor_linear")
    {
        require(child_permutations.count("descriptor"),
                "Only MTM-backed descriptor_linear supports MTP adjustments");
        if (op.basis != "keep")
        {
            auto &field = node.at("state").get<Object>().at("coefficients");
            auto old = values(*g.reader, field), next = old;
            const auto &p = child_permutations.at("descriptor");
            require(old.size() == p.size(), "Invalid descriptor-linear head dimensions");
            for (size_t i = 0; i < p.size(); ++i)
                next[i] = old[p[i]];
            field = g.numeric(next, {uint64_t(next.size()), 1});
        }
    }
    else if (type == "zbl")
    {
        auto &config = node.at("configuration").get<Object>();
        auto old = values(*g.reader, config.at("z"));
        require(old.size() == size_t(s), "Invalid ZBL element dimensions");
        std::vector<double> z(n);
        for (int i = 0; i < n; ++i)
        {
            z[i] = old[plan.source[i]];
            if (plan.fresh[i])
            {
                require(Chemicals::atomic_numbers.count(plan.names[i]),
                        "ZBL requires a valid new element symbol");
                z[i] = Chemicals::atomic_numbers.at(plan.names[i]);
            }
        }
        config["z"] = g.numeric(z);
    }
    else if (type == "hybrid" || type == "hybrid_scale" || type == "hmtp")
    {
        node.at("configuration").get<Object>()["spcount"] = Value(std::to_string(n));
    }
    else if (type == "eampoly")
    {
        require(op.appended.empty() && op.order.empty(),
                "EAMPoly species migration is not defined for MTP workflows");
    }
    else
        require(false, "MTP workflow does not support model type: " + type);
    declare_species(g, node, plan);
    return permutation;
}

std::unique_ptr<AnyPotential> physical_transform(std::shared_ptr<Reader> reader,
                                                 const Adjustment &op, const Species &plan,
                                                 const std::string &root = "")
{
    Graph graph(reader);
    std::mt19937 rng = RNG.mtd;
    const auto id = root.empty() ? string(graph.writer.manifest.at("root_model")) : root;
    morph(graph, id, op, plan, rng);
    return graph.load(id);
}

// A marker pass through the same structural operation obtains the actual model's
// optimizer-coordinate permutation, including request masks and shared storage.
// Tiny distinct ULP perturbations retain valid physical domains during decoding.
std::vector<int> parameter_sources(AnyPotential &old, const Adjustment &op, const Species &plan,
                                   int layout)
{
    auto *mtp = dynamic_cast<PotMTP *>(&old);
    int count = layout == 0   ? old.get_NL_coeffs_count()
                : layout == 1 ? mtp->reg_coeff_size
                              : mtp->lin_coeff_size;
    std::vector<double> values(count);
    if (layout == 0)
        old.nl_get_coeffs(values.data());
    else
        std::copy_n(layout == 1 ? mtp->regression_coeffs : mtp->linear_coeffs, count,
                    values.data());
    auto assign = [&](const std::vector<double> &data)
    {
        if (layout == 0)
            old.nl_set_coeffs(data.data());
        else
        {
            std::copy(data.begin(), data.end(),
                      layout == 1 ? mtp->regression_coeffs : mtp->linear_coeffs);
            if (ModelBundle::type(old) == "mtp_m")
                old.init_coeffs();
        }
    };

    struct Restore
    {
        std::function<void()> f;

        ~Restore()
        {
            f();
        }
    } restore{[&]
              {
                  assign(values);
              }};

    std::unordered_map<uint64_t, int> ids;
    auto markers = values;
    std::vector<NLParameterGroup> unchanged;
    std::vector<bool> skip(count, false);
    if (layout == 0)
        for (const auto &group : old.nl_parameter_groups())
            if (group.name.find("sog_") != std::string::npos)
            {
                unchanged.push_back(group);
                for (int i = group.begin; i < group.end; ++i)
                    skip[i] = true;
            }
    std::unordered_map<uint64_t, double> last;

    for (int i = 0; i < count; ++i)
        if (!skip[i])
        {
            const auto bits = std::bit_cast<uint64_t>(values[i]);
            double marker =
                last.count(bits) ? last[bits] : values[i] + 1e-6 * (1 + std::abs(values[i]));
            while (ids.count(std::bit_cast<uint64_t>(marker)))
                marker = std::nextafter(marker, std::numeric_limits<double>::infinity());
            require(std::isfinite(marker), "Cannot mark a nonfinite parameter");
            markers[i] = marker;
            ids[std::bit_cast<uint64_t>(marker)] = i;
            ids[std::bit_cast<uint64_t>(-marker)] = i;
            last[bits] = std::nextafter(marker, std::numeric_limits<double>::infinity());
        }
    assign(markers);
    Snapshot snapshot(old);
    auto next = physical_transform(snapshot.reader, op, plan);
    auto *new_mtp = dynamic_cast<PotMTP *>(next.get());
    const int size = layout == 0   ? next->get_NL_coeffs_count()
                     : layout == 1 ? new_mtp->reg_coeff_size
                                   : new_mtp->lin_coeff_size;
    std::vector<double> transformed(size);
    if (layout == 0)
        next->nl_get_coeffs(transformed.data());
    else
        std::copy_n(layout == 1 ? new_mtp->regression_coeffs : new_mtp->linear_coeffs, size,
                    transformed.data());
    std::vector<int> map(size, -1);
    for (int i = 0; i < size; ++i)
    {
        auto found = ids.find(std::bit_cast<uint64_t>(transformed[i]));
        if (found != ids.end())
            map[i] = found->second;
    }
    if (layout == 0)
        for (const auto &before : unchanged)
            for (const auto &after : next->nl_parameter_groups())
                if (before.name == after.name)
                {
                    require(before.end - before.begin == after.end - after.begin,
                            "Kernel dimension changed during migration");
                    for (int k = 0; k < before.end - before.begin; ++k)
                        map[after.begin + k] = before.begin + k;
                }
    return map;
}
}

std::unique_ptr<AnyPotential> adjust_tree(AnyPotential &model, const Adjustment &op)
{
    LocalComputation local;
    Snapshot original(model);
    const auto root = string(original.reader->manifest.at("root_model"));
    const auto &nodes = object(original.reader->manifest.at("models"));
    Object source_root = object(nodes.at(root));
    if (array(source_root.at("species")).empty())
        for (const auto &[id, node] : nodes)
            if (object(node).at("type") == Value("mtp"))
            {
                source_root["species"] = object(node).at("species");
                break;
            }
    const auto plan = species_plan(source_root, op);
    Graph changed(original.reader);
    std::mt19937 rng = RNG.mtd;
    morph(changed, root, op, plan, rng);
    for (const auto &[id, value] : nodes)
    {
        const auto &node = object(value);
        auto old = GPMCodecAccess::decode(*original.reader, id);
        const auto &common = object(node.at("common"));
        for (const auto *key : {"reg_L1", "reg_L2"})
        {
            const auto weights = values(*original.reader, common.at(key));
            if (weights.empty())
                continue;
            auto *mtp = dynamic_cast<PotMTP *>(old.get());
            int layout = 0;
            if (weights.size() != size_t(old->get_NL_coeffs_count()))
            {
                if (mtp && weights.size() == size_t(mtp->reg_coeff_size))
                    layout = 1;
                else if (mtp && weights.size() == size_t(mtp->lin_coeff_size))
                    layout = 2;
                else
                    require(false, "Unknown regularization layout at " + id);
            }
            const auto map = parameter_sources(*old, op, plan, layout);
            std::vector<double> next(map.size());
            std::vector<bool> seen(weights.size());
            for (size_t i = 0; i < map.size(); ++i)
                if (map[i] >= 0)
                {
                    next[i] = weights[map[i]];
                    seen[map[i]] = true;
                }
            for (size_t i = 0; i < weights.size(); ++i)
                require(seen[i] || weights[i] == 0.,
                        "Cannot preserve coupled regularization at " + id +
                            "; remove that penalty before adjusting the model");
            changed.node(id).at("common").get<Object>()[key] = changed.numeric(next);
        }
    }
    return changed.load(root);
}
}

namespace MTPWorkflow
{
namespace
{
std::vector<double> numbers(std::string text)
{
    for (char &c : text)
        if (c == '{' || c == '}' || c == ',')
            c = ' ';
    std::istringstream in(text);
    std::vector<double> result;
    double value;
    while (in >> value)
    {
        require(std::isfinite(value), "Nonfinite initialization parameter");
        result.push_back(value);
    }
    require(in.eof(), "Malformed numeric parameter list");
    return result;
}

std::string plain_tag(std::string text)
{
    const auto start = text.find("potential_tag =");
    if (start != std::string::npos)
        text.replace(start, text.find('\n', start) - start, "potential_tag =");
    std::istringstream lines(text);
    std::ostringstream out;
    std::string line;
    while (std::getline(lines, line))
        if (!GPM::trim(line).starts_with("#"))
            out << line << '\n';
    return out.str();
}

std::string model_text(AnyPotential &model)
{
    std::ostringstream text;
    text << std::setprecision(17);
    model.Save(text);
    return text.str();
}

std::vector<std::string> model_elements(AnyPotential &model)
{
    Snapshot source(model);
    const auto &species = array(source.reader->manifest.at("species"));
    std::vector<std::string> names;
    for (const auto &entry : species)
    {
        auto &s = object(entry);
        if (!s.count("symbol"))
            return {};
        names.push_back(string(s.at("symbol")));
    }
    return names;
}

void set_elements(AnyPotential &model, const std::vector<std::string> &names)
{
    if (names.empty())
        return;
    require(model.species_count == int(names.size()), "Model and element count mismatch");
    std::map<int, std::string> map;
    for (size_t i = 0; i < names.size(); ++i)
        map[i] = names[i];
    model.set_element_map(map);
}

struct Builder
{
    Stage stage;
    size_t serial = 0;
    bool reset, randomize;
    double lower, upper, scale;
    std::vector<std::string> names;
    std::string index;
    bool explicit_index;

    std::string file(const std::string &suffix)
    {
        return (stage.path / (std::to_string(serial++) + suffix)).string();
    }

    std::string saved(AnyPotential &model)
    {
        auto path = file(".gpm");
        ModelBundle::save_potential(model, path, ModelBundle::Profile::Inference);
        return path;
    }

    std::unique_ptr<AnyPotential> load(const std::string &input, GPSettings config)
    {
        if (GPM::is_file(input) || GPM::extension(input))
            return ModelBundle::load(input)->release_potential();
        config.Modify("potfile", input);
        if (config["pottype"].empty())
            config.Modify("pottype", "auto");
        potential_wrapper wrapper(config);
        return std::unique_ptr<AnyPotential>(std::exchange(wrapper.pot, nullptr));
    }

    std::unique_ptr<AnyPotential> leaf(GPSettings config, const std::string &input,
                                       const std::string &type)
    {
        GPSettings all;
        for (const auto &[key, value] : config.settings)
            all.Modify("pot:" + key, value);
        const auto path = file(".gpm");
        initialize(all, input, path, type, names, reset, randomize, lower, upper, "", "", index,
                   false, false, scale, explicit_index);
        return ModelBundle::load(path)->release_potential();
    }

    std::unique_ptr<AnyPotential> long_range(GPSettings config, std::string input,
                                             const std::string &type, AnyPotential *source)
    {
        std::unique_ptr<AnyPotential> short_source;
        const bool source_long =
            source && dynamic_cast<PotMTP *>(source) &&
            dynamic_cast<PotMTP *>(source)->model_metadata().count("long_range");
        std::string footer;
        int old_channels = 0;
        if (source_long)
        {
            for (const auto &[key, value] : source->CurrentSettings().settings)
                if ((key.starts_with("req_") || key.starts_with("les_") ||
                     key.starts_with("sog_") || key == "lr_sigma" ||
                     key == "remove_self_interaction") &&
                    config[key].empty())
                    config.Modify(key, value);
            Snapshot snapshot(*source);
            Graph graph(snapshot.reader);
            auto &root = graph.node("root");
            auto &c = root.at("configuration").get<Object>();
            auto &fields = c.at("head_fields").get<Array>();
            Array short_fields;
            for (const auto &entry : fields)
            {
                const auto name = string(object(entry).at("name"));
                if (!name.starts_with("les_") && !name.starts_with("sog_") && name != "lr_sigma" &&
                    name != "remove_self_interaction")
                    short_fields.push_back(entry);
            }
            fields = short_fields;
            c.at("metadata").get<Object>().erase("long_range");
            if (type == "mtp_les")
            {
                root["type"] = Value("mtp");
                c.at("metadata").get<Object>()["type"] = Value("MTP");
            }
            root.at("common").get<Object>()["reg_L1"] = graph.numeric({});
            root.at("common").get<Object>()["reg_L2"] = graph.numeric({});
            short_source = graph.load("root");
            input = saved(*short_source);
            footer = model_text(*source);
            const auto &meta = object(object(snapshot.reader->manifest.at("models")).at("root"));
            const auto &lr =
                object(object(object(meta.at("configuration")).at("metadata")).at("long_range"));
            old_channels = integer(lr.at("channels"));
            if (config["les_kernel"].empty())
                config.Modify("les_kernel", string(lr.at("kernel")));
        }
        auto short_config = config;
        short_config.Modify("lr_les", "false");
        short_config.Modify("req_les", "false");
        short_config.Modify("req_sog", "false");
        auto short_model = leaf(short_config, input, type == "mtp_les" ? "mtp" : "mtp_m");
        auto &mtp = dynamic_cast<PotMTP &>(*short_model);
        const auto short_metadata =
            mtp.potential_tag.output(mtp.model_metadata(), mtp.species_count);
        if (type == "mtp_m")
        {
            config.Modify("extmodel", short_metadata.at("model").get<std::string>());
            config.Modify("n_input", std::to_string(int(short_metadata.at("q").get<double>())));
            if (short_metadata.count("nn"))
            {
                auto &nn = short_metadata.at("nn").get<MTPMetadata::Object>();
                auto &layers = nn.at("hidden_layers").get<MTPMetadata::Array>();
                config.Modify("n_neuron", std::to_string(int(layers[0].get<double>())));
                if (layers.size() > 1)
                    config.Modify("n_neuron2", std::to_string(int(layers[1].get<double>())));
                config.Modify("activation_function", nn.at("activation").get<std::string>());
            }
        }
        const auto kernel = config["les_kernel"].empty() ? "coulomb" : config["les_kernel"];
        int channels = 1;
        std::ostringstream definition_text;
        definition_text << std::setprecision(17);
        std::ostringstream text;
        text << std::setprecision(17) << plain_tag(model_text(*short_model));
        if (kernel == "sog")
        {
            SOGModel definition;
            if (source_long && footer.find("les_kernel = sog") != std::string::npos)
            {
                std::istringstream in(footer);
                definition = SOGModel::read(in);
            }
            if (!config["sog_channels"].empty())
                definition.channels = Parameters::parse_value<int>(config["sog_channels"]);
            if (!config["sog_log_bandwidth_min"].empty())
                definition.lower = std::stod(config["sog_log_bandwidth_min"]);
            if (!config["sog_log_bandwidth_max"].empty())
                definition.upper = std::stod(config["sog_log_bandwidth_max"]);
            if (!config["sog_weights"].empty())
                definition.weights = numbers(config["sog_weights"]);
            if (!config["sog_log_bandwidths"].empty())
                definition.log_bandwidths = numbers(config["sog_log_bandwidths"]);
            require(!definition.weights.empty() && !definition.log_bandwidths.empty(),
                    "New SOG models require sog_weights and sog_log_bandwidths");
            if (!config["les_dl"].empty())
                definition.dl = std::stod(config["les_dl"]);
            if (!config["les_taper_fraction"].empty())
                definition.taper = std::stod(config["les_taper_fraction"]);
            if (!config["les_cutoff_mode"].empty())
                definition.smooth = config["les_cutoff_mode"] == "smooth";
            definition.validate();
            channels = definition.channels;
            definition.write(definition_text);
        }
        else
        {
            require(kernel == "coulomb", "Unknown long-range kernel: " + kernel);
            LES definition;
            if (source_long && footer.find("les_kernel = sog") == std::string::npos)
            {
                std::istringstream lines(footer);
                std::ostringstream filtered;
                std::string line;
                const std::set<std::string> fields{"les_version",
                                                   "les_kernel",
                                                   "les_sigma",
                                                   "lr_sigma",
                                                   "les_dl",
                                                   "les_taper_fraction",
                                                   "les_cutoff_mode",
                                                   "les_charge_treatment",
                                                   "remove_self_interaction"};
                while (std::getline(lines, line))
                {
                    auto eq = line.find('=');
                    if (eq != std::string::npos && fields.count(GPM::trim(line.substr(0, eq))))
                        filtered << line << '\n';
                }
                std::istringstream in(filtered.str());
                definition.load_coulomb_definition(in);
            }
            if (!config["lr_sigma"].empty())
                definition.lr_sigma = std::stod(config["lr_sigma"]);
            if (!config["les_dl"].empty())
                definition.dl = std::stod(config["les_dl"]);
            if (!config["les_taper_fraction"].empty())
                definition.taper_fraction = std::stod(config["les_taper_fraction"]);
            if (!config["les_cutoff_mode"].empty())
                definition.cutoff_mode = LES::parse_cutoff_mode(config["les_cutoff_mode"]);
            definition.validate();
            definition.save_coulomb_definition(definition_text);
        }
        std::vector<double> species(channels * mtp.species_count),
            moments(channels * mtp.alpha_scalar_moments);
        std::uniform_real_distribution<double> uniform(lower, upper);
        for (auto *array : {&species, &moments})
            for (auto &value : *array)
                value = randomize ? uniform(RNG.mtd) : 1e-3;
        if (source_long && !reset && old_channels == channels)
        {
            Object fields;
            std::istringstream lines(footer);
            std::string line;
            while (std::getline(lines, line))
            {
                auto eq = line.find('=');
                if (eq != std::string::npos)
                {
                    auto key = GPM::trim(line.substr(0, eq));
                    if (key == "les_species_coeffs" || key == "les_moment_coeffs")
                        fields[key] = Value(GPM::trim(line.substr(eq + 1)));
                }
            }
            for (const auto &[key, destination] :
                 std::vector<std::pair<std::string, std::vector<double> *>>{
                     {"les_species_coeffs", &species}, {"les_moment_coeffs", &moments}})
            {
                require(fields.count(key), "Long-range coefficient section is missing: " + key);
                auto old = numbers(string(fields.at(key)));
                require(old.size() == destination->size(),
                        "Long-range coefficient dimensions changed");
                *destination = std::move(old);
            }
            Message("Reusing long-range charge parameters and kernel definition");
        }
        auto write = [&](const char *name, const std::vector<double> &values)
        {
            text << name << " = {";
            for (size_t i = 0; i < values.size(); ++i)
                text << (i ? ", " : "") << values[i];
            text << "}\n";
        };
        if (type == "mtp_les")
            text << definition_text.str();
        write("les_species_coeffs", species);
        write("les_moment_coeffs", moments);
        if (type != "mtp_les")
            text << definition_text.str();
        const auto path = file(".mtp");
        std::ofstream(path) << text.str();
        config.Modify("potfile", path);
        config.Modify("pottype", type);
        config.Modify("lr_les", "true");
        // Definitions are authoritative in the new native template. Avoid an
        // independent setting changing channel dimensions after allocation.
        config.Modify("typemap", "");
        potential_wrapper wrapper(config);
        auto result = std::unique_ptr<AnyPotential>(std::exchange(wrapper.pot, nullptr));
        if (auto *p = dynamic_cast<PotMTP *>(result.get()))
        {
            p->empty = false;
            p->nl_inited = true;
        }
        set_elements(*result, model_elements(*short_model));
        return result;
    }

    std::unique_ptr<AnyPotential> build(GPSettings config, std::string input,
                                        std::string requested = "")
    {
        if (input.empty())
            input = config["potfile"];
        std::unique_ptr<AnyPotential> source;
        if (!input.empty() && (GPM::is_file(input) || GPM::extension(input)))
            source = load(input, config);
        std::string type = requested.empty() ? config["pottype"] : requested;
        if (type.empty() || type == "auto")
            type = source ? ModelBundle::type(*source) : "mtp";
        if (!source && !input.empty() && !(GPM::is_file(input) || GPM::extension(input)))
        {
            auto stream = open_ifile(input);
            std::string magic;
            if (stream)
                *stream >> magic;
            if (magic == "MTP")
            {
                const auto header = MTPMetadata::read_header(input);
                auto tag = header.tag.text("type");
                if (tag == "MTP_M" || tag == "MTP_LES")
                {
                    GPSettings factory;
                    factory.Modify("pottype", tag == "MTP_M" ? "mtp_m" : "mtp_les");
                    factory.Modify("typemap", "");
                    if (tag == "MTP_M")
                    {
                        factory.Modify("extmodel", header.tag.text("model"));
                        factory.Modify("n_input", std::to_string(int(
                                                      header.tag.fields.at("q").get<double>())));
                        if (header.tag.fields.count("nn"))
                        {
                            auto &nn = header.tag.fields.at("nn").get<MTPMetadata::Object>();
                            auto &layers = nn.at("hidden_layers").get<MTPMetadata::Array>();
                            factory.Modify("n_neuron",
                                           std::to_string(int(layers[0].get<double>())));
                            if (layers.size() > 1)
                                factory.Modify("n_neuron2",
                                               std::to_string(int(layers[1].get<double>())));
                            factory.Modify("activation_function",
                                           nn.at("activation").get<std::string>());
                        }
                    }
                    source = load(input, factory);
                    if (requested.empty() &&
                        (config["pottype"].empty() || config["pottype"] == "auto"))
                        type = ModelBundle::type(*source);
                }
            }
            else if (magic == "pottype" || fs::exists(input + ".nn") || fs::exists(input + "_sub1"))
            {
                auto factory = config;
                if (magic == "pottype")
                {
                    std::string declared;
                    *stream >> declared;
                    factory.Modify("pottype", declared);
                }
                else if (fs::exists(input + ".nn"))
                    factory.Modify("pottype", "mtpnn1");
                else
                {
                    factory.Modify("pottype", "hmtp");
                    factory.Modify("potlevel", fs::exists(input + "_sub3") ? "3" : "2");
                }
                source = load(input, factory);
                if (requested.empty() && (config["pottype"].empty() || config["pottype"] == "auto"))
                    type = ModelBundle::type(*source);
            }
        }
        if (source)
            require(type == ModelBundle::type(*source) || type == "mtp" || type == "mtp_m" ||
                        type == "mtp_les" || type == "mtpnn1" || type == "descriptor_linear",
                    "Changing composite families requires explicit child templates");
        if (source && names.empty())
            names = model_elements(*source);
        if (source && !reset && !names.empty())
        {
            const auto existing = model_elements(*source);
            require(existing.empty() || existing == names,
                    "Use adjustmtp to change existing element identities");
        }
        if (source && !reset && (type == "mtp_les" || type == "mtp_m"))
        {
            bool changed = false;
            for (const auto &[key, value] : config.settings)
                if (key != "potfile" && key != "pottype" && key != "typemap" && !value.empty())
                    changed = true;
            if (!changed)
                return source;
        }
        if (type == "mtp" || type == "mtp_m")
        {
            require(!input.empty(), "MTP initialization requires a template");
            const bool lr =
                config["lr_les"] == "true" || config["lr_les"] == "1" ||
                (source && dynamic_cast<PotMTP *>(source.get()) &&
                 dynamic_cast<PotMTP *>(source.get())->model_metadata().count("long_range"));
            if (lr)
            {
                require(
                    type == "mtp_m",
                    "Long-range models cannot be converted to ordinary MTP without losing parameters");
                return long_range(config, input, type, source.get());
            }
            return leaf(config, input, type);
        }
        if (type == "mtp_les")
        {
            require(!input.empty(), "LES initialization requires an MTP template");
            if (!source && GPM::is_file(input))
                source = load(input, config);
            return long_range(config, input, type, source.get());
        }
        if (type == "mtpnn1" || type == "descriptor_linear")
        {
            const bool matching = source && ModelBundle::type(*source) == type;
            std::string descriptor_input = input;
            if (matching)
            {
                Snapshot snapshot(*source);
                const auto &root =
                    object(object(snapshot.reader->manifest.at("models")).at("root"));
                const auto &children = object(root.at("children"));
                require(children.count("descriptor"),
                        "Only MTM-backed linear descriptor models support this workflow");
                auto descriptor =
                    GPMCodecAccess::decode(*snapshot.reader, string(children.at("descriptor")));
                descriptor_input = saved(*descriptor);
                if (!reset)
                {
                    if (type == "descriptor_linear")
                        return source;
                    auto &nn = dynamic_cast<PotMTPNN1 &>(*source);
                    const int layers =
                        config["nnlay"].empty() ? nn.nnlay : std::stoi(config["nnlay"]);
                    const int hidden = nn.nnlay == 1 ? nn.p_nn->n_neuron : nn.p_nn2->n_neuron1;
                    if (layers == nn.nnlay &&
                        (config["nnhidden"].empty() || std::stoi(config["nnhidden"]) == hidden) &&
                        (config["nnhidden2"].empty() ||
                         (nn.nnlay == 2 && std::stoi(config["nnhidden2"]) == nn.p_nn2->n_neuron2)))
                        return source;
                }
                if (type == "mtpnn1")
                {
                    auto &nn = dynamic_cast<PotMTPNN1 &>(*source);
                    if (config["nnlay"].empty())
                        config.Modify("nnlay", std::to_string(nn.nnlay));
                    if (config["nnhidden"].empty())
                        config.Modify("nnhidden",
                                      std::to_string(nn.nnlay == 1 ? nn.p_nn->n_neuron
                                                                   : nn.p_nn2->n_neuron1));
                    if (nn.nnlay == 2 && config["nnhidden2"].empty())
                        config.Modify("nnhidden2", std::to_string(nn.p_nn2->n_neuron2));
                }
            }
            require(!descriptor_input.empty(),
                    "Descriptor initialization requires an MTP template");
            auto desc_config = config;
            desc_config.Modify("pottype", "mtp");
            auto descriptor = leaf(desc_config, descriptor_input, "mtp");
            auto *mtp = dynamic_cast<PotMTP *>(descriptor.get());
            require(mtp, "MTM requires an ordinary MTP");
            if (type == "descriptor_linear")
            {
                GPSettings settings;
                settings.Modify("descriptor:dtype", "mtm");
                settings.Modify("species_count", std::to_string(mtp->species_count));
                auto model = std::make_unique<PotentialDescriptorLinear>(
                    settings, std::make_unique<DescriptorMTM>(*mtp));
                model->gpm_resources = std::shared_ptr<AnyPotential>(descriptor.release());
                Eigen::VectorXd weights(model->get_NL_coeffs_count());
                std::uniform_real_distribution<double> uniform(lower, upper);
                for (auto &value : weights)
                    value = randomize ? scale * uniform(RNG.mtd) : 0.;
                model->nl_set_coeffs(weights.data());
                return model;
            }
            require(!config["nnhidden"].empty() && std::stoi(config["nnhidden"]) > 0,
                    "New MTPNN requires positive nnhidden");
            if (config["nnlay"].empty())
                config.Modify("nnlay", "1");
            const int layers = std::stoi(config["nnlay"]);
            require(layers == 1 || layers == 2, "MTPNN nnlay must be 1 or 2");
            if (layers == 2)
                require(!config["nnhidden2"].empty() && std::stoi(config["nnhidden2"]) > 0,
                        "Two-layer MTPNN requires positive nnhidden2");
            config.Modify("mtpcoeff", saved(*descriptor));
            config.Modify("nncoeff", file(".missing.nn"));
            config.Modify("spcoeff", file(".missing.sc"));
            auto model = std::make_unique<PotMTPNN1>(config);
            model->random_init_mtp = false;
            if (randomize)
                model->random_init();
            else
            {
                std::vector<double> weights(model->nn_coeff_size, 1e-3);
                if (layers == 1)
                    model->p_nn->set_coeffs(weights.data());
                else
                    model->p_nn2->set_coeffs(weights.data());
            }
            if (matching && !reset)
                model->species_coeffs = dynamic_cast<PotMTPNN1 &>(*source).species_coeffs;
            model->set_inited();
            return model;
        }
        if (type == "hybrid" || type == "hybrid_scale" || type == "hmtp")
        {
            if (source && !reset)
            {
                bool structural = false;
                for (const auto &[key, value] : config.settings)
                    if (key != "potfile" && key != "pottype" && key != "typemap" && !value.empty())
                        structural = true;
                if (!structural)
                    return source;
            }
            if (source)
            {
                Snapshot snapshot(*source);
                const auto &root =
                    object(object(snapshot.reader->manifest.at("models")).at("root"));
                for (const auto &[key, value] : object(root.at("configuration")))
                    if (config[key].empty())
                        config.Modify(key, string(value));
                for (const auto &[role, reference] : object(root.at("children")))
                {
                    auto child = GPMCodecAccess::decode(*snapshot.reader, string(reference));
                    const auto prefix = role + ":";
                    if (config[prefix + "potfile"].empty())
                        config.Modify(prefix + "potfile", saved(*child));
                }
            }
            require(!config["npot"].empty(),
                    "Composite initialization requires npot and explicit child templates");
            const int count = std::stoi(config["npot"]);
            require(count > 0, "Composite npot must be positive");
            std::vector<std::unique_ptr<AnyPotential>> children;
            for (int i = 1; i <= count; ++i)
            {
                auto child = config.ExtractSubSettings("pot" + std::to_string(i));
                const auto kind = child["pottype"];
                require(!kind.empty(), "Missing composite child type at pot" + std::to_string(i));
                if (kind == "zbl" || kind == "eampoly")
                {
                    if (!child["potfile"].empty())
                        children.push_back(load(child["potfile"], child));
                    else if (kind == "zbl")
                    {
                        if (child["z"].empty())
                        {
                            require(!names.empty(), "ZBL initialization requires elements or z");
                            std::string z;
                            for (auto &name : names)
                            {
                                require(Chemicals::atomic_numbers.count(name),
                                        "Unknown element for ZBL");
                                z += std::to_string(Chemicals::atomic_numbers.at(name)) + " ";
                            }
                            child.Modify("z", z);
                        }
                        potential_wrapper wrapper(child);
                        children.emplace_back(std::exchange(wrapper.pot, nullptr));
                    }
                    else
                        require(false, "An EAMPoly branch must supply a complete model file");
                }
                else
                    children.push_back(build(child, child["potfile"], kind));
            }
            config.Modify("potfile", "");
            std::unique_ptr<AnyPotential> model;
            if (type == "hybrid")
                model = std::make_unique<PotentialHybrid>(config, std::move(children));
            else if (type == "hybrid_scale")
                model = std::make_unique<PotentialHybridScale>(config, std::move(children));
            else
            {
                require(count == 2 || count == 3, "HMTP requires two or three MTP children");
                for (auto &p : children)
                    require(typeid(*p) == typeid(PotMTP), "HMTP children must be ordinary MTP");
                PotMTPHyb3 temporary(static_cast<PotMTP *>(children[0].get()),
                                     static_cast<PotMTP *>(children[1].get()),
                                     count == 3 ? static_cast<PotMTP *>(children[2].get())
                                                : nullptr);
                Snapshot snapshot(temporary);
                return ModelBundle::load((snapshot.stage.path / "source.gpm").string())
                    ->release_potential();
            }
            return model;
        }
        require(false, "Initialization is not defined for model family: " + type);
        return {};
    }
};

std::string resolve(Graph &graph, std::string path)
{
    auto id = string(graph.writer.manifest.at("root_model"));
    if (path.empty() || path == "root")
        return id;
    if (path.starts_with("root/"))
        path.erase(0, 5);
    std::replace(path.begin(), path.end(), '.', '/');
    std::istringstream in(path);
    std::string role;
    while (std::getline(in, role, '/'))
    {
        require(!role.empty(), "Empty offset target path component");
        require(role != "descriptor", "Offset targets must be energy branches, not descriptors");
        const auto &children = object(graph.node(id).at("children"));
        require(children.count(role), "Unknown offset target: " + path);
        id = string(children.at(role));
    }
    return id;
}

void dataset_operations(std::unique_ptr<AnyPotential> &model, const std::string &dataset,
                        const std::string &format, const std::string &index, bool explicit_index,
                        bool fit, bool normalize, const std::string &target)
{
    require(!(fit || normalize) || !dataset.empty(),
            "Offset fitting and normalization require a dataset");
    if (dataset.empty())
        return;
    auto names = model_elements(*model);
    GPIO io;
    if (!names.empty())
        io.setmap(names);
    if (explicit_index)
    {
        require(fs::exists(index), "Explicit element map does not exist: " + index);
        GPIO check;
        check.readmap(index);
        if (!names.empty())
            require(check.getmap() == io.getmap(), "Explicit map conflicts with model elements");
        io.setmap(check.getmap());
    }
    else if (names.empty() && !index.empty() && fs::exists(index))
        io.readmap(index);
    std::vector<Atoms> frames;
    require(io.read(frames, dataset, format) && !frames.empty(),
            "Initialization requires a nonempty dataset");
    for (const auto &frame : frames)
    {
        require(frame.size() > 0, "Empty initialization structure");
        for (int id : frame.chemmap_)
            require(id >= 0 && id < model->species_count,
                    "Dataset element is outside model mapping");
        if (frame.have_energy)
            require(std::isfinite(frame.energy_), "Nonfinite initialization energy");
    }
    Snapshot snapshot(*model);
    Graph graph(snapshot.reader);
    const auto root = string(snapshot.reader->manifest.at("root_model"));
    // Report statistics for every MTP branch without changing reused parameters.
    for (const auto &[id, value] : object(snapshot.reader->manifest.at("models")))
    {
        auto &node = object(value);
        const auto kind = string(node.at("type"));
        if (kind != "mtp" && kind != "mtp_m" && kind != "mtp_les")
            continue;
        auto leaf = GPMCodecAccess::decode(*snapshot.reader, id);
        auto &mtp = dynamic_cast<PotMTP &>(*leaf);
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(mtp.alpha_scalar_moments), m2 = mean;
        size_t sites = 0;
        std::vector<double> basis(mtp.alpha_scalar_moments + 1);
        for (auto frame : frames)
        {
            Neighborhoods neighbors(frame, mtp.cutoff);
            for (int i = 0; i < frame.size(); ++i)
            {
                mtp.compute_descriptors(neighbors[i], basis);
                ++sites;
                for (int k = 0; k < mean.size(); ++k)
                {
                    double delta = basis[k + 1] - mean[k];
                    mean[k] += delta / sites;
                    m2[k] += delta * (basis[k + 1] - mean[k]);
                }
            }
        }
        require(mean.allFinite() && m2.allFinite(), "Nonfinite descriptor statistics at " + id);
        Message("Basis statistics " + id + ": sites=" + std::to_string(sites) +
                ", mean norm=" + std::to_string(mean.norm()));
    }
    if (!fit && !normalize)
        return;
    const auto root_type = string(graph.node(root).at("type"));
    const bool composite =
        root_type == "hybrid" || root_type == "hybrid_scale" || root_type == "hmtp";
    require(!composite || !target.empty(),
            "Composite offset fitting/normalization requires --offset-target");
    const auto id = resolve(graph, target);
    auto &node = graph.node(id);
    const auto kind = string(node.at("type"));
    require(kind == "mtp" || kind == "mtp_m" || kind == "mtp_les" || kind == "mtpnn1",
            "Target has no independent elemental energy offsets: " + id);
    const int species = array(node.at("species")).size();
    require(species > 0, "Offset target has no declared species");
    if (normalize)
    {
        require(
            kind != "mtpnn1",
            "Normalize the MTP descriptor through its explicit model workflow; NN inputs must retain their trained normalization");
        auto selected = GPMCodecAccess::decode(*snapshot.reader, id);
        auto &mtp = dynamic_cast<PotMTP &>(*selected);
        const int b = mtp.alpha_scalar_moments;
        require(b > 0, "Normalization requires scalar moments");
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(b), m2 = mean;
        std::vector<double> basis(b + 1);
        size_t sites = 0;
        for (auto frame : frames)
        {
            Neighborhoods neighbors(frame, mtp.cutoff);
            for (int i = 0; i < frame.size(); ++i)
            {
                mtp.compute_descriptors(neighbors[i], basis);
                ++sites;
                for (int k = 0; k < b; ++k)
                {
                    double d = basis[k + 1] - mean[k];
                    mean[k] += d / sites;
                    m2[k] += d * (basis[k + 1] - mean[k]);
                }
            }
        }
        auto &parameters = node.at("parameters").get<Object>();
        auto moments = values(*snapshot.reader, parameters.at("moments")),
             offsets = values(*snapshot.reader, parameters.at("species"));
        for (int c = 0; c < int(moments.size()) / b; ++c)
        {
            double bias = 0.;
            for (int k = 0; k < b; ++k)
            {
                moments[c * b + k] =
                    m2[k] / sites > 1e-12 ? moments[c * b + k] / std::sqrt(m2[k] / sites) : 0.;
                bias += moments[c * b + k] * mean[k];
            }
            const int first = kind == "mtp_m" ? (c + 1) * species : 0;
            for (int i = 0; i < species; ++i)
                offsets[first + i] -= bias;
        }
        parameters["moments"] = graph.numeric(moments);
        parameters["species"] = graph.numeric(offsets);
    }
    if (fit)
    {
        auto &container =
            kind == "mtpnn1" ? node.at("state").get<Object>() : node.at("parameters").get<Object>();
        const auto key = kind == "mtpnn1" ? "species_coefficients" : "species";
        auto selected = graph.load(id);
        Snapshot current(*selected);
        auto &current_node = object(object(current.reader->manifest.at("models")).at("root"));
        auto &current_container =
            object(current_node.at(kind == "mtpnn1" ? "state" : "parameters"));
        auto coefficients = values(*current.reader, current_container.at(key));
        graph.writer.manifest["root_model"] = Value(root);
        graph.writer.manifest["species"] = graph.node(root).at("species");
        std::fill_n(coefficients.begin(), species, 0.);
        container[key] = graph.numeric(coefficients);
        auto baseline = graph.load();
        std::vector<Atoms> labeled;
        for (const auto &frame : frames)
            if (frame.have_energy)
                labeled.push_back(frame);
        require(!labeled.empty(), "Offset fitting requires energy labels");
        Eigen::VectorXd base(labeled.size()), rhs(labeled.size());
        Eigen::MatrixXd design(labeled.size(), species);
        for (size_t i = 0; i < labeled.size(); ++i)
        {
            auto frame = labeled[i];
            baseline->getEFS(frame);
            base[i] = frame.energy_;
            rhs[i] = (labeled[i].energy_ - base[i]) / std::sqrt(double(frame.size()));
        }
        for (int s = 0; s < species; ++s)
        {
            coefficients[s] = 1.;
            container[key] = graph.numeric(coefficients);
            auto unit = graph.load();
            coefficients[s] = 2.;
            container[key] = graph.numeric(coefficients);
            auto twice = graph.load();
            for (size_t i = 0; i < labeled.size(); ++i)
            {
                auto one = labeled[i], two = one;
                unit->getEFS(one);
                twice->getEFS(two);
                require(std::abs(two.energy_ - base[i] - 2 * (one.energy_ - base[i])) <=
                            1e-9 * (1 + std::abs(two.energy_)),
                        "Offset target is not affine in complete model energy");
                design(i, s) = (one.energy_ - base[i]) / std::sqrt(double(one.size()));
            }
            coefficients[s] = 0.;
        }
        Eigen::FullPivLU<Eigen::MatrixXd> solve(design);
        require(solve.rank() == species, "Composite species offset response is rank deficient");
        Eigen::VectorXd fitted = solve.solve(rhs);
        require(fitted.allFinite(), "Nonfinite composite energy offsets");
        std::copy_n(fitted.data(), species, coefficients.data());
        container[key] = graph.numeric(coefficients);
    }
    model = graph.load();
    Message("Applied initialization changes to energy branch " + id);
}
}

void initialize_model(GPSettings settings, std::string input, std::string output, std::string type,
                      const std::vector<std::string> &elements, bool reset, bool randomize,
                      double lower, double upper, std::string dataset, const std::string &format,
                      const std::string &index, bool fit, bool normalize, double scale,
                      bool explicit_index, const std::string &offset_target)
{
    LocalComputation local;
    require(std::isfinite(lower) && std::isfinite(upper) && lower <= upper &&
                std::isfinite(scale) && scale >= 0,
            "Invalid initialization random bounds/scale");
    require(offset_target.empty() || fit || normalize,
            "--offset-target requires offset fitting or normalization");
    auto config = settings.ExtractSubSettings("pot");
    if (input.empty())
        input = config["potfile"];
    if (output.empty())
    {
        output = "_Initialized_Potential_.gpm";
        require(input.empty() || fs::absolute(input).lexically_normal() !=
                                     fs::absolute(output).lexically_normal(),
                "Explicit output is required to replace input");
    }
    Builder builder{{}, 0, reset, randomize, lower, upper, scale, elements, index, explicit_index};
    if (builder.names.empty() && explicit_index)
    {
        require(fs::exists(index), "Explicit element map does not exist: " + index);
        GPIO io;
        io.readmap(index);
        for (const auto &[id, name] : io.getmap())
        {
            require(id == int(builder.names.size()), "Nonconsecutive element map");
            builder.names.push_back(name);
        }
    }
    auto model = builder.build(config, input, type);
    if (!builder.names.empty())
        set_elements(*model, builder.names);
    dataset_operations(model, dataset, format, index, explicit_index, fit, normalize,
                       offset_target);
    save(*model, output);
}
}
