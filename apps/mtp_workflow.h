#pragma once
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/model_bundle.h"
#include "tools/gpm_mtp.h"
#include "train/collective_error.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>

namespace MTPWorkflow
{
namespace fs = std::filesystem;

// Root-only command work may call model routines containing MPI collectives.
// Keep those collectives local, then restore the world before propagating errors.
struct LocalComputation
{
#ifdef IMRMLP_MPI
    int rank = mpi.rank, size = mpi.size;
    MPI_Comm comm = mpi.comm;

    LocalComputation()
    {
        mpi.rank = 0;
        mpi.size = 1;
        mpi.comm = MPI_COMM_SELF;
    }

    ~LocalComputation()
    {
        mpi.rank = rank;
        mpi.size = size;
        mpi.comm = comm;
    }
#endif
};

struct Stage
{
    fs::path path;

    explicit Stage(const fs::path &parent = fs::temp_directory_path())
    {
        auto pattern = (parent / ".mtp-workflow-XXXXXX").string();
        if (!mkdtemp(pattern.data()))
            GPERROR("Cannot stage MTP workflow output");
        path = pattern;
    }

    ~Stage()
    {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
};

inline void save(Potentials::AnyPotential &model, const std::string &output)
{
    if (output.empty())
        GPERROR("Output must name a model file");
    const auto target = fs::absolute(output).lexically_normal();
    Stage stage(target.parent_path());
    const auto path = (stage.path / "model.gpm").string();
    Potentials::ModelBundle::save_potential(model, path,
                                            Potentials::ModelBundle::Profile::Inference);
    Potentials::ModelBundle::validate(path);
    if (GPM::extension(output))
        fs::rename(path, target);
    else
        Potentials::ModelBundle::load(path)->save_native(
            output, Potentials::ModelBundle::Profile::Inference);
}

inline std::string topology(Potentials::PotMTP &source, int count)
{
    Potentials::PotMTP base(source);
    if (base.radial_basis_split && count != base.species_count)
        GPERROR("Changing a split template's species count requires explicit basis cloning");
    base.species_count = count;
    base.potential_tag.fields.clear();
    std::ostringstream text;
    base.write_header(text);
    base.write_radial_info(text);
    base.write_rules(text);
    auto s = text.str();
    const auto first = s.find("potential_tag =");
    if (first != std::string::npos)
        s.replace(first, s.find('\n', first) - first, "potential_tag =");
    return s;
}

inline void validate_graph(const Potentials::PotMTP &model)
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

inline void reorder_basis(Potentials::PotMTP &model, const std::string &order)
{
    validate_graph(model);
    if (order == "keep")
        return;
    if (order != "level" && order != "level-group")
        GPERROR("Unknown basis order: " + order);
    const int scalars = model.alpha_scalar_moments;
    if (!scalars || model.moment_coeff_size % scalars)
        GPERROR("Unsupported scalar parameter layout");
    for (const auto *regularizer : {&model.reg_L1_vector, &model.reg_L2_vector})
        if (!regularizer->empty() && regularizer->size() != size_t(model.reg_coeff_size))
            GPERROR("Packed regularization requires an explicit basis migration map");
    int nr = 0, ns = 0, nm = 0, nh = 0;
    if (Potentials::ModelBundle::type(model) == "mtp_m")
        model.get_n_coeffs(nr, ns, nm, nh);
    const int first_extra =
        model.radial_coeff_size + model.species_coeff_size + model.moment_coeff_size + nh;
    const int extra = model.reg_coeff_size - first_extra;
    if (extra < 0 || extra % (model.species_count + scalars))
        GPERROR("Unknown long-range parameter layout");
    const int channels = extra / (model.species_count + scalars),
              first_latent = first_extra + channels * model.species_count;
    std::vector<double> latent(model.regression_coeffs + first_latent,
                               model.regression_coeffs + model.reg_coeff_size);
    std::vector<double> coefficients(model.moment_coeffs,
                                     model.moment_coeffs + model.moment_coeff_size);
    for (int i = 0; i < scalars; ++i)
        model.moment_coeffs[i] = i;
    if (order == "level")
        model.ReorderByLevel();
    else
        model.ReorderByLevelGroup();
    std::vector<int> permutation(scalars);
    for (int i = 0; i < scalars; ++i)
        permutation[i] = int(model.moment_coeffs[i]);
    for (int channel = 0; channel < model.moment_coeff_size / scalars; ++channel)
        for (int i = 0; i < scalars; ++i)
            model.moment_coeffs[channel * scalars + i] =
                coefficients[channel * scalars + permutation[i]];
    for (int c = 0; c < channels; ++c)
        for (int i = 0; i < scalars; ++i)
            model.regression_coeffs[first_latent + c * scalars + i] =
                latent[c * scalars + permutation[i]];
    for (auto *regularizer : {&model.reg_L1_vector, &model.reg_L2_vector})
        if (!regularizer->empty())
        {
            auto previous = *regularizer;
            const int first = model.radial_coeff_size + model.species_coeff_size;
            for (int channel = 0; channel < model.moment_coeff_size / scalars; ++channel)
                for (int i = 0; i < scalars; ++i)
                    (*regularizer)[first + channel * scalars + i] =
                        previous[first + channel * scalars + permutation[i]];
            for (int c = 0; c < channels; ++c)
                for (int i = 0; i < scalars; ++i)
                    (*regularizer)[first_latent + c * scalars + i] =
                        previous[first_latent + c * scalars + permutation[i]];
        }
    model.omp_inited = false;
}

inline std::unique_ptr<Potentials::AnyPotential> permute(Potentials::PotMTP &model,
                                                         const std::vector<int> &order)
{
    using namespace GPM;
    const int n = model.species_count;
    if (order.size() != size_t(n))
        GPERROR("Element permutation must contain every existing species");
    std::set<int> seen;
    for (int id : order)
        if (id < 0 || id >= n || !seen.insert(id).second)
            GPERROR("Invalid element permutation");
    Stage stage;
    const auto input = (stage.path / "input.gpm").string(),
               output = (stage.path / "output.gpm").string();
    Potentials::ModelBundle::save_potential(model, input,
                                            Potentials::ModelBundle::Profile::Inference);
    auto reader = std::make_shared<Reader>(input);
    Writer writer;
    writer.preserve(reader);
    auto &node = writer.manifest.at("models")
                     .get<Object>()
                     .at(string(writer.manifest.at("root_model")))
                     .get<Object>();
    auto &config = node.at("configuration").get<Object>();
    auto &parameters = node.at("parameters").get<Object>();
    auto reorder = [&](Value &field, bool pair, const std::string &name)
    {
        auto previous = values(*reader, field), next = previous;
        const int groups = pair ? n * n : n;
        if (previous.size() % groups)
            GPERROR("Inconsistent species parameter dimensions: " + name);
        const int stride = previous.size() / groups;
        if (pair)
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    for (int k = 0; k < stride; ++k)
                        next[(i * n + j) * stride + k] =
                            previous[(order[i] * n + order[j]) * stride + k];
        else
            for (int channel = 0; channel < stride; ++channel)
                for (int i = 0; i < n; ++i)
                    next[channel * n + i] = previous[channel * n + order[i]];
        field = vector_block(writer, "permuted/" + name, next);
    };
    reorder(parameters.at("radial"), true, "radial");
    reorder(parameters.at("species"), false, "species");
    auto &bases = config.at("radial_bases").get<Array>();
    if (bases.size() > 1)
    {
        auto previous = bases;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                bases[i * n + j] = previous[order[i] * n + order[j]];
    }
    auto &common = node.at("common").get<Object>();
    for (const auto *key : {"cutoffs", "r_mins"})
    {
        auto old = values(*reader, common.at(key)), next = old;
        if (!old.empty())
        {
            if (old.size() != size_t(n * n))
                GPERROR("Invalid species matrix");
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    next[i * n + j] = old[order[i] * n + order[j]];
            common[key] = vector_block(writer, std::string("permuted/") + key, next,
                                       {uint64_t(n), uint64_t(n)});
        }
    }
    for (auto &field : config.at("head_fields").get<Array>())
    {
        auto &entry = field.get<Object>();
        const auto name = string(entry.at("name"));
        if (entry.count("array") && name.find("les") != std::string::npos &&
            name.find("species") != std::string::npos)
            reorder(entry.at("array"), false, "head/" + name);
    }
    for (const auto *key : {"reg_L1", "reg_L2"})
    {
        auto old = values(*reader, common.at(key));
        if (old.empty())
            continue;
        if (old.size() != size_t(model.reg_coeff_size))
            GPERROR("Packed regularization species migration is not defined");
        auto next = old;
        const int stride = model.radial_coeff_size / (n * n);
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                for (int k = 0; k < stride; ++k)
                    next[(i * n + j) * stride + k] = old[(order[i] * n + order[j]) * stride + k];
        for (int c = 0; c < model.species_coeff_size / n; ++c)
            for (int i = 0; i < n; ++i)
                next[model.radial_coeff_size + c * n + i] =
                    old[model.radial_coeff_size + c * n + order[i]];
        int nr = 0, ns = 0, nm = 0, nh = 0;
        if (Potentials::ModelBundle::type(model) == "mtp_m")
            model.get_n_coeffs(nr, ns, nm, nh);
        const int first =
            model.radial_coeff_size + model.species_coeff_size + model.moment_coeff_size + nh;
        const int extra = model.reg_coeff_size - first;
        if (extra < 0 || extra % (n + model.alpha_scalar_moments))
            GPERROR("Unknown long-range regularization layout");
        for (int c = 0; c < extra / (n + model.alpha_scalar_moments); ++c)
            for (int i = 0; i < n; ++i)
                next[first + c * n + i] = old[first + c * n + order[i]];
        common[key] = vector_block(writer, std::string("permuted/") + key, next);
    }
    auto species = node.at("species").get<Array>(), next = species;
    for (int i = 0; i < n; ++i)
    {
        next[i] = species[order[i]];
        next[i].get<Object>()["id"] = Value(double(i));
    }
    node["species"] = Value(next);
    writer.manifest["species"] = Value(next);
    auto &meta = config.at("metadata").get<Object>();
    if (meta.count("elements"))
    {
        auto previous = meta.at("elements").get<Array>();
        auto updated = previous;
        for (int i = 0; i < n; ++i)
            updated[i] = previous[order[i]];
        meta["elements"] = Value(updated);
    }
    writer.save(output);
    return Potentials::ModelBundle::load(output)->release_potential();
}

inline void initialize(GPSettings settings, std::string input, std::string output, std::string type,
                       const std::vector<std::string> &elements, bool reset, bool randomize,
                       double lower, double upper, std::string dataset, const std::string &format,
                       const std::string &index, bool fit, bool normalize, double scale = 1.,
                       bool explicit_index = false)
{
    using namespace Potentials;
    LocalComputation local;
    if (!std::isfinite(scale) || scale < 0)
        GPERROR("Invalid random moment scale");
    auto config = settings.ExtractSubSettings("pot");
    if (input.empty())
        input = config["potfile"];
    if (input.empty())
        GPERROR("Initialization requires --input or pot:potfile");
    if (type.empty())
        type = config["pottype"];
    const bool infer_type = type.empty() || type == "auto";
    if (infer_type)
        type = "mtp";
    if (type != "mtp" && type != "mtp_m")
        GPERROR("Template initialization is not defined for target " + type + " in this entry");
    if (!(std::isfinite(lower) && std::isfinite(upper) && lower <= upper))
        GPERROR("Invalid random coefficient bounds");
    if (output.empty())
    {
        output = "_Initialized_Potential_.gpm";
        if (fs::absolute(input).lexically_normal() == fs::absolute(output).lexically_normal())
            GPERROR("Explicit --output is required to overwrite the input");
    }
    Stage stage;
    std::unique_ptr<AnyPotential> model;
    std::shared_ptr<ModelBundle> source_bundle;
    std::unique_ptr<PotMTP> ordinary;
    PotMTP *source = nullptr;
    if (GPM::is_file(input) || GPM::extension(input))
    {
        source_bundle = ModelBundle::load(input);
        source = dynamic_cast<PotMTP *>(&source_bundle->potential());
        if (!source)
            GPERROR("Initialization currently requires a single MTP or MTP-M model");
    }
    else
    {
        const auto header = MTPMetadata::read_header(input);
        if (header.tag.text("type") == "MTP_M")
        {
            GPSettings native;
            native.Modify("potfile", input);
            native.Modify("pottype", "mtp_m");
            native.Modify("typemap", "");
            native.Modify("extmodel", header.tag.text("model"));
            if (header.tag.fields.count("q"))
                native.Modify("n_input",
                              std::to_string(int(header.tag.fields.at("q").get<double>())));
            if (header.tag.fields.count("nn"))
            {
                const auto &nn = header.tag.fields.at("nn").get<MTPMetadata::Object>();
                const auto &layers = nn.at("hidden_layers").get<MTPMetadata::Array>();
                native.Modify("n_neuron", std::to_string(int(layers.at(0).get<double>())));
                if (layers.size() > 1)
                    native.Modify("n_neuron2", std::to_string(int(layers.at(1).get<double>())));
                native.Modify("activation_function", nn.at("activation").get<std::string>());
            }
            potential_wrapper loaded(native);
            model.reset(loaded.pot);
            loaded.pot = nullptr;
            source = dynamic_cast<PotMTP *>(model.get());
        }
        else
        {
            ordinary = std::make_unique<PotMTP>();
            ordinary->Load(input);
            source = ordinary.get();
        }
    }
    const auto source_type = ModelBundle::type(*source);
    if (infer_type && source->inited())
        type = source_type;
    if (type != "mtp" && type != "mtp_m")
        GPERROR("Initialization layout is not yet defined for " + type);
    const auto source_meta =
        source->potential_tag.output(source->model_metadata(), source->species_count);
    if (source_meta.count("long_range") && reset)
        GPERROR("Reinitializing a long-range model is not yet implemented");
    bool compatible_head = true;
    if (type == "mtp_m" && !config["extmodel"].empty() && source_type == type)
        compatible_head = MTPMetadata::canonical_model(config["extmodel"]) ==
                          source_meta.at("model").get<std::string>();
    if (source_type == "mtp_m" && source_meta.count("q") && !config["n_input"].empty())
        compatible_head = compatible_head &&
                          std::stoi(config["n_input"]) == int(source_meta.at("q").get<double>());

    if (type == "mtp_m" && source_type == "mtp_m")
    {
        if (config["extmodel"].empty())
            config.Modify("extmodel", source_meta.at("model").get<std::string>());
        if (config["n_input"].empty() && source_meta.count("q"))
            config.Modify("n_input", std::to_string(int(source_meta.at("q").get<double>())));
        if (source_meta.count("nn"))
        {
            const auto &nn = source_meta.at("nn").get<MTPMetadata::Object>();
            const auto &layers = nn.at("hidden_layers").get<MTPMetadata::Array>();
            for (size_t i = 0; i < layers.size(); ++i)
            {
                const auto key = i ? "n_neuron2" : "n_neuron";
                const int width = layers[i].get<double>();
                if (config[key].empty())
                    config.Modify(key, std::to_string(width));
                else if (std::stoi(config[key]) != width)
                    compatible_head = false;
            }
            if (config["activation_function"].empty())
                config.Modify("activation_function", nn.at("activation").get<std::string>());
            else
            {
                auto requested = config["activation_function"],
                     saved = nn.at("activation").get<std::string>();
                for (auto *value : {&requested, &saved})
                    for (auto &ch : *value)
                        ch = std::toupper(static_cast<unsigned char>(ch));
                if (requested != saved)
                    compatible_head = false;
            }
        }
    }
    if (source_meta.count("long_range") && (!compatible_head || type != source_type))
        GPERROR("Changing the head of a long-range model requires a dedicated layout migration");
    auto names = elements.empty() ? source->potential_tag.elements() : elements;
    if (names.empty() && source->potential_tag.elements().empty() && !index.empty() &&
        fs::exists(index))
    {
        GPIO mapping;
        mapping.readmap(index);
        for (const auto &[id, name] : mapping.getmap())
        {
            if (id != int(names.size()))
                GPERROR("Element map must use consecutive IDs");
            names.push_back(name);
        }
    }
    if (explicit_index)
    {
        if (!fs::exists(index))
            GPERROR("Explicit element map does not exist: " + index);
        GPIO mapping;
        mapping.readmap(index);
        std::vector<std::string> mapped;
        for (const auto &[id, name] : mapping.getmap())
        {
            if (id != int(mapped.size()))
                GPERROR("Element map must use consecutive IDs");
            mapped.push_back(name);
        }
        if (!names.empty() && mapped != names)
            GPERROR("Explicit element map conflicts with declared elements");
        names = std::move(mapped);
    }
    const int count = names.empty() ? source->species_count : names.size();
    if (count <= 0)
        GPERROR("Initialization requires at least one element");
    if (!reset && source->inited() &&
        (count != source->species_count ||
         (!names.empty() && !source->potential_tag.elements().empty() &&
          names != source->potential_tag.elements())))
        GPERROR("Use adjustmtp or --reinitialize to change elements of an initialized model");
    if (!reset && source_type == type && compatible_head && source->inited())
    {
        if (count != source->species_count)
            GPERROR("Use adjustmtp to change elements of an initialized model");
        if (!elements.empty() && !source->potential_tag.elements().empty() &&
            elements != source->potential_tag.elements())
            GPERROR("Reused model element identities cannot be renamed");
        if (source_bundle)
            model = source_bundle->release_potential();
        else if (ordinary)
            model = std::move(ordinary);
        Message("Reusing complete model parameters");
    }
    else
    {
        auto source_owner = std::move(model);
        const auto path = (stage.path / "topology.mtp").string();
        std::ofstream(path) << topology(*source, count);
        config.Modify("potfile", path);
        config.Modify("pottype", type);
        config.Modify("typemap", "");
        if (type == "mtp_m")
        {
            if (config["extmodel"].empty())
                GPERROR("Creating MTP-M requires pot:extmodel");
            const auto head = MTPMetadata::canonical_model(config["extmodel"]);
            if (head == "mtp_m_n1" || head == "mtp_m_n2")
            {
                for (const auto *key : {"n_input", "n_neuron"})
                    if (config[key].empty() || std::stoi(config[key]) <= 0)
                        GPERROR(std::string("Creating an NN head requires positive pot:") + key);
                if (head == "mtp_m_n2" &&
                    (config["n_neuron2"].empty() || std::stoi(config["n_neuron2"]) <= 0))
                    GPERROR("Creating a two-layer NN requires positive pot:n_neuron2");
            }
            model.reset(create_mtp_m_potential(config));
        }
        else
            model = std::make_unique<PotMTP>(path);
        auto &target = dynamic_cast<PotMTP &>(*model);
        target.random_lower = lower;
        target.random_upper = upper;
        if (randomize)
        {
            model->random_init();
            for (int i = 0; i < target.moment_coeff_size; ++i)
                target.moment_coeffs[i] *= scale;
        }
        if (type == "mtp_m")
            model->init_coeffs();
        else
        {
            target.empty = false;
            target.nl_inited = true;
        }
        if (!reset && count == source->species_count)
        {
            // Topology is unchanged; radial coordinates have identical physical meaning.
            if (!source->radial_empty)
            {
                std::copy_n(source->radial_coeffs, target.radial_coeff_size, target.radial_coeffs);
                PotMTP basis_copy(*source);
                std::swap(target.p_radical_basis, basis_copy.p_radical_basis);
                target.p_radical_basis_pair.swap(basis_copy.p_radical_basis_pair);
                target.cutoffs = source->cutoffs;
                target.cutoffsqs = source->cutoffsqs;
                target.r_mins = source->r_mins;
            }
            if (type == source_type && compatible_head &&
                target.moment_coeff_size == source->moment_coeff_size && !source->linear_empty)
                std::copy_n(source->moment_coeffs, target.moment_coeff_size, target.moment_coeffs);
            if (!source->linear_empty)
                std::copy_n(source->species_coeffs, count, target.species_coeffs);
            Message("Reusing radial parameters; target head parameters use their own layout");
        }
    }
    auto &mtp = dynamic_cast<PotMTP &>(*model);
    validate_graph(mtp);
    if (!names.empty())
    {
        std::map<int, std::string> map;
        for (int i = 0; i < count; ++i)
            map[i] = names[i];
        mtp.set_element_map(map);
    }
    if ((fit || normalize) && dataset.empty())
        GPERROR("Offset fitting and descriptor normalization require a dataset");
    if (!dataset.empty())
    {
        GPIO io;
        if (!mtp.potential_tag.elements().empty())
            io.setmap(mtp.potential_tag.elements());
        else if (!index.empty() && fs::exists(index))
            io.readmap(index);
        std::vector<Atoms> atoms;
        if (!io.read(atoms, dataset, format) || atoms.empty())
            GPERROR("Initialization requires a nonempty readable dataset");
        Eigen::MatrixXd design = Eigen::MatrixXd::Zero(atoms.size(), count);
        Eigen::VectorXd rhs = Eigen::VectorXd::Zero(atoms.size());
        int labels = 0;
        const int scalars = mtp.alpha_scalar_moments;
        if (normalize && !scalars)
            GPERROR("Descriptor normalization requires scalar basis functions");
        Eigen::VectorXd means = Eigen::VectorXd::Zero(scalars), squares = means;
        size_t sites = 0;
        std::vector<double> values(scalars + 1);
        for (auto &at : atoms)
        {
            if (!at.size())
                GPERROR("Empty initialization structure");
            for (int id : at.chemmap_)
                if (id < 0 || id >= count)
                    GPERROR("Dataset species is outside the model element map");
            Neighborhoods neighborhoods(at, model->cutoff);
            for (int i = 0; i < at.size(); ++i)
            {
                mtp.compute_descriptors(neighborhoods[i], values);
                ++sites;
                for (int k = 0; k < scalars; ++k)
                {
                    double delta = values[k + 1] - means[k];
                    means[k] += delta / sites;
                    squares[k] += delta * (values[k + 1] - means[k]);
                }
            }
        }
        Eigen::VectorXd variance = squares / double(sites);
        if (!means.allFinite() || !variance.allFinite())
            GPERROR("Nonfinite descriptor statistics");
        Message("Basis means: " + std::to_string(means.norm()) +
                ", variance norm: " + std::to_string(variance.norm()));
        if (normalize)
        {
            const int channels = mtp.moment_coeff_size / scalars;
            for (int c = 0; c < channels; ++c)
            {
                double mean = 0.;
                for (int k = 0; k < scalars; ++k)
                {
                    double &coefficient = mtp.moment_coeffs[c * scalars + k];
                    coefficient = variance[k] > 1e-12 ? coefficient / std::sqrt(variance[k]) : 0.;
                    mean += coefficient * means[k];
                }
                const int offset = type == "mtp" ? 0 : (c + 1) * count;
                for (int i = 0; i < count; ++i)
                    mtp.species_coeffs[offset + i] -= mean;
            }
            Message(
                "Descriptor normalization explicitly changed moment coefficients and channel offsets");
        }
        if (fit)
            std::fill_n(mtp.species_coeffs, count, 0.);
        for (auto &at : atoms)
        {
            if (!at.size())
                GPERROR("Empty initialization structure");
            for (int species : at.chemmap_)
                if (species < 0 || species >= count)
                    GPERROR("Dataset species is outside the model element map");
            if (!at.have_energy)
                continue;
            if (!std::isfinite(at.energy_))
                GPERROR("Nonfinite initialization energy");
            const double energy = at.energy_, scale = 1 / std::sqrt(double(at.size()));
            for (int species : at.chemmap_)
                design(labels, species) += scale;
            if (fit)
            {
                model->getEFS(at);
                rhs[labels] = (energy - at.energy_) * scale;
            }
            ++labels;
        }
        Message("Initialization dataset: " + std::to_string(atoms.size()) + " structures, " +
                std::to_string(labels) + " energies");
        if (fit)
        {
            if (!labels)
                GPERROR("Offset fitting requires energy labels");
            auto matrix = design.topRows(labels).eval();
            Eigen::FullPivLU<Eigen::MatrixXd> solve(matrix);
            if (solve.rank() < count)
                GPERROR("Species offset fitting is rank deficient");
            Eigen::VectorXd values = solve.solve(rhs.head(labels));
            if (!values.allFinite())
                GPERROR("Nonfinite fitted offsets");
            std::copy_n(values.data(), count, mtp.species_coeffs);
        }
    }
    save(*model, output);
}
}
