// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtp_initialization.h"
#include "mtp_adjustment.h"
#include "mtp_model_internal.h"
#include "utils/temporary_directory.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/model_bundle.h"
#include "tools/gpm_mtp.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>

#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_hybrid.h"
#include "potentials/potential_hybrid_scale.h"
#include "potentials/potential_hybmtp.h"
#include "potentials/potential_descriptor_linear.h"
#include "descriptors/descriptor_MTM.h"
#include "tools/chem.h"

namespace Potentials::MTP
{
namespace fs = std::filesystem;
using Stage = Utils::TemporaryDirectory;
using namespace detail;
using GPM::require;

void save_legacy_initialized(AnyPotential &model, const std::string &output,
                             const GPSettings &model_settings, const std::vector<Atoms> &atoms)
{
    const auto target = std::filesystem::absolute(output).lexically_normal();
    Utils::TemporaryDirectory stage(target.parent_path());
    const auto temporary = (stage.path / target.filename()).string();
    model.Save(temporary);
    auto reload = model_settings;
    reload.Modify("potfile", temporary);
    reload.Modify("pottype", Potentials::ModelBundle::type(model));
    Potentials::potential_wrapper check(reload);
    if (check.pot->get_NL_coeffs_count() != model.get_NL_coeffs_count())
        GPERROR("Initialized model parameter count changed on native reload");
    if (!atoms.empty())
    {
        auto before = atoms.front(), after = before;
        model.getEFS(before);
        check.pot->getEFS(after);
        if (std::abs(before.energy_ - after.energy_) > 1e-10 * (1 + std::abs(before.energy_)) ||
            (before.stress_ - after.stress_).norm() > 1e-9 * (1 + before.stress_.norm()))
            GPERROR("Initialized native model changed predictions on reload");
        for (size_t i = 0; i < before.size(); ++i)
            if ((before.forces_[i] - after.forces_[i]).norm() >
                1e-9 * (1 + before.forces_[i].norm()))
                GPERROR("Initialized native model changed forces on reload");
    }
    std::filesystem::rename(temporary, target);
}

static std::string topology(Potentials::PotMTP &source, int count)
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

namespace
{
std::unique_ptr<AnyPotential> initialize_leaf(GPSettings config,
                                              const InitializationOptions &options,
                                              std::unique_ptr<AnyPotential> model = {})
{
    auto input = options.input, type = options.type;
    const auto &elements = options.elements;
    const auto &index = options.index;
    const bool reset = options.reset, randomize = options.randomize;
    const bool explicit_index = options.explicit_index;
    const double lower = options.lower, upper = options.upper, scale = options.scale;

    using namespace Potentials;
    if (!std::isfinite(scale) || scale < 0)
        GPERROR("Invalid random moment scale");
    if (input.empty())
        input = config["potfile"];
    if (input.empty() && !model)
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
    std::shared_ptr<ModelBundle> source_bundle;
    std::unique_ptr<PotMTP> ordinary;
    PotMTP *source = nullptr;
    if (model)
    {
        source = dynamic_cast<PotMTP *>(model.get());
        require(source, "Initialization requires an MTP template");
    }
    else if (GPM::is_file(input) || GPM::extension(input))
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
            native.Modify("pottype", "mtp_m");
            native.Modify("typemap", "");
            potential_wrapper loaded(native_mtp_settings(input, native));
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
        Stage stage;
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
            target.random_init_local();
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
    return model;
}

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
    const InitializationOptions &options;
    std::vector<std::string> names;

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
                                       const std::string &type,
                                       std::unique_ptr<AnyPotential> source = {})
    {
        auto leaf_options = options;
        leaf_options.input = input;
        leaf_options.type = type;
        leaf_options.elements = names;
        // Preserve leaf initialization's last-value-wins handling of repeated keys.
        GPSettings leaf_config;
        for (const auto &[key, value] : config.settings)
            leaf_config.Modify(key, value);
        return initialize_leaf(std::move(leaf_config), leaf_options, std::move(source));
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
            auto &mtp = dynamic_cast<PotMTP &>(*source);
            short_source = copy_mtp(mtp, true);
            footer = model_text(*source);
            const auto metadata = mtp.model_metadata();
            const auto &lr = metadata.at("long_range").get<MTPMetadata::Object>();
            old_channels = int(lr.at("channels").get<double>());
            if (config["les_kernel"].empty())
                config.Modify("les_kernel", lr.at("kernel").get<std::string>());
        }
        auto short_config = config;
        short_config.Modify("lr_les", "false");
        short_config.Modify("req_les", "false");
        short_config.Modify("req_sog", "false");
        auto short_model =
            leaf(short_config, input, type == "mtp_les" ? "mtp" : "mtp_m", std::move(short_source));
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
        std::uniform_real_distribution<double> uniform(options.lower, options.upper);
        for (auto *array : {&species, &moments})
            for (auto &value : *array)
                value = options.randomize ? uniform(RNG.mtd) : 1e-3;
        if (source_long && !options.reset && old_channels == channels)
        {
            std::map<std::string, std::string> fields;
            std::istringstream lines(footer);
            std::string line;
            while (std::getline(lines, line))
            {
                auto eq = line.find('=');
                if (eq != std::string::npos)
                {
                    auto key = GPM::trim(line.substr(0, eq));
                    if (key == "les_species_coeffs" || key == "les_moment_coeffs")
                        fields[key] = GPM::trim(line.substr(eq + 1));
                }
            }
            for (const auto &[key, destination] :
                 std::vector<std::pair<std::string, std::vector<double> *>>{
                     {"les_species_coeffs", &species}, {"les_moment_coeffs", &moments}})
            {
                require(fields.count(key), "Long-range coefficient section is missing: " + key);
                auto old = numbers(fields.at(key));
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
        Stage stage;
        const auto path = (stage.path / "long-range.mtp").string();
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
                                        std::string requested = "",
                                        std::unique_ptr<AnyPotential> source = {})
    {
        if (input.empty())
            input = config["potfile"];
        if (!source && !input.empty() && (GPM::is_file(input) || GPM::extension(input)))
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
                GPSettings factory;
                factory.Modify("pottype", "auto");
                factory.Modify("typemap", "");
                factory = native_mtp_settings(input, factory);
                // Tagged source heads must be loaded as stored, before target conversion.
                if (factory["pottype"] == "mtp_m" || factory["pottype"] == "mtp_les")
                    source = load(input, factory);
            }
            else
            {
                auto factory = config;
                factory.Modify("pottype", "auto");
                factory = native_mtp_settings(input, factory);
                if (magic == "pottype" || factory["pottype"] != "auto")
                    source = load(input, factory);
            }
            if (source && requested.empty() &&
                (config["pottype"].empty() || config["pottype"] == "auto"))
                type = ModelBundle::type(*source);
        }
        if (source)
            require(type == ModelBundle::type(*source) || type == "mtp" || type == "mtp_m" ||
                        type == "mtp_les" || type == "mtpnn1" || type == "descriptor_linear",
                    "Changing composite families requires explicit child templates");
        if (source && names.empty())
            names = model_elements(*source);
        if (source && !options.reset && !names.empty())
        {
            const auto existing = model_elements(*source);
            require(existing.empty() || existing == names,
                    "Use adjustmtp to change existing element identities");
        }
        if (source && !options.reset && (type == "mtp_les" || type == "mtp_m"))
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
            require(source || !input.empty(), "MTP initialization requires a template");
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
            return leaf(config, input, type, std::move(source));
        }
        if (type == "mtp_les")
        {
            require(source || !input.empty(), "LES initialization requires an MTP template");
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
                if (!options.reset)
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
            require(matching || !descriptor_input.empty(),
                    "Descriptor initialization requires an MTP template");
            auto desc_config = config;
            desc_config.Modify("pottype", "mtp");
            std::unique_ptr<AnyPotential> existing_descriptor;
            if (matching)
            {
                PotMTP *mtp = nullptr;
                if (auto *nn = dynamic_cast<PotMTPNN1 *>(source.get()))
                    mtp = nn->p_mtp;
                else if (auto *linear = dynamic_cast<PotentialDescriptorLinear *>(source.get()))
                    if (auto *mtm = dynamic_cast<const DescriptorMTM *>(&linear->descriptor()))
                        mtp = &mtm->pot();
                require(mtp, "Only MTM-backed linear descriptor models support this workflow");
                existing_descriptor = std::make_unique<PotMTP>(*mtp);
            }
            auto descriptor =
                leaf(desc_config, descriptor_input, "mtp", std::move(existing_descriptor));
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
                std::uniform_real_distribution<double> uniform(options.lower, options.upper);
                for (auto &value : weights)
                    value = options.randomize ? options.scale * uniform(RNG.mtd) : 0.;
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
            auto model = std::make_unique<PotMTPNN1>(
                config, std::unique_ptr<PotMTP>(static_cast<PotMTP *>(descriptor.release())));
            model->random_init_mtp = false;
            if (options.randomize)
                model->random_init();
            else
            {
                std::vector<double> weights(model->nn_coeff_size, 1e-3);
                if (layers == 1)
                    model->p_nn->set_coeffs(weights.data());
                else
                    model->p_nn2->set_coeffs(weights.data());
            }
            if (matching && !options.reset)
                model->species_coeffs = dynamic_cast<PotMTPNN1 &>(*source).species_coeffs;
            model->set_inited();
            return model;
        }
        if (type == "hybrid" || type == "hybrid_scale" || type == "hmtp")
        {
            if (source && !options.reset)
            {
                bool structural = false;
                for (const auto &[key, value] : config.settings)
                    if (key != "potfile" && key != "pottype" && key != "typemap" && !value.empty())
                        structural = true;
                if (!structural)
                    return source;
            }
            std::vector<std::unique_ptr<AnyPotential>> existing_children;
            if (source)
            {
                auto previous = composite_settings(*source);
                if (auto *hybrid = dynamic_cast<PotentialHybrid *>(source.get()))
                {
                    for (auto &child : hybrid->ppots)
                        if (auto *mtp = dynamic_cast<PotMTP *>(child.get()))
                            existing_children.push_back(std::make_unique<PotMTP>(*mtp));
                        else
                            existing_children.push_back(std::move(child));
                }
                else if (auto *scaled = dynamic_cast<PotentialHybridScale *>(source.get()))
                {
                    existing_children = std::move(*scaled).take_children();
                }
                else if (auto *hmtp = dynamic_cast<PotMTPHyb3 *>(source.get()))
                    for (auto *child : hmtp->children())
                        existing_children.push_back(
                            std::make_unique<PotMTP>(*const_cast<PotMTP *>(child)));
                for (const auto &[key, value] : previous.settings)
                    if (config[key].empty())
                        config.Modify(key, value);
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
                std::unique_ptr<AnyPotential> existing;
                if (child["potfile"].empty() && i <= int(existing_children.size()))
                    existing = std::move(existing_children[i - 1]);
                require(!kind.empty(), "Missing composite child type at pot" + std::to_string(i));
                if (kind == "zbl" || kind == "eampoly")
                {
                    if (existing)
                        children.push_back(std::move(existing));
                    else if (!child["potfile"].empty())
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
                    children.push_back(build(child, child["potfile"], kind, std::move(existing)));
            }
            config.Modify("potfile", "");
            return make_composite(type, config, std::move(children));
        }
        require(false, "Initialization is not defined for model family: " + type);
        return {};
    }
};

struct OffsetTarget
{
    AnyPotential *model;
    std::string id = "root";
    bool independent_offsets = true;
};

OffsetTarget resolve(AnyPotential &model, std::string path)
{
    OffsetTarget target{&model};
    if (path.empty() || path == "root")
        return target;
    if (path.starts_with("root/"))
        path.erase(0, 5);
    std::replace(path.begin(), path.end(), '.', '/');
    std::istringstream in(path);
    std::string role;
    while (std::getline(in, role, '/'))
    {
        require(!role.empty(), "Empty offset target path component");
        require(role != "charge" && role != "charge_model" && role != "descriptor",
                "Offset targets must be energy branches, not charges or descriptors");
        const auto children = model_children(*target.model);
        require(children.count(role), "Unknown offset target: " + path);
        if (auto *hybrid = dynamic_cast<PotentialHybrid *>(target.model))
            for (size_t i = 1; i < hybrid->train_ids.size(); ++i)
                if (role == "pot" + std::to_string(hybrid->train_ids[i] + 1))
                    target.independent_offsets = false;
        target.model = children.at(role);
        target.id += "/" + role;
    }
    return target;
}

std::vector<Atoms> read_initialization_frames(AnyPotential &model,
                                              const InitializationOptions &options)
{
    auto names = model_elements(model);
    GPIO io;
    if (!names.empty())
        io.setmap(names);
    if (options.explicit_index)
    {
        require(fs::exists(options.index), "Explicit element map does not exist: " + options.index);
        GPIO check;
        check.readmap(options.index);
        if (!names.empty())
            require(check.getmap() == io.getmap(), "Explicit map conflicts with model elements");
        io.setmap(check.getmap());
    }
    else if (names.empty() && !options.index.empty() && fs::exists(options.index))
        io.readmap(options.index);
    std::vector<Atoms> frames;
    require(io.read(frames, options.dataset, options.format) && !frames.empty(),
            "Initialization requires a nonempty dataset");
    for (const auto &frame : frames)
    {
        require(frame.size() > 0, "Empty initialization structure");
        for (int id : frame.chemmap_)
            require(id >= 0 && id < model.species_count,
                    "Dataset element is outside model mapping");
        if (frame.have_energy)
            require(std::isfinite(frame.energy_), "Nonfinite initialization energy");
    }
    return frames;
}

void fit_species_offsets(AnyPotential &model, const std::vector<Atoms> &frames,
                         const OffsetTarget &selected, double *offsets)
{
    const int species = selected.model->species_count;
    std::vector<Atoms> labeled;
    for (const auto &frame : frames)
        if (frame.have_energy)
            labeled.push_back(frame);
    require(!labeled.empty(), "Offset fitting requires energy labels");
    require(selected.independent_offsets, "Composite species offset response is rank deficient");
    std::fill_n(offsets, species, 0.);
    Eigen::VectorXd base(labeled.size()), rhs(labeled.size());
    Eigen::MatrixXd design(labeled.size(), species);
    for (size_t i = 0; i < labeled.size(); ++i)
    {
        auto frame = labeled[i];
        model.getEFS(frame);
        base[i] = frame.energy_;
        rhs[i] = (labeled[i].energy_ - base[i]) / std::sqrt(double(frame.size()));
    }
    for (int s = 0; s < species; ++s)
    {
        for (size_t i = 0; i < labeled.size(); ++i)
        {
            auto one = labeled[i], two = one;
            offsets[s] = 1.;
            model.getEFS(one);
            offsets[s] = 2.;
            model.getEFS(two);
            require(std::abs(two.energy_ - base[i] - 2 * (one.energy_ - base[i])) <=
                        1e-9 * (1 + std::abs(two.energy_)),
                    "Offset target is not affine in complete model energy");
            design(i, s) = (one.energy_ - base[i]) / std::sqrt(double(one.size()));
        }
        offsets[s] = 0.;
    }
    Eigen::FullPivLU<Eigen::MatrixXd> solve(design);
    require(solve.rank() == species, "Composite species offset response is rank deficient");
    Eigen::VectorXd fitted = solve.solve(rhs);
    require(fitted.allFinite(), "Nonfinite composite energy offsets");
    std::copy_n(fitted.data(), species, offsets);
}

void dataset_operations(AnyPotential &model, const InitializationOptions &options)
{
    require(!(options.fit || options.normalize) || !options.dataset.empty(),
            "Offset fitting and normalization require a dataset");
    if (options.dataset.empty())
        return;
    const auto frames = read_initialization_frames(model, options);

    struct Statistics
    {
        Eigen::VectorXd mean, m2;
        size_t sites;
    };

    std::map<AnyPotential *, Statistics> statistics;
    // One descriptor pass per MTP branch; normalization reuses these statistics.
    auto collect = [&](auto &&self, AnyPotential &branch, const std::string &id) -> void
    {
        for (const auto &[role, child] : model_children(branch))
            self(self, *child, id + "/" + role);
        auto *leaf = dynamic_cast<PotMTP *>(&branch);
        if (!leaf)
            return;
        auto &mtp = *leaf;
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
        statistics.emplace(&branch, Statistics{std::move(mean), std::move(m2), sites});
    };
    collect(collect, model, "root");
    if (!options.fit && !options.normalize)
        return;
    const auto root_type = ModelBundle::type(model);
    const bool composite =
        root_type == "hybrid" || root_type == "hybrid_scale" || root_type == "hmtp";
    require(!composite || !options.offset_target.empty(),
            "Composite offset fitting/normalization requires --offset-target");
    const auto selected = resolve(model, options.offset_target);
    const auto &id = selected.id;
    const auto kind = ModelBundle::type(*selected.model);
    auto *mtp = dynamic_cast<PotMTP *>(selected.model);
    auto *nn = dynamic_cast<PotMTPNN1 *>(selected.model);
    require(mtp || nn, "Target has no independent elemental energy offsets: " + id);
    const int species = selected.model->species_count;
    require(species > 0, "Offset target has no declared species");
    // Only the first trainable Hybrid child owns the shared elemental offsets;
    // subsequent trainable children retain zero offsets.
    double *offsets = mtp ? mtp->species_coeffs : nn->species_coeffs.data();
    if (options.normalize)
    {
        require(
            kind != "mtpnn1",
            "Normalize the MTP descriptor through its explicit model workflow; NN inputs must retain their trained normalization");
        const auto &[mean, m2, sites] = statistics.at(mtp);
        const int b = mtp->alpha_scalar_moments;
        require(b > 0, "Normalization requires scalar moments");
        auto *moments = mtp->moment_coeffs;
        for (int c = 0; c < mtp->moment_coeff_size / b; ++c)
        {
            double bias = 0.;
            for (int k = 0; k < b; ++k)
            {
                moments[c * b + k] =
                    m2[k] / sites > 1e-12 ? moments[c * b + k] / std::sqrt(m2[k] / sites) : 0.;
                bias += moments[c * b + k] * mean[k];
            }
            const int first = kind == "mtp_m" ? (c + 1) * species : 0;
            if (selected.independent_offsets)
                for (int i = 0; i < species; ++i)
                    offsets[first + i] -= bias;
        }
    }
    if (options.fit)
        fit_species_offsets(model, frames, selected, offsets);
    Message("Applied initialization changes to energy branch " + id);
}
}

std::unique_ptr<AnyPotential> initialize_model(GPSettings settings,
                                               const InitializationOptions &options)

{
    require(std::isfinite(options.lower) && std::isfinite(options.upper) &&
                options.lower <= options.upper && std::isfinite(options.scale) &&
                options.scale >= 0,
            "Invalid initialization random bounds/scale");
    require(options.offset_target.empty() || options.fit || options.normalize,
            "--offset-target requires offset fitting or normalization");
    auto config = settings.ExtractSubSettings("pot");
    Builder builder{options, options.elements};
    if (builder.names.empty() && options.explicit_index)
    {
        require(fs::exists(options.index), "Explicit element map does not exist: " + options.index);
        GPIO io;
        io.readmap(options.index);
        for (const auto &[id, name] : io.getmap())
        {
            require(id == int(builder.names.size()), "Nonconsecutive element map");
            builder.names.push_back(name);
        }
    }
    auto model = builder.build(config, options.input, options.type);
    if (!builder.names.empty())
        set_elements(*model, builder.names);
    dataset_operations(*model, options);
    return model;
}

std::unique_ptr<AnyPotential> initialize_legacy(GPSettings potsettings, std::vector<Atoms> &atoms,
                                                const InitializationOptions &options)
{
    using namespace std;
    require(!atoms.empty(), "MTP-M initialization requires nonempty training data");
    const bool randomize = options.randomize;
    const double lb = options.lower, ub = options.upper, rand_scale = options.scale;
    int n_labeled = 0;
    for (const auto &at : atoms)
    {
        if (at.size() == 0)
            GPERROR("MTP-M initialization requires nonempty training structures");
        if (at.have_energy)
        {
            if (!std::isfinite(at.energy_))
                GPERROR("MTP-M reference energies must be finite");
            ++n_labeled;
        }
    }
    if (n_labeled < atoms.size())
        Warning("Warning: MTP-M skipped " + std::to_string(atoms.size() - n_labeled) +
                " structures without energy labels when fitting species offsets");
    if (n_labeled == 0)
        Warning("Warning: MTP-M has no energy labels; species offsets are initialized to zero");

    int n_loc_total = 0;
    int n_loc_local = 0;

    for (int i = 0; i < atoms.size(); ++i)
    {
        n_loc_local += atoms[i].size();
    }
    n_loc_total = n_loc_local;

    int n_cfgs_this = atoms.size();
    std::unique_ptr<AnyPotential> pot;
    // Check if LES type is requested
    std::string ptype = "";
    if (potsettings.count("pottype") > 0)
        ptype = potsettings["pottype"];
    if (ptype == "mtp_les")
    {
        std::cerr << "Error: initmtpm does not support mtp_les directly.\n"
                  << "Use 'imrmlp trainnlpot --settings fit.ini' with fit:random_init true\n"
                  << "to initialize LES potentials. Add les_init_range=0 for charge balancing.\n";
        GPERROR("Cannot initialize legacy MTP-M model");
    }
    else
    {
        pot.reset(create_mtp_m_potential(potsettings));
    }
    if (pot == nullptr)
    {
        std::cerr << "Error creating potential.\n";
        GPERROR("Cannot initialize legacy MTP-M model");
    }
    std::map<int, std::string> element_map;
    for (size_t i = 0; i < options.elements.size(); ++i)
        element_map[int(i)] = options.elements[i];
    pot->set_element_map(element_map);
    PotMTP *pmtp = dynamic_cast<PotMTP *>(pot.get());
    for (const auto &at : atoms)
        for (int id : at.chemmap_)
            if (id < 0 || id >= pmtp->species_count)
                GPERROR("Initialization dataset species is outside the model layout");
    {
        Potentials::MTP::validate_graph(*pmtp);
    }
    int nprops;
    pmtp->get_n_props(nprops);
    int n_radial, n_species_coeffs, n_moment, n_model, n_species;
    pmtp->get_n_coeffs(n_radial, n_species_coeffs, n_moment, n_model);
    n_species = n_species_coeffs / (nprops + 1);
    double *radial_ptr, *species_ptr, *moment_ptr, *model_ptr;
    pmtp->get_pointers(radial_ptr, species_ptr, moment_ptr, model_ptr);

    pmtp->random_lower = lb;
    pmtp->random_upper = ub;

    if (randomize)
    {
        pmtp->random_init_local();
        // random init radial coeffs
        uniform_real_distribution<double> uniform(-1, 1);
        for (int i = 0; i < n_radial; ++i)
        {
            radial_ptr[i] = uniform(RNG.mtd);
        }
        pmtp->Orthogonalize();
    }

    // init species coeffs
    Eigen::MatrixXd linmat;
    linmat.resize(n_species, n_species);
    linmat.setZero();
    Eigen::VectorXd sps, linvec;
    sps.resize(n_species);
    linvec.resize(n_species);
    linvec.setZero();
    for (int i = 0; i < n_cfgs_this; ++i)
    {
        if (!atoms[i].have_energy)
            continue;
        int n = atoms[i].size();
        double wgt = 1.0 / n / n_labeled;
        sps.setZero();
        for (int j = 0; j < atoms[i].size(); ++j)
        {
            sps(atoms[i].chemmap_[j]) += 1.;
        }
        for (int m = 0; m < n_species; ++m)
        {
            for (int n = m; n < n_species; ++n)
            {
                linmat(m, n) += wgt * sps(m) * sps(n);
            }
            linvec(m) += wgt * sps(m) * atoms[i].energy_;
        }
    }

    {
        // symmetric
        for (int i = 0; i < n_species; ++i)
        {
            for (int j = 0; j < i; ++j)
            {
                linmat(i, j) = linmat(j, i);
            }
        }
        // solve
        Eigen::VectorXd sol = Eigen::VectorXd::Zero(n_species);
        if (n_labeled > 0)
        {
            Eigen::FullPivLU<Eigen::MatrixXd> lu(linmat);
            sol = lu.solve(linvec);
            if (!sol.allFinite())
                GPERROR("MTP-M species offset fitting produced non-finite coefficients");
        }
        // assign
        for (int i = 0; i < n_species; ++i)
        {
            species_ptr[i] = sol(i);
        }
    }

    // init bias and moment coeffs
    // compute mean of each basis
    int n_scalars = pmtp->get_alpha_scalar_moments();
    Eigen::VectorXd basis_mean, basis_var;
    basis_mean.resize(n_scalars);
    basis_var.resize(n_scalars);
    basis_mean.setZero();
    basis_var.setZero();
    std::vector<double> local_disp(n_scalars + 1, 0);
    for (int i = 0; i < n_cfgs_this; ++i)
    {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j)
        {
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int k = 0; k < n_scalars; ++k)
            {
                basis_mean(k) += local_disp[k + 1] / n_loc_total;
            }
        }
    }

    // compute variance
    for (int i = 0; i < n_cfgs_this; ++i)
    {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j)
        {
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int k = 0; k < n_scalars; ++k)
            {
                double diff = local_disp[k + 1] - basis_mean(k);
                basis_var(k) += diff * diff / n_loc_total;
            }
        }
    }

    {
        cout << "Basis means:\n" << basis_mean.transpose() << "\n";
        cout << "Basis variances:\n" << basis_var.transpose() << "\n";
    }
    // assign moment coeffs
    normal_distribution<double> normal(0, 1);
    if (randomize)
    {
        for (int n = 0; n < nprops; ++n)
        {
            for (int i = 0; i < n_scalars; ++i)
            {
                if (basis_var(i) > 1e-12)
                {
                    moment_ptr[n_scalars * n + i] =
                        normal(RNG.mtd) * rand_scale / sqrt(basis_var(i));
                }
                else
                {
                    moment_ptr[n_scalars * n + i] = 0;
                }
            }
        }
    }

    // adjust species_coeffs to make sure mean of qi = 0
    Eigen::VectorXd qs;
    Eigen::VectorXi tpc;
    tpc.resize(n_species);
    tpc.setZero();
    qs.resize(nprops);
    qs.setZero();
    for (int i = 0; i < atoms.size(); ++i)
    {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j)
        {
            tpc(atoms[i].chemmap_[j]) += 1;
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int n = 0; n < nprops; ++n)
            {
                for (int k = 0; k < n_scalars; ++k)
                {
                    qs(n) += local_disp[k + 1] * moment_ptr[n_scalars * n + k] / n_loc_total;
                }
            }
        }
    }

    {
        for (int n = 0; n < nprops; ++n)
        {
            for (int i = 0; i < n_species; ++i)
            {
                species_ptr[n_species + n_species * n + i] =
                    -qs(n); //* (double)tpc(i) / (double)tpc.sum();
            }
        }
    }

    // calculate var of Qs
    Eigen::VectorXd qs2;
    qs2.resize(nprops);
    qs.setZero();
    qs2.setZero();
    for (int i = 0; i < atoms.size(); ++i)
    {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j)
        {
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int n = 0; n < nprops; ++n)
            {
                double qi = species_ptr[atoms[i].chemmap_[j] + n_species * (n + 1)];
                for (int k = 0; k < n_scalars; ++k)
                {
                    qi += local_disp[k + 1] * moment_ptr[n_scalars * n + k];
                }
                qs[n] += qi / n_loc_total;
                qs2(n) += qi * qi / n_loc_total;
            }
        }
    }

    {
        for (int n = 0; n < nprops; ++n)
        {
            double var = qs2(n) - qs(n) * qs(n);
            cout << "Mean of Q" << n << " after initialization: " << qs(n) << "\n";
            cout << "Var of Q" << n << " after initialization: " << var << "\n";
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    //  LES: auto-balance species coeffs so each structure has net-zero charge
    // ═══════════════════════════════════════════════════════════════════

    pot->init_coeffs();
    return pot;
}
}
