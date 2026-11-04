// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/mtp_initialization.h"
#include "potentials/model_bundle.h"
#include "potentials/model_elements.h"
#include "io/gpio.h"
#include "species_map_utils.h"
#include <memory>

int initmtpm(int argc, char **argv)
{
    using namespace Potentials;
    using namespace std;
    GPSettings settings;
    auto &cmdopts = CMDOption::global_cmdopts;
    string potfile = "potential.mtp";
    string settingfile = "settings.ini";
    string dataset = "train.cfg";
    string format = "";
    string index = "index.map";
    string input, output, target_type, offset_target;
    vector<string> elements;
    bool reinitialize = false, fit_offsets = false, normalize = false;
    bool randomize = true;
    double rand_scale = 1.0;
    double lb = -0.1, ub = 0.1;
    GPIO iobj;
    cmdopts.addop(input, "input", "in", "Input basis template or model")
        .addop(output, "output", "out", "Output model (default _Initialized_Potential_.gpm)")
        .addop(target_type, "type", "", "Target model family")
        .addop(elements, "elements", "", "Ordered element symbols", false, -1)
        .addop(reinitialize, "reinitialize", "", "Reset existing parameters", false, 0)
        .addop(fit_offsets, "fit-species-offsets", "", "Fit elemental energy offsets", false, 0)
        .addop(normalize, "normalize-descriptors", "", "Normalize descriptor statistics", false, 0)
        .addop(offset_target, "offset-target", "", "Energy branch for offset fitting")
        .addop(potfile, "potfile", "pot", "Potential file", false, 1, 0)
        .addop(settingfile, "settings", "", "settings file name", false, 1, 0)
        .addop(dataset, "dataset", "d", "dataset file name", false, 1, 0)
        .addop(format, "format", "f", "dataset file format, mlpcfg, extxyz, ...", false, 1, 0)
        .addop(index, "index", "id", "index file name, for selecting configurations", false, 1, 0)
        .addop(rand_scale, "rand-scale", "", "scale for random coeffs, default 1.0", false, 1, 0)
        .addop(lb, "rand-lb", "", "lower bound for random coeffs, default -0.1", false, 1, 0)
        .addop(ub, "rand-ub", "", "upper bound for random coeffs, default 0.1", false, 1, 0)
        .addop(randomize, "randomize", "r", "randomize coeffs or not, default true", false, 1, 0);
    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp)
    {
        cmdopts.showhelp(cout);
        return 0;
    }
    const bool modern_cli = !input.empty() || !output.empty() || !target_type.empty() ||
                            !elements.empty() || reinitialize || fit_offsets || normalize ||
                            !offset_target.empty();
    if (!settingfile.empty() && (cmdopts.was_provided("settings") || !modern_cli))
    {
        try
        {
            settings.Load(settingfile);
        }
        catch (const GPException &e)
        {
            std::cerr << e.What() << '\n';
            return 1;
        }
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    // Settings may supply the modern-workflow options too; decide the path
    // after they have been applied, not only from the command line.
    const bool modern = modern_cli || !input.empty() || !output.empty() || !target_type.empty() ||
                        !elements.empty() || reinitialize || fit_offsets || normalize ||
                        !offset_target.empty();
    if (modern || (!settings["pot:pottype"].empty() && settings["pot:pottype"] != "mtp_m"))
    {
        if (!output.empty() && cmdopts.was_provided("potfile") && output != potfile)
            GPERROR("Conflicting output and potfile options");
        if (output.empty() && (cmdopts.was_provided("potfile") || settings.count("potfile")))
            output = potfile;
        if (output.empty())
        {
            output = "_Initialized_Potential_.gpm";
            const auto source = input.empty() ? settings["pot:potfile"] : input;
            if (!source.empty() && std::filesystem::absolute(source).lexically_normal() ==
                                       std::filesystem::absolute(output).lexically_normal())
                GPERROR("Explicit output is required to replace input");
        }
        const auto data =
            (cmdopts.was_provided("dataset") || settings.count("dataset")) ? dataset : string();
        auto initialized = Potentials::MTP::initialize_model(
            settings, {.input = input,
                       .type = target_type,
                       .elements = elements,
                       .reset = reinitialize,
                       .randomize = randomize,
                       .lower = lb,
                       .upper = ub,
                       .scale = rand_scale,
                       .index = index,
                       .explicit_index = cmdopts.was_provided("index") || settings.count("index"),
                       .dataset = data,
                       .format = format,
                       .fit = fit_offsets,
                       .normalize = normalize,
                       .offset_target = offset_target});
        Potentials::save_inference_model(*initialized, output);
        return 0;
    }
    vector<Atoms> atoms;
    // Retained for reference.
    // iobj.readmap(index);
    const bool index_explicit = cmdopts.was_provided("index") || settings.count("index");
    iobj.install_species_map(
        index, index_explicit,
        Potentials::probe_model_elements(make_potential_settings(settings, potfile)),
        /*require_if_absent=*/false);
    if (!iobj.read(atoms, dataset, format))
        GPERROR("Failed to read MTP-M training data: " + dataset);
    if (atoms.empty())
        GPERROR("MTP-M initialization requires nonempty training data");
    auto potsettings = settings.ExtractSubSettings("pot");
    Potentials::MTP::InitializationOptions legacy_options;
    legacy_options.randomize = randomize;
    legacy_options.lower = lb;
    legacy_options.upper = ub;
    legacy_options.scale = rand_scale;
    for (const auto &[id, name] : iobj.getmap())
        legacy_options.elements.push_back(name);
    auto pot = Potentials::MTP::initialize_legacy(potsettings, atoms, legacy_options);

    const auto source = potsettings["potfile"];
    if (!cmdopts.was_provided("potfile") && !settings.count("potfile") && !source.empty() &&
        std::filesystem::absolute(source).lexically_normal() ==
            std::filesystem::absolute(potfile).lexically_normal())
        GPERROR("Explicit output is required to overwrite the input model");
    if (GPM::extension(potfile))
        Potentials::save_inference_model(*pot, potfile);
    else
        Potentials::MTP::save_legacy_initialized(*pot, potfile, potsettings, atoms);

    return 0;
}
