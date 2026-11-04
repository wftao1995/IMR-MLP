// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/mtp_adjustment.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_mtp.h"
#include "potentials/model_bundle.h"
using namespace std;

int adjustmtp(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    string infile = "potential.mtp";
    string outfile = "output-adj.mtp";
    string settings_file, type = "auto", basis_order;
    Potentials::MTP::Adjustment operation;
    cmdopts.addop(settings_file, "settings", "", "Model factory configuration")
        .addop(type, "type", "", "Input model family (GPM infers it)")
        .addop(basis_order, "basis-order", "", "keep, level, or level-group")
        .addop(operation.appended, "append-elements", "", "New element names", false, -1)
        .addop(operation.sources, "clone-from", "", "Original species IDs for new species", false,
               -1)
        .addop(operation.initialization, "new-species-init", "", "clone or random")
        .addop(infile, "input", "in", "input file", true, 1)
        .addop(outfile, "output", "out", "output file", true, 1)
        .addop(operation.elements, "elements", "", "Element symbols in output model species order",
               false, -1)
        .addop(operation.order, "neworder", "", "new species order, i.e 2 1 0 means 0 1 2 -> 2 1 0",
               true, -1)
        .phrase_options(argc, argv);
    if (cmdopts.displayhelp)
    {
        cmdopts.showhelp(cout);
        return 0;
    }
    // Apply the settings file before selecting the execution path: input,
    // output and neworder may all come from the deck, not only from the CLI.
    GPSettings all;
    if (!settings_file.empty())
    {
        try
        {
            all.Load(settings_file);
        }
        catch (const GPException &e)
        {
            std::cerr << e.What() << '\n';
            return 1;
        }
        apply_settings_to_cmdopts(cmdopts, all);
    }
    // Keep the historical native defaults unless an extended option selects the tree path.
    const bool extended = GPM::is_file(infile) || GPM::extension(infile) ||
                          GPM::extension(outfile) || type != "auto" || !basis_order.empty() ||
                          !operation.appended.empty() || !settings_file.empty();
    std::unique_ptr<Potentials::AnyPotential> adjusted;
    if (extended)
    {
        auto config = all.ExtractSubSettings("pot");
        if (type != "auto" || config["pottype"].empty())
            config.Modify("pottype", type);
        Potentials::potential_wrapper wrapper(Potentials::native_mtp_settings(infile, config));
        operation.basis = basis_order.empty() ? "keep" : basis_order;
        adjusted = Potentials::MTP::adjust_tree(*wrapper.pot, operation);
    }
    else
    {
        auto native = std::make_unique<Potentials::PotMTP>(infile);
        Potentials::MTP::adjust_legacy(*native, operation);
        adjusted = std::move(native);
    }
    const bool explicit_output = cmdopts.was_provided("output") || all.count("output");
    const auto output = extended && !explicit_output ? "_Adjusted_Potential_.gpm" : outfile;
    if (!explicit_output && std::filesystem::absolute(output).lexically_normal() ==
                                std::filesystem::absolute(infile).lexically_normal())
        GPERROR("Explicit output is required to overwrite the input");
    if (extended)
        Potentials::save_inference_model(*adjusted, output);
    else
        Potentials::MTP::save_legacy_adjusted(static_cast<Potentials::PotMTP &>(*adjusted), output);
    return 0;
}
