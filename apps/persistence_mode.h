#pragma once

#include "potentials/model_bundle.h"
#include "utils/gputils.h"

// Command-local policy: explicit native paths keep their meaning, while the
// switch restores the defaults used before the GPM workflow was introduced.
struct PersistenceMode
{
    bool legacy = false;

    void RegisterConsole(CMDOption::CommandLineOptions &options)
    {
        options.addop(legacy, "legacy", "",
                      "Use native training outputs and legacy .sel persistence", false, 0);
    }

    void training_defaults(GPSettings &settings, std::string &output,
                           const std::string &native_output, const std::string &native_checkpoint,
                           const std::string &gpm_checkpoint, bool save_maxvol) const
    {
        if (legacy)
        {
            if (!CMDOption::global_cmdopts.was_provided("output") && !settings.count("output"))
                output = native_output;
            if (save_maxvol)
                GPERROR(
                    "--legacy conflicts with --save-maxvol-state (GPM embedding); use maxvol_select --legacy to save a .sel sidecar");
            if (GPM::extension(output) || GPM::extension(settings["fit:save_to"]))
                GPERROR("--legacy requires native output and fit:save_to paths, not .gpm");
        }
        if (!settings.count("fit:save_to"))
            settings.Modify("fit:save_to", legacy ? native_checkpoint : gpm_checkpoint);
    }

    void validate_state(const std::string &path) const
    {
        if (legacy && !path.empty() && (GPM::extension(path) || GPM::is_file(path)))
            GPERROR("--legacy requires a legacy .sel state, not GPM: " + path);
    }
};
