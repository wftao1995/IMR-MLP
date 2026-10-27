// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once


#include "potential.h"
#include "utils/gputils.h"
#include "io/gpio.h"
#include <future>
#include <memory>
#include <utility>

/**
 * @brief Shell calculator
 * 
 * Like vaspfile calculator, three scripts provided.
 * pre-script: invoked after structure file written
 * run-script: invoked after pre-script exectuted
 * post-script: invoked after run-script exectuted and result read.
 * 
 * pre-script should generate files for specific program which calculates EFS
 * run-script call programs to calculate, and generate results that could be read by GPIO
 * post-script clean jobs
 * 
 */

namespace Potentials {
    class ShellJob {
        friend class PotShellCalc;
        std::shared_future<Atoms> completion_;
        std::string directory_;
        ShellJob(std::shared_future<Atoms> completion, std::string directory)
            : completion_(std::move(completion)), directory_(std::move(directory)) {}
    public:
        ShellJob(const ShellJob &) = delete;
        ShellJob &operator=(const ShellJob &) = delete;
        ~ShellJob() noexcept;
        bool ready() const;
        Atoms result() const;
        const std::string &run_directory() const { return directory_; }
    };
    
    class PotShellCalc : public AnyPotential {
    friend struct GPMCodecAccess;
    private:
        std::string runscript;
        std::string prescript;
        std::string postscript;
        std::string indexfile = "index.map";
        std::map<std::string,std::string> gpm_scripts_, gpm_inputs_;
        std::vector<std::string> gpm_dependencies_;

        std::string read_outfile = "output.cfg";
        std::string read_outformat = "mlpcfg";
        std::string write_posfile = "POSCAR";
        std::string write_posformat = "POSCAR";

        bool read_energy_sep = false;
        std::string energy_file = "eng.dat";

        GPIO iobj;

    public:
        PotShellCalc(){};
        PotShellCalc(GPSettings settings);
        void set_element_map(const std::map<int, std::string> &mapping) override {
            if (mapping.empty()) return;
            iobj.setmap(mapping); gpm_elements=mapping; species_count=mapping.size();
        }
        ~PotShellCalc(){};

        void InitSettings() {
            Connect(runscript, "runscript");
            Connect(prescript, "prescript");
            Connect(postscript, "postscript");
            Connect(indexfile, "index");

            Connect(read_energy_sep, "read_energy", "read energy data from data file instead of trajectory");
            Connect(energy_file, "engfile");

            Connect(write_posfile, "write_posfile");
            Connect(write_posformat, "write_posformat");
            Connect(read_outfile, "read_outfile");
            Connect(read_outformat, "read_outformat");
        }

        /**
         * @brief After write out poscar, run this script.
         * 
         * It is helpful to generate POTCAR and KPOINTS, as well as INCAR
         * using utils like vaspkit
         * 
         * @return int status
         */
        int exec_pre(std::string script);
        /**
         * @brief Run VASP
         * 
         * Commands read from environment variables
         * 
         * @return int 
         */
        int exec_cmd(std::string script);
        /**
         * @brief Post process
         * 
         * It is helpful to clean files like WAVECAR, CHGCAR, etc.
         * 
         * @return int 
         */
        int exec_post(std::string script);
        

        int read_output(Atoms &atom, std::string file);

        void write_input(Atoms &atom, std::string file);

        // Explicit asynchronous calculation. The job owns its settings, input,
        // I/O mapping and unique directory independently of this calculator.
        std::shared_ptr<ShellJob> submit(const Atoms &atom) const;

        virtual void getEFS(Atoms &atom);

        virtual void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
            getEFS(atom);
        }

    };

}
