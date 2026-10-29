// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "potentials/potential_mtp.h"
#include "descriptors/descriptors.h"
#include "descriptors/descriptor_BispectrumSO3.h"
#include "descriptors/descriptor_BispectrumSO4.h"
#include "descriptors/descriptor_Behler.h"
#include "descriptors/descriptor_BondOrder.h"
#include "descriptors/descriptor_MTM.h"
#include "descriptors/descriptor_wrapper.h"
#include "help_utils.h"
#include <exception>
#include <cmath>

using namespace std;
using namespace Eigen;

int calc_descriptors(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    GPSettings settings;
    std::vector<Atoms> atoms;
    string informat = "";
    string infile = "POSCAR";
    string indexfile = "index.map";
    string settingfile = "input.inp";
    string outformat = "extxyz";
    string localname = "desc";
    string globalname = "desc";
    string potfile = "potential.mtp";
    string outfile = "output-desc.xyz";
    string dtype = "mtp"; // mtp, mtm, biso3, biso4, behler, bondorder
    // biso3 parameters
    int    biso3_lmax = 4;
    double biso3_rmin = 0.0;
    double biso3_rmax = 5.0;
    int    biso3_rsize = 8;
    bool biso3_real_harmonics=false;
    double neig_cutoff = 5.0;
    bool lenergy = false;
    bool fuse_species = false;
    bool discard_local = false;
    cmdopts.addop(infile, "input", "in", "input structure file")
           .addop(outfile, "output", "out", "output structure file")
           .addop(informat, "input-format", "if", "input structure format")
           .addop(outformat, "output-format", "of", "output structure format")
           .addop(indexfile, "index", "id", "index file for each type")
           .addop(potfile, "potfile", "pot", "potential file, or use pot: tag in setting file for some specific potential types")
           .addop(dtype, "dtype", "", "descriptor type: mtp (mtp accepted), mtm, biso3, biso4, behler, bondorder")
           .addop(biso3_lmax, "biso3-lmax", "", "biso3: max angular L (default 4)")
           .addop(biso3_rmin, "biso3-rmin", "", "biso3: minimum radial distance (default 0.0)")
           .addop(biso3_rmax, "biso3-rmax", "", "biso3: cutoff radius (default 5.0)")
           .addop(biso3_rsize, "biso3-rsize", "", "biso3: radial basis size (default 8)")
           .addop(biso3_real_harmonics, "biso3-real-harmonics", "", "biso3: opt-in real harmonic backend")
           .addop(neig_cutoff, "cutoff", "c", "descriptor cutoff override (MTP/MTM use the potential cutoff)")
           .addop(lenergy, "local-energy-components", "le", "use local energy components")
           .addop(fuse_species, "fuse-species-info", "fs", "when use local energy components, whether make species info share one position")
           .addop(discard_local, "discard-local-components", "d", "discard local components")
           .addop(localname, "local-name", "", "local descriptor name")
           .addop(globalname, "global-name", "", "global descriptor name")
           .addop(settingfile, "settings", "", "setting file", true)
           .attach_settings("", "Descriptor settings (type-specific settings use the shown prefix)",
               [](std::ostream &os) {
                   descriptor_wrapper().ListSettings(os, "");
                   {
                       DescriptorBehler b; b.InitSettings();
                       b.ListSettings(os, "behler:");
                   }
               })
           .set_helponexit(false)
           .phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        exit(0);
    }
    if (cmdopts.was_provided("settings") || GPIO::file_exist(settingfile))
    {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    const bool override_cutoff = cmdopts.was_provided("cutoff") || !settings["cutoff"].empty();
    if (override_cutoff) {
        if (!std::isfinite(neig_cutoff) || neig_cutoff <= 0)
            GPERROR("Descriptor --cutoff must be finite and positive");
        if (dtype == "mtp" || dtype == "mtm")
            GPERROR("MTP/MTM descriptors use the potential cutoff; --cutoff is unsupported");
    }
    if (GPIO::file_exist(indexfile))
    {
        iobj.readmap(indexfile);
    }
    else
    {
        cerr << "ERROR: indexfile not found\n";
    }

    if (!iobj.read(atoms, infile, informat))
        GPERROR("Failed to read descriptor input: " + infile);
    if (atoms.empty()) GPERROR("No structures in descriptor input: " + infile);
    for (const auto &atom : atoms)
        if (atom.size() == 0) GPERROR("Descriptor input contains an empty structure");

    // ── MTP (built-in moment tensor potential descriptors) ──────────
    if (dtype == "mtp") {
        // Load outside OpenMP so file/format errors reach the command handler.
        std::vector<std::unique_ptr<Potentials::PotMTP>> potentials;
        for (int t = 0; t < omp_get_max_threads(); ++t)
            potentials.push_back(std::make_unique<Potentials::PotMTP>(potfile));
        neig_cutoff = potentials.front()->get_cutoff();
        std::exception_ptr error;
        #pragma omp parallel for
        for (int i = 0; i < atoms.size(); ++i) {
            try {
                auto &pot = *potentials[omp_get_thread_num()];
                int dsize = lenergy ? pot.get_lin_coeff_size() : pot.get_alpha_scalar_moments();
                if (lenergy && fuse_species) dsize = pot.get_alpha_scalar_moments() + 1;
                const int offset = lenergy ? 0 : 1;
                vector<double> local_disp(dsize + 1, 0);
                VectorXd local_disp_e(dsize);
                VectorXd global_disp = VectorXd::Zero(dsize);
                Neighborhoods nbhs(atoms[i], neig_cutoff);
                if (!discard_local) atoms[i].alloc_lvd(localname, dsize);
                for (int j = 0; j < nbhs.size(); ++j) {
                    if (lenergy)
                        pot.compute_local_ecmpt(nbhs[j], local_disp, fuse_species);
                    else
                        pot.compute_descriptors(nbhs[j], local_disp);
                    memcpy(local_disp_e.data(), local_disp.data() + offset, sizeof(double) * dsize);
                    if (!discard_local)
                        atoms[i].local_attribs_v_[localname].col(j) = local_disp_e;
                    global_disp += local_disp_e;
                }
                atoms[i].global_attribs_v_[globalname] = global_disp / (double)atoms[i].size();
            } catch (...) {
                #pragma omp critical(imrmlp_descriptor_error)
                {
                    if (!error) error = std::current_exception();
                }
            }
        }
        if (error) std::rethrow_exception(error);
    }
    // ── Descriptors-based types: unified factory + compute loop ──────
    else if (dtype == "biso3" || dtype == "biso4" || dtype == "behler" || dtype == "bondorder" || dtype == "mtm") {
        std::unique_ptr<Potentials::PotMTP> mtm_pot;  // must outlive DescriptorMTM
        std::unique_ptr<Descriptors> desc;
        auto desc_settings = settings.ExtractSubSettings(dtype);
        if (override_cutoff)
            desc_settings.Modify("cutoff", fmt::format("{:.17g}", neig_cutoff));
        else if (dtype == "bondorder" && desc_settings["cutoff"].empty())
            desc_settings.Modify("cutoff", fmt::format("{:.17g}", neig_cutoff));
        if (dtype == "biso3")
            desc = std::make_unique<Descriptor_BispectrumSO3>(biso3_lmax, biso3_rmin,
                override_cutoff ? neig_cutoff : biso3_rmax, biso3_rsize, biso3_real_harmonics);
        else if (dtype == "biso4")
            desc = std::make_unique<Descriptor_BispectrumSO4>(desc_settings);
        else if (dtype == "behler")
            desc = std::make_unique<DescriptorBehler>(desc_settings);
        else if (dtype == "bondorder")
            desc = std::make_unique<DescriptorBondOrder>(desc_settings);
        else if (dtype == "mtm") {
            mtm_pot = std::make_unique<Potentials::PotMTP>(potfile);
            desc = std::make_unique<DescriptorMTM>(*mtm_pot);
        }

        auto *d = desc.get();
        int dsize = d->feature_size();
        double rc = d->cutoff_;

        // SO4 can average densities before forming invariants.
        auto *so4_desc=dynamic_cast<Descriptor_BispectrumSO4*>(d);
        bool use_inner_avg = so4_desc && so4_desc->uses_inner_density();

        for (int i = 0; i < atoms.size(); ++i) {
            auto &at = atoms[i];
            if (!discard_local) at.alloc_lvd(localname, dsize);
            Neighborhoods nbhs(at, rc);
            Eigen::VectorXd global_disp(dsize);
            global_disp.setZero();
            Eigen::VectorXd m;

            if (use_inner_avg) {
                // Inner averaging: average C coefficients over all centers
                so4_desc->compute_global_density(nbhs,global_disp);
                if (!discard_local) {
                    if(so4_desc) {
                        for(int j=0;j<nbhs.size();++j) {
                            d->compute(nbhs[j],m);at.local_attribs_v_[localname].col(j)=m;
                        }
                    } else at.local_attribs_v_[localname].setZero();
                }
            } else {
                for (int j = 0; j < nbhs.size(); ++j) {
                    d->compute(nbhs[j], m);
                    if (!discard_local) at.local_attribs_v_[localname].col(j) = m;
                    global_disp += m;
                }
            }
            if(at.size()==0 || use_inner_avg)at.global_attribs_v_[globalname]=global_disp;
            else at.global_attribs_v_[globalname]=global_disp/(double)at.size();
        }
    }
    else {
        cerr << "ERROR: unknown descriptor type \"" << dtype
             << "\". Supported: mtp (mtp accepted), mtm, biso3, biso4, behler, bondorder\n";
        return 1;
    }

    if (discard_local)
        for (auto &at : atoms) at.local_attribs_v_.erase(localname);
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("Failed to write descriptors: " + outfile);

    return 0;
}
