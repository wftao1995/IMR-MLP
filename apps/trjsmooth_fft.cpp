// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "analyse/compute_fftfilter.h"
#include "io/gpio.h"
#include "utils/cmdopts.h"
#include "libalglib/ap.h"
#include "utils/gputils.h"
#include <iomanip>
#include <limits>
#include <sstream>

using namespace std;
using namespace Eigen;

/**
 * @brief 
 * 
 * @param argc 
 * @param argv 
 * @return int 
 * 
 * @todo Window
 */

int smooth_traj(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    string indexfile = "index.map";
    string informat = "";
    string outformat = "";
    string trjfile = "";
    string outfile = "";
    vector<Atoms> atoms;
    bool overwrite = false;
    double ts = 0.1; // fs
    double freq_cut = 1; // fs^-1, matching ComputeSmoothFFT
    int outputrate = 1;
    int interpolation_factor = 1;
    GPSettings settings;
    string settingfile = "settings.ini";
    int wtype = ComputeSmoothFFT::SUPERGAUSSIAN;

    bool scell = false; // scale to same lattice constants, to average

    cmdopts.addop(indexfile, "index", "id", "index file")
        .addop(informat, "input-format", "if", "input format")
        .addop(outformat, "output-format", "of", "output format")
        .addop(trjfile, "input", "i", "input file")
        .addop(outfile, "output", "o", "output file")
        .addop(overwrite, "overwrite", "ov", "", false, 0)
        .addop(ts, "timestep", "ts", "timestep in fs")
        .addop(freq_cut, "freq-cutoff", "fq", "smooth cutoff frequency in fs^-1")
        .addop(outputrate, "output-rate", "or", "output every nth frame after interpolation")
        .addop(interpolation_factor,"interpolation-factor","","positive integer Fourier interpolation factor")
        .addop(settingfile, "settings", "", "", true)
        .addop(wtype, "wtype", "w", "window type")
        .addop(scell, "scale-cell", "sc", "", false, 0)
        .phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(cout);
        return 0;
    }

    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        apply_settings_to_cmdopts(cmdopts, settings);
    }
    if (GPIO::file_exist(indexfile))
    {
        if (!iobj.readmap(indexfile)) // should be with atom radii
            GPERROR("smooth_traj: failed to read index file " + indexfile);
    }
    else
    {
        cerr << "ERROR: indexfile not found\n";
        exit(1);
    }

    if (!GPIO::file_exist(trjfile)) {
        cerr << "ERROR: inputfile " + trjfile + " not found\n";
        exit(1);
    }

    if (GPIO::file_exist(outfile) && !overwrite) {
        cerr << "ERROR: file exists, please use --overwrite to overwrite it\n";
        exit(1);
    }

    // iobj.set(atoms);
    if (!iobj.read(atoms, trjfile, informat)) {
        std::cerr << "Error loading trajectories\n";
        exit(1);
    }
    if (atoms.size() < 2)
        GPERROR("smooth_traj: input must contain at least two frames");
    if (outputrate <= 0)
        GPERROR("smooth_traj: output-rate must be positive");

    if (scell) {
        Matrix3d cell = Matrix3d::Zero();
        int nf = atoms.size();
        for (int i = 0; i < nf; ++i) {
            cell += atoms[i].cell_;
        }
        cell /= nf;
        #pragma omp parallel for
        for (int i = 0; i < nf; ++i) {
            atoms[i].set_cell(cell);
        }
    }

    ComputeSmoothFFT smoother(settings);
    if(cmdopts.was_provided("interpolation-factor") || settings.count("interpolation-factor"))
        smoother.interpolation_factor=interpolation_factor;
    // Both settings spellings and the CLI use fs^-1; explicit CLI takes precedence.
    if (cmdopts.was_provided("freq-cutoff") || settings.count("freq-cutoff") ||
        !settings.count("freq_cutoff"))
        smoother.params.freq_cutoff = freq_cut;
    try
    {
        smoother.compute2(atoms, ts, wtype);
    }
    catch(const alglib::ap_error &e)
    {
        std::cerr << e.msg << '\n';
        return 1;
    }

    // compute2 differentiates with the supplied fs timestep; trajectory I/O
    // stores velocities in Angstrom/ps. Existing filtered velocities keep their units.
    if (smoother.rebuild_velo)
        for (auto &atom : atoms)
            for (auto &velocity : atom.velocities_)
                velocity *= 1000;
    
    if (outputrate > 1) {
        int ct = 0;
        for (int i = 0; i < atoms.size(); i += outputrate) {
            atoms[ct] = atoms[i];
            ++ct;
        }
        atoms.resize(ct);
    }
    // Text CFG serializes arbitrary Feature strings rather than global doubles.
    // Keep the numeric in-memory channel, and bridge only this output format so
    // EXTXYZ does not receive duplicate keys.
    if(smoother.interpolation_factor>1 &&
       (outformat.empty() ? GPIO::guess_format(outfile) : GPIO::format2enum(outformat))==GPIO::MLPCFG) {
        for(auto& atom:atoms) {
            std::ostringstream elapsed;
            elapsed<<std::setprecision(std::numeric_limits<double>::max_digits10)
                   <<atom.global_attribs_d_.at("fft_elapsed_fs");
            atom.features_["fft_elapsed_fs"]=elapsed.str();
        }
    }
    if (!iobj.write(atoms, outfile, outformat))
        GPERROR("smooth_traj: failed to write output file " + outfile);

    return 0;
}
