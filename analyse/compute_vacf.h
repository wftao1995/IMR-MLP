// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef COMPUTE_VACF_H
#define COMPUTE_VACF_H

#include "base/structures.h"
#include "utils/gputils.h"
#include <array>
#include <fstream>
#include <set>
#include <string>
#include <vector>
/**
 * group file like this:
 *  num_groups
 *  [group1]
 *  ids
 *  [group2]
 *  ids
 *  ...
 *
 * filter file like this:
 *  ids ... (natoms in total)
 *
 * We firstly compute per-atom vacf, then add them together
 * 
 * FFT autocorrelation is available via algo=1. Correlations and output retain
 * the Cartesian x/y/z components as well as their total.
 *
 * see:
 * https://www.brown.edu/research/projects/crunch/sites/brown.edu.research.projects.crunch/files/uploads/J.%20Comp%20Phys.pdf
 * **/
class ComputeVACF : public ConnectSettings
{
private:
    // common settings
    int nsteps_total = 0;
    double ts = 1.0; // time steps, used in fitting D0, in unit ps

    int algo = 1; // 0 for nac, 1 for ac-fft, 2 for ac-nofft, 3 for AR (autoregression: fit AR(p) on velocity series)

    int nstart = 0;
    int nstop = -1;
    int ndim = 3;
    // compute settings
    double lal_frac = 1.0; // 0 - 1

    bool proj_only = false;
    bool proj = false; // projection along specific direction
    bool normalize = true;
    bool mass_weighted = false;
    bool remove_drift = false;

    // compute types
    bool compute_total = true;
    bool compute_partial = false;
    bool compute_track = false;
    bool compute_groups = false;
    // bool compute_inte = false;
    bool compute_vdos = false;
    std::vector<Eigen::Vector3d> projection_dir;
    int nprojection = 0;
    std::vector<int> track_filters; // 0 not, 1 yes
    std::vector<int> track_inds;
    std::vector<int> types_tbc;
    std::vector<std::set<int>> group_idxs; // id-index
    std::vector<int> chemmap;
    int group_size = 0;
    int track_size = 0;

    int fft_zero_padding = 0;
    int ar_order = 0; // AR order for algo=3, 0 for auto selection via BIC
    int ar_pmax = 20; // max AR order considered when ar_order=0

    std::vector<int> tbc; // to be computed
    // for smearing
    int ismear = 0;
    double dtau = 0.2;
    double tau0 = 2.0;
    double fmax = 50;

    // for dos
    double freq_step = 0.01; // in unit THz

    // struct param_type {
    //     // gaussian
    //     double m = 1;
    //     double sigma = 0.5; // buffer, will be calculated from freq_cutoff
    //     double x0 = 0; // do not set this
    //     // block
    //     double cutoff_frac = 0.6; // where to cutoff
    //     double dt = 1; // buffer, do not set this
    //     int ncutoff; // buffer, do not set this
    // } params;

    // buffers
    int natoms = 0;
    int noutput = 0;
    int ndos_total = 0;
    int ndos_output = 0;
    int run_start = 0;
    double dv = 0.0; // for dos output, calc freq
    bool using_derived_velocities = false;
    std::vector<std::vector<Eigen::Vector3d>> derived_velocities;
    std::vector<Eigen::Vector3d> velocity_drift;
    void write_header(std::ofstream &ofs, bool projected = false);
    void write_header_dos(std::ofstream &ofs);
    std::vector<std::vector<std::array<double, 4>>> vacf_array;

    Eigen::Vector3d velocity_at(const std::vector<Atoms> &atoms, int iframe, int iatom) const;
    void reset_run_state();
    void prepare_output_selection(const Atoms &first);
    std::string output_prefix(int projection = 0) const;
    static std::string output_path(const std::string &prefix, const std::string &category, int id = 0, bool vdos = false);
    void prepare_velocities(const std::vector<Atoms> &atoms, int requested_start, int requested_stop);
    void prepare_weights();

    void clear_arr()
    {
        vacf_array.resize(natoms);
        for (int i = 0; i < natoms; ++i)
        {
            vacf_array[i].resize(nsteps_total);
            std::fill(vacf_array[i].begin(), vacf_array[i].end(), std::array<double, 4>{0., 0., 0., 0.});
        }
    }

public:
    std::vector<double> atom_masses; // chemical type as index
    std::vector<double> atomic_weight;
    // results
    std::vector<double> time, frquency;
    struct vacf_result
    {
        std::vector<std::array<double, 4>> total; //, total_i, total_s;
        std::vector<std::array<double, 4>> vdos_total;
        std::vector<std::vector<std::array<double, 4>>> vdos_type, vdos_track, vdos_group;
        std::vector<std::vector<std::array<double, 4>>> type; //, type_i, type_s;    // by types, type-vacf
        std::vector<std::vector<std::array<double, 4>>> track; //, track_i, track_s; // by filters
        std::vector<std::vector<std::array<double, 4>>> group; //, group_i, group_s; // by groups
    };
    vacf_result vacf_isotropic;
    std::vector<vacf_result> vacf_projection;

    std::string f_header = "VACF";

    std::string f_track_id = "TRACKID";
    std::string f_group_id = "GROUPID";
    std::string f_proj_dir = "PROJDIR";
    std::string f_weight_id = "";

    void InitSettings(const GPSettings &_settings)
    {
        Connect(algo, "algo");
        Connect(ts, "timestep", "timestep, assume in unit ps, note that velo is assumed in unit A/ps");
        Connect(proj, "projection");
        Connect(proj_only, "noiso");
        Connect(compute_total, "total");
        Connect(compute_partial, "types");
        Connect(compute_track, "track");
        Connect(compute_groups, "groups");
        Connect(fft_zero_padding, "zero_padding");
        Connect(dtau, "dtau", "smearing coeffs, default 0.2");
        Connect(tau0, "tau0", "default 2.0");
        Connect(ndim, "ndim");
        Connect(ismear, "ismear");
        Connect(fmax, "fmax");
        Connect(freq_step, "dfreq", "step size in vdos, in unit THz, default 0.01");
        Connect(lal_frac, "lal_frac", "how many fraction of steps to use for vdos");
        // Connect(compute_inte, "int");
        // window
        // Connect(params.m, "m", "Gaussian m");
        // Connect(params.cutoff_frac, "cutoff_frac");
        Connect(compute_vdos, "vdos");
        Connect(ar_order, "ar_order", "AR order for algo=3, 0 for auto selection via BIC");
        Connect(ar_pmax, "ar_pmax", "max AR order considered when ar_order=0");
        Connect(f_track_id, "trackid");
        Connect(f_group_id, "groupid");
        Connect(f_proj_dir, "projdir");
        Connect(f_header, "output_header");
        Connect(f_weight_id, "weightfile");
        Connect(normalize, "normalize");
        Connect(mass_weighted, "mass_weighted", "use atomic masses as aggregation weights");
        Connect(remove_drift, "remove_drift", "subtract mass-weighted center-of-mass velocity per frame");
        Connect(nstart, "start_from"); // start from 0
        Connect(nstop, "end_to"); // negative to include all, if 1000 frams in total, set to 1000 to include all 
    }

    ComputeVACF(){};
    ~ComputeVACF(){};
    ComputeVACF(const GPSettings &_settings)
    {
        InitSettings(_settings);
        ApplySettings(_settings);
        read_weight();
    }

    void compute(const std::vector<Atoms> &atoms);
    // void compute_int(vacf_result &);
    void compute_vdosfft(std::vector<std::array<double, 4>> &in, std::vector<std::array<double, 4>> &out);


    void compute_vacf_nac(const std::vector<Atoms> &atoms);
    void compute_vacf_ac(const std::vector<Atoms> &atoms);
    void compute_vacf_ac_ar(const std::vector<Atoms> &atoms);
    void compute_vacf_ac_fft(const std::vector<Atoms> &atoms);
    void compute_vacf_nac(std::vector<std::vector<double>> &);
    void compute_vacf_ac(std::vector<std::vector<double>> &);
    void compute_vacf_ac_ar(std::vector<std::vector<double>> &);
    void compute_vacf_ac_fft(std::vector<std::vector<double>> &);
    void postprocessing(vacf_result &);

    // smearing
    double smearing(int istep);
    void read_group();
    void read_filter();
    void read_proj();
    void read_weight();
    std::vector<std::string> planned_output_paths(const Atoms &first);
    void write_results();
    void write_results(vacf_result &res, const std::string &fnm);
};

#endif
