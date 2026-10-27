// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _IMRMLP_MSD_H
#define _IMRMLP_MSD_H

#include "libalglib/ap.h"
#include "base/structures.h"
#include "utils/gputils.h"
#include "libalglib/fasttransforms.h"
#include "libalglib/interpolation.h"
#include <limits>
/**
 * TODO: adaptive
 * 
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
 * We firstly compute per-atom msd, then add them together
 * 
 * see:
 * https://www.neutron-sciences.org/articles/sfn/pdf/2011/01/sfn201112010.pdf
 * **/
// use equation MSD(t) = <|r_i(t) - r_i(0) - [r_{cm}(t) - r_{cm}(0)]|^2>
//         instead of MSD(t) = <|r_i(t) - r_i(0)|^2>
class ComputeMSD : public ConnectSettings
{
private:
    // ═══════════════════════════════════════════════════════════════════
    // Settings
    // ═══════════════════════════════════════════════════════════════════
    int nsteps_total = 0; // number of output lags
    int nframes_total = 0;
    double ts = 1.0; // timestep (should in ps)

    // ── Drift correction ──
    bool drift_cm = true;
    bool drift_vm = false;   // drift = \sum v_m * M_m

    // ── Algorithm selection ──
    bool tac_avg = true;
    bool fft = true;
    int max_length = -1;

    // ── Output control ──
    bool proj = false;
    bool smooth = false;
    bool do_lfit = false;
    bool compute_total = true;
    bool compute_partial = false;
    bool compute_track = false;
    bool compute_groups = false;
    bool compute_deri = false;

    // ── Linear fit ──
    double lfit_timestart = 0;
    double lfit_timeend = 0;
    int lfit_start = 0;
    int lfit_end = 0;
    double fit_regfactor = 0.01;
    int fit_M = 4;

    // ── Finite-size correction for diffusion coefficient ──
    int fsc_mode = 0;          // 0=none, 1=Yeh-Hummer(2004), 2=generalized
    double temperature = 300.0;// simulation temperature (K)
    double viscosity = 0.0;    // shear viscosity (mPa·s), 0=skip
    double fsc_xi = 2.837298;  // hydrodynamic shape factor (cubic=2.837298)

    // ── Input files / track / group ──
    std::string f_track_id = "TRACKID";
    std::string f_group_id = "GROUPID";
    std::string f_weight_id = "";
    std::string f_total = "MSD_TOTAL";
    std::string f_type = "MSD_TYPE";
    std::string f_track = "MSD_TRACK";
    std::string f_group = "MSD_GROUP";

    // ═══════════════════════════════════════════════════════════════════
    // Working state (allocated in compute())
    // ═══════════════════════════════════════════════════════════════════
    int natoms = 0;
    int group_size = 0;
    int track_size = 0;
    std::vector<int> chemmap;
    std::vector<int> tbc;
    std::vector<int> track_filters;
    std::vector<int> track_inds;
    std::vector<int> types_tbc;
    std::map<int, int> types_tbccount;
    std::vector<std::set<int>> group_idxs;

    std::vector<Eigen::RowVector3d> cm;  // center-of-mass per frame
    std::vector<Eigen::RowVector3d> vm;  // center-of-mass velocity per frame
    double atoms_volume = 0;              // cell volume from first frame
    std::vector<std::vector<Eigen::RowVector3d>> positions;
    bool using_default_weights = false;

    // Per-atom MSD buffers: msd_x[i][m] = MSD of atom i at lag m
    std::vector<std::vector<double>> msd_x, msd_y, msd_z;
    void write_header(std::ofstream &ofs);
    void reset_run_state();
    void prepare_output_selection(const Atoms &first);
    std::string output_path(const std::string &category, int id = 0) const;
    void validate_settings_and_input(const std::vector<Atoms> &atoms) const;
    void prepare_positions(const std::vector<Atoms> &atoms);
    void apply_drift(const std::vector<Atoms> &atoms);
    void apply_finite_size_correction();

public:
    struct DiffusionFitResult
    {
        bool fitted = false;
        int begin = 0;
        int end = 0; // exclusive
        double slope = std::numeric_limits<double>::quiet_NaN();
        double intercept = std::numeric_limits<double>::quiet_NaN();
        double slope_stderr = std::numeric_limits<double>::quiet_NaN();
        double D_PBC = std::numeric_limits<double>::quiet_NaN();
        double D_correction = 0.0;
        double D_infinite = std::numeric_limits<double>::quiet_NaN();
    };

    // ═══════════════════════════════════════════════════════════════════
    // Input data
    // ═══════════════════════════════════════════════════════════════════
    std::vector<double> atom_masses;
    std::vector<double> atomic_weight;

    // ═══════════════════════════════════════════════════════════════════
    // Results: time axis
    // ═══════════════════════════════════════════════════════════════════
    std::vector<double> time;

    // ── Total MSD ──
    std::vector<double> msd_total, msd_total_d, msd_total_s;
    std::vector<double> diffusion_total_inst;
    std::vector<double> msd_total_x, msd_total_y, msd_total_z;
    std::vector<double> msd_total_x_d, msd_total_y_d, msd_total_z_d;
    std::vector<double> msd_total_x_s, msd_total_y_s, msd_total_z_s;
    double k_total = std::numeric_limits<double>::quiet_NaN();
    double b_total = std::numeric_limits<double>::quiet_NaN();
    DiffusionFitResult diffusion_total;

    // ── Per-type MSD ──
    std::vector<std::vector<double>> msd_type, msd_type_d, msd_type_s;
    std::vector<std::vector<double>> diffusion_type_inst;
    std::vector<std::vector<double>> msd_type_x, msd_type_y, msd_type_z;
    std::vector<std::vector<double>> msd_type_x_d, msd_type_y_d, msd_type_z_d;
    std::vector<std::vector<double>> msd_type_x_s, msd_type_y_s, msd_type_z_s;
    std::vector<double> k_type, b_type;
    std::vector<DiffusionFitResult> diffusion_type;

    // ── Per-track MSD ──
    std::vector<std::vector<double>> msd_track, msd_track_d, msd_track_s;
    std::vector<std::vector<double>> diffusion_track_inst;
    std::vector<double> k_track, b_track;
    std::vector<DiffusionFitResult> diffusion_track;

    // ── Per-group MSD ──
    std::vector<std::vector<double>> msd_group, msd_group_d, msd_group_s;
    std::vector<std::vector<double>> diffusion_group_inst;
    std::vector<std::vector<double>> msd_group_x, msd_group_y, msd_group_z;
    std::vector<std::vector<double>> msd_group_x_d, msd_group_y_d, msd_group_z_d;
    std::vector<std::vector<double>> msd_group_x_s, msd_group_y_s, msd_group_z_s;
    std::vector<double> k_group, b_group;
    std::vector<DiffusionFitResult> diffusion_group;

    // ── Spline-smoothed per-atom ──
    std::vector<std::vector<double>> msd_x_s, msd_y_s, msd_z_s;
    std::vector<std::vector<double>> msd_x_d, msd_y_d, msd_z_d;

    // ═══════════════════════════════════════════════════════════════════
    // Methods
    // ═══════════════════════════════════════════════════════════════════

    void InitSettings(const GPSettings &_settings)
    {
        Connect(fft, "fft");
        Connect(drift_cm, "drift_cm", "remove per-frame center-of-mass displacement");
        Connect(drift_vm, "drift_vm", "remove integrated center-of-mass velocity; exclusive with drift_cm");
        Connect(tac_avg, "tacavg");
        Connect(ts, "timestep", "raw-frame timestep in ps; calc_prop applies --skip-every");
        Connect(proj, "projection", "output Cartesian x/y/z components");
        Connect(compute_total, "total");
        Connect(compute_partial, "types");
        Connect(compute_track, "track");
        Connect(compute_groups, "groups");
        Connect(compute_deri, "der");
        Connect(smooth, "smooth");
        Connect(fit_regfactor, "reg");
        Connect(fit_M, "bsize");
        Connect(f_track_id, "trackid");
        Connect(f_group_id, "groupid");
        Connect(f_weight_id, "weightfile");
        Connect(f_total, "output_total");
        Connect(f_type, "output_types");
        Connect(f_track, "output_track");
        Connect(f_group, "output_group");
        Connect(max_length, "max_length", "maximum number of lag samples; -1 uses all frames");
        Connect(lfit_timestart, "fit_start", "explicit diffusion-fit start time in ps");
        Connect(lfit_timeend, "fit_end", "explicit diffusion-fit end time in ps");
        Connect(do_lfit, "do_lfit", "fit diffusion coefficient in the explicit fit window");
        Connect(fsc_mode, "fsc_mode");
        Connect(temperature, "temperature");
        Connect(viscosity, "viscosity");
        Connect(fsc_xi, "fsc_xi");
    }

    ComputeMSD() { InitSettings(GPSettings{}); }
    ~ComputeMSD(){};
    ComputeMSD(const GPSettings &_settings)
    {
        InitSettings(_settings);
        ApplySettings(_settings);
    }

    void compute_spline(alglib::real_1d_array &x, std::vector<double> &y, int &fit_M, double &lam, std::vector<double> &s, std::vector<double> &d);
    void compute_spline(alglib::real_1d_array &x, std::vector<double> &y, int &fit_M, double &lam, std::vector<double> &s);

    void compute(const std::vector<Atoms> &atoms);
    void compute(std::vector<Atoms> &atoms)
    {
        compute(static_cast<const std::vector<Atoms> &>(atoms));
    }
    void compute_der();

    // Internal helpers extracted from compute()
    void allocate_buffers(const std::vector<Atoms> &atoms);
    void build_tbc_list();
    void dispatch_algorithm(const std::vector<Atoms> &atoms);

    // Weight normalization helper
    static void normalize_weights(std::vector<double> &wgt, double &total,
                                  const std::vector<double> &atomic_weight,
                                  const std::vector<int> &indices);

    void compute_msd_nac(const std::vector<Atoms> &atoms);
    void compute_msd_ac(const std::vector<Atoms> &atoms);
    void compute_msd_ac_fft(const std::vector<Atoms> &atoms);
    //void compute_msd_ac_fft_nprj(const std::vector<Atoms> &atoms);
    void postprocessing();
    void postprocessing_nprj();

    void lfit(const std::vector<double> &x, const std::vector<double> &y,
              double &k, double &b, DiffusionFitResult *result = nullptr);

    void read_group();
    void read_filter();
    void read_weight();
    std::vector<std::string> planned_output_paths(const Atoms &first);
    void write_results();

};

#endif
