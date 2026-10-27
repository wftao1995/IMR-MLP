// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/structures.h"
#include "potentials/potential.h"
#include "io/gpio.h"
#include "utils/gputils.h"
#include "utils/mpi_stubs.h"
#include "periodic_subspace.h"
#include "potentials/potential_mindist.h"
#include "tools/constants.h"
#include <array>
#include <fstream>
#include <functional>
#include <set>
#include <stdexcept>
#include <memory>
// #include "train/train_mtp.h"

namespace AL { class ALPotential; }
class NEBPathSymmetry;
class NEBPathSeeds;

// Only geometry failures are recoverable optimization trials. Calculator,
// configuration and I/O exceptions retain their original meaning.
class NEBInvalidGeometry : public std::invalid_argument {
public:
    using std::invalid_argument::invalid_argument;
};

/**
 * @brief NEB driver
 * @ingroup mindriver
 */

class NEBDriver : public ConnectSettings {
public:
    enum class StopReason {
        not_run,
        converged,
        max_iterations,
        active_learning_abort,
        no_internal_barrier,
        numerical_failure,
        max_evaluations,
        stalled
    };

    std::vector<double> tangent;
    std::vector<Atoms> images;
    std::vector<double> spconsts; // spring constants for each segment

    // each line represents a configuration
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> x, g;
    Eigen::Matrix3d refcell, ivrefcell;

    // constants during single run
    double Leq; // for EB method

    int natom;
    int npx;   // 3 * natom
    double pressure = 0;
    double cell_weight = 100.0; // metric weight for VC-NEB cell degrees of freedom

    double spring = 5.0;
    double spring_max = 15.0;
    double spring_min = 1.0;
    bool climb = false;
    std::string climb_mode = "highest"; // highest or all_local
    int climb_delay = 0;
    bool vcneb = false;
    bool periodic_subspace = false; // Explicit mixed coordinates for partial PBC VC.
    bool subspace_active = false;
    PeriodicSubspaceCoordinates subspace;
    bool vally = false; // if metastable phase exist, make sure one image on this point
    bool pin = false; // pin transition state or metastable state
    // check if atom move over pbc between different images
    bool check_mic = true;
    bool preserve_complete_path = false; // Explicit fresh-path reference; never set by restart.
    std::string omp_mode = "images";
    bool keep_symm = false;
    double symprec = 1e-5;
    std::string symmetry_operations, symmetry_diagnostics;
    double perturb_amplitude = 0;
    int perturb_seed = 0;
    std::string symmetry_release;
    int symmetry_release_trials = 3, symmetry_release_seed = 0;
    int symmetry_release_steps = 200, symmetry_release_evaluations = 2001;
    double symmetry_release_amplitude = .01, symmetry_release_strain = .001;
    double symmetry_release_energy_tol = 1e-4;
    std::string symmetry_release_status = "disabled";
    // Optional batch evaluator; the CLI supplies its MPI worker service here.
    std::function<void(std::vector<Atoms>&, int, int)> efs_evaluator;
    bool sel_dyn = false;
    std::string cmask = "111111111";
    int iref = -1; // -2: reference cell use average of init and final, n: the n-th image (start from 0), -1: the last image, -3: read from restart file (handled externally)
    bool symm_strain = true;
    bool recast_atomic_force = true;
    bool restore_F_cholesky = true; // use cholesky decomposition to restore strain matrix, otherwise use sqrt of matrix
    bool apply_scal = false; // for vc + symm_strain only
    // xx yy zz yz xz xy
    std::string ciistr;
    std::array<double, 6> ciival {100, 100, 100, 50, 50, 50};
    Eigen::VectorXd scalval;

    int debug_level = 0;

    /**
     * @brief How to calculate spring and tangent force
     * 
     */
    // int force_proj_type = 2; // 1 for original, 2 for adjusted (may be equal distance)
    /**
     * @brief How to calculate tangent
     * 1. J. Chem. Phys. 113, 9978–9985 (2000), https://doi.org/10.1063/1.1323224
     *     Imporved tangent estimate
     * 2. Original definition
     * 3. J. Chem. Phys. 145, 094107 (2016)
     *    EB method
     * 
     * 
     * Usually type 1 is OK (I have fully tested this)
     * sometimes tangent_type = 1 with sproj_type = 2 will give good results, I don't know why
     */
    int tangent_type = 1; // 1 for adjusted (may be faster), 2 for original
    int sproj_type = 1; // 1 for adjusted, 2 for original

    // TODOs
    bool variable_string = false;
    double energy_cutoff;
    double k_delta;
    // end
    bool found_state_site = false;
    std::set<int> iclimbs;  // climbing image indices (persistent, for pin mode)
    std::set<int> ivallys;  // valley image indices (persistent, for pin mode)

    int iter = 0;
    bool write_traj = false;
    int nblock = 5;
    int vcoffset = 0;

    double neb_ftol = 1e-2;    // NEB convergence: max perpendicular force
    double max_perp_force = 0; // tracked per iteration
    double max_spring_force = 0;
    double max_neb_force = 0;
    double saddle_g_proj = 0;  // g_proj at climbing image (saddle), for convergence check
    bool check_saddle_g_proj = false; // require |saddle_g_proj| < neb_ftol for convergence
    int climbing_image = -1;
    bool climbing_active = false;
    bool converged = false;
    StopReason stop_reason = StopReason::not_run;
    int optimizer_iterations = 0;
    int force_evaluations = 0;
    int accepted_steps = 0;
    int rejected_steps = 0;
    long long image_force_evaluations = 0;
    std::string solver_policy = "legacy"; // promote only after material acceptance
    double max_cell_strain = 0.002;
    int max_force_evaluations = 0; // zero derives 10*max_iterations+1
    double last_atomic_move = 0;
    double last_cell_change = 0;
    std::string last_fallback = "none";
    std::string diagnostics_file; // optional accepted-state CSV

    struct RunOptions {
        std::string driver = "fire";
        int max_iterations = 1000;
        double max_move = 0.1;
        int accepted_step_offset = 0; // restart phase clock; optimizer history is fresh
        int max_krylov_directions = 160; // 320/640 are explicit diagnostic extensions
        std::string experimental_acceleration = "none"; // explicit same-build ablations
        GPSettings optimizer_settings;
    };

    // IDPP initialization
    std::string init_method = "linear"; // "linear" or "idpp"
    double idpp_fmax = 0.1;   // IDPP force convergence threshold
    int idpp_max_steps = 100;  // IDPP max optimization steps
    
    double f = 0; // Reported band energy; projected forces are nonconservative.
    int iclimb_ = -1; // cached climbing image index for deletion protection
    Potentials::AnyPotential *pot = nullptr;
    Potentials::PotMDIST pot_mindist;
    bool mindist = false;

    // TODO: make relax like this
    GPIO *iobj = nullptr;
    std::string trajfile = "nebtraj.cfg";
    std::string trjformat = "mlpcfg";
    std::string external_stress_str = ""; // only xx yy and zz can be accepted
    Eigen::Matrix3d external_stress = Eigen::Matrix3d::Zero();

    std::ofstream trjofs;

    // advanced options:
    bool variable_spring = false;
    bool variable_image = false;
    bool needs_image_adjust = false;     // signal that image count changed
    double image_insert_thresh = 2.0;    // insert when segment > thresh * avg
    double image_remove_thresh = 0.3;    // remove when segment < thresh * avg
    int image_adjust_interval = 10;      // check spacing every N iterations
    int max_image_count = 20;            // upper limit on image count
    int min_image_count = 5;             // lower limit, including endpoints

    // Active learning bridge (shared with relax via ALPotential)
    AL::ALPotential *al_pot_ = nullptr;
    bool al_abort_ = false;  // set when grade > threshold_break

public:
    void InitSettings();

    NEBDriver() {
        InitSettings();
    };

    NEBDriver(GPSettings settings);

    double distance(Atoms &a, Atoms &b, double offset);

    // create path using linear interplation
    void fill_path();
    // IDPP (Image Dependent Pair Potential) path initialization
    void idpp_interpolate();
    // Variable image: check and adjust image spacing
    void check_image_spacing();
    void rebuild_path();
    void get_x(Eigen::VectorXd &x_out);

    void LatPos2X(const Eigen::VectorXd &x, Eigen::VectorXd &g);
    void LatPos2X(Eigen::VectorXd &x);
    void X2LatPos_interf(const Eigen::VectorXd &x);
    void X2LatPos(const Eigen::MatrixXd &x);
    void eval();
    int run(const std::string &driver = "fire", int max_iterations = 1000,
            double max_move = 0.1);
    int run(const RunOptions &options);
    // Optional post-run diagnostic on independent drivers; never changes this path.
    void check_symmetry_release(const RunOptions &options);
    // Decode and bound a native-coordinate proposal without an EFS call.
    // A single factor preserves the proposed band direction.
    bool limit_step(const Eigen::VectorXd &base, Eigen::VectorXd &step,
                    double max_move, double &atomic_move, double &cell_change);
    void write_diagnostics(std::ostream &output) const;
    // Shared by the API and CLI: guard accelerated steps for the original
    // tangent's nonconservative force field. The callback starts a fresh
    // optimizer with the supplied iteration budget and updates xin/objective.
    int minimize_guarded(const std::function<int(int)> &minimize, int max_iterations,
                         double max_move, Eigen::VectorXd &xin, double &objective);
    bool check_convergence() const;
    std::string stop_reason_string() const;

    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g);
    void init(std::vector<Atoms>& atoms);
    // Full, already unfolded path; no EFS, interpolation, alignment or optimizer state.
    void prepare_complete_path(const std::vector<Atoms> &, const Eigen::Matrix3d &reference);

    // b is center
    void tagent(int a, int b, int c, Eigen::RowVectorXd &tg);
    // apply MIC so that adjacent images differ by <= 0.5 in fractional coords
    void check_mic_images();
    // update spring constants based on energy profile
    void update_spring_constants();
    // a->b, calcualte projection of gradient along tagent direction
    double tagent_g(int a, int b);
    double tagent_g(int a, int b, int c);

private:
    std::shared_ptr<NEBPathSymmetry> symmetry_;
    void initialize_symmetry();
    void record_symmetry(const std::string &stage, bool append) const;
    Eigen::RowVectorXd project_symmetry(const Eigen::RowVectorXd &) const;
    void project_symmetry_step(Eigen::VectorXd &) const;
    bool stable_evaluation_ = false;
    bool freeze_phase_ = false;
    bool reuse_efs_ = false;
    std::function<void()> legacy_checkpoint_;
    int run_stable(const RunOptions &options);
    int run_legacy(const RunOptions &options);
    void write_accepted_frame();
    void validate_cell(const Eigen::Matrix3d &cell) const;
    std::vector<double> fire_masses() const;
    void validate_images(bool require_potential=true) const;
    friend class NEBPathSeeds;
    void initialize_scaling();
    Eigen::RowVector3d encode_position(const Atoms &image, int atom_index) const;
    Eigen::RowVector3d decode_position(const Eigen::RowVector3d &q,
                                       const Eigen::Matrix3d &cell) const;

public:

    Eigen::Matrix3d get_strain(Eigen::Matrix3d &cell) {
        // return (ivrefcell * cell) - Eigen::Matrix3d::Identity();
        if (symm_strain) {
            return get_align_C(cell);
        } else {
            return get_def(cell);
        }
    }

    Eigen::Matrix3d get_invdef(Eigen::Matrix3d &cell) {
        return cell.inverse() * refcell;
    }

    Eigen::Matrix3d get_def(Eigen::Matrix3d &cell) {
        return ivrefcell * cell;
    }

    Eigen::Matrix3d get_L(Eigen::Matrix3d &C) {
        if (!C.allFinite()) throw NEBInvalidGeometry("NEB: non-finite cell metric");
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> spectrum(C);
        if (spectrum.info() != Eigen::Success ||
            !(spectrum.eigenvalues().minCoeff() > 1e-12 * spectrum.eigenvalues().maxCoeff()))
            throw NEBInvalidGeometry("NEB: cell metric must be positive definite and well conditioned");
        Eigen::Matrix3d L;
        if (restore_F_cholesky) {
            Eigen::LLT<Eigen::Matrix3d> lltOfC(C);
            if (lltOfC.info() != Eigen::Success)
                throw NEBInvalidGeometry("NEB: cell Cholesky decomposition failed");
            L = lltOfC.matrixL();
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(C);
            Eigen::Matrix3d D = solver.eigenvalues().asDiagonal();
            Eigen::Matrix3d V = solver.eigenvectors();
            L = V * D.cwiseSqrt() * V.transpose();
        }
        return L;
    }

    Eigen::Matrix3d get_ncell(Eigen::Matrix3d &C) {
        return refcell * get_L(C);
    }

    Eigen::Matrix3d get_align_C(Eigen::Matrix3d &cell) {
        Eigen::Matrix3d matD = ivrefcell * cell;
        Eigen::Matrix3d matC = matD * matD.transpose();
        return matC;
    }

    Eigen::Matrix3d get_align_L(Eigen::Matrix3d &cell) {
        Eigen::Matrix3d matD = ivrefcell * cell;
        Eigen::Matrix3d matC = matD * matD.transpose();
        Eigen::Matrix3d L = get_L(matC);
        return L;
    }

    Eigen::Matrix3d align_cell(Eigen::Matrix3d &cell) {
        Eigen::Matrix3d matD = ivrefcell * cell;
        Eigen::Matrix3d matC = matD * matD.transpose();
        Eigen::Matrix3d L = get_L(matC);
        Eigen::Matrix3d newcell = refcell * L;
        return newcell;
    }

    Eigen::Matrix3d get_cell(Eigen::Matrix3d &strain) {
        // return refcell * (Eigen::Matrix3d::Identity() + strain);
        if (symm_strain) {
            Eigen::Matrix3d C = strain;
            return get_ncell(C);
        } else {
            Eigen::Matrix3d A = strain;
            Eigen::Matrix3d cell = refcell * A;
            validate_cell(cell);
            return cell;
        }
    }
};
