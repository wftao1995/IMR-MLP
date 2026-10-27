// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * V2 static relaxation: conservative optimizers, AUL or modern SQP constraints,
 * coordinate filters, and optional Hessian/neighbor metrics. Force-only mode
 * projection uses FIRE or the explicitly selected force-only line search.
 * GPMin and constrained native dynamics are deferred; they are not advertised.
 */

/**
 * Linear constraint file format:
 * line1: k (number of constraints), n
 * line2+: kxn matrix A
 * line++: kx1 vector b
 */

#include "base/structures.h"
#include "potentials/potential.h"
#include "utils/gputils.h"
#include "io/gpio.h"
#include "constraints.h"
#include "nonlinear_coordinates.h"
#include "periodic_subspace.h"
#include "relax_preconditioner.h"
#include "constraints_alg.h"
#include "deplibs/LBFGSpp/include/LBFGS.h"
#include "libalglib/ap.h"
#include "libalglib/optimization.h"
#include "tools/constants.h"
#include "symmetry/symmetry.h"

/**
 * About mode constraint:
 * We supply two methods:
 * 1. Project force vectors into normal modes, and set zero or invert directions, and project back
 *  In this method, one should add a fix_cm filter at last to prevent from translation, might be unstable.
 *  One must use force only method, i.e lbfgspp with force only linear search, or FIRE/GD method
 * 2. Change variables.
 *  In this method, one must not add fix_cm filter, and any other filters. The optimization variables now change
 *  into displacements along specified modes. The force is the projected force in method 1
 * 3. fc_update_interval refreshes native atomic curvature at accepted steps.
 */

// ── Forward declaration ───────────────────────────────────────────────
class RelaxationDrvV2;

// ═══════════════════════════════════════════════════════════════════════
// Modifier classes (outside RelaxationDrvV2 for decoupling)
// ═══════════════════════════════════════════════════════════════════════

enum modifier_type { FILTER, BAIS };

class AnyModifier {
public:
    std::string name = "AnyModifier";
    modifier_type type = FILTER;
    virtual ~AnyModifier() = default;
    virtual void forward(RelaxationDrvV2 *) {};
    virtual void reverse(RelaxationDrvV2 *) {};
    virtual void forward_scaling(RelaxationDrvV2 *) {};
    virtual void init(RelaxationDrvV2 *, void * = nullptr) {};
};

class FilterFixCM : public AnyModifier {
public:
    Eigen::RowVector3d center;
    void forward(RelaxationDrvV2 *);
    FilterFixCM() { name = "FixCM"; }
};

class FilterFixSmallGD : public AnyModifier {
public:
    void forward(RelaxationDrvV2 *);
    FilterFixSmallGD() { name = "FixSmallGD"; }
};

class FilterLF : public AnyModifier {
public:
    Eigen::MatrixXd matrixA, invmatrixA;
    Eigen::VectorXd vectorb;
    Eigen::ColPivHouseholderQR<Eigen::MatrixXd> qr, inv_qr;
    void forward(RelaxationDrvV2 *);
    void reverse(RelaxationDrvV2 *);
    void solve();
    void init(RelaxationDrvV2 *, void * = nullptr);
    void forward_scaling(RelaxationDrvV2 *);
    FilterLF() { name = "LinearTransform"; }
};

class FilterLC : public AnyModifier {
public:
    int n = 0, k = 0, algo = 0;
    Eigen::MatrixXd matrixA;
    Eigen::VectorXd vectorb, b0;
    Eigen::MatrixXd T, Q1, Q2, R1;
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> Tcore;
    void forward(RelaxationDrvV2 *);
    void reverse(RelaxationDrvV2 *);
    void init(RelaxationDrvV2 *, void * = nullptr);
    void forward_scaling(RelaxationDrvV2 *);
    void solve();
    FilterLC() { name = "LinearConstraint"; }
};

class FilterSelDyn : public AnyModifier {
public:
    // Read n followed by exactly n binary flags in the full optimizer space.
    // Return a new selection only after the entire stream has been validated.
    static std::vector<int> read_mask(std::istream &, Eigen::Index full_size);
    std::vector<int> ids_pass;
    Eigen::VectorXd x_save, g_new;
    void forward(RelaxationDrvV2 *);
    void reverse(RelaxationDrvV2 *);
    void forward_scaling(RelaxationDrvV2 *);
    void init(RelaxationDrvV2 *, void * = nullptr);
    FilterSelDyn() { name = "SelectiveDynamics"; }
};

/// Input file: natoms, mask[-1,0,1], eigenvalues, init_disp, projection matrix
class FilterVIB : public AnyModifier {
public:
    Eigen::MatrixXd projection_matrix;
    Eigen::VectorXd mode_factor;
    void forward(RelaxationDrvV2 *);
    void init(RelaxationDrvV2 *, void * = nullptr);
    FilterVIB() { name = "VibrationKeepX"; }
};

class FilterVIB2 : public AnyModifier {
public:
    Eigen::MatrixXd projection_matrix;
    Eigen::VectorXd mode_factor, eigenvalues, x0;
    void forward(RelaxationDrvV2 *);
    void reverse(RelaxationDrvV2 *);
    void init(RelaxationDrvV2 *, void * = nullptr);
    void forward_scaling(RelaxationDrvV2 *);
    FilterVIB2() { name = "Vibration"; }
};

class FilterSymmForceStress : public AnyModifier {
public:
    std::vector<Eigen::Matrix3d> rots;
    Eigen::MatrixXi mapping;
    void forward(RelaxationDrvV2 *);
    void reverse(RelaxationDrvV2 *);
    void init(RelaxationDrvV2 *, void * = nullptr);
};

/// @brief Symmetry-constrained relaxation modifier.
/// Reduces optimization variables to the asymmetric unit only.
/// Atomic positions are expanded/reduced by space-group symmetry operations.
/// Cell DOFs are reduced according to the Bravais lattice type.
/// Must be the only coordinate-transform modifier in modifiers_list
/// (mutually exclusive with selective_dynamics, constraint_vib_modes, etc.).
class FilterSymmCoord : public AnyModifier {
public:
    // ── Orbital data (one orbit per set of symmetry-equivalent atoms) ──
    struct Orbit {
        int iat_rep;                                  // representative atom index
        std::vector<int> iat_orbit;                   // all atom indices in this orbit
        std::vector<Eigen::Matrix3d> W_frac;          // rotation matrices (fractional coords)
        std::vector<Eigen::RowVector3d> t_frac;       // translation vectors (fractional coords)
        Eigen::MatrixXd basis;                        // representative x displacement (3 x k)
        std::vector<Eigen::MatrixXd> maps;             // representative q -> atom x (3 x k)
        std::vector<Eigen::Vector3d> origins;          // fixed, exactly symmetric x reference
        int n_free = 3;
    };
    std::vector<Orbit> orbits;

    // ── Cell constraint groups ──
    // Columns span the symmetry-allowed subspace of the six strain components.
    // Voigt order: 0=xx, 1=xy, 2=xz, 3=yy, 4=yz, 5=zz
    Eigen::MatrixXd cell_basis; // reduced modes -> full symmetric strain

    // ── Dimensionality ──
    int n_full_cell = 0;       // xposoffset (6 for symm_strain)
    int n_full_pos = 0;        // 3 * n_atoms
    int n_red_cell = 0;        // number of independent cell DOFs
    int n_red_pos = 0;         // 3 * n_orbits

    // ── Saved state ──
    Eigen::VectorXd x_save;    // saved full x for forward() re-extraction
    int xposoffset_full = 0;   // saved xposoffset

    // ── Methods ──
    FilterSymmCoord() { name = "SymmCoord"; }
    void init(RelaxationDrvV2 *, void * = nullptr) override;
    void forward(RelaxationDrvV2 *) override;
    void reverse(RelaxationDrvV2 *) override;
    void forward_scaling(RelaxationDrvV2 *) override;

private:
    void setup_cell_constraints(const Symmetry3D::symdataset &, const Eigen::Matrix3d &);
    void setup_position_orbits(RelaxationDrvV2 *,
                               const Symmetry3D::symdataset &dataset);
    // Bound to the first init; run() may rebuild coordinates but never detects
    // a new group or reinterprets operation ids/mappings/images on a restart.
    bool selection_bound_ = false, site_only_ = false, bound_recast_ = false;
    std::string bound_operations_, bound_sites_;
    std::vector<int> selected_sites_, bound_species_;
    Symmetry3D::symdataset position_group_, cell_group_;
    std::vector<Eigen::Matrix<double, Eigen::Dynamic, 3>> operation_images_;
    std::vector<Eigen::RowVector3d> initial_fractional_;
    Eigen::Matrix3d coordinate_reference_cell_ = Eigen::Matrix3d::Identity();
};

class FilterNF : public AnyModifier {
public:
    std::shared_ptr<NonlinearCoordinates> map;
    Eigen::VectorXd point, full_gradient;
    void init(RelaxationDrvV2 *, void * = nullptr) override;
    void forward(RelaxationDrvV2 *) override;
    void reverse(RelaxationDrvV2 *) override;
    void forward_scaling(RelaxationDrvV2 *) override;
    FilterNF() { name = "NonlinearCoordinates"; }
};
using FilterRigidBody = FilterNF;

class FilterWhitening : public AnyModifier {
public:
    RelaxPreconditioning::WhiteningTransform transform;
    Eigen::VectorXd point;
    void forward(RelaxationDrvV2 *) override;
    void reverse(RelaxationDrvV2 *) override;
    void forward_scaling(RelaxationDrvV2 *) override;
    FilterWhitening() { name="HessianWhitening"; }
};

// ═══════════════════════════════════════════════════════════════════════"
// RelaxationDrvV2
// ═══════════════════════════════════════════════════════════════════════

class RelaxationDrvV2 : public ConnectSettings {
public:
    Potentials::AnyPotential *pot = nullptr;
    bool initialized = false;
    Atoms atom;

    /* PARAMETERS */
    struct type_tol {
        double xtol = 0;
        double gtol0 = 1e-1;
        double gtol = 1e-3;
        double ftol = 0;
        double ztol = 1e-6;
        double eps = 1e-6;
        double symprec = 1e-4;
    } tol;

    struct {
        std::string driver = "lbfgs";
        double max_iter = 1000;
        int constraint_iter_limit = 5;
        double max_move = 0.1;
        double min_move = 1e-6;
        double cell_factor = 1;
        int couple_flag = 0;    // 0=none 1=ab 2=bc 3=ca 4=abc
        double pressure = 0;    // external pressure (GPa)
        std::string external_stress_str;
        std::string stiffness_str;
        std::string lattice_mask_str;    // 6 or 9 components
        double disp = 0.01;
        std::string trajfile = "relax-trajectory.xyz";
        std::string indexfile = "index.map";
        /// LC file: k, k×n matrix A (row-major, rank k), k×1 vector b
        std::string lc_file = "LINEAR_CONSTRAINT";
        std::string lf_file = "LINEAR_TRANSFORM";
        std::string vib_file = "VIB_CONSTRAINT";
        std::string seldyn_file = "SELECTIVE_DYNAMICS";
        std::string rigid_groups; // Semicolon-separated groups of comma-separated zero-based atom indices.
        std::string symmetry_operations, symmetry_sites; // Initial dataset operation / atom indices.
        std::string hessian_precond_file;
        bool sparse_precond = false, aul_auto = false, thin_cell_scaling = false;
        double sparse_shift = 1e-3, precond_floor = 1e-8;
        int fc_update_interval = 0;
        std::string precond_file; // Full-coordinate diagonal scaling, before masks/LF/LC; empty uses defaults.
        int nblock = 10;
    } control;

    struct {
        bool periodic_subspace = false; // Explicit mixed periodic/open coordinates.
        bool cartesian = false;
        bool relax_cell = false;
        bool relax_cell_use_strain = true;
        bool symm_strain = true;
        bool atomic_force_recast = true;
        bool relax_atom = true;
        bool fix_vol = false;
        bool fix_shape = false;
        bool apply_disp = false;
        bool disp_gaussian = false;
        bool dprecond = true;
        bool force_switchoff_scaling = false;
        bool write_traj = false;
        bool traj_append = false;
        bool with_constraint_aul = false;
        bool with_constraint_sqp = false;
        bool with_constraint_nlc = false;
        bool with_constraint_lc = false;
        bool selective_dynamics = false;
        bool constraint_vib_modes = false;
        bool vibffilter = false;
        bool linear_constraint = false;
        bool linear_transform = false;
        bool fix_cm = false;
        bool freeze_small_grad = false;
        bool symm_force = false;
        bool keep_symm = false; // Symmetry-constrained relaxation (FilterSymmCoord)
    } flags;

    /* OPTIMIZER PARAMETERS */
    struct gd_buffer {
        double alpha = 1.0;
        double max_move_init = 0.1;
        double step_upscale = 1.2;
        double step_downscale = 0.2;
    } buffer_gd;

    double mdmin_dt = 0.2;
    double quickmin_dt = 0.2;

    LBFGSpp::LBFGSParam<double> lbfgspp_param;

    struct fire_buffer {
        double T, cmax, pmax, ts, alpha;
        double ts_max = 10.0, ts_min = 0.001, ts_init = 2;
        int mdint = 0;
        double alpha_start = 0.25, alpha_reduce = 0.99;
        double ts_grow = 1.1, ts_reduce = 0.5;
        size_t vdfitmax = 2000;
        bool halvstepback = true, initaildeyal = true;
        bool downhill_check = false;
        double ts_backward_ratio = 0.5;
        int ndelay = 20;
        double num_it_ppostive = 0, num_it_pnegative = 0;
    } buffer_fire;

    struct newton_buffer { int bfgs_m = 15; } buffer_newton;

    /* INTERNAL STATE */
    struct {
        bool request_alg_restart = false;
    } state_flags;

    struct {
        int curr_iter = 0, curr_constraint_iter = 0;
        bool converged = false;
        int reach_gtol = 0, reach_ftol = 0, reach_xtol = 0;
        int optstate = -1;
        double max_move = 0;
    } state_vars;

    Eigen::MatrixXd full_hessian, reduced_hessian; // True supplied/native curvature; metric is separate.
    std::unique_ptr<RelaxPreconditioning::SparseMetric> sparse_metric;
    RelaxPreconditioning::AcceptedFCUpdate fc_update;
    std::size_t accepted_steps = 0, fc_refresh_count = 0;
    Eigen::VectorXd aul_previous_residual;
    void initialize_metrics();
    void refresh_fc_metric();
    Eigen::VectorXd unwhiten_gradient(const Eigen::VectorXd &) const;
    Eigen::VectorXd convergence_gradient() const { return unwhiten_gradient(g); }
    double optimizer_gtol() const;
    Eigen::VectorXd preconditioned_gradient(const Eigen::VectorXd &) const;
    double metric_inner(const Eigen::VectorXd &, const Eigen::VectorXd &) const;
    Eigen::MatrixXd constraint_derivatives();
    void initialize_aul_parameters();
    PeriodicSubspaceCoordinates subspace;
    bool subspace_active = false;
    std::vector<Eigen::RowVector3d> subspace_fixed_positions;
    int coordinate_offset(const Atoms &) const;
    std::shared_ptr<NonlinearCoordinates> coordinate_map;
    bool nonlinear_coordinates_active() const { return bool(coordinate_map) || !control.rigid_groups.empty(); }
    Eigen::MatrixXd coordinate_jacobian();
    Eigen::VectorXd x, g;
    Eigen::VectorXd x_prev, g_prev;
    Eigen::VectorXd mode_displacement_init;
    Eigen::VectorXd prec_scaling;
    double f = 0, f_prev = 0;
    double symmetry_atom_correction = 0, symmetry_cell_correction = 0;
    double vol0 = 0;
    int xposoffset = 0;
    double scaling = 1.0;
    double energy0 = 0;
    bool use_dprecond = false;
    std::array<double, 6> symm_strain_mask{1, 1, 1, 1, 1, 1}, stiffness{};
    bool cellmask_set_flag = false;
    Eigen::Matrix3d cellmask = Eigen::Matrix3d::Ones();
    Eigen::Matrix3d external_stress = Eigen::Matrix3d::Zero();
    std::vector<Eigen::RowVector3d> pos_orig;
    Eigen::Matrix3d refcell, ivrefcell;
    alglib::minlbfgsstate bfgs_state;
    alglib::mincgstate cg_state;
    Eigen::Matrix3d curr_stress, gstress;
    GPIO iobj;
    std::ofstream ofs;
    double first_step = -1; // Nonpositive: derive the cap from max_move on each FIRE run.
    int first_step_delay = 1;

    /* MODIFIER LISTS */
    class ALModifier *al_modifier_ = nullptr;  // external AL bridge; NOT in modifiers_list
    std::vector<AnyModifier*> modifiers_list_carg;
    std::vector<AnyModifier*> modifiers_list;
    void apply_modifiers(std::vector<AnyModifier*> &);
    void reverse_modifiers(std::vector<AnyModifier*> &);

    /* CONSTRAINTS */
    bool constrained_solver() const;
    void evaluate_regulator(regulator &reg);
    void compute_regulator(regulator &reg);
    Eigen::MatrixXd constraint_jacobian;
    double kkt_residual = std::numeric_limits<double>::infinity();
    void update_regs();
    std::vector<regulator> constraints;
    std::vector<regulator_alg> nlc_constraints;

    /* EFS EVALUATION */
    void LatPos2X();
    void EFS2G();
    void X2LatPos();
    void eval();

    /* UTILS */
    inline void fixvol()   { flags.fix_vol = true; }
    inline void unfixvol() { flags.fix_vol = false; }
    void set_vib_data(const Eigen::MatrixXd &projection_matrix,
                      const Eigen::VectorXd &mode_factor);
    void write_relax_log(double _f);

    /* OPTIMIZATION */
    bool has_checkpoint = false;
    std::string stop_reason = "not_started";
    Atoms checkpoint_atom;
    Eigen::VectorXd checkpoint_x, checkpoint_g;
    double checkpoint_f = 0, checkpoint_scaling = 1;
    std::vector<regulator> checkpoint_constraints;
    int inner_iter_start = 0;
    bool optimizer_initial_report = false;
    void accept_checkpoint(bool count_step = true);
    void restore_checkpoint();
    int run_impl();
    int run();
    int run_aul();
    int run_alg();
    int run_normal();
    void run_lbfgs();
    void run_lbfgspp();
    void run_cg();
    void run_gd();
    void run_fire();
    void run_mdmin(bool quickmin = false);
    void validate_fire_options() const;

    void reset_state();
    bool check_converage(type_tol &ctol);
    bool check_converage_final(type_tol &ctol);
    bool check_cons_convergence();

    /* MEM MANAGEMENT */
    void free_modifiers() {
        for (auto *m : modifiers_list)
            if (m) delete m;
        modifiers_list.clear();
        for (auto *m : modifiers_list_carg)
            if (m) delete m;
        modifiers_list_carg.clear();
    }

    /* CONSTRUCTION */
    void init(Potentials::AnyPotential *_pot, Atoms &_atom,
              std::vector<regulator> *_constraints = nullptr);
    bool inplace_modifiers_init = true;
    void add_modifiers_filter_fixcm();

    RelaxationDrvV2() { InitSettings(); }
    RelaxationDrvV2(const GPSettings &_settings);
    ~RelaxationDrvV2() {
        free_modifiers();
        if (ofs.is_open()) ofs.close();
    }
    double operator()(const Eigen::VectorXd &mx, Eigen::VectorXd &mg);

    void InitSettings();
    // Cell masks use positive=movable, nonpositive=fixed; values are normalized.
    // Parse first, then validate against the final settings/ISIF coordinate mode.
    void configure_cell_mask(const std::string &text);
    void validate_options();
    // Validate file scaling against the full coordinate count without copying geometry.
    void check_preconditioner(const Atoms &) const;
    // Validate and bind without evaluating a potential or changing driver/geometry state.
    std::vector<regulator> prepare_constraints(const Atoms &, const std::vector<regulator> &);
    void set_isif(int isif);
};
