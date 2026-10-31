// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "relax_new.h"
#include "mindriver/strain_gradient.h"
#include "constraint_expression.h"
#include "analyse/trajectory_transform.h"
#include "mindriver/al_modifier.h"
#include "io/gpio.h"
#include "utils/gputils.h"

namespace {
Eigen::VectorXd read_diagonal_scaling(const std::string &path, int expected_count) {
    if (path.empty()) return {};
    std::ifstream input(path);
    int count;
    if (!input || !(input >> count) || count != expected_count)
        throw std::invalid_argument("precond_file must begin with full coordinate count " +
            std::to_string(expected_count) + ": " + path);
    Eigen::VectorXd scaling(expected_count);
    for (int i = 0; i < expected_count; ++i) {
        if (!(input >> scaling[i]) || !std::isfinite(scaling[i]) || scaling[i] <= 0)
            throw std::invalid_argument("precond_file requires finite positive scaling for every coordinate: " + path);
    }
    input >> std::ws;
    if (!input.eof())
        throw std::invalid_argument("Unexpected trailing data in precond_file: " + path);
    return scaling;
}

std::array<double, 6> strain_stiffness(const std::string &text, double cell_factor) {
    std::array<double, 6> result;
    if (text.empty()) result.fill(160 / eVA3_to_GPa / cell_factor);
    else {
        std::vector<std::string> values;
        str_split(text, values);
        if (values.size() != 6) throw std::invalid_argument("stiffness requires 6 components");
        // Input xx yy zz yz xz xy; packed symmetric strain xx xy xz yy yz zz.
        constexpr int packed[] = {0, 3, 5, 4, 2, 1};
        for (int k = 0; k < 6; ++k)
            result[packed[k]] = std::stod(values[k]) / eVA3_to_GPa / cell_factor;
    }
    return result;
}

// The diagonal belongs to the current full coordinate chart, before filters.
void automatic_scaling(RelaxationDrvV2 &d) {
    if (d.use_dprecond) {
        if (d.xposoffset && !d.control.stiffness_str.empty() && d.subspace_active)
            throw std::invalid_argument("Six-component stiffness requires the three-dimensional strain chart; use precond_file for periodic_subspace");
        if (d.xposoffset && !d.subspace_active) {
            d.stiffness = strain_stiffness(d.control.stiffness_str, d.control.cell_factor);
            for (int k = 0; k < d.xposoffset; ++k) d.prec_scaling[k] = d.stiffness[k] * d.vol0;
        }
        // Mixed coordinates have unit Cartesian rows in the open subspace.
        const Eigen::Matrix3d jac = d.subspace_active ? d.subspace.atomic_jacobian(d.refcell).eval() : d.refcell;
        const Eigen::Vector3d diagonal = jac.rowwise().squaredNorm();
        for (int k = d.xposoffset; k < d.prec_scaling.size(); k += 3)
            d.prec_scaling.segment<3>(k) = diagonal;
    }
    if (d.flags.relax_cell && d.xposoffset &&
        (d.control.thin_cell_scaling || (d.use_dprecond && d.subspace_active))) {
        const auto gram = RelaxPreconditioning::periodic_gram(d.refcell, d.atom.pbc_);
        const double measure = gram.axes.empty() ? 1. : std::sqrt(gram.gram.determinant());
        d.prec_scaling.head(d.xposoffset).setConstant(160 / eVA3_to_GPa / d.control.cell_factor * measure);
    }
    if (!d.prec_scaling.allFinite() || (d.prec_scaling.array() <= 0).any())
        throw std::invalid_argument("Automatic preconditioning requires finite positive coordinate scales (check stiffness and cell_factor)");
}

}

void RelaxationDrvV2::InitSettings()
{
    // Relaxation convergence is force-controlled by gtol unless the user
    // explicitly enables one of LBFGSpp's additional stopping criteria.
    lbfgspp_param.epsilon = 0.0;
    lbfgspp_param.epsilon_rel = 0.0;
    Connect(flags.periodic_subspace, "periodic_subspace", "Explicit periodic-fraction/open-Cartesian coordinates; variable cell changes only the periodic Gram, open rows fixed");
    Connect(flags.cartesian, "cartesian", "Whether to use Cartesian coordinates (True) or fractional coordinates (False)");
    Connect(flags.cartesian, "car_mode", "Whether to use Cartesian coordinates (True) or fractional coordinates (False)", 1, 2);
    Connect(flags.dprecond, "dprecond", "Use automatic diagonal scaling for fractional symmetric-strain or periodic-subspace variable-cell coordinates", 1, 2);
    Connect(flags.write_traj, "write_traj", "Whether to write trajectory out to file");
    Connect(flags.traj_append, "traj_append", "Whether to append to existing trajectory file");
    Connect(flags.disp_gaussian, "disp_gaussian", "Whether to use Gaussian distribution for random displacements");
    Connect(flags.force_switchoff_scaling, "force_switchoff_scaling", "Force switch off scaling, usually combined with LF/LC filter to aviod incorrectly scaling. all scaling will set to 1");
    Connect(flags.with_constraint_aul, "with_constraint_aul", "Augmented Lagrangian method, call library func", 1, 2);
    Connect(flags.with_constraint_sqp, "with_constraint_sqp", "Sequential Quadratic Programming, call library func", 1, 2);
    Connect(flags.with_constraint_nlc, "with_constraint_nlc", "Nonlinear Constraints, call library func", 1, 2);
    Connect(flags.with_constraint_lc, "with_constraint_lc", "Linear Constraints, call library func", 1, 2);
    Connect(flags.constraint_vib_modes, "with_constraint_vib", "Whether to use VIB2 filter (transforms coordinates)");
    Connect(flags.vibffilter, "vibffilter", "Whether to use VIB filter (doesn't transform coordinates)");
    Connect(flags.linear_constraint, "with_linear_constraint", "Whether to use linear constraint (LC) filter");
    Connect(flags.linear_transform, "with_linear_transform", "Whether to use linear transform (LF) filter");
    Connect(flags.fix_cm, "fixcm", "Whether to fix center of mass (adjusts gradient vectors only)");
    Connect(flags.relax_atom, "relax_atom", "Whether to relax atomic positions (can be masked)");
    Connect(flags.relax_cell, "relax_cell", "Whether to relax cell parameters (can be masked)");
    Connect(flags.fix_vol, "fix_vol", "Whether to fix volume (conflicts with CONSTANT_VOLUME constraint)");
    Connect(flags.fix_shape, "fix_shape", "Whether to fix cell shape while relaxing volume");
    Connect(flags.relax_cell_use_strain, "cell_strain", "Whether to use strain-based cell relaxation (recommended), default true");
    Connect(flags.symm_strain, "symm_strain", "Whether to use symmetric strain to prevent rotation, default true");
    Connect(flags.atomic_force_recast, "atomic_force_recast", "Whether to recast atomic forces during strain relaxation, default true", 1, 2);
    Connect(flags.selective_dynamics, "selective_dynamics", "Whether to use selective dynamics filter");
    Connect(flags.freeze_small_grad, "freeze_small_gd", "Whether to zero gradients below threshold");
    Connect(flags.apply_disp, "apply_disp", "Whether to apply random initial displacements");
    Connect(flags.symm_force, "symm_force", "Symmetrize force and stress according to space group symmetry");
    Connect(flags.keep_symm, "keep_symm", "Exact symmetry-constrained relaxation using invariant cell and atomic modes. "
        "Detects space group and atom mappings with spglib, retaining coupled site displacements. "
        "Supports cell-only, atom-only, or joint relaxation and both atomic_force_recast modes. "
        "Projects the raw gradient without enabling symm_force; couple_flag is disabled. "
        "Incompatible with other structural constraints, gradient filters, Cartesian coordinates. "
        "Cell relaxation requires symmetric strain, Cholesky restoration and its exact gradient.");
    Connect(control.symmetry_operations, "symmetry_operations", "Comma-separated initial symmetry operation indices; generated subgroup includes identity");
    Connect(control.symmetry_sites, "symmetry_sites", "Comma-separated atom indices; constrain only their selected stabilizers and generated invariant cell modes");
    Connect(control.rigid_groups, "rigid_groups", "Fixed-cell rigid groups: zero-based atom indices separated by commas; separate groups by semicolons");
    // control
    Connect(control.driver, "driver", "elaxation algorithm to use:\n"
        "\t\t- 'cg': Conjugate Gradient\n"
        "\t\t- 'lbfgspp': Limited-memory BFGS, recommended\n"
        "\t\t- 'lbfgs': Limited-memory BFGS, recommended in case lbfgspp failed\n"
        "\t\t- 'gd': Gradient Descent\n"
        "\t\t- 'fire': FIRE algorithm\n"
        "\t\t- 'mdmin': Damped molecular-dynamics minimization\n"
        "\t\t- 'quickmin': Velocity-projection minimization\n"
        "\t\t- 'sqp', 'nlc': Constrained optimization (modern SQP)");
    Connect(control.max_iter, "maxiteration", "Maximum number of optimization iterations");
    Connect(control.constraint_iter_limit, "constraint_loops", "Maximum outer iterations when applying constraints", 1, 2);
    Connect(control.max_move, "stepmax", "Maximum allowable step size per variable update");
    Connect(control.cell_factor, "cell_factor", "Scaling factor for stress preconditioning, default 1.0", 1, 2);
    Connect(control.pressure, "pressure", "External hydrostatic pressure (in GPa)");
    Connect(control.disp, "disp", "Magnitude of random displacement when apply_disp=True");
    Connect(control.external_stress_str, "external_stress", "External stress tensor in GPa: 3 components (xx yy zz) or 6 components (xx yy zz yz xz xy)");
    Connect(control.stiffness_str, "stiffness", "Stiffness for strain modes, (6 components, in GPa)");
    Connect(control.lattice_mask_str, "lattice_mask", "Finite cell-selection flags: positive=movable, nonpositive=fixed. 6 values (xx yy zz yz xz xy) for symmetric strain, 9 row-major values for raw lattice or nonsymmetric strain; enables selective dynamics");
    Connect(control.lattice_mask_str, "strain_mask", "Finite cell-selection flags: positive=movable, nonpositive=fixed. 6 values (xx yy zz yz xz xy) for symmetric strain, 9 row-major values for raw lattice or nonsymmetric strain; enables selective dynamics");
    Connect(control.trajfile, "trajfile", "Output trajectory filename");
    Connect(control.indexfile, "index", "Selective dynamics index filename");
    Connect(control.lc_file, "lc_file", "Linear constraints definition file");
    Connect(control.lf_file, "lf_file", "Linear transforms definition file");
    Connect(control.vib_file, "vib_file", "Vibrational normal modes definition file");
    Connect(control.seldyn_file, "seldyn_file", "Full optimizer-coordinate mask: n followed by exactly n binary 0/1 flags (0=fixed). Used when selective_dynamics=True and the file exists; missing files retain atomic/cell masks. Cell coordinates precede atomic coordinates. Incompatible with keep_symm and VIB/VIB2/LC/LF.");
    Connect(control.hessian_precond_file, "hessian_precond_file", "Full-coordinate Hessian: n followed by row-major n*n symmetric entries, before LF/LC/masks; spectral-absolute positive metric and whitening");
    Connect(control.sparse_precond, "sparse_precond", "Neighbor graph metric with PCG; fixed-cell Cartesian GD/FIRE, optional Cartesian masks");
    Connect(control.sparse_shift, "sparse_shift", "Positive sparse graph diagonal shift");
    Connect(control.precond_floor, "precond_floor", "Positive absolute eigenvalue floor for full Hessian metric");
    Connect(control.fc_update_interval, "fc_update_interval", "Recompute native atomic curvature after this many accepted steps; 0 disables");
    Connect(control.aul_auto, "aul_auto", "Estimate AUL weights from geometry Jacobian and gradient; adapt using residual improvement");
    Connect(control.thin_cell_scaling, "thin_cell_scaling", "Cell preconditioning from periodic Gram measure using unit open thickness, independent of padding vacuum");
    Connect(control.precond_file, "precond_file", "Diagonal optimizer scaling file: n then exactly n finite positive values in full coordinates before masks/LF/LC; overrides dprecond/stiffness. FIRE/GD use gradient divisors, ALGLIB lbfgs/cg use variable scales. Not a full Hessian. Incompatible with lbfgspp, keep_symm, VIB/VIB2 and force_switchoff_scaling", 1, 2);
    Connect(control.nblock, "nblock", "Output interval (write every n steps)");
    Connect(control.couple_flag, "couple", "Lattice vector coupling mode:\n"
        "\t\t- 0: No coupling\n"
        "\t\t- 1: Couple a-b vectors\n"
        "\t\t- 2: Couple b-c vectors\n"
        "\t\t- 3: Couple c-a vectors\n"
        "\t\t- 4: Couple all vectors, in this case use fix_shape is recommended");
    // tolerance
    Connect(tol.gtol0, "gtol0", "Initial AUL inner gradient tolerance (at least gtol)", 1, 2);
    Connect(tol.gtol, "gtol", "Gradient tolerance - stops optimization when max gradient component falls below this value");
    Connect(tol.ftol, "ftol", "Energy tolerance - stops optimization when the energy change between steps falls below this value");
    Connect(tol.xtol, "xtol", "Step tolerance - stops optimization when atomic displacements or lattice changes become smaller than this threshold");
    Connect(tol.ztol, "ztol", "Zero threshold - used for freezing small gradients (`freeze_small_gd`) or other numerical checks requiring a \"zero\" threshold", 1, 2);
    Connect(tol.eps,  "eps",  "Machine epsilon - prevents division by zero", 1, 2);
    // gd related
    Connect(buffer_gd.alpha, "alpha", "gd parameter", 1, 2);
    Connect(buffer_gd.max_move_init, "gd:step_max_init_cell", "initial step_max setting, note that this is not not step_max_cell", 1, 2);
    Connect(buffer_gd.step_upscale, "gd:step_upscale", "factor to upscale the max step size", 1, 2);
    Connect(buffer_gd.step_downscale, "gd:step_downscale", "factor to downscale the max step", 1, 2);
    Connect(mdmin_dt, "mdmin:dt", "Positive timestep for the damped MD minimizer", 1, 2);
    Connect(quickmin_dt, "quickmin:dt", "Positive timestep for QuickMin", 1, 2);
    // lbfgs related
    Connect(buffer_newton.bfgs_m, "bfgs_m", "bfgs parameter", 1, 2);
    Connect(buffer_newton.bfgs_m, "lbfgs:bfgs_m", "", 1, 2);
    Connect(buffer_newton.bfgs_m, "lbfgspp:bfgs_m", "", 1, 2);
    // FIRE parameters
    Connect(buffer_fire.ts_max, "fire:ts_max", "fire parameter", 1, 2);
    Connect(buffer_fire.ts_min, "fire:ts_min", "fire parameter", 1, 2);
    Connect(buffer_fire.ts_init, "fire:ts_init", "fire parameter", 1, 2);
    Connect(buffer_fire.mdint, "fire:mdalgo", "FIRE integration mode; only 0 is implemented", 1, 2);
    Connect(buffer_fire.alpha_start, "fire:alpha_start", "fire parameter", 1, 2);
    Connect(buffer_fire.alpha_reduce, "fire:alpha_reduce", "fire parameter", 1, 2);
    Connect(buffer_fire.ts_grow, "fire:ts_grow", "fire parameter", 1, 2);
    Connect(buffer_fire.ts_reduce, "fire:ts_reduce", "fire parameter", 1, 2);
    Connect(buffer_fire.vdfitmax, "fire:vdfmax", "fire parameter", 1, 2);
    Connect(buffer_fire.halvstepback, "fire:do_backward", "fire parameter", 1, 2);
    Connect(buffer_fire.initaildeyal, "fire:initial_delay", "fire parameter", 1, 2);
    Connect(buffer_fire.ts_backward_ratio, "fire:ts_backratio", "fire parameter", 1, 2);
    Connect(buffer_fire.ndelay, "fire:ndelay", "fire parameter", 1, 2);
    Connect(buffer_fire.downhill_check, "fire:downhill_check", "Accept only non-increasing full objectives; replaces downhill half-step backtracking", 1, 2);
    Connect(first_step, "fire:first_step", "Initial displacement cap; nonpositive uses 0.1*max_move, capped at max_move", 1, 2);
    Connect(first_step_delay, "fire:first_delay", "Number of initial trials using the first_step cap", 1, 2);
    // lbfgspp
    Connect(lbfgspp_param.epsilon, "lbfgspp:epsilon", "", 1, 2);
    Connect(lbfgspp_param.epsilon_rel, "lbfgspp:epsilon_rel", "", 1, 2);
    Connect(lbfgspp_param.past, "lbfgspp:past", "", 1, 2);
    Connect(lbfgspp_param.delta, "lbfgspp:delta", "", 1, 2);
    Connect(lbfgspp_param.max_iterations, "lbfgspp:max_iterations", "", 1, 2);
    Connect(lbfgspp_param.linesearch, "lbfgspp:linesearch", "1 Armijo, 2 Wolfe, 3 strong Wolfe, 4 force-only, 5 Lewis-Overton", 1, 2);
    Connect(lbfgspp_param.max_linesearch, "lbfgspp:max_linesearch", "", 1, 2);
    Connect(lbfgspp_param.min_step, "lbfgspp:min_step", "", 1, 2);
    Connect(lbfgspp_param.max_step, "lbfgspp:max_step", "", 1, 2);
    Connect(lbfgspp_param.ftol, "lbfgspp:ftol", "", 1, 2);
    Connect(lbfgspp_param.wolfe, "lbfgspp:wolfe", "", 1, 2);
    Connect(lbfgspp_param.initial_step, "lbfgspp:initial_step", "", 1, 2);
    Connect(lbfgspp_param.damping, "lbfgspp:damping", "", 1, 2);
    Connect(lbfgspp_param.first_initial_step, "lbfgspp:first_initial_step", "", 1, 2);
    Connect(lbfgspp_param.first_initial_step, "lbfgspp:first_step", "", 1, 2);
    Connect(first_step, "first_step", "initial trial step size", 1, 2);
}

RelaxationDrvV2::RelaxationDrvV2(const GPSettings& _settings)
{
    InitSettings();
    ApplySettings(_settings);

    if (flags.fix_shape) {
        if (flags.relax_cell) {
            flags.relax_cell = false;
            Warning("relax_cell change to false due to fix_shape constraint");
        }
    }
    // Preserve early stiffness syntax errors without caching coordinate-dependent scales.
    if (flags.dprecond && flags.relax_cell && !flags.cartesian && flags.symm_strain &&
        !flags.constraint_vib_modes && !control.stiffness_str.empty())
        (void)strain_stiffness(control.stiffness_str, control.cell_factor);
    lbfgspp_param.epsgmax = optimizer_gtol();
    lbfgspp_param.max_step_move = control.max_move;
    lbfgspp_param.m = buffer_newton.bfgs_m;

    if (!control.lattice_mask_str.empty())
        configure_cell_mask(control.lattice_mask_str);
    // Parse external stress for the cell-gradient and strain-energy contributions.
    if (!control.external_stress_str.empty()) {
        std::vector<std::string> vstr;
        str_split(control.external_stress_str, vstr);
        if (vstr.size() != 6 && vstr.size() != 3) {
            throw std::runtime_error("External stress requires 3 or 6 components, got " + std::to_string(vstr.size()));
        }
        // xx yy zz yz xz xy
        external_stress(0, 0) = std::stod(vstr[0]);
        external_stress(1, 1) = std::stod(vstr[1]);
        external_stress(2, 2) = std::stod(vstr[2]);
        if (vstr.size() == 6) {
            external_stress(1, 2) = std::stod(vstr[3]);
            external_stress(2, 1) = std::stod(vstr[3]);
            external_stress(0, 2) = std::stod(vstr[4]);
            external_stress(2, 0) = std::stod(vstr[4]);
            external_stress(0, 1) = std::stod(vstr[5]);
            external_stress(1, 0) = std::stod(vstr[5]);
        }
        // negative means positive values give negative internal stress
        external_stress /= -eVA3_to_GPa;
        // pressure_external_stress = external_stress.diagonal().sum() / 3.0;
    }
    // convert to energy unit
    control.pressure /= eVA3_to_GPa;

// PrintSettings();
}

void RelaxationDrvV2::configure_cell_mask(const std::string &text)
{
    std::vector<std::string> tokens;
    str_split(text, tokens);
    if (tokens.size() != 6 && tokens.size() != 9)
        throw std::invalid_argument("Cell mask requires 6 or 9 finite components");
    std::vector<double> values;
    values.reserve(tokens.size());
    for (const auto &token : tokens) {
        size_t parsed = 0;
        double value;
        try { value = std::stod(token, &parsed); }
        catch (const std::exception &) {
            throw std::invalid_argument("Invalid cell mask value: " + token);
        }
        if (parsed != token.size() || !std::isfinite(value))
            throw std::invalid_argument("Invalid cell mask value: " + token);
        values.push_back(value > 0.0 ? 1.0 : 0.0);
    }
    // No state changes until every value has been validated.
    control.lattice_mask_str = text;
    cellmask_set_flag = values.size() == 9;
    cellmask.setOnes();
    symm_strain_mask.fill(1.0);
    if (cellmask_set_flag) {
        for (int i = 0; i < 9; ++i) cellmask(i / 3, i % 3) = values[i];
    } else {
        constexpr int order[] = {0, 3, 5, 4, 2, 1};
        for (int i = 0; i < 6; ++i) symm_strain_mask[order[i]] = values[i];
    }
    flags.selective_dynamics = true;
}

namespace {
// Fixed-cell recast atomic coordinates are Cartesian. Avoid the cancelling
// cell/inverse pair for molecules and slabs with a missing open cell direction.
bool cell_free_recast_coordinates(const RelaxationDrvV2 &driver) {
    return !driver.atom.is_fully_periodic() && !driver.flags.relax_cell &&
        !driver.flags.fix_shape && driver.flags.atomic_force_recast &&
        !driver.refcell.fullPivLu().isInvertible();
}

bool lattice_regulator(const regulator &reg) {
    return reg.regulator_t >= regulator::CONSTANT_VOLUME && reg.regulator_t <= regulator::COUPLE_VECTOR;
}
void validate_regulator_values(const regulator &reg, bool runtime) {
    if (reg.type != 0 && reg.type != 1 && reg.type != -1 && reg.type != 10)
        throw std::invalid_argument("Unsupported constraint type; AUL equalities use type=10");
    if (!std::isfinite(reg.tol) || reg.tol <= 0 || !std::isfinite(reg.g0) ||
        !std::isfinite(reg.double_target) || !std::isfinite(reg.lambda_init) ||
        !std::isfinite(reg.weight_init) || reg.weight_init <= 0 ||
        !std::isfinite(reg.weight_step) || !std::isfinite(reg.weight_multply))
        throw std::invalid_argument("Invalid constraint tolerance, target, multiplier, or weight");
    for (double value : reg.double_specs)
        if (!std::isfinite(value)) throw std::invalid_argument("Non-finite constraint specification");
    if (runtime && (!std::isfinite(reg.weight) || reg.weight <= 0 || !std::isfinite(reg.lambda) ||
        !std::isfinite(reg.mos) || !std::isfinite(reg.mos2)))
        throw std::invalid_argument("Non-finite or nonpositive constraint multiplier/weight state");
}
void validate_regulator_layout(const regulator &reg, Eigen::Index nx, Eigen::Index ng) {
    if ((!reg.compute && reg.regulator_t != regulator::DISTANCE && reg.regulator_t != regulator::ANGLE && reg.regulator_t != regulator::FORMULA) || reg.x.size() != reg.inputs.size() || reg.g.size() != reg.outputs.size())
        throw std::invalid_argument("Constraint callback or buffer dimensions are invalid");
    for (int index : reg.inputs)
        if (index < 0 || index >= nx) throw std::invalid_argument("Constraint input index out of bounds");
    for (int index : reg.outputs)
        if (index < 0 || index >= ng) throw std::invalid_argument("Constraint output index out of bounds");
}
}

void RelaxationDrvV2::validate_fire_options() const
{
    const auto &b = buffer_fire;
    if (b.mdint != 0)
        throw std::invalid_argument("V2 FIRE supports only fire:mdalgo=0");
    if (!std::isfinite(b.ts_min) || !std::isfinite(b.ts_init) || !std::isfinite(b.ts_max) ||
        !(0 < b.ts_min && 0 < b.ts_init && b.ts_init <= b.ts_max) ||
        !std::isfinite(b.ts_reduce) || !(0 < b.ts_reduce && b.ts_reduce < 1) ||
        !std::isfinite(b.ts_grow) || b.ts_grow < 1 ||
        !std::isfinite(b.alpha_start) || b.alpha_start < 0 || b.alpha_start > 1 ||
        !std::isfinite(b.alpha_reduce) || !(0 < b.alpha_reduce && b.alpha_reduce <= 1) ||
        !std::isfinite(b.ts_backward_ratio) || b.ts_backward_ratio < 0 || b.ndelay < 0)
        throw std::invalid_argument("Invalid FIRE timestep, mixing, or restart parameters");
}

void RelaxationDrvV2::validate_options()
{
    if(!control.symmetry_operations.empty() || !control.symmetry_sites.empty())flags.keep_symm=true;
    if((!control.hessian_precond_file.empty() && !control.precond_file.empty()) ||
        (control.sparse_precond && (!control.hessian_precond_file.empty() || !control.precond_file.empty() || control.fc_update_interval)))
        throw std::invalid_argument("Diagonal, full-Hessian and sparse preconditioners are mutually exclusive");
    if(!std::isfinite(control.precond_floor) || control.precond_floor<=0 || !std::isfinite(control.sparse_shift) || control.sparse_shift<=0 || control.fc_update_interval<0)
        throw std::invalid_argument("Invalid preconditioner floor, shift or FC update interval");
    if((!control.hessian_precond_file.empty() || control.fc_update_interval) &&
        (flags.vibffilter || flags.constraint_vib_modes || flags.fix_cm || flags.symm_force || flags.freeze_small_grad || !inplace_modifiers_init))
        throw std::invalid_argument("Full Hessian preconditioning requires conservative coordinate filters");
    if(control.sparse_precond && (flags.periodic_subspace || flags.relax_cell || flags.fix_shape || (!flags.cartesian && !flags.atomic_force_recast) ||
        flags.keep_symm || flags.vibffilter || flags.constraint_vib_modes || flags.linear_transform || flags.linear_constraint || nonlinear_coordinates_active() ||
        (control.driver!="gd" && control.driver!="fire")))
        throw std::invalid_argument("Sparse preconditioning supports fixed Cartesian GD/FIRE with Cartesian masks only");
    if(control.fc_update_interval && (flags.relax_cell || flags.fix_shape || nonlinear_coordinates_active() || flags.keep_symm || constrained_solver() || flags.with_constraint_aul || (control.driver!="lbfgs" && control.driver!="cg" && control.driver!="gd" && control.driver!="fire")))
        throw std::invalid_argument("Online native FC supports fixed-cell conservative optimization without nonlinear/symmetry or structural constraints");
    if (control.nblock <= 0)
        throw std::invalid_argument("Relaxation nblock must be positive");
    if (flags.periodic_subspace && (flags.cartesian || flags.fix_shape || flags.fix_vol || control.couple_flag || flags.keep_symm || flags.symm_force || control.pressure!=0 || !external_stress.isZero(0) || !control.lattice_mask_str.empty()))
        throw std::invalid_argument("periodic_subspace requires mixed coordinates with fixed open rows, no 3D pressure/volume/symmetry/coupling or lattice_mask; use a full-coordinate selection file");
    if (nonlinear_coordinates_active() && (flags.keep_symm || flags.vibffilter || flags.constraint_vib_modes || flags.fix_cm || flags.freeze_small_grad))
        throw std::invalid_argument("Nonlinear coordinates require conservative coordinate filters without keep_symm/VIB");
    if (!control.rigid_groups.empty() && (flags.relax_cell || flags.fix_shape || flags.linear_transform || flags.linear_constraint || flags.selective_dynamics || (!flags.cartesian && !flags.atomic_force_recast)))
        throw std::invalid_argument("Rigid groups require fixed-cell Cartesian/recast coordinates without masks/LF/LC");
    if ((!control.rigid_groups.empty() || dynamic_cast<RigidBodyCoordinates *>(coordinate_map.get())) &&
        control.driver!="lbfgs" && control.driver!="cg" && control.driver!="gd" && control.driver!="fire")
        throw std::invalid_argument("Rigid chart updates currently support lbfgs/cg/gd/fire");
    if (constrained_solver()) {
        if(control.driver!="sqp" && control.driver!="nlc")control.driver=flags.with_constraint_sqp?"sqp":"nlc";
        flags.with_constraint_aul = false;
        if (flags.freeze_small_grad || flags.fix_cm || flags.symm_force || flags.vibffilter)
            throw std::invalid_argument("SQP/NLC requires conservative coordinate gradients");
    }
    if (!control.precond_file.empty() &&
        (control.driver == "lbfgspp" || flags.keep_symm || flags.vibffilter ||
         flags.constraint_vib_modes || flags.force_switchoff_scaling || !inplace_modifiers_init))
        throw std::invalid_argument("precond_file is incompatible with lbfgspp, keep_symm, VIB/VIB2, force_switchoff_scaling or external modifiers");
    if (control.driver == "quickmin" && (!std::isfinite(quickmin_dt) || quickmin_dt <= 0))
        throw std::invalid_argument("QuickMin requires a finite positive quickmin:dt");
    if (!std::isfinite(tol.gtol) || tol.gtol < 0 || !std::isfinite(tol.ftol) || tol.ftol < 0 ||
        !std::isfinite(tol.xtol) || tol.xtol < 0 || !std::isfinite(tol.eps) || tol.eps <= 0)
        throw std::invalid_argument("Invalid relaxation tolerances");
    if (flags.with_constraint_aul && (control.constraint_iter_limit <= 0 ||
        !std::isfinite(tol.gtol0) || tol.gtol0 < tol.gtol))
        throw std::invalid_argument("AUL requires positive constraint_loops and finite gtol0 >= gtol");
    if (control.driver == "fire") validate_fire_options();
    if (control.driver == "mdmin" && (!std::isfinite(mdmin_dt) || mdmin_dt <= 0))
        throw std::invalid_argument("MDMin requires a finite positive mdmin:dt");
    if (!control.lattice_mask_str.empty()) {
        // Check the final mode, including any ISIF or programmatic overrides.
        std::vector<std::string> tokens;
        str_split(control.lattice_mask_str, tokens);
        if (!flags.relax_cell || flags.fix_shape)
            throw std::invalid_argument("Cell mask requires cell relaxation without fix_shape");
        const int expected = flags.relax_cell_use_strain &&
            (flags.symm_strain) ? 6 : 9;
        if (tokens.size() != static_cast<size_t>(expected))
            throw std::invalid_argument("Cell mask requires " + std::to_string(expected) +
                                        " components for the final cell coordinate mode");
        if (flags.keep_symm || flags.vibffilter || flags.constraint_vib_modes ||
            flags.linear_constraint || flags.linear_transform)
            throw std::invalid_argument("Cell mask is incompatible with keep_symm and VIB/VIB2/LC/LF modifiers");
        if (!control.seldyn_file.empty() && GPIO::file_exist(control.seldyn_file))
            throw std::invalid_argument("Cell mask is incompatible with an active optimization-coordinate mask file");
        configure_cell_mask(control.lattice_mask_str);
    }
}

int RelaxationDrvV2::coordinate_offset(const Atoms &source) const {
    if(flags.periodic_subspace && flags.relax_cell) {
        const int rank=int(source.pbc_[0])+int(source.pbc_[1])+int(source.pbc_[2]);return rank*(rank+1)/2;
    }
    return flags.fix_shape?1:!flags.relax_cell?0:flags.relax_cell_use_strain && (flags.symm_strain)?6:9;
}

void RelaxationDrvV2::check_preconditioner(const Atoms &source) const
{
    const int offset = coordinate_offset(source);
    (void)read_diagonal_scaling(control.precond_file,
        offset + (flags.relax_atom ? 3 * source.size() : 0));
    if(!control.hessian_precond_file.empty())
        (void)RelaxPreconditioning::read_full_hessian(control.hessian_precond_file,offset+(flags.relax_atom?3*source.size():0));
}

std::vector<regulator> RelaxationDrvV2::prepare_constraints(const Atoms &source, const std::vector<regulator> &requested)
{
    validate_options();
    Atoms _atom = source;
    const int full_offset = coordinate_offset(_atom);
    const int nl = full_offset + (flags.relax_atom ? _atom.size() * 3 : 0);
    (void)read_diagonal_scaling(control.precond_file, nl);
    if(!control.hessian_precond_file.empty())(void)RelaxPreconditioning::read_full_hessian(control.hessian_precond_file,nl);
    auto prepared_constraints = requested;
    for (auto &reg : prepared_constraints) {
        validate_regulator_values(reg, false);
        if (lattice_regulator(reg)) {
            if (!flags.relax_cell || flags.relax_cell_use_strain || flags.fix_shape)
                throw std::invalid_argument("Lattice constraints require raw9 cell coordinates (relax_cell=true, cell_strain=false, fix_shape=false)");
            if (!_atom.cell_.allFinite() || !_atom.is_fully_periodic() || !std::isfinite(_atom.cell_.determinant()) || _atom.cell_.determinant() <= 1e-14)
                throw std::invalid_argument("Lattice constraints require a fully periodic right-handed cell");
            if (flags.fix_vol || control.couple_flag || flags.keep_symm || flags.symm_force ||
                flags.vibffilter || flags.constraint_vib_modes || flags.linear_constraint || flags.linear_transform ||
                flags.selective_dynamics || flags.freeze_small_grad || flags.fix_cm || flags.dprecond)
                throw std::invalid_argument("Raw9 lattice constraints are incompatible with cell/gradient projections, masks, and dprecond");
            if (flags.cartesian || control.pressure != 0 || !external_stress.isZero(0))
                throw std::invalid_argument("Raw9 lattice constraints require cartesian=false and zero external pressure/stress");
        }
        const auto require_axes = [&](size_t count) {
            if (reg.integer_specs.size() != count)
                throw std::invalid_argument("Wrong number of lattice constraint axes");
            for (int axis : reg.integer_specs)
                if (axis < 0 || axis > 2) throw std::invalid_argument("Lattice constraint axis must be 0, 1, or 2");
            if (count == 2 && reg.integer_specs[0] == reg.integer_specs[1])
                throw std::invalid_argument("Lattice constraint axes must differ");
        };
        switch (reg.regulator_t) {
        case regulator::NONE: break; // Checked custom callback and full-coordinate indices below.
        case regulator::CONSTANT_VOLUME:
            require_axes(0); Constraints::constant_volume_setup(reg, _atom); break;
        case regulator::CONSTANT_VECTOR:
            require_axes(1); Constraints::constant_vector_setup(reg, _atom, reg.integer_specs[0]); break;
        case regulator::CONSTANT_ANGLE:
            require_axes(2); Constraints::constant_angle_setup(reg, _atom, reg.integer_specs[0], reg.integer_specs[1]); break;
        case regulator::COUPLE_VECTOR:
            require_axes(2); Constraints::couple_vector_setup(reg, _atom, reg.integer_specs[0], reg.integer_specs[1]); break;
        case regulator::DISTANCE:
        case regulator::ANGLE:
        case regulator::FORMULA:
            if (flags.keep_symm || flags.symm_force || flags.vibffilter || flags.constraint_vib_modes ||
                flags.freeze_small_grad || flags.fix_cm)
                throw std::invalid_argument("Atomic geometry constraints are incompatible with symmetry, VIB/LC/LF and non-coordinate gradient projections");
            if (flags.fix_shape || (flags.relax_cell && (flags.fix_vol || control.couple_flag)))
                throw std::invalid_argument("Atomic geometry constraints do not support fix_shape, fix_vol or coupled cell coordinates");
            if (flags.relax_cell && flags.dprecond && !flags.atomic_force_recast && !flags.periodic_subspace)
                throw std::invalid_argument("Atomic geometry constraints with variable cell require dprecond=false or atomic_force_recast=true");
            if (reg.regulator_t == regulator::FORMULA) {
                if (!reg.expression) throw std::invalid_argument("Unbound constraint expression");
                if (flags.periodic_subspace && reg.expression->uses_volume())
                    throw std::invalid_argument("Periodic-subspace relaxation does not define a three-dimensional volume constraint");
                if (reg.bounded && !constrained_solver())
                    throw std::invalid_argument("V2 bounds require driver=sqp or nlc");
            } else Constraints::atomic_geometry_setup(reg, _atom);
            reg.inputs.resize(nl);
            std::iota(reg.inputs.begin(), reg.inputs.end(), 0);
            reg.outputs = reg.inputs;
            reg.x.resize(nl); reg.g.resize(nl);
            break;
        case regulator::MINDIST:
            if (!flags.relax_atom || flags.relax_cell || flags.fix_shape || (!flags.cartesian && flags.atomic_force_recast) ||
                !std::isfinite(reg.g0) || reg.g0 <= 0 || reg.integer_specs.empty())
                throw std::invalid_argument("MINDIST requires fixed-cell Cartesian or unrecast fractional atomic coordinates and a positive distance");
            for (int index : reg.integer_specs)
                if (index < 0 || index >= _atom.size()) throw std::invalid_argument("MINDIST atom index out of bounds");
            if (flags.cartesian) Constraints::mindist_f_setup(reg, _atom, reg.g0, full_offset);
            else Constraints::mindist_setup(reg, _atom, reg.g0, full_offset);
            break;
        default: throw std::invalid_argument("Requested built-in constraint is not implemented");
        }
        reg.lambda = reg.lambda_init;
        reg.weight = reg.weight_init;
        reg.mos = reg.lambda / reg.weight;
        reg.mos2 = reg.mos * reg.mos;
        reg.f = 0;
        reg.update = true;
        reg.curr_constraint_iter = 0;
        validate_regulator_values(reg, true);
        validate_regulator_layout(reg, nl, nl);
    }
    return prepared_constraints;
}

void RelaxationDrvV2::init(Potentials::AnyPotential* _pot, Atoms& _atom, std::vector<regulator>* _p_aug_constraints)
{
    // A failed reinitialization must not retain a previous successful status.
    initialized = false;
    reset_state();
    validate_options();
    subspace_active=flags.periodic_subspace;
    if(subspace_active) {
        if(_atom.is_fully_periodic())throw std::invalid_argument("periodic_subspace is for rank-deficient periodicity");
        subspace.init(_atom,flags.atomic_force_recast);
    }
    const bool cell_free = !_atom.is_fully_periodic() && !flags.relax_cell && !flags.fix_shape &&
        flags.atomic_force_recast && _atom.cell_.allFinite() && !_atom.cell_.fullPivLu().isInvertible();
    if (!_pot || _atom.chemmap_.size() != _atom.pos_.size() || !_atom.cell_.allFinite() ||
        (!_atom.is_nonperiodic() && !cell_free && !subspace_active &&
         (!std::isfinite(_atom.cell_.determinant()) || std::abs(_atom.cell_.determinant()) < 1e-14)))
        throw std::invalid_argument("Invalid initial relaxation geometry or species");
    for (const auto &p : _atom.pos_)
        if (!p.allFinite()) throw std::invalid_argument("Non-finite initial relaxation coordinates");
    if (cell_free) {
        if (!_atom.cartesian)
            throw std::invalid_argument("Cell-free relaxation requires Cartesian input positions");
        if (flags.keep_symm || flags.symm_force || flags.fix_vol)
            throw std::invalid_argument("Symmetry and fixed-volume relaxation require a nonsingular cell");
        (void)trajectory_periodic_coordinate_map(_atom);
    }
    const int full_offset = coordinate_offset(_atom);
    const int nl = full_offset + (flags.relax_atom ? _atom.size() * 3 : 0);
    auto file_scaling = read_diagonal_scaling(control.precond_file, nl);
    const bool use_mask_file = flags.selective_dynamics && !control.seldyn_file.empty() &&
        GPIO::file_exist(control.seldyn_file);
    std::vector<int> selected_coordinates;
    // Validate before replacing geometry, freeing existing modifiers, or
    // allowing another modifier to transform/displace the full coordinates.
    if (use_mask_file) {
        if (!control.lattice_mask_str.empty())
            throw std::invalid_argument("Cell mask is incompatible with an active optimization-coordinate mask file");
        if (flags.keep_symm || flags.vibffilter || flags.constraint_vib_modes ||
            flags.linear_constraint || flags.linear_transform)
            throw std::invalid_argument("Optimization-coordinate mask is incompatible with keep_symm and VIB/VIB2/LC/LF modifiers");
        std::ifstream input(control.seldyn_file);
        if (!input)
            throw std::invalid_argument("Cannot open optimization-coordinate mask file: " + control.seldyn_file);
        selected_coordinates = FilterSelDyn::read_mask(input, nl);
    }
    Atoms working_atom = _atom;
    if (!working_atom.cartesian) working_atom.to_cartesian();
    auto prepared_constraints = prepare_constraints(working_atom, _p_aug_constraints ? *_p_aug_constraints : std::vector<regulator>{});
    use_dprecond = flags.dprecond && flags.relax_cell && !flags.fix_shape && !flags.cartesian &&
        !flags.atomic_force_recast && !flags.constraint_vib_modes &&
        (subspace_active || (flags.relax_cell_use_strain && flags.symm_strain));
    atom = std::move(working_atom);
    vol0 = atom.get_volume();
    refcell = atom.cell_;
    if (subspace_active || cell_free_recast_coordinates(*this)) ivrefcell.setZero(); // unused in this atomic-only path
    else ivrefcell = refcell.inverse();
    pot = _pot;
    if(subspace_active) {
        subspace_fixed_positions.clear();
        for(const auto &pos:atom.pos_)subspace_fixed_positions.push_back(subspace.encode_position(pos,atom.cell_));
    }
    xposoffset = full_offset;
    if (flags.selective_dynamics) {
        if (cellmask_set_flag)
            atom.sel_dyncell_ = cellmask;
        if (atom.sel_dynpos_.size() == 0) {
            atom.sel_dynpos_.resize(atom.size(), Eigen::RowVector3d::Ones());
        }
    }
    constraints = std::move(prepared_constraints);
    x.resize(nl);
    g.setZero(nl);
    if (flags.fix_shape)
        scaling = 1;
    // scaling settings
    prec_scaling.resize(nl);
    prec_scaling.setConstant(1);
    if (!control.precond_file.empty()) {
        prec_scaling = std::move(file_scaling);
    } else {
        automatic_scaling(*this);
    }
    if (!control.rigid_groups.empty()) {
        std::vector<std::vector<int>> groups; std::istringstream input(control.rigid_groups);std::string block;
        while(std::getline(input,block,';')) {
            std::replace(block.begin(),block.end(),',',' ');std::istringstream ids(block);std::vector<int> group;int id;
            while(ids>>id)group.push_back(id);
            if(!ids.eof() || group.empty())throw std::invalid_argument("rigid_groups requires semicolon-separated nonempty integer lists");
            groups.push_back(std::move(group));
        }
        coordinate_map=std::make_shared<RigidBodyCoordinates>(atom,groups);
    }
    // INIT FILTERS
    LatPos2X();
    if (inplace_modifiers_init) {
        // check conflicts
        if (flags.vibffilter && flags.constraint_vib_modes) {
            throw std::runtime_error("VIB filter and constraint_vib_modes cannot be used together");
        }
        // also modify prec_scaling, for VIB2 filter, we need use frequency w^2 as scaling factor
        // order 1
        free_modifiers();
        modifiers_list.reserve(10);
        modifiers_list_carg.reserve(10);
        if (flags.fix_cm) {
            modifiers_list_carg.resize(modifiers_list_carg.size() + 1);
            modifiers_list_carg.back() = new FilterFixCM();
        }
        if (flags.symm_force && !flags.keep_symm) {
            modifiers_list_carg.resize(modifiers_list_carg.size() + 1);
            modifiers_list_carg.back() = new FilterSymmForceStress();
            modifiers_list_carg.back()->init(this, &tol.symprec);
        }
        // SymmCoord: symmetry-constrained relaxation (reduces DOFs to asymmetric unit)
        if (flags.keep_symm) {
            // Mutual exclusion checks
            if (flags.selective_dynamics) {
                throw std::runtime_error("keep_symm and selective_dynamics are mutually exclusive. "
                    "keep_symm already constrains cell and positions by symmetry.");
            }
            if (flags.with_constraint_aul || flags.with_constraint_sqp || flags.with_constraint_nlc || flags.with_constraint_lc ||
                flags.fix_vol || flags.fix_shape ||
                !constraints.empty()) {
                throw std::runtime_error("keep_symm is not compatible with fixed-volume/shape or "
                    "additional structural constraints");
            }
            if (flags.constraint_vib_modes || flags.vibffilter || flags.fix_cm || flags.freeze_small_grad) {
                throw std::runtime_error("keep_symm is not compatible with VIB2 coordinate transforms.");
            }
            if (flags.linear_constraint || flags.linear_transform) {
                throw std::runtime_error("keep_symm is not compatible with linear constraints/transforms.");
            }
            if (!flags.relax_cell && !flags.relax_atom) {
                throw std::runtime_error("keep_symm requires at least one of relax_cell or relax_atom.");
            }
            if (flags.relax_cell && (!flags.symm_strain || !flags.relax_cell_use_strain)) {
                throw std::runtime_error("keep_symm with relax_cell requires symm_strain and relax_cell_use_strain.");
            }
            if (flags.cartesian) {
                throw std::runtime_error("keep_symm requires fractional coordinate mode (cartesian=false). "
                    "The symmetry operations are defined in fractional space.");
            }
            // The reduced-coordinate Jacobian projects the raw gradient.
            // Pre-symmetrizing it here would apply the projection twice.
            // The symmetry basis already handles coupled strain components.
            control.couple_flag = 0;
            // Create the SymmCoord modifier in modifiers_list
            modifiers_list.resize(modifiers_list.size() + 1);
            modifiers_list.back() = new FilterSymmCoord();
            modifiers_list.back()->init(this, &tol.symprec);
            modifiers_list.back()->forward_scaling(this);
        }
        // order 2, VIB
        if (flags.vibffilter) {
            modifiers_list.resize(modifiers_list.size() + 1);
            modifiers_list.back() = new FilterVIB();
            modifiers_list.back()->init(this, &control.vib_file);
        }
        // order 3, VIB2
        if (flags.constraint_vib_modes) {
            modifiers_list.resize(modifiers_list.size() + 1);
            modifiers_list.back() = new FilterVIB2();
            modifiers_list.back()->init(this, &control.vib_file);
            modifiers_list.back()->forward_scaling(this);
        }
        // order 4, LC/LF. TODO: adjust these filters, this two filter may apper multiple times in any order
        if (flags.linear_transform) {
            modifiers_list.resize(modifiers_list.size() + 1);
            modifiers_list.back() = new FilterLF();
            modifiers_list.back()->init(this, &control.lf_file);
            modifiers_list.back()->forward_scaling(this);
        }
        if (flags.linear_constraint) {
            modifiers_list.resize(modifiers_list.size() + 1);
            modifiers_list.back() = new FilterLC();
            modifiers_list.back()->init(this, &control.lc_file);
            modifiers_list.back()->forward_scaling(this);
        }
        // order 5, selective dynamics
        if (flags.selective_dynamics) {
            modifiers_list.resize(modifiers_list.size() + 1);
            modifiers_list.back() = new FilterSelDyn();
            // now initialize
            modifiers_list.back()->init(this, use_mask_file ? &selected_coordinates : nullptr);
            modifiers_list.back()->forward_scaling(this);
        }
        if (coordinate_map) {
            const auto full_x=x, full_g=g;
            apply_modifiers(modifiers_list);
            auto *filter=new FilterNF();filter->map=coordinate_map;
            try { filter->init(this);filter->forward_scaling(this); }
            catch(...) {delete filter;throw;}
            modifiers_list.push_back(filter);x=full_x;g=full_g;
        }
        // order 6, fix_small_gd
        if (flags.freeze_small_grad) {
            modifiers_list.resize(modifiers_list.size() + 1);
            modifiers_list.back() = new FilterFixSmallGD();
        }
        if (flags.force_switchoff_scaling) {
            prec_scaling.setConstant(1);
        }
    }
    if (!control.precond_file.empty() &&
        (!prec_scaling.allFinite() || (prec_scaling.array() <= 0).any()))
        throw std::invalid_argument("precond_file produced invalid scaling after coordinate filters: " + control.precond_file);
    initialize_metrics();
    initialized = true;
}

void RelaxationDrvV2::set_vib_data(const Eigen::MatrixXd& projection_matrix,
                                   const Eigen::VectorXd& mode_factor)
{
    if (g.size() == 0) {
        throw std::runtime_error("set_vib_data() must be called after RelaxationDrvV2::init()");
    }
    if (flags.constraint_vib_modes) {
        throw std::runtime_error("set_vib_data() cannot be combined with constraint_vib_modes");
    }
    if (projection_matrix.rows() != projection_matrix.cols()) {
        throw std::runtime_error("set_vib_data() requires a square projection matrix");
    }
    if (projection_matrix.rows() != g.size()) {
        throw std::runtime_error("set_vib_data() projection matrix dimension does not match relaxation gradient size");
    }
    if (mode_factor.size() != g.size()) {
        throw std::runtime_error("set_vib_data() mode factor dimension does not match relaxation gradient size");
    }

    FilterVIB *vib_filter = nullptr;
    for (auto *modifier : modifiers_list) {
        vib_filter = dynamic_cast<FilterVIB*>(modifier);
        if (vib_filter) break;
    }
    if (!vib_filter) {
        vib_filter = new FilterVIB();
        modifiers_list.push_back(vib_filter);
    }

    flags.vibffilter = true;
    // V1 stores Q with eigenmodes as columns and applies Q * D * Q^T to column gradients.
    // FilterVIB applies P^T * D * P, so store P = Q^T for programmatic V1-compatible data.
    vib_filter->projection_matrix = projection_matrix.transpose();
    vib_filter->mode_factor = mode_factor;
    mode_displacement_init = Eigen::VectorXd::Zero(mode_factor.size());
}

bool RelaxationDrvV2::constrained_solver() const {
    return control.driver == "sqp" || control.driver == "nlc" ||
        flags.with_constraint_sqp || flags.with_constraint_nlc || flags.with_constraint_lc;
}

void RelaxationDrvV2::evaluate_regulator(regulator& reg)
{
    validate_regulator_values(reg, true);
    validate_regulator_layout(reg, x.size(), g.size());
    std::fill(reg.g.begin(), reg.g.end(), std::numeric_limits<double>::quiet_NaN());
    reg.f = std::numeric_limits<double>::quiet_NaN();
    for (int i = 0; i < reg.inputs.size(); ++i) {
        reg.x[i] = x[reg.inputs[i]];
    }
    if (reg.regulator_t == regulator::DISTANCE || reg.regulator_t == regulator::ANGLE || reg.regulator_t == regulator::FORMULA) {
        auto geometry = Constraints::AtomicGeometryEvaluation{};
        if (reg.regulator_t == regulator::FORMULA) {
            const auto value = reg.expression->evaluate(atom);
            geometry.residual = value.value - reg.g0;
            geometry.physical_value = value.value;
            geometry.cartesian_gradients = value.positions;
            geometry.explicit_cell_gradient = value.cell;
        } else geometry = Constraints::evaluate_atomic_geometry(reg, atom);
        reg.f = geometry.residual;
        std::fill(reg.g.begin(), reg.g.end(), 0.0);
        Eigen::Matrix3d cell_gradient = geometry.explicit_cell_gradient;
        const Eigen::Matrix3d position_map = subspace_active ? subspace.atomic_jacobian(atom.cell_).eval() : (flags.cartesian || cell_free_recast_coordinates(*this))
            ? Eigen::Matrix3d::Identity().eval() :
            (flags.atomic_force_recast ? (ivrefcell * atom.cell_).eval() : atom.cell_);
        for (size_t i = 0; i < reg.integer_specs.size(); ++i) {
            const int id = reg.integer_specs[i];
            const auto &cart_gradient = geometry.cartesian_gradients[i];
            if (flags.relax_atom) {
                const Eigen::RowVector3d derivative = cart_gradient * position_map.transpose();
                for (int a = 0; a < 3; ++a) reg.g[xposoffset + 3 * id + a] = derivative[a];
            }
            if (flags.relax_cell && !flags.cartesian) {
                const Eigen::RowVector3d fractional = atom.pos_[id] * (subspace_active ? trajectory_periodic_coordinate_map(atom).eval() : atom.cell_.inverse().eval());
                cell_gradient += fractional.transpose() * cart_gradient;
            }
        }
        if (flags.relax_cell) {
            if (subspace_active) {
                const auto derivatives=subspace.cell_derivatives(atom.cell_);
                for(int k=0;k<xposoffset;++k)reg.g[k]=(cell_gradient.array()*derivatives[k].array()).sum();
            } else if (!flags.relax_cell_use_strain) {
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) reg.g[3 * a + b] = cell_gradient(a, b);
            } else if (!flags.symm_strain) {
                const Eigen::Matrix3d derivative = refcell.transpose() * cell_gradient;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) reg.g[3 * a + b] = derivative(a, b);
            } else {
                // Differentiate H = H0 chol(I+X) in each independent symmetric direction.
                // This includes the exact Jacobian at finite strain, without stress conventions.
                const Eigen::Matrix3d lower = ivrefcell * atom.cell_;
                const Eigen::Matrix3d inverse = lower.inverse();
                int component = 0;
                for (int a = 0; a < 3; ++a) for (int b = a; b < 3; ++b) {
                    Eigen::Matrix3d direction = Eigen::Matrix3d::Zero();
                    direction(a, b) = direction(b, a) = 1;
                    const Eigen::Matrix3d transformed = inverse * direction * inverse.transpose();
                    Eigen::Matrix3d phi = transformed.triangularView<Eigen::Lower>();
                    phi.diagonal() *= 0.5;
                    const Eigen::Matrix3d dcell = refcell * lower * phi;
                    reg.g[component++] = (cell_gradient.array() * dcell.array()).sum();
                }
            }
        }
    } else {
        reg.compute(reg.x.data(), reg.x.size(), reg.g.data(), reg.g.size(), &reg.f,
            reg.g0, &atom, reg.integer_specs.data(), reg.data);
    }
    if (!std::isfinite(reg.f) || !std::all_of(reg.g.begin(), reg.g.end(), [](double value) { return std::isfinite(value); }))
        throw std::runtime_error("Constraint callback returned a non-finite or incomplete residual/gradient");
}

void RelaxationDrvV2::compute_regulator(regulator& reg)
{
    evaluate_regulator(reg);
    if (constrained_solver()) return;
    if (reg.type == 1) {
        reg.f = std::min(0.0, reg.f);
        f += reg.f * reg.f * reg.weight;
        for (int i = 0; i < reg.outputs.size(); ++i) {
            g[reg.outputs[i]] += 2.0 * reg.g[i] * reg.weight * reg.f;
        }
    } else if (reg.type == 0) {
        f += reg.f * reg.f * reg.weight;
        for (int i = 0; i < reg.outputs.size(); ++i) {
            g[reg.outputs[i]] += 2.0 * reg.g[i] * reg.weight * reg.f;
        }
    } else if (reg.type == -1) {
        reg.f = std::max(0.0, reg.f);
        f += reg.f * reg.f * reg.weight;
        for (int i = 0; i < reg.outputs.size(); ++i) {
            g[reg.outputs[i]] += 2.0 * reg.g[i] * reg.weight * reg.f;
        }
    } else if (reg.type == 10) {
        f += 0.5 * reg.weight * reg.f * reg.f + reg.lambda * reg.f;
        for (int i = 0; i < reg.outputs.size(); ++i) {
            g[reg.outputs[i]] += reg.g[i] * reg.weight * reg.f + reg.lambda * reg.g[i];
        }
    }
}

void RelaxationDrvV2::update_regs()
{
    auto updated = constraints;
    for (auto& it : updated) {
        validate_regulator_values(it, true);
        if (!std::isfinite(it.f)) throw std::runtime_error("Non-finite constraint residual");
        if (!it.update)
            continue;
        if(control.aul_auto && it.type==10) {
            const size_t index=&it-updated.data();
            it.lambda+=it.weight*it.f;
            if(aul_previous_residual.size()==static_cast<int>(updated.size()) && std::abs(it.f)>.5*aul_previous_residual[index])it.weight*=10;
            it.mos=it.lambda/it.weight;it.mos2=it.mos*it.mos;++it.curr_constraint_iter;
            aul_previous_residual[index]=std::abs(it.f);validate_regulator_values(it,true);continue;
        }
        if (it.type == 0 || it.type == 10 || it.type == 1) {
            it.lambda += it.weight * it.f;
            it.weight = it.weight_multply * it.weight + it.weight_step;
        } else if (it.type == 11) {
            it.lambda = std::max(it.lambda + it.weight * it.f, 0.);
        } else if (it.type == 9) {
            it.lambda = std::max(it.lambda - it.weight * it.f, 0.);
        } else if (it.type == -1) {
            it.lambda -= it.weight * it.f;
            it.weight = it.weight_multply * it.weight + it.weight_step;
        }
        it.mos = it.lambda / it.weight;
        it.mos2 = it.mos * it.mos;
        ++it.curr_constraint_iter;
        validate_regulator_values(it, true);
    }
    constraints = std::move(updated);
}

void RelaxationDrvV2::LatPos2X()
{
    if(subspace_active) {
        if(flags.relax_cell)x.head(xposoffset)=subspace.encode_cell(atom.cell_);
        if(flags.relax_atom)for(int i=0;i<atom.size();++i)x.segment<3>(xposoffset+3*i)=subspace.encode_position(atom.pos_[i],atom.cell_);
        return;
    }
    // cell
    if (flags.relax_cell) {
        if (flags.relax_cell_use_strain) {
            Eigen::Matrix3d matD = ivrefcell * atom.cell_;
            Eigen::Matrix3d strain = matD - Eigen::Matrix3d::Identity();
            if (flags.symm_strain) {
                strain = matD * matD.transpose() - Eigen::Matrix3d::Identity();
                int ct = 0;
                for (int a = 0; a < 3; ++a) {
                    for (int b = a; b < 3; ++b) {
                        x[ct] = strain(a, b);
                        ++ct;
                    }
                }
            } else {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        x[3 * a + b] = strain(a, b);
                    }
                }
            }
        } else {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    x[3 * a + b] = atom.cell_(a, b);
                }
            }
        }
    }
    if (flags.fix_shape) {
        x[0] = scaling;
    }
    // pos
    if (flags.relax_atom) {
        if (flags.cartesian || cell_free_recast_coordinates(*this)) {
            for (int i = 0; i < atom.size(); ++i) {
                for (int a = 0; a < 3; ++a) {
                    x[xposoffset + 3 * i + a] = atom.pos_[i](a);
                }
            }
        } else {
            Eigen::RowVector3d xfrac;
            for (int i = 0; i < atom.size(); ++i) {
                xfrac = atom.car2dir(atom.pos_[i]);
                if (flags.atomic_force_recast) {
                    xfrac *= refcell;
                }
                for (int a = 0; a < 3; ++a) {
                    x[xposoffset+3*i+a] = xfrac(a);
                }
            } 
        }
    }
}

void RelaxationDrvV2::X2LatPos()
{
    if(subspace_active) {
        if(flags.relax_cell)atom.cell_=subspace.decode_cell(x.head(xposoffset));
        for(int i=0;i<atom.size();++i) {
            const Eigen::RowVector3d q=flags.relax_atom?Eigen::RowVector3d(x.segment<3>(xposoffset+3*i)):subspace_fixed_positions[i];
            atom.pos_[i]=subspace.decode_position(q,atom.cell_);
        }
        return;
    }
    // TODO: check this, may be incorrect, especially else case
    // Enforce coupling constraint on x[] BEFORE converting to cell.
    // Mirror of the enforcement in LatPos2X.
    // if (flags.relax_cell && control.couple_flag > 0) {
    //     auto couple_x = [&](int i, int j) {
    //         double avg = (x[i] + x[j]) / 2.0;
    //         x[i] = avg; x[j] = avg;
    //     };
    //     if (flags.relax_cell_use_strain) {
    //         if (control.couple_flag == 1)      couple_x(0, 3);
    //         else if (control.couple_flag == 2) couple_x(3, 5);
    //         else if (control.couple_flag == 3) couple_x(5, 0);
    //         else if (control.couple_flag == 4) {
    //             double avg = (x[0] + x[3] + x[5]) / 3.0;
    //             x[0] = avg; x[3] = avg; x[5] = avg;
    //         }
    //     } else {
    //         if (control.couple_flag == 1)      couple_x(3*0+0, 3*1+1);
    //         else if (control.couple_flag == 2) couple_x(3*1+1, 3*2+2);
    //         else if (control.couple_flag == 3) couple_x(3*2+2, 3*0+0);
    //         else if (control.couple_flag == 4) {
    //             double avg = (x[0] + x[4] + x[8]) / 3.0;
    //             x[0] = avg; x[4] = avg; x[8] = avg;
    //         }
    //     }
    // }
    if (flags.relax_cell) {
        Eigen::Matrix3d cell;
        if (flags.relax_cell_use_strain) {
            Eigen::Matrix3d eps;
            if (flags.symm_strain) {
                eps << x[0] + 1., x[1], x[2], 
                       x[1], x[3] + 1., x[4],
                       x[2], x[4], x[5] + 1.;
                Eigen::Matrix3d L;
                {
                    Eigen::LLT<Eigen::Matrix3d> lltOfA(eps);
                    if (lltOfA.info() != Eigen::Success) {
                        throw std::runtime_error(
                            "Illegal trial step: symmetric strain metric is not positive definite");
                    } else {
                        L = lltOfA.matrixL();
                    }
                }
                cell = refcell * L;
            } else {
                eps << x[0] + 1., x[1], x[2], 
                       x[3], x[4] + 1., x[5],
                       x[6], x[7], x[8] + 1.;
                cell = refcell * eps;
            }
        } else {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    cell(a, b) = x[3 * a + b];
                }
            }
        }

        if (flags.relax_atom) {
            atom.cell_ = cell;
        } else if (!flags.cartesian) {
            atom.set_cell(cell);
        } else {
            // Cartesian cell-only relaxation keeps positions fixed while changing the cell.
            atom.cell_ = cell;
        }
    }
    // fix_shape overrides any cell change above, since it controls volume scaling
    if (flags.fix_shape) {
        scaling = x[0];
        Eigen::Matrix3d ncell = scaling * refcell;
        if (!flags.cartesian)
            atom.set_cell(ncell);
        else
            atom.cell_ = ncell;
    }
    if (flags.relax_atom) {
        if (flags.cartesian || cell_free_recast_coordinates(*this)) {
            for (int i = 0; i < atom.size(); ++i) {
                for (int a = 0; a < 3; ++a) {
                    atom.pos_[i](a) = x[xposoffset + 3 * i + a];
                }
            }
        } else {
            Eigen::RowVector3d vec;
            for (int i = 0; i < atom.size(); ++i) {
                for (int a = 0; a < 3; ++a) {
                    vec(a) = x[xposoffset+3*i+a];
                }
                if (flags.atomic_force_recast) {
                    atom.pos_[i] = atom.dir2car(vec * ivrefcell);
                } else {
                    atom.pos_[i] = atom.dir2car(vec);
                }
            }
        }
    }
    if (flags.fix_vol) {
        if (flags.selective_dynamics && flags.symm_strain && flags.relax_cell_use_strain) {
            bool rela = symm_strain_mask[0];
            bool relb = symm_strain_mask[3];
            bool relc = symm_strain_mask[5];
            int irel = symm_strain_mask[0] + symm_strain_mask[3] + symm_strain_mask[5];
            if (irel != 0) {
                Eigen::Matrix3d epsii = Eigen::Matrix3d::Zero();
                if (rela)
                    epsii(0, 0) = 1;
                if (relb)
                    epsii(1, 1) = 1;
                if (relc)
                    epsii(2, 2) = 1;
                double volnew = atom.get_volume();
                double scale = pow(vol0 / volnew, 1.0 / (double)irel) - 1.0;
                epsii *= scale;
                Eigen::Matrix3d cellnew = atom.cell_ * (Eigen::Matrix3d::Identity() + epsii);
                atom.set_cell(cellnew);
            } else {
                ; // volume should be constant in this case
            }
            // scale to volume by strain
            /*             e1 0 0
            a' = a * (I +  0 e2 0 )
                           0 0 e3
            vol' = vol0 * (1+e1) * (1+e2) * (1+e3)
            (1+e)^irel = vol'/vol
            */
            //    cell = refcell * (Eigen::Matrix3d::Identity() + eps);
        } else {
            atom.scale_to_volume(vol0);
        }
    }
}

void RelaxationDrvV2::eval()
{
    if (!x.allFinite()) throw std::runtime_error("Non-finite relaxation variables");
    reverse_modifiers(modifiers_list);
    // _DEBUG_OUTPUT_(g.transpose());
    X2LatPos();
    if (!atom.cell_.allFinite() || !std::isfinite(atom.cell_.determinant()) ||
        (!atom.is_nonperiodic() && !cell_free_recast_coordinates(*this) && !subspace_active &&
         std::abs(atom.cell_.determinant()) < 1e-14))
        throw std::runtime_error("Invalid trial cell");
    for (const auto &p : atom.pos_)
        if (!p.allFinite()) throw std::runtime_error("Non-finite trial coordinates");
    for (const auto &reg : constraints)
        if (lattice_regulator(reg) && atom.cell_.determinant() <= 1e-14)
            throw std::runtime_error("Raw9 lattice constraint trial requires a right-handed nonsingular cell");
    EFS2G();
    // _DEBUG_OUTPUT_(g.transpose());
    if (constrained_solver()) {
        const auto full_x = x, objective_gradient = g;
        constraint_jacobian.resize(constraints.size(), x.size());
        for (size_t i = 0; i < constraints.size(); ++i) {
            x = full_x; g = Eigen::VectorXd::Zero(full_x.size());
            const auto &reg = constraints[i];
            for (size_t j = 0; j < reg.outputs.size(); ++j) g[reg.outputs[j]] += reg.g[j];
            apply_modifiers(modifiers_list);
            if (i == 0) constraint_jacobian.resize(constraints.size(), g.size());
            constraint_jacobian.row(i) = g.transpose();
        }
        x = full_x; g = objective_gradient;
    }
    apply_modifiers(modifiers_list);
    // AL uncertainty check (external bridge, not in modifiers_list)
    if (al_modifier_) al_modifier_->forward(this);
    if (!std::isfinite(f) || !g.allFinite()) throw std::runtime_error("Non-finite objective or gradient");
}

void RelaxationDrvV2::EFS2G()
{
    pos_orig = atom.pos_;
    atom.wrap();
    pot->getEFSOMP(atom);
    if (!std::isfinite(atom.energy_) || !atom.stress_.allFinite() || atom.forces_.size() != atom.pos_.size())
        throw std::runtime_error("Non-finite or incomplete potential output");
    for (const auto &force : atom.forces_)
        if (!force.allFinite()) throw std::runtime_error("Non-finite potential force");
    apply_modifiers(modifiers_list_carg);
    atom.pos_ = pos_orig;
    g.setZero();
    double pscal = control.pressure * atom.get_volume();
    f = atom.energy_ + pscal;
    if(subspace_active) {
        if(flags.relax_cell)g.head(xposoffset)=subspace.cell_gradient(atom.cell_,atom.stress_);
        if(flags.relax_atom) {
            const auto jac=subspace.atomic_jacobian(atom.cell_);
            for(int i=0;i<atom.size();++i)g.segment<3>(xposoffset+3*i)=-jac*atom.forces_[i];
        }
        for(auto &reg:constraints)compute_regulator(reg);
        return;
    }
    // cell
    if (flags.relax_cell) {
        // -(L^-1^T) * S - (L^-1^T) * P = -(l^-1^T) * (S - PI)
        // dPV / dLij = P*(L^-1^T)_{ij}
        curr_stress = atom.stress_ - pscal * Eigen::Matrix3d::Identity();
        // if fix volume, subtract components that change the volume
        if (flags.fix_vol) {
            double press = curr_stress.diagonal().sum() / 3.0;
            if (flags.symm_strain && flags.relax_cell_use_strain && flags.selective_dynamics) {
                bool fixa = true, fixb = true, fixc = true; // only works if symm_strain
                press = 0;
                int ipress = 0;
                // xx
                if (symm_strain_mask[0] > 0) {
                    ++ipress;
                    press += curr_stress(0, 0);
                    fixa = false;
                }
                // yy
                if (symm_strain_mask[3] > 0) {
                    ++ipress;
                    press += curr_stress(1, 1);
                    fixb = false;
                }
                // zz
                if (symm_strain_mask[5] > 0) {
                    ++ipress;
                    press += curr_stress(2, 2);
                    fixc = false;
                }
                if (ipress > 0) {
                    press /= ipress;
                }
                curr_stress -= press * Eigen::Matrix3d::Identity();
                if (fixa)
                    curr_stress(0, 0) = 0;
                if (fixb)
                    curr_stress(1, 1) = 0;
                if (fixc)
                    curr_stress(2, 2) = 0;
            } else {
                curr_stress -= press * Eigen::Matrix3d::Identity();
            }
        }

        // couple: average the generalized forces of coupled axes.
        // Applied in gradient space (after cell⁻ᵀ) so the optimizer sees
        // equal gradient components, ensuring equal step sizes for coupled DOFs.
        // TODO: average curr_stress or gstress?
        if (control.couple_flag > 0) {
            if (control.couple_flag == 1) {
                double gavg = (curr_stress(0, 0) + curr_stress(1, 1)) / 2.0;
                curr_stress(0, 0) = gavg;
                curr_stress(1, 1) = gavg;
            } else if (control.couple_flag == 2) {
                double gavg = (curr_stress(2, 2) + curr_stress(1, 1)) / 2.0;
                curr_stress(2, 2) = gavg;
                curr_stress(1, 1) = gavg;
            } else if (control.couple_flag == 3) {
                double gavg = (curr_stress(0, 0) + curr_stress(2, 2)) / 2.0;
                curr_stress(0, 0) = gavg;
                curr_stress(2, 2) = gavg;
            } else if (control.couple_flag == 4) {
                double gavg = curr_stress.diagonal().sum() / 3.0;
                curr_stress(0, 0) = gavg;
                curr_stress(1, 1) = gavg;
                curr_stress(2, 2) = gavg;
            }
        }
        // Generalized stress: gstress = -cell⁻ᵀ * σ
        // Cache inverse for reuse below
        Eigen::Matrix3d inv_cell_T = atom.cell_.inverse().transpose();
        Eigen::Matrix3d ext_cell_grad = -inv_cell_T * external_stress * atom.get_volume();
        gstress = -inv_cell_T * curr_stress + ext_cell_grad;
        if (flags.relax_cell_use_strain && !flags.symm_strain) {
            gstress = refcell.transpose() * gstress;
            Eigen::Matrix3d ext_grad = refcell.transpose() * ext_cell_grad;
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    g[3*a+b] = gstress(a, b);
                    f += x[3*a+b] * ext_grad(a, b);
                }
            }
        } else if (flags.relax_cell_use_strain && flags.symm_strain) {
            // ── Strain gradient (symm_strain mode) ──
            // Forward: x → ε → eps=I+ε → L=chol(eps) → cell=refcell·L
            const Eigen::Matrix3d L = ivrefcell * atom.cell_;
            const auto strain_grad = SaddleCoordinates::symmetric_pullback(
                L, refcell.transpose() * gstress, true);
            const auto ext_grad = SaddleCoordinates::symmetric_pullback(
                L, refcell.transpose() * ext_cell_grad, true);
            if (!strain_grad.allFinite() || !ext_grad.allFinite())
                throw std::runtime_error("Exact Cholesky gradient is not finite");
            for (int i = 0; i < 6; ++i) {
                g[i] = strain_grad(i);
                f += x[i] * ext_grad(i);
            }
        } else {
            for (int a = 0; a < 3; ++a) {
                for (int b = 0; b < 3; ++b) {
                    g[3 * a + b] = gstress(a, b);
                    f += (x[3*a+b] - refcell(a, b)) * ext_cell_grad(a, b);
                }
            }
        }
    }
    if (flags.fix_shape) {
        // x[0] = scaling;
        g[0] = -atom.stress_.trace() / scaling + 3.0 * pscal / scaling;
    }
    // pos
    if (flags.relax_atom) {
        if (flags.cartesian || cell_free_recast_coordinates(*this)) {
            for (int i = 0; i < atom.size(); ++i) {
                Eigen::RowVector3d gforce = -atom.forces_[i];
                for (int a = 0; a < 3; ++a) {
                    g[xposoffset + 3 * i + a] = gforce(a);
                }
            }
        } else {
            Eigen::Matrix3d matD = ivrefcell * atom.cell_;
            for (int i = 0; i < atom.size(); ++i) {
                Eigen::RowVector3d gfracforce;
                if (flags.atomic_force_recast) {
                    gfracforce = -matD * atom.forces_[i];
                } else {
                    gfracforce = -atom.cell_ * atom.forces_[i];
                } 
                for (int a = 0; a < 3; ++a) {
                    g[xposoffset+3*i+a] = gfracforce(a);
                }
            } 
        }
    }
    // constrain
    for (int i = 0; i < constraints.size(); ++i) {
        compute_regulator(constraints[i]);
    }
}

void RelaxationDrvV2::apply_modifiers(std::vector<AnyModifier*> &modifiers)
{
    for (auto& f : modifiers) {
        f->forward(this);
    }
}

void RelaxationDrvV2::reverse_modifiers(std::vector<AnyModifier*> &modifiers)
{
    for (auto it = modifiers.rbegin(); it != modifiers.rend(); ++it) {
        (*it)->reverse(this);
    }
}

void RelaxationDrvV2::reset_state()
{
    has_checkpoint = false;
    accepted_steps=0;fc_update.reset();
    kkt_residual=std::numeric_limits<double>::infinity();
    stop_reason = "not_started";
    state_vars.curr_iter = 0;
    state_vars.optstate = 0;
    state_vars.converged = false;
    state_vars.reach_ftol = 0;
    state_vars.reach_gtol = 0;
    state_vars.reach_xtol = 0;
    state_vars.curr_constraint_iter = 0;
    inner_iter_start = 0;
    state_vars.max_move = 0;
    state_flags.request_alg_restart = false;
}

bool RelaxationDrvV2::check_converage(type_tol& ctol)
{
    state_vars.reach_ftol = state_vars.reach_gtol = state_vars.reach_xtol = 0;
    state_vars.optstate = -1;
    state_vars.converged = false;

    // ── Explosion / divergence detection ──
    if (!flags.fix_shape) {
        double stress_max = atom.stress_.lpNorm<Eigen::Infinity>();
        double force_max = 0;
        for (int i = 0; i < atom.size(); ++i)
            force_max = std::max(force_max, atom.forces_[i].squaredNorm());
        force_max = std::sqrt(force_max);
        if (stress_max > 1e4 || force_max > 1e4) {
            Message(fmt::format("DIVERGED: max stress={:.2e} GPa, max force={:.2e} eV/A", stress_max, force_max));
            state_vars.optstate = -5;
            return true;
        }
    }

    if (state_vars.curr_iter > control.max_iter) {
        Message(fmt::format("Max iteration reached: {}", control.max_iter));
        state_vars.optstate = -1;
        return true;
    }
    if (ctol.ftol > 0 && state_vars.curr_iter > inner_iter_start + 2 &&
        std::abs(f - f_prev) < ctol.ftol) {
        state_vars.reach_ftol = 1;
    }
    if (g.size() == 0 || convergence_gradient().cwiseAbs().maxCoeff() <= ctol.gtol) {
        state_vars.reach_gtol = 1;
    }
    if (ctol.xtol > 0 && state_vars.curr_iter > inner_iter_start && x.size() == x_prev.size() && (x - x_prev).norm() < ctol.xtol) {
        state_vars.reach_xtol = 1;
    }
    if (state_vars.reach_ftol == 1 || state_vars.reach_gtol == 1 || state_vars.reach_xtol == 1) {
        state_vars.optstate = 1;
        state_vars.converged = state_vars.reach_gtol == 1;
        Message("Stopping criterion reached");
        return true;
    }

    return false;
}

bool RelaxationDrvV2::check_converage_final(type_tol& ctol)
{
    // Check the physical gradient at the accepted point.
    state_vars.optstate = -1;
    state_vars.converged = false;
    for (int i = 0; i < g.size(); ++i) {
        if (!std::isfinite(g[i]) || std::abs(convergence_gradient()[i]) > ctol.gtol) {
            Message(fmt::format("Component {:4d} unconverged: |g| = {:.4e} > {:.2e}", i, std::abs(g[i]), ctol.gtol));
            return false;
        }
    }
    Message("All convergence criteria satisfied — optimization complete");
    state_vars.optstate = 0;
    state_vars.converged = true;
    return true;
}

bool RelaxationDrvV2::check_cons_convergence()
{
    double constraint_voilate = 0.0;
    bool conv = true;
    for (auto& it : constraints) {
        validate_regulator_values(it, true);
        if (!std::isfinite(it.f)) throw std::runtime_error("Non-finite constraint residual");
        it.update = true;
        if (it.bounded) {
            const double violation = std::max({it.lower-it.f, it.f-it.upper, 0.0});
            if (violation > it.tol) conv = false; else it.update = false;
            continue;
        }
        double violation = 0.0;
        switch (it.type) {
        case 10:
            violation = it.f;
            break;
        case 11:
            violation = std::max(it.f, -it.lambda / it.weight);
            break;
        case 9:
            violation = std::max(-it.f, -it.lambda / it.weight);
            break;
        case 0:
            violation = it.f;
            break;
        case 1:
            violation = std::min(0.0, it.f);
            break;
        case -1:
            violation = std::max(0.0, it.f);
            break;
        default:
            it.update = false;
            continue;
        }

        constraint_voilate += violation * violation;

        if (std::abs(violation) > it.tol) {
            conv = false;
        } else {
            it.update = false;
        }
    }
    constraint_voilate = std::sqrt(constraint_voilate);
    return conv;
}

void RelaxationDrvV2::write_relax_log(double _f)
{
    if (state_vars.curr_iter % control.nblock == 0) {
        if (flags.fix_shape)
            Message(fmt::format("Fixed shape: scaling = {}", scaling));
        double gmax = *std::max_element(g.begin(), g.end());
        double gmin = *std::min_element(g.begin(), g.end());
        Message(fmt::format("{} iter {}, f = {:<16f} gmax = {:<8g} gmin = {:<8g}", control.driver, state_vars.curr_iter, _f, gmax, gmin));
        if (flags.write_traj && ofs.is_open() && !iobj.seq_write(ofs, atom))
            GPERROR("Failed to write relaxation trajectory: " + control.trajfile);
    }
}

/**
 * @brief Invoke this function to do relaxation.
 *
 * @return int relaxation state.
 */
void RelaxationDrvV2::accept_checkpoint(bool count_step) {
    const auto previous_steps=accepted_steps;
    try {
        const bool moved=has_checkpoint && (atom.cell_!=checkpoint_atom.cell_ || atom.pos_!=checkpoint_atom.pos_);
        if(moved)++accepted_steps;
        for(size_t i=0;i<modifiers_list.size();++i) if(auto *nf=dynamic_cast<FilterNF *>(modifiers_list[i])) {
            const auto full=nf->map->lift(nf->point);
            if(nf->map->accepted(nf->point)) {
                // The reference chart changes, but its accepted physical point,
                // energy, EFS and full gradient do not. Re-express those values
                // instead of evaluating the potential again at the same point.
                nf->point=nf->map->reduce(full);
                x=nf->point;g=nf->map->jacobian(nf->point).transpose()*nf->full_gradient;
                for(size_t j=i+1;j<modifiers_list.size();++j)modifiers_list[j]->forward(this);
                state_flags.request_alg_restart=true;
            }
        }
        if(moved && fc_update.due(accepted_steps))refresh_fc_metric();
        if (count_step && has_checkpoint) ++state_vars.curr_iter;
        checkpoint_x = x;
        checkpoint_g = g;
        checkpoint_f = f;
        checkpoint_scaling = scaling;
        checkpoint_atom = atom;
        checkpoint_constraints = constraints;
        has_checkpoint = true;
    } catch(...) {
        accepted_steps=previous_steps;
        throw;
    }
}

void RelaxationDrvV2::restore_checkpoint() {
    if (!has_checkpoint) return;
    x = checkpoint_x;
    g = checkpoint_g;
    f = checkpoint_f;
    scaling = checkpoint_scaling;
    atom = checkpoint_atom;
    constraints = checkpoint_constraints;
}

int RelaxationDrvV2::run() {
    reset_state();
    stop_reason = "running";
    try {
        validate_options();
        if (!initialized) throw std::logic_error("Relaxation driver requires successful init before run");
        (void)prepare_constraints(atom, constraints);
        for (const auto &reg : constraints) {
            validate_regulator_values(reg, true);
            validate_regulator_layout(reg, xposoffset + (flags.relax_atom ? atom.size() * 3 : 0), xposoffset + (flags.relax_atom ? atom.size() * 3 : 0));
        }
        if (!std::isfinite(control.max_iter) || control.max_iter < 0 || control.max_iter > std::numeric_limits<int>::max() ||
            !std::isfinite(control.max_move) || control.max_move <= 0 ||
            !std::isfinite(tol.gtol) || tol.gtol < 0)
            throw std::invalid_argument("Invalid relaxation iteration, step, or gradient limits");
        if (!control.driver.empty() && control.driver != "lbfgs" && control.driver != "lbfgspp" &&
            control.driver != "cg" && control.driver != "gd" && control.driver != "fire" && control.driver != "mdmin" && control.driver != "quickmin" && control.driver != "sqp" && control.driver != "nlc")
            throw std::invalid_argument("Unknown relaxation optimizer");
        run_impl();
        if ((flags.with_constraint_aul || flags.with_constraint_sqp || flags.with_constraint_lc || flags.with_constraint_nlc) &&
            !check_cons_convergence()) state_vars.converged = false;
        if (state_vars.converged)
            stop_reason = x.size() ? "gradient_tolerance" : "zero_degrees_of_freedom";
        else if (stop_reason == "running")
            stop_reason = state_vars.curr_iter >= control.max_iter ? "iteration_limit" : "optimizer_stop";
        return state_vars.converged ? 0 : -1;
    } catch (...) {
        if (ofs.is_open()) ofs.close();
        restore_checkpoint();
        state_vars.converged = false;
        stop_reason = "error";
        throw;
    }
}

int RelaxationDrvV2::run_impl()
{
    // initial displacements
    x.resize(xposoffset + (flags.relax_atom ? 3 * atom.size() : 0));
    g = Eigen::VectorXd::Zero(x.size());
    LatPos2X();
    if (flags.keep_symm) {
        for (auto *modifier : modifiers_list) {
            auto *symmetry = dynamic_cast<FilterSymmCoord *>(modifier);
            // init() has already identified and exactified the first reference.
            // A subsequent run rebuilds from the last accepted full structure.
            if (!symmetry || symmetry->x_save.size() != 0)
                modifier->init(this, &tol.symprec);
            modifier->forward_scaling(this);
        }
    }
    apply_modifiers(modifiers_list);
    if (flags.apply_disp) {
        if (flags.disp_gaussian) {
            std::normal_distribution<double> dist(0, control.disp);
            for (int i = 0; i < x.size(); ++i) {
                x[i] += dist(RNG.mtd) / prec_scaling[i];
            }
        } else {
            std::uniform_real_distribution<double> dist(-control.disp, control.disp);
            for (int i = 0; i < x.size(); ++i) {
                x[i] += dist(RNG.mtd);
            }
        }
    }
    if (flags.fix_shape) {
        // A full-coordinate mask can exclude the scale variable, including an
        // empty reduced vector. Reset it in full space, not at reduced x[0].
        reverse_modifiers(modifiers_list);
        x[0] = 1.0;
        scaling = 1.0;
        apply_modifiers(modifiers_list);
    }
    eval();
    if (flags.apply_disp && flags.relax_cell && !flags.keep_symm && !subspace_active) {
        refcell = atom.cell_;
        ivrefcell = refcell.inverse();
        reverse_modifiers(modifiers_list);
        if (control.precond_file.empty() && (use_dprecond || control.thin_cell_scaling)) {
            // Rebuild in full coordinates before applying LF/LC/selection again.
            prec_scaling.setOnes(x.size());
            automatic_scaling(*this);
            for (auto *modifier : modifiers_list) modifier->forward_scaling(this);
            if (flags.force_switchoff_scaling || sparse_metric) prec_scaling.setOnes();
        }
        LatPos2X();
        apply_modifiers(modifiers_list);
        eval();
    }
    // _DEBUG_OUTPUT_(g.transpose());
    accept_checkpoint(false);
    f_prev = f;
    energy0 = atom.energy_;
    iobj.append = flags.traj_append;
    if (flags.write_traj) {
        if (!control.indexfile.empty() && !iobj.readmap(control.indexfile))
            GPERROR("Cannot read relaxation index map: " + control.indexfile);
        if (ofs.is_open()) ofs.close();
        ofs.clear();
        ofs.open(control.trajfile, flags.traj_append ? std::ios::app : std::ios::out);
        if (!ofs) GPERROR("Cannot open relaxation trajectory: " + control.trajfile);
        if (!iobj.seq_write_init(ofs, "", control.trajfile))
            GPERROR("Cannot initialize relaxation trajectory: " + control.trajfile);
    }

    int result;
    if (flags.with_constraint_aul) {
        result = run_aul();
    } else if (constrained_solver()) {
        result = run_alg();
    } else {
        result = run_normal();
    }
    if (flags.write_traj) {
        ofs.close();
        if (!ofs) GPERROR("Failed to write relaxation trajectory: " + control.trajfile);
    }
    return result;
}

int RelaxationDrvV2::run_aul()
{
    validate_options();
    if (!initialized || !has_checkpoint)
        throw std::logic_error("run_aul requires initialized accepted state; use run() to start relaxation");
    if (control.constraint_iter_limit <= 0 || !std::isfinite(tol.gtol0) || tol.gtol0 < tol.gtol)
        throw std::invalid_argument("AUL requires positive constraint_loops and finite gtol0 >= gtol");
    if(control.aul_auto)initialize_aul_parameters();
    const double final_gtol = tol.gtol;
    struct RestoreTolerance { double &value; double saved; ~RestoreTolerance() { value = saved; } } restore{tol.gtol, final_gtol};
    bool final_stage = false;
    for (int consiter = 0; consiter < control.constraint_iter_limit; ++consiter) {
        Message(fmt::format("AUG Loop: {}", consiter));
        state_vars.curr_constraint_iter = consiter + 1;
        inner_iter_start = state_vars.curr_iter;
        state_vars.reach_ftol = state_vars.reach_gtol = state_vars.reach_xtol = 0;
        // Tighten early enough to leave outer iterations for dual corrections.
        // A feasible geometry must also optimize to the final gradient tolerance.
        final_stage = final_stage || check_cons_convergence();
        tol.gtol = final_stage || consiter + 1 == control.constraint_iter_limit ? final_gtol :
            std::max(final_gtol, tol.gtol0 * std::pow(0.1, consiter));
        run_normal();
        bool constraints_met = check_cons_convergence();
        bool final_gradient_met = g.allFinite() && (g.size() == 0 || convergence_gradient().cwiseAbs().maxCoeff() <= final_gtol);
        if (constraints_met && !final_gradient_met && tol.gtol > final_gtol && state_vars.curr_iter < control.max_iter) {
            final_stage = true;
            tol.gtol = final_gtol;
            inner_iter_start = state_vars.curr_iter;
            run_normal();
            constraints_met = check_cons_convergence();
            final_gradient_met = g.allFinite() && (g.size() == 0 || convergence_gradient().cwiseAbs().maxCoeff() <= final_gtol);
        }
        if (constraints_met && final_gradient_met) break;
        if (consiter + 1 == control.constraint_iter_limit || state_vars.curr_iter >= control.max_iter) break;
        update_regs();
        // Multipliers change the objective; keep the geometry and refresh its
        // accepted residual/objective before starting a new inner optimization.
        eval();
        accept_checkpoint(false);
    }
    tol.gtol = final_gtol;
    check_converage_final(tol);
    if (!check_cons_convergence()) state_vars.converged = false;
    state_vars.optstate = state_vars.converged ? 0 : -1;
    return state_vars.optstate;
}


int RelaxationDrvV2::run_normal()
{
    do {
    state_flags.request_alg_restart=false;
    if (x.size() != 0 && !check_converage(tol) && state_vars.curr_iter < control.max_iter) {
        if (control.driver == "lbfgs" || control.driver.empty())
            run_lbfgs();
        else if (control.driver == "cg")
            run_cg();
        else if (control.driver == "gd")
            run_gd();
        else if (control.driver == "fire")
            run_fire();
        else if (control.driver == "mdmin")
            run_mdmin();
        else if (control.driver == "quickmin")
            run_mdmin(true);
        else if (control.driver == "lbfgspp")
            run_lbfgspp();
    }
    } while(state_flags.request_alg_restart && state_vars.curr_iter<control.max_iter && !check_converage(tol));
    restore_checkpoint();

    if (flags.write_traj && !iobj.seq_write(ofs, atom))
        GPERROR("Failed to write relaxation trajectory: " + control.trajfile);

    // Always run the definitive per-component check at the end.
    // check_converage only gives an early exit; final check is definitive.
    check_converage_final(tol);
    return state_vars.optstate;
}

void mindriver_new_bfgsgard(const alglib::real_1d_array& x,
    double& func,
    alglib::real_1d_array& gard,
    void* ptr)
{
    RelaxationDrvV2* pdriver = (RelaxationDrvV2*)ptr;
    // Store previous x for xtol convergence check
    pdriver->x_prev = pdriver->x;
    memcpy(pdriver->x.data(), x.getcontent(), sizeof(double) * x.length());
    pdriver->eval();
    memcpy(gard.getcontent(), pdriver->g.data(), sizeof(double) * pdriver->g.size());
    pdriver->f_prev = func;
    func = pdriver->f;
    pdriver->write_relax_log(func);

};

void mindriver_new_bfgsrep(const alglib::real_1d_array& x,
    double func,
    void* ptr)
{
    auto *driver = static_cast<RelaxationDrvV2 *>(ptr);
    // ALGLIB reports each accepted point, including its final point.
    driver->x = Eigen::Map<const Eigen::VectorXd>(x.getcontent(), x.length());
    driver->eval();
    driver->accept_checkpoint(!driver->optimizer_initial_report);
    driver->optimizer_initial_report = false;
    if(driver->state_flags.request_alg_restart) {
        if(driver->control.driver=="cg") alglib::mincgrequesttermination(driver->cg_state);
        else alglib::minlbfgsrequesttermination(driver->bfgs_state);
    }
}

void RelaxationDrvV2::run_lbfgs()
{
    optimizer_initial_report = true;
    alglib::real_1d_array ae_x;
    ae_x.setcontent(x.size(), x.data());
    if (buffer_newton.bfgs_m == -1 || buffer_newton.bfgs_m > ae_x.length()) {
        buffer_newton.bfgs_m = ae_x.length();
    }
    alglib::minlbfgscreate(ae_x.length(), buffer_newton.bfgs_m, ae_x, bfgs_state);
    alglib::minlbfgssetcond(bfgs_state, optimizer_gtol(), tol.ftol, tol.xtol, std::max(1, static_cast<int>(control.max_iter) - state_vars.curr_iter));
    alglib::minlbfgssetxrep(bfgs_state, true);
    alglib::minlbfgssetstpmax(bfgs_state, control.max_move);
    alglib::real_1d_array scal;
    scal.setcontent(prec_scaling.size(), prec_scaling.data());
    // _DEBUG_OUTPUT_(prec_scaling.transpose());
    if (state_flags.request_alg_restart) {
        alglib::minlbfgsrestartfrom(bfgs_state, ae_x);
    }
    alglib::minlbfgssetscale(bfgs_state, scal);

    // main loop
    write_relax_log(f);
    alglib::minlbfgsoptimize(bfgs_state, mindriver_new_bfgsgard, mindriver_new_bfgsrep, this);
}

void RelaxationDrvV2::run_cg()
{
    optimizer_initial_report = true;
    alglib::real_1d_array ae_x;
    ae_x.setcontent(x.size(), x.data());
    alglib::mincgcreate(ae_x.length(), ae_x, cg_state);
    alglib::mincgsetcond(cg_state, optimizer_gtol(), tol.ftol, tol.xtol, std::max(1, static_cast<int>(control.max_iter) - state_vars.curr_iter));
    alglib::mincgsetxrep(cg_state, true);
    alglib::mincgsetstpmax(cg_state, control.max_move);
    alglib::real_1d_array scal;
    scal.setcontent(prec_scaling.size(), prec_scaling.data());
    if (state_flags.request_alg_restart) {
        alglib::mincgrestartfrom(cg_state, ae_x);
    }
    alglib::mincgsetscale(cg_state, scal);

    // main loop
    write_relax_log(f);
    alglib::mincgoptimize(cg_state, mindriver_new_bfgsgard, mindriver_new_bfgsrep, this);
}

void RelaxationDrvV2::run_gd()
{
    if (!std::isfinite(control.min_move) || control.min_move <= 0)
        throw std::invalid_argument("Invalid gradient-descent minimum step");
    double fnmax = 0.0;
    Eigen::Matrix3d ncell;
    // init fnmax
    fnmax = 1.0 / std::max(fabs(*std::max_element(g.begin(), g.end(), [](const double& a, const double& b) { return fabs(a) < fabs(b); })), 1e-30);
    state_vars.max_move = std::min(control.max_move, buffer_gd.max_move_init);
    write_relax_log(f);
    for (; state_vars.curr_iter < control.max_iter;) {
        x_prev = x;
        g_prev = g;
        f_prev = f;
        const auto direction=control.sparse_precond?preconditioned_gradient(g):g.cwiseQuotient((prec_scaling.array()+tol.eps).matrix()).eval();
        for (int i = 0; i < x.size(); ++i) {
            double gd_step = buffer_gd.alpha * direction[i] * state_vars.max_move * fnmax;
            if (gd_step > control.max_move) {
                x[i] -= control.max_move;
            } else if (gd_step < -control.max_move) {
                x[i] += control.max_move;
            } else {
                x[i] -= gd_step;
            }
        }
        eval();
        write_relax_log(f);
        // accept or denie
        std::cout << "  F F_P: " << f << " " << f_prev << "  ";
        if (f < f_prev) {
            accept_checkpoint();
            if (check_converage(tol)) break;
            Message("Accept");
            // accept step, and enlarge step_max
            fnmax = 1.0 / std::max(fabs(*std::max_element(g.begin(), g.end(), [](const double& a, const double& b) { return fabs(a) < fabs(b); })), 1e-30);
            if (state_vars.max_move < control.max_move) {
                state_vars.max_move *= buffer_gd.step_upscale;
            }
        } else {
            Message("Revert");
            restore_checkpoint();
            if (state_vars.max_move <= control.min_move) { stop_reason = "minimum_step"; break; }
            state_vars.max_move = std::max(control.min_move, state_vars.max_move * buffer_gd.step_downscale);
        }
    }
}

void RelaxationDrvV2::run_fire()
{
    validate_fire_options();
    if (prec_scaling.size() != g.size() || !prec_scaling.allFinite() ||
        (prec_scaling.array() <= 0).any())
        throw std::invalid_argument("FIRE requires finite positive preconditioning for every coordinate");
    // A new AUL inner solve has a refreshed objective and its own FIRE restart.
    if (stop_reason == "minimum_step") stop_reason = "running";
    if (check_converage(tol)) return;
    const double initial_move = std::min(control.max_move,
        first_step > 0 ? first_step : 0.1 * control.max_move);
    buffer_fire.alpha = buffer_fire.alpha_start;
    buffer_fire.ts = buffer_fire.ts_init;
    // Explicit small initial steps remain valid with the default configured floor.
    const double ts_min = std::min(buffer_fire.ts_min, buffer_fire.ts_init);
    buffer_fire.num_it_pnegative = 0;
    buffer_fire.num_it_ppostive = 0;
    // Keep eval's gradient intact for convergence checks and checkpoints.
    Eigen::VectorXd fire_g = preconditioned_gradient(g);
    Eigen::VectorXd v = Eigen::VectorXd::Zero(x.size());
    // The first trial needs a kick, not a restart of the initial zero velocity.
    bool skip_power_update = true;
    for (; state_vars.curr_iter < control.max_iter;) {
        // Startup needs no power scheduling; rejection already restarted FIRE.
        // Do not shrink twice merely because the restarted velocity is zero.
        if (!skip_power_update) {
            // Power is the true objective derivative along the coordinate velocity.
            const double workP = -g.dot(v);
            if (workP > 0) {
                buffer_fire.num_it_ppostive++;
                buffer_fire.num_it_pnegative = 0;
                if (!buffer_fire.initaildeyal || buffer_fire.num_it_ppostive > buffer_fire.ndelay) {
                    buffer_fire.ts = std::min(buffer_fire.ts_max, buffer_fire.ts_grow * buffer_fire.ts);
                    buffer_fire.alpha *= buffer_fire.alpha_reduce;
                }
            } else {
                buffer_fire.num_it_pnegative++;
                buffer_fire.num_it_ppostive = 0;
                if (buffer_fire.num_it_pnegative > buffer_fire.vdfitmax) {
                    state_vars.optstate = -1;
                    break;
                }
                buffer_fire.ts = std::max(ts_min, buffer_fire.ts * buffer_fire.ts_reduce);
                buffer_fire.alpha = buffer_fire.alpha_start;
                if (!buffer_fire.downhill_check && buffer_fire.halvstepback) {
                    Message("Uphill");
                    x -= buffer_fire.ts_backward_ratio * buffer_fire.ts * v;
                    eval();
                    fire_g = preconditioned_gradient(g);
                }
                v.setZero();
            }
        }
        skip_power_update = false;
        // Mix one current velocity, then apply one semi-implicit Euler kick.
        const double vv = std::sqrt(std::max(0.,metric_inner(v,v)));
        v *= 1.0 - buffer_fire.alpha;
        const double gnorm = std::sqrt(std::max(0.,g.dot(fire_g)));
        if (gnorm > 0) v -= buffer_fire.alpha * fire_g * vv / gnorm;
        v -= buffer_fire.ts * fire_g;
        const double move = state_vars.curr_iter < first_step_delay ? initial_move : control.max_move;
        const double vnorm = v.norm();
        if (vnorm > move / buffer_fire.ts)
            v *= move / buffer_fire.ts / vnorm;

        x_prev = checkpoint_x;
        g_prev = checkpoint_g;
        f_prev = checkpoint_f;
        x += buffer_fire.ts * v;
        // Both accepted and rejected trials spend the shared AUL iteration budget.
        ++state_vars.curr_iter;
        eval();
        if (buffer_fire.downhill_check && f > checkpoint_f) {
            restore_checkpoint();
            fire_g = preconditioned_gradient(g);
            v.setZero();
            buffer_fire.alpha = buffer_fire.alpha_start;
            buffer_fire.num_it_ppostive = buffer_fire.num_it_pnegative = 0;
            if (buffer_fire.ts <= ts_min) {
                stop_reason = "minimum_step";
                break;
            }
            buffer_fire.ts = std::max(ts_min, buffer_fire.ts * buffer_fire.ts_reduce);
            skip_power_update = true;
            continue;
        }
        accept_checkpoint(false);
        if(state_flags.request_alg_restart) {
            v.setZero();buffer_fire.alpha=buffer_fire.alpha_start;buffer_fire.ts=buffer_fire.ts_init;
            buffer_fire.num_it_ppostive=buffer_fire.num_it_pnegative=0;
            state_flags.request_alg_restart=false;skip_power_update=true;
        }
        fire_g = preconditioned_gradient(g);
        write_relax_log(f);
        if (check_converage(tol)) break;
    }
}

void RelaxationDrvV2::run_mdmin(bool quickmin)
{
    const double dt = quickmin ? quickmin_dt : mdmin_dt;
    if (!std::isfinite(dt) || dt <= 0 || prec_scaling.size() != g.size() ||
        !prec_scaling.allFinite() || (prec_scaling.array() <= 0).any())
        throw std::invalid_argument("MDMin requires finite positive timestep and coordinate scaling");
    Eigen::VectorXd velocity = Eigen::VectorXd::Zero(x.size());
    bool initial = true;
    for (; state_vars.curr_iter < control.max_iter && !check_converage(tol);) {
        const Eigen::VectorXd force = -g.cwiseQuotient(prec_scaling);
        if (!initial || quickmin) {
            if (!quickmin) velocity += 0.5 * dt * force;
            const double power = velocity.dot(force);
            if (power <= 0) velocity.setZero();
            else velocity = (power / force.squaredNorm()) * force;
        }
        initial = false;
        velocity += (quickmin ? dt : 0.5 * dt) * force;
        Eigen::VectorXd displacement = dt * velocity;
        const double norm = displacement.norm();
        if (norm > control.max_move) displacement *= control.max_move / norm;
        x_prev = checkpoint_x;
        g_prev = checkpoint_g;
        f_prev = checkpoint_f;
        x += displacement;
        ++state_vars.curr_iter;
        eval();
        accept_checkpoint(false);
        write_relax_log(f);
    }
}

void RelaxationDrvV2::run_lbfgspp()
{
    Eigen::VectorXd emx;
    emx.resize(x.size());
    // initial guess
    memcpy(emx.data(), x.data(), sizeof(double) * x.size());
    double emf = f;

    // Python and C++ callers may modify these public settings after init().
    lbfgspp_param.epsgmax = optimizer_gtol();
    lbfgspp_param.max_step_move = control.max_move;
    lbfgspp_param.m = buffer_newton.bfgs_m;
    auto parameters = lbfgspp_param;
    const int remaining = static_cast<int>(control.max_iter) - state_vars.curr_iter;
    parameters.max_iterations = parameters.max_iterations > 0 ?
        std::min(parameters.max_iterations, remaining) : remaining;
    if (parameters.max_iterations <= 0) { stop_reason = "iteration_limit"; return; }
    auto accepted = [this](const Eigen::VectorXd &point, double energy, const Eigen::VectorXd &gradient) {
        x = point; f = energy; g = gradient;
        accept_checkpoint();
    };
    if (parameters.linesearch == LBFGSpp::LBFGS_LINESEARCH_FORCE_ONLY_A) {
        parameters.past = 0; // Projected/nonconservative forces need no energy-decrease stopping rule.
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchForceOnlyA> solver(parameters);
        solver.minimize(*this, emx, emf, accepted);
    } else if (parameters.linesearch == LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBacktracking> solver(parameters);
        solver.minimize(*this, emx, emf, accepted);
    } else if (parameters.linesearch == LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_ARMIJO) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchBracketing> solver(parameters);
        solver.minimize(*this, emx, emf, accepted);
    } else if (parameters.linesearch == LBFGSpp::LBFGS_LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchNocedalWright> solver(parameters);
        solver.minimize(*this, emx, emf, accepted);
    } else if (parameters.linesearch == LBFGSpp::LBFGS_LINESEARCH_LEWIS_OVERTON) {
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchLewisOverton> solver(parameters);
        solver.minimize(*this, emx, emf, accepted);
    } else {
        throw std::invalid_argument("Unknown LBFGSpp line search method");
    }
}

double RelaxationDrvV2::operator()(const Eigen::VectorXd& mx, Eigen::VectorXd& mg)
{
    memcpy(x.data(), mx.data(), sizeof(double) * x.size());
    eval();
    memcpy(mg.data(), g.data(), sizeof(double) * g.size());
    // output information
    write_relax_log(f);
    return f;
}

void RelaxationDrvV2::set_isif(int isif)
{
    switch (isif) {
    case 2:
        flags.relax_atom = true;
        flags.relax_cell = false;
        flags.fix_shape = false;
        flags.fix_vol = false;
        break;
    case 3:
        flags.relax_atom = true;
        flags.relax_cell = true;
        flags.fix_shape = false;
        flags.fix_vol = false;
        break;
    case 4:
        flags.relax_atom = true;
        flags.relax_cell = true;
        flags.fix_shape = false;
        flags.fix_vol = true;
        break;
    case 5:
        flags.relax_atom = false;
        flags.relax_cell = true;
        flags.fix_shape = false;
        flags.fix_vol = true;
        break;
    case 6:
        flags.relax_atom = false;
        flags.relax_cell = true;
        flags.fix_shape = false;
        flags.fix_vol = false;
        break;
    case 7:
        flags.relax_atom = false;
        flags.relax_cell = false;
        flags.fix_shape = true;
        flags.fix_vol = false;
        break;
    case 8:
        flags.relax_atom = true;
        flags.relax_cell = false;
        flags.fix_shape = true;
        flags.fix_vol = false;
        break;
    case 13:
        flags.relax_atom = true;
        flags.relax_cell = true;
        flags.fix_shape = true;
        flags.fix_vol = false;
        flags.selective_dynamics = true;
        flags.symm_strain = true;
        flags.relax_cell_use_strain = true;
        for (int i = 0; i < 6; ++i) {
            symm_strain_mask[i] = 0;
        }
        symm_strain_mask[0] = 1;
        symm_strain_mask[3] = 1;
        symm_strain_mask[5] = 1;
        break;
    default:
        break;
    }
}
