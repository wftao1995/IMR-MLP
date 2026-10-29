# Configuration System {#guide_config}

[中文](configuration_zh.md)

MTP uses `mtp` type names and the `trainmtp` command.

@tableofcontents

## Overview

IMR-MLP uses an ini-style configuration system based on `ConnectSettings`.
Settings files organize parameters under **namespaced prefixes** that route
values to the appropriate subsystem.

## File Format

```text
# Comments start with # or !
key value
section:key value
```

Keys and values are separated by whitespace. An equals sign becomes part of the
value and is usually invalid. This applies to settings files, not to the separate
`.mtp` model-file grammar. Relative paths resolve from the process working directory.

### Bracket Section Syntax

As an alternative to prefixing each key individually, you can use `[section]` headers
to set a namespace prefix for all keys that follow:

```text
[pot]
pottype    mtp
potfile    potential.mtp

[relax]
driver         lbfgs
maxiteration   500
```

Rules:
- A `[section]` header sets the namespace prefix for subsequent key-value lines.
- The section applies **only to consecutive lines** — a blank line ends the current
  section, and subsequent keys revert to the top-level namespace.
- Comment-only lines (`# ...` or `! ...`) do **not** end the section.
- Use explicit `section:key` keys outside a bracket section (insert a blank line first),
  otherwise the active section is prepended again.
- Duplicate keys keep the **first** value. Remove duplicates instead of relying on overrides.

### Value Types

```text
# Scalars
cutoff 5.0             # double
max_iter 500           # int
verbose yes            # bool (yes/no, true/false, y/n, t/f, 1/0)

# Vectors use whitespace, when the receiving setting accepts a vector
relax:strain_mask 1 1 1 0 0 0

# Continue one setting value with a trailing backslash
fit:trainable_groups \
    sr_radial sr_linear \
    les_species les_moment
```

### Namespace Prefixes

Every subsystem prefix routes parameters. Without a prefix, the value goes
to the top-level settings container.

| Prefix | Subsystem |
|--------|-----------|
| *(none)* | Top-level / global settings |
| `relax:` | Structure relaxation |
| `fit:` | Potential training (`trainmtp`, `trainnlpot`, `train_linear`) |
| `al:` | Extrapolation monitoring and candidate collection during relaxation |
| `neb:` | Nudged elastic band |
| `dimer:` | Dimer saddle search (`calc_dimer`) |
| `pot:` | Potential |
| `builder:` | Descriptor/B-matrix construction for MaxVol, FPS/CUR, and crowding |
| `selector:` | MaxVol active-set selection (`maxvol_*`) |
| `behler:` | Behler/ACSF descriptor settings |
| `bondorder:` | Bond-order descriptor settings |
| `shellfile:` | External shell provider (`pottype shellfile`); see @ref reference_potential_types |
| `descriptor_linear:` | Linear descriptor potential (`pottype descriptor_linear`) |
| `potN:` | Sub-potential N of `hybrid`/`hybrid_scale` (N = 1, 2, ...) |

Only `npbc_extent` is registered by the GPIO layer itself. The common keys
(`input`, `output`, `input-format`, `output-format`, `index`) are registered by
each command as top-level settings, so they still have no `io:` prefix.

`mtmbasis` is an exception in the current implementation: its generator settings
are connected as top-level keys, not through an `mtmbasis:` prefix.

MaxVol, FPS/CUR, and crowding commands read their builder and selector options
from the `builder:` and `selector:` blocks. Potential and descriptor subsettings
inside the builder use `pot:`, `behler:`, and `bondorder:`.

---

## Relaxation (`relax:`)

Main configuration for `relax`.

### Basic Control

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax:driver` | str | `lbfgs` | Optimization algorithm: `lbfgs`, `lbfgspp`, `fire`, `cg`, `gd`, `mdmin`, `quickmin`, `sqp`, `nlc` |
| `relax:maxiteration` | int | `1000` | Maximum number of optimization iterations |
| `relax:constraint_loops` | int | `5` | Maximum outer iterations when applying constraints |
| `relax:stepmax` | float | `0.1` | Maximum allowable step per variable update |
| `relax:pressure` | float | 0.0 | External hydrostatic pressure (GPa) |
| `relax:external_stress` | str | — | External stress tensor, 3 or 6 components (GPa) |
| `relax:stiffness` | str | — | Stiffness for strain modes, 6 components (GPa) |

### Geometry Control

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax:relax_atom` | bool | yes | Whether to relax atomic positions (maskable) |
| `relax:relax_cell` | bool | no | Whether to relax cell parameters (maskable) |
| `relax:fix_vol` | bool | no | Fix volume (conflicts with constraints) |
| `relax:fix_shape` | bool | no | Fix cell shape while relaxing volume |
| `relax:cell_strain` | bool | yes | Use strain-based cell relaxation (faster, more stable) |
| `relax:symm_strain` | bool | yes | Use symmetric strain tensor to prevent rigid rotation |
| `relax:atomic_force_recast` | bool | yes | Recast forces for strain consistency |
| `relax:selective_dynamics` | bool | no | Enable selective dynamics filter |
| `relax:cartesian` | bool | no | Use Cartesian (yes) or fractional (no) coordinates |
| `relax:car_mode` | bool | no | Alias for cartesian |
| `relax:couple` | int | — | Lattice vector coupling mode: `0` none, `1` a-b, `2` b-c, `3` c-a, `4` all |
| `relax:strain_mask` | str | — | Mask for lattice/strain components (6 values in strain mode, 9 in lattice mode; 1=relax 0=fix) |
| `relax:lattice_mask` | str | — | Alias for strain_mask |

### Convergence Tolerances

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax:gtol` | float | `1e-3` | Gradient tolerance — stops when max gradient falls below |
| `relax:ftol` | float | `0` | Energy tolerance — stops when energy change falls below |
| `relax:xtol` | float | `0` | Step tolerance — stops when displacement change falls below |
| `relax:ztol` | float | `1e-6` | Zero threshold for freezing small gradients |
| `relax:eps` | float | `1e-6` | Machine epsilon to prevent division by zero |

### Constraints & Filters

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax:with_constraint_aul` | bool | no | Augmented Lagrangian method |
| `relax:with_constraint_sqp` | bool | no | Sequential Quadratic Programming |
| `relax:with_constraint_nlc` | bool | no | Nonlinear constraints |
| `relax:with_constraint_lc` | bool | no | Linear constraints filter |
| `relax:with_constraint_vib` | bool | no | VIB2 filter (transforms coordinates) |
| `relax:vibffilter` | bool | no | VIB filter (no coordinate transform) |
| `relax:with_linear_constraint` | bool | no | Linear constraint (LC) filter |
| `relax:with_linear_transform` | bool | no | Linear transform (LF) filter |
| `relax:fixcm` | bool | no | Fix center of mass (adjusts gradients only) |
| `relax:freeze_small_gd` | bool | no | Zero gradients below threshold |
| `relax:symm_force` | bool | no | Symmetrize forces/stress via space group |

### Constraint Files

| Key | Type | Description |
|-----|------|-------------|
| `relax:lc_file` | path | Linear constraints definition file |
| `relax:lf_file` | path | Linear transforms definition file |
| `relax:vib_file` | path | Vibrational normal modes file |
| `relax:seldyn_file` | path | Selective dynamics config file |
| `relax:precond_file` | path | Preconditioning config file |

### Trajectory Output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax:write_traj` | bool | no | Write optimization trajectory |
| `relax:traj_append` | bool | no | Append to existing trajectory file |
| `relax:trajfile` | path | `relax-trajectory.xyz` | Output trajectory filename |
| `relax:nblock` | int | `10` | Write snapshot every N iterations |

### Initial Conditions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `relax:apply_disp` | bool | no | Apply random initial displacement |
| `relax:disp` | float | `0.01` | Displacement magnitude |
| `relax:disp_gaussian` | bool | no | Gaussian distribution for displacement |

### Driver-Specific

All driver-specific parameters use the `relax:` prefix (the bare prefix shown is shorthand):

| Key | Default | Description |
|-----|---------|-------------|
| `relax:bfgs_m` | `15` | L-BFGS memory parameter |
| `relax:gd:step_max_init_cell` | `0.1` | GD: initial max step for cell |
| `relax:gd:step_upscale` | `1.2` | GD: factor to upscale step |
| `relax:gd:step_downscale` | `0.2` | GD: factor to downscale step |
| `relax:fire:ts_max` | `10.0` | FIRE: max timestep |
| `relax:fire:ts_min` | `0.001` | FIRE: min timestep |
| `relax:fire:ts_init` | `2.0` | FIRE: initial timestep |
| `relax:fire:alpha_start` | `0.25` | FIRE: initial damping |
| `relax:fire:alpha_reduce` | `0.99` | FIRE: damping reduction factor |
| `relax:fire:ts_grow` | `1.1` | FIRE: timestep growth factor |
| `relax:fire:ts_reduce` | `0.5` | FIRE: timestep reduction factor |
| `relax:fire:ndelay` | `20` | FIRE: number of delay steps |
| `relax:fire:do_backward` | `true` | FIRE: enable step backtracking |
| `relax:fire:initial_delay` | `true` | FIRE: enable initial delay |
| `relax:fire:downhill_check` | `false` | FIRE: check downhill condition |
| `relax:lbfgspp:epsilon` | see LBFGSpp | LBFGSpp: gradient tolerance |
| `relax:lbfgspp:past` | see LBFGSpp | LBFGSpp: past steps for convergence |
| `relax:lbfgspp:delta` | see LBFGSpp | LBFGSpp: delta for convergence |
| `relax:lbfgspp:max_iterations` | see LBFGSpp | LBFGSpp: max iterations |
| `relax:lbfgspp:epsilon_rel` | see LBFGSpp | LBFGSpp: relative gradient tolerance |
| `relax:lbfgspp:linesearch` | see LBFGSpp | LBFGSpp: line-search algorithm |
| `relax:lbfgspp:max_linesearch` | see LBFGSpp | LBFGSpp: maximum line-search trials |
| `relax:lbfgspp:min_step` | see LBFGSpp | LBFGSpp: minimum line-search step |
| `relax:lbfgspp:max_step` | see LBFGSpp | LBFGSpp: maximum line-search step |
| `relax:lbfgspp:ftol` | see LBFGSpp | LBFGSpp: Armijo parameter |
| `relax:lbfgspp:wolfe` | see LBFGSpp | LBFGSpp: Wolfe parameter |
| `relax:lbfgspp:initial_step` | see LBFGSpp | LBFGSpp: initial line-search step |
| `relax:lbfgspp:damping` | see LBFGSpp | LBFGSpp: curvature damping |
| `relax:lbfgspp:first_initial_step` | see LBFGSpp | LBFGSpp: first initial step |
| `relax:lbfgspp:first_step` | see LBFGSpp | LBFGSpp: first step |

Additional keys accepted by the relaxation drivers:

| Key | Default | Description |
|-----|---------|-------------|
| `relax:periodic_subspace` | no | Use mixed periodic-fractional/open-Cartesian coordinates; cell changes affect the periodic Gram matrix only |
| `relax:dprecond` | `0` | Diagonal preconditioning intensity (levels above 9 are hidden) |
| `relax:force_switchoff_scaling` | — | Scale forces smoothly near the switching region |
| `relax:keep_symm` | no | Preserve the initial space-group symmetry |
| `relax:cell_factor` | — | Scale factor applied to cell-coordinate steps |
| `relax:hessian_precond_file` | — | Hessian file used as a preconditioner |
| `relax:sparse_precond` | no | Use a sparse preconditioner |
| `relax:sparse_shift` | — | Shift added to the sparse preconditioner diagonal |
| `relax:precond_floor` | — | Lower bound for preconditioner eigenvalues |
| `relax:fc_update_interval` | — | Iterations between force-constant preconditioner updates |
| `relax:aul_auto` | — | Automatic adaptive-step control |
| `relax:thin_cell_scaling` | — | Step scaling for thin/flat cells |
| `relax:symmetry_operations` | — | Explicit symmetry operations for constrained relaxation |
| `relax:symmetry_sites` | — | Symmetry site mapping |
| `relax:rigid_groups` | — | Atom groups kept rigid |
| `relax:gtol0` | — | Initial gradient tolerance for staged convergence |
| `relax:first_step` | — | First step size override |
| `relax:index` | `index.map` | Relaxation-local species map (top-level `index` also applies) |

---

## Training (`fit:`)

Parameters for `trainmtp`, `trainnlpot`, and `train_linear`. All keys below use the `fit:` prefix in the settings file.

The tables cover several trainers; a key is effective only when the selected
trainer exposes it. GPU, staged schedules, VarPro and conditional linear-refit
controls belong to `trainnlpot`. Use `imrmlp <command> --help` to inspect that
command's connected settings. Values below are defaults, not a complete
training recipe; choose label weights and convergence tolerances for your data.

@note The GPU keys (`fit:device`, `fit:gpu_id`, and the `fit:gpu_*` keys below)
apply to `trainnlpot` only; `trainmtp` and `train_linear` run on the CPU.

### Dataset & I/O

| Key | Type | Description |
|-----|------|-------------|
| `fit:save_to` | path | Base path for accepted-iteration coefficient checkpoints (`trainnlpot` appends `.iter-NNNNNN.mtp`) |
| `fit:save_every` | int | Save a coefficient checkpoint every N accepted optimizer iterations |
| `fit:lfit_eng_only` | bool | Linear fit uses energy only |
| `fit:lfit_fastalgo` | bool | Use fast linear fitting algorithm |
| `fit:device` | `cpu` (default), `cuda`, `hip`, or `dcu` | Training backend; `dcu` aliases `hip`. GPU supports compatible ordinary MTP, MTP-M and SOG models; see the [GPU guide](../development/cuda.md) for restrictions |
| `fit:gpu_id` | int | GPU device id; `-1` maps the node-local MPI rank automatically |

### GPU Training

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit:gpu_batch_atoms` | int | 0 | Maximum atoms per structure-preserving GPU chunk; `0` selects automatically |
| `fit:gpu_batch_structures` | int | 0 | Maximum structures per GPU chunk; `0` selects automatically |
| `fit:gpu_memory_fraction` | float | 0.75 | Fraction of currently free device memory available to a chunk |
| `fit:gpu_device_loss` | bool | true | Build E/F/S residuals, metrics, and loss adjoints on the device |
| `fit:gpu_deterministic_reduction` | bool | false | Use ordered, reproducible GPU reductions for diagnostics |
| `fit:gpu_radial_parameter_tile` | int | 128 | Number of radial parameters in each partial-gradient tile |
| `fit:gpu_cache_batches` | bool | true | Cache fixed full-dataset packed batches and labels on the host across optimizer callbacks |

The cache avoids rebuilding CSR topology and labels during repeated full-data
callbacks, at the cost of host memory proportional to the packed dataset.
Shuffled or partial SGD selections bypass the full-dataset cache. One MPI rank
is assigned to each GPU; gradients are reduced by the existing MPI path.

### Loss Weights

Energy, force and stress weights are listed below. HVP labels and their
training weights are described in the [HVP guide](../advanced/hvp-training.md).

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit:energy_weight` | float | 1.0 | Energy loss weight |
| `fit:force_weight` | float | 0.01 | Force loss weight |
| `fit:stress_weight` | float | 0.0001 | Stress loss weight |
| `fit:energy_pow` | float | 1.0 | Energy weight power scaling |
| `fit:force_pow` | float | 0.0 | Force weight power scaling |
| `fit:stress_pow` | float | 1.0 | Stress weight power scaling |
| `fit:penalty_weight` | float | 1e-6 | Orthogonality penalty weight |
| `fit:force_wgt_d` | float | 0.1 | Force weight normalization denominator d |
| `fit:force_wgt_f` | float | 0 | Force weight normalization factor f |
| `fit:force_wgt_p` | float | 1.0 | Force weight normalization power p |

### Regularization

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit:weight_L1` | float | 0 | L1 regularization weight |
| `fit:weight_L2` | float | 1e-10 | L2 regularization weight |
| `fit:weight_rad_w0` | float | 1e-8 | Radial smoothness: 0th derivative |
| `fit:weight_rad_w1` | float | 0 | Radial smoothness: 1st derivative |
| `fit:weight_rad_w2` | float | 0 | Radial smoothness: 2nd derivative |
| `fit:radial_reg_type` | int | 0 | Radial regularization type |
| `fit:quad_count` | int | 100 | Quadrature points for radial integral |
| `fit:safe_fit` | bool | true | Safe (slower) linear fitting mode |

### Initialization

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit:random_init` | bool | no | Randomly initialize coefficients |
| `fit:rand_low` | float | -0.1 | Random init: lower bound |
| `fit:rand_up` | float | 0.1 | Random init: upper bound |
| `fit:force_random` | bool | no | Force random initialization for the generic nonlinear trainer (`trainnlpot`) |
| `fit:force_randomlinear` | bool | no | Force random init for linear coefficients |
| `fit:force_randomradial` | bool | no | Force random init for radial coefficients |
| `fit:skip_preinit` | bool | no | Skip pre-initialization pass |
| `fit:pre_init_iter` | int | 75 | Number of pre-initialization iterations |
| `fit:apply_disp` | bool | no | Apply displacement to training data |
| `fit:disp_sigma` | float | 1e-6 | Displacement sigma |

### SGD Settings

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit:learning_rate` | float | 0.0001 | Initial learning rate |
| `fit:momentum` | float | 0.1 | Momentum coefficient |
| `fit:lr_scheduler` | str | none | LR schedule: `none`, `exponential`, `step`, or `cosine` |
| `fit:lr_decay_rate` | float | 0.95 | `step`: multiplicative factor; `exponential`: rate in `exp(-rate * t)`, with `t` counting mini-batch updates |
| `fit:lr_decay_steps` | int | 10 | Mini-batch updates per decay interval for `step` only |
| `fit:lr_final` | float | 0.0 | Final learning rate floor |
| `fit:lr_warmup_steps` | int | 0 | Optional linear warmup steps applied around the selected scheduler |
| `fit:lr_warmup_init` | float | 0.0 | Initial LR for optional warmup |
| `fit:batch_size` | int | 100 | SGD mini-batch size |
| `fit:epoch_limit` | int | 100 | Maximum training epochs |
| `fit:clip_gradient` | float | 0.0 | Gradient clipping threshold (0 = disabled) |

### BFGS-Specific

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `fit:bfgs_m` | int | 5 | L-BFGS memory (number of saved steps) |
| `fit:bfgs_conv_tol` | float | 1e-3 | BFGS convergence tolerance (function change) |
| `fit:bfgs_grad_tol` | float | 1e-3 | BFGS gradient norm tolerance |
| `fit:bfgs_step_tol` | float | 1e-3 | BFGS step size tolerance |
| `fit:bfgs_maxiteration` | int | 1000 | BFGS maximum iterations |
| `fit:bfgs_iteration_offset` | int | 0 | Add this offset to accepted-iteration logs/checkpoint names after a coefficient restart; ALGLIB history is reset |
| `fit:linear_fit_bfgs` | bool | no | Refit at iterations 25, 100, 200 of each ALGLIB BFGS stage and at training end; ALGLIB counts accepted iterations across refit restarts |
| `fit:lfit_eng_only` | bool | no | Linear fit uses energy only |
| `fit:lfit_fastalgo` | bool | yes | Use fast linear fitting algorithm |
| `fit:bfgs_rescale` | bool | no | Rescale variables after BFGS (`trainmtp`) |
| `fit:set_scale` | bool | no | Set variable scaling from external file |
| `fit:precondition` | bool | no | Use diagonal Hessian preconditioning |
| `fit:fix_linear_coeffs` | bool | no | Keep linear MTP coefficients fixed (`trainmtp`) |
| `fit:fix_species_coeffs` | bool | no | Keep species coefficients fixed (`trainmtp`) |
| `fit:fix_radial_coeffs` | bool | no | Keep radial coefficients fixed (`trainmtp`) |
| `fit:do_orthogonalization` | bool | yes | Orthogonalize the MTP basis during training (`trainmtp`) |


### LBFGS++ Nested Keys

`trainmtp` also connects LBFGS++ controls as nested keys under `[fit]`. With
bracket syntax, write them as `lbfgspp:epsilon`, `lbfgspp:maxit`, etc. Direct
prefix syntax uses the full key, for example `fit:lbfgspp:epsilon`.

| Key | Description |
|-----|-------------|
| `fit:lbfgspp:epsilon` | Absolute gradient tolerance |
| `fit:lbfgspp:epsgmax` | Maximum gradient component tolerance |
| `fit:lbfgspp:epsrel` | Relative gradient tolerance |
| `fit:lbfgspp:m` | LBFGS memory |
| `fit:lbfgspp:damping` | Damping factor |
| `fit:lbfgspp:init_step` | Initial line-search step |
| `fit:lbfgspp:first_step` | First initial step |
| `fit:lbfgspp:past` | Number of previous iterations used for convergence |
| `fit:lbfgspp:delta` | Function-change convergence threshold |
| `fit:lbfgspp:maxit` | Maximum LBFGS++ iterations |
| `fit:lbfgspp:linesearch` | Line-search algorithm selector |
| `fit:lbfgspp:max_linesearch` | Maximum line-search trials |
| `fit:lbfgspp:min_step` | Minimum line-search step |
| `fit:lbfgspp:max_step` | Maximum line-search step |
| `fit:lbfgspp:ftol` | Armijo condition parameter |
| `fit:lbfgspp:wolfe` | Wolfe condition parameter |
| `fit:lbfgspp:f_dec_coeff` | Function decrease coefficient |
| `fit:lbfgspp:s_curv_coeff` | Curvature coefficient |
| `fit:lbfgspp:cautious_factor` | Cautious update factor |
| `fit:lbfgspp:max_move` | Maximum coordinate-space move |

Additional `fit:` keys read by the trainers:

| Key | Default | Description |
|-----|---------|-------------|
| `fit:do_ladder` | no | Enable the radial-fit ladder schedule |
| `fit:ladder_cut` | — | Ladder cutoff level |
| `fit:ladder_last` | — | Last ladder stage |
| `fit:popsize` | — | Differential-evolution population size |
| `fit:df_orthogonal_constraints` | — | Orthogonal constraints for differential evolution |
| `fit:linear_precondition` | — | Linear-fit preconditioning mode |
| `fit:rescale_mode` | — | Coefficient rescaling mode |
| `fit:hvp_weight` | `0` | Hessian-vector label weight |
| `fit:hvp_pow` | `0` | HVP atom-count power |
| `fit:hvp_fd_step` | `1e-4` | HVP finite-difference step |
| `fit:hessian_weight` | `0` | Dense Hessian label weight |
| `fit:hessian_pow` | `0` | Hessian atom-count power |
| `fit:hessian_fd_step` | `1e-4` | Hessian finite-difference step |
| `fit:objective-convention` | `legacy` | `legacy` or shared mean-moments objective |
| `fit:l2-reference` | `native` | `native`, `mean`, or `sum` |
| `fit:objective-report` | — | Optional TSV objective report |
| `fit:validation_interval` | `1` | Iterations between validation evaluations |
| `fit:charge_weight` | `0` | Charge-label weight |
| `fit:charge_pow` | `1` | Charge atom-count power |
| `fit:charge_target_attrib` | `total_charge` | Attribute holding the charge target |
| `fit:charge_default_target` | `0` | Fallback charge target |
| `fit:training_schedule` | `single` | `single`, `mtp_les_from_scratch`, `mtp_les_resume`, `mtp_sog_from_scratch`, or `mtp_sog_resume`; `trainnlpot` only |
| `fit:varpro_report` | `varpro.jsonl` | VarPro report path |
| `fit:varpro_radial_scale_file` | — | Radial scale file for VarPro |
| `fit:varpro_linear_refinement_steps` | `2` | Linear refinement steps inside VarPro |
| `fit:varpro_linear_gradient_tolerance` | `1e-7` | Linear gradient tolerance inside VarPro |
| `fit:validation_metric` | `force` | Metric used to rank validation results |
| `fit:refit_train_valid` | no | Refit on the combined train+valid set |
| `fit:regularization_search` | no | Enable regularization-multiplier search |
| `fit:regularization_multipliers` | — | Candidate regularization multipliers |
| `fit:validation_max_energy` | `-1` | Validation energy ceiling |
| `fit:validation_max_force` | `-1` | Validation force ceiling |
| `fit:validation_max_stress` | `-1` | Validation stress ceiling |
| `fit:linear_refit` | `off` | Disable refitting, or select affine blocks: `sr`, `sr_sog`, `head`, `all`, or model-advertised block names; see [linear refinement](../advanced/linear-refit.md) |
| `fit:linear_refit_solver` | `qr_svd` | Linear refit solver |
| `fit:linear_refit_report` | `linear-refit.json` | Linear refit report path |
| `fit:linear_refit_rtol` | `1e-12` | Relative threshold for numerical rank |
| `fit:linear_refit_atol` | `0` | Absolute threshold for numerical rank |
| `fit:linear_refit_tolerance` | `1e-10` | Relative tolerance for objective verification |
| `fit:trainable_groups` | — | Named parameter groups exposed to the optimizer |
| `fit:sr_stage_iterations` | `100` | Short-range stage iterations |
| `fit:les_stage_iterations` | `50` | LES stage iterations |
| `fit:joint_stage_iterations` | `0` | Joint stage iterations |
| `fit:charge_warmup_factor` | `10` | Charge warm-up multiplier |
| `fit:les_zero_init_range` | `1e-3` | LES zero-initialization range |
| `fit:sog_zero_init_range` | `0.1` | SOG zero-initialization range |
| `fit:les_zero_init_seed` | `5489` | LES initialization seed |
| `fit:curvature_precondition` | — | Curvature preconditioning mode |
| `fit:gn_probes` | — | Gauss-Newton probe count |
| `fit:gn_update` | — | Gauss-Newton update interval |
| `fit:gn_seed` | — | Gauss-Newton probe seed |
| `fit:gn_floor` | — | Gauss-Newton eigenvalue floor |
| `fit:parameter_scaling` | `none` | Per-parameter scaling mode |
| `fit:scale_floor` | `1e-3` | Lower bound for parameter scales |
| `fit:scale_ceiling` | `0.1` | Upper bound for parameter scales |
| `fit:scaled_step_max` | `0` | Maximum step in scaled coordinates |

---

## Active Learning (`al:`)

`relax` uses `ALPotential` and `ALModifier` to monitor extrapolation against a
saved MaxVol active set. Evaluations above `threshold` enter the candidate pool;
those above `threshold_break` are collected and stop that structure's relaxation.
The pool is sparsified and exported after processing the input structures.
Reference labeling and model retraining are separate steps.

| Key | Type | Default | Description |
|---|---|---|---|
| `al:mode` | str | `none` | `maxvol` enables monitoring; `none` disables it |
| `al:state_file` | path | required for MaxVol | Saved active set (`.sel`, v10) |
| `al:index_file` | path | — | Optional species index map for the scorer |
| `al:threshold` | float | 2.01 | Grade above which a candidate is collected |
| `al:threshold_break` | float | 100.0 | Grade above which the caller stops relaxation |
| `al:interval` | int | 1 | Check every N evaluations; must be positive |
| `al:export_file` | path | `al_candidates.xyz` | Candidate output file |
| `al:sparsify_to` | int | 1000 | Maximum exported pool size; nonpositive clears the pool |
| `al:sparsify_method` | str | `fps` | `fps` or `random` |
| `al:source` | str | `external` | `external` active set or `embedded` GPM session |
| `al:committee_prefix` | str | — | Prefix for committee-member model files |

Add this section to an ordinary relaxation settings file and run
`imrmlp relax relax.ini`:

```text
[al]
mode            maxvol
state_file      active.sel
threshold       2.01
threshold_break 100
export_file     candidates.xyz
```

---

## NEB (`neb:`)

Nudged Elastic Band parameters for `calc_neb`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `neb:spring` | float | `5.0` | Spring constant between images |
| `neb:climb` | bool | no | Enable climbing image NEB (CI-NEB) |
| `neb:climb_mode` | string | `highest` | Select the highest image (standard CI-NEB) or `all_local` maxima |
| `neb:climb_delay` | int | `0` | Delay CI-NEB by N accepted steps with `stable-v1`, or N force evaluations with `legacy` |
| `neb:vcneb` | bool | no | Enable variable-cell NEB |
| `neb:variablespring` | bool | no | Variable spring constants |
| `neb:iref` | int | `-1` | Reference cell: -1=final, -2=average |
| `neb:selective_dynamics` | bool | no | Selective dynamics |
| `neb:cmask` | str | — | Cell parameter mask (9 components) |
| `neb:symm_strain` | bool | yes | Symmetric strain tensor |
| `neb:recast_atomic_force` | bool | yes | Recast forces for VC-NEB with symm_strain |
| `neb:pressure` | float | 0.0 | External pressure for VC-NEB |
| `neb:write_traj` | bool | no | Write optimization trajectory |
| `neb:trjfile` | path | — | Trajectory filename |
| `neb:trjformat` | str | `mlpcfg` | Trajectory file format |
| `neb:nblocks` | int | `5` | Trajectory interval in accepted steps with `stable-v1`, or force evaluations with `legacy` |
| `neb:cellweight` | float | 100.0 | Cell-coordinate metric weight used by VC-NEB tangents, springs, and forces |
| `neb:min_images` | int | 5 | Minimum total image count, including endpoints |
| `neb:max_images` | int | 20 | Maximum total image count, including endpoints |
| `neb:solver_policy` | str | `legacy` | Numerical policy: `legacy` or `stable-v1` |
| `neb:max_cell_strain` | float | — | Maximum cell strain per step |
| `neb:max_force_evaluations` | int | — | Cap on force evaluations |
| `neb:diagnostics` | path | empty | Optional accepted-state diagnostic CSV filename |
| `neb:vally` / `neb:valley` | bool | no | Enable valley-following path correction |
| `neb:pin` | bool | no | Pin climbing and valley images |
| `neb:ciival` | 6 floats | `100 100 100 50 50 50` | Positive cell-coordinate scaling weights in `xx yy zz yz xz xy` order |
| `neb:apply_scal` | bool | — | Apply scalar path correction |
| `neb:omp_mode` | str | — | OpenMP scheduling mode for force evaluation |
| `neb:check_mic` | bool | yes | Check minimum-image convention |
| `neb:restore_F_cholesky` | bool | — | Restore forces in the Cholesky frame |
| `neb:periodic_subspace` | bool | no | Use mixed periodic/open coordinates for partial-PBC variable-cell NEB |
| `neb:keep_symm` | bool | no | Preserve initial symmetry along the path |
| `neb:symprec` | float | `1e-5` | Symmetry tolerance |
| `neb:symmetry_operations` | — | — | Explicit symmetry operations |
| `neb:symmetry_diagnostics` | path | empty | Optional symmetry report filename |
| `neb:perturb_amplitude` | float | — | Path perturbation amplitude |
| `neb:perturb_seed` | int | — | Path perturbation seed |
| `neb:symmetry_release` | path | empty | New output directory for a post-run symmetry release check; empty disables it |
| `neb:symmetry_release_trials` | int | — | Symmetry-release trials |
| `neb:symmetry_release_seed` | int | — | Symmetry-release seed |
| `neb:symmetry_release_steps` | int | — | Symmetry-release optimizer steps |
| `neb:symmetry_release_evaluations` | int | — | Symmetry-release force evaluations |
| `neb:symmetry_release_amplitude` | float | — | Symmetry-release displacement amplitude |
| `neb:symmetry_release_strain` | float | — | Symmetry-release cell strain |
| `neb:symmetry_release_energy_tol` | float | — | Symmetry-release energy tolerance |
| `neb:traj` | path | — | Alias of `neb:trjfile` |
| `neb:traj-format` | str | — | Alias of `neb:trjformat` |
| `neb:springmax` | float | — | Maximum spring constant |
| `neb:springmin` | float | — | Minimum spring constant |
| `neb:variableimage` | bool | no | Insert/remove images during optimization |
| `neb:img_insert_thresh` | float | — | Variable-image insertion threshold |
| `neb:img_remove_thresh` | float | — | Variable-image removal threshold |
| `neb:img_adjust_interval` | int | — | Iterations between image adjustments |
| `neb:ttype` | int | 1 | Tangent type: 1 improved, 2 original with safeguards, 3 EB |
| `neb:stype` | int | 1 | Spring projection type; usually matches `ttype` |
| `neb:mdist_r` | float | — | Minimum-distance radius used by the path filter |
| `neb:mdist_factor` | float | — | Minimum-distance penalty factor |
| `neb:mdist_power` | float | — | Minimum-distance penalty power |
| `neb:apply_mindist` | bool | — | Apply the minimum-distance penalty |
| `neb:init_method` | str | — | Path initialization method |
| `neb:idpp_fmax` | float | — | IDPP force convergence target |
| `neb:idpp_max_steps` | int | — | IDPP iteration limit |
| `neb:neb_ftol` | float | — | NEB force tolerance |
| `neb:gtol` | float | — | Alias of `neb:neb_ftol` |
| `neb:check_saddle` | bool | no | Verify the saddle after convergence |
| `neb:debug` | int | 0 | Debug output level |

Top-level `cellweight` and `--cellweight`/`-cwt` are deprecated. Use `neb:cellweight`; if both are supplied, the top-level value takes precedence and emits a warning.


---

## Potential (`pot:`)

Potential settings used by `PotMTP` and other potentials. These keys are
configured under the `[pot]` section of the settings file.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pot:potfile` | path | — | Potential file to load |
| `pot:pottype` | str | `mtp` | Potential type identifier. See @ref reference_potential_types for all 21 types and file formats: `mtp`, `mtpnn1`, `mtp_m`, `hmtp`, `shellfile`, `eamalloy`, `eamfs`, `eampoly`, `openkim`, `acelmp`, `mindist`, `lj612`, `zbl`, `mtp_les`, `descriptor_linear`, `pair_table`, `threebody_table`, `sw`, `tersoff`, `hybrid`, `hybrid_scale` |
| `pot:radial_split` | bool | no | Use separate radial basis per species pair |
| `pot:typemap` | str | — | Explicit element ordering for the potential |
| `pot:kim_model` | str | — | OpenKIM model name for `pottype openkim` |
| `pot:potlevel` | int | — | Number of sub-potentials actually used by `hmtp` |
| `pot:radial_local_cache` | bool | — | Cache local radial values |
| `pot:full_symmetry_reuse` | bool | — | Reuse the full symmetry expansion between evaluations |
| `pot:cutoff` | float | — | Cutoff for `mindist` |
| `pot:factor` | float | — | Minimum-distance penalty factor for `mindist` |
| `pot:power` | float | — | Minimum-distance penalty power for `mindist` |
| `pot:les_kernel` | str | — | LES kernel family |
| `pot:les_recurrence` | — | — | LES recurrence mode |
| `pot:les_cache` | bool | — | Cache LES intermediates |
| `pot:sog_terms` | int | — | Number of SOG terms |
| `pot:sog_channels` | int | — | Number of SOG channels |
| `pot:sog_log_bandwidth_min` | float | — | Lower bound of log-bandwidth |
| `pot:sog_log_bandwidth_max` | float | — | Upper bound of log-bandwidth |
| `pot:lr_sigma` | float | — | Long-range smearing width |
| `pot:les_dl` | float | — | LES finite-difference step |
| `pot:les_taper_fraction` | float | — | LES taper fraction |
| `pot:les_cutoff_mode` | str | — | LES cutoff mode |
| `pot:les_init_range` | float | — | LES initialization range |
| `pot:les_balance_initial_charges` | bool | — | Balance initial charges |
| `pot:les_hessian_fd_nudge` | float | — | Hessian finite-difference nudge |
| `pot:remove_self_interaction` | bool | — | Remove long-range self-interaction |
| `pot:les_response_parity` | str | — | Response parity convention |

@note `level`, `species_count`, `min_dist`, `max_dist`, `radial_basis_size`, and
`radial_basis_type` are tags of the `.mtp` file format, not settings keys — setting
them in an `.ini` file has no effect; they are read from the potential file itself.

---

## Moment Tensor Basis Generation

Settings for `imrmlp mtmbasis`. These keys are read at the top level of the
settings file.

### Pool Generation

The generator enumerates moment tensor products whose weighted level is

```text
level = mu_wei * sum(mu_i) + nv_wei * sum(nu_i) + dim_wei * dim
```

and keeps entries with `level <= max_lev`. `max_dim` is also clamped internally
to `min(max_dim, max_lev / 2)`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `max_lev` | int | 16 | Maximum weighted level for generated scalar invariants |
| `max_dim` | int | 4 | Maximum number of moment tensors in a scalar invariant before the internal clamp |
| `max_mu` | int | 4 | Maximum radial index `mu` when `--moments` is not supplied |
| `max_nv` | int | 10 | Maximum tensor rank `nu` when `--moments` is not supplied |
| `max_poly` | int | 10000 | Maximum sum of tensor ranks in a scalar invariant |
| `seed` | int | — | Random seed for the GA contraction search |
| `elite_count` | int | — | Number of elite individuals kept per generation |
| `initial_population` | int | — | Initial GA population size |
| `mu_wei` | float | 2 | Weight of `mu` in the level formula |
| `nv_wei` | float | 1 | Weight of `nu` in the level formula |
| `dim_wei` | float | 2 | Weight of scalar dimension in the level formula |
| `kselect` | int | 2 | Number of low-dimension decomposition groups retained while building contraction candidates |
| `add_new_scalar_to_pot` | bool | yes | Add newly discovered scalar intermediates to the generated moment list |
| `sample_rates` | str | — | Whitespace-separated rates used by `--advanced-sample`; values map to dimensions 2, 3, ... |

When `--moments`/`-mts` is supplied, the moment-pair file starts with a count and
then lists that many `mu nu` pairs. When it is omitted, all pairs
`0 <= mu <= max_mu` and `0 <= nu <= max_nv` are considered.

The generator enumerates dimensions 1 through 7 in a deterministic canonical
order and removes tensors only when an exact simultaneous vertex permutation
matches both their `mu` labels and contraction matrix. Moment-pair and alpha-pool
inputs are validated strictly. Regenerating a pool with this implementation can
change tensor and moment IDs relative to older hand-enumerated pools; regenerate
restart/select diagnostics together with the pool instead of mixing versions.

### Contraction Search

`mtmbasis` first builds all valid contraction trees, then searches for a compact
layout by minimizing `w1 * moments + w2 * contraction_rules`. Equal weighted
costs prefer fewer contraction rules, then fewer moments. Mode 3 repeats
coordinate sweeps until no branch improves this ordering. Exact per-connection
counts learned from real topology generation may reject non-improving candidates
early; every potential winner is still fully generated and topology-validated.
Exact counting is always enabled.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `mode` | int | 0 | Search mode: 0 simulated annealing-style tempering, 1 random search with local improvement, 2 genetic algorithm, 3 coordinate descent to a local optimum |
| `T_init` | float | 200 | Initial temperature for `mode = 0` |
| `w1` | float | 1.0 | Cost weight for moment count |
| `w2` | float | 4.0 | Cost weight for contraction-rule count |
| `uacc` | float | 0.5 | Acceptance probability for unchanged moves in `mode = 0` |
| `maxiter` | int | 1000000 | Search iteration count |
| `restart` | bool | no | Restart from a previously written selection file when available |
| `randominit` / `rndinit` | bool | no | Use a random initial tree selection |
| `chop_trees_first` | bool | yes | Prune inconsistent trees during initial construction |
| `chop_trees` | bool | yes | Prune inconsistent trees during later search/evaluation |
| `reassign` | bool | no | Reassign selected branches when pruning changes available choices |
| `filter_init` | bool | no | Apply the reuse filter during initial processing |
| `filter_second` | bool | no | Apply the reuse filter during the second processing pass |

### Genetic Algorithm Controls

These keys are used by `mode = 2`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `popsize` | int | 10 | Number of surviving population members |
| `childsize` | int | 30 | Number of child candidates per iteration |
| `kfit` | float | 0.002 | Fitness sigmoid slope |
| `cfit` | float | 0.01 | Fitness sigmoid offset |
| `frac_mutation_min` | float | 0.2 | Lower bound for mutation fraction |
| `frac_mutation_max` | float | 1.0 | Upper bound for mutation fraction |
| `max_mutation_sites` | int | 4 | Maximum number of gene positions mutated at once |
| `min_ex_frac` | float | 0.1 | Minimum crossover exchange fraction |
| `max_ex_frac` | float | 0.5 | Maximum crossover exchange fraction |
| `rstart` | int | 0 | First gene index included in restricted searches |
| `rstop` | int | -1 | Last gene index included in restricted searches; -1 means the end |

Typical minimal file:

```text
max_lev 16
max_dim 4
max_mu 4
max_nv 10

mu_wei 2
nv_wei 1
dim_wei 2

mode 2
popsize 10
childsize 30
maxiter 1000000
```

---

## I/O

General-purpose I/O settings. These are top-level keys (no `io:` prefix):
`npbc_extent` is registered by `GPIO::InitSettings`, while `input`, `output`,
`input-format`, `output-format`, and `index` are registered by each command.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `input` | path | — | Input file (format auto-detected) |
| `output` | path | — | Output file |
| `input-format` | str | — | Force input format |
| `output-format` | str | — | Force output format |
| `npbc_extent` | float | 5.0 | Cell padding for non-PBC structures |
| `index` | path | — | Species index file |

---

## Hybrid Potential

Two composition modes are available:

- `pottype = hybrid` adds the EFS results from its sub-potentials.
- `pottype = hybrid_scale` blends MTP or EAMPoly site energies using a local
  feature and smooth, normalized transition weights.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `npot` | int | — | Number of sub-potentials to combine |
| `spcount` | int | inferred | Number of species; must match every sub-potential |
| `potN:pottype` | str | — | Type of sub-potential `N` |
| `potN:potfile` | path | — | Parameter file for sub-potential `N` |
| `feature` | str | `smooth_coordination` | `hybrid_scale` feature: `smooth_coordination` or `min_distance` |
| `transition` | str | `smoothstep` | Quintic smooth transition |
| `feature_r0` | float | 0 | Inner radius of the smooth-coordination feature |
| `feature_r1` | float | combined cutoff | Outer radius of the smooth-coordination feature |
| `boundaryN` | 2 floats | — | Increasing feature interval transitioning from potential `N` onward |

For three or more sub-potentials, ordered boundaries may overlap. Sequential
stick-breaking weights keep all weights non-negative and normalized. During
`trainnlpot`, `hybrid_scale` concatenates all child parameters into groups such
as `pot1/all`; transition boundaries and feature radii remain fixed. Saving the
trained potential writes a manifest plus adjacent `.potN` child files, and the
manifest can be loaded through `pot:potfile`.

`hybrid` uses a shared-species MTP parameter layout and exposes it to `trainnlpot` as `radial`, `species`, and `moments`
groups. Fixed ZBL children do not add trainable parameters.

```text
pot:pottype hybrid_scale
pot:npot 2
pot:feature smooth_coordination
pot:feature_r0 0.0
pot:feature_r1 5.0
pot:boundary1 0.3 0.8
pot:pot1:pottype mtp
pot:pot1:potfile short_range.mtp
pot:pot2:pottype eampoly
pot:pot2:potfile background.eampoly
```

---

## Programmatic API

Most classes that use `ConnectSettings` follow this pattern:

```cpp
class MyClass : public ConnectSettings {
    void InitSettings() {
        Connect(param, "section:key", "Description of this parameter");
    }

    void InitBySettings(GPSettings &settings) {
        InitSettings();
        ApplySettings(settings);
    }
};

// Usage:
GPSettings settings;
settings["section:key"] = "value";
myobj.InitBySettings(settings);
```

Loading from file:

```cpp
GPSettings settings;
settings.Load("config.ini");
myobj.InitBySettings(settings);
```

## Example Files

### Full Relaxation Config

```text
# relax.ini
relax:driver lbfgs
relax:maxiteration 500
relax:gtol 1e-4
relax:ftol 1e-6
relax:relax_cell yes
relax:cell_strain yes
relax:symm_strain yes
relax:strain_mask 1 1 1 1 1 1
relax:pressure 0.0
relax:write_traj yes
relax:trajfile relax.traj
relax:nblock 10
relax:bfgs_m 5

pot:potfile potential.mtp

input POSCAR
output relaxed.vasp
index index.map
```

### Full Training Config

```text
# train.ini
trainset train.cfg
validset valid.cfg
trainset-format mlpcfg
validset-format mlpcfg
potfile init.mtp
index index.map

[fit]
energy_weight 1.0
force_weight 1.0
stress_weight 0.1
weight_L2 1e-6
weight_rad_w0 1e-6
bfgs_maxiteration 1000
bfgs_m 5
save_to checkpoint.mtp
save_every 50
```

### NEB Config

```text
# neb.ini
neb:spring 5.0
neb:climb yes
neb:climb_delay 50
neb:vcneb no
neb:symm_strain yes
neb:write_traj yes
neb:trjfile neb_path.cfg
neb:nblocks 10

relax:driver lbfgs
relax:gtol 1e-3
relax:maxiteration 200

pot:potfile potential.mtp
```

## Checkpoint compatibility

MaxVol checkpoint version 2 is no longer accepted; version 3 remains supported.

## See Also

- @ref guide_commands — CLI command reference
- @ref guide_workflows — end-to-end workflow examples
- @ref api_core — C++ API examples
