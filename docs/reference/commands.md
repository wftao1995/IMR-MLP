# Command Reference {#guide_commands}

[中文](commands_zh.md)


@tableofcontents

For a first calculation, prepare inputs with [getting started](../guides/getting-started.md).
[Relaxation](../guides/relaxation.md), [NEB](../guides/neb.md),
[properties](../guides/properties.md) and [selection](../guides/active-learning.md)
provide workflows; this page defines commands and options.

## CLI Structure

```
imrmlp <command> [options] <input> <output>
```

Commands support both **short flags** (`-x val`), **long flags** (`--option=val`),
and **positional arguments** (input/output files). Zero-argument Boolean flags
use `--flag` or `--flag=true`; use `--flag=false` to disable one explicitly.
Some Boolean options require a value (`--option true`). Omission keeps the
listed default, which may be true. Do not append a separate `false` to a
zero-argument flag: it will be parsed as a positional argument.

Run `imrmlp --help` for a list of all commands; `imrmlp <command> --help` for
per-command help.

```bash
imrmlp efs -pot potential.mtp POSCAR out.cfg       # short alias
imrmlp relax relax.ini                              # settings file
```

## General Options

These options work with most commands:

| Option | Description |
|--------|-------------|
| `--help` | Show command-specific help |
| `--version` | Show version information (`imrmlp --version`) |
| `--settings` | Load options from settings file (no `-i` alias; many commands also accept the file as a positional argument). `phonopyfc` is the exception: it uses `--settings-file` / `-i`. |
| `--verbose` | Enable verbose logging (a few commands only) |
| `--index` / `-id` | Species index file (often `index.map`) |

### Compressed Input/Output Files

Structure files (POSCAR, dump/lammpstrj, extxyz, cfg, ...), `.ini` settings
files, and potential files support gzip (`.gz`) compression:

- **Reading**: gzip is detected from the file's magic bytes, with the
  extension as a fallback. `POSCAR.gz`, `dump.gz`, and gzip files without a
  compression extension work with the usual `--input` flags.
- **Writing**: output is compressed when the filename ends in `.gz`
  (e.g. `--output out.xyz.gz`).

Gzip is enabled by default; disable it with `make USE_GZIP=0`. Disabled gzip
is still recognized and rejected, rather than treated as plain text. If you
add `-DUSE_GZIP` manually to CXXFLAGS, also add `-lz` to the link flags.

Zstandard support and its build option have been removed. `.zst` paths and
input recognized as Zstandard are rejected. Decompress existing `.zst` files
externally before using them, or convert them to gzip.

---

## Core EFS & Prediction

The EFS (Energy, Forces, Stress) commands are the foundation of all potential-based work.
They evaluate a trained potential on input structures — the first step in validation,
analysis, or production MD runs. Every trajectory analysis, elastic constant computation,
or relaxation pipeline begins with an EFS evaluation under the hood.

**Typical workflow**: Train a potential → validate with `calc_efs` + `calc_err` →
use in production (relaxation, MD, analysis).

### `calc_efs` — Energy, Forces, Stress *(alias: `efs`)*

Compute energy, forces, and stress tensors for one or more structures. This is the
most fundamental command: it takes a potential file and a set of structures, runs
the potential, and appends the computed EFS properties to the output.

@note **When to use `--energy-only`**: use `-eno` when only the total energy is
needed (e.g., energy-volume curves, convex hull analysis). Skipping force and stress
computation cuts runtime significantly for large systems.

```bash
imrmlp efs -pot potential.mtp --input=input.cfg --output=output.cfg
imrmlp calc_efs --potfile=potential.mtp --input=POSCAR --output=out.cfg
imrmlp calc_efs --potfile=potential.mtp --input=POSCAR --output=out.cfg --device=cuda --gpu-id=0
imrmlp efs --settings=calc_efs.ini
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional, required) |
| `--output` | `-out` | `output-efs.cfg` | Output file with EFS appended (positional, required) |
| `--potfile` | `-pot` | `potential.mtp` | Potential parameter file |
| `--pottype` | `-pt` | `mtp` | Potential type identifier |
| `--device` | | `cpu` | Evaluation backend: `cpu`, `cuda`, or `hip` (`dcu` is an alias for `hip`) |
| `--gpu-id` | | `-1` | GPU device id; `-1` uses the current GPU device |
| `--gpu-batch-atoms` | | `0` | Maximum atoms per GPU chunk; `0` selects a memory-bounded size automatically |
| `--gpu-batch-structures` | | `0` | Maximum structures per GPU chunk; `0` selects automatically |
| `--gpu-memory-fraction` | | `0.75` | Fraction of currently free GPU memory available to each chunk |
| `--gpu-reuse-topology` | | `false` | Reuse exactly compatible consecutive single-structure trajectory topologies |
| `--input-format` | `-if` | auto | Input format (e.g., POSCAR, xyz, cif) |
| `--output-format` | `-of` | auto | Output format (e.g., POSCAR, xyz, cif) |
| `--index` | `-id` | `index.map` | Index file mapping atom types to indices |
| `--no-site-energy` | `-ns` | `false` | Exclude site energy from output (bool flag) |
| `--energy-only` | `-eno` | `false` | Output only energy, skip forces, stress, site energies (bool flag) |
| `--show-progress` | `-sp` | `false` | Report completed structure counts (bool flag) |
| `--settings` | | `input.ini` | Configuration settings file |

@note GPU evaluation requires the matching `BUILD_MODE=cuda` / `mpi-cuda`
or `BUILD_MODE=hip` / `mpi-hip` executable. Supported models include ordinary
MTP and MTP-M (`li`, `q2`, `fs`, `n1`, `n2`), including compatible SOG v3/v4
models. Coulomb LES v2 is not supported by this GPU path. See the
[GPU guide](../development/cuda.md) for model and radial-basis restrictions.
Whole structures are grouped into memory-bounded chunks; a single structure
that exceeds the budget is rejected. With `--gpu-reuse-topology`, compatible
consecutive frames are processed one at a time; set `--gpu-batch-structures`
to 0 or 1. `--gpu-memory-fraction` must be in `(0,1]`.
`--energy-only` skips GPU force and virial evaluation.

### `testpot` — Test Potential

Verify potential correctness by checking force consistency (finite-difference test
against analytical gradients).

```bash
imrmlp testpot --settings=test_settings.ini
imrmlp testpot --settings=settings.ini --input=POSCAR --test-which=0 --nugedvalue=0.001
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | Settings file (positional, required) |
| `--input` | | `POSCAR` | Structure file to test |
| `--format` | | `POSCAR` | Structure file format |
| `--index` | | `index.map` | Index file mapping atom types to indices |
| `--test-which` | | `-1` | Atom index to test (0-based); any negative value tests every atom |
| `--nugedvalue` | | `0.001` | Finite-difference step magnitude (Å) |
| `--test-stress` | | `0` | Set to `1` to also test virial via cell finite differences; requires a value |
| `--stress-nudge` | | `1e-6` | Finite-difference displacement for stress test (Å) |

---

## Relaxation & Optimization

Geometry optimization commands find local minima (relaxation) or first-order saddle points
(transition states) on the potential energy surface. These are the most common production
tasks after potential training.

**Choosing the right method for saddle points:** see the comparison table at the end
of this section.

### `relax` — Structure Relaxation

The flagship geometry optimizer. It uses a modular driver architecture supporting
multiple optimization engines (FIRE, L-BFGS, CG, GD), variable-cell relaxation,
selective dynamics (fix/release per-atom and per-cell-strain), and modifiers
(e.g., `ALModifier` for on-the-fly active learning).

```bash
imrmlp relax relax.ini
```

@note **Why use a settings file?** Relaxation involves many interdependent parameters
(optimizer choice, convergence tolerances, cell relaxation flags, constraint masks,
pressure, etc.). Putting them all in a structured `.ini` file avoids CLI clutter
and makes configurations reproducible. See @ref guide_config for every supported
parameter.

While the settings file is the intended interface, these CLI flags can also override
settings file values directly:
`--potfile`/`-pot`, `--pottype`/`-pt`, `--input`/`-in`,
`--output`/`-out`, `--input-format`/`-if`, `--output-format`/`-of`,
`--index`/`-id`, `--gtol`, `--pressure`/`-p`, `--isif`, `--selective_dynamics`/`-sd`,
`--cell-mask`, `--strain_mask`, `--seldyn`,
`--stepmax`, `--dprecond`, `--write-force`, `--write-stress`, `--write-ateng`,
`--create-input`, `--settings`, `--tags`/`--vals` (add `relax:` settings on the command line),
`--seq-read`/`--seq-write` (stream one frame at a time),
`--position-mask` (global 0/1 atomic flags), `--seldyn`, `--filter`
(optimizer-coordinate mask file), `--file-constraints` (JSON constraint set),
and `--discard-all` (drop extra properties on output).

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | — | Potential file |
| `--pottype` | `-pt` | `mtp` | Potential type |
| `--input` | `-in` | `POSCAR` | Input structure |
| `--output` | `-out` | `output-relaxed.cfg` | Output structure |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | from filename | Output format |
| `--index` | `-id` | `index.map` | Species index map |
| `--settings` | | — | Settings file |
| `--gtol` | | — | Force convergence tolerance |
| `--pressure` | `-p` | `0` | External pressure (GPa) |
| `--isif` | | — | Cell/variable relaxation mode |
| `--selective_dynamics` | `-sd` | `false` | Enable selective dynamics |
| `--seldyn` | | — | Per-atom/per-cell constraint mask |
| `--cell-mask` | | — | Cell-coordinate mask |
| `--strain_mask` | | — | Strain-coordinate mask |
| `--stepmax` | | `0.1` | Maximum step per update |
| `--dprecond` | | `0` | Enable diagonal preconditioning (Boolean) |
| `--write-force` | | `true` | Write forces on output |
| `--write-stress` | | `true` | Write stress on output |
| `--write-ateng` | | `true` | Write per-atom energies on output |
| `--create-input` | | `false` | Create a sample input deck and exit |
| `--tags` | | — | Additional `relax:` settings keys |
| `--vals` | | — | Values paired with `--tags` |
| `--seq-read` | | `false` | Stream input structures one frame at a time |
| `--seq-write` | | `false` | Stream output structures one frame at a time |
| `--position-mask` | | — | Global 0/1 atomic flags |
| `--filter` | | — | Optimizer-coordinate mask file |
| `--file-constraints` | | — | JSON constraint set |
| `--discard-all` | | `false` | Drop extra properties on output |

### `modefollowing` — Mode Following

Saddle point search that follows a specific vibrational mode (eigenvector of the
Hessian) uphill from a minimum. Best when you know which mode leads to the desired
transition state — e.g., a soft phonon mode identified from a prior phonon calculation.

```bash
imrmlp modefollowing --settings=modefollowing.ini --potfile=potential.mtp --input=structure.cfg
imrmlp modefollowing --settings=modefollowing.ini --potfile=potential.mtp --input=structure.cfg --imode=1 --amplitude=0.1
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `modefollowing.ini` | Settings file (required) |
| `--potfile` | `-pot` | `potential.mtp` | Potential file (should be MTP-based) |
| `--input` | `-in` | `POSCAR` | Input structure |
| `--input2` | `-in2` | `POSCAR_LAM` | Optional second structure for two-structure mode/overlap comparisons |
| `--input-format` | `-if` | `POSCAR` | Input structure format |
| `--index` | `-id` | `index.map` | Index file |
| `--max-loop` | | `100` | Maximum iterations per mode-following stage; must be nonnegative |
| `--next-tol` | `-ntol` | `-0.1` | Next-step tolerance |
| `--ifac` | | `0` | Force factor for the followed mode |
| `--ofac` | | `1` | Relaxation factor for non-followed orthogonal modes |
| `--sfac` | | `1` | Relaxation factor for non-following modes |
| `--cycle` | | `false` | Use a cyclic loop when generating the mode sequence (bool) |
| `--updateref` | | `false` | Update the reference lattice when the cell changes substantially (bool) |
| `--modecar-out` | `-mcout` | `MODECAR` | MODECAR output file (mode 9) |
| `--imode` | `-i` | `4` | Mode index to follow |
| `--mode` | `-m` | `1` | Job type |
| `--amplitude` | `-a` | `0.1` | Displacement amplitude |
| `--trial-step` | | `0.1` | Trial step size |
| `--symprec` | `-tol` | `1e-3` | Symmetry tolerance |
| `--frames` | | `16` | Number of trajectory frames |
| `--pressure` | `-p` | `0` | External pressure (GPa) |
| `--hessian` | | | Load Hessian from file |
| `--nocell` | | `false` | Fix cell (bool flag) |
| `--linetraj` | | `false` | Create linear trajectory (bool flag) |
| `--car-rep` | | `false` | Use Cartesian coordinates (bool flag) |
| `--request-efs` | `-efs` | `false` | Request EFS output (bool flag) |
| `--eigzero-tol` | | `0.008` | Tolerance for zero eigenvalues |

@note Many additional parameters (ifac, ofac, sfac, cycle, updateref, next-tol, etc.)
are available — run `imrmlp modefollowing --help` for the full list.

Variable-cell fractional mode uses `C-I` (`xx yy zz yz xz xy`) and
fractional atomic coordinates, with `H = H_ref chol(C)`. Hessians, mode
displacements and relaxation use the same convention. `--car-rep` uses
independent raw-cell and Cartesian atomic coordinates.
For two-structure comparisons, the second structure is rigidly rotated into
the same Cholesky cell convention before projecting displacements or modes.

### `calc_neb` — Nudged Elastic Band *(alias: `neb`)*

Minimum energy path search between two endpoint structures.

```bash
imrmlp neb neb_config.ini
imrmlp calc_neb neb_config.ini
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | | Settings file (positional, required) |
| `--create-input` | | `false` | Create a sample input file for NEB calculation (bool) |
| `--input-format` | `-if` | auto | Input structure format |
| `--nimgs` | | `0` | Number of images, including the two endpoints; normally set in the settings file |
| `--index` | `-id` | `index.map` | Index file mapping atom types to species |
| `--ninterp` | | `100` | Interpolation points for the MEP curve |
| `--cellweight` | `-cwt` | `100` | Deprecated; use `neb:cellweight`. The top-level value takes precedence. |
| `--mepout` | | `mep.xyz` | MEP trajectory output file |
| `--mepformat` | | `extxyz` | MEP trajectory format |
| `--summary` | | `neb.dat` | MEP summary output |
| `--neb:path_pbc` | | | Explicit PBC bits for a fresh path imported from POSCAR |
| `--neb:reference_cell` | | | Reference cell for a fresh variable-cell path; incompatible with restart |
| `--neb:profile` | | | MPI+OpenMP timing JSON filename |
| `--driver` | | `lbfgs` | Optimizer: `lbfgs`, `fire`, `cg`, or `sgd` |
| `--gtol` | `-gtol` | `lbfgs:epsgmax` | Active NEB force tolerance; overrides `neb:gtol` |
| `--maxiteration` | `-maxit` | `lbfgs:max_iterations` | Iteration limit per optimization run |
| `--max_total_iter` | | `1000000` | Total iteration budget across image-adjust restarts |
| `--initial_step` | | `lbfgs:initial_step` | Initial step size |
| `--max_step` | | `lbfgs:max_step` | Maximum step size |
| `--max_move` | | `lbfgs:max_step_move` | Maximum cumulative move per iteration |
| `--first_step` | | `lbfgs:first_step` | First step size for the L-BFGS history reset |
| `--epsilon` | | `lbfgs:epsilon` | Absolute force-norm accuracy |
| `--epsilon_rel` | | `lbfgs:epsilon_rel` | Relative force-norm accuracy |
| `--past` | | `lbfgs:past` | Number of past steps in the L-BFGS history |
| `--delta` | | `lbfgs:delta` | Convergence offset added to the tolerance |
| `--damping` | | | L-BFGS mixing parameter |
| `--s_curv_coeff` | | | Curvature condition coefficient |
| `--lbfgs_m` | | `15` | L-BFGS history length |
| `--next_step` | `-r` | `false` | Restart from `nebimage_XX_final.vasp` (bool) |

@note NEB parameters (spring constants, climbing-image options, symmetry release,
variable-image thresholds, etc.) are configured in the `neb:`-prefixed keys of the
settings file. The CLI flags above are explicit overrides for common cases; see
@ref guide_config for the full key list.

### Choosing a Saddle Point Method

| Command | Method | Requires | Best for |
|---------|--------|----------|----------|
| `calc_neb` | NEB + Climbing Image | Two endpoints (initial + final) | Unknown saddle location; full MEP needed |
| `modefollowing` | Mode following | Minimum + target mode direction | Following a specific soft phonon mode |

@note **Typical workflow**: First run `calc_neb` to map the energy barrier and locate
the approximate saddle.
Use `modefollowing` when the reaction coordinate is well-defined by a phonon analysis.

---

## Training & Fitting

Training commands fit potential parameters to reference data (typically DFT).
They form the core of the potential development workflow: after collecting
and curating reference structures, choose a training command based on model
type and optimization needs.

@note **Choosing a training command** — see the comparison table at the end of this section.

### `trainmtp` — Train MTP Potential

The primary training command for Moment Tensor Potentials. Fits both radial basis
coefficients and linear species/moment coefficients to reference DFT data (energies,
forces, stresses).


```bash
imrmlp trainmtp training.ini --do-lbfgs
imrmlp trainmtp --trainset=train.cfg --validset=valid.cfg --potfile=init.mtp --do-lbfgs
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | | Settings file (positional) |
| `--trainset` | `-train` | `train.cfg` | Training dataset file |
| `--validset` | `-valid` | | Validation dataset file |
| `--trainset-format` | `-if` | | Training dataset file format (empty = auto-detect) |
| `--validset-format` | `-vf` | | Validation dataset file format (empty = auto-detect) |
| `--potfile` | `-pot` | `potential.mtp` | Potential file |
| `--index` | `-id` | `index.map` | Index file |
| `--linear-fit` | `-lf` | `false` | Linear fit only, fix radial (bool flag) |
| `--do-rescale` | `-rs` | `false` | Search radial rescaling during linear fitting; requires `--linear-fit` |
| `--balance-at` | `-bw` | `false` | Load balancing for incomplete forces (bool flag) |
| `--do-sgd` | `-sgd` | `false` | Use SGD optimization (bool flag) |
| `--do-lbfgs` | `-lbf` | `false` | Use L-BFGS optimization (bool flag) |
| `--do-cg` | `-cg` | `false` | Use Conjugate Gradient (bool flag) |
| `--do-df` | `-df` | `false` | Use Differential Evolution (bool flag) |
| `--do-random` | `-random` | `false` | Random radial search (bool flag) |
| `--species-only` | | `false` | Fit only species coefficients through the generic L-BFGS trainer (bool flag) |
| `--do-matrix-ata` | `-ata` | `false` | Output A^T A matrix (bool flag) |
| `--do-matrix-a` | `-a` | `false` | Output full A matrix (memory intensive, bool flag) |
| `--calc-grad` | `-g` | `false` | Compute gradient for diagnostics (bool flag) |
| `--check-grad` | | `false` | Verify analytic gradient numerically (bool flag) |
| `--check-grad-all` | | `false` | Full gradient check for all parameters (bool flag) |
| `--test-curv` | | `false` | Test curvature calculation (bool flag) |
| `--test-curv-which` | | `0` | Curvature component index to test |
| `--test-curv-nudge` | | `0.001` | Perturbation magnitude for curvature test |
| `--debug` | | `false` | Debug output (bool flag) |

Choose `--do-lbfgs` (or another optimizer) explicitly for nonlinear fitting;
use `--linear-fit` to fit MTP linear coefficients. `fit:lfit_eng_only true` changes
which labels a linear fit uses; it does not select an optimizer by itself.
`train_linear` is a separate command with its own descriptor/settings requirements.
The final `trainmtp` output is `_Trained_Potential_.mtp` in the working directory;
`fit:save_to` controls trainer checkpoints.

`--do-matrix-a` writes six stress rows per labelled configuration, ordered
`xx, xy, xz, yy, yz, zz`. Off-diagonal rows multiply both the basis coefficients
and target by `sqrt(2)`, so equal row weights reproduce the full symmetric
tensor squared error. Diagonal rows are unchanged. The binary record layout
remains `int configuration_id`, followed by the coefficients and target as
`double` values; older exports used unscaled shear rows. These are geometric
weights only, not the configured training weights or atom-count normalization.

Training hyperparameters are read from the `[fit]` section, while the
potential file is given by the flat `potfile` key (default `potential.mtp`; the
`[pot]` section is not consulted by `trainmtp` — that is `trainnlpot`'s
behavior).

### `trainnlpot` — Train Nonlinear Potential

Train general nonlinear potentials (e.g., NN-MTP, HMTP) using an explicitly
selected optimizer. The potential type is specified in the `[pot]` section
or `pot:` namespace of the settings file (not as a CLI flag).

```bash
imrmlp trainnlpot --settings=training.ini --do-lbfgs
imrmlp trainnlpot --settings=training.ini --trainset=train.cfg --validset=valid.cfg --do-sgd -o output_potential
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | | Settings file (positional, required) |
| `--trainset` | `-train` | `train.cfg` | Training dataset file |
| `--validset` | `-inv` | | Validation dataset file |
| `--trainset-format` | `-if` | | Training data format (empty = auto-detect) |
| `--validset-format` | `-vf` | | Validation dataset format (empty = auto-detect) |
| `--testset` | | | Held-out dataset, evaluated only after the final fit |
| `--testset-format` | | | Held-out dataset format (empty = auto-detect) |
| `--fixed-core` | | | Existing fixed core; fit an ordinary MTP to residual labels and write a Hybrid manifest |
| `--fixed-core-type` | | `zbl` | Fixed core model type: `zbl` or `mtp` |
| `--output` | `-o` | `_Trained_Potential_` | Output potential file prefix |
| `--index` | `-id` | `index.map` | Index file |
| `--do-lbfgs` | `-lbf` | `false` | Use L-BFGS optimizer (bool flag) |
| `--do-sgd` | `-sgd` | `false` | Use SGD optimizer (bool flag) |
| `--do-varpro` | | `false` | Profiled radial L-BFGS with a linear refit at every trial (ordinary CPU MTP) |
| `--linear-refit-only` | | `false` | Refit linear coefficients of an initialized model without nonlinear training; requires `fit:linear_refit` |
| `--do-cg` | `-cg` | `false` | Use Conjugate Gradient (bool flag) |
| `--calc-grad` | `-g` | `false` | Compute gradient for diagnostics (bool flag) |
| `--check-grad` | | `false` | Verify gradient numerically (bool flag) |
| `--check-grad-all` | | `false` | Full gradient check (bool flag) |
| `--balance-atomic-fitness` | `-bw` | `false` | Load balancing for incomplete forces (bool flag) |
| `--test-curv` | | `false` | Test curvature calculation (bool flag) |
| `--test-curv-which` | | `0` | Curvature component index to test |
| `--test-curv-nudge` | | `0.001` | Perturbation magnitude for curvature test |
| `--debug` | | `false` | Debug output (bool flag) |

@note Nonlinear trainer hyperparameters are read from the `[fit]` section.

For a two-component HMTP, use `pot:pottype hmtp`, `pot:potfile source`,
and `pot:potlevel 2`. The loader reads `source_sub1` and `source_sub2`,
which must have the same species count and element ordering. Their radial
ranges may differ: the combined neighbor cutoff is their maximum, and each
component's parameter gradient uses its own neighbor subset. HMTP exposes
the combined coefficient vector to `trainnlpot`, including energy-only and
energy/force/stress gradient paths. With `--output fitted`, it writes
`fitted_sub1` and `fitted_sub2`; both are required to reload the model.

Use a finite `fit:bfgs_maxiteration` (or `fit:epoch_limit` for SGD), and
evaluate the saved model on an independent reference set. Without an
optimizer flag, ordinary nonlinear optimization is not performed.

### `fastfit` — Fast MTP Fitting

Fast linear fitting and basis selection for MTP potentials. Supports multiple modes:
mode 0 computes eigenvalue spectra, mode 1 performs simple weighted least-squares,
mode 2 applies a pre-computed basis selection, mode 3 does brute-force cut-one search,
mode 4 uses CUR-based importance scoring to select basis functions, mode 5 refines
the selection via simulated annealing or greedy swap, mode 6 applies NSGA-II
multi-objective genetic optimization to find the best basis subset, and mode 7
performs correlation analysis and decorrelation.

```bash
imrmlp fastfit training.ini
imrmlp fastfit --settings=settings.ini --potfile=init.mtp --trainset=train.cfg --validset=valid.cfg
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | Settings file (positional, required) |
| `--potfile` | `-pot` | `potential.mtp` | Potential file |
| `--trainset` | `-ts` | `train.cfg` | Training set |
| `--validset` | `-val` | `valid.cfg` | Validation set |
| `--format` | `-fmt` | `mlpcfg` | Data format |
| `--index` | `-id` | `index.map` | Index file |
| `--weight-energy` | `-we` | `1.0` | Energy weight |
| `--weight-force` | `-wf` | `0.1` | Force weight |
| `--weight-stress` | `-ws` | `0.001` | Stress weight |
| `--weight-l2` | `-ww` | `1e-8` | L2 regularization weight |
| `--mode` | `-m` | `0` | Fit mode (0–7) |
| `--method` | | `llt` | Solver method; use `greedy` for greedy swap in mode 5 |
| `--sparse-method` | `-dm` | `0` | Mode 4: 0=select high, 1=discard low |
| `--discard-rate` | `-drate` | `0.1` | Mode 4: fraction of scalars to discard |
| `--epslow` | `-epslow` | `1e-6` | Mode 4: eigenvalue floor |
| `--cur` | `-cur` | `false` | Mode 4: enable CUR random sampling (bool) |
| `--cur-nsearch` | `-ntry` | `10` | Mode 4: maximum CUR search iterations |
| `--addback` | `-addback` | `false` | Mode 4: add back on-path scalars (bool) |
| `--corr-strategy` | `-cs` | `0` | Mode 7: decorrelation strategy, 0=threshold, 1=clustering, 2=greedy VIF |
| `--corr-pearson` | `-cp` | `0.95` | Mode 7: Pearson threshold |
| `--corr-vif` | `-cv` | `10.0` | Mode 7: VIF threshold |
| `--corr-max-samples` | `-cms` | `50000` | Mode 7: maximum atoms sampled for correlation analysis |
| `--corr-max-features` | `-cmf` | `0` | Mode 7: maximum features to keep, 0=no limit |
| `--corr-addback` | `-caddback` | `false` | Mode 7: add back DAG-path scalars after decorrelation (bool) |
| `--corr-save-reduced` | `-csave` | `false` | Mode 7: save reduced potential after selection (bool) |
| `--cmp` | `-cmp` | `false` | Mode 1: compare solver methods (bool) |
| `--objective-convention` | | `legacy` | `legacy` or shared mean-moments EFS/L2 objective |
| `--l2-reference` | | `native` | L2 input reference: `native`, `mean`, or `sum`; `mean`/`sum` require mean-moments |
| `--objective-report` | | | Optional TSV objective contract, including the effective L2 term |
| `--selection-objective` | | `legacy` | `legacy` or `error-cost` (modes 5/6) |
| `--max-valid-loss` | | | Required mean weighted validation-loss ceiling for error-cost selection |
| `--cost-report` | | `fastfit-cost.tsv` | Cost-selection TSV report |
| `--cost-benchmark-topk` | | `8` | Final measured candidates, excluding the baseline |
| `--cost-benchmark-configs` | | `16` | Representative validation configurations |
| `--cost-archive-size` | | `128` | Bounded cross-generation candidate archive |
| `--cost-archive-rule` | | `spread` | `spread` (legacy) or bounded `pareto` archive |
| `--cost-export-dir` | | | Export verified models, Pareto fronts, objective and timing rounds |
| `--cost-benchmark-rounds` | | `5` | Repeated measurement rounds (at least 3) |
| `--cost-benchmark-seconds` | | `0.01` | Minimum seconds per candidate/path/round |
| `--profile` | | `false` | Report exclusive stage times and peak RSS (bool) |
| `--profile-output` | | | Write a profile TSV; also enables profiling |
| `--use-matrices` | | `false` | Read exported unweighted E/F/S matrix blocks (bool) |
| `--energy-power` | | `1` | Energy squared-residual atom-count power |
| `--force-power` | | `0` | Force squared-residual atom-count power |
| `--stress-power` | | `1` | Virial squared-residual atom-count power |
| `--force-norm-d` | | | Force normalization denominator offset |
| `--force-norm-f` | | | Force normalization scale (0 disables) |
| `--force-norm-p` | | | Force normalization power |
| `--seed` | | | Reproducible random seed |
| `--search-steps` | | `-1` | SA/greedy steps (-1 selects an automatic budget) |
| `--population` | | `0` | NSGA-II population (0 = automatic) |
| `--generations` | | `0` | NSGA-II generations (0 = automatic) |
| `--mutation-rate` | | `0.05` | NSGA-II per-scalar flip probability; -1 uses 1/nscalar |
| `--population-init` | | `half` | NSGA-II initialization: `half`, `stratified`, or `dependency` |
| `--group-mutation-rate` | | `0` | Probability of dropping one dependency group after scalar mutation |
| `--group-mutation-kind` | | `basic` | Group family: `basic`, `radial`, or `mixed` |
| `--sa-temperature` | | | Initial temperature for the mean per-configuration loss |
| `--sa-cooling` | | | SA cooling factor in (0, 1] |
| `--matE` | `-me` | | Energy design-matrix block |
| `--matF` | `-mf` | | Force design-matrix block |
| `--matS` | `-ms` | | Stress design-matrix block |
| `--vecE` | `-ve` | | Energy target vector |
| `--vecF` | `-vf` | | Force target vector |
| `--vecS` | `-vs` | | Stress target vector |

Selection files contain a count followed by unshifted scalar-moment IDs; species
coefficients are always retained implicitly. Modes 2, 4, 5, 6, and 7 use this same
format. Under MPI, candidate fitting and error evaluation are collective operations,
while output files are written by rank 0.

### `train_linear` — Train Linear Descriptor Potential

Train a linear descriptor-based potential using least-squares regression.
Solves the normal equations \f$(\mathbf{A}^T\mathbf{W}\mathbf{A} + \lambda\mathbf{I})\mathbf{w} = \mathbf{A}^T\mathbf{W}\mathbf{y}\f$
analytically in one shot, without iterative gradient descent. Supports MPI for
distributed assembly across nodes.

```bash
imrmlp train_linear --settings=training.ini
imrmlp train_linear --trainset=train.cfg --validset=valid.cfg -o fitted_potential
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | | Settings file (positional, required) |
| `--trainset` | `-in` | `train.cfg` | Training dataset file |
| `--validset` | `-inv` | | Validation dataset file |
| `--trainset-format` | `-if` | `mlpcfg` | Training data format: mlpcfg, extxyz, vaspabn |
| `--output` | `-o` | `_Trained_Potential_` | Output potential file prefix |
| `--index` | `-id` | `index.map` | Index file mapping atom types to indices |
| `--debug` | | `false` | Enable debug output (bool) |

@note The settings file uses `[fit]` section for TrainerLinearPotential parameters
(energy_weight, force_weight, stress_weight, weight_L2, etc.) and `[pot]` section
for potential configuration. See @ref theory_linear for the mathematical background.

### `nnmtp` — Create NN-MTP Template

Create a neural-network MTP potential template from an MTP template. This is a
preparatory step before `trainnlpot` with the `nnmtp` potential type.

```bash
imrmlp nnmtp --settings=nnmtp.ini
imrmlp nnmtp -pot init.mtp -n 10 -n2 5 -o nnmtp_template.mtp
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | | Settings file (positional) |
| `--potfile` | `-pot` | `potential.mtp` | Input MTP potential template |
| `--data` | `-d` | `train.cfg` | Training data file for auto-detection |
| `--input-format` | `-f` | auto | Training data format |
| `--nneuron` | `-n` | `10` | Neurons in first hidden layer |
| `--nneuron2` | `-n2` | `5` | Neurons in second hidden layer |
| `--nnlay` | `-l` | `1` | Number of hidden layers: 1 or 2 |
| `--index` | `-id` | `index.map` | Index file |
| `--initmtp` | `-m` | `true` | Initialize with MTP coefficients (bool) |
| `--extent` | `-e` | `1.0` | Normalization standard-deviation multiplier; finite and positive |
| `--output` | `-o` | `potential_specific` | Output NN-MTP template |
| `--normalize` | `-norm` | `true` | Normalize input features (bool) |
| `--cvnv` | `-cvnv` | `false` | Convert the output prefix's `.nn` file to text |

The data file must contain nonempty structures. Structures without energy labels
still contribute to normalization, but are skipped with a warning when fitting
species energy offsets. If all energy labels are missing, offsets are initialized
to zero with a warning. Hidden neuron counts must be positive for each used layer.

### `initmtpm` — Initialize MTP_M

Initialize a PotMTP_M (multi-order MTP) potential file with random coefficients.
Used as a starting point when no pre-trained potential exists. The MTP level and
species count are configured in the `[pot]` section of the settings file.

```bash
imrmlp initmtpm --settings=settings.ini
imrmlp initmtpm -pot initial.mtp --dataset=train.cfg
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | Settings file (positional, required) |
| `--potfile` | `-pot` | `potential.mtp` | Output potential file |
| `--dataset` | `-d` | `train.cfg` | Reference dataset for basis detection |
| `--format` | `-f` | | Dataset format (empty = auto-detect) |
| `--index` | `-id` | `index.map` | Index file |
| `--randomize` | `-r` | `true` | Randomize coefficients (bool) |
| `--rand-scale` | | `1.0` | Random scale factor |
| `--rand-lb` | | `-0.1` | Random lower bound |
| `--rand-ub` | | `0.1` | Random upper bound |

With `--randomize=false`, radial, moment and model coefficients are preserved;
species energy offsets and property biases are still initialized from the dataset.
Structures without energy labels contribute to basis statistics but are skipped
when fitting energy offsets, with a warning. If no energy labels are available,
energy offsets are initialized to zero. Missing, empty or unreadable datasets fail.

### Training Command Selection Guide

| Command | Model type | Method | Use for |
| --- | --- | --- | --- |
| `trainmtp --linear-fit` | Ordinary MTP | Weighted linear least squares | Fixed radial basis; fit species offsets and linear moment coefficients |
| `trainmtp --do-lbfgs` | Ordinary MTP | L-BFGS; also SGD / CG / DE | Fit radial and linear parameters together |
| `trainnlpot` | MTP, MTP-M, NN-MTP, HMTP, LES/SOG and others | Built-in L-BFGS / SGD / CG; VarPro for ordinary CPU MTP | A generic nonlinear path compatible with the model and loss |
| `train_linear` | `descriptor_linear` | Linear least squares | Fit linear coefficients on fixed descriptors |
| `fastfit` | Ordinary MTP | Linear fitting with SA / NSGA-II and other basis selection | Compare errors, basis size and evaluation cost |

For ordinary MTP, choose a compatible template, then run `trainmtp --linear-fit` or
`trainmtp --do-lbfgs`. `initmtpm` targets the separate MTP-M model. See the
[training workflow](../guides/workflows.md) before combining different trainers.

---

## Active Learning (MaxVol / FPS / CUR)

Active learning commands select informative structures from externally prepared
candidate pools, reducing the number of expensive reference calculations.

- **MaxVol selection**: `maxvol_select`, `maxvol_sample` and `maxvol_extend`
  select independent descriptor rows, grade extrapolation and extend active sets.
- **Diversity sampling**: `diverse_select` uses FPS or CUR to select a
  representative subset of a candidate pool.
- **Relaxation monitoring**: `relax` with `[al]` grades extrapolation, collects
  candidates and stops above the configured break threshold. Labeling and
  retraining run separately. See the [AL configuration](configuration.md).

### `maxvol_select` — MAXVOL Selection

Build a maximally-independent training set from a structure pool using MaxVol.

```bash
imrmlp maxvol_select --potfile=potential.mtp --pool=pool.cfg -o selected.cfg
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--pool` | | `pool.xyz` | Input structure pool |
| `--potfile` | | `potential.mtp` | Potential file for B-matrix |
| `--pottype` | | `mtp` | Potential type: mtp, eampoly, lj612, ... |
| `--dtype` | | `mtm` | Descriptor type: mtm, pot_efs, behler, bondorder |
| `--index` | `-id` | `index.map` | Index file |
| `--input-format` | `-if` | auto | Input pool file format |
| `--output-format` | `-of` | auto | Output format |
| `--output` | `-o` | `selected.xyz` | Output selected structures |
| `--score-file` |  | — | Output per-structure MaxVol scores |
| `--threshold` | | `1.001` | Selection threshold |
| `--max-swaps` | | `0` | Swap limit (0 = unlimited) |
| `--weight` | | `1.0` | B-matrix row weight |
| `--pow-scale` | | `0.5` | Atom count scaling power |
| `--init-mode` | | `auto` | Active-set initialization: `auto`, MLIP-compatible virtual identity (`mlip`), or legacy real rows (`data`) |
| `--init-value` | | `1e-6` | Scale of the virtual identity rows in `mlip` mode |
| `--verbose` | `-v` | `false` | Verbose output (bool) |
| `--all-coeffs` | | `false` | Use ALL potential coefficients, not just linear (bool) |
| `--row-scaling` | | `uniform` | `uniform` or MLIP-compatible energy/stress-only atom-count scaling (`mlip`) |
| `--select-mode` | | `stream` | Selection backend: `stream`, `full`, or order-dependent approximate `online` |
| `--stream-mode` | | `exact` | `exact` scans the cached complete pool to convergence; `fast` uses bounded global shortlists |
| `--stream-chunk-size` | | `32` | Number of structures built per parallel cache chunk |
| `--stream-fast-passes` | | `2` | Maximum global passes in `fast` stream mode |
| `--stream-fast-shortlist` | | `0` | Global shortlist size in `fast` mode; 0 selects an automatic size |
| `--stream-cache-mode` | | `auto` | Row cache backend: `auto`, `memory`, `mmap`, or legacy `off` |
| `--stream-memory-limit-mb` | | `3072` | Maximum resident row-cache size per process; `auto` spills later chunks to mmap |
| `--stream-cache-dir` |  | — | Temporary mmap directory; defaults to `TMPDIR`, then `/tmp` |
| `--mode` | | `nbh` | Row mode: `nbh` (per-atom) or `cfg` (per-structure) |
| `--w-energy` | | `1.0` | Energy weight for B-matrix |
| `--w-site` | | `0.0` | Site energy weight for B-matrix |
| `--w-force` | | `0.0` | Force weight for B-matrix |
| `--w-stress` | | `0.0` | Stress weight for B-matrix |
| `--settings` | | `select.ini` | Settings file (positional, required) |

MaxVol state files (`.sel`) use format v10. Model metadata parsing is
independent of the state format.

In settings files, shared MaxVol builder and selector options should use
namespaced keys:

```ini
builder:dtype mtm
builder:potfile potential.mtp
builder:pottype mtp
builder:mode nbh
builder:all-coeffs false
builder:row-scaling uniform
builder:w-energy 1.0
builder:w-site 0.0
builder:w-force 0.0
builder:w-stress 0.0

selector:threshold 1.001
selector:weight 1.0
selector:pow-scale 0.5
selector:init-mode auto
selector:init-value 1e-6
selector:max-swaps 0
selector:select-mode stream
selector:stream-mode exact
selector:stream-cache-mode auto
selector:stream-memory-limit-mb 3072
selector:verbose false
```

`online` mode keeps the active matrix but does not retain rejected historical
candidates. A later exchange can therefore raise an earlier candidate's score;
the result is order-dependent and is an approximation over the cumulative
pool. Use `full` or `stream` with `stream-mode exact` when the final active set
must satisfy the threshold against every input row. Cached exact streaming
builds descriptors once, then rescans cached rows after each accepted exchange.
`auto` keeps up to the configured per-process limit resident and transparently
places additional chunks in temporary mmap storage. MPI launches partition
configurations by rank while keeping deterministic global row ordering.

`init-mode auto` selects MLIP-compatible initialization for `pot_efs` and
`mtp_energy`: the active matrix starts from `init-value * I`, and only virtual
rows whose directions are supported by the pool are replaced by real B rows.
This avoids selecting one real structure per formally present but inactive
coefficient direction. Other descriptor builders retain the legacy `data`
initialization. Selection output reports convergence, the maximum remaining
grade, and the number of virtual rows. A state produced after a swap/pass limit
is saved with `converged=no`; sampling or extending it prints a warning.

### `maxvol_sample` — MAXVOL Sampling

Grade structures against an existing MaxVol active set (extrapolation detection).

```bash
imrmlp maxvol_sample --potfile=potential.mtp --input=pool.cfg --state-file=state.bin -o candidates.cfg
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `sample.ini` | Settings file (positional, required) |
| `--potfile` | | `potential.mtp` | Potential file |
| `--pottype` | | `mtp` | Potential type |
| `--input` | | `candidates.xyz` | Input configuration file |
| `--output` | `-o` | `sampled.xyz` | Output sampled structures |
| `--state-file` | | | Active-set state file (default: `potfile` + `.sel`) |
| `--dtype` | | `mtm` | Descriptor type |
| `--index` | `-id` | `index.map` | Index map file |
| `--input-format` | `-if` | | Input format |
| `--output-format` | `-of` | `extxyz` | Output format |
| `--mode` | | `nbh` | Row mode: `nbh` (per-atom) or `cfg` (per-structure) |
| `--threshold` | | `1.001` | MaxVol selection threshold exposed by the shared selector console |
| `--threshold-save` | | `2.1` | Grade above this threshold → save to output |
| `--threshold-break` | | `11.0` | Grade above this threshold → abort grading |
| `--weight` | | `1.0` | B-matrix row weight |
| `--pow-scale` | | `0.5` | natoms scaling power |
| `--w-energy` | | `1.0` | PotEFSBuilder energy weight |
| `--w-site` | | `0.0` | PotEFSBuilder site-energy weight |
| `--w-force` | | `0.0` | PotEFSBuilder force weight |
| `--w-stress` | | `0.0` | PotEFSBuilder stress weight |
| `--add-grade` | | `true` | Add `MV_grade` feature to output (bool) |
| `--dump-state` | | `false` | Print state file summary, skip grading (bool) |
| `--extract-configs` | | | Export configs from state file to given path |
| `--selected-only` | | `false` | Export only selected configs (with --extract-configs, bool) |
| `--verbose` | `-v` | `false` | Verbose output (bool) |
| `--all-coeffs` | | `false` | Use ALL coefficients, not just linear (bool) |
| `--row-scaling` | | `uniform` | Row scaling: `uniform` or `mlip` |
| `--init-mode` | | `auto` | Active initialization: `auto`, `mlip`, or `data` |
| `--init-value` | | | MLIP virtual identity scale |
| `--max-swaps` | | `0` | Swap limit (0 = unlimited) |
| `--select-mode` | | `stream` | Selection backend: `stream`, `full`, or `online` |
| `--stream-mode` | | `exact` | Stream backend: `exact` or `fast` |
| `--stream-chunk-size` | | | Streaming configuration chunk size |
| `--stream-fast-passes` | | | Maximum passes in stream fast mode |
| `--stream-fast-shortlist` | | `0` | Fast-mode global shortlist size (0 = auto) |
| `--stream-cache-mode` | | `auto` | Stream cache: `auto`, `memory`, `mmap`, or `off` |
| `--stream-memory-limit-mb` | | `3072` | Per-process resident stream-cache limit (MiB) |
| `--stream-cache-dir` | | | Directory for temporary stream mmap files |

`maxvol_sample` uses the same `builder:*` and `selector:*` settings-file
blocks as `maxvol_select`; `threshold-save` and `threshold-break` remain
command-specific top-level keys.

### `maxvol_extend` — MAXVOL Augmentation

Incrementally expand an existing active set with informative candidates.

Two modes:

**Mode 1 — From state file (recommended for iterative AL):**

```bash
imrmlp maxvol_extend --potfile=potential.mtp \
  --state-file=state.bin --candidates=pool.cfg -o new_selected.cfg
```

Loads an existing active set from a `.sel` state file (produced by `maxvol_select`),
evaluates candidates against it, and outputs newly selected structures.
An optional `--train` file can supply extra training structures beyond what is
already in the state file.

**Mode 2 — 3-stage selection (original, no state file):**

```bash
imrmlp maxvol_extend --potfile=potential.mtp \
  --train=train.cfg --candidates=pool.cfg -o new_selected.cfg
```

Builds an active set from `--train`, grades candidates, then re-stabilizes with
the combined pool.  Suitable for one-shot training set construction.

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--potfile` | | `potential.mtp` | Potential file for B-matrix |
| `--pottype` | | `mtp` | Potential type: mtp, eampoly, lj612, ... |
| `--state-file` |  | — | Active-set state file (`.sel`) — skip Stage 1 |
| `--train` |  | — | Training set file |
| `--candidates` | | `candidates.xyz` | Candidate structures to evaluate |
| `--output` | `-o` | `new_selected.xyz` | Output newly selected structures |
| `--dtype` | | `mtm` | Descriptor type: mtm, pot_efs, behler, bondorder |
| `--index` | `-id` | `index.map` | Index file |
| `--input-format` | `-if` | auto | Input file format |
| `--output-format` | `-of` | `extxyz` | Output format |
| `--mode` | | `nbh` | Row mode: `nbh` (per-atom) or `cfg` (per-structure) |
| `--threshold` | | `1.001` | Selection threshold |
| `--weight` | | `1.0` | B-matrix row weight |
| `--pow-scale` | | `0.5` | Atom count scaling power |
| `--w-energy` | | `1.0` | Energy weight for B-matrix |
| `--w-site` | | `0.0` | Site energy weight for B-matrix |
| `--w-force` | | `0.0` | Force weight for B-matrix |
| `--w-stress` | | `0.0` | Stress weight for B-matrix |
| `--max-swaps` | | `0` | Swap limit (0 = unlimited) |
| `--verbose` | `-v` | `false` | Verbose output (bool) |
| `--all-coeffs` | | `false` | Use ALL coefficients, not just linear (bool) |
| `--row-scaling` | | `uniform` | Row scaling: `uniform` or `mlip` |
| `--init-mode` | | `auto` | Active initialization: `auto`, `mlip`, or `data` |
| `--init-value` |  | `1e-6` | MLIP virtual identity scale |
| `--select-mode` | | `stream` | Selection backend: `stream`, `full`, or `online` |
| `--stream-mode` | | `exact` | Stream backend: `exact` or `fast` |
| `--stream-chunk-size` |  | `32` | Streaming configuration chunk size |
| `--stream-fast-passes` |  | `2` | Maximum passes in stream fast mode |
| `--stream-fast-shortlist` | | `0` | Fast-mode global shortlist size (0 = auto) |
| `--stream-cache-mode` | | `auto` | Stream cache: `auto`, `memory`, `mmap`, or `off` |
| `--stream-memory-limit-mb` | | `3072` | Per-process resident stream-cache limit (MiB) |
| `--stream-cache-dir` |  | — | Directory for temporary stream mmap files |
| `--settings` | | `select_add.ini` | Settings file (legacy default name; positional, required) |

`maxvol_extend` uses the same `builder:*` and `selector:*` settings-file
blocks as `maxvol_select`; training/candidate/state paths remain top-level
command keys.

### `diverse_select` — Diversity-Based Representative Selection *(alias: `fps_select`)*

Select a diverse/representative subset via FPS (Farthest Point Sampling) or CUR
(SVD leverage-score) decomposition in descriptor space. Alias: `fps_select`.

```bash
# FPS (default): geometric diversity
imrmlp diverse_select --n-select=50 --pool=pool.xyz --output=selected.xyz

# CUR: statistical leverage-score sampling
imrmlp diverse_select --method=cur --n-select=50 --pool=pool.xyz --output=selected.xyz

# FPS with cosine distance metric
imrmlp diverse_select --method=fps --metric=cosine --n-select=50 --pool=pool.xyz
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--pool` | | `pool.xyz` | Input structure pool file |
| `--potfile` | | `potential.mtp` | Potential file (for descriptor-based B-matrix) |
| `--pottype` | | `mtp` | Potential type |
| `--dtype` | | `mtm` | Descriptor type: mtm, pot_efs, behler, bondorder |
| `--index` | `-id` | `index.map` | Index file |
| `--format` | `-f` | | Input format (empty = auto-detect) |
| `--output` | `-o` | `selected.xyz` | Output selected structures file |
| `--score-file` | | | Output per-structure scores |
| `--n-select` | `-n` | `0` | Maximum number of structures to select (required for CUR) |
| `--method` | | `fps` | Selection method: `fps` (default) or `cur` |
| `--metric` | | `euclidean` | Distance metric for FPS: `euclidean` (default) or `cosine` |
| `--threshold-distance` | | `0.0` | FPS: stop when min distance drops below this |
| `--first-idx` | | `-1` | FPS: index of first selected structure (-1 = random) |
| `--cur-deterministic` | | `false` | CUR: use deterministic top-k selection (bool) |
| `--seed` | | `42` | Random seed for reproducibility |
| `--compute-error` | | `false` | CUR: compute reconstruction error (bool) |
| `--mode` | | `cfg` | Row mode: `cfg` (per-structure) or `nbh` (per-atom) |
| `--verbose` | `-v` | `false` | Verbose output (bool) |
| `--all-coeffs` | | `false` | Use ALL coefficients, not just linear (bool) |
| `--row-scaling` | | `uniform` | Row scaling: `uniform` or `mlip` |
| `--w-energy` | | `1.0` | Energy row weight for `pot_efs` |
| `--w-site` | | `0.0` | Site-energy row weight for `pot_efs` |
| `--w-force` | | `0.0` | Force row weight for `pot_efs` |
| `--w-stress` | | `0.0` | Stress row weight for `pot_efs` |
| `--settings` | | `diverse_select.ini` | Settings file (positional, required) |

In settings files, descriptor/B-matrix construction for `diverse_select` uses
the same `builder:*` block as MaxVol. FPS/CUR controls such as `method`,
`n-select`, `metric`, and `seed` remain top-level keys.

FPS requires a positive `--n-select` or `--threshold-distance`. When both are
provided, selection stops as soon as either limit is reached.

### `calc_crowding` — Descriptor-Space Crowding Weights

Compute bounded-memory per-atom descriptor crowding weights and write them as
`atomic_fitness` (and optionally the configuration `fitness`). The command makes
three streaming passes over the input, so it scales to large datasets without
keeping all descriptors in memory.

```bash
imrmlp calc_crowding --potfile=potential.mtp --input=pool.xyz --output=pool-crowded.xyz
imrmlp calc_crowding --potfile=potential.mtp --input=pool.xyz --kernel=gaussian --apply-to-energy
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `pool.xyz` | Input structures |
| `--output` | `-out` | `pool-crowded.xyz` | Output structures with `atomic_fitness` |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format (`extxyz`, `mlpcfg`, or `mlipbin`) |
| `--settings` | | `crowding.ini` | Settings file (positional) |
| `--index` | `-id` | `index.map` | Species index map |
| `--kernel` | | `cosine` | Kernel: `cosine` or `gaussian` |
| `--sigma` | | auto | Gaussian bandwidth (≤0 = automatic) |
| `--standardize` | | `false` | Scale descriptor components before the kernel (bool) |
| `--alpha` | | | Inverse-density exponent |
| `--clip-min` | | | Lower weight clip |
| `--clip-max` | | | Upper weight clip |
| `--sample` | | | Reference rows per element (-1 = exact small-data mode) |
| `--seed` | | | Deterministic sampling seed |
| `--batch-rows` | | | Density query batch size |
| `--bucket` | | `true` | Bucket density by chemical element |
| `--apply-to-energy` | | `false` | Write the configuration fitness as the mean atomic weight |
| `--potfile` | | `potential.mtp` | Potential file used by `dtype=pot_efs` |
| `--pottype` | | `mtp` | Potential type |
| `--dtype` | | `mtm` | Descriptor type: `mtm`, `mtp`, `mtp_scalar`, `behler`, `bondorder` |
| `--mode` | | `nbh` | Row mode; forced to `nbh` by this command |
| `--all-coeffs` | | `false` | Use all potential coefficients |
| `--row-scaling` | | `uniform` | Row scaling: `uniform` or `mlip` |
| `--w-energy` | | `1.0` | Energy row weight for `pot_efs` |
| `--w-site` | | `0.0` | Site-energy row weight for `pot_efs` |
| `--w-force` | | `0.0` | Force row weight for `pot_efs` |
| `--w-stress` | | `0.0` | Stress row weight for `pot_efs` |

Settings for the descriptor builder are read from the `builder:` block.

### Active Learning Strategy Guide

| Strategy | Command(s) | Principle | Output | Best For |
|----------|-----------|-----------|--------|----------|
| **MaxVol Select** | `maxvol_select` | Maximize determinant of descriptor submatrix | Maximally independent subset | Building initial well-conditioned training set |
| **MaxVol Sample** | `maxvol_sample` | Extrapolation grade vs existing set | Per-structure grade (>1 = extrapolative) | Screening candidate pools for informative additions |
| **MaxVol Augment** | `maxvol_extend` | Sample + add high-grade structures | Augmented training set | Incrementally growing an existing training set |
| **Diversity FPS** | `diverse_select --method fps` | Maximize minimum pairwise distance | Uniformly diverse subset | Ensuring geometric coverage of config space |
| **Diversity CUR** | `diverse_select --method cur` | SVD leverage-score sampling | Statistically representative subset | Large-pool downselection with reconstruction guarantees |

@note **Typical active learning loop**: prepare candidate structures externally →
`maxvol_select` (select an initial reference set) → run reference calculations →
`trainmtp` or `train_linear` (fit a single potential) → `maxvol_sample`
(grade candidates) → `maxvol_extend` (extend the active set) → label selected
structures and repeat fitting.

---

## Structure Manipulation

Structure manipulation commands transform atomic configurations without computing
energies or forces. They are preprocessing steps — prepare supercells, standardize
cells, reorder axes, or insert interstitial atoms with `findint` before EFS evaluation
or relaxation.

@note These commands do NOT require a potential. They operate purely on geometry.

### `mk_super` — Make Supercell

Build a supercell from three diagonal integers or nine row-major integers.

```bash
imrmlp mk_super --input=POSCAR 2 2 2 --output=super.vasp
imrmlp mk_super --input=POSCAR 2 0 0 0 2 0 0 0 2 --output=super.vasp   # 3x3 matrix (row-major)
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | `POSCAR_REV` | Output structure file (positional) |
| `--supercell` | `-s` | | 3 diagonal or 9 row-major integers (positional) |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--index` | `-index` | `index.map` | Species index map |
| `--check` | `-c` | `true` | Flip the first lattice row if the cell is left-handed |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

### `mk_lower` — Make Cell Lower Triangular

Convert the cell matrix to lower-triangular form. Useful for standardizing
periodic cell representations before comparison or symmetry analysis.

```bash
imrmlp mk_lower --input=POSCAR --output=lower.vasp
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | `POSCAR_REV` | Output structure file (positional) |
| `--ax` | `-ax` | | Lattice vector (`a`, `b`, or `c`) to align with x |
| `--order` | `-order` | | Axis order: `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, `zyx` (or `abc` forms) |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--index` | `-index` | `index.map` | Species index map |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

### `mk_orth` — Find (Nearly) Orthogonal Cell

Keeps lattice vector **a** fixed. Searches for an equivalent cell with nearly 90° angles.

```bash
imrmlp mk_orth --input=structure.cfg --output=orth.cfg
imrmlp mk_orth --input=POSCAR --axis-choice 1 1 0 --axis-to x --max-expand 8
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | `POSCAR_ORTH` | Output structure file (positional) |
| `--tol` | `-tol` | `1e-2` | Maximum absolute cosine for orthogonality |
| `--max-expand` | `-mx` | `20` | Coefficient/volume expansion budget |
| `--axis-choice` | | | Three integer lattice-direction coefficients (reduced to primitive); requires full PBC |
| `--axis-to` | | `x` | Align the directed axis to `x`, `y`, or `z` |
| `--index` | `-id` | `index.map` | Species index map |
| `--one` | `-one` | `false` | Stop at the first valid result |
| `--search-only` | `-s` | `false` | Print the supercell matrix without writing structures |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

### `mk_orth_search` — Search Orthogonal Cell

Like `mk_orth` but searches all lattice-vector orientations.

```bash
imrmlp mk_orth_search --input=structure.cfg --output=orth.cfg --tol=1e-4
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | `POSCAR_ORTH` | Output structure file (positional) |
| `--tol` | `-tol` | `1e-2` | Maximum absolute cosine for orthogonality |
| `--max-expand` | `-mx` | `20` | Coefficient/volume expansion budget |
| `--index` | `-id` | `index.map` | Species index map |
| `--one` | `-one` | `false` | Stop at the first valid result |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

### `trans_reduce` — Reduce Translational Multiplicity

Reduce translational multiplicity of a structure along lattice directions.

```bash
imrmlp trans_reduce --input=POSCAR --output=reduced.vasp --tol=1e-4
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | `POSCAR_TRED` | Output structure file (positional) |
| `--tol` | `-tol` | `1e-3` | Symmetry tolerance |
| `--optimize-search` | | `false` | Opt-in exact candidate pruning/cache for 3D/1D searches |
| `--index` | `-id` | `index.map` | Species index map |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

### `primi_reduce` — Reduce to Primitive Cell

Find a primitive cell with all angles between 60° and 120°.

```bash
imrmlp primi_reduce --input=POSCAR --output=primitive.vasp
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | `POSCAR_PRIM` | Output structure file (positional) |
| `--tol` | `-tol` | `1e-5` | Symmetry tolerance |
| `--two-stage` | `-t` | `true` | Run `trans_reduce` before the full primitive search |
| `--correct-cell` | `-c` | `false` | Try to make the resulting cell as orthogonal as possible |
| `--optimize-search` | | `false` | Opt-in exact candidate pruning/cache for 3D/1D searches |
| `--index` | `-id` | `index.map` | Species index map |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

### `swap_axis` — Reorder Lattice Axes

```bash
imrmlp swap_axis --order=132 --input=POSCAR --output=swapped.vasp
# mode: 3-digit permutation. 132 = xyz → xzy
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--order` | `-order` | `xyz` | Axis permutation (positional): `xyz`, `xzy`, `yxz`, `yzx`, `zxy`, `zyx` (or `abc` forms) |
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | `POSCAR_REV` | Output structure file (positional) |
| `--tol` | `-tol` | `1e-6` | Tolerance used when swapping axes |
| `--keephand` | `-k` | `true` | Preserve right-handedness |
| `--tolower` | `-l` | `false` | Convert the result to lower-triangular form |
| `--index` | `-index` | `index.map` | Species index map |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

### `sameq` — Check Structural Equivalence

Check if two structures are equivalent (same geometry).

```bash
imrmlp sameq --input1=structure1.cfg --input2=structure2.cfg --tol=0.1
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input1` | `-in1` | | First structure (positional) |
| `--input2` | `-in2` | | Second structure (positional) |
| `--tol` | `-tol` | `1e-4` | Cell and position tolerance in Å |
| `--input-format1` | `-if1` | auto | Format of the first structure |
| `--input-format2` | `-if2` | auto | Format of the second structure |
| `--index` | `-id` | `index.map` | Species index map |
| `--scale` | `-s` | `false` | Scale both cells to the same volume before comparison |
| `--allow-reflections` | | `true` | Treat mirror images as equivalent |

### `find_mindist` — Minimum Interatomic Distance

Report the minimum interatomic distance per structure and the global minimum.
Prints to stdout unless `--output` is given, in which case one distance per
line followed by the global minimum is written.

```bash
imrmlp find_mindist --input=traj.cfg
imrmlp find_mindist --input=traj.cfg --output=mindist.dat
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure file (positional) |
| `--output` | `-out` | | Optional distance output file |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--index` | `-index` | `index.map` | Species index map |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

## Analysis

Structure analysis commands compute derived properties — from basic structural
descriptors (RDF, bond angles) to quantities such as force constants
and elastic constants. Some commands require a potential for EFS evaluation;
others operate on geometry alone.

`sameq` shares its comparison kernel with batch deduplication: lattice checks
and a species-preserving atom bijection, including translations, atom
permutations and lattice symmetry operations. Only fully periodic structures
are reduced to primitive cells. Partial and nonperiodic structures retain their
cells, including vacuum dimensions. `--allow-reflections=false` distinguishes
mirror images; the default includes them. Fast deduplication compares cached
descriptors within composition groups without quantized descriptor buckets.

### `calc_prop` — Structure Properties

Compute structural properties: RDF, bond angles, dihedrals, SRO, HA index, Voronoi,
CNA, MSD and VACF.


| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | | Settings file (positional, required) |
| `--input` | `-in` | `input.cfg` | Input trajectory file |
| `--input-format` | `-if` | auto | Trajectory file format |
| `--output-xyz` | `-o` | | Output annotated trajectory in EXTXYZ format |
| `--output-sro` | `-os` | `output_sro.txt` | SRO analysis output file |
| `--output-cna` | | `output_cna.txt` | CNA counts for every selected frame and atom |
| `--output-neighbor` | `-on` | `output_neighbor.txt` | Neighbor info output file |
| `--index` | `-id` | `index.map` | Index file |
| `--cutoff` | `-c` | `5.0` | Neighborhood cutoff radius |
| `--cutoff-inner` | `-ci` | `-1.0` | Inner cutoff for shell exclusion |
| `--pair-cutoff` | `-pc` | `false` | Enable type-dependent pairwise cutoff (bool) |
| `--rdf-kd` | `-rkd` | `false` | Use KD-tree for RDF (bool) |
| `--rdf-no-pbc` | `-rnp` | `false` | Disable PBC in KD-tree RDF (bool) |
| `--rdf-rho-all` | | `false` | Density from all-frame average (bool) |
| `--ref-rlimit` | `-rl` | `15.0` | Max radius for RDF |
| `--framestart` | `-s` | `0` | Starting frame index |
| `--frameend` | `-e` | | Ending frame index |
| `--skip-every` | `-skip` | `1` | Frames to skip between analyses |
| `--num-types` | `-nt` | `-1` | Number of atom types (-1 = auto-detect) |
| `--rdf` | `-r` | `false` | Compute RDF / pair correlation (bool) |
| `--bondangle` | `-b` | `false` | Compute bond angle distribution (bool) |
| `--dihedral` | `-d` | `false` | Compute dihedral angle distribution (bool) |
| `--sro` | `-sro` | `false` | Compute short-range order parameters (bool) |
| `--ha-index` | `-ha` | `false` | Compute Honeycutt-Andersen index (bool) |
| `--voro-index` | `-voro` | `false` | Compute Voronoi index (bool) |
| `--voro-neighbor` | `-vb` | `false` | Use Voronoi neighbors for analysis (bool) |
| `--voro-num-print` | `-vn` | `10` | Number of Voronoi indices to output |
| `--bop` | `-bop` | `false` | Compute bond order parameters (bool) |
| `--c-cna` | `-cna` | `false` | Compute common neighbor analysis (bool) |
| `--c-wcsro` | `-wcsro` | `false` | Compute Warren-Cowley SRO (bool) |
| `--avgpos` | `-caps` | `false` | Compute average atom positions (bool) |
| `--msd` | `-msd` | `false` | Compute mean squared displacement (bool) |
| `--vacf` | `-vacf` | `false` | Compute velocity autocorrelation (bool) |
| `--output-all-indices` | | `false` | Output all HA and Voronoi indices (bool) |
| `--avgpos-remove-drift` | | `false` | Remove COM drift when computing average positions (bool) |
| `--avgpos-fractional` | | `false` | Average unwrapped fractional coordinates instead of Cartesian (bool) |


`--c-cna` writes `frame atom central_type neighbor_type cna_index species_index count`
rows to `--output-cna`; frame and atom indices start at zero. Aggregated rows use
`neighbor_type=species_index=-1`. `--output-xyz` also receives integer `cna_total`
and `cna_<index>` columns. `cna:specie_info=true` adds species-resolved rows and
columns. Small periodic cells with repeated neighbor atom indices require a larger
supercell for CNA.

`--c-wcsro` writes the complete directed 2×2 matrix to `--output-sro`, pooled over
selected frames; combining it with `--sro` appends a separate WCSRO section. It
requires exactly two species and neighbors around both species. `wcsro:r_inn` and
`wcsro:r_out` define the strict shell `r_inn < r < r_out`; omitted or `-1` values
use `--cutoff-inner` and `--cutoff`. `wcsro:sro_type` supports `WC` and `PM`.
WC uses `1 - P(j|i)/c_j`, so unequal coordination can give different off-diagonal
values. CNA and SRO outputs must have distinct paths from each other and the
annotated trajectory, including symbolic and hard links.

`bondangle:sep`, `bondangle:low`, and `bondangle:high` use radians; output angles
use degrees. The default grid has 1° spacing and includes 0° and 180°. Observations
use the nearest grid point, with a partial final interval assigned to the last bin.

BOP uses `--cutoff` unless `bop:cutoff` is set; `bop:nnn` selects a fixed number
of nearest neighbors, while `bop:minnn` and `bop:maxnn` bound the radius-based
selection. With `--voro-neighbor`, BOP uses the supplied Voronoi neighbors instead.
`bop:average` includes the center atom when averaging spherical harmonics;
`bop:corase` averages the neighbors' scalar Q/W values and works independently
of `bop:average`. Atoms without neighbors receive zero Q/W values and a warning.
SRO shell radii are determined from sorted neighbor distances across the whole
configuration, so reordering atoms does not change the result.
Each bond belongs to one shell: distances less than `sro:width` above a shell's
minimum distance stay in that shell; the next distance starts a new shell.
`--avgpos` averages continuously unwrapped positions along periodic lattice
directions, using each frame's cell and the nearest image between successive frames.
This assumes motion of less than half a cell per periodic direction between frames;
the output positions may lie outside the cell.

### `calc_descriptors` — Compute Descriptors

Compute ML descriptors (ACSF, bond order, SO(3)/SO(4) bispectrum, MTM) for structures.

```bash
imrmlp calc_descriptors --settings=settings.ini -pot potential.mtp -in POSCAR -out descriptors.txt
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `input.inp` | Settings file (positional, optional) |
| `--input` | `-in` | `POSCAR` | Input structure file (positional, required) |
| `--output` | `-out` | `output-desc.xyz` | Output descriptor file |
| `--input-format` | `-if` | | Input format (empty = auto-detect) |
| `--output-format` | `-of` | `extxyz` | Output format |
| `--index` | `-id` | `index.map` | Index file |
| `--potfile` | `-pot` | `potential.mtp` | Potential file |
| `--dtype` | | `mtp` | Descriptor type: mtp, mtm, biso3, biso4, behler, bondorder |
| `--biso3-lmax` | | `4` | biso3: max angular L |
| `--biso3-rmin` | | `0.0` | biso3: minimum radial distance |
| `--biso3-rmax` | | `5.0` | biso3: cutoff radius |
| `--biso3-rsize` | | `8` | biso3: radial basis size |
| `--biso3-real-harmonics` | | `false` | biso3: opt-in real harmonic backend |
| `--cutoff` | `-c` | `5.0` | Cutoff radius for descriptor computation |
| `--fuse-species-info` | `-fs` | `false` | Fuse species information (bool) |
| `--discard-local-components` | `-d` | `false` | Omit per-atom local descriptor components (bool) |
| `--local-name` | | `desc` | Name of the local descriptor array |
| `--global-name` | | `desc` | Name of the global descriptor array |
| `--local-energy-components` | `-le` | `false` | Output local energy components (bool) |

SO3 / SO4 examples:

```sh
imrmlp calc_descriptors --dtype biso3 --input input.xyz --input-format extxyz \
  --biso3-lmax 3 --biso3-rsize 2 --biso3-rmax 4 --biso3-real-harmonics true
imrmlp calc_descriptors --dtype biso4 --settings so4.ini --input input.xyz
```

`so4.ini` accepts the following `biso4:` settings. Zero-based species indices must fit both weight arrays; local output always contains per-center descriptors.

| Setting | Default | Description |
| --- | --- | --- |
| `biso4:twojmax` | `9` | Maximum doubled angular order |
| `biso4:cutoff` | `5` | Neighbor cutoff |
| `biso4:radius` | `4/3 cutoff` | S3 radius |
| `biso4:weights` | `10 entries, 1` | Species density weights |
| `biso4:selfweights` | `10 entries, 0.6` | Species self weights |
| `biso4:normalize_U` | `false` | Normalize Wigner U matrices |
| `biso4:quaternion_convention` | `legacy` | legacy or q1/q2/q3/q4 (legacy defaults to q4) |
| `biso4:radial_angle_map` | `linear` | linear or atan radial embedding |
| `biso4:average` | `outer` | outer or inner-density |
| `biso4:contraction_backend` | `reference` | reference or exact sparse snap contraction |
| `biso4:wigner_backend` | `direct` | direct or experimental spline (twojmax <= 12) |

`inner-density` averages density coefficients before forming invariants; `outer` averages local features. The SO4 `snap` contraction backend does not require LAMMPS. The experimental spline backend does not support third-order force constants.

### `calc_eos` — Equation of State *(alias: `ev`)*

Fit equation of state from energy-volume or pressure-volume calculations.
Automatically generates strained structures, evaluates the potential, and fits
EOS parameters. Uses `--curve ev|pv` to select mode.

```bash
# Energy-volume curve (default; uses input.ini when --settings is omitted)
imrmlp ev --input=POSCAR --settings=eos.ini
imrmlp calc_eos --input=POSCAR --upper=0.05 --lower=-0.05

# Pressure-volume curve (also uses input.ini by default)
imrmlp calc_eos --curve=pv --input=POSCAR --press-ub=10 --press-lb=-10
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `input.ini` | Optional positional settings file; provides `[pot]` and `[relax]` settings |
| `--curve` | `-c` | `ev` | Curve type: `ev` (energy-volume) or `pv` (pressure-volume) |
| `--input` | `-in` | `POSCAR` | Input equilibrium structure |
| `--output` | `-o` | | EOS curve data output file |
| `--input-format` | | auto | Input format |
| `--index` | `-id` | `index.map` | Index file |
| `--skip-prelax` | | `false` | Skip preliminary relaxation (bool) |
| `--traj-format` | | `mlpcfg` | Output trajectory format |
| `--npos` | | `5` | Number of expansion points (> reference) |
| `--nneg` | | `5` | Number of compression points (< reference) |
| `--upper` | `-ub` | `0.05` | [EV] Upper volume expansion fraction (e.g. 0.05 = +5%) |
| `--lower` | `-lb` | `-0.05` | [EV] Lower volume compression fraction (e.g. -0.05 = -5%) |
| `--ninterp` | `-ninterp` | `100` | [EV] Number of interpolation points for EOS curve (at least 2) |
| `--out-veph` | | `veph.dat` | [EV] Volume-energy-pressure-enthalpy interpolated data file |
| `--fit-only` | `-fit` | `false` | [EV] Fit EOS using existing data only (skip calculation, bool) |
| `--lossless` | | `false` | [EV] Enable lossless structure output mode (bool) |
| `--output-eos` | | `eos.dat` | [EV] EOS fitting parameters output file (Birch-Murnaghan) |
| `--press-ub` | `-pub` | `10` | [PV] Upper pressure bound (GPa) |
| `--press-lb` | `-plb` | `-10` | [PV] Lower pressure bound (GPa) |

@note The settings file is optional at the command-line level and defaults to
`input.ini`. The `[pot]` section configures the potential and `[relax]` controls
the preliminary relaxation; EOS fitting options are command-line options.

EV fitting requires a stable equilibrium at a finite positive volume and a finite
positive bulk modulus. Invalid fits fail before writing fitted parameters or
interpolated data; quadratic fits with a valid minimum are supported.

### `calc_elastic` — Elastic Constants *(aliases: `elastic`, `mlpcalccellhess`)*

Compute elastic constants via analytical Hessian or numerical strain differentiation.
Use `--method` to select mode. Analytical mode requires potential with full analytical
Hessian support (e.g., MTP); numerical mode works with any EFS-capable potential.

```bash
# Analytical mode (auto-detected by default)
imrmlp elastic --potfile=potential.mtp --input=relaxed.vasp

# Numerical mode
imrmlp elastic --potfile=potential.mtp --input=relaxed.vasp --method=numerical --strain-magnitude=0.005
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | `potential.mtp` | Potential file (positional, required) |
| `--input` | `-in` | `POSCAR` | Input structure (positional, required) |
| `--method` | | | Calculation method: `auto` (default), `analytical`, or `numerical` |
| `--input-format` | `-if` | auto | Input format |
| `--index` | `-id` | `index.map` | Index file |
| `--settings` | | | Settings file |
| `--elastic-output` | `-eo` | `elastic.out` | Elastic constant output file |
| `--eigensys-output` | | `eigensys.out` | Analytical eigensystem output file |
| `--symprec` | `-symprec` | `1e-4` | Symmetry tolerance |
| `--eigzero-tol` | `-etol` | `1e-4` | Zero eigenvalue tolerance |
| `--use-gemo-mean` | `-g` | `false` | Use geometric mean for averaging (bool) |
| `--pressure` | `-p` | `0` | External pressure (GPa) |
| `--debug` | | `false` | [Numerical] Debug output (bool) |
| `--strain-magnitude` | `-emag` | `0.01` | [Numerical] Applied strain magnitude |
| `--num-strain-points` | `-n` | `5` | [Numerical] Number of strain points (odd) |
| `--mode` | | `0` | [Numerical] 0=stress-strain (default), 1=energy-strain |
| `--pottype` | `-pt` | `mtp` | [Numerical] Potential type |
| `--relax-correction` | `-cr` | `true` | [Analytical] Ionic relaxation correction (bool) |
| `--correction-l1` | `-c1` | `false` | [Analytical] 1st-order correction 1 (bool) |
| `--correction-l2` | `-c2` | `false` | [Analytical] 1st-order correction 2 (bool) |
| `--compute-eigsys` | | `false` | [Analytical] Compute full super-Hessian eigensystem (bool) |

The primary stiffness and the Voigt/Reuss/Hill properties are the relaxed-ion
material tensor (C).  At nonzero `--pressure`, the output also contains the
Born/Wallace tangent tensor used for the mechanical-stability eigenvalue test.
For the numerical method, place ionic-relaxation controls such as
`relax:gtol` and `relax:maxiteration` in the file passed with `--settings`.

**Method comparison:**

| Aspect | Analytical | Numerical |
|--------|---------------|-------------------|
| **Method** | Analytical Hessian | Numerical strain differentiation |
| **Potential requirement** | Must support `getEFSHessian` | Any EFS-capable potential |
| **Accuracy** | Exact (to numerical precision) | Depends on strain step size |
| **Speed** | Single evaluation | Many EFS evaluations (6+ strain patterns) |
| **Best for** | MTP with Hessian support | EAM, LJ, or any potential without Hessian |

### `calc_fc` — Force Constant Export *(aliases: `fc`, `mlpcalcfc`)*

Calculate atomic Hessians from a potential and export force constants. Use Phonopy
for dispersion, DOS, thermodynamics and mode animation.

```bash
imrmlp calc_fc --potfile potential.mtp --input POSCAR --supercell 3 3 3 --reduced=false
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | `potential.mtp` | Potential file (required) |
| `--input` | `-in` | `POSCAR` | Input structure (required) |
| `--potential-type` | `-pt` | `mtp` | Potential type |
| `--input-format` | `-if` | auto | Input format |
| `--index` | `-id` | `index.map` | Index file |
| `--settings` | | `fc.ini` | Settings file (optional) |
| `--supercell` | `-dim` | | Supercell dimensions (variable length, e.g. `3 3 3` or 9 matrix entries) |
| `--symprec` | `-tol` | `1e-5` | Symmetry tolerance |
| `--write-force-constant` | `-fc` | `true` | Write FORCE_CONSTANT file (bool) |
| `--reduced` | `-rd` | `true` | Use symmetry-reduced FORCE_CONSTANTS (bool) |
| `--hessian-method` | | `analytic` | Atomic Hessian backend: `analytic`, `auto`, `force_fd`, `energy_fd`, or `hybrid` |
| `--hessian-nudge-pos` | | `0.001` | Cartesian finite-difference displacement (Angstrom) |
| `--hessian-cutoff` | | `0` | Finite-difference neighbor cutoff; non-positive uses the potential cutoff |
| `--check-symm` | | `false` | Report tensor acoustic sum-rule residuals |
| `--lowest-modes` | | `0` | Find N lowest internal Gamma modes using analytic HVPs; 0 exports force constants |
| `--lowest-mode-tol` | | `1e-8` | Relative residual tolerance for lowest modes |
| `--lowest-mode-max-matvecs` | | `0` | HVP limit; non-positive uses max(400,20*N) |

`FORCE_CONSTANTS` blocks use eV/Å² and the atom order in `supercell_created.xsf`;
`sup_debug_efs.xyz` records E/F/S for the same cell. Select `--reduced=false` for
a full matrix.

The default `analytic` method fails if the potential lacks an analytical Hessian;
`auto` falls back to force finite differences. `energy_fd` and `hybrid` need
quadratically many evaluations and are intended for small-system validation.
`--check-symm` reports residuals without changing the force constants.

`--lowest-modes=N` is a matrix-free HVP diagnostic. It projects out the three
mass-weighted translations and writes only N internal modes to `EIGENVALS`,
`EIGENVECTORS` and `EIGENSYS`, without force constants. It requires analytic HVPs
and cannot be combined with `--check-symm`. Masses use
the third column of `index.map`, falling back to tabulated elemental masses.

See the [Python adapter](python.md#phonopy-integration) for Phonopy integration.
To import analytic force constants, first generate the supercell in Phonopy and
write it unchanged as the `calc_fc` input, omit `--supercell`, and select
`--reduced=false`. Assign the matrix to `phonon.force_constants` only when its
cell and atom ordering on both axes match `phonon.supercell`; equal atom counts
alone do not establish this correspondence.

### `ti_fit_fc` — positive finite-temperature effective force constants

`ti_fit_fc` fits MD/ensemble force constants and exports LAMMPS harmonic references. Select
`--workflow md|ensemble|md-ensemble|export` explicitly:

```bash
imrmlp ti_fit_fc --workflow md-ensemble \
  --traj md.extxyz --traj-format extxyz \
  --potfile potential.mtp --index index.map \
  --temperature 600 --output-prefix BETA_600K
```

Without `--input`, an MD workflow unwraps a fixed-cell NVT trajectory, removes
mass-weighted center-of-mass drift, and uses its centroid as the reference.
Variable-cell trajectories are rejected; obtain the average NPT cell first and
run a fixed-cell NVT trajectory there. Forces are recomputed with the selected
potential by default. Use `--force-source trajectory` explicitly to consume
stored trajectory forces.

Periodic images are aligned after removing a common translation, including
when center-of-mass drift crosses half a cell. Relative crystalline motion
between atoms must remain below half a period.

Fitting and sampling operate in the mass-weighted internal subspace. The
default staged fit updates modal stiffnesses first, then coupled soft and
near-degenerate blocks using a positive square-root representation. A
self-consistent run counts convergence only after the modal warmup has ended.
Both fitting stages minimize the weighted mean squared force residual plus a
`--ridge` penalty toward the current FC; duplicating samples leaves its strength unchanged. A
self-consistent iteration samples an antithetic Gaussian ensemble, evaluates
true MTP forces, and applies a damped root-space update. The default
`--initial-floor-thz=1` controls both the initial stabilization and the physical
sampling stiffness in every self-consistent iteration, while
`--pd-floor-thz=1e-4` is only the strict numerical positive floor of the fitted FC.
Candidates that violate displacement or minimum-distance limits are rejected as
complete antithetic `+q/-q` pairs and replaced. With
`--max-sampling-attempts=0`, the default budget is twenty times the requested
ensemble size. Exhausting that budget reports rejection reasons and geometry
extrema and fails; it never changes the sampling floor or fits an undersized set.

Outputs include `<prefix>_FORCE_CONSTANTS`, the centroid/reference structure,
iteration CSV and diagnostics, an independent validation ensemble, and the
versioned `<prefix>_LAMMPS_REFERENCE` consumed by LAMMPS
`pair_style harmonic/reference`. Run `imrmlp ti_fit_fc --help` for convergence,
soft-block, and sample-safety options.

The report keeps the MD fitting training/holdout errors and additionally
records `final_md_force_rmse` for the final fitted reference. Training errors
describe the centered fitting objective. Holdout errors, `final_md_force_rmse`,
and `harmonic_validation_force_rmse` use the actual harmonic force residual,
without removing a validation-set force offset. All these errors use
mass-weighted internal force coordinates (eV/Angstrom/sqrt(amu)). The
`lammps_*` diagnostics and `# lammps_mode` table always describe the exact FC
written to `<prefix>_LAMMPS_REFERENCE`; use these values for TI acceptance and
analytic harmonic free energies. The iteration CSV and report also record
sampling attempts, rejected samples, acceptance rates, and geometry extrema;
a persistently low acceptance rate indicates an unsuitable reference or
sampling floor rather than a reason to keep relaxing safety limits.

The LAMMPS file can be exported with sparse 3x3 atom-pair FC blocks. After
truncation, onsite blocks are reconstructed to enforce the acoustic sum rule;
when requested, the smallest isotropic stabilizer on the retained connected
graph raises the minimum internal frequency to the target. Existing local FCs
can be converted without evaluating a potential when the input reference already
contains a finite energy; otherwise `ti_fit_fc` evaluates the selected potential:

```bash
imrmlp ti_fit_fc --workflow export \
  --input REFERENCE.xyz --input-format extxyz \
  --initial-fc FORCE_CONSTANTS --index index.map \
  --lammps-fc-source initial --lammps-fc-block-threshold 0.01 \
  --lammps-fc-min-frequency-thz 0.5 \
  --write-validation-ensemble false --output-prefix TI_SPARSE
```

Use `--lammps-fc-cutoff` for distance truncation. Always inspect the retained
block fraction, relative Hessian change, frequencies before/after stabilization,
and ASR error in the report. An ensemble fit or global spectral projection can
introduce sizable long-range blocks; truncating that `fitted` FC may therefore
change the reference strongly. Select `--lammps-fc-source initial` when the
original local FC is the model that should be preserved.


| Option | Alias | Default | Description |
| --- | --- | --- | --- |
| `--workflow` |  | `—` | required: md, ensemble, md-ensemble, or export |
| `--traj` |  | `—` | MD trajectory |
| `--traj-format` |  | `—` | trajectory format |
| `--input` | `-in` | `—` | optional explicit reference structure |
| `--input-format` | `-if` | `—` | reference structure format |
| `--initial-fc` |  | `—` | optional initial FORCE_CONSTANTS |
| `--potfile` | `-pot` | `potential.mtp` | potential used to evaluate forces |
| `--potential-type` | `-pt` | `mtp` | potential type |
| `--index` | `-id` | `index.map` | index map |
| `--force-source` |  | `potential` | potential or trajectory |
| `--root-representation` |  | `sqrt` | normal, sqrt, or root4 |
| `--initial-stabilization` |  | `spectral-floor` | negative-floor, spectral-floor, or abs |
| `--lammps-fc-source` |  | `fitted` | fitted or initial FC for sparse LAMMPS export |
| `--output-prefix` |  | `TI_FC` | output filename prefix |
| `--start` |  | `1` | first trajectory frame, one-based |
| `--end` |  | `2147483647` | last trajectory frame, one-based and inclusive |
| `--stride` |  | `1` | trajectory frame stride |
| `--temperature` | `-T` | `300.0` | ensemble temperature in K |
| `--ensemble-size` |  | `128` | samples per self-consistent iteration |
| `--max-sampling-attempts` |  | `0` | maximum candidate samples; zero uses 20 times the requested ensemble size |
| `--seed` |  | `5489` | base random seed |
| `--initial-floor-thz` |  | `1.0` | initial and iterative physical sampling floor |
| `--pd-floor-thz` |  | `1e-4` | strict positive-definite floor |
| `--ridge` |  | `1e-8` | ridge toward the current reference |
| `--mixing` |  | `0.25` | root-space update fraction |
| `--warmup-iterations` |  | `5` | modal-only iterations |
| `--min-iterations` |  | `3` | minimum self-consistent iterations |
| `--max-iterations` |  | `20` | maximum self-consistent iterations |
| `--consecutive-converged` |  | `2` | required converged iterations |
| `--relative-tolerance` |  | `1e-3` | relative dynamical-matrix tolerance |
| `--frequency-tolerance-thz` |  | `0.02` | maximum frequency-change tolerance |
| `--soft-block-frequency-thz` |  | `2.0` | soft coupled-block cutoff |
| `--soft-block-max-modes` |  | `32` | maximum soft coupled-block size |
| `--degeneracy-gap-thz` |  | `0.05` | near-degenerate block gap |
| `--degeneracy-block-max-modes` |  | `8` | maximum near-degenerate block size |
| `--max-displacement` |  | `1.0` | hard ensemble displacement limit in Angstrom |
| `--min-distance-ratio` |  | `0.65` | hard pair-distance ratio |
| `--symprec` | `-tol` | `1e-5` | space-group tolerance |
| `--hessian-method` |  | `analytic` | analytic, auto, force_fd, energy_fd, or hybrid |
| `--hessian-nudge-pos` |  | `0.001` | finite-difference displacement |
| `--hessian-cutoff` |  | `0.0` | finite-difference cutoff |
| `--lammps-fc-cutoff` |  | `0.0` | LAMMPS FC real-space cutoff in Angstrom; zero disables |
| `--lammps-fc-block-threshold` |  | `0.0` | LAMMPS FC off-diagonal block norm threshold |
| `--lammps-fc-min-frequency-thz` |  | `-1.0` | minimum sparse-reference frequency; negative uses pd-floor-thz |
| `--lammps-fc-stabilize` |  | `true` | add a sparse isotropic graph stabilizer when needed |
| `--write-validation-ensemble` |  | `true` | write final exact-force validation ensemble |

Exit status is 0 on success, or 2 when self-consistent iterations exhaust the budget without convergence (results and diagnostics are still written).

### `phonopyfc` — Phonopy FORCE_SETS *(alias: `mlpphdisp`)*

Generate FORCE_SETS file from Phonopy displacement input.

```bash
imrmlp mlpphdisp -pot potential.mtp -d phonopy_disp.yaml
imrmlp phonopyfc -pot potential.mtp --disp=phonopy_disp.yaml
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | `potential.mtp` | Potential file (positional) |
| `--disp` | `-d` | `phonopy_disp.yaml` | Phonopy displacement YAML (positional) |
| `--index-file` | `-id` | `index.map` | Species index map |
| `--settings-file` | `-i` | `settings.ini` | Settings file; the potential configuration is read from `pot:` |

### `calc_ferr` — Free Energy Error Estimation

Estimate free energy errors for thermodynamic integration. Both datasets must
have the same frame count, the same (nonzero) atom count, and energy labels.

```bash
imrmlp calc_ferr --input=dft.cfg --prediction=mlp.cfg --temperature=300
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `dft.cfg` | Reference dataset |
| `--prediction` | | `mlp.cfg` | Prediction dataset |
| `--temperature` | `-t` | `300` | Temperature in K; must be positive |
| `--input-format` | `-if` | auto | Format for both datasets |
| `--index` | `-id` | `index.map` | Species index map |

### `calc_trjavg` — Trajectory Average

Compute average structure from trajectory data.

```bash
imrmlp calc_trjavg trajectory.cfg average.cfg
imrmlp calc_trjavg trajectory.cfg average.cfg --start=100 --end=500 --wrap
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input-file` | | | Input trajectory (positional) |
| `--output-file` | | `POSCAR_IDEAL` | Output average structure (positional) |
| `--start` | `-st` | `1` | First frame, 1-based |
| `--end` | `-ed` | last | Last frame, 1-based and inclusive |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--index` | `-id` | `index.map` | Species index map |
| `--loadrate` | `-lr` | `0` | Read every N-th structure; `1` enables sequential one-by-one mode |
| `--wrap` | `-w` | `true` | Wrap the averaged structure into the cell |
| `--average-coordinates` | | `auto` | `auto`, `cartesian`, or `fractional` |
| `--drift` | | `auto` | `auto`, `remove`, or `keep` COM drift |
| `--append` | `-a` | `false` | Append to an existing output file |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

Trajectory averaging shares a streaming kernel with `calc_prop --avgpos`.
`calc_trjavg` removes mass-weighted COM drift for fixed cells and averages
unwrapped fractional coordinates for variable cells. `--avgpos` averages
unwrapped Cartesian positions without drift removal. Masses use the third
column of `index.map` or elemental defaults. The average geometry does not
inherit E/F/S labels from an input frame.

Use `calc_trjavg --average-coordinates=auto|cartesian|fractional` and
`--drift=auto|remove|keep` to select the averaging policy explicitly. Drift
removal requires a fixed cell. `calc_prop --avgpos-fractional` selects fractional
averaging and `--avgpos-remove-drift` removes COM drift; both default to false.
Site energies are averaged only when every frame supplies a complete array.

### `findint` — Find Interstitial Sites

Find interstitial sites in crystal structures using Voronoi tessellation.

```bash
imrmlp findint --input=POSCAR --output=sites.cfg --insert=Li --min-dist=1.0
imrmlp findint --input=POSCAR --insert=H --top=5 --sites-only
imrmlp findint POSCAR --near-elem=O --near-radius=2.0 --local
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | | `POSCAR` | Input structure (positional) |
| `--output` | `-o` | `POSCAR_INT` | Output structure |
| `--index-file` | `-id` | | Species map; use `--near-idx` to select atoms |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | auto | Output format |
| `--insert` | | | Element to insert at the best sites (empty = list sites only) |
| `--top` | `-n` | `-1` | Keep the top-N sites by cavity size (-1 = all) |
| `--min-dist` | `-md` | `0.4` | Minimum distance from host atoms (Å) |
| `--cluster-tol` | `-ct` | `0.8` | Cluster tolerance for merging duplicate sites (Å) |
| `--vertices` | `-v` | `true` | Use Voronoi vertices |
| `--edges` | | `false` | Use Voronoi edge centers |
| `--faces` | | `false` | Use Voronoi face centers |
| `--volprec` | `-vp` | `0.5` | Minimum Voronoi cell volume |
| `--sites-only` | | `false` | Output interstitial sites only, without host atoms |
| `--no-fast-cluster` | | `false` | Use the legacy O(N²) candidate clustering |
| `--verbose` | `-vb` | `0` | Verbosity level |
| `--near-elem` | `-ne` | | Search near element(s), comma-separated (e.g. `Fe,Cr`) |
| `--near-pos` | `-np` | | Search near a fractional position `x,y,z` |
| `--near-pos-car` | `-npc` | | Search near a Cartesian position `x,y,z` |
| `--near-idx` | `-ni` | | Search near 1-based atom indices, comma-separated |
| `--near-radius` | `-nr` | `3.0` | Near-site search radius (Å) |
| `--local` | `-lcl` | `false` | Use local Voronoi construction |

---

## I/O & Conversion

Format conversion and dataset management commands. `convstr` is the universal
format translator — virtually all structure data passes through it. Use `split_set`
to partition datasets for training/validation/test splits, and the de-duplication
commands to clean up redundant structures.

@note **Want to remove duplicates?** Use `remove_duplicate_atoms` (`rmdups`) to
remove duplicate atoms WITHIN each structure (e.g., overlapping atoms after a
supercell operation). Use `remove_duplicate_structures` (`unique`) to remove
duplicate structures FROM a dataset (e.g., symmetry-equivalent configurations).

### `convstr` — Convert Structures

Convert between all supported file formats. Comprehensive filtering and selection.

```bash
imrmlp convstr input.cfg output.extxyz
imrmlp convstr POSCAR output.lammps
imrmlp convstr trajectory.lammpstrj output.cfg
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--inputfiles` | `-in` | — | Input structure file(s) (positional, required, multi-file) |
| `--outputfile` | `-out` | — | Output file (positional, required) |
| `--input-format` | `-if` | auto-detect | Input format |
| `--detect-format` | | `false` | Detect input content when `--input-format` is omitted (bool) |
| `--input-unit-system` | | — | Explicit EXTXYZ source units: `canonical`, `metal`, `atomic`, or `real` |
| `--output-format` | `-of` | auto-detect | Output format |
| `--index` | `-id` | `index.map` | Index file |
| `--wrap` | `-w` | `false` | Wrap coordinates to periodic boundaries (bool) |
| `--resetid` | `-rid` | `true` | Reset atom type IDs to 0,1,2,... |
| `--start` | `-st` | `0` | Starting image index |
| `--end` | `-ed` | `INT_MAX` | Ending image index |
| `--outrate` | `-or` | `0` (all) | Write every Nth frame |
| `--shuffle` | | `false` | Randomly shuffle image order (bool) |
| `--sort-by` | | — | Comma-separated sort keys: `energy`, `fenergy`, `fitness`, `volume`, `size`, `id`, or global attributes |
| `--sort-order` | | `asc` | Sort direction per key; single value or comma-separated |
| `--reverse` | `-r` | `false` | Reverse image order (bool) |
| `--last` | `-last` | `false` | Output only the final frame (bool) |
| `--drop` | `-d` | `false` | Skip non-converged OUTCAR frames (bool) |
| `--overwrite` | `-ov` | `false` | Overwrite existing file (bool) |
| `--append` | `-a` | `false` | Append to existing file (bool) |
| `--select` | `-s` | — | Select specific image indices (variable args) |
| `--loadrate` | `-lr` | `0` | Read images to memory every Nth structure |
| `--ostart` | `-ost` | `0` | Output starting image index |
| `--oend` | `-oed` | `INT_MAX` | Output ending image index |
| `--bstart` | `-b` | `true` | Start IDs from zero (bool) |
| `--outsuffix` | `-suff` | — | Suffix appended to output filenames |
| `--separator` | `-sep` | — | Separator between filename and ID |
| `--magnetic` | `-mag` | `false` | Read magnetic moments (bool) |
| `--magnetic-non-colinear` | `-mncl` | `false` | Assume non-collinear magnetism (bool) |
| `--discard` | | — | Discard specific properties (variable args) |
| `--rextent` | | `5.0` | Cell padding for non-PBC structures |
| `--force-one` | | `false` | Write all frames to single file (bool) |
| `--force-tilt` | | `false` | Force tilted cell format even for orthogonal (bool) |
| `--write-symbol` | | `true` | Write atomic symbols in output (bool) |
| `--pack-atoms` | | `true` | Pack atom types in POSCAR output (bool) |
| `--apply-disp` | `-ad` | `false` | Apply random displacement to images (bool) |
| `--rand-type` | `-rt` | `0` | Random displacement type |
| `--disp-mag` | `-dm` | `0.05` | Displacement magnitude |
| `--label` | `-lb` | | Add custom per-structure labels (variable args) |
| `--value` | `-lv` | | Custom per-structure label values (variable args) |
| `--label-expr` | | | Scalar metadata assignments, e.g. `energy_per_atom=energy/size` (variable args) |

### `split_set` — Split Dataset

Split a structure dataset into N equal subsets or train/test.

```bash
imrmlp split_set --nsplit=2 --shuffle yes --input=dataset.cfg --output-prefix=split_
# produces split_0.xyz, split_1.xyz (outputs are always in extxyz format)
imrmlp split_set dataset.cfg split_ --ratios=8,1,1 --shuffle yes
# produces approximately 80% / 10% / 10% subsets
```

| Option | Alias | Description |
|--------|-------|-------------|
| `--nsplit` | `-n` | Split evenly into N subsets (optional; default 10) |
| `--ratios` | | Comma-separated nonnegative relative weights; at least two, with a positive sum; determines subset count |
| `--shuffle` | `-s` | Random shuffle before splitting (bool) |
| `--input` | | Input structure set (positional, required) |
| `--output-prefix` | | Output file prefix (positional, required) |

Equal mode distributes structures round-robin after any shuffle. Ratio mode
rounds subset sizes by largest remainder (ties favor earlier subsets), then
takes consecutive segments. Zero-sized subsets produce empty output files.
If `--nsplit` is explicitly supplied with `--ratios`, it must equal the number
of ratios. There is no shuffle seed option; save the split files for reproducibility.

### `remove_duplicate_atoms` *(alias: `rmdups`)*

Remove duplicate atoms based on distance criteria.

```bash
imrmlp rmdups --input=overlap.cfg --output=cleaned.cfg --rtol=0.5
imrmlp rmdups --input=overlap.cfg --output=cleaned.cfg --rtol=0.5 --mode=merge
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure (positional) |
| `--output` | `-o` | `POSCAR_REMOVED_DUPS` | Output structure (positional) |
| `--rtol` | `-rt` | `0.5` | Duplicate distance tolerance (Å); must be positive |
| `--mode` | `-m` | `delete` | `delete` or `merge` |
| `--input-format` | `-if` | `POSCAR` | Input format |
| `--output-format` | `-of` | `POSCAR` | Output format |
| `--index` | `-id` | `index.map` | Species index map |

`--mode=delete` keeps the representative atom nearest the duplicate group's
mean position and removes the rest. `--mode=merge` first moves that
representative to the mean position, then removes the rest.

### `remove_duplicate_structures` *(alias: `unique`)*

Remove duplicate structures based on symmetry equivalence.

```bash
imrmlp unique --input=dataset.cfg --output=unique_set.cfg --sym-prec=1e-3
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | | Input dataset (positional) |
| `--output` | `-out` | `output-dedup.xyz` | Output dataset (positional) |
| `--sym-prec` | `-sprec` | `1e-4` | Symmetry tolerance (cell and space group) |
| `--vol-prec` | `-vprec` | `1e-3` | Volume-comparison tolerance |
| `--pos-tol` | `-ptol` | `0.1` | Atomic-position matching tolerance (Å) |
| `--preserve` | `-p` | `true` | Preserve the original cell instead of the standardized one |
| `--allow-reflections` | | `true` | Treat mirror images as equivalent |
| `--fastalgo` | | `true` | Use cached descriptors for fast comparison |
| `--desc-factor` | | `1.0` | Descriptor tolerance multiplier for `--fastalgo` |
| `--index` | `-id` | `index.map` | Species index map |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | `xyz` | Output format |
| `--overwrite` | `-ov` | `false` | Overwrite an existing output file |

The node index file `output-nodup.id` is written alongside the dataset.

---

## Potential Utilities

Maintenance and conversion commands for MTP potential files. They handle format
conversion (to spline tables for LAMMPS), species reordering, coefficient remapping,
and basis inspection — all without modifying the trained model's predictions.

### `mtp2spline` — Convert MTP to Spline

Convert MTP-type potentials to spline representation. Outputs radial functions
as spline coefficients for use in LAMMPS `pair_style mlip spline`.

```bash
imrmlp mtp2spline potential.mtp spline_output
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | | `potential.mtp` | Input potential (positional) |
| `--output` | | `splinedpot.mtp` | Output splined potential (positional) |
| `--delta-r` | `-dr` | `0.01` | Spline interval |
| `--r-min` | `-min` | `-1` | Inner radius kept constant; negative uses each pair's basis minimum |

`--r-min R` keeps radial table values constant below `R`. By default (`-1`),
each species pair uses its radial basis minimum, including potentials with
separate pair bases. Values beyond each pair's cutoff are zero.

### `pot2table` — Export Tabulated Potentials

Export pair and/or three-body interaction tables by sampling a supported potential
on radial/angular grids. It can also write LAMMPS table-format files.

```bash
imrmlp pot2table --potfile=my_pot.sw --pottype=sw --pair --pair-output=pair.table
imrmlp pot2table --potfile=my_pot.sw --pottype=sw --threebody --threebody-output=tb.table
imrmlp pot2table --potfile=my_pot.tersoff --pottype=tersoff --pair --lammps
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | | Potential file path (required) |
| `--pottype` | `-t` | | Potential type, e.g. `lj612`, `sw`, `tersoff`, `eampoly` |
| `--index` | `-id` | `index.map` | Index map file |
| `--pair-output` | `-po` | `pair_table.out` | PairTable output file |
| `--threebody-output` | `-to` | | ThreebodyTable output file |
| `--lammps-prefix` | `-lp` | | Prefix for LAMMPS table files |
| `--nr` | `-nr` | `1000` | Number of radial grid points |
| `--ntheta` | `-nt` | `181` | Number of angular grid points |
| `--rmin` | `-rmin` | `0.01` | Minimum radius |
| `--rmax` | `-rmax` | potential cutoff | Maximum radius override |
| `--pair` | | `false` | Export pair table (bool) |
| `--threebody` | | `false` | Export three-body table (bool) |
| `--lammps` | `-lm` | `false` | Also write LAMMPS table format (bool) |
| `--settings` | | | Settings file for potential loading |

Pair tables contain exactly `N` uniformly spaced grid points, including both
endpoints. `--nr` must be at least 3 for pair tables and at least 2 for three-body
tables; `--ntheta` must be at least 2. Radii must be finite and satisfy
`0 < rmin < rmax`. LAMMPS pair tables include the radial force `-dE/dr`.

The three-body format has one radial dimension and samples equal bond lengths.
For unequal bonds, `ThreebodyTable` uses
`V(rij, rik, theta) = (T(rij, theta) + T(rik, theta)) / 2`, with forces derived
from this energy. This approximation cannot represent a general three-body
potential with two independent radial dimensions.

### `adjustmtp` — Reorder MTP Potential

Reorder an MTP potential's species to match a different ordering convention.
Useful when the index.map used during training differs from the one used in
production — this command permutes the coefficient layout without retraining.

```bash
imrmlp adjustmtp --input=original.mtp --output=reordered.mtp --neworder 0 2 1
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `potential.mtp` | Input potential (positional) |
| `--output` | `-out` | `output-adj.mtp` | Output potential (positional) |
| `--neworder` | | | New species order (positional, multiple values); longer orders append randomly initialized species |
| `--elements` | | | Element symbols in the output species order; required when `--neworder` appends species |

### `mtpbasis` — Compare MTP Bases and Remap Parameters

Check two classic linear MTP bases using exact symbolic integer arithmetic.
Moment IDs, scalar ordering, and contraction paths may differ.  `auto` first
looks for a one-to-one scalar permutation and then checks whether the two
scalar sets span the same space using exact rational elimination.  The
production `mid_tid_mapping` files are optional fast-path hints and are always
verified against the symbolic expressions.

```bash
# Read-only comparison, including a JSON report
imrmlp mtpbasis --source=pot1.mtp --target=pot2.mtp \
  --equivalence=auto --report=basis-report.json

# Verify production mappings and transfer all classic linear MTP parameters
imrmlp mtpbasis --source=pot1.mtp --target=pot2.mtp \
  --source-map=map1.txt --target-map=map2.txt \
  --source-index=index1.map --target-index=index2.map \
  --output=pot2-remapped.mtp
```

| Option | Alias | Description |
|--------|-------|-------------|
| `--source` | `-s` | Source MTP basis or potential (required) |
| `--target` | `-t` | Target MTP basis or potential (required) |
| `--source-map`, `--target-map` | `-sm`, `-tm` | Optional paired `moment_id tensor_id` files |
| `--source-index`, `--target-index` | `-si`, `-ti` | Optional paired species-order maps |
| `--equivalence` | `-eq` | `auto` (default), `item`, or `span` |
| `--output` | `-o` | Write target topology with source parameters; omitted means compare only |
| `--report` | | Write ranks, diagnostics, permutation/transform, and compatibility as JSON |
| `--overwrite` | `-ov` | Permit replacing an existing output/report |

An output file is accepted only when topology, radial layout, and any supplied
species maps are compatible.  It must differ from both inputs.  The command
copies radial and species coefficients and applies the exact scalar transform
to all moment coefficients.  MTP_M, NN, and LES parameter sections are not
silently discarded and cannot be remapped by this command.  Exit status is 0
for equivalence, 1 for a valid but non-equivalent comparison, and 2 for invalid
input or mapping/write failure.



### `mtmbasis` — Generate Moment Tensor Basis

For a runnable route from basis generation through ordinary MTP fitting,
validation and calculations, see the [MTP walkthrough](../../examples/mtp-workflow/README.md).

Generate a moment tensor basis: scalar invariants, intermediate moment tensors,
and contraction rules. The command uses top-level keys in its settings file
(`max_lev`, `max_dim`, `max_mu`, `max_nv`, and GA search controls), not the
regular `[pot]`/`pot:` potential-loading section.

```bash
imrmlp mtmbasis --settings=mtmbasis.ini
imrmlp mtmbasis --settings=mtmbasis.ini --generate-only
imrmlp mtmbasis --settings=mtmbasis.ini --alphapool=alphapool.dat
```

Normal generation builds an alpha pool, searches for an efficient contraction
tree layout with the configured search mode, and writes candidate MTP basis
templates as `potential.mtp_<iteration>`. It also writes `conns_exp.log` with
contraction count diagnostics. The generated MTP files contain the basis metadata
(`alpha_index_basic`, `alpha_index_times`, and `alpha_moment_mapping`) but no
trained coefficients. Complete the species/radial settings of the selected
template before fitting with `trainmtp`. `initmtpm` targets MTP-M, not ordinary MTP.

Pool-only modes are useful for staging expensive basis enumeration:

- `--generate-only` writes the deduplicated alpha pool to `alphapool.dat` and exits.
- `--sample` randomly samples the generated/loaded pool by `--sample-rate` and
  writes `sampled_alphas.dat`.
- `--advanced-sample` uses the top-level `sample_rates` setting, preserves all
  one-moment scalar entries, samples higher tensor dimensions separately, writes
  `sampled_alphas.dat`, and exits.

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | `settings.ini` | Settings file. There is no `-i` alias for this command. |
| `--alphapool` | `-pool` | | Read an existing alpha pool instead of generating one from limits. |
| `--moments` | `-mts` | | File listing allowed `(mu, nu)` moment pairs: first value is count, followed by that many `mu nu` pairs. |
| `--generate-only` | `-g` | false | Generate and save `alphapool.dat`, then exit before GA search. |
| `--sample` | `-s` | false | Sample the current pool and save `sampled_alphas.dat`. |
| `--sample-rate` | `-rate` | 1.0 | Random sampling fraction for `--sample`; values outside `[0,1]` are reset to 1. |
| `--advanced-sample` | `-as` | false | Dimension-aware sampling controlled by `sample_rates`. |
| `--potfile` | `-pot` | `potential.mtp` | Parsed by the CLI, but the current search output path is hard-coded as `potential.mtp_<iteration>`. |
| `--convert` | | false | Legacy conversion path; marked "not working" in the implementation. |

---

## Utility Commands

### `attach_hessian` — Attach Hessian or HVP Labels *(alias: `hessian2xyz`)*

Attach a DFPT (`vasprun.xml`) or finite-difference (`FORCE_CONSTANTS`) Hessian
to an extxyz structure. With `--hvp-probes`, random Hessian-vector labels are
also written for HVP-supervised training.

```bash
imrmlp attach_hessian --input POSCAR --hessian vasprun.xml --output hessian.xyz --source dfpt
imrmlp hessian2xyz --input POSCAR --hessian FORCE_CONSTANTS --output hvp.xyz --hvp-probes 8
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure (positional) |
| `--output` | `-out` | `hessian.xyz` | Output extxyz dataset (positional) |
| `--hessian` | | input file | DFPT `vasprun.xml` or FD `FORCE_CONSTANTS` |
| `--source` | `-s` | `auto` | Hessian source: `auto`, `dfpt`, or `fd` |
| `--vasprun-key` | | `auto` | `varray` name in `vasprun.xml` |
| `--frame` | `-f` | `-1` | Structure frame; -1 uses the last frame |
| `--scale` | | `1.0` | Multiply the Hessian by this factor |
| `--symmetrize` | | `false` | Replace `H` by `0.5*(H+H^T)` |
| `--hvp-probes` | | `0` | Number of random Hessian-vector label pairs |
| `--hvp-seed` | | `42` | Random seed for HVP directions |
| `--hvp-only` | | `false` | Store HVP labels only, omitting the dense Hessian |
| `--input-format` | `-if` | auto | Input format |
| `--output-format` | `-of` | `extxyz` | Output format |

`--hvp-only` requires `--hvp-probes > 0`; HVP probes require at least two atoms.

### `calc_err` — Error Calculation *(alias: `err`)*

Compare potential predictions against reference data. Computes RMSE and MAE
for energy, forces, and stress.

```bash
imrmlp err -pot potential.mtp -in reference.cfg
imrmlp calc_err -pot potential.mtp -in reference.cfg -inprd predicted.cfg -out summary.txt
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--potfile` | `-pot` | | Model used to generate predictions; may instead be set in `pot:potfile`; unnecessary with `--input-predict` |
| `--pottype` | `-pt` | `mtp` | Potential type |
| `--input` | `-in` | `valid.cfg` | Reference data file (required) |
| `--input-format` | `-if` | auto | Input format |
| `--input-predict` | `-inprd` | | Prediction file for comparison |
| `--index` | `-id` | `index.map` | Index file |
| `--output-energy-diff` | `-oe` | | Output energy difference file |
| `--output-energy-at-diff` | `-oeat` | | Output per-atom energy difference file |
| `--output-force-diff` | `-ofce` | | Output force difference file |
| `--output-stress-diff` | `-ofst` | | Output stress difference file |
| `--output-virial-diff` | `-ofvr` | | Output virial difference file |
| `--output-summary` | `-out` | | Output error summary file (default: stdout) |
| `--show-progress` | `-sp` | `false` | Report completed structure counts (bool) |
| `--mpi-mode` | | `replicated` | MPI evaluation mode: `replicated` or `frames` |
| `--percentile` | `-q` | `0` | Percentile ratio for reporting (e.g. `0.95`; 0 disables) |
| `--settings` | | | Optional settings file (positional) |

Only properties present in both paired frames enter the corresponding metrics.
Reference and prediction datasets must contain the same number of frames, with
matching atom counts and species order. `--input-predict` compares saved
predictions without evaluating a potential; both files use `--input-format`.
Energy is reported in eV and eV/atom, forces in eV/Å, stress in GPa, and virial
in eV. Force-vector RMSE and Cartesian-component RMSE use different
normalizations; compare the same reported quantity across runs.

### `smooth_traj` — Smooth Trajectories

Apply FFT-based smoothing to a trajectory and write the smoothed trajectory.

```bash
imrmlp smooth_traj --input=trajectory.cfg --output=smoothed.cfg --timestep=0.1
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-i` | | Input trajectory file |
| `--output` | `-o` | | Output file |
| `--index` | `-id` | `index.map` | Index file |
| `--input-format` | `-if` | auto | Input file format |
| `--output-format` | `-of` | auto | Output file format |
| `--overwrite` | `-ov` | `false` | Overwrite existing output |
| `--timestep` | `-ts` | `0.1` | Trajectory timestep in fs |
| `--freq-cutoff` | `-fq` | `1` | Smoothing cutoff frequency in fs⁻¹ |
| `--wtype` | `-w` | `ComputeSmoothFFT::SUPERGAUSSIAN` | FFT window type |
| `--output-rate` | `-or` | `1` | Output every nth frame |
| `--interpolation-factor` | | `1` | Positive integer Fourier interpolation factor |
| `--scale-cell` | `-sc` | `false` | Rescale all frames to the average cell before smoothing |
| `--settings` | | `settings.ini` | Settings file |

Smoothing requires at least two frames. `--freq-cutoff` and both settings keys
(`freq-cutoff`, `freq_cutoff`) use fs⁻¹; for example, `0.005` corresponds to a period
of 200 fs. An explicit command-line cutoff takes precedence over either setting;
otherwise `freq-cutoff` takes precedence.

---

## Stationary Points and Saddle Searches

### `nroptimize` — Newton–Raphson Stationary-Point Refinement

Refine a stationary point with the analytic PotMTP Hessian. Use `--no-cell` for
fixed-cell refinement; exit code 2 means the iteration budget was exhausted.
[Usage and validation](../advanced/saddle-workflows.md).

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | `POSCAR` | Input structure (positional) |
| `--output` | `-o` | `CONTCAR` | Output structure (positional) |
| `--potfile` | `-pot` | `potential.mtp` | Potential file (positional) |
| `--index` | `-id` | `index.map` | Species index map |
| `--no-cell` | `-nocell` | `false` | Keep the cell fixed |
| `--pressure` | `-p` | `0` | External hydrostatic pressure (GPa) |
| `--external-stress` | `-str` | | External stress tensor in GPa |
| `--gtol` | `-g` | `0.01` | Gradient tolerance (eV/Å for fixed-cell atomic coordinates; mixed units with a variable cell) |
| `--max-iteration` | `-maxit` | `100` | Iteration limit |
| `--alpha` | `-a` | `0.1` | Initial step scale |
| `--alpha-reduce` | `-ar` | `0.9` | Step reduction factor during backtracking |
| `--max-backtracks` | `-mbt` | `10` | Maximum backtracking steps |
| `--fixed-atoms` | `-fix` | | Fixed atom indices, 0-based (multiple values) |
| `--eigzero-tol` | `-eigtol` | | Eigenvalue zero tolerance |

### `calc_dimer` — Dimer Saddle Search *(alias: `dimer`)*

Dimer saddle-point search. Explicit MODECAR input is strict; when MODECAR is
omitted the initial direction uses `dimer:random_seed`. All other controls live
in the `dimer:` settings namespace. [Parameters, output and limitations](../advanced/saddle-workflows.md).

```bash
imrmlp dimer dimer.ini
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--settings` | | | Dimer settings file (positional, required) |
| `--structure` | `-in` | | Initial-guess structure |
| `--input-format` | `-if` | auto | Input structure format |
| `--index` | `-id` | `index.map` | Species index map |
| `--output` | `-out` | `dimer_final.vasp` | Converged saddle-point structure |

Mode following also supports `--mode=9 --modecar-out=MODECAR`; mode 2 accepts
`--max-loop` and reports unsuccessful saddle searches with exit code 2.

---

## GPM Model Bundles

`model_convert` writes a self-contained GPM file; `model_info` and
`model_validate` inspect it without loading numerical blocks. See
[GPM format, save profiles and active learning restoration](gpm-v1.md).

### `model_convert` — Write a GPM Model Bundle

```bash
imrmlp model_convert --input old.mtp --type mtp --output model.gpm
imrmlp model_convert --settings with-al.ini --output resume.gpm --profile resume
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | | Legacy model or GPM input |
| `--output` | `-out` | | Output GPM file (required) |
| `--settings` | | | Existing potential and optional `al:` configuration |
| `--type` | | `auto` | Legacy potential type; GPM infers its type |
| `--index` | | | Explicit ordered element map |
| `--profile` | | `preserve` | `preserve`, `inference`, or `resume` |

### `model_info` — Inspect GPM Metadata

```bash
imrmlp model_info --input model.gpm
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | | GPM model (required) |

### `model_validate` — Validate a GPM Model

```bash
imrmlp model_validate --input resume.gpm
```

| Option | Alias | Default | Description |
|--------|-------|---------|-------------|
| `--input` | `-in` | | GPM model (required) |

---

## See Also

- @ref guide_workflows — end-to-end workflow examples
- @ref guide_config — configuration file format
- @ref guide_install — build and installation
- @ref reference_potential_types — potential type overview and file formats
- @ref apps — all command implementations
- @ref cheatsheet — quick-reference command summary
