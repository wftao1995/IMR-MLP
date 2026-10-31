# Native variable-projection training for ordinary MTP {#doc_docs_mtp_native_varpro}

[中文](mtp-native-varpro_zh.md)

Use `trainnlpot --do-varpro` to optimize radial parameters while solving the
linear coefficients at every trial. This can be useful when a final linear refit
alone leaves substantial error. It supports ordinary MTP on a single CPU process
or MPI CPU ranks. Species offsets and linear moment coefficients are solved by
QR/SVD; the radial gradient then drives L-BFGS.

The method reuses training weights, linear L2 and radial penalties. Input data
and neighbor lists are loaded once; the radial-dependent design matrix is
rebuilt at every trial. See [linear refinement](linear-refit.md) for a cheaper
single solve with all nonlinear parameters fixed.

## Run a fit

Prepare an ordinary `mtp` model, training data, optional validation data and an
`index.map` in the model's species order. Build from the repository root and
use separate input/output paths for each run:

```sh
make -j8 imrmlp
LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
build/release/imrmlp trainnlpot \
  --settings /absolute/fit.ini --do-varpro --output /absolute/exported.mtp --num_threads 2
```

Example `fit.ini` (replace `/absolute/...` with real paths):

```text
# ===== Input/Output =====
index               /absolute/index.map
trainset            /absolute/train.cfg
trainset-format     mlpcfg
validset            /absolute/valid.cfg
validset-format     mlpcfg

# ===== Potential =====
pot:pottype         mtp
pot:potfile         /absolute/initial.mtp
pot:typemap         /absolute/index.map

# ===== Runtime =====
fit:device          cpu

# ===== Initialization =====
fit:random_init     false
fit:force_random    false

# ===== Training Schedule =====
fit:training_schedule single

# ===== Parameter Scaling =====
fit:parameter_scaling none
fit:scaled_step_max 0

# ===== Loss Function =====
fit:energy_weight   1
fit:force_weight    0.05
fit:stress_weight   0.001
fit:energy_pow      1
fit:force_pow       0
fit:stress_pow      1
fit:weight_L2       1e-10
fit:weight_L1       0
fit:penalty_weight  1e-6

# ===== L-BFGS =====
fit:bfgs_m          10
fit:bfgs_maxiteration 40
fit:bfgs_grad_tol   1e-8
fit:bfgs_conv_tol   1e-12
fit:bfgs_step_tol   1e-12

# ===== Diagnostics =====
fit:varpro_report   varpro.jsonl
fit:validation_interval 5
fit:validation_metric force

# ===== Output =====
fit:save_to         fitted.mtp
fit:save_every      5
```

The weights and tolerances are starting points: run with them first, then adjust
against the reported validation RMSE. Do not enable
`linear_refit` separately: VarPro always solves the `sr` block internally.
`linear_refit_solver`, `linear_refit_rtol`, `linear_refit_atol` and
`linear_refit_tolerance` also control this inner solve.

The maximum absolute linear-gradient component must be at most
`fit:varpro_linear_gradient_tolerance` (default `1e-7`). If the first solution
misses this threshold, the solver refits the residual at the same radial point,
up to `fit:varpro_linear_refinement_steps` extra times (default 2; 0 disables).
It solves an increment to the current coefficients and evaluates regularization
at the updated coefficients. This reduces cancellation error without changing
the objective, rank cutoff or gradient threshold. Ill-conditioned or rank-truncated
stalls still need an outer restart or a different model. If checks still fail, training stops
and restores parameters; inconsistent objectives or nonfinite values fail
immediately. MPI uses the maximum linear-gradient norm across ranks.

`bfgs_maxiteration=0` performs only the initial linear projection and reporting.
The three stopping tolerances apply to radial-gradient norm, relative objective
change and step size, respectively. Reaching the iteration limit is not convergence.

## Saved models and validation

For `fit:save_to fitted.mtp`, output includes:

| File | Meaning |
| --- | --- |
| `fitted.mtp` | Latest accepted full coefficients; also saved after initial projection |
| `fitted.mtp.iter-000005.mtp` | Accepted-point snapshot at `save_every` intervals |
| `fitted.mtp.best-training.mtp` | Lowest complete training objective in this run, including penalties |
| `fitted.mtp.best-data.mtp` | Lowest weighted E/F/virial data loss, excluding radial and linear penalties |
| `fitted.mtp.best-validation.mtp` | Best checked validation metric satisfying the configured ceilings |
| CLI `--output` | Final accepted model, matching the `fit:save_to` endpoint |

The initial and final points are always validated; intermediate points follow
`validation_interval`. Best validation means best among **checked** points.
Validation data never enter the linear solve, gradient or line search. Without
validation data, no best-validation model is generated. Best-data is checked at
every accepted point, independently of snapshot and validation intervals.

`validation_metric` accepts `energy`, `force` or `stress`. Optional ceilings are:

| Setting | Unit | Default |
| --- | --- | --- |
| `fit:validation_max_energy` | eV/atom | -1, unrestricted |
| `fit:validation_max_force` | eV/Å | -1, unrestricted |
| `fit:validation_max_stress` | GPa | -1, unrestricted |

A metric or ceiling requiring missing labels is an error. If no checked point
meets the ceilings, no best-validation model is written and
`best_validation_iteration` is -1. Ceilings affect selection, not training loss.

Use a fresh report path and save prefix for each run to avoid mixing old reports
or best models. Restart from a checkpoint via `pot:potfile` and set
`fit:bfgs_iteration_offset` as appropriate. This restores coefficients and
rebuilds L-BFGS history; it need not reproduce uninterrupted training. Best-model
records restart from the new run's initial projection. Compare reports across
runs when choosing a model.

## Read the report

The JSONL report includes `settings`, `initial`, `accepted`, `final` and
`termination` records. Residual corrections also produce `linear_refinement`
records, including successful trials not accepted by L-BFGS. Failures record
`evaluation_failed` / `failed` when possible. Trial points do not update accepted
checkpoints or best models.

Accepted records contain:

- `loss.total/data/energy/force/virial/radial_penalty/linear_penalty`; total is
  the weighted data loss plus both penalties.
- Physical radial-gradient, optimizer-gradient and linear-gradient norms;
  inner rank, optimality residual, assembly/solve/verification times;
  `linear_refinements` and `linear_gradient_initial_inf`. Timings include all
  residual solves; rank and optimality use the final solve.
- Radial-channel coefficient norms across all element pairs and radial bases.
- Unweighted physical RMSE for energy per atom, Cartesian force components and
  the full stress tensor, in eV/atom, eV/Å and GPa. Missing labels are `null`.
- `best_training_iteration`, `best_data_iteration` and
  `best_validation_iteration`, using cumulative iteration numbers.

Stress RMSE is `sqrt(sum(||predicted_stress-reference_stress||_F²)/(9*N))`,
where each virial is first divided by its configuration volume and `N` counts
stress-labelled configurations. This agrees with `calc_err` and is invariant
under a common rotation. Report schema v3 records
`stress_metric="full_tensor_rmse_9"`. Older six-independent-component reports
and their stress ceilings need reevaluation; no single conversion factor applies.

`settings` records effective weights, normalization, regularization, tolerances
and validation rules. `termination.type` uses ALGLIB codes: 1/2/4 indicate a
stopping criterion was met, 5 is the iteration limit, and 0 is initial projection
only, with no L-BFGS step taken; judge usability from the validation RMSE and
`best_validation_iteration` in the JSONL report. `best-data` is weighted and need not have the lowest unweighted force RMSE.
Compare `loss.data` across regularization settings only with the same data
weights; `loss.total` with different regularization weights is not a common
ranking metric.

## Optional fixed radial-coordinate scaling

Set `fit:varpro_radial_scale_file /absolute/radial-scales.txt`. The file must
contain exactly `radial_coeff_size` finite positive numbers, whitespace
separated, in model radial-coefficient order (element pair, radial channel,
radial basis index). Without the file, every scale is 1. MPI root reads and
broadcasts the vector; missing, extra, nonpositive or nonfinite values fail.

The optimizer uses `r_i=s_i*z_i` and `g_z,i=s_i*g_r,i`. Objectives,
regularization and saved models still use physical coefficients `r`. Scales
stay fixed and are recorded in full. Gradient and step stopping conditions
apply in `z` coordinates: equal numerical tolerances under different scales do
not establish equal physical convergence. Compare `radial_gradient_inf` as well
as `optimizer_gradient_inf`. Specify the scale file again on coefficient
restart. Do not combine VarPro with `parameter_scaling`, `set_scale` or
`scaled_step_max` scaling modes.

## Supported scope

Supports ordinary `PotMTP`, `radial_mode=0`, CPU single-process/MPI, E/F/virial,
linear L2 and the fixed diagonal scaling above. LES/other derived potentials,
GPU, HVP/charge loss, linear L1, nonempty `trainable_groups` and other parameter
scaling modes are rejected. Structured reports and these best checkpoints are
specific to `--do-varpro`.

## MPI execution

```sh
make BUILD_MODE=mpi -j6 imrmlp
LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to none -n 4 \
  build/mpi/imrmlp-mpi trainnlpot \
  --settings fit.ini --do-varpro --output exported.mtp --num_threads 1
```

Use the MPI executable; a release executable remains single-process. All ranks
need the same inputs and settings, with relative paths resolved from each
process's working directory. Root owns L-BFGS, model selection and file output;
ranks share configuration work. Validation combines global squared errors and
component counts, including empty local shards correctly. The root QR/SVD solve
and matrix reassembly remain costs, so the speedup varies; compare `-np 1` with `-np N`
on the same settings. Current CLI ranks each read the input before distributing frames.

Numerical and publication failures propagate across ranks. Previously accepted
snapshots remain available. Regression coverage is in `unitest/ut_varpro.cpp`
and `unitest/ut_varpro_mpi.cpp`; to run the MPI test from the repository root:

```sh
make BUILD_MODE=mpi build/mpi/unitest/ut_varpro_mpi.test
LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
mpirun --bind-to none -n 4 build/mpi/unitest/ut_varpro_mpi.test
```

Repeat with 1, 2 and 4 ranks when validating a local MPI installation.
