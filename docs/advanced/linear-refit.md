# Linear refinement after potential training {#doc_docs_mtp_linear_refit}

[中文](linear-refit_zh.md)

Use linear refinement to improve a trained model while keeping its nonlinear
parameters fixed. `trainnlpot` solves a weighted least-squares problem for the
selected coefficients, checks the complete objective, and saves the result.
This feature is disabled by default. It reduces the weighted training objective;
check validation errors separately before choosing the refined model.

## Refine a saved model

Save the following as `refit.ini`. Replace the absolute paths and create the
output directory before running. Data units and the species map must match the
original training; the example weights are illustrative.

```text
index /absolute/path/index.map
trainset /absolute/path/train.xyz
trainset-format extxyz
pot:pottype mtp
pot:potfile /absolute/path/input.mtp
pot:typemap /absolute/path/index.map
fit:device cpu
fit:random_init false
fit:force_random false
fit:training_schedule single
fit:energy_weight 1
fit:force_weight 0.05
fit:stress_weight 0.001
fit:energy_pow 1
fit:force_pow 0
fit:stress_pow 1
fit:penalty_weight 1e-8
fit:linear_refit sr
fit:linear_refit_report /absolute/path/output/refit.json
fit:save_to /absolute/path/output/fitted.mtp
```

```sh
imrmlp trainnlpot --settings refit.ini --linear-refit-only \
  --output /absolute/path/output/fitted.mtp
```

The input must contain initialized coefficients. Refinement-only mode requires
`training_schedule single` and rejects random initialization and input
perturbations. Do not combine `--linear-refit-only` with an optimization flag.

## Choose coefficients to refit

An affine block is a set of coefficients for which predictions are linear once
the other parameters are fixed. Available blocks depend on the model and its
enabled request flags; an unsupported or empty selection is an error.

| Model | `fit:linear_refit` | What changes |
| --- | --- | --- |
| Ordinary `mtp` | `sr` | Species energy offsets and short-range moment coefficients |
| `mtp_les`, Coulomb LES v2 | `sr` | Short-range coefficients; charges and Born responses stay fixed |
| `mtp_les`, SOG v3/v4 | `sr` or `sr_sog` | Short range only, or short range plus Gaussian weights in all channels |
| `mtp_m` (`li`, `q2`, `fs`, `n1`, `n2`) | `head` / `output` | The supported affine output coefficients; hidden features remain fixed |
| `mtp_m` with SOG | `output,sog` | Output coefficients plus Gaussian weights, when their request flags are enabled |
| `mtpnn1` | `head` / `output` | Final network weights and species offsets |
| `descriptor_linear` | `head` / `linear` | Linear descriptor coefficients |
| `hybrid_scale` | e.g. `pot1/sr` | Explicit affine blocks of the selected child potential |

`all` selects all blocks advertised by the model. A comma-separated list selects
exact block names. `head` selects `output`, or `linear` when that is the model's
head name. These choices do not make arbitrary nonlinear parameters linear.

For SOG, radial parameters, latent heads and bandwidths remain fixed. Gaussian
weight fitting includes latent-response forces and virials, signed weights,
zero/self modes and the existing weight regularization. Keep the original
regularization settings: the procedure preserves effective selected L2 terms,
and rejects nonzero selected L1. It does not enable adaptive ridge automatically.

## Run refinement after nonlinear training

Add `fit:linear_refit sr` (or another supported selection) to the training
configuration and retain the usual `--do-lbfgs` flag. Refinement runs after the
selected nonlinear stage. CPU CG with `training_schedule single` performs one
terminal refinement after its final accepted point. The selected coefficients
must also be trainable in the CG mask. GPU refinement and staged CG refinement
are unsupported.

A terminal refit does not change the CG stop reason or make an iteration-limited
CG run converged. To optimize radial parameters while solving the linear problem
at **every trial**, use [variable projection](mtp-native-varpro.md).

MPI uses the same configuration, for example:

```sh
LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH \
  mpirun -np 4 build/mpi/imrmlp-mpi trainnlpot --settings refit.ini \
  --linear-refit-only --output /absolute/path/output/fitted.mtp
```

Run from the repository root after building `BUILD_MODE=mpi`. Each rank needs
access to the same inputs. Parallel results can differ within floating-point
tolerance.

## Labels, checks and output

Refinement reuses energy/force/virial weights, atom-count powers, fitness and
reference-force normalization. It supports CPU HVP and full-Hessian labels using
the same central force differences as nonlinear training. A total-charge term
can remain enabled only when the model supports it and the selected coefficients
do not affect total charge; such a term stays constant during refinement.
SOG itself does not support physical total-charge labels.

All nine internal virial components enter fitting. `calc_err` and VarPro report
stress-component RMSE from the full tensor squared error divided by nine times
the number of labelled configurations; GPa metrics first divide each virial by
its configuration volume. Force-vector and force-component RMSE are distinct.

Before solving, the input is saved as `<fit:save_to>.pre-refit`. Acceptance
requires unchanged frozen parameters, a non-increasing complete objective,
agreement between matrix and full-prediction objectives, and a successful model
save/reload check. Failure restores the coefficients and returns an error.
Successful output goes to `fit:save_to` and the CLI `--output` destination;
models with native sidecar files retain that layout. CG also retains its
`.cg.snapshot` checkpoints. Checkpoints are coefficient restarts, not optimizer
history.

| Setting | Default | Meaning |
| --- | --- | --- |
| `fit:linear_refit_solver` | `qr_svd` | Supported solver |
| `fit:linear_refit_rtol` | `1e-12` | Relative numerical-rank threshold |
| `fit:linear_refit_atol` | `0` | Absolute numerical-rank threshold |
| `fit:linear_refit_tolerance` | `1e-10` | Objective verification tolerance, scaled by `max(1, abs(objective_before))` |
| `fit:linear_refit_report` | `linear-refit.json` | Objective, rank, parameter increments, regularization and timing report |

Fixed-composition data cannot determine every species offset independently.
Refinement preserves the undetermined combinations and solves for a minimum
increment in the remaining scaled coordinates. Compare predictions and the
objective, rather than requiring identical coefficients from rank-deficient
fits. Independent validation/test sets are used for reporting, not solving.
