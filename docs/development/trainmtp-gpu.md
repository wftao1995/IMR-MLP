# trainmtp GPU training

[中文](trainmtp-gpu_zh.md)

FP64 GPU training for ordinary `PotMTP`, `radial_mode=0`, is ported from
atoms-agent `34163139`, using observation kernels from `e843cb32`. CPU remains
the default; explicitly requesting an unavailable backend fails.

```sh
make BUILD_MODE=cuda -j4 imrmlp
OMP_NUM_THREADS=1 python3 examples/quickstart/run.py
cp examples/quickstart/gpu.ini build/quickstart/gpu.ini
cd build/quickstart
LD_LIBRARY_PATH=../cuda:$LD_LIBRARY_PATH ../cuda/imrmlp-cu trainmtp gpu.ini \
  --do-lbfgs --output _Trained_Potential_.mtp
```

The quickstart generates public synthetic labels and a starting model. Other
actions are `--do-cg`, `--do-sgd`, and `--linear-fit` (optionally `--do-rescale`).
`--calc-grad`, `--check-grad` and `--check-grad-all` share the nonlinear GPU
callbacks. `fit:save_to` names checkpoints; the explicit
`--output _Trained_Potential_.mtp` keeps a native final model (the default is the
GPM `_Trained_Potential_.gpm`). Synthetic labels exercise the interface rather
than establishing accuracy for a material.

| Setting | Default | Meaning |
| --- | --- | --- |
| `fit:device` | `cpu` | `cpu`, `cuda`, `hip`, or `dcu` (HIP alias) |
| `fit:gpu_id` | `-1` | Automatic node-local placement; explicit IDs allow intentional sharing |
| `fit:gpu_batch_atoms` / `fit:gpu_batch_structures` | `4096` / `32` | Positive whole-structure batch limits |
| `fit:gpu_memory_mib` / `fit:gpu_workspace_mib` | `256` / `64` | Linear non-observation/observation budgets; their sum bounds the nonlinear owner |
| `fit:gpu_curvature_host_mib` | `128` | Owned host reservation cap for CUDA linear curvature |
| `fit:gpu_device_loss` | `true` | Device residuals/loss; false uses host residuals with GPU predictions/gradients |
| `fit:gpu_deterministic_reduction` | `false` | Ordered diagnostic reduction, without cross-device bitwise guarantees |

MiB means 2²⁰ bytes. Budgets cover tracked allocations, excluding driver/context
storage; they are neither process RSS nor a whole-card memory cap. Nonlinear
allocation failures shrink whole-structure batches; an oversized single
structure fails. SGD evaluates its actual shuffled selection.

E/F/full nine-component raw virial weights, missing labels, fitness and global
structure averaging follow the dedicated CPU trainer. GPU work covers
observations, predictions, data loss and parameter gradients; optimizers,
regularization, linear solves and MPI reductions remain on CPU. Internal linear
refits, orthogonalization, rescale and frozen radial/linear/species parameters
are retained. Linear phases and definition/scaling changes invalidate the
nonlinear device owner.

CUDA `--linear-fit` additionally supports fixed-cell analytic Hessian/HVP
observations. Full Hessian fitting uses the upper triangle; HVPs retain their
saved directions. Independent CPU `H*v` evaluations verify the linear solve.
Nonlinear GPU training rejects curvature labels/losses; HIP linear curvature
is unsupported. Derived/LES/SOG/MTP-M models, ladder topology changes,
corrections and additive EFS are excluded from this GPU entry point; see
[trainnlpot GPU support](cuda.md) for broader model families. DF, random search,
species-only fitting, matrix exports and curvature diagnostics reject GPU
requests. The public branch's removed Shapeev BFGS is not reintroduced.

Use `BUILD_MODE=mpi-cuda` or `mpi-hip` for MPI. Complete structures belong to one
rank, including uneven and empty local shards. Mathematical/model/optimizer
settings must agree across ranks; placement and budgets may differ. Local
failures propagate before reductions. Nonlinear failures restore coefficients,
scaling, effective regularization and SGD state. Checkpoints/final models are
staged and reloaded before replacement. Rank 0 JSON reports describe work,
tracked peak memory and evaluation time, not complete process wall time or a
promise of speedup.

For HIP/DCU use `BUILD_MODE=hip HIP_ARCH=<device architecture>`, `fit:device hip`
and `build/hip/imrmlp-hip`. Local hardware validation of this port uses CUDA;
HIP requires validation on the corresponding hardware.
