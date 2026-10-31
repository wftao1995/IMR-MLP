# MTP GPU evaluation and training (CUDA/HIP) {#development_cuda}

[中文](cuda_zh.md)

The first part of this guide is for users running GPU evaluation or training;
the later sections describe implementation details. You need an available GPU,
a matching CUDA or HIP toolchain and the native dependencies in the
[build guide](../guides/installation.md). Runtime options require a GPU build.

## Build and choose a backend

From the repository root, choose the build matching your hardware:

```sh
make BUILD_MODE=cuda -j2 main  # NVIDIA CUDA: build/cuda/imrmlp-cu
make BUILD_MODE=hip -j2 main   # HIP/DCU: build/hip/imrmlp-hip
```

CUDA builds accept `NVCC` and `CUDA_ARCH`; HIP builds accept `HIPCC` and
`HIP_ARCH`. Values depend on the installed toolchain and device. MPI modes are
`mpi-cuda` and `mpi-hip`; both produce a main executable named `imrmlp-mpi`,
so select them by their full build paths.

`--device` accepts `cpu`, `cuda`, `hip` and `dcu`, with `dcu` an alias for `hip`.
Training uses the same values for `fit:device`. Each build contains one GPU
backend; requesting a different backend raises an error. Examples below use
CUDA. For HIP, use `build/hip/imrmlp-hip` with `--device=hip`, and set
`fit:device hip` for training.

## Current inference implementation

The shared core in `runtime/mtp/gpu-direct` supports CUDA and HIP. The batch
evaluation adapter serves two front ends:

- LAMMPS `pair_style mlip/gpudirect`, using full neighbor lists and LAMMPS
  ghost atoms;
- `imrmlp-cu calc_efs --device=cuda` / `imrmlp-hip calc_efs --device=hip` and the underlying IMRMLP
  `MTPGpuEvaluator`/`MTPGpuBatch` adapter, using contiguous CSR neighbor
  lists and explicit periodic displacement vectors.

For example:

```bash
build/cuda/imrmlp-cu calc_efs --input=POSCAR --output=efs.cfg \
  --potfile=potential.mtp --pottype=mtp \
  --device=cuda --gpu-id=0 --gpu-memory-fraction=0.75
```

The CLI preserves structure boundaries and packs the input into bounded GPU
chunks.  `--gpu-batch-atoms` and `--gpu-batch-structures` impose explicit
limits; zero selects automatic sizing from currently free memory and
`--gpu-memory-fraction`.  `--gpu-reuse-topology` reuses the static CSR arrays
between exactly compatible consecutive single-structure trajectory frames.
`--energy-only` selects a true device energy path and does not launch force or
stress work.  CPU remains the default.  `--gpu-id=-1` uses the current GPU
device; an explicit non-negative id selects a visible device.  A complete
Al-Ni example is available under `examples/mtp-gpu-alni`.

The forward path has three stages.  Basic moments are accumulated in tiles,
composite moments and their energy adjoints are propagated by dependency
level, and pair derivatives are recomputed and contracted directly into
forces.  At initialization the fast path groups product rules by forward
output and reverse destination within each dependency level.  One thread owns
each destination, sums its contributions in a register, and writes once, so
the production moment and adjoint propagation has no contended FP64 atomics.
The original schedule order remains available to the single-thread ordered
diagnostic path.  Pair-derivative recomputation avoids a global
`basic_moment x neighbor x xyz` Jacobian.  Center atoms and moment tiles run in
parallel GPU blocks. FP64 remains the default and reference mode. The
experimental `precision mixed` LAMMPS option changes classic-linear
`moment_vals` and the basic-moment partial workspace to FP32; energy adjoints
(`dmom`), all public interfaces, radial work, coefficients, reductions,
energies, forces, and virials remain FP64. This covers single Chebyshev,
ChebyCos, modern parameterized Poly5, split/type-pair, and configurable radial
representations. Unsupported nonlinear models and GPU fitting reject mixed
precision explicitly. The separate experimental
`precision mixed-aggressive` option also stores `dmom` in FP32, but preserves
FP64 dependency-group reductions and final outputs. Both mixed modes are
experimental and restricted to classic linear inference.
LAMMPS `deterministic yes` exposes the existing ordered diagnostic kernels;
same-device repeats are required to be byte-identical in that mode. The
default `deterministic no` path keeps parallel reductions and is checked with
tight numerical repeat tolerances instead of byte equality.

The implementation currently supports classic linear MTP with
`RBChebyshev`, `RBChebyCos`, `RBChebyshevPoly5`, and
`RBChebyshevConfigurable`. The IMRMLP adapter also supports split/type-pair
radial bases with one ordered basis per center-neighbor species pair. The
pairs may independently select their minimum distance, cutoff, scaling, and
basis type. Configurable descriptors explicitly carry quadratic/cosine/Poly5
cutoffs, linear/exponential coordinate transforms, Poly5 `cutoff_b`, and
exponential `shift_lambda`; all pairs retain the common radial basis size
required by `PotMTP`. The IMRMLP adapter additionally supports MTP-M with the
quadratic `mtp_m_q2`, ArcSinh `mtp_m_fs`, linear `mtp_m_li`, one-hidden-layer `mtp_m_n1`, and
two-hidden-layer `mtp_m_n2` external models for E/F/S and full parameter
gradients. The neural GPU path accepts arbitrary positive descriptor and
hidden dimensions, preserves the native flat coefficient layouts, and
implements tanh, ReLU, sigmoid, plus the NN1 linear activation. The standalone
LAMMPS potential reader now accepts the same single and ordered-pair radial
representations, including configurable cutoff/transform parameters, and the
self-describing q2/fs/NN1/NN2 layouts. The IMR-MLP batch adapter additionally
supports FP64 SOG E/F/S and full parameter gradients for linear MTP and these
MTP-M heads, including multiple channels and frozen parameter groups. SOG
requires complete periodic structures; reciprocal workspaces are included in
inference and training chunk memory budgets. This batch adapter does not support Coulomb LES. Hessian/HVP training objectives
and charge losses also require the CPU. LAMMPS `mlip/gpudirect` accepts only
short-range models. The separate [`mlip/native` interface](../../LAMMPS/USER-MLIP/native/README.md)
has its own distributed GPU long-range evaluation path for supported LES/SOG
models; its capabilities differ from this batch adapter.

Immutable batch geometry, labels and SOG grids are cached with identity and
revision tokens. Refreshing geometry, changing labels, switching evaluators
or failing a forward call invalidates the affected state. Training validation
uses the selected backend and its own batches. `fit:validation_interval`
controls validation at accepted L-BFGS iterations (default 1); initial and
final points are always checked when a validation set is present.

### Runtime contracts

- All threads execute the same neighbor-tile barrier sequence, including
  partial final tiles.
- LAMMPS special-bond bits are masked before neighbor indices are used.
- GPU dimensions, power-of-two reductions, grid limits, dynamic shared
  memory, integer indexing, atom types, centers, and neighbor bounds are
  validated before launch.
- Force output is computed as a device delta and accumulated on the host, so
  `pair_style hybrid` contributions are preserved without uploading the
  existing force array.
- Per-atom virial is split equally between the center and neighbor, matching
  the CPU MLIP core.  Contributions for local and ghost atoms are returned.
- Static center/offset/neighbor arrays can be reused.  Row metadata is still
  checked on each call, while periodic displacements are refreshed.
- The public batch request can disable force/virial construction, so
  energy-only callers skip directional derivative kernels and their outputs.
- `gpu_id auto` uses node-local MPI rank assignment and enforces the selected
  one-rank-per-GPU policy.
- The same GPU core is available through a public contiguous batch request;
  coefficient arrays can be updated without rebuilding the moment schedule.

## GPU MTP training

Classic linear MTP now has complete FP64 GPU gradients for energy, force,
and stress losses.  Enable it in `trainnlpot` with:

```ini
fit:device cuda
fit:gpu_id -1
```

`gpu_id=-1` selects `local MPI rank % visible device count`; an explicit
non-negative id overrides this.  CPU is still the default.  The host
L-BFGS/CG/SGD implementations, coefficient/checkpoint layout, regularization,
trainable masks, and MPI rank reduction are unchanged.

Each rank deterministically partitions its selected structures into
structure-preserving CSR chunks.  Automatic chunk sizing uses the visible
free memory, atoms, directed neighbors, moment dimensions, labels, gradient
partials, and a safety factor.  Explicit atom and structure limits are also
available.  A structure is never split; an individual structure that exceeds
the budget fails with its index and memory estimate.  SGD constructs chunks
only from the selected mini-batch instead of evaluating zero-weight
structures.

By default E/F/S labels and weights are uploaded to the GPU and residuals,
loss adjoints, seven loss/error metrics, and gradients are constructed there.
Label upload uses a transfer stream and event so it can overlap the resident
forward kernels.  A second event protects resident label/center buffers from
being overwritten or reallocated until all reverse consumers have completed.
Gradient and metric accumulators remain on the device across chunks; only the
final metric vector and native-order gradient
(`radial_coeffs`, `species_coeffs`, `moment_coeffs`) are copied to the host.
`fit:gpu_device_loss false` keeps the host-residual path as a diagnostic
fallback.

For fixed full-dataset optimizer callbacks, `fit:gpu_cache_batches true`
caches packed host CSR batches and labels after the first callback.  Later
callbacks reuse them and update only coefficients.  Random or partial SGD
selections bypass this cache.  Device allocations retain capacity across
compatible chunks, and two input slots allow construction/upload of the next
chunk without invalidating the current slot.  Caching reduces repeated host
construction at the cost of host RAM proportional to the packed dataset.

The force/stress adjoints define one directional operator for every directed
neighbor displacement.  Four moment arrays are sufficient:

- `m`: forward moments;
- `q = Dm`: directional moments;
- `a = dU/dm`: energy adjoints;
- `h = Da`: directional adjoints.

For a product rule `m_k += c m_i m_j`, `q` is propagated forward and `h`
backward through the same dependency levels.  Linear gradients are
`lambda_E*m + q`; radial gradients combine `(lambda_E*a+h)*db/dc` with the
pair-local mixed term `a*D(db/dc)`.  This avoids the full
coordinate-by-parameter Jacobian.

### Kernel sequence

1. Reuse inference basic/composite moments and energy adjoints.
2. Build the directional derivative of each basic moment from the force and
   stress loss weights, then propagate it through the composite-moment DAG.
3. Construct E/F/S residuals and adjoints on device and accumulate loss/error
   metrics.
4. Reduce linear gradients in parallel.  In the production radial reverse
   pass, each thread owns one species-local coefficient, visits only basic
   moments grouped under that radial function, and accumulates in a register.
   It writes a compact per-center row that a second kernel maps through the
   center species and reduces.  This avoids both global atom-by-parameter and
   shared FP64 atomics as well as the full four-dimensional Jacobian.
5. Retain metrics and the gradient across all chunks, copy one final result
   per rank, and use the trainer's existing MPI
   reduce/broadcast sequence before the host optimizer step.
6. Call `MTPGpuDirect::update_coefficients()` for the next iteration; do not
   rebuild topology or the static moment schedule.

The ordered diagnostic path retains the original single-thread radial
accumulation so its floating-point order remains stable.  The fast path uses
parallel reductions.  Setting
`fit:gpu_deterministic_reduction true` orders forward moment propagation,
force/virial accumulation, device loss metrics, directional propagation,
radial partials, and final parameter reductions.  It is intended for
byte-repeatability diagnostics and is substantially slower than the default.

### Training settings

```ini
fit:device cuda
fit:gpu_id -1
fit:gpu_batch_atoms 0
fit:gpu_batch_structures 0
fit:gpu_memory_fraction 0.75
fit:gpu_device_loss true
fit:gpu_deterministic_reduction false
fit:gpu_radial_parameter_tile 128
fit:gpu_cache_batches true
```

Zero batch limits mean automatic sizing.  A positive limit is applied in
addition to the memory budget.  The radial tile setting is an upper bound on
the species-local coefficient threads in a partial-row block; the launch is
clipped to the model and device limits and rounded toward a warp-sized block.

### Validation and current limits

The GPU unit test checks combined energy/force/stress gradients for every
coefficient against central differences.  It also covers device loss,
host/device loss equivalence, retained two-chunk gradients, ordered reduction,
coefficient updates, and inference derivatives.  Runtime tests skip cleanly
when no matching GPU device is visible.
