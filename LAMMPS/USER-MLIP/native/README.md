# Native MTP-family LAMMPS interface

`mlip/native` loads the original IMR-MLP model, including companion neural/HMTP
files and native manifests. It is built against the matching IMR-MLP library;
the standalone `mlip/opt` and `mlip/gpudirect` implementations remain available.

To embed native pair and compute styles directly in upstream's standard `lmp`,
without PLUGIN or runtime registration, use the
[standard executable entry point](lammps/README.md) ([中文](lammps/README_zh.md)).
It supports CPU, CUDA and HIP by linking the matching MPI IMR-MLP build.

## Build

Build IMR-MLP in an MPI mode first, then build the plugin against the exact
LAMMPS source headers used by the target executable:

```sh
make BUILD_MODE=mpi -j2 libimrmlp.so
cmake -S LAMMPS/USER-MLIP/native -B build/lammps-native-plugin \
  -DLAMMPS_HEADER_DIR=/path/to/lammps/src \
  -DIMRMLP_LIBRARY="$PWD/build/mpi/libimrmlp-mpi.so" \
  -DIMRMLP_FLAGS_FILE="$PWD/build/mpi/.flags" \
  -DLAMMPS_EXECUTABLE=/path/to/lmp
cmake --build build/lammps-native-plugin -j2
```

The LAMMPS executable needs its `PLUGIN` package and the same MPI
implementation. `IMRMLP_FLAGS_FILE` supplies the actual Eigen/MKL/backend ABI
flags, including CUDA/HIP definitions; selecting an unrelated flag file is
unsupported. Serial IMR-MLP libraries using MPI stubs are rejected. Put the
matching IMR-MLP build directory first in `LD_LIBRARY_PATH`.

## Input

```lammps
plugin load /path/to/mlip_native_plugin.so
units metal
atom_style atomic
boundary p p p
newton on
read_data system.data

pair_style mlip/native device cpu precision fp64 pottype auto
pair_coeff * * model.gpm Al Ni
```

This example assumes two LAMMPS atom types, Al then Ni, and a GPM model whose
element metadata contains both elements. Replace the model and mapping with
your own. GPM accepts omitted `pottype` or `pottype auto`. Legacy files require
the explicit type, for example `pottype mtp` or `pottype mtp_m`; use
`settings model.ini` when additional native `pot:` settings are needed.
`pair_coeff` takes one element name per LAMMPS atom type; legacy models without
element metadata use zero-based numeric species indices. Names and numbers
cannot be mixed, and `NULL` is rejected. The model and settings references are
made absolute when loaded. Model-relative companion references are resolved
by the native loader. To change models, delete dependent observable computes,
issue `pair_style none`, then define the new pair, coefficients and computes.

`device` accepts `cpu`, `cuda`, `hip`, or `dcu` (an alias for `hip`). The selected
GPU backend must match the IMR-MLP build; there is no CPU fallback. New native
inference uses FP64. For GPU builds, `gpu_id auto` places one rank on each
distinct node-local device and rejects accidental oversubscription. Explicit
`gpu_id N` permits intentional sharing for diagnostics; sharing does not
establish multi-GPU performance.

The initialization log states the actual model, device, and execution path.
All CPU models use owned-center domain decomposition. Long-range models
reduce structure factors and charge constraints collectively, and exchange
nonperiodic source blocks in a ring. No complete configuration is gathered.
The native GPU path covers supported MTP/MTP-M heads, legacy neural/HMTP models,
MTP-based composites and LES/SOG. It is separate from the short-range
`mlip/gpudirect` style and the CLI batch adapter. Supported families still require
a compatible saved layout and configuration. A requested backend/model combination
without an implemented evaluator fails explicitly.

The pair retains host workspaces and sorted candidates from LAMMPS's full
neighbor list. It refreshes positions and filters the physical cutoff on every
evaluation, including steps that reuse the neighbor list. Sorting, migration,
species changes, and cell changes invalidate the relevant cached data. GPU
inference receives a packed CSR view directly; an unchanged topology can reuse
device neighbor indices and species while positions are always uploaded again. Same-tag
periodic-image groups are checked against the original coordinate ordering on
each call, including rounding changes at exact nearest-neighbor ties.

## Outputs and restart

The pair supplies ordinary global and per-atom energy/virial flags, including
`compute pe/atom` and `compute stress/atom NULL pair`. Neighbor-edge virials
are symmetric and split equally between endpoints. Ghost force and stress
contributions use LAMMPS's normal reverse communication. Long-range virials
are explicit cell derivatives, including charge/latent response terms.

Returned energy, force, global virial and site outputs honor their independent
request flags. Pure short-range MTP, MTP-M and legacy NN heads avoid unrequested
energy/virial output buffers, reductions and transfers. Global GPU virials are reduced on the
device and returned as six doubles, without downloading site virials when those
are not requested. Virial evaluation still requires device derivative
workspaces, but a virial-only request does not download forces.
Composite and long-range models retain site and response buffers needed by
their dependency chains. CPU short-range outputs are allocated on demand;
the distributed CPU LES/SOG path shares descriptor evaluation and combines the
channel responses before backpropagation. These changes preserve FP64 model
mathematics and the existing per-atom decomposition.

Choose an observable only when the loaded model provides it:

| Loaded model configuration | Available compute |
|---|---|
| Short-range MTP/MTP-M/neural model | Neither charge nor latent |
| Active Coulomb LES | `compute q all mlip/native/charge` |
| Active SOG | `compute latent all mlip/native/latent` |

Do not add both computes to a generic input. Unsupported observables fail when
the compute is defined. Charge is a physical response observable; SOG instead
exposes dimensionless latent variables. A
single latent channel is `c_latent`; multiple channels are
`c_latent[1]`, `c_latent[2]`, etc. Define the pair and coefficients before the
compute so the channel shape is known, and recreate dependent computes after
replacing the pair.

Restart stores the model/settings references, species mapping, device, and GPU
placement policy. The referenced files must remain accessible.
Load the plugin before `read_restart`; the number of MPI ranks may change.

## Regression

```sh
LD_LIBRARY_PATH="$PWD/build/mpi:$LD_LIBRARY_PATH" OMP_NUM_THREADS=1 \
  ctest --test-dir build/lammps-native-plugin --output-on-failure
```

`IMRMLP_NATIVE_TEST_DEVICE` selects the CTest backend (`cpu` by default, `cuda`,
or `hip`). Family, NVE, and workspace tests receive this selection. CUDA/HIP CTest
runs explicitly set `IMRMLP_REQUIRE_GPU=1`; unavailable devices fail validation.
Each test has a default timeout of 7,200 seconds for the full device matrix.
Set `-DIMRMLP_NATIVE_TEST_TIMEOUT=SECONDS` to use another positive integer limit.
For example, configure against the matching MPI CUDA library and expose four
GPUs for the 1/2/4-rank family test and the four-rank NVE test:

```sh
cmake -S LAMMPS/USER-MLIP/native -B build/lammps-native-cuda-plugin \
  -DLAMMPS_HEADER_DIR=/path/to/lammps/src \
  -DIMRMLP_LIBRARY="$PWD/build/mpi-cuda/libimrmlp-mpi.so" \
  -DIMRMLP_FLAGS_FILE="$PWD/build/mpi-cuda/.flags" \
  -DLAMMPS_EXECUTABLE=/path/to/lmp \
  -DIMRMLP_NATIVE_TEST_DEVICE=cuda
cmake --build build/lammps-native-cuda-plugin -j2
ctest --test-dir build/lammps-native-cuda-plugin --output-on-failure
```

For HIP/DCU, use `-DIMRMLP_NATIVE_TEST_DEVICE=hip` and the MPI HIP library and
flags. The device names accepted by this CTest option are exactly
`cpu`, `cuda`, and `hip`; an invalid value fails CMake configuration.

The test generator creates deterministic family models, compares the native
inference results with the original IMR-MLP `getEFS`, and writes energy, forces,
global/site virial, site energy and applicable observables. LAMMPS comparisons
use 1/2/4 ranks, orthogonal/triclinic cells, empty owned ranks, cross-rank
restart, migration, cell
changes, channel-count reloads and collective input failures. No private
datasets are needed.

Run `tests/run_regression.py --help` for model filters, CUDA/HIP testing and an
output directory retaining inputs/results. GPU runs set `IMRMLP_REQUIRE_GPU=1`;
all selected models must pass, without unsupported-model skips. The report
records the requested GPU placement policy; an explicit shared-device run
does not establish coverage on the corresponding number of physical GPUs.

`--threads 1`, `--threads 2`, and `--threads 4` select CPU thread counts. The
result JSON includes the actual owned/ghost atom counts and selected GPU ID
for each rank. The domain split places owned atoms on multiple ranks and also
includes an empty rank.

For four nonempty ranks, set `IMRMLP_NATIVE_FOUR_RANK=1` and pass
`--ranks 4 --geometries four_rank`. This adds five long-range scenarios with
one owned atom on every rank. GPU validation requires actual model evaluations,
force kernel launches and model device allocations on each nonempty rank.

The generated models exercise all five MTP-M heads with both LES and SOG,
one- and two-layer legacy networks with every activation, nontrivial input
normalization and species offsets, and fixed/additive/scaled MTP/ZBL models.
Each model passes a parameter update, Save and reload before reference
evaluation. SOG includes three-channel output and positive/negative weights.

`tests/run_dynamics.py` uses an analytically constructed MTP dimer potential,
`V(r) = 0.1 (r - 2.5)^2 (r - 5)^2`, to check a stable 10 ps trajectory,
1 fs and 0.5 fs timesteps, and restart continuation. It accepts the same
`--lmp`, `--plugin`, `--generator`, `--device`, and `--work` arguments, with a
single `--ranks` value. GPU runs also perform the CPU baseline and enforce the
requested drift-increment bound and final position/velocity agreement. This manufactured system establishes
numerical integration behavior; it is not a material validation dataset.

`tests/run_workspace_regression.py` compares delayed neighbor rebuilding with
a CPU reference that rebuilds every step under prescribed motion. It checks
both directions across the physical and hybrid feature cutoffs without a skin
rebuild, type changes, sorting and migration, continuous triclinic deformation,
repeated periodic images with nearest-neighbor ties, and force-only/full-output
request changes. It accepts the same plugin/model generator arguments and
`--ranks 1 2 4`; `--models-dir` can reuse a previously generated fixture set.
The corresponding CTest is `native_lammps_workspace_mpi`.
