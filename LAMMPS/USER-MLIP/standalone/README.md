# Independent MTP-M inference for LAMMPS

This source package builds `mlip/opt`, optional `mlip/opt/omp`, and optional
`mlip/gpudirect` without linking IMR-MLP. The runtime uses C++17 and the standard
library, with bundled Eigen headers used privately for PolyB initialization.
The LAMMPS adapters also require the matching LAMMPS headers and MPI interface.
CUDA/HIP builds require their respective compiler and runtime. No system Eigen
installation is required. CMake 3.22+ is required. MKL,
Python and the training library are not deployment dependencies. Python 3 is
used only by packaging and optional regression/benchmark scripts.

Supported MTP-M short-range heads are `mtp_m_li`, `mtp_m_q2`, `mtp_m_fs`,
`mtp_m_n1` and `mtp_m_n2`, with FP64 inference. Long-range LES/SOG files must be
rejected by this interface. Existing classic linear models keep their original
CPU execution path. Supported radial formats and runtime option validation are
controlled by the shared reader and executor; an unsupported model must fail
at load, never silently lose a contribution. For Zstandard-compressed GPM blocks,
configure `-DMTP_ZSTD=ON` and install the libzstd development package; uncompressed
GPM blocks do not need it.

## Export and build

From the repository root, export an immutable source snapshot to a new path:

```sh
python3 LAMMPS/USER-MLIP/standalone/package.py --output /tmp/mtp-source
cmake -S /tmp/mtp-source/standalone -B /tmp/mtp-cpu \
  -DMTP_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release \
  -DLAMMPS_SOURCE_DIR=/path/to/matching/lammps
cmake --build /tmp/mtp-cpu --parallel
```

The manifest explicitly lists each copied source, including the standalone
metadata parser and picojson with its embedded license. `source-record.json`
records the source revision and actual file hashes, including uncommitted edits.
The destination must not already exist. The same CMake entry point also works
inside this checkout. All build output belongs outside the source package.

For a CPU runtime without LAMMPS or MPI, add `-DMTP_BUILD_ADAPTERS=OFF`.
The CPU branch does not enable a GPU language or search for a GPU toolkit.
`-DMTP_OPENMP=ON` additionally builds `mlip/opt/omp`; the target LAMMPS executable
must have its OPENMP package enabled. Architecture options are explicit through
`MTP_ARCH_FLAGS`; the build does not enable fast-math.

GPU configurations use exactly one backend per build directory:

```sh
cmake -S /tmp/mtp-source/standalone -B /tmp/mtp-cuda \
  -DMTP_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DLAMMPS_SOURCE_DIR=/path/to/matching/lammps
cmake --build /tmp/mtp-cuda --parallel

cmake -S /tmp/mtp-source/standalone -B /tmp/mtp-hip \
  -DMTP_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DLAMMPS_SOURCE_DIR=/path/to/matching/lammps
cmake --build /tmp/mtp-hip --parallel
```

Select architectures and the CUDA host compiler to match the actual machine.
GPU builds retain the CPU styles, so the same binary can run either backend.

## Build the standard LAMMPS `lmp` executable

The `standalone/lammps` CMake entry point builds upstream LAMMPS's own library
and `main.cpp`, embedding the pair sources through its normal `PairStyle`
factory and generated `style_pair.h`. It produces an ordinary `lmp`, with no
`plugin load`, PLUGIN package, custom runner, or explicit registration call.
It works both in this checkout and in the exported source snapshot:

```sh
cmake -S /tmp/mtp-source/standalone/lammps -B /tmp/lmp-mtp-cpu \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON -DBUILD_OMP=OFF -DPKG_PLUGIN=OFF
cmake --build /tmp/lmp-mtp-cpu --parallel
/tmp/lmp-mtp-cpu/lmp -h
mpirun -np 4 /tmp/lmp-mtp-cpu/lmp -in in.lammps
```

Inside this repository replace the `-S` path with
`LAMMPS/USER-MLIP/standalone/lammps`. `LAMMPS_SOURCE_DIR` accepts the upstream
root or its `src` directory. Use a clean upstream source tree: old installed
copies of these styles are rejected. This entry point writes only to the build
directory, leaves upstream sources and `main.cpp` unchanged, and does not build
`mtp_standalone_plugin.so` or `mtp_lammps`. The standard LAMMPS options remain
available, including `PKG_*`, `BUILD_SHARED_LIBS`, `CMAKE_INSTALL_PREFIX` and
`LAMMPS_SIZES`; adapters inherit the target's MPI and integer-size definitions.
`BUILD_MPI=OFF` selects LAMMPS's serial MPI stubs without requiring an MPI
installation. Do not set `LAMMPS_LIBRARY` or `MTP_LAMMPS_DEFINITIONS` for this
entry point. Install normally with `cmake --install /tmp/lmp-mtp-cpu`.

Build each backend in a separate directory. All configurations include
`mlip/opt`; `MTP_OPENMP=ON` adds `mlip/opt/omp`, and CUDA/HIP add
`mlip/gpudirect`:

```sh
# CPU + OpenMP. MTP_OPENMP enables PKG_OPENMP unless explicitly conflicting.
cmake -S /tmp/mtp-source/standalone/lammps -B /tmp/lmp-mtp-omp \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=CPU -DMTP_OPENMP=ON -DPKG_OPENMP=ON -DBUILD_OMP=ON \
  -DCMAKE_BUILD_TYPE=Release

# NVIDIA CUDA; choose architecture and host compiler for the target machine.
cmake -S /tmp/mtp-source/standalone/lammps -B /tmp/lmp-mtp-cuda \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_TYPE=Release

# AMD HIP/DCU; choose the actual target architecture.
cmake -S /tmp/mtp-source/standalone/lammps -B /tmp/lmp-mtp-hip \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=HIP -DCMAKE_HIP_ARCHITECTURES=gfx90a \
  -DCMAKE_BUILD_TYPE=Release
```

Run `cmake --build BUILD --parallel` after each configure command; the executable
is `BUILD/lmp` (or the normal LAMMPS machine-suffixed name if configured).
OpenMP can also be enabled in CUDA/HIP builds. These GPU backends do not require
LAMMPS's GPU or KOKKOS packages. A CPU build never searches for a GPU toolkit.

Use the same input as below, omitting `plugin load`:

```lammps
units metal
atom_style atomic
newton on
read_data structure.data
# Set masses to match the LAMMPS atom types.
pair_style mlip/opt
pair_coeff * * model.mtp Al Ni
run 0
```

For OpenMP insert `package omp N` before `read_data` and select
`pair_style mlip/opt/omp`, where `N` is the number of threads per MPI rank (see
the [interface guide](../../README.md#21-openmp) for the command-line forms and
thread planning). For CUDA select
`pair_style mlip/gpudirect device cuda precision fp64 gpu_id auto`; HIP uses
`device hip`. All model restrictions and runtime options below apply unchanged.

With tests enabled (the default), build all targets and run:

```sh
ctest --test-dir /tmp/lmp-mtp-cpu --output-on-failure
python3 /tmp/mtp-source/standalone/tests/run_regression.py \
  --lmp /tmp/lmp-mtp-cpu/lmp --ranks 1 2 4 --work /tmp/lmp-regression
```

CTest includes five-head CPU comparisons for the embedded styles, and two-thread
OpenMP comparisons when enabled. It does not launch GPU work automatically;
on allocated devices run the regression with `--device cuda` or `--device hip`
and the corresponding `lmp`. Add `--lifecycle` for the lifecycle suite, with
`-DPKG_MOLECULE=ON` for its molecular rejection check. The independent core and
reader tests remain available. `MTP_BUILD_TESTS=OFF` disables these tests and
their optional Python discovery; Python is never needed to build or run `lmp`.

## LAMMPS plugin and explicit static registration

Load `mtp_standalone_plugin.so` into a LAMMPS executable built with PLUGIN.
Use the same LAMMPS version, MPI ABI, integer sizes and compatible compiler ABI
for the plugin and executable; `MTP_LAMMPS_DEFINITIONS` defaults to
`LAMMPS_SMALLBIG`. A plugin is not a version-independent binary distribution.

```lammps
plugin load /tmp/mtp-cpu/mtp_standalone_plugin.so
units metal
atom_style atomic
newton on
# Define box, atoms and masses before pair_coeff.
pair_style mlip/opt
pair_coeff * * model.mtp Al Ni
```

For OpenMP, select `mlip/opt/omp` and configure `package omp N`. For CUDA/HIP:

```lammps
pair_style mlip/gpudirect device cuda precision fp64 gpu_id auto
pair_coeff * * model.mtp Al Ni
```

For embedding in a custom application, link CMake target `MTP::lammps_adapters` and call
`register_mtp_standalone(lmp)` after creating a LAMMPS instance, before reading
input. This registers the same style names without PLUGIN or editing generated
LAMMPS style headers. Setting `LAMMPS_LIBRARY` builds the example `mtp_lammps`
runner. A static liblammps may require explicit `LAMMPS_EXTRA_LIBRARIES`, matching
its own link command (for example JPEG, PNG, zlib, OpenMP and platform libraries).
No mutation of the supplied LAMMPS source/build is performed.

For MTP-M, restart files preserve the pair settings, species mapping and model
text. `read_restart` restores the pair automatically and checks the external
model against the saved text; a changed model is rejected. Keep the model at its
saved path when restarting. Relative model paths are interpreted from the
working directory, so restart in the same directory or initially supply an
absolute model path. Device buffers and thread workspaces are rebuilt.

## Runtime options and supported inputs

Use space-separated option/value pairs in `pair_style`:

```lammps
pair_style mlip/opt/omp derivative_mode auto
pair_coeff * * model.mtp Al Ni

# Alternative GPU configuration, with a cap on tracked runtime allocations:
pair_style mlip/gpudirect device cuda precision fp64 gpu_id auto centers_per_chunk 1024 memory_budget_mib 4096
pair_coeff * * model.mtp Al Ni
```

| Option | Meaning |
| --- | --- |
| CPU `derivative_mode auto` | Default: cache the basic-moment Jacobian when its per-thread capacity is at most 256 KiB; otherwise recompute it during force accumulation. |
| CPU `derivative_mode cached` / `recompute` | Override the policy for measured workloads. Recompute releases an existing Jacobian buffer. |
| GPU `centers_per_chunk N` | Bound the number of owned centers in moment/head/reverse workspaces. Zero uses all centers unless a budget selects a smaller chunk. Coordinates, topology and force buffers still cover the full local/ghost set. |
| GPU `memory_budget_mib N` | Bound tracked model and workspace device allocations; zero disables the cap. Insufficient fixed storage or space for one center is an error. CUDA/HIP context and external-library allocations are outside this accounting. |
| GPU `force_kernel auto`, `basic-tile`, `center-neighbors` | Existing force-kernel choices for controlled comparisons; chunking works with either explicit kernel. |

Measure `cached` and `recompute` on a representative model before changing the
default. Jacobian storage is per thread, so total memory also depends on the
OpenMP thread count. GPU chunking bounds center workspaces, while coordinates,
neighbor topology and force buffers still require storage for the full local
system.

The five short-range heads accept the same validated model text on CPU and GPU.
NN1 supports TANH/RELU/SIGMOID/LINEAR; NN2 supports TANH/RELU/SIGMOID.
Radial families are RBChebyshev, RBChebyCos, RBChebyshevPoly4,
RBChebyshevPoly5, RBPolyB, RBPolyBCos and RBChebyshevConfigurable, including
all 120 combinations of its 12 cutoffs, five transforms and two polynomial forms,
and ordered species-pair blocks. A radial basis size of one is supported. Each block must have the same radial basis size. Unsupported families,
malformed dimensions, conflicting metadata and long-range coefficients fail at
load. MTP-M models support dynamic CPU type mappings beyond ten LAMMPS types.

The pair requires `units metal`, three dimensions, `newton on` and complete
many-body neighborhoods. Neighbor exclusions and nontrivial molecular
special-bonds scaling are rejected. Centroid stress and pairwise
`single()` are unavailable. GPU MTP-M inference uses FP64; mixed precision and
the gpudirect CPU fallback do not support these nonlinear models.

## Fixed PolyB deployment matrices

A model can carry a `standalone_radial_v1` JSON comment containing fixed PolyB
orthogonalization matrices and normalization arrays. Preserve this footer when
copying the model or moving restart files. It records ordered species pairs,
basis definitions and FP64 matrix values. The reader rejects malformed,
duplicate, incompatible or nonfinite cache data.

This checkout does **not** register the historical `mtp-export-standalone` CLI
command. Do not run it as an `imrmlp` subcommand. If an external compatible
exporter supplied the footer, retain the original model and record that exporter
and build. Re-export a newly trained or saved model rather than transplanting a
footer between models. Ordinary native Save operations do not preserve these
standalone comments.

A model without this footer remains loadable. Its PolyB matrices are initialized
at load time with the bundled Eigen version (see `vendor/eigen3/SOURCE.txt`),
without BLAS/LAPACK/MKL. This compatibility path does not promise agreement with
an independently initialized high-order native PolyB basis. Fixed matrices avoid
that solver difference and are copied to resident GPU storage at initialization.
No Eigen object or matrix decomposition enters the neighbor loop or device code.

Configurable and parameterized Poly4/5 bases are zero below their pair minimum.
Chebyshev, ChebyCos, PolyB and PolyBCos continue below that minimum. Every pair
uses its own upper cutoff, including heterogeneous and asymmetric split blocks.
Standalone inference exposes values and first derivatives for energy, force and
virial; this does not add Hessian or third-derivative interfaces.

## Correctness checks

```sh
ctest --test-dir /tmp/mtp-cpu --output-on-failure
python3 /tmp/mtp-source/standalone/tests/run_regression.py \
  --lmp /path/to/lmp --plugin /tmp/mtp-cpu/mtp_standalone_plugin.so \
  --ranks 1 2 4 --work /tmp/mtp-regression
```

The C++ core tests are standalone executables. `mtp_radial` checks all six
dedicated families, all configurable combinations, all six heads, single bases,
asymmetric split blocks, fixed matrices, and coordinate/strain finite differences.
The Python regression creates
five small analytic head fixtures with ordered species-pair radial coefficients,
duplicate scalar mapping and a repeated square rule. Its independent energy
implementation checks E/site-E, force finite differences and all six symmetric
strain derivatives at three step sizes. It exercises force-only, global energy,
global virial, site energy, site virial and combined requests. These synthetic
fixtures supplement, rather than replace, IMR-MLP oracle comparisons on real
trained potentials. JSON results and every input/log/dump remain in `--work`.

GPU runs add `--device cuda` or `--device hip`. OpenMP runs add
`--style mlip/opt/omp --threads 2`, or another positive thread count. Explicit GPU
selection must fail when no GPU
is available; a CPU fallback is not a passing GPU test. To test chunks, pass
`--pair-options 'centers_per_chunk 3'`. Only run MPI/GPU checks on resources
allocated by the caller.

Configure `-DMTP_BUILD_GPU_TESTS=ON` with CUDA/HIP and a separately installed
Catch2 3 package to build `test_standalone_gpu`. Its CTest sets
`IMRMLP_REQUIRE_GPU=1`, so missing devices fail instead of skipping. Catch2 is
optional test tooling and never a plugin/runtime dependency. Pure CPU builds do
not discover Catch2 or GPU toolkits.

Add `--lifecycle` for sorting, neighbor rebuild/reuse, atom type/count changes,
migration, empty MPI ranks, model reload, hybrid/overlay accumulation and
automatic restart, including changed-model rejection. Restart reads rotate
through `--ranks`, or use `--restart-ranks N`. Rejection checks cover neighbor
exclusions, centroid stress and molecular `special_bonds`; build LAMMPS with
MOLECULE for the latter. Add `--nve` for 10 ps NVE at 1 fs and 0.5 fs,
energy-drift checks and automatic restart continuity for each selected
head/rank count. The recorded `--duration` must remain in acceptance reports.
The fixture tolerance is not a universal production-model stability guarantee.
Use `--heads`, `--modes` and `--ranks` to select a small development check.

Pass `--oracle-plugin /path/to/native.so` and optional `--oracle-lmp /path/to/lmp`
to compare every requested E/F/W flag combination with `mlip/native` on the same
files. Native output is also checked against the independent formula. The oracle
requires IMR-MLP separately; it never enters the candidate runtime/plugin link.
The optional `tests/radial_lammps.py` oracle additionally requires an external
`--gpmain` executable providing `mtp-export-standalone`; this checkout's
`imrmlp` does not provide that command. With that dependency available, it
compares all six heads with an independent native plugin across the six dedicated radial families, a flipped
configurable basis, mixed ordered pairs, one-term bases and eight-term PolyB.
Supply `--gpmain`, `--native-lmp`, `--native-plugin`, `--cpu-lmp`, optional
`--omp-lmp`/`--gpu-lmp`, and a new `--work` directory. `--modes` selects the
energy/force/virial request combinations. Put both the exporter's build directory
and the native plugin's matching MPI IMR-MLP build directory before installed
libraries in `LD_LIBRARY_PATH`. These are oracle tools, not deployment dependencies.

Reader corpus checks use `test_reader.py --probe BUILD/mtp_model_probe
--compatibility-root /path/to/fixtures --report scan.json`; the report records
accepted/rejected models and diagnostics, while crashes/timeouts fail the scan.
These smoke checks cover their selected fixtures.

The benchmark scripts under `benchmarks/` are documented in
[benchmarks/README.md](benchmarks/README.md); they are maintenance tooling for
validating the runtime, not part of deployment.

## IMR-MLP public source layout

In this checkout the shared core is `runtime/mtp`, and metadata/GPM headers
are under `tools`. The exporter retains `mlip-core` inside the standalone
archive. CMake supports both layouts. GPM retains the `GPMLMODL` magic and `gpml.model` schema; these identifiers
belong to the file format.

The adapters follow the LAMMPS GPL-2.0 license; the independent runtime and
format helpers retain their MIT notices. The archive includes the GPL text,
MIT license, picojson notice and pinned Eigen notices. Preserve them when
redistributing the source package.
