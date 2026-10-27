# Native inference in a standard LAMMPS executable

[中文](README_zh.md)

This entry point builds upstream LAMMPS's own `lmp` and `main.cpp`, registering
`mlip/native`, `mlip/native/charge`, and `mlip/native/latent` through the normal
style factory. No `plugin load`, PLUGIN package, or custom runner is needed.
The upstream source tree is not modified. The [plugin entry point](../README.md)
remains available.

## CPU build

Run from the IMR-MLP repository root, using a clean upstream LAMMPS tree,
C++23, and the same MPI implementation and ABI as IMR-MLP:

```sh
make BUILD_MODE=mpi -j2 libimrmlp.so
cmake -S LAMMPS/USER-MLIP/native/lammps -B build/lmp-native-cpu \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps \
  -DIMRMLP_NATIVE_BACKEND=CPU -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON -DPKG_PLUGIN=OFF
cmake --build build/lmp-native-cpu -j2
LD_LIBRARY_PATH="$PWD/build/mpi:$LD_LIBRARY_PATH" \
  mpiexec -n 4 build/lmp-native-cpu/lmp -in in.lammps
```

`LAMMPS_SOURCE_DIR` accepts the upstream root or `src` directory. Standard LAMMPS
CMake options, including `PKG_*`, `BUILD_SHARED_LIBS`, `LAMMPS_SIZES`, and
`CMAKE_INSTALL_PREFIX`, remain available. `BUILD_MPI=OFF` is rejected: native
requires MPI IMR-MLP even for one rank. Duplicate installed native styles are
rejected. Install normally with `cmake --install build/lmp-native-cpu`; IMR-MLP
and its dependencies must remain accessible.

## CUDA and HIP builds

Build IMR-MLP's device kernels first. Replace toolkit paths and GPU architectures
below with those of the target machine. Use a separate build directory per backend:

```sh
make BUILD_MODE=mpi-cuda CUDA_HOME=/usr/local/cuda CUDA_ARCH=sm_90 -j2 libimrmlp.so
cmake -S LAMMPS/USER-MLIP/native/lammps -B build/lmp-native-cuda \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps \
  -DIMRMLP_NATIVE_BACKEND=CUDA -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON -DPKG_PLUGIN=OFF
cmake --build build/lmp-native-cuda -j2

make BUILD_MODE=mpi-hip HIP_ROOT=/opt/rocm HIP_ARCH=gfx90a -j2 libimrmlp.so
cmake -S LAMMPS/USER-MLIP/native/lammps -B build/lmp-native-hip \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps \
  -DIMRMLP_NATIVE_BACKEND=HIP -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_MPI=ON -DPKG_PLUGIN=OFF
cmake --build build/lmp-native-hip -j2
```

Defaults select `build/mpi/libimrmlp-mpi.so`, `build/mpi-cuda/libimrmlp-mpi.so`, or
`build/mpi-hip/libimrmlp-mpi.so`, and the corresponding `.flags`. Override
`IMRMLP_LIBRARY` and `IMRMLP_FLAGS_FILE` together for external builds. Backend/flag
conflicts and serial IMR-MLP builds fail at configuration time. IMR-MLP ABI flags
apply only to native source files, not upstream LAMMPS sources.

Device kernels are already compiled in IMR-MLP. This entry point compiles C++
only, without enabling CUDA/HIP languages or requiring LAMMPS GPU/KOKKOS
packages. The selected GPU runtime and IMR-MLP dependencies are still required
when running `lmp`. GPU builds can also execute on CPU. Model/backend support
boundaries and explicit rejection of unsupported requests match the plugin.

## Input and formats

```lammps
units metal
atom_style atomic
newton on
read_data structure.data
# Set masses for all atom types.
pair_style mlip/native device cpu precision fp64 pottype auto
pair_coeff * * model.gpm Al Ni
run 0
```

For GPU execution replace `device cpu` with `device cuda gpu_id auto` or
`device hip gpu_id auto`. Auto mode requires one distinct device per local
MPI rank; explicit `gpu_id N` permits intentional sharing for diagnostics.

The example assumes two LAMMPS atom types, Al then Ni, and corresponding element
metadata in the model. Replace the model and mapping with your own. Supply one
known element symbol per LAMMPS atom type. For legacy files without element
metadata, use zero-based model species indices such as `0 1`. Symbols and numeric
indices cannot be mixed, and `NULL` is unsupported.

GPM infers the model type and embeds model resources. Legacy inputs require an
explicit `pottype`, such as `mtp`, and optional `settings` for additional model
configuration. Ordinary short-range models have no charge/latent observable.
Only Coulomb LES accepts `compute q all mlip/native/charge`; only
active SOG models accept `compute latent all mlip/native/latent`. Define the
applicable compute after `pair_style` and `pair_coeff`, and recreate it after
replacing the model. See the [plugin guide](../README.md) for observable conditions
and model/GPU support. Restart semantics also match the plugin. Keep
referenced model/settings files available for restart. Loading GPM does not
activate AL. See the [GPM specification](../../../../docs/reference/gpm-v1.md).

## Verification

```sh
LD_LIBRARY_PATH="$PWD/build/mpi:$LD_LIBRARY_PATH" \
  ctest --test-dir build/lmp-native-cpu --output-on-failure

python LAMMPS/USER-MLIP/native/tests/run_regression.py \
  --lmp build/lmp-native-cpu/lmp \
  --generator build/lmp-native-cpu/generate_native_test_models \
  --gpm --ranks 1 2 4 --work build/native-gpm-regression
```

CTest covers geometry, legacy/GPM family comparisons, restart, charge/latent
outputs, NVE, workspace reuse, and configuration guards. The guards use
synthetic CUDA/HIP flag files to test backend selection without a GPU; these
are configuration checks, not device compilation or execution. Runtime scripts
omit `--plugin` for embedded styles; supplying it retains plugin testing.

Tests default to CPU, including with GPU libraries. On allocated devices set
`IMRMLP_NATIVE_TEST_DEVICE=cuda` or `hip` and put the matching IMR-MLP library
directory in `LD_LIBRARY_PATH`. Full MPI suites need up to four devices. On one
GPU add `--device cuda --ranks 1` (or `--device hip --ranks 1`) to the script above.
`IMRMLP_NATIVE_TEST_TIMEOUT` controls each runtime test's timeout in seconds.
`BUILD_TESTING=OFF` disables test targets and Python discovery.
