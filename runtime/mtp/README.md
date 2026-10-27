# Standalone MTP reader

`mlip.cpp` / `mlip.h` provide the MTP reader used by the standalone LAMMPS CPU/OpenMP and GPU interfaces. The MTP 1.1.0 `scaling` field is optional and defaults to **1.0**, matching the native potential reader. Explicit scaling remains supported. Compare energy, forces and virial with the native evaluator before deploying a model through another backend.

The parser regression includes models with an omitted scaling field and an explicit value of 0.5:

```sh
make build/release/unitest/ut_mlip_reader.test
LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" build/release/unitest/ut_mlip_reader.test
make test
```

For CUDA or HIP/DCU deployment, use the same model, structures and species map
when comparing native CPU/GPU and LAMMPS CPU/GPU results. Use separate build
directories for each backend.

`cpu/mtp_cpu.*` implements the short-range MTP-M executor. `radial_basis.h`,
`radial_basis_setup.cpp` and `radial_cache.h` provide shared radial evaluation
and polynomial initialization. The core reads legacy MTP and GPM bundles;
long-range contributions are handled by the native interface.

Build and export through [standalone CMake](../../LAMMPS/USER-MLIP/standalone/README.md).
The exported archive carries its own metadata and Eigen headers.
