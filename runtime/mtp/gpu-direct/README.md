# Direct MTP GPU kernels

This directory contains the standalone CUDA/HIP MTP kernels used by
`potentials/mtp_gpu_adapter.cpp`. Build the main application with
`make BUILD_MODE=cuda imrmlp` or `make BUILD_MODE=hip imrmlp`.

The runtime interface is defined in `mtp_gpu_direct.h`; backend selection and
allocation helpers are in `gpu_backend.h` and `gpu_runtime.h`. `sog_gpu.cu`
provides FP64 reciprocal SOG energies, forces, virials and parameter gradients
for complete structures in the IMR-MLP batch adapter, including multiple
channels and frozen MTP-M parameter groups.

Batch inputs, labels and reciprocal grids carry identity/revision tokens from
`gpu_cache.h`. A zero token disables caching; callers supplying a token must
change its revision whenever the corresponding contents change. The adapter
manages these revisions, detects evaluator changes, and rejects gradients from
stale or failed forward calls.

LAMMPS `mlip/gpudirect` continues to support short-range models with domain
decomposition. Its ghost-atom interface does not enable SOG; use `mlip/opt-les`
for a linear-head CPU/MPI model, or
[`mlip/native`](../../../LAMMPS/USER-MLIP/native/README.md) for supported native
CPU/GPU model graphs. See the [LAMMPS guide](../../../LAMMPS/README.md) for
build steps and model restrictions.

See the [project license](../../../LICENSE) and
[third-party notices](../../../THIRD_PARTY_NOTICES.md).
