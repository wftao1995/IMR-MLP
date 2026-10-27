# MTP+LES/SOG LAMMPS interface

This directory contains the CPU/MPI interface for Coulomb LES v2 and SOG v3/v4:

- `pair_style mlip/opt-les POTENTIAL TYPE_TO_SPECIES...`
- `compute ID GROUP mlip/les/charge` exposes the Coulomb LES signed response
  charge using the model's `les_response_parity` convention.
- `compute ID GROUP mlip/sog/latent` exposes dimensionless SOG latent variables.
  SOG rejects the physical-charge compute.

The pair style requires `units metal`, a three-dimensionally periodic box,
and `newton on`.  It supports orthogonal and restricted-triclinic cells.  Its
reciprocal structure factors are MPI-reduced; descriptor reaction forces on
ghost atoms are returned by LAMMPS's normal reverse communication.

Only linear MTP short-range heads are supported. Nonlinear MTP-M heads
require the IMRMLP implementation and cannot be deployed with this pair style.
Use explicit LES v2 or SOG v3/v4 potentials. Unversioned LES files are rejected.

From the repository root, use the [CPU installer](../install.py), which includes
this pair style, its computes and all shared headers:

```sh
python3 LAMMPS/USER-MLIP/install.py /path/to/lammps/src
```

Reconfigure and rebuild LAMMPS as described in the [interface guide](../../README.md).
For a manual or out-of-tree build, include the four `pair_*` / `compute_*` files,
`mlip_pair_metadata.h`, `tools/mtp_metadata.h`, `tools/picojson.h`,
`potentials/sog_model.h` and `potentials/reciprocal_phases.h` with matching include
paths. Define the simulation box and atoms before the model coefficients.
For a two-species Coulomb LES model, use:

```text
units metal
newton on
pair_style mlip/opt-les model-v2.mtp 0 1
pair_coeff * *
compute qmodel all mlip/les/charge
dump q all custom 10 trajectory.lammpstrj id type x y z c_qmodel
```

The short-range `mlip/gpudirect` style does not evaluate LES/SOG long-range
terms. For complete native model graphs, including supported GPU long-range
paths, see [`mlip/native`](../native/README.md) and its model restrictions.

For SOG, use a v3 or v4 file in `pair_style` and `mlip/sog/latent` in `compute`.
SOG latent values are dimensionless model variables, not physical atomic charges;
do not feed them to a charged pair style or interpret them as Bader charges.
Neither compute replaces LAMMPS `atom->q`.
Zero mode (`include`/`omit`), analytic self subtraction (`onsite`/`none`),
reciprocal cutoff and taper are read from the model. Energy, descriptor
response forces and explicit cell virial use the same conventions as IMRMLP.
Global pressure and the sum of per-atom stresses include both reciprocal and
descriptor-response contributions; this also fixes the prior LES global
virial tally when LAMMPS selected its force-dot-position mode.

For multichannel v4 models, `mlip/sog/latent` produces a per-atom array:

```text
compute q all mlip/sog/latent
dump latent all custom 10 trajectory.lammpstrj id c_q[1] c_q[2] c_q[3]
```

The example assumes three channels. Single-channel models retain vector output
`c_q`. Define the pair style before creating the compute; recreate the compute
and its consuming dumps when switching to a model with a different channel count.
This style does not serialize its model into LAMMPS restart files; after
`read_restart`, define the model and computes again using the same files.
See the [multichannel guide (Chinese original)](../../../docs/advanced/sog-multichannel_zh.md).

## Regression tests

The harness replaces the registered pair/compute factories in an existing
static MPI LAMMPS build; it does not modify that installation. It requires
a Unix Makefiles CMake build (`CMakeFiles/lmp.dir/link.txt`), `mpicxx`, NumPy,
and the release IMRMLP library built in this checkout. Tested with LAMMPS
22Jul2025 Update 5. Run from the repository root:

```sh
python3 LAMMPS/USER-MLIP/opt-les/tests/build_regression.py \
  --lammps-source /path/to/lammps/src --lammps-build /path/to/lammps/build-mpi \
  --work /tmp/sog-lammps
OMP_NUM_THREADS=1 LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  /tmp/sog-lammps/reference /path/to/case-mtp-les/les.mtp /tmp/sog-lammps/cases
OMP_NUM_THREADS=1 python3 LAMMPS/USER-MLIP/opt-les/tests/run_sog_regression.py \
  --lmp /tmp/sog-lammps/lmp-sog --cases /tmp/sog-lammps/cases --ranks 1 2 4
```

The 60 numerical comparisons cover orthogonal/triclinic cells, all four SOG
zero/self combinations, Coulomb regression, atoms across periodic boundaries
and empty MPI ranks, including ranks with no ghost atoms. They check E, F, global virial, per-atom E/virial and q
against IMRMLP, plus malformed-file and observable rejection on two ranks.
Pass an optional channel count (for example `3`) after the reference generator's
output directory to generate v4 cases; the comparison script detects all latent
columns and checks them. The Coulomb cases remain single-channel. Use
`--phase direct` to compare the direct trigonometric path as well.
The tests do not assess long-trajectory stability or distributed IMRMLP training.

The optimized implementation caches cell-dependent reciprocal data, batches
MPI structure-factor reductions and uses integer harmonic recurrence for
phases. Append `phase direct` to `pair_style` to select direct trigonometric
evaluation, or `phase recurrence` (the default). Runtime choices do not change
the potential file. Use the `compare_trajectory.py` and `benchmark_mpi.py` scripts in `tests/`
for moving-cell validation and timings.
