# Harmonic reference for LAMMPS

`pair_style harmonic/reference` evaluates `E0 + 1/2 u^T Phi u` and `-Phi u`
from a version-1 `TI_HARMONIC_REFERENCE` file. It provides a fixed harmonic
reference for thermodynamic integration. Fit and validate the target potential
separately.

## Installation and use

From the repository root, install it with the [CPU installer](../install.py):

```sh
python3 LAMMPS/USER-MLIP/install.py /path/to/lammps/src
```

Alternatively, copy `pair_harmonic_reference.cpp` and `.h` into a clean LAMMPS `src` directory,
then reconfigure and rebuild LAMMPS. No `libimrmlp` link is required.

```text
units metal
# Create/read the simulation box, atoms and masses first.
pair_style harmonic/reference TI_FC_LAMMPS_REFERENCE auto
pair_coeff * *
```

Requirements: periodic boundaries, fixed cell, atom IDs `1..N` matching the
reference, matching atom types and positive per-type masses (`atom_style atomic`).
The style removes the mass-weighted global translational displacement and does
not request a neighbor list. Restart serialization is not supported; specify
the pair style and reference again after reading a restart.

For thermodynamic integration with a target MTP:

```text
variable lambda equal 0.5
variable one equal 1.0-v_lambda
pair_style hybrid/scaled v_one harmonic/reference TI_FC_LAMMPS_REFERENCE &
                         v_lambda mlip/opt
pair_coeff * * harmonic/reference
pair_coeff * * mlip/opt potential.mtp Al Ni
```

The optional storage argument is `auto` (default), `replicated` or `distributed`.
Replicated mode stores all FC rows per MPI rank. Distributed mode partitions
rows and gathers the force response. Auto selects distributed mode with multiple
ranks when the estimated dense FC blocks exceed 256 MiB per rank. Both modes
store positions and displacement buffers on every rank. Row kernels use OpenMP
when compiled with OpenMP; set `OMP_NUM_THREADS` to match allocated cores.

## Reference file

Generate reference files with this repository's `imrmlp ti_fit_fc` command.
For existing force constants and a reference structure carrying a finite energy:

```sh
imrmlp ti_fit_fc --workflow export --input REFERENCE.xyz --input-format extxyz \
  --initial-fc FORCE_CONSTANTS --index index.map \
  --write-validation-ensemble false --output-prefix TI_FC
```

MD fitting and self-consistent ensemble workflows are described in the
`ti_fit_fc` section of the [command reference](../../../docs/reference/commands.md).
Files using the version-1 format below remain compatible. The text format is:

```text
TI_HARMONIC_REFERENCE 1
units metal
natoms 2
energy0 -1
cell 10 0 0 0 10 0 0 0 10
atoms
1 1 2 2 2
2 1 7 2 2
blocks 3
1 1 2 0 0 0 2 0 0 0 2
1 2 -2 0 0 0 -2 0 0 0 -2
2 2 2 0 0 0 2 0 0 0 2
end
```

Cell vectors and each 3x3 FC block are row-major; cell vectors follow LAMMPS's
restricted triclinic convention. Atom rows contain `id type x y z`. Blocks use
one-based `(i,j)` indices with `i <= j`, in strict ascending order; lower blocks
are reconstructed by transposition. Only nonzero blocks need storage. Units are
angstrom and eV (FC: eV/angstrom²). The reference should satisfy the acoustic sum
rule and have stable nontranslational modes; the adapter does not fit or
stabilize the Hessian.

## Regression

Run against a LAMMPS executable containing this style:

```sh
python3 LAMMPS/USER-MLIP/hyb-ti/tests/run_regression.py /path/to/lmp
python3 LAMMPS/USER-MLIP/hyb-ti/tests/run_regression.py /path/to/lmp --ranks 2
```

The test checks an analytic two-atom spring's energy, forces and per-atom energy,
periodic translation, all storage modes, `hybrid/scaled`, and invalid FC input.
`tests/lammps_runner.cpp` can be linked with the pair source and a matching
upstream LAMMPS library to register the style without modifying upstream sources.
Historical `debye/opt` and `ti/debye` adapters are not included.
