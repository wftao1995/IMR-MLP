# Periodic external field with fixed Born effective charges

[中文](README_zh.md)

This example requires external model and response data. Replace the placeholder
paths below with your own. To validate an installation, start with the
[quickstart](../quickstart/README.md).

The training, analytic-BEC and supercell validation scripts of the original
Coulomb LES work are at the end of this page. They measure dynamic response
errors and are separate from the fixed-DFT-BEC field-force entry point below.

This example adds independent DFT Born effective charges (BEC) to an existing
zero-field potential. It supports `units metal`, three-dimensional periodic
boundary conditions, a fixed cell and a constant macroscopic electric field.
Use DFT response tensors for this entry point. This example does not validate
LES finite-wavevector response as physical BEC; integrability and invariance
under lattice-basis changes require separate checks.

Run the Python commands below from the repository root. Generation needs NumPy;
exporting DFT data additionally needs ASE. Running the generated input needs a
LAMMPS build with the matching LES/SOG pair style and its runtime libraries.

## Physical conventions

Rows of the input BEC are polarization/electric-field directions, columns are
displacement/force directions, and values are normalized by the elementary charge
e. Field input is in V/Å, coordinates in Å, and the resulting extra forces are in
eV/Å with energies in eV:

```text
F_i = Z_i^T E
Delta mu = sum_i Z_i (r_i_unwrapped - r_i_reference)
U_field = -E . Delta mu
```

`U_field` is the field energy relative to the given reference configuration,
omitting the constant `-E . mu_reference`. Outputs of different phases therefore
cannot be compared directly for field-dependent free energies; an absolute
reference polarization must be supplied separately. The BEC used here are DFT
values that already represent the physical response, so no
`sqrt(epsilon_infinity)` factor is applied. The field is the macroscopic E inside
the bulk; the boundary problem between electrodes, external voltage and internal
field is not solved.

Fixed BEC is a small-displacement approximation around the reference structure.
LAMMPS image flags keep displacements continuous when atoms cross the box
boundary; do not repeatedly reset the reference displacement with the minimum
image convention. It does not apply when atoms migrate a full lattice period, or
when diffusion and polarization-switching paths are treated with constant BEC.
Cell response, higher-order terms of field-induced electronic polarization and
the full electric-field stress are not implemented; the generated input disables
the extra virial and checks at initialization and at every MD step that the cell
is unchanged.

## Input and generation

The `fixed-bec-reference-v1` JSON explicitly contains `cell_A` (lattice vectors
as rows), `positions_A`, `species`, per-atom `masses_amu`, `bec_e` (N×3×3) and
`bec_axes: "polarization,displacement"`. It must be generated from physical
response data; do not fill in the SOG latent arrays. `export_dfpt.py` is an
adapter for a particular VASP directory layout. For each phase it expects
`PHASE/high_cutoff/dfpt/` under `--source`, containing `POSCAR`, `INCAR`,
`KPOINTS`, `OUTCAR`, `vasprun.xml` and a previously prepared `response.json`.
That JSON must contain `gap.indirect_gap_eV`, `BEC_e` and `epsilon_infinity`.
It is not generated from arbitrary VASP output by this script. The script reads
the final XML tensors and checks them against `response.json`.

```sh
python3 examples/les-response/export_dfpt.py \
  --source /absolute/path/DFT_BEC \
  --output /absolute/path/new-reference-export
```

Default phases are `alpha_2H`, `beta_2H` and `alpha_3R`; use `--phases` to select
other directory names. The exporter requires a gap above 0.01 eV, normal VASP
termination and consistent finite tensors, and saves input hashes and checks.
If your source layout differs, prepare the documented reference JSON directly
or adapt the exporter to your data.

With the reference JSON available locally:

```sh
python3 examples/les-response/fixed_bec_field.py \
  --reference /absolute/path/alpha_2H.json \
  --work /absolute/path/new-field-case \
  --field 0 0 0.01 --replicate 2 2 2 \
  --model /absolute/path/sog3.mtp --type-order In Se
```

`--field` is given in the original Cartesian frame of the reference JSON. The
program converts a general triclinic cell to the LAMMPS restricted triclinic
form, rotating positions, BEC and field together, and writes the transformation
matrix to `manifest.json`. BEC are copied per atom when replicating a supercell,
without multiplying by volume or replica count. `--type-order` must equal the
species index order used in training; here model species 0=In and 1=Se.

The acoustic sum rule is enforced by default: the reference cell must satisfy
`max(abs(sum_i Z_i)) <= 1e-6 e`. To remove a confirmed numerical rounding
residual, pass `--project-asr` explicitly; the original residual, whether
projection was applied and the result are all recorded. Do not use projection to
hide definition errors or unconverged calculations.

Outputs:

- `reference.data`: atom IDs, elements, cell and reference positions.
- `init.inc`: zero-field potential and simulation initialization; `--model`
  uses `mlip/opt-les`, and omitting it uses `pair_style zero` for interface tests.
- `field.inc`, six `*.atom` files: fixed external-field forces and reference
  coordinates stored by atom ID, applied through `fix addforce` together with a
  consistent energy.
- `in.field`: the `run 0` calculation and force output, as a validated minimal
  entry point.
- `manifest.json`: input/model hashes, type mapping, original and rotated field,
  full tensors and reference coordinates.

Run in the result directory with a LAMMPS that includes this library's LES/SOG
pair styles:

```sh
OMP_NUM_THREADS=1 mpirun -np 4 /absolute/path/lmp-sog -in in.field
```

For fixed-cell MD, add `fix nve` or `fix nvt` and the corresponding protocol after
`init.inc` and `field.inc`. On restart, keep the atom IDs, count, image flags and
the original reference files, and redefine `field.inc`; never treat restart
positions as a new reference and lose the work already done by the field. Do not
`replicate`, reorder IDs or add/remove atoms after loading these per-atom data.
The current generator targets fixed-cell MD only.

## Validation

```sh
python3 examples/les-response/validate_fixed_bec.py \
  --lmp /absolute/path/lmp-sog --work /absolute/path/new-checks \
  --references /absolute/path/reference-export \
  --model /absolute/path/sog3.mtp
python3 examples/les-response/validate_fixed_bec.py --dynamics-only \
  --lmp /absolute/path/lmp-sog --work /absolute/path/new-checks \
  --references /absolute/path/reference-export \
  --model /absolute/path/sog3.mtp
```

The first step checks the transpose orientation of asymmetric BEC, triclinic
rotation, ASR rejection and explicit projection, force/energy additivity of real
SOG+DFT BEC for three phases on 1/2/4 MPI ranks, and a synthetic system across
PBC, a 4→2 rank restart and cell-change rejection. The second adds a 100-step,
0.1 fs NVE run for the three phases. These are interface-correctness checks, not
long-time accuracy or phase-transition predictions.

LAMMPS references: [fix addforce](https://docs.lammps.org/fix_addforce.html),
[atomfile variables](https://docs.lammps.org/variable.html), [unwrapped
coordinates](https://docs.lammps.org/compute_property_atom.html).

## Analytic BEC, supercell and mean-force removal checks

This optional analysis needs a compatible external `les_bec_evaluate` helper
passed through `--executable`. Its source and build recipe are not bundled in
this checkout; the main `imrmlp` build does not produce it. If you already have
that helper and its matching `libimrmlp.so`, run:

```sh
python3 examples/les-response/check_les_convergence.py \
  --model /absolute/path/les-model.mtp \
  --executable /absolute/path/les_bec_evaluate \
  --references /absolute/path/dfpt-json-directory \
  --work /absolute/path/new-response-results
```

The inputs reuse the three-phase `fixed-bec-reference-v1` exports above. It uses
the scalar trace(ε∞)/3 without extra fitting for anisotropy, fixes one global
sign on the largest α-2H supercell and then uses it for all phases. α-2H is
therefore the sign anchor, and the three configurations are reference-point
checks, not a broad independent validation set.

The script reports, for 1/2/3/4/6/8/12-fold anisotropic replication, equivalent
basis changes, replica consistency, ASR and nine-component BEC errors. The
primitive cell and the 2-fold cell are cross-checked against independent central
differences. For 0.01 Å perturbations from a fixed seed, it computes the
antisymmetric part of the local force Jacobian and the work around rectangular
loops of two edge lengths on 1/4/12-fold cells, recording both raw forces and
forces with the system-wide mean removed. Perturbed configurations have no DFT
labels and do not enter BEC accuracy statistics.

The loop moves atoms only in the first replica while the other replicas stay
fixed; this is a local degree-of-freedom check, not a coherent displacement of
all replicas. The loop plane follows the largest antisymmetric component of the
raw force Jacobian, and projected loops are compared in the same plane. The
finite-wavevector BEC in the JSON are not used directly for field energies. The
script's preliminary accuracy/size thresholds are not universal physical
tolerances; passing does not establish the accuracy of field-driven MD.

Existing results under `--work` are reused only when the model, shared library,
program and reference-file hashes all match. The program automatically prefers
`libimrmlp.so` next to `--executable`. Run after builds and tests finish; if
parallel builds are still running, copy the program and library to a separate run
directory to avoid loading an incomplete library during relinking.
This workflow calls no DFT, computes no Berry-phase polarization and runs no field
dynamics.
