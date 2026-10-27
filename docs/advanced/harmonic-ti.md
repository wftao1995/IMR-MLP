# Harmonic reference and thermodynamic integration {#advanced_harmonic_ti}

[中文](harmonic-ti_zh.md)

Free-energy differences between a machine-learning potential and a reference
system are obtained by thermodynamic integration (TI). This requires a reference
whose free energy is analytic and whose phase space overlaps the target: a
positive-definite harmonic reference fitted at the temperature of interest.

IMR-MLP provides both halves of that construction:

- `imrmlp ti_fit_fc` fits a positive finite-temperature effective force-constant
  (FC) matrix from MD or a self-consistent ensemble and exports it as a
  version-1 reference file.
- `pair_style harmonic/reference` (`LAMMPS/USER-MLIP/hyb-ti/`, installed by the
  CPU installer) evaluates that reference in LAMMPS, where
  `pair_style hybrid/scaled` mixes it with an MTP at a coupling parameter.

The reference is a fitting and sampling artifact for TI, not a material model:
it reproduces the harmonic part of the target's finite-temperature behavior and
is meaningful only together with the structure, temperature and FC it was
fitted from.

## Fitting the reference with `ti_fit_fc`

The workflow must be selected explicitly — `md`, `ensemble`, `md-ensemble` or
`export`:

```bash
imrmlp ti_fit_fc --workflow md-ensemble \
  --traj md.extxyz --traj-format extxyz \
  --potfile potential.mtp --index index.map \
  --temperature 600 --output-prefix BETA_600K
```

| Workflow | Meaning |
| --- | --- |
| `md` | Fit to a supplied MD trajectory at the given temperature. |
| `ensemble` | Self-consistent Gaussian ensemble sampled from the current reference. |
| `md-ensemble` | MD fit followed by self-consistent refinement; the usual choice. |
| `export` | Convert existing force constants to the LAMMPS reference file without fitting. |

Input requirements, enforced by the command:

- The trajectory is fixed-cell NVT. Variable-cell (NPT) trajectories are
  rejected — obtain the average NPT cell first and run fixed-cell NVT there.
- Without `--input`, the MD workflow unwraps the trajectory, removes the
  mass-weighted center-of-mass drift and uses its centroid as the reference.
- Forces are recomputed with the selected potential by default. Pass
  `--force-source trajectory` explicitly to consume stored trajectory forces.
- Periodic images are aligned after removing a common translation, including
  when the center-of-mass drift crosses half a cell. Relative crystalline
  motion between atoms must remain below half a period.

Options are listed in the `ti_fit_fc` section of the
[command reference](../reference/commands.md); run `imrmlp ti_fit_fc --help`
for the complete table.

## What the fit enforces

Fitting and sampling operate in the mass-weighted internal subspace, so pure
translations are excluded by construction.

- The default staged fit first updates modal stiffnesses, then coupled soft and
  near-degenerate blocks, using a positive square-root representation.
  `--root-representation` selects `normal`, `sqrt` (default) or `root4`.
- Both stages minimize the weighted mean squared force residual plus a
  `--ridge` penalty toward the current FC. Duplicating samples does not change
  its strength.
- `--initial-floor-thz` (default 1 THz) is the physical sampling stiffness used
  in every self-consistent iteration; `--pd-floor-thz` (default 1e-4 THz) is only
  the strict numerical positive floor of the fitted FC. They are not the same
  quantity.
- Self-consistent iterations sample an antithetic Gaussian ensemble, evaluate
  true MTP forces and apply a damped root-space update (`--mixing`). Convergence
  is counted only after the modal warmup (`--warmup-iterations`) has ended.
- Candidates violating displacement or minimum-distance limits are rejected as
  complete `+q/-q` pairs and replaced. With `--max-sampling-attempts=0` the
  budget is twenty times the requested ensemble size; exhausting it reports
  rejection reasons and geometry extrema and fails — it never silently lowers
  the sampling floor or fits an undersized set.

## Outputs and acceptance

A run writes, under `--output-prefix`:

| File | Contents |
| --- | --- |
| `<prefix>_FORCE_CONSTANTS` | Fitted force constants. |
| `<prefix>_LAMMPS_REFERENCE` | Version-1 reference consumed by LAMMPS `pair_style harmonic/reference`. |
| centroid/reference structure | The structure the reference was built around. |
| iteration CSV and diagnostics | Per-iteration convergence, sampling attempts, acceptance rate, geometry extrema. |
| validation ensemble | An independent ensemble used only for validation. |
| report | Training/holdout errors, floors, and the LAMMPS-mode diagnostics. |

How to read the errors:

- Training errors describe the centered fitting objective.
- Holdout errors, `final_md_force_rmse` and `harmonic_validation_force_rmse` use
  the actual harmonic force residual, without removing a validation-set force
  offset. All are in mass-weighted internal force coordinates
  (eV/Å/√amu).
- The `lammps_*` diagnostics and the `# lammps_mode` table always describe the
  exact FC written to `<prefix>_LAMMPS_REFERENCE`. Use those — not the fitting
  errors — for TI acceptance and for analytic harmonic free energies.
- A persistently low acceptance rate indicates an unsuitable reference or
  sampling floor, not a reason to relax safety limits.

Sparse export for LAMMPS can truncate 3×3 atom-pair blocks
(`--lammps-fc-block-threshold`, `--lammps-fc-cutoff`), after which onsite blocks
are reconstructed to enforce the acoustic sum rule and, when requested, the
smallest isotropic stabilizer raises the minimum internal frequency
(`--lammps-fc-min-frequency-thz`). Always inspect the retained block fraction,
the relative Hessian change, the frequencies before/after stabilization and the
ASR error in the report: truncating a `fitted` FC can change the reference
strongly, while `--lammps-fc-source initial` preserves the original local FC.

```bash
imrmlp ti_fit_fc --workflow export \
  --input REFERENCE.xyz --input-format extxyz \
  --initial-fc FORCE_CONSTANTS --index index.map \
  --lammps-fc-source initial --lammps-fc-block-threshold 0.01 \
  --lammps-fc-min-frequency-thz 0.5 \
  --write-validation-ensemble false --output-prefix TI_SPARSE
```

## Using `pair_style harmonic/reference` in LAMMPS

The style is part of the [LAMMPS interfaces](../../LAMMPS/README.md) and is
installed by `python3 LAMMPS/USER-MLIP/install.py` (the `hyb-ti` folder); it can
also be copied manually into a clean LAMMPS `src`. No `libimrmlp` link is
required.

```text
units metal
# Create/read the simulation box, atoms and masses first.
pair_style harmonic/reference BETA_600K_LAMMPS_REFERENCE auto
pair_coeff * *
```

Requirements: periodic boundaries, a fixed cell, atom IDs `1..N` matching the
reference, matching atom types and positive per-type masses
(`atom_style atomic`). The style removes the mass-weighted global translational
displacement and does not request a neighbor list. Restart serialization is not
supported: after reading a restart, specify the pair style and reference again.

The optional storage argument is `auto` (default), `replicated` or
`distributed`. Replicated mode stores all FC rows per MPI rank; distributed mode
partitions rows and gathers the force response. Auto selects distributed mode
with multiple ranks when the estimated dense FC blocks exceed 256 MiB per rank.
Row kernels use OpenMP when compiled with OpenMP; set `OMP_NUM_THREADS` to match
the allocated cores.

## Thermodynamic integration with an MTP

The reference and the target potential are mixed with a coupling parameter
λ, so that λ = 0 is the harmonic reference and λ = 1 is the full MTP:

```text
variable lambda equal 0.5
variable one equal 1.0-v_lambda
pair_style hybrid/scaled v_one harmonic/reference BETA_600K_LAMMPS_REFERENCE &
                         v_lambda mlip/opt
pair_coeff * * harmonic/reference
pair_coeff * * mlip/opt potential.mtp Al Ni
```

Use element names in the MTP `pair_coeff`, as described in
[LAMMPS element mapping](../../LAMMPS/README.md).

Each λ point is an independent LAMMPS run. IMR-MLP supplies the reference and
the mixing-capable pair styles; the λ schedule, the integration of
\(\langle U_{\mathrm{MTP}} - U_{\mathrm{harm}}\rangle_\lambda\) over λ, and the
error estimate belong to the user's workflow. Because the reference is fitted at
one temperature, reuse it only at that temperature and for the cell it was
fitted in.

## Reference file format

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

Cell vectors and each 3×3 FC block are row-major; cell vectors follow LAMMPS's
restricted triclinic convention. Atom rows are `id type x y z`. Blocks use
one-based `(i,j)` indices with `i <= j` in strict ascending order; lower blocks
are reconstructed by transposition, and only nonzero blocks need storage. Units
are Å and eV (FC: eV/Å²). The reference should satisfy the acoustic sum rule and
have stable nontranslational modes — the adapter does not fit or stabilize the
Hessian. Files written by `atoms-agent`'s `gpmain ti_fit_fc` remain compatible.

## Validation and regression

- Unit tests: `unitest/ut_ti_harmonic_reference.cpp` and
  `unitest/ut_ti_reference_io.cpp` cover fitting and reference I/O.
- LAMMPS-side regression, against an executable containing the style:

```bash
python3 LAMMPS/USER-MLIP/hyb-ti/tests/run_regression.py /path/to/lmp
python3 LAMMPS/USER-MLIP/hyb-ti/tests/run_regression.py /path/to/lmp --ranks 2
```

The test checks an analytic two-atom spring's energy, forces and per-atom
energy, periodic translation, all storage modes, `hybrid/scaled`, and invalid FC
input.

## Limits

- The reference is fitted at one temperature and cell; it is not transferable.
- The LAMMPS adapter neither fits nor stabilizes the Hessian; a non-positive or
  ASR-violating reference is used as given.
- Restart of the pair state is not supported.
- Historical `debye/opt` and `ti/debye` adapters are not included in this
  release.
- GPU execution of the harmonic reference requires separate device validation;
  `mlip/gpudirect` does not cover this style.
