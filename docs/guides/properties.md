# Crystal properties {#guide_properties}

[中文](properties_zh.md)

Compute force constants, phonons, elastic constants and equations of state with a
trained model; each property has its own convergence check.

Out of scope: obtaining the equilibrium structure (see
[structure relaxation](relaxation.md)) and model training (see
[training, evaluation and deployment](workflows.md)).

## Required inputs

| File | Content | Source |
| --- | --- | --- |
| `model.mtp` | Trained ordinary MTP model | Training output or a model provider |
| `relaxed.vasp` | Equilibrium structure | POSCAR exported with `convstr` from the relaxation result |
| `index.map` | Model species order | The model or its training records |

Run each calculation in its own directory. Phonons additionally need Phonopy
installed for postprocessing. Path and naming rules are in
[input files and settings conventions](conventions.md). EOS and zero-pressure
elasticity also require the corresponding cell optimization first.

## Quantities and tools

| Goal | Tool | Check first |
| --- | --- | --- |
| Second-order atomic response and harmonic force constants | `calc_fc` | Equilibrium structure, supercell and Hessian backend |
| Phonon dispersion, DOS and vibrational thermodynamics | IMR-MLP forces/force constants with Phonopy postprocessing | Atom order, supercell definition and masses |
| Stiffness and strain response | `calc_elastic` | Equilibrium stress, ionic-relaxation convention and strain amplitude |
| Equilibrium volume, bulk modulus and E–V curve | `calc_eos` | Stable volume interval in one phase and relaxation settings |

## Force constants

Build a 2×2×2 supercell from the input cell and export full force constants:

```sh
imrmlp calc_fc --potfile model.mtp \
  --input relaxed.vasp --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
```

Inspect `FORCE_CONSTANTS`, `supercell_created.xsf` and `sup_debug_efs.xyz`.
Blocks use eV/Å² and follow the **actual exported supercell** order. `2 2 2` is a
starting point; increase the supercell to check convergence. The default analytic
backend errors for unsupported models; `--hessian-method auto` uses analytic
derivatives when available and force finite differences otherwise, or select
`force_fd` explicitly and converge `--hessian-nudge-pos`.

`--lowest-modes N` is a separate matrix-free lowest-Gamma-mode diagnostic: it
exports eigenmodes, not full `FORCE_CONSTANTS`, and does not replace
whole-Brillouin-zone phonon dispersion. More backends and outputs are defined in
the [command reference](../reference/commands.md).

## Phonons: dispersion and DOS

Alternative to the previous section. To let Phonopy generate displacements and
IMR-MLP evaluate forces, copy the same equilibrium structure, model and map into a
fresh directory to avoid mixing in another `FORCE_CONSTANTS`. The following uses
Phonopy 4 command separation:

```sh
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
imrmlp phonopyfc --potfile model.mtp --disp phonopy_disp.yaml \
  --index-file index.map
```

`phonopy-init` creates `phonopy_disp.yaml`; `phonopyfc` writes `FORCE_SETS`. Then
calculate an illustrative DOS mesh:

```sh
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml
```

Check `total_dos.dat` and the log, then vary the mesh or choose a material-specific
band path. Phonopy 3 uses `phonopy -d`; Phonopy 4 moves setup to `phonopy-init` and
reads YAML during postprocessing without repeating `-c` / `--dim`, see the
[official migration guide](https://phonopy.github.io/phonopy/migration-v4.html).
`phonopyfc` uses `--index-file`, not the common `--index`. If `settings.ini` exists
in the directory, check its `pot:` settings against the current model. Meshes, band
paths and plotting are covered by Phonopy's
[command options](https://phonopy.github.io/phonopy/command-options.html) and
[setting tags](https://phonopy.github.io/phonopy/setting-tags.html).

The [Python Phonopy adapter](../reference/python.md#phonopy-integration) can instead
evaluate displaced-supercell forces in memory. For direct import of analytic
`FORCE_CONSTANTS`, use Phonopy's own supercell as `calc_fc` input without expanding
it again, and match both atom axes to `phonon.supercell` exactly; matching atom
counts is insufficient.

Imaginary modes can reflect physical instability, incomplete relaxation,
displacement/supercell errors, or model/unit/order problems. Check equilibrium
forces, acoustic modes and numerical convergence before physical interpretation.
Phonon force constants differ from finite-temperature effective force constants;
see [harmonic references and TI](../advanced/harmonic-ti.md) for the latter.

## Elastic constants

For ordinary MTP, start with the default automatic backend:

```sh
imrmlp calc_elastic --potfile model.mtp --input relaxed.vasp \
  --elastic-output elastic.out > elastic.log 2>&1
```

The default `auto` chooses a supported method. Use `--method analytical` when needed
and confirm atomic/cell-Hessian support. For a finite-difference comparison, create
`elastic.ini`:

```text
relax:cartesian true
relax:gtol 0.0001
relax:maxiteration 500
```

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --method numerical \
  --settings elastic.ini --strain-magnitude 0.005 --elastic-output elastic-numerical.out
```

Compare strain amplitudes and tighter ionic tolerances. Distinguish clamped-ion
stiffness from stiffness with relaxed internal coordinates; analytic
`--relax-correction` controls the related correction. The role and units of each
output block, and the usual misreadings, are covered in
[reading the elastic output files](#elastic-output-format) below; in particular note
the difference between material stiffness and the Born/Wallace tangent at nonzero
pressure. For internal soft-mode or singular-stiffness errors, check structural and
model stability first; disabling the ionic correction changes the physical quantity
and is not a general repair. See the
[command reference](../reference/commands.md) for complete definitions.

### Reading the elastic output files {#elastic-output-format}

The file given by `--elastic-output` (default `elastic.out`) is written as a fixed
sequence of blocks with no blank lines between them:

| Block | Content and units |
| --- | --- |
| `EPS: ...` | Numerical method only, recording the strain amplitude actually used; the analytical method has no such line |
| `Stiffness tensor (GPa)` | 6×6 material stiffness C, symmetrized onto the space group found with `--symprec` |
| `Born/Wallace stiffness tensor (GPa)` | Tangent tensor B used for the mechanical-stability criterion; identical to the previous block at zero pressure |
| `Compliance tensor (GPa^-1)` | Compliance S obtained by inverting C, not B |
| `Elastic properties` | `B`, `G`, `E`, `v` and `G/(9B)` under Voigt, Reuss and Hill averaging |
| `1.0 / S[0][0]` | Uniaxial modulus along the first Cartesian axis; not the same quantity as the polycrystalline average `E` in the table |
| `# BORN/WALLACE STIFFNESS EIGENSYSTEM` | Six eigenvalues of B (GPa) and the stability verdict |

The matrices carry no row or column labels; both follow Voigt order `11 22 33 23 31 12`,
so rows/columns 4, 5 and 6 are yz, xz and xy. Getting that order wrong swaps `C44`,
`C55`, `C66` with `C12`, `C13`. Note also that the Reuss column is spelled `Ruess` in
the output, which matters when you grep for it.

Three points when reading it:

1. "The elastic constants" normally means the first block, C. It is the ion-relaxed
   stiffness when the analytical `--relax-correction` is on (the default), and when
   the numerical method relaxes the ions at every strain point; turning the correction
   off gives clamped-ion stiffness instead. The two can differ substantially and must
   not be compared with each other.
2. B differs from C only at nonzero pressure: diagonal entries lose P and
   normal-stress off-diagonal entries gain P. For one structure at 5 GPa, `C11` =
   97.90 against `B11` = 92.90, and `C12` = 54.57 against `B12` = 59.57. The
   eigenvalues and the "Born elastic stability conditions meet / NOT meet" verdict are
   read from B only; the compliance tensor and the modulus table come from C. Do not
   cross-reference the two sets.
3. Entries below an internal threshold (`1e-10` analytical, `1e-5` numerical) are
   printed as `0`. A symmetry-forbidden component should vanish; if it prints a
   visible number, `--symprec` probably does not match the structure or the structure
   is not well relaxed. Conversely, printing `0.000000` does not mean that component
   was constrained.

Other output files:

| File | Written when | Content |
| --- | --- | --- |
| `eigensys.out` | Analytical method only, always (rename with `--eigensys-output`) | 9×9 `cellhess` (eV/Å²), its eigensystem, 9×9 `C_ijkl` and 6×6 `C_ij` (both in eV, not divided by volume) with their eigensystems |
| `eigtotal.dat` | Analytical method only, with `--compute-eigsys` | Combined (3N+9)×(3N+9) cell-plus-atomic Hessian and its eigensystem; the first nine degrees of freedom are the cell |
| `debuginfo.log` | Numerical method only, with `--debug` | Strain vector and stress vector for every strain point, for refitting by hand |

Tensors in `eigensys.out` are in eV rather than GPa (in the same example `C11` =
35.77 eV corresponds to 97.90 GPa, differing by a volume factor and 160.217662), so
they cannot be compared with `elastic.out` values directly. Their purpose is to show
whether a cell-Hessian eigenvalue softens; to locate the displacement pattern of a
soft mode, use `eigtotal.dat` instead of the 6×6 eigenvalue list.

The numerical method also prints the fit before symmetrization to stdout as
`Original fitted C:` (it is not in `elastic.out`). A large gap between that and the
projected tensor in the file points at the strain points or the ionic relaxation.

Finally, `B` in `elastic.out` is not the `B0` in `eos.dat`: the former is the
stiffness of the current structure at finite strain, the latter the equilibrium bulk
modulus from the EOS fit. They agree only when the structure really sits at the
energy minimum.

### Strain-amplitude check

Keep `elastic-numerical.out` from the `0.005` (0.5%) calculation and repeat with the
same input and internal-coordinate relaxation settings:

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --method numerical \
  --settings elastic.ini --strain-magnitude 0.0025 \
  --elastic-output elastic-half-strain.out > elastic-half-strain.log 2>&1
```

For example, `C11` values of `162.4` and `162.3 GPa` from the first block of each
output differ by `0.1 GPa`, about 0.06%. These illustrate comparison, not expected material values or a universal
acceptance criterion. Also compare other independent components, moduli and ionic
convergence at strained points. Substantial drift on halving the step calls for checking
relaxation tolerances and finite-difference stability; two step sizes agreeing on the same
`Cij` mark the plateau, and shrinking further only amplifies round-off. Apply the same
approach to phonon supercells or DOS meshes: change one numerical setting at a time and
retain separate results.

## Equation of state

Save `eos.ini` with the equilibrium structure's model and relaxation accuracy:

```text
pot:potfile model.mtp
relax:gtol 0.0001
relax:maxiteration 500
```

```sh
imrmlp calc_eos --settings eos.ini --input relaxed.vasp \
  --output ev.dat --output-eos eos.dat
```

The default window samples ±5% around the reference volume and yields curve data plus
Birch–Murnaghan parameters. `calc_eos` only checks that the fitted `V0` and `B0` are
positive and finite; it does not check that the minimum lies inside the sampled window.
When the stationary point falls outside, `eos.dat` still receives extrapolated `V0` and
`B0`, and a different window yields a different set. The check is `ev.dat`: whether
`Energy/at` first falls then rises inside the window, and whether `Press` crosses zero
there — that crossing is `V0`. If not, narrow the window with `--lower`/`--upper` or
relax the equilibrium structure again, then vary `--npos`/`--nneg` to assess stability.
A window without a positive-curvature minimum makes the program report
`EOS fit has no stable finite equilibrium point` and exit nonzero without writing
`eos.dat`. Pressure scans use `--curve pv` with `--press-lb`/`--press-ub`, writing only
`Pressure`, `Volume`, `Energy/at` and `H/at`, with no EOS fit.

### EOS output columns and units

| File / field | Meaning and units |
| --- | --- |
| `ev.dat`: `Volume/at`, `Energy/at`, `Enth/at` | Å³/atom, eV/atom, eV/atom |
| `ev.dat`: `Press` | GPa |
| `eos.dat`: `V0`, `E0` | Fitted equilibrium volume and energy, Å³/atom and eV/atom |
| `eos.dat`: `B0` | eV/Å³; multiply by 160.217662 for GPa |
| `eos.dat`: `B0'` | Pressure derivative of bulk modulus, dimensionless |

These apply to the structure-scanning E–V command. When fitting an external curve,
volume/energy normalization follows its input data and is not automatically per
atom. Do not directly compare `calc_efs` total energy with `Energy/at`.

## Recording results

Retain the equilibrium structure, model, map, settings and logs, recording
supercell, displacement/strain amplitude, pressure and ionic-relaxation convention.
Converge numerics before comparison with reference properties.

## Common errors

| Symptom | Cause | Action |
| --- | --- | --- |
| Imaginary phonon modes | Physical instability, incomplete relaxation, displacement/supercell error or model problem | Check equilibrium forces, acoustic modes and supercell convergence before interpreting |
| Stiffness off by an order of magnitude | Clamped-ion and relaxed-internal-coordinate conventions mixed | Establish the meaning of `--method` and `--relax-correction` before comparing |
| Two stiffness blocks disagree | `--pressure` is nonzero; the second is the Born/Wallace tangent | Read material constants from the first block and stability from the second |
| `eigensys.out` values look an order of magnitude larger | They are in eV and not divided by volume | Convert with the volume and 160.217662 before comparing |
| Internal soft mode or singular stiffness | Structure or model is unstable | Check stability first; disabling the ionic correction is not a repair |
| Results drift when the strain step is halved | Relaxation tolerance too loose or finite differences unstable | Tighten `relax:gtol` and recheck one setting at a time |
| Unreasonable EOS fit | Sampling crosses a phase boundary, or the minimum lies outside the window | Narrow the window with `--lower`/`--upper`, add points with `--npos`/`--nneg` and confirm a single phase |
| `Energy/at` disagrees with single-point energy | Total and per-atom energies mixed | Convert by atom count before comparing |

## Related

- [Structure relaxation](relaxation.md): source of the equilibrium structure
- [Command reference](../reference/commands.md), [Python Phonopy adapter](../reference/python.md#phonopy-integration)
- [Harmonic references and TI](../advanced/harmonic-ti.md)
