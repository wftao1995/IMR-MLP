# Crystal properties {#guide_properties}

[中文](properties_zh.md)

Compute force constants, phonons, elastic constants and equations of state with a
trained model. Small single-point errors for a material do not establish accurate
force constants, elasticity or volume response; each property needs its own check.

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
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
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
  --index index.map --elastic-output elastic.out > elastic.log 2>&1
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
  --input relaxed.vasp --index index.map --method numerical \
  --settings elastic.ini --strain-magnitude 0.005 --elastic-output elastic-numerical.out
```

Compare strain amplitudes and tighter ionic tolerances. Distinguish clamped-ion
stiffness from stiffness with relaxed internal coordinates; analytic
`--relax-correction` controls the related correction. Read stiffness/modulus units
from output headers. At nonzero pressure, distinguish material stiffness from the
Born/Wallace tangent used for mechanical stability. For internal soft-mode or
singular-stiffness errors, check structural and model stability first; disabling
ionic correction changes the physical quantity and is not a general repair. See the
[command reference](../reference/commands.md) for complete definitions.

### Strain-amplitude check

Keep `elastic-numerical.out` from the `0.005` (0.5%) calculation and repeat with the
same input and internal-coordinate relaxation settings:

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --index index.map --method numerical \
  --settings elastic.ini --strain-magnitude 0.0025 \
  --elastic-output elastic-half-strain.out > elastic-half-strain.log 2>&1
```

For example, `C11` values of `162.4` and `162.3 GPa` differ by `0.1 GPa`, about
0.06%. These illustrate comparison, not expected material values or a universal
acceptance criterion. Also compare other independent components, moduli and ionic
convergence at strained points. Substantial drift on halving the step calls for
checking relaxation tolerances and finite-difference stability; smaller steps are
not always more accurate. Apply the same approach to phonon supercells or DOS
meshes: change one numerical setting at a time and retain separate results.

## Equation of state

Save `eos.ini` with the equilibrium structure's model and relaxation accuracy:

```text
pot:potfile model.mtp
relax:gtol 0.0001
relax:maxiteration 500
```

```sh
imrmlp calc_eos --settings eos.ini --input relaxed.vasp --index index.map \
  --output ev.dat --output-eos eos.dat
```

The default window spans roughly ±5% around the reference volume, producing curve
data and EOS parameters. Check that points remain in one phase, bracket an energy
minimum and give sensible fitted volume and bulk modulus. Vary the volume interval
and sample count to assess stability. A fit without a stable positive-volume
minimum is not an equilibrium property. Pressure scans use `--curve pv` with
different options from the E–V route.

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
| Internal soft mode or singular stiffness | Structure or model is unstable | Check stability first; disabling the ionic correction is not a repair |
| Results drift when the strain step is halved | Relaxation tolerance too loose or finite differences unstable | Tighten `relax:gtol` and recheck one setting at a time |
| Unreasonable EOS fit | Sampling crosses a phase boundary or lacks a positive-volume minimum | Narrow the volume window, add points and confirm a single phase |
| `Energy/at` disagrees with single-point energy | Total and per-atom energies mixed | Convert by atom count before comparing |

## Related

- [Structure relaxation](relaxation.md): source of the equilibrium structure
- [Command reference](../reference/commands.md), [Python Phonopy adapter](../reference/python.md#phonopy-integration)
- [Harmonic references and TI](../advanced/harmonic-ti.md)
