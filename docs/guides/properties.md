# Crystal Properties: Phonons, Elasticity and Equation of State {#guide_properties}

[中文](properties_zh.md)

Use this page after obtaining a model and completing [structure relaxation](relaxation.md).
Small single-point errors do not establish accurate force constants, elasticity
or volume response; validate each target property. Examples need your own
`model.mtp`, `relaxed.vasp` and `index.map` in separate calculation directories.

**No matching inputs yet?** The [calculation walkthrough](../../examples/basic-calculations/README.md)
generates and relaxes a teaching LJ crystal, then explicitly copies `bulk.vasp`
to each property directory as `relaxed.vasp`. Use its LJ settings there; this
page's main example uses your ordinary MTP. For real tasks, export the final
[relaxation](relaxation.md) result to POSCAR. EOS/zero-pressure elasticity also
require the corresponding cell optimization.
For basis generation and fitting followed by property checks with the same
model, use the [MTP walkthrough](../../examples/mtp-workflow/README.md).

## Choose the quantity

| Goal | Tool | Check first |
| --- | --- | --- |
| Second-order atomic response and harmonic force constants | `calc_fc` | Equilibrium structure, supercell and Hessian backend |
| Phonon dispersion, DOS and vibrational thermodynamics | IMR-MLP forces/force constants with Phonopy postprocessing | Atom order, supercell definition and masses |
| Stiffness and strain response | `calc_elastic` | Equilibrium stress, ionic-relaxation convention and strain amplitude |
| Equilibrium volume, bulk modulus and E–V curve | `calc_eos` | Stable volume interval in one phase and relaxation settings |

## 1. Export force constants

Build a 2×2×2 supercell from the input cell and export full force constants:

```sh
imrmlp calc_fc --potfile model.mtp \
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
```

Inspect `FORCE_CONSTANTS`, `supercell_created.xsf` and `sup_debug_efs.xyz`.
Blocks use eV/Å² and follow the **actual exported supercell** order.
`2 2 2` is a starting point; increase the supercell to check convergence.
The default analytic backend errors for unsupported models. `--hessian-method auto`
uses analytic derivatives when available and force finite differences otherwise.
You can select `force_fd` explicitly and converge `--hessian-nudge-pos`.

`--lowest-modes N` is a separate matrix-free lowest-Gamma-mode diagnostic:
it exports eigenmodes, not full `FORCE_CONSTANTS`, and does not replace
whole-Brillouin-zone phonon dispersion. More backends and outputs are defined
in the [command reference](../reference/commands.md).

## 2. Use Phonopy for dispersion or DOS

As an alternative to the previous section, let Phonopy generate displacements
and IMR-MLP evaluate forces. Copy the same equilibrium structure, model and map
into a fresh directory to avoid mixing in another `FORCE_CONSTANTS`.
The following uses Phonopy 4 command separation:

```sh
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
imrmlp phonopyfc --potfile model.mtp --disp phonopy_disp.yaml \
  --index-file index.map
```

This ordinary-MTP example requires a separate Phonopy installation.
`phonopy-init` creates `phonopy_disp.yaml`; `phonopyfc` writes `FORCE_SETS`.
Then calculate an illustrative DOS mesh:

```sh
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml
```

Check `total_dos.dat` and the log, then vary the mesh or choose a material-specific
band path. Older Phonopy 3 tutorials use `phonopy -d`. Phonopy 4 moves setup to
`phonopy-init`; postprocessing reads YAML without repeating `-c` / `--dim`.
See the [official migration guide](https://phonopy.github.io/phonopy/migration-v4.html).
 `phonopyfc` uses `--index-file`, rather than the common `--index`.
If `settings.ini` exists in the directory, check its `pot:` settings against
the current model. Meshes, band paths and plotting are covered by Phonopy's
[command options](https://phonopy.github.io/phonopy/command-options.html) and
[setting tags](https://phonopy.github.io/phonopy/setting-tags.html).

The [Python Phonopy adapter](../reference/python.md#phonopy-integration) can
instead evaluate displaced-supercell forces in memory. For direct import of
analytic `FORCE_CONSTANTS`, use Phonopy's own supercell as `calc_fc` input,
without expanding it again. Both atom axes must match `phonon.supercell`
exactly; matching atom counts is insufficient.

Imaginary modes can reflect physical instability, incomplete relaxation,
displacement/supercell errors, or model/unit/order problems. Check equilibrium
forces, acoustic modes and numerical convergence before physical interpretation.
Phonon force constants differ from finite-temperature effective force constants;
see [harmonic references and TI](../advanced/harmonic-ti.md) for the latter.

## 3. Calculate elastic constants

For ordinary MTP, start with the default automatic backend:

```sh
imrmlp calc_elastic --potfile model.mtp --input relaxed.vasp \
  --index index.map --elastic-output elastic.out > elastic.log 2>&1
```

The default `auto` chooses a supported method. Use `--method analytical` when needed and
check atomic/cell-Hessian support. For a finite-difference comparison, create
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
`--relax-correction` controls the related correction. Read stiffness/modulus
units from output headers. At nonzero pressure, distinguish material stiffness
from the Born/Wallace tangent used for mechanical stability. For internal soft-mode
or singular-stiffness errors, check structural/model stability first. Disabling
ionic correction changes the physical quantity and is not a general repair. See the
[elastic command reference](../reference/commands.md) for complete definitions.

### Small exercise: halve the strain amplitude

Keep `elastic-numerical.out` from the `0.005` (0.5%) calculation above and
repeat with the same input and internal-coordinate relaxation settings:

```sh
imrmlp calc_elastic --potfile model.mtp \
  --input relaxed.vasp --index index.map --method numerical \
  --settings elastic.ini --strain-magnitude 0.0025 \
  --elastic-output elastic-half-strain.out > elastic-half-strain.log 2>&1
```

For example, `C11` values of `162.4` and `162.3 GPa` differ by `0.1 GPa`,
about 0.06%. These illustrate comparison, not expected material values or a
universal acceptance criterion. Also compare other independent components,
moduli and ionic convergence at strained points. Substantial drift on halving
the step calls for checking relaxation tolerances and finite-difference
stability; smaller steps are not always more accurate. Apply the same approach
to phonon supercells or DOS meshes: change one numerical setting at a time
and retain separate results.

## 4. Calculate an equation of state

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

The default window spans roughly ±5% around the reference volume, producing curve data and
EOS parameters. Check that points remain in one phase, bracket an energy
minimum and give sensible fitted volume/bulk modulus. Vary the volume interval
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

These apply to the structure-scanning E–V command on this page. When fitting
an external curve, volume/energy normalization follows its input data and is
not automatically per atom. Do not directly compare `calc_efs` total energy
with `Energy/at`.

## Record and compare results

Retain the equilibrium structure, model, map, settings and logs, recording
supercell, displacement/strain amplitude, pressure and ionic-relaxation
convention. Converge numerics before comparison with reference properties.
No C++ API reading is needed; specialized higher derivatives, HVP and
finite-temperature workflows are in [advanced usage](../advanced/index.md).
