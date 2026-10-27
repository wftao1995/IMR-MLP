# Getting Started: Choose a Task and Prepare Inputs {#guide_getting_started}

[中文](getting-started_zh.md)

You do not need to learn C++ classes or every option before using IMR-MLP.
First decide whether you have a suitable potential, then follow one route below.
Commands run in a Linux terminal with `imrmlp` on `PATH`; start with
[installation](installation.md) if it is not available yet.

## Where should I start?

| Your situation | Next step | What you obtain |
| --- | --- | --- |
| No model; check that the program works | [Self-contained quickstart](../../examples/quickstart/README.md) | Automatically generated small model, data, training logs and validation results |
| A model suitable for your material is available | Single-point calculation below, then choose a task | Structures with predicted energy, forces and stress |
| Reference calculations are available; train a model | [Training, evaluation and deployment](workflows.md) | Independently validated model and reproducible training records |
| Run LAMMPS MD with an existing model | [LAMMPS interface](../../LAMMPS/README.md) | Installed interface, pair-style selection and `run 0` comparison |
| Build a Python / ASE workflow | [Python installation and examples](../../README-python.md) | Importable package and ASE calculator |

`Al` in the self-contained example illustrates species mapping; its synthetic
model is not a real Al material potential. Real work needs a model covering
the target elements, structures and conditions, obtained or trained separately.
A model file is not a universal parameter set for all materials.

## Follow this sequence when you have no inputs

1. [Install](installation.md), then run the [training quickstart](../../examples/quickstart/README.md) to create `build/quickstart/`.
2. Use this page's quickstart practice commands to inspect single-point output; do not regenerate a completed step 1.
3. Use the [LJ calculation walkthrough](../../examples/basic-calculations/README.md) for direct calculations, or the [MTP walkthrough](../../examples/mtp-workflow/README.md) for basis generation, training and validation before relaxation, NEB and properties.
4. Use the [active-learning exercise](../../examples/active-learning/README.md) for grades, thresholds and candidate output before reference labeling/retraining.

Each exercise keeps its own model and directory. The ordinary MTP filenames
below are conventions for real tasks; do not rename one exercise's model or
outputs just to fill the next page's placeholders.

## Prepare three files for your calculation

Create a separate directory for each task, put these files inside, then enter
that directory before running commands:

```text
my-calculation/
    model.mtp      Trained ordinary MTP model
    POSCAR         Structure including element symbols
    index.map      Model species order
```

Filenames here and in the task guides are placeholders for your files.
`model.mtp` comes from a provider or training output such as `fitted.mtp` (or update
the settings to its actual name), `POSCAR` from structure preparation/reference
calculations, and `index.map` from the model's species-order records. The
single-point command does not create these inputs. Keep
settings such as `relax.ini` or `neb.ini` and logs in the task directory too.
**Relative paths resolve from the command's working directory, not the settings
file's directory.** Use different input and output filenames.

`index.map` contains one model element per line. For model species 0 = Al and
species 1 = Ni:

```text
Al
Ni
```

Recover this order from the model or training records. POSCAR element blocks
may be ordered differently because element names are mapped on reading.
Numeric types in CFG already represent model species indices: atomic number
13 is not species 0. A successful run after swapping `index.map` does not
establish that the mapping is correct.

## Your first single-point calculation

Run in the task directory above:

```sh
imrmlp calc_efs POSCAR predicted.xyz --potfile model.mtp --index index.map
```

This page and the basic guides use the default ordinary MTP type and infer
formats from filenames. Specify these options only for another potential type
or a filename whose format cannot be recognized.

A single-point calculation preserves the input geometry and predicts total
energy, per-atom forces and stress. `predicted.xyz` is extended XYZ text with
geometry and predictions; inspect it with a text editor or EXTXYZ-compatible
tool. Check successful termination, atom count, species and finite energies
and forces, then compare with reference results for the same geometry.
Creating an output file alone does not establish material accuracy.

For data that already contains reference E/F/stress labels, also evaluate errors:

```sh
imrmlp calc_err --input reference.xyz --potfile model.mtp \
  --index index.map --output-summary errors.txt
```

An unlabeled POSCAR supports prediction, but does not supply validation errors.

### Practice with the quickstart when you have no inputs

First run the [self-contained quickstart](../../examples/quickstart/README.md)
with its default `build/quickstart/` output. From the repository root:

Reuse that directory if installation already created it. If you used a custom
`--work`, substitute its actual path below.

```sh
cd build/quickstart
imrmlp calc_efs valid-input.cfg my-prediction.xyz --potfile fitted.mtp --index index.map
head -n 5 my-prediction.xyz
```

The first line of each frame, `3`, is its atom count. The second line's `energy`
is total energy for that three-atom system. `Properties` declares the columns;
`forces:R:3` means three force components per atom. There are four frames,
not one twelve-atom structure. `valid-input.cfg` contains only geometry;
`valid.cfg` has reference labels for `calc_err`, which you can compare with
`errors.txt`. Run `cd ../..` to return to the repository root before entering
another example or your own task directory.

## How do settings files work?

Separate keys and values with whitespace, **without equals signs**. For example:

```text
input POSCAR
output relaxed.xyz
index index.map
pot:potfile model.mtp
```

This illustrates syntax; the [relaxation guide](relaxation.md) provides complete
settings. `pot:` selects potential settings, `relax:` relaxation settings and
`neb:` path settings. Task guides use explicit prefixes to avoid section and
blank-line ambiguities. Duplicate keys keep their first value; replace the
existing line when editing. `.mtp` model files have their own syntax and
should not be edited as `.ini` files.

## Units, model types and interpretation

| Quantity | Common convention |
| --- | --- |
| Positions and cell | Å |
| Total energy | eV; divide by atom count for per-atom energy |
| Forces | eV/Å |
| Native extensive stress / virial | eV; stress conversion needs volume and the appropriate sign convention |
| External pressure | `pressure` settings use GPa; positive means compression |
| Model type | This page uses ordinary `mtp`; MTP_M, EAM, LES and others need their own type and settings |

Model options differ between CLI commands; do not copy `calc_efs` options
unchanged into another command. See [potential types](../reference/potential-types.md)
for families and legacy formats, and the [configuration reference](../reference/configuration.md)
for units and formats.

## Continue by task

| Goal | Guide |
| --- | --- |
| Optimize atoms at fixed cell, or optimize atoms and cell | [Structure relaxation](relaxation.md) |
| Known initial/final states; calculate a migration or reaction barrier | [NEB minimum-energy paths](neb.md) |
| Phonons, elastic constants or equation of state | [Crystal properties](properties.md) |
| Find unfamiliar structures, add reference labels and retrain | [Active learning and candidate selection](active-learning.md) |
| Look up an option | [Command reference](../reference/commands.md), [cheat sheet](../reference/cheatsheet.md) |

For errors, use [troubleshooting](troubleshooting.md). Retain inputs, settings,
model, species map, logs and `imrmlp --version` output for reproducibility.
