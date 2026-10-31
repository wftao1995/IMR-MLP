# Input files and settings conventions {#guide_conventions}

[中文](conventions_zh.md)

Every task shares one set of input conventions: a dedicated directory, a potential
model, a structure file, a species map, and a few `key value` settings. This page
lists only those shared conventions; the calculation itself is described on each
task page.

Out of scope: the syntax of `.mtp` model files and the call signature of the
Python interface.

## Task directory

Use one directory per calculation, holding inputs, settings, logs and outputs:

```text
my-calculation/
    model.mtp      Trained ordinary MTP model
    POSCAR         Structure including element symbols
    index.map      Model species order
    relax.ini      Settings for this task (named after the task)
```

`model.mtp`, `POSCAR` and similar names are placeholders for your files.
`model.mtp` comes from a provider or training output such as `fitted.mtp` (or
update the settings to its actual name), `POSCAR` from structure preparation or
reference calculations, and `index.map` from the model's species-order records.
No command creates these files.

**Relative paths resolve from the command's working directory, not the settings
file's directory.** Use different input and output filenames to avoid overwriting.

## Model, structure and species map

`index.map` contains one model element per line. For model species 0 = Al and
species 1 = Ni:

```text
Al
Ni
```

Recover this order from the model or its training records. POSCAR element blocks
may be ordered differently because element names are mapped on reading. Numeric
types in CFG already represent model species indices: atomic number 13 is not
species 0. The reader builds name → index row by row without validating the
order, so swapping two lines still runs — with different energies and forces.
The check is one `calc_efs` run on a known structure, then the same run after
swapping two lines: changed E/F prove the order is in use, and each row is then
checked against the model by element name.

`index.map` is the default for both the species-map option and the `index`
settings key in the commands that read a species map; `model_convert` is the
exception, with no default. A file with that name in the working directory is
therefore read without any flag:

```sh
imrmlp calc_efs POSCAR predicted.xyz --potfile model.mtp
```

Pass `--index` only when the file has another name or lives elsewhere.
Forgetting it is quiet rather than fatal: with no `index.map` present, the
command only warns (`ERROR: indexfile not found`, or a similar message) and
continues with whatever species order the structure file supplies, so the first
sign is usually implausible energies rather than an error.

## Settings file syntax

Separate keys and values with whitespace, **without equals signs**:

```text
input POSCAR
output relaxed.xyz
index index.map
pot:potfile model.mtp
```

`pot:` selects potential settings, `relax:` relaxation settings and `neb:` path
settings. Task pages use explicit prefixes to avoid section and blank-line
ambiguities. Duplicate keys keep their first value; replace the existing line
when editing.

`.mtp` model files have their own syntax and should not be edited as `.ini` files.

## Block layout

A deck of thirty keys is read while editing, not while running. Group related keys
and leave one blank line between groups:

```text
# ===== Input/Output =====
index               index.map
mepout              mep.xyz

# ===== Potential =====
pot:potfile         model.mtp

# ===== Images =====
nimgs               7
image:0             initial.vasp
image:6             final.vasp

# ===== Optimizer =====
driver              fire
max_move            0.05
```

The templates in `example-inputs` and the samples on these pages use one order:
input and output, potential, parallelism, images or frames, optimizer, convergence,
then the blocks specific to the task and `Advanced` last. Headers name each block;
a file of a handful of keys needs none of this.

The parser reads keys by name, so the block order does not change the result. A
stable layout only makes two decks comparable and a change easy to locate.

## Units and model types

| Quantity | Common convention |
| --- | --- |
| Positions and cell | Å |
| Total energy | eV; divide by atom count for per-atom energy |
| Forces | eV/Å |
| Native extensive stress / virial | eV; stress conversion needs volume and the appropriate sign convention |
| External pressure | `pressure` settings use GPa; positive means compression |
| Model type | Ordinary `mtp` by default; MTP_M, EAM, LES and others need their own type and settings |

Model options differ between CLI commands; do not copy `calc_efs` options
unchanged into another command. See [potential types](../reference/potential-types.md)
for families and legacy formats, and the [configuration reference](../reference/configuration.md)
for the full key list and units.

## Related

- [Installation and build](installation.md)
- [Single-point calculation and error evaluation](single-point.md)
- [Configuration reference](../reference/configuration.md)
