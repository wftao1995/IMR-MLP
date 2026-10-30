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
species 0. A successful run after swapping `index.map` does not establish that
the mapping is correct.

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
