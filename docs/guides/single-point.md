# Single-point calculation and error evaluation {#guide_single_point}

[中文](single-point_zh.md)

A single-point calculation predicts total energy, per-atom forces and stress at a
fixed geometry without optimizing the structure. When labeled data is available,
`calc_err` quantifies the prediction error for the same geometry.

Out of scope: moving atoms or the cell (see [structure relaxation](relaxation.md))
and energies along a reaction path (see [NEB](neb.md)).

## Required inputs

| File | Content | Source |
| --- | --- | --- |
| `model.mtp` | Trained ordinary MTP model | Model provider or training output |
| `POSCAR` | Structure to predict, including element symbols | Structure preparation or reference calculations |
| `index.map` | Model species order | The model or its training records |
| `reference.xyz` (optional) | Structures with E/F/stress labels | Reference calculations; needed only by `calc_err` |

Naming and path rules are in [input files and settings conventions](conventions.md).
The prediction command does not create these inputs.

## Commands

### Predict energy, forces and stress

```sh
imrmlp calc_efs POSCAR predicted.xyz --potfile model.mtp
```

The species map defaults to `index.map`, so no flag is needed while the file
keeps that name; the potential type defaults to ordinary MTP and formats are
inferred from filenames. Specify explicit options only for another potential
type, a differently named species map, or a filename whose format cannot be
recognized. Options differ between commands; the full list is in the `calc_efs`
section of the [command reference](../reference/commands.md).

### Evaluate errors

```sh
imrmlp calc_err --input reference.xyz --potfile model.mtp \
  --output-summary errors.txt
```

`reference.xyz` must carry reference labels. A POSCAR alone supports prediction,
but there is no reference value to compare with, and validation errors cannot be
produced from it.

## Output and criteria

`predicted.xyz` is extended XYZ text that preserves the input geometry and adds
predicted properties; inspect it with a text editor or an EXTXYZ-compatible tool.
The second line's `energy` is the **total** energy of that structure.
`Properties` declares the columns; `forces:R:3` means three force components per
atom.

Check: the program exits normally, atom count and species are correct, energies
and forces are finite, and deviations from reference results for the same
geometry are acceptable. `calc_err` reports that deviation; without reference
labels only finiteness can be confirmed.

## Verify with the quickstart example

Without your own inputs, the [self-contained quickstart](../../examples/quickstart/README.md)
creates the default directory `build/quickstart/`. From the repository root:

```sh
cd build/quickstart
imrmlp calc_efs valid-input.cfg my-prediction.xyz --potfile fitted.mtp
head -n 5 my-prediction.xyz
```

`valid-input.cfg` contains only geometry; `valid.cfg` carries reference labels for
`calc_err`, which you can compare with `errors.txt`. The file holds four frames of
three atoms, not one twelve-atom structure. `Al` in the example only illustrates
species mapping; its synthetic model is not a usable aluminium potential.

If you used a custom `--work`, substitute its actual output path above; run
`cd ../..` to return to the repository root afterwards.

## Common errors

| Symptom | Cause | Action |
| --- | --- | --- |
| Energy or forces are clearly implausible | `index.map` order does not match the model | Check each line against the model's species order; a run completing says nothing about this |
| Input files not found | Relative paths resolve from the working directory | Run inside the task directory or use absolute paths |
| Output overwrites the previous run | Input and output share a name | Use different output filenames |
| An option is ignored or rejected | `calc_efs` options copied to another command | Check each command in the [command reference](../reference/commands.md) |
| `calc_err` produces no errors | The input carries no reference labels | Use data with E/F/stress labels |

For other symptoms see [troubleshooting](troubleshooting.md).

## Related

- [Input files and settings conventions](conventions.md)
- [Structure relaxation](relaxation.md), [NEB](neb.md), [crystal properties](properties.md)
- [Command reference](../reference/commands.md), [cheat sheet](../reference/cheatsheet.md)
