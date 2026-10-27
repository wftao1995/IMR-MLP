# A self-contained MaxVol selection exercise

[中文](README_zh.md) · [Active-learning guide](../../docs/guides/active-learning.md)

This small example starts with a supplied synthetic MTP and generates two-atom
structures. Three known geometries have separations of 3.8, 3.9 and 4.0 Å;
the exploration pool contains 3.9, 3.5, 3.0 and 2.5 Å. It demonstrates how
an active set represents existing coverage and which new structures pass a
selection threshold. The `Al` label is only a species mapping, not a material model.

## Run

After building the CPU CLI, run from the repository root:

```sh
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/active-learning/run.py
```

Only Python's standard library and the CLI are required. No Python extension,
ASE, LAMMPS or reference dataset is needed. The default output is
`build/active-learning/`; a nonempty directory is never overwritten. Use
`--work build/active-learning-2` for another run, or `--imrmlp /absolute/path/to/imrmlp`
to select a different executable and configure its matching runtime libraries.

## Inspect the result

The script prints the following approximate scores and writes `grades.tsv`:

| Pool frame (zero-based) | Separation (Å) | Grade | Saved at grade > 2.1? |
| --- | --- | --- | --- |
| 0 | 3.9 | 0.854 | No |
| 1 | 3.5 | 1.488 | No |
| 2 | 3.0 | 2.425 | Yes |
| 3 | 2.5 | 3.445 | Yes |

Inspect `select.log` for `converged` and `virtual_rows=0`, then `sample.log`
for `Saved 2 / 4`. This deliberately tiny model has only one active builder
column; the selected structure count is not a rule for larger MTPs.

| File | Meaning |
| --- | --- |
| `coverage.cfg`, `pool.cfg` | Geometry-only known coverage and exploration inputs |
| `model.mtp`, `index.map`, `select.ini` | Model, species order and scoring settings |
| `model.mtp.sel`, `selected.xyz` | Active-set state and structures representing its selected rows |
| `graded-pool.xyz`, `grades.tsv` | All four scored frames and a readable summary |
| `candidates.xyz` | Two frames to consider for reference labeling; **no reference E/F/stress labels** |

## Repeat a step yourself

In the generated directory, with `imrmlp` on `PATH`, the core commands are:

```sh
imrmlp maxvol_select select.ini --pool coverage.cfg --index index.map \
  --output selected.xyz --score-file coverage-grades.txt
imrmlp maxvol_sample select.ini --input pool.cfg --index index.map \
  --state-file model.mtp.sel --output candidates.xyz \
  --threshold-save 2.1 --threshold-break 1000000000
```

Try `--threshold-save 1.2 --output candidates-1.2.xyz`: three of these four
frames should be saved. This changes the labeling budget, not model accuracy.
The high break threshold is for collecting this finite offline pool; it is
not an online MD safety setting. Keep the model and `.sel` paired while changing
only the output threshold.

For real work, inspect candidate geometries, obtain new reference labels with
the same calculation method as the existing data, retrain, validate and rebuild
matching state as described in the [guide](../../docs/guides/active-learning.md).
This exercise stops at selection. A grade is relative to the chosen model,
builder and active set, not a distance cutoff or a physical error estimate.
