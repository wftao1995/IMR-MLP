# Active learning and candidate selection {#guide_active_learning}

[中文](active-learning_zh.md)

Active learning identifies structures poorly covered by current training and worth
labeling, reducing redundant reference work. A complete cycle is
**train → explore → score/select → reference-label → merge/retrain → validate**.

Out of scope: reference calculations and fitting themselves (see
[training, evaluation and deployment](workflows.md)) and the mathematics of MaxVol
(see [MaxVol theory](../theory/maxvol.md)).

## Offline and online entry points

| Situation | Entry point | Precondition |
| --- | --- | --- |
| Candidate structures already exist; score offline | `maxvol_select` + `maxvol_sample` | An inspectable selection cycle |
| Update an existing active set | `maxvol_extend` | Model and active set are consistent |
| Reduce redundancy using geometric/descriptor diversity | FPS / CUR in `diverse_select` | Representative subsets; scores differ from MaxVol |
| Detect unfamiliar configurations during LAMMPS MD | [`fix mlip/maxvol`](../../LAMMPS/USER-MLIP/maxvol/README.md) | Model and interface verified, with a matching active set |

The offline cycle below uses ordinary MTP. Derived models such as MTP_M or LES/SOG
may need the full-parameter-gradient `pot_efs` builder rather than ordinary MTP's
linear shortcut.

## Required inputs

| File | Content | Source |
| --- | --- | --- |
| `model.mtp` | Trained ordinary MTP model | Training output |
| `train.xyz` | Existing training coverage, used to build the active set | The model's training data |
| `pool.xyz` | Structures to explore, geometry only | MD, perturbations or defect generation |
| `index.map` | Model species order | The model or its training records |
| `select.ini` | Builder settings | See below |

Do not use the final test set to establish selection statistics. Selection does not
create the exploration pool.

## Building the active set

`select.ini` only needs the builder:

```text
builder:potfile model.mtp
```

The remaining defaults use per-atom environments for ordinary MTP and exact
streaming selection:

```sh
imrmlp maxvol_select select.ini --pool train.xyz \
  --output selected.xyz --score-file training-grades.txt
```

Outputs are the selected structures and the active-set state `model.mtp.sel`, which is
not a potential file. The check is the `Selection status:` line and the `virtual_rows`
count printed with it: `WARNING: not converged` means the threshold was not reached
within the budget, so rerun with `full` or `stream/exact`; a nonzero `virtual_rows`
count means directions remain uninitialized, which small or strongly correlated
datasets produce.

## Screening exploration structures

```sh
imrmlp maxvol_sample select.ini --input pool.xyz \
  --state-file model.mtp.sel --output candidates.xyz \
  --threshold-save 2.1 --threshold-break 1000000000
```

The example raises the **offline output** break threshold to collect higher-grade
frames. Values such as `2.1` are starting points to calibrate against the model,
active set and reference errors. The CLI default break is 11: a configuration above it
stops output immediately and that extreme frame is not written either, with
`BREAK: cfg … grade …` in the log — `candidates.xyz` then covers only the part before
the stop, and the remaining structures were never scored. Raise `--threshold-break` or
process the pool in chunks to cover all of it. The LAMMPS fix has its own documented
save/break behavior.

`candidates.xyz` may retain input properties and adds `MV_grade`; neither is a reference
label obtained this round — that requires a separate reference calculation. Grade has
no conversion to E/F error; see [MaxVol theory §3.4](../theory/maxvol.md) for reading it
beside `calc_err`.

### Verification with the bundled example

After building the CLI, run from the repository root:

```sh
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/active-learning/run.py
```

Using only Python's standard library, the script prepares the model, mapping,
settings and two-atom structures in a new `build/active-learning/` directory. Known
coverage spans separations of 3.8–4.0 Å. The exploration pool gives:

| Exploration separation (Å) | Approximate grade | Above save threshold 2.1? |
| --- | --- | --- |
| 3.9 | 0.854 | No |
| 3.5 | 1.488 | No |
| 3.0 | 2.425 | Yes |
| 2.5 | 3.445 | Yes |

Inspect `grades.tsv`, `candidates.xyz` and `sample.log`; expect `Saved 2 / 4`. The score
follows the active set and the structure together rather than the separation alone. This
is a synthetic teaching model, with `Al` illustrating mapping only, and candidates
contain geometry and grades without new reference labels or retraining. Manual commands
and a lower-threshold variant are in the
[complete example](../../examples/active-learning/README.md).

The exercise's `coverage.cfg` corresponds to `train.xyz` above and `pool.cfg` to
`pool.xyz`; keep its CFG filenames when repeating it, without changing extensions.

## Labeling, retraining and state updates

1. Inspect candidate geometry and redundancy, then obtain reference E/F/stress with a method consistent with existing training.
2. Check units, mapping and electronic convergence; merge newly labeled data into the training pool.
3. [Retrain](workflows.md), comparing old and new models on independent validation data and target properties.
4. Build a matching active set for the new model before the next exploration cycle.

With a labeled `train.xyz` and a completed reference calculation for one new
candidate at `reference-001/OUTCAR`, convert and merge:

```sh
imrmlp convstr reference-001/OUTCAR new-reference.xyz
cat train.xyz new-reference.xyz > train-next.xyz
```

Both `.xyz` inputs must be EXTXYZ with reference labels; geometry-only candidates are
not newly labeled training data. Inspect labels in `new-reference.xyz`, then change
`trainset` to `train-next.xyz` in a new training configuration and keep the original
validation set. Check convergence for each reference calculation before combining
multiple results, retaining raw references and each round's data.

Keep each round's model, builder settings, map, state, candidates and labels
together. After model parameters change, an old readable `.sel` with matching
dimensions is not sufficient. `maxvol_extend` can update state but does not replace
retraining and validation; retain copies of important states first.

## Extending to production sampling

LAMMPS `fix mlip/maxvol` requires its separate MPI plugin and a matching model,
builder and state. It scores and saves candidates without changing pair forces.
Validate GPU pairs combined with the CPU scorer as described by the interface.
Loading GPM through the CLI does not activate sampling automatically either.

For large pools, `stream` + `exact` caches rows for whole-pool scans. `online` does not
retain every historical candidate and depends on arrival order; use `stream/exact` when
the selection has to be equivalent to a whole-pool scan. See the
[command reference](../reference/commands.md) for memory and MPI controls.

## Common errors

| Symptom | Cause | Action |
| --- | --- | --- |
| Almost no candidates | Structures, model or mapping differ from what was assumed, or exploration misses the target conditions | Check the loaded structures and mapping first, then thresholds and exploration range |
| Almost everything scores highly | Model and active set inconsistent, or builder scaling differs | Verify all three before raising thresholds |
| Many redundant candidates | The exploration pool itself is redundant | Reduce by mechanism or descriptor diversity, then label in batches |
| Retraining does not improve target properties | Label quality or coverage is insufficient | Compare label quality and coverage instead of training error or grade alone |
| Grade treated as an error | Grade is extrapolation relative to one active set | Evaluate E/F errors against reference labels separately |
| An old `.sel` state is reused | Model parameters changed | Rebuild the active set for the new model |

## Related

- [Training, evaluation and deployment](workflows.md): retraining and validation
- [MaxVol theory](../theory/maxvol.md), [command reference](../reference/commands.md)
- [Self-contained MaxVol selection exercise](../../examples/active-learning/README.md)
- [LAMMPS `fix mlip/maxvol`](../../LAMMPS/USER-MLIP/maxvol/README.md)
