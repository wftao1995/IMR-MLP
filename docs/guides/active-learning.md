# Active Learning and Candidate Selection {#guide_active_learning}

[中文](active-learning_zh.md)

Active learning identifies structures poorly covered by current training and
worth labeling, reducing redundant reference work. A complete cycle is
**train → explore → score/select → reference-label → merge/retrain → validate**.
Neither MaxVol nor FPS/CUR supplies DFT labels automatically, and selecting
structures alone does not improve the model.

## Choose offline or online sampling

| Situation | Entry point | When useful |
| --- | --- | --- |
| Candidate structures already exist; score offline | `maxvol_select` + `maxvol_sample` | First inspectable selection cycle |
| Update an existing active set | `maxvol_extend` | Once model/state consistency is understood |
| Reduce redundancy using geometric/descriptor diversity | FPS / CUR in `diverse_select` | Representative subsets; scores differ from MaxVol |
| Detect unfamiliar configurations during LAMMPS MD | [`fix mlip/maxvol`](../../LAMMPS/USER-MLIP/maxvol/README.md) | Model/interface verified and matching active set prepared |

This page uses ordinary MTP for an offline cycle. Derived models such as
MTP_M or LES/SOG may need the full-parameter-gradient `pot_efs` builder rather
than ordinary MTP's linear shortcut.

## Start with a small example requiring no external data

After building the CLI, run from the repository root:

```sh
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/active-learning/run.py
```

Using only Python's standard library, the script prepares the model, mapping,
settings and two-atom structures in a new `build/active-learning/` directory.
Known coverage spans separations of 3.8–4.0 Å. The exploration pool gives:

| Exploration separation (Å) | Approximate grade | Above save threshold 2.1? |
| --- | --- | --- |
| 3.9 | 0.854 | No |
| 3.5 | 1.488 | No |
| 3.0 | 2.425 | Yes |
| 2.5 | 3.445 | Yes |

Inspect `grades.tsv`, `candidates.xyz` and `sample.log`; expect `Saved 2 / 4`.
Different exploration structures receive different scores against the same
active set; shorter distances do not imply extrapolation in general. This is
a synthetic teaching model, with `Al` illustrating mapping only. Candidates
contain geometry and grades, without new reference labels or retraining.
See the [complete example](../../examples/active-learning/README.md) for manual
commands, file meanings and a lower-threshold exercise. The following sections
apply the same process to your own model and data.

The exercise's `coverage.cfg` corresponds to training coverage `train.xyz`
below, and `pool.cfg` to `pool.xyz`. Keep its CFG filenames when repeating it;
do not change extensions. For real work, use the model's training structures
for `train.xyz`; MD, perturbations or defect generation can supply `pool.xyz`.
Selection needs geometry, but does not create the exploration pool for you.

## 1. Build an active set from existing training coverage

Prepare `model.mtp`, training structures `train.xyz`, exploration structures
`pool.xyz` and `index.map`. Do not use the final test set to establish selection
statistics. Save `select.ini`:

```text
builder:potfile model.mtp
```

The remaining defaults use per-atom environments for ordinary MTP and exact
streaming selection.

```sh
imrmlp maxvol_select select.ini --pool train.xyz --index index.map \
  --output selected.xyz --score-file training-grades.txt
```

Outputs include selected structures and the active-set state `model.mtp.sel`;
the state is not a new potential. Check selection completion, remaining virtual
directions and unconverged-state messages. Saving after an exchange/scan budget
is exhausted does not establish convergence. Small or correlated datasets can
also leave parameter directions rank-deficient.

## 2. Screen exploration structures with the same configuration

```sh
imrmlp maxvol_sample select.ini --input pool.xyz --index index.map \
  --state-file model.mtp.sel --output candidates.xyz \
  --threshold-save 2.1 --threshold-break 1000000000
```

The example raises the **offline output** break threshold to collect higher-grade
frames. Values such as `2.1` are starting points to calibrate against model,
active set and reference errors. Check saved counts and `BREAK` messages.
The CLI default break is 11: encountering a configuration above it stops output
before writing that extreme frame. Do not mistake partial output for the entire
candidate pool. The LAMMPS fix has its own documented save/break behavior.

`candidates.xyz` may retain input properties and add `MV_grade`. Predictions
or existing MD properties are not newly obtained reference labels. Grade
measures extrapolation relative to a specific active set, not an eV/Å error;
a low grade alone does not establish reliability.

## 3. Label, retrain and rebuild matching state

1. Inspect candidate geometry and redundancy, then obtain reference E/F/stress using a method consistent with existing training.
2. Check units, mapping and electronic convergence; merge newly labeled data into the training pool.
3. [Retrain](workflows.md), comparing old/new models on independent validation data and target properties.
4. Build a matching active set for the new model before the next exploration cycle.

For example, with a labeled `train.xyz` and a completed reference calculation
for one new candidate at `reference-001/OUTCAR`, convert and merge:

```sh
imrmlp convstr reference-001/OUTCAR new-reference.xyz --index index.map
cat train.xyz new-reference.xyz > train-next.xyz
```

Both `.xyz` inputs must be EXTXYZ with reference labels; geometry-only
candidates are not newly labeled training data. Inspect labels in
`new-reference.xyz`, then change `trainset` to `train-next.xyz` in a new
training configuration, keep the original validation set and follow the
[training guide](workflows.md). Check convergence for each reference calculation
before combining multiple results, retaining raw references and each round's data.

Keep each round's model, builder settings, map, state, candidates and labels
together. After model parameters change, an old readable `.sel` with matching
dimensions is not sufficient. `maxvol_extend` can update state but does not
replace retraining/validation; retain copies of important states first.

## 4. Extend to production sampling

After the offline cycle works, consider online runs. LAMMPS `fix mlip/maxvol`
requires its separate MPI plugin and matching model, builder and state.
It scores/saves candidates without changing pair forces. Validate GPU pairs
combined with the CPU scorer as described by the interface. Loading GPM through
the CLI does not activate sampling automatically either.

For large pools, `stream` + `exact` caches rows for whole-pool scans. `online`
does not retain every historical candidate and depends on arrival order;
it is not the same exact selection. See the [command reference](../reference/commands.md)
for memory/MPI controls and [MaxVol theory](../theory/maxvol.md) for scoring.

## When should the workflow change?

- Almost no candidates: check loaded structures, model and mapping, then thresholds and exploration coverage.
- Almost everything scores highly: check model/state consistency, coverage and builder scaling before raising thresholds.
- Many redundant candidates: reduce by mechanism or descriptor diversity, then label in batches.
- Retraining does not improve target properties: inspect label quality and coverage rather than judging only training errors or grades.
