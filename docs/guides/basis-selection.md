# Generating, optimizing and selecting an MTP basis {#guide_basis_selection}

[中文](basis-selection_zh.md)

The basis bounds the model's functional form: `mtmbasis` builds the candidate scalar
pool and the contraction layout, `trainmtp` turns a template into a fitted model,
`fastfit` drops scalars by fit quality, and `calc_efs`/`calc_err` check the result.
This applies to ordinary MTP with a linear head.

Out of scope: LES/SOG, MTP-M and separate neural-network heads (`fastfit` refits the
ordinary linear part only), and data or label problems (see
[training, evaluation and deployment](workflows.md)).

## Required inputs

| File | Content | Source |
| --- | --- | --- |
| `index.map` | Species order consistent with the training data | The model or its training records |
| `train.xyz`, `valid.xyz` | Training and validation sets | Split produced by the [training workflow](workflows.md) |
| `test.xyz` | Independent final test set, not used to compare candidates | Same as above |
| `template.mtp` | Basis candidate to be fitted | `mtmbasis` output |

Parameters are fitted on the training set, candidates are compared on the validation
set, and the final test set is used only for verification.

## Deciding whether the basis is the problem

| Observation | What to change |
| --- | --- |
| Validation error stops improving with more data, with outliers on particular bond angles or coordinations | The pool may lack the needed invariants: raise `max_lev` or `max_nv` and regenerate |
| The model fits well but evaluation is too expensive | Cut scalars with `fastfit`; radial functions stay unchanged |
| Two bases must be shown to give the same predictions, or reused after a layout change | Compare with `mtpbasis`, not by file length or moment count |
| Data are missing, labels are inconsistent, or radial functions were never optimized | Not a basis problem; return to the [training workflow](workflows.md) |
| LES/SOG, MTP-M or a separate neural-network head | Outside this page |

Changing the basis normally invalidates the fitted model, so rule out data and radial
problems first. A smaller basis is not automatically better: fewer tensors from
`mtmbasis` do not imply lower validation error, and fewer scalars from `fastfit` do
not imply faster evaluation.

## Generating the candidate pool

Create `mtmbasis.ini` in its own working directory. These keys are top-level, not
under `[pot]` or `[fit]`; the small basis below runs quickly rather than reaching
production accuracy:

```ini
max_lev 8
max_dim 3
max_mu 2
max_nv 2
max_poly 4
mu_wei 2
nv_wei 1
dim_wei 2
mode 3
maxiter 20
w1 1
w2 4
```

Generation and the contraction search are separate runs, so an expensive pool can be
built once and searched repeatedly:

```sh
imrmlp mtmbasis mtmbasis.ini --generate-only true
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --potfile basis.mtp
```

Boolean options of `mtmbasis` take an explicit value: `--generate-only` alone reports
a missing argument, so the form is `--generate-only true`. The search writes one file
per accepted candidate, named `<prefix>_<iteration>`; `--potfile basis.mtp` gives
`basis.mtp_0`, and the default prefix gives `potential.mtp_<iteration>`.
`conns_exp.log` and the `*_mapping_*.log` files trace which expressions and
contraction paths each candidate uses.

The search minimises `w1 * moments_count + w2 * rules_count`, a structural count of
intermediate moments and accumulation rules. It does not read energies, forces or
stresses, so it cannot rank candidates by accuracy.

Large pools are usually sampled before the search:

```sh
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --sample true --sample-rate 0.5 --generate-only true
```

`--sample` writes `sampled_alphas.dat` and keeps running; the additional
`--generate-only true` saves the sampled pool as `alphapool.dat` and exits, which
overwrites the pool in the current directory, so run sampling in a copy. The pool
sampler uses `random_device` without a seed option; keep the sampled file itself
rather than the sampling rate alone.

## Training the template

A generated file is a topology, not a potential: one species, `RBChebyshev` radial
functions, a 2–5 Å interval, eight radial basis functions, and no trained
coefficients. Set the species count and order, the distance range and the radial
basis size on the untrained file; `radial_funcs_count` follows the mu indices of the
basis and cannot be changed freely. A trained model cannot gain species or radial
dimensions from header edits. An already-searched topology saves this step; take
one from MTM-new as described in [template selection](workflows.md).

Copy the chosen candidate to `template.mtp` and fit it with the same data used later
for selection:

```ini
trainset train.xyz
validset valid.xyz
trainset-format extxyz
validset-format extxyz
potfile template.mtp
index index.map
fit:energy_weight 1
fit:force_weight 0.1
fit:stress_weight 0.001
fit:weight_L2 1e-8
```

```sh
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp full.mtp
```

Add radial optimization, for example `--do-lbfgs` or `trainnlpot --do-varpro`, when
accuracy matters. This step fixes the radial functions: `fastfit` freezes them and
refits only the linear part, so radial quality decides the ceiling that scalar
selection works within. Select an optimizer or `--linear-fit` explicitly; a bare
`trainmtp train.ini` runs no nonlinear optimization.

## Baseline and scalar reduction

`fastfit` reads a settings file with top-level keys as well:

```ini
potfile full.mtp
trainset train.xyz
validset valid.xyz
format extxyz
index index.map
weight-energy 1
weight-force 0.1
weight-stress 0.001
weight-l2 1e-8
energy-power 1
force-power 0
stress-power 1
seed 42
```

`format` applies to both sets and defaults to `mlpcfg`, so specify `extxyz`
explicitly when the data are extxyz. The model must carry the same species order as
the data. Keep the final test set out of this file.

The shortest baseline → selection → export sequence is:

```sh
imrmlp fastfit fastfit.ini --mode 1 --num_threads 1
imrmlp fastfit fastfit.ini --mode 0 --num_threads 1
imrmlp fastfit fastfit.ini --mode 4 --discard-rate 0.25 --num_threads 1
imrmlp fastfit fastfit.ini --mode 2 --num_threads 1
```

| Step | What it produces |
| --- | --- |
| `--mode 1` | `__Trained__.mtp`: the full-basis weighted refit that serves as the error baseline |
| `--mode 0` | `EIGENSYS`, `EIGVEC` and `EIGVAL` for the scalar normal matrix |
| `--mode 4` | `selected_ones.dat` and `removed.dat`, by CUR importance score |
| `--mode 2` | `__REDUCED__.mtp`, refitted and compressed from that selection |

Mode 4 needs the eigensystem from mode 0 in the same directory, so it cannot follow
mode 1 alone. `--discard-rate` is the fraction to discard, and `--addback` retains
scalars used along evaluation paths, so the final count can differ from that
fraction. Boolean options of `fastfit` are bare flags, unlike those of `mtmbasis`.

To refine a selection instead of cutting once, run `--mode 5` with `--search-steps`
and `--method greedy`, or an independent `--mode 6` with `--population` and
`--generations`. Both accept `selection-objective error-cost` in the settings file
when speed matters under a validation-error ceiling; see
[E/F/S cost-based selection](../advanced/fastfit-cost-selection.md). Mode 7 removes
correlated scalars by threshold, clustering or VIF.

## Verifying the reduced model

Compare the full and reduced models on validation data first, then evaluate the
chosen model once on the independent test set:

```sh
imrmlp calc_efs --potfile __REDUCED__.mtp --input test.xyz --output test-predicted.xyz --index index.map
imrmlp calc_err --input test.xyz --input-predict test-predicted.xyz --index index.map --output-summary test-errors.txt
```

Read energy per atom, force and virial errors separately, and inspect the structures
behind the largest deviations. Throughput follows from the model, system and backend
together; it cannot be inferred from the scalar count. If the reduced model is
trained nonlinearly afterwards, it becomes a new candidate and needs the same
validation again.

Compression renumbers the scalars, so an old `selected_ones.dat` never applies to a
compressed model, and the model, selection and output should be archived together.
For a layout-preserving reordering, use
`mtpbasis --source ... --target ... --report ...` and compare E/F/virial after
transferring parameters.

## Common errors

| Symptom | Cause | Action |
| --- | --- | --- |
| `--generate-only` reports a missing argument | Boolean options of `mtmbasis` need an explicit value | Write `--generate-only true` |
| The existing pool is overwritten | `--sample` with `--generate-only true` rewrites `alphapool.dat` | Run sampling in a copy |
| Training fails or the model behaves oddly | `radial_funcs_count` was edited, or species were added to a trained model | Edit the untrained template only; a model file cannot gain species |
| Selection results are not comparable with the baseline | `EIGVEC`/`EIGVAL` reused after changing model, data, weights or normalization | Rerun `--mode 0` |
| A compressed model cannot load the selection | Compression renumbered the scalars | Store the selection together with its compressed model |
| Tools confused | `mtpbasis` compares bases and moves parameters; `mtmbasis` generates them | Pick by purpose |

## Related

- [Basis generation, selection and verification](../advanced/basis-generation-selection.md): all generator settings, four search modes, fastfit modes 0–7
- [E/F/S cost-based selection](../advanced/fastfit-cost-selection.md): the `error-cost` objective for modes 5 and 6
- [MTP walkthrough](../../examples/mtp-workflow/README.md): generate two templates, fit both and pick one
- [Command reference](../reference/commands.md): every option of `mtmbasis`, `fastfit` and `mtpbasis`
