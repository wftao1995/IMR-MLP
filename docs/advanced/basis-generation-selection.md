# MTP basis generation, selection and verification {#basis_generation_selection}

[中文](basis-generation-selection_zh.md)

This page connects `mtmbasis → trainmtp/trainnlpot → fastfit → calc_efs/calc_err` for MTP with an ordinary linear head. First complete the [training workflow](../guides/workflows.md) and prepare `train.xyz`, `valid.xyz` and an independent `test.xyz` with consistent species mapping. See structure formats for data conventions and MTP formats for model layouts.

For a complete script requiring no external data, run the basis generation and selection example. Initialization, fitting and reload for other heads and long-range models are covered in model-family workflows.

## 1. Distinguish three kinds of basis optimization

| Task | Tool | What changes | Criterion |
|---|---|---|---|
| Define candidate scalar invariants | `mtmbasis` enumeration/sampling | Function space and candidate count | Tensor orders, level and contraction constraints |
| Optimize contraction evaluation | `mtmbasis` search | Intermediate moments, product rules and shared expressions | Structural counts, without training E/F/S |
| Select scalars using fitting quality | `fastfit` | Retained scalars and refitted linear coefficients | Training/validation data, correlation and optional measured cost |
| Check equivalence after reordering | `mtpbasis` | Read-only comparison, or parameter transfer | Exact symbolic expressions and compatibility checks |

Fewer tensors from `mtmbasis` do not imply lower validation error; fewer scalars from `fastfit` do not imply faster evaluation. Distinguish `mtpbasis` (comparison/remapping) from `mtmbasis` (generation). Selecting basis functions also differs from selecting **configurations**, as in `maxvol_select`.

With radial parameters fixed, ordinary MTP is linear in species offsets and scalar moment coefficients. `fastfit` freezes the input model's radial parameters and repeatedly fits its linear part; it does not perform nonlinear radial training. LES/SOG, MTP-M and separate NN heads are outside this ordinary-MTP selection workflow.

## 2. Define a candidate pool with mtmbasis

Create `mtmbasis.ini` in a separate working directory. This small basis demonstrates the workflow; its settings are not an accuracy recommendation for production:

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

These are top-level keys, outside `[pot]` and `[fit]`. For a product of d moment tensors, the current generator uses

\[
\mathrm{level}=w_\mu\sum_{i=1}^{d}\mu_i+w_\nu\sum_{i=1}^{d}\nu_i+w_d d.
\]

| Setting | Default | Meaning |
|---|---|---|
| `max_lev` | 16 | Maximum combined level |
| `max_dim` | 4 | Maximum tensor factors per contracted invariant, supported range 1–7; not spatial dimension |
| `max_mu` | 4 | Maximum mu in the automatically generated `(mu,nu)` list |
| `max_nv` | 10 | Maximum angular tensor rank nu |
| `max_poly` | 10000 | Maximum sum of nu in a combination |
| `mu_wei`, `nv_wei`, `dim_wei` | 2, 1, 2 | The three level weights |

The current implementation additionally clamps `max_dim` to `min(max_dim, max_lev / 2)` using integer division; account for this extra limit even when changing the weights.

mu labels a radial channel; it is not the radial expansion size `radial_basis_size`. nu is tensor rank. A product of d neighbor-sum factors can contain contributions involving up to d+1 atoms, but also contains lower-body terms from repeated neighbors. `max_dim` is not a strict physical body order. Level conventions differ across publications and generators: record the weights and all truncation settings when comparing bases.

Save the candidate pool, then optimize its implementation:

```sh
imrmlp mtmbasis mtmbasis.ini --generate-only true
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --potfile basis.mtp
```

Current `mtmbasis` boolean options take explicit values, such as `--generate-only true`, `--sample true` and `--advanced-sample true`. Bare `--generate-only` reports a missing argument.

To restrict candidate moment tensors, replace automatic enumeration with `--moments moments.txt`. The first value is the count, followed by exactly that many `(mu,nu)` pairs, for example:

```text
3
0 0
1 0
0 2
```

Level, factor-count and total-order limits still apply. Negative values, duplicate pairs and extra/missing data are rejected. An existing `--alphapool` supplies the scalar candidate pool directly; preserve its generating configuration.

## 3. Contraction search, sampling and outputs

The contraction objective is `w1 * moments_count + w2 * rules_count`, with default weights 1 and 4. It counts intermediate moments and product-accumulation rules, not measured execution time on the target machine.

| `mode` | Search | Budget and behavior |
|---|---|---|
| `0` | Simulated annealing | `T_init`, `maxiter`; the default mode |
| `1` | Random starts plus local search | `maxiter` |
| `2` | Genetic search | `popsize`, `childsize`, `maxiter` and mutation/exchange settings |
| `3` | Coordinate descent | Search each branch until a sweep makes no improvement; `maxiter` does not currently limit sweeps |

`--potfile` sets the output prefix: the example writes `basis.mtp_<number>`; the default prefix is `potential.mtp`. Initial and improving candidates are saved, not necessarily one file per search iteration. Select a template using the logged moments, rules and score, and record the exact filename rather than relying on an ambiguous wildcard.

`mid_tid_mapping_<number>.log`, `scalar_id_tensor_id_mapping_<number>.log`, `scalar_pool_<number>.dat` and `conns_exp.log` trace expressions and contraction paths. Options that add path scalars can also affect the resulting scalar space; check it explicitly. Use `mtpbasis` to establish equivalence between outputs, not file length or moment count alone.

Large candidate pools can be sampled first:

```sh
imrmlp mtmbasis mtmbasis.ini --alphapool alphapool.dat --sample true --sample-rate 0.5 --generate-only true
```

`--sample` writes `sampled_alphas.dat` but **does not exit on its own**. The additional `--generate-only true` saves the sampled `alphapool.dat` and exits, overwriting the original pool in that directory. Run this in a copy directory. For factor-count sampling, use `--advanced-sample true` and top-level `sample_rates`: the first six positions correspond to d=2…7, omitted rates are 0, and single-factor scalars are retained. This branch writes `sampled_alphas.dat` and exits. Rates must be in `[0,1]`.

Enumeration and sampling change the candidate function space; contraction search primarily changes evaluation paths. Pool sampling currently uses `random_device` without an exposed seed option. Preserve the sampled pool, not just its sampling rate, for reproduction.

## 4. Turn the template into a model suitable for selection

Generated files are `MTP / version = 1.1.0` templates with default `species_count=1`, `RBChebyshev`, `min_dist=2`, `max_dist=5` and `radial_basis_size=8`. They contain topology but no trained radial or linear coefficients. Successful loading does not make a template a validated material potential.

Check and set species count/order, distance range and radial expansion size on the **untrained template**. `radial_funcs_count` follows the basis mu indices and cannot be changed arbitrarily. Nor can a trained model acquire extra species or radial dimensions through header edits. See MTP formats and the radial-basis reference for multiple species, pair-split and configurable radial bases.

Copy the selected template to `template.mtp` and prepare `index.map`. This minimal training configuration uses data in the working directory:

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
fit:random_init false
fit:force_randomradial false
fit:force_randomlinear false
```

Save it as `train.ini`, then run:

```sh
imrmlp trainmtp train.ini --linear-fit
cp _Trained_Potential_.mtp full.mtp
```

This verifies the connection from generation to training. Production usually also requires radial optimization, for example `trainmtp --do-lbfgs` or `trainnlpot --do-varpro` as described in the training guides. Save the selected model as `full.mtp`; `fastfit` will freeze these radial parameters. Do not substitute `train_linear` for ordinary MTP's dedicated fitter.

## 5. Establish a fastfit baseline, then select

Create `fastfit.ini`, again using **top-level keys**, rather than `fit:` weights or `pot:` model keys:

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

`format` applies to both training and validation; the default is `mlpcfg`, so specify extxyz explicitly. The model must have the data's species order and a fixed radial layout. Fit parameters on training data and compare candidates on validation data. Keep final test data out of this configuration.

| Mode | Purpose | Main output/prerequisite |
|---|---|---|
| `0` | Eigenvalue diagnostics of the scalar normal matrix | `EIGENSYS`, `EIGVEC`, `EIGVAL`; not SVD of the original design matrix |
| `1` | Weighted full-basis linear refit | `__Trained__.mtp`; `--cmp` compares solvers without the usual export branch |
| `2` | Apply an existing selection, refit and compress | Reads `selected_ones.dat`, writes `__REDUCED__.mtp` |
| `3` | Individual-removal sensitivity diagnostics | Loss report for each scalar; not sequential greedy elimination and no final reduced-model export |
| `4` | CUR-score selection | Reads mode 0's `EIGVEC` and `EIGVAL`; writes `selected_ones.dat` and `removed.dat`; use mode 2 to export |
| `5` | Fixed-cardinality SA/greedy exchanges | Reads those complementary, nonempty sets; writes `__TEMPERED__.mtp` |
| `6` | NSGA-II multi-objective subset search | Requires at least two scalars; writes `__NSGA2__.mtp` and selection files |
| `7` | Threshold/clustering/VIF decorrelation | Selection files and `correlation_*`; `--corr-save-reduced` also fits and writes `__DECORR__.mtp` |

The shortest baseline → selection → export workflow is:

```sh
imrmlp fastfit fastfit.ini --mode 1 --num_threads 1
imrmlp fastfit fastfit.ini --mode 0 --num_threads 1
imrmlp fastfit fastfit.ini --mode 4 --discard-rate 0.25 --num_threads 1
imrmlp fastfit fastfit.ini --mode 2 --num_threads 1
```

Mode 4 requires `EIGVEC` and `EIGVAL` generated by mode 0; it cannot run directly after mode 1 in a clean directory. Selection is deterministic by score unless `--cur` enables random sampling. `--discard-rate` is the fraction to discard. `--addback` can retain scalars used along evaluation paths, so the final count may differ from a simple fraction.

Refine the initial selection with `--mode 5 --search-steps 100 --method greedy`, or try an independent `--mode 6 --population 16 --generations 10`. Mode 5's default legacy score combines training and available validation loss. Mode 6's legacy objectives are training/validation loss when validation exists, or training loss/scalar count otherwise. With validation data, this is not an explicit error/speed objective. These example budgets do not guarantee convergence.

For speed under a validation-error ceiling, use `selection-objective error-cost` in mode 5/6; see [E/F/S cost-based selection](fastfit-cost-selection.md). It requires nonempty validation data, searches estimated work and times a small shortlist. Scalar count is only a proxy for work and does not scale with actual CPU/GPU/LAMMPS time.

## 6. Selection files, matrix caching and reproducibility

The first item in `selected_ones.dat` is the count, followed by zero-based **scalar positions**. To retain 0, 2 and 5:

```text
3 0 2 5
```

An ID is a position in the original model's `alpha_moment_mapping`, not its moment ID or a linear-parameter index offset by `species_count`. Species offsets are retained implicitly. Mode 5's `removed.dat` must complement the selection for the same original model, without duplicate, out-of-range, missing or overlapping IDs. Compression renumbers scalars: never apply the old selection directly to the compressed model. Archive the input model, selection and output together.

By default, `fastfit` assembles matrices from the model and data; no pre-generated `MATATA_*` files are needed. `--use-matrices` reads unweighted E/F/S blocks exported by `trainmtp --do-matrix-ata` and reassembles the current problem to validate them. It is not a shortcut that bypasses reading data. Regenerate after changes to data, radial coefficients, species mapping, normalization or the basis. Methods such as `--method svd` solve normal equations here; their numerical properties differ from QR/SVD applied directly to the original design matrix.

Mode 4 checks eigensystem dimensions and finiteness, but does not establish that the files came from the current model and data. Rerun mode 0 after changing the model, data, weights or normalization; do not reuse stale `EIGVEC` and `EIGVAL` merely because their dimensions match.

Record input files, binary version, weights and atom-count powers, seed, thread/MPI-rank counts, selections and benchmark environment. The same seed does not imply bitwise equality across MPI partitions or machine loads. Profiling is described in the [fastfit guide](fastfit-cost-selection.md). Use separate directories because many output names are fixed.

## 7. Independent verification and deployment

Compare the full baseline and reduced model on validation data first. After choosing the model, evaluate the independent test set once:

```sh
imrmlp calc_efs --potfile __REDUCED__.mtp --input test.xyz --output test-predicted.xyz --index index.map
imrmlp calc_err --input test.xyz --input-predict test-predicted.xyz --index index.map --output-summary test-errors.txt
```

Retain separate E, F and virial errors, and inspect extreme configurations and relaxation/MD stability for the target application. Throughput follows from the model, system and backend together; it cannot be inferred from the scalar count. Further nonlinear training after reduction produces a new candidate and requires renewed validation. For graph reordering intended to preserve predictions, use `mtpbasis --source ... --target ... --report ...`; compare E/F/virial after parameter transfer as well.

Complete options are in the [command reference](../reference/commands.md). Implementation entry points are `apps/genmtm.cpp`, `mtmbasis/mtm.cpp` and `apps/fastfit.cpp`; regression coverage includes `unitest/ut_mtmbasis.cpp`, `unitest/ut_fastfit_regression.cpp` and `unitest/ut_fastfit_correlation.cpp`.
