# fastfit basis selection by E/F/S evaluation cost {#doc_docs_fastfit_cost_selection}

[中文](fastfit-cost-selection_zh.md)

`selection-objective error-cost` adds a computational-cost objective to modes 5/6. Search reuses the existing E/F/S quadratic-form scoring, estimates work from the compressed model's dependency structure, then benchmarks a small number of candidates. The default `legacy` retains the original selection logic.

See [basis generation, selection and verification](basis-generation-selection.md) for ordinary MTP generation, fitting, modes and output files. This page covers the error/cost objective in modes 5/6.

## Usage

Prepare a potential, training data and validation data with E/F/S labels. Add to the configuration:

```ini
# ===== Cost-Aware Selection =====
selection-objective error-cost
max-valid-loss      0.02

# ===== Benchmark Budget =====
cost-benchmark-topk 8
cost-benchmark-configs 16
cost-benchmark-rounds 5
cost-benchmark-seconds 0.01

# ===== Archive and Report =====
cost-archive-size   128
cost-report         fastfit-cost.tsv
```

`0.02` is illustrative; choose the limit for your data, weights and acceptable error. Run:

```sh
imrmlp fastfit fastfit.ini -m 6 -np 1 --profile-output fastfit-profile.tsv
# Mode 5 also requires selected_ones.dat and removed.dat.
imrmlp fastfit fastfit.ini -m 5 -np 1 --profile-output fastfit-profile.tsv
```

`max-valid-loss` is a required nonnegative number: **weighted E/F/S squared loss over the full validation set, divided by the number of configurations**, excluding L2. Existing fitness, atomic_fitness, atom-count powers and force normalization apply; virial uses the full 3×3 tensor. Numerical tolerance is `1e-12 * max(1, max-valid-loss)`. The limit constrains combined loss, not each E/F/virial RMSE separately. Validation selects the model; reserve independent test data for final evaluation.

Mode 6 minimizes mean validation loss and relative work, with feasibility-first dominance; the initial population includes the full basis. Mode 5 retains fixed-cardinality exchanges and uses a feasibility-first cost score. SA may explore infeasible regions; the cross-generation archive retains feasible solutions. Annealing temperature in the new mode applies to a dimensionless cost score. NSGA-II error ordering and archive error ties use a fixed binary precision grid (`2^(frexp_exponent(max(1,abs(loss)))-42)`) to suppress ranking changes from tiny reduction errors. Feasibility, reports and final direct checks use unrounded errors.

### NSGA-II initialization and mutation

Mode 6 exposes optional controls while preserving the previous defaults:

```ini
population-init stratified
mutation-rate -1
```

`population-init half` (default) retains each scalar with probability 0.5 in random individuals. `stratified` spreads that probability from 0.05 to 0.95 across population positions; existing file seeds and the full-basis `error-cost` seed still take precedence. Retention is probabilistic, so individuals differ in size; read the actual counts in `selected_ones*.dat` when comparing.

`mutation-rate` is the independent per-scalar flip probability in `[0,1]`, default `0.05`. The sentinel `-1` uses `1/nscalar`, giving **one expected flip** per offspring, not exactly one flip. At 756 scalars, the default gives 37.8 expected flips. Lower rates can help local refinement but may slow broad exploration; compare with equal population/generation budgets across multiple seeds. Logs and cost reports record the effective rate. Both options affect NSGA-II only.

To search basic-moment dependency branches directly:

```ini
population-init dependency
mutation-rate -1
group-mutation-rate 0.5
```

A dependency group contains **every** scalar output using a particular basic moment, computed with the compactor's dependency closure. Identical groups are deduplicated; empty groups and groups covering the entire basis are excluded. Deleting a whole group makes its basic moment removable during compaction, whereas individual scalar deletions often retain these shared dependencies.

`dependency` initialization starts from the full model and removes one group per individual, in seeded shuffled order. The last individual retains the full model; an existing `selected_ones.dat` still takes precedence for individual zero. Groups cycle when there are more individuals than groups; if no group is available, random individuals fall back to `half`. `group-mutation-rate` defaults to `0` and accepts `[0,1]`: after scalar mutation, each offspring drops a uniformly chosen eligible group with this probability. Eligible groups must remove at least one selected scalar and leave a nonempty model. Crossover and scalar mutation can restore deleted scalars. These operations keep the same number of candidate fits per generation. Logs report the deduplicated group count; saved models still undergo the existing loss and reload checks.

Initialization and mutation are independent switches: use `dependency` with `group-mutation-rate 0` to isolate initialization benefits. They only affect mode 6. A group deletion changes the candidate's scalar set only: whether the error stays within the ceiling is decided by the refit in each round, and the speedup appears in the final measurement.

## Cost definition

The shared dependency closure uses the same analysis interface as `remove_orphans()`. It takes the union of dependencies of selected scalars and counts basic moments, all moments, product rules, radial channels and maximum tensor degree. All accumulation rules for a target count. Channel counts reflect the compressed model, including multiple elements and pair-split radial bases.

`efs-ops-v1` estimates operations and writes in the current CPU E/F/S loops:

```
q_active * (6 * channels * rb_size + 2 * channels + 8 * degree + 39 * basics)
+ q_stored * 12 * basics
+ 12 * products + 2 * moments + 4 * scalars + 2 * channels + 2 * rb_size
```

For each channel and radial term, two multiply-add/write sequences count as 6. Basic moments count 32 arithmetic operations and 7 accumulation writes. Jacobian zeroing, contraction and writing count as 12 per stored neighbor and basic moment; forward/backward DAG work counts as 12 per rule. FMA counts multiplication and addition separately. This does not model caches, vectorization or actual instruction latency: **relative work is not a speedup factor**. Benchmarks cover shared radial-basis evaluation, neighbor construction and force reduction.

Representative configurations are stratified by composition, atom count and neighbor density, then sampled with a fixed seed. `q_active` and `q_stored` are normalized by the total atoms in the sample. Stratification constructs each validation configuration's neighbor list once on rank 0. The cost objective does not add per-candidate E/F/S data traversal during search.

The coefficient cache remains limited to 256 entries. The cross-generation archive is bounded by `cost-archive-size` and stores only IDs, losses and structural statistics. It retains representatives spread along cost order, including minimum cost and minimum error; it is not the complete Pareto front. At most `cost-benchmark-topk` archive candidates are benchmarked, plus the full-basis baseline; identical bases are deduplicated.

## Benchmarking and final selection

Shortlisted models are compressed, serialized in `.mtp` format and reloaded. Full training/validation E/F/S are then evaluated directly and checked against cached losses and structural statistics. Serialization uses memory streams to avoid temporary-file conflicts across MPI ranks or concurrent jobs. A failed error check aborts; a model exceeding the limit cannot be selected.

Set threads through imrmlp's global `-np` / `--num_threads` option; setting only `OMP_NUM_THREADS` does not override this program's default. The reported `omp_threads` is the actual configured value.

Rank 0 warms up `PotMTP::getEFSOMP` and interleaves model timings, recording `ns/atom` for two paths:

- `kernel_ns_atom`: reuse existing neighbor lists.
- `total_ns_atom`: reconstruct neighbor lists each time; the default selection metric.

Each path runs for at least `cost-benchmark-seconds` per round (one long evaluation may exceed it), with at least 3 rounds and 5 by default. Reports include medians and the 25%/75% quantiles of total time. Selection starts with the feasible model having the smallest median total time. Models whose interquartile ranges overlap its range are treated as indistinguishable in speed; validation error, then original ID, breaks ties.

Mode 6 may select the full basis. In mode 5, the full basis is only a speed baseline and the final scalar count remains the initial count. "Fastest" holds only for the candidates, configurations and thread/backend measured here, which `fastfit-cost.tsv` records — re-measure after changing them. The fixed-seed structural objective is reproducible; the measured order shifts with machine load, and models inside the fastest median's interquartile range go to validation error, then original ID.

MPI ranks share load statistics and objective definitions. Rank 0 benchmarks and
broadcasts the final choice. Other ranks do not evaluate candidates, but runtime
waiting threads may still affect measurements.

## Outputs and failures

- Mode 6: `__NSGA2__.mtp`, `selected_ones_nsga2.dat`, `removed_nsga2.dat`.
- Mode 5: `__TEMPERED__.mtp`, `selected_ones_tempered.dat`, `removed_tempered.dat`.
- `fastfit-cost.tsv`: baseline, shortlisted candidates, cross-generation archive and measurement environment. Original scalar IDs are listed in compressed order, so list positions are the new scalar IDs.

The report records input-model validation loss, the full basis refitted with identical settings, the error limit, feasibility, structural counts, direct E/F/virial RMSE, estimated work and measured speed. Force RMSE follows the current per-atom force-vector norm convention; virial RMSE follows `ErrorsReport`'s per-configuration atom-count normalization, not pressure-tensor component RMSE.

If no feasible model is found, a failure report is written and the command returns an error, preserving existing potential files. The diagnostic report must be written successfully before final results are published; output-write failures propagate to every rank.

`cost_analysis` and `cost_benchmark` participate in optional section profiling. `scratch_bytes_estimate` estimates only the main E/F/S temporary arrays for the specified maximum neighbor count and thread count. It excludes allocator capacity, radial-basis internal storage, atomic-force buffers and training matrices; it is not process RSS. Process peaks continue to use the existing profiler's historical peak RSS.

## Assembly cache

For repeated searches on the same inputs, the optional C++ entry point
`func_fastfit::fastfit_cached()` accepts a caller-owned `AssemblyCache` from
[`apps/fastfit_cache.h`](../../apps/fastfit_cache.h). It retains at most two
assembled systems, typically training and validation. Exact input model bytes,
parsed structures and labels, per-frame/per-atom weights, normalization and MPI
partition identify each entry; reuse requires a hit on every rank. E/F/S global
weights and L2 are applied after lookup without changing the cached blocks.
Clear the cache before freeing its MPI communicator. Ordinary CLI invocations
are unchanged, and the cache is never persisted across builds or processes.

Regression tests: [`unitest/ut_fastfit_regression.cpp`](../../unitest/ut_fastfit_regression.cpp).

## Explicit shared E/F/V objective and L2

`objective-convention legacy` remains the default. Legacy fastfit uses a summed data objective and penalizes all linear coefficients; ordinary MTP nonlinear training uses a frame-mean objective and penalizes only moments. Equal numeric L2 settings still leave different objectives, because they penalize different parameter groups; select `objective-convention mean-moments` explicitly and align weights and reference convention as in the next section.

`objective-convention mean-moments` defines weighted E/F/full-nine-component virial squared errors, divided by the global number N of loaded training frames, plus `lambda_mean * ||moment_coeffs||²`. Species biases and radial parameters are excluded from this L2 term. Missing labels do not change the frame denominator. Match weights, atom-count powers and reference-force normalization as well.

`l2-reference mean` (the default interpretation in this convention) takes `lambda_mean`; `l2-reference sum` converts the supplied lambda by the actual global N. Fastfit still solves summed normal equations with diagonal increment `N * lambda_mean`. For N=8410, sum-referenced 1e-8 becomes mean-referenced 1.1890606420927468e-12. Removing legacy fastfit's bias penalty changes that historical objective, so record `mean-moments` in the experiment note; otherwise the numbers are not comparable with older runs.

```ini
objective-convention mean-moments
l2-reference sum
weight-l2 1e-8
objective-report objective.tsv
```

For `trainnlpot`, use `fit:objective-convention mean-moments`, `fit:l2-reference sum`, and `fit:weight_L2 1e-8`. Its E/F/V options are `fit:energy_weight`, `fit:force_weight`, and `fit:stress_weight`. This contract currently supports ordinary CPU MTP with `radial_mode=0`, zero `fit:weight_L1`, zero `fit:penalty_weight`, and zero charge/HVP weights. Unsupported combinations fail explicitly; use legacy with a documented objective for other potentials. Standard output and an optional fresh `objective-report` path record normalization, scope, weights and converted L2.

## Deployable model bundles

In error-cost modes 5/6, `cost-archive-rule pareto` enables a capacity-bounded nondominated archive. Default `spread` retains cost-spread exploration and also supports export; it can be preferable when the cost proxy is imperfect. Capacity trimming and shortlisting preserve the cheapest and lowest-loss endpoints; candidates beyond the capacity limit stay out of the archive, so the exported front is built from these representatives.

`cost-export-dir pareto-models` publishes a fresh directory only after every shortlisted model has been saved, reloaded and verified with direct full-data EFS evaluation. Neither the directory nor its `.tmp` path may exist. The usual single-winner output remains. The bundle contains exact input bytes in `parent.mtp`, every verified `candidate-*.mtp`, `objective.tsv`, `manifest.tsv`, and raw `rounds.tsv` timing samples.

The manifest separately marks proxy-cost and measured-latency Pareto fronts within this finite shortlist, and records original scalar IDs, compact graph counts, validation errors, cost, latency and IQR; candidates beyond `cost-archive-size` never enter either front. Force errors keep the per-atom vector-norm convention (square the per-atom Δf, average over atoms, take the root), which is about √3 times the component RMSE for isotropic errors; use the force cmpt columns of `calc_err` for the component convention. Judge speed with the tie rule above (overlap with the fastest median, then validation error, then original ID); the IQR itself only describes the spread of this measurement. Freeze model, dataset and binary SHA256 values in the experiment record; the export does not copy private datasets or commit models.
