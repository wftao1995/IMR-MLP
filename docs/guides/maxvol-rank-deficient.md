# Rank-deficient streaming MaxVol scores

This definition applies to CPU `stream`/`online` selection and grading of their
saved active states. A structure's grade is the maximum of its neighborhood-row
scores. It is not clamped to one. Ordinary full-matrix `MaxVol` uses its existing
truncated-pseudoinverse convention for square matrices; these are different
criteria on rank-deficient matrices.

Let the active matrix be `C` (k rows, p columns, k <= p), with singular values
`s`. Numerical full row rank requires every singular value to exceed
`epsilon * max(k,p) * s_max`. When this fails, use

```
lambda = max(DBL_MIN, max(1e-12, epsilon*max(k,p)) * scale2)
scale2 = s_max^2 if s_max > 0, otherwise 1
H = (C C^T + lambda I)^(-1)
c = b C^T H
rho = ||b - c C||^2 + lambda ||c||^2
gamma(b) = max_i sqrt(c_i^2 + rho H_ii)
```

This is the existing **regularized interpolation score**, now evaluated stably.
It is neither `max(abs(b pinv(C)))` nor the exact regularized determinant ratio
(the latter has an additional `lambda H_ii` term). A pseudoinverse-only score
can miss a candidate in a missing direction; the residual term detects it.
Regularization also damps small nonzero singular directions. Consequently a
truncated SVD score is not a reference for this criterion. Lambda is recomputed
from the new active matrix after every regularized exchange and on reload.
Regularized scores need not be >= 1, even for active rows. A threshold of 1.001
is a stopping criterion, not a guarantee of determinant maximization or of
convergence. Always inspect the final full-pool postcheck.

The implementation forms `c` using `V diag(s/(s^2+lambda)) U^T`, not `C^T H`.
It computes the explicit residual vector, avoiding the unstable subtraction
`||b||^2 - c (b C^T)^T`. Scans and exchange decisions share this calculation.
Batching, BLAS implementations and thread counts may change rounding bits;
agreement means numerical agreement, not bitwise identity. Regression checks
include an independent spectral reference, batching, exchange decisions and
save/load. The supplied 2653-column MTP case is checked with tolerance
`1e-8 * max(1, abs(grade))`.

Well-conditioned square states retain inverse/Sherman–Morrison scoring;
full-row-rank wide states retain the unregularized row-volume criterion.
Saved state formats are unchanged: derived factors are rebuilt from C.
The public CPU implementation uses the same stable regularized criterion.

## Real-data validation (2026-11-10)

The supplied MTP model (2653 columns) and the active matrix from job 38616514
were copied read-only; no cluster files were changed. The saved matrix has
numerical rank 2596 at the threshold above. Model SHA256:
`05f7d527cbb4483532e3d9242fc3ac068a127c831dd154f7003af7c66942c81f`.
Training-file SHA256:
`a7022afe87deb1308e3d8fb334a43607614544cc4bae59dc28ff7145a3295315`.
No private model or structures are part of this repository.

- All 2080 structures (619592 neighborhoods): `grade_all` versus repeated
  `grade`, CPU with 8 OpenMP threads and MKL_NUM_THREADS=1. Maximum absolute
  structure-grade difference: 2.2155e-11 (structure index 1236).
- Original structure index 9, all 416 neighborhood rows: matrix-batch versus
  individual `OnlineMaxVolState::score_rows` calls, MKL/OMP threads=1. Maximum
  absolute row-score difference: 5.1137e-13.
- Independent SciPy SVD reference, using spectral ridge residual
  `sum_j (b V)_j^2 * lambda/(s_j^2+lambda)` for this square matrix: maximum
  absolute row-score difference 4.3371e-12 across those 416 rows.
- Row 5: old batch 177.7890393, old scalar 0.1127337; fixed 0.9761289492239
  in both. Row 3: fixed 1.1906614536338, and a one-candidate exchange at
  threshold 1.001 now performs one swap. The old scalar score was 0.1892529.
- The public implementation produces identical values for all 416 row probes
  and the same successful row-3 exchange.

These are checks of a fixed saved active matrix and an exchange, not a claim
that a new converged active set has been generated. The fixed full-pool maximum
is 61.7315651: the old state remains nonconverged. Its old log reported 303.7718.
The stable criterion still detects genuine extrapolation in that pool.

Local validation artifacts were kept in `/tmp/maxvol-investigation/`:
`validate_pool.cpp`, `validate_rows.cpp`, `fixed_pool.tsv`, `fixed_rows.tsv`,
`pub_fixed_rows.tsv`, `spectrum.py`, and `candidate_reference.npz`. The committed
synthetic regression in `unitest/ut_online_maxvol.cpp` does not need these files.

Validation commands (release, oneMKL 2026.0; MKLROOT supplied by the local
oneAPI environment):

```sh
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" OMP_NUM_THREADS=4 MKL_NUM_THREADS=1 make -j4 test
```

The complete CPU suites passed: atoms-agent 469/469 test programs,
imrmlp-pub 309/309. The changed GPU grader translation units also passed
`make check-syntax` with `SYNTAX_FLAGS='-DGPML_GPU -DGPML_CUDA'` in atoms-agent;
GPU runtime execution was not tested.

## Batched CPU selection

`selector:stream-mode batch` opts into cached, single-process CPU selection
with a bounded candidate working set. `exact` and `fast` keep their existing
selection policies. Batch mode scans the pool, keeps the highest-scoring
unselected rows, and performs at most 64 exchanges before scanning again.
It recomputes the working-set scores after every exchange and keeps outgoing
active rows in the working set. The default working set has 256 rows;
`selector:stream-fast-shortlist` overrides its size. `selector:max-swaps`
limits the total exchanges. The ordinary stream iteration limit caps pool
passes, not exchanges. A final scan includes **all** rows, including active
rows. Working-set exhaustion is never a convergence certificate.

Batch mode additionally enables guarded updates of square deficient states.
The score definition above, including the current-matrix value of lambda,
is unchanged. The implementation maintains three operators:

```
H = (C C^T + lambda I)^(-1)
M = C^T H
R = V diag(lambda/(s^2+lambda)) V^T = I - M C
c = b M
rho = ||b R||^2 + lambda ||c||^2
```

`R` is formed spectrally and updated directly; it is never formed as `I-MC`.
Updating `M` alone and subsequently reconstructing `b-bMC` amplified roundoff
in the supplied data. A single-row replacement instead updates `H`, `M` and
`R` with rank-two formulas using the explicitly evaluated residual. A warm
power iteration estimates `s_max^2` only when its Rayleigh residual and the
Frobenius-norm bound on the second eigenvalue certify a separated maximum.
A retained right-null vector must still certify rank deficiency. Otherwise,
selection falls back to a complete SVD, including when a missing direction
is restored. A resolvent series adjusts **all three operators** for the new
lambda; changes greater than one percent fall back to full factorization.
The series has operator remainder at most `1e-10` in exact arithmetic.

The working accuracy target is `1e-4 * max(1, abs(grade))`. Scores within
`1e-4 * max(1, threshold)` of the stopping threshold trigger a fresh
factorization before the exchange/stop decision. Operator consistency probes
and a mandatory rebuild at least every 64 exchanges limit accumulated drift.
Batch rebuilds use divide-and-conquer SVD. The final published state is rebuilt
with the ordinary stable factorization and the complete pool is checked again;
save/load still serializes the active matrix rather than derived update factors.
Batch mode requires cached, single-process CPU stream selection.

Verbose output reports pool scans, swaps, full factorizations, accepted ridge
updates, and their timings. `score` in the existing stream summary still
includes exchanges and finalization; the additional `scan` timer measures
pool scoring passes. The residual operator and update temporaries require additional O(p^2)
workspace beyond the row cache limit. Private input data and generated
states are not tracked in git.

Between scans, batch mode uses at most `min(8, OMP thread count)` local MKL
threads for dense factors and working-set products. Each pool worker still
uses one MKL thread. The caller's local MKL setting is restored on return.
A repeated physical active set at a batch boundary terminates with `cycle`
unless the final complete-pool check establishes convergence.

## Batch performance on the supplied MTP pool (2026-11-11)

The same model and input hashes above were used for fresh local selection:
2080 structures, 619592 neighborhood rows and 2653 columns. The threshold
was 1.001, with a memory cache, 20 OpenMP workers and MKL_NUM_THREADS=1
(the batch-local MKL team is described above).

| Selection policy, 100-swap budget | Historical exact | Batch |
| --- | ---: | ---: |
| Selection wall time | 4486.964 s | 675.457 s |
| Initialization | 505.969 s | 510.628 s |
| Scoring, exchanges and finalization | 3962.118 s | 146.482 s |
| Complete pool scans, including final check | 101 | 3 |
| Final complete-pool maximum | 4.5649058730 | 8.3641559774 |
| Converged | No | No |

At this budget, batch selection is 6.64 times faster overall and 27.05 times
faster after initialization. Exchange ordering differs: this is a same-budget
comparison, **not** an equal-quality speedup. The batch run accepted 99 ridge
updates and performed 3 complete factorizations, including final publication.
Peak process RSS was 13993.5 MiB. Initialization remains the dominant cost
for this short run and is unchanged by the optimization.

After saving the 100-swap state, all 2080 structure grades were checked in
both repositories. The maximum batch-versus-single difference was 1.3225e-12;
the two repositories produced identical output values. Independent SciPy
SVD recomputation of all 619592 rows gave a maximum structure-grade difference
of 2.1124e-11 (scaled difference 5.1264e-12). For the 416 rows of structure
index 9, the batch/single difference was 6.3583e-13 and the SVD difference
was 2.3635e-12. The spectral reference uses independent SVD factors and the
ridge formula above; switching its dense products from Netlib to MKL agreed
within 2.8689e-13 on 1081 structures.

A corresponding batch configuration is:

```ini
selector:select-mode stream
selector:stream-mode batch
selector:stream-cache-mode memory
selector:stream-memory-limit-mb 32768
selector:threshold 1.001
selector:max-swaps 4096
```

The larger run converged at the unchanged 1.001 threshold. It used 1536
exchanges, 30 complete-pool scans (including the final check), 27 full
factorizations and 1513 accepted ridge updates. The final complete-pool
maximum was **1.0007199559431048**, with 754 selected structures and 2653
active rows. Selection took 2463.758 s (41.06 min); the whole command,
including output publication, took 2481.294 s. Peak RSS was 14084.8 MiB.
This run overlapped validation and test compilation for part of its duration;
its elapsed time is an observed completion time, not an isolated speedup
measurement. Factorizations took 59.485 s, ridge updates 941.410 s and full
pool scans 896.255 s. The configured 4096-swap budget was not exhausted.

The final complete CPU test suites passed: atoms-agent 470/470 programs,
imrmlp-pub 309/309. Both command targets rebuilt successfully. The new
regressions include the no-argument streaming entry point, which must also
reject an uncached batch request.

Final-state validation covered all 2080 structures and 619592 neighborhood
rows. The matrix remained numerically rank deficient (2596/2653), with
lambda = 3.1692524600048273e-10 under the definition above.

- Bulk versus repeated single-structure grades: maximum absolute difference
  2.3759e-13, in both repositories.
- Independent SciPy SVD versus bulk grades: maximum absolute difference
  1.7402e-12 over all structures.
- Structure index 9, all 416 rows: batch/single maximum difference 5.1209e-13;
  batch/SVD maximum difference 1.5274e-12.
- The two repositories' complete grade tables were identical. The tenth
  structure's final grade was 0.9297361442642903.

These final-state errors are observed validation results, not a tighter
working-update tolerance; the intermediate target remains 1e-4 as stated
above. Local inputs, states, timing logs and validation tables are kept under
`build/analysis/maxvol-opt-20261111/` in atoms-agent and excluded from git.
No cluster files were modified.
