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
