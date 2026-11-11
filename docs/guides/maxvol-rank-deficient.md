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

`selector:stream-mode batch` opts into cached CPU selection
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
Batch mode requires cached CPU stream selection; multiple MPI ranks are supported.

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

## MPI batch selection

With an MPI build, `stream-mode=batch` distributes cached pool rows and full
pool scans across ranks. Rank 0 alone updates the candidate working set and
performs factor rebuilds. It broadcasts C, its inverse/ridge operators,
regularization, row references and exchange history before the next scan.
Workers receive the derived operators directly; they do not repeat an SVD at
every batch boundary. The final ordinary factorization is also performed once
on rank 0 and broadcast before every rank checks its entire local pool. An
MPI maximum reduction establishes the final global threshold result.

The scoring definition and 1e-4 intermediate tolerance are unchanged. A mode
or numerical-control disagreement is rejected before ranks can enter different
collective sequences. Canonical update and synchronization failures propagate
to all participating ranks. Caller-provided communicators are supported.
Factor statistics count canonical rebuilds/updates, not their sum over ranks.

Each rank retains the compact active matrix and operators; only the row cache
is partitioned. A square deficient 2653-column state broadcasts approximately
215 MiB of dense matrices per batch. Thus more ranks do not imply a speedup:
rank-0 updates remain serial across MPI ranks, and communication can dominate
on a slow interconnect. Use one OpenMP team per rank and avoid oversubscribing
allocated CPUs. The cache/memory limit is per rank.

For example, after building the current MPI executable, run two ranks with
ten OpenMP threads each and the same batch settings as above:

```sh
export OMP_NUM_THREADS=10 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
# On Slurm, request matching resources first: --ntasks=2 --cpus-per-task=10.
srun --ntasks=2 --cpus-per-task=10 --cpu-bind=cores "$MLP" maxvol_select \
  --settings maxvol_select.ini --pool "$TRAIN" --index index.map \
  --input-format extxyz --output-format extxyz --num_threads 10 \
  --output active.extxyz --state-output final.gpm
```

MPI builds retain sequential MKL by default. To also parallelize the canonical
rank's dense batch updates, build with `MPI_MKL_THREADING=gnu` (atoms-agent:
`make BUILD_MODE=mpi MPI_MKL_THREADING=gnu gpmain`; imrmlp-pub: replace the
target with `imrmlp`). Keep `MKL_NUM_THREADS=1` externally; batch selection
locally enables at most min(8, OpenMP threads per rank) for dense updates and
restores the previous setting. Pool workers continue to use one BLAS thread.
The optional build switch changes MPI linking only; serial builds and the
existing default MPI threading policy are unchanged.

The MPI regression program is `unitest/ut_maxvol_batch_mpi.cpp`. Run it with
both two and four ranks; four ranks exercise empty partitions in the two-row
fixture. It also reverses communicator rank order, checks 130 consecutive
exchanges and rank restoration, compares serial/MPI final results, verifies a
purely virtual active set, and checks collective error propagation and control
agreement. Catch test ordering and seed are fixed across ranks.

```sh
make BUILD_MODE=mpi MPI_MKL_THREADING=gnu build/mpi/unitest/ut_maxvol_batch_mpi.test
# Run from a temporary working directory; use absolute checkout paths.
LD_LIBRARY_PATH="$REPO/build/mpi:$LD_LIBRARY_PATH" OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
  mpirun --bind-to none -np 4 "$REPO/build/mpi/unitest/ut_maxvol_batch_mpi.test"
```

### Local MPI validation on the supplied pool (2026-11-11)

The atoms-agent MPI executable completed the same 2080-structure, 619592-row
case with two ranks and ten OpenMP threads per rank, using GNU-threaded MKL,
external MKL_NUM_THREADS=1, the memory cache and threshold 1.001. The final
full-pool check converged at **1.0007199559431041**. It selected 754 structures
and 2653 active rows, using 1536 exchanges, 30 complete-pool scans, 27 complete
factorizations and 1513 ridge updates. Factor counts describe the canonical
state, not per-rank duplicate work.

The selected structure indices, selected row references and saved active
matrix are identical to the earlier single-process run. The active-matrix raw
SHA256 is `712ac4ede258024b708fec6aaad5d73ea063daccaf769445a3445e7ea56a0530`
in both runs. The final maximum differs by only 6.6613e-16 due to scoring
roundoff. Every printed batch's exchange count, maximum and factor counts
also match the serial trajectory.

| Local layout | Historical 1 rank x 20 threads | 2 ranks x 10 threads |
| --- | ---: | ---: |
| Selection time | 2463.758 s | 2404.470 s |
| Whole command wall time | 2481.294 s | 2423.405 s |
| Full-factor time | 59.485 s | 53.679 s |
| Ridge-update time | 941.410 s | 932.489 s |
| Pool-scan time | 896.255 s | 831.908 s |

These are observed timings, not a controlled scaling result: the historical
serial run shared the machine with other validation/build work. At the same
20-core budget, this comparison establishes no substantial MPI speedup.
MPI partitions row storage and scans; rank-0 exchanges still dominate much
of the run. Multi-node performance has not been measured.

Peak process RSS was 7819.5 MiB on rank 0 and 7132.5 MiB on rank 1. The maximum
combined RSS observed at five-second intervals was 14606.3 MiB (a sampled
value, not an exact simultaneous peak). Replicating the compact state adds
memory overhead, so total node memory is not halved. The printed stream row
cache, builder-call and row counts are rank-local: rank 0 built 1040 structures
and 311032 rows; the global input remains 2080 structures and 619592 rows.

After saving and reloading the MPI result, both repositories checked every
structure with bulk and repeated single-structure grading. Independent SciPy
SVD recomputed all 619592 neighborhood scores with the same ridge definition.
The final matrix has numerical rank 2596/2653 and
lambda = 3.1692524600048273e-10.

- Maximum bulk/single structure-grade difference: 2.3759e-13.
- Maximum bulk/SVD structure-grade difference: 1.7402e-12.
- Structure index 9, all 416 rows: batch/single difference 5.1209e-13 and
  batch/SVD difference 1.5273e-12.
- Both repositories' grade tables are identical. The tenth structure's grade
  remains 0.9297361442642903.

The validation artifacts are local to atoms-agent under
`build/analysis/maxvol-mpi-batch-20261111/` and excluded from git. No files on
the cluster were changed.

Both complete release suites passed after the MPI changes: atoms-agent
471/471 test programs and imrmlp-pub 310/310. In both repositories, the MPI
regressions passed with two and four ranks using GNU-threaded MKL and with
two ranks using the default sequential MKL link. The current MPI command
binaries were then rebuilt with `MPI_MKL_THREADING=gnu`. The full-suite command
was `LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" OMP_NUM_THREADS=4
MKL_NUM_THREADS=1 make -j2 test`, with the appropriate local MKLROOT.
