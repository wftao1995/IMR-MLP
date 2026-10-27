# MAXVOL Active Learning {#theory_maxvol}

[中文](maxvol_zh.md)


@tableofcontents

## 1. Scope

MaxVol (Maximum Volume) selects a set of rows from a candidate matrix aiming for a large
spanning volume and informative directions for the chosen representation. In machine-learned interatomic potentials
(MLIPs), candidate rows are typically descriptors, or derivatives of energy,
forces, and stress with respect to potential parameters. MaxVol is therefore
used for:

- building numerically well-conditioned initial training sets;
- detecting whether a new configuration lies outside the existing training domain;
- incrementally growing the training set during active learning or
  learning-on-the-fly (LOTF);
- covering a very large candidate pool with a small set of representative
  configurations.

For a first selection run, follow the
[training and active-learning workflow](../guides/workflows.md); use the
[command reference](../reference/commands.md) to look up individual options.
This page explains the algorithms behind those commands, including more than
the classical square MaxVol. It covers the classical algorithm, the generalized
volume for rectangular matrices, B-matrix construction, the three selection
backends, extrapolation grades, numerical stability, complexity, command-line
usage, performance tuning, and frequently asked questions.

The most important practical conclusions are:

1. The default `select-mode=stream` with `stream-mode=exact` fits most offline
   selections;
2. `full` fits small/medium matrices, reference results, and debugging;
3. `online` has the lowest memory footprint, but never revisits rejected
   historical candidates, so it is order-dependent;
4. the grade is an algebraic measure of "how much the volume could grow by
   replacing an active row", not an energy error, probability, or confidence
   interval;
5. old grades are not comparable once the B-matrix definition, weights,
   atom-count scaling, or the potential parameterization change.

## 2. Notation and Problem Definition

Let the candidate row matrix be

@f[
A =
\begin{bmatrix}
\mathbf a_1 \\
\mathbf a_2 \\
\vdots \\
\mathbf a_m
\end{bmatrix}
\in \mathbb R^{m\times n},
@f]

where:

- @f$m@f$ is the number of candidate B rows;
- @f$n@f$ is the descriptor dimension or the number of potential parameters
  participating in the selection;
- @f$I=(i_1,\ldots,i_k)@f$ is the set of selected row indices;
- @f$C=A[I,:]\in\mathbb R^{k\times n}@f$ is the active matrix;
- @f$\tau\ge 1@f$ is the user threshold, default `1.001`;
- @f$s@f$ is the number of row swaps actually performed.

Classical MaxVol takes @f$k=n@f$ and aims at

@f[
I_* = \arg\max_{|I|=n} |\det A[I,:]|.
@f]

Exhaustive search over @f$\binom{m}{n}@f$ submatrices is intractable. This
implementation uses pivoted-QR initialization followed by greedy single-row
exchanges, so it returns a **single-exchange local optimum** satisfying the
threshold; it is not a rigorous guarantee of the global maximum determinant.

## 3. Basic Mathematics

### 3.1 Volume and D-optimality

For a square matrix @f$C\in\mathbb R^{n\times n}@f$, the volume spanned by its
rows is

@f[
\operatorname{vol}(C)=|\det C|=\prod_{j=1}^{n}\sigma_j(C),
@f]

where @f$\sigma_j@f$ are the singular values. A large volume means the
directions do not collapse, which usually also implies good linear independence
of the selected rows. However, maximum volume and "minimum condition number"
are not mathematically equivalent optimization problems; the documentation and
logs should not treat them as strictly interchangeable.

If the linearized model is

@f[
\mathbf y \approx B\boldsymbol\theta,
@f]

and the observation noise is i.i.d. after weighting, the Fisher information
matrix / normal matrix is @f$F=B^\mathsf T B@f$. For a square active matrix,

@f[
\det(C^\mathsf T C)=\det(C)^2.
@f]

Hence maximizing @f$|\det C|@f$ is equivalent to maximizing the determinant of
the information matrix — this is the link between MaxVol and D-optimal
experimental design: it tends to shrink the volume of the parameter confidence
ellipsoid rather than optimizing one specific energy or force error.

### 3.2 Expansion Coefficients of Candidate Rows

Assume @f$C@f$ is invertible. The expansion coefficients of all candidate rows
in the active basis are

@f[
R = A C^{-1}\in\mathbb R^{m\times n},
\qquad
\mathbf a_l = \sum_{j=1}^{n}R_{lj}\mathbf c_j.
@f]

Replacing the @f$j@f$-th row of @f$C@f$ by the candidate row @f$\mathbf a_l@f$
gives @f$C'@f$, and by the matrix determinant lemma,

@f[
\frac{|\det C'|}{|\det C|}=|R_{lj}|.
@f]

Thus @f$|R_{lj}|@f$ is both the expansion coefficient and the exact volume
amplification factor of that swap. Classical MaxVol searches each round for

@f[
(l_*,j_*)=\arg\max_{l\notin I,\,j}|R_{lj}|.
@f]

If the maximum exceeds @f$\tau@f$, the swap is performed; otherwise the
algorithm stops.

### 3.3 Convergence Condition and Local Optimality

The stopping condition of the classical square path is

@f[
\max_{l\notin I,j}|(AC^{-1})_{lj}|\le\tau.
@f]

With @f$\tau=1+\varepsilon@f$, any single-row replacement can enlarge the
current volume by at most @f$1+\varepsilon@f$. The user-facing parameter in
this repository is `threshold`=@f$\tau@f$; the low-level `MaxVol::tolerance`
is @f$\varepsilon=\tau-1@f$.

A smaller threshold means stricter local optimality and more scans/swaps; a
larger threshold stops earlier. @f$\tau\ge1@f$ is recommended. The
implementation clamps stop thresholds below 1 to 1, so setting a value below 1
cannot request something "stricter than maximum volume".

### 3.4 Extrapolation Grade

For a new row @f$\mathbf b@f$, the row-level grade on a square active set is
defined as

@f[
\gamma(\mathbf b;C)
=\|\mathbf b C^{-1}\|_\infty
=\max_j |(\mathbf b C^{-1})_j|.
@f]

For a configuration @f$q@f$ whose B matrix has multiple rows @f$B_q@f$, the
configuration-level grade is

@f[
\Gamma(q;C)=\max_{\mathbf b\in\operatorname{rows}(B_q)}
\gamma(\mathbf b;C).
@f]

Engineering interpretation:

- @f$\Gamma<1@f$: all rows lie inside the current active basis with small
  expansion coefficients;
- @f$\Gamma\approx1@f$: near the active-set boundary;
- @f$\Gamma>\tau@f$: at least one row replacement can noticeably increase the
  volume;
- @f$\Gamma\gg1@f$: the configuration is strongly extrapolative with respect to
  the current B-matrix definition.

The grade has no energy units. It does not directly give an energy/force
error, and a small grade does not guarantee the absence of model bias: if the
descriptors, the potential form, or the candidate pool itself miss a physical
state, MaxVol cannot detect bias outside that representation.

### 3.5 Generalized Volume for Rectangular Active Matrices

The low-level `MaxVol::select_rows(A, k)` supports @f$k\ne n@f$. This
implementation defines

@f[
\operatorname{vol}(C)=
\begin{cases}
\sqrt{\det(CC^\mathsf T)}, & k\le n,\\
\sqrt{\det(C^\mathsf T C)}, & k\ge n.
\end{cases}
@f]

Both expressions equal the product of the non-zero singular values, and reduce
to @f$|\det C|@f$ when @f$k=n@f$.

When @f$k<n@f$ and @f$C@f$ has full row rank, write

@f[
G=CC^\mathsf T,
\quad
\boldsymbol\alpha=\mathbf bC^\mathsf T G^{-1},
\quad
r^2=\|\mathbf b\|_2^2-
\boldsymbol\alpha(\mathbf bC^\mathsf T)^\mathsf T.
@f]

The generalized-volume gain of replacing row @f$i@f$ by @f$\mathbf b@f$ is

@f[
g_i^2=\alpha_i^2+r^2(G^{-1})_{ii}.
@f]

The residual term @f$r^2@f$ matters: even if the candidate row has zero
projection into the existing row space, it still scores high as long as it
brings a new orthogonal direction.

When @f$k>n@f$ and @f$C@f$ has full column rank, let

@f[
\boldsymbol\alpha=\mathbf bC^+,
\quad h_b=\|\boldsymbol\alpha\|_2^2,
\quad h_i=\bigl(CC^+\bigr)_{ii},
@f]

then the gain of replacing row @f$i@f$ is

@f[
g_i^2=(1-h_i)(1+h_b)+\alpha_i^2.
@f]

The rectangular path uses @f$\max_i g_i@f$ as the row score and performs exact
single-row swaps until the same threshold holds. Ordinary data initialization
in `ActiveSelector` typically takes @f$k=\min(m,n)@f$, so the square case is
still the most common for users; the rectangular formulas mainly serve the
low-level API, data-sparse regimes, and augmented selection.

## 4. Building the B Matrix from Atomic Configurations

### 4.1 Data Flow and Row-to-Configuration Mapping

The full data flow in this repository is:

```text
Atoms configuration pool
    │
    ▼
BMatrixBuilder ──► B rows + BRowRef metadata
    │
    ▼
full / stream / online MaxVol ──► active row matrix C
    │                                  │
    ├──► de-duplicated selected configs ├──► grade
    └──► score file                    └──► .sel state
```

The unit of selection is a **B row**; the unit of output is an **atomic
configuration**. One configuration may contribute multiple rows, and several
active rows may come from the same configuration; outputs are de-duplicated by
configuration index. Therefore:

- the number of active rows is usually close to the parameter/descriptor
  dimension @f$n@f$;
- the number of output configurations can be significantly smaller than the
  number of active rows;
- the configuration count of `selected.xyz` is not a user-specified fixed
  sample size.

`BRowRef` records the configuration owning a row, the local/global row index,
the row type, the atom index, and the component, keeping the mapping between
online swaps, state restore, and "row selection → configuration output"
consistent.

### 4.2 Descriptor Builders

`dtype=mtm`, `behler`, and `bondorder` use the
`DescriptorBuilder`. Let the local descriptor of atom @f$i@f$ be
@f$\mathbf d(\mathcal N_i)\in\mathbb R^n@f$ and the configuration atom count
be @f$N@f$; the common scaling factor is

@f[
s_N=\frac{w}{N^p},
@f]

where `weight`=@f$w@f$ and `pow-scale`=@f$p@f$.

In `mode=nbh`, each atom contributes one row:

@f[
\mathbf b_i=s_N\mathbf d(\mathcal N_i),
\qquad m_q=N.
@f]

In `mode=cfg`, a configuration contributes a single summed row:

@f[
\mathbf b_q=s_N\sum_{i=1}^{N}\mathbf d(\mathcal N_i),
\qquad m_q=1.
@f]

`nbh` is more sensitive to rare local environments but the total row count is
roughly the total atom count; `cfg` is cheaper in memory and grading cost, but
local anomalies may be averaged out or cancel in the sum.

### 4.3 Potential-Gradient Builder

`dtype=pot_efs` (alias `mtp_energy`) uses the `PotEFSBuilder`. Let
@f$\boldsymbol\theta@f$ be the potential parameters participating in the
selection; the following rows are generated with their non-zero-component
weights:

@f[
\begin{aligned}
\mathbf b_E       &\propto w_E\,\frac{\partial E}{\partial\boldsymbol\theta},\\
\mathbf b_{F,ia}  &\propto w_F\,\frac{\partial F_{ia}}{\partial\boldsymbol\theta},\\
\mathbf b_{S,ab}  &\propto w_S\,\frac{\partial S_{ab}}{\partial\boldsymbol\theta},\\
\mathbf b_{\text{site},i}
                   &\propto w_{\text{site}}\,
                     \frac{\partial E_i}{\partial\boldsymbol\theta}.
\end{aligned}
@f]

The number of rows per configuration is

@f[
m_q=
\mathbb 1_{w_E\ne0}
+3N\mathbb 1_{w_F\ne0}
+9\mathbb 1_{w_S\ne0}
+N\mathbb 1_{w_{\text{site}}\ne0}.
@f]

By default `w-energy=1` and all other component weights are 0, so each
configuration contributes exactly one energy-gradient row. For MTP, only the
linear coefficients are used by default; `all-coeffs=true` also includes
nonlinear coefficients such as the radial ones in the B matrix. Linear
coefficients usually already capture the diversity of the scalar basis and
significantly reduce @f$n@f$ and the selection cost.

### 4.4 Weighting and Atom-Count Scaling

With `row-scaling=uniform`, all enabled rows are multiplied by @f$w/N^p@f$ and
then by the corresponding `w-energy`, `w-force`, `w-stress`, or `w-site`.

`row-scaling=mlip` only changes the scaling of the local quantities of
`pot_efs`:

| Row type | `uniform` | `mlip` |
|---|---:|---:|
| energy | @f$w/N^p@f$ | @f$w/N^p@f$ |
| stress | @f$w/N^p@f$ | @f$w/N^p@f$ |
| force | @f$w/N^p@f$ | @f$w@f$ |
| site energy | @f$w/N^p@f$ | @f$w@f$ |

Weights change the geometry, and therefore the selection result and grades.
Grades from two states are directly comparable only when the builder type,
the potential parameter order, all weights, the scaling mode, and `pow-scale`
are identical.

## 5. Algorithm Design and Implementation

### 5.1 Module Responsibilities

| Class / entry | Responsibility |
|---|---|
| `MaxVol` | generic dense row selection, rectangular generalized volume, scoring, augmentation, and low-level checkpointing |
| `OnlineMaxVolState` | stores the compact active matrix; performs batched online swaps and allocation-free repeated scoring |
| `BMatrixBuilder` | converts `Atoms` into B rows plus `BRowRef` metadata |
| `ActiveSelector` | wires configurations, builder, selection backend, grading, MPI, and `.sel` state |
| `maxvol_select` | builds an active set from a candidate pool |
| `maxvol_sample` | read-only grading against an existing active set |
| `maxvol_extend` | incrementally augments the active set from a state or a training set |

### 5.2 Initialization Strategies

#### Data initialization

`init-mode=data` performs a column-pivoted Householder QR on @f$A^\mathsf T@f$;
the pivot columns correspond exactly to candidate rows of @f$A@f$. It is more
stable and reproducible than random initialization and provides a good initial
volume.

With `init-mode=auto` and a builder other than `PotEFSBuilder`, data
initialization is the default.

#### MLIP virtual-identity initialization

`init-mode=mlip` starts from

@f[
C_0=\delta I_n,
@f]

where `init-value`=@f$\delta@f$, default @f$10^{-6}@f$. Real candidate rows
progressively replace the virtual rows through MaxVol swaps. Parameter
directions not supported by the data may remain as virtual rows instead of
forcibly emitting meaningless configurations to fill the nominal @f$n@f$
directions.

Virtual rows have `global_row=-1` and never map to output configurations; the
command reports `virtual_rows` at the end. `init-mode=auto` selects this
strategy for `pot_efs` by default and data initialization for the other
builders. @f$\delta@f$ also sets the scale of the virtual directions; changing
it changes the initial grades, so keep the default unless you understand its
dimensions and scaling.

### 5.3 Classical Swaps and Fast Updates

Conceptual pseudocode of the square path:

```text
I = pivoted_qr_rows(A, n)
C = A[I, :]
factorize C and form R = A C^{-1}

repeat:
    (l, j, gain) = argmax |R[l, j]| over non-active rows
    if gain <= threshold:
        converged
    replace active row j by A[l, :]
    update C^{-1}, R and log-volume by low-rank formulas
until an iteration/swap cap is reached
```

Re-inverting and recomputing @f$AC^{-1}@f$ after every swap would be expensive.
The implementation uses the Sherman–Morrison formula to update the inverse and
the whole coefficient matrix. If the difference between the new and the old
row is @f$\mathbf d=\mathbf a_l-\mathbf c_j@f$, then

@f[
C'=C+\mathbf e_j\mathbf d,
@f]

so the rank-one formula updates @f$C^{-1}@f$, and the log-volume is updated
directly as

@f[
\log\operatorname{vol}(C')
=\log\operatorname{vol}(C)+\log|R_{lj}|.
@f]

### 5.4 Three ActiveSelector Backends

| Backend | Candidate history | Convergence behavior | Main storage | Use cases |
|---|---|---|---|---|
| `full` | complete B resident in memory | local threshold check against the current full matrix | several @f$m\times n@f$ workspaces | small/medium pools, baselines, debugging |
| `stream/exact` | complete B cached in memory or mmap | repeatedly rescans the full cache; threshold check over the whole pool within the scan cap | chunk cache + compact C | default, large pools, MPI |
| `stream/fast` | complete chunk cache | bounded passes and shortlists; may not converge | chunk cache + shortlist + C | throughput-oriented prefiltering |
| `online` | only the current C; rejected old rows are not kept | order-dependent; cannot prove the threshold over the cumulative historical pool | C, inverse/Gram factors, current batch | LOTF, append-only data, low memory |

"exact" in `stream/exact` refers to repeatedly scanning the complete cached
candidate pool and performing a full-pool post-check at the end — not to
finding the combinatorially global maximum volume. On the data-initialization
path each round receives one deterministic global best swap; the MLIP
virtual-identity initialization swaps in batches from a global shortlist
formed by one full-pool scan, then proceeds to the next full-pool scan. It and
`full` may produce different, but both threshold-satisfying, local solutions
because the initialization and swap paths can differ.

`stream-cache-mode` behavior:

- `auto`: memory first; new chunks move to mmap once the per-process
  `stream-memory-limit-mb` is exceeded;
- `memory`: memory only; errors when the limit is exceeded;
- `mmap`: everything in temporary mapped files;
- `off`: legacy streaming path that may rebuild descriptors every round;
  usually only for compatibility or fallback.

Temporary mmap files are unlinked immediately after mapping and reclaimed when
the process ends or the mapping is released. The directory is chosen in the
order `stream-cache-dir`, `TMPDIR`, `/tmp`.

### 5.5 OnlineMaxVolState

The online state stores @f$C@f$, row references, and factorizations, but not
the rejected historical candidates. B rows of each new configuration only
participate in swaps within the current batch; evicted active rows stay in the
current local pool and may re-enter during later swaps of the same batch.

After the first stream-based initialization, its row cache is released. Later
online updates build only appended configurations; they do not rebuild and
grade the historical pool. An update without new configurations does no
descriptor work. `compute_grades()` / `grade_all()` explicitly request pool
grading when needed. For a retained, complete stream cache, `compute_grades()`
reuses cached rows; a compact online state reconstructs them for this request.

When the current @f$C@f$ has fewer rows than columns, the two-pass
orthogonalized residual picks new independent directions; when it is square
and well conditioned, @f$C^{-1}@f$ is stored; for wide or rank-deficient
matrices, the regularized @f$(CC^\mathsf T+\lambda I)^{-1}@f$ is stored. This
lets the online mode gradually recover missing directions from data-poor
states.

The price: later swaps may make a long-rejected row extrapolative again, while
the online state cannot look it back up. If the threshold must hold over the
full cumulative pool before finalizing the training set, re-run
`stream/exact` or `full`.

### 5.6 Numerical Stability

The implementation includes the following guards:

- pivoted-QR initialization to avoid degenerate random starting points;
- SVD-based pseudo-inverse with truncation threshold about
  @f$\epsilon_{\rm mach}\max(k,n)\sigma_{\max}@f$;
- spectral-scale-dependent regularization for rank-deficient wide active
  matrices;
- full refactorization when the Sherman–Morrison denominator is non-finite or
  its absolute value is below @f$10^{-14}@f$ (the online square path uses
  @f$10^{-12}@f$);
- the low-level `MaxVol` forces a full recomputation every 20 fast updates by
  default;
- the online state samples factorization drift every 16 updates and forces a
  refactorization after at most 128 updates;
- well-conditioned square `MaxVol` states use pivoted LU; a conservative
  reciprocal-condition check retains SVD truncation for ill-conditioned or
  rank-deficient states;
- `log|det|` / generalized log-volume is kept as a sum of logarithms of LU
  pivots or singular values to avoid overflow or underflow;
- selection/grading is rejected when the input contains NaN/Inf, duplicate
  indices, or dimension mismatches.

For strictly rank-deficient matrices, the finite `log|det|` in the logs uses
singular-value floors or the regularization scale; it serves diagnostics only
and should not be read as a real non-zero determinant.

### 5.7 Convergence Status and Persistence

Full and cached-stream selection perform a full-pool post-check and record:

- `converged`: whether the largest remaining grade satisfies the threshold;
- `max_remaining_grade`: the largest configuration/row grade from the
  post-check;
- `termination_reason`: `threshold`, `max-swaps`, `fast-pass-limit`,
  `postcheck-failed`, or `online-approximate`, among others;
- `virtual_rows`: how many virtual rows remain in C after MLIP initialization.

After an incremental online update the historical pool has not been checked:
`converged=false`, `max_remaining_grade=inf`, and
`termination_reason=online-approximate` (or `max-swaps` when capped). Here
infinity means the global maximum is unknown, not a measured extrapolation
grade. Use explicit pool grading to inspect the result, or re-run `full` /
`stream/exact` to obtain a selection with a global convergence check. The
initial stream-based online construction still reports its full-pool check.

States that hit a cap without converging are still saved; later
`sample`/`extend` runs issue a warning. Do not rely only on "selected.xyz was
written" — check the `Selection status` and the maximum remaining grade.

The current `ActiveSelector` state format is v10; `.sel` files store the
compact active matrix, row metadata, training configurations, selected
configurations, builder metadata, scaling parameters, online progress, and
convergence information. Runtime factorization caches are rebuilt after
loading. States support checkpoint/resume but are not generic model files
across different potential parameterizations.

## 6. Characteristics and Limitations

### 6.1 Advantages

- **Directly related to the model linearization**: with gradient B matrices,
  the selection objective directly corresponds to parameter information;
- **pre-selection without labels**: descriptors or model gradients usually do
  not need DFT energies/forces of the candidates;
- **unified extrapolation criterion**: selection and online detection use the
  same volume gain;
- **interpretable**: every swap has an entering row, a leaving row, and a
  volume gain;
- **incrementally updatable**: the online state suits LOTF;
- **large-pool support**: cached streaming, mmap, OpenMP, and MPI reduce
  memory or build-time pressure;
- **strongly deterministic**: deterministic QR / global-winner rules for the
  same input order and run mode.

### 6.2 Limitations

- Returns a single-swap local optimum, not the combinatorially global maximum
  volume;
- sensitive to column scaling; unphysical or un-normalized columns can
  dominate the selection;
- the grade only measures extrapolation within the current representation and
  linearization; it cannot detect model-form errors;
- after nonlinear potential parameters change, the old gradient matrix may be
  invalid and must be rebuilt;
- `online` is order-dependent; `fast` may not converge because of the pass cap;
- very large @f$n@f$ incurs quadratic or even cubic linear-algebra cost;
- when one configuration contributes multiple rows, the number of selected
  rows and the number of output configurations are not in one-to-one
  correspondence.

For MTP, if only the linear coefficients change, old linear B rows remain
meaningful; if nonlinear parameters that affect the descriptors/gradients
(such as the radial basis) change, rebuild the active set completely instead
of merely appending to the old state.

## 7. Efficiency and Complexity

### 7.1 Key Dimensions

Performance is dominated by:

- @f$N_{\rm cfg}@f$: the number of configurations;
- @f$m=\sum_q m_q@f$: the total number of B rows;
- @f$n@f$: the number of B columns;
- @f$k@f$: the number of active rows, commonly @f$n@f$;
- @f$s@f$: the number of swaps;
- @f$q@f$: the number of grading scans over the full row cache;
- @f$T_{\rm build}@f$: the build time of neighbor lists, descriptors, or
  potential gradients.

On scientific-computing data, `pot_efs` gradient construction or complex
descriptor evaluation is often more expensive than a single MaxVol scan, so
"build descriptors once" matters as much as reducing @f$m@f$ and @f$n@f$.

### 7.2 Low-Level Square MaxVol

For @f$m\gg n@f$, @f$k=n@f$:

- QR initialization and the initial coefficient matrix: about
  @f$O(mn^2+n^3)@f$;
- each best-element search: @f$O(mn)@f$;
- each Sherman–Morrison update: @f$O(mn+n^2)@f$;
- periodic full refactorization: @f$O(mn^2+n^3)@f$;
- main working memory: @f$O(mn+n^2)@f$.

While fast updates are effective, the total cost is approximately

@f[
O(mn^2+s\,mn),
@f]

plus periodic full refactorizations. The rectangular path needs to reform the
relevant projections per generalized swap, typically about
@f$O(mnk)@f$, so it is heavier than the square rank-one update.

### 7.3 Backend Costs

Let one full grading scan multiply the @f$m\times n@f$ candidate matrix by an
@f$n\times k@f$ coefficient mapping:

| Backend | Approximate time | Main memory/disk |
|---|---|---|
| `full` | @f$T_{\rm build}+O(mn^2+s\,mn)@f$, plus refactorizations | several @f$O(mn)@f$ matrices in RAM |
| `stream/exact` | @f$T_{\rm build}+O(qmnk)@f$ | RAM/mmap cache about @f$8mn@f$ bytes plus @f$O(kn+k^2)@f$ |
| `stream/fast` | @f$T_{\rm build}+O(pmnk)@f$, @f$p@f$ bounded passes | same, plus a bounded shortlist |
| `online` | build and grade per batch; typical square grading @f$O(m_qn^2)@f$ | @f$O(n^2+m_qn)@f$; no historical pool |

The exact path with data initialization usually has @f$q\lesssim s+1@f$; the
MLIP shortlist path can accept multiple swaps per scan. The default exact
safety cap is @f$q=100@f$. These formulas give order-of-magnitude estimates
only. Eigen/MKL matrix products, builder cost, cache bandwidth, mmap I/O,
OpenMP scheduling, and MPI communication all affect wall-clock time.

A single @f$m\times n@f$ double matrix already needs @f$8mn@f$ bytes; e.g.
@f$m=10^6,n=1000@f$ is about 8 GB (~7.45 GiB), and `full` also keeps copies
and workspaces. For such sizes prefer `stream` + mmap/MPI, or reduce rows and
columns at the modeling level first.

### 7.4 Parallel Design

- **OpenMP**: when the builder is clonable, B-row construction and chunk
  grading parallelize over structures/chunks; internally the MKL local thread
  count is set to 1 to avoid excessive OpenMP/BLAS nesting;
- **MPI full build**: `add_candidates_mpi()` partitions B construction by
  configuration index; rank 0 reassembles the full B, selects, and broadcasts
  the compact grading state;
- **MPI stream**: configurations are distributed by
  `config_index % mpi_size`; each rank caches only its local B rows; exact
  mode reduces a deterministic global winner and broadcasts the entering row,
  avoiding the aggregation of the full matrix on rank 0;
- **MPI fast**: each rank submits a bounded shortlist, then a global shortlist
  is formed.

In MPI stream mode, the row cache ideally scales with the rank count, but all
ranks replicate the compact active matrix and factors. Exact mode still needs
one global-winner communication per swap, so with many swaps and little local
work per rank, communication limits scaling.

### 7.5 Performance Tuning Order

1. Confirm that `dtype` and the row types serve the goal; zero out unneeded
   force/site/stress weights;
2. for MTP keep the default linear coefficients and use `all-coeffs=true`
   only when actually needed;
3. use `mode=cfg` when configuration-level information suffices; use
   `mode=nbh` when local anomaly detection is needed;
4. use `stream/exact` for large pools; `auto` or `mmap` when memory is tight;
5. adjust `stream-chunk-size` so a chunk is large enough to amortize
   scheduling cost without excessive temporary peaks;
6. parallelize the builder with `--num_threads`; use MPI stream only for very
   large pools;
7. use `stream/fast` only for prefiltering, and check the final convergence
   status;
8. use `online` for strict low-memory LOTF paths, and re-check against the
   full cumulative pool at stage ends.

## 8. Usage

### 8.1 Build

```bash
make imrmlp
```

The default executable is `build/release/imrmlp`. Inspect the options actually
supported by the current version:

```bash
build/release/imrmlp maxvol_select --help
build/release/imrmlp maxvol_sample --help
build/release/imrmlp maxvol_extend --help
```

The settings file of the three commands is a positional argument; when not
given explicitly, the defaults are `select.ini`, `sample.ini`, and
`select_add.ini`. Shared options in the settings file should use the
`builder:` and `selector:` namespaces.

### 8.2 Selecting an Active Set from a Pool

A `select.ini` example for MTP scalar descriptors:

```ini
builder:dtype mtm
builder:potfile potential.mtp
builder:pottype mtp
builder:mode nbh
builder:row-scaling uniform

selector:threshold 1.001
selector:weight 1.0
selector:pow-scale 0.5
selector:init-mode auto
selector:max-swaps 0
selector:select-mode stream
selector:stream-mode exact
selector:stream-cache-mode auto
selector:stream-memory-limit-mb 3072
selector:stream-chunk-size 32
selector:verbose true
```

Run:

```bash
build/release/imrmlp maxvol_select select.ini \
  --pool pool.xyz \
  --output selected.xyz \
  --score-file maxvol_scores.dat
```

The command will:

1. write the de-duplicated selected configurations to `selected.xyz`;
2. write `idx grade selected` for every pool configuration to
   `maxvol_scores.dat`;
3. automatically save the state as `potfile + ".sel"`, here
   `potential.mtp.sel`;
4. print the number of selected configurations, log-volume, convergence
   status, maximum remaining grade, and virtual-row count;
5. with `verbose=true` and the stream backend, additionally print build,
   initialization, and scoring times, scan count, swap count, thread count,
   and RAM/mmap bytes.

To make the selection correspond directly to potential-parameter sensitivity,
switch to the gradient builder:

```ini
builder:dtype pot_efs
builder:potfile potential.mtp
builder:pottype mtp
builder:all-coeffs false
builder:row-scaling mlip
builder:w-energy 1.0
builder:w-force 0.1
builder:w-stress 0.0
builder:w-site 0.0

selector:init-mode auto
selector:init-value 1e-6
selector:select-mode stream
selector:stream-mode exact
selector:threshold 1.001
```

Here each configuration produces 1 energy row and @f$3N@f$ force rows.
`w-force=0.1` is only an example; the actual value should be consistent with
the training objective, units, and noise model — do not mechanically mix
default values of different dimensions.

### 8.3 Grading with an Existing Active Set

`maxvol_sample` does not modify the active set; it only writes configurations
whose grade exceeds the save threshold:

```bash
build/release/imrmlp maxvol_sample sample.ini \
  --potfile potential.mtp \
  --state-file potential.mtp.sel \
  --input candidates.xyz \
  --output sampled.xyz \
  --threshold-save 2.1 \
  --threshold-break 11.0
```

Default behavior:

- configurations with grade @f$>2.1@f$ are written to the output;
- output configurations get an `MV_grade` feature;
- sequential scanning stops at the first configuration with grade
  @f$>11.0@f$; later configurations are not processed;
- these two sampling thresholds are workflow policy, not the selection-time
  `selector:threshold=1.001`.

Grade with the same potential, builder, coefficient set, weights, and scaling
that were used for the state. In particular, do not generate a state with
`all-coeffs=false` and interpret it with `all-coeffs=true` B rows.

Inspect or export the state:

```bash
build/release/imrmlp maxvol_sample sample.ini \
  --state-file potential.mtp.sel --dump-state true

build/release/imrmlp maxvol_sample sample.ini \
  --state-file potential.mtp.sel \
  --extract-configs state_selected.xyz --selected-only true
```

### 8.4 Incremental Augmentation

Augmenting from an existing state is recommended:

```bash
build/release/imrmlp maxvol_extend select_add.ini \
  --potfile potential.mtp \
  --state-file potential.mtp.sel \
  --candidates new_pool.xyz \
  --output new_selected.xyz
```

The output only contains candidate configurations newly entering the active
set in this round; the state is updated in place. `--train` can add extra
training configurations beyond the state.

Without a state, a three-stage flow is available:

```bash
build/release/imrmlp maxvol_extend select_add.ini \
  --potfile potential.mtp \
  --train train.xyz \
  --candidates new_pool.xyz \
  --output new_selected.xyz
```

This mode first builds an active set from the training set, then grades the
new candidates, and finally re-stabilizes on the combined data. With
`select-mode=online` or `stream`, the implementation uses the corresponding
backend on the merged training+candidate sequence.

### 8.5 MPI Large-Pool Selection

Under an MPI build, the offline `maxvol_select` automatically uses the
distributed stream when running on multiple ranks with
`select-mode=stream`:

```bash
mpirun -np 4 build/mpi/imrmlp-mpi maxvol_select select.ini \
  --pool huge_pool.xyz --output selected.xyz
```

Shared outputs and the state are written by rank 0 only.
`stream-memory-limit-mb` is a **per-process** limit on every rank, not a total
across all ranks.

### 8.6 Low-Level C++ API

With a numeric matrix in hand, use `MaxVol` directly:

```cpp
#include "maxvol/maxvol.h"

Eigen::MatrixXd A = /* m x n design matrix */;

MaxVol mv;
mv.tolerance = 1e-3;       // stop at gain <= 1.001
mv.max_swaps = 0;          // no extra swap cap
mv.verbose = false;

const int k = std::min<int>(A.rows(), A.cols());
std::vector<int> selected = mv.select_rows(A, k);
Eigen::VectorXd row_scores = mv.compute_scores(A, selected);

double log_volume = mv.log_determinant();
const auto &history = mv.swap_history();
```

To grade only new matrices, first prepare the cache with the same active set
and the original matrix:

```cpp
mv.warm_cache(selected, A);
Eigen::VectorXd candidate_scores = mv.score_rows(B_candidate, selected);
```

`score_rows()` does not silently refactorize when the matching cache is
missing; it returns an empty vector when the cache is not prepared, the column
count changes, the active-row indices change, or the active-row values change.
The high-level `ActiveSelector::prepare_grading()` performs the corresponding
preparation.

## 9. Key Parameters

### 9.1 Builder Parameters

| Parameter | Default | Meaning |
|---|---:|---|
| `dtype` | `mtm` | `mtm`, `pot_efs`, `behler`, `bondorder` |
| `potfile` | `potential.mtp` | potential file; required by MTM and gradient builders |
| `pottype` | `mtp` | potential type |
| `mode` | `nbh` | per-atom or per-configuration descriptors; mainly for descriptor builders |
| `all-coeffs` | `false` | whether the MTP gradient includes all coefficients |
| `row-scaling` | `uniform` | `uniform` or `mlip` |
| `w-energy` | `1.0` | energy-gradient row weight; 0 disables the row |
| `w-force` | `0.0` | force-gradient row weight; 0 disables the row |
| `w-stress` | `0.0` | weight of the 9 stress-gradient rows; 0 disables them |
| `w-site` | `0.0` | site-energy-gradient row weight; 0 disables the row |

### 9.2 Selector Parameters

| Parameter | Default | Meaning and remarks |
|---|---:|---|
| `threshold` | `1.001` | user stop threshold @f$\tau@f$ |
| `weight` | `1.0` | common B-row weight |
| `pow-scale` | `0.5` | atom-count scaling exponent @f$p@f$ |
| `init-mode` | `auto` | `auto`, `mlip`, or `data` |
| `init-value` | `1e-6` | virtual-identity scale @f$\delta@f$ |
| `max-swaps` | `0` | extra swap cap; 0 means no such cap |
| `select-mode` | `stream` | `stream`, `full`, or approximate `online` |
| `stream-mode` | `exact` | `exact` or `fast` |
| `stream-chunk-size` | `32` | number of configurations per build/cache chunk |
| `stream-fast-passes` | `2` | maximum global scan rounds in fast mode |
| `stream-fast-shortlist` | `0` | 0 = automatic; otherwise the global shortlist size |
| `stream-cache-mode` | `auto` | `auto`, `memory`, `mmap`, `off` |
| `stream-memory-limit-mb` | `3072` | per-process resident row-cache limit |
| `stream-cache-dir` | empty | mmap temporary directory; empty uses `TMPDIR` or `/tmp` |
| `verbose` | `false` | print swap and stream performance statistics |

`max-swaps=0` only means there is no user-set swap cap. The low-level `MaxVol`
and the exact stream still have a default safety cap of 100 rounds; if the
post-check is not satisfied before that, the state is marked
`postcheck-failed`. The current CLI does not separately expose the low-level
`max_iterations`.

## 10. Result Interpretation and Recommended Workflows

### 10.1 Initial Training Set

1. generate a candidate pool of physically reasonable structures;
2. run label-free MaxVol selection with `stream/exact`;
3. label only `selected.xyz` with expensive DFT;
4. train the potential and check an independent validation set;
5. when rebuilding a gradient-based active set with this potential, keep the
   weights and the coefficient set traceable.

### 10.2 Iterative Active Learning

```text
initial pool ─► maxvol_select ─► DFT labeling / training
                                   │
                                   ▼
new trajectory / candidates ─► maxvol_sample ─► high-grade configurations
                                                   │
                                                   ▼
                                           DFT + maxvol_extend
                                                   │
                                                   └──► retrain and loop
```

In practice two policy thresholds are used: a low `threshold-save` collects
candidates and a high `threshold-break` immediately terminates untrustworthy
trajectories and triggers high-accuracy calculations. Calibrate the thresholds
via the error–grade correlation of the specific system; the defaults 2.1/11.0
are not universal physical constants for every material.

### 10.3 Reproducible Experiments: What to Record

- potential file version and coefficient order;
- `dtype`, `mode`, per-component weights, `row-scaling`, `pow-scale`;
- selection backend, threshold, initialization mode, and randomness-related
  settings;
- input configuration order, especially for `online`;
- MPI rank count, OpenMP thread count, and the stream cache backend;
- `converged`, `termination_reason`, maximum remaining grade, virtual-row
  count, and swap count;
- the `.sel` state and the final selected configurations.

## 11. FAQ and Troubleshooting

### 11.1 Why is the number of selected configurations smaller than the number of coefficients?

MaxVol selects rows, not configurations. In `nbh`, force, or site modes one
configuration contributes multiple rows; several active rows may come from the
same configuration, and the final output is de-duplicated. MLIP
initialization may also keep virtual rows that do not correspond to any real
configuration.

### 11.2 Why do two modes select different structures?

The maximum-volume subset may not be unique. `full` and `stream/exact` use
different initialization/swap paths; `online` is additionally order-dependent.
Compare whether the threshold is satisfied, the validation error, and the
coverage effect first, instead of requiring the indices to match one by one.

### 11.3 Why are all grades large?

Check, in order:

1. whether the state and the current builder agree on the column count and
   coefficient order;
2. whether the potential has changed nonlinear descriptors/radial parameters;
3. whether `weight`, `pow-scale`, `row-scaling`, and the component weights are
   consistent;
4. whether `init-value` was set too small;
5. whether the state is unconverged or contains many virtual directions;
6. whether the candidates genuinely contain new chemical environments not
   covered by the training set.

### 11.4 Why does the grade return an extreme value?

High-level grading uses `1e10` as a failure/strong-extrapolation sentinel when
a configuration generates no B rows, the cache is not prepared, dimensions
mismatch, or non-finite scores appear. Check whether the weights disable all
rows, whether the builder succeeded, and whether the state is compatible with
the current potential.

### 11.5 Why is `log|det|` very negative?

The volume carries units and scales exponentially with the row/column
scaling; with many rows, small unit scales, or virtual @f$10^{-6}I@f$ rows, a
negative log-volume is normal. It is only comparable for the same @f$k,n@f$,
the same builder, and the same scaling — do not use it as a universal quality
score across settings.

### 11.6 How do I know whether fast/online results can be used directly?

Look at the `Selection status`. If `termination_reason` is
`fast-pass-limit`, `online-approximate`, or `max-swaps`, the threshold over
the full historical pool is not proven. Hand the cumulative pool to
`stream/exact` for a final re-check.

## 12. Tests and Validation

The relevant Catch2 tests live at:

- `unitest/ut_maxvol.cpp`: square/rectangular selection, single-swap
  domination, scoring, caching, augmentation, serialization;
- `unitest/ut_online_maxvol.cpp`: online scoring workspaces, rank-deficient
  recovery, wide-matrix residuals, save/load resume;
- `unitest/ut_active_selector.cpp`: configuration mapping, compact grading
  state, stream/MPI high-level behavior.

After modifying the MaxVol core or the builders, run:

```bash
make test
```

Performance evaluation should at least record `build_seconds`,
`initialization_seconds`, `score_seconds`, `score_passes`, `swaps`, resident /
mapped bytes, and total time separately. Reporting only the total time makes
it hard to tell whether the bottleneck is descriptors, linear algebra, I/O,
or MPI communication.

`score_seconds` and `score_passes` include the final parallel pool check.
Scoring is split into row blocks independently of configuration/cache chunks,
so a single large configuration can use multiple workers. Each worker's
coefficient/Gram scoring intermediates share a 4 MiB budget (at least one row
per matrix); the active matrix, builder buffers, row cache, and output scores
are additional storage. `stream-memory-limit-mb` limits only the resident row
cache, not the process RSS. Linear MTP E/F/S builders use the bulk component
Jacobian; generic/nonlinear builders retain the component-gradient path.

## 13. Comparison with FPS and CUR

| Method | Selection criterion | Depends on the current active set | Typical use |
|---|---|---|---|
| MaxVol | maximize submatrix volume / single-row replacement gain | yes | D-optimal active learning and extrapolation detection |
| FPS | maximize the minimum descriptor distance to the selected set | yes | geometric uniform coverage, fixed-count sampling |
| CUR | SVD leverage scores | can be computed in one pass from the full matrix | representative downsampling and low-rank reconstruction |

`diverse_select --method=fps|cur` suits data-compression tasks where a fixed
number of structures is explicitly selected; the natural goal of
`maxvol_select` is that the active rows reach a local volume threshold, so its
output configuration count is not fixed. The three methods can be combined —
for example, first compress a huge pool with FPS/CUR, then build an active set
correlated with the potential-parameter linearization using MaxVol.

### 13.1 CUR Decomposition

CUR decomposition is an alternative row-selection strategy that complements
the MaxVol algorithm. While MaxVol maximizes the determinant of the selected
submatrix, CUR selects rows based on their **statistical leverage scores**
derived from the singular value decomposition (SVD).

#### Algorithm

Given a descriptor matrix @f$\mathbf{A} \in \mathbb{R}^{m \times n}@f$:

1. Compute the thin SVD: @f$\mathbf{A} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^T@f$
2. Choose the leverage rank @f$r = \min(k, n)@f$, where @f$k@f$ is the number
   of rows to select.
3. Compute leverage scores for each row @f$i@f$:
   @f[
   p_i = \frac{\|\mathbf{U}(i, 1:r)\|^2}{r}
   @f]
   where @f$\mathbf{U}(i, 1:r)@f$ denotes the first @f$r@f$ entries of the @f$i@f$-th
   row of @f$\mathbf{U}@f$.

4. Select @f$k@f$ rows using one of two strategies:
   - **Probabilistic**: weighted random sampling without replacement, where row @f$i@f$
     is sampled with probability proportional to @f$p_i@f$.
   - **Deterministic**: take the top @f$k@f$ rows sorted by leverage score (descending).

#### Reconstruction Guarantee

CUR decomposition with leverage-score sampling provides provable error bounds:

@f[
\|\mathbf{A} - \mathbf{C} \mathbf{C}^+ \mathbf{A}\|_F \leq (1 + \epsilon) \|\mathbf{A} - \mathbf{A}_k\|_F
@f]

where @f$\mathbf{C}@f$ is the selected row submatrix and @f$\mathbf{A}_k@f$ is the
best rank-@f$k@f$ approximation. This guarantee ensures that the selected rows capture
the dominant spectral information in the descriptor matrix.

#### Implementation

`cur_select_rows()` (in `maxvol/cur_selection.{h,cpp}`):

```cpp
CURResult cur_select_rows(
    const Eigen::MatrixXd& A,
    int k,
    bool deterministic = false,
    int seed = 42,
    bool compute_error = false
);
```

The `CURResult` struct contains:
- `selected_indices` — indices of the @f$k@f$ selected rows
- `scores` — leverage score for each row
- `reconstruction_error` — optional **relative** reconstruction error
  @f$\|\mathbf{A} - \mathbf{C}\mathbf{C}^+\mathbf{A}\|_F / \|\mathbf{A}\|_F@f$

#### CUR vs MaxVol

| Aspect | MaxVol | CUR |
|--------|--------|-----|
| **Criterion** | Maximize @f$\det(\mathbf{C})@f$ | Maximize leverage scores |
| **Algorithm** | Greedy row-swap iteration | SVD + score-based selection |
| **Determinism** | Deterministic | Deterministic or probabilistic |
| **Error bounds** | Empirical | Provable reconstruction guarantees |
| **Best for** | Well-conditioned active set | Representative subset of large pool |
| **Complexity** | @f$O(m \cdot n^2)@f$ per swap | @f$O(\min(m,n)^2 \cdot \max(m,n))@f$ (SVD) |
| **CLI access** | `maxvol_select`, `maxvol_sample` | `diverse_select --method cur` |

## 14. References and Code Entry Points

Theory references:

1. S. A. Goreinov, E. E. Tyrtyshnikov, N. L. Zamarashkin,
   "A theory of pseudoskeleton approximations," *Linear Algebra and its
   Applications* 261, 1–21 (1997),
   [DOI: 10.1016/S0024-3795(96)00301-1](https://doi.org/10.1016/S0024-3795(96)00301-1).
2. S. A. Goreinov, I. V. Oseledets, D. V. Savostyanov, E. E. Tyrtyshnikov,
   N. L. Zamarashkin, "How to Find a Good Submatrix," in *Matrix Methods:
   Theory, Algorithms and Applications*, 247–256 (2010),
   [DOI: 10.1142/9789812836021_0015](https://doi.org/10.1142/9789812836021_0015).
3. E. V. Podryabinkin, A. V. Shapeev, "Active learning of linearly
   parametrized interatomic potentials," *Computational Materials Science*
   140, 171–180 (2017),
   [DOI: 10.1016/j.commatsci.2017.08.031](https://doi.org/10.1016/j.commatsci.2017.08.031).

Main code entry points:

- `maxvol/maxvol.{h,cpp}`;
- `maxvol/online_maxvol.{h,cpp}`;
- `maxvol/b_matrix_builder.{h,cpp}`;
- `maxvol/active_selector.{h,cpp}`;
- `apps/maxvol_select.cpp`;
- @ref theory_training (training theory);
- @ref theory_mtp (MTP theory).

**See also:** @ref maxvol, @ref theory_training
