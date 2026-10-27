# Linear Potential Training {#theory_linear}

[中文](linear_zh.md)


@tableofcontents

## Overview

For potentials whose energy is a linear function of the regression coefficients,
the training problem reduces to solving a linear least-squares system. This applies
to Moment Tensor Potentials (MTP) with fixed radial coefficients, where the
site energy is expressed as:

@f[
E_i = \sum_{\alpha} \xi_\alpha B_\alpha(\mathbf{r}_i)
@f]

with linear coefficients @f$\xi_\alpha@f$ and moment tensor basis functions
@f$B_\alpha@f$.

Linear training is typically much faster than nonlinear optimization and
solves the regularized least-squares problem with one assembly and solve,
without iterative gradient descent. Numerical accuracy still depends on the
conditioning of the data and solver.

The equations below describe generic linear least squares. The implementation
details refer to `TrainerLinearPotential`, used by `train_linear` and Python's
`LinearTrainer` for models such as `descriptor_linear`. Ordinary MTP uses its
dedicated `trainmtp --linear-fit` path; it does not implement the generic
`supports_linear_fit()` interface.

For MTP_M, fixing radial parameters does not make joint fitting of nonlinear
heads and projections linear. With the remaining parameters fixed, `li/q2/fs/n1/n2`
each expose supported affine output blocks for
[conditional linear refitting](../advanced/linear-refit.md) through `trainnlpot`.
See [MTP_M theory](mtp-m.md) for each head's coefficients and derivation.

## Normal Equations

The training objective is to minimize the weighted sum of squared errors
with L2 regularization:

@f[
\mathcal{L}(\mathbf{w}) = \frac{1}{2}\|\mathbf{A}\mathbf{w} - \mathbf{y}\|_W^2
+ \frac{\lambda}{2}\|\mathbf{w}\|^2
@f]

where @f$\mathbf{A} \in \mathbb{R}^{d \times p}@f$ is the design matrix
(@f$d@f$ data rows, @f$p@f$ parameters), @f$\mathbf{y}@f$ is the target
vector, @f$W@f$ is the diagonal loss-weight matrix, and @f$\lambda@f$ is the
regularization strength.

@f$\mathbf{A}@f$ and @f$\mathbf{y}@f$ are unweighted. Each observation with
row @f$\mathbf{a}@f$, target @f$y@f$ and weight @f$w@f$ contributes
@f$w\mathbf{a}^T\mathbf{a}@f$ to the Gram matrix and
@f$w\mathbf{a}^Ty@f$ to the right-hand side. Equivalently, multiply both the
row and target by @f$\sqrt{w}@f$; do not multiply rows by @f$w@f$ again.

The minimizer satisfies the normal equations:

@f[
(\mathbf{A}^T \mathbf{W} \mathbf{A} + \lambda \mathbf{I}) \mathbf{w} = \mathbf{A}^T \mathbf{W} \mathbf{y}
@f]

Denoting @f$\mathbf{G} = \mathbf{A}^T \mathbf{W} \mathbf{A}@f$ (the Gram matrix) and
@f$\mathbf{b} = \mathbf{A}^T \mathbf{W} \mathbf{y}@f$, the solution is:

@f[
\mathbf{w}^* = (\mathbf{G} + \lambda \mathbf{I})^{-1} \mathbf{b}
@f]

## Design Matrix Assembly

The design matrix @f$\mathbf{A}@f$ is assembled row-by-row from training
configurations. Each configuration contributes three types of rows:

### Energy Rows (Structure-Level)

For each training configuration, the total energy contributes one row of the
design matrix. For a linear model @f$E = \sum_\alpha \xi_\alpha B_\alpha@f$,
the derivative with respect to coefficient @f$\xi_k@f$ is simply the basis
function value:

@f[
\mathbf{A}_{\text{eng}} = \sum_{i=1}^{N_{\text{atoms}}}
B_k(\mathbf{r}_i)
@f]

Its loss weight is `energy_weight` (default 1.0) divided by
@f$N_{\text{atoms}}^{\text{energy_pow}}@f$, multiplied by `atom.fitness`.

### Force Rows (Per-Atom, Per-Cartesian-Component)

For each atom @f$i@f$, the force gradient contributes @f$3@f$ rows:

@f[
\mathbf{A}_{\text{force}, i, a} = \frac{\partial F_{i,a}}{\partial \xi_k}
@f]

where @f$F_{i,a} = -\partial E / \partial r_{i,a}@f$. The analytical derivative
requires the descriptor Jacobian and backpropagation through the moment tensor
contraction tree. Weighted by `force_weight` (default 0.01) divided by
@f$N_{\text{atoms}}^{\text{force_pow}}@f$, multiplied by the corresponding
`atomic_fitness` entry when present. These are loss weights, not row multipliers.

### Stress Rows (Structure-Level)

The generic trainer uses all 9 entries of the extensive stress tensor (eV),
not 6 Voigt components. Off-diagonal entries are counted separately. For
feature @f$k@f$ and Cartesian components @f$a,b@f$, the unweighted row is:

@f[
\mathbf{A}_{\text{stress}, ab} = -\sum_i \sum_j
\frac{\partial B_k(\mathbf{r}_i)}{\partial r_{ij,a}} r_{ij,b}
@f]

Here @f$\mathbf{r}_{ij}@f$ is the neighbor displacement from center @f$i@f$.
The loss weight is `stress_weight` (default 0.0001) divided by
@f$N_{\text{atoms}}^{\text{stress_pow}}@f$.

## Weighting Scheme

User-configurable weights balance the contributions of different data types:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `energy_weight` | 1.0 | Energy loss weight before normalization |
| `force_weight` | 0.01 | Force loss weight before normalization |
| `stress_weight` | 0.0001 | Stress loss weight before normalization |
| `energy_pow` | 1.0 | Exponent for @f$N_{\text{atoms}}@f$ scaling of energies |
| `force_pow` | 0.0 | Exponent for @f$N_{\text{atoms}}@f$ scaling of forces |
| `stress_pow` | 1.0 | Exponent for @f$N_{\text{atoms}}@f$ scaling of stresses |
| `weight_L2` | 1e-10 | L2 regularization strength @f$\lambda@f$ |

In this generic trainer, `atom.fitness` scales energy observations only;
`atomic_fitness` scales each atom's force observations. Neither scales stress
observations. Missing labels and observations with zero weight are skipped.

## Regularization

L2 (Tikhonov / weight decay) regularization adds @f$\lambda \mathbf{I}@f$ to
the Gram matrix, which:

- Prevents overfitting to noise in the training data.
- Makes a rank-deficient normal matrix positive definite when the penalty is
  positive on every coefficient; the numerical conditioning still depends on
  the data scale and penalty strength.
- Biases the solution toward smaller coefficient norms.

The regularization strength @f$\lambda@f$ is controlled by `weight_L2` (default @f$10^{-10}@f$).

## TrainerLinearPotential

`TrainerLinearPotential` (in `train/train_linear_potential.{h,cpp}`) orchestrates
the linear training pipeline:

1. **`init()`** — Verifies that the potential supports linear fitting, allocates
   the Gram matrix @f$\mathbf{G}@f$ and the right-hand side @f$\mathbf{b}@f$.
2. **`Assemble()`** — Iterates over all training configurations:
   - Computes per-atom descriptors and their parameter gradients via
     `linear_compute_descriptor_grad()`.
   - Accumulates @f$\mathbf{G}@f$ and @f$\mathbf{b}@f$ from weighted observations.
   - Supports MPI for distributed assembly across nodes.
3. **`Solve()`** — Solves the regularized normal equations:
   - Primary solver: LDLT decomposition (Cholesky-like, efficient for symmetric
     positive-definite @f$\mathbf{G} + \lambda\mathbf{I}@f$).
   - Fallback: full-pivot LU decomposition if LDLT fails.
   - Broadcasts the solution vector @f$\mathbf{w}^*@f$ to all MPI ranks.
4. **`Train()`** — `Assemble()` + `Solve()`, saves the model, then evaluates an
   optional held-out validation set configured through the C++ interface.

## Validation

The generic linear trainer performs one assembly and solve, not an iterative
optimization with early stopping. The C++ validation path evaluates the fitted
model once. Python's `LinearTrainer.train()` accepts training data only;
evaluate held-out structures separately. Compare validation errors when choosing
regularization, and reserve final test data for evaluation after selection.

## Comparison with Nonlinear Training

| Aspect | Linear | Nonlinear |
|--------|--------|-----------|
| **When to use** | Linear models; MTP with fixed radial coefficients | MTP radial fitting, NN-MTP, other nonlinear models |
| **Solution** | Linear system solve (subject to numerical conditioning) | Iterative (L-BFGS, SGD) |
| **Speed** | Fast (single assembly + solve) | Slower (requires many epochs) |
| **Regularization** | L2 via normal equations | L1/L2 via objective function |
| **Derivatives** | Analytical design-matrix assembly | Parameter gradients, including radial/tensor or neural derivatives |
| **Training class** | `TrainerLinearPotential` (generic); `Trainer_MTP` (MTP) | `Trainer_MTP`, `TrainNonlinear` |

## Classes Summary

| Class | Module | Role |
|-------|--------|------|
| `TrainerLinearPotential` | @ref train | Generic linear least-squares trainer, e.g. for `descriptor_linear` |
| `ErrorMonitor` | @ref train | Training/validation error tracking |

**See:** @ref theory_mtp, @ref theory_training
