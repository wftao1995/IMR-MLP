# Training Methodology {#theory_training}

[中文](training_zh.md)

@tableofcontents

## Choose a training route

Training adjusts model parameters to reproduce reference energies, forces and
stresses. Ordinary MTP has radial coefficients and linear coefficients; fixing
the radial part gives a linear least-squares problem. Neural and long-range
models may have additional trainable parameters.

| Task | Entry point |
| --- | --- |
| Fit ordinary MTP linear coefficients with a fixed radial basis | `imrmlp trainmtp train.ini --linear-fit` |
| Optimize ordinary MTP radial and linear coefficients | `imrmlp trainmtp train.ini --do-lbfgs` |
| Fit a model exposing the generic linear interface, such as `descriptor_linear` | `imrmlp train_linear train.ini` |
| Train MTP-M, NN-MTP or another supported nonlinear potential | `imrmlp trainnlpot train.ini --do-lbfgs` |

These examples assume an existing model and a settings file containing the
training data and model paths. `trainmtp` requires an explicit training or
diagnostic action. For a complete runnable example, start with the
[quickstart](../../examples/quickstart/README.md), then read the
[training workflow](../guides/workflows.md).

## MTP_M parameter chain and conditional linearity

MTP_M evaluates shared invariants, channel projections and an energy head.
Force-loss gradients must pass through first and second head derivatives and
mixed descriptor derivatives; a nonlinear head cannot be treated as fixed
linear coefficients. See [MTP_M theory](mtp-m.md) for channels, five heads,
Hessian chain rules and parameter groups.

Usually prepare an initial model with `initmtpm`, then optimize with `trainnlpot`.
To improve output coefficients while freezing hidden features, use
`fit:linear_refit output`; see [linear refitting](../advanced/linear-refit.md).
Ordinary MTP [native VarPro](../advanced/mtp-native-varpro.md) instead resolves
linear coefficients during radial trials. These paths have different support
scopes and optimization variables.

## Loss function and weights

For ordinary MTP, the E/F/S data loss is averaged over the number of training
configurations @f$M@f$:

@f[
\mathcal L_{\mathrm{data}} = \frac{1}{M}\sum_{s=1}^{M}
\left[W_{E,s}(E_s-E_s^{\mathrm{ref}})^2
+ W_{F,s}\sum_{i,a}(F_{s,i,a}-F_{s,i,a}^{\mathrm{ref}})^2
+ W_{S,s}\|S_s-S_s^{\mathrm{ref}}\|_F^2\right].
@f]

Here @f$E_s@f$ is total energy, @f$N_s@f$ is the atom count, @f$a@f$ runs over
three Cartesian components, and @f$S_s@f$ is the extensive stress tensor stored
internally in eV. The tensor norm sums **all nine entries**, counting both
members of each off-diagonal pair. A label contributes only when it is present
and its configured weight is positive.

With the optional fitness and force-dependent reweighting left at their defaults,
the effective weights are @f$W_{X,s}=w_X/N_s^{p_X}@f$:

| Setting | Default | Meaning |
| --- | --- | --- |
| `fit:energy_weight` | `1.0` | @f$w_E@f$, total-energy squared-error weight |
| `fit:force_weight` | `0.01` | @f$w_F@f$, force-component squared-error weight |
| `fit:stress_weight` | `0.0001` | @f$w_S@f$, extensive-stress squared-error weight |
| `fit:energy_pow` | `1.0` | @f$p_E@f$, atom-count normalization exponent |
| `fit:force_pow` | `0.0` | @f$p_F@f$, atom-count normalization exponent |
| `fit:stress_pow` | `1.0` | @f$p_S@f$, atom-count normalization exponent |

Thus the default energy loss uses @f$(\Delta E_s)^2/N_s@f$; this differs from
squaring a per-atom energy error, which gives @f$(\Delta E_s)^2/N_s^2@f$.
Weights balance quantities with different units and observation counts. A larger
force weight gives force labels more influence in the fit; it does not set a force-error
ceiling. After changing weights, compare energy, force and stress errors on the
independent validation set to see which weight meets the target.

Structure `fitness` scales energy observations; per-atom `atomic_fitness` scales
force observations. The optional `fit:force_wgt_*` settings apply additional
force-dependent reweighting. See the [configuration reference](../reference/configuration.md)
for these controls. Other training routes have their own objective options;
this formula describes the ordinary MTP E/F/S path.

## Gradients

The optimizer needs the derivative of the loss with respect to each trainable
coefficient. Energy derivatives pass through the radial functions and tensor
contractions. Force fitting also needs mixed position/parameter derivatives,
since force is the **negative** energy gradient:

@f[
F_{i,a}=-\frac{\partial E}{\partial x_{i,a}},\qquad
\frac{\partial F_{i,a}}{\partial c_k}
=-\frac{\partial^2 E}{\partial x_{i,a}\partial c_k}.
@f]

For example, one weighted force residual contributes
@f$2W_F(F_{i,a}-F_{i,a}^{\mathrm{ref}})\,\partial F_{i,a}/\partial c_k@f$
to the loss gradient. The implementation accumulates these derivatives without
requiring users to construct a full coordinate Hessian. Training against Hessian
or Hessian-vector-product labels is a separate option; see
[HVP training](../advanced/hvp-training.md).

## Regularization

Regularization discourages unnecessarily large coefficients or irregular radial
functions; it works with validation rather than replacing it. After raising
`fit:weight_L2`, check whether the independent validation error falls; a rise means
the penalty is too strong.

For ordinary MTP, `fit:weight_L1` and `fit:weight_L2` control penalties on the
linear **moment coefficients**, excluding the per-species constant energy offsets:

@f[
\mathcal L_{\mathrm{coeff}}=
\sum_{k\in\mathrm{moments}}\left(\lambda_{1,k}|c_k|+\lambda_{2,k}c_k^2\right).
@f]

The defaults are `fit:weight_L1 0` and `fit:weight_L2 1e-10`. The linear solve
may scale the L2 diagonal according to the design matrix; do not assume the
same numerical regularization value has identical effects in every trainer.

Radial penalties depend on `fit:radial_reg_type`. Mode `0` penalizes deviations
from radial-coefficient normalization and orthogonality. In mode `1`,
`fit:weight_rad_w0`, `fit:weight_rad_w1` and `fit:weight_rad_w2` penalize sampled
integrals of @f$r^2|f(r)|@f$, @f$r^2|f'(r)|@f$ and @f$r^2|f''(r)|@f$,
respectively. These are absolute-value penalties, not squared first-, second-
and third-derivative penalties.

## Optimizers

### L-BFGS

L-BFGS uses gradients and a limited history of updates to choose a search
direction, then performs a line search. It is a common starting point for MTP
fitting, selected with `--do-lbfgs`. The ordinary MTP command uses the ALGLIB
backend for this flag; other routes expose their own optimizer controls.

Structural relaxation and parameter training are different tasks. The FIRE
structural optimizer is not an automatic fallback for a failed training run.
If training fails, inspect the error, input geometry, initialization and loss
weights before restarting.

### SGD and learning-rate scheduling

`trainmtp --do-sgd` uses mini-batches and can reduce work per update on large
datasets. The following settings illustrate exponential learning-rate decay:

```text
fit:learning_rate 0.0001
fit:batch_size 100
fit:epoch_limit 100
fit:lr_scheduler exponential
fit:lr_decay_rate 0.001
```

Supported scheduler names are `none`, `step`, `exponential` and `cosine`.
`step` uses `lr = lr0 * rate^floor(t / decay_steps)`; `exponential` uses
`lr = lr0 * exp(-rate * t)`, where `t` counts mini-batch updates and `rate` is
`fit:lr_decay_rate`. Thus `0.95` in exponential mode does not mean multiplying
by 0.95 each step; it causes rapid decay. `fit:lr_decay_steps` controls only
the `step` interval. Warmup applies when a scheduler is enabled. Use the [configuration reference](../reference/configuration.md) for
scheduler and stopping controls.

## Read the reported errors

The training loss is weighted; the monitor's RMS errors describe unweighted
prediction errors. In the ordinary MTP training monitor:

| Quantity | Interpretation |
| --- | --- |
| `EATRMS` | RMS of each labeled structure's **per-atom** energy error |
| `FRMS` | RMS force-vector error per labeled atom, not RMS per Cartesian component |
| `SRMS` | RMS Frobenius-norm error per labeled extensive stress tensor |

For the same observations, component force RMS equals `FRMS / sqrt(3)`, and
nine-entry stress component RMS equals `SRMS / 3`. Ordinary MTP progress lines
print `EATRMS` and `FRMS` after multiplying their internal eV/atom and eV/Å values
by 1000; `SRMS` remains in eV. Check the output headings when comparing these
values with `calc_err`, other trainers or external tools.

## Validation and deployment

1. Split data before fitting; keep related trajectory frames together to reduce
   leakage between training and validation.
2. Fit using training labels. Use validation data to choose the basis, weights,
   regularization and stopping point.
3. Inspect energies, forces and stresses separately, including large errors and
   the configurations that produce them.
4. Evaluate a final test set after model selection, then check the intended
   simulation conditions before deployment.

Providing `--validset` to ordinary `trainmtp` enables error monitoring; it does
not enable automatic validation-based early stopping. Keep a final test set
separate from any data used to choose hyperparameters. Active learning can add
unfamiliar structures to a candidate pool, but reference labels must be obtained
before those structures can improve a supervised fit.

For implementation details, see @ref train. For linear least squares, see
@ref theory_linear; for active-set selection, see @ref theory_maxvol.
