# MTP_M: Multiple Channels and Nonlinear Energy Heads {#theory_mtp_m}

[中文](mtp-m_zh.md)

@tableofcontents

MTP_M (also written MTP-M, implemented by `PotMTP_M` with potential type
`mtp_m`) builds trainable channels from shared MTP scalar invariants and maps
them to site energy through an energy head. It retains MTP locality and
symmetries while extending the dependence of energy on the descriptors.
This page describes the implementation in this repository; see
[MTP theory](mtp.md) for moment tensors, radial bases and contraction graphs.

## From ordinary MTP to MTP_M

An ordinary MTP has short-range site energy

@f[
E_i=\epsilon_{z_i}+\sum_{\alpha=1}^{P}\xi_\alpha
B_\alpha(\mathfrak n_i;\mathbf c).
@f]

Here @f$\mathfrak n_i@f$ is the local environment within the cutoff,
@f$z_i@f$ is the central species, @f$P@f$ is the number of scalar invariants,
and @f$\mathbf c@f$ contains shared radial parameters. With radial parameters
fixed, energy, forces and virial are linear in @f$\epsilon,\xi@f$.

MTP_M first projects the same invariants into @f$q@f$ channels:

@f[
Q_{ik}=b_{k,z_i}+\sum_{\alpha=1}^{P}A_{k\alpha}
B_\alpha(\mathfrak n_i;\mathbf c),\qquad k=1,\ldots,q,
@f]

then evaluates

@f[
E_i=\epsilon_{z_i}+F_{\boldsymbol\theta}(\mathbf Q_i),
\qquad E=\sum_i E_i.
@f]

@f$A@f$ is a trainable projection matrix, @f$b_{k,z_i}@f$ is a bias for each
channel and species, and @f$\boldsymbol\theta@f$ contains energy-head parameters.
The external species energy offset @f$\epsilon_{z_i}@f$ is added after the
head; it is a separate parameter block from the channel biases.

**Channel count is distinct from MTP level, tensor rank and body order.**
The integer `q` in `potential_tag` records the short-range channel count:
1 for `li`, 2 for `q2` and `fs`, and `n_input` for neural heads.
Channels are fitted internal variables with no prescribed charge, density or
orbital interpretation, and need not be independent.

All channels share radial functions, basic moments and the contraction graph;
only their final projection coefficients differ. Extra channels therefore do
not require complete copies of descriptor evaluation, but increase projection,
head evaluation and reverse-propagation costs.

## Five energy heads

These are short-range expressions, with species energy offsets outside @f$F@f$.
`pot:extmodel` selects the head; saved model metadata also records its type.

| `extmodel` | Channels | Energy head | Features and costs |
| --- | --- | --- | --- |
| `mtp_m_li` | 1 | @f$F=Q_1@f$ | Linear baseline; no extra head parameters |
| `mtp_m_q2` | 2 | Complete linear and quadratic terms in two channels | Explicit channel coupling; 5 head parameters; quadratic extrapolation can grow rapidly |
| `mtp_m_fs` | 2 | @f$F=Q_1+a\,\operatorname{asinh}(Q_2)@f$ | Linear term plus a smooth nonlinear correction; 1 head parameter |
| `mtp_m_n1` | `n_input` | One hidden layer with linear output | Adjustable channel count and hidden width; more parameters and harder optimization |
| `mtp_m_n2` | `n_input` | Two hidden layers with linear output | Deeper channel mixing; cost depends on both widths and benefits require separate validation |

### li: Relation to ordinary MTP

With @f$q=1@f$ and @f$F=Q_1@f$,

@f[
E_i=(\epsilon_{z_i}+b_{1,z_i})+
\sum_\alpha A_{1\alpha}B_\alpha.
@f]

For the same radial and scalar bases, this has the same energy function form
as ordinary MTP. File layouts need not match. The offsets @f$\epsilon_s@f$
and @f$b_{1,s}@f$ are also redundant: preserving their sum preserves predictions.

### q2: Quadratic channel coupling

The current implementation uses exactly two channels:

@f[
F=c_1Q_1+c_2Q_2+c_{11}Q_1^2+c_{22}Q_2^2+c_{12}Q_1Q_2.
@f]

It contains linear terms as well as squares, and is not a full quadratic
model with an arbitrary channel count. Substituting
@f$\mathbf Q=\mathbf b_s+A\mathbf B@f$ produces terms
@f$B_\alpha B_\beta@f$, but constrains their coefficients. Defining

@f[
H=\begin{pmatrix}c_{11}&c_{12}/2\\c_{12}/2&c_{22}\end{pmatrix},
\qquad C=A^{\mathsf T}HA,
@f]

shows that the quadratic coefficient matrix @f$C@f$ in scalar-basis space has
rank at most 2. This organizes nonlinear interactions through a small channel
space rather than fitting independent coefficients for every one of the
@f$P(P+1)/2@f$ scalar-basis products.

### fs: ArcSinh correction

Here `fs` means the implemented expression
@f$Q_1+a\operatorname{asinh}(Q_2)@f$; it must not be replaced by the square-root
embedding expression of a conventional Finnis–Sinclair model.
It accepts any real @f$Q_2@f$, with

@f[
\frac{\partial F}{\partial Q_1}=1,\qquad
\frac{\partial F}{\partial Q_2}=\frac{a}{\sqrt{1+Q_2^2}},\qquad
\frac{\partial^2F}{\partial Q_2^2}=
-\frac{aQ_2}{(1+Q_2^2)^{3/2}}.
@f]

The correction is smooth at zero, avoiding the square-root derivative's
singularity there. That fixes the derivative at one point; extrapolation
stability still follows from the radial functions, cutoffs and training
coverage, which validation error and grade probe separately.

### n1 / n2: Neural energy heads

The one-hidden-layer head uses

@f[
F=\mathbf v^{\mathsf T}\phi(W\mathbf Q+\mathbf d).
@f]

The two-hidden-layer head uses

@f[
F=\mathbf v^{\mathsf T}\phi\!\left(
W_2\phi(W_1\mathbf Q+\mathbf d_1)+\mathbf d_2\right).
@f]

These expressions use the channel vector directly, as in the current MTP_M
wrapper, which initializes the underlying two-hidden-layer network's input
normalization to the identity. Neither head has a separate output bias;
external parameters such as @f$\epsilon_s@f$ supply species constants.
`n1` supports TANH/RELU/SIGMOID/LINEAR and `n2` supports TANH/RELU/SIGMOID.

A nonlinear head can express more complex energy dependence on the same
local basis, but does not expand the cutoff. ReLU is not differentiable at
zero. For phonons, Hessians or higher derivatives, consider activation and
radial-cutoff smoothness; a numerical branch value does not remove that
mathematical lack of differentiability.

## Symmetry, locality and identifiability

@f$B_\alpha@f$ already has translation, rotation and same-species permutation
invariance, which the projection and scalar head preserve. Summing site
energies makes separated subsystems additive; forces and virial follow from
the same total energy. These statements require consistent species mapping
and geometry definitions; smoothness depends on the chosen basis and head.

**A nonlinear head does not introduce long-range physics.** Two environments with the same central species
and identical surroundings within the cutoff have identical site energies under a short-range
MTP_M. The head alone cannot distinguish charge distributions outside that
cutoff; LES/SOG requires a model containing those additional contributions.

Internal channels are generally not unique. Compatible scaling, mixing or
bias changes in the projection and head can preserve the energy function;
the available freedoms depend on the head. Compare fitted models through
energy, forces and target properties rather than requiring identical
@f$Q_k@f$ values or coefficients.

## Forces, virial and second derivatives

For each site, first propagate the head gradient to the scalar invariants:

@f[
g_{i\alpha}=\frac{\partial E_i}{\partial B_{i\alpha}}
=\sum_{k=1}^{q}\frac{\partial F}{\partial Q_{ik}}A_{k\alpha}.
@f]

Then propagate through the ordinary MTP contraction graph and radial basis.
The full-system force is

@f[
F_{j,a}=-\sum_i\sum_\alpha g_{i\alpha}
\frac{\partial B_{i\alpha}}{\partial x_{j,a}}.
@f]

Channel biases and species energy offsets have zero direct coordinate
derivative. Channel biases can still change nonlinear-model forces through
@f$\partial F/\partial\mathbf Q@f$. Short-range virial uses the same site-neighbor
derivatives; see [MTP forces and stress](mtp.md) for signs and units.

With @f$J_i=\partial\mathbf Q_i/\partial\mathbf x@f$, the site-energy
coordinate Hessian is

@f[
\nabla_{\mathbf x}^2E_i=
J_i^{\mathsf T}\left(\nabla_{\mathbf Q}^2F\right)J_i+
\sum_k\frac{\partial F}{\partial Q_{ik}}
\nabla_{\mathbf x}^2Q_{ik}.
@f]

The first term is the head's own curvature. It vanishes for `li` and generally
must be retained for nonlinear heads. Replacing ordinary MTP linear coefficients
with current @f$g_{i\alpha}@f$ alone does not give the complete Hessian.
Force-label training also needs mixed position/parameter derivatives; see
[training theory](training.md) and [HVP training](../advanced/hvp-training.md).

## Parameter layout and training implications

For @f$S@f$ species and @f$N_r@f$ radial parameters, the complete short-range
parameter count is

@f[
N_{\mathrm{par}}=N_r+S+qS+qP+N_{\mathrm{head}}.
@f]

| Parameters | Count | Implementation storage |
| --- | --- | --- |
| Shared radial parameters @f$\mathbf c@f$ | @f$N_r@f$ | `radial_coeffs` |
| Species energy offsets @f$\epsilon_s@f$ | @f$S@f$ | First @f$S@f$ entries of `species_coeffs` |
| Channel species biases @f$b_{k,s}@f$ | @f$qS@f$ | Remaining `species_coeffs`, ordered by channel |
| Channel projection @f$A_{k\alpha}@f$ | @f$qP@f$ | `moment_coeffs`, ordered by channel |
| Energy-head parameters @f$\boldsymbol\theta@f$ | @f$N_{\mathrm{head}}@f$ | `model_coeffs` |

This is the complete short-range layout. Freezing parameters reduces the vector
exposed to the optimizer. `sr_species` selects external species offsets;
`sr_moment` contains channel biases and projections; `sr_model` selects the
head and `sr_radial` selects radial parameters.

Fixing radial parameters only fixes @f$\mathbf B@f$; it does not make joint
training linear for every MTP_M head. Conditional linear blocks remain once
the nonlinear part is fixed as required by each head:

| Head | Affine coefficients eligible for `output` refitting |
| --- | --- |
| `li` | Species energy offsets, channel biases and projection coefficients |
| `q2` | Species energy offsets and all 5 quadratic-head coefficients |
| `fs` | Species energy offsets and the ArcSinh multiplier @f$a@f$ |
| `n1` / `n2` | Species energy offsets and final linear output weights |

The actual selection also depends on enabled parameter groups. “Linear” here
means **conditionally linear with other parameters fixed**, not joint linear
optimization of projections, hidden layers and radial parameters.
Use `fit:linear_refit output` with `trainnlpot`; see
[linear refitting](../advanced/linear-refit.md). Ordinary MTP native
[variable projection](../advanced/mtp-native-varpro.md) has a separate support
scope and cannot be applied to arbitrary MTP_M models.

`initmtpm` uses training data to initialize radial/projection parameters, fit
species energy offsets and adjust channel biases to center training-set
channel means. Channel variance and inter-channel correlation stay unconstrained,
so do not read the channels as a whitened basis. These statistics come from the
training set only, never from the final test set. Continue with nonlinear
optimization such as
`trainnlpot --do-lbfgs`, comparing model capacity, random initializations and
errors on independent validation data.

## Long-range models and deployment

MTP_M with LES/SOG adds a long-range functional to the short-range energy and
uses separate charge or latent-variable projection coefficients. The
@f$q@f$ short-range head channels and the SOG long-range channel count are
independent concepts. Long-range energy, response forces and cell derivatives
must all enter predictions; retaining only the short-range head is insufficient.
See [LES v2](../reference/les-v2.md) and [SOG](../reference/sog.md).

Supported short-range heads can be deployed independently on CPU/OpenMP or
GPU. Native compositions and long-range paths require compatible interfaces.
Neither a file suffix nor conversion to GPM changes the mathematical model or
executor support. See the [LAMMPS interface guide](../../LAMMPS/README.md).

Implementation sources: [channels, heads and derivatives](../../potentials/potential_mtp_m.h),
[factory and head types](../../potentials/potential_mtp_m.cpp), and
[initialization](../../apps/initmtpm.cpp). API examples are in the
[potential interface guide](../development/api-potential.md).
