# Moment Tensor Potentials (MTP) {#theory_mtp}

[中文](mtp_zh.md)


@tableofcontents

For practical model generation and fitting, start with the
[training workflow](../guides/workflows.md). This page explains ordinary MTP
energies, symmetries, bases and derivatives. Trainable channel projections and
nonlinear energy heads are covered in [MTP_M theory](mtp-m.md).

## References

1. **Wang, J. et al.** "Efficient moment tensor machine-learning interatomic potential
   for accurate description of defects in Ni-Al Alloys." *Phys. Rev. Materials* 9,
   053805 (2025). DOI: [10.1103/PhysRevMaterials.9.053805](https://doi.org/10.1103/PhysRevMaterials.9.053805)
2. Shapeev, A. "Moment Tensor Potentials: A Class of Systematically Improvable
   Interatomic Potentials." *Multiscale Model. Simul.* 14, 1153-1173 (2016).
3. Novikov, I. et al. "The MLIP package: moment tensor potentials with MPI and
   active learning." *Mach. Learn.: Sci. Technol.* 2, 025002 (2021).
4. Gubaev, K. et al. "Machine learning molecular dynamics: A concise review."
   *Phys. Status Solidi B* 255, 1700359 (2018).

## Overview

Moment Tensor Potentials (MTP) are a class of systematically improvable interatomic
potentials introduced by Shapeev (2016). The energy of an atom is expressed as a
linear combination of scalar invariants built from moment tensor descriptors:

@f[
E=\sum_i E_i,\qquad
E_i = \epsilon_{z_i}+\sum_{\alpha} \xi_\alpha B_\alpha(\mathfrak n_i;\mathbf c)
@f]

where @f$\mathfrak n_i@f$ is the local environment of atom @f$i@f$,
@f$B_\alpha@f$ are moment tensor invariants (scalar basis functions),
@f$\mathbf c@f$ contains radial parameters, @f$\xi_\alpha@f$ are moment
coefficients, and @f$\epsilon_{z_i}@f$ is the central species energy offset.
“Linear MTP” refers to this combination with fixed radial parameters;
joint optimization of radial parameters remains nonlinear.

Rotational invariance is obtained by forming scalar basis functions from complete
contractions of symmetric Cartesian moment tensors. This implementation does not
use Clebsch-Gordan coefficient contractions for the MTP basis; Clebsch-Gordan
coefficients appear in other descriptors such as bispectrum descriptors.

## Locality, symmetry and model capacity

The local environment consists of relative positions and species within the cutoff:

@f[
\mathfrak n_i=\{(\mathbf r_{ij},z_j):
0<|\mathbf r_{ij}|<r_{\mathrm{cut}}\},\qquad
\mathbf r_{ij}=\mathbf x_j-\mathbf x_i.
@f]

Relative positions remove overall translation; neighbor sums make same-species
relabeling irrelevant; complete tensor contractions make energy rotationally
invariant. Total energy is a sum of site energies, so separated noninteracting
subsystems are additive. Site energy is a model decomposition, not a unique
atomic-energy label that training data must provide.

The cutoff controls spatial reach, while level, radial basis size and body-order
constraints control local-function capacity. A higher level does not recover
information outside the cutoff, and a nonlinear head does not introduce
long-range interactions by itself. Continuity of energy, forces and higher
derivatives also depends on radial-cutoff and head smoothness.

Two low-order invariants illustrate the transition from distances to angles:

@f[
M_{\mu,0}=\sum_j f_\mu(r_{ij}),\qquad
\mathbf M_{\mu,1}=\sum_j f_\mu(r_{ij})\widehat{\mathbf r}_{ij},
@f]

@f[
\mathbf M_{\mu,1}\cdot\mathbf M_{\lambda,1}
=\sum_{j,k}f_\mu(r_{ij})f_\lambda(r_{ik})
\cos\theta_{jik}.
@f]

The first depends only on distances; the dot product introduces angles between
neighbors. The double sum includes @f$j=k@f$, so it has up to three-body
dependence rather than being a pure three-body term. Higher contractions
combine radial and angular information in the same manner.

## Moment Tensor Construction

### Elementary Moment Tensors

For an atom @f$i@f$ with neighbors @f$j@f$ at positions
@f$\mathbf{r}_{ij} = (x_{ij}, y_{ij}, z_{ij})@f$, an elementary Cartesian tensor
component with exponents @f$(a,b,c)@f$ is evaluated as:

@f[
M_{\mu,a,b,c}(\mathbf{r}_i) = \sum_{j} f_\mu(|\mathbf{r}_{ij}|) \cdot
\frac{x_{ij}^a \, y_{ij}^b \, z_{ij}^c}{|\mathbf{r}_{ij}|^\nu}
@f]

where @f$a + b + c = \nu@f$, and @f$f_\mu(r)@f$ is the radial function. The
normalization by @f$|\mathbf{r}_{ij}|^\nu@f$ makes the Cartesian factor depend
only on bond direction; distance dependence remains in @f$f_\mu(r)@f$. In potential files and code, each elementary component is stored
as an `alpha_index_basic` row `(mu, a, b, c)`.

### Radial Basis Functions

The radial functions are expanded in the basis selected by the model:

@f[
f_\mu(r) = \sum_{k} c_{\mu, z_i, z_j}^{k} \, Q^{k}(r)
@f]

where @f$Q^{k}(r)@f$ are the selected radial basis functions, including the
configured cutoff and coordinate transform, and @f$c_{\mu, z_i, z_j}^{k}@f$ are
species-dependent radial coefficients. Chebyshev-based choices do not imply
orthonormality after a cutoff or nonlinear coordinate transform is applied.
See the [radial-basis reference](../reference/radial-basis.md) for available types.

The code supports multiple radial basis types via the `AnyBasis` hierarchy:
- `RBasis_PolyB` — polynomial basis: @f$\phi_\alpha(r) = (r_c - r)^{\alpha+2} / N_\alpha@f$
- `RBasis_ChebyshevConfigurable` — configurable cutoff, coordinate transform,
  and Chebyshev/one-minus-Chebyshev form

**See:** @ref basis

### Recursive Cartesian Contraction

Non-elementary basis values are built recursively as products and sums of lower-level
moment values generated by Cartesian index contractions. If @f$b_k@f$ and @f$b_l@f$
are lower-level intermediate values, a higher-level intermediate value is:

@f[
b_j = \sum_{k,l} c_{jkl} \cdot b_k \cdot b_l
@f]

where @f$c_{jkl}@f$ are integer multiplicity factors produced by the contraction
enumeration. The final MTP basis functions are scalar entries selected from this
intermediate value array, so they are invariant under rigid rotations.

The decomposition produces a contraction tree stored in:
- `alpha_index_basic` — elementary moment tensor specifications (mu, a, b, c)
- `alpha_index_times` — multiplication/contraction rules {id1, id2, multiplier, output_id}
- `alpha_moment_mapping` — mapping from all moments to scalar invariants

### Level Structure

Basis functions are organized by levels. The level of a moment tensor is defined as
@f$\operatorname{lev}(M_{\mu,\nu}) = w_\mu \mu + w_\nu \nu + w_d@f$, where @f$w_\mu, w_\nu, w_d@f$
are configurable weight parameters. In `mtmbasis`, the current defaults are
@f$w_\mu = 2@f$, @f$w_\nu = 1@f$, and @f$w_d = 2@f$.
For a scalar basis function built from @f$n@f$ moment tensors,
@f$\operatorname{lev}(B_\alpha) = \sum_{i=1}^n \operatorname{lev}(M_{\mu_i,\nu_i})@f$.

All basis functions with level @f$\leq d@f$ (the level threshold) form the basis set.
Additional constraints can be applied:
- Maximum many-body order @f$\text{nbody}_{\max}@f$
- Maximum polynomial degree @f$\text{npoly}_{\max}@f$
- Maximum moment tensor rank @f$\nu_{\max} \leq 3@f$ (recommended, since a @f$\nu@f$-rank
  tensor has @f$(\nu+1)(\nu+2)/2@f$ independent components, growing as @f$O(\nu^2)@f$)

These weights are exposed as the top-level `mu_wei`, `nv_wei`, and `dim_wei`
settings for `imrmlp mtmbasis`.

## Scalar Basis Generation

### Contraction Index Matrix

Given a set of moment tensors @f$\{M_{\mu_1,\nu_1}, \ldots, M_{\mu_n,\nu_n}\}@f$, the
contraction rule is uniquely specified by an @f$(n+1) \times (n+1)@f$ index matrix:

@f[
\boldsymbol{\alpha} = \begin{pmatrix}
0 & \mu_1 & \mu_2 & \cdots & \mu_n \\ \hline
\mu_1 & 0 & \alpha_{12} & \cdots & \alpha_{1n} \\
\mu_2 & \alpha_{12} & \ddots & \ddots & \alpha_{2n} \\
\vdots & \vdots & \ddots & \ddots & \vdots \\
\mu_n & \alpha_{1n} & \alpha_{2n} & \cdots & 0
\end{pmatrix}
@f]

The lower-right @f$n \times n@f$ submatrix @f$\tilde{\boldsymbol{\alpha}}@f$ encodes the
contractions: @f$\tilde{\alpha}_{ij}@f$ specifies how many indices of @f$M_i@f$ and @f$M_j@f$
are contracted. The rank-sum constraint @f$\sum_j \tilde{\alpha}_{ij} = \nu_i@f$ (for all
@f$i@f$) ensures the result is a scalar. Since moment tensors are fully symmetric,
the contraction result is independent of which specific indices are chosen.

**Example:** For @f$\{M_{1,3}, M_{2,2}, M_{3,1}\}@f$ with
@f$\tilde{\boldsymbol{\alpha}} = \begin{pmatrix} 0 & 2 & 1 \\ 2 & 0 & 0 \\ 1 & 0 & 0 \end{pmatrix}@f$,
the scalar basis is @f$B_\alpha = (M_{1,3})_{ijk} (M_{2,2})_{ij} (M_{3,1})_k@f$.

### Equivalence Checking

Two index matrices @f$\boldsymbol{\alpha}@f$ and @f$\boldsymbol{\beta}@f$ represent
equivalent contraction rules iff there exists a permutation matrix @f$\mathbf{P}@f$
that reorders the tensor factors (leaving the auxiliary first row/column fixed) such that
@f$\boldsymbol{\beta} = \mathbf{P} \boldsymbol{\alpha} \mathbf{P}^\mathrm{T}@f$.
The current implementation compares exact canonical representatives under
simultaneous permutations of the tensor factors. It first rejects mismatched
radial-index and rank lists, then compares the canonical integer contraction
matrices. The number of factors is limited to seven, making permutation
enumeration practical. Equal eigenvalues alone do not establish equivalence.

### Computational Path: Tree Decomposition

A scalar basis function @f$B_\alpha@f$ is computed by decomposing it into a binary tree
of intermediate tensor contractions. The root is the scalar output, leaves are elementary
moment tensors, and each internal node is a contraction of two child tensors:

@f[
b_n = \sum_i d_i \cdot b_{k_i} \cdot b_{l_i}
@f]

where @f$d_i@f$ is the symmetry degeneracy factor, and @f$(n, k_i, l_i)@f$ are the
multiplication rules stored in `alpha_index_times`.

The decomposition is not unique — different strategies produce different numbers of
stored moment components @f$n_b@f$ and multiplication rules @f$n_t@f$, directly
impacting computational cost. For a typical level-28 basis set, the number of possible
decomposition paths exceeds @f$10^{2195}@f$, making exhaustive search impossible.

### Contraction Layout Search

The top-level `mode` setting selects the `mtmbasis` search method: `0` is the
default tempering search, `1` is random/local search, `2` is a genetic algorithm
(GA), and `3` is coordinate descent. These are heuristic searches: the layout depends
on the search budget and the seed, so compare structural counts (scalars and product
rules) after changing `mode` or increasing the budget. The GA
follows this outline [Wang et al., 2025]:

1. **Enumeration**: For each scalar function @f$B_\alpha@f$, enumerate all valid
   decomposition trees @f$\{T_i^\alpha\}@f$.
2. **Encoding**: Each individual is a gene sequence @f$\{k_1, \ldots, k_n\}@f$ where
   @f$k_i@f$ selects the decomposition tree for the @f$i@f$-th basis function.
3. **Cost Function**: Minimize @f$F(R)=w_1 n_b+w_2 n_t@f$, where @f$n_b@f$ counts
   stored moment components and @f$n_t@f$ counts multiplication rules. The top-level
   settings `w1` and `w2` control these costs; their current defaults are `1` and `4`.
4. **Fitness**: Sigmoid-type function
   @f$f(R) = 1 - 1/(1 + e^{-k(F(R) - F_{\min}) + c})@f$,
   where @f$F_{\min}@f$ tracks the best solution found.
5. **Operations**: Mutation replaces one or more genes; crossover copies selected
   gene positions from another parent. Elite retention requires `elite_count > 0`;
   its default is `0`, so elite retention is not enabled automatically.
6. **Dynamic Pruning**: Each intermediate tensor must have a unique decomposition
   throughout the entire computational graph. When processing basis functions in order,
   tree branches inconsistent with previously fixed decompositions are pruned, enforcing
   global consistency while maintaining search space efficiency.

**Initialization** normally filters for minimal dimensions and then scores child
reuse and equal-size decompositions. `randominit` selects a random initial guess.
For GA, `initial_population` defaults to `legacy`; `diverse` enables an optional
diverse initial population.

### Implementation

| Class/Method | Role |
|-------------|------|
| `PickTensors::gen_alpha_pool_by_shapeev_level()` | Generate basis set by level threshold |
| `PickTensors::gen_alpha_basic_index()` | Build elementary moment tensor list |
| `PickTensors::gen_alpha_index_times()` | Build contraction multiplication rules |
| `alpha_index_type` struct | Index matrix with exact permutation-based equivalence check |
| `calc_level()` | Compute level of a moment tensor with configurable weights |

`imrmlp mtmbasis` generates the alpha pool with
`PickTensors::gen_alpha_pool_by_shapeev_level()`, then searches the contraction
tree layout with `PickTensors::gen_pot_by_alphapool_GA()`. The command writes
candidate basis templates as `potential.mtp_<iteration>` and diagnostic files
for the selected contraction graphs.

**See:** @ref basis, `mtmbasis/basisgen.cpp`

## Coefficient Layout

The regression coefficient array has the following layout:

```
[radial coeffs | species coeffs | moment coeffs]
```

| Segment | Size | Description |
|---------|------|-------------|
| `radial_coeffs` | `species_count^2 * radial_func_count * rb_size` | Species-pair radial coefficients @f$c_{\mu,z_i,z_j}^{k}@f$ |
| `species_coeffs` | `species_count` | Per-species constant energy offset |
| `moment_coeffs` | `alpha_scalar_moments` | Linear coefficients of scalar invariants |

## Force and Stress Computation

### Force via Backpropagation

Forces are the negative energy derivatives with respect to atomic positions:

@f[
F_{j,a}=-\frac{\partial E}{\partial x_{j,a}} =
-\sum_i \frac{\partial E}{\partial b_i} \cdot
\frac{\partial b_i}{\partial x_{j,a}}
@f]

The backpropagation proceeds in three steps:

1. **Model gradient**: @f$\partial E / \partial B_\alpha@f$ from the energy model (linear or NN)
2. **Recursive backprop**: For every rule @f$b_j\mathrel{+}=c_{jkl}b_kb_l@f$,
   accumulate contributions to both inputs (including both when @f$k=l@f$):
   @f[
   g_k\leftarrow g_k+c_{jkl}g_j b_l,\qquad
   g_l\leftarrow g_l+c_{jkl}g_j b_k,\qquad
   g_j=\frac{\partial E}{\partial b_j}
   @f]
3. **Jacobian multiplication**: Multiply by @f$\partial b_i / \partial x_{j,a}@f$
   for elementary tensors

### Stress Calculation

Define @f$D_{ij,a}=\partial E_i/\partial r_{ij,a}@f$. Each site-neighbor
derivative contributes @f$+\mathbf D_{ij}@f$ to the center and
@f$-\mathbf D_{ij}@f$ to the neighbor, and accumulates the internal virial:

@f[
S_{ab}=-\sum_i\sum_j D_{ij,a}\,r_{ij,b}.
@f]

For a deformation acting as @f$r'_a=r_a+\sum_b\eta_{ab}r_b@f$,
@f$S_{ab}=-\partial E/\partial\eta_{ab}@f$. Internal `Atoms::stress_`
stores this extensive tensor in eV. Comparing with GPa stress or LAMMPS
pressure requires volume and unit conversions and the corresponding output's
sign convention. Short-range MTP_M uses the same construction; long-range
models also need explicit cell-dependent contributions.

## Coefficient Training

### Gradient w.r.t. Coefficients

For a radial coefficient @f$c@f$, holding the linear or external-model
parameters fixed, the energy gradient propagates through the moment values:

@f[
\frac{\partial E}{\partial c} =
\sum_i \frac{\partial E}{\partial b_i} \cdot
\frac{\partial b_i}{\partial c}
@f]

Linear coefficients also have direct derivatives. For ordinary MTP,
@f$E_i=\epsilon_{z_i}+\sum_\alpha\xi_\alpha B_\alpha(\mathfrak n_i)@f$,
where @f$\epsilon_s@f$ is the constant energy offset for species @f$s@f$, so

@f[
\frac{\partial E_i}{\partial\xi_\alpha}=B_\alpha(\mathfrak n_i),\qquad
\frac{\partial E_i}{\partial\epsilon_s}=\delta_{z_i,s}.
@f]

Sum these site derivatives over atoms for the total-energy derivatives.

For force gradient, the second-order mixed derivative is needed:

@f[
\frac{\partial F_{j,a}}{\partial c_k} =
-\frac{\partial^2 E}{\partial x_{j,a}\partial c_k}
@f]

**See:** @ref theory_training

## Neural Network Extension

`PotMTP_M` implements the repository's trainable channels and selectable heads;
see [MTP_M: Multiple Channels and Nonlinear Energy Heads](mtp-m.md).
It projects invariants into channels before evaluating site energy, which should
be distinguished from feeding all invariants directly into a generic network.

The template method `CalcSiteEFS<Model>` allows plugging an external model
(e.g., a neural network) to replace the linear energy expression:

@f[
E_i = \mathcal{N}\!\left(\{B_\alpha(\mathbf{r}_i)\}_\alpha\right)
@f]

The model must provide a `propagate()` method that returns @f$E@f$,
@f$\partial E/\partial B@f$, and optionally
@f$\partial^2 E/\partial B\partial B@f$.

**See:** @ref nnbase

## Implementation Classes

| Class | Role |
|-------|------|
| `PotMTP` | Full MTP implementation with training support |
| `PotMTP_M` | MTP-M with multiple latent channels and selectable linear, polynomial or neural heads |
| `DescriptorMTM` | MTP descriptor wrapped as a `Descriptors` for MaxVol |

**See:** @ref potentials
