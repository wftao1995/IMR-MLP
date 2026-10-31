# MTP+LES v2

[中文](les-v2_zh.md)

For the optional trainable sum-of-Gaussians kernel (v3/v4), see
[MTP + SOG](sog.md). The v2 Coulomb functional below is unchanged.

## Purpose and inputs

LES adds an environment-dependent long-range charge term to the local MTP
energy for fully periodic three-dimensional systems. Evaluate an existing model:

```sh
imrmlp calc_efs --pottype mtp_les --potfile model.mtp \
  --input structure.xyz --input-format extxyz --output predicted.xyz
```

Use the model's species order. Output contains predicted energy, forces and
virial. Train with `trainnlpot` using the charge-constraint settings below.
Latent charges are learned model variables; energy/force fitting alone does not
identify unique physical atomic charges. `les_sigma` and `les_dl` use Å.
The `field = value` syntax below belongs to potential files; settings files
still use `pot:key value`.

## Energy functional

LES v2 evaluates the periodic Gaussian reciprocal functional

\[
E_\mathrm{LES}=\frac{2\pi k_e}{V}\sum_{\mathbf k\ne0}
\frac{e^{-\sigma^2 k^2/2}}{k^2}w(k)|S(\mathbf k)|^2
-\frac{k_e}{\sigma\sqrt{2\pi}}\sum_i q_i^2,
\quad S(\mathbf k)=\sum_i q_i e^{i\mathbf k\cdot\mathbf r_i}.
\]

The self term is controlled by `remove_self_interaction`.  The omitted zero
mode has uniform-background semantics; latent charges are not projected to
zero mean.  Because `q_i` is an MTP descriptor model, forces always include
both the explicit reciprocal derivative and `(dE/dq)(dq/dr)`.

## Potential fields

```text
les_version = 2
les_sigma = 5.0e-1
les_dl = 2.0
les_cutoff_mode = smooth
les_taper_fraction = 1.0e-1
les_charge_treatment = uniform_background
remove_self_interaction = 1
les_response_parity = 1
les_species_coeffs = { ... }
les_moment_coeffs = { ... }
```

`smooth` applies a C2 quintic taper over the final `les_taper_fraction` of
the reciprocal sphere and is the default for training and variable-cell
simulation.  `reference_hard` reproduces the legacy/reference hard sphere
for comparison.  The cutoff is `k_max = 2*pi/les_dl`.

`les_response_parity` must be `+1` or `-1`. It fixes the otherwise
unidentifiable global sign of charge-like observables and is applied to
reported latent charges, total-charge targets and gradients, periodic
dipoles, and Born effective charges. Changing it does not change energy,
force, or virial.

## Periodicity, generalized dipole, and BEC

LES v2 requires a finite right-handed cell with periodic boundaries enabled
in all three lattice directions. Non-periodic and partially periodic
structures are rejected; slab/wire long-range sums are not implemented.
Positions may lie outside the primary cell and are wrapped internally for
neighborhood construction without modifying the input structure.

The periodic generalized dipole is complex. Periodic BEC differentiates the
full complex fractional dipole, removes the displaced atom's reference phase, and only
then takes the real part and transforms to Cartesian axes. Its tensor rows
are response directions and columns are displacement directions.

BEC uses analytic local charge derivatives rather than per-atom finite
differences. Charge centering includes the derivative of the mean charge;
the output is not automatically projected onto the acoustic sum rule.

With a fixed Coulomb LES head, `trainnlpot --linear-refit-only` also supports
`fit:linear_refit sr`: it refits only short-range linear coefficients while
preserving all LES charges and responses. `sr_sog` remains exclusive to SOG.

This is a **finite-wavevector response diagnostic**, not an exact Berry-phase
polarization derivative. With environment-dependent charges it is generally
dependent on the choice of primitive reciprocal vectors and supercell size;
charge centering does not enforce its acoustic sum rule. In particular, the
atom-dependent dephasing need not produce an integrable polarization Jacobian,
so multiplying these tensors by a uniform electric field is not guaranteed to
give a conservative force. For a fixed-BEC field calculation, use
independently calibrated reference tensors as in the
[LAMMPS response example](../../examples/les-response/README.md).

LES coefficients require an explicit supported format version. Unversioned
LES files and obsolete LES fields are rejected.

## LAMMPS

The [LAMMPS LES/SOG adapter](../../LAMMPS/USER-MLIP/opt-les/README.md)
supports restricted-triclinic cells,
MPI-reduced structure factors, global and per-atom energy/virial, full
descriptor reaction forces, and latent-charge output.  GPU LES is not part
of v2.

## Charge-constrained training

`trainnlpot` can penalize the analytically predicted total latent charge of
each configuration.  The default target is neutral; a `GlobalDouble` or
`Feature total_charge VALUE` overrides it for an individual configuration.

```text
fit:charge_weight 1.0
fit:charge_pow 1
fit:charge_target_attrib total_charge
fit:charge_default_target 0
fit:training_schedule mtp_les_from_scratch
fit:sr_stage_iterations 100
fit:les_stage_iterations 50
fit:joint_stage_iterations 200
fit:charge_warmup_factor 10
fit:les_zero_init_range 1e-3
fit:les_zero_init_seed 5489
fit:parameter_scaling group_auto
fit:scale_floor 1e-3
fit:scale_ceiling 0.1
fit:scaled_step_max 1
```

The loss is `charge_weight*(Q-Q_target)^2/N^charge_pow`.  Its gradients with
respect to LES species, LES moment, and MTP radial coefficients are analytic;
finite differences are used only by gradient-verification tests.  Use
`mtp_les_resume` to skip the short-range-only stage when starting from an
already fitted short-range model.  Total-charge neutrality alone does not
make the individual latent charges physically identifiable; dipoles, Born
effective charges, or atomic-charge labels are still needed for that.

For `mtp_les_from_scratch`, an exactly zero LES block is deterministically
perturbed using `les_zero_init_range` and `les_zero_init_seed`, then its
species biases are fitted to the per-configuration total-charge targets.
Existing nonzero coefficients and resume training are left unchanged.

## Hessian support

MTP+LES and compatible MTP-M+LES/SOG models support analytic Cartesian
Hessians and Hessian–vector products (HVPs), including the complete structural
response of charges/latents. Cell Hessians and mixed cell–position derivatives
require `les_cutoff_mode smooth`; hard reciprocal cutoffs do not support
analytic cell derivatives. The legacy `pot:les_hessian_fd_nudge` setting is
still accepted but does not control these analytic paths.

MTP-M with `lr_les=true` includes the full descriptor charge response in
forces, virials and parameter gradients, including radial gradients when the
LES coefficients are frozen. GPU evaluation/training currently supports SOG;
Coulomb LES v2 uses the CPU. See the [GPU guide](../development/cuda.md) for
backend coverage.
