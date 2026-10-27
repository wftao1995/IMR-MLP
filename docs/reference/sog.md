# MTP + trainable SOG (v3/v4)

[中文](sog_zh.md)

SOG supports periodic evaluation, nonlinear E/F/virial training,
MTP-M integration and LAMMPS CPU/MPI deployment.
Independent multichannel SOG is also implemented; see the
[multichannel guide](../advanced/sog-multichannel.md) and
[training example](../../example-inputs/fit-sog-multichannel.ini).
Use the existing `mtp_les` potential type with `pot:les_kernel sog`.
The ordinary Coulomb LES model remains v2 with its original functional.

## Functional

For dimensionless latent variables `q_i`, the Fourier multiplier is

\[
G(k^2)=\sum_m w_m\exp[-k^2/\exp(2u_m)],\qquad
E=\frac{1}{2V}\sum_k T(k^2)G(k^2)|S(k)|^2
-\frac12 C\sum_i q_i^2,
\]

where `S(k)=sum_i q_i exp(ik.r_i)`, `w_m` has units eV Å³, and
`exp(u_m)` is a reciprocal bandwidth in Å⁻¹. The onsite coefficient is
`C=sum_m w_m exp(3u_m)/(8*pi^(3/2))`. Weights may have either sign.
The energy and all derivatives include the dependence of q on MTP descriptors.

The default includes k=0 and subtracts only the analytic onsite self energy.
Saved models can select `sog_zero_mode = include | omit` and
`sog_self_mode = onsite | none`. The reciprocal C2 taper does not modify
the analytic onsite coefficient. Zero frequency contributes energy, virial,
and descriptor response, although its fixed-q positional force is zero.

Only fully periodic, finite right-handed cells are supported. Direct reciprocal
summation supports triclinic cells. This implementation does not use NUFFT.

## Training

See [fit-sog.ini](../../example-inputs/fit-sog.ini). Supply a compatible MTP
template or fitted short-range model, the species map, and training data:

```sh
imrmlp trainnlpot --settings fit-sog.ini --do-lbfgs --output fitted-sog.mtp
```

`pot:les_kernel sog` creates a kernel with `pot:sog_terms` Gaussians
(default 8). Initial weights are `1/M`; log bandwidths are evenly spaced
strictly inside `pot:sog_log_bandwidth_min` and `pot:sog_log_bandwidth_max`
(defaults -4 and 1). Bounds must lie in [-30,30], and M in [1,256].
These are starting settings, not accuracy guarantees; select the bandwidth
range and `pot:les_dl` using the target cells and cutoff convergence.

`fit:training_schedule mtp_sog_from_scratch` runs SR fitting, then a
fixed-bandwidth warm-up, then joint training. `mtp_sog_resume` skips the SR
stage. Both use the existing `sr_stage_iterations`, `les_stage_iterations`
and `joint_stage_iterations` controls. Exactly zero latent heads are seeded
reproducibly using `fit:sog_zero_init_range` (default 0.1) and
`fit:les_zero_init_seed` (default 5489). Nonzero heads are preserved.
Explicit `fit:random_init` resets the kernel to its deterministic initial grid.

Parameter groups are `sr_radial`, `sr_linear`, `les_species`, `les_moment`,
`sog_weights`, and `sog_log_bandwidths`. The single-stage schedule also supports
explicit `fit:trainable_groups`; automatic SOG schedules choose their own groups.
Energy-only and combined energy/force/virial gradients are analytic, including
the mixed derivative of the descriptor response with respect to kernel parameters.

The optimizer uses an unconstrained variable z for each bandwidth:
`u = (u_min+u_max)/2 + (u_max-u_min)*atan(z)/pi`. Its chain derivative is included
in training. Model files store physical u, not z. A coefficient that cannot be
represented strictly inside the bounds is rejected rather than silently clamped.
Changing bounds changes the optimizer coordinates; keep them fixed when resuming.

The existing `fit:penalty_weight` applies L2 to both the latent head and kernel
weights, in addition to existing SR penalties. This helps control the scaling
freedom `q -> c*q, w -> w/c²`. Bandwidths are bounded but not L2-penalized.
Use parameter scaling and tolerances appropriate to the loss magnitude; very
small initial long-range energies can otherwise meet the stopping tolerance
before the kernel learns. Monitor validation errors when choosing regularization.

## Serialization and observables

Single-channel SOG files use `les_version = 3`, `les_kernel = sog`, `sog_channels = 1`,
`sog_terms`, the two arrays, bandwidth bounds, zero/self modes, and the existing
LES latent-head blocks. `sog_terms` must precede its arrays. Incomplete arrays,
nonfinite coefficients, duplicate SOG fields, invalid modes and invalid bandwidths
are rejected. Round-trips preserve the full trainable state. Old LES files
continue to load and save as v2.
Multichannel files use v4, with `sog_channels` in [2,16] and channel-major
coefficient arrays. `sog_terms` is the Gaussian count per channel.

Generic SOG latent variables have no fixed physical charge scale. SOG prediction
clears `Atoms::charges_`; internal q is available as `PotMTP_LES::LES_all_qs`.
For multiple channels, `LES_all_latents` contains the complete N-by-C matrix;
`LES_all_qs` retains the first channel after prediction.
Total-charge training, charge balancing and BEC are rejected for SOG. Use
`fit:charge_weight 0`.

An external, physically calibrated BEC model can be combined with SOG as its
zero-field potential. The [LAMMPS example](../../examples/les-response/README.md)
uses fixed-cell, fixed-BEC DFT reference tensors. It does not derive
charges from SOG channels and is a local linear-response approximation.

`mtp_les` and compatible MTP-M+SOG models support analytic Cartesian Hessians
and HVPs. Analytic cell and mixed cell–position Hessians require a smooth
reciprocal cutoff; see [LES derivative support](les-v2.md#hessian-support).

## GPU evaluation and training

Compatible MTP+SOG and MTP-M+SOG v3/v4 models support CUDA or HIP evaluation
and training. Use `calc_efs --device cuda` (or `hip`) for evaluation and
`fit:device cuda` (or `hip`) in the training settings. Build the matching
backend; see the [GPU guide](../development/cuda.md) for model/radial-basis
restrictions and multi-GPU setup. The standalone `mlip/opt-les` interface below
is a CPU/MPI adapter; GPU-capable model metadata does not enable GPU execution
there automatically.

## MTP-M and LAMMPS deployment

`PotMTP_M<T>` reads/writes SOG v3/v4 metadata alongside its own short-range and
latent-head coefficient blocks. Use a template with the matching MTP-M head
layout; a linear MTP file is not an MTP-M file. `set_sog_definition(SOGModel)`
sets the kernel definition before loading a compatible template. Metadata in
the file activates SOG automatically. The shared `potentials/sog_model.h`
provides portable metadata validation and the standalone Gaussian formulas.

MTP-M uses the ordinary single-stage nonlinear trainer. Its parameter groups
are `sr_radial`, `sr_species`, `sr_moment`, `sr_model`, `les_species`,
`les_moment`, `sog_weights`, and `sog_log_bandwidths`. `pot:req_les false`
freezes the latent head; `pot:req_sog false` freezes kernel weights/bandwidths.
Kernel-only fitting is supported with all SR/head request flags disabled.
The bounded bandwidth mapping and full force/virial response gradients match
MTP. The automatic `mtp_sog_*` schedules above remain specific to `mtp_les`;
MTP-M random initialization preserves the kernel supplied in its template.

The [LAMMPS CPU adapter](../../LAMMPS/USER-MLIP/opt-les/README.md) accepts linear
MTP short-range models with Coulomb LES v2 or SOG v3/v4. It supports SOG zero/self
modes, triclinic cells, per-atom energy and virial, and MPI domain decomposition.
Use `compute ID GROUP mlip/sog/latent` for dimensionless q. The physical-charge
compute `mlip/les/charge` rejects SOG. Nonlinear MTP-M short-range heads are
not supported by this standalone pair style and are rejected.

Reciprocal evaluation uses grid/kernel caching, harmonic recurrence and
batched MPI reductions. Finite SOGs do not enforce a strict Coulomb tail at
arbitrarily large distances.

## Validation

The `ut_sog`, `ut_sog_training`, `ut_sog_channels` and `ut_sog_mtp_m` tests
cover independent reference sums, coordinate and parameter derivatives,
serialization, frozen groups and staged fitting. See the
[multichannel guide](../advanced/sog-multichannel.md) for MPI test commands
and the [LAMMPS adapter](../../LAMMPS/USER-MLIP/opt-les/README.md) for deployment
regressions.
