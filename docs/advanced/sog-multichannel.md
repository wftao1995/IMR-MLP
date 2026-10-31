# Multichannel SOG: training and deployment

[中文](sog-multichannel_zh.md)

Multichannel SOG adds several independent long-range latent channels to a shared
short-range model and descriptor. Each channel has its own latent head, Gaussian
weights and bandwidths. Use it when a single channel is insufficient. More
channels raise the cost of the long-range part only; the short-range cost is
unchanged.

\[
q_{ic}=b_{s_i,c}+\sum_a h_{ac}B_a(i),\qquad
E=E_{\mathrm{SR}}+\sum_{c=1}^{C} E_{\mathrm{SOG}}[q_c,G_c],
\quad G_c(k^2)=\sum_{m=1}^{M}w_{cm}\exp[-k^2e^{-2u_{cm}}].
\]

Each channel follows the [single-channel energy, self and zero-mode
conventions](../reference/sog.md). There are no cross-channel kernel terms.
Shared-descriptor gradients accumulate every channel's contribution, including
the response needed for energy, force and virial training.

## Configure and train

Add the following to an existing `mtp_les` configuration:

```text
pot:les_kernel sog
pot:sog_channels 3
pot:sog_terms 8
fit:charge_weight 0
```

`sog_channels` defaults to 1 and accepts 1–16. `sog_terms` is the Gaussian count
**per channel**, in 1–256. All channels share bandwidth bounds, reciprocal
cutoff/taper and zero/self modes. Latents have no physical charge unit and cannot
be used for total-charge loss or Born effective charges (BEC).

The [complete configuration](../../example-inputs/fit-sog-multichannel.ini)
provides a starting point. Copy it into your run directory, replace data/template
paths, then run:

```sh
imrmlp trainnlpot --settings fit-sog-multichannel.ini --do-lbfgs \
  --output fitted-sog-multichannel.mtp
```

Both `mtp_sog_from_scratch` and `mtp_sog_resume` support multiple channels.
Initialization checks each channel: existing nonzero heads are retained, and
an entirely zero head receives deterministic random coefficients to avoid zero
gradients. New channels start with the same Gaussian grid; different randomized
heads allow them to learn different responses.

Parameter groups remain `les_species`, `les_moment`, `sog_weights` and
`sog_log_bandwidths`. Each group covers all channels; there is no per-channel
freezing option. L2 covers all heads and kernel weights; bandwidths use the
bounded mapping. Channel count is a discrete model-layout choice. Preserve it
and the bandwidth bounds when resuming.

MTP-M also supports multichannel SOG and kernel-only training with a frozen
head. Channel count comes from compatible MTP-M template metadata. Models
cannot change their channel layout after allocation. MTP-M uses ordinary
single-stage training rather than the automatic `mtp_sog_*` stages.

For CUDA/HIP evaluation and training, see the [GPU guide](../development/cuda.md)
and [SOG reference](../reference/sog.md). The standalone LAMMPS adapter below
uses CPU/MPI.

## Model format and latent output

Single-channel SOG remains v3 and Coulomb LES remains v2. Multichannel models
write `les_version = 4`; older programs reject v4. Within each block, coefficients
are channel-major. With S species and A scalar descriptors:

| Field | Length | Order |
| --- | ---: | --- |
| `les_species_coeffs` | C×S | All species of channel 0, then channel 1, and so on |
| `les_moment_coeffs` | C×A | All descriptors of channel 0, then channel 1, and so on |
| `sog_weights` | C×M | All Gaussians of channel 0, then channel 1, and so on |
| `sog_log_bandwidths` | C×M | Same order; files store physical log bandwidth u |

In linear MTP files, `les_kernel = sog` and `sog_channels` must precede the
multichannel coefficients; `sog_terms` must precede Gaussian arrays. Save through
the model API to preserve ordering. MTP-M has its own short-range format and
cannot be passed to the linear-MTP standalone adapter.

After prediction, C++ `LES_all_latents` holds the N×C latent matrix for both
`PotMTP_LES` and `PotMTP_M`. `LES_all_qs` retains channel 0 for compatibility.
SOG clears `Atoms::charges_`, since these variables are not physical charges.

## LAMMPS output

CPU `mlip/opt-les` supports linear MTP + SOG v4. For a three-channel model:

```text
pair_style mlip/opt-les
pair_coeff * * fitted-sog-multichannel.mtp 0 1
compute q all mlip/sog/latent
dump latent all custom 10 trajectory.lammpstrj id type x y z c_q[1] c_q[2] c_q[3]
```

The mapping is one entry per LAMMPS atom type; element names work when the model
carries element metadata, and older files use zero-based indices as shown.
Single-channel output is a per-atom vector `c_q`; multichannel output is a
per-atom array `c_q[1]` through `c_q[C]`. Recreate the compute and dependent dump
when replacing a model with a different channel count. The physical-charge
compute `mlip/les/charge` rejects SOG.

## Cost and verification

The CPU reciprocal sum currently evaluates channels separately, so long-range
time grows approximately linearly with channel count. Reciprocal grids and
harmonic recurrences are reused. LAMMPS caches all channel kernels, but computes
structure factors and MPI reductions separately per channel; IMR-MLP refreshes
its kernel cache for the current channel. Choosing C therefore trades long-range
resolution against that cost.

`unitest/ut_sog_channels.cpp` covers independent channel sums, parameter and
coordinate derivatives, v4 round-trips, invalid layouts, frozen groups and
staged training. Run MPI tests from the repository root:

```sh
make -j8 BUILD_MODE=mpi build/mpi/unitest/ut_sog_channels.test
OMP_NUM_THREADS=2 MKL_NUM_THREADS=1 \
  LD_LIBRARY_PATH=$PWD/build/mpi:$LD_LIBRARY_PATH \
  mpirun -np 2 build/mpi/unitest/ut_sog_channels.test --rng-seed 42
```

LAMMPS tests cover multichannel output, zero/self-mode combinations, empty MPI
partitions and invalid metadata. See the
[LES/SOG adapter](../../LAMMPS/USER-MLIP/opt-les/README.md) for commands.
