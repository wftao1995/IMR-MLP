# HVP-supervised training and data format {#doc_docs_hvp_training}

[中文](hvp-training_zh.md)

Training can use a small number of coordinate Hessian–vector-product labels without storing a full `3N × 3N` Hessian for every structure. Here `H = ∂²E/∂R²`; it is neither the loss Hessian with respect to model parameters nor a mass-weighted dynamical matrix.

## Supported paths

| Entry point | HVP handling |
| --- | --- |
| `imrmlp trainmtp --linear-fit` | Analytic `H_j v` for each linear basis function; supports ordinary and `lfit_fastalgo` linear fitting |
| `imrmlp trainmtp --do-lbfgs` / SGD / CG | Central force differences, reusing force parameter gradients; supports frozen radial or linear parameters |
| `imrmlp trainnlpot --do-lbfgs` / SGD / CG | The same central force differences, for models with correct forces and force parameter gradients, including LES |

HVP training through `trainnlpot` requires `fit:device cpu`. Conditional linear
refinement can include the same HVP observations when the model exposes a
supported affine block; see [linear refinement](linear-refit.md).
`trainmtp` rejects simultaneous `fit:lfit_eng_only true`; its E/F/S block-matrix export also does not support HVP.
These combinations fail explicitly rather than silently ignoring enabled HVP terms.
The generic `train_linear` / Python `LinearTrainer` interface supports CPU, fixed-cell
HVP-only or mixed E/F/S fitting by central differences of the force design matrix.
It requires `supports_linear_fit()`, implemented by `descriptor_linear`.
Use `trainmtp --linear-fit` for ordinary MTP fitting.
Generic linear training uses `force_normalization = 1`, without an extra
`force_weight` or `atomic_fitness` factor. HVP alone cannot identify a constant
energy offset; include energy labels and weight when that offset is needed.
Python `NonlinearTrainer` also accepts HVP-only positive weights, subject to the
same CPU requirement and model-specific restrictions.

## Settings and weights

Add these entries to an existing training configuration:

```text
fit:hvp_weight   0.01
fit:hvp_pow      0
fit:hvp_fd_step  0.0001
```

`hvp_weight` defaults to `0` (disabled), and `hvp_pow` defaults to `0`.
`hvp_fd_step` defaults to `1e-4 Å` and must be finite and positive.
The example weight is illustrative; choose it based on the scales of E/F/S and HVP errors.

The loss for one structure is

```text
L_hvp = hvp_weight * fitness * force_normalization / (N^hvp_pow * m)
        * sum_k || predicted_Hv[k] - target_Hv[k] ||²
```

Here `m` is the number of probe directions in that structure. The existing configuration-level force-normalization factor is reused. There is no additional `atomic_fitness` factor or implicit division by `3N`.
The `HVP monitor` reports this loss and component-wise RMS separately; validation also reports HVP RMS.
E/F/S and HVP labels can be mixed. Structures without HVP labels contribute nothing to this term.

MTP-specific linear fitting uses analytic `Hv`. Generic linear fitting applies
the same difference formula to its force design matrix; nonlinear loss and backpropagation use

```text
Hv ≈ -(F(R + epsilon*v) - F(R - epsilon*v)) / (2*epsilon)
epsilon = hvp_fd_step / max_i ||v_i||
```

The largest single-atom displacement is therefore `hvp_fd_step`; the directions and targets themselves are not renormalized.
Each direction needs two displaced configurations. Gradient evaluation recomputes the forces at one of them to refresh stateful model caches, then performs two force-gradient backpropagations. No full Hessian or full parameter Jacobian is constructed.
Smaller displacements reduce truncation error but eventually amplify numerical noise; check step-size convergence.
Training errors after analytic linear fitting are also reported through this force-difference path and can retain its finite-difference error.


Python generic linear example (structures must already contain HVP labels):

```python
trainer = imrmlp.LinearTrainer(energy_weight=0, force_weight=0, stress_weight=0,
                             hvp_weight=1.0, hvp_pow=0, hvp_fd_step=1e-4)
result = trainer.train(potential, structures)
print(result.train_hvp_rms, result.train_hvp_loss)
```

`assemble()` / `solve()` also support HVP. Training and validation monitoring use
the same difference step. MPI supports empty training shards and validation on
only some ranks, with collective propagation of assembly/monitoring errors.

## Data conventions

C++ `Atoms` stores two `Eigen::MatrixXd` fields:

- `hvp_directions_`: shape `(3N, m)`, with one dimensionless direction per column and rows ordered `atom0.x,y,z, atom1.x,y,z, ...`.
- `hvp_targets_`: the same shape, with the corresponding `Hv` in each column, in `eV/Å²`.

I/O supports `extxyz`, text `cfg`, binary `bin` and GPIO's existing compressed-file handling.
Text flattens columns in probe-major, then atom-major, then xyz order; there is no implicit mass weighting.
The labels must appear together, have lengths that are positive multiples of `3N`, and contain the same number of directions.
Nonfinite values, zero directions, duplicate labels, invalid lengths and incomplete numbers are rejected. Zero responses are valid.
Writing to a format without HVP support fails; call `clear_hvp()` first if only geometry is needed.

Example with two atoms and one direction:

```text
2
Lattice="8 0 0 0 8 0 0 0 8" hvp_directions="1 0 0 -1 0 0" hvp_targets="2 0 0 -2 0 0" Properties=species:S:1:pos:R:3
H 0 0 0
H 1.2 0 0
```

CFG uses the same flattening order, before `END_CFG`:

```text
Feature hvp_directions 1 0 0 -1 0 0
Feature hvp_targets 2 0 0 -2 0 0
```

Binary CFG uses reserved Feature strings and 17-digit numeric roundtrips without changing the existing binary version. Older readers can read them as ordinary Features, but older trainers do not use them.
These two names must not also appear as custom user attributes.

## Python writing and reading

```python
import numpy as np
import imrmlp

atoms = imrmlp.read("structure.xyz")[0]
# One probe in this example; multiple columns are supported.
v = np.zeros((3 * atoms.natoms, 1))
v[0, 0], v[3, 0] = 1.0, -1.0
# Obtain F_plus / F_minus from reference calculations, each with shape (N, 3).
epsilon = 1e-3
hv = -(F_plus - F_minus).reshape(-1, 1) / (2 * epsilon)
atoms.set_hvp(v, hv)
imrmlp.write("train-hvp.xyz", [atoms], format="extxyz")

restored = imrmlp.read("train-hvp.xyz")[0]
print(restored.hvp_directions.shape, restored.hvp_targets.shape)
```

Reference forces must come from `R ± epsilon*v` for the same structure, retaining the cell, atom order and calculation settings.
Remove rigid-translation directions so probe capacity is not spent on zero modes. A small number of probes constrains only a sampled subspace; for a complete Hessian, take directions up to the number of degrees of freedom and compare against `calc_fc` on the same structure.

## Generate labels from an existing full Hessian

`attach_hessian` adds `--hvp-probes`, `--hvp-seed` and `--hvp-only`.
It draws Gaussian directions, removes rigid translation, normalizes each to unit Euclidean norm, then computes `H*v`.
The default seed is `42`; omitting `--hvp-probes` preserves the original full-Hessian output.

```sh
imrmlp attach_hessian --input structure.xyz --hessian FORCE_CONSTANTS --source fd \
    --output train-hvp.xyz --output-format extxyz \
    --hvp-probes 4 --hvp-seed 42 --hvp-only
```

`--hvp-only` omits the full Hessian from the output; this command still reads an existing full Hessian.
To reduce reference-calculation cost, generate each probe's labels directly from the two force calculations described above.

## Validation

`unitest/ut_hvp_training.cpp` covers analytic basis HVP versus the full Hessian, linear fitting, nonlinear and LES parameter gradients, weighting/cache behavior, I/O roundtrips and malformed-data rejection.
`unitest/python/test_hvp_io.py` checks Python label assignment and roundtrips through all three formats.
