# Saddle and path coordinates and numerical tests

[中文](saddle-numerics_zh.md)

The checks use analytic potentials and small synthetic MTP fixtures. They
verify derivatives and solver behavior; material accuracy depends on the model
and data supplied for the calculation.

## Coordinate and solver contracts

- NR uses symmetric linear strain in `xx yy zz yz xz xy` order, with
  `H = H_ref (I + strain)`. Coordinates, gradients and Hessians use the same
  reference cell and include the mixed strain-position derivatives.
- Fractional variable-cell mode following uses metric components `C-I` in the
  same order, with `H = H_ref chol(C)`. Hessians include the second derivative
  of the cell map. `--car-rep` instead uses independent raw-cell and Cartesian
  atomic coordinates. See the [command reference](../reference/commands.md).
- NR constraints reduce the Hessian before inversion. Fixed atoms are
  Cartesian-fixed with `--no-cell` and fractional-fixed otherwise. Fixed DOFs
  are excluded from the convergence residual.
- Backtracking rejects non-finite, inverted-cell and excessive-energy steps;
  exhausting the backtracking budget returns an error.
- Dimer uses the endpoint average for translation and the derivative of its
  curvature for rotation, including the direction-normalization Jacobian.
  Hydrostatic variable-cell objectives use `E+pV` consistently. Symmetric
  VC-Dimer and VCNEB use analytic pullbacks through Cholesky or the symmetric
  matrix square root.
- Dimer results describe the final center and include rotation residual,
  central gradient and negative curvature checks. Negative directional
  curvature alone does not establish that a structure has exactly one
  negative Hessian eigenvalue.
- `random_seed=0` is reproducible in a given build; `-1` requests nondeterministic
  initialization. An invalid explicit MODECAR raises an error.
- `modefollowing --mode=2` returns 2 if its budget is exhausted or the final
  force/one-negative-mode check fails. `--max-loop` bounds each stage.

## Tests

| Test source | Coverage |
|---|---|
| `unitest/ut_saddle_numerics.cpp` | Analytic saddle, independent energy/gradient differences, constrained inversion and Dimer state |
| `unitest/ut_saddle_inputs.cpp` | NR convergence and constraints, strict inputs, output errors and MODECAR tangents |
| `unitest/ut_modefollowing.cpp` | Fixed/variable-cell Hessians, metric derivatives away from the reference cell, mode projection and saddle searches |
| `unitest/ut_neb.cpp` | VCNEB pressure derivatives, CI-NEB double-well barrier and convergence |
| `unitest/python/test_saddle_api.py` | Ownership, repeated runs, nonconvergence, explicit directions and trajectory writers |
| `unitest/python/test_saddle_cli.py` | Command-line output failures |

After configuring dependencies as in the [installation guide](../guides/installation.md),
run from the repository root:

```sh
export LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
make -j8 imrmlp test python
python3 -m pytest
```

## Scope

NR and mode following require MTP analytical Hessians. Their dense Hessians
are intended for small candidate structures. NR finds stationary points
without selecting a Hessian index. The tests target CPU/OpenMP and fixed-cell
or hydrostatic-pressure derivatives; they do not qualify anisotropic external
stress or arbitrary material migration paths. Derivative and MODECAR checks
alone do not establish variable-cell mode-following convergence.
