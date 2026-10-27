# `mlip/opt`, `mlip/opt/omp`, and `mlip/spline`

CPU pair styles for short-range MTP-family models. The shared entry point is
`pair_style mlip/opt`; the OpenMP variant is `mlip/opt/omp` (enable it with
`package omp N`), and `mlip/spline` reads the spline representation written by
`imrmlp mtp2spline`.

```text
units metal
newton on
# Define/read the simulation box and atoms before pair_coeff.
pair_style mlip/opt
pair_coeff * * model.mtp Al Ni
```

`pair_style mlip/opt/omp` and `pair_style mlip/spline` use the same two-line
declaration form. Numeric species maps (`pair_coeff * * model.mtp 0 1`) and the
legacy one-line declaration (`pair_style mlip/opt model.mtp 0 1` followed by
`pair_coeff * *`) remain supported. Declare the model exactly once. To replace a
loaded model, first use `pair_style none`, then redeclare the style and its
coefficients. Element-name mapping requires element metadata in the model;
use numeric indices for older files without it.

`mlip/opt` and `mlip/opt/omp` accept short-range MTP-M heads (li/q2/fs/n1/n2),
and GPM bundles. For these MTP-M/GPM inputs, restart files store model text and
settings, but the original model path must remain readable and its contents must
match. Use an absolute model path when moving restart files between run
directories. `mlip/spline` is restricted to models converted with `mtp2spline`. Long-range LES/SOG models use
[`mlip/opt-les`](../opt-les/README.md); complete native model graphs use
[`mlip/native`](../native/README.md).

The regression harness in `tests/run_metadata_regression.py` compares legacy and
new declarations, numeric and element mappings, 1/2/4 MPI ranks, and all six
virial components. See the [LAMMPS interface guide](../../README.md) for build
and installation steps, and the [command reference](../../../docs/reference/commands.md)
for `mtp2spline`.
