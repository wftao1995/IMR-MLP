# Structure IO and nonlinear training

[中文](shared-workflows_zh.md)

## Structure IO

`imrmlp.iread()` yields trajectory frames without loading the entire file;
`imrmlp.write_iter()` writes an iterable using a single output session. Supported
formats and direction restrictions are listed in `imrmlp.io`. Readers own their
state, and independent sessions do not share frame counters or codec settings.
Malformed frames and truncated compressed streams raise errors when consumed.

`imrmlp.detect_format()` can inspect file contents. Explicit input formats take
precedence; format inference from a filename uses its basename. `open_reader()`
and `open_writer()` provide configured, reusable Python session objects.

Unit conversion is explicit: `unit_conversion_factor()`, `convert_units()`,
`source_unit_system()` and `canonicalize_atoms()` distinguish lengths, energies,
time, force and virial quantities. Native units are Angstrom, eV, ps and amu;
`Atoms.stress` stores the energy virial. Source-unit overrides apply to EXTXYZ.

XSF scalar grids have independent `read_xsf_fields()` and `write_xsf_fields()`
APIs using `ScalarField` and `DataGridBlock`. They preserve grid layout and
spanning vectors without treating scalar grids as atomic structures.

For derived conversion labels, see [scalar expressions](convstr-scalar-labels.md).
For selected Honeycutt–Andersen output, see [HA categories](ha-selected-indices.md).

## Training

`trainnlpot` and Python `NonlinearTrainer` share the CPU nonlinear training engine.
Training settings accept full Cartesian Hessian labels through `hessian_weight`,
`hessian_pow` and `hessian_fd_step`, alongside E/F/S and HVP observations. Training
uses owned structure copies; converting positions does not alter input labels.

A fixed ordinary MTP or ZBL core can be supplied for residual training of an
ordinary MTP. The core remains fixed, while composite publication preserves both
components. Training/validation selection and final held-out test evaluation use
separate datasets. Named trainable groups and supported staged schedules control
which coordinates are fitted.

CPU CG can perform one terminal linear refit after its accepted final point.
The refit preserves radial parameters, checks the objective, and restores the
accepted coefficients if verification fails. It does not alter the CG trajectory.
See [linear refitting](linear-refit.md) for configuration and supported blocks.

## Relaxation and paths

V2 relaxation accepts explicit position/cell masks, versioned physical constraint
files, caller-supplied nonlinear coordinates and supported preconditioners.
Partial-periodic and cell-free geometries use the corresponding coordinate modes;
controls requiring a full cell reject singular geometry before producing output.

NEB exposes solver budgets, stable trial handling, optional symmetry diagnostics,
reproducible perturbations and release checks. See
[parallel NEB and path symmetry](neb-parallel-symmetry.md). These options remain
opt-in; a finite iteration budget is distinguished from convergence.
