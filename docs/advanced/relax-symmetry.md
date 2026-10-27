# Symmetry-constrained relaxation and neighbor safety

[中文](relax-symmetry_zh.md)

Set `relax:keep_symm true` to preserve the initial structure's space-group
constraints during relaxation. Save the settings below as `relax.ini`, then run:

```sh
imrmlp relax --input POSCAR --output relaxed.cfg --potfile potential.mtp \
  --pottype mtp --index index.map --settings relax.ini
```

```ini
relax:keep_symm true
relax:relax_cell true
relax:relax_atom true
relax:driver lbfgs
relax:gtol 1e-5
relax:ftol 0
relax:xtol 0
relax:stepmax 0.01
relax:maxiteration 500
```

## Exact symmetry

Initialization identifies the operations and element-preserving atom mappings,
projects the cell metric onto their invariant space, and aligns the resulting
cell to the input orientation. Atomic coordinates are corrected to the periodic
affine constraints. Atom order, species, and number of atoms are retained.
`symmetry_atom_correction` and `symmetry_cell_correction` record the largest
atomic displacement and lattice-vector correction in Å. Corrections exceeding
`symprec` fail. Frozen components cannot require corrections beyond `1e-10 Å`.

Coordinates follow a fixed reference `x = x0 + J q`; reduced gradients are
`J.transpose() * g`. The small per-orbit Jacobians retain coupled site modes.
Both `atomic_force_recast` settings are supported. The reference cell stays
fixed during a run. Periodic wrapping does not change optimizer variables.
Zero allowed degrees of freedom still require a valid initial evaluation.

`keep_symm` is incompatible with selective dynamics, fixed volume or shape,
additional structural constraints, Cartesian mode, vibration transforms or
filters, linear transforms or constraints, center-of-mass filtering, and
small-gradient filtering. Cell relaxation requires symmetric strain, Cholesky
restoration and its exact strain derivative.
`keep_symm` projects the raw gradient and does not also apply `symm_force`.
Reinitialize after changing coordinate modes or structural constraints.

## Optimizer state

Python results expose `converged`, the accepted-step count `n_iter`, and
`stop_reason`. These states are reset for each run.

Only a finite final gradient meeting `gtol` establishes convergence; augmented
constraints must also meet their tolerances. Energy, step and iteration limits
can stop an unconverged run. LBFGSpp's extra `epsilon` and `epsilon_rel` criteria
default to zero.

The last accepted point is retained internally. An invalid trial restores that
point and propagates the error without automatically restarting the optimizer.
Normal completion restores the accepted structure and its evaluated energy and
gradient without evaluating it again. Python returns that structure, its final
energy and the initial energy of the run.

Line searches respect the displacement limit. A failed search raises an error
and does not present a rejected trial structure as the result.

## Output and exit status

- `0`: all inputs converged, including valid zero-DOF inputs.
- `2`: at least one input stopped without convergence and none failed.
- `1`: an input or calculation failed. Priority is `1 > 2 > 0`.

Normal output includes the last accepted structure of an unconverged run; the
log identifies it as unconverged. Failed inputs are reported and omitted, and
independent inputs continue to run. Allocation failure stops the batch while
preserving previously completed output. There are no automatic JSON status or
per-input checkpoint files.

## Input geometry and neighbors

Check that coordinates and cutoffs are finite, that atoms do not overlap
unexpectedly, and that the model supports the input periodicity. Partially
periodic inputs can have zero lattice vectors in nonperiodic directions;
space-group-constrained relaxation still requires its own valid periodic/cell
conditions. Skew cells can increase neighbor counts and memory use. If neighbor
construction fails, correct geometry or cutoff settings before rerunning.
