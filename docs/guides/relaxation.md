# Structure relaxation {#guide_relaxation}

[中文](relaxation_zh.md)

Relaxation uses an existing potential to optimize atomic positions or the cell,
finding a local minimum under the chosen constraints; which minimum it reaches
follows from the starting structure.

Out of scope: energies and errors at a fixed geometry (see
[single-point calculation](single-point.md)) and paths between two states
(see [NEB](neb.md)).

## Required inputs

| File | Content | Source |
| --- | --- | --- |
| `model.mtp` | Trained ordinary MTP model | Training output such as `fitted.mtp`, or a model provider |
| `POSCAR` | Initial structure including element symbols | Structure preparation or reference calculations |
| `index.map` | Model species order | The model or its training records |

Use one directory per task and keep settings and logs there; relative paths resolve
from the working directory, see [input files and settings conventions](conventions.md).
The examples below assume an ordinary MTP and a valid cell. Check support before
using other model types or boundary conditions.

## Commands and settings

### Degrees of freedom

| Goal | Atomic positions | Cell | Settings to use | `isif` |
| --- | --- | --- | --- | --- |
| Defect or atomic optimization at fixed cell and volume | Free | Fixed | Fixed-cell relaxation (default) | `2` |
| Equilibrium bulk cell at zero or specified pressure | Free | Free | Variable-cell relaxation plus optional `relax:pressure` | `3` |
| Optimize the lattice while retaining fractional positions | Fixed | Free | Disable atomic updates, then enable cell relaxation | `6` |
| Surface with fixed bottom layers or a constrained local optimization | Partially free | Depends on the physical question | Selective dynamics or constraints; confirm frozen components first | — |

The `isif` column is the VASP convention, available as the `--isif` flag or the
top-level `isif` key.
The remaining values follow VASP as well: `4` frees atoms and cell at fixed volume,
`5` fixes the atoms and the volume, `7` scales the volume isotropically with fixed
positions, and `8` combines atoms with isotropic volume changes. `isif` is applied
after the settings file and overrides `relax_atom`, `relax_cell`, `fix_vol` and
`fix_shape`, with a warning when both are present; use one style or the other.

### Fixed-cell relaxation

Save this as `relax.ini`. Input is POSCAR; output is EXTXYZ retaining E/F/stress.
The defaults already optimize atoms at a fixed cell with L-BFGS, so those switches
need not be written explicitly.

```text
# ===== Input/Output =====
input               POSCAR
output              relaxed.xyz
index               index.map

# ===== Potential =====
pot:potfile         model.mtp

# ===== Relaxation =====
relax:cartesian     true
relax:stepmax       0.05
relax:gtol          0.001
relax:maxiteration  500

# ===== Trajectory =====
relax:write_traj    true
```

```sh
imrmlp relax relax.ini > relax.log 2>&1
echo $?
tail -n 20 relax.log
```

Run `echo $?` immediately after the calculation. In fixed-cell Cartesian
coordinates, `gtol` tests the largest absolute active gradient component; without
constraints this means every force component is below `0.001 eV/Å`.
`maxiteration` is the iteration budget and `stepmax` limits optimizer updates;
neither specifies convergence accuracy. Values are illustrative; converge
tolerances for production calculations.

### Variable-cell relaxation

Copy the settings above and **replace** `relax:cartesian true` with:

```text
relax:relax_cell true
relax:cartesian false
```

This optimizes atoms and cell together at zero pressure by default. Add
`relax:pressure` only for a specified external pressure, in GPa, positive for
compression. Variable-cell optimization uses generalized atomic and cell
coordinates, so `gtol` is not a uniform eV/Å force threshold. Alongside logged
convergence, check final stress/target pressure, volume and lattice parameters.
Surfaces or vacuum-containing cells need deliberate cell freedoms along periodic
directions rather than an unchanged fully free bulk-cell configuration.

### Constraints, symmetry and optimizer choice

POSCAR `Selective dynamics` flags require `relax:selective_dynamics true`.
Component meaning depends on optimizer coordinates; do not automatically interpret
any T/F mask as laboratory Cartesian directions. `relax:keep_symm true` restricts
the initial space-group subspace, can exclude real symmetry-breaking distortions,
and has compatibility restrictions with other constraints; see
[constraints and symmetry](../advanced/relax-symmetry.md) and the
[configuration reference](../reference/configuration.md).

`lbfgs` is a starting point for ordinary tasks; `fire`, `lbfgspp` and `cg` are
alternatives, and final energy and gradients should be compared after changing the
optimizer.

The switches on this page cover the ordinary cases. Cell-component masks,
lattice-vector coupling, external stress tensors, constrained solvers and
preconditioning are collected in
[relaxation controls beyond the guide](../advanced/relax-controls.md).

## Output and criteria

| Check | Interpretation |
| --- | --- |
| Exit code `0` | All input structures meet relaxation convergence criteria |
| Exit code `2` | At least one structure stopped normally without convergence; output may still contain its last accepted structure |
| Exit code `1` | Input, evaluation or output failure; inspect the specific error in `relax.log` |
| `relaxed.xyz` | Inspect final geometry, species, energy and forces; it is written even without convergence, so read the exit code |
| `relax-trajectory.xyz` | Look for overlaps, jumps or unreasonable deformation |

Convergence is a force/gradient criterion only: `gtol` compares the largest absolute
gradient component over the allowed degrees of freedom, and energy changes, step sizes
and iteration counts do not enter it. Constraints test only allowed degrees of freedom;
frozen directions may still have forces. Export a final POSCAR when needed:

```sh
imrmlp convstr relaxed.xyz relaxed.vasp
```

The recheck: evaluate the final structure once more with the same model and compare the
largest absolute force component against `gtol`:

```sh
imrmlp calc_efs relaxed.xyz checked.xyz --potfile model.mtp
```

With the unconstrained fixed-cell settings above, `0.0008 eV/Å` meets `gtol=0.001`
while `0.0046 eV/Å` does not; the comparison is per component, not per-atom norms. A
clear disagreement usually means another model or frame was evaluated, or that
constraints are still active.

## Common errors

| Symptom | Cause | Action |
| --- | --- | --- |
| Huge forces or line-search failure | Species map, units, distances or model scope do not match | Check those first, then adjust `stepmax` |
| Iteration exhausted without convergence | Budget too small, or initialization/optimizer unsuitable | Inspect whether the trajectory is still improving before raising the budget or changing initialization/optimizer |
| Structure remains highly symmetric | Symmetry or constraints exclude the distortion of interest | Check `relax:keep_symm` and selective-dynamics settings |
| Inconsistent results across input structures | Frames ended in different states | Inspect each frame's output and stopping status, not just the last frame |

## Related

- [Crystal properties](properties.md): phonons, elasticity and equation of state once the equilibrium structure is fixed
- [NEB](neb.md): paths, after both endpoints are relaxed
- [Relaxation controls beyond the guide](../advanced/relax-controls.md): masks, coupling, external stress, constrained solvers
- [Symmetry-constrained relaxation](../advanced/relax-symmetry.md): what `relax:keep_symm` does and excludes
- [Input files and settings conventions](conventions.md), [command reference](../reference/commands.md)
