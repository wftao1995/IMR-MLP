# Structure Relaxation: From Input to Convergence {#guide_relaxation}

[中文](relaxation_zh.md)

Relaxation uses an existing potential to optimize atomic positions or the cell,
finding a local minimum under the chosen constraints. It does not train the
model or guarantee the global minimum. Complete a [single-point calculation](getting-started.md)
first. For a transition path between two states, use [NEB](neb.md).

**No inputs yet?** Use [steps 1–3 of the calculation walkthrough](../../examples/basic-calculations/README.md).
Its script supplies structures, a potential and both relaxation settings, then
passes the result to property calculations. This page's main example uses your ordinary MTP.
For the complete route starting with basis generation and training, use the
[MTP walkthrough](../../examples/mtp-workflow/README.md).

## 1. Decide what may move

| Goal | Atomic positions | Cell | Starting point |
| --- | --- | --- | --- |
| Defect or atomic optimization at fixed cell and volume | Free | Fixed | Fixed-cell example below |
| Equilibrium bulk cell at zero or specified pressure | Free | Free | Check single-point and fixed-cell runs before variable-cell settings |
| Optimize lattice while retaining fractional positions | Fixed | Free | Account for affine atomic motion with the cell |
| Surface with fixed bottom layers or a constrained local optimization | Partially free | Depends on the physical question | Selective dynamics or constraints; inspect frozen components |

Prepare a suitable `model.mtp`, `POSCAR` and `index.map` in a new task directory.
The model may be a training output such as `fitted.mtp`; prepare the structure
from your modeling/reference calculation and update filenames accordingly.
The example uses ordinary MTP and a valid cell. Check support before using
other model types or boundary conditions.

## 2. Optimize atoms with the cell fixed

Save as `relax.ini`. Input is POSCAR; output is EXTXYZ retaining E/F/stress.
The defaults already optimize atoms at fixed cell with L-BFGS.

```text
input POSCAR
output relaxed.xyz
index index.map
pot:potfile model.mtp
relax:cartesian true
relax:gtol 0.001
relax:maxiteration 500
relax:stepmax 0.05
relax:write_traj true
```

```sh
imrmlp relax relax.ini > relax.log 2>&1
echo $?
tail -n 20 relax.log
```

Run `echo $?` immediately after the calculation. In these fixed-cell Cartesian
coordinates, `gtol` tests the largest absolute active gradient component;
without constraints this means every force component is below `0.001 eV/Å`.
`maxiteration` is the iteration budget and `stepmax` limits optimizer updates;
neither specifies convergence accuracy. Values are illustrative; converge
tolerances for production calculations.

## 3. Check success

| Check | Interpretation |
| --- | --- |
| Exit code `0` | All input structures meet relaxation convergence criteria |
| Exit code `2` | At least one structure stopped normally without convergence; output may still contain its last accepted structure |
| Exit code `1` | Input, evaluation or output failure; inspect the specific error in `relax.log` |
| `relaxed.xyz` | Inspect final geometry, species, energy and forces; file existence is not convergence |
| `relax-trajectory.xyz` | Look for overlaps, jumps or unreasonable deformation |

Small energy changes, small steps and iteration exhaustion do not replace
force/gradient convergence. Constraints test only allowed degrees of freedom;
frozen directions may still have forces. Export a final POSCAR when needed:

```sh
imrmlp convstr relaxed.xyz relaxed.vasp --index index.map
```

A fresh `calc_efs` can independently check final forces, especially after
changing the model, coordinates or constraints.

### Small exercise: recheck final forces

Evaluate the final geometry with the same model:

```sh
imrmlp calc_efs relaxed.xyz checked.xyz --potfile model.mtp --index index.map
```

Inspect the `forces` columns in `checked.xyz`. For the **unconstrained fixed-cell**
settings on this page, a largest absolute force component of `0.0008 eV/Å`
meets `gtol=0.001`, while `0.0046 eV/Å` does not. Compare components, not
three-dimensional force norms. If logged convergence disagrees substantially,
check that the model and final frame match, then inspect effective constraints
and settings. For a tighter run, copy the settings, change input to `relaxed.xyz`,
replace `gtol` and choose a new output filename.
Compare final energy, geometry and forces for stability.

## 4. Enable cell relaxation when needed

Copy the configuration above and **replace** `relax:cartesian true` with:

```text
relax:relax_cell true
relax:cartesian false
```

This optimizes both atoms and cell at zero pressure by default. Add
`relax:pressure` only for a specified external pressure, in GPa, positive for compression.
Variable-cell optimization uses generalized atomic and cell coordinates;
`gtol` is not a uniform eV/Å force threshold. Alongside logged convergence,
check final stress/target pressure, volume and lattice parameters. Surfaces or
vacuum-containing cells need deliberate cell freedoms along periodic directions
rather than an unchanged fully free bulk-cell configuration.

## 5. Constraints, symmetry and optimizer choice

POSCAR `Selective dynamics` flags require `relax:selective_dynamics true`.
Component meaning depends on optimizer coordinates; do not automatically
interpret any T/F mask as laboratory Cartesian directions.
`relax:keep_symm true` restricts the initial space-group subspace and can exclude
real symmetry-breaking distortions. It has compatibility restrictions with
other constraints; see [constraints and symmetry](../advanced/relax-symmetry.md)
and the [configuration reference](../reference/configuration.md).

`lbfgs` is a starting point for ordinary tasks; `fire`, `lbfgspp` and `cg` are
alternatives. Compare final energy and gradients after changing the optimizer.
Avoid copying both `isif` and separate freedom flags: top-level `isif` overrides
related settings such as `relax_atom` and `relax_cell`.

## Common obstacles and next steps

- Huge forces or line-search failure: check species, units, distances and model scope before changing steps.
- Iteration exhaustion: inspect whether the trajectory is improving before increasing the budget or changing initialization/optimizer.
- A structure remains highly symmetric: check whether symmetry or constraints exclude the distortion of interest.
- Multiple input structures: inspect each frame's output and stopping status, not just the last frame.

With a reliable equilibrium structure, continue to [crystal properties](properties.md).
Relax both endpoints before [NEB](neb.md). All CLI options are in the
[command reference](../reference/commands.md).
