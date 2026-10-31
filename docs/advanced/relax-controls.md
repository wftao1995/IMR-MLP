# Relaxation controls beyond the guide

[中文](relax-controls_zh.md)

The [relaxation guide](../guides/relaxation.md) covers what ordinary runs need: atom
and cell freedom, `isif`, selective dynamics, symmetry and the choice of optimizer.
This page maps the rest, so the guide can stay short. It names the controls and says
what they are for; defaults, units and the complete key lists live in the
[configuration reference](../reference/configuration.md), and the matching flags in
the `relax` section of the [command reference](../reference/commands.md).
`imrmlp relax --create-input` writes a commented example deck with the common keys.

## Partly free cells and atoms

Masking individual degrees of freedom is how asymmetric cells, slabs and frozen
substrates are handled.

| Goal | Control |
| --- | --- |
| Free only some strain components | `relax:strain_mask` or `relax:lattice_mask`: 6 values for symmetric strain, 9 row-major values for the raw lattice |
| Same, from the command line | `--cell-mask` |
| Free the same three directions for every atom | `--position-mask`, three 0/1 flags |
| Explicit flag for every atom and cell coordinate | `--filter`, or `relax:seldyn_file` from a settings file |
| Selective dynamics for formats without POSCAR flags | `--seldyn` |
| Keep lattice vectors coupled while the cell relaxes | `relax:couple`: `1` a-b, `2` b-c, `3` c-a, `4` all |
| Volume only, shape fixed | `relax:fix_shape`; `relax:fix_vol` instead holds the volume while the shape changes |
| Slabs with vacuum | `relax:periodic_subspace`, which keeps the open direction Cartesian while the periodic Gram matrix changes |
| Rigid fragments at a fixed cell | `relax:rigid_groups`, zero-based atom indices grouped by semicolons |

The masks are validated against each other rather than silently combined. A strain
mask requires cell relaxation without `fix_shape`; `--filter` refuses
`--position-mask`, `--cell-mask`, `keep_symm` and the filter modifiers; and
`--position-mask` rejects `keep_symm` and `isif` 5 to 7. The error message names the
conflicting pair.

## External loads

`relax:pressure` is isotropic and positive in compression. `relax:external_stress`
takes 3 or 6 components in the order `xx yy zz yz xz xy`, in GPa, for a full tensor.
`relax:stiffness` supplies a strain stiffness for preconditioning, and
`relax:cell_factor` scales cell steps against it.

## Coordinates and preconditioning

Variable-cell relaxation defaults to symmetric strain (`relax:symm_strain`), which
removes the rotational null space, and to `relax:cell_strain`, which updates strain
instead of raw lattice vectors. `relax:atomic_force_recast` keeps atomic forces
consistent with the strain metric. `relax:cartesian true` switches to Cartesian
coordinates and is meant for fixed-cell runs.

The default optimizer needs no scaling. For thin films, layered cells or very stiff
directions these help:

- `relax:dprecond` — automatic diagonal scaling for fractional symmetric-strain coordinates.
- `relax:thin_cell_scaling` — cell preconditioning from the periodic measure, so vacuum padding does not dominate.
- `relax:precond_file`, `relax:hessian_precond_file` — diagonal or full-Hessian metrics read from a file.
- `relax:sparse_precond` — neighbor-graph metric for large fixed-cell runs with `gd` or `fire`.

## Constraints

Three families exist and do not mix freely:

- Coordinate filters (LC, LF, VIB), enabled with `relax:with_linear_constraint`,
  `relax:with_linear_transform`, `relax:with_constraint_vib` and their `*_file`
  companions. They transform the optimization variables or the gradient.
- Constraints solved alongside the optimization: `relax:with_constraint_aul`, or
  `relax:driver sqp` and `nlc`.
- Declarative geometry equalities from `--file-constraints`, a JSON schema whose
  second version adds bounds, walls, formulas, dihedrals and selected initial
  symmetry operations. These require `driver sqp` or `nlc`, run on a single MPI rank,
  and omitted targets bind each frame's initial geometry.

`relax:fixcm` removes centre-of-mass drift by adjusting the gradient, and
`relax:freeze_small_gd` zeroes components below `relax:ztol` so they leave the active
set.

## Optimizers and stopping

`relax:driver` accepts `lbfgs` (the default), `lbfgspp`, `fire`, `cg`, `gd`, `mdmin`
and `quickmin`, plus `sqp` and `nlc` for the constraint families above. LBFGSpp
carries its own line-search and convergence keys under `relax:lbfgspp:`.

`relax:gtol` decides convergence. `relax:ftol` and `relax:xtol` stop a run that has
stopped making progress, and `relax:gtol0` relaxes the inner tolerance of the first
augmented-Lagrangian loops. None of them replaces a gradient check on the final
structure, as the guide describes.

## Multi-frame input and output

`--seq-read` streams one frame at a time, so a large dataset no longer has to fit in
memory, and a failure on a later frame keeps the frames already written. Trajectories
are controlled by `relax:write_traj`, `relax:trajfile`, `relax:traj_append` and the
write interval `relax:nblock`. `relax:apply_disp` with `relax:disp` starts from a
randomly displaced copy, which is a quick check that the minimum does not depend on
the initial guess.

## See also

- [Symmetry-constrained relaxation](relax-symmetry.md): what `relax:keep_symm` does, and what it excludes
- [Configuration reference](../reference/configuration.md): every `relax:` key with its default
- [Relaxation guide](../guides/relaxation.md): the ordinary workflow
