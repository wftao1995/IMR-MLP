# Relaxation under constraints {#advanced_relax_controls}

[中文](relax-controls_zh.md)

Part of a structure should not move: a frozen substrate, a surface where only in-plane motion is
allowed, a bond length or angle that has to stay fixed, an equilibrium under external pressure.
These requirements are implemented at different places, and mixing them usually yields an error that
names the conflicting pair without saying why it conflicts. This page is organized by **what layer
the constraint acts on**: removing degrees of freedom, changing coordinates, and putting the
constraint into the objective, followed by file formats, exclusions and complete examples.

It assumes you can already run fixed-cell and variable-cell relaxation as described in the
[relaxation guide](../guides/relaxation.md), and know what `gtol`, `isif` and `relax:driver` mean.
Key types, defaults and aliases are not repeated here; see the
[configuration reference](../reference/configuration.md) and the `relax` section of the
[command reference](../reference/commands.md). The mechanics of `relax:keep_symm`, its correction
thresholds and the exit-code precedence are in
[symmetry-constrained relaxation](relax-symmetry.md).

| What you want | Main control | Where |
| --- | --- | --- |
| Freeze a substrate, allow in-plane motion only, free directions per atom | `--position-mask`, `--filter`, POSCAR flags | Degree-of-freedom masks |
| Free only some strain components | `--cell-mask`, `relax:strain_mask` | Degree-of-freedom masks |
| Apply a pressure or a stress tensor | `relax:pressure`, `relax:external_stress` | Constraints in the objective, external loads |
| Require a bond length, angle, volume or dihedral to satisfy a relation | `--file-constraints` | Declarative constraints |
| Relax along selected vibrational modes only | `relax:vibffilter`, `relax:with_constraint_vib` | Coordinate transforms |
| Keep the initial space group | `relax:keep_symm` | [Symmetry-constrained relaxation](relax-symmetry.md) |

## What a constraint acts on

A constraint restricts which configurations are legal, but where it is implemented decides what you
can observe. Classifying a control by layer is more useful than memorizing its name: only constraints
placed in the objective expose a residual; for every other layer the only evidence is the output
structure, the mask flags and the gradient.

| Layer | Acts on | Representative controls | Residual | How failure looks |
| --- | --- | --- | --- | --- |
| Variable space, components removed | the optimizer vector | masks, `fix_shape`, `fix_vol`, `couple`, `periodic_subspace`, `rigid_groups`, `keep_symm` | none | force remains along the direction, the structure does not move |
| Variable space, coordinates changed | the basis of the optimizer vector | LC, LF, VIB, VIB2 | none | a wrong dimension fails immediately or crashes |
| Gradient | the search direction | `fixcm`, `freeze_small_gd`, `symm_force` | none | the gradient converges, the structure is still wrong |
| Objective | residuals and multipliers | `--file-constraints` with AUL, SQP or NLC | yes | the residual stops falling, exit code `2` |
| External conditions | the equilibrium condition | `relax:pressure`, `relax:external_stress` | none (check the stress) | converged at the wrong stress |

Choose in this order: first ask whether the degree of freedom can simply be removed, which is exact,
needs no tolerance and does not slow convergence; if it cannot be removed but another coordinate set
works, use LC, LF or VIB; if neither applies — a non-degenerate relation such as "this bond must be
0.98 A" is the typical case — put it in the objective and pay for a solver.

Pressure and stress are not constraints. They change what counts as equilibrium, so they add no
residual and cannot turn the exit code into `2`. Conversely, every constraint with a residual needs
the optimizer to handle objective and constraints together, and the wrong solver fails outright:
bounds are available only under `driver sqp` or `nlc`.

## Degree-of-freedom masks

A mask removes components from the optimizer vector instead of adding a constraint. It therefore has
no residual, no tolerance and cannot be relaxed; the only way to tell whether it took effect is to
read the mask flags in the output structure and the force left along frozen directions.

### The five entry points

| Entry point | Granularity | Value form | When it applies |
| --- | --- | --- | --- |
| POSCAR `Selective dynamics` | three components per atom | `T`/`F` | requires `relax:selective_dynamics true` |
| `--position-mask` | three components shared by all atoms | strict `0`/`1` | requires atomic relaxation, rejects `isif` 5 to 7 |
| `--cell-mask`, `relax:strain_mask`, `relax:lattice_mask` | cell coordinates | 6 or 9 finite numbers, greater than 0 means movable | requires `relax:relax_cell true` and `fix_shape=false` |
| `--filter`, `relax:seldyn_file` | per component, cell coordinates included | `n` followed by exactly n literal `0`/`1` | requires `relax:selective_dynamics`; `--filter` turns it on |
| `--seldyn` | per frame, per atom plus a 3x3 cell block | block format | read only when `relax:selective_dynamics=true` |

`--cell-mask`, `relax:strain_mask` and `relax:lattice_mask` write the same setting. The six-value
order is `xx yy zz yz xz xy`, valid only when `relax:cell_strain` and `relax:symm_strain` are both
true, which is the default; otherwise give the nine row-major components of the raw 3x3 lattice.
Values need not be 0 or 1: any finite number greater than 0 counts as movable. In the six-value mode
the per-frame 3x3 cell mask takes no part in the calculation; only the global mask here applies.

The three flags of `--position-mask` are identical for every atom, so it cannot express "freeze the
bottom layer completely, free in-plane motion in the top layer". It acts on optimization coordinates,
and whether those are Cartesian or fractional depends on `relax:cartesian` and
`relax:atomic_force_recast`, so `1 1 0` must not be read as the laboratory xy plane. Per-atom
distinctions need `--filter` or flags inside the structure file.

`--filter` and `relax:seldyn_file` share one format, and `--filter` overrides the path from the
settings file. The `n` on the first line must equal the **full dimension before any filter**: cell
coordinates first, with a length set by the coordinate mode — 0 for a fixed cell, 1 for `fix_shape`,
6 for symmetric strain, 9 otherwise, `rank(rank+1)/2` for `periodic_subspace` — followed by 3N atomic
coordinates. A four-atom fixed-cell structure is therefore 12. This `n` is not the `n` of an LC or LF
file; see the next section.

`--seldyn` is a different format that allows a different mask per frame. It is read only when
`relax:selective_dynamics` is true; with the switch off the file is silently ignored, without a
warning.

### Mask file formats

The optimization-coordinate mask (`--filter` and `relax:seldyn_file`) is a plain numeric stream with
no comments; a four-atom fixed-cell structure has 12 dimensions:

```text
12
0 0 0  0 0 0  1 1 0  1 1 0
```

The first line is the dimension, followed by exactly that many literal `0` or `1` values. Extra data
or a wrong count fails with a message starting `Invalid optimization-coordinate mask: `;
`0.0` and `+1` are rejected.

The `--seldyn` block format is multi-frame: a frame count, then for each frame the configuration
number (one-based) with the atom count, a 3x3 cell mask, and lines holding an atom index (zero-based)
with three components. Atoms that are not listed stay movable:

```text
1
1 4
1 1 1
1 1 1
1 1 1
0 0 0 0
1 0 0 0
```

This says: frame 1 has four atoms, all three cell components are movable, and atoms 0 and 1 are
frozen in all three components. The two formats are not interchangeable; feeding a `--filter` file
to `--seldyn` fails on the first line.

## Fixed cells, coupling and rigid groups

Besides per-component masks there are several ways to remove degrees of freedom from the structure
as a whole. They change the coordinate parameterization itself, so they cannot be stacked on top of
per-component masks.

| Control | What it does | Cost or side effect | Hard restriction |
| --- | --- | --- | --- |
| `relax:fix_shape` | keeps only isotropic scaling, the cell shape is fixed | forces `relax_cell` off, with a warning | conflicts with a cell mask, raw9 lattice constraints and atomic geometry constraints |
| `relax:fix_vol` | keeps the degree-of-freedom count, rescales the volume at the end of the coordinate transform and removes the hydrostatic part from the stress | one extra rescaling per step | conflicts with `keep_symm`, `periodic_subspace` and raw9 lattice constraints |
| `relax:couple` | `1` a-b, `2` b-c, `3` c-a, `4` all; averages the diagonal stress of the coupled axes so the optimizer sees equal components | does not lock coordinates, the lattice vectors still move | `keep_symm` clears it; conflicts with raw9 lattice constraints, atomic geometry constraints and `periodic_subspace` |
| `relax:periodic_subspace` | fractional coordinates along periodic directions, Cartesian along open ones; cell freedom drops to `rank(rank+1)/2` | only for rank-deficient periodicity | conflicts with `cartesian`, `fix_shape`, `fix_vol`, `couple`, `keep_symm`, `symm_force`, pressure and external stress, and cell masks |
| `relax:rigid_groups` | each group becomes centre-of-mass translation plus rotation, group geometry is exact | fixed cell only | needs Cartesian or recast coordinates, no masks, no LF/LC; driver limited to `lbfgs`, `cg`, `gd`, `fire` |

`relax:rigid_groups` separates groups with semicolons and lists zero-based atom indices separated by
commas or spaces, for example `0,1,2;5,6`. Groups must not overlap and no group may contain
coincident atoms.

`keep_symm` belongs to this layer too: it confines the variables to the invariant subspace of the
initial space group, which is equivalent to removing degrees of freedom by symmetry. That is why it
conflicts with almost every other constraint, including masks, file constraints, `fix_vol`,
`fix_shape`, VIB and linear transforms. Its correction thresholds and failure handling are not
repeated here.

## Coordinate transforms: LC, LF and VIB

This layer leaves the feasible set alone and changes only the variables the optimizer sees. The three
mechanisms differ in which subspace they map onto:

- LC restricts the variables to the affine subspace `x = b0 + Q2 v`. The implementation takes the
  null space from an SVD, and the optimizer works on the `n-k` dimensional `v`; the row count `k`
  must not exceed `n` and `A` must have full row rank.
- LF applies the invertible map `x = A x + b`, pulling the gradient back with `(A^T)^-1`. `k` must
  equal `n` and `A` must be invertible.
- VIB and VIB2 work along a set of normal modes. VIB only projects the gradient, VIB2 really changes
  coordinates and zeroes the modes with `abs(lambda) < relax:ztol`.

| File | First line | Then | Checks |
| --- | --- | --- | --- |
| LC | `k n` | the `k x n` matrix `A` row-major, then `k` values of `b` | `k <= n`, `n` equals the current optimizer dimension, `A` has full row rank |
| LF | `k n` | the `k x n` matrix `A`, then `k` values of `b` | `k == n`, `n` equals the current optimizer dimension, `A` is invertible |
| VIB | `n` | n mode factors, n initial displacements, an `n x n` projection matrix | none |
| VIB2 | `n` | n mode factors, n eigenvalues, n initial displacements, an `n x n` projection matrix | none |

All four are whitespace-separated numeric streams with no comments, and none of them has to be
orthonormal. They are enabled by "switch on and file present": `relax:with_linear_constraint` reads
`relax:lc_file` (default `LINEAR_CONSTRAINT`), `relax:with_linear_transform` reads `relax:lf_file`
(default `LINEAR_TRANSFORM`), and `relax:vibffilter` and `relax:with_constraint_vib` share
`relax:vib_file` (default `VIB_CONSTRAINT`). The last two are mutually exclusive; enabling both fails
with `VIB filter and constraint_vib_modes cannot be used together`.

Dimensions are the easiest thing to copy wrongly here. The `n` in an LC or LF file is the dimension
**at that point in the filter chain**, after `keep_symm` has already reduced the variables and after
VIB has changed coordinates; the `n` in a mask file is the full dimension before any filter. The two
are not equal once symmetry or a vibrational transform is active.

VIB and VIB2 file dimensions are not checked at all: a mismatch raises no readable error and instead
crashes in the following matrix operation. When writing such a file by hand, pin down the combination
of `relax:cartesian`, `keep_symm` and `relax:relax_cell` first, then verify `n`.

## Gradient projections are not constraints

`relax:fixcm` removes the centre-of-mass drift, `relax:freeze_small_gd` zeroes components below
`relax:ztol`, and `relax:symm_force` symmetrizes forces and stress by space group. All three touch
only the gradient: the feasible set is unchanged, the degree-of-freedom count is unchanged, no
residual can be reported, so they are **not constraints**. They can make a run look converged early
without satisfying any geometric relation.

They also conflict with the solvers that do handle constraints. With SQP or NLC active, any of the
three raises `SQP/NLC requires conservative coordinate gradients`; the full list of conflicts with
`keep_symm` is in [symmetry-constrained relaxation](relax-symmetry.md).

Diagnostic: if the gradient has converged but the structure is still wrong, switch these three off
and repeat from the same initial structure before suspecting the potential or the starting geometry.

## Constraints in the objective: AUL and SQP/NLC

Declarative geometry constraints, covered in the next section, are carried by two solvers. The choice
is not stylistic: bounds require SQP or NLC.

### Augmented Lagrangian

AUL changes neither coordinates nor a constraint Jacobian: it folds the constraints into the objective
and wraps a multiplier update around an ordinary optimizer. The objective (external loads included)
and its gradient are

\[
F = E + pV + \sum_i \left( \lambda_i f_i + \tfrac{1}{2} w_i f_i^2 \right)
\]

\[
\frac{\partial F}{\partial x} = \frac{\partial E}{\partial x}
  + \sum_i \left( \lambda_i + w_i f_i \right) \nabla f_i
\]

First-order only: the term \( \sum_i w_i \nabla f_i \otimes \nabla f_i \) is never assembled; that is
what the SQP path does. The constraint gradient is added before the gradient filters run, so any
filter that still gets through projects that part as well.

| Setting | Default | Effect |
| --- | --- | --- |
| `relax:with_constraint_aul` | false | Turns AUL on; also implied when the file declares anything other than `symmetry` and no sqp/nlc driver was chosen |
| `relax:constraint_loops` | 5 | Number of outer multiplier updates, must be positive |
| `relax:gtol0` | 0.1 | Inner gradient tolerance of the first round, must not be below `gtol` |
| `relax:aul_auto` | false | Estimates initial weights from the geometry Jacobian, equalities only |

The inner tolerance tightens per round as `gtol_k = max(gtol, gtol0·0.1^k)`, with the last round
forced back to `gtol`. With the defaults the five rounds are `0.1 → 0.01 → 1e-3 → 1e-3 → 1e-3`:
from the third round on the tolerance already sits at `gtol`, and later outer rounds only do dual
corrections — never looser, never tighter. A zero `constraint_loops` or `gtol0 < gtol` fails with
`AUL requires positive constraint_loops and finite gtol0 >= gtol`.

Multipliers follow `λ += w·f` using the weight **at the start of the round**; the weight is then
advanced as `w ← weight_multply·w + weight_step`. All three constants default to 1, so the weight
grows by `+1` per round. Only constraints that are still violated take part — satisfied ones are
marked on the check and their multipliers stop moving.

`relax:aul_auto` takes another route: it seeds each weight from the reciprocal of
\( \lVert \nabla f_i \rVert^2_{M^{-1}} \), scaled by the gradient magnitude
\( \max(1, \lVert g \rVert) \), and zeroes the multipliers; afterwards any residual that fails to
halve in a round multiplies its weight by 10, with **no cap**. A stuck residual therefore drives the
weight until it overflows, surfacing as
`Non-finite or nonpositive constraint multiplier/weight state`. The automatic mode suits constraints
with very different units whose residuals do fall; it is not a way to force a constraint that cannot
converge.

Every outer round reruns the inner optimization (`run_normal()`) with a fresh optimizer: LBFGS/CG are
recreated, FIRE resets its step size and temperature, velocities are zeroed, so no history carries
across rounds. Inner and outer loops share one `maxiteration` budget — round k receives the remaining
iteration count, so more outer rounds leave fewer inner steps each.

The log gains a single `AUG Loop: k` line and prints neither multipliers, weights, nor per-constraint
residuals. In the step lines `iter N, f = ..., gmax = ..., gmin = ...`, `f` is the **augmented
objective** (including `λf + ½wf²` and the loads, not the energy), and `gmax`/`gmin` are the extreme
gradient components with sign, not the largest absolute value. Per-constraint residuals are visible
only at two moments: the post-run `Constraint j: residual=..., tolerance=...` lines (v1 `distance`
and `angle` also print value/target/deviation), or by reading the constraint array from inside a
program.

| Symptom | Cause | Action |
| --- | --- | --- |
| Outer rounds exhausted, exit code 2 | No residual-stagnation test; the loop simply ends | Raise `constraint_loops`, loosen the tolerance, or change driver |
| `V2 bounds require driver=sqp or nlc` | The file carries `lower`/`upper` | Use SQP/NLC; AUL handles equalities only |
| `Automatic AUL currently supports equality regulators` | `aul_auto` met a bounded constraint | Turn `aul_auto` off |
| `Non-finite or nonpositive constraint multiplier/weight state` | Automatic weights grew without a cap, or a residual went NaN | Turn `aul_auto` off and set weights by hand |

AUL's convergence evidence is just two things: every residual inside its own tolerance and the
augmented gradient within `gtol`. It computes no KKT residual and performs no dual feasibility
check; when you need that evidence, use the solver in the next subsection. Note that `Relaxation` in
Python defaults `constraint_loops` to 10 (C++ uses 5) and does not expose `gtol0`, which stays at
0.1 — copy C++ settings with that in mind.

### SQP and NLC are the same solver

`driver sqp` and `driver nlc` run the same implementation, a serial dense SQP with identical
settings: the difference is historical naming only, and picking either changes nothing.
`relax:with_constraint_sqp` and `relax:with_constraint_nlc` are boolean shells that rename the
driver; `relax:with_constraint_lc` only renames it as well and carries no linear constraint data.

For a fixed cell, purely positional and affine constraints degrade to box bounds or linear
constraints; everything else is treated as a nonlinear constraint. Convergence requires the KKT
residual and every constraint margin at once: `state_vars.converged` needs a normal solver
termination, dual feasibility, all constraint residuals within their own tolerances, and a largest
component of the Lagrangian gradient no greater than `gtol`. That `gtol` is not "the largest force
component", so unconstrained intuition does not carry over.

Enabling SQP or NLC turns AUL off, even when `with_constraint_aul` was set explicitly, and rejects
every non-conservative gradient filter.

### Choosing between them

| Situation | Choice |
| --- | --- |
| Equalities only, memory matters | AUL, or SQP |
| Bounds, walls, intervals | SQP or NLC, mandatory |
| KKT residual and dual feasibility wanted as convergence evidence | SQP |
| LC or LF already active | SQP and NLC both work: the trial step is decoded back through the coordinate map, so the transform need not be rewritten as masks |
| VIB or `keep_symm` already active | cannot use SQP; `keep_symm` also rules out LC, LF, fixed volume/shape and every structural constraint |
| Multiple MPI ranks | file constraints unavailable; use masks or coordinate transforms |

### External loads change the target stress

`relax:pressure` is isotropic, in GPa, positive in compression; `relax:external_stress` takes 3 or 6
components in the order `xx yy zz yz xz xy`, in GPa. Neither enters a residual, both change only the
equilibrium condition, so after convergence check the final stress against the target rather than
checking a residual.

One hard restriction: v1 lattice constraints require zero pressure and zero external stress.
External loads and that family cannot be used together.

## Declarative constraints: `--file-constraints`

### When to use it

Use it when a geometric quantity must be met exactly or fall inside an interval, when targets must
bind to each frame's own initial geometry, or when walls and formulas are needed. The price is a set
of preconditions:

- A single MPI rank only; more fails with `file-constraints supports only a single MPI rank`.
- A CPU build only; a GPU build fails with `file-constraints requires a CPU build`.
- Files up to 64 KiB, at most 32 constraints, strings up to 4096 bytes, pure ASCII, no comments.
- Duplicate keys and unknown keys are hard errors that name the key.

The command-line entry point is `--file-constraints <path>`; the same path can go into a settings
file.

### Document structure and keys

The top level has exactly two keys: `version` and `constraints`. `version` must be an integer token,
`1` or `2`; `1.0`, `"1"` and `true` are all rejected. `constraints` is an array of 1 to 32 entries and
must not be empty.

These are all the keys of a single constraint; anything else fails with `unknown constraint key`:

| Key | Type | Applies to | Meaning |
| --- | --- | --- | --- |
| `kind` | string | all, required | see the next table |
| `target` | number | all but `symmetry` | target value; omitted means bind to the frame's initial geometry |
| `lower`, `upper` | number | v2, all but `symmetry` | residual bounds; mutually exclusive with `target` |
| `residual_tolerance` | number | all but `symmetry`, required | residual tolerance, units in the next table |
| `axis` | `"a"`, `"b"`, `"c"` | `lattice_length` | one axis |
| `axes` | two-element array | `lattice_angle`, `lattice_ratio` | two different axes |
| `atoms` | integer array | `distance` two, `angle` three, `dihedral` four, `position` one, `wall` one | zero-based indices |
| `component` | `"x"`, `"y"`, `"z"` | `position` | which component |
| `normal` | three-element array | `wall` | normalized on read |
| `expression` | string | `formula` | expression, see below |
| `operations`, `sites` | integer arrays | `symmetry` | initial symmetry operations and restricted atoms |

There are no `weight`, `tol` or `scale` fields: the solver maintains weights and multipliers, and
writing them in is an unknown-key error.

### Eleven kinds and their residual units

| kind | Since | Extra keys | `target` unit | Residual (v1 -> v2) |
| --- | --- | --- | --- | --- |
| `volume` | v1 | — | A^3 | `det - V` -> `V - V0` |
| `lattice_length` | v1 | `axis` | A | `L^2 - L0^2` -> `L - L0` |
| `lattice_angle` | v1 | `axes` | degrees | `cos(theta) - cos(theta0)` -> `theta - theta0` |
| `lattice_ratio` | v1 | `axes` | length ratio | `r^2 - r0^2` -> `r - r0` |
| `distance` | v1 | `atoms` x2 | A | `d^2 - d0^2` -> `d - d0` |
| `angle` | v1 | `atoms` x3 | degrees | `cos(theta) - cos(theta0)` -> `theta - theta0` |
| `dihedral` | v2 | `atoms` x4 | degrees | — -> `phi - phi0` |
| `position` | v2 | `atoms`, `component` | A, absolute Cartesian | — -> `x - x0` |
| `wall` | v2 | `atoms`, `normal` | A | — -> `n.r`, with bounds giving the interval |
| `formula` | v2 | `expression` | per expression | — -> `f - f0` |
| `symmetry` | v2 | `operations`, `sites` | — | no residual, turns `keep_symm` on |

The last two columns carry the important point: **the same kind has different residual units in the
two versions**. The six v1 kinds each use a dedicated residual, so a `distance` tolerance is in A^2
and an `angle` tolerance is dimensionless; every v2 kind except `symmetry` is translated into an
expression differentiated automatically, so the residual is the physical quantity difference itself
and the tolerance is back in A, degrees or A^3. `residual_tolerance` describes the residual in that
column — not the physical quantity, and not an energy gradient.

For `angle` the middle index is the vertex, and `[0,1,2]` is the same constraint as `[2,1,0]`. A
`dihedral` target is shifted by whole multiples of 360 degrees to sit near the initial value, which
avoids branch jumps. A `position` target is an absolute Cartesian coordinate, not a displacement, and
atoms bind to their nearest image in the initial frame. A `wall` needs `normal` plus `lower` or
`upper`; `target` alone is rejected.

The `formula` expression language has three ingredients: geometry primitives, five functions, and the
constant `pi`.

| Category | Contents |
| --- | --- |
| Primitives | `x(i)`, `y(i)`, `z(i)`, `distance(i,j)`, `angle(i,j,k)`, `dihedral(i,j,k,l)`, `lattice_angle(i,j)`, `lattice_ratio(i,j)`, `volume()` |
| Functions | `sqrt`, `exp`, `log`, `sin`, `cos`, `^` (the exponent must be constant) |
| Units | lengths in A, angles in degrees, volumes in A^3; `sin` and `cos` take radians |

`volume()` is undefined under `periodic_subspace` and fails with
`Periodic-subspace relaxation does not define a three-dimensional volume constraint`.

### Omitted targets, identity and per-frame binding

Omitting `target` means "bind this constraint to the frame's initial geometry". Nothing binds during
parsing; binding happens at the start of each frame, so with one file over many frames an omitted
target yields a different value per frame and the residual starts at zero. An explicit `target` stays
fixed across frames. Under `--seq-read` every frame rebinds.

Deduplication is by geometric quantity rather than by the whole declaration: identical `kind`, axes,
atom indices and expression, component or normal count as the same constraint, while `target`,
`lower` and `upper` take no part in the comparison. "One bond, two different intervals" therefore
cannot be expressed and fails with `duplicate constraint identity`; two `symmetry` entries also
collide because their identity is the same.

Atom indices are checked for non-negativity and distinctness at parse time; out-of-range indices are
reported at binding time, naming the constraint number.

### Layout restrictions

The v1 lattice kinds (`volume`, `lattice_length`, `lattice_angle`, `lattice_ratio`) live on the raw
nine-component lattice, so they require `relax:relax_cell true`, `relax:cell_strain false`,
`fix_shape` off, `relax:cartesian false` and a fully periodic right-handed cell. They also conflict
with `fix_vol`, `couple`, `keep_symm`, `symm_force`, VIB and VIB2, LC and LF, `selective_dynamics`,
`freeze_small_gd`, `fixcm`, `dprecond`, and with nonzero pressure or external stress. At the
application level they additionally cannot be combined with `--position-mask`, `--filter`,
`--cell-mask`, `--seldyn` or frozen-atom flags inside the structure file.

The atomic kinds (`distance`, `angle`, `dihedral`, `formula`, `position`, `wall`) are looser: they may
coexist with masks but not with `keep_symm`, `symm_force`, VIB, `fixcm` or `freeze_small_gd`, and they
do not support `fix_shape`, nor `fix_vol` or `couple` under a variable cell. A variable cell also
requires `dprecond=false` or `atomic_force_recast=true`, otherwise the run fails with
`Atomic geometry constraints with variable cell require dprecond=false or atomic_force_recast=true`.

A file containing `lower` or `upper` must set `driver sqp` or `nlc` explicitly. Every v2 constraint is
translated into an expression, so the check fires during preparation with
`V2 bounds require driver=sqp or nlc`.

### A complete example

A three-atom model molecule with a fixed cell and `relax:cartesian true`, at `(0.00, 0.00, 0.00)`,
`(0.98, 0.00, 0.00)` and `(0.49, 0.85, 0.00)`: the 0-1 distance is 0.98 A and the angle at atom 1 is
about 60.0 degrees. Both files below use this structure.

v1 equalities, with the second target omitted:

```json
{
  "version": 1,
  "constraints": [
    {"kind": "distance", "atoms": [0, 1], "target": 0.98, "residual_tolerance": 1e-6},
    {"kind": "angle", "atoms": [0, 1, 2], "residual_tolerance": 1e-4}
  ]
}
```

v2 turns the angle into an interval and asks for SQP explicitly:

```json
{
  "version": 2,
  "constraints": [
    {"kind": "distance", "atoms": [0, 1], "target": 0.98, "residual_tolerance": 1e-6},
    {"kind": "angle", "atoms": [0, 1, 2], "lower": 58, "upper": 62, "residual_tolerance": 0.2}
  ]
}
```

```ini
relax:driver        sqp
relax:gtol          1e-4
relax:maxiteration  600
```

The two files differ substantively in two places: the second constrains the angle to 58 to 62 degrees
with a tolerance of 0.2 in degrees, while the first has a dimensionless `1e-4` because it constrains
`cos(theta) - cos(theta0)`.

## Constraint file formats at a glance

The five formats sit in their own sections because each is tied to its dimension semantics. Use this
table to decide what the first line of a file should be.

| Format | Produced by | Switch or key | Dimension | Strictness |
| --- | --- | --- | --- | --- |
| JSON constraint set | hand-written | `--file-constraints` | per constraint, no global dimension | strictest: version, keys, units, indices and identity all checked |
| Optimization-coordinate mask | hand-written or scripted | `--filter`, `relax:seldyn_file` | full optimizer dimension, cell coordinates first | strict: dimension and every value checked |
| `--seldyn` block | hand-written or scripted | `--seldyn` | per frame, per atom | fairly strict: frame numbers, atom counts and duplicates all reported |
| LC and LF | external tool or script | `lc_file`, `lf_file` | the dimension at that point in the filter chain | moderate: dimension plus rank or invertibility |
| VIB and VIB2 | phonon analysis tool | `vib_file` | same as above | none: a wrong dimension crashes in the matrix operations |

## Exclusions and preconditions

The table lists, by dominant control, the combinations the source rejects explicitly. A combination
that is absent is not thereby verified; it simply has no dedicated check.

| Dominant control | Cannot be combined with |
| --- | --- |
| `relax:keep_symm` | selective dynamics; any file constraint; `fix_vol`, `fix_shape`; VIB and VIB2; LC and LF; `fixcm`; `freeze_small_gd`; `cartesian` |
| `--position-mask` | `--filter`, `--seldyn`, `--cell-mask`; `keep_symm`; VIB and VIB2; LC and LF; an active `seldyn_file`; `isif` 5 to 7 |
| `--cell-mask`, `strain_mask` | a fixed cell or `fix_shape`; `keep_symm`; VIB and VIB2; LC and LF; an active mask file |
| `--filter`, an active `seldyn_file` | `--position-mask`; `--seldyn`; a cell mask; `keep_symm`; VIB and VIB2; LC and LF; frozen atoms in the structure file |
| VIB and VIB2 | each other |
| v1 lattice constraints | the mask family; `fix_vol`; `couple`; `keep_symm`; `symm_force`; VIB and VIB2; LC and LF; `fixcm`; `freeze_small_gd`; `dprecond`; non-raw9 coordinates; a cell that is not fully periodic; nonzero pressure or external stress |
| Atomic file constraints | `keep_symm`; `symm_force`; VIB; `fixcm`; `freeze_small_gd`; `fix_shape`; `fix_vol` or `couple` under a variable cell; a variable cell with `dprecond` and no recast |
| `periodic_subspace` | `cartesian`; `fix_shape`; `fix_vol`; `couple`; `keep_symm`; `symm_force`; pressure and external stress; cell masks |
| `rigid_groups` | a variable cell; any mask; LC and LF; a driver other than `lbfgs`, `cg`, `gd`, `fire` |

Four preconditions; getting them wrong is easily mistaken for "the constraint does not work":

1. Single MPI rank and a CPU build belong to `--file-constraints` alone. `driver sqp` and `nlc` have
   neither restriction by themselves.
2. SQP and NLC accept conservative coordinate gradients only; `fixcm`, `freeze_small_gd`, the VIB
   gradient filter and `symm_force` are all rejected.
3. v1 lattice constraints need the raw9 layout: `relax_cell=true`, `cell_strain=false`,
   `fix_shape=false`, `cartesian=false`, a fully periodic right-handed cell, zero pressure and zero
   external stress.
4. Atomic constraints with a variable cell need `dprecond` off, or force recasting on.

## End-to-end examples

### Example 1: three ways to mask the same slab

An orthorhombic four-atom fixed-cell slab, illustrative and not tied to a real material. The goal is
to freeze atoms 0 and 1 in the bottom layer completely and free only in-plane motion for atoms 2 and
3 on top.

```sh
imrmlp relax --input slab.vasp --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --position-mask "1 1 0"
```

This frees xy and freezes z for every atom, including the bottom layer that should be fully frozen.
The granularity is too coarse, so switch to a per-component mask file `slab.mask` (fixed cell,
12 = 3x4):

```text
12
0 0 0  0 0 0  1 1 0  1 1 0
```

```sh
imrmlp relax --input slab.vasp --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --filter slab.mask
```

The third route puts the mask inside the structure file: a `Selective dynamics` line in the POSCAR
plus `T`/`T`/`T` per atom, with `relax:selective_dynamics true` in the settings file.

The difference is granularity and what `n` means. `--position-mask` has three flags shared by all
atoms; the twelve flags of `--filter` map to individual components, and `n` is the full dimension
before filtering; POSCAR flags travel with the structure, so the command line stays unchanged when
the structure changes. All three share one convergence criterion: the largest gradient component over
the allowed degrees of freedom. Forces along frozen directions may be nonzero.

### Example 2: one geometry constraint in v1 and v2

Using the three-atom molecule from the previous section, the v1 file uses equalities: the first gives
an explicit target, the second binds the initial geometry.

```sh
imrmlp relax --input mol.xyz --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --file-constraints eq.json
```

No `driver` is given, so AUL runs, with at most five outer rounds and a tolerance tightened from
`gtol0` each round. The second constraint starts with a zero residual because its target is the
initial angle.

With the v2 interval file the solver must be named:

```sh
imrmlp relax --input mol.xyz --output relaxed.cfg --potfile model.mtp \
  --pottype mtp --file-constraints bounds.json --settings sqp.ini
```

| Aspect | v1 (`eq.json`) | v2 (`bounds.json`) |
| --- | --- | --- |
| Distance residual | `d^2 - d0^2`, tolerance in A^2 | `d - d0`, tolerance in A |
| Angle constraint | equality, target may be omitted | an interval from 58 to 62 degrees |
| Available kinds | 6 | 11 |
| Solver | AUL automatically when unspecified | `sqp` or `nlc`, mandatory |
| Convergence | every residual within tolerance and the outer gradient at `gtol` | KKT residual and dual feasibility plus every residual |

v1 and v2 cannot be mixed in one file: `lower`, `upper`, `position` and the later kinds, `expression`
and similar fields fail under `version: 1` with
`extended constraint fields require version 2`.

## Convergence and troubleshooting

With constraints, "the gradient passed" is no longer sufficient. Three things are checked separately:
whether the gradient over allowed degrees of freedom is below `gtol`, whether each constraint residual
is below its own `residual_tolerance`, and whether the exit code is `0` or `2`. A nonzero force on a
frozen direction is normal. Under SQP and NLC, `gtol` refers to the Lagrangian gradient, which does
not mean what it means unconstrained.

| Message or symptom | Cause | Action |
| --- | --- | --- |
| `file-constraints supports only a single MPI rank` | running on several ranks | drop to one rank |
| `file-constraints requires a CPU build` | GPU build | use a CPU build |
| `file-constraints is incompatible with position-mask, filter, cell-mask and seldyn` | v1 lattice constraints together with the mask family | pick one, or move to v2 |
| `V2 bounds require driver=sqp or nlc` | bounds without the right solver | set `driver sqp` explicitly |
| `Automatic AUL currently supports equality regulators` | `aul_auto` together with an interval | turn `aul_auto` off, or use SQP |
| `SQP/NLC requires conservative coordinate gradients` | `fixcm`, `freeze_small_gd`, the VIB filter or `symm_force` still on | switch them off |
| `AUL requires positive constraint_loops and finite gtol0 >= gtol` | an illegal outer count or inner tolerance | make `constraint_loops` positive and `gtol0 >= gtol` |
| `Cell mask requires 6 components for the final cell coordinate mode` | 6 and 9 values matched to the wrong coordinate mode; the number in the message follows the mode | give the component count implied by `cell_strain` and `symm_strain` |
| `Lattice constraints require raw9 cell coordinates` | strain coordinates or `fix_shape` are on | turn `cell_strain` off and use the raw lattice |
| The residual stops falling, exit code `2` | contradictory constraints, a start far outside the feasible set, or a constraint whose coordinates another control rewrote | identify which residual, then check for other controls in the same layer |

Generic input errors (element mapping, neighbour lists, potential range) are out of scope here; see
the common-errors section of the [relaxation guide](../guides/relaxation.md).

## Remaining controls: optimizers, preconditioning, batches

Everything outside constraints is unchanged; only the parts that interact with this page are listed.

`relax:driver` defaults to `lbfgs`, with `lbfgspp`, `fire`, `cg`, `gd`, `mdmin` and `quickmin` as
alternatives, plus `sqp` and `nlc` for constraints. After changing the optimizer, compare final
energies and gradients rather than iteration counts.

Preconditioning has two constraints of its own: `relax:dprecond` conflicts with v1 lattice
constraints and with atomic constraints under a variable cell (see the previous sections), and
`relax:precond_file` with `hessian_precond_file` also uses the full optimizer dimension.
`relax:thin_cell_scaling` helps cells with vacuum padding, and `relax:sparse_precond` is limited to
fixed-cell `gd` or `fire`.

For batches, `--seq-read` streams one frame at a time and rebinds file constraints per frame;
trajectories are controlled by `relax:write_traj`, `relax:trajfile`, `relax:traj_append` and the
write interval `relax:nblock`.

## See also

- [Relaxation guide](../guides/relaxation.md): the ordinary workflow, convergence checks and gradient verification
- [Symmetry-constrained relaxation](relax-symmetry.md): what `relax:keep_symm` does and excludes
- [Configuration reference](../reference/configuration.md): every `relax:` key, its default and the constraint file paths
- [Command reference](../reference/commands.md): the flags in the `relax` section
