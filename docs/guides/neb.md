# NEB minimum-energy paths {#guide_neb}

[中文](neb_zh.md)

NEB (nudged elastic band) finds migration, reaction or transformation paths between
known initial and final states. It optimizes a sequence of images (structural
copies along a path), not an MD trajectory.

Out of scope: obtaining the endpoint structures (see
[structure relaxation](relaxation.md)) and advanced variable-cell or symmetry
settings (see [parallelism and path symmetry](../advanced/neb-parallel-symmetry.md)).

## Required inputs

| File | Content | Source |
| --- | --- | --- |
| `initial.vasp` | Initial state | POSCAR exported with `convstr` after relaxing the initial state |
| `final.vasp` | Final state | POSCAR exported with `convstr` after relaxing the final state |
| `model.mtp` | Trained ordinary MTP model | Training output or a model provider |
| `index.map` | Model species order | The model or its training records |

Endpoints must have the same atom count, species and **atom identity on every
corresponding row**. Fixed-cell NEB also needs the same cell, coordinate frame and
boundary conditions: relax both endpoints with the same cell, model and
constraints, and do not use different independently relaxed cells for fixed-cell
NEB. You supply `initial.vasp` and `final.vasp`; `neb` does not create them.

NEB does not automatically match equivalent atoms. In vacancy migration an atom may
move between endpoints, but reordering one endpoint's rows for a neater POSCAR
changes the mapping. Under periodic boundaries, check that the displacement branch
represents the intended mechanism; the minimum-image path is not always the
desired path:

> In a 10 Å orthogonal periodic box, the same atom moving along x from 9.8 to
> 0.2 Å has a direct coordinate difference of −9.6 Å but a shortest
> boundary-crossing displacement of +0.4 Å. For that short hop, confirm that
> initialization crosses the boundary in +x and that the atom keeps its row
> identity; reordering atoms does not repair a path. A deliberately long path
> around the box needs explicit unwrapping or intermediate images.

## Commands and settings

Use 7 images **including endpoints**, numbered 0 through 6. Missing interior
images are initialized automatically. Save as `neb.ini`:

```text
# ===== Input/Output =====
index               index.map

# ===== Potential =====
pot:potfile         model.mtp

# ===== Parallel =====
num_threads         1

# ===== Images =====
nimgs               7
image:0             initial.vasp
image:6             final.vasp

# ===== Optimizer =====
driver              fire
max_move            0.05

# ===== Convergence =====
gtol                0.02
maxiteration        500

# ===== NEB =====
neb:solver_policy   stable-v1
neb:init_method     idpp
neb:spring          1
neb:climb           true
```

```sh
imrmlp neb neb.ini > neb.log 2>&1
echo $?
tail -n 20 neb.log
```

`neb` aliases `calc_neb`. This explicitly selects the `stable-v1` numerical policy,
FIRE and a fixed image count. `gtol` tests the complete active NEB residual;
fixed-cell atomic forces use eV/Å. Image count, tolerance, spring and step size are
illustrative and need convergence checks for the target mechanism. NEB does not
read `relax:` settings; top-level `gtol` overrides `neb:gtol`.

### Initialization and climbing images

| Choice | Purpose | Consideration |
| --- | --- | --- |
| `neb:init_method linear` | Interpolate between supplied images | Atom exchange or complex migration may pass through overlaps |
| `neb:init_method idpp` | Improve the initial path using interpolated pair distances | Needs correct atom mapping; it only optimizes distances, so an unhelpful mechanism still needs manual intermediate images |
| `neb:climb true` | Apply climbing forces to the selected high-energy image to refine the saddle region | Still requires whole-path convergence and independent saddle checks |
| Minimum-image unwrapping (enabled by default) | Unwrap adjacent supplied images using minimum-image displacement | Define the unwrapping deliberately for long or prescribed detour paths |

The program writes initialized structures such as `nebimage_0.vasp`; inspect them
before interpreting barriers. If a better path is available, supply interior
structures as `image:1`, etc., instead of relying only on endpoint interpolation.
To run ordinary NEB only, use `neb:climb false` in a separate run, then retain the
full path for refinement.

## Output and criteria

| Output | Purpose |
| --- | --- |
| `neb.log` | Run messages, errors and stopping reason |
| `neb.dat` | Path data and summaries such as `# converged`, `# status`, `# max_neb_force` |
| `mep.xyz` | Final structural sequence for inspecting the path |
| `nebimage_N_final.vasp` | Final images; `N` is a zero-based integer |
| `neb.dat.iterations.csv` | Iteration diagnostics separating accepted states from trial/budget issues |

Exit code 0 means convergence, 2 normal termination without convergence and 1 an
error; the existing missing-settings status is 128. Final paths are written even on
iteration or evaluation-budget exhaustion, so the criteria are `# converged` and
`# status` in `neb.dat`, not the presence of `mep.xyz`. Read the complete NEB residual
(such as `# max_neb_force`); perpendicular forces and image-energy changes do not enter
the criterion.

For fixed cells, the `enthalpy` column in `neb.dat` is model total energy and
`ent_diff` is relative to the initial state. The forward barrier is the highest
path energy minus the initial energy; the reverse barrier subtracts the final
energy. These are total eV for the simulated system, not automatically eV/atom.
A smooth interpolated curve is for presentation only; saddle properties need a separate
evaluation of the highest-energy image.

### Reading barriers from path energies

Suppose a converged fixed-cell path has these illustrative relative energies:

| Position | Energy relative to initial state (eV) |
| --- | --- |
| Initial state | 0.00 |
| Highest-energy path image | 0.46 |
| Final state | −0.05 |

The forward barrier is `0.46 − 0.00 = 0.46 eV`; the reverse barrier is
`0.46 − (−0.05) = 0.51 eV`. The final state is `0.05 eV` lower than the initial
state. Do not substitute the difference between the highest image and its neighbor
or divide by the atom count. When adding images, compare the maximum energy, full
NEB residual and mechanism for stability.

## Checking the result and restarting

### Is the path trustworthy?

Work through these in order; a failure at any step invalidates the ones below it.

1. **Mechanism.** Step through `mep.xyz` frame by frame: which atoms move, in which
   direction, and whether any pair passes through each other or wraps around the cell.
2. **Endpoints.** Both ends must be converged minima obtained with the same cell,
   model and constraints. A path between unconverged endpoints cannot give a barrier.
3. **Convergence.** Read `# converged`, `# status` and the full NEB residual in
   `neb.dat`; perpendicular forces and image-energy changes do not enter the criterion.
4. **Stability.** Only after the three checks above, add images, tighten `gtol` or
   start from another initial path. If the barrier or the mechanism moves, the
   earlier path was not converged.

Passing these four checks makes the barrier self-consistent. To call the top image a
saddle, evaluate its Hessian as well, with `3`
replaced by the index of the highest-energy image:

```sh
imrmlp calc_fc --potfile model.mtp --input nebimage_3_final.vasp --lowest-modes 5
```

`EIGENVALS` then lists the lowest internal modes with the three translations
removed. One negative eigenvalue is consistent with a first-order saddle; two or
more point to a higher-order stationary point or to an unconverged band.

### Continuing a run

`--next_step` picks up from the images the previous run wrote: it reads
`nebimage_<i>_final.vasp` for every image and the `#` lines of `neb.dat`. Keep
`nimgs` unchanged and those files intact.

```sh
imrmlp neb neb.ini --next_step > neb-restart.log 2>&1
```

Outputs are rewritten under the same names, so copy the previous `neb.dat`,
`mep.xyz` and `nebimage_*_final.vasp` elsewhere first.

What carries over is the path and its stage: climbing or valley images, the pinned
image and the accepted-step counter. What does not carry over is the optimizer's own
history, which is rebuilt from the resumed geometry. A continued run follows the same
path, but not the same step sequence as an uninterrupted one.

Variable-cell restarts also need `neb_reference_cell.data`; without it a `stable-v1`
run refuses to restart.

Continuation only applies when the problem itself is unchanged. A new model, a
re-interpolated path or a different setup make it a new calculation. A path with
symmetry constraints or perturbations cannot use `--next_step` at all. The run does
not compare the potential file with the previous run, so check that yourself.

## Extensions: MPI, variable cells and path symmetry

- **MPI/OpenMP:** Distribute images across ranks; `--num_threads` is per rank. Match
  rank × thread count to allocated CPU cores and control BLAS threads separately.
- **VCNEB:** Use variable-cell paths for different endpoint cells or pressure-driven
  transformations; cell metrics, stress and enthalpy enter the problem. Pressure is
  in GPa and the output includes pV already. Fixed-cell `gtol` is not a uniform
  force tolerance in variable-cell coordinates.
- **Symmetry:** Common-subgroup constraints may reduce cost while excluding lower
  paths; compare perturbation/release checks. The space-group number only says both
  endpoints belong to the same group; for the whole path read the common-operation
  indices listed in `neb-symmetry.dat`.

See the [stationary-point and path guide](../advanced/saddle-workflows.md) for Dimer
and mode-following alternatives.

## Common errors

| Symptom | Cause | Action |
| --- | --- | --- |
| Atoms cross or overlap along the path | Endpoint rows do not share atom identity | Keep the same row order for each atom in both endpoints; do not reorder for presentation |
| The mechanism differs from the intended direction | A minimum-image branch was used instead of the intended unwrapping | Prepare explicit unwrapped or intermediate images, or combine `neb:init_method linear` with manual images |
| `nebimage_*` files exist but the path did not converge | The budget ran out and a final path was still written | Check `# converged`, `# status` and the complete NEB residual |
| Barrier off by an order of magnitude | Total eV treated as eV/atom, or divided by the atom count | Fixed-cell barriers are total energy differences of the system |
| Convergence criteria fail in variable-cell runs | Fixed-cell meaning of `gtol` reused | Variable-cell coordinates are generalized; use a separate tolerance check |
| Results after changing model or constraints are not comparable | Treated as a no-change continuation | Rerun as a new calculation and keep a backup of the original results |

## Related

- [Structure relaxation](relaxation.md): endpoint preparation
- [Parallelism and path symmetry](../advanced/neb-parallel-symmetry.md), [stationary points and paths](../advanced/saddle-workflows.md)
- [Command reference](../reference/commands.md)
