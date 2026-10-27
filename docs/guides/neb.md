# NEB: From Two Endpoints to a Minimum-Energy Path {#guide_neb}

[中文](neb_zh.md)

NEB (nudged elastic band) finds migration, reaction or transformation paths
between known initial and final states. It optimizes a sequence of images
(structural copies along a path), not an MD trajectory. Prepare endpoints with
the [relaxation guide](relaxation.md), then start with fixed-cell NEB.
Variable cells and path symmetry can be added after the basic workflow works.

**No endpoints or model yet?** [Walkthrough step 4](../../examples/basic-calculations/README.md)
supplies both guesses, endpoint-relaxation settings and NEB settings for an LJ
vacancy hop. Complete its environment/input preparation first. This page's main
example uses your ordinary MTP and migration problem.
To train before calculating the path, use the [MTP walkthrough](../../examples/mtp-workflow/README.md).

## 1. Prepare endpoints and atom correspondence

Put `initial.vasp`, `final.vasp`, `model.mtp` and `index.map` in a new task
directory. Endpoints must have the same atom count, species and **atom identity
on every corresponding row**. Fixed-cell NEB also needs the same cell,
coordinate frame and boundary conditions. Relax both endpoints with the same
cell, model and constraints; do not directly use different independently
relaxed cells for fixed-cell NEB.
Export each endpoint relaxation's final output to POSCAR with `convstr`, then
copy them into the NEB directory as `initial.vasp` and `final.vasp`.
These are inputs you supply, not inputs automatically created by `neb`.

NEB does not automatically match equivalent atoms. In vacancy migration an
atom may move between endpoints, but reordering one endpoint's rows for a
neater POSCAR changes the mapping. Under periodic boundaries, check that the
displacement branch represents the intended mechanism: the minimum-image
path is not always the desired path.

For example, in a 10 Å orthogonal periodic box, the same atom moving along x
from 9.8 to 0.2 Å has a direct coordinate difference of −9.6 Å but a shortest
boundary-crossing displacement of +0.4 Å. For that short hop, inspect that
initialization crosses the boundary in +x and that the atom keeps its row
identity. Reordering atoms does not repair a path. A deliberately long path
around the box needs explicit unwrapping or intermediate images.

## 2. Save a minimal fixed-cell configuration

Use 7 images **including endpoints**, numbered 0 through 6. Missing interior
images are initialized automatically. Save as `neb.ini`:

```text
index index.map
pot:potfile model.mtp
nimgs 7
image:0 initial.vasp
image:6 final.vasp
driver fire
maxiteration 500
gtol 0.02
max_move 0.05
num_threads 1
neb:solver_policy stable-v1
neb:init_method idpp
neb:spring 1
neb:climb true
```

```sh
imrmlp neb neb.ini > neb.log 2>&1
echo $?
tail -n 20 neb.log
```

`neb` aliases `calc_neb`. This explicitly selects the `stable-v1` numerical
policy, FIRE and a fixed image count as a starting point. `gtol` tests the
complete active NEB residual; fixed-cell atomic forces use eV/Å. Image count,
tolerance, spring and step size are illustrative and need convergence checks
for the target mechanism. NEB does not read `relax:` settings; top-level
`gtol` overrides `neb:gtol`.

## 3. Understand initialization and climbing images

| Choice | Purpose | Consideration |
| --- | --- | --- |
| `neb:init_method linear` | Interpolate between supplied images | Atom exchange or complex migration may pass through overlaps |
| `neb:init_method idpp` | Improve the initial path using interpolated pair distances | Does not fix incorrect atom mapping or guarantee the desired mechanism |
| `neb:climb true` | Apply climbing forces to the selected high-energy image to refine the saddle region | Still requires whole-path convergence and independent saddle checks |
| Minimum-image unwrapping (enabled by default) | Unwrap adjacent supplied images using minimum-image displacement | Define the unwrapping deliberately for long or prescribed detour paths |

Inspect initialized structures such as `nebimage_0.vasp` before interpreting
barriers. If a better path is available, supply interior structures as
`image:1`, etc., instead of relying only on endpoint interpolation.
To start with ordinary NEB, use `neb:climb false` in a separate run, then retain
the full path for refinement.

## 4. Read outputs and check convergence

| Output | Purpose |
| --- | --- |
| `neb.log` | Run messages, errors and stopping reason |
| `neb.dat` | Path data and summaries such as `# converged`, `# status`, `# max_neb_force` |
| `mep.xyz` | Final structural sequence for inspecting the path |
| `nebimage_N_final.vasp` | Final images; `N` is a zero-based integer |
| `neb.dat.iterations.csv` | Iteration diagnostics separating accepted states from trial/budget issues |

Exit code 0 means convergence, 2 normal termination without convergence and
1 an error; the existing missing-settings status is 128. Final paths may still
be written on iteration or evaluation-budget exhaustion. File existence is
not convergence. Check the complete NEB residual, not just small perpendicular
forces or image-energy changes.

For fixed cells, the `enthalpy` column in `neb.dat` is model total energy and
`ent_diff` is relative to the initial state. The forward barrier is the highest
path energy minus the initial energy; the reverse barrier subtracts the final
energy. These are total eV for the simulated system, not automatically eV/atom.
A smooth interpolated curve is for presentation and does not replace an actual
saddle evaluation.

### Small example: read barriers from path energies

Suppose a converged fixed-cell path has these illustrative relative energies:

| Position | Energy relative to initial state (eV) |
| --- | --- |
| Initial state | 0.00 |
| Highest-energy path image | 0.46 |
| Final state | −0.05 |

The forward barrier is `0.46 − 0.00 = 0.46 eV`; the reverse barrier is
`0.46 − (−0.05) = 0.51 eV`. The final state is `0.05 eV` lower than the initial
state. Do not substitute the difference between the highest image and its
neighbor or divide by the atom count. When adding images, compare the maximum
energy, full NEB residual and mechanism for stability.

## 5. Improve reliability and restart

Check the mechanism, distances and endpoint convergence before comparing more
images, tighter tolerances and alternative initial paths. Independently check
whether the highest-energy image is a first-order saddle; exactly one negative
Hessian mode after excluding translations is one necessary check. Numerical
convergence does not establish material barrier accuracy.

Retain the original model, map, `neb.dat` and every `nebimage_N_final.vasp`.
In a backed-up run directory, increase the budget and continue:

```sh
imrmlp neb neb.ini --next_step > neb-restart.log 2>&1
```

The default fixed image count keeps `nimgs` unchanged in this example.
Continuation updates outputs with the same names; retain the original results
first. Path/stage information is restored but optimizer history is rebuilt;
step-by-step agreement with an uninterrupted run is not guaranteed.
Variable-cell restart also needs `neb_reference_cell.data`. Changing models,
adding constraints or perturbing the path should be an explicit new calculation.

## 6. Add MPI, variable cells or path symmetry as needed

- **MPI/OpenMP:** Distribute images across ranks; `--num_threads` is per rank.
  Match rank × thread count to allocated CPU cores and control BLAS threads
  separately. See [parallelism and path symmetry](../advanced/neb-parallel-symmetry.md).
- **VCNEB:** Use variable-cell paths for different endpoint cells or pressure-driven
  transformations. Cell metrics, stress and enthalpy enter the problem.
  Pressure is in GPa; output includes pV already. Fixed-cell `gtol` is not a
  uniform force tolerance in variable-cell coordinates.
- **Symmetry:** Common-subgroup constraints may reduce cost while excluding lower
  paths. Compare perturbation/release checks; the same space-group number does
  not establish symmetry of an entire labeled path.

See the [stationary-point and path guide](../advanced/saddle-workflows.md) for
these settings and alternatives such as Dimer/mode following. Complete options
are in the [command reference](../reference/commands.md).
