# Transition States and Stationary Points {#advanced_saddle_workflows}

[中文](saddle-workflows_zh.md)

Search stationary points and paths with `nroptimize`, Dimer, mode following, and
NEB/CI-NEB/VCNEB. Dimer also has a Python API. Coordinate conventions and
numerical tests are covered in [the validation notes](saddle-numerics.md).

Start ordinary calculations with the [relaxation](../guides/relaxation.md) and
[NEB guides](../guides/neb.md). Build the Python package separately only when
using the Python API below.

## Build and inputs

Configure MKL and other dependencies from the repository root as described in
the [installation guide](../guides/installation.md), then run
`make -j2 imrmlp`. Put this checkout's `build/release` first in
`LD_LIBRARY_PATH` so an older installed library is not loaded. The examples below use CPU/OpenMP builds.

You must supply a potential, structures, and an `index.map` species map suitable
for the target material. NR and mode following use the analytic `PotMTP`
Hessian directly; arbitrary potential types are not accepted. Dimer and NEB use
the generic potential interface, but each model still needs separate validation
of energies, forces, and the stress required by variable-cell searches. Run each
task in its own directory because the native programs write fixed-name
intermediate files.

## NR stationary-point refinement

```sh
imrmlp nroptimize --potfile=potential.mtp --input=POSCAR --output=CONTCAR \
  --index=index.map --no-cell --gtol=0.001 --max-iteration=100
```

Drop `--no-cell` to optimize symmetric cell strain; `--pressure` is in GPa.
With a fixed cell, `gtol` applies to atomic gradient components (eV/Å). Variable
cell searches use mixed generalized coordinates, so a single value cannot be
interpreted as eV/Å for every component. `--fixed-atoms` takes zero-based atom
indices; it fixes Cartesian positions for a fixed cell and fractional
coordinates for a variable cell. The constrained Hessian is inverted in the free
subspace. `example-inputs/nroptimize.ini` is a parameter reference, not a
loadable settings file.

Exit 0 means the gradient criterion was met; 2 means the iteration budget was
exhausted. Both write the final structure. NR can converge to a minimum or a
stationary point of any order, so check the number of negative Hessian modes
before claiming a first-order saddle.

## MODECAR and Dimer

```sh
# mode.ini may be empty; add --hessian=hessianfull when a Hessian exists.
imrmlp modefollowing mode.ini --potfile=potential.mtp --input=POSCAR \
  --index=index.map --nocell --mode=9 --imode=0 --modecar-out=MODECAR
imrmlp dimer dimer.ini
```

`dimer.ini` uses whitespace-separated keys; a minimal example is:

```text
structure POSCAR
input-format POSCAR
index index.map
output CONTCAR
pot:pottype mtp
pot:potfile potential.mtp
dimer:modecarf MODECAR
dimer:maxiteration 200
dimer:gtol 0.001
dimer:rot_gtol 0.001
dimer:trn_gtol 0.001
```

Without `modecarf`, a random direction is used. `dimer:random_seed 0` is the
reproducible default; -1 requests a random seed. An explicitly named direction
file that is missing, truncated, non-finite, zero-length, or inconsistent in
dimension/coordinates is an error. `ddr` is the center-to-endpoint distance; it
is in Å for a fixed cell. The translational gradient averages both endpoints,
and the rotational gradient is consistent with the curvature objective. Older
parameter sets may need retuning, especially rotational thresholds.

Fixed-cell MODECAR files store Cartesian atomic displacements. New files begin
with `GPML_MODECAR 1 N convention`; variable-cell files also store the reference
cell and record the coordinate convention. Legacy plain numeric files still load
but must match the current coordinate settings; they are not an unconditional
replacement for a VASP MODECAR. Variable-cell export reports the required
settings: symmetric mode needs `vcflag true`, `symm_strain true`,
`restore_F_cholesky true`, `recast_atomic_force true`, and the same reference
cell. `--car-rep` export uses `symm_strain false`. See the full
`example-inputs/dimer.ini`.

Dimer exits 0 on convergence and 1 when unconverged; input, numerical, or output
errors produce diagnostics and a nonzero exit. Python raises on errors and
returns `converged=False` for a normal budget exhaustion. `dimer.dat` records the
stop reason, iteration count, center energy/enthalpy, center and rotational
gradients, and curvature; the final structure, endpoints, and
`dimer_modecar_final` are also written. A negative curvature alone does not prove
that there is exactly one negative mode.

```python
from imrmlp import Dimer
# pot and atoms are loaded Potential and Atoms objects.
result = Dimer(pot, modecar="MODECAR", max_iter=200, write_traj=True).run(atoms)
print(result.converged, result.stop_reason, result.max_force, result.curvature)
saddle = result.saddle_atoms  # independent copy, usable after the driver is released
```

## NEB, VCNEB, and mode following

The existing NEB settings and Python `NEB` remain. VCNEB `pressure` is in GPa and
image energies already include pV, so do not add it again during postprocessing.
Validate the path with a converged material model, cell size and image count.
For MPI execution, path symmetry and release checks, see
[parallel NEB and path symmetry](neb-parallel-symmetry.md).

For `modefollowing`, mode 1 is the Hessian/animation mode, mode 2 is iterative
mode following, mode 4 performs a single NR step, and mode 9 exports the
direction. Mode 2 limits each stage with `--max-loop`; repeated initialization
does not read an unspecified `VIB_CONSTRAINT`. Budget exhaustion or a failed
final force/single-negative-mode check returns 2. Fixed-cell Cartesian mode
should combine `--nocell --car-rep` with `relax:relax_cell false`,
`relax:car_mode true`, and `relax:driver fire`. The Hessian/projection
coordinate combination for variable-cell mode following requires additional
validation; prefer a checked VCNEB path when searching between two cells.

## Check a candidate transition state

Confirm convergence, inspect the path and verify that a first-order saddle has
exactly one negative nontranslational Hessian mode. Check force tolerance,
finite-difference steps where used, cell size and potential accuracy against
independent reference data. Numerical regression tests do not establish the
physical accuracy of a material's migration or phase-transition barrier.
