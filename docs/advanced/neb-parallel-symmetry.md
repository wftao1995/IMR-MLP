# NEB parallel execution and path symmetry

[中文](neb-parallel-symmetry_zh.md)

`imrmlp neb` and `imrmlp calc_neb` use the same controller. The MPI build distributes
image evaluations across ranks. Rank zero alone owns the optimizer, changes the
path, writes trajectories and results, and chooses the exit status. Endpoints,
finite-difference probes, rejected trials and adaptive-image evaluations all use
the same evaluator. Idle ranks are supported. Calculator and output failures are
reported to the controller and all ranks terminate with the same status.

## MPI and OpenMP

```bash
make BUILD_MODE=mpi imrmlp
MKL_NUM_THREADS=1 OMP_NUM_THREADS=2 mpiexec -n 4 \
  build/mpi/imrmlp-mpi neb neb.ini --num_threads 2
```

Use the matching build directory first in `LD_LIBRARY_PATH` if required by the
local installation. All ranks must use the same executable and have access to the
same potential files and referenced calculator inputs. Input parsing and output
are performed by rank zero. The existing binary Atoms codec is used only for
same-build MPI transport; it is not a new restart format. PBC flags are explicitly
transported alongside it. Geometry is replicated for each requested batch, while
only the owning rank evaluates each image; results are collected in image order.

| Setting | Default | Meaning |
| --- | --- | --- |
| `neb:omp_mode` | `images` | `images`: OpenMP across images assigned to a rank, using the thread-safe potential entry point. `atoms`: process local images sequentially and call the potential's internal OpenMP entry point. |
| `--num_threads` / `num_threads` | `4` | OpenMP threads per rank; an explicit value overrides the environment's team size. |

Choose `images` for several small images. `atoms` can help large images when the
potential implements internal parallelism; otherwise that evaluator falls back
to serial work. These modes do not nest the two OpenMP levels. Account separately
for BLAS threads and external calculator processes. The driver requests
`MPI_THREAD_FUNNELED`; MPI calls stay outside OpenMP regions. An insufficient
thread-support level is rejected for multithreaded MPI runs.

External shell calculators already use rank/thread-specific directories. Their
scripts remain responsible for allocating their own resources; launching an MPI
calculator from an MPI worker is not automatically configured. On-the-fly active
learning currently requires one rank because model changes are not distributed.
Python `NEB(omp_mode="images"|"atoms")` selects local OpenMP evaluation; MPI image
workers are a CLI facility, not an implicit behavior of Python or `NEBDriver`.

## Diagnose a path

```ini
neb:symmetry_diagnostics neb-symmetry.dat
neb:symprec 1e-5
```

The report contains initial-image space-group operations (row-vector convention
`s' = s W + t`), atom permutations, the common operation indices, the selected
subgroup and allowed atomic/cell mode counts. For each image it records the space
group, operation count and residual against the selected affine subspace in the
scaled NEB coordinates. The residual is a generalized-coordinate norm, not a
Cartesian displacement in Angstrom. The fixed reference common group is labelled `initialized_path`; later sections
also list the current common operation indices. Sections distinguish the initialized,
perturbed/constrained and final paths. `symprec` is a Cartesian positional/lattice
tolerance in Angstrom; near-tolerance classifications may change with this value.

Common operations must preserve **every labelled, unwrapped image**, with the
same atom permutation and integer lattice-image offsets. This is stronger than
sharing a space-group number. Supplied atom order, cell basis and origin are
retained. The implementation neither remaps equivalent atoms nor searches origin
shifts, equivalent cell bases or operations that reverse the order of images.
Such transformations can change the intended diffusion mechanism.

Space-group diagnostics currently require full three-dimensional periodicity;
VC paths also require `symm_strain=true`. Unsupported paths receive an explicit
report status, while ordinary NEB and smooth perturbations remain available.

## Generate a different initial path

```ini
neb:perturb_amplitude 0.02
neb:perturb_seed 17
neb:symmetry_diagnostics neb-symmetry.dat
```

After linear/IDPP initialization, a seeded atomic displacement is modulated by
`sin(pi*i/(n_images-1))`. Endpoints and cells stay fixed. The maximum Cartesian
atomic displacement over the band equals `perturb_amplitude` in Angstrom.
When a nontrivial common subgroup is available, its invariant part is removed
before perturbing. Uniform translation is removed for multi-atom systems, and
selective-dynamics masks are respected. If no displacement survives the filters,
the input is rejected. P1 and unsupported-symmetry paths receive a smooth seeded
perturbation without a claim that an additional symmetry has been broken.

Use several seeds and compare converged paths and barriers. A perturbation is a
new initial guess, not evidence that a lower barrier exists. Amplitude defaults
to zero and seed defaults to zero. The report preserves both values.

## Constrain a chosen common subgroup

```ini
neb:keep_symm true
neb:symmetry_diagnostics neb-symmetry.dat
# Optional: generators indexed in the report's initial-image operation list.
# neb:symmetry_operations 0,3
```

The default preserves the common labelled path group. Explicit generator indices
are closed into a subgroup with identity using the existing relaxation symmetry
selection utility; every resulting operation must be common to the complete
initialized path. Invalid or incompatible choices fail before optimization.
Operation indices belong to the initial endpoint dataset, not each image's own
symmetry search. A subgroup containing only identity provides no reduction.

Atomic orbit modes and symmetric cell modes are projected orthogonally in NEB's
scaled coordinate metric. Position decoding, physical generalized forces,
tangents, spring/climbing forces, guarded step proposals and convergence all use
this allowed subspace. The selected operations remain fixed during optimization,
including adaptive image insertion/removal. Endpoints are retained as supplied;
interior coordinates can be corrected within the detection tolerance. Raw force
components forbidden by the constraint do not determine constrained convergence.
A constrained stationary path does not establish stability against modes excluded
by that constraint; compare with perturbed unconstrained paths.

Constraints require full PBC and, for VC-NEB, symmetric strain coordinates. They
support the cell metric weights and either atomic coordinate representation.
They cannot be combined with selective dynamics or nonzero perturbation amplitude.
`keep_symm` defaults to false. `symm_strain` by itself only removes cell rotation;
it does not enable space-group constraints.

Python exposes the same local options: `keep_symm`, `symprec`,
`symmetry_operations`, `symmetry_diagnostics`, `perturb_amplitude`, and
`perturb_seed`, as keyword-only `NEB` constructor arguments.

## Existing numerical and restart semantics

`climb_delay` counts accepted steps under `stable-v1`, and force evaluations under
`legacy`. Convergence uses the complete active NEB residual; fixed-cell atomic
forces are in eV/Angstrom, while VC residuals depend on the generalized metric.
Top-level `gtol` overrides `neb:gtol`; `relax:` settings are not consumed by NEB.
Exit codes are 0 for convergence, 2 for a completed but unconverged run, and 1 for
an error (the existing missing-settings-file status remains 128).

Restart files and their existing path/phase recovery behavior are unchanged;
optimizer history is reset as before. New symmetry constraints and perturbations
are rejected with `--next_step`, since that format does not persist their fixed
operation/mapping state. Start a fresh calculation from explicitly supplied images
to use these features. A diagnostic-only report is allowed on restart.

## Measure MPI+OMP stages (opt in)

```ini
neb:profile profile.json
```

The empty default performs no profiling collectives or file writes. The CLI JSON
records initialization (after settings parsing), optimization, result output and
optional symmetry release wall time on the controller. Per-rank `main` and
`symmetry_release` records contain EFS wall time, serialization/deserialization,
MPI collective time **including waiting**, batch count and attempted image
calculations. Endpoint evaluations count as images. EFS time is elapsed time for
the local OpenMP batch, not the sum of individual thread CPU times. Configuration
and calculator loading belong to controller initialization, outside rank EFS
statistics; profile collection/writing and MPI startup/finalization are excluded.

Controller stages already include EFS and communication. Rank timers overlap
across processes: do not add them to controller time. `batch_wall_seconds`
includes the measured batch categories and bookkeeping. More ranks can increase
communication cost when images are cheap or ranks are idle. The benchmark suite
uses fixed EFS budgets and reports measured times without a speedup assertion.

## Check a constrained path after releasing symmetry (opt in)

```ini
neb:keep_symm true
neb:symmetry_release release-check
neb:symmetry_release_trials 3
neb:symmetry_release_seed 0
neb:symmetry_release_steps 200
neb:symmetry_release_evaluations 2001
neb:symmetry_release_amplitude 0.01
neb:symmetry_release_strain 0.001
neb:symmetry_release_energy_tol 0.0001
```

`symmetry_release` defaults to empty. A nonempty value names a **new directory**;
existing directories are never overwritten. The CLI saves the main result first,
then checks an independent copy. Python `NEB(symmetry_release="release-check", …)`
uses the same implementation and exposes `NEBResult.symmetry_release_status`.
The C++ post-run entry point is `check_symmetry_release(RunOptions)`.

Only converged `keep_symm` paths with excluded symmetry modes are checked. Other
paths are reported as `skipped`; online active learning and external stress are
unsupported. Each trial preserves the potential, pressure, reference cell,
coordinate metric and endpoints, removes the common subgroup constraint, and
holds the number of images fixed. MIC and IDPP are not reapplied. Trials use the
bounded `stable-v1` controller with the requested optimizer; this does not
change the main solver policy. A completed climbing phase does not repeat its
initial climbing delay. Trial iteration and band EFS budgets are independent;
the EFS budget excludes the two separately counted endpoint evaluations.

Trial 0 releases without perturbation. The remaining trials use successive
seeds, project out the original subgroup's invariant modes, remove uniform
atomic translation for multi-atom systems and apply a smooth endpoint-zero
profile. Cell and atomic components are bounded after decoding. The strain
bound is `||H_old^{-1} H_new - I||_F`; the atomic bound in Å applies to the extra
atomic displacement **after affine motion from cell strain**. Zero amplitudes
disable that component. A trial without a surviving displacement is skipped.

`report.json` contains main and trial barriers (total eV), convergence, residuals,
actual perturbation sizes, completed evaluation counts, attempted image evaluations (including failed batches) and wall time. `image-*.vasp` saves
the main path; `trial-N/image-*.vasp` and `trial-N/final/` save the starting and
final trial paths. VC barriers use enthalpy; fixed-cell barriers use energy.
Only a converged trial lower by more than the configured tolerance produces
`lower_barrier_found`. Other outcomes are `no_lower_barrier_observed`,
`inconclusive` (budget/stall/skipped trial), `failed`, or `skipped`. Failures leave
the main path, counters and CLI convergence exit status intact and emit a warning
when the report itself cannot be written. Check the report/status separately.

These are finite local tests, not Hessian-index or global-minimum certificates.
In particular, a residual solver can return a perturbed band to the same
stationary path even when an excluded unstable direction exists. Test multiple
seeds and amplitudes, inspect incomplete trials, and use independent saddle
validation when a saddle certificate is needed. Restart behavior is unchanged.

## Explicit multi-path search

`python scripts/benchmark/neb/search_paths.py --help` describes the separate,
opt-in search tool. It imports a complete ordered path from a NEB run directory
(`neb.ini`, `neb.dat`, all `nebimage_N_final.vasp` files). The source need not be
converged. `--frames initial` uses all initialized images instead. The output
must be a new directory. The tool freezes inputs, models and executables, records
hashes, library versions, physical CPU affinity and thread binding, and leaves
source files untouched.

```sh
PYTHONPATH="$PWD/build/neb-python:$PWD" \
LD_LIBRARY_PATH="$PWD/build/release:$LD_LIBRARY_PATH" \
python scripts/benchmark/neb/search_paths.py --source build/previous-neb \
  --output build/path-search --binary build/release/imrmlp --threads 8
```

The default is **25 candidates**: an unperturbed baseline plus four seeds
(`0 1 2 3`), three modes (`sine1 sine2 mixed3`) and both signs. Every trial uses
`stable-v1`, the original optimizer and force tolerance, and its own limits of
200 iterations, 2001 full-band EFS evaluations and 3600 seconds. Endpoint EFS
are counted separately. `--steps`, `--evaluations` and `--timeout` override these
budgets. Trials run sequentially; `--ranks`, `--threads` and `--omp-mode` select
existing MPI+OMP execution inside each trial. Profiling remains off unless
`--profile` is supplied. A failed trial does not discard the remaining results.

The generator is also available without any potential or EFS calls:

```python
from imrmlp.neb import make_path_seed
candidate, info = make_path_seed(images, settings={"vcneb": False},
    mode="mixed3", seed=7, amplitude=0.01, atoms=[12, 13])
```

Settings use unprefixed native NEB names. Input images and endpoints are never
modified. `sine1` and `sine2` mean sin(πs) and sin(2πs); `mixed3` combines up to
three resolvable sine modes with independent random directions and 1/k² weights.
Modes unresolved by the image count are omitted; a zero proposal is explicitly
skipped. Seed, mode and sign determine geometry independently of MPI or scheduling.
`--atoms` uses zero-based source labels. `--center` with `--radius` intersects
that support with a cosine-tapered neighborhood chosen once from the middle
source image. Atom selection is frozen throughout the path.

Directions lie in the intersection of this local support, fixed-coordinate
constraints and the complement of the common labelled symmetry projector.
Without excluded symmetry modes, all free directions are used and this is
recorded. Empty intersections are skipped without broadening the support.
Local single-atom motion is not removed by a center-of-mass constraint. Bounds
are measured after decoding: 0.01 Å additional atomic displacement beyond cell
affine motion, and 0.001 for `||H_old^-1 H_new - I||_F`. Cell motion can move atoms
outside the local atomic support affinely. `--amplitudes` and `--strain` explicitly
expand/change these bounds; the expanded trial count is printed before execution.

VC search requires full periodicity, symmetric-strain coordinates, the original
`neb_reference_cell.data`, six effective metric scales, pressure and coordinate
metadata from `neb.dat`. Missing metadata is an error. Both fractional and
reference-Cartesian atomic charts and Cholesky/symmetric-square-root recovery
are supported when the complete source path belongs to the specified chart.
The source chart's fixed-coordinate masks remain in force. General deformation
coordinates, partial-PBC VC, online active learning and external nonhydrostatic
stress are currently unsupported. Fixed-cell paths support partial PBC.

For a fresh ordinary CLI calculation, `neb:reference_cell FILE` accepts a 3×3
reference matrix together with every VC image. It preserves the source frame
and disables MIC, IDPP and cell alignment; it rejects restart, `keep_symm` and
initial perturbation. `neb:path_pbc 110` supplies PBC lost in POSCAR serialization.
`neb:restore_F_cholesky` defaults to true; false selects the existing symmetric
square-root recovery. These options do not restore optimizer history. The batch
tool releases spatial constraints, fixes image count and endpoints, and never
uses `--next_step`. Existing single-run perturbation/release seed sequences and
restart behavior are unchanged.

### Interpreting search reports

Each trial saves its initial/final path, seed diagnostics and `result.json`;
`manifest.json`, incremental `results.json` and final `report.json` retain the
provenance, independent costs and classification. Only converged paths enter
barrier rankings and duplicate groups. Fixed-cell comparisons use total energy;
VC comparisons use total enthalpy E+pV. Timeouts, exhausted budgets, stalled runs,
failures and skipped proposals remain listed separately. A lower-than-baseline
claim requires both paths to converge. Differences close to the energy tolerance
or residuals close to the force threshold are marked `needs_refinement`.

Deduplication preserves atom labels and endpoint correspondence. It permits one
whole-path origin shift, constant per-atom lattice gauges and a single positive
integer unimodular cell-basis change; it never rematches atoms, wraps individual
images, independently aligns images or reverses a path. Comparisons use the union
of both paths' normalized arc-length knots in one source metric. Raw paths are
not resampled on disk. Default thresholds are RMS 0.01 Å, maximum 0.03 Å,
relative cell difference 1e-4 and converged barrier difference 1e-4 eV. Deterministic
complete-link groups require every member pair to pass, preventing chain merges.
One fixed spatial operation plus one fixed atom mapping can be reported as a
geometric relation between distinct groups; it never merges their labels.

These are finite local searches, not global minimum-path or saddle certificates.
Image-reversing path groups and irreducible-representation seeds require a
separate group-action implementation and are not included. Translation, labelled
permutation and unimodular-basis tests compare physical geometry/EFS/projectors;
NEB distances are compared only when the same physical metric is retained.
Gradient tests differentiate E+pV at steps 1e-4, 1e-5 and 1e-6 with explicit
truncation/roundoff allowances. Spring, tangent and climbing residuals retain
separate algebraic tests and are not treated as gradients of a total energy.
