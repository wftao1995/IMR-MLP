# Python API {#api_python}

[中文](python_zh.md)

@tableofcontents

The `imrmlp` Python package provides a Pythonic interface to the IMR-MLP C++ framework
via pybind11 bindings. It follows a two-layer architecture:

| Layer | Module | Description |
|-------|--------|-------------|
| **C++ bindings** | `_imrmlp` (compiled `.so`) | Thin pybind11 wrappers for `Atoms`, potential evaluation, I/O, relaxations, descriptors, active learning |
| **Python wrappers** | `imrmlp` (pure Python package) | Pythonic API with property access, type hints, ASE integration, dataclass results |

## Installation

For a first run, follow [Python and ASE setup](../../README-python.md) through
evaluation of the same geometry, including model provenance and output meaning.
Use this page as an API reference; its snippets are not one sequential script.

Use Python 3.10+, a C++23 compiler and the native development libraries in the
[build guide](../guides/installation.md). Activate the intended virtual environment,
then choose one route from the repository root:

```bash
# Install into the active Python environment for use from other directories
USE_KIM=0 python3 -m pip install .

# Or build and use the package in this source checkout
python3 -m pip install numpy pybind11
make python PYTHON=python3 USE_KIM=0
python3 -c "import imrmlp; print(imrmlp.__version__)"
```

`pip install .` uses CMake to invoke Make, builds a release extension and installs
`libimrmlp.so` alongside it. Native dependencies such as MKL must already be
available. `make python` writes the extension into the source `imrmlp/` directory
and links to `build/release` by default. After changing Python interpreters,
rebuild with the matching `PYTHON=...`. `make install` does not install Python.

Install optional dependencies as needed. Quote extras so shells do not expand
the brackets as filename patterns:

```bash
USE_KIM=0 python3 -m pip install '.[ase]'       # ASE calculator
USE_KIM=0 python3 -m pip install '.[phonopy]'   # Phonopy and ASE
USE_KIM=0 python3 -m pip install '.[pymatgen]'  # pymatgen conversion
USE_KIM=0 python3 -m pip install '.[interop]'   # ASE and pymatgen
USE_KIM=0 python3 -m pip install '.[dev]'       # test tools
```

## Quick Start

This example uses the model from the [CLI quickstart](../../examples/quickstart/README.md).
Run from the repository root after building the Python extension. This synthetic
model checks the interface; it is not a physical Al potential.

```python
import numpy as np
import imrmlp

# Run the CLI quickstart first, from the repository root.
pot = imrmlp.Potential("build/quickstart/fitted.mtp", "mtp")
atoms = imrmlp.Atoms(
    numbers=[13, 13, 13],
    elements=["Al"],  # model species 0 = Al, NOT atomic number 13
    positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9.0, 7.1]],
    cell=np.eye(3) * 20.0,
    pbc=True,
)
pot.calculate(atoms)
print(atoms.energy)  # total eV
print(atoms.forces)  # eV/Å
print(atoms.stress)  # extensive stress, eV
imrmlp.write("build/quickstart/python-predicted.cfg", atoms)
```

The remaining snippets are usage templates requiring your own model and data.
CFG files used for evaluation or relaxation must use the model's numerical species
indices: a one-species Al MTP typically expects 0, rather than atomic number 13.
For element-labeled input such as POSCAR, one option is to read with ASE and
construct `Atoms(numbers=..., positions=..., cell=..., pbc=..., elements=[...])`
to specify model order.

## Atoms

The `imrmlp.Atoms` class wraps the C++ `Atoms` structure with property-based access
and automatic atomic-number to species-index translation.

```python
import numpy as np
import imrmlp

# Create from scratch
atoms = imrmlp.Atoms(
    numbers=np.array([13, 13, 13, 13]),
    positions=np.array([[0,0,0], [1.4,1.4,0], [2.8,0,1.4], [1.4,2.8,1.4]]),
    cell=np.eye(3) * 4.05,
    elements=["Al"],
)

# Properties — readers and setters
atoms.positions          # (N, 3) numpy array, Cartesian by default
atoms.cell               # (3, 3) numpy array
atoms.numbers            # (N,) atomic numbers
atoms.energy             # total energy in eV (None if not computed)
atoms.forces             # (N, 3) forces in eV/Å (None if not computed)
atoms.stress             # (3, 3) extensive stress/virial, eV (None if absent)
atoms.chemmap            # raw species-index list
atoms.velocities         # (N, 3) velocities, Å/ps (optional)
atoms.site_energies      # (N,) per-atom energies (optional)
atoms.natoms             # number of atoms
atoms.volume             # cell volume in Å³

# Methods
atoms.wrap()             # wrap atoms into periodic cell
atoms.copy()             # deep copy
atoms.to_fractional()    # convert to fractional coordinates
atoms.to_cartesian()     # convert back to Cartesian

# Conversion to/from ASE
ase_atoms = atoms.to_ase()
imrmlp_atoms = imrmlp.Atoms.from_ase(ase_atoms)

# Conversion to/from pymatgen (optional dependency)
pmg_structure = atoms.to_pymatgen()
imrmlp_atoms = imrmlp.Atoms.from_pymatgen(pmg_structure)
```

`from_ase()` and `from_pymatgen()` do not accept `elements`; they retain atomic
numbers as types. Before evaluating a model that uses consecutive species
indices, construct an explicitly mapped input, for example:

```python
model_atoms = imrmlp.Atoms(
    numbers=ase_atoms.get_atomic_numbers(),
    positions=ase_atoms.get_positions(),
    cell=ase_atoms.get_cell().array,
    pbc=ase_atoms.get_pbc(),
    elements=["Al"],
)
```

This construction imports geometry only; preserve existing energy/force labels
separately in training workflows. The ASE calculator accepts
`IMRMLPCalculator(..., elements=[...])` to perform the same type mapping directly.

**Note on chemmap vs numbers:** The C++ layer stores species as integer indices
(`chemmap_`). The Python `Atoms` class translates between atomic numbers and
species indices automatically. Provide the `elements` list (in chemmap order)
to enable round-trip conversion:

```python
atoms = imrmlp.Atoms(
    numbers=np.array([13, 12, 13, 12]),
    elements=['Mg', 'Al'],   # index 0 → Mg (Z=12), index 1 → Al (Z=13)
)
```

## Potential

The `imrmlp.Potential` class loads potentials through the C++
`PotentialWrapper` factory. The exact set depends on optional build features
such as OpenKIM and ACELMP support.

```python
import imrmlp
from imrmlp import Potential

# Default auto: GPM supplies its model type; legacy files fall back to mtp
pot = Potential("Al.mtp")                  # legacy MTP file
# Other legacy formats need an explicit type
pot = Potential("Al.eam.alloy", "eamalloy")
pot = Potential("potential.sw", "sw")
pot = Potential("Si.tersoff", "tersoff")
pot = Potential("ZBL", "zbl")
pot = Potential("model.kim", "openkim")

# Properties
pot.cutoff           # cutoff radius
pot.species_count    # number of chemical species
pot.pottype          # potential type string
pot.potfile          # file path

# Compute energy, forces, stress (modifies atoms in-place)
pot.calculate(atoms)

# Energy-only (faster)
energy = pot.calculate_energy(atoms)

# OpenMP parallel evaluation
pot.calculate(atoms, use_omp=True)

# Check if supports linear fitting
if pot.supports_linear_fit():
    print("This potential supports linear training")
```

### Potential type constants

The `Potential` class provides class-level constants for type strings:

| Constant | Type String |
|----------|-------------|
| `Potential.MTP` | `"mtp"` |
| `Potential.EAM_ALLOY` | `"eamalloy"` |
| `Potential.EAM_FS` | `"eamfs"` |
| `Potential.EAM_POLY` | `"eampoly"` |
| `Potential.LJ612` | `"lj612"` |
| `Potential.SW` | `"sw"` |
| `Potential.TERSOFF` | `"tersoff"` |
| `Potential.ZBL` | `"zbl"` |
| `Potential.OPENKIM` | `"openkim"` |
| `Potential.HYBRID` | `"hybrid"` |
| `Potential.HYBRID_SCALE` | `"hybrid_scale"` |
| `Potential.PAIR_TABLE` | `"pair_table"` |
| `Potential.THREEBODY_TABLE` | `"threebody_table"` |


## Structure I/O

`read`/`read_one` preserve file species indices. They do not attach an `elements`
map to the Python wrapper; do not assume `.numbers` contains atomic numbers
or call `to_ase()` without restoring mapping. For model evaluation,
construct `Atoms(..., elements=[...])` with an explicit model species order.


The `imrmlp.io` module wraps the C++ `GPIO` class, supporting 20+ file formats
with automatic format detection.

```python
import imrmlp

# Read all structures
structures = imrmlp.read("training_set.cfg")
for s in structures:
    print(s.natoms, s.energy)

# Read a single structure
atoms = imrmlp.read_one("POSCAR")

# Write structures
imrmlp.write("output.cfg", atoms)
imrmlp.write("trajectory.extxyz", [atoms1, atoms2, atoms3])

# Explicit format specification
atoms = imrmlp.read_one("input.txt", format="poscar")
imrmlp.write("output", atoms, format="mlpcfg")

# Auto-detect format from filename (available as imrmlp.io.guess_format;
# returns a format-name string, e.g. "mlpcfg" for ".cfg", or "" if unknown)
from imrmlp.io import guess_format
fmt = guess_format("structure.cfg")
```

### Supported Formats

| Format | Explicit `format` | Recognized extension / name | Read | Write |
|--------|-------------------|-----------------------------|------|-------|
| VASP POSCAR | `poscar` | `.vasp`, `POSCAR` | ✓ | ✓ |
| VASP XDATCAR | `xdatcar` | `XDATCAR` | ✓ | ✓ |
| VASP OUTCAR | `outcar` | `OUTCAR` | ✓ | |
| VASP XML | `vasprun` | `vasprun.xml` | ✓ (`read`) | |
| VASP ML_ABN | `vaspabn` | `ML_ABN` | ✓ | |
| LAMMPS trajectory | `lmptrj` | `.lammpstrj` | ✓ | ✓ |
| LAMMPS data | `lmpdat` | `.lmpdat`, `.lmp` | ✓ | ✓ |
| LAMMPS YAML | `yaml` | `.yaml` | ✓ | |
| MLIP CFG | `mlpcfg` | `.cfg` | ✓ | ✓ |
| MLIP binary CFG | `bin` | `.cfgbin`, `.bin` | ✓ | ✓ |
| Extended XYZ | `extxyz` | `.extxyz`, `.xyz` | ✓ | ✓ |
| CIF | `cif` | `.cif` | ✓ | ✓ |
| XSF | `xsf` | `.xsf`, `.axsf` | ✓ | ✓ |
| CP2K XYZ | `cp2kxyz` | `.cp2kxyz` | ✓ | ✓ |
| n2p2 data | `n2p2` | `.n2p2` | ✓ | ✓ |
| VTK | `vtk` | `.vtk` | | ✓ |
| DLPOLY4 | `dlpoly` | `.dlpoly` | | ✓ |
| GROMACS | `gro` | `.gro` | ✓ | |
| Materials Studio | `msi`, `xsd`, `car` | `.msi`, `.xsd`, `.car` | ✓ | |

`guess_format()` uses the filename, not file contents. For `data.lammps` or
`input.data`, specify `format="lmpdat"` or `format="n2p2"` as appropriate.
`trajectory.cp2k.xyz` is guessed as `extxyz`; specify `format="cp2kxyz"` for
CP2K data. Explicit content probing is separate: `imrmlp.io.detect_format(path)`
reports matching format candidates, and `read(path, detect_content=True)` uses
an unambiguous content match. The table describes file-based I/O; streaming
sessions support a narrower set (see [shared workflows](../advanced/shared-workflows.md)).

### Streaming and incremental I/O

Use these helpers when a trajectory does not fit in memory, or when frames are
produced and consumed incrementally (for example inside an active-learning loop).

```python
import imrmlp
from imrmlp import iread, write_iter, write_one, open_reader, open_writer

# iread: lazy per-frame iteration; the file opens on the first iteration
for i, atoms in enumerate(iread("trajectory.extxyz")):
    if atoms.energy is not None and atoms.energy < -100.0:
        write_one(f"picked-{i}.cfg", atoms)
    if i > 10000:
        break          # early close is allowed; no full-file check

# write_iter: stream an iterable without materializing a list
def generated():
    for frame in iread("md.extxyz"):
        yield frame

write_iter("filtered.extxyz", generated())

# open_reader / open_writer: independent sessions with explicit close
with open_reader("md.cfg") as reader:
    for atoms in reader:
        process(atoms)

with open_writer("out.extxyz", append=True) as writer:
    for atoms in new_frames:
        writer.write(atoms)   # element mapping must match the writer's
```

For an early exit, use `with contextlib.closing(iread(path)) as frames:` or
call the iterator's `close()` explicitly. Closing releases resources without
validating unread frames.

Behavior and limits:

| API | Purpose | Notes |
|---|---|---|
| `iread(path, format="", *, detect_content=False, input_units=None)` | Iterate frames lazily | Formats: `extxyz`/`xyz`, `cfg`/`mlpcfg`, `bin`/`mlpbin`, `n2p2`, `cp2kxyz`, `gro`, `xdatcar`, `xdatcar6`, `outcar`, `xsf`/`axsf`. Gzip streaming decompression follows native format support; full iteration validates the compressed trailer. Errors on malformed frames raise when consumed. |
| `write_one(path, atoms, format="")` | Write exactly one frame | Convenience wrapper around `write()`; format detection and restrictions are identical. |
| `write_iter(path, structures, format="", *, append=False)` | Write from any iterable | Supports EXTXYZ, CFG, BIN, XDATCAR, XDATCAR6, CP2KXYZ and N2P2. All frames must share one element mapping. An empty iterable does not create or truncate the target. `append=True` currently raises `ValueError`. A late failure can leave the already written prefix. |
| `open_reader(path, format="", *, detect_content=False, input_units=None)` | Bounded reader session | Supports EXTXYZ, CFG, BIN, N2P2, XSF, OUTCAR and XDATCAR/6. Use as an iterator or context manager; separate sessions may run concurrently. |
| `open_writer(path, format="", *, elements=None, append=False)` | Lazy writer session | Supports EXTXYZ, CFG, BIN, N2P2 and XDATCAR/6. `elements` is the same explicit element map as `Atoms`; `None` uses atomic-number types. `append=True` supports EXTXYZ, CFG and BIN only. No file is created before the first frame. |

`write_iter` keeps only one frame in flight, so inputs remain unchanged and memory
stays bounded. For small datasets, `read()` returns a list and `write()` accepts
a structure or list, making them convenient choices. To append frames, use
`open_writer(..., append=True)` with EXTXYZ, CFG or BIN; `write()` and
`write_iter()` reject append requests. Unsupported streaming formats raise
`ValueError`.

## Structure Relaxation

```python
from imrmlp import Potential, read_one, Relaxation

pot = Potential("Al.mtp")
atoms = read_one("initial.cfg")

relax = Relaxation(
    potential=pot,
    driver="fire",         # "lbfgs", "fire", "cg", or "gd"
    max_iter=1000,
    gtol=1e-3,             # gradient component tolerance in active coordinates
    relax_cell=True,       # variable-cell relaxation
    pressure=0.0,          # external pressure in GPa
    trajfile="relax.trj",  # optional trajectory output
)

result = relax.run(atoms)
print(f"Converged: {result.converged}")
print(f"Iterations: {result.n_iter}")
print(f"Final objective: {result.final_energy:.4f} eV")
print(f"Final potential energy: {result.final_atoms.energy:.4f} eV")
print(f"Initial energy: {result.init_energy:.4f} eV")

# In-place: atoms now hold relaxed positions
print(atoms.positions)
```

`gtol` applies to the largest gradient component in the active optimizer
coordinates. Fixed-cell Cartesian atomic coordinates give eV/Å; variable-cell
and other coordinate modes cannot be read directly as a maximum atomic force.
Other available optimizers include `lbfgspp`, `mdmin`, `quickmin`, `sqp` and `nlc`.

`RelaxResult` fields:

- `converged` — `bool`
- `n_iter` — `int`
- `init_energy` — `float` (eV)
- `final_energy` — final objective (eV), including pressure and constraint terms; potential energy is `final_atoms.energy`
- `final_atoms` — `Atoms`
- `stop_reason` — termination reason string

## NEB (Nudged Elastic Band)

```python
from imrmlp import Potential, read_one, NEB

pot = Potential("Al.mtp")
initial = read_one("initial.cfg")
final = read_one("final.cfg")

neb = NEB(
    potential=pot,
    n_images=7,            # including endpoints
    spring=5.0,            # spring constant in eV/Å²
    climb=True,            # climbing-image NEB
    optimizer="fire",      # shared C++ optimizer loop
    max_iter=1000,
    neb_ftol=1e-2,         # force tolerance in eV/Å
)

result = neb.run(initial, final)
print(result.converged, result.stop_reason, result.max_neb_force)
print(f"Converged: {result.converged}")

for i, img in enumerate(result.images):
    print(f"Image {i}: E = {img.energy:.4f} eV")
```

`NEBResult` commonly used fields:

- `converged` — `bool`
- `n_iter` — `int`
- `images` — `List[Atoms]` (includes endpoints)
- `max_perp_force` — maximum perpendicular force
- `max_neb_force` — maximum effective NEB force including spring/climbing terms
- `stop_reason` — termination reason; inspect together with `converged`

## Potential Training

### Linear Trainer

`LinearTrainer` requires the generic linear fitting interface, implemented by
`descriptor_linear`. Ordinary MTP does not expose this interface; fit its linear
coefficients with `imrmlp trainmtp train.ini --linear-fit` instead.
The following example needs labeled `training.cfg` with one Al species (index 0).
For a runnable synthetic example without external data, see
[06_train.py](../../examples-python/06_train.py): run
`PYTHONPATH=. python3 examples-python/06_train.py` from the repository root
after building the Python extension. Outputs go under `build/examples-python/06-train/`.

```python
from imrmlp import Potential, read, LinearTrainer

pot = Potential.from_settings({
    "pottype": "descriptor_linear",
    "descriptor_linear:species_count": "1",
    "descriptor_linear:descriptor:dtype": "behler",
    "descriptor_linear:descriptor:behler:cutoff": "5",
    "descriptor_linear:descriptor:behler:dispcount": "2",
    "descriptor_linear:descriptor:behler:dispselect": "1 2",
    "descriptor_linear:descriptor:behler:eta": "0.3",
})
training_set = read("training.cfg")

trainer = LinearTrainer(
    energy_weight=1.0,
    force_weight=0.01,
    stress_weight=0.0001,
    reg_L2=1e-10,          # L2 regularization
    save_to="fitted.linear",
)

# One-step train
result = trainer.train(pot, training_set)
print(f"Features: {result.feature_count}")
print(f"Energy RMSE: {result.train_erms:.4f} eV/atom")
print(f"Force RMSE:  {result.train_frms:.4f} eV/Å")

# Or two-step for inspection
trainer.assemble(pot, training_set)
# ... inspect matrix ...
result = trainer.solve()
```

`TrainResult` fields:

- `feature_count` — `int`
- `train_erms` — `float`
- `train_frms` — `float`
- `train_srms` — `float`

## Active Learning

### Matrix-Level MaxVol

```python
import numpy as np
from imrmlp import MaxVol

A = np.random.randn(100, 16)
mv = MaxVol()
selected = mv.select_rows(A, k=16)
scores = mv.compute_scores(A, selected)
```

### MaxVol Active Set Selection

```python
from imrmlp import read, ActiveSelector, Potential

pool = read("pool.cfg")

selector = ActiveSelector(
    potfile="model.mtp", pottype="mtp", mode="cfg",
    threshold=1.001,
    weight=1.0,
    pow_scale=0.5,
    max_swaps=0,           # unlimited swaps (default)
)

selector.add_candidates(pool)
selected_count = selector.select()
print(f"Selected {selected_count} structures")
print(f"Selected indices: {selector.selected_indices}")

# Grade new structures
grade = selector.grade(new_structure)
print(f"Grade: {grade:.3f}")
# 1.0 = inside active set
# 1–10 = mild extrapolation
# >>10 = extreme extrapolation

# Batch grading
grades = selector.grade_all(test_structures)
print(f"Max grade: {grades.max():.3f}")

# Persistence
selector.save("active_set.sel")
selector2 = ActiveSelector.load("active_set.sel", potential=Potential("model.mtp"))
```

`select_from_pool()` collapses the three-step sequence above into one call when
no intermediate inspection is needed:

```python
from imrmlp import read, select_from_pool

selector = select_from_pool(
    read("pool.cfg"),
    potfile="model.mtp",
    pottype="mtp",
    mode="cfg",
    threshold=1.001,
)
print(selector.selected_indices)
grade = selector.grade(new_structure)
```

It constructs an `ActiveSelector` from the potential file, adds every structure
as a candidate and runs `select()`. Keyword arguments are routed to the selector:
`threshold`, `weight`, `pow_scale`, `max_swaps`, `pottype`, `dtype`, `mode`,
`params`, `w_energy`, `w_site`, `w_force`, `w_stress` and `all_coeffs`.
Unrecognized keywords are forwarded to `ActiveSelector` unchanged and may be
rejected there. Use the explicit three-step form when candidates must be added
in batches or when the selection needs to be resumed.

@note The matrix-level `MaxVol` API is available in Python. MTP B-matrix
construction is also available through `build_bmatrix()`, while the CLI
commands `maxvol_select`, `maxvol_sample`, and `maxvol_extend` remain the
recommended route for very large production pools.

### Farthest Point Sampling (FPS)

The high-level `FPSSelector` wrapper currently does not expose initialization
of its descriptor/B-matrix builder. Its `metric` and `random_seed` arguments
are stored in Python but are not wired into the C++ selector. Use the `fps_select`
CLI with an explicit potential/descriptor configuration for structure selection;
do not treat a bare `FPSSelector().add_candidates(pool)` as a working pipeline.

### CUR Selection

```python
import numpy as np
from imrmlp import cur_select

# descriptor_matrix: (n_structures, n_features)
result = cur_select(
    descriptor_matrix,
    k=20,
    deterministic=True,
    compute_error=True,
)
print(f"Selected indices: {result.selected_indices}")
print(f"Scores: {result.scores}")
print(f"Reconstruction error: {result.reconstruction_error:.4f}")
```

`cur_select()` returns a `CURResult` dataclass:

| Field | Type | Meaning |
|---|---|---|
| `selected_indices` | `list[int]` | Row indices selected from `A`, in selection order. |
| `scores` | `np.ndarray` | Leverage score per row of `A`. |
| `reconstruction_error` | `float` | Relative reconstruction error; `-1.0` when `compute_error=False`. |

`deterministic=True` takes the top-`k` rows by score, while the default
probability-weighted sampling uses `seed` for reproducibility. With
`deterministic=False`, different seeds may select different rows.
`k` must not exceed the number of rows.

## Descriptors

```python
from imrmlp import Descriptor

# Descriptor parameters are passed through to the C++ descriptor wrappers,
# which read type-specific sub-settings. The keys must carry the descriptor
# prefix ("behler:", ...). SO3 uses the flat keys shown below.

# Behler ACSF
behler = Descriptor("behler", {
    "behler:cutoff": "6.0",
})

# Bond Order Parameters
bo = Descriptor("bondorder", {
    "bondorder:maxl": "8",
})

# SO(3) bispectrum
so3 = Descriptor("biso3", {
    "biso3_lmax": "3", "biso3_rsize": "2", "biso3_rmax": "4",
    "biso3_real_harmonics": "true",
})

# SO(4) bispectrum
so4 = Descriptor("biso4", {
    "biso4:twojmax": "3", "biso4:cutoff": "4",
    "biso4:weights": "1 1.3", "biso4:selfweights": "0.6 0.8",
    "biso4:average": "inner-density", "biso4:contraction_backend": "snap",
})
```

@note The `mtm` descriptor is not available through the Python `Descriptor`
API — it requires a `PotMTP` instance, which the binding does not pass. Use
the C++ `DescriptorMTM` or the `maxvol_*` CLI commands instead.

SO4 `inner-density` averages density coefficients before forming the fingerprint. `compute_averaged_with_gradient` returns the fixed-cell atom Jacobian; `compute_averaged_with_strain_gradient` returns the strain Jacobian. `compute_averaged_with_cell_gradient` differentiates raw cell components at fixed fractional coordinates. `compute_weighted_global_hvp` and `compute_weighted_global_hessian` provide weighted global curvature. The raw-cell and global curvature methods require `inner-density`.

Use `bo.compute(atoms)` for per-atom features with shape `(n_atoms, n_features)`
and `bo.compute_averaged(atoms)` for a structure fingerprint with shape
`(n_features,)`. Descriptor parameters define features; they do not by themselves
create an energy/force model.

## ASE Calculator Integration

Install ASE and always pass the model species order explicitly. The calculator
converts native extensive stress (eV) to ASE stress as
`-atoms.stress / atoms.volume` (eV/Å³). The example uses the
quickstart model; use a validated material potential for relaxation or MD.

```python
from ase import Atoms
from imrmlp.calculator import IMRMLPCalculator

atoms = Atoms("Al3", positions=[[7, 7, 7], [9.5, 7, 7], [7.8, 9, 7.1]],
              cell=[20, 20, 20], pbc=True)
atoms.calc = IMRMLPCalculator(
    filename="build/quickstart/fitted.mtp", pottype="mtp", elements=["Al"],
)
print(atoms.get_potential_energy())  # eV
print(atoms.get_forces())           # eV/Å
print(atoms.get_stress())           # eV/Å³, xx yy zz yz xz xy
```

## Phonopy Integration {#phonopy-integration}

Install the optional dependencies with `python -m pip install '.[phonopy]'`.
`imrmlp.phonopy.calculate_forces(phonon, calculator)` accepts a Phonopy object
with generated displacements and an `IMRMLPCalculator` or other ASE calculator.
It sets `phonon.forces` and returns the force array, preserving displacement and
supercell atom ordering. Use Phonopy's default or `vasp` calculator: lengths are
in Å and forces in eV/Å. Forces are replaced only after all frames succeed;
no drift removal or symmetrization is applied.

```python
from imrmlp.phonopy import calculate_forces

# phonon: Phonopy object; calculator: configured ASE IMRMLPCalculator
phonon.generate_displacements(distance=0.01)
calculate_forces(phonon, calculator)
phonon.produce_force_constants()
phonon.save("phonopy_params.yaml", settings={"force_constants": True})
```

Runnable example: [09_phonopy.py](../../examples-python/09_phonopy.py):

```sh
python examples-python/09_phonopy.py POSCAR fitted.mtp --elements Al Ni --supercell 3 3 3
```

Set `--elements` in model species order and use a structure relaxed for that
potential. Phonopy owns primitive/supercell mapping, force-constant fitting and
postprocessing; the example writes parameters, DOS and thermal-property files.
The CLI `phonopyfc` interface also remains available; see the
[command reference](commands.md).

See the [Phonopy API](https://phonopy.github.io/phonopy/phonopy-module.html) for subsequent analysis.

## Utilities

```python
from imrmlp.utils import (
    Z_TO_SYMBOL,        # dict: atomic_number -> element symbol
    SYMBOL_TO_Z,        # dict: element symbol -> atomic_number
    build_type_mapping,  # build species index map
    chemmap_from_numbers, # convert atomic numbers to chemmap
    numbers_from_chemmap, # convert chemmap to atomic numbers
)

# Example: element translation
al_z = SYMBOL_TO_Z["Al"]     # 13
al_sym = Z_TO_SYMBOL[13]     # "Al"

# Type mapping for a binary system
type_map = build_type_mapping(["Mg", "Al"])
# {12: 0, 13: 1}
```

## OpenMP Thread Control

```python
from imrmlp import set_num_threads, get_num_threads

print(f"Current threads: {get_num_threads()}")
set_num_threads(4)            # set explicit thread count
```

## Model Bundles and Active-Learning Sessions

`ModelBundle` exposes the GPM container without starting active learning implicitly:

```python
from imrmlp import ModelBundle

bundle = ModelBundle.load("resume.gpm")
print(ModelBundle.inspect("resume.gpm"))
ModelBundle.validate("resume.gpm")

session = bundle.create_al()          # explicit activation
result = session.evaluate(atoms)      # returns the EFS result and collects candidates
print(session.should_abort, len(session.pool))
session.save("continued.gpm")

bundle.potential.save("copy.gpm")                    # preserve the AL session
bundle.potential.save("deploy.gpm", profile="inference")
```

`ModelBundle.from_settings(settings)` builds a bundle from a flat `{key: value}`
mapping. `attach_al()` and `attach_selector()` attach an existing session or
selector; `ALSession.selector` returns the active selector used for selection.

## Shell Evaluation Jobs

`Potential.submit()` runs asynchronous jobs for `shellfile` potentials only.
Create `potential` from a shellfile configuration first. Submission copies the
input and leaves the original `atoms` unchanged:

```python
job = potential.submit(atoms)
while not job.ready():
    ...                      # do other Python work
result = job.result()        # returns a new Atoms with the computed results
```

`ShellJob.result()` releases the GIL while waiting. Submit several independent
jobs before collecting their results. In comparison, `calculate(atoms)` modifies
the input in place and returns `None`. Jobs cannot be cancelled; releasing the
last job handle waits for completion.

## Dimer Saddle Search

```python
from imrmlp import Dimer

dimer = Dimer(potential, ddr=0.01, max_iter=200)
result = dimer.run(atoms)
print(result.converged, result.exit_code, result.stop_reason)
saddle = result.saddle_atoms
```

`DimerResult` also reports `energy`, `curvature`, `max_force`,
`max_rot_gradient`, and `n_iter`. See the [saddle workflow](../advanced/saddle-workflows.md).

## Nonlinear Training

```python
from imrmlp import NonlinearTrainer

trainer = NonlinearTrainer(
    energy_weight=1.0, force_weight=0.01, stress_weight=1e-4,
    max_iterations=200, trainable_groups=["radial"],
)
result = trainer.train(potential, training_set, validation_set)
print(result.loss, result.valid_erms, result.test_erms)
```

`NonlinearTrainResult` reports train/validation/test EFS errors, optional Hessian
and HVP RMS values, the selected L2 multiplier, and the hybrid manifest path when
a fixed core is used. Extra keyword arguments use the unprefixed `fit:` key names.

## Units and XSF Fields

```python
from imrmlp.units import (
    convert_units, unit_conversion_factor, source_unit_system,
    canonicalize_atoms, pressure_to_virial_factor,
)
from imrmlp.xsf_fields import (
    GridLayout, ScalarField, DataGridBlock, VectorFieldView,
    read_xsf_fields, write_xsf_fields,
)

scale = unit_conversion_factor("energy", "eV", "hartree")
factor = pressure_to_virial_factor("GPa")
fields = read_xsf_fields("band.xsf")
```

`source_unit_system()` resolves `canonical`, `metal`, `atomic`, and `real`
labels; `canonicalize_atoms()` converts an `Atoms` object to canonical units.

## Extended Atoms and Potential Members

`Atoms` exposes magnetic moments, charges, masses, tags, initial values, total
charge, free energy, PBC flags, and selective-dynamics masks:

```python
atoms.charges, atoms.magmoms, atoms.masses
atoms.tags, atoms.initial_charges, atoms.initial_magmoms
atoms.total_charge, atoms.free_energy, atoms.pbc
atoms.selective_dynamics_mask
atoms.set_hessian(hessian); atoms.set_hvp(directions, targets)
atoms.clear_hessian(); atoms.clear_hvp()
```

`Potential` adds species and parameter access plus Hessian/HVP evaluation:

```python
potential.append_species([0, 1], ["Al", "Ni"])
potential.supports_species_append()
potential.coefficients, potential.parameter_groups
potential.calculate_many(structures, properties=("energy", "forces", "stress"))
potential.hessian(atoms)
potential.hessian_vector_product(atoms, direction)
potential.supports_analytical_hessian()
```

`hessian()` returns a Cartesian `(3*N, 3*N)` matrix;
`hessian_vector_product()` accepts and returns a one-dimensional length-`3*N`
vector. Both use atom-major xyz order and eV/Å² without mass weighting.
The default `method="auto"` uses analytical derivatives when available and
central force differences otherwise. `method="analytic"` requires model support.
Both queries preserve the input geometry and existing results.

`imrmlp.atoms` also provides `from_ase_list`, `to_ase_list`,
`from_pymatgen_list`, and `to_pymatgen_list`; `imrmlp.has_mpi` reports whether
the extension was built with MPI.

## See Also

- @ref api_core — C++ Atoms and Neighborhoods API
- @ref api_io — C++ Structure I/O API
- @ref api_potential — C++ Potential Evaluation API
- @ref api_analysis — C++ Analysis and Descriptor API
- @ref guide_install — build requirements and compilation
- @ref guide_config — configuration file format and reference
