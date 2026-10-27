# Potential Types and File Formats {#reference_potential_types}

[中文](potential-types_zh.md)

The potential factory dispatches on `pot:pottype` (or the flat `pottype` key for
commands that accept it). For a legacy potential, an empty value or `auto` selects `mtp`; GPM files infer
their type from embedded metadata. All types below
are registered in `potentials/potential_wrapper.cpp`; the table lists the input
artifact each type expects and the dedicated documentation, when one exists.

## Selection table

| `pottype` | Family | Input artifact | Notes |
|---|---|---|---|
| `mtp` | Moment tensor potential | `.mtp` text file | Default type; see @ref theory_mtp |
| `mtpnn1` | Neural-network MTP | `.mtp` template plus `.nn` weights | Create the template with `nnmtp`, train with `trainnlpot` |
| `mtp_m` | Multichannel MTP with selectable energy heads | `.mtp` containing channel and energy-head parameters | Initialize with `initmtpm`; see [MTP_M theory](../theory/mtp-m.md) |
| `hmtp` | Two-component hybrid MTP | Source MTP files in `pot:potfile` | Set `pot:potlevel`; components read `source_sub1`/`source_sub2` |
| `shellfile` | External shell provider | Provider configuration | Settings under `shellfile:`; see the shell provider section |
| `eamalloy` | EAM alloy | EAM alloy setfl table | Species order follows the potential file |
| `eamfs` | Finnis–Sinclair EAM | EAM FS table | — |
| `eampoly` | Polynomial EAM | EAM parameter file | Used in hybrid examples |
| `openkim` | OpenKIM model | KIM model name in `pot:kim_model` | Requires a build with `USE_KIM` |
| `acelmp` | ACE/LAMMPS potential | ACE model | Requires `WITH_ACELMP`; no Makefile switch is currently exposed, so it must be enabled manually in `CXXFLAGS` |
| `mindist` | Minimum-distance penalty | None | Parameters `pot:cutoff`, `pot:factor`, `pot:power` |
| `lj612` | Lennard–Jones 6-12 | None | Parameters from the potential section |
| `zbl` | Ziegler–Biersack–Littmark | None | Usually the fixed core of a `hybrid` model |
| `mtp_les` | MTP + long-range electrostatics | LES-enabled `.mtp` | See [MTP+LES v2](les-v2.md) and [SOG](sog.md) |
| `descriptor_linear` | Linear descriptor potential | Descriptor settings | Train with `train_linear` |
| `pair_table` | Tabulated pair potential | Table written by `pot2table` | Pair tables include `-dE/dr` |
| `threebody_table` | Tabulated three-body potential | Table written by `pot2table` | Equal-bond sampling; see `pot2table` |
| `sw` | Stillinger–Weber | SW parameter file | — |
| `tersoff` | Tersoff | Tersoff parameter file | — |
| `hybrid` | Sum of sub-potentials | `pot1:`, `pot2:`, ... settings | See @ref guide_config |
| `hybrid_scale` | Smoothly switched sum | `pot1:`, `pot2:`, ... plus switching keys | Used for MTP + ZBL cores |

## Minimal examples

Each block below is a separate potential configuration. Choose one for your
settings file.

```text
# MTP
[pot]
pottype mtp
potfile potential.mtp

# OpenKIM
[pot]
pottype openkim
kim_model  Sim_LAMMPS_...

# MTP + ZBL core
[pot]
pottype hybrid
pot1:pottype mtp
pot1:potfile potential.mtp
pot2:pottype zbl
pot2:z 6 13
pot2:r0 0.8
pot2:r1 2.0
```

```bash
# Convert a trained MTP to a spline table for LAMMPS
imrmlp mtp2spline potential.mtp spline_output

# Inspect a GPM bundle that embeds the effective potential
imrmlp model_info --input model.gpm
```

## Shell provider (`shellfile`)

`shellfile` turns any program that can read a structure and report energy,
forces and stress into an IMR-MLP potential: a DFT code, a model from another
package, or a wrapper script around both. IMR-MLP writes the structure into a
job directory, runs your scripts there, and reads the result back — the rest of
the program sees the same EFS interface as `mtp`.

### Settings

| Key | Default | Meaning |
| --- | --- | --- |
| `shellfile:runscript` | — | Main script; it must produce `read_outfile` in the job directory |
| `shellfile:prescript` | empty | Runs after the structure is written, e.g. to prepare INCAR/KPOINTS/POTCAR |
| `shellfile:postscript` | empty | Runs after the result is read, e.g. to remove scratch files |
| `shellfile:index` | `index.map` | Element map file |
| `shellfile:write_posfile` | `POSCAR` | Structure file written into the job directory |
| `shellfile:write_posformat` | `POSCAR` | Structure format (`poscar`, `xsf`, `cif`, `extxyz`, ...) |
| `shellfile:read_outfile` | `output.cfg` | Result file read from the job directory |
| `shellfile:read_outformat` | `mlpcfg` | Result format (`extxyz`, `mlpcfg`, `poscar`, `xsf`, `cif`, ...) |
| `shellfile:read_energy` | `false` | Read the total energy from `engfile` instead of the result file |
| `shellfile:engfile` | `eng.dat` | Energy file; its first number is the total energy in eV |

### How scripts are called

- Every script runs as `bash SCRIPT ABSOLUTE_JOB_DIRECTORY`. `runscript` is
  required; `prescript` and `postscript` are optional. A non-zero exit aborts
  the evaluation.
- Order: write structure → `prescript` → `runscript` → read result →
  `postscript`.
- Scripts stay in the submitting directory, so they keep their own relative
  inputs (POTCAR, executables) and reach the structure and results through the
  job-directory argument.
- The last frame of `read_outfile` is used.
- `index.map` lists one element name per line in species order; the first line
  is species 0. A missing file warns and leaves the default mapping.
- Job directories are kept: synchronous runs use `run-job-omp-thread-<n>` (plus
  `mpi-rank-<r>` in MPI builds), asynchronous runs `run-job-async-XXXXXX`. Open
  one when a script misbehaves — it holds exactly what IMR-MLP wrote and what
  your script left behind.

### Example 1: check the wiring with a self-contained script

Replace the calculator with a script that echoes the structure back. Once this
runs, paths, formats and the protocol are right, and only the real program is
left to plug in.

```text
work/
├── index.map
├── run.sh
└── frame.xyz                     # the structure you want to evaluate
```

`index.map` — one element per line, in species order:

```text
Al
Ni
```

`run.sh` — the job directory is the only argument:

```bash
#!/bin/bash
set -e
cp "$1/input.xyz" "$1/output.xyz"
```

What IMR-MLP writes for the script to read, `input.xyz`:

```text
2
Lattice="10 0 0 0 10 0 0 0 10" pbc="T T T" Properties=species:S:1:pos:R:3
Al 0 0 0
Ni 2 0 0
```

Element names come from `index.map`, so the calculator sees species 0 as Al and
species 1 as Ni.

```python
import numpy as np
from imrmlp import Atoms, Potential

potential = Potential.from_settings({
    'pottype': 'shellfile',
    'shellfile:index': 'index.map',
    'shellfile:runscript': 'run.sh',
    'shellfile:write_posfile': 'input.xyz',
    'shellfile:write_posformat': 'extxyz',
    'shellfile:read_outfile': 'output.xyz',
    'shellfile:read_outformat': 'extxyz',
})

atoms = Atoms(numbers=[13, 28], elements=['Al', 'Ni'],
              positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=np.eye(3) * 10)
potential.calculate(atoms)
print(atoms.energy, atoms.forces, atoms.stress)

job = potential.submit(atoms)     # the same evaluation, without blocking
print(job.run_directory)          # open it: input.xyz and output.xyz are there
print(job.result().energy)
```

The echo script writes no energy or forces, so this example validates paths and
formats only: `atoms.energy`, `atoms.forces` and `atoms.stress` stay `None`
until a real calculator writes them.

### Example 2: drive VASP from a settings file

`shell.ini`:

```text
[pot]
pottype shellfile
shellfile:index index.map
shellfile:prescript prepare.sh
shellfile:runscript run_vasp.sh
shellfile:postscript clean.sh
shellfile:write_posfile POSCAR
shellfile:write_posformat poscar
shellfile:read_outfile result.xyz
shellfile:read_outformat extxyz
```

`prepare.sh` — the inputs that do not change between structures:

```bash
#!/bin/bash
set -e
cp INCAR KPOINTS POTCAR "$1"
```

`run_vasp.sh` — run the code, then convert its output:

```bash
#!/bin/bash
set -e
cd "$1"
mpirun -np 4 vasp_std
python3 /opt/tools/vasp2xyz.py vasprun.xml result.xyz
```

`clean.sh` — keep only what you want to read afterwards:

```bash
#!/bin/bash
set -e
rm -f "$1"/CHG* "$1"/WAVECAR "$1"/vasprun.xml
```

Use it like any other potential:

```bash
imrmlp calc_efs --settings=shell.ini --input=frame.xyz --output=out.cfg
```

The result frame carries the numbers IMR-MLP needs — here in `extxyz`:

```text
2
Lattice="10.0 0.0 0.0 0.0 10.0 0.0 0.0 0.0 10.0" Properties=species:S:1:pos:R:3:forces:R:3 energy=-6.42 virial="0.12 0 0 0 0.12 0 0 0 0.12" pbc="T T T"
Al 0.0 0.0 0.0 0.02 0.0 0.0
Ni 2.0 0.0 0.0 -0.02 0.0 0.0
```

`virial` is taken as-is; `stress` is read as a density and scaled by the cell
volume.

### Example 3: a calculator that only prints the energy

Keep writing the result file for forces and stress, and take the energy from a
separate file:

```text
shellfile:read_outfile result.xyz
shellfile:read_outformat extxyz
shellfile:read_energy true
shellfile:engfile eng.dat
```

`eng.dat` — the first number is the total energy in eV, the rest is ignored:

```text
-6.4218   eV
```

The energy file must exist by the time `runscript` returns; a missing or
unparsable file aborts the evaluation.

### Example 4: ship it as one GPM file

Declare what the scripts need, then export — scripts and resources travel inside
the bundle:

```text
shellfile:dependencies vasp vaspkit
shellfile:resource:POTCAR/Al /opt/potentials/Al/POTCAR
shellfile:resource:POTCAR/Ni /opt/potentials/Ni/POTCAR
```

```bash
imrmlp model_convert --settings=shell.ini --output=deploy.gpm
```

The receiving machine only needs the declared dependencies on `PATH`. Bundled
files sit next to the extracted scripts, so resolve them from the script
location, not from the working directory:

```bash
#!/bin/bash
set -e
here="$(cd "$(dirname "$0")" && pwd)"
cp "$here/POTCAR/Al" "$1/POTCAR"
```

Dependencies must be declared before export, and resources must be regular files
whose relative names do not escape the job directory. Extracted scripts live as
long as the loaded potential.

### Concurrent evaluations

`submit()` runs an evaluation in its own `run-job-async-XXXXXX` directory and
returns immediately, which is how you fan out over several structures:

```python
jobs = [potential.submit(frame) for frame in frames]
results = [job.result() for job in jobs]
```

If you do not need polling, `calculate_many()` evaluates a batch synchronously
and returns the results in order; it also accepts ASE `Atoms` and pymatgen
structures:

```python
results = potential.calculate_many(frames)
```

`ShellJob.ready()` polls without blocking and `result()` waits, releasing the
GIL. Each call returns a new `Atoms`; a failed calculation repeats its error on
later calls. Releasing the last job handle waits for completion — there is no
cancellation. The C++ equivalent is `PotShellCalc::submit()` in
`potentials/potential_shell.h`.

## Choosing a type

- Start from `mtp` for short-range chemistry; use `mtp_m`, `hmtp`, or
  `mtpnn1` when the model needs multiple levels or a neural network head.
- Use `mtp_les` for long-range electrostatics, and
  `hybrid`/`hybrid_scale` to combine a machine-learning core with a repulsive
  short-range correction.
- Use `eamalloy`, `eamfs`, `eampoly`, `sw`, or `tersoff` to load classical
  parameterizations, and `pair_table`/`threebody_table` for tabulated models.
- Wrap any supported model in a GPM bundle with `model_convert` when a single
  deployable artifact is required.

## Related

- @ref guide_commands for per-command options such as `pot2table` and `adjustmtp`
- @ref guide_config for `pot:`, `hybrid`, and `potN:` keys
- [Python API](python.md) for `Potential` type constants and `submit()` shell jobs
