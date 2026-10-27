# LJ Calculation Walkthrough: Inputs, Relaxation, NEB and Properties

[中文](README_zh.md) · [Installation](../../docs/guides/installation.md)

Use this route after installing the CPU CLI if you have no model or structures
yet. A supplied script generates every input, using only Python's standard
library. Phonopy is optional for the last step. Run this page first, then use
the task guides to understand the settings.

To start with basis generation, reference labels and training, then calculate
with the same MTP, use the [MTP walkthrough](../mtp-workflow/README.md).
Both routes supply complete inputs in separate directories.

The exercise uses an explicit **Lennard–Jones (`lj612`) teaching potential**,
not a trained MTP. `Al` supplies a species label and mass; the results are not
aluminium properties. The synthetic MTP in the [training quickstart](../quickstart/README.md)
teaches fitting; do not use that tiny model as this exercise's stable crystal
potential. The two routes keep their models and inputs separate.

## 1. Prepare the environment and inputs from the repository root

Complete the CPU build and oneMKL environment setup in the
[installation guide](../../docs/guides/installation.md). Run the first command
below from the repository root containing `Makefile`; variables retain absolute paths:

```sh
imrmlp_root="$PWD"
export PATH="$imrmlp_root/build/release:$PATH"
export LD_LIBRARY_PATH="$imrmlp_root/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
imrmlp --version
python3 examples/basic-calculations/prepare.py
imrmlp_practice="$imrmlp_root/build/basic-calculations"
```

`prepare.py` writes inputs without running calculations or overwriting a nonempty
directory. To repeat, use `--work build/basic-calculations-2` and change
`imrmlp_practice` to that absolute path. Each step changes directory explicitly,
independent of where the previous step ended. Set the environment and variables
again in a new terminal.

| Subdirectory | Supplied inputs | Inputs needed later |
| --- | --- | --- |
| `relax/` | Perturbed four-atom FCC `POSCAR`, `relax.ini`, `bulk.ini` | None |
| `neb/` | Initial guesses for a 31-atom vacancy hop, endpoint relaxation settings, `neb.ini` | Relax and export endpoints first |
| `fc/`, `elastic/`, `eos/`, `phonopy/` | Model, mapping and settings where needed | Copy the relaxed result as `relaxed.vasp` |

Every directory contains `lj.ini` and `index.map`. The potential specifies one
species, `epsilon=0.1 eV`, `sigma=2.5 Å` and a 6 Å cutoff. Generated settings
already select `lj612`; no editing is required. The task guides' ordinary MTP
settings are for your MTP calculations; use the generated LJ settings on this page.

**Each `echo $?` immediately after a main calculation should print `0` before
you continue.** Otherwise inspect the corresponding log. Relaxation/NEB exit
code `2` means unconverged; inspect that step before consuming its outputs.

## 2. Evaluate and relax atoms at fixed cell

```sh
cd "$imrmlp_practice/relax"
imrmlp calc_efs POSCAR initial.xyz --potfile lj.ini --pottype lj612 --index index.map
echo $?
imrmlp relax relax.ini > relax.log 2>&1
echo $?
tail -n 5 relax.log
```

`initial.xyz` contains predictions for the perturbed input; `relaxed.xyz` contains
the optimized fixed-cell structure. Both should contain four atoms and E/F
properties. Inspect the sequence in `relax-trajectory.xyz`. Only atoms move:
cell edges remain 3.9 Å, so this is not yet the zero-pressure equilibrium cell.
See [single-point output](../../docs/guides/getting-started.md) for columns and
[relaxation](../../docs/guides/relaxation.md) for convergence.

## 3. Relax the zero-pressure cell and pass it to property calculations

`bulk.ini` explicitly reads the preceding `relaxed.xyz` and optimizes both atoms and cell:

```sh
cd "$imrmlp_practice/relax"
imrmlp relax bulk.ini > bulk.log 2>&1
echo $?
imrmlp convstr bulk.xyz bulk.vasp --index index.map
echo $?
cp bulk.vasp ../fc/relaxed.vasp
cp bulk.vasp ../elastic/relaxed.vasp
cp bulk.vasp ../eos/relaxed.vasp
cp bulk.vasp ../phonopy/relaxed.vasp
```

`bulk.vasp` contains the four-atom zero-pressure structure, with cell edges
near 3.883 Å. Each property directory's `relaxed.vasp` is a copy of this file.
Fixed-cell relaxation alone does not optimize lattice parameters. Format
conversion transfers geometry; it does not optimize it or create reference labels.

## 4. Relax both endpoints, then run NEB

This separate fixed-cell vacancy-hop task starts with a 2×2×2 FCC cell containing
32 sites, removes the origin atom and moves an adjacent atom into the vacancy,
preserving row-wise atom identity. Endpoint guesses, model and cell are generated;
you do not need to construct a vacancy from the four-atom structure above.

```sh
cd "$imrmlp_practice/neb"
imrmlp relax initial.ini > initial.log 2>&1
echo $?
imrmlp convstr initial.xyz initial.vasp --index index.map
imrmlp relax final.ini > final.log 2>&1
echo $?
imrmlp convstr final.xyz final.vasp --index index.map
imrmlp neb neb.ini > neb.log 2>&1
echo $?
cat neb.dat
```

`neb.ini` reads the exported `initial.vasp` and `final.vasp`: seven images with
31 atoms each. Expect `# status converged`, `# converged true` and a full
`# max_neb_force` no greater than `0.02 eV/Å`. The largest `ent_diff` is about
0.527 eV; equal endpoint energies give approximately equal forward/reverse barriers.
This is a teaching result for this LJ system, not an Al vacancy migration energy.
Inspect `mep.xyz`, then use the [NEB guide](../../docs/guides/neb.md) for barrier
interpretation, image convergence and restarts.

## 5. Calculate force constants, elasticity and EOS

These independent property tasks all read the four-atom structure copied in step 3.

```sh
cd "$imrmlp_practice/fc"
imrmlp calc_fc --potfile lj.ini --potential-type lj612 \
  --input relaxed.vasp --index index.map --supercell 2 2 2 \
  --reduced=false > fc.log 2>&1
echo $?
head -n 3 FORCE_CONSTANTS
```

The first line of `FORCE_CONSTANTS` should be `32 32`: both atom axes of the full
matrix contain 32 atoms after 2×2×2 expansion. Each pair has a 3×3 block in eV/Å².
This is not yet a phonon dispersion.

```sh
cd "$imrmlp_practice/elastic"
imrmlp calc_elastic --potfile lj.ini --pottype lj612 --input relaxed.vasp \
  --index index.map --settings elastic.ini \
  --elastic-output elastic.out > elastic.log 2>&1
echo $?
head -n 8 elastic.out
```

The 6×6 `Stiffness tensor (GPa)` uses Voigt order `xx yy zz yz xz xy`.
Expect `C11` around 97.9 GPa and `C12`, `C44` around 54.6 GPa; read subsequent
moduli according to their headers. Values change with potential and structure.

```sh
cd "$imrmlp_practice/eos"
imrmlp calc_eos --settings eos.ini --input relaxed.vasp --index index.map \
  --output ev.dat --output-eos eos.dat > eos.log 2>&1
echo $?
head -n 5 ev.dat
cat eos.dat
```

`ev.dat` reports volume, energy and enthalpy **per atom**, and pressure in GPa;
this differs from single-point total energy. In `eos.dat`, expect `V0` near
14.63 Å³/atom, `E0` near −0.788 eV/atom and `B0` near 0.432 eV/Å³.
Multiply the latter by 160.217662 to get roughly 69.1 GPa; `B0'` is dimensionless.
See [crystal properties](../../docs/guides/properties.md) for numerical convergence
and complete workflows.

## 6. Optional: calculate a Phonopy DOS

Install Phonopy 4 in the current environment, for example with
`python3 -m pip install phonopy` inside an activated virtual environment.
Skip this section if you do not need phonon postprocessing.

```sh
cd "$imrmlp_practice/phonopy"
phonopy-init -d --dim="2 2 2" -c relaxed.vasp
echo $?
imrmlp phonopyfc --potfile lj.ini --disp phonopy_disp.yaml \
  --index-file index.map --settings-file settings.ini > forces.log 2>&1
echo $?
phonopy --mesh="10 10 10" --dos phonopy_disp.yaml > dos.log 2>&1
echo $?
head -n 5 total_dos.dat
```

This directory's `settings.ini` selects LJ with `pot:pottype lj612`; omitting it
would try to read `lj.ini` as the default MTP. Phonopy defines displaced
supercells and `phonopyfc` writes `FORCE_SETS`; do not copy `FORCE_CONSTANTS`
from `fc/`. The two `total_dos.dat` columns are frequency (default THz) and DOS.
Inspect the whole spectrum and converge supercell/mesh before interpreting it;
negative frequency grid points at the file's start do not alone establish imaginary modes.

## Where next?

| Goal | Input source | Continue with |
| --- | --- | --- |
| Training and error evaluation | Self-contained script generates template, labels and validation data | [Training quickstart](../quickstart/README.md), [training guide](../../docs/guides/workflows.md) |
| Candidate selection | Separate script generates model, coverage and candidate pool | [Active-learning exercise](../active-learning/README.md) |
| Real material calculations | Obtain/train a suitable model and prepare target structures/mapping | [Getting started](../../docs/guides/getting-started.md) and task guides |
| Python/ASE or LAMMPS | Generate `fitted.mtp` with the training quickstart, then follow the interface page for the same geometry | [Python](../../README-python.md), [LAMMPS](../../LAMMPS/README.md) |

You do not need to convert the teaching LJ potential to MTP. For real tasks,
replace the model, potential type, mapping and structures together, retain
validation records, and validate properties and migration paths separately.
