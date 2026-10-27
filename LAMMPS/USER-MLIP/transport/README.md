# MTP transport interface for LAMMPS

This directory is a self-contained CPU/MPI LAMMPS interface. It does not
include or inherit from `USER-MLIP/opt`, does not use the modified `Atom`
class in `MLIP_FLUX`, and can coexist with all existing MLIP style names.
It supports linear short-range MTP models; nonlinear MTP-M heads and LES/SOG
long-range terms are outside this interface. Run the commands below from the
repository root.

Implemented styles:

- `pair_style mlip/transport`: analytic CPU MTP evaluation.
- `pair_style mlip/transport/omp`: OpenMP CPU evaluation with thread-private
  force, standard virial, heat-virial, and interface-current buffers.
- `pair_style mlip/transport/spline`: CPU evaluation of `RBSpline` MTP files.
- `compute ID all mlip/heat/flux`: full-system extensive heat-current vector.
- `compute ID LEFT mlip/heat/current RIGHT`: static LEFT-to-RIGHT energy rate
  plus a three-column per-atom generalized-force array.

## Installation

Install the serial and spline styles plus both computes into a clean LAMMPS
source tree:

```sh
LAMMPS/USER-MLIP/transport/install.sh /path/to/lammps/src
```

Include the OpenMP style when the LAMMPS OPENMP package is enabled:

```sh
LAMMPS/USER-MLIP/transport/install.sh --with-omp /path/to/lammps/src
```

The installer has an explicit file manifest and refuses to overwrite any
existing file. Re-run CMake (or the traditional make package/style generation)
and rebuild LAMMPS after installation. A guarded uninstall removes files only
when their SHA-256 hashes still match the installed copies:

```sh
LAMMPS/USER-MLIP/transport/install.sh --uninstall /path/to/lammps/src
```

For a manual install, copy all base files listed in `base_files` inside
`LAMMPS/USER-MLIP/transport/install.sh` into the top-level LAMMPS `src` directory. Copy the two `_omp`
files as well only for an OPENMP-enabled build. Do not copy `atom.cpp` or
`atom.h` from the historical `MLIP_FLUX` directory.

## Pair styles and standard per-atom virial

Provide one model species mapping for each LAMMPS atom type. The recommended
two-line declaration is:

```text
pair_style mlip/transport
pair_coeff * * model.mtp Al Ni
```

Element names require matching model metadata. Numeric maps also work, for
example `pair_coeff * * model.mtp 0 1`. The legacy one-line declaration remains
supported:

```text
units metal
newton on
pair_style mlip/transport model.mtp 0 1
pair_coeff * *
```

All three styles request a full neighbor list and require `newton pair on`.
For every directed site derivative

```text
r_ij = x_j - x_i
D_ij = d U_i / d r_ij
```

the force update is `f_i += D_ij`, `f_j -= D_ij`. The standard symmetric
six-component virial is

```text
W_ij = sym(-r_ij tensor D_ij)
```

and one half is assigned to the center and one half to the neighbor. Thus
`compute stress/atom NULL pair` works without a custom atom field, including
MPI ghost reduction. `centroid/stress/atom` is deliberately reported as
unavailable because its many-body partition is not implemented.

## Full-system heat current

Define the compute before a `run` command:

```text
compute J all mlip/heat/flux
thermo_style custom step c_J[1] c_J[2] c_J[3] c_J[4] c_J[5] c_J[6]
```

The pair stores the MTP site energy `U_i` and the non-symmetric heat virial

```text
B_j = -sum_i r_ij tensor D_ij .
```

The compute contracts `B_j` with the current atom velocities, avoiding the
half-step velocity ambiguity of doing that work inside the force kernel:

```text
J V = sum_i (K_i + U_i) v_i + sum_j B_j v_j .
```

Components 1--3 are the total extensive heat current `J V`; components 4--6
are its convective part. The compute performs no automatic division by volume.
It has no dependency on separate `ke/atom`, `pe/atom`, or stress computes.

This v1 definition is a total MTP heat current. It rejects pair hybrids and
systems with bonded, angle, dihedral, improper, or k-space energy terms. Force-
adding fixes are outside this compute and must not be interpreted as included
in the reported total heat current.

## Static interface current

LEFT and RIGHT must be nonempty, static, disjoint groups whose union is all
atoms:

```text
group left id 1:1000
group right subtract all left
compute Q left mlip/heat/current right
```

With `h=0` in LEFT and `h=1` in RIGHT, the pair accumulates

```text
G_j = -sum_i (h_j - h_i) D_ij
Q_LEFT->RIGHT = sum_j G_j dot v_j .
```

`c_Q` is the extensive scalar energy rate (positive LEFT to RIGHT).
`c_Q[1]`, `c_Q[2]`, and `c_Q[3]` are the per-atom components of `G_j` in force
units, suitable for dumps and later spectral analysis. Swapping LEFT and RIGHT
reverses both outputs. This is a fixed-membership control-surface current;
dynamic regions, crossing-event convection, moving surfaces, and GPU styles
are intentionally outside v1.

Heat buffers are allocated only when one of the transport computes exists.
Ghost contributions for the 9-component heat virial and every 3-component
interface channel share one reverse communication, cached per timestep.

## Tests

The formula-level finite-difference test is independent of LAMMPS:

```sh
c++ -std=c++17 -I LAMMPS/USER-MLIP/transport \
  LAMMPS/USER-MLIP/transport/tests/test_transport_tally.cpp -o /tmp/test_transport_tally
/tmp/test_transport_tally
```

The following test paths are relative to `LAMMPS/USER-MLIP/transport/`.
`tests/run_lammps_regression.sh LMP POTENTIAL` exercises serial/OMP force,
per-atom energy/virial, full heat current, interface antisymmetry, repeated
scalar/per-atom access, and optional MPI consistency with a potential whose
species 0 is valid. `tests/run_pair_parity.sh` additionally compares energy,
forces, and global virial against an installed `mlip/opt` style.
`tests/run_spline_regression.sh LMP` covers the spline style with the bundled
small test-only potential.
`tests/run_pairwise_heat_flux.sh LMP` checks all six heat-current components
against LAMMPS `compute heat/flux` in that fixture's pairwise limit.
