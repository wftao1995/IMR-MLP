# Native MaxVol sampling in LAMMPS

[中文](README_zh.md)

`fix mlip/maxvol` calls the native `ActiveSelector` inside LAMMPS, reads an
existing `.sel` active set, grades complete configurations during MD and exports
candidates. No Python callback, intermediate trajectory or per-step `imrmlp`
process is needed. Grading uses the same core as `imrmlp maxvol_sample`.

Supported models include ordinary MTP, MTP-M, NN-MTP, LES/SOG and
hybrid MTP. Type names match the selected command's `--pottype` or `pot:pottype`.
Active sets created by full, stream and online selection are supported. The
model, loading settings and `index.map` must match those used to generate the
`.sel`; regenerate it after changing model parameters. Dimension checks cannot
detect a different model with the same parameter count.

## Build

From the repository root, build the MPI library and the release CLI used by the
offline examples. MKL and other dependencies are the same as for the main project:

```sh
make -j4 BUILD_MODE=mpi libimrmlp.so
make -j4 imrmlp
cmake -S LAMMPS/USER-MLIP/maxvol -B build/lammps-maxvol-plugin \
  -DLAMMPS_HEADER_DIR=/path/to/lammps/src \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/lammps-maxvol-plugin -j4
```

Use the exact source headers, MPI implementation and integer-size configuration
of the target LAMMPS executable. The default is the standard SMALLBIG build.
LAMMPS must enable its PLUGIN package. The plugin reuses the Eigen, CPU, MKL and
other ABI flags from `build/mpi/.flags`. `IMRMLP_LIBRARY` and
`IMRMLP_FLAGS_FILE` can select another matching **MPI** build. The serial
`libimrmlp.so` contains MPI stubs and cannot be linked into this MPI plugin.

Put the matching library directory first at runtime:

```sh
export LD_LIBRARY_PATH="$PWD/build/mpi${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mpiexec -n 4 /path/to/lmp -in in.md
```

Alternatively, compile `fix_mlip_maxvol.cpp` and `.h` into LAMMPS with the same
C++23, IMR-MLP ABI flags and MPI library. That route does not require
`maxvol_plugin.cpp` or `plugin load`.

## Input example

First generate a state using the same model and species order:

```sh
build/release/imrmlp maxvol_select --pool train.xyz --potfile model.mtp \
  --dtype mtm --mode nbh --index index.map
```

In an existing MD input, after `create_box` / `read_data` and the pair settings, add:

```text
plugin load /absolute/path/build/lammps-maxvol-plugin/mlip_maxvol_plugin.so

# units metal; index.map lists Al then Ni, with model species 0 and 1.
fix mv all mlip/maxvol 10 model.mtp model.mtp.sel index.map 0 1 &
    save candidates.xyz threshold-save 2.1 threshold-break 11

thermo_style custom step temp pe f_mv f_mv[1] f_mv[2]
run 10000
```

Full syntax:

```text
fix ID all mlip/maxvol NEVERY MODEL STATE INDEX TYPE_TO_SPECIES... &
    [pottype TYPE] [settings FILE] &
    [save FILE] [threshold-save VALUE] [threshold-break VALUE|none]
```

Provide one zero-based model species index per LAMMPS atom type; use `0` for a
single-species model. File paths are relative to the LAMMPS working directory.
`save` defaults to `sampled.xyz`; the save and break thresholds default to `2.1`
and `11.0`, with `0 <= threshold-save <= threshold-break`.
`threshold-break none` disables automatic stopping. The fix requires `units
metal`, three dimensions, `group all` and unique positive atom IDs. Noncontiguous
IDs, empty MPI ranks, orthogonal/restricted-triclinic cells and periodic/open
boundaries are supported. Minimization is not supported.

| Output | Meaning |
| --- | --- |
| `f_ID` | Most recent global MaxVol grade |
| `f_ID[1]` | Number of configurations saved during this fix's lifetime |
| `f_ID[2]` | Timestep of the most recent grading |

These values are intensive and are not normalized by atom count in thermo
output. They retain the previous result between grading steps. Initialization
of every `run`, including `run 0`, refreshes the grade to reflect changed
coordinates or cell. Ordinary saving occurs only at forward integration steps
that are multiples of `NEVERY`, avoiding duplicate saving at consecutive-run
boundaries.

A grade **strictly greater than** `threshold-save` appends one extxyz
configuration with `MV_grade`, `timestep`, cell, PBC and species/positions. Atoms
are sorted by ID and coordinates are shifted by LAMMPS `boxlo`, producing an
equivalent zero-origin cell. Output has no predicted E/F/stress labels and is
suitable for reference relabeling. Output is flushed every
sampling step; existing files are always appended to.

A grade **strictly greater than** `threshold-break` is saved and flushed before
a global LAMMPS error stops the run. This check also runs at initialization, so
an extrapolative initial configuration is retained. For a controlled stop, use
`threshold-break none` and LAMMPS `fix halt` with `f_ID`. The `.sel` file is
read-only: sampling does not extend the active set. Grades describe model/active
set extrapolation and are not calibrated prediction errors.

## Potential types and settings

### Configuration and neighborhood modes

For `dtype mtm`, choose `--mode cfg` or `--mode nbh` when generating the `.sel`,
or set `builder:mode cfg/nbh`. `cfg` sums atomic MTP descriptors into one
configuration row; `nbh` generates one row per neighborhood and the configuration
grade is the maximum row grade. The fix restores the mode from the saved state.
To change it, regenerate the `.sel` rather than changing the LAMMPS input.

For `dtype pot_efs`, row types are chosen with `w-energy`, `w-site`, `w-force`
and `w-stress`, rather than `mode`. Energy-only rows use the complete
configuration energy gradient. For local potentials with site-gradient support,
site-only rows grade each atomic neighborhood. Multiple row types can be
combined; the configuration grade is their maximum. Long-range potentials
support complete-configuration E/F/stress gradients. For them, `mtm/nbh` grades
only the MTP basis and does not assess extrapolation of long-range parameters.

### Model loading

| `pottype` | `MODEL` | MaxVol rows |
| --- | --- | --- |
| `mtp` | Ordinary MTP file | `mtm` or `pot_efs`; linear/all-coeffs and site rows |
| `mtp_m` | MTP-M file | Full parameter gradients with `pot_efs`; `mtm` describes only the MTP basis |
| `mtpnn1` | Prefix for `.mtp`, `.nn`, `.sc` | `pot_efs`; NN1/NN2 depth is detected, site rows supported |
| `mtp_les` | LES or SOG model file | `pot_efs` includes long-range parameters; `mtm` describes only the short-range basis |
| `hmtp` | Prefix for `_sub1`, `_sub2`, optional `_sub3` | `pot_efs`, combining submodel parameter gradients |

Except for ordinary MTP, `pot_efs` uses the model's full trainable parameter
layout, avoiding the ordinary-MTP linear shortcut that would omit nonlinear or
long-range terms. Parameter-freezing settings under `pot:` still apply.
Regenerate old `.sel` files created with the linear shortcut for derived models.
LES/SOG, MTP-M with long-range terms and `hmtp` support only energy,
force and stress rows and require `w-site 0`. Short-range MTP-M supports site
rows. Composite NN-MTP and `hmtp` reject `mtm` states.

`settings FILE` reads the same settings file as the offline command and loads the
model using its `pot:` settings. Type priority is the fix's `pottype`, then
`pot:pottype`, `builder:pottype`, and finally `mtp`. Positional `MODEL` and
`INDEX` always override the model path and type map in settings. Other paths are
relative to the LAMMPS working directory. Row weights and normalization come
from `.sel` and cannot be overridden by these `builder:` settings.

For example, the following MTP-M `select.ini` must use the same `extmodel`,
network dimensions and frozen-parameter options as training:

```text
builder:dtype pot_efs
builder:pottype mtp_m
builder:potfile model.mtp
builder:w-energy 1
builder:w-force 0.01
builder:w-stress 0.001
builder:w-site 0
pot:extmodel mtp_m_n2
pot:n_input 8
pot:n_neuron 16
pot:n_neuron2 8
```

```sh
build/release/imrmlp maxvol_select --settings select.ini --pool train.xyz --index index.map
```

```text
fix mv all mlip/maxvol 10 model.mtp model.mtp.sel index.map 0 1 &
    settings select.ini save candidates.xyz
```

MTP-M `extmodel` supports `mtp_m_fs`, `mtp_m_q2`, `mtp_m_li`, `mtp_m_n1`,
`mtp_m_n2` (or `mtp_m_nn2`), including models with LES/SOG long-range terms.
Retain their training settings such as `pot:lr_les true`. For `hmtp`,
`pot:potlevel` selects two or three submodels, defaulting to two.

Long-range boundary restrictions match the IMR-MLP core; unsupported PBC
combinations produce a global error. Full parameter and force/stress rows can
cost much more than ordinary MTP linear grading. Adjust the sampling interval
for the system size.

## MPI and runtime limits

Each rank sends owned atoms. Rank 0 sorts and builds the complete configuration,
evaluates the grade, writes output and broadcasts the result. Loading, grading
or output failures are synchronized before the next collective call. This
supports offline-equivalent configuration, force and stress rows, but descriptor
work and global configuration memory remain on rank 0. More MPI ranks do not
accelerate grading itself; use `NEVERY` to control overhead.

The fix supplies no force and does not modify the pair style or integrator. It
can accompany a separately installed dynamics pair style. Tests use `pair_style
zero` to isolate grading; GPU/Kokkos pair combinations are not covered by this
regression. Grading runs independently on the CPU.

The fix does not serialize state into a LAMMPS restart. After `read_restart`,
redeclare it with the same `.sel`. Candidate output continues appending and the
saved-count starts from zero. Multiple fixes or LAMMPS partitions must not write
the same output file concurrently.

## Regression tests

Build release `imrmlp` and provide an MPI LAMMPS with PLUGIN support to CMake:

```sh
make -j4 imrmlp
cmake -S LAMMPS/USER-MLIP/maxvol -B build/lammps-maxvol-plugin \
  -DLAMMPS_HEADER_DIR=/path/to/lammps/src \
  -DLAMMPS_EXECUTABLE=/path/to/lmp
cmake --build build/lammps-maxvol-plugin -j4
ctest --test-dir build/lammps-maxvol-plugin --output-on-failure
```

Tests compare against offline `maxvol_sample` at 1/2/4 ranks, including empty
ranks, reverse type mappings, noncontiguous IDs, nonzero `boxlo`, orthogonal and
triclinic boxes, PBC, descriptor/EFS weights, sampling intervals, read-only state,
append output, same-step geometry changes, saving before stopping and coordinated
MPI failures. Family tests generate small deterministic models for NN1/NN2,
five MTP-M heads, LES/SOG, MTP-M plus LES/SOG, two/three-submodel MTP,
type aliases and incompatible-state errors. The core
`ut_maxvol_mtp_family` test independently checks E/F/stress gradients by parameter
finite differences. The LAMMPS regression used the MPI/PLUGIN build of
22 Jul 2025 Update 5.
