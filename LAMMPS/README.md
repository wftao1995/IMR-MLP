# LAMMPS Interface

[中文](README_zh.md)

**For ordinary short-range MTP / MTP-M molecular dynamics or relaxation, start
with the CPU style `mlip/opt`.** It supports MPI, needs no `libimrmlp`, and does
not require conversion to spline or GPM. Run the minimal example below first,
then choose another interface if you need CPU threads, GPUs or special
observables. Models with long-range contributions, such as LES/SOG,
have dedicated choices later on this page.

The project is IMR-MLP and the CLI is `imrmlp`; LAMMPS retains the `mlip/...` style names.

## Start with the common short-range workflow

### 1. Install the CPU interface

Prepare a clean LAMMPS source tree, Python 3, CMake, a C++ compiler, MPI and Eigen headers.
Run from the IMR-MLP repository root, replacing the LAMMPS and Eigen paths:

```sh
python3 LAMMPS/USER-MLIP/install.py /path/to/lammps/src
cmake -S /path/to/lammps/cmake -B /path/to/lammps/build-imrmlp \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON \
  -DCMAKE_CXX_FLAGS="-I/usr/include/eigen3"
cmake --build /path/to/lammps/build-imrmlp -j8
```

This route installs interface sources directly into LAMMPS. You do not need to
build the IMR-MLP executable or training library first. Without MPI, select
`-DBUILD_MPI=OFF` for a serial build.

### 2. Run a single-point calculation with your model

Provide a trained model suitable for your elements and simulation conditions,
and a LAMMPS data file including atom masses. This example assumes two atom
types: type 1 is Al and type 2 is Ni, with matching element metadata in the
model. Save the input as `in.mtp`:

```text
units metal
atom_style atomic
boundary p p p
newton on
read_data system.data

pair_style mlip/opt
pair_coeff * * model.mtp Al Ni

thermo 1
thermo_style custom step atoms pe press
run 0
```

```sh
/path/to/lammps/build-imrmlp/lmp -in in.mtp
```

`model.mtp` and `system.data` are placeholders for your files. For a one-element
Al model, use `pair_coeff * * model.mtp Al`. If the model lacks element metadata,
see “Element mapping and older models” below. Compare energy, forces and
pressure with IMR-MLP on the same configuration before adding the integrator,
thermostat or barostat needed for MD.

### 3. Compare CLI and LAMMPS without your own data

Complete [CPU CLI setup](../docs/guides/installation.md) and the
[training quickstart](../examples/quickstart/README.md), then run from the IMR-MLP
repository root. Reuse an existing successful `build/quickstart/`; substitute
your actual output path if you used a custom `--work`.

```sh
mkdir -p build/lammps-first-run
cp build/quickstart/fitted.mtp build/quickstart/valid-input.cfg \
  build/quickstart/index.map build/lammps-first-run/
cd build/lammps-first-run
imrmlp convstr valid-input.cfg single.vasp --index index.map --force-one --select 0
imrmlp convstr single.vasp system.data --index index.map --output-format lmpdat
imrmlp adjustmtp fitted.mtp named.mtp --elements Al
imrmlp calc_efs single.vasp cli.xyz --potfile named.mtp --pottype mtp \
  --index index.map --input-format POSCAR --output-format extxyz
```

Use a fresh `build/lammps-first-run` directory. `--select 0` extracts the first
of four frames; `--force-one` writes that POSCAR exactly as `single.vasp`,
avoiding numbered filenames from multi-frame export. `adjustmtp` supplies name
mapping for this known single-species teaching model. Save as `in.demo` there:

```text
units metal
atom_style atomic
boundary p p p
newton on
read_data system.data
mass 1 26.9815385
pair_style mlip/opt
pair_coeff * * named.mtp Al
thermo 1
thermo_style custom step atoms pe press
run 0
write_dump all custom forces.dump id type x y z fx fy fz modify sort id
```

The exported data omits masses by default, so set `mass` explicitly. Still in
that directory, run the LAMMPS executable containing the installed interface:

```sh
/path/to/lammps/build-imrmlp/lmp -in in.demo
```

Expect 3 atoms and `PotEng` near `0.46195372 eV` in `log.lammps`, matching
`energy` in `cli.xyz`. Sorted by id, `fx fy fz` in `forces.dump` should match
the CLI `forces` columns within text-rounding precision. LAMMPS `metal` pressure
is in bar; do not compare it directly with GPa. This synthetic model checks
the interface and is not suitable for real Al MD. Run `cd ../..` afterwards
to return to the IMR-MLP repository root.

## Choosing a short-range interface

This table focuses on ordinary short-range MTP and short-range MTP-M.
**Choose an interface compatible with your model, then compare performance;
changing interfaces does not improve the model's physical accuracy.**

| Interface | Scope and advantages | Costs and limits | When to choose it |
| --- | --- | --- | --- |
| **`mlip/opt`** | CPU/MPI with analytic radial bases; ordinary MTP, short-range MTP-M li/q2/fs/n1/n2 heads and their GPM bundles; no `libimrmlp` | No LES/SOG long-range terms or transport heat-current computes | **Default starting point for ordinary MD and relaxation** |
| `mlip/opt/omp` | Same models as `opt`; MPI + OpenMP gives each rank multiple CPU threads | Requires OPENMP and a thread count; per-thread workspaces cost memory, and more threads need not be faster | Compare pure MPI with MPI × threads on CPU nodes |
| `mlip/gpudirect` | CUDA or HIP/DCU; independent short-range MTP/MTP-M inference without `libimrmlp` | Needs a GPU toolchain; no LES/SOG or transport computes; MTP-M uses FP64; transfer and launch costs can offset gains for small systems | Short-range MD on available GPUs, with benchmarks on your model and system |
| `mlip/spline` | CPU/MPI; preconverted `RBSpline` ordinary MTP replaces analytic radial evaluation with interpolation | Requires conversion and introduces grid interpolation error; validate energy, forces and MD stability; not a general MTP-M interface | Try after identifying radial evaluation cost and accepting interpolation validation |
| `mlip/native` | Uses IMR-MLP's native model implementation; preserves supported legacy neural, hybrid and more complete model compositions; CPU/CUDA/HIP for supported models | Links matching MPI `libimrmlp` and its runtime dependencies; backend coverage depends on the model; more deployment setup | Models unsupported by `opt` / `gpudirect`, or integration with a native model workflow |
| `mlip` | Original CPU interface for existing inputs | Do not assume the extended model support of `opt` or native | Maintain existing calculations; start new work with `opt` |

Neither GPU nor OpenMP has a fixed speedup. Measure your model, atom count and
hardware, recording MPI ranks, threads and output frequency. Ordinary energy,
force and pressure calculations do not require the transport interface.
See the [CPU style guide](USER-MLIP/opt/README.md),
[standalone guide](USER-MLIP/standalone/README.md) and
[native guide](USER-MLIP/native/README.md) for complete model restrictions.

### Using OpenMP

Add `--with-omp` to the copy installer and `-DPKG_OPENMP=ON` to the CMake
configuration, then rebuild. Put `package omp 4` before `read_data` and change
the pair declaration to:

```text
pair_style mlip/opt/omp
pair_coeff * * model.mtp Al Ni
```

This uses four threads per MPI rank; plan ranks and threads against the CPU
cores allocated to the job. If the installer has already been used, uninstall
before reinstalling, as described under installation maintenance below.

### Using spline

Try conversion only for an ordinary MTP, then compare the original and converted
models on the same configurations:

```sh
imrmlp mtp2spline model.mtp model.rbspline.mtp
```

```text
pair_style mlip/spline
pair_coeff * * model.rbspline.mtp Al Ni
```

See the [mtp2spline command](../docs/reference/commands.md) for grid spacing and
other conversion options. Conversion does not extend support to other model families.

## Installation routes and GPU deployment

**The style name chooses the evaluator; the installation route puts it into
LAMMPS.** For example, both standalone and copy installation can provide
`mlip/opt`; standalone is not another potential model. Choose one route per
LAMMPS source tree/build to avoid installing duplicate style names.

| Installation route | Suitable use | Dependencies and tradeoffs |
| --- | --- | --- |
| CPU copy installation on this page | Existing LAMMPS sources; install CPU short-range, LES/SOG, transport and harmonic-reference interfaces together | Modifies LAMMPS `src`; needs Eigen headers and a rebuild; these CPU styles need no `libimrmlp` |
| [Standalone standard `lmp` build](USER-MLIP/standalone/README.md) | Deploy only short-range MTP/MTP-M on CPU/OpenMP or CUDA/HIP; export a source package for compute nodes | No training library, MKL or system Eigen; uses bundled Eigen headers; only supported short-range styles; standard `lmp` entry point leaves upstream sources unchanged |
| [Native standard `lmp` build](USER-MLIP/native/lammps/README.md) / [native plugin](USER-MLIP/native/README.md) | Native model compositions, long-range models or their supported GPU paths | Links matching MPI `libimrmlp`; plugins also need PLUGIN in the target LAMMPS and matching version, MPI and compiler ABI |

For short-range GPUs, start with the standalone standard `lmp` build. This
NVIDIA CUDA example requires CMake 3.22+; replace `80` with your device architecture:

```sh
cmake -S LAMMPS/USER-MLIP/standalone/lammps -B build/lmp-standalone-cuda \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/lmp-standalone-cuda -j8
```

Replace the pair declaration in the single-point input above, then run with
the newly built `lmp`:

```text
pair_style mlip/gpudirect device cuda precision fp64 gpu_id auto
pair_coeff * * model.mtp Al Ni
```

HIP/DCU uses `-DMTP_BACKEND=HIP`, `CMAKE_HIP_ARCHITECTURES` and `device hip`.
CPU deployment uses `-DMTP_BACKEND=CPU`, with `-DMTP_OPENMP=ON` for OpenMP.
GPU builds retain the CPU styles for comparisons within the same executable.
See the [standalone guide](USER-MLIP/standalone/README.md) for model and radial
basis restrictions, memory controls, device validation and plugin deployment.

## Element mapping and older models

Give one model element per LAMMPS atom type in type 1, 2, ... order, regardless
of the order atoms appear in the data file. For type 1 = Ni and type 2 = Al:

```text
pair_style mlip/opt
pair_coeff * * model.mtp Ni Al
```

Names must match the model's declared element list. Do not mix names and
numbers in one command; `NULL` is rejected. Three atom types need three
entries, such as `Al Ni Co`; multiple types may map to the same model element.
The same grammar applies to the other `mlip/...` pair styles on this page.

For older models without element metadata, recover the species order from the
training setup first, then use zero-based numeric indices:

```text
pair_coeff * * legacy.mtp 0 1      # type 1 -> species 0, type 2 -> species 1
```

For an ordinary MTP with verified species order `Al Ni`, write metadata explicitly:

```sh
imrmlp adjustmtp legacy.mtp named.mtp --elements Al Ni
```

Check predictions before switching to names. Re-saving alone cannot recover
unknown element identities. The original declaration
`pair_style mlip/opt model.mtp 0 1` followed by `pair_coeff * *` remains supported;
declare the model once. `fix mlip/maxvol` has its own numeric mapping grammar,
documented in its dedicated guide.

## Long-range and native models

Use these choices for models with long-range contributions or native model
compositions. Ordinary short-range MTP needs no charge or latent-variable compute.

| Model and requirement | Interface and advantage | Limits and selection criteria |
| --- | --- | --- |
| Linear MTP short-range head + Coulomb LES v2 / SOG v3, v4 on CPU | [`mlip/opt-les`](USER-MLIP/opt-les/README.md): dedicated CPU/MPI implementation without `libimrmlp` | Requires full 3D periodicity; no nonlinear MTP-M heads; redeclare model and computes after restart |
| Long-range MTP-M, legacy neural models, MTP hybrids or long-range GPU execution | [`mlip/native`](USER-MLIP/native/README.md): native models and their CPU/GPU evaluators | Links MPI `libimrmlp`; saved layout and model/device combination must be supported; no CPU fallback |

For example, SOG through the CPU copy installation:

```text
pair_style mlip/opt-les
pair_coeff * * sog.mtp Ni Al
compute latent all mlip/sog/latent
```

Coulomb LES uses `compute q all mlip/les/charge`. SOG latent variables are
dimensionless and must not be interpreted as physical charges. These models
include long-range contributions; a short-range style is not an equivalent substitute.

Loading GPM through the native plugin (omit `plugin load` when embedded in `lmp`):

```text
plugin load /path/to/mlip_native_plugin.so
# Define the pair after units, box setup and read_data as above:
pair_style mlip/native device cpu precision fp64 pottype auto
pair_coeff * * model.gpm Al Ni
run 0
```

GPM infers the potential type; legacy formats need an explicit `pottype` and may
need `settings` and companion resources. Follow the [native guide](USER-MLIP/native/README.md)
for charge/latent computes, GPU placement and total-charge constraints. GPM is
a model file format; converting to it does not expand interface model coverage.
See the [GPM model guide](../docs/reference/gpm-v1.md).

## Heat current, active learning and thermodynamic integration

Add these features for the task that needs them; ordinary MD does not require them.

| Task | Entry point and advantage | Scope and limits |
| --- | --- | --- |
| Thermal conductivity, full-system heat current or static interface energy current | [`mlip/transport`, `/omp`, `/spline`](USER-MLIP/transport/README.md) with dedicated heat computes | CPU/MPI; only linear short-range MTP / its spline form; no MTP-M, LES/SOG or GPU; total heat-current compute rejects pair hybrids and additional bonded/kspace energy terms |
| Identify unfamiliar configurations during MD and collect candidates | [`fix mlip/maxvol`](USER-MLIP/maxvol/README.md) added to an existing pair | Separate MPI `libimrmlp` plugin; needs a matching active set prepared in advance; scores and saves candidates without changing pair forces or generating reference labels |
| Harmonic reference and thermodynamic integration | [`harmonic/reference` workflow](../docs/advanced/harmonic-ti.md) | Included in the CPU copy installation; prepare the reference structure, force constants and TI settings |

Minimal heat-current declaration (replace the ordinary `opt` pair):

```text
pair_style mlip/transport
pair_coeff * * model.mtp Al Ni
compute J all mlip/heat/flux
```

The compute reports full-system extensive heat current; handle volume
normalization according to the analysis method. Loading GPM does not enable
active learning automatically. See the MaxVol guide for model coverage, active
set preparation and candidate output. GPU pairs with the CPU MaxVol scorer
require separate validation.

## Changing models, restarting and installation maintenance

To replace a loaded model, delete dependent observable computes first, then
clear the pair and declare it again:

```text
pair_style none
pair_style mlip/opt
pair_coeff * * other.mtp Al Ni
```

Recreate the needed computes afterward. Short-range MTP-M/GPM restart through
`opt` / `opt/omp` / `gpudirect` still requires the original model path to be
readable with unchanged contents; native also needs the referenced models and
settings files. Prefer absolute model paths when restarting in another directory.
For `opt-les`, redeclare the model and computes after restart. See each style's
guide for its exact restart restrictions.

The CPU installer never overwrites existing files. For updates, uninstall with
hash verification first, then install files from the same source version:

```sh
python3 LAMMPS/USER-MLIP/install.py --uninstall /path/to/lammps/src
```

Reconfigure and rebuild LAMMPS after installation or removal. Do not mix this
installer with the transport-only installer on the same source tree. MaxVol and
native plugins are not included in the CPU copy installation.

## Interface validation and developer resources

At first deployment, compare energy, forces and all six virial components on
the same configuration, then validate short MD runs under your target conditions.
For interface development or updates, use the regression entry points below.
Passing synthetic fixtures does not establish a potential's material applicability.

- CPU/LES/SOG: [build and MPI regression guide](USER-MLIP/opt-les/README.md).
- Cross-style parity: [opt/transport comparison script](USER-MLIP/transport/tests/run_pair_parity.sh);
  `--all-cpu` in the [CPU build script](USER-MLIP/opt-les/tests/build_regression.py) builds the complete test runner.
- Element mapping, old/new syntax and MPI consistency: [metadata regression script](USER-MLIP/opt/tests/run_metadata_regression.py).
- Short-range CPU/OpenMP/GPU and restart: [standalone tests](USER-MLIP/standalone/README.md).
- Native models and GPUs: [native regression guide](USER-MLIP/native/README.md).
- Heat current and spline: [transport regression guide](USER-MLIP/transport/README.md);
  harmonic reference: [hyb-ti regression guide](USER-MLIP/hyb-ti/README.md).

## Model metadata reference

Models record their element order in a single-line JSON tag. For example:

```text
potential_tag = {"schema":"gpml.mtp.metadata","version":1,"elements":["Al","Ni"],"type":"MTP","model":"linear"}
```

The schema name is intentionally shared; it is not renamed for IMR-MLP.
MTP-M uses `type: "MTP_M"`, its head name in `model`, integer `q`,
`model_coeffs_count`, and `nn.activation` / `nn.hidden_layers` for neural heads.
LES/SOG uses `type: "MTP_LES"` and `long_range.kernel` / `long_range.channels`;
long-range MTP-M retains `type: "MTP_M"` with the separate `long_range` field.

Existing empty/text tags and `imrmlp_meta_v1:` tags remain readable. Saving writes
the canonical schema; old text is preserved in `legacy_tag`, and extension fields
are retained. Old IMR q names/definitions and head extensions are retained in
`imrmlp_q` and `imrmlp_external_model`. Legacy `# mtp_*` footer fields remain
for older GPU readers, and are checked against JSON. Invalid versions, duplicate
keys/elements, inconsistent dimensions and conflicting element orders fail.
A species expansion must explicitly provide every output element with
`imrmlp adjustmtp ... --neworder ... --elements ...`.

## License

The LAMMPS-derived adapters retain their original Sandia/LAMMPS notices and
are distributed under [GPL version 2](COPYING.GPL-2.0). They are not relicensed
under the repository's MIT license. New integration changes carry the
IMR-MLP contributors copyright; headers identify applicable licenses.
The independently maintained `runtime/mtp` core and the project helper
headers copied by the installer retain their MIT notices. A combined
LAMMPS executable must be redistributed consistently with the GPL and its
other dependency licenses. See [third-party notices](../THIRD_PARTY_NOTICES.md).
