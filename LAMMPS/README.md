# LAMMPS Interface

[中文](README_zh.md)

IMR-MLP is exposed to LAMMPS through a set of `mlip/...` pair styles. Besides the
CPU copy installation described on this page, standalone and native installation
routes are available; see Section 3.

For ordinary short-range MTP and MTP-M models in MD or relaxation, start with the
CPU style `mlip/opt`: it supports MPI, needs no `libimrmlp`, and requires no prior
conversion to spline or GPM. Complete the minimal workflow in Section 1 first,
then move to another interface if you need CPU threads, GPUs or special
observables. Models with long-range contributions, such as LES/SOG, are covered
in Section 5.

## 1. Quick start: minimal short-range workflow

The steps below install the interface and run a single-point calculation.
They apply to short-range MTP and MTP-M models.

### 1.1 Install the CPU interface

Prerequisites are a clean LAMMPS source tree, Python 3, CMake, a C++ compiler,
MPI and Eigen headers. Run the following from the IMR-MLP repository root,
replacing the LAMMPS and Eigen paths:

```sh
python3 LAMMPS/USER-MLIP/install.py /path/to/lammps/src
cmake -S /path/to/lammps/cmake -B /path/to/lammps/build-imrmlp \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_MPI=ON \
  -DCMAKE_CXX_FLAGS="-I/usr/include/eigen3"
cmake --build /path/to/lammps/build-imrmlp -j8
```

This route installs the interface sources directly into the LAMMPS source tree.
The IMR-MLP executable and training library do not have to be built first.
Without MPI, select `-DBUILD_MPI=OFF` for a serial build.

### 1.2 Run a single-point calculation with your model

Two inputs are required: a trained model suitable for the target elements and
simulation conditions, and a LAMMPS data file that includes atom masses. The
example below assumes two atom types, type 1 = Al and type 2 = Ni, with matching
element metadata in the model. Save the input as `in.mtp`:

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

`model.mtp` and `system.data` are placeholder names to be replaced by actual
files; a one-element Al model needs only `pair_coeff * * model.mtp Al`. If the
model lacks element metadata, see Section 4. Before adding an integrator,
thermostat or barostat for MD, confirm that the energy, forces and pressure agree
with IMR-MLP on the same configuration.

## 2. Short-range interface selection

This section covers ordinary short-range MTP and short-range MTP-M models. Pick
an interface compatible with the model first, then compare performance; changing
interfaces does not improve the physical accuracy of the model.

| Interface | Scope and advantages | Costs and limits | When it applies |
| --- | --- | --- | --- |
| **`mlip/opt`** | CPU/MPI with analytic radial bases; ordinary MTP, short-range MTP-M li/q2/fs/n1/n2 heads and their GPM bundles; no `libimrmlp` | No LES/SOG long-range terms and no transport heat-current computes | **Default starting point for ordinary MD and relaxation** |
| `mlip/opt/omp` | Same models as `opt`; MPI + OpenMP gives each rank several CPU threads | Requires the OPENMP package and a thread count; per-thread workspaces cost memory, and more threads need not be faster | Extracting more CPU throughput on a single node |
| `mlip/gpudirect` | CUDA or HIP/DCU; independent short-range inference without `libimrmlp`; short-range MTP and MTP-M | Requires a GPU toolchain; no LES/SOG or transport computes; MTP-M runs in FP64; transfer and launch costs can offset gains on small systems | Short-range MD on a machine with a usable GPU |
| `mlip/spline` | CPU/MPI; a preconverted `RBSpline` ordinary MTP replaces analytic radial evaluation with interpolation | Adds a conversion step and introduces grid interpolation error, so energies and forces need checking after conversion; usually not faster than `opt`; not a general MTP-M interface | Radial evaluation dominates the runtime and interpolation error is acceptable |
| `mlip/native` | Uses the native IMR-MLP model implementation; suited to deployments that keep native neural networks, hybrid potentials or more complete model compositions; CPU/CUDA/HIP for supported models | Links a matching MPI `libimrmlp` and its runtime dependencies; backend coverage depends on the model; more deployment steps; slower than `opt` / `gpudirect` for the same model | The model is unsupported by `opt` / `gpudirect`, or must match a native model workflow |
| `mlip` | Original CPU interface that still runs existing inputs | The extended model coverage of `opt` or native does not apply | Maintaining existing calculations; start new work with `opt` |

The speedup from GPU or OpenMP depends on the model, system size and hardware;
switching interface or adding threads does not guarantee it. Ordinary energy,
force and pressure calculations do not need the transport interface.

For standard short-range MTP it is roughly 1.5–2× slower on CPU and more so on
GPU. Native is mainly for models that `opt` cannot read, namely long-range
MTP-M, native neural networks and hybrids.

Complete model restrictions are given in the
[CPU style guide](USER-MLIP/opt/README.md), the
[standalone guide](USER-MLIP/standalone/README.md) and the
[native guide](USER-MLIP/native/README.md).

### 2.1 OpenMP

Add `--with-omp` to the copy installer and `-DPKG_OPENMP=ON -DBUILD_OMP=ON` to the
CMake configuration, then rebuild. Both flags are needed: with only `PKG_OPENMP`
enabled and no OpenMP support at compile time, LAMMPS warns and still runs a
single thread.

Put `package omp N` before `read_data` and `pair_style` in the input, and change
the pair declaration to:

```text
pair_style mlip/opt/omp
pair_coeff * * model.mtp Al Ni
```

`N` is the number of threads **per MPI rank**, not a total core count or a rank
count; the examples below spell it as `4`, which is only a placeholder to be
replaced by the cores actually allocated. Total parallelism is ranks × `N` and
should not exceed the physical cores of a node. Each thread keeps its own
workspace, so more threads cost memory and need not be faster. `N = 0` defers to
`OMP_NUM_THREADS`.

The same settings can be given on the command line, which satisfies the ordering
requirement automatically:

```sh
lmp -pk omp 4 -in in.mtp    # equivalent to "package omp 4" in the input
lmp -sf omp -in in.mtp      # picks mlip/opt/omp for mlip/opt
```

Use one form only. With `-sf omp` the thread count comes from
`OMP_NUM_THREADS`, so an unset variable means a single thread; other styles
without an `/omp` variant fall back to their plain implementation.

`package omp` cannot move after `pair_style` either: LAMMPS rejects the command
once the simulation box is defined, and the `/omp` style reads the thread count
when it is constructed, so a missing declaration is an error rather than a
silent fallback.

If the sources were already installed by the installer, uninstall before
reinstalling, as described in Section 7.

### 2.2 spline

`mlip/spline` replaces analytic radial-basis evaluation with table interpolation.
It is faster only where radial evaluation genuinely dominates the runtime and the
extra memory traffic of the lookup costs less than the floating-point work it
saves; on most systems it is comparable to or slower than `mlip/opt` and is not a
speed-up measure. Conversion mainly serves to produce a spline-form model file for
workflows that need one.

Conversion applies to ordinary MTP only and is run as follows:

```sh
imrmlp mtp2spline model.mtp model.rbspline.mtp
```

```text
pair_style mlip/spline
pair_coeff * * model.rbspline.mtp Al Ni
```

Grid spacing and other conversion options are documented in the
[mtp2spline command reference](../docs/reference/commands.md). A converted file
does not automatically apply to other model families.

## 3. Installation routes and GPU deployment

The interface name selects the evaluator; the installation route determines how
it enters LAMMPS. The standalone route and the copy installation, for example,
both provide `mlip/opt`; standalone is not a different potential. Choose one route
per LAMMPS source tree or build to avoid installing duplicate style names.

| Installation route | Suitable use | Dependencies and tradeoffs |
| --- | --- | --- |
| CPU copy installation on this page | Existing LAMMPS sources; installs CPU short-range, LES/SOG, transport and harmonic-reference interfaces together | Modifies the LAMMPS `src`; needs Eigen headers and a rebuild; these CPU styles need no `libimrmlp` |
| [Standalone standard `lmp` build](USER-MLIP/standalone/README.md) | Deploying only short-range MTP/MTP-M on CPU/OpenMP or CUDA/HIP; exports a source package for compute nodes | No training library, MKL or system Eigen, using bundled Eigen headers; only supported short-range styles; the standard `lmp` entry point leaves upstream sources unchanged; usually faster than native on standard short-range models |
| [Native standard `lmp` build](USER-MLIP/native/lammps/README.md) / [native plugin](USER-MLIP/native/README.md) | Native model compositions, long-range models or their supported GPU paths | Links a matching MPI `libimrmlp`; the plugin additionally requires PLUGIN in the target LAMMPS and matching version, MPI and compiler ABI; usually slower than standalone on standard short-range models |

For short-range GPUs, start from the standalone standard `lmp` build. The
NVIDIA CUDA example below requires CMake 3.22 or newer; replace `80` with the
actual device architecture:

```sh
cmake -S LAMMPS/USER-MLIP/standalone/lammps -B build/lmp-standalone-cuda \
  -DLAMMPS_SOURCE_DIR=/path/to/clean/lammps \
  -DMTP_BACKEND=CUDA -DCMAKE_CUDA_ARCHITECTURES=80 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/lmp-standalone-cuda -j8
```

The architecture number is the compute capability of the device, written without
the decimal point: `80` for an A100, `90` for an H100 and `89` for an RTX 4090.
Values for other models are listed in the
[NVIDIA CUDA GPU compute capability table](https://developer.nvidia.com/cuda-gpus).

Replace the pair declaration in the single-point input above and run with the
newly built `lmp`:

```text
pair_style mlip/gpudirect device cuda precision fp64 gpu_id auto
pair_coeff * * model.mtp Al Ni
```

HIP/DCU uses `-DMTP_BACKEND=HIP`, `CMAKE_HIP_ARCHITECTURES` and `device hip`;
CPU deployment uses `-DMTP_BACKEND=CPU`, with `-DMTP_OPENMP=ON` for OpenMP. GPU
builds retain the CPU styles for comparisons within the same executable. Model
and radial-basis restrictions, memory controls, device checks and plugin
deployment are described in the [standalone guide](USER-MLIP/standalone/README.md).

## 4. Element mapping and older models

Each LAMMPS atom type maps to one model element, written in the order type 1,
2, ... and independent of the order in which atoms appear in the data file. For
type 1 = Ni and type 2 = Al:

```text
pair_style mlip/opt
pair_coeff * * model.mtp Ni Al
```

Names must match the element list declared in the model. Names and numbers cannot
be mixed in one command, and `NULL` is rejected. Three atom types require three
entries, such as `Al Ni Co`; several atom types may map to the same model element.
The same grammar applies to the other `mlip/...` pair styles on this page.

For older models without element metadata, recover the species order from the
training setup first, then use zero-based numeric indices:

```text
pair_coeff * * legacy.mtp 0 1      # type 1 -> species 0, type 2 -> species 1
```

For an ordinary MTP whose species order is verified as `Al Ni`, write the
metadata explicitly:

```sh
imrmlp adjustmtp legacy.mtp named.mtp --elements Al Ni
```

Check the predictions before switching to element names; re-saving alone cannot
recover unknown element identities. The original declaration
`pair_style mlip/opt model.mtp 0 1` followed by `pair_coeff * *` remains
supported, with the model declared once. `fix mlip/maxvol` uses its own numeric
mapping grammar, documented in its dedicated guide.

## 5. Long-range and native models

Use the following choices for models with long-range contributions or for native
model compositions. Ordinary short-range MTP needs no charge or latent-variable
compute.

| Model and requirement | Interface and advantages | Limits and selection criteria |
| --- | --- | --- |
| Linear MTP short-range head + Coulomb LES v2 / SOG v3, v4 on CPU | [`mlip/opt-les`](USER-MLIP/opt-les/README.md): dedicated CPU/MPI implementation without `libimrmlp` | Requires full 3D periodicity; no nonlinear MTP-M heads; model and computes must be redeclared after restart |
| Long-range MTP-M, native neural networks, MTP hybrids or long-range GPU execution | [`mlip/native`](USER-MLIP/native/README.md): native models and their CPU/GPU evaluators | Links MPI `libimrmlp`; the saved layout and the model/device combination must be supported; no CPU fallback |

Taking the SOG interface installed by the CPU copy installation as an example:

```text
pair_style mlip/opt-les
pair_coeff * * sog.mtp Ni Al
compute latent all mlip/sog/latent
```

Coulomb LES uses `compute q all mlip/les/charge`. SOG latent variables are
dimensionless and must not be interpreted as physical charges. These models
already include their long-range contributions; a short-range interface is not an
equivalent substitute.

Loading a GPM through the native plugin (omit `plugin load` when the plugin is
compiled into `lmp`):

```text
plugin load /path/to/mlip_native_plugin.so
# Define the pair after the units, box setup and read_data commands above:
pair_style mlip/native device cpu precision fp64 pottype auto
pair_coeff * * model.gpm Al Ni
run 0
```

GPM allows the potential type to be inferred; legacy formats need an explicit
`pottype` and may need `settings` and companion resources. Charge and
latent-variable computes, GPU placement and total-charge constraints follow the
[native guide](USER-MLIP/native/README.md). GPM is a model file format; converting
to it does not expand the model coverage of an interface. See the
[GPM model guide](../docs/reference/gpm-v1.md).

## 6. Heat current, active learning and thermodynamic integration

These features are added per task; ordinary MD does not require them.

| Task | Entry point and advantage | Scope and limits |
| --- | --- | --- |
| Thermal conductivity, full-system heat current or static interface energy current | [`mlip/transport`, `/omp`, `/spline`](USER-MLIP/transport/README.md) with the dedicated heat computes | CPU/MPI; only linear short-range MTP and its spline form; no MTP-M, LES/SOG or GPU; the total heat-current compute rejects pair hybrids and additional bonded/kspace energy terms |
| Identifying unfamiliar configurations during MD and collecting candidates | [`fix mlip/maxvol`](USER-MLIP/maxvol/README.md), added to an existing pair | Separate MPI `libimrmlp` plugin; requires a matching active set prepared in advance; scores and saves candidates without changing pair forces or generating reference labels |
| Harmonic reference and thermodynamic integration | [`harmonic/reference` workflow](../docs/advanced/harmonic-ti.md) | Included in the CPU copy installation; requires a prepared reference structure, force constants and TI settings |

The minimal declaration for a heat-current calculation, replacing the ordinary
`opt` pair declaration:

```text
pair_style mlip/transport
pair_coeff * * model.mtp Al Ni
compute J all mlip/heat/flux
```

The compute reports the full-system extensive heat current; volume normalization
follows the analysis method. Loading a GPM does not enable active learning.
Model coverage, active-set preparation and candidate output for MaxVol are
described in its dedicated guide; combining a GPU pair with the CPU MaxVol scorer
is not a combination we can recommend.

## 7. Changing models, restarting and installation maintenance

Before replacing a loaded model, delete the observable computes that depend on
it, then clear the pair and declare it again:

```text
pair_style none
pair_style mlip/opt
pair_coeff * * other.mtp Al Ni
```

The required computes are recreated afterwards. Short-range MTP-M/GPM restart
through `opt` / `opt/omp` / `gpudirect` still requires the original model path to
be readable with unchanged contents; native additionally needs the referenced
model and settings files. Prefer absolute model paths when restarting in another
directory. For `opt-les`, the model and computes must be redeclared after restart.
Exact restart restrictions for each style are given in its guide.

The CPU installer never overwrites existing files. To update an interface,
uninstall with hash verification first, then install files from the same source
version:

```sh
python3 LAMMPS/USER-MLIP/install.py --uninstall /path/to/lammps/src
```

LAMMPS must be reconfigured and rebuilt after installation or removal. Do not mix
this installer with the transport-only installer on the same source tree. MaxVol
and the native plugin are not included in the CPU copy installation.

## 8. Model metadata reference

A model records its element order in a single-line JSON tag, for example:

```text
potential_tag = {"schema":"gpml.mtp.metadata","version":1,"elements":["Al","Ni"],"type":"MTP","model":"linear"}
```

The schema name is intentionally shared and is not renamed for IMR-MLP. MTP-M
uses `type: "MTP_M"`, gives the head name in `model`, and provides the integer
`q` and `model_coeffs_count`; neural heads additionally use `nn.activation` and
`nn.hidden_layers`. LES/SOG uses `type: "MTP_LES"` with `long_range.kernel` and
`long_range.channels`; long-range MTP-M keeps `type: "MTP_M"` with a separate
`long_range` field.

Existing empty tags, text tags and `imrmlp_meta_v1:` tags remain readable.
Saving writes the canonical schema, keeps the old text in `legacy_tag` and
retains extension fields; old IMR q names and definitions plus head extensions
are kept in `imrmlp_q` and `imrmlp_external_model`. The `# mtp_*` footer fields
used by older GPU readers are preserved and checked against the JSON. Invalid
versions, duplicate keys or elements, inconsistent dimensions and conflicting
element orders all fail. A species expansion must name every output element
explicitly with `imrmlp adjustmtp ... --neworder ... --elements ...`.

## 9. License

The LAMMPS-derived adapters retain their original Sandia/LAMMPS notices and are
distributed under [GPL version 2](COPYING.GPL-2.0); they are not relicensed under
the repository's MIT license. New integration changes carry the IMR-MLP
contributors copyright, and headers identify the applicable license. The
independently maintained `runtime/mtp` core and the project helper headers copied
by the installer retain their MIT notices. A combined LAMMPS executable must be
redistributed consistently with the GPL and its other dependency licenses; see
the [third-party notices](../THIRD_PARTY_NOTICES.md).
