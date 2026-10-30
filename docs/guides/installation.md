# Installation and build {#guide_install}

[中文](installation_zh.md)

Build a working CPU command-line program and add Python, MPI or GPU support when
needed. Using IMR-MLP does not require changing C++ code. The provided Make build
targets Linux x86_64; Windows and macOS have no native installation route here and
should use a Linux machine or environment.

## Installation scope

| Purpose | Install | Not needed |
| --- | --- | --- |
| Energy/forces, relaxation, NEB or ordinary MTP training | CPU CLI using the steps below | Python extension, MPI, GPU and documentation tools |
| An administrator already provides an executable | Run `imrmlp --version`, then the [quickstart](../../examples/quickstart/README.md) | Rebuilding it yourself |
| Python / ASE calculations | Prepare native dependencies, then install the Python package | LAMMPS |
| LAMMPS MD with an existing short-range model | CPU / standalone route in the [LAMMPS guide](../../LAMMPS/README.md) | Usually the full training library |
| Multiple processes or GPUs | Check the CPU example first, then select a mode below | Configuring every backend at first installation |

Replace `/path/to/...` with real paths. System dependencies are normally
installed once; each new terminal still needs its runtime environment enabled.

## Dependencies

The core CPU program needs a C++23 compiler and standard library, GNU Make
4.3+, Eigen, ALGLIB, FFTW, OpenMP and default-enabled zlib. oneMKL is optional:
it hands Eigen's dense linear algebra to Intel's BLAS/LAPACK and is enabled
automatically once its environment is sourced; without it the same program is
built from Eigen's own kernels, which is slower for large matrix products.
The basic route below disables optional OpenKIM to avoid another dependency
at first installation.

### Ubuntu / Debian development packages

With administrator access, start with common development packages:

```sh
sudo apt update
sudo apt install build-essential git python3 python3-venv \
  libeigen3-dev libalglib-dev libfftw3-dev zlib1g-dev
```

These are distribution package-name examples, not validation of that system's
default compiler. Development headers and libraries are required; runtime
packages alone are insufficient. Check ALGLIB in the
[Ubuntu package directory](https://packages.ubuntu.com/noble/libalglib-dev).
On other distributions use the corresponding package manager; on clusters,
ask whether compiler and dependency modules are already provided rather than
changing system directories yourself.

### Optional: oneMKL for faster dense linear algebra

Skip this section to build against Eigen's own kernels. Install oneMKL when
dense matrix products dominate the workload: the Make build enables the backend
as soon as `MKLROOT` provides the headers, `USE_MKL=0` disables it again, and
`USE_MKL=1` forces it when the environment is provided another way.

Follow [Intel's installation instructions](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl-download.html)
to install oneMKL **development files**. With Intel's APT repository already
configured, use `sudo apt install intel-oneapi-mkl-devel`; the runtime-only
package is insufficient. Using oneMKL here does not require switching to
Intel's C++ compiler.

For a common component-directory installation, run in the current terminal:

```sh
source /opt/intel/oneapi/setvars.sh
```

Replace the directory for a private installation. Unified-layout installations
use their version's `oneapi-vars.sh`; see [Intel's environment guide](https://www.intel.com/content/www/us/en/docs/oneapi-hpc-toolkit/get-started-guide-linux/2025-0/before-you-begin.html).
Check that `MKLROOT` points to a directory containing `include/mkl.h`:

```sh
echo "$MKLROOT"
test -f "$MKLROOT/include/mkl.h" && echo "MKL headers found"
g++ --version
make --version
```

If setting `MKLROOT=/actual/mkl/directory` manually, also configure the oneMKL
runtime-library search path; `MKLROOT` alone does not enable the runtime.
Sources use `-std=c++23`. The current CPU validation environment uses GCC
16.2.1; this does not establish a minimum compatible version for every older
compiler. For standard-library or language-option errors on an older system,
use a compiler/library combination supporting the required features rather
than removing the language-standard option.

## Sources and build

If sources are already available, enter the repository root. Otherwise:

```sh
git clone https://github.com/wftao1995/IMR-MLP.git
cd IMR-MLP
```

Check that the current directory contains `Makefile`, then build:

```sh
make -j2 imrmlp USE_KIM=0
```

`-j2` runs two compiler tasks concurrently; increase it if memory permits.
It does not set calculation threads. Output is in `build/release/`, not the
source root. Do not use `sudo make`. The build picks the oneMKL backend only
when the current terminal has the oneMKL environment; otherwise it compiles the
same sources against Eigen's own kernels, so a missing oneMKL never blocks the
first build. Changing that choice recompiles every file of that build mode.

Add this build's program and library directories for the current terminal,
then check startup:

```sh
export PATH="$PWD/build/release:$PATH"
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
imrmlp --version
imrmlp --help
```

`PATH` locates the executable; `LD_LIBRARY_PATH` gives priority to its matching
shared library. `$PWD` must be the repository root when these commands run;
you may then change to a calculation directory. In a new terminal, re-enable
oneMKL if the build used it and use the repository's absolute path for these
variables.

## Self-contained example

Still from the repository root:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/quickstart/run.py \
  --work build/quickstart
```

Only Python's standard library and the compiled CLI are needed. The example
generates synthetic data and a model, fits and validates it. Each stage should show
`OK`; outputs are in `build/quickstart/`, matching later pages. Use a new `--work`
directory on subsequent runs, since nonempty directories are not overwritten.
Interpretation is in the [quickstart guide](../../examples/quickstart/README.md).

Single-point output in that directory can be inspected with the commands in
[single-point calculation](single-point.md), without rerunning the training script.
Without your own inputs, the [walkthrough with all inputs supplied](../../examples/basic-calculations/README.md)
provides complete files for relaxation, NEB and properties. Test and documentation
tools are not prerequisites for this example.

## Installing into a user directory

Running from the build directory is sufficient. To use an installed copy from
any calculation directory, retain the same build options:

```sh
make -j2 install USE_KIM=0 PREFIX="$HOME/.local"
export PATH="$HOME/.local/bin:$PATH"
export LD_LIBRARY_PATH="$HOME/.local/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
imrmlp --version
```

The personal prefix needs no administrator access. External runtimes must stay
available: ALGLIB, FFTW and, only if that build enabled the backend, oneMKL.
Make installs CLI tools and libraries, not the Python package. Moving to another
machine also requires compatible system dependencies; choose `CPU_ARCH_FLAGS`
again before distributing across CPUs.

## Optional components: Python, MPI and GPU

### Python / ASE

Python requires version 3.10+, CMake and Python development headers
(`cmake python3-dev` packages on Ubuntu / Debian). Create and activate a
virtual environment from the repository root:

```sh
python3 -m venv .venv
source .venv/bin/activate
USE_KIM=0 python -m pip install '.[ase]'
python -c 'import imrmlp; print(imrmlp.__version__)'
```

Native C++ dependencies must already be present: pip does not supply ALGLIB,
FFTW or an optional oneMKL backend. Installation uses CMake to invoke the Make
release build in the source tree. The command retains `USE_KIM=0` from the basic
route. Replace `'.[ase]'` with `.`
if ASE is unnecessary. See the [Python guide](../../README-python.md) for imports
outside the checkout, species mapping and calculator examples.
For source-directory development, another route is
`python3 -m pip install numpy pybind11` followed by
`make -j2 python PYTHON=python3 USE_KIM=0`; new users normally choose one route.

### MPI and GPU modes

Each mode has its own build directory. Match executable and library modes.
MPI needs matching `mpicxx`, launcher and runtime; GPUs need their CUDA/HIP
toolchain.

| `BUILD_MODE` | Main executable in that directory | Use |
| --- | --- | --- |
| `release` (default) | `imrmlp` | Ordinary CPU calculations |
| `mpi` | `imrmlp-mpi` | MPI processes; NEB distributes images among ranks |
| `cuda` | `imrmlp-cu` | Supported NVIDIA CUDA models and commands |
| `hip` | `imrmlp-hip` | Supported AMD HIP / DCU models and commands |
| `mpi-cuda`, `mpi-hip` | `imrmlp-mpi` | MPI with the corresponding GPU backend |

For example, build MPI and run a prepared NEB configuration:

```sh
make -j2 BUILD_MODE=mpi USE_KIM=0 imrmlp
imrmlp_root="$PWD"
cd /path/to/neb-run
LD_LIBRARY_PATH="$imrmlp_root/build/mpi${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  mpirun -np 4 "$imrmlp_root/build/mpi/imrmlp-mpi" neb neb.ini --num_threads 1
```

Relative inputs still resolve from the launch directory. In practice, launch
from the task directory using absolute program and library paths.
See [NEB](neb.md) for ranks and threads, and [GPU support](../development/cuda.md)
for architectures and supported paths. A GPU build does not automatically put
all commands on the GPU: `trainmtp` still runs on CPU; supported GPU fitting
uses `trainnlpot`.

## Common build problems

| Symptom | First check |
| --- | --- |
| Missing `g++` / `make`, or unknown `-std=c++23` | Installed compiler and GNU Make on `PATH` |
| Missing `Eigen/...`, `libalglib/...` or `fftw3.h` | Development packages and matching Eigen/ALGLIB paths |
| `mkl.h` missing although oneMKL is installed | `MKLROOT` and the oneMKL environment, or build with `USE_MKL=0` |
| Linker reports `cannot find -l...` | Development library and link directory; initialized oneMKL environment when that backend is used |
| Missing runtime library / undefined symbol | Matching program and library; current build first in the loader path |
| `cc1plus` killed | Reduce to `-j1` or `-j2` and check memory |
| pip build fails | Check native dependencies, then the first actual compile/link error in its log |

See [troubleshooting](troubleshooting.md) for more explanations.

## Build options and maintenance targets

Ordinary installation does not require editing Makefiles. Override non-system
dependency paths on the command line:

```sh
make -j2 imrmlp USE_KIM=0 EIGEN_DIR=/path/to/eigen3 \
  ALGLIB_INCLUDE_ROOT=/path/to/include \
  ALGLIB_LIBS='-L/path/to/lib -lalglib'
```

`ALGLIB_INCLUDE_ROOT` contains the `libalglib/` directory. fmt, spglib,
TinyXML-2, Voro++ and LBFGS++ are bundled; ALGLIB and FFTW are external, and
oneMKL is an optional external backend.

| Option / target | When needed |
| --- | --- |
| `USE_KIM=1` | OpenKIM use; needs KIM API, enabled by Make by default |
| `USE_GZIP=0` | Disable default gzip support; otherwise zlib is required |
| `USE_ZSTD=1` | Compressed GPM blocks; also needs libzstd |
| `USE_MKL=1`, `USE_MKL=0` | Force or disable the oneMKL backend; by default it follows `MKLROOT` |
| `MKL_LIBDIR` | Override the oneMKL library directory, otherwise located through `MKLROOT`; used only when the backend is enabled |
| `CPU_ARCH_FLAGS=` | Clear default `-march=native`, or select actual target-CPU flags |
| `BUILD_MODE=debug`, `asan`, `relwithdebinfo`, `relwithdebinfo-asan`, `mpi-debug` | Debugging and diagnostics; normal users choose release/mpi |
| `make all` | Library and all CLI/auxiliary programs |
| `make main` | Main program without `libimrmlp`; external dynamic libraries may remain |
| `make clean-<mode>` | Clean a build mode; avoid cleaning directories containing calculation results |

Installed `ccache` is used automatically. `make test` needs Catch2 / Catch2Main;
put the matching build directory first in `LD_LIBRARY_PATH` when running existing
test executables. Full testing, other maintenance targets and licensing are in
[CONTRIBUTING.md](../../CONTRIBUTING.md) and [third-party notices](../../THIRD_PARTY_NOTICES.md).
Reading the manual needs no Doxygen; building it is covered in the
[documentation build guide](../README.md).
