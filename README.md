# IMR-MLP

[中文](README_zh.md) · [Documentation](docs/README.md) · [LAMMPS](LAMMPS/README.md) · [Python](README-python.md)

IMR-MLP is a C++23 framework for training and deploying machine-learning
interatomic potentials. It provides MTP fitting, energy/force/stress evaluation,
active learning, structure relaxation and analysis, Phonopy integration, Python
bindings, and MTP interfaces for LAMMPS.

Developed by the Division of Materials Artificial Intelligence Research, Institute of Metal Research, Chinese Academy of Sciences.

The CLI is `imrmlp`, the shared library is `libimrmlp.so`, and the Python package
is `imrmlp`.

## Build and try it

The default Linux build needs a C++23 compiler, GNU Make 4.3+, Eigen, Intel
oneMKL, ALGLIB and FFTW. OpenKIM and gzip support are
enabled by default. See the [installation guide](docs/guides/installation.md)
for dependency paths, optional features and MPI/CUDA/HIP modes.

From the repository root, with dependencies installed, run the following. The
first line uses the common oneAPI component layout; replace the environment
script for other installation paths/layouts as explained in the installation guide:

```sh
source /opt/intel/oneapi/setvars.sh
make -j2 imrmlp USE_KIM=0
export PATH="$PWD/build/release:$PATH"
export LD_LIBRARY_PATH="$PWD/build/release${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
imrmlp --version
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 examples/quickstart/run.py
```

The example generates a small synthetic dataset, fits an MTP, and evaluates
held-out configurations. It needs only the built CLI and Python's standard
library. Results and logs go to `build/quickstart/`; use a new `--work` directory
to repeat it. This is a workflow demonstration, not a material potential.
See the [walkthrough](examples/quickstart/README.md).

## Choose a workflow

| Task | Start here |
| --- | --- |
| First use: choose a task, prepare files and evaluate a structure | [Getting started](docs/guides/getting-started.md) |
| Fixed-cell or variable-cell structure optimization | [Relaxation guide](docs/guides/relaxation.md) |
| Migration or reaction barriers between endpoints | [NEB guide](docs/guides/neb.md) |
| Phonons, elasticity and equation of state | [Crystal properties](docs/guides/properties.md) |
| Install CPU, MPI or GPU builds | [Build guide](docs/guides/installation.md) |
| Learn configuration syntax and species mapping | [Configuration](docs/reference/configuration.md) |
| Prepare data, train and validate an MTP | [Workflows](docs/guides/workflows.md) |
| Find command arguments | [Command reference](docs/reference/commands.md), `imrmlp COMMAND --help` |
| Deploy MTP in LAMMPS | [CPU/OpenMP, spline, GPU and transport](LAMMPS/README.md) |
| Use Python or ASE | [Python guide](README-python.md) |
| Use LES/SOG models | [LES](docs/reference/les-v2.md), [SOG](docs/reference/sog.md) |
| Select data or sample during LAMMPS MD | [Active-learning guide](docs/guides/active-learning.md), [LAMMPS MaxVol](LAMMPS/USER-MLIP/maxvol/README.md) |
| Find stationary points and transition paths | [NR, Dimer, MODECAR and NEB](docs/advanced/saddle-workflows.md) |
| Answer "which command, unit or type?" questions | [FAQ](docs/guides/faq.md) |
| Diagnose build or runtime problems | [Troubleshooting](docs/guides/troubleshooting.md) |
| Run tests or contribute changes | [Contributing](CONTRIBUTING.md) |

Settings use whitespace, **not an equals sign**:

```text
potfile model.mtp
fit:bfgs_maxiteration 1000
```

For nonlinear MTP optimization, explicitly select an optimizer:

```sh
build/release/imrmlp trainmtp train.ini --do-lbfgs
```

## License

Original project code is [MIT](LICENSE), copyright 2026 IMR-MLP contributors.
LAMMPS-derived adapters retain their [GPLv2 license](LAMMPS/COPYING.GPL-2.0).
Other dependencies and bundled material retain their own licenses; see
[third-party notices](THIRD_PARTY_NOTICES.md). In particular, linking GPL
editions of ALGLIB or FFTW affects redistribution of compiled artifacts.

The repository includes synthetic fixtures and runnable examples. External
research datasets are not bundled.
