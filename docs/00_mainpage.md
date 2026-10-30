# Introduction {#mainpage}

[中文](00_mainpage_zh.md)

\image html IMR-MLP-logo.png "" width=400px
\image latex IMR-MLP-logo.png "" width=0.6\\textwidth

**Moment Tensor Machine Learning Potential Framework**

A C++23 framework for building, training, and deploying machine-learning interatomic potentials.
Implements the Moment Tensor Potential (MTP) formalism with support for nonlinear
neural-network models, active learning, and a broad set of structure analysis tools.

Developed by the Division of Materials Artificial Intelligence Research, Institute of Metal Research, Chinese Academy of Sciences.

## Documentation map

| Section | Purpose |
|---|---|
| [User guides](guides/index.md) | Task-based installation, evaluation, relaxation, NEB, properties, training and selection. |
| [Reference](reference/index.md) | Commands, configuration, Python and model formats. |
| [Advanced usage](advanced/index.md) | Basis selection, HVP, variable projection, long-range models and saddle searches. |
| [Theory and algorithms](theory/index.md) | MTP and MTP_M models, energy derivatives, fitting and active learning. |
| [Development](development/index.md) | C++ architecture, APIs and integration modules. |

---

## Entry points

The complete classification is in the [user guides](guides/index.md), with the four
runnable examples listed at the end.

| Task | Page |
| --- | --- |
| Build the program; add Python, MPI or GPU support | [Installation and build](guides/installation.md) |
| Input files, species mapping, settings syntax and units | [Input files and settings conventions](guides/conventions.md) |
| Evaluate E/F/stress and quantify errors | [Single-point calculation](guides/single-point.md) |
| Optimize atoms or cell | [Structure relaxation](guides/relaxation.md) |
| Migration or reaction barriers | [NEB paths](guides/neb.md) |
| Phonons, elasticity and equation of state | [Crystal properties](guides/properties.md) |
| Train and validate a model | [Training workflow](guides/workflows.md) |
| Selection, reference labeling and retraining | [Active learning](guides/active-learning.md) |
| Run a model in LAMMPS / Python | [LAMMPS](../LAMMPS/README.md) / [Python](../README-python.md) |

Use the [command reference](reference/commands.md) for options and
[troubleshooting](guides/troubleshooting.md) for problems.

---

## Module Reference

| Module | Description |
|--------|-------------|
| @ref core "Core Data Types" | Atoms, Neighborhoods, and fundamental data structures |
| @ref potentials "Potentials" | Interatomic potentials: MTP, EAM, LJ, KIM, and more |
| @ref active_learn "Active Learning" | MaxVol extrapolation grading, ALPotential, ALModifier and online relaxation |
| @ref descriptors "Descriptors" | ML descriptors: ACSF/Behler, Bond Order, MTM |
| @ref io "I/O" | Readers/writers for VASP, LAMMPS, XYZ, CFG, and more |
| @ref analyse "Analysis" | RDF, bond angles, Voronoi, MSD, SRO, force constants |
| @ref cheatsheet "Command Cheat Sheet" | Quick-reference one-page command summary |
| @ref mindriver "Minimization" | L-BFGS, FIRE, conjugate gradient, steepest descent, NEB |
| @ref basis "Basis Functions" | Radial basis functions (Chebyshev, polynomial, GTO) and cutoff functions |
| @ref maxvol "MAXVOL" | Active-set selection for training configuration curation |
| @ref train "Training" | MTP training, nonlinear potential fitting, SGD scheduling |
| @ref symmetry "Symmetry" | Space group detection and Wyckoff position analysis via spglib |
| @ref nnbase "NN Primitives" | Neural network layers (1- and 2-hidden-layer) |
| @ref tools "Utilities" | CLI parsing, chemistry helpers, physical constants |
| @ref apps "Commands" | CLI subcommand implementations for the imrmlp binary |
| @ref trajectory "Trajectory" | Smoothing, unwrapping, and FFT filtering of MD trajectories |

---

## API Reference

- [Class Index](annotated.html) — complete list of documented classes
- [File Index](files.html) — source files with documentation
- [Todo List](todo.html) — known missing features and planned improvements
- [Bug List](bug.html) — known issues

---

## References

- **Wang, J. et al.** "Efficient moment tensor machine-learning interatomic potential
  for accurate description of defects in Ni-Al Alloys."
  *Phys. Rev. Materials* 9, 053805 (2025).
  DOI: [10.1103/PhysRevMaterials.9.053805](https://doi.org/10.1103/PhysRevMaterials.9.053805)
- Shapeev, A. "Moment Tensor Potentials: A Class of Systematically Improvable Interatomic Potentials."
  *Multiscale Model. Simul.* 14, 1153–1173 (2016).
- Bartok, A. et al. "On representing chemical environments." *Phys. Rev. B* 87, 184115 (2013).
- Behler, J. "Atom-centered symmetry functions for constructing high-dimensional neural network potentials."
  *J. Chem. Phys.* 134, 074106 (2011).
