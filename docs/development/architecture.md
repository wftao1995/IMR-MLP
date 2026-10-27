# Architecture Overview {#p04_architecture}

[中文](architecture_zh.md)

This page is an orientation for developers and advanced C++ users.
Ordinary users can begin with the [workflow guide](../guides/workflows.md);
understanding the class hierarchy is not required to run the commands.



## Design Philosophy

IMR-MLP follows a layered architecture:

1. **Data layer** — Atoms, Neighborhoods, masks, and I/O formats define *what* we work with
2. **Compute layer** — Potentials and descriptors compute physical quantities from data
3. **Driver layer** — Minimization drivers and training algorithms orchestrate computations
4. **Application layer** — CLI commands expose functionality to end users

## Data Flow

```
Input files (POSCAR, CFG, XYZ, ...)
    │
    ▼
┌──────────┐    ┌──────────────┐
│   GPIO   │───▶│    Atoms     │
│  (I/O)   │    │ (structures) │
└──────────┘    └──────┬───────┘
                       │
              ┌────────▼────────┐
              │  Neighborhoods  │
              │ (neighbor lists)│
              └────────┬────────┘
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
   ┌──────────┐ ┌───────────┐ ┌──────────┐
   │ Potential│ │Descriptor │ │ Analysis │
   │  (EFS)   │ │ (features)│ │  (RDF,   │
   │          │ │           │ │ Hessian, │
   │          │ │           │ │  ...)    │
   └────┬─────┘ └─────┬─────┘ └────┬─────┘
        │             │            │
        ▼             ▼            ▼
   Energy, forces,  Feature    Analysis
   stress, Hessian  vectors    results
```

## Key Base Classes

| Class | Module | Role |
|-------|--------|------|
| `Atoms` | \ref core | Central data structure: cell, positions, species, properties |
| `AnyNeighborhoods` | \ref core | Type-erased neighbor list interface |
| `AnyPotential` | \ref potentials | Virtual base for all potential energy surfaces |
| `Descriptors` | \ref descriptors | Virtual base for atomic environment descriptors |
| `AnyBasis` | \ref basis | Radial basis function interface |
| `AnyCutoff` | \ref basis | Cutoff function interface |
| `GPIO` | \ref io | File I/O dispatcher |
| `SimpleFireF` | \ref mindriver | FIRE optimizer |
| `NEBDriver` | \ref mindriver | Nudged Elastic Band driver |
| `ALPotential` | \ref active_learn | Owns a potential and an AL scorer/session |
| `ALModifier` | \ref active_learn | On-the-fly active learning modifier for relaxation |

## Module Relationships

```
core ──────────────────────────────────────────────────────
  │  (Atoms, Neighborhoods — used by everything)
  │
  ├── io ── (reads/writes Atoms)
  │
  ├── basis ── (radial/cutoff functions, used by potentials & descriptors)
  │
  ├── potentials ── (virtual AnyPotential interface)
  │      ├── PotMTP (MTP implementation)
  │      ├── PotEAM (EAM implementation)
  │      └── PotLJ, ... (other implementations)
  │
  ├── descriptors ── (ACSF, BondOrder, MTM)
  │      └── Uses basis for cutoff functions
  │
  ├── analyse ── (RDF, Voronoi, Hessian, trajectory analysis/smoothing, ...)
  │
  ├── mindriver ── (L-BFGS, FIRE, CG, NEB)
  │
  ├── train ── (MTP training, nonlinear fitting, linear least-squares)
  │      └── Uses mindriver + potentials + maxvol
  │
  ├── active_learn ── (MaxVol scorer, ALPotential, ALModifier)
  │      └── Uses potentials + train + mindriver
  │
  ├── maxvol ── (Active-set selection: MaxVol, FPS, CUR)
  │
  ├── NNBase ── (NN layers, used by NN-enabled potentials)
  │
  ├── symmetry ── (Space groups, Wyckoff positions)
  │
  ├── tools ── (CLI parsing, chemistry helpers)
  │
  └── apps ── (CLI entry points, dispatch)
```

## Computational Hot Path

For energy/force/stress evaluation, the critical path is:

```
PotMTP::getEFS()
    │
    ├── Neighborhoods::construct()
    │       └── Periodic images and spatial boxes (default Neighborhoods)
    │
    ├── CalcSiteEFS (per atom; getEFSOMP selects parallel evaluation)
    │       ├── Radial basis evaluation (rb_calcder)
    │       ├── Moment tensor construction (elementary → contracted)
    │       ├── Model propagation (linear or NN)
    │       └── Backpropagation (forces & stress via Jacobian)
    │
    └── Accumulate: sum site energies → total, forces, stress
```

## Configuration Model

Most non-trivial classes inherit from `ConnectSettings`, which provides an ini-style
key-value configuration system. Settings files use namespaced prefixes (e.g.,
`relax:`, `mtp:`, `neb:`) to route parameters to the appropriate subsystem.

## Module Details

### `active_learn/` — Active Learning

- `al_scorer.h`: `ALScorer`, `MaxVolScorer` and `ALSetup` provide extrapolation
  grading against a stored active set.
- `al_potential.h`: `ALPotential` manages grading, candidate collection and
  export (`create`, `evaluate`, `check`, `finalize`). It accepts `maxvol` or
  `none`; unsupported modes raise an error.
- `mindriver/al_modifier.h`: `ALModifier` connects `ALPotential` to `relax`
  through the relaxation driver’s optional AL hook.

### `mtp_kim/` — KIM API Model Driver

Standalone KIM API compatible model driver that packages MTP as a standard KIM model
for use with LAMMPS, ASE, or any KIM-aware MD code. Contains:

- `kim_model_driver.cpp`: KIM API v2 model interface (Create, ComputeArgumentsCreate,
  Compute, Destroy, Refresh).
- `mtp_kim_core.{cpp,h}`: Self-contained MTP computation core with minimal dependencies
  — implements `Vec3`, Chebyshev radial basis, moment tensor computation, neighbor list
  building, and `.mtp` file parsing.

### `potentials/potential_openkim.{cpp,h}` — PotKIM

`PotKIM` wraps any KIM API model behind the `AnyPotential` interface, enabling KIM-compatible
potentials to be used as drop-in replacements throughout the framework. Supports auto-detection
of supported species and integrates with the standard `Neighborhoods` infrastructure for
neighbor list construction.
