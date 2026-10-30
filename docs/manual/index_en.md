# IMR-MLP documentation

```{raw} html
<div class="imr-home-hero">
  <div class="imr-home-logo">
    <img src="_static/IMR-MLP-logo.png" alt="IMR-MLP — Institute of Metal Research, Machine Learning Potential" width="200" height="200">
  </div>
  <div class="imr-home-intro">
    <p class="imr-home-tagline">Moment tensor machine-learning potentials</p>
    <p>Use machine-learning interatomic potentials for atomistic calculations: train and validate models, relax structures, find transition paths, calculate crystal properties, and run models through Python, ASE or LAMMPS.</p>
    <div class="imr-home-actions">
      <a class="imr-home-primary" href="guides/index.html">User guides</a>
      <a href="guides/quickstart.html">Run an example</a>
      <a href="reference/index.html">Reference</a>
    </div>
  </div>
  <div class="imr-home-developers">
    <p>Developed by the Division of Materials Artificial Intelligence Research,<br>Institute of Metal Research, Chinese Academy of Sciences.</p>
  </div>
</div>
```

## Entry points

| What you need | Page |
| --- | --- |
| Build the CPU program and verify the environment | [Installation and build](guides/installation.html) |
| Input files, species mapping, settings syntax and units | [Input files and settings conventions](guides/conventions.html) |
| Energy, forces and stress, with error evaluation | [Single-point calculation](guides/single-point.html) |
| Verify the workflow without your own model and data | [Self-contained example](guides/quickstart.html) |

The complete task classification is in the [user guides](guides/index.html), with the
runnable examples listed at the end.

## Complete a calculation

| Goal | Start here |
| --- | --- |
| Fixed-cell / variable-cell optimization | [Structure relaxation](guides/relaxation.html) |
| Diffusion, reaction or transformation barriers | [NEB minimum-energy paths](guides/neb.html) |
| Phonons, elasticity or bulk modulus | [Crystal properties](guides/properties.html) |
| Prepare reference data, fit and validate a model | [Training workflow](guides/workflows.html) |
| Identify unfamiliar structures and add training data | [Active learning and selection](guides/active-learning.html) |
| LAMMPS MD / Python and ASE | [LAMMPS](guides/lammps.html) / [Python](guides/python-installation.html) |

For problems, start with [troubleshooting](guides/troubleshooting.html); for command names,
consult the [cheat sheet](reference/cheatsheet.html).

## Browse by purpose

| Section | Purpose |
|---|---|
| [1. User guides](guides/index.html) | How to complete a task |
| [2. Reference](reference/index.html) | Definitions of commands, options and formats |
| [3. Advanced usage](advanced/index.html) | HVP, SOG, fastfit and specialized examples |
| [4. Theory and algorithms](theory/index.html) | Why the models and methods work |
| [5. Development](development/index.html) | For contributors changing code or extending interfaces |

C++ symbols and source indexes are available in the [API site](../../build/html/annotated.html).

```{toctree}
:hidden:
:maxdepth: 3

guides/index
reference/index
advanced/index
theory/index
development/index
```
