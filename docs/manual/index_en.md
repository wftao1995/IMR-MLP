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
      <a class="imr-home-primary" href="guides/getting-started.html">Get started</a>
      <a href="guides/quickstart.html">Run an example</a>
      <a href="reference/index.html">Reference</a>
    </div>
  </div>
  <div class="imr-home-developers">
    <p>Developed by the Division of Materials Artificial Intelligence Research,<br>Institute of Metal Research, Chinese Academy of Sciences.</p>
  </div>
</div>
```

## First steps

1. [Choose a task and learn which inputs you need](guides/getting-started.html)
2. [Install the CPU program and verify the environment](guides/installation.html)
3. [Without a model, run the self-contained example](guides/quickstart.html)
4. Choose a walkthrough: [direct LJ calculations](guides/basic-calculations.html), or [MTP from basis generation and fitting to calculations](guides/mtp-workflow.html)

With a suitable existing model, start with a [single-point calculation](guides/getting-started.html)
without training a model first.

## Complete a calculation

| Goal | Start here |
| --- | --- |
| Structure formats, species mapping, energy/forces/stress | [First calculation and inputs](guides/getting-started.html) |
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
:maxdepth: 2

guides/index
reference/index
advanced/index
theory/index
development/index
```
