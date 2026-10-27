# API Examples {#p02_api}

[中文](cpp-examples_zh.md)

@tableofcontents

This page is for C++ users embedding IMR-MLP in their own programs. For routine
tasks, start with the [CLI](../reference/commands.md) or
[Python API](../reference/python.md). Complete programs include `main()`; other
snippets assume the inputs and models described on their page.

---

## API Layers

The C++ API mirrors the [architecture](@ref p04_architecture) layers:

| Layer | Page | What You'll Learn |
|-------|------|-------------------|
| **Data** | @ref api_core | Construct `Atoms`, build `Neighborhoods`, transform coordinates, access forces/energy/stress, manage species and extra attributes |
| **I/O** | @ref api_io | Read/write structures, sequential I/O for large files, format detection and species mapping |
| **Compute** | @ref api_potential | Load potentials, compute EFS (energy/forces/stress), OpenMP parallel evaluation, Hessian computation, coefficient access, training interface — *includes a PotMTP\_M case study* |
| **Analysis** | @ref api_analysis | RDF, Voronoi, force constants, bond angles, ACSF/BondOrder descriptors, trajectory smoothing/unwrapping, space group detection |
| **Python** | @ref api_python | Python bindings: `imrmlp` package API, Atoms, Potential, I/O, relaxation, NEB, training, descriptors, active learning, ASE calculator |

---

## Quick Start: Minimal Complete Program

First run `make libimrmlp.so` and `python3 examples/quickstart/run.py`, then save
the following as `my_program.cpp` in the repository root. It uses the generated
synthetic model and validation structure to demonstrate the interface; it is not
a physical Al material model.

```cpp
#include "base/structures.h"
#include "io/gpio.h"
#include "potentials/potential_mtp.h"

#include <iostream>

int main() {
    // 1. Load a structure
    GPIO io;
    io.setmap({"Al"});  // model species 0 = Al
    Atoms atoms;
    io.read_one(atoms, "build/quickstart/valid-input.cfg");

    // 2. Load a potential
    Potentials::PotMTP pot("build/quickstart/fitted.mtp");

    // 3. Compute energy, forces, stress
    pot.getEFS(atoms);

    // 4. Access results
    std::cout << "Energy: " << atoms.energy_ << " eV\n";
    std::cout << "Max force: " << atoms.report_maxforce() << " eV/Å\n";

    // 5. Write results
    io.write_one(atoms, "build/quickstart/cpp-predicted.cfg");
    return 0;
}
```

Compile from the repository root. The compiler response file
`@build/release/.flags` reuses the library's actual include paths and macros to
keep the header and library ABI consistent. This example uses the default
release/GCC build. For custom builds, use the matching compiler,
`build/<mode>/.flags` and link options:
```bash
g++ @build/release/.flags my_program.cpp \
  -Lbuild/release -limrmlp -o build/release/my_program
LD_LIBRARY_PATH=$PWD/build/release:$LD_LIBRARY_PATH build/release/my_program
```

---

## Python Quick Start

Install the extension as described in the [Python API](../reference/python.md)
and continue running from the repository root:

```python
import numpy as np
import imrmlp

# 1. Read a structure
atoms = imrmlp.read_one("build/quickstart/valid-input.cfg")

# 2. Load a potential
pot = imrmlp.Potential("build/quickstart/fitted.mtp", "mtp")

# 3. Compute energy, forces, stress
pot.calculate(atoms)

# 4. Access results
print(f"Energy: {atoms.energy:.4f} eV")
print(f"Max force: {np.linalg.norm(atoms.forces, axis=1).max():.4f} eV/Å")

# 5. Write results
imrmlp.write("build/quickstart/python-predicted.cfg", atoms)
```

---

## Subpages

\subpage api_core
\subpage api_io
\subpage api_potential
\subpage api_analysis
\subpage api_python

---

## See Also

- @ref p04_architecture — architecture overview with data flow diagram
- @ref guide_install — build requirements and compilation
- @ref guide_config — configuration file format and API
- [Class Index](annotated.html) — complete list of documented classes
- [Module Index](topics.html) — classes organized by subsystem
