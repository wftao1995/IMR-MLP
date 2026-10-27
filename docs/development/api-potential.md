# Potential API: Energy, Forces, Stress {#api_potential}

[中文](api-potential_zh.md)

This page is for C++ users integrating potential evaluation or custom training.
For ordinary calculations, use `calc_efs` or the [Python API](../reference/python.md).
Build the library first and use the [C++ compilation instructions](cpp-examples.md).
Examples require a valid MTP model and structures whose species indices match it.
Energy is in eV, forces in eV/Å, and native stress is an extensive virial in eV.


`Potentials::PotMTP` is defined in
`potentials/potential_mtp.h`.


@tableofcontents

## Loading and Using Potentials

### Loading an MTP Potential

```cpp
#include "potentials/potential_mtp.h"

// Load from file (constructor takes filename)
Potentials::PotMTP pot("potential.mtp");

// Or load via stream
std::ifstream ifs("potential.mtp");
pot.Load(ifs);
```

### Basic EFS Calculation

```cpp
#include "io/gpio.h"
#include "potentials/potential_mtp.h"

GPIO io;
io.setmap({"Al"});  // model species 0 = Al
Atoms atoms;
io.read_one(atoms, "POSCAR");

Potentials::PotMTP pot("potential.mtp");

// Compute energy, forces, stress (single-threaded)
pot.getEFS(atoms);

// Access results
double energy = atoms.energy_;
std::vector<Eigen::Vector3d> &forces = atoms.forces_;
Eigen::Matrix3d &stress = atoms.stress_;

std::cout << "Energy: " << energy << " eV" << std::endl;
std::cout << "Max force: " << atoms.report_maxforce() << " eV/A" << std::endl;
```

### OpenMP Parallel EFS

```cpp
// Thread-safe OpenMP evaluation
pot.omp_init();  // initialize OpenMP buffers (call once)

#pragma omp parallel for
for (int i = 0; i < structures.size(); ++i) {
    pot.getEFSOMPthread(structures[i]);
}

// Or use the built-in parallel wrapper
pot.getEFSOMP(atoms);  // parallel evaluation of a single structure
```

### EFS with Pre-built Neighborhoods

```cpp
Neighborhoods nbhs(atoms, pot.cutoff);
pot.getEFS(atoms, nbhs);  // skip neighborhood construction

// Reuse only for the same geometry and a cutoff covering the second potential
Potentials::PotMTP pot2("potential2.mtp");
pot2.getEFS(atoms, nbhs);
```

### Energy-Only Calculation

```cpp
// Faster: skip force and stress computation
pot.getE(atoms);
double e = atoms.energy_;
```

### Additive EFS

```cpp
// Evaluate multiple potentials and accumulate
Potentials::PotMTP pot_short("short_range.mtp");
Potentials::PotMTP pot_long("long_range.mtp");

// First potential
pot_short.getEFS(atoms);

// Second potential: add to existing EFS values
pot_long.add_efs = true;
pot_long.getEFS(atoms);
```

## Hessian Computation

### Full Hessian

```cpp
#include "potentials/potential.h"

Potentials::HessianMatrix hessian;
pot.getEFSHessian(atoms, hessian);

// Access: hessian(i_atom, a, j_atom, b)
double d2E_dxdy = hessian(0, 0, 1, 1);  // d2E / d(x0)d(y1)

// Atomic force constant matrix (3x3 block for atom pair)
Eigen::Matrix3d &fc = hessian(0, 1);
```

### Cell Hessian

```cpp
Eigen::MatrixXd cellhess(9, 9);   // d2E / d(strain)2
Eigen::MatrixXd lrhess(9, 3 * natoms);  // d2E / d(strain)d(pos)

pot.getEFSHessianCell(atoms, hessian, cellhess, lrhess);
```

### Symmetry-Aware Hessian

```cpp
// Exploit crystal symmetry for speed
double symprec = 1e-4;
pot.getEFSHessianSymm(atoms, hessian, symprec);
```

## Coefficient Access

### Getting Coefficients

```cpp
// Total coefficient count
int ncoeffs = pot.get_coeff_size();

// Get all coefficients
Eigen::VectorXd coeffs(ncoeffs);
pot.get_coeffs(coeffs);

// Get as raw array
std::vector<double> coeff_arr(ncoeffs);
pot.get_coeffs(coeff_arr.data());

// Access coefficient segments
double *radial = pot.get_inter_pointer_radial_coeffs();
double *linear = pot.get_inter_pointer_linear_coeffs();
double *all = pot.get_inter_pointer_coeffs();

// Individual linear coefficients
double linear_coeff = pot.get_linear_coeff(0);
```

### Setting Coefficients

```cpp
// Modify and update
Eigen::VectorXd new_coeffs = coeffs;
new_coeffs[0] *= 1.1;  // scale first coefficient
pot.set_coeffs(new_coeffs);

// Set only linear coefficients (fix radial)
// Note: set_linear_coeff takes double*, use .data() for Eigen vectors
pot.set_linear_coeff(new_linear_only.data());
```

## Creating Potentials Programmatically

### Initialize from a Complete Template

An MTP needs a radial basis, moment basis and coefficient layout. Setting only
`cutoff` and `species_count` does not construct them. Load a complete template
before randomizing coefficients; `init_basic()` only prepares working buffers
for an existing basis and is already called during loading.

```cpp
// Run the CLI quickstart first, then run this from the repository root.
// The template provides the radial basis, moment basis and coefficient layout.
Potentials::PotMTP pot("build/quickstart/template.mtp");
pot.random_init();
```

This initializes a model for fitting, not a physical material potential.
Use a template with the intended species order, cutoff and basis for real data.

### Using Settings

```cpp
#include "utils/gputils.h"

GPSettings settings;
settings["potfile"] = "template.mtp";

Potentials::PotMTP pot(settings);
```

## Working with AnyPotential

### Polymorphic Interface

```cpp
#include "potentials/potential.h"

// Use concrete type directly, or wrap in unique_ptr for polymorphism
auto pot = std::make_unique<Potentials::PotMTP>("potential.mtp");
Potentials::AnyPotential *anypot = pot.get();

Atoms atoms;
// ... populate ...

// Use the virtual interface
pot->getEFS(atoms);
pot->getEFSOMP(atoms);  // auto-builds neighborhood
```

### Save and Load

```cpp
// Save to file
pot.Save("potential_saved.mtp");

// Save to stream
std::ofstream ofs("potential_stream.mtp");
pot.Save(ofs);

// Load from stream
std::ifstream ifs("potential.mtp");
pot.Load(ifs);
```

## Training Interface

Standard MTP is linear in moment coefficients at a fixed radial basis; optimizing
radial coefficients is still nonlinear training. The snippets below are for
custom training loops. For routine fitting, prefer `trainmtp`, `trainnlpot` or
the Python trainers.

### Gradient Accumulation

```cpp
// For nonlinear potential training
int ncoeffs = pot.get_coeff_size();                   // number of coefficients
double dLdE = 2.0 * (atoms.energy_ - ref_energy);     // MSE gradient
Eigen::MatrixXd dLdF(force_rows, 3);                  // ColMajor by default
// ... fill dLdF with force differences ...
Eigen::Matrix3d dLdS = 2.0 * (atoms.stress_ - ref_stress);

std::vector<double> gradient(ncoeffs);
Neighborhoods nbhs(atoms, pot.cutoff);
pot.AccumulateEFSCombinationGard(atoms, nbhs,
    dLdE, dLdF, dLdS,
    gradient.data());   // accumulates the gradient in place
```

### Regularization

```cpp
double loss = 0.0;
std::vector<double> loss_grad(ncoeffs);   // ncoeffs from earlier get_coeff_size()
pot.AddPenaltyGard(1.0, loss, loss_grad.data());
// loss: regularization contribution to total loss
// loss_grad: gradient of regularization term
```

## Case Study: Multi-Order MTP (PotMTP\_M)

`PotMTP_M` is a multi-order generalization of the standard MTP. Instead of one
scalar site energy, it produces @f$n@f$ descriptor quantities
@f$Q_k = c_{s,k} + \sum_\alpha c_{k,\alpha} B_\alpha@f$, then passes them
through an external model @f$F(Q_1, \dots, Q_n)@f$ — enabling polynomial,
neural-network, or custom regressors on top of the MTP descriptor.

### Step 1: Create the Potential

```cpp
#include "potentials/potential_mtp_m.h"
#include "base/structures.h"
#include "io/gpio.h"

using namespace Potentials;

// Configure the external model type via settings
GPSettings settings;
settings["potfile"]    = "basis_template.mtp";  // pre-built MTP basis
settings["extmodel"]   = "mtp_m_q2";            // quadratic model
// Other model options: "mtp_m_n1" (1-hidden-layer NN), "mtp_m_li" (linear),
//                      "mtp_m_fs" (ArcSinh finetuning)

// Create the potential — loads basis + initializes the model
PotMTP_M<m_model_mtpq2> pot(settings);
```

### Step 2: Random Initialization (for Training from Scratch)

```cpp
// Seed coefficients randomly within [-0.1, 0.1]
pot.random_init();

// Or control the range
pot.random_lower = -0.05;
pot.random_upper =  0.05;
pot.random_init();
```

### Step 3: Evaluate Energy and Forces

```cpp
Atoms atoms;
GPIO io;
io.setmap({"Al"});  // adapt to the model species order
io.read_one(atoms, "structure.vasp");

// Build neighborhoods (required for EFS)
Neighborhoods nbhs(atoms, pot.get_cutoff());

// Compute energy, forces, stress
pot.getEFS(atoms, nbhs);

std::cout << "Energy: "   << atoms.energy_  << " eV"    << std::endl;
std::cout << "Max force: " << atoms.report_maxforce() << " eV/Å" << std::endl;
```

### Step 4: Training (Gradient Accumulation)

The multi-order MTP training accumulates gradients per-atom through the
moment tensor tree. Use a gradient-based optimizer (L-BFGS, SGD, etc.)
in your training loop:

```cpp
int ncoeffs = pot.get_NL_coeffs_count();
std::vector<double> gradient(ncoeffs, 0.0);

// For each training structure:
for (auto &train_atoms : training_set) {
    Neighborhoods nbhs(train_atoms, pot.get_cutoff());
    pot.getEFS(train_atoms, nbhs);  // forward pass stores intermediate data

    double dLdE = 2.0 * (train_atoms.energy_ - ref_energy);  // MSE derivative

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        pot.AccumulateFPGard(nbhs[ii], gradient.data(), dLdE, nullptr);
    }

}

// Add regularization once per total objective
double loss = 0.0;
pot.AddPenaltyGard(1.0, loss, gradient.data());

// Take an optimizer step using gradient (L-BFGS, SGD, etc.)
// ... optimizer.update(coeffs_ptr, gradient.data()) ...

// Apply updated coefficients back to the potential
pot.nl_set_coeffs(coeffs_ptr);
```

### Step 5: Save and Reuse

```cpp
// Save the trained potential
pot.Save("trained_mtp_m.mtp");

// Reload later
PotMTP_M<m_model_mtpq2> loaded_pot("trained_mtp_m.mtp");
// Note: pure file-load constructor works when model doesn't need init()
```

### Key Differences from PotMTP

| Aspect | PotMTP | PotMTP\_M |
|--------|---------|------------|
| **Site energy** | Linear: `sum c_a * B_a` | Modeled: `F(Q_1, ..., Q_n)` |
| **Coefficients** | species + moment coeffs | + model coeffs (NN weights, polynomial terms) |
| **Model type** | No extra external head; linear in moment coefficients at fixed radial basis | Template parameter: polynomial, NN, custom |
| **Use case** | Standard MTP | Nonlinear regression on MTP descriptors |
| **Training** | `AccumulateEFSCombinationGard` | `AccumulateFPGard` (forward+backward through model) |

## See Also

- @ref potentials — Potential module
- @ref theory_mtp — MTP theory
- @ref theory_training — Training methodology
