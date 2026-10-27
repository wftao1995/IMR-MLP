# Core API: Atoms & Neighborhoods {#api_core}

[中文](api-core_zh.md)

This page is for C++ users extending or embedding IMR-MLP. For routine
structure handling, start with the [Python API](../reference/python.md).
Build `libimrmlp.so` first; compilation and linking are shown in
[API examples](cpp-examples.md). Snippets share the `atoms` object unless stated otherwise.



@tableofcontents

## Creating and Manipulating Atoms

### Basic Construction

```cpp
#include "base/structures.h"

// Create an empty Atoms object
Atoms atoms;

// Set a cubic cell (each row is a lattice vector)
atoms.cell_ << 5.0, 0.0, 0.0,
               0.0, 5.0, 0.0,
               0.0, 0.0, 5.0;

// Add atoms with species indices
atoms.pos_.push_back({0.0, 0.0, 0.0});  // atom 0 at origin
atoms.pos_.push_back({2.5, 2.5, 2.5});  // atom 1 at body center
atoms.chemmap_ = {0, 1};                 // species 0 and 1

// Coordinates are Cartesian; periodicity defaults to all three directions
atoms.cartesian = true;
atoms.pbc_ = {true, true, true};
```

### Reading from File

```cpp
#include "io/gpio.h"

GPIO io;
Atoms atoms;

// Read a single structure
io.read_one(atoms, "POSCAR");

// Read multiple structures
std::vector<Atoms> structures;
io.read(structures, "trajectory.cfg");
```

### Cell Operations

```cpp
// Create a 2x2x2 supercell
Eigen::Matrix3i super_mat;
super_mat << 2, 0, 0,
             0, 2, 0,
             0, 0, 2;
atoms.mksuperL(super_mat);

// Get cell properties
double volume = atoms.get_volume();
Eigen::Matrix3d metric = atoms.get_metric();
Eigen::Matrix3d recip = atoms.get_recip();  // with 2*pi

// Cell parameters (a, b, c, alpha, beta, gamma)
std::vector<double> params = atoms.get_cell_parameters();
double a = params[0], b = params[1], c = params[2];
double alpha = params[3], beta = params[4], gamma = params[5];

// Standardize cell orientation
atoms.to_lower_standard();  // lower triangular, right-handed, a || x
```

### Coordinate Transformations

```cpp
// Fractional <-> Cartesian conversion
atoms.to_fractional();    // pos_ now in fractional coordinates
atoms.to_cartesian();     // pos_ back to Cartesian

// Direct conversion (doesn't modify state)
Eigen::RowVector3d frac = atoms.car2dir(cart_pos);
Eigen::RowVector3d cart = atoms.dir2car(frac_pos);

// Rotate atoms around an axis
atoms.rotate(Eigen::Vector3d(0, 0, 1), M_PI/4);        // 45° around z
atoms.rotate<true, false>(1, 0, 0, M_PI/2);             // rotate atoms only
                                                         // around [100] lattice direction

// Translate
atoms.translate(Eigen::Vector3d(1.0, 0.0, 0.0));

// Wrap atoms into cell
atoms.wrap();
```

### Geometry Queries

```cpp
int natoms = atoms.size();                    // number of atoms
double d = atoms.get_distance(0, 1);          // minimum-image distance
Eigen::RowVector3d rab = atoms.get_rab(0, 1); // minimum-image vector
Eigen::RowVector3d cm = atoms.get_cm();       // center of mass
bool orth = atoms.isorth();                    // orthogonal cell check
bool rh = atoms.is_right_hand();              // right-handed check
int handed = atoms.get_handedness();           // +1 = right, -1 = left
```

### Property Access

```cpp
// Energy, forces, stress
double energy = atoms.energy_;
std::vector<Eigen::Vector3d> &forces = atoms.forces_;
Eigen::Matrix3d &stress = atoms.stress_;

// Check what's available
bool has_forces = atoms.has_forces();
bool has_stress = atoms.has_stresses();
bool has_velocities = atoms.has_velo();

// Clear and reset
atoms.resetEFS();    // set energy, forces, stress to zero
atoms.clearEFS();    // remove energy, forces, stress data
```

### Species Management

```cpp
// Get unique species
std::set<int> types = atoms.get_types_set();

// Count atoms per type
std::map<int, int> counts = atoms.types_count_map();
for (auto &[type, count] : counts) {
    std::cout << "Type " << type << ": " << count << " atoms" << std::endl;
}
```

### Extra Attributes

```cpp
// Per-atom scalar attribute
atoms.alloc_ld("site_energy");
atoms.local_attribs_d_["site_energy"][0] = -3.14;

// Global scalar attribute
atoms.global_attribs_d_["temperature"] = 300.0;

// Per-atom vector attribute (3 components per atom)
atoms.alloc_lvd("dipole", 3);
atoms.local_attribs_v_["dipole"](0, 0) = 0.1;  // component 0, atom 0
atoms.local_attribs_v_["dipole"](1, 0) = 0.2;  // component 1, atom 0
atoms.local_attribs_v_["dipole"](2, 0) = 0.3;  // component 2, atom 0

// Check presence
if (atoms.has_l_attrib_d()) { /* ... */ }
```

### Fitting Weights

```cpp
// Per-structure weight
atoms.fitness = 2.0;  // higher weight for this configuration

// Per-atom weights
atoms.atomic_fitness = {1.0, 0.5, 1.0, 1.0};  // atom 1 has lower weight
```

## Building Neighborhoods

### Basic Usage

```cpp
#include "base/neighborhood.h"

Atoms atoms;
// ... populate atoms ...

double cutoff = 5.0;
Neighborhoods nbhs(atoms, cutoff);  // build neighbor lists

// Alternatively, two-step initialization
Neighborhoods nbhs2;
nbhs2.init(atoms, cutoff);
```

### Accessing Neighbor Info

```cpp
// Iterate over central atoms
for (int i = 0; i < atoms.size(); ++i) {
    // Access the neighborhood of atom i
    auto &nbh = nbhs[i];              // access by operator[]
    int centertype = nbh.centertype;   // species of central atom
    int n_neighbors = nbh.nbhcount_;   // number of neighbors

    for (int j = 0; j < n_neighbors; ++j) {
        double rij = nbh.distances_[j];    // distance
        int jtype = nbh.chemmap_[j];      // neighbor species
        Eigen::Vector3d rvec = nbh.pos_[j]; // relative position vector
    }
}
```

### Selective Dynamics

Initialize every component explicitly: `1` allows movement and `0` fixes it.
Enable selective dynamics in the relaxation driver as well. This does not
constrain the cell.

```cpp
atoms.sel_dynpos_.assign(atoms.size(), Eigen::RowVector3d::Ones());
if (atoms.size() > 0) atoms.sel_dynpos_[0](0) = 0;  // fix the first atom's x component
```

## Merging and Extracting Structures

```cpp
// Merge two structures (concatenate atoms)
Atoms combined = merge_atoms(struct1, struct2);

// Extract substructure by mask
GPMask mask;  // (defined elsewhere)
Atoms subset = atoms.extract_substructures_from_mask(mask);
```

## Serialization

```cpp
// Binary serialization
std::ofstream bofs("atoms.bin", std::ios::binary);
atoms.write_to_bin(bofs);

```

## See Also

- @ref core — Core Data Types module
- @ref api_io — File I/O API
- @ref api_potential — Potential evaluation API
