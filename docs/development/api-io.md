# I/O API: Reading and Writing Structures {#api_io}

[中文](api-io_zh.md)

This page is for C++ integrations. For file conversion without programming,
use `imrmlp convstr`; for scripts, see the [Python I/O API](../reference/python.md).
Build and link the library as described in [API examples](cpp-examples.md).
The following snippets require the named input files.



@tableofcontents

## Basic Read and Write

### Single Structure

```cpp
#include "io/gpio.h"

GPIO io;
Atoms atoms;

// Read from file (auto-detect format by extension)
io.read_one(atoms, "POSCAR");
io.read_one(atoms, "structure.cfg");
io.read_one(atoms, "structure.lmpdat");

// Specify format explicitly
io.read_one(atoms, "input.file", "poscar");
io.read_one(atoms, "input.file", "mlpcfg");
io.read_one(atoms, "data.lammps", "lmpdat");
io.read_one(atoms, "trajectory.cp2k.xyz", "cp2kxyz");

// Write to file (format auto-detected from extension)
io.write_one(atoms, "output.vasp");
io.write_one(atoms, "structure.cfg");
io.write_one(atoms, "data.lammps", "lmpdat");  // explicit format for this name
```

### Multiple Structures

```cpp
std::vector<Atoms> structures;

// Read all structures from file
io.read(structures, "trajectory.cfg");
io.read(structures, "XDATCAR");
io.read(structures, "md.lammpstrj");

// Write all structures
io.write(structures, "output.cfg");
```

### Using Streams

```cpp
std::ifstream ifs("input.cfg");
Atoms atom;
io.read_one(atom, ifs, "mlpcfg");

std::ofstream ofs("output.vasp");
io.write_one(atom, ofs, "poscar");
```

### Compressed Files

File-based reads and writes transparently support gzip (`.gz`): reads
auto-detect gzip from the file's magic bytes, and writes compress when the
output filename ends in `.gz`. Zstandard (`.zst`) is unsupported; detected
Zstandard input and `.zst` output names are rejected.

```cpp
io.read_one(atoms, "POSCAR.gz");          // reads a compressed POSCAR
io.write(structures, "traj.xyz.gz");      // writes a compressed gzip trajectory

```

## Sequential I/O (Large Files)

For trajectories too large to fit in memory:

```cpp
GPIO io;
std::ifstream ifs("md.lammpstrj");
if (!ifs) throw std::runtime_error("Cannot open md.lammpstrj");
io.seq_read_init(ifs, "lmptrj");

Atoms atom;
while (io.seq_read(ifs, atom)) {
    // Process one frame at a time
    process(atom);
}
```

Sequential writing:

```cpp
GPIO io;
std::ofstream ofs("output.cfg");
io.seq_write_init(ofs, "mlpcfg", "output.cfg");

for (auto &atom : structures) {
    io.seq_write(ofs, atom);
}
```

## Supported Formats

### Reading

| Format | Extension / Name | Mode |
|--------|------------------|------|
| VASP POSCAR | `.vasp`, `POSCAR` | Single |
| VASP XDATCAR | `XDATCAR` | Sequential |
| VASP OUTCAR | `OUTCAR` | Single / Sequential |
| VASP vasprun.xml | `vasprun.xml` | File-based `read` (requires USEXML) |
| VASP ML_ABN | `ML_ABN` | Sequential |
| LAMMPS dump | `.lammpstrj` | Sequential |
| LAMMPS data | `.lmpdat`, `.lmp` | Single |
| LAMMPS YAML | `.yaml` | Sequential |
| MLIP CFG | `.cfg` | Sequential |
| MLIP binary CFG | `.cfgbin` | Sequential |
| Extended XYZ | `.extxyz` | Sequential |
| Simple XYZ (EXTXYZ reader) | `.xyz` | Sequential |
| CIF | `.cif` | Single |
| XSF / AXSF | `.xsf`, `.axsf` | Single / Sequential |
| CP2K XYZ | `.cp2kxyz` | Sequential |
| GROMACS | `.gro` | Single / Sequential |
| n2p2 data | `.n2p2` | Sequential |
| MSI (Materials Studio) | `.msi` | Single / Multi-model |
| XSD (Materials Studio) | `.xsd` | Single |
| CAR (Biosym) | `.car` | Single |

### Writing

| Format | Extension / Name | Mode |
|--------|------------------|------|
| VASP POSCAR | `.vasp`, `POSCAR` | Single |
| VASP XDATCAR / v6 | `XDATCAR`, `XDATCAR6` | Sequential |
| LAMMPS data | `.lmpdat`, `.lmp` | Single |
| LAMMPS dump | `.lammpstrj` | Sequential |
| MLIP CFG | `.cfg` | Sequential |
| MLIP binary CFG | `.cfgbin` | Sequential |
| Extended XYZ | `.extxyz` | Sequential |
| CIF | `.cif` | Single |
| XSF / AXSF | `.xsf`, `.axsf` | Single / Animation |
| VTK | `.vtk` | Single |
| CP2K XYZ | `.cp2kxyz` | Sequential |
| n2p2 data | `.n2p2` | Sequential |
| DLPOLY4 | `.dlpoly` | Single |

## Format Notes

### Filename detection and explicit formats

The tables list recognized filenames, not every conventional suffix. Detection
uses the filename unless content detection is explicitly requested. A generic
`data.lammps` or `input.data` needs an explicit `"lmpdat"` or `"n2p2"` argument,
according to its actual format. `trajectory.cp2k.xyz` is guessed as `"extxyz"`;
pass `"cp2kxyz"` for CP2K data. The same names select output codecs. For example:

```cpp
io.read(structures, "input.data", "n2p2");
io.write(structures, "output.data", "n2p2");
```

Read/write direction and filename detection are separate capabilities. See the
[Python format table](../reference/python.md#supported-formats) for explicit
format names, and [shared workflows](../advanced/shared-workflows.md) for the
narrower independent streaming-session format set.

### VASP OUTCAR

- The header's `ions per type` line is read in full: every count on the line is
  used to build the atom counts and species map, so multi-element calculations
  (e.g. `2 3` for Fe2O3) yield the correct total atom number.
- POTCAR species names may carry a valence suffix (`Fe_sv`, `K_pv`, `O_d`,
  ...); the suffix is stripped before mapping to atomic numbers.
- The VASP version is read from the first line (` vasp.5...` / ` vasp.6...`).
  Lines that are too short to contain a version digit are rejected instead of
  being read out of bounds.
- An ionic step that reaches convergence exactly on its NELM-th electronic
  iteration is accepted (VASP still writes the `aborting loop because EDIFF is
  reached` line right after the last iteration). Steps that hit NELM without
  converging are still dropped.

### CIF

- The `loop_` block writes each `_atom_site_*` tag on its own line, which is
  required by the ASE CIF parser and other strict readers.

### Materials Studio formats (`msi`, `xsd`, `car`)

- **MSI** (Cerius2 DataModel): each `(N Model ...)` block is one structure;
  `read()` returns all models. Cell from `A3`/`B3`/`C3` records, elements from
  the `ACL` record, coordinates are Cartesian (Angstrom, as in OpenBabel's
  reader). Bonds and `Charge` records are read when present.
- **XSD**: all layout variants are supported (flat, nested
  `SymmetrySystem`/`IdentityMapping`, and non-periodic `Molecule`). The cell
  comes from `SpaceGroup` `AVector`/`BVector`/`CVector`; `Atom3d` `XYZ` is
  fractional when a cell is present, `CartesianXYZ` is always Cartesian.
  `Bond` elements and `Charge` attributes are read when present.
- **CAR** (Biosym archive): cell from the `PBC a b c alpha beta gamma`
  line, 9-column atom records (`label x y z XXXX mol potential element
  charge`); the charge column is optional. Non-periodic files (`PBC=OFF`) are
  handled.

### LAMMPS trajectory (`lammpstrj`)

- Vector attributes are written with 1-based LAMMPS column indices
  (`desc[1] desc[2]`, not `desc_[0]`), matching what the reader accepts, so
  per-atom vector properties round-trip losslessly.

## Format Detection

```cpp
// From filename extension
GPIO::supported_format fmt = GPIO::guess_format("structure.cfg");
// returns GPIO::MLPCFG

// From format name string
GPIO::supported_format fmt2 = GPIO::format2enum("poscar");
// returns GPIO::POSCAR

// Format enumeration
switch (fmt) {
    case GPIO::POSCAR:    /* ... */ break;
    case GPIO::MLPCFG:    /* ... */ break;
    case GPIO::LMPTRJ:    /* ... */ break;
    case GPIO::EXTXYZ:    /* ... */ break;
    // ...
}
```

## Chemistry Map

The GPIO maintains an integer<->string mapping for chemical species:

```cpp
GPIO io;

// Default map: atomic number -> element symbol
io.setmap();

// For a model with species 0 = Al, 1 = Ni, set the map before reading
io.setmap({"Al", "Ni"});

// Or load from a file (one "id element" pair per line)
io.readmap("index.map");

// Query the map
std::map<int, std::string> id2name = io.getmap();
std::map<std::string, int> name2id = io.getrevmap();

std::string element = io.get_type(0);  // "H", "O", etc.

// Get list of species
std::vector<int> species;
io.report_chem(species);
```

## Configuration Options

```cpp
GPIO io;

// Output control
io.setoutrate(1, 0, INT_MAX);  // every frame, from start to end
io.outrate = 2;                 // write every 2nd frame
io.out_startfrom = 10;          // start from frame 10
io.out_endto = 100;             // end at frame 100

// Format-specific options
io.write_stress = true;          // EXTXYZ: output -virial/volume, eV/Å³
io.write_virtual = true;         // EXTXYZ: output extensive virial (legacy name)
io.absolute_index_write = false; // output the structure's raw species indices
io.write_symbol = true;          // write element symbols (not numbers)
io.header_line = "My calculation";  // custom header

// Reading options
io.read_startfrom = 0;
io.read_endto = INT_MAX;
io.read_irate = 1;               // read every Nth frame
io.drop = false;                 // skip non-converged structures?
```

## Example: Format Conversion

```cpp
#include "io/gpio.h"

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: converter INPUT OUTPUT\n";
        return 1;
    }
    GPIO io;
    std::vector<Atoms> structures;

    // Read all structures
    io.read(structures, argv[1]);

    // Write in a different format
    io.write(structures, argv[2]);

    return 0;
}
```

Usage:
```bash
./converter input.cfg output.lmpdat
./converter POSCAR output.extxyz
```

## See Also

- @ref io — I/O module
- @ref api_core — Atoms API
- @ref guide_commands — CLI commands (including `convstr`)
