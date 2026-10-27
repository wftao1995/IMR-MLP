#pragma once

#include "base/structures.h"
#include "utils/units.h"

namespace GPUnits {
// Source units are an explicit import declaration, never inferred from labels.
// Force/velocity/curvature use the derived E/L, L/T, E/L^2 units.
struct SourceUnits {
    std::string length = "A";
    std::string energy = "eV";
    std::string time = "ps";
    std::string mass = "amu";
};

// Supported presets: canonical (alias metal), atomic, real.
// atomic = Bohr, Hartree, atomic_time, amu; real = A, kcal/mol, fs, amu.
SourceUnits named_source_units(const std::string &name);

// In-place explicit normalization, validating all source units before mutation.
// Arbitrary attributes and dimensionless HVP probes are preserved. The only
// recognized mass field is local_attribs_d_["masses"]. Unset labels stay unset.
// stress_ is the virial in source energy units and is scaled by E alone.
void canonicalize_atoms(Atoms &atoms, const SourceUnits &source);

// Explicit pressure-tensor to virial factor: caller declares the relation
// W = +P*V or W = -P*V; volume must be positive and already in A^3.
// Apply the returned factor to the source pressure tensor to get W in eV.
enum class PressureVirialSign { Same = 1, Opposite = -1 };
double pressure_to_virial_factor(const std::string &pressure_unit,
                                double volume_angstrom3, PressureVirialSign sign);
}
