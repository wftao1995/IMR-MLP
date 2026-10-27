#include "atoms_units.h"
#include <cmath>
#include <stdexcept>

namespace GPUnits {
SourceUnits named_source_units(const std::string &name) {
    if (name == "canonical" || name == "metal") return {};
    if (name == "atomic") return {"Bohr", "Hartree", "atomic_time", "amu"};
    if (name == "real") return {"A", "kcal/mol", "fs", "amu"};
    throw std::invalid_argument("Unknown input unit system: " + name +
                                "; expected canonical, metal, atomic, or real");
}

void canonicalize_atoms(Atoms &atoms, const SourceUnits &source) {
    const double length = conversion_factor(Quantity::Length, source.length, "A");
    const double energy = conversion_factor(Quantity::Energy, source.energy, "eV");
    const double time = conversion_factor(Quantity::Time, source.time, "ps");
    const double mass = conversion_factor(Quantity::Mass, source.mass, "amu");
    // No data is touched until every declared dimension has been validated.
    atoms.cell_ *= length;
    if (atoms.cartesian)
        for (auto &position : atoms.pos_) position *= length;
    if (atoms.have_energy) atoms.energy_ *= energy;
    if (atoms.have_fenergy) atoms.fenergy_ *= energy;
    if (atoms.have_stress) atoms.stress_ *= energy;
    for (auto &site_energy : atoms.site_energies_) site_energy *= energy;
    for (auto &force : atoms.forces_) force *= energy / length;
    for (auto &velocity : atoms.velocities_) velocity *= length / time;
    auto masses = atoms.local_attribs_d_.find("masses");
    if (masses != atoms.local_attribs_d_.end()) masses->second *= mass;
    if (atoms.have_hessian) atoms.hessian_ *= energy / (length * length);
    atoms.hvp_targets_ *= energy / (length * length);
}

double pressure_to_virial_factor(const std::string &pressure_unit,
                                double volume_angstrom3, PressureVirialSign sign) {
    if (!std::isfinite(volume_angstrom3) || volume_angstrom3 <= 0.)
        throw std::invalid_argument("Pressure/virial conversion requires positive finite volume in A^3");
    if (sign != PressureVirialSign::Same && sign != PressureVirialSign::Opposite)
        throw std::invalid_argument("Pressure/virial conversion requires an explicit valid sign");
    return static_cast<int>(sign) * volume_angstrom3 *
           conversion_factor(Quantity::Pressure, pressure_unit, "eV/A^3");
}
}
