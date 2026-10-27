#pragma once
namespace LAMMPS_NS { class LAMMPS; }
// Call after constructing LAMMPS and before processing the input script.
// This explicitly registers the statically linked adapters without modifying
// LAMMPS generated style headers or relying on dynamic PLUGIN support.
void register_mtp_standalone(LAMMPS_NS::LAMMPS *);
