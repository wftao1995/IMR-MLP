#pragma once
#include "io/atoms_units.h"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

inline void bind_unit_conversion(pybind11::module_ &m) {
    namespace py = pybind11;
    m.def("unit_conversion_factor", [](const std::string &quantity,
                                      const std::string &from_unit,
                                      const std::string &to_unit) {
        return GPUnits::conversion_factor(GPUnits::quantity_from_string(quantity), from_unit, to_unit);
    }, py::arg("quantity"), py::arg("from_unit"), py::arg("to_unit"),
       "Explicit fixed-table numeric conversion; pressure and virial are distinct quantities.");
    m.def("canonicalize_atoms", [](const Atoms &atoms, const std::string &length,
                                   const std::string &energy, const std::string &time,
                                   const std::string &mass) {
        Atoms result = atoms;
        GPUnits::canonicalize_atoms(result, {length, energy, time, mass});
        return result;
    }, py::arg("atoms"), py::kw_only(), py::arg("length") = "A", py::arg("energy") = "eV",
       py::arg("time") = "ps", py::arg("mass") = "amu",
       "Return a canonical eV/A/ps/amu copy from explicitly declared source units. "
       "Virial is scaled as energy; arbitrary attributes and dimensionless HVP probes are preserved.");
    m.def("source_unit_system", [](const std::string &name) {
        const auto units = GPUnits::named_source_units(name);
        py::dict result;
        result["length"] = units.length; result["energy"] = units.energy;
        result["time"] = units.time; result["mass"] = units.mass;
        return result;
    }, py::arg("name"), "Return base units for canonical, metal, atomic, or real import.");
    m.def("pressure_to_virial_factor", [](const std::string &pressure_unit,
                                         double volume_angstrom3, int sign) {
        if (sign != 1 && sign != -1)
            throw std::invalid_argument("sign must be +1 (W=P*V) or -1 (W=-P*V)");
        return GPUnits::pressure_to_virial_factor(pressure_unit, volume_angstrom3,
            static_cast<GPUnits::PressureVirialSign>(sign));
    }, py::arg("pressure_unit"), py::arg("volume_angstrom3"), py::arg("sign"),
       "Multiply pressure tensor by this factor to obtain virial in eV, with explicit sign and A^3 volume.");
}
