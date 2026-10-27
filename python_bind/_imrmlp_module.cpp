// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * _imrmlp — pybind11 C++ bindings for IMR-MLP (imrmlp Python package)
 *
 * Phase 1: Atoms, PotentialWrapper, GPSettings, GPIO (basic I/O)
 *
 * Build:
 *   g++ -O2 -std=c++23 -fPIC -shared -fopenmp python_bind/_imrmlp_module.cpp \
 *       -I. -I/usr/include/eigen3 -Ideplibs $(python3 -m pybind11 --includes) \
 *       -o _imrmlp$(python3-config --extension-suffix) \
 *       -Lbuild/release -limrmlp <system-libs>
 */

#include "base/structures.h"
#include "base/neighborhood.h"
#include "analyse/compute_hessian.h"
#include "potentials/potential.h"
#include "potentials/potential_wrapper.h"
#include "potentials/model_bundle.h"
#include "potentials/potential_descriptor_linear.h"
#include "potentials/potential_shell.h"
#include "io/gpio.h"
#include "io/gpio_session.h"
#include "io/format_detection.h"
#include "python_bind/units_bindings.h"
#include "python_bind/xsf_fields_bindings.h"
#include "tools/chem.h"
#include "utils/compressed_stream.h"
#include "utils/gputils.h"
#include "utils/cmdopts.h"
#include "mindriver/relax_new.h"
#include "mindriver/relax_constraints_io.h"
#include "mindriver/neb.h"
#include "mindriver/neb_perturbation.h"
#include "mindriver/dimer_new.h"
#include "train/train_linear_potential.h"
#include "train/train_nonlinear_potential.h"
#include "train/fixed_core.h"
#include "descriptors/descriptor_wrapper.h"
#include "maxvol/active_selector.h"
#include "maxvol/fps_selector.h"
#include "maxvol/cur_selection.h"
#include "maxvol/maxvol.h"
#include "maxvol/b_matrix_builder.h"
#include "active_learn/al_potential.h"
#include "active_learn/al_scorer.h"

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <omp.h>
#include <memory>
#include <sstream>
#include <optional>
#include <stdexcept>

namespace py = pybind11;

// ============================================================================
//  Helper: vector<RowVector3d> <-> MatrixXd conversion
// ============================================================================
static Eigen::MatrixXd pos_to_matrix(const std::vector<Eigen::RowVector3d> &pos) {
    if (pos.empty()) return Eigen::MatrixXd(0, 3);
    Eigen::MatrixXd mat(pos.size(), 3);
    for (size_t i = 0; i < pos.size(); ++i)
        mat.row(i) = pos[i];
    return mat;
}

static void matrix_to_pos(const Eigen::MatrixXd &mat, std::vector<Eigen::RowVector3d> &pos) {
    if (mat.cols() != 3) throw py::value_error("Positions must have shape (N, 3)");
    pos.resize(mat.rows());
    for (int i = 0; i < mat.rows(); ++i)
        pos[i] = mat.row(i);
}

static Eigen::MatrixXd forces_to_matrix(const std::vector<Eigen::Vector3d> &forces) {
    if (forces.empty()) return Eigen::MatrixXd(0, 3);
    Eigen::MatrixXd mat(forces.size(), 3);
    for (size_t i = 0; i < forces.size(); ++i)
        mat.row(i) = forces[i];
    return mat;
}

static void matrix_to_forces(const Eigen::MatrixXd &mat, std::vector<Eigen::Vector3d> &forces,
                             int natoms) {
    if (mat.cols() != 3 || (mat.rows() != 0 && mat.rows() != natoms))
        throw py::value_error("Forces and velocities must have shape (N, 3), or (0, 3) to clear");
    forces.resize(mat.rows());
    for (int i = 0; i < mat.rows(); ++i)
        forces[i] = mat.row(i);
}

static py::array_t<double> vector_to_numpy_copy(const Eigen::VectorXd &v) {
    py::array_t<double> out(v.size());
    auto r = out.mutable_unchecked<1>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size()); ++i)
        r(i) = v(static_cast<int>(i));
    return out;
}

static py::array_t<double> matrix_to_numpy_copy(
    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &m) {
    py::array_t<double> out({
        static_cast<py::ssize_t>(m.rows()),
        static_cast<py::ssize_t>(m.cols())
    });
    auto r = out.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(m.rows()); ++i)
        for (py::ssize_t j = 0; j < static_cast<py::ssize_t>(m.cols()); ++j)
            r(i, j) = m(static_cast<int>(i), static_cast<int>(j));
    return out;
}

// ============================================================================
//  Atoms binding
// ============================================================================
void bind_atoms(py::module_ &m) {
    py::class_<Atoms>(m, "_Atoms")
        .def(py::init<>())

        // ── cell ── (Eigen::Matrix3d -> auto NumPy)
        .def_property("cell",
            [](Atoms &a) -> Eigen::Matrix3d { return a.cell_; },
            [](Atoms &a, const Eigen::Matrix3d &c) { a.cell_ = c; })

        // ── positions ── (vector<RowVector3d> -> Nx3 Matrix)
        .def_property("positions",
            [](Atoms &a) { return pos_to_matrix(a.pos_); },
            [](Atoms &a, const Eigen::MatrixXd &p) { matrix_to_pos(p, a.pos_); })

        .def_property("selective_dynamics_mask",
            [](const Atoms &a) -> py::object {
                if (a.sel_dynpos_.empty()) return py::none();
                if (a.sel_dynpos_.size()!=static_cast<size_t>(a.size()))
                    throw py::value_error("Selective dynamics mask must have shape (N, 3)");
                py::array_t<bool> result({static_cast<py::ssize_t>(a.size()),py::ssize_t(3)});
                auto output=result.mutable_unchecked<2>();
                for (int i=0;i<a.size();++i) for (int j=0;j<3;++j) {
                    const double value=a.sel_dynpos_[i][j];
                    if (value!=0 && value!=1)
                        throw py::value_error("Selective dynamics mask must contain only binary 0 or 1 values");
                    output(i,j)=value==1;
                }
                return result;
            },
            [](Atoms &a, const std::optional<Eigen::MatrixXd> &mask) {
                if (!mask) { a.sel_dynpos_.clear(); return; }
                if (mask->rows()!=a.size() || mask->cols()!=3)
                    throw py::value_error("Selective dynamics mask must have shape (N, 3)");
                if (!mask->allFinite() || ((mask->array()!=0) && (mask->array()!=1)).any())
                    throw py::value_error("Selective dynamics mask must contain only finite binary 0 or 1 values");
                std::vector<Eigen::RowVector3d> next(mask->rows());
                for (int i=0;i<mask->rows();++i) next[i]=mask->row(i);
                a.sel_dynpos_.swap(next);
            })

        // ── chemmap (species indices) ──
        .def_property("chemmap",
            [](const Atoms &a) { return a.chemmap_; },
            [](Atoms &a, const std::vector<int> &c) { a.chemmap_ = c; })

        // ── periodic boundary flags ──
        .def_property("pbc",
            [](const Atoms &a) { return a.pbc_; },
            [](Atoms &a, const std::array<bool,3> &pbc) { a.pbc_ = pbc; })

        // ── energy ──
        .def_property("energy",
            [](const Atoms &a) { return a.energy_; },
            [](Atoms &a, double e) { a.energy_ = e; a.have_energy = true; })

        .def_property("free_energy",
            [](const Atoms &a) -> std::optional<double> {
                if (!a.have_fenergy) return std::nullopt;
                return a.fenergy_;
            },
            [](Atoms &a, std::optional<double> energy) {
                if (!energy) {
                    a.have_fenergy = false;
                } else {
                    if (!std::isfinite(*energy))
                        throw py::value_error("Free energy must be finite");
                    a.fenergy_ = *energy;
                    a.have_fenergy = true;
                }
            })

        .def_property_readonly("hessian", [](const Atoms &a) {
            return a.have_hessian ? a.hessian_ : Eigen::MatrixXd(0,0);
        })
        .def("set_hessian", [](Atoms &a, const Eigen::MatrixXd &target) {
            const int n = 3 * a.size();
            if (n == 0 || target.rows() != n || target.cols() != n || !target.allFinite())
                throw py::value_error("Full Hessian labels require a finite Cartesian 3N by 3N matrix");
            a.hessian_ = target;
            a.have_hessian = true;
        }, py::arg("target"))
        .def("clear_hessian", [](Atoms &a) {
            a.have_hessian = false;
            a.hessian_.resize(0,0);
        })

        // HVP matrices have shape (3N, probes), with atom-major xyz rows.
        .def_property_readonly("hvp_directions", [](const Atoms &a) { return a.hvp_directions_; })
        .def_property_readonly("hvp_targets", [](const Atoms &a) { return a.hvp_targets_; })
        .def("set_hvp", [](Atoms &a, const Eigen::MatrixXd &directions,
                           const Eigen::MatrixXd &targets) {
            Atoms checked;
            checked.pos_ = a.pos_;
            checked.hvp_directions_ = directions;
            checked.hvp_targets_ = targets;
            checked.validate_hvp();
            a.hvp_directions_ = std::move(checked.hvp_directions_);
            a.hvp_targets_ = std::move(checked.hvp_targets_);
        }, py::arg("directions"), py::arg("targets"))
        .def("clear_hvp", &Atoms::clear_hvp)

        // ── forces ──
        .def_property("forces",
            [](const Atoms &a) { return forces_to_matrix(a.forces_); },
            [](Atoms &a, const Eigen::MatrixXd &f) { matrix_to_forces(f, a.forces_, a.size()); })

        // ── stress ──
        .def_property("stress",
            [](const Atoms &a) { return a.stress_; },
            [](Atoms &a, const Eigen::Matrix3d &s) { a.stress_ = s; a.have_stress = true; })

        .def_property("masses",
            [](const Atoms &a) -> py::object {
                const auto found = a.local_attribs_d_.find("masses");
                if (found == a.local_attribs_d_.end()) return py::none();
                return vector_to_numpy_copy(found->second);
            },
            [](Atoms &a, py::object value) {
                if (value.is_none()) { a.local_attribs_d_.erase("masses"); return; }
                const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
                if (!array || array.ndim() != 1 || array.shape(0) != a.size())
                    throw py::value_error("Masses must have shape (N,)");
                Eigen::VectorXd masses(a.size());
                const auto data = array.unchecked<1>();
                for (int i = 0; i < a.size(); ++i) {
                    if (!std::isfinite(data(i))) throw py::value_error("Masses must be finite");
                    masses(i) = data(i);
                }
                a.local_attribs_d_["masses"] = std::move(masses);
            })

        .def_property("tags",
            [](const Atoms &a) -> py::object {
                const auto found = a.local_attribs_i_.find("tags");
                if (found == a.local_attribs_i_.end()) return py::none();
                py::array_t<int> values(found->second.size());
                auto output = values.mutable_unchecked<1>();
                for (Eigen::Index i = 0; i < found->second.size(); ++i) output(i) = found->second(i);
                return values;
            },
            [](Atoms &a, py::object value) {
                if (value.is_none()) { a.local_attribs_i_.erase("tags"); return; }
                const auto array = py::array::ensure(value);
                if (!array || array.ndim() != 1 || array.shape(0) != a.size())
                    throw py::value_error("Tags must have shape (N,)");
                const py::int_ lower(std::numeric_limits<int>::min()), upper(std::numeric_limits<int>::max());
                Eigen::VectorXi tags(a.size());
                for (int i = 0; i < a.size(); ++i) {
                    const py::object item = array.attr("__getitem__")(i);
                    if (!PyIndex_Check(item.ptr())) throw py::value_error("Tags must be integers");
                    const auto number = py::reinterpret_steal<py::object>(PyNumber_Index(item.ptr()));
                    if (!number) throw py::error_already_set();
                    if (PyObject_RichCompareBool(number.ptr(), lower.ptr(), Py_LT) == 1 ||
                        PyObject_RichCompareBool(number.ptr(), upper.ptr(), Py_GT) == 1)
                        throw py::value_error("Tags exceed the native integer range");
                    tags(i) = py::cast<int>(number);
                }
                a.local_attribs_i_["tags"] = std::move(tags);
            })

        .def_property("initial_charges",
            [](const Atoms &a) -> py::object {
                const auto found = a.local_attribs_d_.find("initial_charges");
                if (found == a.local_attribs_d_.end()) return py::none();
                return vector_to_numpy_copy(found->second);
            },
            [](Atoms &a, py::object value) {
                if (value.is_none()) { a.local_attribs_d_.erase("initial_charges"); return; }
                const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
                if (!array || array.ndim() != 1 || array.shape(0) != a.size())
                    throw py::value_error("Initial charges must have shape (N,)");
                Eigen::VectorXd charges(a.size());
                for (int i = 0; i < a.size(); ++i) {
                    if (!std::isfinite(array.data()[i])) throw py::value_error("Initial charges must be finite");
                    charges(i) = array.data()[i];
                }
                a.local_attribs_d_["initial_charges"] = std::move(charges);
            })

        .def_property("initial_magmoms",
            [](const Atoms &a) -> py::object {
                const auto scalar = a.local_attribs_d_.find("initial_magmoms");
                const auto vector = a.local_attribs_v_.find("initial_magmoms");
                if (scalar != a.local_attribs_d_.end() && vector != a.local_attribs_v_.end())
                    throw py::value_error("Simultaneous scalar and vector initial magnetic moments are not supported");
                if (scalar != a.local_attribs_d_.end()) {
                    if (scalar->second.size() != a.size())
                        throw py::value_error("Initial magnetic moments must have shape (N,) or (N, 3)");
                    return vector_to_numpy_copy(scalar->second);
                }
                if (vector == a.local_attribs_v_.end()) return py::none();
                if (vector->second.rows() != 3 || vector->second.cols() != a.size())
                    throw py::value_error("Initial magnetic moments must have shape (N,) or (N, 3)");
                py::array_t<double> values({py::ssize_t(a.size()), py::ssize_t(3)});
                auto output = values.mutable_unchecked<2>();
                for (int i = 0; i < a.size(); ++i)
                    for (int axis = 0; axis < 3; ++axis) output(i, axis) = vector->second(axis, i);
                return values;
            },
            [](Atoms &a, py::object value) {
                if (value.is_none()) {
                    a.local_attribs_d_.erase("initial_magmoms");
                    a.local_attribs_v_.erase("initial_magmoms");
                    return;
                }
                const auto array = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
                if (!array || (array.ndim() != 1 && array.ndim() != 2) || array.shape(0) != a.size() ||
                    (array.ndim() == 2 && array.shape(1) != 3))
                    throw py::value_error("Initial magnetic moments must have shape (N,) or (N, 3)");
                for (py::ssize_t i = 0; i < array.size(); ++i)
                    if (!std::isfinite(array.data()[i])) throw py::value_error("Initial magnetic moments must be finite");
                if (array.ndim() == 1) {
                    Eigen::VectorXd moments(a.size());
                    for (int i = 0; i < a.size(); ++i) moments(i) = array.data()[i];
                    a.local_attribs_d_["initial_magmoms"] = std::move(moments);
                    a.local_attribs_v_.erase("initial_magmoms");
                } else {
                    Eigen::MatrixXd moments(3, a.size());
                    for (int i = 0; i < a.size(); ++i)
                        for (int axis = 0; axis < 3; ++axis) moments(axis, i) = array.data()[3 * i + axis];
                    a.local_attribs_v_["initial_magmoms"] = std::move(moments);
                    a.local_attribs_d_.erase("initial_magmoms");
                }
            })

        // ── velocities ──
        .def_property("velocities",
            [](const Atoms &a) { return forces_to_matrix(a.velocities_); },
            [](Atoms &a, const Eigen::MatrixXd &v) { matrix_to_forces(v, a.velocities_, a.size()); })

        // ── site_energies ──
        .def_property("site_energies",
            [](const Atoms &a) { return a.site_energies_; },
            [](Atoms &a, const std::vector<double> &se) { a.site_energies_ = se; })

        // ── charges ──
        .def_property("charges",
            [](const Atoms &a) { return a.charges_; },
            [](Atoms &a, const std::vector<double> &c) { a.charges_ = c; })

        .def_property("magmoms",
            [](const Atoms &a) -> py::object {
                if (!a.magmom_scalar_.empty() && !a.magmoms_.empty())
                    throw py::value_error("Simultaneous scalar and vector magnetic moments are not supported by magmoms");
                if (!a.magmom_scalar_.empty()) {
                    if (a.magmom_scalar_.size() != a.pos_.size())
                        throw py::value_error("Magnetic moments must have shape (N,) or (N, 3)");
                    py::array_t<double> values(a.size());
                    std::copy(a.magmom_scalar_.begin(), a.magmom_scalar_.end(), values.mutable_data());
                    return values;
                }
                if (!a.magmoms_.empty()) {
                    if (a.magmoms_.size() != a.pos_.size())
                        throw py::value_error("Magnetic moments must have shape (N,) or (N, 3)");
                    py::array_t<double> values({py::ssize_t(a.size()), py::ssize_t(3)});
                    auto output = values.mutable_unchecked<2>();
                    for (int i = 0; i < a.size(); ++i)
                        for (int axis = 0; axis < 3; ++axis)
                            output(i, axis) = a.magmoms_[i](axis);
                    return values;
                }
                return py::none();
            },
            [](Atoms &a, const py::object &value) {
                if (value.is_none()) {
                    a.magmom_scalar_.clear();
                    a.magmoms_.clear();
                    return;
                }
                const auto values = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
                if (!values || (values.ndim() != 1 && values.ndim() != 2) ||
                    values.shape(0) != a.size() || (values.ndim() == 2 && values.shape(1) != 3))
                    throw py::value_error("Magnetic moments must have shape (N,) or (N, 3)");
                for (py::ssize_t i = 0; i < values.size(); ++i)
                    if (!std::isfinite(values.data()[i]))
                        throw py::value_error("Magnetic moments must be finite");
                if (values.ndim() == 1) {
                    std::vector<double> scalar(values.data(), values.data() + values.size());
                    a.magmom_scalar_ = std::move(scalar);
                    a.magmoms_.clear();
                    a.collinear_ = true;
                } else {
                    std::vector<Eigen::Vector3d> vectors(a.size());
                    for (int i = 0; i < a.size(); ++i)
                        for (int axis = 0; axis < 3; ++axis)
                            vectors[i](axis) = values.data()[3 * i + axis];
                    a.magmoms_ = std::move(vectors);
                    a.magmom_scalar_.clear();
                    a.collinear_ = false;
                }
            })

        .def_property("total_charge",
            [](const Atoms &a) -> std::optional<double> {
                const auto found = a.global_attribs_d_.find("total_charge");
                if (found == a.global_attribs_d_.end()) return std::nullopt;
                return found->second;
            },
            [](Atoms &a, std::optional<double> charge) {
                if (!charge) {
                    a.global_attribs_d_.erase("total_charge");
                } else {
                    if (!std::isfinite(*charge))
                        throw py::value_error("Total charge must be finite");
                    a.global_attribs_d_["total_charge"] = *charge;
                }
            })

        // ── Flags (read-only) ──
        .def_property_readonly("has_energy", [](const Atoms &a) { return a.have_energy; })
        .def_property_readonly("has_forces", [](const Atoms &a) { return !a.forces_.empty(); })
        .def_property_readonly("has_stress", [](const Atoms &a) { return a.have_stress; })
        .def_property_readonly("has_velocities", [](const Atoms &a) { return !a.velocities_.empty(); })
        .def_property_readonly("has_site_energies", [](const Atoms &a) { return !a.site_energies_.empty(); })

        // ── cartesian flag ──
        .def_readwrite("cartesian", &Atoms::cartesian)

        // ── masks ──
        .def_property_readonly("masks", [](const Atoms &a) { return a.masks; })
        .def_property_readonly("group_count", [](const Atoms &a) { return a.group_count; })

        // ── Molecular topology ──
        .def_property_readonly("molid", [](const Atoms &a) { return a.molid_; })
        .def_property_readonly("molsize", [](const Atoms &a) { return a.molsize_; })
        .def_property_readonly("bonds", [](const Atoms &a) { return a.bonds_; })
        .def_property_readonly("angles", [](const Atoms &a) { return a.angles_; })

        // ── Methods ──
        .def("size", [](const Atoms &a) { return a.size(); })
        .def("__len__", [](const Atoms &a) { return a.size(); })
        .def("get_volume", [](const Atoms &a) { return a.get_volume(); })
        .def("wrap", [](Atoms &a) { a.wrap(); })
        .def("to_fractional", [](Atoms &a) { a.to_fractional(); })
        .def("to_cartesian", [](Atoms &a) { a.to_cartesian(); })

        // ── Copy support ──
        .def("copy", [](const Atoms &a) { return Atoms(a); })
        .def("__copy__", [](const Atoms &a) { return Atoms(a); })
        .def("__deepcopy__", [](const Atoms &a, py::dict) { return Atoms(a); })

        // ── Repr ──
        .def("__repr__", [](const Atoms &a) {
            return "Atoms(natoms=" + std::to_string(a.size())
                 + ", has_energy=" + (a.have_energy ? "True" : "False")
                 + ", has_forces=" + (a.forces_.empty() ? "False" : "True") + ")";
        });
}

// ============================================================================
//  Potential binding
// ============================================================================
static void validate_potential_atoms(const Potentials::AnyPotential &p, const Atoms &atom) {
    if (atom.chemmap_.size() != atom.pos_.size())
        throw py::value_error("chemmap must contain one species index per atom");
    for (int species : atom.chemmap_) {
        // A zero count means no declared species limit (e.g. mindist).
        if (species < 0 || (p.species_count > 0 && species >= p.species_count))
            throw py::value_error("chemmap species index is outside the potential's species range");
    }
}

static GPSettings curvature_query_settings(const std::string &method, double step) {
    if (method != "auto" && method != "analytic" && method != "force_fd")
        throw py::value_error("method must be auto, analytic, or force_fd");
    if (!std::isfinite(step) || step <= 0.0)
        throw py::value_error("step must be finite and positive");
    GPSettings settings;
    settings.Modify("method", method);
    std::ostringstream step_text;
    step_text.precision(std::numeric_limits<double>::max_digits10);
    step_text << step;
    settings.Modify("nudge_pos", step_text.str());
    return settings;
}

void bind_potential(py::module_ &m) {
    py::class_<Potentials::ShellJob,std::shared_ptr<Potentials::ShellJob>>(m,"_ShellJob")
        .def("ready",&Potentials::ShellJob::ready)
        .def("result",&Potentials::ShellJob::result,py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("run_directory",&Potentials::ShellJob::run_directory);
    // ---- AnyPotential (abstract base, returned by potential_wrapper.inner) ----
    py::class_<Potentials::AnyPotential>(m, "_AnyPotential")
        .def("submit",[](Potentials::AnyPotential &potential,const Atoms &atoms) {
            auto *shell=dynamic_cast<Potentials::PotShellCalc *>(&potential);
            if (!shell) throw py::value_error("submit requires a shellfile potential");
            validate_potential_atoms(potential,atoms);
            return shell->submit(atoms);
        },py::arg("atoms"),"Submit an independent asynchronous shell calculation")
        .def("getEFS", [](Potentials::AnyPotential &p, Atoms &atom) {
            validate_potential_atoms(p, atom);
            py::gil_scoped_release release;
            p.getEFS(atom);
        }, py::arg("atoms"),
           "Compute energy, forces, stress in-place (fills atoms fields)")
        .def("getE", [](Potentials::AnyPotential &p, Atoms &atom) {
            validate_potential_atoms(p, atom);
            py::gil_scoped_release release;
            p.getE(atom);
        }, py::arg("atoms"),
           "Compute energy only in-place (faster than getEFS)")
        .def("getEFSOMP", [](Potentials::AnyPotential &p, Atoms &atom) {
            validate_potential_atoms(p, atom);
            py::gil_scoped_release release;
            p.getEFSOMP(atom);
        }, py::arg("atoms"),
           "Compute EFS with OpenMP parallelism")
        .def("hessian", [](Potentials::AnyPotential &p, const Atoms &atom,
                            const std::string &method, double step) {
            validate_potential_atoms(p, atom);
            ComputeHessian calculator(curvature_query_settings(method, step));
            Atoms work = atom;
            work.resetEFS();
            Eigen::MatrixXd result;
            py::gil_scoped_release release;
            calculator.compute_fc(work, result, &p);
            return result;
        }, py::arg("atoms"), py::arg("method") = "auto", py::arg("step") = 1e-4)
        .def("hessian_vector_product", [](Potentials::AnyPotential &p, const Atoms &atom,
                const py::array_t<double, py::array::c_style | py::array::forcecast> &direction,
                const std::string &method, double step) {
            validate_potential_atoms(p, atom);
            if (direction.ndim() != 1)
                throw py::value_error("direction must be a one-dimensional Cartesian vector");
            ComputeHessian calculator(curvature_query_settings(method, step));
            Eigen::VectorXd probe(direction.size());
            std::copy_n(direction.data(), direction.size(), probe.data());
            Eigen::VectorXd result;
            py::gil_scoped_release release;
            calculator.compute_hvp(atom, probe, result, &p);
            return result;
        }, py::arg("atoms"), py::arg("direction"), py::arg("method") = "auto", py::arg("step") = 1e-4)
        .def("supports_analytical_hessian", &Potentials::AnyPotential::supports_analytical_hessian)
        .def("supports_analytical_hessian_vector_product",
             &Potentials::AnyPotential::supports_analytical_hessian_vector_product)
        .def_property("cutoff",
            [](Potentials::AnyPotential &p) { return p.cutoff; },
            [](Potentials::AnyPotential &p, double c) {
                if (dynamic_cast<Potentials::PotentialDescriptorLinear*>(&p) && c!=p.cutoff)
                    throw py::value_error("Descriptor linear cutoff is fixed by the model");
                p.cutoff = c; p.cutoffsq = c * c;
            })
        .def_property_readonly("cutoffsq", [](Potentials::AnyPotential &p) { return p.cutoffsq; })
        .def_property_readonly("species_count", [](Potentials::AnyPotential &p) { return p.species_count; })
        .def("supports_linear_fit", [](const Potentials::AnyPotential &p) { return p.supports_linear_fit(); })
        .def("supports_species_append", &Potentials::AnyPotential::supports_species_append)
        .def("append_species",[](Potentials::AnyPotential &p,const std::vector<int>& sources,
                                 const std::vector<std::string>& elements) {
            return p.append_species(sources,elements).old_to_new_parameters;
        },py::arg("clone_sources"),py::arg("new_elements")=std::vector<std::string>{})
        .def("save",[](Potentials::AnyPotential &p,const std::string& filename) {
            if(Potentials::save_gpm_if_requested(p,filename))return;
            if(!p.supports_species_append())
                throw py::value_error("This native save entry point requires an appendable MTP model");
            p.Save(filename);
        },py::arg("filename"))
        .def_property("coefficients", [](const Potentials::AnyPotential &p) {
            Eigen::VectorXd c(p.get_NL_coeffs_count()); p.nl_get_coeffs(c.data()); return c;
        }, [](Potentials::AnyPotential &p, const Eigen::VectorXd &c) {
            if (c.size()!=p.get_NL_coeffs_count() || !c.allFinite())
                throw py::value_error("Invalid potential coefficient vector");
            p.nl_set_coeffs(c.data());
        })
        .def_property_readonly("parameter_groups", [](const Potentials::AnyPotential &p) {
            py::dict result;
            for (const auto &g:p.nl_parameter_groups()) result[py::str(g.name)]=py::make_tuple(g.begin,g.end);
            return result;
        })
        .def("__repr__", [](Potentials::AnyPotential &p) {
            return "AnyPotential(cutoff=" + std::to_string(p.cutoff)
                 + ", species=" + std::to_string(p.species_count) + ")";
        });

    py::class_<Potentials::PotentialDescriptorLinear,Potentials::AnyPotential>(m,"_DescriptorLinear")
        .def("save",[](Potentials::PotentialDescriptorLinear &p,const std::string &file) {
            py::gil_scoped_release release;
            p.Save(file);
        },py::arg("filename"));

    // ---- PotentialWrapper (factory) ----
    py::class_<Potentials::potential_wrapper>(m, "_PotentialWrapper")
        .def(py::init<const std::string &, const std::string &>(),
             py::arg("potfile"), py::arg("pottype") = "auto",
             "Load a potential from file. pottype: mtp, eamalloy, eamfs, lj612, tersoff, sw, etc.")
        .def(py::init([](const std::map<std::string, std::string> &values) {
            GPSettings settings;
            for (const auto &[key, value] : values) settings.Modify(key, value);
            return std::make_unique<Potentials::potential_wrapper>(settings);
        }), py::arg("settings"), "Create a potential from unprefixed pot settings.")
        .def_readwrite("pottype", &Potentials::potential_wrapper::pottype)
        .def_readwrite("potfile", &Potentials::potential_wrapper::potfile)
        .def_property_readonly("inner",
            [](Potentials::potential_wrapper &w) -> Potentials::AnyPotential* {
                return w.pot;
            }, py::return_value_policy::reference_internal,
            "Access the underlying AnyPotential object")
        .def("__repr__", [](const Potentials::potential_wrapper &w) {
            return "PotentialWrapper(type='" + w.pottype + "', file='" + w.potfile + "')";
        });
}

// ============================================================================
//  GPSettings binding (configuration dict)
// ============================================================================
void bind_settings(py::module_ &m) {
    py::class_<GPSettings>(m, "_Settings")
        .def(py::init<>())
        .def(py::init([](const std::map<std::string, std::string> &d) {
            GPSettings s;
            for (const auto &[k, v] : d) s.Modify(k, v);
            return s;
        }), py::arg("settings_dict"))
        .def("__getitem__", [](GPSettings &s, const std::string &k) {
            return s[k];
        })
        .def("__setitem__", [](GPSettings &s, const std::string &k, const std::string &v) {
            s.Modify(k, v);
        })
        .def("__contains__", [](GPSettings &s, const std::string &k) {
            return s.settings.find(k) != s.settings.end();
        })
        .def("load", [](GPSettings &s, const std::string &fn) { s.Load(fn); },
             py::arg("filename"), "Load settings from .ini-style file")
        .def("modify", &GPSettings::Modify, py::arg("key"), py::arg("value"))
        .def("extract", &GPSettings::ExtractSubSettings, py::arg("prefix"),
             "Extract sub-settings with given prefix")
        .def("__len__", [](const GPSettings &s) { return s.settings.size(); })
        .def("__repr__", [](const GPSettings &s) {
            std::string r = "Settings({";
            int i = 0;
            for (const auto &[k, v] : s.settings) {
                if (i++ > 0) r += ", ";
                if (i > 10) { r += "..."; break; }
                r += "'" + k + "': '" + v + "'";
            }
            r += "})";
            return r;
        });
}

// ============================================================================
//  GPIO binding (structure I/O)
// ============================================================================
namespace {
std::optional<GPUnits::SourceUnits> python_source_units(const py::object &value) {
    if (value.is_none()) return std::nullopt;
    GPUnits::SourceUnits units;
    if (py::isinstance<py::str>(value)) {
        units = GPUnits::named_source_units(value.cast<std::string>());
    } else if (py::isinstance<py::dict>(value)) {
        for (auto item : value.cast<py::dict>()) {
            const auto key = py::cast<std::string>(item.first);
            const auto unit = py::cast<std::string>(item.second);
            if (key == "length") units.length = unit;
            else if (key == "energy") units.energy = unit;
            else if (key == "time") units.time = unit;
            else if (key == "mass") units.mass = unit;
            else throw py::value_error("Unknown source unit field: " + key);
        }
    } else throw py::type_error("input_units must be a preset string, dict or None");
    Atoms empty;
    GPUnits::canonicalize_atoms(empty, units);
    return units;
}

class GPIOStreamReader {
    GPIO io_{false};
    std::unique_ptr<std::istream> input_;
public:
    GPIOStreamReader(const std::string &filename, const std::string &format) {
        const auto iformat = format.empty() ? GPIO::guess_format(filename) : GPIO::format2enum(format);
        const char *sequential_format;
        switch (iformat) {
        case GPIO::EXTXYZ: sequential_format = "extxyz"; break;
        case GPIO::MLPCFG: sequential_format = "mlpcfg"; break;
        case GPIO::MLIPBIN: sequential_format = "bin"; break;
        case GPIO::N2P2DAT: sequential_format = "n2p2"; break;
        case GPIO::CP2KXYZ: sequential_format = "cp2kxyz"; break;
        case GPIO::XSF: sequential_format = "xsf"; break;
        case GPIO::GRODATA: sequential_format = "gro"; break;
        case GPIO::OUTCAR: sequential_format = "outcar"; break;
        case GPIO::XDATCAR: sequential_format = "xdatcar"; break;
        case GPIO::XDATCAR6: sequential_format = "xdatcar6"; break;
        default:
            throw py::value_error("iread supports only EXTXYZ, MLPCFG, MLIPBIN, N2P2, CP2KXYZ, XSF, GRO, OUTCAR, XDATCAR, XDATCAR6 formats");
        }
        py::gil_scoped_release release;

            input_ = open_seq_ifile(filename);

        if (!input_)
            throw std::runtime_error("Cannot open streaming input: " + filename);
        if (!io_.seq_read_init(*input_, sequential_format))
            throw std::runtime_error("Cannot initialize streaming input: " + filename);
    }

    std::optional<Atoms> read_next() {
        if (!input_) return std::nullopt;
        Atoms atom;
        bool ok;
        try {
            py::gil_scoped_release release;
            ok = io_.seq_read(*input_, atom);
            if (!ok && input_->bad())
                throw std::runtime_error("Streaming input read failed");
        } catch (...) {
            close();
            throw;
        }
        if (!ok) {
            close();
            return std::nullopt;
        }
        return atom;
    }

    void close() { io_.clean_state(); input_.reset(); }
};
} // namespace

void bind_io(py::module_ &m) {
    m.def("_detect_format", [](const std::string &filename, std::size_t limit) {
        GPIOFormat::Detection detected;
        {
            py::gil_scoped_release release;
            detected = GPIOFormat::detect_file(filename, limit);
        }
        py::dict result;
        result["format"] = detected.format;
        result["candidates"] = detected.candidates;
        return result;
    }, py::arg("filename"), py::arg("limit") = GPIOFormat::MAX_PREFIX,
       "Inspect a bounded decoded prefix; unknown and ambiguous inputs have an empty format.");

    py::class_<GPIOReader>(m, "_GPIOReader")
        .def("read_next", &GPIOReader::next, py::call_guard<py::gil_scoped_release>())
        .def("close", &GPIOReader::close, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("format", &GPIOReader::format);
    py::class_<GPIOWriter>(m, "_GPIOWriter")
        .def("write", &GPIOWriter::write, py::call_guard<py::gil_scoped_release>())
        .def("close", &GPIOWriter::close, py::call_guard<py::gil_scoped_release>());
    py::class_<GPIOStreamReader>(m, "_GPIOStreamReader")
        .def(py::init<const std::string &, const std::string &>(),
             py::arg("filename"), py::arg("format") = "")
        .def("read_next", &GPIOStreamReader::read_next,
             "Read an independent frame; return None at clean EOF.")
        .def("close", &GPIOStreamReader::close);

    py::class_<GPIO>(m, "_GPIO")
        .def(py::init([] { return std::make_unique<GPIO>(false); }))
        .def("open_reader", [](const GPIO &io, const std::string &filename,
                                const std::string &format, bool detect_content,
                                const py::object &input_units) {
            auto units = python_source_units(input_units);
            py::gil_scoped_release release;
            return std::make_unique<GPIOReader>(io, filename, format, detect_content, units);
        }, py::arg("filename"), py::arg("format") = "", py::kw_only(),
           py::arg("detect_content") = false, py::arg("input_units") = py::none(),
           "Open a reader with an independent snapshot of this configuration.")
        .def("open_writer", [](const GPIO &io, const std::string &filename,
                                const std::string &format) {
            py::gil_scoped_release release;
            return std::make_unique<GPIOWriter>(io, filename, format);
        }, py::arg("filename"), py::arg("format") = "",
           "Open a lazy writer with an independent configuration and stream.")
        .def_property("read_free_energy", &GPIO::read_free_energy, &GPIO::set_read_free_energy)
        .def_readwrite("read_startfrom", &GPIO::read_startfrom)
        .def_readwrite("read_endto", &GPIO::read_endto)
        .def_readwrite("read_irate", &GPIO::read_irate)
        .def_readwrite("append", &GPIO::append)

        // ── Read all ──
        .def("read", [](GPIO &configuration, const std::string &filename, const std::string &format) {
            auto state = configuration.new_session_state();
            GPIO &io = *state;
            std::vector<Atoms> atoms;
            bool ok;
            {
                py::gil_scoped_release release;
                ok = io.read(atoms, filename, format);
            }
            if (!ok)
                throw std::runtime_error("GPIO::read failed for: " + filename);
            return atoms;
        }, py::arg("filename"), py::arg("format") = "",
           "Read all structures from file. Format auto-detected if empty.")

        // ── Read one ──
        .def("read_one", [](GPIO &configuration, const std::string &filename, const std::string &format) {
            auto state = configuration.new_session_state();
            GPIO &io = *state;
            Atoms atom;
            bool ok;
            {
                py::gil_scoped_release release;
                ok = io.read_one(atom, filename, format);
            }
            if (!ok)
                throw std::runtime_error("GPIO::read_one failed for: " + filename);
            return atom;
        }, py::arg("filename"), py::arg("format") = "",
           "Read a single structure from file.")

        // ── Write list ──
        .def("write", [](GPIO &configuration, const std::vector<Atoms> &atoms,
                         const std::string &filename, const std::string &format) {
            auto state = configuration.new_session_state();
            GPIO &io = *state;
            // Need non-const copy for GPIO::write
            std::vector<Atoms> mutable_atoms = atoms;
            bool ok;
            {
                py::gil_scoped_release release;
                for (auto &atom : mutable_atoms)
                    if (!atom.cartesian) atom.to_cartesian();
                ok = io.write(mutable_atoms, filename, format);
            }
            if (!ok)
                throw std::runtime_error("GPIO::write failed for: " + filename);
        }, py::arg("atoms"), py::arg("filename"), py::arg("format") = "",
           "Write structures to file. Format auto-detected if empty.")

        // ── Write one ──
        .def("write_one", [](GPIO &configuration, const Atoms &atom,
                             const std::string &filename, const std::string &format) {
            auto state = configuration.new_session_state();
            GPIO &io = *state;
            Atoms mutable_atom = atom;
            bool ok;
            {
                py::gil_scoped_release release;
                if (!mutable_atom.cartesian) mutable_atom.to_cartesian();
                ok = io.write_one(mutable_atom, filename, format);
            }
            if (!ok)
                throw std::runtime_error("GPIO::write_one failed for: " + filename);
        }, py::arg("atom"), py::arg("filename"), py::arg("format") = "",
           "Write a single structure to file.")

        .def("write_iter", [](GPIO &io, const py::iterable &atoms,
                              const std::string &filename, const std::string &format) {
            std::unique_ptr<GPIO> state;
            auto session = [&]() -> GPIO& {
                if (!state) state = io.new_session_state();
                return *state;
            };
            const auto oformat = format.empty() ? GPIO::guess_format(filename) : GPIO::format2enum(format);

            if (oformat != GPIO::EXTXYZ && oformat != GPIO::MLPCFG &&
                oformat != GPIO::MLIPBIN && oformat != GPIO::XDATCAR && oformat != GPIO::XDATCAR6 &&
                oformat != GPIO::CP2KXYZ && oformat != GPIO::N2P2DAT)
                throw py::value_error("write_iter supports only EXTXYZ, MLPCFG, MLIPBIN, XDATCAR, XDATCAR6, CP2KXYZ and N2P2 formats");
            std::unique_ptr<std::ostream> output;
            try {
                for (py::handle frame : atoms) {
                    Atoms atom = frame.cast<Atoms>();
                    GPIO &worker = session();
                    py::gil_scoped_release release;
                    if (!atom.cartesian) atom.to_cartesian();
                    if (!output) {
                        output = open_ofile(filename, false, oformat == GPIO::MLIPBIN);
                        if (!output)
                            throw std::runtime_error("Cannot open streaming output: " + filename);
                    }
                    bool ok;
                    if (oformat == GPIO::XDATCAR) {
                        // write_one resets fixed-XDATCAR headers. The bulk
                        // stream path retains the first frame's cell/counts.
                        std::vector<Atoms> current;
                        current.push_back(std::move(atom));
                        ok = worker.write(current, *output, oformat);
                    } else {
                        ok = worker.write_one(atom, *output, oformat);
                    }
                    if (!ok)
                        throw std::runtime_error("Failed to write streaming output: " + filename);
                }
                if (output) {
                    py::gil_scoped_release release;
                    if (!close_ofile(output))
                        throw std::runtime_error("Failed to close streaming output: " + filename);
                }
            } catch (...) {
                if (output) {
                    py::gil_scoped_release release;
                    try { close_ofile(output); } catch (...) {}
                }
                throw;
            }
        }, py::arg("atoms"), py::arg("filename"), py::arg("format") = "",
           "Consume independent frames synchronously without collecting a trajectory.")

        // ── Format detection ──
        .def_static("guess_format", [](const std::string &filename) -> std::string {
            // Return a format string accepted by read/write, as the Python API
            // promises, rather than exposing an unregistered native enum.
            switch (GPIO::guess_format(filename)) {
            case GPIO::UNKNOWN: return "";
            case GPIO::MLPCFG: return "mlpcfg";
            case GPIO::CFG: return "CFG";
            case GPIO::EXTXYZ: return "extxyz";
            case GPIO::XSF: return "xsf";
            case GPIO::CIF: return "cif";
            case GPIO::OUTCAR: return "outcar";
            case GPIO::POSCAR: return "poscar";
            case GPIO::VASPRUN: return "vasprun";
            case GPIO::MLIPBIN: return "bin";
            case GPIO::XDATCAR: return "xdatcar";
            case GPIO::XDATCAR6: return "xdatcar6";
            case GPIO::N2P2DAT: return "n2p2";
            case GPIO::VASPABN: return "vaspabn";
            case GPIO::LMPTRJ: return "lmptrj";
            case GPIO::LMPDAT: return "lmpdat";
            case GPIO::LMPYAML: return "yaml";
            case GPIO::POSCARPACKAGE: return "poscars";
            case GPIO::GRODATA: return "gro";
            case GPIO::CP2KXYZ: return "cp2kxyz";
            case GPIO::VTK: return "vtk";
            case GPIO::DLPOLY4: return "dlpoly";
            case GPIO::MSI: return "msi";
            case GPIO::XSD: return "xsd";
            case GPIO::CAR: return "car";
            default: throw std::runtime_error("No Python format name for detected GPIO format");
            }
        }, py::arg("filename"),
           "Detect the I/O format string; return an empty string if unknown.")

        // ── Settings ──
        .def_readwrite("write_stress", &GPIO::write_stress)
        .def_readwrite("header_line", &GPIO::header_line)
        .def("set_elements", [](GPIO &io, const std::vector<std::string> &elements) {
            auto names = elements;
            const py::object parse_number = py::module_::import("builtins").attr("int");
            const py::int_ zero(0), last_element(118);
            for (auto &name : names) {
                try {
                    // Match imrmlp.utils.build_type_mapping's Python int syntax.
                    const py::object number = parse_number(name);
                    if (PyObject_RichCompareBool(number.ptr(), zero.ptr(), Py_GT) == 1 &&
                        PyObject_RichCompareBool(number.ptr(), last_element.ptr(), Py_LE) == 1)
                        name = Chemicals::chemical_symbols[number.cast<int>()];
                } catch (const py::error_already_set &error) {
                    if (!error.matches(PyExc_ValueError)) throw;
                    // Native GPIO also permits arbitrary species names.
                }
            }
            io.setmap(names);
        }, py::arg("elements"))
        .def("read_map", [](GPIO &io, const std::string &fn) {
            if (!io.readmap(fn))
                throw std::runtime_error("GPIO::read_map failed for: " + fn);
        }, py::arg("map_file"), "Load species name<->index mapping file.")

        .def("__repr__", [](const GPIO &io) {
            return "GPIO()";
        });
}

// ============================================================================
//  RelaxationDrvV2 binding
// ============================================================================
void bind_relax(py::module_ &m) {
    py::class_<RelaxationDrvV2>(m, "_Relaxation")
        .def(py::init<>())
        .def("set_elements", [](RelaxationDrvV2 &r, const std::vector<std::string> &elements) {
            r.iobj.setmap(elements);
            r.control.indexfile.clear();
        }, py::arg("elements"))

        // ── Init ──
        .def("init", [](RelaxationDrvV2 &r, Potentials::AnyPotential *pot, Atoms &atom,
                        const std::string &constraints_json) {
            r.initialized = false;
            r.reset_state();
            std::vector<regulator> constraints;
            if (!constraints_json.empty()) {
                std::istringstream input(constraints_json);
                const auto declarations = Constraints::parse_constraints(input);
                Constraints::configure_symmetry(declarations, r);
                constraints = Constraints::bind_constraints(declarations, atom);
            }
            r.flags.with_constraint_aul = !constraints.empty() && !r.constrained_solver();
            r.init(pot, atom, constraints.empty() ? nullptr : &constraints);
        }, py::arg("potential"), py::arg("atoms"), py::arg("constraints_json") = "",
           py::keep_alive<1, 2>(),
           "Initialize relaxation driver with potential and starting structure.")

        // ── Run ──
        .def("run", [](RelaxationDrvV2 &r) -> int {
            py::gil_scoped_release release;
            return r.run();
        }, "Run relaxation. Returns exit code (0 = converged).")
        .def("run_normal", [](RelaxationDrvV2 &r) -> int {
            py::gil_scoped_release release;
            return r.run_normal();
        }, "Run relaxation without constraints.")

        // ── Optional early cleanup; also safe before reinitialization ──
        .def("cleanup", [](RelaxationDrvV2 &r) {
            r.free_modifiers();
            if (r.ofs.is_open()) r.ofs.close();
        }, "Release modifiers and close trajectory output. Safe to call repeatedly.")

        // ── State (read-only) ──
        .def_property_readonly("converged", [](RelaxationDrvV2 &r) { return r.state_vars.converged; })
        .def_readonly("stop_reason", &RelaxationDrvV2::stop_reason)
        .def_property_readonly("n_iter", [](RelaxationDrvV2 &r) { return r.state_vars.curr_iter; })
        .def_property_readonly("energy", [](RelaxationDrvV2 &r) { return r.f; })
        .def_property_readonly("energy0", [](RelaxationDrvV2 &r) { return r.energy0; })
        .def_property_readonly("atom", [](RelaxationDrvV2 &r) -> Atoms& { return r.atom; },
                               py::return_value_policy::reference_internal)

        .def_readonly("kkt_residual", &RelaxationDrvV2::kkt_residual)
        .def_readonly("accepted_steps", &RelaxationDrvV2::accepted_steps)
        .def_readonly("fc_refresh_count", &RelaxationDrvV2::fc_refresh_count)
        .def("set_coordinate_map", [](RelaxationDrvV2 &r,
            std::function<Eigen::VectorXd(const Eigen::VectorXd &)> reduce,
            std::function<Eigen::VectorXd(const Eigen::VectorXd &)> lift,
            std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> jacobian,
            py::object curvature, py::object initialize) {
            auto map=std::make_shared<CallbackCoordinates>();
            if(!initialize.is_none())map->initializer=initialize.cast<std::function<void(const Eigen::VectorXd &)>>();
            map->to_reduced=std::move(reduce);map->to_full=std::move(lift);map->derivative=std::move(jacobian);
            if(!curvature.is_none())map->second_derivative=curvature.cast<std::function<Eigen::MatrixXd(const Eigen::VectorXd &,const Eigen::VectorXd &)>>();
            r.coordinate_map=std::move(map);
        }, py::arg("to_reduced"),py::arg("to_full"),py::arg("jacobian"),py::arg("curvature")=py::none(),py::arg("initialize")=py::none())
        .def("mapped_hessian", [](const RelaxationDrvV2 &r,const Eigen::VectorXd &q,
            const Eigen::VectorXd &gradient,const Eigen::MatrixXd &hessian) {
            if(!r.coordinate_map)throw std::logic_error("No nonlinear coordinate map is configured");
            return r.coordinate_map->pullback_hessian(q,gradient,hessian);
        },py::arg("point"),py::arg("gradient"),py::arg("hessian"))
        .def_property("hessian_precond_file",
            [](RelaxationDrvV2 &r) {return r.control.hessian_precond_file;},
            [](RelaxationDrvV2 &r,const std::string & value) {r.control.hessian_precond_file=value;})
        .def_property("precond_file",
            [](RelaxationDrvV2 &r) {return r.control.precond_file;},
            [](RelaxationDrvV2 &r,const std::string & value) {r.control.precond_file=value;})
        .def_property("sparse_precond",
            [](RelaxationDrvV2 &r) {return r.control.sparse_precond;},
            [](RelaxationDrvV2 &r,bool value) {r.control.sparse_precond=value;})
        .def_property("sparse_shift",
            [](RelaxationDrvV2 &r) {return r.control.sparse_shift;},
            [](RelaxationDrvV2 &r,double value) {r.control.sparse_shift=value;})
        .def_property("precond_floor",
            [](RelaxationDrvV2 &r) {return r.control.precond_floor;},
            [](RelaxationDrvV2 &r,double value) {r.control.precond_floor=value;})
        .def_property("fc_update_interval",
            [](RelaxationDrvV2 &r) {return r.control.fc_update_interval;},
            [](RelaxationDrvV2 &r,int value) {r.control.fc_update_interval=value;})
        .def_property("aul_auto",
            [](RelaxationDrvV2 &r) {return r.control.aul_auto;},
            [](RelaxationDrvV2 &r,bool value) {r.control.aul_auto=value;})
        .def_property("thin_cell_scaling",
            [](RelaxationDrvV2 &r) {return r.control.thin_cell_scaling;},
            [](RelaxationDrvV2 &r,bool value) {r.control.thin_cell_scaling=value;})
        .def_property("rigid_groups",
            [](RelaxationDrvV2 &r) {return r.control.rigid_groups;},
            [](RelaxationDrvV2 &r,const std::string & value) {r.control.rigid_groups=value;})
        .def_property("symmetry_operations",
            [](RelaxationDrvV2 &r) {return r.control.symmetry_operations;},
            [](RelaxationDrvV2 &r,const std::string & value) {r.control.symmetry_operations=value;})
        .def_property("symmetry_sites",
            [](RelaxationDrvV2 &r) {return r.control.symmetry_sites;},
            [](RelaxationDrvV2 &r,const std::string & value) {r.control.symmetry_sites=value;})
        .def_property("periodic_subspace", [](RelaxationDrvV2 &r){return r.flags.periodic_subspace;}, [](RelaxationDrvV2 &r,bool value){r.flags.periodic_subspace=value;})
        // ── Control parameters (most-used) ──
        .def_property("driver",
            [](RelaxationDrvV2 &r) { return r.control.driver; },
            [](RelaxationDrvV2 &r, const std::string &v) { r.control.driver = v; })
        .def_property("max_iter",
            [](RelaxationDrvV2 &r) { return r.control.max_iter; },
            [](RelaxationDrvV2 &r, double v) { r.control.max_iter = static_cast<int>(v); })
        .def_property("max_move",
            [](RelaxationDrvV2 &r) { return r.control.max_move; },
            [](RelaxationDrvV2 &r, double v) { r.control.max_move = v; })
        .def_property("pressure",
            [](RelaxationDrvV2 &r) { return r.control.pressure; },
            [](RelaxationDrvV2 &r, double v) { r.control.pressure = v; })
        .def_property("downhill_check",
            [](RelaxationDrvV2 &r) { return r.buffer_fire.downhill_check; },
            [](RelaxationDrvV2 &r, bool v) { r.buffer_fire.downhill_check = v; })
        .def_readwrite("mdmin_dt", &RelaxationDrvV2::mdmin_dt)
        .def_readwrite("quickmin_dt", &RelaxationDrvV2::quickmin_dt)
        .def_property("constraint_loops",
            [](RelaxationDrvV2 &r) { return r.control.constraint_iter_limit; },
            [](RelaxationDrvV2 &r, int v) { r.control.constraint_iter_limit = v; })
        .def_property("force_only",
            [](RelaxationDrvV2 &r) { return r.lbfgspp_param.linesearch == LBFGSpp::LBFGS_LINESEARCH_FORCE_ONLY_A; },
            [](RelaxationDrvV2 &r, bool value) { r.lbfgspp_param.linesearch = value ?
                LBFGSpp::LBFGS_LINESEARCH_FORCE_ONLY_A : LBFGSpp::LBFGS_LINESEARCH_LEWIS_OVERTON; })
        .def_property("max_linesearch",
            [](RelaxationDrvV2 &r) { return r.lbfgspp_param.max_linesearch; },
            [](RelaxationDrvV2 &r, int v) { r.lbfgspp_param.max_linesearch = v; })

        // ── Tolerance ──
        .def_property("gtol",
            [](RelaxationDrvV2 &r) { return r.tol.gtol; },
            [](RelaxationDrvV2 &r, double v) { r.tol.gtol = v; })
        .def_property("ftol",
            [](RelaxationDrvV2 &r) { return r.tol.ftol; },
            [](RelaxationDrvV2 &r, double v) { r.tol.ftol = v; })
        .def_property("xtol",
            [](RelaxationDrvV2 &r) { return r.tol.xtol; },
            [](RelaxationDrvV2 &r, double v) { r.tol.xtol = v; })

        // ── Flags ──
        .def_property("relax_cell",
            [](RelaxationDrvV2 &r) { return r.flags.relax_cell; },
            [](RelaxationDrvV2 &r, bool v) { r.flags.relax_cell = v; })
        .def_property("fix_vol",
            [](RelaxationDrvV2 &r) { return r.flags.fix_vol; },
            [](RelaxationDrvV2 &r, bool v) { r.flags.fix_vol = v; })
        .def_property("fix_shape",
            [](RelaxationDrvV2 &r) { return r.flags.fix_shape; },
            [](RelaxationDrvV2 &r, bool v) { r.flags.fix_shape = v; })
        .def_property("selective_dynamics",
            [](RelaxationDrvV2 &r) { return r.flags.selective_dynamics; },
            [](RelaxationDrvV2 &r, bool v) { r.flags.selective_dynamics = v; })
        .def_property("write_traj",
            [](RelaxationDrvV2 &r) { return r.flags.write_traj; },
            [](RelaxationDrvV2 &r, bool v) { r.flags.write_traj = v; })
        .def_property("symm_force",
            [](RelaxationDrvV2 &r) { return r.flags.symm_force; },
            [](RelaxationDrvV2 &r, bool v) { r.flags.symm_force = v; })
        .def_property("keep_symm",
            [](RelaxationDrvV2 &r) { return r.flags.keep_symm; },
            [](RelaxationDrvV2 &r, bool v) { r.flags.keep_symm = v; })
        .def_property("symprec",
            [](RelaxationDrvV2 &r) { return r.tol.symprec; },
            [](RelaxationDrvV2 &r, double v) { r.tol.symprec = v; })

        // ── Trajectory file ──
        .def_property("trajfile",
            [](RelaxationDrvV2 &r) { return r.control.trajfile; },
            [](RelaxationDrvV2 &r, const std::string &v) { r.control.trajfile = v; });
}

// ============================================================================
//  NEBDriver binding
// ============================================================================
void bind_neb(py::module_ &m) {
    m.def("_neb_path_seed", [](const std::vector<Atoms> &images, const std::map<std::string,std::string> &settings,
        const Eigen::Matrix3d &reference,const std::string &mode,std::uint64_t seed,int sign,double amplitude,
        double strain,const std::vector<int> &atoms,int center,double radius) {
        GPSettings values;for(const auto &[key,value]:settings)values.Modify(key,value);
        NEBDriver geometry;geometry.ApplySettings(values);geometry.prepare_complete_path(images,reference);
        NEBSeedOptions options;options.mode=mode;options.seed=seed;options.sign=sign;options.amplitude=amplitude;
        options.strain=strain;options.atoms=atoms;options.center=center;options.radius=radius;
        NEBSeedResult result;
        {py::gil_scoped_release release;result=NEBPathSeeds::generate(geometry,options);}
        py::dict info;info["status"]=result.status;info["reason"]=result.reason;info["subspace"]=result.subspace;
        info["selected_atoms"]=result.selected_atoms;info["atomic_displacement_A"]=result.atomic_displacement;
        info["cell_deformation"]=result.cell_deformation;
        return py::make_tuple(result.images,info);
    },py::arg("images"),py::arg("settings"),py::arg("reference"),py::arg("mode"),py::arg("seed"),
      py::arg("sign"),py::arg("amplitude"),py::arg("strain"),py::arg("atoms"),py::arg("center"),py::arg("radius"));

    py::class_<NEBDriver>(m, "_NEB")
        .def(py::init<>())
        .def_readwrite("periodic_subspace", &NEBDriver::periodic_subspace)
        .def_property("iobj",
            [](NEBDriver &d) { return d.iobj; },
            py::cpp_function([](NEBDriver &d, GPIO *io) { d.iobj = io; },
                             py::keep_alive<1, 2>()),
            py::return_value_policy::reference_internal)

        // ── Init ──
        .def("init", [](NEBDriver &neb, std::vector<Atoms> &images) {
            neb.init(images);
        }, py::arg("images"), "Initialize NEB with image chain.")

        .def_property("pot",
            [](const NEBDriver &d) { return d.pot; },
            py::cpp_function([](NEBDriver &d, Potentials::AnyPotential *pot) {
                d.pot = pot;
            }, py::keep_alive<1, 2>()),
            py::return_value_policy::reference_internal,
            "AnyPotential* for force evaluation")

        // ── Path initialization ──
        .def("fill_path", [](NEBDriver &neb) {
            neb.fill_path();
        }, "Initialize path by linear interpolation between endpoints.")

        // ── Evaluation / optimization ──
        .def("eval", [](NEBDriver &neb) {
            py::gil_scoped_release release;
            neb.eval();
        }, "Evaluate energies, forces, and stresses for the current path.")
        .def("run", [](NEBDriver &neb, const std::string &driver,
                        int max_iterations, double max_move) {
            py::gil_scoped_release release;
            NEBDriver::RunOptions options;
            options.driver=driver;options.max_iterations=max_iterations;options.max_move=max_move;
            const int result=neb.run(options);
            neb.check_symmetry_release(options);
            return result;
        }, py::arg("driver") = "fire", py::arg("max_iterations") = 1000,
           py::arg("max_move") = 0.1,
           "Optimize the NEB path and return 0 only when converged.")

        // ── Results ──
        .def_property_readonly("images", [](NEBDriver &neb) -> std::vector<Atoms>& {
            return neb.images;
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("n_images", [](NEBDriver &neb) { return neb.images.size(); })
        .def_readwrite("iter", &NEBDriver::iter)
        .def_property_readonly("max_perp_force", [](NEBDriver &neb) { return neb.max_perp_force; })
        .def_property_readonly("max_spring_force", [](NEBDriver &neb) { return neb.max_spring_force; })
        .def_property_readonly("max_neb_force", [](NEBDriver &neb) { return neb.max_neb_force; })
        .def_property_readonly("converged", [](NEBDriver &neb) { return neb.converged; })
        .def_property_readonly("stop_reason", [](NEBDriver &neb) { return neb.stop_reason_string(); })
        .def_property_readonly("optimizer_iterations", [](NEBDriver &neb) { return neb.optimizer_iterations; })
        .def_property_readonly("force_evaluations", [](NEBDriver &neb) { return neb.force_evaluations; })
        .def_readwrite("solver_policy", &NEBDriver::solver_policy)
        .def_readwrite("max_cell_strain", &NEBDriver::max_cell_strain)
        .def_readwrite("max_force_evaluations", &NEBDriver::max_force_evaluations)
        .def_readwrite("diagnostics_file", &NEBDriver::diagnostics_file)
        .def_readonly("accepted_steps", &NEBDriver::accepted_steps)
        .def_readonly("rejected_steps", &NEBDriver::rejected_steps)
        .def_readonly("image_force_evaluations", &NEBDriver::image_force_evaluations)
        .def_readonly("last_atomic_move", &NEBDriver::last_atomic_move)
        .def_readonly("last_cell_change", &NEBDriver::last_cell_change)
        .def_readonly("last_fallback", &NEBDriver::last_fallback)
        .def_property_readonly("solver_phase", [](const NEBDriver &neb) {
            return neb.climbing_active ? "climbing" : (neb.vally && !neb.ivallys.empty() ? "valley" : "ordinary");
        })
        .def_property_readonly("climbing_image", [](NEBDriver &neb) { return neb.climbing_image; })

        // ── Parameters ──
        .def_readwrite("spring", &NEBDriver::spring)
        .def_readwrite("climb", &NEBDriver::climb)
        .def_readwrite("climb_mode", &NEBDriver::climb_mode)
        .def_readwrite("climb_delay", &NEBDriver::climb_delay)
        .def_readwrite("vcneb", &NEBDriver::vcneb)
        .def_readwrite("pressure", &NEBDriver::pressure)
        .def_readwrite("cell_weight", &NEBDriver::cell_weight)
        .def_readwrite("neb_ftol", &NEBDriver::neb_ftol)
        .def_readwrite("tangent_type", &NEBDriver::tangent_type)
        .def_readwrite("sproj_type", &NEBDriver::sproj_type)
        .def_readwrite("omp_mode", &NEBDriver::omp_mode)
        .def_readwrite("keep_symm", &NEBDriver::keep_symm)
        .def_readwrite("symprec", &NEBDriver::symprec)
        .def_readwrite("symmetry_operations", &NEBDriver::symmetry_operations)
        .def_readwrite("symmetry_diagnostics", &NEBDriver::symmetry_diagnostics)
        .def_readwrite("perturb_amplitude", &NEBDriver::perturb_amplitude)
        .def_readwrite("perturb_seed", &NEBDriver::perturb_seed)
        .def_readwrite("symmetry_release", &NEBDriver::symmetry_release)
        .def_readwrite("symmetry_release_trials", &NEBDriver::symmetry_release_trials)
        .def_readwrite("symmetry_release_seed", &NEBDriver::symmetry_release_seed)
        .def_readwrite("symmetry_release_steps", &NEBDriver::symmetry_release_steps)
        .def_readwrite("symmetry_release_evaluations", &NEBDriver::symmetry_release_evaluations)
        .def_readwrite("symmetry_release_amplitude", &NEBDriver::symmetry_release_amplitude)
        .def_readwrite("symmetry_release_strain", &NEBDriver::symmetry_release_strain)
        .def_readwrite("symmetry_release_energy_tol", &NEBDriver::symmetry_release_energy_tol)
        .def_readonly("symmetry_release_status", &NEBDriver::symmetry_release_status)
        .def_readwrite("init_method", &NEBDriver::init_method)
        .def_readwrite("check_mic", &NEBDriver::check_mic)
        .def_readwrite("write_traj", &NEBDriver::write_traj)
        .def_readwrite("nblock", &NEBDriver::nblock)
        .def_readwrite("variable_image", &NEBDriver::variable_image)
        .def_readwrite("image_adjust_interval", &NEBDriver::image_adjust_interval)
        .def_readwrite("image_insert_thresh", &NEBDriver::image_insert_thresh)
        .def_readwrite("image_remove_thresh", &NEBDriver::image_remove_thresh)
        .def_readwrite("max_image_count", &NEBDriver::max_image_count)
        .def_readwrite("min_image_count", &NEBDriver::min_image_count);
}

// ============================================================================
//  TrainerLinearPotential binding
// ============================================================================
void bind_dimer(py::module_ &m) {
    py::class_<DimerDriver>(m, "_Dimer")
        .def(py::init<>())
        .def_readwrite("periodic_subspace", &DimerDriver::periodic_subspace)
        .def_property("iobj",
            [](DimerDriver &d) { return d.iobj; },
            py::cpp_function([](DimerDriver &d, GPIO *io) { d.iobj = io; },
                             py::keep_alive<1, 2>()),
            py::return_value_policy::reference_internal)

        // ── Init ──
        .def("init", [](DimerDriver &d, Atoms &center) {
            d.init(center);
        }, py::arg("center_atom"), "Initialize dimer from center structure.")

        .def_property("pot",
            [](const DimerDriver &d) { return d.pot; },
            py::cpp_function([](DimerDriver &d, Potentials::AnyPotential *pot) {
                d.pot = pot;
            }, py::keep_alive<1, 2>()),
            py::return_value_policy::reference_internal,
            "AnyPotential* for force evaluation")

        // ── Run ──
        .def("run", [](DimerDriver &d) -> int {
            py::gil_scoped_release release;
            return d.run();
        }, "Run dimer optimization. Returns exit code.")

        // ── Results ──
        .def_property_readonly("atom_center", [](DimerDriver &d) -> Atoms& {
            return d.atom_center;
        }, py::return_value_policy::reference_internal)
        .def_property_readonly("converged", [](DimerDriver &d) { return d.converged; })
        .def_property_readonly("energy", [](DimerDriver &d) { return d.energy; })
        .def_property_readonly("curvature", [](DimerDriver &d) { return d.curvature; })
        .def_property_readonly("n_iter", [](DimerDriver &d) { return d.iter; })

        .def_readonly("max_trn_force", &DimerDriver::max_trn_force)

        .def_readonly("max_rot_force", &DimerDriver::max_rot_force)

        .def_readonly("max_center_gradient", &DimerDriver::max_center_gradient)

        // ── Parameters ──
        .def_readwrite("modecarf", &DimerDriver::modecarf)
        .def_readwrite("random_seed", &DimerDriver::random_seed)
        .def_readwrite("rot_gtol", &DimerDriver::rot_gtol)
        .def_readwrite("trn_gtol", &DimerDriver::trn_gtol)
        .def_readwrite("rot_force_min", &DimerDriver::rot_force_min)
        .def_readwrite("min_curvature", &DimerDriver::min_curvature)
        .def_readwrite("symm_strain", &DimerDriver::symm_strain)
        .def_readwrite("restore_F_cholesky", &DimerDriver::restore_F_cholesky)
        .def_readwrite("recast_atomic_force", &DimerDriver::recast_atomic_force)
        .def_readwrite("trajfile", &DimerDriver::trajfile)
        .def_readwrite("summary", &DimerDriver::summary)
        .def_readwrite("ddr", &DimerDriver::ddr)
        .def_readwrite("max_iter", &DimerDriver::max_iter)
        .def_readwrite("dimer_gtol", &DimerDriver::dimer_gtol)
        .def_readwrite("rot_driver", &DimerDriver::rot_driver)
        .def_readwrite("trn_driver", &DimerDriver::trn_driver)
        .def_readwrite("rot_max_iter", &DimerDriver::rot_max_iter)
        .def_readwrite("trn_max_iter", &DimerDriver::trn_max_iter)
        .def_readwrite("vc_flag", &DimerDriver::vc_flag)
        .def_readwrite("pressure", &DimerDriver::pressure)
        .def_readwrite("write_traj", &DimerDriver::write_traj);
}

// ============================================================================
//  TrainerLinearPotential binding
// ============================================================================
static void restore_python_training_fitness(Atoms &a) {
    if (const auto it = a.local_attribs_d_.find("atomic_fitness");
        it != a.local_attribs_d_.end()) {
        if (it->second.size() != a.size())
            throw std::invalid_argument("atomic_fitness count differs from atom count");
        a.atomic_fitness.assign(it->second.data(), it->second.data() + it->second.size());
    }
    if (const auto it = a.global_attribs_d_.find("fitness");
        it != a.global_attribs_d_.end()) a.fitness = it->second;
}

void bind_train(py::module_ &m) {
    // A synchronous call owns the trainer and its data until optimization ends;
    // there is no partially initialized Python trainer or dangling potential.
    m.def("_fit_nonlinear",[](Potentials::AnyPotential &pot,std::vector<Atoms> atoms,
        std::vector<Atoms> validation,const std::map<std::string,std::string> &values,
        std::vector<Atoms> test, Potentials::AnyPotential *fixed_core, std::string hybrid_output) {
        if (atoms.empty() || pot.get_NL_coeffs_count()<=0 || !pot.supports_site_parameter_gradients())
            throw py::value_error("Nonlinear fitting requires data and analytical site parameter gradients");
        const auto hessian_setting = values.find("hessian_weight");
        const double hessian_weight = hessian_setting == values.end() ? 0. :
            Parameters::parse_value<double>(hessian_setting->second);
        if (!std::isfinite(hessian_weight) || hessian_weight < 0)
            throw py::value_error("Full Hessian weight must be finite and nonnegative");
        const auto hvp_setting = values.find("hvp_weight");
        const double hvp_weight = hvp_setting == values.end() ? 0. :
            Parameters::parse_value<double>(hvp_setting->second);
        if (!std::isfinite(hvp_weight) || hvp_weight < 0)
            throw py::value_error("HVP weight must be finite and nonnegative");
        for (const auto &a:atoms) {
            validate_potential_atoms(pot,a);
            a.validate_hvp();
            if (a.size()==0 || !(a.has_energy() || a.has_forces() || a.has_stresses() ||
                                (hvp_weight > 0 && a.has_hvp()) || (hessian_weight > 0 && a.has_hessian())))
                throw py::value_error("Every training structure needs atoms and an EFS label or positively weighted HVP/full-Hessian labels");
        }
        for (const auto &a:validation) {
            validate_potential_atoms(pot,a);
            if (a.size()==0) throw py::value_error("Validation structures must contain atoms");
        }
        for (const auto &a:test) {
            validate_potential_atoms(pot,a);
            if (a.size()==0) throw py::value_error("Test structures must contain atoms");
        }
        GPSettings settings;
        for (const auto &[key,value]:values) settings.Modify(key,value);
        TrainNonlinear trainer(settings);
        if(fixed_core) Training::validate_fixed_core(pot,*fixed_core);
        const auto device=values.find("device");
        if (device!=values.end() && device->second!="cpu")
            throw py::value_error("This Python training entry point supports CPU training");
        double loss=0.;
        Training::WorkflowResult workflow;
        TrainNonlinear *active = &trainer;
        {
            py::gil_scoped_release release;
            if(fixed_core) {
                atoms=Training::fixed_core_residuals(*fixed_core,atoms,trainer.hvp_fd_step(),trainer.hessian_fd_step());
                validation=Training::fixed_core_residuals(*fixed_core,validation,trainer.hvp_fd_step(),trainer.hessian_fd_step());
                test=Training::fixed_core_residuals(*fixed_core,test,trainer.hvp_fd_step(),trainer.hessian_fd_step());
            }
            std::vector<Neighborhoods> nbhs,valid_nbhs;
            for (auto &a:atoms) {
                restore_python_training_fitness(a);
                if (!a.cartesian) a.to_cartesian();
                nbhs.emplace_back(a,pot.cutoff);
            }
            for (auto &a:validation) {
                restore_python_training_fitness(a);
                if (!a.cartesian) a.to_cartesian();
                valid_nbhs.emplace_back(a,pot.cutoff);
            }
            for (auto &a:test) if (!a.cartesian) a.to_cartesian();
            trainer.init(atoms,nbhs,&pot);
            if (!validation.empty()) {
                trainer.monit_valid_err=true;
                trainer.init_valid(validation,valid_nbhs);
            }
            auto options = trainer.workflow_options;
            options.validation_metric = trainer.CurrentSettings()["validation_metric"];
            workflow = Training::run_nonlinear_workflow(trainer, test, options,
                [](TrainNonlinear &current) { current.TrainBFGS(); });
            if (workflow.final_trainer) active = workflow.final_trainer.get();
            Eigen::VectorXd gradient(pot.get_NL_coeffs_count()); active->AccumulateGard(loss,gradient);
            if(fixed_core) {
                if(hybrid_output.empty()) hybrid_output=trainer.CurrentSettings()["save_to"]+".hybrid";
                Training::save_fixed_core_hybrid(pot,*fixed_core,hybrid_output,&atoms.front());
            }
        }
        py::dict result;
        result["feature_count"]=pot.get_NL_coeffs_count();
        result["train_erms"]=active->emonitor.erms; result["train_frms"]=active->emonitor.frms;
        result["train_srms"]=active->emonitor.srms; result["loss"]=loss;
        result["train_hessian_rms"]=active->emonitor.hrms;
        result["train_hessian_loss"]=active->emonitor.loss_h;
        result["train_hvp_rms"]=active->emonitor.hvp_rms;
        result["train_hvp_loss"]=active->emonitor.loss_hvp;
        result["iterations"]=active->bfgs_accepted_iter;
        if (!validation.empty() && !trainer.workflow_options.refit_train_valid) {
            result["valid_erms"]=active->emonitor_valid.erms; result["valid_frms"]=active->emonitor_valid.frms;
            result["valid_srms"]=active->emonitor_valid.srms;
            result["valid_hessian_rms"]=active->emonitor_valid.hrms;
            result["valid_hessian_loss"]=active->emonitor_valid.loss_h;
            result["valid_hvp_rms"]=active->emonitor_valid.hvp_rms;
            result["valid_hvp_loss"]=active->emonitor_valid.loss_hvp;
        }
        auto optional_number = [](const std::optional<double> &value) -> py::object {
            return value ? py::cast(*value) : py::none();
        };
        result["test_erms"] = optional_number(workflow.test.energy);
        result["test_frms"] = optional_number(workflow.test.force);
        result["test_stress_gpa"] = optional_number(workflow.test.stress);
        result["selected_l2_multiplier"] = optional_number(workflow.selected_multiplier);
        py::list candidates;
        for (const auto &candidate : workflow.candidates) {
            py::dict entry;
            entry["multiplier"] = candidate.multiplier;
            entry["score"] = candidate.score;
            candidates.append(entry);
        }
        result["regularization_candidates"] = candidates;
        if(fixed_core) result["hybrid_file"]=hybrid_output;
        return result;
    },py::arg("potential"),py::arg("training_set"),py::arg("validation_set"),py::arg("settings"),
      py::arg("test_set")=std::vector<Atoms>{}, py::arg("fixed_core")=nullptr, py::arg("hybrid_output")="");

    py::class_<TrainerLinearPotential>(m, "_LinearTrainer")
        .def(py::init<>())

        // ── Init (auto-builds neighborhoods from atoms) ──
        .def("init", [](TrainerLinearPotential &t,
                         std::vector<Atoms> training_set,
                         Potentials::AnyPotential *pot) {
            if (!pot) throw py::value_error("Linear training requires a potential");
            // Build neighborhoods for each structure
            std::vector<Neighborhoods> nbhs;
            nbhs.reserve(training_set.size());
            for (auto &a : training_set) {
                restore_python_training_fitness(a);
                if (!a.cartesian) a.to_cartesian();
                nbhs.emplace_back(a, pot->cutoff);
            }
            t.init(training_set, nbhs, pot);
        }, py::arg("training_set"), py::arg("potential"), py::keep_alive<1, 3>(),
           "Initialize trainer with training structures (neighborhoods built automatically).")

        // ── Train ──
        .def("assemble", [](TrainerLinearPotential &t) {
            py::gil_scoped_release release;
            t.Assemble();
        }, "Assemble the design matrix A^T A and A^T y.")
        .def("solve", &TrainerLinearPotential::Solve,
             "Solve the regularized normal equations.")
        .def("train", [](TrainerLinearPotential &t) {
            py::gil_scoped_release release;
            t.Train();
        }, "Assemble + Solve in one step.")

        // ── Save ──
        .def("save", &TrainerLinearPotential::save_current, py::arg("filename"),
             "Save the trained potential to file.")

        // ── Parameters (via ConnectSettings keys) ──
        .def("set_params", [](TrainerLinearPotential &t,
                               const std::map<std::string, std::string> &params) {
            GPSettings s;
            for (const auto &[k, v] : params) s.Modify(k, v);
            t.ApplySettings(s);
        }, py::arg("params"),
           "Set training parameters via dict. Keys: energy_weight, force_weight, "
           "stress_weight, energy_pow, force_pow, stress_pow, hvp_weight, hvp_pow, "
           "hvp_fd_step, weight_L2, save_to.")
        .def("print_settings", [](TrainerLinearPotential &t) {
            t.PrintSettings();
        }, "Print current training settings.")

        // ── Info ──
        .def_property_readonly("feature_count", &TrainerLinearPotential::get_feature_count)
        .def_property_readonly("train_erms", [](TrainerLinearPotential &t) { return t.emonitor.erms; })
        .def_property_readonly("train_frms", [](TrainerLinearPotential &t) { return t.emonitor.frms; })
        .def_property_readonly("train_srms", [](TrainerLinearPotential &t) { return t.emonitor.srms; })
        .def_property_readonly("train_hvp_rms", [](TrainerLinearPotential &t) { return t.emonitor.hvp_rms; })
        .def_property_readonly("train_hvp_loss", [](TrainerLinearPotential &t) { return t.emonitor.loss_hvp; });
}

// ============================================================================
//  Descriptor binding
// ============================================================================
static void validate_descriptor_species(const descriptor_wrapper &dw, const Atoms &source) {
    if (source.chemmap_.size() != source.pos_.size())
        throw py::value_error("chemmap must contain one species index per atom");
    const auto *so4 = dynamic_cast<const Descriptor_BispectrumSO4 *>(dw.desc.get());
    const auto *behler = dynamic_cast<const DescriptorBehler *>(dw.desc.get());
    if (behler) behler->validate_species(source.chemmap_);
    for (int species : source.chemmap_)
        if (species < 0 ||
            (so4 && (static_cast<size_t>(species) >= so4->weights_.size() ||
                     static_cast<size_t>(species) >= so4->selfweights_.size())))
            throw py::value_error("chemmap species index is outside the descriptor's species range");
}

void bind_descriptors(py::module_ &m) {
    py::class_<descriptor_wrapper>(m, "_Descriptor")
        .def(py::init([](const std::string &dtype,
                          const std::map<std::string, std::string> &params) {
            GPSettings s;
            s.Modify("dtype", dtype);
            for (const auto &[k, v] : params) s.Modify(k, v);
            return new descriptor_wrapper(s);
        }), py::arg("dtype"), py::arg("params") = std::map<std::string, std::string>(),
           "Create descriptor. dtype: 'biso3', 'behler', 'bondorder', 'biso4', 'mtm'.")
        .def("feature_size", &descriptor_wrapper::feature_size)
        .def_readonly("dtype", &descriptor_wrapper::dtype)
        .def("compute", [](descriptor_wrapper &dw, const Atoms &source) {
            validate_descriptor_species(dw, source);
            const py::ssize_t n = source.size();
            const py::ssize_t d = dw.feature_size();
            py::array_t<double> values({n, d});
            if (n == 0) return values;
            auto output = values.mutable_unchecked<2>();
            {
                py::gil_scoped_release release;
                Atoms atom(source);
                if (!atom.cartesian) atom.to_cartesian();
                Neighborhoods nbhs(atom, dw.desc->cutoff_);
                Eigen::VectorXd local;
                for (int i = 0; i < nbhs.size(); ++i) {
                    dw.compute(nbhs[i], local);
                    for (int f = 0; f < d; ++f) output(i, f) = local(f);
                }
            }
            return values;
        }, py::arg("atoms"),
           "Compute local descriptor values with shape (N, feature_size), in atom order. "
           "SO4 averaging settings do not change this local result.")
        .def("compute_averaged", [](descriptor_wrapper &dw, const Atoms &source) {
            validate_descriptor_species(dw, source);
            Eigen::VectorXd averaged = Eigen::VectorXd::Zero(dw.feature_size());
            if (source.size() > 0) {
                py::gil_scoped_release release;
                Atoms atom(source);
                if (!atom.cartesian) atom.to_cartesian();
                Neighborhoods nbhs(atom, dw.desc->cutoff_);
                if (auto *so4=dynamic_cast<Descriptor_BispectrumSO4 *>(dw.desc.get());so4 && so4->uses_inner_density()) {
                    so4->compute_global_density(nbhs,averaged);
                } else {
                    Eigen::VectorXd local;
                    for (int i = 0; i < nbhs.size(); ++i) {
                        dw.compute(nbhs[i], local);
                        averaged += local;
                    }
                    averaged /= static_cast<double>(nbhs.size());
                }
            }
            return vector_to_numpy_copy(averaged);
        }, py::arg("atoms"),
           "Compute a structure fingerprint with shape (feature_size,). "
           "SO4 inner-density average density coefficients before invariants; other modes average local values. "
           "Empty structures return a zero vector.")
        .def("compute_averaged_with_gradient", [](descriptor_wrapper &dw, const Atoms &source) {
            validate_descriptor_species(dw, source);
            const py::ssize_t d = dw.feature_size();
            const py::ssize_t n = source.size();
            Eigen::VectorXd averaged = Eigen::VectorXd::Zero(d);
            Eigen::Tensor<double, 3, Eigen::RowMajor> jacobian(d, n, 3);
            jacobian.setZero();
            if (n > 0) {
                py::gil_scoped_release release;
                Atoms atom(source);
                if (!atom.cartesian) atom.to_cartesian();
                Neighborhoods nbhs(atom, dw.desc->cutoff_);
                if (auto *so4=dynamic_cast<Descriptor_BispectrumSO4 *>(dw.desc.get());so4 && so4->uses_inner_density()) {
                    so4->compute_global_density(nbhs,averaged,&jacobian);
                } else {
                    Eigen::VectorXd local;
                    DescriptorGradient gradient;
                    const double scale = 1.0 / static_cast<double>(n);
                    for (int i = 0; i < nbhs.size(); ++i) {
                        auto &site = nbhs[i];
                        dw.compute_with_gradient(site, local, gradient);
                        const auto &g = gradient.grad;
                        if (local.size() != d || g.dimension(0) != d ||
                            g.dimension(1) != site.nbhcount_ || g.dimension(2) != 3)
                            throw std::runtime_error("Descriptor returned an incompatible local gradient shape");
                        averaged += scale * local;
                        for (int j = 0; j < site.nbhcount_; ++j) {
                            const int source_index = site.indices_[j];
                            if (source_index == i) continue;
                            for (py::ssize_t f = 0; f < d; ++f)
                                for (int a = 0; a < 3; ++a) {
                                    const double value = scale * g(f, j, a);
                                    jacobian(f, source_index, a) += value;
                                    jacobian(f, i, a) -= value;
                                }
                        }
                    }
                }
            }
            py::array_t<double> derivatives({d, n, py::ssize_t(3)});
            if (jacobian.size() > 0)
                std::copy(jacobian.data(), jacobian.data() + jacobian.size(), derivatives.mutable_data());
            return py::make_tuple(vector_to_numpy_copy(averaged), derivatives);
        }, py::arg("atoms"),
           "Return structure values (D,) and their Cartesian fixed-cell Jacobian (D, N, 3). "
           "Differentiate SO4 inner-density averaging or the local mean for all other modes. "
           "Source atoms retain their order; empty inputs return zeros and arrays own their data.")
        .def("compute_averaged_with_strain_gradient", [](descriptor_wrapper &dw, const Atoms &source) {
            validate_descriptor_species(dw, source);
            const py::ssize_t d = dw.feature_size();
            const py::ssize_t n = source.size();
            Eigen::VectorXd averaged = Eigen::VectorXd::Zero(d);
            Eigen::Tensor<double, 3, Eigen::RowMajor> jacobian(d, 3, 3);
            jacobian.setZero();
            if (n > 0) {
                py::gil_scoped_release release;
                Atoms atom(source);
                if (!atom.cartesian) atom.to_cartesian();
                Neighborhoods nbhs(atom, dw.desc->cutoff_);
                if (auto *so4=dynamic_cast<Descriptor_BispectrumSO4 *>(dw.desc.get());so4 && so4->uses_inner_density()) {
                    so4->compute_global_density(nbhs,averaged,nullptr,&jacobian);
                } else {
                    Eigen::VectorXd local;
                    DescriptorGradient gradient;
                    const double scale = 1.0 / static_cast<double>(n);
                    for (int i = 0; i < nbhs.size(); ++i) {
                        auto &site = nbhs[i];
                        dw.compute_with_gradient(site, local, gradient);
                        const auto &g = gradient.grad;
                        if (local.size() != d || g.dimension(0) != d ||
                            g.dimension(1) != site.nbhcount_ || g.dimension(2) != 3)
                            throw std::runtime_error("Descriptor returned an incompatible local gradient shape");
                        averaged += scale * local;
                        for (int j = 0; j < site.nbhcount_; ++j)
                            for (py::ssize_t f = 0; f < d; ++f)
                                for (int a = 0; a < 3; ++a)
                                    for (int b = 0; b < 3; ++b)
                                        jacobian(f, a, b) += scale * g(f, j, a) * site.pos_[j][b];
                    }
                }
            }
            py::array_t<double> derivatives({d, py::ssize_t(3), py::ssize_t(3)});
            std::copy(jacobian.data(), jacobian.data() + jacobian.size(), derivatives.mutable_data());
            return py::make_tuple(vector_to_numpy_copy(averaged), derivatives);
        }, py::arg("atoms"),
           "Return structure values (D,) and homogeneous strain Jacobian (D,3,3). "
           "Differentiate SO4 inner-density averaging or the local mean for other modes. "
           "Rows transform as R @ (I+epsilon).T; include self images and all nine raw "
           "components, without volume normalization. Empty inputs return zeros; arrays own data.")
        .def("compute_averaged_with_cell_gradient",[](descriptor_wrapper &dw,const Atoms &source) {
            validate_descriptor_species(dw,source);
            auto *so4=dynamic_cast<Descriptor_BispectrumSO4 *>(dw.desc.get());
            if(!so4 || !so4->uses_inner_density())throw std::invalid_argument("Raw cell Jacobian requires biso4:average=inner-density");
            Eigen::VectorXd value;Eigen::Tensor<double,3,Eigen::RowMajor> jac;
            {py::gil_scoped_release release;so4->compute_global_density_cell(source,value,jac);}
            py::array_t<double> result({py::ssize_t(dw.feature_size()),py::ssize_t(3),py::ssize_t(3)});
            std::copy(jac.data(),jac.data()+jac.size(),result.mutable_data());
            return py::make_tuple(vector_to_numpy_copy(value),result);
        },py::arg("atoms"))
        .def("compute_weighted_global_hvp",[](descriptor_wrapper &dw,const Atoms &source,
             const Eigen::VectorXd &weights,const Eigen::VectorXd &direction,bool include_cell) {
            validate_descriptor_species(dw,source);
            auto *so4=dynamic_cast<Descriptor_BispectrumSO4 *>(dw.desc.get());
            if(!so4 || !so4->uses_inner_density())throw std::invalid_argument("Global curvature requires biso4:average=inner-density");
            double value;Eigen::VectorXd gradient,product;
            {py::gil_scoped_release release;so4->compute_weighted_global_hvp(source,weights,direction,include_cell,value,gradient,product);}
            return py::make_tuple(value,vector_to_numpy_copy(gradient),vector_to_numpy_copy(product));
        },py::arg("atoms"),py::arg("weights"),py::arg("direction"),py::arg("include_cell")=false)
        .def("compute_weighted_global_hessian",[](descriptor_wrapper &dw,const Atoms &source,
             const Eigen::VectorXd &weights,bool include_cell) {
            validate_descriptor_species(dw,source);
            auto *so4=dynamic_cast<Descriptor_BispectrumSO4 *>(dw.desc.get());
            if(!so4 || !so4->uses_inner_density())throw std::invalid_argument("Global curvature requires biso4:average=inner-density");
            double value;Eigen::VectorXd gradient;Eigen::MatrixXd hessian;
            {py::gil_scoped_release release;so4->compute_weighted_global_hessian(source,weights,include_cell,value,gradient,hessian);}
            return py::make_tuple(value,vector_to_numpy_copy(gradient),matrix_to_numpy_copy(hessian));
        },py::arg("atoms"),py::arg("weights"),py::arg("include_cell")=false)
        .def("compute_with_gradient", [](descriptor_wrapper &dw, const Atoms &source,
                                         py::ssize_t center) {
            validate_descriptor_species(dw, source);
            if (center < 0 || center >= source.size())
                throw py::index_error("Descriptor center must be an atom index in [0, N)");

            const py::ssize_t d = dw.feature_size();
            Eigen::VectorXd local;
            std::vector<std::int64_t> indices;
            std::vector<double> jacobian;
            {
                py::gil_scoped_release release;
                Atoms atom(source);
                if (!atom.cartesian) atom.to_cartesian();
                Neighborhoods nbhs(atom, dw.desc->cutoff_);
                auto &site = nbhs[static_cast<int>(center)];
                DescriptorGradient gradient;
                dw.compute_with_gradient(site, local, gradient);
                const auto &g = gradient.grad;
                if (local.size() != d || g.dimension(0) != d ||
                    g.dimension(1) != site.nbhcount_ || g.dimension(2) != 3)
                    throw std::runtime_error("Descriptor returned an incompatible local gradient shape");

                indices.assign(site.indices_.begin(), site.indices_.end());
                indices.push_back(center);
                std::sort(indices.begin(), indices.end());
                indices.erase(std::unique(indices.begin(), indices.end()), indices.end());
                const size_t k = indices.size();
                const size_t center_slot = std::lower_bound(indices.begin(), indices.end(), center) - indices.begin();
                jacobian.assign(static_cast<size_t>(d) * k * 3, 0.0);
                for (int j = 0; j < site.nbhcount_; ++j) {
                    const size_t slot = std::lower_bound(indices.begin(), indices.end(), site.indices_[j]) - indices.begin();
                    // A self image has a fixed lattice displacement: its
                    // source and center contributions cancel identically.
                    if (slot == center_slot) continue;
                    // Each image moves with its source atom. Its relative
                    // coordinate also moves oppositely with the center.
                    for (py::ssize_t f = 0; f < d; ++f)
                        for (int a = 0; a < 3; ++a) {
                            const double value = g(f, j, a);
                            jacobian[(f * k + slot) * 3 + a] += value;
                            jacobian[(f * k + center_slot) * 3 + a] -= value;
                        }
                }
            }
            const py::ssize_t k = indices.size();
            py::array_t<std::int64_t> atom_indices(k);
            std::copy(indices.begin(), indices.end(), atom_indices.mutable_data());
            py::array_t<double> derivatives({d, k, py::ssize_t(3)});
            std::copy(jacobian.begin(), jacobian.end(), derivatives.mutable_data());
            return py::make_tuple(vector_to_numpy_copy(local), atom_indices, derivatives);
        }, py::arg("atoms"), py::arg("center"),
           "Return local values (D,), sorted source atom indices (K,) and a Cartesian "
           "position Jacobian (D, K, 3) for one center at fixed cell. "
           "Periodic images are combined and the center derivative is included.")
        .def("compute_with_strain_gradient", [](descriptor_wrapper &dw, const Atoms &source,
                                                py::ssize_t center) {
            validate_descriptor_species(dw, source);
            if (center < 0 || center >= source.size())
                throw py::index_error("Descriptor center must be an atom index in [0, N)");
            const py::ssize_t d = dw.feature_size();
            Eigen::VectorXd local;
            std::vector<double> jacobian(static_cast<size_t>(d) * 9, 0.0);
            {
                py::gil_scoped_release release;
                Atoms atom(source);
                if (!atom.cartesian) atom.to_cartesian();
                Neighborhoods nbhs(atom, dw.desc->cutoff_);
                auto &site = nbhs[static_cast<int>(center)];
                DescriptorGradient gradient;
                dw.compute_with_gradient(site, local, gradient);
                const auto &g = gradient.grad;
                if (local.size() != d || g.dimension(0) != d ||
                    g.dimension(1) != site.nbhcount_ || g.dimension(2) != 3)
                    throw std::runtime_error("Descriptor returned an incompatible local gradient shape");
                // Strain moves each relative image vector, including self images.
                // For column r'=(I+epsilon)r, dp/depsilon_ab = sum_j g_ja * r_jb.
                for (int j = 0; j < site.nbhcount_; ++j)
                    for (py::ssize_t f = 0; f < d; ++f)
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b)
                                jacobian[(f * 3 + a) * 3 + b] +=
                                    g(f, j, a) * site.pos_[j][b];
            }
            py::array_t<double> derivatives({d, py::ssize_t(3), py::ssize_t(3)});
            std::copy(jacobian.begin(), jacobian.end(), derivatives.mutable_data());
            return py::make_tuple(vector_to_numpy_copy(local), derivatives);
        }, py::arg("atoms"), py::arg("center"),
           "Return local values (D,) and homogeneous strain Jacobian (D, 3, 3). "
           "J[f,a,b] differentiates column r'=(I+epsilon)r: row positions and cell "
           "both right-multiply (I+epsilon).T. Include all neighbor images, including "
           "self images. SO4 inner-density settings retain local values. Arrays own their "
           "data and the source representation is preserved.")
        .def("__repr__", [](descriptor_wrapper &dw) {
            return "Descriptor(dtype=" + dw.dtype + ", features=" + std::to_string(dw.feature_size()) + ")";
        });
}

// ============================================================================
//  ActiveSelector (MaxVol) binding
// ============================================================================
void bind_maxvol(py::module_ &m) {
    py::class_<MaxVol::SwapStep>(m, "_MaxVolSwapStep")
        .def_readonly("ind_in", &MaxVol::SwapStep::ind_in)
        .def_readonly("ind_out", &MaxVol::SwapStep::ind_out)
        .def_readonly("gain", &MaxVol::SwapStep::gain)
        .def("__repr__", [](const MaxVol::SwapStep &s) {
            return "MaxVolSwapStep(ind_in=" + std::to_string(s.ind_in)
                 + ", ind_out=" + std::to_string(s.ind_out)
                 + ", gain=" + std::to_string(s.gain) + ")";
        });

    py::class_<MaxVol>(m, "_MaxVol")
        .def(py::init<>())
        .def("select_rows", [](MaxVol &mv, const Eigen::MatrixXd &A, int k) {
            py::gil_scoped_release release;
            return mv.select_rows(A, k);
        }, py::arg("A"), py::arg("k"))
        .def("compute_scores", [](MaxVol &mv, const Eigen::MatrixXd &A,
                                   const std::vector<int> &selected) {
            Eigen::VectorXd scores;
            {
                py::gil_scoped_release release;
                scores = mv.compute_scores(A, selected);
            }
            return vector_to_numpy_copy(scores);
        }, py::arg("A"), py::arg("selected"))
        .def("add_rows", [](MaxVol &mv, const Eigen::MatrixXd &A,
                             const std::vector<int> &current, int add_k) {
            py::gil_scoped_release release;
            return mv.add_rows(A, current, add_k);
        }, py::arg("A"), py::arg("current"), py::arg("add_k"))
        .def_property_readonly("log_determinant", &MaxVol::log_determinant)
        .def_property_readonly("swap_history", [](const MaxVol &mv) {
            return mv.swap_history();
        })
        .def_readwrite("tolerance", &MaxVol::tolerance)
        .def_readwrite("max_iterations", &MaxVol::max_iterations)
        .def_readwrite("max_swaps", &MaxVol::max_swaps)
        .def_readwrite("verbose", &MaxVol::verbose)
        .def_readwrite("sm_recompute_every", &MaxVol::sm_recompute_every);

    m.def("_build_bmatrix", [](py::sequence cfgs,
                               const std::string &potfile,
                               const std::string &pottype,
                               const std::string &dtype,
                               const std::string &mode,
                               const std::map<std::string, std::string> &params,
                               double weight,
                               double pow_scale,
                               double w_energy,
                               double w_site,
                               double w_force,
                               double w_stress,
                               bool all_coeffs) -> py::object {
        GPSettings settings;
        for (const auto &[k, v] : params) settings.Modify(k, v);
        auto builder = make_builder(dtype, potfile, pottype, settings, mode,
                                    w_energy, w_site, w_force, w_stress,
                                    all_coeffs);
        if (!builder) {
            throw std::runtime_error("failed to create B-matrix builder");
        }
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B(0, builder->coeff_count());
        for (py::handle h : cfgs) {
            Atoms &cfg = h.cast<Atoms&>();
            builder->build_rows(cfg, B, weight, pow_scale);
        }
        return matrix_to_numpy_copy(B);
    }, py::arg("structures"), py::arg("potfile"), py::arg("pottype") = "mtp",
       py::arg("dtype") = "mtm", py::arg("mode") = "nbh",
       py::arg("params") = std::map<std::string, std::string>(),
       py::arg("weight") = 1.0, py::arg("pow_scale") = 0.5,
       py::arg("w_energy") = 1.0, py::arg("w_site") = 0.0,
       py::arg("w_force") = 0.0, py::arg("w_stress") = 0.0,
       py::arg("all_coeffs") = false,
       "Build a MaxVol B-matrix from structures.");

    py::class_<ActiveSelector>(m, "_ActiveSelector")
        .def(py::init<>())
        .def(py::init([](const std::string &potfile,
                         const std::string &pottype,
                         const std::string &dtype,
                         const std::string &mode,
                         const std::map<std::string, std::string> &params,
                         double w_energy,
                         double w_site,
                         double w_force,
                         double w_stress,
                         bool all_coeffs) {
            GPSettings settings;
            for (const auto &[k, v] : params) settings.Modify(k, v);
            auto builder = make_builder(dtype, potfile, pottype, settings, mode,
                                        w_energy, w_site, w_force, w_stress,
                                        all_coeffs);
            if (!builder) {
                throw std::runtime_error("failed to create B-matrix builder");
            }
            auto selector = std::make_unique<ActiveSelector>(builder.get());
            builder.release();
            return selector;
        }), py::arg("potfile"), py::arg("pottype") = "mtp",
            py::arg("dtype") = "mtm", py::arg("mode") = "nbh",
            py::arg("params") = std::map<std::string, std::string>(),
            py::arg("w_energy") = 1.0, py::arg("w_site") = 0.0,
            py::arg("w_force") = 0.0, py::arg("w_stress") = 0.0,
            py::arg("all_coeffs") = false)

        .def("add_candidates", [](ActiveSelector &as, py::sequence cfgs) {
            std::vector<Atoms> local;
            local.reserve(py::len(cfgs));
            for (py::handle h : cfgs)
                local.push_back(h.cast<Atoms&>());
            return as.select_mode == "full"
                 ? as.add_candidates_parallel(local)
                 : as.add_candidates_unbuilt(local);
        }, py::arg("structures"), "Add candidate structures to the pool.")

        .def("clear_candidates", &ActiveSelector::clear_candidates)
        .def("num_candidates", &ActiveSelector::num_candidates)
        .def("num_rows", &ActiveSelector::num_rows)

        .def("select", [](ActiveSelector &as, double threshold) -> int {
            return as.select(threshold);
        }, py::arg("threshold") = 1.001, "Run MaxVol selection. Returns number of selected configs.")

        .def("grade", [](ActiveSelector &as, Atoms &atom) {
            return as.grade(atom);
        }, py::arg("structure"), "Compute MaxVol extrapolation grade for one structure.")

        .def("grade_all", [](ActiveSelector &as, py::sequence cfgs) {
            std::vector<Atoms> local;
            local.reserve(py::len(cfgs));
            for (py::handle h : cfgs)
                local.push_back(h.cast<Atoms&>());
            std::vector<double> values;
            as.grade_all(local, values);
            Eigen::VectorXd grades(values.size());
            for (int i = 0; i < grades.size(); ++i)
                grades[i] = values[static_cast<size_t>(i)];
            return vector_to_numpy_copy(grades);
        }, py::arg("structures"), "Batch grade structures.")

        .def("prepare_grading", &ActiveSelector::prepare_grading,
           "Prepare cached MaxVol state for repeated grading.")

        // Accessors
        .def_property_readonly("selected_indices", [](ActiveSelector &as) {
            return as.selected_indices();
        })
        .def("get_selected", [](ActiveSelector &as) {
            return as.get_selected();
        }, "Return selected structures.")

        // Persistence
        .def("save_state", [](ActiveSelector &as, const std::string &fn,
                               const std::string &idx, const std::string &pf,
                               const std::string &pt) {
            as.save_state(fn, idx, pf, pt);
        }, py::arg("filename"), py::arg("index_file") = "",
           py::arg("potfile") = "", py::arg("pottype") = "",
           "Save active set state to .sel file.")

        .def_static("load_state", [](const std::string &fn, const std::string &idx) {
            auto as = std::make_unique<ActiveSelector>();
            as->load_state(fn, idx);
            return as;
        }, py::arg("filename"), py::arg("index_file") = "",
           "Load active set from .sel file.")

        .def("init_builder", [](ActiveSelector &as, Potentials::AnyPotential &pot) {
            as.init_builder(pot);
        }, py::arg("potential"), py::keep_alive<1, 2>(),
           "Initialize B-matrix builder from a potential.")

        // Parameters
        .def_readwrite("threshold", &ActiveSelector::threshold)
        .def_readwrite("weight", &ActiveSelector::weight)
        .def_readwrite("pow_scale", &ActiveSelector::pow_scale)
        .def_readwrite("max_swaps", &ActiveSelector::max_swaps)
        .def_readwrite("select_mode", &ActiveSelector::select_mode)
        .def_readwrite("stream_mode", &ActiveSelector::stream_mode)
        .def_readwrite("stream_chunk_size", &ActiveSelector::stream_chunk_size)
        .def_readwrite("stream_fast_passes", &ActiveSelector::stream_fast_passes);
}

// ============================================================================
//  FPSSelector binding
// ============================================================================
void bind_fps(py::module_ &m) {
    py::class_<FPSSelector>(m, "_FPSSelector")
        .def(py::init<>())
        .def("add_candidates", [](FPSSelector &fps, std::vector<Atoms> &cfgs) {
            fps.add_candidates(cfgs);
        }, py::arg("structures"))
        .def("select", [](FPSSelector &fps, int n, int first_idx) {
            return fps.select(n, first_idx);
        }, py::arg("n"), py::arg("first_idx") = -1)
        .def("select_by_threshold", [](FPSSelector &fps, double thr, int max_n, int first_idx) {
            return fps.select_by_threshold(thr, max_n, first_idx);
        }, py::arg("threshold"), py::arg("max_n") = 0, py::arg("first_idx") = -1)
        .def_property_readonly("selected_indices", [](FPSSelector &fps) {
            return fps.selected_indices();
        })
        .def_property_readonly("selection_scores", [](FPSSelector &fps) {
            return fps.selection_scores();
        })
        .def("get_selected", [](FPSSelector &fps) {
            return fps.get_selected();
        }, "Return selected structures.");
}

// ============================================================================
//  CUR selection binding
// ============================================================================
void bind_cur(py::module_ &m) {
    m.def("cur_select_rows", [](const Eigen::MatrixXd &A, int k, bool det, int seed, bool comp_err) {
        auto result = cur_select_rows(A, k, det, seed, comp_err);
        py::dict d;
        d["selected_indices"] = result.selected_indices;
        d["scores"] = result.scores;
        d["reconstruction_error"] = result.reconstruction_error;
        return d;
    }, py::arg("A"), py::arg("k"),
       py::arg("deterministic") = false, py::arg("seed") = 42,
       py::arg("compute_error") = false,
       "CUR row selection via leverage scores.");
}

// ============================================================================
//  AL Scorer / ALPotential binding
// ============================================================================
void bind_al_potential(py::module_ &m) {
    py::class_<AL::ALPotential>(m, "_ALPotential")
        .def_static("create", [](const std::map<std::string, std::string> &settings) {
            GPSettings s;
            for (const auto &[k, v] : settings) s.Modify(k, v);
            return AL::ALPotential::create(s);
        }, py::arg("settings"), "Create AL potential from settings dict. Returns None if disabled.")
        .def("init_builder", &AL::ALPotential::init_builder,
             py::arg("potential"), py::keep_alive<1, 2>(),
             "Initialize MaxVol scoring with a potential before evaluation.")
        .def("save",&AL::ALPotential::save)
        .def_property_readonly("selector",&AL::ALPotential::maxvol_selector,py::return_value_policy::reference_internal)
        .def_property_readonly("should_abort",&AL::ALPotential::should_abort)
        .def_property_readonly("pool",&AL::ALPotential::pool)
        .def("active", &AL::ALPotential::active)
        .def("evaluate", [](AL::ALPotential &al, Atoms &atom) {
            auto score = al.evaluate(atom);
            py::dict d;
            d["uncertainty"] = score.uncertainty;
            return d;
        })
        .def("is_uncertain", &AL::ALPotential::is_uncertain)
        .def("check", &AL::ALPotential::check)
        .def_property_readonly("pool_size", &AL::ALPotential::pool_size)
        .def("finalize", [](AL::ALPotential &al, std::optional<std::string> file) {
            if (file) al.finalize(*file);
            else al.finalize();
        }, py::arg("export_file") = py::none(),
           "Export with configured sparsification and an optional output path override.");
}

static Potentials::ModelBundle::Profile gpm_profile(const std::string &profile) {
    using P=Potentials::ModelBundle::Profile;
    if(profile=="preserve")return P::Preserve;
    if(profile=="inference")return P::Inference;
    if(profile=="resume")return P::Resume;
    throw py::value_error("profile must be preserve, inference, or resume");
}
void bind_model_bundle(py::module_ &m) {
    using Bundle=Potentials::ModelBundle;
    m.def("_save_gpm",[](Potentials::AnyPotential &p,const std::string &file,const std::string &profile) {
        Bundle::save_potential(p,file,gpm_profile(profile));
    },py::arg("potential"),py::arg("filename"),py::arg("profile")="preserve");
    py::class_<Bundle,std::shared_ptr<Bundle>>(m,"_ModelBundle")
        .def_static("load",&Bundle::load)
        .def_static("from_settings",[](const std::map<std::string,std::string>&values){GPSettings s;for(auto &[k,v]:values)s.Modify(k,v);return Bundle::from_settings(s);})
        .def_static("inspect",[](const std::string &file){return py::module_::import("json").attr("loads")(GPM::Value(Bundle::inspect(file)).serialize());})
        .def_static("validate",&Bundle::validate)
        .def("save",[](Bundle &b,const std::string &file,const std::string &profile){b.save(file,gpm_profile(profile));},py::arg("filename"),py::arg("profile")="preserve")
        .def_property_readonly("potential",&Bundle::potential,py::return_value_policy::reference_internal)
        .def_property_readonly("pottype",[](Bundle&b){return Bundle::type(b.potential());})
        .def("create_al",[](Bundle&b,const std::map<std::string,std::string>&values){GPSettings s;for(auto &[k,v]:values)s.Modify(k,v);return b.create_al(s);},py::arg("overrides")=std::map<std::string,std::string>{})
        .def("attach_al",&Bundle::attach_al)
        .def("attach_selector",&Bundle::attach_selector);
}

// ============================================================================
//  Module registration
// ============================================================================
PYBIND11_MODULE(_imrmlp, m) {
    m.doc() = "imrmlp C++ bindings for IMR-MLP — MLIP simulation framework";

    // Register bindings
    bind_atoms(m);
    bind_unit_conversion(m);
    bind_xsf_fields(m);
    bind_potential(m);
    bind_settings(m);
    bind_io(m);
    bind_relax(m);
    bind_neb(m);
    bind_dimer(m);
    bind_train(m);
    bind_descriptors(m);
    bind_maxvol(m);
    bind_fps(m);
    bind_cur(m);
    bind_al_potential(m);
    bind_model_bundle(m);

    // ── OpenMP helpers ──
    m.def("set_num_threads", &set_omp_num_threads, py::arg("n"),
          "Request a positive OpenMP team size for subsequent calculations");
    m.def("get_num_threads", &get_omp_max_threads,
          "Get this thread's requested OpenMP team size");

    // ── Version ──
    m.attr("__version__") = "1.2.4";

#if IMRMLP_MPI
    m.attr("has_mpi") = true;
#else
    m.attr("has_mpi") = false;
#endif
}
