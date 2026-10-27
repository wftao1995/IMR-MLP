// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpio.h"
#include "hvp_io.h"
#include "io/hessian_io.h"
#include "gpio_binary.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <string>
#include <cmath>
#include "tools/chem.h"
#include "tools/procstr.h"
#include "tools/tools.h"
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>

// Compute lower-triangular cell matrix from an arbitrary cell, without
// the position round-trip that Atoms::to_lower_standard() performs.
// TODO: move or merge with Atoms::to_lower_standard() after refactor.
static Eigen::Matrix3d cell_to_lower_tri(const Eigen::Matrix3d &cell) {
    double a = cell.row(0).norm();
    double b = cell.row(1).norm();
    double c = cell.row(2).norm();
    if (a <= 0 || b <= 0 || c <= 0)
        return cell; // degenerate, return as-is
    double cos_beta = std::clamp(cell.row(2).dot(cell.row(0)) / (c * a), -1.0, 1.0);
    double cos_gamma = std::clamp(cell.row(0).dot(cell.row(1)) / (a * b), -1.0, 1.0);
    double cos_alpha = std::clamp(cell.row(1).dot(cell.row(2)) / (b * c), -1.0, 1.0);
    double sin_gamma = std::sin(std::acos(cos_gamma));
    if (sin_gamma < 1e-12)
        return cell; // degenerate, return as-is
    Eigen::Matrix3d lower;
    lower.row(0) << a, 0, 0;
    lower.row(1) << b * cos_gamma, b * sin_gamma, 0;
    double cx = c * cos_beta;
    double cy = c * (cos_alpha - cos_beta * cos_gamma) / sin_gamma;
    double cz_sq = c * c - cx * cx - cy * cy;
    lower.row(2) << cx, cy, std::sqrt(std::max(0.0, cz_sq));
    return lower;
}

void GPIO::write_next_mlpcfg_bin(std::ostream &ofs, Atoms &atom) {
    HessianIO::validate(atom);
    const std::string hessian=atom.have_hessian?HessianIO::write(atom):std::string{};
    if (hessian.size()>size_t(gpio_binary::MAX_STRING_SIZE)) {
        std::cerr << "Error: BIN full Hessian exceeds the existing named Feature string limit\n";
        ofs.setstate(std::ios::failbit); return;
    }
    using namespace gpio_binary;
    const int natoms = atom.size();

    if (!write_header(ofs, natoms)) return;

    write_tag(ofs, Tag::Cell);
    write_span(ofs, atom.cell_.data(), 9);

    write_tag(ofs, Tag::Positions);
    write_span(ofs, atom.pos_.data(), atom.pos_.size());

    write_tag(ofs, Tag::Types);
    write_span(ofs, atom.chemmap_.data(), atom.chemmap_.size());

    write_tag(ofs, Tag::PBC);
    const std::array<std::uint8_t,3> pbc_flags{{
        static_cast<std::uint8_t>(atom.pbc_[0]),
        static_cast<std::uint8_t>(atom.pbc_[1]),
        static_cast<std::uint8_t>(atom.pbc_[2])}};
    write_span(ofs,pbc_flags.data(),pbc_flags.size());

    if (atom.forces_.size() == static_cast<size_t>(natoms)) {
        write_tag(ofs, Tag::Forces);
        write_span(ofs, atom.forces_.data(), atom.forces_.size());
    }
    if (atom.has_fenergy()) {
        write_tag(ofs, Tag::FreeEnergy);
        write_value(ofs, atom.fenergy_);
    }
    if (atom.site_energies_.size() == static_cast<size_t>(natoms)) {
        write_tag(ofs, Tag::SiteEnergies);
        write_span(ofs, atom.site_energies_.data(), atom.site_energies_.size());
    }
    if (atom.magmoms_.size() == static_cast<size_t>(natoms)) {
        write_tag(ofs, Tag::MagmomVector);
        write_span(ofs, atom.magmoms_.data(), atom.magmoms_.size());
    }
    if (atom.magmom_scalar_.size() == static_cast<size_t>(natoms)) {
        write_tag(ofs, Tag::MagmomScalar);
        write_span(ofs, atom.magmom_scalar_.data(), atom.magmom_scalar_.size());
    }
    if (atom.charges_.size() == static_cast<size_t>(natoms)) {
        write_tag(ofs, Tag::Charges);
        write_span(ofs, atom.charges_.data(), atom.charges_.size());
    }
    if (atom.velocities_.size() == static_cast<size_t>(natoms)) {
        write_tag(ofs, Tag::Velocities);
        write_span(ofs, atom.velocities_.data(), atom.velocities_.size());
    }
    if (atom.ids_.size() == static_cast<size_t>(natoms)) {
        write_tag(ofs, Tag::Ids);
        write_span(ofs, atom.ids_.data(), atom.ids_.size());
    }
    if (atom.has_energy()) {
        write_tag(ofs, Tag::Energy);
        write_value(ofs, atom.energy_);
    }
    if (atom.has_stresses()) {
        write_tag(ofs, Tag::Stress);
        write_span(ofs, atom.stress_.data(), 9);
    }
    atom.validate_hvp();
    if (atom.has_hvp()) {
        write_tag(ofs, Tag::Feature);
        write_named_string(ofs, "hvp_directions", HVPIO::write(atom.hvp_directions_));
        write_tag(ofs, Tag::Feature);
        write_named_string(ofs, "hvp_targets", HVPIO::write(atom.hvp_targets_));
    }
    if (atom.have_hessian) {
        write_tag(ofs, Tag::Feature);
        write_named_string(ofs,"hessian",hessian);
    }
    for (const auto &feature : atom.features_) {
        if (feature.first=="hessian" && atom.have_hessian) continue;
        write_tag(ofs, Tag::Feature);
        write_named_string(ofs, feature.first, feature.second);
    }
    for (const auto &nm : atom.local_attribs_d_) {
        if (nm.second.size() != natoms) {
            ofs.setstate(std::ios::failbit);
            return;
        }
        write_tag(ofs, Tag::LocalDouble);
        write_string(ofs, nm.first);
        write_span(ofs, nm.second.data(), nm.second.size());
    }
    for (const auto &nm : atom.local_attribs_v_) {
        if (nm.second.cols() != natoms || nm.second.rows() > MAX_VECTOR_SIZE ||
            nm.second.size() > MAX_VECTOR_SIZE) {
            ofs.setstate(std::ios::failbit);
            return;
        }
        write_tag(ofs, Tag::LocalDoubleVector);
        write_string(ofs, nm.first);
        write_value(ofs, static_cast<std::int32_t>(nm.second.rows()));
        write_span(ofs, nm.second.data(), nm.second.size());
    }
    for (const auto &nm : atom.global_attribs_d_) {
        write_tag(ofs, Tag::GlobalDouble);
        write_string(ofs, nm.first);
        write_value(ofs, nm.second);
    }
    for (const auto &nm : atom.global_attribs_v_) {
        write_tag(ofs, Tag::GlobalDoubleVector);
        write_string(ofs, nm.first);
        write_vector(ofs, nm.second);
    }
    for (const auto &nm : atom.global_attribs_i_) {
        write_tag(ofs, Tag::GlobalInt);
        write_string(ofs, nm.first);
        write_value(ofs, nm.second);
    }
    for (const auto &nm : atom.global_attribs_vi_) {
        write_tag(ofs, Tag::GlobalIntVector);
        write_string(ofs, nm.first);
        write_vector(ofs, nm.second);
    }
    for (const auto &nm : atom.local_attribs_i_) {
        if (nm.second.size() != natoms) {
            ofs.setstate(std::ios::failbit);
            return;
        }
        write_tag(ofs, Tag::LocalInt);
        write_string(ofs, nm.first);
        write_span(ofs, nm.second.data(), nm.second.size());
    }
    for (const auto &nm : atom.local_attribs_vi_) {
        if (nm.second.cols() != natoms || nm.second.rows() > MAX_VECTOR_SIZE ||
            nm.second.size() > MAX_VECTOR_SIZE) {
            ofs.setstate(std::ios::failbit);
            return;
        }
        write_tag(ofs, Tag::LocalIntVector);
        write_string(ofs, nm.first);
        write_value(ofs, static_cast<std::int32_t>(nm.second.rows()));
        write_span(ofs, nm.second.data(), nm.second.size());
    }

    write_tag(ofs, Tag::End);
}

void GPIO::write_next_mlpcfg(std::ostream &ofs, Atoms &atom) {
    HessianIO::validate(atom);
    if (cfg_have_efs && (!atom.has_energy() || !atom.has_forces() || !atom.has_stresses())) {
        std::cerr << "Error: forced EFS output requires energy, forces and stress\n";
        ofs.setstate(std::ios::failbit);
        return;
    }
    if (!atom.magmom_scalar_.empty()) {
        std::cerr << "Error: CFG output does not support scalar magnetic moments; use BIN to retain them\n";
        ofs.setstate(std::ios::failbit);
        return;
    }
    static const std::set<std::string> reserved_atom_columns{
        "id", "type", "cartes_x", "cartes_y", "cartes_z", "site_en",
        "fx", "fy", "fz", "magmom_x", "magmom_y", "magmom_z",
        "q", "vx", "vy", "vz", "x", "y", "z"
    };
    for (const auto &[name, values] : atom.local_attribs_d_) {
        if (values.size() != atom.size() || reserved_atom_columns.count(name)) {
            ofs.setstate(std::ios::failbit);
            return;
        }
    }
    if ((atom.has_charges() && atom.charges_.size() != atom.size()) ||
        (atom.has_velo() && atom.velocities_.size() != atom.size()) ||
        (!atom.magmoms_.empty() && atom.magmoms_.size() != atom.size())) {
        ofs.setstate(std::ios::failbit);
        return;
    }
    fmt::print(ofs, "BEGIN_CFG\nSize\n{}\nSupercell\n", atom.size());
    const Eigen::Matrix3d &cell = atom.cell_;
    for (int ii = 0; ii < 3; ++ii) {
        if (lossless)
            fmt::print(ofs, "{} {} {}\n", cell(ii, 0), cell(ii, 1), cell(ii, 2));
        else 
            fmt::print(ofs, "{:>15.8g} {:>15.8g} {:>15.8g}\n", cell(ii, 0), cell(ii, 1), cell(ii, 2));
    }
    fmt::print(ofs, "AtomData: id type  cartes_x cartes_y cartes_z");
    if (atom.has_site_energies())
        fmt::print(ofs, " site_en");
    if (atom.has_forces() || cfg_have_efs)
        fmt::print(ofs, " fx fy fz");
    if (atom.has_magmom())
        fmt::print(ofs, " magmom_x magmom_y magmom_z");
    if (atom.has_charges())
        fmt::print(ofs, " q");
    if (atom.has_velo())
        fmt::print(ofs, " vx vy vz");
    for (const auto &[name, values] : atom.local_attribs_d_) {
        (void)values;
        fmt::print(ofs, " {}", name);
    }
    fmt::print(ofs, "\n");
    if (lossless) {
        for (int j = 0; j < atom.size(); ++j) {
            fmt::print(ofs, "{} {} {}", j+1, atom.chemmap_[j], fmt::join(atom.pos_[j], " "));
            if (atom.has_site_energies())
                fmt::print(ofs, " {}", atom.site_energies_[j]);
            if (atom.has_forces())
            fmt::print(ofs, " {}", fmt::join(atom.forces_[j], " "));
            if (atom.has_magmom())
                fmt::print(ofs, " {}", fmt::join(atom.magmoms_[j], " "));
            if (atom.has_charges())
                fmt::print(ofs, " {}", atom.charges_[j]);
            if (atom.has_velo())
                fmt::print(ofs, " {}", fmt::join(atom.velocities_[j], " "));
            for (const auto &[name, values] : atom.local_attribs_d_) {
                (void)name;
                fmt::print(ofs, " {}", values[j]);
            }
            fmt::print(ofs, "\n");
        }
        if (atom.has_energy() || cfg_have_efs)
            fmt::print(ofs, "Energy\n{}\n", atom.energy_);
        if (atom.has_stresses() || cfg_have_efs)
            fmt::print(ofs, "PlusStress: xx yy zz yz xz xy\n{} {} {} {} {} {}\n", 
                                atom.stress_(0, 0), atom.stress_(1, 1), atom.stress_(2, 2), 
                                atom.stress_(1, 2), atom.stress_(0, 2), atom.stress_(0, 1));
    } else {
        for (int j = 0; j < atom.size(); ++j) {
            fmt::print(ofs, " {:<6} {:<6} {:>15.8g}", j+1, atom.chemmap_[j], fmt::join(atom.pos_[j], " "));
            if (atom.has_site_energies())
                fmt::print(ofs, " {:>15.8g}", atom.site_energies_[j]);
            if (atom.has_forces())
            fmt::print(ofs, " {:>15.8g}", fmt::join(atom.forces_[j], " "));
            if (atom.has_magmom())
                fmt::print(ofs, " {:>15.8g}", fmt::join(atom.magmoms_[j], " "));
            if (atom.has_charges())
                fmt::print(ofs, " {:>15.8g}", atom.charges_[j]);
            if (atom.has_velo())
                fmt::print(ofs, " {:>15.8g}", fmt::join(atom.velocities_[j], " "));
            for (const auto &[name, values] : atom.local_attribs_d_) {
                (void)name;
                fmt::print(ofs, " {:>15.8g}", values[j]);
            }
            fmt::print(ofs, "\n");
        }
        if (atom.has_energy() || cfg_have_efs)
            fmt::print(ofs, "Energy\n {:>15.8g}\n", atom.energy_);
        if (atom.has_stresses() || cfg_have_efs)
            fmt::print(ofs, "PlusStress: xx yy zz yz xz xy\n {:>15.8g} {:>15.8g} {:>15.8g} {:>15.8g} {:>15.8g} {:>15.8g}\n", 
                                atom.stress_(0, 0), atom.stress_(1, 1), atom.stress_(2, 2), 
                                atom.stress_(1, 2), atom.stress_(0, 2), atom.stress_(0, 1));
    }
    
    const auto fitness = atom.global_attribs_d_.find("fitness");
    if (fitness != atom.global_attribs_d_.end())
        fmt::print(ofs, "Feature Fitness {}\n", fitness->second);
    const auto total_charge = atom.global_attribs_d_.find("total_charge");
    if (total_charge != atom.global_attribs_d_.end())
        fmt::print(ofs, "Feature total_charge {}\n", total_charge->second);
    if (atom.has_fenergy())
        fmt::print(ofs, "Feature fenergy {}\n", atom.fenergy_);
    if (atom.features_.size() > 0)
        for (auto &it : atom.features_) {
            if (it.first=="hessian" && atom.have_hessian) continue;
            if (it.first == "Fitness" && fitness != atom.global_attribs_d_.end())
                continue;
            if (it.first == "total_charge" && total_charge != atom.global_attribs_d_.end())
                continue;
            if (it.first == "fenergy" && atom.has_fenergy())
                continue;
            fmt::print(ofs, "Feature {} {}\n", it.first, it.second);
        }
    // if (atom.mvgrade_ > 0) {
    // }
    atom.validate_hvp();
    if (atom.has_hvp()) {
        fmt::print(ofs, "Feature hvp_directions {}\n", HVPIO::write(atom.hvp_directions_));
        fmt::print(ofs, "Feature hvp_targets {}\n", HVPIO::write(atom.hvp_targets_));
    }
    if (atom.have_hessian)
        fmt::print(ofs,"Feature hessian {}\n",HessianIO::write(atom));
    fmt::print(ofs, "END_CFG\n\n");
}

bool GPIO::write_mlpcfg_sequence(std::ostream &ofs, Atoms &atom) {
    write_next_mlpcfg(ofs, atom);
    return static_cast<bool>(ofs);
}

bool GPIO::write_mlpcfg_bin_sequence(std::ostream &ofs, Atoms &atom) {
    write_next_mlpcfg_bin(ofs, atom);
    return static_cast<bool>(ofs);
}
/*****
 * write lammps trajectory file
 * *****/
void GPIO::write_next_lammps_trj(std::ostream &ofs, Atoms &atom)
{
    fmt::print(ofs, "ITEM: TIMESTEP\n{}\nITEM: NUMBER OF ATOMS\n{}\n", ncfgs_++, atom.size());
    Eigen::Matrix3d cell = atom.cell_;
    if (atom.isorth() && !force_tilt)
    {
        fmt::print(ofs, "ITEM: BOX BOUNDS pp pp pp\n");
        fmt::print(ofs, "{0:.8f} {1:.8f}\n{0:.8f} {2:.8f}\n{0:.8f} {3:.8f}\n", 0.00,
                        cell(0,0), cell(1,1), cell(2,2));
    }
    else
    {
        fmt::print(ofs, "ITEM: BOX BOUNDS xy xz yz pp pp pp\n");
        cell = cell_to_lower_tri(atom.cell_);
        const double lx = cell(0, 0);
        const double xy = cell(1, 0);
        const double ly = cell(1, 1);
        const double xz = cell(2, 0);
        const double yz = cell(2, 1);
        const double lz = cell(2, 2);
        // LAMMPS dumps store bounding limits, not the true restricted-box
        // limits used by data files.
        const double xlo_bound = std::min({0.0, xy, xz, xy + xz});
        const double xhi_bound = lx + std::max({0.0, xy, xz, xy + xz});
        const double ylo_bound = std::min(0.0, yz);
        const double yhi_bound = ly + std::max(0.0, yz);
        fmt::print(ofs, "{} {} {}\n", xlo_bound, xhi_bound, xy);
        fmt::print(ofs, "{} {} {}\n", ylo_bound, yhi_bound, xz);
        fmt::print(ofs, "{} {} {}\n", 0, lz, yz);
    }
    // writer headers
    fmt::print(ofs, "ITEM: ATOMS id type x y z ");
    if (atom.has_forces())
        ofs << "fx fy fz ";
    if (atom.has_velo())
        ofs << "vx vy vz ";
    // if (atom.has_magmom())
    //     ofs << "c_mx c_my c_mz ";
    // local attributes:
    if (atom.has_site_energies())
        ofs << "site_en ";
    for (auto &it : atom.local_attribs_d_) {
        ofs << it.first << " ";
    }
    for (auto &it : atom.local_attribs_v_) {
        for (int ii = 0; ii < it.second.rows(); ++ii) {
            // LAMMPS vector columns are 1-based: desc[1], desc[2], ...
            // (must match the reader in gpio_lmp.h, which rejects index 0)
            fmt::print(ofs, "{0}[{1}] ", it.first, ii + 1);
        }
    }
    fmt::print(ofs, "\n");
    for (int i = 0; i < atom.size(); ++i)
    {
        ofs << i + 1 << " " << atom.chemmap_[i] + 1 << " ";
        // x y z
        fmt::print(ofs, "{} ", fmt::join(atom.pos_[i], " "));
        // force, velo, etc
        if (atom.has_forces())
            fmt::print(ofs, "{} ", fmt::join(atom.forces_[i], " "));
        if (atom.has_velo())
            fmt::print(ofs, "{} ", fmt::join(atom.velocities_[i], " "));
        if (atom.has_site_energies())
            ofs << atom.site_energies_[i] << " ";
        for (auto &it : atom.local_attribs_d_) {
            ofs << it.second[i] << " ";
        }
        for (auto &it : atom.local_attribs_v_) {
            for (int ii = 0; ii < it.second.rows(); ++ii) {
                ofs << it.second(ii, i) << " ";
            }
        }
        fmt::print(ofs, "\n");
    }
}

bool GPIO::write_vtk(std::ostream &ofs, Atoms &atom) {
    int natoms = atom.size();
    ofs << "# vtk DataFile Version 4.2\n";
    ofs << "Generated by GPIO\n";
    ofs << "ASCII\n";
    ofs << "DATASET POLYDATA\n";
    ofs << "POINTS " << natoms << " float\n";
    // 3 atoms each line
    ofs << std::fixed << std::setprecision(6);
    for (int i = 0; i < natoms; ++i)
    {
        ofs << atom.pos_[i](0) << " "
            << atom.pos_[i](1) << " "
            << atom.pos_[i](2) << " ";
        if (i % 3 == 2)
            fmt::print(ofs, "\n");
    }
    if (natoms % 3 != 0)
        fmt::print(ofs, "\n");
    fmt::print(ofs, "\n");
    ofs << "VERTICES " << natoms << " " << 2 * natoms << '\n';
    for (int i = 0; i < natoms; ++i)
    {
        ofs << "1 " << i << '\n';
    }
    fmt::print(ofs, "\n");
    ofs << "POINT_DATA " << natoms << '\n';
    int filedsize = 2;
    if (atom.has_magmom())
        filedsize++;
    if (atom.has_forces())
        filedsize++;
    ofs << "FIELD FieldData " << filedsize << '\n';
    // output id
    ofs << "id " << 1 << " " << natoms << " int\n";
    // 9 ids in a line
    for (int i = 0; i < natoms; ++i)
    {
        ofs << i + 1 << " ";
        if (i % 9 == 8)
            fmt::print(ofs, "\n");
    }
    if (natoms % 9 != 0)
        fmt::print(ofs, "\n");
    // type info
    ofs << "type " << 1 << " " << natoms << " int\n";
    for (int i = 0; i < natoms; ++i)
    {
        ofs << atom.chemmap_[i] << " ";
        if (i % 9 == 8)
            fmt::print(ofs, "\n");
    }
    if (natoms % 9 != 0)
        fmt::print(ofs, "\n");
    // magmom scalar colinear
    if (atom.has_magmom())
    {
        ofs << "mag " << 1 << " " << natoms << " double\n";
        for (int i = 0; i < natoms; ++i)
        {
            ofs << atom.magmoms_[i](0) << " ";
            if (i % 9 == 8)
                fmt::print(ofs, "\n");
        }
        if (natoms % 9 != 0)
            fmt::print(ofs, "\n");
    }
    // force data as field data
    if (atom.has_forces())
    {
        fmt::print(ofs, "f 3 {} double\n", natoms);
        for (int i = 0; i < natoms; ++i)
        {
            fmt::print(ofs, "{} ", fmt::join(atom.forces_[i], " "));
            if (i % 3 == 2)
                fmt::print(ofs, "\n");
        }
    }
    fmt::print(ofs, "\n");
    return true;
}

void GPIO::write_vasp_xdatcar_header(std::ostream &ofs, Atoms &atom) {
    build_remap(atom);
    ofs << "XDATCAR by GPIO\n";
    ofs << " 1.000000" << '\n';
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            ofs << std::fixed << std::setw(15) << std::setprecision(10) << atom.cell_(i, j);
        }
        fmt::print(ofs, "\n");
    };
    ofs << "  ";
    for (int i = 0; i < chem.size(); ++i)
    {
        ofs << std::setw(4) << mapidname[chem[i]];
    }
    ofs << '\n'
        << "  ";
    for (int i = 0; i < chem.size(); ++i)
    {
        ofs << std::setw(8) << chemcounts[i];
    }
    fmt::print(ofs, "\n");
}

bool GPIO::write_vasp_xdatcar_sequence(std::ostream &ofs, Atoms &atom) {
    if (!xdatcar_output_header_)
    {
        write_vasp_xdatcar_header(ofs, atom);
        seq_writeheader = true;
        xdatcar_output_cell_ = atom.cell_;
        xdatcar_output_species_ = chem;
        xdatcar_output_counts_ = chemcounts;
        xdatcar_output_header_ = true;
    } else {
        // Both the input order and the reader's shared remap may have changed.
        build_remap(atom);
        const double cell_tolerance = 1e-10 + 1e-12 * xdatcar_output_cell_.cwiseAbs().maxCoeff();
        if ((atom.cell_ - xdatcar_output_cell_).cwiseAbs().maxCoeff() > cell_tolerance ||
            chem != xdatcar_output_species_ || chemcounts != xdatcar_output_counts_) {
            std::cerr << "Error: fixed XDATCAR requires the same cell and species counts; use xdatcar6 for changing frames\n";
            return false;
        }
    }
    return write_vasp_xdatcar_positions(ofs, atom, xdatcar_output_cell_);
}

bool GPIO::write_vasp_xdatcar_positions(std::ostream &ofs, Atoms &atom,
                                       const Eigen::Matrix3d &cell) {
    const Eigen::Matrix3d inverse = cell.inverse();
    Eigen::RowVector3d pos;
    ofs << "Direct configuration=    " << atom.id << '\n';
    for (int i : remapid)
    {
        pos = atom.cartesian ? atom.pos_[i] : (atom.pos_[i] * atom.cell_).eval();
        pos = pos * inverse;
        ofs << std::fixed << " " << std::setprecision(10) << pos(0)
            << std::setw(15) << std::setprecision(10) << pos(1)
            << std::setw(15) << std::setprecision(10) << pos(2) << '\n';
    }
    return static_cast<bool>(ofs);
}

void GPIO::write_next_xdatcar6(std::ostream &ofs, Atoms &atom) {
    write_vasp_xdatcar_header(ofs, atom);
    seq_writeheader = true;
    write_vasp_xdatcar_positions(ofs, atom, atom.cell_);
}

bool GPIO::write_vasp_xdatcar6_sequence(std::ostream &ofs, Atoms &atom) {
    write_next_xdatcar6(ofs, atom);
    return static_cast<bool>(ofs);
}

/*****
 * Write lammps datafile
 *
 * For single atom object
 *
 * TODO: non-orthongal box wrap
 *
 * *****/
bool GPIO::write_lammps_dat_full(std::ostream &ofs, Atoms &atom) {
    ofs << std::fixed;
    ofs << "LAMMPS data file generated by GPIO\n";
    ofs << atom.size() << " atoms\n";
    ofs << atom.bonds_.size() << " bonds\n";
    ofs << atom.angles_.size() << " angles\n";
    ofs << "0 dihedrals\n0 impropers\n";
    ofs << std::set<int>(atom.chemmap_.begin(), atom.chemmap_.end()).size() << " atom types\n";
    // TODO : more about this
    ofs << "1 bond types\n1 angle types\n0 dihedral types\n0 improper types\n";

    Eigen::Matrix3d cell = atom.cell_;
    if (atom.isorth())
    {
        ofs << std::fixed << "0.00000000000 " << std::setprecision(10) << std::setw(15) << cell(0, 0) << "  xlo "
            << " xhi " << '\n';
        ofs << "0.00000000000 " << std::setprecision(10) << std::setw(15) << cell(1, 1) << "  ylo "
            << " yhi " << '\n';
        ofs << std::fixed << "0.00000000000 " << std::setprecision(10) << std::setw(15) << cell(2, 2) << "  zlo "
            << " zhi " << '\n';
        if (force_tilt)
            ofs << std::fixed << "0.00000000000 0.00000000000 0.00000000000 xy xz yz\n";
        fmt::print(ofs, "\n");
    }
    else
    {
        cell = cell_to_lower_tri(atom.cell_);
        const double lx = cell(0, 0);
        const double xy = cell(1, 0);
        const double ly = cell(1, 1);
        const double xz = cell(2, 0);
        const double yz = cell(2, 1);
        const double lz = cell(2, 2);
        ofs << std::fixed << "0.0000000 " << std::setprecision(10) << std::setw(15) << lx << "  xlo "
            << " xhi " << '\n';
        ofs << std::fixed << "0.0000000 " << std::setprecision(10) << std::setw(15) << ly << "  ylo "
            << " yhi " << '\n';
        ofs << std::fixed << "0.0000000 " << std::setprecision(10) << std::setw(15) << lz << "  zlo "
            << " zhi " << '\n';
        ofs << std::fixed << std::setprecision(10) << xy << std::setprecision(10) << std::setw(15) << xz << std::setprecision(10) << std::setw(15) << yz << " xy xz yz\n"
            << '\n';
        fmt::print(ofs, "\n");
    }
    if (write_mass) {
        fmt::print(ofs, "Masses\n\n");
        // mass table is indexed by atomic number, not by map row order;
        // Look up the element's atomic number from its name.
        for (const auto &[tid, name] : mapidname) {
            fmt::print(ofs, "        {} {}  # {}\n", tid+1, Chemicals::atomic_masses[Chemicals::atomic_numbers[name]], name);
        }
    }
    /* code here */
    ofs << "Atoms\n"
        << '\n';
    for (int i = 0; i < atom.size(); ++i)
    {
        ofs << std::fixed << std::setw(8) << i + 1;
        ofs << std::setw(8) << atom.molid_[i] + 1;
        ofs << std::setw(6) << atom.chemmap_[i] + 1;
        ofs << std::setw(20) << atom.charges_[i];
        ofs << std::setprecision(10) << std::setw(20) << atom.pos_[i][0]
            << std::setprecision(10) << std::setw(20) << atom.pos_[i][1]
            << std::setprecision(10) << std::setw(20) << atom.pos_[i][2] << '\n';
    }

    // bonds
    ofs << "\nBonds\n\n";
    for (int i = 0; i < atom.bonds_.size(); ++i)
    {
        ofs << i + 1 << std::setw(8) << 1 << std::setw(8) << atom.bonds_[i][0] + 1
            << std::setw(8) << atom.bonds_[i][1] + 1 << '\n';
    }
    // angles
    ofs << "\nAngles\n\n";
    for (int i = 0; i < atom.angles_.size(); ++i)
    {
        ofs << i + 1 << std::setw(8) << 1 << std::setw(8) << atom.angles_[i][0] + 1
            << std::setw(8) << atom.angles_[i][1] + 1
            << std::setw(8) << atom.angles_[i][2] + 1 << '\n';
    }

    return true;
}
/**
 * @brief Write lammps data file, like atomsk's output
 * 
 * @param ofs 
 * @param atom 
 * @return true 
 * @return false 
 */
bool GPIO::write_lammps_dat(std::ostream &ofs, Atoms &atom) {
    // ofs << std::fixed;
    fmt::print(ofs, "# Generated by GPIO\n");
    fmt::print(ofs, "{} atoms\n", atom.size());
    fmt::print(ofs, "{} atom types\n", std::set<int>(atom.chemmap_.begin(), atom.chemmap_.end()).size());
    Eigen::Matrix3d cell = atom.cell_;
    if (atom.isorth())
    {
        fmt::print(ofs, "    {} {} xlo xhi\n", 0, cell(0, 0));
        fmt::print(ofs, "    {} {} ylo yhi\n", 0, cell(1, 1));
        fmt::print(ofs, "    {} {} zlo zhi\n", 0, cell(2, 2));
        if (force_tilt) {
            fmt::print(ofs, "    {} {} {} xy xz yz\n", 0, 0, 0);
        }
        fmt::print(ofs, "\n");
    }
    else
    {
        cell = cell_to_lower_tri(atom.cell_);
        const double lx = cell(0, 0);
        const double xy = cell(1, 0);
        const double ly = cell(1, 1);
        const double xz = cell(2, 0);
        const double yz = cell(2, 1);
        const double lz = cell(2, 2);
        fmt::print(ofs, "    {} {} xlo xhi\n", 0, lx);
        fmt::print(ofs, "    {} {} ylo yhi\n", 0, ly);
        fmt::print(ofs, "    {} {} zlo zhi\n", 0, lz);
        fmt::print(ofs, "    {} {} {} xy xz yz\n", xy, xz, yz);
        fmt::print(ofs, "\n");
    }
    if (write_mass) {
        fmt::print(ofs, "Masses\n\n");
        // mass table is indexed by atomic number, not by map row order;
        // Look up the element's atomic number from its name.
        for (const auto &[tid, name] : mapidname) {
            fmt::print(ofs, "        {} {}  # {}\n", tid+1, Chemicals::atomic_masses[Chemicals::atomic_numbers[name]], name);
        }
    }
    fmt::print(ofs, "\nAtoms # atomic\n\n");
    for (int i = 0; i < atom.size(); ++i)
    {
        if (atom.has_charges()) {
            fmt::print(ofs, "{:8} {:6} {:8} {:20} {:20} {:20}\n", i+1, atom.chemmap_[i]+1, atom.charges_[i],
                atom.pos_[i][0], atom.pos_[i][1], atom.pos_[i][2]);
        } else {
            fmt::print(ofs, "{:8} {:6} {:20} {:20} {:20}\n", i+1, atom.chemmap_[i]+1,
                atom.pos_[i][0], atom.pos_[i][1], atom.pos_[i][2]);
        }
    }
    return true;
}

bool GPIO::write_vasp_poscar(std::ostream &ofs, Atoms &atom) {
    ofs << header_line << '\n';
    ofs << "  " << 1.000 << '\n';
    for (int i = 0; i < 3; ++i)
    {
        if (lossless)
            fmt::print(ofs, "{} {} {}\n", atom.cell_(i, 0), atom.cell_(i, 1), atom.cell_(i, 2));
        else 
            fmt::print(ofs, "{:>16.8f} {:>16.8f} {:>16.8f}\n", atom.cell_(i, 0), atom.cell_(i, 1), atom.cell_(i, 2));
    }
    // types
    if (packatoms)
        build_remap(atom);
    ofs << "  ";
    if (packatoms) {
        for (int i = 0; i < chem.size(); ++i)
            fmt::print(ofs, "{:<4}", mapidname[chem[i]]);
    } else {
        int tpch = atom.chemmap_[0];
        fmt::print(ofs, "{:<4}", mapidname[tpch]);
        chemcounts.clear();
        chemcounts.push_back(1);
        for (int i = 1; i < atom.size(); ++i) {
            if (atom.chemmap_[i] == tpch) {
                chemcounts.back()++;
            } else {
                tpch = atom.chemmap_[i];
                fmt::print(ofs, "{:<4}", mapidname[tpch]);
                chemcounts.push_back(1);
            }
        }
    }
    ofs << "\n  ";
    fmt::print(ofs, "{}\n", fmt::join(chemcounts, " "));
    Eigen::RowVector3d pos;
    bool seldyn = false;
    if (atom.sel_dynpos_.size() != 0) {
        ofs << "Selective dynamics\n";
        seldyn = true;
    }
    if (outdirect)
    {
        ofs << "Direct\n";
        for (int i = 0; i < atom.size(); ++i)
        {
            int cuid = packatoms ? remapid[i] : i;
            pos = atom.car2dir(atom.pos_[cuid]);
            if (lossless)
                fmt::print(ofs, "{}", fmt::join(pos, " "));
            else 
                fmt::print(ofs, "{:>16.8f}", fmt::join(pos, " "));
            if (seldyn) {
                auto &c = atom.sel_dynpos_[cuid];
                for (int j = 0; j < 3; ++j) {
                    if (c(j) == 0) {
                        ofs << " F ";
                    } else {
                        ofs << " T ";
                    }
                }
            }
            ofs << '\n';
        }
    }
    else
    {
        ofs << "Cartesian\n";
        for (int i = 0; i < atom.size(); ++i)
        {
            int cuid = packatoms ? remapid[i] : i;
            pos = atom.pos_[cuid];
            if (lossless)
                fmt::print(ofs, " {}", fmt::join(pos, "  "));
            else 
                fmt::print(ofs, " {:>16.8f}", fmt::join(pos, "  "));
            if (seldyn) {
                auto &c = atom.sel_dynpos_[cuid];
                for (int j = 0; j < 3; ++j) {
                    if (c(j) == 0) {
                        ofs << " F ";
                    } else {
                        ofs << " T ";
                    }
                }
            }
            ofs << '\n';
        }
    }
    return true;
}
/***
 * VESTA format, current use 3.0.0
 * TODO: more feature, including molecular, symmetry, color, light ... (一定不会实现的，windows下编译请删掉这句中文)
 * **/
bool GPIO::write_vesta(std::ostream &ofs, Atoms &atom) {
    ofs << "#VESTA_FORMAT_VERSION 3.0.0\n\n";
    ofs << "CRYSTAL\n\n";
    ofs << "TITLE\n Generated by GPIO\n\n";
    ofs << "GROUP\n1 1 P 1\n";
    ofs << "SYMOP\n";
    ofs << " 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1\n";
    ofs << " -1.0 -1.0 -1.0  0 0 0  0 0 0  0 0 0\n";
    ofs << "TRANM\n";
    ofs << " 0.000000  0.000000  0.000000  1  0  0   0  1  0   0  0  1\n";
    ofs << "LTRANSL\n";
    ofs << " -1\n";
    ofs << " 0.000000  0.000000  0.000000  0.000000  0.000000  0.000000\n";
    ofs << "LORIENT\n";
    ofs << " -1   0   0   0   0\n";
    ofs << " 1.000000  0.000000  0.000000  1.000000  0.000000  0.000000\n";
    ofs << " 0.000000  0.000000  1.000000  0.000000  0.000000  1.000000\n";
    ofs << "LMATRIX\n";
    ofs << " 1.000000  0.000000  0.000000  0.000000\n";
    ofs << " 0.000000  1.000000  0.000000  0.000000\n";
    ofs << " 0.000000  0.000000  1.000000  0.000000\n";
    ofs << " 0.000000  0.000000  0.000000  1.000000\n";
    ofs << " 0.000000  0.000000  0.000000\n";
    ofs << "CELLP\n";
    // a b c alpha beta gamma
    std::vector<double> paras = atom.get_cell_parameters();
    ofs << " ";
    ofs << std::setprecision(6) << std::fixed;
    for (int i = 0; i < 6; ++i)
    {
        ofs << paras[i] << " ";
    }
    ofs << "\n";
    // don't know what is this
    ofs << "  0.000000   0.000000   0.000000   0.000000   0.000000   0.000000\n";
    ofs << "STRUC\n";
    std::map<int, int> hsitem;
    std::vector<Eigen::RowVector3d> pos;
    atom.get_dir_pos(pos);
    for (int i = 0; i < atom.size(); ++i)
    {
        hsitem[atom.chemmap_[i]]++;
        ofs << " " << i + 1 << " " << mapidname[atom.chemmap_[i]] << std::setw(8)
            << mapidname[atom.chemmap_[i]] << hsitem[atom.chemmap_[i]] << "  1.0000  ";
        ofs << std::setw(12) << pos[i](0) << std::setw(12) << pos[i](1) << std::setw(12) << pos[i](2) << "  la    1\n";
        ofs << "                            0.000000   0.000000   0.000000  0.00\n";
    }
    ofs << "\n";
    return true;
}

/****
 * TODO: DO not use build_remap, this in ineffiecient
 *
 * ****/
bool GPIO::write_cif(std::ostream &ofs, Atoms &atom) {
    build_remap(atom);
    std::string chemname = "";
    for (int i = 0; i < chem.size(); ++i)
    {
        chemname += mapidname[chem[i]];
        chemname += std::to_string(chemcounts[i]);
    }
    ofs << "data_GPIO_" + chemname << '\n';
    fmt::print(ofs, "\n");
    ofs << "_audit_creation_method" << std::setw(24) << "GPIO" << '\n';
    fmt::print(ofs, "\n");
    ofs << "_chemical_formula_sum" << std::setw(24) << "'" + chemname + "'" << '\n';
    fmt::print(ofs, "\n");
    const int prec = lossless ? 10 : 6;
    const double a = atom.cell_.row(0).norm();
    const double b = atom.cell_.row(1).norm();
    const double c = atom.cell_.row(2).norm();
    Eigen::Vector3d av = atom.cell_.row(0);
    Eigen::Vector3d bv = atom.cell_.row(1);
    Eigen::Vector3d cv = atom.cell_.row(2);
    double alpha = acos(bv.dot(cv) / bv.norm() / cv.norm()) / M_PI * 180.0;
    double beta = acos(cv.dot(av) / cv.norm() / av.norm()) / M_PI * 180.0;
    double gamma = acos(av.dot(bv) / av.norm() / bv.norm()) / M_PI * 180.0;
    ofs << std::fixed << std::setprecision(prec);
    ofs << "_cell_length_a" << std::setw(24) << a << '\n';
    ofs << "_cell_length_b" << std::setw(24) << b << '\n';
    ofs << "_cell_length_c" << std::setw(24) << c << '\n';
    ofs << "_cell_angle_alpha" << std::setw(24) << alpha << '\n';
    ofs << "_cell_angle_beta" << std::setw(24) << beta << '\n';
    ofs << "_cell_angle_gamma" << std::setw(24) << gamma << '\n';
    ofs << "_cell_volume" << std::setw(24) << atom.cell_.determinant() << '\n';
    fmt::print(ofs, "\n");
    ofs << "_symmetry_space_group_name_H-M" << std::setw(20) << "'P 1'" << '\n';
    ofs << "_symmetry_Int_Tables_number" << std::setw(20) << 1 << '\n';
    fmt::print(ofs, "\n");
    ofs << "loop_\n"
        << "_space_group_symop_operation_xyz\n"
        << "  'x, y, z'\n";
    fmt::print(ofs, "\n");
    // optional per-atom columns carried in local attributes
    const bool write_occ = atom.local_attribs_d_.count("occupancy") &&
                           atom.local_attribs_d_.at("occupancy").size() == static_cast<size_t>(atom.size());
    const bool write_uiso = atom.local_attribs_d_.count("u_iso") &&
                            atom.local_attribs_d_.at("u_iso").size() == static_cast<size_t>(atom.size());
    ofs << "loop_\n"
        << "_atom_site_label\n"
        << "_atom_site_fract_x\n"
        << "_atom_site_fract_y\n"
        << "_atom_site_fract_z\n"
        << "_atom_site_type_symbol\n"
        << "_atom_site_occupancy";
    if (write_uiso) ofs << "\n_atom_site_U_iso_or_equiv";
    ofs << '\n';
    std::map<int, int> ct;
    std::vector<Eigen::RowVector3d> dirpos;
    atom.get_dir_pos(dirpos);
    for (int i = 0; i < atom.size(); ++i)
    {
        ct[atom.chemmap_[i]]++;
        ofs << std::setprecision(prec)
            << mapidname[atom.chemmap_[i]] + std::to_string(ct[atom.chemmap_[i]])
            << std::setw(15) << dirpos[i](0)
            << std::setw(15) << dirpos[i](1)
            << std::setw(15) << dirpos[i](2)
            << std::setw(6) << mapidname[atom.chemmap_[i]];
        if (write_occ)
            ofs << std::setw(10) << std::setprecision(6) << atom.local_attribs_d_.at("occupancy")[i];
        else
            ofs << std::setw(10) << std::setprecision(6) << 1.0;
        if (write_uiso)
            ofs << std::setw(12) << std::setprecision(6) << atom.local_attribs_d_.at("u_iso")[i];
        ofs << '\n';
    }
    return true;
}

bool GPIO::write_xsf(std::ostream &ofs, Atoms &atom) {
    const char *geometry;
    if (atom.pbc_ == std::array<bool, 3>{false, false, false}) geometry = "ATOMS";
    else if (atom.pbc_ == std::array<bool, 3>{true, false, false}) geometry = "POLYMER";
    else if (atom.pbc_ == std::array<bool, 3>{true, true, false}) geometry = "SLAB";
    else if (atom.pbc_ == std::array<bool, 3>{true, true, true}) geometry = "CRYSTAL";
    else {
        std::cerr << "Error: XSF supports only FFF, TFF, TTF and TTT PBC; use EXTXYZ for other axes\n";
        return false;
    }
    const bool have_forces = !atom.forces_.empty();
    if (have_forces && atom.forces_.size() != atom.pos_.size()) {
        std::cerr << "Error: XSF forces must contain one Cartesian vector per atom\n";
        return false;
    }
    std::vector<int> original_chemmap;
    if (!write_symbol && !absolute_index_write)
    {
        original_chemmap = atom.chemmap_;
        for (int i = 0; i < atom.size(); ++i)
        {
            atom.chemmap_[i] = Chemicals::atomic_numbers[mapidname[atom.chemmap_[i]]];
        }
    }
    ofs << "# " << header_line << "\n";
    ofs << geometry << "\n";
    if (!atom.is_nonperiodic()) {
        ofs << "PRIMVEC\n";
        for (int i = 0; i < 3; ++i) {
            if (lossless) {
                fmt::print(ofs, "  {} {} {}\n",
                                    atom.cell_(i, 0), atom.cell_(i, 1), atom.cell_(i, 2));
            } else {
                fmt::print(ofs, "  {: 15.6g} {: 15.6g} {: 15.6g}\n",
                                    atom.cell_(i, 0), atom.cell_(i, 1), atom.cell_(i, 2));
            }
        }
        ofs << "CONVVEC\n";
        for (int i = 0; i < 3; ++i) {
            if (lossless) {
                fmt::print(ofs, "  {} {} {}\n",
                                    atom.cell_(i, 0), atom.cell_(i, 1), atom.cell_(i, 2));
            } else {
                fmt::print(ofs, "  {: 15.6g} {: 15.6g} {: 15.6g}\n",
                                    atom.cell_(i, 0), atom.cell_(i, 1), atom.cell_(i, 2));
            }
        }
        ofs << "PRIMCOORD\n";
        fmt::print(ofs, " {} 1\n", atom.size());
    }
    for (int i = 0; i < atom.size(); ++i) {
        if (write_symbol) fmt::print(ofs, " {:6}", mapidname[atom.chemmap_[i]]);
        else fmt::print(ofs, " {:6}", atom.chemmap_[i]);
        if (lossless) fmt::print(ofs, " {}", fmt::join(atom.pos_[i], " "));
        else fmt::print(ofs, " {: 15.6g}", fmt::join(atom.pos_[i], " "));
        if (have_forces) {
            const Eigen::RowVector3d force = atom.forces_[i] / HARTREE_TO_EV;
            fmt::print(ofs, " {:.17g} {:.17g} {:.17g}", force[0], force[1], force[2]);
        }
        fmt::print(ofs, "\n");
    }
    // if (absolute_index_write)
    if (!write_symbol && !absolute_index_write)
    {
        atom.chemmap_ = std::move(original_chemmap);
    }
    // write_symbol = c;
    return true;
}

void GPIO::write_next_vasp_ABN(std::ostream &ofs, Atoms &atom) {
    build_remap(atom);
    fmt::print(ofs, "**************************************************\n");
    fmt::print(ofs, "     Configuration num.   {}\n", curr_id);
    fmt::print(ofs, "==================================================\n");
    fmt::print(ofs, "     System name\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    fmt::print(ofs, "       LATTICE \n");
    fmt::print(ofs, "==================================================\n");
    fmt::print(ofs, "     The number of atom types\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    int ntypes = chem.size();
    fmt::print(ofs, "          {}\n", ntypes);
    fmt::print(ofs, "==================================================\n");
    fmt::print(ofs, "     The number of atoms\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    fmt::print(ofs, "          {}\n", atom.size());
    fmt::print(ofs, "**************************************************\n");
    fmt::print(ofs, "     Atom types and atom numbers\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    for (int i = 0; i < ntypes; ++i)
    {
        fmt::print(ofs, "{:5} {:5}\n", mapidname[chem[i]], chemcounts[i]);
    }
    fmt::print(ofs, "==================================================\n");
    fmt::print(ofs, "     Primitive lattice vectors (ang.)\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    for (int i = 0; i < 3; ++i)
        fmt::print(ofs, "{: 20.15g} {: 20.15g} {: 20.15g}\n", 
                        atom.cell_(i, 0), atom.cell_(i, 1), atom.cell_(i, 2));
    fmt::print(ofs, "==================================================\n");
    fmt::print(ofs, "     Atomic positions (ang.)\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    for (int i = 0; i < chemmap.size(); ++i)
    {
        auto &pos = atom.pos_[remapid[i]];
        fmt::print(ofs, " {: 20.15g}\n", fmt::join(pos, " "));
    }
    fmt::print(ofs, "==================================================\n");
    fmt::print(ofs, "     Total energy (eV)\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    fmt::print(ofs, " {: 20.15g}\n", atom.energy_);
    fmt::print(ofs, "==================================================\n");
    fmt::print(ofs, "     Forces (eV ang.^-1)\n");
    fmt::print(ofs, "--------------------------------------------------\n");
    if (static_cast<int>(atom.forces_.size()) == atom.size())
        for (int i = 0; i < atom.size(); ++i)
            fmt::print(ofs, " {: 20.15g}\n", fmt::join(atom.forces_[remapid[i]], " "));
    else
        for (int i = 0; i < atom.size(); ++i)
            fmt::print(ofs, " {: 20.15g} {: 20.15g} {: 20.15g}\n", 0.0, 0.0, 0.0);
    if (atom.have_stress)
    {
        fmt::print(ofs, "==================================================\n");
        fmt::print(ofs, "     Stress (kbar)\n");
        fmt::print(ofs, "--------------------------------------------------\n");
        fmt::print(ofs, "     XX YY ZZ\n");
        fmt::print(ofs, "--------------------------------------------------\n");
        double vol = atom.cell_.determinant();
        for (int i = 0; i < 3; ++i)
            fmt::print(ofs, " {: 20.15g}", atom.stress_(i, i) * 1602.1766208 / vol);
        fmt::print(ofs, "\n");
        fmt::print(ofs, "--------------------------------------------------\n");
        fmt::print(ofs, "     XY YZ ZX\n");
        fmt::print(ofs, "--------------------------------------------------\n");
        fmt::print(ofs, " {: 20.15g} {: 20.15g} {: 20.15g}\n", 
                            atom.stress_(1, 0) * 1602.1766208 / vol,
                            atom.stress_(2, 1) * 1602.1766208 / vol,
                            atom.stress_(2, 0) * 1602.1766208 / vol);
    }
    fmt::print(ofs, "\n");
}

bool GPIO::write_dlpoly4(std::ostream &ofs, Atoms &atom) {
    bool have_force = atom.has_forces();
    bool have_v = atom.has_velo();
    bool isorth = atom.isorth();
    bool iscubic = false;
    if (isorth && atom.cell_(0, 0) == atom.cell_(1, 1) && atom.cell_(1, 1) == atom.cell_(2, 2))
    {
        iscubic = true;
    }
    ofs << "Generated by GPIO\n";
    if (have_force && have_v)
    {
        ofs << "  " << 0;
    }
    else if (have_v)
    {
        ofs << "  " << 1;
    }
    else
    {
        ofs << "  " << 0;
    }
    ofs << "    ";
    if (iscubic)
    {
        ofs << 1;
    }
    else if (isorth)
    {
        ofs << 2;
    }
    else
    {
        ofs << 3;
    }
    ofs << "   " << atom.size() << "\n";
    ofs << std::setprecision(10) << std::fixed;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            ofs << std::setw(22) << atom.cell_(i, j);
        }
        fmt::print(ofs, "\n");
    }
    for (int i = 0; i < atom.size(); ++i)
    {
        ofs << mapidname[atom.chemmap_[i]] << "      " << i + 1 << '\n';
        ofs << std::setw(22) << atom.pos_[i](0)
            << std::setw(22) << atom.pos_[i](1)
            << std::setw(22) << atom.pos_[i](2) << '\n';
        if (have_v)
        {
            ofs << std::setw(22) << atom.velocities_[i](0)
                << std::setw(22) << atom.velocities_[i](1)
                << std::setw(22) << atom.velocities_[i](2) << '\n';
        }
        if (have_force)
        {
            ofs << std::setw(22) << atom.forces_[i](0)
                << std::setw(22) << atom.forces_[i](1)
                << std::setw(22) << atom.forces_[i](2) << '\n';
        }
    }
    return true;
}

void GPIO::write_next_extxyz(std::ostream &ofs, Atoms &atom) {
    const bool scalar_magmoms = !atom.magmom_scalar_.empty();
    const bool vector_magmoms = !atom.magmoms_.empty();
    const bool native_magmoms = scalar_magmoms || vector_magmoms;
    if (scalar_magmoms && vector_magmoms) {
        std::cerr << "Error: EXTXYZ magmoms cannot encode scalar and vector moments together; use BIN\n";
        ofs.setstate(std::ios::failbit);
        return;
    }
    if ((scalar_magmoms && atom.magmom_scalar_.size() != atom.size()) ||
        (vector_magmoms && atom.magmoms_.size() != atom.size())) {
        ofs.setstate(std::ios::failbit);
        return;
    }
    fmt::print(ofs, "{}\n", atom.size());
    // Lattice, virial, energy, Properties: species pos force, and pbc
    auto &cell = atom.cell_;
    if (lossless)
    fmt::print(ofs, "Lattice=\"{} {} {} {} {} {} {} {} {}\" ", 
                        cell(0,0), cell(0,1), cell(0,2),
                        cell(1,0), cell(1,1), cell(1,2),
                        cell(2,0), cell(2,1), cell(2,2));
    else 
    fmt::print(ofs, "Lattice=\"{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\" ", 
                        cell(0,0), cell(0,1), cell(0,2),
                        cell(1,0), cell(1,1), cell(1,2),
                        cell(2,0), cell(2,1), cell(2,2));
    if (atom.has_stresses())
    {
        auto &stress = atom.stress_;
        if (lossless) {
            if (write_virtual) {
                fmt::print(ofs, "virial=\"{} {} {} {} {} {} {} {} {}\" ", 
                                    stress(0,0), stress(0,1), stress(0,2), 
                                    stress(1,0), stress(1,1), stress(1,2),
                                    stress(2,0), stress(2,1), stress(2,2));
            }
            if (write_stress) {
                auto rstress = -stress / atom.get_volume();
                fmt::print(ofs, "stress=\"{} {} {} {} {} {} {} {} {}\" ",
                                    rstress(0,0), rstress(0,1), rstress(0,2), 
                                    rstress(1,0), rstress(1,1), rstress(1,2),
                                    rstress(2,0), rstress(2,1), rstress(2,2));
            }
        }
        else {
            if (write_virtual) {
                fmt::print(ofs, "virial=\"{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\" ", 
                                stress(0,0), stress(0,1), stress(0,2), 
                                stress(1,0), stress(1,1), stress(1,2),
                                stress(2,0), stress(2,1), stress(2,2));
            } 
            if (write_stress) {
                auto rstress = -stress / atom.get_volume();
                fmt::print(ofs, "stress=\"{:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\" ", 
                                    rstress(0,0), rstress(0,1), rstress(0,2), 
                                    rstress(1,0), rstress(1,1), rstress(1,2),
                                    rstress(2,0), rstress(2,1), rstress(2,2));
            }
        }
    }
    fmt::print(ofs, "pbc=\"{} {} {}\" ",
        atom.pbc_[0] ? 'T' : 'F', atom.pbc_[1] ? 'T' : 'F',
        atom.pbc_[2] ? 'T' : 'F');
    if (atom.has_energy())
        fmt::print(ofs, "energy={} ", atom.energy_);
    if (atom.has_fenergy()) 
        fmt::print(ofs, "fenergy={} ", atom.fenergy_);
    atom.validate_hvp();
    if (atom.has_hvp()) {
        fmt::print(ofs, "hvp_directions=\"{}\" hvp_targets=\"{}\" ",
            HVPIO::write(atom.hvp_directions_), HVPIO::write(atom.hvp_targets_));
    }
    if (atom.has_hessian()) {
        std::vector<double> flat_hessian;
        const int natoms = atom.size();
        flat_hessian.reserve(static_cast<size_t>(9) * natoms * natoms);
        for (int i = 0; i < natoms; ++i)
            for (int a = 0; a < 3; ++a)
                for (int j = 0; j < natoms; ++j)
                    for (int b = 0; b < 3; ++b)
                        flat_hessian.push_back(atom.hessian_(3 * i + a, 3 * j + b));
        fmt::print(ofs, "hessian=\"{}\" ", fmt::join(flat_hessian, " "));
    }
    // These physical values take precedence over legacy metadata of the
    // same name. The canonical double total_charge itself is written below.
    const auto has_physical_value = [&](const std::string &name) {
        return (name == "energy" && atom.has_energy()) ||
               (name == "fenergy" && atom.has_fenergy()) ||
               (name == "total_charge" && atom.global_attribs_d_.count(name));
    };
    for (auto &it : atom.global_attribs_d_) {
        if (it.first != "total_charge" && has_physical_value(it.first)) continue;
        fmt::print(ofs, "{}={} ", it.first, it.second);
    }
    for (auto &it : atom.global_attribs_i_) {
        if (has_physical_value(it.first)) continue;
        fmt::print(ofs, "{}={} ", it.first, it.second);
    }
    for (auto &it : atom.global_attribs_v_) {
        if (has_physical_value(it.first)) continue;
        fmt::print(ofs, "{}=\"{}\" ", it.first, fmt::join(it.second, " "));
    }
    for (auto &it : atom.global_attribs_vi_) {
        if (has_physical_value(it.first)) continue;
        fmt::print(ofs, "{}=\"{}\" ", it.first, fmt::join(it.second, " "));
    }
    // output features
    if (atom.features_.size() > 0) {
        for (auto &it : atom.features_) {
            if (has_physical_value(it.first)) continue;
            ofs << it.first << '=' << std::quoted(it.second) << ' ';
        }
    }
    // build Properties
    fmt::print(ofs, "Properties=species:S:1:pos:R:3");
    if (atom.has_forces())
        fmt::print(ofs, ":forces:R:3");
    if (atom.has_velo()) 
        fmt::print(ofs, ":velocities:R:3");
    if (atom.has_site_energies())
        fmt::print(ofs, ":site_en:R:1");
    if (atom.has_charges())
        fmt::print(ofs, ":charges:R:1");
    if (atom.has_eff_charges())
        fmt::print(ofs, ":eff_charges:R:9");
    if (native_magmoms)
        fmt::print(ofs, ":magmoms:R:{}", scalar_magmoms ? 1 : 3);
    // Native magnetic moments take precedence over same-named local metadata.
    for (auto &it : atom.local_attribs_d_) {
        if (native_magmoms && it.first == "magmoms") continue;
        fmt::print(ofs, ":{}:R:1", it.first);
    }
    for (auto &it : atom.local_attribs_i_) {
        if (native_magmoms && it.first == "magmoms") continue;
        fmt::print(ofs, ":{}:I:1", it.first);
    }
    for (auto &it : atom.local_attribs_v_) {
        if (native_magmoms && it.first == "magmoms") continue;
        fmt::print(ofs, ":{}:R:{}", it.first, it.second.rows());
    }
    for (auto &it : atom.local_attribs_vi_) {
        if (native_magmoms && it.first == "magmoms") continue;
        fmt::print(ofs, ":{}:I:{}", it.first, it.second.rows());
    }
    fmt::print(ofs, " \n");
    for (int i = 0; i < atom.size(); ++i)
    {
        if (lossless)
            fmt::print(ofs, "{} {} ", mapidname[atom.chemmap_[i]], fmt::join(atom.pos_[i], " "));
        else 
            fmt::print(ofs, "{} {:.4f} ", mapidname[atom.chemmap_[i]], fmt::join(atom.pos_[i], " "));
        if (atom.has_forces()) {
            if (lossless) fmt::print(ofs, "{} ", fmt::join(atom.forces_[i], " "));
            else fmt::print(ofs, "{:.4f} ", fmt::join(atom.forces_[i], " "));
        }
        if (atom.has_velo()) {
            if (lossless)
            fmt::print(ofs, "{} ", fmt::join(atom.velocities_[i], " "));
            else 
            fmt::print(ofs, "{:.4f} ", fmt::join(atom.velocities_[i], " "));
        }
        if (atom.has_site_energies()) {
            if (lossless) fmt::print(ofs, "{} ", atom.site_energies_[i]);
            else fmt::print(ofs, "{:.4f} ", atom.site_energies_[i]);
        }
        if (atom.has_charges()) {
            if (lossless) fmt::print(ofs, "{} ", atom.charges_[i]);
            else fmt::print(ofs, "{:.4f} ", atom.charges_[i]);
        }
        if (atom.has_eff_charges()) {
            if (lossless) {
                for (int a = 0; a < 3; ++a) {
                    fmt::print(ofs, "{} ", fmt::join(atom.born_eff_chgs_[i].row(a), " "));
                }
            } else {
                for (int a = 0; a < 3; ++a) {
                    fmt::print(ofs, "{:.4f} ", fmt::join(atom.born_eff_chgs_[i].row(a), " "));
                }
            }
        }
        if (scalar_magmoms) {
            if (lossless) fmt::print(ofs, "{} ", atom.magmom_scalar_[i]);
            else fmt::print(ofs, "{:.4f} ", atom.magmom_scalar_[i]);
        }
        if (vector_magmoms) {
            if (lossless) fmt::print(ofs, "{} ", fmt::join(atom.magmoms_[i], " "));
            else fmt::print(ofs, "{:.4f} ", fmt::join(atom.magmoms_[i], " "));
        }
        for (auto &it : atom.local_attribs_d_) {
            if (native_magmoms && it.first == "magmoms") continue;
            if (lossless)
            fmt::print(ofs, "{} ", it.second(i));
            else 
            fmt::print(ofs, "{:.4f} ", it.second(i));
        }
        for (auto &it : atom.local_attribs_i_) {
            if (native_magmoms && it.first == "magmoms") continue;
            fmt::print(ofs, "{} ", it.second(i));
        }
        for (auto &it : atom.local_attribs_v_) {
            if (native_magmoms && it.first == "magmoms") continue;
            if (lossless)
            fmt::print(ofs, "{} ", fmt::join(it.second.col(i), " "));
            else 
            fmt::print(ofs, "{:.4f} ", fmt::join(it.second.col(i), " "));
        }
        for (auto &it : atom.local_attribs_vi_) {
            if (native_magmoms && it.first == "magmoms") continue;
            fmt::print(ofs, "{} ", fmt::join(it.second.col(i), " "));
        }
        fmt::print(ofs, "\n");
    }
}

bool GPIO::write_extxyz_sequence(std::ostream &ofs, Atoms &atom) {
    write_next_extxyz(ofs, atom);
    return static_cast<bool>(ofs);
}

void GPIO::write_next_cp2kxyz(std::ostream &ofs, Atoms &atom) {
    fmt::print(ofs, "  {0}\ni = {1}, time = {1}\n", atom.size(), write_icfg);
    for (int iat = 0; iat < atom.size(); ++iat) {
        ofs << " " << mapidname[atom.chemmap_[iat]] << "    " << atom.pos_[iat][0] << " " << atom.pos_[iat][1] << " " << atom.pos_[iat][2] << '\n';
    }
}


bool GPIO::write_n2p2(std::ostream &ofs, Atoms &atom) {
    auto fail = [&](const char *message) {
        std::cerr << "Error: N2P2 " << message << '\n';
        return false;
    };
    if (absolute_index_write) return fail("writes element symbols; absolute_index_write is unsupported");
    if (!atom.is_fully_periodic() && !atom.is_nonperiodic()) return fail("output does not support partial PBC");
    if (!atom.cell_.allFinite()) return fail("cell must be finite");
    if ((atom.is_fully_periodic() || !atom.cartesian) &&
        (!std::isfinite(atom.cell_.determinant()) || atom.cell_.determinant() == 0))
        return fail("periodic or fractional output requires a nonsingular cell");
    if (atom.have_hessian) return fail("cannot retain a full Hessian; use EXTXYZ");
    const size_t n = atom.pos_.size();
    if (atom.chemmap_.size() != n) return fail("requires one element per atom");
    if (atom.forces_.size() != n ||
        (!atom.charges_.empty() && atom.charges_.size() != n) ||
        (!atom.site_energies_.empty() && atom.site_energies_.size() != n))
        return fail("requires forces for every atom and complete charge/atomic-energy arrays when supplied");
    std::vector<Eigen::RowVector3d> positions;
    positions.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        Eigen::RowVector3d position = atom.pos_[i];
        if (!atom.cartesian) position = (position * atom.cell_).eval();
        position /= BOHR_TO_A;
        if (!position.allFinite()) return fail("coordinates must remain finite after conversion");
        positions.push_back(position);
        const auto symbol = mapidname.find(atom.chemmap_[i]);
        if (symbol == mapidname.end() || !Chemicals::atomic_numbers.count(symbol->second) ||
            Chemicals::atomic_numbers.at(symbol->second) <= 0)
            return fail("requires mapped chemical element symbols");
        if ((!atom.forces_.empty() && !(atom.forces_[i] / (HARTREE_TO_EV * A_TO_BOHR)).allFinite()) ||
            (!atom.charges_.empty() && !std::isfinite(atom.charges_[i])) ||
            (!atom.site_energies_.empty() && !std::isfinite(atom.site_energies_[i] / HARTREE_TO_EV)))
            return fail("atom labels must remain finite after conversion");
    }
    const Eigen::Matrix3d lattice = atom.cell_ / BOHR_TO_A;
    if (!lattice.allFinite() || (atom.have_energy && !std::isfinite(atom.energy_ / HARTREE_TO_EV)))
        return fail("cell and energy must remain finite after conversion");
    const auto charge = atom.global_attribs_d_.find("total_charge");
    if (charge != atom.global_attribs_d_.end() && !std::isfinite(charge->second))
        return fail("total_charge must be finite");
    ofs << "begin\n";
    for (int row = 0; row < (atom.is_fully_periodic() ? 3 : 0); ++row)
        fmt::print(ofs, "lattice {:.17g} {:.17g} {:.17g}\n", lattice(row, 0), lattice(row, 1), lattice(row, 2));
    for (size_t i = 0; i < n; ++i) {
        const auto &p = positions[i];
        Eigen::Vector3d force = Eigen::Vector3d::Zero();
        if (!atom.forces_.empty()) force = atom.forces_[i] / (HARTREE_TO_EV * A_TO_BOHR);
        const double q = atom.charges_.empty() ? 0 : atom.charges_[i];
        const double energy = atom.site_energies_.empty() ? 0 : atom.site_energies_[i] / HARTREE_TO_EV;
        fmt::print(ofs, "atom {:.17g} {:.17g} {:.17g} {} {:.17g} {:.17g} {:.17g} {:.17g} {:.17g}\n",
            p[0], p[1], p[2], mapidname.at(atom.chemmap_[i]), q, energy, force[0], force[1], force[2]);
    }
    if (atom.have_energy) fmt::print(ofs, "energy {:.17g}\n", atom.energy_ / HARTREE_TO_EV);
    if (charge != atom.global_attribs_d_.end()) fmt::print(ofs, "charge {:.17g}\n", charge->second);
    ofs << "end\n";
    return static_cast<bool>(ofs);
}

bool GPIO::validate_xsf_animation(const std::vector<const Atoms *> &frames) const {
    if (frames.empty()) return true;
    auto fail = [](const char *message) { std::cerr << "Error: AXSF " << message << '\n'; return false; };
    if (frames.size() > static_cast<size_t>(INT_MAX)) return fail("frame count is not representable");
    const auto pbc = frames.front()->pbc_;
    if (pbc != std::array<bool, 3>{false, false, false} && pbc != std::array<bool, 3>{true, false, false} &&
        pbc != std::array<bool, 3>{true, true, false} && pbc != std::array<bool, 3>{true, true, true})
        return fail("supports only FFF, TFF, TTF and TTT PBC");
    const size_t count = frames.front()->pos_.size();
    if (count > gpio_binary::MAX_ATOMS) return fail("atom count is not representable");
    for (const Atoms *frame : frames) {
        frame->validate_hvp();
        if (frame->have_hessian || frame->has_hvp()) return fail("cannot retain Hessian/HVP labels; use EXTXYZ");
        if (frame->pbc_ != pbc || frame->pos_.size() != count || frame->chemmap_.size() != count)
            return fail("requires constant atom counts and PBC with complete species arrays");
        if (!frame->cell_.allFinite()) return fail("cell must be finite");
        if (!frame->forces_.empty() && frame->forces_.size() != count)
            return fail("forces must be complete when present");
        for (size_t i = 0; i < count; ++i) {
            Eigen::RowVector3d position = frame->pos_[i];
            if (!frame->cartesian) position = (position * frame->cell_).eval();
            if (!position.allFinite()) return fail("Cartesian coordinates must be finite");
            const auto symbol = mapidname.find(frame->chemmap_[i]);
            if (symbol == mapidname.end() || !Chemicals::atomic_numbers.count(symbol->second))
                return fail("requires mapped chemical element symbols");
            if (!frame->forces_.empty() && !(frame->forces_[i] / HARTREE_TO_EV).allFinite())
                return fail("forces must remain finite after conversion");
        }
    }
    return true;
}

bool GPIO::write_xsf_animation(std::ostream &ofs, const std::vector<const Atoms *> &frames) const {
    if (!validate_xsf_animation(frames)) return false;
    if (frames.empty()) return true;
    const Atoms &first = *frames.front();
    bool variable_cell = false;
    if (!first.is_nonperiodic())
        for (const Atoms *frame : frames)
            if (frame->cell_ != first.cell_) variable_cell = true;
    auto cell = [&](const Eigen::Matrix3d &value) {
        for (int i = 0; i < 3; ++i)
            fmt::print(ofs, "{:.17g} {:.17g} {:.17g}\n", value(i, 0), value(i, 1), value(i, 2));
    };
    fmt::print(ofs, "ANIMSTEPS {}\n", frames.size());
    if (!first.is_nonperiodic()) {
        ofs << (first.pbc_[2] ? "CRYSTAL\n" : first.pbc_[1] ? "SLAB\n" : "POLYMER\n");
        if (!variable_cell) { ofs << "PRIMVEC\n"; cell(first.cell_); }
    }
    for (size_t step = 0; step < frames.size(); ++step) {
        const Atoms &frame = *frames[step];
        if (frame.is_nonperiodic()) fmt::print(ofs, "ATOMS {}\n", step + 1);
        else {
            if (variable_cell) { fmt::print(ofs, "PRIMVEC {}\n", step + 1); cell(frame.cell_); }
            fmt::print(ofs, "PRIMCOORD {}\n{} 1\n", step + 1, frame.size());
        }
        for (int i = 0; i < frame.size(); ++i) {
            const auto &symbol = mapidname.at(frame.chemmap_[i]);
            if (write_symbol && !absolute_index_write) ofs << symbol;
            else ofs << Chemicals::atomic_numbers.at(symbol);
            Eigen::RowVector3d position = frame.pos_[i];
            if (!frame.cartesian) position = (position * frame.cell_).eval();
            fmt::print(ofs, " {:.17g} {:.17g} {:.17g}", position[0], position[1], position[2]);
            if (!frame.forces_.empty()) {
                const Eigen::Vector3d force = frame.forces_[i] / HARTREE_TO_EV;
                fmt::print(ofs, " {:.17g} {:.17g} {:.17g}", force[0], force[1], force[2]);
            }
            ofs << '\n';
        }
    }
    return static_cast<bool>(ofs);
}
