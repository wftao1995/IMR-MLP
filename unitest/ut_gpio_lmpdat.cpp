// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "io/gpio.h"

#include <sstream>

namespace {

Atoms make_tio_atom()
{
    Atoms atom;
    atom.cell_ = 5.0 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {{true, true, true}};
    atom.chemmap_ = {22, 8}; // atomic numbers: Ti, O (matches setmap())
    atom.pos_ = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    return atom;
}

Atoms make_triclinic_atom()
{
    Atoms atom;
    atom.cell_ << 20.0, 0.0, 0.0,
                 -8.0, 13.0, 0.0,
                  3.0, -2.0, 37.0;
    atom.pbc_ = {{true, true, true}};
    atom.chemmap_ = {0, 0};
    atom.pos_ = {{1.0, 2.0, 3.0}, {10.0, 6.0, 20.0}};
    return atom;
}

} // namespace

TEST_CASE("GPIO lmpdat Masses uses atomic-number-indexed masses", "[gpio][lmpdat]")
{
    Atoms atom = make_tio_atom();
    GPIO io;
    io.setmap(); // map key == atomic number
    io.write_mass = true;

    std::ostringstream output;
    REQUIRE(io.write_one(atom, output, "lmpdat"));
    std::string text = output.str();

    REQUIRE(text.find("2 atom types") != std::string::npos);
    // Mass must come from the element's atomic number, not from map row
    // order (regression: with map order {Ti, O}, Ti used to get H's mass
    // because atomic_masses[0] is the H placeholder).
    REQUIRE(text.find("23 47.867  # Ti") != std::string::npos);
    REQUIRE(text.find("9 15.999  # O") != std::string::npos);
    REQUIRE(text.find("1.008  # Ti") == std::string::npos);
}

TEST_CASE("GPIO lmpdat Atoms section keeps 1-based type ids from chemmap", "[gpio][lmpdat]")
{
    Atoms atom = make_tio_atom();
    GPIO io;
    io.setmap();

    std::ostringstream output;
    REQUIRE(io.write_one(atom, output, "lmpdat"));
    std::string text = output.str();

    size_t atoms_pos = text.find("\nAtoms # atomic");
    REQUIRE(atoms_pos != std::string::npos);
    std::string body = text.substr(atoms_pos);
    // first atom (map id 22 = Ti) must be type 23, second (id 8 = O) type 9;
    // field widths are {:8} id and {:6} type
    REQUIRE(body.find("       1     23") != std::string::npos);
    REQUIRE(body.find("       2      9") != std::string::npos);
}

TEST_CASE("GPIO lmpdat preserves restricted triclinic box lengths", "[gpio][lmpdat][triclinic]")
{
    Atoms atom = make_triclinic_atom();
    GPIO io;
    io.setmap();

    std::ostringstream output;
    REQUIRE(io.write_one(atom, output, "lmpdat"));
    std::istringstream input(output.str());
    Atoms back;
    REQUIRE(io.read_one(back, input, "lmpdat"));
    REQUIRE(back.cell_.isApprox(atom.cell_, 1e-12));
}

TEST_CASE("GPIO lammpstrj emits bounding bounds for triclinic boxes",
          "[gpio][lammps][triclinic]")
{
    Atoms atom = make_triclinic_atom();
    GPIO io;
    io.setmap();

    std::ostringstream output;
    REQUIRE(io.seq_write_init(output, "lammpstrj"));
    REQUIRE(io.seq_write(output, atom));
    const std::string dump = output.str();
    std::istringstream lines(dump);
    std::string line;
    for (int i = 0; i < 6; ++i) REQUIRE(std::getline(lines, line));
    std::istringstream x_bounds(line);
    double lo, hi, tilt;
    REQUIRE(x_bounds >> lo >> hi >> tilt);
    REQUIRE(lo == Catch::Approx(-8.0));
    REQUIRE(hi == Catch::Approx(23.0));
    REQUIRE(tilt == Catch::Approx(-8.0));
    REQUIRE(std::getline(lines, line));
    std::istringstream y_bounds(line);
    REQUIRE(y_bounds >> lo >> hi >> tilt);
    REQUIRE(lo == Catch::Approx(-2.0));
    REQUIRE(hi == Catch::Approx(13.0));
    REQUIRE(tilt == Catch::Approx(3.0));

    std::istringstream input(dump);
    Atoms back;
    REQUIRE(io.read_one(back, input, "lammpstrj"));
    REQUIRE(back.cell_.isApprox(atom.cell_, 1e-12));
}
