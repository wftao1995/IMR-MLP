// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "io/gpio.h"

#include <cmath>
#include <sstream>

namespace {

Atoms make_cubic_atom()
{
    Atoms atom;
    atom.cell_ = 5.0 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {{true, true, true}};
    return atom;
}

std::string minimal_header(double a, double b, double c)
{
    return "data_test\n"
           "_cell_length_a " + std::to_string(a) + "\n"
           "_cell_length_b " + std::to_string(b) + "\n"
           "_cell_length_c " + std::to_string(c) + "\n"
           "_cell_angle_alpha 90.0\n"
           "_cell_angle_beta 90.0\n"
           "_cell_angle_gamma 90.0\n";
}

} // namespace

TEST_CASE("GPIO CIF round-trips a cubic cell", "[gpio][cif]")
{
    Atoms atom;
    atom.cell_ = 5.0 * Eigen::Matrix3d::Identity();
    atom.chemmap_ = {1, 8}; // H, O
    atom.pos_ = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};

    std::ostringstream output;
    GPIO io;
    REQUIRE(io.write_one(atom, output, "cif"));
    std::istringstream input(output.str());
    Atoms back;
    REQUIRE(io.read_one(back, input, "cif"));
    REQUIRE(back.size() == 2);
    REQUIRE(back.chemmap_ == std::vector<int>({1, 8}));
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(back.cell_(i, j) == Catch::Approx(atom.cell_(i, j)).margin(1e-7));
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(back.pos_[i][j] == Catch::Approx(atom.pos_[i][j]).margin(1e-6));
}

TEST_CASE("GPIO CIF preserves the selected coordinate precision on every atom row", "[gpio][cif]")
{
    const bool lossless = GENERATE(false, true);
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.chemmap_ = {13, 13};
    atom.pos_ = {{1.234567891, 3.456789123, 5.678912345},
                 {2.345678912, 4.567891234, 6.789123456}};
    atom.local_attribs_d_["occupancy"] = Eigen::Vector2d(.87654321, .65432109);
    atom.local_attribs_d_["u_iso"] = Eigen::Vector2d(.0123456789, .0234567891);
    const auto positions = atom.pos_;
    GPIO io;
    io.lossless = lossless;
    std::ostringstream output;
    REQUIRE(io.write_one(atom, output, "cif"));
    const std::vector<std::vector<std::string>> expected = lossless
        ? std::vector<std::vector<std::string>>{{"0.1234567891", "0.3456789123", "0.5678912345"},
                                                {"0.2345678912", "0.4567891234", "0.6789123456"}}
        : std::vector<std::vector<std::string>>{{"0.123457", "0.345679", "0.567891"},
                                                {"0.234568", "0.456789", "0.678912"}};
    std::istringstream lines(output.str());
    std::string line;
    int rows = 0;
    while (std::getline(lines, line)) {
        std::istringstream row(line);
        std::string label, x, y, z, element, occupancy, u_iso;
        if (!(row >> label) || (label != "Al1" && label != "Al2")) continue;
        REQUIRE(row >> x >> y >> z >> element >> occupancy >> u_iso);
        REQUIRE(rows < 2);
        CHECK(std::vector<std::string>{x, y, z} == expected[rows]);
        CHECK(element == "Al");
        CHECK(occupancy == (rows == 0 ? "0.876543" : "0.654321"));
        CHECK(u_iso == (rows == 0 ? "0.012346" : "0.023457"));
        ++rows;
    }
    REQUIRE(rows == 2);
    std::istringstream input(output.str());
    Atoms back;
    REQUIRE(io.read_one(back, input, "cif"));
    REQUIRE(back.chemmap_ == atom.chemmap_);
    REQUIRE(back.size() == 2);
    for (int i = 0; i < 2; ++i)
        for (int axis = 0; axis < 3; ++axis) {
            CHECK(back.pos_[i][axis] == Catch::Approx(10 * std::stod(expected[i][axis])).margin(1e-10));
            CHECK(std::abs(back.pos_[i][axis] - positions[i][axis]) < (lossless ? 1e-9 : 5.1e-6));
        }
    CHECK(atom.pos_ == positions);
}

TEST_CASE("GPIO CIF accepts quoted numeric coordinates without dropping atoms", "[gpio][cif]")
{
    const bool cartesian = GENERATE(false, true);
    const std::string coordinates = cartesian
        ? "_atom_site_Cartn_x\n_atom_site_Cartn_y\n_atom_site_Cartn_z\n"
        : "_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\n";
    // Leading plus and uncertainty suffixes retain stod's existing numeric
    // prefix convention; only surrounding quotes are newly stripped.
    const std::string second = cartesian ? "Al '+1.0(2)' \"+2.0\" 3.0\n"
                                         : "Al '+0.25(2)' \"+0.5\" 0.75\n";
    std::istringstream input(minimal_header(4, 4, 4) +
        "loop_\n_atom_site_type_symbol\n" + coordinates + "Al 0 0 0\n" + second);
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.chemmap_ == std::vector<int>{13, 13});
    REQUIRE(atom.pos_[0].isZero(1e-12));
    CHECK(atom.pos_[1][0] == Catch::Approx(1).margin(1e-12));
    CHECK(atom.pos_[1][1] == Catch::Approx(2).margin(1e-12));
    CHECK(atom.pos_[1][2] == Catch::Approx(3).margin(1e-12));
}

TEST_CASE("GPIO CIF round-trips a triclinic cell", "[gpio][cif]")
{
    Eigen::Matrix3d cell;
    cell << 6.5, 1.2, 0.3,
            0.1, 5.0, 1.4,
            0.7, -0.2, 7.3;
    Atoms atom;
    atom.cell_ = cell;
    atom.chemmap_ = {26, 8}; // Fe, O
    atom.pos_ = {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}};
    std::vector<Eigen::RowVector3d> frac;
    atom.get_dir_pos(frac);

    std::ostringstream output;
    GPIO io;
    REQUIRE(io.write_one(atom, output, "cif"));
    std::istringstream input(output.str());
    Atoms back;
    REQUIRE(io.read_one(back, input, "cif"));
    REQUIRE(back.size() == 2);
    // cell parameters (lengths and angles) are preserved
    std::vector<double> p1 = atom.get_cell_parameters();
    std::vector<double> p2 = back.get_cell_parameters();
    for (int i = 0; i < 6; ++i)
        REQUIRE(p2[i] == Catch::Approx(p1[i]).margin(1e-7));
    // fractional coordinates are preserved exactly
    std::vector<Eigen::RowVector3d> backfrac;
    back.get_dir_pos(backfrac);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 3; ++j)
            REQUIRE(backfrac[i][j] == Catch::Approx(frac[i][j]).margin(1e-7));
}

TEST_CASE("GPIO CIF handles quotes, comments and '?' occupancy", "[gpio][cif]")
{
    std::istringstream input(
        "# top comment\n"
        "data_test\n"
        "_cell_length_a '5.0'\n"
        "_cell_length_b \"5.0\"\n"
        "_cell_length_c 5.0\n"
        "_cell_angle_alpha 90.0\n"
        "_cell_angle_beta 90.0\n"
        "_cell_angle_gamma 90.0\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Fe1 'Fe' 0.0 0.0 0.0 1.0  # inline comment\n"
        "Fe2 \"Fe\" 0.5 0.5 0.5 ?\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.local_attribs_d_.count("occupancy") == 0);
    REQUIRE(atom.pos_[1][0] == Catch::Approx(2.5));
    REQUIRE(atom.pos_[1][1] == Catch::Approx(2.5));
    REQUIRE(atom.pos_[1][2] == Catch::Approx(2.5));
}

TEST_CASE("GPIO CIF reads cell parameters inside loop_ (Materials Studio)", "[gpio][cif]")
{
    std::istringstream input(
        "data_test\n"
        "loop_\n"
        "_cell_length_a\n"
        "_cell_length_b\n"
        "_cell_length_c\n"
        "_cell_angle_alpha\n"
        "_cell_angle_beta\n"
        "_cell_angle_gamma\n"
        "5.0 6.0 7.0 90.0 90.0 90.0\n"
        "loop_\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Fe 0 0 0\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(5.0));
    REQUIRE(atom.cell_(1, 1) == Catch::Approx(6.0));
    REQUIRE(atom.cell_(2, 2) == Catch::Approx(7.0));
    REQUIRE(atom.chemmap_ == std::vector<int>{io.getrevmap().at("Fe")});
}

TEST_CASE("GPIO CIF skips symmetry loop before the atom loop", "[gpio][cif]")
{
    std::istringstream input(
        minimal_header(5.0, 5.0, 5.0) +
        "loop_\n"
        "_space_group_symop_operation_xyz\n"
        "x, y, z\n"
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Fe1 0 0 0\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.chemmap_ == std::vector<int>{io.getrevmap().at("Fe")});
}

TEST_CASE("GPIO CIF reads Cartesian atom coordinates", "[gpio][cif]")
{
    std::istringstream input(
        minimal_header(5.0, 5.0, 5.0) +
        "loop_\n"
        "_atom_site_type_symbol\n"
        "_atom_site_Cartn_x\n"
        "_atom_site_Cartn_y\n"
        "_atom_site_Cartn_z\n"
        "Fe 1.0 2.0 3.0\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.pos_[0][0] == Catch::Approx(1.0));
    REQUIRE(atom.pos_[0][1] == Catch::Approx(2.0));
    REQUIRE(atom.pos_[0][2] == Catch::Approx(3.0));
}

TEST_CASE("GPIO CIF normalizes element symbols and skips unknowns", "[gpio][cif]")
{
    std::istringstream input(
        minimal_header(5.0, 5.0, 5.0) +
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "D1 0.0 0.0 0.0\n"
        "Fe1 0.25 0.25 0.25\n"
        "Zz1 0.5 0.5 0.5\n"
        "Fe2 0.75 0.75 0.75\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 3); // Zz is skipped
    REQUIRE(atom.chemmap_ ==
            std::vector<int>({io.getrevmap().at("H"), io.getrevmap().at("Fe"), io.getrevmap().at("Fe")}));
}

TEST_CASE("GPIO CIF stores partial occupancy as an attribute", "[gpio][cif]")
{
    std::istringstream input(
        minimal_header(5.0, 5.0, 5.0) +
        "loop_\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "_atom_site_occupancy\n"
        "Fe 0.0 0.0 0.0 0.5\n"
        "O  0.5 0.5 0.5 1.0\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.local_attribs_d_.count("occupancy") == 1);
    REQUIRE(atom.local_attribs_d_.at("occupancy")(0) == Catch::Approx(0.5));
    REQUIRE(atom.local_attribs_d_.at("occupancy")(1) == Catch::Approx(1.0));
}

TEST_CASE("GPIO CIF aligns optional attributes across loops and missing columns", "[gpio][cif]")
{
    const std::string atom_columns =
        "loop_\n_atom_site_type_symbol\n_atom_site_fract_x\n"
        "_atom_site_fract_y\n_atom_site_fract_z\n";
    std::istringstream input(minimal_header(5.0, 5.0, 5.0) +
        atom_columns + "O 0 0 0\n" +
        atom_columns + "_atom_site_occupancy\n_atom_site_U_iso_or_equiv\n"
        "Zz 0.1 0.1 0.1 0.2 0.03\n"
        "Fe 0.2 0.2 0.2 0.5 0.02\n"
        "H 0.4 0.4 0.4 ?\n" +
        atom_columns + "C 0.6 0.6 0.6\n");
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 4);
    REQUIRE(atom.chemmap_ == std::vector<int>{8, 26, 1, 6});
    REQUIRE(atom.local_attribs_d_.at("occupancy").size() == 4);
    REQUIRE(atom.local_attribs_d_.at("u_iso").size() == 4);
    REQUIRE(atom.local_attribs_d_.at("occupancy").isApprox(Eigen::Vector4d(1, 0.5, 1, 1)));
    REQUIRE(atom.local_attribs_d_.at("u_iso").isApprox(Eigen::Vector4d(0, 0.02, 0, 0)));

    for (const std::string format : {"cif", "extxyz"}) {
        INFO(format);
        std::stringstream output;
        REQUIRE(io.write_one(atom, output, format));
        Atoms back;
        REQUIRE(io.read_one(back, output, format));
        REQUIRE(back.chemmap_ == atom.chemmap_);
        for (const auto &name : {"occupancy", "u_iso"}) {
            REQUIRE(back.local_attribs_d_.at(name).size() == atom.size());
            REQUIRE(back.local_attribs_d_.at(name).isApprox(atom.local_attribs_d_.at(name)));
        }
    }
}

TEST_CASE("GPIO CIF reads multiple data_ blocks", "[gpio][cif]")
{
    std::istringstream input(
        "data_first\n"
        "_cell_length_a 5.0\n"
        "_cell_length_b 5.0\n"
        "_cell_length_c 5.0\n"
        "_cell_angle_alpha 90.0\n"
        "_cell_angle_beta 90.0\n"
        "_cell_angle_gamma 90.0\n"
        "loop_\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Fe 0 0 0\n"
        "data_second\n"
        "_cell_length_a 6.0\n"
        "_cell_length_b 6.0\n"
        "_cell_length_c 6.0\n"
        "_cell_angle_alpha 90.0\n"
        "_cell_angle_beta 90.0\n"
        "_cell_angle_gamma 90.0\n"
        "loop_\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "O 0.5 0.5 0.5\n");
    std::vector<Atoms> atoms;
    GPIO io;
    REQUIRE(io.read(atoms, input, "cif"));
    REQUIRE(atoms.size() == 2);
    REQUIRE(atoms[0].cell_(0, 0) == Catch::Approx(5.0));
    REQUIRE(atoms[1].cell_(0, 0) == Catch::Approx(6.0));
    REQUIRE(atoms[0].chemmap_ == std::vector<int>{io.getrevmap().at("Fe")});
    REQUIRE(atoms[1].chemmap_ == std::vector<int>{io.getrevmap().at("O")});
}

TEST_CASE("GPIO CIF defaults missing cell angles to 90 degrees", "[gpio][cif]")
{
    std::istringstream input(
        "data_test\n"
        "_cell_length_a 5.0\n"
        "_cell_length_b 6.0\n"
        "_cell_length_c 7.0\n"
        "loop_\n"
        "_atom_site_type_symbol\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Fe 0 0 0\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(5.0));
    REQUIRE(atom.cell_(1, 1) == Catch::Approx(6.0));
    REQUIRE(atom.cell_(2, 2) == Catch::Approx(7.0));
    REQUIRE(atom.cell_(0, 1) == Catch::Approx(0.0).margin(1e-12));
    REQUIRE(atom.cell_(2, 0) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("GPIO CIF falls back to _atom_site_label for elements", "[gpio][cif]")
{
    std::istringstream input(
        minimal_header(5.0, 5.0, 5.0) +
        "loop_\n"
        "_atom_site_label\n"
        "_atom_site_fract_x\n"
        "_atom_site_fract_y\n"
        "_atom_site_fract_z\n"
        "Fe1 0 0 0\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "cif"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.chemmap_ == std::vector<int>{io.getrevmap().at("Fe")});
}

TEST_CASE("GPIO CIF writes volume, symmetry and occupancy columns", "[gpio][cif]")
{
    Atoms atom = make_cubic_atom();
    atom.chemmap_ = {26};
    atom.pos_ = {{0.0, 0.0, 0.0}};
    Eigen::VectorXd occ(1);
    occ << 0.75;
    atom.local_attribs_d_["occupancy"] = occ;
    Eigen::VectorXd u(1);
    u << 0.01;
    atom.local_attribs_d_["u_iso"] = u;

    std::ostringstream output;
    GPIO io;
    REQUIRE(io.write_one(atom, output, "cif"));
    const std::string cif = output.str();
    REQUIRE(cif.find("_cell_volume") != std::string::npos);
    REQUIRE(cif.find("_symmetry_space_group_name_H-M") != std::string::npos);
    REQUIRE(cif.find("_space_group_symop_operation_xyz") != std::string::npos);
    REQUIRE(cif.find("_atom_site_U_iso_or_equiv") != std::string::npos);
    REQUIRE(cif.find("0.750000") != std::string::npos);
    REQUIRE(cif.find("0.010000") != std::string::npos);

    std::istringstream input(cif);
    Atoms back;
    REQUIRE(io.read_one(back, input, "cif"));
    REQUIRE(back.size() == 1);
    REQUIRE(back.local_attribs_d_.count("occupancy") == 1);
    REQUIRE(back.local_attribs_d_.at("occupancy")(0) == Catch::Approx(0.75));
    REQUIRE(back.local_attribs_d_.count("u_iso") == 1);
    REQUIRE(back.local_attribs_d_.at("u_iso")(0) == Catch::Approx(0.01));
}
