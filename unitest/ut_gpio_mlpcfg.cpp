// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <sstream>

using Catch::Approx;

TEST_CASE("GPIO text mlpcfg reads mixed optional training labels",
          "[gpio][mlpcfg]")
{
    const bool first_has_force = GENERATE(false, true);
    const bool separate_files = GENERATE(false, true);
    GPIO reader;
    GPIO writer;
    std::stringstream buffer;
    std::vector<Atoms> output;
    for (int i = 0; i < 4; ++i) {
        Atoms atom;
        atom.cell_ = 5.0 * Eigen::Matrix3d::Identity();
        atom.pos_ = {{0.0, 0.0, 0.0}};
        atom.chemmap_ = {0};
        atom.energy_ = -1.0 - i;
        atom.have_energy = true;
        if ((i % 2 == 0) == first_has_force)
            atom.forces_ = {{0.1, 0.2, 0.3}};
        atom.have_stress = i < 2;
        atom.stress_ = 2.0 * Eigen::Matrix3d::Identity();
        REQUIRE(writer.write_one(atom, buffer, "mlpcfg"));
        if (separate_files) {
            REQUIRE(reader.read(output, buffer, "mlpcfg"));
            buffer.str("");
            buffer.clear();
        }
    }
    if (!separate_files)
        REQUIRE(reader.read(output, buffer, "mlpcfg"));
    REQUIRE(output.size() == 4);
    for (int i = 0; i < 4; ++i) {
        CAPTURE(i, first_has_force, separate_files);
        REQUIRE(output[i].has_energy());
        REQUIRE(output[i].energy_ == Approx(-1.0 - i));
        REQUIRE(output[i].has_forces() == ((i % 2 == 0) == first_has_force));
        if (output[i].has_forces())
            REQUIRE(output[i].forces_[0].isApprox(Eigen::Vector3d(0.1, 0.2, 0.3)));
        REQUIRE(output[i].has_stresses() == (i < 2));
        if (output[i].has_stresses())
            REQUIRE(output[i].stress_.isApprox(2.0 * Eigen::Matrix3d::Identity()));
    }
}

TEST_CASE("GPIO text mlpcfg reports truncated training labels",
          "[gpio][mlpcfg][errors]")
{
    const auto invalid_frame = GENERATE(
        "BEGIN_CFG\nSize\n1\nAtomData: id type cartes_x cartes_y cartes_z fx fy fz\n1 0 0 0 0\nEND_CFG\n",
        "BEGIN_CFG\nSize\n1\nAtomData: id type cartes_x cartes_y cartes_z\n1 0 0 0 0\nPlusStress: xx yy zz yz xz xy\n1 2\nEND_CFG\n");
    std::stringstream buffer;
    buffer << "BEGIN_CFG\nSize\n1\nAtomData: id type cartes_x cartes_y cartes_z\n1 0 0 0 0\nEnergy\n-1\nEND_CFG\n"
           << invalid_frame;
    GPIO reader;
    std::vector<Atoms> output;
    REQUIRE_FALSE(reader.read(output, buffer, "mlpcfg"));
    REQUIRE(output.size() == 1);
}

TEST_CASE("GPIO text mlpcfg round-trips atomic and configuration fitness",
          "[gpio][mlpcfg]")
{
    Atoms input;
    input.cell_ = 5.0 * Eigen::Matrix3d::Identity();
    input.pos_ = {{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {2.0, 2.0, 2.0}};
    input.chemmap_ = {0, 1, 0};
    input.forces_ = {{0.1, 0.0, 0.0}, {0.0, 0.2, 0.0}, {0.0, 0.0, 0.3}};
    input.local_attribs_d_["atomic_fitness"].resize(3);
    input.local_attribs_d_["atomic_fitness"] << 0.5, 1.25, 2.0;
    input.global_attribs_d_["fitness"] = 1.25;

    std::stringstream buffer;
    GPIO writer;
    REQUIRE(writer.write_one(input, buffer, "mlpcfg"));

    buffer.seekg(0);
    Atoms output;
    GPIO reader;
    REQUIRE(reader.read_one(output, buffer, "mlpcfg"));
    REQUIRE(output.local_attribs_d_.count("atomic_fitness") == 1);
    REQUIRE(output.local_attribs_d_.at("atomic_fitness").isApprox(
        input.local_attribs_d_.at("atomic_fitness")));
    REQUIRE(output.global_attribs_d_.at("fitness") == Approx(1.25));
    REQUIRE(output.fitness == Approx(1.25));
}

TEST_CASE("GPIO text mlpcfg rejects local attributes colliding with AtomData",
          "[gpio][mlpcfg][errors]")
{
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity();
    atom.pos_ = {{0.0, 0.0, 0.0}};
    atom.chemmap_ = {0};
    const std::string column = GENERATE("fx", "q", "vx", "vy", "vz", "x", "y", "z");
    atom.local_attribs_d_[column] = Eigen::VectorXd::Ones(1);
    std::stringstream output;
    GPIO io;
    REQUIRE_FALSE(io.write_one(atom, output, "mlpcfg"));
}

namespace {
std::string simple_cfg(const std::string &type = "0", const std::string &energy = "-2") {
    return "BEGIN_CFG\nSize\n1\nSupercell\n20 0 0\n0 20 0\n0 0 20\n"
           "AtomData: id type cartes_x cartes_y cartes_z\n1 " + type +
           " 0 0 0\nEnergy\n" + energy + "\nEND_CFG\n";
}
}

TEST_CASE("CFG distinguishes truncated frame headers from clean EOF", "[gpio][mlpcfg][errors]") {
    const std::string suffix = GENERATE("BEGIN_CFG", "BEGIN_CFG\n", "BEGIN_CFG\nSize",
        "BEGIN_CFG\nSize\n", "BEGIN_CFG\nSize\n0\n", "BEGIN_CFG\nSize\n1\n");
    const bool valid_prefix = GENERATE(false, true);
    GPIO reader;
    std::vector<Atoms> frames;
    std::istringstream input((valid_prefix ? simple_cfg() : "") + suffix);
    REQUIRE_FALSE(reader.read(frames, input, "mlpcfg"));
    REQUIRE(frames.size() == (valid_prefix ? 1 : 0));
}

TEST_CASE("CFG accepts clean EOF and trailing blank lines", "[gpio][mlpcfg]") {
    const std::string ending = GENERATE("", "\n", "\n \t\r\n");
    auto text = simple_cfg();
    text.pop_back(); // END_CFG without a final newline is a complete frame.
    std::istringstream input(text + ending);
    GPIO reader;
    std::vector<Atoms> frames;
    REQUIRE(reader.read(frames, input, "mlpcfg"));
    REQUIRE(frames.size() == 1);
    std::istringstream empty(ending);
    frames.clear();
    REQUIRE(reader.read(frames, empty, "mlpcfg"));
    REQUIRE(frames.empty());
}

TEST_CASE("CFG rejects invalid atom types before integer conversion", "[gpio][mlpcfg][errors]") {
    const std::string type = GENERATE("0.5", "nan", "inf", "-1", "2147483648", "1junk");
    std::istringstream input(simple_cfg(type));
    GPIO reader;
    std::vector<Atoms> frames;
    REQUIRE_FALSE(reader.read(frames, input, "mlpcfg"));
    REQUIRE(frames.empty());
}

TEST_CASE("CFG accepts nonnegative integral type values", "[gpio][mlpcfg]") {
    const std::string type = GENERATE("0", "1.0", "2e0", "2147483647");
    std::istringstream input(simple_cfg(type));
    GPIO reader;
    Atoms frame;
    REQUIRE(reader.read_one(frame, input, "mlpcfg"));
    REQUIRE(frame.chemmap_[0] == static_cast<int>(std::stod(type)));
}

TEST_CASE("CFG rejects non-finite labels and malformed numeric tokens", "[gpio][mlpcfg][errors]") {
    const std::string energy = GENERATE("-2junk", "-2,", "nan", "inf", "-inf", "1e999");
    std::istringstream input(simple_cfg("0", energy));
    GPIO reader;
    std::vector<Atoms> frames;
    REQUIRE_FALSE(reader.read(frames, input, "mlpcfg"));
    REQUIRE(frames.empty());
}

TEST_CASE("MLPCFG sequential reader distinguishes clean EOF and resets after corrupt frames", "[gpio][mlpcfg][sequential]") {
    const std::string good = "BEGIN_CFG\nSize\n1\nSupercell\n10 0 0\n0 10 0\n0 0 10\n"
                             "AtomData: id type cartes_x cartes_y cartes_z\n1 0 0 0 0\nEnergy\n-2\nEND_CFG";
    GPIO reader;
    Atoms atom;
    SECTION("clean EOF without final newline") {
        std::istringstream input(good);
        REQUIRE(reader.seq_read_init(input, "mlpcfg"));
        REQUIRE(reader.seq_read(input, atom));
        REQUIRE(atom.id == 0);
        REQUIRE_FALSE(reader.seq_read(input, atom));
        REQUIRE(atom.size() == 0);
        REQUIRE_FALSE(reader.seq_read(input, atom));
    }
    SECTION("truncated frame is not EOF") {
        std::istringstream input(good + "\nBEGIN_CFG\nSize\n1\nSupercell\n10 0\n");
        REQUIRE(reader.seq_read_init(input, "mlpcfg"));
        REQUIRE(reader.seq_read(input, atom));
        REQUIRE_THROWS_WITH(reader.seq_read(input, atom), Catch::Matchers::ContainsSubstring("frame 1"));
        REQUIRE(atom.size() == 0);
        REQUIRE_FALSE(reader.seq_read(input, atom));
        std::istringstream next(good);
        REQUIRE(reader.seq_read_init(next, "mlpcfg"));
        REQUIRE(reader.seq_read(next, atom));
        REQUIRE(atom.id == 0);
    }
}

TEST_CASE("CFG preserves charges velocities and independent total charge", "[gpio][mlpcfg]") {
    const bool lossless = GENERATE(false, true);
    // Independent hand-written input exercises column order and atom IDs.
    std::istringstream source(
        "BEGIN_CFG\nSize\n2\nSupercell\n5 0 0\n0 5 0\n0 0 5\n"
        "AtomData: id type q vx vy vz cartes_x cartes_y cartes_z\n"
        "2 1 0.05 -0.1 0.5 0.6 1 2 3\n"
        "1 0 0.1 0.2 0.3 0.4 0 0 0\n"
        "Feature total_charge 0.8\nFeature label first  frame\nEND_CFG\n");
    GPIO reader;
    Atoms input;
    REQUIRE(reader.read_one(input, source, "mlpcfg"));
    REQUIRE(input.charges_.size() == 2);
    REQUIRE(input.charges_[0] == Approx(0.1));
    REQUIRE(input.charges_[1] == Approx(0.05));
    REQUIRE(input.velocities_.size() == 2);
    REQUIRE(input.velocities_[0].isApprox(Eigen::Vector3d(0.2, 0.3, 0.4)));
    REQUIRE(input.velocities_[1].isApprox(Eigen::Vector3d(-0.1, 0.5, 0.6)));
    REQUIRE(input.global_attribs_d_.at("total_charge") == Approx(0.8));
    REQUIRE(input.features_.count("total_charge") == 0);
    REQUIRE(input.features_.at("label") == "first  frame");

    // The canonical typed value takes precedence over stale legacy text.
    input.features_["total_charge"] = "legacy charge text";
    GPIO writer;
    writer.lossless = lossless;
    std::stringstream written;
    REQUIRE(writer.write_one(input, written, "mlpcfg"));
    const auto text = written.str();
    const auto charge_tag = text.find("Feature total_charge ");
    REQUIRE(charge_tag != std::string::npos);
    REQUIRE(text.find("Feature total_charge ", charge_tag + 1) == std::string::npos);
    REQUIRE(text.find("legacy charge text") == std::string::npos);
    REQUIRE(text.find(" q vx vy vz") != std::string::npos);
    Atoms output;
    REQUIRE(reader.read_one(output, written, "mlpcfg"));
    REQUIRE(output.charges_ == input.charges_);
    REQUIRE(output.velocities_.size() == 2);
    REQUIRE(output.velocities_[0].isApprox(input.velocities_[0]));
    REQUIRE(output.velocities_[1].isApprox(input.velocities_[1]));
    REQUIRE(output.global_attribs_d_.at("total_charge") == Approx(0.8));
    REQUIRE(output.features_.at("label") == "first  frame");
}

TEST_CASE("CFG optional charge and velocity labels do not leak between frames", "[gpio][mlpcfg]") {
    const bool lossless = GENERATE(false, true);
    GPIO writer;
    writer.lossless = lossless;
    std::stringstream stream;
    for (int frame = 0; frame < 3; ++frame) {
        Atoms atom;
        atom.cell_ = 5.0 * Eigen::Matrix3d::Identity();
        atom.pos_ = {{0.0, 0.0, 0.0}};
        atom.chemmap_ = {0};
        if (frame != 1) {
            atom.charges_ = {frame == 0 ? 0.1 : 0.0};
            atom.velocities_ = {{frame == 0 ? 0.2 : 0.0,
                                 frame == 0 ? 0.3 : 0.0,
                                 frame == 0 ? 0.4 : 0.0}};
            atom.global_attribs_d_["total_charge"] = frame == 0 ? 0.8 : 0.0;
        }
        REQUIRE(writer.write_one(atom, stream, "mlpcfg"));
    }
    GPIO reader;
    std::vector<Atoms> frames;
    REQUIRE(reader.read(frames, stream, "mlpcfg"));
    REQUIRE(frames.size() == 3);
    REQUIRE(frames[0].charges_[0] == Approx(0.1));
    REQUIRE(frames[0].global_attribs_d_.at("total_charge") == Approx(0.8));
    REQUIRE_FALSE(frames[1].has_charges());
    REQUIRE_FALSE(frames[1].has_velo());
    REQUIRE(frames[1].global_attribs_d_.count("total_charge") == 0);
    REQUIRE(frames[2].has_charges());
    REQUIRE(frames[2].charges_[0] == 0.0);
    REQUIRE(frames[2].has_velo());
    REQUIRE(frames[2].velocities_[0].isZero());
    REQUIRE(frames[2].global_attribs_d_.at("total_charge") == 0.0);
}

TEST_CASE("CFG promotes only complete finite total charge Features", "[gpio][mlpcfg]") {
    const std::string value = GENERATE("0", "8e-1 \t", "first frame", "0.8 note", "nan", "inf", "-inf", "1e999");
    std::string text = simple_cfg();
    text.insert(text.find("END_CFG"), "Feature total_charge " + value + "\n");
    GPIO io;
    std::istringstream source(text);
    Atoms atom;
    REQUIRE(io.read_one(atom, source, "mlpcfg"));
    const bool numeric = value == "0" || value == "8e-1 \t";
    REQUIRE(atom.global_attribs_d_.count("total_charge") == (numeric ? 1 : 0));
    if (numeric) {
        REQUIRE(atom.global_attribs_d_.at("total_charge") == Approx(value == "0" ? 0.0 : 0.8));
        REQUIRE(atom.features_.count("total_charge") == 0);
    } else {
        REQUIRE(atom.features_.at("total_charge") == value);
        std::stringstream written;
        REQUIRE(io.write_one(atom, written, "mlpcfg"));
        Atoms restored;
        REQUIRE(io.read_one(restored, written, "mlpcfg"));
        REQUIRE(restored.features_.at("total_charge") == value);
        REQUIRE(restored.global_attribs_d_.count("total_charge") == 0);
    }
}

TEST_CASE("CFG rejects mis-sized charge velocity and magnetic arrays before writing", "[gpio][mlpcfg][errors]") {
    const std::string field = GENERATE("charge", "velocity", "magnetic moment");
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity();
    atom.pos_ = {{0.0, 0.0, 0.0}};
    atom.chemmap_ = {0};
    if (field == "charge")
        atom.charges_ = {0.1, 0.2};
    else if (field == "velocity")
        atom.velocities_ = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    else
        atom.magmoms_ = {Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero()};
    GPIO io;
    std::stringstream output;
    REQUIRE_FALSE(io.write_one(atom, output, "mlpcfg"));
    REQUIRE(output.str().empty());
}

TEST_CASE("CFG retains coordinate aliases and ordinary scalar attributes", "[gpio][mlpcfg]") {
    const bool lossless = GENERATE(false, true);
    std::istringstream source(
        "BEGIN_CFG\nSize\n1\nSupercell\n5 0 0\n0 5 0\n0 0 5\n"
        "AtomData: id type x y z score\n1 1 1 2 3 9\nEND_CFG\n");
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, source, "mlpcfg"));
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3)));
    REQUIRE(atom.local_attribs_d_.at("score")[0] == 9);
    io.lossless = lossless;
    std::stringstream written;
    REQUIRE(io.write_one(atom, written, "mlpcfg"));
    Atoms restored;
    REQUIRE(io.read_one(restored, written, "mlpcfg"));
    REQUIRE(restored.pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3)));
    REQUIRE(restored.local_attribs_d_.at("score")[0] == 9);
}

TEST_CASE("CFG rejects scalar magnetic moments from valid BIN before writing", "[gpio][mlpcfg][errors]") {
    const bool lossless = GENERATE(false, true);
    const bool vector_also_present = GENERATE(false, true);
    Atoms original;
    original.cell_ = 10 * Eigen::Matrix3d::Identity();
    original.pos_ = {{1, 2, 3}};
    original.chemmap_ = {1};
    original.magmom_scalar_ = {1.5};
    if (vector_also_present) original.magmoms_ = {{0.1, 0.2, 0.3}};
    GPIO io;
    std::stringstream binary(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(io.write_one(original, binary, "bin"));
    Atoms from_binary;
    REQUIRE(io.read_one(from_binary, binary, "bin"));
    REQUIRE(from_binary.magmom_scalar_ == std::vector<double>{1.5});
    REQUIRE(from_binary.magmoms_.size() == (vector_also_present ? 1 : 0));
    io.lossless = lossless;
    std::stringstream cfg;
    REQUIRE_FALSE(io.write_one(from_binary, cfg, "mlpcfg"));
    REQUIRE(cfg.str().empty());
    REQUIRE(from_binary.magmom_scalar_ == original.magmom_scalar_);
}

TEST_CASE("CFG vector magnetic moments retain their components in both precisions", "[gpio][mlpcfg]") {
    const bool lossless = GENERATE(false, true);
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1, 2, 3}, {4, 5, 6}};
    atom.chemmap_ = {1, 2};
    atom.magmoms_ = {{0.1, -0.2, 0.3}, {-0.4, 0.5, -0.6}};
    GPIO io;
    io.lossless = lossless;
    std::stringstream cfg;
    REQUIRE(io.write_one(atom, cfg, "mlpcfg"));
    REQUIRE(cfg.str().find("magmom_x magmom_y magmom_z") != std::string::npos);
    Atoms restored;
    REQUIRE(io.read_one(restored, cfg, "mlpcfg"));
    REQUIRE(restored.magmom_scalar_.empty());
    REQUIRE(restored.magmoms_.size() == 2);
    REQUIRE(restored.magmoms_[0].isApprox(Eigen::Vector3d(0.1, -0.2, 0.3)));
    REQUIRE(restored.magmoms_[1].isApprox(Eigen::Vector3d(-0.4, 0.5, -0.6)));
}

TEST_CASE("CFG canonical free energy survives BIN and native precedence", "[gpio][mlpcfg]") {
    const bool lossless = GENERATE(false, true);
    std::istringstream source(
        "BEGIN_CFG\nSize\n1\nSupercell\n10 0 0\n0 10 0\n0 0 10\n"
        "AtomData: id type cartes_x cartes_y cartes_z\n1 1 1 2 3\n"
        "Energy\n-1\nFeature fenergy -1.2\nEND_CFG\n");
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, source, "mlpcfg"));
    REQUIRE(atom.has_energy());
    REQUIRE(atom.energy_ == -1);
    REQUIRE(atom.has_fenergy());
    REQUIRE(atom.fenergy_ == Approx(-1.2));
    REQUIRE(atom.features_.count("fenergy") == 0);
    atom.features_["fenergy"] = "stale free energy text";
    std::stringstream binary(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(io.write_one(atom, binary, "bin"));
    Atoms from_binary;
    REQUIRE(io.read_one(from_binary, binary, "bin"));
    REQUIRE(from_binary.has_fenergy());
    REQUIRE(from_binary.fenergy_ == Approx(-1.2));
    REQUIRE(from_binary.features_.at("fenergy") == "stale free energy text");
    io.lossless = lossless;
    std::stringstream cfg;
    REQUIRE(io.write_one(from_binary, cfg, "mlpcfg"));
    const auto text = cfg.str();
    const auto first = text.find("Feature fenergy ");
    REQUIRE(first != std::string::npos);
    REQUIRE(text.find("Feature fenergy ", first + 1) == std::string::npos);
    REQUIRE(text.find("stale free energy text") == std::string::npos);
    Atoms restored;
    REQUIRE(io.read_one(restored, cfg, "mlpcfg"));
    REQUIRE(restored.has_fenergy());
    REQUIRE(restored.fenergy_ == Approx(-1.2));
    REQUIRE(restored.energy_ == -1);
    std::stringstream extxyz;
    REQUIRE(io.write_one(restored, extxyz, "extxyz"));
    Atoms final;
    REQUIRE(io.read_one(final, extxyz, "extxyz"));
    REQUIRE(final.has_fenergy());
    REQUIRE(final.fenergy_ == Approx(-1.2));
    REQUIRE(final.energy_ == -1);
}

TEST_CASE("CFG distinguishes absent and zero free energy in mixed frames", "[gpio][mlpcfg]") {
    const bool lossless = GENERATE(false, true);
    GPIO io;
    io.lossless = lossless;
    std::stringstream cfg;
    for (int frame = 0; frame < 3; ++frame) {
        Atoms atom;
        atom.cell_ = 10 * Eigen::Matrix3d::Identity();
        atom.pos_ = {{1, 2, 3}};
        atom.chemmap_ = {1};
        atom.energy_ = -1 - frame;
        atom.have_energy = true;
        if (frame != 1) {
            atom.have_fenergy = true;
            atom.fenergy_ = frame == 0 ? -1.2 : 0;
        }
        REQUIRE(io.write_one(atom, cfg, "mlpcfg"));
    }
    std::vector<Atoms> restored;
    REQUIRE(io.read(restored, cfg, "mlpcfg"));
    REQUIRE(restored.size() == 3);
    REQUIRE(restored[0].has_fenergy());
    REQUIRE(restored[0].fenergy_ == Approx(-1.2));
    REQUIRE_FALSE(restored[1].has_fenergy());
    REQUIRE(restored[1].features_.count("fenergy") == 0);
    REQUIRE(restored[2].has_fenergy());
    REQUIRE(restored[2].fenergy_ == 0);
    for (int frame = 0; frame < 3; ++frame)
        REQUIRE(restored[frame].energy_ == -1 - frame);
}

TEST_CASE("CFG promotes only complete finite free energy Features", "[gpio][mlpcfg]") {
    const std::string value = GENERATE("0", "-1.2e0 \t", "old  free energy", "-1.2 note", "nan", "inf", "1e999");
    std::string text = simple_cfg();
    text.insert(text.find("END_CFG"), "Feature fenergy " + value + "\n");
    GPIO io;
    std::istringstream source(text);
    Atoms atom;
    REQUIRE(io.read_one(atom, source, "mlpcfg"));
    const bool numeric = value == "0" || value == "-1.2e0 \t";
    REQUIRE(atom.has_fenergy() == numeric);
    REQUIRE(atom.energy_ == -2);
    if (numeric) {
        REQUIRE(atom.fenergy_ == Approx(value == "0" ? 0 : -1.2));
        REQUIRE(atom.features_.count("fenergy") == 0);
    } else {
        REQUIRE(atom.features_.at("fenergy") == value);
        std::stringstream cfg;
        REQUIRE(io.write_one(atom, cfg, "mlpcfg"));
        Atoms restored;
        REQUIRE(io.read_one(restored, cfg, "mlpcfg"));
        REQUIRE_FALSE(restored.has_fenergy());
        REQUIRE(restored.features_.at("fenergy") == value);
    }
}

TEST_CASE("CFG preserves declared full Hessians and HVP-only labels", "[gpio][mlpcfg][hessian]") {
    const bool lossless = GENERATE(false, true);
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1, 2, 3}, {4, 5, 6}};
    atom.chemmap_ = {0, 0};
    atom.hessian_ = Eigen::MatrixXd::Identity(6, 6);
    atom.have_hessian = true;
    GPIO io;
    io.lossless = lossless;
    std::stringstream rejected;
    REQUIRE(io.write_one(atom, rejected, "mlpcfg"));
    Atoms dense;
    REQUIRE(io.read_one(dense,rejected,"mlpcfg"));
    REQUIRE(dense.have_hessian);
    REQUIRE(dense.hessian_.isApprox(atom.hessian_));
    REQUIRE(atom.have_hessian);
    REQUIRE(atom.hessian_.isApprox(Eigen::MatrixXd::Identity(6, 6)));

    atom.have_hessian = false;
    atom.hvp_directions_ = Eigen::MatrixXd::Ones(6, 1);
    atom.hvp_targets_ = 2 * atom.hvp_directions_;
    std::stringstream supported;
    REQUIRE(io.write_one(atom, supported, "mlpcfg"));
    Atoms restored;
    REQUIRE(io.read_one(restored, supported, "mlpcfg"));
    REQUIRE_FALSE(restored.have_hessian);
    REQUIRE(restored.hvp_directions_.isApprox(atom.hvp_directions_));
    REQUIRE(restored.hvp_targets_.isApprox(atom.hvp_targets_));
}
