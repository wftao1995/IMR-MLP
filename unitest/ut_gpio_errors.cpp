// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "io/gpio.h"
#include "io/gpio_binary.h"
#include "io/gpio_tokenizer.h"

#include <sstream>

namespace {

Atoms make_atom()
{
    Atoms atom;
    atom.cell_ = 5.0 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1.0, 2.0, 3.0}};
    atom.chemmap_ = {1};
    atom.pbc_ = {{true, true, true}};
    return atom;
}

std::string lammps_dump(std::string_view rows)
{
    return "ITEM: TIMESTEP\n0\n"
           "ITEM: NUMBER OF ATOMS\n2\n"
           "ITEM: BOX BOUNDS pp pp pp\n0 5\n0 5\n0 5\n"
           "ITEM: ATOMS id type x y z q c_desc[1] c_desc[2]\n" + std::string(rows);
}

class FailingStreamBuf : public std::streambuf {
protected:
    std::streamsize xsputn(const char *, std::streamsize) override { return 0; }
    int_type overflow(int_type) override { return traits_type::eof(); }
};

} // namespace

TEST_CASE("GPIO reads charges and bracket vectors from LAMMPS dump", "[gpio][lammps]")
{
    std::istringstream input(lammps_dump("2 2 2 2 2 -0.2 20 21\n1 1 1 1 1 0.1 10 11\n"));
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "lammpstrj"));
    REQUIRE(atom.charges_ == std::vector<double>{0.1, -0.2});
    REQUIRE(atom.local_attribs_v_.at("c_desc")(0, 0) == Catch::Approx(10.0));
    REQUIRE(atom.local_attribs_v_.at("c_desc")(1, 1) == Catch::Approx(21.0));
}

TEST_CASE("GPIO rejects invalid LAMMPS atom ids and vector labels", "[gpio][lammps][errors]")
{
    SECTION("duplicate id") {
        std::istringstream input(lammps_dump("1 1 1 1 1 0.1 10 11\n1 2 2 2 2 -0.2 20 21\n"));
        Atoms atom;
        GPIO io;
        REQUIRE_FALSE(io.read_one(atom, input, "lammpstrj"));
    }
    SECTION("zero vector component") {
        std::string dump = lammps_dump("1 1 1 1 1 0.1 10 11\n2 2 2 2 2 -0.2 20 21\n");
        const auto pos = dump.find("c_desc[1]");
        dump.replace(pos, std::string("c_desc[1]").size(), "c_desc[0]");
        std::istringstream input(dump);
        Atoms atom;
        GPIO io;
        REQUIRE_FALSE(io.read_one(atom, input, "lammpstrj"));
    }
}

TEST_CASE("GPIO writes CIF loop_ tags one per line", "[gpio][cif]")
{
    Atoms atom = make_atom();
    atom.chemmap_ = {1, 2};
    atom.pos_ = {{0.0, 0.0, 0.0}, {0.5, 0.5, 0.5}};
    std::ostringstream output;
    GPIO io;
    REQUIRE(io.write_one(atom, output, "cif"));
    const std::string cif = output.str();
    // ASE's CIF parser requires each _atom_site_* tag on its own line
    REQUIRE(cif.find("loop_\n_atom_site_label\n_atom_site_fract_x\n"
                     "_atom_site_fract_y\n_atom_site_fract_z\n"
                     "_atom_site_type_symbol\n_atom_site_occupancy\n") != std::string::npos);
    REQUIRE(cif.find("_atom_site_label _atom_site_fract_x") == std::string::npos);
}

TEST_CASE("GPIO lammpstrj vector labels round-trip with 1-based indices", "[gpio][lammps]")
{
    Atoms atom = make_atom();
    atom.chemmap_ = {1, 2};
    atom.pos_ = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
    Eigen::MatrixXd desc(2, 2);
    desc << 10, 11, 20, 21;
    atom.local_attribs_v_["desc"] = desc;

    std::ostringstream output;
    GPIO io;
    REQUIRE(io.seq_write_init(output, "lammpstrj"));
    REQUIRE(io.seq_write(output, atom));
    const std::string dump = output.str();
    // LAMMPS vector columns are 1-based (v_name[N]); 0-based tags must not
    // be written because the reader rejects them
    REQUIRE(dump.find("desc[1] desc[2]") != std::string::npos);
    REQUIRE(dump.find("desc[0]") == std::string::npos);

    std::istringstream input(dump);
    Atoms back;
    REQUIRE(io.read_one(back, input, "lammpstrj"));
    REQUIRE(back.size() == 2);
    REQUIRE(back.local_attribs_v_.at("desc")(0, 0) == Catch::Approx(10.0));
    REQUIRE(back.local_attribs_v_.at("desc")(0, 1) == Catch::Approx(11.0));
    REQUIRE(back.local_attribs_v_.at("desc")(1, 0) == Catch::Approx(20.0));
    REQUIRE(back.local_attribs_v_.at("desc")(1, 1) == Catch::Approx(21.0));
}

TEST_CASE("GPIO handles duplicate EXTXYZ pbc metadata", "[gpio][extxyz][errors]")
{
    SECTION("a malformed duplicate does not replace a valid value") {
        std::istringstream input(
            "1\n"
            "Lattice=\"5 0 0 0 5 0 0 0 5\" pbc=\"T F T\" pbc=T T T "
            "Properties=species:S:1:pos:R:3\n"
            "Al 1 2 3\n");
        Atoms atom;
        GPIO io;
        REQUIRE(io.read_one(atom, input, "extxyz"));
        REQUIRE(atom.pbc_ == std::array<bool, 3>{true, false, true});
    }

    SECTION("a malformed sole value is rejected") {
        std::istringstream input(
            "1\n"
            "Lattice=\"5 0 0 0 5 0 0 0 5\" pbc=T "
            "Properties=species:S:1:pos:R:3\n"
            "Al 1 2 3\n");
        Atoms atom;
        GPIO io;
        REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    }
}

TEST_CASE("GPIO MLPCFG validates ids, charges, and truncation", "[gpio][mlpcfg][errors]")
{
    const std::string prefix =
        "BEGIN_CFG\nSize\n2\nSupercell\n5 0 0\n0 5 0\n0 0 5\n"
        "AtomData: id type cartes_x cartes_y cartes_z q\n";

    SECTION("valid reordered ids") {
        std::istringstream input(prefix + "2 1 2 2 2 -0.2\n1 0 1 1 1 0.1\nEND_CFG\n");
        Atoms atom;
        GPIO io;
        REQUIRE(io.read_one(atom, input, "mlpcfg"));
        REQUIRE(atom.charges_ == std::vector<double>{0.1, -0.2});
    }
    SECTION("out of range id") {
        std::istringstream input(prefix + "3 1 2 2 2 -0.2\n1 0 1 1 1 0.1\nEND_CFG\n");
        Atoms atom;
        GPIO io;
        REQUIRE_FALSE(io.read_one(atom, input, "mlpcfg"));
    }
    SECTION("truncated frame") {
        std::istringstream input(prefix + "1 0 1 1 1 0.1\n");
        std::vector<Atoms> atoms;
        GPIO io;
        REQUIRE_FALSE(io.read(atoms, input, "mlpcfg"));
    }
}

TEST_CASE("GPIO binary rejects oversized and truncated records", "[gpio][binary][errors]")
{
    using namespace gpio_binary;
    SECTION("oversized atom count") {
        std::stringstream input(std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(write_span(input, MAGIC, sizeof(MAGIC)));
        REQUIRE(write_value(input, VERSION));
        REQUIRE(write_value(input, static_cast<std::int32_t>(MAX_ATOMS + 1)));
        input.seekg(0);
        Atoms atom;
        GPIO io;
        REQUIRE_FALSE(io.read_one(atom, input, "bin"));
    }
    SECTION("truncated header") {
        std::stringstream input(std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(write_span(input, MAGIC, 3));
        input.seekg(0);
        std::vector<Atoms> atoms;
        GPIO io;
        REQUIRE_FALSE(io.read(atoms, input, "bin"));
    }
}

TEST_CASE("GPIO propagates output errors and rejects unknown formats", "[gpio][write][errors]")
{
    Atoms atom = make_atom();
    GPIO io;

    FailingStreamBuf buffer;
    std::ostream failing(&buffer);
    REQUIRE_FALSE(io.write_one(atom, failing, "extxyz"));

    std::ostringstream output;
    REQUIRE_FALSE(io.write_one(atom, output, "not-a-format"));
    REQUIRE(output.str().empty());
}

TEST_CASE("IStreamTokenizer handles tokens crossing its buffer boundary", "[gpio][tokenizer]")
{
    std::istringstream input("123456789 42.125 trailing-token");
    IStreamTokenizer<WS_Default, 5> tokenizer(input);
    std::string_view token;
    REQUIRE(tokenizer.next_token(token));
    REQUIRE(token == "123456789");
    double value = 0.0;
    REQUIRE(tokenizer.next_numeric(value));
    REQUIRE(value == Catch::Approx(42.125));
    REQUIRE(tokenizer.next_token(token));
    REQUIRE(token == "trailing-token");
    REQUIRE_FALSE(tokenizer.next_token(token));
}

TEST_CASE("GPIO sequential LAMMPS writing is available", "[gpio][lammps][sequential]")
{
    GPIO io;
    std::ofstream unused;
    REQUIRE(io.seq_write_init(unused, "lammpstrj"));
    std::ostringstream output;
    Atoms atom = make_atom();
    REQUIRE(io.seq_write(output, atom));
    REQUIRE(output.str().find("ITEM: TIMESTEP") != std::string::npos);
}

TEST_CASE("Line numeric tokens must end at a valid separator", "[gpio][tokenizer]") {
    double value;
    GPIO_LineTokenizer plain("-2junk 3");
    REQUIRE_FALSE(plain.next_numeric(value));
    GPIO_LineTokenizer valid("-2.5e-1\t3 \r");
    REQUIRE(valid.next_numeric(value));
    REQUIRE(value == Catch::Approx(-0.25));
    REQUIRE(valid.next_numeric(value));
    REQUIRE(value == Catch::Approx(3));
    GPIO_LineTokenizer<WS_LmpYaml> yaml("[1.5, -2.5]");
    REQUIRE(yaml.next_numeric(value));
    REQUIRE(value == Catch::Approx(1.5));
    REQUIRE(yaml.next_numeric(value));
    REQUIRE(value == Catch::Approx(-2.5));
}
