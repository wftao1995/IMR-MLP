// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "io/gpio.h"

#include <sstream>

namespace {
struct MSIAtomRecord {
    int id;
    const char *acl;
};

std::string msi_model(const std::vector<MSIAtomRecord> &atoms,
                      const std::vector<std::array<int, 2>> &bonds,
                      bool with_charges = true)
{
    std::ostringstream output;
    output << "(1 Model\n";
    for (size_t i = 0; i < atoms.size(); ++i) {
        output << "  (" << atoms[i].id << " Atom\n"
               << "    (A C ACL \"" << atoms[i].acl << "\")\n"
               << "    (A D XYZ (" << i << " 0 0))\n";
        if (with_charges)
            output << "    (A D Charge " << i + 0.25 << ")\n";
        output << "  )\n";
    }
    for (size_t i = 0; i < bonds.size(); ++i)
        output << "  (" << i + 1 << " Bond\n"
               << "    (A I Atom1 " << bonds[i][0] << ")\n"
               << "    (A I Atom2 " << bonds[i][1] << ")\n"
               << "  )\n";
    output << ")\n";
    return output.str();
}
}

TEST_CASE("GPIO CAR reads a periodic crystal", "[gpio][car]")
{
    std::istringstream input(
        "!BIOSYM archive 3\n"
        "PBC=ON\n"
        "Materials Studio Generated CAR File\n"
        "!DATE Sat Sep 28 14:53:25 2013\n"
        "PBC    9.4214   18.8428    6.8814   90.0000   90.0000  90.0000 (P1)\n"
        "O1      -1.624219154    8.097696078    1.776747746 XXXX 1      o       O  -0.587\n"
        "Ca1     -2.382573017    9.453046931    0.006968682 XXXX 1      ca+     Ca  1.255\n"
        "P1      -0.385120417    7.101979887    1.743525377 XXXX 1      p       P   0.480\n"
        "end\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "car"));
    REQUIRE(atom.size() == 3);
    REQUIRE(atom.chemmap_ == std::vector<int>({io.getrevmap().at("O"),
                                               io.getrevmap().at("Ca"),
                                               io.getrevmap().at("P")}));
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(9.4214).margin(1e-8));
    REQUIRE(atom.cell_(1, 1) == Catch::Approx(18.8428).margin(1e-8));
    REQUIRE(atom.cell_(2, 2) == Catch::Approx(6.8814).margin(1e-8));
    REQUIRE(atom.pos_[0][0] == Catch::Approx(-1.624219154));
    REQUIRE(atom.pos_[0][1] == Catch::Approx(8.097696078));
    REQUIRE(atom.charges_.size() == 3);
    REQUIRE(atom.charges_[0] == Catch::Approx(-0.587));
    REQUIRE(atom.charges_[1] == Catch::Approx(1.255));
    REQUIRE(atom.charges_[2] == Catch::Approx(0.480));
}
TEST_CASE("GPIO CAR reads a triclinic cell", "[gpio][car]")
{
    std::istringstream input(
        "!BIOSYM archive 3\n"
        "PBC=ON\n"
        "comment\n"
        "!DATE Mon Jan 01 00:00:00 2001\n"
        "PBC 10.0 8.0 6.0 80.0 70.0 60.0\n"
        "Fe1 0.0 0.0 0.0 XXXX 1 fe Fe 0.0\n"
        "end\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "car"));
    REQUIRE(atom.size() == 1);
    // a along x
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(10.0));
    REQUIRE(atom.cell_(1, 0) == Catch::Approx(8.0 * std::cos(60.0 * M_PI / 180.0)));
    REQUIRE(atom.cell_(2, 0) == Catch::Approx(6.0 * std::cos(70.0 * M_PI / 180.0)));
}
TEST_CASE("GPIO CAR reads a non-periodic file", "[gpio][car]")
{
    std::istringstream input(
        "!BIOSYM archive 3\n"
        "PBC=OFF\n"
        "comment\n"
        "!DATE Mon Jan 01 00:00:00 2001\n"
        "C1 0.0 0.0 0.0 XXXX 1 c C 0.0\n"
        "H1 1.0 0.0 0.0 XXXX 1 h H 0.0\n"
        "end\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "car"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.pbc_ == std::array<bool, 3>{false, false, false});
    REQUIRE(atom.cell_.isZero());
}
TEST_CASE("GPIO CAR reads the element column and skips unknowns", "[gpio][car]")
{
    std::istringstream input(
        "!BIOSYM archive 3\n"
        "PBC=ON\n"
        "comment\n"
        "!DATE Mon Jan 01 00:00:00 2001\n"
        "PBC 5.0 5.0 5.0 90.0 90.0 90.0\n"
        "O1 1.0 1.0 1.0 XXXX 1 o O\n"
        "Zz1 2.0 2.0 2.0 XXXX 1 zz Zz\n"
        "end\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "car"));
    // Zz1 cannot be resolved and is skipped with a warning
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.chemmap_ == std::vector<int>{io.getrevmap().at("O")});
    REQUIRE(atom.charges_.empty()); // no charge column
}
TEST_CASE("GPIO CAR rejects non-BIOSYM files", "[gpio][car]")
{
    std::istringstream input("not a car file\n");
    Atoms atom;
    GPIO io;
    REQUIRE_FALSE(io.read_one(atom, input, "car"));
}
TEST_CASE("GPIO CAR aligns optional charges with retained atoms", "[gpio][car]")
{
    bool retained_charges = true;
    SECTION("mixed charge columns") {}
    SECTION("only skipped atoms have charges") { retained_charges = false; }

    std::stringstream input;
    input << "!BIOSYM archive 3\n"
          << "PBC=ON\n"
          << "PBC 5 5 5 90 90 90\n"
          << "Zz1 4 0 0 XXXX 1 zz Zz 9.0\n"
          << "C1 0 0 0 XXXX 1 c C\n"
          << "H1 1 0 0 XXXX 1 h H" << (retained_charges ? " 0.4" : "") << '\n'
          << "Zz2 4 1 0 XXXX 1 zz Zz -9.0\n"
          << "O1 2 0 0 XXXX 1 o O\n"
          << "Si1 3 0 0 XXXX 1 si Si" << (retained_charges ? " -0.2" : "") << '\n'
          << "end\n";
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "car"));
    REQUIRE(atom.chemmap_ == std::vector<int>{io.getrevmap().at("C"),
        io.getrevmap().at("H"), io.getrevmap().at("O"), io.getrevmap().at("Si")});
    const std::vector<double> expected = retained_charges
        ? std::vector<double>{0.0, 0.4, 0.0, -0.2} : std::vector<double>{};
    REQUIRE(atom.charges_.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        REQUIRE(atom.charges_[i] == Catch::Approx(expected[i]));

    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    Atoms reread;
    REQUIRE(io.read_one(reread, output, "extxyz"));
    REQUIRE(reread.chemmap_ == atom.chemmap_);
    REQUIRE(reread.charges_.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i)
        REQUIRE(reread.charges_[i] == Catch::Approx(expected[i]));
    for (size_t i = 0; i < reread.size(); ++i)
        REQUIRE(reread.pos_[i][0] == Catch::Approx(i));
}
TEST_CASE("GPIO MSI reads a periodic crystal model", "[gpio][msi]")
{
    std::istringstream input(
        "# MSI CERIUS2 DataModel File Version 4 0\n"
        "(1 Model\n"
        "  (A I PeriodicType 100)\n"
        "  (A D A3 (5.0 0.0 0.0))\n"
        "  (A D B3 (0.0 5.0 0.0))\n"
        "  (A D C3 (0.0 0.0 5.0))\n"
        "  (A C SpaceGroup \"1 1\")\n"
        "  (1 Atom\n"
        "    (A C ACL \"6 C\")\n"
        "    (A D XYZ (0.1 0.2 0.3))\n"
        "  )\n"
        "  (2 Atom\n"
        "    (A D XYZ (2.0 2.0 2.0))\n"
        "    (A C ACL \"14 Si\")\n"
        "  )\n"
        ")\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "msi"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.chemmap_ ==
            std::vector<int>({io.getrevmap().at("C"), io.getrevmap().at("Si")}));
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(5.0));
    REQUIRE(atom.cell_(1, 1) == Catch::Approx(5.0));
    REQUIRE(atom.cell_(2, 2) == Catch::Approx(5.0));
    REQUIRE(atom.pos_[0][0] == Catch::Approx(0.1));
    REQUIRE(atom.pos_[0][1] == Catch::Approx(0.2));
    REQUIRE(atom.pos_[0][2] == Catch::Approx(0.3));
    REQUIRE(atom.pos_[1][0] == Catch::Approx(2.0));
}
TEST_CASE("GPIO MSI reads bonds and charges", "[gpio][msi]")
{
    std::istringstream input(
        "# MSI CERIUS2 DataModel File Version 4 0\n"
        "(1 Model\n"
        "  (1 Atom\n"
        "    (A C ACL \"6 C\")\n"
        "    (A D XYZ (0.0 0.0 0.0))\n"
        "    (A D Charge -0.5)\n"
        "  )\n"
        "  (2 Atom\n"
        "    (A C ACL \"1 H\")\n"
        "    (A D XYZ (1.0 0.0 0.0))\n"
        "    (A D Charge 0.5)\n"
        "  )\n"
        "  (1 Bond\n"
        "    (A I Atom1 1)\n"
        "    (A I Atom2 2)\n"
        "    (A I Type 1)\n"
        "  )\n"
        ")\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "msi"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.charges_.size() == 2);
    REQUIRE(atom.charges_[0] == Catch::Approx(-0.5));
    REQUIRE(atom.charges_[1] == Catch::Approx(0.5));
    REQUIRE(atom.bonds_.size() == 1);
    REQUIRE(atom.bonds_[0][0] == 0);
    REQUIRE(atom.bonds_[0][1] == 1);
}
TEST_CASE("GPIO MSI reads multiple models", "[gpio][msi]")
{
    std::istringstream input(
        "# MSI CERIUS2 DataModel File Version 4 0\n"
        "(1 Model\n"
        "  (A D A3 (5.0 0.0 0.0))\n"
        "  (A D B3 (0.0 5.0 0.0))\n"
        "  (A D C3 (0.0 0.0 5.0))\n"
        "  (1 Atom\n"
        "    (A C ACL \"6 C\")\n"
        "    (A D XYZ (0.0 0.0 0.0))\n"
        "  )\n"
        ")\n"
        "(2 Model\n"
        "  (A D A3 (7.0 0.0 0.0))\n"
        "  (A D B3 (0.0 7.0 0.0))\n"
        "  (A D C3 (0.0 0.0 7.0))\n"
        "  (1 Atom\n"
        "    (A C ACL \"8 O\")\n"
        "    (A D XYZ (0.5 0.5 0.5))\n"
        "  )\n"
        ")\n");
    std::vector<Atoms> atoms;
    GPIO io;
    REQUIRE(io.read(atoms, input, "msi"));
    REQUIRE(atoms.size() == 2);
    REQUIRE(atoms[0].cell_(0, 0) == Catch::Approx(5.0));
    REQUIRE(atoms[1].cell_(0, 0) == Catch::Approx(7.0));
    REQUIRE(atoms[0].chemmap_ == std::vector<int>{io.getrevmap().at("C")});
    REQUIRE(atoms[1].chemmap_ == std::vector<int>{io.getrevmap().at("O")});
}

TEST_CASE("GPIO MSI remaps bonds after filtering any atom position", "[gpio][msi]")
{
    const std::vector<MSIAtomRecord> records{{41, "6 C"}, {7, "1 H"},
                                            {29, "8 O"}, {13, "7 N"}};
    std::vector<std::array<int, 2>> bonds;
    for (const auto &a : records)
        for (const auto &b : records)
            bonds.push_back({a.id, b.id});
    bonds.push_back({999, 41});
    bonds.push_back({41, 999});
    bonds.push_back({999, 998});

    for (size_t skipped = 0; skipped < records.size(); ++skipped) {
        DYNAMIC_SECTION("unknown atom at position " << skipped) {
            auto filtered = records;
            filtered[skipped].acl = "0 Zz";
            std::istringstream input(msi_model(filtered, bonds));
            Atoms atom;
            GPIO io;
            REQUIRE(io.read_one(atom, input, "msi"));
            REQUIRE(atom.size() == 3);
            REQUIRE(atom.charges_.empty());
            std::vector<std::array<int, 2>> expected;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    expected.push_back({a, b});
            REQUIRE(atom.bonds_ == expected);
            size_t retained = 0;
            for (size_t old = 0; old < records.size(); ++old) {
                if (old == skipped) continue;
                REQUIRE(atom.pos_[retained][0] == Catch::Approx(old));
                REQUIRE(atom.chemmap_[retained] ==
                        io.getrevmap().at(std::string(records[old].acl).substr(2)));
                ++retained;
            }
            for (const auto &bond : atom.bonds_)
                for (int endpoint : bond) {
                    REQUIRE(endpoint >= 0);
                    REQUIRE(endpoint < atom.size());
                }
        }
    }
}

TEST_CASE("GPIO MSI remaps custom element maps and drops every dangling bond", "[gpio][msi]")
{
    const std::vector<MSIAtomRecord> records{{41, "6 C"}, {7, "1 H"},
                                            {29, "8 O"}, {13, "7 N"}};
    std::istringstream input(msi_model(records, {{41, 29}, {29, 41}, {7, 29},
                                                {13, 41}, {7, 13}, {41, 999}}));
    GPIO io;
    io.setmap({"O", "C"});
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "msi"));
    REQUIRE(atom.chemmap_ == std::vector<int>{1, 0});
    REQUIRE(atom.pos_[0][0] == Catch::Approx(0.0));
    REQUIRE(atom.pos_[1][0] == Catch::Approx(2.0));
    REQUIRE(atom.bonds_ == std::vector<std::array<int, 2>>{{0, 1}, {1, 0}});
    REQUIRE(atom.charges_.empty());

    std::istringstream dangling(msi_model({{41, "0 Zz"}, {7, "6 C"}},
                                         {{41, 7}, {7, 41}, {999, 7}}, false));
    REQUIRE(io.read_one(atom, dangling, "msi"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.bonds_.empty());
    REQUIRE(atom.charges_.empty());
}

TEST_CASE("GPIO MSI keeps bond maps and charges local to each model", "[gpio][msi]")
{
    const auto filtered = msi_model({{9, "0 Zz"}, {4, "6 C"}, {2, "1 H"}},
                                     {{4, 2}, {9, 2}});
    const auto complete = msi_model({{9, "8 O"}, {4, "1 H"}, {2, "6 C"}},
                                     {{2, 9}, {9, 4}});
    std::istringstream input(filtered + complete + filtered);
    std::vector<Atoms> atoms;
    GPIO io;
    REQUIRE(io.read(atoms, input, "msi"));
    REQUIRE(atoms.size() == 3);
    for (int index : {0, 2}) {
        REQUIRE(atoms[index].size() == 2);
        REQUIRE(atoms[index].bonds_ == std::vector<std::array<int, 2>>{{0, 1}});
        REQUIRE(atoms[index].charges_.empty());
    }
    REQUIRE(atoms[1].size() == 3);
    REQUIRE(atoms[1].bonds_ == std::vector<std::array<int, 2>>{{2, 0}, {0, 1}});
    REQUIRE(atoms[1].charges_ == std::vector<double>{0.25, 1.25, 2.25});
}

TEST_CASE("GPIO MSI skips unknown records", "[gpio][msi]")
{
    std::istringstream input(
        "# MSI CERIUS2 DataModel File Version 4 0\n"
        "(1 Model\n"
        "  (1 Foo)\n"
        "  (A C Name \"Test\")\n"
        "  (A I Number 42)\n"
        "  (1 Atom\n"
        "    (A C ACL \"26 Fe\")\n"
        "    (A D XYZ (0.25 0.25 0.25))\n"
        "  )\n"
        ")\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "msi"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.chemmap_ == std::vector<int>{io.getrevmap().at("Fe")});
    REQUIRE(atom.pos_[0][0] == Catch::Approx(0.25));
}
TEST_CASE("GPIO XSD reads a flat periodic structure with fractional XYZ", "[gpio][xsd]")
{
    std::istringstream input(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<XSD Version=\"18.1\">\n"
        "  <AtomisticTreeRoot ID=\"1\" Name=\"CHA\"/>\n"
        "  <SpaceGroup ID=\"2\" Name=\"R-3M\" ITNumber=\"166\" System=\"Trigonal\"\n"
        "    AVector=\"9.36631997375662,-0.742504027486964,-0.689977326669816\"\n"
        "    BVector=\"0,9.39570444839697,-0.689977326669816\"\n"
        "    CVector=\"0,0,9.42100476557278\"\n"
        "    Operators=\"1,0,0,0,0,1,0,0,0,0,1,0\"/>\n"
        "  <Atom3d ID=\"8\" Name=\"Si1\" XYZ=\"0.104399914,0.333800129,0.8748990473\" Components=\"Si\"/>\n"
        "  <Atom3d ID=\"4\" Name=\"O4\" XYZ=\"0.2515000353,0.2515000353,0.8945997644\" Components=\"O\"/>\n"
        "  <Bond ID=\"10\" Connects=\"8,4\"/>\n"
        "</XSD>\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "xsd"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.chemmap_ ==
            std::vector<int>({io.getrevmap().at("Si"), io.getrevmap().at("O")}));
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(9.36631997375662));
    REQUIRE(atom.cell_(1, 1) == Catch::Approx(9.39570444839697));
    REQUIRE(atom.cell_(2, 2) == Catch::Approx(9.42100476557278));
    // fractional XYZ is converted to Cartesian
    REQUIRE(atom.pos_[0][0] == Catch::Approx(0.104399914 * 9.36631997375662));
    Eigen::RowVector3d frac1(0.2515000353, 0.2515000353, 0.8945997644);
    Eigen::RowVector3d exp1 = frac1 * atom.cell_;
    REQUIRE(atom.pos_[1][0] == Catch::Approx(exp1[0]));
    REQUIRE(atom.pos_[1][1] == Catch::Approx(exp1[1]));
    REQUIRE(atom.pos_[1][2] == Catch::Approx(exp1[2]));
    REQUIRE(atom.bonds_.size() == 1);
    REQUIRE(atom.bonds_[0][0] == 0);
    REQUIRE(atom.bonds_[0][1] == 1);
}
TEST_CASE("GPIO XSD reads CartesianXYZ coordinates", "[gpio][xsd]")
{
    std::istringstream input(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<XSD Version=\"18.1\">\n"
        "  <SpaceGroup Name=\"P1\" ITNumber=\"1\" System=\"Triclinic\" Lattice=\"3D Triclinic\"\n"
        "    AVector=\"10,0,0\" BVector=\"0,10,0\" CVector=\"0,0,10\"\n"
        "    Operators=\"1,0,0,0,0,1,0,0,0,0,1,0\"/>\n"
        "  <Atom3d ID=\"1\" Name=\"C1\" CartesianXYZ=\"2,3,4\" Components=\"C\"/>\n"
        "</XSD>\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "xsd"));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.chemmap_ == std::vector<int>{io.getrevmap().at("C")});
    REQUIRE(atom.pos_[0][0] == Catch::Approx(2.0));
    REQUIRE(atom.pos_[0][1] == Catch::Approx(3.0));
    REQUIRE(atom.pos_[0][2] == Catch::Approx(4.0));
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(10.0));
}
TEST_CASE("GPIO XSD reads the nested IdentityMapping layout", "[gpio][xsd]")
{
    std::istringstream input(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<XSD Version=\"4.0\">\n"
        "  <AtomisticTreeRoot ID=\"1\">\n"
        "    <SymmetrySystem ID=\"2\">\n"
        "      <MappingSet ID=\"8\">\n"
        "        <MappingFamily ID=\"10\">\n"
        "          <IdentityMapping ID=\"11\" Element=\"1,0,0,0,0,1,0,0,0,0,1,0\">\n"
        "            <SpaceGroup ID=\"9\" AVector=\"5,0,0\" BVector=\"0,5,0\" CVector=\"0,0,5\"\n"
        "              ITNumber=\"1\" LongName=\"P 1\"/>\n"
        "            <Atom3d ID=\"4\" Name=\"Fe1\" XYZ=\"0.0,0.0,0.0\" Components=\"Fe\"/>\n"
        "            <Atom3d ID=\"5\" Name=\"Fe2\" XYZ=\"0.5,0.5,0.5\" Components=\"Fe\" Charge=\"0.5\"/>\n"
        "          </IdentityMapping>\n"
        "        </MappingFamily>\n"
        "      </MappingSet>\n"
        "    </SymmetrySystem>\n"
        "  </AtomisticTreeRoot>\n"
        "</XSD>\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "xsd"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.chemmap_ == std::vector<int>({io.getrevmap().at("Fe"), io.getrevmap().at("Fe")}));
    REQUIRE(atom.cell_(0, 0) == Catch::Approx(5.0));
    REQUIRE(atom.pos_[1][0] == Catch::Approx(2.5));
    REQUIRE(atom.charges_.size() == 2);
    REQUIRE(atom.charges_[0] == Catch::Approx(0.0));
    REQUIRE(atom.charges_[1] == Catch::Approx(0.5));
}
TEST_CASE("GPIO XSD reads a non-periodic Molecule", "[gpio][xsd]")
{
    std::istringstream input(
        "<?xml version=\"1.0\" encoding=\"utf-8\"?>\n"
        "<XSD Version=\"18.1\">\n"
        "  <AtomisticTreeRoot ID=\"1\">\n"
        "    <Molecule ID=\"2\">\n"
        "      <Atom3d ID=\"3\" Name=\"O1\" XYZ=\"0.0,0.0,0.0\" Components=\"O\"/>\n"
        "      <Atom3d ID=\"4\" Name=\"H1\" XYZ=\"0.96,0.0,0.0\" Components=\"H\"/>\n"
        "    </Molecule>\n"
        "  </AtomisticTreeRoot>\n"
        "</XSD>\n");
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "xsd"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.chemmap_ == std::vector<int>({io.getrevmap().at("O"), io.getrevmap().at("H")}));
    REQUIRE(atom.pos_[1][0] == Catch::Approx(0.96));
    REQUIRE(atom.pbc_ == std::array<bool, 3>{false, false, false});
}
