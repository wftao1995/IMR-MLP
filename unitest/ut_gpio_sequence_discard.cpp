#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <sstream>

namespace {
std::string ordinary_input(const std::string &format) {
    if (format == "outcar") {
        std::ostringstream out;
        out << " vasp.6.3.2\n INCAR: SYSTEM = discard\n POTCAR: PAW_PBE H 2000\n"
               " VRHFIN =H: s1\n ions per type = 1\n NELM = 60\n EDIFF = .00001\n"
               " EDIFFG = -.1\n NSW = 2\n IBRION = 0\n ISIF = 2\n";
        for (int i = 0; i < 2; ++i)
            out << "--------------------------------------- Iteration 1(1)\n"
                   "------------------------ aborting loop because EDIFF is reached\n"
                   " VOLUME and BASIS-vectors are now :\n ---\n energy-cutoff : 400\n"
                   " volume of cell : 27\n direct lattice vectors reciprocal lattice vectors\n"
                   " 3 0 0 0 0 0\n 0 3 0 0 0 0\n 0 0 3 0 0 0\n"
                   " POSITION                                       TOTAL-FORCE (eV/Angst)\n ---\n"
                << " " << i + 1 << " 2 3 .1 .2 .3\n"
                << "  free  energy   TOTEN  = " << -10 - i << " eV\n";
        return out.str();
    }
    if (format == "n2p2")
        return "begin\nlattice 6 0 0\nlattice 0 6 0\nlattice 0 0 6\n"
               "atom 1 2 3 H .25 -1 .1 .2 .3\nenergy -1\ncharge .25\nend\n"
               "begin\nlattice 6 0 0\nlattice 0 6 0\nlattice 0 0 6\n"
               "atom 2 3 4 H -.25 -2 .2 .3 .4\nenergy -2\ncharge -.25\nend\n";
    GPIO writer;
    writer.lossless = true;
    std::vector<Atoms> atoms;
    for (int i = 0; i < 2; ++i) {
        Atoms atom;
        atom.cell_ = 3. * Eigen::Matrix3d::Identity();
        atom.pos_ = {Eigen::RowVector3d(1 + i, 2 + i, 3 + i)};
        atom.chemmap_ = {1};
        atom.forces_ = {Eigen::Vector3d(.1 + i, .2 + i, .3 + i)};
        atom.energy_ = -9 - i;
        atom.have_energy = true;
        atom.fenergy_ = -10 - i;
        atom.have_fenergy = true;
        atom.stress_ = (i + 1.) * Eigen::Matrix3d::Identity();
        atom.have_stress = true;
        atom.site_energies_ = {-9. - i};
        atom.charges_ = {.25};
        atom.global_attribs_d_["temperature"] = 300 + i;
        atom.global_attribs_i_["keep"] = 7 + i;
        atoms.push_back(atom);
    }
    std::ostringstream output(std::ios::out | std::ios::binary);
    REQUIRE(writer.write(atoms, output, format));
    return output.str();
}
void check_selected(const Atoms &actual, const Atoms &before, const std::string &key) {
    REQUIRE(actual.id == before.id);
    REQUIRE(actual.pos_ == before.pos_);
    REQUIRE(actual.cell_.isApprox(before.cell_));
    REQUIRE(actual.pbc_ == before.pbc_);
    REQUIRE(actual.chemmap_ == before.chemmap_);
    REQUIRE(actual.have_energy == (before.have_energy && key != "energy"));
    REQUIRE(actual.have_fenergy == (before.have_fenergy && key != "fenergy" && key != "free_energy"));
    REQUIRE(actual.have_stress == (before.have_stress && key != "virial"));
    if (actual.have_energy) REQUIRE(actual.energy_ == before.energy_);
    if (actual.have_fenergy) REQUIRE(actual.fenergy_ == before.fenergy_);
    if (actual.have_stress) REQUIRE(actual.stress_.isApprox(before.stress_));
    if (key == "forces") REQUIRE(actual.forces_.empty());
    else REQUIRE(actual.forces_ == before.forces_);
    if (key == "site_en") REQUIRE(actual.site_energies_.empty());
    else REQUIRE(actual.site_energies_ == before.site_energies_);
    REQUIRE(actual.charges_ == before.charges_);
    REQUIRE(actual.global_attribs_i_ == before.global_attribs_i_);
    auto metadata = before.global_attribs_d_;
    if (key == "temperature") metadata.erase("temperature");
    REQUIRE(actual.global_attribs_d_ == metadata);
}
}

TEST_CASE("Ordinary sequential codecs apply the same requested discards as bulk", "[gpio][sequence][discard]") {
    const std::string format = GENERATE("extxyz", "mlpcfg", "bin", "outcar", "n2p2");
    const std::string key = GENERATE("", "energy", "fenergy", "free_energy", "forces", "virial", "site_en", "temperature");
    const auto text = ordinary_input(format);
    GPIO control, bulk, sequential;
    if (!key.empty()) bulk.discard_props = sequential.discard_props = {key};
    std::vector<Atoms> before, filtered;
    std::istringstream control_input(text, std::ios::in | std::ios::binary);
    std::istringstream bulk_input(text, std::ios::in | std::ios::binary);
    REQUIRE(control.read(before, control_input, format));
    REQUIRE(bulk.read(filtered, bulk_input, format));
    REQUIRE(before.size() == 2);
    REQUIRE(filtered.size() == 2);
    std::istringstream stream(text, std::ios::in | std::ios::binary);
    REQUIRE(sequential.seq_read_init(stream, format));
    Atoms first, second;
    REQUIRE(sequential.seq_read(stream, first));
    REQUIRE(sequential.seq_read(stream, second));
    for (int i = 0; i < 2; ++i) check_selected(filtered[i], before[i], key);
    check_selected(first, before[0], key);
    check_selected(second, before[1], key);
    second.pos_[0].setConstant(99);
    check_selected(first, before[0], key);
    Atoms end;
    REQUIRE_FALSE(sequential.seq_read(stream, end));
}

TEST_CASE("Discarding forces never bypasses sequential EXTXYZ frame validation", "[gpio][sequence][discard]") {
    GPIO io;
    io.discard_props = {"forces"};
    std::istringstream input("1\nProperties=species:S:1:pos:R:3:forces:R:3\nH 1 2 3 .1 .2\n");
    REQUIRE(io.seq_read_init(input, "extxyz"));
    Atoms atom;
    REQUIRE_THROWS(io.seq_read(input, atom));
    REQUIRE(atom.size() == 0);
}
