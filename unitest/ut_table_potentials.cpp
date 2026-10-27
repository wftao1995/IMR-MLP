// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "base/neighborhood.h"
#include "analyse/compute_hessian.h"
#include "analyse/lowest_eigenmodes.h"
#include "tools/hessian_utils.h"
#include "potentials/potential_mindist.h"
#include "potentials/potential_pair_table.h"
#include "potentials/potential_threebody_table.h"
#include "potentials/potential_sw.h"
#include "potentials/potential_tersoff.h"
#include "potentials/potential_lj.h"
#include "potentials/potential_eam_poly.h"
#include "potentials/potential_hybrid_scale.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_zbl.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>

using Catch::Approx;

namespace {
Atoms pair_atoms(int i = 0, int j = 0)
{
    Atoms atoms;
    atoms.cell_ = Eigen::Matrix3d::Identity() * 20;
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{1, 1, 1}, {2.37, 1.24, 1.19}};
    atoms.chemmap_ = {i, j};
    return atoms;
}

double evaluate(Potentials::AnyPotential &pot, Atoms &atoms)
{
    Neighborhoods nbhs(atoms, pot.cutoff);
    pot.getEFS(atoms, nbhs);
    return atoms.energy_;
}

void check_derivatives(Potentials::AnyPotential &pot, Atoms atoms)
{
    evaluate(pot, atoms);
    const double h = 1e-6;
    for (int i = 0; i < atoms.size(); ++i) {
        for (int a = 0; a < 3; ++a) {
            auto plus = atoms, minus = atoms;
            plus.pos_[i](a) += h;
            minus.pos_[i](a) -= h;
            const double force = -(evaluate(pot, plus) - evaluate(pot, minus)) / (2 * h);
            INFO("atom " << i << ", axis " << a);
            REQUIRE(atoms.forces_[i](a) == Approx(force).margin(2e-7));
        }
    }
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            auto plus = atoms, minus = atoms;
            for (int i = 0; i < atoms.size(); ++i) {
                plus.pos_[i](a) += h * atoms.pos_[i](b);
                minus.pos_[i](a) -= h * atoms.pos_[i](b);
            }
            const double virial = -(evaluate(pot, plus) - evaluate(pot, minus)) / (2 * h);
            INFO("virial " << a << ", " << b);
            REQUIRE(atoms.stress_(a, b) == Approx(virial).margin(2e-7));
        }
    }
}

void write_pair_table()
{
    std::ofstream file("pair.table");
    file << "H He\nN 4\n";
    for (int i = 0; i < 4; ++i) {
        const double r = 0.5 + i;
        file << i << " " << r << " " << r*r << " " << 2*r*r << " " << 3*r*r << "\n";
    }
}

void write_sw(const std::string &file, bool pair, bool triplet)
{
    std::ofstream out(file);
    if (pair) out << "H H 2body 1 1 2 0.5 4 0 3 3\n";
    if (triplet) out << "H H H 1 1 3 2 1 -0.3333333333333333 3\n";
}

void write_tersoff(const std::string &file, bool brenner, double radius = 3)
{
    std::ofstream out(file);
    out << (brenner ? "mode = brenner\n" : "mode=tersoff\n");
    out << "H H A=10 B=3 lambda1=2 lambda2=1 lambda3=0.7 n=1 beta=1 "
        << "D0=0.8123456789012345 S=2 r0=1.4 gamma=0.7 R=" << radius
        << " D=0.3 c=1 d=1 h=-0.3\n";
}

Atoms triple_atoms()
{
    auto atoms = pair_atoms();
    atoms.pos_.push_back({1.3, 2.5, 1.2});
    atoms.chemmap_.push_back(0);
    return atoms;
}

GPSettings constant_eam_settings()
{
    GPSettings settings;
    for (const auto &[key, value] : std::map<std::string, std::string>{
        {"nelems", "1"}, {"npoly_rho", "2"}, {"npoly_f", "1"}, {"npoly_pair", "2"},
        {"rmax_pair_0_0", "6"}, {"rmax_rho_0_0", "6"}, {"coeffs", "0 0 0 0 0 1"}})
        settings.Modify(key, value);
    return settings;
}
}

TEST_CASE("Minimum-distance forces repel and match energy derivatives", "[potential][mindist]")
{
    Potentials::PotMDIST pot(3);
    auto atoms = pair_atoms(0, 4);
    check_derivatives(pot, atoms);
    atoms.pos_[1] = {2, 1, 1};
    REQUIRE(evaluate(pot, atoms) == Approx(4));
    REQUIRE(atoms.forces_[0](0) == Approx(-4));
    REQUIRE(atoms.stress_(0, 0) == Approx(4));
}

TEST_CASE("Pair tables respect species mappings and energy derivatives", "[potential][table]")
{
    write_pair_table();
    Potentials::PairTable original("pair.table", {{"H", 0}, {"He", 1}});
    Potentials::PairTable swapped("pair.table", {{"H", 1}, {"He", 0}});
    Potentials::PairTable subset("pair.table", {{"He", 0}});
    REQUIRE(original.cutoff == Approx(3.5));
    for (int i = 0; i < 2; ++i) {
        for (int j = i; j < 2; ++j) {
            auto a = pair_atoms(i, j), b = pair_atoms(1-i, 1-j);
            REQUIRE(evaluate(original, a) == Approx(evaluate(swapped, b)));
            REQUIRE((a.forces_[0] - b.forces_[0]).norm() < 1e-12);
            check_derivatives(original, a);
        }
    }
    auto a = pair_atoms(1, 1), b = pair_atoms();
    REQUIRE(evaluate(original, a) == Approx(evaluate(subset, b)));
    REQUIRE_THROWS(Potentials::PairTable("pair.table", {{"Li", 0}}));
    REQUIRE_THROWS(Potentials::PairTable("pair.table", {{"H", 0}, {"He", 0}}));
}

TEST_CASE("Pair tables reject malformed grids and incomplete data", "[potential][table]")
{
    const auto data = GENERATE(
        "", "H\nN 2\n0 1 0\n1 2 0\n", "H\nN 3\n0 1 0\n1 2 0\n",
        "H\nN 3\n0 1 0\n1 1 0\n2 3 0\n", "H\nN 3\n0 1 0\n1 1.5 0\n2 3 0\n",
        "H\nN 3\n0 1 0\n1 2 nan\n2 3 0\n", "H H\nN 3\n");
    { std::ofstream file("invalid-pair.table"); file << data; }
    REQUIRE_THROWS(Potentials::PairTable("invalid-pair.table", {}));
}

TEST_CASE("Three-body forces match energy for unequal bonds and reordered atoms", "[potential][table]")
{
    {
        std::ofstream file("threebody.table");
        file << std::setprecision(17);
        file << "H\nnr 5 ntheta 5\nrmin 0.5 rmax 4.5\ntheta_min 0 theta_max 3.141592653589793\n";
        for (int ir = 0; ir < 5; ++ir) {
            for (int it = 0; it < 5; ++it) {
                const double r = 0.5 + ir, theta = it * std::acos(-1.) / 4;
                file << r*r + 0.2*r*theta + 0.3*theta << " ";
            }
            file << "\n";
        }
    }
    Potentials::ThreebodyTable pot("threebody.table", {});
    auto atoms = pair_atoms();
    atoms.pos_.push_back({1.31, 2.68, 1.42});
    atoms.chemmap_.push_back(0);
    check_derivatives(pot, atoms);
    evaluate(pot, atoms);
    auto reordered = atoms;
    std::swap(reordered.pos_[1], reordered.pos_[2]);
    REQUIRE(evaluate(pot, reordered) == Approx(atoms.energy_));
    REQUIRE((reordered.forces_[1] - atoms.forces_[2]).norm() < 1e-12);
    REQUIRE((reordered.stress_ - atoms.stress_).norm() < 1e-12);
    atoms.pos_[1] = {1.17, 1.21, 1.08}; // A bond below rmin exercises clamped radial derivatives.
    check_derivatives(pot, atoms);
}

TEST_CASE("Potential file saving reports buffered output failures", "[potential][io]")
{
    if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full is unavailable");
    std::string suffix = GENERATE(std::string(""), std::string(".gz"), std::string(".zst"));
#ifndef USE_GZIP
    if (suffix == ".gz") SKIP("gzip support disabled");
#endif
#ifndef USE_ZSTD
    if (suffix == ".zst") SKIP("zstd support disabled");
#endif
    const auto path = "full-potential" + suffix;
    INFO("output " << path);
    std::filesystem::remove(path);
    std::filesystem::create_symlink("/dev/full", path);
    Potentials::PotMDIST pot(3);
    Potentials::AnyPotential &base = pot;
    REQUIRE_THROWS_WITH(base.Save(path), Catch::Matchers::ContainsSubstring("Failed to write"));
    std::filesystem::remove(path);
    REQUIRE_NOTHROW(base.Save(path));
    auto input = open_ifile(path);
    REQUIRE(input != nullptr);
    std::string first_line;
    REQUIRE(static_cast<bool>(std::getline(*input, first_line)));
    REQUIRE(first_line == "pot:pottype mindist");
}

TEST_CASE("SW pair and angular forces match energy and strain derivatives", "[potential][sw]")
{
    const int interaction = GENERATE(0, 1, 2);
    write_sw("derivatives.sw", interaction != 1, interaction != 0);
    Potentials::SW pot("derivatives.sw", {{"H", 0}});
    check_derivatives(pot, interaction == 0 ? pair_atoms() : triple_atoms());
}

TEST_CASE("Tersoff and Brenner forces include angular and cutoff derivatives", "[potential][tersoff]")
{
    const bool brenner = GENERATE(false, true);
    const int geometry = GENERATE(0, 1, 2);
    write_tersoff("derivatives.tersoff", brenner, geometry == 2 ? 1.5 : 3);
    Potentials::Tersoff pot("derivatives.tersoff", {{"H", 0}});
    REQUIRE(pot.get_mode() == (brenner ? Potentials::Tersoff::Mode::Brenner : Potentials::Tersoff::Mode::Tersoff));
    check_derivatives(pot, geometry == 0 ? pair_atoms() : triple_atoms());
}

TEST_CASE("SW preserves the center species when loading and updating triplet parameters", "[potential][sw]")
{
    const std::vector<std::string> names{"H", "He", "Li"};
    {
        std::ofstream out("multispecies.sw");
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                for (int k = 0; k < 3; ++k)
                    out << names[i] << " " << names[j] << " " << names[k]
                        << " 1 1 3 " << i + 1 << " 1 -0.3333333333333333 3\n";
    }
    Potentials::SW pot("multispecies.sw", {{"H", 0}, {"He", 1}, {"Li", 2}});
    auto atoms = triple_atoms();
    atoms.chemmap_ = {0, 1, 2};
    const double initial_energy = evaluate(pot, atoms);
    const bool update = GENERATE(false, true);
    if (update) {
        pot.train_init();
        std::vector<double> coeffs(pot.get_NL_coeffs_count());
        pot.nl_get_coeffs(coeffs.data());
        for (size_t i = 6 * 8 + 3; i < coeffs.size(); i += 7) coeffs[i] *= 1.123456789;
        pot.nl_set_coeffs(coeffs.data());
        REQUIRE(evaluate(pot, atoms) == Approx(initial_energy * 1.123456789).margin(1e-12));
    }
    check_derivatives(pot, atoms);
    evaluate(pot, atoms);
    std::vector<int> order{0, 1, 2};
    do {
        auto reordered = atoms;
        for (int i = 0; i < 3; ++i) {
            reordered.pos_[i] = atoms.pos_[order[i]];
            reordered.chemmap_[i] = atoms.chemmap_[order[i]];
        }
        REQUIRE(evaluate(pot, reordered) == Approx(atoms.energy_).margin(1e-12));
        for (int i = 0; i < 3; ++i)
            REQUIRE((reordered.forces_[i] - atoms.forces_[order[i]]).norm() < 1e-12);
        REQUIRE((reordered.stress_ - atoms.stress_).norm() < 1e-12);
    } while (std::next_permutation(order.begin(), order.end()));
    pot.Save("multispecies-saved.sw");
    Potentials::SW restored("multispecies-saved.sw", {{"H", 0}, {"He", 1}, {"Li", 2}});
    auto copy = atoms;
    REQUIRE(evaluate(restored, copy) == Approx(atoms.energy_).margin(1e-12));
    for (int i = 0; i < 3; ++i) REQUIRE((copy.forces_[i] - atoms.forces_[i]).norm() < 1e-12);
    REQUIRE((copy.stress_ - atoms.stress_).norm() < 1e-12);
}

TEST_CASE("SW and Tersoff saved files and streams reload with the same EFS", "[potential][io]")
{
    const int kind = GENERATE(0, 1, 2);
    const std::string suffix = GENERATE(std::string(""), std::string(".gz"), std::string(".zst"));
#ifndef USE_GZIP
    if (suffix == ".gz") SKIP("gzip support disabled");
#endif
#ifndef USE_ZSTD
    if (suffix == ".zst") SKIP("zstd support disabled");
#endif
    write_sw("roundtrip.sw", true, true);
    write_tersoff("roundtrip.tersoff", kind == 2);
    std::unique_ptr<Potentials::AnyPotential> original, restored;
    if (kind == 0) {
        original = std::make_unique<Potentials::SW>("roundtrip.sw", std::map<std::string, int>{{"H", 1}, {"He", 0}});
        restored = std::make_unique<Potentials::SW>();
    } else {
        original = std::make_unique<Potentials::Tersoff>("roundtrip.tersoff", std::map<std::string, int>{{"H", 1}, {"He", 0}});
        restored = std::make_unique<Potentials::Tersoff>();
    }
    auto atoms = triple_atoms();
    atoms.chemmap_ = {1, 1, 1};
    evaluate(*original, atoms);
    auto check = [&] {
        auto copy = atoms;
        REQUIRE(restored->species_count == 2);
        REQUIRE(restored->cutoff == Approx(original->cutoff));
        REQUIRE(evaluate(*restored, copy) == Approx(atoms.energy_).margin(1e-12));
        for (int i = 0; i < 3; ++i) REQUIRE((copy.forces_[i] - atoms.forces_[i]).norm() < 1e-12);
        REQUIRE((copy.stress_ - atoms.stress_).norm() < 1e-12);
    };
    const std::string file = "saved.potential" + suffix;
    original->Save(file);
    restored->Load(file);
    check();
    // The factory path supplies a map; it must also read the saved format.
    { std::ofstream map("roundtrip-index.map"); map << "He\nH\n"; }
    GPSettings settings;
    settings.Modify("typemap", "roundtrip-index.map");
    Potentials::potential_wrapper wrapper(file, kind == 0 ? "sw" : "tersoff", settings);
    auto copy = atoms;
    REQUIRE(evaluate(*wrapper.pot, copy) == Approx(atoms.energy_).margin(1e-12));
    std::stringstream stream;
    original->Save(stream);
    restored->Load(stream);
    check();
}

TEST_CASE("Three-body tables remap all species combinations and subsets", "[potential][table]")
{
    {
        std::ofstream out("mapped-threebody.table");
        out << "H He\nnr 3 ntheta 3\nrmin 0.5 rmax 3.5\ntheta_min 0 theta_max 3.141592653589793\n";
        for (int t = 0; t < 4; ++t)
            for (int ir = 0; ir < 3; ++ir)
                for (int it = 0; it < 3; ++it) out << (t + 1) * (1 + ir + 0.2*it) << " ";
    }
    Potentials::ThreebodyTable original("mapped-threebody.table", {});
    Potentials::ThreebodyTable swapped("mapped-threebody.table", {{"He", 0}, {"H", 1}});
    Potentials::ThreebodyTable subset("mapped-threebody.table", {{"He", 0}});
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            for (int k = 0; k < 2; ++k) {
                auto a = triple_atoms(), b = a;
                a.chemmap_ = {i, j, k}; b.chemmap_ = {1-i, 1-j, 1-k};
                REQUIRE(evaluate(original, a) == Approx(evaluate(swapped, b)));
                for (int n = 0; n < 3; ++n) REQUIRE((a.forces_[n] - b.forces_[n]).norm() < 1e-12);
                REQUIRE((a.stress_ - b.stress_).norm() < 1e-12);
            }
    auto a = triple_atoms(), b = a;
    a.chemmap_ = {1, 1, 1};
    REQUIRE(evaluate(original, a) == Approx(evaluate(subset, b)));
    REQUIRE_THROWS(Potentials::ThreebodyTable("mapped-threebody.table", {{"Li", 0}}));
    REQUIRE_THROWS(Potentials::ThreebodyTable("mapped-threebody.table", {{"H", 1}, {"He", 1}}));
    REQUIRE_THROWS(Potentials::ThreebodyTable("mapped-threebody.table", {{"H", 2}}));
}

TEST_CASE("Derived potential saves report open and buffered write failures", "[potential][io]")
{
    const int kind = GENERATE(0, 1, 2, 3, 4);
    const std::string suffix = GENERATE(std::string(""), std::string(".gz"), std::string(".zst"));
#ifndef USE_GZIP
    if (suffix == ".gz") SKIP("gzip support disabled");
#endif
#ifndef USE_ZSTD
    if (suffix == ".zst") SKIP("zstd support disabled");
#endif
    write_sw("failure.sw", true, true);
    write_tersoff("failure.tersoff", false);
    std::unique_ptr<Potentials::AnyPotential> pot;
    if (kind == 0) pot = std::make_unique<Potentials::SW>("failure.sw", std::map<std::string,int>{{"H",0}});
    if (kind == 1) pot = std::make_unique<Potentials::Tersoff>("failure.tersoff", std::map<std::string,int>{{"H",0}});
    if (kind == 2) {
        GPSettings settings;
        settings.Modify("ntypes", "1"); settings.Modify("0-0", "1 1"); settings.Modify("cut:0-0", "3");
        pot = std::make_unique<Potentials::LJ612>(settings);
    }
    if (kind == 3) pot = std::make_unique<Potentials::PotentialEAMPoly>(constant_eam_settings());
    if (kind == 4) {
        GPSettings settings;
        settings.Modify("npot", "1"); settings.Modify("pot1:pottype", "eampoly");
        for (const auto &[key, value] : constant_eam_settings().settings) settings.Modify("pot1:" + key, value);
        pot = std::make_unique<Potentials::PotentialHybridScale>(settings);
    }
    REQUIRE_THROWS(pot->Save("missing-directory/potential" + suffix));
    if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full is unavailable");
    const auto file = "failed-save" + suffix;
    std::filesystem::remove(file);
    std::filesystem::create_symlink("/dev/full", file);
    REQUIRE_THROWS(pot->Save(file));
    std::filesystem::remove(file);
    REQUIRE_NOTHROW(pot->Save(file));
    if (kind == 4) {
        std::filesystem::remove(file + ".pot1");
        std::filesystem::create_symlink("/dev/full", file + ".pot1");
        REQUIRE_THROWS(pot->Save(file));
        std::filesystem::remove(file + ".pot1");
    }
}

TEST_CASE("Settings potentials reload saved files and streams with identical EFS", "[potential][io]")
{
    const std::string kind = GENERATE(std::string("lj612"), std::string("mindist"), std::string("zbl"));
    const std::string suffix = GENERATE(std::string(""), std::string(".gz"), std::string(".zst"));
#ifndef USE_GZIP
    if (suffix == ".gz") SKIP("gzip support disabled");
#endif
#ifndef USE_ZSTD
    if (suffix == ".zst") SKIP("zstd support disabled");
#endif
    GPSettings settings;
    std::unique_ptr<Potentials::AnyPotential> original;
    if (kind == "lj612") {
        settings.Modify("ntypes", "1");
        settings.Modify("0-0", "0.8123456789012345 1.123456789012345");
        settings.Modify("cut:0-0", "3.123456789012345");
        settings.Modify("ele:0", "0.2123456789012345");
        original = std::make_unique<Potentials::LJ612>(settings);
    } else if (kind == "mindist") {
        settings.Modify("cutoff", "3.123456789012345");
        settings.Modify("factor", "2.123456789012345");
        settings.Modify("power", "4");
        original = std::make_unique<Potentials::PotMDIST>(settings);
    } else {
        settings.Modify("z", "13");
        settings.Modify("r0", "0.8123456789012345");
        settings.Modify("r1", "2.123456789012345");
        original = std::make_unique<Potentials::PotentialZBL>(settings);
    }
    auto atoms = pair_atoms();
    evaluate(*original, atoms);
    auto check = [&](Potentials::AnyPotential &restored) {
        auto copy = atoms;
        REQUIRE(restored.species_count == original->species_count);
        REQUIRE(restored.cutoff == Approx(original->cutoff).margin(1e-14));
        REQUIRE(evaluate(restored, copy) == Approx(atoms.energy_).margin(1e-12));
        for (int i = 0; i < atoms.size(); ++i) {
            REQUIRE((copy.forces_[i] - atoms.forces_[i]).norm() < 1e-12);
            REQUIRE(copy.site_energies_[i] == Approx(atoms.site_energies_[i]).margin(1e-12));
        }
        REQUIRE((copy.stress_ - atoms.stress_).norm() < 1e-12);
    };
    const std::string file = "settings-" + kind + ".pot" + suffix;
    original->Save(file);
    Potentials::potential_wrapper wrapper(file, kind);
    check(*wrapper.pot);
    REQUIRE_THROWS(Potentials::potential_wrapper("missing-directory/settings.pot" + suffix, kind));
    if (kind != "lj612") {
        std::stringstream stream;
        original->Save(stream);
        wrapper.pot->Load(stream);
        check(*wrapper.pot);
        // Legacy unprefixed key=value syntax remains readable, including whitespace.
        std::stringstream saved;
        original->Save(saved);
        std::stringstream legacy;
        std::string line;
        while (std::getline(saved, line)) {
            line.erase(0, 4);
            line.replace(line.find(' '), 1, " = ");
            legacy << "  " << line << " # comment\n";
        }
        wrapper.pot->Load(legacy);
        check(*wrapper.pot);
    }
}

TEST_CASE("Explicit settings override settings potential files", "[potential][io]")
{
    { std::ofstream out("mindist-override.pot"); out << "pot:cutoff 3\npot:factor 2\npot:power 4\n"; }
    GPSettings settings;
    settings.Modify("factor", "3");
    Potentials::potential_wrapper wrapper("mindist-override.pot", "mindist", settings);
    auto *pot = dynamic_cast<Potentials::PotMDIST *>(wrapper.pot);
    REQUIRE(pot != nullptr);
    REQUIRE(pot->cutoff == Approx(3));
    REQUIRE(pot->power == Approx(4));
    REQUIRE(pot->factor == Approx(3));
}

TEST_CASE("Table potentials preserve mapped species and EFS when saved and reloaded", "[potential][table][io]")
{
    const std::string kind = GENERATE(std::string("pair_table"), std::string("threebody_table"));
    const int mapping = GENERATE(0, 1, 2);
    const std::string suffix = GENERATE(std::string(""), std::string(".gz"), std::string(".zst"));
#ifndef USE_GZIP
    if (suffix == ".gz") SKIP("gzip support disabled");
#endif
#ifndef USE_ZSTD
    if (suffix == ".zst") SKIP("zstd support disabled");
#endif
    const std::string input = "roundtrip-input-" + kind + ".table";
    {
        std::ofstream out(input);
        out << std::setprecision(17) << "H He\n";
        if (kind == "pair_table") {
            out << "N 4\n";
            for (int i = 0; i < 4; ++i) {
                const double r = 0.5123456789012345 + i;
                out << i << " " << r;
                for (int p = 0; p < 3; ++p)
                    out << " " << (p + 1.123456789012345) * r * r;
                out << "\n";
            }
        } else {
            out << "nr 4 ntheta 3\nrmin 0.5123456789012345 rmax 3.5123456789012345\n"
                << "theta_min 0.123456789012345 theta_max 2.7123456789012345\n";
            for (int t = 0; t < 4; ++t)
                for (int ir = 0; ir < 4; ++ir)
                    for (int it = 0; it < 3; ++it)
                        out << (t + 1.123456789012345) * (1 + ir * ir + 0.3 * ir * it + it)
                            << "\n";
        }
    }
    std::map<std::string, int> typemap;
    if (mapping == 1) typemap = {{"He", 0}, {"H", 1}};
    if (mapping == 2) typemap = {{"He", 0}};
    std::unique_ptr<Potentials::AnyPotential> original, restored;
    if (kind == "pair_table") {
        original = std::make_unique<Potentials::PairTable>(input, typemap);
        restored = std::make_unique<Potentials::PairTable>();
    } else {
        original = std::make_unique<Potentials::ThreebodyTable>(input, typemap);
        restored = std::make_unique<Potentials::ThreebodyTable>();
    }
    auto check = [&](Potentials::AnyPotential &loaded) {
        REQUIRE(loaded.species_count == original->species_count);
        REQUIRE(loaded.cutoff == original->cutoff);
        for (int i = 0; i < original->species_count; ++i)
            for (int j = 0; j < original->species_count; ++j)
                for (int k = 0; k < original->species_count; ++k) {
                    auto expected = triple_atoms();
                    expected.chemmap_ = {i, j, k};
                    auto actual = expected;
                    evaluate(*original, expected);
                    evaluate(loaded, actual);
                    REQUIRE(std::abs(actual.energy_ - expected.energy_) < 1e-12);
                    REQUIRE((actual.stress_ - expected.stress_).norm() < 1e-12);
                    for (int a = 0; a < expected.size(); ++a) {
                        REQUIRE((actual.forces_[a] - expected.forces_[a]).norm() < 1e-12);
                        REQUIRE(std::abs(actual.site_energies_[a] - expected.site_energies_[a]) < 1e-12);
                    }
                }
    };
    std::stringstream stream;
    original->Save(stream);
    const std::string header = stream.str().substr(0, stream.str().find('\n'));
    REQUIRE(header == (mapping == 0 ? "H He" : mapping == 1 ? "He H" : "He"));
    restored->Load(stream);
    check(*restored);

    const std::string saved = "roundtrip-saved-" + kind + ".table" + suffix;
    original->Save(saved);
    restored->Load(saved);
    check(*restored);
    GPSettings settings;
    settings.Modify("typemap", "unused-roundtrip-index.map");
    Potentials::potential_wrapper wrapper(saved, kind, settings);
    check(*wrapper.pot);
}

namespace {
Potentials::PairTable curvature_table()
{
    // Non-unit spacing and nonzero cubic first interval, with distinct pair columns.
    std::istringstream input("H He\nN 4\n0 .5 .25 .5 .75\n1 1 1 2 3\n"
                             "2 1.5 2.25 4.5 6.75\n3 2 4 8 12\n");
    Potentials::PairTable pot;
    pot.Load(input);
    return pot;
}

Atoms table_dimer(double distance)
{
    auto atom = pair_atoms();
    atom.pos_ = {{0, 0, 0}, {distance, 0, 0}};
    return atom;
}

Eigen::VectorXd table_forces(Potentials::PairTable &pot, Atoms atom)
{
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    Eigen::VectorXd result(3 * atom.size());
    for (int i = 0; i < atom.size(); ++i) result.segment<3>(3 * i) = atom.forces_[i];
    return result;
}

Eigen::VectorXd table_force_fd(Potentials::PairTable &pot, const Atoms &atom,
                              const Eigen::VectorXd &direction, double step)
{
    auto plus = atom, minus = atom;
    for (int i = 0; i < atom.size(); ++i) {
        plus.pos_[i] += step * direction.segment<3>(3 * i).transpose();
        minus.pos_[i] -= step * direction.segment<3>(3 * i).transpose();
    }
    return -(table_forces(pot, plus) - table_forces(pot, minus)) / (2 * step);
}

void check_table_source(const Atoms &atom, const Atoms &source)
{
    REQUIRE(atom.cell_.isApprox(source.cell_, 0));
    REQUIRE(atom.cartesian == source.cartesian);
    REQUIRE(atom.pbc_ == source.pbc_);
    REQUIRE(atom.chemmap_ == source.chemmap_);
    REQUIRE(atom.energy_ == source.energy_);
    REQUIRE(atom.stress_.isApprox(source.stress_, 0));
    REQUIRE(atom.has_energy() == source.has_energy());
    REQUIRE(atom.has_stresses() == source.has_stresses());
    REQUIRE(atom.has_forces() == source.has_forces());
    for (int i = 0; i < atom.size(); ++i) {
        REQUIRE(atom.pos_[i].isApprox(source.pos_[i], 0));
        REQUIRE(atom.forces_[i].isApprox(source.forces_[i], 0));
        REQUIRE(atom.site_energies_[i] == source.site_energies_[i]);
    }
}
}

TEST_CASE("PairTable analytical cubic curvature has independent scaling and EFS semantics", "[potential][pairtable][hvp]")
{
    auto pot = curvature_table();
    Potentials::AnyPotential &api = pot;
    REQUIRE(api.supports_analytical_hessian());
    REQUIRE(api.supports_analytical_hessian_vector_product());
    auto atom = table_dimer(5.0 / 8);
    Neighborhoods nbhs(atom, pot.cutoff);
    auto direct = atom;
    pot.getEFS(direct, nbhs);
    // First interval V=.25+.75p-.25p^2+.25p^3, p=(r-.5)/.5.
    // At r=5/8: V=109/256, V'=43/32, V''=-1/2.
    Eigen::Matrix3d pair = Eigen::Matrix3d::Zero();
    pair.diagonal() << -.5, 43.0 / 20, 43.0 / 20;
    Eigen::MatrixXd expected(6, 6);
    expected << pair, -pair, -pair, pair;
    Potentials::HessianMatrix blocks;
    blocks.resize(1);
    blocks(0, 0).setConstant(42);
    api.getEFSHessian(atom, nbhs, blocks);
    REQUIRE(HessianUtils::to_dense(blocks).isApprox(expected, 1e-13));
    REQUIRE(atom.energy_ == Approx(109.0 / 256).margin(1e-13));
    REQUIRE(atom.forces_[0](0) == Approx(43.0 / 32));
    check_table_source(atom, direct);
    api.getEFSHessian(atom, blocks);
    REQUIRE(HessianUtils::to_dense(blocks).isApprox(expected, 1e-13));
    api.getEFSHessianSymmOMP(atom, blocks, 1e-4);
    REQUIRE(HessianUtils::to_dense(blocks).isApprox(expected, 1e-13));
    pot.add_efs = true;
    pot.getEFSHessian(atom, blocks);
    REQUIRE(HessianUtils::to_dense(blocks).isApprox(expected, 1e-13));
    REQUIRE(atom.energy_ == Approx(2 * direct.energy_));
    REQUIRE(atom.stress_.isApprox(2 * direct.stress_, 1e-13));
    for (int i = 0; i < atom.size(); ++i) {
        REQUIRE(atom.forces_[i].isApprox(2 * direct.forces_[i], 1e-13));
        REQUIRE(atom.site_energies_[i] == Approx(2 * direct.site_energies_[i]));
    }
    const auto source = atom;
    Eigen::VectorXd direction(6), product = Eigen::VectorXd::Constant(2, 42);
    direction << 0, 0, 0, 1, 2, 3;
    Eigen::VectorXd known(6);
    known << .5, -43.0 / 10, -129.0 / 20, -.5, 43.0 / 10, 129.0 / 20;
    pot.applyHessianVectorProduct(atom, direction, product);
    REQUIRE(product.isApprox(known, 1e-13));
    api.applyHessianVectorProduct(atom, nbhs, direction, direction);
    REQUIRE(direction.isApprox(known, 1e-13));
    check_table_source(atom, source);
    REQUIRE_THROWS(api.applyHessianVectorProduct(atom, nbhs, Eigen::VectorXd::Zero(3), product));
}

TEST_CASE("PairTable curvature matches forces for mixed pairs and repeated periodic images", "[potential][pairtable][hvp]")
{
    auto pot = curvature_table();
    auto atom = table_dimer(.75);
    atom.pos_ = {{0, 0, 0}, {.74, .13, .08}, {.31, 1.23, .17}};
    atom.chemmap_ = {0, 1, 0};
    SECTION("noncollinear distinct pair columns") {}
    SECTION("multiple images of the same target") {
        atom.cell_ = Eigen::Vector3d(2.8, 8, 9).asDiagonal();
        atom.pbc_ = {true, false, false};
        atom.pos_ = {{0, 0, 0}, {1.2, .15, 0}};
        atom.chemmap_ = {0, 1};
    }
    Neighborhoods nbhs(atom, pot.cutoff);
    Potentials::HessianMatrix blocks;
    pot.getEFSHessian(atom, nbhs, blocks);
    const auto source = atom;
    const Eigen::MatrixXd dense = HessianUtils::to_dense(blocks);
    REQUIRE(dense.isApprox(dense.transpose(), 1e-13));
    Eigen::VectorXd u(dense.cols()), v(dense.cols()), hu, hv;
    for (int k = 0; k < u.size(); ++k) {
        u[k] = (k % 4 - 1.5) / 3;
        v[k] = (k % 5 - 2.0) / 4;
    }
    pot.applyHessianVectorProduct(atom, nbhs, u, hu);
    pot.applyHessianVectorProduct(atom, nbhs, v, hv);
    REQUIRE(hu.isApprox(dense * u, 1e-12));
    REQUIRE(hv.isApprox(dense * v, 1e-12));
    REQUIRE(u.dot(hv) == Approx(v.dot(hu)).margin(1e-12));
    for (double step : {2e-5, 1e-5}) {
        Eigen::MatrixXd fd(dense.rows(), dense.cols());
        for (int k = 0; k < dense.cols(); ++k)
            fd.col(k) = table_force_fd(pot, atom, Eigen::VectorXd::Unit(dense.cols(), k), step);
        REQUIRE((dense - fd).cwiseAbs().maxCoeff() < 3e-7);
        REQUIRE((hu - table_force_fd(pot, atom, u, step)).cwiseAbs().maxCoeff() < 3e-7);
        REQUIRE((hv - table_force_fd(pot, atom, v, step)).cwiseAbs().maxCoeff() < 3e-7);
    }
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(dense.cols());
        for (int i = 0; i < atom.size(); ++i) translation[3 * i + axis] = 1;
        pot.applyHessianVectorProduct(atom, nbhs, translation, hv);
        REQUIRE(hv.cwiseAbs().maxCoeff() < 1e-13);
    }
    check_table_source(atom, source);
}

TEST_CASE("PairTable curvature preserves endpoint branches and self image energy", "[potential][pairtable][hvp]")
{
    auto pot = curvature_table();
    for (double distance : {.3, .5, 2.0, 2.2}) {
        for (int species : {0, 1}) {
            auto atom = table_dimer(distance);
            atom.chemmap_ = {0, species};
            Neighborhoods nbhs(atom, 3); // Include the upper endpoint in the supplied list.
            Potentials::HessianMatrix blocks;
            pot.getEFSHessian(atom, nbhs, blocks);
            REQUIRE(atom.energy_ == Approx(distance <= .5 ? .25 * (species + 1) : 0).margin(1e-13));
            REQUIRE(HessianUtils::to_dense(blocks).isZero(0));
            REQUIRE(atom.forces_[0].isZero(0));
            Eigen::VectorXd product;
            pot.applyHessianVectorProduct(atom, nbhs, Eigen::VectorXd::LinSpaced(6, 0, 1), product);
            REQUIRE(product.isZero(0));
        }
    }
    Atoms atom;
    atom.cell_ = Eigen::Vector3d(1.2, 8, 9).asDiagonal();
    atom.pbc_ = {true, false, false};
    atom.pos_ = {{.1, .2, .3}};
    atom.chemmap_ = {0};
    Potentials::HessianMatrix blocks;
    pot.getEFSHessian(atom, blocks);
    REQUIRE(atom.energy_ == Approx(1.44).margin(1e-13));
    REQUIRE(std::abs(atom.stress_(0, 0)) > 1);
    REQUIRE(blocks(0, 0).isZero(0));
    const auto source = atom;
    Eigen::VectorXd product;
    pot.applyHessianVectorProduct(atom, Eigen::VectorXd::Ones(3), product);
    REQUIRE(product.isZero(0));
    check_table_source(atom, source);
}

TEST_CASE("PairTable Auto and analytic consumers select the exact cubic Hessian", "[potential][pairtable][hvp]")
{
    auto pot = curvature_table();
    auto atom = table_dimer(5.0 / 8);
    evaluate(pot, atom);
    const auto source = atom;
    Eigen::Matrix3d pair = Eigen::Matrix3d::Zero();
    pair.diagonal() << -.5, 43.0 / 20, 43.0 / 20;
    Eigen::MatrixXd expected(6, 6);
    expected << pair, -pair, -pair, pair;
    GPSettings settings;
    settings.Modify("nudge_pos", "0.02");
    ComputeHessian hess(settings);
    hess.set_cutoff(0);
    for (auto method : {ComputeHessian::HessianMethod::Auto, ComputeHessian::HessianMethod::Analytic}) {
        hess.method = method;
        Eigen::MatrixXd result;
        hess.compute_fc(atom, result, &pot);
        REQUIRE(result.isApprox(expected, 1e-13));
        Potentials::HessianMatrix blocks;
        hess.compute_fc_blocks(atom, blocks, &pot);
        REQUIRE(HessianUtils::to_dense(blocks).isApprox(expected, 1e-13));
    }
    for (double step : {2e-5, 1e-5}) {
        GPSettings force_settings;
        force_settings.Modify("nudge_pos", std::to_string(step));
        ComputeHessian force(force_settings);
        force.method = ComputeHessian::HessianMethod::ForceFD;
        Eigen::MatrixXd result;
        force.compute_fc(atom, result, &pot);
        REQUIRE((result - expected).cwiseAbs().maxCoeff() < 3e-7);
    }
    check_table_source(atom, source);
}

TEST_CASE("PairTable curvature follows reordered and subset native species mappings", "[potential][pairtable][hvp]")
{
    write_pair_table();
    Potentials::PairTable original("pair.table", {{"H", 0}, {"He", 1}});
    Potentials::PairTable swapped("pair.table", {{"H", 1}, {"He", 0}});
    Potentials::PairTable subset("pair.table", {{"He", 0}});
    for (int i = 0; i < 2; ++i) {
        for (int j = i; j < 2; ++j) {
            auto a = pair_atoms(i, j), b = pair_atoms(1 - i, 1 - j);
            Potentials::HessianMatrix ha, hb;
            original.getEFSHessian(a, ha);
            swapped.getEFSHessian(b, hb);
            REQUIRE(HessianUtils::to_dense(ha).isApprox(HessianUtils::to_dense(hb), 1e-13));
            Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(6, -.5, 1), va, vb;
            original.applyHessianVectorProduct(a, direction, va);
            swapped.applyHessianVectorProduct(b, direction, vb);
            REQUIRE(va.isApprox(vb, 1e-13));
        }
    }
    auto a = pair_atoms(1, 1), b = pair_atoms();
    Potentials::HessianMatrix ha, hb;
    original.getEFSHessian(a, ha);
    subset.getEFSHessian(b, hb);
    REQUIRE(HessianUtils::to_dense(ha).isApprox(HessianUtils::to_dense(hb), 1e-13));
}

namespace {
std::string threebody_hvp_grid(double lower = .5, int types = 1)
{
    std::ostringstream out;
    out << std::setprecision(17) << (types == 1 ? "H" : "H He")
        << "\nnr 3 ntheta 3\nrmin " << lower << " rmax 3\ntheta_min 0 theta_max " << M_PI << '\n';
    const int triplets = types == 1 ? 1 : 4;
    for (int t = 0; t < triplets; ++t)
        for (int ir = 0; ir < 3; ++ir)
            for (int it = 0; it < 3; ++it)
                out << (t+1)*(lower+(3-lower)*ir/2)*(M_PI*it/2) << ' ';
    return out.str();
}

Atoms threebody_hvp_star()
{
    Atoms atom;
    atom.cell_ = 20*Eigen::Matrix3d::Identity();
    atom.pbc_ = {false,false,false};
    atom.pos_ = {{0,0,0},{1.5,0,0},{-1,std::sqrt(3.),0}};
    atom.chemmap_ = {0,0,0};
    return atom;
}

Eigen::MatrixXd threebody_full_force_fd(Potentials::ThreebodyTable &pot, const Atoms &atom, double step)
{
    Eigen::MatrixXd result(3*atom.size(),3*atom.size());
    for (int k = 0; k < result.cols(); ++k) {
        auto plus = atom, minus = atom;
        plus.pos_[k/3][k%3] += step;
        minus.pos_[k/3][k%3] -= step;
        evaluate(pot,plus); evaluate(pot,minus);
        for (int i = 0; i < atom.size(); ++i)
            result.block<3,1>(3*i,k) = -(plus.forces_[i]-minus.forces_[i])/(2*step);
    }
    return result;
}
}

TEST_CASE("ThreebodyTable HVP retains the independent mixed angular cross block", "[potential][table][hvp]")
{
    std::istringstream input(threebody_hvp_grid());
    Potentials::ThreebodyTable pot; pot.Load(input);
    auto atom = threebody_hvp_star();
    Neighborhoods nbhs(atom,pot.cutoff);
    REQUIRE(nbhs[0].nbhcount_ == 2);
    REQUIRE(nbhs[1].nbhcount_ == 1);
    REQUIRE(nbhs[2].nbhcount_ == 1);
    evaluate(pot,atom);
    REQUIRE(atom.energy_ == Approx(3.5*(2*M_PI/3)/6).margin(1e-12));
    atom.features_["hvp_label"] = "source";
    atom.local_attribs_d_["marker"] = Eigen::Vector3d(1,2,3);
    const auto source = atom;
    std::ostringstream saved_table; pot.Save(saved_table);
    Potentials::AnyPotential &base = pot;
    REQUIRE(base.supports_analytical_hessian_vector_product());
    REQUIRE(base.supports_analytical_hessian());
    REQUIRE(base.supports_analytical_cell_hessian());
    // T=r*theta gives U=(r+s)*theta/6. This full block is hand-derived,
    // including its nonsymmetric planar mixed terms and out-of-plane curvature.
    Eigen::Matrix3d expected;
    expected << -std::sqrt(3.)/24,-1./24,0,
                1./18,-std::sqrt(3.)/18,0,
                0,0,-7./(18*std::sqrt(3.));
    Eigen::MatrixXd raw(9,9);
    for (int k = 0; k < 9; ++k) {
        Eigen::VectorXd result;
        base.applyHessianVectorProduct(atom,nbhs,Eigen::VectorXd::Unit(9,k),result);
        raw.col(k) = result;
    }
    REQUIRE((raw.block<3,3>(3,6)-expected).norm() < 1e-11);
    REQUIRE((raw.block<3,3>(6,3)-expected.transpose()).norm() < 1e-11);
    REQUIRE((raw-raw.transpose()).norm() < 1e-11);
    Eigen::VectorXd u(9),v(9),hu = Eigen::VectorXd::Constant(2,99),hv,combined;
    u << .2,-.1,.4,.7,.3,-.2,-.4,.5,.1;
    v << -.3,.4,.1,.2,-.6,.5,.8,.1,-.2;
    base.applyHessianVectorProduct(atom,nbhs,u,hu);
    base.applyHessianVectorProduct(atom,v,hv);
    REQUIRE(u.dot(hv) == Approx(v.dot(hu)).margin(1e-11));
    auto alias = u;
    pot.add_efs = true; // HVP is independent of EFS accumulation mode.
    base.applyHessianVectorProduct(atom,nbhs,alias,alias);
    REQUIRE((alias-hu).norm() < 1e-12);
    base.applyHessianVectorProduct(atom,2*u-.3*v,combined);
    REQUIRE((combined-(2*hu-.3*hv)).norm() < 1e-11);
    pot.add_efs = false;
    for (double step : {1e-5,5e-6}) {
        const Eigen::MatrixXd fd = threebody_full_force_fd(pot,atom,step);
        REQUIRE((raw-fd).norm() < 3e-6);
        REQUIRE((hu-fd*u).norm() < 3e-6);
        REQUIRE((hv-fd*v).norm() < 3e-6);
    }
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(9),zero;
        for (int i = 0; i < 3; ++i) translation[3*i+axis] = 1;
        base.applyHessianVectorProduct(atom,nbhs,translation,zero);
        REQUIRE(zero.isZero(1e-12));
    }
    check_table_source(atom,source);
    REQUIRE(atom.features_ == source.features_);
    REQUIRE(atom.local_attribs_d_.at("marker").isApprox(source.local_attribs_d_.at("marker"),0));
    std::ostringstream after; pot.Save(after);
    REQUIRE(after.str() == saved_table.str());
}

TEST_CASE("ThreebodyTable HVP respects radial clamp and repeated periodic sources", "[potential][table][hvp]")
{
    SECTION("one radial branch is clamped while angular and other radial terms remain") {
        std::istringstream input(threebody_hvp_grid(1.));
        Potentials::ThreebodyTable pot; pot.Load(input);
        auto atom = threebody_hvp_star();
        atom.pos_[1] = {.7,0,0};
        atom.pos_[2] = {-1.4,1.4*std::sqrt(3.),0};
        Neighborhoods nbhs(atom,pot.cutoff);
        REQUIRE(nbhs[0].nbhcount_ == 2);
        REQUIRE(nbhs[1].nbhcount_ == 1);
        REQUIRE(nbhs[2].nbhcount_ == 1);
        evaluate(pot,atom);
        const auto source = atom;
        Eigen::VectorXd direction = Eigen::VectorXd::LinSpaced(9,-.4,.7),product;
        pot.applyHessianVectorProduct(atom,nbhs,direction,product);
        REQUIRE(product.norm() > 1e-6);
        for (double step : {1e-5,5e-6})
            REQUIRE((product-threebody_full_force_fd(pot,atom,step)*direction).norm() < 3e-6);
        check_table_source(atom,source);
    }
    SECTION("mapped triplets preserve self and repeated source images") {
        { std::ofstream file("hvp-threebody-mapped.table"); file << threebody_hvp_grid(.5,2); }
        Potentials::ThreebodyTable pot("hvp-threebody-mapped.table",{});
        Potentials::ThreebodyTable reversed("hvp-threebody-mapped.table",{{"He",0},{"H",1}});
        Atoms atom;
        atom.cell_ = Eigen::Vector3d(2.8,10,10).asDiagonal(); atom.pbc_ = {true,false,false};
        atom.pos_ = {{.2,.3,.4},{1,.8,.6}}; atom.chemmap_ = {0,1};
        Neighborhoods nbhs(atom,pot.cutoff);
        bool self = false,repeated = false;
        for (int i = 0; i < atom.size(); ++i) {
            int others = 0;
            for (int source : nbhs[i].indices_) { self |= source == i; if (source != i) ++others; }
            repeated |= others > 1;
        }
        REQUIRE(self); REQUIRE(repeated);
        evaluate(pot,atom);
        const auto source = atom;
        Eigen::VectorXd direction(6),product,remapped;
        direction << .2,-.3,.5,-.4,.6,.1;
        pot.applyHessianVectorProduct(atom,nbhs,direction,product);
        auto mapped = atom; mapped.chemmap_ = {1,0};
        reversed.applyHessianVectorProduct(mapped,direction,remapped);
        REQUIRE((product-remapped).norm() < 1e-10);
        REQUIRE(product.norm() > 1e-6);
        for (double step : {1e-5,5e-6})
            REQUIRE((product-threebody_full_force_fd(pot,atom,step)*direction).norm() < 1e-5);
        check_table_source(atom,source);
        Atoms single = atom; single.pos_.resize(1); single.chemmap_.resize(1);
        Eigen::VectorXd zero;
        pot.applyHessianVectorProduct(single,Eigen::VectorXd::Ones(3),zero);
        REQUIRE(zero.isZero(0));
    }
}

TEST_CASE("ThreebodyTable analytical HVP drives the actual matrix-free lowest-mode solver", "[potential][table][hvp]")
{
    std::istringstream input(threebody_hvp_grid());
    Potentials::ThreebodyTable pot; pot.Load(input);
    auto atom = threebody_hvp_star();
    Neighborhoods nbhs(atom,pot.cutoff);
    evaluate(pot,atom);
    const auto source = atom;
    const Eigen::MatrixXd fd = threebody_full_force_fd(pot,atom,5e-6);
    Eigen::MatrixXd translations = Eigen::MatrixXd::Zero(9,3);
    for (int i = 0; i < 3; ++i)
        for (int axis = 0; axis < 3; ++axis) translations(3*i+axis,axis) = 1/std::sqrt(3.);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(translations);
    const Eigen::MatrixXd q = qr.householderQ()*Eigen::MatrixXd::Identity(9,9);
    const Eigen::MatrixXd internal = q.rightCols(6);
    // Uniform explicit mass 2; remove the same three translations as calc_fc.
    const Eigen::MatrixXd reference = internal.transpose()*(.5*(fd+fd.transpose())/2)*internal;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> exact(reference);
    REQUIRE(exact.info() == Eigen::Success);
    auto project = [&](Eigen::VectorXd &value) {
        const Eigen::VectorXd overlap = translations.transpose()*value;
        value -= translations*overlap;
    };
    MatrixFree::LowestEigenmodesSettings settings;
    settings.count = 2; settings.tolerance = 1e-9; settings.max_matvecs = 100;
    auto apply = [&](const Eigen::VectorXd &direction,Eigen::VectorXd &product) {
        Eigen::VectorXd projected = direction;
        project(projected);
        pot.applyHessianVectorProduct(atom,nbhs,projected/std::sqrt(2.),product);
        product /= std::sqrt(2.);
        project(product);
    };
    const auto result = MatrixFree::solve_lowest_symmetric(9,apply,settings,project);
    REQUIRE(result.converged);
    REQUIRE(result.converged_count == 2);
    REQUIRE((result.eigenvalues-exact.eigenvalues().head(2)).norm() < 3e-6);
    check_table_source(atom,source);
}
