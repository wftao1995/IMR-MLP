#include <catch2/catch_all.hpp>
#include "potentials/potential_hybrid.h"
#include "potentials/potential_lj.h"
#include "potentials/potential_pair_table.h"
#include "potentials/potential_threebody_table.h"
#include "potentials/potential_zbl.h"
#include "unitest/pair_cell_fixture.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>

using namespace Potentials;

namespace {
struct ForwardingTables {
    std::filesystem::path directory;
    ForwardingTables() {
        char pattern[] = "/tmp/imrmlp_hessian_forwarding_XXXXXX";
        const char *made = mkdtemp(pattern);
        if (!made) throw std::runtime_error("mkdtemp failed");
        directory = made;
        std::ofstream pair(directory / "pair.table");
        pair << "He H\nN 10\n";
        for (int i = 0; i < 10; ++i) {
            const double r = .5 + .5*i;
            pair << i+1 << ' ' << r << ' ' << .03*r*r << ' '
                 << .05*r*r << ' ' << .08*r*r << '\n';
        }
        std::ofstream triplet(directory / "triplet.table");
        triplet << "He H\nnr 3 ntheta 3\nrmin .5 rmax 3.5\ntheta_min .2 theta_max 2.8\n";
        for (int type = 0; type < 4; ++type)
            for (int r = 0; r < 3; ++r)
                for (int t = 0; t < 3; ++t)
                    triplet << (.5+1.5*r)*(.2+1.3*t)*(.04+.03*type) << ' ';
    }
    ~ForwardingTables() { std::filesystem::remove_all(directory); }

    std::unique_ptr<AnyPotential> make(int kind) const {
        GPSettings settings;
        if (kind == 0) {
            settings.Modify("ntypes", "2");
            settings.Modify("0-0", ".05 1.1");
            settings.Modify("0-1", ".08 1.2");
            settings.Modify("1-1", ".03 1.0");
            settings.Modify("cut:0-0", "4.6");
            settings.Modify("cut:0-1", "3.5");
            settings.Modify("cut:1-1", "4.0");
            return std::make_unique<LJ612>(settings);
        }
        if (kind == 1) {
            settings.Modify("z", "2 3");
            settings.Modify("r0", "1.2");
            settings.Modify("r1", "3.7");
            return std::make_unique<PotentialZBL>(settings);
        }
        if (kind == 2)
            return std::make_unique<PairTable>((directory / "pair.table").string(),
                                               std::map<std::string,int>{{"H",0},{"He",1}});
        if (kind == 3)
            return std::make_unique<ThreebodyTable>((directory / "triplet.table").string(),
                                                    std::map<std::string,int>{{"H",0},{"He",1}});
        settings.Modify("npot", "2");
        for (int i = 1; i <= 2; ++i) {
            const auto prefix = "pot" + std::to_string(i) + ":";
            settings.Modify(prefix + "pottype", "zbl");
            settings.Modify(prefix + "z", "2 3");
            settings.Modify(prefix + "r0", "1.2");
            settings.Modify(prefix + "r1", i == 1 ? "3.7" : "2.7");
        }
        return std::make_unique<PotentialHybrid>(settings);
    }
};

void same_efs(const Atoms &actual, const Atoms &expected) {
    REQUIRE(std::abs(actual.energy_ - expected.energy_) < 1e-12);
    REQUIRE((actual.stress_ - expected.stress_).norm() < 1e-12);
    for (int i = 0; i < actual.size(); ++i)
        REQUIRE((actual.forces_[i] - expected.forces_[i]).norm() < 1e-12);
}
}

TEST_CASE("Native Hessian overloads dispatch through AnyPotential and preserve EFS",
          "[hessian][forwarding][cell]") {
    const int kind = GENERATE(0, 1, 2, 3, 4);
    const bool add = GENERATE(false, true);
    const bool partial = GENERATE(false, true);
    // A shorter supplied list detects wrappers that silently rebuild neighborhoods.
    const bool shorter = GENERATE(false, true);
    INFO("potential=" << kind << " add_efs=" << add << " partial=" << partial
         << " shorter neighborhoods=" << shorter);
    ForwardingTables tables;
    auto owner = tables.make(kind);
    AnyPotential &api = *owner;
    auto seed = PairCellTest::geometry(partial);
    api.getEFS(seed);
    api.add_efs = add;
    const double cutoff = shorter ? 2.5 : api.cutoff;
    Neighborhoods nbhs(seed, cutoff);

    auto expected = seed, native = seed, supplied = seed, atomic = seed;
    api.getEFS(expected, nbhs);
    HessianMatrix href, hsupplied, hatomic;
    Eigen::MatrixXd cref, xref, csupplied, xsupplied;
    api.getEFSHessianCellOMP(native, nbhs, href, cref, xref);
    api.getEFSHessianCell(supplied, nbhs, hsupplied, csupplied, xsupplied);
    REQUIRE((PairCellTest::native_matrix(href, cref, xref)
             - PairCellTest::native_matrix(hsupplied, csupplied, xsupplied)).norm() < 1e-12);
    same_efs(native, expected);
    same_efs(supplied, expected);
    api.getEFSHessianOMP(atomic, nbhs, hatomic);
    REQUIRE((HessianUtils::to_dense(hatomic) - HessianUtils::to_dense(href)).norm() < 1e-12);
    same_efs(atomic, expected);

    auto automatic = seed, full = seed, full_expected = seed;
    Neighborhoods full_nbhs(seed, api.cutoff);
    api.getEFS(full_expected, full_nbhs);
    api.getEFSHessianCellOMP(full, full_nbhs, href, cref, xref);
    api.getEFSHessianCell(automatic, hsupplied, csupplied, xsupplied);
    REQUIRE((PairCellTest::native_matrix(href, cref, xref)
             - PairCellTest::native_matrix(hsupplied, csupplied, xsupplied)).norm() < 1e-12);
    same_efs(automatic, full_expected);
}
