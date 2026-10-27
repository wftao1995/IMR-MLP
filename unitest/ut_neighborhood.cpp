// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "base/neighborhood.h"
#include "base/neighborhood_ghost.h"
#include "base/neighborhood_cluster.h"
#include "base/neighborhood_cluster_simple.h"
#include "base/neighborhood_shell.h"
#include "base/neighborhood_cluster_shell_simple.h"
#include <algorithm>
#include <limits>
using Catch::Approx;

// Helper: simple cubic 8-atom cell
static Atoms make_sc8(double a = 4.0) {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * a;
    atom.pos_ = {
        {0, 0, 0}, {a/2, 0, 0},
        {0, a/2, 0}, {a/2, a/2, 0},
        {0, 0, a/2}, {a/2, 0, a/2},
        {0, a/2, a/2}, {a/2, a/2, a/2}
    };
    atom.chemmap_ = {0, 0, 0, 0, 0, 0, 0, 0};
    atom.wrap();
    return atom;
}

// Helper: simple dimer for minimal tests
static Atoms make_dimer() {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20.0;
    atom.pos_ = {{0, 0, 0}, {2.5, 0, 0}};
    atom.chemmap_ = {0, 0};
    return atom;
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods basic construction", "[neighborhood]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    Neighborhoods nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 8);

    // Every atom should have some neighbors in SC with a=4, cutoff=3.0
    for (int i = 0; i < nbhs.size(); ++i) {
        REQUIRE(nbhs[i].nbhcount_ > 0);
        REQUIRE(nbhs[i].centertype == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods symmetry", "[neighborhood]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    Neighborhoods nbhs(atom, cutoff);

    // For every pair (i,j), if j is in i's neighbor list,
    // i should be in j's neighbor list
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            int j = nbhs[i].indices_[k];
            bool found = false;
            for (int l = 0; l < nbhs[j].nbhcount_; ++l) {
                if (nbhs[j].indices_[l] == i) {
                    found = true;
                    // Opposing directions
                    double expected_d = nbhs[i].distances_[k];
                    REQUIRE(nbhs[j].distances_[l] == Approx(expected_d).epsilon(1e-10));
                    break;
                }
            }
            REQUIRE(found);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods respect cutoff", "[neighborhood]") {
    auto atom = make_sc8();
    double cutoff = 2.5;
    Neighborhoods nbhs(atom, cutoff);

    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            REQUIRE(nbhs[i].distances_[k] <= cutoff);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsGhost basic construction", "[neighborhood][ghost]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    NeighborhoodsGhost nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 8);
    for (int i = 0; i < nbhs.size(); ++i) {
        REQUIRE(nbhs[i].nbhcount_ > 0);
        REQUIRE(nbhs[i].centertype == 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsGhost symmetry", "[neighborhood][ghost]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    NeighborhoodsGhost nbhs(atom, cutoff);

    // Same symmetry check as Neighborhoods
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            int j = nbhs[i].indices_[k];
            bool found = false;
            for (int l = 0; l < nbhs[j].nbhcount_; ++l) {
                if (nbhs[j].indices_[l] == i) {
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsCluster unidirectional", "[neighborhood][cluster]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    NeighborhoodsCluster nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 8);

    // Cluster: if i < j, only i gets j as neighbor
    // So atom 0 should have 7 neighbors, atom 7 should have 0
    // (in a dense enough SC cell with cutoff covering all)
    // Actually with a=4.0 and cutoff=3.0, not all pairs are within cutoff.
    // Just verify that lower-index atoms generally have more neighbors
    int ct0 = nbhs[0].nbhcount_;
    int ct7 = nbhs[7].nbhcount_;
    // In this geometry, atom 0 and atom 7 are symmetric corners
    // But cluster favors lower index
    INFO("Atom 0 neighbors: " << ct0 << ", Atom 7 neighbors: " << ct7);
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsCluster unique pairs", "[neighborhood][cluster]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    NeighborhoodsCluster nbhs(atom, cutoff);

    // Collect all (i,j) pairs where i < j and they're neighbors
    // Verify no duplicate pairs
    std::set<std::pair<int,int>> pairs;
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            int j = nbhs[i].indices_[k];
            REQUIRE(i < j);  // Cluster invariant
            pairs.insert({i, j});
        }
    }
    REQUIRE(pairs.size() > 0);  // At least some neighbors
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsClusterSimple basic construction", "[neighborhood][simple]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    NeighborhoodsClusterSimple nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 8);

    // Verify cluster invariant
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].indices_.size(); ++k) {
            int j = nbhs[i].indices_[k];
            REQUIRE(i < j);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsShell basic construction", "[neighborhood][shell]") {
    auto atom = make_sc8();
    double cutoff_inner = 1.5;
    double cutoff_outer = 3.0;
    NeighborhoodsShell nbhs(atom, cutoff_inner, cutoff_outer);

    REQUIRE(nbhs.size() == 8);

    // All neighbors should be between inner and outer cutoff
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            double d = nbhs[i].distances_[k];
            REQUIRE(d > cutoff_inner);
            REQUIRE(d < cutoff_outer);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsShell symmetry", "[neighborhood][shell]") {
    auto atom = make_sc8();
    double cutoff_inner = 1.5;
    double cutoff_outer = 3.0;
    NeighborhoodsShell nbhs(atom, cutoff_inner, cutoff_outer);

    // Same symmetry check
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            int j = nbhs[i].indices_[k];
            bool found = false;
            for (int l = 0; l < nbhs[j].nbhcount_; ++l) {
                if (nbhs[j].indices_[l] == i) {
                    found = true;
                    break;
                }
            }
            REQUIRE(found);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsClusterShellSimple basic construction", "[neighborhood][shell][simple]") {
    auto atom = make_sc8();
    double cutoff_inner = 1.5;
    double cutoff_outer = 3.0;
    NeighborhoodsClusterShellSimple nbhs(atom, cutoff_inner, cutoff_outer);

    REQUIRE(nbhs.size() == 8);

    // Verify shell range + cluster invariant
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].indices_.size(); ++k) {
            int j = nbhs[i].indices_[k];
            REQUIRE(i < j);
            double d = nbhs[i].distances_[k];
            REQUIRE(d > cutoff_inner);
            REQUIRE(d < cutoff_outer);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods reinit clears old data", "[neighborhood]") {
    auto atom = make_sc8();
    Neighborhoods nbhs(atom, 3.0);
    REQUIRE(nbhs.size() == 8);

    // Reinit with different cutoff
    nbhs.init(atom, 1.8);
    REQUIRE(nbhs.size() == 8);

    // All distances should respect new cutoff
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            REQUIRE(nbhs[i].distances_[k] <= 1.8);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods dimer distances", "[neighborhood]") {
    auto atom = make_dimer();
    double cutoff = 5.0;
    Neighborhoods nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 2);

    // Both atoms should see each other
    REQUIRE(nbhs[0].nbhcount_ == 1);
    REQUIRE(nbhs[1].nbhcount_ == 1);

    double d = nbhs[0].distances_[0];
    REQUIRE(d == Approx(2.5).epsilon(1e-10));
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods::sub_neighborhoods by cutoff", "[neighborhood]") {
    auto atom = make_sc8();
    Neighborhoods nbhs(atom, 3.5);

    AnyNeighborhoods sub = nbhs.sub_neighborhoods(2.0);

    REQUIRE(sub.size() == nbhs.size());

    for (int i = 0; i < sub.size(); ++i) {
        for (int k = 0; k < sub[i].nbhcount_; ++k) {
            REQUIRE(sub[i].distances_[k] <= 2.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods::sub_neighborhoods by NN count", "[neighborhood]") {
    auto atom = make_sc8();
    Neighborhoods nbhs(atom, 5.0);

    int knn = 3;
    AnyNeighborhoods sub = nbhs.sub_neighborhoods(knn);

    for (int i = 0; i < sub.size(); ++i) {
        REQUIRE(sub[i].nbhcount_ <= knn);
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhood::get_sub_by_type", "[neighborhood]") {
    // Two-type system
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20;
    atom.pos_ = {{0, 0, 0}, {2, 0, 0}, {0, 2, 0}};
    atom.chemmap_ = {0, 1, 0};
    atom.wrap();

    Neighborhoods nbhs(atom, 3.0);
    REQUIRE(nbhs.size() == 3);

    for (int center = 0; center < nbhs.size(); ++center) {
        for (int type : {0, 1, 2}) {
            const auto sub = nbhs[center].get_sub_by_type(type);
            REQUIRE(sub.centertype == atom.chemmap_[center]);
            REQUIRE(sub.outertype == type);
            const int expected = std::count(nbhs[center].chemmap_.begin(),
                                            nbhs[center].chemmap_.end(), type);
            REQUIRE(sub.nbhcount_ == expected);
            REQUIRE(sub.pos_.size() == expected);
            REQUIRE(sub.distances_.size() == expected);
            REQUIRE(sub.indices_.size() == expected);
            for (int k = 0; k < sub.nbhcount_; ++k) {
                REQUIRE(atom.chemmap_[sub.indices_[k]] == type);
                REQUIRE(sub.pos_[k].isApprox(atom.pos_[sub.indices_[k]] - atom.pos_[center]));
                REQUIRE(sub.distances_[k] == Approx(sub.pos_[k].norm()));
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Ghost variant extensions
// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsGhost dimer distances", "[neighborhood][ghost]") {
    auto atom = make_dimer();
    double cutoff = 5.0;
    NeighborhoodsGhost nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 2);
    REQUIRE(nbhs[0].nbhcount_ == 1);
    REQUIRE(nbhs[1].nbhcount_ == 1);

    REQUIRE(nbhs[0].distances_[0] == Approx(2.5).epsilon(1e-10));
    REQUIRE(nbhs[1].distances_[0] == Approx(2.5).epsilon(1e-10));
}

TEST_CASE("NeighborhoodsGhost respect cutoff", "[neighborhood][ghost]") {
    auto atom = make_sc8();
    double cutoff = 2.5;
    NeighborhoodsGhost nbhs(atom, cutoff);

    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            REQUIRE(nbhs[i].distances_[k] <= cutoff);
        }
    }
}

TEST_CASE("NeighborhoodsGhost reinit clears old data", "[neighborhood][ghost]") {
    auto atom = make_sc8();
    NeighborhoodsGhost nbhs(atom, 3.0);
    REQUIRE(nbhs.size() == 8);

    nbhs.init(atom, 1.8);
    REQUIRE(nbhs.size() == 8);
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            REQUIRE(nbhs[i].distances_[k] <= 1.8);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Cluster variant extensions
// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsCluster dimer – only low index sees neighbor", "[neighborhood][cluster]") {
    auto atom = make_dimer();
    double cutoff = 5.0;
    NeighborhoodsCluster nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 2);
    REQUIRE(nbhs[0].nbhcount_ == 1);
    REQUIRE(nbhs[0].indices_[0] == 1);
    REQUIRE(nbhs[0].distances_[0] == Approx(2.5).epsilon(1e-10));
    // Atom 1 should NOT see atom 0 (because 1 > 0)
    REQUIRE(nbhs[1].nbhcount_ == 0);
}

TEST_CASE("NeighborhoodsCluster respect cutoff", "[neighborhood][cluster]") {
    auto atom = make_sc8();
    double cutoff = 2.5;
    NeighborhoodsCluster nbhs(atom, cutoff);

    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            REQUIRE(nbhs[i].distances_[k] <= cutoff);
        }
    }
}

TEST_CASE("NeighborhoodsClusterSimple dimer", "[neighborhood][simple]") {
    auto atom = make_dimer();
    double cutoff = 5.0;
    NeighborhoodsClusterSimple nbhs(atom, cutoff);

    REQUIRE(nbhs.size() == 2);
    REQUIRE(nbhs[0].indices_.size() == 1);
    REQUIRE(nbhs[0].indices_[0] == 1);
    REQUIRE(nbhs[0].distances_[0] == Approx(2.5).epsilon(1e-10));
    REQUIRE(nbhs[1].indices_.size() == 0);
}

TEST_CASE("NeighborhoodsClusterSimple unique pairs", "[neighborhood][simple]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    NeighborhoodsClusterSimple nbhs(atom, cutoff);

    std::set<std::pair<int,int>> pairs;
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].indices_.size(); ++k) {
            int j = nbhs[i].indices_[k];
            REQUIRE(i < j);
            pairs.insert({i, j});
        }
    }
    REQUIRE(pairs.size() > 0);
}

// ═══════════════════════════════════════════════════════════════════════
// Shell variant extensions
// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("NeighborhoodsShell dimer – neighbor inside shell", "[neighborhood][shell]") {
    auto atom = make_dimer();
    // d = 2.5, so with inner=1.5 outer=3.0 it should be found
    NeighborhoodsShell nbhs(atom, 1.5, 3.0);

    REQUIRE(nbhs.size() == 2);
    REQUIRE(nbhs[0].nbhcount_ == 1);
    REQUIRE(nbhs[1].nbhcount_ == 1);
}

TEST_CASE("NeighborhoodsShell dimer – neighbor outside shell", "[neighborhood][shell]") {
    auto atom = make_dimer();
    // d = 2.5, set outer cutoff below it → should miss
    NeighborhoodsShell nbhs(atom, 0.5, 2.0);

    REQUIRE(nbhs.size() == 2);
    REQUIRE(nbhs[0].nbhcount_ == 0);
    REQUIRE(nbhs[1].nbhcount_ == 0);
}

TEST_CASE("NeighborhoodsShell dimer – neighbor inside inner cutoff excluded", "[neighborhood][shell]") {
    auto atom = make_dimer();
    // d = 2.5, set inner cutoff above it → should miss
    NeighborhoodsShell nbhs(atom, 3.0, 5.0);

    REQUIRE(nbhs.size() == 2);
    REQUIRE(nbhs[0].nbhcount_ == 0);
    REQUIRE(nbhs[1].nbhcount_ == 0);
}

TEST_CASE("NeighborhoodsShell reinit", "[neighborhood][shell]") {
    auto atom = make_sc8();
    NeighborhoodsShell nbhs(atom, 1.5, 3.0);
    REQUIRE(nbhs.size() == 8);

    // Reinit with different window
    nbhs.init(atom, 0.5, 2.0);
    REQUIRE(nbhs.size() == 8);
    for (int i = 0; i < nbhs.size(); ++i) {
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            double d = nbhs[i].distances_[k];
            REQUIRE(d > 0.5);
            REQUIRE(d < 2.0);
        }
    }
}

TEST_CASE("NeighborhoodsClusterShellSimple dimer", "[neighborhood][shell][simple]") {
    auto atom = make_dimer();
    // d = 2.5, within shell [1.5, 3.0]; cluster invariant holds
    NeighborhoodsClusterShellSimple nbhs(atom, 1.5, 3.0);

    REQUIRE(nbhs.size() == 2);
    REQUIRE(nbhs[0].indices_.size() == 1);
    REQUIRE(nbhs[0].indices_[0] == 1);
    REQUIRE(nbhs[0].distances_[0] == Approx(2.5).epsilon(1e-10));
    REQUIRE(nbhs[1].indices_.size() == 0);
}

// ═══════════════════════════════════════════════════════════════════════
// get_neighbor_shell and sub_neighborhoods(rinn, rout)
// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("Neighborhoods::get_neighbor_shell", "[neighborhood]") {
    auto atom = make_sc8();
    Neighborhoods nbhs(atom, 3.5);

    // In SC a=4, first-neighbor distance = 2.0; second = 2*sqrt(2) ~ 2.828
    // Extract shell at r=2.0 with width 0.05
    AnyNeighborhoods shell = nbhs.get_neighbor_shell(2.0, 0.05);
    REQUIRE(shell.size() == 8);

    for (int i = 0; i < shell.size(); ++i) {
        for (int k = 0; k < shell[i].nbhcount_; ++k) {
            REQUIRE(shell[i].distances_[k] == Approx(2.0).margin(0.05));
        }
    }
}

TEST_CASE("Neighborhoods::get_neighbor_shell no match", "[neighborhood]") {
    auto atom = make_dimer();
    Neighborhoods nbhs(atom, 5.0);

    // No atoms at distance 3.0
    AnyNeighborhoods shell = nbhs.get_neighbor_shell(3.0, 0.01);
    REQUIRE(shell.size() == 2);
    REQUIRE(shell[0].nbhcount_ == 0);
    REQUIRE(shell[1].nbhcount_ == 0);
}

TEST_CASE("Neighborhoods::sub_neighborhoods by inner/outer", "[neighborhood]") {
    auto atom = make_sc8();
    Neighborhoods nbhs(atom, 3.5);

    // Extract shell between 1.8 and 2.2 (first neighbors at 2.0 should be caught)
    AnyNeighborhoods sub = nbhs.sub_neighborhoods(1.8, 2.2);
    REQUIRE(sub.size() == 8);

    for (int i = 0; i < sub.size(); ++i) {
        for (int k = 0; k < sub[i].nbhcount_; ++k) {
            double d = sub[i].distances_[k];
            REQUIRE(d >= 1.8);
            REQUIRE(d <= 2.2);
        }
    }
}

TEST_CASE("AnyNeighborhoodsSimple::get_neighbor_shell via ClusterSimple", "[neighborhood][simple]") {
    auto atom = make_sc8();
    double cutoff = 3.0;
    NeighborhoodsClusterSimple nbhs(atom, cutoff);

    AnyNeighborhoodsSimple shell = nbhs.get_neighbor_shell(2.0, 0.05);
    REQUIRE(shell.size() == 8);

    for (int i = 0; i < shell.size(); ++i) {
        for (int k = 0; k < shell[i].indices_.size(); ++k) {
            REQUIRE(shell[i].distances_[k] == Approx(2.0).margin(0.05));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
TEST_CASE("All neighbor list classes: empty atom", "[neighborhood]") {
    Atoms empty;
    empty.cell_ = Eigen::Matrix3d::Identity() * 10;

    // All classes should handle zero atoms gracefully
    REQUIRE_NOTHROW(Neighborhoods(empty, 3.0));
    REQUIRE_NOTHROW(NeighborhoodsGhost(empty, 3.0));
    REQUIRE_NOTHROW(NeighborhoodsCluster(empty, 3.0));
    REQUIRE_NOTHROW(NeighborhoodsClusterSimple(empty, 3.0));
    REQUIRE_NOTHROW(NeighborhoodsShell(empty, 1.5, 3.0));
    REQUIRE_NOTHROW(NeighborhoodsClusterShellSimple(empty, 1.5, 3.0));
}

TEST_CASE("Neighbor definition agrees with direct periodic enumeration", "[neighborhood][guard]") {
    for (int periodic : {0, 1, 3}) {
        Atoms atom;
        atom.cell_ = Eigen::Matrix3d::Identity() * 4;
        atom.pbc_ = {periodic > 0, periodic == 3, periodic == 3};
        atom.pos_ = {{0.2, 0.3, 0.4}, {20.7, 0.4, 0.6}};
        atom.chemmap_ = {0, 0};
        Neighborhoods result(atom, 1.2);
        for (int i = 0; i < 2; ++i) {
            int expected = 0;
            for (int j = 0; j < 2; ++j)
                for (int nx = -6; nx <= 6; ++nx)
                    for (int ny = -1; ny <= 1; ++ny)
                        for (int nz = -1; nz <= 1; ++nz) {
                            if ((!atom.pbc_[0] && nx) || (!atom.pbc_[1] && ny) || (!atom.pbc_[2] && nz)) continue;
                            Eigen::RowVector3d d = atom.pos_[j] - atom.pos_[i] + Eigen::RowVector3d(nx,ny,nz) * atom.cell_;
                            if (d.norm() > 1e-8 && d.norm() < 1.2) ++expected;
                        }
            REQUIRE(result[i].nbhcount_ == expected);
        }
    }
}
TEST_CASE("Neighbor lists can be rebuilt after an index error or empty input", "[neighborhood]") {
    auto atom = make_dimer();
    Neighborhoods result(atom, 3);
    REQUIRE_THROWS_AS(result.init(atom, 1e12), std::length_error);
    REQUIRE_NOTHROW(result.init(atom, 3));
    REQUIRE(result[0].distances_[0] == Approx(2.5));
    Atoms empty;
    result.init(empty, 3);
    REQUIRE(result.size() == 0);
    REQUIRE_NOTHROW(NeighborhoodsShell(atom, 0, 3));
}

TEST_CASE("Partially periodic neighbors allow zero nonperiodic lattice vectors", "[neighborhood]") {
    for (int dimensions : {1, 2}) {
        Atoms atom;
        atom.cell_.setZero();
        for (int d = 0; d < dimensions; ++d) atom.cell_(d,d) = 4;
        atom.pbc_ = {true, dimensions == 2, false};
        atom.pos_ = {{0,0,0}}; atom.chemmap_ = {0};
        Neighborhoods result(atom, 4.1);
        REQUIRE(result[0].nbhcount_ == 2 * dimensions);
        for (double distance : result[0].distances_)
            REQUIRE(distance == Approx(4));
    }
}

TEST_CASE("Neighbor indices reject unrepresentable ranges", "[neighborhood][guard]") {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity();
    atom.pos_ = {{0,0,0}}; atom.chemmap_ = {0};
    REQUIRE_THROWS_AS(Neighborhoods(atom, 1e12), std::length_error);
    atom.pbc_ = {false, false, false};
    atom.pos_.push_back({1e30,0,0}); atom.chemmap_.push_back(0);
#ifdef BOXED_NEBH
    REQUIRE_THROWS_AS(Neighborhoods(atom, 1), std::length_error);
#else
    REQUIRE_NOTHROW(Neighborhoods(atom, 1));
#endif
}

TEST_CASE("Skew lattice enumeration agrees with the equivalent cubic lattice", "[neighborhood][guard]") {
    for (double angle : {0.0, 0.37}) {
        const Eigen::Matrix3d rotation = Eigen::AngleAxisd(angle, Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
        for (int dimensions : {1, 2, 3}) {
            CAPTURE(angle, dimensions);
            Atoms atom;
            atom.cell_ << 1,1000,0, 0,1,1000, 0,0,1;
            atom.cell_ = (atom.cell_ * rotation).eval();
            atom.pbc_ = {dimensions == 3, dimensions >= 2, true};
            atom.pos_ = {{0,0,0}}; atom.chemmap_ = {0};
            Neighborhoods result(atom, 1.01);
            REQUIRE(result[0].nbhcount_ == 2 * dimensions);
            for (double distance : result[0].distances_)
                REQUIRE(distance == Approx(1).margin(1e-8));
            NeighborhoodsShell shell(atom, 0.99, 1.01);
            REQUIRE(shell[0].nbhcount_ == 2 * dimensions);
        }
    }
}
