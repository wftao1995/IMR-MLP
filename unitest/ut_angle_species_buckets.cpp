#include <catch2/catch_all.hpp>
#include "analyse/compute_bondangle.h"
#include "base/neighborhood.h"
#include <cmath>

namespace {
void check_selected(const AnyNeighborhoods &graph, int types, bool cutoff) {
    ComputeAngles direct(M_PI/180, 0, M_PI, types), selected(M_PI/180, 0, M_PI, types);
    // Cover unequal/equal center species, reversed chains, repeated endpoint
    // species and an absent center species. Select every observed key in turn.
    std::vector<std::vector<double>> radii(types, std::vector<double>(types, 1.8));
    std::vector<double *> rows;
    for (auto &row : radii) rows.push_back(row.data());
    bondangle_count bc;
    dihedral_count dc;
    const auto bonds = cutoff ? direct.compute(graph, bc, rows.data()) : direct.compute(graph, bc);
    const auto dihedrals = cutoff ? direct.compute(graph, dc, rows.data()) : direct.compute(graph, dc);
    for (const auto &[key, bins] : bonds) {
        selected.set_bondangle_types({key});
        bondangle_count count;
        const auto actual = cutoff ? selected.compute(graph, count, rows.data()) : selected.compute(graph, count);
        REQUIRE(actual.size() == 1);
        REQUIRE(actual.at(key) == bins);
        REQUIRE(count.size() == bc.count(key));
        if (bc.contains(key)) REQUIRE(count.at(key) == bc.at(key));
    }
    for (const auto &[key, bins] : dihedrals) {
        selected.set_dihedral_types({key, dihedral_type_index(key.chem[3], key.chem[2], key.chem[1], key.chem[0])});
        dihedral_count count;
        const auto actual = cutoff ? selected.compute(graph, count, rows.data()) : selected.compute(graph, count);
        REQUIRE(actual.size() == 1);
        REQUIRE(actual.at(key) == bins);
        REQUIRE(count.size() == dc.count(key));
        if (dc.contains(key)) REQUIRE(count.at(key) == dc.at(key));
    }
}
}

TEST_CASE("Species buckets exactly preserve dense mixed-species angle histograms", "[angles][selection][buckets]") {
    const bool cutoff = GENERATE(false, true);
    Atoms atom;
    atom.cell_ = 8 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{1,1,1}, {2,1,1}, {1,2,1}, {1,1,2}, {2,2,1}, {2,1,2}, {1,2,2}, {2,2,2}};
    atom.chemmap_ = {0,1,2,0,1,2,0,1};
    Neighborhoods graph(atom, 2.1);
    // Exact cutoff equality is accepted, a nextafter smaller cutoff is not.
    graph[0].distances_[0] = 1.8;
    if (graph[0].nbhcount_ > 1) graph[0].distances_[1] = std::nextafter(1.8, 2.0);
    check_selected(graph, 4, cutoff);
}

TEST_CASE("Species buckets preserve skew partial-periodic and duplicate image slots", "[angles][selection][buckets][periodic]") {
    const bool cutoff = GENERATE(false, true);
    const bool partial = GENERATE(false, true);
    Atoms atom;
    atom.cell_ << 1.6,0,0, .3,1.7,0, .2,.1,1.8;
    atom.pbc_ = {true, true, !partial};
    atom.pos_ = {{.1,.1,.1}, {.7,.6,.8}, {1.1,.9,.2}};
    atom.chemmap_ = {0,1,0};
    Neighborhoods graph(atom, 1.85);
    // Retain two identical raw slots as well as multiple periodic images of
    // each source ID. Selection must not deduplicate either kind of edge.
    auto &site = graph[0];
    REQUIRE(site.nbhcount_ > 0);
    site.indices_.push_back(site.indices_[0]);
    site.chemmap_.push_back(site.chemmap_[0]);
    site.pos_.push_back(site.pos_[0]);
    site.distances_.push_back(site.distances_[0]);
    ++site.nbhcount_;
    check_selected(graph, 3, cutoff);
}

TEST_CASE("Multiple sparse selections retain raw slot arithmetic and accumulated counts", "[angles][selection][buckets]") {
    AnyNeighborhoods graph;
    graph.resize(18);
    for (int i = 0; i < graph.size(); ++i) {
        graph[i].centertype = i % 6;
        const Eigen::RowVector3d center(.19*i, std::sin(i*.7), std::cos(i*.4));
        for (int j = 0; j < graph.size(); ++j) {
            if (i == j) continue;
            const Eigen::RowVector3d other(.19*j, std::sin(j*.7), std::cos(j*.4));
            graph[i].indices_.push_back(j);
            graph[i].chemmap_.push_back(j % 6);
            graph[i].pos_.push_back(other-center);
            graph[i].distances_.push_back((other-center).norm());
            ++graph[i].nbhcount_;
        }
    }
    ComputeAngles direct(M_PI/180, 0, M_PI, 6), selected(M_PI/180, 0, M_PI, 6);
    const std::set<bondangle_type_index> bs{{5,0,1}, {2,2,2}, {1,0,1}};
    const std::set<dihedral_type_index> ds{{5,0,1,2}, {2,2,2,5}, {1,0,1,5}};
    selected.set_bondangle_types(bs);
    selected.set_dihedral_types(ds);
    bondangle_count bc, sc;
    dihedral_count dc, tc;
    const auto b = direct.compute(graph, bc);
    const auto d = direct.compute(graph, dc);
    // Counts are additive across calls, including preexisting values.
    for (const auto &key : bs) sc[key] = 7;
    for (const auto &key : ds) tc[key] = 11;
    const auto sb = selected.compute(graph, sc);
    const auto sd = selected.compute(graph, tc);
    REQUIRE(sb.size() == bs.size());
    REQUIRE(sd.size() == ds.size());
    for (const auto &key : bs) {
        REQUIRE(sb.at(key) == b.at(key));
        REQUIRE(sc.at(key) == bc[key] + 7);
    }
    for (const auto &key : ds) {
        REQUIRE(sd.at(key) == d.at(key));
        REQUIRE(tc.at(key) == dc[key] + 11);
    }
}
