// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "analyse/compute_ha.h"
#include "base/neighborhood.h"
#include "base/neighborhood_voro_direct.h"
#include <cmath>
#include <sstream>

namespace {
Atoms fcc_cell(bool primitive) {
    Atoms atom;
    if (primitive) {
        atom.cell_ << 0, .5, .5, .5, 0, .5, .5, .5, 0;
        atom.pos_ = {{0, 0, 0}};
    } else {
        atom.cell_ = Eigen::Matrix3d::Identity();
        atom.pos_ = {{0, 0, 0}, {0, .5, .5}, {.5, 0, .5}, {.5, .5, 0}};
    }
    atom.chemmap_.assign(atom.size(), 0);
    return atom;
}

void require_fcc_pairs(const ha_index_image &result, int count) {
    REQUIRE(result.all_count == 6 * count);
    REQUIRE(result.total.at(1421) == 6 * count);
    for (const auto &[index, value] : result.total)
        if (index != 1421) REQUIRE(value == 0);
    const haindex_bond_type partial{1421, {0, 0}};
    REQUIRE(result.partial.at(partial) == 6 * count);
#ifndef NO_SAMPLE_STR
    // Independently inspect the six-point FCC pair cluster: four common
    // neighbors, with exactly two disjoint nearest-neighbor bonds among them.
    const auto &positions = result.structure.at(1421).pos;
    REQUIRE(positions.size() == 6);
    REQUIRE(positions[0].isZero());
    REQUIRE(positions[1].norm() == Catch::Approx(std::sqrt(.5)));
    int bonds = 0;
    for (int k = 2; k < 6; ++k) {
        REQUIRE(positions[k].norm() == Catch::Approx(std::sqrt(.5)));
        REQUIRE((positions[k] - positions[1]).norm() == Catch::Approx(std::sqrt(.5)));
        int degree = 0;
        for (int l = 2; l < 6; ++l) {
            if (k == l) continue;
            const double distance = (positions[k] - positions[l]).norm();
            REQUIRE(distance > 1e-8);
            if (distance < .75) {
                ++degree;
                if (k < l) ++bonds;
            }
        }
        REQUIRE(degree == 1);
    }
    REQUIRE(bonds == 2);
#endif
}

std::map<std::string, std::pair<int, double>> ha_output_rows(const std::string &output) {
    std::map<std::string, std::pair<int, double>> rows;
    std::istringstream input(output);
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string index, fraction_text, extra;
        int count;
        REQUIRE(static_cast<bool>(row >> index >> count >> fraction_text));
        REQUIRE_FALSE(static_cast<bool>(row >> extra));
        REQUIRE_FALSE(rows.contains(index));
        rows[index] = {count, std::stod(fraction_text)};
    }
    return rows;
}
}

TEST_CASE("HA pair cutoffs filter the complete bond graph", "[ha]")
{
    const bool unbond = GENERATE(false, true);
    Atoms atoms;
    atoms.cell_ = 10 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{2, 2, 2}, {3, 2, 2}, {3, 3, 2}, {2, 3, 2}};
    atoms.chemmap_ = {0, 1, 0, 1};
    Neighborhoods short_neighbors(atoms, 1.1), long_neighbors(atoms, 1.5);
    GPSettings settings;
    settings.Modify("unbond", unbond ? "true" : "false");
    settings.Modify("store_str", "true");
    ComputeHA compute(settings);
    double same[] = {1.1, 1.1}, other[] = {1.1, 1.1};
    double *cutoffs[] = {same, other};
    const auto expected = compute.compute(short_neighbors);
    const auto actual = compute.compute(long_neighbors, cutoffs);
    REQUIRE(actual.total == expected.total);
    REQUIRE(actual.partial == expected.partial);
    REQUIRE(actual.all_count == expected.all_count);
    REQUIRE(actual.total.at(1001) == 4);
#ifndef NO_SAMPLE_STR
    for (const auto &[index, structure] : expected.structure)
    {
        const auto &positions = actual.structure.at(index).pos;
        REQUIRE(positions.size() == structure.pos.size());
        for (int i = 0; i < positions.size(); ++i)
            REQUIRE((positions[i] - structure.pos[i]).norm() < 1e-12);
    }
#endif
}

TEST_CASE("HA resolves primitive and conventional FCC periodic images", "[ha][periodic]") {
    const bool primitive = GENERATE(false, true);
    CAPTURE(primitive);
    auto atom = fcc_cell(primitive);
    Neighborhoods neighbors(atom, .75), long_neighbors(atom, 1.01);
    GPSettings settings;
    settings.Modify("store_str", "true");
    ComputeHA compute(settings);
    const auto result = compute.compute(neighbors);
    require_fcc_pairs(result, atom.size());
    // The same image-aware graph must be used after filtering every edge.
    double cutoff = .75;
    double *cutoffs[] = {&cutoff};
    const auto filtered = compute.compute(long_neighbors, cutoffs);
    require_fcc_pairs(filtered, atom.size());
    REQUIRE(filtered.total == result.total);
    REQUIRE(filtered.partial == result.partial);
}

TEST_CASE("HA unbonded periodic images use translated two-hop positions", "[ha][periodic]") {
    Atoms atom;
    atom.cell_.row(0) << 1, 0, 0;
    atom.pbc_ = {true, false, false};
    atom.pos_ = {{0, 0, 0}};
    atom.chemmap_ = {0};
    Neighborhoods neighbors(atom, 1.1);
    GPSettings settings;
    settings.Modify("unbond", "true");
    settings.Modify("store_str", "true");
    ComputeHA compute(settings);
    const auto result = compute.compute(neighbors);
    REQUIRE(result.all_count == 2);
    REQUIRE(result.total.at(1001) == 1);
    REQUIRE(result.total.at(2101) == 1);
    for (const auto &[index, value] : result.total)
        if (index != 1001 && index != 2101) REQUIRE(value == 0);
    REQUIRE(result.partial.at(haindex_bond_type{1001, {0, 0}}) == 1);
    REQUIRE(result.partial.at(haindex_bond_type{2101, {0, 0}}) == 1);
#ifndef NO_SAMPLE_STR
    const auto &bonded = result.structure.at(1001).pos;
    REQUIRE(bonded.size() == 2);
    REQUIRE(bonded[0].isZero());
    REQUIRE(bonded[1].isApprox(Eigen::RowVector3d(1, 0, 0), 1e-12));
    const auto &unbonded = result.structure.at(2101).pos;
    REQUIRE(unbonded.size() == 3);
    REQUIRE(unbonded[0].isZero());
    REQUIRE(unbonded[1].isApprox(Eigen::RowVector3d(2, 0, 0), 1e-12));
    REQUIRE(unbonded[2].isApprox(Eigen::RowVector3d(1, 0, 0), 1e-12));
#endif
}

TEST_CASE("HA consumes actual unrefined Voronoi FCC image geometry", "[ha][periodic][voro]") {
    auto atom = fcc_cell(true);
    NeighborhoodsVoroDir voronoi(atom, 12, 0, 0, 0);
    REQUIRE(voronoi[0].nbhcount_ == 12);
    for (int source : voronoi[0].indices_) REQUIRE(source == 0);
    GPSettings settings;
    settings.Modify("store_str", "true");
    ComputeHA compute(settings);
    const auto actual = compute.compute(voronoi);
    require_fcc_pairs(actual, atom.size());
    Neighborhoods radial(atom, .75);
    const auto expected = compute.compute(radial);
    REQUIRE(actual.total == expected.total);
    REQUIRE(actual.partial == expected.partial);
}

TEST_CASE("HA formatters preserve empty rows with finite zero fractions", "[ha][format]") {
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{0, 0, 0}, {5, 0, 0}};
    atoms.chemmap_ = {0, 0};
    Neighborhoods neighbors(atoms, 1.1);
    ComputeHA compute(GPSettings{});
    auto result = compute.compute(neighbors);
    REQUIRE(result.all_count == 0);
    REQUIRE(result.partial.empty());
    auto require_empty_rows = [](const std::string &output, size_t count) {
        const auto rows = ha_output_rows(output);
        REQUIRE(rows.size() == count);
        for (const auto &[index, row] : rows) {
            CAPTURE(index);
            REQUIRE(row.first == 0);
            REQUIRE(std::isfinite(row.second));
            REQUIRE(row.second == 0);
        }
    };
    std::ostringstream all, common, partial, partial_common, partial_after_common;
    result.outall(all);
    require_empty_rows(all.str(), result.total.size());
    result.outallcomm(common, compute.bflag());
    require_empty_rows(common.str(), 15);
    REQUIRE(ha_output_rows(common.str()).contains("Total"));
    // An empty partial map has no rows; formatting it must not invent any.
    result.outpart(partial, 0, 0);
    REQUIRE(partial.str().empty());
    result.outpartcomm(partial_common, 0, 0, compute.bflag());
    require_empty_rows(partial_common.str(), 15);
    REQUIRE(ha_output_rows(partial_common.str()).contains("Total"));
    // The common formatter inserts its existing zero-index entries. A later
    // full partial output must normalize those rows without changing shape.
    result.outpart(partial_after_common, 0, 0);
    require_empty_rows(partial_after_common.str(), 14);
    REQUIRE_FALSE(ha_output_rows(partial_after_common.str()).contains("Total"));
}

TEST_CASE("HA partial fractions keep the nonempty global pair denominator", "[ha][format]") {
    Atoms atoms;
    atoms.cell_ = 20 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    // Six bonds in a regular tetrahedron; three connect type0 to type1.
    atoms.pos_ = {{0, 0, 0}, {1, 0, 0}, {.5, .866025403784, 0},
                  {.5, .288675134595, .816496580928}};
    atoms.chemmap_ = {0, 0, 0, 1};
    Neighborhoods neighbors(atoms, 1.1);
    ComputeHA compute(GPSettings{});
    auto result = compute.compute(neighbors);
    REQUIRE(result.all_count == 6);
    REQUIRE(result.total.at(1211) == 6);
    REQUIRE(result.partial.at(haindex_bond_type{1211, {0, 1}}) == 3);
    std::ostringstream all, partial, common, partial_common;
    result.outall(all);
    result.outpart(partial, 1, 0);
    result.outallcomm(common, compute.bflag());
    result.outpartcomm(partial_common, 1, 0, compute.bflag());
    auto require_fraction = [](const std::string &output, int count, double fraction, bool total) {
        const auto rows = ha_output_rows(output);
        REQUIRE(rows.at("1211").first == count);
        REQUIRE(rows.at("1211").second == Catch::Approx(fraction));
        REQUIRE(rows.contains("Total") == total);
        if (total) {
            REQUIRE(rows.at("Total").first == count);
            REQUIRE(rows.at("Total").second == Catch::Approx(fraction));
        }
        for (const auto &[index, row] : rows) REQUIRE(std::isfinite(row.second));
    };
    require_fraction(all.str(), 6, 1., false);
    require_fraction(partial.str(), 3, .5, false);
    require_fraction(common.str(), 6, 1., true);
    require_fraction(partial_common.str(), 3, .5, true);
}

namespace {
void require_selected_ha(const ha_index_image &full, const ha_index_image &selected,
                         const std::set<haindexT> &codes) {
    REQUIRE(selected.all_count == full.all_count);
    REQUIRE(selected.total.size() == codes.size());
    for (int code : codes) {
        const auto found=full.total.find(code);
        REQUIRE(selected.total.at(code) == (found == full.total.end() ? 0 : found->second));
    }
    ha_partial_count expected;
    for (const auto &[key,count] : full.partial) if (codes.contains(key.index)) expected[key]=count;
    REQUIRE(selected.partial == expected);
#ifndef NO_SAMPLE_STR
    for (const auto &[code,cluster] : selected.structure) {
        REQUIRE(codes.contains(code));
        const auto &reference=full.structure.at(code).pos;
        REQUIRE(reference.size()==cluster.pos.size());
        for (std::size_t i=0;i<reference.size();++i) REQUIRE(reference[i]==cluster.pos[i]);
    }
#endif
}
}

TEST_CASE("HA selected categories preserve periodic ownership and full denominators", "[ha][selection][periodic]") {
    const bool primitive=GENERATE(false,true),pair_cutoff=GENERATE(false,true);
    auto atoms=fcc_cell(primitive);
    Neighborhoods short_neighbors(atoms,.75),long_neighbors(atoms,1.01);
    GPSettings settings;settings.Modify("store_str","true");
    ComputeHA full(settings),selected(settings);
    const std::set<haindexT> codes{1001,1421,1551};selected.set_selected_indices(codes);
    double row[]={.75};double *cutoffs[]={row};
    auto &graph=pair_cutoff?long_neighbors:short_neighbors;
    const auto baseline=pair_cutoff?full.compute(graph,cutoffs):full.compute(graph);
    const auto result=pair_cutoff?selected.compute(graph,cutoffs):selected.compute(graph);
    require_selected_ha(baseline,result,codes);
    REQUIRE(result.total.at(1421)==6*atoms.size());
    selected.set_selected_indices({});
    const auto reset=pair_cutoff?selected.compute(graph,cutoffs):selected.compute(graph);
    REQUIRE(reset.total==baseline.total);REQUIRE(reset.partial==baseline.partial);
    REQUIRE(reset.all_count==baseline.all_count);
}

TEST_CASE("HA selected unbonded categories skip work without changing image identity", "[ha][selection][periodic]") {
    Atoms atoms;atoms.cell_.row(0)<<1,0,0;atoms.pbc_={true,false,false};
    atoms.pos_={{0,0,0}};atoms.chemmap_={0};Neighborhoods graph(atoms,1.1);
    GPSettings settings;settings.Modify("unbond","true");settings.Modify("store_str","true");
    ComputeHA full(settings),selected(settings);selected.set_selected_indices({2101,1551});
    HAWorkStats base_work,selected_work;
    const auto baseline=full.compute(graph,base_work),result=selected.compute(graph,selected_work);
    require_selected_ha(baseline,result,{2101,1551});
    REQUIRE(result.all_count==2);REQUIRE(result.total.at(2101)==1);
    // Bonded and unbonded first digits are both selected, so use a purely
    // unbonded set to prove ii pruning also avoids common construction.
    selected.set_selected_indices({2101});
    const auto unbonded=selected.compute(graph,selected_work);
    require_selected_ha(baseline,unbonded,{2101});
    REQUIRE(selected_work.owned_pairs==base_work.owned_pairs);
    REQUIRE(selected_work.common_sets<base_work.common_sets);
    REQUIRE(selected_work.fourth_classifications<base_work.fourth_classifications);
}

TEST_CASE("HA selection includes missing rows and keeps global partial fractions", "[ha][selection][format]") {
    Atoms atoms;atoms.cell_=20*Eigen::Matrix3d::Identity();atoms.pbc_={false,false,false};
    atoms.pos_={{0,0,0},{1,0,0},{.5,.866025403784,0},{.5,.288675134595,.816496580928}};
    atoms.chemmap_={0,0,0,1};Neighborhoods graph(atoms,1.1);
    GPSettings settings;settings.Modify("indices","1551 1211 1551");
    ComputeHA selected(settings),full(GPSettings{});
    const auto result=selected.compute(graph);require_selected_ha(full.compute(graph),result,{1211,1551});
    std::ostringstream total,partial;result.outselected(total);result.outpartselected(partial,1,0);
    const auto rows=ha_output_rows(total.str()),part=ha_output_rows(partial.str());
    REQUIRE(rows.size()==3);REQUIRE(rows.at("1211").first==6);REQUIRE(rows.at("1551").first==0);
    REQUIRE(rows.at("Total").second==Catch::Approx(1.));
    REQUIRE(part.at("1211").first==3);REQUIRE(part.at("1211").second==Catch::Approx(.5));
    REQUIRE(part.at("1551").second==0.);REQUIRE(part.at("Total").second==Catch::Approx(.5));
    REQUIRE(total.str().find("1211")<total.str().find("1551"));
    selected.set_selected_indices({1551});const auto absent=selected.compute(graph);
    std::ostringstream output;absent.outselected(output);
    REQUIRE(ha_output_rows(output.str()).at("Total").second==0.);
    auto sum=result;sum+=result;REQUIRE(sum.all_count==12);REQUIRE(sum.total.at(1211)==12);
    std::ostringstream pooled;sum.outpartselected(pooled,0,1);
    REQUIRE(ha_output_rows(pooled.str()).at("1211").second==Catch::Approx(.5));
    selected.set_selected_indices({0});REQUIRE_THROWS(selected.compute(graph));
}

TEST_CASE("HA extended category selection falls back safely to full classification", "[ha][selection][extended]") {
    Atoms atoms;atoms.cell_=100*Eigen::Matrix3d::Identity();atoms.pbc_={false,false,false};
    for(int i=0;i<12;++i) {atoms.pos_.push_back({double(i%3),double((i/3)%2),double(i/6)});atoms.chemmap_.push_back(i%2);}
    Neighborhoods graph(atoms,10.);
    ComputeHA full(GPSettings{}),selected(GPSettings{});
    // Complete twelve-vertex graph: each pair has ten common neighbors
    // joined by45bonds, using the existing extended encoder unchanged.
    const int code=100000+1000*10+10*45+1;
    selected.set_selected_indices({code,1421});HAWorkStats baseline_work,selected_work;
    const auto baseline=full.compute(graph,baseline_work),result=selected.compute(graph,selected_work);
    REQUIRE(baseline.total.at(code)==66);require_selected_ha(baseline,result,{code,1421});
    REQUIRE(selected_work.common_sets==baseline_work.common_sets);
    REQUIRE(selected_work.common_bond_checks==baseline_work.common_bond_checks);
    REQUIRE(selected_work.fourth_classifications==baseline_work.fourth_classifications);
    selected.set_selected_indices({1421});const auto ordinary=selected.compute(graph,selected_work);
    require_selected_ha(baseline,ordinary,{1421});REQUIRE(selected_work.common_bond_checks==0);
}

TEST_CASE("HA selected classification reports actual reduced bond work on mixed environments", "[ha][selection][work]") {
    Atoms atoms;atoms.cell_=30*Eigen::Matrix3d::Identity();atoms.pbc_={false,false,false};
    // Four FCC cells plus a disordered mixed-species region give both matching
    // and nonmatching common-neighbor environments without selecting atoms.
    auto fcc=fcc_cell(false);
    for(int copy=0;copy<4;++copy) for(std::size_t i=0;i<fcc.pos_.size();++i) {
        atoms.pos_.push_back(fcc.pos_[i]+Eigen::RowVector3d(3.*copy,0,0));
        atoms.chemmap_.push_back((copy+i)%2);
    }
    for(int i=0;i<32;++i) {atoms.pos_.push_back({double(i%4)*.46,4.+double((i/4)%4)*.51,double(i/16)*.58});atoms.chemmap_.push_back(i%2);}
    Neighborhoods graph(atoms,.8);ComputeHA full(GPSettings{}),selected(GPSettings{});
    selected.set_selected_indices({1421,1551});HAWorkStats a,b;
    const auto baseline=full.compute(graph,a),result=selected.compute(graph,b);
    require_selected_ha(baseline,result,{1421,1551});
    REQUIRE(b.owned_pairs==a.owned_pairs);REQUIRE(b.common_bond_checks<a.common_bond_checks);
    REQUIRE(b.fourth_classifications<a.fourth_classifications);
    std::cout<<"HA selection work: owned="<<a.owned_pairs
             <<" common="<<a.common_sets<<"->"<<b.common_sets
             <<" common-bonds="<<a.common_bond_checks<<"->"<<b.common_bond_checks
             <<" fourth="<<a.fourth_classifications<<"->"<<b.fourth_classifications<<'\n';
}
