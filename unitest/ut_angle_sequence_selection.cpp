#include <catch2/catch_all.hpp>
#include "analyse/compute_bondangle.h"
#include "base/neighborhood.h"
#include <numeric>

TEST_CASE("Typed bond selection initializes and counts only canonical requested combinations", "[angles][selection]") {
    const bool pair_cutoff = GENERATE(false, true);
    Atoms atom;
    atom.cell_ = 10*Eigen::Matrix3d::Identity(); atom.pbc_={false,false,false};
    atom.pos_={{3,3,3},{4,3,3},{3,4,3}};atom.chemmap_={0,1,2};
    Neighborhoods neighbors(atom,1.1);
    ComputeAngles baseline(M_PI/180,0,M_PI,3), selected(M_PI/180,0,M_PI,3);
    // Reverse endpoints canonicalize to the same key; a second key has no observations.
    const bondangle_type_index key(2,0,1), empty(0,0,0);
    selected.set_bondangle_types({key,bondangle_type_index(1,0,2),empty});
    double row[3]={1.1,1.1,1.1};double *cutoffs[]={row,row,row};
    bondangle_count all_count, count;
    const auto all=pair_cutoff?baseline.compute(neighbors,all_count,cutoffs):baseline.compute(neighbors,all_count);
    const auto result=pair_cutoff?selected.compute(neighbors,count,cutoffs):selected.compute(neighbors,count);
    REQUIRE(result.size()==2);
    REQUIRE(result.at(key)==all.at(key));REQUIRE(count.at(key)==1);
    REQUIRE(result.at(key)[90]==1);REQUIRE(count.size()==1);
    REQUIRE(std::accumulate(result.at(empty).begin(),result.at(empty).end(),0)==0);
    selected.set_bondangle_types({});
    bondangle_type reset;selected.initbret(reset);REQUIRE(reset.size()==18);
}

TEST_CASE("Typed dihedral selection preserves chain reversal and existing torsion counts", "[angles][selection]") {
    const bool pair_cutoff = GENERATE(false,true);
    Atoms atom;atom.cell_=10*Eigen::Matrix3d::Identity();atom.pbc_={false,false,false};
    atom.pos_={{2.3,3.7,3},{3,3,3},{4,3,3},{4.7,3,3.7}};atom.chemmap_={3,1,2,0};
    Neighborhoods neighbors(atom,1.1);
    ComputeAngles baseline(M_PI/180,0,M_PI,4),selected(M_PI/180,0,M_PI,4);
    const dihedral_type_index key(3,1,2,0),empty(0,0,0,0);
    selected.set_dihedral_types({key,dihedral_type_index(0,2,1,3),empty});
    double row[4]={1.1,1.1,1.1,1.1};double *cutoffs[]={row,row,row,row};
    dihedral_count all_count,count;
    const auto all=pair_cutoff?baseline.compute(neighbors,all_count,cutoffs):baseline.compute(neighbors,all_count);
    const auto result=pair_cutoff?selected.compute(neighbors,count,cutoffs):selected.compute(neighbors,count);
    REQUIRE(result.size()==2);REQUIRE(count.size()==1);
    REQUIRE(result.at(key)==all.at(key));REQUIRE(count.at(key)==1);REQUIRE(result.at(key)[90]==1);
    REQUIRE(std::accumulate(result.at(empty).begin(),result.at(empty).end(),0)==0);
}

TEST_CASE("Selected periodic dihedrals retain existing image ownership", "[angles][selection][periodic]") {
    Atoms atom;atom.cell_=Eigen::Matrix3d::Identity();atom.pos_={{0,0,0}};atom.chemmap_={0};
    Neighborhoods neighbors(atom,1.1);
    ComputeAngles baseline(M_PI/180,0,M_PI,1), selected(M_PI/180,0,M_PI,1);
    const dihedral_type_index key(0,0,0,0);selected.set_dihedral_types({key});
    dihedral_count all_count,count;
    const auto all=baseline.compute(neighbors,all_count), result=selected.compute(neighbors,count);
    REQUIRE(count.at(key)>0);REQUIRE(count.at(key)==all_count.at(key));REQUIRE(result.at(key)==all.at(key));
}
