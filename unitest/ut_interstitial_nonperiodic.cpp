#include <catch2/catch_all.hpp>
#include "symmetry/intersitials.h"
#include <Eigen/Geometry>

namespace {
Atoms cluster(bool cube, bool fractional=false) {
    Atoms atom;atom.pbc_={false,false,false};atom.cell_=Eigen::Matrix3d::Identity()*12;
    if(cube) for(int x:{-1,1}) for(int y:{-1,1}) for(int z:{-1,1}) atom.pos_.emplace_back(x*2.,y*2.,z*2.);
    else atom.pos_={Eigen::RowVector3d(2,0,0),Eigen::RowVector3d(-2,0,0),
        Eigen::RowVector3d(0,2,0),Eigen::RowVector3d(0,-2,0),Eigen::RowVector3d(0,0,2),Eigen::RowVector3d(0,0,-2)};
    atom.chemmap_.assign(atom.size(),0);atom.have_energy=true;atom.energy_=5.;
    if(fractional) atom.to_fractional();
    return atom;
}
}
TEST_CASE("FFF Voronoi keeps only the known internal cavity and no container artifacts", "[interstitial][nonperiodic]") {
    const bool cube=GENERATE(false,true),fractional=GENERATE(false,true),local=GENERATE(false,true);
    auto atom=cluster(cube,fractional);const Atoms before=atom;
    const Eigen::RowVector3d query=Eigen::RowVector3d::Zero();
    std::vector<Eigen::RowVector3d> sites;
    get_intersitial_sites_nonperiodic(atom,sites,{{0,.3}},true,true,true,.6,.1,.1,.1,local?&query:nullptr,4.);
    REQUIRE(sites.size()==1);REQUIRE(sites[0].norm()<1e-10);
    REQUIRE(atom.cartesian==before.cartesian);REQUIRE(atom.pos_==before.pos_);
    REQUIRE(atom.cell_==before.cell_);REQUIRE(atom.energy_==5.);
}
TEST_CASE("FFF cavity follows rigid motions and has no dependence on a box", "[interstitial][nonperiodic]") {
    auto atom=cluster(false);
    const Eigen::Matrix3d rotation=Eigen::AngleAxisd(.71,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    const Eigen::RowVector3d translation(17,-8,31);
    for(auto &p:atom.pos_) p=(p*rotation+translation).eval();
    atom.cell_.setZero();
    std::vector<Eigen::RowVector3d> sites;
    get_intersitial_sites_nonperiodic(atom,sites,{{0,.3}},true,true,true,.6,.1,.1,.1);
    REQUIRE(sites.size()==1);REQUIRE((sites[0]-translation).norm()<1e-9);
    const Eigen::RowVector3d distant=translation+Eigen::RowVector3d(10,0,0);
    get_intersitial_sites_nonperiodic(atom,sites,{{0,.3}},true,true,true,.6,.1,.1,.1,&distant,1.);
    REQUIRE(sites.empty());
}
TEST_CASE("FFF convex hull requires a three dimensional host", "[interstitial][nonperiodic]") {
    auto atom=cluster(false);for(auto &p:atom.pos_) p[2]=0;
    std::vector<Eigen::RowVector3d> sites;
    REQUIRE_THROWS_WITH(get_intersitial_sites_nonperiodic(atom,sites,{{0,.3}}),
        Catch::Matchers::ContainsSubstring("three-dimensional atomic convex hull"));
}

TEST_CASE("Local periodic Voronoi uses physical KD images without periodic computation walls", "[interstitial][local]") {
    Atoms atom;atom.cell_=4*Eigen::Matrix3d::Identity();atom.pbc_={true,true,true};
    atom.pos_={Eigen::RowVector3d::Zero()};atom.chemmap_={0};
    std::vector<Eigen::RowVector3d> sites;
    get_intersitial_sites_near(atom,Eigen::RowVector3d(2,2,2),sites,{{0,.3}},4.,true,false,false,.6,.1,.1,.1);
    REQUIRE(sites.size()==1);
    REQUIRE((sites[0]-Eigen::RowVector3d(2,2,2)).norm()<1e-9);
}
