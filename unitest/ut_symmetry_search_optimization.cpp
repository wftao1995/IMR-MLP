#include <catch2/catch_all.hpp>
#include "symmetry/symmetry.h"
#include <algorithm>
#include <numeric>
#include <random>
using Catch::Approx;
namespace {
Atoms repeated(bool tilt,bool fractional,bool partial,int copies) {
    Atoms atom;
    atom.cell_<<3,0,0,tilt ? .7 : 0.,3.7,0,tilt ? -.3 : 0.,tilt ? .5 : 0.,4.1;
    const Eigen::Matrix3d primitive=atom.cell_;
    for(int copy=0;copy<copies;++copy)for(int i=0;i<3;++i) {
        const Eigen::RowVector3d position=i==0 ? Eigen::RowVector3d(.2,.3,.4) : i==1 ? Eigen::RowVector3d(1.1,1.3,1.7) : Eigen::RowVector3d(2.2,.9,2.6);
        atom.pos_.push_back(position+copy*primitive.row(0));atom.chemmap_.push_back(i==2 ? 1 : 0);
    }
    atom.cell_.row(0)*=copies;
    if(partial)atom.pbc_={true,true,false};
    if(fractional)atom.to_fractional();
    atom.have_energy=true;atom.energy_=7*copies;atom.id=17;
    return atom;
}
int run(int api,Atoms& atom,const Symmetry3D::ReductionSearchOptions& options,Symmetry3D::ReductionSearchStats& stats) {
    if(api==0)return Symmetry3D::trans_reduce(atom,1e-5,options,&stats);
    if(api==1)return Symmetry3D::trans_reduce_1d(atom,0,1e-5,options,&stats);
    if(api==2)return Symmetry3D::find_primitive(atom,1e-5,options,&stats);
    return Symmetry3D::all_reduce(atom,1e-5,options,&stats);
}
void equivalent(Atoms a,Atoms b) {
    REQUIRE(a.size()==b.size());REQUIRE(a.types_count_map()==b.types_count_map());
    REQUIRE(a.pbc_==b.pbc_);REQUIRE(a.cartesian==b.cartesian);REQUIRE(a.id==b.id);
    REQUIRE(a.energy_==Approx(b.energy_).margin(1e-12));
    REQUIRE(std::abs(a.cell_.determinant())==Approx(std::abs(b.cell_.determinant())).margin(1e-9));
    a.to_cartesian();b.to_cartesian();
    const auto inverse=a.cell_.inverse().eval();std::vector<bool> used(b.size());
    for(int i=0;i<a.size();++i) {
        bool found=false;
        for(int j=0;j<b.size();++j)if(!used[j] && a.chemmap_[i]==b.chemmap_[j]) {
            Eigen::RowVector3d difference=(a.pos_[i]-b.pos_[j])*inverse;
            for(int k=0;k<3;++k)if(a.pbc_[k])difference[k]-=std::round(difference[k]);
            if(difference.norm()<1e-5){used[j]=true;found=true;break;}
        }
        REQUIRE(found);
    }
}
}
TEST_CASE("Opt-in reduction search agrees with the legacy oracle under permutations and representations", "[symmetry][search]") {
    for(int api:{0,1,2,3})for(bool tilt:{false,true})for(bool fractional:{false,true})for(int copies:{2,3}) {
        auto original=repeated(tilt,fractional,false,copies);
        std::mt19937 rng(17);
        for(int trial=0;trial<4;++trial) {
            std::vector<int> indices(original.size());std::iota(indices.begin(),indices.end(),0);std::shuffle(indices.begin(),indices.end(),rng);
            auto a=original;
            for(int i=0;i<a.size();++i){a.pos_[i]=original.pos_[indices[i]];a.chemmap_[i]=original.chemmap_[indices[i]];}
            auto b=a;Symmetry3D::ReductionSearchOptions off,on;on.enabled=true;
            Symmetry3D::ReductionSearchStats before,after;
            REQUIRE(run(api,a,off,before)==run(api,b,on,after));equivalent(a,b);
            REQUIRE(after.geometry_stages>=2);
        }
    }
}
TEST_CASE("Translation search necessary species filters reduce full checks and exact caches stay local", "[symmetry][search]") {
    Atoms atom;atom.cell_=8*Eigen::Matrix3d::Identity();
    for(int i=0;i<6;++i){atom.pos_.push_back(Eigen::RowVector3d(i,0,0));atom.chemmap_.push_back(i<4 ? 0 : 1);}
    auto a=atom,b=atom;Symmetry3D::ReductionSearchOptions off,on;on.enabled=true;
    Symmetry3D::ReductionSearchStats before,after;
    REQUIRE(run(0,a,off,before)==1);REQUIRE(run(0,b,on,after)==1);
    REQUIRE(after.cache_hits>0);equivalent(a,b);
    atom.cell_(0,0)=6;atom.pos_={{0,0,0},{3,0,0},{.5,0,0},{1.1,0,0},{2.2,0,0},{.8,0,0}};
    atom.chemmap_={0,1,0,0,0,1};a=atom;b=atom;before={};after={};
    REQUIRE(run(0,a,off,before)==1);REQUIRE(run(0,b,on,after)==1);
    REQUIRE(after.full_checks<before.full_checks);equivalent(a,b);
    on.cache_capacity=0;a=atom;b=atom;before={};after={};
    REQUIRE(run(0,a,off,before)==run(0,b,on,after));REQUIRE(after.cache_hits==0);equivalent(a,b);
}
TEST_CASE("Optimized translation search preserves slab and near-tolerance no-reduction behavior", "[symmetry][search]") {
    for(bool partial:{false,true})for(double noise:{0.,1e-7,2e-5}) {
        auto a=repeated(true,false,partial,3);a.pos_.back()[1]+=noise;auto b=a;
        Symmetry3D::ReductionSearchOptions off,on;on.enabled=true;Symmetry3D::ReductionSearchStats x,y;
        REQUIRE(run(0,a,off,x)==run(0,b,on,y));equivalent(a,b);
    }
}
