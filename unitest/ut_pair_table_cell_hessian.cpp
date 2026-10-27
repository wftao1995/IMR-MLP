#include <catch2/catch_all.hpp>
#include "potentials/potential_pair_table.h"
#include "unitest/pair_cell_fixture.h"
#include <filesystem>
#include <fstream>
using namespace Potentials;
namespace {
struct PairCellFixture {
    std::filesystem::path directory;
    PairCellFixture(){char name[]="/tmp/imrmlp_pair_cell_XXXXXX";const char *made=mkdtemp(name);if(!made)throw std::runtime_error("mkdtemp failed");directory=made;
        std::ofstream out(directory/"pair.table");out<<"He H\nN 10\n";
        for(int i=0;i<10;++i){const double r=.5+.5*i;out<<i+1<<' '<<r<<' '<<.03*r*r<<' '<<.05*r*r<<' '<<.08*r*r<<'\n';}}
    ~PairCellFixture(){std::filesystem::remove_all(directory);}
    PairTable make(){return PairTable((directory/"pair.table").string(),{{"H",0},{"He",1}});}
};
}
TEST_CASE("Mapped PairTable native cell atomic and mixed blocks differentiate actual spline energy","[pairtable][cell][batch18]") {
    const bool partial=GENERATE(false,true);const double step=GENERATE(.001,.0005);PairCellFixture file;auto pot=file.make();AnyPotential &api=pot;
    auto a=PairCellTest::geometry(partial),reference=a;api.getEFS(reference);REQUIRE(api.supports_analytical_cell_hessian());
    HessianMatrix h;Eigen::MatrixXd cell,cross;Neighborhoods nbhs(a,pot.cutoff);api.getEFSHessianCellOMP(a,nbhs,h,cell,cross);
    const auto full=PairCellTest::native_matrix(h,cell,cross);REQUIRE(cell.norm()>1e-4);REQUIRE(cross.norm()>1e-4);
    REQUIRE((full-PairCellTest::energy_fd(api,a,step,true)).norm()<30*step*step+1e-6);
    REQUIRE(std::abs(a.energy_-reference.energy_)<1e-12);REQUIRE((a.stress_-reference.stress_).norm()<1e-12);
    api.add_efs=true;api.getEFSHessianCellSymmOMP(a,h,cell,cross);REQUIRE((full-PairCellTest::native_matrix(h,cell,cross)).norm()<1e-12);
    REQUIRE(std::abs(a.energy_-2*reference.energy_)<1e-12);for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*reference.forces_[i]).norm()<1e-12);
}
TEST_CASE("PairTable analytic cell consumer maps fixed positions strain and effective cutoff","[pairtable][cell][batch18]") {
    const bool strain=GENERATE(false,true);const double cutoff=GENERATE(0.,3.1);PairCellFixture file;auto pot=file.make();AnyPotential &api=pot;auto a=PairCellTest::geometry(true);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;c.request_cell=true;c.cell_strain=strain;c.engineering_shear=true;c.set_cutoff(cutoff);
    Eigen::MatrixXd result;c.compute(a,result,&api);
    for(double step:{.001,.0005})REQUIRE((result-PairCellTest::energy_fd(api,a,step,false,strain,cutoff)).norm()<100*step*step+1e-6);
    c.request_response_pos_cell=false;c.compute(a,result,&api);REQUIRE(result.topRightCorner(strain?6:9,3*a.size()).isZero(1e-14));
}
TEST_CASE("PairTable cell query preserves clamp cutoff and periodic self image branches","[pairtable][cell][batch18]") {
    PairCellFixture file;auto pot=file.make();AnyPotential &api=pot;Atoms a;a.cell_=3.2*Eigen::Matrix3d::Identity();a.pos_={{.2,.3,.4}};a.chemmap_={0};
    HessianMatrix h;Eigen::MatrixXd cell,cross;api.getEFSHessianCellSymm(a,h,cell,cross);REQUIRE(cell.norm()>1e-4);REQUIRE(h(0,0).isZero(1e-12));REQUIRE(cross.isZero(1e-12));
    for(double step:{.001,.0005})REQUIRE((PairCellTest::native_matrix(h,cell,cross)-PairCellTest::energy_fd(api,a,step,true)).norm()<20*step*step+1e-6);
    a.pbc_={false,false,false};a.pos_.push_back({.5,.3,.4});a.chemmap_.push_back(1);
    for(double distance:{.3,5.,5.2}){a.pos_[1]=a.pos_[0]+Eigen::RowVector3d(distance,0,0);Neighborhoods nbhs(a,6.);api.getEFSHessianCellOMP(a,nbhs,h,cell,cross);
        REQUIRE(PairCellTest::native_matrix(h,cell,cross).isZero(0.));}
    a.cell_.setZero();REQUIRE_THROWS(api.getEFSHessianCellSymmOMP(a,h,cell,cross));
}
