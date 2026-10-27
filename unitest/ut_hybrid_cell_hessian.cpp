#include <catch2/catch_all.hpp>
#include "potentials/potential_hybrid.h"
#include "unitest/many_body_mtp_fixture.h"
#include "unitest/pair_cell_fixture.h"
#include <filesystem>
#include <fstream>
using namespace Potentials;
namespace {
struct HybridFixture {
    std::filesystem::path directory;
    HybridFixture(){char pattern[]="/tmp/imrmlp_hybrid_cell_XXXXXX";const char *p=mkdtemp(pattern);if(!p)throw std::runtime_error("mkdtemp failed");directory=p;
        PotMTP ordinary;load_many_body_mtp_fixture(ordinary,2);ordinary.Save((directory/"model.mtp").string());}
    ~HybridFixture(){std::filesystem::remove_all(directory);}
    GPSettings settings(){GPSettings s;s.Modify("npot","2");s.Modify("pot1:pottype","mtp");s.Modify("pot1:potfile",(directory/"model.mtp").string());
        s.Modify("pot2:pottype","zbl");s.Modify("pot2:z","2 3");s.Modify("pot2:r0","1.2");s.Modify("pot2:r1","3.7");return s;}
};
class NoCell : public AnyPotential { void getEFS(Atoms &, AnyNeighborhoods &) override {} };
}
TEST_CASE("Hybrid native cell curvature sums real MTP and ZBL energy derivatives", "[hybrid][cell][batch08]") {
    const bool partial=GENERATE(false,true);const double step=GENERATE(.001,.0005);
    HybridFixture files;PotentialHybrid pot(files.settings());AnyPotential &api=pot;
    REQUIRE(api.supports_analytical_cell_hessian());auto a=PairCellTest::geometry(partial),reference=a;api.getEFS(reference);
    HessianMatrix h;Eigen::MatrixXd cell,cross;api.getEFSHessianCellSymmOMP(a,h,cell,cross);
    REQUIRE((PairCellTest::native_matrix(h,cell,cross)-PairCellTest::energy_fd(api,a,step,true)).norm()<60*step*step+5e-7);
    REQUIRE(std::abs(a.energy_-reference.energy_)<1e-12);REQUIRE((a.stress_-reference.stress_).norm()<1e-12);
    auto sum=Eigen::MatrixXd::Zero(9+3*a.size(),9+3*a.size()).eval();
    for(auto &child:pot.ppots){auto w=a;w.resetEFS();HessianMatrix hc;Eigen::MatrixXd cc,xc;
        Neighborhoods child_nbhs(w,child->cutoff);child->getEFSHessianCellOMP(w,child_nbhs,hc,cc,xc);REQUIRE(cc.norm()>1e-5);sum+=PairCellTest::native_matrix(hc,cc,xc);}
    REQUIRE((sum-PairCellTest::native_matrix(h,cell,cross)).norm()<1e-11);
    const auto previous=cell;api.add_efs=true;api.getEFSHessianCellSymmOMP(a,h,cell,cross);
    REQUIRE((cell-previous).norm()<1e-12);REQUIRE(std::abs(a.energy_-2*reference.energy_)<1e-12);
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*reference.forces_[i]).norm()<1e-12);
}
TEST_CASE("Hybrid analytic cell consumer maps strain fixed positions and effective cutoffs", "[hybrid][cell][batch08]") {
    const bool strain=GENERATE(false,true);const double cutoff=GENERATE(0.,3.1);
    HybridFixture files;PotentialHybrid pot(files.settings());auto a=PairCellTest::geometry(true);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;c.request_cell=true;c.cell_strain=strain;c.engineering_shear=true;c.set_cutoff(cutoff);
    Eigen::MatrixXd result;c.compute(a,result,&pot);
    for(double step:{.001,.0005})REQUIRE((result-PairCellTest::energy_fd(pot,a,step,false,strain,cutoff)).norm()<150*step*step+5e-7);
    c.request_response_pos_cell=false;c.compute(a,result,&pot);
    REQUIRE(result.topRightCorner(strain?6:9,3*a.size()).isZero(1e-14));
}
TEST_CASE("Hybrid cell capability excludes unsupported and legacy correction children", "[hybrid][cell][batch08]") {
    HybridFixture files;PotentialHybrid pot(files.settings());auto a=PairCellTest::geometry();
    HessianMatrix h;Eigen::MatrixXd cell,cross;
    auto *mtp=dynamic_cast<PotMTP*>(pot.ppots[0].get());REQUIRE(mtp);
    mtp->hc_correction=true;REQUIRE_FALSE(pot.supports_analytical_cell_hessian());
    REQUIRE_THROWS(pot.getEFSHessianCellSymmOMP(a,h,cell,cross));mtp->hc_correction=false;
    mtp->rR_correction=true;REQUIRE_FALSE(pot.supports_analytical_cell_hessian());mtp->rR_correction=false;
    REQUIRE(pot.supports_analytical_cell_hessian());pot.ppots[1]=std::make_unique<NoCell>();
    REQUIRE_FALSE(pot.supports_analytical_cell_hessian());REQUIRE_THROWS(pot.getEFSHessianCellSymmOMP(a,h,cell,cross));
    pot.ppots.clear();REQUIRE_FALSE(pot.supports_analytical_cell_hessian());
}
