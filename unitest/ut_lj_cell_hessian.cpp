#include <catch2/catch_all.hpp>
#include "potentials/potential_lj.h"
#include "unitest/pair_cell_fixture.h"
using namespace Potentials;
namespace {
GPSettings lj_cell_settings(){GPSettings s;s.Modify("ntypes","2");
    s.Modify("0-0",".05 1.1");s.Modify("0-1",".08 1.2");s.Modify("1-1",".03 1.0");
    s.Modify("cut:0-0","4.6");s.Modify("cut:0-1","3.5");s.Modify("cut:1-1","4.0");
    s.Modify("ele:0",".4");s.Modify("ele:1","-.2");return s;}
}
TEST_CASE("LJ native cell blocks preserve species cutoffs and complete energy curvature","[lj][cell][batch17]") {
    const bool partial=GENERATE(false,true);const double step=GENERATE(.001,.0005);
    LJ612 pot(lj_cell_settings());AnyPotential &api=pot;auto a=PairCellTest::geometry(partial),reference=a;api.getEFS(reference);
    REQUIRE(api.supports_analytical_cell_hessian());HessianMatrix h;Eigen::MatrixXd cell,cross;Neighborhoods nbhs(a,pot.cutoff);
    api.getEFSHessianCellOMP(a,nbhs,h,cell,cross);const auto full=PairCellTest::native_matrix(h,cell,cross);
    REQUIRE(cell.norm()>1e-5);REQUIRE(cross.norm()>1e-5);
    REQUIRE((full-PairCellTest::energy_fd(api,a,step,true)).norm()<150*step*step+5e-7);
    REQUIRE(std::abs(a.energy_-reference.energy_)<1e-13);REQUIRE((a.stress_-reference.stress_).norm()<1e-13);
    api.add_efs=true;api.getEFSHessianCellSymmOMP(a,h,cell,cross);
    REQUIRE((full-PairCellTest::native_matrix(h,cell,cross)).norm()<1e-13);
    REQUIRE(std::abs(a.energy_-2*reference.energy_)<1e-13);
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*reference.forces_[i]).norm()<1e-13);
}
TEST_CASE("LJ analytic cell consumer maps fixed Cartesian positions strain and effective cutoff","[lj][cell][batch17]") {
    const bool strain=GENERATE(false,true);const double cutoff=GENERATE(0.,3.1);
    LJ612 pot(lj_cell_settings());AnyPotential &api=pot;auto a=PairCellTest::geometry(true);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;c.request_cell=true;c.cell_strain=strain;c.engineering_shear=true;c.set_cutoff(cutoff);
    Eigen::MatrixXd result;c.compute(a,result,&api);
    for(double step:{.001,.0005})REQUIRE((result-PairCellTest::energy_fd(api,a,step,false,strain,cutoff)).norm()<150*step*step+5e-7);
    c.request_response_pos_cell=false;c.compute(a,result,&api);REQUIRE(result.topRightCorner(strain?6:9,3*a.size()).isZero(1e-14));
}
TEST_CASE("LJ native cell retains self image curvature while atomic and mixed terms cancel","[lj][cell][batch17]") {
    LJ612 pot(lj_cell_settings());AnyPotential &api=pot;Atoms a;a.cell_=3.2*Eigen::Matrix3d::Identity();a.pos_={{.2,.3,.4}};a.chemmap_={0};
    HessianMatrix h;Eigen::MatrixXd cell,cross;api.getEFSHessianCellSymm(a,h,cell,cross);
    REQUIRE(cell.norm()>1e-5);REQUIRE(h(0,0).isZero(1e-13));REQUIRE(cross.isZero(1e-13));
    for(double step:{.001,.0005})REQUIRE((PairCellTest::native_matrix(h,cell,cross)-PairCellTest::energy_fd(api,a,step,true)).norm()<20*step*step+5e-7);
    a.cell_.setZero();REQUIRE_THROWS(api.getEFSHessianCellSymmOMP(a,h,cell,cross));
}
