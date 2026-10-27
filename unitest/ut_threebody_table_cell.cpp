#include <catch2/catch_all.hpp>
#include "potentials/potential_threebody_table.h"
#include "unitest/pair_cell_fixture.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
using namespace Potentials;
namespace {
struct TripletCellFixture {
    std::filesystem::path directory;
    TripletCellFixture(){char name[]="/tmp/imrmlp_triplet_cell_XXXXXX";const char *p=mkdtemp(name);if(!p)throw std::runtime_error("mkdtemp failed");directory=p;
        std::ofstream out(directory/"triplet.table");out<<std::setprecision(17)<<"He H\nnr 3 ntheta 3\nrmin .5 rmax 3.5\ntheta_min .2 theta_max 2.8\n";
        for(int type=0;type<4;++type)for(int r=0;r<3;++r)for(int t=0;t<3;++t)out<<(.5+1.5*r)*(.2+1.3*t)*(.04+.03*type)<<' ';}
    ~TripletCellFixture(){std::filesystem::remove_all(directory);}
    ThreebodyTable make(){return ThreebodyTable((directory/"triplet.table").string(),{{"H",0},{"He",1}});}
};
void close(const Eigen::MatrixXd &actual,const Eigen::MatrixXd &oracle,double step){const double scale=std::max(1.,oracle.norm());INFO("error="<<(actual-oracle).norm()<<" scale="<<scale);REQUIRE((actual-oracle).norm()<80*step*step*scale+2e-6);}
}
TEST_CASE("ThreebodyTable native cell retains mixed radial angular and source curvature","[threebody][cell][batch27]") {
    const bool partial=GENERATE(false,true);TripletCellFixture files;auto pot=files.make();AnyPotential &api=pot;auto a=PairCellTest::geometry(partial),expected=a;api.getEFS(expected);
    HessianMatrix h;Eigen::MatrixXd cell,cross;Neighborhoods nbhs(a,pot.cutoff);api.getEFSHessianCellOMP(a,nbhs,h,cell,cross);const auto full=PairCellTest::native_matrix(h,cell,cross);
    REQUIRE(cell.norm()>1e-5);REQUIRE(cross.norm()>1e-5);for(double step:{.001,.0005})close(full,PairCellTest::energy_fd(api,a,step,true),step);
    auto atom=a;HessianMatrix atomic;api.getEFSHessian(atom,atomic);REQUIRE((HessianUtils::to_dense(h)-HessianUtils::to_dense(atomic)).norm()<1e-12);
    REQUIRE(std::abs(a.energy_-expected.energy_)<1e-12);REQUIRE((a.stress_-expected.stress_).norm()<1e-12);for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-expected.forces_[i]).norm()<1e-12);
    api.add_efs=true;api.getEFSHessianCellSymmOMP(a,h,cell,cross);REQUIRE((full-PairCellTest::native_matrix(h,cell,cross)).norm()<1e-12);
    REQUIRE(std::abs(a.energy_-2*expected.energy_)<1e-12);REQUIRE((a.stress_-2*expected.stress_).norm()<1e-12);
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*expected.forces_[i]).norm()<1e-12);
}
TEST_CASE("ThreebodyTable analytic cell consumer maps fixed positions strain and effective cutoff","[threebody][cell][batch27]") {
    const bool strain=GENERATE(false,true);const double cutoff=GENERATE(0.,3.1);TripletCellFixture files;auto pot=files.make();AnyPotential &api=pot;auto a=PairCellTest::geometry(true);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;c.request_cell=true;c.cell_strain=strain;c.engineering_shear=true;c.set_cutoff(cutoff);
    Eigen::MatrixXd result;c.compute(a,result,&api);for(double step:{.001,.0005})close(result,PairCellTest::energy_fd(api,a,step,false,strain,cutoff),step);
    c.request_response_pos_cell=false;c.compute(a,result,&api);REQUIRE(result.topRightCorner(strain?6:9,3*a.size()).isZero(1e-14));
}
TEST_CASE("ThreebodyTable cell retains affine self images and angular clamp branch","[threebody][cell][batch27]") {
    TripletCellFixture files;auto pot=files.make();AnyPotential &api=pot;Atoms a;a.cell_=Eigen::Vector3d(2.8,10,10).asDiagonal();a.pbc_={true,false,false};a.pos_={{.2,.3,.4}};a.chemmap_={0};
    HessianMatrix h;Eigen::MatrixXd cell,cross;api.getEFSHessianCellSymm(a,h,cell,cross);REQUIRE(cell.norm()>1e-5);REQUIRE(h(0,0).isZero(1e-12));REQUIRE(cross.isZero(1e-12));
    for(double step:{.001,.0005})close(PairCellTest::native_matrix(h,cell,cross),PairCellTest::energy_fd(api,a,step,true),step);
    REQUIRE(std::abs(a.energy_)>1e-5);a.cell_.setZero();REQUIRE_THROWS(api.getEFSHessianCellSymmOMP(a,h,cell,cross));
}
