#include <catch2/catch_all.hpp>
#include "potentials/potential_eam_poly.h"
#include "unitest/pair_cell_fixture.h"
using namespace Potentials;
namespace {
GPSettings settings(int basis){GPSettings s;s.Modify("nelems","2");s.Modify("basis_poly",std::to_string(basis));s.Modify("npoly_rho","3");s.Modify("npoly_pair","3");s.Modify("npoly_f","3");
    s.Modify("rmax_pair_0_0","5.5");s.Modify("rmax_pair_0_1","3.0");s.Modify("rmax_pair_1_0","3.0");s.Modify("rmax_pair_1_1","4.3");
    s.Modify("rmax_rho_0_0","4.0");s.Modify("rmax_rho_0_1","5.2");s.Modify("rmax_rho_1_0","2.7");s.Modify("rmax_rho_1_1","5.5");return s;}
void coeffs(PotentialEAMPoly &p,int sign){std::fill(p.regression_coeffs.begin(),p.regression_coeffs.end(),0.);
    for(int a=0;a<2;++a){for(int b=0;b<2;++b){double *d=p.density_coeffs+p.get_rhoid(a,b)*3;d[0]=sign*(.2+.03*a-.02*b);d[1]=sign*.03*(a+1);d[2]=-sign*.015*(b+1);}
        p.embedding_coeffs[3*a]=.07*(a+1);p.embedding_coeffs[3*a+1]=.12+.03*a;p.embedding_coeffs[3*a+2]=-.01*(a+1);p.species_coeffs[a]=.1*a;}
    for(int pair=0;pair<3;++pair){double *d=p.pair_coeffs+3*pair;d[0]=.08+.02*pair;d[1]=-.03;d[2]=.015;}}
void close(const Eigen::MatrixXd &a,const Eigen::MatrixXd &b,double h){const double scale=std::max(1.,b.norm());INFO("error="<<(a-b).norm()<<" scale="<<scale);REQUIRE((a-b).norm()<80*h*h*scale+2e-6);}
}
TEST_CASE("EAMPoly native cell preserves density sign basis and independent cutoffs","[eam_poly][cell][batch33]") {
    const int basis=GENERATE(1,2,3,4,5),sign=GENERATE(-1,0,1);const bool partial=GENERATE(false,true);PotentialEAMPoly pot(settings(basis));coeffs(pot,sign);AnyPotential &api=pot;
    auto a=PairCellTest::geometry(partial),expected=a;api.getEFS(expected);HessianMatrix h;Eigen::MatrixXd cell,cross;Neighborhoods nbhs(a,pot.cutoff);api.getEFSHessianCellOMP(a,nbhs,h,cell,cross);
    const auto full=PairCellTest::native_matrix(h,cell,cross);REQUIRE(cell.norm()>1e-6);REQUIRE(cross.norm()>1e-6);
    for(double step:{.001,.0005})close(full,PairCellTest::energy_fd(api,a,step,true),step);
    auto atom=a;HessianMatrix atomic;api.getEFSHessian(atom,atomic);REQUIRE((HessianUtils::to_dense(h)-HessianUtils::to_dense(atomic)).norm()<1e-11);
    REQUIRE(std::abs(a.energy_-expected.energy_)<1e-10);REQUIRE((a.stress_-expected.stress_).norm()<1e-10);for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-expected.forces_[i]).norm()<1e-10);
    api.add_efs=true;api.getEFSHessianCellSymmOMP(a,h,cell,cross);REQUIRE((full-PairCellTest::native_matrix(h,cell,cross)).norm()<1e-11);
    REQUIRE(std::abs(a.energy_-2*expected.energy_)<1e-10);REQUIRE((a.stress_-2*expected.stress_).norm()<1e-10);for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*expected.forces_[i]).norm()<1e-10);
}
TEST_CASE("EAMPoly analytic cell consumer maps fixed Cartesian positions strain and cutoff","[eam_poly][cell][batch33]") {
    const int basis=GENERATE(1,2,3,4,5);const bool strain=GENERATE(false,true);PotentialEAMPoly pot(settings(basis));coeffs(pot,1);AnyPotential &api=pot;auto a=PairCellTest::geometry(true);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;c.request_cell=true;c.cell_strain=strain;c.engineering_shear=true;c.set_cutoff(3.1);
    Eigen::MatrixXd result;c.compute(a,result,&api);for(double step:{.001,.0005})close(result,PairCellTest::energy_fd(api,a,step,false,strain,3.1),step);
    c.request_response_pos_cell=false;c.compute(a,result,&api);REQUIRE(result.topRightCorner(strain?6:9,3*a.size()).isZero(1e-14));
}
TEST_CASE("EAMPoly native self cell curvature keeps periodic embedding backgrounds","[eam_poly][cell][batch33]") {
    const int basis=GENERATE(1,2,3,4,5),sign=GENERATE(-1,0,1);PotentialEAMPoly pot(settings(basis));coeffs(pot,sign);AnyPotential &api=pot;Atoms a;a.cell_=3.2*Eigen::Matrix3d::Identity();a.pos_={{.2,.3,.4}};a.chemmap_={0};
    HessianMatrix h;Eigen::MatrixXd cell,cross;api.getEFSHessianCellSymm(a,h,cell,cross);REQUIRE(cell.norm()>1e-6);REQUIRE(h(0,0).isZero(1e-11));REQUIRE(cross.isZero(1e-11));
    for(double step:{.001,.0005})close(PairCellTest::native_matrix(h,cell,cross),PairCellTest::energy_fd(api,a,step,true),step);
    a.cell_.setZero();REQUIRE_THROWS(api.getEFSHessianCellSymmOMP(a,h,cell,cross));
}
