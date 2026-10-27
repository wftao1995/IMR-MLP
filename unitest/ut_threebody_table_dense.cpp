#include <catch2/catch_all.hpp>
#include "potentials/potential_threebody_table.h"
#include "tools/hessian_utils.h"
#include "analyse/compute_hessian.h"
#include <sstream>
#include <fstream>
#include <filesystem>
#include <iomanip>
using namespace Potentials;
namespace {
std::string grid(double lower=.5,bool clamp_angle=false) {
    std::ostringstream out;out<<std::setprecision(17)<<"H He\nnr 3 ntheta 3\nrmin "<<lower<<" rmax 3\ntheta_min "<<(clamp_angle?.4:0.)<<" theta_max "<<(clamp_angle?2.8:M_PI)<<'\n';
    for(int type=0;type<4;++type)for(int r=0;r<3;++r)for(int t=0;t<3;++t){
        const double rr=lower+(3-lower)*r/2,theta=clamp_angle?.4+1.2*t:M_PI*t/2;
        out<<(type+1)*rr*theta<<' ';}return out.str();
}
Atoms star(){Atoms a;a.cell_=20*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};
    a.pos_={{0,0,0},{1.5,0,0},{-1,std::sqrt(3.),0}};a.chemmap_={0,1,0};return a;}
Eigen::MatrixXd force_fd(AnyPotential &pot,const Atoms &a,double step,double cutoff=0.) {
    Eigen::MatrixXd result(3*a.size(),3*a.size());
    for(int k=0;k<result.cols();++k){auto p=a,m=a;p.pos_[k/3][k%3]+=step;m.pos_[k/3][k%3]-=step;
        Neighborhoods pn(p,cutoff>0?cutoff:pot.cutoff),mn(m,cutoff>0?cutoff:pot.cutoff);pot.getEFS(p,pn);pot.getEFS(m,mn);
        for(int i=0;i<a.size();++i)result.block<3,1>(3*i,k)=-(p.forces_[i]-m.forces_[i]).transpose()/(2*step);}
    return result;
}
void verify(AnyPotential &api,Atoms a) {
    api.getEFS(a);const auto expected=a;HessianMatrix blocks;Neighborhoods nbhs(a,api.cutoff);api.getEFSHessian(a,nbhs,blocks);
    const auto dense=HessianUtils::to_dense(blocks);REQUIRE(dense.norm()>1e-5);REQUIRE((dense-dense.transpose()).norm()<1e-11);
    for(double step:{1e-5,5e-6})REQUIRE((dense-force_fd(api,a,step)).norm()<3e-5);
    for(int mode=0;mode<2;++mode){Eigen::VectorXd direction(dense.cols()),product;
        for(int k=0;k<direction.size();++k)direction[k]=std::sin(.7*k+.3*mode);
        api.applyHessianVectorProduct(a,nbhs,direction,product);REQUIRE((product-dense*direction).norm()<1e-11);}
    for(int d=0;d<3;++d){Eigen::VectorXd translation=Eigen::VectorXd::Zero(dense.cols());for(int i=0;i<a.size();++i)translation[3*i+d]=1;
        REQUIRE((dense*translation).norm()<1e-11);}
    REQUIRE(std::abs(a.energy_-expected.energy_)<1e-12);REQUIRE((a.stress_-expected.stress_).norm()<1e-12);
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-expected.forces_[i]).norm()<1e-12);
    api.add_efs=true;api.getEFSHessianSymm(a,blocks);REQUIRE((dense-HessianUtils::to_dense(blocks)).norm()<1e-12);
    REQUIRE(std::abs(a.energy_-2*expected.energy_)<1e-12);REQUIRE((a.stress_-2*expected.stress_).norm()<1e-12);
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*expected.forces_[i]).norm()<1e-12);
    api.add_efs=false;
}
}
TEST_CASE("ThreebodyTable dense atomic Hessian retains mixed angular and clamp curvature","[threebody][dense][batch19]") {
    const bool clamp=GENERATE(false,true);std::istringstream in(grid(clamp?1.:.5));ThreebodyTable pot;pot.Load(in);AnyPotential &api=pot;
    REQUIRE(api.supports_analytical_hessian());auto a=star();if(clamp){a.pos_[1]={.7,0,0};a.pos_[2]={-1.4,1.4*std::sqrt(3.),0};}
    verify(api,a);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;
    Eigen::MatrixXd dense;c.compute_fc(a,dense,&api);REQUIRE((dense-force_fd(api,a,5e-6)).norm()<3e-5);
    c.set_cutoff(.6);c.compute_fc(a,dense,&api);REQUIRE(dense.isZero(0.));
}
TEST_CASE("ThreebodyTable dense scattering retains periodic repeated mapped sources","[threebody][dense][batch19]") {
    char pattern[]="/tmp/imrmlp_threebody_dense_XXXXXX";const char *made=mkdtemp(pattern);REQUIRE(made);const std::filesystem::path directory=made;
    struct Guard{std::filesystem::path p;~Guard(){std::filesystem::remove_all(p);}}guard{directory};std::ofstream(directory/"grid.table")<<grid();
    ThreebodyTable pot((directory/"grid.table").string(),{}),mapped((directory/"grid.table").string(),{{"He",0},{"H",1}});
    Atoms a;a.cell_=Eigen::Vector3d(2.8,10,10).asDiagonal();a.pbc_={true,false,false};a.pos_={{.2,.3,.4},{1,.8,.6}};a.chemmap_={0,1};
    verify(pot,a);HessianMatrix original,reversed;AnyPotential &api=pot,&mapapi=mapped;api.getEFSHessian(a,original);a.chemmap_={1,0};mapapi.getEFSHessian(a,reversed);
    REQUIRE((HessianUtils::to_dense(original)-HessianUtils::to_dense(reversed)).norm()<1e-11);
    a.pos_.resize(1);a.chemmap_.resize(1);api.getEFSHessian(a,original);REQUIRE(original(0,0).isZero(1e-12));REQUIRE(std::abs(a.energy_)>1e-5);
}
TEST_CASE("ThreebodyTable dense query preserves angular clamp small sine and empty branches","[threebody][dense][batch19]") {
    std::istringstream in(grid(.5,true));ThreebodyTable pot;pot.Load(in);AnyPotential &api=pot;
    auto a=star();a.pos_[1]={1.,0,0};a.pos_[2]={2.,0,0};verify(api,a);
    HessianMatrix h;Atoms empty;api.getEFSHessian(empty,h);REQUIRE(h.size==0);REQUIRE(h.hessian.empty());
}
