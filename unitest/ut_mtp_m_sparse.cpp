#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_m.h"
#include "unitest/many_body_mtp_fixture.h"
#include <sstream>

using namespace Potentials;
namespace {
using Model = PotMTP_M<m_model_mtpq2>;
void init_model(Model &pot) {
    PotMTP ordinary; load_many_body_mtp_fixture(ordinary, 2);
    std::ostringstream serialized; ordinary.Save(serialized);
    std::string source = serialized.str(); source.resize(source.find("species_coeffs"));
    source += "species_coeffs = {0,0,0,0,0,0}\n"
              "moment_coeffs = {-0.15,0.004,0.005,0.075,-0.002,-0.0025}\n"
              "model_coeffs = {1.0,0.3,0.2,-0.1,0.15}\n";
    std::istringstream stream(source); pot.Load(stream);
    pot.model.coeffs = pot.model_coeffs; pot.omp_init();
}
Atoms geometry(bool periodic) {
    Atoms a;
    a.cell_ = Eigen::Matrix3d::Identity() * (periodic ? 3.7 : 20.);
    a.pbc_ = {periodic,periodic,periodic};
    a.pos_ = {{.3,.5,.4},{1.8,.8,.7},{2.2,2.4,1.2},{12.,12.,12.}};
    a.chemmap_ = {0,1,0,1};
    if (periodic) { a.pos_.resize(3); a.chemmap_.resize(3); }
    return a;
}
Eigen::Matrix3d force_block(AnyPotential &pot, const Atoms &a,int i,int j,double step) {
    Eigen::Matrix3d block;
    for (int b=0;b<3;++b) {
        auto plus=a,minus=a; plus.pos_[j][b]+=step; minus.pos_[j][b]-=step;
        pot.getEFS(plus); pot.getEFS(minus);
        block.col(b)=-(plus.forces_[i]-minus.forces_[i]).transpose()/(2*step);
    }
    return block;
}
}
TEST_CASE("MTP-M sparse curvature matches nonlinear force differences", "[mtp_m][sparse][batch02][oracle]") {
    Model pot; init_model(pot); AnyPotential &api=pot;
    auto a=geometry(false); Neighborhoods nbhs(a,pot.cutoff);
    HessianSparse sparse; api.getHessian(a,nbhs,{{0,0}},sparse);
    const auto &result=sparse;
    const auto fd=force_block(api,a,0,0,1e-4);
    REQUIRE(fd.norm()>1e-6);
    REQUIRE((result(0,0)-fd).norm()<2e-7);
}
TEST_CASE("MTP-M selected sparse Hessian retains nonlinear head curvature", "[mtp_m][sparse][batch02]") {
    const bool periodic=GENERATE(false,true);
    const double step=GENERATE(1e-4,5e-5);
    Model pot; init_model(pot); AnyPotential &api=pot;
    auto a=geometry(periodic); api.getEFS(a); const auto source=a;
    HessianMatrix dense; auto prediction=a; api.getEFSHessian(prediction,dense);
    std::vector<std::pair<int,int>> requested={{0,0},{0,1},{2,1},{0,1},{2,2}};
    if(!periodic) requested.push_back({0,3});
    HessianSparse sparse; sparse.size=99; sparse.hessian[{99,99}]=Eigen::Matrix3d::Ones();
    Neighborhoods nbhs(a,pot.cutoff); api.getHessian(a,nbhs,requested,sparse);
    REQUIRE(sparse.size==a.size()); REQUIRE(sparse.hessian.size()==(periodic ? 4 : 5));
    const auto &result=sparse;
    for (auto [i,j]:requested) {
        const auto fd=force_block(api,a,i,j,step);
        REQUIRE((fd.norm()>1e-6 || j==3));
        REQUIRE((result(i,j)-fd).norm()<2e-7);
        REQUIRE((result(i,j)-dense(i,j)).norm()<1e-10);
    }
    REQUIRE(a.energy_==source.energy_); REQUIRE(a.stress_==source.stress_);
    REQUIRE(a.forces_==source.forces_);
    api.getHessian(a,nbhs,{},sparse); REQUIRE(sparse.hessian.empty());
}
TEST_CASE("MTP-M sparse Hessian validates indices and retains LES rejection", "[mtp_m][sparse][batch02]") {
    Model pot; init_model(pot); auto a=geometry(false); Neighborhoods nbhs(a,pot.cutoff);
    HessianSparse sparse; sparse.size=1; sparse.hessian[{0,0}]=Eigen::Matrix3d::Ones();
    REQUIRE_THROWS(pot.getHessian(a,nbhs,{{0,a.size()}},sparse));
    REQUIRE(sparse.hessian.at({0,0}).isOnes());
    pot.lr_les=true; REQUIRE_THROWS(pot.getHessian(a,nbhs,{{0,0}},sparse));
}
TEST_CASE("MTP-M diagonal query uses nonlinear curvature without a global dense matrix", "[mtp_m][diagonal][batch03]") {
    const bool periodic=GENERATE(false,true);
    const double step=GENERATE(1e-4,5e-5);
    Model pot; init_model(pot); AnyPotential &api=pot;
    auto a=geometry(periodic); api.getEFS(a); const auto source=a;
    auto dense_state=a; HessianMatrix dense; api.getEFSHessian(dense_state,dense);
    std::vector<double> diagonal(999,123.);
    api.getEFSHessianSymmDiaOMP(a,diagonal);
    REQUIRE(diagonal.size()==3*a.size());
    for(int i=0;i<a.size();++i) {
        const auto fd=force_block(api,source,i,i,step);
        for(int d=0;d<3;++d) {
            REQUIRE(std::abs(diagonal[3*i+d]-fd(d,d))<2e-7);
            REQUIRE(std::abs(diagonal[3*i+d]-dense(i,d,i,d))<1e-10);
        }
    }
    REQUIRE(std::abs(a.energy_-source.energy_)<1e-12);
    REQUIRE((a.stress_-source.stress_).norm()<1e-11);
    for(int i=0;i<a.size();++i) REQUIRE((a.forces_[i]-source.forces_[i]).norm()<1e-11);
    const auto first=diagonal;
    api.add_efs=true; api.getEFSHessianSymmDiaOMP(a,diagonal);
    REQUIRE(diagonal==first);
    REQUIRE(std::abs(a.energy_-2*source.energy_)<1e-12);
    REQUIRE((a.stress_-2*source.stress_).norm()<1e-11);
    for(int i=0;i<a.size();++i) REQUIRE((a.forces_[i]-2*source.forces_[i]).norm()<1e-11);
}
TEST_CASE("MTP-M diagonal retains periodic self cancellation empty output and LES guards", "[mtp_m][diagonal][batch03]") {
    Model pot; init_model(pot); Atoms a; a.cell_=3.7*Eigen::Matrix3d::Identity();
    a.pbc_={true,true,true}; a.pos_={{.3,.5,.4}}; a.chemmap_={0};
    std::vector<double> diagonal;
    pot.getEFSHessianSymmDiaOMP(a,diagonal);
    REQUIRE(diagonal.size()==3);
    for(double d:diagonal) REQUIRE(std::abs(d)<1e-12);
    REQUIRE(std::abs(a.energy_)>1e-6);
    Atoms empty; pot.getEFSHessianSymmDiaOMP(empty,diagonal);
    REQUIRE(diagonal.empty()); REQUIRE(empty.energy_==0.);
    pot.lr_les=true; REQUIRE_THROWS(pot.getEFSHessianSymmDiaOMP(a,diagonal));
}
