#include <catch2/catch_all.hpp>
#include "unitest/les_curvature_fixture.h"
#include "unitest/pair_cell_fixture.h"

using namespace Potentials;
using namespace LESCurvature;
namespace {
Eigen::MatrixXd gradient_fd(AnyPotential& api,const Atoms& reference,double step) {
    const int d=9+3*reference.size();
    Eigen::MatrixXd f(reference.size(),3);
    for(int i=0;i<reference.size();++i) f.row(i)=reference.pos_[i]*reference.cell_.inverse();
    auto gradient=[&](const Eigen::VectorXd& q) {
        auto work=reference;
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) work.cell_(a,b)+=q[3*a+b];
        for(int i=0;i<work.size();++i) work.pos_[i]=f.row(i)*work.cell_+q.segment<3>(9+3*i).transpose();
        api.getEFS(work);
        const Eigen::Matrix3d inverse=work.cell_.inverse();
        Eigen::Matrix3d gc=-inverse.transpose()*work.stress_.transpose();
        Eigen::VectorXd result(d);
        for(int i=0;i<work.size();++i) {
            const Eigen::RowVector3d residual=q.segment<3>(9+3*i).transpose()*inverse;
            gc+=residual.transpose()*work.forces_[i].transpose();
            result.segment<3>(9+3*i)=-work.forces_[i];
        }
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) result[3*a+b]=gc(a,b);
        return result;
    };
    Eigen::MatrixXd out(d,d);Eigen::VectorXd q=Eigen::VectorXd::Zero(d);
    for(int j=0;j<d;++j) {
        q[j]=step;const auto p=gradient(q);q[j]=-step;const auto m=gradient(q);q[j]=0.;
        out.col(j)=(p-m)/(2*step);
    }
    return out;
}
void verify_model(AnyPotential& api) {
    REQUIRE(api.supports_analytical_hessian());REQUIRE(api.supports_analytical_cell_hessian());
    auto atom=geometry(true),reference=atom;
    Neighborhoods nbhs(atom,api.cutoff);api.getEFS(reference,nbhs);
    HessianMatrix h;Eigen::MatrixXd cell,cross;
    api.getEFSHessianCell(atom,nbhs,h,cell,cross);
    const auto full=PairCellTest::native_matrix(h,cell,cross),dense=HessianUtils::to_dense(h);
    REQUIRE((full-full.transpose()).norm()<2e-9*std::max(1.,full.norm()));
    REQUIRE(cell.norm()>1e-7);REQUIRE(cross.norm()>1e-7);
    REQUIRE(std::abs(atom.energy_-reference.energy_)<1e-10);
    REQUIRE((atom.stress_-reference.stress_).norm()<1e-10);
    for(int i=0;i<atom.size();++i) REQUIRE((atom.forces_[i]-reference.forces_[i]).norm()<1e-10);
    for(double step:{2e-5,1e-5}) {
        const auto expected=gradient_fd(api,atom,step);
        INFO("native gradient FD h="<<step);
        REQUIRE((full-expected).norm()<3e-6*std::max(1.,expected.norm()));
    }
    for(double step:{.0005,.00025}) {
        const auto expected=PairCellTest::energy_fd(api,atom,step,true);
        REQUIRE((full-expected).norm()<2e-4*std::max(1.,expected.norm()));
    }
    Eigen::VectorXd direction=Eigen::VectorXd::LinSpaced(3*atom.size(),-.4,.7),product;
    api.applyHessianVectorProduct(atom,nbhs,direction,product);
    REQUIRE((dense*direction-product).norm()<1e-9*std::max(1.,product.norm()));
    auto work=atom;HessianMatrix atomic;api.getEFSHessianOMP(work,nbhs,atomic);
    REQUIRE((HessianUtils::to_dense(atomic)-dense).norm()<1e-9*std::max(1.,dense.norm()));
    HessianSparse sparse;api.getHessian(work,nbhs,{{0,0},{0,2},{2,1}},sparse);
    REQUIRE(sparse.hessian.size()==3);
    REQUIRE((sparse(0,0)-h(0,0)).norm()<1e-9);
    REQUIRE((sparse(0,2)-h(0,2)).norm()<1e-9);
    REQUIRE((sparse(1,2)-h(1,2)).norm()<1e-9);
    std::vector<double> diagonal;api.getEFSHessianSymmDiaOMP(work,diagonal);
    REQUIRE(diagonal.size()==direction.size());
    for(int i=0;i<direction.size();++i) REQUIRE(diagonal[i]==Catch::Approx(dense(i,i)).margin(1e-9));
    api.add_efs=true;api.getEFSHessianCellSymmOMP(atom,h,cell,cross);api.add_efs=false;
    REQUIRE((PairCellTest::native_matrix(h,cell,cross)-full).norm()<1e-9);
    REQUIRE(atom.energy_==Catch::Approx(2*reference.energy_).margin(1e-10));
    REQUIRE((atom.stress_-2*reference.stress_).norm()<1e-10);
    for(int i=0;i<atom.size();++i) REQUIRE((atom.forces_[i]-2*reference.forces_[i]).norm()<1e-10);
    ComputeHessian consumer;
    consumer.method=ComputeHessian::HessianMethod::Analytic;consumer.request_cell=true;
    consumer.use_symmetry=false;consumer.cell_strain=true;consumer.engineering_shear=true;
    Eigen::MatrixXd strain;auto clean=geometry(true);consumer.compute(clean,strain,&api);
    for(double step:{.0005,.00025}) {
        const auto expected=PairCellTest::energy_fd(api,clean,step,false,true);
        REQUIRE((strain-expected).norm()<3e-4*std::max(1.,expected.norm()));
    }
}
}

TEST_CASE("LES reciprocal cell differential retains kernel curvature volume and moving reciprocal vectors",
          "[les][sog][cell][native]") {
    const bool sog=GENERATE(false,true),self=GENERATE(false,true),zero=GENERATE(false,true);
    LES engine;configure_engine(engine,sog,self);engine.sog_include_zero=zero;
    const auto atom=geometry();Eigen::VectorXd q(4),qd(4);q<<.31,-.17,.23,.11;qd<<-.13,.07,.19,-.11;
    Eigen::Matrix3d dc;dc<<.2,-.1,.3,.1,.3,-.2,-.2,.1,-.1;
    SiteCoordinateMatrix velocity(4,3);
    for(int i=0;i<4;++i) velocity.row(i)=atom.pos_[i]*atom.cell_.inverse()*dc+Eigen::RowVector3d(.03*i,-.02*i,.01*i);
    const auto response=engine.evaluate_native_directional(atom,q,qd,velocity,dc);
    const auto base=engine.evaluate(atom,q);
    REQUIRE((response.forces-base.forces).norm()<1e-11);REQUIRE((response.virial-base.virial).norm()<1e-11);
    for(double step:{2e-5,1e-5}) {
        auto plus=atom,minus=atom;plus.cell_+=step*dc;minus.cell_-=step*dc;
        for(int i=0;i<4;++i) {plus.pos_[i]+=step*velocity.row(i);minus.pos_[i]-=step*velocity.row(i);}
        const auto p=engine.evaluate(plus,q+step*qd),m=engine.evaluate(minus,q-step*qd);
        REQUIRE((response.force_directional-(p.forces-m.forces)/(2*step)).norm()<2e-6);
        REQUIRE((response.virial_directional-(p.virial-m.virial)/(2*step)).norm()<2e-6);
        REQUIRE((response.dEdq_directional-(p.dEdq-m.dEdq)/(2*step)).norm()<2e-6);
    }
}
TEST_CASE("Ordinary and model-aware LES provide full native dense cell mixed and requested curvature",
          "[les][sog][cell][dense]") {
    const bool sog=GENERATE(false,true);const int model=GENERATE(0,1,2);
    INFO("SOG="<<sog<<" model="<<model);
    if(model==0) {PotMTP_LES p("unitest/test-data/portable-mtp/les.mtp");initialize(p,sog);verify_model(p);}
    if(model==1) {PotMTP_M<m_model_mtpq2> p;initialize(p,sog);verify_model(p);}
    if(model==2) {PotMTP_M<m_model_nn1> p;initialize(p,sog);verify_model(p);}
}
TEST_CASE("LES hard cutoff keeps Cartesian curvature while rejecting native cell curvature",
          "[les][sog][cell][contract]") {
    PotMTP_LES p("unitest/test-data/portable-mtp/les.mtp");initialize(p,false);p.les_cutoff_mode="reference_hard";
    REQUIRE(p.supports_analytical_hessian());REQUIRE_FALSE(p.supports_analytical_cell_hessian());
    auto atom=geometry();HessianMatrix h;Eigen::MatrixXd c,x;
    REQUIRE_THROWS(p.getEFSHessianCell(atom,h,c,x));
    PotMTP_M<m_model_mtpq2> m;initialize(m,true);m.model_les.cutoff_mode=LES::CutoffMode::ReferenceHard;
    REQUIRE(m.supports_analytical_hessian());REQUIRE_FALSE(m.supports_analytical_cell_hessian());
    REQUIRE_THROWS(m.getEFSHessianCell(atom,h,c,x));
}
