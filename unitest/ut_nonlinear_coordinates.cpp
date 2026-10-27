#include <catch2/catch_all.hpp>
#include "mindriver/nonlinear_coordinates.h"
#include "mindriver/relax_new.h"
#include <Eigen/Geometry>

namespace {
Eigen::VectorXd flatten(const Atoms &a) {
    Eigen::VectorXd x(3*a.size());for(int i=0;i<a.size();++i)x.segment<3>(3*i)=a.pos_[i].transpose();return x;
}
Atoms group_atoms(int count,bool periodic=false) {
    Atoms a;a.cell_=10*Eigen::Matrix3d::Identity();a.pbc_={periodic,false,false};
    a.pos_={{.2,.3,.4}};if(count>1)a.pos_.push_back({1.2,.3,.4});
    if(count>2)a.pos_.push_back({.4,1.3,.7});
    if(periodic && count>1)a.pos_[1][0]=9.2;
    a.pos_.push_back({2.,2.,2.});a.chemmap_.assign(a.size(),0);return a;
}
void check_derivatives(NonlinearCoordinates &map,const Eigen::VectorXd &q) {
    const auto x=map.lift(q);const auto j=map.jacobian(q);
    Eigen::VectorXd g=Eigen::VectorXd::LinSpaced(x.size(),.2,1.4);
    const auto curvature=map.curvature(q,g);
    const auto hessian=map.pullback_hessian(q,x,Eigen::MatrixXd::Identity(x.size(),x.size()));
    for(double h:{1e-5,2e-6})for(int k=0;k<q.size();++k) {
        auto plus=q,minus=q;plus[k]+=h;minus[k]-=h;
        REQUIRE(((map.lift(plus)-map.lift(minus))/(2*h)-j.col(k)).norm()<2e-8);
        REQUIRE(((map.jacobian(plus).transpose()*g-map.jacobian(minus).transpose()*g)/(2*h)-curvature.col(k)).norm()<2e-8);
        const Eigen::VectorXd gp=map.jacobian(plus).transpose()*map.lift(plus);
        const Eigen::VectorXd gm=map.jacobian(minus).transpose()*map.lift(minus);
        REQUIRE(((gp-gm)/(2*h)-hessian.col(k)).norm()<3e-8);
    }
}
struct RigidTarget:Potentials::AnyPotential {
    std::vector<Eigen::RowVector3d> target;
    void getEFS(Atoms &a,AnyNeighborhoods &) override {getEFSOMP(a);}
    void getEFSOMP(Atoms &a) override {
        a.energy_=0;a.stress_.setZero();a.forces_.resize(a.size());
        for(int i=0;i<a.size();++i) {
            const Eigen::RowVector3d delta=a.pos_[i]-target[i];a.energy_+=.5*delta.squaredNorm();a.forces_[i]=-delta.transpose();
        }
    }
};
}
TEST_CASE("Generic nonlinear map includes analytic curvature in an exact Hessian", "[nonlinear][coordinates]") {
    CallbackCoordinates map;
    map.to_full=[](const Eigen::VectorXd &q){Eigen::VectorXd x(2);x<<q[0],q[0]*q[0];return x;};
    map.to_reduced=[](const Eigen::VectorXd &x){return Eigen::VectorXd::Constant(1,x[0]);};
    map.derivative=[](const Eigen::VectorXd &q){Eigen::MatrixXd j(2,1);j<<1,2*q[0];return j;};
    Eigen::VectorXd q=Eigen::VectorXd::Constant(1,.7);
    REQUIRE_THROWS_WITH(map.pullback_hessian(q,map.lift(q),Eigen::MatrixXd::Identity(2,2)),Catch::Matchers::ContainsSubstring("curvature callback"));
    map.second_derivative=[](const Eigen::VectorXd &,const Eigen::VectorXd &g){return Eigen::MatrixXd::Constant(1,1,2*g[1]);};
    check_derivatives(map,q);
    REQUIRE(map.pullback_hessian(q,map.lift(q),Eigen::MatrixXd::Identity(2,2))(0,0)==Catch::Approx(1+6*.7*.7));
}
TEST_CASE("Rigid single linear and noncollinear groups have exact analytic rotation derivatives", "[rigid][coordinates][fd]") {
    const int count=GENERATE(1,2,3);const bool periodic=GENERATE(false,true);
    auto a=group_atoms(count,periodic);std::vector<int> ids;for(int i=0;i<count;++i)ids.push_back(i);
    RigidBodyCoordinates map(a,{ids});const auto original=flatten(a);map.initialize(original);
    REQUIRE(map.dimension()==(count==1?3:count==2?5:6)+3);
    auto q=map.reduce(original);REQUIRE((map.lift(q)-original).norm()<1e-12);
    q.array()+=.07;check_derivatives(map,q);
    const auto full=map.lift(q);REQUIRE((map.lift(map.reduce(full))-full).norm()<1e-10);
    if(count>1) {
        Atoms pair;pair.cell_=a.cell_;pair.pbc_=a.pbc_;pair.pos_={a.pos_[0],a.pos_[1]};
        const double length=pair.get_rab(0,1).norm();
        pair.pos_[0]=full.segment<3>(0).transpose();pair.pos_[1]=full.segment<3>(3).transpose();
        REQUIRE(pair.get_rab(0,1).norm()==Catch::Approx(length).margin(1e-11));
        q[3]=.4;const auto before=map.lift(q);REQUIRE(map.accepted(q));
        const auto reset=map.reduce(before);REQUIRE(reset.segment(3,count==2?2:3).norm()<1e-10);
        REQUIRE((map.lift(reset)-before).norm()<1e-10);check_derivatives(map,reset);
    }
}
TEST_CASE("V2 rigid groups converge with GD FIRE and LBFGS while retaining exact geometry", "[rigid][relax]") {
    const std::string optimizer=GENERATE("gd","fire","lbfgs");
    auto a=group_atoms(3);RigidTarget p;p.target=a.pos_;
    const Eigen::Matrix3d rotation=Eigen::AngleAxisd(.45,Eigen::Vector3d::UnitZ()).toRotationMatrix();
    const Eigen::RowVector3d center=(a.pos_[0]+a.pos_[1]+a.pos_[2])/3.;
    for(int i=0;i<3;++i)p.target[i]=((a.pos_[i]-center)*rotation+center+Eigen::RowVector3d(.2,.3,.4)).eval();
    p.target[3]+=Eigen::RowVector3d(-.1,.2,-.1);
    RelaxationDrvV2 d;d.flags.cartesian=true;d.control.rigid_groups="0,1,2";
    d.control.driver=optimizer;d.control.max_iter=4000;d.control.max_move=.1;d.control.nblock=100000;d.control.indexfile.clear();d.tol.gtol=2e-5;
    d.init(&p,a);REQUIRE(d.run()==0);
    for(int i=0;i<a.size();++i)REQUIRE((d.atom.pos_[i]-p.target[i]).norm()<1e-4);
    for(int i=0;i<3;++i)for(int k=i+1;k<3;++k)
        REQUIRE((d.atom.pos_[i]-d.atom.pos_[k]).norm()==Catch::Approx((a.pos_[i]-a.pos_[k]).norm()).margin(2e-11));
    REQUIRE(d.atom.energy_<1e-8);REQUIRE(d.state_vars.converged);
}

TEST_CASE("V2 generic nonlinear callbacks optimize on their physical manifold", "[nonlinear][relax]") {
    const std::string optimizer=GENERATE("gd","fire","lbfgs");
    auto map=std::make_shared<CallbackCoordinates>();
    map->to_full=[](const Eigen::VectorXd &q){Eigen::VectorXd x(3);x<<q[0],q[1],q.squaredNorm();return x;};
    map->to_reduced=[](const Eigen::VectorXd &x){return x.head(2).eval();};
    map->derivative=[](const Eigen::VectorXd &q){Eigen::MatrixXd j=Eigen::MatrixXd::Zero(3,2);j.topRows(2).setIdentity();j.row(2)=2*q.transpose();return j;};
    map->second_derivative=[](const Eigen::VectorXd &,const Eigen::VectorXd &g){return (2*g[2]*Eigen::MatrixXd::Identity(2,2)).eval();};
    Atoms a;a.cell_=10*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};a.chemmap_={0};a.pos_={{.2,.3,.13}};
    RigidTarget p;p.target={{.6,.5,.61}};
    RelaxationDrvV2 d;d.flags.cartesian=true;d.coordinate_map=map;d.control.driver=optimizer;
    d.control.max_iter=3000;d.control.max_move=.1;d.control.nblock=100000;d.control.indexfile.clear();d.tol.gtol=1e-6;
    d.init(&p,a);REQUIRE(d.run()==0);
    REQUIRE((d.atom.pos_[0]-p.target[0]).norm()<3e-6);
    REQUIRE(d.atom.pos_[0][2]==Catch::Approx(std::pow(d.atom.pos_[0][0],2)+std::pow(d.atom.pos_[0][1],2)).margin(1e-12));
}

TEST_CASE("Callback coordinate initialization receives its reference before reduction", "[nonlinear][initialization]") {
    auto a=group_atoms(1);a.pos_.resize(1);a.chemmap_.resize(1);RigidTarget pot;pot.target=a.pos_;
    Eigen::VectorXd origin;int initialized=0;
    auto map=std::make_shared<CallbackCoordinates>();
    map->initializer=[&](const Eigen::VectorXd &x){origin=x;++initialized;};
    map->to_reduced=[&](const Eigen::VectorXd &x){REQUIRE(initialized==1);return Eigen::VectorXd(x-origin);};
    map->to_full=[&](const Eigen::VectorXd &q){return Eigen::VectorXd(origin+q);};
    map->derivative=[](const Eigen::VectorXd &q){return Eigen::MatrixXd(Eigen::MatrixXd::Identity(q.size(),q.size()));};
    RelaxationDrvV2 d;d.flags.cartesian=true;d.coordinate_map=map;d.control.max_iter=0;
    d.init(&pot,a);REQUIRE(initialized==1);REQUIRE(origin.isApprox(a.pos_[0].transpose()));
    REQUIRE(d.run()==0);REQUIRE(initialized==1);
}
