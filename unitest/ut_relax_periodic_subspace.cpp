#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include <Eigen/Geometry>

namespace {
Atoms geometry(int rank) {
    Atoms a;a.cell_.setZero();a.pbc_={false,false,false};
    const auto rotation=Eigen::AngleAxisd(.29,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    if(rank) {a.cell_.row(0)=Eigen::RowVector3d(7,0,0)*rotation;a.pbc_[0]=true;}
    if(rank==2) {a.cell_.row(2)=Eigen::RowVector3d(.8,8,0)*rotation;a.pbc_[2]=true;}
    a.pos_={Eigen::RowVector3d(.4,.6,1.2)*rotation,Eigen::RowVector3d(1.2,.8,2.1)*rotation};
    a.chemmap_={0,0};return a;
}
struct IndependentPotential:Potentials::AnyPotential {
    std::vector<Eigen::RowVector3d> positions;
    Eigen::Matrix3d cell=Eigen::Matrix3d::Zero();
    void getEFS(Atoms &a,AnyNeighborhoods &) override {getEFSOMP(a);}
    void getEFSOMP(Atoms &a) override {
        a.energy_=0;a.stress_.setZero();a.forces_.resize(a.size());
        for(int i=0;i<a.size();++i) {
            const Eigen::RowVector3d delta=a.pos_[i]-positions[i];
            a.energy_+=.5*delta.squaredNorm();a.forces_[i]=-delta.transpose();
            a.stress_-=a.pos_[i].transpose()*delta;
        }
        for(int i=0;i<3;++i) if(a.pbc_[i]) {
            const Eigen::RowVector3d delta=a.cell_.row(i)-cell.row(i);
            a.energy_+=.5*delta.squaredNorm();a.stress_-=a.cell_.row(i).transpose()*delta;
        }
    }
};
void configure(RelaxationDrvV2 &d,bool recast,bool variable) {
    d.flags.periodic_subspace=true;d.flags.atomic_force_recast=recast;d.flags.relax_cell=variable;
    d.flags.cartesian=false;d.control.indexfile.clear();d.control.nblock=100000;
}
}
TEST_CASE("V2 periodic-subspace objective and atomic constraints have independent exact derivatives", "[relax][subspace][fd]") {
    const int rank=GENERATE(0,1,2);const bool recast=GENERATE(false,true),variable=GENERATE(false,true);
    auto a=geometry(rank);IndependentPotential p;p.positions={Eigen::RowVector3d(.1,.2,.3),Eigen::RowVector3d(.8,.4,1.)};p.cell=1.03*a.cell_;
    RelaxationDrvV2 d;configure(d,recast,variable);d.control.driver="sqp";
    regulator distance;distance.regulator_t=regulator::DISTANCE;distance.integer_specs={0,1};distance.type=10;
    distance.to_target_g0=true;distance.double_target=1.4;distance.tol=1e-8;
    std::vector<regulator> constraints{distance};d.init(&p,a,&constraints);
    d.apply_modifiers(d.modifiers_list);auto point=d.x;point[point.size()-1]+=.03;
    if(d.xposoffset)point[0]+=.04;
    Eigen::VectorXd gradient=Eigen::VectorXd::Zero(point.size());d(point,gradient);const auto jac=d.constraint_jacobian;
    REQUIRE(d.xposoffset==(variable?rank*(rank+1)/2:0));
    REQUIRE(d.atom.cartesian);REQUIRE_FALSE(d.flags.cartesian);REQUIRE(d.atom.cell_.row(1).isZero());
    for(double h:{1e-5,2e-6}) for(int j=0;j<point.size();++j) {
        auto plus=point,minus=point;plus[j]+=h;minus[j]-=h;Eigen::VectorXd scratch=Eigen::VectorXd::Zero(point.size());
        const double fp=d(plus,scratch),cp=d.constraints[0].f;
        const double fm=d(minus,scratch),cm=d.constraints[0].f;
        REQUIRE(gradient[j]==Catch::Approx((fp-fm)/(2*h)).margin(3e-7));
        REQUIRE(jac(0,j)==Catch::Approx((cp-cm)/(2*h)).margin(2e-8));
    }
    d(point,gradient);REQUIRE(d.atom.cell_.row(1).isZero());
}
TEST_CASE("V2 periodic-subspace minimization reaches the independent Cartesian and periodic-cell target", "[relax][subspace][optimization]") {
    const int rank=GENERATE(0,1,2);const bool recast=GENERATE(false,true),variable=GENERATE(false,true);
    auto a=geometry(rank);IndependentPotential p;p.positions=a.pos_;
    for(auto &pos:p.positions)pos+=Eigen::RowVector3d(.1,.15,-.2);
    p.cell=(variable?1.02:1.)*a.cell_;
    RelaxationDrvV2 d;configure(d,recast,variable);d.control.driver="lbfgs";d.control.max_iter=500;
    d.control.max_move=.2;d.tol.gtol=1e-7;d.init(&p,a);REQUIRE(d.run()==0);
    REQUIRE(d.atom.cartesian);REQUIRE(d.atom.cell_.row(1).isZero());
    REQUIRE((d.atom.cell_-p.cell).norm()<2e-6);
    for(int i=0;i<a.size();++i)REQUIRE((d.atom.pos_[i]-p.positions[i]).norm()<2e-6);
    REQUIRE(d.atom.energy_<1e-10);REQUIRE(a.cell_.row(1).isZero());
}
TEST_CASE("Periodic-subspace V2 rejects singular stored fractions and 3D pressure or volume", "[relax][subspace]") {
    auto a=geometry(2);IndependentPotential p;p.positions=a.pos_;p.cell=a.cell_;
    const int invalid=GENERATE(0,1,2);RelaxationDrvV2 d;configure(d,true,true);
    if(invalid==0)a.cartesian=false;
    if(invalid==1)d.control.pressure=1;
    if(invalid==2)d.flags.fix_vol=true;
    REQUIRE_THROWS(d.init(&p,a));
}
