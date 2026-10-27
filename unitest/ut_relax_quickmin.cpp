#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"

namespace {
struct Anisotropic : Potentials::AnyPotential {
    Eigen::Vector3d stiffness{1,1,1};
    Anisotropic(){cutoff=2;}
    void getEFS(Atoms &a,AnyNeighborhoods &) override {
        a.resetEFS();
        for(int i=0;i<a.size();++i){a.energy_+=.5*(a.pos_[i].array().square()*stiffness.transpose().array()).sum();
            a.forces_[i]=-(a.pos_[i].array()*stiffness.transpose().array()).matrix().transpose();}
    }
    void getEFSOMP(Atoms &a) override {Neighborhoods n(a,cutoff);getEFS(a,n);}
};
Atoms initial(){Atoms a;a.pbc_={false,false,false};a.cell_.setZero();a.pos_={{1,0,0}};a.chemmap_={0};return a;}
GPSettings settings(int steps){GPSettings s;s.Modify("driver","quickmin");s.Modify("cartesian","true");
    s.Modify("atomic_force_recast","false");s.Modify("maxiteration",std::to_string(steps));s.Modify("stepmax","100");
    s.Modify("write_traj","false");s.Modify("gtol","0");return s;}
}
TEST_CASE("QuickMin full-kick harmonic steps differ from MDMin half kicks", "[relax][quickmin]"){
    const int steps=GENERATE(0,1,2);const double expected[]={1.,.96,.8816};
    auto a=initial();Anisotropic p;RelaxationDrvV2 d(settings(steps));d.init(&p,a);d.run();
    REQUIRE(d.state_vars.curr_iter==steps);REQUIRE(d.atom.pos_[0][0]==Catch::Approx(expected[steps]));
    REQUIRE(d.f==Catch::Approx(.5*expected[steps]*expected[steps]));REQUIRE(a.pos_[0][0]==1.);
}
TEST_CASE("QuickMin projects the complete velocity and quenches negative power", "[relax][quickmin]"){
    auto a=initial();a.pos_[0][1]=2;Anisotropic p;p.stiffness<<1,3,1;
    RelaxationDrvV2 d(settings(2));d.init(&p,a);d.run();
    // F0=(-1,-6), x1=(.96,1.76), projection coefficient6.528/28.8.
    REQUIRE(d.atom.pos_[0][0]==Catch::Approx(.87808));
    REQUIRE(d.atom.pos_[0][1]==Catch::Approx(1.30944));
    a=initial();p.stiffness.setOnes();auto s=settings(2);s.Modify("quickmin:dt","1.2");
    RelaxationDrvV2 quench(s);quench.init(&p,a);quench.run();
    // x1=-.44, uphilloldvelocity -1.2 is discarded; x2=-.44+1.44*.44.
    REQUIRE(quench.atom.pos_[0][0]==Catch::Approx(.1936));
}
TEST_CASE("QuickMin respects movement limits, masks and convergence", "[relax][quickmin]"){
    auto a=initial();Anisotropic p;auto s=settings(1);s.Modify("quickmin:dt","10");s.Modify("stepmax",".03");
    RelaxationDrvV2 clipped(s);clipped.init(&p,a);clipped.run();REQUIRE(clipped.atom.pos_[0][0]==Catch::Approx(.97));
    s=settings(300);s.Modify("gtol","1e-8");s.Modify("selective_dynamics","true");a.pos_[0][1]=2;a.sel_dynpos_={{1,0,1}};
    RelaxationDrvV2 d(s);d.init(&p,a);REQUIRE(d.run()==0);REQUIRE(std::abs(d.atom.pos_[0][0])<1e-8);
    REQUIRE(d.atom.pos_[0][1]==2);REQUIRE(d.f==Catch::Approx(2));
    d.quickmin_dt=0;REQUIRE_THROWS_AS(d.run(),std::invalid_argument);REQUIRE(d.atom.pos_[0][1]==2);
}
