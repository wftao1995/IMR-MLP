#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include <filesystem>
#include <fstream>

namespace {
struct Harmonic : Potentials::AnyPotential {
    int calls=0,fail_after=0;
    Harmonic(){ cutoff=2; }
    void getEFS(Atoms &a,AnyNeighborhoods &) override {
        if(fail_after && ++calls>fail_after) throw std::runtime_error("injected harmonic failure");
        a.resetEFS();
        for(int i=0;i<a.size();++i) {a.energy_+=.5*a.pos_[i].squaredNorm();a.forces_[i]=-a.pos_[i];}
    }
    void getEFSOMP(Atoms &a) override { Neighborhoods n(a,cutoff);getEFS(a,n); }
};
Atoms start(){ Atoms a;a.pbc_={false,false,false};a.cell_.setZero();a.pos_={{1,0,0}};a.chemmap_={0};return a; }
GPSettings settings(int n=3){ GPSettings s; s.Modify("driver","mdmin");s.Modify("cartesian","true");
    s.Modify("atomic_force_recast","false");s.Modify("write_traj","false");s.Modify("maxiteration",std::to_string(n));
    s.Modify("stepmax","100");s.Modify("gtol","0");return s; }
}
TEST_CASE("V2 MDMin matches independent velocity-Verlet harmonic steps", "[relax][mdmin]") {
    const int n=GENERATE(0,1,2,3);
    const double expected[]={1.,.98,.9208,.824768};
    Harmonic p;auto a=start();RelaxationDrvV2 d(settings(n));d.init(&p,a);
    REQUIRE(d.run()!=0);
    REQUIRE(d.state_vars.curr_iter==n);
    REQUIRE(d.atom.pos_[0][0]==Catch::Approx(expected[n]).margin(1e-13));
    REQUIRE(d.f==Catch::Approx(.5*expected[n]*expected[n]));
    REQUIRE(d.g[0]==Catch::Approx(expected[n]));
    REQUIRE(a.pos_[0][0]==1.);
    REQUIRE(d.stop_reason=="iteration_limit");
}
TEST_CASE("MDMin quenches uphill momentum and limits the actual displacement", "[relax][mdmin]") {
    Harmonic p;auto a=start();auto s=settings(2);s.Modify("mdmin:dt","1.6");
    RelaxationDrvV2 d(s);d.init(&p,a);d.run();
    // x1=-.28; old v=-.8, half kick .224 has negative power => reset.
    // The final half kick .224 gives x2=.0784.
    REQUIRE(d.atom.pos_[0][0]==Catch::Approx(.0784).margin(1e-12));
    s=settings(1);s.Modify("mdmin:dt","10");s.Modify("stepmax",".03");
    RelaxationDrvV2 clipped(s);clipped.init(&p,a);clipped.run();
    REQUIRE(clipped.atom.pos_[0][0]==Catch::Approx(.97));
}
TEST_CASE("MDMin convergence, repeated run and selective coordinates use current accepted state", "[relax][mdmin]") {
    Harmonic p;auto a=start();a.pos_[0][1]=2;a.sel_dynpos_={{1,0,1}};
    auto s=settings(200);s.Modify("gtol","1e-8");s.Modify("selective_dynamics","true");
    RelaxationDrvV2 d(s);d.init(&p,a);REQUIRE(d.run()==0);
    REQUIRE(std::abs(d.atom.pos_[0][0])<1e-8);REQUIRE(d.atom.pos_[0][1]==2);
    REQUIRE(d.f==Catch::Approx(2));
    REQUIRE(d.run()==0);REQUIRE(d.state_vars.curr_iter==0);
}
TEST_CASE("MDMin invalid timestep and failed force calls do not publish a trial", "[relax][mdmin]") {
    const double dt=GENERATE(0.,-1.,std::numeric_limits<double>::infinity());
    Harmonic p;auto a=start();RelaxationDrvV2 d(settings());d.mdmin_dt=dt;
    REQUIRE_THROWS_AS(d.init(&p,a),std::invalid_argument);
    auto s=settings(3);Harmonic failing;failing.fail_after=2;
    RelaxationDrvV2 f(s);f.init(&failing,a);
    REQUIRE_THROWS_WITH(f.run(),Catch::Matchers::ContainsSubstring("injected"));
    REQUIRE(f.atom.pos_[0][0]==Catch::Approx(.98));REQUIRE(f.f==Catch::Approx(.5*.98*.98));
    REQUIRE(f.stop_reason=="error");REQUIRE_FALSE(f.state_vars.converged);
}
