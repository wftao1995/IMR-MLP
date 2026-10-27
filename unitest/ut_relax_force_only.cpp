#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
namespace {
struct ProjectedForces:Potentials::AnyPotential {
 double fail_below=-1;
 ProjectedForces(){cutoff=1;}
 void getEFS(Atoms&a,AnyNeighborhoods&)override{
  if(a.pos_[0][0]<fail_below)throw std::runtime_error("force trial failure");
  a.resetEFS();a.energy_=17;for(int i=0;i<a.size();++i)a.forces_[i]=-a.pos_[i];
 }
 void getEFSOMP(Atoms&a)override{Neighborhoods n(a,cutoff);getEFS(a,n);}
};
Atoms geometry(){Atoms a;a.pbc_={false,false,false};a.cell_.setZero();a.pos_={{1,0,0}};a.chemmap_={0};return a;}
GPSettings settings(){GPSettings s;s.Modify("driver","lbfgspp");s.Modify("cartesian","true");s.Modify("atomic_force_recast","false");s.Modify("lbfgspp:linesearch","4");s.Modify("lbfgspp:epsilon","0");s.Modify("lbfgspp:epsilon_rel","0");s.Modify("gtol","1e-9");s.Modify("stepmax",".2");s.Modify("maxiteration","30");s.Modify("write_traj","false");return s;}
}
TEST_CASE("V2 force-only LBFGSpp ignores constant reported energy and retains masks", "[relax][force-only]"){
 auto a=geometry();a.pos_[0][1]=2;a.sel_dynpos_={{1,0,1}};auto s=settings();s.Modify("selective_dynamics","true");s.Modify("lbfgspp:past","1");s.Modify("lbfgspp:delta","1");
 ProjectedForces p;RelaxationDrvV2 d(s);d.init(&p,a);REQUIRE(d.run()==0);REQUIRE(d.state_vars.converged);REQUIRE(std::abs(d.atom.pos_[0][0])<1e-9);REQUIRE(d.atom.pos_[0][1]==2);REQUIRE(d.f==17);REQUIRE(d.state_vars.curr_iter>1);REQUIRE(a.pos_[0][0]==1);
}
TEST_CASE("V2 force-only LBFGSpp retains current state at a budget limit", "[relax][force-only]"){
 auto a=geometry();auto s=settings();s.Modify("maxiteration","1");ProjectedForces p;RelaxationDrvV2 d(s);d.init(&p,a);REQUIRE(d.run()!=0);REQUIRE(d.state_vars.curr_iter==1);REQUIRE(d.atom.pos_[0][0]==Catch::Approx(.8));REQUIRE(d.g[0]==Catch::Approx(.8));REQUIRE(d.f==17);REQUIRE_FALSE(d.state_vars.converged);
}
TEST_CASE("V2 force-only LBFGSpp restores the last accepted state after a failed trial", "[relax][force-only]"){
 auto a=geometry();ProjectedForces p;p.fail_below=.7;RelaxationDrvV2 d(settings());d.init(&p,a);REQUIRE_THROWS_WITH(d.run(),Catch::Matchers::ContainsSubstring("force trial failure"));REQUIRE(d.atom.pos_[0][0]==Catch::Approx(.8));REQUIRE(d.x[0]==Catch::Approx(.8));REQUIRE(d.g[0]==Catch::Approx(.8));REQUIRE(d.state_vars.curr_iter==1);REQUIRE(d.f==17);
}
