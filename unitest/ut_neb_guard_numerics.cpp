// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "mindriver/neb.h"
#include <cmath>
#include <limits>
#include <string>

namespace {
class NonfiniteBarrier : public Potentials::AnyPotential {
public:
    double injected_value = 0;
    int field = 0; // force, energy, or virial
    NonfiniteBarrier() { cutoff=3; cutoffsq=9; species_count=1; }
    void evaluate(Atoms &a) {
        a.resetEFS();
        const double x=a.pos_[0](0)-10;
        a.energy_=(x*x-1)*(x*x-1);
        a.forces_[0](0)=-4*x*(x*x-1);
        // Endpoints remain valid. Only the middle image has a failed EFS
        // evaluation, so this exercises the NEB convergence reduction.
        if (std::abs(x)<0.1) {
            if (field==0) a.forces_[0](1)=injected_value;
            else if (field==1) a.energy_+=injected_value;
            else a.stress_(1,2)=injected_value;
        }
    }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { evaluate(a); }
    void getEFSOMPthread(Atoms &a) override { evaluate(a); }
};

class TransverseQuadratic : public Potentials::AnyPotential {
public:
    double z_stiffness = 0;
    TransverseQuadratic() { cutoff=3; cutoffsq=9; species_count=1; }
    void evaluate(Atoms &a) {
        a.resetEFS();
        const double y=a.pos_[0](1);
        const double z=a.pos_[0](2);
        a.energy_=.5*(y*y+z_stiffness*z*z);
        a.forces_[0](1)=-y;
        a.forces_[0](2)=-z_stiffness*z;
    }
    void getEFS(Atoms &a, AnyNeighborhoods &) override { evaluate(a); }
    void getEFSOMPthread(Atoms &a) override { evaluate(a); }
};
}

TEST_CASE("NEB guard tries Newton after a small accepted force-flow step", "[neb][numerics][guard]") {
    TransverseQuadratic pot;
    pot.z_stiffness=1e4;
    Atoms a;
    a.cell_=Eigen::Matrix3d::Identity(); a.pbc_={false,false,false};
    a.pos_={{-1,0,0}}; a.chemmap_={0};
    std::vector<Atoms> path(3,a);
    path[1].pos_[0]<<0,1,1e-6; path[2].pos_[0]<<1,0,0;
    NEBDriver driver;
    driver.pot=&pot; driver.check_mic=false; driver.tangent_type=2;
    driver.spring=0; driver.climb=false; driver.neb_ftol=1e-12;
    driver.init(path);
    Eigen::VectorXd x,g;
    driver.LatPos2X(x); g.resize(x.size());
    double objective=driver(x,g);
    const Eigen::VectorXd initial=x;
    const double initial_residual=g.squaredNorm();
    REQUIRE(initial_residual==Catch::Approx(1.0001));
    // For diag(1, 1e4), the first descending force-flow trial uses
    // alpha=(.01/sqrt(1.0001))/32 and improves the residual by only 0.027%.
    // A capped exact Newton correction improves it by 1.99%. Retaining a
    // valid small step must not prevent the guard from trying that correction.
    constexpr double max_move=.01;
    const double alpha=max_move/std::sqrt(initial_residual)/32;
    const double flow_residual=std::pow(1-alpha,2)+1e-4*std::pow(1-1e4*alpha,2);
    REQUIRE(flow_residual<initial_residual);
    REQUIRE(flow_residual>.99*initial_residual);
    int callbacks=0;
    auto stationary_optimizer=[&](int) { ++callbacks; return 1; };
    const int used=driver.minimize_guarded(stationary_optimizer,2,max_move,x,objective);
    driver(x,g);
    CAPTURE(used,callbacks,initial_residual,flow_residual,g.squaredNorm());
    CHECK(used==2);
    CHECK(callbacks==1);
    CHECK(driver.stop_reason!=NEBDriver::StopReason::numerical_failure);
    CHECK(g.squaredNorm()<.99*initial_residual);
    CHECK((x-initial).norm()<=max_move+1e-12);
    CHECK(driver.images.front().pos_[0].isApprox(path.front().pos_[0],1e-14));
    CHECK(driver.images.back().pos_[0].isApprox(path.back().pos_[0],1e-14));
    for (int i=0;i<3;++i) CHECK(driver.images[i].cell_.isApprox(path[i].cell_,1e-14));
}

TEST_CASE("NEB guard never accepts an uncapped GMRES probe", "[neb][numerics][guard]") {
    TransverseQuadratic pot;
    Atoms a;
    a.cell_=Eigen::Matrix3d::Identity(); a.pbc_={false,false,false};
    a.pos_={{-1,0,0}}; a.chemmap_={0};
    std::vector<Atoms> path(3,a);
    path[1].pos_[0]<<0,1e-5,0; path[2].pos_[0]<<1,0,0;
    NEBDriver driver;
    driver.pot=&pot; driver.check_mic=false; driver.tangent_type=2;
    driver.spring=0; driver.climb=false; driver.neb_ftol=1e-12;
    driver.init(path);
    Eigen::VectorXd x,g;
    driver.LatPos2X(x); g.resize(x.size());
    double objective=driver(x,g);
    const Eigen::VectorXd initial=x;
    const double initial_residual=g.squaredNorm();
    constexpr double max_move=1e-9;
    // The first finite-difference probe (delta=1e-5) lands at the root,
    // but it exceeds max_move by 1e4. Probes are not accepted iterations.
    int callbacks=0;
    auto stationary_optimizer=[&](int) { ++callbacks; return 1; };
    const int used=driver.minimize_guarded(stationary_optimizer,2,max_move,x,objective);
    driver(x,g);
    CAPTURE(used,callbacks,(x-initial).norm(),g.squaredNorm());
    CHECK(used==2);
    CHECK(callbacks==1);
    CHECK(driver.stop_reason!=NEBDriver::StopReason::numerical_failure);
    CHECK((x-initial).norm()<=max_move*(1+1e-6));
    CHECK_FALSE(driver.check_convergence());
    CHECK(g.squaredNorm()<initial_residual);
    CHECK(driver.images.front().pos_[0].isApprox(path.front().pos_[0],1e-14));
    CHECK(driver.images.back().pos_[0].isApprox(path.back().pos_[0],1e-14));
}

TEST_CASE("NEB guard never converts nonfinite EFS into convergence", "[neb][numerics][guard]") {
    const int invalid=GENERATE(0,1,2);
    const bool fractional_vc=GENERATE(false,true);
    const int tangent=GENERATE(1,2);
    NonfiniteBarrier pot;
    pot.field=GENERATE(0,1,2);
    if (invalid==1) pot.injected_value=std::numeric_limits<double>::quiet_NaN();
    if (invalid==2) pot.injected_value=std::numeric_limits<double>::infinity();
    Atoms a;
    a.cell_=20*Eigen::Matrix3d::Identity();
    a.pos_={{9,10,10}}; a.chemmap_={0};
    std::vector<Atoms> path(3,a);
    path[1].pos_[0](0)=10; path[2].pos_[0](0)=11;
    NEBDriver driver;
    driver.pot=&pot; driver.check_mic=false; driver.tangent_type=tangent;
    driver.vcneb=fractional_vc; driver.recast_atomic_force=false;
    driver.sel_dyn=true; driver.cmask="000000000";
    driver.neb_ftol=1e-6;
    driver.init(path);
    int status=-1;
    bool rejected=false;
    std::string error;
    try { status=driver.run("lbfgs",9,.03); }
    catch (const std::exception &e) { rejected=true; error=e.what(); }
    CAPTURE(invalid,pot.field,fractional_vc,tangent,status,rejected,error,driver.max_neb_force,driver.stop_reason_string());
    if (invalid==0) {
        CHECK_FALSE(rejected);
        CHECK(status==0);
        CHECK(driver.converged);
    } else {
        CHECK((rejected || status!=0));
        CHECK_FALSE(driver.converged);
        if (!rejected) CHECK(driver.stop_reason==NEBDriver::StopReason::numerical_failure);
    }
}

TEST_CASE("NEB finite evaluation recovers after an invalid trial", "[neb][numerics][guard]") {
    const bool fractional_vc=GENERATE(false,true);
    const int tangent=GENERATE(1,2);
    const bool infinite=GENERATE(false,true);
    NonfiniteBarrier pot;
    pot.field=GENERATE(0,1,2);
    Atoms a;
    a.cell_=20*Eigen::Matrix3d::Identity();
    a.pos_={{9,10,10}}; a.chemmap_={0};
    a.sel_dynpos_={Eigen::RowVector3d::Ones()};
    std::vector<Atoms> path(3,a);
    path[1].pos_[0](0)=10; path[2].pos_[0](0)=11;
    NEBDriver driver;
    driver.pot=&pot; driver.check_mic=false; driver.tangent_type=tangent;
    driver.vcneb=fractional_vc; driver.recast_atomic_force=false;
    driver.sel_dyn=true; driver.cmask="000000000";
    driver.init(path);
    Eigen::VectorXd x,g;
    driver.LatPos2X(x); g.resize(x.size());
    const double initial_objective=driver(x,g);
    REQUIRE(driver.check_convergence());
    pot.injected_value=infinite ? std::numeric_limits<double>::infinity()
                               : std::numeric_limits<double>::quiet_NaN();
    const double invalid_objective=driver(x,g);
    CAPTURE(fractional_vc,tangent,infinite,pot.field);
    CHECK_FALSE(std::isfinite(invalid_objective));
    CHECK_FALSE(g.allFinite());
    CHECK_FALSE(driver.check_convergence());
    CHECK(driver.stop_reason==NEBDriver::StopReason::not_run);
    pot.injected_value=0;
    const double recovered_objective=driver(x,g);
    CHECK(recovered_objective==Catch::Approx(initial_objective));
    CHECK(g.allFinite());
    CHECK(g.isZero());
    CHECK(driver.check_convergence());
    CHECK(driver.stop_reason==NEBDriver::StopReason::not_run);
    for (int i=0;i<3;++i) {
        CHECK(driver.images[i].cell_.isApprox(path[i].cell_,1e-14));
        CHECK(driver.images[i].pos_[0].isApprox(path[i].pos_[0],1e-14));
    }
}

TEST_CASE("NEB guard accepts bounded progress below a fixed relative reduction", "[neb][numerics][guard]") {
    const double height=GENERATE(100.,250.);
    // Zero selects fixed-cell Cartesian NEB. Nonzero lengths select the
    // fractional VC representation while constraining every cell component.
    const double cell_length=GENERATE(0.,.2,20.);
    const double length=cell_length==0 ? 1 : cell_length;
    constexpr double max_move=.01;
    TransverseQuadratic pot;
    Atoms a;
    a.cell_=length*Eigen::Matrix3d::Identity(); a.pbc_={false,false,false};
    a.pos_={{-1,0,0}}; a.chemmap_={0};
    a.sel_dynpos_={Eigen::RowVector3d::Ones()};
    std::vector<Atoms> path(3,a);
    path[1].pos_[0]<<0,height,0; path[2].pos_[0]<<1,0,0;
    NEBDriver driver;
    driver.pot=&pot; driver.check_mic=false; driver.tangent_type=2;
    driver.vcneb=cell_length!=0; driver.recast_atomic_force=false;
    driver.sel_dyn=true; driver.cmask="000000000";
    driver.neb_ftol=1e-8;
    driver.init(path);
    Eigen::VectorXd x,g;
    driver.LatPos2X(x); g.resize(x.size());
    double objective=driver(x,g);
    const int yindex=driver.vcoffset+1;
    REQUIRE(g(yindex)==Catch::Approx(length*height));
    const Eigen::VectorXd initial=x;
    const double initial_residual=g.squaredNorm();
    // At fixed x=0, the original tangent is exactly horizontal and both
    // spring lengths agree. The complete NEB residual is therefore |y|.
    // A .01 move from y=250 improves its square by only 7.99984e-5 relative:
    // rejecting all such small improvements makes both force flow and exact
    // Newton directions fail despite a perfectly conditioned linear problem.
    const double greatest_relative_reduction=1-std::pow(1-length*max_move/height,2);
    if (height==250 && length<=1) REQUIRE(greatest_relative_reduction<1e-4);
    int callbacks=0;
    auto stationary_optimizer=[&](int) { ++callbacks; return 1; };
    const int used=driver.minimize_guarded(stationary_optimizer,2,max_move,x,objective);
    CAPTURE(height,cell_length,greatest_relative_reduction,used,callbacks,driver.stop_reason_string());
    CHECK(used==2);
    CHECK(callbacks==1);
    CHECK(driver.stop_reason!=NEBDriver::StopReason::numerical_failure);
    CHECK(x(yindex)<initial(yindex));
    CHECK((x-initial).norm()<=max_move+1e-12);
    CHECK(driver.images[1].pos_[0].transpose().isApprox(length*x.tail(3),1e-14));
    const double physical_y=driver.images[1].pos_[0](1);
    CHECK(physical_y<height);
    CHECK(objective==Catch::Approx(.5*physical_y*physical_y+
          driver.spring*(1+physical_y*physical_y)/(length*length)));
    for (int i=0;i<3;++i) CHECK(driver.images[i].cell_.isApprox(path[i].cell_,1e-14));
    CHECK(driver.images.front().pos_[0].isApprox(path.front().pos_[0],1e-14));
    CHECK(driver.images.back().pos_[0].isApprox(path.back().pos_[0],1e-14));
    driver(x,g);
    CHECK(g.squaredNorm()<initial_residual);
}
