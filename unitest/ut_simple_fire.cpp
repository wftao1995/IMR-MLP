#include <catch2/catch_all.hpp>
#include "mindriver/fire.h"
#include <cmath>

using Catch::Approx;

namespace {
struct CoupledQuadratic {
    std::vector<Eigen::VectorXd> samples;
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        samples.push_back(x);
        g.resize(2);
        g << 2*x[0]+x[1], x[0]+3*x[1];
        // A force-only caller may return a reporting value which is not an
        // objective for its effective force (as in NEB or dimer).
        return static_cast<double>(samples.size());
    }
};
struct ScalarQuadratic {
    double stiffness;
    std::vector<double> samples;
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        samples.push_back(x[0]);
        g = stiffness*x;
        return .5*stiffness*x.squaredNorm();
    }
};
}

TEST_CASE("Shared force-only FIRE starts at zero velocity and mixes a coupled two dimensional force", "[simple_fire]") {
    CoupledQuadratic potential;
    Eigen::VectorXd x(2); x << 1,-1;
    SimpleFireF fire;
    fire.ts_init=.1;
    fire.max_step=10;
    fire.max_iter=2;
    fire.gtol=0;
    double report=0;
    REQUIRE(fire.minimize(potential,x,report)==2);
    REQUIRE(potential.samples.size()==2);
    REQUIRE(potential.samples[1][0]==Approx(.99));
    REQUIRE(potential.samples[1][1]==Approx(-.98));
    // First g=(1,-2), v=(-.1,.2). At (.99,-.98), g=(1,-1.95).
    // FIRE mixes the old v, of norm sqrt(5)/10, before the next EI kick.
    const double ratio=std::sqrt(20.0/1921.0);
    REQUIRE(x[0]==Approx(.9725-.025*ratio).margin(1e-13));
    REQUIRE(x[1]==Approx(-.9455+.04875*ratio).margin(1e-13));
    REQUIRE(report==2); // Increasing report values must not reject effective-force steps.
}

TEST_CASE("Shared FIRE converges on a coupled quadratic without an energy acceptance gate", "[simple_fire]") {
    CoupledQuadratic potential;
    Eigen::VectorXd x(2); x << 1,-1;
    SimpleFireF fire;
    fire.ts_init=.1;
    fire.ts_max=.2;
    fire.max_step=.2;
    fire.max_iter=2000;
    fire.gtol=1e-8;
    double report=0;
    REQUIRE(fire.minimize(potential,x,report)<fire.max_iter);
    Eigen::Vector2d residual;
    residual << 2*x[0]+x[1], x[0]+3*x[1];
    REQUIRE(residual.cwiseAbs().maxCoeff()<fire.gtol);
    REQUIRE(x.norm()<1e-8);
}

TEST_CASE("Shared FIRE small timestep restart clears velocity without raising the configured initial step", "[simple_fire]") {
    ScalarQuadratic potential{2e10};
    Eigen::VectorXd x=Eigen::VectorXd::Ones(1);
    SimpleFireF fire;
    fire.ts_init=1e-5;
    fire.ts_max=5e-4;
    fire.half_step_back=false;
    fire.max_iter=2;
    fire.gtol=0;
    double f=0;
    REQUIRE(fire.minimize(potential,x,f)==2);
    REQUIRE(potential.samples.size()==2); // No extra evaluation without coordinate rollback.
    REQUIRE(potential.samples[1]==Approx(-1));
    REQUIRE(fire.ts==Approx(1e-5));
    REQUIRE(fire.ts<=fire.ts_max);
    REQUIRE(x[0]==Approx(1)); // Zeroed velocity plus a fresh kick reverses the first move.
}

TEST_CASE("Shared FIRE first-step delay uses a norm cap no larger than max_step after settings", "[simple_fire]") {
    const double first=GENERATE(.05,1.0);
    const int delay=GENERATE(0,1);
    GPSettings settings;
    settings.Modify("first_step",std::to_string(first));
    settings.Modify("first_delay",std::to_string(delay));
    settings.Modify("max_move",".2");
    SimpleFireF fire(settings);
    fire.ts_init=1;
    fire.max_iter=2;
    fire.gtol=0;
    std::vector<Eigen::VectorXd> samples;
    auto force=[&](const Eigen::VectorXd &x,Eigen::VectorXd &g) {
        samples.push_back(x); g.resize(2); g<<3,4;
        return 3*x[0]+4*x[1];
    };
    Eigen::VectorXd x=Eigen::VectorXd::Zero(2);
    double f=0;
    REQUIRE(fire.minimize(force,x,f)==2);
    REQUIRE(samples.size()==2);
    const double first_cap=delay ? std::min(first,.2) : .2;
    REQUIRE(samples[1][0]==Approx(-.6*first_cap));
    REQUIRE(samples[1][1]==Approx(-.8*first_cap));
    REQUIRE(samples[1].norm()==Approx(first_cap));
    REQUIRE((x-samples[1]).norm()==Approx(.2));
    REQUIRE(x[0]==Approx(-.6*(first_cap+.2)));
    REQUIRE(x[1]==Approx(-.8*(first_cap+.2)));
}

TEST_CASE("Shared FIRE rollback ratio uses the previous actual capped displacement", "[simple_fire]") {
    const double ratio=GENERATE(0.0,.5,1.0);
    ScalarQuadratic potential{4};
    Eigen::VectorXd x=Eigen::VectorXd::Ones(1);
    SimpleFireF fire;
    fire.ts_init=1;
    fire.max_step=1.5;
    fire.max_iter=2;
    fire.gtol=0;
    fire.backward_ratio=ratio;
    double f=0;
    REQUIRE(fire.minimize(potential,x,f)==2);
    REQUIRE(potential.samples.size()==3);
    REQUIRE(potential.samples[1]==Approx(-.5)); // Raw first displacement -4 is capped to -1.5.
    REQUIRE(potential.samples[2]==Approx(-.5+1.5*ratio));
    REQUIRE(fire.ts==Approx(.5));
    REQUIRE(x[0]==Approx(0).margin(1e-12));
}

TEST_CASE("Shared FIRE keeps the raw gradient convergence test with nonuniform masses", "[simple_fire]") {
    const bool act_all=GENERATE(false,true);
    SimpleFireF fire;
    fire.assignmass=true;
    fire.massactall=act_all;
    fire.mass={40,20};
    fire.ts_init=.1;
    fire.max_iter=1;
    fire.gtol=.03;
    auto potential=[](const Eigen::VectorXd &x,Eigen::VectorXd &g) {
        g.resize(2); g<<.1*x[0],.2*x[1];
        return .05*x[0]*x[0]+.1*x[1]*x[1];
    };
    Eigen::VectorXd x=Eigen::VectorXd::Ones(2);
    double f=0;
    REQUIRE(fire.minimize(potential,x,f)==1); // Raw (.1,.2), not scaled (.0025,.01).
    REQUIRE(x[0]==Approx(1-.000025));
    REQUIRE(x[1]==Approx(1-.0001));
}
