#include <catch2/catch_all.hpp>
#include "deplibs/LBFGSpp/include/LBFGS.h"
#include <limits>
namespace {
using Vector=Eigen::VectorXd;
struct Gradient {
 int calls=0;
 double operator()(const Vector &x,Vector &g){++calls;g=2*(x.array()-2).matrix();return -999.-x.sum();}
};
}
TEST_CASE("Force-only line search returns one coherent evaluated point at a trial limit", "[lbfgs][force-only]") {
 Gradient function;Vector origin=Vector::Zero(1),direction=Vector::Constant(1,4),gradient=Vector::Constant(1,-4),point;
 double step=.1,value=123,dot=-16;LBFGSpp::LBFGSParam<double> p;p.linesearch=4;p.max_linesearch=1;p.s_curv_coeff=.1;p.damping=0;p.max_step_move=10;
 LBFGSpp::LineSearchForceOnlyA<double>::LineSearch(function,p,origin,direction,10.,step,value,gradient,dot,point);
 REQUIRE(function.calls==1);REQUIRE(step==.1);REQUIRE(point[0]==Catch::Approx(.4));
 REQUIRE(point.isApprox(origin+step*direction));REQUIRE(dot==Catch::Approx(gradient.dot(direction)));REQUIRE(value==-999.-point.sum());
}
TEST_CASE("Force-only bracketing and damping preserve bounds and directional derivatives", "[lbfgs][force-only]") {
 const double damping=GENERATE(0.,.5,1.);Gradient function;
 Vector origin=Vector::Zero(1),direction=Vector::Constant(1,4),gradient=Vector::Constant(1,-4),point;
 double step=2,value=123,dot=-16;LBFGSpp::LBFGSParam<double> p;p.linesearch=4;p.s_curv_coeff=.01;p.damping=damping;p.max_step_move=3;
 LBFGSpp::LineSearchForceOnlyA<double>::LineSearch(function,p,origin,direction,10.,step,value,gradient,dot,point);
 REQUIRE((point-origin).lpNorm<Eigen::Infinity>()<=3);REQUIRE(point.isApprox(origin+step*direction));
 REQUIRE(dot==Catch::Approx(gradient.dot(direction)));REQUIRE(value==-999.-point.sum());
 REQUIRE(std::abs(point[0]-2)<(damping==.5?1.1:.03));
}
TEST_CASE("Force-only solver converges from gradients even as reported energy increases", "[lbfgs][force-only]") {
 struct OpposedEnergy {double operator()(const Vector &x,Vector &g){g=x;return -x.squaredNorm();}} function;
 LBFGSpp::LBFGSParam<double> p;p.linesearch=4;p.max_iterations=30;p.max_step_move=.2;p.past=0;p.epsilon=p.epsilon_rel=0;p.epsgmax=1e-10;
 Vector x=Vector::Ones(3);double value;LBFGSpp::LBFGSSolver<double,LBFGSpp::LineSearchForceOnlyA> solver(p);
 REQUIRE(solver.minimize(function,x,value)>0);REQUIRE(x.norm()<1e-10);REQUIRE(value==Catch::Approx(0).margin(1e-20));
}
TEST_CASE("Force-only damping cannot enlarge a capped step", "[lbfgs][force-only]") {
 const double damping=GENERATE(-.1,1.1,std::numeric_limits<double>::quiet_NaN());Gradient function;
 Vector origin=Vector::Zero(1),direction=Vector::Ones(1),gradient=-Vector::Ones(1),point;
 double step=.1,value=123,dot=-1;LBFGSpp::LBFGSParam<double> p;p.linesearch=4;p.damping=damping;
 REQUIRE_THROWS_AS(LBFGSpp::LineSearchForceOnlyA<double>::LineSearch(function,p,origin,direction,10.,step,value,gradient,dot,point),std::invalid_argument);
 REQUIRE(function.calls==0);
}
