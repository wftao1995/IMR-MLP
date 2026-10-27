#include <catch2/catch_all.hpp>
#include "mindriver/constraint_expression.h"
#include "mindriver/relax_constraints_io.h"
#include "mindriver/relax_new.h"
#include <numbers>
using Catch::Approx;

namespace {
Atoms expression_atoms() {
    Atoms a; a.cell_ << 5,.3,.2, .4,4.8,.1, .1,.2,6;
    a.pos_={{.2,.3,.1},{1.1,.4,.3},{1.3,1.4,.5},{2.1,1.5,1.6}};
    a.chemmap_={0,0,0,0}; a.cartesian=true; return a;
}
struct ExpressionQuadratic : Potentials::AnyPotential {
    void getEFS(Atoms &a,AnyNeighborhoods &) override {getEFSOMP(a);}
    void getEFSOMP(Atoms &a) override {
        a.energy_=0; a.stress_.setZero();a.forces_.resize(a.size());
        for(int i=0;i<a.size();++i) {
            auto delta=(a.pos_[i]-Eigen::RowVector3d(2,2,0)).eval();
            a.energy_+=.5*delta.squaredNorm();a.forces_[i]=-delta;
        }
    }
};
}

TEST_CASE("Physical constraint expression uses analytic chain derivatives", "[constraint_expression][fd]") {
    const std::string source=GENERATE("distance(0,1)","angle(0,1,2)","dihedral(0,1,2,3)",
        "volume()/lattice_length(0)^2+lattice_ratio(1,2)+lattice_angle(0,1)",
        "sqrt(distance(0,1)^2)+exp(x(2))/3-log(distance(1,2))+sin(angle(0,1,2)*pi/180)+cos(y(3))",
        "-x(0)^2+2^3^2");
    auto a=expression_atoms();Constraints::Expression expr(source,a);const auto value=expr.evaluate(a);
    for(double h:{1e-5,2e-6}) {
        for(int i=0;i<a.size();++i) for(int c=0;c<3;++c) {
            auto plus=a,minus=a;plus.pos_[i][c]+=h;minus.pos_[i][c]-=h;
            CAPTURE(source,i,c,h);
            REQUIRE(value.positions[i][c]==Approx((expr.evaluate(plus).value-expr.evaluate(minus).value)/(2*h)).margin(3e-6).epsilon(2e-6));
        }
        for(int i=0;i<3;++i) for(int c=0;c<3;++c) {
            auto plus=a,minus=a;plus.cell_(i,c)+=h;minus.cell_(i,c)-=h;
            REQUIRE(value.cell(i,c)==Approx((expr.evaluate(plus).value-expr.evaluate(minus).value)/(2*h)).margin(3e-6).epsilon(2e-6));
        }
    }
}

TEST_CASE("Expression periodic images and dihedral branch remain bound", "[constraint_expression]") {
    auto a=expression_atoms(); a.pos_[0]+=a.cell_.row(0);a.pos_[3]-=a.cell_.row(1);
    Constraints::Expression expr("dihedral(0,1,2,3)",a);
    const auto initial=expr.evaluate(a);
    // A full common translation changes neither the physical torsion nor images.
    auto moved=a;for(auto &p:moved.pos_)p+=Eigen::RowVector3d(13,-4,2);
    REQUIRE(expr.evaluate(moved).value==Approx(initial.value).margin(1e-11));
    for(auto &p:a.pos_)p*=1.01;
    REQUIRE(std::abs(expr.evaluate(a).value-initial.value)<10);
    REQUIRE_THROWS_AS(Constraints::Expression("distance(0,9)",a),std::invalid_argument);
    REQUIRE_THROWS_AS(Constraints::Expression("__import__(0)",a),std::invalid_argument);
    REQUIRE_THROWS_AS(Constraints::Expression("x(0)^x(1)",a),std::invalid_argument);
    REQUIRE_THROWS_AS(Constraints::Expression("distance(0,0)",a),std::invalid_argument);
    REQUIRE_THROWS(Constraints::Expression("log(-1)",a));
}

TEST_CASE("V2 box plane and nonlinear bounds converge with KKT checks", "[sqp][constraint_v2]") {
    auto a=expression_atoms();a.pos_.resize(1);a.chemmap_.resize(1);a.pbc_={false,false,false};
    ExpressionQuadratic p;RelaxationDrvV2 d;d.control.driver="sqp";d.flags.cartesian=true;
    d.control.max_move=.2;d.tol.gtol=2e-6;
    std::istringstream input(R"({"version":2,"constraints":[
        {"kind":"position","atoms":[0],"component":"x","lower":0,"upper":0.8,"residual_tolerance":1e-7},
        {"kind":"wall","atoms":[0],"normal":[1,1,0],"upper":1,"residual_tolerance":1e-7},
        {"kind":"formula","expression":"z(0)^2","upper":0.01,"residual_tolerance":1e-7}
    ]})");
    auto regs=Constraints::bind_constraints(Constraints::parse_constraints(input),a);d.init(&p,a,&regs);
    REQUIRE(d.run()==0);REQUIRE(d.kkt_residual<=d.tol.gtol);
    REQUIRE(d.atom.pos_[0][0]==Approx(1/std::sqrt(2.)).margin(3e-6));
    REQUIRE(d.atom.pos_[0][1]==Approx(1/std::sqrt(2.)).margin(3e-6));
    REQUIRE(d.atom.pos_[0][2]==Approx(0).margin(3e-6));
}

TEST_CASE("V2 geometric intervals are physical and V1 remains versioned", "[constraint_v2]") {
    auto a=expression_atoms();
    std::istringstream input(R"({"constraints":[{"kind":"angle","atoms":[0,1,2],"lower":90,"upper":150,"residual_tolerance":0.001}],"version":2})");
    auto regs=Constraints::bind_constraints(Constraints::parse_constraints(input),a);
    REQUIRE(regs[0].lower==90);REQUIRE(regs[0].upper==150);
    REQUIRE(regs[0].g0==0);
    for(const auto source:{
        R"({"version":1,"constraints":[{"kind":"position","atoms":[0],"component":"x","target":0,"residual_tolerance":0.001}]})",
        R"({"version":2,"constraints":[{"kind":"distance","atoms":[0,1],"lower":2,"upper":1,"residual_tolerance":0.001}]})",
        R"({"version":2,"constraints":[{"kind":"angle","atoms":[0,1,2],"target":90,"upper":100,"residual_tolerance":0.001}]})"}) {
        std::istringstream invalid(source);REQUIRE_THROWS_AS(Constraints::parse_constraints(invalid),std::invalid_argument);
    }
}

TEST_CASE("V2 dihedral targets and intervals bind the initial continuous branch", "[constraint_v2][dihedral]") {
    auto a=expression_atoms();a.pbc_={false,false,false};
    const double angle=-170*std::numbers::pi/180;
    a.pos_={{1,0,0},{0,0,0},{0,0,1},{std::cos(angle),std::sin(angle),1}};
    for(const auto spec:{std::string("\"target\":170"),std::string("\"lower\":170,\"upper\":190")}) {
        std::istringstream text("{\"version\":2,\"constraints\":[{\"kind\":\"dihedral\",\"atoms\":[0,1,2,3],\"residual_tolerance\":1e-6,"+spec+"}]}");
        auto regs=Constraints::bind_constraints(Constraints::parse_constraints(text),a);
        if(regs[0].bounded) {REQUIRE(regs[0].lower==Approx(-190));REQUIRE(regs[0].upper==Approx(-170));}
        else REQUIRE(regs[0].g0==Approx(-190));
    }
}

TEST_CASE("Periodic-subspace formula does not silently constrain a fictitious 3D volume", "[constraint_v2][subspace]") {
    auto a=expression_atoms();a.cell_.row(2).setZero();a.pbc_={true,true,false};
    std::istringstream text(R"json({"version":2,"constraints":[{"kind":"formula","expression":"volume()","target":1,"residual_tolerance":1e-6}]})json");
    auto regs=Constraints::bind_constraints(Constraints::parse_constraints(text),a);
    ExpressionQuadratic p;RelaxationDrvV2 d;d.flags.periodic_subspace=true;d.flags.relax_cell=true;d.control.driver="sqp";
    REQUIRE_THROWS_WITH(d.init(&p,a,&regs),Catch::Matchers::ContainsSubstring("three-dimensional volume"));
}
