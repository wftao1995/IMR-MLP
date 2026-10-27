#include <catch2/catch_all.hpp>
#include "potentials/potential_sw.h"
#include "unitest/pair_cell_fixture.h"
#include <map>
#include <sstream>

using namespace Potentials;
namespace {
void load(SW& model, int terms = 0) {
    std::ostringstream text;
    const char* names[] = {"H", "He"};
    for (int i = 0; i < 2; ++i) {
        for (int j = i; j < 2; ++j)
            text << names[i] << ' ' << names[j] << " 2body " << (terms == 2 ? 0. : 1.1+.1*i+.2*j)
                 << " .8 2.3 .6 4.3 1.2 4.5 3.4\n";
        for (int j = 0; j < 2; ++j)
            for (int k = j; k < 2; ++k)
                text << names[i] << ' ' << names[j] << ' ' << names[k] << ' '
                     << (terms == 1 ? 0. : .7+.2*i+.1*j+.3*k)
                     << " .75 4.8 1.3 .9 -.2 3.25\n";
    }
    std::istringstream input(text.str());
    model.Load(input);
}
Eigen::VectorXd force_fd(AnyPotential& pot, const Atoms& atom,
    const Eigen::VectorXd& direction, double step, double cutoff) {
    auto plus = atom, minus = atom;
    for (int i = 0; i < atom.size(); ++i) {
        plus.pos_[i] += step*direction.segment<3>(3*i).transpose();
        minus.pos_[i] -= step*direction.segment<3>(3*i).transpose();
    }
    Neighborhoods np(plus,cutoff), nm(minus,cutoff);
    pot.getEFS(plus,np);
    pot.getEFS(minus,nm);
    Eigen::VectorXd result(direction.size());
    for (int i = 0; i < atom.size(); ++i)
        result.segment<3>(3*i) = -(plus.forces_[i]-minus.forces_[i])/(2*step);
    return result;
}
void same_efs(const Atoms& actual, const Atoms& expected, double factor = 1.) {
    REQUIRE(std::abs(actual.energy_-factor*expected.energy_) < 1e-11);
    REQUIRE((actual.stress_-factor*expected.stress_).norm() < 1e-11);
    for (int i = 0; i < actual.size(); ++i)
        REQUIRE((actual.forces_[i]-factor*expected.forces_[i]).norm() < 1e-11);
}
void energy_oracle(AnyPotential& pot, const Atoms& atom, const Eigen::MatrixXd& result) {
    for (double step : {.001,.0005}) {
        const auto expected = PairCellTest::energy_fd(pot,atom,step,true);
        INFO("cell energy step=" << step);
        REQUIRE((result-expected).norm() < 100*step*step*std::max(1.,expected.norm())+2e-6);
    }
}
}

TEST_CASE("SW dense and native cell curvature differentiate pair and triplet energy",
          "[sw][hessian][cell]") {
    const int terms = GENERATE(0,1,2);
    const bool partial = GENERATE(false,true);
    INFO("terms=" << terms << " partial PBC=" << partial);
    SW model; load(model,terms); AnyPotential& pot = model;
    REQUIRE(pot.supports_analytical_hessian());
    REQUIRE(pot.supports_analytical_cell_hessian());
    auto atom = PairCellTest::geometry(partial), reference = atom;
    Neighborhoods nbhs(atom,pot.cutoff);
    // Both cells contain repeated periodic sources, including image triplets.
    bool repeated = false;
    for (int i = 0; i < atom.size(); ++i) {
        std::map<int,int> counts;
        for (int source : nbhs[i].indices_) repeated |= ++counts[source] > 1;
    }
    REQUIRE(repeated);
    pot.getEFS(reference,nbhs);
    HessianMatrix h;
    Eigen::MatrixXd cell,cross;
    pot.getEFSHessianCell(atom,nbhs,h,cell,cross);
    const auto dense = HessianUtils::to_dense(h);
    const auto native = PairCellTest::native_matrix(h,cell,cross);
    REQUIRE((dense-dense.transpose()).norm() < 1e-11);
    REQUIRE((cell-cell.transpose()).norm() < 1e-11);
    REQUIRE(cell.norm() > 1e-5);
    REQUIRE(cross.norm() > 1e-5);
    same_efs(atom,reference);
    for (int probe = 1; probe <= 2; ++probe) {
        Eigen::VectorXd direction(3*atom.size()), product;
        for (int i = 0; i < direction.size(); ++i) direction[i] = std::sin(.71*(i+1)*probe);
        pot.applyHessianVectorProduct(atom,nbhs,direction,product);
        REQUIRE((dense*direction-product).norm() < 1e-11*std::max(1.,product.norm()));
        for (double step : {1e-5,5e-6}) {
            const auto expected = force_fd(pot,atom,direction,step,pot.cutoff);
            REQUIRE((dense*direction-expected).norm() < 2e-7*std::max(1.,expected.norm()));
        }
    }
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(3*atom.size());
        for (int i = 0; i < atom.size(); ++i) translation[3*i+axis] = 1.;
        REQUIRE((dense*translation).norm() < 1e-11);
    }
    energy_oracle(pot,atom,native);
    // Every public route dispatches virtually to the same analytical kernel.
    for (int route = 0; route < 7; ++route) {
        auto work = atom;
        HessianMatrix other;
        if (route == 0) pot.getEFSHessian(work,other);
        if (route == 1) pot.getEFSHessian(work,nbhs,other);
        if (route == 2) pot.getEFSHessianOMP(work,nbhs,other);
        if (route == 3) pot.getEFSHessianSymm(work,other);
        if (route == 4) pot.getEFSHessianSymmOMP(work,other);
        if (route == 5) pot.getEFSHessianCell(work,other,cell,cross);
        if (route == 6) pot.getEFSHessianCellSymmOMP(work,other,cell,cross);
        REQUIRE((HessianUtils::to_dense(other)-dense).norm() < 1e-11);
        same_efs(work,reference);
    }
    pot.add_efs = true;
    pot.getEFSHessianCellSymm(atom,h,cell,cross);
    REQUIRE((PairCellTest::native_matrix(h,cell,cross)-native).norm() < 1e-11);
    same_efs(atom,reference,2.);
}

TEST_CASE("SW periodic self images retain cell curvature and cancel Cartesian residuals",
          "[sw][hessian][cell][self-image]") {
    SW model; load(model); AnyPotential& pot = model;
    Atoms atom;
    atom.cell_ = 2.5*Eigen::Matrix3d::Identity();
    atom.pos_ = {{.2,.3,.4}};
    atom.chemmap_ = {0};
    HessianMatrix h;
    Eigen::MatrixXd cell,cross;
    Neighborhoods nbhs(atom,pot.cutoff);
    REQUIRE(nbhs[0].nbhcount_ > 1);
    pot.getEFSHessianCellOMP(atom,nbhs,h,cell,cross);
    REQUIRE(h(0,0).isZero(1e-12));
    REQUIRE(cross.isZero(1e-12));
    REQUIRE(cell.norm() > 1e-5);
    energy_oracle(pot,atom,PairCellTest::native_matrix(h,cell,cross));
    atom.cell_.setZero();
    REQUIRE_THROWS(pot.getEFSHessianCellOMP(atom,nbhs,h,cell,cross));
}

TEST_CASE("SW cell consumer maps strain fixed positions and effective cutoff",
          "[sw][hessian][cell][consumer]") {
    const bool strain = GENERATE(false,true);
    const double cutoff = GENERATE(0.,2.9);
    SW model; load(model); AnyPotential& pot = model;
    auto atom = PairCellTest::geometry(true);
    ComputeHessian consumer;
    consumer.method = ComputeHessian::HessianMethod::Analytic;
    consumer.use_symmetry = false;
    consumer.request_cell = true;
    consumer.cell_strain = strain;
    consumer.engineering_shear = true;
    consumer.set_cutoff(cutoff);
    Eigen::MatrixXd result;
    consumer.compute(atom,result,&pot);
    for (double step : {.001,.0005}) {
        const auto expected = PairCellTest::energy_fd(pot,atom,step,false,strain,cutoff);
        REQUIRE((result-expected).norm() < 150*step*step*std::max(1.,expected.norm())+2e-6);
    }
}
