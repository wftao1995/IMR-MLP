// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "train/hvp_fit.h"
#include "train/train_mtp.h"
#include "train/train_nonlinear_potential.h"
#include "potentials/potential_mtp_LES.h"
#include <sstream>

using Catch::Approx;
namespace {
Atoms structure() {
    Atoms a;
    a.cell_ = 12. * Eigen::Matrix3d::Identity();
    a.pos_ = {{2.,2.,2.}, {4.1,2.2,2.}, {2.3,4.2,2.4}, {5.,4.8,3.1}};
    a.chemmap_ = {0,1,0,1}; a.id = 0;
    a.hvp_directions_.resize(12, 2);
    for (int k = 0; k < 2; ++k)
        for (int i = 0; i < 12; ++i) a.hvp_directions_(i,k) = std::sin(.3 * (i+1) * (k+1));
    a.hvp_targets_ = 1.7 * a.hvp_directions_;
    return a;
}
GPSettings settings() {
    GPSettings s;
    for (const auto *key : {"energy_weight", "force_weight", "stress_weight", "weight_L2",
                            "penalty_weight", "weight_rad_w0", "weight_rad_w1", "weight_rad_w2"})
        s.Modify(key, "0");
    s.Modify("hvp_weight", "0.7");
    s.Modify("hvp_fd_step", "0.0001");
    s.Modify("do_orthogonalization", "false");
    s.Modify("skip_preinit", "true");
    return s;
}
GPSettings generic_settings() {
    auto s = settings();
    // Radial regularization settings belong to the dedicated MTP trainer.
    for (const auto *key : {"weight_rad_w0", "weight_rad_w1", "weight_rad_w2"})
        s.settings.erase(key);
    return s;
}
// A stateful E = theta^2 |R|^2 / 2 model makes cache misuse observable.
class Quadratic : public Potentials::AnyPotential {
public:
    double theta = 1.1;
    std::vector<Eigen::RowVector3d> cached;
    Quadratic() { cutoff=4.; set_inited(); }
    int get_NL_coeffs_count() const override { return 1; }
    void nl_get_coeffs(double *p) const override { *p=theta; }
    void nl_set_coeffs(const double *p) override { theta=*p; }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        if (!add_efs) a.resetEFS();
        cached=a.pos_;
        for (int i=0;i<a.size();++i) {
            a.energy_ += .5 * theta * theta * a.pos_[i].squaredNorm();
            a.forces_[i] += -theta * theta * a.pos_[i].transpose();
        }
    }
    void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &, double,
        Eigen::Matrix<double,-1,3> &fw, Eigen::Matrix3d &,
        Eigen::Matrix<double,-1,3,Eigen::RowMajor> &, double *g) override {
        REQUIRE(cached == a.pos_);
        for (int i=0;i<a.size();++i) *g += -2 * theta * fw.row(i).dot(cached[i]);
    }
};
}

TEST_CASE("HVP labels round trip in extxyz, CFG and binary", "[hvp][io]") {
    const auto format = GENERATE(std::string("extxyz"),std::string("cfg"),std::string("bin"));
    auto a=structure(); GPIO io;
    std::stringstream stream;
    REQUIRE(io.write_one(a,stream,format));
    auto plain=a; plain.clear_hvp();
    REQUIRE(io.write_one(plain,stream,format));
    Atoms b;
    REQUIRE(io.read_one(b,stream,format));
    REQUIRE(b.hvp_directions_.isApprox(a.hvp_directions_,1e-15));
    REQUIRE(b.hvp_targets_.isApprox(a.hvp_targets_,1e-15));
    REQUIRE(io.read_one(b,stream,format));
    REQUIRE_FALSE(b.has_hvp());
    REQUIRE(b.hvp_targets_.size()==0);
}

TEST_CASE("HVP IO rejects malformed paired data", "[hvp][io]") {
    const std::string tags = GENERATE(
        "hvp_directions=\"1 2 3\"",
        "hvp_targets=\"1 2 3\"",
        "hvp_directions=\"1 2\" hvp_targets=\"1 2 3\"",
        "hvp_directions=\"1 2 3\" hvp_targets=\"1 2 3 4 5 6\"",
        "hvp_directions=\"0 0 0\" hvp_targets=\"1 2 3\"",
        "hvp_directions=\"nan 2 3\" hvp_targets=\"1 2 3\"",
        "hvp_directions=\"1 2 3\" hvp_targets=\"1 2 inf\"",
        "hvp_directions=\"1 2 3junk\" hvp_targets=\"1 2 3\"",
        "hvp_directions=\"1 2 3\" hvp_directions=\"4 5 6\" hvp_targets=\"1 2 3\"");
    std::stringstream stream("1\nLattice=\"8 0 0 0 8 0 0 0 8\" "+tags+" Properties=species:S:1:pos:R:3\nH 0 0 0\n");
    Atoms a; GPIO io;
    REQUIRE_THROWS(io.read_one(a,stream,"extxyz"));
}

TEST_CASE("HVP output validates shapes and refuses lossy formats", "[hvp][io]") {
    GPIO io; auto a=structure(); std::stringstream out;
    REQUIRE_THROWS(io.write_one(a,out,"poscar"));
    a.hvp_targets_.resize(12,1);
    REQUIRE_THROWS(io.write_one(a,out,"extxyz"));
    a=structure(); a.clear(); REQUIRE_FALSE(a.has_hvp());
    a=structure(); a.resize(2); REQUIRE_FALSE(a.has_hvp());
}

TEST_CASE("MTP basis HVP reconstructs dense basis Hessians and total product", "[hvp][mtp]") {
    Potentials::PotMTP pot("tests/case-mtp-les/les.mtp"); pot.train_init();
    auto a=structure(); Neighborhoods nbhs(a,pot.cutoff);
    Eigen::MatrixXd components;
    Eigen::VectorXd direction=a.hvp_directions_.col(0);
    pot.CalcHessianVectorCmpt(a,nbhs,direction,components);
    for (const auto &buffer:pot.omp_buffers_list) REQUIRE(buffer.buff_site_hessian.hessian.empty());
    Eigen::VectorXd actual;
    pot.applyHessianVectorProduct(a,nbhs,direction,actual);
    Eigen::Map<const Eigen::VectorXd> coeff(pot.linear_coeffs,pot.lin_coeff_size);
    REQUIRE((components*coeff).isApprox(actual,1e-9));
    const Eigen::VectorXd saved_coefficients = coeff;
    for (int j=0;j<pot.lin_coeff_size;++j) {
        Eigen::VectorXd unit = Eigen::VectorXd::Zero(pot.lin_coeff_size);
        unit[j] = 1.;
        Eigen::Map<Eigen::VectorXd>(pot.linear_coeffs, pot.lin_coeff_size) = unit;
        Potentials::HessianMatrix dense;
        pot.getEFSHessian(a, nbhs, dense);
        Eigen::VectorXd expected=Eigen::VectorXd::Zero(direction.size());
        for(int r=0;r<direction.size();++r) for(int c=0;c<direction.size();++c)
            expected[r]+=dense(r/3,r%3,c/3,c%3)*direction[c];
        REQUIRE(components.col(j).isApprox(expected,1e-9));
    }
    Eigen::Map<Eigen::VectorXd>(pot.linear_coeffs, pot.lin_coeff_size) = saved_coefficients;
}

TEST_CASE("HVP finite-difference loss has correct sign, weight and stateful pullback", "[hvp][training]") {
    Quadratic pot; pot.add_efs=GENERATE(false,true);
    auto a=structure(); a.fitness=1.3;
    const auto saved=a.pos_;
    Training::HVPFitSettings fit{.7,1.,1e-4};
    double grad=0.;
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> scratch;
    Eigen::Matrix3d stress=Eigen::Matrix3d::Zero();
    auto pullback=[&](Atoms &p,Neighborhoods &n,Eigen::Matrix<double,-1,3>& w) {
        pot.AccumulateEFSCombinationGard(p,n,0.,w,stress,scratch,&grad);
    };
    const auto result=Training::finite_difference_hvp_loss(pot,a,fit,.8,pullback);
    const double coeff=.7*1.3*.8/4/2;
    const double diff=pot.theta*pot.theta-1.7;
    REQUIRE(result.loss==Approx(coeff*diff*diff*a.hvp_directions_.squaredNorm()).epsilon(1e-9));
    REQUIRE(grad==Approx(coeff*4*pot.theta*diff*a.hvp_directions_.squaredNorm()).epsilon(1e-8));
    REQUIRE(result.observations==24);
    REQUIRE(a.pos_==saved);
    fit.weight=0.; REQUIRE(Training::finite_difference_hvp_loss(pot,a,fit).loss==0.);
    fit.step=0.; REQUIRE_THROWS(fit.validate());
}

TEST_CASE("Generic nonlinear trainer includes HVP-only data in loss and gradient", "[hvp][training]") {
    Quadratic pot; std::vector<Atoms> atoms{structure()};
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(atoms[0],pot.cutoff);
    TrainNonlinear trainer(generic_settings()); trainer.init(atoms,nbhs,&pot);
    Eigen::VectorXd g(1); double loss=0, value=0;
    trainer.AccumulateGard(loss,g); trainer.AccumulateLoss(value);
    REQUIRE(value==Approx(loss).epsilon(1e-10)); REQUIRE(loss>0.);
    const double diff=pot.theta*pot.theta-1.7;
    REQUIRE(g[0]==Approx(.7/2*4*pot.theta*diff*atoms[0].hvp_directions_.squaredNorm()).epsilon(1e-8));
    REQUIRE(trainer.emonitor.nhvp==24);
}

TEST_CASE("MTP HVP-only linear fit reduces projected Hessian error", "[hvp][training][mtp]") {
    const bool fast=GENERATE(false,true);
    Potentials::PotMTP pot("tests/case-mtp-les/les.mtp"); pot.train_init(); pot.set_inited();
    std::vector<Atoms> atoms; std::vector<Neighborhoods> nbhs;
    for(int k=0;k<5;++k) {
        auto a=structure(); a.pos_[1][0]+=.1*k; a.id=k;
        Neighborhoods n(a,pot.cutoff);
        for(int j=0;j<2;++j) {
            Eigen::VectorXd product; pot.applyHessianVectorProduct(a,n,a.hvp_directions_.col(j),product);
            a.hvp_targets_.col(j)=product;
        }
        atoms.push_back(a); nbhs.emplace_back(a,pot.cutoff);
    }
    for(int i=pot.species_count;i<pot.lin_coeff_size;++i) pot.linear_coeffs[i]+=.2*std::sin(i);
    auto s=settings(); s.Modify("lfit_fastalgo",fast?"true":"false");
    s.Modify("fix_species_coeffs", "true"); s.Modify("weight_L2","1e-12");
    Trainer_MTP trainer(s); trainer.init(atoms,nbhs,&pot);
    double before=0, after=0; trainer.AccumulateLoss(before);
    trainer.TrainLinear(); trainer.AccumulateLoss(after);
    REQUIRE(before>1e-8); REQUIRE(after<before*1e-6);
}

TEST_CASE("MTP nonlinear HVP gradient matches parameter differences", "[hvp][training][mtp]") {
    Potentials::PotMTP pot("tests/case-mtp-les/les.mtp"); pot.train_init(); pot.set_inited();
    auto a=structure(); std::vector<Atoms> atoms{a};
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(a,pot.cutoff);
    Trainer_MTP trainer(settings()); trainer.init(atoms,nbhs,&pot);
    alglib::real_1d_array g; g.setlength(pot.reg_coeff_size); double loss=0;
    trainer.AccumulateGard(loss,g);
    for(int i=0;i<pot.reg_coeff_size;++i) {
        CAPTURE(i); const double old=pot.regression_coeffs[i], h=1e-5;
        pot.regression_coeffs[i]=old+h; double plus; trainer.AccumulateLoss(plus);
        pot.regression_coeffs[i]=old-h; double minus; trainer.AccumulateLoss(minus);
        pot.regression_coeffs[i]=old;
        REQUIRE(g[i]==Approx((plus-minus)/(2*h)).margin(3e-5).epsilon(1e-4));
    }
}

TEST_CASE("HVP CFG rejects incomplete and invalid labels", "[hvp][io]") {
    const std::string features=GENERATE(
        "Feature hvp_directions 1 2 3\n",
        "Feature hvp_directions 1 2 3\nFeature hvp_targets 1 2 junk\n",
        "Feature hvp_directions 1 2 3\nFeature hvp_targets 1 2 3\nFeature hvp_targets 4 5 6\n");
    std::stringstream stream("BEGIN_CFG\nSize\n1\nSupercell\n8 0 0\n0 8 0\n0 0 8\nAtomData: id type cartes_x cartes_y cartes_z\n1 0 0 0 0\n"+features+"END_CFG\n");
    GPIO io; Atoms a; REQUIRE_THROWS(io.read_one(a,stream,"cfg"));
}

TEST_CASE("HVP extxyz supports either tag order and rejects unterminated quotes", "[hvp][io]") {
    GPIO io; Atoms a;
    std::stringstream valid("1\nhvp_targets=\"0 0 0\" hvp_directions=\"1 2 3\" Properties=species:S:1:pos:R:3\nH 0 0 0\n");
    REQUIRE(io.read_one(a,valid,"extxyz")); REQUIRE(a.has_hvp());
    REQUIRE(a.hvp_targets_.isZero());
    std::stringstream invalid("1\nProperties=species:S:1:pos:R:3 hvp_directions=\"1 2 3\nH 0 0 0\n");
    REQUIRE_THROWS(io.read_one(a,invalid,"extxyz"));
}

TEST_CASE("HVP loss averages probes and disabled training leaves no contribution", "[hvp][training]") {
    Quadratic pot; auto a=structure(); Training::HVPFitSettings fit{.7,0.,1e-4};
    const double expected=Training::finite_difference_hvp_loss(pot,a,fit).loss;
    a.hvp_directions_=a.hvp_directions_.replicate(1,3).eval();
    a.hvp_targets_=a.hvp_targets_.replicate(1,3).eval();
    REQUIRE(Training::finite_difference_hvp_loss(pot,a,fit).loss==Approx(expected).epsilon(1e-12));
    auto s=generic_settings(); s.Modify("hvp_weight","0");
    std::vector<Atoms> atoms{a}; std::vector<Neighborhoods> nbhs; nbhs.emplace_back(a,pot.cutoff);
    TrainNonlinear trainer(s); trainer.init(atoms,nbhs,&pot);
    double loss; Eigen::VectorXd gradient(1); trainer.AccumulateGard(loss,gradient);
    REQUIRE(loss==0.); REQUIRE(gradient[0]==0.);
    REQUIRE(trainer.emonitor.nhvp==0);
}

TEST_CASE("LES HVP supervision differentiates full long-range forces", "[hvp][training][les]") {
    Potentials::PotMTP_LES pot("tests/case-mtp-les/les.mtp");
    pot.les_species_coeffs[0]=.23; pot.les_species_coeffs[1]=-.17;
    std::fill_n(pot.les_moment_coeffs,pot.get_alpha_scalar_moments(),0.);
    pot.les_moment_coeffs[0]=.35; pot.les_moment_coeffs[1]=-.21;
    pot.train_init(); pot.set_inited();
    auto a=structure(); std::vector<Atoms> atoms{a};
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(a,pot.cutoff);
    TrainNonlinear trainer(generic_settings()); trainer.init(atoms,nbhs,&pot);
    Eigen::VectorXd x(pot.get_NL_coeffs_count()),g(x.size()); pot.nl_get_coeffs(x.data());
    double loss; trainer.AccumulateGard(loss,g);
    for(int i=0;i<x.size();++i) {
        CAPTURE(i); const double h=1e-5;
        auto plus=x,minus=x; plus[i]+=h; minus[i]-=h;
        pot.nl_set_coeffs(plus.data()); double fp; trainer.AccumulateLoss(fp);
        pot.nl_set_coeffs(minus.data()); double fm; trainer.AccumulateLoss(fm);
        REQUIRE(g[i]==Approx((fp-fm)/(2*h)).margin(5e-5).epsilon(3e-4));
    }
    pot.nl_set_coeffs(x.data());
}
