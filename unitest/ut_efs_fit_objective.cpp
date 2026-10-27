// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/train_nonlinear_potential.h"
#include "potentials/potential_mtp_LES.h"

using namespace Potentials;
using Catch::Approx;

TEST_CASE("EFS mixed missing labels preserve loss and gradient", "[training][missing-labels]") {
    PotMTP p("tests/case-mtp-les/les.mtp");
    Atoms a;
    a.cell_=Eigen::Matrix3d::Identity()*10.;
    a.pos_={{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
    a.chemmap_={0,1,0,1};
    p.getEFS(a);
    a.energy_+=.13;
    a.stress_(0,0)+=.17;
    for(auto& f:a.forces_) f+=Eigen::RowVector3d(.1,-.2,.3);
    std::vector<Atoms> mixed;
    double expected=0., single_sum=0.;
    Eigen::VectorXd gradient_sum=Eigen::VectorXd::Zero(p.get_NL_coeffs_count());
    GPSettings settings;
    settings.Modify("penalty_weight","0");
    settings.Modify("weight_L1","0");
    settings.Modify("weight_L2","0");
    EFSFitWeights weights;
    // Start with all labels, then remove them in every possible combination.
    for(int mask=7;mask>=0;--mask) {
        CAPTURE(mask);
        Atoms ref=a;
        ref.have_energy=mask&1;
        ref.have_stress=mask&4;
        if(!(mask&2)) ref.forces_.clear();
        mixed.push_back(ref);
        double target=0.;
        if(mask&1) target+=weights.energy_weight(ref)*.13*.13;
        if(mask&2) target+=weights.force_weight(ref)*ref.size()*.14;
        if(mask&4) target+=weights.stress_weight(ref)*.17*.17;
        expected+=target;
        std::vector<Atoms> data={ref};
        std::vector<Neighborhoods> nbhs;
        nbhs.emplace_back(data[0],p.cutoff);
        TrainNonlinear trainer(settings); trainer.init(data,nbhs,&p);
        double f=0.,fg=0.;
        Eigen::VectorXd g(p.get_NL_coeffs_count());
        trainer.AccumulateLoss(f); trainer.AccumulateGard(fg,g);
        REQUIRE(f==Approx(target).margin(1e-10));
        REQUIRE(fg==Approx(f).margin(1e-12));
        REQUIRE(g.allFinite());
        if(mask==0) REQUIRE(g.norm()==0.);
        single_sum+=fg;
        gradient_sum+=g;
    }
    std::vector<Neighborhoods> nbhs;
    for(auto& ref:mixed) nbhs.emplace_back(ref,p.cutoff);
    TrainNonlinear trainer(settings); trainer.init(mixed,nbhs,&p);
    double f=0.,fg=0.;
    Eigen::VectorXd g(p.get_NL_coeffs_count());
    trainer.AccumulateLoss(f); trainer.AccumulateGard(fg,g);
    REQUIRE(f==Approx(expected/mixed.size()).margin(1e-10));
    REQUIRE(fg==Approx(single_sum/mixed.size()).margin(1e-12));
    REQUIRE((g-gradient_sum/mixed.size()).norm()<1e-10);
    // Check the actual derivative and a descent step on the mixed dataset.
    Eigen::VectorXd x(g.size()); p.nl_get_coeffs(x.data());
    REQUIRE(g.norm()>0.);
    const Eigen::VectorXd direction=g.normalized();
    const double step=1e-6;
    Eigen::VectorXd trial=x+step*direction;
    trainer.SetOptimizerCoeffs(trial.data());
    double plus=0.,minus=0.; trainer.AccumulateLoss(plus);
    trial=x-step*direction;
    trainer.SetOptimizerCoeffs(trial.data()); trainer.AccumulateLoss(minus);
    trainer.SetOptimizerCoeffs(x.data());
    REQUIRE((plus-minus)/(2*step)==Approx(g.dot(direction)).epsilon(1e-5).margin(1e-8));
    REQUIRE(minus<f);
}

TEST_CASE("MTP penalty dispatch matches explicit implementation", "[training][penalty]") {
    PotMTP p("tests/case-mtp-les/les.mtp");
    p.train_init();
    p.reg_L2_vector[p.species_count]=.3;
    double direct=0.,dispatched=0.;
    Eigen::VectorXd a=Eigen::VectorXd::Zero(p.get_NL_coeffs_count()),b=a;
    p.AddPenaltyGard(.02,direct,a.data(),false,false,false);
    AnyPotential* base=&p;
    base->AddPenaltyGard(.02,dispatched,b.data());
    REQUIRE(direct>0.);
    REQUIRE(dispatched==Approx(direct));
    REQUIRE((a-b).norm()==0.);
}

TEST_CASE("EFS scalar loss agrees with gradient objective for atomic weights", "[training][weights]") {
    PotMTP p("tests/case-mtp-les/les.mtp");
    Atoms a;
    a.cell_=Eigen::Matrix3d::Identity()*10.;
    a.pos_={{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
    a.chemmap_={0,1,0,1}; a.fitness=.4; a.id=0;
    p.getEFS(a);
    a.energy_+=.13;
    a.stress_(0,1)+=.17; a.stress_(1,0)-=.23;
    a.atomic_fitness={0.,.3,2.,4.};
    for(int i=0;i<a.size();++i) a.forces_[i]+=Eigen::RowVector3d(.1,-.2,.3);
    std::vector<Atoms> data={a};
    std::vector<Neighborhoods> nbhs; nbhs.emplace_back(data[0],p.cutoff);
    GPSettings settings;
    settings.Modify("force_wgt_f","2");
    settings.Modify("penalty_weight","0.02");
    TrainNonlinear trainer(settings); trainer.init(data,nbhs,&p);
    double f=0.,gvalue=0.; Eigen::VectorXd g(p.get_NL_coeffs_count());
    trainer.AccumulateLoss(f); trainer.AccumulateGard(gvalue,g);
    REQUIRE(f==Approx(gvalue).margin(1e-12));
    EFSFitWeights weights; weights.norm_f=2.;
    double expected=weights.energy_weight(a)*.13*.13
        +weights.stress_weight(a)*(.17*.17+.23*.23);
    for(int i=0;i<a.size();++i)
        expected+=EFSFitWeights::atomic_weight(a,i,weights.force_weight(a))*.14;
    expected*=weights.normalization(a);
    REQUIRE(f-trainer.get_ploss()==Approx(expected).margin(1e-10));
}
