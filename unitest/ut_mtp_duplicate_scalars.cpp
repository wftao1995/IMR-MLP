// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "fixtures/mtp_linear_efs.h"
#include "train/train_mtp.h"

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); mpi.InitComm(MPI_COMM_SELF);
    const int result = Catch::Session().run(argc, argv);
    MPI_Finalize(); return result;
}
#endif

namespace {
Atoms geometry() {
    Atoms a; a.cell_ = 12.*Eigen::Matrix3d::Identity();
    a.pos_ = {{2.,2.,2.},{4.1,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
    a.chemmap_ = {0,1,0,1}; return a;
}
}

TEST_CASE("Repeated MTP scalar mappings sum all energy force and virial columns",
          "[mtp][derivatives][duplicate-scalars]") {
    Potentials::PotMTP model; MTPLinearEFSFixture::load(model,2);
    model.train_init(); model.set_inited();
    auto a = geometry(); Neighborhoods neighbors(a,model.cutoff);
    model.getEFS(a,neighbors);
    const auto basis = MTPLinearEFSFixture::reference(model,a,neighbors);
    double energy = 0.; Eigen::Matrix3d virial = Eigen::Matrix3d::Zero();
    for (int j=0; j<model.lin_coeff_size; ++j) {
        energy += model.linear_coeffs[j]*basis.energy[j];
        for (int d=0; d<3; ++d) for (int e=0; e<3; ++e)
            virial(d,e) += model.linear_coeffs[j]*basis.virial(d,e,j);
    }
    REQUIRE(a.energy_ == Catch::Approx(energy).epsilon(1e-11));
    REQUIRE((a.stress_-virial).norm() < 1e-10*std::max(1.,virial.norm()));
    for (int i=0; i<a.size(); ++i) for (int d=0; d<3; ++d) {
        double force = 0.;
        for (int j=0; j<model.lin_coeff_size; ++j)
            force += model.linear_coeffs[j]*basis.force(i,d,j);
        REQUIRE(a.forces_[i][d] == Catch::Approx(force).margin(1e-10));
        auto plus=a, minus=a; constexpr double step=1e-6;
        plus.pos_[i][d] += step; minus.pos_[i][d] -= step;
        model.getEFS(plus); model.getEFS(minus);
        REQUIRE(force == Catch::Approx(-(plus.energy_-minus.energy_)/(2*step))
                         .margin(2e-6*std::max(1.,std::abs(force))));
    }
}

TEST_CASE("Repeated MTP scalar mappings preserve full and frozen parameter pullbacks",
          "[mtp][training][duplicate-scalars]") {
    const int frozen = GENERATE(0,1,2);
    Potentials::PotMTP model; MTPLinearEFSFixture::load(model,2);
    model.train_init(); model.set_inited();
    auto target=geometry(); model.getEFS(target);
    target.energy_ += .03; target.forces_[1][0] += .011;
    target.stress_(0,1) += .017;
    std::vector<Atoms> data{target};
    std::vector<Neighborhoods> neighbors{Neighborhoods(target,model.cutoff)};
    GPSettings settings;
    settings.Modify("penalty_weight","0"); settings.Modify("weight_L2",".02");
    settings.Modify("force_weight",".17"); settings.Modify("stress_weight",".03");
    settings.Modify("fix_radial_coeffs",frozen==1?"true":"false");
    settings.Modify("fix_linear_coeffs",frozen==2?"true":"false");
    Trainer_MTP trainer(settings); trainer.init(data,neighbors,&model);
    const int n=trainer.get_relaxed_coeffs_count();
    alglib::real_1d_array x,g; x.setlength(n); g.setlength(n);
    model.ae_get_coeffs(x,trainer.fix_radial_coeffs,trainer.fix_linear_coeffs);
    double loss=0.; trainer.AccumulateGard(loss,g);
    Eigen::VectorXd direction=Eigen::VectorXd::NullaryExpr(n,
        [](Eigen::Index i) {return std::sin(.71*(i+1));});
    direction.normalize(); double dot=0.;
    for (int i=0; i<n; ++i) dot += g[i]*direction[i];
    constexpr double step=1e-6;
    for (int i=0; i<n; ++i) x[i] += step*direction[i];
    model.ae_set_coeffs(x,trainer.fix_radial_coeffs,trainer.fix_linear_coeffs);
    double plus=0.; trainer.AccumulateLoss(plus);
    for (int i=0; i<n; ++i) x[i] -= 2*step*direction[i];
    model.ae_set_coeffs(x,trainer.fix_radial_coeffs,trainer.fix_linear_coeffs);
    double minus=0.; trainer.AccumulateLoss(minus);
    REQUIRE(dot == Catch::Approx((plus-minus)/(2*step)).margin(3e-5*std::max(1.,std::abs(dot))));
}
