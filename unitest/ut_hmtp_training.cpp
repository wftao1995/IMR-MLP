// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#include <catch2/catch_all.hpp>
#include "potentials/potential_hybmtp.h"
#include <sstream>

namespace {
void load_analytic(Potentials::PotMTP &pot, double cutoff) {
    std::ostringstream text;
    text << "MTP\nversion = 1.1.0\npotential_name = hmtp_training_test\n"
         << "species_count = 1\npotential_tag = test-only\nradial_basis_type = RBChebyshev\n"
         << "min_dist = 0\nmax_dist = " << cutoff << "\nradial_basis_size = 2\nradial_funcs_count = 1\n"
         << "radial_coeffs\n0-0\n{0.25, 0.125}\nalpha_moments_count = 1\n"
         << "alpha_index_basic_count = 1\nalpha_index_basic = {{0,0,0,0}}\n"
         << "alpha_index_times_count = 0\nalpha_index_times = {}\nalpha_scalar_moments = 1\n"
         << "alpha_moment_mapping = {0}\nspecies_coeffs = {0.125}\nmoment_coeffs = {0.75}\n";
    std::istringstream input(text.str());pot.Load(input);
}
Atoms structure() {
    Atoms at;at.cell_ = Eigen::Matrix3d::Identity()*20;
    at.pos_ = {{7,7,7},{9.2,7.1,7.2},{7.3,10.4,7.4}};
    at.chemmap_ = {0,0,0};return at;
}
}

TEST_CASE("HMTP exposes combined parameters and cutoff to the general trainer", "[hmtp][training]") {
    Potentials::PotMTP a,b,c;load_analytic(a,3);load_analytic(b,5);load_analytic(c,6);
    const bool three = GENERATE(false,true);
    Potentials::PotMTPHyb3 hmtp(&a,&b,three?&c:nullptr);
    Potentials::AnyPotential &potential=hmtp;
    REQUIRE(potential.species_count==1);REQUIRE(potential.cutoff==Catch::Approx(three?6:5));
    REQUIRE(potential.get_NL_coeffs_count()==a.get_NL_coeffs_count()+b.get_NL_coeffs_count()+(three?c.get_NL_coeffs_count():0));
    std::vector<double> coefficients(potential.get_NL_coeffs_count());potential.nl_get_coeffs(coefficients.data());
    coefficients.back()+=0.1;potential.nl_set_coeffs(coefficients.data());
    std::vector<double> restored(coefficients.size());potential.nl_get_coeffs(restored.data());REQUIRE(restored==coefficients);
    auto at=structure();potential.getEFS(at);const auto site=at.site_energies_;potential.getEFS(at);REQUIRE(at.site_energies_==site);
}

TEST_CASE("HMTP EFS and energy-only parameter gradients match finite differences", "[hmtp][training][gradient]") {
    Potentials::PotMTP a,b;load_analytic(a,3);load_analytic(b,5);
    Potentials::PotMTPHyb3 hmtp(&a,&b,nullptr);Potentials::AnyPotential &potential=hmtp;
    potential.train_init();auto at=structure();Neighborhoods nbhs(at,potential.cutoff);
    const bool energy_only=GENERATE(false,true);const double energy_weight=0.37;
    Eigen::Matrix<double,-1,3> force_weight(3,3);force_weight<<0.2,-0.1,0.3,0.4,0.2,-0.3,-0.2,0.1,0.5;
    Eigen::Matrix3d stress_weight=Eigen::Matrix3d::Constant(0.02);
    Eigen::Matrix<double,-1,3,Eigen::RowMajor> buffer;
    std::vector<double> coefficients(potential.get_NL_coeffs_count()),gradient(coefficients.size(),0);
    potential.nl_get_coeffs(coefficients.data());
    if(energy_only)potential.AccumulateEFSCombinationGard(at,nbhs,energy_weight,gradient.data());
    else potential.AccumulateEFSCombinationGard(at,nbhs,energy_weight,force_weight,stress_weight,buffer,gradient.data());
    auto loss=[&](){potential.getEFS(at,nbhs);double value=energy_weight*at.energy_;if(!energy_only){for(int i=0;i<at.size();++i)value+=at.forces_[i].dot(force_weight.row(i));value+=(at.stress_.array()*stress_weight.array()).sum();}return value;};
    constexpr double delta=1e-6;
    for(size_t i=0;i<coefficients.size();++i){auto perturbed=coefficients;perturbed[i]+=delta;potential.nl_set_coeffs(perturbed.data());const auto plus=loss();perturbed[i]-=2*delta;potential.nl_set_coeffs(perturbed.data());const auto minus=loss();INFO("parameter "<<i);REQUIRE(gradient[i]==Catch::Approx((plus-minus)/(2*delta)).margin(1e-7));}
    potential.nl_set_coeffs(coefficients.data());
}
