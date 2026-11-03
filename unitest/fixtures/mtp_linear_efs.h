// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "mtp_linear_energy.h"

namespace MTPLinearEFSFixture {
inline constexpr const char *version = "mtp-linear-efs-v1";
inline constexpr double force_atol = 5e-9, virial_atol = 2e-8, rtol = 5e-10;
inline bool close(double actual, double expected, bool virial = false) {
    return std::isfinite(actual) && std::isfinite(expected) &&
        std::abs(actual-expected) <= (virial ? virial_atol : force_atol) + rtol*std::abs(expected);
}
struct Reference {
    std::vector<double> energy;
    Eigen::Tensor<double, 3, Eigen::RowMajor> force, virial;
};
inline Reference reference(Potentials::PotMTP &model, Atoms &atoms, AnyNeighborhoods &neighbors) {
    Reference result;
    model.CalcEFSCmpt_R(atoms, neighbors, result.energy, result.force, result.virial);
    return result;
}
inline void load(Potentials::PotMTP &model, int variant) {
    if (variant == 4) {
        // Larger synthetic head for workspace splitting and allocation retry.
        // It has 29 scalar columns, three radial channels and two species.
        std::ostringstream out;
        out << "MTP\nversion = 1.1.0\npotential_name = synthetic-large\nscaling = 1\nspecies_count = 2\npotential_tag =\n"
            << "radial_basis_type = RBChebyshev\nmin_dist = .5\nmax_dist = 5\nradial_basis_size = 8\nradial_funcs_count = 3\nradial_coeffs\n";
        for (int a=0; a<2; ++a) for (int b=0; b<2; ++b) {
            out << a << '-' << b << '\n';
            for (int mu=0; mu<3; ++mu) {
                out << '{';
                for (int n=0; n<8; ++n) {
                    if (n) out << ',';
                    out << (n==mu ? .07 : .015*std::sin((mu+1.)*(n+1.)));
                }
                out << "}\n";
            }
        }
        out << "alpha_moments_count = 87\nalpha_index_basic_count = 12\nalpha_index_basic = {";
        for (int mu=0; mu<3; ++mu) for (int j=0; j<4; ++j) {
            if (mu || j) out << ',';
            out << '{' << mu << ',' << (j==1) << ',' << (j==2) << ',' << (j==3) << '}';
        }
        out << "}\nalpha_index_times_count = 225\nalpha_index_times = {";
        for (int k=12; k<87; ++k) for (int j=1; j<=3; ++j) {
            if (k!=12 || j!=1) out << ',';
            const int axis=4*(k%3)+j;
            out << '{' << axis << ',' << axis << ",1," << k << '}';
        }
        out << "}\nalpha_scalar_moments = 29\nalpha_moment_mapping = {0";
        for (int k=59; k<87; ++k) out << ',' << k;
        out << "}\nspecies_coeffs = {.3,-.1}\nmoment_coeffs = {";
        for (int k=0; k<29; ++k) { if (k) out << ','; out << .01*std::sin(k+.3); }
        out << "}\n";
        std::istringstream input(out.str()); model.Load(input); return;
    }
    MTPLinearEnergyFixture::load(model, variant == 0 || variant == 3 ? 1 : 2, variant == 2);
    if (variant == 3) {
        // Retain the same moment graph but exercise configurable radial derivatives.
        std::ostringstream saved; model.Save(saved);
        auto text = saved.str();
        const std::string original = "radial_basis_type = RBChebyshev";
        const auto position = text.find(original);
        if (position == std::string::npos) throw std::runtime_error("missing fixture radial declaration");
        text.replace(position, original.size(), "radial_basis_type = RBChebyshevConfigurable");
        const auto radial = text.find("radial_funcs_count");
        text.insert(radial, "cutoff_type = cosine_left\nshift_type = exponential\nshift_lambda = 0.7\n");
        std::istringstream input(text); model.Load(input);
    }
}
inline std::vector<Atoms> frames(int species, bool orthogonal = false) {
    std::vector<Atoms> result;
    result.push_back(MTPLinearEnergyFixture::frame(0, species, 2));
    result.push_back(MTPLinearEnergyFixture::frame(1, species, 3));
    Atoms isolated;
    isolated.cell_ = 20.*Eigen::Matrix3d::Identity();
    isolated.pos_ = {{1., 1., 1.}}; isolated.chemmap_ = {species-1};
    result.push_back(isolated);
    result.push_back(MTPLinearEnergyFixture::frame(2, species, 1)); // Periodic self images.
    if (orthogonal) for (auto &atoms : result) atoms.cell_ = atoms.cell_.diagonal().asDiagonal().toDenseMatrix().eval();
    return result;
}
} // namespace MTPLinearEFSFixture
