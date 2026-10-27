#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BispectrumSO4.h"
#include "descriptors/so4_wigner_spline.h"
#include <iomanip>
#include <sstream>

namespace {
Eigen::VectorXd spline_global_values(const Descriptor_BispectrumSO4 &descriptor, const Atoms &atom) {
    Neighborhoods sites(atom, descriptor.cutoff_);
    Eigen::VectorXd value;
    descriptor.compute_global_density(sites, value);
    return value;
}
std::string precise(double value) {
    std::ostringstream text; text << std::setprecision(17) << value; return text.str();
}
}

TEST_CASE("SO4 spline fast global Jacobians cross blend boundaries and the cutoff",
          "[descriptor][so4][spline][global-density][derivative]") {
    constexpr int maximum = 4;
    constexpr double cutoff = 3.2;
    const auto table = SO4Density::make_wigner_spline(maximum);
    const double spacing = M_PI / table->cells();
    const int boundary = GENERATE(2, 4);
    const bool above = GENERATE(false, true), upper_pole = GENERATE(false, true);
    const double r = GENERATE(1.25, 3.1998, 3.2);
    const double distance = boundary * spacing * (above ? 1. + 1e-5 : 1. - 1e-5);
    const double beta = upper_pole ? M_PI - distance : distance;
    CAPTURE(boundary, above, upper_pole, r, spacing);

    GPSettings settings;
    settings.Modify("cutoff", precise(cutoff)); settings.Modify("twojmax", std::to_string(maximum));
    // theta=pi/2 on the selected edge makes either actual beta pole accessible.
    // Radius is held fixed in every displacement used by finite differences.
    settings.Modify("radius", precise(2. * r));
    settings.Modify("quaternion_convention", "q4");
    settings.Modify("contraction_backend", "snap"); settings.Modify("normalize_U", "true");
    settings.Modify("weights", "0.7 1.3"); settings.Modify("selfweights", "0.4 0.8");
    settings.Modify("average", "inner-density");
    Descriptor_BispectrumSO4 direct(settings);
    settings.Modify("wigner_backend", "spline"); Descriptor_BispectrumSO4 spline(settings);

    Atoms atom; atom.cell_ << 8., .4, -.2, .1, 7., .3, -.3, .2, 9.;
    atom.pbc_ = {false, false, false}; atom.chemmap_ = {0, 1, 0};
    atom.pos_ = {{0., 0., 0.},
        {r * std::cos(beta / 2.), .8 * r * std::sin(beta / 2.), .6 * r * std::sin(beta / 2.)},
        {-.4, .7, 1.1}};
    // Verify the sampled geometry really lies on the selected 2h/4h side,
    // rather than testing an assumed beta based only on the position angle.
    const auto &edge = atom.pos_[1];
    double ar, ai, br, bi;
    SO4Density::parameters(edge[0], edge[1], edge[2], edge.norm(), 2. * r, 4, false, ar, ai, br, bi);
    const double actual_beta = 2. * std::atan2(std::hypot(br, bi), std::hypot(ar, ai));
    REQUIRE(std::abs(actual_beta - beta) < 2e-13);

    Neighborhoods sites(atom, cutoff);
    Eigen::VectorXd value, reference_value, cell_value;
    Eigen::Tensor<double, 3, Eigen::RowMajor> cart, strain, reference_cart, reference_strain, cell;
    spline.compute_global_density(sites, value, &cart, &strain);
    direct.compute_global_density(sites, reference_value, &reference_cart, &reference_strain);
    spline.compute_global_density_cell(atom, cell_value, cell);
    REQUIRE((value - reference_value).norm() < 1e-9 * (1. + reference_value.norm()));
    REQUIRE((value - cell_value).norm() < 1e-12 * (1. + value.norm()));
    const Eigen::Map<const Eigen::VectorXd> jac(cart.data(), cart.size()), exact(reference_cart.data(), reference_cart.size());
    REQUIRE((jac - exact).norm() < 1e-8 * (1. + exact.norm()));

    for (double step : {1e-6, 5e-7}) {
        // The cosine cutoff is C1; a centered first difference crossing it has
        // O(step) truncation rather than the interior O(step^2) error.
        const double tolerance = r == cutoff ? 2e-6 : 3e-7;
        for (int i = 0; i < atom.size(); ++i) for (int axis = 0; axis < 3; ++axis) {
            auto plus = atom, minus = atom;
            plus.pos_[i][axis] += step; minus.pos_[i][axis] -= step;
            const Eigen::VectorXd fd = (spline_global_values(spline, plus) - spline_global_values(spline, minus)) / (2. * step);
            Eigen::VectorXd analytic(value.size());
            for (int f = 0; f < value.size(); ++f) analytic[f] = cart(f, i, axis);
            REQUIRE((analytic - fd).norm() < tolerance * (1. + fd.norm()));
        }
        for (int row = 0; row < 3; ++row) for (int column = 0; column < 3; ++column) {
            auto plus = atom, minus = atom;
            Eigen::Matrix3d fp = Eigen::Matrix3d::Identity(), fm = fp;
            fp(column, row) += step; fm(column, row) -= step;
            plus.cell_ = atom.cell_ * fp; minus.cell_ = atom.cell_ * fm;
            for (int i = 0; i < atom.size(); ++i) {
                plus.pos_[i] = atom.pos_[i] * fp; minus.pos_[i] = atom.pos_[i] * fm;
            }
            const Eigen::VectorXd fd = (spline_global_values(spline, plus) - spline_global_values(spline, minus)) / (2. * step);
            Eigen::VectorXd analytic(value.size());
            for (int f = 0; f < value.size(); ++f) analytic[f] = strain(f, row, column);
            REQUIRE((analytic - fd).norm() < tolerance * (1. + fd.norm()));

            plus = minus = atom;
            plus.cell_(row, column) += step; minus.cell_(row, column) -= step;
            for (int i = 0; i < atom.size(); ++i) {
                plus.pos_[i] = atom.pos_[i] * atom.cell_.inverse() * plus.cell_;
                minus.pos_[i] = atom.pos_[i] * atom.cell_.inverse() * minus.cell_;
            }
            const Eigen::VectorXd cell_fd = (spline_global_values(spline, plus) - spline_global_values(spline, minus)) / (2. * step);
            for (int f = 0; f < value.size(); ++f) analytic[f] = cell(f, row, column);
            REQUIRE((analytic - cell_fd).norm() < tolerance * (1. + cell_fd.norm()));
        }
    }
}
