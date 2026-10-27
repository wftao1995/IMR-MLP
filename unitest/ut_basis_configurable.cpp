// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "basis/basis.h"
#include "base/neighborhood.h"
#include "potentials/potential_mtp.h"

#include <array>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using Catch::Approx;

namespace {

constexpr double rmin = 0.5;
constexpr double rmax = 5.0;
constexpr int basis_size = 6;

std::unique_ptr<RBasis_ChebyshevConfigurable> make_basis(
    Cutoff::CutoffType cutoff_type, XTransForm::ShiftType shift_type,
    double cutoff_b = -0.4, double shift_lambda = 1.2,
    double min_val = rmin, double max_val = rmax, int size = basis_size,
    double scaling = 1.0)
{
    return std::make_unique<RBasis_ChebyshevConfigurable>(
        min_val, max_val, size,
        Cutoff::make_cutoff(cutoff_type, min_val, max_val, cutoff_b),
        XTransForm::make_shift(shift_type, min_val, max_val, shift_lambda),
        scaling);
}

template <class Function>
void validate_scalar_function(const Function &function, double r)
{
    double val, der, derder;
    function.compute_derder(r, val, der, derder);

    double val1, der1;
    function.compute_der(r, val1, der1);
    REQUIRE(val1 == Approx(val).margin(1e-13));
    REQUIRE(der1 == Approx(der).margin(1e-13));

    constexpr double h1 = 1e-6;
    const double num_der =
        (function.compute(r + h1) - function.compute(r - h1)) / (2.0 * h1);
    REQUIRE(der == Approx(num_der).epsilon(1e-7).margin(1e-9));

    constexpr double h2 = 1e-5;
    auto first = [&](double x) {
        double v, d;
        function.compute_der(x, v, d);
        return d;
    };
    const double num_second = (first(r + h2) - first(r - h2)) / (2.0 * h2);
    REQUIRE(derder == Approx(num_second).epsilon(2e-6).margin(1e-8));
}

void compare_bases(AnyBasis &left, AnyBasis &right,
                   const std::vector<double> &points)
{
    REQUIRE(left.basis_size == right.basis_size);
    const int size = left.basis_size;
    for (double r : points) {
        std::vector<double> lv(size), ld(size), ldd(size);
        std::vector<double> rv(size), rd(size), rdd(size);
        left.calc_derder(r, lv, ld, ldd);
        right.calc_derder(r, rv, rd, rdd);
        for (int i = 0; i < size; ++i) {
            INFO("r=" << r << " i=" << i);
            REQUIRE(lv[i] == Approx(rv[i]).margin(2e-12));
            REQUIRE(ld[i] == Approx(rd[i]).margin(2e-12));
            REQUIRE(ldd[i] == Approx(rdd[i]).margin(2e-11));
        }
    }
}

std::string potential_text(const std::string &basis_definition,
                           bool split = false)
{
    std::ostringstream output;
    output << "MTP\n"
           << "version = 1.1.0\n"
           << "potential_name = MTP1m\n"
           << "scaling = 1\n"
           << "species_count = 1\n"
           << "potential_tag =\n";
    if (split) output << "radial_basis_split = 1\npair 0 0\n";
    output << basis_definition
           << "radial_funcs_count = 2\n"
           << "radial_coeffs\n"
           << "  0-0\n"
           << "    {1, 0, 0}\n"
           << "    {0, 1, 0}\n"
           << "alpha_moments_count = 4\n"
           << "alpha_index_basic_count = 2\n"
           << "alpha_index_basic = {{0, 0, 0, 0}, {1, 0, 0, 0}}\n"
           << "alpha_index_times_count = 2\n"
           << "alpha_index_times = {{0, 0, 1, 2}, {1, 1, 1, 3}}\n"
           << "alpha_scalar_moments = 2\n"
           << "alpha_moment_mapping = {2, 3}\n"
           << "species_coeffs = {0.25}\n"
           << "moment_coeffs = {0.7, -0.2}\n";
    return output.str();
}

const std::string configurable_quadratic =
    "radial_basis_type = RBChebyshevConfigurable\n"
    "  min_dist = 0.5\n"
    "  max_dist = 5\n"
    "  radial_basis_size = 3\n"
    "  cutoff_type = quadratic\n"
    "  shift_type = linear\n";

const std::string legacy_quadratic =
    "radial_basis_type = RBChebyshev\n"
    "  min_dist = 0.5\n"
    "  max_dist = 5\n"
    "  radial_basis_size = 3\n";

Atoms small_atoms()
{
    Atoms atoms;
    atoms.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {{1.0, 1.0, 1.0},
                  {2.8, 1.2, 1.1},
                  {1.7, 3.0, 1.4}};
    atoms.chemmap_ = {0, 0, 0};
    atoms.resetEFS();
    return atoms;
}

} // namespace

TEST_CASE("Configurable radial cutoff component derivatives",
          "[basis][configurable][cutoff]")
{
    Cutoff::QuadraticCutoff quadratic(rmin, rmax);
    Cutoff::CosineCutoff cosine(rmin, rmax);
    Cutoff::Poly5Cutoff poly5(rmin, rmax, -0.4);
    validate_scalar_function(quadratic, 2.1);
    validate_scalar_function(cosine, 2.1);
    validate_scalar_function(poly5, 2.1);
}

TEST_CASE("Configurable radial shift component derivatives and endpoints",
          "[basis][configurable][shift]")
{
    XTransForm::LinearShift linear(rmin, rmax);
    XTransForm::ExponentialShift exponential(rmin, rmax, 1.2);
    XTransForm::ExponentialShift exponential_zero(rmin, rmax, 0.0);
    validate_scalar_function(linear, 2.1);
    validate_scalar_function(exponential, 2.1);
    validate_scalar_function(exponential_zero, 2.1);
    for (const auto *shift : std::array<const XTransForm::AnyTrans *, 3>{
             &linear, &exponential, &exponential_zero}) {
        REQUIRE(shift->compute(rmin) == Approx(-1.0).margin(1e-14));
        REQUIRE(shift->compute(rmax) == Approx(1.0).margin(1e-14));
    }
    for (double r : {rmin, 1.1, 2.8, rmax})
        REQUIRE(exponential_zero.compute(r) ==
                Approx(linear.compute(r)).margin(1e-14));
}

TEST_CASE("All configurable radial basis combinations have consistent derivatives",
          "[basis][configurable][derivatives]")
{
    for (const auto cutoff_type : {Cutoff::CutoffType::Quadratic,
                                   Cutoff::CutoffType::Cosine,
                                   Cutoff::CutoffType::Poly5}) {
        for (const auto shift_type : {XTransForm::ShiftType::Linear,
                                      XTransForm::ShiftType::Exponential}) {
            auto basis = make_basis(cutoff_type, shift_type);
            for (double r : {1.0, 2.2, 4.0}) {
                std::vector<double> val, der, derder;
                basis->calc_derder(r, val, der, derder);
                constexpr double h = 1e-5;
                std::vector<double> vp, dp;
                std::vector<double> vm, dm;
                basis->calc_der(r + h, vp, dp);
                basis->calc_der(r - h, vm, dm);
                for (int i = 0; i < basis_size; ++i) {
                    INFO("cutoff=" << Cutoff::cutoff_type_name(cutoff_type)
                         << " shift=" << XTransForm::shift_type_name(shift_type)
                         << " r=" << r << " i=" << i);
                    REQUIRE(derder[i] ==
                            Approx((dp[i] - dm[i]) / (2.0 * h))
                                .epsilon(2e-5).margin(2e-7));
                }
            }
        }
    }
}

TEST_CASE("Configurable radial basis reproduces legacy combinations",
          "[basis][configurable][compatibility]")
{
    auto quadratic = make_basis(Cutoff::CutoffType::Quadratic,
                                XTransForm::ShiftType::Linear,
                                -0.4, 0.0, rmin, rmax, basis_size, 1.7);
    RBasis_Chebyshev legacy_quadratic_basis(rmin, rmax, basis_size, 1.7);
    compare_bases(*quadratic, legacy_quadratic_basis, {0.5, 1.0, 2.7, 5.0});

    auto poly5 = make_basis(Cutoff::CutoffType::Poly5,
                            XTransForm::ShiftType::Linear,
                            -0.4, 0.0, rmin, rmax, basis_size, 1.7);
    RBasis_ChebyshevPoly5 legacy_poly5(rmin, rmax, basis_size, -0.4, 1.7);
    compare_bases(*poly5, legacy_poly5, {0.5, 1.0, 2.7, 5.0});

    auto cosine = make_basis(Cutoff::CutoffType::Cosine,
                             XTransForm::ShiftType::Linear,
                             -0.4, 0.0, 0.0, rmax, basis_size, 1.7);
    RBasis_ChebyCos legacy_cosine(0.0, rmax, basis_size, 1.7);
    compare_bases(*cosine, legacy_cosine, {0.0, 1.0, 2.7, 5.0});

    auto legacy_formula_cosine = make_basis(
        Cutoff::CutoffType::CosineLeft,
        XTransForm::ShiftType::Linear,
        -0.4, 0.0, rmin, rmax, basis_size, 1.7);
    RBasis_ChebyCos legacy_nonzero_min(
        rmin, rmax, basis_size, 1.7);
    compare_bases(
        *legacy_formula_cosine, legacy_nonzero_min,
        {0.5, 1.0, 2.7, 5.0});

    Cutoff::Parameters poly4_parameters;
    poly4_parameters.b = -0.4;
    auto poly4 = std::make_unique<RBasis_ChebyshevConfigurable>(
        rmin, rmax, basis_size,
        Cutoff::make_cutoff(
            Cutoff::CutoffType::Poly4, rmin, rmax,
            poly4_parameters),
        XTransForm::make_shift(
            XTransForm::ShiftType::Linear, rmin, rmax),
        1.7);
    RBasis_ChebyshevPoly4 legacy_poly4(
        rmin, rmax, basis_size, -0.4, 1.7);
    compare_bases(*poly4, legacy_poly4, {0.5, 1.0, 2.7, 5.0});
}

TEST_CASE("Extended configurable radial basis format round trips",
          "[basis][configurable][io]")
{
    Cutoff::Parameters cutoff_parameters;
    cutoff_parameters.r1 = 1.5;
    cutoff_parameters.r2 = 3.5;
    XTransForm::Parameters shift_parameters;
    shift_parameters.lambda = 1.2;
    shift_parameters.scale_lambda = 0.8;
    RBasis_ChebyshevConfigurable basis(
        rmin, rmax, basis_size,
        Cutoff::make_cutoff(
            Cutoff::CutoffType::Smoothstep7, rmin, rmax,
            cutoff_parameters),
        XTransForm::make_shift(
            XTransForm::ShiftType::ExponentialTanh, rmin, rmax,
            shift_parameters),
        1.3, BasisFunctionType::OneMinusChebyshev);

    std::ostringstream saved;
    saved << std::scientific << std::setprecision(17);
    basis.SaveToStream(saved);
    std::istringstream input(saved.str());
    std::string key, equals, type;
    REQUIRE(static_cast<bool>(input >> key >> equals >> type));
    std::unique_ptr<AnyBasis> restored(
        create_basis_by_name(type, input, 1.3));
    compare_bases(basis, *restored, {0.5, 1.2, 2.7, 3.5, 5.0});
}

TEST_CASE("Configurable radial basis zeroes out-of-range outputs and supports size one",
          "[basis][configurable][bounds]")
{
    auto basis = make_basis(Cutoff::CutoffType::Cosine,
                            XTransForm::ShiftType::Exponential);
    for (double r : {rmin - 1e-6, rmax + 1e-6}) {
        std::vector<double> val{1.0}, der{1.0}, derder{1.0};
        basis->calc_derder(r, val, der, derder);
        REQUIRE(val == std::vector<double>(basis_size, 0.0));
        REQUIRE(der == std::vector<double>(basis_size, 0.0));
        REQUIRE(derder == std::vector<double>(basis_size, 0.0));
    }

    auto size_one = make_basis(Cutoff::CutoffType::Quadratic,
                               XTransForm::ShiftType::Linear,
                               -0.4, 0.0, rmin, rmax, 1);
    std::vector<double> val, der, derder;
    REQUIRE_NOTHROW(size_one->calc_derder(2.0, val, der, derder));
    REQUIRE(val.size() == 1);
}

TEST_CASE("Configurable radial basis stream format round trips and validates tags",
          "[basis][configurable][io]")
{
    auto basis = make_basis(Cutoff::CutoffType::Poly5,
                            XTransForm::ShiftType::Exponential, -0.4, 1.2);
    std::ostringstream saved;
    saved << std::scientific << std::setprecision(17);
    basis->SaveToStream(saved);

    std::istringstream input(saved.str());
    std::string key, equals, type;
    REQUIRE(static_cast<bool>(input >> key >> equals >> type));
    REQUIRE(key == "radial_basis_type");
    REQUIRE(equals == "=");
    REQUIRE(type == "RBChebyshevConfigurable");
    std::unique_ptr<AnyBasis> restored(create_basis_by_name(type, input, 1.0));
    compare_bases(*basis, *restored, {0.5, 1.3, 3.1, 5.0});

    std::istringstream bad_cutoff(
        "min_dist = 0.5 max_dist = 5 radial_basis_size = 3 "
        "cutoff_type = missing shift_type = linear");
    REQUIRE_THROWS_WITH(
        RBasis_ChebyshevConfigurable(bad_cutoff),
        Catch::Matchers::ContainsSubstring("cutoff type"));

    std::istringstream missing_lambda(
        "min_dist = 0.5 max_dist = 5 radial_basis_size = 3 "
        "cutoff_type = quadratic shift_type = exponential");
    REQUIRE_THROWS_WITH(
        RBasis_ChebyshevConfigurable(missing_lambda),
        Catch::Matchers::ContainsSubstring("shift_lambda"));

    REQUIRE_THROWS(Cutoff::make_cutoff(
        Cutoff::CutoffType::Quadratic, 5.0, 5.0));
    REQUIRE_THROWS(XTransForm::make_shift(
        XTransForm::ShiftType::Exponential, rmin, rmax, 1000.0));
}

TEST_CASE("Removed radial basis names are not accepted",
          "[basis][configurable][compatibility]")
{
    const std::array removed_names = {
        "RBChebyDoubleCosLeft",
        "RBChebyDoubleCosShiftedScaledLeft",
        "RBChebyDoubleCosShiftedScaled",
        "RBChebyCosShiftedScaled",
        "RBChebyDoubleSinMiddle",
        "RBChebyDoubleSinDoubleScaledShiftedMiddle",
        "RBChebyshevI2",
        "RBChebyshevPoly2",
        "RBChebyshevPoly3",
        "RBChebyshevPoly5BSide",
        "RBChebyshevPoly7BSideC",
        "RBChebyshevFlip",
        "RBChebyshevShiftedFlip",
        "RBGTO",
        "RBGTOl",
        "RBRCTAny",
        "RBConstant",
    };
    for (const char *name : removed_names) {
        INFO(name);
        REQUIRE(basis_name_to_code(name) == RB_Unknown);
    }
}

TEST_CASE("PotMTP loads configurable and split radial bases",
          "[basis][configurable][mtp]")
{
    std::istringstream configured_stream(
        potential_text(configurable_quadratic));
    std::istringstream legacy_stream(potential_text(legacy_quadratic));
    Potentials::PotMTP configured;
    Potentials::PotMTP legacy;
    configured.Load(configured_stream);
    legacy.Load(legacy_stream);
    REQUIRE(configured.btype == RB_ChebyshevConfigurable);
    REQUIRE(configured.rb_size == 3);

    Atoms configured_atoms = small_atoms();
    Atoms legacy_atoms = configured_atoms;
    Neighborhoods configured_nbhs(configured_atoms, configured.cutoff);
    Neighborhoods legacy_nbhs(legacy_atoms, legacy.cutoff);
    configured.getEFS(configured_atoms, configured_nbhs);
    legacy.getEFS(legacy_atoms, legacy_nbhs);
    REQUIRE(configured_atoms.energy_ ==
            Approx(legacy_atoms.energy_).margin(2e-12));
    for (int i = 0; i < configured_atoms.size(); ++i)
        for (int axis = 0; axis < 3; ++axis)
            REQUIRE(configured_atoms.forces_[i][axis] ==
                    Approx(legacy_atoms.forces_[i][axis]).margin(2e-12));
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 3; ++column)
            REQUIRE(configured_atoms.stress_(row, column) ==
                    Approx(legacy_atoms.stress_(row, column)).margin(2e-12));

    std::ostringstream saved;
    configured.Save(saved);
    std::istringstream reloaded_stream(saved.str());
    Potentials::PotMTP reloaded;
    REQUIRE_NOTHROW(reloaded.Load(reloaded_stream));
    REQUIRE(reloaded.btype == RB_ChebyshevConfigurable);

    Atoms derivative_atoms;
    derivative_atoms.cell_ = 12.0 * Eigen::Matrix3d::Identity();
    derivative_atoms.pos_ = {{0.0, 0.0, 0.0}, {1.8, 0.0, 0.0}};
    derivative_atoms.chemmap_ = {0, 0};
    Potentials::HessianMatrix hessian;
    Neighborhoods derivative_nbhs(derivative_atoms, configured.cutoff);
    REQUIRE_NOTHROW(configured.getEFSHessian(
        derivative_atoms, derivative_nbhs, hessian));

    std::istringstream split_stream(
        potential_text(configurable_quadratic, true));
    Potentials::PotMTP split;
    REQUIRE_NOTHROW(split.Load(split_stream));
    REQUIRE(split.radial_basis_split);
    REQUIRE(split.rb_size == 3);
    REQUIRE(split.p_radical_basis_pair.size() == 1);
    REQUIRE(split.p_radical_basis_pair[0]->name ==
            "RBChebyshevConfigurable");
    std::ostringstream split_saved;
    split.Save(split_saved);
    std::istringstream split_reloaded_stream(split_saved.str());
    Potentials::PotMTP split_reloaded;
    REQUIRE_NOTHROW(split_reloaded.Load(split_reloaded_stream));
    REQUIRE(split_reloaded.radial_basis_split);
    REQUIRE(split_reloaded.rb_size == 3);
}
