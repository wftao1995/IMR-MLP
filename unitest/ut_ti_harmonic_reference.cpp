#include <catch2/catch_all.hpp>

#include "tools/constants.h"
#include "tools/hessian_utils.h"
#include "tools/ti_harmonic_reference.h"

using Catch::Approx;

namespace {

Eigen::MatrixXd make_hessian(const std::vector<double> &masses,
                             const Eigen::Vector3d &internal_values)
{
    const Eigen::MatrixXd basis = HessianUtils::mass_weighted_internal_basis(masses);
    Eigen::MatrixXd dynamical = basis * internal_values.asDiagonal() * basis.transpose();
    return HessianUtils::from_dynamical_matrix(dynamical, masses);
}

TIHarmonicReference::ForceSample make_force_sample(
    const TIHarmonicReference::ReferenceResult &reference,
    const Eigen::Vector3d &q,
    const Eigen::Matrix3d &target,
    const Eigen::Vector3d &intercept = Eigen::Vector3d::Zero())
{
    TIHarmonicReference::ForceSample sample;
    sample.displacement_cartesian = (reference.internal_basis * q).array()
        / reference.sqrt_mass_coordinates.array();
    sample.force_cartesian = (reference.internal_basis * (intercept - target * q)).array()
        * reference.sqrt_mass_coordinates.array();
    return sample;
}

std::vector<TIHarmonicReference::ForceSample> axis_force_samples(
    const TIHarmonicReference::ReferenceResult &reference,
    const Eigen::Matrix3d &target)
{
    std::vector<TIHarmonicReference::ForceSample> samples;
    for (int axis = 0; axis < 3; ++axis)
        for (double sign : {-1.0, 1.0})
            samples.push_back(make_force_sample(
                reference, sign * 0.1 * Eigen::Vector3d::Unit(axis), target));
    return samples;
}

} // namespace

TEST_CASE("TI reference floors unstable internal modes and preserves translations", "[ti-reference]")
{
    const std::vector<double> masses{1.0, 4.0};
    TIHarmonicReference::Options options;
    options.unstable_frequency_thz = TIHarmonicReference::frequency_conversion_thz * 0.5;
    const auto result = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(-4.0, 1.0, 2.0)), masses, options);
    REQUIRE(result.diagnostics.corrected_modes == 1);
    REQUIRE(result.diagnostics.eigenvalues_after.minCoeff() == Approx(0.25).epsilon(1e-11));
    REQUIRE(result.diagnostics.acoustic_sum_rule_error_after < 1e-11);
    REQUIRE((result.hessian - result.hessian.transpose()).norm() < 1e-11);
}

TEST_CASE("TI reference applies the strict PD floor after every stabilization policy",
          "[ti-reference][stabilization]")
{
    const std::vector<double> masses{1.0, 4.0};
    const Eigen::MatrixXd hessian = make_hessian(
        masses, Eigen::Vector3d(-4.0, 0.01, 2.0));
    for (const auto policy : {
             TIHarmonicReference::StabilizationPolicy::NegativeFloor,
             TIHarmonicReference::StabilizationPolicy::SpectralFloor,
             TIHarmonicReference::StabilizationPolicy::AbsoluteValue}) {
        TIHarmonicReference::Options options;
        options.stabilization = policy;
        options.unstable_frequency_thz = TIHarmonicReference::frequency_conversion_thz;
        options.positive_definite_floor_thz =
            0.5 * TIHarmonicReference::frequency_conversion_thz;
        const auto result = TIHarmonicReference::build_reference(
            hessian, masses, options);
        const double expected_minimum =
            policy == TIHarmonicReference::StabilizationPolicy::SpectralFloor
            ? 1.0 : 0.25;
        REQUIRE(result.diagnostics.eigenvalues_after.minCoeff()
                == Approx(expected_minimum).epsilon(1e-11));
        REQUIRE(result.diagnostics.corrected_modes == 2);
    }
}

TEST_CASE("TI root representations round trip and expose correct gradients", "[ti-reference][root]")
{
    Eigen::Matrix2d matrix;
    matrix << 2.0, 0.3, 0.3, 1.0;
    Eigen::Matrix2d gradient;
    gradient << 0.7, -0.2, -0.2, 0.4;
    for (auto representation : {TIHarmonicReference::RootRepresentation::SquareRoot,
                                TIHarmonicReference::RootRepresentation::FourthRoot}) {
        const Eigen::MatrixXd root = TIHarmonicReference::to_root(matrix, representation);
        REQUIRE(TIHarmonicReference::from_root(root, representation).isApprox(matrix, 1e-11));
        const Eigen::MatrixXd analytic = TIHarmonicReference::pullback_gradient(root, gradient, representation);
        const double epsilon = 1e-6;
        for (int i = 0; i < 2; ++i) {
            Eigen::MatrixXd plus = root, minus = root;
            plus(i, i) += epsilon;
            minus(i, i) -= epsilon;
            const double numeric = ((TIHarmonicReference::from_root(plus, representation)
                - TIHarmonicReference::from_root(minus, representation))
                .cwiseProduct(gradient)).sum() / (2.0 * epsilon);
            REQUIRE(analytic(i, i) == Approx(numeric).epsilon(2e-6));
        }
    }
}

TEST_CASE("Classical TI ensemble has zero COM and canonical modal variance", "[ti-reference][ensemble]")
{
    const std::vector<double> masses{1.0, 4.0};
    TIHarmonicReference::Options options;
    const auto reference = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 4.0)), masses, options);
    TIHarmonicReference::EnsembleOptions ensemble_options;
    ensemble_options.temperature_kelvin = 500.0;
    ensemble_options.size = 12000;
    ensemble_options.seed = 1234;
    const auto ensemble = TIHarmonicReference::sample_classical_ensemble(reference, ensemble_options);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(reference.internal_dynamical_matrix);
    Eigen::Vector3d modal_variance = Eigen::Vector3d::Zero();
    for (const auto &sample : ensemble) {
        Eigen::Vector3d com = masses[0] * sample.displacement_cartesian.segment<3>(0)
                            + masses[1] * sample.displacement_cartesian.segment<3>(3);
        REQUIRE(com.norm() < 1e-11);
        REQUIRE(sample.reference_force_cartesian.isApprox(
            -reference.hessian * sample.displacement_cartesian, 1e-12));
        REQUIRE(sample.harmonic_energy == Approx(0.5 * sample.displacement_cartesian.dot(
            reference.hessian * sample.displacement_cartesian)).epsilon(1e-12));
        const Eigen::VectorXd weighted = reference.sqrt_mass_coordinates.array()
            * sample.displacement_cartesian.array();
        const Eigen::Vector3d q = solver.eigenvectors().transpose()
            * reference.internal_basis.transpose() * weighted;
        modal_variance.array() += q.array().square();
    }
    modal_variance /= static_cast<double>(ensemble.size());
    const Eigen::Vector3d expected = (Boltzmann * ensemble_options.temperature_kelvin
        / solver.eigenvalues().array()).matrix();
    for (int i = 0; i < 3; ++i)
        REQUIRE(modal_variance(i) == Approx(expected(i)).epsilon(0.04));
}

TEST_CASE("TI ensemble sampling floor limits soft-mode variance without changing forces",
          "[ti-reference][ensemble]")
{
    const std::vector<double> masses{1.0, 4.0};
    const auto reference = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(0.01, 1.0, 4.0)), masses,
        TIHarmonicReference::Options{});
    TIHarmonicReference::EnsembleOptions options;
    options.temperature_kelvin = 500.0;
    options.sampling_floor_thz = 0.5 * TIHarmonicReference::frequency_conversion_thz;
    options.size = 12000;
    options.seed = 4321;
    options.antithetic = true;
    const auto ensemble = TIHarmonicReference::sample_classical_ensemble(reference, options);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(reference.internal_dynamical_matrix);
    Eigen::Vector3d modal_variance = Eigen::Vector3d::Zero();
    for (const auto &sample : ensemble) {
        REQUIRE(sample.reference_force_cartesian.isApprox(
            -reference.hessian * sample.displacement_cartesian, 1e-12));
        const Eigen::VectorXd weighted = reference.sqrt_mass_coordinates.array()
            * sample.displacement_cartesian.array();
        const Eigen::Vector3d q = solver.eigenvectors().transpose()
            * reference.internal_basis.transpose() * weighted;
        modal_variance.array() += q.array().square();
    }
    modal_variance /= static_cast<double>(ensemble.size());
    const Eigen::Vector3d sampling_values(0.25, 1.0, 4.0);
    const Eigen::Vector3d expected = (Boltzmann * options.temperature_kelvin
        / sampling_values.array()).matrix();
    for (int i = 0; i < 3; ++i)
        REQUIRE(modal_variance(i) == Approx(expected(i)).epsilon(0.04));
}

TEST_CASE("Root full-force fitting recovers coupled internal force constants", "[ti-reference][fit]")
{
    const std::vector<double> masses{1.0, 2.0};
    const Eigen::MatrixXd initial_hessian = make_hessian(masses, Eigen::Vector3d(1.0, 1.5, 2.0));
    TIHarmonicReference::Options base_options;
    const auto initial = TIHarmonicReference::build_reference(initial_hessian, masses, base_options);
    Eigen::Matrix3d target;
    target << 1.2, 0.25, -0.1,
              0.25, 1.8, 0.15,
             -0.1, 0.15, 2.4;
    std::vector<TIHarmonicReference::TrajectorySample> samples;
    for (int isample = 0; isample < 18; ++isample) {
        Eigen::Vector3d q;
        q << std::sin(0.7 * isample), std::cos(0.4 * isample), std::sin(1.1 * isample + 0.2);
        const Eigen::VectorXd weighted_displacement = initial.internal_basis * q;
        const Eigen::VectorXd weighted_force = initial.internal_basis * (-target * q);
        TIHarmonicReference::TrajectorySample sample;
        sample.displacement_cartesian = weighted_displacement.array()
            / initial.sqrt_mass_coordinates.array();
        sample.delta_force_cartesian = weighted_force.array()
            * initial.sqrt_mass_coordinates.array();
        samples.push_back(std::move(sample));
    }
    TIHarmonicReference::Options fit_options;
    fit_options.fit_method = TIHarmonicReference::FitMethod::FullForce;
    fit_options.root_representation = TIHarmonicReference::RootRepresentation::SquareRoot;
    fit_options.ridge = 0.0;
    fit_options.gradient_tolerance = 1e-12;
    fit_options.max_iterations = 500;
    const auto fitted = TIHarmonicReference::build_reference(
        initial_hessian, masses, fit_options, {}, samples);
    REQUIRE(fitted.internal_dynamical_matrix.isApprox(target, 2e-6));
    REQUIRE(fitted.diagnostics.force_rmse < 1e-6);
}

TEST_CASE("TI space-group averaging respects atom permutations", "[ti-reference][symmetry]")
{
    const std::vector<double> masses{1.0, 1.0};
    TIHarmonicReference::SymmetryOperation identity;
    identity.atom_mapping = {0, 1};
    TIHarmonicReference::SymmetryOperation exchange;
    exchange.atom_mapping = {1, 0};
    const auto result = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 4.0)), masses,
        TIHarmonicReference::Options{}, {identity, exchange});
    Eigen::MatrixXd permutation = Eigen::MatrixXd::Zero(6, 6);
    permutation.block<3, 3>(0, 3).setIdentity();
    permutation.block<3, 3>(3, 0).setIdentity();
    REQUIRE((permutation * result.dynamical_matrix * permutation.transpose()
             - result.dynamical_matrix).norm() < 1e-11);
    REQUIRE(result.diagnostics.space_group_error_after < 1e-11);
    REQUIRE_THROWS_AS(TIHarmonicReference::build_reference(
        make_hessian({1.0, 2.0}, Eigen::Vector3d(1.0, 2.0, 4.0)), {1.0, 2.0},
        TIHarmonicReference::Options{}, {exchange}), std::invalid_argument);
}

TEST_CASE("TI reference handles a one-atom cell", "[ti-reference]")
{
    const auto reference = TIHarmonicReference::build_reference(
        Eigen::Matrix3d::Random(), {2.0}, TIHarmonicReference::Options{});
    REQUIRE(reference.hessian.isZero(0.0));
    REQUIRE(reference.internal_dynamical_matrix.size() == 0);
}

TEST_CASE("Antithetic TI sampling emits exact displacement pairs", "[ti-reference][ensemble]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto reference = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 3.0)), masses,
        TIHarmonicReference::Options{});
    TIHarmonicReference::EnsembleOptions options;
    options.size = 7;
    options.antithetic = true;
    options.seed = 42;
    const auto samples = TIHarmonicReference::sample_classical_ensemble(reference, options);
    REQUIRE(samples.size() == 7);
    for (int i = 0; i < 6; i += 2) {
        REQUIRE((samples[i].displacement_cartesian
                 + samples[i + 1].displacement_cartesian).norm() < 1e-13);
        REQUIRE(samples[i].harmonic_energy == Approx(samples[i + 1].harmonic_energy));
    }
}

TEST_CASE("TI ensemble rejection sampling preserves antithetic pairs",
          "[ti-reference][ensemble][rejection]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto reference = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 3.0)), masses,
        TIHarmonicReference::Options{});
    TIHarmonicReference::EnsembleOptions options;
    options.size = 6;
    options.antithetic = true;
    options.seed = 42;
    int groups = 0;
    const auto result = TIHarmonicReference::sample_classical_ensemble_filtered(
        reference, options, 10,
        [&](std::span<const TIHarmonicReference::EnsembleSample> candidates) {
            REQUIRE(candidates.size() == 2);
            return groups++ != 0;
        });
    REQUIRE(result.samples.size() == 6);
    REQUIRE(result.attempted_samples == 8);
    REQUIRE(result.rejected_samples == 2);
    for (int i = 0; i < 6; i += 2)
        REQUIRE((result.samples[i].displacement_cartesian
                 + result.samples[i + 1].displacement_cartesian).norm() < 1e-13);
}

TEST_CASE("TI ensemble rejection sampling reports an exhausted attempt budget",
          "[ti-reference][ensemble][rejection]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto reference = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 3.0)), masses,
        TIHarmonicReference::Options{});
    TIHarmonicReference::EnsembleOptions options;
    options.size = 4;
    options.antithetic = true;
    const auto result = TIHarmonicReference::sample_classical_ensemble_filtered(
        reference, options, 6,
        [](std::span<const TIHarmonicReference::EnsembleSample>) { return false; });
    REQUIRE(result.samples.empty());
    REQUIRE(result.attempted_samples == 6);
    REQUIRE(result.rejected_samples == 6);
}

TEST_CASE("Unfiltered TI ensemble wrapper preserves the filtered sample sequence",
          "[ti-reference][ensemble][rejection]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto reference = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 3.0)), masses,
        TIHarmonicReference::Options{});
    TIHarmonicReference::EnsembleOptions options;
    options.size = 7;
    options.antithetic = true;
    options.seed = 9876;
    const auto original = TIHarmonicReference::sample_classical_ensemble(reference, options);
    const auto filtered = TIHarmonicReference::sample_classical_ensemble_filtered(
        reference, options, options.size,
        [](std::span<const TIHarmonicReference::EnsembleSample>) { return true; });
    REQUIRE(filtered.attempted_samples == options.size);
    REQUIRE(filtered.rejected_samples == 0);
    REQUIRE(filtered.samples.size() == original.size());
    for (size_t i = 0; i < original.size(); ++i)
        REQUIRE(filtered.samples[i].displacement_cartesian.isApprox(
            original[i].displacement_cartesian, 0.0));
}

TEST_CASE("Positive force matching removes the constant force intercept",
          "[ti-reference][force-match]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto initial = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 1.5, 2.0)), masses,
        TIHarmonicReference::Options{});
    Eigen::Matrix3d target;
    target << 1.2, 0.2, -0.1,
              0.2, 1.8, 0.15,
             -0.1, 0.15, 2.4;
    const Eigen::Vector3d intercept(0.4, -0.2, 0.3);
    std::vector<TIHarmonicReference::ForceSample> samples;
    for (int isample = 0; isample < 24; ++isample) {
        Eigen::Vector3d q;
        q << std::sin(0.7 * isample), std::cos(0.4 * isample),
             std::sin(1.1 * isample + 0.2);
        const Eigen::VectorXd weighted_displacement = initial.internal_basis * q;
        const Eigen::VectorXd weighted_force = initial.internal_basis * (intercept - target * q);
        TIHarmonicReference::ForceSample sample;
        sample.displacement_cartesian = weighted_displacement.array()
            / initial.sqrt_mass_coordinates.array();
        sample.force_cartesian = weighted_force.array()
            * initial.sqrt_mass_coordinates.array();
        samples.push_back(std::move(sample));
    }
    TIHarmonicReference::ForceMatchOptions options;
    options.ridge = 0.0;
    options.coupled_blocks = true;
    options.soft_block_frequency_thz = 100.0;
    options.soft_block_max_modes = 3;
    options.gradient_tolerance = 1e-12;
    options.max_iterations = 500;
    const auto fitted = TIHarmonicReference::fit_force_samples(
        initial, masses, options, samples);
    REQUIRE(fitted.reference.internal_dynamical_matrix.isApprox(target, 3e-6));
    REQUIRE(fitted.diagnostics.force_rmse < 1e-6);
    REQUIRE(fitted.diagnostics.mean_force_rms > 0.1);
    REQUIRE(fitted.reference.diagnostics.acoustic_sum_rule_error_after < 1e-11);
}

TEST_CASE("Self-consistent force matching converges for an exact harmonic evaluator",
          "[ti-reference][self-consistent]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto initial = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(0.8, 1.1, 1.6)), masses,
        TIHarmonicReference::Options{});
    Eigen::Matrix3d target;
    target << 1.4, 0.12, -0.08,
              0.12, 1.9, 0.05,
             -0.08, 0.05, 2.3;
    TIHarmonicReference::SelfConsistentOptions options;
    options.ensemble.size = 48;
    options.ensemble.antithetic = true;
    options.fit.ridge = 0.0;
    options.fit.soft_block_frequency_thz = 100.0;
    options.fit.soft_block_max_modes = 3;
    options.fit.gradient_tolerance = 1e-12;
    options.fit.max_iterations = 500;
    options.warmup_iterations = 0;
    options.minimum_iterations = 2;
    options.maximum_iterations = 5;
    options.consecutive_converged = 1;
    options.mixing = 1.0;
    options.relative_dynamical_tolerance = 1e-6;
    options.maximum_frequency_change_thz = 1e-5;
    auto evaluator = [&](const TIHarmonicReference::ReferenceResult &reference,
                         const std::vector<TIHarmonicReference::EnsembleSample> &ensemble) {
        std::vector<TIHarmonicReference::ForceSample> samples;
        for (const auto &draw : ensemble) {
            const Eigen::VectorXd weighted = reference.sqrt_mass_coordinates.array()
                * draw.displacement_cartesian.array();
            const Eigen::VectorXd q = reference.internal_basis.transpose() * weighted;
            const Eigen::VectorXd weighted_force = reference.internal_basis * (-target * q);
            TIHarmonicReference::ForceSample sample;
            sample.displacement_cartesian = draw.displacement_cartesian;
            sample.force_cartesian = weighted_force.array()
                * reference.sqrt_mass_coordinates.array();
            samples.push_back(std::move(sample));
        }
        return samples;
    };
    const auto result = TIHarmonicReference::iterate_reference(
        initial, masses, {}, options, evaluator);
    REQUIRE(result.converged);
    REQUIRE(result.reference.internal_dynamical_matrix.isApprox(target, 4e-6));
    REQUIRE(result.reference.diagnostics.acoustic_sum_rule_error_after < 1e-11);
}

TEST_CASE("Self-consistent convergence waits for the coupled stage",
          "[ti-reference][self-consistent]")
{
    const std::vector<double> masses{1.0, 2.0};
    const Eigen::Vector3d diagonal(0.005, 0.008, 0.012);
    const auto initial = TIHarmonicReference::build_reference(
        make_hessian(masses, diagonal), masses, TIHarmonicReference::Options{});
    Eigen::Matrix3d target = diagonal.asDiagonal();
    target(0, 1) = target(1, 0) = 0.001;
    TIHarmonicReference::SelfConsistentOptions options;
    options.ensemble.size = 128;
    options.ensemble.antithetic = true;
    options.ensemble.sampling_floor_thz = 1.0;
    auto evaluator = [&](const TIHarmonicReference::ReferenceResult &reference,
                         const std::vector<TIHarmonicReference::EnsembleSample> &ensemble) {
        std::vector<TIHarmonicReference::ForceSample> samples;
        for (const auto &draw : ensemble) {
            const Eigen::Vector3d q = reference.internal_basis.transpose()
                * (reference.sqrt_mass_coordinates.array()
                   * draw.displacement_cartesian.array()).matrix();
            samples.push_back(make_force_sample(reference, q, target));
        }
        return samples;
    };
    SECTION("Default stopping criteria include consecutive coupled iterations") {
        const auto result = TIHarmonicReference::iterate_reference(
            initial, masses, {}, options, evaluator);
        REQUIRE(result.converged);
        REQUIRE(result.iterations.size()
                >= options.warmup_iterations + options.consecutive_converged);
        REQUIRE(result.iterations.back().coupled_blocks > 0);
        REQUIRE(result.reference.internal_dynamical_matrix(0, 1) > 0.0003);
    }
    SECTION("Exhausting the budget during warmup does not report convergence") {
        options.maximum_iterations = options.warmup_iterations;
        const auto result = TIHarmonicReference::iterate_reference(
            initial, masses, {}, options, evaluator);
        REQUIRE_FALSE(result.converged);
        REQUIRE(result.iterations.size() == options.maximum_iterations);
    }
}

TEST_CASE("Coupled force matching recovers stiffness from the positive floor",
          "[ti-reference][force-match]")
{
    const std::vector<double> masses{1.0, 2.0};
    TIHarmonicReference::ForceMatchOptions options;
    const double floor = std::pow(options.positive_definite_floor_thz
        / TIHarmonicReference::frequency_conversion_thz, 2);
    const auto initial = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d::Constant(floor)), masses,
        TIHarmonicReference::Options{});
    const Eigen::Matrix3d target = 0.005 * Eigen::Matrix3d::Identity();
    const auto samples = axis_force_samples(initial, target);
    options.ridge = 0.0;
    options.coupled_blocks = true;
    for (const auto root : {TIHarmonicReference::RootRepresentation::SquareRoot,
                            TIHarmonicReference::RootRepresentation::FourthRoot}) {
        CAPTURE(TIHarmonicReference::root_name(root));
        options.root_representation = root;
        const auto fitted = TIHarmonicReference::fit_force_samples(
            initial, masses, options, samples);
        REQUIRE(fitted.diagnostics.coupled_blocks == 1);
        REQUIRE(fitted.reference.internal_dynamical_matrix.isApprox(target, 1e-6));
        REQUIRE(fitted.diagnostics.force_rmse < 1e-8);
    }
}

TEST_CASE("Modal and coupled fits share a sample-count independent ridge penalty",
          "[ti-reference][force-match]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto initial = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d::Constant(2.0)), masses,
        TIHarmonicReference::Options{});
    const auto samples = axis_force_samples(initial, Eigen::Matrix3d::Identity());
    // Mean q_i^2 = 0.01/3: the ridge optimum is (0.01/3 + 0.01*2)/(0.01/3 + 0.01).
    const Eigen::Matrix3d expected = 1.75 * Eigen::Matrix3d::Identity();
    TIHarmonicReference::ForceMatchOptions options;
    options.ridge = 0.01;
    options.soft_block_frequency_thz = 100.0;
    for (bool coupled : {false, true}) {
        CAPTURE(coupled);
        options.coupled_blocks = coupled;
        auto repeated = samples;
        for (int copy = 1; copy < 4; ++copy)
            repeated.insert(repeated.end(), samples.begin(), samples.end());
        auto reweighted = samples;
        for (auto &sample : reweighted) sample.weight = 7.0;
        auto split_weight = samples;
        split_weight.front().weight = 0.25;
        split_weight.push_back(samples.front());
        split_weight.back().weight = 0.75;
        const std::vector<TIHarmonicReference::ForceSample> *datasets[] = {
            &samples, &repeated, &reweighted, &split_weight};
        for (const auto *data : datasets) {
            const auto fitted = TIHarmonicReference::fit_force_samples(
                initial, masses, options, *data);
            REQUIRE(fitted.reference.internal_dynamical_matrix.isApprox(expected, 1e-6));
        }
    }
}

TEST_CASE("Harmonic validation retains force offsets and single-frame errors",
          "[ti-reference][validation]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto reference = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d::Constant(0.005)), masses,
        TIHarmonicReference::Options{});
    const Eigen::Matrix3d target = reference.internal_dynamical_matrix;
    const Eigen::Vector3d intercept(0.4, -0.2, 0.3);
    std::vector<TIHarmonicReference::ForceSample> samples;
    samples.push_back(make_force_sample(reference, Eigen::Vector3d(0.1, 0.2, 0.3),
                                       target, intercept));
    const double expected = intercept.norm() / std::sqrt(3.0);
    REQUIRE(TIHarmonicReference::force_sample_rmse(reference, samples)
            == Approx(expected).margin(1e-12));
    samples.push_back(make_force_sample(reference, Eigen::Vector3d(-0.2, 0.1, 0.2),
                                       target, intercept));
    REQUIRE(TIHarmonicReference::force_sample_rmse(reference, samples)
            == Approx(expected).margin(1e-12));
    samples.back().force_cartesian *= 2.0;
    samples.back().weight = 3.0;
    const double second = TIHarmonicReference::force_sample_rmse(reference, {samples.back()});
    REQUIRE(TIHarmonicReference::force_sample_rmse(reference, samples)
            == Approx(std::sqrt((expected * expected + 3.0 * second * second) / 4.0)));
    REQUIRE(TIHarmonicReference::force_sample_rmse(reference, axis_force_samples(reference, target))
            < 1e-12);
    REQUIRE(std::isnan(TIHarmonicReference::force_sample_rmse(reference, {})));
}

TEST_CASE("Self-consistent force matching records rejected replacement samples",
          "[ti-reference][self-consistent][rejection]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto initial = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 3.0)), masses,
        TIHarmonicReference::Options{});
    TIHarmonicReference::SelfConsistentOptions options;
    options.ensemble.size = 6;
    options.ensemble.antithetic = true;
    options.minimum_iterations = 1;
    options.maximum_iterations = 1;
    options.consecutive_converged = 1;
    options.maximum_sampling_attempts = 10;
    int filter_calls = 0;
    auto filter = [&](std::span<const TIHarmonicReference::EnsembleSample>) {
        return filter_calls++ != 0;
    };
    auto evaluator = [](const TIHarmonicReference::ReferenceResult &,
                        const std::vector<TIHarmonicReference::EnsembleSample> &ensemble) {
        std::vector<TIHarmonicReference::ForceSample> samples;
        samples.reserve(ensemble.size());
        for (const auto &draw : ensemble) {
            TIHarmonicReference::ForceSample sample;
            sample.displacement_cartesian = draw.displacement_cartesian;
            sample.force_cartesian = draw.reference_force_cartesian;
            samples.push_back(std::move(sample));
        }
        return samples;
    };
    const auto result = TIHarmonicReference::iterate_reference(
        initial, masses, {}, options, evaluator, filter);
    REQUIRE(result.iterations.size() == 1);
    REQUIRE(result.iterations[0].sampling_attempts == 8);
    REQUIRE(result.iterations[0].rejected_samples == 2);
    REQUIRE(result.iterations[0].sampling_acceptance_rate == Approx(0.75));
}

TEST_CASE("Self-consistent rejection budget fails before force evaluation",
          "[ti-reference][self-consistent][rejection]")
{
    const std::vector<double> masses{1.0, 2.0};
    const auto initial = TIHarmonicReference::build_reference(
        make_hessian(masses, Eigen::Vector3d(1.0, 2.0, 3.0)), masses,
        TIHarmonicReference::Options{});
    TIHarmonicReference::SelfConsistentOptions options;
    options.ensemble.size = 4;
    options.ensemble.antithetic = true;
    options.minimum_iterations = 1;
    options.maximum_iterations = 1;
    options.maximum_sampling_attempts = 6;
    int evaluator_calls = 0;
    auto evaluator = [&](const TIHarmonicReference::ReferenceResult &,
                         const std::vector<TIHarmonicReference::EnsembleSample> &) {
        ++evaluator_calls;
        return std::vector<TIHarmonicReference::ForceSample>{};
    };
    REQUIRE_THROWS_WITH(
        TIHarmonicReference::iterate_reference(
            initial, masses, {}, options, evaluator,
            [](std::span<const TIHarmonicReference::EnsembleSample>) { return false; }),
        Catch::Matchers::ContainsSubstring(
            "accepted 0 of 4 samples after 6 attempts"));
    REQUIRE(evaluator_calls == 0);
}
