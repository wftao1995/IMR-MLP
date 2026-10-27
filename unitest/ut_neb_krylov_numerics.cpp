// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "mindriver/neb.h"
#include <cmath>
#include <numbers>
#include <utility>
#include <vector>

namespace {
// This external quadratic field tests an NEB force-root solver. Its stationary
// point has many unstable directions: it is not a material transition-state or
// minimum-energy-path fixture, and translation invariance is not assumed.
class NegativeTransverseQuadratic : public Potentials::AnyPotential {
public:
    std::vector<double> curvature;

    explicit NegativeTransverseQuadratic(std::vector<double> values)
        : curvature(std::move(values)) {
        cutoff = 1;
        cutoffsq = 1;
        species_count = 1;
    }

    void evaluate(Atoms &atoms) {
        atoms.resetEFS();
        for (int i = 0; i < atoms.size(); ++i) {
            const double y = atoms.pos_[i](1);
            atoms.energy_ -= 0.5 * curvature[i] * y * y;
            atoms.forces_[i](1) = curvature[i] * y;
        }
    }
    void getEFS(Atoms &atoms, AnyNeighborhoods &) override { evaluate(atoms); }
    void getEFSOMPthread(Atoms &atoms) override { evaluate(atoms); }
};

class ShiftedQuadraticSaddle : public Potentials::AnyPotential {
public:
    ShiftedQuadraticSaddle() { cutoff = 1; cutoffsq = 1; species_count = 1; }
    void evaluate(Atoms &atoms) {
        atoms.resetEFS();
        const double x = atoms.pos_[0](0) - 0.2;
        const double y = atoms.pos_[0](1);
        atoms.energy_ = 1 - 0.5 * x * x + 0.5 * y * y;
        atoms.forces_[0] << x, -y, 0;
    }
    void getEFS(Atoms &atoms, AnyNeighborhoods &) override { evaluate(atoms); }
    void getEFSOMPthread(Atoms &atoms) override { evaluate(atoms); }
};
}

TEST_CASE("NEB residual correction expands beyond eighty Krylov directions",
          "[neb][numerics][guard][krylov]") {
    constexpr int degree = 80;
    constexpr int count = degree + 1;
    constexpr double lambda_min = 0.000625;
    constexpr double lambda_max = 100;
    std::vector<double> eigenvalues(count), weights(count);
    double weight_sum = 0;
    for (int i = 0; i < count; ++i) {
        eigenvalues[i] = 0.5 * (lambda_min + lambda_max) +
            0.5 * (lambda_max - lambda_min) *
                std::cos(std::numbers::pi * i / degree);
        // The endpoints of a Chebyshev-Lobatto grid have half barycentric
        // weight. Division by lambda makes the minimax polynomial also the
        // least-squares residual polynomial for this initial residual.
        weights[i] = (i == 0 || i == degree ? 0.5 : 1) / eigenvalues[i];
        weight_sum += weights[i];
    }

    // Independent oracle: at degree m, p(lambda)=T_m(t)/T_m(t(0)).
    // Its values at the m+1 nodes have equal modulus eta. For k=1..m,
    // sum_i w_i*p(lambda_i)*lambda_i^k is proportional to the Lobatto
    // barycentric sum of a polynomial of degree k-1, hence zero. Thus p is
    // the least-squares minimizer, not merely an upper bound. Lower-degree
    // Krylov spaces cannot do better. The 81st direction can solve exactly.
    const double eta_eighty = 1 / std::cosh(degree * std::acosh(
        (lambda_max + lambda_min) / (lambda_max - lambda_min)));
    REQUIRE(eta_eighty > 0.9);
    REQUIRE(eta_eighty < 0.95);

    NegativeTransverseQuadratic potential(eigenvalues);
    Atoms atoms;
    atoms.cell_ = 300 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.chemmap_.assign(count, 0);
    atoms.pos_.resize(count);
    for (int i = 0; i < count; ++i) atoms.pos_[i] << 3 * i, 0, 0;
    std::vector<Atoms> path(3, atoms);
    Eigen::VectorXd initial_y(count);
    for (int i = 0; i < count; ++i) {
        initial_y(i) = std::sqrt(weights[i] / weight_sum) / eigenvalues[i];
        path[0].pos_[i](0) -= 1;
        path[1].pos_[i](1) = initial_y(i);
        path[2].pos_[i](0) += 1;
    }

    NEBDriver driver;
    driver.pot = &potential;
    driver.check_mic = false;
    driver.tangent_type = 2;
    driver.spring = 0;
    driver.climb = false;
    driver.neb_ftol = 1e-10;
    driver.init(path);
    Eigen::VectorXd x, gradient;
    driver.LatPos2X(x);
    gradient.resize(x.size());
    double objective = driver(x, gradient);
    const Eigen::VectorXd initial_x = x;
    REQUIRE(x.size() == 3 * count);
    REQUIRE(gradient.squaredNorm() == Catch::Approx(1).margin(1e-12));
    for (int i = 0; i < count; ++i) {
        // Symmetric endpoints fix the original tangent along x, so projection
        // does not modify these y forces: the Jacobian is -diag(lambda).
        REQUIRE(gradient(3 * i + 1) ==
            Catch::Approx(-eigenvalues[i] * initial_y(i)).margin(1e-12));
    }
    Eigen::VectorXd flow_gradient(gradient.size());
    driver(x - 1e-4 * gradient, flow_gradient);
    REQUIRE(flow_gradient.squaredNorm() > gradient.squaredNorm());

    int callbacks = 0;
    auto stationary_optimizer = [&](int budget) { ++callbacks; return budget; };
    // Permit a complete Newton correction, while retaining a finite explicit
    // move bound. Nine iterations allow one eight-step rejected batch and one
    // fallback; many short restarted solves cannot stand in for expansion.
    const double max_move = 2 * initial_y.norm();
    const int used = driver.minimize_guarded(stationary_optimizer, 9, max_move, x, objective);
    driver(x, gradient);
    Eigen::VectorXd final_y(count);
    for (int i = 0; i < count; ++i) final_y(i) = driver.images[1].pos_[i](1);
    CAPTURE(eta_eighty, used, callbacks, max_move, gradient.norm(), final_y.norm(),
            driver.stop_reason_string());
    CHECK(used == 9);
    CHECK(callbacks == 1);
    CHECK(driver.stop_reason != NEBDriver::StopReason::numerical_failure);
    REQUIRE(x.allFinite());
    REQUIRE(gradient.allFinite());
    CHECK((x - initial_x).norm() <= max_move * (1 + 1e-12));
    // Fixed thresholds include forward-difference roundoff in the smallest
    // eigenvalue; they are much stronger than any <=80-direction model can
    // attain (its squared residual is at least eta_eighty^2 > .81).
    CHECK(gradient.squaredNorm() < 1e-6);
    CHECK(final_y.norm() < 1e-2);
    for (int i = 0; i < count; ++i) {
        CHECK(driver.images[1].pos_[i](0) == Catch::Approx(atoms.pos_[i](0)).margin(1e-12));
        CHECK(driver.images[1].pos_[i](2) == 0);
        CHECK(driver.images.front().pos_[i].isApprox(path.front().pos_[i], 1e-14));
        CHECK(driver.images.back().pos_[i].isApprox(path.back().pos_[i], 1e-14));
    }
}

TEST_CASE("NEB guard restarts its residual model when climbing activates",
          "[neb][numerics][guard][climb-delay]") {
    const int delay = GENERATE(0, 1, 2, 3);
    constexpr double max_move = 0.01;
    ShiftedQuadraticSaddle potential;
    Atoms atoms;
    atoms.cell_ = 10 * Eigen::Matrix3d::Identity();
    atoms.pbc_ = {false, false, false};
    atoms.pos_ = {{-1, 0, 0}};
    atoms.chemmap_ = {0};
    std::vector<Atoms> path(3, atoms);
    path[1].pos_[0] << 0, 0.01, 0;
    path[2].pos_[0] << 1, 0, 0;
    NEBDriver driver;
    driver.pot = &potential;
    driver.check_mic = false;
    driver.tangent_type = 2;
    driver.spring = 0;
    driver.climb = true;
    driver.climb_delay = delay;
    driver.neb_ftol = 1e-12;
    driver.init(path);
    Eigen::VectorXd x;
    driver.LatPos2X(x);
    const Eigen::VectorXd initial_x = x;
    double objective = 0;
    int callbacks = 0;
    auto stationary_optimizer = [&](int) { ++callbacks; return 1; };

    // Initially the tangent is horizontal. Before climbing, r=(0,.01,0)
    // has squared norm .0001. Activating the parallel-force reflection gives
    // r=(-.2,.01,0), with squared norm .0401 at the SAME coordinates. Comparing
    // trial residuals in the latter field with the former field's baseline
    // rejects every bounded descent step and falsely reports a numerical
    // failure. The short budget permits a phase restart without requiring it
    // to perform an extra optimizer step beyond the requested two.
    const int used = driver.minimize_guarded(stationary_optimizer, 2, max_move, x, objective);
    CAPTURE(delay, used, callbacks, driver.force_evaluations, driver.stop_reason_string());
    CHECK(used == 2);
    CHECK(driver.stop_reason != NEBDriver::StopReason::numerical_failure);
    REQUIRE(driver.climbing_active);
    REQUIRE(x.allFinite());
    CHECK((x - initial_x).norm() <= max_move * (1 + 1e-12));

    Eigen::VectorXd gradient(x.size());
    objective = driver(x, gradient);
    const auto middle = driver.images[1].pos_[0];
    const Eigen::RowVector3d backward = (middle - path.front().pos_[0]).normalized();
    const Eigen::RowVector3d forward = (path.back().pos_[0] - middle).normalized();
    const Eigen::RowVector3d tangent = (backward + forward).normalized();
    const Eigen::RowVector3d physical_force(middle(0) - 0.2, -middle(1), 0);
    const Eigen::Vector3d expected_gradient =
        (-physical_force + 2 * physical_force.dot(tangent) * tangent).transpose();
    REQUIRE(gradient.allFinite());
    CHECK(gradient.isApprox(expected_gradient, 1e-12));
    CHECK(driver.max_neb_force == Catch::Approx(expected_gradient.norm()).margin(1e-12));
    CHECK(driver.max_neb_force <= std::sqrt(0.0401) + 1e-12);
    CHECK_FALSE(driver.check_convergence());
    if (delay == 0) CHECK(driver.max_neb_force < 0.999 * std::sqrt(0.0401));
    const double expected_energy = 0.28 + 0.68 +
        1 - 0.5 * std::pow(middle(0) - 0.2, 2) + 0.5 * middle(1) * middle(1);
    CHECK(objective == Catch::Approx(expected_energy).margin(1e-12));
    CHECK(driver.images.front().pos_[0].isApprox(path.front().pos_[0], 1e-14));
    CHECK(driver.images.back().pos_[0].isApprox(path.back().pos_[0], 1e-14));
}
