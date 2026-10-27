#include <catch2/catch_all.hpp>
#include "train/gn_diagonal.h"
#include <limits>

using Catch::Approx;

TEST_CASE("GN diagonal residual probes reproduce identity and zero columns exactly", "[training][gn-diagonal]") {
    const int probes = GENERATE(1, 7, 100);
    Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(3,5);
    jacobian(0,0) = 1.; jacobian(1,1) = -2.; jacobian(2,2) = 3.;
    int calls = 0;
    const auto diagonal = Training::estimate_gn_diagonal(3,5,probes,17,
        [&](const Eigen::VectorXd &z, Eigen::VectorXd &product) {
            ++calls;
            REQUIRE(z.size() == 3);
            for (const double value : z) REQUIRE(std::abs(value) == 1.);
            product = jacobian.transpose() * z;
        });
    REQUIRE(calls == probes);
    REQUIRE(diagonal[0] == Approx(2.).epsilon(1e-13));
    REQUIRE(diagonal[1] == Approx(8.).epsilon(1e-13));
    REQUIRE(diagonal[2] == Approx(18.).epsilon(1e-13));
    REQUIRE(diagonal[3] == 0.);
    REQUIRE(diagonal[4] == 0.);
}

TEST_CASE("GN diagonal estimates a coupled weighted Jacobian reproducibly", "[training][gn-diagonal]") {
    Eigen::Matrix<double,4,3> jacobian;
    jacobian << 1., -2., .3, 2., 3., -.4, -.5, 1., 2., 3., .1, -1.;
    // Residual weights/mean reduction have already been applied to J.
    jacobian.row(0) *= std::sqrt(.2 / 7.);
    jacobian.row(1) *= std::sqrt(2. / 7.);
    jacobian.row(2) *= std::sqrt(.5 / 7.);
    jacobian.row(3) *= std::sqrt(3. / 7.);
    const Training::ResidualVJP vjp = [&](const Eigen::VectorXd &z, Eigen::VectorXd &out) {
        out = jacobian.transpose() * z;
    };
    const auto first = Training::estimate_gn_diagonal(4,3,32768,1729,vjp);
    const auto repeated = Training::estimate_gn_diagonal(4,3,32768,1729,vjp);
    const auto other = Training::estimate_gn_diagonal(4,3,32768,1730,vjp);
    REQUIRE(first == repeated);
    REQUIRE_FALSE(first == other);
    const Eigen::VectorXd exact = 2. * jacobian.array().square().colwise().sum().transpose().matrix();
    for (int i = 0; i < 3; ++i) {
        CAPTURE(i);
        REQUIRE(first[i] == Approx(exact[i]).epsilon(.04));
        REQUIRE(other[i] == Approx(exact[i]).epsilon(.04));
    }
}

TEST_CASE("GN diagonal keeps independent residual blocks additive", "[training][gn-diagonal]") {
    // Single-row blocks give exact estimates independent of seed and expose the
    // cross terms that would appear if rank-local VJPs were summed then squared.
    Eigen::Vector3d a(1.,2.,3.), b(-2.,1.,.5);
    auto block = [](const Eigen::Vector3d &j, std::uint64_t seed) {
        return Training::estimate_gn_diagonal(1,3,11,seed,
            [&](const Eigen::VectorXd &z, Eigen::VectorXd &out) { out = j * z[0]; });
    };
    const auto combined = (block(a,3) + block(b,4)).eval();
    for (int i = 0; i < 3; ++i)
        REQUIRE(combined[i] == Approx(2. * (a[i]*a[i] + b[i]*b[i])).epsilon(1e-13));
}

TEST_CASE("GN diagonal handles empty dimensions without calling the VJP", "[training][gn-diagonal]") {
    int calls = 0;
    const Training::ResidualVJP vjp = [&](const auto &, auto &) { ++calls; };
    REQUIRE(Training::estimate_gn_diagonal(0,4,3,0,vjp).isZero());
    REQUIRE(Training::estimate_gn_diagonal(4,0,3,0,vjp).size() == 0);
    REQUIRE(calls == 0);
    REQUIRE_THROWS_AS(Training::estimate_gn_diagonal(-1,3,2,0,vjp),std::invalid_argument);
    REQUIRE_THROWS_AS(Training::estimate_gn_diagonal(3,-1,2,0,vjp),std::invalid_argument);
    REQUIRE_THROWS_AS(Training::estimate_gn_diagonal(3,3,0,0,vjp),std::invalid_argument);
    REQUIRE_THROWS_AS(Training::estimate_gn_diagonal(0,3,0,0,vjp),std::invalid_argument);
    REQUIRE_THROWS_AS(Training::estimate_gn_diagonal(3,3,2,0,{}),std::invalid_argument);
}

TEST_CASE("GN diagonal refuses malformed nonfinite and unrepresentable VJPs", "[training][gn-diagonal]") {
    const int mode = GENERATE(0,1,2,3,4);
    REQUIRE_THROWS(Training::estimate_gn_diagonal(2,2,3,0,
        [&](const Eigen::VectorXd &, Eigen::VectorXd &out) {
            if (mode == 0) out.resize(1);
            if (mode == 1) out[0] = 1.; // Missing second coefficient.
            if (mode == 2) out.setConstant(std::numeric_limits<double>::infinity());
            if (mode == 3) out.setConstant(1e200); // The GN diagonal overflows.
            if (mode == 4) out.setConstant(1e-200); // A nonzero diagonal underflows.
        }));
}

TEST_CASE("GN diagonal avoids overflow of an unnormalized square sum", "[training][gn-diagonal]") {
    const double large = std::sqrt(std::numeric_limits<double>::max() / 8.);
    const auto diagonal = Training::estimate_gn_diagonal(1,1,32,0,
        [&](const Eigen::VectorXd &z, Eigen::VectorXd &out) { out[0] = large * z[0]; });
    REQUIRE(std::isfinite(diagonal[0]));
    REQUIRE(diagonal[0] == Approx(2. * large * large).epsilon(1e-13));
}
