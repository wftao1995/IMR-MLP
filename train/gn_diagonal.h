#pragma once

#include <Eigen/Core>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>

namespace Training {
using ResidualVJP = std::function<void(const Eigen::VectorXd &, Eigen::VectorXd &)>;

// For a local weighted residual r(theta), estimate diag(2 J^T J) using
// E[(J^T z)_i^2] = (J^T J)_ii, with independent Rademacher residual probes.
// The callback must overwrite every output coefficient with J^T z, where J
// differentiates sqrt(weight / global_configuration_count) * (prediction-target).
// Supply no loss-residual factor, extra factor 2, penalty, or parameter scaling.
//
// MPI callers use a distinct seed per rank and SUM the returned diagonals:
// squaring a globally reduced VJP would introduce unwanted cross-rank terms.
// This function never builds J. It stores O(residuals + parameters) values and
// invokes vjp exactly probes times unless either dimension is zero. Exact zero
// columns remain zero; damping and handling unobserved parameters belong to the
// caller. A finite-probe estimate can also be zero through probe cancellation.
inline Eigen::VectorXd estimate_gn_diagonal(Eigen::Index residuals,
    Eigen::Index parameters, int probes, std::uint64_t seed,
    const ResidualVJP &vjp) {
    if (residuals < 0 || parameters < 0 || probes <= 0)
        throw std::invalid_argument("GN diagonal requires nonnegative dimensions and positive probe count");
    Eigen::VectorXd norm = Eigen::VectorXd::Zero(parameters);
    if (residuals == 0 || parameters == 0) return norm;
    if (!vjp) throw std::invalid_argument("GN diagonal requires a residual VJP callback");

    std::mt19937_64 random(seed);
    Eigen::VectorXd z(residuals), product(parameters);
    const double inverse_root_probes = 1. / std::sqrt(double(probes));
    for (int sample = 0; sample < probes; ++sample) {
        for (Eigen::Index row = 0; row < residuals; ++row)
            z[row] = (random() & 1) ? 1. : -1.;
        // Detect callbacks which only accumulate, omit an entry, or resize to
        // the wrong parameter count instead of returning a complete VJP.
        product.setConstant(parameters, std::numeric_limits<double>::quiet_NaN());
        vjp(z, product);
        if (product.size() != parameters || !product.allFinite())
            throw std::runtime_error("GN residual VJP must return one finite value per parameter");
        // hypot avoids overflow in a sum of squares when its mean is finite.
        for (Eigen::Index column = 0; column < parameters; ++column) {
            const double scaled = product[column] * inverse_root_probes;
            if (product[column] != 0. && scaled == 0.)
                throw std::overflow_error("GN diagonal probe normalization underflowed");
            norm[column] = std::hypot(norm[column], scaled);
        }
    }
    for (Eigen::Index column = 0; column < parameters; ++column) {
        const double value = norm[column] * norm[column] * 2.;
        if (!std::isfinite(value) || (norm[column] != 0. && value == 0.))
            throw std::overflow_error("GN diagonal is outside the finite nonzero double range");
        norm[column] = value;
    }
    return norm;
}
} // namespace Training
