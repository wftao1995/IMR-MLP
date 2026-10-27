#include "lower_energy_envelope.h"
#include "libalglib/optimization.h"
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>

namespace {
constexpr double domain_tolerance = 1e-8;
constexpr double solution_tolerance = 1e-7;

void finite_coordinates(const LowerEnergyEnvelope::Coordinates &point, std::size_t dimension) {
    if (point.size() != dimension)
        throw std::invalid_argument("Lower energy envelope coordinate dimension mismatch");
    for (double value : point)
        if (!std::isfinite(value))
            throw std::invalid_argument("Lower energy envelope coordinates must be finite");
}

[[noreturn]] void outside_domain() {
    throw std::out_of_range("Query is outside the reference convex coordinate domain");
}
}

struct LowerEnergyEnvelope::Impl {
    Coordinates origin;
    std::vector<long double> scales;
    Eigen::MatrixXd scaled_points, basis;
    Eigen::VectorXd scaled_energies, reference_energies;
    double energy_origin = 0;
    long double energy_scale = 1;
    alglib::minlpstate solver;
    alglib::real_1d_array cost, solution;
    alglib::minlpreport report;
    std::map<Coordinates, double> cache;

    Eigen::VectorXd scaled(const Coordinates &point) const {
        Eigen::VectorXd result(point.size());
        for (std::size_t k = 0; k < point.size(); ++k)
            result[k] = (static_cast<long double>(point[k]) - origin[k]) / scales[k];
        return result;
    }
};

LowerEnergyEnvelope::LowerEnergyEnvelope(const std::vector<Coordinates> &coordinates,
                                       const std::vector<double> &energies)
    : impl_(std::make_unique<Impl>()) {
    if (coordinates.empty() || coordinates.size() != energies.size())
        throw std::invalid_argument("Lower energy envelope requires matching nonempty references and energies");
    const auto dimension = coordinates.front().size();
    std::map<Coordinates, double> unique;
    for (std::size_t j = 0; j < coordinates.size(); ++j) {
        finite_coordinates(coordinates[j], dimension);
        if (!std::isfinite(energies[j]))
            throw std::invalid_argument("Lower energy envelope reference energies must be finite");
        auto [it, inserted] = unique.emplace(coordinates[j], energies[j]);
        if (!inserted) it->second = std::min(it->second, energies[j]);
    }
    auto &p = *impl_;
    p.origin = unique.begin()->first;
    p.scales.assign(dimension, 0);
    p.energy_origin = std::numeric_limits<double>::infinity();
    for (const auto &[point, energy] : unique) {
        p.energy_origin = std::min(p.energy_origin, energy);
        for (std::size_t k = 0; k < dimension; ++k)
            p.scales[k] = std::max(p.scales[k], std::fabs(static_cast<long double>(point[k]) - p.origin[k]));
    }
    for (std::size_t k = 0; k < dimension; ++k)
        if (p.scales[k] == 0) p.scales[k] = std::max(1.0L, std::fabs(static_cast<long double>(p.origin[k])));
    for (const auto &[point, energy] : unique)
        p.energy_scale = std::max(p.energy_scale, static_cast<long double>(energy) - p.energy_origin);
    p.scaled_points.resize(unique.size(), dimension);
    p.scaled_energies.resize(unique.size());
    p.reference_energies.resize(unique.size());
    std::size_t row = 0;
    for (const auto &[point, energy] : unique) {
        p.scaled_points.row(row) = p.scaled(point).transpose();
        p.reference_energies[row] = energy;
        p.scaled_energies[row++] = (static_cast<long double>(energy) - p.energy_origin) / p.energy_scale;
    }
    if (dimension == 0) {
        p.basis.resize(0, 0);
    } else {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(p.scaled_points, Eigen::ComputeThinV);
        // Coordinates were scaled independently; resolve affine dimension
        // here rather than passing redundant directions to the LP solver.
        svd.setThreshold(1e-10);
        p.basis = svd.matrixV().leftCols(svd.rank());
    }
    const int variables = p.basis.cols() + 1;
    alglib::minlpcreate(variables, p.solver);
    const double inf = std::numeric_limits<double>::infinity();
    alglib::minlpsetbcall(p.solver, -inf, inf);
    alglib::minlpsetalgodss(p.solver, 1e-9);
    alglib::real_2d_array constraints;
    alglib::real_1d_array lower, upper;
    constraints.setlength(unique.size(), variables);
    lower.setlength(unique.size());
    upper.setlength(unique.size());
    const Eigen::MatrixXd projected = p.scaled_points * p.basis;
    for (std::size_t j = 0; j < unique.size(); ++j) {
        constraints[j][0] = 1;
        for (int k = 1; k < variables; ++k) constraints[j][k] = projected(j, k - 1);
        lower[j] = -inf;
        upper[j] = p.scaled_energies[j];
    }
    alglib::minlpsetlc2dense(p.solver, constraints, lower, upper);
    p.cost.setlength(variables);
}

LowerEnergyEnvelope::~LowerEnergyEnvelope() = default;

double LowerEnergyEnvelope::energy_at(const Coordinates &coordinates) {
    auto &p = *impl_;
    finite_coordinates(coordinates, p.origin.size());
    if (const auto it = p.cache.find(coordinates); it != p.cache.end()) return it->second;
    const Eigen::VectorXd scaled = p.scaled(coordinates);
    if (!scaled.allFinite()) outside_domain();
    const Eigen::VectorXd projected = p.basis.transpose() * scaled;
    const Eigen::VectorXd residual = scaled - p.basis * projected;
    if (residual.size() && residual.cwiseAbs().maxCoeff() > domain_tolerance) outside_domain();
    // Dual: maximize a+b.x subject to a+b.x_j <= e_j. Only the
    // objective changes between queries; ALGLIB still starts a new solve.
    p.cost[0] = -1;
    for (int k = 0; k < projected.size(); ++k) p.cost[k + 1] = -projected[k];
    try {
        alglib::minlpsetcost(p.solver, p.cost);
        alglib::minlpoptimize(p.solver);
        alglib::minlpresultsbuf(p.solver, p.solution, p.report);
    } catch (const alglib::ap_error &error) {
        throw std::runtime_error("Lower energy envelope LP failed: " + error.msg);
    }
    // ALGLIB's presolver can report -2 (infeasible or unbounded), even
    // for DSS. This dual is always feasible: b=0, a=min(e_j). Thus its
    // infeasible/unbounded result can only mean an unbounded query.
    if (p.report.terminationtype == -4 || p.report.terminationtype == -2) outside_domain();
    if (p.report.terminationtype < 1 || p.report.terminationtype > 4)
        throw std::runtime_error("Lower energy envelope LP failed with status " + std::to_string(p.report.terminationtype));
    // LP multipliers are the primal mixture weights. Check them in every
    // original coordinate, including directions removed by rank reduction.
    Eigen::VectorXd weights(p.scaled_points.rows());
    for (int j = 0; j < weights.size(); ++j) weights[j] = p.report.laglc[j];
    if (!weights.allFinite() || weights.minCoeff() < -solution_tolerance ||
        std::fabs(weights.sum() - 1) > solution_tolerance)
        throw std::runtime_error("Lower energy envelope LP has invalid mixture weights");
    const Eigen::VectorXd mixing_error = p.scaled_points.transpose() * weights - scaled;
    if (mixing_error.size() && mixing_error.cwiseAbs().maxCoeff() > solution_tolerance)
        outside_domain();
    const double mixture_energy = weights.dot(p.scaled_energies);
    if (!std::isfinite(p.report.f) || std::fabs(mixture_energy + p.report.f) > solution_tolerance ||
        !std::isfinite(p.report.primalerror) || p.report.primalerror > solution_tolerance)
        throw std::runtime_error("Lower energy envelope LP residual exceeds tolerance");
    // Return the feasible primal mixture in the original energy units. Undoing
    // the solver's energy shift/scale can lose an ulp even at an exact reference
    // endpoint, creating a spurious positive energy excess there. Long-double
    // accumulation and normalization also preserve a single reference energy
    // when the solver's weight sum differs slightly from one.
    long double raw_energy = 0, weight_sum = 0;
    for (int j = 0; j < weights.size(); ++j) {
        raw_energy += static_cast<long double>(weights[j]) * p.reference_energies[j];
        weight_sum += static_cast<long double>(weights[j]);
    }
    const double result = raw_energy / weight_sum;
    if (!std::isfinite(result)) throw std::runtime_error("Lower energy envelope result must be finite");
    p.cache.emplace(coordinates, result);
    return result;
}

std::size_t LowerEnergyEnvelope::affine_rank() const { return impl_->basis.cols(); }
std::size_t LowerEnergyEnvelope::reference_size() const { return impl_->scaled_points.rows(); }
