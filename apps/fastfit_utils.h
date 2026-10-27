// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_APPS_FASTFIT_UTILS_H
#define IMRMLP_APPS_FASTFIT_UTILS_H

#include <Eigen/Dense>
#include "train/efs_factorization_observer.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace func_fastfit::detail {

// C C^+ M (C C^+)^T = Q (Q^T M Q) Q^T. Working with an orthonormal
// range avoids multiplying two large pseudoinverses on ill-conditioned CUR
// columns. COD retains its existing numerical-rank rule.
inline double StableCURResidual(const Eigen::MatrixXd &M,const Eigen::MatrixXd &C,int &rank) {
    Eigen::CompleteOrthogonalDecomposition<Eigen::MatrixXd> cod(C);
    Training::count_efs_factorization(Training::EFSFactorization::cur_range_qr);
    rank=cod.rank();
    Eigen::MatrixXd Q=cod.matrixQ()*Eigen::MatrixXd::Identity(C.rows(),rank);
    Eigen::MatrixXd middle=Q.transpose()*M*Q;
    return (M-Q*middle*Q.transpose()).norm();
}
inline double CURRoundoffTolerance(const Eigen::MatrixXd &M) {
    return 128*std::numeric_limits<double>::epsilon()*M.rows()*M.norm();
}

inline void BuildReducedSystem(const Eigen::MatrixXd &mat,
                               const Eigen::VectorXd &vec,
                               const std::vector<int> &indices,
                               Eigen::MatrixXd &reduced_mat,
                               Eigen::VectorXd &reduced_vec) {
    if (mat.rows() != mat.cols() || vec.size() != mat.rows()) {
        throw std::invalid_argument("normal-equation matrix/vector dimensions do not match");
    }
    std::vector<bool> seen(static_cast<std::size_t>(mat.rows()), false);
    for (int idx : indices) {
        if (idx < 0 || idx >= mat.rows()) throw std::out_of_range("reduced-system index out of range");
        if (seen[static_cast<std::size_t>(idx)]) {
            throw std::invalid_argument("reduced-system indices must be unique");
        }
        seen[static_cast<std::size_t>(idx)] = true;
    }

    const Eigen::Index size = static_cast<Eigen::Index>(indices.size());
    reduced_mat.resize(size, size);
    reduced_vec.resize(size);
    for (Eigen::Index i = 0; i < size; ++i) {
        reduced_vec(i) = vec(indices[static_cast<std::size_t>(i)]);
        for (Eigen::Index j = 0; j < size; ++j) {
            reduced_mat(i, j) = mat(indices[static_cast<std::size_t>(i)],
                                    indices[static_cast<std::size_t>(j)]);
        }
    }
}

inline void RemoveOneBasis(const Eigen::MatrixXd &mat,
                           const Eigen::VectorXd &vec,
                           int removed,
                           Eigen::MatrixXd &reduced_mat,
                           Eigen::VectorXd &reduced_vec) {
    if (removed < 0 || removed >= mat.rows()) throw std::out_of_range("basis index out of range");
    std::vector<int> keep;
    keep.reserve(static_cast<std::size_t>(mat.rows() - 1));
    for (int i = 0; i < mat.rows(); ++i) if (i != removed) keep.push_back(i);
    BuildReducedSystem(mat, vec, keep, reduced_mat, reduced_vec);
}

inline std::set<int> ReadScalarIds(const std::string &filename, int nscalar) {
    std::ifstream input(filename);
    if (!input) throw std::runtime_error("cannot open selection file: " + filename);
    long long count = -1;
    if (!(input >> count) || count < 0 || count > nscalar) {
        throw std::runtime_error("invalid selection count in " + filename);
    }
    std::set<int> ids;
    for (long long i = 0; i < count; ++i) {
        int id = -1;
        if (!(input >> id) || id < 0 || id >= nscalar || !ids.insert(id).second) {
            throw std::runtime_error("invalid, duplicate, or out-of-range scalar id in " + filename);
        }
    }
    std::string extra;
    if (input >> extra) throw std::runtime_error("unexpected trailing data in " + filename);
    return ids;
}

inline void ValidatePartition(const std::set<int> &selected,
                              const std::set<int> &removed,
                              int nscalar) {
    if (nscalar < 0) throw std::invalid_argument("scalar count must be non-negative");
    const auto invalid_id = [nscalar](int id) { return id < 0 || id >= nscalar; };
    if (std::any_of(selected.begin(), selected.end(), invalid_id) ||
        std::any_of(removed.begin(), removed.end(), invalid_id)) {
        throw std::runtime_error("selected or removed scalar id is out of range");
    }
    for (int id = 0; id < nscalar; ++id) {
        if (selected.count(id) + removed.count(id) != 1) {
            throw std::runtime_error("selected and removed ids must form an exact partition");
        }
    }
}

inline void WriteScalarIds(const std::string &filename, const std::set<int> &ids) {
    std::ofstream output(filename);
    if (!output) throw std::runtime_error("cannot write selection file: " + filename);
    output << ids.size();
    for (int id : ids) output << ' ' << id;
    output << '\n';
    if (!output) throw std::runtime_error("failed while writing selection file: " + filename);
}

inline std::vector<int> AllocateSampleQuotas(const std::vector<int> &local_sites,
                                              int max_samples) {
    if (max_samples <= 0) throw std::invalid_argument("corr-max-samples must be positive");
    long long total = 0;
    for (int count : local_sites) {
        if (count < 0) throw std::invalid_argument("site count must be non-negative");
        total += count;
    }
    std::vector<int> quotas(local_sites.size(), 0);
    if (total <= max_samples) return local_sites;

    struct Fraction { std::size_t rank; double value; };
    std::vector<Fraction> fractions;
    int assigned = 0;
    for (std::size_t rank = 0; rank < local_sites.size(); ++rank) {
        const double exact = static_cast<double>(max_samples) * local_sites[rank] /
                             static_cast<double>(total);
        quotas[rank] = std::min(local_sites[rank], static_cast<int>(std::floor(exact)));
        assigned += quotas[rank];
        fractions.push_back({rank, exact - std::floor(exact)});
    }
    std::stable_sort(fractions.begin(), fractions.end(), [](const Fraction &a, const Fraction &b) {
        return a.value > b.value;
    });
    for (const Fraction &fraction : fractions) {
        if (assigned == max_samples) break;
        if (quotas[fraction.rank] < local_sites[fraction.rank]) {
            ++quotas[fraction.rank];
            ++assigned;
        }
    }
    return quotas;
}

inline int SelectKneePoint(const std::vector<std::pair<double, double>> &objectives) {
    if (objectives.empty()) throw std::invalid_argument("empty Pareto front");
    double min0 = objectives.front().first, max0 = min0;
    double min1 = objectives.front().second, max1 = min1;
    for (const auto &[o0, o1] : objectives) {
        min0 = std::min(min0, o0); max0 = std::max(max0, o0);
        min1 = std::min(min1, o1); max1 = std::max(max1, o1);
    }
    const double range0 = max0 > min0 ? max0 - min0 : 1.0;
    const double range1 = max1 > min1 ? max1 - min1 : 1.0;
    int best = 0;
    double best_distance = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < objectives.size(); ++i) {
        const double n0 = (objectives[i].first - min0) / range0;
        const double n1 = (objectives[i].second - min1) / range1;
        const double distance = std::hypot(n0, n1);
        if (distance < best_distance) {
            best_distance = distance;
            best = static_cast<int>(i);
        }
    }
    return best;
}

inline std::vector<double> CrowdingDistances(
    const std::vector<std::pair<double, double>> &objectives) {
    const std::size_t size = objectives.size();
    std::vector<double> distances(size, 0.0);
    if (size <= 2) {
        std::fill(distances.begin(), distances.end(), std::numeric_limits<double>::infinity());
        return distances;
    }
    for (int objective = 0; objective < 2; ++objective) {
        std::vector<std::size_t> order(size);
        for (std::size_t i = 0; i < size; ++i) order[i] = i;
        std::stable_sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
            const double left = objective == 0 ? objectives[lhs].first : objectives[lhs].second;
            const double right = objective == 0 ? objectives[rhs].first : objectives[rhs].second;
            return left < right;
        });
        const auto value = [&](std::size_t index) {
            return objective == 0 ? objectives[index].first : objectives[index].second;
        };
        const double range = value(order.back()) - value(order.front());
        distances[order.front()] = std::numeric_limits<double>::infinity();
        distances[order.back()] = std::numeric_limits<double>::infinity();
        if (range <= 0.0) continue;
        for (std::size_t i = 1; i + 1 < size; ++i) {
            if (!std::isinf(distances[order[i]])) {
                distances[order[i]] +=
                    (value(order[i + 1]) - value(order[i - 1])) / range;
            }
        }
    }
    return distances;
}

} // namespace func_fastfit::detail

#endif
