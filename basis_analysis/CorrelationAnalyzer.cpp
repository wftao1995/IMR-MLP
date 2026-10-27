// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "CorrelationAnalyzer.h"
#include <Eigen/Eigenvalues>
#include "FactorizationObserver.h"
#include <cmath>
#include <limits>
#include <algorithm>
#include <stdexcept>

namespace BasisAnalysis {
namespace { thread_local FactorizationCounts *factorization_counts = nullptr; }
void observe_factorizations(FactorizationCounts *counts) { factorization_counts = counts; }
void count_eigen_factorization() { if (factorization_counts) ++factorization_counts->eigen; }
void count_ldlt_factorization() { if (factorization_counts) ++factorization_counts->ldlt; }


// ── CorrelationResult ──────────────────────────────────────────────────

int CorrelationResult::rank(double eps) const {
    if (eigenvalues.size() == 0) return 0;
    double threshold = eps * eigenvalues[0];
    int r = 0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (eigenvalues[i] > threshold) ++r;
    }
    return r;
}

std::vector<std::pair<int, int>>
CorrelationResult::highly_correlated_pairs(double threshold) const {
    std::vector<std::pair<int, int>> pairs;
    for (int i = 0; i < nfeatures; ++i) {
        for (int j = i + 1; j < nfeatures; ++j) {
            if (std::abs(pearson(i, j)) > threshold) {
                pairs.emplace_back(i, j);
            }
        }
    }
    return pairs;
}

// ── CorrelationAnalyzer ────────────────────────────────────────────────

CorrelationAnalyzer::CorrelationAnalyzer(const Eigen::MatrixXd &descriptors)
    : desc_(descriptors) {}

namespace {
using Solver = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>;
void validate_correlation(const Eigen::MatrixXd &corr) {
    if (corr.rows() != corr.cols() || !corr.allFinite() || !corr.isApprox(corr.transpose(), 1e-12))
        throw std::invalid_argument("Correlation matrix must be square, finite and symmetric");
}
Eigen::VectorXd vif_from_spectrum(const Solver &es) {
    const auto &ev = es.eigenvalues();
    const auto &vectors = es.eigenvectors();
    const int n = ev.size();
    Eigen::VectorXd vif(n);
    if (!n) return vif;
    const double cutoff = std::max(1e-12, std::numeric_limits<double>::epsilon() * n) * std::max(0., ev[n-1]);
    if (ev[0] < -std::max(1e-14, cutoff))
        throw std::invalid_argument("Correlation matrix is not positive semidefinite");
    const int nullity = (ev.array() <= cutoff).count();
    const double projection_tolerance = 64 * std::numeric_limits<double>::epsilon() * n;
    for (int i = 0; i < n; ++i) {
        // The projection norm is invariant under rotations of a degenerate
        // null-space basis, unlike a threshold on an individual eigenvector.
        if (vectors.row(i).head(nullity).squaredNorm() > projection_tolerance) {
            vif[i] = std::numeric_limits<double>::infinity();
        } else {
            double sum = 0.;
            for (int j = nullity; j < n; ++j) sum += vectors(i,j) * vectors(i,j) / ev[j];
            vif[i] = std::max(1., sum);
        }
    }
    return vif;
}
double condition_from_spectrum(const Eigen::VectorXd &ev) {
    if (ev.size() == 0) return std::numeric_limits<double>::infinity();
    const double cutoff = std::max(1e-12, std::numeric_limits<double>::epsilon() * ev.size()) * ev[ev.size()-1];
    return ev[0] <= cutoff ? std::numeric_limits<double>::infinity() : ev[ev.size()-1] / ev[0];
}
}

CorrelationResult CorrelationAnalyzer::compute_all() {
    CorrelationResult result;
    result.nfeatures = desc_.rows();
    result.nsamples = desc_.cols();
    result.pearson = compute_pearson(desc_);
    if (!result.nfeatures) return result;
    count_eigen_factorization();
    Solver es(result.pearson);
    if (es.info() != Eigen::Success) throw std::runtime_error("Correlation eigensolver failed");
    result.vif = vif_from_spectrum(es);
    result.cond_number = condition_from_spectrum(es.eigenvalues());
    result.eigenvalues = es.eigenvalues().reverse();
    result.eigenvectors = es.eigenvectors().rowwise().reverse();
    return result;
}

Eigen::MatrixXd CorrelationAnalyzer::compute_pearson(const Eigen::MatrixXd &desc) {
    if (!desc.allFinite()) throw std::invalid_argument("Descriptor matrix contains non-finite values");
    const int nfeat = desc.rows(), nsamp = desc.cols();
    if (nsamp < 2) return Eigen::MatrixXd::Zero(nfeat, nfeat);
    Eigen::MatrixXd centered = desc;
    std::vector<bool> varying(nfeat, false);
    for (int i = 0; i < nfeat; ++i) {
        centered.row(i).array() -= desc.row(i).mean();
        const double norm = centered.row(i).stableNorm();
        const double tolerance = 8 * std::numeric_limits<double>::epsilon() * std::sqrt(double(nsamp)) * desc.row(i).cwiseAbs().maxCoeff();
        if (norm > tolerance) { centered.row(i) /= norm; varying[i] = true; }
        else centered.row(i).setZero();
    }
    Eigen::MatrixXd pearson = centered * centered.transpose();
    for (int i = 0; i < nfeat; ++i) {
        // Zero-variance features have no estimable Pearson correlation.
        pearson(i,i) = varying[i] ? 1. : 0.;
        for (int j = i+1; j < nfeat; ++j)
            pearson(j,i) = pearson(i,j) = std::clamp(pearson(i,j), -1., 1.);
    }
    return pearson;
}

Eigen::VectorXd CorrelationAnalyzer::compute_vif(const Eigen::MatrixXd &corr) {
    validate_correlation(corr);
    if (!corr.rows()) return Eigen::VectorXd();
    count_eigen_factorization();
    Solver es(corr);
    if (es.info() != Eigen::Success) throw std::runtime_error("VIF eigensolver failed");
    return vif_from_spectrum(es);
}

double CorrelationAnalyzer::compute_condition_number(const Eigen::MatrixXd &corr) {
    validate_correlation(corr);
    if (!corr.rows()) return std::numeric_limits<double>::infinity();
    count_eigen_factorization();
    Solver es(corr, Eigen::EigenvaluesOnly);
    if (es.info() != Eigen::Success) throw std::runtime_error("Correlation eigensolver failed");
    return condition_from_spectrum(es.eigenvalues());
}

void CorrelationAnalyzer::compute_eigensystem(const Eigen::MatrixXd &corr,
                                             Eigen::VectorXd &values, Eigen::MatrixXd &vectors) {
    validate_correlation(corr);
    if (!corr.rows()) { values.resize(0); vectors.resize(0,0); return; }
    count_eigen_factorization();
    Solver es(corr);
    if (es.info() != Eigen::Success) throw std::runtime_error("Correlation eigensolver failed");
    values = es.eigenvalues().reverse();
    vectors = es.eigenvectors().rowwise().reverse();
}

} // namespace BasisAnalysis
