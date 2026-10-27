// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_CORRELATION_ANALYZER_H
#define IMRMLP_BASIS_ANALYSIS_CORRELATION_ANALYZER_H

#include <Eigen/Dense>
#include <vector>
#include <utility>

namespace BasisAnalysis {

/// Result container for a full correlation analysis of a descriptor matrix.
struct CorrelationResult {
    Eigen::MatrixXd pearson;       // (nscalar x nscalar) Pearson correlation coefficients
    Eigen::VectorXd vif;           // (nscalar) Variance Inflation Factors
    double cond_number = 0.0;      // Condition number of the correlation matrix
    Eigen::VectorXd eigenvalues;   // Eigenvalues of the correlation matrix (descending)
    Eigen::MatrixXd eigenvectors;  // Corresponding eigenvectors (columns)
    int nsamples = 0;              // Number of samples (atoms) used
    int nfeatures = 0;             // Number of scalar moments

    /// Numerical rank: count of eigenvalues > machine-epsilon * max_eig
    int rank(double eps = 1e-14) const;

    /// Return pairs (i, j) where |pearson(i,j)| > threshold and i < j.
    std::vector<std::pair<int, int>> highly_correlated_pairs(double threshold = 0.95) const;
};

/// Stateless analysis of a descriptor matrix: Pearson correlation, VIF,
/// condition number, and eigenvalue spectrum.
///
/// The input is a (nfeatures x nsamples) matrix where each column is one
/// atom's raw scalar-moment descriptor vector.
class CorrelationAnalyzer {
public:
    /// Construct with a raw descriptor matrix (nfeatures rows x nsamples cols).
    /// Borrows the matrix; it must outlive this analyzer.
    explicit CorrelationAnalyzer(const Eigen::MatrixXd &descriptors);

    /// Run all analyses and populate `result`.
    CorrelationResult compute_all();

    /// Compute the Pearson correlation matrix.
    /// Handles zero-variance columns gracefully (returns 0 corr for those).
    static Eigen::MatrixXd compute_pearson(const Eigen::MatrixXd &desc);

    /// Compute VIF from a correlation matrix: vif[i] = diag(inv(corr))[i].
    /// Features participating in the numerical null space have infinite VIF.
    static Eigen::VectorXd compute_vif(const Eigen::MatrixXd &corr);

    /// Compute condition number = max_eigenvalue / min_eigenvalue of the
    /// correlation matrix. Returns INF if the matrix is numerically singular.
    static double compute_condition_number(const Eigen::MatrixXd &corr);

    /// Compute full eigenvalue decomposition of the correlation matrix.
    /// eigenvalues are sorted descending.
    static void compute_eigensystem(const Eigen::MatrixXd &corr,
                                    Eigen::VectorXd &eigenvalues,
                                    Eigen::MatrixXd &eigenvectors);

private:
    const Eigen::MatrixXd &desc_;
};

} // namespace BasisAnalysis

#endif // IMRMLP_BASIS_ANALYSIS_CORRELATION_ANALYZER_H
