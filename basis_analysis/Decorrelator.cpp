// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "Decorrelator.h"
#include "FactorizationObserver.h"
#include <cmath>
#include <stdexcept>

namespace BasisAnalysis {
void prune_vif(const Eigen::MatrixXd &correlation, std::set<int> &selected,
               std::set<int> &removed, double target, int max_features) {
    if (!std::isfinite(target) || target < 1 || max_features < 0)
        throw std::invalid_argument("Invalid VIF target or feature limit");
    Eigen::MatrixXd inverse;
    int updates = 0;
    for (auto it = selected.begin(); it != selected.end();) {
        if (correlation(*it,*it) == 0.) { removed.insert(*it); it = selected.erase(it); }
        else ++it;
    }
    while (selected.size() > 1) {
        const std::vector<int> ids(selected.begin(), selected.end());
        const int n = ids.size();
        auto submatrix = [&] {
            Eigen::MatrixXd sub(n,n);
            for (int j = 0; j < n; ++j) for (int i = 0; i < n; ++i)
                sub(i,j) = correlation(ids[i],ids[j]);
            return sub;
        };
        Eigen::VectorXd vif;
        if (inverse.rows() != n || updates >= 32) {
            const auto sub = submatrix();
            vif = CorrelationAnalyzer::compute_vif(sub);
            inverse.resize(0,0); updates = 0;
            if (vif.allFinite()) {
                count_ldlt_factorization();
                Eigen::LDLT<Eigen::MatrixXd> ldlt(sub);
                if (ldlt.info() == Eigen::Success && ldlt.isPositive()) {
                    inverse = ldlt.solve(Eigen::MatrixXd::Identity(n,n));
                    if (!inverse.allFinite() || !inverse.diagonal().isApprox(vif, 1e-8)) inverse.resize(0,0);
                }
            }
        } else vif = inverse.diagonal();
        Eigen::Index worst;
        double maximum = vif.maxCoeff(&worst); // +infinity is the worst, not ignored.
        const bool count_ok = max_features == 0 || n <= max_features;
        if (maximum <= target && count_ok) {
            // Verify the stopping condition independently after inverse updates.
            vif = CorrelationAnalyzer::compute_vif(submatrix());
            maximum = vif.maxCoeff(&worst);
            if (maximum <= target) break;
            inverse.resize(0,0);
        }
        removed.insert(ids[worst]); selected.erase(ids[worst]);
        if (inverse.rows() == n) {
            Eigen::MatrixXd next(n-1,n-1);
            for (int j=0,jj=0;j<n;++j) {
                if (j==worst) continue;
                for (int i=0,ii=0;i<n;++i) {
                    if (i==worst) continue;
                    next(ii++,jj)=inverse(i,j)-inverse(i,worst)*inverse(worst,j)/inverse(worst,worst);
                }
                ++jj;
            }
            inverse = std::move(next); ++updates;
        }
    }
}
} // namespace BasisAnalysis
