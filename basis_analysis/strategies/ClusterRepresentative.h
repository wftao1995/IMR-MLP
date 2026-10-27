// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_BASIS_ANALYSIS_STRATEGIES_CLUSTER_REPRESENTATIVE_H
#define IMRMLP_BASIS_ANALYSIS_STRATEGIES_CLUSTER_REPRESENTATIVE_H

#include "../Decorrelator.h"

namespace BasisAnalysis {

/// Cluster features by correlation distance (1 - |pearson|), then pick one
/// representative per cluster.
///
/// Uses a greedy clustering approach: build connected components in the graph
/// where edges exist when |pearson| > threshold. For each cluster, pick the
/// feature with the lowest mean absolute correlation to other cluster members
/// (i.e., the most "central" feature).
class ClusterRepresentative : public Decorrelator {
public:
    /// @param correlation_threshold  Edge threshold for clustering (default 0.80)
    /// @param min_cluster_size        Clusters smaller than this are kept intact
    ClusterRepresentative(double correlation_threshold = 0.80,
                          int min_cluster_size = 2);

    std::string name() const override;
    std::pair<std::set<int>, std::set<int>>
    select(const CorrelationResult &corr,
           int nscalar) override;

private:
    double correlation_threshold_;
    int min_cluster_size_;

    /// Build connected components from the correlation graph.
    static std::vector<std::vector<int>>
    find_clusters(const Eigen::MatrixXd &pearson, double threshold, int n);

    /// Select the most representative feature in a cluster.
    /// Representative = lowest mean |correlation| to other members.
    static int pick_representative(const std::vector<int> &cluster,
                                   const Eigen::MatrixXd &pearson);
};

} // namespace BasisAnalysis

#endif
