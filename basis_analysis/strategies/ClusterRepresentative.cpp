// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "ClusterRepresentative.h"
#include <cmath>
#include <queue>
#include <limits>

namespace BasisAnalysis {

ClusterRepresentative::ClusterRepresentative(double correlation_threshold,
                                             int min_cluster_size)
    : correlation_threshold_(correlation_threshold),
      min_cluster_size_(min_cluster_size) {}

std::string ClusterRepresentative::name() const {
    return "ClusterRepresentative(threshold=" + std::to_string(correlation_threshold_) +
           ", min_size=" + std::to_string(min_cluster_size_) + ")";
}

std::vector<std::vector<int>>
ClusterRepresentative::find_clusters(const Eigen::MatrixXd &pearson,
                                      double threshold, int n) {
    // Build adjacency list
    std::vector<std::vector<int>> adj(n);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (std::abs(pearson(i, j)) > threshold) {
                adj[i].push_back(j);
                adj[j].push_back(i);
            }
        }
    }

    // BFS to find connected components
    std::vector<bool> visited(n, false);
    std::vector<std::vector<int>> clusters;
    for (int i = 0; i < n; ++i) {
        if (visited[i]) continue;
        std::vector<int> cluster;
        std::queue<int> q;
        q.push(i);
        visited[i] = true;
        while (!q.empty()) {
            int v = q.front(); q.pop();
            cluster.push_back(v);
            for (int u : adj[v]) {
                if (!visited[u]) {
                    visited[u] = true;
                    q.push(u);
                }
            }
        }
        clusters.push_back(std::move(cluster));
    }
    return clusters;
}

int ClusterRepresentative::pick_representative(const std::vector<int> &cluster,
                                                const Eigen::MatrixXd &pearson) {
    if (cluster.size() == 1) return cluster[0];

    int best_idx = cluster[0];
    double best_score = std::numeric_limits<double>::max();

    for (int idx : cluster) {
        double mean_abs_corr = 0.0;
        for (int other : cluster) {
            if (idx == other) continue;
            mean_abs_corr += std::abs(pearson(idx, other));
        }
        mean_abs_corr /= (cluster.size() - 1);
        if (mean_abs_corr < best_score) {
            best_score = mean_abs_corr;
            best_idx = idx;
        }
    }
    return best_idx;
}

std::pair<std::set<int>, std::set<int>>
ClusterRepresentative::select(const CorrelationResult &corr,
                               int nscalar) {
    std::set<int> selected;
    std::set<int> removed;

    auto clusters = find_clusters(corr.pearson, correlation_threshold_, nscalar);

    for (const auto &cluster : clusters) {
        if (cluster.size() == 1 && corr.pearson(cluster[0], cluster[0]) == 0.) {
            removed.insert(cluster[0]);
            continue;
        }
        if (static_cast<int>(cluster.size()) < min_cluster_size_) {
            // Keep all members of very small clusters
            for (int idx : cluster) {
                selected.insert(idx);
            }
        } else {
            // Pick one representative, remove the rest
            int rep = pick_representative(cluster, corr.pearson);
            selected.insert(rep);
            for (int idx : cluster) {
                if (idx != rep) removed.insert(idx);
            }
        }
    }
    return {selected, removed};
}

} // namespace BasisAnalysis
