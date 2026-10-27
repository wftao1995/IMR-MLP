// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/fps_selector.h"
#include "base/structures.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <fmt/format.h>

FPSSelector::FPSSelector(BMatrixBuilder *builder)
    : builder_(builder), own_builder_(true)
{
    ncols_ = builder_->coeff_count();
}

FPSSelector::~FPSSelector()
{
    if (own_builder_) delete builder_;
}

// ── Candidate management ──────────────────────────────────────────────────────

void FPSSelector::add_candidates(std::vector<Atoms> &cfgs)
{
    if (!builder_) return;

    // Force structure-level rows for DescriptorBuilder (FPS needs 1 row/structure)
    auto *db = dynamic_cast<DescriptorBuilder *>(builder_);
    if (db && db->mode != DescriptorBuilder::Cfg) {
        if (verbose)
            std::cerr << "FPSSelector: forcing DescriptorBuilder to Cfg mode\n";
        db->mode = DescriptorBuilder::Cfg;
    }

    int n_existing = static_cast<int>(cfgs_.size());
    int n_new = static_cast<int>(cfgs.size());
    int new_rows = n_existing + n_new;

    D_.conservativeResize(new_rows, ncols_);

    for (int i = 0; i < n_new; ++i) {
        cfgs_.push_back(cfgs[i]);

        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_tmp;
        int nr = builder_->build_rows(cfgs_.back(), B_tmp, 1.0, 0.0);

        if (nr == 0) {
            D_.row(n_existing + i).setZero();
            if (verbose)
                std::cerr << "FPSSelector: cfg " << i << " produced 0 rows, using zeros\n";
        } else {
            // Use the first row (or sum if multiple — shouldn't happen in Cfg mode)
            D_.row(n_existing + i) = B_tmp.row(0);
            for (int r = 1; r < nr; ++r)
                D_.row(n_existing + i) += B_tmp.row(r);
        }
    }

    if (verbose)
        fmt::print(stderr, "FPSSelector: added {} candidates, total {} (dim={})\n",
                   n_new, cfgs_.size(), ncols_);
}

void FPSSelector::clear_candidates()
{
    D_.resize(0, ncols_);
    cfgs_.clear();
    selected_cfgs_.clear();
    selected_indices_.clear();
    selection_scores_.clear();
    selection_order_.clear();
}

int FPSSelector::n_candidates() const { return static_cast<int>(cfgs_.size()); }
int FPSSelector::feat_dim() const { return ncols_; }

// ── Distance helpers ──────────────────────────────────────────────────────────

double FPSSelector::compute_distance_vec(const Eigen::RowVectorXd &a,
                                          const Eigen::RowVectorXd &b) const
{
    if (metric == Cosine) {
        double na = a.norm(), nb = b.norm();
        if (na < 1e-30 || nb < 1e-30) return (a - b).norm();
        double cs = a.dot(b) / (na * nb);
        if (cs > 1.0) cs = 1.0;
        if (cs < -1.0) cs = -1.0;
        return std::sqrt(std::max(0.0, 2.0 - 2.0 * cs));
    }
    return (a - b).norm();
}

double FPSSelector::compute_distance(int i, int j) const
{
    return compute_distance_vec(D_.row(i), D_.row(j));
}

void FPSSelector::ensure_rng()
{
    if (!rng_seeded_) {
        rng_.seed(random_seed);
        rng_seeded_ = true;
    }
}

// ── Core FPS algorithm ────────────────────────────────────────────────────────

int FPSSelector::compute_fps_order(int first_idx, int max_k, double dist_thr,
                                    bool use_threshold)
{
    int n = static_cast<int>(D_.rows());
    if (n == 0) return 0;

    ensure_rng();

    // Choose initial point
    if (first_idx < 0 || first_idx >= n)
        first_idx = std::uniform_int_distribution<int>(0, n - 1)(rng_);

    selected_indices_.clear();
    selection_scores_.clear();
    selection_order_.clear();

    selected_indices_.push_back(first_idx);
    selection_scores_.push_back(std::numeric_limits<double>::infinity());
    selection_order_.push_back(0);

    // Track min distance from each candidate to the selected set
    std::vector<double> min_dist(n, std::numeric_limits<double>::infinity());
    for (int i = 0; i < n; ++i) {
        if (i == first_idx) { min_dist[i] = 0.0; continue; }
        min_dist[i] = compute_distance(i, first_idx);
    }

    if (verbose)
        fmt::print(stderr, "FPS: step 0 selected idx={} (n={})\n", first_idx, n);

    int actual_k = (max_k > 0) ? std::min(max_k, n) : n;

    for (int step = 1; step < actual_k; ++step) {
        // Find the candidate with maximum min-distance
        int best_idx = -1;
        double best_dist = -1.0;

        for (int i = 0; i < n; ++i) {
            if (min_dist[i] > best_dist) {
                best_dist = min_dist[i];
                best_idx = i;
            }
        }

        // Threshold-based early stop
        if (use_threshold && best_dist < dist_thr) {
            if (verbose)
                fmt::print(stderr, "FPS: step {} best_dist {:.6f} < threshold {:.6f}, stopping\n",
                           step, best_dist, dist_thr);
            break;
        }

        if (best_dist <= 0.0) {
            if (verbose)
                fmt::print(stderr, "FPS: step {} all remaining distances are 0, stopping\n", step);
            break;
        }

        selected_indices_.push_back(best_idx);
        selection_scores_.push_back(best_dist);
        selection_order_.push_back(step);
        min_dist[best_idx] = 0.0; // mark as selected

        if (verbose && step < 10)
            fmt::print(stderr, "FPS: step {} selected idx={} dist={:.6f}\n",
                       step, best_idx, best_dist);

        // Update min_dist for all remaining candidates
        for (int i = 0; i < n; ++i) {
            if (min_dist[i] <= 0.0) continue; // already selected
            double d = compute_distance(i, best_idx);
            if (d < min_dist[i]) min_dist[i] = d;
        }
    }

    // Build selected_cfgs_
    selected_cfgs_.clear();
    for (int idx : selected_indices_)
        selected_cfgs_.push_back(cfgs_[idx]);

    return static_cast<int>(selected_indices_.size());
}

// ── Public selection API ──────────────────────────────────────────────────────

int FPSSelector::select(int n, int first_idx)
{
    return compute_fps_order(first_idx, n, 0.0, false);
}

int FPSSelector::select_by_threshold(double dist_thr, int max_n, int first_idx)
{
    return compute_fps_order(first_idx, max_n, dist_thr, true);
}

// ── Result accessors ──────────────────────────────────────────────────────────

const std::vector<Atoms> &FPSSelector::get_selected() const { return selected_cfgs_; }
const std::vector<int>    &FPSSelector::selected_indices() const { return selected_indices_; }
const std::vector<double> &FPSSelector::selection_scores() const { return selection_scores_; }
const std::vector<int>    &FPSSelector::selection_order() const { return selection_order_; }
