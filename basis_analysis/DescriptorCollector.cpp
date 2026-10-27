// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "DescriptorCollector.h"
#include "potentials/potential_mtp.h"
#include <random>
#include <numeric>
#include <cstring>

namespace BasisAnalysis {

DescriptorCollector::DescriptorCollector(Potentials::PotMTP &pot, size_t max_samples, unsigned seed)
    : pot_(pot), max_samples_(max_samples), seed_(seed) {}

std::vector<int> DescriptorCollector::build_selection(int total_sites) {
    std::vector<int> select;
    if (total_sites <= static_cast<int>(max_samples_)) {
        select.assign(total_sites, 1);
        return select;
    }
    // Random subsample
    select.assign(total_sites, 0);
    std::vector<int> indices(total_sites);
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 gen(seed_);
    std::shuffle(indices.begin(), indices.end(), gen);
    indices.resize(max_samples_);
    for (int idx : indices) {
        select[idx] = 1;
    }
    return select;
}

Eigen::MatrixXd DescriptorCollector::collect(const std::vector<Atoms> &atoms) {
    nsites_ = 0;
    int nscalar = pot_.alpha_scalar_moments;
    if (max_samples_ == 0) return Eigen::MatrixXd(nscalar, 0);
    if (nscalar == 0) {
        return Eigen::MatrixXd();
    }

    // Count total atoms
    int total_sites = 0;
    for (const auto &atom : atoms) {
        total_sites += atom.size();
    }
    if (total_sites == 0) {
        return Eigen::MatrixXd(nscalar, 0);
    }

    std::vector<int> select = build_selection(total_sites);
    int nsampled = 0;
    for (int v : select) if (v) ++nsampled;

    // Allocate descriptor matrix: (nscalar x nsampled)
    Eigen::MatrixXd desc(nscalar, nsampled);
    desc.setZero();

    std::vector<double> dbuffer;
    int site_idx = 0;
    int col = 0;

    for (const auto &atom : atoms) {
        if (std::none_of(select.begin() + site_idx, select.begin() + site_idx + atom.size(), [](int v) { return v != 0; })) {
            site_idx += atom.size();
            continue;
        }
        auto nbhs = Neighborhoods(atom, pot_.cutoff);
        for (int iat = 0; iat < atom.size(); ++iat) {
            if (!select[site_idx]) {
                ++site_idx;
                continue;
            }
            pot_.compute_descriptors(nbhs[iat], dbuffer);
            // dbuffer layout: [1.0, B_0, B_1, ..., B_{nscalar-1}]
            for (int f = 0; f < nscalar; ++f) {
                desc(f, col) = dbuffer[f + 1];
            }
            ++col;
            ++site_idx;
        }
    }
    nsites_ = nsampled;
    return desc;
}

} // namespace BasisAnalysis
