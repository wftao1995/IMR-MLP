// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "train/linear_efs_system.h"
#include <memory>
#include <functional>
#include <string>
#include <vector>

namespace func_fastfit {
// Explicitly owned by a batch caller; never persisted across processes/builds.
// Clear/destroy before freeing the MPI communicator used by the batch.
// Keys store exact input bytes, avoiding hash collisions or timestamp checks.
class AssemblyCache {
    struct Entry {
        std::string key;
        std::shared_ptr<Training::LinearEFSSystem> system;
    };
    std::vector<Entry> entries_;
public:
    std::shared_ptr<Training::LinearEFSSystem> find(const std::string &key) const {
        for (const auto &entry : entries_) if (entry.key == key) return entry.system;
        return {};
    }
    void put(std::string key, std::shared_ptr<Training::LinearEFSSystem> system) {
        std::erase_if(entries_, [&](const auto &entry) { return entry.key == key; });
        if (entries_.size() == 2) entries_.erase(entries_.begin());
        entries_.push_back({std::move(key), std::move(system)});
    }
    void clear() { entries_.clear(); }
};

// Mathematical inputs used by plain PotMTP's EFS assembly. Frame order,
// periodicity, label presence, per-frame/per-atom weights and normalization
// are significant. A rank's exact local key is checked collectively by callers.
inline std::string assembly_key(const std::string &model_bytes,
        const std::vector<Atoms> &data, const EFSFitWeights &w, int rank, int ranks) {
    std::string key;
    auto add = [&](const auto &value) {
        key.append(reinterpret_cast<const char *>(&value), sizeof(value));
    };
    add(model_bytes.size()); key += model_bytes;
    add(rank); add(ranks);
    for (double x : {w.energy,w.force,w.stress,w.energy_power,w.force_power,w.stress_power,w.norm_d,w.norm_f,w.norm_p}) add(x);
    add(data.size());
    for (const auto &a : data) {
        if (a.have_ghost) throw std::runtime_error("Assembly cache does not accept ghost structures");
        add(a.cartesian);
        for (bool x : a.pbc_) add(x);
        for (int i=0;i<9;++i) add(a.cell_.data()[i]);
        add(a.pos_.size()); for (const auto &p : a.pos_) for (int d=0;d<3;++d) add(p[d]);
        add(a.chemmap_.size()); for (int x : a.chemmap_) add(x);
        add(a.has_energy()); if (a.has_energy()) add(a.energy_);
        add(a.forces_.size()); for (const auto &f : a.forces_) for (int d=0;d<3;++d) add(f[d]);
        add(a.has_stresses()); if (a.has_stresses()) for (int i=0;i<9;++i) add(a.stress_.data()[i]);
        add(a.fitness);
        add(a.atomic_fitness.size()); for (double x : a.atomic_fitness) add(x);
    }
    return key;
}

// Same settings/output behavior as fastfit, with at most two global assembled
// systems retained by the caller. All MPI ranks must use the same entry point.
int fastfit_cached(int argc, char *argv[], AssemblyCache &cache, bool assemble_only = false);

// Opt-in, root-only observations for offline benchmarks. Callbacks must not
// enter MPI collectives or mutate the supplied objects. Exceptions propagate
// collectively; the ordinary CLI and cached entry point retain their behavior.
struct BenchmarkObserver {
    std::function<void(const Potentials::PotMTP&,
        std::shared_ptr<const Training::LinearEFSSystem>,
        std::shared_ptr<const Training::LinearEFSSystem>,
        const Eigen::MatrixXd&, const Eigen::VectorXd&, int, int)> systems;
    std::function<void(const std::vector<int>&, double, double, bool)> candidate;
    std::function<void(const std::vector<int>&, double)> cur_sample;
};
int fastfit_observed(int argc, char *argv[], AssemblyCache &cache,
                    BenchmarkObserver &observer, bool assemble_only = false,
                    // Explicit experiment-only numerical path. False preserves
                    // the ordinary CLI/cached implementation and RNG sequence.
                    bool experimental_square_root = false);
} // namespace func_fastfit
