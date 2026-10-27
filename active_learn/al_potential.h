// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "active_learn/al_scorer.h"
#include "base/structures.h"
#include "utils/gputils.h"
#include <memory>
#include <string>
#include <vector>

class GPIO;
namespace Potentials { class ModelBundle; }

namespace AL {

/// @brief Active learning manager — load state, evaluate prediction reliability,
/// collect uncertain candidates, sparsify the pool.
///
/// One-stop entry point for on-the-fly active learning in any simulation.
///
/// Usage:
///   auto al = AL::ALPotential::create(settings);
///   if (al) al->check(atom);      // evaluate + auto-collect if uncertain
///   if (al) al->finalize();        // sparsify pool + export at the end
///
/// MaxVol scoring uses an ActiveSelector loaded from a .sel state file.
class ALPotential {
    friend struct GPMALAccess;
    friend class Potentials::ModelBundle;
public:
    ALPotential() = default;
    ~ALPotential() = default;
    ALPotential(const ALPotential &) = delete;
    ALPotential &operator=(const ALPotential &) = delete;

    // ── Factory ──────────────────────────────────────────────────────────────

    /// Create from GPSettings (al: section, with prefix stripped).
    ///
    /// Recognised keys:
    ///   mode             = maxvol | none                  (default: none)
    ///   threshold        = 2.01     MaxVol grade threshold (raw grade)
    ///   threshold_break  = 100.0    grade above which the caller should abort
    ///   interval         = 1        check every N steps
    ///   export_file      = al_candidates.xyz
    ///   sparsify_to      = 1000     max pool size after sparsification
    ///   sparsify_method  = fps      fps | random
    ///   # MaxVol mode:
    ///   state_file       = X.sel    required ActiveSelector state file
    ///   index_file       =          optional index.map for species names
    ///
    /// Returns nullptr only when disabled; invalid configuration or load failures throw.
    /// Throws std::invalid_argument for an unsupported mode.
    static std::unique_ptr<ALPotential> create(const GPSettings &al_settings);

    /// True when scorer is loaded and active.
    ActiveSelector *maxvol_selector();
    void save(const std::string &path, Potentials::AnyPotential &model);
    bool active() const { return embedded_pending_ || static_cast<bool>(scorer_); }

    // ── Uncertainty evaluation ───────────────────────────────────────────────

    /// Evaluate uncertainty for one structure.
    /// Preserves atom energy/forces/stress (restored after evaluation).
    /// @return ALScore with the MaxVol extrapolation grade.
    ALScore evaluate(Atoms &atom);

    /// Check if structure is uncertain (uncertainty > threshold).
    /// Does NOT modify the pool. Use check() for auto-collect.
    bool is_uncertain(Atoms &atom);

    /// Evaluate uncertainty and, if above threshold, copy atom into the
    /// candidate pool.  Safe to call every step — built-in interval throttling.
    /// @return true if uncertainty > threshold_break (caller should abort).
    bool check(Atoms &atom);

    /// True after check() returns true (grade exceeded threshold_break).
    bool should_abort() const { return should_abort_; }

    // ── Candidate pool ───────────────────────────────────────────────────────

    int pool_size() const { return static_cast<int>(pool_.size()); }
    const std::vector<Atoms> &pool() const { return pool_; }

    /// Direct insertion (no threshold check).
    void add_to_pool(const Atoms &atom);

    /// Sparsify the pool to `sparsify_to_` structures using the configured sampling method,
    /// then export to `export_file_`.
    /// Call once after the simulation ends.
    /// @param iobj  External GPIO for I/O (avoids duplicate GPIO construction).
    ///              If nullptr, a temporary one is created (may warn in multi-GPIO scenarios).
    void finalize(GPIO *iobj = nullptr);

    /// Override only the export path, retaining the configured pool size.
    void finalize(const std::string &export_file, GPIO *iobj = nullptr);

    /// Override sparsify/export parameters for one-shot use.
    void finalize(int sparsify_to, const std::string &export_file, GPIO *iobj = nullptr);

    /// Initialize builder from stored metadata + a user-provided potential.
    /// Must be called after create() and before check()/evaluate().
    void init_builder(Potentials::AnyPotential &pot);

    // ── Accessors (for adapters) ─────────────────────────────────────────────

    ALScorer *scorer() { return scorer_.get(); }
    const ALScorer *scorer() const { return scorer_.get(); }
    std::string mode_name() const { return mode_name_; }
    double threshold() const { return threshold_; }

private:
    std::shared_ptr<Potentials::ModelBundle> bundle_owner_;
    GPSettings embedded_settings_;
    bool embedded_pending_ = false;
    Potentials::AnyPotential *bound_model_ = nullptr;
    std::string bound_fingerprint_;

    std::unique_ptr<ALScorer> scorer_;

    // Configuration
    double threshold_ = 2.01;
    double threshold_break_ = 100.0;  // grade above this → abort simulation
    bool   should_abort_ = false;
    std::string export_file_ = "al_candidates.xyz";
    std::string sparsify_method_ = "fps";
    int sparsify_to_ = 1000;
    int check_interval_ = 1;
    int step_counter_ = 0;
    std::string mode_name_ = "none";

    // Candidate pool buffer
    std::vector<Atoms> pool_;

    // Private factory helpers
    static std::unique_ptr<ALPotential> load_maxvol(
        const std::string &state_file, const std::string &index_file,
        double threshold, double threshold_break, int interval,
        const std::string &export_file, int sparsify_to,
        const std::string &sparsify_method);
};

} // namespace AL
