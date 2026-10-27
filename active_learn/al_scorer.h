// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "base/structures.h"
#include "utils/gputils.h"
#include <memory>
#include <string>

class ActiveSelector;  // fwd (global scope)

namespace Potentials { class AnyPotential; }

namespace AL { class ALPotential; }

namespace AL {

struct ALScore {
    double uncertainty = 0;   // scalar: higher = more interesting
};

/// @brief Abstract scoring interface for on-the-fly active learning.
///
/// MaxVolScorer grades extrapolation against a stored active set.
class ALScorer {
public:
    virtual ~ALScorer() = default;
    virtual ALScore evaluate(Atoms &atom) = 0;
    virtual double cutoff() const = 0;
    virtual std::string name() const = 0;
};

// ── MaxVol-based scorer ──────────────────────────────────────────────────

class MaxVolScorer : public ALScorer {
    friend struct GPMALAccess;
public:
    MaxVolScorer() = default;
    ~MaxVolScorer() override;
    explicit MaxVolScorer(const std::string &state_file,
                          const std::string &index_file = "");

    void load_state(const std::string &state_file,
                    const std::string &index_file = "");

    /// Reconstruct the internal builder from stored descriptor metadata
    /// + a user-provided potential.  Pass through to ActiveSelector.
    void init_builder(Potentials::AnyPotential &pot);

    ALScore evaluate(Atoms &atom) override;
    double cutoff() const override { return cutoff_; }
    std::string name() const override { return "maxvol"; }

    void set_cutoff(double r) { cutoff_ = r; }
    ActiveSelector *selector() { return as_; }
    bool loaded() const { return loaded_; }

private:
    ActiveSelector *as_ = nullptr;
    double cutoff_ = 5.0;
    bool loaded_ = false;
    bool builder_initialized_ = false;
    bool own_selector_ = false;
};

// ── Unified factory ───────────────────────────────────────────────────────

/// Holds the scorer and its owning ALPotential.
/// Created by al_create_setup().
///
/// Usage (NEB):
///   auto al = al_create_setup(settings.ExtractSubSettings("al"));
///   if (al.scorer) { neb.al_scorer_ = al.scorer; neb.on_the_fly_flag = true; }
struct ALSetup {
    ALScorer *scorer = nullptr;               // points into al_pot
    std::unique_ptr<ALPotential> al_pot;       // owns scorer
};

/// Create an AL scorer + modifier from GPSettings (one-stop entry).
///
/// Settings keys (under [al] section):
///   mode = maxvol | none   (default: none)
///   threshold = 2.01        # uncertainty threshold (MaxVol: raw grade)
///   interval = 1            # check every N steps
///   export_file = al_candidates.xyz
///   sparsify_to = 1000
///
///   # MaxVol mode:
///   state_file = X.sel      # ActiveSelector state file
///   index_file = index.map  # optional
///
/// Returns empty ALSetup if disabled or the state cannot be loaded.
/// Throws std::invalid_argument for an unsupported mode.
ALSetup al_create_setup(const GPSettings &al_settings);

} // namespace AL
