// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "active_learn/al_scorer.h"
#include "tools/gpm_container.h"
#include "active_learn/al_potential.h"
#include "maxvol/active_selector.h"

namespace AL {

MaxVolScorer::~MaxVolScorer()
{
    if (own_selector_ && as_) { delete as_; as_ = nullptr; }
}

MaxVolScorer::MaxVolScorer(const std::string &state_file,
                           const std::string &index_file)
{
    load_state(state_file, index_file);
}

void MaxVolScorer::load_state(const std::string &state_file,
                               const std::string &index_file)
{
    if (own_selector_ && as_) { delete as_; as_ = nullptr; }
    own_selector_ = false; loaded_ = false; builder_initialized_ = false;

    auto p = std::make_unique<ActiveSelector>();
    p->load_state(state_file, index_file);
    // Loading errors throw; a valid but empty state cannot be used for scoring.
    if (p->num_candidates() == 0) {
        GPERROR("al:state_file '" + state_file + "' is missing, invalid or contains no candidates");
    }
    as_ = p.release();
    own_selector_ = true;
    loaded_ = true;
}

void MaxVolScorer::init_builder(Potentials::AnyPotential &pot)
{
    builder_initialized_ = false;
    if (!as_) GPERROR("MaxVol scorer has no loaded state");
    as_->init_builder(pot);
    if (!as_->prepare_grading()) GPERROR("Cannot initialize MaxVol scoring from this state and potential");
    builder_initialized_ = true;
}

ALScore MaxVolScorer::evaluate(Atoms &atom)
{
    ALScore s;
    if (loaded_ && as_) {
        if (!builder_initialized_) GPERROR("MaxVol scorer requires init_builder(potential) before evaluation");
        double grade = as_->grade(atom);
        // Use raw MaxVol grade as uncertainty (same convention as maxvol_sample).
        // grade < 1.0  → inside active set
        // grade ~ 1-10 → mild extrapolation
        // grade >> 10  → extreme extrapolation (threshold_break territory)
        s.uncertainty = grade;
    }
    return s;
}

// ═══════════════════════════════════════════════════════════════════════════
// Unified factory
// ═══════════════════════════════════════════════════════════════════════════

ALSetup al_create_setup(const GPSettings &al_settings)
{
    ALSetup setup;
    auto al = ALPotential::create(al_settings);
    if (!al || !al->active()) return setup;

    GPM::require(al->scorer()!=nullptr,"embedded AL requires init_builder(potential); use ALPotential::create directly");
    setup.scorer = al->scorer();
    setup.al_pot = std::move(al);
    return setup;
}

} // namespace AL
