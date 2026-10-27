// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "active_learn/al_potential.h"
#include "active_learn/gpm_al.h"
#include "potentials/model_bundle.h"
#include "potentials/gpm_codec.h"
#include "maxvol/active_selector.h"
#include "io/gpio.h"
#include "utils/gputils.h"
#include <fmt/format.h>
#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>

namespace AL {

// ═══════════════════════════════════════════════════════════════════════════
// Factory: create from GPSettings
// ═══════════════════════════════════════════════════════════════════════════

std::unique_ptr<ALPotential> ALPotential::create(const GPSettings &al_settings)
{
    if(al_settings["source"] == "embedded") {
        GPM::require(al_settings["state_file"].empty() && al_settings["committee_prefix"].empty(),
                     "source=embedded conflicts with state_file/committee_prefix");
        auto al=std::make_unique<ALPotential>();al->embedded_settings_=al_settings;
        al->embedded_pending_=true;return al;
    }
    GPM::require(al_settings["source"].empty(),"unknown AL source");
    std::string mode = al_settings["mode"];
    // empty / "none" → AL not configured, bail out immediately
    if (mode.empty() || mode == "none") return nullptr;
    if (mode != "maxvol")
        throw std::invalid_argument("Unsupported active-learning mode: " + mode +
                                    "; supported modes are maxvol and none");

    // Helper: read numeric key with default fallback
    auto get_double = [&](const std::string &key, double def) -> double {
        auto v = al_settings[key];
        if (v.empty()) return def;
        Parameters::bind(def, "al:" + key)->apply_setting(v, "active learning");
        return def;
    };
    auto get_int = [&](const std::string &key, int def) -> int {
        auto v = al_settings[key];
        if (v.empty()) return def;
        Parameters::bind(def, "al:" + key)->apply_setting(v, "active learning");
        return def;
    };

    double threshold = get_double("threshold", 2.01);
    double threshold_break = get_double("threshold_break", 100.0);
    int interval = get_int("interval", 1);
    if (interval <= 0)
        GPERROR("al:interval must be positive");
    std::string export_file = al_settings["export_file"];
    if (export_file.empty())
        export_file = "al_candidates.xyz";
    int sparsify_to = get_int("sparsify_to", 1000);
    std::string sparsify_method = al_settings["sparsify_method"];
    if (sparsify_method.empty()) sparsify_method = "fps";

    if (mode == "maxvol") {
        std::string state_file = al_settings["state_file"];
        if (state_file.empty()) GPERROR("al:state_file is required in maxvol mode");
        std::string idx = al_settings["index_file"];
        return load_maxvol(state_file, idx,
                           threshold, threshold_break, interval,
                           export_file, sparsify_to, sparsify_method);
    }

    return nullptr;
}

// ═══════════════════════════════════════════════════════════════════════════
// Private factory helpers
// ═══════════════════════════════════════════════════════════════════════════

std::unique_ptr<ALPotential> ALPotential::load_maxvol(
    const std::string &state_file, const std::string &index_file,
    double threshold, double threshold_break, int interval,
    const std::string &export_file, int sparsify_to,
    const std::string &sparsify_method)
{
    auto al = std::make_unique<ALPotential>();
    al->mode_name_ = "maxvol";
    al->threshold_ = threshold;
    al->threshold_break_ = threshold_break;
    al->check_interval_ = interval;
    al->export_file_ = export_file;
    al->sparsify_to_ = sparsify_to;
    al->sparsify_method_ = sparsify_method;

    auto mv = std::make_unique<MaxVolScorer>();
    mv->load_state(state_file, index_file);

    al->scorer_ = std::move(mv);
    return al;
}

// ═══════════════════════════════════════════════════════════════════════════
// Evaluation
// ═══════════════════════════════════════════════════════════════════════════

ALScore ALPotential::evaluate(Atoms &atom)
{
    GPM::require(!embedded_pending_,"initialize embedded AL with its potential before scoring");
    if(bound_model_)GPM::require(Potentials::ModelBundle::fingerprint(*bound_model_)==bound_fingerprint_,"AL target changed; rebuild state");
    if (scorer_)
        return scorer_->evaluate(atom);
    return ALScore{};
}

bool ALPotential::is_uncertain(Atoms &atom)
{
    auto score = evaluate(atom);
    return score.uncertainty > threshold_;
}

bool ALPotential::check(Atoms &atom)
{
    if (!scorer_) return false;

    // Interval throttling
    if (++step_counter_ % check_interval_ != 0) return false;

    auto score = scorer_->evaluate(atom);

    // Extreme extrapolation — signal caller to abort the simulation
    if (score.uncertainty > threshold_break_) {
        should_abort_ = true;
        atom.features_["AL_uncertainty"] = std::to_string(score.uncertainty);
        atom.features_["AL_scorer"]     = scorer_->name();
        pool_.push_back(atom);
        return true;
    }

    if (score.uncertainty > threshold_) {
        atom.features_["AL_uncertainty"] = std::to_string(score.uncertainty);
        atom.features_["AL_scorer"]     = scorer_->name();
        pool_.push_back(atom);
    }
    return false;
}

// ═══════════════════════════════════════════════════════════════════════════
// Candidate pool
// ═══════════════════════════════════════════════════════════════════════════

void ALPotential::init_builder(Potentials::AnyPotential &pot)
{
    if(embedded_pending_) {
        GPM::require(bool(pot.gpm_source),"source=embedded requires a GPM model");
        auto restored=GPMALAccess::decode(*pot.gpm_source,pot,embedded_settings_);
        scorer_=std::move(restored->scorer_);
        threshold_=restored->threshold_;threshold_break_=restored->threshold_break_;
        should_abort_=restored->should_abort_;export_file_=restored->export_file_;
        sparsify_method_=restored->sparsify_method_;sparsify_to_=restored->sparsify_to_;
        check_interval_=restored->check_interval_;step_counter_=restored->step_counter_;
        mode_name_=restored->mode_name_;pool_=std::move(restored->pool_);embedded_pending_=false;
        bound_model_=&pot;bound_fingerprint_=Potentials::ModelBundle::fingerprint(pot);return;
    }
    if(bound_model_) {
        GPM::require(Potentials::ModelBundle::fingerprint(pot)==bound_fingerprint_,"AL target model changed");
        bound_model_=&pot;
    }

    auto *mv = dynamic_cast<MaxVolScorer *>(scorer_.get());
    if (mv) mv->init_builder(pot);
}

void ALPotential::add_to_pool(const Atoms &atom)
{
    pool_.push_back(atom);
}

void ALPotential::finalize(GPIO *iobj)
{
    finalize(sparsify_to_, export_file_, iobj);
}

void ALPotential::finalize(const std::string &export_file, GPIO *iobj)
{
    finalize(sparsify_to_, export_file, iobj);
}

void ALPotential::finalize(int sparsify_to, const std::string &export_file, GPIO *iobj)
{
    if (pool_.empty()) {
        Message("ALPotential: pool is empty, nothing to export.");
        return;
    }
    if (sparsify_to <= 0) {
        Message("ALPotential: sparsify_to <= 0, clearing candidate pool.");
        pool_.clear();
        return;
    }

    auto sparsify_random = [&](int target) {
        std::mt19937 rng{42};
        std::vector<Atoms> subset;
        subset.reserve(target);
        std::sample(pool_.begin(), pool_.end(),
                    std::back_inserter(subset),
                    target, rng);
        pool_ = std::move(subset);
    };

    auto sparsify_fps = [&](int target) {
        const int n = static_cast<int>(pool_.size());
        constexpr int dim = 12;
        Eigen::Matrix<double, -1, dim, Eigen::RowMajor> X(n, dim);
        for (int i = 0; i < n; ++i) {
            const Atoms &a = pool_[i];
            const double nat = std::max(1, a.size());
            X(i, 0) = static_cast<double>(a.size());
            X(i, 1) = std::abs(a.cell_.determinant());
            X(i, 2) = a.has_energy() ? a.energy_ / nat : 0.0;
            int c = 3;
            for (int r = 0; r < 3; ++r)
                for (int col = 0; col < 3; ++col)
                    X(i, c++) = a.has_stresses() ? a.stress_(r, col) : 0.0;
        }
        if (!X.allFinite()) GPERROR("AL candidate features must be finite for FPS");
        for (int j = 0; j < dim; ++j) {
            double mean = X.col(j).mean();
            Eigen::ArrayXd centered = X.col(j).array() - mean;
            double scale = std::sqrt(centered.square().mean());
            if (!std::isfinite(scale) || scale < 1e-12) scale = 1.0;
            X.col(j) = centered / scale;
        }

        std::vector<int> selected;
        selected.reserve(target);
        selected.push_back(0);
        std::vector<double> min_dist(n, std::numeric_limits<double>::infinity());
        min_dist[0] = -1.0; // Exclude selected rows even when all distances are zero.
        for (int i = 1; i < n; ++i)
            min_dist[i] = (X.row(i) - X.row(0)).norm();

        while (static_cast<int>(selected.size()) < target) {
            int best = -1;
            double best_dist = -1.0;
            for (int i = 0; i < n; ++i) {
                if (min_dist[i] > best_dist) {
                    best_dist = min_dist[i];
                    best = i;
                }
            }
            if (best < 0) break;
            selected.push_back(best);
            min_dist[best] = -1.0;
            for (int i = 0; i < n; ++i) {
                if (min_dist[i] <= 0.0) continue;
                double d = (X.row(i) - X.row(best)).norm();
                if (d < min_dist[i]) min_dist[i] = d;
            }
        }

        std::vector<Atoms> subset;
        subset.reserve(selected.size());
        for (int idx : selected) subset.push_back(pool_[idx]);
        pool_ = std::move(subset);
    };

    // Sparsify if pool exceeds target.
    if (static_cast<int>(pool_.size()) > sparsify_to) {
        Message(fmt::format("ALPotential: sparsifying pool ({} → {})",
                            pool_.size(), sparsify_to));
        if (sparsify_method_ == "random") {
            sparsify_random(sparsify_to);
        } else {
            if (sparsify_method_ != "fps")
                Message(fmt::format("ALPotential: unknown sparsify_method '{}', using fps",
                                    sparsify_method_));
            sparsify_fps(sparsify_to);
        }
    }

    std::unique_ptr<GPIO> owner;
    if (!iobj) {
        owner = std::make_unique<GPIO>();
        iobj = owner.get();
    }
    std::ofstream ofs(export_file, std::ios::binary);
    if (!ofs) GPERROR("Cannot open AL candidate output: " + export_file);
    if (!iobj->write(pool_, ofs, "extxyz"))
        GPERROR("Failed to export AL candidates: " + export_file);
    ofs.close();
    if (!ofs) GPERROR("Failed to export AL candidates: " + export_file);
    Message(fmt::format("ALPotential: exported {} candidate structures to {}",
                        pool_.size(), export_file));
}

ActiveSelector *ALPotential::maxvol_selector() {auto *s=dynamic_cast<MaxVolScorer*>(scorer_.get());return s?s->selector():nullptr;}
void ALPotential::save(const std::string &path,Potentials::AnyPotential &model) {
    GPM::Writer writer;Potentials::GPMCodecAccess::encode(writer,model);
    writer.manifest["species"]=writer.manifest["models"].get<GPM::Object>().at("root").get<GPM::Object>().at("species");
    writer.manifest["model_fingerprint"]=GPM::Value(Potentials::ModelBundle::fingerprint(model));
    GPMALAccess::encode(writer,*this,model);writer.save(path);
}
} // namespace AL
