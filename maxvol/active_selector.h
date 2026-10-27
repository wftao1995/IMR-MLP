// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
namespace AL { struct GPMALAccess; }

#include "maxvol/maxvol.h"
#include "maxvol/b_matrix_builder.h"
#include "maxvol/online_maxvol.h"
#include "base/structures.h"
#include "utils/mpi_stubs.h"
#include <Eigen/Core>
#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace Potentials { class AnyPotential; }
class StreamRowStore;

/// Stored builder metadata recovered from a state file.
/// Contains everything needed to reconstruct a BMatrixBuilder,
/// except the potential itself (injected later via init_builder).
struct BuilderParams {
    std::string dtype;    // "mtm", "mtp", "pot_efs", …
    std::string mode;     // "cfg" or "nbh" (mtm family)
    GPSettings settings;  // descriptor construction settings
    // PotEFSBuilder extras
    double w_energy = 1.0, w_site = 0.0, w_force = 0.0, w_stress = 0.0;
    bool   linear_only = false;
    RowScalingMode row_scaling = RowScalingMode::Uniform;
    bool   valid = false;
};

/// @brief Active learning selector built on top of the MaxVol algorithm.
/// @ingroup maxvol
///
/// Generic over the descriptor / basis type via a BMatrixBuilder.
///
/// Three workflows (mirroring MLIP-3):
///   1. select     — build active set from a pool
///   2. sample     — grade candidates without modifying active set
///   3. select_add — augment training set with informative candidates
class ActiveSelector : public ConnectSettings
{
    friend struct AL::GPMALAccess;
public:
    struct StreamSelectionStats {
        std::size_t builder_calls = 0;
        std::size_t rows_built = 0;
        int descriptor_passes = 0;
        int score_passes = 0;
        int swaps = 0;
        std::size_t resident_bytes = 0;
        std::size_t mapped_bytes = 0;
        int configured_threads = 1;
        int build_threads = 1;
        int score_threads = 1;
        double build_seconds = 0.0;
        double initialization_seconds = 0.0;
        double score_seconds = 0.0;
        double total_seconds = 0.0;
        double mpi_seconds = 0.0;
        std::string cache_backend = "off";
    };

    double threshold  = 1.001;
    double weight     = 1.0;     // B-matrix row weight
    double pow_scale  = 0.5;     // natoms scaling: row *= weight / natoms^pow
    std::string init_mode = "auto"; // auto | mlip | data
    double init_value = 1e-6;        // MLIP-compatible virtual identity scale
    int    max_swaps  = 0;
    bool   verbose    = false;
    // online is order-dependent and does not revisit rejected historical rows.
    std::string select_mode = "stream";   // stream | full | online (approximate)
    // exact rescans the cached complete pool until convergence or the cap.
    std::string stream_mode = "exact";    // exact | fast
    int    stream_chunk_size = 32;
    int    stream_fast_passes = 2;
    std::string stream_cache_mode = "auto"; // auto | memory | mmap | off
    int    stream_memory_limit_mb = 3072;
    std::string stream_cache_dir;
    int    stream_fast_shortlist = 0;       // 0 = automatic

    ActiveSelector();
    void RegisterConsole(CMDOption::CommandLineOptions &cmdopts);

    /// Construct with a B-matrix builder (takes ownership).
    explicit ActiveSelector(BMatrixBuilder *builder);

    ~ActiveSelector();
    ActiveSelector(const ActiveSelector &) = delete;
    ActiveSelector &operator=(const ActiveSelector &) = delete;

    // ── Candidate management ─────────────────────────────────────────────
    int  add_candidate(Atoms &cfg);
    int  add_candidates(std::vector<Atoms> &cfgs);
    int  add_candidates_parallel(std::vector<Atoms> &cfgs);
    int  add_candidates_mpi(std::vector<Atoms> &cfgs, MPI_Comm comm, int root = 0);
    int  add_candidates_unbuilt(std::vector<Atoms> &cfgs);
    bool append_built_candidates_from(const ActiveSelector &other);
    void clear_candidates();
    int  num_candidates() const { return static_cast<int>(cfg_row_start_.size()) - 1; }
    int  num_rows() const { return static_cast<int>(B_.rows()); }
    int  num_coeffs() const { return n_coeff_; }

    // ── Selection ───────────────────────────────────────────────────────
    int  select(double threshold);
    int  select();
    int  select_streaming(std::vector<Atoms> &cfgs);
    int  select_streaming();
    int  select_streaming_mpi(MPI_Comm comm);
    int  select_online();

    // ── Grading (read-only) ──────────────────────────────────────────────
    double grade(Atoms &cfg);
    void   grade_all(std::vector<Atoms> &cfgs, std::vector<double> &grades);

    /// Prepare the online factorization or fallback MaxVol cache used by
    /// grade_one().
    /// Returns false if the selector has no active set to grade against.
    bool   prepare_grading();

    /// Replace runtime state with a compact active-row matrix sufficient for
    /// grade()/grade_one().  This is used by MPI non-root ranks after rank0 has
    /// selected the active rows.
    bool   set_active_rows_for_grading(
        const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &C,
        const std::vector<int> &selected_config_indices,
        const std::vector<Atoms> *all_cfgs = nullptr);

    /// Broadcast selected rows from root and initialize compact grading state
    /// on non-root ranks. Root keeps its full selector state.
    bool   broadcast_grading_state(MPI_Comm comm,
                                   const std::vector<Atoms> *all_cfgs = nullptr,
                                   int root = 0);

    /// Compute per-structure grades, reusing full B or a complete stream cache
    /// when available. Compact online states rebuild rows only when this
    /// explicit full-pool grading operation is requested.
    void   compute_grades(std::vector<double> &grades);

    /// Rebuild B_ and cfg_row_start_ from cfgs_ by recomputing descriptors.
    /// Called after load_state() so that select() sees the full training matrix.
    void rebuild_B_from_configs();

    // ── Results ──────────────────────────────────────────────────────────
    const std::vector<Atoms> &get_selected() const { return selected_cfgs_; }
    const std::vector<int> &selected_indices() const { return selected_config_indices_; }
    const std::vector<BRowRef> &selected_row_refs() const { return selected_row_refs_; }
    bool online_capable() const { return online_state_.valid(); }
    double log_determinant() const { return log_det_; }
    bool converged() const { return converged_; }
    double max_remaining_grade() const { return max_remaining_grade_; }
    const std::string &termination_reason() const { return termination_reason_; }
    int virtual_row_count() const;
    const StreamSelectionStats &stream_stats() const { return stream_stats_; }
    const std::vector<MaxVol::SwapStep> &swap_history() const {
        return using_stream_state_ ? stream_swap_history_ : mv_.swap_history();
    }

    static constexpr const char *kNewTag = "!mv_is_new";

    // ── Persistence ──────────────────────────────────────────────────────
    /// Throws on open, serialization or write failure.
    void save_state(const std::string &filename,
                    const std::string &index_file = "",
                    const std::string &pot_file = "",
                    const std::string &pottype = "") const;
    /// Throws on missing or invalid state files and clears loaded selection data.
    void load_state(const std::string &filename,
                    const std::string &index_file = "");

    /// Reconstruct the builder from stored params + a user-provided
    /// potential. Must be called after load_state() and before grade().
    /// Throws if the potential type or coefficient count is incompatible.
    void init_builder(Potentials::AnyPotential &pot);

    /// Print human-readable summary of a state file (static, no instance needed).
    static void dump_state(const std::string &filename, std::ostream &out);

    /// Extract training or selected configs from a state file.
    /// @param filename       Input state file path.
    /// @param output_file    Output file path for exported configs.
    /// @param selected_only  If true, export only selected configs; else all training.
    /// @param index_file     Optional index.map for correct species names in output.
    static void export_configs(const std::string &filename,
                               const std::string &output_file,
                               bool selected_only,
                               const std::string &index_file = "");

    /// Core scoring logic shared by grade() and grade_all().
    /// Uses the online row-volume score when available, otherwise MaxVol scoring.
    double grade_one(Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B_cfg,
                     int nrows);

private:
    bool gpm_rebuild_full_ = false;
    bool gpm_legacy_unverified_ = false;
    int select_streaming_current();
    int select_streaming_current(MPI_Comm comm, bool distributed);
    int select_streaming_legacy_current();
    int select_online_current();
    int select_mlip_full_current();
    bool uses_mlip_initialization() const;
    void reset_selection_status();
    void release_stream_rows();
    void set_selection_status(bool converged, double max_grade,
                              std::string reason);
    bool build_compact_selected_state(
        const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &C,
        const std::vector<BRowRef> &refs);
    void invalidate_grading() { grading_prepared_ = false; }
    void score_candidate_chunk(std::vector<Atoms> &cfgs,
                               std::size_t beg,
                               std::size_t end,
                               BMatrixBuilder &builder,
                               std::vector<double> &grades);

    BMatrixBuilder *builder_ = nullptr;
    bool own_builder_ = false;
    MaxVol mv_;
    int n_coeff_ = 0;
    double log_det_ = 0.0;
    bool using_stream_state_ = false;
    bool has_full_B_ = false;
    bool grading_prepared_ = false;
    bool converged_ = false;
    double max_remaining_grade_ = std::numeric_limits<double>::infinity();
    std::string termination_reason_ = "not-run";

    // Descriptor metadata recovered from state file; consumed by init_builder()
    BuilderParams pending_builder_;

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_;
    std::vector<int>    cfg_row_start_{0};
    std::vector<Atoms>  cfgs_;
    std::vector<bool>   is_new_;

    std::vector<Atoms> selected_cfgs_;
    std::vector<int>   selected_config_indices_;
    std::vector<int>   selected_row_indices_;
    std::vector<BRowRef> selected_row_refs_;
    std::vector<MaxVol::SwapStep> stream_swap_history_;
    StreamSelectionStats stream_stats_;
    std::unique_ptr<StreamRowStore> stream_rows_;

    OnlineMaxVolState online_state_;
    std::size_t online_processed_cfgs_ = 0;

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_full_;
};
