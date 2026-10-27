// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "runtime/mtp/gpu-direct/gpu_backend.h"

#ifdef IMRMLP_GPU

#include "runtime/mtp/gpu-direct/mtp_gpu_direct.h"
#include "base/anyneighborhoods.h"
#include "base/neighborhood.h"
#include "base/structures.h"
#include "potentials/potential_mtp.h"
#include "potentials/les.h"

#include <memory>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace Potentials {

class PotMTPNN1;

// IMR-MLP-facing owner for the backend-neutral GPU MTP core.  It converts the
// native PotMTP representation once, while MTPGpuBatch owns the contiguous
// CSR buffers used for repeated evaluations.
class MTPGpuEvaluator {
public:
    bool initialize(const AnyPotential &potential, int device_id = -1,
                    std::string *error = nullptr);
    bool initialize(
        const AnyPotential &potential, int device_id,
        MLIPCore::GpuDirect::MTPGpuPrecision precision,
        std::string *error = nullptr);
    // Explicit inference primitive for composite backends: compile only the
    // local MTP energy, leaving LES/Coulomb/SOG to the caller's field backend.
    bool initialize_short_range(const AnyPotential &potential, int device_id = -1,
                                std::string *error = nullptr);
    bool initialize_short_range(const AnyPotential &potential, int device_id,
                                MLIPCore::GpuDirect::MTPGpuPrecision precision,
                                std::string *error = nullptr);
    bool initialize_linear_head(const PotMTP &potential,
                                const double *species_coeffs, const double *moment_coeffs,
                                int device_id = -1, std::string *error = nullptr);
    bool initialize_legacy_nn(const PotMTPNN1 &potential, int device_id = -1,
                              std::string *error = nullptr);
    bool update_coefficients(const AnyPotential &potential,
                             std::string *error = nullptr);

    MLIPCore::GpuDirect::MTPGpuDirect &context() { return *context_; }
    const MLIPCore::GpuDirect::MTPGpuDirect &context() const { return *context_; }
    bool initialized() const { return context_ != nullptr; }
    const LES *sog_definition() const { return has_sog_ ? &sog_definition_ : nullptr; }
    size_t reciprocal_bytes(const Atoms &atoms) const;
    bool accumulate_gradient(MLIPCore::GpuDirect::MTPGpuGradientRequest request,
                             std::string *error = nullptr);
    int full_gradient_count() const { return full_gradient_count_; }


private:
    std::unique_ptr<MLIPCore::GpuDirect::MTPGpuDirect> context_;
    std::vector<int> species_map_;
    bool has_sog_ = false;
    bool short_range_only_ = false;
    bool custom_head_ = false;
    bool initialize_impl(const AnyPotential &, int,
                         MLIPCore::GpuDirect::MTPGpuPrecision, bool,
                         std::string *,
                         const MLIPCore::GpuDirect::MTPGpuPotentialView *head_override = nullptr);
    bool inference_only_radial_mode_ = false;
    int full_gradient_count_ = 0;
    std::vector<int> gradient_indices_;
    std::vector<double> full_gradient_;
    void configure_gradient_layout(const AnyPotential &);
    LES sog_definition_;
    bool update_sog(const AnyPotential &, std::string *error);

};

class MTPGpuBatch {
public:
    bool build(const Atoms &atoms, const AnyNeighborhoods &neighborhoods,
               std::string *error = nullptr);
    bool build(const std::vector<Atoms> &atoms,
               const std::vector<Neighborhoods> &neighborhoods,
               std::string *error = nullptr);
    bool build(const std::vector<Atoms> &atoms,
               const std::vector<Neighborhoods> &neighborhoods,
               const std::vector<int> &indices,
               std::string *error = nullptr);
    // Refresh coordinates, types, and displacement vectors while requiring
    // an exactly compatible centers/offsets/neighbors topology. This enables
    // trajectory reuse without making a same-sized but changed neighbor list
    // unsafe.
    bool refresh_compatible(const Atoms &atoms,
                            const AnyNeighborhoods &neighborhoods,
                            std::string *error = nullptr);
    bool evaluate(MTPGpuEvaluator &evaluator, Atoms &result,
                  bool reuse_topology = true, std::string *error = nullptr);
    bool evaluate(MTPGpuEvaluator &evaluator, std::vector<Atoms> &results,
                  bool reuse_topology = true, std::string *error = nullptr);
    bool evaluate_energy(MTPGpuEvaluator &evaluator, Atoms &result,
                         bool reuse_topology = true,
                         std::string *error = nullptr);
    bool evaluate_energy(MTPGpuEvaluator &evaluator,
                         std::vector<Atoms> &results,
                         const std::vector<int> &indices,
                         bool reuse_topology = true,
                         std::string *error = nullptr);
    bool evaluate(MTPGpuEvaluator &evaluator, std::vector<Atoms> &results,
                  const std::vector<int> &indices,
                  bool reuse_topology = true, std::string *error = nullptr);
    bool evaluate_loss(MTPGpuEvaluator &evaluator,
                       const std::vector<double> &target_energy,
                       const std::vector<double> &energy_weight,
                       const std::vector<double> &target_force,
                       const std::vector<double> &force_weight,
                       const std::vector<double> &target_stress,
                       const std::vector<double> &stress_weight,
                       std::array<double,
                           MLIPCore::GpuDirect::MTPGpuLossRequest::metric_count> &metrics,
                       bool reset_metrics = true,
                       bool copy_metrics_to_host = true,
                       std::string *error = nullptr);
    bool accumulate_gradient(MTPGpuEvaluator &evaluator,
                             const std::vector<double> &dloss_denergy,
                             const std::vector<double> &dloss_dforce,
                             const std::vector<double> &dloss_dstress,
                             double *gradient, int gradient_count,
                             bool accumulate = false,
                             std::string *error = nullptr);
    bool accumulate_gradient_chunks(MTPGpuEvaluator &evaluator,
                                    const std::vector<double> &dloss_denergy,
                                    const std::vector<double> &dloss_dforce,
                                    const std::vector<double> &dloss_dstress,
                                    double *gradient, int gradient_count,
                                    bool reset, bool finalize,
                                    std::string *error = nullptr);
    bool accumulate_prepared_gradient(MTPGpuEvaluator &evaluator,
                                      double *gradient, int gradient_count,
                                      bool reset, bool finalize,
                                      std::string *error = nullptr);
    void configure_gradient_reduction(bool deterministic,
                                      int radial_parameter_tile) {
        deterministic_reduction_ = deterministic;
        radial_parameter_tile_ = radial_parameter_tile;
    }

    MLIPCore::GpuDirect::MTPGpuBatchRequest request();
    int structure_count() const {
        return structure_offsets_.empty()
            ? 0 : static_cast<int>(structure_offsets_.size()) - 1;
    }
    int atom_count() const { return static_cast<int>(types_.size()); }
    int neighbor_count() const { return static_cast<int>(neighbors_.size()); }

private:
    bool built_ = false;
    MLIPCore::GpuDirect::GpuCacheIdentity identity_;
    std::uint64_t topology_revision_ = 1, input_revision_ = 1;
    std::uint64_t grid_revision_ = 1, labels_revision_ = 1;
    std::uint64_t uploaded_context_ = 0;
    std::array<std::vector<double>, 6> cached_labels_;
    MLIPCore::GpuDirect::GpuCacheToken input_token() const {
        return {identity_.value, input_revision_};
    }
    bool require_forward(MTPGpuEvaluator &, std::string *error) const;
    bool prepare_sog(MTPGpuEvaluator &, MLIPCore::GpuDirect::MTPGpuBatchRequest &, std::string *error);
    std::vector<Atoms> reciprocal_geometry_;
    std::vector<int> wave_offsets_;
    std::vector<double> wave_vectors_, wave_factors_, wave_taper_, wave_tp_, volumes_;
    MLIPCore::GpuDirect::SOGGpuGridView sog_grid_;
    bool grid_valid_ = false;
    LES grid_definition_;
    bool build_impl(const std::vector<const Atoms *> &atoms,
                    const std::vector<const AnyNeighborhoods *> &neighborhoods,
                    std::string *error);
    bool evaluate_impl(MTPGpuEvaluator &evaluator,
                       const std::vector<Atoms *> &results,
                       bool reuse_topology, std::string *error);
    bool evaluate_energy_impl(MTPGpuEvaluator &evaluator,
                              const std::vector<Atoms *> &results,
                              bool reuse_topology, std::string *error);
    bool can_reuse_topology(const MTPGpuEvaluator &evaluator,
                            bool requested) const;
    void record_uploaded_topology(const MTPGpuEvaluator &evaluator);
    std::vector<int> centers_;
    std::vector<int> center_structure_;
    std::vector<int> structure_offsets_;
    std::vector<int> offsets_;
    std::vector<int> neighbors_;
    std::vector<int> types_;
    std::vector<double> positions_;
    std::vector<double> neighbor_vectors_;
    std::vector<double> forces_;
    std::vector<double> site_energies_;
    std::vector<double> site_virials_;
    bool deterministic_reduction_ = false;
    int radial_parameter_tile_ = 128;
    bool uploaded_once_ = false;
    std::uint64_t uploaded_generation_ = 0;
};

} // namespace Potentials

#endif // IMRMLP_GPU
