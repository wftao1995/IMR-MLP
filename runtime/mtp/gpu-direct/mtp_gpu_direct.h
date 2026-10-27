// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_backend.h"
#include "sog_gpu.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace MLIPCore {
struct mlip;

namespace GpuDirect {

enum class MTPGpuPrecision {
    FP64 = 0,
    Mixed = 1,
    MixedAggressive = 2,
};

enum class MTPGpuForceKernel {
    Auto = 0,
    BasicTile = 1,
    CenterNeighbors = 2,
};

inline constexpr const char *force_kernel_name(MTPGpuForceKernel kernel)
{
    switch (kernel) {
    case MTPGpuForceKernel::Auto:
        return "auto";
    case MTPGpuForceKernel::BasicTile:
        return "basic-tile";
    case MTPGpuForceKernel::CenterNeighbors:
        return "center-neighbors";
    }
    return "unknown";
}

inline bool parse_force_kernel(std::string_view value,
                               MTPGpuForceKernel *kernel)
{
    if (kernel == nullptr) return false;
    if (value == "auto") {
        *kernel = MTPGpuForceKernel::Auto;
        return true;
    }
    if (value == "basic-tile") {
        *kernel = MTPGpuForceKernel::BasicTile;
        return true;
    }
    if (value == "center-neighbors") {
        *kernel = MTPGpuForceKernel::CenterNeighbors;
        return true;
    }
    return false;
}

inline constexpr const char *precision_name(MTPGpuPrecision precision)
{
    switch (precision) {
    case MTPGpuPrecision::FP64:
        return "fp64";
    case MTPGpuPrecision::Mixed:
        return "mixed";
    case MTPGpuPrecision::MixedAggressive:
        return "mixed-aggressive";
    }
    return "unknown";
}

inline bool parse_precision(std::string_view value, MTPGpuPrecision *precision)
{
    if (precision == nullptr) return false;
    if (value == "fp64") {
        *precision = MTPGpuPrecision::FP64;
        return true;
    }
    if (value == "mixed") {
        *precision = MTPGpuPrecision::Mixed;
        return true;
    }
    if (value == "mixed-aggressive") {
        *precision = MTPGpuPrecision::MixedAggressive;
        return true;
    }
    return false;
}

struct MTPGpuInitializationOptions {
    MTPGpuPrecision precision = MTPGpuPrecision::FP64;
    // Validate model upload before allocation; compute requests carry the
    // corresponding total resident-memory budget for dynamic workspaces.
    std::size_t memory_budget_bytes = 0;
};

struct MTPGpuSOGView {
    SOGGpuModelView model;
    const double *species_coeffs = nullptr;
    const double *moment_coeffs = nullptr;
    const double *bandwidth_jacobian = nullptr; // du/dz, one per Gaussian
};

enum class RadialBasis {
    Chebyshev = 0,
    ChebyCos = 1,
    ChebyshevPoly5 = 2,
    ChebyshevConfigurable = 3,
    ChebyshevPoly4 = 4,
    PolyB = 5,
    PolyBCos = 6,
};

enum class RadialCutoff {
    Quadratic = 0,
    Cosine = 1,
    Poly5 = 2,
    NormalizedQuadratic = 3,
    CosineLeft = 4,
    CosineUnscaled = 5,
    CosineSquared = 6,
    SineSquared = 7,
    Poly3 = 8,
    Poly4 = 9,
    Smoothstep5 = 10,
    Smoothstep7 = 11,
};

enum class RadialTransform {
    Linear = 0,
    Exponential = 1,
    ExponentialLeft = 2,
    I2 = 3,
    ExponentialTanh = 4,
};

// Site-energy model applied to one or more linear scalar-moment descriptors.
// Linear is the classic MTP layout. The remaining values use the native
// MTP-M descriptor and model-parameter layout transported by the IMRMLP
// adapter. LES remains outside this view.
enum class MTPGpuModel {
    Linear = 0,
    MTPMQuadratic = 1,
    MTPMAsinh = 2,
    MTPMNN1 = 3,
    MTPMNN2 = 4,
    MTPMLinear = 5,
};

enum class MTPGpuActivation {
    Tanh = 0,
    ReLU = 1,
    Sigmoid = 2,
    Linear = 3,
};

// Optional per-ordered-species-pair radial-basis descriptor. A split MTP, a
// non-split configurable MTP, or a modern parameterized Poly5 MTP supplies
// species_count^2 entries in (center species, neighbor species) order. The
// IMRMLP adapter and standalone LAMMPS reader both normalize those cases;
// legacy single-basis callers use the scalar fields in MTPGpuPotentialView.
struct MTPGpuRadialBasisView {
    RadialBasis radial_basis = RadialBasis::Chebyshev;
    double min_dist = 0.0;
    double max_dist = 0.0;
    double scaling = 1.0;
    // Configurable bases transport their concrete components explicitly so
    // device code never depends on host-side polymorphic objects.
    RadialCutoff cutoff = RadialCutoff::Quadratic;
    RadialTransform transform = RadialTransform::Linear;
    // RBChebyshevPoly5's rb_b, or a configurable Poly5 cutoff's cutoff_b.
    double cutoff_parameter = 0.0;
    // A configurable exponential transform's shift_lambda.
    double transform_parameter = 0.0;
    bool one_minus_chebyshev = false;
    double cutoff_r1 = 0.0;
    double cutoff_r2 = 0.0;
    double transform_scale_parameter = 0.0;
    // PolyB/PolyBCos: rb_size^2 column-major orthogonalization matrix and
    // rb_size normalizations, copied by initialize before it returns.
    const double *polynomial_matrix = nullptr;
    const double *polynomial_normalizations = nullptr;
};

struct MTPGpuPotentialView {
    int species_count = 0;
    int max_type = 0;
    const int *spmap = nullptr;

    int alpha_moments_count = 0;
    int alpha_index_basic_count = 0;
    int alpha_index_times_count = 0;
    int alpha_scalar_moments = 0;

    int radial_func_count = 0;
    int rb_size = 0;
    int max_mu = 0;
    int max_alpha_index_basic = 0;

    double min_dist = 0.0;
    double max_dist = 0.0;
    double cutoff = 0.0;
    double scaling = 1.0;
    RadialBasis radial_basis = RadialBasis::Chebyshev;

    const MTPGpuRadialBasisView *pair_radial_bases = nullptr;
    int pair_radial_basis_count = 0;

    const double *radial_coeffs = nullptr;
    int radial_coeffs_count = 0;
    const double *species_coeffs = nullptr;
    const double *moment_coeffs = nullptr;

    MTPGpuModel model = MTPGpuModel::Linear;
    int descriptor_count = 0;
    const double *descriptor_species_coeffs = nullptr;
    const double *descriptor_moment_coeffs = nullptr;
    const double *model_coeffs = nullptr;
    int model_coeffs_count = 0;
    MTPGpuActivation model_activation = MTPGpuActivation::Tanh;
    int model_hidden1 = 0;
    int model_hidden2 = 0;

    // Flat arrays with four ints per entry:
    // alpha_index_basic: {mu, b1, b2, b3}
    // alpha_index_times: {l1, l2, c, id}
    const int *alpha_index_basic = nullptr;
    const int *alpha_index_times = nullptr;
    const int *alpha_moment_mapping = nullptr;
    const int *nu_arr_list = nullptr;
};

inline MTPGpuPotentialView make_lammps_pair_view(
    int species_count,
    int max_type,
    const int *spmap,
    int alpha_moments_count,
    int alpha_index_basic_count,
    const int *alpha_index_basic,
    int alpha_index_times_count,
    const int *alpha_index_times,
    int alpha_scalar_moments,
    const int *alpha_moment_mapping,
    int radial_func_count,
    int rb_size,
    int max_mu,
    int max_alpha_index_basic,
    double min_dist,
    double max_dist,
    double cutoff,
    double scaling,
    const double *radial_coeffs,
    const double *linear_coeffs,
    const int *nu_arr_list = nullptr,
    RadialBasis radial_basis = RadialBasis::Chebyshev)
{
    MTPGpuPotentialView view;
    view.species_count = species_count;
    view.max_type = max_type;
    view.spmap = spmap;
    view.alpha_moments_count = alpha_moments_count;
    view.alpha_index_basic_count = alpha_index_basic_count;
    view.alpha_index_times_count = alpha_index_times_count;
    view.alpha_scalar_moments = alpha_scalar_moments;
    view.radial_func_count = radial_func_count;
    view.rb_size = rb_size;
    view.max_mu = max_mu;
    view.max_alpha_index_basic = max_alpha_index_basic;
    view.min_dist = min_dist;
    view.max_dist = max_dist;
    view.cutoff = cutoff;
    view.scaling = scaling;
    view.radial_basis = radial_basis;
    view.radial_coeffs = radial_coeffs;
    view.radial_coeffs_count = species_count * species_count * radial_func_count * rb_size;
    view.species_coeffs = linear_coeffs;
    view.moment_coeffs = linear_coeffs + species_count;
    view.alpha_index_basic = alpha_index_basic;
    view.alpha_index_times = alpha_index_times;
    view.alpha_moment_mapping = alpha_moment_mapping;
    view.nu_arr_list = nu_arr_list;
    return view;
}

struct MTPGpuComputeRequest {
    const SOGGpuGridView *sog_grid = nullptr; // complete structures only
    const int *ilist = nullptr;
    const int *numneigh = nullptr;
    int **firstneigh = nullptr;
    // Optional row-major displacement vectors aligned with the flattened
    // neighbor rows.  When null, displacements are computed from x[j]-x[i].
    const double *neighbor_vectors = nullptr;
    int inum = 0;
    int nlocal = 0;
    int ntotal = 0;

    // Optional host array of ntotal fixed weights, indexed by atom. The
    // output is sum_i weight[i] E_i; force and virial use its adjoint.
    // Weighted forward requests are inference-only. Null means unit weights.
    const double *center_energy_weights = nullptr;
    const int *type = nullptr;
    // LAMMPS-style row pointers whose backing x/force storage must be
    // contiguous from row zero.  site_virial, when requested, follows the
    // same rule with six doubles per atom.
    double **x = nullptr;
    double **force = nullptr;

    double *total_energy = nullptr;
    double *site_energy = nullptr;
    double **site_virial = nullptr;
    // Direct rank-local virial: xx, yy, zz, xy, xz, yz; overwritten per call.
    double *global_virial = nullptr;
    // Explicit inference contract; gradient/loss calls are invalid afterwards.
    // Keeps batch/training requests on their original full cache layout.
    bool inference_only = false;
    int centers_per_chunk = 0; // zero selects all centers (or budget-derived size)
    std::size_t memory_budget_bytes = 0; // zero disables the tracked-device cap

    // Optional mask applied to every neighbor-list entry before it is used as
    // an atom index.  LAMMPS encodes special-bond flags in the high bits of a
    // neighbor index; generic callers can leave the default value unchanged.
    int neighbor_index_mask = -1;

    // true matches LAMMPS pair style semantics: add this pair contribution
    // onto the force array already held by atom->f.
    bool accumulate_force = true;

    // Reuse the centers/offsets/neighbors uploaded by the previous call.  The
    // core automatically rebuilds if atom or total-neighbor counts changed.
    // Coordinates and neighbor_vectors are always refreshed.
    bool reuse_topology = false;
    // Training can keep forward outputs resident for device-side loss
    // construction. LAMMPS and inference callers retain the default copies.
    bool copy_outputs_to_host = true;
    // Virial-only inference needs force adjoints, but no force host output.
    bool copy_forces_to_host = true;
    // Energy-only inference skips energy adjoints, direct-force kernels,
    // force/virial buffers, and their host copies. Site energies remain
    // available so multi-structure callers can recover per-structure energy.
    bool compute_forces = true;
    // Keep global and per-atom energy work independent. LAMMPS normally
    // requests neither on the force-only steps between thermo samples.
    // Device-resident training keeps both defaults enabled.
    bool compute_global_energy = true;
    bool compute_site_energy = true;
    // Diagnostic execution mode that orders every reduction contributing to
    // training predictions. Inference/LAMMPS callers keep the fast default.
    bool deterministic_reduction = false;
    // Auto uses CenterNeighbors only for sufficiently large launches and
    // models with more than one basic-moment tile. Explicit selections are
    // retained for regression and capacity work.
    MTPGpuForceKernel force_kernel = MTPGpuForceKernel::Auto;

    // Tuning knobs. The defaults are conservative for double precision MTPs.
    int basic_tile = 128;
    int neighbor_lanes = 8;
    int basic_lanes = 16;
    int propagate_threads = 256;
};

// Backend-neutral contiguous/CSR request used by IMRMLP batch evaluation and
// training.  `offsets` has center_count + 1 entries and indexes `neighbors`.
// Coordinates, forces, and virials use row-major AoS layouts (xyz and 6-vector
// respectively).
struct MTPGpuBatchRequest {
    const SOGGpuGridView *sog_grid = nullptr;
    const int *centers = nullptr;
    const int *offsets = nullptr;
    const int *neighbors = nullptr;
    const double *neighbor_vectors = nullptr;
    int center_count = 0;
    int nlocal = 0;
    int ntotal = 0;

    // Optional host array of ntotal fixed weights, indexed by atom. The
    // output is sum_i weight[i] E_i; force and virial use its adjoint.
    // Weighted forward requests are inference-only. Null means unit weights.
    const double *center_energy_weights = nullptr;
    const int *type = nullptr;
    const double *x = nullptr;
    double *force = nullptr;
    double *total_energy = nullptr;
    // Optional six-component symmetric virial (xx, yy, zz, xy, xz, yz).
    // Reduced on device; does not require downloading per-atom virials.
    double *global_virial = nullptr;
    double *site_energy = nullptr;
    double *site_virial = nullptr;
    bool accumulate_force = true;
    bool reuse_topology = false;
    bool copy_outputs_to_host = true;
    // Virial-only inference needs force adjoints, but no force host output.
    bool copy_forces_to_host = true;
    bool compute_forces = true;
    bool compute_global_energy = true;
    bool compute_site_energy = true;
    bool deterministic_reduction = false;
    MTPGpuForceKernel force_kernel = MTPGpuForceKernel::Auto;

    int basic_tile = 128;
    int neighbor_lanes = 8;
    int basic_lanes = 16;
    int propagate_threads = 256;
    // Optional immutable-input contracts. Change the corresponding revision
    // whenever contents change; zero preserves the uncached API behavior.
    GpuCacheToken topology_token;
    // Opt-in: this topology token also promises immutable atom types. Legacy
    // batch refreshes may change chemistry without changing CSR indices.
    bool topology_includes_types = false;
    GpuCacheToken input_token;
};

// Loss adjoints for a multi-structure training batch.  The immediately
// preceding compute_batch() call must have used the same flattened topology.
// Parameter order is the native PotMTP order:
// radial coefficients, species coefficients, scalar-moment coefficients.
struct MTPGpuGradientRequest {
    int structure_count = 0;
    const int *center_structure = nullptr;
    const double *dloss_denergy = nullptr;
    const double *dloss_dforce = nullptr;
    // Full row-major 3x3 tensors, one per structure.  The forward stress is
    // symmetric, so an arbitrary adjoint is contracted through its symmetric
    // part.  This matches PotMTP::AccumulateEFSCombinationGard.
    const double *dloss_dstress = nullptr;

    double *gradient = nullptr;
    int gradient_count = 0;
    bool accumulate_gradient = false;
    // Chunked training can retain the previous chunk's gradient on the
    // device and defer the only D2H copy until the final chunk.
    bool accumulate_gradient_on_device = false;
    bool copy_gradient_to_host = true;
    // Use adjoints produced by prepare_loss() without uploading host arrays.
    bool adjoints_on_device = false;

    // The fast mode uses parallel forward, loss, directional, and parameter
    // reductions. Deterministic mode orders all of those reductions for
    // byte-repeatable diagnostics and is intentionally much slower.
    bool deterministic_reduction = false;
    int radial_parameter_tile = 128;

    int basic_tile = 128;
    int neighbor_lanes = 8;
    int basic_lanes = 16;
    int propagate_threads = 256;
};

// Supervised E/F/S labels and weights for constructing loss adjoints directly
// from the resident forward outputs. Metrics are returned in this order:
// total loss, energy loss, force loss, stress loss, squared per-atom energy
// error, squared force error, squared stress error.
struct MTPGpuLossRequest {
    static constexpr int metric_count = 7;

    int structure_count = 0;
    const int *structure_offsets = nullptr;
    const int *center_structure = nullptr;
    const double *target_energy = nullptr;
    const double *energy_weight = nullptr;
    const double *target_force = nullptr;
    const double *force_weight = nullptr;
    const double *target_stress = nullptr;
    const double *stress_weight = nullptr;
    double *metrics = nullptr;
    bool reset_metrics = true;
    bool copy_metrics_to_host = true;
    bool deterministic_reduction = false;
    GpuCacheToken labels_token;
};

struct MTPGpuRuntimeStats {
    std::uint64_t buffer_reallocations = 0;
    std::uint64_t chunks_executed = 0;
    std::uint64_t output_download_bytes = 0;
    int last_centers_per_chunk = 0;
    int last_chunk_count = 0;
    std::uint64_t topology_uploads = 0;
    std::uint64_t topology_reuses = 0;
    std::uint64_t pipelined_input_uploads = 0;
    std::uint64_t topology_upload_bytes = 0;
    std::uint64_t type_uploads = 0;
    std::uint64_t type_upload_bytes = 0;
    std::uint64_t coordinate_uploads = 0;
    std::uint64_t coordinate_upload_bytes = 0;
    std::uint64_t basic_tile_force_launches = 0;
    std::uint64_t center_neighbors_force_launches = 0;
    std::uint64_t input_reuses = 0;
    std::uint64_t label_reuses = 0;
    std::uint64_t label_upload_bytes = 0;
    std::uint64_t sog_grid_upload_bytes = 0;
    std::uint64_t sog_grid_reuses = 0;
};

struct MTPGpuPotentialStats {
    MTPGpuPrecision precision = MTPGpuPrecision::FP64;
    int declared_moment_slots = 0;
    int active_moment_slots = 0;
    int forward_block_groups = 0;
    int reverse_block_groups = 0;
    int neural_cache_doubles_per_center = 0;
    int moment_element_bytes = static_cast<int>(sizeof(double));
    int adjoint_element_bytes = static_cast<int>(sizeof(double));
    std::size_t potential_bytes = 0;
};

// Actual allocated GPU bytes. Workspaces grow lazily, so values obtained
// immediately after initialize() contain the resident potential but no
// per-batch storage yet.
struct MTPGpuMemoryStats {
    MTPGpuPrecision precision = MTPGpuPrecision::FP64;
    std::size_t potential_bytes = 0;
    std::size_t topology_bytes = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t moment_workspace_bytes = 0;
    std::size_t adjoint_workspace_bytes = 0;
    std::size_t directional_workspace_bytes = 0;
    std::size_t neural_cache_bytes = 0;
    std::size_t loss_gradient_bytes = 0;
    std::size_t conversion_buffer_bytes = 0;
    std::size_t tracked_device_bytes = 0;
};

struct MTPGpuDeviceInfo {
    std::string name;
    std::string architecture;
    std::string identity;
    std::size_t total_memory = 0;
    int wave_size = 0;
    int compute_units = 0;
};

class MTPGpuDirect {
public:
    MTPGpuDirect();
    ~MTPGpuDirect();

    MTPGpuDirect(const MTPGpuDirect &) = delete;
    MTPGpuDirect &operator=(const MTPGpuDirect &) = delete;

    // Select the CUDA device owned by this context.  A negative id keeps the
    // device selected by the caller/runtime.  This must be called before
    // initialize().
    bool set_device(int device_id, std::string *error = nullptr);
    static bool available_device_count(int *count, std::string *error = nullptr);

    static const char *backend_name();
    static bool device_info(int device, MTPGpuDeviceInfo *info,
                            std::string *error = nullptr);

    bool initialize(const MTPGpuPotentialView &potential, std::string *error = nullptr);
    bool initialize(const MTPGpuPotentialView &potential,
                    const MTPGpuInitializationOptions &options,
                    std::string *error = nullptr);
    bool initialize_from_mlip(const mlip &potential, std::string *error = nullptr);
    bool initialize_from_mlip(const mlip &potential,
                              const MTPGpuInitializationOptions &options,
                              std::string *error = nullptr);
    bool update_coefficients(const double *radial_coeffs,
                             const double *species_coeffs,
                             const double *moment_coeffs,
                             std::string *error = nullptr);
    bool update_coefficients(const double *radial_coeffs,
                             const double *species_coeffs,
                             const double *moment_coeffs,
                             const double *descriptor_species_coeffs,
                             const double *descriptor_moment_coeffs,
                             const double *model_coeffs,
                             std::string *error = nullptr);
    bool memory_info(std::size_t *free_bytes, std::size_t *total_bytes,
                     std::string *error = nullptr) const;
    bool configure_sog(const MTPGpuSOGView &, std::string *error = nullptr);
    // Monotonically changes whenever a different centers/offsets/neighbors
    // topology is uploaded. Adapters use this to avoid stale reuse when
    // multiple batches share one evaluator.
    std::uint64_t topology_generation() const;
    std::uint64_t context_identity() const;
    bool has_input(GpuCacheToken token) const;
    void invalidate_forward();
    MTPGpuRuntimeStats runtime_stats() const;
    MTPGpuPrecision precision() const;
    // The CUDA core compacts gaps in the potential's moment numbering.  This
    // reports the file-level slot count and the resident contiguous count so
    // callers can size batches against the actual device workspace.
    MTPGpuPotentialStats potential_stats() const;
    MTPGpuMemoryStats memory_stats() const;
    bool compute_batch(const MTPGpuBatchRequest &request, std::string *error = nullptr);
    bool prepare_loss(const MTPGpuLossRequest &request,
                      std::string *error = nullptr);
    bool accumulate_gradient(const MTPGpuGradientRequest &request,
                             std::string *error = nullptr);
    bool compute_lammps(const MTPGpuComputeRequest &request, std::string *error = nullptr);

private:
    bool compute_impl(const MTPGpuComputeRequest &request,
                      const MTPGpuBatchRequest *batch, std::string *error);
    struct State;
    State *state_ = nullptr;
};

} // namespace GpuDirect
} // namespace MLIPCore
