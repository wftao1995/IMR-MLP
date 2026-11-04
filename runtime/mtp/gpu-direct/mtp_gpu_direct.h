// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_backend.h"
#include "sog_gpu.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace MLIPCore
{
struct mlip;

namespace GpuDirect
{

enum class MTPGpuPrecision
{
    FP64 = 0,
    Mixed = 1,
    MixedAggressive = 2,
    MixedCompute = 3,
    FP32 = 4,
};

enum class MTPGpuForceKernel
{
    Auto = 0,
    BasicTile = 1,
    CenterNeighbors = 2,
};

inline constexpr const char *force_kernel_name(MTPGpuForceKernel kernel)
{
    switch (kernel)
    {
    case MTPGpuForceKernel::Auto:
        return "auto";
    case MTPGpuForceKernel::BasicTile:
        return "basic-tile";
    case MTPGpuForceKernel::CenterNeighbors:
        return "center-neighbors";
    }
    return "unknown";
}

inline bool parse_force_kernel(std::string_view value, MTPGpuForceKernel *kernel)
{
    if (kernel == nullptr)
        return false;
    if (value == "auto")
    {
        *kernel = MTPGpuForceKernel::Auto;
        return true;
    }
    if (value == "basic-tile")
    {
        *kernel = MTPGpuForceKernel::BasicTile;
        return true;
    }
    if (value == "center-neighbors")
    {
        *kernel = MTPGpuForceKernel::CenterNeighbors;
        return true;
    }
    return false;
}

inline constexpr const char *precision_name(MTPGpuPrecision precision)
{
    switch (precision)
    {
    case MTPGpuPrecision::FP64:
        return "fp64";
    case MTPGpuPrecision::Mixed:
        return "mixed";
    case MTPGpuPrecision::MixedAggressive:
        return "mixed-aggressive";
    case MTPGpuPrecision::MixedCompute:
        return "mixed-compute";
    case MTPGpuPrecision::FP32:
        return "fp32";
    }
    return "unknown";
}

inline bool parse_precision(std::string_view value, MTPGpuPrecision *precision)
{
    if (precision == nullptr)
        return false;
    if (value == "fp64")
    {
        *precision = MTPGpuPrecision::FP64;
        return true;
    }
    if (value == "mixed")
    {
        *precision = MTPGpuPrecision::Mixed;
        return true;
    }
    if (value == "mixed-aggressive")
    {
        *precision = MTPGpuPrecision::MixedAggressive;
        return true;
    }
    return false;
}

inline bool fp32_compute(MTPGpuPrecision p)
{
    return p == MTPGpuPrecision::MixedCompute || p == MTPGpuPrecision::FP32;
}

inline const char *precision_description(MTPGpuPrecision p)
{
    if (p == MTPGpuPrecision::FP32)
        return "geometry=fp32 compute=fp32 moment=fp32 adjoint=fp32 accumulate=fp32 long_range=fp64 experimental";
    if (p == MTPGpuPrecision::MixedCompute)
        return "geometry=fp64 compute=fp32 moment=fp32 adjoint=fp32 accumulate=fp64 long_range=fp64 experimental";
    if (p == MTPGpuPrecision::Mixed)
        return "geometry=fp64 compute=fp64 moment=fp32 adjoint=fp64 accumulate=fp64 long_range=fp64";
    if (p == MTPGpuPrecision::MixedAggressive)
        return "geometry=fp64 compute=fp64 moment=fp32 adjoint=fp32 accumulate=fp64 long_range=fp64 experimental";
    return "geometry=fp64 compute=fp64 moment=fp64 adjoint=fp64 accumulate=fp64 long_range=fp64";
}

struct MTPGpuInitializationOptions
{
    MTPGpuPrecision precision = MTPGpuPrecision::FP64;
    // Validate model upload before allocation; compute requests carry the
    // corresponding total resident-memory budget for dynamic workspaces.
    std::size_t memory_budget_bytes = 0;
    // Persistent cap for an ordinary FP64 training owner, including model,
    // forward caches, labels, prepared adjoints and backward workspaces.
    // Fixed before initialization; zero preserves the legacy unbounded owner.
    std::size_t training_memory_budget_bytes = 0;
};

struct MTPGpuSOGView
{
    SOGGpuModelView model;
    const double *species_coeffs = nullptr;
    const double *moment_coeffs = nullptr;
    const double *bandwidth_jacobian = nullptr; // du/dz, one per Gaussian
};

enum class RadialBasis
{
    Chebyshev = 0,
    ChebyCos = 1,
    ChebyshevPoly5 = 2,
    ChebyshevConfigurable = 3,
    ChebyshevPoly4 = 4,
    PolyB = 5,
    PolyBCos = 6,
};

enum class RadialCutoff
{
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

enum class RadialTransform
{
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
enum class MTPGpuModel
{
    Linear = 0,
    MTPMQuadratic = 1,
    MTPMAsinh = 2,
    MTPMNN1 = 3,
    MTPMNN2 = 4,
    MTPMLinear = 5,
};

enum class MTPGpuActivation
{
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
struct MTPGpuRadialBasisView
{
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

struct MTPGpuPotentialView
{
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
    // Legacy NN inference gathers scalar moments directly, with O(K) storage.
    bool scalar_gather = false;
    const double *input_mean = nullptr, *input_inverse_std = nullptr, *input_species = nullptr;
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
    int species_count, int max_type, const int *spmap, int alpha_moments_count,
    int alpha_index_basic_count, const int *alpha_index_basic, int alpha_index_times_count,
    const int *alpha_index_times, int alpha_scalar_moments, const int *alpha_moment_mapping,
    int radial_func_count, int rb_size, int max_mu, int max_alpha_index_basic, double min_dist,
    double max_dist, double cutoff, double scaling, const double *radial_coeffs,
    const double *linear_coeffs, const int *nu_arr_list = nullptr,
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

// Borrowed device geometry is shared by all heads in a family execution plan.
// The owner keeps these arrays alive until all default-stream work completes.
struct MTPGpuDeviceInput
{
    const int *centers = nullptr, *offsets = nullptr, *neighbors = nullptr, *type = nullptr;
    const double *positions = nullptr, *neighbor_vectors = nullptr;
    int centers_count = 0, total = 0, edges = 0;
    bool explicit_neighbor_vectors = false;
};

struct MTPGpuMomentView
{
    // Center-major, compacted FP64 moments, valid until the owner's next call.
    // The caller must use the same graph, scalar mapping and center ordering.
    const double *values = nullptr;
    int centers = 0, stride = 0;
    const float *values_fp32 = nullptr;
    MTPGpuPrecision precision = MTPGpuPrecision::FP64;

    bool present() const
    {
        return values || values_fp32;
    }
};

struct MTPGpuDeviceOutput
{
    double *force = nullptr, *site_energy = nullptr, *site_virial = nullptr, *energy = nullptr,
           *virial = nullptr;
    MTPGpuMomentView moments;
};

struct MTPGpuComputeRequest
{
    const MTPGpuDeviceInput *device_input = nullptr;
    MTPGpuMomentView device_moments;
    const double *device_center_energy_weights = nullptr;
    // Optional borrowed ntotal buffer for unweighted linear site energies.
    // Requires FP64 inference and compute_site_energy; owned centers are written.
    double *device_unweighted_site_energy = nullptr;
    // Atom-major response and channel-major scalar coefficients; device arrays.
    const double *device_response = nullptr, *device_response_coefficients = nullptr;
    int response_channels = 0;
    // Optional atom-major projection emitted before each center chunk is reused.
    double *device_projection = nullptr;
    const double *device_projection_species = nullptr, *device_projection_coefficients = nullptr;
    int projection_channels = 0;

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
    int centers_per_chunk = 0;           // zero selects all centers (or budget-derived size)
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
struct MTPGpuBatchRequest
{
    const MTPGpuDeviceInput *device_input = nullptr;
    MTPGpuMomentView device_moments;
    const double *device_center_energy_weights = nullptr;
    // Same optional linear-head output contract as MTPGpuComputeRequest.
    double *device_unweighted_site_energy = nullptr;
    // Atom-major response and channel-major scalar coefficients; device arrays.
    const double *device_response = nullptr, *device_response_coefficients = nullptr;
    int response_channels = 0;
    // Optional atom-major projection emitted before each center chunk is reused.
    double *device_projection = nullptr;
    const double *device_projection_species = nullptr, *device_projection_coefficients = nullptr;
    int projection_channels = 0;

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
    bool inference_only = false;
    int centers_per_chunk = 0;
    std::size_t memory_budget_bytes = 0;
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
struct MTPGpuGradientRequest
{
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
    GpuCacheToken prepared_loss_token;

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
struct MTPGpuLossRequest
{
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

// Two complete ordered structures: plus then minus. The host target contains
// 3*N components; q=-(Fplus-Fminus)/(2*epsilon), loss=weight*sum((q-target)^2).
struct MTPGpuHVPForcePairLossRequest
{
    int atoms_per_structure = 0;
    const int *structure_offsets = nullptr;
    const int *center_structure = nullptr;
    const double *target = nullptr;
    std::size_t target_count = 0;
    double epsilon = 0.;
    double observation_weight = 0.;
    bool deterministic_reduction = false;
    GpuCacheToken input_token, labels_token;
};

struct MTPGpuHVPPairMetrics
{
    double loss = 0.;
    double squared_error = 0.;
};

// One fixed-cell Cartesian direction. The host product is caller-owned scratch:
// a failed download may partially write it. Public adapters publish by swap.
struct MTPGpuAnalyticHvpRequest
{
    const double *direction = nullptr;
    std::size_t direction_count = 0;
    double *product = nullptr;
    std::size_t product_count = 0;
};

// Selected native linear columns of H*v, with the same complete fixed-cell
// geometry as the contracted analytic request. Native columns are species
// offsets followed by scalar moments; species columns have zero curvature.
// Rows select [first_row, first_row + row_count) from the flattened Cartesian
// 3N product. Column order and duplicates are preserved. Output is unweighted
// row-major row_count x column_count, and is caller-owned scratch on failure.
struct MTPGpuLinearHvpRequest
{
    const double *direction = nullptr;
    std::size_t direction_count = 0;
    int first_row = 0, row_count = 0;
    const int *columns = nullptr;
    int column_count = 0;
    double *output = nullptr;
    std::size_t output_count = 0;
};

struct MTPGpuRuntimeStats
{
    std::uint64_t linear_hvp_calls = 0;
    std::uint64_t linear_hvp_direction_upload_bytes = 0;
    std::uint64_t linear_hvp_output_download_bytes = 0;
    std::uint64_t linear_hvp_diagnostic_download_bytes = 0;
    std::uint64_t linear_hvp_basic_directional_launches = 0;
    std::uint64_t linear_hvp_propagation_launches = 0;
    std::uint64_t linear_hvp_product_launches = 0;
    std::uint64_t linear_hvp_duplicate_copy_launches = 0;
    std::uint64_t analytic_hvp_calls = 0;
    std::uint64_t analytic_direction_upload_bytes = 0;
    std::uint64_t analytic_output_download_bytes = 0;
    std::uint64_t analytic_diagnostic_download_bytes = 0;
    std::uint64_t analytic_basic_directional_launches = 0;
    std::uint64_t analytic_propagation_launches = 0;
    std::uint64_t analytic_product_launches = 0;
    std::uint64_t buffer_reallocations = 0;
    std::uint64_t moment_reuses = 0;
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
    std::uint64_t linear_resident_blocks = 0;
    std::uint64_t linear_score_blocks = 0;
    std::uint64_t linear_host_row_uploads = 0;
    std::uint64_t linear_host_row_upload_bytes = 0;
    std::uint64_t linear_factor_uploads = 0;
    std::uint64_t linear_factor_upload_bytes = 0;
    std::uint64_t linear_observation_download_bytes = 0;
    std::uint64_t linear_score_download_bytes = 0;
    std::uint64_t linear_diagnostic_download_bytes = 0;
    std::uint64_t linear_candidate_download_bytes = 0;
    std::uint64_t hvp_pair_loss_calls = 0;
    std::uint64_t hvp_target_upload_bytes = 0;
    std::uint64_t hvp_metric_download_bytes = 0;
    std::uint64_t training_diagnostic_download_bytes = 0;
    std::uint64_t training_gradient_download_bytes = 0;
    std::uint64_t training_force_download_bytes = 0;
    std::uint64_t training_force_adjoint_upload_bytes = 0;
};

struct MTPGpuPotentialStats
{
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
enum class MTPGpuLinearDerivative
{
    Force,
    Virial
};
enum class MTPGpuLinearKind
{
    Energy,
    Site,
    Force,
    Virial,
    Host
};

// Supplied CPU factorization only: no rank/regularization decisions on device.
// All dense arrays are row-major. Native columns are an exact layout identity,
// including order and repetitions; host-matrix rows may use arbitrary IDs.
enum class MTPGpuScoreFormula
{
    Square,
    MaxVolWide,
    MaxVolTall,
    OnlineGram
};

struct MTPGpuScoreFactor
{
    MTPGpuScoreFormula formula = MTPGpuScoreFormula::Square;
    std::uint64_t owner_identity = 0, generation = 0;
    int active_rows = 0;
    std::vector<int> native_columns;
    std::vector<double> active, coefficient_map, gram_inverse;
    std::vector<double> gram_diagonal, active_leverage;
};

struct MTPGpuLinearRows
{
    std::uint64_t context_identity = 0, generation = 0;
    GpuCacheToken input;
    int rows = 0, columns = 0;
    bool host_source = false;
};

struct MTPGpuScoreToken
{
    std::uint64_t context_identity = 0, owner_identity = 0, generation = 0;
};

struct MTPGpuScoreDiagnostics
{
    double row_norm2 = 0., projection = 0., residual2_unclamped = 0., candidate_leverage = 0.;
};

struct MTPGpuMemoryStats
{
    std::size_t parameter_copy_bytes = 0; // included in potential_bytes
    MTPGpuPrecision precision = MTPGpuPrecision::FP64;
    std::size_t potential_bytes = 0;
    std::size_t topology_bytes = 0;
    std::size_t input_bytes = 0;
    std::size_t output_bytes = 0;
    std::size_t moment_workspace_bytes = 0;
    std::size_t adjoint_workspace_bytes = 0;
    std::size_t directional_workspace_bytes = 0;
    std::size_t analytic_hvp_io_bytes =
        0; // one 3N direction and one product or selected row/column block
    std::size_t neural_cache_bytes = 0;
    std::size_t loss_gradient_bytes = 0;
    std::size_t conversion_buffer_bytes = 0;
    std::size_t linear_observation_bytes = 0;
    std::size_t linear_factor_bytes = 0;
    std::size_t linear_score_workspace_bytes = 0;
    std::size_t tracked_device_bytes = 0;
    std::size_t training_memory_budget_bytes = 0;
    std::size_t training_peak_device_bytes = 0;
};

struct MTPGpuDeviceInfo
{
    std::string name;
    std::string architecture;
    std::string identity;
    std::size_t total_memory = 0;
    int wave_size = 0;
    int compute_units = 0;
};

class MTPGpuDirect
{
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
    static bool device_info(int device, MTPGpuDeviceInfo *info, std::string *error = nullptr);

    bool initialize(const MTPGpuPotentialView &potential, std::string *error = nullptr);
    bool initialize(const MTPGpuPotentialView &potential,
                    const MTPGpuInitializationOptions &options, std::string *error = nullptr);
    bool initialize_from_mlip(const mlip &potential, std::string *error = nullptr);
    bool initialize_from_mlip(const mlip &potential, const MTPGpuInitializationOptions &options,
                              std::string *error = nullptr);
    bool update_coefficients(const double *radial_coeffs, const double *species_coeffs,
                             const double *moment_coeffs, std::string *error = nullptr);
    bool update_coefficients(const double *radial_coeffs, const double *species_coeffs,
                             const double *moment_coeffs, const double *descriptor_species_coeffs,
                             const double *descriptor_moment_coeffs, const double *model_coeffs,
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
    bool has_linear_energy_input(GpuCacheToken token) const;
    // Unweighted native linear columns, row-major; offsets index resident
    // centers. Repeated/reordered columns are permitted. Budget covers this
    // projection's device buffer, separately from the resident forward pass.
    bool linear_energy_block(GpuCacheToken token, const int *offsets, int rows, const int *columns,
                             int column_count, double *output, size_t workspace_budget_bytes,
                             std::string *error = nullptr);
    // Force rows: 3*atom+axis. Virial rows: 9*structure+3*a+b, raw -dE/dstrain.
    // Same prepared moments as linear_energy_block; one-column reverse scratch
    // and the requested row-major output are both charged to the block budget.
    bool linear_derivative_block(GpuCacheToken token, MTPGpuLinearDerivative kind,
                                 const int *structure_offsets, int structures, int first_row,
                                 int rows, const int *columns, int column_count, double *output,
                                 size_t workspace_budget_bytes, std::string *error = nullptr);
    // Synchronous resident preparation: no numerical B values downloaded.
    // A token expires on next row preparation, input/coefficient changes or
    // failure. total_cap=0 preserves the legacy host wrapper's independent cap.
    bool prepare_linear_energy_device(GpuCacheToken input, const int *offsets, int rows,
                                      const int *columns, int column_count, MTPGpuLinearRows &token,
                                      size_t workspace_cap, size_t total_cap,
                                      std::string *error = nullptr);
    bool prepare_linear_derivative_device(GpuCacheToken input, MTPGpuLinearDerivative kind,
                                          const int *structure_offsets, int structures,
                                          int first_row, int rows, const int *columns,
                                          int column_count, MTPGpuLinearRows &token,
                                          size_t workspace_cap, size_t total_cap,
                                          std::string *error = nullptr);
    // Explicit materialized-row path, separately counted from resident physics.
    // Does not require an initialized potential; select the device first.
    bool upload_linear_rows(const std::vector<double> &rows, int row_count,
                            const std::vector<int> &native_columns, MTPGpuLinearRows &token,
                            size_t workspace_cap, size_t total_cap, std::string *error = nullptr);
    bool upload_score_factor(const MTPGpuScoreFactor &, MTPGpuScoreToken &, size_t total_cap,
                             std::string *error = nullptr);
    // Free transient row/products arenas while retaining factor and forward
    // storage. Useful before reserving workspace in the next forward budget.
    bool release_linear_workspaces(std::string *error = nullptr);
    // Exact allocation sizes for a fresh block, without allocating or changing
    // tokens. Physics kinds require the current complete FP64 forward batch.
    bool estimate_linear_workspace(MTPGpuLinearKind kind, int structures, int rows,
                                   const std::vector<int> &columns, int active_rows,
                                   MTPGpuScoreFormula formula, bool diagnostics,
                                   size_t &observation_bytes, size_t &score_bytes,
                                   std::string *error = nullptr) const;
    // scales empty / one / rows means unit / uniform / per-row scale.
    // workspace_cap bounds the sum of retained observation and scoring arenas;
    // total_cap also includes the factor, parameters, geometry and forward state.
    bool score_linear_block(const MTPGpuLinearRows &, const MTPGpuScoreToken &,
                            const std::vector<double> &scales, std::vector<double> &scores,
                            size_t total_cap, size_t workspace_cap, std::string *error = nullptr,
                            std::vector<MTPGpuScoreDiagnostics> *diagnostics = nullptr);
    // Only explicitly selected raw rows are copied, in requested order.
    bool download_linear_rows(const MTPGpuLinearRows &, const std::vector<int> &selected,
                              std::vector<double> &output, std::string *error = nullptr);
    void invalidate_forward();
    // An adapter-side rejected training call also discards accumulated gradients.
    void invalidate_training_state();
    MTPGpuRuntimeStats runtime_stats() const;
    MTPGpuPrecision precision() const;
    // The CUDA core compacts gaps in the potential's moment numbering.  This
    // reports the file-level slot count and the resident contiguous count so
    // callers can size batches against the actual device workspace.
    MTPGpuPotentialStats potential_stats() const;
    MTPGpuMemoryStats memory_stats() const;
    bool compute_batch(const MTPGpuBatchRequest &request, std::string *error = nullptr);
    // Ordinary FP64 linear model, complete ordered geometry and positive
    // aggregate memory_budget_bytes. Streams centers; no finite differences.
    bool compute_analytic_hvp(const MTPGpuBatchRequest &geometry,
                              const MTPGpuAnalyticHvpRequest &hvp, std::string *error = nullptr);
    // Streams centers, sharing primal/tangent moments across selected columns.
    // Uses the same positive aggregate geometry budget as analytic HVP. No
    // dense coordinate Hessian, per-edge Jacobian, or model coefficient edits.
    bool compute_linear_hvp_block(const MTPGpuBatchRequest &geometry,
                                  const MTPGpuLinearHvpRequest &block,
                                  std::string *error = nullptr);
    bool prepare_loss(const MTPGpuLossRequest &request, std::string *error = nullptr);
    bool prepare_hvp_pair_loss(const MTPGpuHVPForcePairLossRequest &request,
                               MTPGpuHVPPairMetrics &metrics, GpuCacheToken &prepared_token,
                               std::string *error = nullptr);
    bool accumulate_gradient(const MTPGpuGradientRequest &request, std::string *error = nullptr);
    MTPGpuDeviceOutput device_outputs() const;
    bool project_channels(const MTPGpuDeviceInput &, int channels, const double *species,
                          const double *coefficients, double *output, std::string *error = nullptr);
    bool compute_lammps(const MTPGpuComputeRequest &request, std::string *error = nullptr);

  private:
    struct AnalyticHvpOperation;
    bool compute_analytic_hvp_impl(const MTPGpuBatchRequest &geometry,
                                   const AnalyticHvpOperation &operation, std::string *error);
    bool compute_batch_impl(const MTPGpuBatchRequest &request, const AnalyticHvpOperation *hvp,
                            std::string *error);
    bool compute_impl(const MTPGpuComputeRequest &request, const MTPGpuBatchRequest *batch,
                      std::string *error, const AnalyticHvpOperation *hvp = nullptr);
    struct State;
    State *state_ = nullptr;
};

} // namespace GpuDirect
} // namespace MLIPCore
