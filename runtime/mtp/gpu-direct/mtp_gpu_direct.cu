// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtp_gpu_direct.h"
#include "radial_gpu_math.h"

#include "../mlip.h"

#include "gpu_runtime.h"

#ifdef MTP_GPU_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <type_traits>
#include <vector>

// Independent experimental builds: 0 = legacy, 1 = partitioned groups,
// 2 = partitioned + classic-linear FP64; 3 = classic-linear only.
#ifndef MTP_GPU_PROPAGATION_VARIANT
#if defined(IMRMLP_HIP)
#define MTP_GPU_PROPAGATION_VARIANT 1
#else
#define MTP_GPU_PROPAGATION_VARIANT 2
#endif
#endif
static_assert(MTP_GPU_PROPAGATION_VARIANT >= 0 && MTP_GPU_PROPAGATION_VARIANT <= 3);

namespace MLIPCore::GpuDirect {
using namespace Runtime;
namespace {

constexpr double kPi = 3.141592653589793238462643383279502884;

class NvtxRange {
public:
    explicit NvtxRange(const char *name)
    {
#ifdef MTP_GPU_ENABLE_NVTX
        nvtxRangePushA(name);
#else
        (void) name;
#endif
    }
    ~NvtxRange()
    {
#ifdef MTP_GPU_ENABLE_NVTX
        nvtxRangePop();
#endif
    }
    NvtxRange(const NvtxRange &) = delete;
    NvtxRange &operator=(const NvtxRange &) = delete;
};

// Fixed reduction order, independent of force-kernel scheduling. The input is
// the same symmetric site virial used by the per-atom output path.
__global__ void global_virial_kernel(int count, const double *sites, double *total)
{
    __shared__ double partial[128];
    double value = 0.;
    for (int atom = threadIdx.x; atom < count; atom += blockDim.x)
        value += sites[6 * atom + blockIdx.x];
    partial[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride; stride /= 2) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) total[blockIdx.x] = partial[0];
}

struct DeviceRadialBasis {
    int rb_type = 0;
    int cutoff_type = 0;
    int transform_type = 0;
    double min_dist = 0.0;
    double min_distsq = 0.0;
    double max_dist = 0.0;
    double cutoffsq = 0.0;
    double scaling = 1.0;
    double plus_r = 0.0;
    double sub_r = 1.0;
    double mult = 1.0;
    double irpi = 0.0;
    double cutoff_parameter = 0.0;
    double transform_parameter = 0.0;
    double transform_denominator = 0.0;
    bool one_minus_chebyshev = false;
    double cutoff_r1 = 0.0, cutoff_r2 = 0.0;
    double transform_scale_parameter = 0.0;
    size_t polynomial_offset = 0;
};

// Execution-ordered terms avoid an extra schedule-index load in hot loops.
struct ForwardTerm { int l1, l2, c; };
struct ReverseTerm { int source, other, c; };

struct DevicePotential {
    int species_count = 0;
    int max_type = 0;
    // A single potential has a compact (int-sized) moment schedule, but the
    // flattened center x moment workspace can exceed INT_MAX elements.
    // Keeping the stride unsigned widens all device pointer arithmetic.
    size_t alpha_moments_count = 0;
    int alpha_index_basic_count = 0;
    int alpha_index_times_count = 0;
    int alpha_scalar_moments = 0;
    int radial_func_count = 0;
    int rb_size = 0;
    int max_mu = 0;
    int max_alpha_index_basic = 0;
    int coord_power_count = 0;
    int max_level = 0;
    int rb_type = 0;
    int pair_radial_basis_count = 0;
    int model_type = 0;
    int model_activation = 0;
    int model_hidden1 = 0;
    int model_hidden2 = 0;
    int descriptor_count = 0;
    int model_coeffs_count = 0;
    int nonradial_parameter_count = 0;
    bool unique_scalar_mapping = true;
    bool inference_only = false;
    double *global_virial = nullptr; // borrowed per-call six-component output
    const double *center_energy_weights = nullptr; // borrowed per-call buffer

    double min_dist = 0.0;
    double max_dist = 0.0;
    double cutoff = 0.0;
    double cutoffsq = 0.0;
    double scaling = 1.0;
    double plus_r = 0.0;
    double sub_r = 1.0;
    double mult = 1.0;
    double irpi = 0.0;
    double p0 = 1.0;
    double pp = 1.0;

    int *spmap = nullptr;
    int *basic_mu = nullptr;
    int *basic_b1 = nullptr;
    int *basic_b2 = nullptr;
    int *basic_b3 = nullptr;
    int *basic_nu = nullptr;
    int *basic_mu_offsets = nullptr;
    int *basic_mu_indices = nullptr;
    int *times_l1 = nullptr;
    int *times_l2 = nullptr;
    int *times_c = nullptr;
    int *times_id = nullptr;
    int *level_offsets = nullptr;
    int *forward_level_offsets = nullptr;
    int *forward_ids = nullptr;
    int *forward_term_offsets = nullptr;
    ForwardTerm *forward_terms = nullptr;
    int *forward_block_level_offsets = nullptr;
    int *forward_block_groups = nullptr;
    int *reverse_level_offsets = nullptr;
    int *reverse_ids = nullptr;
    int *reverse_entry_offsets = nullptr;
    ReverseTerm *reverse_entries = nullptr;
    int *reverse_block_level_offsets = nullptr;
    int *reverse_block_groups = nullptr;
    int *forward_short_level_offsets = nullptr;
    int *forward_short_groups = nullptr;
    int *forward_warp_level_offsets = nullptr;
    int *forward_warp_groups = nullptr;
    int *reverse_short_level_offsets = nullptr;
    int *reverse_short_groups = nullptr;
    int *reverse_warp_level_offsets = nullptr;
    int *reverse_warp_groups = nullptr;
    int *alpha_moment_mapping = nullptr;
    DeviceRadialBasis *pair_radial_bases = nullptr;
    double *radial_basis_parameters = nullptr;
    double *radial_coeffs = nullptr;
    double *species_coeffs = nullptr;
    double *moment_coeffs = nullptr;
    double *descriptor_species_coeffs = nullptr;
    double *descriptor_moment_coeffs = nullptr;
    double *model_coeffs = nullptr;
};

struct HostSchedule {
    int max_level = 0;
    std::vector<int> level_offsets;
    std::vector<int> times_l1;
    std::vector<int> times_l2;
    std::vector<int> times_c;
    std::vector<int> times_id;
    std::vector<int> forward_level_offsets;
    std::vector<int> forward_ids;
    std::vector<int> forward_term_offsets;
    std::vector<int> forward_terms;
    std::vector<int> forward_block_level_offsets;
    std::vector<int> forward_block_groups;
    std::vector<int> reverse_level_offsets;
    std::vector<int> reverse_ids;
    std::vector<int> reverse_entry_offsets;
    std::vector<int> reverse_entries;
    std::vector<int> reverse_block_level_offsets;
    std::vector<int> reverse_block_groups;
    std::vector<int> forward_short_level_offsets;
    std::vector<int> forward_short_groups;
    std::vector<int> forward_warp_level_offsets;
    std::vector<int> forward_warp_groups;
    std::vector<int> reverse_short_level_offsets;
    std::vector<int> reverse_short_groups;
    std::vector<int> reverse_warp_level_offsets;
    std::vector<int> reverse_warp_groups;
};

constexpr int propagation_warp_size = 32;
constexpr int propagation_warp_group_threshold = 32;
// Real MTM-new bases top out below this limit except for the irregular basis,
// whose reverse schedule contains groups as large as 4216 entries.  Give two
// warps to those extreme groups while retaining multiple concurrent groups in
// a block, so regular/reference bases keep the lower-synchronization path.
constexpr int propagation_block_group_threshold = 512;
constexpr int propagation_multiwarp_group_warps = 2;

struct HostPotential {
    DevicePotential dims;
    std::vector<int> spmap;
    std::vector<int> basic_mu;
    std::vector<int> basic_b1;
    std::vector<int> basic_b2;
    std::vector<int> basic_b3;
    std::vector<int> basic_nu;
    std::vector<int> basic_mu_offsets;
    std::vector<int> basic_mu_indices;
    std::vector<int> alpha_moment_mapping;
    std::vector<DeviceRadialBasis> pair_radial_bases;
    std::vector<double> radial_basis_parameters;
    HostSchedule schedule;
};

void set_error(std::string *error, const std::string &message)
{
    if (error != nullptr) {
        *error = message;
    }
}

bool check_gpu(gpuError_t err, const char *expr, std::string *error)
{
    if (err == gpuSuccess) {
        return true;
    }
    std::ostringstream oss;
    oss << expr << " failed: " << gpuGetErrorString(err);
    set_error(error, oss.str());
    return false;
}

#define MTP_GPU_CHECK(expr) \
    do { \
        if (!check_gpu((expr), #expr, error)) { \
            return false; \
        } \
    } while (0)

template <typename T>
bool upload_array(T **dst, const T *src, int count, std::string *error)
{
    if (count <= 0) {
        *dst = nullptr;
        return true;
    }
    MTP_GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(dst), count * sizeof(T)));
    MTP_GPU_CHECK(gpuMemcpy(*dst, src, count * sizeof(T), gpuMemcpyHostToDevice));
    return true;
}

template <typename T>
bool upload_vector(T **dst, const std::vector<T> &src, std::string *error)
{
    return upload_array(dst, src.data(), static_cast<int>(src.size()), error);
}

template <typename T>
bool ensure_capacity(T **ptr, size_t *capacity, size_t needed,
                     std::string *error)
{
    if (needed <= *capacity) {
        return true;
    }
    if (needed > std::numeric_limits<size_t>::max() / sizeof(T)) {
        set_error(error, "GPU buffer size overflows size_t");
        return false;
    }
    const size_t old_capacity = *capacity;
    if (*ptr != nullptr) {
        MTP_GPU_CHECK(gpuFree(*ptr));
        *ptr = nullptr;
        *capacity = 0;
    }
    size_t new_capacity = std::max<size_t>(needed, 1);
    // Grow reusable buffers geometrically, but cap spare capacity at 64 MiB
    // per allocation so million-atom workspaces do not lose useful capacity
    // to a large multiplicative reserve.
    if (old_capacity > 0) {
        const size_t half = old_capacity / 2;
        const size_t geometric = old_capacity <=
                std::numeric_limits<size_t>::max() - half
            ? old_capacity + half
            : std::numeric_limits<size_t>::max();
        constexpr size_t max_spare_bytes = size_t{64} << 20;
        const size_t max_spare_elements =
            std::max<size_t>(1, max_spare_bytes / sizeof(T));
        const size_t spare_cap = needed <=
                std::numeric_limits<size_t>::max() - max_spare_elements
            ? needed + max_spare_elements
            : std::numeric_limits<size_t>::max();
        new_capacity = std::max(needed, std::min(geometric, spare_cap));
    }
    if (new_capacity > std::numeric_limits<size_t>::max() / sizeof(T)) {
        new_capacity = needed;
    }
    MTP_GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(ptr), new_capacity * sizeof(T)));
    *capacity = new_capacity;
    return true;
}

void free_device_potential(DevicePotential *pot)
{
    if (pot == nullptr) return;
    gpuFree(pot->spmap);
    gpuFree(pot->basic_mu);
    gpuFree(pot->basic_b1);
    gpuFree(pot->basic_b2);
    gpuFree(pot->basic_b3);
    gpuFree(pot->basic_nu);
    gpuFree(pot->basic_mu_offsets);
    gpuFree(pot->basic_mu_indices);
    gpuFree(pot->times_l1);
    gpuFree(pot->times_l2);
    gpuFree(pot->times_c);
    gpuFree(pot->times_id);
    gpuFree(pot->level_offsets);
    gpuFree(pot->forward_level_offsets);
    gpuFree(pot->forward_ids);
    gpuFree(pot->forward_term_offsets);
    gpuFree(pot->forward_terms);
    gpuFree(pot->forward_block_level_offsets);
    gpuFree(pot->forward_block_groups);
    gpuFree(pot->reverse_level_offsets);
    gpuFree(pot->reverse_ids);
    gpuFree(pot->reverse_entry_offsets);
    gpuFree(pot->reverse_entries);
    gpuFree(pot->reverse_block_level_offsets);
    gpuFree(pot->reverse_block_groups);
    gpuFree(pot->forward_short_level_offsets);
    gpuFree(pot->forward_short_groups);
    gpuFree(pot->forward_warp_level_offsets);
    gpuFree(pot->forward_warp_groups);
    gpuFree(pot->reverse_short_level_offsets);
    gpuFree(pot->reverse_short_groups);
    gpuFree(pot->reverse_warp_level_offsets);
    gpuFree(pot->reverse_warp_groups);
    gpuFree(pot->alpha_moment_mapping);
    gpuFree(pot->pair_radial_bases);
    gpuFree(pot->radial_basis_parameters);
    gpuFree(pot->radial_coeffs);
    gpuFree(pot->species_coeffs);
    gpuFree(pot->moment_coeffs);
    gpuFree(pot->descriptor_species_coeffs);
    gpuFree(pot->descriptor_moment_coeffs);
    gpuFree(pot->model_coeffs);
    *pot = {};
}

bool validate_potential_view(const MTPGpuPotentialView &view, std::string *error)
{
    if (view.species_count <= 0 || view.max_type <= 0 || view.spmap == nullptr) {
        set_error(error, "invalid species_count/max_type/spmap in MTPGpuPotentialView");
        return false;
    }
    if (view.alpha_moments_count <= 0 || view.alpha_index_basic_count <= 0 ||
        view.alpha_index_basic_count > view.alpha_moments_count ||
        view.alpha_scalar_moments <= 0) {
        set_error(error, "invalid MTP moment counts in MTPGpuPotentialView");
        return false;
    }
    if (view.radial_func_count <= 0 || view.rb_size <= 0) {
        set_error(error, "invalid radial_func_count/rb_size in MTPGpuPotentialView");
        return false;
    }
    if (view.max_dist <= 0.0) {
        set_error(error, "invalid max_dist in MTPGpuPotentialView");
        return false;
    }
    if (view.radial_coeffs == nullptr || view.species_coeffs == nullptr) {
        set_error(error, "missing coefficient pointer in MTPGpuPotentialView");
        return false;
    }
    const int model_type = static_cast<int>(view.model);
    if (model_type < static_cast<int>(MTPGpuModel::Linear) ||
        model_type > static_cast<int>(MTPGpuModel::MTPMLinear)) {
        set_error(error, "unsupported site-energy model in MTPGpuPotentialView");
        return false;
    }
    if (view.model == MTPGpuModel::Linear) {
        if (view.moment_coeffs == nullptr || view.descriptor_count != 0 ||
            view.descriptor_species_coeffs != nullptr ||
            view.descriptor_moment_coeffs != nullptr ||
            view.model_coeffs != nullptr || view.model_coeffs_count != 0) {
            set_error(error, "invalid classic linear coefficient layout");
            return false;
        }
    } else {
        if (view.descriptor_count <= 0 ||
            view.descriptor_species_coeffs == nullptr ||
            view.descriptor_moment_coeffs == nullptr ||
            (view.model_coeffs == nullptr && view.model_coeffs_count != 0) ||
            view.moment_coeffs != nullptr) {
            set_error(error,
                "invalid MTP-M descriptor/model coefficient layout");
            return false;
        }
        long long expected_model_coeffs = 0;
        if (view.model == MTPGpuModel::MTPMLinear) {
            if (view.descriptor_count != 1 || view.model_hidden1 != 0 || view.model_hidden2 != 0) {
                set_error(error, "invalid linear MTP-M dimensions"); return false;
            }
        } else if (view.model == MTPGpuModel::MTPMQuadratic ||
            view.model == MTPGpuModel::MTPMAsinh) {
            expected_model_coeffs =
                view.model == MTPGpuModel::MTPMQuadratic ? 5 : 1;
            if (view.descriptor_count != 2 || view.model_hidden1 != 0 ||
                view.model_hidden2 != 0) {
                set_error(error, "invalid analytic MTP-M model dimensions");
                return false;
            }
        } else {
            const int activation = static_cast<int>(view.model_activation);
            if (activation < static_cast<int>(MTPGpuActivation::Tanh) ||
                activation > static_cast<int>(MTPGpuActivation::Linear) ||
                view.model_hidden1 <= 0) {
                set_error(error, "invalid neural MTP-M activation or hidden size");
                return false;
            }
            if (view.model == MTPGpuModel::MTPMNN1) {
                if (view.model_hidden2 != 0) {
                    set_error(error, "NN1 MTP-M must not define a second hidden layer");
                    return false;
                }
                expected_model_coeffs =
                    static_cast<long long>(view.descriptor_count) *
                        view.model_hidden1 + 2LL * view.model_hidden1;
            } else {
                if (view.model_hidden2 <= 0 ||
                    view.model_activation == MTPGpuActivation::Linear) {
                    set_error(error,
                        "NN2 MTP-M requires two hidden layers and a supported nonlinear activation");
                    return false;
                }
                expected_model_coeffs =
                    static_cast<long long>(view.descriptor_count) *
                        view.model_hidden1 + view.model_hidden1 +
                    static_cast<long long>(view.model_hidden1) *
                        view.model_hidden2 + 2LL * view.model_hidden2;
            }
        }
        if (expected_model_coeffs > std::numeric_limits<int>::max() ||
            view.model_coeffs_count != expected_model_coeffs) {
            set_error(error,
                "MTP-M model coefficient count does not match its dimensions");
            return false;
        }
    }
    const long long expected_radial_count =
        static_cast<long long>(view.species_count) * view.species_count *
        view.radial_func_count * view.rb_size;
    if (expected_radial_count > std::numeric_limits<int>::max() ||
        (view.radial_coeffs_count > 0 &&
         view.radial_coeffs_count != expected_radial_count)) {
        set_error(error, "radial coefficient count does not match the MTP dimensions");
        return false;
    }
    if (view.alpha_index_basic == nullptr || view.alpha_moment_mapping == nullptr) {
        set_error(error, "missing alpha-index pointer in MTPGpuPotentialView");
        return false;
    }
    if (view.alpha_index_times_count > 0 && view.alpha_index_times == nullptr) {
        set_error(error, "alpha_index_times_count is nonzero but alpha_index_times is null");
        return false;
    }
    for (int i = 0; i < view.max_type; ++i) {
        if (view.spmap[i] < -1 || view.spmap[i] >= view.species_count) {
            set_error(error, "species mapping is outside [-1, species_count)");
            return false;
        }
    }
    for (int i = 0; i < view.alpha_scalar_moments; ++i) {
        if (view.alpha_moment_mapping[i] < 0 ||
            view.alpha_moment_mapping[i] >= view.alpha_moments_count) {
            set_error(error, "alpha_moment_mapping entry is out of range");
            return false;
        }
    }
    const int rb = static_cast<int>(view.radial_basis);
    if (rb < static_cast<int>(RadialBasis::Chebyshev) ||
        rb > static_cast<int>(RadialBasis::PolyBCos)) {
        set_error(error, "unsupported radial basis in MTPGpuPotentialView");
        return false;
    }
    if ((view.radial_basis == RadialBasis::ChebyshevConfigurable ||
         view.radial_basis == RadialBasis::ChebyshevPoly4 ||
         view.radial_basis == RadialBasis::PolyB || view.radial_basis == RadialBasis::PolyBCos) &&
        view.pair_radial_basis_count == 0) {
        set_error(error,
            "configurable radial basis requires explicit pair descriptors");
        return false;
    }
    if (view.pair_radial_basis_count != 0) {
        const long long expected_pair_count =
            static_cast<long long>(view.species_count) * view.species_count;
        if (view.pair_radial_bases == nullptr ||
            view.pair_radial_basis_count != expected_pair_count) {
            set_error(error,
                "split radial basis count does not match species_count^2");
            return false;
        }
        for (int pair = 0; pair < view.pair_radial_basis_count; ++pair) {
            const auto &basis = view.pair_radial_bases[pair];
            const int pair_type = static_cast<int>(basis.radial_basis);
            const int cutoff_type = static_cast<int>(basis.cutoff);
            const int transform_type = static_cast<int>(basis.transform);
            if (pair_type < static_cast<int>(RadialBasis::Chebyshev) ||
                pair_type > static_cast<int>(RadialBasis::PolyBCos) ||
                cutoff_type < static_cast<int>(RadialCutoff::Quadratic) ||
                cutoff_type > static_cast<int>(RadialCutoff::Smoothstep7) ||
                transform_type < static_cast<int>(RadialTransform::Linear) ||
                transform_type > static_cast<int>(RadialTransform::ExponentialTanh) ||
                !std::isfinite(basis.min_dist) ||
                !std::isfinite(basis.max_dist) ||
                !std::isfinite(basis.scaling) ||
                !std::isfinite(basis.cutoff_parameter) ||
                !std::isfinite(basis.transform_parameter) ||
                !std::isfinite(basis.transform_scale_parameter) ||
                !std::isfinite(basis.cutoff_r1) || !std::isfinite(basis.cutoff_r2) ||
                basis.max_dist <= basis.min_dist) {
                set_error(error, "invalid split radial basis descriptor");
                return false;
            }
            if ((basis.radial_basis == RadialBasis::PolyB || basis.radial_basis == RadialBasis::PolyBCos) &&
                (!basis.polynomial_matrix || !basis.polynomial_normalizations)) {
                set_error(error, "PolyB radial basis requires its native orthogonalization matrix"); return false;
            }
            if (basis.radial_basis == RadialBasis::ChebyshevConfigurable &&
                basis.cutoff == RadialCutoff::Smoothstep7 &&
                !(basis.min_dist < basis.cutoff_r1 && basis.cutoff_r1 <= basis.cutoff_r2 && basis.cutoff_r2 < basis.max_dist)) {
                set_error(error, "invalid smoothstep7 radial interval"); return false;
            }
            if (basis.radial_basis == RadialBasis::ChebyshevConfigurable &&
                (basis.transform == RadialTransform::Exponential ||
                 basis.transform == RadialTransform::ExponentialLeft ||
                 basis.transform == RadialTransform::ExponentialTanh) &&
                basis.transform_parameter != 0.0) {
                const double range = basis.max_dist - basis.min_dist;
                const double lambda = basis.transform_parameter;
                const double denominator = std::expm1(lambda);
                const double rate = lambda / range;
                const double first = 2.0 * rate / denominator;
                const double second = -rate * first;
                const double third = -rate * second;
                if (!std::isfinite(denominator) || denominator == 0.0 ||
                    !std::isfinite(first) || !std::isfinite(second) ||
                    !std::isfinite(third)) {
                    set_error(error,
                        "invalid configurable exponential transform descriptor");
                    return false;
                }
            }
        }
    } else if (view.pair_radial_bases != nullptr) {
        set_error(error,
            "pair_radial_bases is non-null but pair_radial_basis_count is zero");
        return false;
    }
    return true;
}

bool build_host_potential(const MTPGpuPotentialView &view, HostPotential *host, std::string *error)
{
    if (!validate_potential_view(view, error)) {
        return false;
    }

    host->dims = {};
    host->dims.species_count = view.species_count;
    host->dims.max_type = view.max_type;
    host->dims.alpha_index_basic_count = view.alpha_index_basic_count;
    host->dims.alpha_index_times_count = view.alpha_index_times_count;
    host->dims.alpha_scalar_moments = view.alpha_scalar_moments;
    host->dims.radial_func_count = view.radial_func_count;
    host->dims.rb_size = view.rb_size;
    host->dims.min_dist = view.min_dist;
    host->dims.max_dist = view.max_dist;
    host->dims.cutoff = view.cutoff > 0.0 ? view.cutoff : view.max_dist;
    host->dims.cutoffsq = host->dims.cutoff * host->dims.cutoff;
    host->dims.scaling = view.scaling;
    host->dims.plus_r = view.min_dist + view.max_dist;
    host->dims.sub_r = view.max_dist - view.min_dist;
    if (host->dims.sub_r <= 0.0) {
        set_error(error, "max_dist must be larger than min_dist");
        return false;
    }
    host->dims.mult = 2.0 / host->dims.sub_r;
    host->dims.irpi = kPi / view.max_dist;
    host->dims.p0 = 1.0 - view.min_dist / view.max_dist;
    host->dims.pp = 1.0 / host->dims.p0 / view.max_dist;
    host->dims.rb_type = static_cast<int>(view.radial_basis);
    host->dims.model_type = static_cast<int>(view.model);
    host->dims.model_activation = static_cast<int>(view.model_activation);
    host->dims.model_hidden1 = view.model_hidden1;
    host->dims.model_hidden2 = view.model_hidden2;
    host->dims.descriptor_count = view.descriptor_count;
    host->dims.model_coeffs_count = view.model_coeffs_count;
    host->dims.nonradial_parameter_count = view.model == MTPGpuModel::Linear
        ? view.species_count + view.alpha_scalar_moments
        : view.species_count + view.descriptor_count * view.species_count +
              view.descriptor_count * view.alpha_scalar_moments +
              view.model_coeffs_count;
    host->dims.pair_radial_basis_count = view.pair_radial_basis_count;
    host->pair_radial_bases.resize(view.pair_radial_basis_count);
    for (int pair = 0; pair < view.pair_radial_basis_count; ++pair) {
        const auto &source = view.pair_radial_bases[pair];
        auto &target = host->pair_radial_bases[pair];
        target.rb_type = static_cast<int>(source.radial_basis);
        target.cutoff_type = static_cast<int>(source.cutoff);
        target.transform_type = static_cast<int>(source.transform);
        target.min_dist = source.min_dist;
        target.min_distsq = source.min_dist * source.min_dist;
        target.max_dist = source.max_dist;
        target.cutoffsq = source.max_dist * source.max_dist;
        target.scaling = source.scaling;
        target.plus_r = source.min_dist + source.max_dist;
        target.sub_r = source.max_dist - source.min_dist;
        target.mult = 2.0 / target.sub_r;
        target.irpi = kPi / source.max_dist;
        target.cutoff_parameter = source.cutoff_parameter;
        target.transform_parameter = source.transform_parameter;
        target.one_minus_chebyshev = source.one_minus_chebyshev;
        target.cutoff_r1 = source.cutoff_r1; target.cutoff_r2 = source.cutoff_r2;
        target.transform_scale_parameter = source.transform_scale_parameter;
        if (source.radial_basis == RadialBasis::PolyB || source.radial_basis == RadialBasis::PolyBCos) {
            target.polynomial_offset = host->radial_basis_parameters.size();
            const size_t matrix_count = size_t(view.rb_size) * view.rb_size;
            for (size_t i=0;i<matrix_count;++i) {
                if (!std::isfinite(source.polynomial_matrix[i])) {
                    set_error(error, "non-finite PolyB orthogonalization matrix"); return false;
                }
                host->radial_basis_parameters.push_back(source.polynomial_matrix[i]);
            }
            for (int i=0;i<view.rb_size;++i) {
                if (!std::isfinite(source.polynomial_normalizations[i])) {
                    set_error(error, "non-finite PolyB normalization"); return false;
                }
                host->radial_basis_parameters.push_back(source.polynomial_normalizations[i]);
            }
        }
        target.transform_denominator = source.transform_parameter == 0.0
            ? 0.0 : std::expm1(source.transform_parameter);
    }

    host->spmap.assign(view.spmap, view.spmap + view.max_type);
    host->basic_mu.resize(view.alpha_index_basic_count);
    host->basic_b1.resize(view.alpha_index_basic_count);
    host->basic_b2.resize(view.alpha_index_basic_count);
    host->basic_b3.resize(view.alpha_index_basic_count);
    host->basic_nu.resize(view.alpha_index_basic_count);

    int max_mu = view.max_mu;
    int max_nu = 0;
    for (int i = 0; i < view.alpha_index_basic_count; ++i) {
        const int mu = view.alpha_index_basic[4 * i + 0];
        const int b1 = view.alpha_index_basic[4 * i + 1];
        const int b2 = view.alpha_index_basic[4 * i + 2];
        const int b3 = view.alpha_index_basic[4 * i + 3];
        const int nu = view.nu_arr_list != nullptr ? view.nu_arr_list[i] : b1 + b2 + b3;
        if (mu < 0 || b1 < 0 || b2 < 0 || b3 < 0 || nu < 0) {
            set_error(error, "negative alpha_index_basic entry");
            return false;
        }
        host->basic_mu[i] = mu;
        host->basic_b1[i] = b1;
        host->basic_b2[i] = b2;
        host->basic_b3[i] = b3;
        host->basic_nu[i] = nu;
        max_mu = std::max(max_mu, mu + 1);
        max_nu = std::max(max_nu, std::max({nu, b1, b2, b3}));
    }
    host->dims.max_mu = max_mu;
    host->dims.max_alpha_index_basic = std::max(view.max_alpha_index_basic, max_nu + 1);
    host->dims.coord_power_count = host->dims.max_alpha_index_basic + 1;
    if (host->dims.max_mu > view.radial_func_count) {
        set_error(error, "max_mu exceeds radial_func_count");
        return false;
    }

    host->basic_mu_offsets.assign(host->dims.max_mu + 1, 0);
    for (int mu : host->basic_mu) {
        ++host->basic_mu_offsets[mu + 1];
    }
    std::partial_sum(host->basic_mu_offsets.begin(),
                     host->basic_mu_offsets.end(),
                     host->basic_mu_offsets.begin());
    host->basic_mu_indices.resize(view.alpha_index_basic_count);
    std::vector<int> mu_cursor = host->basic_mu_offsets;
    for (int bi = 0; bi < view.alpha_index_basic_count; ++bi) {
        host->basic_mu_indices[mu_cursor[host->basic_mu[bi]]++] = bi;
    }

    std::vector<int> moment_level(view.alpha_moments_count, -1);
    for (int i = 0; i < view.alpha_index_basic_count; ++i) {
        moment_level[i] = 0;
    }
    for (int i = 0; i < view.alpha_index_times_count; ++i) {
        const int l1 = view.alpha_index_times[4 * i + 0];
        const int l2 = view.alpha_index_times[4 * i + 1];
        const int id = view.alpha_index_times[4 * i + 3];
        if (l1 < 0 || l1 >= view.alpha_moments_count ||
            l2 < 0 || l2 >= view.alpha_moments_count ||
            id < 0 || id >= view.alpha_moments_count) {
            set_error(error, "alpha_index_times entry is out of range");
            return false;
        }
        if (moment_level[l1] < 0 || moment_level[l2] < 0) {
            set_error(error, "alpha_index_times is not topologically ordered");
            return false;
        }
        const int lev = std::max(moment_level[l1], moment_level[l2]) + 1;
        moment_level[id] = std::max(moment_level[id], lev);
    }
    for (int i = 0; i < view.alpha_scalar_moments; ++i) {
        if (moment_level[view.alpha_moment_mapping[i]] < 0) {
            set_error(error, "alpha_moment_mapping refers to an undefined moment");
            return false;
        }
    }

    // Potentials produced by basis searches may retain a large declared
    // moment range even though only a sparse subset is addressed.  Keep all
    // basic moments first (and therefore identity-mapped for the tiled basic
    // kernels), then compact every defined composite moment in source-index
    // order.  This monotonic mapping also preserves grouped sort order and
    // leaves coefficient/scalar ordering unchanged.
    std::vector<int> moment_remap(view.alpha_moments_count, -1);
    int active_moment_count = 0;
    for (int index = 0; index < view.alpha_moments_count; ++index) {
        if (moment_level[index] >= 0) {
            moment_remap[index] = active_moment_count++;
        }
    }
    if (active_moment_count < view.alpha_index_basic_count) {
        set_error(error, "active moment count is smaller than the basic prefix");
        return false;
    }
    host->dims.alpha_moments_count = active_moment_count;
    host->alpha_moment_mapping.resize(view.alpha_scalar_moments);
    for (int i = 0; i < view.alpha_scalar_moments; ++i) {
        host->alpha_moment_mapping[i] =
            moment_remap[view.alpha_moment_mapping[i]];
    }
    auto sorted_mapping = host->alpha_moment_mapping;
    std::sort(sorted_mapping.begin(), sorted_mapping.end());
    host->dims.unique_scalar_mapping =
        std::adjacent_find(sorted_mapping.begin(), sorted_mapping.end()) ==
        sorted_mapping.end();
    int max_level = 0;
    for (int lev : moment_level) {
        max_level = std::max(max_level, lev);
    }
    host->schedule.max_level = max_level;
    host->dims.max_level = max_level;
    host->schedule.level_offsets.assign(max_level + 2, 0);
    for (int i = 0; i < view.alpha_index_times_count; ++i) {
        const int id = view.alpha_index_times[4 * i + 3];
        ++host->schedule.level_offsets[moment_level[id] + 1];
    }
    std::partial_sum(host->schedule.level_offsets.begin(), host->schedule.level_offsets.end(),
                     host->schedule.level_offsets.begin());
    std::vector<int> cursor = host->schedule.level_offsets;
    host->schedule.times_l1.resize(view.alpha_index_times_count);
    host->schedule.times_l2.resize(view.alpha_index_times_count);
    host->schedule.times_c.resize(view.alpha_index_times_count);
    host->schedule.times_id.resize(view.alpha_index_times_count);
    for (int i = 0; i < view.alpha_index_times_count; ++i) {
        const int id = view.alpha_index_times[4 * i + 3];
        const int lev = moment_level[id];
        const int pos = cursor[lev]++;
        host->schedule.times_l1[pos] =
            moment_remap[view.alpha_index_times[4 * i + 0]];
        host->schedule.times_l2[pos] =
            moment_remap[view.alpha_index_times[4 * i + 1]];
        host->schedule.times_c[pos] = view.alpha_index_times[4 * i + 2];
        host->schedule.times_id[pos] = moment_remap[id];
    }

    // The production kernels assign one thread to each forward output and
    // reverse destination.  Terms remain in their original schedule arrays so
    // the one-thread ordered kernels retain their exact accumulation order.
    auto &schedule = host->schedule;
    schedule.forward_level_offsets.assign(max_level + 2, 0);
    schedule.forward_term_offsets.push_back(0);
    for (int level = 0; level <= max_level; ++level) {
        schedule.forward_level_offsets[level] =
            static_cast<int>(schedule.forward_ids.size());
        std::vector<int> terms;
        const int begin = schedule.level_offsets[level];
        const int end = schedule.level_offsets[level + 1];
        terms.reserve(end - begin);
        for (int t = begin; t < end; ++t) {
            terms.push_back(t);
        }
        std::stable_sort(terms.begin(), terms.end(), [&](int lhs, int rhs) {
            return schedule.times_id[lhs] < schedule.times_id[rhs];
        });
        for (int pos = 0; pos < static_cast<int>(terms.size());) {
            const int id = schedule.times_id[terms[pos]];
            schedule.forward_ids.push_back(id);
            do {
                schedule.forward_terms.push_back(terms[pos++]);
            } while (pos < static_cast<int>(terms.size()) &&
                     schedule.times_id[terms[pos]] == id);
            schedule.forward_term_offsets.push_back(
                static_cast<int>(schedule.forward_terms.size()));
        }
    }
    schedule.forward_level_offsets[max_level + 1] =
        static_cast<int>(schedule.forward_ids.size());
    schedule.forward_block_level_offsets.assign(max_level + 2, 0);
    for (int level = 0; level <= max_level; ++level) {
        schedule.forward_block_level_offsets[level] =
            static_cast<int>(schedule.forward_block_groups.size());
        const int begin = schedule.forward_level_offsets[level];
        const int end = schedule.forward_level_offsets[level + 1];
        std::vector<int> block_groups;
        for (int group = begin; group < end; ++group) {
            const int term_count = schedule.forward_term_offsets[group + 1] -
                                   schedule.forward_term_offsets[group];
            if (term_count > propagation_block_group_threshold) {
                block_groups.push_back(group);
            }
        }
        std::stable_sort(block_groups.begin(), block_groups.end(),
                         [&](int lhs, int rhs) {
            const int lhs_count = schedule.forward_term_offsets[lhs + 1] -
                                  schedule.forward_term_offsets[lhs];
            const int rhs_count = schedule.forward_term_offsets[rhs + 1] -
                                  schedule.forward_term_offsets[rhs];
            return lhs_count > rhs_count;
        });
        schedule.forward_block_groups.insert(
            schedule.forward_block_groups.end(),
            block_groups.begin(), block_groups.end());
    }
    schedule.forward_block_level_offsets[max_level + 1] =
        static_cast<int>(schedule.forward_block_groups.size());

    // A reverse entry encodes (term_index, side) as 2 * term_index + side.
    // side 0 targets l1 and reads l2; side 1 targets l2 and reads l1.  Keeping
    // both entries when l1 == l2 preserves the factor of two in the chain rule.
    schedule.reverse_level_offsets.assign(max_level + 2, 0);
    schedule.reverse_entry_offsets.push_back(0);
    for (int level = 0; level <= max_level; ++level) {
        schedule.reverse_level_offsets[level] =
            static_cast<int>(schedule.reverse_ids.size());
        std::vector<int> entries;
        const int begin = schedule.level_offsets[level];
        const int end = schedule.level_offsets[level + 1];
        entries.reserve(2 * (end - begin));
        for (int t = begin; t < end; ++t) {
            entries.push_back(2 * t);
            entries.push_back(2 * t + 1);
        }
        auto destination = [&](int entry) {
            const int t = entry >> 1;
            return (entry & 1) != 0
                ? schedule.times_l2[t]
                : schedule.times_l1[t];
        };
        std::stable_sort(entries.begin(), entries.end(),
                         [&](int lhs, int rhs) {
            return destination(lhs) < destination(rhs);
        });
        for (int pos = 0; pos < static_cast<int>(entries.size());) {
            const int id = destination(entries[pos]);
            schedule.reverse_ids.push_back(id);
            do {
                schedule.reverse_entries.push_back(entries[pos++]);
            } while (pos < static_cast<int>(entries.size()) &&
                     destination(entries[pos]) == id);
            schedule.reverse_entry_offsets.push_back(
                static_cast<int>(schedule.reverse_entries.size()));
        }
    }
    schedule.reverse_level_offsets[max_level + 1] =
        static_cast<int>(schedule.reverse_ids.size());
    schedule.reverse_block_level_offsets.assign(max_level + 2, 0);
    for (int level = 0; level <= max_level; ++level) {
        schedule.reverse_block_level_offsets[level] =
            static_cast<int>(schedule.reverse_block_groups.size());
        const int begin = schedule.reverse_level_offsets[level];
        const int end = schedule.reverse_level_offsets[level + 1];
        std::vector<int> block_groups;
        for (int group = begin; group < end; ++group) {
            const int entry_count =
                schedule.reverse_entry_offsets[group + 1] -
                schedule.reverse_entry_offsets[group];
            if (entry_count > propagation_block_group_threshold) {
                block_groups.push_back(group);
            }
        }
        std::stable_sort(block_groups.begin(), block_groups.end(),
                         [&](int lhs, int rhs) {
            const int lhs_count = schedule.reverse_entry_offsets[lhs + 1] -
                                  schedule.reverse_entry_offsets[lhs];
            const int rhs_count = schedule.reverse_entry_offsets[rhs + 1] -
                                  schedule.reverse_entry_offsets[rhs];
            return lhs_count > rhs_count;
        });
        schedule.reverse_block_groups.insert(
            schedule.reverse_block_groups.end(),
            block_groups.begin(), block_groups.end());
    }
    schedule.reverse_block_level_offsets[max_level + 1] =
        static_cast<int>(schedule.reverse_block_groups.size());

#if MTP_GPU_PROPAGATION_VARIANT == 1 || MTP_GPU_PROPAGATION_VARIANT == 2
    // Keep each group's arithmetic order. Only partition independent groups
    // within a level; dependencies still cross the original level barrier.
    auto partition_groups = [&](const std::vector<int> &levels,
                                const std::vector<int> &terms,
                                std::vector<int> &short_levels,
                                std::vector<int> &short_groups,
                                std::vector<int> &warp_levels,
                                std::vector<int> &warp_groups) {
        short_levels.assign(max_level + 2, 0);
        warp_levels.assign(max_level + 2, 0);
        for (int level = 0; level <= max_level; ++level) {
            short_levels[level] = static_cast<int>(short_groups.size());
            warp_levels[level] = static_cast<int>(warp_groups.size());
            for (int group = levels[level]; group < levels[level + 1]; ++group) {
                const int count = terms[group + 1] - terms[group];
                if (count <= propagation_warp_group_threshold)
                    short_groups.push_back(group);
                else if (count <= propagation_block_group_threshold)
                    warp_groups.push_back(group);
            }
        }
        short_levels[max_level + 1] = static_cast<int>(short_groups.size());
        warp_levels[max_level + 1] = static_cast<int>(warp_groups.size());
    };
    partition_groups(schedule.forward_level_offsets, schedule.forward_term_offsets,
                     schedule.forward_short_level_offsets, schedule.forward_short_groups,
                     schedule.forward_warp_level_offsets, schedule.forward_warp_groups);
    partition_groups(schedule.reverse_level_offsets, schedule.reverse_entry_offsets,
                     schedule.reverse_short_level_offsets, schedule.reverse_short_groups,
                     schedule.reverse_warp_level_offsets, schedule.reverse_warp_groups);
#endif

    return true;
}

__device__ double atomic_add_double(double *address, double value)
{
#if defined(__HIP_DEVICE_COMPILE__) || (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 600)
    return atomicAdd(address, value);
#else
    unsigned long long int *address_as_ull =
        reinterpret_cast<unsigned long long int *>(address);
    unsigned long long int old = *address_as_ull;
    unsigned long long int assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        static_cast<unsigned long long int>(
                            __double_as_longlong(value + __longlong_as_double(assumed))));
    } while (assumed != old);
    return __longlong_as_double(old);
#endif
}

// A thread owns short groups; a full warp cooperates only after the serial
// term count exceeds a warp.  This keeps regular bases on their inexpensive
// grouped path while bounding the long tail in irregular schedules.
__device__ double warp_sum(double value)
{
    for (int offset = propagation_warp_size / 2; offset > 0; offset >>= 1) {
        value += shuffle_down_32(value, offset);
    }
    return value;
}

__device__ void rb_calc(const DevicePotential &pot, int pair, double r,
                        double *rb_vals, double *rb_ders, bool need_derivatives)
{
    const bool split = pot.pair_radial_basis_count != 0;
    const DeviceRadialBasis *basis = split ? pot.pair_radial_bases + pair : nullptr;
    const auto type = static_cast<RadialBasis>(split ? basis->rb_type : pot.rb_type);
    const double lo = split ? basis->min_dist : pot.min_dist;
    const double hi = split ? basis->max_dist : pot.max_dist;
    const double scaling = split ? basis->scaling : pot.scaling;
    if (type == RadialBasis::PolyB || type == RadialBasis::PolyBCos) {
        const double *matrix = pot.radial_basis_parameters + basis->polynomial_offset;
        const double *normalization = matrix + size_t(pot.rb_size) * pot.rb_size;
        const double delta = hi - r;
        double fc = 1.0, dfc = 0.0;
        const int first_power = type == RadialBasis::PolyB ? 3 : 2;
        if (type == RadialBasis::PolyBCos)
            RadialMath::envelope(RadialCutoff::CosineLeft, r, lo, hi, 0, 0, 0, fc, dfc);
        for (int i=0;i<pot.rb_size;++i) {
            double value = 0.0, derivative = 0.0;
            double previous = scaling * delta;
            if (first_power == 3) previous *= delta;
            for (int j=0;j<pot.rb_size;++j) {
                const double power = previous * delta;
                const double coefficient = matrix[size_t(j) * pot.rb_size + i];
                value += coefficient * (normalization[j] * power);
                if (need_derivatives) derivative += coefficient * (-(normalization[j] * (j + first_power)) * previous);
                previous = power;
            }
            rb_vals[i] = value * fc;
            if (need_derivatives) rb_ders[i] = derivative * fc + value * dfc;
        }
        return;
    }
    double coordinate = (2.0 * r - lo - hi) / (hi - lo);
    double dx = 2.0 / (hi - lo), fc = 0.0, dfc = 0.0;
    bool flipped = false;
    if (type == RadialBasis::ChebyshevConfigurable) {
        RadialMath::coordinate(static_cast<RadialTransform>(basis->transform_type),
            r, lo, hi, basis->transform_parameter, basis->transform_scale_parameter, coordinate, dx);
        RadialMath::envelope(static_cast<RadialCutoff>(basis->cutoff_type),
            r, lo, hi, basis->cutoff_parameter, basis->cutoff_r1, basis->cutoff_r2, fc, dfc);
        flipped = basis->one_minus_chebyshev;
    } else if (type == RadialBasis::ChebyCos) {
        RadialMath::envelope(RadialCutoff::CosineLeft, r, lo, hi, 0, 0, 0, fc, dfc);
    } else if (type == RadialBasis::ChebyshevPoly4 || (type == RadialBasis::ChebyshevPoly5 && split)) {
        RadialMath::envelope(type == RadialBasis::ChebyshevPoly4 ? RadialCutoff::Poly4 : RadialCutoff::Poly5,
            r, lo, hi, basis->cutoff_parameter, 0, 0, fc, dfc);
    } else if (type == RadialBasis::ChebyshevPoly5) {
        // Preserve the historical bare-view Poly5 convention. Native models
        // always supply their explicit parameterized pair descriptors.
        const double q = (1.0-r/pot.cutoff)/pot.p0;
        fc = q*q*q*(10.0+q*(-15.0+6.0*q));
        dfc = -30.0*q*q*(1.0-q)*(1.0-q)*pot.pp;
    } else {
        RadialMath::envelope(RadialCutoff::Quadratic, r, lo, hi, 0, 0, 0, fc, dfc);
    }
    double previous = 1.0, current = coordinate;
    double dprevious = 0.0, dcurrent = dx;
    for (int i=0;i<pot.rb_size;++i) {
        const double polynomial = flipped ? 1.0-current : previous;
        const double derivative = flipped ? -dcurrent : dprevious;
        rb_vals[i] = scaling * polynomial * fc;
        if (need_derivatives) rb_ders[i] = scaling * (derivative * fc + polynomial * dfc);
        const double next = 2.0 * coordinate * current - previous;
        const double dnext = 2.0 * (dx * current + coordinate * dcurrent) - dprevious;
        previous = current; current = next; dprevious = dcurrent; dcurrent = dnext;
    }
}

__device__ void fill_pair_cache(const DevicePotential &pot, int lane,
                                int center_idx, int ati, int jj,
                                const int *offsets, const int *neighbors,
                                const double *neighbor_vectors,
                                const int *type, const double *x,
                                double *rb_vals, double *rb_ders,
                                double *val_mu, double *der_mu,
                                double *dist_powers, double *coord_powers,
                                double *rvec_cache, double *rijinv_cache,
                                double *valid_cache, bool need_derivatives,
                                bool need_radial_values = true)
{
    const int atj = neighbors[offsets[center_idx] + jj];
    const int itype = pot.spmap[type[ati]];
    const int jtype = pot.spmap[type[atj]];
    if (itype < 0 || jtype < 0) {
        valid_cache[lane] = 0.0;
        return;
    }

    const int pair = itype * pot.species_count + jtype;
    const int edge = offsets[center_idx] + jj;
    const double rx = neighbor_vectors != nullptr
        ? neighbor_vectors[3 * edge + 0] : x[3 * atj + 0] - x[3 * ati + 0];
    const double ry = neighbor_vectors != nullptr
        ? neighbor_vectors[3 * edge + 1] : x[3 * atj + 1] - x[3 * ati + 1];
    const double rz = neighbor_vectors != nullptr
        ? neighbor_vectors[3 * edge + 2] : x[3 * atj + 2] - x[3 * ati + 2];
    const double rijsq = rx * rx + ry * ry + rz * rz;
    const DeviceRadialBasis *pair_basis =
        pot.pair_radial_basis_count != 0
            ? pot.pair_radial_bases + pair : nullptr;
    const double pair_cutoffsq =
        pair_basis != nullptr ? pair_basis->cutoffsq : pot.cutoffsq;
    const bool below_pair_min = pair_basis != nullptr &&
        pair_basis->min_dist > 0.0 &&
        (pair_basis->rb_type == static_cast<int>(RadialBasis::ChebyshevPoly4) ||
         pair_basis->rb_type ==
             static_cast<int>(RadialBasis::ChebyshevPoly5) ||
         pair_basis->rb_type ==
             static_cast<int>(RadialBasis::ChebyshevConfigurable)) &&
        rijsq < pair_basis->min_distsq;
    if (rijsq <= 0.0 || rijsq >= pair_cutoffsq || below_pair_min) {
        valid_cache[lane] = 0.0;
        return;
    }

    const double rij = sqrt(rijsq);
    const double rijinv = 1.0 / rij;
    valid_cache[lane] = 1.0;
    rijinv_cache[lane] = rijinv;
    rvec_cache[3 * lane + 0] = rx;
    rvec_cache[3 * lane + 1] = ry;
    rvec_cache[3 * lane + 2] = rz;

    double *rbv = rb_vals + lane * pot.rb_size;
    double *rbd = need_derivatives ? rb_ders + lane * pot.rb_size : nullptr;
    rb_calc(pot, pair, rij, rbv, rbd, need_derivatives);

    if (need_radial_values) {
        double *vm = val_mu + lane * pot.max_mu;
        double *dm = need_derivatives ? der_mu + lane * pot.max_mu : nullptr;
        const int pair_offset =
            (itype * pot.species_count + jtype) *
            pot.radial_func_count * pot.rb_size;
        for (int mu = 0; mu < pot.max_mu; ++mu) {
            double v = 0.0;
            double d = 0.0;
            const int coeff_offset = pair_offset + mu * pot.rb_size;
            for (int xi = 0; xi < pot.rb_size; ++xi) {
                const double coeff = pot.radial_coeffs[coeff_offset + xi];
                v += coeff * rbv[xi];
                if (need_derivatives) {
                    d += coeff * rbd[xi];
                }
            }
            vm[mu] = v;
            if (need_derivatives) {
                dm[mu] = d;
            }
        }
    }

    double *dp = dist_powers + lane * pot.max_alpha_index_basic;
    double *cp = coord_powers + lane * pot.coord_power_count * 3;
    dp[0] = 1.0;
    cp[0] = 0.0;
    cp[1] = 0.0;
    cp[2] = 0.0;
    cp[3] = 1.0;
    cp[4] = 1.0;
    cp[5] = 1.0;
    for (int k = 1; k < pot.max_alpha_index_basic; ++k) {
        dp[k] = dp[k - 1] * rijinv;
        cp[3 * (k + 1) + 0] = cp[3 * k + 0] * rx;
        cp[3 * (k + 1) + 1] = cp[3 * k + 1] * ry;
        cp[3 * (k + 1) + 2] = cp[3 * k + 2] * rz;
    }
}

// Neighbors occupy x lanes; distribute radial channels over the y lanes.
// Keep each channel's coefficient sum in its original order. The caller
// synchronizes the block before reading the basis and after writing channels.
template <bool Derivatives>
__device__ void contract_pair_radial(
    const DevicePotential &pot, int ati, int jj, int center_idx,
    const int *offsets, const int *neighbors, const int *type,
    const double *valid_cache, const double *rb_vals, const double *rb_ders,
    double *val_mu, double *der_mu)
{
    const int lane = threadIdx.x;
    if (valid_cache[lane] == 0.0) return;
    const int atj = neighbors[offsets[center_idx] + jj];
    const int pair = pot.spmap[type[ati]] * pot.species_count + pot.spmap[type[atj]];
    for (int mu = threadIdx.y; mu < pot.max_mu; mu += blockDim.y) {
        const int coeff_offset = (pair * pot.radial_func_count + mu) * pot.rb_size;
        double val = 0.0;
        double der = 0.0;
        for (int xi = 0; xi < pot.rb_size; ++xi) {
            const double coeff = pot.radial_coeffs[coeff_offset + xi];
            val += coeff * rb_vals[lane * pot.rb_size + xi];
            if constexpr (Derivatives) der += coeff * rb_ders[lane * pot.rb_size + xi];
        }
        val_mu[lane * pot.max_mu + mu] = val;
        if constexpr (Derivatives) der_mu[lane * pot.max_mu + mu] = der;
    }
}

template <typename WorkReal>
__global__ void basic_moments_kernel(DevicePotential pot, const int *centers,
                                     const int *offsets, const int *neighbors,
                                     const double *neighbor_vectors,
                                     const int *type, const double *x,
                                     WorkReal *moment_vals, int basic_tile)
{
    extern __shared__ unsigned char shared_bytes[];
    const int center_idx = blockIdx.x;
    const int tile_idx = blockIdx.y;
    const int tile_begin = tile_idx * basic_tile;
    const int remaining = pot.alpha_index_basic_count - tile_begin;
    const int tile_count = remaining < basic_tile ? remaining : basic_tile;
    const int nlane = blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;
    const int partial_stride = basic_tile + 1;

    WorkReal *partial = reinterpret_cast<WorkReal *>(shared_bytes);
    const size_t partial_bytes = static_cast<size_t>(nlane) * partial_stride *
        sizeof(WorkReal);
    const size_t double_offset =
        (partial_bytes + alignof(double) - 1) & ~(alignof(double) - 1);
    double *rb_vals = reinterpret_cast<double *>(shared_bytes + double_offset);
    double *val_mu = rb_vals + nlane * pot.rb_size;
    double *dist_powers = val_mu + nlane * pot.max_mu;
    double *coord_powers = dist_powers + nlane * pot.max_alpha_index_basic;
    double *rvec_cache = coord_powers + nlane * pot.coord_power_count * 3;
    double *rijinv_cache = rvec_cache + nlane * 3;
    double *valid_cache = rijinv_cache + nlane;

    for (int idx = tid; idx < nlane * partial_stride; idx += nthreads) {
        partial[idx] = static_cast<WorkReal>(0.0);
    }
    __syncthreads();

    const int ati = centers[center_idx];
    const int start = offsets[center_idx];
    const int end = offsets[center_idx + 1];
    const int jnum = end - start;

    // Every thread in the block must execute every barrier below.  In
    // particular, do not use `for (jj = tx; ...)`: the final partial neighbor
    // tile would make some x lanes leave the loop before the others.
    for (int neighbor_begin = 0; neighbor_begin < jnum; neighbor_begin += nlane) {
        const int jj = neighbor_begin + tx;
        if (ty == 0) {
            if (jj < jnum) {
                fill_pair_cache(pot, tx, center_idx, ati, jj,
                                offsets, neighbors,
                                neighbor_vectors, type, x,
                                rb_vals, nullptr, val_mu, nullptr, dist_powers,
                                coord_powers, rvec_cache, rijinv_cache, valid_cache, false, false);
            } else {
                valid_cache[tx] = 0.0;
            }
        }
        __syncthreads();
        contract_pair_radial<false>(
            pot, ati, jj, center_idx, offsets, neighbors, type,
            valid_cache, rb_vals, nullptr, val_mu, nullptr);
        __syncthreads();

        if (valid_cache[tx] != 0.0) {
            const double *vm = val_mu + tx * pot.max_mu;
            const double *dp = dist_powers + tx * pot.max_alpha_index_basic;
            const double *cp = coord_powers + tx * pot.coord_power_count * 3;

            for (int local_b = ty; local_b < tile_count; local_b += blockDim.y) {
                const int bi = tile_begin + local_b;
                const int mu = pot.basic_mu[bi];
                const int b1 = pot.basic_b1[bi];
                const int b2 = pot.basic_b2[bi];
                const int b3 = pot.basic_b3[bi];
                const int nu = pot.basic_nu[bi];
                double val = vm[mu];
                const double rpow = dp[nu];
                val *= rpow;
                const double pow0 = cp[3 * (b1 + 1) + 0];
                const double pow1 = cp[3 * (b2 + 1) + 1];
                const double pow2 = cp[3 * (b3 + 1) + 2];
                partial[tx * partial_stride + local_b] +=
                    static_cast<WorkReal>(val * pow0 * pow1 * pow2);
            }
        }
        __syncthreads();
    }

    for (int local_b = tid; local_b < tile_count; local_b += nthreads) {
        double sum = 0.0;
        for (int lane = 0; lane < nlane; ++lane) {
            sum += static_cast<double>(
                partial[lane * partial_stride + local_b]);
        }
        const int bi = tile_begin + local_b;
        moment_vals[center_idx * pot.alpha_moments_count + bi] =
            static_cast<WorkReal>(sum);
    }
}

__device__ void mtpm_descriptor_values(const DevicePotential &pot,
                                       const double *mom, int itype,
                                       double *q0, double *q1)
{
    *q0 = pot.descriptor_species_coeffs[itype];
    *q1 = pot.descriptor_count > 1 ? pot.descriptor_species_coeffs[pot.species_count + itype] : 0.0;
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        const double value = mom[pot.alpha_moment_mapping[scalar]];
        *q0 += pot.descriptor_moment_coeffs[scalar] * value;
        if (pot.descriptor_count > 1)
            *q1 += pot.descriptor_moment_coeffs[pot.alpha_scalar_moments + scalar] * value;
    }
}

__device__ void mtpm_model_derivatives(
    const DevicePotential &pot, double q0, double q1,
    double *energy, double *d0, double *d1,
    double *h00, double *h01, double *h11)
{
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMLinear)) {
        *energy = q0; *d0 = 1.0; *d1 = *h00 = *h01 = *h11 = 0.0;
    } else if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMQuadratic)) {
        const double c0 = pot.model_coeffs[0];
        const double c1 = pot.model_coeffs[1];
        const double c00 = pot.model_coeffs[2];
        const double c11 = pot.model_coeffs[3];
        const double c01 = pot.model_coeffs[4];
        *energy = c0 * q0 + c1 * q1 + c00 * q0 * q0 +
            c11 * q1 * q1 + c01 * q0 * q1;
        *d0 = c0 + 2.0 * c00 * q0 + c01 * q1;
        *d1 = c1 + 2.0 * c11 * q1 + c01 * q0;
        *h00 = 2.0 * c00;
        *h01 = c01;
        *h11 = 2.0 * c11;
    } else {
        const double coefficient = pot.model_coeffs[0];
        const double inverse_root = 1.0 / sqrt(1.0 + q1 * q1);
        *energy = q0 + coefficient * asinh(q1);
        *d0 = 1.0;
        *d1 = coefficient * inverse_root;
        *h00 = 0.0;
        *h01 = 0.0;
        *h11 = -coefficient * q1 * inverse_root * inverse_root * inverse_root;
    }
}

__device__ double mtpm_descriptor_value(const DevicePotential &pot,
                                        const double *mom, int itype,
                                        int descriptor)
{
    double value = pot.descriptor_species_coeffs[
        descriptor * pot.species_count + itype];
    const int offset = descriptor * pot.alpha_scalar_moments;
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        value += pot.descriptor_moment_coeffs[offset + scalar] *
            mom[pot.alpha_moment_mapping[scalar]];
    }
    return value;
}

__device__ double mtpm_descriptor_direction(const DevicePotential &pot,
                                            const double *dir,
                                            int descriptor)
{
    double value = 0.0;
    const int offset = descriptor * pot.alpha_scalar_moments;
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        value += pot.descriptor_moment_coeffs[offset + scalar] *
            dir[pot.alpha_moment_mapping[scalar]];
    }
    return value;
}

__device__ void mtpm_activation(int activation, double z,
                                double *value, double *first,
                                double *second)
{
    if (activation == static_cast<int>(MTPGpuActivation::Tanh)) {
        *value = tanh(z);
        *first = 1.0 - *value * *value;
        *second = -2.0 * *value * *first;
    } else if (activation == static_cast<int>(MTPGpuActivation::ReLU)) {
        *value = z > 0.0 ? z : 0.0;
        *first = z > 0.0 ? 1.0 : 0.0;
        *second = 0.0;
    } else if (activation == static_cast<int>(MTPGpuActivation::Sigmoid)) {
        if (z >= 0.0) {
            const double exponential = exp(-z);
            *value = 1.0 / (1.0 + exponential);
        } else {
            const double exponential = exp(z);
            *value = exponential / (1.0 + exponential);
        }
        *first = *value * (1.0 - *value);
        *second = *first * (1.0 - 2.0 * *value);
    } else {
        *value = z;
        *first = 1.0;
        *second = 0.0;
    }
}

__device__ void mtpm_nn1_hidden(const DevicePotential &pot,
                                const double *mom, const double *dir,
                                int itype, int hidden, double *value,
                                double *first, double *second,
                                double *direction)
{
    const int input_count = pot.descriptor_count;
    const int output_offset = input_count * pot.model_hidden1;
    const int bias_offset = output_offset + pot.model_hidden1;
    double z = pot.model_coeffs[bias_offset + hidden];
    double dz = 0.0;
    for (int input = 0; input < input_count; ++input) {
        const double weight = pot.model_coeffs[hidden * input_count + input];
        z += weight * mtpm_descriptor_value(pot, mom, itype, input);
        if (dir != nullptr) {
            dz += weight * mtpm_descriptor_direction(pot, dir, input);
        }
    }
    mtpm_activation(pot.model_activation, z, value, first, second);
    *direction = dz;
}

__device__ double mtpm_nn1_energy(const DevicePotential &pot,
                                  const double *mom, int itype)
{
    const int output_offset = pot.descriptor_count * pot.model_hidden1;
    double energy = 0.0;
    for (int hidden = 0; hidden < pot.model_hidden1; ++hidden) {
        double value, first, second, direction;
        mtpm_nn1_hidden(pot, mom, nullptr, itype, hidden,
                        &value, &first, &second, &direction);
        energy += pot.model_coeffs[output_offset + hidden] * value;
    }
    return energy;
}

__device__ double mtpm_nn1_gradient_component(
    const DevicePotential &pot, const double *mom, int itype, int input)
{
    const int input_count = pot.descriptor_count;
    const int output_offset = input_count * pot.model_hidden1;
    double gradient = 0.0;
    for (int hidden = 0; hidden < pot.model_hidden1; ++hidden) {
        double value, first, second, direction;
        mtpm_nn1_hidden(pot, mom, nullptr, itype, hidden,
                        &value, &first, &second, &direction);
        gradient += pot.model_coeffs[output_offset + hidden] * first *
            pot.model_coeffs[hidden * input_count + input];
    }
    return gradient;
}

__device__ double mtpm_nn1_hessian_direction_component(
    const DevicePotential &pot, const double *mom, const double *dir,
    int itype, int input)
{
    const int input_count = pot.descriptor_count;
    const int output_offset = input_count * pot.model_hidden1;
    double result = 0.0;
    for (int hidden = 0; hidden < pot.model_hidden1; ++hidden) {
        double value, first, second, direction;
        mtpm_nn1_hidden(pot, mom, dir, itype, hidden,
                        &value, &first, &second, &direction);
        result += pot.model_coeffs[output_offset + hidden] * second *
            pot.model_coeffs[hidden * input_count + input] * direction;
    }
    return result;
}

__device__ void mtpm_nn2_first_hidden(
    const DevicePotential &pot, const double *mom, const double *dir,
    int itype, int hidden, double *value, double *first, double *second,
    double *direction)
{
    const int input_count = pot.descriptor_count;
    const int bias_offset = input_count * pot.model_hidden1;
    double z = pot.model_coeffs[bias_offset + hidden];
    double dz = 0.0;
    for (int input = 0; input < input_count; ++input) {
        const double weight = pot.model_coeffs[hidden * input_count + input];
        z += weight * mtpm_descriptor_value(pot, mom, itype, input);
        if (dir != nullptr) {
            dz += weight * mtpm_descriptor_direction(pot, dir, input);
        }
    }
    mtpm_activation(pot.model_activation, z, value, first, second);
    *direction = dz;
}

__device__ void mtpm_nn2_second_hidden(
    const DevicePotential &pot, const double *mom, const double *dir,
    int itype, int hidden, double *value, double *first, double *second,
    double *direction)
{
    const int input_count = pot.descriptor_count;
    const int first_bias_offset = input_count * pot.model_hidden1;
    const int second_weight_offset = first_bias_offset + pot.model_hidden1;
    const int second_bias_offset = second_weight_offset +
        pot.model_hidden1 * pot.model_hidden2;
    double z = pot.model_coeffs[second_bias_offset + hidden];
    double dz = 0.0;
    for (int first_hidden = 0; first_hidden < pot.model_hidden1;
         ++first_hidden) {
        double h1, d1, d2, dh1_input;
        mtpm_nn2_first_hidden(pot, mom, dir, itype, first_hidden,
                              &h1, &d1, &d2, &dh1_input);
        const double weight = pot.model_coeffs[
            second_weight_offset + hidden * pot.model_hidden1 + first_hidden];
        z += weight * h1;
        dz += weight * d1 * dh1_input;
    }
    mtpm_activation(pot.model_activation, z, value, first, second);
    *direction = dz;
}

template <bool FullCache = false>
__host__ __device__ int mtpm_neural_cache_stride(
    const DevicePotential &pot)
{
    if constexpr (!FullCache) {
        if (pot.inference_only) {
            if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1))
                return 2 * pot.descriptor_count + 2 * pot.model_hidden1;
            if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2))
                return 2 * pot.descriptor_count + 3 * pot.model_hidden1 + 2 * pot.model_hidden2;
        }
    }
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)) {
        // q, Dq, dE/dq, H*Dq, then h, f', f'', Dz.
        return 4 * pot.descriptor_count + 4 * pot.model_hidden1;
    }
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2)) {
        // Common descriptor data; four values per neuron in each hidden
        // layer; and first-layer reverse seeds plus their directions.
        return 4 * pot.descriptor_count + 6 * pot.model_hidden1 +
            4 * pot.model_hidden2;
    }
    return 0;
}

struct NeuralCachePointers {
    double *q = nullptr;
    double *dq = nullptr;
    double *gradient = nullptr;
    double *hessian_direction = nullptr;
    double *h1 = nullptr;
    double *d1 = nullptr;
    double *dd1 = nullptr;
    double *du = nullptr;
    double *h2 = nullptr;
    double *d2 = nullptr;
    double *dd2 = nullptr;
    double *dz = nullptr;
    double *upstream = nullptr;
    double *upstream_direction = nullptr;
};

template <bool FullCache = false>
__device__ NeuralCachePointers mtpm_neural_cache_pointers(
    const DevicePotential &pot, double *base)
{
    NeuralCachePointers cache;
    cache.q = base;
    if constexpr (!FullCache) {
        if (pot.inference_only) {
            cache.gradient = cache.q + pot.descriptor_count;
            cache.h1 = cache.gradient + pot.descriptor_count;
            cache.d1 = cache.h1 + pot.model_hidden1;
            if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2)) {
                cache.h2 = cache.d1 + pot.model_hidden1;
                cache.d2 = cache.h2 + pot.model_hidden2;
                cache.upstream = cache.d2 + pot.model_hidden2;
            }
            return cache;
        }
    }
    cache.dq = cache.q + pot.descriptor_count;
    cache.gradient = cache.dq + pot.descriptor_count;
    cache.hessian_direction = cache.gradient + pot.descriptor_count;
    cache.h1 = cache.hessian_direction + pot.descriptor_count;
    cache.d1 = cache.h1 + pot.model_hidden1;
    cache.dd1 = cache.d1 + pot.model_hidden1;
    cache.du = cache.dd1 + pot.model_hidden1;
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2)) {
        cache.h2 = cache.du + pot.model_hidden1;
        cache.d2 = cache.h2 + pot.model_hidden2;
        cache.dd2 = cache.d2 + pot.model_hidden2;
        cache.dz = cache.dd2 + pot.model_hidden2;
        cache.upstream = cache.dz + pot.model_hidden2;
        cache.upstream_direction = cache.upstream + pot.model_hidden1;
    }
    return cache;
}

// Compact inference cache: no directional or second-derivative arrays.
// A one-thread block also serves the ordered diagnostic path.
template <bool ParallelDescriptors>
__device__ void mtpm_build_inference_cache(
    const DevicePotential &pot, int itype, const double *mom, double *base)
{
    const int tid = threadIdx.x;
    const auto c = mtpm_neural_cache_pointers(pot, base);
    if (ParallelDescriptors && blockDim.x >= propagation_warp_size) {
        const int lane = tid % propagation_warp_size;
        const int warp = tid / propagation_warp_size;
        const int warps = blockDim.x / propagation_warp_size;
        for (int q = warp; q < pot.descriptor_count; q += warps) {
            double value = 0.0;
            for (int a = lane; a < pot.alpha_scalar_moments; a += propagation_warp_size)
                value += pot.descriptor_moment_coeffs[q * pot.alpha_scalar_moments + a] *
                    mom[pot.alpha_moment_mapping[a]];
            value = warp_sum(value);
            if (lane == 0) c.q[q] = value + pot.descriptor_species_coeffs[q * pot.species_count + itype];
        }
    } else {
        for (int q = tid; q < pot.descriptor_count; q += blockDim.x)
            c.q[q] = mtpm_descriptor_value(pot, mom, itype, q);
    }
    __syncthreads();
    const bool nn1 = pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1);
    const int first_end = pot.descriptor_count * pot.model_hidden1;
    const int b1 = first_end + (nn1 ? pot.model_hidden1 : 0);
    for (int h = tid; h < pot.model_hidden1; h += blockDim.x) {
        double z = pot.model_coeffs[b1 + h], unused;
        for (int q = 0; q < pot.descriptor_count; ++q)
            z += pot.model_coeffs[h * pot.descriptor_count + q] * c.q[q];
        mtpm_activation(pot.model_activation, z, &c.h1[h], &c.d1[h], &unused);
    }
    __syncthreads();
    if (!nn1) {
        const int w2 = first_end + pot.model_hidden1;
        const int b2 = w2 + pot.model_hidden1 * pot.model_hidden2;
        const int out = b2 + pot.model_hidden2;
        for (int h2 = tid; h2 < pot.model_hidden2; h2 += blockDim.x) {
            double z = pot.model_coeffs[b2 + h2], unused;
            for (int h1 = 0; h1 < pot.model_hidden1; ++h1)
                z += pot.model_coeffs[w2 + h2 * pot.model_hidden1 + h1] * c.h1[h1];
            mtpm_activation(pot.model_activation, z, &c.h2[h2], &c.d2[h2], &unused);
        }
        __syncthreads();
        for (int h1 = tid; h1 < pot.model_hidden1; h1 += blockDim.x) {
            double u = 0.0;
            for (int h2 = 0; h2 < pot.model_hidden2; ++h2)
                u += pot.model_coeffs[out + h2] * c.d2[h2] *
                    pot.model_coeffs[w2 + h2 * pot.model_hidden1 + h1];
            c.upstream[h1] = u;
        }
        __syncthreads();
    }
    for (int q = tid; q < pot.descriptor_count; q += blockDim.x) {
        double gradient = 0.0;
        for (int h = 0; h < pot.model_hidden1; ++h)
            gradient += (nn1 ? pot.model_coeffs[first_end + h] : c.upstream[h]) *
                c.d1[h] * pot.model_coeffs[h * pot.descriptor_count + q];
        c.gradient[q] = gradient;
    }
    __syncthreads();
}

// All threads of one center's block participate. The ordered diagnostic
// wrapper retains serial descriptor reductions; production uses one warp
// per descriptor, including partial scalar/descriptor tiles.
template <bool ParallelDescriptors, bool FullCache = false>
__device__ void mtpm_build_neural_cache(
    const DevicePotential &pot, int itype, const double *mom,
    const double *dir, double *cache_base)
{
    if constexpr (!FullCache) {
        if (pot.inference_only) {
            mtpm_build_inference_cache<ParallelDescriptors>(pot, itype, mom, cache_base);
            return;
        }
    }
    const int tid = threadIdx.x;
    NeuralCachePointers cache = mtpm_neural_cache_pointers<FullCache>(pot, cache_base);
    if (ParallelDescriptors && blockDim.x >= propagation_warp_size) {
        const int lane = tid % propagation_warp_size;
        const int warp = tid / propagation_warp_size;
        const int warps = blockDim.x / propagation_warp_size;
        for (int input = warp; input < pot.descriptor_count; input += warps) {
            double q = 0.0;
            double dq = 0.0;
            for (int scalar = lane; scalar < pot.alpha_scalar_moments;
                 scalar += propagation_warp_size) {
                const int mid = pot.alpha_moment_mapping[scalar];
                const double weight = pot.descriptor_moment_coeffs[
                    input * pot.alpha_scalar_moments + scalar];
                q += weight * mom[mid];
                if (dir != nullptr) dq += weight * dir[mid];
            }
            q = warp_sum(q);
            dq = warp_sum(dq);
            if (lane == 0) {
                cache.q[input] = q + pot.descriptor_species_coeffs[
                    input * pot.species_count + itype];
                cache.dq[input] = dq;
            }
        }
    } else {
        for (int input = tid; input < pot.descriptor_count;
             input += blockDim.x) {
            cache.q[input] = mtpm_descriptor_value(pot, mom, itype, input);
            cache.dq[input] = dir != nullptr
                ? mtpm_descriptor_direction(pot, dir, input) : 0.0;
        }
    }
    __syncthreads();

    const int first_bias_offset =
        pot.descriptor_count * pot.model_hidden1;
    const int nn1_output_offset = first_bias_offset;
    const int nn1_bias_offset = nn1_output_offset + pot.model_hidden1;
    for (int hidden = tid; hidden < pot.model_hidden1;
         hidden += blockDim.x) {
        const int bias_offset =
            pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)
                ? nn1_bias_offset : first_bias_offset;
        double z = pot.model_coeffs[bias_offset + hidden];
        double dz = 0.0;
        for (int input = 0; input < pot.descriptor_count; ++input) {
            const double weight = pot.model_coeffs[
                hidden * pot.descriptor_count + input];
            z += weight * cache.q[input];
            dz += weight * cache.dq[input];
        }
        mtpm_activation(pot.model_activation, z,
                        &cache.h1[hidden], &cache.d1[hidden],
                        &cache.dd1[hidden]);
        cache.du[hidden] = dz;
    }
    __syncthreads();

    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)) {
        for (int input = tid; input < pot.descriptor_count;
             input += blockDim.x) {
            double gradient = 0.0;
            double hessian_direction = 0.0;
            for (int hidden = 0; hidden < pot.model_hidden1; ++hidden) {
                const double output_weight =
                    pot.model_coeffs[nn1_output_offset + hidden];
                const double input_weight = pot.model_coeffs[
                    hidden * pot.descriptor_count + input];
                gradient += output_weight * cache.d1[hidden] * input_weight;
                hessian_direction += output_weight * cache.dd1[hidden] *
                    input_weight * cache.du[hidden];
            }
            cache.gradient[input] = gradient;
            cache.hessian_direction[input] = hessian_direction;
        }
        __syncthreads();
        return;
    }

    const int second_weight_offset =
        first_bias_offset + pot.model_hidden1;
    const int second_bias_offset = second_weight_offset +
        pot.model_hidden1 * pot.model_hidden2;
    const int output_offset = second_bias_offset + pot.model_hidden2;
    for (int hidden2 = tid; hidden2 < pot.model_hidden2;
         hidden2 += blockDim.x) {
        double z = pot.model_coeffs[second_bias_offset + hidden2];
        double dz = 0.0;
        for (int hidden1 = 0; hidden1 < pot.model_hidden1; ++hidden1) {
            const double weight = pot.model_coeffs[
                second_weight_offset + hidden2 * pot.model_hidden1 + hidden1];
            z += weight * cache.h1[hidden1];
            dz += weight * cache.d1[hidden1] * cache.du[hidden1];
        }
        mtpm_activation(pot.model_activation, z,
                        &cache.h2[hidden2], &cache.d2[hidden2],
                        &cache.dd2[hidden2]);
        cache.dz[hidden2] = dz;
    }
    __syncthreads();

    for (int hidden1 = tid; hidden1 < pot.model_hidden1;
         hidden1 += blockDim.x) {
        double upstream = 0.0;
        double upstream_direction = 0.0;
        for (int hidden2 = 0; hidden2 < pot.model_hidden2; ++hidden2) {
            const double output_weight =
                pot.model_coeffs[output_offset + hidden2];
            const double second_weight = pot.model_coeffs[
                second_weight_offset + hidden2 * pot.model_hidden1 + hidden1];
            upstream += output_weight * cache.d2[hidden2] * second_weight;
            upstream_direction += output_weight * cache.dd2[hidden2] *
                cache.dz[hidden2] * second_weight;
        }
        cache.upstream[hidden1] = upstream;
        cache.upstream_direction[hidden1] = upstream_direction;
    }
    __syncthreads();

    for (int input = tid; input < pot.descriptor_count;
         input += blockDim.x) {
        double gradient = 0.0;
        double hessian_direction = 0.0;
        for (int hidden1 = 0; hidden1 < pot.model_hidden1; ++hidden1) {
            const double first_weight = pot.model_coeffs[
                hidden1 * pot.descriptor_count + input];
            const double delta =
                cache.d1[hidden1] * cache.upstream[hidden1];
            const double delta_direction =
                cache.dd1[hidden1] * cache.du[hidden1] *
                    cache.upstream[hidden1] +
                cache.d1[hidden1] * cache.upstream_direction[hidden1];
            gradient += first_weight * delta;
            hessian_direction += first_weight * delta_direction;
        }
        cache.gradient[input] = gradient;
        cache.hessian_direction[input] = hessian_direction;
    }
    __syncthreads();
}

__global__ void mtpm_build_neural_cache_kernel(
    DevicePotential pot, int inum, const int *centers, const int *type,
    const double *moment_vals, const double *directional,
    double *neural_cache)
{
    const int center_idx = blockIdx.x;
    if (center_idx >= inum) return;
    mtpm_build_neural_cache<false>(
        pot, pot.spmap[type[centers[center_idx]]],
        moment_vals + center_idx * pot.alpha_moments_count,
        directional + center_idx * pot.alpha_moments_count,
        neural_cache + center_idx * mtpm_neural_cache_stride(pot));
}

// Each scalar owns its seed, unless a caller supplies repeated mappings.
// Retain the old scalar order for that uncommon case rather than introducing
// races or atomic reductions into the cache-to-moment contraction.
__device__ void mtpm_seed_neural_adjoint(
    const DevicePotential &pot, const double *seed, double *adj)
{
    const int tid = threadIdx.x;
    if (!pot.unique_scalar_mapping && tid != 0) return;
    const int stride = pot.unique_scalar_mapping ? blockDim.x : 1;
    for (int scalar = tid; scalar < pot.alpha_scalar_moments; scalar += stride) {
        double value = 0.0;
        for (int input = 0; input < pot.descriptor_count; ++input) {
            value += seed[input] * pot.descriptor_moment_coeffs[
                input * pot.alpha_scalar_moments + scalar];
        }
        adj[pot.alpha_moment_mapping[scalar]] += value;
    }
}

__device__ double mtpm_neural_energy_from_cache(
    const DevicePotential &pot, const NeuralCachePointers &cache)
{
    const bool nn1 = pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1);
    const int output_offset = nn1 ? pot.descriptor_count * pot.model_hidden1
        : pot.descriptor_count * pot.model_hidden1 + pot.model_hidden1 +
            pot.model_hidden1 * pot.model_hidden2 + pot.model_hidden2;
    const int count = nn1 ? pot.model_hidden1 : pot.model_hidden2;
    const double *hidden = nn1 ? cache.h1 : cache.h2;
    double energy = 0.0;
    for (int h = 0; h < count; ++h) {
        energy += pot.model_coeffs[output_offset + h] * hidden[h];
    }
    return energy;
}

__device__ double mtpm_nn2_energy(const DevicePotential &pot,
                                  const double *mom, int itype)
{
    const int output_offset = pot.descriptor_count * pot.model_hidden1 +
        pot.model_hidden1 + pot.model_hidden1 * pot.model_hidden2 +
        pot.model_hidden2;
    double energy = 0.0;
    for (int hidden = 0; hidden < pot.model_hidden2; ++hidden) {
        double value, first, second, direction;
        mtpm_nn2_second_hidden(pot, mom, nullptr, itype, hidden,
                               &value, &first, &second, &direction);
        energy += pot.model_coeffs[output_offset + hidden] * value;
    }
    return energy;
}

__device__ double mtpm_nn2_gradient_component(
    const DevicePotential &pot, const double *mom, int itype, int input)
{
    const int first_bias_offset = pot.descriptor_count * pot.model_hidden1;
    const int second_weight_offset = first_bias_offset + pot.model_hidden1;
    const int output_offset = second_weight_offset +
        pot.model_hidden1 * pot.model_hidden2 + pot.model_hidden2;
    double gradient = 0.0;
    for (int hidden2 = 0; hidden2 < pot.model_hidden2; ++hidden2) {
        double h2, d2, dd2, unused;
        mtpm_nn2_second_hidden(pot, mom, nullptr, itype, hidden2,
                               &h2, &d2, &dd2, &unused);
        double dz_dq = 0.0;
        for (int hidden1 = 0; hidden1 < pot.model_hidden1; ++hidden1) {
            double h1, d1, dd1, direction;
            mtpm_nn2_first_hidden(pot, mom, nullptr, itype, hidden1,
                                  &h1, &d1, &dd1, &direction);
            dz_dq += pot.model_coeffs[
                second_weight_offset + hidden2 * pot.model_hidden1 + hidden1] *
                d1 * pot.model_coeffs[
                    hidden1 * pot.descriptor_count + input];
        }
        gradient += pot.model_coeffs[output_offset + hidden2] * d2 * dz_dq;
    }
    return gradient;
}

__device__ double mtpm_nn2_hessian_direction_component(
    const DevicePotential &pot, const double *mom, const double *dir,
    int itype, int input)
{
    const int first_bias_offset = pot.descriptor_count * pot.model_hidden1;
    const int second_weight_offset = first_bias_offset + pot.model_hidden1;
    const int output_offset = second_weight_offset +
        pot.model_hidden1 * pot.model_hidden2 + pot.model_hidden2;
    double result = 0.0;
    for (int hidden2 = 0; hidden2 < pot.model_hidden2; ++hidden2) {
        double h2, d2, dd2, dz_direction;
        mtpm_nn2_second_hidden(pot, mom, dir, itype, hidden2,
                               &h2, &d2, &dd2, &dz_direction);
        double dz_dq = 0.0;
        double d_dz_dq = 0.0;
        for (int hidden1 = 0; hidden1 < pot.model_hidden1; ++hidden1) {
            double h1, d1, dd1, du_direction;
            mtpm_nn2_first_hidden(pot, mom, dir, itype, hidden1,
                                  &h1, &d1, &dd1, &du_direction);
            const double second_weight = pot.model_coeffs[
                second_weight_offset + hidden2 * pot.model_hidden1 + hidden1];
            const double first_weight = pot.model_coeffs[
                hidden1 * pot.descriptor_count + input];
            dz_dq += second_weight * d1 * first_weight;
            d_dz_dq += second_weight * dd1 * du_direction * first_weight;
        }
        result += pot.model_coeffs[output_offset + hidden2] *
            (dd2 * dz_direction * dz_dq + d2 * d_dz_dq);
    }
    return result;
}

__device__ double mtpm_external_model_energy(
    const DevicePotential &pot, const double *mom, int itype)
{
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)) {
        return mtpm_nn1_energy(pot, mom, itype);
    }
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2)) {
        return mtpm_nn2_energy(pot, mom, itype);
    }
    double q0 = 0.0;
    double q1 = 0.0;
    mtpm_descriptor_values(pot, mom, itype, &q0, &q1);
    double energy, d0, d1, h00, h01, h11;
    mtpm_model_derivatives(pot, q0, q1, &energy, &d0, &d1,
                           &h00, &h01, &h11);
    return energy;
}

__device__ double mtpm_external_model_gradient_component(
    const DevicePotential &pot, const double *mom, int itype, int descriptor)
{
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)) {
        return mtpm_nn1_gradient_component(pot, mom, itype, descriptor);
    }
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2)) {
        return mtpm_nn2_gradient_component(pot, mom, itype, descriptor);
    }
    double q0 = 0.0;
    double q1 = 0.0;
    mtpm_descriptor_values(pot, mom, itype, &q0, &q1);
    double energy, d0, d1, h00, h01, h11;
    mtpm_model_derivatives(pot, q0, q1, &energy, &d0, &d1,
                           &h00, &h01, &h11);
    return descriptor == 0 ? d0 : d1;
}

__device__ double mtpm_external_model_hessian_direction_component(
    const DevicePotential &pot, const double *mom, const double *dir,
    int itype, int descriptor)
{
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMLinear)) return 0.0;
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)) {
        return mtpm_nn1_hessian_direction_component(
            pot, mom, dir, itype, descriptor);
    }
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2)) {
        return mtpm_nn2_hessian_direction_component(
            pot, mom, dir, itype, descriptor);
    }
    const double q0 = mtpm_descriptor_value(pot, mom, itype, 0);
    const double q1 = mtpm_descriptor_value(pot, mom, itype, 1);
    const double dq0 = mtpm_descriptor_direction(pot, dir, 0);
    const double dq1 = mtpm_descriptor_direction(pot, dir, 1);
    double energy, d0, d1, h00, h01, h11;
    mtpm_model_derivatives(pot, q0, q1, &energy, &d0, &d1,
                           &h00, &h01, &h11);
    return descriptor == 0 ? h00 * dq0 + h01 * dq1
                           : h01 * dq0 + h11 * dq1;
}

__device__ void mtpm_seed_energy_and_adjoint(
    const DevicePotential &pot, const double *mom, int itype,
    double *adj, double *model_energy)
{
    if (pot.model_type != static_cast<int>(MTPGpuModel::MTPMNN1) &&
        pot.model_type != static_cast<int>(MTPGpuModel::MTPMNN2)) {
        // Analytic heads need just one projection, not one per scalar seed.
        double q0, q1, d0, d1, h00, h01, h11;
        mtpm_descriptor_values(pot, mom, itype, &q0, &q1);
        mtpm_model_derivatives(pot, q0, q1, model_energy,
                              &d0, &d1, &h00, &h01, &h11);
        for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
            double seed = d0 * pot.descriptor_moment_coeffs[scalar];
            if (pot.descriptor_count > 1)
                seed += d1 * pot.descriptor_moment_coeffs[pot.alpha_scalar_moments + scalar];
            adj[pot.alpha_moment_mapping[scalar]] += seed;
        }
        return;
    }
    *model_energy = mtpm_external_model_energy(pot, mom, itype);
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        const int mid = pot.alpha_moment_mapping[scalar];
        double seed = 0.0;
        for (int descriptor = 0; descriptor < pot.descriptor_count;
             ++descriptor) {
            seed += mtpm_external_model_gradient_component(
                pot, mom, itype, descriptor) *
                pot.descriptor_moment_coeffs[
                    descriptor * pot.alpha_scalar_moments + scalar];
        }
        adj[mid] += seed;
    }
}

__device__ void mtpm_seed_energy_and_adjoint_cached(
    const DevicePotential &pot, const double *mom, int itype,
    double *adj, double *model_energy, double *cache_base)
{
    NeuralCachePointers cache = mtpm_neural_cache_pointers(pot, cache_base);
    *model_energy = mtpm_external_model_energy(pot, mom, itype);
    for (int descriptor = 0; descriptor < pot.descriptor_count;
         ++descriptor) {
        cache.gradient[descriptor] = mtpm_external_model_gradient_component(
            pot, mom, itype, descriptor);
    }
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        const int mid = pot.alpha_moment_mapping[scalar];
        double seed = 0.0;
        for (int descriptor = 0; descriptor < pot.descriptor_count;
             ++descriptor) {
            seed += cache.gradient[descriptor] *
                pot.descriptor_moment_coeffs[
                    descriptor * pot.alpha_scalar_moments + scalar];
        }
        adj[mid] += seed;
    }
}

__device__ void mtpm_seed_directional_adjoint(
    const DevicePotential &pot, const double *mom, const double *dir,
    int itype, double *hadj)
{
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        const int mid = pot.alpha_moment_mapping[scalar];
        double seed = 0.0;
        for (int descriptor = 0; descriptor < pot.descriptor_count;
             ++descriptor) {
            seed += pot.descriptor_moment_coeffs[
                        descriptor * pot.alpha_scalar_moments + scalar] *
                mtpm_external_model_hessian_direction_component(
                    pot, mom, dir, itype, descriptor);
        }
        hadj[mid] += seed;
    }
}

__device__ void mtpm_seed_directional_adjoint_cached(
    const DevicePotential &pot, const double *mom, const double *dir,
    int itype, double *hadj, double *cache_base)
{
    NeuralCachePointers cache = mtpm_neural_cache_pointers(pot, cache_base);
    for (int descriptor = 0; descriptor < pot.descriptor_count;
         ++descriptor) {
        cache.hessian_direction[descriptor] =
            mtpm_external_model_hessian_direction_component(
                pot, mom, dir, itype, descriptor);
    }
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        const int mid = pot.alpha_moment_mapping[scalar];
        double seed = 0.0;
        for (int descriptor = 0; descriptor < pot.descriptor_count;
             ++descriptor) {
            seed += pot.descriptor_moment_coeffs[
                        descriptor * pot.alpha_scalar_moments + scalar] *
                cache.hessian_direction[descriptor];
        }
        hadj[mid] += seed;
    }
}

// Reuse the block reduction storage for analytic descriptor projections.
__device__ void mtpm_seed_analytic_parallel(const DevicePotential &pot,
    int itype, const double *mom, double *adj, double *head, double *work)
{
    const int tid = threadIdx.x;
    for (int q = 0; q < pot.descriptor_count; ++q) {
        double value = 0.0;
        for (int a = tid; a < pot.alpha_scalar_moments; a += blockDim.x)
            value += pot.descriptor_moment_coeffs[q * pot.alpha_scalar_moments + a] *
                mom[pot.alpha_moment_mapping[a]];
        work[tid] = value;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride; stride >>= 1) {
            if (tid < stride) work[tid] += work[tid + stride];
            __syncthreads();
        }
        if (tid == 0) head[q] = work[0] + pot.descriptor_species_coeffs[q * pot.species_count + itype];
        __syncthreads();
    }
    if (tid == 0) {
        double energy, d0, d1, h00, h01, h11;
        mtpm_model_derivatives(pot, head[0], pot.descriptor_count > 1 ? head[1] : 0.0,
            &energy, &d0, &d1, &h00, &h01, &h11);
        head[0] = d0; head[1] = d1; head[2] = energy;
    }
    __syncthreads();
    if (pot.unique_scalar_mapping || tid == 0) {
        const int stride = pot.unique_scalar_mapping ? blockDim.x : 1;
        for (int a = tid; a < pot.alpha_scalar_moments; a += stride) {
            double seed = head[0] * pot.descriptor_moment_coeffs[a];
            if (pot.descriptor_count > 1)
                seed += head[1] * pot.descriptor_moment_coeffs[pot.alpha_scalar_moments + a];
            adj[pot.alpha_moment_mapping[a]] += seed;
        }
    }
}

template <typename MomentReal, typename AdjointReal,
          bool Partitioned = false, bool LinearOnly = false,
          bool FullNeuralCache = false>
__global__ void propagate_kernel(DevicePotential pot, const int *centers,
                                 const int *type, MomentReal *moment_vals,
                                 AdjointReal *dmom, double *site_energy,
                                 double *total_energy,
                                 double *neural_cache, int phase = 0)
{
    static_assert(!FullNeuralCache || (!LinearOnly &&
        std::is_same_v<MomentReal, double> && std::is_same_v<AdjointReal, double>));
    extern __shared__ double block_sum[];
    __shared__ double analytic_head[3];
    const int center_idx = blockIdx.x;
    const int tid = threadIdx.x;
    // The per-model stride is validated to fit int; only the flattened offset
    // must stay wide when the center count crosses the old 32-bit limit.
    const int moment_stride = static_cast<int>(pot.alpha_moments_count);
    const size_t moment_offset = static_cast<size_t>(center_idx) *
        static_cast<unsigned int>(moment_stride);
    MomentReal *mom = moment_vals + moment_offset;
    AdjointReal *adj = dmom + moment_offset;

    if (phase != 2) {
    for (int level = 1; level <= pot.max_level; ++level) {
        const int begin = pot.forward_level_offsets[level];
        const int end = pot.forward_level_offsets[level + 1];
        const bool classified = Partitioned && blockDim.x >= propagation_warp_size;
        const int short_begin = classified ? pot.forward_short_level_offsets[level] : begin;
        const int short_end = classified ? pot.forward_short_level_offsets[level + 1] : end;
        for (int index = short_begin + tid; index < short_end;
             index += blockDim.x) {
            const int group = classified ? pot.forward_short_groups[index] : index;
            const int term_begin = pot.forward_term_offsets[group];
            const int term_end = pot.forward_term_offsets[group + 1];
            if (!Partitioned && blockDim.x >= propagation_warp_size &&
                term_end - term_begin > propagation_warp_group_threshold) {
                continue;
            }
            double sum = 0.0;
            for (int pos = term_begin; pos < term_end; ++pos) {
                const ForwardTerm term = pot.forward_terms[pos];
                sum += static_cast<double>(term.c) *
                    mom[term.l1] * mom[term.l2];
            }
            mom[pot.forward_ids[group]] += sum;
        }
        if (blockDim.x >= propagation_warp_size) {
            const int lane = tid & (propagation_warp_size - 1);
            const int warp = tid / propagation_warp_size;
            const int warps = blockDim.x / propagation_warp_size;
            const int warp_begin = Partitioned ? pot.forward_warp_level_offsets[level] : begin;
            const int warp_end = Partitioned ? pot.forward_warp_level_offsets[level + 1] : end;
            for (int index = warp_begin + warp; index < warp_end; index += warps) {
                const int group = Partitioned ? pot.forward_warp_groups[index] : index;
                const int term_begin = pot.forward_term_offsets[group];
                const int term_end = pot.forward_term_offsets[group + 1];
                const int term_count = term_end - term_begin;
                if (!Partitioned && (term_count <= propagation_warp_group_threshold ||
                    term_count > propagation_block_group_threshold)) {
                    continue;
                }
                double sum = 0.0;
                for (int pos = term_begin + lane; pos < term_end;
                     pos += propagation_warp_size) {
                    const ForwardTerm term = pot.forward_terms[pos];
                    sum += static_cast<double>(term.c) *
                        mom[term.l1] * mom[term.l2];
                }
                sum = warp_sum(sum);
                if (lane == 0) mom[pot.forward_ids[group]] += sum;
            }
            const int block_begin =
                pot.forward_block_level_offsets[level];
            const int block_end =
                pot.forward_block_level_offsets[level + 1];
            const int group_warps = warps < propagation_multiwarp_group_warps
                ? warps : propagation_multiwarp_group_warps;
            const int group_threads = group_warps * propagation_warp_size;
            const int groups_per_batch = warps / group_warps;
            const int group_slot = tid / group_threads;
            const int group_lane = tid % group_threads;
            for (int batch = block_begin; batch < block_end;
                 batch += groups_per_batch) {
                const int block_group = batch + group_slot;
                const bool active = block_group < block_end;
                int group = 0;
                int term_begin = 0;
                int term_end = 0;
                if (active) {
                    group = pot.forward_block_groups[block_group];
                    term_begin = pot.forward_term_offsets[group];
                    term_end = pot.forward_term_offsets[group + 1];
                }
                double sum = 0.0;
                for (int pos = term_begin + group_lane; pos < term_end;
                     pos += group_threads) {
                    const ForwardTerm term = pot.forward_terms[pos];
                    sum += static_cast<double>(term.c) *
                        mom[term.l1] * mom[term.l2];
                }
                sum = warp_sum(sum);
                if (lane == 0) block_sum[warp] = sum;
                __syncthreads();
                if (active && group_lane == 0) {
                    double group_sum = 0.0;
                    const int first_warp = group_slot * group_warps;
                    for (int index = 0; index < group_warps; ++index) {
                        group_sum += block_sum[first_warp + index];
                    }
                    mom[pot.forward_ids[group]] += group_sum;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    if constexpr (FullNeuralCache) {
        // Native/batch NN requests retain the training cache. Keep the compact
        // inference and analytic heads out of this kernel's register budget.
        const int ati = centers[center_idx];
        const int itype = pot.spmap[type[ati]];
        double *cache_base = neural_cache +
            center_idx * mtpm_neural_cache_stride<true>(pot);
        mtpm_build_neural_cache<true, true>(pot, itype, mom, nullptr, cache_base);
        const auto cache = mtpm_neural_cache_pointers<true>(pot, cache_base);
        mtpm_seed_neural_adjoint(pot, cache.gradient, adj);
        if (tid == 0 && (site_energy != nullptr || total_energy != nullptr)) {
            const double energy = pot.species_coeffs[itype] +
                mtpm_neural_energy_from_cache(pot, cache);
            const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
            if (site_energy != nullptr) site_energy[ati] = weight * energy;
            if (total_energy != nullptr) atomic_add_double(total_energy, weight * energy);
        }
    } else if (LinearOnly || pot.model_type == static_cast<int>(MTPGpuModel::Linear)) {
        const bool need_energy =
            site_energy != nullptr || total_energy != nullptr;
        double local_energy = 0.0;
        for (int s = tid; s < pot.alpha_scalar_moments; s += blockDim.x) {
            const int mid = pot.alpha_moment_mapping[s];
            const double coeff = pot.moment_coeffs[s];
            if (need_energy) local_energy += coeff * mom[mid];
            if (pot.unique_scalar_mapping) adj[mid] = static_cast<AdjointReal>(coeff);
        }
        if (!pot.unique_scalar_mapping && tid == 0) {
            for (int s = 0; s < pot.alpha_scalar_moments; ++s)
                adj[pot.alpha_moment_mapping[s]] += static_cast<AdjointReal>(pot.moment_coeffs[s]);
        }
        if (need_energy) {
            block_sum[tid] = local_energy;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) block_sum[tid] += block_sum[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const int ati = centers[center_idx];
                const int itype = pot.spmap[type[ati]];
                const double e =
                    (itype >= 0 ? pot.species_coeffs[itype] : 0.0) + block_sum[0];
                const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
                if (site_energy != nullptr) site_energy[ati] = weight * e;
                if (total_energy != nullptr) atomic_add_double(total_energy, weight * e);
            }
        }
    } else if (neural_cache != nullptr) {
        if constexpr (std::is_same_v<MomentReal, double> &&
                      std::is_same_v<AdjointReal, double>) {
            const int ati = centers[center_idx];
            const int itype = pot.spmap[type[ati]];
            double *cache_base = neural_cache +
                center_idx * mtpm_neural_cache_stride(pot);
            mtpm_build_neural_cache<true>(pot, itype, mom, nullptr, cache_base);
            const auto cache = mtpm_neural_cache_pointers(pot, cache_base);
            mtpm_seed_neural_adjoint(pot, cache.gradient, adj);
            if (tid == 0 && (site_energy != nullptr || total_energy != nullptr)) {
                const double energy = pot.species_coeffs[itype] +
                    mtpm_neural_energy_from_cache(pot, cache);
                const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
                if (site_energy != nullptr) site_energy[ati] = weight * energy;
                if (total_energy != nullptr) atomic_add_double(total_energy, weight * energy);
            }
        }
    } else if (pot.inference_only &&
        pot.model_type != static_cast<int>(MTPGpuModel::MTPMNN1) &&
        pot.model_type != static_cast<int>(MTPGpuModel::MTPMNN2)) {
        if constexpr (std::is_same_v<MomentReal, double> &&
                      std::is_same_v<AdjointReal, double>) {
            const int ati = centers[center_idx];
            const int itype = pot.spmap[type[ati]];
            mtpm_seed_analytic_parallel(pot, itype, mom, adj, analytic_head, block_sum);
            if (tid == 0) {
                const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
                const double energy = weight * (pot.species_coeffs[itype] + analytic_head[2]);
                if (site_energy) site_energy[ati] = energy;
                if (total_energy) atomic_add_double(total_energy, energy);
            }
        }
    } else if (tid == 0) {
        if constexpr (std::is_same_v<MomentReal, double> &&
                      std::is_same_v<AdjointReal, double>) {
            const int ati = centers[center_idx];
            const int itype = pot.spmap[type[ati]];
            double model_energy = 0.0;
            mtpm_seed_energy_and_adjoint(
                pot, mom, itype, adj, &model_energy);
            const double energy = pot.species_coeffs[itype] + model_energy;
            const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
                if (site_energy != nullptr) site_energy[ati] = weight * energy;
            if (total_energy != nullptr) atomic_add_double(total_energy, weight * energy);
        }
    }
    __syncthreads();

    if (pot.center_energy_weights != nullptr) {
        const double weight = pot.center_energy_weights[centers[center_idx]];
        for (int m = tid; m < pot.alpha_moments_count; m += blockDim.x) adj[m] *= weight;
        __syncthreads();
    }
    }
    if (phase == 1) return;
    for (int level = pot.max_level; level >= 1; --level) {
        const int begin = pot.reverse_level_offsets[level];
        const int end = pot.reverse_level_offsets[level + 1];
        const bool classified = Partitioned && blockDim.x >= propagation_warp_size;
        const int short_begin = classified ? pot.reverse_short_level_offsets[level] : begin;
        const int short_end = classified ? pot.reverse_short_level_offsets[level + 1] : end;
        for (int index = short_begin + tid; index < short_end;
             index += blockDim.x) {
            const int group = classified ? pot.reverse_short_groups[index] : index;
            const int entry_begin = pot.reverse_entry_offsets[group];
            const int entry_end = pot.reverse_entry_offsets[group + 1];
            if (!Partitioned && blockDim.x >= propagation_warp_size &&
                entry_end - entry_begin > propagation_warp_group_threshold) {
                continue;
            }
            double sum = 0.0;
            for (int pos = entry_begin; pos < entry_end; ++pos) {
                const ReverseTerm term = pot.reverse_entries[pos];
                sum += adj[term.source] *
                    static_cast<double>(term.c) * mom[term.other];
            }
            adj[pot.reverse_ids[group]] += sum;
        }
        if (blockDim.x >= propagation_warp_size) {
            const int lane = tid & (propagation_warp_size - 1);
            const int warp = tid / propagation_warp_size;
            const int warps = blockDim.x / propagation_warp_size;
            const int warp_begin = Partitioned ? pot.reverse_warp_level_offsets[level] : begin;
            const int warp_end = Partitioned ? pot.reverse_warp_level_offsets[level + 1] : end;
            for (int index = warp_begin + warp; index < warp_end; index += warps) {
                const int group = Partitioned ? pot.reverse_warp_groups[index] : index;
                const int entry_begin = pot.reverse_entry_offsets[group];
                const int entry_end = pot.reverse_entry_offsets[group + 1];
                const int entry_count = entry_end - entry_begin;
                if (!Partitioned && (entry_count <= propagation_warp_group_threshold ||
                    entry_count > propagation_block_group_threshold)) {
                    continue;
                }
                double sum = 0.0;
                for (int pos = entry_begin + lane; pos < entry_end;
                     pos += propagation_warp_size) {
                    const ReverseTerm term = pot.reverse_entries[pos];
                    sum += adj[term.source] *
                        static_cast<double>(term.c) * mom[term.other];
                }
                sum = warp_sum(sum);
                if (lane == 0) adj[pot.reverse_ids[group]] += sum;
            }
            const int block_begin =
                pot.reverse_block_level_offsets[level];
            const int block_end =
                pot.reverse_block_level_offsets[level + 1];
            const int group_warps = warps < propagation_multiwarp_group_warps
                ? warps : propagation_multiwarp_group_warps;
            const int group_threads = group_warps * propagation_warp_size;
            const int groups_per_batch = warps / group_warps;
            const int group_slot = tid / group_threads;
            const int group_lane = tid % group_threads;
            for (int batch = block_begin; batch < block_end;
                 batch += groups_per_batch) {
                const int block_group = batch + group_slot;
                const bool active = block_group < block_end;
                int group = 0;
                int entry_begin = 0;
                int entry_end = 0;
                if (active) {
                    group = pot.reverse_block_groups[block_group];
                    entry_begin = pot.reverse_entry_offsets[group];
                    entry_end = pot.reverse_entry_offsets[group + 1];
                }
                double sum = 0.0;
                for (int pos = entry_begin + group_lane; pos < entry_end;
                     pos += group_threads) {
                    const ReverseTerm term = pot.reverse_entries[pos];
                    sum += adj[term.source] *
                        static_cast<double>(term.c) * mom[term.other];
                }
                sum = warp_sum(sum);
                if (lane == 0) block_sum[warp] = sum;
                __syncthreads();
                if (active && group_lane == 0) {
                    double group_sum = 0.0;
                    const int first_warp = group_slot * group_warps;
                    for (int index = 0; index < group_warps; ++index) {
                        group_sum += block_sum[first_warp + index];
                    }
                    adj[pot.reverse_ids[group]] += group_sum;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

// Forward-only composite propagation for inference that does not request
// forces or virials.  It deliberately avoids allocating or constructing the
// energy-adjoint array used by direct_force_kernel.
template <typename WorkReal>
__global__ void propagate_energy_kernel(DevicePotential pot, const int *centers,
                                        const int *type, WorkReal *moment_vals,
                                        double *site_energy,
                                        double *total_energy, double *neural_cache)
{
    extern __shared__ double block_sum[];
    const int center_idx = blockIdx.x;
    const int tid = threadIdx.x;
    WorkReal *mom = moment_vals + center_idx * pot.alpha_moments_count;

    for (int level = 1; level <= pot.max_level; ++level) {
        const int begin = pot.forward_level_offsets[level];
        const int end = pot.forward_level_offsets[level + 1];
        for (int group = begin + tid; group < end;
             group += blockDim.x) {
            const int term_begin = pot.forward_term_offsets[group];
            const int term_end = pot.forward_term_offsets[group + 1];
            if (blockDim.x >= propagation_warp_size &&
                term_end - term_begin > propagation_warp_group_threshold) {
                continue;
            }
            double sum = 0.0;
            for (int pos = term_begin; pos < term_end; ++pos) {
                const ForwardTerm term = pot.forward_terms[pos];
                sum += static_cast<double>(term.c) *
                    mom[term.l1] * mom[term.l2];
            }
            mom[pot.forward_ids[group]] += sum;
        }
        if (blockDim.x >= propagation_warp_size) {
            const int lane = tid & (propagation_warp_size - 1);
            const int warp = tid / propagation_warp_size;
            const int warps = blockDim.x / propagation_warp_size;
            for (int group = begin + warp; group < end; group += warps) {
                const int term_begin = pot.forward_term_offsets[group];
                const int term_end = pot.forward_term_offsets[group + 1];
                const int term_count = term_end - term_begin;
                if (term_count <= propagation_warp_group_threshold ||
                    term_count > propagation_block_group_threshold) {
                    continue;
                }
                double sum = 0.0;
                for (int pos = term_begin + lane; pos < term_end;
                     pos += propagation_warp_size) {
                    const ForwardTerm term = pot.forward_terms[pos];
                    sum += static_cast<double>(term.c) *
                        mom[term.l1] * mom[term.l2];
                }
                sum = warp_sum(sum);
                if (lane == 0) mom[pot.forward_ids[group]] += sum;
            }
            const int block_begin =
                pot.forward_block_level_offsets[level];
            const int block_end =
                pot.forward_block_level_offsets[level + 1];
            const int group_warps = warps < propagation_multiwarp_group_warps
                ? warps : propagation_multiwarp_group_warps;
            const int group_threads = group_warps * propagation_warp_size;
            const int groups_per_batch = warps / group_warps;
            const int group_slot = tid / group_threads;
            const int group_lane = tid % group_threads;
            for (int batch = block_begin; batch < block_end;
                 batch += groups_per_batch) {
                const int block_group = batch + group_slot;
                const bool active = block_group < block_end;
                int group = 0;
                int term_begin = 0;
                int term_end = 0;
                if (active) {
                    group = pot.forward_block_groups[block_group];
                    term_begin = pot.forward_term_offsets[group];
                    term_end = pot.forward_term_offsets[group + 1];
                }
                double sum = 0.0;
                for (int pos = term_begin + group_lane; pos < term_end;
                     pos += group_threads) {
                    const ForwardTerm term = pot.forward_terms[pos];
                    sum += static_cast<double>(term.c) *
                        mom[term.l1] * mom[term.l2];
                }
                sum = warp_sum(sum);
                if (lane == 0) block_sum[warp] = sum;
                __syncthreads();
                if (active && group_lane == 0) {
                    double group_sum = 0.0;
                    const int first_warp = group_slot * group_warps;
                    for (int index = 0; index < group_warps; ++index) {
                        group_sum += block_sum[first_warp + index];
                    }
                    mom[pot.forward_ids[group]] += group_sum;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    if (pot.model_type == static_cast<int>(MTPGpuModel::Linear)) {
        const bool need_energy =
            site_energy != nullptr || total_energy != nullptr;
        double local_energy = 0.0;
        for (int s = tid; s < pot.alpha_scalar_moments; s += blockDim.x) {
            if (need_energy) {
                local_energy += pot.moment_coeffs[s] *
                    mom[pot.alpha_moment_mapping[s]];
            }
        }
        if (need_energy) {
            block_sum[tid] = local_energy;
            __syncthreads();
            for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
                if (tid < stride) block_sum[tid] += block_sum[tid + stride];
                __syncthreads();
            }
            if (tid == 0) {
                const int ati = centers[center_idx];
                const int itype = pot.spmap[type[ati]];
                const double energy = pot.species_coeffs[itype] + block_sum[0];
                const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
                if (site_energy != nullptr) site_energy[ati] = weight * energy;
                if (total_energy != nullptr) atomic_add_double(total_energy, weight * energy);
            }
        }
    } else if (neural_cache != nullptr) {
        if constexpr (std::is_same_v<WorkReal, double>) {
            const int ati = centers[center_idx];
            const int itype = pot.spmap[type[ati]];
            double *cache_base = neural_cache +
                center_idx * mtpm_neural_cache_stride(pot);
            mtpm_build_neural_cache<true>(pot, itype, mom, nullptr, cache_base);
            if (tid == 0) {
                const double energy = pot.species_coeffs[itype] +
                    mtpm_neural_energy_from_cache(
                        pot, mtpm_neural_cache_pointers(pot, cache_base));
                const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
                if (site_energy != nullptr) site_energy[ati] = weight * energy;
                if (total_energy != nullptr) atomic_add_double(total_energy, weight * energy);
            }
        }
    } else if (tid == 0) {
        const int ati = centers[center_idx];
        const int itype = pot.spmap[type[ati]];
        if constexpr (std::is_same_v<WorkReal, double>) {
            const double model_energy =
                mtpm_external_model_energy(pot, mom, itype);
            const double energy = pot.species_coeffs[itype] + model_energy;
            const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
                if (site_energy != nullptr) site_energy[ati] = weight * energy;
            if (total_energy != nullptr) atomic_add_double(total_energy, weight * energy);
        }
    }
}

// One thread owns a center and visits the moment schedule in index order.
// This deliberately trades throughput for repeatable training diagnostics.
template <typename MomentReal, typename AdjointReal>
__global__ void propagate_ordered_kernel(
    DevicePotential pot, const int *centers, const int *type,
    MomentReal *moment_vals, AdjointReal *dmom, double *site_energy,
    double *neural_cache, int phase = 0)
{
    const int center_idx = blockIdx.x;
    MomentReal *mom = moment_vals + center_idx * pot.alpha_moments_count;
    AdjointReal *adj = dmom + center_idx * pot.alpha_moments_count;

    if (phase != 2) {
    for (int level = 1; level <= pot.max_level; ++level) {
        const int begin = pot.level_offsets[level];
        const int end = pot.level_offsets[level + 1];
        for (int t = begin; t < end; ++t) {
            const int l1 = pot.times_l1[t];
            const int l2 = pot.times_l2[t];
            const int id = pot.times_id[t];
            const double c = static_cast<double>(pot.times_c[t]);
            mom[id] += c * mom[l1] * mom[l2];
        }
    }

    const int ati = centers[center_idx];
    const int itype = pot.spmap[type[ati]];
    double energy = 0.0;
    if (pot.model_type == static_cast<int>(MTPGpuModel::Linear)) {
        for (int s = 0; s < pot.alpha_scalar_moments; ++s) {
            const int mid = pot.alpha_moment_mapping[s];
            const double coefficient = pot.moment_coeffs[s];
            energy += coefficient * mom[mid];
            adj[mid] += static_cast<AdjointReal>(coefficient);
        }
    } else {
        if constexpr (std::is_same_v<MomentReal, double> &&
                      std::is_same_v<AdjointReal, double>) {
            if (neural_cache != nullptr) {
                const int stride = mtpm_neural_cache_stride(pot);
                mtpm_seed_energy_and_adjoint_cached(
                    pot, mom, itype, adj, &energy,
                    neural_cache + center_idx * stride);
            } else {
                mtpm_seed_energy_and_adjoint(pot, mom, itype, adj, &energy);
            }
        }
    }
    if (site_energy != nullptr) {
        const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
        site_energy[ati] = weight * (pot.species_coeffs[itype] + energy);
    }

    if (pot.center_energy_weights != nullptr) {
        const double weight = pot.center_energy_weights[centers[center_idx]];
        for (int m = 0; m < pot.alpha_moments_count; ++m) adj[m] *= weight;
    }
    }
    if (phase == 1) return;
    for (int level = pot.max_level; level >= 1; --level) {
        const int begin = pot.level_offsets[level];
        const int end = pot.level_offsets[level + 1];
        for (int t = begin; t < end; ++t) {
            const int l1 = pot.times_l1[t];
            const int l2 = pot.times_l2[t];
            const int id = pot.times_id[t];
            const double dadj =
                adj[id] * static_cast<double>(pot.times_c[t]);
            adj[l1] += dadj * mom[l2];
            adj[l2] += dadj * mom[l1];
        }
    }
}

template <typename WorkReal>
__global__ void propagate_energy_ordered_kernel(
    DevicePotential pot, const int *centers, const int *type,
    WorkReal *moment_vals, double *site_energy)
{
    const int center_idx = blockIdx.x;
    WorkReal *mom = moment_vals + center_idx * pot.alpha_moments_count;
    for (int level = 1; level <= pot.max_level; ++level) {
        const int begin = pot.level_offsets[level];
        const int end = pot.level_offsets[level + 1];
        for (int t = begin; t < end; ++t) {
            const int l1 = pot.times_l1[t];
            const int l2 = pot.times_l2[t];
            const int id = pot.times_id[t];
            mom[id] += static_cast<double>(pot.times_c[t]) *
                mom[l1] * mom[l2];
        }
    }
    const int ati = centers[center_idx];
    const int itype = pot.spmap[type[ati]];
    double energy = 0.0;
    if (pot.model_type == static_cast<int>(MTPGpuModel::Linear)) {
        for (int s = 0; s < pot.alpha_scalar_moments; ++s) {
            energy += pot.moment_coeffs[s] *
                mom[pot.alpha_moment_mapping[s]];
        }
    } else {
        if constexpr (std::is_same_v<WorkReal, double>) {
            energy = mtpm_external_model_energy(pot, mom, itype);
        }
    }
    if (site_energy != nullptr) {
        const double weight = pot.center_energy_weights ? pot.center_energy_weights[ati] : 1.0;
        site_energy[ati] = weight * (pot.species_coeffs[itype] + energy);
    }
}

__global__ void total_energy_ordered_kernel(
    int inum, const int *centers, const double *site_energy,
    double *total_energy)
{
    double sum = 0.0;
    for (int center_idx = 0; center_idx < inum; ++center_idx) {
        sum += site_energy[centers[center_idx]];
    }
    *total_energy = sum;
}

// The optimized launch is restricted to 8 neighbor x 16 basic lanes. Each
// logical warp contains four basic lanes for each neighbor, so only its first
// eight lanes need to publish partials. All 128 threads execute both shuffles
// and the barrier, including invalid neighbor lanes at the end of a list.
template <bool WarpReduction>
__device__ inline void store_force_partials(double *partial,
    double fx, double fy, double fz)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    if constexpr (WarpReduction) {
        fx += shuffle_down_32(fx, 16);
        fy += shuffle_down_32(fy, 16);
        fz += shuffle_down_32(fz, 16);
        fx += shuffle_down_32(fx, 8);
        fy += shuffle_down_32(fy, 8);
        fz += shuffle_down_32(fz, 8);
        if ((ty & 3) == 0) {
            const int index = ((ty / 4) * 8 + tx) * 3;
            partial[index] = fx;
            partial[index + 1] = fy;
            partial[index + 2] = fz;
        }
    } else {
        const int index = (ty * blockDim.x + tx) * 3;
        partial[index] = fx;
        partial[index + 1] = fy;
        partial[index + 2] = fz;
    }
    __syncthreads();
}

template <typename WorkReal, bool ComputeSiteVirial, bool WarpReduction = false>
__global__ void direct_force_kernel(DevicePotential pot, const int *centers,
                                    const int *offsets, const int *neighbors,
                                    const double *neighbor_vectors,
                                    const int *type, const double *x,
                                    const WorkReal *dmom, double *force,
                                    double *site_virial, int basic_tile,
                                    int center_offset, int tile_offset)
{
    extern __shared__ double shared[];
    const int center_idx = blockIdx.x + center_offset;
    const int tile_idx = blockIdx.y + tile_offset;
    const int tile_begin = tile_idx * basic_tile;
    const int remaining = pot.alpha_index_basic_count - tile_begin;
    const int tile_count = remaining < basic_tile ? remaining : basic_tile;
    const int nlane = blockDim.x;
    const int blane = blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    double *rb_vals = shared;
    double *rb_ders = rb_vals + nlane * pot.rb_size;
    double *val_mu = rb_ders + nlane * pot.rb_size;
    double *der_mu = val_mu + nlane * pot.max_mu;
    double *dist_powers = der_mu + nlane * pot.max_mu;
    double *coord_powers = dist_powers + nlane * pot.max_alpha_index_basic;
    double *rvec_cache = coord_powers + nlane * pot.coord_power_count * 3;
    double *rijinv_cache = rvec_cache + nlane * 3;
    double *valid_cache = rijinv_cache + nlane;
    double *force_partial = valid_cache + nlane;

    const int ati = centers[center_idx];
    const int start = offsets[center_idx];
    const int end = offsets[center_idx + 1];
    const int jnum = end - start;
    const WorkReal *adj = dmom + center_idx * pot.alpha_moments_count;

    double center_fx = 0.0;
    double center_fy = 0.0;
    double center_fz = 0.0;
    double center_vxx = 0.0;
    double center_vyy = 0.0;
    double center_vzz = 0.0;
    double center_vxy = 0.0;
    double center_vxz = 0.0;
    double center_vyz = 0.0;

    for (int neighbor_begin = 0; neighbor_begin < jnum; neighbor_begin += nlane) {
        const int jj = neighbor_begin + tx;
        if (ty == 0) {
            if (jj < jnum) {
                fill_pair_cache(pot, tx, center_idx, ati, jj,
                                offsets, neighbors,
                                neighbor_vectors, type, x,
                                rb_vals, rb_ders, val_mu, der_mu, dist_powers,
                                coord_powers, rvec_cache, rijinv_cache, valid_cache, true, false);
            } else {
                valid_cache[tx] = 0.0;
            }
        }
        __syncthreads();
        contract_pair_radial<true>(
            pot, ati, jj, center_idx, offsets, neighbors, type,
            valid_cache, rb_vals, rb_ders, val_mu, der_mu);
        __syncthreads();

        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        if (valid_cache[tx] != 0.0) {
            const double *vm = val_mu + tx * pot.max_mu;
            const double *dm = der_mu + tx * pot.max_mu;
            const double *dp = dist_powers + tx * pot.max_alpha_index_basic;
            const double *cp = coord_powers + tx * pot.coord_power_count * 3;
            const double *rv = rvec_cache + tx * 3;
            const double rijinv = rijinv_cache[tx];

            for (int local_b = ty; local_b < tile_count; local_b += blane) {
                const int bi = tile_begin + local_b;
                const int mu = pot.basic_mu[bi];
                const int b1 = pot.basic_b1[bi];
                const int b2 = pot.basic_b2[bi];
                const int b3 = pot.basic_b3[bi];
                const int nu = pot.basic_nu[bi];
                double val = vm[mu];
                double der = dm[mu];
                const double rpow = dp[nu];
                val *= rpow;
                der = der * rpow - nu * val * rijinv;

                const double pow0 = cp[3 * (b1 + 1) + 0];
                const double pow1 = cp[3 * (b2 + 1) + 1];
                const double pow2 = cp[3 * (b3 + 1) + 2];
                const double common = pow0 * pow1 * pow2 * der * rijinv;

                double dmx = common * rv[0];
                double dmy = common * rv[1];
                double dmz = common * rv[2];
                dmx += b1 * val * cp[3 * b1 + 0] * pow1 * pow2;
                dmy += b2 * val * cp[3 * b2 + 1] * pow0 * pow2;
                dmz += b3 * val * cp[3 * b3 + 2] * pow0 * pow1;

                const double coeff = adj[bi];
                fx += coeff * dmx;
                fy += coeff * dmy;
                fz += coeff * dmz;
            }
        }

        store_force_partials<WarpReduction>(force_partial, fx, fy, fz);

        if (ty == 0 && valid_cache[tx] != 0.0) {
            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
            const int partial_lanes = WarpReduction ? 4 : blane;
            for (int y = 0; y < partial_lanes; ++y) {
                const int idx = (y * nlane + tx) * 3;
                sx += force_partial[idx + 0];
                sy += force_partial[idx + 1];
                sz += force_partial[idx + 2];
            }
            const int atj = neighbors[start + jj];
            center_fx += sx;
            center_fy += sy;
            center_fz += sz;
            if (force) atomic_add_double(&force[3 * atj + 0], -sx);
            if (force) atomic_add_double(&force[3 * atj + 1], -sy);
            if (force) atomic_add_double(&force[3 * atj + 2], -sz);

            if constexpr (ComputeSiteVirial) {
                const double rx = rvec_cache[3 * tx + 0];
                const double ry = rvec_cache[3 * tx + 1];
                const double rz = rvec_cache[3 * tx + 2];
                const double vxx = -0.5 * sx * rx;
                const double vyy = -0.5 * sy * ry;
                const double vzz = -0.5 * sz * rz;
                const double vxy = -0.25 * (sy * rx + sx * ry);
                const double vxz = -0.25 * (sz * rx + sx * rz);
                const double vyz = -0.25 * (sz * ry + sy * rz);
                center_vxx += vxx;
                center_vyy += vyy;
                center_vzz += vzz;
                center_vxy += vxy;
                center_vxz += vxz;
                center_vyz += vyz;
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 0], vxx);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 1], vyy);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 2], vzz);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 3], vxy);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 4], vxz);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 5], vyz);
            }
        }
        __syncthreads();
    }

    // Reduce the center contribution once per center/basic tile. Neighbor
    // atom atomics remain immediate because multiple center blocks can write
    // the same neighbor and no reverse-edge cache is allocated.
    if (ty == 0) {
        const int fp = tx * 3;
        force_partial[fp + 0] = center_fx;
        force_partial[fp + 1] = center_fy;
        force_partial[fp + 2] = center_fz;
    }
    __syncthreads();
    if (ty == 0 && tx == 0) {
        double sx = 0.0;
        double sy = 0.0;
        double sz = 0.0;
        for (int xlane = 0; xlane < nlane; ++xlane) {
            sx += force_partial[3 * xlane + 0];
            sy += force_partial[3 * xlane + 1];
            sz += force_partial[3 * xlane + 2];
        }
        if (force) atomic_add_double(&force[3 * ati + 0], sx);
        if (force) atomic_add_double(&force[3 * ati + 1], sy);
        if (force) atomic_add_double(&force[3 * ati + 2], sz);
    }
    __syncthreads();

    if constexpr (ComputeSiteVirial) {
        if (ty == 0) {
            force_partial[3 * tx + 0] = center_vxx;
            force_partial[3 * tx + 1] = center_vyy;
            force_partial[3 * tx + 2] = center_vzz;
        }
        __syncthreads();
        if (ty == 0 && tx == 0) {
            for (int component = 0; component < 3; ++component) {
                double sum = 0.0;
                for (int xlane = 0; xlane < nlane; ++xlane) {
                    sum += force_partial[3 * xlane + component];
                }
                if (site_virial) atomic_add_double(&site_virial[6 * ati + component], sum);
                if (pot.global_virial) atomic_add_double(&pot.global_virial[component], 2.0 * sum);
            }
        }
        __syncthreads();
        if (ty == 0) {
            force_partial[3 * tx + 0] = center_vxy;
            force_partial[3 * tx + 1] = center_vxz;
            force_partial[3 * tx + 2] = center_vyz;
        }
        __syncthreads();
        if (ty == 0 && tx == 0) {
            for (int component = 0; component < 3; ++component) {
                double sum = 0.0;
                for (int xlane = 0; xlane < nlane; ++xlane) {
                    sum += force_partial[3 * xlane + component];
                }
                if (site_virial) atomic_add_double(&site_virial[6 * ati + 3 + component], sum);
                if (pot.global_virial) atomic_add_double(&pot.global_virial[3 + component], 2.0 * sum);
            }
        }
        __syncthreads();
    }
}

template <typename WorkReal, bool ComputeSiteVirial, bool WarpReduction = false>
__global__ void center_neighbors_force_kernel(
    DevicePotential pot, const int *centers, const int *offsets,
    const int *neighbors, const double *neighbor_vectors, const int *type,
    const double *x, const WorkReal *dmom, double *force,
    double *site_virial)
{
    extern __shared__ double shared[];
    const int center_idx = blockIdx.x;
    const int nlane = blockDim.x;
    const int blane = blockDim.y;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    double *rb_vals = shared;
    double *rb_ders = rb_vals + nlane * pot.rb_size;
    double *val_mu = rb_ders + nlane * pot.rb_size;
    double *der_mu = val_mu + nlane * pot.max_mu;
    double *dist_powers = der_mu + nlane * pot.max_mu;
    double *coord_powers = dist_powers + nlane * pot.max_alpha_index_basic;
    double *rvec_cache = coord_powers + nlane * pot.coord_power_count * 3;
    double *rijinv_cache = rvec_cache + nlane * 3;
    double *valid_cache = rijinv_cache + nlane;
    double *force_partial = valid_cache + nlane;

    const int ati = centers[center_idx];
    const int start = offsets[center_idx];
    const int jnum = offsets[center_idx + 1] - start;
    const WorkReal *adj = dmom + center_idx * pot.alpha_moments_count;
    double center_fx = 0.0;
    double center_fy = 0.0;
    double center_fz = 0.0;
    double center_vxx = 0.0;
    double center_vyy = 0.0;
    double center_vzz = 0.0;
    double center_vxy = 0.0;
    double center_vxz = 0.0;
    double center_vyz = 0.0;

    for (int neighbor_begin = 0; neighbor_begin < jnum;
         neighbor_begin += nlane) {
        const int jj = neighbor_begin + tx;
        if (ty == 0) {
            if (jj < jnum) {
                fill_pair_cache(pot, tx, center_idx, ati, jj, offsets,
                                neighbors, neighbor_vectors, type, x,
                                rb_vals, rb_ders, val_mu, der_mu, dist_powers,
                                coord_powers, rvec_cache, rijinv_cache,
                                valid_cache, true, false);
            } else {
                valid_cache[tx] = 0.0;
            }
        }
        __syncthreads();
        contract_pair_radial<true>(
            pot, ati, jj, center_idx, offsets, neighbors, type,
            valid_cache, rb_vals, rb_ders, val_mu, der_mu);
        __syncthreads();

        double fx = 0.0;
        double fy = 0.0;
        double fz = 0.0;
        if (valid_cache[tx] != 0.0) {
            const double *vm = val_mu + tx * pot.max_mu;
            const double *dm = der_mu + tx * pot.max_mu;
            const double *dp = dist_powers + tx * pot.max_alpha_index_basic;
            const double *cp = coord_powers + tx * pot.coord_power_count * 3;
            const double *rv = rvec_cache + tx * 3;
            const double rijinv = rijinv_cache[tx];
            for (int bi = ty; bi < pot.alpha_index_basic_count; bi += blane) {
                const int mu = pot.basic_mu[bi];
                const int b1 = pot.basic_b1[bi];
                const int b2 = pot.basic_b2[bi];
                const int b3 = pot.basic_b3[bi];
                const int nu = pot.basic_nu[bi];
                double val = vm[mu];
                double der = dm[mu];
                const double rpow = dp[nu];
                val *= rpow;
                der = der * rpow - nu * val * rijinv;
                const double pow0 = cp[3 * (b1 + 1) + 0];
                const double pow1 = cp[3 * (b2 + 1) + 1];
                const double pow2 = cp[3 * (b3 + 1) + 2];
                const double common = pow0 * pow1 * pow2 * der * rijinv;
                double dmx = common * rv[0];
                double dmy = common * rv[1];
                double dmz = common * rv[2];
                dmx += b1 * val * cp[3 * b1 + 0] * pow1 * pow2;
                dmy += b2 * val * cp[3 * b2 + 1] * pow0 * pow2;
                dmz += b3 * val * cp[3 * b3 + 2] * pow0 * pow1;
                const double coeff = adj[bi];
                fx += coeff * dmx;
                fy += coeff * dmy;
                fz += coeff * dmz;
            }
        }

        store_force_partials<WarpReduction>(force_partial, fx, fy, fz);
        if (ty == 0 && valid_cache[tx] != 0.0) {
            double sx = 0.0;
            double sy = 0.0;
            double sz = 0.0;
            const int partial_lanes = WarpReduction ? 4 : blane;
            for (int y = 0; y < partial_lanes; ++y) {
                const int idx = (y * nlane + tx) * 3;
                sx += force_partial[idx + 0];
                sy += force_partial[idx + 1];
                sz += force_partial[idx + 2];
            }
            const int atj = neighbors[start + jj];
            center_fx += sx;
            center_fy += sy;
            center_fz += sz;
            if (force) atomic_add_double(&force[3 * atj + 0], -sx);
            if (force) atomic_add_double(&force[3 * atj + 1], -sy);
            if (force) atomic_add_double(&force[3 * atj + 2], -sz);
            if constexpr (ComputeSiteVirial) {
                const double rx = rvec_cache[3 * tx + 0];
                const double ry = rvec_cache[3 * tx + 1];
                const double rz = rvec_cache[3 * tx + 2];
                const double vxx = -0.5 * sx * rx;
                const double vyy = -0.5 * sy * ry;
                const double vzz = -0.5 * sz * rz;
                const double vxy = -0.25 * (sy * rx + sx * ry);
                const double vxz = -0.25 * (sz * rx + sx * rz);
                const double vyz = -0.25 * (sz * ry + sy * rz);
                center_vxx += vxx;
                center_vyy += vyy;
                center_vzz += vzz;
                center_vxy += vxy;
                center_vxz += vxz;
                center_vyz += vyz;
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 0], vxx);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 1], vyy);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 2], vzz);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 3], vxy);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 4], vxz);
                if (site_virial) atomic_add_double(&site_virial[6 * atj + 5], vyz);
            }
        }
        __syncthreads();
    }

    if (ty == 0) {
        force_partial[3 * tx + 0] = center_fx;
        force_partial[3 * tx + 1] = center_fy;
        force_partial[3 * tx + 2] = center_fz;
    }
    __syncthreads();
    if (ty == 0 && tx == 0) {
        for (int component = 0; component < 3; ++component) {
            double sum = 0.0;
            for (int xlane = 0; xlane < nlane; ++xlane) {
                sum += force_partial[3 * xlane + component];
            }
            if (force) atomic_add_double(&force[3 * ati + component], sum);
        }
    }
    __syncthreads();
    if constexpr (ComputeSiteVirial) {
        if (ty == 0) {
            force_partial[3 * tx + 0] = center_vxx;
            force_partial[3 * tx + 1] = center_vyy;
            force_partial[3 * tx + 2] = center_vzz;
        }
        __syncthreads();
        if (ty == 0 && tx == 0) {
            for (int component = 0; component < 3; ++component) {
                double sum = 0.0;
                for (int xlane = 0; xlane < nlane; ++xlane) {
                    sum += force_partial[3 * xlane + component];
                }
                if (site_virial) atomic_add_double(&site_virial[6 * ati + component], sum);
                if (pot.global_virial) atomic_add_double(&pot.global_virial[component], 2.0 * sum);
            }
        }
        __syncthreads();
        if (ty == 0) {
            force_partial[3 * tx + 0] = center_vxy;
            force_partial[3 * tx + 1] = center_vxz;
            force_partial[3 * tx + 2] = center_vyz;
        }
        __syncthreads();
        if (ty == 0 && tx == 0) {
            for (int component = 0; component < 3; ++component) {
                double sum = 0.0;
                for (int xlane = 0; xlane < nlane; ++xlane) {
                    sum += force_partial[3 * xlane + component];
                }
                if (site_virial) atomic_add_double(&site_virial[6 * ati + 3 + component], sum);
                if (pot.global_virial) atomic_add_double(&pot.global_virial[3 + component], 2.0 * sum);
            }
        }
        __syncthreads();
    }
}

__device__ void loss_direction(const int ati, const int atj, const int structure,
                               const double *rvec, const double *dloss_dforce,
                               const double *dloss_dstress,
                               double *wx, double *wy, double *wz)
{
    *wx = dloss_dforce[3 * ati + 0] - dloss_dforce[3 * atj + 0];
    *wy = dloss_dforce[3 * ati + 1] - dloss_dforce[3 * atj + 1];
    *wz = dloss_dforce[3 * ati + 2] - dloss_dforce[3 * atj + 2];
    const double *ds = dloss_dstress + 9 * structure;
    // The forward path exposes a symmetric virial tensor.  Contract its
    // derivative with the symmetric part of an arbitrary row-major adjoint.
    const double sxy = 0.5 * (ds[1] + ds[3]);
    const double sxz = 0.5 * (ds[2] + ds[6]);
    const double syz = 0.5 * (ds[5] + ds[7]);
    *wx -= ds[0] * rvec[0] + sxy * rvec[1] + sxz * rvec[2];
    *wy -= sxy * rvec[0] + ds[4] * rvec[1] + syz * rvec[2];
    *wz -= sxz * rvec[0] + syz * rvec[1] + ds[8] * rvec[2];
}

// q = Dm, where D is the force/stress loss direction for each structure.
__global__ void basic_directional_kernel(
    DevicePotential pot, const int *centers, const int *offsets,
    const int *neighbors, const double *neighbor_vectors, const int *type,
    const double *x, const int *center_structure, const double *dloss_dforce,
    const double *dloss_dstress, double *directional, int basic_tile)
{
    extern __shared__ double shared[];
    const int center_idx = blockIdx.x;
    const int tile_begin = blockIdx.y * basic_tile;
    const int remaining = pot.alpha_index_basic_count - tile_begin;
    const int tile_count = remaining < basic_tile ? remaining : basic_tile;
    const int nlane = blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;
    const int partial_stride = basic_tile + 1;

    double *partial = shared;
    double *rb_vals = partial + nlane * partial_stride;
    double *rb_ders = rb_vals + nlane * pot.rb_size;
    double *val_mu = rb_ders + nlane * pot.rb_size;
    double *der_mu = val_mu + nlane * pot.max_mu;
    double *dist_powers = der_mu + nlane * pot.max_mu;
    double *coord_powers = dist_powers + nlane * pot.max_alpha_index_basic;
    double *rvec_cache = coord_powers + nlane * pot.coord_power_count * 3;
    double *rijinv_cache = rvec_cache + nlane * 3;
    double *valid_cache = rijinv_cache + nlane;

    for (int idx = tid; idx < nlane * partial_stride; idx += nthreads) {
        partial[idx] = 0.0;
    }
    __syncthreads();

    const int ati = centers[center_idx];
    const int structure = center_structure[center_idx];
    const int start = offsets[center_idx];
    const int jnum = offsets[center_idx + 1] - start;
    for (int neighbor_begin = 0; neighbor_begin < jnum; neighbor_begin += nlane) {
        const int jj = neighbor_begin + tx;
        if (ty == 0) {
            if (jj < jnum) {
                fill_pair_cache(pot, tx, center_idx, ati, jj,
                                offsets, neighbors,
                                neighbor_vectors, type, x, rb_vals, rb_ders,
                                val_mu, der_mu, dist_powers, coord_powers,
                                rvec_cache, rijinv_cache, valid_cache, true, false);
            } else {
                valid_cache[tx] = 0.0;
            }
        }
        __syncthreads();
        contract_pair_radial<true>(
            pot, ati, jj, center_idx, offsets, neighbors, type,
            valid_cache, rb_vals, rb_ders, val_mu, der_mu);
        __syncthreads();

        if (valid_cache[tx] != 0.0) {
            const int atj = neighbors[start + jj];
            const double *vm = val_mu + tx * pot.max_mu;
            const double *dm = der_mu + tx * pot.max_mu;
            const double *dp = dist_powers + tx * pot.max_alpha_index_basic;
            const double *cp = coord_powers + tx * pot.coord_power_count * 3;
            const double *rv = rvec_cache + tx * 3;
            const double rijinv = rijinv_cache[tx];
            double wx, wy, wz;
            loss_direction(ati, atj, structure, rv, dloss_dforce,
                           dloss_dstress, &wx, &wy, &wz);

            for (int local_b = ty; local_b < tile_count; local_b += blockDim.y) {
                const int bi = tile_begin + local_b;
                const int mu = pot.basic_mu[bi];
                const int b1 = pot.basic_b1[bi];
                const int b2 = pot.basic_b2[bi];
                const int b3 = pot.basic_b3[bi];
                const int nu = pot.basic_nu[bi];
                double val = vm[mu];
                double der = dm[mu];
                const double rpow = dp[nu];
                val *= rpow;
                der = der * rpow - nu * val * rijinv;
                const double pow0 = cp[3 * (b1 + 1) + 0];
                const double pow1 = cp[3 * (b2 + 1) + 1];
                const double pow2 = cp[3 * (b3 + 1) + 2];
                const double common = pow0 * pow1 * pow2 * der * rijinv;
                double dmx = common * rv[0];
                double dmy = common * rv[1];
                double dmz = common * rv[2];
                dmx += b1 * val * cp[3 * b1 + 0] * pow1 * pow2;
                dmy += b2 * val * cp[3 * b2 + 1] * pow0 * pow2;
                dmz += b3 * val * cp[3 * b3 + 2] * pow0 * pow1;
                partial[tx * partial_stride + local_b] +=
                    wx * dmx + wy * dmy + wz * dmz;
            }
        }
        __syncthreads();
    }

    for (int local_b = tid; local_b < tile_count; local_b += nthreads) {
        double sum = 0.0;
        for (int lane = 0; lane < nlane; ++lane) {
            sum += partial[lane * partial_stride + local_b];
        }
        directional[center_idx * pot.alpha_moments_count + tile_begin + local_b] = sum;
    }
}

// Propagate q forward and h = D(dU/dm) backward through the moment DAG.
__global__ void propagate_directional_kernel(DevicePotential pot,
                                             const int *centers,
                                             const int *type,
                                             const double *moment_vals,
                                             const double *dmom,
                                             double *directional,
                                             double *directional_adjoint,
                                             double *neural_cache, int phase = 0)
{
    extern __shared__ double block_sum[];
    const int center_idx = blockIdx.x;
    const int tid = threadIdx.x;
    const double *mom = moment_vals + center_idx * pot.alpha_moments_count;
    const double *adj = dmom + center_idx * pot.alpha_moments_count;
    double *dir = directional + center_idx * pot.alpha_moments_count;
    double *hadj = directional_adjoint + center_idx * pot.alpha_moments_count;

    if (phase != 2) {
    for (int level = 1; level <= pot.max_level; ++level) {
        const int begin = pot.forward_level_offsets[level];
        const int end = pot.forward_level_offsets[level + 1];
        for (int group = begin + tid; group < end;
             group += blockDim.x) {
            const int term_begin = pot.forward_term_offsets[group];
            const int term_end = pot.forward_term_offsets[group + 1];
            if (blockDim.x >= propagation_warp_size &&
                term_end - term_begin > propagation_warp_group_threshold) {
                continue;
            }
            double sum = 0.0;
            for (int pos = term_begin; pos < term_end; ++pos) {
                const ForwardTerm term = pot.forward_terms[pos];
                const int l1 = term.l1;
                const int l2 = term.l2;
                sum += static_cast<double>(term.c) *
                    (dir[l1] * mom[l2] + mom[l1] * dir[l2]);
            }
            dir[pot.forward_ids[group]] += sum;
        }
        if (blockDim.x >= propagation_warp_size) {
            const int lane = tid & (propagation_warp_size - 1);
            const int warp = tid / propagation_warp_size;
            const int warps = blockDim.x / propagation_warp_size;
            for (int group = begin + warp; group < end; group += warps) {
                const int term_begin = pot.forward_term_offsets[group];
                const int term_end = pot.forward_term_offsets[group + 1];
                const int term_count = term_end - term_begin;
                if (term_count <= propagation_warp_group_threshold ||
                    term_count > propagation_block_group_threshold) {
                    continue;
                }
                double sum = 0.0;
                for (int pos = term_begin + lane; pos < term_end;
                     pos += propagation_warp_size) {
                    const ForwardTerm term = pot.forward_terms[pos];
                    const int l1 = term.l1;
                    const int l2 = term.l2;
                    sum += static_cast<double>(term.c) *
                        (dir[l1] * mom[l2] + mom[l1] * dir[l2]);
                }
                sum = warp_sum(sum);
                if (lane == 0) dir[pot.forward_ids[group]] += sum;
            }
            const int block_begin =
                pot.forward_block_level_offsets[level];
            const int block_end =
                pot.forward_block_level_offsets[level + 1];
            const int group_warps = warps < propagation_multiwarp_group_warps
                ? warps : propagation_multiwarp_group_warps;
            const int group_threads = group_warps * propagation_warp_size;
            const int groups_per_batch = warps / group_warps;
            const int group_slot = tid / group_threads;
            const int group_lane = tid % group_threads;
            for (int batch = block_begin; batch < block_end;
                 batch += groups_per_batch) {
                const int block_group = batch + group_slot;
                const bool active = block_group < block_end;
                int group = 0;
                int term_begin = 0;
                int term_end = 0;
                if (active) {
                    group = pot.forward_block_groups[block_group];
                    term_begin = pot.forward_term_offsets[group];
                    term_end = pot.forward_term_offsets[group + 1];
                }
                double sum = 0.0;
                for (int pos = term_begin + group_lane; pos < term_end;
                     pos += group_threads) {
                    const ForwardTerm term = pot.forward_terms[pos];
                    const int l1 = term.l1;
                    const int l2 = term.l2;
                    sum += static_cast<double>(term.c) *
                        (dir[l1] * mom[l2] + mom[l1] * dir[l2]);
                }
                sum = warp_sum(sum);
                if (lane == 0) block_sum[warp] = sum;
                __syncthreads();
                if (active && group_lane == 0) {
                    double group_sum = 0.0;
                    const int first_warp = group_slot * group_warps;
                    for (int index = 0; index < group_warps; ++index) {
                        group_sum += block_sum[first_warp + index];
                    }
                    dir[pot.forward_ids[group]] += group_sum;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }

    }
    if (phase == 1) return;
    if (neural_cache != nullptr) {
        double *cache_base = neural_cache +
            center_idx * mtpm_neural_cache_stride(pot);
        mtpm_build_neural_cache<true>(
            pot, pot.spmap[type[centers[center_idx]]], mom, dir, cache_base);
        const auto cache = mtpm_neural_cache_pointers(pot, cache_base);
        mtpm_seed_neural_adjoint(pot, cache.hessian_direction, hadj);
    } else if (pot.model_type != static_cast<int>(MTPGpuModel::Linear) && tid == 0) {
        const int ati = centers[center_idx];
        const int itype = pot.spmap[type[ati]];
        mtpm_seed_directional_adjoint(pot, mom, dir, itype, hadj);
    }
    __syncthreads();

    for (int level = pot.max_level; level >= 1; --level) {
        const int begin = pot.reverse_level_offsets[level];
        const int end = pot.reverse_level_offsets[level + 1];
        for (int group = begin + tid; group < end;
             group += blockDim.x) {
            const int entry_begin = pot.reverse_entry_offsets[group];
            const int entry_end = pot.reverse_entry_offsets[group + 1];
            if (blockDim.x >= propagation_warp_size &&
                entry_end - entry_begin > propagation_warp_group_threshold) {
                continue;
            }
            double sum = 0.0;
            for (int pos = entry_begin; pos < entry_end; ++pos) {
                const ReverseTerm term = pot.reverse_entries[pos];
                const int source = term.source;
                sum += static_cast<double>(term.c) *
                    (hadj[source] * mom[term.other] + adj[source] * dir[term.other]);
            }
            hadj[pot.reverse_ids[group]] += sum;
        }
        if (blockDim.x >= propagation_warp_size) {
            const int lane = tid & (propagation_warp_size - 1);
            const int warp = tid / propagation_warp_size;
            const int warps = blockDim.x / propagation_warp_size;
            for (int group = begin + warp; group < end; group += warps) {
                const int entry_begin = pot.reverse_entry_offsets[group];
                const int entry_end = pot.reverse_entry_offsets[group + 1];
                const int entry_count = entry_end - entry_begin;
                if (entry_count <= propagation_warp_group_threshold ||
                    entry_count > propagation_block_group_threshold) {
                    continue;
                }
                double sum = 0.0;
                for (int pos = entry_begin + lane; pos < entry_end;
                     pos += propagation_warp_size) {
                    const ReverseTerm term = pot.reverse_entries[pos];
                    const int source = term.source;
                    sum += static_cast<double>(term.c) *
                        (hadj[source] * mom[term.other] +
                         adj[source] * dir[term.other]);
                }
                sum = warp_sum(sum);
                if (lane == 0) hadj[pot.reverse_ids[group]] += sum;
            }
            const int block_begin =
                pot.reverse_block_level_offsets[level];
            const int block_end =
                pot.reverse_block_level_offsets[level + 1];
            const int group_warps = warps < propagation_multiwarp_group_warps
                ? warps : propagation_multiwarp_group_warps;
            const int group_threads = group_warps * propagation_warp_size;
            const int groups_per_batch = warps / group_warps;
            const int group_slot = tid / group_threads;
            const int group_lane = tid % group_threads;
            for (int batch = block_begin; batch < block_end;
                 batch += groups_per_batch) {
                const int block_group = batch + group_slot;
                const bool active = block_group < block_end;
                int group = 0;
                int entry_begin = 0;
                int entry_end = 0;
                if (active) {
                    group = pot.reverse_block_groups[block_group];
                    entry_begin = pot.reverse_entry_offsets[group];
                    entry_end = pot.reverse_entry_offsets[group + 1];
                }
                double sum = 0.0;
                for (int pos = entry_begin + group_lane; pos < entry_end;
                     pos += group_threads) {
                    const ReverseTerm term = pot.reverse_entries[pos];
                    const int source = term.source;
                    sum += static_cast<double>(term.c) *
                        (hadj[source] * mom[term.other] +
                         adj[source] * dir[term.other]);
                }
                sum = warp_sum(sum);
                if (lane == 0) block_sum[warp] = sum;
                __syncthreads();
                if (active && group_lane == 0) {
                    double group_sum = 0.0;
                    const int first_warp = group_slot * group_warps;
                    for (int index = 0; index < group_warps; ++index) {
                        group_sum += block_sum[first_warp + index];
                    }
                    hadj[pot.reverse_ids[group]] += group_sum;
                }
                __syncthreads();
            }
        }
        __syncthreads();
    }
}

__global__ void propagate_directional_ordered_kernel(
    DevicePotential pot, const int *centers, const int *type,
    const double *moment_vals, const double *dmom,
    double *directional, double *directional_adjoint,
    double *neural_cache, int phase = 0)
{
    const int center_idx = blockIdx.x;
    const double *mom =
        moment_vals + center_idx * pot.alpha_moments_count;
    const double *adj = dmom + center_idx * pot.alpha_moments_count;
    double *dir = directional + center_idx * pot.alpha_moments_count;
    double *hadj =
        directional_adjoint + center_idx * pot.alpha_moments_count;

    if (phase != 2) {
    for (int level = 1; level <= pot.max_level; ++level) {
        const int begin = pot.level_offsets[level];
        const int end = pot.level_offsets[level + 1];
        for (int t = begin; t < end; ++t) {
            const int l1 = pot.times_l1[t];
            const int l2 = pot.times_l2[t];
            const int id = pot.times_id[t];
            const double c = static_cast<double>(pot.times_c[t]);
            dir[id] += c *
                (dir[l1] * mom[l2] + mom[l1] * dir[l2]);
        }
    }

    }
    if (phase == 1) return;
    if (pot.model_type != static_cast<int>(MTPGpuModel::Linear)) {
        const int ati = centers[center_idx];
        const int itype = pot.spmap[type[ati]];
        if (neural_cache != nullptr) {
            const int stride = mtpm_neural_cache_stride(pot);
            mtpm_seed_directional_adjoint_cached(
                pot, mom, dir, itype, hadj,
                neural_cache + center_idx * stride);
        } else {
            mtpm_seed_directional_adjoint(pot, mom, dir, itype, hadj);
        }
    }

    for (int level = pot.max_level; level >= 1; --level) {
        const int begin = pot.level_offsets[level];
        const int end = pot.level_offsets[level + 1];
        for (int t = begin; t < end; ++t) {
            const int l1 = pot.times_l1[t];
            const int l2 = pot.times_l2[t];
            const int id = pot.times_id[t];
            const double c = static_cast<double>(pot.times_c[t]);
            hadj[l1] +=
                c * (hadj[id] * mom[l2] + adj[id] * dir[l2]);
            hadj[l2] +=
                c * (hadj[id] * mom[l1] + adj[id] * dir[l1]);
        }
    }
}

__device__ double mtpm_nn1_model_parameter_derivative(
    const DevicePotential &pot, int parameter, int itype, double lambda,
    const double *mom, const double *dir)
{
    const int input_count = pot.descriptor_count;
    const int input_weight_count = input_count * pot.model_hidden1;
    const int output_offset = input_weight_count;
    const int bias_offset = output_offset + pot.model_hidden1;
    int hidden = 0;
    int input = 0;
    if (parameter < input_weight_count) {
        hidden = parameter / input_count;
        input = parameter - hidden * input_count;
    } else if (parameter < bias_offset) {
        hidden = parameter - output_offset;
    } else {
        hidden = parameter - bias_offset;
    }
    double value, first, second, hidden_direction;
    mtpm_nn1_hidden(pot, mom, dir, itype, hidden,
                    &value, &first, &second, &hidden_direction);
    const double output_weight = pot.model_coeffs[output_offset + hidden];
    double derivative = 0.0;
    double directional_derivative = 0.0;
    if (parameter < input_weight_count) {
        const double q = mtpm_descriptor_value(pot, mom, itype, input);
        const double dq = mtpm_descriptor_direction(pot, dir, input);
        derivative = q * output_weight * first;
        directional_derivative = output_weight *
            (dq * first + q * second * hidden_direction);
    } else if (parameter < bias_offset) {
        derivative = value;
        directional_derivative = first * hidden_direction;
    } else {
        derivative = output_weight * first;
        directional_derivative =
            output_weight * second * hidden_direction;
    }
    return lambda * derivative + directional_derivative;
}

__device__ double mtpm_nn2_model_parameter_derivative(
    const DevicePotential &pot, int parameter, int itype, double lambda,
    const double *mom, const double *dir)
{
    const int input_count = pot.descriptor_count;
    const int first_weight_count = input_count * pot.model_hidden1;
    const int first_bias_offset = first_weight_count;
    const int second_weight_offset = first_bias_offset + pot.model_hidden1;
    const int second_weight_count = pot.model_hidden1 * pot.model_hidden2;
    const int second_bias_offset = second_weight_offset + second_weight_count;
    const int output_offset = second_bias_offset + pot.model_hidden2;
    double derivative = 0.0;
    double directional_derivative = 0.0;

    if (parameter < second_weight_offset) {
        const bool is_weight = parameter < first_weight_count;
        const int hidden1 = is_weight
            ? parameter / input_count : parameter - first_bias_offset;
        const int input = is_weight
            ? parameter - hidden1 * input_count : 0;
        double h1, d1, dd1, du_direction;
        mtpm_nn2_first_hidden(pot, mom, dir, itype, hidden1,
                              &h1, &d1, &dd1, &du_direction);
        double upstream = 0.0;
        double upstream_direction = 0.0;
        for (int hidden2 = 0; hidden2 < pot.model_hidden2; ++hidden2) {
            double h2, d2, dd2, dz_direction;
            mtpm_nn2_second_hidden(pot, mom, dir, itype, hidden2,
                                   &h2, &d2, &dd2, &dz_direction);
            const double output_weight =
                pot.model_coeffs[output_offset + hidden2];
            const double second_weight = pot.model_coeffs[
                second_weight_offset + hidden2 * pot.model_hidden1 + hidden1];
            upstream += output_weight * d2 * second_weight;
            upstream_direction +=
                output_weight * dd2 * dz_direction * second_weight;
        }
        const double delta = d1 * upstream;
        const double delta_direction =
            dd1 * du_direction * upstream + d1 * upstream_direction;
        if (is_weight) {
            const double q = mtpm_descriptor_value(pot, mom, itype, input);
            const double dq = mtpm_descriptor_direction(pot, dir, input);
            derivative = q * delta;
            directional_derivative = dq * delta + q * delta_direction;
        } else {
            derivative = delta;
            directional_derivative = delta_direction;
        }
    } else if (parameter < second_bias_offset) {
        const int relative = parameter - second_weight_offset;
        const int hidden2 = relative / pot.model_hidden1;
        const int hidden1 = relative - hidden2 * pot.model_hidden1;
        double h1, d1, dd1, du_direction;
        mtpm_nn2_first_hidden(pot, mom, dir, itype, hidden1,
                              &h1, &d1, &dd1, &du_direction);
        double h2, d2, dd2, dz_direction;
        mtpm_nn2_second_hidden(pot, mom, dir, itype, hidden2,
                               &h2, &d2, &dd2, &dz_direction);
        const double output_weight =
            pot.model_coeffs[output_offset + hidden2];
        derivative = output_weight * d2 * h1;
        directional_derivative = output_weight *
            (dd2 * dz_direction * h1 + d2 * d1 * du_direction);
    } else if (parameter < output_offset) {
        const int hidden2 = parameter - second_bias_offset;
        double h2, d2, dd2, dz_direction;
        mtpm_nn2_second_hidden(pot, mom, dir, itype, hidden2,
                               &h2, &d2, &dd2, &dz_direction);
        const double output_weight =
            pot.model_coeffs[output_offset + hidden2];
        derivative = output_weight * d2;
        directional_derivative = output_weight * dd2 * dz_direction;
    } else {
        const int hidden2 = parameter - output_offset;
        double h2, d2, dd2, dz_direction;
        mtpm_nn2_second_hidden(pot, mom, dir, itype, hidden2,
                               &h2, &d2, &dd2, &dz_direction);
        derivative = h2;
        directional_derivative = d2 * dz_direction;
    }
    return lambda * derivative + directional_derivative;
}

__device__ double mtpm_model_parameter_derivative(
    const DevicePotential &pot, int parameter, int itype, double lambda,
    const double *mom, const double *dir)
{
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMLinear)) return 0.0;
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)) {
        return mtpm_nn1_model_parameter_derivative(
            pot, parameter, itype, lambda, mom, dir);
    }
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2)) {
        return mtpm_nn2_model_parameter_derivative(
            pot, parameter, itype, lambda, mom, dir);
    }
    const double q0 = mtpm_descriptor_value(pot, mom, itype, 0);
    const double q1 = mtpm_descriptor_value(pot, mom, itype, 1);
    const double dq0 = mtpm_descriptor_direction(pot, dir, 0);
    const double dq1 = mtpm_descriptor_direction(pot, dir, 1);
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMQuadratic)) {
        if (parameter == 0) return lambda * q0 + dq0;
        if (parameter == 1) return lambda * q1 + dq1;
        if (parameter == 2)
            return lambda * q0 * q0 + 2.0 * q0 * dq0;
        if (parameter == 3)
            return lambda * q1 * q1 + 2.0 * q1 * dq1;
        return lambda * q0 * q1 + q1 * dq0 + q0 * dq1;
    }
    const double inverse_root = 1.0 / sqrt(1.0 + q1 * q1);
    return lambda * asinh(q1) + inverse_root * dq1;
}

__device__ double mtpm_nn1_model_parameter_derivative_cached(
    const DevicePotential &pot, int parameter, double lambda,
    const NeuralCachePointers &cache)
{
    const int input_count = pot.descriptor_count;
    const int input_weight_count = input_count * pot.model_hidden1;
    const int output_offset = input_weight_count;
    const int bias_offset = output_offset + pot.model_hidden1;
    int hidden = 0;
    int input = 0;
    if (parameter < input_weight_count) {
        hidden = parameter / input_count;
        input = parameter - hidden * input_count;
    } else if (parameter < bias_offset) {
        hidden = parameter - output_offset;
    } else {
        hidden = parameter - bias_offset;
    }
    const double output_weight = pot.model_coeffs[output_offset + hidden];
    double derivative = 0.0;
    double directional_derivative = 0.0;
    if (parameter < input_weight_count) {
        derivative = cache.q[input] * output_weight * cache.d1[hidden];
        directional_derivative = output_weight *
            (cache.dq[input] * cache.d1[hidden] + cache.q[input] *
             cache.dd1[hidden] * cache.du[hidden]);
    } else if (parameter < bias_offset) {
        derivative = cache.h1[hidden];
        directional_derivative = cache.d1[hidden] * cache.du[hidden];
    } else {
        derivative = output_weight * cache.d1[hidden];
        directional_derivative =
            output_weight * cache.dd1[hidden] * cache.du[hidden];
    }
    return lambda * derivative + directional_derivative;
}

__device__ double mtpm_nn2_model_parameter_derivative_cached(
    const DevicePotential &pot, int parameter, double lambda,
    const NeuralCachePointers &cache)
{
    const int input_count = pot.descriptor_count;
    const int first_weight_count = input_count * pot.model_hidden1;
    const int first_bias_offset = first_weight_count;
    const int second_weight_offset = first_bias_offset + pot.model_hidden1;
    const int second_weight_count = pot.model_hidden1 * pot.model_hidden2;
    const int second_bias_offset = second_weight_offset + second_weight_count;
    const int output_offset = second_bias_offset + pot.model_hidden2;
    double derivative = 0.0;
    double directional_derivative = 0.0;

    if (parameter < second_weight_offset) {
        const bool is_weight = parameter < first_weight_count;
        const int hidden1 = is_weight
            ? parameter / input_count : parameter - first_bias_offset;
        const int input = is_weight
            ? parameter - hidden1 * input_count : 0;
        const double delta =
            cache.d1[hidden1] * cache.upstream[hidden1];
        const double delta_direction =
            cache.dd1[hidden1] * cache.du[hidden1] *
                cache.upstream[hidden1] +
            cache.d1[hidden1] * cache.upstream_direction[hidden1];
        if (is_weight) {
            derivative = cache.q[input] * delta;
            directional_derivative =
                cache.dq[input] * delta + cache.q[input] * delta_direction;
        } else {
            derivative = delta;
            directional_derivative = delta_direction;
        }
    } else if (parameter < second_bias_offset) {
        const int relative = parameter - second_weight_offset;
        const int hidden2 = relative / pot.model_hidden1;
        const int hidden1 = relative - hidden2 * pot.model_hidden1;
        const double output_weight =
            pot.model_coeffs[output_offset + hidden2];
        derivative = output_weight * cache.d2[hidden2] * cache.h1[hidden1];
        directional_derivative = output_weight *
            (cache.dd2[hidden2] * cache.dz[hidden2] * cache.h1[hidden1] +
             cache.d2[hidden2] * cache.d1[hidden1] * cache.du[hidden1]);
    } else if (parameter < output_offset) {
        const int hidden2 = parameter - second_bias_offset;
        const double output_weight =
            pot.model_coeffs[output_offset + hidden2];
        derivative = output_weight * cache.d2[hidden2];
        directional_derivative =
            output_weight * cache.dd2[hidden2] * cache.dz[hidden2];
    } else {
        const int hidden2 = parameter - output_offset;
        derivative = cache.h2[hidden2];
        directional_derivative = cache.d2[hidden2] * cache.dz[hidden2];
    }
    return lambda * derivative + directional_derivative;
}

__device__ double mtpm_nonradial_parameter_derivative_cached(
    const DevicePotential &pot, int parameter, int itype, double lambda,
    const double *mom, const double *dir, const double *cache_base)
{
    NeuralCachePointers cache = mtpm_neural_cache_pointers(
        pot, const_cast<double *>(cache_base));
    if (parameter < pot.species_count) {
        return parameter == itype ? lambda : 0.0;
    }
    parameter -= pot.species_count;
    const int descriptor_species_count =
        pot.descriptor_count * pot.species_count;
    if (parameter < descriptor_species_count) {
        const int descriptor = parameter / pot.species_count;
        const int species = parameter - descriptor * pot.species_count;
        if (species != itype) return 0.0;
        return lambda * cache.gradient[descriptor] +
            cache.hessian_direction[descriptor];
    }
    parameter -= descriptor_species_count;
    const int descriptor_moment_count =
        pot.descriptor_count * pot.alpha_scalar_moments;
    if (parameter < descriptor_moment_count) {
        const int descriptor = parameter / pot.alpha_scalar_moments;
        const int scalar = parameter -
            descriptor * pot.alpha_scalar_moments;
        const int mid = pot.alpha_moment_mapping[scalar];
        return lambda * cache.gradient[descriptor] * mom[mid] +
            cache.gradient[descriptor] * dir[mid] +
            cache.hessian_direction[descriptor] * mom[mid];
    }
    parameter -= descriptor_moment_count;
    if (pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1)) {
        return mtpm_nn1_model_parameter_derivative_cached(
            pot, parameter, lambda, cache);
    }
    return mtpm_nn2_model_parameter_derivative_cached(
        pot, parameter, lambda, cache);
}

__device__ double mtpm_nonradial_parameter_derivative(
    const DevicePotential &pot, int parameter, int itype, double lambda,
    const double *mom, const double *dir)
{
    if (parameter < pot.species_count) {
        return parameter == itype ? lambda : 0.0;
    }
    parameter -= pot.species_count;
    const int descriptor_species_count =
        pot.descriptor_count * pot.species_count;
    if (parameter < descriptor_species_count) {
        const int descriptor = parameter / pot.species_count;
        const int species = parameter - descriptor * pot.species_count;
        if (species != itype) return 0.0;
        return lambda * mtpm_external_model_gradient_component(
                            pot, mom, itype, descriptor) +
            mtpm_external_model_hessian_direction_component(
                pot, mom, dir, itype, descriptor);
    }
    parameter -= descriptor_species_count;
    const int descriptor_moment_count =
        pot.descriptor_count * pot.alpha_scalar_moments;
    if (parameter < descriptor_moment_count) {
        const int descriptor = parameter / pot.alpha_scalar_moments;
        const int scalar = parameter - descriptor * pot.alpha_scalar_moments;
        const int mid = pot.alpha_moment_mapping[scalar];
        const double gradient = mtpm_external_model_gradient_component(
            pot, mom, itype, descriptor);
        const double gradient_direction =
            mtpm_external_model_hessian_direction_component(
                pot, mom, dir, itype, descriptor);
        return lambda * gradient * mom[mid] + gradient * dir[mid] +
            gradient_direction * mom[mid];
    }
    parameter -= descriptor_moment_count;
    return mtpm_model_parameter_derivative(
        pot, parameter, itype, lambda, mom, dir);
}

__global__ void mtpm_gradient_reduction_kernel(
    DevicePotential pot, int inum, const int *centers, const int *type,
    const int *center_structure, const double *dloss_denergy,
    const double *moment_vals, const double *directional,
    const double *neural_cache, double *gradient, int radial_count)
{
    extern __shared__ double partial[];
    const int parameter = blockIdx.x;
    const int tid = threadIdx.x;
    double sum = 0.0;
    for (int center_idx = tid; center_idx < inum; center_idx += blockDim.x) {
        const int ati = centers[center_idx];
        const int itype = pot.spmap[type[ati]];
        const double *mom = moment_vals +
            center_idx * pot.alpha_moments_count;
        const double *dir = directional +
            center_idx * pot.alpha_moments_count;
        const double lambda =
            dloss_denergy[center_structure[center_idx]];
        if (neural_cache != nullptr) {
            const int stride = mtpm_neural_cache_stride(pot);
            sum += mtpm_nonradial_parameter_derivative_cached(
                pot, parameter, itype, lambda, mom, dir,
                neural_cache + center_idx * stride);
        } else {
            sum += mtpm_nonradial_parameter_derivative(
                pot, parameter, itype, lambda, mom, dir);
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) gradient[radial_count + parameter] += partial[0];
}

__global__ void mtpm_gradient_ordered_kernel(
    DevicePotential pot, int inum, const int *centers, const int *type,
    const int *center_structure, const double *dloss_denergy,
    const double *moment_vals, const double *directional,
    const double *neural_cache, double *gradient, int radial_count)
{
    const int parameter = blockIdx.x * blockDim.x + threadIdx.x;
    if (parameter >= pot.nonradial_parameter_count) return;
    double sum = 0.0;
    for (int center_idx = 0; center_idx < inum; ++center_idx) {
        const int ati = centers[center_idx];
        const int itype = pot.spmap[type[ati]];
        const double *mom = moment_vals +
            center_idx * pot.alpha_moments_count;
        const double *dir = directional +
            center_idx * pot.alpha_moments_count;
        const double lambda =
            dloss_denergy[center_structure[center_idx]];
        if (neural_cache != nullptr) {
            const int stride = mtpm_neural_cache_stride(pot);
            sum += mtpm_nonradial_parameter_derivative_cached(
                pot, parameter, itype, lambda, mom, dir,
                neural_cache + center_idx * stride);
        } else {
            sum += mtpm_nonradial_parameter_derivative(
                pot, parameter, itype, lambda, mom, dir);
        }
    }
    gradient[radial_count + parameter] += sum;
}

__global__ void linear_gradient_reduction_kernel(
    DevicePotential pot, int inum, const int *centers, const int *type,
    const int *center_structure, const double *dloss_denergy,
    const double *moment_vals, const double *directional,
    double *gradient, int radial_count)
{
    extern __shared__ double partial[];
    const int parameter = blockIdx.x;
    const int tid = threadIdx.x;
    double sum = 0.0;
    if (parameter < pot.species_count) {
        for (int center_idx = tid; center_idx < inum;
             center_idx += blockDim.x) {
            const int ati = centers[center_idx];
            if (pot.spmap[type[ati]] == parameter) {
                sum += dloss_denergy[center_structure[center_idx]];
            }
        }
    } else {
        const int scalar = parameter - pot.species_count;
        const int mid = pot.alpha_moment_mapping[scalar];
        for (int center_idx = tid; center_idx < inum;
             center_idx += blockDim.x) {
            const double lambda =
                dloss_denergy[center_structure[center_idx]];
            const double *mom =
                moment_vals + center_idx * pot.alpha_moments_count;
            const double *dir =
                directional + center_idx * pot.alpha_moments_count;
            sum += lambda * mom[mid] + dir[mid];
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        gradient[radial_count + parameter] += partial[0];
    }
}

__global__ void linear_gradient_ordered_kernel(
    DevicePotential pot, int inum, const int *centers, const int *type,
    const int *center_structure, const double *dloss_denergy,
    const double *moment_vals, const double *directional,
    double *gradient, int radial_count)
{
    const int parameter = blockIdx.x * blockDim.x + threadIdx.x;
    const int linear_count = pot.species_count + pot.alpha_scalar_moments;
    if (parameter >= linear_count) return;
    double sum = 0.0;
    if (parameter < pot.species_count) {
        for (int center_idx = 0; center_idx < inum; ++center_idx) {
            const int ati = centers[center_idx];
            if (pot.spmap[type[ati]] == parameter) {
                sum += dloss_denergy[center_structure[center_idx]];
            }
        }
    } else {
        const int scalar = parameter - pot.species_count;
        const int mid = pot.alpha_moment_mapping[scalar];
        for (int center_idx = 0; center_idx < inum; ++center_idx) {
            const double lambda =
                dloss_denergy[center_structure[center_idx]];
            const double *mom =
                moment_vals + center_idx * pot.alpha_moments_count;
            const double *dir =
                directional + center_idx * pot.alpha_moments_count;
            sum += lambda * mom[mid] + dir[mid];
        }
    }
    gradient[radial_count + parameter] += sum;
}

// Fast radial reverse pass. Each thread owns one species-local radial
// parameter and accumulates it in a register. Basic moments are grouped by mu
// during initialization, so a parameter thread only visits moments that can
// contribute to it. Pair geometry and loss directions are shared across the
// block. This preserves the compact center row while removing the contended
// FP64 shared-memory atomics from the production path.
__global__ void radial_gradient_partial_kernel(
    DevicePotential pot, const int *centers, const int *offsets,
    const int *neighbors, const double *neighbor_vectors, const int *type,
    const double *x, const int *center_structure, const double *dloss_denergy,
    const double *dloss_dforce, const double *dloss_dstress,
    const double *dmom, const double *directional_adjoint,
    double *radial_partials, int center_radial_count,
    int radial_parameter_tile, int neighbor_lanes)
{
    extern __shared__ double shared[];
    const int center_idx = blockIdx.x;
    const int tile_begin = blockIdx.y * radial_parameter_tile;
    const int local_parameter = tile_begin + threadIdx.x;
    const bool parameter_valid = local_parameter < center_radial_count;

    double *rb_vals = shared;
    double *rb_ders = rb_vals + neighbor_lanes * pot.rb_size;
    double *dist_powers = rb_ders + neighbor_lanes * pot.rb_size;
    double *coord_powers =
        dist_powers + neighbor_lanes * pot.max_alpha_index_basic;
    double *rvec_cache =
        coord_powers + neighbor_lanes * pot.coord_power_count * 3;
    double *rijinv_cache = rvec_cache + neighbor_lanes * 3;
    double *valid_cache = rijinv_cache + neighbor_lanes;
    double *jtype_cache = valid_cache + neighbor_lanes;
    double *loss_direction_cache = jtype_cache + neighbor_lanes;

    const int pair_width = pot.radial_func_count * pot.rb_size;
    const int target_jtype = parameter_valid
        ? local_parameter / pair_width : -1;
    const int pair_parameter = parameter_valid
        ? local_parameter - target_jtype * pair_width : 0;
    const int mu = pair_parameter / pot.rb_size;
    const int xi = pair_parameter - mu * pot.rb_size;

    const int ati = centers[center_idx];
    const int structure = center_structure[center_idx];
    const double lambda = dloss_denergy[structure];
    const double *adj = dmom + center_idx * pot.alpha_moments_count;
    const double *hadj =
        directional_adjoint + center_idx * pot.alpha_moments_count;
    const int start = offsets[center_idx];
    const int jnum = offsets[center_idx + 1] - start;
    double parameter_sum = 0.0;

    for (int neighbor_begin = 0; neighbor_begin < jnum;
         neighbor_begin += neighbor_lanes) {
        if (threadIdx.x < neighbor_lanes) {
            const int lane = threadIdx.x;
            const int jj = neighbor_begin + lane;
            if (jj < jnum) {
                fill_pair_cache(
                    pot, lane, center_idx, ati, jj, offsets, neighbors,
                    neighbor_vectors, type, x, rb_vals, rb_ders,
                    nullptr, nullptr, dist_powers, coord_powers,
                    rvec_cache, rijinv_cache, valid_cache, true, false);
                if (valid_cache[lane] != 0.0) {
                    const int atj = neighbors[start + jj];
                    jtype_cache[lane] =
                        static_cast<double>(pot.spmap[type[atj]]);
                    loss_direction(
                        ati, atj, structure, rvec_cache + 3 * lane,
                        dloss_dforce, dloss_dstress,
                        &loss_direction_cache[3 * lane + 0],
                        &loss_direction_cache[3 * lane + 1],
                        &loss_direction_cache[3 * lane + 2]);
                }
            } else {
                valid_cache[lane] = 0.0;
            }
        }
        __syncthreads();

        if (parameter_valid && mu < pot.max_mu) {
            const int mu_begin = pot.basic_mu_offsets[mu];
            const int mu_end = pot.basic_mu_offsets[mu + 1];
            for (int lane = 0; lane < neighbor_lanes; ++lane) {
                if (valid_cache[lane] == 0.0 ||
                    static_cast<int>(jtype_cache[lane]) != target_jtype) {
                    continue;
                }
                const double *rbv = rb_vals + lane * pot.rb_size;
                const double *rbd = rb_ders + lane * pot.rb_size;
                const double *dp =
                    dist_powers + lane * pot.max_alpha_index_basic;
                const double *cp =
                    coord_powers + lane * pot.coord_power_count * 3;
                const double *rv = rvec_cache + lane * 3;
                const double rijinv = rijinv_cache[lane];
                const double wx = loss_direction_cache[3 * lane + 0];
                const double wy = loss_direction_cache[3 * lane + 1];
                const double wz = loss_direction_cache[3 * lane + 2];

                for (int grouped = mu_begin; grouped < mu_end; ++grouped) {
                    const int bi = pot.basic_mu_indices[grouped];
                    const int b1 = pot.basic_b1[bi];
                    const int b2 = pot.basic_b2[bi];
                    const int b3 = pot.basic_b3[bi];
                    const int nu = pot.basic_nu[bi];
                    const double rpow = dp[nu];
                    const double pow0 = cp[3 * (b1 + 1) + 0];
                    const double pow1 = cp[3 * (b2 + 1) + 1];
                    const double pow2 = cp[3 * (b3 + 1) + 2];
                    const double monomial = pow0 * pow1 * pow2;
                    const double basis = rbv[xi] * rpow;
                    const double dbdc = basis * monomial;
                    const double tail =
                        rbd[xi] * rpow * monomial -
                        rbv[xi] * nu * rpow * monomial * rijinv;
                    double dx = rv[0] * rijinv * tail;
                    double dy = rv[1] * rijinv * tail;
                    double dz = rv[2] * rijinv * tail;
                    dx += basis * b1 * cp[3 * b1 + 0] * pow1 * pow2;
                    dy += basis * b2 * cp[3 * b2 + 1] * pow0 * pow2;
                    dz += basis * b3 * cp[3 * b3 + 2] * pow0 * pow1;
                    const double coeff_weight =
                        lambda * adj[bi] + hadj[bi];
                    parameter_sum += coeff_weight * dbdc + adj[bi] *
                        (wx * dx + wy * dy + wz * dz);
                }
            }
        }
        __syncthreads();
    }

    if (parameter_valid) {
        radial_partials[
            static_cast<size_t>(center_idx) * center_radial_count +
            local_parameter] = parameter_sum;
    }
}

// Ordered diagnostics retain the original basic-lane decomposition and its
// shared accumulation. The fast path above is deterministic within each
// parameter thread but intentionally has a different floating-point order.
__global__ void radial_gradient_partial_atomic_kernel(
    DevicePotential pot, const int *centers, const int *offsets,
    const int *neighbors, const double *neighbor_vectors, const int *type,
    const double *x, const int *center_structure, const double *dloss_denergy,
    const double *dloss_dforce, const double *dloss_dstress,
    const double *dmom, const double *directional_adjoint,
    double *radial_partials, int center_radial_count,
    int radial_parameter_tile)
{
    extern __shared__ double shared[];
    const int center_idx = blockIdx.x;
    const int tile_begin = blockIdx.y * radial_parameter_tile;
    const int remaining = center_radial_count - tile_begin;
    const int tile_count = remaining < radial_parameter_tile
        ? remaining : radial_parameter_tile;
    const int nlane = blockDim.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * blockDim.x + tx;
    const int nthreads = blockDim.x * blockDim.y;

    double *parameter_partial = shared;
    double *rb_vals = parameter_partial + radial_parameter_tile;
    double *rb_ders = rb_vals + nlane * pot.rb_size;
    double *val_mu = rb_ders + nlane * pot.rb_size;
    double *der_mu = val_mu + nlane * pot.max_mu;
    double *dist_powers = der_mu + nlane * pot.max_mu;
    double *coord_powers = dist_powers + nlane * pot.max_alpha_index_basic;
    double *rvec_cache = coord_powers + nlane * pot.coord_power_count * 3;
    double *rijinv_cache = rvec_cache + nlane * 3;
    double *valid_cache = rijinv_cache + nlane;

    const int ati = centers[center_idx];
    const int structure = center_structure[center_idx];
    const double lambda = dloss_denergy[structure];
    const int itype = pot.spmap[type[ati]];
    const double *adj = dmom + center_idx * pot.alpha_moments_count;
    const double *hadj = directional_adjoint + center_idx * pot.alpha_moments_count;
    const int start = offsets[center_idx];
    const int jnum = offsets[center_idx + 1] - start;

    for (int parameter = tid; parameter < tile_count;
         parameter += nthreads) {
        parameter_partial[parameter] = 0.0;
    }
    __syncthreads();

    for (int neighbor_begin = 0; neighbor_begin < jnum; neighbor_begin += nlane) {
        const int jj = neighbor_begin + tx;
        if (ty == 0) {
            if (jj < jnum) {
                fill_pair_cache(pot, tx, center_idx, ati, jj,
                                offsets, neighbors,
                                neighbor_vectors, type, x, rb_vals, rb_ders,
                                val_mu, der_mu, dist_powers, coord_powers,
                                rvec_cache, rijinv_cache, valid_cache, true);
            } else {
                valid_cache[tx] = 0.0;
            }
        }
        __syncthreads();

        if (valid_cache[tx] != 0.0) {
            const int atj = neighbors[start + jj];
            const int jtype = pot.spmap[type[atj]];
            const double *rbv = rb_vals + tx * pot.rb_size;
            const double *rbd = rb_ders + tx * pot.rb_size;
            const double *dp = dist_powers + tx * pot.max_alpha_index_basic;
            const double *cp = coord_powers + tx * pot.coord_power_count * 3;
            const double *rv = rvec_cache + tx * 3;
            const double rijinv = rijinv_cache[tx];
            double wx, wy, wz;
            loss_direction(ati, atj, structure, rv, dloss_dforce,
                           dloss_dstress, &wx, &wy, &wz);

            if (itype >= 0 && jtype >= 0) {
            for (int bi = ty; bi < pot.alpha_index_basic_count;
                 bi += blockDim.y) {
                const int mu = pot.basic_mu[bi];
                const int b1 = pot.basic_b1[bi];
                const int b2 = pot.basic_b2[bi];
                const int b3 = pot.basic_b3[bi];
                const int nu = pot.basic_nu[bi];
                const double rpow = dp[nu];
                const double pow0 = cp[3 * (b1 + 1) + 0];
                const double pow1 = cp[3 * (b2 + 1) + 1];
                const double pow2 = cp[3 * (b3 + 1) + 2];
                const double monomial = pow0 * pow1 * pow2;
                const double coeff_weight = lambda * adj[bi] + hadj[bi];
                const int coeff_offset =
                    jtype * pot.radial_func_count * pot.rb_size +
                    mu * pot.rb_size;
                const int xi_begin = tile_begin > coeff_offset
                    ? tile_begin - coeff_offset : 0;
                const int xi_end = tile_begin + tile_count <
                        coeff_offset + pot.rb_size
                    ? tile_begin + tile_count - coeff_offset
                    : pot.rb_size;
                for (int xi = xi_begin; xi < xi_end; ++xi) {
                    const double basis = rbv[xi] * rpow;
                    const double dbdc = basis * monomial;
                    const double tail =
                        rbd[xi] * rpow * monomial -
                        rbv[xi] * nu * rpow * monomial * rijinv;
                    double dx = rv[0] * rijinv * tail;
                    double dy = rv[1] * rijinv * tail;
                    double dz = rv[2] * rijinv * tail;
                    dx += basis * b1 * cp[3 * b1 + 0] * pow1 * pow2;
                    dy += basis * b2 * cp[3 * b2 + 1] * pow0 * pow2;
                    dz += basis * b3 * cp[3 * b3 + 2] * pow0 * pow1;
                    const double directional_dbdc = wx * dx + wy * dy + wz * dz;
                    atomic_add_double(
                        &parameter_partial[coeff_offset + xi - tile_begin],
                        coeff_weight * dbdc + adj[bi] * directional_dbdc);
                }
            }
            }
        }
        __syncthreads();
    }

    for (int parameter = tid; parameter < tile_count;
         parameter += nthreads) {
        radial_partials[static_cast<size_t>(center_idx) * center_radial_count +
                        tile_begin + parameter] =
            parameter_partial[parameter];
    }
}

__global__ void radial_gradient_reduction_kernel(
    DevicePotential pot, int inum, const int *centers, const int *type,
    int center_radial_count, const double *radial_partials, double *gradient)
{
    extern __shared__ double partial[];
    const int parameter = blockIdx.x;
    const int itype = parameter / center_radial_count;
    const int local_parameter = parameter - itype * center_radial_count;
    const int tid = threadIdx.x;
    double sum = 0.0;
    for (int center = tid; center < inum; center += blockDim.x) {
        const int ati = centers[center];
        if (pot.spmap[type[ati]] == itype) {
            sum += radial_partials[
                static_cast<size_t>(center) * center_radial_count +
                local_parameter];
        }
    }
    partial[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) gradient[parameter] += partial[0];
}

__global__ void radial_gradient_ordered_kernel(
    DevicePotential pot, int inum, const int *centers, const int *type,
    int radial_count, int center_radial_count,
    const double *radial_partials, double *gradient)
{
    const int parameter = blockIdx.x * blockDim.x + threadIdx.x;
    if (parameter >= radial_count) return;
    const int itype = parameter / center_radial_count;
    const int local_parameter = parameter - itype * center_radial_count;
    double sum = 0.0;
    for (int center = 0; center < inum; ++center) {
        const int ati = centers[center];
        if (pot.spmap[type[ati]] == itype) {
            sum += radial_partials[
                static_cast<size_t>(center) * center_radial_count +
                local_parameter];
        }
    }
    gradient[parameter] += sum;
}

__global__ void supervised_force_loss_kernel(
    int natoms, const double *prediction, const double *target,
    const double *weight, double *adjoint, double *metrics)
{
    const int atom = blockIdx.x * blockDim.x + threadIdx.x;
    if (atom >= natoms) return;
    const double w = weight[atom];
    double squared_error = 0.0;
    for (int a = 0; a < 3; ++a) {
        const int index = 3 * atom + a;
        const double error = prediction[index] - target[index];
        adjoint[index] = 2.0 * w * error;
        squared_error += error * error;
    }
    if (w != 0.0) {
        const double loss = w * squared_error;
        atomic_add_double(&metrics[0], loss);
        atomic_add_double(&metrics[2], loss);
        atomic_add_double(&metrics[5], squared_error);
    }
}

__global__ void supervised_structure_loss_kernel(
    int structure_count, const int *structure_offsets,
    const double *site_energy, const double *site_virial,
    const double *target_energy, const double *energy_weight,
    const double *target_stress, const double *stress_weight,
    double *energy_adjoint, double *stress_adjoint, double *metrics)
{
    const int structure = blockIdx.x * blockDim.x + threadIdx.x;
    if (structure >= structure_count) return;
    const int begin = structure_offsets[structure];
    const int end = structure_offsets[structure + 1];
    const int natoms = end - begin;

    double energy = 0.0;
    double stress[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    for (int atom = begin; atom < end; ++atom) {
        energy += site_energy[atom];
        stress[0] += site_virial[6 * atom + 0];
        stress[4] += site_virial[6 * atom + 1];
        stress[8] += site_virial[6 * atom + 2];
        stress[1] += site_virial[6 * atom + 3];
        stress[2] += site_virial[6 * atom + 4];
        stress[5] += site_virial[6 * atom + 5];
    }
    stress[3] = stress[1];
    stress[6] = stress[2];
    stress[7] = stress[5];

    const double ew = energy_weight[structure];
    const double energy_error = energy - target_energy[structure];
    energy_adjoint[structure] = 2.0 * ew * energy_error;
    if (ew != 0.0) {
        const double squared_error = energy_error * energy_error;
        const double loss = ew * squared_error;
        atomic_add_double(&metrics[0], loss);
        atomic_add_double(&metrics[1], loss);
        if (natoms > 0) {
            atomic_add_double(&metrics[4], squared_error /
                (static_cast<double>(natoms) * natoms));
        }
    }

    const double sw = stress_weight[structure];
    double stress_squared_error = 0.0;
    for (int component = 0; component < 9; ++component) {
        const int index = 9 * structure + component;
        const double error = stress[component] - target_stress[index];
        stress_adjoint[index] = 2.0 * sw * error;
        stress_squared_error += error * error;
    }
    if (sw != 0.0) {
        const double loss = sw * stress_squared_error;
        atomic_add_double(&metrics[0], loss);
        atomic_add_double(&metrics[3], loss);
        atomic_add_double(&metrics[6], stress_squared_error);
    }
}

__global__ void supervised_loss_ordered_kernel(
    int natoms, int structure_count, const int *structure_offsets,
    const double *force_prediction, const double *target_force,
    const double *force_weight, const double *site_energy,
    const double *site_virial, const double *target_energy,
    const double *energy_weight, const double *target_stress,
    const double *stress_weight, double *force_adjoint,
    double *energy_adjoint, double *stress_adjoint, double *metrics)
{
    for (int atom = 0; atom < natoms; ++atom) {
        const double weight = force_weight[atom];
        double squared_error = 0.0;
        for (int component = 0; component < 3; ++component) {
            const int index = 3 * atom + component;
            const double error =
                force_prediction[index] - target_force[index];
            force_adjoint[index] = 2.0 * weight * error;
            squared_error += error * error;
        }
        if (weight != 0.0) {
            const double loss = weight * squared_error;
            metrics[0] += loss;
            metrics[2] += loss;
            metrics[5] += squared_error;
        }
    }

    for (int structure = 0; structure < structure_count; ++structure) {
        const int begin = structure_offsets[structure];
        const int end = structure_offsets[structure + 1];
        const int structure_atoms = end - begin;
        double energy = 0.0;
        double stress[9] = {
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int atom = begin; atom < end; ++atom) {
            energy += site_energy[atom];
            stress[0] += site_virial[6 * atom + 0];
            stress[4] += site_virial[6 * atom + 1];
            stress[8] += site_virial[6 * atom + 2];
            stress[1] += site_virial[6 * atom + 3];
            stress[2] += site_virial[6 * atom + 4];
            stress[5] += site_virial[6 * atom + 5];
        }
        stress[3] = stress[1];
        stress[6] = stress[2];
        stress[7] = stress[5];

        const double ew = energy_weight[structure];
        const double energy_error = energy - target_energy[structure];
        energy_adjoint[structure] = 2.0 * ew * energy_error;
        if (ew != 0.0) {
            const double squared_error = energy_error * energy_error;
            const double loss = ew * squared_error;
            metrics[0] += loss;
            metrics[1] += loss;
            if (structure_atoms > 0) {
                metrics[4] += squared_error /
                    (static_cast<double>(structure_atoms) * structure_atoms);
            }
        }

        const double sw = stress_weight[structure];
        double stress_squared_error = 0.0;
        for (int component = 0; component < 9; ++component) {
            const int index = 9 * structure + component;
            const double error = stress[component] - target_stress[index];
            stress_adjoint[index] = 2.0 * sw * error;
            stress_squared_error += error * error;
        }
        if (sw != 0.0) {
            const double loss = sw * stress_squared_error;
            metrics[0] += loss;
            metrics[3] += loss;
            metrics[6] += stress_squared_error;
        }
    }
}

// SOG uses the same compact scalar-moment mapping as the SR head.
__global__ void sog_latents_kernel(DevicePotential pot, int n, int channels,
    const int *centers, const int *types, const double *moments,
    const double *coeffs, double *values, bool directional)
{
    extern __shared__ double shared[];
    const int i=blockIdx.x, t=threadIdx.x;
    const double *mom=moments+size_t(i)*pot.alpha_moments_count;
    const double *b=coeffs+channels*pot.species_count;
    for(int c=0;c<channels;++c) {
        double value=0.;
        for(int a=t;a<pot.alpha_scalar_moments;a+=blockDim.x)
            value+=b[c*pot.alpha_scalar_moments+a]*mom[pot.alpha_moment_mapping[a]];
        shared[t]=value;__syncthreads();
        for(int stride=blockDim.x/2;stride;stride/=2){if(t<stride)shared[t]+=shared[t+stride];__syncthreads();}
        if(t==0)values[size_t(c)*n+i]=shared[0]+(directional?0.:coeffs[c*pot.species_count+pot.spmap[types[centers[i]]]]);
        __syncthreads();
    }
}

__global__ void sog_seed_kernel(DevicePotential pot,int n,int channels,
    const double *coeffs,const double *weights,double *adjoints)
{
    const int i=blockIdx.x,t=threadIdx.x;
    if(!pot.unique_scalar_mapping && t!=0)return;
    const int step=pot.unique_scalar_mapping ? blockDim.x : 1;
    const double *b=coeffs+channels*pot.species_count;
    for(int a=pot.unique_scalar_mapping?t:0;a<pot.alpha_scalar_moments;a+=step) {
        double value=0.;for(int c=0;c<channels;++c)value+=weights[size_t(c)*n+i]*b[c*pot.alpha_scalar_moments+a];
        adjoints[size_t(i)*pot.alpha_moments_count+pot.alpha_moment_mapping[a]]+=value;
    }
}

__global__ void sog_head_gradient_kernel(DevicePotential pot,int n,int channels,
    const int *centers,const int *type,const int *structure,const double *lambda,
    const double *moments,const double *directional,const double *phi,const double *gq,
    double *gradient,int base)
{
    extern __shared__ double shared[];
    const int p=blockIdx.x,t=threadIdx.x,ns=channels*pot.species_count;
    const bool species=p<ns;
    const int c=species?p/pot.species_count:(p-ns)/pot.alpha_scalar_moments;
    const int a=species?p%pot.species_count:(p-ns)%pot.alpha_scalar_moments;
    double sum=0.;
    for(int i=t;i<n;i+=blockDim.x) {
        const size_t at=size_t(c)*n+i;
        const double g=lambda[structure[i]]*phi[at]+gq[at];
        if(species){if(pot.spmap[type[centers[i]]]==a)sum+=g;}
        else {const size_t mid=size_t(i)*pot.alpha_moments_count+pot.alpha_moment_mapping[a];sum+=g*moments[mid]+phi[at]*directional[mid];}
    }
    shared[t]=sum;__syncthreads();
    for(int stride=blockDim.x/2;stride;stride/=2){if(t<stride)shared[t]+=shared[t+stride];__syncthreads();}
    if(t==0)gradient[base+p]+=shared[0];
}
__global__ void sog_kernel_gradient_add(int count,const double *physical,const double *jacobian,double *out)
{
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){out[i]+=physical[i];out[count+i]+=physical[count+i]*jacobian[i];}
}

constexpr bool uses_fp32_moments(MTPGpuPrecision precision)
{
    return precision != MTPGpuPrecision::FP64;
}

constexpr bool uses_fp32_adjoint(MTPGpuPrecision precision)
{
    return precision == MTPGpuPrecision::MixedAggressive;
}

size_t basic_shared_bytes(const DevicePotential &pot, int neighbor_lanes,
                          int basic_tile, MTPGpuPrecision precision)
{
    const size_t partial_elements =
        static_cast<size_t>(neighbor_lanes) * (basic_tile + 1);
    const size_t work_bytes = partial_elements *
        (uses_fp32_moments(precision) ? sizeof(float) : sizeof(double));
    const size_t aligned_work_bytes =
        (work_bytes + alignof(double) - 1) & ~(alignof(double) - 1);
    const size_t doubles =
        static_cast<size_t>(neighbor_lanes) * pot.rb_size +
        static_cast<size_t>(neighbor_lanes) * pot.max_mu +
        static_cast<size_t>(neighbor_lanes) * pot.max_alpha_index_basic +
        static_cast<size_t>(neighbor_lanes) * pot.coord_power_count * 3 +
        static_cast<size_t>(neighbor_lanes) * 3 +
        static_cast<size_t>(neighbor_lanes) +
        static_cast<size_t>(neighbor_lanes);
    return aligned_work_bytes + doubles * sizeof(double);
}

size_t force_shared_bytes(const DevicePotential &pot, int neighbor_lanes, int basic_lanes)
{
    const size_t doubles =
        static_cast<size_t>(neighbor_lanes) * pot.rb_size * 2 +
        static_cast<size_t>(neighbor_lanes) * pot.max_mu * 2 +
        static_cast<size_t>(neighbor_lanes) * pot.max_alpha_index_basic +
        static_cast<size_t>(neighbor_lanes) * pot.coord_power_count * 3 +
        static_cast<size_t>(neighbor_lanes) * 3 +
        static_cast<size_t>(neighbor_lanes) +
        static_cast<size_t>(neighbor_lanes) +
        static_cast<size_t>(neighbor_lanes) * basic_lanes * 3;
    return doubles * sizeof(double);
}

size_t directional_shared_bytes(const DevicePotential &pot, int neighbor_lanes,
                                int basic_tile)
{
    const size_t doubles =
        static_cast<size_t>(neighbor_lanes) * (basic_tile + 1) +
        static_cast<size_t>(neighbor_lanes) * pot.rb_size * 2 +
        static_cast<size_t>(neighbor_lanes) * pot.max_mu * 2 +
        static_cast<size_t>(neighbor_lanes) * pot.max_alpha_index_basic +
        static_cast<size_t>(neighbor_lanes) * pot.coord_power_count * 3 +
        static_cast<size_t>(neighbor_lanes) * 3 +
        static_cast<size_t>(neighbor_lanes) +
        static_cast<size_t>(neighbor_lanes);
    return doubles * sizeof(double);
}

size_t radial_gradient_shared_bytes(const DevicePotential &pot, int neighbor_lanes)
{
    const size_t doubles =
        static_cast<size_t>(neighbor_lanes) * pot.rb_size * 2 +
        static_cast<size_t>(neighbor_lanes) * pot.max_mu * 2 +
        static_cast<size_t>(neighbor_lanes) * pot.max_alpha_index_basic +
        static_cast<size_t>(neighbor_lanes) * pot.coord_power_count * 3 +
        static_cast<size_t>(neighbor_lanes) * 3 +
        static_cast<size_t>(neighbor_lanes) +
        static_cast<size_t>(neighbor_lanes);
    return doubles * sizeof(double);
}

size_t radial_parameter_shared_bytes(const DevicePotential &pot,
                                     int neighbor_lanes)
{
    const size_t doubles =
        static_cast<size_t>(neighbor_lanes) * pot.rb_size * 2 +
        static_cast<size_t>(neighbor_lanes) * pot.max_alpha_index_basic +
        static_cast<size_t>(neighbor_lanes) * pot.coord_power_count * 3 +
        static_cast<size_t>(neighbor_lanes) * 3 +
        static_cast<size_t>(neighbor_lanes) * 6;
    return doubles * sizeof(double);
}

} // namespace

struct MTPGpuDirect::State {
    DevicePotential pot;
    std::unique_ptr<SOGGpu> sog;
    int sog_channels=0, sog_terms=0;
    std::vector<double> sog_weights, sog_widths;
    bool sog_onsite=true;
    double *d_sog_coeffs=nullptr;
    size_t capacity_sog_coeffs=0;
    MTPGpuPrecision precision = MTPGpuPrecision::FP64;
    size_t potential_bytes = 0;
    int declared_moment_slots = 0;
    int forward_block_groups = 0;
    int reverse_block_groups = 0;
    bool initialized = false;
    int device_id = -1;
    gpuDeviceProp device_props{};
    bool device_props_valid = false;
    size_t configured_basic_float_shmem = 0;
    size_t configured_basic_double_shmem = 0;
    size_t configured_force_float_shmem = 0;
    size_t configured_force_double_shmem = 0;
    size_t configured_center_force_float_shmem = 0;
    size_t configured_center_force_double_shmem = 0;
    size_t configured_directional_shmem = 0;
    size_t configured_radial_atomic_shmem = 0;
    size_t configured_radial_shmem = 0;
    bool topology_valid = false;
    int topology_inum = 0;
    int topology_ntotal = 0;
    int topology_neighbors = 0;
    std::uint64_t topology_generation = 0;
    std::uint64_t identity = next_cache_identity();
    GpuCacheToken topology_token, input_token, labels_token;
    int labels_structures = 0;
    bool input_has_vectors = false;
    bool prepared_loss_valid = false;
    MTPGpuRuntimeStats runtime_stats;
    struct CallGuard {
        State &s;
        bool success = false;
        ~CallGuard() {
            if (success) return;
            if (s.device_id >= 0) gpuSetDevice(s.device_id);
            if (s.input_transfer_stream) gpuStreamSynchronize(s.input_transfer_stream);
            if (s.label_transfer_stream) gpuStreamSynchronize(s.label_transfer_stream);
            s.forward_valid = s.prepared_loss_valid = s.gradient_valid = false;
            s.topology_valid = false;
            s.topology_token = s.input_token = s.labels_token = {};
        }
    };
    bool forward_valid = false;
    bool gradient_valid = false;
    bool forward_deterministic_reduction = false;
    bool prepared_loss_deterministic_reduction = false;
    int forward_inum = 0;
    int forward_ntotal = 0;
    int forward_neighbors = 0;
    bool forward_has_neighbor_vectors = false;
    bool forward_weighted = false;

    size_t capacity_centers = 0;
    size_t capacity_offsets = 0;
    size_t capacity_neighbors = 0;
    size_t capacity_neighbor_vectors = 0;
    size_t capacity_type = 0;
    size_t capacity_x = 0;
    size_t staging_capacity_centers = 0;
    size_t staging_capacity_offsets = 0;
    size_t staging_capacity_neighbors = 0;
    size_t staging_capacity_neighbor_vectors = 0;
    size_t staging_capacity_type = 0;
    size_t staging_capacity_x = 0;
    size_t capacity_force = 0;
    size_t capacity_site_energy = 0;
    size_t capacity_center_energy_weights = 0;
    size_t capacity_site_virial = 0;
    size_t capacity_moment_vals = 0;
    size_t capacity_dmom = 0;
    size_t capacity_moment_vals_mixed = 0;
    size_t capacity_dmom_mixed = 0;
    size_t capacity_directional = 0;
    size_t capacity_directional_adjoint = 0;
    size_t capacity_center_structure = 0;
    size_t capacity_dloss_denergy = 0;
    size_t capacity_dloss_dforce = 0;
    size_t capacity_dloss_dstress = 0;
    size_t capacity_gradient = 0;
    size_t capacity_structure_offsets = 0;
    size_t capacity_target_energy = 0;
    size_t capacity_energy_weight = 0;
    size_t capacity_target_force = 0;
    size_t capacity_force_weight = 0;
    size_t capacity_target_stress = 0;
    size_t capacity_stress_weight = 0;
    size_t capacity_radial_partials = 0;
    size_t capacity_neural_cache = 0;

    int *d_centers = nullptr;
    int *d_offsets = nullptr;
    int *d_neighbors = nullptr;
    double *d_neighbor_vectors = nullptr;
    int *d_type = nullptr;
    double *d_x = nullptr;
    int *staging_d_centers = nullptr;
    int *staging_d_offsets = nullptr;
    int *staging_d_neighbors = nullptr;
    double *staging_d_neighbor_vectors = nullptr;
    int *staging_d_type = nullptr;
    double *staging_d_x = nullptr;
    double *d_force = nullptr;
    double *d_site_energy = nullptr;
    double *d_center_energy_weights = nullptr;
    double *d_site_virial = nullptr;
    double *d_total_energy = nullptr;
    double *d_global_virial = nullptr;
    double *d_moment_vals = nullptr;
    double *d_dmom = nullptr;
    float *d_moment_vals_mixed = nullptr;
    float *d_dmom_mixed = nullptr;
    double *d_directional = nullptr;
    double *d_directional_adjoint = nullptr;
    int *d_center_structure = nullptr;
    double *d_dloss_denergy = nullptr;
    double *d_dloss_dforce = nullptr;
    double *d_dloss_dstress = nullptr;
    double *d_gradient = nullptr;
    int *d_structure_offsets = nullptr;
    double *d_target_energy = nullptr;
    double *d_energy_weight = nullptr;
    double *d_target_force = nullptr;
    double *d_force_weight = nullptr;
    double *d_target_stress = nullptr;
    double *d_stress_weight = nullptr;
    double *d_loss_metrics = nullptr;
    double *d_radial_partials = nullptr;
    double *d_neural_cache = nullptr;
    gpuStream_t label_transfer_stream = nullptr;
    gpuEvent_t labels_ready_event = nullptr;
    gpuEvent_t label_buffers_safe_event = nullptr;
    bool label_buffers_safe_recorded = false;
    gpuStream_t input_transfer_stream = nullptr;
    gpuEvent_t input_ready_event = nullptr;
    gpuEvent_t active_input_safe_event = nullptr;
    gpuEvent_t staging_input_safe_event = nullptr;
    bool active_input_used = false;
    bool staging_input_used = false;
    std::vector<double> host_force_delta;

    void free_potential()
    {
        if (device_id >= 0) gpuSetDevice(device_id);
        free_device_potential(&pot);
        sog.reset(); sog_channels=sog_terms=0;
        if(d_sog_coeffs)gpuFree(d_sog_coeffs);
        d_sog_coeffs=nullptr;capacity_sog_coeffs=0;
        declared_moment_slots = 0;
        forward_block_groups = 0;
        reverse_block_groups = 0;
        potential_bytes = 0;
        initialized = false;
        forward_valid = false;
        gradient_valid = false;
        forward_deterministic_reduction = false;
        prepared_loss_deterministic_reduction = false;
    }

    void free_buffers()
    {
        if (device_id >= 0) gpuSetDevice(device_id);
        gpuFree(d_centers);
        gpuFree(d_offsets);
        gpuFree(d_neighbors);
        gpuFree(d_neighbor_vectors);
        gpuFree(d_type);
        gpuFree(d_x);
        gpuFree(staging_d_centers);
        gpuFree(staging_d_offsets);
        gpuFree(staging_d_neighbors);
        gpuFree(staging_d_neighbor_vectors);
        gpuFree(staging_d_type);
        gpuFree(staging_d_x);
        gpuFree(d_force);
        gpuFree(d_site_energy);
        gpuFree(d_center_energy_weights);
        gpuFree(d_site_virial);
        gpuFree(d_total_energy);
        gpuFree(d_global_virial);
        gpuFree(d_moment_vals);
        gpuFree(d_dmom);
        gpuFree(d_moment_vals_mixed);
        gpuFree(d_dmom_mixed);
        gpuFree(d_directional);
        gpuFree(d_directional_adjoint);
        gpuFree(d_center_structure);
        gpuFree(d_dloss_denergy);
        gpuFree(d_dloss_dforce);
        gpuFree(d_dloss_dstress);
        gpuFree(d_gradient);
        gpuFree(d_structure_offsets);
        gpuFree(d_target_energy);
        gpuFree(d_energy_weight);
        gpuFree(d_target_force);
        gpuFree(d_force_weight);
        gpuFree(d_target_stress);
        gpuFree(d_stress_weight);
        gpuFree(d_loss_metrics);
        gpuFree(d_radial_partials);
        gpuFree(d_neural_cache);
        if (labels_ready_event != nullptr) gpuEventDestroy(labels_ready_event);
        if (label_buffers_safe_event != nullptr)
            gpuEventDestroy(label_buffers_safe_event);
        if (label_transfer_stream != nullptr) gpuStreamDestroy(label_transfer_stream);
        if (active_input_safe_event != nullptr)
            gpuEventDestroy(active_input_safe_event);
        if (staging_input_safe_event != nullptr)
            gpuEventDestroy(staging_input_safe_event);
        if (input_ready_event != nullptr) gpuEventDestroy(input_ready_event);
        if (input_transfer_stream != nullptr)
            gpuStreamDestroy(input_transfer_stream);
        d_centers = nullptr;
        d_offsets = nullptr;
        d_neighbors = nullptr;
        d_neighbor_vectors = nullptr;
        d_type = nullptr;
        d_x = nullptr;
        staging_d_centers = nullptr;
        staging_d_offsets = nullptr;
        staging_d_neighbors = nullptr;
        staging_d_neighbor_vectors = nullptr;
        staging_d_type = nullptr;
        staging_d_x = nullptr;
        d_force = nullptr;
        d_site_energy = nullptr;
        d_center_energy_weights = nullptr;
        d_site_virial = nullptr;
        d_total_energy = nullptr;
        d_global_virial = nullptr;
        d_moment_vals = nullptr;
        d_dmom = nullptr;
        d_moment_vals_mixed = nullptr;
        d_dmom_mixed = nullptr;
        d_directional = nullptr;
        d_directional_adjoint = nullptr;
        d_center_structure = nullptr;
        d_dloss_denergy = nullptr;
        d_dloss_dforce = nullptr;
        d_dloss_dstress = nullptr;
        d_gradient = nullptr;
        d_structure_offsets = nullptr;
        d_target_energy = nullptr;
        d_energy_weight = nullptr;
        d_target_force = nullptr;
        d_force_weight = nullptr;
        d_target_stress = nullptr;
        d_stress_weight = nullptr;
        d_loss_metrics = nullptr;
        d_radial_partials = nullptr;
        d_neural_cache = nullptr;
        labels_ready_event = nullptr;
        label_buffers_safe_event = nullptr;
        label_buffers_safe_recorded = false;
        label_transfer_stream = nullptr;
        input_transfer_stream = nullptr;
        input_ready_event = nullptr;
        active_input_safe_event = nullptr;
        staging_input_safe_event = nullptr;
        active_input_used = false;
        staging_input_used = false;
        capacity_centers = 0;
        capacity_offsets = 0;
        capacity_neighbors = 0;
        capacity_neighbor_vectors = 0;
        capacity_type = 0;
        capacity_x = 0;
        staging_capacity_centers = 0;
        staging_capacity_offsets = 0;
        staging_capacity_neighbors = 0;
        staging_capacity_neighbor_vectors = 0;
        staging_capacity_type = 0;
        staging_capacity_x = 0;
        capacity_force = 0;
        capacity_site_energy = 0;
        capacity_center_energy_weights = 0;
        capacity_site_virial = 0;
        capacity_moment_vals = 0;
        capacity_dmom = 0;
        capacity_moment_vals_mixed = 0;
        capacity_dmom_mixed = 0;
        capacity_directional = 0;
        capacity_directional_adjoint = 0;
        capacity_center_structure = 0;
        capacity_dloss_denergy = 0;
        capacity_dloss_dforce = 0;
        capacity_dloss_dstress = 0;
        capacity_gradient = 0;
        capacity_structure_offsets = 0;
        capacity_target_energy = 0;
        capacity_energy_weight = 0;
        capacity_target_force = 0;
        capacity_force_weight = 0;
        capacity_target_stress = 0;
        capacity_stress_weight = 0;
        capacity_radial_partials = 0;
        capacity_neural_cache = 0;
        topology_valid = false;
        topology_inum = 0;
        topology_ntotal = 0;
        topology_neighbors = 0;
        forward_valid = false;
        gradient_valid = false;
        forward_deterministic_reduction = false;
        prepared_loss_deterministic_reduction = false;
        forward_inum = 0;
        forward_ntotal = 0;
        forward_neighbors = 0;
        forward_has_neighbor_vectors = false;
    }
};

MTPGpuDirect::MTPGpuDirect() : state_(new State)
{
}

MTPGpuDirect::~MTPGpuDirect()
{
    if (state_ != nullptr) {
        state_->free_potential();
        state_->free_buffers();
        delete state_;
        state_ = nullptr;
    }
}

bool MTPGpuDirect::available_device_count(int *count, std::string *error)
{
    if (count == nullptr) {
        set_error(error, "GPU device-count output pointer is null");
        return false;
    }
    *count = 0;
    return check_gpu(gpuGetDeviceCount(count), "gpuGetDeviceCount(count)", error);
}

const char *MTPGpuDirect::backend_name() { return compiled_backend(); }

bool MTPGpuDirect::device_info(int device, MTPGpuDeviceInfo *info, std::string *error)
{
    if (info == nullptr) {
        set_error(error, "GPU device-info output pointer is null");
        return false;
    }
    if (device < 0) MTP_GPU_CHECK(gpuGetDevice(&device));
    gpuDeviceProp props{};
    MTP_GPU_CHECK(gpuGetDeviceProperties(&props, device));
    info->name = props.name;
    info->total_memory = props.totalGlobalMem;
    info->wave_size = props.warpSize;
    info->compute_units = props.multiProcessorCount;
#if defined(IMRMLP_HIP)
    info->architecture = props.gcnArchName;
    char pci[64]{};
    MTP_GPU_CHECK(gpuDeviceGetPCIBusId(pci, sizeof(pci), device));
    info->identity = pci;
#else
    info->architecture = "sm_" + std::to_string(props.major * 10 + props.minor);
    // UUID also distinguishes CUDA MIG instances sharing a PCI address.
    constexpr char digits[] = "0123456789abcdef";
    info->identity.clear();
    for (unsigned char byte : props.uuid.bytes) {
        info->identity += digits[byte >> 4];
        info->identity += digits[byte & 15];
    }
#endif
    return true;
}

bool MTPGpuDirect::set_device(int device_id, std::string *error)
{
    if (state_->initialized || state_->d_total_energy != nullptr) {
        set_error(error, "GPU device must be selected before MTPGpuDirect initialization");
        return false;
    }
    if (device_id < 0) {
        MTP_GPU_CHECK(gpuGetDevice(&state_->device_id));
        MTP_GPU_CHECK(gpuGetDeviceProperties(
            &state_->device_props, state_->device_id));
        state_->device_props_valid = true;
        return true;
    }
    int count = 0;
    MTP_GPU_CHECK(gpuGetDeviceCount(&count));
    if (device_id >= count) {
        set_error(error, "requested GPU device id is not visible");
        return false;
    }
    MTP_GPU_CHECK(gpuSetDevice(device_id));
    state_->device_id = device_id;
    MTP_GPU_CHECK(gpuGetDeviceProperties(
        &state_->device_props, state_->device_id));
    state_->device_props_valid = true;
    return true;
}

bool MTPGpuDirect::initialize(const MTPGpuPotentialView &potential, std::string *error)
{
    return initialize(potential, MTPGpuInitializationOptions{}, error);
}

bool MTPGpuDirect::initialize(
    const MTPGpuPotentialView &potential,
    const MTPGpuInitializationOptions &options,
    std::string *error)
{
    invalidate_forward();
    state_->identity = next_cache_identity();
    state_->topology_token = state_->input_token = state_->labels_token = {};
    state_->topology_valid = false;
    if (options.precision != MTPGpuPrecision::FP64 &&
        options.precision != MTPGpuPrecision::Mixed &&
        options.precision != MTPGpuPrecision::MixedAggressive) {
        set_error(error, "unknown GPU MTP precision mode");
        return false;
    }
    if (uses_fp32_moments(options.precision) &&
        potential.model != MTPGpuModel::Linear) {
        set_error(error,
            "mixed precision currently supports classic linear PotMTP inference only");
        return false;
    }
    if (state_->device_id < 0) {
        MTP_GPU_CHECK(gpuGetDevice(&state_->device_id));
    } else {
        MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
    }
    if (!state_->device_props_valid) {
        MTP_GPU_CHECK(gpuGetDeviceProperties(
            &state_->device_props, state_->device_id));
        state_->device_props_valid = true;
    }
    HostPotential host;
    if (!build_host_potential(potential, &host, error)) {
        return false;
    }
    if (uses_fp32_moments(options.precision)) {
        for (const auto &basis : host.pair_radial_bases) {
            if (basis.rb_type > static_cast<int>(RadialBasis::ChebyshevConfigurable) ||
                basis.one_minus_chebyshev ||
                (basis.rb_type == static_cast<int>(RadialBasis::ChebyshevConfigurable) &&
                 (basis.cutoff_type > static_cast<int>(RadialCutoff::Poly5) ||
                  basis.transform_type > static_cast<int>(RadialTransform::Exponential)))) {
                set_error(error, "extended radial bases currently require precision fp64");
                return false;
            }
        }
    }

    std::vector<ForwardTerm> forward_terms;
    forward_terms.reserve(host.schedule.forward_terms.size());
    for (int t : host.schedule.forward_terms) {
        forward_terms.push_back({host.schedule.times_l1[t],
                                 host.schedule.times_l2[t],
                                 host.schedule.times_c[t]});
    }
    std::vector<ReverseTerm> reverse_entries;
    reverse_entries.reserve(host.schedule.reverse_entries.size());
    for (int entry : host.schedule.reverse_entries) {
        const int t = entry >> 1;
        reverse_entries.push_back({host.schedule.times_id[t],
            (entry & 1) ? host.schedule.times_l1[t] : host.schedule.times_l2[t],
            host.schedule.times_c[t]});
    }

    const bool precision_changed = state_->precision != options.precision;
    state_->free_potential();
    if (precision_changed) {
        state_->free_buffers();
    }
    DevicePotential pot = host.dims;

    const int radial_count = potential.radial_coeffs_count > 0
        ? potential.radial_coeffs_count
        : potential.species_count * potential.species_count *
              potential.radial_func_count * potential.rb_size;

    const size_t integer_entries =
        host.spmap.size() + host.basic_mu.size() + host.basic_b1.size() +
        host.basic_b2.size() + host.basic_b3.size() + host.basic_nu.size() +
        host.basic_mu_offsets.size() + host.basic_mu_indices.size() +
        host.alpha_moment_mapping.size() + host.schedule.times_l1.size() +
        host.schedule.times_l2.size() + host.schedule.times_c.size() +
        host.schedule.times_id.size() + host.schedule.level_offsets.size() +
        host.schedule.forward_level_offsets.size() +
        host.schedule.forward_ids.size() +
        host.schedule.forward_term_offsets.size() +
        host.schedule.forward_block_level_offsets.size() +
        host.schedule.forward_block_groups.size() +
        host.schedule.reverse_level_offsets.size() +
        host.schedule.reverse_ids.size() +
        host.schedule.reverse_entry_offsets.size() +
        host.schedule.reverse_block_level_offsets.size() +
        host.schedule.reverse_block_groups.size() +
        host.schedule.forward_short_level_offsets.size() +
        host.schedule.forward_short_groups.size() +
        host.schedule.forward_warp_level_offsets.size() +
        host.schedule.forward_warp_groups.size() +
        host.schedule.reverse_short_level_offsets.size() +
        host.schedule.reverse_short_groups.size() +
        host.schedule.reverse_warp_level_offsets.size() +
        host.schedule.reverse_warp_groups.size();
    const size_t coefficient_entries = static_cast<size_t>(radial_count) +
        potential.species_count +
        (potential.model == MTPGpuModel::Linear
             ? static_cast<size_t>(potential.alpha_scalar_moments) : 0) +
        static_cast<size_t>(potential.descriptor_count) *
            potential.species_count +
        static_cast<size_t>(potential.descriptor_count) *
            potential.alpha_scalar_moments +
        potential.model_coeffs_count;
    const size_t new_potential_bytes = integer_entries * sizeof(int) +
        forward_terms.size() * sizeof(ForwardTerm) +
        reverse_entries.size() * sizeof(ReverseTerm) +
        host.pair_radial_bases.size() * sizeof(DeviceRadialBasis) +
        host.radial_basis_parameters.size() * sizeof(double) +
        coefficient_entries * sizeof(double);
    if (options.memory_budget_bytes && new_potential_bytes > options.memory_budget_bytes) {
        set_error(error, "model GPU storage exceeds memory_budget_bytes before upload");
        return false;
    }
    if (options.memory_budget_bytes && memory_stats().tracked_device_bytes >
        options.memory_budget_bytes - new_potential_bytes)
        state_->free_buffers();

    if (!upload_vector(&pot.spmap, host.spmap, error) ||
        !upload_vector(&pot.basic_mu, host.basic_mu, error) ||
        !upload_vector(&pot.basic_b1, host.basic_b1, error) ||
        !upload_vector(&pot.basic_b2, host.basic_b2, error) ||
        !upload_vector(&pot.basic_b3, host.basic_b3, error) ||
        !upload_vector(&pot.basic_nu, host.basic_nu, error) ||
        !upload_vector(&pot.basic_mu_offsets, host.basic_mu_offsets, error) ||
        !upload_vector(&pot.basic_mu_indices, host.basic_mu_indices, error) ||
        !upload_vector(&pot.times_l1, host.schedule.times_l1, error) ||
        !upload_vector(&pot.times_l2, host.schedule.times_l2, error) ||
        !upload_vector(&pot.times_c, host.schedule.times_c, error) ||
        !upload_vector(&pot.times_id, host.schedule.times_id, error) ||
        !upload_vector(&pot.level_offsets, host.schedule.level_offsets, error) ||
        !upload_vector(&pot.forward_level_offsets,
                       host.schedule.forward_level_offsets, error) ||
        !upload_vector(&pot.forward_ids, host.schedule.forward_ids, error) ||
        !upload_vector(&pot.forward_term_offsets,
                       host.schedule.forward_term_offsets, error) ||
        !upload_vector(&pot.forward_terms, forward_terms, error) ||
        !upload_vector(&pot.forward_block_level_offsets,
                       host.schedule.forward_block_level_offsets, error) ||
        !upload_vector(&pot.forward_block_groups,
                       host.schedule.forward_block_groups, error) ||
        !upload_vector(&pot.reverse_level_offsets,
                       host.schedule.reverse_level_offsets, error) ||
        !upload_vector(&pot.reverse_ids, host.schedule.reverse_ids, error) ||
        !upload_vector(&pot.reverse_entry_offsets,
                       host.schedule.reverse_entry_offsets, error) ||
        !upload_vector(&pot.reverse_entries, reverse_entries, error) ||
        !upload_vector(&pot.reverse_block_level_offsets,
                       host.schedule.reverse_block_level_offsets, error) ||
        !upload_vector(&pot.reverse_block_groups,
                       host.schedule.reverse_block_groups, error) ||
        !upload_vector(&pot.forward_short_level_offsets, host.schedule.forward_short_level_offsets, error) ||
        !upload_vector(&pot.forward_short_groups, host.schedule.forward_short_groups, error) ||
        !upload_vector(&pot.forward_warp_level_offsets, host.schedule.forward_warp_level_offsets, error) ||
        !upload_vector(&pot.forward_warp_groups, host.schedule.forward_warp_groups, error) ||
        !upload_vector(&pot.reverse_short_level_offsets, host.schedule.reverse_short_level_offsets, error) ||
        !upload_vector(&pot.reverse_short_groups, host.schedule.reverse_short_groups, error) ||
        !upload_vector(&pot.reverse_warp_level_offsets, host.schedule.reverse_warp_level_offsets, error) ||
        !upload_vector(&pot.reverse_warp_groups, host.schedule.reverse_warp_groups, error) ||
        !upload_vector(&pot.alpha_moment_mapping,
                       host.alpha_moment_mapping, error) ||
        !upload_vector(&pot.pair_radial_bases,
                       host.pair_radial_bases, error) ||
        !upload_vector(&pot.radial_basis_parameters, host.radial_basis_parameters, error) ||
        !upload_array(&pot.radial_coeffs, potential.radial_coeffs, radial_count, error) ||
        !upload_array(&pot.species_coeffs, potential.species_coeffs,
                      potential.species_count, error) ||
        !upload_array(&pot.moment_coeffs, potential.moment_coeffs,
                      potential.model == MTPGpuModel::Linear
                          ? potential.alpha_scalar_moments : 0, error) ||
        !upload_array(&pot.descriptor_species_coeffs,
                      potential.descriptor_species_coeffs,
                      potential.descriptor_count * potential.species_count,
                      error) ||
        !upload_array(&pot.descriptor_moment_coeffs,
                      potential.descriptor_moment_coeffs,
                      potential.descriptor_count *
                          potential.alpha_scalar_moments,
                      error) ||
        !upload_array(&pot.model_coeffs, potential.model_coeffs,
                      potential.model_coeffs_count, error)) {
        free_device_potential(&pot);
        return false;
    }

    state_->pot = pot;
    state_->precision = options.precision;
    state_->declared_moment_slots = potential.alpha_moments_count;
    state_->forward_block_groups =
        static_cast<int>(host.schedule.forward_block_groups.size());
    state_->reverse_block_groups =
        static_cast<int>(host.schedule.reverse_block_groups.size());
    state_->potential_bytes = new_potential_bytes;
    state_->initialized = true;
    return true;
}

bool MTPGpuDirect::initialize_from_mlip(const mlip &potential, std::string *error)
{
    return initialize_from_mlip(
        potential, MTPGpuInitializationOptions{}, error);
}

bool MTPGpuDirect::initialize_from_mlip(
    const mlip &potential, const MTPGpuInitializationOptions &options,
    std::string *error)
{
    std::vector<int> alpha_basic(4 * potential.alpha_index_basic_count);
    for (int i = 0; i < potential.alpha_index_basic_count; ++i) {
        alpha_basic[4 * i + 0] = potential.alpha_index_basic[i].mu;
        alpha_basic[4 * i + 1] = potential.alpha_index_basic[i].b1;
        alpha_basic[4 * i + 2] = potential.alpha_index_basic[i].b2;
        alpha_basic[4 * i + 3] = potential.alpha_index_basic[i].b3;
    }
    std::vector<int> alpha_times(4 * potential.alpha_index_times_count);
    for (int i = 0; i < potential.alpha_index_times_count; ++i) {
        alpha_times[4 * i + 0] = potential.alpha_index_times[i].l1;
        alpha_times[4 * i + 1] = potential.alpha_index_times[i].l2;
        alpha_times[4 * i + 2] = potential.alpha_index_times[i].c;
        alpha_times[4 * i + 3] = potential.alpha_index_times[i].id;
    }
    std::vector<MTPGpuRadialBasisView> pair_radial_bases(
        potential.pair_radial_basis_count);
    for (int pair = 0; pair < potential.pair_radial_basis_count; ++pair) {
        const auto &source = potential.pair_radial_bases[pair];
        auto &target = pair_radial_bases[pair];
        target.radial_basis = static_cast<RadialBasis>(source.rb);
        target.min_dist = source.min_dist;
        target.max_dist = source.max_dist;
        target.scaling = source.scaling;
        target.cutoff = static_cast<RadialCutoff>(source.cutoff);
        target.transform = static_cast<RadialTransform>(source.transform);
        target.cutoff_parameter = source.cutoff_parameter;
        target.transform_parameter = source.transform_parameter;
        target.one_minus_chebyshev = source.one_minus_chebyshev;
        target.cutoff_r1 = source.cutoff_r1; target.cutoff_r2 = source.cutoff_r2;
        target.transform_scale_parameter = source.transform_scale_parameter;
        target.polynomial_matrix = source.polynomial_matrix.empty() ? nullptr : source.polynomial_matrix.data();
        target.polynomial_normalizations = source.polynomial_normalizations.empty() ? nullptr : source.polynomial_normalizations.data();
    }

    MTPGpuPotentialView view;
    view.species_count = potential.species_count;
    view.max_type = static_cast<int>(potential.mapping_size());
    view.spmap = potential.mapping_data();
    view.alpha_moments_count = potential.alpha_moments_count;
    view.alpha_index_basic_count = potential.alpha_index_basic_count;
    view.alpha_index_times_count = potential.alpha_index_times_count;
    view.alpha_scalar_moments = potential.alpha_scalar_moments;
    view.radial_func_count = potential.radial_func_count;
    view.rb_size = potential.rb_size;
    view.max_mu = potential.max_mu;
    view.max_alpha_index_basic = potential.max_alpha_index_basic;
    view.min_dist = potential.min_dist;
    view.max_dist = potential.max_dist;
    view.cutoff = potential.cutoff;
    view.scaling = potential.scaling;
    view.radial_basis = static_cast<RadialBasis>(potential.rb);
    if (!pair_radial_bases.empty()) {
        view.pair_radial_bases = pair_radial_bases.data();
        view.pair_radial_basis_count =
            static_cast<int>(pair_radial_bases.size());
    }
    view.radial_coeffs = potential.radial_coeffs;
    view.radial_coeffs_count = potential.radial_coeffs_count;
    view.species_coeffs = potential.species_coeffs;
    switch (potential.site_model) {
    case mlip::MODEL_LINEAR:
        view.model = MTPGpuModel::Linear;
        break;
    case mlip::MODEL_MTPM_Q2:
        view.model = MTPGpuModel::MTPMQuadratic;
        break;
    case mlip::MODEL_MTPM_FS:
        view.model = MTPGpuModel::MTPMAsinh;
        break;
    case mlip::MODEL_MTPM_NN1:
        view.model = MTPGpuModel::MTPMNN1;
        break;
    case mlip::MODEL_MTPM_NN2:
        view.model = MTPGpuModel::MTPMNN2;
        break;
    case mlip::MODEL_MTPM_LI:
        view.model = MTPGpuModel::MTPMLinear;
        break;
    }
    view.moment_coeffs = potential.site_model == mlip::MODEL_LINEAR
        ? potential.moment_coeffs : nullptr;
    view.descriptor_count = potential.descriptor_count;
    view.descriptor_species_coeffs = potential.descriptor_species_coeffs;
    view.descriptor_moment_coeffs = potential.descriptor_moment_coeffs;
    view.model_coeffs = potential.model_coeffs;
    view.model_coeffs_count = potential.model_coeffs_count;
    view.model_activation = static_cast<MTPGpuActivation>(
        potential.model_activation);
    view.model_hidden1 = potential.model_hidden1;
    view.model_hidden2 = potential.model_hidden2;
    view.alpha_index_basic = alpha_basic.data();
    view.alpha_index_times = alpha_times.data();
    view.alpha_moment_mapping = potential.alpha_moment_mapping;
    view.nu_arr_list = potential.nu_arr_list;

    return initialize(view, options, error);
}

bool MTPGpuDirect::update_coefficients(const double *radial_coeffs,
                                       const double *species_coeffs,
                                       const double *moment_coeffs,
                                       std::string *error)
{
    return update_coefficients(radial_coeffs, species_coeffs, moment_coeffs,
                               nullptr, nullptr, nullptr, error);
}

bool MTPGpuDirect::update_coefficients(
    const double *radial_coeffs, const double *species_coeffs,
    const double *moment_coeffs, const double *descriptor_species_coeffs,
    const double *descriptor_moment_coeffs, const double *model_coeffs,
    std::string *error)
{
    invalidate_forward();
    State::CallGuard guard{*state_};
    if (!state_->initialized) {
        set_error(error, "MTPGpuDirect is not initialized");
        return false;
    }
    const bool linear =
        state_->pot.model_type == static_cast<int>(MTPGpuModel::Linear);
    if (radial_coeffs == nullptr || species_coeffs == nullptr ||
        (linear && moment_coeffs == nullptr) ||
        (!linear && (descriptor_species_coeffs == nullptr ||
                     descriptor_moment_coeffs == nullptr ||
                     (model_coeffs == nullptr && state_->pot.model_coeffs_count != 0)))) {
        set_error(error, "coefficient update contains a null pointer");
        return false;
    }
    if (state_->device_id >= 0) MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
    const int center_radial_count = state_->pot.species_count *
        state_->pot.radial_func_count * state_->pot.rb_size;
    const int radial_count = state_->pot.species_count * center_radial_count;
    MTP_GPU_CHECK(gpuMemcpy(state_->pot.radial_coeffs, radial_coeffs,
                             radial_count * sizeof(double), gpuMemcpyHostToDevice));
    MTP_GPU_CHECK(gpuMemcpy(state_->pot.species_coeffs, species_coeffs,
                             state_->pot.species_count * sizeof(double), gpuMemcpyHostToDevice));
    if (linear) {
        MTP_GPU_CHECK(gpuMemcpy(state_->pot.moment_coeffs, moment_coeffs,
                                 state_->pot.alpha_scalar_moments * sizeof(double),
                                 gpuMemcpyHostToDevice));
    } else {
        MTP_GPU_CHECK(gpuMemcpy(
            state_->pot.descriptor_species_coeffs, descriptor_species_coeffs,
            state_->pot.descriptor_count * state_->pot.species_count *
                sizeof(double), gpuMemcpyHostToDevice));
        MTP_GPU_CHECK(gpuMemcpy(
            state_->pot.descriptor_moment_coeffs, descriptor_moment_coeffs,
            state_->pot.descriptor_count * state_->pot.alpha_scalar_moments *
                sizeof(double), gpuMemcpyHostToDevice));
        if (state_->pot.model_coeffs_count > 0)
            MTP_GPU_CHECK(gpuMemcpy(
                state_->pot.model_coeffs, model_coeffs,
                state_->pot.model_coeffs_count * sizeof(double),
                gpuMemcpyHostToDevice));
    }
    state_->forward_valid = false;
    state_->gradient_valid = false;
    guard.success = true;
    return true;
}

bool MTPGpuDirect::memory_info(std::size_t *free_bytes, std::size_t *total_bytes,
                               std::string *error) const
{
    if (free_bytes == nullptr || total_bytes == nullptr) {
        set_error(error, "GPU memory-info output pointer is null");
        return false;
    }
    if (state_->device_id >= 0) {
        MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
    }
    MTP_GPU_CHECK(gpuMemGetInfo(free_bytes, total_bytes));
    return true;
}

std::uint64_t MTPGpuDirect::topology_generation() const
{
    return state_ != nullptr ? state_->topology_generation : 0;
}

std::uint64_t MTPGpuDirect::context_identity() const { return state_->identity; }
bool MTPGpuDirect::has_input(GpuCacheToken token) const {
    return token.valid() && token == state_->input_token && state_->forward_valid;
}
void MTPGpuDirect::invalidate_forward() {
    state_->forward_valid = state_->prepared_loss_valid = false;
}

MTPGpuRuntimeStats MTPGpuDirect::runtime_stats() const
{
    auto stats = state_->runtime_stats;
    if (state_->sog) {
        stats.sog_grid_upload_bytes = state_->sog->grid_upload_bytes();
        stats.sog_grid_reuses = state_->sog->grid_reuses();
    }
    return stats;
}

MTPGpuPrecision MTPGpuDirect::precision() const
{
    return state_ != nullptr ? state_->precision : MTPGpuPrecision::FP64;
}

MTPGpuPotentialStats MTPGpuDirect::potential_stats() const
{
    if (state_ == nullptr || !state_->initialized) {
        return {};
    }
    MTPGpuPotentialStats stats;
    stats.precision = state_->precision;
    stats.declared_moment_slots = state_->declared_moment_slots;
    stats.active_moment_slots =
        static_cast<int>(state_->pot.alpha_moments_count);
    stats.forward_block_groups = state_->forward_block_groups;
    stats.reverse_block_groups = state_->reverse_block_groups;
    stats.neural_cache_doubles_per_center =
        mtpm_neural_cache_stride(state_->pot);
    stats.moment_element_bytes = uses_fp32_moments(state_->precision)
        ? static_cast<int>(sizeof(float)) : static_cast<int>(sizeof(double));
    stats.adjoint_element_bytes = uses_fp32_adjoint(state_->precision)
        ? static_cast<int>(sizeof(float)) : static_cast<int>(sizeof(double));
    stats.potential_bytes = state_->potential_bytes;
    return stats;
}

MTPGpuMemoryStats MTPGpuDirect::memory_stats() const
{
    MTPGpuMemoryStats stats;
    if (state_ == nullptr) return stats;
    stats.precision = state_->precision;
    stats.potential_bytes = state_->potential_bytes;
    stats.topology_bytes = static_cast<size_t>(
        state_->capacity_centers + state_->capacity_offsets +
        state_->capacity_neighbors + state_->staging_capacity_centers +
        state_->staging_capacity_offsets + state_->staging_capacity_neighbors) *
        sizeof(int);
    stats.input_bytes = static_cast<size_t>(
        state_->capacity_type + state_->staging_capacity_type) * sizeof(int) +
        static_cast<size_t>(
            state_->capacity_neighbor_vectors + state_->capacity_x +
            state_->staging_capacity_neighbor_vectors +
            state_->staging_capacity_x) * sizeof(double);
    stats.output_bytes = static_cast<size_t>(
        state_->capacity_force + state_->capacity_site_energy +
        state_->capacity_center_energy_weights +
        state_->capacity_site_virial) * sizeof(double) +
        (state_->d_total_energy != nullptr ? sizeof(double) : 0) +
        (state_->d_global_virial != nullptr ? 6 * sizeof(double) : 0);
    stats.moment_workspace_bytes =
        static_cast<size_t>(state_->capacity_moment_vals) * sizeof(double) +
        static_cast<size_t>(state_->capacity_moment_vals_mixed) * sizeof(float);
    stats.adjoint_workspace_bytes =
        static_cast<size_t>(state_->capacity_dmom) * sizeof(double) +
        static_cast<size_t>(state_->capacity_dmom_mixed) * sizeof(float);
    stats.directional_workspace_bytes = static_cast<size_t>(
        state_->capacity_directional + state_->capacity_directional_adjoint) *
        sizeof(double);
    stats.neural_cache_bytes =
        static_cast<size_t>(state_->capacity_neural_cache) * sizeof(double);
    stats.loss_gradient_bytes = static_cast<size_t>(
        state_->capacity_center_structure + state_->capacity_structure_offsets) *
        sizeof(int) + static_cast<size_t>(
            state_->capacity_dloss_denergy + state_->capacity_dloss_dforce +
            state_->capacity_dloss_dstress + state_->capacity_gradient +
            state_->capacity_target_energy + state_->capacity_energy_weight +
            state_->capacity_target_force + state_->capacity_force_weight +
            state_->capacity_target_stress + state_->capacity_stress_weight +
            state_->capacity_radial_partials) * sizeof(double) +
        (state_->d_loss_metrics != nullptr
             ? MTPGpuLossRequest::metric_count * sizeof(double) : 0);
    stats.potential_bytes += state_->capacity_sog_coeffs*sizeof(double);
    stats.loss_gradient_bytes += state_->sog ? state_->sog->memory_bytes() : 0;
    stats.tracked_device_bytes = stats.potential_bytes + stats.topology_bytes +
        stats.input_bytes + stats.output_bytes +
        stats.moment_workspace_bytes + stats.adjoint_workspace_bytes +
        stats.directional_workspace_bytes + stats.neural_cache_bytes +
        stats.loss_gradient_bytes + stats.conversion_buffer_bytes;
    return stats;
}

bool MTPGpuDirect::configure_sog(const MTPGpuSOGView &view,std::string *error)
{
    invalidate_forward();
    State::CallGuard guard{*state_};
    if(!state_->initialized || state_->precision!=MTPGpuPrecision::FP64) {
        set_error(error,"SOG requires an initialized FP64 GPU MTP evaluator");return false;
    }
    MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
    const auto &m=view.model;
    if(m.channels<1||m.channels>16||m.terms<1||m.terms>256||!m.weights||!m.log_bandwidths
       ||!view.species_coeffs||!view.moment_coeffs||!view.bandwidth_jacobian) {
        set_error(error,"Invalid SOG GPU coefficient view");return false;
    }
    const int ns=m.channels*state_->pot.species_count;
    const int nm=m.channels*state_->pot.alpha_scalar_moments,nk=m.channels*m.terms;
    if(!ensure_capacity(&state_->d_sog_coeffs,&state_->capacity_sog_coeffs,size_t(ns)+nm+nk,error))return false;
    MTP_GPU_CHECK(gpuMemcpy(state_->d_sog_coeffs,view.species_coeffs,ns*sizeof(double),gpuMemcpyHostToDevice));
    MTP_GPU_CHECK(gpuMemcpy(state_->d_sog_coeffs+ns,view.moment_coeffs,nm*sizeof(double),gpuMemcpyHostToDevice));
    MTP_GPU_CHECK(gpuMemcpy(state_->d_sog_coeffs+ns+nm,view.bandwidth_jacobian,nk*sizeof(double),gpuMemcpyHostToDevice));
    state_->sog_channels=m.channels;state_->sog_terms=m.terms;state_->sog_onsite=m.onsite;
    state_->sog_weights.assign(m.weights,m.weights+nk);state_->sog_widths.assign(m.log_bandwidths,m.log_bandwidths+nk);
    if(!state_->sog)state_->sog=std::make_unique<SOGGpu>();
    state_->forward_valid=false;
    guard.success = true;
    return true;
}

bool MTPGpuDirect::compute_batch(const MTPGpuBatchRequest &request, std::string *error)
{
    State::CallGuard guard{*state_};
    invalidate_forward();
    if (request.center_count < 0 || request.center_count > request.nlocal ||
        request.nlocal < 0 || request.ntotal < request.nlocal ||
        request.ntotal > std::numeric_limits<int>::max() / 6 ||
        request.centers == nullptr || request.offsets == nullptr || request.type == nullptr ||
        request.x == nullptr || (request.compute_global_energy && request.total_energy == nullptr) ||
        (request.compute_forces && request.copy_forces_to_host && request.force == nullptr)) {
        set_error(error, "invalid pointer or atom count in MTPGpuBatchRequest");
        return false;
    }
    if (request.offsets[0] != 0) {
        set_error(error, "MTPGpuBatchRequest offsets must start at zero");
        return false;
    }

    std::vector<unsigned char> center_seen(request.nlocal, 0);
    for (int ii = 0; ii < request.center_count; ++ii) {
        const int ati = request.centers[ii];
        if (ati < 0 || ati >= request.nlocal || request.offsets[ii + 1] < request.offsets[ii]
            || center_seen[ati]++) {
            set_error(error, "invalid or duplicate center or offsets in MTPGpuBatchRequest");
            return false;
        }
    }
    if (request.offsets[request.center_count] && !request.neighbors) {
        set_error(error, "MTPGpuBatchRequest neighbors pointer is null");
        return false;
    }
    double *x = const_cast<double *>(request.x);
    double *force = request.force;
    double *virial = request.site_virial;

    MTPGpuComputeRequest lammps_request;
    lammps_request.sog_grid = request.sog_grid;
    lammps_request.ilist = request.centers;

    lammps_request.neighbor_vectors = request.neighbor_vectors;
    lammps_request.inum = request.center_count;
    lammps_request.nlocal = request.nlocal;
    lammps_request.ntotal = request.ntotal;
    lammps_request.type = request.type;
    lammps_request.center_energy_weights = request.center_energy_weights;
    lammps_request.x = &x;
    lammps_request.force = force ? &force : nullptr;
    lammps_request.total_energy = request.total_energy;
    lammps_request.global_virial = request.global_virial;
    lammps_request.site_energy = request.site_energy;
    lammps_request.site_virial = virial ? &virial : nullptr;
    lammps_request.accumulate_force = request.accumulate_force;
    lammps_request.reuse_topology = request.reuse_topology;
    lammps_request.copy_outputs_to_host = request.copy_outputs_to_host;
    lammps_request.copy_forces_to_host = request.copy_forces_to_host;
    lammps_request.compute_forces = request.compute_forces;
    lammps_request.compute_global_energy = request.compute_global_energy;
    lammps_request.compute_site_energy = request.compute_site_energy;
    lammps_request.deterministic_reduction = request.deterministic_reduction;
    lammps_request.force_kernel = request.force_kernel;
    lammps_request.neighbor_index_mask = -1;
    lammps_request.basic_tile = request.basic_tile;
    lammps_request.neighbor_lanes = request.neighbor_lanes;
    lammps_request.basic_lanes = request.basic_lanes;
    lammps_request.propagate_threads = request.propagate_threads;
    guard.success = compute_impl(lammps_request, &request, error);
    return guard.success;
}

bool MTPGpuDirect::prepare_loss(const MTPGpuLossRequest &request,
                                std::string *error)
{
    State::CallGuard guard{*state_};
    state_->prepared_loss_valid = false;
    if (state_->forward_weighted) {
        set_error(error, "weighted GPU forward requests are inference-only");
        return false;
    }
    if (state_->device_id >= 0) {
        MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
    }
    if (!state_->initialized || !state_->forward_valid) {
        set_error(error, "GPU MTP loss requires a successful preceding compute_batch call");
        return false;
    }
    if (request.deterministic_reduction &&
        !state_->forward_deterministic_reduction) {
        set_error(error,
            "deterministic GPU MTP loss requires a deterministic forward call");
        return false;
    }
    if (request.structure_count <= 0 || request.structure_count > std::numeric_limits<int>::max() / 9 ||
        request.structure_offsets == nullptr ||
        request.center_structure == nullptr || request.target_energy == nullptr ||
        request.energy_weight == nullptr || request.target_force == nullptr ||
        request.force_weight == nullptr || request.target_stress == nullptr ||
        request.stress_weight == nullptr || request.metrics == nullptr) {
        set_error(error, "MTPGpuLossRequest contains a null pointer or invalid structure count");
        return false;
    }
    const int inum = state_->forward_inum;
    const int ntotal = state_->forward_ntotal;
    if (request.structure_offsets[0] != 0 ||
        request.structure_offsets[request.structure_count] != ntotal) {
        set_error(error, "MTPGpuLossRequest structure offsets do not cover the resident batch");
        return false;
    }
    for (int s = 0; s < request.structure_count; ++s) {
        if (request.structure_offsets[s + 1] < request.structure_offsets[s]) {
            set_error(error, "MTPGpuLossRequest structure offsets are not monotonic");
            return false;
        }
    }
    for (int i = 0; i < inum; ++i) {
        if (request.center_structure[i] < 0 ||
            request.center_structure[i] >= request.structure_count) {
            set_error(error, "MTPGpuLossRequest center structure is out of range");
            return false;
        }
    }

    const bool reuse_labels = request.labels_token.valid() &&
        request.labels_token == state_->labels_token &&
        request.structure_count == state_->labels_structures;
    const bool resize_label_buffers =
        request.structure_count + 1 > state_->capacity_structure_offsets ||
        inum > state_->capacity_center_structure ||
        request.structure_count > state_->capacity_target_energy ||
        request.structure_count > state_->capacity_energy_weight ||
        ntotal * 3 > state_->capacity_target_force ||
        ntotal > state_->capacity_force_weight ||
        request.structure_count * 9 > state_->capacity_target_stress ||
        request.structure_count > state_->capacity_stress_weight ||
        request.structure_count > state_->capacity_dloss_denergy ||
        ntotal * 3 > state_->capacity_dloss_dforce ||
        request.structure_count * 9 > state_->capacity_dloss_dstress;
    if (resize_label_buffers && state_->label_buffers_safe_recorded) {
        // ensure_capacity may release an old allocation.  It must not do so
        // while the preceding chunk's reverse kernels still consume labels
        // or prepared adjoints.
        MTP_GPU_CHECK(gpuEventSynchronize(state_->label_buffers_safe_event));
    }

    if (!ensure_capacity(&state_->d_structure_offsets,
                         &state_->capacity_structure_offsets,
                         request.structure_count + 1, error) ||
        !ensure_capacity(&state_->d_center_structure,
                         &state_->capacity_center_structure, inum, error) ||
        !ensure_capacity(&state_->d_target_energy,
                         &state_->capacity_target_energy,
                         request.structure_count, error) ||
        !ensure_capacity(&state_->d_energy_weight,
                         &state_->capacity_energy_weight,
                         request.structure_count, error) ||
        !ensure_capacity(&state_->d_target_force,
                         &state_->capacity_target_force, ntotal * 3, error) ||
        !ensure_capacity(&state_->d_force_weight,
                         &state_->capacity_force_weight, ntotal, error) ||
        !ensure_capacity(&state_->d_target_stress,
                         &state_->capacity_target_stress,
                         request.structure_count * 9, error) ||
        !ensure_capacity(&state_->d_stress_weight,
                         &state_->capacity_stress_weight,
                         request.structure_count, error) ||
        !ensure_capacity(&state_->d_dloss_denergy,
                         &state_->capacity_dloss_denergy,
                         request.structure_count, error) ||
        !ensure_capacity(&state_->d_dloss_dforce,
                         &state_->capacity_dloss_dforce, ntotal * 3, error) ||
        !ensure_capacity(&state_->d_dloss_dstress,
                         &state_->capacity_dloss_dstress,
                         request.structure_count * 9, error)) {
        return false;
    }
    if (state_->d_loss_metrics == nullptr) {
        MTP_GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(&state_->d_loss_metrics),
                                 MTPGpuLossRequest::metric_count * sizeof(double)));
    }
    if (state_->label_transfer_stream == nullptr) {
        MTP_GPU_CHECK(gpuStreamCreateWithFlags(
            &state_->label_transfer_stream, gpuStreamNonBlocking));
    }
    if (state_->labels_ready_event == nullptr) {
        MTP_GPU_CHECK(gpuEventCreateWithFlags(
            &state_->labels_ready_event, gpuEventDisableTiming));
    }
    if (state_->label_buffers_safe_event == nullptr) {
        MTP_GPU_CHECK(gpuEventCreateWithFlags(
            &state_->label_buffers_safe_event, gpuEventDisableTiming));
    }

    const gpuStream_t transfer = state_->label_transfer_stream;
    if (!reuse_labels) {
    if (state_->label_buffers_safe_recorded) {
        // The next chunk may upload labels while its forward pass runs, but
        // it may not overwrite buffers retained by the previous reverse
        // pass.  accumulate_gradient() re-records this event after its last
        // consumer kernel.
        MTP_GPU_CHECK(gpuStreamWaitEvent(
            transfer, state_->label_buffers_safe_event, 0));
    }
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_structure_offsets, request.structure_offsets,
        (request.structure_count + 1) * sizeof(int), gpuMemcpyHostToDevice,
        transfer));
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_center_structure, request.center_structure,
        inum * sizeof(int), gpuMemcpyHostToDevice, transfer));
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_target_energy, request.target_energy,
        request.structure_count * sizeof(double), gpuMemcpyHostToDevice,
        transfer));
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_energy_weight, request.energy_weight,
        request.structure_count * sizeof(double), gpuMemcpyHostToDevice,
        transfer));
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_target_force, request.target_force,
        ntotal * 3 * sizeof(double), gpuMemcpyHostToDevice, transfer));
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_force_weight, request.force_weight,
        ntotal * sizeof(double), gpuMemcpyHostToDevice, transfer));
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_target_stress, request.target_stress,
        request.structure_count * 9 * sizeof(double), gpuMemcpyHostToDevice,
        transfer));
    MTP_GPU_CHECK(gpuMemcpyAsync(
        state_->d_stress_weight, request.stress_weight,
        request.structure_count * sizeof(double), gpuMemcpyHostToDevice,
        transfer));
    MTP_GPU_CHECK(gpuEventRecord(state_->labels_ready_event, transfer));
    state_->runtime_stats.label_upload_bytes +=
        (static_cast<std::uint64_t>(request.structure_count) + 1 + inum) * sizeof(int) +
        (static_cast<std::uint64_t>(request.structure_count) * 12 +
         static_cast<std::uint64_t>(ntotal) * 4) * sizeof(double);
    } else {
        ++state_->runtime_stats.label_reuses;
    }
    MTP_GPU_CHECK(gpuMemset(state_->d_dloss_denergy, 0,
                            request.structure_count * sizeof(double)));
    MTP_GPU_CHECK(gpuMemset(state_->d_dloss_dforce, 0,
                            ntotal * 3 * sizeof(double)));
    MTP_GPU_CHECK(gpuMemset(state_->d_dloss_dstress, 0,
                            request.structure_count * 9 * sizeof(double)));
    if (request.reset_metrics) {
        MTP_GPU_CHECK(gpuMemset(state_->d_loss_metrics, 0,
                                MTPGpuLossRequest::metric_count * sizeof(double)));
    }
    // Forward kernels from compute_batch() are already ordered on the default
    // stream. Label upload is independent and can overlap them; loss begins
    // only after both streams reach this event.
    MTP_GPU_CHECK(gpuStreamWaitEvent(nullptr, state_->labels_ready_event, 0));

    if (request.deterministic_reduction) {
        supervised_loss_ordered_kernel<<<1, 1>>>(
            ntotal, request.structure_count, state_->d_structure_offsets,
            state_->d_force, state_->d_target_force,
            state_->d_force_weight, state_->d_site_energy,
            state_->d_site_virial, state_->d_target_energy,
            state_->d_energy_weight, state_->d_target_stress,
            state_->d_stress_weight, state_->d_dloss_dforce,
            state_->d_dloss_denergy, state_->d_dloss_dstress,
            state_->d_loss_metrics);
        MTP_GPU_CHECK(gpuGetLastError());
    } else {
        constexpr int threads = 256;
        if (ntotal > 0) {
            supervised_force_loss_kernel<<<
                (ntotal + threads - 1) / threads, threads>>>(
                ntotal, state_->d_force, state_->d_target_force,
                state_->d_force_weight, state_->d_dloss_dforce,
                state_->d_loss_metrics);
            MTP_GPU_CHECK(gpuGetLastError());
        }
        supervised_structure_loss_kernel<<<
            (request.structure_count + threads - 1) / threads, threads>>>(
            request.structure_count, state_->d_structure_offsets,
            state_->d_site_energy, state_->d_site_virial,
            state_->d_target_energy, state_->d_energy_weight,
            state_->d_target_stress, state_->d_stress_weight,
            state_->d_dloss_denergy, state_->d_dloss_dstress,
            state_->d_loss_metrics);
        MTP_GPU_CHECK(gpuGetLastError());
    }
    MTP_GPU_CHECK(gpuEventRecord(state_->label_buffers_safe_event, nullptr));
    state_->label_buffers_safe_recorded = true;
    state_->prepared_loss_deterministic_reduction =
        request.deterministic_reduction;
    if (request.copy_metrics_to_host) {
        MTP_GPU_CHECK(gpuMemcpy(
            request.metrics, state_->d_loss_metrics,
            MTPGpuLossRequest::metric_count * sizeof(double),
            gpuMemcpyDeviceToHost));
    } else {
        // The caller owns the pageable label vectors and may reuse them for
        // the next chunk immediately after return.
        MTP_GPU_CHECK(gpuStreamSynchronize(transfer));
    }
    state_->labels_token = request.labels_token;
    state_->labels_structures = request.structure_count;
    state_->prepared_loss_valid = true;
    guard.success = true;
    return true;
}

bool MTPGpuDirect::accumulate_gradient(const MTPGpuGradientRequest &request,
                                       std::string *error)
{
    State::CallGuard guard{*state_};
    if (request.adjoints_on_device && !state_->prepared_loss_valid) {
        set_error(error, "GPU MTP gradient requires a successful prepared loss");
        return false;
    }
    // Host adjoints overwrite the center-structure array used by device loss.
    if (!request.adjoints_on_device) state_->labels_token = {};
    if (state_->forward_weighted) {
        set_error(error, "weighted GPU forward requests are inference-only");
        return false;
    }
    if (state_->device_id >= 0) {
        MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
    }
    if (state_->initialized && uses_fp32_moments(state_->precision)) {
        set_error(error,
            "mixed precision GPU fitting is not implemented; reinitialize the evaluator with precision fp64");
        return false;
    }
    if (!state_->initialized || !state_->forward_valid || !state_->topology_valid) {
        set_error(error, "GPU MTP gradient requires a successful preceding compute_batch call");
        return false;
    }
    if (request.deterministic_reduction &&
        !state_->forward_deterministic_reduction) {
        set_error(error,
            "deterministic GPU MTP gradient requires a deterministic forward call");
        return false;
    }
    if (request.deterministic_reduction && request.adjoints_on_device &&
        !state_->prepared_loss_deterministic_reduction) {
        set_error(error,
            "deterministic GPU MTP gradient requires deterministic prepared loss");
        return false;
    }
    if (request.structure_count <= 0 || request.gradient == nullptr ||
        (!request.adjoints_on_device &&
         (request.center_structure == nullptr || request.dloss_denergy == nullptr ||
          request.dloss_dforce == nullptr || request.dloss_dstress == nullptr))) {
        set_error(error, "MTPGpuGradientRequest contains a null pointer or invalid structure count");
        return false;
    }
    if (state_->forward_inum != state_->topology_inum ||
        state_->forward_ntotal != state_->topology_ntotal ||
        state_->forward_neighbors != state_->topology_neighbors) {
        set_error(error, "GPU MTP retained forward data does not match the resident topology");
        return false;
    }
    if (request.neighbor_lanes <= 0 || request.basic_lanes <= 0 ||
        request.basic_tile <= 0 || request.radial_parameter_tile <= 0 ||
        request.propagate_threads <= 0) {
        set_error(error, "gradient kernel tuning parameters must be positive");
        return false;
    }
    if ((request.propagate_threads & (request.propagate_threads - 1)) != 0) {
        set_error(error, "propagate_threads must be a power of two");
        return false;
    }
    const int inum = state_->forward_inum;
    const int ntotal = state_->forward_ntotal;
    if (!request.adjoints_on_device) {
        for (int i = 0; i < inum; ++i) {
            if (request.center_structure[i] < 0 ||
                request.center_structure[i] >= request.structure_count) {
                set_error(error, "center_structure entry is outside the gradient batch");
                return false;
            }
        }
    }
    const int center_radial_count = state_->pot.species_count *
        state_->pot.radial_func_count * state_->pot.rb_size;
    const int radial_count = state_->pot.species_count * center_radial_count;
    const long long expected_count = static_cast<long long>(radial_count) +
        state_->pot.nonradial_parameter_count +
        state_->sog_channels*(state_->pot.species_count+state_->pot.alpha_scalar_moments+2*state_->sog_terms);
    if (expected_count > std::numeric_limits<int>::max() ||
        request.gradient_count != static_cast<int>(expected_count)) {
        set_error(error, "gradient_count does not match the MTP coefficient layout");
        return false;
    }
    if (inum > std::numeric_limits<int>::max() / state_->pot.alpha_moments_count ||
        request.structure_count > std::numeric_limits<int>::max() / 9) {
        set_error(error, "MTPGpuGradientRequest is too large for 32-bit GPU indexing");
        return false;
    }

    const gpuDeviceProp &props = state_->device_props;
    const long long pair_threads =
        static_cast<long long>(request.neighbor_lanes) * request.basic_lanes;
    if (pair_threads > props.maxThreadsPerBlock ||
        request.neighbor_lanes > props.maxThreadsDim[0] ||
        request.basic_lanes > props.maxThreadsDim[1] ||
        request.propagate_threads > props.maxThreadsPerBlock) {
        set_error(error, "gradient kernel thread dimensions exceed GPU device limits");
        return false;
    }

    const size_t max_dynamic_shared = max_dynamic_shared_memory(props);
    int basic_tile = request.basic_tile;
    while (basic_tile > 1 &&
           directional_shared_bytes(state_->pot, request.neighbor_lanes,
                                    basic_tile) > max_dynamic_shared) {
        basic_tile = std::max(1, basic_tile / 2);
    }
    const size_t directional_shmem =
        directional_shared_bytes(state_->pot, request.neighbor_lanes, basic_tile);
    int radial_parameter_tile = std::min(request.radial_parameter_tile,
                                         center_radial_count);
    if (!request.deterministic_reduction) {
        radial_parameter_tile = std::min(radial_parameter_tile,
                                         props.maxThreadsPerBlock);
    }
    size_t radial_shmem = request.deterministic_reduction
        ? radial_gradient_shared_bytes(state_->pot, 1) +
              static_cast<size_t>(radial_parameter_tile) * sizeof(double)
        : radial_parameter_shared_bytes(state_->pot, request.neighbor_lanes);
    while (request.deterministic_reduction && radial_parameter_tile > 1 &&
           radial_shmem > max_dynamic_shared) {
        radial_parameter_tile = std::max(1, radial_parameter_tile / 2);
        radial_shmem = radial_gradient_shared_bytes(state_->pot, 1) +
            static_cast<size_t>(radial_parameter_tile) * sizeof(double);
    }
    if (directional_shmem > max_dynamic_shared || radial_shmem > max_dynamic_shared) {
        set_error(error, "MTP gradient kernel requires more shared memory than the GPU device provides");
        return false;
    }
    if (directional_shmem > props.sharedMemPerBlock) {
        if (directional_shmem > state_->configured_directional_shmem) {
            MTP_GPU_CHECK(gpuFuncSetAttribute(
                basic_directional_kernel,
                gpuFuncAttributeMaxDynamicSharedMemorySize,
                static_cast<int>(directional_shmem)));
            state_->configured_directional_shmem = directional_shmem;
        }
    }
    if (radial_shmem > props.sharedMemPerBlock) {
        if (request.deterministic_reduction) {
            if (radial_shmem > state_->configured_radial_atomic_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    radial_gradient_partial_atomic_kernel,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(radial_shmem)));
                state_->configured_radial_atomic_shmem = radial_shmem;
            }
        } else {
            if (radial_shmem > state_->configured_radial_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    radial_gradient_partial_kernel,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(radial_shmem)));
                state_->configured_radial_shmem = radial_shmem;
            }
        }
    }

    const int moment_slots = static_cast<int>(
        static_cast<size_t>(inum) * state_->pot.alpha_moments_count);
    const int neural_cache_stride = mtpm_neural_cache_stride(state_->pot);
    const long long neural_cache_count_ll =
        static_cast<long long>(inum) * neural_cache_stride;
    if (neural_cache_count_ll > std::numeric_limits<int>::max()) {
        set_error(error,
            "GPU MTP neural cache exceeds 32-bit allocation indexing");
        return false;
    }
    const int neural_cache_count =
        static_cast<int>(neural_cache_count_ll);
    if (!ensure_capacity(&state_->d_directional, &state_->capacity_directional,
                         moment_slots, error) ||
        !ensure_capacity(&state_->d_directional_adjoint,
                         &state_->capacity_directional_adjoint, moment_slots, error) ||
        !ensure_capacity(&state_->d_center_structure,
                         &state_->capacity_center_structure, inum, error) ||
        !ensure_capacity(&state_->d_dloss_denergy,
                         &state_->capacity_dloss_denergy,
                         request.structure_count, error) ||
        !ensure_capacity(&state_->d_dloss_dforce,
                         &state_->capacity_dloss_dforce, ntotal * 3, error) ||
        !ensure_capacity(&state_->d_dloss_dstress,
                         &state_->capacity_dloss_dstress,
                         request.structure_count * 9, error) ||
        !ensure_capacity(&state_->d_gradient, &state_->capacity_gradient,
                         request.gradient_count, error)) {
        return false;
    }
    if (neural_cache_count > 0 &&
        !ensure_capacity(&state_->d_neural_cache,
                         &state_->capacity_neural_cache,
                         neural_cache_count, error)) {
        return false;
    }
    const long long radial_partial_count =
        static_cast<long long>(inum) * center_radial_count;
    if (radial_partial_count > std::numeric_limits<int>::max()) {
        set_error(error,
            "GPU MTP radial partial buffer exceeds 32-bit allocation indexing");
        return false;
    }
    if (!ensure_capacity(&state_->d_radial_partials,
                         &state_->capacity_radial_partials,
                         static_cast<int>(radial_partial_count), error)) {
        return false;
    }

    if (!request.adjoints_on_device) {
        MTP_GPU_CHECK(gpuMemcpy(state_->d_center_structure, request.center_structure,
                                 inum * sizeof(int), gpuMemcpyHostToDevice));
        MTP_GPU_CHECK(gpuMemcpy(state_->d_dloss_denergy, request.dloss_denergy,
                                 request.structure_count * sizeof(double),
                                 gpuMemcpyHostToDevice));
        MTP_GPU_CHECK(gpuMemcpy(state_->d_dloss_dforce, request.dloss_dforce,
                                 ntotal * 3 * sizeof(double), gpuMemcpyHostToDevice));
        MTP_GPU_CHECK(gpuMemcpy(state_->d_dloss_dstress, request.dloss_dstress,
                                 request.structure_count * 9 * sizeof(double),
                                 gpuMemcpyHostToDevice));
    }
    MTP_GPU_CHECK(gpuMemset(state_->d_directional, 0,
                            moment_slots * sizeof(double)));
    MTP_GPU_CHECK(gpuMemset(state_->d_directional_adjoint, 0,
                            moment_slots * sizeof(double)));
    if (request.accumulate_gradient && request.accumulate_gradient_on_device) {
        if (!state_->gradient_valid) {
            set_error(error, "GPU MTP device gradient accumulation has no initial chunk");
            return false;
        }
    } else if (request.accumulate_gradient) {
        MTP_GPU_CHECK(gpuMemcpy(state_->d_gradient, request.gradient,
                                 request.gradient_count * sizeof(double),
                                 gpuMemcpyHostToDevice));
    } else {
        MTP_GPU_CHECK(gpuMemset(state_->d_gradient, 0,
                                request.gradient_count * sizeof(double)));
    }
    state_->gradient_valid = true;

    if (inum > 0) {
        const int basic_tiles =
            (state_->pot.alpha_index_basic_count + basic_tile - 1) / basic_tile;
        if (basic_tiles > props.maxGridSize[1]) {
            set_error(error, "number of gradient basic-moment tiles exceeds GPU grid-y limit");
            return false;
        }
        const dim3 grid_basic(inum, basic_tiles);
        const dim3 block_pair(request.neighbor_lanes, request.basic_lanes);
        basic_directional_kernel<<<grid_basic, block_pair, directional_shmem>>>(
            state_->pot, state_->d_centers, state_->d_offsets, state_->d_neighbors,
            state_->forward_has_neighbor_vectors ? state_->d_neighbor_vectors : nullptr,
            state_->d_type, state_->d_x,
            state_->d_center_structure, state_->d_dloss_dforce,
            state_->d_dloss_dstress, state_->d_directional, basic_tile);
        MTP_GPU_CHECK(gpuGetLastError());

        if (request.deterministic_reduction) {
            propagate_directional_ordered_kernel<<<inum, 1>>>(
                state_->pot, state_->d_centers, state_->d_type,
                state_->d_moment_vals, state_->d_dmom,
                state_->d_directional, state_->d_directional_adjoint,
                neural_cache_count > 0 ? state_->d_neural_cache : nullptr, state_->sog ? 1 : 0);
        } else {
            propagate_directional_kernel<<<
                inum, request.propagate_threads,
                request.propagate_threads * sizeof(double)>>>(
                state_->pot, state_->d_centers, state_->d_type,
                state_->d_moment_vals, state_->d_dmom,
                state_->d_directional, state_->d_directional_adjoint,
                neural_cache_count > 0 ? state_->d_neural_cache : nullptr, state_->sog ? 1 : 0);
        }
        MTP_GPU_CHECK(gpuGetLastError());

        if(state_->sog) {
            const int c=state_->sog_channels;
            const int head_count=c*(state_->pot.species_count+state_->pot.alpha_scalar_moments);
            const int nk=c*state_->sog_terms;
            const int base=radial_count+state_->pot.nonradial_parameter_count;
            sog_latents_kernel<<<inum,128,128*sizeof(double)>>>(state_->pot,inum,c,state_->d_centers,state_->d_type,
                state_->d_directional,state_->d_sog_coeffs,state_->sog->response(),true);
            MTP_GPU_CHECK(gpuGetLastError());
            if(!state_->sog->backward(state_->d_x,state_->d_dloss_denergy,state_->d_dloss_dforce,state_->d_dloss_dstress,error))return false;
            sog_seed_kernel<<<inum,128>>>(state_->pot,inum,c,state_->d_sog_coeffs,state_->sog->charge_vjp(),state_->d_directional_adjoint);
            if(request.deterministic_reduction)
                propagate_directional_ordered_kernel<<<inum,1>>>(state_->pot,state_->d_centers,state_->d_type,state_->d_moment_vals,state_->d_dmom,
                    state_->d_directional,state_->d_directional_adjoint,neural_cache_count>0?state_->d_neural_cache:nullptr,2);
            else propagate_directional_kernel<<<inum,request.propagate_threads,request.propagate_threads*sizeof(double)>>>(
                state_->pot,state_->d_centers,state_->d_type,state_->d_moment_vals,state_->d_dmom,
                state_->d_directional,state_->d_directional_adjoint,neural_cache_count>0?state_->d_neural_cache:nullptr,2);
            sog_head_gradient_kernel<<<head_count,128,128*sizeof(double)>>>(state_->pot,inum,c,state_->d_centers,state_->d_type,
                state_->d_center_structure,state_->d_dloss_denergy,state_->d_moment_vals,state_->d_directional,
                state_->sog->phi(),state_->sog->charge_vjp(),state_->d_gradient,base);
            sog_kernel_gradient_add<<<(nk+127)/128,128>>>(nk,state_->sog->kernel_gradient(),state_->d_sog_coeffs+head_count,state_->d_gradient+base+head_count);
            MTP_GPU_CHECK(gpuGetLastError());
        }

        if (neural_cache_count > 0 && request.deterministic_reduction) {
            constexpr int neural_cache_threads = 128;
            mtpm_build_neural_cache_kernel<<<inum, neural_cache_threads>>>(
                state_->pot, inum, state_->d_centers, state_->d_type,
                state_->d_moment_vals, state_->d_directional,
                state_->d_neural_cache);
            MTP_GPU_CHECK(gpuGetLastError());
        }

        const int nonradial_count = state_->pot.nonradial_parameter_count;
        if (state_->pot.model_type == static_cast<int>(MTPGpuModel::Linear)) {
            if (request.deterministic_reduction) {
                constexpr int ordered_threads = 128;
                linear_gradient_ordered_kernel<<<
                    (nonradial_count + ordered_threads - 1) / ordered_threads,
                    ordered_threads>>>(
                    state_->pot, inum, state_->d_centers, state_->d_type,
                    state_->d_center_structure, state_->d_dloss_denergy,
                    state_->d_moment_vals, state_->d_directional,
                    state_->d_gradient, radial_count);
            } else {
                linear_gradient_reduction_kernel<<<
                    nonradial_count, request.propagate_threads,
                    request.propagate_threads * sizeof(double)>>>(
                    state_->pot, inum, state_->d_centers, state_->d_type,
                    state_->d_center_structure, state_->d_dloss_denergy,
                    state_->d_moment_vals, state_->d_directional,
                    state_->d_gradient, radial_count);
            }
        } else if (request.deterministic_reduction) {
            constexpr int ordered_threads = 128;
            mtpm_gradient_ordered_kernel<<<
                (nonradial_count + ordered_threads - 1) / ordered_threads,
                ordered_threads>>>(
                state_->pot, inum, state_->d_centers, state_->d_type,
                state_->d_center_structure, state_->d_dloss_denergy,
                state_->d_moment_vals, state_->d_directional,
                neural_cache_count > 0 ? state_->d_neural_cache : nullptr,
                state_->d_gradient, radial_count);
        } else {
            mtpm_gradient_reduction_kernel<<<
                nonradial_count, request.propagate_threads,
                request.propagate_threads * sizeof(double)>>>(
                state_->pot, inum, state_->d_centers, state_->d_type,
                state_->d_center_structure, state_->d_dloss_denergy,
                state_->d_moment_vals, state_->d_directional,
                neural_cache_count > 0 ? state_->d_neural_cache : nullptr,
                state_->d_gradient, radial_count);
        }
        MTP_GPU_CHECK(gpuGetLastError());

        const int radial_tiles =
            (center_radial_count + radial_parameter_tile - 1) /
            radial_parameter_tile;
        if (radial_tiles > props.maxGridSize[1]) {
            set_error(error,
                "number of radial-parameter tiles exceeds GPU grid-y limit");
            return false;
        }
        const dim3 grid_radial(inum, radial_tiles);
        if (request.deterministic_reduction) {
            const dim3 ordered_pair(1, 1);
            radial_gradient_partial_atomic_kernel<<<
                grid_radial, ordered_pair, radial_shmem>>>(
                state_->pot, state_->d_centers, state_->d_offsets,
                state_->d_neighbors,
                state_->forward_has_neighbor_vectors
                    ? state_->d_neighbor_vectors : nullptr,
                state_->d_type, state_->d_x,
                state_->d_center_structure, state_->d_dloss_denergy,
                state_->d_dloss_dforce, state_->d_dloss_dstress,
                state_->d_dmom, state_->d_directional_adjoint,
                state_->d_radial_partials, center_radial_count,
                radial_parameter_tile);
        } else {
            const int radial_threads = std::max(
                request.neighbor_lanes,
                ((radial_parameter_tile + 31) / 32) * 32);
            const int launch_threads = std::min(
                radial_threads, props.maxThreadsPerBlock);
            radial_gradient_partial_kernel<<<
                grid_radial, launch_threads, radial_shmem>>>(
                state_->pot, state_->d_centers, state_->d_offsets,
                state_->d_neighbors,
                state_->forward_has_neighbor_vectors
                    ? state_->d_neighbor_vectors : nullptr,
                state_->d_type, state_->d_x,
                state_->d_center_structure, state_->d_dloss_denergy,
                state_->d_dloss_dforce, state_->d_dloss_dstress,
                state_->d_dmom, state_->d_directional_adjoint,
                state_->d_radial_partials, center_radial_count,
                radial_parameter_tile, request.neighbor_lanes);
        }
        MTP_GPU_CHECK(gpuGetLastError());
        if (request.deterministic_reduction) {
            constexpr int ordered_threads = 128;
            radial_gradient_ordered_kernel<<<
                (radial_count + ordered_threads - 1) / ordered_threads,
                ordered_threads>>>(
                state_->pot, inum, state_->d_centers, state_->d_type,
                radial_count, center_radial_count, state_->d_radial_partials,
                state_->d_gradient);
        } else {
            radial_gradient_reduction_kernel<<<
                radial_count, request.propagate_threads,
                request.propagate_threads * sizeof(double)>>>(
                state_->pot, inum, state_->d_centers, state_->d_type,
                center_radial_count, state_->d_radial_partials,
                state_->d_gradient);
        }
        MTP_GPU_CHECK(gpuGetLastError());
    }

    if (state_->label_buffers_safe_event != nullptr) {
        MTP_GPU_CHECK(gpuEventRecord(
            state_->label_buffers_safe_event, nullptr));
        state_->label_buffers_safe_recorded = true;
    }
    if (request.copy_gradient_to_host) {
        MTP_GPU_CHECK(gpuMemcpy(request.gradient, state_->d_gradient,
                                 request.gradient_count * sizeof(double),
                                 gpuMemcpyDeviceToHost));
    }
    guard.success = true;
    return true;
}

bool MTPGpuDirect::compute_lammps(const MTPGpuComputeRequest &request, std::string *error)
{
    return compute_impl(request, nullptr, error);
}

bool MTPGpuDirect::compute_impl(const MTPGpuComputeRequest &request,
                               const MTPGpuBatchRequest *batch, std::string *error)
{
    // Declare host staging before the guard so failed asynchronous uploads are
    // drained while their source buffers are still alive.
    std::vector<int> centers, offsets, neighbors;
    State::CallGuard guard{*state_};
    invalidate_forward();
    if (state_->device_id >= 0) {
        MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
    }
    if (state_->sog && request.center_energy_weights != nullptr) {
        set_error(error, "SOG site-energy weighting requires its nonlocal field adjoint");
        return false;
    }
    if(state_->sog) {
        state_->forward_valid=false;
        if(!request.sog_grid || !request.ilist || request.inum<=0 || request.inum!=request.ntotal || request.nlocal!=request.ntotal
            ||request.sog_grid->atoms!=request.ntotal) {
            set_error(error,"SOG GPU requires complete structures and reciprocal geometry");return false;
        }
        for(int i=0;i<request.inum;++i)if(request.ilist[i]!=i){set_error(error,"SOG GPU requires atom-ordered centers");return false;}
        MTP_GPU_CHECK(gpuSetDevice(state_->device_id));
        SOGGpuModelView model{state_->sog_channels,state_->sog_terms,state_->sog_weights.data(),state_->sog_widths.data(),state_->sog_onsite};
        if(!state_->sog->prepare(model,*request.sog_grid,error))return false;
    }

    NvtxRange pair_total_range("pair_total");
    if (!state_->initialized) {
        set_error(error, "MTPGpuDirect is not initialized");
        return false;
    }
    state_->forward_valid = false;
    state_->pot.inference_only = request.inference_only;
    state_->pot.global_virial = nullptr;
    if (request.centers_per_chunk < 0 ||
        ((request.centers_per_chunk || request.memory_budget_bytes) && !request.inference_only)) {
        set_error(error, "center chunking and memory budgets require inference_only");
        return false;
    }
    if (state_->sog && (request.centers_per_chunk || request.memory_budget_bytes ||
        (request.inference_only && request.global_virial))) {
        set_error(error, "chunking, memory budgets and direct global virial require short-range inference");
        return false;
    }
    const bool need_virial = request.site_virial != nullptr || request.global_virial != nullptr;
    // Native batch calls retain the fixed-order site-virial reduction. The
    // standalone inference contract accumulates global virial directly.
    const bool reduce_global_virial = request.global_virial && !request.inference_only;
    const bool need_site_virial = request.site_virial != nullptr || reduce_global_virial;
    const bool need_derivatives = request.compute_forces || need_virial;
    state_->prepared_loss_deterministic_reduction = false;
    if (request.inum < 0 || request.nlocal < 0 || request.ntotal < request.nlocal) {
        set_error(error, "invalid atom counts in MTPGpuComputeRequest");
        return false;
    }
    if ((request.inum > 0 && (request.ilist == nullptr ||
        (!batch && (request.numneigh == nullptr || request.firstneigh == nullptr)))) ||
        (request.ntotal > 0 && (request.type == nullptr || request.x == nullptr || request.x[0] == nullptr)) ||
        (request.compute_global_energy && request.total_energy == nullptr)) {
        set_error(error, "missing required pointer in MTPGpuComputeRequest");
        return false;
    }
    if (request.compute_forces && request.copy_forces_to_host && request.ntotal > 0 &&
        (request.force == nullptr || request.force[0] == nullptr)) {
        set_error(error, "force output pointer is required");
        return false;
    }
    if (state_->sog && !request.compute_forces && need_virial) {
        set_error(error, "SOG virial output requires force computation");
        return false;
    }
    if (request.neighbor_lanes <= 0 || request.basic_lanes <= 0 ||
        request.basic_tile <= 0 || request.propagate_threads <= 0) {
        set_error(error, "kernel tuning parameters must be positive");
        return false;
    }
    if ((request.propagate_threads & (request.propagate_threads - 1)) != 0) {
        set_error(error, "propagate_threads must be a power of two");
        return false;
    }
    if (request.force_kernel != MTPGpuForceKernel::Auto &&
        request.force_kernel != MTPGpuForceKernel::BasicTile &&
        request.force_kernel != MTPGpuForceKernel::CenterNeighbors) {
        set_error(error, "unknown MTP GPU force kernel selection");
        return false;
    }
    if (request.deterministic_reduction &&
        request.force_kernel == MTPGpuForceKernel::CenterNeighbors) {
        set_error(error,
            "CenterNeighbors force kernel is unavailable with deterministic reduction");
        return false;
    }

    if (request.ntotal > std::numeric_limits<int>::max() / 6) {
        set_error(error,
            "MTPGpuComputeRequest atom outputs exceed 32-bit GPU indexing");
        return false;
    }
    if (state_->pot.alpha_moments_count != 0 &&
        static_cast<size_t>(request.inum) >
            std::numeric_limits<size_t>::max() /
                state_->pot.alpha_moments_count) {
        set_error(error, "MTPGpuComputeRequest moment workspace size overflows size_t");
        return false;
    }

    // Even when the caller promises that the topology is unchanged, scan the
    // inexpensive row metadata.  This prevents an outdated reuse flag from
    // making the displacement upload read past a shorter neighbor-vector
    // array.  The neighbor indices themselves are still left on the device.
    int requested_neighbors = 0;
    for (int ii = 0; ii < request.inum; ++ii) {
        const int ati = request.ilist[ii];
        if (ati < 0 || ati >= request.nlocal) {
            set_error(error, "center atom index is outside the local-atom range");
            return false;
        }
        const int count = batch ? batch->offsets[ii + 1] - batch->offsets[ii] : request.numneigh[ati];
        if (count < 0 || requested_neighbors >
                             std::numeric_limits<int>::max() - count) {
            set_error(error, "invalid or overflowing neighbor count");
            return false;
        }
        requested_neighbors += count;
    }
    if (request.neighbor_vectors != nullptr &&
        requested_neighbors > std::numeric_limits<int>::max() / 3) {
        set_error(error,
            "MTPGpuComputeRequest neighbor vectors exceed 32-bit GPU indexing");
        return false;
    }

    const GpuCacheToken topology_token = batch ? batch->topology_token : GpuCacheToken{};
    const GpuCacheToken input_token = batch ? batch->input_token : GpuCacheToken{};
    const bool rebuild_topology = !request.reuse_topology || !state_->topology_valid ||
        (topology_token.valid() && !(topology_token == state_->topology_token)) ||
        state_->topology_inum != request.inum || state_->topology_ntotal != request.ntotal ||
        state_->topology_neighbors != requested_neighbors;
    const bool reuse_input = !rebuild_topology && input_token.valid() &&
        input_token == state_->input_token &&
        state_->input_has_vectors == (request.neighbor_vectors != nullptr);
    const int *center_data = request.ilist;
    const int *offset_data = batch ? batch->offsets : nullptr;
    const int *neighbor_data = batch ? batch->neighbors : nullptr;
    if (rebuild_topology && !batch) {
        NvtxRange topology_pack_range("topology_pack");
        centers.resize(request.inum);
        offsets.assign(request.inum + 1, 0);
        for (int ii = 0; ii < request.inum; ++ii) {
            const int ati = request.ilist[ii];
            if (request.type[ati] < 0 || request.type[ati] >= state_->pot.max_type ||
                state_->pot.spmap == nullptr) {
                set_error(error, "center atom type is outside the species-map range");
                return false;
            }
            centers[ii] = ati;
            offsets[ii + 1] = offsets[ii] + request.numneigh[ati];
        }
        neighbors.resize(offsets.back());
        for (int ii = 0; ii < request.inum; ++ii) {
            const int ati = centers[ii];
            const int jnum = request.numneigh[ati];
            if (jnum > 0 && request.firstneigh[ati] == nullptr) {
                set_error(error, "neighbor-list row is null");
                return false;
            }
            for (int jj = 0; jj < jnum; ++jj) {
                int atj = request.firstneigh[ati][jj];
                if (request.neighbor_index_mask >= 0) {
                    atj &= request.neighbor_index_mask;
                }
                if (atj < 0 || atj >= request.ntotal) {
                    set_error(error, "neighbor atom index is outside [0, ntotal)");
                    return false;
                }
                neighbors[offsets[ii] + jj] = atj;
            }
        }
    }
    if (rebuild_topology) {
        if (!batch) {
            center_data = centers.data(); offset_data = offsets.data(); neighbor_data = neighbors.data();
        }
        for (int edge = 0; edge < requested_neighbors; ++edge) {
            if (neighbor_data[edge] < 0 || neighbor_data[edge] >= request.ntotal) {
                set_error(error, "neighbor atom index is outside [0, ntotal)");
                return false;
            }
        }
        state_->labels_token = {};
    }
    for (int i = 0; i < request.ntotal; ++i) {
        const int atom_type = request.type[i];
        if (atom_type < 0 || atom_type >= state_->pot.max_type) {
            set_error(error, "atom type is outside the species-map range");
            return false;
        }
    }

    const gpuDeviceProp &props = state_->device_props;
    const long long pair_threads =
        static_cast<long long>(request.neighbor_lanes) * request.basic_lanes;
    if (pair_threads > props.maxThreadsPerBlock ||
        request.neighbor_lanes > props.maxThreadsDim[0] ||
        request.basic_lanes > props.maxThreadsDim[1] ||
        request.propagate_threads > props.maxThreadsPerBlock) {
        set_error(error, "kernel thread dimensions exceed GPU device limits");
        return false;
    }
    if (request.inum > props.maxGridSize[0]) {
        set_error(error, "number of center atoms exceeds GPU grid-x limit");
        return false;
    }

    int basic_tile = request.basic_tile;
    const bool fp32_moments = uses_fp32_moments(state_->precision);
    const bool fp32_adjoint = uses_fp32_adjoint(state_->precision);
    const size_t max_dynamic_shared = max_dynamic_shared_memory(props);
    while (basic_tile > 1 &&
           (basic_shared_bytes(state_->pot, request.neighbor_lanes, basic_tile,
                               state_->precision) >
                max_dynamic_shared ||
            (need_derivatives &&
             force_shared_bytes(state_->pot, request.neighbor_lanes,
                                request.basic_lanes) > max_dynamic_shared))) {
        basic_tile = std::max(1, basic_tile / 2);
    }
    const size_t basic_shmem =
        basic_shared_bytes(state_->pot, request.neighbor_lanes, basic_tile,
                           state_->precision);
    const size_t force_shmem = need_derivatives
        ? force_shared_bytes(state_->pot, request.neighbor_lanes,
                             request.basic_lanes)
        : 0;
    if (basic_shmem > max_dynamic_shared ||
        (need_derivatives && force_shmem > max_dynamic_shared)) {
        set_error(error, "MTP kernel requires more shared memory than the GPU device provides");
        return false;
    }
    if (basic_shmem > props.sharedMemPerBlock) {
        if (fp32_moments) {
            if (basic_shmem > state_->configured_basic_float_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    basic_moments_kernel<float>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(basic_shmem)));
                state_->configured_basic_float_shmem = basic_shmem;
            }
        } else {
            if (basic_shmem > state_->configured_basic_double_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    basic_moments_kernel<double>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(basic_shmem)));
                state_->configured_basic_double_shmem = basic_shmem;
            }
        }
    }
    if (need_derivatives && force_shmem > props.sharedMemPerBlock) {
        if (fp32_adjoint) {
            if (force_shmem > state_->configured_force_float_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    direct_force_kernel<float, false>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    direct_force_kernel<float, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                state_->configured_force_float_shmem = force_shmem;
            }
            if (force_shmem > state_->configured_center_force_float_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    center_neighbors_force_kernel<float, false>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    center_neighbors_force_kernel<float, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                state_->configured_center_force_float_shmem = force_shmem;
            }
        } else {
            if (force_shmem > state_->configured_force_double_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    direct_force_kernel<double, false>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    direct_force_kernel<double, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
#if defined(IMRMLP_CUDA)
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    direct_force_kernel<double, false, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    direct_force_kernel<double, true, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
#endif
                state_->configured_force_double_shmem = force_shmem;
            }
            if (force_shmem > state_->configured_center_force_double_shmem) {
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    center_neighbors_force_kernel<double, false>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    center_neighbors_force_kernel<double, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
#if defined(IMRMLP_CUDA)
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    center_neighbors_force_kernel<double, false, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
                MTP_GPU_CHECK(gpuFuncSetAttribute(
                    center_neighbors_force_kernel<double, true, true>,
                    gpuFuncAttributeMaxDynamicSharedMemorySize,
                    static_cast<int>(force_shmem)));
#endif
                state_->configured_center_force_double_shmem = force_shmem;
            }
        }
    }

    bool reserving_centers = false;
    const auto release_center_workspace = [&]() -> bool {
        const auto release = [&](auto **ptr, size_t *capacity) -> bool {
            if (*ptr) MTP_GPU_CHECK(gpuFree(*ptr));
            *ptr = nullptr; *capacity = 0; return true;
        };
        return release(&state_->d_moment_vals, &state_->capacity_moment_vals) &&
            release(&state_->d_dmom, &state_->capacity_dmom) &&
            release(&state_->d_moment_vals_mixed, &state_->capacity_moment_vals_mixed) &&
            release(&state_->d_dmom_mixed, &state_->capacity_dmom_mixed) &&
            release(&state_->d_neural_cache, &state_->capacity_neural_cache);
    };
    // Under an explicit budget allocate exact capacities and count retained
    // buffers too. Never briefly exceed the cap through geometric reserve.
    const auto reserve_call = [&](auto **ptr, size_t *capacity, size_t needed,
                                  std::string *detail) -> bool {
        using Element = std::remove_pointer_t<std::remove_pointer_t<decltype(ptr)>>;
        const bool exact_centers = reserving_centers && request.inference_only &&
            (request.memory_budget_bytes || request.centers_per_chunk);
        if (exact_centers && *capacity > needed) {
            if (*ptr) MTP_GPU_CHECK(gpuFree(*ptr));
            *ptr = nullptr; *capacity = 0;
        }
        if (needed <= *capacity) return true;
        if (!request.memory_budget_bytes && !exact_centers) {
            if (!ensure_capacity(ptr, capacity, needed, detail)) return false;
            ++state_->runtime_stats.buffer_reallocations;
            return true;
        }
        if (needed > std::numeric_limits<size_t>::max() / sizeof(Element)) {
            set_error(detail, "GPU buffer size overflows size_t"); return false;
        }
        size_t used = memory_stats().tracked_device_bytes;
        const size_t previous = *capacity * sizeof(Element);
        const size_t replacement = needed * sizeof(Element);
        const auto fits = [&]() {
            return used >= previous && used - previous <= request.memory_budget_bytes &&
                replacement <= request.memory_budget_bytes - (used - previous);
        };
        // A larger neighbor list may need the space held by an older chunk.
        // Its cache is already invalid; reclaim it only when the cap requires it.
        if (request.memory_budget_bytes && !fits() && !reserving_centers) {
            if (!release_center_workspace()) return false;
            used = memory_stats().tracked_device_bytes;
        }
        if (request.memory_budget_bytes && !fits()) {
            set_error(detail, "fixed GPU buffers exceed memory_budget_bytes"); return false;
        }
        if (*ptr) { MTP_GPU_CHECK(gpuFree(*ptr)); *ptr = nullptr; *capacity = 0; }
        MTP_GPU_CHECK(gpuMalloc(reinterpret_cast<void **>(ptr), replacement));
        *capacity = needed;
        ++state_->runtime_stats.buffer_reallocations;
        return true;
    };
    if (request.memory_budget_bytes && memory_stats().tracked_device_bytes > request.memory_budget_bytes) {
        if (!release_center_workspace()) return false;
        if (memory_stats().tracked_device_bytes > request.memory_budget_bytes) {
            set_error(error, "retained fixed GPU allocations exceed memory_budget_bytes; use a fresh context or a larger budget");
            return false;
        }
    }

    if (rebuild_topology) {
        if (state_->input_transfer_stream == nullptr) {
            MTP_GPU_CHECK(gpuStreamCreateWithFlags(
                &state_->input_transfer_stream, gpuStreamNonBlocking));
            MTP_GPU_CHECK(gpuEventCreateWithFlags(
                &state_->input_ready_event, gpuEventDisableTiming));
            MTP_GPU_CHECK(gpuEventCreateWithFlags(
                &state_->active_input_safe_event, gpuEventDisableTiming));
            MTP_GPU_CHECK(gpuEventCreateWithFlags(
                &state_->staging_input_safe_event, gpuEventDisableTiming));
        }

        // The previous loss/gradient work has been enqueued before the next
        // compute call. Record its completion before this call queues new
        // kernels, so the old active slot can safely become a future staging
        // slot without forcing the host to wait now.
        if (state_->active_input_used) {
            MTP_GPU_CHECK(gpuEventRecord(state_->active_input_safe_event, nullptr));
        }

        const int neighbor_values = requested_neighbors;
        const int vector_values = request.neighbor_vectors != nullptr
            ? neighbor_values * 3 : 0;
        const bool staging_grows =
            request.inum > state_->staging_capacity_centers ||
            request.inum + 1 > state_->staging_capacity_offsets ||
            neighbor_values > state_->staging_capacity_neighbors ||
            vector_values > state_->staging_capacity_neighbor_vectors ||
            request.ntotal > state_->staging_capacity_type ||
            request.ntotal * 3 > state_->staging_capacity_x;
        if (state_->staging_input_used && staging_grows) {
            MTP_GPU_CHECK(gpuEventSynchronize(
                state_->staging_input_safe_event));
        }
        if (!reserve_call(&state_->staging_d_centers,
                             &state_->staging_capacity_centers,
                             request.inum, error) ||
            !reserve_call(&state_->staging_d_offsets,
                             &state_->staging_capacity_offsets,
                             request.inum + 1, error) ||
            !reserve_call(&state_->staging_d_neighbors,
                             &state_->staging_capacity_neighbors,
                             neighbor_values, error) ||
            !reserve_call(&state_->staging_d_neighbor_vectors,
                             &state_->staging_capacity_neighbor_vectors,
                             vector_values, error) ||
            !reserve_call(&state_->staging_d_type,
                             &state_->staging_capacity_type,
                             request.ntotal, error) ||
            !reserve_call(&state_->staging_d_x,
                             &state_->staging_capacity_x,
                             request.ntotal * 3, error)) {
            return false;
        }
        if (state_->staging_input_used) {
            MTP_GPU_CHECK(gpuStreamWaitEvent(
                state_->input_transfer_stream,
                state_->staging_input_safe_event, 0));
        }
        {
            NvtxRange topology_upload_range("topology_upload");
            if (request.inum > 0) {
                MTP_GPU_CHECK(gpuMemcpyAsync(
                    state_->staging_d_centers, center_data,
                    request.inum * sizeof(int), gpuMemcpyHostToDevice,
                    state_->input_transfer_stream));
            }
            MTP_GPU_CHECK(gpuMemcpyAsync(
                state_->staging_d_offsets, offset_data,
                (static_cast<size_t>(request.inum) + 1) * sizeof(int), gpuMemcpyHostToDevice,
                state_->input_transfer_stream));
            if (requested_neighbors > 0) {
                MTP_GPU_CHECK(gpuMemcpyAsync(
                    state_->staging_d_neighbors, neighbor_data,
                    static_cast<size_t>(requested_neighbors) * sizeof(int), gpuMemcpyHostToDevice,
                    state_->input_transfer_stream));
            }
        }
        {
            NvtxRange coordinates_upload_range("coordinates_upload");
            if (request.neighbor_vectors != nullptr && neighbor_values > 0) {
                MTP_GPU_CHECK(gpuMemcpyAsync(
                    state_->staging_d_neighbor_vectors,
                    request.neighbor_vectors,
                    vector_values * sizeof(double), gpuMemcpyHostToDevice,
                    state_->input_transfer_stream));
            }
            if (request.ntotal > 0) {
                MTP_GPU_CHECK(gpuMemcpyAsync(
                    state_->staging_d_type, request.type,
                    request.ntotal * sizeof(int), gpuMemcpyHostToDevice,
                    state_->input_transfer_stream));
                MTP_GPU_CHECK(gpuMemcpyAsync(
                    state_->staging_d_x, request.x[0],
                    request.ntotal * 3 * sizeof(double),
                    gpuMemcpyHostToDevice, state_->input_transfer_stream));
            }
        }
        MTP_GPU_CHECK(gpuEventRecord(
            state_->input_ready_event, state_->input_transfer_stream));
        MTP_GPU_CHECK(gpuStreamWaitEvent(nullptr, state_->input_ready_event, 0));

        std::swap(state_->d_centers, state_->staging_d_centers);
        std::swap(state_->d_offsets, state_->staging_d_offsets);
        std::swap(state_->d_neighbors, state_->staging_d_neighbors);
        std::swap(state_->d_neighbor_vectors,
                  state_->staging_d_neighbor_vectors);
        std::swap(state_->d_type, state_->staging_d_type);
        std::swap(state_->d_x, state_->staging_d_x);
        std::swap(state_->capacity_centers,
                  state_->staging_capacity_centers);
        std::swap(state_->capacity_offsets,
                  state_->staging_capacity_offsets);
        std::swap(state_->capacity_neighbors,
                  state_->staging_capacity_neighbors);
        std::swap(state_->capacity_neighbor_vectors,
                  state_->staging_capacity_neighbor_vectors);
        std::swap(state_->capacity_type, state_->staging_capacity_type);
        std::swap(state_->capacity_x, state_->staging_capacity_x);
        std::swap(state_->active_input_safe_event,
                  state_->staging_input_safe_event);
        std::swap(state_->active_input_used,
                  state_->staging_input_used);
        state_->active_input_used = true;
        state_->topology_valid = true;
        state_->topology_inum = request.inum;
        state_->topology_ntotal = request.ntotal;
        state_->topology_neighbors = neighbor_values;
        ++state_->topology_generation;
        ++state_->runtime_stats.topology_uploads;
        ++state_->runtime_stats.pipelined_input_uploads;
        state_->runtime_stats.topology_upload_bytes +=
            (static_cast<std::uint64_t>(request.inum) * 2 + 1 + requested_neighbors) * sizeof(int);
        if (request.ntotal > 0) {
            ++state_->runtime_stats.type_uploads;
            state_->runtime_stats.type_upload_bytes +=
                static_cast<std::uint64_t>(request.ntotal) * sizeof(int);
            ++state_->runtime_stats.coordinate_uploads;
            state_->runtime_stats.coordinate_upload_bytes +=
                static_cast<std::uint64_t>(request.ntotal) * 3 * sizeof(double) +
                static_cast<std::uint64_t>(vector_values) * sizeof(double);
        }
    } else {
        ++state_->runtime_stats.topology_reuses;
        if (!reserve_call(&state_->d_neighbor_vectors,
                             &state_->capacity_neighbor_vectors,
                             request.neighbor_vectors != nullptr
                                 ? state_->topology_neighbors * 3 : 0,
                             error) ||
            !reserve_call(&state_->d_type, &state_->capacity_type,
                             request.ntotal, error) ||
            !reserve_call(&state_->d_x, &state_->capacity_x,
                             request.ntotal * 3, error)) {
            return false;
        }
        if (!reuse_input && request.neighbor_vectors != nullptr &&
            state_->topology_neighbors > 0) {
            NvtxRange coordinates_upload_range("coordinates_upload");
            MTP_GPU_CHECK(gpuMemcpy(
                state_->d_neighbor_vectors, request.neighbor_vectors,
                state_->topology_neighbors * 3 * sizeof(double),
                gpuMemcpyHostToDevice));
        }
        if (!reuse_input && request.ntotal > 0) {
            NvtxRange coordinates_upload_range("coordinates_upload");
            const bool reuse_types = batch && batch->topology_includes_types &&
                topology_token.valid() && topology_token == state_->topology_token;
            if (!reuse_types) {
                MTP_GPU_CHECK(gpuMemcpy(state_->d_type, request.type,
                                         request.ntotal * sizeof(int), gpuMemcpyHostToDevice));
                ++state_->runtime_stats.type_uploads;
                state_->runtime_stats.type_upload_bytes +=
                    static_cast<std::uint64_t>(request.ntotal) * sizeof(int);
            }
            MTP_GPU_CHECK(gpuMemcpy(state_->d_x, request.x[0],
                                     request.ntotal * 3 * sizeof(double),
                                     gpuMemcpyHostToDevice));
            ++state_->runtime_stats.coordinate_uploads;
            state_->runtime_stats.coordinate_upload_bytes +=
                static_cast<std::uint64_t>(request.ntotal) * 3 * sizeof(double) +
                (request.neighbor_vectors != nullptr
                    ? static_cast<std::uint64_t>(state_->topology_neighbors) *
                          3 * sizeof(double)
                    : 0);
        }
    }

    state_->pot.center_energy_weights = nullptr;
    if (request.center_energy_weights != nullptr) {
        for (int atom = 0; atom < request.ntotal; ++atom) {
            if (!std::isfinite(request.center_energy_weights[atom])) {
                set_error(error, "center energy weights must be finite");
                return false;
            }
        }
        if (!reserve_call(&state_->d_center_energy_weights,
                             &state_->capacity_center_energy_weights,
                             request.ntotal, error)) return false;
        MTP_GPU_CHECK(gpuMemcpy(state_->d_center_energy_weights,
            request.center_energy_weights, request.ntotal * sizeof(double),
            gpuMemcpyHostToDevice));
        state_->pot.center_energy_weights = state_->d_center_energy_weights;
    }

    const bool needs_site_energy_buffer = state_->sog || request.compute_site_energy ||
        (request.deterministic_reduction && request.compute_global_energy);
    if (!reserve_call(&state_->d_force, &state_->capacity_force,
                         request.compute_forces ? request.ntotal * 3 : 0, error) ||
        !reserve_call(&state_->d_site_energy, &state_->capacity_site_energy,
                         needs_site_energy_buffer ? request.nlocal : 0, error) ||
        !reserve_call(&state_->d_site_virial, &state_->capacity_site_virial,
                         need_site_virial ? request.ntotal * 6 : 0, error)) {
        return false;
    }
    if (request.compute_global_energy && state_->d_total_energy == nullptr) {
        size_t capacity = 0;
        if (!reserve_call(&state_->d_total_energy, &capacity, 1, error)) return false;
    }
    if (request.global_virial && state_->d_global_virial == nullptr) {
        size_t capacity = 0;
        if (!reserve_call(&state_->d_global_virial, &capacity, 6, error)) return false;
    }
    state_->pot.global_virial = request.global_virial && !reduce_global_virial
        ? state_->d_global_virial : nullptr;
    if (request.global_virial) MTP_GPU_CHECK(gpuMemset(state_->d_global_virial, 0, 6 * sizeof(double)));

    const int neural_cache_stride = mtpm_neural_cache_stride(state_->pot);
    const bool need_neural_cache = need_derivatives || !request.deterministic_reduction;
    int chunk_centers = request.centers_per_chunk > 0
        ? std::min(request.centers_per_chunk, request.inum) : request.inum;
    if (request.memory_budget_bytes && request.inum > 0) {
        const auto stats = memory_stats();
        const size_t reusable = stats.moment_workspace_bytes + stats.adjoint_workspace_bytes + stats.neural_cache_bytes;
        // Reserve room for both topology/input slots from the first step;
        // otherwise the next normal neighbor rebuild could exhaust the cap.
        const auto missing = [](size_t need, size_t capacity) { return need > capacity ? need - capacity : 0; };
        const size_t future_staging = sizeof(int) *
            (missing(request.inum, state_->staging_capacity_centers) +
             missing(static_cast<size_t>(request.inum) + 1, state_->staging_capacity_offsets) +
             missing(requested_neighbors, state_->staging_capacity_neighbors) +
             missing(request.ntotal, state_->staging_capacity_type)) + sizeof(double) *
            (missing(static_cast<size_t>(request.ntotal) * 3, state_->staging_capacity_x) +
             missing(request.neighbor_vectors ? static_cast<size_t>(requested_neighbors) * 3 : 0,
                     state_->staging_capacity_neighbor_vectors));
        const size_t fixed = stats.tracked_device_bytes - reusable + future_staging;
        const size_t per_center = state_->pot.alpha_moments_count *
            ((fp32_moments ? sizeof(float) : sizeof(double)) +
             (need_derivatives ? (fp32_adjoint ? sizeof(float) : sizeof(double)) : 0)) +
            (need_neural_cache ? static_cast<size_t>(neural_cache_stride) * sizeof(double) : 0);
        if (fixed >= request.memory_budget_bytes || !per_center ||
            (request.memory_budget_bytes - fixed) / per_center == 0) {
            set_error(error, "memory_budget_bytes cannot fit fixed buffers and one center"); return false;
        }
        chunk_centers = static_cast<int>(std::min<size_t>(chunk_centers,
            (request.memory_budget_bytes - fixed) / per_center));
    }
    const size_t moment_slots = static_cast<size_t>(chunk_centers) * state_->pot.alpha_moments_count;
    const long long neural_cache_count_ll = need_neural_cache
        ? static_cast<long long>(chunk_centers) * neural_cache_stride : 0;
    if (neural_cache_count_ll > std::numeric_limits<int>::max()) {
        set_error(error, "GPU MTP neural cache exceeds 32-bit allocation indexing"); return false;
    }
    const int neural_cache_count = static_cast<int>(neural_cache_count_ll);
    state_->runtime_stats.last_centers_per_chunk = chunk_centers;
    state_->runtime_stats.last_chunk_count = chunk_centers ? (request.inum - 1) / chunk_centers + 1 : 0;
    reserving_centers = true;
    if (request.inference_only && (request.memory_budget_bytes || request.centers_per_chunk)) {
        // Trim all oversize center buffers before growing any one of them,
        // so changing output flags cannot create a transient budget excess.
        const auto trim = [&](auto **ptr, size_t *capacity, size_t need) -> bool {
            if (*capacity > need) {
                if (*ptr) MTP_GPU_CHECK(gpuFree(*ptr));
                *ptr = nullptr; *capacity = 0;
            }
            return true;
        };
        if (!trim(&state_->d_moment_vals, &state_->capacity_moment_vals, fp32_moments ? 0 : moment_slots) ||
            !trim(&state_->d_moment_vals_mixed, &state_->capacity_moment_vals_mixed, fp32_moments ? moment_slots : 0) ||
            !trim(&state_->d_dmom, &state_->capacity_dmom, need_derivatives && !fp32_adjoint ? moment_slots : 0) ||
            !trim(&state_->d_dmom_mixed, &state_->capacity_dmom_mixed, need_derivatives && fp32_adjoint ? moment_slots : 0) ||
            !trim(&state_->d_neural_cache, &state_->capacity_neural_cache, neural_cache_count)) return false;
    }
    if (fp32_moments) {
        if (!reserve_call(&state_->d_moment_vals_mixed,
                             &state_->capacity_moment_vals_mixed,
                             moment_slots, error)) {
            return false;
        }
    } else if (!reserve_call(&state_->d_moment_vals,
                                &state_->capacity_moment_vals,
                                moment_slots, error)) {
        return false;
    }
    if (fp32_adjoint) {
        if (!reserve_call(&state_->d_dmom_mixed,
                             &state_->capacity_dmom_mixed,
                             need_derivatives ? moment_slots : 0,
                             error)) {
            return false;
        }
    } else if (!reserve_call(&state_->d_dmom, &state_->capacity_dmom,
                                need_derivatives ? moment_slots : 0,
                                error)) {
        return false;
    }
    if (!reserve_call(&state_->d_neural_cache,
                         &state_->capacity_neural_cache,
                         neural_cache_count, error)) {
        return false;
    }
    // The GPU core produces a force delta.  Keeping the existing LAMMPS
    // force array on the host avoids an otherwise unnecessary H2D copy and is
    // also correct when this style participates in a hybrid pair potential.
    if (request.compute_forces && request.ntotal > 0) {
        MTP_GPU_CHECK(gpuMemset(state_->d_force, 0,
                                request.ntotal * 3 * sizeof(double)));
    }
    if (needs_site_energy_buffer && request.nlocal > 0) {
        MTP_GPU_CHECK(gpuMemset(state_->d_site_energy, 0,
                                request.nlocal * sizeof(double)));
    }
    if (need_site_virial && request.ntotal > 0) {
        MTP_GPU_CHECK(gpuMemset(state_->d_site_virial, 0,
                                request.ntotal * 6 * sizeof(double)));
    }
    if (request.compute_global_energy) {
        MTP_GPU_CHECK(gpuMemset(state_->d_total_energy, 0, sizeof(double)));
    }
    double *const device_site_energy =
        needs_site_energy_buffer ? state_->d_site_energy : nullptr;
    double *const device_total_energy =
        request.compute_global_energy ? state_->d_total_energy : nullptr;

    const auto finish_sog = [&]() -> bool {
        if(!state_->sog)return true;
        const int n=request.inum,c=state_->sog_channels;
        sog_latents_kernel<<<n,128,128*sizeof(double)>>>(state_->pot,n,c,state_->d_centers,state_->d_type,
            state_->d_moment_vals,state_->d_sog_coeffs,state_->sog->latents(),false);
        MTP_GPU_CHECK(gpuGetLastError());
        if(!state_->sog->forward(state_->d_x,device_site_energy,
            request.compute_forces?state_->d_force:nullptr,
            request.compute_forces&&need_site_virial?state_->d_site_virial:nullptr,error))return false;
        if(request.compute_forces) {
            sog_seed_kernel<<<n,128>>>(state_->pot,n,c,state_->d_sog_coeffs,state_->sog->phi(),state_->d_dmom);
            if(request.deterministic_reduction)
                propagate_ordered_kernel<double,double><<<n,1>>>(state_->pot,state_->d_centers,state_->d_type,
                    state_->d_moment_vals,state_->d_dmom,nullptr,nullptr,2);
            else propagate_kernel<double,double><<<n,request.propagate_threads,request.propagate_threads*sizeof(double)>>>(
                state_->pot,state_->d_centers,state_->d_type,state_->d_moment_vals,state_->d_dmom,nullptr,nullptr,nullptr,2);
        }
        if(request.compute_global_energy)
            total_energy_ordered_kernel<<<1,1>>>(n,state_->d_centers,device_site_energy,device_total_energy);
        MTP_GPU_CHECK(gpuGetLastError());return true;
    };

    for (int first_center = 0; first_center < request.inum; first_center += chunk_centers) {
        const int active_centers = std::min(chunk_centers, request.inum - first_center);
        const int *chunk_center_ids = state_->d_centers + first_center;
        const int *chunk_offsets = state_->d_offsets + first_center;
        ++state_->runtime_stats.chunks_executed;
    if (fp32_moments) {
        MTP_GPU_CHECK(gpuMemset(state_->d_moment_vals_mixed, 0,
                                moment_slots * sizeof(float)));
    } else {
        MTP_GPU_CHECK(gpuMemset(state_->d_moment_vals, 0,
                                moment_slots * sizeof(double)));
    }
    if (need_derivatives) {
        if (fp32_adjoint) {
            MTP_GPU_CHECK(gpuMemset(state_->d_dmom_mixed, 0,
                                    moment_slots * sizeof(float)));
        } else {
            MTP_GPU_CHECK(gpuMemset(state_->d_dmom, 0,
                                    moment_slots * sizeof(double)));
        }
    }

        const int basic_tiles =
            (state_->pot.alpha_index_basic_count + basic_tile - 1) / basic_tile;
        if (basic_tiles > props.maxGridSize[1]) {
            set_error(error, "number of basic-moment tiles exceeds GPU grid-y limit");
            return false;
        }
        const dim3 grid_basic(active_centers, basic_tiles);
        const dim3 block_pair(request.neighbor_lanes, request.basic_lanes);
        {
            NvtxRange basic_moments_range("basic_moments");
            if (fp32_moments) {
                basic_moments_kernel<float><<<grid_basic, block_pair, basic_shmem>>>(
                    state_->pot, chunk_center_ids, chunk_offsets,
                    state_->d_neighbors,
                    request.neighbor_vectors != nullptr
                        ? state_->d_neighbor_vectors : nullptr,
                    state_->d_type, state_->d_x,
                    state_->d_moment_vals_mixed, basic_tile);
            } else {
                basic_moments_kernel<double><<<grid_basic, block_pair, basic_shmem>>>(
                    state_->pot, chunk_center_ids, chunk_offsets,
                    state_->d_neighbors,
                    request.neighbor_vectors != nullptr
                        ? state_->d_neighbor_vectors : nullptr,
                    state_->d_type, state_->d_x,
                    state_->d_moment_vals, basic_tile);
            }
            MTP_GPU_CHECK(gpuGetLastError());
        }

        if (!need_derivatives) {
            NvtxRange model_forward_seed_range("model_forward_seed");
            if (request.deterministic_reduction) {
                if (fp32_moments) {
                    propagate_energy_ordered_kernel<float><<<active_centers, 1>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals_mixed, device_site_energy);
                } else {
                    propagate_energy_ordered_kernel<double><<<active_centers, 1>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals, device_site_energy);
                }
                MTP_GPU_CHECK(gpuGetLastError());
            } else {
                if (fp32_moments) {
                    propagate_energy_kernel<float><<<
                        active_centers, request.propagate_threads,
                        request.propagate_threads * sizeof(double)>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals_mixed, device_site_energy,
                        device_total_energy, nullptr);
                } else {
                    propagate_energy_kernel<double><<<
                        active_centers, request.propagate_threads,
                        request.propagate_threads * sizeof(double)>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals, device_site_energy,
                        device_total_energy,
                        neural_cache_count > 0 ? state_->d_neural_cache : nullptr);
                }
                MTP_GPU_CHECK(gpuGetLastError());
            }
            if(!finish_sog())return false;
        } else if (request.deterministic_reduction) {
            {
                NvtxRange model_forward_seed_range("model_forward_seed");
                NvtxRange moment_reverse_range("moment_reverse");
                if (fp32_adjoint) {
                // With one thread per center, the grouped propagation
                // schedule is repeatable while accumulating each dependency
                // group in double before its single FP32 adjoint write.  The
                // generic ordered kernel writes after every term, which
                // needlessly amplifies FP32-adjoint rounding.  Its atomic
                // energy total is overwritten below by the ordered reduction.
                propagate_kernel<float, float><<<
                    active_centers, 1, sizeof(double)>>>(
                    state_->pot, chunk_center_ids, state_->d_type,
                    state_->d_moment_vals_mixed, state_->d_dmom_mixed,
                    device_site_energy, device_total_energy, nullptr);
                } else if (fp32_moments) {
                    propagate_ordered_kernel<float, double><<<active_centers, 1>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals_mixed, state_->d_dmom,
                        device_site_energy, nullptr);
                } else {
                    propagate_ordered_kernel<double, double><<<active_centers, 1>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals, state_->d_dmom,
                        device_site_energy,
                        neural_cache_count > 0
                            ? state_->d_neural_cache : nullptr, state_->sog ? 1 : 0);
                }
                MTP_GPU_CHECK(gpuGetLastError());
            }

            if(!finish_sog())return false;

            // One neighbor lane and one block at a time give force/virial
            // atomics a fixed center, tile, and edge order.
            const dim3 ordered_pair(1, request.basic_lanes);
            const size_t ordered_force_shmem = force_shared_bytes(
                state_->pot, 1, request.basic_lanes);
            NvtxRange direct_force_range("direct_force");
            const auto launch_ordered_force = [&](
                const auto *adjoint, auto site_virial_tag,
                int center, int tile) {
                using WorkReal = std::remove_cv_t<
                    std::remove_pointer_t<decltype(adjoint)>>;
                constexpr bool compute_site_virial =
                    decltype(site_virial_tag)::value;
                direct_force_kernel<WorkReal, compute_site_virial><<<
                    1, ordered_pair, ordered_force_shmem>>>(
                    state_->pot, chunk_center_ids, chunk_offsets,
                    state_->d_neighbors,
                    request.neighbor_vectors != nullptr
                        ? state_->d_neighbor_vectors : nullptr,
                    state_->d_type, state_->d_x, adjoint,
                    request.compute_forces ? state_->d_force : nullptr,
                    need_site_virial ? state_->d_site_virial : nullptr,
                    basic_tile, center, tile);
            };
            for (int center = 0; center < active_centers; ++center) {
                for (int tile = 0; tile < basic_tiles; ++tile) {
                    if (need_virial) {
                        if (fp32_adjoint) {
                            launch_ordered_force(
                                state_->d_dmom_mixed, std::true_type{},
                                center, tile);
                        } else {
                            launch_ordered_force(
                                state_->d_dmom, std::true_type{},
                                center, tile);
                        }
                    } else {
                        if (fp32_adjoint) {
                            launch_ordered_force(
                                state_->d_dmom_mixed, std::false_type{},
                                center, tile);
                        } else {
                            launch_ordered_force(
                                state_->d_dmom, std::false_type{},
                                center, tile);
                        }
                    }
                    MTP_GPU_CHECK(gpuGetLastError());
                    ++state_->runtime_stats.basic_tile_force_launches;
                }
            }
        } else {
            {
                NvtxRange model_forward_seed_range("model_forward_seed");
                NvtxRange moment_reverse_range("moment_reverse");
                if (fp32_adjoint) {
                propagate_kernel<float, float><<<
                    active_centers, request.propagate_threads,
                    request.propagate_threads * sizeof(double)>>>(
                    state_->pot, chunk_center_ids, state_->d_type,
                    state_->d_moment_vals_mixed, state_->d_dmom_mixed,
                    device_site_energy, device_total_energy, nullptr);
                } else if (fp32_moments) {
                    propagate_kernel<float, double><<<
                        active_centers, request.propagate_threads,
                        request.propagate_threads * sizeof(double)>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals_mixed, state_->d_dmom,
                        device_site_energy, device_total_energy, nullptr);
                } else {
#if MTP_GPU_PROPAGATION_VARIANT > 0
                    if (state_->pot.model_type == static_cast<int>(MTPGpuModel::Linear)) {
                        propagate_kernel<double, double, MTP_GPU_PROPAGATION_VARIANT <= 2,
                            MTP_GPU_PROPAGATION_VARIANT >= 2><<<
                            active_centers, request.propagate_threads,
                            request.propagate_threads * sizeof(double)>>>(
                            state_->pot, chunk_center_ids, state_->d_type,
                            state_->d_moment_vals, state_->d_dmom,
                            device_site_energy, device_total_energy, nullptr, state_->sog ? 1 : 0);
                    } else
#endif
#if MTP_GPU_PROPAGATION_VARIANT == 1 || MTP_GPU_PROPAGATION_VARIANT == 2
                    if (request.inference_only) {
                        // These per-level partitions are already resident and
                        // included in the potential memory budget. Reuse them
                        // for MTP_M inference without rescanning every group
                        // in both the short and warp loops. Training keeps the
                        // unpartitioned path and its original cache semantics.
                        propagate_kernel<double, double, true><<<
                            active_centers, request.propagate_threads,
                            request.propagate_threads * sizeof(double)>>>(
                            state_->pot, chunk_center_ids, state_->d_type,
                            state_->d_moment_vals, state_->d_dmom,
                            device_site_energy, device_total_energy,
                            neural_cache_count > 0
                                ? state_->d_neural_cache : nullptr, state_->sog ? 1 : 0);
                    } else
#endif
                    if (!request.inference_only &&
                        (state_->pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN1) ||
                         state_->pot.model_type == static_cast<int>(MTPGpuModel::MTPMNN2))) {
                        propagate_kernel<double, double, false, false, true><<<
                            active_centers, request.propagate_threads,
                            request.propagate_threads * sizeof(double)>>>(
                            state_->pot, chunk_center_ids, state_->d_type,
                            state_->d_moment_vals, state_->d_dmom,
                            device_site_energy, device_total_energy,
                            state_->d_neural_cache, state_->sog ? 1 : 0);
                    } else {
                    propagate_kernel<double, double><<<
                        active_centers, request.propagate_threads,
                        request.propagate_threads * sizeof(double)>>>(
                        state_->pot, chunk_center_ids, state_->d_type,
                        state_->d_moment_vals, state_->d_dmom,
                        device_site_energy, device_total_energy,
                        neural_cache_count > 0
                            ? state_->d_neural_cache : nullptr, state_->sog ? 1 : 0);
                    }
                }
                MTP_GPU_CHECK(gpuGetLastError());
            }

            if(!finish_sog())return false;

            const bool auto_center_neighbors =
                request.force_kernel == MTPGpuForceKernel::Auto &&
                active_centers >= 4 * props.multiProcessorCount &&
                state_->pot.alpha_index_basic_count > basic_tile;
            const MTPGpuForceKernel selected_force_kernel =
                request.force_kernel == MTPGpuForceKernel::CenterNeighbors ||
                    auto_center_neighbors
                    ? MTPGpuForceKernel::CenterNeighbors
                    : MTPGpuForceKernel::BasicTile;
#if defined(IMRMLP_CUDA)
            const bool use_warp_reduction = request.inference_only &&
                !fp32_moments && !fp32_adjoint &&
                state_->pot.model_type != static_cast<int>(MTPGpuModel::Linear) &&
                request.neighbor_lanes == 8 && request.basic_lanes == 16;
#else
            // Keep HIP on its existing reduction until separately profiled.
            constexpr bool use_warp_reduction = false;
#endif
            NvtxRange direct_force_range("direct_force");
            const auto launch_center_force = [&](
                const auto *adjoint, auto site_virial_tag) {
                using WorkReal = std::remove_cv_t<
                    std::remove_pointer_t<decltype(adjoint)>>;
                constexpr bool compute_site_virial =
                    decltype(site_virial_tag)::value;
                const auto launch = [&](auto reduction_tag) {
                    constexpr bool warp_reduction = decltype(reduction_tag)::value &&
                        std::is_same_v<WorkReal, double>;
                    center_neighbors_force_kernel<
                        WorkReal, compute_site_virial, warp_reduction><<<
                        dim3(active_centers), block_pair, force_shmem>>>(
                        state_->pot, chunk_center_ids, chunk_offsets,
                        state_->d_neighbors,
                        request.neighbor_vectors != nullptr
                            ? state_->d_neighbor_vectors : nullptr,
                        state_->d_type, state_->d_x, adjoint,
                        request.compute_forces ? state_->d_force : nullptr,
                        need_site_virial ? state_->d_site_virial : nullptr);
                };
                if (use_warp_reduction) launch(std::true_type{});
                else launch(std::false_type{});
            };
            const auto launch_basic_force = [&](
                const auto *adjoint, auto site_virial_tag) {
                using WorkReal = std::remove_cv_t<
                    std::remove_pointer_t<decltype(adjoint)>>;
                constexpr bool compute_site_virial =
                    decltype(site_virial_tag)::value;
                const auto launch = [&](auto reduction_tag) {
                    constexpr bool warp_reduction = decltype(reduction_tag)::value &&
                        std::is_same_v<WorkReal, double>;
                    direct_force_kernel<WorkReal, compute_site_virial, warp_reduction><<<
                        grid_basic, block_pair, force_shmem>>>(
                        state_->pot, chunk_center_ids, chunk_offsets,
                        state_->d_neighbors,
                        request.neighbor_vectors != nullptr
                            ? state_->d_neighbor_vectors : nullptr,
                        state_->d_type, state_->d_x, adjoint,
                        request.compute_forces ? state_->d_force : nullptr,
                        need_site_virial ? state_->d_site_virial : nullptr,
                        basic_tile, 0, 0);
                };
                if (use_warp_reduction) launch(std::true_type{});
                else launch(std::false_type{});
            };
            if (selected_force_kernel ==
                MTPGpuForceKernel::CenterNeighbors) {
                if (need_virial) {
                    if (fp32_adjoint) {
                        launch_center_force(
                            state_->d_dmom_mixed, std::true_type{});
                    } else {
                        launch_center_force(
                            state_->d_dmom, std::true_type{});
                    }
                } else {
                    if (fp32_adjoint) {
                        launch_center_force(
                            state_->d_dmom_mixed, std::false_type{});
                    } else {
                        launch_center_force(
                            state_->d_dmom, std::false_type{});
                    }
                }
                ++state_->runtime_stats.center_neighbors_force_launches;
            } else {
                if (need_virial) {
                    if (fp32_adjoint) {
                        launch_basic_force(
                            state_->d_dmom_mixed, std::true_type{});
                    } else {
                        launch_basic_force(
                            state_->d_dmom, std::true_type{});
                    }
                } else {
                    if (fp32_adjoint) {
                        launch_basic_force(
                            state_->d_dmom_mixed, std::false_type{});
                    } else {
                        launch_basic_force(
                            state_->d_dmom, std::false_type{});
                    }
                }
                ++state_->runtime_stats.basic_tile_force_launches;
            }
            MTP_GPU_CHECK(gpuGetLastError());
        }
    }

    if (reduce_global_virial) {
        global_virial_kernel<<<6, 128>>>(request.ntotal, state_->d_site_virial, state_->d_global_virial);
        MTP_GPU_CHECK(gpuGetLastError());
    }
    if (request.deterministic_reduction && request.compute_global_energy && request.inum > 0) {
        total_energy_ordered_kernel<<<1, 1>>>(request.inum, state_->d_centers,
            device_site_energy, device_total_energy);
        MTP_GPU_CHECK(gpuGetLastError());
    }
    if (request.memory_budget_bytes && memory_stats().tracked_device_bytes > request.memory_budget_bytes) {
        set_error(error, "GPU memory budget exceeded"); return false;
    }

    if (request.copy_outputs_to_host) {
        NvtxRange output_download_range("output_download");
        if (request.global_virial) {
            MTP_GPU_CHECK(gpuMemcpy(request.global_virial, state_->d_global_virial,
                6 * sizeof(double), gpuMemcpyDeviceToHost));
            state_->runtime_stats.output_download_bytes += 6 * sizeof(double);
        }
        state_->runtime_stats.output_download_bytes += sizeof(double) *
            ((request.compute_global_energy ? size_t{1} : 0) +
             (request.compute_forces && request.copy_forces_to_host ? static_cast<size_t>(request.ntotal) * 3 : 0) +
             (request.compute_site_energy && request.site_energy ? static_cast<size_t>(request.nlocal) : 0) +
             (request.site_virial ? static_cast<size_t>(request.ntotal) * 6 : 0));
        if (request.compute_global_energy) {
            MTP_GPU_CHECK(gpuMemcpy(request.total_energy, state_->d_total_energy,
                                     sizeof(double), gpuMemcpyDeviceToHost));
        }
        if (request.compute_forces && request.copy_forces_to_host && request.ntotal > 0) {
            const int force_values = request.ntotal * 3;
            if (request.accumulate_force) {
                state_->host_force_delta.resize(force_values);
                MTP_GPU_CHECK(gpuMemcpy(state_->host_force_delta.data(), state_->d_force,
                                         force_values * sizeof(double), gpuMemcpyDeviceToHost));
                for (int i = 0; i < force_values; ++i)
                    request.force[0][i] += state_->host_force_delta[i];
            } else {
                MTP_GPU_CHECK(gpuMemcpy(request.force[0], state_->d_force,
                                         force_values * sizeof(double), gpuMemcpyDeviceToHost));
            }
        }
        if (request.compute_site_energy && request.site_energy != nullptr && request.nlocal > 0) {
            MTP_GPU_CHECK(gpuMemcpy(request.site_energy, state_->d_site_energy,
                                     request.nlocal * sizeof(double), gpuMemcpyDeviceToHost));
        }
        if (request.site_virial != nullptr && request.ntotal > 0) {
            MTP_GPU_CHECK(gpuMemcpy(request.site_virial[0], state_->d_site_virial,
                                     request.ntotal * 6 * sizeof(double),
                                     gpuMemcpyDeviceToHost));
        }
    }
    if (rebuild_topology) {
        MTP_GPU_CHECK(gpuStreamSynchronize(state_->input_transfer_stream));
    }
    if (reuse_input) ++state_->runtime_stats.input_reuses;
    state_->topology_token = topology_token;
    state_->input_token = input_token;
    state_->input_has_vectors = request.neighbor_vectors != nullptr;
    guard.success = true;
    // Training loss consumes the resident force buffer. A virial-only call has
    // adjoints, but deliberately leaves that buffer absent or stale.
    state_->forward_valid = request.compute_forces && !request.inference_only;
    state_->forward_inum = request.inum;
    state_->forward_ntotal = request.ntotal;
    state_->forward_neighbors = requested_neighbors;
    state_->forward_has_neighbor_vectors = request.neighbor_vectors != nullptr;
    state_->forward_weighted = request.center_energy_weights != nullptr;
    state_->forward_deterministic_reduction =
        request.deterministic_reduction;
    return true;
}

} // namespace MLIPCore::GpuDirect
