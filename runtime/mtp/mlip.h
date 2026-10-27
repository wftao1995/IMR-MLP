// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include <cstddef>
#include <string>
#include <vector>
/**
 * @brief 
 * TODO: memalign, in LAMMPS, memory->create() will create alligned memory thus it is fast
 */

// TODO: spline support
namespace MLIPCore
{
    constexpr int mlip_g_s = 10;
    constexpr int align = 64;
    constexpr int mlip_neighbor_index_mask = 0x3fffffff;

    struct alpha_basic_t {
        int mu, b1, b2, b3;
    };
    struct alpha_times_t {
        int l1,l2,c,id;
    };
    struct coordpow_t {
        double x, y, z;
    };

    struct mlip_buffers {
        double *moment_vals = nullptr;
        double *site_energy_ders_wrt_moments_ = nullptr;
        double *rb_vals = nullptr;
        double *rb_ders = nullptr;
        double *val_mu = nullptr;
        double *der_mu = nullptr;
        double *dist_powers = nullptr;
        coordpow_t *coords_powers = nullptr;
        int jnum_old = 0;
        int *jlist_true = nullptr;
        double *buff_site_energy_ders = nullptr;
        double *moment_jacobian = nullptr;
    };

    // setups 
    struct mlip {
        enum site_model_type {
            MODEL_LINEAR = 0,
            MODEL_MTPM_Q2 = 1,
            MODEL_MTPM_FS = 2,
            MODEL_MTPM_NN1 = 3,
            MODEL_MTPM_NN2 = 4,
            MODEL_MTPM_LI = 5,
        };
        enum activation_type {
            ACTIVATION_TANH = 0,
            ACTIVATION_RELU = 1,
            ACTIVATION_SIGMOID = 2,
            ACTIVATION_LINEAR = 3,
        };

        int alpha_moments_count;
        int alpha_index_basic_count;
        int alpha_index_times_count;
        int alpha_scalar_moments;
        int radial_func_count;
        int rb_size;
        int species_count;
        int radial_coeffs_count, moment_coeffs_count, linear_coeffs_count, coeffs_count;
        double max_dist;
        double min_dist;
        double scaling = 1.0;
        int max_mu;
        int max_alpha_index_basic;
        double *regression_coeffs = nullptr;
        double *radial_coeffs;
        double *species_coeffs;
        double *moment_coeffs;
        // MTP-M stores one site-energy offset per species first,
        // followed by per-descriptor species biases, descriptor-to-moment
        // weights, and the external-model coefficients.  Classic MTP leaves
        // these fields at their defaults and uses moment_coeffs directly.
        site_model_type site_model = MODEL_LINEAR;
        activation_type model_activation = ACTIVATION_TANH;
        int descriptor_count = 0;
        int model_hidden1 = 0;
        int model_hidden2 = 0;
        int descriptor_species_coeffs_count = 0;
        int descriptor_moment_coeffs_count = 0;
        int model_coeffs_count = 0;
        double *descriptor_species_coeffs = nullptr;
        double *descriptor_moment_coeffs = nullptr;
        double *model_coeffs = nullptr;
        alpha_basic_t *alpha_index_basic = nullptr;
        alpha_times_t *alpha_index_times = nullptr;
        int *nu_arr_list = nullptr;
        int *alpha_moment_mapping = nullptr;
        int spmap[mlip_g_s]; // species mapping
        // The fixed array remains for legacy callers constructing a view by
        // hand. Loaded models use a mapping sized to the actual atom types.
        std::vector<int> type_mapping;
        const int *mapping_data() const {
            return type_mapping.empty() ? spmap : type_mapping.data();
        }
        std::size_t mapping_size() const {
            return type_mapping.empty() ? mlip_g_s : type_mapping.size();
        }
        // for radial basis
        double cutoff, cutoffsq, sub_r, plus_r, mult, rmult;
        double irpi, p0, pp;
        enum rb_type {
            RB_CHEBVHAR,
            RB_CHEBVCOS,
            RB_CHEBYPOLY5,
            RB_CHEBYCONFIGURABLE,
            RB_CHEBYPOLY4,
            RB_POLYB,
            RB_POLYBCOS,
            RB_LEGACY_I2,
            RB_LEGACY_POLY2,
            RB_LEGACY_POLY3,
        } rb;
        enum rb_cutoff_type {
            RB_CUTOFF_QUADRATIC,
            RB_CUTOFF_COSINE,
            RB_CUTOFF_POLY5,
            RB_CUTOFF_NORMALIZED_QUADRATIC,
            RB_CUTOFF_COSINE_LEFT,
            RB_CUTOFF_COSINE_UNSCALED,
            RB_CUTOFF_COSINE_SQUARED,
            RB_CUTOFF_SINE_SQUARED,
            RB_CUTOFF_POLY3,
            RB_CUTOFF_POLY4,
            RB_CUTOFF_SMOOTHSTEP5,
            RB_CUTOFF_SMOOTHSTEP7,
        };
        enum rb_transform_type {
            RB_TRANSFORM_LINEAR,
            RB_TRANSFORM_EXPONENTIAL,
            RB_TRANSFORM_EXPONENTIAL_LEFT,
            RB_TRANSFORM_I2,
            RB_TRANSFORM_EXPONENTIAL_TANH,
        };
        struct radial_basis_t {
            rb_type rb = RB_CHEBVHAR;
            rb_cutoff_type cutoff = RB_CUTOFF_QUADRATIC;
            rb_transform_type transform = RB_TRANSFORM_LINEAR;
            int size = 0;
            double min_dist = 0.0;
            double max_dist = 0.0;
            double scaling = 1.0;
            // Prepared at load time; zero permits programmatically built bases
            // to use the evaluator's fallback. Immutable with the interval.
            double inverse_range = 0.0, cosine_frequency = 0.0;
            double cutoff_parameter = 0.0;
            double transform_parameter = 0.0;
            bool one_minus_chebyshev = false;
            double cutoff_r1 = 0.0, cutoff_r2 = 0.0;
            double transform_scale_parameter = 0.0;
            // Owned, column-major PolyB data; immutable during evaluation.
            std::vector<double> polynomial_matrix, polynomial_normalizations;
        };
        // Split/type-pair bases, non-split configurable bases, and modern
        // parameterized Poly5 bases are normalized to species_count^2
        // descriptors. Legacy single-basis potentials keep this empty.
        int pair_radial_basis_count = 0;
        radial_basis_t *pair_radial_bases = nullptr;
        void (*rbcalc)(const mlip *, mlip_buffers *, const double &);
        // level specific
        int max_level;
        int *lev_scalar_count = nullptr;
        int *lev_scalar_nodes = nullptr;
        int *lev_scalar_accum = nullptr;
        int *lev_times_count = nullptr;
        int *lev_times_nodes = nullptr;

        bool have_order_info = false;
    };

    mlip_buffers* create_mlip_buffer(const mlip *);
    void extent_mlip_buffer(const mlip *, mlip_buffers *, int);
    void destroy_mlip_buffer(mlip_buffers *);
    // Throws std::runtime_error on malformed or unsupported input; the caller
    // owns MPI error propagation. No process exit occurs inside the reader.
    mlip *setup(const char *mlip_fnm, const int *sp, double *cutoff,
                std::size_t mapping_count = mlip_g_s);
    std::string read_model_text(const char *filename);
    mlip *setup_from_text(const std::string &text, const int *sp, double *cutoff,
                         std::size_t mapping_count = mlip_g_s);
    void destroy_mlip(mlip *);

    void compute(const mlip *,
                 mlip_buffers*,
                 const int *ilist,
                 const int *numneigh,
                 const int inum,
                 const int *type,
                 double **x,
                 int **firstneigh,
                 double *en,
                 double *p_site_en,
                 double **p_site_virial,
                 double **f);

    // radial basis
    void RB_Calc(const mlip *, mlip_buffers *, const double &r);
    void RB_CalcDer(const mlip *, mlip_buffers *, const double &r);
    void RB_CalcDerCos(const mlip *, mlip_buffers *, const double &r);
    void RB_CalcDerPoly5(const mlip *, mlip_buffers *, const double &r);
    void RB_CalcDerPair(const mlip *, const mlip::radial_basis_t *,
                        mlip_buffers *, const double &r);
}
// THIS IS ONLY A TEST
