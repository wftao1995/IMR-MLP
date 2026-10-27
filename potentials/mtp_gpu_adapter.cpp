// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtp_gpu_adapter.h"
#include "potential_mtp_m.h"
#include "potential_mtp_LES.h"
#include "potential_mtpnn1.h"

#ifdef IMRMLP_GPU

#include <algorithm>
#include <limits>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <typeinfo>

namespace Potentials {
namespace {

using MLIPCore::GpuDirect::MTPGpuPotentialView;
using MLIPCore::GpuDirect::MTPGpuRadialBasisView;
using MLIPCore::GpuDirect::MTPGpuInitializationOptions;
using MLIPCore::GpuDirect::MTPGpuPrecision;
using MLIPCore::GpuDirect::MTPGpuActivation;
using MLIPCore::GpuDirect::MTPGpuModel;
using MLIPCore::GpuDirect::RadialBasis;
using MLIPCore::GpuDirect::RadialCutoff;
using MLIPCore::GpuDirect::RadialTransform;

bool set_error(std::string *error, const std::string &message)
{
    if (error != nullptr) *error = message;
    return false;
}

MTPGpuActivation gpu_activation(m_model_nn1::ActivationFunction activation)
{
    switch (activation) {
    case m_model_nn1::TANH: return MTPGpuActivation::Tanh;
    case m_model_nn1::RELU: return MTPGpuActivation::ReLU;
    case m_model_nn1::SIGMOID: return MTPGpuActivation::Sigmoid;
    case m_model_nn1::LINEAR: return MTPGpuActivation::Linear;
    }
    return MTPGpuActivation::Linear;
}

MTPGpuActivation gpu_activation(IMRMLPNN::ActivationFunction activation)
{
    switch (activation) {
    case IMRMLPNN::TANH: return MTPGpuActivation::Tanh;
    case IMRMLPNN::RELU: return MTPGpuActivation::ReLU;
    case IMRMLPNN::SIGMOID: return MTPGpuActivation::Sigmoid;
    }
    return MTPGpuActivation::Tanh;
}

// Private formula cutoff/transform classes expose their exact parameters through
// SaveToStream. Read that portable representation once at initialization; never
// sample or approximate the basis and never serialize on the evaluation path.
std::map<std::string, double> radial_parameters(const AnyBasis &basis)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10);
    const_cast<AnyBasis &>(basis).SaveToStream(output);
    std::map<std::string, double> values;
    std::istringstream input(output.str());
    std::string line;
    while (std::getline(input, line)) {
        std::istringstream record(line);
        std::string key, equals;
        double value;
        if (record >> key >> equals >> value && equals == "=") values[key] = value;
    }
    return values;
}

bool radial_basis_for(const AnyBasis &source, MTPGpuRadialBasisView *target,
                      std::string *error)
{
    *target = {};
    target->min_dist = source.min_val;
    target->max_dist = source.max_val;
    target->scaling = source.scaling;
    switch (basis_name_to_code(source.name)) {
    case RB_Chebyshev:
        target->radial_basis = RadialBasis::Chebyshev; return true;
    case RB_ChebyCos:
        target->radial_basis = RadialBasis::ChebyCos; return true;
    case RB_ChebyshevPoly4: {
        const auto *basis = dynamic_cast<const RBasis_ChebyshevPoly4 *>(&source);
        if (!basis) return set_error(error, "Invalid RBChebyshevPoly4 type");
        target->radial_basis = RadialBasis::ChebyshevPoly4;
        target->cutoff_parameter = basis->b; return true;
    }
    case RB_ChebyshevPoly5: {
        const auto *basis = dynamic_cast<const RBasis_ChebyshevPoly5 *>(&source);
        if (!basis) return set_error(error, "Invalid RBChebyshevPoly5 type");
        target->radial_basis = RadialBasis::ChebyshevPoly5;
        target->cutoff_parameter = basis->b; return true;
    }
    case RB_PolyB: {
        const auto *basis = dynamic_cast<const RBasis_PolyB *>(&source);
        if (!basis) return set_error(error, "Invalid RBPolyB type");
        target->radial_basis = RadialBasis::PolyB;
        target->polynomial_matrix = basis->coeffs.data();
        target->polynomial_normalizations = basis->invNalpha.data(); return true;
    }
    case RB_PolyBCos: {
        const auto *basis = dynamic_cast<const RBasis_PolyBCos *>(&source);
        if (!basis) return set_error(error, "Invalid RBPolyBCos type");
        target->radial_basis = RadialBasis::PolyBCos;
        target->polynomial_matrix = basis->coeffs.data();
        target->polynomial_normalizations = basis->invNalpha.data(); return true;
    }
    case RB_ChebyshevConfigurable: {
        const auto *basis = dynamic_cast<const RBasis_ChebyshevConfigurable *>(&source);
        if (!basis) return set_error(error, "Invalid RBChebyshevConfigurable type");
        target->radial_basis = RadialBasis::ChebyshevConfigurable;
        target->one_minus_chebyshev = basis->basis_function_type() == BasisFunctionType::OneMinusChebyshev;
        switch (basis->cutoff_function().type()) {
#define IMRMLP_GPU_CUTOFF(name) case Cutoff::CutoffType::name: target->cutoff = RadialCutoff::name; break;
        IMRMLP_GPU_CUTOFF(Quadratic)
        IMRMLP_GPU_CUTOFF(NormalizedQuadratic)
        IMRMLP_GPU_CUTOFF(Cosine)
        IMRMLP_GPU_CUTOFF(CosineLeft)
        IMRMLP_GPU_CUTOFF(CosineUnscaled)
        IMRMLP_GPU_CUTOFF(CosineSquared)
        IMRMLP_GPU_CUTOFF(SineSquared)
        IMRMLP_GPU_CUTOFF(Poly3)
        IMRMLP_GPU_CUTOFF(Poly4)
        IMRMLP_GPU_CUTOFF(Poly5)
        IMRMLP_GPU_CUTOFF(Smoothstep5)
        IMRMLP_GPU_CUTOFF(Smoothstep7)
#undef IMRMLP_GPU_CUTOFF
        }
        switch (basis->shift_function().type()) {
#define IMRMLP_GPU_TRANSFORM(name) case XTransForm::ShiftType::name: target->transform = RadialTransform::name; break;
        IMRMLP_GPU_TRANSFORM(Linear)
        IMRMLP_GPU_TRANSFORM(Exponential)
        IMRMLP_GPU_TRANSFORM(ExponentialLeft)
        IMRMLP_GPU_TRANSFORM(I2)
        IMRMLP_GPU_TRANSFORM(ExponentialTanh)
#undef IMRMLP_GPU_TRANSFORM
        }
        auto parameters = radial_parameters(source);
        target->cutoff_parameter = parameters["cutoff_b"];
        target->cutoff_r1 = parameters["cutoff_r1"];
        target->cutoff_r2 = parameters["cutoff_r2"];
        target->transform_parameter = parameters["shift_lambda"];
        target->transform_scale_parameter = parameters["shift_scale_lambda"];
        return true;
    }
    default:
        return set_error(error, "GPU MTP does not support radial basis '" + source.name + "'");
    }
}

} // namespace

bool MTPGpuEvaluator::initialize(const AnyPotential &source_potential,
                                   int device_id,
                                   std::string *error)
{
    return initialize(source_potential, device_id,
                      MTPGpuPrecision::FP64, error);
}

bool MTPGpuEvaluator::initialize(const AnyPotential &source_potential,
                                   int device_id,
                                   MTPGpuPrecision precision,
                                   std::string *error)
{
    return initialize_impl(source_potential, device_id, precision, false, error);
}

bool MTPGpuEvaluator::initialize_short_range(const AnyPotential &source_potential,
                                             int device_id, std::string *error)
{
    return initialize_short_range(source_potential, device_id, MTPGpuPrecision::FP64, error);
}

bool MTPGpuEvaluator::initialize_short_range(const AnyPotential &source_potential,
                                             int device_id, MTPGpuPrecision precision,
                                             std::string *error)
{
    return initialize_impl(source_potential, device_id, precision, true, error);
}

bool MTPGpuEvaluator::initialize_impl(const AnyPotential &source_potential,
                                      int device_id, MTPGpuPrecision precision,
                                      bool short_range_only, std::string *error,
                                      const MTPGpuPotentialView *head_override)
{
    context_.reset();
    short_range_only_ = short_range_only;
    custom_head_ = head_override != nullptr;
    const auto *potential_ptr = dynamic_cast<const PotMTP *>(&source_potential);
    if (potential_ptr == nullptr) {
        return set_error(error, "GPU MTP adapter requires PotMTP or PotMTP_M");
    }
    const PotMTP &potential = *potential_ptr;
    MTPGpuModel model_type = MTPGpuModel::Linear;
    MTPGpuActivation model_activation = MTPGpuActivation::Tanh;
    int model_hidden1 = 0;
    int model_hidden2 = 0;
    int descriptor_count = 0;
    int model_coeffs_count = 0;
    const double *descriptor_species_coeffs = nullptr;
    const double *descriptor_moment_coeffs = nullptr;
    const double *model_coeffs = nullptr;
    const auto *sog_source = dynamic_cast<const PotMTP_LES *>(&source_potential);
    if (!short_range_only_ && sog_source && (!sog_source->uses_sog() || !sog_source->lr_les))
        return set_error(error,"GPU PotMTP_LES requires an active SOG kernel");
    if (!head_override && typeid(source_potential) != typeid(PotMTP) && !sog_source) {
        if (const auto *linear = dynamic_cast<const PotMTP_M<m_model_mtpli> *>(&source_potential)) {
            if (!short_range_only_ && (linear->lr_les || linear->req_les) && !linear->uses_sog())
                return set_error(error, "GPU MTP-M supports SOG long range; Coulomb LES is unsupported");
            model_type = MTPGpuModel::MTPMLinear;
            descriptor_count = linear->n_props;
            model_coeffs_count = linear->model_coeff_size;
            descriptor_species_coeffs = linear->species_coeffs + linear->species_count;
            descriptor_moment_coeffs = linear->moment_coeffs;
            model_coeffs = linear->model_coeffs;
        } else if (const auto *quadratic = dynamic_cast<
                const PotMTP_M<m_model_mtpq2> *>(&source_potential)) {
            if (!short_range_only_ && (quadratic->lr_les || quadratic->req_les) && !quadratic->uses_sog()) {
                return set_error(error, "GPU MTP-M supports SOG long range; Coulomb LES is unsupported");
            }
            model_type = MTPGpuModel::MTPMQuadratic;
            descriptor_count = quadratic->n_props;
            model_coeffs_count = quadratic->model_coeff_size;
            descriptor_species_coeffs =
                quadratic->species_coeffs + quadratic->species_count;
            descriptor_moment_coeffs = quadratic->moment_coeffs;
            model_coeffs = quadratic->model_coeffs;
        } else if (const auto *asinh_model = dynamic_cast<
                       const PotMTP_M<m_model_mtpfs> *>(&source_potential)) {
            if (!short_range_only_ && (asinh_model->lr_les || asinh_model->req_les) && !asinh_model->uses_sog()) {
                return set_error(error, "GPU MTP-M supports SOG long range; Coulomb LES is unsupported");
            }
            model_type = MTPGpuModel::MTPMAsinh;
            descriptor_count = asinh_model->n_props;
            model_coeffs_count = asinh_model->model_coeff_size;
            descriptor_species_coeffs =
                asinh_model->species_coeffs + asinh_model->species_count;
            descriptor_moment_coeffs = asinh_model->moment_coeffs;
            model_coeffs = asinh_model->model_coeffs;
        } else if (const auto *nn1 = dynamic_cast<
                       const PotMTP_M<m_model_nn1> *>(&source_potential)) {
            if (!short_range_only_ && (nn1->lr_les || nn1->req_les) && !nn1->uses_sog()) {
                return set_error(error, "GPU MTP-M supports SOG long range; Coulomb LES is unsupported");
            }
            model_type = MTPGpuModel::MTPMNN1;
            model_activation = gpu_activation(nn1->model.acfunc_enum);
            model_hidden1 = nn1->model.n_neuron;
            descriptor_count = nn1->n_props;
            model_coeffs_count = nn1->model_coeff_size;
            descriptor_species_coeffs =
                nn1->species_coeffs + nn1->species_count;
            descriptor_moment_coeffs = nn1->moment_coeffs;
            model_coeffs = nn1->model_coeffs;
        } else if (const auto *nn2 = dynamic_cast<
                       const PotMTP_M<m_model_nn2> *>(&source_potential)) {
            if (!short_range_only_ && (nn2->lr_les || nn2->req_les) && !nn2->uses_sog()) {
                return set_error(error, "GPU MTP-M supports SOG long range; Coulomb LES is unsupported");
            }
            model_type = MTPGpuModel::MTPMNN2;
            model_activation = gpu_activation(nn2->model.nn.acfunc);
            model_hidden1 = nn2->model.n_neuron;
            model_hidden2 = nn2->model.n_neuron2;
            descriptor_count = nn2->n_props;
            model_coeffs_count = nn2->model_coeff_size;
            descriptor_species_coeffs =
                nn2->species_coeffs + nn2->species_count;
            descriptor_moment_coeffs = nn2->moment_coeffs;
            model_coeffs = nn2->model_coeffs;
        } else {
            return set_error(error,
                "GPU supports mtp_m_li, mtp_m_q2, mtp_m_fs, mtp_m_n1, and mtp_m_n2; Coulomb LES and other derived PotMTP models remain unsupported");
        }
    }
    const bool has_split_basis = potential.radial_basis_split &&
        potential.p_radical_basis_pair.size() ==
            static_cast<size_t>(potential.species_count * potential.species_count) &&
        std::all_of(potential.p_radical_basis_pair.begin(),
                    potential.p_radical_basis_pair.end(),
                    [](const AnyBasis *basis) { return basis != nullptr; });
    if (!potential.inited() ||
        (!potential.radial_basis_split && potential.p_radical_basis == nullptr) ||
        (potential.radial_basis_split && !has_split_basis)) {
        return set_error(error, "cannot initialize GPU from an empty MTP potential");
    }
    // radial_mode controls native training regularization, not inference.
    inference_only_radial_mode_ = potential.radial_mode != 0;

    const AnyBasis &first_basis = potential.radial_basis_split
        ? *potential.p_radical_basis_pair.front()
        : *potential.p_radical_basis;
    MTPGpuRadialBasisView first_descriptor;
    if (!radial_basis_for(first_basis, &first_descriptor, error)) return false;

    std::vector<MTPGpuRadialBasisView> pair_radial_bases;
    if (potential.radial_basis_split ||
        (first_descriptor.radial_basis != RadialBasis::Chebyshev &&
         first_descriptor.radial_basis != RadialBasis::ChebyCos)) {
        const size_t pair_count = static_cast<size_t>(potential.species_count) *
            static_cast<size_t>(potential.species_count);
        pair_radial_bases.assign(pair_count, first_descriptor);
    }
    if (potential.radial_basis_split) {
        pair_radial_bases.resize(potential.p_radical_basis_pair.size());
        for (size_t pair = 0; pair < pair_radial_bases.size(); ++pair) {
            const AnyBasis &source = *potential.p_radical_basis_pair[pair];
            auto &target = pair_radial_bases[pair];
            if (!radial_basis_for(source, &target, error)) {
                if (error != nullptr && !error->empty()) {
                    *error += " for ordered species pair " +
                        std::to_string(pair / potential.species_count) + "-" +
                        std::to_string(pair % potential.species_count);
                }
                return false;
            }
        }
    }

    species_map_.resize(potential.species_count);
    std::iota(species_map_.begin(), species_map_.end(), 0);

    MTPGpuPotentialView view;
    view.species_count = potential.species_count;
    view.max_type = potential.species_count;
    view.spmap = species_map_.data();
    view.alpha_moments_count = potential.alpha_moments_count;
    view.alpha_index_basic_count = potential.alpha_index_basic_count;
    view.alpha_index_times_count = potential.alpha_index_times_count;
    view.alpha_scalar_moments = potential.alpha_scalar_moments;
    view.radial_func_count = potential.radial_func_count;
    view.rb_size = potential.rb_size;
    view.max_mu = potential.max_mu;
    view.max_alpha_index_basic = potential.max_alpha_index_basic;
    view.min_dist = first_basis.min_val;
    view.max_dist = first_basis.max_val;
    view.cutoff = potential.cutoff;
    view.scaling = first_basis.scaling;
    view.radial_basis = first_descriptor.radial_basis;
    if (!pair_radial_bases.empty()) {
        view.pair_radial_bases = pair_radial_bases.data();
        view.pair_radial_basis_count =
            static_cast<int>(pair_radial_bases.size());
    }
    view.radial_coeffs = potential.radial_coeffs;
    view.radial_coeffs_count = potential.radial_coeff_size;
    view.species_coeffs = potential.species_coeffs;
    view.moment_coeffs = model_type == MTPGpuModel::Linear
        ? potential.moment_coeffs : nullptr;
    view.model = model_type;
    view.descriptor_count = descriptor_count;
    view.descriptor_species_coeffs = descriptor_species_coeffs;
    view.descriptor_moment_coeffs = descriptor_moment_coeffs;
    view.model_coeffs = model_coeffs;
    view.model_coeffs_count = model_coeffs_count;
    view.model_activation = model_activation;
    view.model_hidden1 = model_hidden1;
    view.model_hidden2 = model_hidden2;
    view.alpha_index_basic = potential.alpha_index_basic.data();
    view.alpha_index_times = potential.alpha_index_times.data();
    view.alpha_moment_mapping = potential.alpha_moment_mapping;
    view.nu_arr_list = potential.nu_arr_list;

    if (head_override) {
        view.model = head_override->model;
        view.species_coeffs = head_override->species_coeffs;
        view.moment_coeffs = head_override->moment_coeffs;
        view.descriptor_count = head_override->descriptor_count;
        view.descriptor_species_coeffs = head_override->descriptor_species_coeffs;
        view.descriptor_moment_coeffs = head_override->descriptor_moment_coeffs;
        view.model_coeffs = head_override->model_coeffs;
        view.model_coeffs_count = head_override->model_coeffs_count;
        view.model_activation = head_override->model_activation;
        view.model_hidden1 = head_override->model_hidden1;
        view.model_hidden2 = head_override->model_hidden2;
    }

    auto new_context = std::make_unique<MLIPCore::GpuDirect::MTPGpuDirect>();
    MTPGpuInitializationOptions options;
    options.precision = precision;
    if (!new_context->set_device(device_id, error) ||
        !new_context->initialize(view, options, error)) {
        return false;
    }
    context_ = std::move(new_context);
    has_sog_ = false;
    if(!short_range_only_ && !update_sog(source_potential,error)){context_.reset();return false;}
    configure_gradient_layout(source_potential);
    return true;
}

bool MTPGpuEvaluator::initialize_linear_head(const PotMTP &potential,
                                              const double *species, const double *moments,
                                              int device_id, std::string *error)
{
    if (!species || !moments) return set_error(error, "linear GPU head requires species and moment coefficients");
    MTPGpuPotentialView head;
    head.model = MTPGpuModel::Linear;
    head.species_coeffs = species;
    head.moment_coeffs = moments;
    return initialize_impl(potential, device_id, MTPGpuPrecision::FP64, true, error, &head);
}

bool MTPGpuEvaluator::initialize_legacy_nn(const PotMTPNN1 &potential,
                                            int device_id, std::string *error)
{
    if (!potential.p_mtp || (potential.nnlay > 1 ? !potential.p_nn2 : !potential.p_nn))
        return set_error(error, "legacy neural GPU model is not initialized");
    const PotMTP &mtp = *potential.p_mtp;
    const int inputs = mtp.alpha_scalar_moments + 1;
    const auto &mean = potential.nnlay > 1 ? potential.p_nn2->norm_mean : potential.p_nn->norm_mean;
    const auto &inverse_std = potential.nnlay > 1 ? potential.p_nn2->norm_irstd : potential.p_nn->norm_irstd;
    const int nn_inputs = potential.nnlay > 1 ? potential.p_nn2->n_input : potential.p_nn->n_input;
    if (nn_inputs != inputs || mean.size() != inputs || inverse_std.size() != inputs ||
        potential.species_coeffs.size() != static_cast<size_t>(mtp.species_count))
        return set_error(error, "legacy neural GPU descriptor or species layout mismatch");
    std::vector<double> species(static_cast<size_t>(inputs) * mtp.species_count);
    std::vector<double> moments(static_cast<size_t>(inputs) * mtp.alpha_scalar_moments, 0.);
    for (int descriptor = 0; descriptor < inputs; ++descriptor) {
        for (int type = 0; type < mtp.species_count; ++type) {
            const double raw_bias = descriptor == 0 ? mtp.species_coeffs[type] : 0.;
            species[descriptor * mtp.species_count + type] =
                (raw_bias - mean[descriptor]) * inverse_std[descriptor];
        }
        if (descriptor > 0)
            moments[descriptor * mtp.alpha_scalar_moments + descriptor - 1] = inverse_std[descriptor];
    }
    MTPGpuPotentialView head;
    head.model = potential.nnlay > 1 ? MTPGpuModel::MTPMNN2 : MTPGpuModel::MTPMNN1;
    head.species_coeffs = potential.species_coeffs.data();
    head.descriptor_count = inputs;
    head.descriptor_species_coeffs = species.data();
    head.descriptor_moment_coeffs = moments.data();
    head.model_coeffs_count = potential.nnlay > 1 ? potential.p_nn2->n_coeffs : potential.p_nn->n_coeffs;
    std::vector<double> coefficients(head.model_coeffs_count);
    if (potential.nnlay > 1) {
        potential.p_nn2->get_coeffs(coefficients.data());
        head.model_hidden1 = potential.p_nn2->n_neuron1;
        head.model_hidden2 = potential.p_nn2->n_neuron2;
        head.model_activation = gpu_activation(potential.p_nn2->acfunc);
    } else {
        // The legacy NN1 serializes [input weights, hidden biases, output
        // weights]; MTP-M NN1 uses [input weights, output weights, biases].
        potential.p_nn->get_coeffs(coefficients.data());
        head.model_hidden1 = potential.p_nn->n_neuron;
        const int offset = inputs * head.model_hidden1;
        for (int hidden = 0; hidden < head.model_hidden1; ++hidden)
            std::swap(coefficients[offset + hidden], coefficients[offset + head.model_hidden1 + hidden]);
        head.model_activation = gpu_activation(potential.p_nn->acfunc);
    }
    head.model_coeffs = coefficients.data();
    return initialize_impl(mtp, device_id, MTPGpuPrecision::FP64, true, error, &head);
}

// The device always differentiates the full physical model. MTP-M exposes
// a compact optimizer vector when req_* groups are frozen; gather only those
// slots on host, while preserving full device accumulation across chunks.
void MTPGpuEvaluator::configure_gradient_layout(const AnyPotential &source)
{
    gradient_indices_.clear(); full_gradient_count_ = 0;
    const auto add = [&](int count, bool enabled) {
        if (enabled) for (int i=0;i<count;++i)
            gradient_indices_.push_back(full_gradient_count_+i);
        full_gradient_count_ += count;
    };
    const auto mtpm = [&](const auto *p) {
        if (!p) return false;
        add(p->radial_coeff_size,p->req_radical);
        add(p->species_count,p->req_species);
        add(p->species_coeff_size-p->species_count+p->moment_coeff_size,p->req_moment);
        add(p->model_coeff_size,p->req_model);
        if (has_sog_) {
            add(p->les_channels()*p->species_count,p->req_les);
            add(p->les_channels()*p->alpha_scalar_moments,p->req_les);
            add(2*p->model_les.sog_weights.size(),p->req_sog);
        }
        return true;
    };
    if (!mtpm(dynamic_cast<const PotMTP_M<m_model_mtpli> *>(&source)) &&
        !mtpm(dynamic_cast<const PotMTP_M<m_model_mtpq2> *>(&source)) &&
        !mtpm(dynamic_cast<const PotMTP_M<m_model_mtpfs> *>(&source)) &&
        !mtpm(dynamic_cast<const PotMTP_M<m_model_nn1> *>(&source)) &&
        !mtpm(dynamic_cast<const PotMTP_M<m_model_nn2> *>(&source)))
        add(source.get_NL_coeffs_count(),true);
}

bool MTPGpuEvaluator::accumulate_gradient(MLIPCore::GpuDirect::MTPGpuGradientRequest request, std::string *error)
{
    if (short_range_only_) return set_error(error, "short-range-only GPU models are inference-only");
    if (!context_) return set_error(error,"GPU MTP evaluator is not initialized");
    if (inference_only_radial_mode_)
        return set_error(error,"GPU training still requires radial_mode=0");
    if (request.gradient_count != static_cast<int>(gradient_indices_.size()) || !request.gradient) {
        context_->invalidate_forward();
        return set_error(error,"GPU gradient does not match the native optimizer layout");
    }
    if (request.gradient_count == full_gradient_count_)
        return context_->accumulate_gradient(request,error);
    double *native = request.gradient;
    full_gradient_.assign(full_gradient_count_,0.);
    if (request.accumulate_gradient && !request.accumulate_gradient_on_device)
        for (size_t i=0;i<gradient_indices_.size();++i) full_gradient_[gradient_indices_[i]]=native[i];
    request.gradient=full_gradient_.data(); request.gradient_count=full_gradient_count_;
    if (!context_->accumulate_gradient(request,error)) return false;
    if (request.copy_gradient_to_host)
        for (size_t i=0;i<gradient_indices_.size();++i) native[i]=full_gradient_[gradient_indices_[i]];
    return true;
}

bool MTPGpuEvaluator::update_sog(const AnyPotential &source,std::string *error)
{
    const double *species=nullptr,*moments=nullptr;
    double lower=0.,upper=0.;
    bool found=false;
    if(const auto *p=dynamic_cast<const PotMTP_LES *>(&source)) {
        if(!p->uses_sog()||!p->lr_les)return set_error(error,"GPU requires an active SOG kernel");
        sog_definition_=p->model_les;
        sog_definition_.kernel=LES::Kernel::SOG;sog_definition_.sog_channels=p->channels();
        sog_definition_.dl=p->dl;sog_definition_.taper_fraction=p->les_taper_fraction;
        sog_definition_.cutoff_mode=LES::parse_cutoff_mode(p->les_cutoff_mode);
        sog_definition_.remove_self_interaction=p->remove_self_interaction;
        species=p->les_species_coeffs;moments=p->les_moment_coeffs;
        lower=p->sog_log_bandwidth_min;upper=p->sog_log_bandwidth_max;found=true;
    }
    const auto mtpm=[&](const auto *p){
        if(!p||!p->uses_sog())return;
        sog_definition_=p->model_les;species=p->les_species;moments=p->les_moment;
        lower=p->sog_log_bandwidth_min;upper=p->sog_log_bandwidth_max;found=true;
    };
    mtpm(dynamic_cast<const PotMTP_M<m_model_mtpli> *>(&source));
    mtpm(dynamic_cast<const PotMTP_M<m_model_mtpq2> *>(&source));
    mtpm(dynamic_cast<const PotMTP_M<m_model_mtpfs> *>(&source));
    mtpm(dynamic_cast<const PotMTP_M<m_model_nn1> *>(&source));
    mtpm(dynamic_cast<const PotMTP_M<m_model_nn2> *>(&source));
    if(!found){if(has_sog_)return set_error(error,"GPU model kind changed; reinitialize the evaluator");return true;}
    if(has_sog_ && (sog_definition_.sog_channels<1))return set_error(error,"Invalid SOG layout");
    sog_definition_.validate();
    const double span=upper-lower,mid=.5*(upper+lower);
    if(!(span>0.)||!std::isfinite(span))return set_error(error,"Invalid SOG bandwidth bounds");
    std::vector<double> jacobian(sog_definition_.sog_weights.size());
    for(size_t i=0;i<jacobian.size();++i){
        const double u=sog_definition_.sog_log_bandwidths[i];
        if(!(u>lower&&u<upper))return set_error(error,"SOG bandwidth is outside its optimizer bounds");
        const double c=std::cos(M_PI*(u-mid)/span);jacobian[i]=span/M_PI*c*c;
    }
    MLIPCore::GpuDirect::MTPGpuSOGView view;
    view.model={sog_definition_.sog_channels,sog_definition_.channel_terms(),sog_definition_.sog_weights.data(),
        sog_definition_.sog_log_bandwidths.data(),sog_definition_.remove_self_interaction};
    view.species_coeffs=species;view.moment_coeffs=moments;view.bandwidth_jacobian=jacobian.data();
    if(!context_->configure_sog(view,error))return false;
    has_sog_=true;return true;
}

size_t MTPGpuEvaluator::reciprocal_bytes(const Atoms &atoms) const
{
    if(!has_sog_)return 0;
    LES grid=sog_definition_;LES::KVectorList waves;grid.generate_k_vectors(atoms,waves,false);
    const long double c=grid.sog_channels,n=atoms.size(),k=waves.size(),m=grid.channel_terms();
    const long double bytes=1.15L*((4*n*c+8*k*c+7*k+2*c*m+c)*sizeof(double)+(n+k+4)*sizeof(int)+1024);
    return bytes>=std::numeric_limits<size_t>::max()?std::numeric_limits<size_t>::max():static_cast<size_t>(std::ceil(bytes));
}

bool MTPGpuEvaluator::update_coefficients(const AnyPotential &source_potential,
                                            std::string *error)
{
    if (!context_) return set_error(error, "GPU MTP evaluator is not initialized");
    if (custom_head_) return set_error(error, "custom GPU heads require explicit reinitialization after coefficient changes");
    context_->invalidate_forward();
    const auto *potential = dynamic_cast<const PotMTP *>(&source_potential);
    if (potential == nullptr)
        return set_error(error, "GPU coefficient update requires PotMTP");
    inference_only_radial_mode_ = potential->radial_mode != 0;
    if(!short_range_only_ && !update_sog(source_potential,error))return false;
    configure_gradient_layout(source_potential);
    if (typeid(source_potential) == typeid(PotMTP) || dynamic_cast<const PotMTP_LES *>(&source_potential)) {
        return context_->update_coefficients(
            potential->radial_coeffs, potential->species_coeffs,
            potential->moment_coeffs, error);
    }
    if (const auto *linear = dynamic_cast<const PotMTP_M<m_model_mtpli> *>(&source_potential)) {
        return context_->update_coefficients(linear->radial_coeffs, linear->species_coeffs, nullptr,
            linear->species_coeffs + linear->species_count, linear->moment_coeffs, linear->model_coeffs, error);
    }
    if (const auto *quadratic = dynamic_cast<
            const PotMTP_M<m_model_mtpq2> *>(&source_potential)) {
        return context_->update_coefficients(
            quadratic->radial_coeffs, quadratic->species_coeffs, nullptr,
            quadratic->species_coeffs + quadratic->species_count,
            quadratic->moment_coeffs, quadratic->model_coeffs, error);
    }
    if (const auto *asinh_model = dynamic_cast<
            const PotMTP_M<m_model_mtpfs> *>(&source_potential)) {
        return context_->update_coefficients(
            asinh_model->radial_coeffs, asinh_model->species_coeffs, nullptr,
            asinh_model->species_coeffs + asinh_model->species_count,
            asinh_model->moment_coeffs, asinh_model->model_coeffs, error);
    }
    if (const auto *nn1 = dynamic_cast<
            const PotMTP_M<m_model_nn1> *>(&source_potential)) {
        return context_->update_coefficients(
            nn1->radial_coeffs, nn1->species_coeffs, nullptr,
            nn1->species_coeffs + nn1->species_count,
            nn1->moment_coeffs, nn1->model_coeffs, error);
    }
    if (const auto *nn2 = dynamic_cast<
            const PotMTP_M<m_model_nn2> *>(&source_potential)) {
        return context_->update_coefficients(
            nn2->radial_coeffs, nn2->species_coeffs, nullptr,
            nn2->species_coeffs + nn2->species_count,
            nn2->moment_coeffs, nn2->model_coeffs, error);
    }
    return set_error(error,
        "GPU coefficient update received an unsupported PotMTP-derived model");
}

bool MTPGpuBatch::build(const Atoms &atoms, const AnyNeighborhoods &neighborhoods,
                          std::string *error)
{
    std::vector<const Atoms *> atom_ptrs{&atoms};
    std::vector<const AnyNeighborhoods *> neighborhood_ptrs{&neighborhoods};
    return build_impl(atom_ptrs, neighborhood_ptrs, error);
}

bool MTPGpuBatch::build(const std::vector<Atoms> &atoms,
                          const std::vector<Neighborhoods> &neighborhoods,
                          std::string *error)
{
    built_ = false;
    if (atoms.size() != neighborhoods.size()) {
        return set_error(error, "GPU MTP batch requires one neighborhood set per structure");
    }
    std::vector<const Atoms *> atom_ptrs(atoms.size());
    std::vector<const AnyNeighborhoods *> neighborhood_ptrs(neighborhoods.size());
    for (size_t i = 0; i < atoms.size(); ++i) {
        atom_ptrs[i] = &atoms[i];
        neighborhood_ptrs[i] = &neighborhoods[i];
    }
    return build_impl(atom_ptrs, neighborhood_ptrs, error);
}

bool MTPGpuBatch::refresh_compatible(
    const Atoms &atoms, const AnyNeighborhoods &neighborhoods,
    std::string *error)
{
    built_ = false;
    ++input_revision_;

    if (structure_count() != 1 || atom_count() != static_cast<int>(atoms.size()) ||
        neighborhoods.size() != static_cast<int>(atoms.size())) {
        return set_error(error,
            "GPU MTP refreshed structure has a different topology shape");
    }
    if (atoms.chemmap_.size() != atoms.size()) {
        return set_error(error, "GPU MTP refreshed structure has no complete type map");
    }
    for (int i = 0; i < atom_count(); ++i) {
        const Neighborhood &nbh = neighborhoods[i];
        if (nbh.nbhcount_ < 0 ||
            offsets_[i + 1] - offsets_[i] != nbh.nbhcount_ ||
            nbh.indices_.size() != static_cast<size_t>(nbh.nbhcount_) ||
            nbh.pos_.size() != static_cast<size_t>(nbh.nbhcount_)) {
            return set_error(error,
                "GPU MTP refreshed structure has different neighbor row sizes");
        }
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            if (neighbors_[offsets_[i] + jj] != nbh.indices_[jj]) {
                return set_error(error,
                    "GPU MTP refreshed structure has different neighbor indices");
            }
        }
    }
    for (int i = 0; i < atom_count(); ++i) {
        types_[i] = atoms.chemmap_[i];
        for (int a = 0; a < 3; ++a) positions_[3 * i + a] = atoms.pos_[i][a];
        const Neighborhood &nbh = neighborhoods[i];
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            const int edge = offsets_[i] + jj;
            for (int a = 0; a < 3; ++a) {
                neighbor_vectors_[3 * edge + a] = nbh.pos_[jj][a];
            }
        }
    }
    if(reciprocal_geometry_[0].cell_!=atoms.cell_ || reciprocal_geometry_[0].pbc_!=atoms.pbc_)grid_valid_=false;
    reciprocal_geometry_[0].cell_=atoms.cell_;reciprocal_geometry_[0].pbc_=atoms.pbc_;
    built_ = true;
    return true;
}

bool MTPGpuBatch::build(const std::vector<Atoms> &atoms,
                          const std::vector<Neighborhoods> &neighborhoods,
                          const std::vector<int> &indices,
                          std::string *error)
{
    built_ = false;
    if (atoms.size() != neighborhoods.size()) {
        return set_error(error, "GPU MTP batch requires one neighborhood set per structure");
    }
    if (indices.empty()) return set_error(error, "GPU MTP selected batch is empty");
    std::vector<const Atoms *> atom_ptrs;
    std::vector<const AnyNeighborhoods *> neighborhood_ptrs;
    atom_ptrs.reserve(indices.size());
    neighborhood_ptrs.reserve(indices.size());
    for (int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= atoms.size()) {
            return set_error(error, "GPU MTP selected batch index is out of range");
        }
        atom_ptrs.push_back(&atoms[index]);
        neighborhood_ptrs.push_back(&neighborhoods[index]);
    }
    return build_impl(atom_ptrs, neighborhood_ptrs, error);
}

bool MTPGpuBatch::build_impl(
    const std::vector<const Atoms *> &atoms,
    const std::vector<const AnyNeighborhoods *> &neighborhoods,
    std::string *error)
{
    built_ = false;
    ++topology_revision_;
    ++grid_revision_;
    ++labels_revision_;
    grid_valid_ = false;
    cached_labels_ = {};

    ++input_revision_;
    uploaded_once_ = false;

    if (atoms.empty()) return set_error(error, "GPU MTP batch is empty");
    structure_offsets_.assign(atoms.size() + 1, 0);
    for (size_t s = 0; s < atoms.size(); ++s) {
        if (atoms[s] == nullptr || neighborhoods[s] == nullptr) {
            return set_error(error, "GPU MTP batch contains a null structure");
        }
        const size_t n = atoms[s]->size();
        if (n > static_cast<size_t>(std::numeric_limits<int>::max()) ||
            structure_offsets_[s] > std::numeric_limits<int>::max() - static_cast<int>(n)) {
            return set_error(error, "GPU MTP batch has too many atoms for 32-bit indexing");
        }
        structure_offsets_[s + 1] = structure_offsets_[s] + static_cast<int>(n);
    }
    reciprocal_geometry_.resize(atoms.size());
    for(size_t b=0;b<atoms.size();++b){reciprocal_geometry_[b].cell_=atoms[b]->cell_;reciprocal_geometry_[b].pbc_=atoms[b]->pbc_;}
    grid_valid_=false;
    const int natoms = structure_offsets_.back();
    centers_.resize(natoms);
    std::iota(centers_.begin(), centers_.end(), 0);
    center_structure_.resize(natoms);
    offsets_.assign(natoms + 1, 0);
    types_.resize(natoms);
    positions_.resize(static_cast<size_t>(natoms) * 3);

    for (size_t s = 0; s < atoms.size(); ++s) {
        const Atoms &structure = *atoms[s];
        const AnyNeighborhoods &nbhs = *neighborhoods[s];
        const int count = structure_offsets_[s + 1] - structure_offsets_[s];
        if (nbhs.size() != count || structure.chemmap_.size() != static_cast<size_t>(count)) {
            return set_error(error, "GPU MTP batch has inconsistent atoms or neighborhoods");
        }
        const int atom_base = structure_offsets_[s];
        for (int i = 0; i < count; ++i) {
            const int global_i = atom_base + i;
            center_structure_[global_i] = static_cast<int>(s);
            types_[global_i] = structure.chemmap_[i];
            for (int a = 0; a < 3; ++a) {
                positions_[3 * global_i + a] = structure.pos_[i][a];
            }
            const Neighborhood &nbh = nbhs[i];
            if (nbh.nbhcount_ < 0 ||
                nbh.indices_.size() != static_cast<size_t>(nbh.nbhcount_) ||
                nbh.pos_.size() != static_cast<size_t>(nbh.nbhcount_)) {
                return set_error(error, "GPU MTP batch contains an inconsistent neighborhood");
            }
            if (nbh.nbhcount_ > std::numeric_limits<int>::max() - offsets_[global_i]) {
                return set_error(error, "GPU MTP batch neighbor count overflows 32-bit indexing");
            }
            offsets_[global_i + 1] = offsets_[global_i] + nbh.nbhcount_;
        }
    }

    neighbors_.resize(offsets_.back());
    neighbor_vectors_.resize(static_cast<size_t>(offsets_.back()) * 3);
    for (size_t s = 0; s < atoms.size(); ++s) {
        const int atom_base = structure_offsets_[s];
        const int count = structure_offsets_[s + 1] - atom_base;
        const AnyNeighborhoods &nbhs = *neighborhoods[s];
        for (int i = 0; i < count; ++i) {
            const int global_i = atom_base + i;
            const Neighborhood &nbh = nbhs[i];
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                const int edge = offsets_[global_i] + jj;
                const int atj = nbh.indices_[jj];
                if (atj < 0 || atj >= count) {
                    return set_error(error, "GPU MTP neighbor index is outside its structure");
                }
                neighbors_[edge] = atom_base + atj;
                for (int a = 0; a < 3; ++a) {
                    neighbor_vectors_[3 * edge + a] = nbh.pos_[jj][a];
                }
            }
        }
    }

    forces_.assign(static_cast<size_t>(natoms) * 3, 0.0);
    site_energies_.assign(natoms, 0.0);
    site_virials_.assign(static_cast<size_t>(natoms) * 6, 0.0);
    uploaded_once_ = false;
    uploaded_generation_ = 0;
    built_ = true;
    return true;
}

bool MTPGpuBatch::can_reuse_topology(
    const MTPGpuEvaluator &evaluator, bool requested) const
{
    return requested && uploaded_once_ &&
        evaluator.context().context_identity() == uploaded_context_ &&
        evaluator.context().topology_generation() == uploaded_generation_;
}

void MTPGpuBatch::record_uploaded_topology(
    const MTPGpuEvaluator &evaluator)
{
    uploaded_context_ = evaluator.context().context_identity();
    uploaded_once_ = true;
    uploaded_generation_ = evaluator.context().topology_generation();
}

bool MTPGpuBatch::prepare_sog(MTPGpuEvaluator &evaluator,
    MLIPCore::GpuDirect::MTPGpuBatchRequest &request,std::string *error)
{
    if (!built_) return set_error(error,"GPU batch requires a successful build or refresh");
    const LES *definition=evaluator.sog_definition();
    if(!definition)return true;
    if(!std::all_of(positions_.begin(),positions_.end(),[](double x){return std::isfinite(x);}))
        return set_error(error,"SOG GPU positions contain non-finite values");
    const bool same=grid_valid_ && grid_definition_.dl==definition->dl
        && grid_definition_.cutoff_mode==definition->cutoff_mode
        && grid_definition_.taper_fraction==definition->taper_fraction
        && grid_definition_.sog_include_zero==definition->sog_include_zero;
    if(!same) {
        grid_valid_=false;
        ++grid_revision_;
        grid_definition_=*definition;
        wave_offsets_.assign(structure_count()+1,0);volumes_.resize(structure_count());
        wave_vectors_.clear();wave_factors_.clear();wave_taper_.clear();wave_tp_.clear();
        for(int b=0;b<structure_count();++b){
            LES::KVectorList waves;grid_definition_.generate_k_vectors(reciprocal_geometry_[b],waves,false);
            if(waves.size()>size_t(std::numeric_limits<int>::max()-wave_offsets_[b]))return set_error(error,"SOG GPU reciprocal grid overflows 32-bit indexing");
            wave_offsets_[b+1]=wave_offsets_[b]+static_cast<int>(waves.size());
            volumes_[b]=reciprocal_geometry_[b].get_volume();
            for(const auto &k:waves)for(int d=0;d<3;++d)wave_vectors_.push_back(k[d]);
            wave_factors_.insert(wave_factors_.end(),grid_definition_.kvec_factors.begin(),grid_definition_.kvec_factors.end());
            const auto &t=grid_definition_.reciprocal_taper(),&tp=grid_definition_.reciprocal_taper_derivative();
            wave_taper_.insert(wave_taper_.end(),t.begin(),t.end());wave_tp_.insert(wave_tp_.end(),tp.begin(),tp.end());
        }
        grid_valid_=true;
    }
    sog_grid_={atom_count(),structure_count(),wave_offsets_.back(),structure_offsets_.data(),wave_offsets_.data(),volumes_.data(),
        wave_vectors_.data(),wave_factors_.data(),wave_taper_.data(),wave_tp_.data()};
    sog_grid_.cache_token = {identity_.value, grid_revision_};
    request.sog_grid=&sog_grid_;return true;
}

MLIPCore::GpuDirect::MTPGpuBatchRequest MTPGpuBatch::request()
{
    MLIPCore::GpuDirect::MTPGpuBatchRequest req;
    if (!built_) return req;
    req.topology_token = {identity_.value, topology_revision_};
    req.input_token = input_token();
    req.centers = centers_.data();
    req.offsets = offsets_.data();
    req.neighbors = neighbors_.empty() ? nullptr : neighbors_.data();
    req.neighbor_vectors = neighbor_vectors_.empty() ? nullptr : neighbor_vectors_.data();
    req.center_count = static_cast<int>(centers_.size());
    req.nlocal = static_cast<int>(types_.size());
    req.ntotal = static_cast<int>(types_.size());
    req.type = types_.data();
    req.x = positions_.data();
    req.force = forces_.data();
    req.site_energy = site_energies_.data();
    req.site_virial = site_virials_.data();
    req.accumulate_force = false;
    req.deterministic_reduction = deterministic_reduction_;
    return req;
}

bool MTPGpuBatch::evaluate(MTPGpuEvaluator &evaluator, Atoms &result,
                             bool reuse_topology, std::string *error)
{
    if (evaluator.initialized()) evaluator.context().invalidate_forward();
    if (!evaluator.initialized()) return set_error(error, "GPU MTP evaluator is not initialized");
    if (result.size() != types_.size()) {
        return set_error(error, "GPU MTP result structure has a different atom count");
    }

    auto req = request();
    if(!prepare_sog(evaluator,req,error))return false;
    double energy = 0.0;
    req.total_energy = &energy;
    req.reuse_topology = can_reuse_topology(evaluator, reuse_topology);
    if (!evaluator.context().compute_batch(req, error)) return false;
    record_uploaded_topology(evaluator);

    result.resetEFS();
    if(evaluator.sog_definition())result.charges_.clear();
    result.energy_ = energy;
    result.site_energies_ = site_energies_;
    for (int i = 0; i < req.ntotal; ++i) {
        for (int a = 0; a < 3; ++a) result.forces_[i][a] = forces_[3 * i + a];
    }
    for (int i = 0; i < req.ntotal; ++i) {
        result.stress_(0, 0) += site_virials_[6 * i + 0];
        result.stress_(1, 1) += site_virials_[6 * i + 1];
        result.stress_(2, 2) += site_virials_[6 * i + 2];
        result.stress_(0, 1) += site_virials_[6 * i + 3];
        result.stress_(0, 2) += site_virials_[6 * i + 4];
        result.stress_(1, 2) += site_virials_[6 * i + 5];
    }
    result.stress_(1, 0) = result.stress_(0, 1);
    result.stress_(2, 0) = result.stress_(0, 2);
    result.stress_(2, 1) = result.stress_(1, 2);
    return true;
}

bool MTPGpuBatch::evaluate(MTPGpuEvaluator &evaluator,
                             std::vector<Atoms> &results,
                             bool reuse_topology, std::string *error)
{
    if (evaluator.initialized()) evaluator.context().invalidate_forward();
    if (results.size() != static_cast<size_t>(structure_count())) {
        return set_error(error, "GPU MTP result batch has a different structure count");
    }
    std::vector<Atoms *> result_ptrs(results.size());
    for (size_t i = 0; i < results.size(); ++i) result_ptrs[i] = &results[i];
    return evaluate_impl(evaluator, result_ptrs, reuse_topology, error);
}

bool MTPGpuBatch::evaluate(MTPGpuEvaluator &evaluator,
                             std::vector<Atoms> &results,
                             const std::vector<int> &indices,
                             bool reuse_topology, std::string *error)
{
    if (evaluator.initialized()) evaluator.context().invalidate_forward();
    if (indices.size() != static_cast<size_t>(structure_count())) {
        return set_error(error, "GPU MTP selected result count differs from the resident batch");
    }
    std::vector<Atoms *> result_ptrs;
    result_ptrs.reserve(indices.size());
    for (int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= results.size()) {
            return set_error(error, "GPU MTP selected result index is out of range");
        }
        result_ptrs.push_back(&results[index]);
    }
    return evaluate_impl(evaluator, result_ptrs, reuse_topology, error);
}

bool MTPGpuBatch::evaluate_impl(MTPGpuEvaluator &evaluator,
                                  const std::vector<Atoms *> &results,
                                  bool reuse_topology, std::string *error)
{
    if (evaluator.initialized()) evaluator.context().invalidate_forward();
    if (!evaluator.initialized()) return set_error(error, "GPU MTP evaluator is not initialized");
    if (results.size() != static_cast<size_t>(structure_count())) {
        return set_error(error, "GPU MTP result batch has a different structure count");
    }
    for (int s = 0; s < structure_count(); ++s) {
        const int expected = structure_offsets_[s + 1] - structure_offsets_[s];
        if (results[s] == nullptr || results[s]->size() != static_cast<size_t>(expected)) {
            return set_error(error, "GPU MTP result batch has a different atom count");
        }
    }

    auto req = request();
    if(!prepare_sog(evaluator,req,error))return false;
    double total_energy = 0.0;
    req.total_energy = &total_energy;
    req.reuse_topology = can_reuse_topology(evaluator, reuse_topology);
    if (!evaluator.context().compute_batch(req, error)) return false;
    record_uploaded_topology(evaluator);

    for (int s = 0; s < structure_count(); ++s) {
        Atoms &result = *results[s];
        result.resetEFS();
    if(evaluator.sog_definition())result.charges_.clear();
        const int begin = structure_offsets_[s];
        const int end = structure_offsets_[s + 1];
        for (int global_i = begin; global_i < end; ++global_i) {
            const int local_i = global_i - begin;
            result.energy_ += site_energies_[global_i];
            result.site_energies_[local_i] = site_energies_[global_i];
            for (int a = 0; a < 3; ++a) {
                result.forces_[local_i][a] = forces_[3 * global_i + a];
            }
            result.stress_(0, 0) += site_virials_[6 * global_i + 0];
            result.stress_(1, 1) += site_virials_[6 * global_i + 1];
            result.stress_(2, 2) += site_virials_[6 * global_i + 2];
            result.stress_(0, 1) += site_virials_[6 * global_i + 3];
            result.stress_(0, 2) += site_virials_[6 * global_i + 4];
            result.stress_(1, 2) += site_virials_[6 * global_i + 5];
        }
        result.stress_(1, 0) = result.stress_(0, 1);
        result.stress_(2, 0) = result.stress_(0, 2);
        result.stress_(2, 1) = result.stress_(1, 2);
    }
    return true;
}

bool MTPGpuBatch::evaluate_energy(MTPGpuEvaluator &evaluator,
                                    Atoms &result, bool reuse_topology,
                                    std::string *error)
{
    std::vector<Atoms *> results{&result};
    return evaluate_energy_impl(evaluator, results, reuse_topology, error);
}

bool MTPGpuBatch::evaluate_energy(MTPGpuEvaluator &evaluator,
                                    std::vector<Atoms> &results,
                                    const std::vector<int> &indices,
                                    bool reuse_topology,
                                    std::string *error)
{
    if (evaluator.initialized()) evaluator.context().invalidate_forward();
    if (indices.size() != static_cast<size_t>(structure_count())) {
        return set_error(error,
            "GPU MTP selected energy result count differs from the resident batch");
    }
    std::vector<Atoms *> result_ptrs;
    result_ptrs.reserve(indices.size());
    for (int index : indices) {
        if (index < 0 || static_cast<size_t>(index) >= results.size()) {
            return set_error(error,
                "GPU MTP selected energy result index is out of range");
        }
        result_ptrs.push_back(&results[index]);
    }
    return evaluate_energy_impl(evaluator, result_ptrs, reuse_topology, error);
}

bool MTPGpuBatch::evaluate_energy_impl(
    MTPGpuEvaluator &evaluator, const std::vector<Atoms *> &results,
    bool reuse_topology, std::string *error)
{
    if (evaluator.initialized()) evaluator.context().invalidate_forward();
    if (!evaluator.initialized()) {
        return set_error(error, "GPU MTP evaluator is not initialized");
    }
    if (results.size() != static_cast<size_t>(structure_count())) {
        return set_error(error,
            "GPU MTP energy result batch has a different structure count");
    }
    for (int s = 0; s < structure_count(); ++s) {
        const int expected = structure_offsets_[s + 1] - structure_offsets_[s];
        if (results[s] == nullptr ||
            results[s]->size() != static_cast<size_t>(expected)) {
            return set_error(error,
                "GPU MTP energy result batch has a different atom count");
        }
    }

    auto req = request();
    if(!prepare_sog(evaluator,req,error))return false;
    double total_energy = 0.0;
    req.total_energy = &total_energy;
    req.force = nullptr;
    req.site_virial = nullptr;
    req.compute_forces = false;
    req.reuse_topology = can_reuse_topology(evaluator, reuse_topology);
    if (!evaluator.context().compute_batch(req, error)) return false;
    record_uploaded_topology(evaluator);

    for (int s = 0; s < structure_count(); ++s) {
        Atoms &result = *results[s];
        result.resetEFS();
    if(evaluator.sog_definition())result.charges_.clear();
        result.energy_ = 0.0;
        for (int global_i = structure_offsets_[s];
             global_i < structure_offsets_[s + 1]; ++global_i) {
            result.energy_ += site_energies_[global_i];
        }
        result.forces_.clear();
        result.site_energies_.clear();
        result.stress_.setZero();
        result.have_stress = false;
    }
    return true;
}

bool MTPGpuBatch::evaluate_loss(
    MTPGpuEvaluator &evaluator,
    const std::vector<double> &target_energy,
    const std::vector<double> &energy_weight,
    const std::vector<double> &target_force,
    const std::vector<double> &force_weight,
    const std::vector<double> &target_stress,
    const std::vector<double> &stress_weight,
    std::array<double, MLIPCore::GpuDirect::MTPGpuLossRequest::metric_count> &metrics,
    bool reset_metrics, bool copy_metrics_to_host,
    std::string *error)
{
    if (evaluator.initialized()) evaluator.context().invalidate_forward();
    if (!evaluator.initialized())
        return set_error(error, "GPU MTP evaluator is not initialized");
    if (target_energy.size() != static_cast<size_t>(structure_count()) ||
        energy_weight.size() != target_energy.size() ||
        target_force.size() != static_cast<size_t>(atom_count()) * 3 ||
        force_weight.size() != static_cast<size_t>(atom_count()) ||
        target_stress.size() != static_cast<size_t>(structure_count()) * 9 ||
        stress_weight.size() != target_energy.size()) {
        return set_error(error,
            "GPU MTP supervised labels do not match the resident batch");
    }

    auto compute = request();
    if(!prepare_sog(evaluator,compute,error))return false;
    double ignored_energy = 0.0;
    compute.total_energy = &ignored_energy;
    compute.reuse_topology = can_reuse_topology(evaluator, true);
    compute.copy_outputs_to_host = false;
    compute.compute_global_energy = false;
    if (!evaluator.context().compute_batch(compute, error)) return false;
    record_uploaded_topology(evaluator);

    const std::array<const std::vector<double> *, 6> labels = {
        &target_energy, &energy_weight, &target_force, &force_weight,
        &target_stress, &stress_weight};
    bool labels_changed = false;
    for (size_t i = 0; i < labels.size(); ++i)
        labels_changed |= cached_labels_[i] != *labels[i];
    if (labels_changed) {
        ++labels_revision_;
        for (size_t i = 0; i < labels.size(); ++i) cached_labels_[i] = *labels[i];
    }
    MLIPCore::GpuDirect::MTPGpuLossRequest loss;
    loss.labels_token = {identity_.value, labels_revision_};
    loss.structure_count = structure_count();
    loss.structure_offsets = structure_offsets_.data();
    loss.center_structure = center_structure_.data();
    loss.target_energy = target_energy.data();
    loss.energy_weight = energy_weight.data();
    loss.target_force = target_force.data();
    loss.force_weight = force_weight.data();
    loss.target_stress = target_stress.data();
    loss.stress_weight = stress_weight.data();
    loss.metrics = metrics.data();
    loss.reset_metrics = reset_metrics;
    loss.copy_metrics_to_host = copy_metrics_to_host;
    loss.deterministic_reduction = deterministic_reduction_;
    return evaluator.context().prepare_loss(loss, error);
}

bool MTPGpuBatch::require_forward(MTPGpuEvaluator &evaluator, std::string *error) const
{
    if (!built_ || !evaluator.initialized() || !evaluator.context().has_input(input_token())) {
        if (evaluator.initialized()) evaluator.context().invalidate_forward();
        return set_error(error, "GPU gradient requires a successful forward of this batch version");
    }
    return true;
}

bool MTPGpuBatch::accumulate_gradient(
    MTPGpuEvaluator &evaluator,
    const std::vector<double> &dloss_denergy,
    const std::vector<double> &dloss_dforce,
    const std::vector<double> &dloss_dstress,
    double *gradient, int gradient_count, bool accumulate,
    std::string *error)
{
    if (!require_forward(evaluator, error)) return false;
    if (!evaluator.initialized()) return set_error(error, "GPU MTP evaluator is not initialized");
    if (dloss_denergy.size() != static_cast<size_t>(structure_count()) ||
        dloss_dforce.size() != static_cast<size_t>(atom_count()) * 3 ||
        dloss_dstress.size() != static_cast<size_t>(structure_count()) * 9) {
        evaluator.context().invalidate_forward();
        return set_error(error, "GPU MTP loss-adjoint arrays do not match the resident batch");
    }
    MLIPCore::GpuDirect::MTPGpuGradientRequest req;
    req.structure_count = structure_count();
    req.center_structure = center_structure_.data();
    req.dloss_denergy = dloss_denergy.data();
    req.dloss_dforce = dloss_dforce.data();
    req.dloss_dstress = dloss_dstress.data();
    req.gradient = gradient;
    req.gradient_count = gradient_count;
    req.accumulate_gradient = accumulate;
    req.deterministic_reduction = deterministic_reduction_;
    req.radial_parameter_tile = radial_parameter_tile_;
    return evaluator.accumulate_gradient(req, error);
}

bool MTPGpuBatch::accumulate_gradient_chunks(
    MTPGpuEvaluator &evaluator,
    const std::vector<double> &dloss_denergy,
    const std::vector<double> &dloss_dforce,
    const std::vector<double> &dloss_dstress,
    double *gradient, int gradient_count, bool reset, bool finalize,
    std::string *error)
{
    if (!require_forward(evaluator, error)) return false;
    if (!evaluator.initialized())
        return set_error(error, "GPU MTP evaluator is not initialized");
    if (dloss_denergy.size() != static_cast<size_t>(structure_count()) ||
        dloss_dforce.size() != static_cast<size_t>(atom_count()) * 3 ||
        dloss_dstress.size() != static_cast<size_t>(structure_count()) * 9) {
        evaluator.context().invalidate_forward();
        return set_error(error,
            "GPU MTP loss-adjoint arrays do not match the resident batch");
    }
    MLIPCore::GpuDirect::MTPGpuGradientRequest req;
    req.structure_count = structure_count();
    req.center_structure = center_structure_.data();
    req.dloss_denergy = dloss_denergy.data();
    req.dloss_dforce = dloss_dforce.data();
    req.dloss_dstress = dloss_dstress.data();
    req.gradient = gradient;
    req.gradient_count = gradient_count;
    req.accumulate_gradient = !reset;
    req.accumulate_gradient_on_device = true;
    req.copy_gradient_to_host = finalize;
    req.deterministic_reduction = deterministic_reduction_;
    req.radial_parameter_tile = radial_parameter_tile_;
    return evaluator.accumulate_gradient(req, error);
}

bool MTPGpuBatch::accumulate_prepared_gradient(
    MTPGpuEvaluator &evaluator, double *gradient, int gradient_count,
    bool reset, bool finalize, std::string *error)
{
    if (!require_forward(evaluator, error)) return false;
    if (!evaluator.initialized())
        return set_error(error, "GPU MTP evaluator is not initialized");
    MLIPCore::GpuDirect::MTPGpuGradientRequest req;
    req.structure_count = structure_count();
    req.gradient = gradient;
    req.gradient_count = gradient_count;
    req.accumulate_gradient = !reset;
    req.accumulate_gradient_on_device = true;
    req.copy_gradient_to_host = finalize;
    req.adjoints_on_device = true;
    req.deterministic_reduction = deterministic_reduction_;
    req.radial_parameter_tile = radial_parameter_tile_;
    return evaluator.accumulate_gradient(req, error);
}

} // namespace Potentials

#endif // IMRMLP_GPU
