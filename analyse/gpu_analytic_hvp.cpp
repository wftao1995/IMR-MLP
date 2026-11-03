// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpu_analytic_hvp.h"

#include "base/neighborhood.h"
#include "potentials/mtp_gpu_adapter.h"
#include "potentials/potential_mtp.h"
#include "utils/comm.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <locale>
#include <istream>
#include <ostream>
#include <streambuf>
#include <typeinfo>
#include <vector>

namespace {
using Clock = std::chrono::steady_clock;

std::size_t add(std::size_t a, std::size_t b)
{
    if (b > std::numeric_limits<std::size_t>::max() - a)
        GPERROR("GPU analytic HVP host size overflow");
    return a + b;
}

std::size_t multiply(std::size_t a, std::size_t b)
{
    if (a && b > std::numeric_limits<std::size_t>::max() / a)
        GPERROR("GPU analytic HVP storage size overflow");
    return a * b;
}

std::size_t mib(int value)
{
    if (value <= 0) GPERROR("GPU analytic HVP memory budgets must be positive");
    return multiply(static_cast<std::size_t>(value), std::size_t{1} << 20);
}

void check_cap(std::size_t bytes, std::size_t cap)
{
    if (bytes > cap)
        GPERROR("GPU analytic HVP owned host workspace exceeds host_memory_mib");
}

std::size_t checked_dof(const Atoms &atom)
{
    return HessianGpu::checked_analytic_hvp_dof(atom.pos_.size());
}

// Native formula parameters are private; their portable descriptors are small.
// Bound even a custom writer before parsing with the existing constructor
// validators, without serializing or copying the complete model.
class FormulaDescriptorBuffer : public std::streambuf {
    char bytes_[512];
public:
    FormulaDescriptorBuffer() { setp(bytes_, bytes_ + sizeof(bytes_)); }
    void rewind() { setg(bytes_, bytes_, pptr()); }
};

template<class Formula, class Load>
void validate_formula_parameters(const Formula &formula, Load load)
{
    FormulaDescriptorBuffer buffer;
    std::ostream output(&buffer);
    output.imbue(std::locale::classic());
    output.precision(std::numeric_limits<double>::max_digits10);
    output.exceptions(std::ios::badbit | std::ios::failbit);
    formula.SaveToStream(output);
    buffer.rewind();
    std::istream input(&buffer);
    input.imbue(std::locale::classic());
    const auto checked = load(input, formula.min_value(), formula.max_value());
    if (checked->type() != formula.type())
        GPERROR("GPU analytic HVP radial formula type does not match its descriptor");
    input >> std::ws;
    if (!input.eof()) GPERROR("GPU analytic HVP radial formula descriptor has trailing data");
}

void validate_radial_bases(const Potentials::PotMTP &model)
{
    const auto check = [&](const AnyBasis *basis) {
        if (!basis || basis->basis_size != model.rb_size || model.rb_size <= 0 ||
            !std::isfinite(basis->min_val) || !std::isfinite(basis->max_val) ||
            !(basis->min_val < basis->max_val) || !std::isfinite(basis->scaling))
            GPERROR("GPU analytic HVP requires valid radial basis dimensions and interval");
        const auto &type = typeid(*basis);
        const auto code = basis_name_to_code(basis->name);
        const bool supported = (type == typeid(RBasis_Chebyshev) && code == RB_Chebyshev) ||
            (type == typeid(RBasis_ChebyCos) && code == RB_ChebyCos) ||
            (type == typeid(RBasis_ChebyshevPoly4) && code == RB_ChebyshevPoly4) ||
            (type == typeid(RBasis_ChebyshevPoly5) && code == RB_ChebyshevPoly5) ||
            (type == typeid(RBasis_PolyB) && code == RB_PolyB) ||
            (type == typeid(RBasis_PolyBCos) && code == RB_PolyBCos) ||
            (type == typeid(RBasis_ChebyshevConfigurable) && code == RB_ChebyshevConfigurable);
        if (!supported) GPERROR("GPU analytic HVP does not support this radial basis type");
        if (const auto *b = dynamic_cast<const RBasis_ChebyshevPoly4 *>(basis); b && !std::isfinite(b->b))
            GPERROR("GPU analytic HVP requires finite radial parameters");
        if (const auto *b = dynamic_cast<const RBasis_ChebyshevPoly5 *>(basis); b && !std::isfinite(b->b))
            GPERROR("GPU analytic HVP requires finite radial parameters");
        const auto polynomial = [&](const auto &b) {
            if (b.coeffs.rows() != model.rb_size || b.coeffs.cols() != model.rb_size ||
                !b.coeffs.allFinite() || b.invNalpha.size() != std::size_t(model.rb_size))
                GPERROR("GPU analytic HVP requires complete finite polynomial radial coefficients");
            for (double value : b.invNalpha)
                if (!std::isfinite(value)) GPERROR("GPU analytic HVP radial normalization must be finite");
        };
        if (const auto *b = dynamic_cast<const RBasis_PolyB *>(basis)) polynomial(*b);
        if (const auto *b = dynamic_cast<const RBasis_PolyBCos *>(basis)) polynomial(*b);
        if (const auto *b = dynamic_cast<const RBasis_ChebyshevConfigurable *>(basis)) {
            const auto &cutoff = b->cutoff_function();
            const auto &shift = b->shift_function();
            if (cutoff.min_value() != b->min_val || cutoff.max_value() != b->max_val ||
                shift.min_value() != b->min_val || shift.max_value() != b->max_val)
                GPERROR("GPU analytic HVP configurable radial component intervals must match the basis");
            if ((b->basis_function_type() != BasisFunctionType::Chebyshev &&
                 b->basis_function_type() != BasisFunctionType::OneMinusChebyshev) ||
                int(cutoff.type()) < int(Cutoff::CutoffType::Quadratic) ||
                int(cutoff.type()) > int(Cutoff::CutoffType::Smoothstep7) ||
                int(shift.type()) < int(XTransForm::ShiftType::Linear) ||
                int(shift.type()) > int(XTransForm::ShiftType::ExponentialTanh))
                GPERROR("GPU analytic HVP configurable radial formula enum is unsupported");
            validate_formula_parameters(cutoff, Cutoff::load_cutoff);
            validate_formula_parameters(shift, XTransForm::load_shift);
        }
    };
    if (model.radial_basis_split) {
        const auto pairs = multiply(std::size_t(model.species_count), std::size_t(model.species_count));
        if (model.p_radical_basis_pair.size() != pairs)
            GPERROR("GPU analytic HVP requires every species-pair radial basis");
        for (const auto *basis : model.p_radical_basis_pair) check(basis);
    } else {
        check(model.p_radical_basis);
    }
}

double elapsed(Clock::time_point start)
{
    return std::chrono::duration<double>(Clock::now() - start).count();
}

} // namespace

namespace HessianGpu {

void validate_analytic_radial_bases(const Potentials::PotMTP &model)
{
    validate_radial_bases(model);
}

std::size_t checked_analytic_hvp_dof(std::size_t atoms)
{
    if (atoms > std::size_t(std::numeric_limits<int>::max() / 6) ||
        atoms > std::size_t(std::numeric_limits<Eigen::Index>::max() / 3))
        GPERROR("GPU analytic HVP atom count exceeds the Cartesian/packed index range");
    return multiply(atoms, 3);
}

std::size_t analytic_geometry_bytes(const Atoms &atom, bool copied)
{
    (void)checked_dof(atom);
    auto bytes = add(sizeof(Atoms), multiply(copied ? atom.pos_.size() : atom.pos_.capacity(),
                                           sizeof(Eigen::RowVector3d)));
    return add(bytes, multiply(copied ? atom.chemmap_.size() : atom.chemmap_.capacity(), sizeof(int)));
}

std::size_t analytic_hvp_preflight(const Atoms &atom, const double *direction,
    std::size_t direction_count, const Potentials::AnyPotential &potential,
    double cutoff, const HessianExecutionSettings &settings, std::size_t external_host_bytes)
{
    const auto dof = checked_dof(atom);
    if (direction_count != dof || (dof && !direction))
        GPERROR("GPU analytic HVP requires a Cartesian direction of size 3*N");
    bool nonzero = false;
    for (std::size_t i = 0; i < dof; ++i) {
        if (!std::isfinite(direction[i])) GPERROR("GPU analytic HVP direction must be finite");
        nonzero = nonzero || direction[i] != 0.;
    }
    HessianExecutionReport unused;
    if (!begin(settings, unused) || unused.backend != "cuda")
        GPERROR("GPU analytic HVP currently requires device=cuda");
    if (mpi.size != 1) GPERROR("GPU analytic HVP requires a single MPI rank");
    validate(atom, potential, cutoff, settings);
    validate_radial_bases(static_cast<const Potentials::PotMTP &>(potential));
    if (!atom.cartesian) {
        for (const auto &position : atom.pos_) {
            const Eigen::RowVector3d cartesian = position * atom.cell_;
            if (!cartesian.allFinite())
                GPERROR("GPU analytic HVP Cartesian conversion produced nonfinite positions");
        }
    }
    auto required = add(external_host_bytes, add(sizeof(Eigen::VectorXd), multiply(dof, sizeof(double))));
    if (nonzero) {
        required = add(required, analytic_geometry_bytes(atom, true));
        required = add(required, multiply(dof, sizeof(double)));
    }
    check_cap(required, mib(settings.host_memory_mib));
    return dof;
}

Atoms analytic_geometry_copy(const Atoms &atom, const HessianExecutionSettings &settings,
                             std::size_t reserved_host_bytes)
{
    check_cap(add(reserved_host_bytes, analytic_geometry_bytes(atom, true)), mib(settings.host_memory_mib));
    Atoms result;
    result.cell_ = atom.cell_;
    result.pos_ = atom.pos_;
    result.chemmap_ = atom.chemmap_;
    result.pbc_ = atom.pbc_;
    result.cartesian = atom.cartesian;
    if (!result.cartesian) result.to_cartesian();
    result.cartesian = true;
    for (const auto &position : result.pos_)
        if (!position.allFinite()) GPERROR("GPU analytic HVP Cartesian conversion produced nonfinite positions");
    check_cap(add(reserved_host_bytes, analytic_geometry_bytes(result)), mib(settings.host_memory_mib));
    return result;
}

Eigen::VectorXd analytic_hvp(const Atoms &input, const Eigen::VectorXd &direction,
    const Potentials::AnyPotential &potential, double cutoff,
    const HessianExecutionSettings &settings, HessianExecutionReport &report,
    std::size_t external_host_bytes)
{
    const auto dof = analytic_hvp_preflight(input, direction.data(), std::size_t(direction.size()),
        potential, cutoff, settings, external_host_bytes);
    const auto output_bytes = multiply(dof, sizeof(double));
    const auto cap = mib(settings.host_memory_mib);
    report.host_hessian_bytes = output_bytes;
    report.method = "analytic";
    const auto record = [&](std::size_t bytes) {
        check_cap(bytes, cap);
        report.host_workspace_bytes = std::max(report.host_workspace_bytes, bytes);
    };
    bool nonzero = false;
    for (Eigen::Index i = 0; i < direction.size(); ++i) nonzero = nonzero || direction[i] != 0.;
    if (!nonzero) {
        record(add(external_host_bytes, add(sizeof(Eigen::VectorXd), output_bytes)));
        return Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dof));
    }
#ifdef IMRMLP_GPU
    // Either adapter normalization+pending result, or its returned vector+
    // final Eigen result coexist. Both phases need two complete 3*N buffers.
    const auto result_reserve = multiply(output_bytes, 2);
    const auto containers = sizeof(Potentials::MTPGpuBatch) + sizeof(Neighborhoods) +
        sizeof(std::vector<const Atoms *>) + sizeof(std::vector<const AnyNeighborhoods *>) +
        sizeof(std::vector<double>) + sizeof(Eigen::VectorXd) + sizeof(const Atoms *) +
        sizeof(const AnyNeighborhoods *);
    const auto fixed = add(external_host_bytes, add(containers, result_reserve));
    record(add(fixed, analytic_geometry_bytes(input, true)));
    auto start = Clock::now();
    Atoms geometry = analytic_geometry_copy(input, settings, fixed);
    const auto geometry_bytes = analytic_geometry_bytes(geometry);
    const auto base = add(fixed, geometry_bytes);
    record(base);
    const auto construction = Neighborhoods::construction_memory_bound(geometry, cutoff, cap - base);
    record(add(base, construction));
    Neighborhoods neighborhoods(geometry, cutoff);
    const auto neighbor_bytes = neighborhoods.owned_memory_bytes();
    // owned_memory_bytes includes the inline object already in containers.
    auto live = add(base, neighbor_bytes - sizeof(Neighborhoods));
    record(live);
    report.neighborhoods_seconds += elapsed(start);
    std::size_t edges = 0;
    for (int center = 0; center < neighborhoods.size(); ++center) {
        const auto &site = neighborhoods[center];
        if (site.nbhcount_ < 0) GPERROR("GPU analytic HVP invalid neighbor count");
        edges = add(edges, std::size_t(site.nbhcount_));
        if (edges > std::size_t(std::numeric_limits<int>::max() / 3))
            GPERROR("GPU analytic HVP neighbor count exceeds packed indexing");
        for (double distance : site.distances_)
            if (!std::isfinite(distance) || distance <= 0.)
                GPERROR("GPU analytic HVP requires finite positive neighbor distances");
    }
    const std::vector<const Atoms *> atoms{&geometry};
    const std::vector<const AnyNeighborhoods *> nbhs{&neighborhoods};
    live = add(live - sizeof(const Atoms *) - sizeof(const AnyNeighborhoods *),
        add(multiply(atoms.capacity(), sizeof(const Atoms *)),
            multiply(nbhs.capacity(), sizeof(const AnyNeighborhoods *))));
    record(live);
    Potentials::MTPGpuBatch batch;
    std::string error;
    const auto require = [&](bool ok) { if (!ok) GPERROR("GPU analytic HVP: " + error); };
    start = Clock::now();
    require(batch.build_bounded(atoms, nbhs, cap - live, &error));
    record(add(live, batch.owned_host_bytes()));
    report.packing_seconds += elapsed(start);
    Potentials::MTPGpuEvaluator evaluator;
    start = Clock::now();
    require(evaluator.initialize_bounded(potential, settings.gpu_id, mib(settings.gpu_memory_mib), &error));
    ++report.evaluator_initializations;
    report.setup_seconds += elapsed(start);
    report.tracked_device_bytes = evaluator.context().memory_stats().tracked_device_bytes;
    std::vector<double> pending;
    start = Clock::now();
    // live already reserves the two result arrays: release that reservation
    // only in the sub-budget passed to the adapter, which counts them itself.
    const auto adapter_cap = cap - (live - result_reserve);
    require(batch.evaluate_analytic_hvp(evaluator, direction.data(), dof, pending,
        mib(settings.gpu_memory_mib), adapter_cap, &error));
    record(add(live - result_reserve, batch.last_analytic_hvp_host_bytes()));
    report.evaluation_seconds += elapsed(start);
    const auto &stats = evaluator.context().runtime_stats();
    report.analytic_hvp_calls = stats.analytic_hvp_calls;
    report.direction_upload_bytes = stats.analytic_direction_upload_bytes;
    report.geometry_uploads = stats.coordinate_uploads;
    report.geometry_reuses = stats.input_reuses;
    report.download_bytes = stats.analytic_output_download_bytes;
    report.tracked_device_bytes = evaluator.context().memory_stats().tracked_device_bytes;
    report.evaluated_structures = 1;
    report.batches = 1;
    report.peak_batch_atoms = input.pos_.size();
    report.peak_batch_structures = 1;
    report.peak_batch_neighbors = edges;
    if (pending.size() != dof) GPERROR("GPU analytic HVP returned an incomplete result");
    record(add(live - result_reserve, add(batch.owned_host_bytes(),
        add(multiply(pending.capacity(), sizeof(double)), output_bytes))));
    start = Clock::now();
    Eigen::VectorXd result(static_cast<Eigen::Index>(dof));
    for (std::size_t i = 0; i < dof; ++i) {
        if (!std::isfinite(pending[i])) GPERROR("GPU analytic HVP produced nonfinite curvature");
        result[static_cast<Eigen::Index>(i)] = pending[i];
    }
    report.reduction_seconds += elapsed(start);
    return result;
#else
    GPERROR("GPU analytic HVP is unavailable in this CPU-only build");
    return {};
#endif
}

} // namespace HessianGpu
