// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "fixtures/gpu_linear_curvature.h"
#include "train/linear_curvature_observations.h"
#include "apps/apps.h"
#include "io/gpio.h"
#include "tools/picojson.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <limits>
#include <set>
#include <tuple>
#include <unistd.h>

namespace
{
namespace C = GpuLinearCurvatureFixture;
namespace F = GpuHessianFixture;
namespace A = GpuAnalyticHvpFixture;
namespace G = MLIPCore::GpuDirect;
using Potentials::PotMTP;

void close(double actual, double expected, double atol = 2e-8, double rtol = 3e-9)
{
    CAPTURE(actual, expected);
    REQUIRE(std::isfinite(actual));
    REQUIRE(std::isfinite(expected));
    REQUIRE(std::abs(actual - expected) <= atol + rtol * std::abs(expected));
}

void component_close(const Eigen::MatrixXd &actual, const Eigen::MatrixXd &reference,
                     const Eigen::VectorXd &v, bool cpu = false)
{
    REQUIRE(actual.rows() == reference.rows());
    REQUIRE(actual.cols() == reference.cols());
    REQUIRE(actual.allFinite());
    REQUIRE(reference.allFinite());
    const double scale = F::maximum_atom_norm(v);
    if (scale == 0.)
    {
        REQUIRE(actual.isZero(0.));
        return;
    }
    if (cpu)
    {
        REQUIRE(((actual - reference) / scale).cwiseAbs().maxCoeff() <=
                C::cpu_tolerance * std::max(1., (reference / scale).cwiseAbs().maxCoeff()));
    }
    else
        for (int r = 0; r < actual.rows(); ++r)
            for (int j = 0; j < actual.cols(); ++j)
            {
                CAPTURE(r, j, scale);
                close(actual(r, j) / scale, reference(r, j) / scale, C::gpu_atol, C::gpu_rtol);
            }
}

void qualify(PotMTP &model, const Atoms &input, bool fd)
{
    auto a = input;
    const auto v = F::direction(a.size());
    const auto matrix = C::components(model, a, v);
    REQUIRE(matrix.cols() == model.lin_coeff_size);
    REQUIRE(matrix.leftCols(model.species_count).isZero(0.));
    component_close(matrix, C::dense_components(model, a, v), v, true);
    auto one = C::copy(model);
    Neighborhoods nbhs(a, model.cutoff);
    std::vector<double> weights(model.lin_coeff_size, 0.);
    for (int j = 0; j < model.lin_coeff_size; ++j)
    {
        CAPTURE(j);
        std::fill(weights.begin(), weights.end(), 0.);
        weights[j] = 1.;
        one->set_linear_coeff(weights.data());
        Eigen::VectorXd value;
        one->applyHessianVectorProduct(a, nbhs, v, value);
        component_close(matrix.col(j), value, v, true);
    }
    Eigen::VectorXd physical;
    model.applyHessianVectorProduct(a, nbhs, v, physical);
    component_close(
        matrix * Eigen::Map<const Eigen::VectorXd>(model.linear_coeffs, model.lin_coeff_size),
        physical, v, true);
    for (int d = 0; d < 3; ++d)
        REQUIRE(C::components(model, a, F::translation(a.size(), d)).isZero(0.));
    component_close(C::components(model, a, -3.7 * v), -3.7 * matrix, -3.7 * v, true);
    if (!fd)
        return;
    std::array<double, 4> errors{};
    const double scale = F::maximum_atom_norm(v);
    for (size_t i = 0; i < A::steps.size(); ++i)
    {
        CAPTURE(A::steps[i]);
        const auto difference = C::force_difference(model, a, v, A::steps[i]);
        errors[i] = ((difference - matrix) / scale).cwiseAbs().maxCoeff();
        if (i + 1 == A::steps.size())
            for (int r = 0; r < matrix.rows(); ++r)
                for (int j = 0; j < matrix.cols(); ++j)
                    close(difference(r, j) / scale, matrix(r, j) / scale, 2e-6, 2e-6);
    }
    const double floor = 2e-9 * std::max(1., (matrix / scale).cwiseAbs().maxCoeff());
    REQUIRE(errors[1] <= .35 * errors[0] + floor);
    REQUIRE(errors[3] <= .35 * errors[2] + floor);
}
}

TEST_CASE("Native linear curvature CPU references qualify every basis column",
          "[linear-curvature][cpu-oracle]")
{
    for (int variant : {1, 2, 3, 4})
    {
        CAPTURE(variant);
        PotMTP model;
        F::load(model, variant);
        model.train_init();
        model.set_inited();
        for (int n : {3, 5})
        {
            auto a = F::cluster(n);
            if (model.species_count == 1)
                std::fill(a.chemmap_.begin(), a.chemmap_.end(), 0);
            CAPTURE(n);
            qualify(model, a, true);
        }
        if (variant == 4)
            continue;
        for (int n : {1, 4})
        {
            auto a = F::periodic(n);
            if (model.species_count == 1)
                std::fill(a.chemmap_.begin(), a.chemmap_.end(), 0);
            CAPTURE(n);
            qualify(model, a, false);
        }
    }
}

TEST_CASE("Native linear curvature CPU consumer preserves upper triangle and probe averaging",
          "[linear-curvature][cpu-oracle][training]")
{
    const bool fast = GENERATE(false, true);
    const int kind = GENERATE(0, 1, 2);
    C::Sample sample(kind);
    C::Options options;
    options.fast = fast;
    options.norm_p = 0.;
    if (kind == 0)
        options.hvp = 0.;
    if (kind == 1)
        options.hessian = 0.;
    if (kind != 2)
        options.energy = options.force = options.virial = 0.;
    auto base = sample.initial(), changed = C::copy(*base);
    auto modified = sample.data;
    for (auto &a : modified)
    {
        if (a.has_hessian())
            for (int r = 1; r < a.hessian_.rows(); ++r)
                for (int c = 0; c < r; ++c)
                    a.hessian_(r, c) += 1000. + r;
        if (a.has_hvp())
        {
            a.hvp_directions_ = a.hvp_directions_.replicate(1, 2).eval();
            a.hvp_targets_ = a.hvp_targets_.replicate(1, 2).eval();
        }
    }
    close(C::data_loss(*base, sample.data, sample.neighbors, options),
          C::data_loss(*base, modified, sample.neighbors, options), 1e-12, 1e-12);
    Trainer_MTP first(options.settings()), second(options.settings());
    first.init(sample.data, sample.neighbors, base.get());
    second.init(modified, sample.neighbors, changed.get());
    auto original = C::copy(*base);
    first.TrainLinear();
    second.TrainLinear();
    original->reg_L2_vector = base->reg_L2_vector;
    const double before = C::objective(*original, sample.data, sample.neighbors, options);
    const double after = C::objective(*base, sample.data, sample.neighbors, options);
    REQUIRE(after <= before + 1e-10 * std::max(1., std::abs(before)));
    close(after, C::objective(*changed, sample.data, sample.neighbors, options));
    for (size_t k = 0; k < sample.data.size(); ++k)
    {
        const auto v = F::direction(sample.data[k].size());
        component_close(A::apply(*base, sample.data[k], v), A::apply(*changed, sample.data[k], v),
                        v);
    }
}

#ifdef IMRMLP_GPU
namespace
{
bool available()
{
    int count = 0;
    std::string error;
    const bool ok = std::string(G::compiled_backend()) == "cuda" &&
                    G::MTPGpuDirect::available_device_count(&count, &error) && count > 0;
    if (std::getenv("IMRMLP_REQUIRE_GPU"))
    {
        INFO(error);
        REQUIRE(ok);
    }
    return ok;
}

Eigen::MatrixXd selected(const Eigen::MatrixXd &matrix, int first, int rows,
                         const std::vector<int> &columns)
{
    Eigen::MatrixXd result(rows, columns.size());
    for (size_t j = 0; j < columns.size(); ++j)
        result.col(j) = matrix.col(columns[j]).segment(first, rows);
    return result;
}

Eigen::MatrixXd unpack(const std::vector<double> &values, int rows, int columns)
{
    REQUIRE(values.size() == size_t(rows) * size_t(columns));
    return Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>(values.data(), rows,
                                                                            columns);
}

void fit_predictions(PotMTP &actual, PotMTP &reference, const std::vector<Atoms> &data)
{
    for (const auto &a : data)
    {
        auto left = a, right = a;
        actual.getEFS(left);
        reference.getEFS(right);
        close(left.energy_, right.energy_, 2e-7, 2e-8);
        double force_error = 0., force_scale = 0.;
        for (size_t i = 0; i < a.size(); ++i)
        {
            force_error =
                std::max(force_error, (left.forces_[i] - right.forces_[i]).cwiseAbs().maxCoeff());
            force_scale = std::max(force_scale, right.forces_[i].cwiseAbs().maxCoeff());
        }
        REQUIRE(force_error <= 2e-7 + 2e-8 * force_scale);
        REQUIRE((left.stress_ - right.stress_).cwiseAbs().maxCoeff() <=
                2e-7 + 2e-8 * right.stress_.cwiseAbs().maxCoeff());
        if (a.has_hessian())
            for (int j = 0; j < 3 * a.size(); ++j)
            {
                Eigen::VectorXd v = Eigen::VectorXd::Zero(3 * a.size());
                v[j] = 1.;
                const auto x = A::apply(actual, a, v), y = A::apply(reference, a, v);
                component_close(x.head(j + 1), y.head(j + 1), v);
            }
        if (a.has_hvp())
            for (int j = 0; j < a.hvp_directions_.cols(); ++j)
            {
                const Eigen::VectorXd v = a.hvp_directions_.col(j);
                component_close(A::apply(actual, a, v), A::apply(reference, a, v), v);
            }
    }
}
}

#define REQUIRE_CURVATURE_GPU()                                                                    \
    do                                                                                             \
    {                                                                                              \
        if (!available())                                                                          \
            SKIP("Required CUDA device unavailable");                                              \
    } while (false)

TEST_CASE("Selected GPU native curvature blocks retain rows columns and physical direction",
          "[gpu][linear-curvature][producer]")
{
    REQUIRE_CURVATURE_GPU();
    PotMTP model;
    F::load(model, 2);
    std::string error;
    INFO(error);
    Potentials::MTPGpuEvaluator owner;
    REQUIRE(owner.initialize_bounded(model, 0, size_t{64} << 20, &error));
    std::vector<Atoms> atoms{F::cluster(2), F::cluster(5), F::periodic(1), F::periodic(4)};
    std::vector<Neighborhoods> nbhs;
    for (const auto &a : atoms)
        nbhs.emplace_back(a, model.cutoff);
    std::vector<const Atoms *> ap;
    std::vector<const AnyNeighborhoods *> np;
    std::vector<double> direction;
    Eigen::MatrixXd reference(36, model.lin_coeff_size);
    int offset = 0;
    for (size_t k = 0; k < atoms.size(); ++k)
    {
        ap.push_back(&atoms[k]);
        np.push_back(&nbhs[k]);
        const auto v = F::direction(atoms[k].size(), .1 * k);
        reference.middleRows(offset, v.size()) = C::components(model, atoms[k], v);
        offset += v.size();
        direction.insert(direction.end(), v.data(), v.data() + v.size());
    }
    const Eigen::VectorXd packed =
        Eigen::Map<const Eigen::VectorXd>(direction.data(), direction.size());
    Potentials::MTPGpuBatch batch;
    REQUIRE(batch.build_bounded(ap, np, size_t{16} << 20, &error));
    const auto columns = C::columns(model);
    std::vector<double> output{31., 37.};
    const auto before = owner.context().runtime_stats();
    REQUIRE(batch.linear_hvp_block(owner, direction.data(), direction.size(), 0, 36, columns,
                                   output, size_t{64} << 20, size_t{16} << 20, &error));
    component_close(unpack(output, 36, columns.size()), selected(reference, 0, 36, columns),
                    packed);
    for (int r = 0; r < 36; ++r)
    {
        REQUIRE(output[5 * r] == output[5 * r + 3]);
        REQUIRE(output[5 * r + 1] == 0.);
        REQUIRE(output[5 * r + 4] == 0.);
    }
    const auto after = owner.context().runtime_stats();
    REQUIRE(after.linear_hvp_calls - before.linear_hvp_calls == 1);
    REQUIRE(after.linear_hvp_direction_upload_bytes - before.linear_hvp_direction_upload_bytes ==
            direction.size() * sizeof(double));
    REQUIRE(after.linear_hvp_output_download_bytes - before.linear_hvp_output_download_bytes ==
            output.size() * sizeof(double));
    REQUIRE(after.training_force_download_bytes == before.training_force_download_bytes);
    REQUIRE(after.linear_hvp_basic_directional_launches >
            before.linear_hvp_basic_directional_launches);
    REQUIRE(after.linear_hvp_product_launches > before.linear_hvp_product_launches);
    REQUIRE(batch.last_linear_hvp_host_bytes() <= size_t{16} << 20);
    for (int first = 0; first < 36; first += 7)
    {
        const int rows = std::min(7, 36 - first);
        REQUIRE(batch.linear_hvp_block(owner, direction.data(), direction.size(), first, rows,
                                       columns, output, size_t{64} << 20, size_t{16} << 20,
                                       &error));
        component_close(unpack(output, rows, columns.size()),
                        selected(reference, first, rows, columns), packed);
    }
    output = direction;
    REQUIRE(batch.linear_hvp_block(owner, output.data(), output.size(), 3, 11, columns, output,
                                   size_t{64} << 20, size_t{16} << 20, &error));
    component_close(unpack(output, 11, columns.size()), selected(reference, 3, 11, columns),
                    packed);
    auto scaled = direction;
    for (auto &v : scaled)
        v *= -3.7;
    REQUIRE(batch.linear_hvp_block(owner, scaled.data(), scaled.size(), 0, 36, columns, output,
                                   size_t{64} << 20, size_t{16} << 20, &error));
    component_close(unpack(output, 36, columns.size()), -3.7 * selected(reference, 0, 36, columns),
                    -3.7 * packed);
    std::fill(scaled.begin(), scaled.end(), 0.);
    REQUIRE(batch.linear_hvp_block(owner, scaled.data(), scaled.size(), 0, 36, columns, output,
                                   size_t{64} << 20, size_t{16} << 20, &error));
    for (double v : output)
        REQUIRE(v == 0.);
    for (int chunk : {1, 2, 64})
    {
        auto geometry = batch.request();
        geometry.memory_budget_bytes = size_t{64} << 20;
        geometry.centers_per_chunk = chunk;
        G::MTPGpuLinearHvpRequest block;
        block.direction = direction.data();
        block.direction_count = direction.size();
        block.first_row = 2;
        block.row_count = 13;
        block.columns = columns.data();
        block.column_count = columns.size();
        output.assign(13 * columns.size(), 31.);
        block.output = output.data();
        block.output_count = output.size();
        const auto old = owner.context().runtime_stats();
        REQUIRE(owner.context().compute_linear_hvp_block(geometry, block, &error));
        const auto now = owner.context().runtime_stats();
        const int chosen = std::min(chunk, 12), chunks = (12 + chosen - 1) / chosen;
        REQUIRE(now.last_centers_per_chunk == chosen);
        REQUIRE(now.last_chunk_count == chunks);
        REQUIRE(now.linear_hvp_basic_directional_launches -
                    old.linear_hvp_basic_directional_launches ==
                size_t(chunks));
        component_close(unpack(output, 13, columns.size()), selected(reference, 2, 13, columns),
                        packed);
    }
    for (int failure = 0; failure < 7; ++failure)
    {
        CAPTURE(failure);
        output = {31., 37.};
        auto invalid_direction = direction;
        auto invalid_columns = columns;
        if (failure == 0)
            invalid_direction[0] = std::numeric_limits<double>::infinity();
        if (failure == 1)
            invalid_columns[0] = model.lin_coeff_size;
        const auto count = failure == 2 ? direction.size() - 1 : direction.size();
        REQUIRE_FALSE(batch.linear_hvp_block(
            owner, invalid_direction.data(), count, failure == 3 ? -1 : 0, failure == 4 ? 37 : 36,
            invalid_columns, output, failure == 5 ? 1 : size_t{64} << 20,
            failure == 6 ? 1 : size_t{16} << 20, &error));
        REQUIRE((output == std::vector<double>{31., 37.}));
    }
    REQUIRE(batch.linear_hvp_block(owner, direction.data(), direction.size(), 36, 0, columns,
                                   output, size_t{64} << 20, size_t{16} << 20, &error));
    REQUIRE(output.empty());
    REQUIRE(batch.linear_hvp_block(owner, direction.data(), direction.size(), 0, 36, {}, output,
                                   size_t{64} << 20, size_t{16} << 20, &error));
    REQUIRE(output.empty());
    auto invalid = direction;
    invalid[0] = std::numeric_limits<double>::quiet_NaN();
    output = {31., 37.};
    REQUIRE_FALSE(batch.linear_hvp_block(owner, invalid.data(), invalid.size(), 0, 0, {}, output,
                                         size_t{64} << 20, size_t{16} << 20, &error));
    REQUIRE((output == std::vector<double>{31., 37.}));
    for (int j = 0; j < model.lin_coeff_size; ++j)
        model.linear_coeffs[j] += .13 * std::sin(j + .1);
    REQUIRE(owner.update_coefficients(model, &error));
    REQUIRE(batch.linear_hvp_block(owner, direction.data(), direction.size(), 0, 36, columns,
                                   output, size_t{64} << 20, size_t{16} << 20, &error));
    component_close(unpack(output, 36, columns.size()), selected(reference, 0, 36, columns),
                    packed);
    model.radial_coeffs[0] += .017;
    REQUIRE(owner.update_coefficients(model, &error));
    offset = 0;
    for (size_t k = 0; k < atoms.size(); ++k)
    {
        const auto v = F::direction(atoms[k].size(), .1 * k);
        reference.middleRows(offset, v.size()) = C::components(model, atoms[k], v);
        offset += v.size();
    }
    REQUIRE(batch.linear_hvp_block(owner, direction.data(), direction.size(), 0, 36, columns,
                                   output, size_t{64} << 20, size_t{16} << 20, &error));
    component_close(unpack(output, 36, columns.size()), selected(reference, 0, 36, columns),
                    packed);
    model.hc_correction = true;
    REQUIRE(owner.update_coefficients(model, &error));
    output = {31., 37.};
    REQUIRE_FALSE(batch.linear_hvp_block(owner, direction.data(), direction.size(), 0, 36, columns,
                                         output, size_t{64} << 20, size_t{16} << 20, &error));
    REQUIRE((output == std::vector<double>{31., 37.}));
}

TEST_CASE("Linear curvature visitor streams upper Hessian and HVP through unequal tight workspaces",
          "[gpu][linear-curvature][visitor]")
{
    REQUIRE_CURVATURE_GPU();
    PotMTP model;
    F::load(model, 2);
    const auto columns = C::columns(model);
    std::vector<Atoms> atoms{F::cluster(5), F::cluster(2)};
    std::vector<Neighborhoods> nbhs;
    for (auto &a : atoms)
    {
        a.hessian_ = Eigen::MatrixXd::Zero(3 * a.size(), 3 * a.size());
        a.have_hessian = true;
        a.hvp_directions_.resize(3 * a.size(), 2);
        a.hvp_targets_ = Eigen::MatrixXd::Zero(3 * a.size(), 2);
        for (int j = 0; j < 2; ++j)
            a.hvp_directions_.col(j) = F::direction(a.size(), .2 * j);
        nbhs.emplace_back(a, model.cutoff);
    }
    const auto original = atoms;
    Training::LinearEFSOptions options;
    options.device = "cuda";
    options.gpu_id = 0;
    options.gpu_batch_atoms = 5;
    options.gpu_batch_structures = 2;
    options.gpu_memory_bytes = size_t{64} << 20;
    // Large direction(120B)+small output(80B), then small direction(48B)+
    // larger output(120B): retaining independent maxima would violate 200B.
    options.gpu_workspace_bytes = 200;
    const size_t host_cap = size_t{1} << 20;
    Training::LinearCurvatureStatistics stats;
    std::set<std::tuple<size_t, bool, int, int>> seen;
    size_t total = 0;
    auto consume = [&](const Training::LinearCurvatureObservationBlock &block)
    {
        REQUIRE(block.configuration < atoms.size());
        REQUIRE(block.rows > 0);
        REQUIRE(std::vector<int>(block.native_columns.begin(), block.native_columns.end()) ==
                columns);
        const auto &a = atoms[block.configuration];
        Eigen::VectorXd v;
        if (block.hessian)
        {
            v = Eigen::VectorXd::Zero(3 * a.size());
            v[block.probe] = 1.;
            REQUIRE(block.first_row + block.rows <= block.probe + 1);
        }
        else
        {
            v = a.hvp_directions_.col(block.probe);
            REQUIRE(block.first_row + block.rows <= 3 * a.size());
        }
        const std::vector<double> values(block.values.begin(), block.values.end());
        component_close(unpack(values, block.rows, columns.size()),
                        selected(C::components(model, a, v), block.first_row, block.rows, columns),
                        v);
        for (int r = block.first_row; r < block.first_row + block.rows; ++r)
            REQUIRE(seen.emplace(block.configuration, block.hessian, block.probe, r).second);
        total += values.size() * sizeof(double);
    };
    Training::visit_gpu_linear_curvature(model, atoms, {{true, true}, {true, true}}, columns,
                                         options, stats, consume, &nbhs, host_cap, 128);
    const size_t expected_rows = 15 * 16 / 2 + 15 * 2 + 6 * 7 / 2 + 6 * 2;
    REQUIRE(seen.size() == expected_rows);
    REQUIRE(stats.download_bytes == total);
    REQUIRE(total == expected_rows * columns.size() * 8);
    REQUIRE(stats.probes == 15 + 2 + 6 + 2);
    REQUIRE(stats.blocks > stats.probes);
    REQUIRE(stats.output_workspace_bytes <= options.gpu_workspace_bytes);
    REQUIRE(stats.output_workspace_bytes > 0);
    REQUIRE(stats.tracked_device_bytes <= options.gpu_memory_bytes + options.gpu_workspace_bytes);
    REQUIRE(stats.host_workspace_bytes <= host_cap);
    REQUIRE(stats.host_workspace_bytes >= 128);
    for (size_t k = 0; k < atoms.size(); ++k)
        REQUIRE(F::unchanged(atoms[k], original[k]));
    size_t calls = 0;
    auto count = [&](const auto &)
    {
        ++calls;
    };
    Training::LinearCurvatureStatistics rejected;
    REQUIRE_THROWS(Training::visit_gpu_linear_curvature(
        model, atoms, {{true, true}, {true, true}}, columns, options, rejected, count, &nbhs, 1));
    REQUIRE(calls == 0);
    auto bad = atoms;
    bad[1].hvp_targets_(0, 0) = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS(Training::visit_gpu_linear_curvature(model, bad, {{true, true}, {true, true}},
                                                        columns, options, rejected, count, &nbhs,
                                                        host_cap));
    REQUIRE(calls == 0);
    REQUIRE_THROWS(Training::visit_gpu_linear_curvature(
        model, atoms, {{true, true}, {true, true}}, columns, options, rejected,
        [&](const auto &)
        {
            ++calls;
            throw std::runtime_error("intentional consumer failure");
        },
        &nbhs, host_cap));
    REQUIRE(calls == 1);
    Training::visit_gpu_linear_curvature(model, atoms, {{false, true}, {false, false}}, columns,
                                         options, rejected, count, &nbhs, host_cap);
    REQUIRE(calls > 1);
}

TEST_CASE("Native linear curvature failures retire prepared training tokens",
          "[gpu][linear-curvature][token]")
{
    REQUIRE_CURVATURE_GPU();
    PotMTP model;
    F::load(model);
    auto a = F::cluster(2);
    Neighborhoods nbhs(a, model.cutoff);
    Potentials::MTPGpuBatch batch;
    Potentials::MTPGpuEvaluator owner;
    std::string error;
    INFO(error);
    REQUIRE(owner.initialize_training_bounded(model, 0, size_t{64} << 20, &error));
    REQUIRE(batch.build_bounded({&a, &a}, {&nbhs, &nbhs}, size_t{16} << 20, &error));
    std::vector<double> target(6, .13), direction(12, .2), output(12, 37.),
        gradient(model.get_NL_coeffs_count(), 31.);
    G::MTPGpuHVPPairMetrics metrics;
    const std::vector<int> columns{model.species_count};
    REQUIRE(batch.evaluate_hvp_pair_loss(owner, target.data(), target.size(), 1e-4, .7, metrics,
                                         &error));
    auto geometry = batch.request();
    geometry.memory_budget_bytes = size_t{64} << 20;
    G::MTPGpuLinearHvpRequest request;
    request.direction = direction.data();
    request.direction_count = direction.size();
    request.row_count = 12;
    request.columns = columns.data();
    request.column_count = 1;
    request.output = output.data();
    request.output_count = output.size();
    REQUIRE_FALSE(owner.context().compute_linear_hvp_block(geometry, request, &error));
    REQUIRE_FALSE(batch.accumulate_prepared_gradient(owner, gradient.data(), gradient.size(), true,
                                                     true, &error));
    for (double v : gradient)
        REQUIRE(v == 31.);
    REQUIRE(batch.evaluate_hvp_pair_loss(owner, target.data(), target.size(), 1e-4, .7, metrics,
                                         &error));
    REQUIRE(batch.accumulate_prepared_gradient(owner, gradient.data(), gradient.size(), true, true,
                                               &error));
    for (double v : gradient)
        REQUIRE(std::isfinite(v));
}

TEST_CASE("Linear curvature matches split pair cutoffs at exact equality",
          "[gpu][linear-curvature][producer][cutoff]")
{
    REQUIRE_CURVATURE_GPU();
    PotMTP model;
    F::load(model, 1);
    Potentials::MTPGpuEvaluator owner;
    std::string error;
    INFO(error);
    REQUIRE(owner.initialize_bounded(model, 0, size_t{64} << 20, &error));
    const auto columns = C::columns(model);
    for (int species : {0, 1})
        for (double delta : {-1e-7, 0., 1e-7})
        {
            CAPTURE(species, delta);
            auto a = F::cluster(2);
            a.chemmap_ = {0, species};
            a.pos_[0] = {0., 0., 0.};
            a.pos_[1] = {double(species ? 5 : 4) + delta, 0., 0.};
            Neighborhoods nbhs(a, model.cutoff);
            Potentials::MTPGpuBatch batch;
            REQUIRE(batch.build_bounded({&a}, {&nbhs}, size_t{16} << 20, &error));
            const auto v = F::direction(a.size());
            const auto reference = C::components(model, a, v);
            std::vector<double> output;
            REQUIRE(batch.linear_hvp_block(owner, v.data(), v.size(), 0, v.size(), columns, output,
                                           size_t{64} << 20, size_t{16} << 20, &error));
            component_close(unpack(output, v.size(), columns.size()),
                            selected(reference, 0, v.size(), columns), v);
        }
}

TEST_CASE("Dedicated GPU curvature fits retain both CPU algorithms and analytic objectives",
          "[gpu][linear-curvature][training]")
{
    REQUIRE_CURVATURE_GPU();
    const bool fast = GENERATE(false, true);
    const int kind = GENERATE(0, 1, 2);
    CAPTURE(fast, kind);
    C::Sample sample(kind, kind == 1 ? 2 : 1);
    C::Options options;
    options.fast = fast;
    options.fixed_species = kind != 2;
    options.diagonal = kind == 1;
    options.norm_p = kind == 2 ? 1.4 : 0.;
    if (kind == 0)
        options.hvp = 0.;
    if (kind == 1)
        options.hessian = 0.;
    if (kind != 2)
        options.energy = options.force = options.virial = 0.;
    auto cpu_model = sample.initial(), gpu_model = C::copy(*cpu_model);
    const auto original = C::coefficients(*gpu_model);
    Trainer_MTP cpu(options.settings()), gpu(options.settings(true));
    cpu.init(sample.data, sample.neighbors, cpu_model.get());
    gpu.init(sample.data, sample.neighbors, gpu_model.get());
    auto before = C::copy(*gpu_model);
    cpu.TrainLinear();
    gpu.TrainLinear();
    const auto &report = gpu.GetGpuLinearReport();
    REQUIRE(report.verified);
    REQUIRE(report.backend == "cuda");
    REQUIRE(report.algorithm == (fast ? "fast/eigen_ldlt" : "ordinary/alglib"));
    REQUIRE(report.effective_l2.size() == size_t(gpu_model->lin_coeff_size));
    for (int j = 0; j < gpu_model->lin_coeff_size; ++j)
        close(report.effective_l2[j], cpu_model->reg_L2_vector[j], 1e-12, 2e-8);
    before->reg_L2_vector = report.effective_l2;
    close(report.data_before, C::data_loss(*before, sample.data, sample.neighbors, options));
    close(report.data_after, C::data_loss(*gpu_model, sample.data, sample.neighbors, options));
    close(report.objective_before, C::objective(*before, sample.data, sample.neighbors, options));
    close(report.objective_after, C::objective(*gpu_model, sample.data, sample.neighbors, options));
    close(report.objective_after, C::objective(*cpu_model, sample.data, sample.neighbors, options));
    close(report.predicted_data_before, report.data_before);
    close(report.predicted_data_after, report.data_after);
    REQUIRE(report.objective_after <=
            report.objective_before + 1e-10 * std::max(1., std::abs(report.objective_before)));
    REQUIRE(report.curvature_stats.probes > 0);
    REQUIRE(report.curvature_stats.blocks > 0);
    REQUIRE(report.curvature_stats.download_bytes > 0);
    REQUIRE(report.curvature_stats.output_workspace_bytes <= report.options.gpu_workspace_bytes);
    REQUIRE(report.curvature_stats.tracked_device_bytes <=
            report.options.gpu_memory_bytes + report.options.gpu_workspace_bytes);
    REQUIRE(report.curvature_host_cap_bytes == size_t{16} << 20);
    REQUIRE(report.curvature_host_peak_bytes <= report.curvature_host_cap_bytes);
    REQUIRE(report.curvature_stats.host_workspace_bytes <= report.curvature_host_cap_bytes);
    fit_predictions(*gpu_model, *cpu_model, sample.data);
    auto reload = C::copy(*gpu_model);
    fit_predictions(*reload, *gpu_model, sample.data);
    const auto final = C::coefficients(*gpu_model);
    REQUIRE((final.head(gpu_model->radial_coeff_size).array() ==
             original.head(gpu_model->radial_coeff_size).array())
                .all());
    if (options.fixed_species)
        for (int j = 0; j < gpu_model->species_count; ++j)
            REQUIRE(gpu_model->species_coeffs[j] == original[gpu_model->radial_coeff_size + j]);
    for (double seconds : {report.assembly_seconds, report.solve_seconds,
                           report.verification_seconds, report.total_seconds})
    {
        REQUIRE(std::isfinite(seconds));
        REQUIRE(seconds >= 0.);
    }
    REQUIRE(report.verification_seconds > 0.);
    if (fast)
        REQUIRE(report.host_design_bytes > 0.);
}

TEST_CASE("Mixed GPU curvature fits use the independently orthogonalized CPU anchor",
          "[gpu][linear-curvature][training][orthogonalization]")
{
    REQUIRE_CURVATURE_GPU();
    const bool fast = GENERATE(false, true);
    CAPTURE(fast);
    C::Sample sample(2, 1);
    C::Options options;
    options.fast = fast;
    options.fixed_species = false;
    options.diagonal = true;
    auto cpu_model = sample.initial(), gpu_model = C::copy(*cpu_model);
    auto cpu_settings = options.settings(), gpu_settings = options.settings(true);
    cpu_settings.Modify("do_orthogonalization", "true");
    gpu_settings.Modify("do_orthogonalization", "true");
    Trainer_MTP cpu(cpu_settings), gpu(gpu_settings);
    cpu.init(sample.data, sample.neighbors, cpu_model.get());
    gpu.init(sample.data, sample.neighbors, gpu_model.get());
    const auto original = C::coefficients(*gpu_model);
    // This separate native CPU model defines the before point, including
    // physical analytic H*v. It never consumes the GPU observation blocks.
    auto anchor = C::copy(*cpu_model);
    anchor->Orthogonalize();
    const auto anchor_coefficients = C::coefficients(*anchor);
    REQUIRE((anchor_coefficients.head(anchor->radial_coeff_size) -
             original.head(anchor->radial_coeff_size))
                .norm() > 1e-8);
    const double data_before = C::data_loss(*anchor, sample.data, sample.neighbors, options);
    cpu.TrainLinear();
    gpu.TrainLinear();
    const auto &report = gpu.GetGpuLinearReport();
    REQUIRE(report.verified);
    REQUIRE(report.backend == "cuda");
    REQUIRE(report.algorithm == (fast ? "fast/eigen_ldlt" : "ordinary/alglib"));
    REQUIRE(report.effective_l2.size() == size_t(gpu_model->lin_coeff_size));
    for (int j = 0; j < gpu_model->lin_coeff_size; ++j)
        close(report.effective_l2[j], cpu_model->reg_L2_vector[j], 1e-12, 2e-8);
    anchor->reg_L2_vector = report.effective_l2;
    close(report.data_before, data_before);
    close(report.data_after, C::data_loss(*gpu_model, sample.data, sample.neighbors, options));
    close(report.objective_before, C::objective(*anchor, sample.data, sample.neighbors, options));
    close(report.objective_after, C::objective(*gpu_model, sample.data, sample.neighbors, options));
    close(report.objective_after, C::objective(*cpu_model, sample.data, sample.neighbors, options));
    close(report.predicted_data_before, report.data_before);
    close(report.predicted_data_after, report.data_after);
    REQUIRE(report.objective_after <=
            report.objective_before + 1e-10 * std::max(1., std::abs(report.objective_before)));
    REQUIRE(report.gpu_stats.observation_blocks > 0);
    REQUIRE(report.curvature_stats.blocks > 0);
    REQUIRE(report.curvature_stats.output_workspace_bytes <= report.options.gpu_workspace_bytes);
    REQUIRE(report.curvature_host_peak_bytes <= report.curvature_host_cap_bytes);
    const auto final = C::coefficients(*gpu_model), cpu_final = C::coefficients(*cpu_model);
    for (int j = 0; j < gpu_model->radial_coeff_size; ++j)
    {
        close(final[j], anchor_coefficients[j], 1e-12, 0.);
        close(final[j], cpu_final[j], 1e-12, 0.);
    }
    fit_predictions(*gpu_model, *cpu_model, sample.data);
}

TEST_CASE("Dedicated curvature failures preserve the model and allow budget retry",
          "[gpu][linear-curvature][training][rollback]")
{
    REQUIRE_CURVATURE_GPU();
    const bool fast = GENERATE(false, true);
    C::Sample sample;
    C::Options options;
    options.fast = fast;
    auto model = sample.initial();
    Trainer_MTP gpu(options.settings(true));
    gpu.init(sample.data, sample.neighbors, model.get());
    const auto initial = C::coefficients(*model);
    const auto l1 = model->reg_L1_vector, l2 = model->reg_L2_vector;
    for (const auto &[key, value] :
         std::vector<std::pair<std::string, std::string>>{{"gpu_batch_atoms", "1"},
                                                          {"gpu_id", "2147483647"},
                                                          {"hvp_weight", ".74"},
                                                          {"hessian_pow", "1.3"}})
    {
        CAPTURE(key, value);
        GPSettings settings;
        settings.Modify(key, value);
        gpu.ApplySettings(settings);
        REQUIRE_THROWS(gpu.TrainLinear());
        REQUIRE_FALSE(gpu.GetGpuLinearReport().verified);
        REQUIRE((C::coefficients(*model).array() == initial.array()).all());
        REQUIRE(model->reg_L1_vector == l1);
        REQUIRE(model->reg_L2_vector == l2);
        gpu.ApplySettings(options.settings(true));
    }
    auto extra = gpu.training_set_.back();
    gpu.training_set_.push_back(extra);
    REQUIRE_THROWS(gpu.TrainLinear());
    REQUIRE((C::coefficients(*model).array() == initial.array()).all());
    gpu.training_set_.pop_back();
    gpu.TrainLinear();
    REQUIRE(gpu.GetGpuLinearReport().verified);
    auto reference = sample.initial();
    Trainer_MTP cpu(options.settings());
    cpu.init(sample.data, sample.neighbors, reference.get());
    cpu.TrainLinear();
    fit_predictions(*model, *reference, sample.data);
    // A bounded 1.1 MiB metadata payload crosses a one MiB owned-copy budget
    // without a huge geometry, Hessian or process memory pressure.
    auto small = std::vector<Atoms>{sample.data.front()};
    small[0].features_["budget"] = std::string(1100 * 1024, 'x');
    std::vector<Neighborhoods> neighborhoods{sample.neighbors.front()};
    auto fresh = sample.initial();
    const auto unchanged = C::coefficients(*fresh);
    auto capped = options.settings(true);
    capped.Modify("gpu_curvature_host_mib", "1");
    Trainer_MTP rejected(capped);
    REQUIRE_THROWS(rejected.init(small, neighborhoods, fresh.get()));
    REQUIRE((C::coefficients(*fresh).array() == unchanged.array()).all());
    rejected.ApplySettings(options.settings(true));
    REQUIRE_NOTHROW(rejected.init(small, neighborhoods, fresh.get()));
    REQUIRE_NOTHROW(rejected.TrainLinear());
    REQUIRE(rejected.GetGpuLinearReport().verified);
    REQUIRE(small[0].features_.at("budget") == std::string(1100 * 1024, 'x'));
}

namespace
{
namespace fs = std::filesystem;

struct Workspace
{
    fs::path original = fs::current_path(), directory;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;

    Workspace()
    {
        const char *tmp = std::getenv("TMPDIR");
        const fs::path parent = tmp ? fs::path(tmp) : original / "build";
        fs::create_directories(parent);
        std::string pattern = (parent / "gpml-linear-curvature-XXXXXX").string();
        std::vector<char> name(pattern.begin(), pattern.end());
        name.push_back('\0');
        const char *path = mkdtemp(name.data());
        if (!path)
            throw std::runtime_error("Cannot create disk CLI fixture");
        directory = path;
        fs::current_path(directory);
    }

    ~Workspace()
    {
        CMDOption::global_cmdopts = saved;
        std::error_code error;
        fs::current_path(original, error);
        fs::remove_all(directory, error);
    }
};

struct Capture
{
    std::ostringstream stream;
    std::streambuf *buffer = std::cout.rdbuf(stream.rdbuf());
    std::ostream *output = SetStreamForOutput(&stream);

    ~Capture()
    {
        SetStreamForOutput(output);
        std::cout.rdbuf(buffer);
    }
};

int invoke()
{
    CMDOption::global_cmdopts = CMDOption::CommandLineOptions();
    std::vector<std::string> args{"trainmtp",
                                  "--output=_Trained_Potential_.mtp",
                                  "--linear-fit",
                                  "--potfile=input.mtp",
                                  "--trainset=train.bin",
                                  "--trainset-format=bin",
                                  "--index=index.map",
                                  "--settings=fit.ini"};
    std::vector<char *> argv;
    for (auto &arg : args)
        argv.push_back(arg.data());
    return trainmtp(argv.size(), argv.data());
}

void cli_settings(const C::Options &options, bool gpu, bool reject = false)
{
    std::ofstream output("fit.ini");
    auto settings = options.settings(gpu);
    if (reject)
        settings.Modify("weight_L1", ".1");
    for (const auto &[key, value] : settings.settings)
        output << "fit:" << key << ' ' << value << '\n';
}

std::string contents(const fs::path &path)
{
    std::ifstream input(path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
}

TEST_CASE("trainmtp curvature CLI publishes independently verified reloadable native models",
          "[gpu][linear-curvature][cli]")
{
    REQUIRE_CURVATURE_GPU();
    const bool fast = GENERATE(false, true);
    C::Sample sample(2, 4, 4);
    C::Options options;
    options.fast = fast;
    auto initial = sample.initial();
    auto data = sample.data;
    for (auto &a : data)
    {
        a.global_attribs_d_["fitness"] = a.fitness;
        a.local_attribs_d_["atomic_fitness"] =
            Eigen::Map<Eigen::VectorXd>(a.atomic_fitness.data(), a.size());
    }
    Workspace work;
    initial->Save("input.mtp");
    GPIO io;
    io.setmap();
    REQUIRE(io.write(data, "train.bin", "bin"));
    {
        std::ofstream output("index.map");
        const auto map = io.getmap();
        output << map.at(0) << '\n' << map.at(1) << '\n';
    }
    cli_settings(options, false);
    REQUIRE(invoke() == 0);
    fs::rename("_Trained_Potential_.mtp", "cpu.mtp");
    cli_settings(options, true);
    fs::create_directory("_Trained_Potential_.mtp");
    std::ofstream("_Trained_Potential_.mtp/keep") << "keep";
    REQUIRE_THROWS(invoke());
    REQUIRE(contents("_Trained_Potential_.mtp/keep") == "keep");
    fs::remove_all("_Trained_Potential_.mtp");
    std::string log;
    {
        Capture capture;
        REQUIRE(invoke() == 0);
        log = capture.stream.str();
    }
    const std::string marker = "trainmtp GPU linear report ";
    const auto at = log.find(marker);
    REQUIRE(at != std::string::npos);
    const auto begin = at + marker.size();
    picojson::value report;
    REQUIRE(picojson::parse(report, log.substr(begin, log.find('\n', begin) - begin)).empty());
    REQUIRE(report.get("verified").get<bool>());
    REQUIRE(report.get("acceptance_backend").get<std::string>() == "cpu/direct");
    REQUIRE(report.get("solve_backend").get<std::string>() ==
            (fast ? "cpu/fast/eigen_ldlt" : "cpu/ordinary/alglib"));
    PotMTP cpu("cpu.mtp"), gpu("_Trained_Potential_.mtp");
    cpu.train_init();
    gpu.train_init();
    cpu.set_inited();
    gpu.set_inited();
    fit_predictions(gpu, cpu, data);
    close(report.get("data_after").get<double>(),
          C::data_loss(gpu, data, sample.neighbors, options));
    const auto saved = contents("_Trained_Potential_.mtp");
    cli_settings(options, true, true);
    REQUIRE_THROWS(invoke());
    REQUIRE(contents("_Trained_Potential_.mtp") == saved);
    for (const auto &entry : fs::directory_iterator("."))
        REQUIRE_FALSE(entry.path().filename().string().starts_with(".gpml-affine-save-"));
}
#endif
