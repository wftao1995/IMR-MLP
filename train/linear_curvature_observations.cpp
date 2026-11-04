// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "linear_curvature_observations.h"
#include "analyse/gpu_analytic_hvp.h"
#include "potentials/mtp_gpu_adapter.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <typeinfo>

namespace Training
{
namespace
{
size_t add(size_t a, size_t b)
{
    if (b > std::numeric_limits<size_t>::max() - a)
        GPERROR("GPU linear curvature size overflow");
    return a + b;
}

size_t mul(size_t a, size_t b)
{
    if (a && b > std::numeric_limits<size_t>::max() / a)
        GPERROR("GPU linear curvature size overflow");
    return a * b;
}

void fit(size_t bytes, size_t cap)
{
    if (bytes > cap)
        GPERROR("GPU linear curvature exceeds the owned host memory budget");
}
#ifdef IMRMLP_GPU
template <class F> void timed(double &seconds, F &&operation)
{
    const auto begin = std::chrono::steady_clock::now();
    operation();
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now() - begin).count();
}
#endif
}

void visit_gpu_linear_curvature(
    Potentials::PotMTP &pot, const std::vector<Atoms> &atoms,
    const std::vector<std::array<bool, 2>> &requested, std::span<const int> native_columns,
    const LinearEFSOptions &options, LinearCurvatureStatistics &stats,
    const std::function<void(const LinearCurvatureObservationBlock &)> &consume,
    const std::vector<Neighborhoods> *neighborhoods, size_t host_cap, size_t external_host_bytes)
{
    options.validate();
    if (options.device != "cuda")
        GPERROR("GPU linear curvature currently requires device=cuda");
    if (!host_cap)
        GPERROR("GPU linear curvature requires a positive host budget");
    fit(external_host_bytes, host_cap);
    if (typeid(pot) != typeid(Potentials::PotMTP) || pot.radial_mode != 0 || pot.hc_correction ||
        pot.rR_correction || pot.add_efs)
        GPERROR("GPU linear curvature requires ordinary radial_mode=0 MTP without corrections");
    if (requested.size() != atoms.size() ||
        (neighborhoods && neighborhoods->size() != atoms.size()))
        GPERROR("GPU linear curvature requires one mask and neighborhood set per structure");
    if (pot.lin_coeff_size <= 0 || native_columns.size() > size_t(std::numeric_limits<int>::max()))
        GPERROR("Invalid GPU linear curvature column count");
    for (int column : native_columns)
        if (column < 0 || column >= pot.lin_coeff_size)
            GPERROR("Invalid native GPU linear curvature column");
    HessianGpu::validate_analytic_radial_bases(pot);
    HessianExecutionSettings geometry_settings;
    geometry_settings.device = options.device;
    geometry_settings.gpu_id = options.gpu_id;
    geometry_settings.gpu_batch_atoms = options.gpu_batch_atoms;
    bool active = false;
    // Validate labels and geometry for all requested frames before publication.
    // Supplied neighbor contents are additionally checked when each frame packs.
    for (size_t k = 0; k < atoms.size(); ++k)
    {
        if (!requested[k][0] && !requested[k][1])
            continue;
        const auto dof = HessianGpu::checked_analytic_hvp_dof(atoms[k].pos_.size());
        if (!dof)
            GPERROR("GPU linear curvature requires nonempty structures");
        HessianGpu::validate(atoms[k], pot, pot.cutoff, geometry_settings);
        if (atoms[k].hvp_directions_.cols() > std::numeric_limits<int>::max() ||
            atoms[k].hvp_targets_.cols() > std::numeric_limits<int>::max())
            GPERROR("GPU linear curvature probe count exceeds the index range");
        if (!atoms[k].cartesian)
            for (const auto &position : atoms[k].pos_)
            {
                const Eigen::RowVector3d cartesian = position * atoms[k].cell_;
                if (!cartesian.allFinite())
                    GPERROR("GPU linear curvature Cartesian conversion is nonfinite");
            }
        atoms[k].validate_hvp();
        if (requested[k][0] && (!atoms[k].has_hessian() || !atoms[k].hessian_.allFinite()))
            GPERROR("GPU linear curvature Hessian mask requires finite 3N by 3N labels");
        if (requested[k][1] && !atoms[k].has_hvp())
            GPERROR("GPU linear curvature HVP mask requires labeled directions");
        if (neighborhoods && (*neighborhoods)[k].size() != atoms[k].size())
            GPERROR("GPU linear curvature neighborhood count differs from the structure");
        active = true;
    }
    if (!active || native_columns.empty())
        return;
    if (!consume)
        GPERROR("GPU linear curvature requires a consumer callback");
#ifdef IMRMLP_GPU
    const size_t columns_bytes = mul(native_columns.size(), sizeof(int));
    fit(add(external_host_bytes, columns_bytes), host_cap);
    const std::vector<int> columns(native_columns.begin(), native_columns.end());
    const size_t fixed = add(external_host_bytes, mul(columns.capacity(), sizeof(int)));
    fit(fixed, host_cap);
    const size_t device_cap = add(options.gpu_memory_bytes, options.gpu_workspace_bytes);
    Potentials::MTPGpuEvaluator evaluator;
    std::string error;
    auto require = [&](bool ok)
    {
        if (!ok)
            GPERROR("GPU linear curvature: " + error);
    };
    timed(stats.gpu_initialize_seconds,
          [&]
          {
              require(evaluator.initialize_bounded(pot, options.gpu_id, device_cap, &error));
          });
    if (!evaluator.supports_analytic_hvp() ||
        evaluator.linear_observation_column_count() != pot.lin_coeff_size)
        GPERROR("GPU linear curvature model does not support native analytic columns");
    auto record_host = [&](size_t bytes)
    {
        fit(bytes, host_cap);
        stats.host_workspace_bytes = std::max(stats.host_workspace_bytes, bytes);
    };
    for (size_t k = 0; k < atoms.size(); ++k)
    {
        if (!requested[k][0] && !requested[k][1])
            continue;
        const auto &input = atoms[k];
        const size_t dof = HessianGpu::checked_analytic_hvp_dof(input.pos_.size());
        const size_t vector_bytes = mul(dof, sizeof(double));
        size_t live = fixed;
        Atoms geometry;
        Neighborhoods owned_neighbors;
        const AnyNeighborhoods *neighbors = nullptr;
        Potentials::MTPGpuBatch batch;
        std::vector<double> unit;
        timed(stats.geometry_seconds,
              [&]
              {
                  record_host(add(live, HessianGpu::analytic_geometry_bytes(input, true)));
                  geometry.cell_ = input.cell_;
                  geometry.pos_ = input.pos_;
                  geometry.chemmap_ = input.chemmap_;
                  geometry.pbc_ = input.pbc_;
                  geometry.cartesian = input.cartesian;
                  if (!geometry.cartesian)
                      geometry.to_cartesian();
                  for (const auto &position : geometry.pos_)
                      if (!position.allFinite())
                          GPERROR("GPU linear curvature Cartesian conversion is nonfinite");
                  geometry.cartesian = true;
                  live = add(live, HessianGpu::analytic_geometry_bytes(geometry));
                  record_host(live);
                  if (neighborhoods)
                      neighbors = &(*neighborhoods)[k];
                  else
                  {
                      record_host(add(live, Neighborhoods::construction_memory_bound(
                                                geometry, pot.cutoff, host_cap - live)));
                      owned_neighbors.init(geometry, pot.cutoff);
                      live = add(live, owned_neighbors.owned_memory_bytes());
                      record_host(live);
                      neighbors = &owned_neighbors;
                  }
                  if (requested[k][0])
                  {
                      record_host(add(live, vector_bytes));
                      unit.assign(dof, 0.);
                      live = add(live, mul(unit.capacity(), sizeof(double)));
                      record_host(live);
                  }
                  // One bounded pack; caller labels and all other probes stay borrowed.
                  const size_t pointer_bytes =
                      sizeof(const Atoms *) + sizeof(const AnyNeighborhoods *);
                  record_host(add(live, pointer_bytes));
                  require(batch.build_bounded({&geometry}, {neighbors},
                                              host_cap - live - pointer_bytes, &error));
                  record_host(add(add(live, pointer_bytes), batch.owned_host_bytes()));
              });
        const size_t packed = batch.owned_host_bytes();
        const size_t minimum = add(add(live, packed), vector_bytes);
        record_host(minimum);
        const size_t row_bytes = mul(columns.size(), sizeof(double));
        if (options.gpu_workspace_bytes <= vector_bytes ||
            row_bytes > options.gpu_workspace_bytes - vector_bytes ||
            row_bytes > host_cap - minimum)
            GPERROR("GPU linear curvature workspace cannot fit one selected row and direction");
        const size_t row_limit =
            std::min({dof, (host_cap - minimum) / row_bytes,
                      (options.gpu_workspace_bytes - vector_bytes) / row_bytes});
        auto emit_probe = [&](bool hessian, int probe, const double *direction, size_t total_rows)
        {
            ++stats.probes;
            for (size_t first = 0; first < total_rows;)
            {
                const int rows = static_cast<int>(std::min(row_limit, total_rows - first));
                std::vector<double> output;
                timed(stats.observation_seconds,
                      [&]
                      {
                          require(batch.linear_hvp_block(
                              evaluator, direction, dof, static_cast<int>(first), rows, columns,
                              output, device_cap, host_cap - live, &error));
                      });
                record_host(add(live, batch.last_linear_hvp_host_bytes()));
                const auto memory = evaluator.context().memory_stats();
                if (memory.tracked_device_bytes > device_cap ||
                    memory.analytic_hvp_io_bytes > options.gpu_workspace_bytes)
                    GPERROR("GPU linear curvature exceeded its device allocation budgets");
                stats.tracked_device_bytes =
                    std::max(stats.tracked_device_bytes, memory.tracked_device_bytes);
                stats.output_workspace_bytes =
                    std::max(stats.output_workspace_bytes, memory.analytic_hvp_io_bytes);
                ++stats.blocks;
                stats.download_bytes =
                    add(stats.download_bytes, mul(output.size(), sizeof(double)));
                const LinearCurvatureObservationBlock block{
                    k, hessian, probe, static_cast<int>(first), rows, columns, output};
                timed(stats.accumulation_seconds,
                      [&]
                      {
                          consume(block);
                      });
                first += size_t(rows);
            }
        };
        if (requested[k][0])
            for (size_t j = 0; j < dof; ++j)
            {
                unit[j] = 1.;
                emit_probe(true, static_cast<int>(j), unit.data(), j + 1);
                unit[j] = 0.;
            }
        if (requested[k][1])
            for (Eigen::Index j = 0; j < input.hvp_directions_.cols(); ++j)
                emit_probe(false, static_cast<int>(j), input.hvp_directions_.col(j).data(), dof);
    }
#else
    (void)stats;
    GPERROR("GPU linear curvature is unavailable in this CPU build");
#endif
}
} // namespace Training
