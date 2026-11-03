// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "linear_efs_observations.h"
#include "potentials/mtp_gpu_adapter.h"
#include <algorithm>
#include <chrono>
#include <limits>
#include <typeinfo>

namespace Training {

void LinearEFSOptions::validate() const {
    using namespace MLIPCore::GpuDirect;
    if (!valid_device(device)) GPERROR("Linear EFS device must be cpu, cuda, hip or dcu");
    if (!device_supported(device))
        GPERROR("Linear EFS device=" + device + " is unavailable in this " + compiled_backend() + " build");
    if (gpu_id < -1 || gpu_batch_atoms <= 0 || gpu_batch_structures <= 0 ||
        !gpu_memory_bytes || !gpu_workspace_bytes ||
        gpu_workspace_bytes > std::numeric_limits<size_t>::max()-gpu_memory_bytes)
        GPERROR("Invalid GPU linear EFS device, batch limit or memory budget");
}

namespace {
#ifdef IMRMLP_GPU
template<class F> void timed(double &seconds, F &&operation) {
    const auto start = std::chrono::steady_clock::now();
    operation();
    seconds += std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
#endif
}

void visit_gpu_linear_efs(Potentials::PotMTP &pot,
    const std::vector<Atoms> &atoms,
    const std::vector<std::array<bool, 3>> &requested,
    std::span<const int> native_columns,
    const LinearEFSOptions &options, LinearEFSStatistics &stats,
    const std::function<void(std::span<const std::size_t>)> &begin_chunk,
    const std::function<void(const LinearEFSObservationBlock &)> &consume,
    const std::vector<Neighborhoods> *neighborhoods) {
    options.validate();
    if (options.device == "cpu") GPERROR("GPU linear observation visitor requires an explicit GPU backend");
    if (typeid(pot) != typeid(Potentials::PotMTP) || pot.radial_mode != 0)
        GPERROR("GPU linear assembly requires ordinary PotMTP with radial_mode=0");
    if (requested.size() != atoms.size() || (neighborhoods && neighborhoods->size() != atoms.size()))
        GPERROR("GPU linear observations require one channel mask and neighborhood set per structure");
    if (pot.lin_coeff_size <= 0 || native_columns.size() > size_t(std::numeric_limits<int>::max()))
        GPERROR("Invalid GPU linear observation column count");
    for (int column : native_columns)
        if (column < 0 || column >= pot.lin_coeff_size)
            GPERROR("Invalid native GPU linear observation column");
    if (native_columns.empty()) return;
#ifdef IMRMLP_GPU
    std::vector<size_t> active;
    for (size_t i = 0; i < atoms.size(); ++i) {
        if (!requested[i][0] && !requested[i][1] && !requested[i][2]) continue;
        if (atoms[i].size() == 0) GPERROR("GPU linear observations require nonempty structures");
        if (atoms[i].size() > options.gpu_batch_atoms)
            GPERROR("Structure exceeds gpu-batch-atoms; increase the explicit limit");
        if (neighborhoods && i > size_t(std::numeric_limits<int>::max()))
            GPERROR("GPU linear observation structure index exceeds adapter limits");
        active.push_back(i);
    }
    if (active.empty()) return;
    if (!consume) GPERROR("GPU linear observations require a consumer callback");
    Potentials::MTPGpuEvaluator evaluator;
    Potentials::MTPGpuBatch batch;
    std::string error;
    auto require = [&](bool ok) { if (!ok) GPERROR("GPU linear assembly: " + error); };
    timed(stats.gpu_initialize_seconds, [&] {
        int devices = 0;
        require(MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&devices, &error));
        if (!devices || options.gpu_id >= devices)
            GPERROR("GPU linear assembly device index is outside the visible device count");
        require(evaluator.initialize(pot, options.gpu_id, &error));
    });
    if (evaluator.linear_observation_column_count() != pot.lin_coeff_size)
        GPERROR("GPU linear assembly does not support this model's linear columns");
    const size_t columns = native_columns.size();
    const size_t moments = evaluator.context().potential_stats().active_moment_slots;
    const bool scalar = std::any_of(native_columns.begin(), native_columns.end(),
        [&](int column) { return column >= pot.species_count; });
    const std::vector<int> selected(native_columns.begin(), native_columns.end());
    auto sample_memory = [&] {
        const auto memory = evaluator.context().memory_stats();
        if (memory.linear_observation_bytes > options.gpu_workspace_bytes ||
            memory.tracked_device_bytes < memory.linear_observation_bytes ||
            memory.tracked_device_bytes-memory.linear_observation_bytes > options.gpu_memory_bytes)
            GPERROR("GPU linear assembly exceeded its tracked memory budgets");
        stats.tracked_device_bytes = std::max(stats.tracked_device_bytes, memory.tracked_device_bytes);
        stats.observation_workspace_bytes = std::max(stats.observation_workspace_bytes, memory.linear_observation_bytes);
    };
    // Match the core's aligned offset/column headers, selected-column output
    // and one-column reverse scratch. Species-only derivatives need no scratch.
    // Shrink batches before repeating derivative work for smaller row blocks.
    auto row_limit = [&](size_t centers, size_t structures, int channel) -> size_t {
        const size_t budget = options.gpu_workspace_bytes;
        size_t fixed, per_row;
        if (channel == 0) {
            fixed = (columns+1)*sizeof(int);
            per_row = sizeof(int)+columns*sizeof(double);
            if (fixed > budget) return 0;
            size_t rows = std::min((budget-fixed)/per_row, size_t(std::numeric_limits<int>::max())/columns);
            const size_t header = ((rows+1+columns)*sizeof(int)+7)&~size_t(7);
            if (rows && (header > budget || rows*columns > (budget-header)/sizeof(double))) --rows;
            return rows;
        }
        fixed = ((structures+1)*sizeof(int)+7)&~size_t(7);
        const size_t per_center = scalar ? (moments+(channel == 1 ? 3 : 9))*sizeof(double) : 0;
        if (fixed > budget || (per_center && centers > (budget-fixed)/per_center)) return 0;
        fixed += centers*per_center;
        per_row = columns*sizeof(double);
        return std::min((budget-fixed)/per_row, size_t(std::numeric_limits<int>::max())/columns);
    };
    for (size_t begin = 0; begin < active.size();) {
        size_t end = begin, centers = 0;
        std::array<bool, 3> needed{};
        for (; end < active.size() && end-begin < size_t(options.gpu_batch_structures); ++end) {
            const size_t next_centers = centers+size_t(atoms[active[end]].size()), structures = end-begin+1;
            if (next_centers > size_t(options.gpu_batch_atoms)) break;
            auto next_needed = needed;
            for (int channel = 0; channel < 3; ++channel)
                next_needed[channel] = next_needed[channel] || requested[active[end]][channel];
            const size_t total_rows[] = {structures, 3*next_centers, 9*structures};
            bool full = true;
            for (int channel = 0; channel < 3; ++channel)
                if (next_needed[channel] && row_limit(next_centers, structures, channel) < total_rows[channel]) full = false;
            if (end > begin && !full) break;
            centers = next_centers; needed = next_needed;
            if (!full) { ++end; break; }
        }
        const size_t structures = end-begin;
        if (!structures) GPERROR("GPU linear assembly cannot fit a structure in the batch");
        if (centers > size_t(std::numeric_limits<int>::max())/3 ||
            structures > size_t(std::numeric_limits<int>::max())/9)
            GPERROR("GPU linear assembly row count exceeds index limits");
        for (int channel = 0; channel < 3; ++channel)
            if (needed[channel] && !row_limit(centers, structures, channel))
                GPERROR("GPU linear assembly workspace cannot fit one observation row and its scratch");
        const std::span<const size_t> ids(active.data()+begin, structures);
        std::vector<int> offsets{0};
        timed(stats.neighborhoods_seconds, [&] {
            for (size_t id : ids) offsets.push_back(offsets.back()+atoms[id].size());
            if (neighborhoods) {
                const std::vector<int> indices(ids.begin(), ids.end());
                require(batch.build(atoms, *neighborhoods, indices, &error));
            } else {
                std::vector<Atoms> chunk;
                std::vector<Neighborhoods> neighbors;
                chunk.reserve(structures); neighbors.reserve(structures);
                for (size_t id : ids) {
                    chunk.push_back(atoms[id]);
                    neighbors.emplace_back(chunk.back(), pot.cutoff);
                }
                require(batch.build(chunk, neighbors, &error));
            }
        });
        timed(stats.gpu_forward_seconds, [&] {
            // Keep public budgets independent when the previous observation
            // buffer remains resident during the next forward preparation.
            const size_t retained = evaluator.context().memory_stats().linear_observation_bytes;
            require(batch.prepare_linear_observations(evaluator, options.gpu_memory_bytes+retained, &error));
        });
        sample_memory(); ++stats.batches;
        if (begin_chunk) timed(stats.accumulation_seconds, [&] { begin_chunk(ids); });
        for (int channel = 0; channel < 3; ++channel) {
            if (!needed[channel]) continue;
            const int total_rows = channel == 0 ? structures : channel == 1 ? 3*centers : 9*structures;
            const int block_rows = std::min(size_t(total_rows), row_limit(centers, structures, channel));
            size_t structure = 0;
            for (int first = 0; first < total_rows;) {
                const int rows = std::min(block_rows, total_rows-first);
                std::vector<double> output;
                timed(stats.gpu_observation_seconds, [&] {
                    if (channel == 0) require(batch.linear_energy_block(evaluator, first, rows, selected,
                        output, options.gpu_workspace_bytes, &error));
                    else if (channel == 1) require(batch.linear_force_block(evaluator, first, rows, selected,
                        output, options.gpu_workspace_bytes, &error));
                    else require(batch.linear_virial_block(evaluator, first, rows, selected,
                        output, options.gpu_workspace_bytes, &error));
                });
                ++stats.observation_blocks; stats.download_bytes += output.size()*sizeof(double);
                sample_memory();
                std::vector<LinearEFSObservationRow> references;
                references.reserve(rows);
                for (int r = 0; r < rows; ++r) {
                    const int row = first+r;
                    if (channel == 1) {
                        while (row/3 >= offsets[structure+1]) ++structure;
                    } else structure = channel == 0 ? row : row/9;
                    const size_t id = ids[structure];
                    if (!requested[id][channel]) continue;
                    const size_t destination = references.size()*columns;
                    if (references.size() != size_t(r))
                        std::copy_n(output.data()+size_t(r)*columns, columns, output.data()+destination);
                    references.push_back({id, channel, channel == 1 ? row/3-offsets[structure] : -1,
                        channel == 0 ? -1 : channel == 1 ? row%3 : row%9});
                }
                if (!references.empty()) {
                    const LinearEFSObservationBlock block{references, selected,
                        std::span<const double>(output.data(), references.size()*columns)};
                    timed(stats.accumulation_seconds, [&] { consume(block); });
                }
                first += rows;
            }
        }
        begin = end;
    }
#else
    (void)stats; (void)begin_chunk; (void)consume;
    GPERROR("GPU linear observations are unavailable in this CPU build");
#endif
}

} // namespace Training
