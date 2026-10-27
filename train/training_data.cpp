// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "training_data.h"

#include <algorithm>
#include <stdexcept>

namespace Training {

std::vector<Atoms> load_training_data(GPIO &io, const std::string &filename,
                                      const std::string &format)
{
    std::vector<Atoms> frames;
    if (!io.read(frames, filename, format))
        throw std::runtime_error("Failed to read complete training dataset: " + filename);
    return frames;
}

void restore_training_fitness(std::vector<Atoms> &frames)
{
    for (auto &frame : frames) {
        if (const auto it = frame.local_attribs_d_.find("atomic_fitness");
            it != frame.local_attribs_d_.end()) {
            if (it->second.size() != frame.size())
                throw std::invalid_argument("atomic_fitness count differs from atom count");
            frame.atomic_fitness.assign(it->second.data(), it->second.data() + it->second.size());
        }
        const auto it = frame.global_attribs_d_.find("fitness");
        frame.fitness = it == frame.global_attribs_d_.end() ? 1.0 : it->second;
    }
}

void distribute_training_data(std::vector<Atoms> &frames, bool balance_atomic_fitness)
{
#ifdef IMRMLP_MPI
    const auto active_atoms = [](const Atoms &frame) {
        if (frame.atomic_fitness.empty()) return frame.size();
        return static_cast<int>(std::count_if(frame.atomic_fitness.begin(), frame.atomic_fitness.end(),
                                             [](double weight) { return weight > 0.0; }));
    };
    std::sort(frames.begin(), frames.end(), [&](const Atoms &a, const Atoms &b) {
        if (balance_atomic_fitness && active_atoms(a) != active_atoms(b))
            return active_atoms(a) < active_atoms(b);
        return a.size() < b.size();
    });
    std::erase_if(frames, [](const Atoms &frame) { return frame.id % mpi.size != mpi.rank; });
#else
    (void)frames;
    (void)balance_atomic_fitness;
#endif
}

std::vector<Neighborhoods> training_neighborhoods(std::vector<Atoms> &frames, double cutoff)
{
    std::vector<Neighborhoods> neighborhoods;
    neighborhoods.reserve(frames.size());
    for (auto &frame : frames) {
        frame.wrap();
        neighborhoods.emplace_back(frame, cutoff);
    }
    return neighborhoods;
}

} // namespace Training
