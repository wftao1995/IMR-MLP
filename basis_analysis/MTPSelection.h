// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potentials/potential_mtp.h"
#include <array>
#include <memory>
#include <random>
#include <vector>
#include <utility>

namespace BasisAnalysis::MTPSelection
{
using Potentials::BasisCostStats;
using Potentials::PotMTP;

// A group contains every scalar that depends on a particular basic moment.
// Reuse the compactor's dependency closure, including all additive rules for
// each target. Identical groups (e.g. Cartesian tensor components) occur once.
std::vector<std::vector<int>> dependency_groups(const PotMTP &pot);

// Delete all scalar outputs that transitively use a radial channel. The
// closure comes from the compactor, so shared/additive product rules cannot
// leave a supposedly removed channel alive. Keep this family separate from
// basic groups: uniform sampling of the union would underweight channels.
std::vector<std::vector<int>> radial_dependency_groups(const PotMTP &pot);

// Only choose groups that remove at least one scalar and leave a nonempty
// model. Crossover and scalar mutation can reintroduce deleted dependencies.
bool drop_dependency_group(std::vector<bool> &chrom, const std::vector<std::vector<int>> &groups,
                           std::mt19937 &rng);

// Ordering only: suppress insignificant reduction/BLAS roundoff. Feasibility
// and reported losses always use the original, unrounded value.
double order_loss(double loss);

double tolerance(double limit);

bool feasible(double loss, double limit);

double violation(double loss, double limit);

double sa_score(double loss, double relative_cost, double limit);

bool dominates(double loss_a, double cost_a, double loss_b, double cost_b, double limit);

struct Entry
{
    std::vector<int> ids;
    double train_loss = 0, valid_loss = 0, relative = 0; // losses per configuration
    BasisCostStats stats;
};

// Cost-spread representatives, preserving cheapest and most accurate entries.
// Index/ID tie-breaks are independent of wall-clock measurements.
std::vector<Entry> representatives(std::vector<Entry> entries, std::size_t count);

class Archive
{
    std::size_t capacity_;
    double limit_;
    bool pareto_;
    std::vector<Entry> entries_;

  public:
    Archive(std::size_t capacity, double limit, bool pareto = false)
        : capacity_(capacity), limit_(limit), pareto_(pareto)
    {
    }

    const std::vector<Entry> &entries() const
    {
        return entries_;
    }

    void add(Entry entry);

    std::vector<Entry> shortlist(std::size_t count) const;
};

struct Workload
{
    std::vector<Atoms> frames;
    std::vector<int> global_ids;
    double active = 0, stored = 0, max_neighbors = 0;
    std::size_t atoms = 0;
};

// Stratify by composition, log2 atom count and log2 mean active coordination.
// Only rank 0 calls this; selection uses original global frame indices.
Workload sample_workload(PotMTP &pot, std::vector<Atoms> &pool, int count, int seed);

struct Timing
{
    double median = 0, low = 0, high = 0;
};

Timing summarize(std::vector<double> times);

struct Measurement
{
    Timing kernel, total;
    std::vector<double> kernel_rounds, total_rounds;
};

// Nondominated feasible entries for one specified cost metric. Ties in the
// rounded loss and exact cost retain the lexicographically smallest mask.
std::vector<std::size_t> pareto_front(const std::vector<Entry> &entries,
                                      const std::vector<double> &costs, double limit);

// Timing-overlap heuristic, not a statistical significance test. Compare to
// the fastest median to avoid non-transitive interval-based sorting.
std::size_t choose_measured(const std::vector<Entry> &entries,
                            const std::vector<Measurement> &times, double limit);

std::vector<Measurement> benchmark(std::vector<std::unique_ptr<PotMTP>> &models, Workload &work,
                                   int rounds, double seconds, int seed);
double StableCURResidual(const Eigen::MatrixXd &M, const Eigen::MatrixXd &C, int &rank);

double CURRoundoffTolerance(const Eigen::MatrixXd &M);

int SelectKneePoint(const std::vector<std::pair<double, double>> &objectives);

std::vector<double> CrowdingDistances(const std::vector<std::pair<double, double>> &objectives);

} // namespace BasisAnalysis::MTPSelection
