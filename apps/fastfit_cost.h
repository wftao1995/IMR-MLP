// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "potentials/potential_mtp.h"
#include <algorithm>
#include <array>
#include <memory>
#include <limits>
#include <set>
#include <stdexcept>
#include <chrono>
#include <cmath>
#include <map>
#include <numeric>
#include <random>
#include <sstream>

namespace func_fastfit::cost {
using Potentials::BasisCostStats;
using Potentials::PotMTP;

// A group contains every scalar that depends on a particular basic moment.
// Reuse the compactor's dependency closure, including all additive rules for
// each target. Identical groups (e.g. Cartesian tensor components) occur once.
inline std::vector<std::vector<int>> dependency_groups(const PotMTP &pot) {
    std::vector<std::vector<int>> groups(pot.alpha_index_basic_count);
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        const auto analysis = pot.analyze_basis({scalar});
        for (int basic = 0; basic < pot.alpha_index_basic_count; ++basic)
            if (analysis.keep[basic]) groups[basic].push_back(scalar);
    }
    std::erase_if(groups, [&](const auto &ids) {
        return ids.empty() || ids.size() == std::size_t(pot.alpha_scalar_moments);
    });
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    return groups;
}

// Delete all scalar outputs that transitively use a radial channel. The
// closure comes from the compactor, so shared/additive product rules cannot
// leave a supposedly removed channel alive. Keep this family separate from
// basic groups: uniform sampling of the union would underweight channels.
inline std::vector<std::vector<int>> radial_dependency_groups(const PotMTP &pot) {
    std::vector<std::vector<int>> groups(pot.radial_func_count);
    for (int scalar = 0; scalar < pot.alpha_scalar_moments; ++scalar) {
        const auto analysis = pot.analyze_basis({scalar});
        std::set<int> channels;
        for (int basic = 0; basic < pot.alpha_index_basic_count; ++basic)
            if (analysis.keep[basic]) channels.insert(pot.alpha_index_basic(basic, 0));
        for (int channel : channels) groups.at(channel).push_back(scalar);
    }
    std::erase_if(groups, [&](const auto &ids) {
        return ids.empty() || ids.size() == std::size_t(pot.alpha_scalar_moments);
    });
    std::sort(groups.begin(), groups.end());
    groups.erase(std::unique(groups.begin(), groups.end()), groups.end());
    return groups;
}

// Only choose groups that remove at least one scalar and leave a nonempty
// model. Crossover and scalar mutation can reintroduce deleted dependencies.
inline bool drop_dependency_group(std::vector<bool> &chrom,
        const std::vector<std::vector<int>> &groups, std::mt19937 &rng) {
    const auto selected = std::count(chrom.begin(), chrom.end(), true);
    std::vector<std::size_t> eligible;
    for (std::size_t g = 0; g < groups.size(); ++g) {
        const auto removed = std::count_if(groups[g].begin(), groups[g].end(),
                                          [&](int id) { return chrom[id]; });
        if (removed > 0 && removed < selected) eligible.push_back(g);
    }
    if (eligible.empty()) return false;
    const auto g = eligible[std::uniform_int_distribution<std::size_t>(0, eligible.size()-1)(rng)];
    for (int id : groups[g]) chrom[id] = false;
    return true;
}

// Ordering only: suppress insignificant reduction/BLAS roundoff. Feasibility
// and reported losses always use the original, unrounded value.
inline double order_loss(double loss) {
    int exponent = 0;
    std::frexp(std::max(1., std::abs(loss)), &exponent);
    const double step = std::ldexp(1., exponent - 42);
    return std::min(std::numeric_limits<double>::max(), std::round(loss / step) * step);
}
inline double tolerance(double limit) { return 1e-12 * std::max(1., limit); }
inline bool feasible(double loss, double limit) {
    return std::isfinite(loss) && loss <= limit + tolerance(limit);
}
inline double violation(double loss, double limit) {
    return feasible(loss, limit) ? 0. : (loss - limit) / std::max(limit, tolerance(limit));
}
inline double sa_score(double loss, double relative_cost, double limit) {
    // A subset cannot exceed the full compact basis's deterministic cost.
    return feasible(loss, limit) ? std::clamp(relative_cost, 0., 1.)
        : 2. + std::log1p(violation(loss, limit));
}
inline bool dominates(double loss_a, double cost_a, double loss_b, double cost_b, double limit) {
    const bool a = feasible(loss_a, limit), b = feasible(loss_b, limit);
    if (a != b) return a;
    loss_a = order_loss(loss_a); loss_b = order_loss(loss_b);
    if (!a) return loss_a < loss_b;
    return loss_a <= loss_b && cost_a <= cost_b && (loss_a < loss_b || cost_a < cost_b);
}
struct Entry {
    std::vector<int> ids;
    double train_loss = 0, valid_loss = 0, relative = 0; // losses per configuration
    BasisCostStats stats;
};
// Cost-spread representatives, preserving cheapest and most accurate entries.
// Index/ID tie-breaks are independent of wall-clock measurements.
inline std::vector<Entry> representatives(std::vector<Entry> entries, std::size_t count) {
    if (!count || entries.empty()) return {};
    std::sort(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
        if (a.relative != b.relative) return a.relative < b.relative;
        if (order_loss(a.valid_loss) != order_loss(b.valid_loss)) return order_loss(a.valid_loss) < order_loss(b.valid_loss);
        return a.ids < b.ids;
    });
    if (entries.size() <= count) return entries;
    std::set<std::size_t> selected{0};
    if (count > 1) selected.insert(std::min_element(entries.begin(), entries.end(), [](const Entry &a, const Entry &b) {
        return order_loss(a.valid_loss) < order_loss(b.valid_loss) ||
            (order_loss(a.valid_loss) == order_loss(b.valid_loss) && a.ids < b.ids);
    }) - entries.begin());
    // Incremental nearest selected rank distances: O(pool_size * count),
    // including archive eviction where pool_size is capacity + 1.
    std::vector<std::size_t> distances(entries.size(), entries.size());
    auto update = [&](std::size_t j) {
        for (std::size_t i = 0; i < entries.size(); ++i)
            distances[i] = std::min(distances[i], i > j ? i-j : j-i);
    };
    for (auto i : selected) update(i);
    while (selected.size() < count) {
        auto best = std::max_element(distances.begin(), distances.end()) - distances.begin();
        selected.insert(best);
        update(best);
    }
    std::vector<Entry> result;
    for (auto i : selected) result.push_back(std::move(entries[i]));
    return result;
}
class Archive {
    std::size_t capacity_;
    double limit_;
    bool pareto_;
    std::vector<Entry> entries_;
public:
    Archive(std::size_t capacity, double limit, bool pareto = false)
        : capacity_(capacity), limit_(limit), pareto_(pareto) {}
    const std::vector<Entry> &entries() const { return entries_; }
    void add(Entry entry) {
        for (const auto &e : entries_) if (e.ids == entry.ids) return;
        if (pareto_) {
            const auto better = [&](const Entry& a, const Entry& b) {
                return dominates(a.valid_loss,a.relative,b.valid_loss,b.relative,limit_)
                    || (feasible(a.valid_loss,limit_) == feasible(b.valid_loss,limit_)
                        && order_loss(a.valid_loss) == order_loss(b.valid_loss)
                        && a.relative == b.relative && a.ids < b.ids);
            };
            if (std::any_of(entries_.begin(), entries_.end(), [&](const Entry& e) {return better(e,entry);})) return;
            std::erase_if(entries_, [&](const Entry& e) {return better(entry,e);});
        }
        entries_.push_back(std::move(entry));
        const bool any = std::any_of(entries_.begin(), entries_.end(), [&](const Entry &e) { return feasible(e.valid_loss, limit_); });
        if (any) std::erase_if(entries_, [&](const Entry &e) { return !feasible(e.valid_loss, limit_); });
        if (entries_.size() > capacity_) entries_ = representatives(std::move(entries_), capacity_);
    }
    std::vector<Entry> shortlist(std::size_t count) const {
        std::vector<Entry> result;
        for (const auto &e : entries_) if (feasible(e.valid_loss, limit_)) result.push_back(e);
        return representatives(std::move(result), count);
    }
};

struct Workload {
    std::vector<Atoms> frames;
    std::vector<int> global_ids;
    double active = 0, stored = 0, max_neighbors = 0;
    std::size_t atoms = 0;
};
// Stratify by composition, log2 atom count and log2 mean active coordination.
// Only rank 0 calls this; selection uses original global frame indices.
inline Workload sample_workload(PotMTP &pot, std::vector<Atoms> &pool, int count, int seed) {
    std::map<std::vector<int>, std::vector<int>> strata;
    std::vector<std::array<double, 3>> counts(pool.size());
    for (std::size_t i = 0; i < pool.size(); ++i) {
        auto &a = pool[i];
        if (a.size() == 0) throw std::runtime_error("Empty cost benchmark structure");
        Neighborhoods nbhs(a, pot.cutoff);
        std::vector<int> key(pot.species_count, 0);
        for (int type : a.chemmap_) {
            if (type < 0 || type >= pot.species_count) throw std::runtime_error("Invalid benchmark atom type");
            ++key[type];
        }
        // Composition fractions in quarters keep similar sizes in one stratum.
        for (int &n : key) n = n == 0 ? 0 : 1 + (4 * n / a.size());
        auto &c = counts[i];
        for (int j = 0; j < nbhs.size(); ++j) {
            const auto &nbh = nbhs[j];
            c[1] += nbh.nbhcount_;
            c[2] = std::max(c[2], double(nbh.nbhcount_));
            for (int k = 0; k < nbh.nbhcount_; ++k)
                if (nbh.distances_[k] <= pot.cutoffs(nbh.centertype, nbh.chemmap_[k])) ++c[0];
        }
        key.push_back(int(std::log2(a.size())));
        key.push_back(int(std::log2(1. + c[0] / a.size())));
        strata[key].push_back(i);
    }
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    std::vector<std::vector<int>> groups;
    for (auto &[key, ids] : strata) { std::shuffle(ids.begin(), ids.end(), rng); groups.push_back(std::move(ids)); }
    std::shuffle(groups.begin(), groups.end(), rng);
    Workload w;
    const auto target = std::min<std::size_t>(count, pool.size());
    for (std::size_t round = 0; w.frames.size() < target; ++round) {
        for (const auto &group : groups) {
            if (round >= group.size() || w.frames.size() == target) continue;
            int id = group[round];
            w.global_ids.push_back(id);
            w.atoms += pool[id].size();
            w.active += counts[id][0]; w.stored += counts[id][1];
            w.max_neighbors = std::max(w.max_neighbors, counts[id][2]);
            w.frames.push_back(std::move(pool[id]));
        }
    }
    if (!w.atoms) throw std::runtime_error("No structures for cost benchmark");
    w.active /= w.atoms; w.stored /= w.atoms;
    return w;
}
struct Timing {
    double median = 0, low = 0, high = 0;
};
inline Timing summarize(std::vector<double> times) {
    if (times.size() < 3) throw std::runtime_error("Insufficient benchmark rounds");
    for (double t : times) if (!std::isfinite(t) || t <= 0) throw std::runtime_error("Invalid benchmark time");
    std::sort(times.begin(), times.end());
    auto quantile = [&](double q) {
        double p = q * (times.size()-1);
        auto lo = std::size_t(p), hi = std::min(lo+1, times.size()-1);
        return times[lo] + (times[hi]-times[lo]) * (p-lo);
    };
    return {quantile(.5), quantile(.25), quantile(.75)};
}
struct Measurement {
    Timing kernel, total;
    std::vector<double> kernel_rounds, total_rounds;
};
// Nondominated feasible entries for one specified cost metric. Ties in the
// rounded loss and exact cost retain the lexicographically smallest mask.
inline std::vector<std::size_t> pareto_front(const std::vector<Entry>& entries,
                                           const std::vector<double>& costs, double limit) {
    if (entries.size() != costs.size()) throw std::runtime_error("Pareto size mismatch");
    std::vector<std::size_t> front;
    for (std::size_t i=0; i<entries.size(); ++i) {
        if (!feasible(entries[i].valid_loss,limit) || !std::isfinite(costs[i]) || costs[i]<0) continue;
        bool dominated=false;
        for (std::size_t j=0; j<entries.size(); ++j) {
            if (j==i || !feasible(entries[j].valid_loss,limit) || !std::isfinite(costs[j]) || costs[j]<0) continue;
            if (dominates(entries[j].valid_loss,costs[j],entries[i].valid_loss,costs[i],limit)
                || (order_loss(entries[j].valid_loss)==order_loss(entries[i].valid_loss)
                    && costs[j]==costs[i] && entries[j].ids<entries[i].ids)) {dominated=true;break;}
        }
        if (!dominated) front.push_back(i);
    }
    std::sort(front.begin(),front.end(),[&](auto i,auto j) {
        if (costs[i]!=costs[j]) return costs[i]<costs[j];
        return entries[i].ids<entries[j].ids;
    });
    return front;
}
// Timing-overlap heuristic, not a statistical significance test. Compare to
// the fastest median to avoid non-transitive interval-based sorting.
inline std::size_t choose_measured(const std::vector<Entry> &entries,
                                   const std::vector<Measurement> &times, double limit) {
    if (entries.size() != times.size()) throw std::runtime_error("Benchmark size mismatch");
    std::size_t fastest = entries.size();
    for (std::size_t i = 0; i < entries.size(); ++i)
        if (feasible(entries[i].valid_loss, limit) && (fastest == entries.size() ||
            times[i].total.median < times[fastest].total.median)) fastest = i;
    if (fastest == entries.size()) throw std::runtime_error("No feasible measured candidate");
    std::size_t best = fastest;
    for (std::size_t i = 0; i < entries.size(); ++i) {
        if (!feasible(entries[i].valid_loss, limit)) continue;
        if (times[i].total.low > times[fastest].total.high || times[fastest].total.low > times[i].total.high) continue;
        if (entries[i].valid_loss < entries[best].valid_loss ||
            (entries[i].valid_loss == entries[best].valid_loss && entries[i].ids < entries[best].ids)) best = i;
    }
    return best;
}
inline std::vector<Measurement> benchmark(std::vector<std::unique_ptr<PotMTP>> &models,
        Workload &work, int rounds, double seconds, int seed) {
    using Clock = std::chrono::steady_clock;
    std::vector<Neighborhoods> neighborhoods;
    for (auto &a : work.frames) neighborhoods.emplace_back(a, models.front()->cutoff);
    auto evaluate = [&](PotMTP &pot, bool kernel) {
        for (std::size_t j = 0; j < work.frames.size(); ++j) {
            if (kernel) pot.getEFSOMP(work.frames[j], neighborhoods[j]);
            else pot.getEFSOMP(work.frames[j]);
        }
    };
    for (auto &pot : models) { pot->omp_init(); evaluate(*pot, false); evaluate(*pot, true); }
    std::vector<std::array<std::vector<double>, 2>> samples(models.size());
    std::vector<std::size_t> order(models.size());
    std::iota(order.begin(), order.end(), 0);
    std::mt19937 rng(static_cast<std::mt19937::result_type>(seed));
    for (int round = 0; round < rounds; ++round) {
        std::shuffle(order.begin(), order.end(), rng);
        for (auto i : order) for (int path = 0; path < 2; ++path) {
            auto begin = Clock::now();
            std::size_t iterations = 0;
            double elapsed;
            do {
                evaluate(*models[i], path == 0);
                ++iterations;
                elapsed = std::chrono::duration<double>(Clock::now()-begin).count();
            } while (elapsed < seconds);
            samples[i][path].push_back(elapsed * 1e9 / (iterations * double(work.atoms)));
        }
    }
    std::vector<Measurement> result;
    for (auto &s : samples) result.push_back({summarize(s[0]), summarize(s[1]), s[0], s[1]});
    return result;
}
} // namespace func_fastfit::cost
