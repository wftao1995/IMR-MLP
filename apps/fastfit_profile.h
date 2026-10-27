// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/comm.h"
#include <array>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>
#if defined(__linux__) || defined(__APPLE__)
#include <sys/resource.h>
#endif

namespace func_fastfit {

// Main-thread scopes measure exclusive wall time. OpenMP work is measured by
// its enclosing scope; no instrumentation or synchronization in worker loops.
class Profile {
public:
    using Clock = std::chrono::steady_clock;
    enum class Stage {
        Input, Assembly, MatrixValidation, Solve, CachedScore, DirectEFS, FinalEFS,
        Descriptors, Correlation, CostAnalysis, CostBenchmark, Selection, Output, Communication, Other, Count
    };
    static constexpr int count = static_cast<int>(Stage::Count);
    inline static constexpr std::array<const char *, count> names{
        "input", "assembly", "matrix_validation", "solve", "cached_score",
        "direct_efs", "final_efs", "descriptors", "correlation", "cost_analysis", "cost_benchmark", "selection", "output",
        "communication", "other"
    };

    class Scope {
    public:
        Scope(Profile &profile, Stage stage) : profile_(profile.enabled_ ? &profile : nullptr) {
            if (profile_) {
                previous_ = profile_->current_;
                profile_->switch_to(stage);
                ++profile_->calls_[static_cast<int>(stage)];
            }
        }
        Scope(const Scope &) = delete;
        Scope &operator=(const Scope &) = delete;
        ~Scope() { stop(); }
        void stop() {
            if (profile_) { profile_->switch_to(previous_); profile_ = nullptr; }
        }
    private:
        Profile *profile_;
        Stage previous_ = Stage::Other;
    };

    void start(bool enabled, Clock::time_point application_start) {
        enabled_ = enabled;
        if (!enabled_) return;
        seconds_.fill(0.); calls_.fill(0.);
        start_ = last_ = application_start;
        current_ = Stage::Other;
        initial_peak_ = peak_rss_mib();
    }
    Scope scope(Stage stage) { return Scope(*this, stage); }
    template<class Function> decltype(auto) measure(Stage stage, Function &&function) {
        auto timing = scope(stage);
        return std::forward<Function>(function)();
    }
    template<class Function> decltype(auto) communicate(Function &&function) {
        if (mpi.size == 1) return std::forward<Function>(function)();
        return measure(Stage::Communication, std::forward<Function>(function));
    }

    // Call collectively after the mode completes, outside all timing scopes.
    // Report aggregation and report-file I/O are excluded from measured time.
    void report(std::ostream &output, const std::string &filename, int mode,
                const std::string &mode_error = {}) {
        if (!enabled_) {
            if (!mode_error.empty()) throw std::runtime_error(mode_error);
            return;
        }
        switch_to(Stage::Other);
        enabled_ = false;
        constexpr int total_index = count;
        constexpr int peak_index = count + 1;
        constexpr int valid_index = count + 2;
        constexpr int calls_index = count + 3;
        constexpr int failed_index = 2 * count + 3;
        constexpr int fields = failed_index + 1;
        std::array<double, fields> local{}, sums{}, maxima{};
        for (int i = 0; i < count; ++i) {
            local[i] = seconds_[i];
            local[calls_index + i] = calls_[i];
        }
        local[total_index] = std::chrono::duration<double>(last_ - start_).count();
        local[peak_index] = peak_rss_mib();
        local[valid_index] = local[peak_index] >= 0 ? 1. : 0.;
        local[failed_index] = mode_error.empty() ? 0. : 1.;
        MPI_Reduce(local.data(), sums.data(), fields, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        MPI_Reduce(local.data(), maxima.data(), fields, MPI_DOUBLE, MPI_MAX, 0, mpi.comm);

        int status[2] = {1, 0}; // report written, mode failed on any rank
        if (mpi.rank == 0) {
            status[1] = sums[failed_index] > 0;
            try {
                std::ostringstream text;
                text.imbue(std::locale::classic());
                text << std::setprecision(10)
                     << "# fastfit_profile_version\t1\n# mode\t" << mode
                     << "\n# ranks\t" << mpi.size
                     << "\n# status\t" << (status[1] ? "failed" : "ok")
                     << "\n# timing\texclusive wall seconds; communication includes MPI waiting"
                     << "\n# aggregation\tstage maxima need not add up to total maximum"
                     << "\n# memory\tprocess-lifetime peak RSS in MiB; rank-peak sum is not a simultaneous peak\n";
                if (sums[valid_index] == mpi.size) {
                    text << "# rss_peak_at_start_rank0_mib\t" << initial_peak_
                         << "\n# rss_peak_rank0_mib\t" << local[peak_index]
                         << "\n# rss_peak_rank_max_mib\t" << maxima[peak_index]
                         << "\n# rss_peak_rank_sum_mib\t" << sums[peak_index] << '\n';
                } else text << "# rss_peak\tunavailable\n";
                text << "stage\trank0_seconds\tmean_seconds\tmax_seconds\tcalls_rank0\tcalls_sum\n";
                for (int i = 0; i < count; ++i)
                    text << names[i] << '\t' << local[i] << '\t' << sums[i] / mpi.size
                         << '\t' << maxima[i] << '\t' << local[calls_index + i]
                         << '\t' << sums[calls_index + i] << '\n';
                text << "total\t" << local[total_index] << '\t' << sums[total_index] / mpi.size
                     << '\t' << maxima[total_index] << "\t1\t" << mpi.size << '\n';
                output << text.str();
                if (!filename.empty()) {
                    std::ofstream file(filename);
                    file << text.str();
                    file.close();
                    if (!file) status[0] = 0;
                }
            } catch (...) { status[0] = 0; }
        }
        MPI_Bcast(status, 2, MPI_INT, 0, mpi.comm);
        if (status[1]) throw std::runtime_error(mode_error.empty()
            ? "fastfit mode failed on another MPI rank" : mode_error);
        if (!status[0]) throw std::runtime_error("Cannot write fastfit profile: " + filename);
    }

private:
    bool enabled_ = false;
    Stage current_ = Stage::Other;
    Clock::time_point start_, last_;
    std::array<double, count> seconds_{}, calls_{};
    double initial_peak_ = -1.;

    void switch_to(Stage next) {
        const auto now = Clock::now();
        seconds_[static_cast<int>(current_)] += std::chrono::duration<double>(now - last_).count();
        last_ = now;
        current_ = next;
    }
    static double peak_rss_mib() {
#if defined(__linux__) || defined(__APPLE__)
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) return -1.;
#ifdef __APPLE__
        return usage.ru_maxrss / (1024. * 1024.); // bytes on macOS
#else
        return usage.ru_maxrss / 1024.; // KiB on Linux
#endif
#else
        return -1.;
#endif
    }
};
} // namespace func_fastfit
