// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "utils/gputils.h"
#include "apps.h"
#include "config/version.h"
#include <algorithm>
#include <string_view>

int main(int argc, char *argv[])
{
    #ifdef IMRMLP_MPI
    int thread_support = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_support);
    mpi.InitComm(MPI_COMM_WORLD);
    struct MPILifetime { ~MPILifetime() { MPI_Finalize(); } } mpi_lifetime;
    #endif

    std::string command = "";
    if (argc < 2) {
        if (mpi.rank != 0) return 1;
        std::cerr << "Usage: " << argv[0] << " <command> [options]" << std::endl;
        std::cerr << "Use '" << argv[0] << " --help' to see available commands." << std::endl;
        return 1;
    }
    command = std::string(argv[1]);
    if (mpi.rank != 0 && (command == "help" || command == "--help" || command == "-h"
        || command == "--version" || command == "-v" || command == "version")) return 0;
    if (command == "help" || command == "--help" || command == "-h") {
        std::cout << "IMR-MLP commands: " << std::endl;
        for (const auto &cmd : command_maps) {
            if (!cmd.second.second.empty())
                std::cout << fmt::format("  {:<20} - {}", cmd.first, cmd.second.second) << std::endl;
        }
        #ifdef IMRMLP_MPI
        std::cout << "MPI enabled commands: neb calc_neb trainmtp trainnlpot fastfit calc_err (--mpi-mode frames)" << std::endl;
        #endif
        return 0;
    } else if (command == "--version" || command == "-v" || command == "version") {
        std::cout << "IMR-MLP version: " << PROJECT_VERSION << "." << GIT_COMMIT_ID << std::endl;
        std::cout << "Target:     " << BUILD_TARGET    << std::endl;
        std::cout << "Date:       " << BUILD_DATE      << std::endl;
        std::string compiler;
        #if defined(__clang__)
        compiler = "Clang " + std::string(__clang_version__);
        #elif defined(__GNUC__)
        compiler = "GCC " + std::to_string(__GNUC__) + "." + std::to_string(__GNUC_MINOR__) + "." + std::to_string(__GNUC_PATCHLEVEL__);
        #elif defined(_MSC_VER)
        compiler = "MSVC " + std::to_string(_MSC_VER);
        #else
        compiler = "Unknown Compiler";
        #endif
        std::cout << "Compiler:   " << compiler << std::endl;
        std::string mpi_enables = "No\n";
        std::string omp_enables = "No";

        #ifdef IMRMLP_MPI
        char version_str[MPI_MAX_LIBRARY_VERSION_STRING];
        int version_len;
        MPI_Get_library_version(version_str, &version_len);
        mpi_enables = std::string(version_str, version_len);
        #endif
        #ifdef USE_OMP
        omp_enables = "Yes";
        #endif
        std::cout << "MPI:        " << mpi_enables;
        std::cout << "OpenMP:     " << omp_enables << std::endl;
        return 0;
    }
    if (command_maps.count(command) == 0) {
        std::cerr << "Unknown command: " << command << std::endl;

        // Levenshtein (edit) distance — fallback for fuzzy matching
        auto levenshtein = [](const std::string &a, const std::string &b) -> int {
            int n = (int)a.size(), m = (int)b.size();
            std::vector<int> prev(m + 1), cur(m + 1);
            for (int j = 0; j <= m; ++j) prev[j] = j;
            for (int i = 1; i <= n; ++i) {
                cur[0] = i;
                for (int j = 1; j <= m; ++j) {
                    cur[j] = std::min({prev[j] + 1, cur[j-1] + 1,
                                       prev[j-1] + (a[i-1] != b[j-1])});
                }
                std::swap(prev, cur);
            }
            return prev[m];
        };

        // Multi-strategy fuzzy match score: lower = better, negative = no match.
        //   0       — exact prefix  (e.g. "remove" → "remove_duplicate_structures")
        //   1 .. 99 — _-token prefix (e.g. "maxvol" → "maxvol_select")
        //   100+    — substring      (e.g. "vol" → "maxvol_select")
        //   200+    — normalised Levenshtein similarity ≥ 0.5
        auto fuzzy_match_score = [&](const std::string &input, const std::string &candidate) -> int {
            if (candidate.starts_with(input))
                return 0;

            // token-prefix: any _-separated token starts with input?
            size_t pos = 0;
            int token_idx = 0;
            while (pos < candidate.size()) {
                size_t next = candidate.find('_', pos);
                if (next == std::string::npos) next = candidate.size();
                if (next > pos) {
                    std::string_view token(candidate.data() + pos, next - pos);
                    if (token.starts_with(input))
                        return 1 + token_idx;
                }
                pos = next + 1;
                ++token_idx;
            }

            // substring anywhere
            size_t found = candidate.find(input);
            if (found != std::string::npos)
                return 100 + (int)found;

            // normalised edit distance fallback
            int d = levenshtein(input, candidate);
            int max_len = std::max((int)input.size(), (int)candidate.size());
            if (max_len > 0) {
                double similarity = 1.0 - (double)d / max_len;
                if (similarity >= 0.5)
                    return 200 + d;
            }

            return -1;
        };

        // Score every registered command name; collect those that match.
        std::vector<std::pair<int, std::string>> candidates;
        for (const auto &[name, _] : command_maps) {
            int score = fuzzy_match_score(command, name);
            if (score >= 0)
                candidates.emplace_back(score, name);
        }

        if (!candidates.empty()) {
            std::sort(candidates.begin(), candidates.end());
            std::cerr << "Did you mean:";
            int shown = 0;
            for (auto &[score, name] : candidates) {
                if (shown >= 5) break;
                std::cerr << "  " << name;
                ++shown;
            }
            std::cerr << std::endl;
        } else {
            std::cerr << "Use '" << argv[0] << " --help' to see available commands." << std::endl;
        }
        return 1;
    }
    int ret = 0;
    try {
        ret = command_maps[command].first(argc - 1, argv + 1);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
        ret = 1;
    }

    return ret;
}
