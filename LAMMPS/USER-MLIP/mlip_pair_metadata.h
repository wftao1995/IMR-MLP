#pragma once

#if __has_include("mtp_metadata.h")
#include "mtp_metadata.h"
#else
#include "../../tools/mtp_metadata.h"
#endif

namespace MTPMetadata {
// Command state shared by the independent LAMMPS pair implementations. Parsing
// does not depend on LAMMPS, and exceptions are translated by each adapter.
struct PairArguments {
    bool loaded = false;
    int mapping_count = 0;
    std::vector<std::string> options;

    bool defer(int narg, char **arg, bool options_only = false) {
        require(!loaded, "potential cannot be reset; create a new pair style first");
        if (narg && !options_only) return false;
        options.clear();
        for (int i = 0; i < narg; ++i) options.emplace_back(arg[i]);
        return true;
    }
    void load(int narg, char **arg, int map_end, int *mapping, size_t capacity,
              bool allow_long_range = false, bool allow_nonlinear = false,
              const Header *supplied_header = nullptr) {
        require(narg > 0 && map_end > 1, "a potential file and atom-type mappings are required");
        require(static_cast<size_t>(map_end) <= capacity, "too many atom-type mappings");
        const auto header = supplied_header ? *supplied_header : read_header(arg[0]);
        require_model(header.tag, allow_long_range, allow_nonlinear);
        std::vector<std::string> names;
        for (int i = 1; i < map_end; ++i) names.emplace_back(arg[i]);
        const auto resolved = map_types(header, names);
        std::fill(mapping, mapping + capacity, -1);
        mapping_count = static_cast<int>(resolved.size());
        std::copy(resolved.begin(), resolved.end(), mapping + 1);
    }
    void validate(int types) const {
        require(loaded, "no potential loaded; use pair_coeff * * FILE ELEMENTS...");
        require(mapping_count == types, "mapping count must equal the number of LAMMPS atom types");
    }
    std::vector<std::string> coefficients(int narg, char **arg, int types) const {
        require(narg >= 2 && std::string(arg[0]) == "*" && std::string(arg[1]) == "*",
                "pair_coeff must start with '* *'");
        if (narg == 2) { validate(types); return {}; }
        require(!loaded, "potential was already supplied; use pair_coeff * * with the legacy pair_style form");
        require(narg == 3 + types, "pair_coeff requires a filename and one mapping per LAMMPS atom type");
        std::vector<std::string> result;
        for (int i = 2; i < narg; ++i) result.emplace_back(arg[i]);
        result.insert(result.end(), options.begin(), options.end());
        return result;
    }
};
} // namespace MTPMetadata
