// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef _MLGP_CHEM_H
#define _MLGP_CHEM_H

#include <string>
#include <map>
#include <algorithm>

namespace Chemicals
{
    extern std::map<std::string, int> atomic_numbers;
    extern std::string chemical_symbols[119];
    extern double atomic_masses[119];
    extern double covalent_radii[119]; // Cordero et al., Dalton Trans. (2008)
} // namespace Chemicals

// ── Ordered chemical-type pair (canonical: a <= b) ─────────────────────
// Used across analyse/ as map keys for pair-counting histograms
// (rdf_partial_type, sro_counter, tsro_counter, haindex_bond_type, cna_index_sp, etc.)
struct ChemPair {
    int a, b;
    ChemPair(int x, int y) : a(std::min(x, y)), b(std::max(x, y)) {}
    ChemPair() : a(0), b(0) {}

    bool operator<(const ChemPair &o) const {
        if (a < o.a) return true;
        if (a > o.a) return false;
        return b < o.b;
    }

    bool operator==(const ChemPair &o) const {
        return a == o.a && b == o.b;
    }

    bool operator!=(const ChemPair &o) const {
        return !(*this == o);
    }
};

#endif