// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/****
 * This is an C++ implementation of PA.F with many imporvements
 * 
 * Written by Jiantao Wang (wftao1995@gmail.com)
 * 
 * Optional indices select HA category codes, never atom or neighbor indices.
 * 
*****/

#ifndef _CALC_HA_UTILS_H
#define _CALC_HA_UTILS_H

#include "base/anyneighborhoods.h"
#include "utils/gputils.h"
#include <map>
#include <array>
#include <algorithm>
#include <cstdint>
#include <set>
#include <tuple>
#include <iostream>
#include <iomanip>
#include <Eigen/Dense>

#define INITIAL_HA_MAP                                                                                                                                                                                                                                                                                                       \
    {                                                                                                                                                                                                                                                                                                                        \
        {1201, 0}, {1211, 0}, {1301, 0}, {1311, 0}, {1321, 0}, {1331, 0}, {1421, 0}, {1422, 0}, {1431, 0}, {1441, 0}, {1541, 0}, {1551, 0}, {1661, 0}, {1771, 0}, {2201, 0}, {2211, 0}, {2301, 0}, {2311, 0}, {2321, 0}, {2331, 0}, {2421, 0}, {2422, 0}, {2431, 0}, {2441, 0}, {2541, 0}, {2551, 0}, {2661, 0}, { 2771, 0 } \
    }
#define INITIAL_HA_LIST                                                                                                                                                        \
    {                                                                                                                                                                          \
        1201, 1211, 1301, 1311, 1321, 1331, 1421, 1422, 1431, 1441, 1541, 1551, 1661, 1771, 2201, 2211, 2301, 2311, 2321, 2331, 2421, 2422, 2431, 2441, 2541, 2551, 2661, 2771 \
    }

typedef int ha_int;   // integer type used
typedef int haindexT; // HA index type, for example 1551

struct haindex_bond_type
{
    haindexT index;
    int chem[2];
    bool operator<(const haindex_bond_type &opt) const
    {
        if (index < opt.index)
            return true;
        if (index > opt.index)
            return false;
        if (chem[0] < opt.chem[0])
            return true;
        if (chem[0] > opt.chem[0])
            return false;
        if (chem[1] < opt.chem[1])
            return true;
        return false;
    }
    bool operator==(const haindex_bond_type &opt) const
    {
        if (index != opt.index)
            return false;
        if (chem[0] != opt.chem[0])
            return false;
        if (chem[1] != opt.chem[1])
            return false;
        return true;
    }
    bool operator!=(const haindex_bond_type &opt) const
    {
        return !(*this == opt);
    }
};
struct haindex_pos
{
    haindexT index;
    std::vector<Eigen::RowVector3d> pos;
    bool operator<(const haindex_pos &opt) const
    {
        if (index < opt.index)
            return true;
        return false;
    }
    bool operator!=(const haindex_pos &opt) const
    {
        if (index != opt.index)
            return true;
        return false;
    }
};
typedef std::map<haindexT, ha_int> ha_count; // index types count
typedef std::map<haindex_bond_type, ha_int> ha_partial_count;

// Optional caller-owned work counters, reset by each counted computation.
// They measure classification work, independently of histogram output size.
struct HAWorkStats {
    std::uint64_t owned_pairs = 0;
    std::uint64_t common_sets = 0;
    std::uint64_t common_bond_checks = 0;
    std::uint64_t fourth_classifications = 0;
};

struct ha_index_image // ha info for each configuration
{
    ha_count total;           // count for each index type
    ha_partial_count partial; // count for each index with bound info
    ha_int all_count = 0;     // total index counts
#ifndef NO_SAMPLE_STR
    std::map<haindexT, haindex_pos> structure; // smapled structures
#endif

    void operator+=(const ha_index_image &opt)
    {
        for (auto &it : opt.total)
        {
            total[it.first] += it.second;
        }
        for (auto &it : opt.partial)
        {
            partial[it.first] += it.second;
        }
        all_count += opt.all_count;
#ifndef NO_SAMPLE_STR
        structure.insert(opt.structure.begin(), opt.structure.end()); // in C++17, one may use structure.merge(opt.structure)
#endif
    }

    // output all, sorted by count
    std::ostream &outall(std::ostream &os)
    {
        std::vector<std::pair<haindexT, ha_int>> temp_sort;
        for (auto &it : total)
        {
            temp_sort.emplace_back(it);
        }
        std::sort(temp_sort.begin(), temp_sort.end(),
                  [](std::pair<haindexT, ha_int> &a, std::pair<haindexT, ha_int> &b) { return a.second > b.second; });
        for (auto &it : temp_sort)
        {
            os << std::setw(7) << it.first << std::setw(15) << it.second << std::setw(15)
               << (all_count == 0 ? 0.0 : static_cast<double>(it.second) / static_cast<double>(all_count)) << '\n';
        }
        return os;
    }
    // out partial, sorted by count, make sure that chem1 <= chem2
    std::ostream &outpart(std::ostream &os, int chem1, int chem2)
    {
        if (chem1 > chem2)
        {
            std::swap(chem1, chem2);
        }
        std::vector<std::pair<haindex_bond_type, ha_int>> temp_sort;
        for (auto &it : partial)
        {
            if (it.first.chem[0] == chem1 && it.first.chem[1] == chem2)
            {
                temp_sort.emplace_back(it);
            }
        }
        std::sort(temp_sort.begin(), temp_sort.end(),
                  [](std::pair<haindex_bond_type, ha_int> &a, std::pair<haindex_bond_type, ha_int> &b) { return a.second > b.second; });
        for (auto &it : temp_sort)
        {
            os << std::setw(7) << it.first.index << std::setw(15) << it.second << std::setw(15)
               << (all_count == 0 ? 0.0 : static_cast<double>(it.second) / static_cast<double>(all_count)) << '\n';
        }
        return os;
    }
    // Explicitly selected categories, ascending and including absent rows.
    // Fractions retain the full owned-pair denominator, also for partials.
    std::ostream &outselected(std::ostream &os) const {
        ha_int selected_count = 0;
        for (const auto &[index, count] : total) {
            os << std::setw(7) << index << std::setw(15) << count << std::setw(15)
               << (all_count == 0 ? 0. : double(count) / all_count) << '\n';
            selected_count += count;
        }
        os << "Total" << std::setw(15) << selected_count << std::setw(15)
           << (all_count == 0 ? 0. : double(selected_count) / all_count) << '\n';
        return os;
    }
    std::ostream &outpartselected(std::ostream &os, int chem1, int chem2) const {
        if (chem1 > chem2) std::swap(chem1, chem2);
        ha_int selected_count = 0;
        for (const auto &[index, ignored] : total) {
            const auto found = partial.find(haindex_bond_type{index, {chem1, chem2}});
            const ha_int count = found == partial.end() ? 0 : found->second;
            os << std::setw(7) << index << std::setw(15) << count << std::setw(15)
               << (all_count == 0 ? 0. : double(count) / all_count) << '\n';
            selected_count += count;
        }
        os << "Total" << std::setw(15) << selected_count << std::setw(15)
           << (all_count == 0 ? 0. : double(selected_count) / all_count) << '\n';
        return os;
    }
    ha_int commom_count()
    {
        haindexT comminds[] = INITIAL_HA_LIST;
        ha_int result = 0;
        for (int i = 0; i < _length(comminds); ++i)
        {
            result += total[comminds[i]];
        }
        return result;
    }
    // out total, common, sorted by index
    std::ostream &outallcomm(std::ostream &os, int bflag)
    {
        haindexT comminds[] = INITIAL_HA_LIST;
        size_t num_present = 0;
        for (int i = 0; i < _length(comminds) / (3 - bflag); ++i)
        {
            os << std::setw(7) << comminds[i] << std::setw(15) << total[comminds[i]] << std::setw(15)
               << (all_count == 0 ? 0.0 : static_cast<double>(total[comminds[i]]) / static_cast<double>(all_count)) << '\n';
            num_present += total[comminds[i]];
        }
        os << "Total" << std::setw(15) << num_present << std::setw(15)
           << (all_count == 0 ? 0.0 : static_cast<double>(num_present) / static_cast<double>(all_count)) << '\n';
        return os;
    }
    // out parital, common, sorted by index
    std::ostream &outpartcomm(std::ostream &os, int chem1, int chem2, int bflag)
    {
        if (chem1 > chem2)
        {
            std::swap(chem1, chem2);
        }
        size_t num_present = 0;
        haindexT comminds[] = INITIAL_HA_LIST;
        for (int i = 0; i < _length(comminds) / (3 - bflag); ++i)
        {
            haindex_bond_type gind;
            gind.index = comminds[i];
            gind.chem[0] = chem1;
            gind.chem[1] = chem2;
            os << std::setw(7) << comminds[i] << std::setw(15) << partial[gind] << std::setw(15)
               << (all_count == 0 ? 0.0 : static_cast<double>(partial[gind]) / static_cast<double>(all_count)) << '\n';
            num_present += partial[gind];
        }
        os << "Total" << std::setw(15) << num_present << std::setw(15)
           << (all_count == 0 ? 0.0 : static_cast<double>(num_present) / static_cast<double>(all_count)) << '\n';
        return os;
    }
};

// main ComputeHA computer
class ComputeHA : public ConnectSettings
{
private:
    bool count_ubond = false;
    bool sample_one = false;
    bool sample_all = false;
    std::vector<haindexT> selected_indices_;
    struct Selection {
        std::set<haindexT> codes;
        std::set<int> first;
        std::set<std::pair<int, int>> common;
        std::set<std::tuple<int, int, int>> bonds;
        bool ordinary_only = true;
        explicit Selection(const std::vector<haindexT> &);
        bool active() const { return !codes.empty(); }
    };
    ha_index_image compute_selected(const AnyNeighborhoods &, HAWorkStats *);

    // ── Common implementation shared by both compute() overloads ──────
    // Image geometry uses the given graph; cutoff filtering is the caller's
    // responsibility and precedes this computation.
    void compute_impl(const AnyNeighborhoods &nbhs,
                      ha_index_image &result, int atom_i, const Selection &, HAWorkStats *);

public:
    ha_index_image compute(AnyNeighborhoods &nbhs);
    ha_index_image compute(AnyNeighborhoods &nbhs, double **pair_cutoff);
    ha_index_image compute(AnyNeighborhoods &nbhs, HAWorkStats &work);
    ha_index_image compute(AnyNeighborhoods &nbhs, double **pair_cutoff, HAWorkStats &work);
    // Empty selections restore complete classification. Duplicates are harmless.
    void set_selected_indices(const std::set<haindexT> &indices) {
        selected_indices_.assign(indices.begin(), indices.end());
    }
    bool has_selection() const { return !selected_indices_.empty(); }
    void InitSettings()
    {
        Connect(selected_indices_, "indices", "HA category codes (e.g. 1421 1551); empty selects all; fractions use all pairs", -1);
        Connect(count_ubond, "unbond");
        Connect(sample_one, "store_str");
        Connect(sample_all, "sample_all");
    }
    ComputeHA(){};
    ComputeHA(const GPSettings &_settings);

    int bflag()
    {
        return count_ubond ? 2 : 1;
    };
};

#endif //_CALC_HA_UTILS_H
