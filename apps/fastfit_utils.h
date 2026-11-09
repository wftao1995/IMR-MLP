// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_APPS_FASTFIT_UTILS_H
#define IMRMLP_APPS_FASTFIT_UTILS_H

#include <algorithm>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>

namespace func_fastfit::detail
{

inline std::set<int> ReadScalarIds(const std::string &filename, int nscalar)
{
    std::ifstream input(filename);
    if (!input)
        throw std::runtime_error("cannot open selection file: " + filename);
    long long count = -1;
    if (!(input >> count) || count < 0 || count > nscalar)
    {
        throw std::runtime_error("invalid selection count in " + filename);
    }
    std::set<int> ids;
    for (long long i = 0; i < count; ++i)
    {
        int id = -1;
        if (!(input >> id) || id < 0 || id >= nscalar || !ids.insert(id).second)
        {
            throw std::runtime_error("invalid, duplicate, or out-of-range scalar id in " +
                                     filename);
        }
    }
    std::string extra;
    if (input >> extra)
        throw std::runtime_error("unexpected trailing data in " + filename);
    return ids;
}

inline void ValidatePartition(const std::set<int> &selected, const std::set<int> &removed,
                              int nscalar)
{
    if (nscalar < 0)
        throw std::invalid_argument("scalar count must be non-negative");
    const auto invalid_id = [nscalar](int id)
    {
        return id < 0 || id >= nscalar;
    };
    if (std::any_of(selected.begin(), selected.end(), invalid_id) ||
        std::any_of(removed.begin(), removed.end(), invalid_id))
    {
        throw std::runtime_error("selected or removed scalar id is out of range");
    }
    for (int id = 0; id < nscalar; ++id)
    {
        if (selected.count(id) + removed.count(id) != 1)
        {
            throw std::runtime_error("selected and removed ids must form an exact partition");
        }
    }
}

inline void WriteScalarIds(const std::string &filename, const std::set<int> &ids)
{
    std::ofstream output(filename);
    if (!output)
        throw std::runtime_error("cannot write selection file: " + filename);
    output << ids.size();
    for (int id : ids)
        output << ' ' << id;
    output << '\n';
    if (!output)
        throw std::runtime_error("failed while writing selection file: " + filename);
}

} // namespace func_fastfit::detail

#endif
