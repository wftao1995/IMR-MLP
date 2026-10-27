// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mathextra/spline.h"
#include "utils/gputils.h"
#include <cmath>
#include <set>
#include <sstream>

namespace Potentials::EAMTable {

inline int pair_index(int i, int j)
{
    if (i < j) std::swap(i, j);
    return i * (i + 1) / 2 + j;
}

struct Header {
    int species_count, nr, nrho;
    double dr, drho, cutoff, rhomin = 0.0, rhomax;
    bool he = false;
    std::vector<std::string> names;
    std::vector<int> types; // file element -> model type; -1 means unselected
};

inline Header read_header(std::istream &is, const std::string &file,
                          const std::map<std::string, int> &typemap)
{
    auto fail = [&] { throw std::runtime_error("Invalid EAM header or species map in " + file); };
    std::string line;
    for (int i = 0; i < 3; ++i) if (!std::getline(is, line)) fail();
    int count = 0;
    if (!(is >> count) || count <= 0) fail();
    std::vector<std::string> names(count);
    std::set<std::string> unique_names;
    for (auto &name : names)
        if (!(is >> name) || !unique_names.insert(name).second) fail();

    Header header;header.names=names;
    header.species_count = typemap.empty() ? count : static_cast<int>(typemap.size());
    std::set<int> ids;
    for (const auto &[name, id] : typemap)
        if (!unique_names.count(name) || id < 0 || id >= header.species_count
            || !ids.insert(id).second) fail();
    for (int i = 0; i < count; ++i) {
        const auto found = typemap.find(names[i]);
        header.types.push_back(typemap.empty() ? i : found == typemap.end() ? -1 : found->second);
    }

    if (!std::getline(is >> std::ws, line)) fail();
    trim_comment(line, "#");
    std::istringstream grid(line);
    if (!(grid >> header.nrho >> header.drho >> header.nr >> header.dr >> header.cutoff)) fail();
    if (header.nr < 3 || header.nrho < 3 || !std::isfinite(header.dr)
        || !std::isfinite(header.drho) || !std::isfinite(header.cutoff)
        || header.dr <= 0 || header.drho <= 0 || header.cutoff <= 0) fail();
    header.rhomax = (header.nrho - 1) * header.drho;
    grid >> std::ws;
    if (!grid.eof()) {
        if (!(grid >> header.rhomax)) fail();
        header.he = true;
        header.rhomin = header.rhomax - (header.nrho - 1) * header.drho;
        grid >> std::ws;
        if (!grid.eof()) fail();
    }
    if (!std::isfinite(header.rhomax) || !std::isfinite(header.rhomin)
        || header.rhomax <= header.rhomin) fail();
    return header;
}

inline void read_element_header(std::istream &is, const std::string &file)
{
    int atomic_number;
    double mass, lattice_constant;
    std::string lattice_type;
    if (!(is >> atomic_number >> mass >> lattice_constant >> lattice_type)
        || !std::isfinite(mass) || !std::isfinite(lattice_constant))
        throw std::runtime_error("Invalid EAM element header in " + file);
}

inline Eigen::MatrixXd read_spline(std::istream &is, int count, double delta,
                                 const std::string &file)
{
    std::vector<double> values(count + 1, 0.0);
    for (int i = 1; i <= count; ++i)
        if (!(is >> values[i]) || !std::isfinite(values[i]))
            throw std::runtime_error("Incomplete or invalid EAM table in " + file);
    Eigen::MatrixXd spline;
    Spline::build_cubic_spline(count, delta, values, spline);
    return spline;
}

} // namespace Potentials::EAMTable
