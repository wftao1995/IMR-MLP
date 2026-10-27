// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace IMRMLPNN::CoeffIO {

// A negative marker cannot be confused with a legacy hidden-layer size.
inline constexpr std::int32_t marker = -0x4e4e;
struct Header { int layers; int activation = 0; bool legacy = false; };

[[noreturn]] inline void fail(const std::string &file) {
    throw std::runtime_error("Invalid or incomplete neural network file: " + file);
}

inline Header read_header(std::istream &is, const std::string &file, int legacy_layers) {
    std::int32_t first;
    if (!is.read(reinterpret_cast<char *>(&first), sizeof(first))) fail(file);
    if (first != marker) {
        if (first <= 0 || (legacy_layers != 1 && legacy_layers != 2)) fail(file);
        is.seekg(0);
        return {legacy_layers, 0, true}; // Legacy files use tanh.
    }
    std::int32_t fields[3]; // version, layers, activation
    if (!is.read(reinterpret_cast<char *>(fields), sizeof(fields))
        || fields[0] != 1 || (fields[1] != 1 && fields[1] != 2)
        || fields[2] < 0 || fields[2] > 2) fail(file);
    return {fields[1], fields[2]};
}

inline int layers(const std::string &file, int legacy_layers) {
    std::ifstream is(file, std::ios::binary);
    return read_header(is, file, legacy_layers).layers;
}

inline void write_header(std::ostream &os, int layers, int activation) {
    if (activation < 0 || activation > 2)
        throw std::runtime_error("Invalid neural network activation function");
    const std::int32_t fields[] = {marker, 1, layers, activation};
    os.write(reinterpret_cast<const char *>(fields), sizeof(fields));
}

inline bool check_payload(std::istream &is, const std::string &file, bool legacy,
                          int input, int hidden1, int hidden2 = 0) {
    if (!is || input <= 0 || hidden1 <= 0 || hidden2 < 0) fail(file);
    const std::uint64_t ni = input, n1 = hidden1, n2 = hidden2;
    const auto coefficients = n2 == 0 ? ni * n1 + 2 * n1
                                     : ni * n1 + n1 + n1 * n2 + 2 * n2;
    if (coefficients > std::numeric_limits<int>::max()) fail(file);
    const auto start = is.tellg();
    is.seekg(0, std::ios::end);
    const auto end = is.tellg();
    if (start < 0 || end < start) fail(file);
    const auto bytes = static_cast<std::uint64_t>(end - start);
    const bool has_normalization = bytes == (coefficients + 2 * ni) * sizeof(double);
    if (!has_normalization) {
        if (!legacy || bytes != coefficients * sizeof(double)) fail(file);
        std::cerr << "Warning: legacy neural network file has no normalization; using identity: "
                  << file << '\n';
    }
    is.seekg(start);
    return has_normalization;
}

template<class Matrix>
void read_array(std::istream &is, Matrix &array, const std::string &file) {
    if (!is.read(reinterpret_cast<char *>(array.data()), array.size() * sizeof(double))
        || !array.allFinite()) fail(file);
}

inline void finish_output(std::ofstream &os, const std::string &file) {
    os.close();
    if (!os) throw std::runtime_error("Failed to write neural network file: " + file);
}

} // namespace IMRMLPNN::CoeffIO
