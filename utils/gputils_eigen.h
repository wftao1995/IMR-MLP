// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file gputils_eigen.h
 * @brief Eigen/fmt-dependent template utilities extracted from gputils.h.
 *
 * Include this file when you need matrix I/O, vector I/O, or the
 * buildOrthogonalFromVector helper.
 */

#include "gputils.h"
#include "utils/compressed_stream.h"
#include <Eigen/Dense>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <limits>

// row or col major does not matter
template <typename MatrixType>
void writeMatrixToFile(const std::string& filename, MatrixType &matrix) {
    auto ofs = open_ofile(filename);
    if (!ofs) GPERROR("Cannot open output matrix: " + filename);
    int size1, size2;
    size1 = matrix.rows();
    size2 = matrix.cols();
    *ofs << fmt::format("{} {}\n", size1, size2);
    for (int i = 0; i < size1; ++i) {
        *ofs << fmt::format("{}\n", fmt::join(matrix.row(i).begin(), matrix.row(i).end(), " "));
    }
    if (!close_ofile(ofs)) GPERROR("Failed to write output matrix: " + filename);
}

void readMatrixFromFile(const std::string& filename, Eigen::MatrixXd &matrix);

namespace detail_binary_eigen {
inline void read_exact(std::istream &input, char *data, std::size_t bytes,
                       const std::string &filename) {
    constexpr std::size_t chunk = 1 << 20;
    while (bytes) {
        const std::size_t n = std::min(bytes, chunk);
        if (!input.read(data, static_cast<std::streamsize>(n)))
            GPERROR("Truncated binary matrix/vector: " + filename);
        bytes -= n; data += n;
    }
}
inline std::size_t payload_size(int rows, int cols, std::size_t scalar_size,
                                const std::string &filename) {
    if (rows < 0 || cols < 0 || (rows && static_cast<std::size_t>(cols) >
        static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()) / scalar_size / rows))
        GPERROR("Invalid binary matrix/vector dimensions: " + filename);
    return static_cast<std::size_t>(rows) * cols * scalar_size;
}
}

template <typename MatrixType>
void readMatrixFromBinFile(const std::string &filename, MatrixType &matrix) {
    auto input = open_ifile(filename);
    if (!input || !*input) GPERROR("Cannot open binary matrix: " + filename);
    int rows = 0, cols = 0;
    unsigned char rowmajor = 0;
    using namespace detail_binary_eigen;
    read_exact(*input, reinterpret_cast<char*>(&rows), sizeof(rows), filename);
    read_exact(*input, reinterpret_cast<char*>(&cols), sizeof(cols), filename);
    read_exact(*input, reinterpret_cast<char*>(&rowmajor), sizeof(rowmajor), filename);
    if (rowmajor > 1) GPERROR("Invalid binary matrix layout: " + filename);
    using Scalar = typename MatrixType::Scalar;
    const auto bytes = payload_size(rows, cols, sizeof(Scalar), filename);
    Eigen::Matrix<Scalar, -1, -1> raw(rows, cols);
    read_exact(*input, reinterpret_cast<char*>(raw.data()), bytes, filename);
    if (!raw.allFinite()) GPERROR("Non-finite binary matrix: " + filename);
    if (rowmajor) matrix = Eigen::Map<const Eigen::Matrix<Scalar, -1, -1, Eigen::RowMajor>>(raw.data(), rows, cols);
    else matrix = raw;
}

template <typename VectorType>
void readVectorFromBinFile(const std::string &filename, VectorType &vector) {
    auto input = open_ifile(filename);
    if (!input || !*input) GPERROR("Cannot open binary vector: " + filename);
    int size = 0;
    using namespace detail_binary_eigen;
    read_exact(*input, reinterpret_cast<char*>(&size), sizeof(size), filename);
    const auto bytes = payload_size(size, 1, sizeof(typename VectorType::Scalar), filename);
    VectorType temporary(size);
    read_exact(*input, reinterpret_cast<char*>(temporary.data()), bytes, filename);
    if (!temporary.allFinite()) GPERROR("Non-finite binary vector: " + filename);
    vector = std::move(temporary);
}

template <typename MatrixType>
void writeMatrixToBinFile(const std::string &filename, MatrixType &matrix) {
    auto ofs = open_ofile(filename, false, true);
    if (!ofs) GPERROR("Cannot open output matrix: " + filename);
    int size1 = matrix.rows();
    int size2 = matrix.cols();
    bool rowmajor = false;
    if (matrix.IsRowMajor) {
        rowmajor = true;
    } else {
        rowmajor = false;
    }
    ofs->write(reinterpret_cast<const char*>(&size1), sizeof(size1));
    ofs->write(reinterpret_cast<const char*>(&size2), sizeof(size2));
    ofs->write(reinterpret_cast<const char*>(&rowmajor), sizeof(bool));
    ofs->write(reinterpret_cast<const char*>(matrix.data()), sizeof(typename MatrixType::Scalar)*size1*size2);
    if (!close_ofile(ofs)) GPERROR("Failed to write output matrix: " + filename);
}

template <typename VectorType>
void writeVectorToBinFile(const std::string &filename, VectorType &vector) {
    auto ofs = open_ofile(filename, false, true);
    if (!ofs) GPERROR("Cannot open output vector: " + filename);
    int size = vector.size();
    ofs->write(reinterpret_cast<const char*>(&size), sizeof(size));
    ofs->write(reinterpret_cast<const char*>(vector.data()), sizeof(typename VectorType::Scalar)*size);
    if (!close_ofile(ofs)) GPERROR("Failed to write output vector: " + filename);
}

// input vector will be the first row of the output orthogonal matrix
template<typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic>
buildOrthogonalFromVector(const Eigen::MatrixBase<Derived>& vec) {
    using Scalar = typename Derived::Scalar;
    using Index = typename Derived::Index;

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> v;
    if constexpr (Derived::IsVectorAtCompileTime && Derived::ColsAtCompileTime == 1) {
        v = vec;
    } else if constexpr (Derived::IsVectorAtCompileTime && Derived::RowsAtCompileTime == 1) {
        v = vec.transpose();
    } else {
        static_assert(Derived::IsVectorAtCompileTime,
                     "Input must be a vector (row or column)");
        v = vec;
    }

    const Index n = v.size();

    Scalar norm = v.norm();
    if (norm < Eigen::NumTraits<Scalar>::epsilon()) {
        throw std::invalid_argument("Input vector has zero norm");
    }
    v.normalize();

    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> e1 =
        Eigen::Matrix<Scalar, Eigen::Dynamic, 1>::Zero(n);
    e1(0) = Scalar(1);

    // Householder w
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> w = v - e1;
    Scalar w_norm = w.norm();

    if (w_norm < Eigen::NumTraits<Scalar>::epsilon()) {
        return Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n);
    }

    w.normalize();

    // Householder Q = I - 2ww^T
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> Q =
        Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>::Identity(n, n) -
        Scalar(2) * w * w.transpose();

    return Q;
}
