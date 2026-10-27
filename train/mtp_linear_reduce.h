#pragma once

#include <limits>
#include <stdexcept>
#include <vector>

namespace Training {

// Legacy Trainer_MTP normal systems: send RHS and only the solver's triangle
// in one collective. Buffers belong to the trainer and survive repeated fits.
template<class Matrix, class Vector, class Reduce>
void reduce_mtp_linear_system(const Matrix &local_matrix, const Vector &local_rhs,
                               Matrix &matrix, Vector &rhs, int n, bool upper,
                               bool root, std::vector<double> &send,
                               std::vector<double> &receive, Reduce reduce) {
    const auto width = static_cast<std::size_t>(n);
    const auto count = width + width * (width + 1) / 2;
    if (n < 0 || count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("MTP linear reduction exceeds MPI count range");
    send.resize(count);
    receive.resize(root ? count : 0);
    std::size_t position = 0;
    for (int i = 0; i < n; ++i) send[position++] = local_rhs[i];
    for (int i = 0; i < n; ++i)
        for (int j = upper ? i : 0; j < (upper ? n : i + 1); ++j)
            send[position++] = local_matrix(i, j);
    reduce(send.data(), receive.data(), static_cast<int>(count));
    if (!root) return;
    position = 0;
    for (int i = 0; i < n; ++i) rhs[i] = receive[position++];
    for (int i = 0; i < n; ++i)
        for (int j = upper ? i : 0; j < (upper ? n : i + 1); ++j)
            matrix(i, j) = receive[position++];
}

} // namespace Training
