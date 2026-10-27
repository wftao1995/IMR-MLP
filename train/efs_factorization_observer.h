#pragma once
#include <array>

namespace Training {
// Optional process-local accounting for the serial experimental EFS path.
// One event is one decomposition, not an estimate of equal computational work.
enum class EFSFactorization { rows_qr, mpi_merge_qr, channels_qr, bias_svd,
    fit_qr, cur_range_qr, selector_qr, deletion_ldlt, count };
using EFSFactorizationCounts=std::array<unsigned long long,static_cast<int>(EFSFactorization::count)>;
void observe_efs_factorizations(EFSFactorizationCounts *counts);
void count_efs_factorization(EFSFactorization kind);
} // namespace Training
