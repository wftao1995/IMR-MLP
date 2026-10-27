#pragma once
#include <cstddef>

namespace BasisAnalysis {
// Opt-in instrumentation for finite offline experiments. No matrix, solver,
// sampling, or random-number decisions depend on these counters. The caller
// owns the counter and must reset it before its lifetime ends.
struct FactorizationCounts { std::size_t eigen = 0, ldlt = 0; };
void observe_factorizations(FactorizationCounts *counts);
void count_eigen_factorization();
void count_ldlt_factorization();
}
