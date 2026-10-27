// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "mathextra/combinatorics.h"

TEST_CASE("factorial", "[combinatorics]") {
    REQUIRE(Combinatorics::factorial(0) == 1);
    REQUIRE(Combinatorics::factorial(1) == 1);
    REQUIRE(Combinatorics::factorial(3) == 6);
    REQUIRE(Combinatorics::factorial(5) == 120);
}

TEST_CASE("ElementaryPermutation count and range", "[combinatorics]") {
    const size_t n = 5;
    Combinatorics::ElementaryPermutation perm(n);

    size_t count = 1; // initial permutation already counted
    while (perm.next() != std::pair<size_t, size_t>(0, 0)) {
        for (size_t i = 0; i < n; ++i) {
            REQUIRE(perm[i] < n);
        }
        ++count;
    }

    REQUIRE(count == Combinatorics::factorial(n));
}

TEST_CASE("ElementaryPermutation parity balanced", "[combinatorics]") {
    Combinatorics::ElementaryPermutation perm(3);
    int even = 0, odd = 0;

    // parity(): 1 = even, -1 = odd
    if (perm.parity() == 1) even++; else odd++;

    while (perm.next() != std::pair<size_t, size_t>(0, 0)) {
        if (perm.parity() == 1)
            even++;
        else
            odd++;
    }
    REQUIRE(even == 3);
    REQUIRE(odd == 3);
}
