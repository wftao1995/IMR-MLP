// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

// Pre-main stdio configuration for unit-test binaries.
//
// GPIO's constructor calls std::ios_base::sync_with_stdio(false), which swaps
// std::cout's rdbuf. Catch2's console reporter captures std::cout.rdbuf() when
// the session starts, so a later swap silently orphans all reporter output
// (test names, failures, summaries) while the process still exits with the
// correct code. Applying the same setting here — before main runs — makes the
// lazy call in GPIO() a no-op and keeps the reporter alive.
//
// Linked into every test binary via unitest/tests.mk (main_init.o).

#include <ios>
#include <iostream>

namespace {

struct MainInit {
    MainInit() {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(nullptr);
    }
};
MainInit g_main_init;

}  // namespace
