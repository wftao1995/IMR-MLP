// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "utils/mpi_stubs.h"
#include <array>
#include <type_traits>

// tests.mk excludes this executable from real-MPI builds. These are the
// serial stub contracts; an MPI implementation must not stand in for them.
#ifndef IMRMLP_MPI
namespace {
void collective(int which, void *send, void *receive, int count, MPI_Datatype type)
{
    int counts[] = {count};
    int displacements[] = {0};
    switch (which)
    {
    case 0: MPI_Reduce(send, receive, count, type, MPI_SUM, 0, MPI_COMM_WORLD); break;
    case 1: MPI_Allreduce(send, receive, count, type, MPI_SUM, MPI_COMM_WORLD); break;
    case 2: MPI_Allgather(send, count, type, receive, count, type, MPI_COMM_WORLD); break;
    case 3: MPI_Allgatherv(send, count, type, receive, counts, displacements, type, MPI_COMM_WORLD); break;
    case 4: MPI_Gather(send, count, type, receive, count, type, 0, MPI_COMM_WORLD); break;
    case 5: MPI_Gatherv(send, count, type, receive, counts, displacements, type, 0, MPI_COMM_WORLD); break;
    }
}
}

TEST_CASE("Serial MPI collectives accept null buffers for zero counts", "[mpi-stubs]")
{
    const int which = GENERATE(0, 1, 2, 3, 4, 5);
    CAPTURE(which);
    REQUIRE_NOTHROW(collective(which, nullptr, nullptr, 0, MPI_DOUBLE));
    int sent = 17, received = 29;
    REQUIRE_NOTHROW(collective(which, &sent, &received, 0, MPI_INT));
    REQUIRE(sent == 17);
    REQUIRE(received == 29);
    // Only one side may have an allocated buffer in a caller's empty branch.
    REQUIRE_NOTHROW(collective(which, nullptr, &received, 0, MPI_INT));
    REQUIRE_NOTHROW(collective(which, &sent, nullptr, 0, MPI_INT));
    REQUIRE(received == 29);
}

TEMPLATE_TEST_CASE("Serial MPI collectives copy nonempty buffers exactly", "[mpi-stubs]", int, double)
{
    const int which = GENERATE(0, 1, 2, 3, 4, 5);
    CAPTURE(which);
    const MPI_Datatype type = std::is_same_v<TestType, int> ? MPI_INT : MPI_DOUBLE;
    std::array<TestType, 3> sent{3, -4, 7};
    std::array<TestType, 5> received{-100, 0, 0, 0, 200};
    collective(which, sent.data(), received.data() + 1, sent.size(), type);
    REQUIRE(received.front() == -100);
    REQUIRE(received.back() == 200);
    for (int i = 0; i < 3; ++i) REQUIRE(received[i + 1] == sent[i]);
    const auto saved = received;
    collective(which, nullptr, received.data() + 1, 0, type);
    REQUIRE(received == saved);
}
#endif
