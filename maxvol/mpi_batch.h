#pragma once
#include "utils/comm.h"
#include <array>
#include <stdexcept>

namespace MaxVolMPI
{
// All ranks enter before the next collective. Allocation/factorization failures
// on one rank must not leave the others waiting in a broadcast.
template <class F> void local(MPI_Comm comm, F &&operation)
{
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size == 1)
    {
        operation();
        return;
    }
    std::array<int, 256> message{};
    int source = size;
    try
    {
        operation();
    }
    catch (const std::exception &e)
    {
        source = rank;
        for (size_t i = 0; i + 1 < message.size() && e.what()[i]; ++i)
            message[i] = static_cast<unsigned char>(e.what()[i]);
    }
    catch (...)
    {
        source = rank;
        message[0] = '?';
    }
    int first = size;
    MPI_Allreduce(&source, &first, 1, MPI_INT, MPI_MIN, comm);
    if (first == size)
        return;
    MPI_Bcast(message.data(), int(message.size()), MPI_INT, first, comm);
    std::string detail;
    for (int c : message)
    {
        if (!c)
            break;
        detail.push_back(char(c));
    }
    throw std::runtime_error("MaxVol MPI failure on rank " + std::to_string(first) + ": " + detail);
}
}
