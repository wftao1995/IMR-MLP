#pragma once

#include "mlip.h"
#include <mpi.h>
#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace MLIPCore {

// Run a rank-local initialization stage, then propagate the first error before
// any rank enters the next collective (for example GPU selection). The action
// must not itself call MPI collectives. Fixed error storage also covers bad_alloc.
template<class Action>
inline void collective_try(MPI_Comm world, Action &&action)
{
    char message[2048] = {};
    bool failed = false;
    try { action(); }
    catch (const std::exception &e) {
        failed = true;
        std::strncpy(message, e.what(), sizeof(message)-1);
    }
    catch (...) { failed = true; }
    if (failed && !message[0]) std::strcpy(message, "Unknown error while initializing the MTP runtime");
    int rank = 0;
    MPI_Comm_rank(world, &rank);
    // Older LAMMPS MPI stubs take a non-const send buffer for Allreduce.
    int failed_rank = failed ? rank : std::numeric_limits<int>::max();
    int first_failed_rank = 0;
    MPI_Allreduce(&failed_rank, &first_failed_rank, 1, MPI_INT, MPI_MIN, world);
    if (first_failed_rank == std::numeric_limits<int>::max()) return;
    MPI_Bcast(message, sizeof(message), MPI_CHAR, first_failed_rank, world);
    throw std::runtime_error(message);
}

// Only rank zero accesses the filesystem. Broadcast the portable text prefix,
// never object bytes containing pointers, and leave the MVS binary payload out.
inline std::string read_model_text_mpi(MPI_Comm world, const char *filename)
{
    int rank = 0;
    MPI_Comm_rank(world, &rank);
    std::string text, error;
    if (rank == 0) {
        try { text = read_model_text(filename); }
        catch (const std::exception &e) { error = e.what(); }
    }
    int error_length = rank == 0 ? static_cast<int>(error.size()) : 0;
    MPI_Bcast(&error_length, 1, MPI_INT, 0, world);
    if (error_length) {
        error.resize(error_length);
        MPI_Bcast(error.data(), error_length, MPI_CHAR, 0, world);
        throw std::runtime_error(error);
    }
    unsigned long long length = text.size();
#ifndef MPI_STUBS
    // LAMMPS serial stubs have no MPI_UNSIGNED_LONG_LONG. There is only
    // rank zero in that build, so the local length already has its final value.
    MPI_Bcast(&length, 1, MPI_UNSIGNED_LONG_LONG, 0, world);
#endif
    int failed = 0;
    try {
        if (length > text.max_size()) throw std::length_error("MTP model text is too large");
        text.resize(static_cast<std::size_t>(length));
    } catch (const std::exception &) { failed = 1; }
    int any_failed = 0;
    MPI_Allreduce(&failed, &any_failed, 1, MPI_INT, MPI_MAX, world);
    if (any_failed) throw std::runtime_error("Cannot allocate MTP model text on every MPI rank");
    for (std::size_t offset = 0; offset < text.size();) {
        const auto size = static_cast<int>(std::min<std::size_t>(
            text.size() - offset, std::numeric_limits<int>::max()));
        MPI_Bcast(text.data() + offset, size, MPI_CHAR, 0, world);
        offset += size;
    }
    return text;
}

} // namespace MLIPCore
