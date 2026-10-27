// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mtp_gpu_direct.h"
#include <mpi.h>
#include <array>
#include <cstdio>
#include <vector>

namespace MLIPCore::GpuDirect {

// Collective over world. Handles either all node devices visible to each rank
// or a scheduler giving each rank its own single-device visibility mask.
inline bool select_mpi_device(MPI_Comm world, int requested, int *selected,
                              std::string *error)
{
    MPI_Comm local;
#ifdef MPI_STUBS
    // LAMMPS's serial stubs do not implement shared-memory communicator split.
    MPI_Comm_dup(world, &local);
#else
    MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
#endif
    int rank = 0, size = 1, count = 0;
    MPI_Comm_rank(local, &rank);
    MPI_Comm_size(local, &size);
    std::string detail;
    int failed = !MTPGpuDirect::available_device_count(&count, &detail);
    *selected = requested >= 0 ? requested : (count == 1 ? 0 : rank);
    if (*selected < 0 || *selected >= count) {
        failed = 1;
        detail = "MPI rank has no matching visible GPU";
    }
    MTPGpuDeviceInfo info;
    if (!failed && !MTPGpuDirect::device_info(*selected, &info, &detail)) failed = 1;
    std::array<char, 128> identity{};
    std::snprintf(identity.data(), identity.size(), "%s", info.identity.c_str());
    std::vector<char> identities(static_cast<std::size_t>(size) * identity.size());
    MPI_Allgather(identity.data(), identity.size(), MPI_CHAR,
                  identities.data(), identity.size(), MPI_CHAR, local);
    if (!failed && requested < 0) {
        for (int other = 0; other < size; ++other) {
            if (other != rank && info.identity ==
                    identities.data() + other * identity.size()) {
                failed = 1;
                detail = "automatic MPI placement assigned multiple ranks to GPU " +
                    info.identity + "; request distinct devices or explicit sharing";
            }
        }
    }
    MPI_Comm_free(&local);
    int world_rank=0, world_size=0;
    MPI_Comm_rank(world, &world_rank);
    MPI_Comm_size(world, &world_size);
    int source=failed ? world_rank : world_size, first=world_size;
    MPI_Allreduce(&source, &first, 1, MPI_INT, MPI_MIN, world);
    if (first < world_size) {
        int length=world_rank==first ? static_cast<int>(detail.size()) : 0;
        MPI_Bcast(&length, 1, MPI_INT, first, world);
        detail.resize(length);
        MPI_Bcast(detail.data(), length, MPI_CHAR, first, world);
        if (error) *error="GPU placement failed on rank " + std::to_string(first) + ": " + detail;
        return false;
    }
    return true;
}
} // namespace MLIPCore::GpuDirect
