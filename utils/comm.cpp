// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "comm.h"

MPI_data mpi;

void MPI_data::InitComm(MPI_Comm _comm)
{
    comm = _comm;
    MPI_Comm_size(comm, &size);
    MPI_Comm_rank(comm, &rank);
#ifdef IMRMLP_MPI
    fnm_ending = "_" + std::to_string(rank);
#else
    fnm_ending = "";
#endif
}