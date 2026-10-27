// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_COMM_H
#define IMRMLP_COMM_H

#include "mpi_stubs.h"
#include <string>

struct MPI_data
{
    int rank = 0; // Number of this processor in the mpi grop
    int size = 1; //  Number of ranks in MPI gruop
    std::string fnm_ending = "";
    MPI_Comm comm; // mpi communication group
    void InitComm(MPI_Comm _comm);  // Initialization, updating the object
};

// Common communicator
extern MPI_data mpi; 

#endif