// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "utils/comm.h"
#include <stdexcept>
#include <utility>

namespace Training {
// All ranks call at the same phase boundary. Broadcast the first failing
// rank's message so a local failure cannot strand peers in the next reduction.
inline void collective_check(const std::string& local) {
#ifdef IMRMLP_MPI
    int source=local.empty() ? mpi.size : mpi.rank;
    int first=mpi.size;
    MPI_Allreduce(&source,&first,1,MPI_INT,MPI_MIN,mpi.comm);
    if(first==mpi.size) return;
    int length=mpi.rank==first ? static_cast<int>(local.size()) : 0;
    MPI_Bcast(&length,1,MPI_INT,first,mpi.comm);
    std::string message=mpi.rank==first ? local : std::string(length,'\0');
    MPI_Bcast(message.data(),length,MPI_CHAR,first,mpi.comm);
    throw std::runtime_error("Training failure on rank "+std::to_string(first)+": "+message);
#else
    if(!local.empty()) throw std::runtime_error(local);
#endif
}
// The operation must be local, or already handle failures inside its own
// collectives. This wrapper cannot recover from a failed MPI communicator.
template<class F> void collective_try(F&& operation) {
    std::string error;
    try {std::forward<F>(operation)();}
    catch(const std::exception& e) {error=e.what();}
    catch(...) {error="Unknown training exception";}
    collective_check(error);
}
}
