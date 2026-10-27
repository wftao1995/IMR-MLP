// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_MPI_STUBS_H
#define IMRMLP_MPI_STUBS_H


#ifdef IMRMLP_MPI
#include <mpi.h>
#else


#define MPI_COMM_WORLD     0
#define MPI_COMM_SELF      1
#define MPI_SUCCESS        0
#define MPI_INT            (sizeof(int))
#define MPI_DOUBLE         (sizeof(double))
#define MPI_C_BOOL         (sizeof(bool))
#define MPI_STATUS_IGNORE  0


enum { MPI_MAX, MPI_MIN, MPI_SUM };

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Status;
int MPI_Init(int *, char ***);
int MPI_Finalize();
int MPI_Comm_rank(MPI_Comm, int *);
int MPI_Comm_size(MPI_Comm, int *);
void MPI_Barrier(MPI_Comm);
void MPI_Bcast(void* buffer, int count, MPI_Datatype datatype,
              int root, MPI_Comm comm);
void MPI_Reduce(void* sendbuf, void* recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root,
               MPI_Comm comm);
void MPI_Allreduce(void* sendbuf, void* recvbuf, int count,
                  MPI_Datatype datatype, MPI_Op op, MPI_Comm comm);
void MPI_Allgather(void* sendbuf, int sendcount,
                  MPI_Datatype sendtype, void* recvbuf, int recvcount,
                  MPI_Datatype recvtype, MPI_Comm comm);
void MPI_Allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, 
                    void* recvbuf, int* recvcounts, int* displs, 
                    MPI_Datatype recvtype, MPI_Comm comm);
void MPI_Gather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, 
                   int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm);
void MPI_Gatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, 
                 int* recvcounts, int* displs, MPI_Datatype recvtype, int root, MPI_Comm comm);
// buf is send/recive
// One should not call this without a real MPI library
void MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
                               int dest, int sendtag, int source, int recvtag,
                               MPI_Comm comm, MPI_Status *status);

#endif // IMRMLP_MPI

#endif // MPI_STUBS
