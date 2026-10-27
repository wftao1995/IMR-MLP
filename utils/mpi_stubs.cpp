// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mpi_stubs.h"
#include <cstring>
#ifdef IMRMLP_DEBUG_MPI_STUB
#include <stdexcept>
#endif


#ifndef IMRMLP_MPI


using namespace std;


int stubs_sizeof(MPI_Datatype datatype)
{
    return datatype;
}

int MPI_Init(int *, char ***) {
    return 0;
}

int MPI_Finalize() {
    return 0;
}

int MPI_Comm_rank(MPI_Comm comm, int * rank) 
{
    *rank = 0;
    return 0;
}

int MPI_Comm_size(MPI_Comm comm, int * size) 
{
    *size = 1;
    return 0;
}

void MPI_Barrier(MPI_Comm) {
    ;
}

void MPI_Bcast(void* buffer, int count, MPI_Datatype datatype, int root, MPI_Comm comm) 
{
#ifdef IMRMLP_DEBUG_MPI_STUB
    if (root != 0)
        throw std::runtime_error("Unhandled case in MPI STUBS");
#else
    ;
#endif 
}

void MPI_Reduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, 
                int root, MPI_Comm comm) 
{
    if (count == 0) return;
#ifdef IMRMLP_DEBUG_MPI_STUB
    if ((op == MPI_MAX || op == MPI_MIN || op == MPI_SUM) && (sendbuf != recvbuf) && (root == 0))
        memcpy(recvbuf, sendbuf, count*stubs_sizeof(datatype));
    else
        throw std::runtime_error("Unhandled case in MPI STUBS");
#else
    memcpy(recvbuf, sendbuf, count*stubs_sizeof(datatype));
#endif 
}

void MPI_Allreduce(void* sendbuf, void* recvbuf, int count, MPI_Datatype datatype, MPI_Op op, 
                   MPI_Comm comm) 
{
    if (count == 0) return;
#ifdef IMRMLP_DEBUG_MPI_STUB
    if ((op == MPI_MAX || op == MPI_MIN || op == MPI_SUM) && (sendbuf != recvbuf))
        memcpy(recvbuf, sendbuf, count*stubs_sizeof(datatype));
    else
        throw std::runtime_error("Unhandled case in MPI STUBS");
#else
    memcpy(recvbuf, sendbuf, count*stubs_sizeof(datatype));
#endif 
}

void MPI_Allgather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, 
                   int recvcount, MPI_Datatype recvtype, MPI_Comm comm) 
{
    if (sendcount == 0) return;
#ifdef IMRMLP_DEBUG_MPI_STUB
    if (sendcount == recvcount && sendtype == recvtype && sendbuf != recvbuf)
        memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
    else
        throw std::runtime_error("Unhandled case in MPI STUBS");
#else
    memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
#endif 
}

void MPI_Allgatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf,
                    int* recvcounts, int* displs, MPI_Datatype recvtype, MPI_Comm comm) 
{
    if (sendcount == 0) return;
#ifdef IMRMLP_DEBUG_MPI_STUB
    if (sendcount == recvcounts[0] && sendtype == recvtype && sendbuf != recvbuf)
        memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
    else
        throw std::runtime_error("Unhandled case in MPI STUBS");
#else
    memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
#endif 
}

void MPI_Gather(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, 
                int recvcount, MPI_Datatype recvtype, int root, MPI_Comm comm) 
{
    if (sendcount == 0) return;
#ifdef IMRMLP_DEBUG_MPI_STUB
    if (sendcount == recvcount && sendtype == recvtype && sendbuf != recvbuf && root == 0)
        memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
    else
        throw std::runtime_error("Unhandled case in MPI STUBS");
#else
    memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
#endif 
}

void MPI_Gatherv(void* sendbuf, int sendcount, MPI_Datatype sendtype, void* recvbuf, 
                 int* recvcounts, int* displs, MPI_Datatype recvtype, int root, MPI_Comm comm) 
{
    if (sendcount == 0) return;
#ifdef IMRMLP_DEBUG_MPI_STUB
    if (sendcount == recvcounts[0] && sendtype == recvtype && sendbuf != recvbuf && root == 0)
        memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
    else
        throw std::runtime_error("Unhandled case in MPI STUBS");
#else
    memcpy(recvbuf, sendbuf, sendcount*stubs_sizeof(sendtype));
#endif 
}
// One should not call this without a real MPI library
void MPI_Sendrecv_replace(void *buf, int count, MPI_Datatype datatype,
                         int dest, int sendtag, int source, int recvtag,
                         MPI_Comm comm, MPI_Status *status) {
    ;
}


#endif