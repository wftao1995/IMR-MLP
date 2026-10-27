// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_backend.h"
#if defined(IMRMLP_HIP)
#include <hip/hip_runtime.h>
#else
#include <cuda_runtime.h>
#endif

// Only this header knows the vendor runtime. Kernel code shares typed wrappers;
// no CUDA-name macros leak into callers or third-party headers.
namespace MLIPCore::GpuDirect::Runtime {
#if defined(IMRMLP_HIP)
using gpuError_t = hipError_t;
using gpuDeviceProp = hipDeviceProp_t;
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
#define IMRMLP_GPU_CONSTANT(name) inline constexpr auto gpu##name = hip##name
#define IMRMLP_GPU_API(name) template<class... Args> inline auto gpu##name(Args... args) { return hip##name(args...); }
#else
using gpuError_t = cudaError_t;
using gpuDeviceProp = cudaDeviceProp;
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
#define IMRMLP_GPU_CONSTANT(name) inline constexpr auto gpu##name = cuda##name
#define IMRMLP_GPU_API(name) template<class... Args> inline auto gpu##name(Args... args) { return cuda##name(args...); }
#endif
IMRMLP_GPU_CONSTANT(Success);
IMRMLP_GPU_CONSTANT(MemcpyHostToDevice);
IMRMLP_GPU_CONSTANT(MemcpyDeviceToHost);
IMRMLP_GPU_CONSTANT(StreamNonBlocking);
IMRMLP_GPU_CONSTANT(EventDisableTiming);
IMRMLP_GPU_CONSTANT(FuncAttributeMaxDynamicSharedMemorySize);
IMRMLP_GPU_API(Malloc);
IMRMLP_GPU_API(Free);
IMRMLP_GPU_API(Memcpy);
IMRMLP_GPU_API(MemcpyAsync);
IMRMLP_GPU_API(Memset);
IMRMLP_GPU_API(MemGetInfo);
IMRMLP_GPU_API(GetDevice);
IMRMLP_GPU_API(SetDevice);
IMRMLP_GPU_API(GetDeviceCount);
IMRMLP_GPU_API(GetDeviceProperties);
IMRMLP_GPU_API(DeviceGetPCIBusId);
IMRMLP_GPU_API(GetErrorString);
IMRMLP_GPU_API(GetLastError);
IMRMLP_GPU_API(StreamCreateWithFlags);
IMRMLP_GPU_API(StreamDestroy);
IMRMLP_GPU_API(StreamSynchronize);
IMRMLP_GPU_API(StreamWaitEvent);
IMRMLP_GPU_API(EventCreateWithFlags);
IMRMLP_GPU_API(EventDestroy);
IMRMLP_GPU_API(EventRecord);
IMRMLP_GPU_API(EventSynchronize);
template<class Kernel>
inline gpuError_t gpuFuncSetAttribute(Kernel kernel,
    decltype(gpuFuncAttributeMaxDynamicSharedMemorySize) attribute, int value)
{
#if defined(IMRMLP_HIP)
    return hipFuncSetAttribute(reinterpret_cast<const void *>(kernel), attribute, value);
#else
    return cudaFuncSetAttribute(kernel, attribute, value);
#endif
}
#undef IMRMLP_GPU_API
#undef IMRMLP_GPU_CONSTANT

inline size_t max_dynamic_shared_memory(const gpuDeviceProp &props)
{
#if defined(IMRMLP_HIP)
    // DTK does not expose CUDA's sharedMemPerBlockOptin field. Its reported
    // per-block LDS limit is the allocation limit; do not invent an opt-in.
    return props.sharedMemPerBlock;
#else
    return props.sharedMemPerBlockOptin > 0
        ? props.sharedMemPerBlockOptin : props.sharedMemPerBlock;
#endif
}

__device__ inline double shuffle_down_32(double value, unsigned int offset)
{
#if defined(IMRMLP_HIP)
    // Two independent logical groups on wave64. Omitting width silently
    // mixes unrelated groups in the propagation and neural-cache kernels.
    return __shfl_down(value, offset, 32);
#else
    return __shfl_down_sync(0xffffffffu, value, offset, 32);
#endif
}
} // namespace MLIPCore::GpuDirect::Runtime
