// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <string_view>

#if defined(IMRMLP_CUDA) && defined(IMRMLP_HIP)
#error "Select exactly one GPU backend per build"
#endif

// Preserve consumers that select a backend with the original vendor macro.
#if (defined(IMRMLP_CUDA) || defined(IMRMLP_HIP)) && !defined(IMRMLP_GPU)
#define IMRMLP_GPU
#endif

namespace MLIPCore::GpuDirect {
// Backend macros describe this translation unit. Internal linkage also keeps
// standalone consumers from merging this helper with a different build mode.
static constexpr const char *compiled_backend()
{
#if defined(IMRMLP_HIP)
    return "hip";
#elif defined(IMRMLP_CUDA)
    return "cuda";
#else
    return "cpu";
#endif
}

inline constexpr std::string_view canonical_device(std::string_view device)
{
    return device == "dcu" ? std::string_view("hip") : device;
}

inline constexpr bool valid_device(std::string_view device)
{
    device = canonical_device(device);
    return device == "cpu" || device == "cuda" || device == "hip";
}

static constexpr bool device_supported(std::string_view device)
{
    device = canonical_device(device);
    return device == "cpu" || device == compiled_backend();
}
} // namespace MLIPCore::GpuDirect
