// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <cstdint>

namespace MLIPCore::GpuDirect {

// Process-local identities, never pointer values: allocator address reuse must
// not revive a cache from an earlier object or initialization cycle.
inline std::uint64_t next_cache_identity()
{
    static std::atomic<std::uint64_t> next{1};
    return next.fetch_add(1, std::memory_order_relaxed);
}

struct GpuCacheToken {
    std::uint64_t owner = 0;
    std::uint64_t revision = 0;
    bool valid() const { return owner != 0 && revision != 0; }
    bool operator==(const GpuCacheToken &other) const {
        return owner == other.owner && revision == other.revision;
    }
};

// Copying a batch copies its data, not its device residency. Moving also gets
// a fresh identity; this is inexpensive and leaves the moved-from owner safe.
struct GpuCacheIdentity {
    std::uint64_t value = next_cache_identity();
    GpuCacheIdentity() = default;
    GpuCacheIdentity(const GpuCacheIdentity &) {}
    GpuCacheIdentity &operator=(const GpuCacheIdentity &) {
        value = next_cache_identity();
        return *this;
    }
};

} // namespace MLIPCore::GpuDirect
