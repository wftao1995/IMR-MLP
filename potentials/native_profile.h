#ifndef IMRMLP_NATIVE_PROFILE_H
#define IMRMLP_NATIVE_PROFILE_H

// Host-only diagnostic instrumentation. Production builds do not define
// IMRMLP_NATIVE_HOST_PROFILE: scope arguments are then not evaluated at all.
// Scopes are inclusive and nested. No GPU synchronization is added here.
#ifdef IMRMLP_NATIVE_HOST_PROFILE
#include <array>
#include <cstddef>
#include <cstdint>

namespace Potentials {
struct NativeHostProfileRecord {
  struct Entry {
    const char *label = nullptr;
    int parent = -1;
    std::uint64_t calls = 0, inclusive_ns = 0, child_ns = 0;
  };
  std::array<Entry,128> entries{};
  std::size_t count = 0;
  int active_entry = -1;
  bool overflow = false;
};

class NativeHostProfileFrame {
 public:
  explicit NativeHostProfileFrame(NativeHostProfileRecord &) noexcept;
  ~NativeHostProfileFrame();
  NativeHostProfileFrame(const NativeHostProfileFrame &) = delete;
  NativeHostProfileFrame &operator=(const NativeHostProfileFrame &) = delete;
 private:
  NativeHostProfileRecord *previous_;
};

class NativeHostProfileScope {
 public:
  // Labels must be static strings, retained until the enclosing frame is written.
  explicit NativeHostProfileScope(const char *) noexcept;
  ~NativeHostProfileScope();
  NativeHostProfileScope(const NativeHostProfileScope &) = delete;
  NativeHostProfileScope &operator=(const NativeHostProfileScope &) = delete;
 private:
  NativeHostProfileRecord *record_ = nullptr;
  int entry_ = -1, previous_ = -1;
  std::uint64_t start_ = 0;
};

struct NativeHostProfileMetadata {
  int rank = 0, owned = 0, total = 0;
  std::int64_t step = 0;
  std::uint64_t call = 0, candidates = 0, active_edges = 0;
  std::uint64_t image_groups = 0, topology_revision = 0;
  int eflag = 0, vflag = 0;
  bool rebuilt = false;
};

// Call after all measured scopes have ended. JSONL goes to
// $IMRMLP_NATIVE_PROFILE_DIR/native-host-profile.rank-R.pid-P.jsonl (default '.').
// Output I/O is deliberately outside the timings but can perturb later calls.
// Returns false on write/open failure, without throwing during MPI evaluation.
bool write_native_host_profile(const NativeHostProfileRecord &,
                               const NativeHostProfileMetadata &) noexcept;
}

#define IMRMLP_NATIVE_PROFILE_JOIN_I(a,b) a##b
#define IMRMLP_NATIVE_PROFILE_JOIN(a,b) IMRMLP_NATIVE_PROFILE_JOIN_I(a,b)
#define IMRMLP_NATIVE_PROFILE_SCOPE(label) \
  ::Potentials::NativeHostProfileScope IMRMLP_NATIVE_PROFILE_JOIN(gpml_native_profile_scope_,__COUNTER__)(label)
#else
#define IMRMLP_NATIVE_PROFILE_SCOPE(label) ((void)0)
#endif
#endif
