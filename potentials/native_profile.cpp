#include "native_profile.h"

#ifdef IMRMLP_NATIVE_HOST_PROFILE
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace Potentials {
namespace {
// One out-of-line TLS instance shared by the IMR-MLP library and LAMMPS plugin.
thread_local NativeHostProfileRecord *active_record = nullptr;
std::uint64_t now_ns() noexcept
{
  return static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count());
}
void quoted(FILE *file,const char *text)
{
  fputc('"',file);
  for (const unsigned char *p = reinterpret_cast<const unsigned char *>(text); *p; ++p) {
    if (*p == '"' || *p == '\\') { fputc('\\',file); fputc(*p,file); }
    else if (*p < 32) fprintf(file,"\\u%04x",static_cast<unsigned>(*p));
    else fputc(*p,file);
  }
  fputc('"',file);
}
struct Output {
  FILE *file = nullptr;
  int rank = -1;
  bool failed = false;
  ~Output() { if (file) fclose(file); }
};
thread_local Output output;
}

NativeHostProfileFrame::NativeHostProfileFrame(NativeHostProfileRecord &record) noexcept
  : previous_(active_record)
{ active_record = &record; }
NativeHostProfileFrame::~NativeHostProfileFrame() { active_record = previous_; }

NativeHostProfileScope::NativeHostProfileScope(const char *label) noexcept
  : record_(active_record)
{
  if (!record_) return;
  previous_ = record_->active_entry;
  for (std::size_t i = 0; i < record_->count; ++i)
    if (record_->entries[i].parent == previous_ && !std::strcmp(record_->entries[i].label,label)) {
      entry_ = static_cast<int>(i); break;
    }
  if (entry_ < 0) {
    if (record_->count == record_->entries.size()) {
      record_->overflow = true; record_ = nullptr; return;
    }
    entry_ = static_cast<int>(record_->count++);
    auto &entry = record_->entries[entry_];
    entry.label = label; entry.parent = previous_;
  }
  record_->active_entry = entry_;
  start_ = now_ns();
}
NativeHostProfileScope::~NativeHostProfileScope()
{
  if (!record_) return;
  const auto elapsed = now_ns()-start_;
  auto &entry = record_->entries[entry_];
  ++entry.calls; entry.inclusive_ns += elapsed;
  if (previous_ >= 0) record_->entries[previous_].child_ns += elapsed;
  record_->active_entry = previous_;
}

bool write_native_host_profile(const NativeHostProfileRecord &record,
                               const NativeHostProfileMetadata &m) noexcept
{
  if (output.failed) return false;
  try {
    if (!output.file || output.rank != m.rank) {
      if (output.file) { fclose(output.file); output.file = nullptr; }
      const char *directory = std::getenv("IMRMLP_NATIVE_PROFILE_DIR");
      std::filesystem::path root = directory && *directory ? directory : ".";
      std::filesystem::create_directories(root);
      const auto path = root/("native-host-profile.rank-"+std::to_string(m.rank)+
        ".pid-"+std::to_string(getpid())+".jsonl");
      output.file = fopen(path.c_str(),"a"); output.rank = m.rank;
      if (!output.file) throw std::runtime_error("cannot open host profile JSONL");
    }
    FILE *f = output.file;
    fprintf(f,"{\"schema\":1,\"clock\":\"steady_clock\",\"unit\":\"ns\",\"rank\":%d,\"pid\":%lld,"
      "\"step\":%lld,\"call\":%llu,\"owned\":%d,\"total\":%d,\"eflag\":%d,\"vflag\":%d,"
      "\"candidates\":%llu,\"active_edges\":%llu,\"image_groups\":%llu,\"topology_revision\":%llu,"
      "\"rebuilt\":%s,\"overflow\":%s,\"stages\":[",m.rank,static_cast<long long>(getpid()),
      static_cast<long long>(m.step),static_cast<unsigned long long>(m.call),m.owned,m.total,m.eflag,m.vflag,
      static_cast<unsigned long long>(m.candidates),static_cast<unsigned long long>(m.active_edges),
      static_cast<unsigned long long>(m.image_groups),static_cast<unsigned long long>(m.topology_revision),
      m.rebuilt ? "true" : "false",record.overflow ? "true" : "false");
    for (std::size_t i = 0; i < record.count; ++i) {
      const auto &e = record.entries[i];
      if (i) fputc(',',f);
      fprintf(f,"{\"id\":%zu,\"parent\":%d,\"label\":",i,e.parent); quoted(f,e.label);
      fprintf(f,",\"calls\":%llu,\"inclusive_ns\":%llu,\"exclusive_ns\":%llu}",
        static_cast<unsigned long long>(e.calls),static_cast<unsigned long long>(e.inclusive_ns),
        static_cast<unsigned long long>(e.inclusive_ns-e.child_ns));
    }
    fputs("]}\n",f);
    if (fflush(f) || ferror(f)) throw std::runtime_error("cannot write host profile JSONL");
    return true;
  } catch (...) {
    fprintf(stderr,"IMR-MLP native host profiling: output failed for rank %d pid %lld\n",m.rank,
      static_cast<long long>(getpid()));
    output.failed = true;
    return false;
  }
}
}
#endif
