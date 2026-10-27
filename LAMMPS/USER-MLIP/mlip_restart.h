#pragma once

#include "mlip_model_mpi.h"
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace MLIPCore {

struct RestartState {
    std::string filename;
    std::string model_text;
    // Effective pair_style arguments: filename, type mappings, then options.
    std::vector<std::string> arguments;
};

namespace RestartDetail {
constexpr std::uint64_t max_payload = std::uint64_t(1) << 30;
constexpr std::uint64_t max_arguments = 1000000;

inline std::uint64_t content_hash(const std::string &text) {
    std::uint64_t hash = 14695981039346656037ULL;
    for (unsigned char byte : text) { hash ^= byte; hash *= 1099511628211ULL; }
    return hash;
}
inline void append_integer(std::string &out, std::uint64_t value) {
    if (out.size() > max_payload-8) throw std::length_error("MTP restart payload exceeds 1 GiB");
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<char>((value >> (8*i)) & 255));
}
inline std::uint64_t take_integer(const std::string &input, std::size_t &position) {
    if (position > input.size() || input.size()-position < 8)
        throw std::runtime_error("Truncated MTP restart integer");
    std::uint64_t value = 0;
    for (int i = 0; i < 8; ++i)
        value |= std::uint64_t(static_cast<unsigned char>(input[position++])) << (8*i);
    return value;
}
inline void append_string(std::string &out, const std::string &value) {
    if (value.size() > max_payload || out.size() > max_payload-value.size() ||
        max_payload-out.size()-value.size() < 8)
        throw std::length_error("MTP restart payload exceeds 1 GiB");
    append_integer(out,value.size()); out.append(value);
}
inline std::string take_string(const std::string &input, std::size_t &position) {
    const auto size = take_integer(input,position);
    if (size > input.size()-position) throw std::runtime_error("Truncated MTP restart string");
    const auto begin = position; position += static_cast<std::size_t>(size);
    return input.substr(begin,static_cast<std::size_t>(size));
}
inline void validate(const RestartState &state) {
    if (state.filename.empty() || state.model_text.empty() || state.arguments.empty() ||
        state.arguments.front() != state.filename || state.arguments.size() > max_arguments)
        throw std::runtime_error("Invalid MTP restart model reference or settings");
}
inline std::string encode(const RestartState &state) {
    validate(state);
    std::string out;
    append_integer(out,content_hash(state.model_text));
    append_string(out,state.filename);
    append_string(out,state.model_text);
    append_integer(out,state.arguments.size());
    for (const auto &argument : state.arguments) append_string(out,argument);
    return out;
}
inline RestartState decode(const std::string &payload) {
    std::size_t position = 0;
    const auto hash = take_integer(payload,position);
    RestartState state;
    state.filename = take_string(payload,position);
    state.model_text = take_string(payload,position);
    const auto count = take_integer(payload,position);
    if (!count || count > max_arguments || count > (payload.size()-position)/8)
        throw std::runtime_error("Invalid MTP restart argument count");
    state.arguments.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) state.arguments.push_back(take_string(payload,position));
    if (position != payload.size() || hash != content_hash(state.model_text))
        throw std::runtime_error("MTP restart model checksum or record length mismatch");
    validate(state);
    return state;
}
inline void write_bytes(FILE *file, const char *data, std::size_t size) {
    if (!file || std::fwrite(data,1,size,file) != size)
        throw std::runtime_error("Cannot write MTP restart state");
}
inline std::string read_payload(FILE *file) {
    char header[16];
    if (!file || std::fread(header,1,sizeof(header),file) != sizeof(header) ||
        std::memcmp(header,"MTPST01",8))
        throw std::runtime_error("Invalid or truncated MTP restart header");
    std::string prefix(header+8,8);
    std::size_t position = 0;
    const auto size = take_integer(prefix,position);
    if (size > max_payload) throw std::length_error("MTP restart payload exceeds 1 GiB");
    std::string payload(static_cast<std::size_t>(size),'\0');
    if (std::fread(payload.data(),1,payload.size(),file) != payload.size())
        throw std::runtime_error("Truncated MTP restart payload");
    return payload;
}
} // namespace RestartDetail

// LAMMPS invokes write_restart on rank zero only. On failure the adapter must
// use error->one(), not the collective error->all(). No GPU/neighbor state is saved.
inline void write_restart_state(FILE *file, const RestartState &state) {
    const std::string payload = RestartDetail::encode(state);
    std::string header("MTPST01",8);
    RestartDetail::append_integer(header,payload.size());
    RestartDetail::write_bytes(file,header.data(),header.size());
    RestartDetail::write_bytes(file,payload.data(),payload.size());
}

// Rank zero verifies both the stored FNV-1a64 checksum and byte-for-byte equality
// with the external model prefix. The exact comparison makes hash collisions
// irrelevant. MVS payloads are excluded by read_model_text(). Callers should
// replay settings using state.model_text, avoiding a second filesystem read.
inline RestartState read_restart_state(FILE *file, MPI_Comm world) {
    int rank = 0;
    MPI_Comm_rank(world,&rank);
    std::string payload, failure;
    if (!rank) try {
        payload = RestartDetail::read_payload(file);
        const auto state = RestartDetail::decode(payload);
        if (read_model_text(state.filename.c_str()) != state.model_text)
            throw std::runtime_error("External MTP model differs from the model saved in restart");
    } catch (const std::exception &e) { failure = e.what(); }
    int error_length = static_cast<int>(failure.size());
    MPI_Bcast(&error_length,1,MPI_INT,0,world);
    if (error_length) {
        failure.resize(error_length);
        MPI_Bcast(failure.data(),error_length,MPI_CHAR,0,world);
        throw std::runtime_error(failure);
    }
    unsigned long long size = payload.size();
#ifndef MPI_STUBS
    // A serial LAMMPS build has already read the payload on its only rank.
    MPI_Bcast(&size,1,MPI_UNSIGNED_LONG_LONG,0,world);
#endif
    int local_failure = 0, any_failure = 0;
    try { payload.resize(static_cast<std::size_t>(size)); }
    catch (const std::exception &) { local_failure = 1; }
    MPI_Allreduce(&local_failure,&any_failure,1,MPI_INT,MPI_MAX,world);
    if (any_failure) throw std::runtime_error("Cannot allocate MTP restart state on every rank");
    // max_payload is below INT_MAX, so the portable record fits one broadcast.
    MPI_Bcast(payload.data(),static_cast<int>(size),MPI_CHAR,0,world);
    RestartState state;
    local_failure = 0;
    try { state = RestartDetail::decode(payload); }
    catch (const std::exception &) { local_failure = 1; }
    MPI_Allreduce(&local_failure,&any_failure,1,MPI_INT,MPI_MAX,world);
    if (any_failure) throw std::runtime_error("Cannot decode MTP restart state on every rank");
    return state;
}

} // namespace MLIPCore
