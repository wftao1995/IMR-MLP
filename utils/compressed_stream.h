// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file compressed_stream.h
 * @brief Transparent gzip compressed file I/O (optional feature).
 *
 * Provides factory functions that open a file and return a std::istream /
 * std::ostream with transparent compression:
 *   - Reading: the codec is detected from the file's magic bytes first,
 *     with the file extension (.gz) as a fallback.
 *   - Writing: the codec is chosen from the output file extension.
 *
 * open_ifile decompresses the whole compressed file into memory on open,
 * supporting random access (seekg/tellg). open_seq_ifile instead decodes on
 * demand with fixed I/O buffers and does not seek within compressed input.
 *
 * Gzip is enabled by default in Makefile.inc (-DUSE_GZIP, links -lz).
 * Disable it with USE_GZIP=0. Disabled gzip and unsupported Zstandard files
 * are still recognized (magic bytes / extension) but open_ifile/open_ofile
 * report an error instead of silently treating them as plain text. All API
 * declarations below are unconditional so callers compile unchanged.
 *
 * Error convention: open_ifile/open_ofile return nullptr when the file cannot
 * be opened or its codec is not compiled in; corrupt/truncated compressed
 * data throws std::runtime_error.
 */

#include <cstdio>
#include <istream>
#include <memory>
#include <ostream>
#include <string>

#ifdef USE_GZIP
#include <zlib.h>
#endif

/// @brief Compression detection result; Unsupported is recognized but rejected.
enum class CompressionKind { None, Gzip, Unsupported };

/// @brief Codec implied by the file extension (.gz / .zst), None if none.
CompressionKind compression_of_extension(const std::string& path) noexcept;

/// @brief Codec detected from the file's magic bytes (gzip 1f 8b, zstd 28 b5 2f fd).
///        Returns None if the file is unreadable or not compressed.
CompressionKind probe_compression(const std::string& path) noexcept;

/// @brief Removes a trailing compression extension (.gz / .zst) from path.
///        Returns path unchanged if it has no compression extension.
std::string strip_compression_suffix(const std::string& path) noexcept;

/// @brief Opens a file for reading with transparent decompression.
///        Returns nullptr if the file cannot be opened; throws
///        std::runtime_error if the compressed content is corrupt/truncated.
std::unique_ptr<std::istream> open_ifile(const std::string& path);

/// @brief Opens a file for sequential reading with on-demand decompression.
///        Plain files use std::ifstream; gzip use two fixed 64 KiB I/O
///        buffers plus the codec's bounded history window. Compressed streams
///        do not support seekg/tellg; concatenated members are supported.
///        Returns nullptr for open failures or disabled codecs. Decode/read
///        errors set badbit and throw (badbit exceptions are enabled by default);
///        only a complete final member followed by physical EOF is clean EOF.
///        Callers disabling exceptions must check badbit after reading. Read to
///        EOF to validate the complete file; destruction does not drain it.
std::unique_ptr<std::istream> open_seq_ifile(const std::string& path);

/// @brief Opens a file for writing; compressed iff path ends in .gz.
///        Returns nullptr if the file cannot be opened.
///        For compressed output, append=true appends a new compressed member
///        (multi-member gzip, readable back).
std::unique_ptr<std::ostream> open_ofile(const std::string& path, bool append = false, bool binary = false);

/// @brief Closes a stream opened by open_ofile, finalizing the compressed
///        trailer if needed. Returns false if closing/writing failed.
///        The stream is released (set to nullptr) on success.
bool close_ofile(std::unique_ptr<std::ostream>& os);

/// @brief std::istream that owns its streambuf (std::istream does not).
class OwnedIStream : public std::istream {
public:
    OwnedIStream(std::unique_ptr<std::streambuf> sb) : std::istream(nullptr), sb_(std::move(sb)) {
        rdbuf(sb_.get());
    }
    OwnedIStream(const OwnedIStream&) = delete;
    OwnedIStream& operator=(const OwnedIStream&) = delete;

private:
    std::unique_ptr<std::streambuf> sb_;
};

/// @brief std::ostream that owns a compressing streambuf; finalizes the
///        compressed trailer on destruction.
class CompressedOStream : public std::ostream {
public:
    CompressedOStream(FILE* f, CompressionKind kind, int level);
    ~CompressedOStream() override;
    CompressedOStream(const CompressedOStream&) = delete;
    CompressedOStream& operator=(const CompressedOStream&) = delete;

    /// @brief Flushes remaining data and writes the compression trailer.
    ///        Returns false on failure. Safe to call once.
    bool finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
