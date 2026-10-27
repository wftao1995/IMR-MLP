// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "utils/compressed_stream.h"

#ifdef USE_GZIP
#include <zlib.h>
#endif

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <fstream>
#include <exception>
#include <iostream>
#include <stdexcept>
#include <vector>

// ── Extension / magic-byte helpers ──────────────────────────────────────────

CompressionKind compression_of_extension(const std::string& path) noexcept {
    size_t dot = path.rfind('.');
    if (dot == std::string::npos || dot + 1 >= path.size()) return CompressionKind::None;
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    if (ext == "gz") return CompressionKind::Gzip;
    if (ext == "zst") return CompressionKind::Unsupported;
    return CompressionKind::None;
}

CompressionKind probe_compression(const std::string& path) noexcept {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return CompressionKind::None;
    unsigned char magic[4] = {0, 0, 0, 0};
    size_t n = fread(magic, 1, 4, f);
    fclose(f);
    if (n >= 2 && magic[0] == 0x1f && magic[1] == 0x8b) return CompressionKind::Gzip;
    if (n >= 4 && magic[0] == 0x28 && magic[1] == 0xb5 && magic[2] == 0x2f && magic[3] == 0xfd)
        return CompressionKind::Unsupported;
    return CompressionKind::None;
}

std::string strip_compression_suffix(const std::string& path) noexcept {
    size_t dot = path.rfind('.');
    if (dot == std::string::npos || dot == 0) return path;
    std::string ext = path.substr(dot + 1);
    for (char& c : ext) c = (char)std::tolower((unsigned char)c);
    if (ext == "gz" || ext == "zst") return path.substr(0, dot);
    return path;
}

// ── DecompressBuf: whole-file decompression into memory ────────────────────

namespace {

/// @brief std::streambuf over the fully decompressed file contents.
///        Mirrors MMapStreamBuf (io/gpio_tokenizer.h): the whole file is
///        resident in memory, so random access (seekg/tellg) is supported.
class DecompressBuf : public std::streambuf {
public:
    /// @brief Takes ownership of f (closed by the constructor). The file is
    ///        rewound to the beginning, so the caller may have consumed a few
    ///        magic bytes already. Throws std::runtime_error on corrupt data
    ///        or when the codec was not compiled in (defensive — open_ifile
    ///        already rejects disabled codecs before constructing this).
    DecompressBuf(FILE* f, CompressionKind kind) {
        rewind(f);
        try {
            if (kind == CompressionKind::Gzip) {
#ifdef USE_GZIP
                inflate_gzip(f);
#else
                throw std::runtime_error("gzip support not enabled (compile with -DUSE_GZIP)");
#endif
            } else {
                throw std::runtime_error("unsupported compression");
            }
        } catch (...) {
            fclose(f);
            throw;
        }
        fclose(f);
        if (data_.empty()) {
            setg(nullptr, nullptr, nullptr);
        } else {
            setg(data_.data(), data_.data(), data_.data() + data_.size());
        }
    }

    int_type underflow() override {
        if (gptr() == egptr()) return traits_type::eof();
        return traits_type::to_int_type(*gptr());
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode mode = std::ios_base::in) override {
        if (mode != std::ios_base::in) return pos_type(off_type(-1));
        off_type new_pos = 0;
        if (dir == std::ios_base::beg) {
            new_pos = off;
        } else if (dir == std::ios_base::cur) {
            new_pos = (gptr() ? gptr() - eback() : 0) + off;
        } else if (dir == std::ios_base::end) {
            new_pos = (off_type)data_.size() + off;
        } else {
            return pos_type(off_type(-1));
        }
        if (new_pos < 0 || (size_t)new_pos > data_.size()) return pos_type(off_type(-1));
        setg(eback(), eback() + new_pos, egptr());
        return pos_type(new_pos);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode mode = std::ios_base::in) override {
        return seekoff(off_type(pos), std::ios_base::beg, mode);
    }

private:
#ifdef USE_GZIP
    void inflate_gzip(FILE* f) {
        z_stream zs{};
        if (inflateInit2(&zs, 15 + 16) != Z_OK) throw std::runtime_error("gzip: inflateInit2 failed");
        std::array<char, 65536> in, out;
        bool finished = false;
        for (;;) {
            if (zs.avail_in == 0) {
                size_t n = fread(in.data(), 1, in.size(), f);
                if (ferror(f)) {
                    inflateEnd(&zs);
                    throw std::runtime_error("gzip: read failed");
                }
                if (n == 0) break;
                zs.next_in = (Bytef*)in.data();
                zs.avail_in = (uInt)n;
            }
            zs.next_out = (Bytef*)out.data();
            zs.avail_out = (uInt)out.size();
            finished = false;
            int ret = inflate(&zs, Z_NO_FLUSH);
            data_.insert(data_.end(), out.data(), out.data() + (out.size() - zs.avail_out));
            if (ret == Z_STREAM_END) {
                finished = true;
                // concatenated gzip member (e.g. from append-mode writes)
                // Refill even when a member ends exactly at the input buffer
                // boundary; only fread reaching EOF proves this was the last.
                if (inflateReset2(&zs, 15 + 16) != Z_OK) {
                    inflateEnd(&zs);
                    throw std::runtime_error("gzip: member reset failed");
                }
                continue;
            }
            if (ret != Z_OK && ret != Z_BUF_ERROR) {
                inflateEnd(&zs);
                throw std::runtime_error("gzip: decompression error");
            }
            // Z_OK implies either input exhausted (refill next iteration)
            // or output full (drain again with the same input).
        }
        inflateEnd(&zs);
        if (!finished) throw std::runtime_error("gzip: truncated file");
    }
#endif  // USE_GZIP


    std::vector<char> data_;
};

// ── SequentialDecompressBuf: bounded, on-demand decompression ────────────────

struct InputFileCloser {
    void operator()(FILE* file) const noexcept { fclose(file); }
};
using InputFile = std::unique_ptr<FILE, InputFileCloser>;

class SequentialDecompressBuf : public std::streambuf {
public:
    SequentialDecompressBuf(InputFile file, CompressionKind kind,
                            const char* prefix, size_t prefix_size)
        : file_(std::move(file)), kind_(kind), input_size_(prefix_size) {
        std::copy_n(prefix, prefix_size, input_.data());
        if (kind_ == CompressionKind::Gzip) {
#ifdef USE_GZIP
            gzip_ = std::make_unique<GzipDecoder>();
#else
            throw std::runtime_error("gzip support not enabled (compile with -DUSE_GZIP)");
#endif
        } else {
#ifdef USE_ZSTD
            zstd_.reset(ZSTD_createDStream());
            if (!zstd_) throw std::runtime_error("zstd: createDStream failed");
            check_zstd(ZSTD_initDStream(zstd_.get()));
#else
            throw std::runtime_error("zstd support not enabled (compile with -DUSE_ZSTD)");
#endif
        }
    }

protected:
    int_type underflow() override {
        if (gptr() != egptr()) return traits_type::to_int_type(*gptr());
        if (failure_) std::rethrow_exception(failure_);
        if (finished_) return traits_type::eof();
        try {
            while (!finished_) {
                refill();
                if (input_eof_ && input_pos_ == input_size_ && member_finished_) {
                    finished_ = true;
                    return traits_type::eof();
                }
                const size_t produced = decode();
                if (produced != 0) {
                    setg(output_.data(), output_.data(), output_.data() + produced);
                    return traits_type::to_int_type(*gptr());
                }
            }
            return traits_type::eof();
        } catch (...) {
            // clear() on the istream cannot turn a corrupt member into EOF.
            failure_ = std::current_exception();
            throw;
        }
    }

private:
    void refill() {
        if (input_pos_ != input_size_ || input_eof_) return;
        input_size_ = fread(input_.data(), 1, input_.size(), file_.get());
        input_pos_ = 0;
        if (ferror(file_.get())) throw std::runtime_error("compressed input: read failed");
        input_eof_ = feof(file_.get());
    }

    size_t decode() {
        const size_t previous_pos = input_pos_;
        size_t produced = 0;
        if (kind_ == CompressionKind::Gzip) {
#ifdef USE_GZIP
            auto& stream = gzip_->stream;
            if (member_finished_ && inflateReset2(&stream, 15 + 16) != Z_OK)
                throw std::runtime_error("gzip: member reset failed");
            member_finished_ = false;
            stream.next_in = reinterpret_cast<Bytef*>(input_.data() + input_pos_);
            stream.avail_in = static_cast<uInt>(input_size_ - input_pos_);
            stream.next_out = reinterpret_cast<Bytef*>(output_.data());
            stream.avail_out = static_cast<uInt>(output_.size());
            const int result = inflate(&stream, Z_NO_FLUSH);
            input_pos_ = input_size_ - stream.avail_in;
            produced = output_.size() - stream.avail_out;
            if (result == Z_STREAM_END) member_finished_ = true;
            else if (result != Z_OK && result != Z_BUF_ERROR)
                throw std::runtime_error("gzip: decompression error");
#endif
        } else {
#ifdef USE_ZSTD
            if (member_finished_) check_zstd(ZSTD_initDStream(zstd_.get()));
            member_finished_ = false;
            ZSTD_inBuffer input{input_.data(), input_size_, input_pos_};
            ZSTD_outBuffer output{output_.data(), output_.size(), 0};
            const size_t result = ZSTD_decompressStream(zstd_.get(), &output, &input);
            check_zstd(result);
            input_pos_ = input.pos;
            produced = output.pos;
            member_finished_ = result == 0;
#endif
        }
        // A decoder may retain output after consuming its final input byte.
        // Drain it even at physical EOF; only no progress before a member end
        // proves truncation. This matters for high compression-ratio frames.
        if (!member_finished_ && produced == 0 && input_pos_ == previous_pos) {
            const char* codec = kind_ == CompressionKind::Gzip ? "gzip" : "zstd";
            if (input_eof_ && input_pos_ == input_size_)
                throw std::runtime_error(std::string(codec) + ": truncated file");
            throw std::runtime_error(std::string(codec) + ": decompression made no progress");
        }
        return produced;
    }

#ifdef USE_GZIP
    struct GzipDecoder {
        z_stream stream{};
        GzipDecoder() {
            if (inflateInit2(&stream, 15 + 16) != Z_OK)
                throw std::runtime_error("gzip: inflateInit2 failed");
        }
        ~GzipDecoder() { inflateEnd(&stream); }
    };
    std::unique_ptr<GzipDecoder> gzip_;
#endif
#ifdef USE_ZSTD
    static void check_zstd(size_t result) {
        if (ZSTD_isError(result))
            throw std::runtime_error(std::string("zstd: ") + ZSTD_getErrorName(result));
    }
    std::unique_ptr<ZSTD_DStream, decltype(&ZSTD_freeDStream)> zstd_{nullptr, &ZSTD_freeDStream};
#endif
    InputFile file_;
    CompressionKind kind_;
    std::array<char, 65536> input_, output_;
    size_t input_pos_ = 0, input_size_ = 0;
    bool input_eof_ = false, member_finished_ = false, finished_ = false;
    std::exception_ptr failure_;
};

// ── CompressBuf: streaming compression to a FILE* ───────────────────────────

class CompressBuf : public std::streambuf {
public:
    CompressBuf(FILE* f, CompressionKind kind, int level) : f_(f), kind_(kind), obuf_(65536) {
        setp(pbuf_, pbuf_ + sizeof(pbuf_));
        // A codec that is not compiled in can never be constructed through
        // open_ofile (it rejects the extension); this is a defensive failure.
        if (kind_ == CompressionKind::Gzip) {
#ifdef USE_GZIP
            if (deflateInit2(&zs_, level == 0 ? Z_DEFAULT_COMPRESSION : level, Z_DEFLATED, 15 + 16, 8,
                             Z_DEFAULT_STRATEGY) != Z_OK)
                ok_ = false;
#else
            ok_ = false;
#endif
        } else {
            ok_ = false;
        }
    }

    ~CompressBuf() override { finish(); }

    /// @brief Flushes remaining input, writes the compression trailer and
    ///        closes the file. Returns false on failure. Idempotent.
    bool finish() noexcept {
        if (finished_) return ok_;
        finished_ = true;
        if (ok_ && f_) {
            if (kind_ == CompressionKind::Gzip) {
#ifdef USE_GZIP
                if (flush_put_area(Z_NO_FLUSH)) ok_ = compress_chunk(nullptr, 0, Z_FINISH);
#endif
            }
        }
        if (f_) {
            if (fclose(f_) != 0) ok_ = false;
            f_ = nullptr;
        }
#ifdef USE_GZIP
        if (kind_ == CompressionKind::Gzip) deflateEnd(&zs_);
#endif
        return ok_;
    }

protected:
    int_type overflow(int_type c) override {
        if (!flush_put_area(put_flush())) return traits_type::eof();
        if (!traits_type::eq_int_type(c, traits_type::eof())) {
            *pptr() = traits_type::to_char_type(c);
            pbump(1);
        }
        return c;
    }

    std::streamsize xsputn(const char* s, std::streamsize n) override {
        std::streamsize written = 0;
        while (written < n) {
            std::streamsize avail = epptr() - pptr();
            if (avail == 0) {
                if (!flush_put_area(put_flush())) return written;
                avail = epptr() - pptr();
                if (avail == 0) return written;
            }
            std::streamsize chunk = std::min(avail, n - written);
            std::memcpy(pptr(), s + written, (size_t)chunk);
            pbump(chunk);
            written += chunk;
        }
        return written;
    }

    int sync() override {
        return flush_put_area(sync_flush()) ? 0 : -1;
    }

private:
    /// @brief Gzip flush directive for the put area.
    int put_flush() const noexcept {
#ifdef USE_GZIP
        if (kind_ == CompressionKind::Gzip) return Z_NO_FLUSH;
#endif
        return 0;
    }

    int sync_flush() const noexcept {
#ifdef USE_GZIP
        if (kind_ == CompressionKind::Gzip) return Z_SYNC_FLUSH;
#endif
        return 0;
    }

    /// @brief Compresses the buffered put area, then rewinds it.
    ///        flush is a zlib flush code.
    bool flush_put_area(int flush) noexcept {
        std::streamsize n = pptr() - pbase();
        if (n <= 0) return ok_;
        bool r = true;
#ifdef USE_GZIP
        if (kind_ == CompressionKind::Gzip) r = compress_chunk(pbase(), (size_t)n, flush);
#endif
        setp(pbuf_, pbuf_ + sizeof(pbuf_));
        return r;
    }

#ifdef USE_GZIP
    bool compress_chunk(const char* data, size_t n, int flush) noexcept {
        if (!ok_) return false;
        size_t off = 0;
        for (;;) {
            bool no_progress = false;
            if (off < n) {
                size_t chunk = std::min(n - off, (size_t)0x7FFFFFFF);  // stay within uInt
                zs_.next_in = (Bytef*)(data + off);
                zs_.avail_in = (uInt)chunk;
                off += chunk;
            } else {
                zs_.next_in = nullptr;
                zs_.avail_in = 0;
            }
            for (;;) {
                zs_.next_out = (Bytef*)obuf_.data();
                zs_.avail_out = (uInt)obuf_.size();
                int ret = deflate(&zs_, flush);
                size_t produced = obuf_.size() - zs_.avail_out;
                if (produced && fwrite(obuf_.data(), 1, produced, f_) != produced) {
                    ok_ = false;
                    return false;
                }
                if (ret == Z_STREAM_END) return true;
                if (ret == Z_BUF_ERROR) {
                    no_progress = true;
                    break;
                }
                if (ret != Z_OK) {
                    ok_ = false;
                    return false;
                }
                if (zs_.avail_out != 0) break;  // drained: input consumed, no pending output
            }
            if (no_progress) return ok_;        // sync-flush finished / nothing left to drain
            if (off == n && flush == Z_NO_FLUSH) return ok_;
        }
    }
#endif  // USE_GZIP


    FILE* f_;
    CompressionKind kind_;
    bool finished_ = false;
    bool ok_ = true;
    std::vector<char> obuf_;
    char pbuf_[8192];
#ifdef USE_GZIP
    z_stream zs_{};
#endif
};

}  // namespace

// ── CompressedOStream ───────────────────────────────────────────────────────

struct CompressedOStream::Impl {
    std::unique_ptr<CompressBuf> buf;
};

CompressedOStream::CompressedOStream(FILE* f, CompressionKind kind, int level)
    : std::ostream(nullptr), impl_(std::make_unique<Impl>()) {
    impl_->buf = std::make_unique<CompressBuf>(f, kind, level);
    rdbuf(impl_->buf.get());
}

CompressedOStream::~CompressedOStream() = default;

bool CompressedOStream::finish() { return impl_->buf->finish(); }

// ── Factories ───────────────────────────────────────────────────────────────

std::unique_ptr<std::istream> open_ifile(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return nullptr;
    unsigned char magic[4] = {0, 0, 0, 0};
    size_t n = fread(magic, 1, 4, f);
    CompressionKind kind = CompressionKind::None;
    if (n >= 2 && magic[0] == 0x1f && magic[1] == 0x8b) {
        kind = CompressionKind::Gzip;
    } else if (n >= 4 && magic[0] == 0x28 && magic[1] == 0xb5 && magic[2] == 0x2f && magic[3] == 0xfd) {
        kind = CompressionKind::Unsupported;
    }
    if (kind == CompressionKind::None) kind = compression_of_extension(path);
    if (kind == CompressionKind::None) {
        fclose(f);
        return std::make_unique<std::ifstream>(path);
    }
    // Codec recognition (magic bytes / extension) is always on so that a
    // compressed file is never silently parsed as plain text; gzip decoding
    // requires the compile-time switch and unsupported codecs are rejected.
#ifndef USE_GZIP
    if (kind == CompressionKind::Gzip) {
        std::cerr << "Error: gzip support not enabled (compile with -DUSE_GZIP): " << path << std::endl;
        fclose(f);
        return nullptr;
    }
#endif
    if (kind == CompressionKind::Unsupported || compression_of_extension(path) == CompressionKind::Unsupported) {
        std::cerr << "Error: Zstandard compression is not supported: " << path << std::endl;
        fclose(f);
        return nullptr;
    }
    rewind(f);
    return std::make_unique<OwnedIStream>(std::make_unique<DecompressBuf>(f, kind));
}

std::unique_ptr<std::istream> open_seq_ifile(const std::string& path) {
    InputFile file(fopen(path.c_str(), "rb"));
    if (!file) return nullptr;
    std::array<char, 4> prefix{};
    const size_t count = fread(prefix.data(), 1, prefix.size(), file.get());
    if (ferror(file.get())) throw std::runtime_error("input: read failed: " + path);
    CompressionKind kind = CompressionKind::None;
    const auto* magic = reinterpret_cast<const unsigned char*>(prefix.data());
    if (count >= 2 && magic[0] == 0x1f && magic[1] == 0x8b)
        kind = CompressionKind::Gzip;
    else if (count >= 4 &&
             ((magic[0] == 0x28 && magic[1] == 0xb5 && magic[2] == 0x2f && magic[3] == 0xfd) ||
              ((magic[0] & 0xf0) == 0x50 && magic[1] == 0x2a && magic[2] == 0x4d && magic[3] == 0x18)))
        kind = CompressionKind::Unsupported;
    if (kind == CompressionKind::None) kind = compression_of_extension(path);
    if (kind == CompressionKind::Unsupported || compression_of_extension(path) == CompressionKind::Unsupported) {
        std::cerr << "Error: Zstandard compression is not supported: " << path << std::endl;
        return nullptr;
    }
    if (kind == CompressionKind::None) {
        file.reset();
        auto stream = std::make_unique<std::ifstream>(path, std::ios::binary);
        if (!*stream) return nullptr;
        stream->exceptions(std::ios::badbit);
        return stream;
    }
#ifndef USE_GZIP
    if (kind == CompressionKind::Gzip) {
        std::cerr << "Error: gzip support not enabled (compile with -DUSE_GZIP): " << path << std::endl;
        return nullptr;
    }
#endif
    // Keep the probed prefix in the input buffer; compressed input never seeks.
    auto buffer = std::make_unique<SequentialDecompressBuf>(std::move(file), kind, prefix.data(), count);
    auto stream = std::make_unique<OwnedIStream>(std::move(buffer));
    stream->exceptions(std::ios::badbit);
    return stream;
}

std::unique_ptr<std::ostream> open_ofile(const std::string& path, bool append, bool binary) {
    CompressionKind kind = compression_of_extension(path);
    if (kind == CompressionKind::None) {
        std::ios::openmode mode = std::ios::out;
        if (append) mode |= std::ios::app;
        if (binary) mode |= std::ios::binary;
        auto stream = std::make_unique<std::ofstream>(path, mode);
        if (!*stream) return nullptr;
        return stream;
    }
    // Reject unsupported or disabled compression before touching the output.
#ifndef USE_GZIP
    if (kind == CompressionKind::Gzip) {
        std::cerr << "Error: gzip support not enabled (compile with -DUSE_GZIP): " << path << std::endl;
        return nullptr;
    }
#endif
    if (kind == CompressionKind::Unsupported) {
        std::cerr << "Error: Zstandard compression is not supported: " << path << std::endl;
        return nullptr;
    }
    FILE* f = fopen(path.c_str(), append ? "ab" : "wb");
    if (!f) return nullptr;
    return std::make_unique<CompressedOStream>(f, kind, 0);
}

bool close_ofile(std::unique_ptr<std::ostream>& os) {
    if (!os) return false;
    bool ok = false;
    if (auto* cos = dynamic_cast<CompressedOStream*>(os.get())) {
        ok = cos->finish() && !cos->fail();
    } else if (auto* ofs = dynamic_cast<std::ofstream*>(os.get())) {
        ofs->close();
        ok = !ofs->fail();
    } else {
        os->flush();
        ok = !os->fail();
    }
    os.reset();
    return ok;
}
