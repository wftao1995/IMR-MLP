// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file gpio_tokenizer.h
 * @brief Memory-mapped stream buffer and fast line-based tokenizers for GPIO.
 *
 * Extracted from gpio.h to reduce recompilation cascade — only io/ .cpp
 * files that actually parse input need to include this.
 *
 * @see gpio.h
 */

#include <sys/stat.h>
#include <mio/mmap.hpp>
#include <fast_float/fast_float.h>
#include <charconv>
#include <string>
#include <string_view>
#include <istream>

/// @brief Memory-mapped file stream buffer for fast sequential reading.
class MMapStreamBuf : public std::streambuf {
public:
    explicit MMapStreamBuf(const char* filename) : data_(nullptr), size_(0) {
        int fd = open(filename, O_RDONLY);
        if (fd == -1) {
            throw std::runtime_error("open failed: " + std::string(strerror(errno)));
        }

        struct stat sb;
        if (fstat(fd, &sb) == -1) {
            int err = errno;
            close(fd);
            throw std::runtime_error("fstat failed: " + std::string(strerror(err)));
        }

        size_ = sb.st_size;
        if (size_ == 0) {
            close(fd);
            setg(nullptr, nullptr, nullptr);
            return;
        }

        data_ = static_cast<char*>(mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0));
        close(fd);

        if (data_ == MAP_FAILED) {
            int err = errno;
            throw std::runtime_error("mmap failed: " + std::string(strerror(err)));
        }

        setg(data_, data_, data_ + size_);
    }

    ~MMapStreamBuf() override {
        if (data_ && data_ != MAP_FAILED) {
            munmap(data_, size_);
            data_ = nullptr;
        }
    }

    int_type underflow() override {
        if (gptr() == egptr()) {
            return traits_type::eof();
        }
        return traits_type::to_int_type(*gptr());
    }

    pos_type seekoff(off_type off, std::ios_base::seekdir dir,
                     std::ios_base::openmode mode = std::ios_base::in) override {
        if (mode != std::ios_base::in) {
            return pos_type(off_type(-1));
        }

        off_type new_pos = 0;
        if (dir == std::ios_base::beg) {
            new_pos = off;
        } else if (dir == std::ios_base::cur) {
            new_pos = gptr() - eback() + off;
        } else if (dir == std::ios_base::end) {
            new_pos = size_ + off;
        } else {
            return pos_type(off_type(-1));
        }

        if (new_pos < 0 || static_cast<size_t>(new_pos) > size_) {
            return pos_type(off_type(-1));
        }

        setg(eback(), eback() + new_pos, egptr());
        return pos_type(new_pos);
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode mode = std::ios_base::in) override {
        return seekoff(off_type(pos), std::ios_base::beg, mode);
    }

private:
    char* data_;
    size_t size_;
};

struct WS_Default {
    static inline bool is(char c) noexcept {
        return c == ' ' || c == '\n' || c == '\t' || c == '\r';
    }
};

struct WS_LmpYaml {
    static inline bool is(char c) noexcept {
        return c == ' ' || c == ',' || c == '[' || c == ']' ||
               c == '\n' || c == '\t' || c == '\r';
    }
};

/// @brief Fast line-based tokenizer with configurable whitespace detection.
/// @tparam WS Whitespace predicate type (e.g., WS_Default, WS_LmpYaml)
template <typename WS = WS_Default>
struct GPIO_LineTokenizer {
    std::string_view line;
    size_t pos = 0;

    explicit GPIO_LineTokenizer(std::string_view l) : line(l) {}

    inline void skip_ws() noexcept {
        while (pos < line.size() && WS::is(line[pos])) {
            ++pos;
        }
    }

    inline bool eof() const noexcept {
        return pos >= line.size();
    }

    std::string_view next_token() noexcept {
        skip_ws();
        if (eof()) return {};

        size_t start = pos;
        while (pos < line.size() && !WS::is(line[pos])) {
            ++pos;
        }
        return line.substr(start, pos - start);
    }

    template<class T>
    bool next_numeric(T &val) noexcept {
        skip_ws();
        if (eof()) return false;

        auto res = fast_float::from_chars(
            line.data() + pos,
            line.data() + line.size(),
            val
        );

        if (res.ec != std::errc{} ||
            (res.ptr != line.data() + line.size() && !WS::is(*res.ptr))) return false;

        pos = static_cast<size_t>(res.ptr - line.data());
        return true;
    }

    template<class T>
    inline void next_numeric_unsafe(T& val) noexcept {
        skip_ws();
        auto res = fast_float::from_chars(
            line.data() + pos,
            line.data() + line.size(),
            val
        );
        pos = static_cast<size_t>(res.ptr - line.data());
    }

    template<class T>
    bool until_next_numeric(T& val) noexcept {
        skip_ws();
        if (eof()) return false;
        auto is_num_start = [](unsigned char c) noexcept {
            return (c >= '0' && c <= '9') || c == '+' || c == '-' || c == '.'
                   || c == 'n' || c == 'N' || c == 'i' || c == 'I';
        };
        const char* begin = line.data();
        const char* end   = begin + line.size();
        size_t i = pos;
        for (; i < line.size(); ++i) {
            unsigned char c = static_cast<unsigned char>(line[i]);
            if (!is_num_start(c)) continue;
            auto res = fast_float::from_chars(begin + i, end, val);
            if (res.ec == std::errc{}) {
                pos = static_cast<size_t>(res.ptr - begin);
                return true;
            }
        }
        pos = line.size();
        return false;
    }
};

/// @brief Buffered stream tokenizer for large sequential reads.
/// @tparam WS Whitespace predicate type
/// @tparam BufSize Internal buffer size in bytes (default 64KB)
template <typename WS = WS_Default, size_t BufSize = 1 << 16>
class IStreamTokenizer {
public:
    explicit IStreamTokenizer(std::istream& is)
        : is_(is) {
        refill();
    }

    bool eof() const noexcept {
        return cur_ >= end_ && is_.eof();
    }

    bool next_token(std::string_view& tok) {
        skip_ws();
        if (!ensure_data()) return false;

        scratch_.clear();
        while (true) {
            const char* start = cur_;
            while (cur_ < end_ && !WS::is(*cur_)) ++cur_;

            if (cur_ < end_) {
                if (scratch_.empty())
                    tok = {start, static_cast<size_t>(cur_ - start)};
                else {
                    scratch_.append(start, cur_);
                    tok = scratch_;
                }
                return true;
            }

            scratch_.append(start, cur_);
            if (!refill()) {
                tok = scratch_;
                return !tok.empty();
            }
        }
    }

    template<class T>
    bool next_numeric(T& val) {
        std::string_view token;
        if (!next_token(token)) return false;
        const auto res = fast_float::from_chars(token.data(), token.data() + token.size(), val);
        return res.ec == std::errc{} && res.ptr == token.data() + token.size();
    }

    template<class T>
    void next_numeric_unsafe(T& val) {
        (void)next_numeric(val);
    }

private:
    std::istream& is_;
    char buffer_[BufSize];
    const char* cur_ = nullptr;
    const char* end_ = nullptr;
    std::string scratch_;

    inline void skip_ws() {
        while (true) {
            while (cur_ < end_ && WS::is(*cur_)) {
                ++cur_;
            }
            if (cur_ < end_ || !refill()) break;
        }
    }

    inline bool ensure_data() {
        if (cur_ < end_) return true;
        return refill();
    }

    bool refill() {
        if (!is_) return false;

        is_.read(buffer_, BufSize);
        std::streamsize n = is_.gcount();
        if (n <= 0) return false;

        cur_ = buffer_;
        end_ = buffer_ + n;
        return true;
    }
};
