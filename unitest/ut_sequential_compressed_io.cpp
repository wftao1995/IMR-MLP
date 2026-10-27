#include <catch2/catch_all.hpp>
#include "utils/compressed_stream.h"
#include <algorithm>
#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <random>
#include <vector>

namespace {
struct Directory {
    std::filesystem::path path;
    Directory() {
        char pattern[] = "/tmp/imrmlp_seq_codec_XXXXXX";
        const char* value = mkdtemp(pattern);
        if (!value) throw std::runtime_error("Cannot create sequential codec test directory");
        path = value;
    }
    ~Directory() { std::filesystem::remove_all(path); }
    std::string file(const std::string& name) const { return (path / name).string(); }
};

std::vector<std::string> compiled_codecs() {
    std::vector<std::string> result;
#ifdef USE_GZIP
    result.push_back(".gz");
#endif
#ifdef USE_ZSTD
    result.push_back(".zst");
#endif
    return result;
}

std::string raw_read(const std::string& path) {
    std::ifstream stream(path, std::ios::binary);
    REQUIRE(stream.is_open());
    return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

void raw_write(const std::string& path, const std::string& bytes) {
    std::ofstream stream(path, std::ios::binary);
    REQUIRE(stream.is_open());
    stream.write(bytes.data(), bytes.size());
    REQUIRE(stream.good());
}

std::string member(const std::string& path, const std::string& text) {
    auto stream = open_ofile(path);
    REQUIRE(stream != nullptr);
    stream->write(text.data(), text.size());
    REQUIRE(close_ofile(stream));
    return raw_read(path);
}

std::string read_chunks(std::istream& stream) {
    std::array<char, 137> buffer;
    std::string output;
    while (stream.read(buffer.data(), buffer.size()) || stream.gcount() != 0)
        output.append(buffer.data(), static_cast<size_t>(stream.gcount()));
    return output;
}

void drain(std::istream& stream) {
    std::array<char, 32768> buffer;
    while (stream.read(buffer.data(), buffer.size())) {}
}
}

TEST_CASE("Sequential plain input uses filebuf and handles missing and empty files", "[seq_compressed][plain]") {
    Directory directory;
    REQUIRE(open_seq_ifile(directory.file("missing")) == nullptr);
    raw_write(directory.file("empty"), "");
    auto empty = open_seq_ifile(directory.file("empty"));
    REQUIRE(empty != nullptr);
    REQUIRE(dynamic_cast<std::ifstream*>(empty.get()) != nullptr);
    REQUIRE(empty->get() == std::char_traits<char>::eof());
    REQUIRE(empty->eof());
    REQUIRE_FALSE(empty->bad());

    const std::string content("plain\0data\n", 11);
    raw_write(directory.file("plain"), content);
    auto input = open_seq_ifile(directory.file("plain"));
    REQUIRE(input != nullptr);
    REQUIRE(read_chunks(*input) == content);
    REQUIRE(input->eof());
    REQUIRE_FALSE(input->bad());
}

TEST_CASE("Sequential codecs preserve chunked binary data and magic detection", "[seq_compressed]") {
    Directory directory;
    std::mt19937 generator(789);
    std::string text(3 * 65536 + 29, '\0');
    for (char& value : text) value = static_cast<char>(generator() & 255);
    for (const auto& extension : compiled_codecs()) {
        INFO(extension);
        const auto bytes = member(directory.file("member" + extension), text);
        for (const std::string& name : {"same" + extension, std::string("magic.dat")}) {
            raw_write(directory.file(name), bytes);
            auto input = open_seq_ifile(directory.file(name));
            REQUIRE(input != nullptr);
            REQUIRE((input->exceptions() & std::ios::badbit) != 0);
            REQUIRE(read_chunks(*input) == text);
            REQUIRE(input->eof());
            REQUIRE_FALSE(input->bad());
        }
    }
}

TEST_CASE("Sequential codecs concatenate empty and nonempty members", "[seq_compressed][members]") {
    Directory directory;
    for (const auto& extension : compiled_codecs()) {
        INFO(extension);
        const auto path = directory.file("concat" + extension);
        for (const auto& text : {std::string{}, std::string("first\n"), std::string{}, std::string("last\n"), std::string{}}) {
            auto output = open_ofile(path, true);
            REQUIRE(output != nullptr);
            *output << text;
            REQUIRE(close_ofile(output));
        }
        auto input = open_seq_ifile(path);
        REQUIRE(input != nullptr);
        REQUIRE(read_chunks(*input) == "first\nlast\n");
        REQUIRE(input->eof());
        REQUIRE_FALSE(input->bad());
        const auto empty_path = directory.file("empty" + extension);
        member(empty_path, "");
        input = open_seq_ifile(empty_path);
        REQUIRE(input->get() == std::char_traits<char>::eof());
        REQUIRE(input->eof());
        REQUIRE_FALSE(input->bad());
    }
}

TEST_CASE("Sequential codecs drain large expansions without retaining output", "[seq_compressed][large]") {
    Directory directory;
    const std::string block(65536, 'a');
    constexpr size_t blocks = 384; // 24 MiB logical content, fixed caller buffer.
    for (const auto& extension : compiled_codecs()) {
        INFO(extension);
        const auto path = directory.file("large" + extension);
        auto output = open_ofile(path);
        REQUIRE(output != nullptr);
        for (size_t i = 0; i < blocks; ++i) output->write(block.data(), block.size());
        REQUIRE(close_ofile(output));
        REQUIRE(std::filesystem::file_size(path) < 65536);
        auto input = open_seq_ifile(path);
        REQUIRE(input != nullptr);
        std::array<char, 32768> buffer;
        size_t total = 0;
        bool correct = true;
        while (input->read(buffer.data(), buffer.size()) || input->gcount() != 0) {
            const size_t count = static_cast<size_t>(input->gcount());
            total += count;
            correct = correct && std::all_of(buffer.begin(), buffer.begin() + count, [](char c) { return c == 'a'; });
        }
        REQUIRE(total == block.size() * blocks);
        REQUIRE(correct);
        REQUIRE(input->eof());
        REQUIRE_FALSE(input->bad());
    }
}

TEST_CASE("Sequential decode is lazy and reports corrupt trailers instead of clean EOF", "[seq_compressed][errors]") {
    Directory directory;
    for (const auto& extension : compiled_codecs()) {
        INFO(extension);
        const auto path = directory.file("bad" + extension);
        const std::string first(131072, 'x');
        const auto valid = member(path, first);
        auto damaged = member(directory.file("tail" + extension), "second member\n");
        damaged.resize(damaged.size() - 1);
        raw_write(path, valid + damaged);
        std::unique_ptr<std::istream> input;
        REQUIRE_NOTHROW(input = open_seq_ifile(path));
        REQUIRE(input != nullptr);
        REQUIRE(input->get() == 'x'); // Opening did not eagerly read the corrupt tail.
        REQUIRE_THROWS_AS(drain(*input), std::runtime_error);
        REQUIRE(input->bad());
        REQUIRE_FALSE(input->eof());
        input->clear();
        REQUIRE_THROWS_AS(input->get(), std::runtime_error);
        REQUIRE(input->bad());
    }
}

TEST_CASE("Sequential malformed and truncated members retain badbit with exceptions disabled", "[seq_compressed][errors]") {
    Directory directory;
    for (const auto& extension : compiled_codecs()) {
        INFO(extension);
        const auto path = directory.file("bad" + extension);
        const auto valid = member(path, "some text\n");
        auto corrupt = valid;
        corrupt.back() ^= 1; // gzip size trailer or zstd checksum
        const std::vector<std::string> bad_inputs{
            "", "uncompressed data", valid.substr(0, 1), valid.substr(0, 4),
            valid.substr(0, valid.size() / 2), valid.substr(0, valid.size() - 1),
            corrupt, valid + "garbage", valid + valid.substr(0, 4)};
        for (const auto& bytes : bad_inputs) {
            INFO(bytes.size());
            raw_write(path, bytes);
            auto input = open_seq_ifile(path);
            REQUIRE(input != nullptr);
            input->exceptions(std::ios::goodbit);
            REQUIRE_NOTHROW(drain(*input));
            REQUIRE(input->bad());
            REQUIRE_FALSE(input->eof());
            input->clear();
            input->exceptions(std::ios::badbit);
            REQUIRE_THROWS_AS(input->peek(), std::runtime_error);
        }
    }
}

TEST_CASE("Sequential compressed streams do not seek and legacy streams still do", "[seq_compressed][compatibility]") {
    Directory directory;
    for (const auto& extension : compiled_codecs()) {
        const auto path = directory.file("seek" + extension);
        member(path, "0123456789");
        auto input = open_seq_ifile(path);
        REQUIRE(input->rdbuf()->pubseekpos(0, std::ios::in) == std::streampos(-1));
        REQUIRE(input->get() == '0');
        auto legacy = open_ifile(path);
        REQUIRE(legacy != nullptr);
        legacy->seekg(7);
        REQUIRE(legacy->get() == '7');
        legacy->seekg(2);
        REQUIRE(legacy->get() == '2');
    }
}

#ifdef USE_GZIP
TEST_CASE("Sequential gzip crosses exact input-buffer boundaries between members", "[seq_compressed][gzip][members]") {
    Directory directory;
    // Four bytes are probed separately; subsequent input buffers are 64 KiB.
    const size_t boundary = GENERATE(65539, 65540, 65541);
    const auto path = directory.file("boundary.gz");
    auto first = member(path, "first\n");
    const auto second = member(directory.file("second.gz"), "second\n");
    const size_t padding = boundary - first.size() - 2;
    REQUIRE(padding <= 65535);
    first[3] |= 4; // FEXTRA, an optional gzip header field.
    std::string extra;
    extra.push_back(static_cast<char>(padding & 255));
    extra.push_back(static_cast<char>(padding >> 8));
    extra.append(padding, '\0');
    first.insert(10, extra);
    REQUIRE(first.size() == boundary);
    raw_write(path, first + second);
    auto input = open_seq_ifile(path);
    REQUIRE(read_chunks(*input) == "first\nsecond\n");
    REQUIRE_FALSE(input->bad());
    raw_write(path, first);
    input = open_seq_ifile(path);
    REQUIRE(read_chunks(*input) == "first\n");
    REQUIRE_FALSE(input->bad());
}
#endif

#ifdef USE_ZSTD
TEST_CASE("Sequential zstd crosses buffer boundaries including skippable frames", "[seq_compressed][zstd][members]") {
    Directory directory;
    const size_t boundary = GENERATE(65539, 65540, 65541);
    const auto path = directory.file("boundary.zst");
    auto first = member(path, "first\n");
    const auto second = member(directory.file("second.zst"), "second\n");
    const size_t padding = boundary - first.size() - 8;
    std::string skip("\x50\x2a\x4d\x18", 4);
    for (int shift = 0; shift < 32; shift += 8)
        skip.push_back(static_cast<char>((padding >> shift) & 255));
    skip.append(padding, '\0');
    first += skip;
    REQUIRE(first.size() == boundary);
    raw_write(path, first + second);
    auto input = open_seq_ifile(path);
    REQUIRE(read_chunks(*input) == "first\nsecond\n");
    REQUIRE_FALSE(input->bad());
    raw_write(path, first);
    input = open_seq_ifile(path);
    REQUIRE(read_chunks(*input) == "first\n");
    REQUIRE_FALSE(input->bad());
    const auto magic_path = directory.file("skip-first.dat");
    raw_write(magic_path, skip + second);
    input = open_seq_ifile(magic_path);
    REQUIRE(input != nullptr);
    REQUIRE(read_chunks(*input) == "second\n");
    REQUIRE_FALSE(input->bad());
}
#endif

#ifndef USE_GZIP
TEST_CASE("Sequential disabled gzip is rejected by suffix and magic", "[seq_compressed][disabled]") {
    Directory directory;
    for (const auto& name : {"disabled.gz", "magic.dat"}) {
        const auto path = directory.file(name);
        raw_write(path, std::string("\x1f\x8b\x08\x00", 4));
        REQUIRE(open_seq_ifile(path) == nullptr);
    }
}
#endif
#ifndef USE_ZSTD
TEST_CASE("Sequential disabled zstd is rejected by suffix and magic", "[seq_compressed][disabled]") {
    Directory directory;
    for (const auto& name : {"disabled.zst", "magic.dat"}) {
        const auto path = directory.file(name);
        raw_write(path, std::string("\x28\xb5\x2f\xfd", 4));
        REQUIRE(open_seq_ifile(path) == nullptr);
    }
    const auto skip_path = directory.file("skip.dat");
    raw_write(skip_path, std::string("\x50\x2a\x4d\x18", 4));
    REQUIRE(open_seq_ifile(skip_path) == nullptr);
}
#endif
