#include <catch2/catch_all.hpp>
#include "io/format_detection.h"
#include "utils/compressed_stream.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>

namespace {
const std::string xyz = "2\nProperties=species:S:1:pos:R:3\nH 0 0 0\nHe 1 2 3\n";
std::string remainder(std::istream &input) {
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
struct Counts { std::size_t bytes = 0, seeks = 0; };
class NonSeekingBuffer final : public std::stringbuf {
public:
    NonSeekingBuffer(const std::string &text, Counts &counts) : std::stringbuf(text), counts_(counts) {}
protected:
    std::streamsize xsgetn(char *data, std::streamsize count) override {
        auto got = std::stringbuf::xsgetn(data, count);
        counts_.bytes += got;
        return got;
    }
    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override {
        ++counts_.seeks; return pos_type(off_type(-1));
    }
    pos_type seekpos(pos_type, std::ios_base::openmode) override {
        ++counts_.seeks; return pos_type(off_type(-1));
    }
private:
    Counts &counts_;
};
class FailingBuffer final : public std::stringbuf {
public:
    FailingBuffer() : std::stringbuf("BEGIN_CFG\nsource read failure") {}
protected:
    std::streamsize xsgetn(char *data, std::streamsize count) override {
        if (reads_++) throw std::runtime_error("synthetic source failure");
        return std::stringbuf::xsgetn(data, count);
    }
private:
    int reads_ = 0;
};
struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("imrmlp-detection-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { std::filesystem::create_directory(path); }
    ~TemporaryDirectory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
}

TEST_CASE("Format detection uses reliable content signatures", "[io][detection]") {
    using GPIOFormat::detect_prefix;
    REQUIRE(detect_prefix(std::string_view("GPMLBIN2\0binary", 15)).format == "bin");
    REQUIRE(detect_prefix("IMRMLPBIN").format.empty());
    REQUIRE(detect_prefix("\n BEGIN_CFG\n Size\n 1\n").format == "cfg");
    REQUIRE(detect_prefix("<?xml version=\"1.0\"?>\n<!--metadata-->\n<modeling>\n").format == "vasprun");
    REQUIRE(detect_prefix("<!-- <modeling> -->\n<other>\n").format.empty());
    REQUIRE(detect_prefix("<modeling_data>\n").format.empty());
    REQUIRE(detect_prefix("CRYSTAL\nPRIMVEC\n1 0 0\n0 1 0\n0 0 1\nPRIMCOORD\n1 1\n").format == "xsf");
    REQUIRE(detect_prefix("ATOMS\nH 0 0 0\n").format == "xsf");
    REQUIRE(detect_prefix("ANIMSTEPS 2\nATOMS 1\nH 0 0 0\n").format == "xsf");
    REQUIRE(detect_prefix("begin\ncomment n2p2\natom 0 0 0 H 0 0 1 2 3\nend\n").format == "n2p2");
    REQUIRE(detect_prefix("begin\natom nonsense\n").format.empty());
    REQUIRE(detect_prefix(xyz).format == "extxyz");
    REQUIRE(detect_prefix("1\n\nH 0 0 0\n").format == "extxyz");
    REQUIRE(detect_prefix("1\nheader\nH 0 x 0\n").format.empty());
    REQUIRE(detect_prefix("1\nheader\nH 0 nan 0\n").format.empty());
    REQUIRE(detect_prefix("0\nheader\n").format.empty());
}

TEST_CASE("Unknown and conflicting content never picks a default format", "[io][detection]") {
    using GPIOFormat::detect_prefix;
    for (const std::string text : {"", "\n\n", "1", "1\ncomment\n", "begin\n",
        "title\n1\n1 0 0\n0 1 0\n0 0 1\nH\n1\nDirect\n0 0 0\n",
        "ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n",
        "BEGIN_BLOCK_BANDGRID_3D\nexample\nBEGIN_BANDGRID_3D\n",
        "BEGIN_BLOCK_DATAGRID_3D\nexample\nBEGIN_DATAGRID_3D\n",
        "1\ni = 1, time = 0.0, E = -2.0\nH 0 0 0\n"}) {
        const auto detection = detect_prefix(text);
        INFO(text);
        REQUIRE(detection.format.empty());
        REQUIRE(detection.candidates.empty());
    }
    const auto conflict = detect_prefix("BEGIN_CFG\nATOMS\nH 0 0 0\n");
    REQUIRE(conflict.format.empty());
    REQUIRE(conflict.candidates == std::vector<std::string>{"cfg", "xsf"});
    REQUIRE(detect_prefix(std::string(GPIOFormat::MAX_PREFIX, '#') + "\nBEGIN_CFG\n").format.empty());
}

TEST_CASE("Bounded probe replays every byte without seeking", "[io][detection]") {
    for (const std::size_t limit : {std::size_t(1), std::size_t(8), std::size_t(53), GPIOFormat::MAX_PREFIX}) {
        Counts counts;
        std::string text = xyz + xyz + std::string(90000, 't');
        auto input = std::make_unique<OwnedIStream>(std::make_unique<NonSeekingBuffer>(text, counts));
        auto probed = GPIOFormat::probe(std::move(input), limit);
        REQUIRE(counts.bytes == limit);
        REQUIRE(counts.seeks == 0);
        REQUIRE(remainder(*probed.input) == text);
        REQUIRE(counts.seeks == 0);
    }
    auto short_input = GPIOFormat::probe(std::make_unique<std::istringstream>(xyz));
    REQUIRE(short_input.detection.format == "extxyz");
    REQUIRE(remainder(*short_input.input) == xyz);
    auto empty = GPIOFormat::probe(std::make_unique<std::istringstream>(""));
    REQUIRE(empty.detection.format.empty());
    REQUIRE(remainder(*empty.input).empty());
    REQUIRE_THROWS_AS(GPIOFormat::probe(nullptr), std::invalid_argument);
    REQUIRE_THROWS_AS(GPIOFormat::probe(std::make_unique<std::istringstream>(xyz), 0), std::invalid_argument);
    REQUIRE_THROWS_AS(GPIOFormat::probe(std::make_unique<std::istringstream>(xyz), GPIOFormat::MAX_PREFIX + 1), std::invalid_argument);
}

TEST_CASE("Probe supports caller EOF exception policy", "[io][detection]") {
    auto source = std::make_unique<std::istringstream>(xyz);
    source->exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
    auto probed = GPIOFormat::probe(std::move(source));
    REQUIRE(remainder(*probed.input) == xyz);
}

TEST_CASE("Replay preserves source read failures", "[io][detection]") {
    auto source = std::make_unique<OwnedIStream>(std::make_unique<FailingBuffer>());
    auto probed = GPIOFormat::probe(std::move(source), 10);
    REQUIRE(probed.detection.format == "cfg");
    REQUIRE_THROWS_AS(remainder(*probed.input), std::runtime_error);
}

TEST_CASE("File detection ignores extensions and progressively decodes compressed files", "[io][detection]") {
    TemporaryDirectory temporary;
    std::vector<std::string> names = {"without_extension", "wrong.poscar"};
#ifdef USE_GZIP
    names.push_back("compressed.gz");
#endif
#ifdef USE_ZSTD
    names.push_back("compressed.zst");
#endif
    for (const auto &name : names) {
        const auto path = (temporary.path / name).string();
        auto output = open_ofile(path);
        REQUIRE(output);
        *output << xyz << xyz;
        REQUIRE(close_ofile(output));
        REQUIRE(GPIOFormat::detect_file(path).format == "extxyz");
        auto probed = GPIOFormat::probe(open_seq_ifile(path), 40);
        REQUIRE(probed.detection.format == "extxyz");
        REQUIRE(remainder(*probed.input) == xyz + xyz);
    }
    REQUIRE_THROWS_AS(GPIOFormat::detect_file((temporary.path / "missing").string()), std::runtime_error);
}
