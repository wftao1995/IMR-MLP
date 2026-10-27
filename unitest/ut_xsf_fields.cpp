#include <catch2/catch_all.hpp>
#include "io/xsf_fields.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace {
// Independent literal file oracle: x is fastest, f(i,j,k)=100*i+10*j+k.
const std::string grid3 =
    "BEGIN_BLOCK_DATAGRID_3D\nblock#1\nBEGIN_DATAGRID_3D_rho#1\n"
    "2 3 4\n1 2 3\n2 .5 0\n0 3 .25\n.5 0 4\n"
    "0 100 10 110 20 120\n1 101 11 111 21 121\n"
    "2 102 12 112 22 122\n3 103 13 113 23 123\n"
    "END_DATAGRID_3D\nEND_BLOCK_DATAGRID_3D\n";
const std::string grid2 =
    "BEGIN_BLOCK_DATAGRID_2D\nplane\nBEGIN_DATAGRID_2D_potential\n"
    "2 3\n-1 2 3\n2 .5 1\n.25 3 .5\n"
    "0e0 100 10\n110 20 120\nEND_DATAGRID_2D\nEND_BLOCK_DATAGRID_2D\n";
std::vector<XsfDataGridBlock> parse(const std::string& text) {
    std::istringstream input(text);
    return read_xsf_fields(input);
}
std::string replace(std::string text, const std::string& from, const std::string& to) {
    text.replace(text.find(from), from.size(), to);
    return text;
}
class NonseekBuffer : public std::stringbuf {
public:
    using std::stringbuf::stringbuf;
protected:
    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override { return pos_type(off_type(-1)); }
    pos_type seekpos(pos_type, std::ios_base::openmode) override { return pos_type(off_type(-1)); }
};
}

TEST_CASE("XSF fields read independent nonsymmetric 2D and 3D oracles", "[xsf_fields]") {
    auto blocks = parse("CRYSTAL\nPRIMVEC\n1 0 0\n0 1 0\n0 0 1\nPRIMCOORD\n1 1\nH 0 0 0\n" + grid3 + "# between blocks\n" + grid2);
    REQUIRE(blocks.size() == 2);
    REQUIRE(blocks[0].identifier == "block#1");
    REQUIRE(blocks[0].fields[0].identifier == "rho#1");
    REQUIRE(blocks[0].layout.shape == std::vector<std::size_t>{2, 3, 4});
    const auto& values = blocks[0].fields[0].values;
    for (std::size_t i = 0; i < 2; ++i)
        for (std::size_t j = 0; j < 3; ++j)
            for (std::size_t k = 0; k < 4; ++k)
                REQUIRE(values[(i * 3 + j) * 4 + k] == 100*i + 10*j + k);
    REQUIRE(blocks[1].fields[0].values == std::vector<double>{0,10,20,100,110,120});
    REQUIRE(blocks[0].layout.coordinate({1,2,3}) == std::array<double,3>{3.5,5.5,7.25});
    REQUIRE(blocks[1].layout.coordinate({1,2}) == std::array<double,3>{1.25,5.5,4.5});
    REQUIRE(blocks[0].layout.coordinate({0,0,0}) == blocks[0].layout.origin);
    REQUIRE_THROWS(blocks[0].layout.coordinate({2,0,0}));
    REQUIRE_THROWS(blocks[0].layout.coordinate({0,0}));
}

TEST_CASE("XSF field writer independently emits x-fast order and ordered components", "[xsf_fields]") {
    XsfDataGridBlock b;
    b.identifier = "vector";
    b.layout.shape = {2,3,4}; b.layout.origin = {1,2,3};
    b.layout.spanning_vectors = {{2,.5,0},{0,3,.25},{.5,0,4}};
    b.fields = {{"x",{}},{"y",{}},{"z",{}}};
    for (auto& field : b.fields)
        for (std::size_t i = 0; i < 2; ++i)
            for (std::size_t j = 0; j < 3; ++j)
                for (std::size_t k = 0; k < 4; ++k) field.values.push_back(100*i+10*j+k);
    std::ostringstream output;
    write_xsf_fields(output, {b});
    // Direct text assertion (not a writer-reader agreement).
    REQUIRE(output.str().find("0 100 10 110 20 120\n1 101 11 111 21 121\n2 102 12 112 22 122\n3 103 13 113 23 123\n") != std::string::npos);
    auto again = parse(output.str());
    REQUIRE(again[0].fields.size() == 3);
    REQUIRE(again[0].fields[2].identifier == "z");
    REQUIRE(again[0].fields[1].values == b.fields[1].values);
}

TEST_CASE("XSF fields accept nonseekable input arbitrary numeric lines and singleton axes", "[xsf_fields]") {
    NonseekBuffer buffer(replace(grid2, "2 3\n-1 2 3", "2\n3 -1\n2 3"));
    std::istream input(&buffer);
    REQUIRE(read_xsf_fields(input)[0].layout.shape == std::vector<std::size_t>{2,3});
    auto blocks = parse("ATOMS\nH 0 0 0\n" + replace(replace(grid2, "2 3", "1 3"), "0e0 100 10\n110 20 120", "1 2 3"));
    REQUIRE(blocks[0].layout.coordinate({0,2}) == std::array<double,3>{-.75,5,3.5});
    REQUIRE(blocks[0].fields[0].values == std::vector<double>{1,2,3});
}

TEST_CASE("XSF fields reject malformed unsupported and inconsistent input", "[xsf_fields]") {
    auto second = grid2.substr(grid2.find("BEGIN_DATAGRID"), grid2.find("END_BLOCK") - grid2.find("BEGIN_DATAGRID"));
    for (const auto& bad : {
        std::string{}, std::string("garbage\n") + grid3, std::string("ANIMSTEPS 1\n") + grid3,
        std::string("ATOMS 1\nH 0 0 0\n") + grid3,
        std::string("BEGIN_BLOCK_BANDGRID_3D\n"),
        replace(grid3, "2 3 4", "0 3 4"), replace(grid3, "2 3 4", "-2 3 4"),
        replace(grid3, "2 3 4", "18446744073709551615 3 4"),
        replace(grid3, "0 100", "nan 100"), replace(grid3, "2 .5 0", "inf .5 0"),
        replace(grid3, "0 100", "100"), replace(grid3, "0 100", "999 0 100"),
        replace(grid3, "END_DATAGRID_3D", "END_DATAGRID_2D"),
        replace(grid3, "END_BLOCK_DATAGRID_3D", "END_BLOCK_DATAGRID_2D"),
        grid3.substr(0, grid3.find("23 123")),
        replace(grid2, "END_BLOCK_DATAGRID_2D", replace(second, "2 .5 1", "3 .5 1") + "END_BLOCK_DATAGRID_2D"),
        grid2 + "garbage\n"}) {
        CAPTURE(bad);
        REQUIRE_THROWS(parse(bad));
    }
    auto b = parse(grid2)[0];
    b.fields[0].values.pop_back();
    std::ostringstream output;
    REQUIRE_THROWS(write_xsf_fields(output, {parse(grid3)[0], b}));
    REQUIRE(output.str().empty());
    b = parse(grid2)[0]; b.fields[0].values[0] = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS(write_xsf_fields(output, {b}));
}

TEST_CASE("XSF fields path I/O reuses compressed streams and preserves invalid output", "[xsf_fields]") {
    auto blocks = parse(grid2 + grid3);
    const auto base = std::filesystem::temp_directory_path() / "imrmlp-ut-xsf-fields";
    std::vector<std::string> suffixes{".xsf"};
#ifdef USE_GZIP
    suffixes.push_back(".xsf.gz");
#endif
#ifdef USE_ZSTD
    suffixes.push_back(".xsf.zst");
#endif
    for (const auto& suffix : suffixes) {
        const auto path = base.string() + suffix;
        write_xsf_fields(path, blocks);
        auto again = read_xsf_fields(path);
        REQUIRE(again[1].fields[0].values == blocks[1].fields[0].values);
        auto invalid_blocks = blocks;
        invalid_blocks[0].fields[0].values.clear();
        REQUIRE_THROWS(write_xsf_fields(path, invalid_blocks));
        REQUIRE(read_xsf_fields(path)[1].fields[0].values == blocks[1].fields[0].values);
        std::filesystem::remove(path);
    }
    REQUIRE_THROWS(read_xsf_fields(base.string() + "/missing.xsf"));
    REQUIRE_THROWS(write_xsf_fields(base.string() + "/missing.xsf", blocks));
}
