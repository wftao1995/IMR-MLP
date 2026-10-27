#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "io/gpio_binary.h"
#include <limits>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <streambuf>

namespace {
std::string cp2k(const std::string &atoms = "H 1 2 3\nO 4 5 6\n", const std::string &count = "2") {
    return count + "\ni = 0, time = 0\n" + atoms;
}
void reject_cp2k(const std::string &data) {
    GPIO reader;
    reader.setmap({"H", "O"});
    Atoms atom;
    atom.resize(1);
    atom.features_["stale"] = "data";
    std::istringstream one(data);
    REQUIRE_FALSE(reader.read_one(atom, one, "cp2kxyz"));
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.features_.empty());
    std::istringstream bulk(data);
    std::vector<Atoms> frames;
    REQUIRE_FALSE(reader.read(frames, bulk, "cp2kxyz"));
    REQUIRE(frames.empty());
    std::istringstream tail(cp2k() + data);
    REQUIRE_FALSE(reader.read(frames, tail, "cp2kxyz"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].size() == 2);
}
struct TextBuffer : std::streambuf {
    std::string data;
    size_t position = 0;
    size_t fail_at = std::numeric_limits<size_t>::max();
    explicit TextBuffer(std::string text) : data(std::move(text)) {}
    int_type underflow() override {
        if (position == fail_at) throw std::runtime_error("injected text I/O error");
        return position == data.size() ? traits_type::eof() : traits_type::to_int_type(data[position]);
    }
    int_type uflow() override {
        const auto value = underflow();
        if (!traits_type::eq_int_type(value, traits_type::eof())) ++position;
        return value;
    }
};
}

TEST_CASE("CP2KXYZ writer round trips multiple and zero atom frames", "[gpio][cp2kxyz]") {
    const auto newline = GENERATE(false, true);
    GPIO io;
    io.setmap({"H", "O"});
    Atoms original;
    original.resize(2);
    original.chemmap_ = {0, 1};
    original.pos_ = {{1, 2, 3}, {4, 5, 6}};
    Atoms empty;
    empty.clear();
    std::ostringstream out;
    REQUIRE(io.write_one(original, out, "cp2kxyz"));
    REQUIRE(io.write_one(empty, out, "cp2kxyz"));
    REQUIRE(io.write_one(original, out, "cp2kxyz"));
    auto text = out.str();
    if (!newline) text.pop_back();
    std::istringstream input(text);
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "cp2kxyz"));
    REQUIRE(frames.size() == 3);
    REQUIRE(frames[1].size() == 0);
    REQUIRE(frames[0].pos_ == original.pos_);
    REQUIRE(frames[2].chemmap_ == original.chemmap_);
    for (const auto &frame : frames) {
        REQUIRE(frame.is_nonperiodic());
        REQUIRE(frame.cell_.isZero());
    }
}

TEST_CASE("CP2KXYZ geometry without a cell supports nonperiodic distances", "[gpio][cp2kxyz]") {
    GPIO io;
    std::istringstream input("2\nmolecule geometry only\nH 0 0 0\nH 1.5 0 0\n");
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "cp2kxyz"));
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.is_nonperiodic());
    REQUIRE(atom.cell_.isZero());
    REQUIRE(atom.get_rab(0, 1).isApprox(Eigen::RowVector3d(1.5, 0, 0)));
    REQUIRE(atom.get_distance(0, 1) == Catch::Approx(1.5));
}

TEST_CASE("CP2KXYZ distinguishes clean EOF complete empty frame and malformed empty tail", "[gpio][cp2kxyz]") {
    const auto mask = GENERATE(std::ios::goodbit, std::ios::eofbit, std::ios::failbit,
        std::ios::badbit, std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    for (const auto &text : {"", "0\n\n", "0\ncomment", "0\n\n0\nlast comment"}) {
        CAPTURE(text, mask);
        GPIO io;
        TextBuffer buffer(text);
        std::istream input(&buffer);
        input.exceptions(mask);
        std::vector<Atoms> frames;
        REQUIRE(io.read(frames, input, "cp2kxyz"));
        const auto expected = std::string(text).empty() ? 0 : std::string(text).find("last") != std::string::npos ? 2 : 1;
        REQUIRE(frames.size() == expected);
    }
    for (const auto &text : {"0", "0\n", "\n", " \n", "1\n", "1\n\n"}) {
        // Invalid-input propagation with exception masks may throw; either
        // path must reject, never return success with a phantom empty frame.
        GPIO io;
        TextBuffer buffer(text);
        std::istream input(&buffer);
        input.exceptions(mask);
        std::vector<Atoms> frames;
        bool success = false;
        try { success = io.read(frames, input, "cp2kxyz"); } catch (const std::exception &) {}
        REQUIRE_FALSE(success);
        REQUIRE(frames.empty());
    }
}

TEST_CASE("CP2KXYZ rejects incomplete counts species coordinates and frames", "[gpio][cp2kxyz]") {
    for (const auto &count : {"-1", "2147483648", "2147483647", "1junk", "1 2", "nan", ""}) {
        CAPTURE(count);
        reject_cp2k(cp2k("", count));
    }
    reject_cp2k(cp2k("", std::to_string(gpio_binary::MAX_ATOMS)));
    reject_cp2k(cp2k("", std::to_string(gpio_binary::MAX_ATOMS + 1)));
    for (const auto &row : {"", "H", "H 1", "H 1 2", "Alien 1 2 3", "H nan 2 3", "H 1 inf 3",
                           "H 1 2 1e999", "H 1 2 3junk", "H 1 2 3 extra"}) {
        CAPTURE(row);
        reject_cp2k(cp2k(std::string(row) + "\n", "1"));
    }
    for (const auto &tail : {"0", "0\n", "\n", "1\n", "1\ncomment", "1\ncomment\nH 1 2"})
        reject_cp2k(tail);
}

TEST_CASE("CP2KXYZ refuses double mapping unknown-map pollution and stale filename outputs", "[gpio][cp2kxyz]") {
    GPIO io;
    std::vector<std::string> map(121, "Unused");
    map.back() = "H";
    io.setmap(map);
    Atoms atom;
    std::istringstream input(cp2k("H +1 +2 +3", "+1"));
    REQUIRE(io.read_one(atom, input, "cp2kxyz"));
    REQUIRE(atom.chemmap_ == std::vector<int>{120});
    io.absolute_index_read = true;
    std::istringstream absolute(cp2k());
    REQUIRE_FALSE(io.read_one(atom, absolute, "cp2kxyz"));
    REQUIRE(atom.size() == 0);
    io.absolute_index_read = false;
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::istringstream unknown(cp2k("Alien 1 2 3\n", "1"));
        REQUIRE_FALSE(io.read_one(atom, unknown, "cp2kxyz"));
    }
    atom.resize(1);
    REQUIRE_FALSE(io.read_one(atom, "/nonexistent/imrmlp-cp2k-input", "cp2kxyz"));
    REQUIRE(atom.size() == 0);
}

TEST_CASE("Checked text codecs reject stream failures and retain only complete frames", "[gpio][cp2kxyz]") {
    const auto mask = GENERATE(std::ios::goodbit, std::ios::eofbit, std::ios::failbit,
        std::ios::badbit, std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    const std::string data = cp2k();
    for (size_t position : {size_t(0), size_t(1), size_t(4), data.size() - 2, data.size()}) {
        GPIO io;
        io.setmap({"H", "O"});
        TextBuffer buffer(data);
        buffer.fail_at = position;
        std::istream input(&buffer);
        input.exceptions(mask);
        std::vector<Atoms> frames;
        bool success = false;
        try { success = io.read(frames, input, "cp2kxyz"); } catch (const std::exception &) {}
        REQUIRE_FALSE(success);
        REQUIRE(frames.size() == (position == data.size() ? 1 : 0));
    }
    for (auto state : {std::ios::badbit, std::ios::failbit, std::ios::eofbit | std::ios::failbit}) {
        GPIO io;
        std::istringstream input("");
        input.setstate(state);
        std::vector<Atoms> frames;
        REQUIRE_FALSE(io.read(frames, input, "cp2kxyz"));
    }
}

TEST_CASE("CP2KXYZ range stops after skipped end frame and across files", "[gpio][cp2kxyz]") {
    GPIO io;
    io.setmap({"H", "O"});
    io.read_irate = 2;
    io.read_endto = 1;
    std::istringstream input(cp2k() + cp2k() + "BAD TAIL\n");
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "cp2kxyz"));
    REQUIRE(frames.size() == 1);
    REQUIRE(input.peek() == 'B');
    REQUIRE(io.read(frames, input, "cp2kxyz"));
    REQUIRE(frames.size() == 1);
    REQUIRE(input.peek() == 'B');
    // An invalid skipped frame within the requested range must still fail.
    GPIO bad;
    bad.setmap({"H", "O"});
    bad.read_irate = 2;
    bad.read_endto = 1;
    std::istringstream bad_input(cp2k() + "BAD TAIL\n");
    frames.clear();
    REQUIRE_FALSE(bad.read(frames, bad_input, "cp2kxyz"));
    REQUIRE(frames.size() == 1);

    struct TemporaryDirectory {
        std::filesystem::path path;
        TemporaryDirectory() {
            auto name = (std::filesystem::temp_directory_path() / "imrmlp-cp2k-XXXXXX").string();
            const auto made = mkdtemp(name.data());
            if (!made) throw std::runtime_error("failed to create temporary directory");
            path = made;
        }
        ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
    } directory;
    const auto first = (directory.path / "first.xyz").string();
    const auto second = (directory.path / "second.xyz").string();
    { std::ofstream output(first); output << cp2k(); }
    { std::ofstream output(second); output << cp2k() << "BAD TAIL\n"; }
    GPIO files;
    files.setmap({"H", "O"});
    files.reset_status_every_read = false;
    files.read_irate = 2;
    files.read_endto = 1;
    frames.clear();
    REQUIRE(files.read(frames, first, "cp2kxyz"));
    REQUIRE(files.read(frames, second, "cp2kxyz"));
    REQUIRE(frames.size() == 1);
    REQUIRE(files.read(frames, (directory.path / "not-opened.xyz").string(), "cp2kxyz"));
    REQUIRE(frames.size() == 1);
}


TEST_CASE("CP2KXYZ sequential input retains frame IDs and complete empty frames", "[gpio][cp2kxyz][sequence]") {
    GPIO io;
    io.setmap({"H", "O"});
    auto text = cp2k() + cp2k("", "0") + cp2k();
    text.pop_back();
    TextBuffer buffer(text);
    std::istream input(&buffer);
    REQUIRE(io.seq_read_init(input, "cp2kxyz"));
    Atoms atom;
    for (int i = 0; i < 3; ++i) {
        REQUIRE(io.seq_read(input, atom));
        REQUIRE(atom.id == i);
        REQUIRE(atom.size() == (i == 1 ? 0 : 2));
        REQUIRE(atom.is_nonperiodic());
        REQUIRE(atom.cell_.isZero());
        if (i != 1) {
            REQUIRE(atom.chemmap_ == std::vector<int>{0, 1});
            REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3)));
            REQUIRE(atom.pos_[1].isApprox(Eigen::RowVector3d(4, 5, 6)));
        }
    }
    REQUIRE_FALSE(io.seq_read(input, atom));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
}

TEST_CASE("CP2KXYZ sequential input reports broken tails and can restart", "[gpio][cp2kxyz][sequence]") {
    const bool transport_error = GENERATE(false, true);
    GPIO io;
    TextBuffer buffer(cp2k() + "2\ncomment\nH 1 2 3\n");
    if (transport_error) buffer.fail_at = cp2k().size() + 2;
    std::istream input(&buffer);
    input.exceptions(std::ios::badbit);
    REQUIRE(io.seq_read_init(input, "cp2kxyz"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    REQUIRE_THROWS(io.seq_read(input, atom));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
    std::istringstream fresh(cp2k());
    REQUIRE(io.seq_read_init(fresh, "cp2kxyz"));
    REQUIRE(io.seq_read(fresh, atom));
    REQUIRE(atom.id == 0);
    REQUIRE_FALSE(io.seq_read(fresh, atom));
}
