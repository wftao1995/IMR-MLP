#include <catch2/catch_all.hpp>

#include "io/gpio.h"
#include "utils/compressed_stream.h"

#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

// Independent VASP text fixtures, including a tilted cell and a negative
// POSCAR-style volume scale. Input configuration labels need not be contiguous.
const std::vector<std::string> headers = {
    "frame zero\n2\n2 0 0\n0.5 3 0\n0 0 4\nAl O\n1 2\n",
    "frame one\n1\n5 0.2 0\n0 7 0.1\n0.3 0 9\nO Al\n2 1\n",
    "frame two\n-8\n1 0 0\n0 1 0\n0 0 1\nAl O\n1 2\n",
};
const std::vector<std::string> configurations = {
    "Direct configuration= 7\n0.1 0.2 0.3\n0.4 0.5 0.6\n-0.1 1.1 0\n",
    "Direct configuration = 41\n0 0 0\n0.2 0.3 0.4\n1 1 1\n",
    "Direct configuration=103\n0.1 0 0\n0 0.2 0\n0 0 0.3\n",
};

std::string variable_trajectory(bool duplicate_first_header = false)
{
    std::string result;
    for (size_t i = 0; i < headers.size(); ++i) {
        result += headers[i];
        if (i == 0 && duplicate_first_header) result += headers[i];
        result += configurations[i];
    }
    return result;
}

void check_same(const Atoms &actual, const Atoms &expected)
{
    CHECK(actual.id == expected.id);
    CHECK(actual.chemmap_ == expected.chemmap_);
    CHECK(actual.pbc_ == expected.pbc_);
    CHECK(actual.cartesian == expected.cartesian);
    CHECK(actual.cell_.isApprox(expected.cell_, 1e-12));
    REQUIRE(actual.size() == expected.size());
    for (int i = 0; i < actual.size(); ++i)
        CHECK(actual.pos_[i].isApprox(expected.pos_[i], 1e-12));
}

std::vector<Atoms> read_batch(const std::string &text, const std::string &format)
{
    GPIO reader;
    std::istringstream input(text);
    std::vector<Atoms> frames;
    REQUIRE(reader.read(frames, input, format));
    return frames;
}

void check_sequence(GPIO &reader, std::istream &input, const std::string &format,
                    const std::vector<Atoms> &expected)
{
    REQUIRE(reader.seq_read_init(input, format));
    Atoms frame;
    for (const auto &reference : expected) {
        REQUIRE(reader.seq_read(input, frame));
        check_same(frame, reference);
    }
    REQUIRE_FALSE(reader.seq_read(input, frame));
    CHECK(frame.size() == 0);
    REQUIRE_FALSE(reader.seq_read(input, frame));
}

std::string replace_once(std::string text, const std::string &from, const std::string &to)
{
    const auto position = text.find(from);
    REQUIRE(position != std::string::npos);
    text.replace(position, from.size(), to);
    return text;
}

struct Workspace {
    std::filesystem::path path;
    Workspace() {
        char pattern[] = "/tmp/imrmlp_xdatcar6_XXXXXX";
        const char *created = mkdtemp(pattern);
        if (!created) throw std::runtime_error("mkdtemp failed");
        path = created;
    }
    ~Workspace() { std::filesystem::remove_all(path); }
};

std::vector<Atoms> fixed_output_frames() {
    Atoms first;
    first.cell_ = 2 * Eigen::Matrix3d::Identity();
    first.pbc_ = {true, true, true};
    first.pos_ = {{.2, .1, .4}, {.8, .3, .6}};
    first.chemmap_ = {1, 2};
    first.id = 0;
    auto second = first;
    second.id = 1;
    second.chemmap_ = {2, 1};
    second.pos_ = {{.7, .4, .5}, {.3, .2, .1}};
    return {first, second};
}

} // namespace

TEST_CASE("Fixed XDATCAR groups every frame without reusing old atom indices", "[gpio][xdatcar][write]") {
    const bool sequential = GENERATE(false, true);
    const bool fractional = GENERATE(false, true);
    auto frames = fixed_output_frames();
    if (fractional) for (auto &frame : frames) frame.to_fractional();
    const auto source = frames;
    GPIO writer;
    std::ostringstream output;
    if (sequential) {
        REQUIRE(writer.seq_write_init(output, "xdatcar"));
        for (auto &frame : frames) REQUIRE(writer.seq_write(output, frame));
    } else REQUIRE(writer.write(frames, output, "xdatcar"));
    const auto actual = read_batch(output.str(), "xdatcar");
    REQUIRE(actual.size() == 2);
    for (const auto &frame : actual) CHECK(frame.chemmap_ == std::vector<int>{1, 2});
    CHECK(actual[1].pos_[0].isApprox(Eigen::RowVector3d(.3, .2, .1), 1e-10));
    CHECK(actual[1].pos_[1].isApprox(Eigen::RowVector3d(.7, .4, .5), 1e-10));
    for (size_t i = 0; i < frames.size(); ++i) check_same(frames[i], source[i]);
}

TEST_CASE("Fixed XDATCAR rejects incompatible frames before writing their coordinates", "[gpio][xdatcar][write]") {
    const bool sequential = GENERATE(false, true);
    const int change = GENERATE(0, 1, 2, 3);
    auto frames = fixed_output_frames();
    auto &second = frames[1];
    if (change == 0) second.cell_ *= 2;
    if (change == 1) second.chemmap_ = {1, 1};
    if (change == 2) { second.pos_.pop_back(); second.chemmap_.pop_back(); }
    if (change == 3) { second.pos_.emplace_back(.4, .2, .8); second.chemmap_.push_back(1); }
    GPIO writer;
    std::ostringstream output;
    if (sequential) {
        REQUIRE(writer.seq_write_init(output, "xdatcar"));
        REQUIRE(writer.seq_write(output, frames[0]));
        const auto prefix = output.str();
        REQUIRE_FALSE(writer.seq_write(output, second));
        CHECK(output.str() == prefix);
    } else REQUIRE_FALSE(writer.write(frames, output, "xdatcar"));
    const auto actual = read_batch(output.str(), "xdatcar");
    REQUIRE(actual.size() == 1);
    check_same(actual[0], frames[0]);
}

TEST_CASE("Fixed XDATCAR takes its header from the first selected output frame", "[gpio][xdatcar][write]") {
    auto frames = fixed_output_frames();
    auto skipped = frames[0];
    skipped.cell_ *= 2;
    frames.insert(frames.begin(), skipped);
    GPIO writer;
    writer.out_startfrom = 1;
    std::ostringstream output;
    REQUIRE(writer.write(frames, output, "xdatcar"));
    const auto actual = read_batch(output.str(), "xdatcar");
    REQUIRE(actual.size() == 2);
    CHECK(actual[0].cell_.isApprox(frames[1].cell_, 1e-10));
    CHECK(actual[0].pos_[0].isApprox(frames[1].pos_[0], 1e-10));
}

TEST_CASE("Fixed XDATCAR tolerates cell roundoff and resets for new outputs", "[gpio][xdatcar][write]") {
    auto frames = fixed_output_frames();
    frames[1].cell_(0, 0) += 1e-12;
    GPIO writer;
    std::ostringstream first;
    REQUIRE(writer.seq_write_init(first, "xdatcar"));
    for (auto &frame : frames) REQUIRE(writer.seq_write(first, frame));
    auto actual = read_batch(first.str(), "xdatcar");
    CHECK(actual[1].pos_[0].isApprox(Eigen::RowVector3d(.3, .2, .1), 1e-10));
    frames[0].cell_ *= 3;
    std::ostringstream second;
    REQUIRE(writer.seq_write_init(second, "xdatcar"));
    REQUIRE(writer.seq_write(second, frames[0]));
    actual = read_batch(second.str(), "xdatcar");
    REQUIRE(actual.size() == 1);
    CHECK(actual[0].cell_.isApprox(frames[0].cell_, 1e-10));

    Workspace workspace;
    std::vector<Atoms> one{frames[0]};
    REQUIRE(writer.write(one, (workspace.path / "first.xdatcar").string(), "xdatcar"));
    one[0].cell_ *= 2;
    REQUIRE(writer.write(one, (workspace.path / "second.xdatcar").string(), "xdatcar"));
    std::vector<Atoms> reloaded;
    GPIO reader;
    REQUIRE(reader.read(reloaded, (workspace.path / "second.xdatcar").string(), "xdatcar"));
    REQUIRE(reloaded.size() == 1);
    CHECK(reloaded[0].cell_.isApprox(one[0].cell_, 1e-10));
}

TEST_CASE("XDATCAR6 keeps per-frame cells and counts with the same writer", "[gpio][xdatcar6][write]") {
    auto frames = fixed_output_frames();
    frames[1].cell_ *= 2;
    frames[1].pos_.pop_back();
    frames[1].chemmap_.pop_back();
    GPIO writer;
    std::ostringstream output;
    REQUIRE(writer.seq_write_init(output, "xdatcar6"));
    for (auto &frame : frames) REQUIRE(writer.seq_write(output, frame));
    const auto actual = read_batch(output.str(), "xdatcar6");
    REQUIRE(actual.size() == frames.size());
    for (size_t i = 0; i < frames.size(); ++i) check_same(actual[i], frames[i]);
}

TEST_CASE("XDATCAR6 sequential frames match batch cells species and positions", "[gpio][xdatcar6]")
{
    const bool final_newline = GENERATE(false, true);
    std::string text = variable_trajectory();
    if (!final_newline) text.pop_back();
    const auto expected = read_batch(text, "xdatcar6");
    REQUIRE(expected.size() == 3);
    for (int i = 0; i < 3; ++i) CHECK(expected[i].id == i);
    CHECK(expected[0].chemmap_ == std::vector<int>{13, 8, 8});
    CHECK(expected[1].chemmap_ == std::vector<int>{8, 8, 13});
    CHECK(expected[2].chemmap_ == std::vector<int>{13, 8, 8});
    CHECK(expected[0].cell_(1, 0) == Catch::Approx(1));
    CHECK(expected[0].pos_[0].isApprox(Eigen::RowVector3d{0.6, 1.2, 2.4}, 1e-12));
    CHECK(expected[1].cell_(2, 0) == Catch::Approx(0.3));
    CHECK(expected[1].pos_[1].isApprox(Eigen::RowVector3d{1.12, 2.14, 3.63}, 1e-12));
    CHECK(expected[2].cell_.isApprox(2 * Eigen::Matrix3d::Identity(), 1e-12));
    CHECK(expected[2].pos_[2].isApprox(Eigen::RowVector3d{0, 0, 0.6}, 1e-12));

    GPIO reader;
    std::istringstream input(text);
    check_sequence(reader, input, "XDATCAR6", expected);
    // Reinitializing the same reader resets both the ID and pending header.
    std::istringstream again(text);
    check_sequence(reader, again, "xdatcar6", expected);
}

TEST_CASE("XDATCAR6 respects custom element maps", "[gpio][xdatcar6]")
{
    GPIO reader;
    reader.setmap(std::vector<std::string>{"O", "Al"});
    std::istringstream input(variable_trajectory());
    REQUIRE(reader.seq_read_init(input, "xdatcar6"));
    Atoms atom;
    REQUIRE(reader.seq_read(input, atom));
    CHECK(atom.chemmap_ == std::vector<int>{1, 0, 0});
    REQUIRE(reader.seq_read(input, atom));
    CHECK(atom.chemmap_ == std::vector<int>{0, 0, 1});
}

TEST_CASE("Fixed cell XDATCAR remains compatible with sequential and batch reads", "[gpio][xdatcar6][xdatcar]")
{
    const auto ending = GENERATE(std::string{}, std::string{"\n"}, std::string{"\n \t\n"});
    std::string text = headers[0] + configurations[0] + configurations[1] + configurations[2];
    text.pop_back();
    text += ending;
    const auto expected = read_batch(text, "xdatcar");
    REQUIRE(expected.size() == 3);
    CHECK(expected[2].cell_.isApprox(expected[0].cell_, 1e-12));
    CHECK(expected[2].pos_[2].isApprox(Eigen::RowVector3d{0, 0, 2.4}, 1e-12));
    GPIO reader;
    std::istringstream input(text);
    check_sequence(reader, input, "xdatcar", expected);
    std::istringstream again(text);
    check_sequence(reader, again, "xdatcar", expected);
}

TEST_CASE("Fixed cell XDATCAR preserves header-only clean EOF", "[gpio][xdatcar]")
{
    CHECK(read_batch(headers[0], "xdatcar").empty());
    GPIO reader;
    std::istringstream input(headers[0]);
    REQUIRE(reader.seq_read_init(input, "xdatcar"));
    Atoms atom;
    REQUIRE_FALSE(reader.seq_read(input, atom));
    CHECK(atom.size() == 0);
    REQUIRE_FALSE(reader.seq_read(input, atom));
}

TEST_CASE("Fixed cell XDATCAR rejects begun malformed frames and can restart", "[gpio][xdatcar][errors]")
{
    const std::string first = headers[0] + configurations[0];
    const auto expected = read_batch(first + configurations[1], "xdatcar");
    const std::vector<std::string> tails = {
        "Direct garbage= 41\n0 0 0\n0 0 0\n0 0 0\n",
        "Direct configuration= 41\n0 0 0\n",
        replace_once(configurations[1], "0.2 0.3 0.4", "0.2 invalid 0.4"),
    };
    for (const auto &tail : tails) {
        CAPTURE(tail);
        GPIO batch;
        std::istringstream batch_input(first + tail);
        std::vector<Atoms> frames;
        REQUIRE_FALSE(batch.read(frames, batch_input, "xdatcar"));
        REQUIRE(frames.size() == 1);
        check_same(frames[0], expected[0]);

        GPIO sequential;
        std::istringstream input(first + tail);
        REQUIRE(sequential.seq_read_init(input, "xdatcar"));
        Atoms atom;
        REQUIRE(sequential.seq_read(input, atom));
        const Atoms saved_first = atom;
        REQUIRE_THROWS_WITH(sequential.seq_read(input, atom),
                            "Invalid or truncated XDATCAR frame 1");
        CHECK(atom.size() == 0);
        CHECK_FALSE(sequential.seq_read(input, atom));
        check_same(saved_first, expected[0]);

        std::istringstream restarted(first + configurations[1]);
        check_sequence(sequential, restarted, "xdatcar", expected);
    }
}

TEST_CASE("Fixed cell XDATCAR stops after an input failure", "[gpio][xdatcar][errors]")
{
    GPIO reader;
    std::istringstream input(headers[0] + configurations[0] + configurations[1]);
    REQUIRE(reader.seq_read_init(input, "xdatcar"));
    Atoms atom;
    REQUIRE(reader.seq_read(input, atom));
    input.setstate(std::ios::badbit);
    REQUIRE_THROWS_WITH(reader.seq_read(input, atom), "Invalid or truncated XDATCAR frame 1");
    CHECK(atom.size() == 0);
    CHECK_FALSE(reader.seq_read(input, atom));
}

TEST_CASE("XDATCAR6 basename detection precedes fixed XDATCAR", "[gpio][xdatcar6]")
{
    CHECK(GPIO::guess_format("xdatcar6") == GPIO::XDATCAR6);
    CHECK(GPIO::guess_format("xdatcar6.gz") == GPIO::XDATCAR6);
}

TEST_CASE("XDATCAR6 accepts complete legacy duplicate headers but writes one header", "[gpio][xdatcar6]")
{
    const auto expected = read_batch(variable_trajectory(), "xdatcar6");
    const std::string legacy = variable_trajectory(true);
    const auto batch = read_batch(legacy, "xdatcar6");
    REQUIRE(batch.size() == expected.size());
    for (size_t i = 0; i < batch.size(); ++i) check_same(batch[i], expected[i]);
    GPIO reader;
    std::istringstream input(legacy);
    check_sequence(reader, input, "xdatcar6", expected);

    GPIO writer;
    std::ostringstream output;
    auto frames = expected;
    REQUIRE(writer.write(frames, output, "xdatcar6"));
    const std::string written = output.str();
    size_t count = 0, position = 0;
    while ((position = written.find("XDATCAR by GPIO", position)) != std::string::npos) {
        ++count;
        ++position;
    }
    CHECK(count == expected.size());
    const auto roundtrip = read_batch(written, "xdatcar6");
    REQUIRE(roundtrip.size() == expected.size());
    // Writers group atoms by element, so compare the cell and atom count here.
    for (size_t i = 0; i < roundtrip.size(); ++i) {
        CHECK(roundtrip[i].cell_.isApprox(expected[i].cell_, 1e-12));
        CHECK(roundtrip[i].size() == expected[i].size());
    }
}

TEST_CASE("XDATCAR rejects malformed or truncated initial headers", "[gpio][xdatcar6][errors]")
{
    std::vector<std::string> invalid = {
        replace_once(headers[0], "\n2\n", "\nbad\n"),
        replace_once(headers[0], "\n2\n", "\n0\n"),
        replace_once(headers[0], "\n2\n", "\nnan\n"),
        replace_once(headers[0], "2 0 0", "2 0"),
        replace_once(headers[0], "2 0 0", "nan 0 0"),
        replace_once(headers[0], "2 0 0", "0 0 0"),
        replace_once(headers[0], "Al O", "Unknown O"),
        replace_once(headers[0], "Al O", ""),
        replace_once(headers[0], "1 2", "1 -2"),
        replace_once(headers[0], "1 2", "1"),
        replace_once(headers[0], "1 2", "1 2 3"),
        replace_once(headers[0], "1 2", "1 two"),
        replace_once(headers[0], "1 2", "2147483647 2"),
        replace_once(headers[0], "1 2", "0 0"),
    };
    for (size_t end = headers[0].find('\n'); end != std::string::npos;
         end = headers[0].find('\n', end + 1)) {
        if (end + 1 < headers[0].size()) invalid.push_back(headers[0].substr(0, end + 1));
    }
    for (const auto &header : invalid) {
        CAPTURE(header);
        for (const auto *format : {"xdatcar", "xdatcar6"}) {
            GPIO sequential;
            std::istringstream seq_input(header);
            if (std::string_view(format) == "xdatcar6")
                REQUIRE_THROWS_WITH(sequential.seq_read_init(seq_input, format),
                                    "Invalid or truncated XDATCAR6 header");
            else
                REQUIRE_FALSE(sequential.seq_read_init(seq_input, format));
            GPIO batch;
            std::istringstream batch_input(header);
            std::vector<Atoms> output;
            REQUIRE_FALSE(batch.read(output, batch_input, format));
            CHECK(output.empty());
        }
    }
}

TEST_CASE("XDATCAR6 distinguishes truncated frames from clean EOF", "[gpio][xdatcar6][errors]")
{
    const std::string first = headers[0] + configurations[0];
    const std::vector<std::string> tails = {
        "unfinished comment\n", headers[1],
        headers[1] + "Direct configuration= 41\n",
        headers[1] + "Direct configuration= 41\n0 0 0\n0.2 0.3\n",
        headers[1] + "Direct configuration= 41\n0 0 0\n0.2 0.3 nan\n1 1 1\n",
        headers[1] + "Direct garbage= 41\n0 0 0\n0 0 0\n0 0 0\n",
        headers[1] + "garbage\n" + configurations[1],
        // A changed header cannot silently replace an empty frame.
        headers[1] + headers[2] + configurations[2],
    };
    for (const auto &tail : tails) {
        CAPTURE(tail);
        GPIO batch;
        std::istringstream batch_input(first + tail);
        std::vector<Atoms> frames;
        REQUIRE_FALSE(batch.read(frames, batch_input, "xdatcar6"));
        REQUIRE(frames.size() == 1);
        CHECK(frames[0].id == 0);

        GPIO sequential;
        std::istringstream seq_input(first + tail);
        REQUIRE(sequential.seq_read_init(seq_input, "xdatcar6"));
        Atoms atom;
        REQUIRE(sequential.seq_read(seq_input, atom));
        REQUIRE_THROWS_WITH(sequential.seq_read(seq_input, atom),
                            "Invalid or truncated XDATCAR6 frame 1");
        CHECK(atom.size() == 0);
        CHECK_FALSE(sequential.seq_read(seq_input, atom));
    }
    // A valid first header alone also starts a frame and must not look like EOF.
    GPIO reader;
    std::istringstream input(headers[0]);
    REQUIRE(reader.seq_read_init(input, "xdatcar6"));
    Atoms atom;
    REQUIRE_THROWS_WITH(reader.seq_read(input, atom), "Invalid or truncated XDATCAR6 frame 0");
    CHECK(atom.size() == 0);
}

TEST_CASE("XDATCAR6 requires at least one frame", "[gpio][xdatcar6][errors]")
{
    const auto text = GENERATE(std::string{}, std::string{"\n \t\n"});
    GPIO batch;
    std::istringstream input(text);
    std::vector<Atoms> frames;
    REQUIRE_FALSE(batch.read(frames, input, "xdatcar6"));
    CHECK(frames.empty());
    GPIO sequential;
    std::istringstream seq_input(text);
    REQUIRE_THROWS_WITH(sequential.seq_read_init(seq_input, "xdatcar6"),
                        "Invalid or truncated XDATCAR6 header");
}

TEST_CASE("XDATCAR6 allows blank separators and blank comments without seeking", "[gpio][xdatcar6]")
{
    const auto expected = read_batch(variable_trajectory(), "xdatcar6");
    std::string text = "\n \t\n" + headers[0] + configurations[0] + "\n \r\n" +
                       headers[1] + configurations[1] + "\n" + headers[2] + configurations[2] + "\n \t\n";
    SECTION("blank comment") {
        text = replace_once(text, "frame one", "");
    }
    SECTION("numeric comment after a separator") {
        text = replace_once(text, "frame one", "123");
    }
    SECTION("ordinary comments") {}
    // std::streambuf's default seek implementation fails, unlike stringstream.
    class NonSeekingBuffer : public std::streambuf {
    public:
        explicit NonSeekingBuffer(std::string &storage) {
            setg(storage.data(), storage.data(), storage.data() + storage.size());
        }
    } buffer(text);
    std::istream input(&buffer);
    GPIO reader;
    check_sequence(reader, input, "xdatcar6", expected);
    const auto batch = read_batch(text, "xdatcar6");
    REQUIRE(batch.size() == expected.size());
    for (size_t i = 0; i < batch.size(); ++i) check_same(batch[i], expected[i]);
}

TEST_CASE("XDATCAR6 sequential reads support compressed streams", "[gpio][xdatcar6][compressed]")
{
    Workspace workspace;
    const std::vector<std::string> extensions = {
        ".plain",
#ifdef USE_GZIP
        ".gz",
#endif
#ifdef USE_ZSTD
        ".zst",
#endif
    };
    const auto expected = read_batch(variable_trajectory(), "xdatcar6");
    for (const auto &extension : extensions) {
        CAPTURE(extension);
        const auto path = (workspace.path / ("XDATCAR6" + extension)).string();
        auto output = open_ofile(path);
        REQUIRE(output);
        *output << variable_trajectory();
        REQUIRE(close_ofile(output));
        auto input = open_ifile(path);
        REQUIRE(input);
        GPIO sequential;
        check_sequence(sequential, *input, "xdatcar6", expected);
        GPIO batch;
        std::vector<Atoms> actual;
        REQUIRE(batch.read(actual, path, "xdatcar6"));
        REQUIRE(actual.size() == expected.size());
        for (size_t i = 0; i < actual.size(); ++i) check_same(actual[i], expected[i]);
    }
}
