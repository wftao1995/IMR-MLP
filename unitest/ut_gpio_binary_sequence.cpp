#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "io/gpio_binary.h"
#include "utils/compressed_stream.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <sstream>
#include <streambuf>

namespace {
using namespace gpio_binary;

Atoms binary_atom(int count = 2) {
    Atoms atom;
    atom.clear();
    atom.resize(count);
    atom.cell_ = 7 * Eigen::Matrix3d::Identity();
    for (int i = 0; i < count; ++i) {
        atom.pos_[i] << i + 0.1, i + 0.2, i + 0.3;
        atom.chemmap_[i] = i;
    }
    atom.pbc_ = {{true, false, true}};
    return atom;
}

std::string encode(Atoms atom) {
    std::ostringstream out(std::ios::out | std::ios::binary);
    GPIO writer;
    REQUIRE(writer.write_one(atom, out, "bin"));
    return out.str();
}

std::string complete_record() {
    auto atom = binary_atom();
    atom.energy_ = -3.25;
    atom.have_energy = true;
    atom.fenergy_ = -3.5;
    atom.have_fenergy = true;
    atom.stress_ = Eigen::Matrix3d::Identity();
    atom.have_stress = true;
    atom.forces_ = {{1, 2, 3}, {4, 5, 6}};
    atom.velocities_ = atom.forces_;
    atom.magmoms_ = atom.forces_;
    atom.site_energies_ = {-1.25, -2};
    atom.magmom_scalar_ = {1, -1};
    atom.charges_ = {0.5, -0.5};
    atom.ids_ = {3, 5};
    atom.features_["label"] = "binary frame";
    atom.local_attribs_d_["local_double"] = Eigen::VectorXd::Ones(2);
    atom.local_attribs_i_["local_int"] = Eigen::VectorXi::Ones(2);
    atom.local_attribs_v_["local_vector"] = Eigen::MatrixXd::Ones(3, 2);
    atom.local_attribs_vi_["local_ivector"] = Eigen::MatrixXi::Ones(3, 2);
    atom.global_attribs_d_["global_double"] = 9.5;
    atom.global_attribs_i_["global_int"] = 9;
    atom.global_attribs_v_["global_vector"] = Eigen::VectorXd::Ones(3);
    atom.global_attribs_vi_["global_ivector"] = Eigen::VectorXi::Ones(3);
    atom.hvp_directions_ = Eigen::MatrixXd::Ones(6, 1);
    atom.hvp_targets_ = Eigen::MatrixXd::Constant(6, 1, 2);
    return encode(atom);
}

void empty_output(const Atoms &atom) {
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.chemmap_.empty());
    REQUIRE(atom.forces_.empty());
    REQUIRE_FALSE(atom.have_energy);
    REQUIRE(atom.features_.empty());
    REQUIRE(atom.hvp_directions_.size() == 0);
}

void frame_error(GPIO &reader, std::istream &input, Atoms &atom, int frame) {
    REQUIRE_THROWS_WITH(reader.seq_read(input, atom),
        Catch::Matchers::ContainsSubstring("MLIPBIN frame " + std::to_string(frame) + ":"));
    empty_output(atom);
    // Failed readers remain stopped even if a new healthy stream is supplied.
    std::istringstream replacement(encode(binary_atom()));
    REQUIRE_FALSE(reader.seq_read(replacement, atom));
    REQUIRE(replacement.tellg() == 0);
    REQUIRE(reader.seq_read_init(replacement, "bin"));
    REQUIRE(reader.seq_read(replacement, atom));
    REQUIRE(atom.id == 0);
    REQUIRE(atom.size() == 2);
}

std::string header(std::int32_t count = 2, std::uint32_t version = VERSION) {
    std::ostringstream out(std::ios::out | std::ios::binary);
    REQUIRE(write_span(out, MAGIC, sizeof(MAGIC)));
    REQUIRE(write_value(out, version));
    REQUIRE(write_value(out, count));
    return out.str();
}

template<typename T>
std::string bytes(const T &value) {
    return std::string(reinterpret_cast<const char *>(&value), sizeof(value));
}

std::string tagged(Tag tag, const std::string &payload = {}) {
    return std::string(1, static_cast<char>(tag)) + payload;
}

std::string named(const std::string &name) {
    return bytes(static_cast<std::int32_t>(name.size())) + name;
}

struct NonseekableBuffer : std::streambuf {
    std::string data;
    std::size_t pos = 0;
    std::size_t failure_at = std::numeric_limits<std::size_t>::max();
    bool nonstandard_error = false;
    explicit NonseekableBuffer(std::string content) : data(std::move(content)) {}
    int_type underflow() override {
        if (pos >= failure_at) {
            if (nonstandard_error) throw 42;
            throw std::runtime_error("injected binary transport failure");
        }
        return pos == data.size() ? traits_type::eof() : traits_type::to_int_type(data[pos]);
    }
    int_type uflow() override {
        const auto value = underflow();
        if (!traits_type::eq_int_type(value, traits_type::eof())) ++pos;
        return value;
    }
    std::streamsize xsgetn(char *destination, std::streamsize count) override {
        std::streamsize copied = 0;
        while (copied < count) {
            const auto value = uflow();
            if (traits_type::eq_int_type(value, traits_type::eof())) break;
            destination[copied++] = traits_type::to_char_type(value);
        }
        return copied;
    }
};

struct TemporaryDirectory {
    std::filesystem::path path;
    TemporaryDirectory() {
        auto pattern = (std::filesystem::temp_directory_path() / "imrmlp-binary-sequence-XXXXXX").string();
        const auto created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("cannot create binary test directory");
        path = created;
    }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};
}

TEST_CASE("MLIPBIN sequence distinguishes zero atom records and normal EOF", "[gpio][binary][sequence]") {
    const auto mask = GENERATE(std::ios::goodbit, std::ios::eofbit, std::ios::failbit,
        std::ios::badbit, std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    const auto count = GENERATE(0, 1, 3);
    std::string data;
    for (int i = 0; i < count; ++i) data += encode(binary_atom(i == 1 ? 0 : 2));
    NonseekableBuffer buffer(data);
    std::istream input(&buffer);
    input.exceptions(mask);
    GPIO reader;
    REQUIRE(reader.seq_read_init(input, "bin"));
    auto atom = binary_atom();
    for (int i = 0; i < count; ++i) {
        REQUIRE(reader.seq_read(input, atom));
        REQUIRE(atom.id == i);
        REQUIRE(atom.size() == (i == 1 ? 0 : 2));
    }
    REQUIRE_FALSE(reader.seq_read(input, atom));
    empty_output(atom);
    REQUIRE_FALSE(reader.seq_read(input, atom));
    REQUIRE(input.exceptions() == mask);
    REQUIRE(buffer.pos == data.size());
    NonseekableBuffer batch_buffer(data);
    std::istream batch_input(&batch_buffer);
    batch_input.exceptions(mask);
    std::vector<Atoms> frames;
    GPIO batch_reader;
    REQUIRE(batch_reader.read(frames, batch_input, "bin"));
    REQUIRE(frames.size() == static_cast<std::size_t>(count));
    if (count == 3) REQUIRE(frames[1].size() == 0);
}

TEST_CASE("MLIPBIN sequence rejects every nonempty truncated record prefix", "[gpio][binary][sequence][truncation]") {
    const auto record = complete_record();
    const auto prefix = encode(binary_atom());
    for (std::size_t length = 1; length < record.size(); ++length) {
        CAPTURE(length, record.size());
        for (int preceding : {0, 1}) {
            CAPTURE(preceding);
            std::istringstream input((preceding ? prefix : "") + record.substr(0, length));
            GPIO reader;
            REQUIRE(reader.seq_read_init(input, "bin"));
            auto atom = binary_atom();
            if (preceding) REQUIRE(reader.seq_read(input, atom));
            REQUIRE_THROWS_WITH(reader.seq_read(input, atom),
                Catch::Matchers::ContainsSubstring("MLIPBIN frame " + std::to_string(preceding) + ":"));
            REQUIRE(atom.size() == 0);
            REQUIRE_FALSE(atom.have_energy);
            REQUIRE(atom.features_.empty());
            REQUIRE_FALSE(reader.seq_read(input, atom));
        }
        std::istringstream single_input(record.substr(0, length));
        GPIO single_reader;
        auto single = binary_atom();
        REQUIRE_FALSE(single_reader.read_one(single, single_input, "bin"));
        REQUIRE(single.size() == 0);
        std::istringstream batch_input(prefix + record.substr(0, length));
        GPIO batch_reader;
        std::vector<Atoms> frames;
        REQUIRE_FALSE(batch_reader.read(frames, batch_input, "bin"));
        REQUIRE(frames.size() == 1);
    }
}

TEST_CASE("MLIPBIN rejects malformed headers tags lengths and missing required fields", "[gpio][binary][sequence][errors]") {
    const std::string record = encode(binary_atom());
    const std::string cell = tagged(Tag::Cell, std::string(9 * sizeof(double), '\0'));
    const std::string positions = tagged(Tag::Positions, std::string(6 * sizeof(double), '\0'));
    std::vector<std::string> malformed = {
        "badmagic", header(2, 1), header(2, VERSION + 1), header(-1), header(MAX_ATOMS + 1),
        header() + "?", header() + tagged(Tag::End),
        header() + cell + tagged(Tag::End), header() + positions + tagged(Tag::End),
        record.substr(0, record.size() - 1),
        header() + tagged(Tag::PBC, std::string("\0\2\1", 3)),
        header() + tagged(Tag::Feature, bytes(std::int32_t(-1))),
        header() + tagged(Tag::Feature, bytes(std::int32_t(MAX_STRING_SIZE + 1))),
        header() + tagged(Tag::Feature, named("name") + bytes(std::int32_t(-1))),
        header() + tagged(Tag::Feature, named("name") + bytes(std::int32_t(MAX_STRING_SIZE + 1))),
    };
    for (auto tag : {Tag::GlobalDoubleVector, Tag::GlobalIntVector}) {
        for (auto size : {std::int32_t(-1), std::int32_t(MAX_VECTOR_SIZE + 1)})
            malformed.push_back(header() + tagged(tag, named("v") + bytes(size)));
    }
    for (auto tag : {Tag::LocalDoubleVector, Tag::LocalIntVector}) {
        for (auto size : {std::int32_t(-1), std::int32_t(MAX_VECTOR_SIZE + 1), std::int32_t(MAX_VECTOR_SIZE / 2 + 1)})
            malformed.push_back(header() + tagged(tag, named("v") + bytes(size)));
    }
    for (std::size_t i = 0; i < malformed.size(); ++i) {
        CAPTURE(i);
        for (int preceding : {0, 1}) {
            std::istringstream input((preceding ? record : "") + malformed[i] + record);
            GPIO reader;
            REQUIRE(reader.seq_read_init(input, "bin"));
            auto atom = binary_atom();
            if (preceding) REQUIRE(reader.seq_read(input, atom));
            frame_error(reader, input, atom, preceding);
        }
        std::istringstream one_input(malformed[i]);
        GPIO one;
        auto atom = binary_atom();
        REQUIRE_FALSE(one.read_one(atom, one_input, "bin"));
        empty_output(atom);
    }
}

TEST_CASE("MLIPBIN sequential failures carry context for all stream exception masks", "[gpio][binary][sequence][stream]") {
    const auto mask = GENERATE(std::ios::goodbit, std::ios::eofbit, std::ios::failbit,
        std::ios::badbit, std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    const auto prefix = encode(binary_atom());
    const auto record = complete_record();
    for (auto failure_at : {std::size_t(0), std::size_t(5), std::size_t(35), prefix.size()}) {
        CAPTURE(mask, failure_at);
        NonseekableBuffer buffer(prefix + record);
        buffer.failure_at = failure_at;
        std::istream input(&buffer);
        input.exceptions(mask);
        GPIO reader;
        REQUIRE(reader.seq_read_init(input, "bin"));
        auto atom = binary_atom();
        if (failure_at == prefix.size()) REQUIRE(reader.seq_read(input, atom));
        frame_error(reader, input, atom, failure_at == prefix.size() ? 1 : 0);
        REQUIRE(input.exceptions() == mask);
    }
}

TEST_CASE("MLIPBIN truncated reads with enabled stream exceptions still clear and stop", "[gpio][binary][sequence][stream]") {
    const auto mask = GENERATE(std::ios::eofbit, std::ios::failbit,
        std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    const auto record = complete_record();
    for (auto length : {std::size_t(1), std::size_t(11), std::size_t(35), record.size() - 1}) {
        CAPTURE(length, mask);
        std::istringstream input(record.substr(0, length));
        input.exceptions(mask);
        GPIO reader;
        REQUIRE(reader.seq_read_init(input, "bin"));
        auto atom = binary_atom();
        frame_error(reader, input, atom, 0);
        REQUIRE(input.exceptions() == mask);
    }
}

TEST_CASE("MLIPBIN preexisting failed stream states cannot be mistaken for clean EOF", "[gpio][binary][sequence][stream]") {
    const auto state = GENERATE(std::ios::badbit, std::ios::failbit,
        std::ios::eofbit | std::ios::badbit, std::ios::eofbit | std::ios::failbit,
        std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    std::istringstream input;
    input.setstate(state);
    GPIO reader;
    REQUIRE(reader.seq_read_init(input, "bin"));
    auto atom = binary_atom();
    frame_error(reader, input, atom, 0);
    std::istringstream batch_input;
    batch_input.setstate(state);
    GPIO batch_reader;
    std::vector<Atoms> frames;
    REQUIRE_FALSE(batch_reader.read(frames, batch_input, "bin"));
    REQUIRE(frames.empty());
    std::istringstream one_input;
    one_input.setstate(state);
    GPIO one;
    REQUIRE_FALSE(one.read_one(atom, one_input, "bin"));
    empty_output(atom);
}

TEST_CASE("MLIPBIN bulk reads report swallowed streambuf errors as failure", "[gpio][binary][sequence][stream]") {
    const auto record = complete_record();
    const auto prefix = encode(binary_atom());
    for (auto failure_at : {std::size_t(0), std::size_t(35), prefix.size()}) {
        NonseekableBuffer buffer(prefix + record);
        buffer.failure_at = failure_at;
        std::istream input(&buffer);
        GPIO reader;
        std::vector<Atoms> frames;
        REQUIRE_FALSE(reader.read(frames, input, "bin"));
        REQUIRE(frames.size() == (failure_at == prefix.size() ? 1 : 0));
    }
}

TEST_CASE("MLIPBIN nonstandard stream errors also terminate the sequential reader", "[gpio][binary][sequence][stream]") {
    NonseekableBuffer buffer("content");
    buffer.failure_at = 0;
    buffer.nonstandard_error = true;
    std::istream input(&buffer);
    input.exceptions(std::ios::badbit);
    GPIO reader;
    REQUIRE(reader.seq_read_init(input, "bin"));
    auto atom = binary_atom();
    frame_error(reader, input, atom, 0);
}

TEST_CASE("MLIPBIN HVP semantic failures discard the frame and include frame context", "[gpio][binary][sequence][hvp]") {
    const auto record = encode(binary_atom());
    for (auto payload : {std::string("nan"), std::string("1 2"), std::string("1 2 3 4 5 6")}) {
        const auto data = header() + tagged(Tag::Cell, std::string(9 * sizeof(double), '\0')) +
            tagged(Tag::Positions, std::string(6 * sizeof(double), '\0')) +
            tagged(Tag::Feature, named("hvp_targets") + named(payload)) + tagged(Tag::End);
        std::istringstream input(record + data);
        GPIO reader;
        REQUIRE(reader.seq_read_init(input, "bin"));
        auto atom = binary_atom();
        REQUIRE(reader.seq_read(input, atom));
        frame_error(reader, input, atom, 1);
    }
}

TEST_CASE("MLIPBIN v2 v3 and full field sequences agree with batch reads", "[gpio][binary][sequence][compatibility]") {
    auto old = encode(binary_atom(0));
    const std::uint32_t version = 2;
    std::memcpy(old.data() + sizeof(MAGIC), &version, sizeof(version));
    const auto record = complete_record();
    const auto data = old + record;
    GPIO batch, sequential;
    std::istringstream batch_input(data), sequence_input(data);
    std::vector<Atoms> frames;
    REQUIRE(batch.read(frames, batch_input, "bin"));
    REQUIRE(frames.size() == 2);
    REQUIRE(frames[0].size() == 0);
    REQUIRE(frames[0].pbc_ == std::array<bool, 3>{{true, true, true}});
    REQUIRE(sequential.seq_read_init(sequence_input, "bin"));
    Atoms atom;
    for (const auto &expected : frames) {
        REQUIRE(sequential.seq_read(sequence_input, atom));
        REQUIRE(atom.id == expected.id);
        REQUIRE(encode(atom) == encode(expected));
    }
    REQUIRE(atom.hvp_directions_.isApprox(Eigen::MatrixXd::Ones(6, 1)));
    REQUIRE(atom.hvp_targets_.isApprox(Eigen::MatrixXd::Constant(6, 1, 2)));
    REQUIRE_FALSE(sequential.seq_read(sequence_input, atom));
}

TEST_CASE("MLIPBIN absolute type remapping bounds are checked before lookup", "[gpio][binary][sequence][errors]") {
    const auto type = GENERATE(-1, 119, std::numeric_limits<int>::max());
    auto atom = binary_atom();
    atom.chemmap_[0] = type;
    const auto data = encode(atom);
    GPIO reader;
    reader.absolute_index_read = true;
    std::istringstream input(data);
    REQUIRE(reader.seq_read_init(input, "bin"));
    frame_error(reader, input, atom, 0);
    // Internal type indices are not atomic numbers and retain their old range.
    GPIO ordinary;
    std::istringstream regular_input(data);
    REQUIRE(ordinary.read_one(atom, regular_input, "bin"));
    REQUIRE(atom.chemmap_[0] == type);
}

TEST_CASE("MLIPBIN bounded batch reads do not inspect frames beyond the requested range", "[gpio][binary][sequence][range]") {
    const auto stride = GENERATE(1, 2);
    const auto data = encode(binary_atom()) + (stride == 1 ? "" : encode(binary_atom())) + "bad tail";
    std::istringstream input(data);
    GPIO reader;
    reader.read_irate = stride;
    reader.read_endto = stride - 1;
    REQUIRE_FALSE(reader.binary_read_limit_reached());
    std::vector<Atoms> frames;
    REQUIRE(reader.read(frames, input, "bin"));
    REQUIRE(frames.size() == 1);
    REQUIRE(reader.binary_read_limit_reached());
    REQUIRE(input.peek() == 'b');
    REQUIRE(reader.read(frames, input, "bin"));
    REQUIRE(input.peek() == 'b');
    REQUIRE(frames.size() == 1);
    reader.reset_status_every_read = false;
    TemporaryDirectory directory;
    REQUIRE(reader.read(frames, (directory.path / "nonexistent.bin").string(), "bin"));
    REQUIRE(frames.size() == 1);
    reader.reset_status_every_read = true;
    REQUIRE_FALSE(reader.read(frames, (directory.path / "nonexistent.bin").string(), "bin"));
    REQUIRE_FALSE(reader.binary_read_limit_reached());
}

TEST_CASE("MLIPBIN frame limits follow consumed frames across input streams", "[gpio][binary][sequence][range]") {
    GPIO reader;
    reader.read_irate = 2;
    reader.read_endto = 1;
    std::vector<Atoms> frames;
    std::istringstream empty;
    REQUIRE(reader.read(frames, empty, "bin"));
    REQUIRE_FALSE(reader.binary_read_limit_reached());
    std::istringstream first(encode(binary_atom()));
    REQUIRE(reader.read(frames, first, "bin"));
    REQUIRE_FALSE(reader.binary_read_limit_reached());
    REQUIRE(frames.size() == 1);
    std::istringstream second(encode(binary_atom()) + "bad tail");
    REQUIRE(reader.read(frames, second, "bin"));
    REQUIRE(reader.binary_read_limit_reached());
    REQUIRE(frames.size() == 1);
    REQUIRE(second.peek() == 'b');
}

TEST_CASE("MLIPBIN rejected frames inside the requested range cannot be hidden by stride", "[gpio][binary][sequence][range]") {
    const auto data = encode(binary_atom()) + "bad tail";
    std::istringstream input(data);
    GPIO reader;
    reader.read_irate = 2;
    reader.read_endto = 1;
    std::vector<Atoms> frames;
    REQUIRE_FALSE(reader.read(frames, input, "bin"));
    REQUIRE(frames.size() == 1);
}

TEST_CASE("MLIPBIN compressed filename reads honor the requested range and reject corrupt tails", "[gpio][binary][sequence][compressed]") {
    std::vector<std::string> extensions;
#ifdef USE_GZIP
    extensions.push_back(".gz");
#endif
#ifdef USE_ZSTD
    extensions.push_back(".zst");
#endif
    if (extensions.empty()) SKIP("No compressed input codec enabled");
    TemporaryDirectory directory;
    const auto record = complete_record();
    // Keep the damaged trailer beyond the decoder's fixed output buffer.
    std::string data;
    for (int i = 0; i < 300; ++i) data += record;
    REQUIRE(data.size() > 2 * 65536);
    for (const auto &extension : extensions) {
        CAPTURE(extension);
        const auto file = directory.path / ("trajectory.bin" + extension);
        auto output = open_ofile(file.string(), false, true);
        REQUIRE(output != nullptr);
        output->write(data.data(), data.size());
        REQUIRE(close_ofile(output));
        GPIO complete;
        std::vector<Atoms> frames;
        REQUIRE(complete.read(frames, file.string(), "bin"));
        REQUIRE(frames.size() == 300);
        const auto size = std::filesystem::file_size(file);
        REQUIRE(size > 3);
        std::filesystem::resize_file(file, size - 3);

        GPIO one;
        Atoms atom;
        REQUIRE(one.read_one(atom, file.string(), "bin"));
        REQUIRE(atom.size() == 2);
        GPIO bounded;
        bounded.read_endto = 0;
        frames.clear();
        REQUIRE(bounded.read(frames, file.string(), "bin"));
        REQUIRE(frames.size() == 1);

        GPIO all;
        frames.clear();
        REQUIRE_FALSE(all.read(frames, file.string(), "bin"));
        auto input = open_seq_ifile(file.string());
        REQUIRE(input != nullptr);
        GPIO sequential;
        REQUIRE(sequential.seq_read_init(*input, "bin"));
        int count = 0;
        while (true) {
            try {
                const bool success = sequential.seq_read(*input, atom);
                REQUIRE(success);
                ++count;
                REQUIRE(count <= 300);
            } catch (const std::runtime_error &error) {
                REQUIRE(std::string(error.what()).find("MLIPBIN frame " + std::to_string(count) + ":") != std::string::npos);
                break;
            }
        }
        REQUIRE(count > 1);
        empty_output(atom);
        REQUIRE_FALSE(sequential.seq_read(*input, atom));
    }
}
