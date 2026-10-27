#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "tools/tools.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <streambuf>

namespace {
const std::string lattice = "lattice 6 0 0\nlattice .2 7 0\nlattice .1 .3 8\n";
const std::string atom_row = "atom 1 2 3 H .25 -.5 .1 -.2 .3\n";
std::string frame(const std::string &body = atom_row + "energy -.25\ncharge 0\n") {
    return "begin\ncomment frame\n" + lattice + body + "end\n";
}
void reject_n2p2(const std::string &text) {
    CAPTURE(text);
    GPIO io;
    io.setmap({"H", "O"});
    Atoms atom;
    atom.resize(1);
    atom.features_["stale"] = "data";
    std::istringstream one(text);
    REQUIRE_FALSE(io.read_one(atom, one, "n2p2"));
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.forces_.empty());
    REQUIRE(atom.charges_.empty());
    REQUIRE(atom.site_energies_.empty());
    REQUIRE(atom.global_attribs_d_.empty());
    REQUIRE(atom.features_.empty());
    REQUIRE_FALSE(atom.have_energy);
    std::istringstream bulk(text);
    std::vector<Atoms> frames;
    REQUIRE_FALSE(io.read(frames, bulk, "n2p2"));
    REQUIRE(frames.empty());
    std::istringstream tail(frame() + text);
    REQUIRE_FALSE(io.read(frames, tail, "n2p2"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].size() == 1);
    // Failed frames never consume an ID, including a zero-atom partial frame.
    std::istringstream good(frame());
    REQUIRE(io.read_one(atom, good, "n2p2"));
    REQUIRE(atom.id == 1);
}
struct NonseekText : std::streambuf {
    std::string data;
    size_t position = 0, fail_at = std::numeric_limits<size_t>::max();
    explicit NonseekText(std::string text) : data(std::move(text)) {}
    int_type underflow() override {
        if (position == fail_at) throw std::runtime_error("injected N2P2 transport error");
        return position == data.size() ? traits_type::eof() : traits_type::to_int_type(data[position]);
    }
    int_type uflow() override {
        const auto value = underflow();
        if (!traits_type::eq_int_type(value, traits_type::eof())) ++position;
        return value;
    }
};
}

TEST_CASE("N2P2 commits complete periodic frames and retains Hartree Bohr units", "[gpio][n2p2]") {
    const auto newline = GENERATE(false, true);
    GPIO io;
    io.setmap({"O", "H"});
    const auto body = atom_row + "atom 2 3 4 O 0 0 -.4 .5 -.6\nenergy -.25\ncharge +0\n";
    auto text = frame(body) + frame("") + frame(body);
    if (!newline) text.pop_back();
    NonseekText buffer(text);
    std::istream input(&buffer);
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "n2p2"));
    REQUIRE(frames.size() == 3);
    REQUIRE(frames[1].size() == 0);
    REQUIRE_FALSE(frames[1].have_energy);
    REQUIRE(frames[1].charges_.empty());
    REQUIRE(frames[1].site_energies_.empty());
    REQUIRE_FALSE(frames[1].global_attribs_d_.contains("total_charge"));
    for (int i : {0, 2}) {
        const auto &atom = frames[i];
        REQUIRE(atom.id == i);
        REQUIRE(atom.chemmap_ == std::vector<int>{1, 0});
        REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3) * BOHR_TO_A));
        REQUIRE(atom.pos_[1].isApprox(Eigen::RowVector3d(2, 3, 4) * BOHR_TO_A));
        REQUIRE(atom.cell_.row(1).isApprox(Eigen::RowVector3d(.2, 7, 0) * BOHR_TO_A));
        REQUIRE(atom.forces_[0].isApprox(Eigen::Vector3d(.1, -.2, .3) * HARTREE_TO_EV * A_TO_BOHR));
        REQUIRE(atom.forces_[1].isApprox(Eigen::Vector3d(-.4, .5, -.6) * HARTREE_TO_EV * A_TO_BOHR));
        REQUIRE(atom.have_energy);
        REQUIRE(atom.energy_ == Catch::Approx(-.25 * HARTREE_TO_EV));
        REQUIRE(atom.charges_ == std::vector<double>{.25, 0});
        REQUIRE(atom.site_energies_ == std::vector<double>{-.5 * HARTREE_TO_EV, 0});
        REQUIRE(atom.global_attribs_d_.at("total_charge") == 0);
        REQUIRE_FALSE(atom.have_stress);
    }
}

TEST_CASE("N2P2 retains independent atomic and total charge labels through EXTXYZ", "[gpio][n2p2]") {
    GPIO io;
    const auto body = atom_row + "atom 2 3 4 O -.1 .125 -.4 .5 -.6\n";
    std::istringstream input(frame(body + "charge .8\n"));
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "n2p2"));
    REQUIRE(atom.charges_ == std::vector<double>{.25, -.1});
    REQUIRE(atom.site_energies_ == std::vector<double>{-.5 * HARTREE_TO_EV, .125 * HARTREE_TO_EV});
    REQUIRE(atom.global_attribs_d_.at("total_charge") == .8);

    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    Atoms back;
    REQUIRE(io.read_one(back, output, "extxyz"));
    REQUIRE(back.chemmap_ == atom.chemmap_);
    REQUIRE(back.charges_ == atom.charges_);
    REQUIRE(back.site_energies_.size() == atom.site_energies_.size());
    for (size_t i = 0; i < atom.site_energies_.size(); ++i)
        REQUIRE(back.site_energies_[i] == Catch::Approx(atom.site_energies_[i]));
    REQUIRE(back.global_attribs_d_.at("total_charge") == .8);

    std::istringstream without_total(frame(body));
    REQUIRE(io.read_one(back, without_total, "n2p2"));
    REQUIRE(back.charges_ == atom.charges_);
    REQUIRE_FALSE(back.global_attribs_d_.contains("total_charge"));
}

TEST_CASE("N2P2 permits whitespace optional comments and unique trailing metadata", "[gpio][n2p2]") {
    const auto text = " \r\n\tbegin \r\n\n" + lattice +
        "comment " + std::string(12000, 'x') + "\n" + atom_row +
        "\ncharge .5\ncomment after charge\nenergy +.25\n end \r\n\t \n";
    GPIO io;
    std::istringstream input(text);
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "n2p2"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].energy_ == Catch::Approx(.25 * HARTREE_TO_EV));
    std::istringstream no_comment("begin\n" + lattice + "end");
    Atoms atom;
    REQUIRE(io.read_one(atom, no_comment, "n2p2"));
    REQUIRE(atom.size() == 0);
}

TEST_CASE("N2P2 distinguishes clean EOF from every incomplete frame boundary", "[gpio][n2p2]") {
    const auto mask = GENERATE(std::ios::goodbit, std::ios::eofbit, std::ios::failbit,
        std::ios::badbit, std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    for (const auto &text : {std::string(""), std::string(" \n\t"), frame(), frame() + " \n\t", frame("")}) {
        GPIO io;
        NonseekText buffer(text);
        std::istream input(&buffer);
        input.exceptions(mask);
        std::vector<Atoms> frames;
        REQUIRE(io.read(frames, input, "n2p2"));
        REQUIRE(frames.size() == (text.find("begin") == std::string::npos ? 0 : 1));
    }
    const auto text = frame();
    // Exact line cuts and partial-token cuts must all reject before the full end.
    for (size_t end = 1; end < text.size() - 1; ++end) {
        GPIO io;
        NonseekText buffer(text.substr(0, end));
        std::istream input(&buffer);
        input.exceptions(mask);
        std::vector<Atoms> frames;
        bool success = false;
        try { success = io.read(frames, input, "n2p2"); } catch (const std::exception &) {}
        REQUIRE_FALSE(success);
        REQUIRE(frames.empty());
    }
}

TEST_CASE("N2P2 rejects malformed labels columns order duplicates and unknown species", "[gpio][n2p2]") {
    for (const auto &text : {"begin extra\n", "comment outside\n", "end\n", "beginning\n",
                            "begin\nlattice\n", "begin\nlattice 1 2\n", "begin\nlattice 1 2 3 extra\n"})
        reject_n2p2(text);
    for (const auto &row : {"", "atom", "atom 1 2 3 H 0 0 1 2", "other 1 2 3 H 0 0 1 2 3",
                           "atom 1 2 3 Unknown 0 0 1 2 3", "atom 1 2 3 H 0 0 1 2 3 extra",
                           "energy", "energy 1 extra", "charge", "charge 0 extra", "end extra",
                           "energyx 1", "begin", "lattice 1 2 3"}) {
        if (std::string(row).empty()) continue;
        reject_n2p2(frame(std::string(row) + "\n"));
    }
    reject_n2p2("begin\n" + atom_row + lattice + "end\n");
    reject_n2p2("begin\nenergy 0\n" + lattice + "end\n");
    reject_n2p2(frame("energy 0\n" + atom_row));
    reject_n2p2(frame("charge 0\n" + atom_row));
    reject_n2p2(frame("energy 0\nenergy 1\n"));
    reject_n2p2(frame("charge 0\ncharge 1\n"));
    reject_n2p2(frame(atom_row + "\nunknown\n"));
    reject_n2p2(frame("atom " + std::string(400, ' ') + std::string(600, '1') + "\n"));
    std::string many = "atom";
    for (int i = 0; i < 250; ++i) many += " 0";
    reject_n2p2(frame(many + "\n"));
    reject_n2p2(frame("energy " + std::string(12000, '1') + "\n"));
}

TEST_CASE("N2P2 validates every numeric column and conversion overflow", "[gpio][n2p2]") {
    for (const auto &number : {"junk", "1junk", "nan", "inf", "-inf", "1e999", "+-1", "++1"}) {
        for (int field : {1, 2, 3, 5, 6, 7, 8, 9}) {
            std::vector<std::string> tokens = {"atom", "1", "2", "3", "H", "0", "0", "1", "2", "3"};
            tokens[field] = number;
            std::string row;
            for (const auto &token : tokens) row += token + " ";
            reject_n2p2(frame(row + "\n"));
        }
        reject_n2p2(frame(std::string("energy ") + number + "\n"));
        reject_n2p2(frame(std::string("charge ") + number + "\n"));
        reject_n2p2(std::string("begin\nlattice ") + number + " 0 0\n" + lattice + "end\n");
    }
    reject_n2p2(frame("energy 1e308\n"));
    reject_n2p2(frame("atom 1 2 3 H 0 1e308 0 0 0\n"));
    reject_n2p2(frame("atom 1 2 3 H 0 0 1e308 0 0\n"));
}

TEST_CASE("N2P2 rejects I O failures and never confuses them with clean EOF", "[gpio][n2p2]") {
    const auto mask = GENERATE(std::ios::goodbit, std::ios::eofbit, std::ios::failbit,
        std::ios::badbit, std::ios::eofbit | std::ios::failbit | std::ios::badbit);
    const auto text = frame();
    for (const size_t offset : {size_t(0), size_t(1), size_t(6), text.size() - 2, text.size()}) {
        GPIO io;
        NonseekText buffer(text);
        buffer.fail_at = offset;
        std::istream input(&buffer);
        input.exceptions(mask);
        std::vector<Atoms> frames;
        bool success = false;
        try { success = io.read(frames, input, "n2p2"); } catch (const std::exception &) {}
        REQUIRE_FALSE(success);
        REQUIRE(frames.size() == (offset == text.size() ? 1 : 0));
    }
    for (auto state : {std::ios::badbit, std::ios::failbit, std::ios::eofbit | std::ios::failbit}) {
        GPIO io;
        std::istringstream input("");
        input.setstate(state);
        std::vector<Atoms> frames;
        REQUIRE_FALSE(io.read(frames, input, "n2p2"));
    }
}

TEST_CASE("N2P2 keeps custom mapping rejects absolute remapping and clears filename failures", "[gpio][n2p2]") {
    GPIO io;
    std::vector<std::string> names(121, "Unused");
    names.back() = "H";
    io.setmap(names);
    Atoms atom;
    std::istringstream good(frame());
    REQUIRE(io.read_one(atom, good, "n2p2"));
    REQUIRE(atom.chemmap_ == std::vector<int>{120});
    io.absolute_index_read = true;
    std::istringstream absolute(frame());
    REQUIRE_FALSE(io.read_one(atom, absolute, "n2p2"));
    REQUIRE(atom.size() == 0);
    io.absolute_index_read = false;
    for (int attempt = 0; attempt < 2; ++attempt) {
        std::istringstream unknown(frame("atom 1 2 3 Alien 0 0 0 0 0\n"));
        REQUIRE_FALSE(io.read_one(atom, unknown, "n2p2"));
    }
    atom.resize(1);
    REQUIRE_FALSE(io.read_one(atom, "/nonexistent/imrmlp-n2p2-input", "n2p2"));
    REQUIRE(atom.size() == 0);
    std::istringstream empty("");
    REQUIRE_FALSE(io.read_one(atom, empty, "n2p2"));
    REQUIRE(atom.size() == 0);
}

TEST_CASE("N2P2 ranges stop at skipped end frames and carry IDs across files", "[gpio][n2p2]") {
    GPIO io;
    io.read_irate = 2;
    io.read_endto = 1;
    std::istringstream input(frame() + frame() + "BAD TAIL\n");
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "n2p2"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].id == 0);
    REQUIRE(input.peek() == 'B');
    REQUIRE(io.read(frames, input, "n2p2"));
    REQUIRE(frames.size() == 1);
    REQUIRE(input.peek() == 'B');
    GPIO bad;
    bad.read_irate = 2;
    bad.read_endto = 1;
    std::istringstream bad_input(frame() + "BAD TAIL\n");
    frames.clear();
    REQUIRE_FALSE(bad.read(frames, bad_input, "n2p2"));
    REQUIRE(frames.size() == 1);

    struct TemporaryDirectory {
        std::filesystem::path path;
        TemporaryDirectory() {
            auto name = (std::filesystem::temp_directory_path() / "imrmlp-n2p2-XXXXXX").string();
            const auto made = mkdtemp(name.data());
            if (!made) throw std::runtime_error("failed to create temporary directory");
            path = made;
        }
        ~TemporaryDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
    } directory;
    const auto first = (directory.path / "first.n2p2").string();
    const auto second = (directory.path / "second.n2p2").string();
    { std::ofstream output(first); output << frame(); }
    { std::ofstream output(second); output << frame() << frame() << "BAD TAIL\n"; }
    GPIO files;
    files.reset_status_every_read = false;
    files.read_startfrom = 1;
    files.read_endto = 2;
    files.read_irate = 2;
    frames.clear();
    REQUIRE(files.read(frames, first, "n2p2"));
    REQUIRE(files.read(frames, second, "n2p2"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].id == 1);
    REQUIRE(files.read(frames, (directory.path / "not-opened.n2p2").string(), "n2p2"));
    REQUIRE(frames.size() == 1);
}

TEST_CASE("N2P2 validates ranges before input and preserves a caller supplied prefix", "[gpio][n2p2]") {
    for (const auto &range : {std::array<int, 3>{-1, 2, 1}, {0, -1, 1}, {2, 1, 1},
                              {0, INT_MAX, 0}, {0, INT_MAX, -1}}) {
        GPIO io;
        io.read_startfrom = range[0];
        io.read_endto = range[1];
        io.read_irate = range[2];
        std::istringstream input(frame());
        std::vector<Atoms> frames(1);
        frames[0].features_["caller"] = "prefix";
        REQUIRE_FALSE(io.read(frames, input, "n2p2"));
        REQUIRE(input.peek() == 'b');
        REQUIRE(frames.size() == 1);
        REQUIRE(frames[0].features_.at("caller") == "prefix");
    }
    GPIO io;
    io.read_startfrom = INT_MAX;
    io.read_endto = INT_MAX;
    std::istringstream input(frame());
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "n2p2"));
    REQUIRE(frames.empty());
}


TEST_CASE("N2P2 sequential input retains labels IDs and complete empty frames", "[gpio][n2p2][sequence]") {
    GPIO io;
    io.setmap({"H", "O"});
    auto text = frame() + frame("") + frame();
    text.pop_back(); // A complete final end record does not require a newline.
    NonseekText buffer(text);
    std::istream input(&buffer);
    REQUIRE(io.seq_read_init(input, "n2p2"));
    Atoms atom;
    for (int i = 0; i < 3; ++i) {
        REQUIRE(io.seq_read(input, atom));
        REQUIRE(atom.id == i);
        REQUIRE(atom.size() == (i == 1 ? 0 : 1));
        if (i == 1) {
            REQUIRE(atom.charges_.empty());
            REQUIRE(atom.site_energies_.empty());
            REQUIRE_FALSE(atom.global_attribs_d_.contains("total_charge"));
        } else {
            REQUIRE(atom.chemmap_ == std::vector<int>{0});
            REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3) * BOHR_TO_A));
            REQUIRE(atom.charges_ == std::vector<double>{.25});
            REQUIRE(atom.site_energies_ == std::vector<double>{-.5 * HARTREE_TO_EV});
            REQUIRE(atom.global_attribs_d_.at("total_charge") == 0);
            REQUIRE(atom.energy_ == Catch::Approx(-.25 * HARTREE_TO_EV));
        }
    }
    REQUIRE_FALSE(io.seq_read(input, atom));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
}

TEST_CASE("N2P2 sequential input reports broken tails and can restart", "[gpio][n2p2][sequence]") {
    const bool transport_error = GENERATE(false, true);
    GPIO io;
    NonseekText buffer(frame() + "begin\n" + lattice + atom_row);
    if (transport_error) buffer.fail_at = frame().size() + 2;
    std::istream input(&buffer);
    input.exceptions(std::ios::badbit);
    REQUIRE(io.seq_read_init(input, "n2p2"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    REQUIRE_THROWS(io.seq_read(input, atom));
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.charges_.empty());
    REQUIRE(atom.site_energies_.empty());
    REQUIRE_FALSE(io.seq_read(input, atom));
    std::istringstream fresh(frame());
    REQUIRE(io.seq_read_init(fresh, "n2p2"));
    REQUIRE(io.seq_read(fresh, atom));
    REQUIRE(atom.id == 0);
    REQUIRE_FALSE(io.seq_read(fresh, atom));
}
