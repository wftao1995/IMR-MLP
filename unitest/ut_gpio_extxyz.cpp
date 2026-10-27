#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "io/gpio_binary.h"
#include "utils/compressed_stream.h"
#include <atomic>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

namespace {
const std::string lattice = "Lattice=\"5 0 0 0 6 0 0 0 7\" ";
const std::string properties = "Properties=species:S:1:pos:R:3";
std::string frame(const std::string &header = properties, const std::string &row = "H 1 2 3\n") {
    return "1\n" + header + "\n" + row;
}
Atoms sentinel() {
    Atoms atom;
    atom.cell_ = 8 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{8, 7, 6}, {5, 4, 3}};
    atom.chemmap_ = {0, 0};
    atom.energy_ = 123;
    atom.have_energy = true;
    atom.features_["sentinel"] = "keep";
    return atom;
}
void unchanged(const Atoms &atom) {
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.pos_[0][0] == 8);
    REQUIRE(atom.energy_ == 123);
    REQUIRE(atom.features_.at("sentinel") == "keep");
}
struct Directory {
    std::filesystem::path path;
    Directory() {
        static std::atomic<unsigned> count{0};
        path = std::filesystem::temp_directory_path() / ("imrmlp-extxyz-" + std::to_string(getpid()) + "-" + std::to_string(count++));
        std::filesystem::create_directory(path);
    }
    ~Directory() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};
struct BadRead : std::streambuf {
    int_type underflow() override { throw std::runtime_error("injected read failure"); }
};
}

TEST_CASE("EXTXYZ accepts whitespace EOF and an unterminated final atom line", "[extxyz_strict][eof]") {
    const auto trailer = GENERATE(std::string(""), std::string("\n"), std::string("\r\n \t\r\n"));
    GPIO io;
    std::istringstream input("\n \t\r\n" + frame(lattice + properties, "H +1 2 3") + trailer);
    Atoms atom;
    REQUIRE(io.seq_read_init(input, "extxyz"));
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.size() == 1);
    REQUIRE(atom.pos_[0][0] == 1);
    REQUIRE_FALSE(io.seq_read(input, atom));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
}

TEST_CASE("EXTXYZ handles EOF with caller enabled stream exceptions", "[extxyz_strict][eof]") {
    GPIO io;
    std::istringstream input(frame(properties, "H 1 2 3"));
    input.exceptions(std::ios::badbit | std::ios::failbit | std::ios::eofbit);
    REQUIRE(io.seq_read_init(input, "extxyz"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    REQUIRE_FALSE(io.seq_read(input, atom));
    REQUIRE(input.exceptions() == (std::ios::badbit | std::ios::failbit | std::ios::eofbit));
}

TEST_CASE("EXTXYZ rejects bad frame headers without changing the destination", "[extxyz_strict][errors]") {
    const auto text = GENERATE(
        std::string("-1\n"), std::string("2147483648\n"), std::string("999999999999999999999\n"),
        std::string("1junk\n"), std::string("1.0\n"), std::string("+\n"), std::string("1\n"),
        std::string("1\ncomment only\nH 1 2 3\n"),
        std::to_string(gpio_binary::MAX_ATOMS + 1) + "\n");
    GPIO io;
    auto atom = sentinel();
    std::istringstream input(text);
    REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    unchanged(atom);
}

TEST_CASE("EXTXYZ rejects invalid Properties declarations", "[extxyz_strict][errors]") {
    const auto declaration = GENERATE(
        "species:S:1:pos:R:2", "species:R:1:pos:R:3", "species:S:2:pos:R:3",
        "species:S:1:pos:I:3", "species:S:1:pos:R:3:", "species:S:1:pos:R:3:bad",
        "species:S:1:pos:R:3:foo:R:0", "species:S:1:pos:R:3:foo:R:-2",
        "species:S:1:pos:R:3:foo:R:999999999999", "species:S:1:pos:R:3:foo:R:1junk",
        "species:S:1:pos:R:3:foo:RR:1", "species:S:1:pos:R:3:foo:Q:1",
        "species:S:1:pos:R:3:pos:R:3", "species:S:1:pos:R:3:force:R:3:forces:R:3",
        "species:S:1:pos:R:3:forces:R:2", "species:S:1:pos:R:3:charges:R:2",
        "species:S:1:pos:R:3:eff_charges:R:3", "pos:R:3", "species:S:1");
    GPIO io; auto atom = sentinel();
    std::istringstream input(frame("Properties=" + std::string(declaration)));
    REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    unchanged(atom);
}

TEST_CASE("EXTXYZ validates every numeric atom column and exact row width", "[extxyz_strict][errors]") {
    const auto row = GENERATE("H 1 2\n", "H 1 2 3 extra\n", "H 1 2 nan\n", "H inf 2 3\n",
        "H 1x 2 3\n", "H 1e999 2 3\n", "H --1 2 3\n", "H +-1 2 3\n", "Unknown 1 2 3\n", "\n", "");
    GPIO io; auto atom = sentinel();
    std::istringstream input(frame(properties, row));
    REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    unchanged(atom);
}

TEST_CASE("EXTXYZ rejects malformed numeric metadata and unterminated quotes", "[extxyz_strict][errors]") {
    const auto metadata = GENERATE("energy=nan", "energy=1e+", "fenergy=2junk", "energy=\"\"",
        "Lattice=\"1 0 0\"", "lattice=\"1 0 0 0 1 0 0 0 inf\"", "Lattice=\"1 0 0 0 1 0 0 0 1junk\"",
        "stress=\"1 2 3 4 5\"", "virial=\"1 2 3 4 5 6 7\"", "virial=\"1 2 3 4 5 nan\"",
        "label=\"unterminated", "label=\"x\"trailing", "Properties=species:S:1:pos:R:3");
    GPIO io; auto atom = sentinel();
    std::istringstream input(frame(properties + " " + metadata));
    REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    unchanged(atom);
}

TEST_CASE("EXTXYZ batch rejects broken suffixes instead of returning a successful prefix", "[extxyz_strict][batch]") {
    const auto tail = GENERATE(std::string("junk"), std::string("1"), std::string("1\n"),
        frame(properties, "H 1 2"), frame("pbc=T " + properties), frame(properties, "H nan 2 3\n"));
    GPIO io;
    std::istringstream input(frame(lattice + properties) + tail);
    std::vector<Atoms> atoms;
    REQUIRE_FALSE(io.read(atoms, input, "extxyz"));
    REQUIRE(atoms.size() == 1);
    REQUIRE(atoms[0].pos_[0][2] == 3);
}

TEST_CASE("EXTXYZ sequence rejects partial frames and requires reinitialization", "[extxyz_strict][sequence]") {
    GPIO io;
    std::istringstream input(frame(lattice + properties) + frame(properties, "H 1 2\n") + frame());
    REQUIRE(io.seq_read_init(input, "extxyz"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    REQUIRE_THROWS_WITH(io.seq_read(input, atom), Catch::Matchers::ContainsSubstring("EXTXYZ frame 1"));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
    REQUIRE(io.seq_read_init(input, "extxyz"));
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.id == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
}

TEST_CASE("EXTXYZ propagates read failures with and without stream exceptions", "[extxyz_strict][sequence]") {
    const bool exceptions = GENERATE(false, true);
    BadRead buffer;
    std::istream input(&buffer);
    if (exceptions) input.exceptions(std::ios::badbit);
    GPIO io;
    REQUIRE(io.seq_read_init(input, "extxyz"));
    Atoms atom = sentinel();
    REQUIRE_THROWS(io.seq_read(input, atom));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
    BadRead batch_buffer;
    std::istream batch_input(&batch_buffer);
    std::vector<Atoms> atoms;
    REQUIRE_FALSE(io.read(atoms, batch_input, "extxyz"));
    REQUIRE(atoms.empty());
}

TEST_CASE("EXTXYZ preserves no-Lattice bounding boxes and accepts complete empty frames", "[extxyz_strict][geometry]") {
    GPIO io;
    io.r_extent = 2;
    std::istringstream input("2\n" + properties + " pbc=\"F F F\"\nH -1 2 3\nH 2 4 7\n");
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.cell_.diagonal().isApprox(Eigen::Vector3d(7, 6, 8)));
    REQUIRE(atom.pos_[0][0] == -1);
    REQUIRE(atom.pbc_ == std::array<bool, 3>{false, false, false});
    std::istringstream empty("0\n" + properties + "\n");
    REQUIRE(io.read_one(atom, empty, "extxyz"));
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.cell_.isApprox(4 * Eigen::Matrix3d::Identity()));
}

TEST_CASE("EXTXYZ preserves duplicate pbc compatibility and aliases", "[extxyz_strict][metadata]") {
    GPIO io;
    Atoms atom;
    std::istringstream valid(frame("lattice=\"5 0 0 0 6 0 0 0 7\" pbc=\"T F T\" pbc=T T T "
        "energy=+2.5 fenergy=-1 label=\"two words\" config_type=sample " + properties + ":force:R:3", "H 1 2 3 4 5 6\n"));
    REQUIRE(io.read_one(atom, valid, "extxyz"));
    REQUIRE(atom.pbc_ == std::array<bool, 3>{true, false, true});
    REQUIRE(atom.forces_[0].isApprox(Eigen::Vector3d(4, 5, 6)));
    REQUIRE(atom.energy_ == 2.5);
    REQUIRE(atom.fenergy_ == -1);
    REQUIRE(atom.features_.at("label") == "two words");
    std::istringstream words(frame("pbc=\"True FALSE true\" " + properties));
    REQUIRE(io.read_one(atom, words, "extxyz"));
    REQUIRE(atom.pbc_ == std::array<bool, 3>{true, false, true});
    atom.energy_ = 2.5;
    std::istringstream invalid(frame("pbc=T " + properties));
    REQUIRE_FALSE(io.read_one(atom, invalid, "extxyz"));
    REQUIRE(atom.energy_ == 2.5);
}

TEST_CASE("EXTXYZ preserves typed custom properties and untyped metadata", "[extxyz_strict][metadata]") {
    GPIO io;
    Atoms atom;
    std::istringstream input(frame(lattice + "temperature=300 custom=\"some text\" vector=\"1 2 3\" " + properties +
        ":scalar:R:1:ivec:I:2:rvec:R:2:tag:S:2:velocities:R:3:site_en:R:1:charges:R:1:eff_charges:R:9",
        "H 1 2 3 +4.5 +2 -3 1e-3 2e-3 \"quoted text\" second 4 5 6 7 8 1 2 3 4 5 6 7 8 9\n"));
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.global_attribs_d_.at("temperature") == 300);
    REQUIRE(atom.features_.at("custom") == "some text");
    REQUIRE(atom.features_.at("vector") == "1 2 3");
    REQUIRE(atom.local_attribs_d_.at("scalar")[0] == 4.5);
    REQUIRE(atom.local_attribs_vi_.at("ivec")(1, 0) == -3);
    REQUIRE(atom.local_attribs_v_.at("rvec")(1, 0) == .002);
    REQUIRE(atom.velocities_[0][2] == 6);
    REQUIRE(atom.site_energies_[0] == 7);
    REQUIRE(atom.charges_[0] == 8);
    REQUIRE(atom.born_eff_chgs_[0](2, 2) == 9);
    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    Atoms reread;
    REQUIRE(io.read_one(reread, output, "extxyz"));
    REQUIRE(reread.local_attribs_vi_.at("ivec") == atom.local_attribs_vi_.at("ivec"));
    REQUIRE(reread.local_attribs_v_.at("rvec") == atom.local_attribs_v_.at("rvec"));
}

TEST_CASE("EXTXYZ validates custom integer and real columns", "[extxyz_strict][errors]") {
    const auto extra = GENERATE(std::string("bad:R:1"), std::string("bad:I:1"), std::string("bad:R:2"), std::string("bad:I:2"));
    GPIO io; auto atom = sentinel();
    std::istringstream input(frame(properties + ":" + extra, "H 1 2 3 invalid" + (extra.back() == '2' ? std::string(" 1") : std::string()) + "\n"));
    REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    unchanged(atom);
}

TEST_CASE("EXTXYZ reads native and ASE per-atom energy names", "[extxyz_strict][energies]") {
    const auto name = GENERATE(std::string("site_en"), std::string("energies"));
    GPIO io;
    Atoms atom;
    std::istringstream input("2\n" + lattice + properties + ":" + name + ":R:1 energy=3\n"
                             "H 1 2 3 1\nH 2 3 4 2\n");
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.site_energies_ == std::vector<double>{1, 2});
    REQUIRE(atom.have_energy);
    REQUIRE(atom.energy_ == 3);
    REQUIRE_FALSE(atom.local_attribs_d_.count(name));

    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    REQUIRE(output.str().find(":site_en:R:1") != std::string::npos);
    Atoms reread;
    REQUIRE(io.read_one(reread, output, "extxyz"));
    REQUIRE(reread.site_energies_ == atom.site_energies_);
}

TEST_CASE("EXTXYZ rejects conflicting or nonscalar per-atom energies", "[extxyz_strict][energies]") {
    const auto declaration = GENERATE(
        ":site_en:R:1:energies:R:1", ":energies:R:1:site_en:R:1",
        ":energies:R:2", ":energies:I:1");
    GPIO io;
    auto atom = sentinel();
    const auto row = std::string(declaration) == ":energies:I:1" ? "H 1 2 3 1\n" : "H 1 2 3 1 2\n";
    std::istringstream input(frame(properties + declaration, row));
    REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    unchanged(atom);
}

TEST_CASE("EXTXYZ reads native and ASE per-atom charge names", "[extxyz_strict][charges]") {
    const auto name = GENERATE(std::string("charge"), std::string("charges"));
    GPIO io;
    Atoms atom;
    std::istringstream input("2\n" + lattice + properties + ":" + name + ":R:1\n"
                             "H 1 2 3 .25\nH 2 3 4 -.1\n");
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.charges_ == std::vector<double>{.25, -.1});
    REQUIRE_FALSE(atom.local_attribs_d_.count("charge"));
    REQUIRE_FALSE(atom.local_attribs_d_.count("charges"));
    REQUIRE_FALSE(atom.global_attribs_d_.count("total_charge"));

    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    REQUIRE(output.str().find(":charges:R:1") != std::string::npos);
    Atoms reread;
    REQUIRE(io.read_one(reread, output, "extxyz"));
    REQUIRE(reread.charges_ == atom.charges_);
}

TEST_CASE("EXTXYZ rejects duplicate or nonscalar charge aliases", "[extxyz_strict][charges]") {
    const auto declaration = GENERATE(
        ":charge:R:1:charges:R:1", ":charge:R:2", ":charge:I:1");
    GPIO io;
    auto atom = sentinel();
    const auto row = std::string(declaration) == ":charge:I:1" ? "H 1 2 3 1\n" : "H 1 2 3 1 2\n";
    std::istringstream input(frame(properties + declaration, row));
    REQUIRE_FALSE(io.read_one(atom, input, "extxyz"));
    unchanged(atom);
}

TEST_CASE("EXTXYZ preserves numeric-looking untyped text", "[extxyz_strict][metadata]") {
    GPIO io;
    Atoms atom;
    std::istringstream input(frame(lattice + properties +
        " batch=\"2026-10-18\" identifier=\"1-2\" foo=1e999 status=nan temperature=300"));
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.features_.at("batch") == "2026-10-18");
    REQUIRE(atom.features_.at("identifier") == "1-2");
    REQUIRE(atom.features_.at("foo") == "1e999");
    REQUIRE(atom.features_.at("status") == "nan");
    REQUIRE(atom.global_attribs_d_.at("temperature") == 300);

    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    Atoms reread;
    REQUIRE(io.read_one(reread, output, "extxyz"));
    REQUIRE(reread.features_ == atom.features_);
    REQUIRE(reread.global_attribs_d_.at("temperature") == 300);
}

TEST_CASE("EXTXYZ text metadata survives quoted output", "[extxyz_strict][metadata]") {
    GPIO io;
    Atoms atom;
    std::istringstream input(frame(lattice + properties +
        R"( label="" config_type="liquid water" identifier="1-2" note='it\'s "fine"' path="C:\tmp\\sample" bare=C:\tmp)"));
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.features_.at("label").empty());
    REQUIRE(atom.features_.at("config_type") == "liquid water");
    REQUIRE(atom.features_.at("identifier") == "1-2");
    REQUIRE(atom.features_.at("note") == "it's \"fine\"");
    REQUIRE(atom.features_.at("path") == R"(C:\tmp\sample)");
    REQUIRE(atom.features_.at("bare") == R"(C:\tmp)");

    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    Atoms back;
    REQUIRE(io.read_one(back, output, "extxyz"));
    REQUIRE(back.features_ == atom.features_);
}

TEST_CASE("EXTXYZ retains high order labels and commits none on high order errors", "[extxyz_strict][hvp]") {
    Atoms original = sentinel();
    original.hvp_directions_ = Eigen::MatrixXd::Ones(6, 2);
    original.hvp_targets_ = Eigen::MatrixXd::Constant(6, 2, .4);
    original.hessian_ = Eigen::MatrixXd::Identity(6, 6);
    original.have_hessian = true;
    GPIO io;
    std::stringstream output;
    REQUIRE(io.write_one(original, output, "extxyz"));
    Atoms parsed;
    REQUIRE(io.read_one(parsed, output, "extxyz"));
    REQUIRE(parsed.hvp_directions_.isApprox(original.hvp_directions_));
    REQUIRE(parsed.hvp_targets_.isApprox(original.hvp_targets_));
    REQUIRE(parsed.hessian_.isApprox(original.hessian_));
    const auto bad = GENERATE("hessian=\"1 2 3\"", "hessian=\"nan 0 0 0 1 0 0 0 1\"",
        "hvp_directions=\"1 2 3\"", "hvp_directions=\"1 2 3", "hvp_directions=\"1 2 inf\" hvp_targets=\"1 2 3\"");
    auto kept = sentinel();
    std::istringstream input(frame(properties + " " + bad));
    REQUIRE_THROWS(io.read_one(kept, input, "extxyz"));
    unchanged(kept);
}

TEST_CASE("EXTXYZ reads long property lines across stream buffer boundaries", "[extxyz_strict][large]") {
    constexpr int width = 18000;
    std::string row = "H 1 2 3";
    for (int i = 0; i < width; ++i) row += " 1.125";
    std::istringstream input(frame(properties + ":large:R:" + std::to_string(width), row));
    GPIO io; Atoms atom;
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.local_attribs_v_.at("large").rows() == width);
    REQUIRE(atom.local_attribs_v_.at("large")(width - 1, 0) == 1.125);
}

TEST_CASE("EXTXYZ sequential input checks compressed trailers", "[extxyz_strict][compressed]") {
    Directory directory;
    const auto extension = GENERATE(std::string(".gz"), std::string(".zst"));
#if !defined(USE_GZIP)
    if (extension == ".gz") return;
#endif
#if !defined(USE_ZSTD)
    if (extension == ".zst") return;
#endif
    const auto path = directory.path / ("frames.xyz" + extension);
    auto output = open_ofile(path.string());
    REQUIRE(output);
    *output << frame(lattice + properties) << frame(lattice + properties);
    REQUIRE(close_ofile(output));
    {
        auto input = open_seq_ifile(path.string());
        REQUIRE(input);
        GPIO io; Atoms atom;
        REQUIRE(io.seq_read_init(*input, "extxyz"));
        REQUIRE(io.seq_read(*input, atom));
        REQUIRE(io.seq_read(*input, atom));
        REQUIRE_FALSE(io.seq_read(*input, atom));
    }
    std::filesystem::resize_file(path, std::filesystem::file_size(path) - 1);
    auto input = open_seq_ifile(path.string());
    REQUIRE(input);
    GPIO io; Atoms atom;
    REQUIRE(io.seq_read_init(*input, "extxyz"));
    REQUIRE_THROWS([&] { while (io.seq_read(*input, atom)) {} }());
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(*input, atom));
}

TEST_CASE("EXTXYZ stress and virial use the final cell and retain six component order", "[extxyz_strict][metadata]") {
    GPIO io; Atoms atom;
    std::istringstream virial(frame(lattice + properties + " virial=\"1 2 3 4 5 6\""));
    REQUIRE(io.read_one(atom, virial, "extxyz"));
    Eigen::Matrix3d expected;
    expected << 1,4,5, 4,2,6, 5,6,3;
    REQUIRE(atom.stress_.isApprox(expected));
    std::istringstream stress(frame(lattice + properties + " stress=\"1 2 3 4 5 6\""));
    REQUIRE(io.read_one(atom, stress, "extxyz"));
    REQUIRE(atom.stress_.isApprox(-210 * expected));
    std::istringstream no_cell(frame(properties + " stress=\"1 2 3 4 5 6\""));
    REQUIRE(io.read_one(atom, no_cell, "extxyz"));
    REQUIRE(atom.stress_.isApprox(-atom.get_volume() * expected));
    std::istringstream last_value(frame(lattice + properties + " stress=\"1 2 3 4 5 6\" virial=\"1 2 3 4 5 6\""));
    REQUIRE(io.read_one(atom, last_value, "extxyz"));
    REQUIRE(atom.stress_.isApprox(expected));
}

TEST_CASE("EXTXYZ sequential high order exceptions report the frame and reset the reader", "[extxyz_strict][sequence]") {
    GPIO io; Atoms atom;
    std::istringstream input(frame() + frame(properties + " hessian=\"1 2\""));
    REQUIRE(io.seq_read_init(input, "extxyz"));
    REQUIRE(io.seq_read(input, atom));
    REQUIRE_THROWS_WITH(io.seq_read(input, atom), Catch::Matchers::ContainsSubstring("EXTXYZ frame 1"));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));
}

TEST_CASE("EXTXYZ physical values take precedence over conflicting metadata", "[extxyz_strict][metadata]") {
    const std::string legacy = GENERATE("99", "old label");
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1, 2, 3}};
    atom.chemmap_ = {1};
    atom.energy_ = -1;
    atom.have_energy = true;
    atom.fenergy_ = -1.2;
    atom.have_fenergy = true;
    atom.global_attribs_d_["energy"] = 98;
    atom.global_attribs_d_["fenergy"] = 97;
    atom.global_attribs_d_["total_charge"] = 0.8;
    for (const auto &key : {"energy", "fenergy", "total_charge"}) {
        atom.global_attribs_i_[key] = 96;
        atom.global_attribs_v_[key] = Eigen::VectorXd::Constant(2, 95);
        atom.global_attribs_vi_[key] = Eigen::VectorXi::Constant(2, 94);
        atom.features_[key] = legacy;
    }
    atom.features_["note"] = "ordinary text stays";
    GPIO io;
    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    const auto text = output.str();
    for (const auto &key : {"energy", "fenergy", "total_charge"}) {
        const std::string token = " " + std::string(key) + "=";
        const auto first = text.find(token);
        REQUIRE(first != std::string::npos);
        REQUIRE(text.find(token, first + token.size()) == std::string::npos);
    }
    Atoms restored;
    REQUIRE(io.read_one(restored, output, "extxyz"));
    REQUIRE(restored.has_energy());
    REQUIRE(restored.energy_ == -1);
    REQUIRE(restored.has_fenergy());
    REQUIRE(restored.fenergy_ == Catch::Approx(-1.2));
    REQUIRE(restored.global_attribs_d_.at("total_charge") == Catch::Approx(0.8));
    REQUIRE(restored.features_.at("note") == "ordinary text stays");
    REQUIRE(atom.features_.at("energy") == legacy);
    REQUIRE(atom.global_attribs_d_.at("energy") == 98);
}

TEST_CASE("EXTXYZ retains legacy metadata when no physical value takes precedence", "[extxyz_strict][metadata]") {
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1, 2, 3}};
    atom.chemmap_ = {1};
    atom.features_["energy"] = "99";
    atom.features_["fenergy"] = "88";
    atom.features_["total_charge"] = "old charge text";
    atom.features_["note"] = "ordinary text stays";
    GPIO io;
    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    REQUIRE(output.str().find(" energy=\"99\"") != std::string::npos);
    REQUIRE(output.str().find(" fenergy=\"88\"") != std::string::npos);
    REQUIRE(output.str().find(" total_charge=\"old charge text\"") != std::string::npos);
    Atoms restored;
    REQUIRE(io.read_one(restored, output, "extxyz"));
    REQUIRE(restored.energy_ == 99);
    REQUIRE(restored.fenergy_ == 88);
    REQUIRE(restored.global_attribs_d_.count("total_charge") == 0);
    REQUIRE(restored.features_.at("total_charge") == "old charge text");
    REQUIRE(restored.features_.at("note") == "ordinary text stays");
}

TEST_CASE("CFG energy survives legacy metadata through BIN and EXTXYZ", "[extxyz_strict][metadata]") {
    std::istringstream cfg(
        "BEGIN_CFG\nSize\n1\nSupercell\n10 0 0\n0 10 0\n0 0 10\n"
        "AtomData: id type cartes_x cartes_y cartes_z\n1 1 1 2 3\n"
        "Energy\n-1\nFeature energy old label\nFeature note ordinary text stays\nEND_CFG\n");
    GPIO io;
    Atoms original;
    REQUIRE(io.read_one(original, cfg, "mlpcfg"));
    REQUIRE(original.energy_ == -1);
    REQUIRE(original.features_.at("energy") == "old label");
    std::stringstream binary(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(io.write_one(original, binary, "bin"));
    Atoms from_binary;
    REQUIRE(io.read_one(from_binary, binary, "bin"));
    REQUIRE(from_binary.energy_ == -1);
    REQUIRE(from_binary.features_.at("energy") == "old label");
    std::stringstream extxyz;
    REQUIRE(io.write_one(from_binary, extxyz, "extxyz"));
    Atoms restored;
    REQUIRE(io.read_one(restored, extxyz, "extxyz"));
    REQUIRE(restored.energy_ == -1);
    REQUIRE(restored.features_.count("energy") == 0);
    REQUIRE(restored.features_.at("note") == "ordinary text stays");
    REQUIRE(restored.pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3)));
}

TEST_CASE("EXTXYZ canonical magmoms retain scalar and vector representations", "[extxyz_strict][magmoms]") {
    const bool scalar = GENERATE(false, true);
    const bool lossless = GENERATE(false, true);
    const std::string declaration = scalar ? ":magmoms:R:1" : ":magmoms:R:3";
    const std::string rows = scalar ? "H 1 2 3 -1.5\nH 4 5 6 0\n" :
        "H 1 2 3 .1 -.2 .3\nH 4 5 6 0 0 0\n";
    std::istringstream input("2\n" + lattice + properties + declaration + "\n" + rows);
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.magmom_scalar_.size() == (scalar ? 2 : 0));
    REQUIRE(atom.magmoms_.size() == (scalar ? 0 : 2));
    REQUIRE(atom.local_attribs_d_.count("magmoms") == 0);
    REQUIRE(atom.local_attribs_v_.count("magmoms") == 0);
    if (scalar) {
        REQUIRE(atom.magmom_scalar_ == std::vector<double>{-1.5, 0});
    } else {
        REQUIRE(atom.magmoms_[0].isApprox(Eigen::Vector3d(.1, -.2, .3)));
        REQUIRE(atom.magmoms_[1].isZero());
    }
    // BIN is an existing independent native source for both representations.
    std::stringstream binary(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(io.write_one(atom, binary, "bin"));
    Atoms from_binary;
    REQUIRE(io.read_one(from_binary, binary, "bin"));
    io.lossless = lossless;
    std::stringstream output;
    REQUIRE(io.write_one(from_binary, output, "extxyz"));
    REQUIRE(output.str().find(declaration) != std::string::npos);
    Atoms restored;
    REQUIRE(io.read_one(restored, output, "extxyz"));
    REQUIRE(restored.magmom_scalar_ == atom.magmom_scalar_);
    REQUIRE(restored.magmoms_.size() == atom.magmoms_.size());
    if (!scalar) {
        REQUIRE(restored.magmoms_[0].isApprox(Eigen::Vector3d(.1, -.2, .3)));
        REQUIRE(restored.magmoms_[1].isZero());
    }
}

TEST_CASE("EXTXYZ magnetic representations do not leak between frames", "[extxyz_strict][magmoms]") {
    GPIO io;
    std::stringstream output;
    for (int frame = 0; frame < 3; ++frame) {
        Atoms atom;
        atom.cell_ = 10 * Eigen::Matrix3d::Identity();
        atom.pos_ = {{1, 2, 3}};
        atom.chemmap_ = {1};
        if (frame == 0) atom.magmom_scalar_ = {-1.5};
        if (frame == 2) atom.magmoms_ = {Eigen::Vector3d::Zero()};
        REQUIRE(io.write_one(atom, output, "extxyz"));
    }
    std::vector<Atoms> restored;
    REQUIRE(io.read(restored, output, "extxyz"));
    REQUIRE(restored.size() == 3);
    REQUIRE(restored[0].magmom_scalar_ == std::vector<double>{-1.5});
    REQUIRE(restored[0].magmoms_.empty());
    REQUIRE_FALSE(restored[1].has_magmom());
    REQUIRE(restored[2].magmom_scalar_.empty());
    REQUIRE(restored[2].magmoms_.size() == 1);
    REQUIRE(restored[2].magmoms_[0].isZero());
}

TEST_CASE("EXTXYZ native magmoms take precedence over same-named local metadata", "[extxyz_strict][magmoms]") {
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1, 2, 3}};
    atom.chemmap_ = {1};
    atom.magmoms_ = {{.1, -.2, .3}};
    atom.local_attribs_d_["magmoms"] = Eigen::VectorXd::Constant(1, 9);
    atom.local_attribs_i_["magmoms"] = Eigen::VectorXi::Constant(1, 8);
    atom.local_attribs_v_["magmoms"] = Eigen::MatrixXd::Constant(2, 1, 7);
    atom.local_attribs_vi_["magmoms"] = Eigen::MatrixXi::Constant(2, 1, 6);
    atom.local_attribs_d_["score"] = Eigen::VectorXd::Constant(1, 5);
    GPIO io;
    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    const auto text = output.str();
    const auto first = text.find(":magmoms:");
    REQUIRE(first != std::string::npos);
    REQUIRE(text.find(":magmoms:", first + 1) == std::string::npos);
    Atoms restored;
    REQUIRE(io.read_one(restored, output, "extxyz"));
    REQUIRE(restored.magmoms_.size() == 1);
    REQUIRE(restored.magmoms_[0].isApprox(Eigen::Vector3d(.1, -.2, .3)));
    REQUIRE(restored.local_attribs_d_.at("score")[0] == 5);
    REQUIRE(atom.local_attribs_d_.at("magmoms")[0] == 9);
}

TEST_CASE("EXTXYZ noncanonical magnetic names and shapes remain generic", "[extxyz_strict][magmoms]") {
    const bool integer = GENERATE(false, true);
    const std::string extra = integer ? ":magmoms:I:1" : ":magmoms:R:2";
    const std::string values = integer ? " 4" : " 4 5";
    std::istringstream input(frame(lattice + properties +
        ":magmom:R:3:magmom_scalar:R:1" + extra,
        "H 1 2 3 .1 -.2 .3 -1.5" + values + "\n"));
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE_FALSE(atom.has_magmom());
    REQUIRE(atom.local_attribs_v_.at("magmom")(2, 0) == .3);
    REQUIRE(atom.local_attribs_d_.at("magmom_scalar")[0] == -1.5);
    if (integer) REQUIRE(atom.local_attribs_i_.at("magmoms")[0] == 4);
    else REQUIRE(atom.local_attribs_v_.at("magmoms")(1, 0) == 5);
    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    Atoms restored;
    REQUIRE(io.read_one(restored, output, "extxyz"));
    REQUIRE_FALSE(restored.has_magmom());
    REQUIRE(restored.local_attribs_v_.at("magmom")(2, 0) == .3);
    REQUIRE(restored.local_attribs_d_.at("magmom_scalar")[0] == -1.5);
    if (integer) REQUIRE(restored.local_attribs_i_.at("magmoms")[0] == 4);
    else REQUIRE(restored.local_attribs_v_.at("magmoms")(1, 0) == 5);
}

TEST_CASE("EXTXYZ rejects simultaneous native scalar and vector magnetic moments", "[extxyz_strict][magmoms]") {
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{1, 2, 3}};
    atom.chemmap_ = {1};
    atom.magmom_scalar_ = {1.5};
    atom.magmoms_ = {{.1, -.2, .3}};
    GPIO io;
    std::stringstream binary(std::ios::in | std::ios::out | std::ios::binary);
    REQUIRE(io.write_one(atom, binary, "bin"));
    Atoms from_binary;
    REQUIRE(io.read_one(from_binary, binary, "bin"));
    REQUIRE(from_binary.magmom_scalar_.size() == 1);
    REQUIRE(from_binary.magmoms_.size() == 1);
    std::stringstream output;
    REQUIRE_FALSE(io.write_one(from_binary, output, "extxyz"));
    REQUIRE(output.str().empty());
}

TEST_CASE("EXTXYZ accepts finite free_energy alias with canonical fenergy priority", "[extxyz_strict][metadata]") {
    const std::string energy = GENERATE(
        "free_energy=0", "free_energy=-2 fenergy=-1.2", "fenergy=-1.2 free_energy=-2");
    std::istringstream input(frame(lattice + properties + " energy=-1 " + energy));
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.energy_ == -1);
    REQUIRE(atom.has_fenergy());
    REQUIRE(atom.fenergy_ == Catch::Approx(energy == "free_energy=0" ? 0 : -1.2));
    REQUIRE(atom.global_attribs_d_.count("free_energy") == 0);
    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    REQUIRE(output.str().find(" fenergy=") != std::string::npos);
    REQUIRE(output.str().find(" free_energy=") == std::string::npos);
}

TEST_CASE("EXTXYZ legacy free_energy text remains metadata", "[extxyz_strict][metadata]") {
    std::istringstream input(frame(lattice + properties +
        " fenergy=-1.2 free_energy=\"old free energy\""));
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "extxyz"));
    REQUIRE(atom.fenergy_ == Catch::Approx(-1.2));
    REQUIRE(atom.features_.at("free_energy") == "old free energy");
    std::stringstream output;
    REQUIRE(io.write_one(atom, output, "extxyz"));
    Atoms restored;
    REQUIRE(io.read_one(restored, output, "extxyz"));
    REQUIRE(restored.fenergy_ == Catch::Approx(-1.2));
    REQUIRE(restored.features_.at("free_energy") == "old free energy");
}
