#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <sstream>

TEST_CASE("CP2KXYZ native sequential output reuses mapped geometry codec frame by frame", "[gpio][cp2kxyz][sequence]") {
    const bool guessed = GENERATE(false, true);
    GPIO writer;
    writer.setmap({"He", "H"});
    std::ostringstream out;
    REQUIRE(writer.seq_write_init(out, guessed ? "" : "cp2kxyz", "output.cp2kxyz"));
    Atoms first;
    first.chemmap_ = {1, 0};
    first.pos_ = {{1.25, -2.5, .375}, {4, 5, 6}};
    first.energy_ = -7; first.have_energy = true;
    Atoms empty;
    Atoms last;
    last.chemmap_ = {0}; last.pos_ = {{1, -3, 2}};
    REQUIRE(writer.seq_write(out, first));
    const auto first_text = out.str();
    REQUIRE(first_text.find("H    1.25 -2.5 0.375") != std::string::npos);
    REQUIRE(first_text.find("He    4 5 6") != std::string::npos);
    REQUIRE(writer.seq_write(out, empty));
    REQUIRE(writer.seq_write(out, last));
    std::istringstream input(out.str());
    GPIO reader;
    reader.setmap({"He", "H"});
    std::vector<Atoms> frames;
    REQUIRE(reader.read(frames, input, "cp2kxyz"));
    REQUIRE(frames.size() == 3);
    REQUIRE(frames[0].chemmap_ == first.chemmap_);
    REQUIRE(frames[0].pos_ == first.pos_);
    REQUIRE(frames[1].size() == 0);
    REQUIRE(frames[2].chemmap_ == last.chemmap_);
    REQUIRE(frames[2].pos_ == last.pos_);
    for (const auto &frame : frames) {
        REQUIRE(frame.is_nonperiodic());
        REQUIRE(frame.cell_.isZero());
        REQUIRE_FALSE(frame.have_energy);
    }
    REQUIRE(first.have_energy);
    REQUIRE(first.energy_ == -7);
    REQUIRE(first.chemmap_ == std::vector<int>{1, 0});
}

TEST_CASE("CP2KXYZ sequential output retains common dispatch failures", "[gpio][cp2kxyz][sequence]") {
    GPIO writer;
    Atoms atom;
    atom.pos_ = {{1, 2, 3}}; atom.chemmap_ = {1};
    std::ostringstream out;
    REQUIRE(writer.seq_write_init(out, "cp2kxyz"));
    SECTION("stream failure") {
        out.setstate(std::ios::badbit);
        REQUIRE_FALSE(writer.seq_write(out, atom));
    }
    SECTION("paired HVP cannot be silently lost") {
        atom.hvp_directions_ = Eigen::MatrixXd::Ones(3, 1);
        atom.hvp_targets_ = Eigen::MatrixXd::Zero(3, 1);
        REQUIRE_THROWS_WITH(writer.seq_write(out, atom), Catch::Matchers::ContainsSubstring("HVP labels"));
        REQUIRE(out.str().empty());
    }
    SECTION("reserved HVP keys cannot bypass common checks") {
        atom.features_["hvp_directions"] = "reserved";
        REQUIRE_THROWS_WITH(writer.seq_write(out, atom), Catch::Matchers::ContainsSubstring("reserved"));
        REQUIRE(out.str().empty());
    }
}
