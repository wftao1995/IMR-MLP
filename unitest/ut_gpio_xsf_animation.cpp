#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <sstream>

namespace {
const std::string variable =
    "ANIMSTEPS 2\nSLAB\nPRIMVEC 1\n4 0 0\n.5 5 0\n0 0 8\n"
    "CONVVEC 1\n8 0 0\n0 10 0\n0 0 16\nPRIMCOORD 1\n2 1\n"
    "H .25 .5 .75 .1 -.2 .3\n8 1 2 3 -.4 .5 -.6\n"
    "PRIMVEC 2\n6 0 .2\n.4 7 0\n0 0 9\nPRIMCOORD 2\n2 1\n"
    "1 .5 .75 1.25 0 0 0\nO 2 3 4 .2 .3 .4\n";
const std::string fixed =
    "ANIMSTEPS 2\nCRYSTAL\nPRIMVEC\n4 0 0\n0 5 0\n0 0 6\n"
    "PRIMCOORD 1\n1 1\nH 0 0 0\nPRIMCOORD 2\n1 1\nH 1 2 3\n";
}

TEST_CASE("XSF animation uses indexed primitive cells and Hartree per Angstrom forces", "[gpio][xsf][animation]") {
    GPIO io;
    std::vector<Atoms> frames;
    std::istringstream input(variable);
    REQUIRE(io.read(frames, input, "axsf"));
    REQUIRE(frames.size() == 2);
    REQUIRE(frames[0].id == 0); REQUIRE(frames[1].id == 1);
    REQUIRE(frames[0].pbc_ == std::array<bool, 3>{true, true, false});
    REQUIRE(frames[1].pbc_ == frames[0].pbc_);
    REQUIRE(frames[0].cell_(0, 0) == 4); REQUIRE(frames[0].cell_(1, 0) == .5);
    REQUIRE(frames[1].cell_(0, 2) == .2); REQUIRE(frames[1].cell_(1, 1) == 7);
    REQUIRE(frames[0].cell_(2, 2) == 8); // CONVVEC never replaces PRIMVEC.
    REQUIRE(frames[1].pos_[0].isApprox(Eigen::RowVector3d(.5, .75, 1.25)));
    REQUIRE(frames[0].forces_[0][0] == Catch::Approx(2.7211396641308));
    REQUIRE(frames[0].forces_[0][1] == Catch::Approx(-5.4422793282616));
    REQUIRE(frames[1].forces_[1][2] == Catch::Approx(10.8845586565232));
    frames[1].forces_[0].setOnes(); REQUIRE(frames[0].forces_[0][0] == Catch::Approx(2.7211396641308));
}

TEST_CASE("XSF fixed-cell and molecular animations retain exact frame geometry", "[gpio][xsf][animation]") {
    GPIO io;
    std::vector<Atoms> frames;
    std::istringstream input(fixed);
    REQUIRE(io.read(frames, input, "xsf")); REQUIRE(frames.size() == 2);
    REQUIRE(frames[0].cell_ == frames[1].cell_); REQUIRE(frames[0].is_fully_periodic());
    REQUIRE(frames[1].pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3)));
    std::istringstream molecule("ANIMSTEPS 2\nATOMS 1\nH 0 0 0 .1 0 0\nO 1 0 0 0 .2 0\nATOMS 2\n1 0 0 .5\n8 1 0 .5\n");
    frames.clear(); REQUIRE(io.read(frames, molecule, "xsf")); REQUIRE(frames.size() == 2);
    REQUIRE(frames[0].is_nonperiodic()); REQUIRE(frames[1].cell_.isZero());
    REQUIRE(frames[1].pos_[1][2] == .5); REQUIRE(frames[1].forces_.empty());
    REQUIRE(frames[0].forces_[1][1] == Catch::Approx(5.4422793282616));
    std::istringstream empty("ANIMSTEPS 2\nATOMS 1\nATOMS 2\n");
    frames.clear(); REQUIRE(io.read(frames, empty, "xsf")); REQUIRE(frames.size() == 2);
    REQUIRE(frames[0].size() == 0); REQUIRE(frames[1].size() == 0);
}

TEST_CASE("XSF animation rejects wrong counts order incomplete frames and grids", "[gpio][xsf][animation]") {
    for (const auto &text : {
        std::string("ANIMSTEPS 0\nATOMS 1\n"), std::string("ANIMSTEPS 2\nATOMS 1\nH 0 0 0\n"),
        std::string("ANIMSTEPS 1\nATOMS 1\nH 0 0 0\nATOMS 2\nH 1 0 0\n"),
        std::string("ANIMSTEPS 2\nATOMS 1\nH 0 0 0\nATOMS 3\nH 1 0 0\n"),
        std::string("ANIMSTEPS 2\nATOMS 1\nH 0 0 0\nATOMS 2\nH 1 0 0\nO 2 0 0\n"),
        fixed.substr(0, fixed.rfind("H 1 2 3")) + "H 1 2\n",
        fixed + "BEGIN_BLOCK_DATAGRID_3D\n"}) {
        CAPTURE(text);
        GPIO io;
        std::vector<Atoms> frames;
        std::istringstream input(text);
        REQUIRE_FALSE(io.read(frames, input, "xsf"));
    }
    std::string wrong = variable;
    wrong.replace(wrong.find("PRIMVEC 2"), 9, "PRIMVEC 3");
    GPIO io; std::vector<Atoms> frames; std::istringstream input(wrong);
    REQUIRE_FALSE(io.read(frames, input, "xsf")); REQUIRE(frames.size() == 1);
}

TEST_CASE("XSF animation inclusive bounds stop before unread malformed frames", "[gpio][xsf][animation]") {
    GPIO io; io.read_endto = 0;
    const auto start = fixed.find("PRIMCOORD 2");
    std::istringstream input(fixed.substr(0, start) + "BAD UNREAD TAIL\n");
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "xsf")); REQUIRE(frames.size() == 1);
    REQUIRE(input.peek() == 'B');
    GPIO skipped; skipped.read_irate = 2; skipped.read_endto = 1;
    std::istringstream skip_input(fixed + "BAD UNREAD TAIL\n"); frames.clear();
    REQUIRE(skipped.read(frames, skip_input, "xsf")); REQUIRE(frames.size() == 1);
    REQUIRE(skip_input.peek() == 'B');
}
