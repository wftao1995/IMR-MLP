#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <limits>
#include <sstream>

namespace {
Atoms frame() {
    Atoms a;
    a.cell_ = Eigen::Matrix3d::Identity() * 4;
    a.pos_ = {{1, 2, 3}};
    a.chemmap_ = {1};
    a.forces_ = {{54.422793282616, -27.211396641308, 0}};
    return a;
}
}

TEST_CASE("AXSF vector writer emits independent count index and force-unit oracle", "[gpio][xsf][animation][write]") {
    GPIO io;
    io.setmap({"O", "C", "H"});
    auto a = frame(); a.chemmap_[0] = 2;
    auto b = a;
    b.pos_[0] = Eigen::RowVector3d(2, 3, 4); b.forces_.clear();
    std::vector<Atoms> frames{a, b};
    std::ostringstream output;
    REQUIRE(io.write(frames, output, "axsf"));
    std::istringstream text(output.str());
    std::string tag, symbol;
    int number;
    text >> tag >> number; REQUIRE(tag == "ANIMSTEPS"); REQUIRE(number == 2);
    text >> tag; REQUIRE(tag == "CRYSTAL"); text >> tag; REQUIRE(tag == "PRIMVEC");
    for (int i = 0; i < 3; ++i) {
        double x, y, z; text >> x >> y >> z;
        REQUIRE(x == (i == 0 ? 4 : 0)); REQUIRE(y == (i == 1 ? 4 : 0)); REQUIRE(z == (i == 2 ? 4 : 0));
    }
    text >> tag >> number; REQUIRE(tag == "PRIMCOORD"); REQUIRE(number == 1);
    int count, sets; text >> count >> sets; REQUIRE(count == 1); REQUIRE(sets == 1);
    double x, y, z, fx, fy, fz; text >> symbol >> x >> y >> z >> fx >> fy >> fz;
    REQUIRE(symbol == "H"); REQUIRE(x == 1); REQUIRE(y == 2); REQUIRE(z == 3);
    REQUIRE(fx == Catch::Approx(2)); REQUIRE(fy == Catch::Approx(-1)); REQUIRE(fz == 0);
    text >> tag >> number; REQUIRE(tag == "PRIMCOORD"); REQUIRE(number == 2);
    REQUIRE(frames[0].pos_ == a.pos_); REQUIRE(frames[0].chemmap_ == a.chemmap_);
    REQUIRE(frames[0].forces_ == a.forces_);
}

TEST_CASE("AXSF variable cells fractional inputs and molecular padding remain explicit", "[gpio][xsf][animation][write]") {
    GPIO io;
    auto a = frame(), b = a;
    a.cartesian = false; a.pos_[0] = Eigen::RowVector3d(.25, .5, .75);
    b.cell_ *= 2;
    std::vector<Atoms> frames{a, b};
    std::ostringstream output; REQUIRE(io.write(frames, output, "axsf"));
    REQUIRE(output.str().find("PRIMVEC 1\n4 0 0") != std::string::npos);
    REQUIRE(output.str().find("PRIMVEC 2\n8 0 0") != std::string::npos);
    REQUIRE(output.str().find("H 1 2 3 2 -1 0") != std::string::npos);
    REQUIRE_FALSE(frames[0].cartesian); REQUIRE(frames[0].pos_ == a.pos_);
    for (auto &f : frames) { f.pbc_ = {false, false, false}; f.cartesian = true; }
    output.str(""); output.clear(); REQUIRE(io.write(frames, output, "axsf"));
    REQUIRE(output.str().find("PRIMVEC") == std::string::npos);
    REQUIRE(output.str().find("ATOMS 1\n") != std::string::npos);
    REQUIRE(output.str().find("ATOMS 2\n") != std::string::npos);
    REQUIRE(frames[1].cell_(0, 0) == 8);
}

TEST_CASE("AXSF validates all frames before writing any animation bytes", "[gpio][xsf][animation][write]") {
    GPIO io;
    std::vector<Atoms> frames{frame(), frame()};
    SECTION("changed atom count") { frames[1].pos_.clear(); frames[1].chemmap_.clear(); frames[1].forces_.clear(); }
    SECTION("changed PBC") { frames[1].pbc_ = {false, false, false}; }
    SECTION("unsupported axes") { for (auto &a : frames) a.pbc_ = {false, true, false}; }
    SECTION("unknown mapping") { frames[1].chemmap_[0] = -1; }
    SECTION("incomplete forces") { frames[1].forces_.push_back(Eigen::Vector3d::Zero()); }
    SECTION("nonfinite position") { frames[1].pos_[0][0] = std::numeric_limits<double>::quiet_NaN(); }
    SECTION("nonfinite force") { frames[1].forces_[0][0] = std::numeric_limits<double>::infinity(); }
    SECTION("full Hessian") { frames[1].have_hessian = true; }
    SECTION("HVP") { frames[1].hvp_directions_ = Eigen::MatrixXd::Ones(3, 1); frames[1].hvp_targets_ = Eigen::MatrixXd::Ones(3, 1); }
    std::ostringstream output; output << "keep";
    REQUIRE_FALSE(io.write(frames, output, "axsf")); REQUIRE(output.str() == "keep");
}

TEST_CASE("AXSF selected frame count is known and empty selection is a no-op", "[gpio][xsf][animation][write]") {
    GPIO io; io.setoutrate(2, 0, 2);
    std::vector<Atoms> frames{frame(), frame(), frame()};
    frames[1].pos_[0][0] = 7; frames[2].pos_[0][0] = 9;
    std::ostringstream output; REQUIRE(io.write(frames, output, "axsf"));
    REQUIRE(output.str().starts_with("ANIMSTEPS 2\n"));
    REQUIRE(output.str().find("H 7 2 3") == std::string::npos);
    REQUIRE(output.str().find("H 9 2 3") != std::string::npos);
    io.setoutrate(1, 5, 6); output.str("keep");
    REQUIRE(io.write(frames, output, "axsf")); REQUIRE(output.str() == "keep");
    io.setoutrate(1, 0, 2); io.append = true;
    REQUIRE_FALSE(io.write(frames, output, "axsf")); REQUIRE(output.str() == "keep");
    io.append = false; output.setstate(std::ios::badbit);
    REQUIRE_FALSE(io.write(frames, output, "axsf"));
}
