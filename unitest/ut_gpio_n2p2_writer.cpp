#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <limits>
#include <sstream>

namespace {
Atoms training_frame(GPIO &io) {
    Atoms a;
    a.cell_ = Eigen::Matrix3d::Identity() * 1.058354421806; // two Bohr
    a.pos_ = {{0.529177210903, -1.058354421806, 0}};
    a.chemmap_ = {io.getrevmap().at("H")};
    a.forces_ = {{51.422087120633655, -102.84417424126731, 0}};
    a.charges_ = {0.25};
    a.site_energies_ = {54.422793282616}; // two Hartree
    a.have_energy = true;
    a.energy_ = -81.634189923924; // minus three Hartree
    a.global_attribs_d_["total_charge"] = -0.75;
    return a;
}
}

TEST_CASE("N2P2 writer emits independently checked atomic units", "[gpio][n2p2]") {
    GPIO io;
    auto a = training_frame(io);
    const auto source = a;
    std::ostringstream output;
    REQUIRE(io.write_one(a, output, "n2p2"));
    std::istringstream text(output.str());
    std::string tag, symbol;
    text >> tag;
    REQUIRE(tag == "begin");
    for (int i = 0; i < 3; ++i) {
        double x, y, z;
        text >> tag >> x >> y >> z;
        REQUIRE(tag == "lattice");
        REQUIRE(x == Catch::Approx(i == 0 ? 2 : 0));
        REQUIRE(y == Catch::Approx(i == 1 ? 2 : 0));
        REQUIRE(z == Catch::Approx(i == 2 ? 2 : 0));
    }
    double x, y, z, q, e, fx, fy, fz;
    text >> tag >> x >> y >> z >> symbol >> q >> e >> fx >> fy >> fz;
    REQUIRE(tag == "atom"); REQUIRE(symbol == "H");
    REQUIRE(x == Catch::Approx(1)); REQUIRE(y == Catch::Approx(-2)); REQUIRE(z == 0);
    REQUIRE(q == 0.25); REQUIRE(e == Catch::Approx(2));
    REQUIRE(fx == Catch::Approx(1).margin(1e-9));
    REQUIRE(fy == Catch::Approx(-2).margin(2e-9)); REQUIRE(fz == 0);
    text >> tag >> e; REQUIRE(tag == "energy"); REQUIRE(e == Catch::Approx(-3));
    text >> tag >> q; REQUIRE(tag == "charge"); REQUIRE(q == -0.75);
    text >> tag; REQUIRE(tag == "end");
    REQUIRE(a.pos_ == source.pos_); REQUIRE(a.cell_ == source.cell_);
    REQUIRE(a.forces_ == source.forces_); REQUIRE(a.chemmap_ == source.chemmap_);
}

TEST_CASE("N2P2 writer handles fractional sources and standard reserved slots", "[gpio][n2p2]") {
    GPIO io;
    auto a = training_frame(io);
    a.pos_[0] = Eigen::RowVector3d(0.5, -1, 0);
    a.cartesian = false;
    a.charges_.clear(); a.site_energies_.clear();
    a.have_energy = false; a.global_attribs_d_.clear();
    std::vector<Atoms> frames{a, a};
    std::ostringstream output;
    REQUIRE(io.write(frames, output, "n2p2"));
    REQUIRE(frames[0].pos_ == a.pos_); REQUIRE_FALSE(frames[0].cartesian);
    std::istringstream input(output.str());
    std::vector<Atoms> read;
    REQUIRE(io.read(read, input, "n2p2"));
    REQUIRE(read.size() == 2);
    REQUIRE(read[0].pos_[0][0] == Catch::Approx(0.529177210903));
    REQUIRE(read[0].charges_[0] == 0); REQUIRE(read[0].site_energies_[0] == 0);
    REQUIRE_FALSE(read[0].have_energy);
    REQUIRE(read[0].global_attribs_d_.count("total_charge") == 0);
}

TEST_CASE("N2P2 writer rejects unrepresentable training frames before a block", "[gpio][n2p2]") {
    GPIO io;
    auto a = training_frame(io);
    SECTION("missing force labels") { a.forces_.clear(); }
    SECTION("partial force labels") { a.forces_.push_back(Eigen::Vector3d::Zero()); }
    SECTION("partial charges") { a.charges_.push_back(0); }
    SECTION("partial site energy") { a.site_energies_.push_back(0); }
    SECTION("partial PBC") { a.pbc_[2] = false; }
    SECTION("singular cell") { a.cell_.setZero(); }
    SECTION("unknown element") { a.chemmap_[0] = -1; }
    SECTION("nonfinite coordinate") { a.pos_[0][0] = std::numeric_limits<double>::infinity(); }
    SECTION("nonfinite force") { a.forces_[0][0] = std::numeric_limits<double>::quiet_NaN(); }
    SECTION("nonfinite total charge") { a.global_attribs_d_["total_charge"] = std::numeric_limits<double>::infinity(); }
    SECTION("full Hessian") { a.have_hessian = true; }
    SECTION("absolute index output") { io.absolute_index_write = true; }
    std::ostringstream output;
    REQUIRE_FALSE(io.write_one(a, output, "n2p2"));
    REQUIRE(output.str().empty());
}

TEST_CASE("N2P2 sequential writer reports output failure and permits variable counts", "[gpio][n2p2]") {
    GPIO io;
    auto a = training_frame(io);
    std::ostringstream output;
    REQUIRE(io.seq_write_init(output, "n2p2"));
    REQUIRE(io.seq_write(output, a));
    a.pos_.clear(); a.chemmap_.clear(); a.forces_.clear(); a.charges_.clear(); a.site_energies_.clear();
    REQUIRE(io.seq_write(output, a));
    std::istringstream input(output.str());
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "n2p2"));
    REQUIRE(frames.size() == 2); REQUIRE(frames[1].size() == 0);
    output.setstate(std::ios::badbit);
    REQUIRE_FALSE(io.seq_write(output, a));
}
