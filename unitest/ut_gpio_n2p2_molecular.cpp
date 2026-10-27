#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <sstream>

namespace {
const std::string molecule = "begin\ncomment isolated H\natom 1 -2 0 H .25 2 1 -2 0\nenergy -3\ncharge -.5\nend\n";
}

TEST_CASE("N2P2 reads standard molecules and mixed periodicity without lattice guesses", "[gpio][n2p2][molecular]") {
    const std::string periodic = "begin\nlattice 2 0 0\nlattice 0 2 0\nlattice 0 0 2\natom 0 0 0 O 0 0 0 0 0\nend\n";
    const std::string text = molecule + "begin\nend\n" + periodic + molecule;
    GPIO io;
    std::vector<Atoms> frames;
    std::istringstream input(text);
    REQUIRE(io.read(frames, input, "n2p2"));
    REQUIRE(frames.size() == 4);
    for (int index : {0, 1, 3}) {
        REQUIRE(frames[index].is_nonperiodic()); REQUIRE(frames[index].cell_.isZero());
    }
    REQUIRE(frames[1].size() == 0); REQUIRE_FALSE(frames[1].have_energy);
    REQUIRE(frames[2].is_fully_periodic());
    REQUIRE(frames[2].cell_(0, 0) == Catch::Approx(1.058354421806));
    for (int index : {0, 3}) {
        REQUIRE(frames[index].pos_[0][0] == Catch::Approx(.529177210903));
        REQUIRE(frames[index].pos_[0][1] == Catch::Approx(-1.058354421806));
        REQUIRE(frames[index].forces_[0][0] == Catch::Approx(51.422087120633655));
        REQUIRE(frames[index].energy_ == Catch::Approx(-81.634189923924));
        REQUIRE(frames[index].site_energies_[0] == Catch::Approx(54.422793282616));
        REQUIRE(frames[index].charges_[0] == .25);
        REQUIRE(frames[index].global_attribs_d_.at("total_charge") == -.5);
    }
    std::istringstream sequence(text);
    REQUIRE(io.seq_read_init(sequence, "n2p2"));
    Atoms current;
    for (int i = 0; i < 4; ++i) {
        REQUIRE(io.seq_read(sequence, current)); REQUIRE(current.id == i);
        REQUIRE(current.pbc_ == frames[i].pbc_);
    }
    REQUIRE_FALSE(io.seq_read(sequence, current));
}

TEST_CASE("N2P2 molecular grammar rejects partial and late lattice sections", "[gpio][n2p2][molecular]") {
    for (const auto &text : {
        "begin\nlattice 1 0 0\nend\n",
        "begin\nlattice 1 0 0\nlattice 0 1 0\nend\n",
        "begin\nlattice 1 0 0\natom 0 0 0 H 0 0 0 0 0\nend\n",
        "begin\natom 0 0 0 H 0 0 0 0 0\nlattice 1 0 0\nlattice 0 1 0\nlattice 0 0 1\nend\n",
        "begin\nenergy 0\nlattice 1 0 0\nlattice 0 1 0\nlattice 0 0 1\nend\n"}) {
        CAPTURE(text);
        GPIO io;
        Atoms a;
        std::istringstream input(text);
        REQUIRE_FALSE(io.read_one(a, input, "n2p2"));
        REQUIRE(a.size() == 0);
    }
}

TEST_CASE("N2P2 molecular output omits lattice and preserves measured labels", "[gpio][n2p2][molecular]") {
    GPIO io;
    Atoms a;
    std::istringstream authored(molecule);
    REQUIRE(io.read_one(a, authored, "n2p2"));
    const auto original = a;
    std::ostringstream output;
    REQUIRE(io.seq_write_init(output, "n2p2"));
    REQUIRE(io.seq_write(output, a));
    REQUIRE(output.str().find("lattice") == std::string::npos);
    REQUIRE(output.str().find("comment") == std::string::npos);
    Atoms back;
    std::istringstream written(output.str());
    REQUIRE(io.read_one(back, written, "n2p2"));
    REQUIRE(back.is_nonperiodic()); REQUIRE(back.cell_.isZero());
    REQUIRE(back.forces_[0].isApprox(original.forces_[0], 1e-14));
    REQUIRE(back.pos_[0].isApprox(original.pos_[0], 1e-14));
    REQUIRE(back.energy_ == Catch::Approx(original.energy_));
    REQUIRE(a.pos_ == original.pos_); REQUIRE(a.pbc_ == original.pbc_);
    a.cell_ = Eigen::Matrix3d::Identity() * 9; // Nonperiodic padding is not a periodic box.
    std::ostringstream padded;
    REQUIRE(io.write_one(a, padded, "n2p2"));
    REQUIRE(padded.str().find("lattice") == std::string::npos);
    REQUIRE(a.cell_(0, 0) == 9);
    a.pbc_ = {true, false, false};
    std::ostringstream partial;
    REQUIRE_FALSE(io.write_one(a, partial, "n2p2")); REQUIRE(partial.str().empty());
}
