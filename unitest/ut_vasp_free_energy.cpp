#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "io/gpio_session.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

namespace {
struct RestoreEnergySelection {
    bool value = GPIO::vasp_use_free_eng;
    ~RestoreEnergySelection() { GPIO::vasp_use_free_eng = value; }
};
std::string outcar() {
    std::ostringstream text;
    text << " vasp.6.3.2\n INCAR: SYSTEM = explicit F\n POTCAR: PAW_PBE H 2000\n"
            " VRHFIN =H: s1\n ions per type = 1\n NELM = 60\n EDIFF = .00001\n"
            " EDIFFG = -.1\n NSW = 2\n IBRION = 0\n ISIF = 2\n";
    for (int i = 0; i < 2; ++i) {
        text << "--------------------------------------- Iteration 1(1)\n"
                "------------------------ aborting loop because EDIFF is reached\n"
                " VOLUME and BASIS-vectors are now :\n ---\n energy-cutoff : 400\n"
                " volume of cell : 27\n direct lattice vectors reciprocal lattice vectors\n"
                " 3 0 0 0 0 0\n 0 3 0 0 0 0\n 0 0 3 0 0 0\n"
                " POSITION                                       TOTAL-FORCE (eV/Angst)\n ---\n"
             << " " << i + 1 << " 2 3 .1 .2 .3\n"
             << "  free  energy   TOTEN  = " << -10 - i << " eV\n\n"
             << "  energy  without entropy= " << -9.75 - i
             << "  energy(sigma->0) = " << -9.5 - i << "\n";
    }
    return text.str();
}
#ifdef USEXML
struct XMLFile {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("imrmlp_vasp_fenergy_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path path = directory / "vasprun.xml";
    XMLFile(int version, bool reordered, bool omit_second_free = false) {
        std::filesystem::create_directory(directory);
        std::ofstream text(path);
        text << "<modeling><generator><i name=\"program\">vasp</i><i name=\"version\">"
             << version << ".3.2</i></generator><atominfo><atoms>1</atoms><types>1</types>"
                "<array name=\"atomtypes\"><set><rc><c>1</c><c>H</c></rc></set></array></atominfo>";
        for (int i = 0; i < 2; ++i) {
            text << "<calculation><scstep/><structure><crystal><varray name=\"basis\">"
                    "<v>3 0 0</v><v>0 3 0</v><v>0 0 3</v></varray><i name=\"volume\">27</i>"
                    "</crystal><varray name=\"positions\"><v>0.1 0.2 0.3</v></varray></structure>"
                    "<varray name=\"forces\"><v>1 2 3</v></varray><varray name=\"stress\">"
                    "<v>1 0 0</v><v>0 2 0</v><v>0 0 3</v></varray><energy>";
            const std::string free = omit_second_free && i == 1 ? "" :
                "<i name=\"e_fr_energy\">" + std::to_string(-10. - i) + "</i>";
            const std::string without = "<i name=\"e_wo_entrp\">" + std::to_string(-9.75 - i) + "</i>";
            const std::string zero = "<i name=\"e_0_energy\">" + std::to_string(-9.5 - i) + "</i>";
            text << (reordered ? zero + free + without : free + without + zero) << "</energy></calculation>";
        }
        text << "</modeling>";
    }
    ~XMLFile() { std::filesystem::remove_all(directory); }
};
#endif
}

TEST_CASE("Ordinary OUTCAR preserves explicit free energy under either energy selection", "[gpio][outcar][fenergy]") {
    RestoreEnergySelection restore;
    GPIO::vasp_use_free_eng = GENERATE(false, true);
    const auto sequential = GENERATE(false, true);
    GPIO io;
    std::istringstream input(outcar());
    std::vector<Atoms> frames;
    if (sequential) {
        REQUIRE(io.seq_read_init(input, "outcar"));
        Atoms atom;
        while (io.seq_read(input, atom)) frames.push_back(atom);
    } else REQUIRE(io.read(frames, input, "outcar"));
    REQUIRE(frames.size() == 2);
    for (int i = 0; i < 2; ++i) {
        REQUIRE(frames[i].have_fenergy);
        REQUIRE(frames[i].fenergy_ == -10 - i);
        REQUIRE(frames[i].energy_ == (GPIO::vasp_use_free_eng ? -10. : -9.5) - i);
        REQUIRE(frames[i].pos_[0].isApprox(Eigen::RowVector3d(i + 1, 2, 3)));
        REQUIRE(frames[i].forces_[0].isApprox(Eigen::Vector3d(.1, .2, .3)));
    }
}

TEST_CASE("OUTCAR sessions freeze energy selection independently", "[gpio][outcar][sessions]") {
    RestoreEnergySelection restore;
    GPIO configuration(false);
    GPIO::vasp_use_free_eng = true;
    GPIOReader free_energy(configuration, std::make_unique<std::istringstream>(outcar()), "outcar");
    GPIO::vasp_use_free_eng = false;
    GPIOReader zero_energy(configuration, std::make_unique<std::istringstream>(outcar()), "outcar");
    GPIO::vasp_use_free_eng = true;
    for (int i = 0; i < 2; ++i) {
        auto a = free_energy.next(), b = zero_energy.next();
        REQUIRE(a); REQUIRE(b);
        REQUIRE(a->energy_ == -10. - i);
        REQUIRE(b->energy_ == -9.5 - i);
        REQUIRE(a->fenergy_ == b->fenergy_);
    }
    REQUIRE_FALSE(free_energy.next());
    REQUIRE_FALSE(zero_energy.next());
}

#ifdef USEXML
TEST_CASE("Public XML retains named free energy independently of version and energy item order", "[gpio][vasprun][fenergy]") {
    RestoreEnergySelection restore;
    GPIO::vasp_use_free_eng = GENERATE(false, true);
    const auto version = GENERATE(5, 6);
    const auto reordered = GENERATE(false, true);
    XMLFile input(version, reordered);
    GPIO io;
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input.path.string(), "vasprun"));
    REQUIRE(frames.size() == 2);
    for (int i = 0; i < 2; ++i) {
        REQUIRE(frames[i].have_fenergy);
        REQUIRE(frames[i].fenergy_ == -10 - i);
        const double selected = GPIO::vasp_use_free_eng ? -10. : (version == 6 ? -9.5 : -9.75);
        REQUIRE(frames[i].energy_ == selected - i);
        REQUIRE(frames[i].have_stress);
        REQUIRE(frames[i].forces_[0].isApprox(Eigen::Vector3d(1, 2, 3)));
    }
}

TEST_CASE("XML without a source free energy never carries over an earlier frame label", "[gpio][vasprun][fenergy]") {
    RestoreEnergySelection restore;
    GPIO::vasp_use_free_eng = false;
    XMLFile input(6, true, true);
    GPIO io;
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input.path.string(), "xml"));
    REQUIRE(frames.size() == 2);
    REQUIRE(frames[0].have_fenergy);
    REQUIRE_FALSE(frames[1].have_fenergy);
    REQUIRE(frames[1].energy_ == -10.5);
}
#endif

TEST_CASE("Free energy discard aliases clear only the independent native free energy", "[gpio][fenergy][discard]") {
    const std::string key = GENERATE("fenergy", "free_energy", "energy");
    std::istringstream input("1\nLattice=\"3 0 0 0 3 0 0 0 3\" energy=-9 fenergy=-10 "
                             "Properties=species:S:1:pos:R:3:forces:R:3 virial=\"1 0 0 0 2 0 0 0 3\"\nH 1 2 3 .1 .2 .3\n");
    GPIO io;
    io.discard_props = {key};
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "extxyz"));
    REQUIRE(frames.size() == 1);
    const auto &frame = frames.front();
    REQUIRE(frame.have_energy == (key != "energy"));
    REQUIRE(frame.have_fenergy == (key == "energy"));
    if (frame.have_energy) REQUIRE(frame.energy_ == -9);
    if (frame.have_fenergy) REQUIRE(frame.fenergy_ == -10);
    REQUIRE(frame.have_stress);
    REQUIRE(frame.stress_(1, 1) == 2);
    REQUIRE(frame.forces_[0].isApprox(Eigen::Vector3d(.1, .2, .3)));
}
