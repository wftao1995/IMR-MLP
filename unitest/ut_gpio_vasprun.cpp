#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>

#ifdef USEXML
namespace {
struct XMLFile {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("imrmlp_vasprun_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path path = directory / "vasprun.xml";
    explicit XMLFile(int frames) {
        std::filesystem::create_directory(directory);
        std::ofstream output(path);
        output << "<modeling><generator><i name=\"program\">vasp</i>"
                  "<i name=\"version\">6.3.2</i></generator>"
                  "<atominfo><atoms>1</atoms><types>1</types><array name=\"atomtypes\">"
                  "<set><rc><c>1</c><c>H</c></rc></set></array></atominfo>";
        for (int i = 0; i < frames; ++i) {
            output << "<calculation><scstep/><structure><crystal><varray name=\"basis\">"
                      "<v>3 0 0</v><v>0 3 0</v><v>0 0 3</v></varray><i name=\"volume\">27</i>"
                      "</crystal><varray name=\"positions\"><v>" << .1 * i << " 0.2 0.3</v>"
                      "</varray></structure><varray name=\"forces\"><v>1 2 3</v></varray>"
                      "<varray name=\"stress\"><v>1 0 0</v><v>0 2 0</v><v>0 0 3</v></varray>"
                      "<energy><i name=\"e_fr_energy\">" << -10 - i << "</i>"
                      "<i name=\"e_wo_entrp\">-9</i><i name=\"e_0_energy\">-9.5</i>"
                      "</energy></calculation>";
        }
        output << "</modeling>";
    }
    ~XMLFile() { std::filesystem::remove_all(directory); }
};
}

TEST_CASE("Public vasprun file reader keeps source frame IDs through selection and files", "[gpio][vasprun]") {
    XMLFile xml(5);
    GPIO io;
    io.read_startfrom = 1;
    io.read_irate = 2;
    io.read_endto = 3;
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, xml.path.string(), "vasprun"));
    REQUIRE(frames.size() == 2);
    REQUIRE(frames[0].id == 1);
    REQUIRE(frames[1].id == 3);
    REQUIRE(frames[0].pos_[0].isApprox(Eigen::RowVector3d(.3, .6, .9)));
    REQUIRE(frames[1].pos_[0].isApprox(Eigen::RowVector3d(.9, .6, .9)));
    REQUIRE(frames[0].energy_ == -11);
    REQUIRE(frames[1].energy_ == -13);

    GPIO across;
    across.reset_status_every_read = false;
    std::vector<Atoms> all;
    REQUIRE(across.read(all, xml.path.string(), "xml"));
    REQUIRE(across.read(all, xml.path.string(), "vasprun.xml"));
    REQUIRE(all.size() == 10);
    for (int i = 0; i < 10; ++i) REQUIRE(all[i].id == i);
}

TEST_CASE("Public vasprun format aliases honor selected label discards after parsing", "[gpio][vasprun]") {
    XMLFile xml(2);
    const auto format = GENERATE("xml", "vasprun", "vasprun.xml");
    const auto property = GENERATE("energy", "forces", "virial", "unrelated");
    GPIO io;
    io.discard_props = {property};
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, xml.path.string(), format));
    REQUIRE(frames.size() == 2);
    for (int i = 0; i < 2; ++i) {
        const auto &frame = frames[i];
        REQUIRE(frame.id == i);
        REQUIRE(frame.have_energy == (std::string(property) != "energy"));
        REQUIRE(frame.forces_.empty() == (std::string(property) == "forces"));
        REQUIRE(frame.have_stress == (std::string(property) != "virial"));
        if (frame.have_energy) REQUIRE(frame.energy_ == -10 - i);
        if (!frame.forces_.empty()) REQUIRE(frame.forces_[0].isApprox(Eigen::Vector3d(1, 2, 3)));
        if (frame.have_stress) REQUIRE(frame.stress_(1, 1) == Catch::Approx(54. / 1602.1766208));
        REQUIRE(frame.cell_.isApprox(3. * Eigen::Matrix3d::Identity()));
        REQUIRE(frame.chemmap_ == std::vector<int>{1});
    }
}
#endif

TEST_CASE("Shared read discards preserve unselected structure metadata", "[gpio][discard]") {
    GPIO io;
    io.discard_props = {"remove", "forces"};
    std::istringstream input("1\nLattice=\"3 0 0 0 3 0 0 0 3\" Properties=species:S:1:pos:R:3:forces:R:3 "
                             "energy=-5 remove=4 keep=7\nH 1 2 3 .1 .2 .3\n");
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "extxyz"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].forces_.empty());
    REQUIRE(frames[0].have_energy);
    REQUIRE(frames[0].energy_ == -5);
    REQUIRE(frames[0].global_attribs_d_.count("remove") == 0);
    REQUIRE(frames[0].global_attribs_d_.at("keep") == 7);
}
