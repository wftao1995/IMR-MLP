#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "utils/compressed_stream.h"
#include <chrono>
#include <filesystem>
#include <fstream>

#ifdef USEXML
namespace {
std::string xml_header() {
    // Deliberately reorder version and program, and put unrelated arrays first.
    return "<modeling><generator><i name=\"version\">6.3.2</i><i name=\"program\">vasp</i></generator>"
           "<atominfo><atoms>1</atoms><types>1</types><array name=\"atoms\"><set/></array>"
           "<array name=\"atomtypes\"><set><rc><c>1</c><c>H</c></rc></set></array></atominfo>";
}
std::string xml_frame() {
    return "<calculation><scstep/><structure><crystal><i name=\"volume\">27</i>"
           "<varray name=\"basis\"><v>3 0 0</v><v>0 3 0</v><v>0 0 3</v></varray>"
           "</crystal><varray name=\"positions\"><v>0.1 0.2 0.3</v></varray></structure>"
           "<varray name=\"unused\"><v>99 99 99</v></varray>"
           "<varray name=\"forces\"><v>1 2 3</v></varray>"
           "<energy><i name=\"e_fr_energy\">-10</i><i name=\"e_0_energy\">-9.5</i></energy></calculation>";
}
void replace_xml(std::string &text, const std::string &before, const std::string &after) {
    const auto at = text.find(before);
    REQUIRE(at != std::string::npos);
    text.replace(at, before.size(), after);
}
struct ValidationXMLFile {
    std::filesystem::path directory = std::filesystem::temp_directory_path() /
        ("imrmlp_vasprun_validation_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::path path;
    ValidationXMLFile(const std::string &text, bool compressed = false) {
        std::filesystem::create_directory(directory);
        path = directory / (compressed ? "vasprun.xml.gz" : "vasprun.xml");
        auto output = open_ofile(path.string());
        REQUIRE(output);
        *output << text;
        REQUIRE(*output);
    }
    ~ValidationXMLFile() { std::filesystem::remove_all(directory); }
};
}

TEST_CASE("Vasprun named sections preserve optional stress, compression and frame selection", "[gpio][vasprun]") {
#ifdef USE_GZIP
    const bool compressed = GENERATE(false, true);
#else
    const bool compressed = false;
#endif
    std::string frame = xml_frame();
    replace_xml(frame, "<scstep/>", ""); // Geometry/labels need not contain electronic SCF steps.
    ValidationXMLFile input(xml_header() + frame + frame + "</modeling>", compressed);
    GPIO io;
    io.read_startfrom = 1;
    io.read_endto = 1;
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input.path.string(), "xml"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].id == 1);
    REQUIRE(frames[0].energy_ == -10.0);
    REQUIRE(frames[0].fenergy_ == -10.0);
    REQUIRE(frames[0].have_fenergy);
    REQUIRE_FALSE(frames[0].have_stress);
    REQUIRE(frames[0].pos_[0].isApprox(Eigen::RowVector3d(.3, .6, .9)));
    REQUIRE(frames[0].forces_[0].isApprox(Eigen::Vector3d(1, 2, 3)));
}

TEST_CASE("Vasprun malformed required metadata is rejected without publishing atoms", "[gpio][vasprun]") {
    const int damage = GENERATE(range(0, 14));
    std::string text = xml_header() + xml_frame() + "</modeling>";
    switch (damage) {
    case 0: text = "<modeling/>"; break;
    case 1: replace_xml(text, "name=\"version\"", "name=\"missing\""); break;
    case 2: replace_xml(text, ">6.3.2<", "><"); break;
    case 3: replace_xml(text, ">6.3.2<", ">x<"); break;
    case 4: replace_xml(text, "<atoms>1</atoms>", ""); break;
    case 5: replace_xml(text, "<types>1</types>", "<types>2</types>"); break;
    case 6: replace_xml(text, "name=\"atomtypes\"", "name=\"unknown\""); break;
    case 7: replace_xml(text, "<c>1</c>", "<c>2</c>"); break;
    case 8: replace_xml(text, "<c>H</c>", "<c/>"); break;
    case 9: replace_xml(text, "<c>H</c>", "<c>Unmapped</c>"); break;
    case 10: replace_xml(text, "<c>1</c>", "<c>-1</c>"); break;
    case 11: replace_xml(text, "<generator>", "<generator><i>ignored</i>"); break;
    case 12: replace_xml(text, "<atoms>1</atoms>", "<atoms>1 trailing</atoms>"); break;
    case 13: replace_xml(text, "<types>1</types>", "<types>1.2</types>"); break;
    }
    ValidationXMLFile input(text);
    GPIO io;
    Atoms sentinel;
    sentinel.pos_ = {{7, 8, 9}};
    sentinel.chemmap_ = {1};
    std::vector<Atoms> frames{sentinel};
    // Unrelated nameless fields are allowed; they cannot replace the required named version.
    if (damage == 11) {
        REQUIRE(io.read(frames, input.path.string(), "xml"));
        REQUIRE(frames.size() == 2);
    } else {
        REQUIRE_FALSE(io.read(frames, input.path.string(), "xml"));
        REQUIRE(frames.size() == 1);
    }
    REQUIRE(frames[0].pos_[0].isApprox(Eigen::RowVector3d(7, 8, 9)));
}

TEST_CASE("Vasprun malformed calculation never publishes a partial frame", "[gpio][vasprun]") {
    const int damage = GENERATE(range(0, 20));
    std::string broken = xml_frame();
    switch (damage) {
    case 0: replace_xml(broken, "<v>0.1 0.2 0.3</v>", "<v>nan 0.2 0.3</v>"); break;
    case 1: replace_xml(broken, "<structure>", "<unknown>"); replace_xml(broken, "</structure>", "</unknown>"); break;
    case 2: replace_xml(broken, "<crystal>", "<unknown>"); replace_xml(broken, "</crystal>", "</unknown>"); break;
    case 3: replace_xml(broken, "name=\"basis\"", "name=\"unknown\""); break;
    case 4: replace_xml(broken, "<v>0 0 3</v>", ""); break;
    case 5: replace_xml(broken, "<v>0 0 3</v>", "<v>0 0 3</v><v>0 0 3</v>"); break;
    case 6: replace_xml(broken, "<v>3 0 0</v>", "<v>nan 0 0</v>"); break;
    case 7: replace_xml(broken, "<v>0.1 0.2 0.3</v>", "<v>0.1 0.2</v>"); break;
    case 8: replace_xml(broken, "<v>0.1 0.2 0.3</v>", "<v>0.1 0.2 0.3 0.4</v>"); break;
    case 9: replace_xml(broken, "<v>0.1 0.2 0.3</v>", "<v>0 0 0</v><v>1 1 1</v>"); break;
    case 10: replace_xml(broken, "name=\"positions\"", "name=\"unknown\""); break;
    case 11: replace_xml(broken, "name=\"forces\"", "name=\"unknown\""); break;
    case 12: replace_xml(broken, "<v>1 2 3</v>", ""); break;
    case 13: replace_xml(broken, "<v>1 2 3</v>", "<v>1 inf 3</v>"); break;
    case 14: replace_xml(broken, "<energy>", "<varray name=\"stress\"><v>1 2 3</v></varray><energy>"); break;
    case 15: replace_xml(broken, "name=\"e_fr_energy\"", "name=\"unknown\""); break;
    case 16: replace_xml(broken, ">-10<", ">nan<"); break;
    case 17: replace_xml(broken, "name=\"volume\"", "name=\"unknown\""); break;
    case 18: replace_xml(broken, ">27<", ">0<"); break;
    case 19: replace_xml(broken, "<v>1 2 3</v>", "<v/>"); break;
    }
    ValidationXMLFile input(xml_header() + xml_frame() + broken + "</modeling>");
    GPIO io;
    std::vector<Atoms> frames;
    REQUIRE_FALSE(io.read(frames, input.path.string(), "xml"));
    REQUIRE(frames.size() == 1); // The preceding complete frame remains usable.
    REQUIRE(frames[0].pos_.size() == 1);
    REQUIRE(frames[0].forces_.size() == 1);
    REQUIRE(frames[0].energy_ == -10);
}
#endif
