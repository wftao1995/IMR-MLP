#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "io/gpio_binary.h"
#include <sstream>
#include <limits>

namespace {
std::string xsf(const std::string &atoms = "H 1 2 3\nO 4 5 6\n", const std::string &count = "2 1") {
    return "# fixture\nCRYSTAL\nPRIMVEC\n3 0.2 0.1\n0.3 4 0.4\n0.5 0.6 5\nCONVVEC\n"
           "3 0.2 0.1\n0.3 4 0.4\n0.5 0.6 5\nPRIMCOORD\n" + count + "\n" + atoms;
}
void reject_xsf(const std::string &input) {
    GPIO reader;
    reader.setmap({"H", "O"});
    Atoms atom;
    atom.resize(1);
    atom.features_["stale"] = "data";
    std::istringstream one(input);
    REQUIRE_FALSE(reader.read_one(atom, one, "xsf"));
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.chemmap_.empty());
    REQUIRE(atom.features_.empty());
    std::vector<Atoms> frames;
    std::istringstream bulk(input);
    REQUIRE_FALSE(reader.read(frames, bulk, "xsf"));
    REQUIRE(frames.empty());
}
}

TEST_CASE("XSF writer crystal subset round trips symbol and numeric species", "[gpio][xsf]") {
    const auto symbols = GENERATE(false, true);
    const auto lossless = GENERATE(false, true);
    GPIO io;
    io.setmap({"H", "O"});
    io.write_symbol = symbols;
    io.lossless = lossless;
    Atoms original;
    original.resize(2);
    original.chemmap_ = {0, 1};
    original.pos_ = {{1, 2, 3}, {4, 5, 6}};
    original.cell_ << 3, .2, .1, .3, 4, .4, .5, .6, 5;
    std::ostringstream encoded;
    REQUIRE(io.write_one(original, encoded, "xsf"));
    std::istringstream input(encoded.str());
    Atoms decoded;
    REQUIRE(io.read_one(decoded, input, "xsf"));
    REQUIRE(decoded.chemmap_ == original.chemmap_);
    REQUIRE(decoded.cell_.isApprox(original.cell_));
    REQUIRE(decoded.pos_ == original.pos_);
    REQUIRE(decoded.is_fully_periodic());
    REQUIRE(decoded.forces_.empty());
}

TEST_CASE("XSF maps legal boundary species and keeps complete empty structures", "[gpio][xsf]") {
    GPIO io;
    io.setmap({"Og", "X", "H"});
    for (const auto &atoms : {"0 1 2 3\n118 4 5 6\n", "X 1 2 3\nOg 4 5 6\n"}) {
        std::istringstream input(xsf(atoms));
        Atoms result;
        REQUIRE(io.read_one(result, input, "xsf"));
        REQUIRE(result.chemmap_ == std::vector<int>{1, 0});
    }
    auto empty = xsf("", "0 1");
    empty.pop_back();
    std::istringstream input(empty);
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, input, "xsf"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].size() == 0);
    GPIO defaults;
    std::istringstream upper(xsf("118 1 2 3\n", "1 1"));
    Atoms result;
    REQUIRE(defaults.read_one(result, upper, "xsf"));
    REQUIRE(result.chemmap_ == std::vector<int>{118});
}

TEST_CASE("XSF invalid numbers counts tags and incomplete data fail atomically", "[gpio][xsf]") {
    for (const auto &token : {"-1", "119", "9999999999999999999", "8junk", "NoSuchElement", "Og", ""}) {
        CAPTURE(token);
        reject_xsf(xsf(std::string(token) + " 1 2 3\n", "1 1"));
    }
    for (const auto &count : {"-1 1", "2147483648 1", "2147483647 1", "1junk 1", "1", "1 2", "1 1 junk", ""}) {
        CAPTURE(count);
        reject_xsf(xsf("", count));
    }
    for (const auto &position : {"bad 2 3", "nan 2 3", "inf 2 3", "1e999 2 3", "1 2", "1 2 3 4"}) {
        CAPTURE(position);
        reject_xsf(xsf(std::string("H ") + position + "\n", "1 1"));
    }
    const auto good = xsf();
    for (const auto &tag : {"CRYSTAL", "PRIMVEC", "CONVVEC", "PRIMCOORD"}) {
        auto bad = good;
        bad.replace(bad.find(tag), std::string(tag).size(), "INVALID");
        reject_xsf(bad);
    }
    for (const auto &row : {"nan 0.2 0.1", "3 0.2"}) {
        auto bad = good;
        bad.replace(bad.find("3 0.2 0.1"), 9, row);
        reject_xsf(bad);
    }
    reject_xsf(xsf("", std::to_string(gpio_binary::MAX_ATOMS) + " 1"));
    reject_xsf(xsf("", std::to_string(gpio_binary::MAX_ATOMS + 1) + " 1"));
    reject_xsf("");
    for (size_t end = 0; (end = good.find('\n', end)) != std::string::npos; ++end) {
        if (end + 1 < good.size()) reject_xsf(good.substr(0, end + 1));
    }
}

TEST_CASE("XSF supports whitespace comments and rejects double mapping without pollution", "[gpio][xsf]") {
    GPIO io;
    std::vector<std::string> map(121, "Unused");
    map.back() = "H";
    io.setmap(map);
    const auto content = "\n # leading\n" + xsf("H +1 +2 +3", "+1 +1");
    std::istringstream input(content);
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "xsf"));
    REQUIRE(atom.chemmap_ == std::vector<int>{120});
    io.absolute_index_read = true;
    std::istringstream absolute(content);
    REQUIRE_FALSE(io.read_one(atom, absolute, "xsf"));
    REQUIRE(atom.size() == 0);
    io.absolute_index_read = false;
    std::istringstream unknown(xsf("Alien 1 2 3\n", "1 1"));
    REQUIRE_FALSE(io.read_one(atom, unknown, "xsf"));
    std::istringstream again(xsf("Alien 1 2 3\n", "1 1"));
    REQUIRE_FALSE(io.read_one(atom, again, "xsf"));
    atom.resize(1);
    REQUIRE_FALSE(io.read_one(atom, "/nonexistent/imrmlp-xsf-input", "xsf"));
    REQUIRE(atom.size() == 0);
}

TEST_CASE("XSF ATOMS retains nonperiodic geometry without a padding cell", "[gpio][xsf]") {
    GPIO io;
    io.setmap({"H", "O"});
    std::istringstream handwritten("# molecule\nATOMS\nH 0 0 0\nO 0 0 9");
    Atoms decoded;
    REQUIRE(io.read_one(decoded, handwritten, "xsf"));
    REQUIRE(decoded.is_nonperiodic());
    REQUIRE(decoded.cell_.isZero());
    REQUIRE(decoded.chemmap_ == std::vector<int>{0, 1});
    REQUIRE(decoded.get_rab(0, 1).isApprox(Eigen::RowVector3d(0, 0, 9)));
    REQUIRE(decoded.get_distance(0, 1) == Catch::Approx(9));
    Atoms original = decoded;
    original.cell_ = 10 * Eigen::Matrix3d::Identity();
    const Atoms before = original;
    io.write_symbol = true;
    std::ostringstream encoded;
    REQUIRE(io.write_one(original, encoded, "xsf"));
    REQUIRE(encoded.str().find("ATOMS\n") != std::string::npos);
    REQUIRE(encoded.str().find("PRIM") == std::string::npos);
    REQUIRE(encoded.str().find("CRYSTAL") == std::string::npos);
    std::istringstream input(encoded.str());
    REQUIRE(io.read_one(decoded, input, "xsf"));
    REQUIRE(decoded.is_nonperiodic());
    REQUIRE(decoded.cell_.isZero());
    REQUIRE(decoded.pos_ == before.pos_);
    REQUIRE(decoded.get_distance(0, 1) == Catch::Approx(9));
    REQUIRE(original.pos_ == before.pos_);
    REQUIRE(original.cell_ == before.cell_);
    REQUIRE(original.pbc_ == before.pbc_);
    REQUIRE(original.chemmap_ == before.chemmap_);
    std::istringstream empty("ATOMS\n # empty molecule\n");
    REQUIRE(io.read_one(decoded, empty, "xsf"));
    REQUIRE(decoded.size() == 0);
    REQUIRE(decoded.is_nonperiodic());
}

TEST_CASE("XSF canonical partial boundaries preserve open directions", "[gpio][xsf]") {
    const bool slab = GENERATE(false, true);
    const std::string label = slab ? "SLAB" : "POLYMER";
    const std::array<bool, 3> pbc{true, slab, false};
    const std::string rows = slab ? "H 0 0 0\nO 0 0 9\n" : "H 0 0 0\nO 0 9 0\n";
    const std::string vectors = "10 0 0\n0 10 0\n0 0 10\n";
    const std::string text = label + "\nPRIMVEC\n" + vectors + "CONVVEC\n" + vectors +
                             "PRIMCOORD\n2 1\n" + rows;
    GPIO io;
    io.setmap({"H", "O"});
    std::istringstream input(text);
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "xsf"));
    REQUIRE(atom.pbc_ == pbc);
    REQUIRE(atom.cell_ == 10 * Eigen::Matrix3d::Identity());
    REQUIRE(atom.get_distance(0, 1) == Catch::Approx(9));
    const Atoms before = atom;
    std::ostringstream encoded;
    REQUIRE(io.write_one(atom, encoded, "xsf"));
    REQUIRE(encoded.str().find(label + "\n") != std::string::npos);
    std::istringstream reread(encoded.str());
    Atoms roundtrip;
    REQUIRE(io.read_one(roundtrip, reread, "xsf"));
    REQUIRE(roundtrip.pbc_ == pbc);
    REQUIRE(roundtrip.cell_ == before.cell_);
    REQUIRE(roundtrip.pos_ == before.pos_);
    REQUIRE(roundtrip.get_distance(0, 1) == Catch::Approx(9));
    REQUIRE(atom.pbc_ == before.pbc_);
    REQUIRE(atom.pos_ == before.pos_);
    REQUIRE(atom.cell_ == before.cell_);
    REQUIRE(atom.chemmap_ == before.chemmap_);
    std::istringstream crystal("CRYSTAL" + text.substr(label.size()));
    REQUIRE(io.read_one(roundtrip, crystal, "xsf"));
    REQUIRE(roundtrip.is_fully_periodic());
    REQUIRE(roundtrip.get_distance(0, 1) == Catch::Approx(1));
}

TEST_CASE("XSF rejects unsupported axes and malformed ATOMS instead of publishing a prefix", "[gpio][xsf]") {
    GPIO io;
    Atoms atom;
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pos_ = {{0, 0, 0}};
    atom.chemmap_ = {1};
    atom.pbc_ = {false, true, false};
    const Atoms before = atom;
    std::ostringstream encoded;
    REQUIRE_FALSE(io.write_one(atom, encoded, "xsf"));
    REQUIRE(encoded.str().empty());
    REQUIRE(atom.pbc_ == before.pbc_);
    REQUIRE(atom.pos_ == before.pos_);
    REQUIRE(atom.cell_ == before.cell_);
    REQUIRE(atom.chemmap_ == before.chemmap_);
    reject_xsf("ATOMS\nH 1 2");
    reject_xsf("ATOMS\nH 0 0 0\nATOMS\nH 1 2 3\n");
    std::istringstream broken("ATOMS\nH 0 0 0\n");
    broken.setstate(std::ios::badbit);
    REQUIRE_FALSE(io.read_one(atom, broken, "xsf"));
    REQUIRE(atom.size() == 0);
}


TEST_CASE("XSF reads complete Cartesian forces in Hartree per Angstrom", "[gpio][xsf]") {
    const bool periodic = GENERATE(false, true);
    const std::string rows = "H 1 2 3 .01 -.02 .03\n8 4 5 6 -.04 .05 0\n";
    std::istringstream input(periodic ? xsf(rows) : "ATOMS\n" + rows);
    GPIO io;
    io.setmap({"H", "O"});
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "xsf"));
    REQUIRE(atom.forces_.size() == 2);
    // Independent numeric eV/Angstrom reference for 1 Ha=27.211396641308 eV.
    REQUIRE(atom.forces_[0].isApprox(Eigen::Vector3d(.27211396641308,
        -.54422793282616, .81634189923924), 1e-13));
    REQUIRE(atom.forces_[1].isApprox(Eigen::Vector3d(-1.08845586565232,
        1.3605698320654, 0), 1e-13));
    REQUIRE(atom.pos_ == std::vector<Eigen::RowVector3d>{{1, 2, 3}, {4, 5, 6}});
}

TEST_CASE("XSF force rows reject mixed missing extra and nonfinite components", "[gpio][xsf]") {
    const bool periodic = GENERATE(false, true);
    for (const std::string rows : {
        "H 1 2 3 .01 .02\nO 4 5 6 .01 .02 .03\n",
        "H 1 2 3 .01 .02 .03 .04\nO 4 5 6 .01 .02 .03\n",
        "H 1 2 3 .01 .02 .03\nO 4 5 6\n",
        "H 1 2 3\nO 4 5 6 .01 .02 .03\n",
        "H 1 2 3 nan 0 0\nO 4 5 6 .01 .02 .03\n",
        "H 1 2 3 .01 inf 0\nO 4 5 6 .01 .02 .03\n",
        "H 1 2 3 .01 .02 1e308\nO 4 5 6 .01 .02 .03\n"}) {
        CAPTURE(rows);
        reject_xsf(periodic ? xsf(rows) : "ATOMS\n" + rows);
    }
}

TEST_CASE("XSF writer force columns use Hartree per Angstrom and retain source state", "[gpio][xsf]") {
    const bool periodic = GENERATE(false, true);
    const bool symbols = GENERATE(false, true);
    const bool lossless = GENERATE(false, true);
    GPIO io;
    io.setmap({"H", "O"});
    io.write_symbol = symbols;
    io.lossless = lossless;
    Atoms atom;
    atom.resize(2);
    atom.chemmap_ = {0, 1};
    atom.pos_ = {{1, 2, 3}, {4, 5, 6}};
    atom.cell_ = 10 * Eigen::Matrix3d::Identity();
    atom.pbc_ = {periodic, periodic, periodic};
    atom.forces_ = {{.27211396641308, -.54422793282616, .81634189923924},
                    {-1.08845586565232, 1.3605698320654, 0}};
    atom.stress_ = Eigen::Matrix3d::Identity();
    atom.have_stress = true;
    const Atoms before = atom;
    std::ostringstream encoded;
    REQUIRE(io.write_one(atom, encoded, "xsf"));
    const auto text = encoded.str();
    std::istringstream rows(text.substr(text.find(periodic ? "PRIMCOORD" : "ATOMS")));
    std::string row;
    REQUIRE(static_cast<bool>(std::getline(rows, row)));
    if (periodic) REQUIRE(static_cast<bool>(std::getline(rows, row)));
    const std::array<Eigen::RowVector3d, 2> expected{
        Eigen::RowVector3d(.01, -.02, .03), Eigen::RowVector3d(-.04, .05, 0)};
    for (int i = 0; i < 2; ++i) {
        REQUIRE(static_cast<bool>(std::getline(rows, row)));
        std::istringstream fields(row);
        std::string species, extra;
        double x, y, z, fx, fy, fz;
        REQUIRE(static_cast<bool>(fields >> species >> x >> y >> z >> fx >> fy >> fz));
        REQUIRE_FALSE(static_cast<bool>(fields >> extra));
        REQUIRE(fx == Catch::Approx(expected[i][0]).epsilon(1e-13));
        REQUIRE(fy == Catch::Approx(expected[i][1]).epsilon(1e-13));
        REQUIRE(fz == Catch::Approx(expected[i][2]).epsilon(1e-13));
    }
    Atoms roundtrip;
    std::istringstream input(text);
    REQUIRE(io.read_one(roundtrip, input, "xsf"));
    REQUIRE(roundtrip.forces_.size() == 2);
    for (int i = 0; i < 2; ++i) REQUIRE(roundtrip.forces_[i].isApprox(before.forces_[i], 1e-13));
    REQUIRE(atom.forces_ == before.forces_);
    REQUIRE(atom.stress_ == before.stress_);
    REQUIRE(atom.have_stress == before.have_stress);
    REQUIRE(atom.pos_ == before.pos_);
    REQUIRE(atom.cell_ == before.cell_);
    REQUIRE(atom.pbc_ == before.pbc_);
    REQUIRE(atom.chemmap_ == before.chemmap_);
}

TEST_CASE("XSF writer rejects partial force arrays before emitting a header", "[gpio][xsf]") {
    GPIO io;
    Atoms atom;
    atom.resize(2);
    atom.forces_ = {{1, 2, 3}};
    const Atoms before = atom;
    std::ostringstream output;
    REQUIRE_FALSE(io.write_one(atom, output, "xsf"));
    REQUIRE(output.str().empty());
    REQUIRE(atom.forces_ == before.forces_);
    REQUIRE(atom.chemmap_ == before.chemmap_);
}

TEST_CASE("XSF primitive coordinates keep their cell with independent or absent conventional vectors", "[gpio][xsf]") {
    const auto geometry = GENERATE("CRYSTAL", "SLAB", "POLYMER");
    const auto vectors = GENERATE("same", "different", "absent");
    auto text = xsf("S 0 0 0\nZn 1.355 -1.355 -1.355\n");
    text.replace(text.find("CRYSTAL"), 7, geometry);
    const auto begin = text.find("CONVVEC");
    const auto end = text.find("PRIMCOORD");
    if (std::string(vectors) == "different")
        text.replace(begin, end - begin, "CONVVEC\n5.42 0 0\n0 5.42 0\n0 0 5.42\n");
    else if (std::string(vectors) == "absent")
        text.erase(begin, end - begin);
    GPIO io;
    std::istringstream input(text);
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "xsf"));
    Eigen::Matrix3d primitive;
    primitive << 3, .2, .1, .3, 4, .4, .5, .6, 5;
    REQUIRE(atom.cell_.isApprox(primitive));
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d::Zero()));
    REQUIRE(atom.pos_[1].isApprox(Eigen::RowVector3d(1.355, -1.355, -1.355)));
    REQUIRE(atom.chemmap_ == std::vector<int>{16, 30});
    const std::array<bool, 3> expected_pbc = {true, std::string(geometry) != "POLYMER", std::string(geometry) == "CRYSTAL"};
    REQUIRE(atom.pbc_ == expected_pbc);
}

TEST_CASE("Present XSF conventional blocks still reject invalid and truncated vectors", "[gpio][xsf]") {
    for (const auto &vectors : {"5.42 0 0\n0 nan 0\n0 0 5.42\n",
                               "5.42 0 0\n0 5.42\n0 0 5.42\n",
                               "5.42 0 0\n0 5.42 0\n",
                               "5.42 0 0\n"}) {
        auto text = xsf();
        const auto begin = text.find("CONVVEC") + std::string("CONVVEC\n").size();
        const auto end = text.find("PRIMCOORD");
        text.replace(begin, end - begin, vectors);
        reject_xsf(text);
    }
}
