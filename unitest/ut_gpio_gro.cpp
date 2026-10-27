#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "io/gpio_binary.h"
#include "analyse/compute_vacf.h"
#include <sstream>

namespace {
std::string gro(const std::string &count = "3", const std::string &atoms =
                "1WAT H 1 .1 .2 .3\n1WAT O 2 .4 .5 .6 1 2 3\n2MOL H 3 .7 .8 .9\n",
                const std::string &box = "1 2 3") {
    return "water\n" + count + "\n" + atoms + box;
}
void reject_gro(const std::string &data) {
    GPIO io;
    io.setmap({"H", "O"});
    Atoms atom;
    atom.resize(2);
    atom.molid_ = {1, 2};
    atom.features_["stale"] = "data";
    std::istringstream one(data);
    REQUIRE_FALSE(io.read_one(atom, one, "gro"));
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.chemmap_.empty());
    REQUIRE(atom.molid_.empty());
    REQUIRE(atom.features_.empty());
    std::istringstream input(data);
    std::vector<Atoms> frames;
    REQUIRE_FALSE(io.read(frames, input, "gro"));
    REQUIRE(frames.empty());
}
}

TEST_CASE("GRO whitespace subset preserves units boxes and explicit residue sizes", "[gpio][gro]") {
    const auto triclinic = GENERATE(false, true);
    const auto final_newline = GENERATE(false, true);
    GPIO io;
    io.setmap({"O", "H"});
    auto data = gro("3", "1WAT H 1 .1 .2 .3\n1WAT O 2 .4 .5 .6 1 2 3\n2MOL H 3 .7 .8 .9\n",
                    triclinic ? "1 2 3 .1 .2 .3 .4 .5 .6" : "1 2 3");
    if (final_newline) data += '\n';
    std::istringstream input(data);
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "gro"));
    REQUIRE(atom.chemmap_ == std::vector<int>{1, 0, 1});
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(1, 2, 3)));
    REQUIRE(atom.pos_[2].isApprox(Eigen::RowVector3d(7, 8, 9)));
    Eigen::Matrix3d cell = Eigen::Vector3d(10, 20, 30).asDiagonal();
    if (triclinic) cell << 10, 1, 2, 3, 20, 4, 5, 6, 30;
    REQUIRE(atom.cell_.isApprox(cell));
    REQUIRE(atom.molid_ == std::vector<int>{0, 0, 1});
    REQUIRE(atom.molsize_ == std::vector<int>{2, 1});
    REQUIRE(atom.mol_types_ == std::vector<int>{0, 1});
    REQUIRE(atom.molnum_ == 2);
    REQUIRE(atom.velocities_.empty());
}

TEST_CASE("GRO commits the full size of a single residue group", "[gpio][gro]") {
    GPIO io;
    io.setmap({"H", "O"});
    std::istringstream input(gro("3",
        "1WAT H 1 .1 .2 .3\n1WAT O 2 .4 .5 .6\n1WAT H 3 .7 .8 .9\n"));
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "gro"));
    REQUIRE(atom.molsize_ == std::vector<int>{3});
    REQUIRE(atom.molid_ == std::vector<int>{0, 0, 0});
    REQUIRE(atom.molnum_ == 1);
    REQUIRE(atom.mol_types_ == std::vector<int>{0});
}

TEST_CASE("GRO sequential reads preserve frame identities and clean EOF", "[gpio][gro][stream]") {
    const std::string data = gro() + '\n' + gro("2",
        "3MOL O 7 1.0 .2 .3 8 9 10\n3MOL H 8 1.1 .4 .5\n", "4 5 6");
    GPIO io;
    io.setmap({"H", "O"});
    std::istringstream input(data);
    REQUIRE(io.seq_read_init(input, "gro"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.id == 0);
    REQUIRE(atom.size() == 3);
    REQUIRE(atom.molsize_ == std::vector<int>{2, 1});
    REQUIRE(atom.molid_ == std::vector<int>{0, 0, 1});
    REQUIRE(atom.pos_[2].isApprox(Eigen::RowVector3d(7, 8, 9)));
    const Atoms first = atom;
    atom.features_["stale"] = "first";
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.id == 1);
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.chemmap_ == std::vector<int>{1, 0});
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(10, 2, 3)));
    REQUIRE(atom.pos_[1].isApprox(Eigen::RowVector3d(11, 4, 5)));
    const Eigen::Matrix3d cell = Eigen::Vector3d(40, 50, 60).asDiagonal();
    REQUIRE(atom.cell_.isApprox(cell));
    REQUIRE(atom.molid_ == std::vector<int>{0, 0});
    REQUIRE(atom.molsize_ == std::vector<int>{2});
    REQUIRE(atom.molnum_ == 1);
    REQUIRE(atom.mol_types_ == std::vector<int>{0});
    REQUIRE(atom.features_.empty());
    REQUIRE(atom.velocities_.empty());
    REQUIRE(first.size() == 3);
    REQUIRE(first.pos_[2].isApprox(Eigen::RowVector3d(7, 8, 9)));
    for (int repeat = 0; repeat < 2; ++repeat) {
        REQUIRE_FALSE(io.seq_read(input, atom));
        REQUIRE(atom.size() == 0);
        REQUIRE(atom.molid_.empty());
        REQUIRE(atom.molsize_.empty());
    }
    // Bulk read keeps its existing single-frame behavior independently of seq.
    GPIO bulk;
    bulk.setmap({"H", "O"});
    std::istringstream batch_input(data);
    std::vector<Atoms> frames;
    REQUIRE(bulk.read(frames, batch_input, "gro"));
    REQUIRE(frames.size() == 1);
    REQUIRE(frames[0].molsize_ == std::vector<int>{2, 1});
}

TEST_CASE("GRO sequential malformed frames clear outputs and allow a fresh restart", "[gpio][gro][stream]") {
    GPIO io;
    io.setmap({"H", "O"});
    std::istringstream input(gro() + "\nnext\n2\n1WAT H 1 .1 .2 .3\n");
    REQUIRE(io.seq_read_init(input, "gro"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.id == 0);
    REQUIRE_THROWS_WITH(io.seq_read(input, atom),
                        Catch::Matchers::ContainsSubstring("GRO invalid or truncated frame 1"));
    REQUIRE(atom.size() == 0);
    REQUIRE(atom.molid_.empty());
    REQUIRE(atom.molsize_.empty());
    REQUIRE_FALSE(io.seq_read(input, atom));

    std::istringstream restarted(gro());
    REQUIRE(io.seq_read_init(restarted, "gro"));
    REQUIRE(io.seq_read(restarted, atom));
    REQUIRE(atom.id == 0);
    REQUIRE(atom.molsize_ == std::vector<int>{2, 1});
    REQUIRE_FALSE(io.seq_read(restarted, atom));
    REQUIRE(atom.size() == 0);
    // A blank title starts a frame; it is not clean inter-frame whitespace.
    std::istringstream blank_title("\n");
    REQUIRE(io.seq_read_init(blank_title, "gro"));
    REQUIRE_THROWS_WITH(io.seq_read(blank_title, atom),
                        Catch::Matchers::ContainsSubstring("GRO invalid or truncated frame 0"));
    REQUIRE(atom.size() == 0);
    std::istringstream failed_transport(gro());
    REQUIRE(io.seq_read_init(failed_transport, "gro"));
    failed_transport.setstate(std::ios::badbit);
    REQUIRE_THROWS_WITH(io.seq_read(failed_transport, atom),
                        Catch::Matchers::ContainsSubstring("GRO invalid or truncated frame 0"));
    REQUIRE(atom.size() == 0);
}

TEST_CASE("GRO validates counts before allocation and refuses incomplete atom lines", "[gpio][gro]") {
    for (const auto &count : {"", "0", "-1", "2147483648", "2147483647", "1x", "nan", "1 2"}) {
        CAPTURE(count);
        reject_gro(gro(count, "", ""));
    }
    for (const auto &row : {"", "1WAT", "1WAT H", "1WAT H 1", "1WAT H 1 0", "1WAT H 1 0 0",
                           "1WAT Unknown 1 0 0 0", "1WAT H x 0 0 0", "1WAT H -1 0 0 0",
                           "1WAT H 1 nan 0 0", "1WAT H 1 0 inf 0", "1WAT H 1 0 0 1e308",
                           "1WAT H 1 0 0 0x", "1WAT H 1 0 0 1e999"}) {
        CAPTURE(row);
        reject_gro(gro("1", std::string(row) + "\n"));
    }
    reject_gro(gro(std::to_string(gpio_binary::MAX_ATOMS), "", ""));
    reject_gro(gro(std::to_string(gpio_binary::MAX_ATOMS + 1), "", ""));
    reject_gro("");
    const auto good = gro();
    for (size_t end = 0; (end = good.find('\n', end)) != std::string::npos; ++end)
        reject_gro(good.substr(0, end + 1));
}

TEST_CASE("GRO requires exactly three or nine finite box entries", "[gpio][gro]") {
    for (const auto &box : {"", "1", "1 2", "1 2 3 4", "1 2 3 4 5", "1 2 3 4 5 6",
                           "1 2 3 4 5 6 7", "1 2 3 4 5 6 7 8", "1 2 3 4 5 6 7 8 9 10",
                           "nan 2 3", "1 inf 3", "1 2 1e308", "1 2 abc", "1 2 3junk"}) {
        CAPTURE(box);
        reject_gro(gro("1", "1WAT H 1 0 0 0\n", box));
    }
}

TEST_CASE("GRO accepts long ignored suffix and refuses double mapping and stale outputs", "[gpio][gro]") {
    GPIO io;
    std::vector<std::string> map(121, "Unused");
    map.back() = "H";
    io.setmap(map);
    const auto data = gro("+1", "1WAT H +1 +.1 +.2 +.3 " + std::string(5000, 'x') + "\n");
    Atoms atom;
    std::istringstream input(data);
    REQUIRE(io.read_one(atom, input, "gro"));
    REQUIRE(atom.chemmap_ == std::vector<int>{120});
    io.absolute_index_read = true;
    std::istringstream absolute(data);
    REQUIRE_FALSE(io.read_one(atom, absolute, "gro"));
    REQUIRE(atom.size() == 0);
    atom.resize(1);
    REQUIRE_FALSE(io.read_one(atom, "/nonexistent/imrmlp-gro-input", "gro"));
    REQUIRE(atom.size() == 0);
}

TEST_CASE("GRO standard fixed-width rows retain coordinates boxes and residue pairs", "[gpio][gro]") {
    // Metadata fields have widths 5/5/5/5. Full-width serials and negative
    // eight-column positions deliberately have no whitespace separators.
    const std::string rows =
        "    1WAT      H12345-123.456-234.567 345.678\n"
        "    1WAT      O12346   0.125  -0.250   0.375  1.0000  2.0000  3.0000\n"
        "    2WAT      H12347   0.500   0.625   0.750\n";
    GPIO io;
    io.setmap({"O", "H"});
    std::istringstream input(gro("3", rows, "1 2 3 .1 .2 .3 .4 .5 .6"));
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "gro"));
    REQUIRE(atom.chemmap_ == std::vector<int>{1, 0, 1});
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(-1234.56, -2345.67, 3456.78)));
    REQUIRE(atom.pos_[1].isApprox(Eigen::RowVector3d(1.25, -2.5, 3.75)));
    REQUIRE(atom.pos_[2].isApprox(Eigen::RowVector3d(5, 6.25, 7.5)));
    Eigen::Matrix3d expected;
    expected << 10, 1, 2, 3, 20, 4, 5, 6, 30;
    REQUIRE(atom.cell_.isApprox(expected));
    REQUIRE(atom.molid_ == std::vector<int>{0, 0, 1});
    REQUIRE(atom.molsize_ == std::vector<int>{2, 1});
    REQUIRE(atom.molnum_ == 2);
    REQUIRE(atom.velocities_.empty());
}

TEST_CASE("GRO fixed-width frames are independent and atom names map exactly", "[gpio][gro][stream]") {
    GPIO io;
    io.setmap({"OW", "HW"});
    const std::string first = gro("2",
        "    1WAT     OW99999   0.125  -0.250   0.375\n"
        "    1WAT     HW    0   0.500   0.625   0.750\n", "1 2 3");
    const std::string second = gro("1",
        "    2WAT     HW    1  -0.125   0.250  -0.375\n", "4 5 6");
    std::istringstream input(first + '\n' + second);
    REQUIRE(io.seq_read_init(input, "gro"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    const Atoms saved = atom;
    REQUIRE(atom.chemmap_ == std::vector<int>{0, 1});
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.id == 1);
    REQUIRE(atom.chemmap_ == std::vector<int>{1});
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(-1.25, 2.5, -3.75)));
    REQUIRE(atom.cell_.isApprox(Eigen::Vector3d(40, 50, 60).asDiagonal().toDenseMatrix()));
    REQUIRE(saved.pos_[0].isApprox(Eigen::RowVector3d(1.25, -2.5, 3.75)));
    REQUIRE(saved.molsize_ == std::vector<int>{2});
    REQUIRE_FALSE(io.seq_read(input, atom));
    // OW is an explicit internal type here, not an inferred oxygen element.
    GPIO defaults;
    std::istringstream unknown(first);
    REQUIRE_FALSE(defaults.read_one(atom, unknown, "gro"));
}

TEST_CASE("GRO fixed-width residue numbers and names remain separate identities", "[gpio][gro]") {
    std::istringstream input(gro("2",
        "    12WAT     H    1   0.100   0.200   0.300\n"
        "   12WAT      O    2   0.400   0.500   0.600\n"));
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom, input, "gro"));
    REQUIRE(atom.molid_ == std::vector<int>{0, 1});
    REQUIRE(atom.molsize_ == std::vector<int>{1, 1});
}

TEST_CASE("GRO fixed metadata does not fall back on malformed coordinates", "[gpio][gro]") {
    for (const auto &row : {
            "    1WAT      H12345   0.100   0.200\n",
            "    1WAT      H12345     bad   0.200   0.300\n",
            "    1WAT      H12345   0.100  1 2 3    0.300\n"})
        reject_gro(gro("1", row));
}


TEST_CASE("GRO complete standard velocity columns convert nm per ps to Angstrom per ps", "[gpio][gro]") {
    const bool zero = GENERATE(false, true);
    const std::string first = "    1WAT      H    1   0.100   0.200   0.300";
    const std::string second = "    1WAT      O    2   0.400   0.500   0.600";
    const std::string zeros = "  0.0000  0.0000  0.0000";
    const auto rows = first + (zero ? zeros : "  0.1250 -0.2500  1.5000") + "\n" +
                      second + (zero ? zeros : " -1.0000  0.0000  2.3456") + "\n";
    std::istringstream input(gro("2", rows));
    GPIO io;
    io.setmap({"H", "O"});
    Atoms atom;
    REQUIRE(io.read_one(atom,input,"gro"));
    REQUIRE(atom.velocities_.size() == 2);
    if (zero) {
        REQUIRE(atom.velocities_[0].isZero());
        REQUIRE(atom.velocities_[1].isZero());
    } else {
        REQUIRE(atom.velocities_[0].isApprox(Eigen::Vector3d(1.25,-2.5,15),1e-13));
        REQUIRE(atom.velocities_[1].isApprox(Eigen::Vector3d(-10,0,23.456),1e-13));
    }
    REQUIRE(atom.pos_[0].isApprox(Eigen::RowVector3d(1,2,3),1e-13));
    REQUIRE(atom.pos_[1].isApprox(Eigen::RowVector3d(4,5,6),1e-13));
    REQUIRE(atom.molsize_ == std::vector<int>{2});
}

TEST_CASE("GRO incomplete or nonstandard velocity suffixes preserve geometry-only compatibility", "[gpio][gro]") {
    const std::string suffix = GENERATE(
        "", "  0.1250", "  0.1250 -0.2500", "     bad  0.0000  0.0000",
        "     nan  0.0000  0.0000", "   1e308  0.0000  0.0000",
        "  0.1250 -0.2500  1.5000 ignored");
    const std::string rows =
        "    1WAT      H    1   0.100   0.200   0.300  0.1250 -0.2500  1.5000\n"
        "    1WAT      O    2   0.400   0.500   0.600" + suffix + "\n";
    std::istringstream input(gro("2",rows));
    GPIO io;
    Atoms atom;
    REQUIRE(io.read_one(atom,input,"gro"));
    REQUIRE(atom.velocities_.empty());
    REQUIRE(atom.size() == 2);
    REQUIRE(atom.pos_[1].isApprox(Eigen::RowVector3d(4,5,6),1e-13));
}

TEST_CASE("GRO sequential velocity presence is independent for every frame", "[gpio][gro][stream]") {
    const std::string row = "    1WAT      H    1   0.100   0.200   0.300";
    std::istringstream input(gro("1",row + "  0.1250 -0.2500  1.5000\n") + "\n" +
                             gro("1",row + "\n"));
    GPIO io;
    REQUIRE(io.seq_read_init(input,"gro"));
    Atoms atom;
    REQUIRE(io.seq_read(input,atom));
    const Atoms first = atom;
    REQUIRE(io.seq_read(input,atom));
    REQUIRE(atom.velocities_.empty());
    REQUIRE(first.velocities_.size() == 1);
    REQUIRE(first.velocities_[0].isApprox(Eigen::Vector3d(1.25,-2.5,15),1e-13));
}

TEST_CASE("GRO supplied standard velocities feed VACF without position differencing", "[gpio][gro][vacf]") {
    const std::string frame = gro("1",
        "    1WAT      H    1   0.100   0.200   0.300  0.1000  0.2000  0.0000\n") + "\n";
    std::string text;
    for (int i = 0; i < 8; ++i) text += frame;
    GPIO io;
    std::istringstream input(text);
    REQUIRE(io.seq_read_init(input,"gro"));
    std::vector<Atoms> trajectory;
    Atoms atom;
    while (io.seq_read(input,atom)) trajectory.push_back(atom);
    REQUIRE(trajectory.size() == 8);
    GPSettings settings;
    settings.Modify("algo","2");
    settings.Modify("timestep","1.0");
    settings.Modify("total","true");
    settings.Modify("types","false");
    settings.Modify("track","false");
    settings.Modify("groups","false");
    settings.Modify("vdos","false");
    settings.Modify("normalize","false");
    settings.Modify("ndim","2");
    ComputeVACF vacf(settings);
    vacf.compute(trajectory);
    REQUIRE_FALSE(vacf.vacf_isotropic.total.empty());
    for (const auto &lag : vacf.vacf_isotropic.total) {
        REQUIRE(lag[0] == Catch::Approx(1).margin(1e-12));
        REQUIRE(lag[1] == Catch::Approx(4).margin(1e-12));
        REQUIRE(lag[3] == Catch::Approx(2.5).margin(1e-12));
    }
}
