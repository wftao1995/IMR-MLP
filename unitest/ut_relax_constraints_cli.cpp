#include <catch2/catch_all.hpp>
#include "mindriver/relax_constraints_io.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>
#include <numbers>
#include <limits>

int relaxstrv2(int argc, char *argv[]);

namespace {
struct Directory {
    std::filesystem::path previous = std::filesystem::current_path(), path;
    Directory() {
        char name[] = "/tmp/imrmlp_constraints_cli_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("mkdtemp failed");
        path = created; std::filesystem::current_path(path);
    }
    ~Directory() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};
Atoms structure(double scale = 1) {
    Atoms a;
    a.cell_ << 3, 0.2, 0.1, 0.6, 3.5, 0.3, 0.4, 0.5, 4;
    a.cell_ *= scale;
    a.pos_ = {{0.7, 0.6, 0.5}, {1.9, 1.3, 1.1}};
    a.chemmap_ = {0, 0};
    return a;
}
void input(std::vector<Atoms> frames = {structure()}) {
    std::ofstream("index.map") << "H 0\n";
    GPIO io; REQUIRE(io.readmap("index.map"));
    REQUIRE(io.write(frames, "input.cfg", "mlpcfg"));
}
void settings(const std::string &extra = "") {
    std::istringstream text(
        "input input.cfg\ninput-format mlpcfg\noutput output.cfg\noutput-format mlpcfg\n"
        "pot:pottype lj612\npot:ntypes 1\npot:0-0 0 1\npot:cut:0-0 1\n"
        "file-constraints constraints.json\nisif 6\n"
        "relax:cell_strain False\nrelax:dprecond False\nrelax:driver lbfgspp\n"
        "relax:gtol 1e-7\nrelax:maxiteration 600\nrelax:constraint_loops 12\n"
        "relax:lbfgspp:max_linesearch 80\n"
        "relax:write_traj False\n" + extra);
    std::map<std::string,std::string> values;
    for (std::string line; std::getline(text,line);) {
        const auto sep=line.find(' '); values[line.substr(0,sep)]=line.substr(sep+1);
    }
    std::ofstream out("settings.ini");
    for (const auto &[key,value] : values) out << key << ' ' << value << '\n';
}
void json(const std::string &entries) {
    std::ofstream("constraints.json") << "{\"version\":1,\"constraints\":[" << entries << "]}";
}
int run(std::vector<std::string> extra = {}) {
    std::vector<std::string> args{"relax", "settings.ini", "--num_threads", "1"};
    args.insert(args.end(),extra.begin(),extra.end());
    std::vector<char *> argv;
    for (auto &a:args) argv.push_back(a.data());
    CMDOption::global_cmdopts.reset();
    return relaxstrv2(argv.size(),argv.data());
}
std::vector<Atoms> output() {
    GPIO io; REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> result;
    REQUIRE(io.read(result,"output.cfg","mlpcfg"));
    return result;
}
std::string bytes(const std::string &file) {
    std::ifstream in(file); return {std::istreambuf_iterator<char>(in), {}};
}
}

TEST_CASE("V2 JSON lattice equalities move a real cell to noninitial targets", "[relax_constraints_cli]") {
    Directory dir; input(); settings();
    const std::string kind=GENERATE("volume","lattice_length","lattice_angle","lattice_ratio");
    double target=0;
    std::string entry;
    if(kind=="volume") { target=45; entry="\"kind\":\"volume\",\"target\":45"; }
    if(kind=="lattice_length") { target=3.3; entry="\"kind\":\"lattice_length\",\"axis\":\"a\",\"target\":3.3"; }
    if(kind=="lattice_angle") { target=70; entry="\"kind\":\"lattice_angle\",\"axes\":[\"a\",\"b\"],\"target\":70"; }
    if(kind=="lattice_ratio") { target=0.95; entry="\"kind\":\"lattice_ratio\",\"axes\":[\"a\",\"b\"],\"target\":0.95"; }
    json("{"+entry+",\"residual_tolerance\":1e-6}");
    REQUIRE(run()==0);
    auto result=output(); REQUIRE(result.size()==1);
    const auto &cell=result[0].cell_;
    REQUIRE_FALSE(cell.isApprox(structure().cell_,1e-7));
    if(kind=="volume") REQUIRE(cell.determinant()==Catch::Approx(target).margin(1e-6));
    if(kind=="lattice_length") REQUIRE(cell.row(0).squaredNorm()==Catch::Approx(target*target).margin(1e-6));
    if(kind=="lattice_angle") REQUIRE(cell.row(0).dot(cell.row(1))/(cell.row(0).norm()*cell.row(1).norm())==
        Catch::Approx(std::cos(target*std::numbers::pi/180)).margin(1e-6));
    if(kind=="lattice_ratio") REQUIRE(cell.row(0).squaredNorm()/cell.row(1).squaredNorm()==Catch::Approx(target*target).margin(1e-6));
}

TEST_CASE("V2 implicit lattice targets bind independently for each frame", "[relax_constraints_cli]") {
    Directory dir;
    auto second=structure(1.2); second.pos_.push_back({2.5,2.7,2.9}); second.chemmap_.push_back(0);
    input({structure(),second}); settings();
    json("{\"kind\":\"volume\",\"residual_tolerance\":1e-7}");
    REQUIRE(run()==0);
    auto result=output(); REQUIRE(result.size()==2);
    REQUIRE(result[0].cell_.isApprox(structure().cell_,1e-10));
    REQUIRE(result[1].cell_.isApprox(second.cell_,1e-10));
    REQUIRE(result[1].size()==3);
}

TEST_CASE("V2 sequential lattice constraints match batch for changing cells and atom counts", "[relax_constraints_cli][relax_stream]") {
    Directory dir;
    auto second=structure(1.2); second.pos_.push_back({2.5,2.7,2.9}); second.chemmap_.push_back(0);
    input({structure(),second}); settings();
    json("{\"kind\":\"volume\",\"target\":45,\"residual_tolerance\":1e-6}");
    REQUIRE(run()==0);
    const auto batch=output();
    REQUIRE(run({"--seq-read"})==0);
    const auto sequential=output();
    REQUIRE(batch.size()==2);
    REQUIRE(sequential.size()==2);
    for (size_t i=0; i<sequential.size(); ++i) {
        CAPTURE(i);
        REQUIRE(sequential[i].size()==2+i);
        REQUIRE(sequential[i].cell_.determinant()==Catch::Approx(45).margin(1e-6));
        REQUIRE(sequential[i].cell_.isApprox(batch[i].cell_,1e-9));
        for (int j=0; j<sequential[i].size(); ++j)
            REQUIRE(sequential[i].pos_[j].isApprox(batch[i].pos_[j],1e-9));
    }
}

TEST_CASE("V2 JSON constraints reject unsupported combinations before output access", "[relax_constraints_cli][errors]") {
    Directory dir; input();
    const std::string extra=GENERATE("relax:cell_strain True\n", "relax:cell_strain True\nrelax:symm_strain False\n",
        "isif 2\n", "isif 7\n", "relax:fix_vol True\nisif 0\nrelax:relax_cell True\n",
        "relax:keep_symm True\n", "relax:symm_force True\n", "relax:couple 1\n",
        "relax:dprecond True\n", "relax:freeze_small_gd True\n", "relax:pressure 1\n",
        "relax:cartesian True\n", "relax:fixcm True\n",
        "relax:external_stress 1 0 0 0 0 0\n", "relax:selective_dynamics True\n");
    settings(extra);
    json("{\"kind\":\"volume\",\"target\":45,\"residual_tolerance\":1e-6}");
    std::ofstream("output.cfg") << "preserve old result";
    REQUIRE_THROWS(run());
    REQUIRE(bytes("output.cfg")=="preserve old result");
}

TEST_CASE("V2 constraints validate every frame and embedded frozen flags before output", "[relax_constraints_cli][errors]") {
    Directory dir;
    auto second=structure(); second.sel_dynpos_.assign(second.size(),Eigen::RowVector3d::Ones());
    second.sel_dynpos_[1][0]=0;
    input();
    GPIO io; io.force_write_all_to_one = true; REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> frames{structure(),second};
    REQUIRE(io.write(frames,"input.poscars","poscar"));
    settings("input input.poscars\ninput-format poscars\nseq-write True\n");
    json("{\"kind\":\"volume\",\"residual_tolerance\":1e-6}");
    std::ofstream("output.cfg") << "preserve old result";
    REQUIRE_THROWS(run());
    REQUIRE(bytes("output.cfg")=="preserve old result");
}

TEST_CASE("V2 constraint file command line overrides settings and can be cleared", "[relax_constraints_cli]") {
    Directory dir; input(); settings("file-constraints missing.json\n");
    json("{\"kind\":\"volume\",\"residual_tolerance\":1e-6}");
    REQUIRE(run({"--file-constraints","constraints.json"})==0);
    REQUIRE(run({"--file-constraints",""})==0);
    REQUIRE(run({"--file-constraints","constraints.json","--seq-read"})==0);
    std::ofstream("output.cfg") << "preserve old result";
    REQUIRE_THROWS(run({"--file-constraints","constraints.json","--tags","cell_strain","--vals","True"}));
    REQUIRE(bytes("output.cfg")=="preserve old result");
}

TEST_CASE("V2 constraints reject later-frame species outside the potential before output", "[relax_constraints_cli][errors]") {
    Directory dir;
    const int invalid_species = GENERATE(1, std::numeric_limits<int>::max());
    auto second = structure(1.2);
    second.chemmap_[1] = invalid_species;
    input({structure(), second});
    settings("seq-write True\nrelax:write_traj True\nrelax:trajfile trajectory.cfg\n");
    json("{\"kind\":\"volume\",\"residual_tolerance\":1e-6}");
    std::ofstream("output.cfg") << "preserve old result";
    std::ofstream("trajectory.cfg") << "preserve old trajectory";
    REQUIRE_THROWS_WITH(run(), Catch::Matchers::ContainsSubstring("frame 1 atom 1") &&
                              Catch::Matchers::ContainsSubstring("species index"));
    REQUIRE(bytes("output.cfg") == "preserve old result");
    REQUIRE(bytes("trajectory.cfg") == "preserve old trajectory");
}

TEST_CASE("V2 sequential lattice constraints preserve only the completed prefix on preparation failure", "[relax_constraints_cli][relax_stream][errors]") {
    Directory dir;
    auto second=structure(1.2); second.chemmap_[1]=1;
    input({structure(),second}); settings();
    json("{\"kind\":\"volume\",\"target\":45,\"residual_tolerance\":1e-6}");
    REQUIRE(run({"--seq-read"})==1);
    const auto result=output();
    REQUIRE(result.size()==1);
    REQUIRE(result[0].size()==2);
    REQUIRE(result[0].cell_.determinant()==Catch::Approx(45).margin(1e-6));

    // A first-frame preparation failure must still precede output access.
    input({second,structure()});
    std::ofstream("output.cfg") << "preserve old result";
    REQUIRE_THROWS_WITH(run({"--seq-read"}), Catch::Matchers::ContainsSubstring("frame 0 atom 1"));
    REQUIRE(bytes("output.cfg")=="preserve old result");
}

TEST_CASE("V2 constraint parsing fails before input and preserves existing output", "[relax_constraints_cli][errors]") {
    Directory dir; settings();
    std::ofstream("constraints.json") << "{\"version\":1,\"version\":1,\"constraints\":[]}";
    std::ofstream("output.cfg") << "preserve old result";
    REQUIRE_THROWS(run());
    REQUIRE(bytes("output.cfg")=="preserve old result");
}

TEST_CASE("V2 unsatisfied lattice constraints report budget exhaustion", "[relax_constraints_cli]") {
    Directory dir; input(); settings("relax:maxiteration 1\n");
    json("{\"kind\":\"volume\",\"target\":80,\"residual_tolerance\":1e-12},"
         "{\"kind\":\"lattice_length\",\"axis\":\"a\",\"target\":2,\"residual_tolerance\":1e-12}");
    REQUIRE(run()==2);
    auto result=output(); REQUIRE(result.size()==1);
    REQUIRE(result[0].cell_.allFinite());
}

TEST_CASE("V2 atomic JSON equalities move fixed-cell positions to physical targets", "[relax_constraints_cli][atomic]") {
    Directory dir;
    const bool angle = GENERATE(false, true);
    auto frame = structure();
    frame.cell_ = 20 * Eigen::Matrix3d::Identity();
    frame.pos_ = {{6,5,5}, {5,5,5}, {5,6,5}};
    frame.chemmap_ = {0,0,0};
    input({frame});
    settings("isif 2\nrelax:cartesian True\n");
    json(angle ? R"({"kind":"angle","atoms":[0,1,2],"target":60,"residual_tolerance":1e-6})" :
                 R"({"kind":"distance","atoms":[0,1],"target":1.5,"residual_tolerance":1e-6})");
    REQUIRE(run()==0);
    auto result = output(); REQUIRE(result.size()==1);
    REQUIRE(result[0].cell_.isApprox(frame.cell_,1e-12));
    const auto u = result[0].pos_[0]-result[0].pos_[1];
    if (angle) {
        const auto v = result[0].pos_[2]-result[0].pos_[1];
        REQUIRE(u.dot(v)/(u.norm()*v.norm()) == Catch::Approx(.5).margin(1e-6));
    } else REQUIRE(u.squaredNorm() == Catch::Approx(2.25).margin(1e-6));
}

TEST_CASE("V2 atomic constraints support position masks and an embedded frozen endpoint", "[relax_constraints_cli][atomic]") {
    Directory dir;
    const bool embedded = GENERATE(false, true);
    auto frame = structure();
    frame.cell_ = 20 * Eigen::Matrix3d::Identity();
    frame.pos_ = {{5,5,5}, {6,5,5}};
    input({frame});
    if (embedded) {
        frame.sel_dynpos_ = {Eigen::RowVector3d::Zero(), Eigen::RowVector3d(1,0,0)};
        GPIO io; REQUIRE(io.readmap("index.map"));
        io.force_write_all_to_one = true;
        std::vector<Atoms> frames{frame};
        REQUIRE(io.write(frames,"masked.poscars","poscar"));
        settings("isif 2\nrelax:cartesian True\nrelax:selective_dynamics True\n"
                 "input masked.poscars\ninput-format poscars\n");
    } else settings("isif 2\nrelax:cartesian True\nposition-mask 1 0 0\n");
    json(R"({"kind":"distance","atoms":[0,1],"target":1.5,"residual_tolerance":1e-6})");
    REQUIRE(run()==0);
    const auto result = output(); REQUIRE(result.size()==1);
    REQUIRE((result[0].pos_[1]-result[0].pos_[0]).squaredNorm() == Catch::Approx(2.25).margin(1e-6));
    if (embedded) REQUIRE(result[0].pos_[0].isApprox(frame.pos_[0],1e-10));
    for (int i=0;i<2;++i) {
        REQUIRE(result[0].pos_[i][1] == Catch::Approx(5).margin(1e-10));
        REQUIRE(result[0].pos_[i][2] == Catch::Approx(5).margin(1e-10));
    }
}

TEST_CASE("V2 atomic implicit targets bind per frame and sequential errors preserve the completed prefix", "[relax_constraints_cli][atomic][relax_stream]") {
    Directory dir;
    auto first = structure(), second = structure();
    first.cell_ = second.cell_ = 20 * Eigen::Matrix3d::Identity();
    first.pos_ = {{5,5,5},{6,5,5}};
    second.pos_ = {{5,5,5},{7,5,5}};
    input({first,second}); settings("isif 2\nrelax:cartesian True\n");
    json(R"({"kind":"distance","atoms":[0,1],"residual_tolerance":1e-6})");
    REQUIRE(run({"--seq-read"})==0);
    auto result = output(); REQUIRE(result.size()==2);
    REQUIRE((result[0].pos_[1]-result[0].pos_[0]).norm() == Catch::Approx(1).margin(1e-10));
    REQUIRE((result[1].pos_[1]-result[1].pos_[0]).norm() == Catch::Approx(2).margin(1e-10));
    second.pos_.resize(1); second.chemmap_.resize(1);
    input({first,second});
    REQUIRE(run({"--seq-read"})==1);
    result = output(); REQUIRE(result.size()==1);
    REQUIRE(result[0].pos_[0].isApprox(first.pos_[0],1e-10));
    // First-frame index failure precedes output access.
    input({second});
    std::ofstream("output.cfg") << "preserve old result";
    REQUIRE_THROWS_WITH(run(),Catch::Matchers::ContainsSubstring("frame 0"));
    REQUIRE(bytes("output.cfg")=="preserve old result");
}

TEST_CASE("V2 all-frozen atomic equality does not report an unmet target as converged", "[relax_constraints_cli][atomic]") {
    Directory dir;
    auto frame=structure();
    frame.cell_=20*Eigen::Matrix3d::Identity();
    frame.pos_={{5,5,5},{6,5,5}};
    frame.sel_dynpos_.assign(2,Eigen::RowVector3d::Zero());
    input({frame});
    GPIO io; REQUIRE(io.readmap("index.map")); io.force_write_all_to_one=true;
    std::vector<Atoms> frames{frame};
    REQUIRE(io.write(frames,"frozen.poscars","poscar"));
    settings("isif 2\nrelax:cartesian True\nrelax:selective_dynamics True\n"
             "input frozen.poscars\ninput-format poscars\nrelax:maxiteration 3\nrelax:constraint_loops 2\n");
    json(R"({"kind":"distance","atoms":[0,1],"target":1.5,"residual_tolerance":1e-6})");
    REQUIRE(run()==2);
    const auto result=output(); REQUIRE(result.size()==1);
    REQUIRE(result[0].pos_[0].isApprox(frame.pos_[0],1e-10));
    REQUIRE(result[0].pos_[1].isApprox(frame.pos_[1],1e-10));
}
