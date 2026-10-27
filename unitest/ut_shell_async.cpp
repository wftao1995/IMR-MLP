#include <catch2/catch_all.hpp>
#include "potentials/potential_shell.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>
#include <unistd.h>

namespace {
namespace fs=std::filesystem;
struct AsyncDirectory {
    fs::path original=fs::current_path(),path;
    AsyncDirectory() {
        char pattern[]="/tmp/imrmlp_shell_async_XXXXXX";
        const auto *directory=mkdtemp(pattern);
        if(!directory) throw std::runtime_error("Cannot create async shell test directory");
        path=directory;fs::current_path(path); write("index.map","X\n");
    }
    ~AsyncDirectory() { fs::current_path(original);fs::remove_all(path); }
    static void write(const fs::path &file,const std::string &value) {
        std::ofstream output(file);output<<value;
        if(!output) throw std::runtime_error("Cannot write async shell test file");
    }
};
GPSettings settings() {
    GPSettings s;
    s.Modify("index","index.map"); s.Modify("runscript","run.sh");
    s.Modify("write_posfile","input.xyz"); s.Modify("write_posformat","extxyz");
    s.Modify("read_outfile","output.xyz"); s.Modify("read_outformat","extxyz");
    return s;
}
Atoms atom(double energy) {
    Atoms a;a.cell_=7.*Eigen::Matrix3d::Identity();a.pos_={{1.,2.,3.}};a.chemmap_={0};
    a.energy_=energy;a.have_energy=true;a.forces_={{.2,-.4,.7}};return a;
}
void wait_for_file(const fs::path &file) {
    for(int i=0;i<500 && !fs::exists(file);++i) std::this_thread::sleep_for(std::chrono::milliseconds(10));
    REQUIRE(fs::exists(file));
}
std::string failure(const std::shared_ptr<Potentials::ShellJob> &job) {
    try { static_cast<void>(job->result()); }
    catch(const std::exception &error) { return error.what(); }
    return {};
}
std::string read_line(const fs::path &path) { std::ifstream file(path);std::string text;std::getline(file,text);return text; }
}

TEST_CASE("Shell jobs own settings input and mapping after their calculator is destroyed", "[shell][async-shell]") {
    AsyncDirectory workspace;
    AsyncDirectory::write("run.sh",
        "touch \"$1/started\"\n"
        "for i in $(seq 1 500); do [ -f \"$1/release\" ] && break; sleep .01; done\n"
        "[ -f \"$1/release\" ] || exit 21\n"
        "pwd > \"$1/submission-cwd\"\ncp \"$1/input.xyz\" \"$1/output.xyz\"\n");
    std::shared_ptr<Potentials::ShellJob> job;
    auto input=atom(-3.5);
    {
        Potentials::PotShellCalc calculator(settings());
        job=calculator.submit(input);
        GPSettings changed;changed.Modify("runscript","missing.sh");changed.Modify("read_outfile","missing.xyz");
        calculator.ApplySettings(changed);
    }
    input.energy_=91.;input.pos_[0][0]=12.;
    wait_for_file(fs::path(job->run_directory())/"started");
    REQUIRE_FALSE(job->ready());
    AsyncDirectory::write("index.map","H\n"); // The owned GPIO mapping must still recognize X.
    fs::create_directory("different-cwd");fs::current_path("different-cwd");
    AsyncDirectory::write(fs::path(job->run_directory())/"release","");
    auto first=job->result();
    REQUIRE(job->ready());REQUIRE(first.energy_==Catch::Approx(-3.5));
    REQUIRE(first.pos_[0][0]==Catch::Approx(1.));
    REQUIRE(first.forces_[0].isApprox(Eigen::Vector3d(.2,-.4,.7),1e-12));
    REQUIRE(read_line(fs::path(job->run_directory())/"submission-cwd")==workspace.path.string());
    REQUIRE(fs::current_path()==workspace.path/"different-cwd");
    first.energy_=100.;first.pos_[0][0]=-20.;
    const auto second=job->result();
    REQUIRE(second.energy_==Catch::Approx(-3.5));REQUIRE(second.pos_[0][0]==Catch::Approx(1.));
}

TEST_CASE("Concurrent shell jobs have independent directories and repeatable results", "[shell][async-shell]") {
    AsyncDirectory workspace;
    fs::create_directory("scripts space's");
    AsyncDirectory::write("scripts space's/run.sh",
        "[ \"$#\" -eq 1 ] || exit 31\n"
        "cp \"$1/input.xyz\" \"$1/output.xyz\"\nprintf '%s\\n' \"$1\" > \"$1/job-arg\"\n");
    auto config=settings();config.Modify("runscript","scripts space's/run.sh");
    Potentials::PotShellCalc calculator(config);
    std::vector<std::shared_ptr<Potentials::ShellJob>> jobs;
    for(int i=0;i<4;++i) jobs.push_back(calculator.submit(atom(-i-1.)));
    for(int i=0;i<4;++i) {
        const auto output=jobs[i]->result();REQUIRE(output.energy_==Catch::Approx(-i-1.));
        REQUIRE(read_line(fs::path(jobs[i]->run_directory())/"job-arg")==jobs[i]->run_directory());
        for(int j=0;j<i;++j) REQUIRE(jobs[i]->run_directory()!=jobs[j]->run_directory());
    }
    REQUIRE(fs::current_path()==workspace.path);
}

TEST_CASE("Shell job failures repeat from result without returning stale outputs", "[shell][async-shell]") {
    const auto stage=GENERATE(std::string("input"),std::string("pre"),std::string("run"),std::string("output"),std::string("energy"),std::string("post"));
    AsyncDirectory workspace;
    AsyncDirectory::write("pre.sh",stage=="pre" ? "exit 7\n" : "touch \"$1/pre-ran\"\n");
    AsyncDirectory::write("run.sh",stage=="run" ? "exit 9\n" : stage=="output" ? "exit 0\n" : "cp \"$1/input.xyz\" \"$1/output.xyz\"\n");
    AsyncDirectory::write("post.sh",stage=="post" ? "exit 11\n" : "touch \"$1/post-ran\"\n");
    auto config=settings();config.Modify("prescript","pre.sh");config.Modify("postscript","post.sh");
    if(stage=="input") config.Modify("write_posfile","missing/input.xyz");
    if(stage=="energy") { config.Modify("read_energy","true");config.Modify("engfile","missing-energy.dat"); }
    Potentials::PotShellCalc calculator(config);const auto job=calculator.submit(atom(-3.));
    const auto first=failure(job);REQUIRE_FALSE(first.empty());REQUIRE(job->ready());
    REQUIRE(failure(job)==first);
    REQUIRE_FALSE(fs::exists(fs::path(job->run_directory())/"post-ran"));
}

TEST_CASE("Shell job destruction joins successful and failing work without throwing", "[shell][async-shell]") {
    const bool fail=GENERATE(false,true);
    AsyncDirectory workspace;
    AsyncDirectory::write("run.sh",
        std::string("sleep .05\ncp \"$1/input.xyz\" \"$1/output.xyz\"\ntouch \"$1/completed\"\n")+(fail ? "exit 9\n" : ""));
    Potentials::PotShellCalc calculator(settings());auto job=calculator.submit(atom(-2.));
    const auto directory=job->run_directory();
    REQUIRE_NOTHROW(job.reset());
    REQUIRE(fs::exists(fs::path(directory)/"completed"));
}

TEST_CASE("Shell job snapshots preserve sparse species IDs and their inverse mapping", "[shell][async-shell]") {
    AsyncDirectory workspace;
    AsyncDirectory::write("run.sh","cp \"$1/input.xyz\" \"$1/output.xyz\"\n");
    Potentials::PotShellCalc calculator(settings());
    const std::map<int,std::string> mapping{{2,"He"},{7,"Ne"}};
    calculator.set_element_map(mapping);
    auto input=atom(-3.);
    input.pos_.push_back(Eigen::RowVector3d(2.,3.,4.));
    input.forces_.push_back(Eigen::RowVector3d(-.2,.4,-.7));
    input.chemmap_={7,2};
    const auto job=calculator.submit(input);
    calculator.set_element_map({{0,"H"}});
    const auto output=job->result();
    REQUIRE(output.chemmap_==input.chemmap_);
    REQUIRE(output.energy_==Catch::Approx(input.energy_));
    REQUIRE(output.pos_==input.pos_);
    GPIO io;
    io.setmap(mapping);
    REQUIRE(io.getmap()==mapping);
    REQUIRE(io.getrevmap().at("He")==2);
    REQUIRE(io.getrevmap().at("Ne")==7);
}
