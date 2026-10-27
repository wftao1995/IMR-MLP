#include <catch2/catch_all.hpp>
#include "mindriver/neb.h"
#include <filesystem>
#include <fstream>
#include <chrono>

namespace {
struct Directory {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("neb-release-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ~Directory() {std::filesystem::remove_all(path);}
    std::string report() const {std::ifstream in(path/"report.json");return {std::istreambuf_iterator<char>(in),{}};}
};
class BranchPotential: public Potentials::AnyPotential {
public:
    bool unstable=false, fail=false, zero=false;
    void calculate(Atoms &a) {
        if(fail) throw std::runtime_error("deliberate calculator failure");
        a.resetEFS();if(zero)return;
        const double t=a.pos_[0][1]+a.pos_[1][1]-3;
        const double z=(a.pos_[0][2]-a.pos_[1][2])/2;
        const double k=unstable?.2-2*(1-t*t):1.;
        a.energy_=(t*t-1)*(t*t-1)+k*z*z+z*z*z*z;
        const double dy=4*t*(t*t-1)+(unstable?4*t*z*z:0);
        const double dz=2*k*z+4*z*z*z;
        a.forces_[0][1]=a.forces_[1][1]=-dy;
        a.forces_[0][2]=-dz/2;a.forces_[1][2]=dz/2;
    }
    void getEFS(Atoms &a,AnyNeighborhoods &) override {calculate(a);}
    void getEFSOMPthread(Atoms &a) override {calculate(a);}
    void getEFSOMP(Atoms &a) override {calculate(a);}
};
std::vector<Atoms> path() {
    Atoms a;a.cell_=Eigen::Vector3d(6,7,8).asDiagonal();a.chemmap_={0,0};
    a.pos_={Eigen::RowVector3d(1.5,1,2),Eigen::RowVector3d(4.5,1,2)};
    auto b=a;for(auto &r:b.pos_)r[1]=2;
    return {a,Atoms{},Atoms{},Atoms{},b};
}
}

TEST_CASE("Symmetry release distinguishes stable paths and lower branches without mutating the main result", "[neb][release]") {
    for(bool unstable:{false,true}) {
        BranchPotential potential;potential.unstable=unstable;
        auto input=path();NEBDriver d;d.pot=&potential;d.keep_symm=true;d.climb=true;
        d.solver_policy="stable-v1";d.neb_ftol=1e-4;d.init(input);
        REQUIRE(d.run("fire",5,.04)==0);
        const auto original=d.images;const auto x=d.x;const auto calls=d.image_force_evaluations;
        Directory output;d.symmetry_release=output.path.string();d.symmetry_release_trials=1;
        d.symmetry_release_amplitude=unstable?1.:.01;
        d.symmetry_release_seed=5;
        d.symmetry_release_steps=1000;d.symmetry_release_evaluations=10001;
        NEBDriver::RunOptions options;options.max_move=.04;d.check_symmetry_release(options);
        INFO(output.report());
        REQUIRE(d.symmetry_release_status==(unstable?"lower_barrier_found":"no_lower_barrier_observed"));
        REQUIRE(d.image_force_evaluations==calls);REQUIRE(d.converged);REQUIRE(d.x==x);
        for(int i=0;i<original.size();++i) {REQUIRE(d.images[i].pos_==original[i].pos_);REQUIRE(d.images[i].energy_==original[i].energy_);}
        REQUIRE(std::filesystem::exists(output.path/"trial-1/final/image-2.vasp"));
    }
}

TEST_CASE("Release budgets, failures, skips and opt-in preserve the main calculation", "[neb][release]") {
    BranchPotential potential;potential.unstable=true;auto input=path();
    NEBDriver d;d.pot=&potential;d.keep_symm=true;d.climb=true;d.solver_policy="stable-v1";d.init(input);
    d.run("fire",5,.03);NEBDriver::RunOptions options;
    d.check_symmetry_release(options);REQUIRE(d.symmetry_release_status=="disabled");
    SECTION("independent EFS budget") {
        Directory output;d.symmetry_release=output.path.string();d.symmetry_release_evaluations=1;
        d.symmetry_release_trials=1;d.check_symmetry_release(options);
        REQUIRE(d.symmetry_release_status=="inconclusive");
        REQUIRE(output.report().find("\"force_evaluations\":1")!=std::string::npos);
    }
    SECTION("failed calculator") {
        Directory output;d.symmetry_release=output.path.string();potential.fail=true;
        d.check_symmetry_release(options);REQUIRE(d.symmetry_release_status=="failed");
        REQUIRE(output.report().find("deliberate calculator failure")!=std::string::npos);
    }
    SECTION("output failure") {
        Directory output;std::filesystem::create_directory(output.path);
        d.symmetry_release=output.path.string();d.check_symmetry_release(options);
        REQUIRE(d.symmetry_release_status=="failed");
    }
    SECTION("unconverged main") {
        Directory output;d.symmetry_release=output.path.string();d.converged=false;d.check_symmetry_release(options);
        REQUIRE(d.symmetry_release_status=="skipped");
    }
    REQUIRE(d.images.front().pos_==input.front().pos_);
    REQUIRE(d.images.back().pos_==input.back().pos_);
}

TEST_CASE("Cell-only symmetry release bounds deformation and preserves endpoint cells", "[neb][release][vcneb]") {
    BranchPotential potential;potential.zero=true;
    Atoms a;a.cell_=4*Eigen::Matrix3d::Identity();a.chemmap_={0};a.pos_={Eigen::RowVector3d::Zero()};
    auto b=a;b.cell_=4.2*Eigen::Matrix3d::Identity();std::vector<Atoms> input{a,Atoms{},b};
    NEBDriver d;d.pot=&potential;d.vcneb=true;d.keep_symm=true;d.solver_policy="stable-v1";d.init(input);
    REQUIRE(d.run("fire",5,.03)==0);
    Directory output;d.symmetry_release=output.path.string();d.symmetry_release_amplitude=0;
    d.symmetry_release_trials=1;d.symmetry_release_evaluations=1;
    NEBDriver::RunOptions options;d.check_symmetry_release(options);
    INFO(output.report());REQUIRE(d.symmetry_release_status!="failed");
    GPIO io;std::vector<Atoms> perturbed;REQUIRE(io.read(perturbed,(output.path/"trial-1/image-1.vasp").string(),"POSCAR"));
    REQUIRE(perturbed.size()==1);
    const double change=(d.images[1].cell_.inverse()*perturbed[0].cell_-Eigen::Matrix3d::Identity()).norm();
    REQUIRE(change>.0009);REQUIRE(change<=.00100001);
    REQUIRE(d.images.front().cell_==a.cell_);REQUIRE(d.images.back().cell_.isApprox(b.cell_));
}

TEST_CASE("Release seeds reproduce the decoded trial path", "[neb][release]") {
    BranchPotential potential;auto input=path();NEBDriver d;d.pot=&potential;
    d.keep_symm=true;d.climb=true;d.solver_policy="stable-v1";d.init(input);d.run("fire",5,.03);
    d.symmetry_release_trials=1;d.symmetry_release_evaluations=1;
    Directory first,second,third;NEBDriver::RunOptions options;
    d.symmetry_release=first.path.string();d.symmetry_release_seed=17;d.check_symmetry_release(options);
    d.symmetry_release=second.path.string();d.check_symmetry_release(options);
    d.symmetry_release=third.path.string();d.symmetry_release_seed=18;d.check_symmetry_release(options);
    GPIO io;std::vector<Atoms> a,b,c;
    REQUIRE(io.read(a,(first.path/"trial-1/image-2.vasp").string(),"POSCAR"));
    REQUIRE(io.read(b,(second.path/"trial-1/image-2.vasp").string(),"POSCAR"));
    REQUIRE(io.read(c,(third.path/"trial-1/image-2.vasp").string(),"POSCAR"));
    REQUIRE(a[0].pos_==b[0].pos_);REQUIRE(a[0].pos_!=c[0].pos_);
    double maximum=0;for(int atom=0;atom<2;++atom) maximum=std::max(maximum,(a[0].pos_[atom]-d.images[2].pos_[atom]).norm());
    REQUIRE(maximum<=d.symmetry_release_amplitude*(1+1e-8));REQUIRE(maximum>0);
}

TEST_CASE("Release does not truncate the main driver's debug log", "[neb][release]") {
    BranchPotential potential;auto input=path();NEBDriver d;d.pot=&potential;
    d.keep_symm=true;d.climb=true;d.solver_policy="stable-v1";d.init(input);d.run("fire",5,.03);
    Directory scratch;std::filesystem::create_directory(scratch.path);
    struct RestoreDirectory {
        std::filesystem::path old=std::filesystem::current_path();
        ~RestoreDirectory(){std::filesystem::current_path(old);}
    } restore;
    std::filesystem::current_path(scratch.path);
    std::ofstream("debug_neb_stream.log")<<"main diagnostic sentinel\n";
    d.debug_level=1;
    Directory output;d.symmetry_release=output.path.string();d.symmetry_release_trials=0;
    NEBDriver::RunOptions options;d.check_symmetry_release(options);
    std::ifstream log("debug_neb_stream.log");std::string line;std::getline(log,line);
    REQUIRE(line=="main diagnostic sentinel");
    std::filesystem::remove("debug_neb_stream.log");
}
