#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <limits>
namespace {
struct Workspace {
    std::filesystem::path old=std::filesystem::current_path(),path;
    Workspace() { char name[]="/tmp/imrmlp_findint_mic_XXXXXX"; auto p=mkdtemp(name); REQUIRE(p); path=p; std::filesystem::current_path(path); }
    ~Workspace() { CMDOption::global_cmdopts.reset(); std::filesystem::current_path(old); std::filesystem::remove_all(path); }
    int run(std::vector<std::string> args) {
        CMDOption::global_cmdopts.reset(); std::vector<char*> argv;
        for (auto &arg:args) argv.push_back(arg.data());
        return command_maps.at("findint").first(argv.size(),argv.data());
    }
};
Eigen::Matrix3d cell(bool orth=false) {
    Eigen::Matrix3d c; c<<8,0,0,6,6,0,4,3.4,6;
    if (orth) c=Eigen::Vector3d(8,6,6).asDiagonal(); return c;
}
std::vector<Eigen::RowVector3d> hosts() { return {{.4,.8,1.2},{4.2,1,3.4},{2.2,4.2,4.6},{8.4,3,.6}}; }
void input(const std::string &name,const Eigen::Matrix3d &c,double shift=0) {
    std::ofstream out(name);out<<std::setprecision(17)<<"4\nLattice=\"";
    for (int i=0;i<3;++i) for (int j=0;j<3;++j) out<<c(i,j)<<' ';
    out<<"\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n";
    for (auto p:hosts()) out<<"Si "<<p[0]+shift<<' '<<p[1]+shift<<' '<<p[2]+shift<<'\n';
}
std::vector<std::string> args(const std::string &source,const std::string &output,double tolerance=0) {
    return {"findint",source,"--input-format=extxyz","--sites-only","--output-format=extxyz","--output="+output,"--cluster-tol="+std::to_string(tolerance),"--min-dist=0.4","--top=1000"};
}
Atoms read(const std::string &name) {
    GPIO io;io.setmap();std::vector<Atoms> frames;
    REQUIRE(io.read(frames,name,"extxyz"));REQUIRE(frames.size()==1);return frames[0];
}
// Independent full Cartesian integer-image enumeration, no native MIC helpers.
double distance(const Eigen::RowVector3d &a,const Eigen::RowVector3d &b,const Eigen::Matrix3d &c) {
    double best=std::numeric_limits<double>::infinity();
    for (int x=-5;x<=5;++x) for (int y=-5;y<=5;++y) for (int z=-5;z<=5;++z)
        best=std::min(best,(a-b+Eigen::RowVector3d(x,y,z)*c).norm());
    return best;
}
double cavity(const Eigen::RowVector3d &point,const Eigen::Matrix3d &c) {
    double best=std::numeric_limits<double>::infinity();
    for (const auto &host:hosts()) best=std::min(best,distance(point,host,c));
    return best;
}
void same_sites(const Atoms &a,const Atoms &b,const Eigen::Matrix3d &c,double shift=0) {
    REQUIRE(a.size()==b.size());
    for (const auto &p:a.pos_) {
        double best=std::numeric_limits<double>::infinity();
        for (const auto &q:b.pos_) best=std::min(best,distance(p+Eigen::RowVector3d::Constant(shift),q,c));
        CHECK(best<1e-9);
    }
}
std::string contents(const std::string &name) { std::ifstream in(name); return {std::istreambuf_iterator<char>(in),{}}; }
}
TEST_CASE("findint ranks and selects actual nearest-image cavities in skew crystals", "[apps][findint][mic]") {
    const bool orth=GENERATE(false,true);Workspace w;const auto c=cell(orth);input("input.xyz",c);
    const auto source=contents("input.xyz");REQUIRE(w.run(args("input.xyz","all.xyz"))==0);
    const auto all=read("all.xyz");REQUIRE(all.size()>1);
    double best=0;for (const auto &p:all.pos_) best=std::max(best,cavity(p,c));
    if (!orth) CHECK(best==Catch::Approx(3.843444395228492).margin(1e-10));
    auto top=args("input.xyz","top.xyz");top.push_back("--top=1");
    REQUIRE(w.run(top)==0);const auto selected=read("top.xyz");REQUIRE(selected.size()==1);
    CHECK(cavity(selected.pos_[0],c)==Catch::Approx(best).margin(1e-10));
    CHECK(cavity(all.pos_[0],c)==Catch::Approx(best).margin(1e-10));
    CHECK(contents("input.xyz")==source);
}
TEST_CASE("findint min-dist rejects skew sites below Cartesian image threshold", "[apps][findint][mic]") {
    Workspace w;const auto c=cell();input("input.xyz",c);auto command=args("input.xyz","filtered.xyz");
    command.push_back("--min-dist=3.8");REQUIRE(w.run(command)==0);const auto filtered=read("filtered.xyz");
    REQUIRE(filtered.size()==1);
    for (const auto &p:filtered.pos_) CHECK(cavity(p,c)>=3.8-1e-10);
    CHECK(cavity(filtered.pos_[0],c)==Catch::Approx(3.843444395228492).margin(1e-10));
}
TEST_CASE("findint skew clustering uses the same exact metric for grouping and aligned centers", "[apps][findint][mic]") {
    Workspace w;const auto c=cell();input("input.xyz",c);input("shift.xyz",c,.37);
    auto fast=args("input.xyz","fast.xyz",.8);REQUIRE(w.run(fast)==0);const auto original=read("fast.xyz");
    auto slow=args("input.xyz","slow.xyz",.8);slow.push_back("--no-fast-cluster");
    REQUIRE(w.run(slow)==0);same_sites(original,read("slow.xyz"),c);
    REQUIRE(w.run(args("shift.xyz","shift.xyz.out",.8))==0);same_sites(original,read("shift.xyz.out"),c,.37);
    auto shifted_slow=args("shift.xyz","shift-slow.xyz",.8);shifted_slow.push_back("--no-fast-cluster");
    REQUIRE(w.run(shifted_slow)==0);same_sites(original,read("shift-slow.xyz"),c,.37);
}
