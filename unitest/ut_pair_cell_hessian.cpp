#include <catch2/catch_all.hpp>
#include "potentials/potential_zbl.h"
#include "unitest/pair_cell_fixture.h"
#include "apps/apps.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
using namespace Potentials;
namespace {
GPSettings zbl_settings() { GPSettings s;s.Modify("z","2 3");s.Modify("r0","1.2");s.Modify("r1","4.5");return s; }
}
TEST_CASE("ZBL native cell atomic and mixed curvature follows total energy", "[zbl][cell][batch07]") {
    const bool partial=GENERATE(false,true); const double step=GENERATE(.001,.0005);
    PotentialZBL pot(zbl_settings()); AnyPotential &api=pot; auto a=PairCellTest::geometry(partial);
    auto source=a; api.getEFS(source);
    HessianMatrix h;Eigen::MatrixXd cell,cross; Neighborhoods nbhs(a,pot.cutoff);
    api.getEFSHessianCellOMP(a,nbhs,h,cell,cross);
    const auto oracle=PairCellTest::energy_fd(api,a,step,true);
    REQUIRE(cell.norm()>.01);REQUIRE(cross.norm()>.01);
    REQUIRE((PairCellTest::native_matrix(h,cell,cross)-oracle).norm()<60*step*step+5e-7);
    REQUIRE(std::abs(a.energy_-source.energy_)<1e-12);
    REQUIRE((a.stress_-source.stress_).norm()<1e-12);
    const auto first=cell;api.add_efs=true;api.getEFSHessianCellSymmOMP(a,h,cell,cross);
    REQUIRE((cell-first).norm()<1e-12); REQUIRE(std::abs(a.energy_-2*source.energy_)<1e-12);
    for(int i=0;i<a.size();++i)REQUIRE((a.forces_[i]-2*source.forces_[i]).norm()<1e-12);
}
TEST_CASE("ZBL analytical cell consumer maps fixed positions strain and cutoff", "[zbl][cell][batch07]") {
    const bool strain=GENERATE(false,true);const double cutoff=GENERATE(0.,3.1);
    PotentialZBL pot(zbl_settings()); AnyPotential &api=pot;auto a=PairCellTest::geometry(true);
    ComputeHessian c;c.method=ComputeHessian::HessianMethod::Analytic;c.use_symmetry=false;
    c.request_cell=true;c.cell_strain=strain;c.engineering_shear=true;c.set_cutoff(cutoff);
    Eigen::MatrixXd result;c.compute(a,result,&api);
    for(double step:{.001,.0005}) {
        const auto oracle=PairCellTest::energy_fd(api,a,step,false,strain,cutoff);
        REQUIRE((result-oracle).norm()<150*step*step+5e-7);
    }
    c.request_response_pos_cell=false;c.compute(a,result,&api);
    REQUIRE(result.topRightCorner(strain?6:9,3*a.size()).isZero(1e-14));
}
TEST_CASE("ZBL cell curvature keeps periodic self images and rejects singular cell", "[zbl][cell][batch07]") {
    PotentialZBL pot(zbl_settings());Atoms a;a.cell_=3.2*Eigen::Matrix3d::Identity();a.pos_={{.1,.2,.3}};a.chemmap_={0};
    HessianMatrix h;Eigen::MatrixXd cell,cross;pot.getEFSHessianCellSymmOMP(a,h,cell,cross);
    REQUIRE(cell.norm()>.01);REQUIRE(h(0,0).isZero(1e-13));REQUIRE(cross.isZero(1e-13));
    const auto oracle=PairCellTest::energy_fd(pot,a,.0005,true);
    REQUIRE((PairCellTest::native_matrix(h,cell,cross)-oracle).norm()<2e-5);
    a.cell_.setZero(); REQUIRE_THROWS(pot.getEFSHessianCellSymmOMP(a,h,cell,cross));
}
TEST_CASE("calc_elastic auto uses ZBL cell provider and rejects MTP-only corrections", "[zbl][cell][cli][batch07]") {
    const auto original=std::filesystem::current_path();char pattern[]="/tmp/imrmlp_zbl_cell_XXXXXX";
    const char *made=mkdtemp(pattern);REQUIRE(made);const std::filesystem::path path=made;
    struct Guard {std::filesystem::path cwd,path;CMDOption::CommandLineOptions saved=CMDOption::global_cmdopts;
        ~Guard(){CMDOption::global_cmdopts=saved;std::filesystem::current_path(cwd);std::filesystem::remove_all(path);}} guard{original,path};
    std::filesystem::current_path(path);std::ofstream("index.map")<<"He\n";
    std::ofstream("pot.ini")<<"pot:pottype zbl\npot:z 2\npot:r0 1.2\npot:r1 4.5\n";
    Atoms a;a.cell_ << 0,2.,2.,2.,0,2.,2.,2.,0;a.pos_={{0,0,0}};a.chemmap_={0};
    GPIO io;io.readmap("index.map");REQUIRE(io.write_one(a,"input.cfg","mlpcfg"));
    auto run=[&](std::string correction){
        CMDOption::global_cmdopts=CMDOption::CommandLineOptions{};
        std::vector<std::string> args={"mlpcalccellhess","--potfile=pot.ini","--pottype=zbl","--input=input.cfg","--input-format=mlpcfg","--index=index.map","--relax-correction=false","--elastic-output=elastic.out"};
        if(!correction.empty())args.push_back(correction);
        std::vector<char*> argv;for(auto &s:args)argv.push_back(s.data());return calc_elastic(argv.size(),argv.data());};
    REQUIRE(run("")==0);REQUIRE(std::filesystem::file_size("elastic.out")>0);
    REQUIRE_THROWS_WITH(run("--correction-l1=true"),Catch::Matchers::ContainsSubstring("require a PotMTP"));
    REQUIRE_THROWS_WITH(run("--correction-l2=true"),Catch::Matchers::ContainsSubstring("require a PotMTP"));
}
