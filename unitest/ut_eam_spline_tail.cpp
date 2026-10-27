#include <catch2/catch_all.hpp>
#include "potentials/potential_eamalloy_spline.h"
#include "potentials/potential_eamfs_spline.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#ifdef USE_OMP
#include <omp.h>
#endif
using namespace Potentials;
namespace {
struct Fixture {
    std::filesystem::path directory;std::string file;
    Fixture(bool fs){char name[]="/tmp/imrmlp_eam_tail_XXXXXX";const char *p=mkdtemp(name);if(!p)throw std::runtime_error("mkdtemp failed");directory=p;file=(directory/"model.table").string();
        std::ofstream out(file);out<<std::setprecision(17)<<"a\nb\nc\n2 H He\n9 1 5 1 5\n";
        for(int neighbor=0;neighbor<2;++neighbor){out<<neighbor+1<<" 1 1 fcc\n";for(int k=0;k<9;++k)out<<(.1+.05*neighbor)*k*k<<' ';out<<'\n';
            for(int host=0;host<(fs?2:1);++host){for(int r=0;r<5;++r)out<<(.25+.25*neighbor+.5*host)*r<<' ';out<<'\n';}}
        for(int pair=0;pair<3;++pair){for(int r=0;r<5;++r)out<<(.3+.2*pair)*r*r<<' ';out<<'\n';}}
    ~Fixture(){std::filesystem::remove_all(directory);}
    std::unique_ptr<AnyPotential> make(bool fs){std::map<std::string,int> map;if(fs)return std::make_unique<EAMFsSpline>(file,map);return std::make_unique<EAMAlloySpline>(file,map);}
};
Atoms geometry(bool mixed){Atoms a;a.cell_=20*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};a.pos_={{0,0,0},{4.5,0,0}};a.chemmap_={0,1};if(mixed){a.pos_.push_back({1.2,1.1,.4});a.chemmap_.push_back(0);}return a;}
void evaluate(AnyPotential &p,Atoms &a,bool parallel){if(parallel)p.getEFSOMP(a);else p.getEFS(a);}
}
TEST_CASE("Ordinary EAM radial tail forces differentiate unchanged clamped energy","[eam][tail][batch34]") {
    const bool fs=GENERATE(false,true),parallel=GENERATE(false,true),mixed=GENERATE(false,true);
    Fixture files(fs);auto p=files.make(fs);auto a=geometry(mixed);evaluate(*p,a,parallel);
    if(!mixed){const double rho0=2.,rho1=fs?3.:1.;
        const double exact=.1*rho0*rho0+.15*rho1*rho1+.5*16/4.5;
        REQUIRE(std::abs(a.energy_-exact)<1e-12);}
    for(double h:{1e-5,5e-6})for(int i=0;i<a.size();++i)for(int d=0;d<3;++d){auto plus=geometry(mixed),minus=plus;plus.pos_[i][d]+=h;minus.pos_[i][d]-=h;evaluate(*p,plus,parallel);evaluate(*p,minus,parallel);
        REQUIRE(std::abs(a.forces_[i][d]+(plus.energy_-minus.energy_)/(2*h))<1e-8);}
    for(double h:{1e-5,5e-6}){auto plus=geometry(mixed),minus=plus;for(int i=0;i<a.size();++i){plus.pos_[i][0]*=1+h;minus.pos_[i][0]*=1-h;}evaluate(*p,plus,parallel);evaluate(*p,minus,parallel);
        REQUIRE(std::abs(a.stress_(0,0)+(plus.energy_-minus.energy_)/(2*h))<1e-8);}
    p->add_efs=true;auto doubled=a;evaluate(*p,doubled,parallel);REQUIRE(std::abs(doubled.energy_-2*a.energy_)<1e-12);
    for(int i=0;i<a.size();++i)REQUIRE((doubled.forces_[i]-2*a.forces_[i]).norm()<1e-12);
}
