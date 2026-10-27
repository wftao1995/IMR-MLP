#include <catch2/catch_all.hpp>
#include "mindriver/neb.h"
#include "mindriver/neb_symmetry.h"
#include "mindriver/neb_evaluate.h"
#include <atomic>

namespace {
class SymmetryField : public Potentials::AnyPotential {
public:
    bool force=false, harmonic=false;
    std::atomic<bool> symmetry_violation{false};
    std::atomic<int> image_calls{0}, atom_calls{0};
    void calculate(Atoms &a) {
        a.resetEFS();
        if(force) for(int i=0;i<a.size();++i) {a.energy_-=a.pos_[i][0];a.forces_[i][0]=1;}
        if(harmonic) {
            if(std::abs(a.pos_[0][0]+a.pos_[1][0]-6)>1e-9 || std::abs(a.pos_[0][2]-1.52)>1e-9) symmetry_violation=true;
            for(int i=0;i<a.size();++i) {a.energy_+=.5*a.pos_[i][1]*a.pos_[i][1];a.forces_[i][1]=-a.pos_[i][1];}
        }
    }
    void getEFS(Atoms &a,AnyNeighborhoods &) override {calculate(a);}
    void getEFSOMPthread(Atoms &a) override {++image_calls;calculate(a);}
    void getEFSOMP(Atoms &a) override {++atom_calls;calculate(a);}
};
std::vector<Atoms> pair_path() {
    Atoms a; a.cell_=Eigen::Vector3d(6,7,8).asDiagonal();
    a.chemmap_={0,0}; a.pos_={Eigen::RowVector3d(.23,.17,.19)*a.cell_,Eigen::RowVector3d(.77,.17,.19)*a.cell_};
    auto b=a;
    b.pos_={Eigen::RowVector3d(.28,.21,.19)*b.cell_,Eigen::RowVector3d(.72,.21,.19)*b.cell_};
    return {a,Atoms{},Atoms{},Atoms{},b};
}
}

TEST_CASE("NEB OpenMP evaluator selects one parallel level and preserves geometry", "[neb][parallel]") {
    SymmetryField potential; auto path=pair_path(); path={path.front(),path.back()};
    path[0].pos_[0][0]+=12; const auto original=path[0].pos_;
    evaluate_neb_images(potential,path,0,2,"images");
    REQUIRE(potential.image_calls==2); REQUIRE(potential.atom_calls==0);
    REQUIRE(path[0].pos_[0].isApprox(original[0]));
    evaluate_neb_images(potential,path,0,2,"atoms");
    REQUIRE(potential.image_calls==2); REQUIRE(potential.atom_calls==2);
    REQUIRE_THROWS_AS(evaluate_neb_images(potential,path,0,2,"nested"),std::invalid_argument);
}

TEST_CASE("NEB common path symmetry uses consistent atom mappings and projects forbidden force", "[neb][symmetry]") {
    SymmetryField potential; potential.force=true;
    auto path=pair_path(); NEBDriver d; d.pot=&potential; d.keep_symm=true;
    d.solver_policy="stable-v1"; d.init(path);
    NEBPathSymmetry symmetry(d,"");
    REQUIRE(symmetry.common_operations.size()>1);
    REQUIRE(symmetry.common_operations.size()<symmetry.initial.n_operations);
    Eigen::RowVectorXd v=Eigen::RowVectorXd::Random(d.x.cols());
    auto p=symmetry.project(v);
    REQUIRE(symmetry.project(p).isApprox(p,1e-10));
    REQUIRE(std::abs(p.dot(v-p))<1e-10);
    REQUIRE(d.run("fire",4,.03)==0);
    REQUIRE(d.max_neb_force<1e-10);
    for(const auto &image:d.images) {
        REQUIRE(image.pos_[0][0]+image.pos_[1][0]==Catch::Approx(6));
        REQUIRE(image.pos_[0][2]==Catch::Approx(1.52));
    }
    REQUIRE(d.images.front().pos_==path.front().pos_);
    REQUIRE(d.images.back().pos_==path.back().pos_);
}

TEST_CASE("NEB perturbations are bounded reproducible smooth and break the common group", "[neb][symmetry]") {
    SymmetryField potential; auto path=pair_path();
    NEBDriver reference; reference.pot=&potential; reference.init(path);
    auto make=[&](int seed) {
        NEBDriver d; d.pot=&potential; d.perturb_amplitude=.04; d.perturb_seed=seed; d.init(path);
        return d.images;
    };
    auto a=make(17),b=make(17),c=make(18);
    double maximum=0,difference=0;
    for(int i=0;i<a.size();++i) for(int atom=0;atom<2;++atom) {
        REQUIRE(a[i].pos_[atom].isApprox(b[i].pos_[atom],1e-14));
        maximum=std::max(maximum,(a[i].pos_[atom]-reference.images[i].pos_[atom]).norm());
        difference+=(a[i].pos_[atom]-c[i].pos_[atom]).squaredNorm();
    }
    REQUIRE(maximum==Catch::Approx(.04).margin(1e-12)); REQUIRE(difference>1e-6);
    REQUIRE(a.front().pos_==reference.images.front().pos_);
    REQUIRE(a.back().pos_==reference.images.back().pos_);
    NEBDriver perturbed; perturbed.pot=&potential; perturbed.check_mic=false; perturbed.init(a);
    REQUIRE(NEBPathSymmetry(perturbed,"").common_operations.size()<NEBPathSymmetry(reference,"").common_operations.size());
}

TEST_CASE("VC NEB symmetry projection respects the scaled cell metric", "[neb][symmetry][vcneb]") {
    SymmetryField potential;
    Atoms a; a.cell_=4*Eigen::Matrix3d::Identity(); a.chemmap_={0};a.pos_={Eigen::RowVector3d::Zero()};
    auto b=a; b.cell_=5*Eigen::Matrix3d::Identity(); std::vector<Atoms> path{a,Atoms{},b};
    NEBDriver d; d.pot=&potential;d.vcneb=true;d.keep_symm=true;d.apply_scal=true;
    d.ciival={100,200,300,50,60,70};d.init(path);
    NEBPathSymmetry symmetry(d,"");
    REQUIRE(symmetry.cell_dofs()==1); REQUIRE(symmetry.atomic_dofs()==0);
    const Eigen::RowVectorXd v=Eigen::RowVectorXd::Random(d.x.cols());
    const auto p=symmetry.project(v);
    REQUIRE(symmetry.project(p).isApprox(p,1e-10));
    REQUIRE(std::abs(p.dot(v-p))<1e-10);
    REQUIRE(p[0]/d.scalval[0]==Catch::Approx(p[3]/d.scalval[3]).margin(1e-12));
    REQUIRE(p[0]/d.scalval[0]==Catch::Approx(p[5]/d.scalval[5]).margin(1e-12));
    REQUIRE(std::abs(p[1])+std::abs(p[2])+std::abs(p[4])<1e-10);
    Eigen::VectorXd base,step=Eigen::VectorXd::Random(d.x.cols());d.get_x(base);
    double move,cell;REQUIRE(d.limit_step(base,step,.1,move,cell));
    d.X2LatPos_interf(base+step);
    const auto &h=d.images[1].cell_;
    REQUIRE(h(0,0)==Catch::Approx(h(1,1)).margin(1e-10));
    REQUIRE(h(0,0)==Catch::Approx(h(2,2)).margin(1e-10));
}

TEST_CASE("NEB rejects incompatible symmetry options and a noncommon selected operation", "[neb][symmetry]") {
    SymmetryField potential; auto path=pair_path();
    NEBDriver d;d.pot=&potential;d.init(path);NEBPathSymmetry common(d,"");
    int noncommon=-1;
    for(int i=0;i<common.initial.n_operations;++i)
        if(std::find(common.common_operations.begin(),common.common_operations.end(),i)==common.common_operations.end()) {noncommon=i;break;}
    REQUIRE(noncommon>=0);
    d.keep_symm=true;d.symmetry_operations=std::to_string(noncommon);REQUIRE_THROWS_AS(d.init(path),std::invalid_argument);
    d.symmetry_operations="";d.perturb_amplitude=.1;REQUIRE_THROWS_AS(d.init(path),std::invalid_argument);
    d.perturb_amplitude=0;d.sel_dyn=true;REQUIRE_THROWS_AS(d.init(path),std::invalid_argument);
    d.sel_dyn=false;for(auto &image:path) image.pbc_={true,true,false};
    REQUIRE_THROWS_AS(d.init(path),std::invalid_argument);
}

TEST_CASE("NEB constrained iterations preserve symmetry in every calculator trial", "[neb][symmetry]") {
    const std::string policy=GENERATE("legacy","stable-v1");
    const std::string driver=GENERATE("fire","lbfgs");
    SymmetryField potential;potential.force=true;potential.harmonic=true;
    auto path=pair_path();NEBDriver d;d.pot=&potential;d.keep_symm=true;d.solver_policy=policy;d.init(path);
    const auto before=d.images;
    d.run(driver,4,.03);
    REQUIRE_FALSE(potential.symmetry_violation.load());
    REQUIRE((d.images[2].pos_[0]-before[2].pos_[0]).norm()>1e-8);
}
