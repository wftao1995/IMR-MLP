#include <catch2/catch_all.hpp>
#include "mindriver/neb.h"
#include <atomic>
#include <limits>
#include <sstream>

namespace {
class StableQuadratic : public Potentials::AnyPotential {
public:
    std::atomic<int> calls{0};
    double stiffness=1;
    bool double_well=false;
    bool nonfinite=false, throw_error=false;
    std::vector<double> sampled_y;
    StableQuadratic() {cutoff=1;cutoffsq=1;species_count=1;}
    void getEFS(Atoms &a,AnyNeighborhoods &) override {evaluate(a);}
    void getEFSOMPthread(Atoms &a) override {evaluate(a);}
    void evaluate(Atoms &a) {
        ++calls;a.resetEFS();
        for(int i=0;i<a.size();++i) {
            double y=a.pos_[i][1];
            #pragma omp critical(neb_test_samples)
            sampled_y.push_back(y);
            if(throw_error && y<.7999)throw std::runtime_error("calculator failure must propagate");
            a.energy_+=.5*stiffness*y*y;
            a.forces_[i][1]=-stiffness*y;
            if(nonfinite && y<.7999)a.forces_[i][1]=std::numeric_limits<double>::quiet_NaN();
            if(double_well) {
                double x=a.pos_[i][0]-1;
                a.energy_+=(x*x-1)*(x*x-1);
                a.forces_[i][0]=-4*x*(x*x-1);
            }
        }
    }
};

std::vector<Atoms> quadratic_path(int count=3,int atoms=1) {
    Atoms a;a.cell_=20*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};
    a.chemmap_.assign(atoms,0);a.pos_.resize(atoms);
    std::vector<Atoms> path(count,a);
    for(int i=0;i<count;++i)for(int j=0;j<atoms;++j)path[i].pos_[j]<<double(i)+2*j,.8,0;
    return path;
}
}

TEST_CASE("NEB rejects finite nonpositive cell metrics before EFS", "[neb][stable][geometry]") {
    StableQuadratic pot;
    auto path=quadratic_path();
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.vcneb=true;d.check_mic=false;d.spring=0;
    d.restore_F_cholesky=GENERATE(false,true);
    d.init(path);
    Eigen::VectorXd x,g;d.get_x(x);const auto valid=x;
    const int calls=pot.calls;
    x[3]=GENERATE(-.1,0.,1e-14,std::numeric_limits<double>::infinity());
    CHECK_FALSE(std::isfinite(d(x,g)));
    CHECK_FALSE(g.allFinite());
    CHECK_FALSE(d.check_convergence());
    CHECK(pot.calls==calls);
    CHECK(std::isfinite(d(valid,g)));
    CHECK(g.allFinite());
    CHECK(d.images[1].cell_.isApprox(path[1].cell_,1e-12));
}

TEST_CASE("NEB validates complete-cell handedness and conditioning", "[neb][stable][geometry]") {
    StableQuadratic pot;auto path=quadratic_path();
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.vcneb=true;d.symm_strain=false;d.check_mic=false;
    d.init(path);Eigen::VectorXd x,g;d.get_x(x);
    x[0]=GENERATE(-1.,0.,1e-14);
    const int calls=pot.calls;
    CHECK_FALSE(std::isfinite(d(x,g)));
    CHECK(pot.calls==calls);
    CHECK_FALSE(d.check_convergence());
}

TEST_CASE("NEB rejects ill-conditioned initial periodic cells before endpoint EFS", "[neb][stable][geometry]") {
    StableQuadratic pot;auto path=quadratic_path();
    for(auto &a:path) {a.pbc_={true,true,true};a.cell_(0,0)=1e14;}
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;
    CHECK_THROWS_AS(d.init(path),NEBInvalidGeometry);
    CHECK(pot.calls==0);
}

TEST_CASE("NEB stable FIRE retains explicit masses and initial step controls", "[neb][stable][options]") {
    StableQuadratic pot;auto path=quadratic_path();
    for(auto &a:path)a.local_attribs_d_["masses"]=Eigen::VectorXd::Constant(1,4.);
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.spring=0;d.check_mic=false;d.init(path);
    NEBDriver::RunOptions options;options.max_iterations=1;options.max_move=.1;
    options.optimizer_settings=GPSettings({{"assign_mass","true"},{"ts_init",".1"}});
    CHECK(d.run(options)==2);
    CHECK(d.last_atomic_move==Catch::Approx(.002).margin(1e-12));
    d.init(path);
    options.optimizer_settings.Modify("first_delay","1");
    options.optimizer_settings.Modify("first_step",".0001");
    CHECK(d.run(options)==2);
    CHECK(d.last_atomic_move==Catch::Approx(.0001).margin(1e-12));
}

TEST_CASE("NEB physical move is independent of atom and image counts", "[neb][stable][steps]") {
    const int atoms=GENERATE(1,8), count=GENERATE(3,7);
    const std::string driver=GENERATE("fire","lbfgs","cg","sgd");
    StableQuadratic pot;auto path=quadratic_path(count,atoms);
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;d.neb_ftol=1e-12;
    d.init(path);CHECK(d.run(driver,1,.03)==2);
    REQUIRE(d.accepted_steps==1);
    CHECK(d.last_atomic_move==Catch::Approx(.03).margin(1e-12));
    CHECK(d.last_cell_change==0);
    const int evaluations=driver=="cg"?3:2;
    CHECK(d.force_evaluations==evaluations);
    CHECK(d.image_force_evaluations==2+evaluations*(count-2));
    for(int i=1;i<count-1;++i)for(int j=0;j<atoms;++j)
        CHECK((d.images[i].pos_[j]-path[i].pos_[j]).norm()<=.03*(1+1e-12));
    CHECK(d.images.front().pos_==path.front().pos_);
    CHECK(d.images.back().pos_==path.back().pos_);
}

TEST_CASE("VCNEB bounds affine displacement and cell deformation without EFS", "[neb][stable][steps]") {
    StableQuadratic pot;auto path=quadratic_path();
    for(auto &a:path){a.cell_=Eigen::Vector3d(28,4,112).asDiagonal();a.pos_[0]<<1,1,100;}
    path[0].pos_[0][0]=0;path[2].pos_[0][0]=2;
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.vcneb=true;d.check_mic=false;
    d.symm_strain=GENERATE(false,true);d.recast_atomic_force=GENERATE(false,true);d.init(path);
    Eigen::VectorXd x;d.get_x(x);Eigen::VectorXd step=Eigen::VectorXd::Zero(x.size());step[2]=.03;
    const int calls=pot.calls;
    double atomic=0,cell=0;
    REQUIRE(d.limit_step(x,step,.03,atomic,cell));
    CHECK(pot.calls==calls);
    CHECK(atomic<=.03);
    CHECK(cell<=d.max_cell_strain);
    CHECK(step[2]>0);
    CHECK(step[2]<.03);
    CHECK(d.images[1].cell_.isApprox(path[1].cell_,1e-12));
    d.X2LatPos_interf(x+step);
    CHECK((d.images[1].pos_[0]-path[1].pos_[0]).norm()<=.03);
}

TEST_CASE("NEB budget restores the last valid EFS and geometry", "[neb][stable][budget]") {
    StableQuadratic pot;pot.nonfinite=true;
    auto path=quadratic_path();NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;
    d.max_force_evaluations=3;d.init(path);
    const auto original=d.images[1];
    NEBDriver::RunOptions options;options.max_iterations=100;options.max_move=.03;
    options.experimental_acceleration=GENERATE("none","secant","anderson","combined","multisecant","guarded","adaptive","cell-block","graph","colored","retained");
    CHECK(d.run(options)==2);
    CHECK(d.stop_reason==NEBDriver::StopReason::max_evaluations);
    CHECK(d.force_evaluations==3);
    CHECK(d.accepted_steps==0);
    CHECK(d.rejected_steps>0);
    CHECK_FALSE(d.converged);
    CHECK(d.images[1].pos_==original.pos_);
    CHECK(d.images[1].forces_[0].allFinite());
    CHECK(d.max_neb_force==Catch::Approx(.8));
}

TEST_CASE("NEB does not swallow calculator exceptions", "[neb][stable][errors]") {
    StableQuadratic pot;pot.throw_error=true;
    auto path=quadratic_path();NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;d.init(path);
    NEBDriver::RunOptions options;options.max_iterations=10;options.max_move=.03;
    options.experimental_acceleration=GENERATE("none","secant","anderson","combined","multisecant","guarded","adaptive","cell-block","graph","colored","retained");
    CHECK_THROWS_WITH(d.run(options),"calculator failure must propagate");
    CHECK(d.images[1].pos_==path[1].pos_);
    CHECK(d.images[1].forces_[0].allFinite());
}

TEST_CASE("NEB Krylov probes cannot bypass physical displacement limits", "[neb][stable][krylov]") {
    StableQuadratic pot;auto path=quadratic_path();
    for(auto &a:path)a.pos_[0][1]=1.0008e-5;
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;d.tangent_type=2;d.neb_ftol=1e-12;
    d.init(path);
    NEBDriver::RunOptions options;options.max_iterations=9;options.max_move=1e-9;
    options.experimental_acceleration=GENERATE("none","secant","anderson","combined","multisecant","guarded","adaptive","cell-block","graph","colored","retained");
    CHECK(d.run(options)==2);
    CHECK(d.accepted_steps==9);
    CHECK_FALSE(d.converged);
    CHECK(d.last_atomic_move<=1e-9*(1+1e-10));
    CHECK(std::abs(d.images[1].pos_[0][1]-path[1].pos_[0][1])<=9e-9*(1+1e-10));
    CHECK(d.images[1].pos_[0][1]>9.99e-6);
    CHECK(d.force_evaluations>10); // probes actually ran
    for(double y:pot.sampled_y) CHECK(std::abs(y-path[1].pos_[0][1])<=10e-9*(1+1e-6));
}

TEST_CASE("NEB CG convergence at a probe cannot replace the trusted output", "[neb][stable][phase]") {
    StableQuadratic pot;auto path=quadratic_path();
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;d.max_force_evaluations=2;d.init(path);
    NEBDriver::RunOptions options;options.driver="cg";options.max_iterations=10;options.max_move=1.;
    options.optimizer_settings.Modify("finite_step",".8");
    CHECK(d.run(options)==2);
    CHECK(d.stop_reason==NEBDriver::StopReason::max_evaluations);
    CHECK_FALSE(d.converged);
    CHECK(d.accepted_steps==0);
    CHECK(d.images[1].pos_==path[1].pos_);
    CHECK(d.max_neb_force==Catch::Approx(.8));
}

TEST_CASE("NEB climb delay advances accepted states without artificial EFS", "[neb][stable][phase]") {
    StableQuadratic pot;pot.double_well=true;auto path=quadratic_path();
    for(auto &a:path)a.pos_[0][1]=0;
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;d.climb=true;d.climb_delay=3;d.init(path);
    CHECK(d.run("fire",10,.03)==0);
    CHECK(d.climbing_active);
    CHECK(d.climbing_image==1);
    CHECK(d.accepted_steps==3);
    CHECK(d.force_evaluations==1);
    CHECK(d.images[1].pos_==path[1].pos_);
    d.init(path);
    NEBDriver::RunOptions options;options.max_iterations=10;options.max_move=.03;options.accepted_step_offset=3;
    CHECK(d.run(options)==0);
    CHECK(d.climbing_active);
    CHECK(d.accepted_steps==3);
    CHECK(d.optimizer_iterations==0);
    CHECK(d.force_evaluations==1);
}

TEST_CASE("NEB stable policies report physical cost and retain legacy entry point", "[neb][stable]") {
    StableQuadratic pot;auto path=quadratic_path();
    for(auto &a:path) a.pos_[0][1]=0;
    for(const std::string policy:{"stable-v1","legacy"}) {
        NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.solver_policy=policy;d.check_mic=false;d.init(path);
        CHECK(d.run("fire",10,.03)==0);
        CHECK(d.converged);
        CHECK(d.stop_reason_string()=="converged");
        std::ostringstream output;d.write_diagnostics(output);
        CHECK(output.str().find("equivalent_A22")!=std::string::npos);
    }
}

TEST_CASE("VCNEB cell diagnostics measure periodic-subspace stretches", "[neb][stable][diagnostics]") {
    StableQuadratic pot;auto path=quadratic_path();
    for(auto &a:path) {a.pbc_={true,true,false};a.cell_=Eigen::Vector3d(20,20,0).asDiagonal();}
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.vcneb=true;d.periodic_subspace=true;d.check_mic=false;d.init(path);
    d.images[1].cell_(0,0)*=1.02;
    std::ostringstream stream;d.write_diagnostics(stream);
    std::istringstream lines(stream.str());std::string row;
    std::getline(lines,row);std::getline(lines,row);std::getline(lines,row);
    std::replace(row.begin(),row.end(),',',' ');std::istringstream fields(row);
    int image;double volume,bx,cx,cy,minimum,maximum;
    fields>>image>>volume>>bx>>cx>>cy>>minimum>>maximum;
    CHECK(image==1);CHECK(volume==0);
    CHECK(minimum==Catch::Approx(1.));CHECK(maximum==Catch::Approx(1.02));
}

TEST_CASE("NEB adaptive-image budget preserves the last valid band stage", "[neb][stable][images]") {
    const int budget=GENERATE(2,3);
    StableQuadratic pot;auto path=quadratic_path(4);
    path[1].pos_[0][0]=.1;path[2].pos_[0][0]=.2;path[3].pos_[0][0]=4;
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;
    d.variable_image=true;d.image_adjust_interval=1;d.image_insert_thresh=1.2;d.max_image_count=6;
    d.max_force_evaluations=budget;d.init(path);
    const double old_equilibrium=d.Leq;
    CHECK(d.run("fire",1,.03)==2);
    CHECK(d.accepted_steps==1);
    CHECK(d.force_evaluations==budget);
    CHECK(d.images.size()==(budget==2?4:5));
    CHECK(d.x.rows()==d.images.size());CHECK(d.g.rows()==d.images.size());
    if(budget==2) {
        CHECK(d.stop_reason==NEBDriver::StopReason::max_evaluations);
        CHECK(d.Leq==old_equilibrium);
    }
    for(const auto &image:d.images) {
        CHECK(std::isfinite(image.energy_));CHECK(image.forces_[0].allFinite());
    }
    CHECK(d.images.front().pos_==path.front().pos_);CHECK(d.images.back().pos_==path.back().pos_);
}

TEST_CASE("NEB reserves candidate EFS after a Krylov probe", "[neb][stable][budget]") {
    const int budget=GENERATE(18,19);
    StableQuadratic pot;pot.stiffness=-1.;auto path=quadratic_path();
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;
    d.max_force_evaluations=budget;d.init(path);
    CHECK(d.run("fire",10,.03)==2);
    CHECK(d.stop_reason==NEBDriver::StopReason::max_evaluations);
    CHECK(d.force_evaluations==budget-1);
    CHECK_FALSE(d.converged);CHECK(d.accepted_steps==0);
    CHECK(d.images[1].pos_==path[1].pos_);
    CHECK(d.max_neb_force==Catch::Approx(.8));
}

TEST_CASE("NEB experimental residual histories preserve the physical solution", "[neb][stable][history]") {
    StableQuadratic pot;auto path=quadratic_path(5,3);
    pot.stiffness=7;
    for(int i=1;i<4;++i)for(int j=0;j<3;++j)path[i].pos_[j][1]=.2+.07*i+.11*j;
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;
    d.neb_ftol=1e-7;d.max_force_evaluations=400;d.init(path);
    NEBDriver::RunOptions options;options.max_iterations=400;options.max_move=.03;
    options.driver=GENERATE("fire","lbfgs");
    options.experimental_acceleration=GENERATE("none","secant","anderson","combined","multisecant","guarded","adaptive","cell-block","graph","colored","retained");
    REQUIRE(d.run(options)==0);
    CHECK(d.force_evaluations<=400);
    CHECK(d.max_neb_force<=1e-7);
    CHECK(d.last_atomic_move<=.03*(1+1e-12));
    CHECK(d.images.front().pos_==path.front().pos_);
    CHECK(d.images.back().pos_==path.back().pos_);
}

TEST_CASE("NEB rejects unknown experimental variants", "[neb][stable][options]") {
    StableQuadratic pot;NEBDriver d;d.pot=&pot;auto path=quadratic_path();d.init(path);
    NEBDriver::RunOptions options;options.experimental_acceleration="typo";
    CHECK_THROWS_AS(d.run(options),std::invalid_argument);
    options.experimental_acceleration="anderson";
    CHECK_THROWS_AS(d.run(options),std::invalid_argument); // legacy remains unmodified
}

TEST_CASE("NEB guarded acceleration preserves productive FIRE momentum", "[neb][stable][history]") {
    std::vector<double> reference_samples;
    Eigen::RowVector3d reference_position;
    int reference_evaluations=0;
    for(const std::string variant:{"none","guarded","adaptive","graph","colored","retained"}) {
        StableQuadratic pot;
        NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;d.neb_ftol=1e-12;
        auto path=quadratic_path();d.init(path);
        NEBDriver::RunOptions options;options.driver="fire";options.max_iterations=10;options.max_move=.03;
        options.experimental_acceleration=variant;
        CHECK(d.run(options)==2);
        REQUIRE(d.accepted_steps==10);
        if(variant=="none") {
            reference_samples=pot.sampled_y;reference_position=d.images[1].pos_[0];reference_evaluations=d.force_evaluations;
        } else {
            CHECK(pot.sampled_y==reference_samples);
            CHECK(d.images[1].pos_[0].isApprox(reference_position,1e-14));
            CHECK(d.force_evaluations==reference_evaluations);
        }
    }
}

TEST_CASE("VCNEB cell model probes retain physical bounds and budget", "[neb][stable][krylov][budget]") {
    const int count=GENERATE(3,5), budget=GENERATE(18,64);
    StableQuadratic pot;auto path=quadratic_path(count);
    for(auto &a:path)a.pos_[0][1]=1.0008e-5;
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;d.spring=0;
    d.vcneb=true;d.symm_strain=GENERATE(false,true);d.cell_weight=100;d.neb_ftol=1e-12;
    d.max_force_evaluations=budget;d.init(path);
    NEBDriver::RunOptions options;options.max_iterations=9;options.max_move=1e-9;
    options.experimental_acceleration="cell-block";
    CHECK(d.run(options)==2);
    CHECK_FALSE(d.converged);
    CHECK(d.force_evaluations<=budget);
    CHECK(d.force_evaluations>10);
    CHECK(d.accepted_steps<=9);
    for(int i=1;i<count-1;++i) {
        CHECK((d.images[i].pos_[0]-path[i].pos_[0]).norm()<=9e-9*(1+1e-6));
        CHECK(d.images[i].forces_[0].allFinite());
        CHECK(d.images[i].cell_.allFinite());
    }
    for(double y:pot.sampled_y)CHECK(std::abs(y-path[1].pos_[0][1])<=10e-9*(1+1e-6));
}

TEST_CASE("NEB separated image probes have disjoint residual supports", "[neb][stable][krylov]") {
    StableQuadratic pot;auto path=quadratic_path(8);
    for(int i=0;i<path.size();++i)path[i].pos_[0][1]=.8+.1*std::sin(double(i));
    NEBDriver d;d.solver_policy="stable-v1";d.pot=&pot;d.check_mic=false;
    d.vcneb=GENERATE(false,true);d.tangent_type=GENERATE(1,2,3);d.climb=GENERATE(false,true);d.climb_delay=0;
    d.init(path);
    Eigen::VectorXd base,gradient;d.get_x(base);d(base,gradient);
    const int stride=base.size()/6;
    Eigen::VectorXd first=Eigen::VectorXd::Zero(base.size()),second=first;
    first[d.vcoffset+1]=1e-4;second[3*stride+d.vcoffset+1]=-2e-4;
    if(d.vcneb) {first[0]=1e-5;second[3*stride+3]=-1e-5;}
    Eigen::VectorXd a,b,together;
    d(base+first,a);d(base+second,b);d(base+first+second,together);
    CHECK((together-gradient-(a-gradient)-(b-gradient)).lpNorm<Eigen::Infinity>()<1e-9);
}
