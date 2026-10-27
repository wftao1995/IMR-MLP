#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "mindriver/relax_constraints_io.h"
#include "mindriver/constraint_expression.h"
#include "potentials/potential_lj.h"
#include <filesystem>
#include <fstream>
#include <numbers>

namespace {
struct Files {
    std::filesystem::path previous = std::filesystem::current_path();
    std::filesystem::path path;
    Files() {
        char pattern[] = "/tmp/imrmlp-sqp-transform-XXXXXX";
        path = mkdtemp(pattern);
        std::filesystem::current_path(path);
    }
    ~Files() {
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};
Atoms source() {
    Atoms a;
    a.cell_ = 10*Eigen::Matrix3d::Identity();
    a.pbc_ = {false,false,false}; a.cartesian = true;
    a.pos_ = {{1.,1.,1.}}; a.chemmap_ = {0};
    return a;
}
struct RecordingHarmonic : Potentials::AnyPotential {
    RelaxationDrvV2 *driver = nullptr;
    Eigen::RowVector3d target{2.,1.,1.};
    double largest_atom_trial = 0, largest_cell_trial = 0;
    int calls = 0, fail_after = -1;
    RecordingHarmonic() {cutoff=2;species_count=1;}
    void getEFS(Atoms &a, AnyNeighborhoods &) override {getEFSOMP(a);}
    void getEFSOMP(Atoms &a, AnyNeighborhoods &) override {getEFSOMP(a);}
    void getEFSOMP(Atoms &a) override {
        if (fail_after >= 0 && calls >= fail_after) throw std::runtime_error("recorded trial failure");
        ++calls;
        if (driver && driver->has_checkpoint) {
            for (int i=0;i<a.size();++i)
                largest_atom_trial=std::max(largest_atom_trial,(a.pos_[i]-driver->checkpoint_atom.pos_[i]).norm());
            for (int i=0;i<3;++i)
                largest_cell_trial=std::max(largest_cell_trial,(a.cell_.row(i)-driver->checkpoint_atom.cell_.row(i)).norm());
        }
        const auto delta=(a.pos_[0]-target).eval();
        a.energy_=.5*delta.squaredNorm();
        a.forces_={-delta};
        a.stress_=-a.pos_[0].transpose()*delta;
    }
};
void configure(RelaxationDrvV2 &d, RecordingHarmonic &p) {
    d.control.driver="sqp";d.flags.write_traj=false;
    d.control.max_move=.05;d.control.max_iter=1200;
    d.tol.gtol=1e-6;p.driver=&d;
}
void matrix_file(const std::string &name,const Eigen::MatrixXd &matrix,bool affine=false) {
    std::ofstream out(name);out.precision(17);
    if(affine) out<<matrix.rows()<<' '<<matrix.cols()<<'\n';
    else out<<matrix.rows()<<'\n';
    out<<matrix<<'\n';
    if(affine)out<<Eigen::VectorXd::Zero(matrix.rows()).transpose()<<'\n';
}
std::vector<regulator> declarations(const std::string &entries,const Atoms &a) {
    std::istringstream text("{\"version\":2,\"constraints\":"+entries+"}");
    return Constraints::bind_constraints(Constraints::parse_constraints(text),a);
}
}

TEST_CASE("SQP Cartesian trial cap survives LF LC mask fractional coordinates and whitening", "[sqp-transform][step]") {
    Files files;
    const int mode=GENERATE(0,1,2,3,4,5);
    auto a=source();RecordingHarmonic p;RelaxationDrvV2 d;configure(d,p);
    if(mode==0) { // x = 100*y: small active steps are large physical moves
        d.flags.linear_transform=true;
        matrix_file(d.control.lf_file,.01*Eigen::Matrix3d::Identity(),true);
    }
    if(mode==1) {
        d.flags.linear_constraint=true;
        Eigen::Matrix<double,1,3> row;row<<1,-1,0;
        matrix_file(d.control.lc_file,row,true);
        p.target={2.,2.,1.};
    }
    if(mode==2) {
        d.flags.selective_dynamics=true;
        std::ofstream(d.control.seldyn_file)<<"3 1 0 0\n";
    }
    if(mode==3)d.flags.atomic_force_recast=false;
    if(mode==4 || mode==5) {
        d.control.hessian_precond_file="metric.txt";
        matrix_file("metric.txt",(mode==4?.0001:1e8)*Eigen::Matrix3d::Identity());
    }
    d.init(&p,a);
    REQUIRE(d.run()==0);
    REQUIRE((d.atom.pos_[0]-p.target).norm()<1e-5);
    REQUIRE(p.largest_atom_trial<=d.control.max_move*(1+1e-9));
    REQUIRE(d.state_vars.curr_iter<=d.control.max_iter);
    REQUIRE(p.calls>3);
}

TEST_CASE("Initially violated position bounds approach a distant feasible interval without projection jumps", "[sqp-transform][bounds]") {
    Files files;auto a=source();RecordingHarmonic p;RelaxationDrvV2 d;configure(d,p);
    p.target={4.,1.,1.};
    auto regs=declarations(R"([{"kind":"position","atoms":[0],"component":"x","lower":2,"upper":3,"residual_tolerance":1e-8}])",a);
    d.init(&p,a,&regs);
    REQUIRE(d.run()==0);
    REQUIRE(d.atom.pos_[0][0]==Catch::Approx(3).margin(1e-7));
    REQUIRE(d.atom.forces_[0].norm()>.9);
    REQUIRE(d.kkt_residual<=d.tol.gtol);
    REQUIRE(p.largest_atom_trial<=d.control.max_move*(1+1e-9));
}

TEST_CASE("Distant infeasible distance equality advances through local SQP trust boxes", "[sqp-transform][constraints]") {
    auto a=source();a.pos_.push_back({3.,1.,1.});a.chemmap_.push_back(0);
    struct PairHarmonic : RecordingHarmonic {
        void getEFSOMP(Atoms &a) override {
            RecordingHarmonic::getEFSOMP(a);
            const Eigen::RowVector3d delta=a.pos_[1]-Eigen::RowVector3d(4.,1.,1.);
            a.energy_+=.5*delta.squaredNorm();a.forces_.push_back(-delta);
            a.stress_-=a.pos_[1].transpose()*delta;
        }
    } p;
    RelaxationDrvV2 d;configure(d,p);p.target={0.,1.,1.};
    auto regs=declarations(R"([{"kind":"distance","atoms":[0,1],"target":1,"residual_tolerance":1e-7}])",a);
    d.init(&p,a,&regs);
    REQUIRE(d.run()==0);
    REQUIRE((d.atom.pos_[1]-d.atom.pos_[0]).norm()==Catch::Approx(1.).margin(1e-7));
    REQUIRE(d.kkt_residual<=d.tol.gtol);
    REQUIRE(d.atom.forces_[0].norm()>1.);
    REQUIRE(p.largest_atom_trial<=d.control.max_move*(1+1e-9));
}

TEST_CASE("SQP nonlinear previews reject geometry before potential evaluation and share total budget", "[sqp-transform][nonlinear]") {
    const int budget=GENERATE(3,300,1200);
    auto a=source();RecordingHarmonic p;RelaxationDrvV2 d;configure(d,p);
    d.control.max_iter=budget;
    if(budget==1200)d.control.min_move=.02;
    struct ExponentialMap : NonlinearCoordinates {
        Eigen::VectorXd reduce(const Eigen::VectorXd &x) const override {
            Eigen::VectorXd q(1);q[0]=std::log(x[0])/100;return q;
        }
        Eigen::VectorXd lift(const Eigen::VectorXd &q) const override {
            Eigen::VectorXd x(3);x<<std::exp(100*q[0]),1,1;return x;
        }
        Eigen::MatrixXd jacobian(const Eigen::VectorXd &q) const override {
            Eigen::MatrixXd j=Eigen::MatrixXd::Zero(3,1);j(0,0)=100*std::exp(100*q[0]);return j;
        }
    };
    auto map=std::make_shared<ExponentialMap>();
    d.coordinate_map=map;d.init(&p,a);
    const int result=d.run();
    REQUIRE(p.largest_atom_trial<=d.control.max_move*(1+1e-9));
    REQUIRE(d.state_vars.curr_iter<=budget);
    if(budget==3) {
        REQUIRE(result!=0);REQUIRE_FALSE(d.state_vars.converged);
        REQUIRE(d.accepted_steps==0);REQUIRE(d.state_vars.curr_iter==3);
        REQUIRE(d.atom.pos_[0].isApprox(a.pos_[0]));
    } else if(budget==1200) {
        REQUIRE(result!=0);REQUIRE(d.stop_reason=="minimum_step");
        REQUIRE(d.accepted_steps==0);REQUIRE(d.state_vars.curr_iter==2);
        REQUIRE(d.atom.pos_[0].isApprox(a.pos_[0]));
    } else {
        REQUIRE(result==0);REQUIRE(d.atom.pos_[0][0]==Catch::Approx(2.).margin(2e-6));
    }
}

TEST_CASE("SQP failed trial restores accepted geometry and clears convergence", "[sqp-transform][failure]") {
    auto a=source();RecordingHarmonic p;RelaxationDrvV2 d;configure(d,p);
    d.init(&p,a);p.fail_after=3;
    REQUIRE_THROWS_WITH(d.run(),"recorded trial failure");
    REQUIRE(d.atom.pos_==d.checkpoint_atom.pos_);
    REQUIRE(d.x.isApprox(d.checkpoint_x));
    REQUIRE_FALSE(d.state_vars.converged);
    REQUIRE(d.stop_reason=="error");
}

TEST_CASE("Dihedral sign matches independent projected-vector physical convention across branch cut", "[sqp-transform][dihedral]") {
    Atoms a=source();a.pos_={{1,0,0},{0,0,0},{0,0,1},{0,1,1}};a.chemmap_={0,0,0,0};
    Constraints::Expression expression("dihedral(0,1,2,3)",a);
    REQUIRE(expression.evaluate(a).value==Catch::Approx(90.).margin(1e-12));
    const auto reference=[](const Atoms &atom) {
        const Eigen::Vector3d axis=(atom.pos_[2]-atom.pos_[1]).normalized();
        const Eigen::Vector3d left=atom.pos_[0]-atom.pos_[1],right=atom.pos_[3]-atom.pos_[2];
        const Eigen::Vector3d u=left-axis*axis.dot(left),v=right-axis*axis.dot(right);
        return std::atan2(axis.cross(u).dot(v),u.dot(v))*180/std::numbers::pi;
    };
    for(double degrees:{-120.,-45.,60.,150.}) {
        const double radians=degrees*std::numbers::pi/180;
        a.pos_[3]={std::cos(radians),std::sin(radians),1};
        Constraints::Expression bound("dihedral(0,1,2,3)",a);
        REQUIRE(bound.evaluate(a).value==Catch::Approx(reference(a)).margin(1e-11));
    }
    const double initial=179*std::numbers::pi/180;
    a.pos_[3]={std::cos(initial),std::sin(initial),1};
    Constraints::Expression crossing("dihedral(0,1,2,3)",a);
    const double after=181*std::numbers::pi/180;
    a.pos_[3]={std::cos(after),std::sin(after),1};
    REQUIRE(crossing.evaluate(a).value==Catch::Approx(181.).margin(1e-11));
    const auto derivative=crossing.evaluate(a).positions[3];
    for(double h:{1e-5,5e-6})for(int j=0;j<3;++j) {
        Atoms plus=a,minus=a;plus.pos_[3][j]+=h;minus.pos_[3][j]-=h;
        REQUIRE(derivative[j]==Catch::Approx((crossing.evaluate(plus).value-crossing.evaluate(minus).value)/(2*h)).margin(1e-7));
    }
}

TEST_CASE("Physical position bounds retain offsets through mixed LF LC masks and whitening", "[sqp-transform][classification]") {
    Files files;const int mode=GENERATE(0,1,2,3);
    auto a=source();RecordingHarmonic p;RelaxationDrvV2 d;configure(d,p);
    if(mode==0) {
        d.flags.linear_transform=true;
        std::ofstream(d.control.lf_file)<<"3 3\n.02 .01 0\n0 .01 0\n0 0 .02\n.25 -.4 .1\n";
    } else if(mode==1) {
        d.flags.linear_constraint=true;
        Eigen::Matrix<double,1,3> row;row<<0,1,-1;
        matrix_file(d.control.lc_file,row,true);
    } else if(mode==2) {
        d.flags.selective_dynamics=true;
        std::ofstream(d.control.seldyn_file)<<"3 1 0 0\n";
    } else {
        d.control.hessian_precond_file="metric.txt";
        Eigen::Matrix3d metric;metric<<.02,.01,0,.01,.03,.005,0,.005,.02;
        matrix_file("metric.txt",metric);
    }
    auto regs=declarations(R"([{"kind":"position","atoms":[0],"component":"x","lower":1.3,"upper":1.4,"residual_tolerance":1e-8}])",a);
    d.init(&p,a,&regs);
    REQUIRE(d.run()==0);
    REQUIRE(d.atom.pos_[0][0]==Catch::Approx(1.4).margin(1e-7));
    REQUIRE(d.kkt_residual<=d.tol.gtol);
    REQUIRE(p.largest_atom_trial<=d.control.max_move*(1+1e-9));
}

TEST_CASE("Variable cell SQP previews cap real cell rows and atomic affine displacement", "[sqp-transform][cell]") {
    struct CellHarmonic : RecordingHarmonic {
        void getEFSOMP(Atoms &a) override {
            RecordingHarmonic::getEFSOMP(a);
            const double delta=a.cell_(0,0)-11.;
            a.energy_=.5*delta*delta;
            a.forces_.assign(a.size(),Eigen::Vector3d::Zero());
            a.stress_.setZero();a.stress_(0,0)=-a.cell_(0,0)*delta;
        }
    } p;
    auto a=source();RelaxationDrvV2 d;configure(d,p);
    d.flags.relax_cell=true;d.flags.relax_atom=false;
    d.init(&p,a);
    REQUIRE(d.run()==0);
    REQUIRE(d.atom.cell_(0,0)==Catch::Approx(11.).margin(1e-5));
    REQUIRE(p.largest_cell_trial<=d.control.max_move*(1+1e-9));
    REQUIRE(p.largest_atom_trial<=d.control.max_move*(1+1e-9));
    REQUIRE(d.state_vars.curr_iter<=d.control.max_iter);
}

TEST_CASE("Local SQP segments converge with real LJ nonzero-force equality and interval", "[sqp-transform][lj]") {
    const bool interval=GENERATE(false,true);
    auto a=source();a.pos_.push_back({2.4,1.,1.});a.chemmap_.push_back(0);
    GPSettings settings;settings.Modify("ntypes","1");settings.Modify("0-0","1 1");settings.Modify("cut:0-0","3");
    Potentials::LJ612 potential(settings);
    RelaxationDrvV2 d;d.control.driver="sqp";d.flags.write_traj=false;
    d.control.max_move=.05;d.control.max_iter=600;d.tol.gtol=1e-6;
    auto regs=declarations(interval?
        R"([{"kind":"distance","atoms":[0,1],"lower":1.6,"upper":1.8,"residual_tolerance":1e-7}])":
        R"([{"kind":"distance","atoms":[0,1],"target":1.8,"residual_tolerance":1e-7}])",a);
    d.init(&potential,a,&regs);
    REQUIRE(d.run()==0);
    REQUIRE((d.atom.pos_[1]-d.atom.pos_[0]).norm()==Catch::Approx(interval?1.6:1.8).margin(2e-6));
    REQUIRE(d.atom.forces_[0].norm()>.1);
    REQUIRE(d.kkt_residual<=d.tol.gtol);
    REQUIRE(d.state_vars.curr_iter<=d.control.max_iter);
}
