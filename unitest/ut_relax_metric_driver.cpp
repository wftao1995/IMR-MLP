#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"
#include "mindriver/relax_constraints_io.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

using Catch::Approx;
namespace {
struct MetricFiles {
    std::filesystem::path path=std::filesystem::temp_directory_path()/
        ("imrmlp-driver-metric-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    MetricFiles() {std::filesystem::create_directory(path);}
    ~MetricFiles() {std::error_code ec;std::filesystem::remove_all(path,ec);}
    std::string name(const std::string &file) const {return (path/file).string();}
    void matrix(const std::string &file,const Eigen::MatrixXd &h) const {
        std::ofstream out(name(file));out<<h.rows()<<'\n'<<std::setprecision(17);
        for(int i=0;i<h.rows();++i)for(int j=0;j<h.cols();++j)out<<h(i,j)<<' ';
    }
};
Atoms metric_atoms(int count=1) {
    Atoms a;a.cell_=10*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};
    for(int i=0;i<count;++i) {a.pos_.push_back({1.+i,.5,-.3});a.chemmap_.push_back(0);}return a;
}
struct DriverQuadratic : Potentials::AnyPotential {
    Eigen::Matrix3d h=Eigen::Matrix3d::Identity();double quartic=0;
    bool dense=false,hvp=false;int dense_calls=0,hvp_calls=0,efs_calls=0;
    DriverQuadratic() {cutoff=4;cutoffsq=16;species_count=1;}
    Eigen::MatrixXd curvature(const Atoms &a) const {
        Eigen::MatrixXd result=Eigen::MatrixXd::Zero(3*a.size(),3*a.size());
        for(int i=0;i<a.size();++i) {
            result.block<3,3>(3*i,3*i)=h;
            result.block<3,3>(3*i,3*i).diagonal().array()+=3*quartic*a.pos_[i].array().square().transpose();
        }return result;
    }
    void getEFS(Atoms &a,AnyNeighborhoods &) override {
        ++efs_calls;a.energy_=0;a.stress_.setZero();a.forces_.resize(a.size());
        for(int i=0;i<a.size();++i) {
            const Eigen::Vector3d x=a.pos_[i];a.energy_+=.5*x.dot(h*x)+.25*quartic*x.array().pow(4).sum();
            a.forces_[i]=-(h*x+quartic*x.array().cube().matrix());
        }
    }
    bool supports_analytical_hessian() const override {return dense;}
    bool supports_analytical_hessian_vector_product() const override {return hvp;}
    void getEFSHessian(Atoms &a,AnyNeighborhoods &,Potentials::HessianMatrix &out) override {
        ++dense_calls;const Eigen::MatrixXd matrix=curvature(a);out.resize(a.size());
        for(int i=0;i<a.size();++i)for(int j=0;j<a.size();++j)out(i,j)=matrix.block<3,3>(3*i,3*j);
    }
    void applyHessianVectorProduct(Atoms &a,AnyNeighborhoods &,const Eigen::VectorXd &v,Eigen::VectorXd &out) override {
        ++hvp_calls;out=curvature(a)*v;
    }
};
void basic(RelaxationDrvV2 &d,const MetricFiles &files,const std::string &algorithm="gd") {
    d.control.driver=algorithm;d.flags.cartesian=true;d.flags.dprecond=false;d.flags.write_traj=false;
    d.control.max_iter=0;d.control.seldyn_file=files.name("absent-mask");
    d.control.max_move=.02;d.buffer_gd.max_move_init=.01;d.tol.gtol=1e-9;
}
}

TEST_CASE("Driver full non-diagonal metric pulls LF LC masks through actual displacement and gradients", "[relax_metric_driver][full]") {
    const std::string mode=GENERATE("plain","lf","lc","lf-lc","mask");
    INFO(mode);MetricFiles files;DriverQuadratic pot;pot.h<<4,1,.2,1,3,.4,.2,.4,2;
    Atoms a=metric_atoms();const Eigen::Vector3d initial=a.pos_[0];files.matrix("hessian",pot.h);
    RelaxationDrvV2 d;basic(d,files);d.control.hessian_precond_file=files.name("hessian");
    Eigen::Matrix3d transform;transform<<2,.3,0,0,1.5,.2,0,0,1;
    Eigen::MatrixXd j=Eigen::Matrix3d::Identity();
    if(mode=="lf" || mode=="lf-lc") {
        std::ofstream out(files.name("lf"));out<<"3 3\n"<<std::setprecision(17);
        for(int i=0;i<3;++i)for(int k=0;k<3;++k)out<<transform(i,k)<<' ';
        out<<"\n0 0 0\n";out.close();d.flags.linear_transform=true;d.control.lf_file=files.name("lf");j=transform.inverse();
    }
    if(mode=="lc" || mode=="lf-lc") {
        // Constraint on LF output when LF is enabled; initial point is feasible.
        const Eigen::RowVector3d row(1,1,0);const Eigen::Vector3d u=d.flags.linear_transform?(transform*initial).eval():initial;
        std::ofstream out(files.name("lc"));out<<"1 3\n1 1 0\n"<<std::setprecision(17)<<row.dot(u)<<'\n';out.close();
        d.flags.linear_constraint=true;d.control.lc_file=files.name("lc");
        Eigen::Matrix<double,3,2> null;null<<1/std::sqrt(2.),0,-1/std::sqrt(2.),0,0,1;j=j*null;
    }
    if(mode=="mask") {
        a.sel_dynpos_={{1,1,0}};d.flags.selective_dynamics=true;j=Eigen::Matrix3d::Identity().leftCols(2);
    }
    d.init(&pot,a);d.run();
    const Eigen::MatrixXd projected=j.transpose()*pot.h*j;
    // For an SPD full metric, one white steepest step is the exact projected Newton direction.
    const Eigen::Vector3d expected=initial-.01*j*projected.ldlt().solve(j.transpose()*pot.h*initial);
    const Eigen::VectorXd white_origin=d.x,white_gradient=d.g;
    const double eps=1e-6;
    for(int k=0;k<d.x.size();++k) {
        d.x=white_origin;d.x[k]+=eps;d.eval();const double plus=d.f;
        d.x=white_origin;d.x[k]-=eps;d.eval();const double minus=d.f;
        REQUIRE((plus-minus)/(2*eps)==Approx(white_gradient[k]).margin(1e-8));
    }
    d.x=white_origin-.01*white_gradient;d.eval();
    REQUIRE(d.atom.pos_[0].transpose().isApprox(expected,1e-10));
    REQUIRE(d.f<.5*initial.dot(pot.h*initial));
    if(mode=="mask")REQUIRE(d.atom.pos_[0][2]==Approx(initial[2]).margin(1e-13));
}
TEST_CASE("Driver judges true gradient despite very small whitened optimizer gradient", "[relax_metric_driver][full][convergence]") {
    MetricFiles files;DriverQuadratic pot;Atoms a=metric_atoms();files.matrix("huge",1e12*Eigen::Matrix3d::Identity());
    RelaxationDrvV2 d;basic(d,files);d.tol.gtol=1e-3;d.control.hessian_precond_file=files.name("huge");
    d.init(&pot,a);d.run();REQUIRE(d.g.cwiseAbs().maxCoeff()<d.tol.gtol);
    REQUIRE(d.convergence_gradient().isApprox(a.pos_[0].transpose(),1e-12));
    REQUIRE_FALSE(d.state_vars.converged);REQUIRE_FALSE(d.check_converage_final(d.tol));
    REQUIRE_FALSE(d.check_converage(d.tol));
    // The optimizer must also attempt a real step rather than stop on tiny white g.
    RelaxationDrvV2 optimizer;basic(optimizer,files,"lbfgspp");optimizer.tol.gtol=1e-3;
    optimizer.control.hessian_precond_file=files.name("huge");optimizer.control.max_iter=2;
    optimizer.control.max_move=1000;optimizer.init(&pot,a);optimizer.run();
    REQUIRE(optimizer.accepted_steps>0);
    REQUIRE((optimizer.atom.pos_[0]-a.pos_[0]).norm()>1e-8);
    REQUIRE(optimizer.atom.energy_<.5*a.pos_[0].squaredNorm());
}
TEST_CASE("Sparse GD and FIRE descend with an active Cartesian mask", "[relax_metric_driver][sparse]") {
    const std::string algorithm=GENERATE("gd","fire");MetricFiles files;DriverQuadratic pot;Atoms a=metric_atoms(3);
    a.pos_[2][0]=3.5;a.sel_dynpos_.assign(3,Eigen::RowVector3d(1,0,0));
    RelaxationDrvV2 d;basic(d,files,algorithm);d.control.sparse_precond=true;d.control.sparse_shift=.2;
    d.flags.selective_dynamics=true;d.control.max_iter=5;d.buffer_fire.ts_init=.01;d.buffer_fire.downhill_check=true;
    double initial=0;for(const auto &x:a.pos_)initial+=.5*x.squaredNorm();d.init(&pot,a);d.run();
    REQUIRE(d.sparse_metric);REQUIRE(d.sparse_metric->matrix().rows()==3);REQUIRE(d.accepted_steps>0);REQUIRE(d.f<initial);
    for(int i=0;i<3;++i) {
        REQUIRE(d.atom.pos_[i][0]<a.pos_[i][0]);REQUIRE(d.atom.pos_[i][1]==a.pos_[i][1]);REQUIRE(d.atom.pos_[i][2]==a.pos_[i][2]);
    }
    REQUIRE(d.atom.cell_.isApprox(a.cell_));
}
TEST_CASE("Thin-cell scaling gives identical real planar steps with different vacuum padding", "[relax_metric_driver][thin]") {
    struct PlanarPotential : Potentials::AnyPotential {
        PlanarPotential() {cutoff=2;}
        void getEFS(Atoms &a,AnyNeighborhoods &) override {
            const double residual=a.cell_(0,0)-3;a.energy_=.5*residual*residual;
            a.forces_.assign(a.size(),Eigen::Vector3d::Zero());a.stress_.setZero();a.stress_(0,0)=-a.cell_(0,0)*residual;
        }
    } pot;
    MetricFiles files;Eigen::VectorXd scale;Eigen::Matrix3d final;double energy=0;
    for(double vacuum:{20.,200.}) {
        Atoms a=metric_atoms();a.cell_=Eigen::Vector3d(4,5,vacuum).asDiagonal();a.pbc_={true,true,false};
        RelaxationDrvV2 d;basic(d,files);d.flags.cartesian=false;d.flags.periodic_subspace=true;
        d.flags.relax_cell=true;d.flags.relax_atom=false;d.control.thin_cell_scaling=true;d.control.max_iter=1;
        d.init(&pot,a);const Eigen::VectorXd this_scale=d.prec_scaling;d.run();
        REQUIRE(d.f<.5);REQUIRE(d.accepted_steps==1);REQUIRE(d.atom.cell_.row(2).isApprox(a.cell_.row(2)));
        if(vacuum==20) {scale=this_scale;final=d.atom.cell_;energy=d.f;}
        else {REQUIRE(this_scale.isApprox(scale));REQUIRE(d.atom.cell_.topRows(2).isApprox(final.topRows(2),1e-12));REQUIRE(d.f==Approx(energy).margin(1e-12));}
    }
}
TEST_CASE("Automatic AUL estimates actual constraint weights and reduces distance residual", "[relax_metric_driver][aul]") {
    MetricFiles files;DriverQuadratic pot;Atoms a=metric_atoms(2);a.pos_[0]={-.8,0,0};a.pos_[1]={1.2,0,0};
    std::istringstream text(R"({"version":1,"constraints":[{"kind":"distance","atoms":[0,1],"target":1,"residual_tolerance":1e-5}]})");
    auto constraints=Constraints::bind_constraints(Constraints::parse_constraints(text),a);
    RelaxationDrvV2 d;basic(d,files,"lbfgs");d.flags.with_constraint_aul=true;d.control.aul_auto=true;
    d.control.max_iter=500;d.control.max_move=.1;d.control.constraint_iter_limit=12;d.tol.gtol=1e-5;d.tol.gtol0=1e-3;
    d.init(&pot,a,&constraints);d.run();
    REQUIRE(d.state_vars.curr_constraint_iter>0);REQUIRE(d.aul_previous_residual.size()==1);
    REQUIRE(std::isfinite(d.constraints[0].weight));REQUIRE(d.constraints[0].weight>0);
    REQUIRE(std::abs(d.constraints[0].f)<1e-3);REQUIRE(std::abs((d.atom.pos_[1]-d.atom.pos_[0]).norm()-1)<1e-3);
}
TEST_CASE("Accepted driver steps recompute native dense or HVP force constants at current geometry", "[relax_metric_driver][fc]") {
    const bool dense=GENERATE(true,false);MetricFiles files;DriverQuadratic pot;pot.quartic=.3;pot.dense=dense;pot.hvp=true;
    Atoms a=metric_atoms();const Eigen::MatrixXd initial=pot.curvature(a);
    RelaxationDrvV2 d;basic(d,files);d.control.fc_update_interval=1;d.control.max_iter=3;
    d.init(&pot,a);d.run();REQUIRE(d.accepted_steps==3);REQUIRE_FALSE(d.full_hessian.isApprox(initial));
    REQUIRE(d.full_hessian.isApprox(pot.curvature(d.atom),1e-10));
    if(dense) {REQUIRE(pot.dense_calls==4);REQUIRE(pot.hvp_calls==0);}
    else {REQUIRE(pot.dense_calls==0);REQUIRE(pot.hvp_calls==12);}
    REQUIRE(d.atom.energy_<.5*a.pos_[0].squaredNorm()+.075*a.pos_[0].array().pow(4).sum());
    REQUIRE(d.convergence_gradient().isApprox(-d.atom.forces_[0],1e-10));
}

TEST_CASE("Few AUL constraint rows stay linear in the atom count", "[relax_metric_driver][aul][sparse]") {
    MetricFiles files;DriverQuadratic potential;Atoms a=metric_atoms(1500);
    RelaxationDrvV2 d;basic(d,files);d.flags.with_constraint_aul=true;
    regulator row;row.type=10;row.inputs={0,3*a.size()-1};row.outputs=row.inputs;row.x.resize(2);row.g.resize(2);
    row.compute=[](double *x,int,double *g,int,double *f,double target,Atoms *,int *,void *){g[0]=1;g[1]=2;*f=x[0]+2*x[1]-target;};
    std::vector<regulator> constraints{row};d.init(&potential,a,&constraints);d.eval();
    const auto point=d.x,gradient=d.g;const auto jac=d.constraint_derivatives();
    REQUIRE(jac.rows()==1);REQUIRE(jac.cols()==3*a.size());
    REQUIRE(jac(0,0)==1);REQUIRE(jac(0,jac.cols()-1)==2);
    REQUIRE(jac.array().abs().sum()==3);REQUIRE(d.x.isApprox(point));REQUIRE(d.g.isApprox(gradient));
}

TEST_CASE("Automatic diagonal uses the active mixed coordinate layout", "[relax_metric_driver][automatic]") {
    const int rank=GENERATE(0,1,2,3);
    const bool recast=GENERATE(false,true), atoms=GENERATE(false,true);
    INFO(rank << " recast=" << recast << " atoms=" << atoms);
    MetricFiles files;DriverQuadratic pot;Atoms a=metric_atoms();a.cell_.setZero();
    const Eigen::Matrix3d rows=(Eigen::Matrix3d()<<4,-4,0,4,4,0,0,0,5).finished();
    for(int k=0;k<3;++k) {a.pbc_[k]=k<rank;if(k<rank)a.cell_.row(k)=rows.row(k);}
    RelaxationDrvV2 d;basic(d,files,"lbfgs");d.flags.cartesian=false;
    d.flags.periodic_subspace=rank<3;d.flags.relax_cell=true;d.flags.relax_atom=atoms;
    d.flags.atomic_force_recast=recast;d.flags.dprecond=true;d.init(&pot,a);
    const int nc=rank==3?6:rank*(rank+1)/2;
    REQUIRE(d.prec_scaling.size()==nc+(atoms?3:0));
    REQUIRE(d.prec_scaling.allFinite());REQUIRE((d.prec_scaling.array()>0).all());
    REQUIRE(d.use_dprecond==!recast);REQUIRE(d.flags.dprecond);
    if(atoms) {
        const Eigen::Vector3d expected=recast?Eigen::Vector3d::Ones().eval():
            (rank<3?d.subspace.atomic_jacobian(a.cell_).rowwise().squaredNorm().eval():a.cell_.rowwise().squaredNorm().eval());
        REQUIRE(d.prec_scaling.tail<3>().isApprox(expected));
        // Open coordinates have Cartesian units, even with zero open cell rows.
        if(!recast && rank<3)REQUIRE(d.prec_scaling.tail(3-rank).isApprox(Eigen::VectorXd::Ones(3-rank)));
    }
    if(!recast && nc) {
        double measure=rank==1?std::sqrt(32.):rank==2?32.:160.;
        REQUIRE(d.prec_scaling[0]==Approx(160/eVA3_to_GPa*measure));
    }
    // Reusing a driver must derive the next scale from the new flags.
    d.flags.atomic_force_recast=!recast;d.init(&pot,a);
    REQUIRE(d.use_dprecond==recast);REQUIRE(d.flags.dprecond);
    d.flags.dprecond=false;d.init(&pot,a);REQUIRE_FALSE(d.use_dprecond);
    REQUIRE(d.prec_scaling.isApprox(Eigen::VectorXd::Ones(nc+(atoms?3:0))));
}

TEST_CASE("Automatic strain scale follows current settings and preserves coordinate masks on displacement", "[relax_metric_driver][automatic]") {
    MetricFiles files;DriverQuadratic pot;Atoms a=metric_atoms();a.pbc_={true,true,true};
    a.cell_<<4,-4,0,4,4,0,0,0,5;
    RelaxationDrvV2 d;basic(d,files,"lbfgs");d.flags.cartesian=false;d.flags.atomic_force_recast=false;
    d.flags.relax_cell=true;d.flags.dprecond=true;d.control.stiffness_str="160 320 480 640 800 960";
    d.init(&pot,a);
    const Eigen::VectorXd original=d.prec_scaling;
    const Eigen::Vector<double,6> expected=(Eigen::Vector<double,6>()<<160,960,800,320,640,480).finished()/eVA3_to_GPa*160;
    REQUIRE(d.prec_scaling.head(6).isApprox(expected));
    d.control.cell_factor=2;d.init(&pot,a);REQUIRE(d.prec_scaling.head(6).isApprox(.5*expected));
    d.flags.relax_cell=false;d.init(&pot,a);REQUIRE_FALSE(d.use_dprecond);REQUIRE(d.prec_scaling.isOnes());
    d.flags.relax_cell=true;d.flags.relax_cell_use_strain=false;d.init(&pot,a);
    REQUIRE_FALSE(d.use_dprecond);REQUIRE(d.prec_scaling.size()==12);REQUIRE(d.prec_scaling.isOnes());
    d.flags.relax_cell_use_strain=true;d.flags.selective_dynamics=true;
    a.sel_dynpos_={{1,0,1}};d.configure_cell_mask("0 0 0 0 0 0");
    d.flags.apply_disp=true;d.flags.disp_gaussian=true;d.control.disp=.001;
    d.control.stiffness_str.clear();d.init(&pot,a);REQUIRE(d.prec_scaling.size()==2);
    REQUIRE_NOTHROW(d.run());REQUIRE(d.prec_scaling.size()==2);
    REQUIRE(d.prec_scaling.isApprox(Eigen::Vector2d(32,25)));
    REQUIRE(d.atom.cell_.isApprox(a.cell_,1e-12));
}
