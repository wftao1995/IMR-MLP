#include <catch2/catch_all.hpp>
#include "mindriver/relax_new.h"

using Catch::Approx;
namespace {
struct RefreshPotential : Potentials::AnyPotential {
    bool dense=true,fail_native=false,forbid_efs=false;
    int dense_calls=0,hvp_calls=0,efs_calls=0;
    RefreshPotential() {cutoff=3;}
    Eigen::Matrix3d curvature(const Atoms &a) const {
        return (Eigen::Vector3d::Constant(2)+.9*a.pos_[0].array().square().matrix().transpose()).asDiagonal();
    }
    void getEFS(Atoms &a,AnyNeighborhoods &) override {
        ++efs_calls;if(forbid_efs)throw std::runtime_error("Unexpected duplicate EFS during FC refresh");
        const Eigen::Vector3d x=a.pos_[0];a.energy_=x.squaredNorm()+.075*x.array().pow(4).sum();
        a.forces_={-(2*x+.3*x.array().cube().matrix())};a.stress_.setZero();
    }
    bool supports_analytical_hessian() const override {return dense;}
    bool supports_analytical_hessian_vector_product() const override {return true;}
    void getEFSHessian(Atoms &a,AnyNeighborhoods &,Potentials::HessianMatrix &out) override {
        ++dense_calls;if(fail_native)throw std::runtime_error("Native FC failure");out.resize(a.size());out(0,0)=curvature(a);
    }
    void applyHessianVectorProduct(Atoms &a,AnyNeighborhoods &,const Eigen::VectorXd &v,Eigen::VectorXd &out) override {
        ++hvp_calls;if(fail_native)throw std::runtime_error("Native HVP failure");out=curvature(a)*v;
    }
};
Atoms refresh_atom() {
    Atoms a;a.cell_=8*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};a.pos_={{1,.5,-.2}};a.chemmap_={0};return a;
}
void initialize(RelaxationDrvV2 &d,RefreshPotential &p,Atoms &a) {
    d.flags.cartesian=true;d.flags.dprecond=false;d.control.driver="gd";d.control.fc_update_interval=1;
    d.control.max_iter=0;d.init(&p,a);d.run();
}
FilterWhitening *white(RelaxationDrvV2 &d) {
    auto *filter=dynamic_cast<FilterWhitening *>(d.modifiers_list.back());REQUIRE(filter);return filter;
}
void next_trial(RelaxationDrvV2 &d) {d.x-=.02*d.g;d.eval();}
void replay_checkpoint(RelaxationDrvV2 &d) {
    const Atoms accepted=d.checkpoint_atom;const Eigen::VectorXd gradient=d.checkpoint_g;const double f=d.checkpoint_f;
    d.x=d.checkpoint_x;d.eval();REQUIRE(d.atom.pos_[0].isApprox(accepted.pos_[0],1e-12));
    REQUIRE(d.atom.cell_.isApprox(accepted.cell_,1e-12));REQUIRE(d.g.isApprox(gradient,1e-11));REQUIRE(d.f==Approx(f).margin(1e-12));
}
}
TEST_CASE("Accepted native FC refresh reexpresses coordinates without duplicate EFS", "[relax_fc_failure][transaction]") {
    const bool dense=GENERATE(true,false);RefreshPotential p;p.dense=dense;Atoms a=refresh_atom();RelaxationDrvV2 d;initialize(d,p,a);
    next_trial(d);const Atoms trial=d.atom;const double f=d.f;const Eigen::VectorXd physical_gradient=d.convergence_gradient();
    const int calls=p.efs_calls;p.forbid_efs=true;
    REQUIRE_NOTHROW(d.accept_checkpoint());REQUIRE(p.efs_calls==calls);REQUIRE(d.fc_refresh_count==2);REQUIRE(d.accepted_steps==1);
    REQUIRE(d.x.isZero(0));REQUIRE(d.atom.pos_[0].isApprox(trial.pos_[0]));REQUIRE(d.atom.energy_==trial.energy_);
    REQUIRE(d.atom.forces_[0].isApprox(trial.forces_[0]));REQUIRE(d.atom.stress_.isApprox(trial.stress_));REQUIRE(d.f==f);
    REQUIRE(d.convergence_gradient().isApprox(physical_gradient,1e-12));
    REQUIRE(d.full_hessian.isApprox(p.curvature(d.atom),1e-12));
    if(dense) {REQUIRE(p.dense_calls==2);REQUIRE(p.hvp_calls==0);}else {REQUIRE(p.hvp_calls==6);REQUIRE(p.dense_calls==0);}
    p.forbid_efs=false;replay_checkpoint(d);
}
TEST_CASE("Native FC failure preserves metric checkpoint and permits accepted-step retry", "[relax_fc_failure][transaction]") {
    const bool dense=GENERATE(true,false);RefreshPotential p;p.dense=dense;Atoms a=refresh_atom();RelaxationDrvV2 d;initialize(d,p,a);
    auto *filter=white(d);const Eigen::MatrixXd metric=filter->transform.metric().matrix(),h=d.full_hessian,reduced=d.reduced_hessian;
    const Eigen::VectorXd checkpoint=d.checkpoint_x,gradient=d.checkpoint_g;const auto list=d.modifiers_list;
    next_trial(d);p.fail_native=true;const int calls=p.efs_calls;
    REQUIRE_THROWS(d.accept_checkpoint());REQUIRE(p.efs_calls==calls);REQUIRE(d.fc_refresh_count==1);REQUIRE(d.accepted_steps==0);
    REQUIRE(d.modifiers_list==list);REQUIRE(filter->transform.metric().matrix().isApprox(metric));
    REQUIRE(d.full_hessian.isApprox(h));REQUIRE(d.reduced_hessian.isApprox(reduced));
    REQUIRE(d.checkpoint_x.isApprox(checkpoint));REQUIRE(d.checkpoint_g.isApprox(gradient));REQUIRE(d.fc_update.due(1));
    d.restore_checkpoint();replay_checkpoint(d);p.fail_native=false;next_trial(d);REQUIRE_NOTHROW(d.accept_checkpoint());
    REQUIRE(d.accepted_steps==1);REQUIRE(d.fc_refresh_count==2);replay_checkpoint(d);
}
TEST_CASE("Failure constructing a candidate FC whitening leaves old coordinate representation intact", "[relax_fc_failure][transaction]") {
    RefreshPotential p;Atoms a=refresh_atom();RelaxationDrvV2 d;initialize(d,p,a);next_trial(d);
    auto *filter=white(d);const auto list=d.modifiers_list;
    const Eigen::MatrixXd metric=filter->transform.metric().matrix(),h=d.full_hessian,reduced=d.reduced_hessian;
    const Eigen::VectorXd point=filter->point,x=d.x,g=d.g;const Atoms trial=d.atom;const double f=d.f;
    d.accepted_steps=1;d.control.precond_floor=-1;const int calls=p.efs_calls;
    REQUIRE_THROWS(d.refresh_fc_metric());REQUIRE(p.efs_calls==calls);REQUIRE(d.fc_refresh_count==1);REQUIRE(d.fc_update.due(1));
    REQUIRE(d.modifiers_list==list);REQUIRE(d.x.isApprox(x));REQUIRE(d.g.isApprox(g));REQUIRE(filter->point.isApprox(point));
    REQUIRE(filter->transform.metric().matrix().isApprox(metric));REQUIRE(d.full_hessian.isApprox(h));REQUIRE(d.reduced_hessian.isApprox(reduced));
    REQUIRE(d.atom.pos_[0].isApprox(trial.pos_[0]));REQUIRE(d.f==f);
    d.control.precond_floor=1e-8;d.accepted_steps=0;d.restore_checkpoint();replay_checkpoint(d);
    next_trial(d);REQUIRE_NOTHROW(d.accept_checkpoint());REQUIRE(d.fc_refresh_count==2);replay_checkpoint(d);
}
