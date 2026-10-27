#include <catch2/catch_all.hpp>
#include "unitest/les_curvature_fixture.h"
#include <future>

using namespace Potentials;
using namespace LESCurvature;
namespace {
struct Inputs {
    Atoms atom=geometry();
    Eigen::VectorXd q=Eigen::VectorXd(4),qd=Eigen::VectorXd(4),qw=Eigen::VectorXd(4),qvw=Eigen::VectorXd(4);
    SiteCoordinateMatrix v=SiteCoordinateMatrix(4,3),w=SiteCoordinateMatrix(4,3);
    Eigen::Matrix3d dc=Eigen::Matrix3d::Zero();
    Inputs() {
        q<<.31,-.17,.23,.11;qd<<-.13,.07,.19,-.11;
        qw<<.04,-.12,.11,.08;qvw<<.03,.07,-.04,.02;
        for(int i=0;i<4;++i) for(int d=0;d<3;++d) {
            v(i,d)=.2*std::sin(3*i+d+1.);w(i,d)=.3*std::cos(3*i+d+1.);
        }
        dc<<.07,.02,-.01,-.03,.04,.02,.01,-.02,.03;
    }
};
// A single checksum exercises every reciprocal derivative API without sharing
// any caller-owned output or scratch across workers.
Eigen::VectorXd evaluate_all(const LES& engine,const Inputs& x,LES::Workspace& workspace,int channel) {
    const auto e=engine.evaluate(x.atom,x.q,true,workspace,channel);
    const auto h=engine.apply_charge_hessian(x.atom,x.qd,workspace,channel);
    const auto r=engine.evaluate_cartesian_directional(x.atom,x.q,x.qd,x.v,workspace,channel);
    const auto n=engine.evaluate_native_directional(x.atom,x.q,x.qd,x.v,x.dc,workspace,channel);
    Eigen::VectorXd dq;engine.compute_les_dLdq_from_forces(x.atom,x.q,x.v,x.dc,dq,workspace,channel);
    Eigen::VectorXd gradient=Eigen::VectorXd::Zero(2*engine.sog_weights.size());
    engine.accumulate_kernel_vjp(x.atom,x.q,.37,x.v,x.dc,x.qw,gradient.data(),workspace,channel);
    Eigen::VectorXd result(14);
    result<<e.energy,e.forces.squaredNorm(),e.virial.squaredNorm(),e.dEdq.sum(),
        e.charge_hessian.squaredNorm(),e.per_atom_energy.sum(),h.sum(),
        r.force_directional.squaredNorm(),r.dEdq_directional.sum(),
        n.forces.squaredNorm(),n.force_directional.squaredNorm(),n.virial_directional.squaredNorm(),
        dq.sum(),gradient.sum();
    return result;
}
}

TEST_CASE("LES shared owner has independent explicit channel workspaces","[les][workspace]") {
    const bool sog=GENERATE(false,true);
    LES engine;configure_engine(engine,sog);
    if(sog) {
        engine.sog_channels=2;engine.sog_channel=1;
        engine.sog_weights={4.2,-1.3,3.1,-.8};
        engine.sog_log_bandwidths={-.4,.2,-.2,.1};
    }
    Inputs x;engine.evaluate(x.atom,x.q,true);
    const auto grid=engine.kvecs;const auto factors=engine.kvec_factors;
    const auto kernel=engine.k_kernel,derivative=engine.k_kernel_derivative;
    const auto taper=engine.reciprocal_taper();const int selected=engine.sog_channel;
    const LES& shared=engine;
    const int channels=sog ? 2 : 1;
    std::vector<Eigen::VectorXd> expected;
    for(int c=0;c<channels;++c) {
        LES::Workspace fresh;expected.push_back(evaluate_all(shared,x,fresh,c));
        LES legacy;configure_engine(legacy,sog);
        if(sog) {legacy.sog_channels=2;legacy.sog_weights=engine.sog_weights;
            legacy.sog_log_bandwidths=engine.sog_log_bandwidths;legacy.sog_channel=c;}
        const auto e=legacy.evaluate(x.atom,x.q,true);
        LES::Workspace workspace;const auto actual=shared.evaluate(x.atom,x.q,true,workspace,c);
        REQUIRE(actual.energy==Catch::Approx(e.energy).margin(1e-13));
        REQUIRE((actual.forces-e.forces).norm()<1e-13);
        REQUIRE((actual.virial-e.virial).norm()<1e-13);
        REQUIRE((actual.charge_hessian-e.charge_hessian).norm()<1e-13);
        const auto old_native=legacy.evaluate_native_directional(x.atom,x.q,x.qd,x.v,x.dc);
        const auto native=shared.evaluate_native_directional(x.atom,x.q,x.qd,x.v,x.dc,workspace,c);
        REQUIRE((native.force_directional-old_native.force_directional).norm()<1e-13);
    }
    std::vector<std::future<Eigen::VectorXd>> workers;
    for(int i=0;i<6;++i) workers.push_back(std::async(std::launch::async,[&,i] {
        LES::Workspace workspace;Eigen::VectorXd out;
        for(int repeat=0;repeat<4;++repeat) out=evaluate_all(shared,x,workspace,i%channels);
        return out;
    }));
    for(int i=0;i<6;++i) REQUIRE((workers[i].get()-expected[i%channels]).norm()<1e-12);
    REQUIRE(engine.sog_channel==selected);REQUIRE(engine.kvecs==grid);
    REQUIRE(engine.kvec_factors==factors);REQUIRE(engine.k_kernel==kernel);
    REQUIRE(engine.k_kernel_derivative==derivative);REQUIRE(engine.reciprocal_taper()==taper);
    LES::Workspace workspace;
    Eigen::MatrixXd q(4,channels);for(int c=0;c<channels;++c) q.col(c)=(1.+c)*x.q;
    const auto aggregate=shared.evaluate_channels(x.atom,q,workspace);
    double energy=0.;for(int c=0;c<channels;++c) energy+=shared.evaluate(x.atom,q.col(c),workspace,c).energy;
    REQUIRE(aggregate.energy==Catch::Approx(energy).margin(1e-13));
    REQUIRE((shared.apply_charge_hessian_channels(x.atom,q,workspace)-engine.apply_charge_hessian_channels(x.atom,q)).norm()<1e-13);
    if(sog) {
        const auto explicit_components=shared.kernel_components(x.atom,q,workspace);
        const auto old_components=engine.kernel_components(x.atom,q);
        REQUIRE((explicit_components.efs-old_components.efs).norm()<1e-13);
    }
}

TEST_CASE("LES workspace cache keys follow geometry and physical settings","[les][workspace][cache]") {
    LES engine;configure_engine(engine,true);Inputs x;LES::Workspace reused;
    auto check=[&] {
        LES::Workspace fresh;
        const auto a=evaluate_all(engine,x,reused,0),b=evaluate_all(engine,x,fresh,0);
        REQUIRE((a-b).norm()<1e-12*std::max(1.,b.norm()));
    };
    check();x.atom.cell_(0,1)+=.17;check();x.atom.pos_[0][1]+=.13;check();
    engine.dl=2.3;check();engine.taper_fraction=.27;check();
    engine.sog_weights[0]+=.41;check();engine.sog_log_bandwidths[1]-=.11;check();
    engine.sog_include_zero=false;check();engine.remove_self_interaction=false;check();
    engine.sog_channels=2;check();engine.sog_channels=1;check();
    engine.use_cache=false;check();engine.use_cache=true;check();
    engine.use_recurrence=false;check();engine.use_recurrence=true;check();
    engine.kernel=LES::Kernel::Coulomb;engine.lr_sigma=.83;check();engine.lr_sigma=1.14;check();
    // Hard cutoffs support fixed-cell derivatives, but never cell curvature.
    engine.cutoff_mode=LES::CutoffMode::ReferenceHard;
    LES::Workspace fresh;
    REQUIRE((engine.evaluate(x.atom,x.q,true,reused).forces-engine.evaluate(x.atom,x.q,true,fresh).forces).norm()<1e-13);
    REQUIRE_THROWS(engine.evaluate_native_directional(x.atom,x.q,x.qd,x.v,x.dc,reused));
}
