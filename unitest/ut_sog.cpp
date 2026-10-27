// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/les.h"

using namespace Potentials;
using Catch::Approx;

namespace {
Atoms cell() {
    Atoms a;
    a.cell_ << 7.,0,0, .8,7.5,0, -.3,.6,8.;
    a.pos_={{1.1,1.3,1.7},{3.2,2.1,2.4},{2.4,4.1,3.3}};
    a.chemmap_={0,0,0};
    return a;
}
LES sog() {
    LES l;
    l.kernel=LES::Kernel::SOG;
    l.sog_weights={4.2,-1.3};
    l.sog_log_bandwidths={std::log(.7),std::log(1.2)};
    l.dl=1.0;
    return l;
}
Eigen::VectorXd charges() { Eigen::VectorXd q(3); q<<.7,-.3,.4; return q; }
double reference(const Atoms& a,const Eigen::VectorXd& q,const LES& l) {
    double e=0.;
    for(int nx=-4;nx<=4;++nx) for(int ny=-4;ny<=4;++ny) for(int nz=-4;nz<=4;++nz)
        for(int i=0;i<a.size();++i) for(int j=0;j<a.size();++j) {
            if(l.remove_self_interaction && i==j && nx==0 && ny==0 && nz==0) continue;
            Eigen::Vector3d r=(a.pos_[i]-a.pos_[j]+nx*a.cell_.row(0)+ny*a.cell_.row(1)+nz*a.cell_.row(2)).transpose();
            for(size_t m=0;m<l.sog_weights.size();++m) {
                double s=std::exp(l.sog_log_bandwidths[m]);
                e+=.5*q(i)*q(j)*l.sog_weights[m]*s*s*s/(8*std::pow(M_PI,1.5))*std::exp(-.25*s*s*r.squaredNorm());
            }
        }
    return e;
}
}

TEST_CASE("SOG reciprocal energy converges to independent periodic Gaussian sum","[sog]") {
    auto l=sog(); auto a=cell(); auto q=charges();
    l.remove_self_interaction=GENERATE(false,true);
    l.dl=2.;
    const double exact=reference(a,q,l);
    const double coarse=std::abs(l.evaluate(a,q).energy-exact);
    l.dl=1.;
    const auto r=l.evaluate(a,q);
    REQUIRE(r.energy==Approx(exact).margin(1e-10));
    REQUIRE(std::abs(r.energy-exact)<coarse);
    REQUIRE(r.per_atom_energy.sum()==Approx(r.energy).margin(1e-12));
}

TEST_CASE("SOG forces virial and charge derivatives differentiate the defined energy","[sog][grad]") {
    auto l=sog(); auto a=cell(); auto q=charges();
    l.sog_include_zero=GENERATE(false,true);
    l.remove_self_interaction=GENERATE(false,true);
    l.dl=2.3; // Exercise the taper, not just the untruncated Gaussian.
    const auto r=l.evaluate(a,q,true);
    constexpr double h=1e-6;
    for(int i=0;i<a.size();++i) {
        auto qp=q,qm=q;qp(i)+=h;qm(i)-=h;
        auto ep=l.evaluate(a,qp),em=l.evaluate(a,qm);
        REQUIRE(r.dEdq(i)==Approx((ep.energy-em.energy)/(2*h)).margin(1e-9));
        for(int j=0;j<a.size();++j)
            REQUIRE(r.charge_hessian(j,i)==Approx((ep.dEdq(j)-em.dEdq(j))/(2*h)).margin(1e-9));
        for(int d=0;d<3;++d) {
            auto ap=a,am=a;ap.pos_[i](d)+=h;am.pos_[i](d)-=h;
            REQUIRE(r.forces(i,d)==Approx(-(l.evaluate(ap,q).energy-l.evaluate(am,q).energy)/(2*h)).margin(1e-9));
        }
    }
    for(int i=0;i<3;++i) for(int j=0;j<3;++j) {
        auto ap=a,am=a;Eigen::Matrix3d dp=Eigen::Matrix3d::Identity(),dm=dp;
        dp(i,j)+=h;dm(i,j)-=h;
        ap.cell_=a.cell_*dp.transpose();am.cell_=a.cell_*dm.transpose();
        for(int n=0;n<a.size();++n){ap.pos_[n]=a.pos_[n]*dp.transpose();am.pos_[n]=a.pos_[n]*dm.transpose();}
        REQUIRE(r.virial(i,j)==Approx(-(l.evaluate(ap,q).energy-l.evaluate(am,q).energy)/(2*h)).margin(1e-9));
    }
    REQUIRE(r.forces.colwise().sum().norm()<1e-12);
}

TEST_CASE("SOG kernel VJP includes mixed response and self derivatives","[sog][grad]") {
    auto l=sog();auto a=cell();auto q=charges();
    l.sog_include_zero=GENERATE(false,true);
    l.remove_self_interaction=GENERATE(false,true);
    l.dl=2.3;
    Eigen::Matrix<double,-1,3> fw(3,3);fw<<.2,-.1,.3,-.5,.7,.1,.3,.2,-.4;
    Eigen::Matrix3d sw;sw<<.1,.2,-.1,.3,-.2,.5,-.1,.2,.7;
    Eigen::VectorXd response(3);response<<.2,-.6,.1;
    std::vector<double> grad(4,0.);
    l.accumulate_kernel_vjp(a,q,.37,fw,sw,response,grad.data());
    auto loss=[&](LES& model){auto r=model.evaluate(a,q);return .37*r.energy+(r.forces.array()*fw.array()).sum()+(r.virial.array()*sw.array()).sum()+response.dot(r.dEdq);};
    for(int m=0;m<4;++m) {
        auto p=l,n=l;double h=1e-6;
        if(m<2){p.sog_weights[m]+=h;n.sog_weights[m]-=h;}
        else{p.sog_log_bandwidths[m-2]+=h;n.sog_log_bandwidths[m-2]-=h;}
        REQUIRE(grad[m]==Approx((loss(p)-loss(n))/(2*h)).margin(1e-9));
    }
    Eigen::VectorXd direct;
    l.compute_les_dLdq_from_forces(a,q,fw,sw,direct);
    auto fixed_loss=[&](const Eigen::VectorXd& v){auto r=l.evaluate(a,v);return (r.forces.array()*fw.array()).sum()+(r.virial.array()*sw.array()).sum();};
    for(int i=0;i<3;++i){auto p=q,n=q;p(i)+=1e-6;n(i)-=1e-6;REQUIRE(direct(i)==Approx((fixed_loss(p)-fixed_loss(n))/2e-6).margin(1e-9));}
}

TEST_CASE("SOG zero mode and periodic replication have explicit semantics","[sog]") {
    auto l=sog();auto a=cell();auto q=charges();
    const auto included=l.evaluate(a,q);
    l.sog_include_zero=false;
    const auto omitted=l.evaluate(a,q);
    const double g0=l.sog_weights[0]+l.sog_weights[1];
    const double e0=.5*g0*q.sum()*q.sum()/a.get_volume();
    REQUIRE(included.energy-omitted.energy==Approx(e0).margin(1e-12));
    REQUIRE((included.virial-omitted.virial-e0*Eigen::Matrix3d::Identity()).norm()<1e-12);
    REQUIRE((included.forces-omitted.forces).norm()<1e-12);
    REQUIRE((included.dEdq-omitted.dEdq-Eigen::VectorXd::Constant(3,g0*q.sum()/a.get_volume())).norm()<1e-12);
    l.sog_include_zero=true;
    auto b=a;b.cell_.row(0)*=2;
    for(int i=0;i<a.size();++i){b.pos_.push_back(a.pos_[i]+a.cell_.row(0));b.chemmap_.push_back(0);}
    Eigen::VectorXd qb(6);qb<<q,q;
    auto replicated=l.evaluate(b,qb);
    REQUIRE(replicated.energy==Approx(2*included.energy).margin(1e-12));
    REQUIRE((replicated.forces.topRows(3)-included.forces).norm()<1e-12);
    l.sog_log_bandwidths[0]=std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(l.evaluate(a,q),GPException);
    l=sog();l.sog_weights.pop_back();REQUIRE_THROWS_AS(l.evaluate(a,q),GPException);
}
