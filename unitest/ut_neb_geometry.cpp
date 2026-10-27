#include <catch2/catch_all.hpp>
#include "mindriver/neb.h"
#include "mindriver/neb_symmetry.h"
#include "tools/constants.h"

namespace {
// Periodic, translation-invariant pair energy with an independently known virial.
struct PairField : Potentials::AnyPotential {
    void calculate(Atoms &a) {
        a.resetEFS();
        for(int i=0;i<a.size();++i) for(int j=i+1;j<a.size();++j) {
            const Eigen::RowVector3d r=a.get_rab(i,j);
            a.energy_+=.5*r.squaredNorm();a.forces_[i]+=r.transpose();a.forces_[j]-=r.transpose();
            a.stress_-=r.transpose()*r;
        }
    }
    void getEFS(Atoms &a,AnyNeighborhoods &) override {calculate(a);}
    void getEFSOMP(Atoms &a) override {calculate(a);}
    void getEFSOMPthread(Atoms &a) override {calculate(a);}
};
std::vector<Atoms> band() {
    Atoms a;a.cell_=Eigen::Vector3d(6,7,8).asDiagonal();a.chemmap_={0,0};
    a.pos_={Eigen::RowVector3d(1.2,1.3,1.4),Eigen::RowVector3d(4.8,1.3,1.4)};
    std::vector<Atoms> result(5,a);
    for(int i=0;i<5;++i) {result[i].pos_[0][0]+=.05*i;result[i].pos_[1][0]-=.05*i;}
    return result;
}
}

TEST_CASE("NEB labelled geometry and symmetry are covariant under translations basis changes and global relabelling", "[neb][geometry][symmetry]") {
    auto source=band();PairField potential;NEBDriver base;base.pot=&potential;base.check_mic=false;base.init(source);base.eval();
    NEBPathSymmetry p(base,"");
    Eigen::RowVectorXd v(6);v<<.2,-.3,.7,-.6,.4,.1;
    for(int mode=0;mode<4;++mode) {
        auto transformed=source;Eigen::Matrix3d u=Eigen::Matrix3d::Identity();
        if(mode==1)u<<1,2,0,0,1,0,0,0,1;
        if(mode==2)u<<0,1,0,0,0,1,1,0,0;
        for(auto &a:transformed) {
            a.cell_=u*a.cell_;
            if(mode==0)for(auto &r:a.pos_)r+=Eigen::RowVector3d(.37,-.23,.41);
            if(mode==3)std::swap(a.pos_[0],a.pos_[1]);
        }
        NEBDriver d;d.pot=&potential;d.check_mic=false;d.init(transformed);d.eval();NEBPathSymmetry q(d,"");
        REQUIRE(q.atomic_dofs()==p.atomic_dofs());REQUIRE(q.common_operations.size()==p.common_operations.size());
        Eigen::RowVectorXd w=v;if(mode==3){w.head<3>()=v.tail<3>();w.tail<3>()=v.head<3>();}
        auto expected=p.project(v);if(mode==3){const auto old=expected;expected.head<3>()=old.tail<3>();expected.tail<3>()=old.head<3>();}
        REQUIRE(q.project(w).isApprox(expected,1e-10));
        for(int i=0;i<5;++i) {
            REQUIRE(d.images[i].energy_==Catch::Approx(base.images[i].energy_).margin(1e-10));
            for(int a=0;a<2;++a) REQUIRE(d.images[i].forces_[a].isApprox(base.images[i].forces_[mode==3?1-a:a],1e-10));
        }
        auto a=d.images.front(),b=d.images.back(),c=base.images.front(),e=base.images.back();
        REQUIRE(d.distance(a,b,0)==Catch::Approx(base.distance(c,e,0)).margin(1e-10));
    }
    auto reordered=source;std::swap(reordered[2].pos_[0],reordered[2].pos_[1]);
    NEBDriver other;other.pot=&potential;other.check_mic=false;other.init(reordered);
    REQUIRE(other.images[2].pos_==reordered[2].pos_);
    REQUIRE((other.x.row(2)-base.x.row(2)).norm()>1.);
}

TEST_CASE("NEB unfolded skew paths retain labelled winding and partial periodic boundaries", "[neb][geometry][mic]") {
    PairField pot;auto source=band();
    for(int i=0;i<5;++i) {
        source[i].cell_<<6,0,0,19,7,0,1,2,8;
        source[i].pbc_={true,true,false};source[i].pos_[0]+=double(i)*source[i].cell_.row(0);
        source[i].pos_[0][2]+=.3*i;
    }
    NEBDriver d;d.pot=&pot;d.check_mic=false;d.init(source);
    for(int i=0;i<5;++i)REQUIRE(d.images[i].pos_==source[i].pos_);
    NEBDriver mic;mic.pot=&pot;mic.init(source);
    REQUIRE((mic.images.back().pos_[0]-d.images.back().pos_[0]).norm()>20);
    REQUIRE(mic.images.back().pos_[0][2]==source.back().pos_[0][2]);
}

TEST_CASE("NEB approximate symmetry tolerance scans preserve projector and mapping contracts", "[neb][geometry][symmetry]") {
    PairField pot;
    for(double noise:{0.,2e-7,2e-5})for(double tolerance:{1e-7,1e-6,1e-4}) {
        auto source=band();source[2].pos_[0][2]+=noise;
        NEBDriver d;d.pot=&pot;d.check_mic=false;d.symprec=tolerance;d.init(source);NEBPathSymmetry p(d,"");
        Eigen::RowVectorXd v(6);v<<.2,.3,-.1,.7,.6,.4;const auto projected=p.project(v);
        REQUIRE(p.project(projected).isApprox(projected,1e-10));REQUIRE(std::abs(projected.dot(v-projected))<1e-10);
        for(int op:p.selected_operations) {
            std::vector<int> seen(2,0);for(int a=0;a<2;++a){const int b=p.initial.symop_mapping(op,a);REQUIRE(b>=0);REQUIRE(b<2);++seen[b];}
            REQUIRE(seen==std::vector<int>{1,1});
        }
        for(int left:p.selected_operations)for(int right:p.selected_operations) {
            const auto &[r1,t1]=p.initial.symm_ops[left];const auto &[r2,t2]=p.initial.symm_ops[right];
            const Eigen::Matrix3d rotation=r1*r2;const Eigen::RowVector3d translation=t1*r2+t2;
            bool found=false;
            for(int candidate:p.selected_operations) {
                const auto &[r,t]=p.initial.symm_ops[candidate];const Eigen::RowVector3d delta=translation-t;
                if((r-rotation).norm()>1e-12 || (delta-delta.array().round().matrix()).norm()>1e-8)continue;
                bool mapping=true;
                for(int a=0;a<2;++a)mapping &= p.initial.symop_mapping(candidate,a)==p.initial.symop_mapping(right,p.initial.symop_mapping(left,a));
                found |= mapping;
            }
            REQUIRE(found);
        }
        for(int i=0;i<5;++i)REQUIRE((p.project_point(d.x.row(i))-d.x.row(i)).norm()<4*tolerance);
    }
}

TEST_CASE("VC NEB physical enthalpy gradients obey finite difference pullbacks", "[neb][geometry][gradient]") {
    const bool recast=GENERATE(false,true),cholesky=GENERATE(false,true),constrained=GENERATE(false,true);
    PairField potential;auto source=band();source={source[0],source[2],source[4]};
    NEBDriver d;d.pot=&potential;d.vcneb=true;d.check_mic=false;d.recast_atomic_force=recast;
    d.restore_F_cholesky=cholesky;d.apply_scal=true;d.ciival={83,127,59,34,42,61};d.pressure=1.7;d.keep_symm=constrained;
    d.init(source);
    auto x=d.x;
    // Positive-definite strain with shear exercises the non-diagonal pullback.
    if(!constrained){x(1,0)+=.03;x(1,1)+=.012;x(1,2)-=.008;x(1,3)-=.02;}
    d.X2LatPos(x);d.x=x;d.eval();const auto analytic=d.g.row(1).eval();
    NEBPathSymmetry symmetry(d,"");
    for(int type=0;type<3;++type) {
        Eigen::RowVectorXd direction=Eigen::RowVectorXd::Zero(x.cols());
        for(int k=0;k<x.cols();++k)if(type==2 || (type==0)==(k>=6))direction[k]=std::sin(1.7*(k+1));
        if(constrained)direction=symmetry.project(direction);
        if(direction.norm()<1e-12)continue;
        direction.normalize();const auto native=direction.cwiseQuotient(d.scalval.transpose()).eval();
        const double expected=-analytic.dot(native);std::vector<double> derivatives,energies;
        for(double h:{1e-4,1e-5,1e-6}) {
            double values[2];for(int sign=0;sign<2;++sign) {
                auto trial=x;trial.row(1)+=(sign?1:-1)*h*native;d.X2LatPos(trial);d.eval();values[sign]=d.images[1].energy_;
            }
            derivatives.push_back((values[1]-values[0])/(2*h));energies.push_back(std::max(std::abs(values[0]),std::abs(values[1])));
        }
        for(int j=1;j<3;++j) {
            const double h=j==1?1e-5:1e-6;
            const double budget=3e-7+2e-6*std::abs(expected)+32*std::numeric_limits<double>::epsilon()*std::max(1.,energies[j])/h+std::abs(derivatives[j-1]-derivatives[j])/99.;
            REQUIRE(std::abs(derivatives[j]-expected)<=budget);
        }
    }
}

TEST_CASE("VC reference Cartesian metric is invariant under positive unimodular basis changes", "[neb][geometry][vcneb]") {
    PairField potential;auto source=band();NEBDriver base;base.pot=&potential;base.vcneb=true;base.check_mic=false;
    base.apply_scal=true;base.ciival={83,127,59,34,42,61};base.init(source);base.eval();
    NEBPathSymmetry p(base,"");Eigen::RowVectorXd v(base.x.cols());
    for(int j=0;j<v.size();++j)v[j]=std::sin(1.3*j);
    for(int mode=0;mode<2;++mode) {
        Eigen::Matrix3d u;if(mode==0)u<<1,2,0,0,1,0,0,0,1;else u<<0,1,0,0,0,1,1,0,0;
        auto path=base.images;for(auto &image:path)image.cell_=u*image.cell_;
        NEBDriver d;d.ApplySettings(base.CurrentSettings());d.ciival=base.ciival;d.pot=&potential;
        d.prepare_complete_path(path,u*base.refcell);d.eval();NEBPathSymmetry q(d,"");
        REQUIRE(d.x.isApprox(base.x,1e-10));REQUIRE(q.project(v).isApprox(p.project(v),1e-10));
        REQUIRE(d.g.isApprox(base.g,1e-10));
        for(int i=0;i<path.size();++i)REQUIRE(d.images[i].energy_==Catch::Approx(base.images[i].energy_).margin(1e-10));
    }
}
