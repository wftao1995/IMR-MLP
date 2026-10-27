#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BispectrumSO4.h"
#include "unitest/pair_cell_fixture.h"

namespace {
GPSettings settings() {
    GPSettings s;s.Modify("cutoff","3.2");s.Modify("twojmax","2");
    s.Modify("weights","0.7 1.3");s.Modify("selfweights","0.4 0.8");
    s.Modify("average","inner-density");return s;
}
Eigen::VectorXd values(Descriptor_BispectrumSO4 &d,const Atoms &a) {
    Neighborhoods nbhs(a,d.cutoff_);Eigen::VectorXd out;d.compute_global_density(nbhs,out);return out;
}
Atoms displaced(const Atoms &a,const Eigen::VectorXd &q,bool cell) {
    Atoms out=a;const int offset=cell?9:0;
    if (cell) {
        out.cell_+=Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(q.data());
        for (int i=0;i<a.size();++i) out.pos_[i]=a.pos_[i]*a.cell_.inverse()*out.cell_;
    }
    for (int i=0;i<a.size();++i) out.pos_[i]+=q.segment<3>(offset+3*i).transpose();
    return out;
}
}

TEST_CASE("SO4 global density differs from local mean and has geometric invariances",
          "[descriptor][so4][global-density]") {
    Descriptor_BispectrumSO4 d(settings());auto a=PairCellTest::geometry(true);
    const auto reference=values(d,a);
    Neighborhoods sites(a,d.cutoff_);Eigen::VectorXd local,mean=Eigen::VectorXd::Zero(d.size());
    for (int i=0;i<sites.size();++i) {d.compute(sites[i],local);mean+=local/double(a.size());}
    REQUIRE((reference-mean).norm()>1e-5);
    auto shift=a;for (auto &p:shift.pos_)p+=Eigen::RowVector3d(.3,-.4,.6);
    REQUIRE((values(d,shift)-reference).norm()<1e-11);
    auto rotated=a;
    const Eigen::Matrix3d rotation=Eigen::AngleAxisd(.7,Eigen::Vector3d(1,2,3).normalized()).toRotationMatrix();
    rotated.cell_=a.cell_*rotation;for (auto &p:rotated.pos_)p=p*rotation;
    REQUIRE((values(d,rotated)-reference).norm()<1e-10);
    auto reordered=a;std::swap(reordered.pos_[0],reordered.pos_[1]);std::swap(reordered.chemmap_[0],reordered.chemmap_[1]);
    REQUIRE((values(d,reordered)-reference).norm()<1e-11);
    auto replica=a;replica.cell_.row(0)*=2.;
    for(int i=0;i<a.size();++i) {replica.pos_.push_back(a.pos_[i]+a.cell_.row(0));replica.chemmap_.push_back(a.chemmap_[i]);}
    REQUIRE((values(d,replica)-reference).norm()<1e-10);
    Atoms empty;REQUIRE(values(d,empty).isZero());
    Eigen::VectorXd empty_value;Eigen::Tensor<double,3,Eigen::RowMajor> empty_cell;
    d.compute_global_density_cell(empty,empty_value,empty_cell);REQUIRE(empty_value.isZero());
    double e;Eigen::VectorXd g;Eigen::MatrixXd h;
    d.compute_weighted_global_hessian(empty,Eigen::VectorXd::Ones(d.size()),true,e,g,h);
    REQUIRE(e==0.);REQUIRE(g.size()==9);REQUIRE(g.isZero());REQUIRE(h.isZero());
}

TEST_CASE("SO4 global atom and strain Jacobians include repeated and self images",
          "[descriptor][so4][global-density][derivative]") {
    Descriptor_BispectrumSO4 d(settings());auto a=PairCellTest::geometry();
    a.cell_.row(0)*=.55;
    Neighborhoods sites(a,d.cutoff_);Eigen::VectorXd v;
    Eigen::Tensor<double,3,Eigen::RowMajor> jac,strain;
    d.compute_global_density(sites,v,&jac,&strain);
    Eigen::VectorXd cv;Eigen::Tensor<double,3,Eigen::RowMajor> cell_jac;
    d.compute_global_density_cell(a,cv,cell_jac);
    REQUIRE((cv-v).norm()<1e-12);
    for(double h:{5e-5,2e-5}) {
        for(int i=0;i<a.size();++i) for(int axis=0;axis<3;++axis) {
            auto p=a,m=a;p.pos_[i][axis]+=h;m.pos_[i][axis]-=h;
            const Eigen::VectorXd fd=(values(d,p)-values(d,m))/(2*h);
            for(int f=0;f<v.size();++f) REQUIRE(jac(f,i,axis)==Catch::Approx(fd[f]).margin(2e-7*(1.+fd.norm())));
        }
        for(int axis=0;axis<3;++axis) for(int b=0;b<3;++b) {
            auto p=a,m=a;Eigen::Matrix3d fp=Eigen::Matrix3d::Identity(),fm=fp;
            fp(b,axis)+=h;fm(b,axis)-=h;
            p.cell_=a.cell_*fp;m.cell_=a.cell_*fm;
            for(int i=0;i<a.size();++i) {p.pos_[i]=a.pos_[i]*fp;m.pos_[i]=a.pos_[i]*fm;}
            const Eigen::VectorXd fd=(values(d,p)-values(d,m))/(2*h);
            for(int f=0;f<v.size();++f) REQUIRE(strain(f,axis,b)==Catch::Approx(fd[f]).margin(2e-7*(1.+fd.norm())));
            Eigen::VectorXd qp=Eigen::VectorXd::Zero(9+3*a.size());qp[3*axis+b]=h;
            const Eigen::VectorXd cell_fd=(values(d,displaced(a,qp,true))-values(d,displaced(a,-qp,true)))/(2*h);
            for(int f=0;f<v.size();++f) REQUIRE(cell_jac(f,axis,b)==Catch::Approx(cell_fd[f]).margin(2e-7*(1.+cell_fd.norm())));
        }
    }
}

TEST_CASE("SO4 weighted global Hessian includes cross center and raw cell coupling",
          "[descriptor][so4][global-density][curvature]") {
    const bool cell=GENERATE(false,true),partial=GENERATE(false,true);
    Descriptor_BispectrumSO4 d(settings());auto a=PairCellTest::geometry(partial);
    const Eigen::VectorXd w=Eigen::VectorXd::LinSpaced(d.size(),-.02,.07);
    const int dim=3*a.size()+(cell?9:0);
    Eigen::VectorXd v=Eigen::VectorXd::LinSpaced(dim,-.13,.19),g,hv;
    double energy;d.compute_weighted_global_hvp(a,w,v,cell,energy,g,hv);
    REQUIRE(energy==Catch::Approx(w.dot(values(d,a))).margin(1e-11));
    Eigen::MatrixXd H;Eigen::VectorXd hg;double he;
    d.compute_weighted_global_hessian(a,w,cell,he,hg,H);
    REQUIRE((H-H.transpose()).norm()<1e-10*(1.+H.norm()));
    REQUIRE((H*v-hv).norm()<1e-10*(1.+hv.norm()));
    for(double h:{7e-5,3e-5}) {
        auto p=displaced(a,(h*v).eval(),cell),m=displaced(a,(-h*v).eval(),cell);
        // Energy is evaluated independently through the structure fingerprint.
        const double ep=w.dot(values(d,p)),em=w.dot(values(d,m));
        REQUIRE(g.dot(v)==Catch::Approx((ep-em)/(2*h)).margin(2e-7*(1.+g.norm())));
        REQUIRE(v.dot(hv)==Catch::Approx((ep-2*energy+em)/(h*h)).margin(2e-5*(1.+hv.norm())));
        for(const auto &[row,column]:std::vector<std::pair<int,int>>{{0,dim-1},{1,dim-4},{dim-3,dim-5}}) {
            Eigen::VectorXd q=Eigen::VectorXd::Zero(dim);q[row]=h;q[column]=h;
            const double pp=w.dot(values(d,displaced(a,q,cell)));
            q[column]=-h;const double pm=w.dot(values(d,displaced(a,q,cell)));
            q[row]=-h;const double mm=w.dot(values(d,displaced(a,q,cell)));
            q[column]=h;const double mp=w.dot(values(d,displaced(a,q,cell)));
            REQUIRE(H(row,column)==Catch::Approx((pp-pm+mm-mp)/(4*h*h)).margin(2e-5*(1.+H.norm())));
        }
        if (!cell) {
            Eigen::VectorXd gp,gm,unused;double e;
            d.compute_weighted_global_hvp(p,w,v,false,e,gp,unused);
            d.compute_weighted_global_hvp(m,w,v,false,e,gm,unused);
            REQUIRE((hv-(gp-gm)/(2*h)).norm()<2e-6*(1.+hv.norm()));
        }
    }
    Eigen::VectorXd translation=Eigen::VectorXd::Zero(dim);
    for(int i=0;i<a.size();++i)translation[(cell?9:0)+3*i+1]=1.;
    REQUIRE((H*translation).norm()<1e-10*(1.+H.norm()));
}
