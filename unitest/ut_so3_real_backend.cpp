#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BispectrumSO3.h"
#include "descriptors/descriptor_wrapper.h"
#include <Eigen/Geometry>

namespace {
Neighborhood real_so3_site() {
    Neighborhood site;site.centertype=0;
    for(const auto& p:std::vector<Eigen::RowVector3d>{{.71,.12,.38},{-.23,1.14,.51},
        {.19,-.42,1.67},{1.31,.65,-.36},{-.83,-1.07,-.29}}) {
        site.pos_.push_back(p);site.distances_.push_back(p.norm());site.indices_.push_back(site.nbhcount_);
        site.chemmap_.push_back(0);++site.nbhcount_;
    }
    return site;
}
Eigen::VectorXd tensor_vector(const DescriptorGradient& gradient) {
    return Eigen::Map<const Eigen::VectorXd>(gradient.grad.data(),gradient.grad.size());
}
void move_so3_site(Neighborhood& site,const DescriptorSiteMatrix& direction,double step) {
    for(int i=0;i<site.nbhcount_;++i) {site.pos_[i]+=step*direction.row(i);site.distances_[i]=site.pos_[i].norm();}
}
}

TEST_CASE("SO3 real coupling retains complete complex output layout and derivatives","[descriptor][so3][real]") {
    const int lmax=GENERATE(1,3,4);
    Descriptor_BispectrumSO3 reference(lmax,0.,4.,2),real(lmax,0.,4.,2,true);
    auto site=real_so3_site();
    if(lmax==3) {site.pos_[0]={1e-12,-2e-12,1.1};site.distances_[0]=site.pos_[0].norm();}
    Eigen::VectorXd a,b;DescriptorGradient ga,gb;ComputeOptions oa,ob;
    oa.compute_derivative=ob.compute_derivative=true;oa.gradient=&ga;ob.gradient=&gb;
    reference.compute(site,a,oa);real.compute(site,b,ob);
    REQUIRE(real.uses_real_harmonics());REQUIRE_FALSE(reference.uses_real_harmonics());
    REQUIRE((a-b).norm()<3e-11*std::max(1.,a.norm()));
    REQUIRE((tensor_vector(ga)-tensor_vector(gb)).norm()<2e-10*std::max(1.,tensor_vector(ga).norm()));
    // The count is observable work, not a timing-based claim of a speedup.
    INFO("complex nonzero CG terms="<<real.complex_coupling_count()<<", transformed real terms="<<real.real_coupling_count());
    REQUIRE(real.complex_coupling_count()>0);REQUIRE(real.real_coupling_count()>0);
    const auto old=real.compute(site);for(int i=0;i<old.size();++i) {
        REQUIRE(old[i].real()==Catch::Approx(b[2*i]).margin(1e-13));
        REQUIRE(old[i].imag()==Catch::Approx(b[2*i+1]).margin(1e-13));
    }
    auto clone=real.clone();Eigen::VectorXd copied;clone->compute(site,copied);
    REQUIRE((copied-b).norm()<1e-13);
    // Never infer a zero imaginary channel from real harmonics. Odd-parity
    // components in this asymmetric environment are part of the old contract.
    if(lmax==4) {
        double imaginary_norm=0.;for(int i=1;i<a.size();i+=2) imaginary_norm+=a[i]*a[i];
        REQUIRE(std::sqrt(imaginary_norm)>1e-9);
    }
    const auto rotation=Eigen::AngleAxisd(.63,Eigen::Vector3d(.3,-.4,.7).normalized()).toRotationMatrix();
    auto rotated=site;for(auto& p:rotated.pos_) p=p*rotation.transpose();
    Eigen::VectorXd invariant;real.compute(rotated,invariant);
    REQUIRE((invariant-b).norm()<3e-11*std::max(1.,b.norm()));
    auto permuted=site;std::reverse(permuted.pos_.begin(),permuted.pos_.end());
    std::reverse(permuted.distances_.begin(),permuted.distances_.end());real.compute(permuted,invariant);
    REQUIRE((invariant-b).norm()<3e-11*std::max(1.,b.norm()));
}

TEST_CASE("SO3 real weighted curvature matches complex and independent directional oracle","[descriptor][so3][real][hessian]") {
    Descriptor_BispectrumSO3 reference(4,0.,4.,2),real(4,0.,4.,2,true);auto site=real_so3_site();
    const auto weights=Eigen::VectorXd::LinSpaced(real.feature_size(),-.02,.03).eval();
    DescriptorSiteMatrix direction(site.nbhcount_,3);for(int i=0;i<direction.size();++i) direction.data()[i]=.2*std::sin(i+1.);
    double ea,eb;DescriptorSiteMatrix ga,gb,ha,hb;
    reference.compute_weighted_site_hvp(site,weights,direction,ea,ga,ha);
    real.compute_weighted_site_hvp(site,weights,direction,eb,gb,hb);
    REQUIRE(ea==Catch::Approx(eb).margin(2e-11));
    REQUIRE((ga-gb).norm()<2e-10*std::max(1.,ga.norm()));REQUIRE((ha-hb).norm()<3e-10*std::max(1.,ha.norm()));
    for(double step:{2e-4,1e-4}) {
        auto plus=site,minus=site;move_so3_site(plus,direction,step);move_so3_site(minus,direction,-step);
        Eigen::VectorXd discarded;DescriptorGradient gp,gm;ComputeOptions options;options.compute_derivative=true;
        options.gradient=&gp;real.compute(plus,discarded,options);options.gradient=&gm;real.compute(minus,discarded,options);
        DescriptorSiteMatrix difference=DescriptorSiteMatrix::Zero(site.nbhcount_,3);
        for(int f=0;f<weights.size();++f) for(int j=0;j<site.nbhcount_;++j) for(int a=0;a<3;++a)
            difference(j,a)+=weights[f]*(gp.grad(f,j,a)-gm.grad(f,j,a))/(2*step);
        REQUIRE((difference-hb).norm()<2e-7*std::max(1.,hb.norm()));
    }
    Eigen::MatrixXd dense;real.compute_weighted_site_hessian(site,weights,eb,gb,dense);
    REQUIRE((dense*Eigen::Map<const Eigen::VectorXd>(direction.data(),direction.size())
        -Eigen::Map<const Eigen::VectorXd>(hb.data(),hb.size())).norm()<2e-10);
    REQUIRE((dense-dense.transpose()).norm()<2e-10);
    Neighborhood empty;Eigen::VectorXd zeros;real.compute(empty,zeros);REQUIRE(zeros.isZero());
    DescriptorSiteMatrix empty_direction(0,3);real.compute_weighted_site_hvp(empty,weights,empty_direction,eb,gb,hb);
    REQUIRE(eb==0.);REQUIRE(gb.rows()==0);REQUIRE(hb.rows()==0);
}

TEST_CASE("SO3 real setting forwards through descriptor wrapper and clone","[descriptor][so3][real]") {
    GPSettings settings;settings.Modify("dtype","biso3");settings.Modify("biso3_lmax","3");
    settings.Modify("biso3_rsize","2");settings.Modify("biso3_real_harmonics","true");
    descriptor_wrapper wrapper(settings);const auto* real=dynamic_cast<Descriptor_BispectrumSO3*>(wrapper.desc.get());
    REQUIRE(real);REQUIRE(real->uses_real_harmonics());auto cloned=wrapper.desc->clone();
    auto site=real_so3_site();Eigen::VectorXd a,b;wrapper.compute(site,a);cloned->compute(site,b);REQUIRE((a-b).norm()<1e-13);
}
