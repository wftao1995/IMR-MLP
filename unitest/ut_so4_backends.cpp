#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_BispectrumSO4.h"
#include "unitest/pair_cell_fixture.h"
#include <iostream>

TEST_CASE("SO4 exact sparse and spline backends preserve full geometric derivatives",
          "[descriptor][so4][backend]") {
    const auto angular=GENERATE(3,4);
    const auto convention=GENERATE("q1","q2","q3","q4");
    GPSettings s;s.Modify("cutoff","3.2");s.Modify("twojmax",std::to_string(angular));
    s.Modify("quaternion_convention",convention);s.Modify("radial_angle_map","atan");
    s.Modify("normalize_U","true");s.Modify("weights","0.7 1.3");s.Modify("selfweights","0.4 0.8");
    Descriptor_BispectrumSO4 reference(s);
    s.Modify("contraction_backend","snap");Descriptor_BispectrumSO4 exact(s);
    s.Modify("wigner_backend","spline");Descriptor_BispectrumSO4 spline(s);
    auto a=PairCellTest::geometry(true);Neighborhoods sites(a,reference.cutoff_);
    const Eigen::VectorXd weights=Eigen::VectorXd::LinSpaced(reference.size(),-.03,.05);
    for(int i=0;i<sites.size();++i) {
        Eigen::VectorXd v;DescriptorGradient jac;ComputeOptions options;
        options.compute_derivative=true;options.gradient=&jac;
        reference.compute(sites[i],v,options);
        double energy;DescriptorSiteMatrix g,product;
        const DescriptorSiteMatrix direction=DescriptorSiteMatrix::Constant(sites[i].nbhcount_,3,.13);
        reference.compute_weighted_site_hvp(sites[i],weights,direction,energy,g,product);
        for(auto *backend:{&exact,&spline}) {
            const bool approximate=backend==&spline;
            Eigen::VectorXd w;DescriptorGradient gj;ComputeOptions o=options;o.gradient=&gj;
            backend->compute(sites[i],w,o);
            REQUIRE((v-w).norm()/(1.+v.norm())<(approximate?1e-8:1e-10));
            REQUIRE(Eigen::Map<Eigen::VectorXd>(jac.grad.data(),jac.grad.size()).isApprox(
                Eigen::Map<Eigen::VectorXd>(gj.grad.data(),gj.grad.size()),approximate?1e-6:1e-8));
            double e;DescriptorSiteMatrix bg,bp;
            backend->compute_weighted_site_hvp(sites[i],weights,direction,e,bg,bp);
            REQUIRE((g-bg).norm()/(1.+g.norm())<(approximate?1e-6:1e-8));
            REQUIRE((product-bp).norm()/(1.+product.norm())<(approximate?1e-4:1e-6));
        }
    }
    const Eigen::VectorXd direction=Eigen::VectorXd::LinSpaced(9+3*a.size(),-.05,.07);
    double e;Eigen::VectorXd g,hv;reference.compute_weighted_global_hvp(a,weights,direction,true,e,g,hv);
    Eigen::VectorXd global;reference.compute_global_density(sites,global);
    for(auto *backend:{&exact,&spline}) {
        Eigen::VectorXd bg,bhv,bvalue;double be;
        backend->compute_weighted_global_hvp(a,weights,direction,true,be,bg,bhv);
        backend->compute_global_density(sites,bvalue);
        const bool approximate=backend==&spline;
        REQUIRE((bvalue-global).norm()/(1.+global.norm())<(approximate?1e-8:1e-10));
        REQUIRE((g-bg).norm()/(1.+g.norm())<(approximate?1e-6:1e-8));
        REQUIRE((hv-bhv).norm()/(1.+hv.norm())<(approximate?1e-4:1e-6));
        const Eigen::Matrix3d rotation=Eigen::AngleAxisd(.81,Eigen::Vector3d(.3,.7,-.2).normalized()).toRotationMatrix();
        auto r=a;r.cell_=a.cell_*rotation;for(auto &p:r.pos_)p=p*rotation;
        Neighborhoods rotated(r,backend->cutoff_);Eigen::VectorXd rv;backend->compute_global_density(rotated,rv);
        REQUIRE((rv-bvalue).norm()/(1.+bvalue.norm())<(approximate?1e-8:1e-10));
    }
}

TEST_CASE("SO4 sparse plan removes zero couplings and repeated derivative contractions",
          "[descriptor][so4][performance]") {
    GPSettings s;s.Modify("twojmax","6");s.Modify("contraction_backend","snap");
    Descriptor_BispectrumSO4 d(s);
    REQUIRE(d.sparse_contraction_count()<d.reference_contraction_count());
    std::cout<<"SO4 J=6 reference scalar terms="<<d.reference_contraction_count()
             <<" nonzero stored terms="<<d.sparse_contraction_count()
             <<"; reference Jacobian repeats the coupling loops for each neighbor/axis,"
             <<" SNAP uses one coefficient adjoint per feature\n";
}

TEST_CASE("SO4 spline maximum order passes Cartesian endpoint and rotation gates",
          "[descriptor][so4][spline][maximum-order]") {
    GPSettings s;s.Modify("cutoff","3.2");s.Modify("twojmax","12");
    s.Modify("contraction_backend","snap");Descriptor_BispectrumSO4 exact(s);
    s.Modify("wigner_backend","spline");Descriptor_BispectrumSO4 spline(s);
    const Eigen::VectorXd weights=Eigen::VectorXd::LinSpaced(exact.size(),-.0003,.0005);
    Neighborhood site;site.centertype=0;site.nbhcount_=2;site.indices_={1,2};site.chemmap_={0,1};
    const double radius=3.2*4./3.,r=1.25,spacing=M_PI/1024.;
    for(double beta:{0.,2.1*spacing,3.9*spacing,.73}) {
        const double alpha=std::asin(std::sin(beta/2.)/std::sin(M_PI*r/radius));
        site.pos_={{r*std::cos(alpha),r*std::sin(alpha),0.},{-.4,.7,1.1}};
        site.distances_={r,site.pos_[1].norm()};
        const DescriptorSiteMatrix direction=DescriptorSiteMatrix::Constant(2,3,.013);
        double e,be;DescriptorSiteMatrix g,hv,bg,bhv;
        exact.compute_weighted_site_hvp(site,weights,direction,e,g,hv);
        spline.compute_weighted_site_hvp(site,weights,direction,be,bg,bhv);
        REQUIRE(std::abs(e-be)/(1.+std::abs(e))<1e-8);
        REQUIRE((g-bg).norm()/(1.+g.norm())<1e-6);
        REQUIRE((hv-bhv).norm()/(1.+hv.norm())<1e-4);
        const Eigen::Matrix3d rotation=Eigen::AngleAxisd(.83,Eigen::Vector3d(.3,.7,-.2).normalized()).toRotationMatrix();
        auto rotated=site;for(auto &p:rotated.pos_)p=p*rotation;
        Eigen::VectorXd v,rv;spline.compute(site,v);spline.compute(rotated,rv);
        REQUIRE((rv-v).norm()/(1.+v.norm())<1e-8);
        for(double h:{1e-4,5e-5}) {
            auto p=site,m=site;
            for(int i=0;i<2;++i) {p.pos_[i]+=h*direction.row(i);m.pos_[i]-=h*direction.row(i);
                p.distances_[i]=p.pos_[i].norm();m.distances_[i]=m.pos_[i].norm();}
            double unused;DescriptorSiteMatrix gp,gm,discard;
            exact.compute_weighted_site_hvp(p,weights,direction,unused,gp,discard);
            exact.compute_weighted_site_hvp(m,weights,direction,unused,gm,discard);
            REQUIRE((bhv-(gp-gm)/(2*h)).norm()/(1.+bhv.norm())<1e-4);
        }
    }
}
