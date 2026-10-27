#include <catch2/catch_all.hpp>
#include "descriptors/so4_wigner_spline.h"
#include <iostream>

TEST_CASE("SO4 reduced Wigner spline passes interior value and derivative gates",
          "[descriptor][so4][spline]") {
    const int maximum=GENERATE(0,4,8,12);
    auto table=SO4Density::make_wigner_spline(maximum);
    REQUIRE(table->cells()>=64);REQUIRE(table->cells()<=8192);
    for(double e:table->error())REQUIRE(e<=1e-10);
    std::cout<<"SO4 reduced J="<<maximum<<" cells="<<table->cells()<<" error="
             <<table->error()[0]<<","<<table->error()[1]<<","<<table->error()[2]<<"\n";
    using namespace DescriptorCurvature;
    const auto offsets=SO4Density::offsets(maximum);
    const double spacing=M_PI/table->cells();
    for(double beta:{0.,1e-10,1.9*spacing,2.1*spacing,2.8*spacing,3.9*spacing,
                     .37,.7,1.7,2.9,M_PI-1.9*spacing,M_PI-2.1*spacing,
                     M_PI-2.8*spacing,M_PI-3.9*spacing,M_PI-4.1*spacing,M_PI}) {
        // Independent Cartesian chart on the unit sphere: beta and both
        // phases vary, so differentiation tests phase and blend cross terms.
        auto x=Jet6::variable(beta,0),p=Jet6::variable(.47,1),q=Jet6::variable(-.63,2);
        x.gradient[3]=.17;p.gradient[4]=-.23;q.gradient[5]=.31;
        const auto c=SO4Density::cosine(Jet6(.5)*x),s=SO4Density::sine(Jet6(.5)*x);
        const ComplexJet a(c*SO4Density::cosine(p),c*SO4Density::sine(p));
        const ComplexJet b(s*SO4Density::cosine(q),s*SO4Density::sine(q));
        const auto exact=SO4Density::wigner(a,b,maximum,offsets),approx=table->evaluate(a,b);
        const auto first=table->evaluate_jacobian(a,b);
        const auto values=table->evaluate_values({a.real.value,a.imag.value},{b.real.value,b.imag.value});
        for(int k=0;k<offsets.back();++k) {
            CAPTURE(maximum,beta,k);
            REQUIRE(std::abs(exact[k].real.value-approx[k].real.value)<1e-10);
            REQUIRE(std::abs(exact[k].imag.value-approx[k].imag.value)<1e-10);
            REQUIRE((exact[k].real.gradient-approx[k].real.gradient).norm()<1e-8);
            REQUIRE((exact[k].imag.gradient-approx[k].imag.gradient).norm()<1e-8);
            REQUIRE((exact[k].real.hessian-approx[k].real.hessian).norm()<1e-6);
            REQUIRE((exact[k].imag.hessian-approx[k].imag.hessian).norm()<1e-6);
            REQUIRE(std::abs(values[k]-std::complex<double>(exact[k].real.value,exact[k].imag.value))<1e-10);
            REQUIRE(std::abs(first[k].real.value-approx[k].real.value)<1e-12);
            REQUIRE(std::abs(first[k].imag.value-approx[k].imag.value)<1e-12);
            REQUIRE((first[k].real.gradient-approx[k].real.gradient).norm()<1e-9);
            REQUIRE((first[k].imag.gradient-approx[k].imag.gradient).norm()<1e-9);
        }
    }
}

TEST_CASE("SO4 spline Cartesian phase derivatives match independent spin one polynomials",
          "[descriptor][so4][spline][independent]") {
    using namespace DescriptorCurvature;
    const auto table=SO4Density::make_wigner_spline(2);
    const double spacing=M_PI/table->cells();
    const auto polynomial=[](const Eigen::Vector3d &q) {
        using C=std::complex<double>;
        const C a=std::cos(q[0]/2.)*std::polar(1.,q[1]);
        const C b=std::sin(q[0]/2.)*std::polar(1.,q[2]);
        const C ac=std::conj(a),bc=std::conj(b);const double r=std::sqrt(2.);
        return std::vector<C>{1.,a,bc,-b,ac,
            a*a,r*a*bc,bc*bc,-r*a*b,a*ac-b*bc,r*ac*bc,b*b,-r*ac*b,ac*ac};
    };
    for(double beta:{0.,2.*spacing*(1.-1e-7),2.*spacing*(1.+1e-7),
        4.*spacing*(1.-1e-7),4.*spacing*(1.+1e-7),.71,1.83,
        M_PI-2.*spacing*(1.-1e-7),M_PI-2.*spacing*(1.+1e-7),
        M_PI-4.*spacing*(1.-1e-7),M_PI-4.*spacing*(1.+1e-7),M_PI}) {
        const Eigen::Vector3d q(beta,.47,-.63);
        const auto x=Jet6::variable(q[0],0),p=Jet6::variable(q[1],1),z=Jet6::variable(q[2],2);
        const auto c=SO4Density::cosine(Jet6(.5)*x),s=SO4Density::sine(Jet6(.5)*x);
        const auto got=table->evaluate({c*SO4Density::cosine(p),c*SO4Density::sine(p)},
                                        {s*SO4Density::cosine(z),s*SO4Density::sine(z)});
        const auto ref=polynomial(q);
        for(int k=0;k<static_cast<int>(got.size());++k) {
            CAPTURE(beta,k);
            REQUIRE(std::abs(std::complex<double>(got[k].real.value,got[k].imag.value)-ref[k])<1e-10);
            for(double h:{1e-4,5e-5})for(int a=0;a<3;++a) {
                Eigen::Vector3d p=q,m=q;p[a]+=h;m[a]-=h;
                const auto plus=polynomial(p),minus=polynomial(m);
                const auto first=(plus[k]-minus[k])/(2*h);
                const auto second=(plus[k]-2.*ref[k]+minus[k])/(h*h);
                REQUIRE(std::abs(std::complex<double>(got[k].real.gradient[a],got[k].imag.gradient[a])-first)<2e-8);
                REQUIRE(std::abs(std::complex<double>(got[k].real.hessian(a,a),got[k].imag.hessian(a,a))-second)<8e-7);
            }
        }
    }
}

TEST_CASE("SO4 spline has explicit angular availability and shares immutable plans",
          "[descriptor][so4][spline]") {
    REQUIRE_THROWS(SO4Density::make_wigner_spline(13));
    auto a=SO4Density::make_wigner_spline(4),b=SO4Density::make_wigner_spline(4);
    REQUIRE(a.get()==b.get());
}
