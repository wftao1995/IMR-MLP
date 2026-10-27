#include <catch2/catch_all.hpp>
#include "mathextra/angular_functions.h"
#include "mathextra/real_spherical_recurrence.h"
#include "descriptors/local_curvature_jet.h"
#include <array>
#include <cmath>
#include <limits>
#include <numbers>
#include <vector>
using Catch::Approx;
namespace {
int yi(int l,int m){return l*(l+1)+m;}
double reference(int l,int m,double x,double y,double z) {
    const double r=std::hypot(x,y,z),phi=std::atan2(y,x);
    const int a=std::abs(m);
    const double factor=std::sqrt((2.*l+1)/(4*std::numbers::pi)*std::exp(std::lgamma(l-a+1.)-std::lgamma(l+a+1.)));
    // C++ associated Legendre omits the Condon--Shortley phase.
    return factor*std::assoc_legendre(l,a,z/r)*(a%2 ? -1 : 1)*
        (m>0 ? std::sqrt(2.)*std::cos(a*phi) : m<0 ? std::sqrt(2.)*std::sin(a*phi) : 1.);
}
std::vector<std::pair<double,double>> gauss(int n) {
    std::vector<std::pair<double,double>> result;
    for(int i=0;i<n;++i) {
        double x=std::cos(std::numbers::pi*(i+.75)/(n+.5)),p=0,dp=0;
        for(int iteration=0;iteration<30;++iteration) {
            double previous=1,current=x;
            for(int l=2;l<=n;++l){double next=((2*l-1)*x*current-(l-1)*previous)/l;previous=current;current=next;}
            p=current;dp=n*(x*current-previous)/(x*x-1);
            const double step=p/dp;x-=step;if(std::abs(step)<1e-15)break;
        }
        result.emplace_back(x,2/((1-x*x)*dp*dp));
    }
    return result;
}
}
TEST_CASE("Real spherical Cartesian values preserve independent tesseral convention through l32", "[angular][real][contract]") {
    AngularFuncs::RealSphericalHarmoics sh(32);
    for(const auto& point:std::vector<std::array<double,3>>{{.3,-.4,.7},{-.8,.2,-.5},{0,0,1},{0,0,-1},{1e-9,-2e-9,-1}}) {
        sh.computedYCar(point[0],point[1],point[2],std::hypot(point[0],point[1],point[2]));
        for(int l:{0,1,2,4,8,16,32}) {
            double sum=0;
            for(int m=-l;m<=l;++m) {
                const double value=sh.Y[yi(l,m)];sum+=value*value;
                // Independent angular reference loses near-axis transverse
                // digits; exact-axis/ordinary directions retain their values.
                if(std::hypot(point[0],point[1])>1e-6 || (point[0]==0 && point[1]==0))
                    REQUIRE(value==Approx(reference(l,m,point[0],point[1],point[2])).margin(2e-12));
                REQUIRE(std::isfinite(sh.dYdx[yi(l,m)]));
                REQUIRE(std::isfinite(sh.dYdy[yi(l,m)]));
                REQUIRE(std::isfinite(sh.dYdz[yi(l,m)]));
            }
            REQUIRE(sum==Approx((2.*l+1)/(4*std::numbers::pi)).margin(2e-12));
        }
    }
    AngularFuncs::RealSphericalHarmoics truncated(8,2),full(8);
    truncated.computedP(.37);full.computedP(.37);
    for(int l=0;l<=8;++l)for(int m=0;m<=std::min(l,2);++m) {
        const int index=l*(l+1)/2+m;
        REQUIRE(truncated.P[index]==Approx(full.P[index]).margin(1e-14));
        REQUIRE(truncated.dP[index]==Approx(full.dP[index]).margin(1e-14));
    }
    truncated.computedYCar(.2,.3,.4,std::hypot(.2,.3,.4));full.computedYCar(.2,.3,.4,std::hypot(.2,.3,.4));
    for(int l=0;l<=8;++l)for(int m=-std::min(l,2);m<=std::min(l,2);++m)
        REQUIRE(truncated.Y[yi(l,m)]==Approx(full.Y[yi(l,m)]).margin(1e-14));
}
TEST_CASE("Real spherical polar parity scale and two-step gradients agree independently", "[angular][real][contract]") {
    AngularFuncs::RealSphericalHarmoics sh(32),values(32);
    for(double z:{-1.,1.})for(double x:{0.,1e-9,1e-4}) {
        sh.computedYCar(x,0,z,std::hypot(x,z));
        const double c=std::sqrt(15/(4*std::numbers::pi)),s=x*x+z*z;
        REQUIRE(sh.Y[yi(2,1)]==Approx(-c*x*z/s).margin(1e-14));
        REQUIRE(sh.dYdx[yi(2,1)]==Approx(-c*z*(s-2*x*x)/(s*s)).margin(1e-13));
        for(double h:{2e-5,1e-5})for(int coordinate=0;coordinate<3;++coordinate) {
            std::array<double,3> p{x,0,z},plus=p,minus=p;plus[coordinate]+=h;minus[coordinate]-=h;
            values.computeYCarSafe(plus[0],plus[1],plus[2],std::hypot(plus[0],plus[1],plus[2]));
            const std::vector<double> yp(values.Y,values.Y+1089);
            values.computeYCarSafe(minus[0],minus[1],minus[2],std::hypot(minus[0],minus[1],minus[2]));
            const double* gradient=coordinate==0 ? sh.dYdx : coordinate==1 ? sh.dYdy : sh.dYdz;
            for(int l:{1,2,8,16,32})for(int m=-l;m<=l;++m)
                REQUIRE(gradient[yi(l,m)]==Approx((yp[yi(l,m)]-values.Y[yi(l,m)])/(2*h)).margin(3e-5));
        }
    }
    sh.computedYCar(.3,-.4,.5,std::hypot(.3,.4,.5));
    std::vector<double> y(sh.Y,sh.Y+1089),dx(sh.dYdx,sh.dYdx+1089);
    for(double scale:{1e-150,1e150}) {
        values.computedYCar(.3*scale,-.4*scale,.5*scale,std::hypot(.3*scale,.4*scale,.5*scale));
        for(int i=0;i<1089;++i) {
            REQUIRE(values.Y[i]==Approx(y[i]).margin(2e-13));
            REQUIRE(values.dYdx[i]*scale==Approx(dx[i]).margin(2e-11));
        }
    }
    REQUIRE_THROWS_AS(sh.computeYCar(0,0,0,0),std::invalid_argument);
    REQUIRE_THROWS_AS(sh.computeYCarSafe(1,0,0,2),std::invalid_argument);
    REQUIRE_THROWS_AS(sh.computedYCar(1,0,0,std::numeric_limits<double>::infinity()),std::invalid_argument);
    REQUIRE_THROWS_AS(AngularFuncs::RealSphericalHarmoics(-1),std::invalid_argument);
    REQUIRE_THROWS_AS(AngularFuncs::RealSphericalHarmoics(2,3),std::invalid_argument);
}
TEST_CASE("Real spherical independent quadrature resolves selected orthonormal modes through l32", "[angular][real][contract]") {
    AngularFuncs::RealSphericalHarmoics sh(32);
    std::vector<int> selected{0};
    for(int l:{1,2,4,8,16,32})for(int m:{-l,0,l})selected.push_back(yi(l,m));
    std::vector<double> gram(selected.size()*selected.size());
    for(const auto& [z,w]:gauss(34))for(int j=0;j<66;++j) {
        const double phi=2*std::numbers::pi*(j+.5)/66,r=std::sqrt(1-z*z);
        sh.computeYCar(r*std::cos(phi),r*std::sin(phi),z,1.);
        for(size_t a=0;a<selected.size();++a)for(size_t b=0;b<selected.size();++b)
            gram[a*selected.size()+b]+=w*2*std::numbers::pi/66*sh.Y[selected[a]]*sh.Y[selected[b]];
    }
    for(size_t a=0;a<selected.size();++a)for(size_t b=0;b<selected.size();++b)
        REQUIRE(gram[a*selected.size()+b]==Approx(a==b ? 1. : 0.).margin(4e-13));
}
TEST_CASE("Real solid recurrence supports local second-order scalars without changing phase", "[angular][real][contract]") {
    using DescriptorCurvature::Jet6;
    const auto result=AngularFuncs::detail::real_solid_harmonics(2,2,Jet6::variable(.2,0),Jet6::variable(-.3,1),Jet6::variable(-.7,2));
    const double c=std::sqrt(15/(4*std::numbers::pi));
    REQUIRE(result[yi(2,1)].value==Approx(c*.14).margin(1e-15));
    REQUIRE(result[yi(2,1)].gradient[0]==Approx(c*.7).margin(1e-15));
    REQUIRE(result[yi(2,1)].hessian(0,2)==Approx(-c).margin(1e-15));
}
