#include <catch2/catch_all.hpp>
#include "descriptors/so4_density_kernel.h"
#include "descriptors/CutoffFuncs.h"
#include <array>

namespace {
using namespace SO4Density;
using DescriptorCurvature::Jet6;
using DescriptorCurvature::ComplexJet;
using Complex=std::complex<double>;

void first_order(const Eigen::RowVector3d &p,int maximum,int convention,bool atan_map,
                 bool normalize,const CutoffFunctions &cutoff,std::span<ComplexFirstOrder> result) {
    const auto x=FirstOrder::variable(p[0],0),y=FirstOrder::variable(p[1],1),z=FirstOrder::variable(p[2],2);
    const auto r=SO4Density::sqrt(x*x+y*y+z*z);
    FirstOrder ar,ai,br,bi;
    parameters(x,y,z,r,3.1,convention,atan_map,ar,ai,br,bi);
    const auto offset=offsets(maximum);
    wigner_first_order(ComplexFirstOrder(ar,ai),ComplexFirstOrder(br,bi),maximum,offset,result);
    const auto fc=SO4Density::compose(r,cutoff.fcut(r.value),cutoff.dfcut(r.value));
    for(int j=0;j<=maximum;++j)for(int k=offset[j];k<offset[j+1];++k)
        result[k]=FirstOrder(.7*(normalize?1./std::sqrt(j+1.):1.))*fc*result[k];
}
std::vector<ComplexJet> full_order(const Eigen::RowVector3d &p,int maximum,int convention,bool atan_map,
                                 bool normalize,const CutoffFunctions &cutoff) {
    const auto x=Jet6::variable(p[0],0),y=Jet6::variable(p[1],1),z=Jet6::variable(p[2],2);
    const auto r=DescriptorCurvature::sqrt(x*x+y*y+z*z);
    Jet6 ar,ai,br,bi;
    parameters(x,y,z,r,3.1,convention,atan_map,ar,ai,br,bi);
    const auto offset=offsets(maximum);
    auto result=wigner(ComplexJet(ar,ai),ComplexJet(br,bi),maximum,offset);
    const auto fc=DescriptorCurvature::compose(r,cutoff.fcut(r.value),cutoff.dfcut(r.value),0.);
    for(int j=0;j<=maximum;++j)for(int k=offset[j];k<offset[j+1];++k)
        result[k]=Jet6(.7*(normalize?1./std::sqrt(j+1.):1.))*fc*result[k];
    return result;
}
std::vector<Complex> values(const Eigen::RowVector3d &p,int maximum,int convention,bool atan_map,
                            bool normalize,const CutoffFunctions &cutoff) {
    double ar,ai,br,bi;
    parameters(p[0],p[1],p[2],p.norm(),3.1,convention,atan_map,ar,ai,br,bi);
    const auto offset=offsets(maximum);
    auto result=wigner_values(Complex(ar,ai),Complex(br,bi),maximum,offset);
    for(int j=0;j<=maximum;++j)for(int k=offset[j];k<offset[j+1];++k)
        result[k]*=.7*(normalize?1./std::sqrt(j+1.):1.)*cutoff.fcut(p.norm());
    return result;
}
}

TEST_CASE("SO4 Cartesian first-order coefficients match full jets across runtime conventions", "[so4][first-order]") {
    const int convention=GENERATE(1,2,3,4),maximum=GENERATE(4,8,12);
    const bool atan_map=GENERATE(false,true),normalize=GENERATE(false,true);
    const auto cutoff_type=GENERATE(CutoffFunctions::CT_HARD,CutoffFunctions::CT_COS_BASIC,CutoffFunctions::CT_POLY2);
    CutoffFunctions cutoff;cutoff.setCutoffType(cutoff_type);cutoff.setCutoffRadius(4.);
    const std::array<Eigen::RowVector3d,6> positions{{{.43,-.71,.92},{0.,0.,1.2},{0.,-1.4,0.},{.2,.1,3.5},{4.,0.,0.},{4.1,0.,0.}}};
    const auto offset=offsets(maximum);
    std::vector<ComplexFirstOrder> result(offset.back());
    const auto *storage=result.data();
    for(const auto &p:positions) {
        first_order(p,maximum,convention,atan_map,normalize,cutoff,result);
        REQUIRE(result.data()==storage);
        const auto reference=full_order(p,maximum,convention,atan_map,normalize,cutoff);
        double value_error=0.,jac_error=0.;
        for(std::size_t k=0;k<result.size();++k) {
            value_error=std::max(value_error,std::abs(Complex(result[k].real.value,result[k].imag.value)
                -Complex(reference[k].real.value,reference[k].imag.value)));
            for(int a=0;a<3;++a)jac_error=std::max(jac_error,std::abs(
                Complex(result[k].real.gradient[a],result[k].imag.gradient[a])
                -Complex(reference[k].real.gradient[a],reference[k].imag.gradient[a])));
        }
        REQUIRE(value_error<1e-13);
        REQUIRE(jac_error<1e-12);
    }
}

TEST_CASE("SO4 Cartesian first-order Jacobian matches independent scalar finite differences", "[so4][first-order][fd]") {
    const int convention=GENERATE(1,2,3,4);
    const bool atan_map=GENERATE(false,true),normalize=GENERATE(false,true);
    CutoffFunctions cutoff;cutoff.setCutoffType(CutoffFunctions::CT_COS_BASIC);cutoff.setCutoffRadius(4.);
    const Eigen::RowVector3d p(.43,-.71,.92);
    const auto offset=offsets(8);
    std::vector<ComplexFirstOrder> result(offset.back());
    first_order(p,8,convention,atan_map,normalize,cutoff,result);
    for(double step:{2e-5,7e-6})for(int a=0;a<3;++a) {
        auto plus=p,minus=p;plus[a]+=step;minus[a]-=step;
        const auto vp=values(plus,8,convention,atan_map,normalize,cutoff);
        const auto vm=values(minus,8,convention,atan_map,normalize,cutoff);
        double error=0.;
        for(std::size_t k=0;k<result.size();++k)
            error=std::max(error,std::abs(Complex(result[k].real.gradient[a],result[k].imag.gradient[a])
                -(vp[k]-vm[k])/(2*step)));
        REQUIRE(error<2e-8);
    }
}

TEST_CASE("SO4 first-order map retains radial sign branch errors and compact scratch", "[so4][first-order][branch]") {
    const int convention=GENERATE(1,2);
    const auto x=FirstOrder::variable(3.1,0),y=FirstOrder(0.),z=FirstOrder(0.);
    const auto r=SO4Density::sqrt(x*x+y*y+z*z);
    FirstOrder ar,ai,br,bi;
    REQUIRE_THROWS_AS(parameters(x,y,z,r,3.1,convention,false,ar,ai,br,bi),std::invalid_argument);
    REQUIRE_NOTHROW(parameters(x,y,z,r,3.1,convention,true,ar,ai,br,bi));
    REQUIRE(sizeof(ComplexFirstOrder)<sizeof(ComplexJet)/5);
    std::vector<ComplexFirstOrder> wrong(1);
    REQUIRE_THROWS_AS(wigner_first_order(ComplexFirstOrder(1.),ComplexFirstOrder(0.),4,offsets(4),wrong),std::invalid_argument);
}
