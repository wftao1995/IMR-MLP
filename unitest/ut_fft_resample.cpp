#include <catch2/catch_all.hpp>
#include "analyse/fft_resample.h"

namespace {
std::vector<std::complex<double>> direct_dft(const std::vector<double>& source) {
    std::vector<std::complex<double>> result(source.size());const double pi=std::acos(-1.);
    for(int k=0;k<int(source.size());++k) for(int j=0;j<int(source.size());++j)
        result[k]+=source[j]*std::exp(std::complex<double>(0.,-2*pi*k*j/source.size()));
    return result;
}
std::vector<double> mirrored(std::vector<double> samples,bool even) {
    const int n=samples.size();samples.resize(2*n-(even?0:1));
    for(int j=even?0:1;j<n;++j) samples[2*n-j-1]=samples[j];return samples;
}
}
TEST_CASE("FFT padding preserves DC and both source Nyquist signs","[fft][resample]") {
    const int length=GENERATE(7,8),factor=GENERATE(2,3);
    std::vector<double> source(length);
    for(int j=0;j<length;++j) source[j]=2+.3*std::cos(2*std::acos(-1.)*(length/2)*j/length);
    const auto padded=FFTResample::pad_spectrum(direct_dft(source),factor);
    for(int m=0;m<length*factor;++m) {
        std::complex<double> value=0.;
        for(int k=0;k<length*factor;++k) value+=padded[k]*std::exp(std::complex<double>(0.,2*std::acos(-1.)*k*m/(length*factor)));
        value/=length*factor;
        REQUIRE(value.real()==Catch::Approx(2+.3*std::cos(2*std::acos(-1.)*(length/2)*m/(length*factor))).margin(2e-13));
        REQUIRE(std::abs(value.imag())<2e-13);
    }
}
TEST_CASE("FFT interpolation and derivatives retain the original mirrored period","[fft][resample]") {
    const bool even=GENERATE(false,true);const int factor=GENERATE(2,3);const int derivative=GENERATE(0,1,2);
    const double dt=.7;const std::vector<double> source{.2,1.3,-.7,.9,.4};
    const auto extended=mirrored(source,even);const auto spectrum=direct_dft(extended);
    const auto actual=FFTResample::interpolate(source,factor,even,dt,derivative);
    const int length=extended.size();REQUIRE(actual.size()==size_t(factor*(source.size()-1)+1));
    for(int m=0;m<int(actual.size());++m) {
        std::complex<double> expected=0.;
        for(int k=0;k<length;++k) {
            const int frequency=k<=length/2?k:k-length;const double omega=2*std::acos(-1.)*frequency/(length*dt);
            if(length%2==0 && k==length/2) {
                const double phase=std::acos(-1.)*m/factor;
                expected+=spectrum[k].real()*(derivative==0?std::cos(phase):derivative==1?-omega*std::sin(phase):-omega*omega*std::cos(phase));
            } else {
                auto term=spectrum[k]*std::exp(std::complex<double>(0.,omega*m*dt/factor));
                if(derivative==1) term*=std::complex<double>(0.,omega);else if(derivative==2) term*=-omega*omega;
                expected+=term;
            }
        }
        REQUIRE(actual[m]==Catch::Approx(expected.real()/length).margin(3e-12));
        if(derivative==0 && m%factor==0) REQUIRE(actual[m]==Catch::Approx(source[m/factor]).margin(2e-13));
    }
}
TEST_CASE("FFT interpolation rejects invalid dimensions and nonfinite input","[fft][resample]") {
    REQUIRE_THROWS(FFTResample::interpolate({1.},2,true,1.));
    REQUIRE_THROWS(FFTResample::interpolate({1.,2.},0,true,1.));
    REQUIRE_THROWS(FFTResample::interpolate({1.,2.},2,true,0.));
    REQUIRE_THROWS(FFTResample::interpolate({1.,NAN},2,true,1.));
    REQUIRE_THROWS(FFTResample::output_size(2,std::numeric_limits<int>::max(),true));
}
