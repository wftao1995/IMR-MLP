#pragma once

#include "libalglib/fasttransforms.h"
#include <cmath>
#include <complex>
#include <limits>
#include <stdexcept>
#include <vector>

namespace FFTResample {
inline int checked_length(int samples,int factor,bool even) {
    if(samples<2 || factor<1) throw std::invalid_argument("FFT interpolation requires at least two samples and a positive integer factor");
    const long long length=2LL*samples-(even ? 0 : 1);
    if(length*factor>std::numeric_limits<int>::max()) throw std::length_error("FFT interpolation length exceeds integer range");
    return static_cast<int>(length);
}
inline int output_size(int samples,int factor,bool even) {
    checked_length(samples,factor,even);
    return static_cast<int>(1LL*factor*(samples-1)+1);
}

// Full Hermitian DFT convention: forward is unnormalized, inverse divides by
// its length. Preserve the physical period and scale amplitudes by new/old.
inline std::vector<std::complex<double>> pad_spectrum(
    const std::vector<std::complex<double>>& source,int factor) {
    const int length=static_cast<int>(source.size());
    if(length<2 || factor<1 || source.size()>size_t(std::numeric_limits<int>::max()/factor))
        throw std::invalid_argument("Invalid FFT spectrum interpolation dimensions");
    if(factor==1) return source;
    std::vector<std::complex<double>> result(size_t(length)*factor);
    for(int k=0;k<length;++k) {
        if(!std::isfinite(source[k].real()) || !std::isfinite(source[k].imag()))
            throw std::invalid_argument("FFT interpolation spectrum must be finite");
        if(length%2==0 && k==length/2) {
            // This bin represented both signs on the source grid. Its cosine
            // interpolant has two distinct bins on the larger grid.
            const auto half=.5*factor*source[k].real();
            result[k]=half;result[result.size()-k]=half;
        } else result[k<=length/2 ? k : int(result.size())+k-length]=double(factor)*source[k];
    }
    return result;
}

// Uses exactly ComputeSmoothFFT::fill_data's mirrored sequence. No low-pass
// window is applied: inputs have already been smoothed on their original grid.
inline std::vector<double> interpolate(const std::vector<double>& samples,int factor,
    bool even,double dt,int derivative=0) {
    if(samples.size()>size_t(std::numeric_limits<int>::max())) throw std::length_error("Too many FFT samples");
    const int n=static_cast<int>(samples.size()),length=checked_length(n,factor,even);
    if(!(dt>0.) || !std::isfinite(dt) || !std::isfinite(length*dt) || derivative<0 || derivative>2)
        throw std::invalid_argument("Invalid FFT interpolation timestep or derivative order");
    alglib::real_1d_array data;data.setlength(length);
    for(int j=0;j<n;++j) {
        if(!std::isfinite(samples[j])) throw std::invalid_argument("FFT interpolation samples must be finite");
        data[j]=samples[j];
    }
    for(int j=even ? 0 : 1;j<n;++j) data[2*n-j-1]=data[j];
    alglib::complex_1d_array transform;alglib::fftr1d(data,transform);
    std::vector<std::complex<double>> spectrum(length);
    for(int k=0;k<length;++k) spectrum[k]={transform[k].x,transform[k].y};
    auto padded=pad_spectrum(spectrum,factor);
    const int target=length*factor;
    transform.setlength(target);
    const double delta=2*std::acos(-1.)/(length*dt);
    for(int k=0;k<target;++k) {
        const int frequency=k<=target/2 ? k : k-target;
        const double omega=frequency*delta;
        auto value=padded[k];
        if(derivative==1) {
            if(target%2==0 && k==target/2) value=0.;
            else value*=std::complex<double>(0.,omega);
        } else if(derivative==2) value*=-omega*omega;
        transform[k]=alglib::complex(value.real(),value.imag());
    }
    alglib::fftr1dinv(transform,data);
    std::vector<double> result(output_size(n,factor,even));
    for(int m=0;m<int(result.size());++m) {
        if(!std::isfinite(data[m])) throw std::invalid_argument("FFT interpolation produced non-finite samples");
        result[m]=data[m];
    }
    return result;
}
}
