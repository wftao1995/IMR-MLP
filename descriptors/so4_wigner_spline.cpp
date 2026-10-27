#include "so4_wigner_spline.h"
#include <algorithm>
#include <map>
#include <mutex>
#include <stdexcept>

namespace SO4Density {
namespace {
using Triple=std::array<long double,3>;
Triple mul(const Triple &a,const Triple &b) {
    return {a[0]*b[0],a[1]*b[0]+a[0]*b[1],a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]};
}
std::vector<Triple> reduced(int maximum,const std::vector<int> &offset,long double beta) {
    const long double c=std::cos(.5L*beta),s=std::sin(.5L*beta);
    const Triple a{c,-.5*s,-.25*c},b{s,.5*c,-.25*s};
    std::vector<Triple> u(offset.back());u[0]={1.,0.,0.};
    const auto ix=[&](int j,int ma,int mb){return index(offset,j,ma,mb);};
    for(int j=1;j<=maximum;++j) {
        for(int mb=0;mb<=j/2;++mb) {
            const auto left=mul(a,u[ix(j-1,0,mb)]),right=mul(b,u[ix(j-1,j-1,mb)]);
            const long double r=std::sqrt(static_cast<long double>(j)/(j-mb));
            for(int k=0;k<3;++k) {u[ix(j,0,mb)][k]=r*left[k];u[ix(j,j,mb)][k]=-r*right[k];}
            for(int ma=1;ma<j;++ma) {
                const auto p=mul(a,u[ix(j-1,ma,mb)]),q=mul(b,u[ix(j-1,ma-1,mb)]);
                const long double x=std::sqrt(static_cast<long double>(j-ma)/(j-mb)),y=std::sqrt(static_cast<long double>(ma)/(j-mb));
                for(int k=0;k<3;++k)u[ix(j,ma,mb)][k]=x*p[k]-y*q[k];
            }
        }
        for(int mb=j/2+1;mb<=j;++mb)for(int ma=0;ma<=j;++ma)
            for(int k=0;k<3;++k)u[ix(j,ma,mb)][k]=((ma+mb)%2?-1.:1.)*u[ix(j,j-ma,j-mb)][k];
    }
    return u;
}
template<class T> auto phase_powers(const T &phase,int maximum) {
    std::array<T,25> powers;
    const auto conjugate_phase=[](const auto &v) {
        if constexpr(std::is_same_v<T,std::complex<double>>)return std::conj(v);
        else return DescriptorCurvature::conjugate(v);
    };
    if constexpr(std::is_same_v<T,std::complex<double>>)powers[maximum]=1.;
    else powers[maximum]=T(DescriptorCurvature::Jet6(1.));
    for(int i=1;i<=maximum;++i) {
        powers[maximum+i]=powers[maximum+i-1]*phase;
        powers[maximum-i]=conjugate_phase(powers[maximum+i]);
    }
    return powers;
}
}

std::array<double,3> WignerSpline::interpolate(const std::array<double,6> &p,double t) const {
    return {((((p[5]*t+p[4])*t+p[3])*t+p[2])*t+p[1])*t+p[0],
            ((((5*p[5]*t+4*p[4])*t+3*p[3])*t+2*p[2])*t+p[1])*inverse_spacing_,
            (((20*p[5]*t+12*p[4])*t+6*p[3])*t+2*p[2])*inverse_spacing_*inverse_spacing_};
}

WignerSpline::WignerSpline(int maximum):maximum_(maximum),offsets_(maximum>=0 && maximum<=12 ? offsets(maximum) : std::vector<int>{}) {
    if(maximum<0 || maximum>12)throw std::invalid_argument("SO4 spline supports 0<=twojmax<=12; FC3 is unavailable");
    for(cells_=64;cells_<=8192;cells_*=2) {
        spacing_=M_PI/cells_;inverse_spacing_=1./spacing_;
        // Compute cancellation-sensitive Hermite coefficients once in extended
        // precision. Runtime queries only load a double polynomial and use Horner.
        polynomials_.resize(std::size_t(cells_)*offsets_.back());
        auto left=reduced(maximum_,offsets_,0.);
        for(int i=0;i<cells_;++i) {
          auto right=reduced(maximum_,offsets_,static_cast<long double>(i+1)*spacing_);
          for(int k=0;k<offsets_.back();++k) {
            const auto &a=left[k];
            const auto &b=right[k];
            const long double h=spacing_,c0=a[0],c1=h*a[1],c2=.5L*h*h*a[2];
            const long double A=b[0]-c0-c1-c2,B=h*b[1]-c1-2*c2,C=h*h*b[2]-2*c2;
            polynomials_[std::size_t(i)*offsets_.back()+k]={double(c0),double(c1),double(c2),
                double(10*A-4*B+.5L*C),double(-15*A+7*B-C),double(6*A-3*B+.5L*C)};
          }
          left=std::move(right);
        }
        error_={0.,0.,0.};
        for(int i=0;i<cells_;++i)for(double t:{.1127016653792583,.276393202250021,.5,.723606797749979,.887298334620742}) {
            const double beta=(i+t)*spacing_;const auto exact=reduced(maximum_,offsets_,beta);
            const double actual_t=(beta-i*spacing_)/spacing_;
            for(int j=0;j<=maximum_;++j)for(int c=offsets_[j];c<offsets_[j+1];++c) {
                const auto approximate=interpolate(polynomials_[std::size_t(i)*offsets_.back()+c],actual_t);
                for(int k=0;k<3;++k)error_[k]=std::max(error_[k],
                    double(std::abs(exact[c][k]-approximate[k])/std::max(1.,std::pow(.5*j,k))));
            }
        }
        if(*std::max_element(error_.begin(),error_.end())<=1e-10)return;
    }
    throw std::runtime_error("SO4 spline could not meet value/first/second error tolerance within 8192 cells");
}

std::vector<DescriptorCurvature::ComplexJet> WignerSpline::evaluate(
    const DescriptorCurvature::ComplexJet &a,const DescriptorCurvature::ComplexJet &b) const {
    using namespace DescriptorCurvature;
    const double av=std::hypot(a.real.value,a.imag.value),bv=std::hypot(b.real.value,b.imag.value);
    const double beta_value=2.*std::atan2(bv,av),distance=std::min(beta_value,M_PI-beta_value);
    // Branch before dividing by either phase amplitude; both endpoint limits
    // (including their Cartesian derivatives) are the native recurrence.
    if(distance<=2.*spacing_)return wigner(a,b,maximum_,offsets_);
    const auto am=sqrt(a.real*a.real+a.imag*a.imag),bm=sqrt(b.real*b.real+b.imag*b.imag);
    const auto beta=Jet6(2.)*arctangent(bm/am);
    const ComplexJet ap(a.real/am,a.imag/am),bp(b.real/bm,b.imag/bm);
    const int cell=std::min(cells_-1,static_cast<int>(beta_value/spacing_));
    const double t=(beta_value-cell*spacing_)/spacing_;
    const auto *polynomials=polynomials_.data()+std::size_t(cell)*offsets_.back();
    const auto a_powers=phase_powers(ap,maximum_),b_powers=phase_powers(bp,maximum_);
    std::vector<ComplexJet> result(offsets_.back());
    for(int j=0;j<=maximum_;++j) {
      for(int ma=0;ma<=j;++ma)for(int mb=0;mb<=j/2;++mb) {
        const int k=index(offsets_,j,ma,mb);const auto p=interpolate(polynomials[k],t);
        // This descriptor recurrence consumes conjugated Cayley--Klein
        // parameters relative to public WignerUTrans, hence these signs.
        result[k]=compose(beta,p[0],p[1],p[2])*(a_powers[maximum_+j-ma-mb]*b_powers[maximum_+ma-mb]);
      }
      for(int mb=j/2+1;mb<=j;++mb)for(int ma=0;ma<=j;++ma) {
        const auto value=conjugate(result[index(offsets_,j,j-ma,j-mb)]);
        result[index(offsets_,j,ma,mb)]=((ma+mb)%2) ? ComplexJet((-1.)*value.real,(-1.)*value.imag) : value;
      }
    }
    if(distance<4.*spacing_) {
        const auto native=wigner(a,b,maximum_,offsets_);
        const auto dist=beta_value<M_PI/2.?beta:Jet6(M_PI)-beta;
        const auto t=(dist-Jet6(2.*spacing_))/Jet6(2.*spacing_);
        const auto weight=t*t*t*(Jet6(10.)+t*(Jet6(-15.)+Jet6(6.)*t));
        for(int k=0;k<offsets_.back();++k)result[k]=native[k]+weight*(result[k]-native[k]);
    }
    return result;
}

std::vector<DescriptorCurvature::ComplexJet> WignerSpline::evaluate_jacobian(
    const DescriptorCurvature::ComplexJet &a,const DescriptorCurvature::ComplexJet &b) const {
    using namespace DescriptorCurvature;
    using Complex=std::complex<double>;
    const Complex avalue(a.real.value,a.imag.value),bvalue(b.real.value,b.imag.value);
    const double av=std::abs(avalue),bv=std::abs(bvalue);
    const double beta=2.*std::atan2(bv,av),distance=std::min(beta,M_PI-beta);
    if(distance<=2.*spacing_)return wigner(a,b,maximum_,offsets_);
    std::array<double,6> beta_gradient,a_phase_gradient,b_phase_gradient;
    for(int axis=0;axis<6;++axis) {
        const Complex da(a.real.gradient[axis],a.imag.gradient[axis]);
        const Complex db(b.real.gradient[axis],b.imag.gradient[axis]);
        const auto aa=std::conj(avalue)*da,bb=std::conj(bvalue)*db;
        beta_gradient[axis]=2.*(av*bb.real()/bv-bv*aa.real()/av)/(av*av+bv*bv);
        a_phase_gradient[axis]=aa.imag()/(av*av);
        b_phase_gradient[axis]=bb.imag()/(bv*bv);
    }
    const auto a_powers=phase_powers(avalue/av,maximum_),b_powers=phase_powers(bvalue/bv,maximum_);
    const int cell=std::min(cells_-1,static_cast<int>(beta/spacing_));
    const double t=(beta-cell*spacing_)/spacing_;
    const auto *polynomials=polynomials_.data()+std::size_t(cell)*offsets_.back();
    std::vector<ComplexJet> result(offsets_.back());
    for(int j=0;j<=maximum_;++j) {
      for(int ma=0;ma<=j;++ma)for(int mb=0;mb<=j/2;++mb) {
        const int k=index(offsets_,j,ma,mb),pa=j-ma-mb,pb=ma-mb;
        const auto &p=polynomials[k];
        const double value=(((((p[5]*t+p[4])*t+p[3])*t+p[2])*t+p[1])*t+p[0]);
        const double first=((((5*p[5]*t+4*p[4])*t+3*p[3])*t+2*p[2])*t+p[1])*inverse_spacing_;
        const auto phase=a_powers[maximum_+pa]*b_powers[maximum_+pb];
        const auto v=value*phase;
        auto &out=result[k];out.real.value=v.real();out.imag.value=v.imag();
        for(int axis=0;axis<6;++axis) {
            const auto g=phase*Complex(first*beta_gradient[axis],
                value*(pa*a_phase_gradient[axis]+pb*b_phase_gradient[axis]));
            out.real.gradient[axis]=g.real();out.imag.gradient[axis]=g.imag();
        }
      }
      for(int mb=j/2+1;mb<=j;++mb)for(int ma=0;ma<=j;++ma) {
        const auto &source=result[index(offsets_,j,j-ma,j-mb)];
        auto &out=result[index(offsets_,j,ma,mb)];const double sign=(ma+mb)%2?-1.:1.;
        out.real.value=sign*source.real.value;out.imag.value=-sign*source.imag.value;
        out.real.gradient=sign*source.real.gradient;out.imag.gradient=-sign*source.imag.gradient;
      }
    }
    if(distance<4.*spacing_) {
        const auto native=wigner(a,b,maximum_,offsets_);
        const double t=(distance-2.*spacing_)/(2.*spacing_);
        const double weight=t*t*t*(10.+t*(-15.+6.*t));
        const double first=30.*t*t*(1.-t)*(1.-t)/(2.*spacing_)*(beta<M_PI/2.?1.:-1.);
        for(int k=0;k<offsets_.back();++k) {
            for(auto pair:{std::pair{&result[k].real,&native[k].real},
                           std::pair{&result[k].imag,&native[k].imag}}) {
                auto &out=*pair.first;const auto &ref=*pair.second;
                const double delta=out.value-ref.value;
                out.gradient=ref.gradient+weight*(out.gradient-ref.gradient);
                for(int axis=0;axis<6;++axis)out.gradient[axis]+=first*beta_gradient[axis]*delta;
                out.value=ref.value+weight*delta;
            }
        }
    }
    return result;
}

std::shared_ptr<const WignerSpline> make_wigner_spline(int maximum) {
    static std::mutex mutex;
    static std::map<int,std::weak_ptr<const WignerSpline>> tables;
    std::lock_guard lock(mutex);
    auto table=tables[maximum].lock();
    if(!table) {table=std::make_shared<const WignerSpline>(maximum);tables[maximum]=table;}
    return table;
}

std::vector<std::complex<double>> WignerSpline::evaluate_values(std::complex<double> a,std::complex<double> b) const {
    const double av=std::abs(a),bv=std::abs(b),beta=2.*std::atan2(bv,av),distance=std::min(beta,M_PI-beta);
    if(distance<=2.*spacing_)return wigner_values(a,b,maximum_,offsets_);
    a/=av;b/=bv;
    const auto a_powers=phase_powers(a,maximum_),b_powers=phase_powers(b,maximum_);
    const int cell=std::min(cells_-1,static_cast<int>(beta/spacing_));
    const double t=(beta-cell*spacing_)/spacing_;
    const auto *polynomials=polynomials_.data()+std::size_t(cell)*offsets_.back();
    std::vector<std::complex<double>> result(offsets_.back());
    for(int j=0;j<=maximum_;++j) {
      for(int ma=0;ma<=j;++ma)for(int mb=0;mb<=j/2;++mb) {
        const int k=index(offsets_,j,ma,mb);const auto &p=polynomials[k];
        const double value=(((((p[5]*t+p[4])*t+p[3])*t+p[2])*t+p[1])*t+p[0]);
        result[k]=value*a_powers[maximum_+j-ma-mb]*b_powers[maximum_+ma-mb];
      }
      for(int mb=j/2+1;mb<=j;++mb)for(int ma=0;ma<=j;++ma)
        result[index(offsets_,j,ma,mb)]=((ma+mb)%2?-1.:1.)*std::conj(result[index(offsets_,j,j-ma,j-mb)]);
    }
    if(distance<4.*spacing_) {
        const auto native=wigner_values(av*a,bv*b,maximum_,offsets_);
        const double t=(distance-2.*spacing_)/(2.*spacing_),w=t*t*t*(10.+t*(-15.+6.*t));
        for(int k=0;k<offsets_.back();++k)result[k]=native[k]+w*(result[k]-native[k]);
    }
    return result;
}
}
