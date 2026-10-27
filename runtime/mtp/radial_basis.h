#pragma once

// Shared host-side radial representation. No Eigen or device runtime in this interface.
#include "mlip.h"
#if __has_include("gpu-direct/radial_gpu_math.h")
#include "gpu-direct/radial_gpu_math.h"
#else
#include "radial_gpu_math.h"
#endif
#include <algorithm>
#include <cmath>
#include <istream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace MLIPCore::Radial {
[[noreturn]] inline void radial_parse_error(const std::string &message) {
    throw std::runtime_error("Invalid MTP radial basis: " + message);
}
template<class T> inline void read_assignment_value(std::istream &input, const char *key, T *value) {
    std::string eq;
    if (!(input >> eq >> *value) || eq != "=") radial_parse_error(std::string("expected '")+key+" = ...'");
}
template<class T> inline void read_assignment(std::istream &input, const char *key, T *value) {
    std::string actual;
    if (!(input >> actual) || actual != key) radial_parse_error(std::string("expected '")+key+" = ...'");
    read_assignment_value(input,key,value);
}
struct ParsedRadialBasis {
    MLIPCore::mlip::radial_basis_t basis;
    bool has_poly5_parameter = false;
    std::string name;
    std::string next_token;
};

inline ParsedRadialBasis read_radial_basis(std::istream &input, double scaling, bool legacy = false)
{
    ParsedRadialBasis parsed;
    parsed.basis.scaling = scaling;

    std::string basis_name;
    read_assignment_value(input, "radial_basis_type", &basis_name);
    parsed.name = basis_name;
    if (basis_name == "RBChebyshev") {
        parsed.basis.rb = MLIPCore::mlip::RB_CHEBVHAR;
    } else if (basis_name == "RBChebyCos") {
        parsed.basis.rb = MLIPCore::mlip::RB_CHEBVCOS;
    } else if (basis_name == "RBChebyshevPoly4") {
        parsed.basis.rb = MLIPCore::mlip::RB_CHEBYPOLY4;
    } else if (basis_name == "RBChebyshevPoly5") {
        parsed.basis.rb = MLIPCore::mlip::RB_CHEBYPOLY5;
    } else if (basis_name == "RBChebyshevConfigurable") {
        parsed.basis.rb = MLIPCore::mlip::RB_CHEBYCONFIGURABLE;
    } else if (basis_name == "RBPolyB") {
        parsed.basis.rb = MLIPCore::mlip::RB_POLYB;
    } else if (basis_name == "RBPolyBCos") {
        parsed.basis.rb = MLIPCore::mlip::RB_POLYBCOS;
    } else if (legacy && basis_name == "RBChebyshevI2") {
        parsed.basis.rb = MLIPCore::mlip::RB_LEGACY_I2;
    } else if (legacy && basis_name == "RBChebyshevPoly2") {
        parsed.basis.rb = MLIPCore::mlip::RB_LEGACY_POLY2;
    } else if (legacy && basis_name == "RBChebyshevPoly3") {
        parsed.basis.rb = MLIPCore::mlip::RB_LEGACY_POLY3;
        parsed.basis.cutoff_parameter = -1.5;
    } else {
        radial_parse_error("unsupported radial_basis_type '" + basis_name + "'");
    }

    read_assignment(input, "min_dist", &parsed.basis.min_dist);
    read_assignment(input, "max_dist", &parsed.basis.max_dist);
    read_assignment(input, "radial_basis_size", &parsed.basis.size);
    if (parsed.basis.size <= 0 || !std::isfinite(scaling) ||
        !std::isfinite(parsed.basis.min_dist) || !std::isfinite(parsed.basis.max_dist) ||
        parsed.basis.min_dist < 0 || parsed.basis.max_dist <= 0 ||
        !(parsed.basis.max_dist > parsed.basis.min_dist)) {
        radial_parse_error("invalid distance interval or radial_basis_size");
    }

    if (!(input >> parsed.next_token)) {
        radial_parse_error("unexpected end of radial basis block");
    }
    if ((parsed.basis.rb == MLIPCore::mlip::RB_CHEBYPOLY5 ||
         parsed.basis.rb == MLIPCore::mlip::RB_CHEBYPOLY4 ||
         parsed.basis.rb == MLIPCore::mlip::RB_LEGACY_POLY3) &&
        parsed.next_token == "rb_b") {
        read_assignment_value(input, "rb_b", &parsed.basis.cutoff_parameter);
        parsed.has_poly5_parameter = true;
        if (!(input >> parsed.next_token)) {
            radial_parse_error("unexpected end after rb_b");
        }
    }

    if (parsed.basis.rb == MLIPCore::mlip::RB_CHEBYPOLY4 && !parsed.has_poly5_parameter)
        radial_parse_error("RBChebyshevPoly4 requires rb_b");
    if (parsed.basis.rb == MLIPCore::mlip::RB_CHEBYCONFIGURABLE) {
        if (parsed.next_token == "basis_function_type") {
            std::string name; read_assignment_value(input, "basis_function_type", &name);
            if (name != "chebyshev" && name != "one_minus_chebyshev") radial_parse_error("unknown basis_function_type '"+name+"'");
            parsed.basis.one_minus_chebyshev = name == "one_minus_chebyshev";
            input >> parsed.next_token;
        }
        if (parsed.next_token != "cutoff_type") radial_parse_error("configurable basis is missing cutoff_type");
        const std::map<std::string, MLIPCore::mlip::rb_cutoff_type> cutoffs{
            {"quadratic",MLIPCore::mlip::RB_CUTOFF_QUADRATIC},
            {"normalized_quadratic",MLIPCore::mlip::RB_CUTOFF_NORMALIZED_QUADRATIC},
            {"cosine",MLIPCore::mlip::RB_CUTOFF_COSINE}, {"cosine_left",MLIPCore::mlip::RB_CUTOFF_COSINE_LEFT},
            {"cosine_unscaled",MLIPCore::mlip::RB_CUTOFF_COSINE_UNSCALED}, {"cosine_squared",MLIPCore::mlip::RB_CUTOFF_COSINE_SQUARED},
            {"sine_squared",MLIPCore::mlip::RB_CUTOFF_SINE_SQUARED},
            {"poly3",MLIPCore::mlip::RB_CUTOFF_POLY3}, {"poly4",MLIPCore::mlip::RB_CUTOFF_POLY4}, {"poly5",MLIPCore::mlip::RB_CUTOFF_POLY5},
            {"smoothstep5",MLIPCore::mlip::RB_CUTOFF_SMOOTHSTEP5}, {"smoothstep7",MLIPCore::mlip::RB_CUTOFF_SMOOTHSTEP7}};
        std::string name; read_assignment_value(input, "cutoff_type", &name);
        const auto cutoff = cutoffs.find(name);
        if (cutoff == cutoffs.end()) radial_parse_error("unknown configurable cutoff_type '"+name+"'");
        parsed.basis.cutoff = cutoff->second;
        if (name == "poly3" || name == "poly4" || name == "poly5")
            read_assignment(input,"cutoff_b",&parsed.basis.cutoff_parameter);
        if (name == "smoothstep7") {
            read_assignment(input,"cutoff_r1",&parsed.basis.cutoff_r1);
            read_assignment(input,"cutoff_r2",&parsed.basis.cutoff_r2);
            if (!(parsed.basis.min_dist < parsed.basis.cutoff_r1 && parsed.basis.cutoff_r1 <= parsed.basis.cutoff_r2 && parsed.basis.cutoff_r2 < parsed.basis.max_dist))
                radial_parse_error("invalid smoothstep7 interval");
        }
        read_assignment(input,"shift_type",&name);
        const std::map<std::string, MLIPCore::mlip::rb_transform_type> transforms{
            {"linear",MLIPCore::mlip::RB_TRANSFORM_LINEAR}, {"exponential",MLIPCore::mlip::RB_TRANSFORM_EXPONENTIAL},
            {"exponential_left",MLIPCore::mlip::RB_TRANSFORM_EXPONENTIAL_LEFT}, {"i2",MLIPCore::mlip::RB_TRANSFORM_I2},
            {"exponential_tanh",MLIPCore::mlip::RB_TRANSFORM_EXPONENTIAL_TANH}};
        const auto transform = transforms.find(name);
        if (transform == transforms.end()) radial_parse_error("unknown configurable shift_type '"+name+"'");
        parsed.basis.transform = transform->second;
        if (name != "linear" && name != "i2") read_assignment(input,"shift_lambda",&parsed.basis.transform_parameter);
        if (name == "exponential_tanh") read_assignment(input,"shift_scale_lambda",&parsed.basis.transform_scale_parameter);
        if (!(input >> parsed.next_token)) radial_parse_error("unexpected end after configurable basis");
    }
    if (legacy && parsed.basis.rb == MLIPCore::mlip::RB_CHEBYPOLY5 && !parsed.has_poly5_parameter)
        radial_parse_error("RBChebyshevPoly5 is missing rb_b");
    for (double v : {parsed.basis.cutoff_parameter, parsed.basis.transform_parameter,
                     parsed.basis.transform_scale_parameter, parsed.basis.cutoff_r1, parsed.basis.cutoff_r2})
        if (!std::isfinite(v)) radial_parse_error("non-finite radial parameter");
    parsed.basis.inverse_range=1/(parsed.basis.max_dist-parsed.basis.min_dist);
    parsed.basis.cosine_frequency=3.141592653589793238462643383279502884/parsed.basis.max_dist;
    return parsed;
}

inline bool is_poly(const mlip::radial_basis_t &b) {
    return b.rb == mlip::RB_POLYB || b.rb == mlip::RB_POLYBCOS;
}
inline bool excludes(const mlip::radial_basis_t &b, double r) {
    return r >= b.max_dist || (r < b.min_dist &&
        (b.rb == mlip::RB_CHEBYCONFIGURABLE || b.rb == mlip::RB_CHEBYPOLY4 ||
         b.rb == mlip::RB_CHEBYPOLY5 || b.rb == mlip::RB_LEGACY_POLY2 || b.rb == mlip::RB_LEGACY_POLY3));
}
// Defined in a host-only translation unit using the bundled Eigen headers.
void initialize_polynomial(mlip::radial_basis_t &basis);

// Keep the dense contraction out of the Chebyshev function's register allocation
// and instruction footprint. "inline" is for header linkage, not call expansion.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline void evaluate_polynomial(const mlip::radial_basis_t &b, double r,
                                double *values, double *derivatives) {
    using namespace GpuDirect;
    const double delta=b.max_dist-r;
    const int first=b.rb==mlip::RB_POLYB ? 3 : 2;
    double fc=1,dfc=0;
    if (first==2) RadialMath::envelope(RadialCutoff::CosineLeft,r,b.min_dist,b.max_dist,0,0,0,fc,dfc);
    double previous=b.scaling*delta;
    if(first==3) previous*=delta;
    const double first_power=previous*delta;
    const double first_normalization=b.polynomial_normalizations[0];
    const double first_value=first_normalization*first_power;
    const double first_derivative=-(first_normalization*first)*previous;
    // For one or two columns, keep each dot product in registers. General
    // column streaming would otherwise read the tiny output vectors back.
    if(b.size<=2) {
        const double second_value=b.size==2?b.polynomial_normalizations[1]*(first_power*delta):0;
        const double second_derivative=b.size==2?
            -(b.polynomial_normalizations[1]*(first+1))*first_power:0;
        for(int i=0;i<b.size;++i) {
            double value=0.0+b.polynomial_matrix[i]*first_value;
            double derivative=0.0+b.polynomial_matrix[i]*first_derivative;
            if(b.size==2) {
                value+=b.polynomial_matrix[2+i]*second_value;
                derivative+=b.polynomial_matrix[2+i]*second_derivative;
            }
            values[i]=value*fc;derivatives[i]=derivative*fc+value*dfc;
        }
        return;
    }
    // Seed from column zero instead of clearing both output arrays. Retain the
    // initial +0 operation of the native sum, including signed-zero behavior.
    for(int i=0;i<b.size;++i) {
        values[i]=0.0+b.polynomial_matrix[i]*first_value;
        derivatives[i]=0.0+b.polynomial_matrix[i]*first_derivative;
    }
    previous=first_power;
    for(int j=1;j<b.size;++j) {
        const double power=previous*delta;
        const double normalization=b.polynomial_normalizations[j];
        const double value=normalization*power;
        const double derivative=-(normalization*(j+first))*previous;
        const double *column=b.polynomial_matrix.data()+std::size_t(j)*b.size;
        // Traverse contiguous columns and generate each normalized power once.
        // Each output still sums in native j order, with the same products.
        for(int i=0;i<b.size;++i) {
            values[i]+=column[i]*value;
            derivatives[i]+=column[i]*derivative;
        }
        previous=power;
    }
    if(first==2)for(int i=0;i<b.size;++i) {
        derivatives[i]=derivatives[i]*fc+values[i]*dfc;
        values[i]*=fc;
    }
}

// Shared envelope-weighted recurrence for fixed and configurable coordinates.
inline void evaluate_chebyshev(int size, double scaling, double x, double dx,
                                double fc, double dfc, double *values, double *derivatives) {
    // Carry the envelope through the recurrence: no per-order product rule,
    // basis dispatch or extra polynomial recurrence in the common path.
    double previous=scaling*fc,dprevious=scaling*dfc;
    values[0]=previous;derivatives[0]=dprevious;
    if(size==1) return;
    double current=scaling*x*fc,dcurrent=scaling*(dx*fc+x*dfc);
    values[1]=current;derivatives[1]=dcurrent;
    const double twice_x=2*x,twice_dx=2*dx;
    for(int i=2;i<size;++i) {
        const double next=twice_x*current-previous;
        const double dnext=twice_dx*current+twice_x*dcurrent-dprevious;
        values[i]=next;derivatives[i]=dnext;
        previous=current;current=next;dprevious=dcurrent;dcurrent=dnext;
    }
}

// Keep the trigonometric call before computing the recurrence state, so x/dx
// need not survive a libm call inside the larger configurable dispatcher.
#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
inline void evaluate_cosine(const mlip::radial_basis_t &b, double r,
                            double *values, double *derivatives) {
    constexpr double pi=3.141592653589793238462643383279502884;
    const double frequency=b.cosine_frequency!=0?b.cosine_frequency:pi/b.max_dist;
    const double fc=.5*(1+std::cos(r*frequency));
    const double dfc=-.5*frequency*std::sin(r*frequency);
    const double inverse_range=b.inverse_range!=0?b.inverse_range:1/(b.max_dist-b.min_dist);
    const double x=(2*r-b.min_dist-b.max_dist)*inverse_range;
    evaluate_chebyshev(b.size,b.scaling,x,2*inverse_range,fc,dfc,values,derivatives);
}

// CheckBounds=false is only for callers that already applied excludes(b,r).
template<bool CheckBounds = true>
inline void evaluate(const mlip::radial_basis_t &b, double r, double *values, double *derivatives) {
    using namespace GpuDirect;
    if constexpr(CheckBounds) {
        if (excludes(b,r)) {
            std::fill_n(values,b.size,0.0); std::fill_n(derivatives,b.size,0.0); return;
        }
    }
    if (is_poly(b)) {
        evaluate_polynomial(b,r,values,derivatives);
        return;
    }
    if (b.rb==mlip::RB_CHEBVCOS) {
        evaluate_cosine(b,r,values,derivatives);
        return;
    }
    const double inverse_range=b.inverse_range!=0?b.inverse_range:1/(b.max_dist-b.min_dist);
    double x=(2*r-b.min_dist-b.max_dist)*inverse_range;
    double dx=2*inverse_range,fc=0,dfc=0;
    bool flipped=false;
    // Dedicated families have fixed envelopes. Avoid the configurable cutoff
    // dispatch and reuse the reciprocal already needed by their coordinate.
    switch(b.rb) {
    case mlip::RB_CHEBYCONFIGURABLE:
        RadialMath::coordinate(static_cast<RadialTransform>(b.transform),r,b.min_dist,b.max_dist,
            b.transform_parameter,b.transform_scale_parameter,x,dx);
        RadialMath::envelope(static_cast<RadialCutoff>(b.cutoff),r,b.min_dist,b.max_dist,
            b.cutoff_parameter,b.cutoff_r1,b.cutoff_r2,fc,dfc);
        flipped=b.one_minus_chebyshev;
        break;
    case mlip::RB_CHEBYPOLY4:
    case mlip::RB_CHEBYPOLY5:
    case mlip::RB_LEGACY_POLY3: {
        const int degree=b.rb==mlip::RB_LEGACY_POLY3?2:(b.rb==mlip::RB_CHEBYPOLY4?3:4);
        const double t=(r-b.min_dist)*inverse_range,q=1-t;
        const double previous=degree==2?q:(degree==3?q*q:q*q*q);
        const double power=previous*q;
        fc=power*(1+(b.cutoff_parameter+degree)*t);
        dfc=(-degree*previous*(1+(b.cutoff_parameter+degree)*t)
             +power*(b.cutoff_parameter+degree))*inverse_range;
        break;
    }
    case mlip::RB_LEGACY_I2: {
        const double q=(r-b.max_dist)*inverse_range;
        x=2*q*q-1;dx=4*q*inverse_range;
        fc=q*q;dfc=2*q*inverse_range;
        break;
    }
    case mlip::RB_LEGACY_POLY2: {
        const double q=1-(r-b.min_dist)*inverse_range;
        fc=q*q;dfc=-2*q*inverse_range;
        break;
    }
    default:
        const double delta=r-b.max_dist;
        fc=delta*delta;dfc=2*delta;
    }
    if (!flipped) {
        evaluate_chebyshev(b.size,b.scaling,x,dx,fc,dfc,values,derivatives);
        return;
    }
    // Keep 1-T_(i+1) separate: subtracting two envelope-weighted terms would
    // amplify cancellation near T=1. This family starts at T_1, not T_0.
    double previous=1,current=x,dprevious=0,dcurrent=dx;
    for(int i=0;i<b.size;++i) {
        const double polynomial=1-current,derivative=-dcurrent;
        values[i]=b.scaling*polynomial*fc;
        derivatives[i]=b.scaling*(derivative*fc+polynomial*dfc);
        if(i+1==b.size) break;
        const double next=2*x*current-previous,dnext=2*(dx*current+x*dcurrent)-dprevious;
        previous=current;current=next;dprevious=dcurrent;dcurrent=dnext;
    }
}
} // namespace MLIPCore::Radial
