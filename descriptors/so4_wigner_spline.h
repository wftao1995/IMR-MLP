#pragma once

#include "so4_density_kernel.h"
#include <array>
#include <memory>

namespace SO4Density {
// Experimental descriptor-only approximation. Public, non-unit Wigner
// functions do not use this table. Independent analytic beta derivatives at
// construction nodes determine each cached quintic Hermite polynomial.
class WignerSpline {
    int maximum_,cells_;
    std::vector<int> offsets_;
    double spacing_,inverse_spacing_;
    std::vector<std::array<double,6>> polynomials_;
    std::array<double,3> error_{};
    std::array<double,3> interpolate(const std::array<double,6> &p,double t) const;
public:
    explicit WignerSpline(int maximum);
    std::vector<DescriptorCurvature::ComplexJet> evaluate(
        const DescriptorCurvature::ComplexJet &a,const DescriptorCurvature::ComplexJet &b) const;
    // First-order consumer: Hessian fields are not part of this result's contract.
    std::vector<DescriptorCurvature::ComplexJet> evaluate_jacobian(
        const DescriptorCurvature::ComplexJet &a,const DescriptorCurvature::ComplexJet &b) const;
    std::vector<std::complex<double>> evaluate_values(std::complex<double> a,std::complex<double> b) const;
    int cells() const { return cells_; }
    std::array<double,3> error() const { return error_; }
};
std::shared_ptr<const WignerSpline> make_wigner_spline(int maximum);
}
