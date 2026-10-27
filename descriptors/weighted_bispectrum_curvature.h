#pragma once

#include "descriptors.h"
#include "solid_harmonic_curvature.h"
#include <complex>
#include <stdexcept>

namespace DescriptorCurvature {
// A bispectrum is a weighted sum of conj(C_a)*C_b*C_c. Density coefficients
// are sums of independent neighbor contributions, so only local 3x3 second
// derivatives and directional coefficient/adjoint vectors are needed.
class WeightedBispectrum {
    using Complex = std::complex<double>;
    const DescriptorSiteMatrix &direction_;
    std::vector<std::vector<ComplexJet>> local_;
    std::vector<Complex> value_, tangent_, adjoint_, tangent_adjoint_;
    double energy_ = 0.;
public:
    WeightedBispectrum(int neighbors, int coefficients, const DescriptorSiteMatrix &direction)
        : direction_(direction), local_(neighbors,std::vector<ComplexJet>(coefficients)),
          value_(coefficients), tangent_(coefficients), adjoint_(coefficients), tangent_adjoint_(coefficients) {}
    void self(int coefficient, double value) { value_[coefficient] += value; }
    void add(int neighbor, int coefficient, const ComplexJet &local) {
        local_[neighbor][coefficient] = local;
        value_[coefficient] += Complex(local.real.value,local.imag.value);
        tangent_[coefficient] += Complex(local.real.gradient.head<3>().dot(direction_.row(neighbor)),
                                         local.imag.gradient.head<3>().dot(direction_.row(neighbor)));
    }
    void cubic(int a, int b, int c, Complex weight) {
        if (weight == Complex{}) return;
        const auto av = value_[a], bv = value_[b], cv = value_[c];
        const auto ad = tangent_[a], bd = tangent_[b], cd = tangent_[c];
        energy_ += std::real(weight*std::conj(av)*bv*cv);
        adjoint_[a] += std::conj(weight*bv*cv);
        adjoint_[b] += weight*std::conj(av)*cv;
        adjoint_[c] += weight*std::conj(av)*bv;
        tangent_adjoint_[a] += std::conj(weight*(bd*cv+bv*cd));
        tangent_adjoint_[b] += weight*(std::conj(ad)*cv+std::conj(av)*cd);
        tangent_adjoint_[c] += weight*(std::conj(ad)*bv+std::conj(av)*bd);
    }
    // Terms are grouped by output coefficient a. Reuse its Z and directional
    // Z contraction for value and reverse adjoints, instead of multiplying
    // conj(C_a) for every individual coupling contribution.
    template<class Terms> void cubic_plan(const Terms &terms,double feature_weight) {
        int a=-1;Complex z{},zd{};
        const auto flush=[&] {
            if(a<0)return;
            energy_+=std::real(std::conj(value_[a])*z);
            adjoint_[a]+=std::conj(z);tangent_adjoint_[a]+=std::conj(zd);
        };
        for(const auto &term:terms) {
            if(term.a!=a){flush();a=term.a;z={};zd={};}
            const int b=term.b,c=term.c;const double w=feature_weight*term.weight;
            const auto av=std::conj(value_[a]),ad=std::conj(tangent_[a]);
            const auto bv=value_[b],cv=value_[c],bd=tangent_[b],cd=tangent_[c];
            z+=w*bv*cv;zd+=w*(bd*cv+bv*cd);
            adjoint_[b]+=w*av*cv;adjoint_[c]+=w*av*bv;
            tangent_adjoint_[b]+=w*(ad*cv+av*cd);
            tangent_adjoint_[c]+=w*(ad*bv+av*bd);
        }
        flush();
    }
    void finish(double &value, DescriptorSiteMatrix &gradient, DescriptorSiteMatrix &product) const {
        DescriptorSiteMatrix g = DescriptorSiteMatrix::Zero(direction_.rows(),3), hv = g;
        for (int neighbor = 0; neighbor < direction_.rows(); ++neighbor)
            for (int k = 0; k < static_cast<int>(value_.size()); ++k) {
                const auto &coefficient = local_[neighbor][k];
                for (int a = 0; a < 3; ++a) {
                    const Complex derivative(coefficient.real.gradient[a],coefficient.imag.gradient[a]);
                    const Complex derivative_dot(
                        coefficient.real.hessian.block<1,3>(a,0).dot(direction_.row(neighbor)),
                        coefficient.imag.hessian.block<1,3>(a,0).dot(direction_.row(neighbor)));
                    g(neighbor,a) += std::real(adjoint_[k]*derivative);
                    hv(neighbor,a) += std::real(tangent_adjoint_[k]*derivative+adjoint_[k]*derivative_dot);
                }
            }
        if (!std::isfinite(energy_) || !g.allFinite() || !hv.allFinite())
            throw std::runtime_error("Non-finite weighted bispectrum curvature");
        value = energy_; gradient = std::move(g); product = std::move(hv);
    }
};

// Analytical HVP columns materialize only the requested weighted site matrix.
// In particular, there is no tensor indexed by both feature and neighbor pair.
template<class Descriptor>
void bispectrum_hessian(Descriptor &descriptor, Neighborhood &nbh, const Eigen::VectorXd &weights,
    double &value, DescriptorSiteMatrix &gradient, Eigen::MatrixXd &hessian) {
    Eigen::MatrixXd local(3*nbh.nbhcount_,3*nbh.nbhcount_);
    DescriptorSiteMatrix direction = DescriptorSiteMatrix::Zero(nbh.nbhcount_,3), g, product;
    double energy = 0.;
    for (int column = 0; column < local.cols(); ++column) {
        direction.data()[column] = 1.;
        descriptor.compute_weighted_site_hvp(nbh,weights,direction,energy,g,product);
        local.col(column) = Eigen::Map<const Eigen::VectorXd>(product.data(),product.size());
        direction.data()[column] = 0.;
    }
    if (local.cols() == 0) descriptor.compute_weighted_site_hvp(nbh,weights,direction,energy,g,product);
    value = energy; gradient = std::move(g); hessian = std::move(local);
}
} // namespace DescriptorCurvature
