#include "descriptor_BondOrder.h"
#include "solid_harmonic_curvature.h"

#include <limits>

void DescriptorBondOrder::compute_weighted_site_hvp(Neighborhood &nbh, const Eigen::VectorXd &weights,
    const DescriptorSiteMatrix &direction, double &value, DescriptorSiteMatrix &gradient,
    DescriptorSiteMatrix &product) {
    if (weights.size() != feature_size() || !weights.allFinite() ||
        direction.rows() != nbh.nbhcount_ || !direction.allFinite())
        GPERROR("Invalid BondOrder weighted curvature weights or direction");
    DescriptorSiteMatrix g = DescriptorSiteMatrix::Zero(nbh.nbhcount_, 3), hv = g;
    if (nbh.nbhcount_ == 0) { value = 0.; gradient = g; product = hv; return; }
    const int count = (maxl_+1)*(maxl_+1), features = maxl_/2;
    const double inverse_n = 1./nbh.nbhcount_;
    using Complex = std::complex<double>;
    std::vector<std::vector<DescriptorCurvature::ComplexJet>> local;
    std::vector<Complex> q(count), tangent(count), adjoint(count), tangent_adjoint(count);
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        if (nbh.distances_[j] <= 0.) GPERROR("BondOrder curvature requires nonzero neighbor distance");
        local.push_back(DescriptorCurvature::solid_harmonics(nbh.pos_[j], maxl_, true));
        for (int k = 0; k < count; ++k) {
            const auto &harmonic = local.back()[k];
            q[k] += inverse_n*Complex(harmonic.real.value, harmonic.imag.value);
            tangent[k] += inverse_n*Complex(harmonic.real.gradient.head<3>().dot(direction.row(j)),
                                            harmonic.imag.gradient.head<3>().dot(direction.row(j)));
        }
    }
    for (int l = 2; l <= maxl_; l += 2) {
        const double qw = weights[l/2-1], ww = weights[features+l/2-1];
        if (qw == 0. && ww == 0.) continue;
        double sq = 0., sq_dot = 0.;
        for (int m = -l; m <= l; ++m) {
            const int k = get_idx(l,m);
            sq += std::norm(q[k]);
            sq_dot += 2.*std::real(std::conj(q[k])*tangent[k]);
        }
        const double zero = std::numeric_limits<double>::epsilon()*std::numeric_limits<double>::epsilon();
        if (sq <= zero && (qw != 0. || (!normalize_w_outer && ww != 0.)))
            GPERROR("BondOrder weighted curvature is undefined at zero harmonic norm");
        double raw = 0., raw_dot = 0.;
        std::fill(adjoint.begin(), adjoint.end(), Complex{});
        std::fill(tangent_adjoint.begin(), tangent_adjoint.end(), Complex{});
        if (ww != 0.)
            for (int m1 = -l; m1 <= l; ++m1)
                for (int m2 = -l; m2 <= l; ++m2) {
                    const int m3 = -m1-m2;
                    if (std::abs(m3) > l) continue;
                    const int i = get_idx(l,m1), j = get_idx(l,m2), k = get_idx(l,m3);
                    const double c = AngularFuncs::Wigner3jIntCache(l,l,l,m1,m2,m3);
                    raw += c*std::real(q[i]*q[j]*q[k]);
                    raw_dot += c*std::real(tangent[i]*q[j]*q[k]+q[i]*tangent[j]*q[k]+q[i]*q[j]*tangent[k]);
                    const int indices[3] = {i,j,k};
                    for (int slot = 0; slot < 3; ++slot) {
                        const int a = indices[(slot+1)%3], b = indices[(slot+2)%3];
                        adjoint[indices[slot]] += c*q[a]*q[b];
                        tangent_adjoint[indices[slot]] += c*(tangent[a]*q[b]+q[a]*tangent[b]);
                    }
                }
        const double qfactor = qw == 0. ? 0. : std::sqrt(4.*M_PI/(2*l+1)*sq)/sq;
        const double qfactor_dot = qw == 0. ? 0. : -.5*qfactor*sq_dot/sq;
        const double normalization = normalize_w_outer || ww == 0. ? 1. : std::pow(sq,-1.5);
        const double normalization_dot = normalize_w_outer || ww == 0. ? 0. : -1.5*normalization*sq_dot/sq;
        const double normalized = raw*normalization;
        const double normalized_dot = raw_dot*normalization+raw*normalization_dot;
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int axis = 0; axis < 3; ++axis) {
                double a = 0., da = 0., b = 0., db = 0.;
                for (int m = -l; m <= l; ++m) {
                    const int k = get_idx(l,m);
                    const auto &s = local[j][k];
                    const Complex derivative = inverse_n*Complex(s.real.gradient[axis],s.imag.gradient[axis]);
                    const Complex derivative_dot = inverse_n*Complex(
                        s.real.hessian.block<1,3>(axis,0).dot(direction.row(j)),
                        s.imag.hessian.block<1,3>(axis,0).dot(direction.row(j)));
                    a += std::real(std::conj(q[k])*derivative);
                    da += std::real(std::conj(tangent[k])*derivative+std::conj(q[k])*derivative_dot);
                    b += std::real(adjoint[k]*derivative);
                    db += std::real(tangent_adjoint[k]*derivative+adjoint[k]*derivative_dot);
                }
                g(j,axis) += qw*qfactor*a+ww*normalization*b;
                hv(j,axis) += qw*(qfactor_dot*a+qfactor*da)+ww*(normalization_dot*b+normalization*db);
                if (!normalize_w_outer && ww != 0.) {
                    g(j,axis) -= 3.*ww*normalized*a/sq;
                    hv(j,axis) -= 3.*ww*((normalized_dot*a+normalized*da)/sq-normalized*a*sq_dot/(sq*sq));
                }
            }
    }
    Eigen::VectorXd features_value;
    compute(nbh, features_value);
    value = weights.dot(features_value);
    gradient = std::move(g);
    product = std::move(hv);
}

void DescriptorBondOrder::compute_weighted_site_hessian(Neighborhood &nbh, const Eigen::VectorXd &weights,
    double &value, DescriptorSiteMatrix &gradient, Eigen::MatrixXd &hessian) {
    Eigen::MatrixXd local(3*nbh.nbhcount_,3*nbh.nbhcount_);
    DescriptorSiteMatrix direction = DescriptorSiteMatrix::Zero(nbh.nbhcount_,3), g, hv;
    double energy = 0.;
    for (int column = 0; column < local.cols(); ++column) {
        direction.data()[column] = 1.;
        compute_weighted_site_hvp(nbh,weights,direction,energy,g,hv);
        local.col(column) = Eigen::Map<const Eigen::VectorXd>(hv.data(),hv.size());
        direction.data()[column] = 0.;
    }
    if (local.cols() == 0) compute_weighted_site_hvp(nbh,weights,direction,energy,g,hv);
    value = energy; gradient = std::move(g); hessian = std::move(local);
}
