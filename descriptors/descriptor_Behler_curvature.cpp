#include "descriptor_Behler.h"
#include "local_curvature_jet.h"

#include <array>
#include <algorithm>

using DescriptorCurvature::Jet6;

void DescriptorBehler::weighted_curvature(Neighborhood &nbh, const Eigen::VectorXd &weights,
    const DescriptorSiteMatrix *direction, double &value, DescriptorSiteMatrix &gradient,
    DescriptorSiteMatrix *product, Eigen::MatrixXd *hessian) {
    if (weights.size() != dispcount_ || !weights.allFinite() || (direction &&
        (direction->rows() != nbh.nbhcount_ || !direction->allFinite())))
        GPERROR("Invalid Behler weighted curvature weights or direction");
    validate_neighborhood(nbh);
    // Validate all feature domains, including zero-contracted features.
    Eigen::VectorXd features;
    compute(nbh, features);
    DescriptorSiteMatrix local_gradient = DescriptorSiteMatrix::Zero(nbh.nbhcount_, 3);
    DescriptorSiteMatrix local_product;
    Eigen::MatrixXd local_hessian;
    if (direction) local_product.setZero(nbh.nbhcount_, 3);
    else local_hessian.setZero(3 * nbh.nbhcount_, 3 * nbh.nbhcount_);
    const auto norm = [](const std::array<Jet6, 3> &x) {
        return DescriptorCurvature::sqrt(x[0]*x[0] + x[1]*x[1] + x[2]*x[2]);
    };
    const auto cutoff = [](const Jet6 &r, double rc) {
        return r.value >= rc ? Jet6(0.)
            : .5 * (DescriptorCurvature::cos((M_PI / rc) * r) + 1.);
    };
    const auto add = [&](const Jet6 &term, double weight, int i, int j) {
        if (!term.gradient.allFinite() || !term.hessian.allFinite())
            GPERROR("Behler weighted curvature is undefined at this angular configuration");
        const int neighbors[2] = {i, j};
        const int count = j < 0 ? 1 : 2;
        for (int p = 0; p < count; ++p) {
            local_gradient.row(neighbors[p]) += weight * term.gradient.segment<3>(3*p).transpose();
            for (int q = 0; q < count; ++q) {
                const Eigen::Matrix3d block = weight * term.hessian.block<3, 3>(3*p, 3*q);
                if (direction)
                    local_product.row(neighbors[p]) +=
                        (block * direction->row(neighbors[q]).transpose()).transpose();
                else local_hessian.block<3, 3>(3*neighbors[p], 3*neighbors[q]) += block;
            }
        }
    };
    for (int f = 0; f < dispcount_; ++f) {
        if (weights[f] == 0.) continue;
        const int gi = dispselect_[f];
        const double rc = featureparams_(f, 0), eta = featureparams_(f, 1);
        const double shift = featureparams_(f, 2), kappa = featureparams_(f, 3);
        const double zeta = featureparams_(f, 4), lambda = featureparams_(f, 5);
        const double angular_shift = weighted_ ? shift : 0.;
        const double pre = (weighted_ ? 2. : 1.) * std::pow(2., 1. - zeta);
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            if (nbh.distances_[i] <= 0. || nbh.distances_[i] >= rc) continue;
            std::array<Jet6, 3> pi;
            for (int a = 0; a < 3; ++a) pi[a] = Jet6::variable(nbh.pos_[i][a], a);
            const auto ri = norm(pi);
            const auto fi = cutoff(ri, rc);
            if (gi <= 3) {
                Jet6 term = fi;
                if (gi == 2) {
                    const auto dr = ri - shift;
                    term = DescriptorCurvature::exp((-eta) * dr * dr) * fi;
                } else if (gi == 3) term = DescriptorCurvature::cos(kappa * ri) * fi;
                add(term, weights[f] * (weighted_ ? species_weight(nbh.chemmap_[i]) : 1.), i, -1);
                continue;
            }
            for (int j = i + 1; j < nbh.nbhcount_; ++j) {
                if (nbh.distances_[j] <= 0. || nbh.distances_[j] >= rc) continue;
                std::array<Jet6, 3> pj, difference;
                for (int a = 0; a < 3; ++a) {
                    pj[a] = Jet6::variable(nbh.pos_[j][a], 3+a);
                    difference[a] = pi[a] - pj[a];
                }
                const auto rj = norm(pj);
                const auto fj = cutoff(rj, rc);
                auto cosine = (pi[0]*pj[0] + pi[1]*pj[1] + pi[2]*pj[2]) / (ri*rj);
                cosine.value = std::clamp(cosine.value, -1., 1.);
                const auto base = 1. + lambda * cosine;
                // At a collinear zero the base has zero Cartesian gradient.
                // For zeta>1 the Cartesian Hessian is zero, even when the
                // scalar power's second derivative is singular (1<zeta<2).
                const auto angular = base.value == 0. && zeta > 1. ? Jet6(0.)
                    : DescriptorCurvature::pow(base, zeta);
                const auto di = ri - angular_shift, dj = rj - angular_shift;
                auto squared_distance = di*di + dj*dj;
                auto cutoffs = fi*fj;
                if (gi == 4) {
                    const auto rij_squared=difference[0]*difference[0]+difference[1]*difference[1]+difference[2]*difference[2];
                    if(rij_squared.value==0.) {
                        if(eta*angular_shift!=0.)
                            GPERROR("Behler shifted G4 has a cusp at coincident neighbor rows");
                        // fc(sqrt(q))=1-pi^2*q/(4*rc^2)+O(q^2).
                        // q has zero Cartesian gradient here, so this first
                        // q coefficient supplies the exact Cartesian Hessian.
                        cutoffs=cutoffs*DescriptorCurvature::compose(rij_squared,1.,-M_PI*M_PI/(4.*rc*rc),0.);
                        squared_distance=squared_distance+rij_squared+angular_shift*angular_shift;
                    } else {
                        const auto rij=DescriptorCurvature::sqrt(rij_squared);
                        if(rij.value>=rc) continue;
                        const auto dij=rij-angular_shift;
                        squared_distance=squared_distance+dij*dij;
                        cutoffs=cutoffs*cutoff(rij,rc);
                    }
                }
                const auto term = pre * angular * DescriptorCurvature::exp((-eta)*squared_distance) * cutoffs;
                const double chemical = weighted_ ? species_weight(nbh.chemmap_[i]) * species_weight(nbh.chemmap_[j]) : 1.;
                add(term, weights[f] * chemical, i, j);
            }
        }
    }
    const double local_value = weights.dot(features);
    if (!std::isfinite(local_value) || !local_gradient.allFinite() ||
        (direction ? !local_product.allFinite() : !local_hessian.allFinite()))
        GPERROR("Non-finite Behler weighted descriptor value or curvature");
    value = local_value;
    gradient = std::move(local_gradient);
    if (direction) *product = std::move(local_product);
    else *hessian = std::move(local_hessian);
}

void DescriptorBehler::compute_weighted_site_hvp(Neighborhood &nbh, const Eigen::VectorXd &weights,
    const DescriptorSiteMatrix &direction, double &value, DescriptorSiteMatrix &gradient,
    DescriptorSiteMatrix &product) {
    weighted_curvature(nbh, weights, &direction, value, gradient, &product, nullptr);
}

void DescriptorBehler::compute_weighted_site_hessian(Neighborhood &nbh, const Eigen::VectorXd &weights,
    double &value, DescriptorSiteMatrix &gradient, Eigen::MatrixXd &hessian) {
    weighted_curvature(nbh, weights, nullptr, value, gradient, nullptr, &hessian);
}
