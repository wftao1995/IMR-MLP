#include "descriptor_BispectrumSO4.h"
#include "weighted_bispectrum_curvature.h"
#include "so4_density_kernel.h"

bool Descriptor_BispectrumSO4::supports_weighted_site_hvp() const {
    return radius_ > 0. && std::isfinite(radius_) &&
        (cutfun_.getCutoffType() == CutoffFunctions::CT_COS_BASIC ||
         cutfun_.getCutoffType() == CutoffFunctions::CT_HARD);
}
bool Descriptor_BispectrumSO4::supports_weighted_site_hessian() const { return supports_weighted_site_hvp(); }

void Descriptor_BispectrumSO4::compute_weighted_site_hvp(Neighborhood &nbh, const Eigen::VectorXd &weights,
    const DescriptorSiteMatrix &direction, double &value, DescriptorSiteMatrix &gradient,
    DescriptorSiteMatrix &product) {
    if (!supports_weighted_site_hvp()) GPERROR("SO4 weighted curvature requires cosine/basic or hard cutoff");
    if (weights.size() != feature_size() || !weights.allFinite() ||
        direction.rows() != nbh.nbhcount_ || !direction.allFinite())
        GPERROR("Invalid SO4 weighted curvature weights or direction");
    if (nbh.centertype < 0 || nbh.centertype >= static_cast<int>(selfweights_.size()))
        GPERROR("SO4 central species is outside the configured range");
    using namespace DescriptorCurvature;
    std::vector<int> offsets(twojmax_+1);
    int count = 0;
    for (int j = 0; j <= twojmax_; ++j) { offsets[j] = count; count += (j+1)*(j+1); }
    const auto index = [&](int j, int ma, int mb) { return offsets[j]+ma*(j+1)+mb; };
    WeightedBispectrum contraction(nbh.nbhcount_,count,direction);
    for (int j = 0; j <= twojmax_; ++j)
        for (int m = 0; m <= j; ++m) contraction.self(index(j,m,m),basis_scale(j)*selfweights_[nbh.centertype]);
    for(int neighbor=0;neighbor<nbh.nbhcount_;++neighbor) {
        const auto coefficients=neighbor_coefficients(nbh.pos_[neighbor],nbh.chemmap_[neighbor]);
        for(int k=0;k<count;++k)contraction.add(neighbor,k,coefficients[k]);
    }
    for (int feature = 0; feature < bi_size; ++feature) {
        if (uses_snap()) {
            contraction.cubic_plan(contraction_terms(feature),weights[feature]);
            continue;
        }
        if (weights[feature] == 0.) continue;
        const int j1 = bindex(feature,0), j2 = bindex(feature,1), j = bindex(feature,2);
        const int shift = (j1+j2-j)/2;
        for (int m = 0; m <= j; ++m)
            for (int mp = 0; mp <= j/2; ++mp) {
                const double symmetry = j%2 == 0 && mp == j/2 ? 1. : 2.;
                for (int m2 = 0; m2 <= j2; ++m2) {
                    const int m1 = m-m2+shift;
                    if (m1 < 0 || m1 > j1) continue;
                    const double cg = AngularFuncs::ClebschGordanTransCache(j1,j2,j,m1,m2,m);
                    for (int m2p = 0; m2p <= j2; ++m2p) {
                        const int m1p = mp-m2p+shift;
                        if (m1p < 0 || m1p > j1) continue;
                        const double cgp = AngularFuncs::ClebschGordanTransCache(j1,j2,j,m1p,m2p,mp);
                        contraction.cubic(index(j,m,mp),index(j1,m1,m1p),index(j2,m2,m2p),
                                          weights[feature]*symmetry*cg*cgp);
                    }
                }
            }
    }
    contraction.finish(value,gradient,product);
}

void Descriptor_BispectrumSO4::compute_weighted_site_hessian(Neighborhood &nbh, const Eigen::VectorXd &weights,
    double &value, DescriptorSiteMatrix &gradient, Eigen::MatrixXd &hessian) {
    DescriptorCurvature::bispectrum_hessian(*this,nbh,weights,value,gradient,hessian);
}
