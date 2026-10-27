#include "descriptor_BispectrumSO3.h"
#include "weighted_bispectrum_curvature.h"

bool Descriptor_BispectrumSO3::supports_weighted_site_hvp() const { return true; }
bool Descriptor_BispectrumSO3::supports_weighted_site_hessian() const { return true; }

void Descriptor_BispectrumSO3::compute_weighted_site_hvp(Neighborhood &nbh, const Eigen::VectorXd &weights,
    const DescriptorSiteMatrix &direction, double &value, DescriptorSiteMatrix &gradient,
    DescriptorSiteMatrix &product) {
    if (weights.size() != feature_size() || !weights.allFinite() ||
        direction.rows() != nbh.nbhcount_ || !direction.allFinite())
        GPERROR("Invalid SO3 weighted curvature weights or direction");
    if(real_harmonics_) {
        Eigen::VectorXd density;std::vector<std::vector<DescriptorCurvature::Jet6>> local;
        real_density(nbh,density,&local);
        Eigen::VectorXd tangent=Eigen::VectorXd::Zero(density.size()),adjoint=tangent,tangent_adjoint=tangent;
        for(int neighbor=0;neighbor<nbh.nbhcount_;++neighbor) for(int coefficient=0;coefficient<density.size();++coefficient)
            tangent[coefficient]+=local[neighbor][coefficient].gradient.head<3>().dot(direction.row(neighbor));
        double energy=0.;
        for(const auto& term:real_couplings_) {
            const double weight=weights[2*term.feature]*term.real+weights[2*term.feature+1]*term.imag;
            if(weight==0.) continue;
            const double a=density[term.a],b=density[term.b],c=density[term.c],
                da=tangent[term.a],db=tangent[term.b],dc=tangent[term.c];
            energy+=weight*a*b*c;
            adjoint[term.a]+=weight*b*c;adjoint[term.b]+=weight*a*c;adjoint[term.c]+=weight*a*b;
            tangent_adjoint[term.a]+=weight*(db*c+b*dc);tangent_adjoint[term.b]+=weight*(da*c+a*dc);
            tangent_adjoint[term.c]+=weight*(da*b+a*db);
        }
        DescriptorSiteMatrix g=DescriptorSiteMatrix::Zero(nbh.nbhcount_,3),hv=g;
        for(int neighbor=0;neighbor<nbh.nbhcount_;++neighbor) for(int coefficient=0;coefficient<density.size();++coefficient) {
            const auto& jet=local[neighbor][coefficient];
            g.row(neighbor)+=adjoint[coefficient]*jet.gradient.head<3>().transpose();
            hv.row(neighbor)+=tangent_adjoint[coefficient]*jet.gradient.head<3>().transpose()
                +adjoint[coefficient]*(jet.hessian.topLeftCorner<3,3>()*direction.row(neighbor).transpose()).transpose();
        }
        if(!std::isfinite(energy) || !g.allFinite() || !hv.allFinite()) GPERROR("Non-finite SO3 real weighted curvature");
        value=energy;gradient=std::move(g);product=std::move(hv);return;
    }
    using namespace DescriptorCurvature;
    WeightedBispectrum contraction(nbh.nbhcount_,r_size*ysize,direction);
    std::vector<double> radial_value(r_size), radial_first(r_size), radial_second(r_size);
    for (int neighbor = 0; neighbor < nbh.nbhcount_; ++neighbor) {
        const auto &position = nbh.pos_[neighbor];
        const double radius = position.norm();
        if (!(radius > 0.)) GPERROR("SO3 curvature requires nonzero neighbor distances");
        const auto harmonics = solid_harmonics(position,l_max,true);
        const auto x = Jet6::variable(position[0],0), y = Jet6::variable(position[1],1),
                   z = Jet6::variable(position[2],2), r = DescriptorCurvature::sqrt(x*x+y*y+z*z);
        radical_basis->calc_derder(radius,radial_value,radial_first,radial_second);
        for (int n = 0; n < r_size; ++n) {
            const auto radial = compose(r,radial_value[n],radial_first[n],radial_second[n]);
            for (int lm = 0; lm < ysize; ++lm) contraction.add(neighbor,n*ysize+lm,radial*harmonics[lm]);
        }
    }
    for (int feature = 0; feature < b_index.rows(); ++feature) {
        // The public layout is raw real/imaginary pairs, including those
        // components that vanish by symmetry. No extra l normalization applies.
        const std::complex<double> weight(weights[2*feature],-weights[2*feature+1]);
        if (weight == std::complex<double>{}) continue;
        const int n = b_index(feature,0), l1 = b_index(feature,1),
                  l2 = b_index(feature,2), l = b_index(feature,3);
        const auto index = [&](int degree, int order) { return n*ysize+degree*(degree+1)+order; };
        for (int m = -l; m <= l; ++m)
            for (int m1 = -l1; m1 <= l1; ++m1) {
                const int m2 = m-m1;
                if (std::abs(m2) > l2) continue;
                const double cg = AngularFuncs::ClebschGordanInt(l1,l2,l,m1,m2,m);
                contraction.cubic(index(l,m),index(l1,m1),index(l2,m2),cg*weight);
            }
    }
    contraction.finish(value,gradient,product);
}

void Descriptor_BispectrumSO3::compute_weighted_site_hessian(Neighborhood &nbh, const Eigen::VectorXd &weights,
    double &value, DescriptorSiteMatrix &gradient, Eigen::MatrixXd &hessian) {
    DescriptorCurvature::bispectrum_hessian(*this,nbh,weights,value,gradient,hessian);
}
