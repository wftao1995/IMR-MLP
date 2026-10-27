#include "descriptor_BispectrumSO4.h"
#include "so4_density_kernel.h"
#include "so4_wigner_spline.h"

namespace {
using Complex=std::complex<double>;
using DescriptorCurvature::ComplexJet;
using DescriptorCurvature::Jet6;
Complex value_of(const ComplexJet &v) { return {v.real.value,v.imag.value}; }
Complex derivative(const ComplexJet &v,int a) { return {v.real.gradient[a],v.imag.gradient[a]}; }
Complex value_of(const SO4Density::ComplexFirstOrder &v) { return {v.real.value,v.imag.value}; }
Complex derivative(const SO4Density::ComplexFirstOrder &v,int a) { return {v.real.gradient[a],v.imag.gradient[a]}; }
Complex second_direction(const ComplexJet &v,int a,const Eigen::RowVector3d &direction) {
    return {v.real.hessian.block<1,3>(a,0).dot(direction),
            v.imag.hessian.block<1,3>(a,0).dot(direction)};
}
// Coefficient-space reverse contraction. Size depends on angular order, not
// on atom count; local 3x3 jets are generated one edge at a time in two passes.
struct CubicDensity {
    std::vector<Complex> value,tangent,adjoint,dot_adjoint;
    double energy=0.;
    explicit CubicDensity(int n):value(n),tangent(n),adjoint(n),dot_adjoint(n) {}
    void term(int a,int b,int c,double weight) {
        const auto av=value[a],bv=value[b],cv=value[c];
        const auto ad=tangent[a],bd=tangent[b],cd=tangent[c];
        energy+=std::real(weight*std::conj(av)*bv*cv);
        adjoint[a]+=std::conj(weight*bv*cv);
        adjoint[b]+=weight*std::conj(av)*cv;
        adjoint[c]+=weight*std::conj(av)*bv;
        dot_adjoint[a]+=std::conj(weight*(bd*cv+bv*cd));
        dot_adjoint[b]+=weight*(std::conj(ad)*cv+std::conj(av)*cd);
        dot_adjoint[c]+=weight*(std::conj(ad)*bv+std::conj(av)*bd);
    }
    void snap(std::span<const Descriptor_BispectrumSO4::CouplingTerm> terms,double weight=1.) {
        int a=-1;Complex z{},zd{};
        const auto flush=[&] {
            if(a<0)return;
            energy+=std::real(std::conj(value[a])*z);
            adjoint[a]+=std::conj(z);dot_adjoint[a]+=std::conj(zd);
        };
        for(const auto &t:terms) {
            if(t.a!=a){flush();a=t.a;z={};zd={};}
            const auto av=std::conj(value[a]),ad=std::conj(tangent[a]);
            const auto bv=value[t.b],cv=value[t.c],bd=tangent[t.b],cd=tangent[t.c];
            const double w=weight*t.weight;z+=w*bv*cv;zd+=w*(bd*cv+bv*cd);
            adjoint[t.b]+=w*av*cv;adjoint[t.c]+=w*av*bv;
            dot_adjoint[t.b]+=w*(ad*cv+av*cd);dot_adjoint[t.c]+=w*(ad*bv+av*bd);
        }
        flush();
    }

};
double sparse_value(const std::vector<Complex> &density,
    std::span<const Descriptor_BispectrumSO4::CouplingTerm> terms) {
    int a=-1;Complex z{};double value=0.;
    const auto flush=[&]{if(a>=0)value+=std::real(std::conj(density[a])*z);};
    for(const auto &t:terms) {
        if(t.a!=a){flush();a=t.a;z={};}
        z+=t.weight*density[t.b]*density[t.c];
    }
    flush();return value;
}
template<class Visit>
void coupling_terms(const Descriptor_BispectrumSO4 &descriptor,const std::vector<int> &offset,
                    int feature,Visit &&visit) {
    if (descriptor.uses_snap()) {
        for (const auto &t:descriptor.contraction_terms(feature))visit(t.a,t.b,t.c,t.weight);
        return;
    }
    const auto js=descriptor.get_bi_ind(feature);
    const int j1=js[0],j2=js[1],j=js[2],shift=(j1+j2-j)/2;
    for (int m=0;m<=j;++m)
        for (int mp=0;mp<=j/2;++mp)
            for (int m2=0;m2<=j2;++m2) {
                const int m1=m-m2+shift;
                if (m1<0 || m1>j1) continue;
                const double cg=AngularFuncs::ClebschGordanTransCache(j1,j2,j,m1,m2,m);
                for (int m2p=0;m2p<=j2;++m2p) {
                    const int m1p=mp-m2p+shift;
                    if (m1p<0 || m1p>j1) continue;
                    const double cgp=AngularFuncs::ClebschGordanTransCache(j1,j2,j,m1p,m2p,mp);
                    const double factor=(j%2==0 && mp==j/2)?1.:2.;
                    visit(SO4Density::index(offset,j,m,mp),SO4Density::index(offset,j1,m1,m1p),
                          SO4Density::index(offset,j2,m2,m2p),factor*cg*cgp);
                }
            }
}
}

std::vector<Descriptor_BispectrumSO4::cmpld> Descriptor_BispectrumSO4::neighbor_coefficient_values(
    const Eigen::RowVector3d &position,int type) const {
    if(type<0 || type>=static_cast<int>(weights_.size()))
        throw std::invalid_argument("SO4 neighbor species outside configured weights");
    const double r=position.norm();if(!(r>0.))throw std::invalid_argument("SO4 requires nonzero neighbor distances");
    double ar,ai,br,bi;
    SO4Density::parameters(position[0],position[1],position[2],r,radius_,convention_,radial_angle_map_=="atan",ar,ai,br,bi);
    auto c=spline_ ? spline_->evaluate_values(cmpld(ar,ai),cmpld(br,bi))
        : SO4Density::wigner_values(cmpld(ar,ai),cmpld(br,bi),twojmax_,coefficient_offsets_);
    const double fc=cutfun_.fcut(r)*weights_[type];
    for(int j=0;j<=twojmax_;++j)for(int k=coefficient_offsets_[j];k<coefficient_offsets_[j+1];++k)
        c[k]*=fc*basis_scale(j);
    return c;
}

std::vector<DescriptorCurvature::ComplexJet> Descriptor_BispectrumSO4::neighbor_coefficients(
    const Eigen::RowVector3d &position,int type,bool second_order) const {
    if (type<0 || type>=static_cast<int>(weights_.size()))
        throw std::invalid_argument("SO4 neighbor species outside configured weights");
    using namespace DescriptorCurvature;
    const auto x=Jet6::variable(position[0],0),y=Jet6::variable(position[1],1),
               z=Jet6::variable(position[2],2);
    const auto r=DescriptorCurvature::sqrt(x*x+y*y+z*z);
    if (!(r.value>0.)) throw std::invalid_argument("SO4 requires nonzero neighbor distances");
    Jet6 ar,ai,br,bi;
    SO4Density::parameters(x,y,z,r,radius_,convention_,radial_angle_map_=="atan",ar,ai,br,bi);
    auto result=spline_ ? (second_order ? spline_->evaluate(ComplexJet(ar,ai),ComplexJet(br,bi))
        : spline_->evaluate_jacobian(ComplexJet(ar,ai),ComplexJet(br,bi)))
        : SO4Density::wigner(ComplexJet(ar,ai),ComplexJet(br,bi),twojmax_,coefficient_offsets_);
    if(spline_ && !second_order) {
        const double fc=cutfun_.fcut(r.value),df=cutfun_.dfcut(r.value);
        for(int j=0;j<=twojmax_;++j)for(int k=coefficient_offsets_[j];k<coefficient_offsets_[j+1];++k) {
            const double scale=weights_[type]*basis_scale(j);
            for(auto *component:{&result[k].real,&result[k].imag}) {
                component->gradient=scale*(fc*component->gradient+df*component->value*r.gradient);
                component->value*=scale*fc;
            }
        }
        return result;
    }
    const double second=r.value<cutoff_ && cutfun_.getCutoffType()==CutoffFunctions::CT_COS_BASIC
        ? -.5*(M_PI/cutoff_)*(M_PI/cutoff_)*std::cos(M_PI*r.value/cutoff_) : 0.;
    const auto fc=DescriptorCurvature::compose(r,cutfun_.fcut(r.value),cutfun_.dfcut(r.value),second);
    for (int j=0;j<=twojmax_;++j)
        for (int k=coefficient_offsets_[j];k<coefficient_offsets_[j+1];++k)
            result[k]=Jet6(weights_[type]*basis_scale(j))*fc*result[k];
    return result;
}

void Descriptor_BispectrumSO4::neighbor_coefficient_jacobian(
    const Eigen::RowVector3d &position,int type,std::span<SO4Density::ComplexFirstOrder> result) const {
    if(type<0 || type>=static_cast<int>(weights_.size()))
        throw std::invalid_argument("SO4 neighbor species outside configured weights");
    if(result.size()!=static_cast<std::size_t>(coefficient_offsets_.back()))
        throw std::invalid_argument("SO4 first-order coefficient buffer has incompatible size");
    // Preserve the experimental spline evaluator and its existing contracts.
    // The direct path below never constructs Jet6 or second derivatives.
    if(spline_) {
        const auto coefficients=neighbor_coefficients(position,type,false);
        for(std::size_t k=0;k<result.size();++k) {
            result[k].real.value=coefficients[k].real.value;
            result[k].imag.value=coefficients[k].imag.value;
            result[k].real.gradient=coefficients[k].real.gradient.head<3>();
            result[k].imag.gradient=coefficients[k].imag.gradient.head<3>();
        }
        return;
    }
    using namespace SO4Density;
    const auto x=FirstOrder::variable(position[0],0),y=FirstOrder::variable(position[1],1),
               z=FirstOrder::variable(position[2],2);
    const auto r=SO4Density::sqrt(x*x+y*y+z*z);
    if(!(r.value>0.))throw std::invalid_argument("SO4 requires nonzero neighbor distances");
    FirstOrder ar,ai,br,bi;
    SO4Density::parameters(x,y,z,r,radius_,convention_,radial_angle_map_=="atan",ar,ai,br,bi);
    wigner_first_order(ComplexFirstOrder(ar,ai),ComplexFirstOrder(br,bi),twojmax_,coefficient_offsets_,result);
    const auto fc=SO4Density::compose(r,cutfun_.fcut(r.value),cutfun_.dfcut(r.value));
    for(int j=0;j<=twojmax_;++j)for(int k=coefficient_offsets_[j];k<coefficient_offsets_[j+1];++k)
        result[k]=FirstOrder(weights_[type]*basis_scale(j))*fc*result[k];
}

void Descriptor_BispectrumSO4::compute_global_density(const AnyNeighborhoods &sites,Eigen::VectorXd &out,
    Eigen::Tensor<double,3,Eigen::RowMajor> *atom_jacobian,
    Eigen::Tensor<double,3,Eigen::RowMajor> *strain_jacobian) const {
    const int n=sites.size(),count=coefficient_offsets_.back();
    out.setZero(bi_size);
    if (atom_jacobian) { atom_jacobian->resize(bi_size,n,3); atom_jacobian->setZero(); }
    if (strain_jacobian) { strain_jacobian->resize(bi_size,3,3); strain_jacobian->setZero(); }
    if (!n) return;
    const double scale=1./n;
    CubicDensity density(count);
    for (int i=0;i<n;++i) {
        const auto &site=sites[i];
        if (site.centertype<0 || site.centertype>=static_cast<int>(selfweights_.size()))
            throw std::invalid_argument("SO4 center species outside configured self weights");
        for (int j=0;j<=twojmax_;++j)
            for (int m=0;m<=j;++m)
                density.value[SO4Density::index(coefficient_offsets_,j,m,m)]+=scale*basis_scale(j)*selfweights_[site.centertype];
        for (int e=0;e<site.nbhcount_;++e) {
            const auto coefficients=neighbor_coefficient_values(site.pos_[e],site.chemmap_[e]);
            for (int k=0;k<count;++k) density.value[k]+=scale*coefficients[k];
        }
    }
    const bool derivatives=atom_jacobian || strain_jacobian;
    std::vector<std::vector<Complex>> adjoints;
    if(derivatives)adjoints.resize(bi_size);
    for(int f=0;f<bi_size;++f) {
        std::fill(density.adjoint.begin(),density.adjoint.end(),Complex{});density.energy=0.;
        if(uses_snap()) {
            if(derivatives)density.snap(contraction_terms(f));
            else density.energy=sparse_value(density.value,contraction_terms(f));
        } else coupling_terms(*this,coefficient_offsets_,f,[&](int a,int b,int c,double w){
            if(derivatives)density.term(a,b,c,w);
            else density.energy+=std::real(w*std::conj(density.value[a])*density.value[b]*density.value[c]);
        });
        out[f]=density.energy;
        if(derivatives)adjoints[f]=density.adjoint;
    }
    if(!derivatives)return;
    std::vector<SO4Density::ComplexFirstOrder> coefficients(count);
    for(int i=0;i<n;++i)for(int e=0;e<sites[i].nbhcount_;++e) {
        const auto &site=sites[i];const int neighbor=site.indices_[e];
        neighbor_coefficient_jacobian(site.pos_[e],site.chemmap_[e],coefficients);
        for(int f=0;f<bi_size;++f) {
            Eigen::RowVector3d g=Eigen::RowVector3d::Zero();
            for(int k=0;k<count;++k)for(int a=0;a<3;++a)
                g[a]+=scale*std::real(adjoints[f][k]*derivative(coefficients[k],a));
            if(atom_jacobian)for(int a=0;a<3;++a) {
                (*atom_jacobian)(f,neighbor,a)+=g[a];(*atom_jacobian)(f,i,a)-=g[a];
            }
            if(strain_jacobian)for(int a=0;a<3;++a)for(int b=0;b<3;++b)
                (*strain_jacobian)(f,a,b)+=g[a]*site.pos_[e][b];
        }
    }
}

void Descriptor_BispectrumSO4::compute_global_density_cell(const Atoms &source,Eigen::VectorXd &value,
    Eigen::Tensor<double,3,Eigen::RowMajor> &jacobian) const {
    if(source.size()==0) {value.setZero(bi_size);jacobian.resize(bi_size,3,3);jacobian.setZero();return;}
    Atoms atom=source;if(!atom.cartesian)atom.to_cartesian();
    if(!atom.cell_.fullPivLu().isInvertible())
        throw std::invalid_argument("SO4 raw cell derivatives require a nonsingular cell");
    Neighborhoods sites(atom,cutoff_);
    Eigen::Tensor<double,3,Eigen::RowMajor> strain;
    compute_global_density(sites,value,nullptr,&strain);
    jacobian.resize(bi_size,3,3);
    const Eigen::Matrix3d inverse_transpose=atom.cell_.inverse().transpose();
    for(int f=0;f<bi_size;++f) {
        Eigen::Matrix3d m;
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)m(a,b)=strain(f,a,b);
        const Eigen::Matrix3d g=inverse_transpose*m.transpose();
        for(int a=0;a<3;++a)for(int b=0;b<3;++b)jacobian(f,a,b)=g(a,b);
    }
}

void Descriptor_BispectrumSO4::compute_weighted_global_hvp(const Atoms &source,
    const Eigen::VectorXd &weights,const Eigen::VectorXd &direction,bool include_cell,
    double &value,Eigen::VectorXd &gradient,Eigen::VectorXd &product) const {
    if (!supports_weighted_site_hvp()) throw std::invalid_argument("SO4 global curvature requires a supported smooth cutoff");
    const int n=source.size(),cell_dof=include_cell?9:0,ndim=cell_dof+3*n;
    if (weights.size()!=bi_size || direction.size()!=ndim || !weights.allFinite() || !direction.allFinite())
        throw std::invalid_argument("SO4 global weights or direction have incompatible shape");
    if(n==0) {value=0.;gradient.setZero(ndim);product.setZero(ndim);return;}
    Atoms atom=source;
    if (!atom.cartesian) atom.to_cartesian();
    Eigen::Matrix3d inverse=Eigen::Matrix3d::Zero(),cell_direction=inverse;
    if (include_cell) {
        if (!atom.cell_.fullPivLu().isInvertible())
            throw std::invalid_argument("SO4 raw cell derivatives require a nonsingular cell");
        inverse=atom.cell_.inverse();
        cell_direction=Eigen::Map<const Eigen::Matrix<double,3,3,Eigen::RowMajor>>(direction.data());
    }
    Neighborhoods sites(atom,cutoff_);
    const double scale=n?1./n:0.;
    CubicDensity density(coefficient_offsets_.back());
    const auto edge_direction=[&](int center,const Neighborhood &site,int e) {
        Eigen::RowVector3d v=direction.segment<3>(cell_dof+3*site.indices_[e]).transpose()
                              -direction.segment<3>(cell_dof+3*center).transpose();
        if (include_cell) v+=(site.pos_[e]*inverse)*cell_direction;
        return v;
    };
    for (int i=0;i<n;++i) {
        const auto &site=sites[i];
        if (site.centertype<0 || site.centertype>=static_cast<int>(selfweights_.size()))
            throw std::invalid_argument("SO4 center species outside configured self weights");
        for (int j=0;j<=twojmax_;++j)
            for (int m=0;m<=j;++m)
                density.value[SO4Density::index(coefficient_offsets_,j,m,m)]+=scale*basis_scale(j)*selfweights_[site.centertype];
        for (int e=0;e<site.nbhcount_;++e) {
            const auto coefficients=neighbor_coefficients(site.pos_[e],site.chemmap_[e]);
            const auto v=edge_direction(i,site,e);
            for (int k=0;k<static_cast<int>(coefficients.size());++k) {
                density.value[k]+=scale*value_of(coefficients[k]);
                for (int a=0;a<3;++a) density.tangent[k]+=scale*derivative(coefficients[k],a)*v[a];
            }
        }
    }
    for (int f=0;f<bi_size;++f) {
        if (weights[f]==0.) continue;
        if(uses_snap())density.snap(contraction_terms(f),weights[f]);
        else coupling_terms(*this,coefficient_offsets_,f,[&](int a,int b,int c,double w){density.term(a,b,c,weights[f]*w);});
    }
    Eigen::VectorXd g=Eigen::VectorXd::Zero(ndim),hv=g;
    for (int i=0;i<n;++i)
        for (int e=0;e<sites[i].nbhcount_;++e) {
            const auto &site=sites[i];
            const auto coefficients=neighbor_coefficients(site.pos_[e],site.chemmap_[e]);
            const auto v=edge_direction(i,site,e);
            Eigen::RowVector3d eg=Eigen::RowVector3d::Zero(),eh=eg;
            for (int k=0;k<static_cast<int>(coefficients.size());++k)
                for (int a=0;a<3;++a) {
                    const auto d=derivative(coefficients[k],a);
                    eg[a]+=scale*std::real(density.adjoint[k]*d);
                    eh[a]+=scale*std::real(density.dot_adjoint[k]*d+density.adjoint[k]*second_direction(coefficients[k],a,v));
                }
            g.segment<3>(cell_dof+3*site.indices_[e])+=eg.transpose();
            g.segment<3>(cell_dof+3*i)-=eg.transpose();
            hv.segment<3>(cell_dof+3*site.indices_[e])+=eh.transpose();
            hv.segment<3>(cell_dof+3*i)-=eh.transpose();
            if (include_cell) {
                const Eigen::RowVector3d f=site.pos_[e]*inverse;
                for (int a=0;a<3;++a)
                    for (int b=0;b<3;++b) { g[3*a+b]+=f[a]*eg[b]; hv[3*a+b]+=f[a]*eh[b]; }
            }
        }
    value=density.energy; gradient=std::move(g); product=std::move(hv);
}

void Descriptor_BispectrumSO4::compute_weighted_global_hessian(const Atoms &atom,
    const Eigen::VectorXd &weights,bool include_cell,double &value,Eigen::VectorXd &gradient,
    Eigen::MatrixXd &hessian) const {
    const int dim=3*atom.size()+(include_cell?9:0);
    Eigen::MatrixXd result(dim,dim);
    Eigen::VectorXd direction=Eigen::VectorXd::Zero(dim),product,g;
    double energy=0.;
    for (int c=0;c<dim;++c) {
        direction[c]=1.;
        compute_weighted_global_hvp(atom,weights,direction,include_cell,energy,g,product);
        result.col(c)=product; direction[c]=0.;
    }
    if (!dim) compute_weighted_global_hvp(atom,weights,direction,include_cell,energy,g,product);
    value=energy; gradient=std::move(g); hessian=std::move(result);
}

void Descriptor_BispectrumSO4::compute_snap(Neighborhood &site,bi_type &out,dbi_type *jac) const {
    const int count=coefficient_offsets_.back();
    CubicDensity density(count);
    if(site.centertype<0 || site.centertype>=static_cast<int>(selfweights_.size()))
        throw std::invalid_argument("SO4 center species outside configured self weights");
    for(int j=0;j<=twojmax_;++j)for(int m=0;m<=j;++m)
        density.value[SO4Density::index(coefficient_offsets_,j,m,m)]=basis_scale(j)*selfweights_[site.centertype];
    // Dispatch once so ordinary values do not enter derivative-buffer or
    // adjoint loops. Both modes retain the same existing coefficient kernels.
    if(!jac) {
        for(int e=0;e<site.nbhcount_;++e) {
            const auto c=neighbor_coefficient_values(site.pos_[e],site.chemmap_[e]);
            for(int k=0;k<count;++k)density.value[k]+=c[k];
        }
        out.assign(bi_size,0.);
        for(int f=0;f<bi_size;++f)out[f]=sparse_value(density.value,contraction_terms(f));
        return;
    }
    // One call-local allocation: all edge first derivatives persist for the
    // feature contractions. No owner-shared scratch or per-edge vectors.
    std::vector<SO4Density::ComplexFirstOrder> local(static_cast<std::size_t>(site.nbhcount_)*count);
    for(int e=0;e<site.nbhcount_;++e) {
        auto c=std::span(local).subspan(static_cast<std::size_t>(e)*count,count);
        neighbor_coefficient_jacobian(site.pos_[e],site.chemmap_[e],c);
        for(int k=0;k<count;++k)density.value[k]+=value_of(c[k]);
    }
    out.assign(bi_size,0.);
    jac->assign(bi_size,DescriptorSiteMatrix::Zero(site.nbhcount_,3));
    for(int f=0;f<bi_size;++f) {
        std::fill(density.adjoint.begin(),density.adjoint.end(),Complex{});density.energy=0.;
        density.snap(contraction_terms(f));
        out[f]=density.energy;
        for(int e=0;e<site.nbhcount_;++e)for(int k=0;k<count;++k)
            for(int a=0;a<3;++a)(*jac)[f](e,a)+=std::real(density.adjoint[k]*derivative(local[static_cast<std::size_t>(e)*count+k],a));
    }
}
