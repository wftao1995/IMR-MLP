#include "potential_hybrid_scale.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace Potentials {
namespace {
// Ordinary derivatives through order two, with the product rule coefficients.
using ScalarDerivatives = std::array<double, 3>;
ScalarDerivatives multiply(const ScalarDerivatives &a, const ScalarDerivatives &b) {
    return {a[0]*b[0], a[1]*b[0]+a[0]*b[1],
            a[2]*b[0]+2*a[1]*b[1]+a[0]*b[2]};
}
ScalarDerivatives step_derivatives(double left, double right, double x) {
    if (x <= left) return {0,0,0};
    if (x >= right) return {1,0,0};
    const double length = right-left, t = (x-left)/length;
    return {t*t*t*(10+t*(-15+6*t)), 30*t*t*(1-t)*(1-t)/length,
            60*t*(1-t)*(1-2*t)/(length*length)};
}
struct FeatureCurvature {
    double value = 0;
    SiteCoordinateMatrix gradient;
    std::vector<Eigen::Matrix3d> hessian;
};
FeatureCurvature feature_curvature(Neighborhood &nbh, bool nearest,
                                   double left, double right) {
    FeatureCurvature q;
    const int n = nbh.nbhcount_;
    q.gradient = SiteCoordinateMatrix::Zero(n,3);
    q.hessian.assign(n,Eigen::Matrix3d::Zero());
    int selected = -1;
    if (nearest) {
        q.value = std::numeric_limits<double>::infinity();
        for (int j=0;j<n;++j) if (nbh.distances_[j] < q.value) {
            q.value=nbh.distances_[j]; selected=j;
        }
        for (int j=0;j<n;++j) if (j!=selected &&
            std::abs(nbh.distances_[j]-q.value) <= 16*std::numeric_limits<double>::epsilon()
                *std::max(1.,q.value))
            GPERROR("hybrid_scale: analytical min_distance requires a unique nearest neighbor");
    }
    for (int j=0;j<n;++j) {
        if (nearest && j!=selected) continue;
        const double r = nbh.distances_[j];
        if (!(r>0.)) GPERROR("hybrid_scale: analytical feature derivatives require positive distances");
        ScalarDerivatives f{r,1,0};
        if (!nearest) {
            f=step_derivatives(left,right,r);
            f={1-f[0],-f[1],-f[2]};
            q.value+=f[0];
        }
        const Eigen::Vector3d u=nbh.pos_[j].transpose()/r;
        q.gradient.row(j)=f[1]*u.transpose();
        q.hessian[j]=(f[1]/r)*Eigen::Matrix3d::Identity()
            +(f[2]-f[1]/r)*u*u.transpose();

    }
    return q;
}
}

bool PotentialHybridScale::supports_site_hvp() const {
    return std::all_of(ppots_.begin(),ppots_.end(),[](const auto &p){return p->supports_site_hvp();});
}
bool PotentialHybridScale::supports_site_hessian() const {
    return std::all_of(ppots_.begin(),ppots_.end(),[](const auto &p){return p->supports_site_hessian();});
}


bool PotentialHybridScale::feature_regular_for(const Atoms &source) const {
    if (feature_kind_ != FeatureKind::MinDistance) return true;
    auto atom = source;
    if (!atom.cartesian) atom.to_cartesian();
    Neighborhoods neighborhoods(atom, cutoff);
    for (int i = 0; i < neighborhoods.size(); ++i) {
        const auto &site = neighborhoods[i];
        double nearest = std::numeric_limits<double>::infinity();
        for (double r : site.distances_) nearest = std::min(nearest, r);
        int matches = 0;
        for (double r : site.distances_)
            matches += std::abs(r - nearest) <= 16. * std::numeric_limits<double>::epsilon() * std::max(1., nearest);
        if (matches > 1) return false;
    }
    return true;
}

void PotentialHybridScale::site_curvature(Neighborhood &nbh, const SiteCoordinateMatrix *direction,
    double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix *product,
    HessianMatrix *hessian) {
    const int n=nbh.nbhcount_;
    auto q=feature_curvature(nbh,feature_kind_==FeatureKind::MinDistance,
                             feature_r0,feature_r1);
    std::vector<ScalarDerivatives> weights(npot);
    ScalarDerivatives prefix{1,0,0};
    for (int p=0;p<npot-1;++p) {
        const auto step=step_derivatives(boundaries_[p].left,boundaries_[p].right,q.value);
        weights[p]=multiply(prefix,{1-step[0],-step[1],-step[2]});
        prefix=multiply(prefix,step);
    }
    weights.back()=prefix;
    energy=0;
    gradient=SiteCoordinateMatrix::Zero(n,3);
    if (product) *product=SiteCoordinateMatrix::Zero(n,3);
    if (hessian) {hessian->resize(n);hessian->setzero();}
    const double dqv=direction ? (q.gradient.array()*direction->array()).sum() : 0.;
    SiteCoordinateMatrix hqv;
    if (direction) {
        hqv.resize(n,3);
        for (int i=0;i<n;++i) hqv.row(i)=(*direction).row(i)*q.hessian[i];
    }
    Neighborhood sub;
    std::vector<int> rows;
    for (int p=0;p<npot;++p) {
        const auto &w=weights[p];
        make_site_neighborhood(nbh,pot_cutoffs_[p],sub,rows);
        double e=0;
        SiteCoordinateMatrix child_gradient, child_product, child_direction;
        HessianMatrix child_hessian;
        if (direction) {
            child_direction.resize(rows.size(),3);
            for (int j=0;j<int(rows.size());++j) child_direction.row(j)=direction->row(rows[j]);
            ppots_[p]->apply_site_hvp(sub,child_direction,e,child_gradient,child_product);
        } else ppots_[p]->evaluate_site_hessian(sub,e,child_gradient,child_hessian);
        SiteCoordinateMatrix g=SiteCoordinateMatrix::Zero(n,3);
        std::vector<int> local(n,-1);
        for (int j=0;j<int(rows.size());++j) {g.row(rows[j])=child_gradient.row(j);local[rows[j]]=j;}
        energy+=w[0]*e;
        gradient+=w[0]*g+e*w[1]*q.gradient;
        if (product) {
            const double dEv=(g.array()*direction->array()).sum();
            for (int j=0;j<int(rows.size());++j) product->row(rows[j])+=w[0]*child_product.row(j);
            *product+=w[1]*(dqv*g+dEv*q.gradient+e*hqv)+e*w[2]*dqv*q.gradient;
            continue;
        }
        auto child_h=[&](int i,int a,int j,int b) {
            return local[i]<0 || local[j]<0 ? 0. : child_hessian(local[i],a,local[j],b);
        };
        auto qh=[&](int i,int a,int j,int b) {return i==j ? q.hessian[i](a,b) : 0.;};
        for (int i=0;i<n;++i) for (int j=0;j<n;++j) for (int a=0;a<3;++a) for (int b=0;b<3;++b) {
            (*hessian)(i,a,j,b)+=w[0]*child_h(i,a,j,b)
                +w[1]*(g(i,a)*q.gradient(j,b)+q.gradient(i,a)*g(j,b)+e*qh(i,a,j,b))
                +e*w[2]*q.gradient(i,a)*q.gradient(j,b);

        }
    }
}

void PotentialHybridScale::apply_site_hvp(Neighborhood &n,const SiteCoordinateMatrix &v,
    double &e,SiteCoordinateMatrix &g,SiteCoordinateMatrix &product) {
    if (!supports_site_hvp()) GPERROR("hybrid_scale: site HVP requires native site curvature in every child");
    if (v.rows()!=n.nbhcount_ || v.cols()!=3 || !v.allFinite()) GPERROR("Invalid hybrid_scale site HVP direction");
    const SiteCoordinateMatrix direction=v;
    site_curvature(n,&direction,e,g,&product,nullptr);
}
void PotentialHybridScale::evaluate_site_hessian(Neighborhood &n,double &e,SiteCoordinateMatrix &g,HessianMatrix &h) {
    if (!supports_site_hessian()) GPERROR("hybrid_scale: Hessian requires native site curvature in every child");
    site_curvature(n,nullptr,e,g,nullptr,&h);
}


void PotentialHybridScale::applyHessianVectorProduct(Atoms &atom,AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction,Eigen::VectorXd &product) {
    if (!supports_site_hvp()) GPERROR("hybrid_scale: analytical HVP requires native site curvature in every child");
    if (direction.size()!=3*atom.size() || !direction.allFinite()) GPERROR("Invalid hybrid_scale HVP direction");
    Eigen::VectorXd result=Eigen::VectorXd::Zero(direction.size());
    for (int ii=0;ii<nbhs.size();++ii) {
        auto &n=nbhs[ii];
        SiteCoordinateMatrix v(n.nbhcount_,3),g,p;
        for (int j=0;j<n.nbhcount_;++j)
            v.row(j)=(direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*ii)).transpose();
        double e;apply_site_hvp(n,v,e,g,p);
        for (int j=0;j<n.nbhcount_;++j) {
            result.segment<3>(3*n.indices_[j])+=p.row(j).transpose();
            result.segment<3>(3*ii)-=p.row(j).transpose();
        }
    }
    product=std::move(result);
}

void PotentialHybridScale::assemble_curvature(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &h,
    Eigen::MatrixXd *cell,Eigen::MatrixXd *mixed) {
    if (!supports_site_hessian())
        GPERROR("hybrid_scale: analytical curvature requires native site derivatives in every child");
    Eigen::Matrix3d inverse;
    if (cell) {
        if (!atom.cell_.fullPivLu().isInvertible()) GPERROR("hybrid_scale: native cell Hessian requires an invertible cell");
        inverse=atom.cell_.inverse();cell->setZero(9,9);mixed->setZero(9,3*atom.size());
    }
    h.resize(atom.size());h.setzero();
    if (!add_efs) atom.resetEFS();else atom.setEFSflag();
    for (int ii=0;ii<nbhs.size();++ii) {
        auto &n=nbhs[ii];double e;SiteCoordinateMatrix g;HessianMatrix local;
        evaluate_site_hessian(n,e,g,local);
        atom.energy_+=e;atom.site_energies_[ii]+=e;
        std::vector<Eigen::RowVector3d> fractions;
        if (cell) for (const auto &r:n.pos_) fractions.push_back(r*inverse);
        for (int i=0;i<n.nbhcount_;++i) {
            const int ai=n.indices_[i];
            atom.forces_[ii]+=g.row(i);atom.forces_[ai]-=g.row(i);
            atom.stress_-=g.row(i).transpose()*n.pos_[i];
            for (int j=0;j<n.nbhcount_;++j) {
                const int aj=n.indices_[j];const auto &block=local(i,j);
                h(ai,aj)+=block;h(ai,ii)-=block;h(ii,aj)-=block;h(ii,ii)+=block;
                if (cell) for (int a=0;a<3;++a) for (int b=0;b<3;++b) for (int c=0;c<3;++c) {
                    (*mixed)(3*a+b,3*aj+c)+=fractions[i][a]*block(b,c);
                    (*mixed)(3*a+b,3*ii+c)-=fractions[i][a]*block(b,c);
                    for (int d=0;d<3;++d) (*cell)(3*a+b,3*d+c)+=fractions[i][a]*fractions[j][d]*block(b,c);
                }

            }
        }
    }
}

void PotentialHybridScale::getEFSHessian(Atoms &a,HessianMatrix &h) {Neighborhoods n(a,cutoff);getEFSHessian(a,n,h);}
void PotentialHybridScale::getEFSHessian(Atoms &a,AnyNeighborhoods &n,HessianMatrix &h) {assemble_curvature(a,n,h,nullptr,nullptr);}
void PotentialHybridScale::getEFSHessianOMP(Atoms &a,AnyNeighborhoods &n,HessianMatrix &h) {getEFSHessian(a,n,h);}
void PotentialHybridScale::getEFSHessianSymm(Atoms &a,HessianMatrix &h,double) {getEFSHessian(a,h);}
void PotentialHybridScale::getEFSHessianCell(Atoms &a,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &m) {Neighborhoods n(a,cutoff);getEFSHessianCell(a,n,h,c,m);}
void PotentialHybridScale::getEFSHessianCell(Atoms &a,AnyNeighborhoods &n,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &m) {assemble_curvature(a,n,h,&c,&m);}
void PotentialHybridScale::getEFSHessianCellOMP(Atoms &a,AnyNeighborhoods &n,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &m) {getEFSHessianCell(a,n,h,c,m);}
void PotentialHybridScale::getEFSHessianCellSymm(Atoms &a,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &m,double) {getEFSHessianCell(a,h,c,m);}
void PotentialHybridScale::getEFSHessianCellSymmOMP(Atoms &a,HessianMatrix &h,Eigen::MatrixXd &c,Eigen::MatrixXd &m,double) {getEFSHessianCell(a,h,c,m);}

}
