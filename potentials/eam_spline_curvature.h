#pragma once
#include "potential.h"
#include <algorithm>
#include <map>
#include <vector>

// Internal curvature operations for the two fixed spline EAM adapters. The
// callbacks retain each adapter's species and density-direction conventions.
namespace Potentials::EAMSplineCurvature {
struct Jet { double value, first, second; };
inline Jet polynomial(const Eigen::MatrixXd &table,int interval,double fraction,double inverse_step) {
    return {((table(interval,3)*fraction+table(interval,4))*fraction+table(interval,5))*fraction+table(interval,6),
        (table(interval,0)*fraction+table(interval,1))*fraction+table(interval,2),
        (2*table(interval,0)*fraction+table(interval,1))*inverse_step};
}
inline Jet radial(const Eigen::MatrixXd &table,int count,double inverse_step,double gridmax,double r) {
    double p=r*inverse_step+1.;
    const int m=std::min(static_cast<int>(p),count-1);
    p=std::min(p-m,1.);
    auto result=polynomial(table,m,p,inverse_step);
    if(r>gridmax)result.first=result.second=0.;
    return result;
}
inline Jet embedding(const Eigen::MatrixXd &table,int count,double inverse_step,double rhomin,double rhomax,double rho) {
    double p=(rho-rhomin)*inverse_step+1.;
    const int m=std::max(1,std::min(static_cast<int>(p),count-1));
    p=std::min(p-m,1.);
    auto result=polynomial(table,m,p,inverse_step);
    if(rho>rhomax){result.value+=result.first*(rho-rhomax);result.second=0.;}
    return result;
}
inline Jet pair_from_rphi(const Jet &z,double r) {
    const double inverse=1./r;
    return {z.value*inverse,z.first*inverse-z.value*inverse*inverse,
        z.second*inverse-2*z.first*inverse*inverse+2*z.value*inverse*inverse*inverse};
}

template<class Density,class Pair,class Embedding>
void assemble(const Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian,
    Eigen::MatrixXd *cellhess,Eigen::MatrixXd *lrhess,Density density,Pair pair,Embedding embedding) {
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse=Eigen::Matrix3d::Zero();
    if(cellhess){inverse=atom.cell_.inverse();*cellhess=Eigen::MatrixXd::Zero(9,9);*lrhess=Eigen::MatrixXd::Zero(9,3*atom.size());}
    struct Radial {Jet density,pair;};
    std::vector<Radial> radial_values;
    for(int i=0;i<nbhs.size();++i){
        const auto &n=nbhs[i];radial_values.resize(n.nbhcount_);double rho=0.;
        for(int j=0;j<n.nbhcount_;++j){auto &v=radial_values[j];
            v.density=density(n.chemmap_[j],n.centertype,n.distances_[j]);
            v.pair=pair(n.centertype,n.chemmap_[j],n.distances_[j]);rho+=v.density.value;}
        const auto f=embedding(n.centertype,rho);
        std::map<int,Eigen::Vector3d> gradient;gradient.emplace(i,Eigen::Vector3d::Zero());
        Eigen::Matrix<double,9,1> cell_gradient=Eigen::Matrix<double,9,1>::Zero();
        for(int j=0;j<n.nbhcount_;++j){const auto &v=radial_values[j];const int source=n.indices_[j];const double r=n.distances_[j];
            const Eigen::Vector3d u=n.pos_[j].transpose()/r;
            const double first=.5*v.pair.first+f.first*v.density.first,second=.5*v.pair.second+f.first*v.density.second;
            const Eigen::Matrix3d block=(first/r)*Eigen::Matrix3d::Identity()+(second-first/r)*u*u.transpose();
            if(cellhess){const Eigen::RowVector3d fractional=n.pos_[j]*inverse;
                for(int a=0;a<3;++a)for(int b=0;b<3;++b){cell_gradient[3*a+b]+=fractional[a]*v.density.first*u[b];
                    for(int l=0;l<3;++l)for(int q=0;q<3;++q)(*cellhess)(3*a+b,3*l+q)+=fractional[a]*block(b,q)*fractional[l];
                    for(int q=0;q<3;++q){const double value=fractional[a]*block(b,q);(*lrhess)(3*a+b,3*source+q)+=value;(*lrhess)(3*a+b,3*i+q)-=value;}}}
            if(source==i)continue;
            hessian(source,source)+=block;hessian(i,i)+=block;hessian(source,i)-=block;hessian(i,source)-=block;
            gradient.try_emplace(source,Eigen::Vector3d::Zero()).first->second+=v.density.first*u;gradient.at(i)-=v.density.first*u;
        }
        for(const auto &[a,qa]:gradient)for(const auto &[b,qb]:gradient)hessian(a,b)+=f.second*qa*qb.transpose();
        if(cellhess){*cellhess+=f.second*cell_gradient*cell_gradient.transpose();
            for(const auto &[source,q]:gradient)lrhess->middleCols(3*source,3)+=f.second*cell_gradient*q.transpose();}
    }
}
// Direct local curvature contraction: no atomic Hessian is constructed.
template<class Density,class Pair,class Embedding>
void apply(const Atoms &atom,AnyNeighborhoods &nbhs,const Eigen::VectorXd &direction,
    Eigen::VectorXd &product,Density density,Pair pair,Embedding embedding) {
    Eigen::VectorXd result=Eigen::VectorXd::Zero(direction.size());
    struct Radial {Jet density,pair;double alpha;};
    std::vector<Radial> radial_values;
    for(int i=0;i<nbhs.size();++i){
        const auto &n=nbhs[i];radial_values.resize(n.nbhcount_);double rho=0.,delta_rho=0.;
        for(int j=0;j<n.nbhcount_;++j){auto &v=radial_values[j];const double r=n.distances_[j];
            v.density=density(n.chemmap_[j],n.centertype,r);v.pair=pair(n.centertype,n.chemmap_[j],r);
            const Eigen::Vector3d w=direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i);
            v.alpha=n.pos_[j].dot(w)/r;rho+=v.density.value;delta_rho+=v.density.first*v.alpha;
        }
        const auto f=embedding(n.centertype,rho);
        for(int j=0;j<n.nbhcount_;++j){const auto &v=radial_values[j];const double r=n.distances_[j];
            const Eigen::Vector3d u=n.pos_[j].transpose()/r;
            const Eigen::Vector3d w=direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i);
            const double first=.5*v.pair.first+f.first*v.density.first,second=.5*v.pair.second+f.first*v.density.second;
            const Eigen::Vector3d contribution=second*v.alpha*u+(first/r)*(w-v.alpha*u)
                +f.second*delta_rho*v.density.first*u;
            result.segment<3>(3*n.indices_[j])+=contribution;result.segment<3>(3*i)-=contribution;
        }
    }
    product=std::move(result);
}

}
