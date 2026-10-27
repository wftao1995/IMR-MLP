#pragma once
#include "../base/neighborhood.h"
#include <Eigen/SparseCholesky>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>

namespace NEBNumerics {
// Carry geometric directions, never stale Jacobian products. The next model
// re-evaluates each column against its own trusted state and frozen phase.
inline Eigen::MatrixXd soft_directions(const Eigen::MatrixXd &directions,
                                      const Eigen::MatrixXd &model,int count) {
    Eigen::MatrixXd result(directions.rows(),std::min<int>(count,model.cols()));
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(model,Eigen::ComputeThinV);
    if(svd.info()!=Eigen::Success || !svd.singularValues().allFinite())return result.leftCols(0);
    const auto &singular=svd.singularValues();
    int retained=0;
    for(int j=singular.size()-1;j>=0 && retained<result.cols();--j) {
        if(singular[j]<=1e-8*singular[0])continue;
        Eigen::VectorXd direction=directions*svd.matrixV().col(j);
        const double initial=direction.norm();
        for(int pass=0;pass<2;++pass)for(int k=0;k<retained;++k)
            direction-=result.col(k).dot(direction)*result.col(k);
        if(direction.allFinite() && direction.norm()>1e-8*initial)
            result.col(retained++)=direction.normalized();
    }
    return result.leftCols(retained);
}

// Distance-weighted graph of Packwood et al., arXiv:1604.02570, Eq. (16).
// Used only as a right preconditioner of the existing NEB residual Jacobian;
// it does not define a new path metric or remove translational freedoms.
class AtomicGraph {
    Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> factor;
    bool valid_=false;
public:
    bool build(const Atoms &atoms,double cutoff) {
        valid_=false;
        if(atoms.size()<2 || !(cutoff>0) || !std::isfinite(cutoff))return false;
        Atoms geometry=atoms;geometry.wrap();
        Neighborhoods neighbors(geometry,cutoff);
        double nearest=0;
        for(int i=0;i<atoms.size();++i) {
            double local=std::numeric_limits<double>::infinity();
            const auto &site=neighbors[i];
            for(int j=0;j<site.nbhcount_;++j)
                if(site.indices_[j]!=i && site.distances_[j]>0)local=std::min(local,site.distances_[j]);
            if(std::isfinite(local))nearest=std::max(nearest,local);
        }
        if(!(nearest>0))return false;
        if(2*nearest>cutoff)neighbors.init(geometry,2*nearest);
        std::vector<Eigen::Triplet<double>> entries;
        for(int i=0;i<atoms.size();++i) {
            double diagonal=.1;
            const auto &site=neighbors[i];
            for(int j=0;j<site.nbhcount_;++j) {
                const int other=site.indices_[j];const double distance=site.distances_[j];
                if(other==i || !(distance>0) || distance>=2*nearest)continue;
                const double weight=std::exp(-3*(distance/nearest-1));
                entries.emplace_back(i,other,-weight);diagonal+=weight;
            }
            entries.emplace_back(i,i,diagonal);
        }
        Eigen::SparseMatrix<double> matrix(atoms.size(),atoms.size());
        matrix.setFromTriplets(entries.begin(),entries.end());
        factor.compute(matrix);
        valid_=factor.info()==Eigen::Success && factor.vectorD().allFinite() && factor.vectorD().minCoeff()>0;
        return valid_;
    }
    bool valid() const {return valid_;}
    Eigen::MatrixXd solve(const Eigen::MatrixXd &rhs) const {return factor.solve(rhs);}
};
}
