#pragma once
// Opt-in numerical support for the experimental basis-selection benchmark.
// Factors contain [weighted design, residual at the immutable input anchor].
#include <Eigen/Dense>
#include "efs_factorization_observer.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace Training {
inline Eigen::MatrixXd efs_root(const Eigen::MatrixXd &rows,EFSFactorization kind=EFSFactorization::rows_qr) {
    if (!rows.allFinite()) throw std::runtime_error("Nonfinite square-root rows");
    const int n=rows.cols(), m=std::min<int>(rows.rows(),n);
    Eigen::MatrixXd out=Eigen::MatrixXd::Zero(n,n);
    if (m) {
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(rows);
        count_efs_factorization(kind);
        out.topRows(m)=qr.matrixQR().topRows(m).template triangularView<Eigen::Upper>();
    }
    return out;
}

inline Eigen::MatrixXd merge_efs_roots(const Eigen::MatrixXd &a,const Eigen::MatrixXd &b,
                                     EFSFactorization kind=EFSFactorization::channels_qr) {
    if (a.cols()!=b.cols()) throw std::runtime_error("Square-root dimension mismatch");
    int na=a.rows(),nb=b.rows();
    while(na && a.row(na-1).isZero(0.))--na;
    while(nb && b.row(nb-1).isZero(0.))--nb;
    Eigen::MatrixXd stacked(na+nb,a.cols());stacked<<a.topRows(na),b.topRows(nb);
    return efs_root(stacked,kind);
}

class EFSSquareRootFit {
    Eigen::MatrixXd A_, projected_, bias_inverse_;
    Eigen::VectorXd residual_, anchor_;
    double l2_;
    int bias_;
public:
    struct Fit {
        Eigen::VectorXd coefficients;
        Eigen::MatrixXd moment_covariance;
    };
    EFSSquareRootFit(const Eigen::MatrixXd &factor,const Eigen::VectorXd &anchor,
                    int bias,double l2):A_(factor.leftCols(anchor.size())),
                    residual_(factor.col(anchor.size())),anchor_(anchor),l2_(l2),bias_(bias) {
        if (factor.cols()!=anchor.size()+1 || bias<=0 || bias>=anchor.size() || !(l2>0))
            throw std::runtime_error("Invalid square-root fit inputs");
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A_.leftCols(bias),Eigen::ComputeThinU|Eigen::ComputeThinV);
        count_efs_factorization(EFSFactorization::bias_svd);
        const auto s=svd.singularValues();
        const double tolerance=128*std::numeric_limits<double>::epsilon()*bias*s.maxCoeff();
        int rank=0;while(rank<s.size() && s[rank]>tolerance)++rank;
        const Eigen::MatrixXd U=svd.matrixU().leftCols(rank);
        bias_inverse_=svd.matrixV().leftCols(rank)*s.head(rank).cwiseInverse().asDiagonal()*U.transpose();
        projected_=A_.rightCols(anchor.size()-bias)-U*(U.transpose()*A_.rightCols(anchor.size()-bias));
    }
    const Eigen::MatrixXd &projected_design() const {return projected_;}
    Fit fit(const std::vector<int> &ids,bool covariance=false) const {
        const int n=anchor_.size(),k=ids.size(),rows=A_.rows();
        if (!k || !std::is_sorted(ids.begin(),ids.end()) || ids.front()<0 || ids.back()>=n-bias_ ||
            std::adjacent_find(ids.begin(),ids.end())!=ids.end())throw std::runtime_error("Invalid square-root subset");
        Eigen::MatrixXd design=Eigen::MatrixXd::Zero(rows+k,k);
        Eigen::VectorXd target=residual_, selected_anchor(k);
        int selected=0;
        for (int j=0;j<n-bias_;++j) {
            if (selected<k && ids[selected]==j) {
                design.col(selected).head(rows)=projected_.col(j);
                selected_anchor[selected]=anchor_[bias_+j];++selected;
            } else target+=A_.col(bias_+j)*anchor_[bias_+j];
        }
        target-=A_.leftCols(bias_)*(bias_inverse_*target);
        Eigen::VectorXd rhs(rows+k);rhs.head(rows)=target;rhs.tail(k)=-std::sqrt(l2_)*selected_anchor;
        design.bottomRows(k).diagonal().array()=std::sqrt(l2_);
        Eigen::VectorXd scales(k);
        for(int i=0;i<k;++i){scales[i]=1/design.col(i).norm();design.col(i)*=scales[i];}
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(design);
        count_efs_factorization(EFSFactorization::fit_qr);
        Eigen::MatrixXd R=qr.matrixQR().topRows(k).template triangularView<Eigen::Upper>();
        Eigen::VectorXd transformed=(qr.householderQ().adjoint()*rhs).head(k);
        Eigen::VectorXd z=R.template triangularView<Eigen::Upper>().solve(transformed);
        Fit result;result.coefficients=Eigen::VectorXd::Zero(n);
        for(int i=0;i<k;++i)result.coefficients[bias_+ids[i]]=selected_anchor[i]+scales[i]*z[i];
        // Determine only the identifiable bias component; no penalty is added.
        result.coefficients.head(bias_)=bias_inverse_*(residual_+A_*(anchor_-result.coefficients));
        if(covariance) {
            Eigen::MatrixXd inverse=R.template triangularView<Eigen::Upper>().solve(Eigen::MatrixXd::Identity(k,k));
            inverse=scales.asDiagonal()*inverse;
            result.moment_covariance=inverse*inverse.transpose();
        }
        if(!result.coefficients.allFinite())throw std::runtime_error("Nonfinite square-root fit");
        return result;
    }
    double objective_increment(const Eigen::VectorXd &before,const Eigen::VectorXd &after) const {
        const Eigen::VectorXd change=A_*(after-before),res=A_*(before-anchor_)-residual_;
        // Extended accumulation avoids cancellation when a deletion has tiny cost.
        long double value=0;
        for(int i=0;i<change.size();++i)value+=static_cast<long double>(change[i])*(change[i]+2*res[i]);
        for(int i=bias_;i<before.size();++i)value+=static_cast<long double>(l2_)*(after[i]-before[i])*(after[i]+before[i]);
        return static_cast<double>(value);
    }
    double projected_increment(const Eigen::VectorXd &before,const Eigen::VectorXd &after) const {
        const int n=anchor_.size()-bias_;
        const Eigen::VectorXd r=residual_-A_.leftCols(bias_)*(bias_inverse_*residual_);
        const Eigen::VectorXd change=projected_*(after-before).tail(n);
        const Eigen::VectorXd res=projected_*(before-anchor_).tail(n)-r;
        long double value=0;
        for(int i=0;i<change.size();++i)value+=static_cast<long double>(change[i])*(change[i]+2*res[i]);
        for(int i=bias_;i<before.size();++i)value+=static_cast<long double>(l2_)*(after[i]-before[i])*(after[i]+before[i]);
        return static_cast<double>(value);
    }
};
} // namespace Training
