#pragma once
#include "potentials/potential.h"
#include "tools/hessian_utils.h"
#include "analyse/compute_hessian.h"
namespace PairCellTest {
inline Atoms geometry(bool partial=false) {
    Atoms a;
    a.cell_ << 3.8,.35,.2,.1,4.1,.25,.15,.2,4.3;
    a.pbc_={true,!partial,true};
    a.pos_={{.3,.5,.4},{1.65,.8,.7},{2.2,2.4,1.2}}; a.chemmap_={0,1,0};
    return a;
}
inline Eigen::MatrixXd native_matrix(const Potentials::HessianMatrix &h,
    const Eigen::MatrixXd &cell,const Eigen::MatrixXd &cross) {
    Eigen::MatrixXd out=Eigen::MatrixXd::Zero(9+3*h.size,9+3*h.size);
    out.topLeftCorner(9,9)=cell; out.topRightCorner(9,3*h.size)=cross;
    out.bottomLeftCorner(3*h.size,9)=cross.transpose();
    out.bottomRightCorner(3*h.size,3*h.size)=HessianUtils::to_dense(h);
    return out;
}
// Independent total-energy oracle; affine native coordinates and fixed
// Cartesian consumer coordinates are perturbed directly, without J^T H J.
inline Eigen::MatrixXd energy_fd(Potentials::AnyPotential &pot,const Atoms &a,
    double step,bool native,bool strain=false,double cutoff=0.) {
    const int nc=strain ? 6 : 9,n=nc+3*a.size();
    const Eigen::Matrix3d inverse=a.cell_.inverse();
    auto energy=[&](const Eigen::VectorXd &q) {
        Atoms w=a;
        if(strain) {
            Eigen::Matrix3d f=Eigen::Matrix3d::Identity(); f.diagonal()+=q.head<3>();
            f(1,2)=f(2,1)=q[3]/2; f(0,2)=f(2,0)=q[4]/2; f(0,1)=f(1,0)=q[5]/2;
            w.cell_=a.cell_*f;
        } else for(int i=0;i<3;++i) for(int j=0;j<3;++j) w.cell_(i,j)+=q[3*i+j];
        for(int i=0;i<a.size();++i)
            w.pos_[i]=(native ? (a.pos_[i]*inverse*w.cell_).eval() : a.pos_[i])
                +q.segment<3>(nc+3*i).transpose();
        Neighborhoods nbhs(w,cutoff>0 ? cutoff : pot.cutoff); pot.getEFS(w,nbhs); return w.energy_;
    };
    Eigen::VectorXd q=Eigen::VectorXd::Zero(n); const double center=energy(q);
    Eigen::MatrixXd h(n,n);
    for(int i=0;i<n;++i) {
        q[i]=step; const double plus=energy(q); q[i]=-step; const double minus=energy(q); q[i]=0;
        h(i,i)=(plus-2*center+minus)/(step*step);
        for(int j=0;j<i;++j) {
            q[i]=step;q[j]=step;const double pp=energy(q);q[j]=-step;const double pm=energy(q);
            q[i]=-step;const double mm=energy(q);q[j]=step;const double mp=energy(q);q[i]=q[j]=0;
            h(i,j)=h(j,i)=(pp-pm+mm-mp)/(4*step*step);
        }
    }
    return h;
}
}
