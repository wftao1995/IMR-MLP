#pragma once
#include "base/structures.h"
#include <Eigen/Cholesky>
#include <stdexcept>
#include <algorithm>
#include <cmath>

// Explicit mixed coordinates: periodic fractions (optionally recast in the
// reference length metric), followed by Cartesian coordinates normal to the
// periodic span. Atoms itself always stores Cartesian positions.
class PeriodicSubspaceCoordinates {
    std::vector<int> axes_;
    Eigen::Matrix3d reference_=Eigen::Matrix3d::Zero();
    Eigen::MatrixXd frame_,normal_,lower_;
    bool recast_=true;
    Eigen::MatrixXd basis(const Eigen::Matrix3d &cell) const {
        Eigen::MatrixXd b(rank(),3);
        for(int i=0;i<rank();++i) b.row(i)=cell.row(axes_[i]);
        return b;
    }
    Eigen::MatrixXd factor(const Eigen::Matrix3d &cell) const {
        const auto b=basis(cell);
        Eigen::MatrixXd l=Eigen::MatrixXd::Zero(rank(),rank());
        if(rank()) {
            Eigen::LLT<Eigen::MatrixXd> chol(b*b.transpose());
            if(chol.info()!=Eigen::Success) throw std::invalid_argument("Periodic cell vectors must be independent");
            l=chol.matrixL();
            if(!(l.diagonal().array()>0).all() || !l.allFinite()) throw std::invalid_argument("Invalid periodic Gram Cholesky factor");
        }
        return l;
    }
public:
    void init(const Atoms &reference,bool recast=true) {
        if(!reference.cell_.allFinite()) throw std::invalid_argument("Periodic subspace requires a finite cell");
        if(!reference.cartesian && !reference.cell_.fullPivLu().isInvertible())
            throw std::invalid_argument("Singular cells require Cartesian positions, not stored mixed fractional coordinates");
        reference_=reference.cell_;recast_=recast;axes_.clear();
        for(int i=0;i<3;++i) if(reference.pbc_[i]) axes_.push_back(i);
        lower_=factor(reference_);
        if(rank()) frame_=lower_.triangularView<Eigen::Lower>().solve(basis(reference_));
        else frame_.resize(0,3);
        normal_.resize(3-rank(),3);int count=0;
        for(int axis=0;axis<3 && count<3-rank();++axis) {
            Eigen::RowVector3d v=Eigen::RowVector3d::Unit(axis);
            for(int i=0;i<rank();++i) v-=v.dot(frame_.row(i))*frame_.row(i);
            for(int i=0;i<count;++i) v-=v.dot(normal_.row(i))*normal_.row(i);
            if(v.norm()>1e-10) normal_.row(count++)=v.normalized();
        }
        if(count!=3-rank()) throw std::invalid_argument("Cannot construct periodic/open coordinate frame");
    }
    int rank() const {return axes_.size();}
    int cell_dofs() const {return rank()*(rank()+1)/2;}
    const std::vector<int> &axes() const {return axes_;}
    Eigen::VectorXd encode_cell(const Eigen::Matrix3d &cell) const {
        if(!cell.allFinite()) throw std::invalid_argument("Periodic subspace requires a finite cell");
        for(int i=0;i<3;++i) if(std::find(axes_.begin(),axes_.end(),i)==axes_.end() &&
            !cell.row(i).isApprox(reference_.row(i),1e-10))
            throw std::invalid_argument("Open cell directions must remain fixed");
        const Eigen::MatrixXd l=factor(cell);
        if(!(basis(cell)-l*frame_).isZero(1e-8*std::max(1.,cell.norm())))
            throw std::invalid_argument("Periodic cell must share the reference Cholesky orientation");
        Eigen::VectorXd q(cell_dofs());int k=0;
        for(int i=0;i<rank();++i) for(int j=0;j<=i;++j)
            q[k++]=i==j?std::log(l(i,i)/lower_(i,i)):(l(i,j)-lower_(i,j))/lower_(i,i);
        return q;
    }
    Eigen::Matrix3d decode_cell(const Eigen::VectorXd &q) const {
        if(q.size()!=cell_dofs() || !q.allFinite()) throw std::invalid_argument("Invalid periodic cell coordinates");
        Eigen::MatrixXd l=lower_;int k=0;
        for(int i=0;i<rank();++i) for(int j=0;j<=i;++j) {
            l(i,j)=i==j?lower_(i,i)*std::exp(q[k]):lower_(i,j)+lower_(i,i)*q[k];++k;
        }
        if(!l.allFinite() || (rank() && !(l.diagonal().array()>0).all())) throw std::invalid_argument("Invalid periodic Gram coordinates");
        Eigen::Matrix3d result=reference_;const Eigen::MatrixXd b=l*frame_;
        for(int i=0;i<rank();++i) result.row(axes_[i])=b.row(i);
        return result;
    }
    Eigen::Matrix3d cell_derivative(const Eigen::VectorXd &q,int coordinate,bool second=false) const {
        if(coordinate<0 || coordinate>=cell_dofs()) throw std::invalid_argument("Periodic cell derivative index out of range");
        Eigen::Matrix3d result=Eigen::Matrix3d::Zero();int k=0;
        for(int i=0;i<rank();++i) for(int j=0;j<=i;++j,++k) if(k==coordinate) {
            const double coefficient=i==j?lower_(i,i)*std::exp(q[k]):(second?0.:lower_(i,i));
            result.row(axes_[i])=coefficient*frame_.row(j);
        }
        return result;
    }
    std::vector<Eigen::Matrix3d> cell_derivatives(const Eigen::Matrix3d &cell) const {
        const Eigen::VectorXd q=encode_cell(cell);std::vector<Eigen::Matrix3d> result;
        for(int k=0;k<cell_dofs();++k) result.push_back(cell_derivative(q,k));
        return result;
    }
    Eigen::Matrix3d atomic_jacobian(const Eigen::Matrix3d &cell) const {
        Eigen::Matrix3d j=Eigen::Matrix3d::Zero();
        if(rank()) {
            if(recast_) j.topRows(rank())=lower_.triangularView<Eigen::Lower>().solve(basis(cell));
            else j.topRows(rank())=basis(cell);
        }
        j.bottomRows(3-rank())=normal_;return j;
    }
    Eigen::RowVector3d encode_position(const Eigen::RowVector3d &p,const Eigen::Matrix3d &cell) const {
        Eigen::RowVector3d q=Eigen::RowVector3d::Zero();
        if(rank()) {
            const Eigen::MatrixXd b=basis(cell);
            const Eigen::RowVectorXd s=p*b.transpose()*(b*b.transpose()).llt().solve(Eigen::MatrixXd::Identity(rank(),rank()));
            if(recast_) q.head(rank())=s*lower_;else q.head(rank())=s;
        }
        q.tail(3-rank())=p*normal_.transpose();return q;
    }
    Eigen::RowVector3d decode_position(const Eigen::RowVector3d &q,const Eigen::Matrix3d &cell) const {
        return q*atomic_jacobian(cell);
    }
    Eigen::RowVector3d displacement(const Eigen::RowVector3d &a,const Eigen::RowVector3d &b,
                                    const Eigen::Matrix3d &cell,bool mic=true) const {
        Eigen::RowVector3d delta=b-a;
        if(!mic || rank()==0) return delta;
        Atoms pair;pair.cell_=cell;pair.pbc_={false,false,false};
        for(int axis:axes_) pair.pbc_[axis]=true;
        pair.pos_={decode_position(Eigen::RowVector3d::Zero(),cell),decode_position(delta,cell)};
        return encode_position(pair.get_rab(0,1),cell);
    }
    Eigen::VectorXd cell_gradient(const Eigen::Matrix3d &cell,const Eigen::Matrix3d &virial) const {
        const Eigen::VectorXd q=encode_cell(cell);Eigen::VectorXd g(cell_dofs());
        const Eigen::MatrixXd b=basis(cell);
        Eigen::MatrixXd gb(rank(),3);
        if(rank()) gb=-(b*b.transpose()).llt().solve(b)*virial;
        for(int k=0;k<cell_dofs();++k) {
            const Eigen::Matrix3d derivative=cell_derivative(q,k);g[k]=0;
            for(int i=0;i<rank();++i) g[k]+=gb.row(i).dot(derivative.row(axes_[i]));
        }
        return g;
    }
};
