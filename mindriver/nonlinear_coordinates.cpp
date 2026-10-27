#include "nonlinear_coordinates.h"
#include <numbers>

Eigen::MatrixXd NonlinearCoordinates::curvature(const Eigen::VectorXd &, const Eigen::VectorXd &) const {
    throw std::logic_error("Exact nonlinear Hessian requires a curvature callback");
}
Eigen::MatrixXd NonlinearCoordinates::pullback_hessian(const Eigen::VectorXd &q,
        const Eigen::VectorXd &g, const Eigen::MatrixXd &h) const {
    if(!has_curvature()) throw std::logic_error("Exact nonlinear Hessian requires a curvature callback");
    const auto j=jacobian(q);
    if(j.rows()!=g.size() || h.rows()!=g.size() || h.cols()!=g.size())
        throw std::invalid_argument("Nonlinear Hessian dimensions do not match the map");
    const auto c=curvature(q,g);
    if(c.rows()!=q.size() || c.cols()!=q.size() || !c.allFinite())
        throw std::invalid_argument("Nonlinear curvature callback has invalid dimensions or values");
    return j.transpose()*h*j+c;
}
Eigen::MatrixXd CallbackCoordinates::curvature(const Eigen::VectorXd &q,const Eigen::VectorXd &g) const {
    if(!second_derivative) return NonlinearCoordinates::curvature(q,g);
    return second_derivative(q,g);
}

namespace {
// Three-variable second-order jets differentiate Rodrigues' map analytically,
// including its zero-angle limit. This is independent of the potential Hessian.
struct Jet {
    double v=0;
    Eigen::Vector3d g=Eigen::Vector3d::Zero();
    Eigen::Matrix3d h=Eigen::Matrix3d::Zero();
    Jet()=default;
    Jet(double value):v(value){}
};
Jet operator+(const Jet&a,const Jet&b){Jet c;c.v=a.v+b.v;c.g=a.g+b.g;c.h=a.h+b.h;return c;}
Jet operator-(const Jet&a,const Jet&b){Jet c;c.v=a.v-b.v;c.g=a.g-b.g;c.h=a.h-b.h;return c;}
Jet operator*(const Jet&a,const Jet&b){Jet c;c.v=a.v*b.v;c.g=a.g*b.v+b.g*a.v;c.h=a.h*b.v+b.h*a.v+a.g*b.g.transpose()+b.g*a.g.transpose();return c;}
Jet unary(const Jet&a,double v,double d,double dd){Jet c;c.v=v;c.g=d*a.g;c.h=d*a.h+dd*a.g*a.g.transpose();return c;}
std::array<Jet,3> rotated(const Eigen::Vector3d &w,const Eigen::Vector3d &v) {
    std::array<Jet,3> x;for(int i=0;i<3;++i){x[i].v=w[i];x[i].g[i]=1;}
    const Jet s=x[0]*x[0]+x[1]*x[1]+x[2]*x[2];Jet a,b;
    if(s.v<1e-4) {
        a=Jet(1)-s*Jet(1./6)+s*s*Jet(1./120)-s*s*s*Jet(1./5040)+s*s*s*s*Jet(1./362880);
        b=Jet(.5)-s*Jet(1./24)+s*s*Jet(1./720)-s*s*s*Jet(1./40320)+s*s*s*s*Jet(1./3628800);
    } else {
        const double t=std::sqrt(s.v),sin=std::sin(t),cos=std::cos(t);
        a=unary(s,sin/t,(t*cos-sin)/(2*t*t*t),(-t*t*sin-3*t*cos+3*sin)/(4*std::pow(t,5)));
        b=unary(s,(1-cos)/s.v,(t*sin-2+2*cos)/(2*s.v*s.v),
            (t*t*cos-5*t*sin+8-8*cos)/(4*std::pow(t,6)));
    }
    const std::array<Jet,3> cross={x[1]*Jet(v[2])-x[2]*Jet(v[1]),x[2]*Jet(v[0])-x[0]*Jet(v[2]),x[0]*Jet(v[1])-x[1]*Jet(v[0])};
    const Jet dot=x[0]*Jet(v[0])+x[1]*Jet(v[1])+x[2]*Jet(v[2]);
    std::array<Jet,3> out;for(int i=0;i<3;++i)out[i]=Jet(v[i])+a*cross[i]+b*(x[i]*dot-s*Jet(v[i]));return out;
}
Eigen::MatrixXd rotation_basis(const std::vector<Eigen::Vector3d> &body,double tol) {
    if(body.size()==1)return Eigen::MatrixXd(3,0);
    Eigen::MatrixXd points(body.size(),3);for(size_t i=0;i<body.size();++i)points.row(i)=body[i];
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(points,Eigen::ComputeFullV);
    if(svd.singularValues()[0]<tol)throw std::invalid_argument("Rigid group has coincident atoms");
    const bool linear=svd.singularValues().size()<2 || svd.singularValues()[1]<=tol*svd.singularValues()[0];
    if(linear)return svd.matrixV().rightCols(2);
    return Eigen::Matrix3d::Identity();
}
}

RigidBodyCoordinates::RigidBodyCoordinates(const Atoms &source,const std::vector<std::vector<int>> &groups,double tol) {
    if(!source.cartesian)throw std::invalid_argument("Rigid groups require Cartesian input positions");
    Atoms image_frame=source;natoms_=source.size();std::vector<bool> used(natoms_,false);
    for(const auto &ids:groups) {
        if(ids.empty())throw std::invalid_argument("Rigid groups must not be empty");
        Group group;group.ids=ids;group.center.setZero();
        for(int id:ids) {
            if(id<0 || id>=natoms_ || used[id])throw std::invalid_argument("Rigid groups require distinct in-range nonoverlapping atom indices");used[id]=true;
            const Eigen::RowVector3d displacement=image_frame.get_rab(ids.front(),id);
            const Eigen::RowVector3d unwrapped=source.pos_[ids.front()]+displacement;
            group.image_shifts.push_back(unwrapped-source.pos_[id]);group.body.push_back(unwrapped.transpose());group.center+=unwrapped.transpose();
        }
        group.center/=ids.size();for(auto &p:group.body)p-=group.center;
        group.rotation_basis=rotation_basis(group.body,tol);group.offset=ndim_;
        ndim_+=3+group.rotation_basis.cols();groups_.push_back(std::move(group));
    }
    for(int i=0;i<natoms_;++i)if(!used[i]){free_.push_back(i);ndim_+=3;}
}
void RigidBodyCoordinates::initialize(const Eigen::VectorXd &full) {
    if(full.size()!=3*natoms_ || !lift(reduce(full)).isApprox(full,1e-7))
        throw std::invalid_argument("Rigid map input differs from its bound initial Cartesian frame");
}
Eigen::VectorXd RigidBodyCoordinates::reduce(const Eigen::VectorXd &full) const {
    if(full.size()!=3*natoms_)throw std::invalid_argument("Rigid map full dimension changed");
    Eigen::VectorXd q=Eigen::VectorXd::Zero(ndim_);
    for(const auto &group:groups_) {
        Eigen::Vector3d center=Eigen::Vector3d::Zero();Eigen::Matrix3d cov=Eigen::Matrix3d::Zero();
        for(size_t i=0;i<group.ids.size();++i)center+=full.segment<3>(3*group.ids[i])+group.image_shifts[i].transpose();center/=group.ids.size();
        q.segment<3>(group.offset)=center-group.center;
        if(group.rotation_basis.cols()) {
            for(size_t i=0;i<group.ids.size();++i)cov+=group.body[i]*(full.segment<3>(3*group.ids[i])+group.image_shifts[i].transpose()-center).transpose();
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(cov,Eigen::ComputeFullU|Eigen::ComputeFullV);
            Eigen::Matrix3d sign=Eigen::Matrix3d::Identity();sign(2,2)=(svd.matrixV()*svd.matrixU().transpose()).determinant();
            Eigen::Matrix3d r=svd.matrixV()*sign*svd.matrixU().transpose();
            // A linear group has an arbitrary rotation about its own axis. Use
            // the unique shortest rotation of the directed line instead.
            if(group.rotation_basis.cols()==2) {
                const Eigen::Vector3d from=group.body.back()-group.body.front();
                const Eigen::Vector3d to=full.segment<3>(3*group.ids.back())+group.image_shifts.back().transpose()-full.segment<3>(3*group.ids.front())-group.image_shifts.front().transpose();
                r=Eigen::Quaterniond::FromTwoVectors(from,to).toRotationMatrix();
            }
            Eigen::AngleAxisd angle(r);q.segment(group.offset+3,group.rotation_basis.cols())=group.rotation_basis.transpose()*(angle.angle()*angle.axis());
        }
    }
    int offset=ndim_-3*free_.size();for(int id:free_){q.segment<3>(offset)=full.segment<3>(3*id);offset+=3;}return q;
}
Eigen::VectorXd RigidBodyCoordinates::lift(const Eigen::VectorXd &q) const {
    if(q.size()!=ndim_)throw std::invalid_argument("Rigid map reduced dimension changed");
    Eigen::VectorXd full(3*natoms_);
    for(const auto &group:groups_) {
        const Eigen::Vector3d w=group.rotation_basis*q.segment(group.offset+3,group.rotation_basis.cols());
        for(size_t i=0;i<group.ids.size();++i) {
            const auto r=rotated(w,group.body[i]);
            for(int c=0;c<3;++c)full[3*group.ids[i]+c]=group.center[c]+q[group.offset+c]+r[c].v-group.image_shifts[i][c];
        }
    }
    int offset=ndim_-3*free_.size();for(int id:free_){full.segment<3>(3*id)=q.segment<3>(offset);offset+=3;}return full;
}
Eigen::MatrixXd RigidBodyCoordinates::jacobian(const Eigen::VectorXd &q) const {
    if(q.size()!=ndim_)throw std::invalid_argument("Rigid map reduced dimension changed");
    Eigen::MatrixXd j=Eigen::MatrixXd::Zero(3*natoms_,ndim_);
    for(const auto &group:groups_) {
        const Eigen::Vector3d w=group.rotation_basis*q.segment(group.offset+3,group.rotation_basis.cols());
        for(size_t i=0;i<group.ids.size();++i) {
            j.block<3,3>(3*group.ids[i],group.offset).setIdentity();const auto r=rotated(w,group.body[i]);
            for(int c=0;c<3;++c)j.block(3*group.ids[i]+c,group.offset+3,1,group.rotation_basis.cols())=r[c].g.transpose()*group.rotation_basis;
        }
    }
    int offset=ndim_-3*free_.size();for(int id:free_){j.block<3,3>(3*id,offset).setIdentity();offset+=3;}return j;
}
Eigen::MatrixXd RigidBodyCoordinates::curvature(const Eigen::VectorXd &q,const Eigen::VectorXd &g) const {
    if(q.size()!=ndim_ || g.size()!=3*natoms_)throw std::invalid_argument("Rigid curvature dimensions changed");
    Eigen::MatrixXd h=Eigen::MatrixXd::Zero(ndim_,ndim_);
    for(const auto &group:groups_) {
        const Eigen::Vector3d w=group.rotation_basis*q.segment(group.offset+3,group.rotation_basis.cols());
        Eigen::Matrix3d local=Eigen::Matrix3d::Zero();
        for(size_t i=0;i<group.ids.size();++i) {const auto r=rotated(w,group.body[i]);for(int c=0;c<3;++c)local+=g[3*group.ids[i]+c]*r[c].h;}
        h.block(group.offset+3,group.offset+3,group.rotation_basis.cols(),group.rotation_basis.cols())=group.rotation_basis.transpose()*local*group.rotation_basis;
    }return h;
}
bool RigidBodyCoordinates::accepted(const Eigen::VectorXd &q) {
    bool changed=false;
    for(auto &group:groups_) {
        const Eigen::Vector3d w=group.rotation_basis*q.segment(group.offset+3,group.rotation_basis.cols());
        if(w.norm()<.25)continue;
        for(auto &body:group.body){const auto r=rotated(w,body);for(int c=0;c<3;++c)body[c]=r[c].v;}
        group.center+=q.segment<3>(group.offset);group.rotation_basis=rotation_basis(group.body,1e-8);changed=true;
    }return changed;
}
