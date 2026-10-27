#include "relax_preconditioner.h"
#include "potentials/potential.h"
#include "base/neighborhood.h"
#include <Eigen/Eigenvalues>
#include <Eigen/IterativeLinearSolvers>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <numeric>

namespace RelaxPreconditioning {
namespace {
void require(bool condition,const std::string &message) {
    if (!condition) throw std::invalid_argument("Relaxation preconditioner: "+message);
}
void square(const Eigen::MatrixXd &matrix) {
    require(matrix.rows()==matrix.cols() && matrix.allFinite(),"expected a finite square matrix");
    if (matrix.size()) {
        const double scale=std::max(1.0,matrix.cwiseAbs().maxCoeff());
        require((matrix-matrix.transpose()).cwiseAbs().maxCoeff()<=1e-10*scale,"matrix must be symmetric");
    }
}
void vector(const Eigen::VectorXd &value,Eigen::Index size) {
    require(value.size()==size && value.allFinite(),"vector shape or finite-value mismatch");
}
Atoms geometry(const Atoms &atoms) {
    require(atoms.size()>0 && atoms.size()<=std::numeric_limits<int>::max()/3 &&
            atoms.chemmap_.size()==atoms.pos_.size() && atoms.cartesian && atoms.cell_.allFinite(),
            "requires a nonempty complete Cartesian structure");
    for (const auto &p:atoms.pos_) require(p.allFinite(),"non-finite atomic position");
    (void)periodic_gram(atoms.cell_,atoms.pbc_);
    return atoms;
}
}
Eigen::MatrixXd read_full_hessian(const std::string &path,Eigen::Index expected) {
    require(expected>=0 && expected<=4096,"full matrix file exceeds the 4096-coordinate dense budget");
    std::ifstream input(path); Eigen::Index count;
    require(bool(input) && bool(input>>count) && count==expected,"full Hessian file must begin with the full coordinate count: "+path);
    Eigen::MatrixXd result(count,count);
    for (Eigen::Index i=0;i<count;++i) for (Eigen::Index j=0;j<count;++j)
        require(bool(input>>result(i,j)) && std::isfinite(result(i,j)),"incomplete/non-finite full Hessian file: "+path);
    input>>std::ws;
    require(input.eof(),"trailing full Hessian data: "+path);
    square(result); return result;
}
Eigen::MatrixXd pullback(const Eigen::MatrixXd &hessian,const Eigen::MatrixXd &jacobian) {
    square(hessian);
    require(jacobian.rows()==hessian.rows() && jacobian.allFinite(),"Jacobian shape or finite-value mismatch");
    Eigen::MatrixXd result=jacobian.transpose()*hessian*jacobian;
    require(result.allFinite(),"matrix pullback overflow");
    return result;
}
Eigen::MatrixXd select_coordinates(const Eigen::MatrixXd &hessian,const std::vector<int> &indices) {
    square(hessian); std::vector<bool> seen(hessian.rows(),false);
    for (int i:indices) {
        require(i>=0 && i<hessian.rows() && !seen[i],"coordinate selection contains an invalid or duplicate index");
        seen[i]=true;
    }
    Eigen::MatrixXd result(indices.size(),indices.size());
    for (size_t i=0;i<indices.size();++i) for (size_t j=0;j<indices.size();++j)
        result(i,j)=hessian(indices[i],indices[j]);
    return result;
}
void DenseMetric::reset(const Eigen::MatrixXd &hessian,double floor) {
    square(hessian); require(std::isfinite(floor) && floor>0,"spectral floor must be finite and positive");
    if (!hessian.rows()) { matrix_.resize(0,0); root_.resize(0,0); inverse_root_.resize(0,0); return; }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen((hessian+hessian.transpose())*.5);
    require(eigen.info()==Eigen::Success && eigen.eigenvalues().allFinite(),"Hessian eigensolve failed");
    const Eigen::VectorXd spectrum=eigen.eigenvalues().cwiseAbs().cwiseMax(floor);
    const auto &u=eigen.eigenvectors();
    Eigen::MatrixXd matrix=u*spectrum.asDiagonal()*u.transpose();
    Eigen::MatrixXd root=u*spectrum.cwiseSqrt().asDiagonal()*u.transpose();
    Eigen::MatrixXd inverse_root=u*spectrum.cwiseSqrt().cwiseInverse().asDiagonal()*u.transpose();
    require(matrix.allFinite() && root.allFinite() && inverse_root.allFinite(),"spectral metric overflow");
    matrix_.swap(matrix);root_.swap(root);inverse_root_.swap(inverse_root);
}
Eigen::VectorXd DenseMetric::solve(const Eigen::VectorXd &gradient) const {
    vector(gradient,matrix_.rows());
    Eigen::VectorXd result=inverse_root_*(inverse_root_*gradient);
    require(result.allFinite(),"dense solve overflow"); return result;
}
void WhiteningTransform::reset(const Eigen::MatrixXd &hessian,const Eigen::VectorXd &origin,double floor) {
    vector(origin,hessian.rows()); DenseMetric metric(hessian,floor);
    origin_=origin; metric_=std::move(metric);
}
Eigen::VectorXd WhiteningTransform::reduce(const Eigen::VectorXd &physical) const {
    vector(physical,origin_.size()); Eigen::VectorXd result=metric_.root()*(physical-origin_);
    require(result.allFinite(),"whitening overflow");return result;
}
Eigen::VectorXd WhiteningTransform::lift(const Eigen::VectorXd &white) const {
    vector(white,origin_.size());Eigen::VectorXd result=origin_+metric_.inverse_root()*white;
    require(result.allFinite(),"unwhitening overflow");return result;
}
Eigen::VectorXd WhiteningTransform::gradient(const Eigen::VectorXd &physical_gradient) const {
    vector(physical_gradient,origin_.size());Eigen::VectorXd result=metric_.inverse_root().transpose()*physical_gradient;
    require(result.allFinite(),"gradient whitening overflow");return result;
}
void SparseMetric::reset(const Atoms &atoms,double probe_cutoff,const SparseOptions &options,const std::vector<int> &active_indices) {
    require(std::isfinite(probe_cutoff) && probe_cutoff>0 && std::isfinite(options.shift) && options.shift>0 &&
            std::isfinite(options.relative_tolerance) && options.relative_tolerance>0 && options.relative_tolerance<1 && options.max_iterations>0,
            "invalid sparse cutoff/shift/PCG options");
    Atoms work=geometry(atoms); const int ndof=3*work.size();
    double nearest=0,cutoff=0;
    std::vector<Eigen::Triplet<double>> entries;
    if (work.size()!=1 || !work.is_nonperiodic()) {
        Neighborhoods initial(work,probe_cutoff); std::vector<double> distances;
        distances.reserve(work.size());
        for (int i=0;i<work.size();++i) {
            const auto &site=initial[i]; double distance=std::numeric_limits<double>::infinity();
            for (int j=0;j<site.nbhcount_;++j)
                if (site.distances_[j]>0 && std::isfinite(site.distances_[j])) distance=std::min(distance,site.distances_[j]);
            require(std::isfinite(distance),"probe cutoff did not find every atom's nearest neighbor");
            distances.push_back(distance);
        }
        std::sort(distances.begin(),distances.end()); nearest=distances[distances.size()/2];
        if (distances.size()%2==0) nearest=.5*nearest+.5*distances[distances.size()/2-1];
        cutoff=2*nearest;
        require(std::isfinite(cutoff) && cutoff>0,"invalid median-neighbor cutoff");
        Neighborhoods neighbors(work,cutoff);
        for (int i=0;i<work.size();++i) {
            const auto &site=neighbors[i];
            for (int p=0;p<site.nbhcount_;++p) {
                const int j=site.indices_[p];
                require(j>=0 && j<work.size(),"neighbor source index is invalid");
                if (j<=i) continue; // one source orientation; self images cancel for atomic translations
                const double weight=std::exp(-3*(site.distances_[p]/nearest-1));
                require(std::isfinite(weight) && weight>0,"invalid sparse neighbor weight");
                for (int a=0;a<3;++a) {
                    const int x=3*i+a,y=3*j+a;
                    entries.emplace_back(x,x,weight);entries.emplace_back(y,y,weight);
                    entries.emplace_back(x,y,-weight);entries.emplace_back(y,x,-weight);
                }
            }
        }
    }
    for (int i=0;i<ndof;++i) entries.emplace_back(i,i,options.shift);
    Eigen::SparseMatrix<double> matrix(ndof,ndof);matrix.setFromTriplets(entries.begin(),entries.end());
    if (!active_indices.empty()) {
        std::vector<int> map(ndof,-1);
        for (size_t i=0;i<active_indices.size();++i) {
            const int index=active_indices[i];require(index>=0 && index<ndof && map[index]==-1,"invalid sparse coordinate selection");
            map[index]=i;
        }
        entries.clear();
        for (int k=0;k<matrix.outerSize();++k) for (Eigen::SparseMatrix<double>::InnerIterator entry(matrix,k);entry;++entry)
            if (map[entry.row()]>=0 && map[entry.col()]>=0) entries.emplace_back(map[entry.row()],map[entry.col()],entry.value());
        matrix.resize(active_indices.size(),active_indices.size());matrix.setFromTriplets(entries.begin(),entries.end());
    }
    matrix.makeCompressed();
    for (int k=0;k<matrix.outerSize();++k) for (Eigen::SparseMatrix<double>::InnerIterator entry(matrix,k);entry;++entry)
        require(std::isfinite(entry.value()),"sparse metric accumulation overflow");
    matrix_.swap(matrix); nearest_scale_=nearest;cutoff_=cutoff;options_=options;
}
Eigen::VectorXd SparseMetric::solve(const Eigen::VectorXd &gradient) const {
    vector(gradient,matrix_.rows()); if (!gradient.size() || gradient.isZero(0)) return Eigen::VectorXd::Zero(gradient.size());
    Eigen::ConjugateGradient<Eigen::SparseMatrix<double>,Eigen::Lower|Eigen::Upper,Eigen::DiagonalPreconditioner<double>> solver;
    solver.setTolerance(options_.relative_tolerance);solver.setMaxIterations(options_.max_iterations);solver.compute(matrix_);
    require(solver.info()==Eigen::Success,"PCG factor initialization failed");
    Eigen::VectorXd result=solver.solve(gradient);
    const double residual=(matrix_*result-gradient).stableNorm();
    require(solver.info()==Eigen::Success && result.allFinite() && std::isfinite(residual) &&
            residual<=options_.relative_tolerance*gradient.stableNorm()*2,"PCG did not reach the requested residual");
    return result;
}
PeriodicGram periodic_gram(const Eigen::Matrix3d &cell,const std::array<bool,3> &pbc) {
    require(cell.allFinite(),"non-finite periodic cell");PeriodicGram result;
    for (int i=0;i<3;++i) if (pbc[i]) result.axes.push_back(i);
    const Eigen::Index k=result.axes.size(); result.rows.resize(k,3);
    Eigen::MatrixXd units(k,3);Eigen::VectorXd lengths(k);
    for (Eigen::Index i=0;i<k;++i) {
        result.rows.row(i)=cell.row(result.axes[i]);lengths[i]=result.rows.row(i).stableNorm();
        require(std::isfinite(lengths[i]) && lengths[i]>0,"zero periodic lattice row");
        units.row(i)=result.rows.row(i)/lengths[i];
    }
    if (!k) { result.gram.resize(0,0);result.inverse.resize(0,0);result.dual.resize(3,0);result.heights.resize(0);return result; }
    const Eigen::MatrixXd correlation=units*units.transpose();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(correlation);
    require(eigen.info()==Eigen::Success && eigen.eigenvalues().minCoeff()>1e-12,"periodic rows are rank deficient");
    const Eigen::MatrixXd inverse_correlation=correlation.llt().solve(Eigen::MatrixXd::Identity(k,k));
    result.gram=lengths.asDiagonal()*correlation*lengths.asDiagonal();
    result.inverse=lengths.cwiseInverse().asDiagonal()*inverse_correlation*lengths.cwiseInverse().asDiagonal();
    result.dual=units.transpose()*inverse_correlation*lengths.cwiseInverse().asDiagonal();
    result.heights.resize(k);
    for (Eigen::Index i=0;i<k;++i) result.heights[i]=lengths[i]/std::sqrt(inverse_correlation(i,i));
    result.condition=result.gram.cwiseAbs().colwise().sum().maxCoeff()*result.inverse.cwiseAbs().colwise().sum().maxCoeff();
    require(result.gram.allFinite() && result.inverse.allFinite() && result.dual.allFinite() && result.heights.allFinite() &&
            std::isfinite(result.condition),"periodic Gram scale overflow");
    return result;
}
AULEstimate estimate_aul(const Eigen::MatrixXd &jacobian,const Eigen::VectorXd &initial_residual,
                         const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &solve_metric,double compliance_floor) {
    require(jacobian.allFinite() && initial_residual.size()==jacobian.rows() && initial_residual.allFinite() &&
            std::isfinite(compliance_floor) && compliance_floor>0,"invalid AUL Jacobian/residual/floor");
    AULEstimate result;result.weights.resize(jacobian.rows());result.correction_norm.resize(jacobian.rows());
    for (Eigen::Index i=0;i<jacobian.rows();++i) {
        const Eigen::VectorXd row=jacobian.row(i).transpose();
        if (row.isZero(0)) {
            require(initial_residual[i]==0,"cannot estimate a violated constraint with zero gradient");
            result.weights[i]=0;result.correction_norm[i]=0;continue;
        }
        const Eigen::VectorXd solved=solve_metric?solve_metric(row):row;
        vector(solved,row.size());const double compliance=row.dot(solved);
        require(std::isfinite(compliance) && compliance>0,"AUL metric has nonpositive constraint compliance");
        const double scale=std::max(compliance,compliance_floor);
        result.weights[i]=1/scale;result.correction_norm[i]=std::abs(initial_residual[i])/std::sqrt(scale);
    }
    require(result.weights.allFinite() && result.correction_norm.allFinite(),"AUL scale overflow");return result;
}
Eigen::MatrixXd native_cartesian_hessian(Potentials::AnyPotential &potential,const Atoms &atoms,Eigen::Index max_dofs) {
    require(max_dofs>0 && atoms.size()>0 && int64_t(atoms.size())*3<=max_dofs,"native Hessian exceeds the dense coordinate budget");
    Atoms work=geometry(atoms);
    require(std::isfinite(potential.cutoff) && potential.cutoff>0,"native Hessian requires a positive finite potential cutoff");
    const Eigen::Index n=3*work.size(); Eigen::MatrixXd result(n,n);
    Neighborhoods neighbors(work,potential.cutoff);
    if (potential.supports_analytical_hessian()) {
        Potentials::HessianMatrix blocks;blocks.size=0;
        potential.getEFSHessian(work,neighbors,blocks);
        require(blocks.size==work.size() && blocks.hessian.size()==size_t(work.size())*work.size(),"native dense Hessian has invalid block shape");
        for (int i=0;i<work.size();++i) for (int j=0;j<work.size();++j) result.block<3,3>(3*i,3*j)=blocks(i,j);
    } else {
        require(potential.supports_analytical_hessian_vector_product(),"potential has neither native dense Hessian nor native HVP");
        for (Eigen::Index i=0;i<n;++i) {
            Eigen::VectorXd direction=Eigen::VectorXd::Zero(n),product;direction[i]=1;
            potential.applyHessianVectorProduct(work,neighbors,direction,product);
            vector(product,n); result.col(i)=product;
        }
    }
    square(result);return result;
}
void AcceptedFCUpdate::configure(int interval) {
    require(interval>=0,"force-constant update interval must be nonnegative");interval_=interval;last_=0;
}
bool AcceptedFCUpdate::due(std::size_t accepted_steps) const {
    return interval_>0 && accepted_steps>last_ && accepted_steps%size_t(interval_)==0;
}
std::optional<Eigen::MatrixXd> AcceptedFCUpdate::compute(Potentials::AnyPotential &potential,const Atoms &atoms,
                                                       std::size_t accepted_steps,Eigen::Index max_dofs) {
    if (!due(accepted_steps)) return std::nullopt;
    Eigen::MatrixXd hessian=native_cartesian_hessian(potential,atoms,max_dofs);
    last_=accepted_steps;return hessian;
}
}
