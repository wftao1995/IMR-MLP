#pragma once
#include "base/structures.h"
#include <Eigen/Sparse>
#include <functional>
#include <optional>
#include <string>

namespace Potentials { class AnyPotential; }
namespace RelaxPreconditioning {

// All matrices below use the driver's current coordinate ordering. A physical
// 3N Cartesian Hessian must be pulled back before using it in other coordinates.
Eigen::MatrixXd read_full_hessian(const std::string &path,Eigen::Index expected);
Eigen::MatrixXd pullback(const Eigen::MatrixXd &hessian,const Eigen::MatrixXd &jacobian);
Eigen::MatrixXd select_coordinates(const Eigen::MatrixXd &hessian,const std::vector<int> &indices);

class DenseMetric {
    Eigen::MatrixXd matrix_,root_,inverse_root_;
public:
    DenseMetric()=default;
    explicit DenseMetric(const Eigen::MatrixXd &hessian,double floor=1e-8) { reset(hessian,floor); }
    // Spectral absolute value with a positive absolute floor; true gradients
    // remain separate and convergence must not be judged using solve(g).
    void reset(const Eigen::MatrixXd &hessian,double floor=1e-8);
    const Eigen::MatrixXd &matrix() const { return matrix_; }
    const Eigen::MatrixXd &root() const { return root_; }
    const Eigen::MatrixXd &inverse_root() const { return inverse_root_; }
    Eigen::VectorXd solve(const Eigen::VectorXd &gradient) const;
};

class WhiteningTransform {
    DenseMetric metric_;
    Eigen::VectorXd origin_;
public:
    WhiteningTransform()=default;
    WhiteningTransform(const Eigen::MatrixXd &hessian,const Eigen::VectorXd &origin,double floor=1e-8) { reset(hessian,origin,floor); }
    void reset(const Eigen::MatrixXd &hessian,const Eigen::VectorXd &origin,double floor=1e-8);
    Eigen::VectorXd reduce(const Eigen::VectorXd &physical) const;
    Eigen::VectorXd lift(const Eigen::VectorXd &white) const;
    Eigen::VectorXd gradient(const Eigen::VectorXd &physical_gradient) const;
    const DenseMetric &metric() const { return metric_; }
};

struct SparseOptions {
    double shift=1e-3;
    double relative_tolerance=1e-10;
    int max_iterations=1000;
};
class SparseMetric {
    Eigen::SparseMatrix<double> matrix_;
    double nearest_scale_=0,cutoff_=0;
    SparseOptions options_;
public:
    // Atom-major Cartesian Laplacian: weights=exp[-3*(r/medianNN-1)], range 2*medianNN.
    // Probe cutoff must find each atom's first neighbor. No-neighbor atoms
    // cause a clear failure (except a lone nonperiodic atom: shifted identity).
    // Empty active_indices means all coordinates; pass an explicit vector to
    // restrict a Cartesian mask. No LF/LC mixing is implied here.
    SparseMetric()=default;
    SparseMetric(const Atoms &atoms,double probe_cutoff,const SparseOptions &options={},
                 const std::vector<int> &active_indices={}) { reset(atoms,probe_cutoff,options,active_indices); }
    void reset(const Atoms &atoms,double probe_cutoff,const SparseOptions &options={},
               const std::vector<int> &active_indices={});
    const Eigen::SparseMatrix<double> &matrix() const { return matrix_; }
    double nearest_scale() const { return nearest_scale_; }
    double cutoff() const { return cutoff_; }
    Eigen::VectorXd solve(const Eigen::VectorXd &gradient) const;
};

struct PeriodicGram {
    std::vector<int> axes;
    Eigen::MatrixXd rows,gram,inverse,dual;
    Eigen::VectorXd heights;
    double condition=1;
};
PeriodicGram periodic_gram(const Eigen::Matrix3d &cell,const std::array<bool,3> &pbc);

struct AULEstimate {
    Eigen::VectorXd weights; // inverse metric compliance, before driver's penalty conventions
    Eigen::VectorXd correction_norm; // |initial residual| / sqrt(compliance)
};
AULEstimate estimate_aul(const Eigen::MatrixXd &jacobian,const Eigen::VectorXd &initial_residual,
                         const std::function<Eigen::VectorXd(const Eigen::VectorXd &)> &solve_metric={},
                         double compliance_floor=1e-12);

// Native curvature only: dense analytical interface first, then one native HVP
// per basis direction. Never force-FD; owns all temporary Atoms/EFS state.
Eigen::MatrixXd native_cartesian_hessian(Potentials::AnyPotential &potential,const Atoms &atoms,
                                         Eigen::Index max_dofs=4096);
class AcceptedFCUpdate {
    int interval_=0;
    std::size_t last_=0;
public:
    explicit AcceptedFCUpdate(int interval=0) { configure(interval); }
    void configure(int interval);
    void reset() { last_=0; }
    bool due(std::size_t accepted_steps) const;
    std::optional<Eigen::MatrixXd> compute(Potentials::AnyPotential &potential,const Atoms &atoms,
                                         std::size_t accepted_steps,Eigen::Index max_dofs=4096);
};
}
