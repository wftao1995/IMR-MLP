#pragma once
#include "base/structures.h"
#include <functional>
#include <memory>

class NonlinearCoordinates {
public:
    virtual ~NonlinearCoordinates() = default;
    virtual void initialize(const Eigen::VectorXd &) {}
    virtual Eigen::VectorXd reduce(const Eigen::VectorXd &) const = 0;
    virtual Eigen::VectorXd lift(const Eigen::VectorXd &) const = 0;
    virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd &) const = 0;
    virtual bool has_curvature() const { return false; }
    virtual Eigen::MatrixXd curvature(const Eigen::VectorXd &, const Eigen::VectorXd &) const;
    virtual bool accepted(const Eigen::VectorXd &) { return false; }
    Eigen::MatrixXd pullback_hessian(const Eigen::VectorXd &q, const Eigen::VectorXd &gradient,
                                   const Eigen::MatrixXd &hessian) const;
};

class CallbackCoordinates : public NonlinearCoordinates {
public:
    std::function<void(const Eigen::VectorXd &)> initializer;
    void initialize(const Eigen::VectorXd &x) override {if(initializer)initializer(x);}
    std::function<Eigen::VectorXd(const Eigen::VectorXd &)> to_reduced, to_full;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd &)> derivative;
    std::function<Eigen::MatrixXd(const Eigen::VectorXd &, const Eigen::VectorXd &)> second_derivative;
    Eigen::VectorXd reduce(const Eigen::VectorXd &x) const override { return to_reduced(x); }
    Eigen::VectorXd lift(const Eigen::VectorXd &q) const override { return to_full(q); }
    Eigen::MatrixXd jacobian(const Eigen::VectorXd &q) const override { return derivative(q); }
    bool has_curvature() const override { return bool(second_derivative); }
    Eigen::MatrixXd curvature(const Eigen::VectorXd &, const Eigen::VectorXd &) const override;
};

// Fixed-cell rigid groups in Cartesian optimizer coordinates. Periodic images
// are unwrapped once from the initial group anchor, never reassigned in trials.
class RigidBodyCoordinates : public NonlinearCoordinates {
    struct Group {
        std::vector<int> ids;
        std::vector<Eigen::Vector3d> body;
        std::vector<Eigen::RowVector3d> image_shifts;
        Eigen::Vector3d center;
        Eigen::MatrixXd rotation_basis;
        int offset = 0;
    };
    std::vector<Group> groups_;
    std::vector<int> free_;
    int natoms_ = 0, ndim_ = 0;
public:
    RigidBodyCoordinates(const Atoms &, const std::vector<std::vector<int>> &, double tolerance=1e-8);
    void initialize(const Eigen::VectorXd &) override;
    Eigen::VectorXd reduce(const Eigen::VectorXd &) const override;
    Eigen::VectorXd lift(const Eigen::VectorXd &) const override;
    Eigen::MatrixXd jacobian(const Eigen::VectorXd &) const override;
    bool has_curvature() const override { return true; }
    Eigen::MatrixXd curvature(const Eigen::VectorXd &, const Eigen::VectorXd &) const override;
    bool accepted(const Eigen::VectorXd &) override;
    int dimension() const { return ndim_; }
};
