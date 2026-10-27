#pragma once
#include "external/spglib_driver.h"

class NEBDriver;

// A fixed, labelled common subgroup of the initialized, unwrapped path. All
// projectors are orthogonal in NEB's scaled coordinates, including VC weights.
class NEBPathSymmetry {
    struct Operation {
        int index;
        Eigen::Matrix3d fractional, coordinate;
        Eigen::RowVector3d translation;
        std::vector<int> permutation;
        std::vector<Eigen::RowVector3d> lattice_images;
    };
    struct Orbit {
        std::vector<int> atoms;
        std::vector<Eigen::MatrixXd> maps;
    };
    std::vector<Operation> operations_;
    std::vector<Orbit> orbits_;
    Eigen::MatrixXd cell_basis_;
    Eigen::RowVectorXd origin_;
    int cell_dofs_=0, atom_dofs_=0;
    double tolerance_;
public:
    Symmetry3D::symdataset initial;
    std::vector<int> common_operations, selected_operations;
    NEBPathSymmetry(const NEBDriver &, const std::string &selection);
    Eigen::RowVectorXd project(const Eigen::RowVectorXd &metric_vector) const;
    Eigen::RowVectorXd project_point(const Eigen::RowVectorXd &metric_point) const;
    int atomic_dofs() const { return atom_dofs_; }
    int cell_dofs() const { return cell_basis_.cols(); }
    void report(std::ostream &, const NEBDriver &, const std::string &stage) const;
};
