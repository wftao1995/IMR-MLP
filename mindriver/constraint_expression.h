#pragma once
#include "base/structures.h"
#include <memory>

namespace Constraints {
// Parsed once, with physical geometry images and a continuous dihedral branch
// bound to the initial frame. Values use A, degrees, and A^3; sin/cos use radians.
class Expression {
    struct Impl;
    std::shared_ptr<const Impl> impl_;
public:
    struct Value {
        double value = 0;
        std::vector<Eigen::RowVector3d> positions;
        Eigen::Matrix3d cell = Eigen::Matrix3d::Zero();
    };
    Expression(const std::string &, const Atoms &);
    Value evaluate(const Atoms &) const;
    bool affine_positions() const;
    bool uses_volume() const;
};
}
