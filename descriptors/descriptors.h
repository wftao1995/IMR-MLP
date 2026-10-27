// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_DESCRIPTORS_H
#define IMRMLP_DESCRIPTORS_H

#include "base/neighborhood.h"
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <stdexcept>

/// \brief Descriptor position gradient: d(descriptor) / d(atomic positions).
///
/// Shape: (feature_size, n_neighbors, 3), RowMajor layout.
///   grad(f, j, a) = d(desc_f) / d(pos_{j,a})
struct DescriptorGradient
{
    Eigen::Tensor<double, 3, Eigen::RowMajor> grad;
};

using DescriptorSiteMatrix = Eigen::Matrix<double, -1, 3, Eigen::RowMajor>;

/// \brief Options controlling descriptor evaluation.
///
/// Pass to compute() overloads that accept it. Not all descriptors support
/// every option — unsupported flags are silently ignored (descriptor
/// implementations document their supported set).
struct ComputeOptions
{
    /// Whether to compute the descriptor derivative w.r.t. atomic positions.
    bool compute_derivative = false;

    /// Optional output: when compute_derivative is true and this is non-null,
    /// the descriptor resizes this tensor and fills it with position gradients.
    DescriptorGradient *gradient = nullptr;
};

/// \brief Abstract base for all atomic-environment descriptors.
/// \ingroup descriptors
///
/// Each subclass implements a specific descriptor family (ACSF,
/// bond-order parameters, etc.).  The public contract is:
///
///   1. Call \ref compute(Neighborhood&, Eigen::VectorXd&) to obtain the
///      descriptor vector for a single atomic environment.
///   2. Use \ref feature_size() to query the vector length.
///
/// For performance-critical code the virtual dispatch overhead can be avoided
/// by calling the concrete class methods directly.
class Descriptors
{
public:
    double cutoff_;

    Descriptors() = default;
    virtual ~Descriptors() = default;

    /// Compute descriptors for a single atomic neighborhood.
    /// @param nbh  Neighbor list for the center atom.
    ///              The list must satisfy \c nbh.distances_[i] <= cutoff_.
    /// @param out  Output descriptor vector. The implementation resizes it
    ///              to feature_size() and fills it with descriptor values.
    virtual void compute(Neighborhood &nbh, Eigen::VectorXd &out) = 0;

    /// Compute descriptors with additional options (derivatives).
    /// The default implementation discards \p opts and calls the two-argument
    /// overload.  Subclasses that support options override this method.
    virtual void compute(Neighborhood &nbh, Eigen::VectorXd &out,
                         const ComputeOptions & /*opts*/)
    {
        compute(nbh, out);
    }

    /// Dimension of the descriptor vector produced by compute().
    virtual int feature_size() const = 0;

    // Opt-in contractions of fixed feature weights with local coordinate
    // derivatives. Rows refer to independent relative neighbor coordinates;
    // the Hessian uses flattened neighbor-major xyz indexing. Unsupported
    // implementations throw without changing outputs. No feature Hessian
    // tensor is part of this interface.
    virtual bool supports_weighted_site_hvp() const { return false; }
    virtual bool supports_weighted_site_hessian() const { return false; }
    virtual void compute_weighted_site_hvp(Neighborhood &, const Eigen::VectorXd &,
        const DescriptorSiteMatrix &, double &, DescriptorSiteMatrix &, DescriptorSiteMatrix &) {
        throw std::runtime_error("Descriptor weighted site HVP is not supported");
    }
    virtual void compute_weighted_site_hessian(Neighborhood &, const Eigen::VectorXd &,
        double &, DescriptorSiteMatrix &, Eigen::MatrixXd &) {
        throw std::runtime_error("Descriptor weighted site Hessian is not supported");
    }

    /// Create an independent copy for thread-safe parallel use.
    /// Returns nullptr if this descriptor type cannot be cloned.
    virtual std::unique_ptr<Descriptors> clone() const = 0;
};

#endif // IMRMLP_DESCRIPTORS_H
