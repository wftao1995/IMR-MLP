/**
 * By wftao (Jiantao Wang).
 * For gradient only cases, i.e. NEB optimization
*/

// Copyright (C) 2023-2024 Jiantao Wang <wftao1995@gmail.com>
// Under MIT license

#pragma once

#include <Eigen/Core>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <limits>

namespace LBFGSpp {

///
/// A line search algorithm for graident only
///
///   search steps that change the sign of
///   F'(\lambda) = [\nambda^T f(x) + \lambda u] \cdot u
///   minimize F aka find F' = 0
///
template <typename Scalar>
class LineSearchForceOnlyA
{
private:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> Vector;

public:
    ///
    /// Line search by Nocedal and Wright (2006).
    ///
    /// \param f        A function object such that `f(x, grad)` returns the
    ///                 objective function value at `x`, and overwrites `grad` with
    ///                 the gradient.
    /// \param param    Parameters for the L-BFGS algorithm.
    /// \param xp       The current point.
    /// \param drt      The current moving direction.
    /// \param step_max The upper bound for the step size that makes x feasible.
    ///                 Can be ignored for the L-BFGS solver.
    /// \param step     In: The initial step length.
    ///                 Out: The calculated step length.
    /// \param fx       In: The objective function value at the current point.
    ///                 Out: The function value at the new point.
    /// \param grad     In: The current gradient vector.
    ///                 Out: The gradient at the new point.
    /// \param dg       In: The inner product between drt and grad.
    ///                 Out: The inner product between drt and the new gradient.
    /// \param x        Out: The new point moved to.
    ///
    template <typename Foo>
    static void LineSearch(Foo& f, const LBFGSParam<Scalar>& param,
                           const Vector& xp, const Vector& drt, const Scalar& step_max,
                           Scalar& step, Scalar& fx, Vector& grad, Scalar& dg, Vector& x)
    {
        if (!std::isfinite(step) || step <= Scalar(0))
            throw std::invalid_argument("'step' must be finite and positive");
        if (param.linesearch != LBFGS_LINESEARCH_FORCE_ONLY_A)
            throw std::invalid_argument("force-only line search requires LBFGS_LINESEARCH_FORCE_ONLY_A");
        if (!std::isfinite(param.damping) || param.damping < 0 || param.damping > 1)
            throw std::invalid_argument("force-only damping must lie in [0, 1]");
        const Scalar dg_init = grad.dot(drt);
        if (!std::isfinite(dg_init) || dg_init >= 0)
            throw std::logic_error("force-only search requires a finite descent direction");
        const Scalar maximum = std::min(step_max, param.max_step_move / drt.cwiseAbs().maxCoeff());
        step = std::min(step, maximum);
        const Scalar threshold = -param.s_curv_coeff * dg_init;
        Scalar lower = 0, upper = std::numeric_limits<Scalar>::infinity();
        for (int iteration = 0; iteration < param.max_linesearch; ++iteration) {
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            dg = grad.dot(drt);
            if (!grad.allFinite() || !std::isfinite(dg))
                throw std::runtime_error("non-finite gradient in force-only line search");
            if (std::abs(dg) <= threshold || (dg < 0 && step >= maximum)) break;
            if (dg > 0) upper = step;
            else lower = step;
            const Scalar next = std::isfinite(upper) ? (lower + upper) * Scalar(.5) :
                std::min(maximum, step * Scalar(2));
            // The returned step, x, gradient and directional derivative must all
            // describe the last evaluated point, also at a trial budget limit.
            if (iteration + 1 == param.max_linesearch || next == step) break;
            step = next;
        }
        if (param.damping > 0 && param.damping < 1) {
            step *= param.damping;
            x.noalias() = xp + step * drt;
            fx = f(x, grad);
            dg = grad.dot(drt);
            if (!grad.allFinite() || !std::isfinite(dg))
                throw std::runtime_error("non-finite gradient in damped force-only line search");
        }
    }
};

}  // namespace LBFGSpp
