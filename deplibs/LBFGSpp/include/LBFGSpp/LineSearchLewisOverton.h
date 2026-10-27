// Copyright (C) 2016-2022 Yixuan Qiu <yixuan.qiu@cos.name>
// Copyright (C) 2016-2022 Dirk Toewe <DirkToewe@GoogleMail.com>
// Under MIT license

#pragma once

#include <Eigen/Core>
#include <stdexcept>

namespace LBFGSpp {

///
/// A line search algorithm for the strong Wolfe condition. Implementation based on:
///
///   "Numerical Optimization" 2nd Edition,
///   Jorge Nocedal Stephen J. Wright,
///   Chapter 3. Line Search Methods, page 60f.
///
template <typename Scalar>
class LineSearchLewisOverton
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
        // Check the value of step
        if (step <= Scalar(0))
            throw std::invalid_argument("'step' must be positive");

        if (param.linesearch != LBFGS_LINESEARCH_LEWIS_OVERTON)
            throw std::invalid_argument("'param.linesearch' must be 'LBFGS_LINESEARCH_LEWIS_OVERTON' for LineSearchLewisOverton");

        int count = 0;
        bool brackt = false;
        // translation: s -> drt

        // The expansion rate of the step size
        const Scalar expansion = Scalar(2);

        // Save the function value at the current x
        const Scalar fx_init = fx;
        // Projection of gradient on the search direction
        const Scalar dg_init = grad.dot(drt);
        // Make sure d points to a descent direction
        if (dg_init > 0)
            throw std::logic_error("the moving direction increases the objective function value");

        Scalar dgtest = param.f_dec_coeff * dg_init; // decrease
        Scalar dstest = param.s_curv_coeff * dg_init; // curvature

        // change max_step according to max_move
        Scalar drmax = drt.cwiseAbs().maxCoeff();
        Scalar cur_max_step = step_max;
        if (cur_max_step * drmax > param.max_step_move) {
            cur_max_step = param.max_step_move / drmax;
        }
        step = step > cur_max_step ? cur_max_step : step;

        Scalar mu = 0.0, nu = cur_max_step;

        for (;;) {
            x = xp + step * drt;
            
            // Evaluate function value and its gradient
            fx = f(x, grad);
            ++count;

            if (std::isinf(fx) || std::isnan(fx)) {
                throw std::logic_error("inf or nan");
            }

            // Armijo condition
            if (fx > fx_init + step * dgtest) {
                nu = step;
                brackt = true;
            } else {
                // A decreasing boundary step is valid even if curvature
                // cannot be satisfied within the displacement limit.
                if (step >= cur_max_step) return;
                // weak wolfe condition
                if (grad.dot(drt) < dstest) {
                    mu = step;
                } else {
                    return;
                }
            }
            // check max number of iteration, should return a error
            if (count >= param.max_linesearch) {
                throw std::logic_error("exceed max linesearch");
            }
            if (brackt && (nu - mu) < param.machine_prec * nu) {
                // relative interval width is at least machine_prec, should return a error code. TODO
                throw std::logic_error("width too small");
            }

            if (brackt) {
                step = 0.5 * (mu + nu);
            } else {
                step *= 2.0;
            }

            if (step < param.min_step) {
                throw std::logic_error("linesearch step below minimum allowed value");
            } 
            step = std::min(step, cur_max_step);
        }

    }

};

}  // namespace LBFGSpp
