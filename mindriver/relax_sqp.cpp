#include "relax_new.h"
#include "constraint_expression.h"
#include <Eigen/SVD>
#include <exception>

namespace {
struct SQPContext {
    RelaxationDrvV2 &driver;
    std::vector<int> nonlinear;
    alglib::minnlcstate *solver;
    Eigen::VectorXd center;
    bool progress = false;
    double halfwidth = 0;
    std::exception_ptr callback_error;
};
struct OversizedTrial {};

Atoms decoded_geometry(RelaxationDrvV2 &d, const Eigen::VectorXd &point) {
    const auto saved_x = d.x, saved_g = d.g;
    const Atoms saved_atom = d.atom;
    const double saved_scaling = d.scaling;
    try {
        d.x = point;
        d.reverse_modifiers(d.modifiers_list);
        d.X2LatPos();
        Atoms result = d.atom;
        d.x = saved_x; d.g = saved_g; d.atom = saved_atom; d.scaling = saved_scaling;
        return result;
    } catch (...) {
        d.x = saved_x; d.g = saved_g; d.atom = saved_atom; d.scaling = saved_scaling;
        throw;
    }
}

void check_physical_step(RelaxationDrvV2 &d, const Eigen::VectorXd &point) {
    if (!d.has_checkpoint) return;
    // Decode both points on the same unwrapped coordinate branch. Neither
    // preview calls the potential or replaces the solver's requested point.
    const Atoms previous = decoded_geometry(d, d.checkpoint_x);
    const Atoms trial = decoded_geometry(d, point);
    double largest = 0;
    for (int i = 0; i < trial.size(); ++i)
        largest = std::max(largest, (trial.pos_[i]-previous.pos_[i]).norm());
    if (d.flags.relax_cell || d.flags.fix_shape)
        for (int i = 0; i < 3; ++i)
            largest = std::max(largest, (trial.cell_.row(i)-previous.cell_.row(i)).norm());
    if (!std::isfinite(largest) || largest > d.control.max_move*(1+1e-10)) throw OversizedTrial{};
}

double initial_active_radius(RelaxationDrvV2 &d) {
    double norm = 0.;
    if (!d.flags.relax_cell && !d.flags.fix_shape && !d.nonlinear_coordinates_active() && d.flags.relax_atom) {
        Eigen::MatrixXd jac = d.coordinate_jacobian();
        Eigen::Matrix3d atomic = Eigen::Matrix3d::Identity();
        if (d.subspace_active) atomic = d.subspace.atomic_jacobian(d.atom.cell_).transpose();
        else if (!d.flags.cartesian && !d.flags.atomic_force_recast) atomic = d.atom.cell_.transpose();
        for (int i = 0; i < d.atom.size(); ++i)
            jac.middleRows(3*i, 3) = (atomic*jac.middleRows(3*i, 3)).eval();
        if (!jac.allFinite()) throw std::runtime_error("SQP Cartesian coordinate Jacobian is non-finite");
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(jac);
        if (svd.singularValues().size()) norm = svd.singularValues()[0];
    }
    return d.control.max_move/(norm > 0 ? norm : 1.);
}
void sqp_jacobian(const alglib::real_1d_array &point, alglib::real_1d_array &values,
                  alglib::real_2d_array &jacobian, void *data) {
    auto &ctx = *static_cast<SQPContext *>(data);
    auto &d = ctx.driver;
    try {
    d.x = Eigen::Map<const Eigen::VectorXd>(point.getcontent(), point.length());
    check_physical_step(d, d.x);
    d.eval();
    values[0] = d.f;
    for (int j = 0; j < d.g.size(); ++j) jacobian[0][j] = d.g[j];
    for (size_t i = 0; i < ctx.nonlinear.size(); ++i) {
        const int id = ctx.nonlinear[i];
        values[i + 1] = d.constraints[id].f;
        for (int j = 0; j < d.g.size(); ++j)
            jacobian[i + 1][j] = d.constraint_jacobian(id, j);
    }
    } catch (...) {
        ctx.callback_error = std::current_exception();
        for (int i = 0; i < values.length(); ++i) {
            values[i] = 0.;
            for (int j = 0; j < point.length(); ++j) jacobian[i][j] = 0.;
        }
        alglib::minnlcrequesttermination(*ctx.solver);
    }
}

void sqp_report(const alglib::real_1d_array &point, double, void *data) {
    auto &ctx = *static_cast<SQPContext *>(data);
    auto &d = ctx.driver;
    try {
    d.x = Eigen::Map<const Eigen::VectorXd>(point.getcontent(), point.length());
    check_physical_step(d, d.x);
    d.eval();
    d.accept_checkpoint(!d.optimizer_initial_report);
    d.optimizer_initial_report = false;
    d.write_relax_log(d.f);
    if ((d.x-ctx.center).norm() > 1e-14) {
        ctx.progress = true;
        // Preserve interior BFGS learning (important for strongly scaled
        // coordinates), but carry boundary progress to a fresh local box.
        // A distant infeasible equality cannot be solved inside this box.
        if ((d.x-ctx.center).cwiseAbs().maxCoeff() >= .9*ctx.halfwidth)
            alglib::minnlcrequesttermination(*ctx.solver);
    }
    } catch (...) {
        ctx.callback_error = std::current_exception();
        alglib::minnlcrequesttermination(*ctx.solver);
    }
}
}

int RelaxationDrvV2::run_alg() {
    if (x.size() == 0 || control.max_iter == 0) {
        kkt_residual = g.size() ? g.cwiseAbs().maxCoeff() : 0;
        state_vars.converged = check_cons_convergence() && kkt_residual <= tol.gtol;
        return state_vars.optstate = state_vars.converged ? 0 : -1;
    }
    if (!std::isfinite(control.min_move) || control.min_move <= 0)
        throw std::invalid_argument("SQP requires a finite positive minimum step");
    double radius = initial_active_radius(*this);
    while (state_vars.curr_iter < control.max_iter) {
        if (radius < control.min_move) {
            stop_reason = "minimum_step";
            state_vars.converged = false;
            return state_vars.optstate = -1;
        }
        alglib::minnlcstate solver;
        alglib::real_1d_array point, lower, upper, scales, box_lower, box_upper, lin_lower, lin_upper;
        alglib::real_2d_array linear;
        SQPContext ctx{*this,{},&solver,x};
        const int iterations_before = state_vars.curr_iter;
        std::vector<int> affine;
        const double infinity = std::numeric_limits<double>::infinity();
        box_lower.setlength(x.size()); box_upper.setlength(x.size());
        for (int j=0;j<x.size();++j) {box_lower[j]=-infinity;box_upper[j]=infinity;}
        const auto bounds=[&](const regulator &r) {
            return std::pair{r.bounded?r.lower:(r.type==-1?-infinity:0.),
                             r.bounded?r.upper:(r.type==1?infinity:0.)};
        };
        for (size_t i=0;i<constraints.size();++i) {
            const auto &r=constraints[i];
            // Physical positions remain affine under fixed-cell LF/LC/masks.
            // Variable-cell products and nonlinear coordinate maps must stay NLCs.
            if (!flags.relax_cell && !flags.fix_shape && !nonlinear_coordinates_active() && r.expression && r.expression->affine_positions()) {
                const auto [lo,hi]=bounds(r);
                const auto row=constraint_jacobian.row(i);
                const double offset=r.f-row.dot(x);
                int component=-1, count=0;
                for (int j=0;j<x.size();++j) if (std::abs(row[j])>1e-14) {component=j;++count;}
                if (count==0) {
                    if(r.f<lo-r.tol || r.f>hi+r.tol) throw std::invalid_argument("Constraint is infeasible in the selected coordinate subspace");
                } else if (r.f < lo || r.f > hi) {
                    // Hard box/linear projection can jump to a distant feasible
                    // point before any step limit. Approach it as an NLC first.
                    ctx.nonlinear.push_back(i);
                } else if (count==1) {
                    double a=(lo-offset)/row[component], b=(hi-offset)/row[component];
                    if(a>b) std::swap(a,b);
                    box_lower[component]=std::max(box_lower[component],a);
                    box_upper[component]=std::min(box_upper[component],b);
                } else affine.push_back(i);
            } else ctx.nonlinear.push_back(i);
        }
        for(int j=0;j<x.size();++j) if(box_lower[j]>box_upper[j])
            throw std::invalid_argument("Position bounds have an empty intersection");
        const auto physical_lower = box_lower, physical_upper = box_upper;
        // This vendor's modern SQP does not consume the StpMax field. Local box
        // diameter, not that setter, enforces the active Euclidean radius.
        const double halfwidth = radius/(2*std::sqrt(static_cast<double>(x.size())));
        ctx.halfwidth = halfwidth;
        for (int j = 0; j < x.size(); ++j) {
            box_lower[j] = std::max(box_lower[j], x[j]-halfwidth);
            box_upper[j] = std::min(box_upper[j], x[j]+halfwidth);
        }
        point.setcontent(x.size(), x.data());
        alglib::minnlccreate(point, solver);
        alglib::minnlcsetalgosqp(solver);
        alglib::minnlcsetbc(solver,box_lower,box_upper);
        linear.setlength(affine.size(),x.size()); lin_lower.setlength(affine.size());lin_upper.setlength(affine.size());
        for(size_t i=0;i<affine.size();++i) {
            const int id=affine[i]; const auto row=constraint_jacobian.row(id);
            const double offset=constraints[id].f-row.dot(x);const auto [lo,hi]=bounds(constraints[id]);
            lin_lower[i]=lo-offset;lin_upper[i]=hi-offset;
            for(int j=0;j<x.size();++j)linear[i][j]=row[j];
        }
        if(!affine.empty()) alglib::minnlcsetlc2dense(solver,linear,lin_lower,lin_upper);
        lower.setlength(ctx.nonlinear.size()); upper.setlength(ctx.nonlinear.size());
        for (size_t i = 0; i < ctx.nonlinear.size(); ++i) {
            const auto [lo,hi]=bounds(constraints[ctx.nonlinear[i]]);
            lower[i]=lo;upper[i]=hi;
        }
        alglib::minnlcsetnlc2(solver, lower, upper);
        // Trust-radius termination is an internal solver condition. Physical
        // feasibility and the Lagrangian gradient are checked independently below.
        alglib::minnlcsetcond(solver, std::max(1e-12, std::min(1e-8, tol.gtol * .01)),
                            std::max(1, static_cast<int>(control.max_iter) - state_vars.curr_iter));
        scales.setlength(x.size());
        for (int i = 0; i < x.size(); ++i) scales[i] = 1;
        alglib::minnlcsetscale(solver, scales);
        alglib::minnlcsetstpmax(solver, radius);
        alglib::minnlcsetxrep(solver, true);
        optimizer_initial_report = true;
        try {
            try {
                alglib::minnlcoptimize(solver, sqp_jacobian, sqp_report, &ctx);
            } catch (...) {
                if (ctx.callback_error) std::rethrow_exception(ctx.callback_error);
                throw;
            }
            if (ctx.callback_error) std::rethrow_exception(ctx.callback_error);
        } catch (const OversizedTrial &) {
            restore_checkpoint();
            ++state_vars.curr_iter; // rejected previews share the total budget
            radius *= .5;
            continue;
        } catch (...) {
            restore_checkpoint();
            throw;
        }
        alglib::minnlcreport report;
        alglib::minnlcresults(solver, point, report);
        state_vars.curr_iter = std::min(static_cast<int>(control.max_iter),
            std::max(state_vars.curr_iter, iterations_before+static_cast<int>(report.iterationscount)));
        // Results are the solver's accepted point, never its last line-search trial.
        x = Eigen::Map<const Eigen::VectorXd>(point.getcontent(), point.length());
        eval(); accept_checkpoint(false);
        Eigen::VectorXd lagrangian = g;
        bool dual_ok=true;
        const auto dual=[&](double value,double lo,double hi,double multiplier,double tolerance) {
            if(lo==hi) return;
            if(multiplier>tol.gtol && (!std::isfinite(hi) || std::abs(value-hi)>tolerance)) dual_ok=false;
            if(multiplier<-tol.gtol && (!std::isfinite(lo) || std::abs(value-lo)>tolerance)) dual_ok=false;
        };
        for(int j=0;j<x.size();++j) {
            const double multiplier = report.lagbcnz[j];
            const bool real_bound = multiplier >= 0 ? box_upper[j] == physical_upper[j]
                                                    : box_lower[j] == physical_lower[j];
            // Artificial trust-bound multipliers cannot certify stationarity of
            // the physical problem. Original bounds retain their own multipliers.
            if (real_bound) {
                lagrangian[j] += multiplier;
                dual(x[j],physical_lower[j],physical_upper[j],multiplier,std::max(tol.gtol,1e-8));
            }
        }
        for(size_t i=0;i<affine.size();++i) {
            const int id=affine[i]; const auto [lo,hi]=bounds(constraints[id]);
            lagrangian+=report.laglc[i]*constraint_jacobian.row(id).transpose();
            dual(constraints[id].f,lo,hi,report.laglc[i],constraints[id].tol);
        }
        for(size_t i=0;i<ctx.nonlinear.size();++i) {
            const int id=ctx.nonlinear[i];const auto [lo,hi]=bounds(constraints[id]);
            lagrangian+=report.lagnlc[i]*constraint_jacobian.row(id).transpose();
            dual(constraints[id].f,lo,hi,report.lagnlc[i],constraints[id].tol);
        }
        kkt_residual = unwhiten_gradient(lagrangian).cwiseAbs().maxCoeff();
        state_vars.converged = report.terminationtype > 0 && dual_ok && check_cons_convergence() &&
            std::isfinite(kkt_residual) && kkt_residual <= tol.gtol;
        if (flags.write_traj && !iobj.seq_write(ofs, atom))
            GPERROR("Failed to write relaxation trajectory: " + control.trajfile);
        if (state_vars.converged) return state_vars.optstate = 0;
        if (!ctx.progress) {
            // No accepted move: shrink without retrying an identical subproblem
            // forever. Count this failed segment if ALGLIB reported no iteration.
            if (state_vars.curr_iter == iterations_before) ++state_vars.curr_iter;
            radius *= .5;
        }
    }
    state_vars.converged = false;
    return state_vars.optstate = -1;
}
