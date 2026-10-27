// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "varpro.h"
#include "conditional_physical_state.h"
#include "collective_error.h"
#include "potentials/potential_mtp.h"
#include <typeinfo>

VarProRadialScaling::VarProRadialScaling(Eigen::VectorXd scales):scales_(std::move(scales)) {
    if(!scales_.allFinite() || (scales_.array()<=0.).any())
        GPERROR("VP radial scales must be finite and positive");
}
void VarProRadialScaling::check(const Eigen::VectorXd& v) const {
    if(v.size()!=scales_.size() || !v.allFinite()) GPERROR("Invalid VP scaled coordinate vector");
}
Eigen::VectorXd VarProRadialScaling::physical(const Eigen::VectorXd& z) const {
    check(z);Eigen::VectorXd r=z.cwiseProduct(scales_);check(r);return r;
}
Eigen::VectorXd VarProRadialScaling::optimizer(const Eigen::VectorXd& radial) const {
    check(radial);Eigen::VectorXd z=radial.cwiseQuotient(scales_);check(z);return z;
}
Eigen::VectorXd VarProRadialScaling::gradient(const Eigen::VectorXd& physical_gradient) const {
    return physical(physical_gradient); // chain rule: grad_z = S^T grad_r
}

MTPProfiledObjective::MTPProfiledObjective(Potentials::AnyPotential& potential,
    std::vector<Atoms>& reference, std::vector<Neighborhoods>& neighborhoods,
    const EFSFitWeights& weights, double penalty_weight,
    LinearRefitSettings settings, Gradient gradient, double linear_gradient_tolerance, int linear_refinement_steps)
    : potential_(potential), reference_(reference), neighborhoods_(neighborhoods),
      weights_(weights), penalty_weight_(penalty_weight),
      linear_gradient_tolerance_(linear_gradient_tolerance), settings_(std::move(settings)),
      gradient_(std::move(gradient)), linear_refinement_steps_(linear_refinement_steps) {
    if (auto *mtp=dynamic_cast<Potentials::PotMTP*>(&potential); mtp && mtp->radial_mode!=0)
        GPERROR("VP currently requires radial_mode=0 for MTP models");
    if (!std::isfinite(linear_gradient_tolerance_) || linear_gradient_tolerance_<=0.)
        GPERROR("VP linear gradient tolerance must be finite and positive");
    if (linear_refinement_steps_<0) GPERROR("VP linear refinement steps must be nonnegative");
    if (settings_.solver!="qr_svd" || !std::isfinite(settings_.rtol) || settings_.rtol<0. || settings_.rtol>=1.
        || !std::isfinite(settings_.atol) || settings_.atol<0.
        || !std::isfinite(settings_.acceptance_tolerance) || settings_.acceptance_tolerance<=0.)
        GPERROR("Invalid VP linear solver or tolerances");
    if (!gradient_) GPERROR("VP requires a gradient evaluator");
    if (settings_.mode=="off") {
        const auto available=potential.conditional_linear_blocks(penalty_weight_);
        settings_.mode="all";
        for(const auto &block:available) if(block.name=="output" || block.name=="linear") settings_.mode="head";
        for(const auto &block:available) if(block.name=="sr") settings_.mode="sr";
    }
    Training::ConditionalLinearAdapter selected(potential,
        Training::conditional_linear_selection(potential,settings_.mode),penalty_weight_);
    if((selected.l1.array()!=0.).any()) GPERROR("VP does not support nonzero selected L1 regularization");
    if(!selected.l2.allFinite() || selected.l2.minCoeff()<0. || !std::isfinite(penalty_weight_) || penalty_weight_<0.)
        GPERROR("Invalid VP quadratic regularization");
    inner_indices_=selected.indices;inner_l2_=selected.l2;
    for(int i=0;i<potential.get_NL_coeffs_count();++i) if(!selected.selected(i)) outer_indices_.push_back(i);
    radial_size_=outer_indices_.size();
    settings_.report.clear(); // per-evaluation diagnostics belong to the VP report
    source_.resize(potential.get_NL_coeffs_count());potential.nl_get_coeffs(source_.data());
    if (!source_.allFinite()) GPERROR("VP requires finite initial coefficients");
}

Eigen::VectorXd MTPProfiledObjective::outer_coefficients(const Eigen::VectorXd& full) const {
    if(full.size()!=source_.size()) GPERROR("Invalid VP full parameter layout");
    Eigen::VectorXd outer(radial_size_);
    for(int i=0;i<radial_size_;++i) outer[i]=full[outer_indices_[i]];
    return outer;
}

VarProEvaluation MTPProfiledObjective::evaluate(const Eigen::VectorXd& radial) {
    Training::collective_try([&] {
        if(potential_.get_NL_coeffs_count()!=source_.size())
            GPERROR("Potential parameter layout changed; reconstruct the profiled objective");
        if (radial.size()!=radial_size_ || !radial.allFinite()) GPERROR("Invalid VP outer trial");
    });
    auto restore=Training::conditional_physical_restore(potential_);
    try {
        auto trial=source_.eval();
        for(int i=0;i<radial_size_;++i) trial[outer_indices_[i]]=radial[i];
        Training::collective_try([&] {potential_.nl_set_coeffs(trial.data());});
        VarProEvaluation result;
        Eigen::VectorXd full_gradient;
        for(int refinement=0;;++refinement) {
            auto fit=linear_refit(potential_, reference_, neighborhoods_, weights_,
                                 penalty_weight_, settings_);
            if(refinement>0) {
                fit.before=result.linear.before;
                fit.data_before=result.linear.data_before;
                fit.penalty_before=result.linear.penalty_before;
                fit.delta+=result.linear.delta;
                fit.assembly_seconds+=result.linear.assembly_seconds;
                fit.solve_seconds+=result.linear.solve_seconds;
                fit.verification_seconds+=result.linear.verification_seconds;
            }
            result.linear=std::move(fit);
            full_gradient=Eigen::VectorXd::Zero(source_.size());
            // The callback must synchronize any errors before its own collectives.
            Training::collective_try([&] {gradient_(result.objective, full_gradient);});
            Training::collective_try([&] {
                if (full_gradient.size()!=source_.size() || !full_gradient.allFinite()
                    || !std::isfinite(result.objective)) GPERROR("Nonfinite VP objective or gradient");
                if (std::abs(result.objective-result.linear.after)>1e-9*std::max(1.,std::abs(result.objective)))
                    GPERROR("VP linear refit and gradient objectives disagree");
            });
            result.linear_gradient_inf=0.;
            for(int index:inner_indices_) result.linear_gradient_inf=std::max(result.linear_gradient_inf,std::abs(full_gradient[index]));
#ifdef IMRMLP_MPI
            // All ranks must take the same retry branch, even for rank-local callbacks.
            MPI_Allreduce(MPI_IN_PLACE,&result.linear_gradient_inf,1,MPI_DOUBLE,MPI_MAX,mpi.comm);
#endif
            result.linear_gradient_history.push_back(result.linear_gradient_inf);
            if(result.linear_gradient_inf<=linear_gradient_tolerance_) break;
            if(refinement==linear_refinement_steps_)
                GPERROR(fmt::format("VP linear coefficients are not stationary enough for the profiled gradient: "
                    "gradient_inf={:.17g}, tolerance={:.17g}, refinements={}",
                    result.linear_gradient_inf,linear_gradient_tolerance_,refinement));
            // Refit the residual around the current linear solution at fixed radial
            // parameters. The linear solver preserves composition-null directions.
            if(mpi.rank==0) Message(fmt::format("VP linear refinement {}: gradient_inf={:.17g}, tolerance={:.17g}",
                refinement+1,result.linear_gradient_inf,linear_gradient_tolerance_));
        }
        Training::collective_try([&] {
        result.gradient=outer_coefficients(full_gradient);
        result.coefficients.resize(source_.size());potential_.nl_get_coeffs(result.coefficients.data());
        result.linear_penalty=0.;
        for(int j=0;j<int(inner_indices_.size());++j)
            result.linear_penalty+=inner_l2_[j]*std::pow(result.coefficients[inner_indices_[j]],2);
        double full_penalty=0.;full_gradient.setZero();
        potential_.AddPenaltyGard(penalty_weight_,full_penalty,full_gradient.data());
        result.radial_penalty=full_penalty-result.linear_penalty;
        // Retain the ordinary-MTP radial diagnostic. Other models report the
        // actual outer index list instead of inventing a radial interpretation.
        if(typeid(potential_)==typeid(Potentials::PotMTP)) {
            auto &p=static_cast<Potentials::PotMTP&>(potential_);
            result.radial_norms=Eigen::VectorXd::Zero(p.radial_func_count);
            for(int pair=0;pair<p.species_count*p.species_count;++pair)
                for(int k=0;k<p.radial_func_count;++k)
                    result.radial_norms[k]+=Eigen::Map<const Eigen::VectorXd>(p.radial_coeffs+
                        (pair*p.radial_func_count+k)*p.rb_size,p.rb_size).squaredNorm();
            result.radial_norms=result.radial_norms.cwiseSqrt();
        }
        });
        restore();
        return result;
    } catch (...) {
        restore();
        throw;
    }
}
