#include "relax_new.h"

void FilterWhitening::forward(RelaxationDrvV2 *d) {
    point=d->x;d->x=transform.reduce(point);d->g=transform.gradient(d->g);
}
void FilterWhitening::reverse(RelaxationDrvV2 *d) {d->x=transform.lift(d->x);d->g.resize(d->x.size());}
void FilterWhitening::forward_scaling(RelaxationDrvV2 *d) {d->prec_scaling.setOnes(point.size());}

namespace {
FilterWhitening *whitening(const RelaxationDrvV2 &d) {
    for(auto *m:d.modifiers_list)if(auto *w=dynamic_cast<FilterWhitening *>(m))return w;
    return nullptr;
}
Eigen::MatrixXd cartesian_to_full(const RelaxationDrvV2 &d) {
    Eigen::MatrixXd j=Eigen::MatrixXd::Zero(3*d.atom.size(),3*d.atom.size());
    Eigen::Matrix3d block=Eigen::Matrix3d::Identity();
    if(d.subspace_active)block=d.subspace.atomic_jacobian(d.atom.cell_).transpose();
    else if(!d.flags.cartesian && !d.flags.atomic_force_recast)block=d.atom.cell_.transpose();
    for(int i=0;i<d.atom.size();++i)j.block<3,3>(3*i,3*i)=block;
    return j;
}
}

void RelaxationDrvV2::initialize_metrics() {
    fc_refresh_count=0;
    sparse_metric.reset();full_hessian.resize(0,0);reduced_hessian.resize(0,0);
    fc_update.configure(control.fc_update_interval);
    if(!control.hessian_precond_file.empty())
        full_hessian=RelaxPreconditioning::read_full_hessian(control.hessian_precond_file,x.size());
    else if(control.fc_update_interval) {
        if(!flags.relax_atom)throw std::invalid_argument("Online FC requires atomic degrees of freedom");
        full_hessian=RelaxPreconditioning::pullback(RelaxPreconditioning::native_cartesian_hessian(*pot,atom),cartesian_to_full(*this));
        ++fc_refresh_count;
    }
    if(full_hessian.size()==0 && !control.sparse_precond)return;
    const auto initial_full=x,initial_g=g;
    apply_modifiers(modifiers_list);
    if(control.sparse_precond && x.size()) {
        std::vector<int> ids;
        for(auto *m:modifiers_list)if(auto *selection=dynamic_cast<FilterSelDyn *>(m))ids=selection->ids_pass;
        RelaxPreconditioning::SparseOptions options;options.shift=control.sparse_shift;
        sparse_metric=std::make_unique<RelaxPreconditioning::SparseMetric>(atom,pot->cutoff,options,ids);
        prec_scaling.setOnes(x.size());
    }
    if(full_hessian.size() && x.size()) {
        const auto jac=coordinate_jacobian();
        reduced_hessian=RelaxPreconditioning::pullback(full_hessian,jac);
        auto *filter=new FilterWhitening();
        try {filter->transform.reset(reduced_hessian,x,control.precond_floor);}
        catch(...) {delete filter;throw;}
        filter->point=x;filter->forward_scaling(this);modifiers_list.push_back(filter);
    }
    x=initial_full;g=initial_g;
}

Eigen::VectorXd RelaxationDrvV2::unwhiten_gradient(const Eigen::VectorXd &gradient) const {
    if(auto *w=whitening(*this))return w->transform.metric().root()*gradient;
    return gradient;
}
double RelaxationDrvV2::optimizer_gtol() const {
    double threshold=tol.gtol*(prec_scaling.size()?std::min(1.,prec_scaling.minCoeff()):1.);
    if(auto *w=whitening(*this))threshold/=std::max(1.,w->transform.metric().root().norm());
    return threshold;
}
Eigen::VectorXd RelaxationDrvV2::preconditioned_gradient(const Eigen::VectorXd &gradient) const {
    if(sparse_metric)return sparse_metric->solve(gradient);
    return gradient.cwiseQuotient(prec_scaling);
}
double RelaxationDrvV2::metric_inner(const Eigen::VectorXd &a,const Eigen::VectorXd &b) const {
    if(sparse_metric)return a.dot(sparse_metric->matrix()*b);
    return a.dot(prec_scaling.cwiseProduct(b));
}

void RelaxationDrvV2::refresh_fc_metric() {
    if(!fc_update.due(accepted_steps))return;
    auto *white=whitening(*this);
    if(!white || modifiers_list.empty() || modifiers_list.back()!=white)
        throw std::logic_error("Online FC requires a final whitening filter");
    // Neither the update controller nor the old metric is committed until all
    // native curvature, pullbacks and coordinate transformations have succeeded.
    auto candidate_update=fc_update;
    const auto native=candidate_update.compute(*pot,atom,accepted_steps);
    if(!native)return;
    const auto saved_x=x,saved_g=g;
    Eigen::VectorXd point=white->transform.lift(saved_x);
    const Eigen::VectorXd gradient=white->transform.metric().root()*saved_g;
    if(!gradient.allFinite())throw std::runtime_error("Non-finite unwhitened FC gradient");
    modifiers_list.pop_back();
    try {
        x=point;g=gradient;
        const auto jac=coordinate_jacobian();
        Eigen::MatrixXd candidate_full=RelaxPreconditioning::pullback(*native,cartesian_to_full(*this));
        Eigen::MatrixXd candidate_reduced=RelaxPreconditioning::pullback(candidate_full,jac);
        RelaxPreconditioning::WhiteningTransform candidate_transform(candidate_reduced,point,control.precond_floor);
        Eigen::VectorXd candidate_g=candidate_transform.gradient(gradient);
        Eigen::VectorXd candidate_x=Eigen::VectorXd::Zero(point.size());
        // pop_back retained capacity, so restoring this pointer cannot allocate.
        modifiers_list.push_back(white);
        white->transform=std::move(candidate_transform);white->point.swap(point);
        full_hessian.swap(candidate_full);reduced_hessian.swap(candidate_reduced);
        x.swap(candidate_x);g.swap(candidate_g);
        fc_update=candidate_update;++fc_refresh_count;
    } catch(...) {
        if(modifiers_list.empty() || modifiers_list.back()!=white)modifiers_list.push_back(white);
        x=saved_x;g=saved_g;throw;
    }
    // Atom/E/F/S and the objective already belong to this accepted point. Only
    // their coordinate representation changes; another potential call is wrong.
    state_flags.request_alg_restart=true;
}

Eigen::MatrixXd RelaxationDrvV2::constraint_derivatives() {
    const auto saved_x=x,saved_g=g;
    struct SavedNF {FilterNF *filter;Eigen::VectorXd point,gradient;};
    std::vector<SavedNF> saved_nf;
    for(auto *m:modifiers_list)if(auto *nf=dynamic_cast<FilterNF *>(m))saved_nf.push_back({nf,nf->point,nf->full_gradient});
    const auto restore=[&] {
        x=saved_x;g=saved_g;
        for(auto &state:saved_nf){state.filter->point=state.point;state.filter->full_gradient=state.gradient;}
    };
    try {
        reverse_modifiers(modifiers_list);const auto full=x;
        Eigen::MatrixXd derivatives(constraints.size(),saved_x.size());
        // Only pull back the requested rows. A handful of constraints on a
        // large sparse-preconditioned system must not allocate a 3N-by-3N map.
        for(size_t i=0;i<constraints.size();++i) {
            x=full;g=Eigen::VectorXd::Zero(full.size());
            for(size_t j=0;j<constraints[i].outputs.size();++j)g[constraints[i].outputs[j]]+=constraints[i].g[j];
            apply_modifiers(modifiers_list);derivatives.row(i)=g.transpose();
        }
        restore();return derivatives;
    } catch(...) {restore();throw;}
}

void RelaxationDrvV2::initialize_aul_parameters() {
    const auto jac=constraint_derivatives();Eigen::VectorXd residual(constraints.size());
    Eigen::VectorXd objective=g;
    for(size_t i=0;i<constraints.size();++i) {
        const auto &r=constraints[i];
        if(r.type!=10 || r.bounded)throw std::invalid_argument("Automatic AUL currently supports equality regulators");
        residual[i]=r.f;objective-=(r.weight*r.f+r.lambda)*jac.row(i).transpose();
    }
    const auto estimate=RelaxPreconditioning::estimate_aul(jac,residual,[&](const Eigen::VectorXd &v){return preconditioned_gradient(v);});
    aul_previous_residual=residual.cwiseAbs();
    for(size_t i=0;i<constraints.size();++i) {
        auto &r=constraints[i];r.weight=estimate.weights[i]*std::max(1.,objective.norm());
        r.lambda=0;r.mos=r.mos2=0;
    }
    eval();accept_checkpoint(false);
}
