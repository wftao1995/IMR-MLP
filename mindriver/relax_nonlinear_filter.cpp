#include "relax_new.h"

void FilterNF::init(RelaxationDrvV2 *d,void *) {
    if(!map)throw std::invalid_argument("Nonlinear coordinate map is missing");
    map->initialize(d->x);point=map->reduce(d->x);
    const auto full=map->lift(point);const auto jac=map->jacobian(point);
    if(!point.allFinite() || full.size()!=d->x.size() || !full.isApprox(d->x,1e-7) ||
        jac.rows()!=d->x.size() || jac.cols()!=point.size() || !jac.allFinite())
        throw std::invalid_argument("Nonlinear coordinate map does not reconstruct its initial frame");
}
void FilterNF::forward(RelaxationDrvV2 *d) {
    point=map->reduce(d->x);const auto jac=map->jacobian(point);
    if(!point.allFinite() || jac.rows()!=d->g.size() || jac.cols()!=point.size() || !jac.allFinite())
        throw std::runtime_error("Nonlinear coordinate Jacobian has invalid dimensions or values");
    full_gradient=d->g;d->g=jac.transpose()*full_gradient;d->x=point;
}
void FilterNF::reverse(RelaxationDrvV2 *d) {
    point=d->x;d->x=map->lift(point);d->g.resize(d->x.size());
}
void FilterNF::forward_scaling(RelaxationDrvV2 *d) {
    const auto jac=map->jacobian(point);
    if(jac.rows()!=d->prec_scaling.size())throw std::invalid_argument("Nonlinear scaling dimensions do not match");
    d->prec_scaling=(jac.array().square().matrix().transpose()*d->prec_scaling).cwiseMax(1e-12);
}

Eigen::MatrixXd RelaxationDrvV2::coordinate_jacobian() {
    const auto saved_x=x,saved_g=g;
    reverse_modifiers(modifiers_list);const auto full=x;
    Eigen::MatrixXd jac(full.size(),saved_x.size());
    for(int i=0;i<full.size();++i) {
        x=full;g=Eigen::VectorXd::Unit(full.size(),i);apply_modifiers(modifiers_list);
        if(g.size()!=saved_x.size())throw std::runtime_error("Coordinate map dimension changed during Jacobian pullback");
        jac.row(i)=g.transpose();
    }
    x=saved_x;g=saved_g;return jac;
}
