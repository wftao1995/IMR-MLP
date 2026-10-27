#include "potential_descriptor_linear.h"

#include <Eigen/LU>

using namespace Potentials;

bool PotentialDescriptorLinear::supports_site_hvp() const {
    return desc_wrapper_->desc->supports_weighted_site_hvp();
}
bool PotentialDescriptorLinear::supports_site_hessian() const {
    return desc_wrapper_->desc->supports_weighted_site_hessian();
}

void PotentialDescriptorLinear::CalcSiteEFS(Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) {
    ComputeOptions options;
    options.compute_derivative = true;
    options.gradient = &grad_buf_;
    desc_wrapper_->compute(nbh, desc_buf_, options);
    energy = coeffs_.dot(desc_buf_);
    gradient.setZero(nbh.nbhcount_, 3);
    for (int f = 0; f < n_features_; ++f)
        for (int j = 0; j < nbh.nbhcount_; ++j)
            for (int a = 0; a < 3; ++a) gradient(j, a) += coeffs_[f] * grad_buf_.grad(f, j, a);
}

void PotentialDescriptorLinear::apply_site_hvp(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
    double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix &product) {
    if (!supports_site_hvp()) AnyPotential::apply_site_hvp(nbh, direction, energy, gradient, product);
    desc_wrapper_->desc->compute_weighted_site_hvp(nbh, coeffs_, direction, energy, gradient, product);
}

void PotentialDescriptorLinear::evaluate_site_hessian(Neighborhood &nbh, double &energy,
    SiteCoordinateMatrix &gradient, HessianMatrix &hessian) {
    if (!supports_site_hessian()) AnyPotential::evaluate_site_hessian(nbh, energy, gradient, hessian);
    Eigen::MatrixXd local;
    desc_wrapper_->desc->compute_weighted_site_hessian(nbh, coeffs_, energy, gradient, local);
    hessian.resize(nbh.nbhcount_);
    for (int i = 0; i < nbh.nbhcount_; ++i)
        for (int j = 0; j < nbh.nbhcount_; ++j) hessian(i, j) = local.block<3, 3>(3*i, 3*j);
}

void PotentialDescriptorLinear::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    if (!supports_site_hvp()) GPERROR("Descriptor potential does not support analytical HVP");
    if (nbhs.size() != atom.size() || direction.size() != 3*atom.size() || !direction.allFinite())
        GPERROR("Invalid descriptor potential HVP direction or neighborhoods");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    SiteCoordinateMatrix local_direction, local_gradient, local_product;
    for (int center = 0; center < nbhs.size(); ++center) {
        auto &nbh = nbhs[center];
        local_direction.resize(nbh.nbhcount_, 3);
        for (int j = 0; j < nbh.nbhcount_; ++j)
            local_direction.row(j) = (direction.segment<3>(3*nbh.indices_[j])
                                    -direction.segment<3>(3*center)).transpose();
        double energy = 0.;
        apply_site_hvp(nbh, local_direction, energy, local_gradient, local_product);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            result.segment<3>(3*nbh.indices_[j]) += local_product.row(j).transpose();
            result.segment<3>(3*center) -= local_product.row(j).transpose();
        }
    }
    product = std::move(result);
}

void PotentialDescriptorLinear::assemble_curvature(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd *cell, Eigen::MatrixXd *mixed) {
    if (!supports_site_hessian()) GPERROR("Descriptor potential does not support analytical Hessian");
    if (nbhs.size() != atom.size()) GPERROR("Descriptor Hessian neighborhood count differs from atom count");
    if (cell && (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible()))
        GPERROR("Descriptor cell Hessian requires a finite invertible cell");
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    if (cell) {
        inverse = atom.cell_.inverse();
        cell->setZero(9, 9);
        mixed->setZero(9, 3*atom.size());
    }
    Eigen::MatrixXd local;
    SiteCoordinateMatrix gradient;
    for (int center = 0; center < nbhs.size(); ++center) {
        auto &nbh = nbhs[center];
        double energy = 0.;
        desc_wrapper_->desc->compute_weighted_site_hessian(nbh, coeffs_, energy, gradient, local);
        atom.energy_ += energy;
        atom.site_energies_[center] += energy;
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            const int source = nbh.indices_[j];
            atom.forces_[source] -= gradient.row(j).transpose();
            atom.forces_[center] += gradient.row(j).transpose();
            atom.stress_ -= gradient.row(j).transpose() * nbh.pos_[j];
            for (int k = 0; k < nbh.nbhcount_; ++k) {
                const int other = nbh.indices_[k];
                const Eigen::Matrix3d block = local.block<3, 3>(3*j, 3*k);
                hessian(source, other) += block;
                hessian(center, other) -= block;
                hessian(source, center) -= block;
                hessian(center, center) += block;
            }
        }
        if (cell) {
            Eigen::MatrixXd affine = Eigen::MatrixXd::Zero(3*nbh.nbhcount_, 9);
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                const Eigen::RowVector3d fractional = nbh.pos_[j] * inverse;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) affine(3*j+b, 3*a+b) = fractional[a];
            }
            *cell += affine.transpose() * local * affine;
            const Eigen::MatrixXd contraction = affine.transpose() * local;
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                mixed->middleCols<3>(3*nbh.indices_[j]) += contraction.middleCols<3>(3*j);
                mixed->middleCols<3>(3*center) -= contraction.middleCols<3>(3*j);
            }
        }
    }
    atom.setEFSflag();
}

void PotentialDescriptorLinear::getEFSHessian(Atoms &a, AnyNeighborhoods &n, HessianMatrix &h) {
    assemble_curvature(a, n, h, nullptr, nullptr);
}
void PotentialDescriptorLinear::getEFSHessian(Atoms &a, HessianMatrix &h) {
    Neighborhoods n(a, cutoff); getEFSHessian(a, n, h);
}
void PotentialDescriptorLinear::getEFSHessianOMP(Atoms &a, AnyNeighborhoods &n, HessianMatrix &h) {
    getEFSHessian(a, n, h);
}
void PotentialDescriptorLinear::getEFSHessianSymm(Atoms &a, HessianMatrix &h, double) { getEFSHessian(a, h); }
void PotentialDescriptorLinear::getEFSHessianCell(Atoms &a, AnyNeighborhoods &n, HessianMatrix &h,
    Eigen::MatrixXd &c, Eigen::MatrixXd &m) { assemble_curvature(a, n, h, &c, &m); }
void PotentialDescriptorLinear::getEFSHessianCell(Atoms &a, HessianMatrix &h,
    Eigen::MatrixXd &c, Eigen::MatrixXd &m) { Neighborhoods n(a, cutoff); getEFSHessianCell(a, n, h, c, m); }
void PotentialDescriptorLinear::getEFSHessianCellOMP(Atoms &a, AnyNeighborhoods &n, HessianMatrix &h,
    Eigen::MatrixXd &c, Eigen::MatrixXd &m) { getEFSHessianCell(a, n, h, c, m); }
void PotentialDescriptorLinear::getEFSHessianCellSymm(Atoms &a, HessianMatrix &h,
    Eigen::MatrixXd &c, Eigen::MatrixXd &m, double) { getEFSHessianCell(a, h, c, m); }
void PotentialDescriptorLinear::getEFSHessianCellSymmOMP(Atoms &a, HessianMatrix &h,
    Eigen::MatrixXd &c, Eigen::MatrixXd &m, double) { getEFSHessianCell(a, h, c, m); }
