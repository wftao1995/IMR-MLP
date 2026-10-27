// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential.h"

void Potentials::AnyPotential::apply_site_hvp(Neighborhood &, const SiteCoordinateMatrix &,
    double &, SiteCoordinateMatrix &, SiteCoordinateMatrix &) {
    GPERROR("apply_site_hvp is not supported by this potential");
}

void Potentials::AnyPotential::evaluate_site_hessian(Neighborhood &, double &,
    SiteCoordinateMatrix &, HessianMatrix &) {
    GPERROR("evaluate_site_hessian is not supported by this potential");
}



void Potentials::AnyPotential::AccumulateEFSCombinationGard(Atoms &atom,
                                          AnyNeighborhoods &nbhs,
                                          double dLdE,
                                          Eigen::Matrix<double, -1, 3> &dLdF,
                                          Eigen::Matrix3d &dLdS,
                                          Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &wgt_d2udrdx, // buffer
                                          double *out_gard_accumulator)
{
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        wgt_d2udrdx.resize(nbh.nbhcount_, 3);
        wgt_d2udrdx.setZero();
        int itype = nbh.centertype;

        for (int j = 0; j < nbh.nbhcount_; ++j) {
            int jtype = nbh.chemmap_[j];
            if (nbh.distances_[j] < cutoffs(itype, jtype)) {
                for (int a = 0; a < 3; ++a) {
                    wgt_d2udrdx(j, a) += dLdF(i, a);
                    wgt_d2udrdx(j, a) -= dLdF(nbh.indices_[j], a);
                }
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        wgt_d2udrdx(j, a) -= dLdS(a, b) * nbh.pos_[j][b];
            }
        }
        AccumulateFPGard(nbh, out_gard_accumulator, dLdE, &wgt_d2udrdx);
    }
}

void Potentials::AnyPotential::AccumulateEFSCombinationGard(Atoms &atom, AnyNeighborhoods &nbhs, double dLdE, double *out_gard_accumulator)
{
    for (int i = 0; i < nbhs.size(); ++i) {
        Neighborhood &nbh = nbhs[i];
        int itype = nbh.centertype;
        AccumulateFPGard(nbh, out_gard_accumulator, dLdE, nullptr);
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  Numerical gradient helper — central finite differences
// ═════════════════════════════════════════════════════════════════════════

void Potentials::AnyPotential::numerical_gradient_efs(
    Atoms &atom, AnyNeighborhoods &nbhs,
    double dLdE,
    Eigen::Matrix<double, -1, 3> &dLdF,
    Eigen::Matrix3d &dLdS,
    double *out_gard_accumulator,
    int n_params,
    const std::function<void(int, double)> &set_param)
{
    const double h = 1e-6;
    const double inv_2h = 1.0 / (2.0 * h);

    // Save original EFS state (caller expects it unchanged)
    const double e0 = atom.energy_;
    const std::vector<Eigen::Vector3d> f0 = atom.forces_;
    const Eigen::Matrix3d s0 = atom.stress_;
    const std::vector<double> se0 = atom.site_energies_;
    const bool have_energy0 = atom.have_energy, have_stress0 = atom.have_stress;

    for (int idx = 0; idx < n_params; ++idx) {
        // Forward perturbation: +h
        set_param(idx, h);
        atom.resetEFS();
        getEFS(atom, nbhs);
        const double ep = atom.energy_;
        const std::vector<Eigen::Vector3d> fp = atom.forces_;
        const Eigen::Matrix3d sp = atom.stress_;

        // Backward perturbation: -h (from forward)
        set_param(idx, -2.0 * h);
        atom.resetEFS();
        getEFS(atom, nbhs);
        const double em = atom.energy_;
        const std::vector<Eigen::Vector3d> fm = atom.forces_;
        const Eigen::Matrix3d sm = atom.stress_;

        // Restore original value
        set_param(idx, h);

        // Accumulate gradient: dL/dp = dLdE * dE/dp + sum(dLdF * dF/dp) + sum(dLdS * dS/dp)
        double grad = dLdE * (ep - em) * inv_2h;

        // Force contributions
        for (int jj = 0; jj < atom.size(); ++jj) {
            for (int a = 0; a < 3; ++a) {
                grad += dLdF(jj, a) * (fp[jj][a] - fm[jj][a]) * inv_2h;
            }
        }

        // Stress contributions
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                grad += dLdS(a, b) * (sp(a, b) - sm(a, b)) * inv_2h;
            }
        }

        out_gard_accumulator[idx] += grad;
    }

    // Restore original EFS state
    atom.energy_ = e0;
    atom.forces_ = f0;
    atom.stress_ = s0;
    atom.site_energies_ = se0;
    atom.have_energy = have_energy0;
    atom.have_stress = have_stress0;
}

void Potentials::AnyPotential::numerical_gradient_eonly(
    Atoms &atom, AnyNeighborhoods &nbhs,
    double dLdE,
    double *out_gard_accumulator,
    int n_params,
    const std::function<void(int, double)> &set_param)
{
    const double h = 1e-6;
    const double inv_2h = 1.0 / (2.0 * h);

    const double e0 = atom.energy_;
    const std::vector<Eigen::Vector3d> f0 = atom.forces_;
    const Eigen::Matrix3d s0 = atom.stress_;
    const std::vector<double> se0 = atom.site_energies_;
    const bool have_energy0 = atom.have_energy, have_stress0 = atom.have_stress;

    for (int idx = 0; idx < n_params; ++idx) {
        set_param(idx, h);
        atom.resetEFS();
        getEFS(atom, nbhs);
        const double ep = atom.energy_;

        set_param(idx, -2.0 * h);
        atom.resetEFS();
        getEFS(atom, nbhs);
        const double em = atom.energy_;

        set_param(idx, h);

        out_gard_accumulator[idx] += dLdE * (ep - em) * inv_2h;
    }

    atom.energy_ = e0;
    atom.forces_ = f0;
    atom.stress_ = s0;
    atom.site_energies_ = se0;
    atom.have_energy = have_energy0;
    atom.have_stress = have_stress0;
}

void Potentials::AnyPotential::conditional_linear_set_coeffs(const std::vector<int>& indices,
                                                            const Eigen::VectorXd& values) {
    if (values.size()!=static_cast<Eigen::Index>(indices.size()) || !values.allFinite())
        GPERROR("Invalid conditional linear parameter update");
    Eigen::VectorXd full(get_NL_coeffs_count());nl_get_coeffs(full.data());
    for (int index : indices)
        if (index<0 || index>=full.size()) GPERROR("Invalid conditional linear parameter index");
    for (int j=0;j<values.size();++j) full[indices[j]]=values[j];
    nl_set_coeffs(full.data());
}
