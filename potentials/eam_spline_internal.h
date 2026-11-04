// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "potential.h"
#include "eam_spline_io.h"
#include <algorithm>
#include <map>
#include <vector>

// Internal EFS and curvature operations for the two fixed spline EAM adapters.
// Density callbacks preserve each adapter's neighbor/host indexing convention.
namespace Potentials::EAMSpline
{
template <class Density>
void compute_efs(Atoms &atom, AnyNeighborhoods &nbhs, bool add_efs, int Nr, int Nrho, double rdr,
                 double rdrho, double rmax, double rhomin, double rhomax,
                 const std::vector<Eigen::MatrixXd> &frho_spline,
                 const std::vector<Eigen::MatrixXd> &z2r_spline, const Density &density)
{
    if (!add_efs)
        atom.resetEFS();
    else
        atom.setEFSflag();
    // compute EAM
    // 1. compute rho
    double p;
    int m;
    static thread_local std::vector<double> rho, fp;
    rho.assign(atom.size(), 0);
    fp.assign(atom.size(), 0);
    for (int ii = 0; ii < nbhs.size(); ++ii)
    {
        auto &nbh = nbhs[ii];
        int itype = nbh.centertype;
        int iind = ii;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj)
        {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            p = nbh.distances_[jj] * rdr + 1.0;
            m = static_cast<int>(p);
            m = __min(m, Nr - 1);
            p -= m;
            p = __min(p, 1.0);
            auto &coeff = density(jtype, itype);
            rho[iind] += ((coeff(m, 3) * p + coeff(m, 4)) * p + coeff(m, 5)) * p + coeff(m, 6);
            // auto &coeff2 = density(itype, jtype);
            // rho[jind] += ((coeff2(m, 3) * p + coeff2(m, 4)) * p + coeff2(m, 5)) * p + coeff2(m, 6);
        }
    }
    // compute embedding energy
    double phi = 0;
    for (int ii = 0; ii < nbhs.size(); ++ii)
    {
        int indi = ii;
        p = (rho[indi] - rhomin) * rdrho + 1.0;
        m = static_cast<int>(p);
        m = __max(1, _min(m, Nrho - 1));
        p -= m;
        p = __min(p, 1.0);
        int itype = nbhs[ii].centertype;
        auto &coeff = frho_spline[itype];
        fp[indi] = (coeff(m, 0) * p + coeff(m, 1)) * p + coeff(m, 2);
        phi = ((coeff(m, 3) * p + coeff(m, 4)) * p + coeff(m, 5)) * p + coeff(m, 6);
        if (rho[indi] > rhomax)
        {
            phi += fp[indi] * (rho[indi] - rhomax);
        }
        // std::cout << "phi[" << indi << "] = " << phi << std::endl;
        // std::cout << "rho[" << indi << "] = " << rho[indi] << std::endl;
        // printf("rho[%d] = %f\n", indi, rho[indi]);
        // printf("phi[%d] = %f\n", indi, phi);
        // printf("fp[%d] = %f\n", indi, fp[indi]);
        atom.energy_ += phi;
        atom.site_energies_[indi] += phi;
    }
    // compute force
    for (int ii = 0; ii < nbhs.size(); ++ii)
    {
        auto &nbh = nbhs[ii];
        int iind = ii;
        int itype = nbh.centertype;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj)
        {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            int rtype = EAMTable::pair_index(itype, jtype);
            double r = nbh.distances_[jj];
            p = r * rdr + 1.0;
            m = static_cast<int>(p);
            m = __min(m, Nr - 1);
            p -= m;
            p = __min(p, 1.0);
            auto &coeff = density(itype, jtype);
            double rhoip = (coeff(m, 0) * p + coeff(m, 1)) * p + coeff(m, 2);
            auto &coeff2 = density(jtype, itype);
            double rhojp = (coeff2(m, 0) * p + coeff2(m, 1)) * p + coeff2(m, 2);
            auto &coeff3 = z2r_spline[rtype];
            double z2p = (coeff3(m, 0) * p + coeff3(m, 1)) * p + coeff3(m, 2);
            double z2 = ((coeff3(m, 3) * p + coeff3(m, 4)) * p + coeff3(m, 5)) * p + coeff3(m, 6);

            // The existing energy clamps density and r*phi to their last
            // grid values beyond rmax. Differentiate that same branch:
            // constant density and a 1/r pair tail.
            if (r > rmax)
            {
                rhoip = rhojp = z2p = 0.;
            }
            double recip = 1.0 / r;
            double phi = z2 * recip;
            double phip = z2p * recip - phi * recip;
            double psip = fp[iind] * rhojp + fp[jind] * rhoip + phip;
            double fpair = -psip * recip;
            fpair *= 0.5;

            atom.energy_ += 0.5 * phi;
            atom.site_energies_[ii] += 0.5 * phi;

            atom.forces_[iind](0) -= nbh.pos_[jj](0) * fpair;
            atom.forces_[iind](1) -= nbh.pos_[jj](1) * fpair;
            atom.forces_[iind](2) -= nbh.pos_[jj](2) * fpair;
            atom.forces_[jind](0) += nbh.pos_[jj](0) * fpair;
            atom.forces_[jind](1) += nbh.pos_[jj](1) * fpair;
            atom.forces_[jind](2) += nbh.pos_[jj](2) * fpair;

            for (int a = 0; a < 3; a++)
            {
                for (int b = 0; b < 3; b++)
                {
                    atom.stress_(a, b) += nbh.pos_[jj](a) * fpair * nbh.pos_[jj](b);
                }
            }
        }
    }
}

template <class Density>
void compute_efs_omp(Atoms &atom, AnyNeighborhoods &nbhs, bool add_efs, int Nr, int Nrho,
                     double rdr, double rdrho, double rmax, double rhomin, double rhomax,
                     const std::vector<Eigen::MatrixXd> &frho_spline,
                     const std::vector<Eigen::MatrixXd> &z2r_spline, const Density &density)
{
    if (!add_efs)
        atom.resetEFS();
    else
        atom.setEFSflag();
    // compute EAM
    // 1. compute rho
    std::vector<double> rho(atom.size(), 0);
    std::vector<double> fp(atom.size(), 0);
#pragma omp parallel for
    for (int ii = 0; ii < nbhs.size(); ++ii)
    {
        double p;
        int m;
        auto &nbh = nbhs[ii];
        int itype = nbh.centertype;
        int iind = ii;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj)
        {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            p = nbh.distances_[jj] * rdr + 1.0;
            m = static_cast<int>(p);
            m = __min(m, Nr - 1);
            p -= m;
            p = __min(p, 1.0);
            auto &coeff = density(jtype, itype);
            rho[iind] += ((coeff(m, 3) * p + coeff(m, 4)) * p + coeff(m, 5)) * p + coeff(m, 6);
            // auto &coeff2 = density(itype, jtype);
            // rho[jind] += ((coeff2(m, 3) * p + coeff2(m, 4)) * p + coeff2(m, 5)) * p + coeff2(m, 6);
        }
    }
// compute embedding energy
#pragma omp parallel for
    for (int ii = 0; ii < nbhs.size(); ++ii)
    {
        double p;
        int m;
        int indi = ii;
        double phi = 0;
        p = (rho[indi] - rhomin) * rdrho + 1.0;
        m = static_cast<int>(p);
        m = __max(1, _min(m, Nrho - 1));
        p -= m;
        p = __min(p, 1.0);
        int itype = nbhs[ii].centertype;
        auto &coeff = frho_spline[itype];
        fp[indi] = (coeff(m, 0) * p + coeff(m, 1)) * p + coeff(m, 2);
        phi = ((coeff(m, 3) * p + coeff(m, 4)) * p + coeff(m, 5)) * p + coeff(m, 6);
        if (rho[indi] > rhomax)
        {
            phi += fp[indi] * (rho[indi] - rhomax);
        }
// std::cout << "phi[" << indi << "] = " << phi << std::endl;
// std::cout << "rho[" << indi << "] = " << rho[indi] << std::endl;
// printf("rho[%d] = %f\n", indi, rho[indi]);
// printf("phi[%d] = %f\n", indi, phi);
// printf("fp[%d] = %f\n", indi, fp[indi]);
#pragma omp critical
        atom.energy_ += phi;
        atom.site_energies_[indi] += phi;
    }
// compute force
#pragma omp parallel for
    for (int ii = 0; ii < nbhs.size(); ++ii)
    {
        int m;
        double p;
        auto &nbh = nbhs[ii];
        int iind = ii;
        int itype = nbh.centertype;
        for (int jj = 0; jj < nbh.nbhcount_; ++jj)
        {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            int rtype = EAMTable::pair_index(itype, jtype);
            double r = nbh.distances_[jj];
            p = r * rdr + 1.0;
            m = static_cast<int>(p);
            m = __min(m, Nr - 1);
            p -= m;
            p = __min(p, 1.0);
            auto &coeff = density(itype, jtype);
            double rhoip = (coeff(m, 0) * p + coeff(m, 1)) * p + coeff(m, 2);
            auto &coeff2 = density(jtype, itype);
            double rhojp = (coeff2(m, 0) * p + coeff2(m, 1)) * p + coeff2(m, 2);
            auto &coeff3 = z2r_spline[rtype];
            double z2p = (coeff3(m, 0) * p + coeff3(m, 1)) * p + coeff3(m, 2);
            double z2 = ((coeff3(m, 3) * p + coeff3(m, 4)) * p + coeff3(m, 5)) * p + coeff3(m, 6);

            // The existing energy clamps density and r*phi to their last
            // grid values beyond rmax. Differentiate that same branch:
            // constant density and a 1/r pair tail.
            if (r > rmax)
            {
                rhoip = rhojp = z2p = 0.;
            }
            double recip = 1.0 / r;
            double phi = z2 * recip;
            double phip = z2p * recip - phi * recip;
            double psip = fp[iind] * rhojp + fp[jind] * rhoip + phip;
            double fpair = -psip * recip;
            fpair *= 0.5;
#pragma omp critical
            {
                atom.energy_ += 0.5 * phi;
                atom.site_energies_[ii] += 0.5 * phi;

                atom.forces_[iind](0) -= nbh.pos_[jj](0) * fpair;
                atom.forces_[iind](1) -= nbh.pos_[jj](1) * fpair;
                atom.forces_[iind](2) -= nbh.pos_[jj](2) * fpair;
                atom.forces_[jind](0) += nbh.pos_[jj](0) * fpair;
                atom.forces_[jind](1) += nbh.pos_[jj](1) * fpair;
                atom.forces_[jind](2) += nbh.pos_[jj](2) * fpair;

                for (int a = 0; a < 3; a++)
                {
                    for (int b = 0; b < 3; b++)
                    {
                        atom.stress_(a, b) += nbh.pos_[jj](a) * fpair * nbh.pos_[jj](b);
                    }
                }
            }
        }
    }
}

struct Jet
{
    double value, first, second;
};

inline Jet polynomial(const Eigen::MatrixXd &table, int interval, double fraction,
                      double inverse_step)
{
    return {((table(interval, 3) * fraction + table(interval, 4)) * fraction + table(interval, 5)) *
                    fraction +
                table(interval, 6),
            (table(interval, 0) * fraction + table(interval, 1)) * fraction + table(interval, 2),
            (2 * table(interval, 0) * fraction + table(interval, 1)) * inverse_step};
}

inline Jet radial(const Eigen::MatrixXd &table, int count, double inverse_step, double gridmax,
                  double r)
{
    double p = r * inverse_step + 1.;
    const int m = std::min(static_cast<int>(p), count - 1);
    p = std::min(p - m, 1.);
    auto result = polynomial(table, m, p, inverse_step);
    if (r > gridmax)
        result.first = result.second = 0.;
    return result;
}

inline Jet embedding(const Eigen::MatrixXd &table, int count, double inverse_step, double rhomin,
                     double rhomax, double rho)
{
    double p = (rho - rhomin) * inverse_step + 1.;
    const int m = std::max(1, std::min(static_cast<int>(p), count - 1));
    p = std::min(p - m, 1.);
    auto result = polynomial(table, m, p, inverse_step);
    if (rho > rhomax)
    {
        result.value += result.first * (rho - rhomax);
        result.second = 0.;
    }
    return result;
}

inline Jet pair_from_rphi(const Jet &z, double r)
{
    const double inverse = 1. / r;
    return {z.value * inverse, z.first * inverse - z.value * inverse * inverse,
            z.second * inverse - 2 * z.first * inverse * inverse +
                2 * z.value * inverse * inverse * inverse};
}

template <class Density, class Pair, class Embedding>
void assemble(const Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
              Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess, Density density, Pair pair,
              Embedding embedding)
{
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    if (cellhess)
    {
        inverse = atom.cell_.inverse();
        *cellhess = Eigen::MatrixXd::Zero(9, 9);
        *lrhess = Eigen::MatrixXd::Zero(9, 3 * atom.size());
    }

    struct Radial
    {
        Jet density, pair;
    };

    std::vector<Radial> radial_values;
    for (int i = 0; i < nbhs.size(); ++i)
    {
        const auto &n = nbhs[i];
        radial_values.resize(n.nbhcount_);
        double rho = 0.;
        for (int j = 0; j < n.nbhcount_; ++j)
        {
            auto &v = radial_values[j];
            v.density = density(n.chemmap_[j], n.centertype, n.distances_[j]);
            v.pair = pair(n.centertype, n.chemmap_[j], n.distances_[j]);
            rho += v.density.value;
        }
        const auto f = embedding(n.centertype, rho);
        std::map<int, Eigen::Vector3d> gradient;
        gradient.emplace(i, Eigen::Vector3d::Zero());
        Eigen::Matrix<double, 9, 1> cell_gradient = Eigen::Matrix<double, 9, 1>::Zero();
        for (int j = 0; j < n.nbhcount_; ++j)
        {
            const auto &v = radial_values[j];
            const int source = n.indices_[j];
            const double r = n.distances_[j];
            const Eigen::Vector3d u = n.pos_[j].transpose() / r;
            const double first = .5 * v.pair.first + f.first * v.density.first,
                         second = .5 * v.pair.second + f.first * v.density.second;
            const Eigen::Matrix3d block = (first / r) * Eigen::Matrix3d::Identity() +
                                          (second - first / r) * u * u.transpose();
            if (cellhess)
            {
                const Eigen::RowVector3d fractional = n.pos_[j] * inverse;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                    {
                        cell_gradient[3 * a + b] += fractional[a] * v.density.first * u[b];
                        for (int l = 0; l < 3; ++l)
                            for (int q = 0; q < 3; ++q)
                                (*cellhess)(3 * a + b, 3 * l + q) +=
                                    fractional[a] * block(b, q) * fractional[l];
                        for (int q = 0; q < 3; ++q)
                        {
                            const double value = fractional[a] * block(b, q);
                            (*lrhess)(3 * a + b, 3 * source + q) += value;
                            (*lrhess)(3 * a + b, 3 * i + q) -= value;
                        }
                    }
            }
            if (source == i)
                continue;
            hessian(source, source) += block;
            hessian(i, i) += block;
            hessian(source, i) -= block;
            hessian(i, source) -= block;
            gradient.try_emplace(source, Eigen::Vector3d::Zero()).first->second +=
                v.density.first * u;
            gradient.at(i) -= v.density.first * u;
        }
        for (const auto &[a, qa] : gradient)
            for (const auto &[b, qb] : gradient)
                hessian(a, b) += f.second * qa * qb.transpose();
        if (cellhess)
        {
            *cellhess += f.second * cell_gradient * cell_gradient.transpose();
            for (const auto &[source, q] : gradient)
                lrhess->middleCols(3 * source, 3) += f.second * cell_gradient * q.transpose();
        }
    }
}

// Direct local curvature contraction: no atomic Hessian is constructed.
template <class Density, class Pair, class Embedding>
void apply(const Atoms &atom, AnyNeighborhoods &nbhs, const Eigen::VectorXd &direction,
           Eigen::VectorXd &product, Density density, Pair pair, Embedding embedding)
{
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());

    struct Radial
    {
        Jet density, pair;
        double alpha;
    };

    std::vector<Radial> radial_values;
    for (int i = 0; i < nbhs.size(); ++i)
    {
        const auto &n = nbhs[i];
        radial_values.resize(n.nbhcount_);
        double rho = 0., delta_rho = 0.;
        for (int j = 0; j < n.nbhcount_; ++j)
        {
            auto &v = radial_values[j];
            const double r = n.distances_[j];
            v.density = density(n.chemmap_[j], n.centertype, r);
            v.pair = pair(n.centertype, n.chemmap_[j], r);
            const Eigen::Vector3d w =
                direction.segment<3>(3 * n.indices_[j]) - direction.segment<3>(3 * i);
            v.alpha = n.pos_[j].dot(w) / r;
            rho += v.density.value;
            delta_rho += v.density.first * v.alpha;
        }
        const auto f = embedding(n.centertype, rho);
        for (int j = 0; j < n.nbhcount_; ++j)
        {
            const auto &v = radial_values[j];
            const double r = n.distances_[j];
            const Eigen::Vector3d u = n.pos_[j].transpose() / r;
            const Eigen::Vector3d w =
                direction.segment<3>(3 * n.indices_[j]) - direction.segment<3>(3 * i);
            const double first = .5 * v.pair.first + f.first * v.density.first,
                         second = .5 * v.pair.second + f.first * v.density.second;
            const Eigen::Vector3d contribution = second * v.alpha * u +
                                                 (first / r) * (w - v.alpha * u) +
                                                 f.second * delta_rho * v.density.first * u;
            result.segment<3>(3 * n.indices_[j]) += contribution;
            result.segment<3>(3 * i) -= contribution;
        }
    }
    product = std::move(result);
}

}
