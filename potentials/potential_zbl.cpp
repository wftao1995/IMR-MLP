// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_zbl.h"
#include "potential_settings.h"
#include "base/neighborhood.h"
#include "utils/gputils.h"
#include <cmath>
#include <limits>
#include <iomanip>

using namespace Potentials;

PotentialZBL::PotentialZBL(GPSettings settings) {
    initialize(std::move(settings));
}

void PotentialZBL::initialize(GPSettings settings) {
    // Read element atomic numbers
    if (!settings.count("z")) {
        GPERROR("zbl: z must list one atomic number for each species");
    }

    std::string zstr = settings["z"];
    std::vector<std::string> spl;
    str_split(zstr, spl);
    if (spl.empty()) {
        GPERROR("zbl: z must list one atomic number for each species");
    }

    z.clear();
    z.reserve(spl.size());
    for (const auto &token : spl) {
        std::size_t parsed = 0;
        double atomic_number = 0.0;
        try {
            atomic_number = std::stod(token, &parsed);
        } catch (const std::exception &) {
            GPERROR("zbl: invalid atomic number: " + token);
        }
        if (parsed != token.size() || !std::isfinite(atomic_number)
            || atomic_number <= 0.0 || std::floor(atomic_number) != atomic_number
            || atomic_number > std::numeric_limits<int>::max()) {
            GPERROR("zbl: atomic numbers must be finite positive integers");
        }
        z.push_back(atomic_number);
    }
    species_count = static_cast<int>(z.size());

    r0 = 1.0; r1 = 2.0;
    try {
        if (settings.count("r0")) r0 = std::stod(settings["r0"]);
        if (settings.count("r1")) r1 = std::stod(settings["r1"]);
    } catch (const std::exception &) {
        GPERROR("zbl: r0 and r1 must be finite numbers");
    }
    if (!std::isfinite(r0) || !std::isfinite(r1)
        || r0 <= 0.0 || r1 <= r0) {
        GPERROR("zbl: switching radii must satisfy 0 < r0 < r1");
    }

    cutoff = r1;
    cutoffsq = cutoff * cutoff;
    cutoffs.resize(species_count, species_count);
    cutoffsqs.resize(species_count, species_count);
    r_mins.resize(species_count, species_count);
    for (int i = 0; i < species_count; ++i) {
        for (int j = 0; j < species_count; ++j) {
            cutoffs(i, j) = cutoff;
            cutoffsqs(i, j) = cutoffsq;
            r_mins(i, j) = 0.0;
        }
    }
    nl_inited = true;
}

void PotentialZBL::Load(std::istream &is) {
    initialize(read_potential_settings(is));
}

void PotentialZBL::Save(std::ostream &os) {
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << "pot:pottype zbl\n";
    os << "pot:z";
    for (auto &zi : z) os << " " << zi;
    os << "\n";
    os << "pot:r0 " << r0 << "\n";
    os << "pot:r1 " << r1 << "\n";
}

inline double PotentialZBL::switch_fn(double r) const {
    if (r <= r0) return 1.0;
    if (r >= r1) return 0.0;
    double frac = (r - r0) / (r1 - r0);
    return 1.0 - frac * frac * (3.0 - 2.0 * frac);
}

inline double PotentialZBL::switch_fn_deriv(double r) const {
    if (r <= r0 || r >= r1) return 0.0;
    double frac = (r - r0) / (r1 - r0);
    double inv_delta = 1.0 / (r1 - r0);
    return -6.0 * frac * (1.0 - frac) * inv_delta;
}

double PotentialZBL::zbl_energy(double r, int zi, int zj) const {
    double a = a0 / (std::pow(static_cast<double>(zi), zp)
                   + std::pow(static_cast<double>(zj), zp));
    double x = r / a;
    double sum = c1 * std::exp(p1 * x) + c2 * std::exp(p2 * x)
               + c3 * std::exp(p3 * x) + c4 * std::exp(p4 * x);
    return cc_ev * zi * zj / r * sum;
}

double PotentialZBL::zbl_energy_deriv(double r, int zi, int zj) const {
    double a = a0 / (std::pow(static_cast<double>(zi), zp)
                   + std::pow(static_cast<double>(zj), zp));
    double x = r / a;
    double sum = c1 * std::exp(p1 * x) + c2 * std::exp(p2 * x)
               + c3 * std::exp(p3 * x) + c4 * std::exp(p4 * x);
    double dsum = c1 * p1 * std::exp(p1 * x) + c2 * p2 * std::exp(p2 * x)
                + c3 * p3 * std::exp(p3 * x) + c4 * p4 * std::exp(p4 * x);
    double inv_a = 1.0 / a;
    return -cc_ev * zi * zj / (r * r) * sum
           + cc_ev * zi * zj / r * dsum * inv_a;
}

double PotentialZBL::zbl_energy_second_deriv(double r, int zi, int zj) const {
    const double a = a0 / (std::pow(static_cast<double>(zi), zp)
                          + std::pow(static_cast<double>(zj), zp));
    double sum = 0., first = 0., second = 0.;
    const double coefficients[] = {c1, c2, c3, c4};
    const double exponents[] = {p1, p2, p3, p4};
    for (int k = 0; k < 4; ++k) {
        const double lambda = exponents[k] / a;
        const double term = coefficients[k] * std::exp(lambda * r);
        sum += term;
        first += lambda * term;
        second += lambda * lambda * term;
    }
    return cc_ev * zi * zj * (second / r - 2 * first / (r * r) + 2 * sum / (r * r * r));
}

Eigen::Matrix3d PotentialZBL::pair_curvature(double r, const Eigen::RowVector3d &delta,
                                            int zi, int zj) const {
    if (r >= r1) return Eigen::Matrix3d::Zero();
    const double energy = zbl_energy(r, zi, zj);
    const double first = zbl_energy_deriv(r, zi, zj);
    const double s = switch_fn(r), ds = switch_fn_deriv(r);
    double dds = 0.;
    // The original cubic switch is C1. At r0 retain its r<=r0 branch;
    // r>=r1 is excluded above, rather than assigning a C2 endpoint limit.
    if (r > r0) {
        const double width = r1 - r0;
        const double t = (r - r0) / width;
        dds = (-6 + 12 * t) / (width * width);
    }
    const double tangential = (s * first + ds * energy) / r;
    const double longitudinal = s * zbl_energy_second_deriv(r, zi, zj)
        + 2 * ds * first + dds * energy;
    const Eigen::Vector3d u = delta.transpose() / r;
    return tangential * Eigen::Matrix3d::Identity()
        + (longitudinal - tangential) * u * u.transpose();
}

void PotentialZBL::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    if (z.empty()) return;
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        const int zi = static_cast<int>(z[nbh.centertype]);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            const int source = nbh.indices_[j];
            if (source == i || nbh.distances_[j] >= r1) continue;
            const Eigen::Matrix3d curvature = .5 * pair_curvature(nbh.distances_[j], nbh.pos_[j],
                zi, static_cast<int>(z[nbh.chemmap_[j]]));
            hessian(i, i) += curvature;
            hessian(source, source) += curvature;
            hessian(i, source) -= curvature;
            hessian(source, i) -= curvature;
        }
    }
}

void PotentialZBL::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}

void PotentialZBL::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double) {
    getEFSHessian(atom, hessian);
}

void PotentialZBL::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    if (direction.size() != 3 * atom.size()) GPERROR("Hessian-vector direction has the wrong size");
    if (nbhs.size() != atom.size()) GPERROR("Hessian-vector neighborhoods have the wrong size");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    const int nz = static_cast<int>(z.size());
    if (nz != 0) {
        for (int i = 0; i < nbhs.size(); ++i) {
            const auto &nbh = nbhs[i];
            if (nbh.centertype < 0 || nbh.centertype >= nz)
                GPERROR("zbl: central atom species index is out of range");
            const int zi = static_cast<int>(z[nbh.centertype]);
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                const int type = nbh.chemmap_[j], source = nbh.indices_[j];
                if (type < 0 || type >= nz) GPERROR("zbl: neighbor species index is out of range");
                if (source == i || nbh.distances_[j] >= r1) continue;
                const Eigen::Vector3d difference = direction.segment<3>(3 * i)
                    - direction.segment<3>(3 * source);
                const Eigen::Vector3d contribution = .5 * pair_curvature(nbh.distances_[j], nbh.pos_[j],
                    zi, static_cast<int>(z[type])) * difference;
                result.segment<3>(3 * i) += contribution;
                result.segment<3>(3 * source) -= contribution;
            }
        }
    }
    product = std::move(result);
}

void PotentialZBL::CalcSiteEFS(
    Neighborhood &nbh, double &energy, SiteCoordinateMatrix &gradient) {
    if (!supports_site_efs()) GPERROR("zbl: site EFS requires an initialized model");
    const int count = static_cast<int>(z.size());
    if (nbh.centertype < 0 || nbh.centertype >= count)
        GPERROR("zbl: central atom species index is out of range");
    energy = 0.;
    gradient.setZero(nbh.nbhcount_, 3);
    const int zi = static_cast<int>(z[nbh.centertype]);
    for (int j = 0; j < nbh.nbhcount_; ++j) {
        const int type = nbh.chemmap_[j];
        if (type < 0 || type >= count) GPERROR("zbl: neighbor species index is out of range");
        const double r = nbh.distances_[j];
        if (r >= r1) continue;
        if (!(r > 0.)) GPERROR("zbl: site EFS requires positive neighbor distances");
        const int zj = static_cast<int>(z[type]);
        const double pair_energy = zbl_energy(r, zi, zj);
        const double value = switch_fn(r), derivative = switch_fn_deriv(r);
        energy += .5 * value * pair_energy;
        gradient.row(j) = .5 * (value * zbl_energy_deriv(r, zi, zj)
            + derivative * pair_energy) / r * nbh.pos_[j];
    }
}

void PotentialZBL::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    int nz = static_cast<int>(z.size());
    if (nz == 0) return;

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        int itype = nbh.centertype;
        int iind = ii;
        if (itype < 0 || itype >= nz) {
            GPERROR("zbl: central atom species index is out of range");
        }
        int zi = static_cast<int>(z[itype]);

        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            if (jtype < 0 || jtype >= nz) {
                GPERROR("zbl: neighbor species index is out of range");
            }
            int zj = static_cast<int>(z[jtype]);
            double r = nbh.distances_[jj];
            if (r >= r1) continue;

            double zbl_e = zbl_energy(r, zi, zj);
            double zbl_deriv_over_r = zbl_energy_deriv(r, zi, zj) / r;

            double s = switch_fn(r);
            double ds = switch_fn_deriv(r);

            atom.energy_ += 0.5 * s * zbl_e;
            atom.site_energies_[ii] += 0.5 * s * zbl_e;

            // Each pair occurs in both atoms' neighborhoods.  This is half of
            // the force contribution for this directed visit, consistent with
            // the 0.5 pair-energy contribution above.
            double dUdr_over_r = s * zbl_deriv_over_r + ds * zbl_e / r;
            Eigen::RowVector3d force_i = 0.5 * dUdr_over_r * nbh.pos_[jj];
            atom.forces_[iind] += force_i;
            atom.forces_[jind] -= force_i;

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= force_i(a) * nbh.pos_[jj](b);
                }
            }
        }
    }
}

void PotentialZBL::getE(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetE(); else atom.setEflag();

    const int nz = static_cast<int>(z.size());
    if (nz == 0) return;

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        if (nbh.centertype < 0 || nbh.centertype >= nz) {
            GPERROR("zbl: central atom species index is out of range");
        }
        const int zi = static_cast<int>(z[nbh.centertype]);
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            const int jtype = nbh.chemmap_[jj];
            if (jtype < 0 || jtype >= nz) {
                GPERROR("zbl: neighbor species index is out of range");
            }
            const int zj = static_cast<int>(z[jtype]);
            const double r = nbh.distances_[jj];
            if (r >= r1) continue;
            const double site_energy = 0.5 * switch_fn(r)
                                     * zbl_energy(r, zi, zj);
            atom.energy_ += site_energy;
            atom.site_energies_[ii] += site_energy;
        }
    }
}

void PotentialZBL::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    if (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Analytical ZBL cell Hessian requires a finite invertible cell");
    const Eigen::Matrix3d inverse = atom.cell_.inverse();
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    cellhess = Eigen::MatrixXd::Zero(9, 9);
    lrhess = Eigen::MatrixXd::Zero(9, 3 * atom.size());
    if (z.empty()) return;
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        const int zi = static_cast<int>(z[nbh.centertype]);
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            if (nbh.distances_[j] >= r1) continue;
            const int source = nbh.indices_[j];
            const Eigen::Matrix3d curvature = .5 * pair_curvature(
                nbh.distances_[j], nbh.pos_[j], zi, static_cast<int>(z[nbh.chemmap_[j]]));
            const Eigen::RowVector3d fractional = nbh.pos_[j] * inverse;
            // Native cell coordinates move all images affinely; Cartesian
            // residuals move source atoms independently. Self images therefore
            // contribute to the cell block while their atomic terms cancel.
            hessian(i, i) += curvature;
            hessian(source, source) += curvature;
            hessian(i, source) -= curvature;
            hessian(source, i) -= curvature;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) {
                    for (int l = 0; l < 3; ++l)
                        for (int p = 0; p < 3; ++p)
                            cellhess(3*a+b, 3*l+p) += fractional(a) * curvature(b,p) * fractional(l);
                    for (int g = 0; g < 3; ++g) {
                        const double cross = fractional(a) * curvature(b,g);
                        lrhess(3*a+b, 3*source+g) += cross;
                        lrhess(3*a+b, 3*i+g) -= cross;
                    }
                }
        }
    }
}

void PotentialZBL::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}
void PotentialZBL::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec) {
    getEFSHessianCellSymmOMP(atom, hessian, cellhess, lrhess, symprec);
}

void Potentials::PotentialZBL::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::PotentialZBL::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::PotentialZBL::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian) {
    getEFSHessian(atom, nbhs, hessian);
}
