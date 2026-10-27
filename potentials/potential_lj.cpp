// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_lj.h"
#include "potential_settings.h"
#include "utils/compressed_stream.h"
#include <iomanip>
#include <limits>
#include <utility>

using namespace Potentials;

Potentials::LJ612::LJ612(GPSettings settings)
{
    InitSettings();
    ApplySettings(settings);
    ncoeffs = ntypes * (ntypes + 2);
    nljcoeffs = ntypes * (ntypes + 1);
    sig.resize(ntypes, ntypes);
    eps.resize(ntypes, ntypes);
    elemoffset.resize(ntypes);
    elemoffset.setZero();
    sig.setZero();
    eps.setZero();
    lj1 = sig;
    lj2 = sig;
    lj3 = sig;
    lj4 = sig;
    cutoffs.resize(ntypes, ntypes);
    cutoffs.setZero();
    cutoffsqs.resize(ntypes, ntypes);
    // give :i-j eps sig
    for (int i = 0; i < ntypes; ++i) {
        for (int j = 0; j < ntypes; ++j) {
            std::string ijp = std::to_string(i) + "-" + std::to_string(j);
            int cc = settings.count(ijp);
            if (cc == 0) continue;
            std::string val = settings[ijp];
            std::vector<std::string> splstr;
            str_split(val, splstr);
            if (splstr.size() != 2)
                GPERROR("LJ pair " + ijp + " requires 2 values (epsilon sigma), got " + std::to_string(splstr.size()));
            eps(i, j) = std::stod(splstr[0]);
            sig(i, j) = std::stod(splstr[1]);
        }
    }
    // An explicitly supplied upper pair, including zero, takes precedence.
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i + 1; j < ntypes; ++j) {
            const std::string upper = std::to_string(i) + "-" + std::to_string(j);
            const std::string lower = std::to_string(j) + "-" + std::to_string(i);
            if (!settings.count(upper) && settings.count(lower)) {
                eps(i, j) = eps(j, i);
                sig(i, j) = sig(j, i);
            }
            eps(j, i) = eps(i, j);
            sig(j, i) = sig(i, j);
        }
    }
    convert_eps2lj();
    // get cutoff
    cutoff = 0;
    GPSettings cutoffsetting = settings.ExtractSubSettings("cut");
    for (int i = 0; i < ntypes; ++i) {
        for (int j = 0; j < ntypes; ++j) {
            std::string ijp = std::to_string(i) + "-" + std::to_string(j);
            int cc = cutoffsetting.count(ijp);
            if (cc == 0) continue;
            std::string val = cutoffsetting[ijp];
            cutoffs(i, j) = std::stod(val);
            cutoffs(j, i) = cutoffs(i, j);
            if (cutoff < cutoffs(i, j)) cutoff = cutoffs(i, j);
        }
    }
    // get elem
    GPSettings elemsettings = settings.ExtractSubSettings("ele");
    for (int i = 0; i < ntypes; ++i) {
        std::string ielm = std::to_string(i);
        int cc = elemsettings.count(ielm);
        if (cc == 0) continue;
        std::string val = elemsettings[ielm];
        elemoffset[i] = std::stod(val);
    }
    cutoffsq = cutoff * cutoff;
    for (int i = 0; i < ntypes; ++i) {
        for (int j = 0; j < ntypes; ++j) {
            cutoffsqs(i, j) = cutoffs(i, j) * cutoffs(i, j);
        }
    }
    species_count = ntypes;
    r_mins.resize(ntypes, ntypes);
    r_mins.setZero();
    nl_inited = true;
}

void Potentials::LJ612::convert_eps2lj()
{
    dlj1deps.resize(ntypes, ntypes);
    dlj2deps = dlj1deps;
    dlj3deps = dlj1deps;
    dlj4deps = dlj1deps;
    dlj1dsig = dlj1deps;
    dlj2dsig = dlj1deps;
    dlj3dsig = dlj1deps;
    dlj4dsig = dlj1deps;
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            double sig2 = sig(i, j) * sig(i, j);
            double sig6 = sig2 * sig2 * sig2;
            double sig5 = sig2 * sig2 * sig(i, j);
            double sig11 = sig5 * sig6;
            double sig12 = sig6 * sig6;
            lj1(i, j) = 48 * eps(i, j) * sig12;
            lj1(j, i) = lj1(i, j);
            lj2(i, j) = 24 * eps(i, j) * sig6;
            lj2(j, i) = lj2(i, j);
            lj3(i, j) = 4 * eps(i, j) * sig12;
            lj3(j, i) = lj3(i, j);
            lj4(i, j) = 4 * eps(i, j) * sig6;
            lj4(j, i) = lj4(i, j);
            dlj1deps(i, j) = 48 * sig12;
            dlj1deps(j, i) = dlj1deps(i, j);
            dlj2deps(i, j) = 24 * sig6;
            dlj2deps(j, i) = dlj2deps(i, j);
            dlj3deps(i, j) = 4 * sig12;
            dlj3deps(j, i) = dlj3deps(i, j);
            dlj4deps(i, j) = 4 * sig6;
            dlj4deps(j, i) = dlj4deps(i, j);
            dlj1dsig(i, j) = 12.0 * 48 * eps(i, j) * sig11;
            dlj1dsig(j, i) = dlj1dsig(i, j);
            dlj2dsig(i, j) = 6.0 * 24 * eps(i, j) * sig5;
            dlj2dsig(j, i) = dlj2dsig(i, j);
            dlj3dsig(i, j) = 12.0 * 4 * eps(i, j) * sig11;
            dlj3dsig(j, i) = dlj3dsig(i, j);
            dlj4dsig(i, j) = 6.0 * 4 * eps(i, j) * sig5;
            dlj4dsig(j, i) = dlj4dsig(i, j);
        }
    }
}

void Potentials::LJ612::convert_lj2eps()
{
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            sig(i, j) = pow(lj1(i, j) / lj2(i, j) * 0.5, 1.0 / 6.0);
            sig(j, i) = sig(i, j);
            eps(i, j) = lj2(i, j) * lj2(i, j) / 12.0 / lj1(i, j);
            eps(j, i) = eps(i, j);
        }
    }
}

void Potentials::LJ612::getEFS(Atoms &atom, AnyNeighborhoods &nbhs)
{
    if (!add_efs) atom.resetEFS();
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &nbh = nbhs[i];
        int nneigh = nbh.nbhcount_;
        int itype = atom.chemmap_[i];
        atom.energy_ += elemoffset[itype];
        atom.site_energies_[i] += elemoffset[itype];
        for (int j = 0; j < nneigh; ++j) {
            int jtype = nbh.chemmap_[j];
            int atj = nbh.indices_[j];
            double rc = cutoffs(itype, jtype);
            if (nbh.distances_[j] < rc) {
                double rinv = 1.0 / nbh.distances_[j];
                double r2inv = rinv * rinv;
                double r6inv = r2inv * r2inv * r2inv;
                double force = r2inv * r6inv * (lj1(itype, jtype) * r6inv - lj2(itype, jtype));
                // 0.5: each pair is visited twice (from i and from j),
                // so energy, force, and stress must be halved.
                const double site_energy = 0.5 * r6inv * (lj3(itype,jtype) * r6inv - lj4(itype,jtype));
                atom.energy_ += site_energy;
                atom.site_energies_[i] += site_energy;
                // fce = -(0.5*dV/dr)*r_hat: force points AWAY from neighbor
                auto fce = -0.5 * force * nbh.pos_[j];
                atom.forces_[i] += fce;
                atom.forces_[atj] -= fce;
                // add stress term, here we can not use fdotr after all eng and force calcs because we do not build ghost list
                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        atom.stress_(a, b) -= fce(a) * nbh.pos_[j](b);
                    }
                }
            } 
        }
    }
}

Eigen::Matrix3d LJ612::pair_curvature(int itype, int jtype, double distance,
                                     const Eigen::RowVector3d &displacement) const
{
    const double r2inv = 1.0 / (distance * distance);
    const double r6inv = r2inv * r2inv * r2inv;
    const double c12r12 = lj3(itype, jtype) * r6inv * r6inv;
    const double c6r6 = lj4(itype, jtype) * r6inv;
    const double transverse = (-12.0 * c12r12 + 6.0 * c6r6) * r2inv;
    const double radial = (156.0 * c12r12 - 42.0 * c6r6) * r2inv;
    return transverse * Eigen::Matrix3d::Identity()
         + ((radial - transverse) * r2inv) * displacement.transpose() * displacement;
}

void LJ612::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs,
                          HessianMatrix &hessian)
{
    if (nbhs.size() != atom.size())
        GPERROR("Hessian neighborhoods have the wrong size");
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        const int itype = atom.chemmap_[i];
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            const int atj = nbh.indices_[j];
            const int jtype = nbh.chemmap_[j];
            // A self-image displacement is constant under atomic motion.
            if (atj == i || !(nbh.distances_[j] < cutoffs(itype, jtype))) continue;
            // The complete neighborhood visits each pair from both ends.
            const Eigen::Matrix3d curvature = 0.5 * pair_curvature(
                itype, jtype, nbh.distances_[j], nbh.pos_[j]);
            hessian(i, i) += curvature;
            hessian(atj, atj) += curvature;
            hessian(i, atj) -= curvature;
            hessian(atj, i) -= curvature;
        }
    }
}

void LJ612::getEFSHessian(Atoms &atom, HessianMatrix &hessian)
{
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}

void LJ612::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double)
{
    getEFSHessian(atom, hessian);
}

void LJ612::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
                                     const Eigen::VectorXd &direction,
                                     Eigen::VectorXd &product)
{
    if (direction.size() != 3 * atom.size())
        GPERROR("Hessian-vector direction has the wrong size");
    if (nbhs.size() != atom.size())
        GPERROR("Hessian-vector neighborhoods have the wrong size");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        const int itype = atom.chemmap_[i];
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            const int atj = nbh.indices_[j];
            const int jtype = nbh.chemmap_[j];
            if (atj == i || !(nbh.distances_[j] < cutoffs(itype, jtype))) continue;
            const Eigen::Vector3d value = 0.5 * pair_curvature(
                itype, jtype, nbh.distances_[j], nbh.pos_[j])
                * (direction.segment<3>(3 * i) - direction.segment<3>(3 * atj));
            result.segment<3>(3 * i) += value;
            result.segment<3>(3 * atj) -= value;
        }
    }
    product = std::move(result);
}

int Potentials::LJ612::get_NL_coeffs_count() const
{
    return ncoeffs;
}

void Potentials::LJ612::nl_get_coeffs(double *x) const
{
    int ct = 0;
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            x[ct] = eps(i, j);
            ++ct;
            x[ct] = sig(i, j);
            ++ct;
        }
    }
    for (int i = 0; i < ntypes; ++i) {
        x[ct] = elemoffset[i];
        ++ct;
    }
}

void Potentials::LJ612::nl_set_coeffs(const double *x)
{
    int ct = 0;
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            eps(i, j) = x[ct];
            ++ct;
            sig(i, j) = x[ct];
            ++ct;
            eps(j, i) = eps(i, j);
            sig(j, i) = sig(i, j);
        }
    }
    for (int i = 0; i < ntypes; ++i) {
        elemoffset(i) = x[ct];
        ++ct;
    }
    convert_eps2lj();
}

void Potentials::LJ612::AccumulateFPGard(Neighborhood &nbh, double *out_gard_accumulator, double se_weight, Eigen::Matrix<double, -1, 3, Eigen::RowMajor> *f_weight)
{
    int nneigh = nbh.nbhcount_;
    int itype = nbh.centertype;
    int idt = nljcoeffs + itype;
    out_gard_accumulator[idt] += se_weight;
    for (int j = 0; j < nneigh; ++j) {
        int jtype = nbh.chemmap_[j];
        double rc = cutoffs(itype, jtype);
        if (nbh.distances_[j] < rc) { 
            double rinv = 1.0 / nbh.distances_[j];
            double r2inv = rinv * rinv;
            double r6inv = r2inv * r2inv * r2inv;
            int idg = mapoid2gid[std::make_pair(itype, jtype)] * 2;
            // energy part 
            // r6inv * (lj3(itype,jtype) * r6inv - lj4(itype,jtype));
            out_gard_accumulator[idg] += 0.5 * se_weight * r6inv * (dlj3deps(itype, jtype) * r6inv - dlj4deps(itype, jtype));
            out_gard_accumulator[idg+1] += 0.5 * se_weight * r6inv * (dlj3dsig(itype, jtype) * r6inv - dlj4dsig(itype, jtype));
            // f part 
            if (f_weight != nullptr) {
                double r8inv = r2inv * r6inv;
                double dfdeps = r8inv * (dlj1deps(itype, jtype) * r6inv - dlj2deps(itype, jtype));
                double dfdsig = r8inv * (dlj1dsig(itype, jtype) * r6inv - dlj2dsig(itype, jtype));
                for (int a = 0; a < 3; ++a) {
                    double wgt = (*f_weight)(j, a);
                    out_gard_accumulator[idg] -= 0.5 * wgt * dfdeps * nbh.pos_[j](a);
                    out_gard_accumulator[idg+1] -= 0.5 * wgt * dfdsig * nbh.pos_[j](a);
                }
            }
        }
    }
}

void Potentials::LJ612::Load(std::istream &is)
{
    const auto settings = read_potential_settings(is);
    if (settings.count("pottype") && settings["pottype"] != "lj612")
        throw std::runtime_error("Not an LJ612 potential");
    LJ612 loaded(settings);
    loaded.train_init();
    loaded.add_efs = add_efs;
    *this = std::move(loaded);
    // ConnectSettings bindings are non-owning: bind this object's members
    // again instead of retaining the temporary model's ntypes address.
    InitSettings();
}

void Potentials::LJ612::Save(const std::string &fnm)
{
    if (save_gpm_if_requested(*this, fnm)) return;
    auto ofs = open_ofile(fnm);
    if (!ofs) throw std::runtime_error("Failed to open potential file: " + fnm);
    Save(*ofs);
    if (!close_ofile(ofs)) throw std::runtime_error("Failed to write potential file: " + fnm);
}

void Potentials::LJ612::Save(std::ostream &os)
{
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << "pot:pottype lj612\n";
    os << "pot:ntypes " << ntypes << "\n";
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            std::string ff = std::to_string(i) + "-" + std::to_string(j);
            os << "pot:" << ff << " " << eps(i, j) << " " << sig(i, j) << "\n";
        }
    }
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            std::string ff = std::to_string(i) + "-" + std::to_string(j);
            os << "pot:cut:" << ff << " " << cutoffs(i, j) << " \n";
        }
    }
    for (int i = 0; i < ntypes; ++i) {
        std::string ff = std::to_string(i);
        os << "pot:ele:" << ff << " " << elemoffset(i) << "\n";
    }
    if (!os) throw std::runtime_error("Failed to write LJ612 potential stream");
}

void Potentials::LJ612::train_init()
{
    int ct = 0;
    mapgid2oid.clear();
    mapoid2gid.clear();
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            mapgid2oid[ct] = std::make_pair(i, j);
            mapoid2gid[std::make_pair(i, j)] = ct;
            mapoid2gid[std::make_pair(j, i)] = ct;
            ++ct;
        }
    }
    reg_L1_vector.resize(get_NL_coeffs_count(), 0);
    reg_L2_vector.resize(get_NL_coeffs_count(), 0);
}

void Potentials::LJ612::random_init()
{
    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);
    for (int i = 0; i < ntypes; ++i) {
        for (int j = i; j < ntypes; ++j) {
            eps(i, j) = std::abs(dis(gen)) * 2.0;
            sig(i, j) = std::abs(dis(gen)) * 2.0 + 1.0;
            eps(j, i) = eps(i, j);
            sig(j, i) = sig(i, j);
        }
    }
    convert_eps2lj();
    nl_inited = true;
}

void LJ612::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    if (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Analytical LJ612 cell Hessian requires a finite invertible cell");
    const Eigen::Matrix3d inverse = atom.cell_.inverse();
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    cellhess = Eigen::MatrixXd::Zero(9, 9);
    lrhess = Eigen::MatrixXd::Zero(9, 3 * atom.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            if (!(nbh.distances_[j] < cutoffs(nbh.centertype, nbh.chemmap_[j]))) continue;
            const int source = nbh.indices_[j];
            const Eigen::Matrix3d curvature = .5 * pair_curvature(
                nbh.centertype, nbh.chemmap_[j], nbh.distances_[j], nbh.pos_[j]);
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

void LJ612::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}
void LJ612::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec) {
    getEFSHessianCellSymmOMP(atom, hessian, cellhess, lrhess, symprec);
}

void Potentials::LJ612::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::LJ612::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::LJ612::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian) {
    getEFSHessian(atom, nbhs, hessian);
}
