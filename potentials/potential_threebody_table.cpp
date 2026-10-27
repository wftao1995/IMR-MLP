// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_threebody_table.h"
#include "base/neighborhood.h"
#include "utils/compressed_stream.h"
#include <iomanip>
#include <limits>
#include <sstream>
#include <cmath>
#include <algorithm>

using namespace Potentials;

ThreebodyTable::ThreebodyTable(const std::string &filename,
                               const std::map<std::string, int> &typemap) {
    load_file(filename, typemap);
}

void ThreebodyTable::load_file(const std::string &file,
                                const std::map<std::string, int> &typemap) {
    auto ifs = open_ifile(file);
    if (!ifs)
        throw std::runtime_error("ThreebodyTable: cannot open " + file);

    load_stream(*ifs, typemap, file);
}

void ThreebodyTable::load_stream(std::istream &is,
                             const std::map<std::string, int> &typemap,
                             const std::string &source) {
    std::string buffer;
    do {
        if (!getline(is, buffer))
            throw std::runtime_error("ThreebodyTable: missing element header in " + source);
    } while (buffer.empty() || buffer[0] == '#');

    // Parse element list
    std::istringstream iss(buffer);
    std::vector<std::string> elenames;
    std::string tok;
    while (iss >> tok) elenames.push_back(tok);
    if (elenames.empty()) throw std::runtime_error("ThreebodyTable: empty element header");

    const int file_types = elenames.size();
    nelements = typemap.empty() ? file_types : static_cast<int>(typemap.size());
    species_count = nelements;
    std::vector<int> file_index(nelements, -1);
    for (int i = 0; i < file_types; ++i) {
        if (std::find(elenames.begin(), elenames.begin() + i, elenames[i]) != elenames.begin() + i)
            throw std::runtime_error("ThreebodyTable: duplicate element " + elenames[i]);
        if (typemap.empty()) file_index[i] = i;
        else if (auto it = typemap.find(elenames[i]); it != typemap.end()) {
            const int id = it->second;
            if (id < 0 || id >= nelements || file_index[id] != -1)
                throw std::runtime_error("ThreebodyTable: invalid species indices in type map");
            file_index[id] = i;
        }
    }
    if (std::find(file_index.begin(), file_index.end(), -1) != file_index.end())
        throw std::runtime_error("ThreebodyTable: type map contains an element absent from the table");
    element_names.resize(nelements);
    for (int i = 0; i < nelements; ++i)
        element_names[i] = elenames[file_index[i]];

    // Read grid parameters
    is >> tok >> nr_points;
    is >> tok >> ntheta_points;
    is >> tok >> rmin >> tok >> rmax;
    is >> tok >> theta_min >> tok >> theta_max;

    if (!is || nr_points < 2 || ntheta_points < 2 ||
        !std::isfinite(rmin) || !std::isfinite(rmax) || rmin <= 0 || rmax <= rmin ||
        !std::isfinite(theta_min) || !std::isfinite(theta_max) || theta_max <= theta_min)
        throw std::runtime_error("ThreebodyTable: invalid radial or angular grid in " + source);

    dr = (rmax - rmin) / (nr_points - 1);
    dtheta = (theta_max - theta_min) / (ntheta_points - 1);
    inv_dr = 1.0 / dr;
    inv_dtheta = 1.0 / dtheta;
    cutoff = rmax;
    cutoffsq = cutoff * cutoff;

    // Read tables: each triplet has nr_points * ntheta_points values
    // stored row-major (r slow, theta fast)
    const int file_triplets = file_types * (file_types + 1) * (file_types + 2) / 6;
    std::vector<Eigen::MatrixXd> file_tables(file_triplets,
                                           Eigen::MatrixXd::Zero(nr_points, ntheta_points));
    for (int t = 0; t < file_triplets; ++t) {
        for (int ir = 0; ir < nr_points; ++ir) {
            for (int it = 0; it < ntheta_points; ++it) {
                if (!(is >> file_tables[t](ir, it)) || !std::isfinite(file_tables[t](ir, it)))
                    throw std::runtime_error("ThreebodyTable: missing or nonfinite value in " + source);
            }
        }
    }
    tables.resize(nelements * (nelements + 1) * (nelements + 2) / 6);
    for (int i = 0; i < nelements; ++i)
        for (int j = 0; j <= i; ++j)
            for (int k = 0; k <= j; ++k)
                tables[get_triplet_id(i, j, k)] = std::move(
                    file_tables[get_triplet_id(file_index[i], file_index[j], file_index[k])]);

    cutoffs.resize(nelements, nelements);
    cutoffsqs.resize(nelements, nelements);
    r_mins.resize(nelements, nelements);
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            cutoffs(i, j) = rmax;
            cutoffsqs(i, j) = rmax * rmax;
            r_mins(i, j) = rmin;
        }
    }
    nl_inited = true;
}

void ThreebodyTable::eval_3body(int tid, double rij, double rik, double theta,
                                 double &val,
                                 double &dV_drij, double &dV_drik,
                                 double &dV_dtheta,
                                 double *dV_drij_dtheta, double *dV_drik_dtheta) const {
    // The 2D table specifies equal bond lengths. Extend it symmetrically as
    // V(rij, rik, theta) = (T(rij, theta) + T(rik, theta)) / 2.

    // Clamp to grid bounds
    double r1 = (rij < rmin) ? rmin : (rij > rmax ? rmax : rij);
    double r2 = (rik < rmin) ? rmin : (rik > rmax ? rmax : rik);
    double th = (theta < theta_min) ? theta_min :
                (theta > theta_max ? theta_max : theta);

    double fr = (r1 - rmin) * inv_dr;
    double ft = (th - theta_min) * inv_dtheta;
    int ir = static_cast<int>(fr);
    int it = static_cast<int>(ft);
    if (ir >= nr_points - 1) ir = nr_points - 2;
    if (it >= ntheta_points - 1) it = ntheta_points - 2;
    if (ir < 0) ir = 0;
    if (it < 0) it = 0;
    double u = fr - ir;
    double v = ft - it;

    const auto &T = tables[tid];
    double f00 = T(ir,     it);
    double f10 = T(ir + 1, it);
    double f01 = T(ir,     it + 1);
    double f11 = T(ir + 1, it + 1);

    const double value1 = (1 - u) * (1 - v) * f00 + u * (1 - v) * f10
        + (1 - u) * v * f01 + u * v * f11;

    double df_dr1 = ((1 - v) * (f10 - f00) + v * (f11 - f01)) * inv_dr;
    const double df_dt1 = ((1 - u) * (f01 - f00) + u * (f11 - f10)) * inv_dtheta;

    // Evaluate df/dr2 by interpolating on the r-dimension at the rik position.
    // The table V(r, theta) is symmetric in r1, r2.
    double fr2 = (r2 - rmin) * inv_dr;
    int ir2 = static_cast<int>(fr2);
    if (ir2 >= nr_points - 1) ir2 = nr_points - 2;
    if (ir2 < 0) ir2 = 0;
    double u2 = fr2 - ir2;
    double g00 = T(ir2,     it);
    double g10 = T(ir2 + 1, it);
    double g01 = T(ir2,     it + 1);
    double g11 = T(ir2 + 1, it + 1);
    const double value2 = (1 - u2) * (1 - v) * g00 + u2 * (1 - v) * g10
        + (1 - u2) * v * g01 + u2 * v * g11;
    val = 0.5 * (value1 + value2);
    double df_dt = 0.5 * (df_dt1 +
        ((1 - u2) * (g01 - g00) + u2 * (g11 - g10)) * inv_dtheta);
    double df_dr2 = ((1 - v) * (g10 - g00) + v * (g11 - g01)) * inv_dr;

    // If rij or rik was clamped, derivatives in those directions are zero
    if (rij < rmin || rij > rmax) df_dr1 = 0.0;
    if (rik < rmin || rik > rmax) df_dr2 = 0.0;
    if (theta < theta_min || theta > theta_max) df_dt = 0.0;

    dV_drij = 0.5 * df_dr1;
    dV_drik = 0.5 * df_dr2;
    dV_dtheta = df_dt;
    const bool angle_active = theta >= theta_min && theta <= theta_max;
    if (dV_drij_dtheta)
        *dV_drij_dtheta = angle_active && rij >= rmin && rij <= rmax
            ? .5 * (f11-f10-f01+f00) * inv_dr * inv_dtheta : 0.;
    if (dV_drik_dtheta)
        *dV_drik_dtheta = angle_active && rik >= rmin && rik <= rmax
            ? .5 * (g11-g10-g01+g00) * inv_dr * inv_dtheta : 0.;
}

Eigen::Matrix<double,6,6> ThreebodyTable::triplet_curvature(
    const Neighborhood &n, int j, int k) const {
    using Vector6 = Eigen::Matrix<double,6,1>;
    using Matrix6 = Eigen::Matrix<double,6,6>;
    const Eigen::Matrix3d identity = Eigen::Matrix3d::Identity();
    const double r = n.distances_[j], s = n.distances_[k];
    const Eigen::Vector3d u = n.pos_[j].transpose()/r;
    const Eigen::Vector3d v = n.pos_[k].transpose()/s;
    double c = n.pos_[j].dot(n.pos_[k])/(r*s);
    c = std::clamp(c,-1.,1.);
    const double theta = std::acos(c), sine = std::sin(theta);
    double value, first_r, first_s, first_theta, mixed_r, mixed_s;
    eval_3body(get_triplet_id(n.centertype,n.chemmap_[j],n.chemmap_[k]),
        r,s,theta,value,first_r,first_s,first_theta,&mixed_r,&mixed_s);
    Vector6 gr = Vector6::Zero(), gs = Vector6::Zero();
    gr.head<3>() = u; gs.tail<3>() = v;
    Matrix6 local = Matrix6::Zero();
    local.topLeftCorner<3,3>() = first_r*(identity-u*u.transpose())/r;
    local.bottomRightCorner<3,3>() = first_s*(identity-v*v.transpose())/s;
    // Match the existing small-sine force branch. Grid/clamp
    // boundaries and collinear angles need not have a classical Hessian.
    if (sine > 1e-12) {
        Vector6 gc;
        gc.head<3>() = (v-c*u)/r;
        gc.tail<3>() = (u-c*v)/s;
        Matrix6 hc;
        hc.topLeftCorner<3,3>() = (-v*u.transpose()-u*v.transpose()
            -c*identity+3*c*u*u.transpose())/(r*r);
        hc.bottomRightCorner<3,3>() = (-u*v.transpose()-v*u.transpose()
            -c*identity+3*c*v*v.transpose())/(s*s);
        hc.topRightCorner<3,3>() = (identity-v*v.transpose()-u*u.transpose()
            +c*u*v.transpose())/(r*s);
        hc.bottomLeftCorner<3,3>() = hc.topRightCorner<3,3>().transpose();
        const Vector6 gt = -gc/sine;
        const Matrix6 ht = -hc/sine-c/(sine*sine*sine)*gc*gc.transpose();
        local += first_theta*ht + mixed_r*(gr*gt.transpose()+gt*gr.transpose())
            + mixed_s*(gs*gt.transpose()+gt*gs.transpose());
    }
    return local;
}

void ThreebodyTable::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    if (direction.size() != 3*atom.size()) GPERROR("Hessian-vector direction has the wrong size");
    if (nbhs.size() != atom.size()) GPERROR("Hessian-vector neighborhoods have the wrong size");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    using Vector6 = Eigen::Matrix<double,6,1>;
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &n = nbhs[i];
        for (int j = 0; j < n.nbhcount_; ++j) {
            if (n.distances_[j] >= rmax) continue;
            for (int k = j+1; k < n.nbhcount_; ++k) {
                if (n.distances_[k] >= rmax) continue;
                const auto local = triplet_curvature(n, j, k);
                Vector6 displacement;
                displacement.head<3>() = direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i);
                displacement.tail<3>() = direction.segment<3>(3*n.indices_[k])-direction.segment<3>(3*i);
                const Vector6 contribution = (local*displacement)/3.;
                result.segment<3>(3*n.indices_[j]) += contribution.head<3>();
                result.segment<3>(3*n.indices_[k]) += contribution.tail<3>();
                result.segment<3>(3*i) -= contribution.head<3>()+contribution.tail<3>();
            }
        }
    }
    product = std::move(result);
}

void ThreebodyTable::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    assemble_hessian(atom,nbhs,hessian,nullptr,nullptr);
}

void ThreebodyTable::assemble_hessian(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian,
    Eigen::MatrixXd *cellhess,Eigen::MatrixXd *lrhess) {
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    if (cellhess) {
        inverse = atom.cell_.inverse();
        *cellhess = Eigen::MatrixXd::Zero(9,9);
        *lrhess = Eigen::MatrixXd::Zero(9,3*atom.size());
    }
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &n = nbhs[i];
        for (int j = 0; j < n.nbhcount_; ++j) {
            if (n.distances_[j] >= rmax) continue;
            for (int k = j+1; k < n.nbhcount_; ++k) {
                if (n.distances_[k] >= rmax) continue;
                const auto local = triplet_curvature(n, j, k);
                const int sources[2] = {n.indices_[j], n.indices_[k]};
                // Scatter each relative-bond block through x_source - x_i.
                // Coincident source indices and periodic self images use the
                // same signs, rather than assuming three distinct atoms.
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b) {
                        const Eigen::Matrix3d block = local.block<3,3>(3*a, 3*b)/3.;
                        hessian(sources[a], sources[b]) += block;
                        hessian(i, sources[b]) -= block;
                        hessian(sources[a], i) -= block;
                        hessian(i, i) += block;
                    }
                if (cellhess) {
                    Eigen::Matrix<double,6,9> affine = Eigen::Matrix<double,6,9>::Zero();
                    const int bonds[2] = {j,k};
                    for (int bond=0;bond<2;++bond) {
                        const Eigen::RowVector3d fractional = n.pos_[bonds[bond]]*inverse;
                        for (int a=0;a<3;++a)
                            for (int b=0;b<3;++b)
                                affine(3*bond+b,3*a+b) = fractional(a);
                    }
                    *cellhess += affine.transpose()*local*affine/3.;
                    const Eigen::Matrix<double,9,6> mixed = affine.transpose()*local/3.;
                    for (int bond=0;bond<2;++bond) {
                        lrhess->middleCols(3*sources[bond],3) += mixed.middleCols<3>(3*bond);
                        lrhess->middleCols(3*i,3) -= mixed.middleCols<3>(3*bond);
                    }
                }
            }
        }
    }
}

void ThreebodyTable::getEFSHessianCellOMP(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess) {
    if (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Analytical ThreebodyTable cell Hessian requires a finite invertible cell");
    assemble_hessian(atom,nbhs,hessian,&cellhess,&lrhess);
}
void ThreebodyTable::getEFSHessianCellSymmOMP(Atoms &atom,HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessianCellOMP(atom,nbhs,hessian,cellhess,lrhess);
}
void ThreebodyTable::getEFSHessianCellSymm(Atoms &atom,HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double symprec) {
    getEFSHessianCellSymmOMP(atom,hessian,cellhess,lrhess,symprec);
}

void ThreebodyTable::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}
void ThreebodyTable::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double) {
    getEFSHessian(atom, hessian);
}

void ThreebodyTable::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        int itype = nbh.centertype;
        int ncount = nbh.nbhcount_;

        for (int jj = 0; jj < ncount; ++jj) {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            double rij = nbh.distances_[jj];
            if (rij >= rmax) continue;

            for (int kk = jj + 1; kk < ncount; ++kk) {
                int ktype = nbh.chemmap_[kk];
                int kind = nbh.indices_[kk];
                double rik = nbh.distances_[kk];
                if (rik >= rmax) continue;

                // cos(theta) = r_ij_hat · r_ik_hat
                double cos_theta = nbh.pos_[jj].dot(nbh.pos_[kk]) / (rij * rik);
                if (cos_theta > 1.0) cos_theta = 1.0;
                if (cos_theta < -1.0) cos_theta = -1.0;
                double theta = std::acos(cos_theta);

                int tid = get_triplet_id(itype, jtype, ktype);
                double val, dV_drij, dV_drik, dV_dtheta;
                eval_3body(tid, rij, rik, theta, val, dV_drij, dV_drik, dV_dtheta);

                double third = 1.0 / 3.0;
                atom.energy_ += third * val;
                atom.site_energies_[ii] += third * val;

                // Convert dV/dtheta to dV/d(cos_theta) for analytical forces
                double sin_theta = std::sin(theta);
                double dV_dcosth = (sin_theta > 1e-12) ? -dV_dtheta / sin_theta : 0.0;

                // Forces from dV/drij and dV/drik
                double fp_rij = dV_drij;
                double fp_rik = dV_drik;

                // Angular force contributions: d(cos_theta)/d(pos)
                double inv_rij = 1.0 / rij;
                double inv_rik = 1.0 / rik;

                Eigen::Vector3d r_ij_hat = nbh.pos_[jj] * inv_rij;
                Eigen::Vector3d r_ik_hat = nbh.pos_[kk] * inv_rik;

                Eigen::Vector3d dcosth_drj = (r_ik_hat - cos_theta * r_ij_hat) * inv_rij;
                Eigen::Vector3d dcosth_drk = (r_ij_hat - cos_theta * r_ik_hat) * inv_rik;

                // Force on j from radial + angular
                Eigen::Vector3d fj = -r_ij_hat * fp_rij - dcosth_drj * dV_dcosth;
                // Force on k from radial + angular
                Eigen::Vector3d fk = -r_ik_hat * fp_rik - dcosth_drk * dV_dcosth;
                // Force on i = -(fj + fk)
                Eigen::Vector3d fi = -(fj + fk);

                fj *= third; fk *= third; fi *= third;

                atom.forces_[ii] += fi;
                atom.forces_[jind] += fj;
                atom.forces_[kind] += fk;

                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        atom.stress_(a, b) += fj(a) * nbh.pos_[jj](b)
                                            + fk(a) * nbh.pos_[kk](b);
                    }
                }
            }
        }
    }
}

void ThreebodyTable::Load(std::istream &is) {
    load_stream(is, {}, "stream");
}

void ThreebodyTable::Save(std::ostream &os) {
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < nelements; ++i)
        os << (i == 0 ? "" : " ") << element_names[i];
    os << "\nnr " << nr_points << " ntheta " << ntheta_points << "\n";
    os << "rmin " << rmin << " rmax " << rmax << "\n";
    os << "theta_min " << theta_min << " theta_max " << theta_max << "\n";
    for (const auto &table : tables) {
        for (int ir = 0; ir < nr_points; ++ir) {
            for (int it = 0; it < ntheta_points; ++it)
                os << (it == 0 ? "" : " ") << table(ir, it);
            os << "\n";
        }
    }
}

void Potentials::ThreebodyTable::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::ThreebodyTable::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::ThreebodyTable::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian) {
    getEFSHessian(atom, nbhs, hessian);
}
