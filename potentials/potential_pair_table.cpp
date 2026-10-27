// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_pair_table.h"
#include "base/neighborhood.h"
#include "mathextra/spline.h"
#include "utils/compressed_stream.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <utility>

using namespace Potentials;

PairTable::PairTable(const std::string &filename,
                     const std::map<std::string, int> &typemap) {
    load_file(filename, typemap);
}

void PairTable::load_file(const std::string &file,
                           const std::map<std::string, int> &typemap) {
    auto ifs = open_ifile(file);
    if (!ifs)
        throw std::runtime_error("PairTable: cannot open " + file);

    load_stream(*ifs, typemap, file);
}

void PairTable::load_stream(std::istream &is,
                             const std::map<std::string, int> &typemap,
                             const std::string &source) {
    std::string buffer;
    // Element names follow the order of the pair columns in the file.
    do {
        if (!std::getline(is, buffer))
            throw std::runtime_error("PairTable: missing element header in " + source);
    } while (buffer.empty() || buffer[0] == '#');
    std::istringstream names(buffer);
    std::vector<std::string> elements;
    std::string name;
    while (names >> name) elements.push_back(name);
    if (elements.empty()) throw std::runtime_error("PairTable: empty element header");
    const int file_types = elements.size();
    nelements = typemap.empty() ? file_types : static_cast<int>(typemap.size());
    species_count = nelements;
    std::vector<int> file_index(nelements, -1);
    for (int i = 0; i < file_types; ++i) {
        if (std::find(elements.begin(), elements.begin() + i, elements[i]) != elements.begin() + i)
            throw std::runtime_error("PairTable: duplicate element " + elements[i]);
        if (typemap.empty()) file_index[i] = i;
        else if (auto it = typemap.find(elements[i]); it != typemap.end()) {
            const int id = it->second;
            if (id < 0 || id >= nelements || file_index[id] != -1)
                throw std::runtime_error("PairTable: invalid species indices in type map");
            file_index[id] = i;
        }
    }
    if (std::find(file_index.begin(), file_index.end(), -1) != file_index.end())
        throw std::runtime_error("PairTable: type map contains an element absent from the table");
    element_names.resize(nelements);
    for (int i = 0; i < nelements; ++i)
        element_names[i] = elements[file_index[i]];

    int count = 0;
    while (std::getline(is, buffer)) {
        if (buffer.empty() || buffer[0] == '#') continue;
        std::istringstream row(buffer);
        std::string tag;
        if (!(row >> tag >> count) || tag != "N")
            throw std::runtime_error("PairTable: expected N followed by the grid point count");
        break;
    }
    if (count < 3)
        throw std::runtime_error("PairTable: at least 3 grid points are required");
    npoints = count - 1; // Number of intervals; the file has exactly count rows.
    const int file_pairs = file_types * (file_types + 1) / 2;
    std::vector<double> radii(count);
    std::vector<std::vector<double>> values(file_pairs, std::vector<double>(count));
    for (int m = 0; m < count; ++m) {
        int row_index;
        if (!(is >> row_index >> radii[m]) || !std::isfinite(radii[m]))
            throw std::runtime_error("PairTable: missing or invalid grid row in " + source);
        for (auto &column : values) {
            if (!(is >> column[m]) || !std::isfinite(column[m]))
                throw std::runtime_error("PairTable: missing or nonfinite energy in " + source);
        }
    }
    std::string extra;
    if (is >> extra)
        throw std::runtime_error("PairTable: extra data after declared grid rows in " + source);
    rmin = radii.front();
    rmax = radii.back();
    dr = (rmax - rmin) / npoints;
    if (!(rmin > 0) || !std::isfinite(dr) || dr <= 0)
        throw std::runtime_error("PairTable: expected 0 < rmin < rmax");
    for (int m = 1; m < count; ++m) {
        const double expected = rmin + m * dr;
        if (radii[m] <= radii[m - 1] ||
            std::abs(radii[m] - expected) > 1e-6 * std::max(1.0, std::abs(expected)))
            throw std::runtime_error("PairTable: grid must be increasing and uniformly spaced");
    }
    inv_dr = 1.0 / dr;
    cutoff = rmax;
    cutoffsq = cutoff * cutoff;

    pair_splines.resize(nelements * (nelements + 1) / 2);
    std::vector<double> shifted(count + 1, 0.0);
    for (int j = 0; j < nelements; ++j) {
        for (int i = 0; i <= j; ++i) {
            const auto &column = values[get_pairid(file_index[i], file_index[j])];
            std::copy(column.begin(), column.end(), shifted.begin() + 1);
            Spline::build_cubic_spline(count, dr, shifted, pair_splines[get_pairid(i, j)]);
        }
    }

    // Set per-pair cutoffs
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

void PairTable::Load(std::istream &is) {
    load_stream(is, {}, "stream");
}

void PairTable::Save(std::ostream &os) {
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (int i = 0; i < nelements; ++i)
        os << (i == 0 ? "" : " ") << element_names[i];
    os << "\nN " << npoints + 1 << "\n";
    for (int m = 0; m <= npoints; ++m) {
        os << m << " " << (m == npoints ? rmax : rmin + m * dr);
        for (const auto &spline : pair_splines)
            os << " " << spline(m + 1, 6);
        os << "\n";
    }
}

void PairTable::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        int itype = nbh.centertype;
        int iind = ii;

        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            double r = nbh.distances_[jj];
            if (r >= rmax) continue;

            int pid = get_pairid(itype, jtype);
            double val, deriv;
            Spline::eval_cubic_spline(r, rmin, inv_dr, pair_splines[pid],
                                       npoints, val, deriv);

            atom.energy_ += 0.5 * val;
            atom.site_energies_[ii] += 0.5 * val;

            double fpair = 0.5 * deriv / r;
            atom.forces_[iind](0) += nbh.pos_[jj](0) * fpair;
            atom.forces_[iind](1) += nbh.pos_[jj](1) * fpair;
            atom.forces_[iind](2) += nbh.pos_[jj](2) * fpair;
            atom.forces_[jind](0) -= nbh.pos_[jj](0) * fpair;
            atom.forces_[jind](1) -= nbh.pos_[jj](1) * fpair;
            atom.forces_[jind](2) -= nbh.pos_[jj](2) * fpair;

            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= nbh.pos_[jj](a) * fpair *
                                           nbh.pos_[jj](b);
                }
            }
        }
    }
}

Eigen::Matrix3d PairTable::pair_curvature(
    int itype, int jtype, double distance,
    const Eigen::RowVector3d &displacement) const {
    // Preserve the evaluator's constant lower branch and hard upper cutoff.
    if (distance <= rmin || distance >= rmax) return Eigen::Matrix3d::Zero();
    // At an internal knot, select the same right interval as the evaluator.
    double p = (distance - rmin) * inv_dr;
    int m = static_cast<int>(p);
    if (m >= npoints) return Eigen::Matrix3d::Zero();
    p -= m;
    ++m;
    p = p > 1.0 ? 1.0 : p;
    const auto &spline = pair_splines[get_pairid(itype, jtype)];
    double value, first;
    Spline::eval_cubic_spline(distance, rmin, inv_dr, spline, npoints, value, first);
    // Columns 0/1 already contain one inverse grid-spacing factor.
    const double second = (2.0 * spline(m, 0) * p + spline(m, 1)) * inv_dr;
    const double transverse = first / distance;
    return transverse * Eigen::Matrix3d::Identity()
         + ((second - transverse) / (distance * distance))
             * displacement.transpose() * displacement;
}

void PairTable::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs,
                             HessianMatrix &hessian) {
    if (nbhs.size() != atom.size())
        GPERROR("Hessian neighborhoods have the wrong size");
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            const int atj = nbh.indices_[j];
            if (atj == i || nbh.distances_[j] >= rmax) continue;
            // Full neighborhoods visit each physical pair from both ends.
            const Eigen::Matrix3d curvature = 0.5 * pair_curvature(
                nbh.centertype, nbh.chemmap_[j], nbh.distances_[j], nbh.pos_[j]);
            hessian(i, i) += curvature;
            hessian(atj, atj) += curvature;
            hessian(i, atj) -= curvature;
            hessian(atj, i) -= curvature;
        }
    }
}

void PairTable::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}

void PairTable::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double) {
    getEFSHessian(atom, hessian);
}

void PairTable::applyHessianVectorProduct(
    Atoms &atom, AnyNeighborhoods &nbhs, const Eigen::VectorXd &direction,
    Eigen::VectorXd &product) {
    if (direction.size() != 3 * atom.size())
        GPERROR("Hessian-vector direction has the wrong size");
    if (nbhs.size() != atom.size())
        GPERROR("Hessian-vector neighborhoods have the wrong size");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            const int atj = nbh.indices_[j];
            if (atj == i || nbh.distances_[j] >= rmax) continue;
            const Eigen::Vector3d value = 0.5 * pair_curvature(
                nbh.centertype, nbh.chemmap_[j], nbh.distances_[j], nbh.pos_[j])
                * (direction.segment<3>(3 * i) - direction.segment<3>(3 * atj));
            result.segment<3>(3 * i) += value;
            result.segment<3>(3 * atj) -= value;
        }
    }
    product = std::move(result);
}

void PairTable::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    if (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Analytical PairTable cell Hessian requires a finite invertible cell");
    const Eigen::Matrix3d inverse = atom.cell_.inverse();
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    cellhess = Eigen::MatrixXd::Zero(9, 9);
    lrhess = Eigen::MatrixXd::Zero(9, 3 * atom.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &nbh = nbhs[i];
        for (int j = 0; j < nbh.nbhcount_; ++j) {
            if (nbh.distances_[j] >= rmax) continue;
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

void PairTable::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}
void PairTable::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double symprec) {
    getEFSHessianCellSymmOMP(atom, hessian, cellhess, lrhess, symprec);
}

void Potentials::PairTable::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::PairTable::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void Potentials::PairTable::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian) {
    getEFSHessian(atom, nbhs, hessian);
}
