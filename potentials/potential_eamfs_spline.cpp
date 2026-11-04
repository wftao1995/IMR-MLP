// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>
#include "potential_eamfs_spline.h"
#include "base/neighborhood.h"
#include "eam_spline_io.h"
#include "eam_spline_internal.h"
#include "utils/compressed_stream.h"

namespace Potentials
{
void EAMFsSpline::Load(const std::string &file, std::map<std::string, int> &typemap)
{
    auto ifs = open_ifile(file);
    if (!ifs)
        throw std::runtime_error("Cannot open EAM potential file: " + file);
    const auto header = EAMTable::read_header(*ifs, file, typemap);
    gpm_elements.clear();
    for (size_t i = 0; i < header.types.size(); ++i)
        if (header.types[i] >= 0)
            gpm_elements[header.types[i]] = header.names[i];
    nelements = header.species_count;
    Nr = header.nr;
    Nrho = header.nrho;
    dr = header.dr;
    drho = header.drho;
    cutoff = header.cutoff;
    rhomin = header.rhomin;
    rhomax = header.rhomax;
    is_he_format = header.he;
    rmax = dr * (Nr - 1);
    frho_spline.resize(nelements);
    rhor_spline.assign(nelements, std::vector<Eigen::MatrixXd>(nelements));
    z2r_spline.resize(nelements * (nelements + 1) / 2);
    for (size_t i = 0; i < header.types.size(); ++i)
    {
        EAMTable::read_element_header(*ifs, file);
        auto embedding = EAMTable::read_spline(*ifs, Nrho, drho, file);
        const int type = header.types[i];
        if (type >= 0)
            frho_spline[type] = std::move(embedding);
        // Each file section is a neighbor species; rows enumerate hosts.
        for (size_t j = 0; j < header.types.size(); ++j)
        {
            auto density = EAMTable::read_spline(*ifs, Nr, dr, file);
            const int host = header.types[j];
            if (type >= 0 && host >= 0)
                rhor_spline[type][host] = std::move(density);
        }
    }
    for (size_t i = 0; i < header.types.size(); ++i)
    {
        for (size_t j = 0; j <= i; ++j)
        {
            auto pair = EAMTable::read_spline(*ifs, Nr, dr, file);
            const int itype = header.types[i], jtype = header.types[j];
            if (itype >= 0 && jtype >= 0)
                z2r_spline[EAMTable::pair_index(itype, jtype)] = std::move(pair);
        }
    }
    if (ifs->bad())
        throw std::runtime_error("Failed to read EAM potential file: " + file);
    rdr = 1.0 / dr;
    rdrho = 1.0 / drho;
    species_count = nelements;
    cutoffsq = cutoff * cutoff;
    cutoffs = Eigen::MatrixXd::Constant(nelements, nelements, cutoff);
    cutoffsqs = Eigen::MatrixXd::Constant(nelements, nelements, cutoffsq);
    r_mins = Eigen::MatrixXd::Zero(nelements, nelements);
    nl_inited = true;
}

void EAMFsSpline::getEFS(Atoms &atom)
{
    Neighborhoods nbhs(atom, cutoff);
    getEFS(atom, nbhs);
}

// E_i = F\alpha (\sum rho_\beta (r_ij)) + \sum_{i > j} \phi_{\alpha,\beta}(r_ij)
// TODO: OMP PARALLIZE
void EAMFsSpline::getEFS(Atoms &atom, AnyNeighborhoods &nbhs)
{
    EAMSpline::compute_efs(atom, nbhs, add_efs, Nr, Nrho, rdr, rdrho, rmax, rhomin, rhomax,
                           frho_spline, z2r_spline,
                           [&](int neighbor, int host) -> const Eigen::MatrixXd &
                           {
                               return rhor_spline[neighbor][host];
                           });
}

void EAMFsSpline::getEFSOMP(Atoms &atom)
{
    Neighborhoods nbhs(atom, cutoff);
    getEFSOMP(atom, nbhs);
}

// E_i = F\alpha (\sum rho_\beta (r_ij)) + \sum_{i > j} \phi_{\alpha,\beta}(r_ij)
// TODO: OMP PARALLIZE
void EAMFsSpline::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs)
{
    EAMSpline::compute_efs_omp(atom, nbhs, add_efs, Nr, Nrho, rdr, rdrho, rmax, rhomin, rhomax,
                               frho_spline, z2r_spline,
                               [&](int neighbor, int host) -> const Eigen::MatrixXd &
                               {
                                   return rhor_spline[neighbor][host];
                               });
}

}

namespace Potentials
{
void EAMFsSpline::assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
                                   Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess)
{
    if (nbhs.size() != atom.size())
        GPERROR("Hessian neighborhoods have the wrong size");
    getEFS(atom, nbhs);
    EAMSpline::assemble(
        atom, nbhs, hessian, cellhess, lrhess,
        [&](int neighbor, int host, double r)
        {
            return EAMSpline::radial(rhor_spline[neighbor][host], Nr, rdr, rmax, r);
        },
        [&](int a, int b, double r)
        {
            return EAMSpline::pair_from_rphi(
                EAMSpline::radial(z2r_spline[EAMTable::pair_index(a, b)], Nr, rdr, rmax, r), r);
        },
        [&](int a, double rho)
        {
            return EAMSpline::embedding(frho_spline[a], Nrho, rdrho, rhomin, rhomax, rho);
        });
}

void EAMFsSpline::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian)
{
    assemble_hessian(atom, nbhs, hessian, nullptr, nullptr);
}

void EAMFsSpline::getEFSHessian(Atoms &atom, HessianMatrix &hessian)
{
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessian(atom, nbhs, hessian);
}

void EAMFsSpline::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double)
{
    getEFSHessian(atom, hessian);
}

void EAMFsSpline::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
                                       Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess)
{
    if (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Analytical spline EAM cell Hessian requires a finite invertible cell");
    assemble_hessian(atom, nbhs, hessian, &cellhess, &lrhess);
}

void EAMFsSpline::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
                                           Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess,
                                           double)
{
    Neighborhoods nbhs(atom, cutoff);
    getEFSHessianCellOMP(atom, nbhs, hessian, cellhess, lrhess);
}

void EAMFsSpline::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
                                        Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess,
                                        double symprec)
{
    getEFSHessianCellSymmOMP(atom, hessian, cellhess, lrhess, symprec);
}
}

namespace Potentials
{
void EAMFsSpline::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
                                            const Eigen::VectorXd &direction,
                                            Eigen::VectorXd &product)
{
    if (direction.size() != 3 * atom.size() || !direction.allFinite())
        GPERROR("Hessian-vector direction must be finite with size 3*N");
    if (nbhs.size() != atom.size())
        GPERROR("Hessian-vector neighborhoods have the wrong size");
    EAMSpline::apply(
        atom, nbhs, direction, product,
        [&](int neighbor, int host, double r)
        {
            return EAMSpline::radial(rhor_spline[neighbor][host], Nr, rdr, rmax, r);
        },
        [&](int a, int b, double r)
        {
            return EAMSpline::pair_from_rphi(
                EAMSpline::radial(z2r_spline[EAMTable::pair_index(a, b)], Nr, rdr, rmax, r), r);
        },
        [&](int a, double rho)
        {
            return EAMSpline::embedding(frho_spline[a], Nrho, rdrho, rhomin, rhomax, rho);
        });
}
}
