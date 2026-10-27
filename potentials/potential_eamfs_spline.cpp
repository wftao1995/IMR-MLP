// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>
#include "potential_eamfs_spline.h"
#include "base/neighborhood.h"
#include "eam_spline_io.h"
#include "eam_spline_curvature.h"
#include "utils/compressed_stream.h"
namespace Potentials {
    void EAMFsSpline::Load(const std::string &file, std::map<std::string, int> &typemap) {
        auto ifs = open_ifile(file);
        if (!ifs) throw std::runtime_error("Cannot open EAM potential file: " + file);
        const auto header = EAMTable::read_header(*ifs, file, typemap);
        gpm_elements.clear();for(size_t i=0;i<header.types.size();++i)if(header.types[i]>=0)gpm_elements[header.types[i]]=header.names[i];
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
        for (size_t i = 0; i < header.types.size(); ++i) {
            EAMTable::read_element_header(*ifs, file);
            auto embedding = EAMTable::read_spline(*ifs, Nrho, drho, file);
            const int type = header.types[i];
            if (type >= 0) frho_spline[type] = std::move(embedding);
            // Each file section is a neighbor species; rows enumerate hosts.
            for (size_t j = 0; j < header.types.size(); ++j) {
                auto density = EAMTable::read_spline(*ifs, Nr, dr, file);
                const int host = header.types[j];
                if (type >= 0 && host >= 0) rhor_spline[type][host] = std::move(density);
            }
        }
        for (size_t i = 0; i < header.types.size(); ++i) {
            for (size_t j = 0; j <= i; ++j) {
                auto pair = EAMTable::read_spline(*ifs, Nr, dr, file);
                const int itype = header.types[i], jtype = header.types[j];
                if (itype >= 0 && jtype >= 0)
                    z2r_spline[EAMTable::pair_index(itype, jtype)] = std::move(pair);
            }
        }
        if (ifs->bad()) throw std::runtime_error("Failed to read EAM potential file: " + file);
        rdr = 1.0 / dr;
        rdrho = 1.0 / drho;
        species_count = nelements;
        cutoffsq = cutoff * cutoff;
        cutoffs = Eigen::MatrixXd::Constant(nelements, nelements, cutoff);
        cutoffsqs = Eigen::MatrixXd::Constant(nelements, nelements, cutoffsq);
        r_mins = Eigen::MatrixXd::Zero(nelements, nelements);
        nl_inited = true;
    }

    void EAMFsSpline::getEFS(Atoms &atom) {
        Neighborhoods nbhs(atom, cutoff);
        getEFS(atom, nbhs);
    }
    // E_i = F\alpha (\sum rho_\beta (r_ij)) + \sum_{i > j} \phi_{\alpha,\beta}(r_ij)
    // TODO: OMP PARALLIZE
    void EAMFsSpline::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
        if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
        // compute EAM 
        // 1. compute rho
        double p;
        int m;
        static thread_local std::vector<double> rho, fp;
        rho.assign(atom.size(), 0);
        fp.assign(atom.size(), 0);
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            auto &nbh = nbhs[ii];
            int itype = nbh.centertype;
            int iind = ii;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int jtype = nbh.chemmap_[jj];
                int jind = nbh.indices_[jj];
                p = nbh.distances_[jj] * rdr + 1.0;
                m = static_cast<int>(p);
                m = __min(m, Nr - 1);
                p -= m;
                p = __min(p, 1.0);
                auto &coeff = rhor_spline[jtype][itype];
                rho[iind] += ((coeff(m, 3) * p + coeff(m, 4)) * p + coeff(m, 5)) * p + coeff(m, 6);
                // auto &coeff2 = rhor_spline[itype];
                // rho[jind] += ((coeff2(m, 3) * p + coeff2(m, 4)) * p + coeff2(m, 5)) * p + coeff2(m, 6);
            }
        }  
        // compute embedding energy
        double phi = 0;
        for (int ii = 0; ii < nbhs.size(); ++ii) {
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
            if (rho[indi] > rhomax) {
                phi += fp[indi] * (rho[indi] - rhomax);
            }
            atom.energy_ += phi;
            atom.site_energies_[indi] += phi;
        }
        // compute force
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            auto &nbh = nbhs[ii];
            int iind = ii;
            int itype = nbh.centertype;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int jtype = nbh.chemmap_[jj];
                int jind = nbh.indices_[jj];
                int rtype = EAMTable::pair_index(itype, jtype);
                double r = nbh.distances_[jj];
                p = r * rdr + 1.0;
                m = static_cast<int> (p);
                m = __min(m, Nr - 1);
                p -= m;
                p = __min(p, 1.0);
                auto &coeff = rhor_spline[itype][jtype];
                double rhoip = (coeff(m, 0) * p + coeff(m, 1)) * p + coeff(m, 2);
                auto &coeff2 = rhor_spline[jtype][itype];
                double rhojp = (coeff2(m, 0) * p + coeff2(m, 1)) * p + coeff2(m, 2);
                auto &coeff3 = z2r_spline[rtype];
                double z2p = (coeff3(m, 0) * p + coeff3(m, 1)) * p + coeff3(m, 2);
                double z2 = ((coeff3(m, 3) * p + coeff3(m ,4))* p + coeff3(m, 5)) * p + coeff3(m, 6);

                // The existing energy clamps density and r*phi to their last
                // grid values beyond rmax. Differentiate that same branch:
                // constant density and a 1/r pair tail.
                if (r > rmax) {
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

                for (int a = 0; a < 3; a++) {
                    for (int b = 0; b < 3; b++) {
                        atom.stress_(a, b) += nbh.pos_[jj](a) * fpair * nbh.pos_[jj](b);
                    }
                }
            }
        } 
    }

    void EAMFsSpline::getEFSOMP(Atoms &atom) {
        Neighborhoods nbhs(atom, cutoff);
        getEFSOMP(atom, nbhs);
    }
    // E_i = F\alpha (\sum rho_\beta (r_ij)) + \sum_{i > j} \phi_{\alpha,\beta}(r_ij)
    // TODO: OMP PARALLIZE
    void EAMFsSpline::getEFSOMP(Atoms &atom, AnyNeighborhoods &nbhs) {
        if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

        // compute EAM
        // 1. compute rho

        std::vector<double> rho(atom.size(), 0);
        std::vector<double> fp(atom.size(), 0);
        #pragma omp parallel for
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            double p;
            int m;
            auto &nbh = nbhs[ii];
            int itype = nbh.centertype;
            int iind = ii;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int jtype = nbh.chemmap_[jj];
                int jind = nbh.indices_[jj];
                p = nbh.distances_[jj] * rdr + 1.0;
                m = static_cast<int>(p);
                m = __min(m, Nr - 1);
                p -= m;
                p = __min(p, 1.0);
                auto &coeff = rhor_spline[jtype][itype];
                rho[iind] += ((coeff(m, 3) * p + coeff(m, 4)) * p + coeff(m, 5)) * p + coeff(m, 6);
                // auto &coeff2 = rhor_spline[itype];
                // rho[jind] += ((coeff2(m, 3) * p + coeff2(m, 4)) * p + coeff2(m, 5)) * p + coeff2(m, 6);
            }
        }  
        // compute embedding energy
        #pragma omp parallel for
        for (int ii = 0; ii < nbhs.size(); ++ii) {
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
            if (rho[indi] > rhomax) {
                phi += fp[indi] * (rho[indi] - rhomax);
            }
            #pragma omp critical
            atom.energy_ += phi;
            atom.site_energies_[indi] += phi;
        }
        // compute force
        #pragma omp parallel for
        for (int ii = 0; ii < nbhs.size(); ++ii) {
            int m;
            double p;
            auto &nbh = nbhs[ii];
            int iind = ii;
            int itype = nbh.centertype;
            for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
                int jtype = nbh.chemmap_[jj];
                int jind = nbh.indices_[jj];
                int rtype = EAMTable::pair_index(itype, jtype);
                double r = nbh.distances_[jj];
                p = r * rdr + 1.0;
                m = static_cast<int> (p);
                m = __min(m, Nr - 1);
                p -= m;
                p = __min(p, 1.0);
                auto &coeff = rhor_spline[itype][jtype];
                double rhoip = (coeff(m, 0) * p + coeff(m, 1)) * p + coeff(m, 2);
                auto &coeff2 = rhor_spline[jtype][itype];
                double rhojp = (coeff2(m, 0) * p + coeff2(m, 1)) * p + coeff2(m, 2);
                auto &coeff3 = z2r_spline[rtype];
                double z2p = (coeff3(m, 0) * p + coeff3(m, 1)) * p + coeff3(m, 2);
                double z2 = ((coeff3(m, 3) * p + coeff3(m ,4))* p + coeff3(m, 5)) * p + coeff3(m, 6);

                // The existing energy clamps density and r*phi to their last
                // grid values beyond rmax. Differentiate that same branch:
                // constant density and a 1/r pair tail.
                if (r > rmax) {
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

                    for (int a = 0; a < 3; a++) {
                        for (int b = 0; b < 3; b++) {
                            atom.stress_(a, b) += nbh.pos_[jj](a) * fpair * nbh.pos_[jj](b);
                        }
                    }
                }
            }
        } 
    }

}
namespace Potentials {
void EAMFsSpline::assemble_hessian(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian,
    Eigen::MatrixXd *cellhess,Eigen::MatrixXd *lrhess) {
    if(nbhs.size()!=atom.size())GPERROR("Hessian neighborhoods have the wrong size");
    getEFS(atom,nbhs);
    EAMSplineCurvature::assemble(atom,nbhs,hessian,cellhess,lrhess,
        [&](int neighbor,int host,double r){return EAMSplineCurvature::radial(rhor_spline[neighbor][host],Nr,rdr,rmax,r);},
        [&](int a,int b,double r){return EAMSplineCurvature::pair_from_rphi(
            EAMSplineCurvature::radial(z2r_spline[EAMTable::pair_index(a,b)],Nr,rdr,rmax,r),r);},
        [&](int a,double rho){return EAMSplineCurvature::embedding(frho_spline[a],Nrho,rdrho,rhomin,rhomax,rho);});
}
void EAMFsSpline::getEFSHessian(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian){assemble_hessian(atom,nbhs,hessian,nullptr,nullptr);}
void EAMFsSpline::getEFSHessian(Atoms &atom,HessianMatrix &hessian){Neighborhoods nbhs(atom,cutoff);getEFSHessian(atom,nbhs,hessian);}
void EAMFsSpline::getEFSHessianSymm(Atoms &atom,HessianMatrix &hessian,double){getEFSHessian(atom,hessian);}
void EAMFsSpline::getEFSHessianCellOMP(Atoms &atom,AnyNeighborhoods &nbhs,HessianMatrix &hessian,Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess){
    if(!atom.cell_.allFinite()||!atom.cell_.fullPivLu().isInvertible())GPERROR("Analytical spline EAM cell Hessian requires a finite invertible cell");
    assemble_hessian(atom,nbhs,hessian,&cellhess,&lrhess);
}
void EAMFsSpline::getEFSHessianCellSymmOMP(Atoms &atom,HessianMatrix &hessian,Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double){
    Neighborhoods nbhs(atom,cutoff);getEFSHessianCellOMP(atom,nbhs,hessian,cellhess,lrhess);
}
void EAMFsSpline::getEFSHessianCellSymm(Atoms &atom,HessianMatrix &hessian,Eigen::MatrixXd &cellhess,Eigen::MatrixXd &lrhess,double symprec){
    getEFSHessianCellSymmOMP(atom,hessian,cellhess,lrhess,symprec);
}
}

namespace Potentials {
void EAMFsSpline::applyHessianVectorProduct(Atoms &atom,AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction,Eigen::VectorXd &product) {
    if(direction.size()!=3*atom.size()||!direction.allFinite())GPERROR("Hessian-vector direction must be finite with size 3*N");
    if(nbhs.size()!=atom.size())GPERROR("Hessian-vector neighborhoods have the wrong size");
    EAMSplineCurvature::apply(atom,nbhs,direction,product,
        [&](int neighbor,int host,double r){return EAMSplineCurvature::radial(rhor_spline[neighbor][host],Nr,rdr,rmax,r);},
        [&](int a,int b,double r){return EAMSplineCurvature::pair_from_rphi(
            EAMSplineCurvature::radial(z2r_spline[EAMTable::pair_index(a,b)],Nr,rdr,rmax,r),r);},
        [&](int a,double rho){return EAMSplineCurvature::embedding(frho_spline[a],Nrho,rdrho,rhomin,rhomax,rho);});
}
}
