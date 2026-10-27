// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_eam_any.h"
#include "tools/chem.h"
#include "utils/compressed_stream.h"

void Potentials::PotEAMAny::CalcSiteEFS(Neighborhood& nbh, double& site_en, Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders)
{
    site_energy_ders.resize(nbh.nbhcount_, 3);
    site_energy_ders.setZero();
    int itype = nbh.centertype;
    site_en = 0;
    double rho_all = 0;
    buff_drho_dr.resize(nbh.nbhcount_);

    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double& rij = nbh.distances_[jj];
        int jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype))
            continue;
        const double rinv = 1.0 / rij;
        // pair energy
        double fpair = 0;
        double dfpair_dr = 0;
        func_pair_d(itype, jtype, rij, fpair, dfpair_dr);
        site_en += 0.5 * fpair;
        site_energy_ders(jj, 0) += dfpair_dr * nbh.pos_[jj](0) * rinv;
        site_energy_ders(jj, 1) += dfpair_dr * nbh.pos_[jj](1) * rinv;
        site_energy_ders(jj, 2) += dfpair_dr * nbh.pos_[jj](2) * rinv;
        // density
        double rho = 0;
        double drho_dr = 0;
        func_density_d(itype, jtype, rij, rho, drho_dr);
        rho_all += rho;
        buff_drho_dr[jj] = drho_dr;
    }
    if (rho_min > rho_all) [[unlikely]] {
        rho_min = rho_all;
    }
    if (rho_max < rho_all) [[unlikely]] {
        rho_max = rho_all;
    }
    // embeeding
    double femb = 0;
    double dfemb_drho = 0;
    func_emb_d(itype, rho_all, femb, dfemb_drho);
    // embeeding force
    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double& rij = nbh.distances_[jj];
        int jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype))
            continue;
        const double rinv = 1.0 / rij;
        double dfemb_dr = dfemb_drho * buff_drho_dr[jj];
        site_energy_ders(jj, 0) += dfemb_dr * nbh.pos_[jj](0) * rinv;
        site_energy_ders(jj, 1) += dfemb_dr * nbh.pos_[jj](1) * rinv;
        site_energy_ders(jj, 2) += dfemb_dr * nbh.pos_[jj](2) * rinv;
    }
    site_en += femb + species_coeffs[nbh.centertype];
}

void Potentials::PotEAMAny::CalcSiteE(Neighborhood& nbh, double& site_en)
{
    int itype = nbh.centertype;
    site_en = 0;
    double rho_all = 0;

    for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
        const double& rij = nbh.distances_[jj];
        int jtype = nbh.chemmap_[jj];
        if (rij > cutoffs(itype, jtype))
            continue;
        const double rinv = 1.0 / rij;
        // pair energy
        double fpair = func_pair(itype, jtype, rij);
        site_en += 0.5 * fpair;
        // density
        double rho = func_density(itype, jtype, rij);
        rho_all += rho;
    }
    site_en += func_emb(itype, rho_all) + species_coeffs[nbh.centertype];
}

void Potentials::PotEAMAny::OutputLAMMPSset(std::string outfile, int nr, int nrho, double rhomin, double rhomax)
{
    auto ofs = open_ofile(outfile);
    if (!ofs) return;
    std::string lat = "  fcc  ";
    *ofs << "Created by IMRMLP potential module\n";
    *ofs << "For LAMMPS\n";
    *ofs << "In setfl format\n";
    *ofs << "   " << species_count;
    if (type_name.empty()) {
        for (int i = 0; i < species_count; ++i) {
            *ofs << " " << Chemicals::chemical_symbols[i+1];
        }
    } else {
        for (auto &it : type_name) {
            *ofs << " " << it;
        }
    }
    *ofs << "\n";
    double delta_r = cutoff / static_cast<double>(nr - 1);
    double drho = (rhomax - rhomin) / static_cast<double>(nrho - 1);
    int mlrs = 5;
    *ofs << "  " << nrho << "  " << drho << "  " << nr << "  "
        << delta_r << "  " << cutoff << '\n';
    for (int i = 0; i < species_count; ++i) {
        // atomic number, mass, lattice constant, lattice type
        *ofs << "  " << type_name[i] << "  " << 5.0 << "  "
            << 5.0 << "  "
            << lat;
        // embedding function (Nrho values) 
        for (int ii = 0; ii < nrho; ++ii) {
            double rho = rhomin + drho * static_cast<double>(ii);
            double fembval = func_emb(i, rho);
            if (ii % mlrs == 0) {
                *ofs << '\n';
            }
            *ofs << fembval << " ";
        }
        // density functions
        if (eam_fs) {
            for (int j = 0; j < species_count; ++j) {
                for (int ii = 0; ii < nr; ++ii) {
                    double rcurr = delta_r * static_cast<double>(ii);
                    double fdenval = func_density(j, i, rcurr);
                    if (ii % mlrs == 0) {
                        *ofs << '\n';
                    }
                    *ofs << fdenval << " ";
                }
            }
        } else {
            for (int ii = 0; ii < nr; ++ii) {
                double rcurr = delta_r * static_cast<double>(ii);
                double fdenval = func_density(i, i, rcurr);
                if (ii % mlrs == 0) {
                    *ofs << '\n';
                }
                *ofs << fdenval << " ";
            }
        }
    }
    // pair functions
    for (int i = 0; i < species_count; ++i) {
        for (int j = 0; j <= i; ++j) {
            for (int ii = 0; ii < nr; ++ii) {
                double rcurr = delta_r * static_cast<double>(ii);
                double fpairval = func_pair(i, j, rcurr);
                if (ii % mlrs == 0) {
                    *ofs << '\n';
                }
                *ofs << fpairval << " ";
            }
        }
    }
}

void Potentials::PotEAMAny::getEFS(Atoms& atom, AnyNeighborhoods& nbhs)
{
    if (!add_efs)
        atom.resetEFS();
    else
        atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood& nbh = nbhs[ii];
        double buff_site_energy = 0;
        CalcSiteEFS(nbh, buff_site_energy, buff_site_energy_ders);
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;

        for (int i = 0; i < nbh.nbhcount_; ++i) {
            atom.forces_[ii] += buff_site_energy_ders.row(i);
            atom.forces_[nbh.indices_[i]] -= buff_site_energy_ders.row(i);
            for (int a = 0; a < 3; a++) {
                for (int b = 0; b < 3; b++) {
                    atom.stress_(a, b) -= buff_site_energy_ders(i, a) * nbh.pos_[i](b);
                }
            }
        }
    }
}

void Potentials::PotEAMAny::getE(Atoms& atom, AnyNeighborhoods& nbhs)
{
    if (!add_efs)
        atom.resetE();
    else
        atom.setEflag();

    for (int ii = 0; ii < nbhs.size(); ii++) {
        Neighborhood& nbh = nbhs[ii];
        double buff_site_energy = 0;
        CalcSiteE(nbh, buff_site_energy);
        // extract energy, force, virual, etc
        atom.energy_ += buff_site_energy;
        atom.site_energies_[ii] += buff_site_energy;
    }
}

// Stub implementations for virtual methods not overridden by EAMPoly
double Potentials::PotEAMAny::func_pair(int, int, double) {
    throw std::runtime_error("PotEAMAny::func_pair is abstract, use derived class");
}
void Potentials::PotEAMAny::func_pair_d(int, int, double, double&, double&) {
    throw std::runtime_error("PotEAMAny::func_pair_d is abstract, use derived class");
}
double Potentials::PotEAMAny::func_emb(int, double) {
    throw std::runtime_error("PotEAMAny::func_emb is abstract, use derived class");
}
void Potentials::PotEAMAny::func_emb_d(int, double, double&, double&) {
    throw std::runtime_error("PotEAMAny::func_emb_d is abstract, use derived class");
}
double Potentials::PotEAMAny::func_density(int, int, double) {
    throw std::runtime_error("PotEAMAny::func_density is abstract, use derived class");
}
void Potentials::PotEAMAny::func_density_d(int, int, double, double&, double&) {
    throw std::runtime_error("PotEAMAny::func_density_d is abstract, use derived class");
}
void Potentials::PotEAMAny::Load(std::istream&) {
    throw std::runtime_error("PotEAMAny::Load(istream) is abstract, use derived class");
}
void Potentials::PotEAMAny::Save(std::ostream&) {
    throw std::runtime_error("PotEAMAny::Save(ostream) is abstract, use derived class");
}
void Potentials::PotEAMAny::getE(Atoms& atom) {
    Neighborhoods nbhs(atom, cutoff);
    getE(atom, nbhs);
}
