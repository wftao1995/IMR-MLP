// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 
#include "potential.h"
#include <unordered_map>
#include <set>

namespace Potentials
{
    class PotEAMAny : public AnyPotential
    {
        friend struct GPMCodecAccess;
    private:
        double *regression_coeffs;
        double *species_coeffs;

        // buffer
        std::vector<double> buff_drho_dr;
        Eigen::Matrix<double, -1, 3, Eigen::RowMajor> buff_site_energy_ders;
    protected:
        bool eam_fs = true; // if true, rho is i-j type depentent. Only affect output
        std::vector<std::string> type_name;
    public:
        double rho_min = 0.0;
        double rho_max = 0.0;
        PotEAMAny(){};
        ~PotEAMAny(){};
        // pair function
        virtual double func_pair(int typei, int typej, double rij);
        virtual void func_pair_d(int typei, int typej, double rij, double &f, double &dfdr);
        // embeeding function
        virtual double func_emb(int typei, double rho);
        virtual void func_emb_d(int typei, double rho, double &f, double &dfdrho);
        // density function
        virtual double func_density(int typei, int typej, double rij);
        virtual void func_density_d(int typei, int typej, double rij, double &f, double &dfdr);
        //
        bool supports_site_efs() const override { return true; }
        virtual void CalcSiteEFS(Neighborhood &nbh, double &site_en, Eigen::Matrix<double, -1, 3, Eigen::RowMajor>& site_energy_ders) override;
        virtual void CalcSiteE(Neighborhood &nbh, double &site_en) override;
        // Save LAMMPS readable file
        virtual void OutputLAMMPSset(std::string outfile, int nr, int nrho, double rhomin = 0.0, double rhomax = 100);

        virtual void getEFS(Atoms &atom, AnyNeighborhoods &nbhs);
        virtual void getE(Atoms &atom, AnyNeighborhoods &nbhs);
        virtual void Load(std::istream &is);
        virtual void Save(std::ostream &os);

        void getE(Atoms &atom);
    };
};
