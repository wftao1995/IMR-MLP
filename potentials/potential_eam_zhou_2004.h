// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_POTENTIAL_EAM_ZHOU2004_H
#define IMRMLP_POTENTIAL_EAM_ZHOU2004_H

// See https://journals.aps.org/prb/pdf/10.1103/PhysRevB.69.144113

#include "potential.h"
#include <unordered_map>
#include <set>

namespace Potentials
{
    struct EAMParametersZhouSingle
    {
        double re;
        double fe;
        double rhoe;
        double rhos;
        double alpha;
        double beta;
        double A;
        double B;
        double kppa;
        double lambda;
        double Fn0;
        double Fn1;
        double Fn2;
        double Fn3;
        double F0;
        double F1;
        double F2;
        double F3;
        double yta;
        double Fe;
        double ielement; // Nothing
        double amass;    // nothing
        double fm4;
        double beta1;
        double ramda1;
        double rhol;
        double rhoh;
        double rhoin;  // rhol * rhoe
        double rhoout; // rhoh * rhoe
        double rc;     // sqrt(5) * re
    };

    class EAMZhou2004 
    {
    private:
        std::unordered_map<int, EAMParametersZhouSingle> eleparas;
        std::set<int> elements;
        double rcut, rhoemax;
        bool inited = false;

    public:
        EAMZhou2004(){};
        ~EAMZhou2004(){};
        // density function
        double frho(int type, double r) const;
        void dfrhodr(int type, double r, double &val, double &dval) const;
        // pair potential single
        double phi_s(int type, double r) const;
        void dphi_sdr(int type, double r, double &val, double &dval) const;
        // pair potential
        double phi(int t1, int t2, double r) const;
        void dphidr(int t1, int t2, double r, double &val, double &dval) const;
        // embdding function F_i, type depent
        double femb(int type, double rho) const;
        void dfembdrho(int type, double rho, double &val, double &dval) const;
        // add element
        void add_element(std::string str);
        void add_element(int type) { elements.insert(type); }
        void init();

        void LoadParameterSet(const std::string &filename);
        void SaveParameterSet(const std::string &filename) const;

        void OutputLAMMPSset(std::string outfile, int nr, int nrho, double rmin = 0.5);

        void getEFS(Atoms &atom);
        void Load(const std::string &filename);
        void getE(Atoms &atom);
    };
};

#endif