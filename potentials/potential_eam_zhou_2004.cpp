// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_eam_zhou_2004.h"
#include "gpm_state.h"
#include <cmath>
#include "tools/chem.h"
#include <fstream>
#include "base/neighborhood.h"
#include <numeric>
#include <iomanip>
#include "tools/tools.h"
#include "utils/compressed_stream.h"

namespace Potentials
{
    double EAMZhou2004::frho(int type, double r) const
    {
        const auto &para = eleparas.at(type);
        double ratio = r / para.re;
        double exponential_term = std::exp(-para.beta1 * (ratio - 1.0));
        double polynomial_term = std::pow(ratio - para.ramda1, 20);
        return para.fe * exponential_term / (1.0 + polynomial_term);
        // return para.fe * std::exp(-para.beta1 * (r / para.re - 1.0)) / (1.0 + std::pow(r / para.re - para.ramda1, 20));
    }

    void EAMZhou2004::dfrhodr(int type, double r, double &val, double &dval) const
    {
        const EAMParametersZhouSingle &para = eleparas.at(type);
        double rr = r / para.re;
        double rl = rr - para.ramda1;
        // double upper = -expval * (para.beta1 * (1 + powval20) + 20.0 * std::pow(rl, 19));
        // double loweri = 1 + powval20;
        // dval = upper / loweri / loweri;
        double expval = para.fe * std::exp(-para.beta1 * (rr - 1.0));
        double powval19 = std::pow(rl, 19);
        double powval20 = powval19 * rl;
        double denominator = 1.0 + powval20;
        val = expval / denominator;
        dval = expval * (-para.beta1 * denominator - 20.0 * powval19) / (denominator * denominator);
    }

    double EAMZhou2004::phi_s(int type, double r) const
    {
        const EAMParametersZhouSingle &para = eleparas.at(type);
        double rr = r / para.re;
        double term1 = para.A * std::exp(-para.alpha * (rr - 1.0)) / (1.0 + std::pow(rr - para.kppa, 20));
        double term2 = para.B * std::exp(-para.beta * (rr - 1.0)) / (1.0 + std::pow(rr - para.lambda, 20));
        return term1 - term2;
    }

    void EAMZhou2004::dphi_sdr(int type, double r, double &val, double &dval) const
    {
        // double expleft = para.A * std::exp(-para.alpha * (rr - 1));
        // double expright = para.B * std::exp(-para.beta * (rr - 1));
        // double pow20left = std::pow(rr - para.kppa, 20);
        // double pow20right = std::pow(rr - para.lambda, 20);
        // val = expleft / (1.0 + pow20left) - expright / (1.0 + pow20right);
        // double dleft = -expleft * (para.alpha * (1 + pow20left) + 20.0 * pow(rr - para.kppa, 19)) / (1.0 + pow20left) / (1.0 + pow20left);
        // double dright = -expleft * (para.beta * (1 + pow20right) + 20.0 * pow(rr - para.lambda, 19)) / (1.0 + pow20right) / (1.0 + pow20right);
        // dval = dleft - dright;
        const EAMParametersZhouSingle &para = eleparas.at(type);
        double rr = r / para.re;
        double exponent_left = std::exp(-para.alpha * (rr - 1.0));
        double exponent_right = std::exp(-para.beta * (rr - 1.0));
        double pow20_left = std::pow(rr - para.kppa, 20);
        double pow20_right = std::pow(rr - para.lambda, 20);
        double denominator_left = 1.0 + pow20_left;
        double denominator_right = 1.0 + pow20_right;
        val = para.A * exponent_left / denominator_left - para.B * exponent_right / denominator_right;
        double dpow19_left = 20.0 * std::pow(rr - para.kppa, 19);
        double dpow19_right = 20.0 * std::pow(rr - para.lambda, 19);
        dval = -para.A * exponent_left * (para.alpha * denominator_left + dpow19_left) / (denominator_left * denominator_left) +
               para.B * exponent_right * (para.beta * denominator_right + dpow19_right) / (denominator_right * denominator_right);
    }

    double EAMZhou2004::phi(int t1, int t2, double r) const
    {
        if (t1 == t2)
        {
            return phi_s(t1, r);
        }

        double rat1 = frho(t1, r);
        double rat2 = frho(t2, r);
        double common_factor = 0.5 * rat2 / rat1;
        return common_factor * phi_s(t1, r) + phi_s(t2, r) / common_factor;

        // if (t1 != t2)
        // {
        //     double rat1 = frho(t1, r);
        //     double rat2 = frho(t2, r);
        //     return 0.5 * (rat2 / rat1 * phi_s(t1, r) + rat1 / rat2 * phi_s(t2, r));
        // }
        // else
        //     return phi_s(t1, r);
    }

    void EAMZhou2004::dphidr(int t1, int t2, double r, double &val, double &dval) const
    {
        if (t1 != t2)
            dphi_sdr(t1, r, val, dval);
        else
        {
            double fa, fb, dfa, dfb;
            double pha, phb, dpha, dphb;
            dfrhodr(t1, r, fa, dfa);
            dfrhodr(t2, r, fb, dfb);
            dphi_sdr(t1, r, pha, dpha);
            dphi_sdr(t2, r, phb, dphb);
            double rat1 = fb / fa;
            double rat2 = fa / fb;
            val = 0.5 * (rat1 * pha + rat2 * phb);
            double fa2 = fa * fa;
            double fb2 = fb * fb;
            double rtc1 = -fb / fa2 * dfa + 1 / fa * dfb;
            double rtc2 = -fa / fb2 * dfb + 1 / fb * dfa;
            dval = 0.5 * (rtc1 * pha + rat1 * dpha + rtc2 * phb + rat2 * dphb);
        }
    }

    double EAMZhou2004::femb(int type, double rho) const
    {
        const EAMParametersZhouSingle &para = eleparas.at(type);
        double fm33;
        if (rho < para.rhoe)
            fm33 = para.F3;
        else
            fm33 = para.fm4;

        if (rho < para.rhoin)
        {
            return para.Fn0 +
                   para.Fn1 * (rho / para.rhoin - 1.0) +
                   para.Fn2 * std::pow(rho / para.rhoin - 1.0, 2) +
                   para.Fn3 * std::pow(rho / para.rhoin - 1.0, 3);
        }
        else if (rho < para.rhoout)
        {
            return para.F0 +
                   para.F1 * (rho / para.rhoe - 1.0) +
                   para.F2 * std::pow(rho / para.rhoe - 1.0, 2) +
                   fm33 * std::pow(rho / para.rhoe - 1.0, 3);
        }
        else
        {
            return para.Fe *
                   (1 - para.yta * std::log(rho / para.rhos)) *
                   std::pow(rho / para.rhos, para.yta);
        }
    }

    void EAMZhou2004::dfembdrho(int type, double rho, double &val, double &dval) const
    {
        const EAMParametersZhouSingle &para = eleparas.at(type);
        double fm33;
        if (rho < para.rhoe)
            fm33 = para.F3;
        else
            fm33 = para.fm4;

        if (rho < para.rhoin)
        {
            val = para.Fn0 +
                  para.Fn1 * (rho / para.rhoin - 1.0) +
                  para.Fn2 * std::pow(rho / para.rhoin - 1.0, 2) +
                  para.Fn3 * std::pow(rho / para.rhoin - 1.0, 3);
            dval = para.Fn1 / para.rhoin +
                   2.0 * para.Fn2 * (rho / para.rhoin - 1.0) / para.rhoin +
                   3.0 * para.Fn3 * std::pow(rho / para.rhoin - 1.0, 2) / para.rhoin;
        }
        else if (rho < para.rhoout)
        {
            val = para.F0 +
                  para.F1 * (rho / para.rhoe - 1.0) +
                  para.F2 * std::pow(rho / para.rhoe - 1.0, 2) +
                  fm33 * std::pow(rho / para.rhoe - 1.0, 3);
            dval = para.F1 / para.rhoe +
                   2.0 * para.F2 * (rho / para.rhoe - 1) / para.rhoe +
                   3.0 * fm33 * std::pow(rho / para.rhoe - 1.0, 2) / para.rhoe;
        }
        else
        {
            double rr = rho / para.rhos;
            val = para.Fe *
                  (1 - para.yta * std::log(rr)) *
                  std::pow(rr, para.yta);
            dval = -para.Fe *
                   para.yta * para.yta *
                   std::pow(rr, para.yta) * std::log(rr) / rho;
        }
    }

    void EAMZhou2004::Load(const std::string &filename) {
        return;
    }

    namespace {
    void zhou_fields(GPM::State &a,EAMParametersZhouSingle &p) {
#define F(x) a.field(#x,p.x)
        F(re);F(fe);F(rhoe);F(rhos);F(alpha);F(beta);F(A);F(B);F(kppa);F(lambda);
        F(Fn0);F(Fn1);F(Fn2);F(Fn3);F(F0);F(F1);F(F2);F(F3);F(yta);F(Fe);F(ielement);F(amass);F(fm4);F(beta1);F(ramda1);F(rhol);F(rhoh);F(rhoin);F(rhoout);F(rc);
#undef F
        GPM::require(p.re>0&&p.rhoe>0&&p.rc>0,"invalid Zhou parameter domain");
    }
    }
    void EAMZhou2004::SaveParameterSet(const std::string &filename) const {
        GPM::require(!eleparas.empty(),"empty Zhou parameter set");GPM::Writer w;GPM::Object records;
        std::map<int,EAMParametersZhouSingle> ordered(eleparas.begin(),eleparas.end());
        for(auto &[z,p]:ordered){GPM::State a(w,"parameters/"+std::to_string(z));zhou_fields(a,p);records[std::to_string(z)]=GPM::Value(a.fields());}
        std::vector<int> active(elements.begin(),elements.end());GPM::State state(w,"parameters");state.field("active_elements",active);
        bool initialized=inited;state.field("initialized",initialized);
        if(inited){double cut=rcut,density=rhoemax;state.field("cutoff",cut);state.field("max_density",density);}
        w.manifest["models"]=GPM::Value(GPM::Object{{"root",GPM::Value(GPM::Object{{"type",GPM::Value("eam_zhou2004_parameters")},{"version",GPM::Value(1.)},{"parameters",GPM::Value(records)},{"state",GPM::Value(state.fields())}})}});
        w.save(filename);
    }
    void EAMZhou2004::LoadParameterSet(const std::string &filename)
    {
        if(GPM::is_file(filename)||GPM::extension(filename)) {
            GPM::Reader r(filename);r.validate();const auto &node=GPM::object(GPM::at(GPM::object(GPM::at(r.manifest,"models")),GPM::string(GPM::at(r.manifest,"root_model"))));
            GPM::require(GPM::string(GPM::at(node,"type"))=="eam_zhou2004_parameters"&&GPM::integer(GPM::at(node,"version"))==1,"unsupported Zhou parameter object");
            std::unordered_map<int,EAMParametersZhouSingle> parameters;
            for(const auto &[key,values]:GPM::object(GPM::at(node,"parameters"))){int z=std::stoi(key);GPM::require(z>0&&z<=118&&std::to_string(z)==key,"invalid atomic number");GPM::State a(r,GPM::object(values));zhou_fields(a,parameters[z]);}
            GPM::State state(r,GPM::object(GPM::at(node,"state")));std::vector<int> active;bool initialized;double cut=0,density=0;
            state.field("active_elements",active);state.field("initialized",initialized);for(int z:active)GPM::require(parameters.count(z),"missing active Zhou element");
            if(initialized){state.field("cutoff",cut);state.field("max_density",density);GPM::require(!active.empty()&&cut>0&&density>0,"invalid Zhou active state");}
            eleparas=std::move(parameters);elements=std::set<int>(active.begin(),active.end());inited=initialized;rcut=cut;rhoemax=density;return;
        }

        auto ifs = open_ifile(filename);
        if (!ifs) return;
        EAMParametersZhouSingle para;
        std::string buffer;
        std::cout << "Loading parameters form file " << filename << std::endl;
        while (!ifs->eof())
        {
            *ifs >> buffer;
            if (buffer == "")
                break;
            int type = Chemicals::atomic_numbers.at(buffer);
            *ifs >> para.re >> para.fe >> para.rhoe >>
                para.rhos >> para.alpha >> para.beta >>
                para.A >> para.B >> para.kppa >>
                para.lambda >> para.Fn0 >> para.Fn1 >>
                para.Fn2 >> para.Fn3 >> para.F0 >>
                para.F1 >> para.F2 >> para.F3 >>
                para.yta >> para.Fe >> para.ielement >>
                para.amass >> para.fm4 >> para.beta1 >>
                para.ramda1 >> para.rhol >> para.rhoh;
            para.rhoin = para.rhol * para.rhoe;
            para.rhoout = para.rhoh * para.rhoe;
            para.rc = std::sqrt(5.0) * para.re;
            eleparas[type] = para;
        }
    }

    void EAMZhou2004::add_element(std::string type)
    {
        int id = Chemicals::atomic_numbers.at(type);
        elements.insert(id);
        inited = false;
    }

    void EAMZhou2004::init()
    {
        rcut = eleparas.at(*elements.begin()).rc;
        rhoemax = eleparas.at(*elements.begin()).rhoe;
        for (auto it = elements.begin(); it != elements.end(); ++it)
        {
            if (rcut < eleparas.at(*it).rc)
                rcut = eleparas.at(*it).rc;
            if (rhoemax < eleparas.at(*it).rhoe)
                rhoemax = eleparas.at(*it).rhoe;
        }
        inited = true;
    }
    // will not check species!!
    void EAMZhou2004::getE(Atoms &atom)
    {
        if (!inited)
            init();
        Neighborhoods nbhs(atom, rcut);
        atom.energy_ = 0;
        atom.have_energy = true;
        atom.have_stress = false;
        atom.forces_.clear();
        atom.site_energies_.clear();
        atom.site_energies_.resize(atom.size());

        double total_eng = 0;
        for (int i = 0; i < nbhs.size(); ++i)
        {
            const Neighborhood &nbh = nbhs[i];
            int ctype = nbh.centertype;
            double density = 0;
            for (int j = 0; j < nbh.nbhcount_; ++j)
            {
                int jtype = nbh.chemmap_[j];
                double rij = nbh.distances_[j];
                // density
                density += frho(ctype, rij);
                if (i > nbh.indices_[j])
                    continue;
                // energy - pair
                double pair_eng = phi(ctype, jtype, rij);
                atom.site_energies_[i] += 0.5 * pair_eng;
                atom.site_energies_[j] += 0.5 * pair_eng;
                atom.energy_ += pair_eng;
            }
            // energy - density
            double emb_eng = femb(ctype, density);
            atom.site_energies_[i] += emb_eng;
            atom.energy_ += emb_eng;
        }
    }

    void EAMZhou2004::getEFS(Atoms &atom)
    {
        if (!inited)
            init();
        Neighborhoods nbhs(atom, rcut);
        atom.energy_ = 0;
        atom.have_energy = true;
        atom.have_stress = false;
        atom.forces_.clear();
        atom.site_energies_.clear();
        atom.forces_.resize(atom.size());
        atom.site_energies_.resize(atom.size());
        atom.stress_ = Eigen::Matrix3d::Zero();

        std::vector<double> rhoarr;
        rhoarr.resize(atom.size());
        double total_eng = 0;
        for (int i = 0; i < nbhs.size(); ++i)
        {
            const Neighborhood &nbh = nbhs[i];
            int ctype = nbh.centertype;
            double density = 0;
            double emb_eng = 0;
            double dembdrho = 0;
            std::vector<double> dparidr;
            std::vector<double> dfrdr;
            dparidr.resize(nbh.nbhcount_);
            dfrdr.resize(nbh.nbhcount_);
            for (int j = 0; j < nbh.nbhcount_; ++j)
            {
                int jtype = nbh.chemmap_[j];
                double rij = nbh.distances_[j];
                // energy - pair
                double pair_eng, dpairedr;
                dphidr(ctype, jtype, rij, pair_eng, dpairedr);
                atom.site_energies_[i] += 0.5 * pair_eng;
                atom.energy_ += 0.5 * pair_eng;
                // force - pair
                dparidr[j] = dpairedr;
                // density
                double tmpdensity;
                double dtmpdensity;
                dfrhodr(jtype, rij, tmpdensity, dtmpdensity);
                density += tmpdensity;
                dfrdr[j] = dtmpdensity;
            }
            dfembdrho(ctype, density, emb_eng, dembdrho);
            atom.site_energies_[i] += emb_eng;
            atom.energy_ += emb_eng;
            // forces and viral stress
            for (int j = 0; j < nbh.nbhcount_; ++j)
            {
                Eigen::Vector3d raddir = nbh.pos_[j].transpose() / nbh.pos_[j].norm();
                double force_emb = dembdrho * dfrdr[j];
                double force_me = dparidr[j] + force_emb;
                // viral stress
                for (int a = 0; a < 3; a++)
                {
                    double forcescalar = force_me * nbh.pos_[j](a) / nbh.distances_[j];
                    atom.forces_[i](a) += forcescalar;
                    for (int b = 0; b < 3; b++)
                        atom.stress_(a, b) -= forcescalar * nbh.pos_[j](b);
                }
            }
        }
    }

    void EAMZhou2004::OutputLAMMPSset(std::string outfile, int nr, int nrho, double rmin)
    {
        if (!inited)
            init();
        auto ofs = open_ofile(outfile);
        if (!ofs) return;
        std::string lat = "  fcc  ";
        *ofs << "Created by GPIO potential module\n";
        *ofs << "See zhou 2004\n";
        *ofs << "setfl format\n";
        *ofs << std::setprecision(18);
        *ofs << std::fixed;
        *ofs << std::scientific;
        int ntypes = elements.size();
        std::vector<int> velements;
        for (auto it = elements.begin(); it != elements.end(); ++it)
        {
            velements.push_back(*it);
        }
        *ofs << "   " << ntypes;
        for (auto it = elements.begin(); it != elements.end(); ++it)
        {
            *ofs << "  " << Chemicals::chemical_symbols[*it];
        }
        *ofs << "\n";
        double delta_r = rcut / static_cast<double>(nr - 1);
        double rhomax = 2.0 * rhoemax;
        if (rhomax < 100)
            rhomax = 100;
        // tabulate density function and pair interactions
        std::vector<std::vector<double>> rho_table(ntypes);
        std::for_each(rho_table.begin(), rho_table.end(), [&nr](std::vector<double> &a)
                      { a.resize(nr); });
        std::vector<std::vector<double>> pair_table(ntypes * (ntypes + 1) / 2);
        std::for_each(pair_table.begin(), pair_table.end(), [&nr](std::vector<double> &a)
                      { a.resize(nr); });
        int ct = 0;
        int ct2 = 0;
        std::vector<double> rvalarr(nr, 0.0);
        for (int i = 0; i < nr; ++i)
        {
            rvalarr[i] = delta_r * static_cast<double>(i);
            if (rvalarr[i] < rmin)
                rvalarr[i] = rmin;
        }
        for (auto it = velements.begin(); it != velements.end(); ++it)
        {
            std::vector<double> &rtable = rho_table[ct];
            ++ct;
            for (int i = 0; i < nr; ++i)
            {
                rtable[i] = frho(*it, rvalarr[i]);
                if (rhomax < rtable[i])
                {
                    rhomax = rtable[i];
                }
            }
            for (auto it2 = velements.begin(); it2 <= it; ++it2)
            {
                std::vector<double> &ptable = pair_table[ct2];
                ct2++;
                for (int i = 0; i < nr; ++i)
                {
                    ptable[i] = rvalarr[i] * phi(*it, *it2, rvalarr[i]);
                }
            }
        }
        double drho = rhomax / static_cast<double>(nrho - 1);
        std::vector<std::vector<double>> emb_table(ntypes);
        std::for_each(emb_table.begin(), emb_table.end(), [&nrho](std::vector<double> &a)
                      { a.resize(nrho); });
        for (int ct = 0; ct < velements.size(); ++ct)
        {
            std::vector<double> &rtable = emb_table[ct];
            for (int i = 0; i < nrho; ++i)
            {
                rtable[i] = femb(velements[ct], static_cast<double>(i) * drho);
            }
        }
        *ofs << "  " << nrho << "  " << drho << "  " << nr << "  "
            << delta_r << "  " << rcut << '\n';
        int mlrs = 5;
        for (int i = 0; i < ntypes; ++i)
        {
            *ofs << "  " << velements[i] << "  " << eleparas[velements[i]].amass << "  "
                << sqrt(2.0) * eleparas[velements[i]].re << "  "
                << lat;
            // embedding function
            for (int j = 0; j < nrho; ++j)
            {
                if (j % mlrs == 0)
                    *ofs <<'\n';
                *ofs <<std::setw(27) << emb_table[i][j];
            }
            *ofs <<'\n';
            // density function
            for (int j = 0; j < nr; ++j)
            {
                if (j % mlrs == 0)
                    *ofs <<'\n';
                *ofs <<std::setw(27) << rho_table[i][j];
            }
            *ofs <<'\n';
        }
        // pair interactions * r
        for (int i = 0; i < (ntypes + 1) * ntypes / 2; ++i)
        {
            for (int j = 0; j < nr; ++j)
            {
                if (j % mlrs == 0)
                    *ofs <<'\n';
                *ofs <<std::setw(27) << pair_table[i][j];
            }
        }
    }
};

