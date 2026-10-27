// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_mtp_m.h"
#include "io/gpio.h"
#include <cmath>
#include <memory>

int initmtpm(int argc, char** argv)
{
    using namespace Potentials;
    using namespace std;
    GPSettings settings;
    auto &cmdopts = CMDOption::global_cmdopts;
    string potfile = "pot.mtp";
    string settingfile = "settings.ini";
    string dataset = "train.cfg";
    string format = "";
    string index = "index.map";
    bool randomize = true;
    double rand_scale = 1.0;
    double lb = -0.1, ub = 0.1;
    GPIO iobj;
    cmdopts.addop(potfile, "potfile", "pot", "Potential file", false, 1, 0)
        .addop(settingfile, "settings", "", "settings file name", true, 1, 0)
        .addop(dataset, "dataset", "d", "dataset file name", false, 1, 0)
        .addop(format, "format", "f", "dataset file format, mlpcfg, extxyz, ...", false, 1, 0)
        .addop(index, "index", "id", "index file name, for selecting configurations", false, 1, 0)
        .addop(rand_scale, "rand-scale", "", "scale for random coeffs, default 1.0", false, 1, 0)
        .addop(lb, "rand-lb", "", "lower bound for random coeffs, default -0.1", false, 1, 0)
        .addop(ub, "rand-ub", "", "upper bound for random coeffs, default 0.1", false, 1, 0)
        .addop(randomize, "randomize", "r", "randomize coeffs or not, default true", false, 1, 0);
    cmdopts.phrase_options(argc, argv);
    if (!settingfile.empty()) {
        try {
            settings.Load(settingfile);
        } catch (const GPException& e) {
            std::cerr << e.What() << '\n';
            return 1;
        }
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    vector<Atoms> atoms;
    iobj.readmap(index);
    if (!iobj.read(atoms, dataset, format))
        GPERROR("Failed to read MTP-M training data: " + dataset);
    if (atoms.empty())
        GPERROR("MTP-M initialization requires nonempty training data");
    int n_labeled = 0;
    for (const auto &at : atoms) {
        if (at.size() == 0)
            GPERROR("MTP-M initialization requires nonempty training structures");
        if (at.have_energy) {
            if (!std::isfinite(at.energy_))
                GPERROR("MTP-M reference energies must be finite");
            ++n_labeled;
        }
    }
    if (mpi.rank == 0 && n_labeled < atoms.size())
        Warning("Warning: MTP-M skipped " + std::to_string(atoms.size() - n_labeled) +
                " structures without energy labels when fitting species offsets");
    if (mpi.rank == 0 && n_labeled == 0)
        Warning("Warning: MTP-M has no energy labels; species offsets are initialized to zero");

    int n_cfgs_total = atoms.size();
    int n_loc_total = 0;
    int n_loc_local = 0;

    #ifdef IMRMLP_MPI
    Message("Balancing");
    {
        // first sort by size to balance total number of atoms
        sort(atoms.begin(), atoms.end(), [](const Atoms &a, const Atoms &b) {return a.size() < b.size();});
        auto itrm = remove_if(atoms.begin(), atoms.end(), [](Atoms &at) {
            return at.id % mpi.size == mpi.rank ? false : true;});
        atoms.erase(itrm, atoms.end());
    }
    #endif 
    for (int i = 0; i < atoms.size(); ++i) {
        n_loc_local+= atoms[i].size();
    }
    #ifdef IMRMLP_MPI
    MPI_Allreduce(&n_loc_local, &n_loc_total, 1, MPI_INT, MPI_SUM, mpi.comm);
    #else
    n_loc_total = n_loc_local;
    #endif

    int n_cfgs_this = atoms.size();
    auto potsettings = settings.ExtractSubSettings("pot");
    std::unique_ptr<AnyPotential> pot;
    // Check if LES type is requested
    string ptype = "";
    if (potsettings.count("pottype") > 0)
        ptype = potsettings["pottype"];
    if (ptype == "mtp_les") {
        std::cerr << "Error: initmtpm does not support mtp_les directly.\n"
                  << "Use 'imrmlp trainnlpot --settings fit.ini' with fit:random_init true\n"
                  << "to initialize LES potentials. Add les_init_range=0 for charge balancing.\n";
        return 1;
    } else {
        pot.reset(create_mtp_m_potential(potsettings));
    }
    if (pot == nullptr) {
        std::cerr << "Error creating potential.\n";
        return 1;
    }
    PotMTP *pmtp = dynamic_cast<PotMTP*>(pot.get());
    int nprops;
    pmtp->get_n_props(nprops);
    int n_radial, n_species_coeffs, n_moment, n_model, n_species;
    pmtp->get_n_coeffs(n_radial, n_species_coeffs, n_moment, n_model);
    n_species = n_species_coeffs / (nprops + 1);
    double *radial_ptr, *species_ptr, *moment_ptr, *model_ptr;
    pmtp->get_pointers(radial_ptr, species_ptr, moment_ptr, model_ptr);

    pmtp->random_lower = lb;
    pmtp->random_upper = ub;

    if (randomize) {
        pot->random_init();
        // random init radial coeffs
        uniform_real_distribution<double> uniform(-1, 1);
        for (int i = 0; i < n_radial; ++i) {
            radial_ptr[i] = uniform(RNG.mtd);
        }
        pmtp->Orthogonalize();
    }
    #ifdef IMRMLP_MPI
    MPI_Barrier(mpi.comm);
    MPI_Bcast(radial_ptr, n_radial, MPI_DOUBLE, 0, mpi.comm);
    #endif 
    // init species coeffs
    Eigen::MatrixXd linmat;
    linmat.resize(n_species, n_species);
    linmat.setZero();
    Eigen::VectorXd sps, linvec;
    sps.resize(n_species);
    linvec.resize(n_species);
    linvec.setZero();
    for (int i = 0; i < n_cfgs_this; ++i) {
        if (!atoms[i].have_energy) continue;
        int n = atoms[i].size();
        double wgt = 1.0 / n / n_labeled;
        sps.setZero();
        for (int j = 0; j < atoms[i].size(); ++j) {
            sps(atoms[i].chemmap_[j]) += 1.;
        }
        for (int m = 0; m < n_species; ++m) {
            for (int n = m; n < n_species; ++n) {
                linmat(m,n) += wgt * sps(m) * sps(n);
            }
            linvec(m) += wgt * sps(m) * atoms[i].energy_;
        }
    }
    // sum over MPI, linvec
    #ifdef IMRMLP_MPI
    Eigen::VectorXd globvec;
    globvec.resize(n_species);
    MPI_Allreduce(linvec.data(), globvec.data(), n_species, MPI_DOUBLE, MPI_SUM, mpi.comm);
    linvec = globvec;
    // linmat
    Eigen::MatrixXd globmat;
    globmat.resize(n_species, n_species);
    MPI_Allreduce(linmat.data(), globmat.data(), n_species * n_species, MPI_DOUBLE, MPI_SUM, mpi.comm);
    linmat = globmat;
    #endif
    if (mpi.rank == 0) {
        // symmetric
        for (int i = 0; i < n_species; ++i) {
            for (int j = 0; j < i; ++j) {
                linmat(i,j) = linmat(j,i);
            }
        }
        // solve
        Eigen::VectorXd sol = Eigen::VectorXd::Zero(n_species);
        if (n_labeled > 0) {
            Eigen::FullPivLU<Eigen::MatrixXd> lu(linmat);
            sol = lu.solve(linvec);
            if (!sol.allFinite())
                GPERROR("MTP-M species offset fitting produced non-finite coefficients");
        }
        // assign
        for (int i = 0; i < n_species; ++i) {
            species_ptr[i] = sol(i);
        }
    }
    // bcast 
    #ifdef IMRMLP_MPI
    MPI_Barrier(mpi.comm);
    MPI_Bcast(species_ptr, n_species, MPI_DOUBLE, 0, mpi.comm);
    #endif
    // init bias and moment coeffs
    // compute mean of each basis
    int n_scalars = pmtp->get_alpha_scalar_moments();
    Eigen::VectorXd basis_mean, basis_var;
    basis_mean.resize(n_scalars);
    basis_var.resize(n_scalars);
    basis_mean.setZero();
    basis_var.setZero();
    std::vector<double> local_disp(n_scalars + 1, 0);
    for (int i = 0; i < n_cfgs_this; ++i) {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j) {
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int k = 0; k < n_scalars; ++k) {
                basis_mean(k) += local_disp[k + 1] / n_loc_total;
            }
        }
    }
    // sum over MPI
    #ifdef IMRMLP_MPI
    Eigen::VectorXd globvec2;
    globvec2.resize(n_scalars);
    globvec2.setZero();
    MPI_Allreduce(basis_mean.data(), globvec2.data(), n_scalars, MPI_DOUBLE, MPI_SUM, mpi.comm);
    basis_mean = globvec2;
    #endif
    // compute variance
    for (int i = 0; i < n_cfgs_this; ++i) {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j) {
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int k = 0; k < n_scalars; ++k) {
                double diff = local_disp[k + 1] - basis_mean(k);
                basis_var(k) += diff * diff / n_loc_total;
            }
        }
    }
    // sum over MPI
    #ifdef IMRMLP_MPI
    globvec2.setZero();
    MPI_Allreduce(basis_var.data(), globvec2.data(), n_scalars, MPI_DOUBLE, MPI_SUM, mpi.comm);
    basis_var = globvec2;
    #endif
    if (mpi.rank == 0) {
        cout << "Basis means:\n" << basis_mean.transpose() << "\n";
        cout << "Basis variances:\n" << basis_var.transpose() << "\n";
    }
    // assign moment coeffs
    normal_distribution<double> normal(0, 1);
    if (randomize && mpi.rank == 0) {
        for (int n = 0; n < nprops; ++n) {
            for (int i = 0; i < n_scalars; ++i) {
                if (basis_var(i) > 1e-12) {
                    moment_ptr[n_scalars * n + i] = normal(RNG.mtd) * rand_scale / sqrt(basis_var(i));
                } else {
                    moment_ptr[n_scalars * n + i] = 0;
                }
            }
        }
    }
    // bcast
    #ifdef IMRMLP_MPI
    MPI_Barrier(mpi.comm);
    MPI_Bcast(moment_ptr, n_moment, MPI_DOUBLE, 0, mpi.comm);
    #endif
    // adjust species_coeffs to make sure mean of qi = 0
    Eigen::VectorXd qs;
    Eigen::VectorXi tpc;
    tpc.resize(n_species);
    tpc.setZero();
    qs.resize(nprops);
    qs.setZero();
    for (int i = 0; i < atoms.size(); ++i) {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j) {
            tpc(atoms[i].chemmap_[j]) += 1;
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int n = 0; n < nprops; ++n) {
                for (int k = 0; k < n_scalars; ++k) {
                    qs(n) += local_disp[k + 1] * moment_ptr[n_scalars * n + k] / n_loc_total;
                }
            }
        }
    }
    // sum over MPI
    #ifdef IMRMLP_MPI
    Eigen::VectorXd globvec3;
    Eigen::VectorXi globveci;
    globveci.resize(n_species);
    globveci.setZero();
    MPI_Allreduce(tpc.data(), globveci.data(), n_species, MPI_INT, MPI_SUM, mpi.comm);
    tpc = globveci;
    globvec3.resize(nprops);
    globvec3.setZero();
    MPI_Allreduce(qs.data(), globvec3.data(), nprops, MPI_DOUBLE, MPI_SUM, mpi.comm);
    qs = globvec3;
    #endif
    if (mpi.rank == 0) {
        for (int n = 0; n < nprops; ++n) {
            for (int i = 0; i < n_species; ++i) {
                species_ptr[n_species + n_species * n + i] = -qs(n) ;//* (double)tpc(i) / (double)tpc.sum();
            }
        }
    }
    // bcast
    #ifdef IMRMLP_MPI
    MPI_Barrier(mpi.comm);
    MPI_Bcast(species_ptr, n_species * (nprops + 1), MPI_DOUBLE, 0, mpi.comm);
    #endif
    
    // calculate var of Qs
    Eigen::VectorXd qs2;
    qs2.resize(nprops);
    qs.setZero();
    qs2.setZero();
    for (int i = 0; i < atoms.size(); ++i) {
        Neighborhoods nbhs(atoms[i], pot->cutoff);
        for (int j = 0; j < atoms[i].size(); ++j) {
            pmtp->compute_descriptors(nbhs[j], local_disp);
            for (int n = 0; n < nprops; ++n) {
                double qi = species_ptr[atoms[i].chemmap_[j] + n_species * (n + 1)];
                for (int k = 0; k < n_scalars; ++k) {
                    qi += local_disp[k + 1] * moment_ptr[n_scalars * n + k];
                }
                qs[n] += qi / n_loc_total;
                qs2(n) += qi * qi / n_loc_total;
            }
        }
    }
    // sum over MPI
    #ifdef IMRMLP_MPI
    globvec3.resize(nprops);
    globvec3.setZero();
    MPI_Allreduce(qs.data(), globvec3.data(), nprops, MPI_DOUBLE, MPI_SUM, mpi.comm);
    qs = globvec3;
    globvec3.setZero();
    MPI_Allreduce(qs2.data(), globvec3.data(), nprops, MPI_DOUBLE, MPI_SUM, mpi.comm);
    qs2 = globvec3;
    #endif
    if (mpi.rank == 0) {
        for (int n = 0; n < nprops; ++n) {
            double var = qs2(n) - qs(n) * qs(n);
            cout << "Mean of Q" << n << " after initialization: " << qs(n) <<  "\n";
            cout << "Var of Q" << n << " after initialization: " << var <<  "\n";
        }
    }

    if (mpi.rank == 0) {
        pot->Save(potfile);
    }

    return 0;
}
