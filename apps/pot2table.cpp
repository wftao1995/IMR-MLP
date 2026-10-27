// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/// \file pot2table.cpp
/// \brief Export pair/three-body potentials as tabulated PairTable / ThreebodyTable
///        and LAMMPS-compatible tabulated potential files.
///
/// Usage:
///   imrmlp pot2table --potfile potential.sw --pottype sw --typemap index.map \
///                    --pair_output pair.table --threebody_output threebody.table \
///                    --nr 1000 --ntheta 181 --rmin 0.01 --lammps

#include "potentials/potential_wrapper.h"
#include "base/neighborhood.h"
#include "io/gpio.h"
#include <fstream>
#include <cmath>
#include <iomanip>

using namespace std;

/// Create a 2-atom system for pair energy evaluation.
static Atoms make_pair_system(int type_i, int type_j, double r) {
    Atoms atom;
    atom.pbc_ = {false, false, false};
    // Large box to avoid periodic images
    double L = r + 20.0;
    atom.cell_ = Eigen::Matrix3d::Identity() * L;
    atom.chemmap_ = {type_i, type_j};
    atom.pos_.resize(2);
    atom.pos_[0] = Eigen::RowVector3d(0, 0, 0);
    atom.pos_[1] = Eigen::RowVector3d(r, 0, 0);
    return atom;
}

/// Create a 3-atom system for three-body energy evaluation.
/// Atom i at origin, j at (rij, 0, 0), k at (rik*cos(theta), rik*sin(theta), 0).
static Atoms make_triplet_system(int type_i, int type_j, int type_k,
                                  double rij, double rik, double theta) {
    Atoms atom;
    atom.pbc_ = {false, false, false};
    double L = max(rij, rik) + 20.0;
    atom.cell_ = Eigen::Matrix3d::Identity() * L;
    atom.chemmap_ = {type_i, type_j, type_k};
    atom.pos_.resize(3);
    atom.pos_[0] = Eigen::RowVector3d(0, 0, 0);
    atom.pos_[1] = Eigen::RowVector3d(rij, 0, 0);
    atom.pos_[2] = Eigen::RowVector3d(rik * cos(theta), rik * sin(theta), 0);
    return atom;
}

/// Evaluate pair energy V(r) for types (i,j) at distance r.
static double eval_pair(Potentials::AnyPotential *pot, int type_i, int type_j, double r,
                        double *force = nullptr) {
    Atoms atom = make_pair_system(type_i, type_j, r);
    Neighborhoods nbh(atom, pot->cutoff);
    pot->getEFS(atom, nbh);
    if (!std::isfinite(atom.energy_) || !atom.forces_[1].allFinite())
        throw std::runtime_error("pot2table: nonfinite pair energy or force");
    if (force) *force = atom.forces_[1](0);
    return atom.energy_;
}

/// Evaluate total energy of a 3-atom triplet (i,j,k) with distances rij, rik and angle theta.
static double eval_triplet(Potentials::AnyPotential *pot,
                            int type_i, int type_j, int type_k,
                            double rij, double rik, double theta) {
    Atoms atom = make_triplet_system(type_i, type_j, type_k, rij, rik, theta);
    Neighborhoods nbh(atom, pot->cutoff);
    pot->getEFS(atom, nbh);
    return atom.energy_;
}

static void finish_output(ofstream &stream, const string &filename) {
    stream.close();
    if (!stream) throw std::runtime_error("pot2table: failed to write " + filename);
}

/// Get unique pair count for the upper triangle.
static int npairs(int nelements) {
    return nelements * (nelements + 1) / 2;
}

/// Get unique triplet count.
static int ntriplets(int nelements) {
    return nelements * (nelements + 1) * (nelements + 2) / 6;
}

int pot2table(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;

    string potfile, pottype, typemap_file = "index.map";
    string pair_output, threebody_output;
    string lammps_prefix;
    int nr = 1000;
    int ntheta = 181;
    double rmin = 0.01;
    double rmax_override = -1;
    bool do_pair = false;
    bool do_threebody = false;
    bool do_lammps = false;
    string settingfile;

    cmdopts.addop(potfile, "potfile", "pot", "Potential file path", true)
        .addop(pottype, "pottype", "t", "Potential type (lj612, sw, tersoff, eampoly, ...)", true)
        .addop(typemap_file, "index", "id", "Index map file")
        .addop(pair_output, "pair-output", "po", "Output PairTable format file for pair potential")
        .addop(threebody_output, "threebody-output", "to", "Output ThreebodyTable format file for 3-body potential")
        .addop(lammps_prefix, "lammps-prefix", "lp", "Prefix for LAMMPS tabulated potential output files")
        .addop(nr, "nr", "nr", "Number of radial grid points (default 1000)")
        .addop(ntheta, "ntheta", "nt", "Number of angular grid points (default 181)")
        .addop(rmin, "rmin", "rmin", "Minimum radius (default 0.01)")
        .addop(rmax_override, "rmax", "rmax", "Maximum radius override (default: from potential cutoff)")
        .addop(do_pair, "pair", "", "Export pair table", false, 0)
        .addop(do_threebody, "threebody", "", "Export three-body table", false, 0)
        .addop(do_lammps, "lammps", "lm", "Also write LAMMPS tabulated format files", false, 0)
        .addop(settingfile, "settings", "", "Settings file for potential", false)
        .phrase_options(argc, argv);

    if (cmdopts.displayhelp) {
        cmdopts.showhelp(cout);
        cout << "\nExamples:\n"
             << "  imrmlp pot2table --potfile my_pot.sw --pottype sw --pair --pair_output pair.table\n"
             << "  imrmlp pot2table --potfile my_pot.sw --pottype sw --threebody --threebody_output tb.table\n"
             << "  imrmlp pot2table --potfile my_pot.tersoff --pottype tersoff --pair --lammps\n";
        return 0;
    }

    if (!do_pair && !do_threebody) {
        cerr << "ERROR: specify at least one of --pair or --threebody\n";
        return 1;
    }

    if (nr < (do_pair ? 3 : 2) || (do_threebody && ntheta < 2) ||
        !std::isfinite(rmin) || rmin <= 0 ||
        !std::isfinite(rmax_override) || (rmax_override != -1 && rmax_override <= 0)) {
        cerr << "ERROR: invalid grid; nr must be >= 3 for pair tables (>= 2 otherwise), "
             << "ntheta >= 2 for three-body tables, and radii must be finite and positive\n";
        return 1;
    }

    // Load settings if provided
    GPSettings settings;
    if (!settingfile.empty() && GPIO::file_exist(settingfile))
        settings.Load(settingfile);

    // Load type map
    GPIO iobj;
    map<string, int> typemap;
    if (GPIO::file_exist(typemap_file)) {
        iobj.readmap(typemap_file);
        typemap = iobj.getrevmap();
    }

    // Load potential
    GPSettings potsettings = settings.ExtractSubSettings("pot");
    if (!potfile.empty()) potsettings.Modify("potfile", potfile);
    if (!pottype.empty()) potsettings.Modify("pottype", pottype);

    Potentials::potential_wrapper wrapper(potsettings);
    Potentials::AnyPotential *pot = wrapper.pot;
    if (!pot) {
        cerr << "ERROR: failed to load potential\n";
        return 1;
    }

    int nelements = pot->species_count;
    double rmax = (rmax_override > 0) ? rmax_override : pot->cutoff;
    if (nelements <= 0 || !std::isfinite(rmax) || rmax <= rmin ||
        !std::isfinite(pot->cutoff) || pot->cutoff <= 0) {
        cerr << "ERROR: invalid species count or grid; require finite 0 < rmin < rmax and cutoff > 0\n";
        return 1;
    }

    cout << "Potential loaded: species=" << nelements
         << ", cutoff=" << pot->cutoff
         << ", rmax=" << rmax << "\n";

    // ═══════════════════════════════════════════════════════════════════
    // Pair table export
    // ═══════════════════════════════════════════════════════════════════
    if (do_pair) {
        int np = npairs(nelements);
        double dr = (rmax - rmin) / (nr - 1);
        cout << "Building pair table: " << nelements << " elements, "
             << np << " pairs, " << nr << " radial points, "
             << "r=[" << rmin << ", " << rmax << "], dr=" << dr << "\n";

        // Evaluate V(r) for each pair type
        vector<vector<double>> pair_vals(np, vector<double>(nr));

        for (int i = 0; i < nelements; ++i) {
            for (int j = i; j < nelements; ++j) {
                int pid = j * (j + 1) / 2 + i;
                cout << "  Pair (" << i << "," << j << "): sampling...\n";
                for (int ir = 0; ir < nr; ++ir) {
                    double r = rmin + ir * dr;
                    pair_vals[pid][ir] = eval_pair(pot, i, j, r);
                }
            }
        }

        // Build splines and write PairTable format
        string outfile = pair_output.empty() ? "pair_table.out" : pair_output;
        ofstream ofs(outfile);
        if (!ofs.is_open()) {
            cerr << "ERROR: cannot open " << outfile << " for writing\n";
            return 1;
        }

        ofs << std::setprecision(17);
        // Header: element names
        if (!typemap.empty()) {
            vector<string> elenames(nelements);
            for (const auto &[name, idx] : typemap)
                if (idx >= 0 && idx < nelements) elenames[idx] = name;
            for (int i = 0; i < nelements; ++i)
                ofs << (elenames[i].empty() ? "X" + to_string(i) : elenames[i]) << " ";
            ofs << "\n";
        } else {
            for (int i = 0; i < nelements; ++i)
                ofs << "X" << i << " ";
            ofs << "\n";
        }

        // Grid line
        ofs << "N " << nr << "\n";

        // Data lines
        for (int ir = 0; ir < nr; ++ir) {
            double r = rmin + ir * dr;
            ofs << ir << " " << r;
            for (int p = 0; p < np; ++p)
                ofs << " " << pair_vals[p][ir];
            ofs << "\n";
        }
        finish_output(ofs, outfile);
        cout << "Pair table written to: " << outfile << "\n";

        // LAMMPS format
        if (do_lammps || !lammps_prefix.empty()) {
            string prefix = lammps_prefix.empty() ? "lammps_pair" : lammps_prefix;
            // LAMMPS expects N equally-spaced points from rmin to rmax
            int lmp_n = nr;
            double lmp_dr = (rmax - rmin) / (lmp_n - 1);
            for (int i = 0; i < nelements; ++i) {
                for (int j = i; j < nelements; ++j) {
                    int pid = j * (j + 1) / 2 + i;
                    string lmpfile = prefix + "_" + to_string(i) + "_" + to_string(j);
                    ofstream lmfs(lmpfile);
                    if (!lmfs.is_open()) {
                        cerr << "ERROR: cannot open " << lmpfile << "\n";
                        return 1;
                    }
                    lmfs << std::setprecision(17);
                    lmfs << "# LAMMPS pair_style table format\n";
                    lmfs << "# Generated by pot2table\n";
                    lmfs << "PAIR_" << i << "_" << j << "\n";
                    lmfs << "N " << lmp_n << " R " << rmin << " " << rmax << "\n\n";
                    for (int ir = 0; ir < lmp_n; ++ir) {
                        double r = rmin + ir * lmp_dr;
                        double force;
                        double val = eval_pair(pot, i, j, r, &force);
                        lmfs << (ir + 1) << " " << r << " " << val << " " << force << "\n";
                    }
                    finish_output(lmfs, lmpfile);
                }
            }
            cout << "LAMMPS pair tables written with prefix: " << prefix << "\n";
        }
    }

    // ═══════════════════════════════════════════════════════════════════
    // Three-body table export
    // ═══════════════════════════════════════════════════════════════════
    if (do_threebody) {
        int nt = ntriplets(nelements);
        double dr = (rmax - rmin) / (nr - 1);
        double dtheta = M_PI / (ntheta - 1);
        cout << "Building three-body table: " << nelements << " elements, "
             << nt << " triplets, " << nr << " x " << ntheta << " grid, "
             << "r=[" << rmin << ", " << rmax << "], "
             << "theta=[0, pi]\n";

        // Tables per triplet: nr x ntheta (row-major: r slow, theta fast)
        vector<Eigen::MatrixXd> tables(nt,
            Eigen::MatrixXd::Zero(nr, ntheta));

        // Evaluate for each triplet: table V(r, theta) at symmetric rij=rik=r
        for (int i = 0; i < nelements; ++i) {
            for (int j = i; j < nelements; ++j) {
                for (int k = j; k < nelements; ++k) {
                    // triplet id matching ThreebodyTable::get_triplet_id
                    // (sorts to descending: ii >= jj >= kk)
                    int ii = i, jj = j, kk = k;
                    if (ii < jj) std::swap(ii, jj);
                    if (jj < kk) std::swap(jj, kk);
                    if (ii < jj) std::swap(ii, jj);
                    int tid = ii * (ii + 1) * (ii + 2) / 6 + jj * (jj + 1) / 2 + kk;
                    cout << "  Triplet (" << i << "," << j << "," << k
                         << "): sampling " << nr << "x" << ntheta << " grid...\n";

                    for (int ir = 0; ir < nr; ++ir) {
                        double r = rmin + ir * dr;
                        for (int it = 0; it < ntheta; ++it) {
                            double theta = it * dtheta;
                            if (theta < 1e-12) theta = 1e-12;
                            if (theta > M_PI - 1e-12) theta = M_PI - 1e-12;
                            // For symmetric case rij=rik=r, evaluate total energy
                            // of 3-atom system and store
                            double e_total = eval_triplet(pot, i, j, k, r, r, theta);
                            tables[tid](ir, it) = e_total;
                        }
                    }
                }
            }
        }

        // Write ThreebodyTable format
        string outfile = threebody_output.empty() ? "threebody_table.out" : threebody_output;
        ofstream ofs(outfile);
        if (!ofs.is_open()) {
            cerr << "ERROR: cannot open " << outfile << " for writing\n";
            return 1;
        }

        ofs << std::setprecision(17);
        // Header: element names
        if (!typemap.empty()) {
            vector<string> elenames(nelements);
            for (const auto &[name, idx] : typemap)
                if (idx >= 0 && idx < nelements) elenames[idx] = name;
            for (int i = 0; i < nelements; ++i)
                ofs << (elenames[i].empty() ? "X" + to_string(i) : elenames[i]) << " ";
            ofs << "\n";
        } else {
            for (int i = 0; i < nelements; ++i)
                ofs << "X" << i << " ";
            ofs << "\n";
        }

        // Grid parameters
        ofs << "nr " << nr << " ntheta " << ntheta << "\n";
        ofs << "rmin " << rmin << " rmax " << rmax << "\n";
        ofs << "theta_min 0.0 theta_max " << M_PI << "\n";

        // Data: row-major per triplet
        for (int t = 0; t < nt; ++t) {
            for (int ir = 0; ir < nr; ++ir) {
                for (int it = 0; it < ntheta; ++it) {
                    ofs << tables[t](ir, it);
                    if (it < ntheta - 1) ofs << " ";
                }
                ofs << "\n";
            }
        }
        finish_output(ofs, outfile);
        cout << "Three-body table written to: " << outfile << "\n";
    }

    cout << "Done.\n";
    return 0;
}
