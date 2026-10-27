// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

// Newton-Raphson optimization based on force constant

#include "io/gpio.h"
#include "apps/nroptimize.h"
#include "potentials/potential_mtp.h"
#include "external/spglib_driver.h"
#include "tools/chem.h"
#include "tools/procstr.h"
#include "tools/tools.h"
#include "tools/constants.h"
#include <cmath>
using namespace Eigen;
using namespace std;

namespace NROptimizeAPP {

// Static Voigt index mapping: Voigt index -> (cartesian row, col)
inline const std::array<std::pair<int, int>, VOIGT_SIZE>& voigt_map() {
    static const std::array<std::pair<int, int>, VOIGT_SIZE> m = {{
        {0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}
    }};
    return m;
}

// Compute pressure-dependent correction to the cell Hessian.
// Must be recomputed whenever the cell changes during optimization.
MatrixXd compute_pdl(const Matrix3d &cell, double pressure) {
    MatrixXd pdl = MatrixXd::Zero(9, 9);
    if (pressure == 0.0) return pdl;

    pdl(0, 4) =  cell(2, 2);
    pdl(0, 5) = -cell(2, 1);
    pdl(0, 7) = -cell(1, 2);
    pdl(0, 8) =  cell(1, 1);
    pdl(1, 3) = -cell(2, 2);
    pdl(1, 5) =  cell(2, 0);
    pdl(1, 6) =  cell(1, 2);
    pdl(1, 8) = -cell(1, 0);
    pdl(2, 3) =  cell(2, 1);
    pdl(2, 4) = -cell(2, 0);
    pdl(2, 6) = -cell(1, 1);
    pdl(2, 7) =  cell(1, 0);
    pdl(3, 7) =  cell(0, 2);
    pdl(3, 8) = -cell(0, 1);
    pdl(4, 6) = -cell(0, 2);
    pdl(4, 8) =  cell(0, 0);
    pdl(5, 6) =  cell(0, 1);
    pdl(5, 7) = -cell(0, 0);
    MatrixXd pdlT = pdl.transpose();
    pdl += pdlT;
    pdl *= pressure;
    return pdl;
}

void compute_hessian(Atoms &atom,              // atom object
                     const Matrix3d &ref,        // reference cell to calculate strain
                     double pressure,          // pressure for pdl correction
                     Potentials::PotMTP &pot, // MTP potential
                     double symprec,           // tolerance for symmetry
                     VectorXd &eigval,         // output: eigenvalues
                     MatrixXd &eigvec,         // output: eigenvectors
                     MatrixXd &fullhessian,    // output: hessian matrix
                     const NROptions &opts)
{
    Message("Computing full HESSIAN matrix");
    auto cell = atom.cell_;
    Potentials::HessianMatrix hess;
    MatrixXd hesscell, lrhess;
    int offset = opts.offset;

    if (opts.nocell) {
        pot.getEFSHessianSymmOMP(atom, hess);
    } else {
        pot.getEFSHessianCellSymmOMP(atom, hess, hesscell, lrhess, symprec);
        // Recompute pdl each call since cell may have changed
        hesscell += compute_pdl(atom.cell_, pressure);
    }

    if (opts.nocell) {
        fullhessian.resize(3 * atom.size(), 3 * atom.size());
        fullhessian.setZero();
        for (int i = 0; i < atom.size(); ++i) {
            for (int j = 0; j <= i; ++j) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        fullhessian(3 * i + a, 3 * j + b) = hess(i, a, j, b);
                    }
                }
            }
        }
    } else {
        const auto &mapid = voigt_map();
        fullhessian.resize(offset + 3 * atom.size(), offset + 3 * atom.size());
        fullhessian.setZero();

        // 1. Convert lattice deformation Hessian to strain Hessian
        Matrix<double, 6, 6> ss = Matrix<double, 6, 6>::Zero();
        for (int i = 0; i < VOIGT_SIZE; ++i) {
            int a = mapid[i].first;
            int b = mapid[i].second;
            double p1 = (a == b) ? 0.5 : 1.0;
            for (int j = 0; j <= i; ++j) {
                int l = mapid[j].first;
                int p = mapid[j].second;
                double p2 = (l == p) ? 0.5 : 1.0;
                for (int t = 0; t < 3; ++t) {
                    for (int w = 0; w < 3; ++w) {
                        ss(i, j) += p1 * p2 *
                                    (ref(w, l) * hesscell(3 * t + b, 3 * w + p) * ref(t, a) +
                                     ref(w, l) * hesscell(3 * t + a, 3 * w + p) * ref(t, b) +
                                     ref(w, p) * hesscell(3 * t + b, 3 * w + l) * ref(t, a) +
                                     ref(w, p) * hesscell(3 * t + a, 3 * w + l) * ref(t, b));
                    }
                }
            }
        }

        // 2. Convert lrhess (lattice-row Hessian) to strain-position coupling
        MatrixXd sp = MatrixXd::Zero(VOIGT_SIZE, 3 * atom.size());
        for (int i = 0; i < VOIGT_SIZE; ++i) {
            int a = mapid[i].first;
            int b = mapid[i].second;
            double p1 = (a == b) ? 0.5 : 1.0;
            for (int iat = 0; iat < 3 * atom.size(); ++iat) {
                for (int t = 0; t < 3; ++t) {
                    sp(i, iat) += p1 * (lrhess(3 * t + b, iat) * ref(t, a) +
                                        lrhess(3 * t + a, iat) * ref(t, b));
                }
            }
        }

        // 3. Build strain-position coupling with force contributions
        MatrixXd sr = MatrixXd::Zero(offset, 3 * atom.size());
        for (int i = 0; i < VOIGT_SIZE; ++i) {
            int a = mapid[i].first;
            int b = mapid[i].second;
            double p1 = (a == b) ? 0.5 : 1.0;
            for (int iat = 0; iat < atom.size(); ++iat) {
                for (int l = 0; l < 3; ++l) {
                    for (int m = 0; m < 3; ++m) {
                        sr(i, 3 * iat + l) += sp(i, 3 * iat + m) * cell(l, m);
                    }
                    sr(i, 3 * iat + l) -= p1 * (atom.forces_[iat](b) * ref(l, a) +
                                                 atom.forces_[iat](a) * ref(l, b));
                }
            }
        }

        // Fill cell-cell block (upper triangle)
        for (int i = 0; i < VOIGT_SIZE; ++i) {
            for (int j = 0; j <= i; ++j) {
                fullhessian(i, j) = ss(i, j);
            }
        }
        // Fill cell-position coupling block
        for (int i = 0; i < VOIGT_SIZE; ++i) {
            for (int j = 0; j < atom.size() * 3; ++j) {
                fullhessian(offset + j, i) = sr(i, j);
            }
        }

        // 4. Convert Cartesian atom Hessian to direct coordinates
        MatrixXd rr = MatrixXd::Zero(atom.size() * 3, atom.size() * 3);
        for (int i = 0; i < atom.size(); ++i) {
            for (int j = 0; j <= i; ++j) {
                for (int a = 0; a < 3; ++a) {
                    for (int b = 0; b < 3; ++b) {
                        for (int l = 0; l < 3; ++l) {
                            for (int m = 0; m < 3; ++m) {
                                rr(3 * i + a, 3 * j + b) += cell(b, m) * hess(i, l, j, m) * cell(a, l);
                            }
                        }
                    }
                }
            }
        }
        for (int i = 0; i < 3 * atom.size(); ++i) {
            for (int j = 0; j <= i; ++j) {
                fullhessian(offset + i, offset + j) = rr(i, j);
            }
        }
    }

    fullhessian = MatrixXd(fullhessian.selfadjointView<Eigen::Lower>());
    SelfAdjointEigenSolver<MatrixXd> solver(fullhessian);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite())
        GPERROR("NR Hessian eigensolver failed");
    eigval = solver.eigenvalues();
    eigvec = solver.eigenvectors();
}

void inverse_hessian(const MatrixXd &fullhessian,
                     const VectorXd &eigval,
                     const MatrixXd &eigvec,
                     MatrixXd &invhessian,
                     const set<int> &fixed_dofs,
                     double eigzero_tol)
{
    const int size = fullhessian.rows();
    std::vector<int> free_dofs;
    for (int i = 0; i < size; ++i)
        if (!fixed_dofs.count(i)) free_dofs.push_back(i);
    invhessian = MatrixXd::Zero(size, size);
    if (free_dofs.empty()) return;
    MatrixXd symmetric = fullhessian.selfadjointView<Eigen::Lower>();
    MatrixXd reduced(free_dofs.size(), free_dofs.size());
    for (int i = 0; i < reduced.rows(); ++i)
        for (int j = 0; j < reduced.cols(); ++j)
            reduced(i, j) = symmetric(free_dofs[i], free_dofs[j]);
    SelfAdjointEigenSolver<MatrixXd> solver(reduced);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite())
        GPERROR("NR free-subspace Hessian eigensolver failed");
    VectorXd inverse_values = solver.eigenvalues();
    for (auto &v : inverse_values)
        v = std::abs(v) < eigzero_tol ? 0.0 : 1.0 / v;
    MatrixXd inverse = solver.eigenvectors() * inverse_values.asDiagonal() * solver.eigenvectors().transpose();
    for (int i = 0; i < reduced.rows(); ++i)
        for (int j = 0; j < reduced.cols(); ++j)
            invhessian(free_dofs[i], free_dofs[j]) = inverse(i, j);
}

void X2LatPos(const VectorXd &x, const Matrix3d &ref, Atoms &atom, const NROptions &opts)
{
    int offset = opts.offset;
    if (opts.nocell) {
        for (int i = 0; i < atom.size(); ++i)
            atom.pos_[i] = x.segment<3>(3 * i).transpose();
    } else {
        // xx yy zz yz zx xy -> symmetric strain matrix
        Eigen::Matrix3d stm;
        stm(0, 0) = x[0] + 1;
        stm(1, 1) = x[1] + 1;
        stm(2, 2) = x[2] + 1;
        stm(1, 2) = x[3];
        stm(2, 1) = x[3];
        stm(0, 2) = x[4];
        stm(2, 0) = x[4];
        stm(0, 1) = x[5];
        stm(1, 0) = x[5];
        atom.cell_ = ref * stm;
        atom.cartesian = false;
        for (int i = 0; i < atom.size(); ++i)
            atom.pos_[i] = x.segment<3>(offset + 3 * i).transpose();
        atom.to_cartesian();
    }
}

void LatPos2X(const Atoms &atom, const Matrix3d &ivref, VectorXd &x, const NROptions &opts)
{
    int offset = opts.offset;
    x.resize(atom.size() * 3 + offset);
    if (opts.nocell) {
        for (int i = 0; i < atom.size(); ++i)
            x.segment<3>(3 * i) = atom.pos_[i];
    } else {
        Eigen::Matrix3d stm = ivref * atom.cell_ - Eigen::Matrix3d::Identity();
        x[0] = stm(0, 0);
        x[1] = stm(1, 1);
        x[2] = stm(2, 2);
        x[3] = stm(1, 2);
        x[4] = stm(0, 2);
        x[5] = stm(0, 1);
        vector<Eigen::RowVector3d> pp;
        atom.get_dir_pos(pp);
        for (int i = 0; i < atom.size(); ++i)
            x.segment<3>(offset + 3 * i) = pp[i];
    }
}

void LatPos2G(const Atoms &atom, const double pressure, const Matrix3d &refcell,
              const Matrix3d &external_stress, VectorXd &g, const NROptions &opts)
{
    int offset = opts.offset;
    g.setZero(opts.offset + 3 * atom.size());
    double pscal = pressure * atom.get_volume();

    if (opts.nocell) {
        for (int i = 0; i < atom.size(); ++i) {
            Eigen::RowVector3d gforce = -atom.forces_[i];
            for (int a = 0; a < 3; ++a)
                g[3 * i + a] = gforce(a);
        }
    } else {
        Matrix3d curr_stress = atom.stress_ - pscal * Eigen::Matrix3d::Identity();
        Matrix3d gstress = -refcell.transpose() * atom.cell_.inverse().transpose() * curr_stress;
        Eigen::Matrix3d ext_str = -refcell.transpose() * atom.cell_.inverse().transpose()
                                  * external_stress * atom.get_volume();
        for (int i = 0; i < VOIGT_SIZE; ++i) {
            const auto [a, b] = voigt_map()[i];
            g(i) = gstress(a, b) + ext_str(a, b);
            if (a != b) g(i) += gstress(b, a) + ext_str(b, a);
        }
        for (int i = 0; i < atom.size(); ++i) {
            Eigen::RowVector3d gfracforce = -atom.cell_ * atom.forces_[i];
            for (int a = 0; a < 3; ++a)
                g[offset + 3 * i + a] = gfracforce(a);
        }
    }
}

// Trust-region safeguard: scale back step if it causes unreasonably large
// energy change (divergence). Does NOT enforce energy decrease — required
// for transition-state searches where the system must climb uphill along
// the unstable mode.
double trust_region_safeguard(Atoms &atom, const VectorXd &x, const VectorXd &dx,
                              const Matrix3d &refcell, Potentials::PotMTP &pot,
                              const NROptions &opts, double alpha, double alpha_reduce,
                              int max_backtracks = 10) {
    const double max_energy_change_per_atom = 10.0; // eV/atom — above this, suspect divergence
    double current_energy = atom.energy_;
    Atoms trial_atom = atom;  // copy

    for (int bt = 0; bt < max_backtracks; ++bt) {
        VectorXd x_trial = x - alpha * dx;
        if (!x_trial.allFinite()) GPERROR("NR step contains non-finite coordinates");
        X2LatPos(x_trial, refcell, trial_atom, opts);
        const double determinant = trial_atom.cell_.determinant();
        if (!trial_atom.cell_.allFinite() || std::abs(determinant) < 1e-12 ||
            determinant * refcell.determinant() <= 0) {
            alpha *= alpha_reduce;
            continue;
        }
        trial_atom.wrap();
        pot.getEFSOMP(trial_atom);
        double de = fabs(trial_atom.energy_ - current_energy) / atom.size();
        bool finite = std::isfinite(de) && trial_atom.stress_.allFinite();
        for (const auto &force : trial_atom.forces_) finite = finite && force.allFinite();
        if (finite && de < max_energy_change_per_atom) {
            atom = trial_atom;
            return alpha;
        }
        alpha *= alpha_reduce;
    }
    GPERROR("NR backtracking exhausted without an admissible finite step");

}

}

int nroptimize(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    Atoms atom;
    GPIO iobj;
    string indexfile = "index.map";
    string infile = "POSCAR";
    string outfile = "CONTCAR";
    string inpot = "potential.mtp";
    string extstress_str = "";
    double pressure = 0.0; // GPa
    double gtol = 0.01;    // eV/A
    int max_iteration = 100;
    double alpha = 0.1;
    double alpha_reduce = 0.9;
    int max_backtracks = 10;
    set<int> fixed_atom_indices;
    NROptimizeAPP::NROptions opts;

    cmdopts.addop(indexfile, "index", "id", "Index file", false, 1)
        .addop(inpot, "potfile", "pot", "Potential file", true, 1)
        .addop(infile, "input", "in", "Input file", true, 1)
        .addop(outfile, "output", "o", "Output file", true, 1)
        .addop(opts.nocell, "no-cell", "nocell", "no cell", false, 0)
        .addop(pressure, "pressure", "p", "pressure in GPa", false, 1)
        .addop(extstress_str, "external-stress", "str", "external stress in GPa", false, 1)
        .addop(gtol, "gtol", "g", "force tolerance", false, 1)
        .addop(max_iteration, "max-iteration", "maxit", "max iteration", false, 1)
        .addop(alpha, "alpha", "a", "alpha", false, 1)
        .addop(alpha_reduce, "alpha-reduce", "ar", "alpha reduce", false, 1)
        .addop(max_backtracks, "max-backtracks", "mbt", "max backtracking steps", false, 1)
        .addop(fixed_atom_indices, "fixed-atoms", "fix", "fixed atom indices (0-based)", false, -1)
        .addop(opts.eigzero_tol, "eigzero-tol", "eigtol", "eigenvalue zero tolerance", false, 1)
        .phrase_options(argc, argv);
    if (max_iteration < 0)
        GPERROR("max-iteration must be nonnegative");
    if (!std::isfinite(gtol) || gtol <= 0)
        GPERROR("gtol must be finite and positive");
    if (!std::isfinite(alpha) || alpha <= 0 || !std::isfinite(alpha_reduce) ||
        alpha_reduce <= 0 || alpha_reduce >= 1 || max_backtracks <= 0 ||
        !std::isfinite(opts.eigzero_tol) || opts.eigzero_tol <= 0 || !std::isfinite(pressure))
        GPERROR("Invalid NR step, backtracking, eigenvalue tolerance or pressure");
    if (GPIO::file_exist(indexfile))
    {
        iobj.readmap(indexfile);
    }
    else
    {
        std::cerr << "ERROR: no index found\n"
                  << std::endl;
        exit(1);
    }

    if (!iobj.read_one(atom, infile) || atom.size() == 0)
        GPERROR("Cannot read a nonempty structure from: " + infile);
    if (!atom.cell_.allFinite() || std::abs(atom.cell_.determinant()) < 1e-12)
        GPERROR("NR requires a finite nonsingular cell");
    if (!atom.cartesian) atom.to_cartesian();
    for (const auto &position : atom.pos_)
        if (!position.allFinite()) GPERROR("NR requires finite positions");
    atom.wrap();
    for (int index : fixed_atom_indices)
        if (index < 0 || index >= atom.size()) GPERROR("fixed-atoms index out of range");

    // Convert fixed atom indices to DOF indices
    set<int> fixed_dofs;
    if (!opts.nocell) {
        // In cell-opt mode, DOFs 0-5 are cell strain, then atomic DOFs
        for (int idx : fixed_atom_indices) {
            for (int a = 0; a < 3; ++a)
                fixed_dofs.insert(opts.offset + 3 * idx + a);
        }
    } else {
        for (int idx : fixed_atom_indices) {
            for (int a = 0; a < 3; ++a)
                fixed_dofs.insert(3 * idx + a);
        }
    }

    Matrix3d external_stress = Matrix3d::Zero();
    if (!extstress_str.empty())
    {
        std::vector<std::string> vstr;
        str_split(extstress_str, vstr);
        if (vstr.size() != 6 && vstr.size() != 3)
        {
            Warning("ERROR: External stress incorrectly set");
            exit(1);
        }
        // xx yy zz yz xz xy
        external_stress(0, 0) = std::stod(vstr[0]);
        external_stress(1, 1) = std::stod(vstr[1]);
        external_stress(2, 2) = std::stod(vstr[2]);
        if (vstr.size() == 6)
        {
            external_stress(1, 2) = std::stod(vstr[3]);
            external_stress(2, 1) = std::stod(vstr[3]);
            external_stress(0, 2) = std::stod(vstr[4]);
            external_stress(2, 0) = std::stod(vstr[4]);
            external_stress(0, 1) = std::stod(vstr[5]);
            external_stress(1, 0) = std::stod(vstr[5]);
        }
        // negative means positive values give negative internal stress
        external_stress /= -eVA3_to_GPa;
    }

    Potentials::PotMTP pot(inpot);
    pot.omp_init();
    pressure *= GPa_to_eVA3;

    // Set up options
    if (opts.nocell)
        opts.offset = 0;
    else
        opts.offset = NROptimizeAPP::VOIGT_SIZE;

    Matrix3d refcell = atom.cell_;
    Matrix3d ivref = refcell.inverse();

    int ndof = opts.offset + 3 * atom.size();
    MatrixXd fullhessian = MatrixXd::Zero(ndof, ndof);
    MatrixXd invhessian = MatrixXd::Zero(ndof, ndof);
    VectorXd eigval = VectorXd::Zero(ndof);
    MatrixXd eigvec = MatrixXd::Zero(ndof, ndof);
    VectorXd x = VectorXd::Zero(ndof);
    VectorXd g = VectorXd::Zero(ndof);
    VectorXd dx = VectorXd::Zero(ndof);

    pot.getEFSOMP(atom);
    NROptimizeAPP::LatPos2X(atom, ivref, x, opts);
    NROptimizeAPP::LatPos2G(atom, pressure, refcell, external_stress, g, opts);
    std::cout << "Iteration " << 0 << " : " << g.cwiseAbs().maxCoeff() << std::endl;

    // Check both the initial structure and the last permitted step.
    int iterations = 0;
    bool converged = false;
    for (;;)
    {
        for (int dof : fixed_dofs) g(dof) = 0;
        if (!g.allFinite() || !std::isfinite(atom.energy_))
            GPERROR("NR produced non-finite energy or gradient");
        double gmax = g.cwiseAbs().maxCoeff();
        if (gmax < gtol)
        {
            converged = true;
            std::cout << "Converged in " << iterations << " iterations" << std::endl;
            break;
        }
        if (iterations == max_iteration) {
            Warning(fmt::format("NR reached maximum iterations ({}) without convergence: "
                                "max gradient = {}, gtol = {}", iterations, gmax, gtol));
            break;
        }

        NROptimizeAPP::compute_hessian(atom, refcell, pressure, pot, 1e-5,
                                       eigval, eigvec, fullhessian, opts);
        NROptimizeAPP::inverse_hessian(fullhessian, eigval, eigvec, invhessian,
                                       fixed_dofs, opts.eigzero_tol);

        dx = invhessian * g;

        // Zero out step for fixed DOFs
        for (int dof : fixed_dofs) {
            if (dof >= 0 && dof < dx.size())
                dx[dof] = 0;
        }

        // Trust-region safeguard: scale back step if diverging
        double accepted_alpha = NROptimizeAPP::trust_region_safeguard(
            atom, x, dx, refcell, pot, opts, alpha, alpha_reduce, max_backtracks);

        x -= accepted_alpha * dx;

        const auto stepfile = "str.iteration." + std::to_string(iterations) + ".cfg";
        if (!iobj.write_one(atom, stepfile, "mlpcfg"))
            GPERROR("Failed to write NR iteration structure: " + stepfile);
        ++iterations;
        NROptimizeAPP::LatPos2G(atom, pressure, refcell, external_stress, g, opts);
        std::cout << "Iteration " << iterations << " : " << g.cwiseAbs().maxCoeff()
                  << "  energy: " << atom.energy_ << "  alpha: " << accepted_alpha << std::endl;

        // Start each Newton step with the requested damping. Reducing alpha
        // unconditionally gives a finite total step budget and can stall at
        // a nonzero gradient; alpha_reduce is for rejected trial steps only.
    }

    if (!iobj.write_one(atom, outfile))
        GPERROR("Failed to write NR final structure: " + outfile);
    return converged ? 0 : 2;
}
