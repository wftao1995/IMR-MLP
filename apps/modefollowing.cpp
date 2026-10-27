// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "apps/modefollowing.h"
#include "tools/tools.h"
#include "mindriver/relax_new.h"
#include "mindriver/dimer_new.h"
#include "mindriver/strain_gradient.h"
#include "analyse/trajectory_transform.h"
#include "tools/constants.h"
#include <array>
#include <cmath>
using namespace std;

/**
 * @brief 
 * 
 * MODE FOLLOWING WITH CELL
 * mode1: read/compute full hessian, then output a movie for specific mode
 * mode2: mode following optimization on specified mode
 * mode3: projection of displacement
 * mode4: Single step NR-iteration
 * @param argc 
 * @param argv 
 * @return int 
 * 
 * set trial_step to zero if rational guess has been given
 */

namespace AppsModeFollowing {

int offset = 6;
bool car_rep = false;
bool nocell = false;

namespace {
constexpr std::array<std::pair<int, int>, 6> strain_indices = {{
    {0, 0}, {1, 1}, {2, 2}, {1, 2}, {0, 2}, {0, 1}
}};

Eigen::Matrix3d cholesky_direction(const Eigen::Matrix3d &L, const Eigen::Matrix3d &rhs) {
    const Eigen::Matrix3d inverse = L.inverse();
    Eigen::Matrix3d phi = inverse * rhs * inverse.transpose();
    phi.triangularView<Eigen::StrictlyUpper>().setZero();
    phi.diagonal() *= .5;
    return L * phi;
}

std::array<Eigen::Matrix3d, 6> cell_directions(const Eigen::Matrix3d &L) {
    std::array<Eigen::Matrix3d, 6> directions;
    for (int k = 0; k < 6; ++k) {
        const auto [a, b] = strain_indices[k];
        Eigen::Matrix3d basis = Eigen::Matrix3d::Zero();
        basis(a,b) = basis(b,a) = 1.0;
        directions[k] = cholesky_direction(L, basis);
    }
    return directions;
}

void validate_structure(const Atoms &atom) {
    if (atom.size() == 0 || atom.chemmap_.size() != atom.pos_.size() ||
        !atom.cell_.allFinite() || !std::isfinite(atom.cell_.determinant()) ||
        atom.cell_.determinant() <= 1e-12)
        GPERROR("Mode following requires a nonempty structure with a positive, nonsingular cell");
    for (const auto &pos : atom.pos_)
        if (!pos.allFinite()) GPERROR("Non-finite mode-following coordinates");
}

void diagonalize(const Eigen::MatrixXd &hessian, Eigen::VectorXd &values, Eigen::MatrixXd &vectors) {
    if (!hessian.allFinite()) GPERROR("Non-finite mode-following Hessian");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(hessian);
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite())
        GPERROR("Mode-following Hessian eigensolver failed");
    values = solver.eigenvalues();
    vectors = solver.eigenvectors();
}
}

void compute_hessian(Atoms &atom, const Eigen::Matrix3d &ref, double pressure,
                     Potentials::PotMTP &pot, double symprec, Eigen::VectorXd &eigval,
                     Eigen::MatrixXd &eigvec, Eigen::MatrixXd &fullhessian) {
    validate_structure(atom);
    Message("Computing full HESSIAN matrix");
    Potentials::HessianMatrix hess;
    Eigen::MatrixXd cellhess, mixed;
    if (nocell) pot.getEFSHessianSymmOMP(atom, hess);
    else pot.getEFSHessianCellSymmOMP(atom, hess, cellhess, mixed, symprec);

    const int n = 3 * atom.size();
    Eigen::MatrixXd cart(n, n);
    for (int i = 0; i < atom.size(); ++i)
        for (int j = 0; j < atom.size(); ++j)
            cart.block<3, 3>(3 * i, 3 * j) = hess(i, j);
    fullhessian.setZero(offset + n, offset + n);

    if (!nocell) {
        // d²(p det H)/dH_ab dH_cd depends on the CURRENT cell, not the initial cell.
        const Eigen::Matrix3d inv = atom.cell_.inverse();
        const double pv = pressure * atom.get_volume();
        for (int a = 0; a < 3; ++a)
            for (int b = 0; b < 3; ++b)
                for (int c = 0; c < 3; ++c)
                    for (int d = 0; d < 3; ++d)
                        cellhess(3*a+b, 3*c+d) += pv *
                            (inv(b,a)*inv(d,c) - inv(b,c)*inv(d,a));
    }
    if (car_rep) {
        fullhessian.bottomRightCorner(n, n) = cart;
        if (!nocell) {
            // The MTP cell derivatives deform fractional positions with H.
            // For independent (H,r), subtract the affine displacement s dH.
            Eigen::MatrixXd affine = Eigen::MatrixXd::Zero(n, 9);
            for (int i = 0; i < atom.size(); ++i) {
                Eigen::RowVector3d frac = atom.car2dir(atom.pos_[i]);
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        affine(3*i+b, 3*a+b) = frac(a);
            }
            fullhessian.topLeftCorner(9, 9) = cellhess - mixed * affine -
                affine.transpose() * mixed.transpose() + affine.transpose() * cart * affine;
            fullhessian.topRightCorner(9, n) = mixed - affine.transpose() * cart;
        }
    } else {
        Eigen::MatrixXd atomic = Eigen::MatrixXd::Zero(n, n);
        for (int i = 0; i < atom.size(); ++i)
            atomic.block<3, 3>(3*i, 3*i) = atom.cell_.transpose();
        fullhessian.bottomRightCorner(n, n) = atomic.transpose() * cart * atomic;
        if (!nocell) {
            const Eigen::Matrix3d L = ref.inverse() * atom.cell_;
            const auto directions = cell_directions(L);
            Eigen::MatrixXd cell(9, 6);
            for (int k = 0; k < 6; ++k) {
                const Eigen::Matrix3d tangent = ref * directions[k];
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) cell(3*a+b,k) = tangent(a,b);
            }
            fullhessian.topLeftCorner(6, 6) = cell.transpose() * cellhess * cell;
            fullhessian.topRightCorner(6, n) = cell.transpose() * mixed * atomic;
            const Eigen::Matrix3d cell_gradient = -atom.cell_.inverse().transpose() *
                (atom.stress_ - pressure * atom.get_volume() * Eigen::Matrix3d::Identity());
            for (int i = 0; i < 6; ++i)
                for (int j = 0; j < 6; ++j) {
                    const Eigen::Matrix3d second = ref * cholesky_direction(L,
                        -directions[i]*directions[j].transpose() - directions[j]*directions[i].transpose());
                    fullhessian(i,j) += (cell_gradient.array() * second.array()).sum();
                }
            // r = s H(C) also contributes mixed coordinate derivatives.
            // k indexes SIX strain components; i indexes the atoms independently.
            for (int k = 0; k < 6; ++k)
                for (int i = 0; i < atom.size(); ++i)
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b)
                            fullhessian(k, offset+3*i+a) -= atom.forces_[i](b) * cell(3*a+b, k);
        }
    }
    fullhessian.bottomLeftCorner(n, offset) = fullhessian.topRightCorner(offset, n).transpose();
    fullhessian = (0.5 * (fullhessian + fullhessian.transpose())).eval();
    diagonalize(fullhessian, eigval, eigvec);
}

Eigen::VectorXd gradient(const Atoms &atom, const Eigen::Matrix3d &ref, double pressure) {
    Eigen::VectorXd g(offset + 3 * atom.size());
    if (!nocell) {
        Eigen::Matrix3d cellgrad = -atom.cell_.inverse().transpose() *
            (atom.stress_ - pressure * atom.get_volume() * Eigen::Matrix3d::Identity());
        if (car_rep) {
            for (int i = 0; i < atom.size(); ++i)
                cellgrad += atom.car2dir(atom.pos_[i]).transpose() * atom.forces_[i].transpose();
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b) g(3*a+b) = cellgrad(a,b);
        } else {
            const auto metric_gradient = SaddleCoordinates::symmetric_pullback(
                ref.inverse() * atom.cell_, ref.transpose() * cellgrad, true);
            constexpr int order[] = {0, 3, 5, 4, 2, 1};
            for (int k = 0; k < 6; ++k) g(k) = metric_gradient(order[k]);
        }
    }
    for (int i = 0; i < atom.size(); ++i) {
        if (car_rep) g.segment<3>(offset+3*i) = -atom.forces_[i];
        else g.segment<3>(offset+3*i) = -atom.cell_ * atom.forces_[i];
    }
    if (!g.allFinite()) GPERROR("Non-finite mode-following gradient");
    return g;
}

Eigen::MatrixXd optimizer_modes(const Eigen::MatrixXd &modes) {
    Eigen::MatrixXd result = modes;
    if (!nocell && !car_rep) {
        // RelaxationDrvV2's symmetric metric order is xx xy xz yy yz zz.
        constexpr int order[] = {0, 5, 4, 1, 3, 2};
        for (int i = 0; i < 6; ++i) result.row(i) = modes.row(order[i]);
    }
    return result;
}

namespace {
class ModeGradient : public AnyModifier {
    Eigen::Matrix3d ref;
    double pressure;
public:
    ModeGradient(const Eigen::Matrix3d &reference, double p) : ref(reference), pressure(p) {}
    void forward(RelaxationDrvV2 *opt) override {
        opt->g = optimizer_modes(gradient(opt->atom, ref, pressure));
    }
};
}

void init_relaxation(RelaxationDrvV2 &opt, Potentials::PotMTP &pot, Atoms &atom,
                     const Eigen::Matrix3d &ref, double pressure) {
    const auto &f = opt.flags;
    if (!f.relax_atom || f.keep_symm || f.fix_vol || f.fix_shape || f.selective_dynamics ||
        f.constraint_vib_modes || f.linear_constraint || f.linear_transform || f.freeze_small_grad ||
        f.fix_cm || f.symm_force || f.apply_disp || f.with_constraint_aul || f.with_constraint_sqp ||
        f.with_constraint_nlc || f.with_constraint_lc || opt.control.couple_flag != 0 ||
        !opt.control.external_stress_str.empty())
        GPERROR("Mode following does not support additional relaxation constraints or anisotropic stress");
    // The CLI coordinate convention is authoritative for BOTH Hessian and optimizer.
    opt.flags.cartesian = car_rep;
    opt.flags.relax_cell = !nocell;
    opt.flags.relax_cell_use_strain = !car_rep;
    opt.flags.symm_strain = !car_rep;
    opt.flags.atomic_force_recast = false;
    opt.flags.dprecond = false;
    opt.use_dprecond = false;
    opt.flags.vibffilter = false;
    opt.control.pressure = pressure; // Already eV/A^3; constructor conversion has finished.
    opt.init(&pot, atom);
    opt.refcell = ref;
    opt.ivrefcell = ref.inverse();
    opt.LatPos2X();
    // run_impl applies modifiers once before eval(), including for a file-loaded Hessian.
    pot.getEFSOMP(opt.atom);
    // Recompute the exact canonical gradient before applying the mode filter.
    // In particular, a Cartesian cell change holds absolute atomic positions fixed.
    opt.modifiers_list.insert(opt.modifiers_list.begin(), new ModeGradient(ref, pressure));
}

void inverse_hessian(Eigen::MatrixXd &fullhessian, 
            Eigen::VectorXd &eigval,
            Eigen::MatrixXd &eigvec,
            Eigen::MatrixXd &invhessian,
            double eigzero_tol) {
    int size = eigval.size();
    invhessian.resize(size, size);
    invhessian.setZero();
    for (int i = 0; i < size; ++i) {
        if (fabs(eigval(i)) < eigzero_tol) {
            invhessian(i, i) = 0;
        } else {
            invhessian(i, i) = 1.0 / eigval(i);
        }
    }
    invhessian = eigvec * invhessian * eigvec.transpose();
}

void LatPos2X(Atoms &atom, Eigen::RowVectorXd &x) {
    x.resize(atom.size() * 3 + offset);
    if (!nocell) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                x[3*a+b] = atom.cell_(a, b);
            }
        }
    }
    memcpy(x.data() + offset, atom.pos_[0].data(), 3*atom.size() * sizeof(double));
}

void X2LatPos(Atoms &atom, Eigen::RowVectorXd &x) {
    if (!nocell) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                atom.cell_(a, b) = x[3*a+b];
            }
        }
    }
    memcpy(atom.pos_[0].data(), x.data() + offset, 3*atom.size() * sizeof(double));
    validate_structure(atom);
} 

void LatPos2x(Atoms &atom, Eigen::Matrix3d &ivref, Eigen::RowVectorXd &x) {
    x.resize(atom.size() * 3 + offset);
    if (!nocell) {
        const Eigen::Matrix3d L = ivref * atom.cell_;
        Eigen::Matrix3d stm = L * L.transpose() - Eigen::Matrix3d::Identity();
        x[0] = stm(0, 0);
        x[1] = stm(1, 1);
        x[2] = stm(2, 2);
        x[3] = stm(1, 2);
        x[4] = stm(0, 2);
        x[5] = stm(0, 1);
    }
    vector<Eigen::RowVector3d> pp;
    atom.get_dir_pos(pp);
    memcpy(x.data() + offset, pp[0].data(), 3*atom.size() * sizeof(double));
}

void x2LatPos(Atoms &atom, Eigen::Matrix3d &ref, Eigen::RowVectorXd &x) {
    if (!nocell) {
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
        Eigen::LLT<Eigen::Matrix3d> factor(stm);
        if (factor.info() != Eigen::Success)
            GPERROR("Mode-following strain metric is not positive definite");
        atom.cell_ = ref * Eigen::Matrix3d(factor.matrixL());
    }
    atom.cartesian = false;
    memcpy(atom.pos_[0].data(), x.data() + offset, 3*atom.size() * sizeof(double));
    atom.to_cartesian();
    validate_structure(atom);
}

void NugeMode(Atoms &atom, Eigen::MatrixXd &mode, Eigen::Matrix3d &ref, Eigen::Matrix3d &ivref, int imode, double step) {
    if (imode < 0 || imode >= mode.cols() || mode.rows() != offset + 3*atom.size() ||
        !mode.allFinite() || !std::isfinite(step)) GPERROR("Invalid mode displacement");
    static Eigen::RowVectorXd dx;
    dx.resize(offset + 3*atom.size());
    dx.setZero();
    dx[imode] = step;
    dx *= mode.transpose();
    Eigen::RowVectorXd xx;
    if (car_rep) {
        LatPos2X(atom, xx);
        xx += dx;
        X2LatPos(atom, xx);
    } else {
        LatPos2x(atom, ivref, xx);
        xx += dx;
        x2LatPos(atom, ref, xx);
    }
}

int FindMode(Eigen::MatrixXd &eigvec, Eigen::VectorXd &lastmode) {
    if (eigvec.rows() != lastmode.size() || eigvec.cols() == 0 || !eigvec.allFinite() ||
        !lastmode.allFinite() || lastmode.norm() < 1e-14) GPERROR("Invalid mode for overlap tracking");
    double maxcrit = 0.0;
    double sgn = 1.0;
    int maxmode = 0;
    double llast = lastmode.norm();
    for (int i = 0; i < eigvec.cols(); ++i) {
        Eigen::VectorXd mm = eigvec.col(i);
        double crit = mm.dot(lastmode) / mm.norm() / llast;
        if (fabs(crit) > fabs(maxcrit)) {
            maxcrit = crit;
            maxmode = i;
            sgn = crit > 0 ? 1.0 : -1.0;
        }
    }
    if (sgn < 0) {
        eigvec.col(maxmode) *= -1;
    }
    lastmode = eigvec.col(maxmode);
    return maxmode;
}

}

int modefollowing(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    Atoms atom, supercell;
    std::vector<Atoms> atoms;
    string infile = "POSCAR";
    string infile2 = "POSCAR_LAM";
    string informat = "POSCAR";
    string indexfile = "index.map";
    string settingfile = "modefollowing.ini";
    string inpot = "potential.mtp";
    GPIO iobj;
    GPSettings settings;
    double symprec = 1e-3; // ok
    bool updateref = false; // ok
    string read_hessian = ""; // ok

    double eigzero_tol = 0.008; // ok
    double next_step_tol = -0.1; // ok

    double ifac = 0; // ok, relax following modes
    double ofac = 1; // ok, relax + modes
    double sfac = 1; // ok, relax non-following - modes
    
    bool cycle = false;
    bool linetrajectory = false;

    string modecar_out = "MODECAR";
    AppsModeFollowing::offset = 6;
    AppsModeFollowing::car_rep = false;
    AppsModeFollowing::nocell = false;

    int mode = 1; // 1 compute/read hessian; 2 compute mode following, ok
    int imode = 4; // ok
    double amp = 0.1; // displacement magnitude, ok
    int frames = 16; // ok
    double pressure = 0.0; // ok
    bool request_efs = false; // ok

    double trial_step = 0.1; // step, ok
    int maxloop = 100;
    int result_code = 0;

    cmdopts.addop(indexfile, "index", "id", "index file")
        .addop(infile, "input", "in", "input structure file")
        .addop(infile2, "input2", "in2", "second structure for modes 3 and 7")
        .addop(informat, "input-format", "if", "format, use XDATCAR or lammpstrj or mlpcfg")
        .addop(maxloop, "max-loop", "", "Maximum iterations per mode-following stage")
        .addop(trial_step, "trial-step", "", "trial step size")
        .addop(symprec, "symprec", "tol", "tolerance of symmetry finding")
        .addop(next_step_tol, "next-tol", "ntol", "next step tol")
        .addop(AppsModeFollowing::car_rep, "car-rep", "", "use cartsian coordinates", false, 0)
        .addop(AppsModeFollowing::nocell, "nocell", "", "do not compute cell part of hessian matrix", false, 0)
        .addop(ifac, "ifac", "", "ifac, default = 0")
        .addop(ofac, "ofac", "", "ofac, default = 1")
        .addop(sfac, "sfac", "", "sfac, default = 1")
        .addop(frames, "frames", "", "number of frames")
        .addop(cycle, "cycle", "", "when generation vibration mode structure sequence, whather use cyclic loop", false, 0)
        .addop(pressure, "pressure", "p", "external pressure")
        .addop(updateref, "updateref", "", "update reference lattice parameter or not, when cell changes too large, this might help", false, 0)
        .addop(read_hessian, "hessian", "", "load hessian matrix from this file")
        .addop(linetrajectory, "linetraj", "", "create trajectory using linear relation", false, 0)
        .addop(imode, "imode", "i", "which mode to follow")
        .addop(modecar_out, "modecar-out", "mcout", "MODECAR output file (for mode 9)")
        .addop(mode, "mode", "m", "1=animation, 2=saddle, 3=displacement, 4=Newton, 6=nudge, 7=correlation, 8=gradient, 9=MODECAR; 5 is unimplemented")
        .addop(amp, "amplitude", "a", "amplitude")
        .addop(eigzero_tol, "eigzero-tol", "", "tolerance of zero eigenvalues")
        .addop(request_efs, "request-efs", "efs", "request efs")
        .addop(inpot, "potfile", "pot", "potential file, should be MTP-based type currently")
        .addop(settingfile, "settings", "", "input setting file", true)
        .add_request("settings")
        .attach_settings("relax:", "Relaxation driver",
            [](std::ostream &os) { RelaxationDrvV2().ListSettings(os, "relax:"); });
    cmdopts.phrase_options(argc, argv);
    if (!GPIO::file_exist(settingfile)) GPERROR("Cannot open mode-following settings: " + settingfile);
    if (GPIO::file_exist(settingfile))
    {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    if (maxloop < 0) GPERROR("max-loop must be nonnegative");
    if (GPIO::file_exist(indexfile)) {
        iobj.readmap(indexfile);
    } else {
        GPERROR("Cannot open mode-following index file: " + indexfile);
    }
    if (AppsModeFollowing::car_rep) AppsModeFollowing::offset = 9;
    if (AppsModeFollowing::nocell) AppsModeFollowing::offset = 0;
    auto &offset = AppsModeFollowing::offset;

    if (next_step_tol > 0) next_step_tol *= -1.0;

    if (mode < 1 || mode > 9 || mode == 5)
        GPERROR("Unsupported mode: " + to_string(mode) + "; supported modes are 1,2,3,4,6,7,8,9 (5 is unimplemented)");
    if (maxloop < 0) GPERROR("max-loop must be nonnegative");
    if (mode == 1 && frames < 2) GPERROR("frames must be at least 2");
    if (!std::isfinite(eigzero_tol) || eigzero_tol <= 0 ||
        !std::isfinite(symprec) || symprec <= 0)
        GPERROR("eigzero-tol and symprec must be finite and positive");
    for (double value : {pressure, trial_step, amp, next_step_tol, ifac, ofac, sfac})
        if (!std::isfinite(value)) GPERROR("Mode-following parameters must be finite");

    if (!iobj.read(atoms, infile, informat))
        GPERROR("Failed to read complete mode-following input: " + infile);
    if (atoms.empty() || atoms[0].size() == 0) GPERROR("Mode following requires a nonempty input structure");
    atom = atoms[0];
    AppsModeFollowing::validate_structure(atom);
    atom.wrap();

    Potentials::PotMTP pot(inpot);
    pot.omp_init();
    pressure *= GPa_to_eVA3;

    Eigen::MatrixXd fullhessian;
    Eigen::VectorXd eigval;
    Eigen::MatrixXd eigvec;
    Eigen::VectorXd lastmode;

    fullhessian.resize(offset + 3*atom.size(), offset + 3*atom.size());
    eigval.resize(offset + 3*atom.size());
    eigvec.resize(offset + 3*atom.size(), offset + 3*atom.size());
    lastmode.resize(eigval.size());

    auto relax_settings = settings.ExtractSubSettings("relax");
    if (!relax_settings.count("driver")) relax_settings.Modify("driver", "fire");
    RelaxationDrvV2 opt(relax_settings);
    if (mode == 2 && opt.control.driver != "fire")
        GPERROR("Mode following requires relax:driver fire; energy-decreasing line searches cannot climb a mode");
    if (mode == 2 && (!std::isfinite(opt.tol.gtol) || opt.tol.gtol <= 0))
        GPERROR("relax:gtol must be finite and positive");

    Eigen::Matrix3d ref = atom.cell_;
    Eigen::Matrix3d ivref = ref.inverse();

    if (read_hessian.empty())
        AppsModeFollowing::compute_hessian(atom, ref, pressure, pot, symprec, eigval, eigvec, fullhessian);
    else {
        ifstream ifs(read_hessian);
        if (!ifs.is_open()) GPERROR("Cannot open hessian file: " + read_hessian);
        fullhessian.resize(offset + atom.size() * 3, offset + atom.size() * 3);
        fullhessian.setZero();
        for (int i = 0; i < offset + atom.size() * 3; ++i) {
            for (int j = 0; j < offset + atom.size() * 3; ++j) {
                ifs >> fullhessian(i, j);
            }
        }
        if (ifs.fail()) GPERROR("Failed to read hessian matrix: " + read_hessian);
        ifs >> std::ws;
        if (!ifs.eof()) GPERROR("Unexpected trailing data in hessian file");
        if (!fullhessian.allFinite() || !fullhessian.isApprox(fullhessian.transpose(), 1e-10))
            GPERROR("Input hessian must be finite and symmetric");
        AppsModeFollowing::diagonalize(fullhessian, eigval, eigvec);
    }
    if ((mode == 2 || mode == 6 || mode == 9) && (imode < 0 || imode >= eigvec.cols()))
        GPERROR("Mode index is out of range");
    auto write_structure = [&](Atoms &a, const string &file) {
        if (!iobj.write_one(a, file)) GPERROR("Failed to write " + file);
    };
    auto read_second = [&]() {
        vector<Atoms> second;
        if (!iobj.read(second, infile2, informat))
            GPERROR("Failed to read complete second input: " + infile2);
        if (second.empty()) GPERROR("Second input structure is empty");
        Atoms result = second.back();
        AppsModeFollowing::validate_structure(result);
        if (result.size() != atom.size() || result.chemmap_ != atom.chemmap_ || result.pbc_ != atom.pbc_)
            GPERROR("Input structures must have matching atom counts, species order and periodicity");
        if (AppsModeFollowing::nocell && !result.cell_.isApprox(atom.cell_, 1e-10))
            GPERROR("Fixed-cell comparison requires matching input cells");
        if (!AppsModeFollowing::nocell && !AppsModeFollowing::car_rep) {
            const Eigen::Matrix3d deformation = ivref * result.cell_;
            if (deformation.triangularView<Eigen::StrictlyUpper>().toDenseMatrix().norm() > 1e-10 ||
                deformation.diagonal().minCoeff() <= 0) {
                Eigen::LLT<Eigen::Matrix3d> factor(deformation * deformation.transpose());
                if (factor.info() != Eigen::Success)
                    GPERROR("Second cell strain metric is not positive definite");
                const Eigen::Matrix3d lower = factor.matrixL();
                const Eigen::Matrix3d rotation = deformation.inverse() * lower;
                result.cell_ = ref * lower;
                for (auto &position : result.pos_) position *= rotation;
            }
        }
        return result;
    };

    if (mode == 1) {
        // compute hessian info
        if (read_hessian.empty()) {
            ofstream ofsa("hessianfull");
            for (int i = 0; i < offset+3*atom.size(); ++i) {
                ofsa << fmt::format(" {} \n", fmt::join(fullhessian.row(i), " "));
            }
            ofsa.close();
            if (!ofsa) GPERROR("Failed to write hessianfull");
        }
        // output eigenvalues and eigenvectors
        ofstream ofs("EigensystemFull.dat");
        for (int i = 0; i < eigval.size(); ++i) {
            ofs << fmt::format(" {:.17g}  {:.17g} \n", eigval(i), fmt::join(eigvec.col(i), " "));
        }
        ofs.close();
        if (!ofs) GPERROR("Failed to write EigensystemFull.dat");
        ofstream ofsb("EIGSYS");
        for (int i = 0; i < eigval.size(); ++i) {
            ofsb << fmt::format(" {:.17g} \n", fmt::join(eigvec.col(i), " "));
        }
        ofsb.close();
        if (!ofsb) GPERROR("Failed to write EIGSYS");
        // create animate for each mode
        // -1 to 1
        vector<double> phi(frames);
        if (linetrajectory) {
            double dphi = 2. / (frames - 1.);
            for (int i = 0; i < frames; ++i) {
                phi[i] = amp * (-1. + i * dphi);
            }
        } else {
            for (int i = 0; i < frames; ++i) {
                if (cycle) 
                    phi[i] = amp * sin(-M_PI/2 + 2*M_PI / frames * i);
                else
                    phi[i] = amp * sin(-M_PI/2 + M_PI / (frames - 1) * i);
            }
        }
        for (int i = 0; i < eigval.size(); ++i) {
            string fnm = "vibmode_meta_" + to_string(i) + ".xyz";
            double eval = eigval(i);
            vector<Atoms> ats(frames);
            for (int ia = 0; ia < frames; ++ia) {
                ats[ia] = atom;
                ats[ia].clearEFS();
                ats[ia].have_fenergy = ats[ia].have_hessian = false;
                ats[ia].fenergy_ = 0.;
                ats[ia].hessian_.resize(0, 0);
                ats[ia].clear_hvp();
                AppsModeFollowing::NugeMode(ats[ia], eigvec, ref, ivref, i, phi[ia]);
                if (request_efs) {
                    ats[ia].wrap();
                    pot.getEFS(ats[ia]);
                    ats[ia].global_attribs_d_["fenergy"] = ats[ia].energy_ + pressure * ats[ia].get_volume();
                } 
                ats[ia].global_attribs_d_["freq"] = eval;
                ats[ia].global_attribs_d_["volume"] = ats[ia].get_volume();
            }
            if (!iobj.write(ats, fnm)) GPERROR("Failed to write " + fnm);
        } 
    }
    else if (mode == 2) {
        lastmode = eigvec.col(imode);
        auto update_modes = [&]() {
            if (updateref) {
                if (!AppsModeFollowing::nocell && !AppsModeFollowing::car_rep) {
                    Eigen::Matrix3d strain;
                    strain << lastmode(0), lastmode(5), lastmode(4),
                              lastmode(5), lastmode(1), lastmode(3),
                              lastmode(4), lastmode(3), lastmode(2);
                    Eigen::Matrix3d rebased = atom.cell_.inverse() * ref * strain;
                    rebased = (0.5 * (rebased + rebased.transpose())).eval();
                    for (int k = 0; k < 6; ++k) {
                        auto [a,b] = AppsModeFollowing::strain_indices[k];
                        lastmode(k) = rebased(a,b);
                    }
                    lastmode.normalize();
                }
                ref = atom.cell_;
                ivref = ref.inverse();
            }
            AppsModeFollowing::compute_hessian(atom, ref, pressure, pot, symprec, eigval, eigvec, fullhessian);
            imode = AppsModeFollowing::FindMode(eigvec, lastmode);
        };
        // First relax orthogonal modes, then take a trial step along the followed mode.
        for (int iloop = 0; iloop < maxloop; ++iloop) {
            write_structure(atom, "POSCAR" + to_string(iloop) + "_stage.vasp");
            AppsModeFollowing::init_relaxation(opt, pot, atom, ref, pressure);
            Eigen::VectorXd factors = Eigen::VectorXd::Constant(eigvec.cols(), ofac);
            for (int i = 0; i < eigval.size(); ++i)
                if (eigval(i) < -eigzero_tol) factors(i) = sfac;
            factors(imode) = ifac;
            opt.set_vib_data(AppsModeFollowing::optimizer_modes(eigvec), factors);
            opt.run();
            atom = opt.atom;
            write_structure(atom, "CONTCAR_stage" + to_string(iloop) + ".vasp");
            update_modes();
            if (eigval(imode) < next_step_tol) break;
            AppsModeFollowing::NugeMode(atom, eigvec, ref, ivref, imode, trial_step);
            // Eigenvalue ordering can change at the trial point; track BEFORE the next filter.
            update_modes();
        }
        write_structure(atom, "POSCAR_SAD_INI");
        Message("Second loop");
        bool saddle_converged = false;
        for (int iloop = 0; iloop < maxloop; ++iloop) {
            AppsModeFollowing::init_relaxation(opt, pot, atom, ref, pressure);
            Eigen::VectorXd factors = Eigen::VectorXd::Ones(eigvec.cols());
            factors(imode) = -1;
            opt.set_vib_data(AppsModeFollowing::optimizer_modes(eigvec), factors);
            const int optstate = opt.run();
            atom = opt.atom;
            update_modes();
            if (optstate == 0) { saddle_converged = true; break; }
        }
        // Test the actual enthalpy gradient, including the cell, and the full Hessian index.
        AppsModeFollowing::compute_hessian(atom, ref, pressure, pot, symprec, eigval, eigvec, fullhessian);
        pot.getEFSOMP(atom);
        const double residual = AppsModeFollowing::gradient(atom, ref, pressure).cwiseAbs().maxCoeff();
        double max_force = 0;
        for (const auto &force : atom.forces_) max_force = std::max(max_force, force.cwiseAbs().maxCoeff());
        const int negative_modes = (eigval.array() < -eigzero_tol).count();
        saddle_converged = saddle_converged && residual <= opt.tol.gtol &&
            std::isfinite(max_force) && max_force <= opt.tol.gtol && negative_modes == 1;
        Message(fmt::format("Mode following status: {}, max atomic force = {}, max gradient = {}, negative modes = {}",
                            saddle_converged ? "converged" : "not_converged", max_force, residual, negative_modes));
        result_code = saddle_converged ? 0 : 2;
    }
    else if (mode == 3) {
        Atoms atom2 = read_second();
        vector<Atoms> path{atom, atom2};
        unwarp(path);
        atom2 = path[1];
        Eigen::RowVectorXd first, second;
        if (AppsModeFollowing::car_rep) {
            AppsModeFollowing::LatPos2X(atom, first);
            AppsModeFollowing::LatPos2X(atom2, second);
        } else {
            AppsModeFollowing::LatPos2x(atom, ivref, first);
            AppsModeFollowing::LatPos2x(atom2, ivref, second);
        }
        // Preserve the historical input1-input2 displacement sign.
        Eigen::RowVectorXd dx = first - second;
        dx *= eigvec;
        vector<pair<int, double>> dxx;
        dxx.resize(dx.size());
        for (int i = 0; i < dx.size(); ++i) {
            dxx[i].first = i;
            dxx[i].second = dx[i];
        }
        sort(dxx.begin(), dxx.end(), [](const pair<int, double> &a, const pair<int, double> &b) {
            if (fabs(a.second) > fabs(b.second)) return true; return false;
        });
        for (int i = 0; i < dxx.size(); ++i) {
            std::cout << " " << dxx[i].first << " " << dxx[i].second << std::endl;
        }
    } else if (mode == 4) {
        // Do a single NR iteration
        Eigen::MatrixXd invhessian;
        AppsModeFollowing::inverse_hessian(fullhessian, eigval, eigvec, invhessian, eigzero_tol);
        pot.getEFSOMP(atom);
        const Eigen::VectorXd g = AppsModeFollowing::gradient(atom, ref, pressure);
        Eigen::RowVectorXd dx = (invhessian * g).transpose();
        Eigen::RowVectorXd xx;
        if (AppsModeFollowing::car_rep) {
            AppsModeFollowing::LatPos2X(atom, xx);
            xx -= dx;
            AppsModeFollowing::X2LatPos(atom, xx);
        } else {
            AppsModeFollowing::LatPos2x(atom, ivref, xx);
            xx -= dx;
            AppsModeFollowing::x2LatPos(atom, ref, xx);
        }
        AppsModeFollowing::validate_structure(atom);
        write_structure(atom, "POSCAR_NR_STEP");
    } else if (mode == 6) {
        AppsModeFollowing::NugeMode(atom, eigvec, ref, ivref, imode, trial_step);
        write_structure(atom, "POSCAR_NUGED");
    } else if (mode == 7) {
        Atoms atom2 = read_second();
        auto eigval2 = eigval;
        auto eigvec2 = eigvec;
        auto fullhessian2 = fullhessian;
        AppsModeFollowing::compute_hessian(atom2, ref, pressure, pot, symprec, eigval2, eigvec2, fullhessian2);
        Eigen::MatrixXd senmat;
        senmat.resize(offset + atom.size() * 3, offset + atom.size() * 3);
        senmat.setZero();
        map<int, int> bestmatch;
        for (int i = 0; i < offset + atom.size() * 3; ++i) {
            double maxmatch = 0;
            int maxid = 0;
            for (int j = 0; j < offset + atom.size() * 3; ++j) {
                senmat(i, j) = 
                    fabs(eigvec.col(i).dot(eigvec2.col(j)) / eigvec.col(i).norm() / eigvec2.col(j).norm());
                if (senmat(i, j) > maxmatch) {
                    maxmatch = senmat(i, j);
                    maxid = j;
                }
            }
            bestmatch[i] = maxid;
        }
        cout << "     ";
        for (int i = 0; i < eigval.size(); ++i) {
            cout << fmt::format(" {:10} ", i);
        }
        cout << "\n";
        for (int i = 0; i < eigval.size(); ++i) {
            cout << fmt::format(" {:5} {:10.6f} \n", i, fmt::join(senmat.row(i), " "));
        }
        // output best match 
        cout << "BEST MATCH: in1 in2 \n";
        for (auto &it : bestmatch) {
            cout << it.first << " " << it.second << "\n";
        }
    } else if (mode == 8) {
        pot.getEFSOMP(atom);
        Eigen::RowVectorXd gp = AppsModeFollowing::gradient(atom, ref, pressure).transpose() * eigvec;
        for (int i = 0; i < gp.size(); ++i) {
            std::cout << fmt::format("{} {}\n", i, gp(i));
        }
    }
    else if (mode == 9) {
        if (imode < 0 || imode >= eigvec.cols())
            GPERROR("MODECAR mode index is out of range");

        // Preserve the cell map when converting to DIMER optimization coordinates.
        DimerDriver output;
        output.vc_flag = !AppsModeFollowing::nocell;
        output.symm_strain = !AppsModeFollowing::car_rep;
        output.restore_F_cholesky = output.symm_strain;
        output.recast_atomic_force = true;
        output.natom = atom.size();
        output.offset = offset;
        output.nlen = offset + 3 * atom.size();
        output.refcell = ref;
        output.modedir = Eigen::VectorXd::Zero(output.nlen);
        Eigen::Matrix3d dcell = Eigen::Matrix3d::Zero();
        if (output.vc_flag) {
            if (AppsModeFollowing::car_rep) {
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        dcell(a, b) = eigvec(3 * a + b, imode);
                Eigen::Matrix3d deformation = ivref * dcell;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        output.modedir(3 * a + b) = deformation(a, b);
            } else {
                // Both use C-I; only the component order differs.
                const int source_index[] = {0, 5, 4, 1, 3, 2};
                for (int i = 0; i < 6; ++i)
                    output.modedir(i) = eigvec(source_index[i], imode);
            }
        }
        for (int i = 0; i < atom.size(); ++i) {
            Eigen::RowVector3d displacement = eigvec.col(imode).segment<3>(offset + 3 * i).transpose();
            if (!AppsModeFollowing::car_rep)
                displacement = (displacement * ref).eval();
            else if (output.vc_flag)
                // q = s H_ref, r = s H: dq = dr - s dH at H = H_ref.
                displacement -= atom.car2dir(atom.pos_[i]) * dcell;
            output.modedir.segment<3>(offset + 3 * i) = displacement.transpose();
        }
        const double norm = output.modedir.stableNorm();
        if (!output.modedir.allFinite() || !std::isfinite(norm) || norm < 1e-12)
            GPERROR("Cannot export a non-finite or zero MODECAR direction");
        output.modedir /= norm;
        output.write_modecar(modecar_out);
        if (output.vc_flag)
            Message(fmt::format("MODECAR requires the same reference cell and dimer:vcflag true, "
                                "dimer:symm_strain {}, dimer:restore_F_cholesky {}, "
                                "dimer:recast_atomic_force true", output.symm_strain,
                                output.restore_F_cholesky));
        Message(fmt::format("MODECAR written to '{}', mode index = {}, eigenvalue = {:.6f}",
                            modecar_out, imode, eigval(imode)));
    }
    else {
        Message("Unknown mode");
    }

    // complex 

    // iobj.set(atom);
    if (mode == 1) {
        // Mode 1 publishes the unchanged reference as geometry after animation.
        Atoms geometry = atom;
        geometry.have_hessian = false;
        geometry.hessian_.resize(0, 0);
        geometry.clear_hvp();
        write_structure(geometry, "POSCAR_SAD");
    } else {
        write_structure(atom, "POSCAR_SAD");
    }

    return result_code;
}
