// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "dimer_new.h"
#include "analyse/trajectory_transform.h"
#include "mindriver/fire.h"
#include "mindriver/cg.h"
#include "mindriver/sgd.h"
#include "deplibs/LBFGSpp/include/LBFGS.h"
#include "tools/constants.h"
#include <fmt/format.h>
#include <cmath>
#include "mindriver/strain_gradient.h"
#include <iomanip>
#include <limits>
#include <random>

using namespace std;

namespace {
std::string modecar_coordinates(const DimerDriver &driver) {
    if (!driver.vc_flag) return "fixed-cartesian";
    if(driver.subspace_active) return "vc-periodic-gram-r"+std::to_string(driver.subspace.rank())+
        (driver.recast_atomic_force?"-mixed-recast":"-mixed-fractional");
    std::string result = driver.symm_strain
        ? (driver.restore_F_cholesky ? "vc-symmetric-cholesky" : "vc-symmetric-sqrt")
        : "vc-full";
    return result + (driver.recast_atomic_force ? "-recast" : "-fractional");
}
}

// ═══════════════════════════════════════════════════════════════════════
// Construction & Settings
// ═══════════════════════════════════════════════════════════════════════

DimerDriver::DimerDriver(GPSettings settings) {
    InitSettings();
    ApplySettings(settings);
// PrintSettings();
}

void DimerDriver::InitSettings() {
    // Dimer geometry
    Connect(ddr, "ddr", "Dimer separation delta_R");
    Connect(rot_force_min, "rot_force_min", "Skip rotation if |F_rot| below this");

    Connect(random_seed, "random_seed", "Random direction seed (default 0; -1 nondeterministic)");
    // MODECAR
    Connect(modecarf, "modecarf", "Initial direction in DIMER optimization coordinates; invalid files are errors (empty setting: random)");

    // VC-DIMER
    Connect(vc_flag, "vcflag", "Variable cell DIMER");
    Connect(periodic_subspace,"periodic_subspace","Use periodic Gram cell and explicit periodic/open mixed coordinates");
    Connect(symm_strain, "symm_strain", "Use symmetric strain tensor");
    Connect(restore_F_cholesky, "restore_F_cholesky", "Cholesky decomposition for strain");
    Connect(recast_atomic_force, "recast_atomic_force", "Recast atomic forces with strain");
    Connect(pressure, "pressure", "External pressure (GPa)");
    Connect(cell_factor, "cell_factor", "Cell preconditioning factor");

    // Rotation optimizer
    Connect(rot_driver, "rot_driver", "Rotation optimizer: fire, lbfgs, cg, sgd");
    Connect(rot_max_iter, "rot_max_iter", "Max iterations per rotation sub-opt");
    Connect(rot_gtol, "rot_gtol", "Rotation force tolerance");
    Connect(rot_max_step, "rot_max_step", "Rotation max step");
    Connect(rot_first_step, "rot_first_step", "Rotation first step");

    // Translation optimizer
    Connect(trn_driver, "trn_driver", "Translation optimizer: fire, lbfgs, cg, sgd");
    Connect(trn_max_iter, "trn_max_iter", "Max iterations per translation sub-opt");
    Connect(trn_gtol, "trn_gtol", "Translation force tolerance");
    Connect(trn_max_step, "trn_max_step", "Translation max step");
    Connect(trn_first_step, "trn_first_step", "Translation first step");

    // Outer convergence
    Connect(dimer_gtol, "gtol", "Outer loop force tolerance");
    Connect(max_iter, "maxiteration", "Max outer iterations");
    Connect(min_curvature, "min_curvature", "Minimum curvature for saddle detection");

    // Output
    Connect(write_traj, "write_traj", "Write trajectory");
    Connect(trajfile, "trajfile", "Trajectory output file");
    Connect(summary, "summary", "Summary output file");
    Connect(nblock, "nblock", "Output interval");
    Connect(debug_level, "debug", "Debug level");
}

// ═══════════════════════════════════════════════════════════════════════
// Initialization
// ═══════════════════════════════════════════════════════════════════════

void DimerDriver::validate_settings() const {
    for (double value : {ddr, dimer_gtol, rot_gtol, trn_gtol, rot_max_step,
                         trn_max_step, rot_first_step, trn_first_step})
        if (!std::isfinite(value) || value <= 0)
            GPERROR("Dimer distances, steps and tolerances must be finite and positive");
    if (max_iter < 0 || rot_max_iter <= 0 || trn_max_iter <= 0 || nblock <= 0 || random_seed < -1)
        GPERROR("Invalid Dimer iteration limit, nblock or random_seed");
    if (!std::isfinite(pressure) || !std::isfinite(min_curvature) ||
        !std::isfinite(rot_force_min) || rot_force_min < 0)
        GPERROR("Invalid Dimer pressure, curvature or rotation threshold");
    for (const auto &driver : {rot_driver, trn_driver})
        if (driver != "fire" && driver != "cg" && driver != "sgd" && driver != "lbfgs")
            GPERROR("Unknown Dimer optimizer: " + driver);
}

void DimerDriver::init(Atoms &center_atom) {
    validate_settings();
    subspace_active=vc_flag && periodic_subspace && !center_atom.is_fully_periodic();
    if(subspace_active) {
        if(pressure!=0) GPERROR("Periodic-subspace Dimer forbids three-dimensional pressure");
        subspace.init(center_atom,recast_atomic_force);
    }
    const bool cell_free = !vc_flag && !center_atom.is_fully_periodic() && center_atom.cartesian &&
        center_atom.cell_.allFinite() && !center_atom.cell_.fullPivLu().isInvertible();
    if (center_atom.size() == 0 || !center_atom.cell_.allFinite() ||
        (!cell_free && !subspace_active && std::abs(center_atom.cell_.determinant()) < 1e-12))
        GPERROR("Dimer requires a nonempty structure with a finite nonsingular cell");
    for (const auto &position : center_atom.pos_)
        if (!position.allFinite()) GPERROR("Dimer requires finite positions");
    if (cell_free) (void)trajectory_periodic_coordinate_map(center_atom);
    if (trjofs.is_open()) trjofs.close();
    trjofs.clear();
    converged = false;
    iter = 0;
    energy = curvature = max_rot_force = max_trn_force = 0.0;
    max_center_gradient = 0;
    atom_center = center_atom;
    if (!atom_center.cartesian) atom_center.to_cartesian();
    natom = atom_center.size();
    npx = 3 * natom;

    // Determine cell DOF count
    if(subspace_active) offset=subspace.cell_dofs();
    else if (vc_flag) {
        if (symm_strain)
            offset = 6;
        else
            offset = 9;
    } else {
        offset = 0;
    }
    nlen = offset + npx;

    // Set up reference cell
    refcell = atom_center.cell_;
    if (cell_free || subspace_active) ivrefcell.setZero(); // Fixed-cell Cartesian coordinates never use it.
    else ivrefcell = refcell.inverse();

    // Allocate buffers
    xx_center.resize(nlen);
    modedir.resize(nlen);
    modedir_init.resize(nlen);
    g_plus.resize(nlen);
    g_minus.resize(nlen);

    // Encode center structure
    LatPos2X(atom_center, xx_center);

    // Read or generate mode direction
    if (!modecarf.empty()) {
        read_modecar(modecarf);
    } else {
        random_modedir();
    }
    modedir_init = modedir;

    // Copy chemical info to dimer endpoints
    atom_plus = atom_center;
    atom_minus = atom_center;

    // Open trajectory
    if (write_traj) {
        trjofs.open(trajfile);
        if (!trjofs) GPERROR("Cannot open DIMER trajectory: " + trajfile);
    }

    if (debug_level > 0) {
        fmt::print("DimerDriver initialized: natom={}, nlen={}, offset={}\n",
                   natom, nlen, offset);
        fmt::print("  vc_flag={}, symm_strain={}\n", vc_flag, symm_strain);
    }
}

void DimerDriver::read_modecar(const std::string &fnm) {
    ifstream ifs(fnm);
    if (!ifs.is_open()) GPERROR("Cannot open MODECAR file: " + fnm);

    // Legacy files contain only numbers, interpreted in the current driver's
    // optimization coordinates. New files identify that convention explicitly.
    ifs >> ws;
    if (ifs.peek() == 'G') {
        std::string magic, coordinates;
        int version, count;
        if (!(ifs >> magic >> version >> count >> coordinates) ||
            magic != "GPML_MODECAR" || version != 1)
            GPERROR("Invalid or unsupported MODECAR header: " + fnm);
        if (count != natom)
            GPERROR("MODECAR atom count does not match the structure: " + fnm);
        if (coordinates != modecar_coordinates(*this))
            GPERROR("MODECAR coordinate convention '" + coordinates + "' does not match '" +
                    modecar_coordinates(*this) + "'; check vcflag, symm_strain, "
                    "restore_F_cholesky and recast_atomic_force: " + fnm);
        if (vc_flag) {
            Eigen::Matrix3d reference;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    if (!(ifs >> reference(a, b)) || !std::isfinite(reference(a, b)))
                        GPERROR("Invalid MODECAR reference cell: " + fnm);
            if(subspace_active) {
                Atoms reference_atom=atom_center;reference_atom.cell_=reference;
                subspace.init(reference_atom,recast_atomic_force);refcell=reference;ivrefcell.setZero();
                LatPos2X(atom_center,xx_center);
            } else if (!reference.isApprox(refcell, 1e-10))
                GPERROR("MODECAR reference cell does not match the initial structure: " + fnm);
        }
    }

    Eigen::VectorXd direction(nlen);
    for (int i = 0; i < nlen; ++i)
        if (!(ifs >> direction(i)) || !std::isfinite(direction(i)))
            GPERROR(fmt::format("Invalid MODECAR '{}': expected {} finite components, "
                                "failed at component {}", fnm, nlen, i + 1));
    std::string extra;
    if (ifs >> extra)
        GPERROR(fmt::format("Invalid MODECAR '{}': extra data after {} components", fnm, nlen));
    if (!ifs.eof() || ifs.bad()) GPERROR("Failed to read MODECAR: " + fnm);

    double norm = direction.stableNorm();
    if (!std::isfinite(norm) || norm < 1e-12)
        GPERROR("Invalid MODECAR direction norm (must be finite and >= 1e-12): " + fnm);
    modedir = direction / norm;
    Message(fmt::format("MODECAR loaded from {}, coordinates = {}, norm = {:.6f}",
                        fnm, modecar_coordinates(*this), norm));
}

void DimerDriver::write_modecar(const std::string &fnm) const {
    const double norm = modedir.stableNorm();
    if (modedir.size() != nlen || !modedir.allFinite() || !std::isfinite(norm) || norm < 1e-12)
        GPERROR("Cannot write invalid MODECAR direction: " + fnm);
    ofstream mfs(fnm);
    if (!mfs) GPERROR("Cannot open DIMER mode output: " + fnm);
    mfs << "GPML_MODECAR 1 " << natom << ' ' << modecar_coordinates(*this) << '\n';
    mfs << std::setprecision(std::numeric_limits<double>::max_digits10);
    if (vc_flag) {
        for (int a = 0; a < 3; ++a)
            mfs << refcell(a, 0) << ' ' << refcell(a, 1) << ' ' << refcell(a, 2) << '\n';
        for (int i = 0; i < offset; ++i)
            mfs << modedir(i) << (i + 1 == offset ? '\n' : ' ');
    }
    for (int i = 0; i < natom; ++i)
        mfs << modedir(offset + 3 * i) << ' ' << modedir(offset + 3 * i + 1)
            << ' ' << modedir(offset + 3 * i + 2) << '\n';
    mfs.close();
    if (!mfs) GPERROR("Failed to write DIMER mode: " + fnm);
}

void DimerDriver::random_modedir() {
    // Generate random direction using normal distribution (isotropic)
    std::mt19937 rng(random_seed < 0 ? std::random_device{}() : static_cast<unsigned>(random_seed));
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int i = 0; i < nlen; ++i) {
        modedir(i) = dist(rng);
    }

    // For fixed-cell, zero out cell components (they don't exist in modedir anyway since offset=0)
    double norm = modedir.norm();
    if (norm > 1e-12) {
        modedir /= norm;
    } else {
        modedir.setZero();
        modedir(0) = 1.0; // fallback
    }
    Message(fmt::format("Generated random mode direction, norm = {:.6f}", modedir.norm()));
}

// ═══════════════════════════════════════════════════════════════════════
// Coordinate Transforms
// ═══════════════════════════════════════════════════════════════════════

void DimerDriver::LatPos2X(const Atoms &at, Eigen::VectorXd &x) {
    x.resize(nlen);
    x.setZero();
    if(subspace_active) {
        x.head(offset)=subspace.encode_cell(at.cell_);
        for(int i=0;i<natom;++i) x.segment<3>(offset+3*i)=subspace.encode_position(at.pos_[i],at.cell_).transpose();
        return;
    }

    if (vc_flag) {
        // Cell → strain
        if (symm_strain) {
            Eigen::Matrix3d matD = ivrefcell * at.cell_;
            Eigen::Matrix3d strain_mat = matD * matD.transpose() - Eigen::Matrix3d::Identity();
            // Upper triangle: xx, xy, xz, yy, yz, zz
            x(0) = strain_mat(0, 0);
            x(1) = strain_mat(0, 1);
            x(2) = strain_mat(0, 2);
            x(3) = strain_mat(1, 1);
            x(4) = strain_mat(1, 2);
            x(5) = strain_mat(2, 2);
        } else {
            // Full 9-component strain
            Eigen::Matrix3d matD = ivrefcell * at.cell_;
            Eigen::Matrix3d strain_mat = matD - Eigen::Matrix3d::Identity();
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    x(3 * a + b) = strain_mat(a, b);
        }
    }

    // Positions
    if (vc_flag) {
        // Fractional with recast
        for (int i = 0; i < natom; ++i) {
            Eigen::RowVector3d xfrac = at.car2dir(at.pos_[i]);
            if (recast_atomic_force)
                xfrac = xfrac * refcell;
            for (int a = 0; a < 3; ++a)
                x(offset + 3 * i + a) = xfrac(a);
        }
    } else {
        // Cartesian directly
        for (int i = 0; i < natom; ++i)
            for (int a = 0; a < 3; ++a)
                x(3 * i + a) = at.pos_[i](a);
    }
}

void DimerDriver::X2LatPos(const Eigen::VectorXd &x, Atoms &at) {
    if (x.size() != nlen || !x.allFinite()) GPERROR("Invalid Dimer coordinates");
    if(subspace_active) {
        at.cell_=subspace.decode_cell(x.head(offset));at.cartesian=true;at.pos_.resize(natom);
        for(int i=0;i<natom;++i) at.pos_[i]=subspace.decode_position(x.segment<3>(offset+3*i).transpose(),at.cell_);
        return;
    }
    if (vc_flag) {
        // Cell reconstruction from strain
        Eigen::Matrix3d eps;
        if (symm_strain) {
            eps << x(0) + 1.0, x(1),        x(2),
                   x(1),        x(3) + 1.0,  x(4),
                   x(2),        x(4),        x(5) + 1.0;

            Eigen::Matrix3d L;
            if (restore_F_cholesky) {
                Eigen::LLT<Eigen::Matrix3d> llt(eps);
                if (llt.info() == Eigen::Success) {
                    L = llt.matrixL();
                } else {
                    GPERROR("VC-Dimer strain must remain positive definite");
                }
            } else {
                Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(eps);
                if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 0)
                    GPERROR("VC-Dimer strain must remain positive definite");
                L = solver.operatorSqrt();
            }
            at.cell_ = refcell * L;
        } else {
            eps << x(0) + 1.0, x(1),        x(2),
                   x(3),        x(4) + 1.0,  x(5),
                   x(6),        x(7),        x(8) + 1.0;
            at.cell_ = refcell * eps;
        }

        if (!at.cell_.allFinite() || std::abs(at.cell_.determinant()) < 1e-12 ||
            at.cell_.determinant() * refcell.determinant() <= 0)
            GPERROR("VC-Dimer produced a singular or inverted cell");
        // Fractional positions with recast
        for (int i = 0; i < natom; ++i) {
            Eigen::RowVector3d vec;
            for (int a = 0; a < 3; ++a)
                vec(a) = x(offset + 3 * i + a);
            if (recast_atomic_force) {
                at.pos_[i] = at.dir2car(vec * ivrefcell);
            } else {
                at.pos_[i] = at.dir2car(vec);
            }
        }
    } else {
        // Cartesian positions directly
        at.pos_.resize(natom);
        for (int i = 0; i < natom; ++i)
            for (int a = 0; a < 3; ++a)
                at.pos_[i](a) = x(3 * i + a);
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Gradient Mapping: Atom forces + stress → gradient vector g
// ═══════════════════════════════════════════════════════════════════════

void DimerDriver::LatPos2G(Atoms &at, Eigen::VectorXd &g, double &ene) {
    g.resize(nlen);
    g.setZero();

    ene = at.energy_;
    if(subspace_active) {
        g.head(offset)=subspace.cell_gradient(at.cell_,at.stress_);
        const auto j=subspace.atomic_jacobian(at.cell_);
        for(int i=0;i<natom;++i) g.segment<3>(offset+3*i)=-j*at.forces_[i];
        return;
    }

    if (vc_flag) {
        // ── Cell gradient from stress ──
        double pscal = pressure * GPa_to_eVA3 * at.get_volume();
        ene += pscal;
        Eigen::Matrix3d curr_stress = at.stress_ - pscal * Eigen::Matrix3d::Identity();

        // Generalized stress: gstress = -cell^(-T) * curr_stress
        Eigen::Matrix3d gstress = -at.cell_.inverse().transpose() * curr_stress;

        if (symm_strain) {
            g.head<6>() = SaddleCoordinates::symmetric_pullback(
                ivrefcell * at.cell_, refcell.transpose() * gstress, restore_F_cholesky);
        } else {
            // Full strain gradient
            Eigen::Matrix3d c = refcell.transpose() * gstress;
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    g(3 * a + b) = c(a, b);
        }

        // ── Atomic gradient from forces ──
        if (recast_atomic_force) {
            Eigen::Matrix3d matD = ivrefcell * at.cell_;
            for (int i = 0; i < natom; ++i) {
                Eigen::RowVector3d gfrac = -matD * at.forces_[i];
                for (int a = 0; a < 3; ++a)
                    g(offset + 3 * i + a) = gfrac(a);
            }
        } else {
            for (int i = 0; i < natom; ++i) {
                Eigen::RowVector3d gfrac = -at.cell_ * at.forces_[i];
                for (int a = 0; a < 3; ++a)
                    g(offset + 3 * i + a) = gfrac(a);
            }
        }
    } else {
        // Fixed cell: g = -force (negative of force = gradient)
        for (int i = 0; i < natom; ++i) {
            for (int a = 0; a < 3; ++a) {
                g(3 * i + a) = -at.forces_[i](a);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════
// Dimer Endpoint Management
// ═══════════════════════════════════════════════════════════════════════

void DimerDriver::build_dimer_endpoints() {
    X2LatPos(xx_center + ddr * modedir, atom_plus);
    X2LatPos(xx_center - ddr * modedir, atom_minus);
}

void DimerDriver::eval_endpoints() {
    pot->getEFSOMP(atom_plus);
    pot->getEFSOMP(atom_minus);
}

// ═══════════════════════════════════════════════════════════════════════
// Force & Curvature Computation
// ═══════════════════════════════════════════════════════════════════════

double DimerDriver::compute_curvature() {
    // C = (F₂ - F₁)·n̂ / (2*ddr) = (g_plus - g_minus)·n̂ / (2*ddr)
    // Negative curvature indicates a saddle point along this mode
    return (g_plus - g_minus).dot(modedir) / (2.0 * ddr);
}

void DimerDriver::compute_rotational_force(Eigen::VectorXd &g_out) {
    // Perpendicular component of g_diff → rotates dimer toward lowest curvature mode
    g_out = (g_plus - g_minus) / ddr;
    double proj = g_out.dot(modedir);
    g_out -= proj * modedir;
    max_rot_force = g_out.lpNorm<Eigen::Infinity>();
}

void DimerDriver::compute_translational_force(Eigen::VectorXd &g_out) {
    // Reverse parallel component → climb toward saddle along mode
    g_out = 0.5 * (g_plus + g_minus);
    double proj = g_out.dot(modedir);
    g_out -= 2.0 * proj * modedir;
    max_trn_force = g_out.lpNorm<Eigen::Infinity>();
}

// ═══════════════════════════════════════════════════════════════════════
// Functor Interfaces for Optimizers
// ═══════════════════════════════════════════════════════════════════════

double DimerDriver::operator_rot(const Eigen::VectorXd &mode_in, Eigen::VectorXd &g_out) {
    // Copy and normalize mode direction
    modedir = mode_in;
    double norm = modedir.norm();
    if (std::isfinite(norm) && norm > 1e-12) {
        modedir /= norm;
    } else {
        GPERROR("Dimer rotation produced an invalid direction");
    }

    // Build dimer endpoints with current mode
    build_dimer_endpoints();

    // Evaluate forces
    eval_endpoints();

    // Map forces to gradients
    double e_plus, e_minus;
    LatPos2G(atom_plus, g_plus, e_plus);
    LatPos2G(atom_minus, g_minus, e_minus);

    // Compute curvature (minimize this)
    curvature = compute_curvature();

    // Compute rotational gradient (perpendicular to n̂)
    compute_rotational_force(g_out);
    g_out /= norm; // mode_in is normalized above: apply the normalization Jacobian.

    // Return curvature as "energy" (optimizer minimizes it)
    return curvature;
}

double DimerDriver::operator_trn(const Eigen::VectorXd &x_in, Eigen::VectorXd &g_out) {
    // Copy center position
    xx_center = x_in;

    // Build dimer endpoints
    build_dimer_endpoints();

    // Evaluate forces
    eval_endpoints();

    // Map forces to gradients
    double e_plus, e_minus;
    LatPos2G(atom_plus, g_plus, e_plus);
    LatPos2G(atom_minus, g_minus, e_minus);

    // Compute effective translational gradient
    compute_translational_force(g_out);

    // Compute curvature for monitoring
    curvature = compute_curvature();

    // Energy is the average of the two endpoints
    energy = 0.5 * (e_plus + e_minus);

    // Return 0 for force-only optimizers (FIRE/SGD), or energy for LBFGS/CG
    return energy;
}

// ═══════════════════════════════════════════════════════════════════════
// Convergence Check
// ═══════════════════════════════════════════════════════════════════════

bool DimerDriver::check_convergence() {
    return max_trn_force < dimer_gtol && max_center_gradient < dimer_gtol &&
           max_rot_force <= std::max(rot_force_min, rot_gtol) && curvature < min_curvature;
}

// ═══════════════════════════════════════════════════════════════════════
// Optimizer Helper: run a sub-optimizer instance
// ═══════════════════════════════════════════════════════════════════════

namespace {

// Helper to build GPSettings from a flat string map
GPSettings make_opt_settings(const std::string &driver_type,
                              int max_iter, double gtol,
                              double max_step, double first_step) {
    GPSettings s;
    s.Modify("driver", driver_type);
    s.Modify("maxiteration", to_string(max_iter));
    s.Modify("gtol", fmt::format("{:.17g}", gtol));
    s.Modify("max_move", fmt::format("{:.17g}", max_step));
    s.Modify("first_step", fmt::format("{:.17g}", first_step));
    // FIRE applies first_step only during first_delay; cap the first sub-step.
    if (driver_type == "fire") s.Modify("first_delay", "1");
    return s;
}

// Wrapper functor for rotation sub-optimizer
struct RotationFunctor {
    DimerDriver &dimer;
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        return dimer.operator_rot(x, g);
    }
};

// Wrapper functor for translation sub-optimizer
struct TranslationFunctor {
    DimerDriver &dimer;
    double operator()(const Eigen::VectorXd &x, Eigen::VectorXd &g) {
        return dimer.operator_trn(x, g);
    }
};

// Run a sub-optimizer on a functor. Returns the number of iterations used.
template <typename Functor>
int run_sub_optimizer(const std::string &driver_type,
                      int max_iter, double gtol,
                      const GPSettings &opt_settings,
                      Functor &functor,
                      Eigen::VectorXd &x, double &f) {
    if (driver_type == "fire") {
        SimpleFireF opt(opt_settings);
        return opt.minimize(functor, x, f);
    } else if (driver_type == "cg") {
        SimpleCGF opt(opt_settings);
        return opt.minimize(functor, x, f);
    } else if (driver_type == "sgd") {
        SimpleSGD opt(opt_settings);
        return opt.minimize(functor, x, f);
    } else if (driver_type == "lbfgs") {
        LBFGSpp::LBFGSParam<double> lbfgs_param;
        lbfgs_param.max_iterations = max_iter;
        lbfgs_param.linesearch = LBFGSpp::LBFGS_LINESEARCH_FORCE_ONLY_A;
        lbfgs_param.epsgmax = gtol;
        lbfgs_param.epsilon = 0;
        lbfgs_param.epsilon_rel = 0;
        LBFGSpp::LBFGSSolver<double, LBFGSpp::LineSearchForceOnlyA> solver(lbfgs_param);
        return solver.minimize(functor, x, f);
    } else {
        Message("Warning: unknown driver '" + driver_type + "', using FIRE");
        SimpleFireF opt(opt_settings);
        return opt.minimize(functor, x, f);
    }
}

} // anonymous namespace

// ═══════════════════════════════════════════════════════════════════════
// Main Algorithm
// ═══════════════════════════════════════════════════════════════════════

void DimerDriver::refresh_state() {
    build_dimer_endpoints();
    eval_endpoints();
    double ep, em;
    LatPos2G(atom_plus, g_plus, ep);
    LatPos2G(atom_minus, g_minus, em);
    Eigen::VectorXd g;
    compute_translational_force(g);
    compute_rotational_force(g);
    curvature = compute_curvature();
    X2LatPos(xx_center, atom_center);
    pot->getEFSOMP(atom_center);
    LatPos2G(atom_center, g, energy);
    max_center_gradient = g.lpNorm<Eigen::Infinity>();
    if (!g_plus.allFinite() || !g_minus.allFinite() || !g.allFinite() ||
        !std::isfinite(energy) || !std::isfinite(curvature))
        GPERROR("Dimer produced non-finite energy or gradients");
}

int DimerDriver::run() {
    converged = false;
    validate_settings();
    if (!pot) GPERROR("No potential assigned to DimerDriver");
    if (natom == 0 || xx_center.size() != nlen) GPERROR("Initialize Dimer before running");

    // ── Open summary file ──
    ofstream sumfs;
    if (!summary.empty()) {
        sumfs.open(summary);
        if (!sumfs) GPERROR("Cannot open DIMER summary: " + summary);
        if (sumfs.is_open()) {
            sumfs << fmt::format("# {:>4s} {:>14s} {:>14s} {:>12s} {:>12s} {:>12s}\n",
                                 "iter", "energy", "curvature", "|F_rot|", "|F_trn|", "volume");
        }
    }

    // ── Main iteration loop ──
    for (iter = 0; iter < max_iter; ++iter) {
        refresh_state();
        if (check_convergence()) { converged = true; break; }

        // Step 5: Rotation sub-step
        {
            Eigen::VectorXd tmp(nlen);
            compute_rotational_force(tmp);
        }
        if (max_rot_force > rot_force_min) {
            GPSettings rot_settings = make_opt_settings(
                rot_driver, rot_max_iter, rot_gtol, rot_max_step, rot_first_step);

            double f_rot = 0.0;
            Eigen::VectorXd mode_opt = modedir;
            RotationFunctor rot_functor{*this};
            int nrot = run_sub_optimizer(rot_driver, rot_max_iter, rot_gtol,
                                         rot_settings, rot_functor, mode_opt, f_rot);

            Eigen::VectorXd accepted_gradient;
            operator_rot(mode_opt, accepted_gradient);
            if (debug_level > 1) {
                fmt::print("  Rotation: {} steps, final curvature = {:.6f}\n", nrot, curvature);
            }
        }

        // Step 6: Translation sub-step
        {
            GPSettings trn_settings = make_opt_settings(
                trn_driver, trn_max_iter, trn_gtol, trn_max_step, trn_first_step);

            double f_trn = 0.0;
            TranslationFunctor trn_functor{*this};
            Eigen::VectorXd position_opt = xx_center;
            int ntrn = run_sub_optimizer(trn_driver, trn_max_iter, trn_gtol,
                                         trn_settings, trn_functor, position_opt, f_trn);
            xx_center = position_opt;

            if (debug_level > 1) {
                fmt::print("  Translation: {} steps, energy = {:.6f}\n", ntrn, energy);
            }
        }

        refresh_state();
        // Step 7: Output
        if (iter % nblock == 0 || (debug_level > 0)) {
            print_status();
        }

        if (sumfs.is_open()) {
            double vol = atom_center.get_volume();
            sumfs << fmt::format("  {:>4d} {:>14.6f} {:>14.6f} {:>12.6f} {:>12.6f} {:>12.4f}\n",
                                 iter, energy, curvature, max_rot_force, max_trn_force, vol);
        }

        if (write_traj && trjofs.is_open()) {
            X2LatPos(xx_center, atom_center);
            if (iobj) {
                pot->getEFS(atom_center);
                if (!iobj->write_one(atom_center, trjofs, "extxyz"))
                    GPERROR("Failed to write DIMER trajectory: " + trajfile);
            }
        }
    }

    // Include the final allowed translation and zero-iteration runs.
    refresh_state();
    converged = check_convergence();
    if (write_traj && trjofs.is_open() && iobj && iter == 0)
        if (!iobj->write_one(atom_center, trjofs, "extxyz"))
            GPERROR("Failed to write DIMER trajectory: " + trajfile);

    // ── Finalization ──
    if (sumfs.is_open()) {
        sumfs.close();
        if (!sumfs) GPERROR("Failed to write DIMER summary: " + summary);
    }

    // Write final center structure
    if (iobj) {
        if (!iobj->write_one(atom_center, "dimer_final.vasp"))
            GPERROR("Failed to write DIMER final structure: dimer_final.vasp");
    }

    // Write final mode direction
    write_modecar("dimer_modecar_final");

    // Write dimer endpoint structures
    if (iobj) {
        build_dimer_endpoints();
        if (!iobj->write_one(atom_plus, "dimer_plus_final.vasp"))
            GPERROR("Failed to write DIMER endpoint: dimer_plus_final.vasp");
        if (!iobj->write_one(atom_minus, "dimer_minus_final.vasp"))
            GPERROR("Failed to write DIMER endpoint: dimer_minus_final.vasp");
    }

    if (write_traj && trjofs.is_open()) {
        trjofs.close();
        if (!trjofs) GPERROR("Failed to write DIMER trajectory: " + trajfile);
    }

    if (!summary.empty()) write_summary(summary);
    return converged ? 0 : 1;
}

// ═══════════════════════════════════════════════════════════════════════
// Output Helpers
// ═══════════════════════════════════════════════════════════════════════

void DimerDriver::print_status() {
    double vol = atom_center.get_volume();
    fmt::print("DIMER iter {:>4d}: energy = {:>12.6f}, curvature = {:>10.6f}, "
               "|F_rot|max = {:>8.6f}, |F_trn|max = {:>8.6f}, vol = {:>10.4f}\n",
               iter, energy, curvature, max_rot_force, max_trn_force, vol);
}

void DimerDriver::write_summary(const std::string &fnm) {
    ofstream ofs(fnm);
    if (!ofs) GPERROR("Cannot open DIMER summary: " + fnm);
    double vol = atom_center.get_volume();
    ofs << fmt::format("# DIMER summary\n");
    ofs << fmt::format("# converged: {}\n", converged ? "yes" : "no");
    ofs << fmt::format("# status: {}\n", converged ? "converged" : "max_iterations");
    ofs << fmt::format("# iterations: {}\n", iter);
    ofs << fmt::format("# random_seed: {}\n", random_seed);
    ofs << fmt::format("# energy_kind: {}\n", vc_flag ? "enthalpy" : "potential_energy");
    ofs << fmt::format("# final |gradient_center|max: {:.12g}\n", max_center_gradient);
    ofs << fmt::format("# final |gradient_rotation|max: {:.12g}\n", max_rot_force);
    ofs << fmt::format("# final energy: {:.8f}\n", energy);
    ofs << fmt::format("# final curvature: {:.8f}\n", curvature);
    ofs << fmt::format("# final |F_trn|max: {:.8f}\n", max_trn_force);
    ofs << fmt::format("# final volume: {:.4f}\n", vol);
    ofs.close();
    if (!ofs) GPERROR("Failed to write DIMER summary: " + fnm);
}
