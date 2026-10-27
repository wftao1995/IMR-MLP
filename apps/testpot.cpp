// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "potentials/potential_wrapper.h"
#include "tools/constants.h"

using namespace std;
using namespace Eigen;

// ═══════════════════════════════════════════════════════════════════
// Numerical force test via atomic position finite difference.
// Uses central difference: F_a = -(E(x+dx) - E(x-dx)) / (2*dx)
// Restores atom position after each component to avoid accumulated
// drift across atoms in the all-atoms test loop.
// ═══════════════════════════════════════════════════════════════════
void test_force(Atoms &atom, Potentials::AnyPotential *pot,
                std::vector<Eigen::Vector3d> &frefs, double e0,
                int test_which, double nugedvalue) {
    int iat = test_which;
    auto fref = frefs[iat];
    fmt::print("\n--- Atom {} ---\n", iat);
    fmt::print("{:<4s} {:>16s} {:>16s} {:>16s} {:>14s}\n",
               "Comp", "F_analytical", "F_numerical", "Diff", "RelDiff");
    for (int a = 0; a < 3; ++a) {
        atom.pos_[iat][a] += nugedvalue;
        pot->getE(atom);
        double ep = atom.energy_;
        atom.pos_[iat][a] -= 2 * nugedvalue;
        pot->getE(atom);
        double em = atom.energy_;
        atom.pos_[iat][a] += nugedvalue;  // RESTORE position
        double f1 = (e0 - em) / nugedvalue;
        double f2 = (ep - e0) / nugedvalue;
        double ff = -(f1 + f2) / 2.0;  // F = -dE/dx, central difference
        double diff = std::abs(fref[a] - ff);
        double reldiff = (std::abs(fref[a]) > 1e-12) ? diff / std::abs(fref[a]) : diff;
        char comp = 'x' + a;
        fmt::print("{:<4c} {:16.10f} {:16.10f} {:16.10f} {:14.6e}\n",
                   comp, fref[a], ff, diff, reldiff);
    }
}

// ═══════════════════════════════════════════════════════════════════
// Numerical stress tensor test via cell matrix finite difference.
//
// Cell vectors are rows: positions r_i = s_i * h (fractional coords).
// the energy derivative w.r.t. h is:
//   dE/dh_ab = -Sigma_d S_bd * (h^{-1})_{da}
// where S = atom.stress_ (virial stress, energy units).
//
// FD: perturb h(a,b) by +/- stress_nudge, recompute energy with
//     positions rescaled via set_cell (preserves fractional coords).
//     dE/dh_ab ≈ (E(h+δ) - E(h-δ)) / (2*δ)
// ═══════════════════════════════════════════════════════════════════
void test_stress(Atoms &atom, Potentials::AnyPotential *pot,
                 const Eigen::Matrix3d &analytical_stress,
                 double e0, double stress_nudge) {
    Eigen::Matrix3d cell0 = atom.cell_;
    double vol = std::abs(cell0.determinant());
    Eigen::Matrix3d inv_cell = cell0.inverse();
    const char *comp_names[9] = {"xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"};

    // ── Symmetry check ──
    fmt::print("Stress Symmetry Check:\n");
    double max_asym = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int b = a + 1; b < 3; ++b) {
            double asym = std::abs(analytical_stress(a, b) - analytical_stress(b, a));
            max_asym = std::max(max_asym, asym);
            fmt::print("  S_{}{} = {:14.10f},  S_{}{} = {:14.10f},  |diff| = {:.2e}\n",
                       char('x' + a), char('x' + b), analytical_stress(a, b),
                       char('x' + b), char('x' + a), analytical_stress(b, a), asym);
        }
    }
    fmt::print("  Max asymmetry = {:.2e}  (should be ~0 for symmetric tensor)\n\n", max_asym);

    // ── Print analytical stress (Voigt form for convenience) ──
    fmt::print("Analytical stress tensor (eV, energy units):\n");
    fmt::print("  xx={:14.10f}  yy={:14.10f}  zz={:14.10f}\n",
               analytical_stress(0, 0), analytical_stress(1, 1), analytical_stress(2, 2));
    fmt::print("  yz={:14.10f}  xz={:14.10f}  xy={:14.10f}\n",
               analytical_stress(1, 2), analytical_stress(0, 2), analytical_stress(0, 1));
    fmt::print("  Volume = {:.6f} A^3\n", vol);
    if (vol > 1e-12) {
        // stress_ is the virial W: positive-compressive p = tr(W)/(3V).
        fmt::print("  Pressure = {:.6f} GPa (trace/3)\n",
                   (analytical_stress.trace() / 3.0) / vol * eVA3_to_GPa);
    }
    fmt::print("\n");

    // ── Compute FD dE/dh for all 9 components once ──
    Eigen::Matrix3d dE_dh_fd;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            Eigen::Matrix3d cell_p = cell0;
            cell_p(a, b) += stress_nudge;
            atom.set_cell(cell_p);
            pot->getE(atom);
            double ep = atom.energy_;

            Eigen::Matrix3d cell_m = cell0;
            cell_m(a, b) -= stress_nudge;
            atom.set_cell(cell_m);
            pot->getE(atom);
            double em = atom.energy_;

            atom.set_cell(cell0);
            dE_dh_fd(a, b) = (ep - em) / (2.0 * stress_nudge);
        }
    }

    // ── dE/dh comparison table ──
    fmt::print("Stress dE/dh Comparison (nudge = {:.2e} A):\n", stress_nudge);
    fmt::print("{:<5s} {:>16s} {:>16s} {:>16s} {:>14s}\n",
               "Comp", "dE/dh (FD)", "dE/dh (ana)", "Diff", "RelDiff");

    double max_abs_diff = 0.0, max_rel_diff = 0.0;
    for (int a = 0; a < 3; ++a) {
        for (int b = 0; b < 3; ++b) {
            // r = s*h, so dE/dh_ab = -sum_i F_ib*s_ia.
            double dE_dh_ana = 0.0;
            for (int d = 0; d < 3; ++d)
                dE_dh_ana -= analytical_stress(b, d) * inv_cell(d, a);

            double diff = std::abs(dE_dh_fd(a, b) - dE_dh_ana);
            double reldiff = (std::abs(dE_dh_ana) > 1e-12)
                                 ? diff / std::abs(dE_dh_ana)
                                 : diff;
            max_abs_diff = std::max(max_abs_diff, diff);
            max_rel_diff = std::max(max_rel_diff, reldiff);

            fmt::print("{:<5s} {:16.10f} {:16.10f} {:16.10f} {:14.6e}\n",
                       comp_names[a * 3 + b], dE_dh_fd(a, b), dE_dh_ana, diff, reldiff);
        }
    }
    fmt::print("  Max |diff| = {:.6e},  Max |rel diff| = {:.6e}\n", max_abs_diff, max_rel_diff);

    // ── For orthogonal / near-orthogonal cell, recover S directly from FD ──
    bool nearly_orth = true;
    for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b)
            if (a != b && std::abs(cell0(a, b)) > 1e-8)
                nearly_orth = false;

    if (nearly_orth) {
        fmt::print("\nCell is (nearly) orthogonal → direct stress recovery:\n");
        fmt::print("{:<5s} {:>16s} {:>16s} {:>16s} {:>14s}\n",
                   "Comp", "S_FD (eV)", "S_ana (eV)", "Diff", "RelDiff");
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                // For a diagonal cell: S_ab = -h_bb * dE/dh_ba.
                double S_fd = -cell0(b, b) * dE_dh_fd(b, a);
                double S_ana = analytical_stress(a, b);
                double diff = std::abs(S_fd - S_ana);
                double reldiff = (std::abs(S_ana) > 1e-12)
                                     ? diff / std::abs(S_ana)
                                     : diff;
                fmt::print("{:<5s} {:16.10f} {:16.10f} {:16.10f} {:14.6e}\n",
                           comp_names[a * 3 + b], S_fd, S_ana, diff, reldiff);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Main: test potential forces (and optionally stress) by finite
// difference numerical differentiation against analytical values.
// ═══════════════════════════════════════════════════════════════════
int testpot(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    string settingfile = "settings.ini";
    string fname = "POSCAR";
    string format = "POSCAR";
    string indexfile = "index.map";
    int test_which = -1;
    double nugedvalue = 0.001;
    int test_stress_flag = 0;
    double stress_nudge = 1e-6;

    cmdopts.addop(settingfile, "settings", "", "input setting file", true)
           .addop(fname, "input")
           .addop(format, "format")
           .addop(indexfile, "index")
           .addop(test_which, "test-which")
           .addop(nugedvalue, "nugedvalue")
           .addop(test_stress_flag, "test-stress", "", "also test stress tensor by cell FD")
           .addop(stress_nudge, "stress-nudge", "", "FD displacement for stress test (A)");
    cmdopts.phrase_options(argc, argv);

    GPSettings settings;
    settings.Load(settingfile);
    apply_settings_to_cmdopts(cmdopts, settings);

    iobj.readmap(indexfile);
    Atoms atom;
    if (!iobj.read_one(atom, fname, format)) {
        std::cerr << "Error: failed to read input structure from " << fname << std::endl;
        return 1;
    }
    Potentials::potential_wrapper wrapper(settings.ExtractSubSettings("pot"));

    // ── Compute analytical EFS once ──
    wrapper.pot->getEFS(atom);
    double e0 = atom.energy_;
    std::vector<Eigen::Vector3d> fce = atom.forces_;
    Eigen::Matrix3d analytical_stress = atom.stress_;

    fmt::print("System: {} atoms,  energy = {:.10f} eV\n", atom.size(), e0);

    // ── Test forces (FD by atomic position perturbation) ──
    fmt::print("\n=== Force Numerical Test (nudge = {} A) ===\n", nugedvalue);
    if (test_which >= 0)
        test_force(atom, wrapper.pot, fce, e0, test_which, nugedvalue);
    else {
        for (int i = 0; i < atom.size(); ++i)
            test_force(atom, wrapper.pot, fce, e0, i, nugedvalue);
    }

    // ── Test stress (FD by cell perturbation) ──
    if (test_stress_flag) {
        fmt::print("\n=== Stress Numerical Test (nudge = {:.2e} A) ===\n", stress_nudge);
        test_stress(atom, wrapper.pot, analytical_stress, e0, stress_nudge);
    }

    return 0;
}
