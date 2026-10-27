// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_wrapper.h"
#include "io/gpio.h"
#include "mindriver/relax_new.h"
#include <libalglib/stdafx.h>
#include <libalglib/interpolation.h>
#include "tools/constants.h"
#include <algorithm>
#include <cmath>
#include <set>

using namespace std;

int calc_eos(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    GPSettings settings;
    Atoms atom;
    std::vector<Atoms> atoms;
    string informat;
    string infile = "POSCAR";
    string outfile;
    string eosfit = "eos.dat";
    string indexfile = "index.map";
    string settingfile = "input.ini";
    string trjfile;
    string trjformat = "mlpcfg";
    string vephfile = "veph.dat";
    int ninterp = 100;

    string curve_type = "ev";
    bool fit_eos_only = false;
    double frac_ub = 0.05;
    double frac_lb = -0.05;
    double press_ub = 10;
    double press_lb = -10;
    int npositive = 5;
    int nnegitave = 5;
    bool skip_pre_relax = false;

    cmdopts.addop(indexfile, "index", "id", "Index file mapping atom types to indices")
        .addop(infile, "input", "in", "Input structure file")
        .addop(informat, "input-format", "", "Input structure file format")
        .addop(trjformat, "traj-format", "", "Output trajectory format")
        .addop(npositive, "npos", "", "Number of expansion points (> reference)")
        .addop(nnegitave, "nneg", "", "Number of compression points (< reference)")
        .addop(skip_pre_relax, "skip-prelax", "", "Skip preliminary structure relaxation", false, 0)
        .addop(outfile, "output", "o", "EOS curve data output file")
        .addop(settingfile, "settings", "", "Main configuration settings file", true)
        // Mode selection
        .addop(curve_type, "curve", "c", "Curve type: ev (energy-volume) or pv (pressure-volume)", false, 1)
        // EV-specific
        .addop(frac_ub, "upper", "ub", "[EV] Upper bound of volume expansion fraction (e.g. 0.05 = +5%)")
        .addop(frac_lb, "lower", "lb", "[EV] Lower bound of volume compression fraction (e.g. -0.05 = -5%)")
        .addop(ninterp, "ninterp", "ninterp", "[EV] Number of interpolation points for EOS curve")
        .addop(vephfile, "out-veph", "", "[EV] Volume-energy-pressure-enthalpy interpolated data file")
        .addop(fit_eos_only, "fit-only", "fit", "[EV] Fit EOS using existing data (skip calculation)", false, 0)
        .addop(iobj.lossless, "lossless", "", "[EV] Enable lossless output mode", false, 0, 10)
        .addop(eosfit, "output-eos", "", "[EV] EOS fitting parameters output file (Birch-Murnaghan only)")
        // PV-specific
        .addop(press_ub, "press-ub", "pub", "[PV] Upper pressure bound in GPa")
        .addop(press_lb, "press-lb", "plb", "[PV] Lower pressure bound in GPa")
        .attach_settings("pot:", "Potential settings",
            [](std::ostream &os) { Potentials::potential_wrapper().ListSettings(os, "pot:"); })
        .attach_settings("relax:", "Relaxation driver",
            [](std::ostream &os) { RelaxationDrvV2().ListSettings(os, "relax:"); })
        .set_helponexit(false);

    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        exit(0);
    }

    if (GPIO::file_exist(settingfile)) {
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);

    // Both command-line and settings-file curve selections determine defaults.
    if (outfile.empty()) {
        outfile = (curve_type == "pv") ? "p-e.dat" : "e-v.dat";
    }
    if (trjfile.empty()) {
        trjfile = (curve_type == "pv") ? "p-e.cfg" : "e-v.cfg";
    }

    if (GPIO::file_exist(indexfile)) {
        iobj.readmap(indexfile);
    } else {
        cerr << "ERROR: indexfile not found\n";
    }

    auto check_relaxation = [](const RelaxationDrvV2 &driver, const string &stage) {
        if (driver.state_vars.converged) return true;
        cerr << "EOS relaxation did not converge at " << stage << ": "
             << driver.stop_reason << endl;
        return false;
    };

    // ===== EV mode =====
    if (curve_type == "ev") {
        if (ninterp < 2)
            GPERROR("EOS --ninterp must be at least 2");
        alglib::real_1d_array fit_x, fit_y;
        if (!fit_eos_only) {
            if (!iobj.read(atoms, infile, informat) || atoms.empty() || atoms.back().size() == 0) {
                std::cerr << "Error: no structures loaded from " << infile << '\n';
                return 1;
            }
            atom = atoms.back();
            Potentials::potential_wrapper wrapper("", "", settings.ExtractSubSettings("pot"));
            RelaxationDrvV2 optdrv(settings.ExtractSubSettings("relax"));

            if (!skip_pre_relax) {
                optdrv.init(wrapper.pot, atom);
                optdrv.run();
                if (!check_relaxation(optdrv, "EV preliminary")) return 2;
                atom = optdrv.atom;
            } else {
                wrapper.pot->getEFS(atom);
            }

            std::vector<double> scale_positive(npositive);
            std::vector<double> scale_negative(nnegitave);
            optdrv.fixvol();
            atoms.clear();

            double vol0 = atom.get_volume();
            for (int i = 1; i <= npositive; ++i) {
                scale_positive[i-1] = vol0 * (1 + frac_ub * i / npositive);
            }
            for (int i = 1; i <= nnegitave; ++i) {
                scale_negative[i-1] = vol0 * (1 + frac_lb * i / nnegitave);
            }
            std::vector<double> results(npositive + 1 + nnegitave);
            atoms.resize(nnegitave + npositive + 1);
            atoms[nnegitave] = atom;
            results[nnegitave] = atom.energy_;
            Atoms at = atom;
            for (int i = 0; i < npositive; ++i) {
                at.scale_to_volume(scale_positive[i]);
                optdrv.init(wrapper.pot, at);
                optdrv.run();
                if (!check_relaxation(optdrv, fmt::format("EV expansion volume {}", scale_positive[i]))) return 2;
                at = optdrv.atom;
                atoms[nnegitave + 1 + i] = at;
            }
            at = atom;
            for (int i = 0; i < nnegitave; ++i) {
                at.scale_to_volume(scale_negative[i]);
                optdrv.init(wrapper.pot, at);
                optdrv.run();
                if (!check_relaxation(optdrv, fmt::format("EV compression volume {}", scale_negative[i]))) return 2;
                at = optdrv.atom;
                atoms[nnegitave - 1 - i] = at;
            }

            optdrv.init(wrapper.pot, atoms[nnegitave]);
            optdrv.run();
            if (!check_relaxation(optdrv, "EV reference")) return 2;
            atoms[nnegitave] = optdrv.atom;

            if (!iobj.write(atoms, trjfile, trjformat))
                GPERROR("Failed to write EOS trajectory: " + trjfile);

            ofstream ofs(outfile);
            if (!ofs) GPERROR("Cannot open EOS curve output: " + outfile);
            double eref = atom.energy_;
            fit_x.setlength(atoms.size());
            fit_y.setlength(atoms.size());
            ofs << "# Volume/at    Energy/at   Press   Enth/at\n";
            for (int i = 0; i < atoms.size(); ++i) {
                double vv = atoms[i].get_volume() / atoms[i].size();
                double ee = atoms[i].energy_ / atoms[i].size();
                double pp = atoms[i].stress_.diagonal().mean() / atoms[i].get_volume();
                double hh = ee + pp * vv;
                ofs << fmt::format("{:10.6f} {:10.6f} {:10.6f} {:10.6f}\n", vv, ee, pp * eVA3_to_GPa, hh);
                fit_x[i] = pow(vv, -2./3.);
                fit_y[i] = ee;
            }
            ofs.close();
            if (!ofs) GPERROR("Failed to write EOS curve: " + outfile);
        } else {
            ifstream ifs(outfile);
            if (!ifs) GPERROR("Cannot open EOS curve input: " + outfile);
            string buffer;
            vector<string> vbuf;
            vector<double> fx, fy;
            std::set<double> volumes;
            size_t line = 0;
            while (getline(ifs, buffer)) {
                ++line;
                trim_comment(buffer, "#!");
                trim_all(buffer);
                if (buffer.empty()) continue;
                str_split(buffer, vbuf);
                if (vbuf.size() < 2)
                    GPERROR("EOS curve requires volume and energy at line " + to_string(line));
                size_t used_v = 0, used_e = 0;
                const double volume = stod(vbuf[0], &used_v);
                const double energy = stod(vbuf[1], &used_e);
                if (used_v != vbuf[0].size() || used_e != vbuf[1].size() ||
                    !std::isfinite(volume) || volume <= 0.0 || !std::isfinite(energy))
                    GPERROR("Invalid EOS volume or energy at line " + to_string(line));
                volumes.insert(volume);
                fx.push_back(pow(volume, -2./3.));
                fy.push_back(energy);
            }
            if (!ifs.eof()) GPERROR("Failed to read EOS curve: " + outfile);
            if (volumes.size() < 4) {
                std::cerr << "EOS cubic fit requires at least four distinct positive volumes\n";
                return 1;
            }
            fit_x.setlength(fx.size());
            fit_y.setlength(fy.size());
            fit_x.setcontent(fx.size(), fx.data());
            fit_y.setcontent(fy.size(), fy.data());
        }

        // Fit Birch-Murnaghan isothermal equation of state
        alglib::barycentricinterpolant p;
        alglib::polynomialfitreport rep;
        try {
            alglib::polynomialfit(fit_x, fit_y, 4, p, rep);
        } catch(const alglib::ap_error& e) {
            std::cerr << "EOS fit failed: " << e.msg << std::endl;
            return 1;
        }

        alglib::real_1d_array coeff;
        alglib::polynomialbar2pow(p, coeff);
        double rmse = rep.rmserror;
        double avge = rep.avgerror;
        double maxe = rep.maxerror;
        double a0 = coeff[0];
        double a1 = coeff[1];
        double a2 = coeff[2];
        double a3 = coeff[3];
        const double discriminant = a2*a2 - 3*a1*a3;
        if (!std::isfinite(a0) || !std::isfinite(a1) || !std::isfinite(a2) || !std::isfinite(a3) ||
            !std::isfinite(discriminant) || discriminant <= 0.0)
            GPERROR("EOS fit has no stable finite equilibrium point");
        const double curvature = sqrt(discriminant); // half of d2E/dx2 at the minimum
        // Rationalize the positive-curvature root to avoid cancellation. This
        // also covers a quadratic fit (a3 == 0, a2 > 0).
        if (a2 < 0.0 && a3 == 0.0)
            GPERROR("EOS fit has no stable finite equilibrium point");
        double x0 = a2 >= 0.0 ? -a1 / (a2 + curvature) : (-a2 + curvature) / (3*a3);
        if (!std::isfinite(x0) || x0 <= 0.0)
            GPERROR("EOS fit has no positive finite equilibrium volume");
        double V0 = pow(x0, -1.5);
        double B0p = 4.0 + 2.0 * a3 * x0 / curvature;
        double B0 = 8.0 / 9.0 * curvature * pow(x0, 3.5);
        double E0 = a0 + x0 * (a1 + x0 * (a2 + x0 * a3));
        if (!std::isfinite(V0) || V0 <= 0.0 || !std::isfinite(B0p) ||
            !std::isfinite(B0) || B0 <= 0.0 || !std::isfinite(E0))
            GPERROR("EOS fit produced invalid equilibrium parameters");

        ofstream ofseos(eosfit);
        if (!ofseos) GPERROR("Cannot open EOS fit output: " + eosfit);
        ofseos << "Birch-Murnaghan isothermal equation of state\n";
        ofseos << fmt::format("E  = {:10.6f} + {:10.6f} x + {:10.6f} x^2 + {:10.6f} x^3, where x = V^(-2/3)\n",
                              a0, a1, a2, a3);
        ofseos << fmt::format("Summary: RMSE = {:.5g} ; AVGE = {:.5g} ; MAXE = {:.5g}\n", rmse, avge, maxe);
        ofseos << "Fitted parameters: \n";
        ofseos << fmt::format("V0 = {:10.6f}\n", V0);
        ofseos << fmt::format("B0'= {:10.6f}\n", B0p);
        ofseos << fmt::format("B0 = {:10.6f}\n", B0);
        ofseos << fmt::format("E0 = {:10.6f}\n", E0);
        ofseos.close();
        if (!ofseos) GPERROR("Failed to write EOS fit: " + eosfit);

        // Output interpolated V-E-P-H
        {
            ofstream ofs(vephfile);
            if (!ofs) GPERROR("Cannot open EOS interpolation output: " + vephfile);
            ofs << "# V(A3) E(eV) P(GPa) H(eV) All per-atom\n";
            const auto [min_x, max_x] = std::minmax_element(
                fit_x.getcontent(), fit_x.getcontent() + fit_x.length());
            double vstart = pow(*max_x, -1.5);
            double vend = pow(*min_x, -1.5);
            double dv = (vend - vstart) / (ninterp - 1.0);
            for (int iv = 0; iv < ninterp; ++iv) {
                double v = vstart + iv * dv;
                double x = pow(v, -2./3.);
                double e = a0 + a1 * x + a2 * x * x + a3 * x * x * x;
                double dedx = a1 + 2 * a2 * x + 3 * a3 * x * x;
                double dxdv = -2./3. * pow(v, -2./3.-1.);
                double pp = -dedx * dxdv;
                double h = e + pp * v;
                ofs << fmt::format("{:20.16f} {:20.16f} {:20.16f} {:20.16f}\n", v, e, pp * eVA3_to_GPa, h);
            }
            ofs.close();
            if (!ofs) GPERROR("Failed to write EOS interpolation: " + vephfile);
        }
        return 0;
    }

    // ===== PV mode =====
    if (!iobj.read(atoms, infile, informat) || atoms.empty() || atoms.back().size() == 0) {
        std::cerr << "Error: no structures loaded from " << infile << '\n';
        return 1;
    }
    atom = atoms.back();
    Potentials::potential_wrapper wrapper("", "", settings.ExtractSubSettings("pot"));
    RelaxationDrvV2 optdrv(settings.ExtractSubSettings("relax"));

    if (!skip_pre_relax) {
        optdrv.init(wrapper.pot, atom);
        optdrv.run();
        if (!check_relaxation(optdrv, "PV preliminary")) return 2;
        atom = optdrv.atom;
    } else {
        wrapper.pot->getEFS(atom);
    }

    std::vector<double> press_positive(npositive);
    std::vector<double> press_negative(nnegitave);

    atoms.clear();
    for (int i = 1; i <= npositive; ++i) {
        press_positive[i-1] = press_ub * i / npositive / eVA3_to_GPa;
    }
    for (int i = 1; i <= nnegitave; ++i) {
        press_negative[i-1] = press_lb * i / nnegitave / eVA3_to_GPa;
    }
    std::vector<double> results(npositive + 1 + nnegitave);
    std::vector<double> pressures(npositive + 1 + nnegitave, 0);
    atoms.resize(nnegitave + npositive + 1);
    atoms[nnegitave] = atom;
    // The reference may be unrelaxed (--skip-prelax), or relaxed at a nonzero
    // relax:pressure. Report its actual hydrostatic pressure and E + PV rather
    // than silently labelling that state as zero pressure with H = E.
    const double reference_pressure = atom.stress_.trace() / (3.0 * atom.get_volume());
    pressures[nnegitave] = reference_pressure * eVA3_to_GPa;
    results[nnegitave] = atom.energy_ + reference_pressure * atom.get_volume();
    Atoms at = atom;
    for (int i = 0; i < npositive; ++i) {
        optdrv.control.pressure = press_positive[i];
        optdrv.init(wrapper.pot, at);
        optdrv.run();
        if (!check_relaxation(optdrv, fmt::format("PV pressure {} GPa", press_positive[i] * eVA3_to_GPa))) return 2;
        at = optdrv.atom;
        atoms[nnegitave + 1 + i] = at;
        results[nnegitave + 1 + i] = at.energy_ + optdrv.control.pressure * at.get_volume();
        pressures[nnegitave + 1 + i] = press_positive[i] * eVA3_to_GPa;
    }
    at = atom;
    for (int i = 0; i < nnegitave; ++i) {
        optdrv.control.pressure = press_negative[i];
        optdrv.init(wrapper.pot, at);
        optdrv.run();
        if (!check_relaxation(optdrv, fmt::format("PV pressure {} GPa", press_negative[i] * eVA3_to_GPa))) return 2;
        at = optdrv.atom;
        atoms[nnegitave - 1 - i] = at;
        results[nnegitave - 1 - i] = at.energy_ + optdrv.control.pressure * at.get_volume();
        pressures[nnegitave - 1 - i] = press_negative[i] * eVA3_to_GPa;
    }

    if (!iobj.write(atoms, trjfile, trjformat))
        GPERROR("Failed to write EOS trajectory: " + trjfile);

    ofstream ofs(outfile);
    if (!ofs) GPERROR("Cannot open EOS curve output: " + outfile);
    ofs << "# Pressure   Volume    Energy/at  H/at\n";
    for (int i = 0; i < atoms.size(); ++i) {
        double vv = atoms[i].get_volume() / atoms[i].size();
        double ee = atoms[i].energy_ / atoms[i].size();
        double hh = results[i] / atoms[i].size();
        double pp = pressures[i];
        ofs << fmt::format("{:10.6f} {:10.6f} {:10.6f} {:10.6f}\n", pp, vv, ee, hh);
    }
    ofs.close();
    if (!ofs) GPERROR("Failed to write EOS curve: " + outfile);

    return 0;
}
