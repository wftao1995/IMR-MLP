// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file mlpcalccellhess.cpp
 * @author wftao (wftao1995@gmail.com)
 * @brief Calculate elastic constants analytically (Hessian-based) or numerically
 *        (stress-strain / energy-strain finite difference).
 * @version 0.2
 * @date 2023-2026
 * @copyright Copyright (c) 2023-2026
 *
 * Analytical: direct Hessian computation from MTP potential, with relaxation
 * correction via pseudo-inverse of force-constant matrix.
 *
 * Numerical: finite-difference stress-strain or energy-strain fitting using
 * the generic potential wrapper, with spacegroup-based symmetry averaging.
 */
#include "potentials/potential_mtp.h"
#include "potentials/potential_wrapper.h"
#include "io/gpio.h"
#include "external/spglib_driver.h"
#include "mindriver/relax_new.h"
#include "utils/algos.h"
#include "tools/constants.h"
#include "tools/elastic_utils.h"
using namespace std;

int calc_elastic(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    double pressure = 0;
    string infile = "POSCAR";
    string informat = "";
    string potfile = "pot.mtp";
    string pottype = "mtp";
    string felastic = "elastic.out";
    string feigensys = "eigensys.out";
    string fsettings = "";
    Atoms atom;
    vector<Atoms> atoms;
    GPIO iobj;
    string indexfile = "index.map";
    double symprec = 1e-4;
    double eigzero_tol = 1e-4;
    bool geom_avg = false;

    // Method selection: empty = auto-detect, "analytical" = force, "numerical" = force
    string method = "auto";

    // Analytical-specific
    bool relax_correction = true;
    bool correction_l1 = false;
    bool correction_l2 = false;
    bool compute_eigsys = false;

    // Numerical-specific
    int mode = 0; // 0=stress-strain, 1=energy-strain
    int npoints = 5;
    double estrain = 0.01;
    bool debug = false;

    cmdopts.addop(informat, "input-format", "if", "Input file format")
           .addop(indexfile, "index", "id", "Index file")
           .addop(symprec, "symprec", "symprec", "Tolerance for symmetry finding")
           .addop(geom_avg, "use-gemo-mean", "g", "Use geometric mean instead of arithmetic")
           .addop(eigzero_tol, "eigzero-tol", "etol", "Tolerance for zero valued eigenvalues")
           .addop(pressure, "pressure", "p", "External pressure in GPa")
           .addop(potfile, "potfile", "pot", "Potential file", true, 1)
           .addop(infile, "input", "in", "Input structure file", true, 1)
           .addop(felastic, "elastic-output", "eo", "Elastic constant output file")
           .addop(feigensys, "eigensys-output", "", "Analytical eigensystem output file")
           .addop(fsettings, "settings", "", "Settings file")
           // Method selection
           .addop(method, "method", "", "Calculation method: auto-detect (default), analytical, or numerical")
           // Analytical
           .addop(compute_eigsys, "compute-eigsys", "", "[Analytical] Compute full super-Hessian eigensystem", false, 0)
           .addop(relax_correction, "relax-correction", "cr", "[Analytical] Ionic relaxation correction (default: yes)")
           .addop(correction_l1, "correction-l1", "c1", "[Analytical] 1st-order correction 1")
           .addop(correction_l2, "correction-l2", "c2", "[Analytical] 1st-order correction 2")
           // Numerical
           .addop(pottype, "pottype", "pt", "Potential type (default: mtp)")
           .addop(estrain, "strain-magnitude", "emag", "[Numerical] Magnitude of applied strain")
           .addop(npoints, "num-strain-points", "n", "[Numerical] Number of strain points (odd)")
           .addop(mode, "mode", "", "[Numerical] Mode: 0=stress-strain (default), 1=energy-strain")
           .addop(debug, "debug", "", "[Numerical] Debug output enabled", false, 1, 10)
           .add_request("potfile").add_request("input")
           .attach_settings("pot:", "Potential settings",
               [](std::ostream &os) { Potentials::potential_wrapper().ListSettings(os, "pot:"); })
           .attach_settings("relax:", "Numerical-method relaxation settings",
               [](std::ostream &os) { RelaxationDrvV2().ListSettings(os, "relax:"); });

    cmdopts.set_helponexit(false);
    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        exit(0);
    }

    // Load settings file (shared by both paths)
    GPSettings settings;
    if (!fsettings.empty() && !GPIO::file_exist(fsettings))
        throw std::runtime_error("Settings file does not exist: " + fsettings);
    if (!fsettings.empty()) {
        settings.Load(fsettings);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    if (method != "auto" && method != "analytical" && method != "numerical")
        throw std::invalid_argument("--method must be auto, analytical, or numerical");
    if (!std::isfinite(pressure))
        throw std::invalid_argument("--pressure must be finite");
    if (mode != 0 && mode != 1)
        throw std::invalid_argument("--mode must be 0 or 1");
    if (!std::isfinite(estrain) || estrain <= 0.0)
        throw std::invalid_argument("--strain-magnitude must be finite and positive");
    if (mode == 0 && (npoints < 3 || npoints % 2 == 0))
        throw std::invalid_argument("stress-strain mode requires an odd --num-strain-points >= 3");
    if (!std::isfinite(eigzero_tol) || eigzero_tol <= 0.0)
        throw std::invalid_argument("--eigzero-tol must be finite and positive");

    // Read structure
    iobj.readmap(indexfile);
    if (!iobj.read(atoms, infile, informat))
        throw std::runtime_error("Failed to read complete input: " + infile);
    if (atoms.empty())
        throw std::runtime_error("No structure was read from " + infile);
    atom = atoms.back();

    // Load potential (shared by both paths)
    Potentials::potential_wrapper wrapper(potfile, pottype, settings.ExtractSubSettings("pot"));
    Potentials::AnyPotential* pot = wrapper.pot;
    pot->omp_init();

    // Auto-fallback: detect if potential supports analytical cell Hessian
    bool has_analytical = pot->supports_analytical_cell_hessian();
    if (method == "analytical" && !has_analytical) {
        cerr << "ERROR: Potential type \"" << pottype << "\" does not support analytical cell Hessian. "
             << "Use --method numerical or omit --method for auto-fallback." << endl;
        exit(128);
    }
    if (method == "auto") {
        if (has_analytical) {
            method = "analytical";
        } else {
            Message("Potential does not support analytical Hessian, falling back to numerical method");
            method = "numerical";
        }
    }

    // ===== Numerical method =====
    if (method == "numerical") {
        if (atom.size() == 0 || atom.get_volume() < 1e-4) {
            cerr << "ERROR: no atoms or not PBC cell\n";
            exit(128);
        }
        RelaxationDrvV2 relaxdrv(settings.ExtractSubSettings("relax"));
        relaxdrv.set_isif(2);

        // Get symmetry info
        auto symmetry_dataset = Symmetry3D::get_sym_dataset(atom, symprec);
        std::vector<Eigen::Matrix3d> symmetry_rotations;
        symmetry_rotations.reserve(symmetry_dataset.symm_ops.size());
        for (const auto &operation : symmetry_dataset.symm_ops)
            symmetry_rotations.push_back(operation.first);
        double vol0 = atom.get_volume();

        // Material stiffness is differentiated with respect to Green-Lagrange
        // strain eta. With row-vector cells, eta = (F F^T - I) / 2.
        auto genstrainmat = [](const Eigen::VectorXd &estrain) -> Eigen::Matrix3d {
            Eigen::Matrix3d mat;
            mat.setZero();
            mat(0, 0) = 1 + 2*estrain(0); mat(1, 1) = 1 + 2*estrain(1); mat(2, 2) = 1 + 2*estrain(2);
            mat(1, 2) = mat(2, 1) = estrain(3);
            mat(0, 2) = mat(2, 0) = estrain(4);
            mat(0, 1) = mat(1, 0) = estrain(5);
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> solver(mat);
            if (solver.info() != Eigen::Success || solver.eigenvalues().minCoeff() <= 0.0)
                throw std::invalid_argument("Applied Green-Lagrange strain requires positive definite I + 2*eta; reduce --strain-magnitude");
            return solver.eigenvectors() * solver.eigenvalues().cwiseSqrt().asDiagonal()
                * solver.eigenvectors().transpose();
        };
        int unconverged_points = 0;
        auto relax_at_strain = [&](Atoms &satom, const Eigen::VectorXd &strain) {
            relaxdrv.init(pot, satom);
            if (relaxdrv.run() != 0) {
                ++unconverged_points;
                const double max_gradient = relaxdrv.g.size() ? relaxdrv.g.cwiseAbs().maxCoeff() : 0.0;
                Warning(fmt::format(
                    "WARNING: calc_elastic: ionic relaxation unconverged at Green-Lagrange strain "
                    "[{}] (xx yy zz 2yz 2xz 2xy): reason={}, max|gradient|={:.6e}, gtol={:.6e}. "
                    "Continuing with the last iterate; elastic constants may be unreliable.",
                    fmt::join(strain, " "), relaxdrv.stop_reason, max_gradient, relaxdrv.tol.gtol));
            }
            satom = relaxdrv.atom;
        };

        Eigen::MatrixXd cmat_main(6, 6);
        cmat_main.setZero();
        ofstream ofsdebug;
        if (debug) ofsdebug.open("debuginfo.log");

        if (mode == 0) {
            for (int ci = 0; ci < 6; ++ci) {
                Eigen::VectorXd epslist;
                linspace(-estrain, estrain, npoints, epslist);
                vector<Eigen::VectorXd> siglist;
                for (auto &ieps : epslist) {
                    Eigen::VectorXd epspos(6);
                    epspos.setZero();
                    epspos(ci) = ieps;
                    Eigen::Matrix3d strainmat = genstrainmat(epspos);
                    Eigen::Matrix3d newcell = atom.cell_ * strainmat;
                    Atoms satom = atom;
                    satom.set_cell(newcell);
                    relax_at_strain(satom, epspos);
                    // Second Piola-Kirchhoff stress S = J F^-T sigma F^-1
                    // is conjugate to eta, unlike the Cauchy stress at finite
                    // prestress. stress_ stores -V*sigma, so J/V = 1/vol0.
                    const Eigen::Matrix3d invstrain = strainmat.inverse();
                    const Eigen::Matrix3d stress = -invstrain.transpose() * satom.stress_ * invstrain / vol0;
                    Eigen::VectorXd sig(6);
                    sig << stress(0, 0), stress(1, 1), stress(2, 2),
                           stress(1, 2), stress(0, 2), stress(0, 1);
                    siglist.push_back(sig);
                    if (debug) ofsdebug << epspos.transpose() << " " << siglist.back().transpose() << endl;
                }
                for (int cj = 0; cj < 6; ++cj) {
                    Eigen::VectorXd sigma(siglist.size());
                    for (size_t si = 0; si < siglist.size(); ++si) sigma(si) = siglist[si](cj);
                    Eigen::MatrixXd design(epslist.size(), 2);
                    design.col(0) = epslist;
                    design.col(1).setOnes();
                    cmat_main(cj, ci) = design.colPivHouseholderQr().solve(sigma)(0);
                }
            }
        } else if (mode == 1) {
            auto geth = [&](Eigen::VectorXd &eps) {
                auto strainmat = genstrainmat(eps);
                Eigen::Matrix3d newcell = atom.cell_ * strainmat;
                Atoms satom = atom;
                satom.set_cell(newcell);
                relax_at_strain(satom, eps);
                return satom.energy_;
            };
            Eigen::VectorXd zero_strain = Eigen::VectorXd::Zero(6);
            const double h0 = geth(zero_strain);
            for (int i = 0; i < 6; ++i) {
                for (int j = i; j < 6; ++j) {
                    Eigen::VectorXd epspos(6);
                    epspos.setZero();
                    if (i == j) {
                        double hpp, hmm;
                        epspos(i) = estrain;  hpp = geth(epspos);
                        epspos(i) = -estrain; hmm = geth(epspos);
                        cmat_main(i, i) = (hpp + hmm - 2.0 * h0) / (estrain * estrain);
                    } else {
                        double hpp, hmm, hpm, hmp;
                        epspos(i) = estrain;  epspos(j) = estrain;  hpp = geth(epspos);
                        epspos(i) = -estrain; epspos(j) = -estrain; hmm = geth(epspos);
                        epspos(i) = estrain;  epspos(j) = -estrain; hpm = geth(epspos);
                        epspos(i) = -estrain; epspos(j) = estrain;  hmp = geth(epspos);
                        cmat_main(i, j) = (hpp + hmm - hpm - hmp) / (4.0 * estrain * estrain);
                        cmat_main(j, i) = cmat_main(i, j);
                    }
                }
            }
        } else {
            cerr << "Unknown mode " << mode << endl;
            exit(128);
        }
        cout << "Original fitted C:\n" << cmat_main << endl;
        cmat_main = 0.5 * (cmat_main + cmat_main.transpose()).eval();

        Eigen::Matrix<double, 6, 6> fitted = cmat_main;
        fitted = ElasticUtils::project_by_symmetry(fitted, atom.cell_, symmetry_rotations);
        Eigen::Matrix<double, 6, 6> Cmat = fitted * (mode == 0 ? eVA3_to_GPa : eVA3_to_GPa / vol0);
        const Eigen::Matrix<double, 6, 6> Bmat = ElasticUtils::born_stiffness(Cmat, pressure);

        double tol = 1e-5;
        Eigen::Matrix<double, 6, 6> Smat = ElasticUtils::compliance(Cmat);
        ofstream ofselastic(felastic);
        if (!ofselastic)
            throw std::runtime_error("Cannot open elastic output: " + felastic);
        ofselastic << "EPS: " << estrain << '\n';
        if (unconverged_points > 0) {
            const std::string warning = fmt::format(
                "calc_elastic: {} strain evaluations had unconverged ionic relaxation; "
                "results include their last iterates and may be unreliable.", unconverged_points);
            Warning("WARNING: " + warning);
            ofselastic << "WARNING: " << warning << '\n';
        }
        string buffer;
        Message("Stiffness tensor (GPa)");
        ofselastic << "Stiffness tensor (GPa)\n";
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (fabs(Cmat(i, j)) < tol) Cmat(i, j) = 0.0;
                buffer = fmt::format("{:> 15.6f} ", Cmat(i, j));
                cout << buffer; ofselastic << buffer;
            }
            cout << endl; ofselastic << '\n';
        }
        Message("Born/Wallace stiffness tensor (GPa)");
        ofselastic << "Born/Wallace stiffness tensor (GPa)\n";
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                buffer = fmt::format("{:> 15.6f} ", std::abs(Bmat(i, j)) < tol ? 0.0 : Bmat(i, j));
                cout << buffer; ofselastic << buffer;
            }
            cout << endl; ofselastic << '\n';
        }
        Message("Compliance tensor (GPa^-1)");
        ofselastic << "Compliance tensor (GPa^-1)\n";
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                if (fabs(Smat(i, j)) < tol) Smat(i, j) = 0.0;
                buffer = fmt::format("{:> 15.6g} ", Smat(i, j));
                cout << buffer; ofselastic << buffer;
            }
            cout << endl; ofselastic << '\n';
        }

        // Elastic properties (shared)
        double c1d = Cmat(0,0)+Cmat(1,1)+Cmat(2,2), c2s = Cmat(0,1)+Cmat(1,2)+Cmat(2,0), c3d = Cmat(3,3)+Cmat(4,4)+Cmat(5,5);
        double s1d = Smat(0,0)+Smat(1,1)+Smat(2,2), s2s = Smat(0,1)+Smat(1,2)+Smat(2,0), s3d = Smat(3,3)+Smat(4,4)+Smat(5,5);
        double Bv = (c1d+2*c2s)/9., Gv = (c1d-c2s+3*c3d)/15., Ev = 9*Bv*Gv/(3*Bv+Gv), vv = 0.5*(3*Bv-2*Gv)/(3*Bv+Gv), Hv = (1.-2.*vv)/(1.+vv)/6.;
        double vBr = s1d+2*s2s, vGr = (4*s1d-4*s2s+3*s3d)/15., Br = 1./vBr, Gr = 1./vGr, Er = 9*Br*Gr/(3*Br+Gr), vr = 0.5*(3.*Br-2.*Gr)/(3.*Br+Gr), Hr = (1.-2.*vr)/(1.+vr)/6.;
        double Bh, Gh;
        if (geom_avg) {
            if (Bv * Br < 0.0 || Gv * Gr < 0.0)
                throw std::runtime_error("Geometric Voigt-Reuss average is undefined for opposite-sign moduli");
            Bh = sqrt(Bv*Br); Gh = sqrt(Gv*Gr);
        }
        else { Bh = (Bv+Br)/2.; Gh = (Gv+Gr)/2.; }
        double Eh = 9.*Bh*Gh/(3.*Bh+Gh), vh = 0.5*(3.*Bh-2.*Gh)/(3.*Bh+Gh), Hh = (1.-2.*vh)/(1.+vh)/6.;
        Message("Elastic properties");
        ofselastic << "Elastic properties\n";
        cout << fmt::format("    {:<15} {:<15} {:<15} \n", "Voigt", "Ruess", "Hill");
        cout << fmt::format("B   {:<15.5f} {:<15.5f} {:<15.5f} \n", Bv, Br, Bh);
        cout << fmt::format("G   {:<15.5f} {:<15.5f} {:<15.5f} \n", Gv, Gr, Gh);
        cout << fmt::format("E   {:<15.5f} {:<15.5f} {:<15.5f} \n", Ev, Er, Eh);
        cout << fmt::format("v   {:<15.5f} {:<15.5f} {:<15.5f} \n", vv, vr, vh);
        cout << fmt::format("G/(9B) {:<15.5f} {:<15.5f} {:<15.5f} \n", Hv, Hr, Hh);
        ofselastic << fmt::format("    {:<15} {:<15} {:<15} \n", "Voigt", "Ruess", "Hill");
        ofselastic << fmt::format("B   {:<15.5f} {:<15.5f} {:<15.5f} \n", Bv, Br, Bh);
        ofselastic << fmt::format("G   {:<15.5f} {:<15.5f} {:<15.5f} \n", Gv, Gr, Gh);
        ofselastic << fmt::format("E   {:<15.5f} {:<15.5f} {:<15.5f} \n", Ev, Er, Eh);
        ofselastic << fmt::format("v   {:<15.5f} {:<15.5f} {:<15.5f} \n", vv, vr, vh);
        ofselastic << fmt::format("G/(9B) {:<15.5f} {:<15.5f} {:<15.5f} \n", Hv, Hr, Hh);
        cout << fmt::format("1.0 / S[0][0] = {}\n", 1.0/Smat(0,0));
        ofselastic << fmt::format("1.0 / S[0][0] = {}\n", 1.0/Smat(0,0));

        // Stiffness eigenvalues + Born stability
        ofselastic << "# BORN/WALLACE STIFFNESS EIGENSYSTEM\n";
        cout << "Eigenvalues of Born/Wallace stiffness" << endl;
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver(Bmat, Eigen::ComputeEigenvectors);
        if (eigensolver.info() != Eigen::Success)
            throw std::runtime_error("Failed to diagonalize Born/Wallace stiffness");
        auto eigval = eigensolver.eigenvalues();
        bool positive_definite = true;
        for (int i = 0; i < 6; ++i) {
            if (eigval(i) < 0) positive_definite = false;
            ofselastic << fmt::format("{:> 16.8f}\n", eigval(i));
            cout << fmt::format("{:> 16.8f}", eigval(i)) << endl;
        }
        if (positive_definite) {
            cout << "Born elastic stability conditions meet" << endl;
            ofselastic << "Born elastic stability conditions meet\n";
        } else {
            cout << "Born elastic stability conditions NOT meet" << endl;
            ofselastic << "Born elastic stability conditions NOT meet\n";
        }
        ofselastic.close();
        if (!ofselastic)
            throw std::runtime_error("Failed to write elastic output: " + felastic);
        return 0;
    }

    // ===== Analytical method =====
    auto *mtp = dynamic_cast<Potentials::PotMTP*>(pot);
    if (!mtp && (correction_l1 || correction_l2))
        throw std::runtime_error("correction-l1 and correction-l2 require a PotMTP potential");
    if (mtp) {
        mtp->hc_correction = correction_l1;
        mtp->rR_correction = correction_l2;
    }

    Message("Computing full HESSIAN matrix");
    Potentials::HessianMatrix hess;
    Eigen::MatrixXd hesscell, lrhess;
    pot->getEFSHessianCellSymmOMP(atom, hess, hesscell, lrhess, symprec);

    if (compute_eigsys) {
        Eigen::MatrixXd eigsys;
        eigsys.resize(atom.size()*3+9, atom.size()*3+9);
        eigsys.setZero();
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 9; ++j)
                eigsys(i, j) = hesscell(i, j);
        for (int i = 0; i < atom.size(); ++i)
            for (int j = 0; j < atom.size(); ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        eigsys(9+3*i+a, 9+3*j+b) = hess(i, a, j, b);
        for (int i = 0; i < 9; ++i)
            for (int j = 0; j < 3*atom.size(); ++j) {
                eigsys(i, 9+j) = lrhess(i, j);
                eigsys(9+j, i) = lrhess(i, j);
            }
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigsol(eigsys, Eigen::ComputeEigenvectors);
        if (eigsol.info() != Eigen::Success)
            throw std::runtime_error("Failed to diagonalize the combined cell/atomic Hessian");
        auto eigval = eigsol.eigenvalues();
        auto eigvec = eigsol.eigenvectors();
        ofstream ofs("eigtotal.dat");
        if (!ofs)
            throw std::runtime_error("Cannot open full Hessian output: eigtotal.dat");
        ofs << "FULL HESSIAN\n";
        for (int i = 0; i < atom.size()*3+9; ++i)
            ofs << fmt::format("{:> 16.8f}\n", fmt::join(eigsys.row(i), " "));
        ofs << "EIGEN SYSTEM\n";
        for (int i = 0; i < atom.size()*3+9; ++i)
            ofs << fmt::format("{:> 16.8f}  {:> 16.8f}\n", eigval(i), fmt::join(eigvec.col(i).transpose(), " "));
        ofs.close();
        if (!ofs)
            throw std::runtime_error("Failed to write full Hessian output: eigtotal.dat");
    }

    Eigen::MatrixXd fcmatrix, fcpinv;
    if (relax_correction) {
        Message("Computing ionic correction");
        fcmatrix.resize(3*atom.size(), 3*atom.size());
        fcpinv.resize(3*atom.size(), 3*atom.size());
        for (int i = 0; i < atom.size(); ++i)
            for (int j = 0; j < atom.size(); ++j)
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        fcmatrix(i*3+a, j*3+b) = hess(i, a, j, b);

        fcmatrix = 0.5 * (fcmatrix + fcmatrix.transpose()).eval();
        const int dim = 3 * atom.size();
        Eigen::MatrixXd translations = Eigen::MatrixXd::Zero(dim, 3);
        for (int i = 0; i < atom.size(); ++i)
            for (int axis = 0; axis < 3; ++axis)
                translations(3 * i + axis, axis) = 1.0 / std::sqrt(static_cast<double>(atom.size()));
        Eigen::HouseholderQR<Eigen::MatrixXd> qr(translations);
        const Eigen::MatrixXd qfull = qr.householderQ() * Eigen::MatrixXd::Identity(dim, dim);
        const Eigen::MatrixXd internal_basis = qfull.rightCols(dim - 3);
        if (dim == 3) {
            fcpinv.setZero();
        } else {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(
                internal_basis.transpose() * fcmatrix * internal_basis, Eigen::ComputeEigenvectors);
            if (eigensolver.info() != Eigen::Success)
                throw std::runtime_error("Failed to diagonalize the internal force-constant matrix");
            const auto eigval = eigensolver.eigenvalues();
            cout << fmt::format("Internal force-constant eigenvalues: {:10f}\n", fmt::join(eigval, " "));
            const double scale = std::max(1.0, eigval.cwiseAbs().maxCoeff());
            const double threshold = eigzero_tol * scale;
            for (int i = 0; i < eigval.size(); ++i) {
                if (eigval(i) <= threshold)
                    throw std::runtime_error("Ionic relaxation correction is undefined for a soft or singular internal mode; use --relax-correction=false");
            }
            fcpinv = internal_basis * eigensolver.eigenvectors()
                * eigval.cwiseInverse().asDiagonal() * eigensolver.eigenvectors().transpose()
                * internal_basis.transpose();
        }
        hesscell -= lrhess * fcpinv * lrhess.transpose();
    }

    Message("Computing elastic constants");
    Eigen::MatrixXd cmatall;
    cmatall.resize(9, 9);
    cmatall.setZero();
    for (int mu = 0; mu < 3; ++mu)
        for (int beta = 0; beta < 3; ++beta)
            for (int nu = 0; nu < 3; ++nu)
                for (int phi = 0; phi < 3; ++phi) {
                    int id1 = 3*mu + beta, id2 = 3*nu + phi;
                    for (int alpha = 0; alpha < 3; ++alpha)
                        for (int lambda = 0; lambda < 3; ++lambda)
                            cmatall(id1, id2) += atom.cell_(alpha, mu) *
                                hesscell(3*alpha+beta, 3*lambda+phi) * atom.cell_(lambda, nu);
                }

    // E_(F_ab F_cd) = V*C_abcd + V*sigma_ac*delta_bd.
    // Convert to the Green-Lagrange material tensor before Voigt mapping.
    // The stored extensive virial is -V*sigma (not the requested pressure).
    for (int a = 0; a < 3; ++a)
        for (int c = 0; c < 3; ++c)
            for (int b = 0; b < 3; ++b)
                cmatall(3*a+b, 3*c+b) += atom.stress_(a, c);
    Eigen::Matrix<double, 6, 6> Cmat = ElasticUtils::tensor_to_voigt(cmatall);
    auto symmetry_dataset = Symmetry3D::get_sym_dataset(atom, symprec);
    std::vector<Eigen::Matrix3d> symmetry_rotations;
    symmetry_rotations.reserve(symmetry_dataset.symm_ops.size());
    for (const auto &operation : symmetry_dataset.symm_ops)
        symmetry_rotations.push_back(operation.first);
    Cmat = ElasticUtils::project_by_symmetry(Cmat, atom.cell_, symmetry_rotations);
    const Eigen::Matrix<double, 6, 6> Cmatorg = Cmat;

    double tol = 1e-10;
    Cmat *= eVA3_to_GPa / atom.get_volume();
    const Eigen::Matrix<double, 6, 6> Bmat = ElasticUtils::born_stiffness(Cmat, pressure);
    Eigen::Matrix<double, 6, 6> Smat = ElasticUtils::compliance(Cmat);
    ofstream ofselastic(felastic);
    if (!ofselastic)
        throw std::runtime_error("Cannot open elastic output: " + felastic);
    string buffer;
    Message("Stiffness tensor (GPa)");
    ofselastic << "Stiffness tensor (GPa)\n";
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (fabs(Cmat(i, j)) < tol) Cmat(i, j) = 0.0;
            buffer = fmt::format("{:> 15.6f} ", Cmat(i, j));
            cout << buffer; ofselastic << buffer;
        }
        cout << endl; ofselastic << '\n';
    }
    Message("Born/Wallace stiffness tensor (GPa)");
    ofselastic << "Born/Wallace stiffness tensor (GPa)\n";
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            buffer = fmt::format("{:> 15.6f} ", std::abs(Bmat(i, j)) < tol ? 0.0 : Bmat(i, j));
            cout << buffer; ofselastic << buffer;
        }
        cout << endl; ofselastic << '\n';
    }

    Message("Compliance tensor (GPa^-1)");
    ofselastic << "Compliance tensor (GPa^-1)\n";
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            if (fabs(Smat(i, j)) < tol) Smat(i, j) = 0.0;
            buffer = fmt::format("{:> 15.6g} ", Smat(i, j));
            cout << buffer; ofselastic << buffer;
        }
        cout << endl; ofselastic << '\n';
    }

    // Elastic properties
    double c1d = Cmat(0,0)+Cmat(1,1)+Cmat(2,2), c2s = Cmat(0,1)+Cmat(1,2)+Cmat(2,0), c3d = Cmat(3,3)+Cmat(4,4)+Cmat(5,5);
    double s1d = Smat(0,0)+Smat(1,1)+Smat(2,2), s2s = Smat(0,1)+Smat(1,2)+Smat(2,0), s3d = Smat(3,3)+Smat(4,4)+Smat(5,5);
    double Bv = (c1d+2*c2s)/9., Gv = (c1d-c2s+3*c3d)/15., Ev = 9*Bv*Gv/(3*Bv+Gv), vv = 0.5*(3*Bv-2*Gv)/(3*Bv+Gv), Hv = (1.-2.*vv)/(1.+vv)/6.;
    double vBr = s1d+2*s2s, vGr = (4*s1d-4*s2s+3*s3d)/15., Br = 1./vBr, Gr = 1./vGr, Er = 9*Br*Gr/(3*Br+Gr), vr = 0.5*(3.*Br-2.*Gr)/(3.*Br+Gr), Hr = (1.-2.*vr)/(1.+vr)/6.;
    double Bh, Gh;
    if (geom_avg) {
        if (Bv * Br < 0.0 || Gv * Gr < 0.0)
            throw std::runtime_error("Geometric Voigt-Reuss average is undefined for opposite-sign moduli");
        Bh = sqrt(Bv*Br); Gh = sqrt(Gv*Gr);
    }
    else { Bh = (Bv+Br)/2.; Gh = (Gv+Gr)/2.; }
    double Eh = 9.*Bh*Gh/(3.*Bh+Gh), vh = 0.5*(3.*Bh-2.*Gh)/(3.*Bh+Gh), Hh = (1.-2.*vh)/(1.+vh)/6.;
    Message("Elastic properties");
    ofselastic << "Elastic properties\n";
    cout << fmt::format("    {:<15} {:<15} {:<15} \n", "Voigt", "Ruess", "Hill");
    cout << fmt::format("B   {:<15.5f} {:<15.5f} {:<15.5f} \n", Bv, Br, Bh);
    cout << fmt::format("G   {:<15.5f} {:<15.5f} {:<15.5f} \n", Gv, Gr, Gh);
    cout << fmt::format("E   {:<15.5f} {:<15.5f} {:<15.5f} \n", Ev, Er, Eh);
    cout << fmt::format("v   {:<15.5f} {:<15.5f} {:<15.5f} \n", vv, vr, vh);
    cout << fmt::format("G/(9B) {:<15.5f} {:<15.5f} {:<15.5f} \n", Hv, Hr, Hh);
    ofselastic << fmt::format("    {:<15} {:<15} {:<15} \n", "Voigt", "Ruess", "Hill");
    ofselastic << fmt::format("B   {:<15.5f} {:<15.5f} {:<15.5f} \n", Bv, Br, Bh);
    ofselastic << fmt::format("G   {:<15.5f} {:<15.5f} {:<15.5f} \n", Gv, Gr, Gh);
    ofselastic << fmt::format("E   {:<15.5f} {:<15.5f} {:<15.5f} \n", Ev, Er, Eh);
    ofselastic << fmt::format("v   {:<15.5f} {:<15.5f} {:<15.5f} \n", vv, vr, vh);
    ofselastic << fmt::format("G/(9B) {:<15.5f} {:<15.5f} {:<15.5f} \n", Hv, Hr, Hh);
    cout << fmt::format("1.0 / S[0][0] = {}\n", 1.0/Smat(0,0));
    ofselastic << fmt::format("1.0 / S[0][0] = {}\n", 1.0/Smat(0,0));

    // Stiffness eigenvalues + Born stability
    ofselastic << "# BORN/WALLACE STIFFNESS EIGENSYSTEM\n";
    cout << "Eigenvalues of Born/Wallace stiffness" << endl;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver2(Bmat, Eigen::ComputeEigenvectors);
    if (eigensolver2.info() != Eigen::Success)
        throw std::runtime_error("Failed to diagonalize Born/Wallace stiffness");
    auto eigval2 = eigensolver2.eigenvalues();
    bool positive_definite = true;
    for (int i = 0; i < 6; ++i) {
        if (eigval2(i) < 0) positive_definite = false;
        ofselastic << fmt::format("{:> 16.8f}\n", eigval2(i));
        cout << fmt::format("{:> 16.8f}", eigval2(i)) << endl;
    }
    if (positive_definite) {
        cout << "Born elastic stability conditions meet" << endl;
        ofselastic << "Born elastic stability conditions meet\n";
    } else {
        cout << "Born elastic stability conditions NOT meet" << endl;
        ofselastic << "Born elastic stability conditions NOT meet\n";
    }

    ofselastic.close();
    if (!ofselastic)
        throw std::runtime_error("Failed to write elastic output: " + felastic);

    // Analytical eigensystem output
    ofstream ofs(feigensys);
    if (!ofs)
        throw std::runtime_error("Cannot open eigensystem output: " + feigensys);
    ofs << "# cellhess written in 9x9 matrix. l11 l12 l13 l21 l22 l23 l31 l32 l33  (eV/A^2)\n";
    for (int i = 0; i < 9; ++i)
        ofs << fmt::format("{:> 16.8f}\n", fmt::join(hesscell.row(i), " "));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(hesscell, Eigen::ComputeEigenvectors);
    if (eigensolver.info() != Eigen::Success)
        throw std::runtime_error("Failed to diagonalize cell Hessian");
    auto eigval_hc = eigensolver.eigenvalues();
    Eigen::MatrixXd eigvec_hc = eigensolver.eigenvectors();
    ofs << "# CELL EIGENSYSTEM\n";
    for (int i = 0; i < 9; ++i)
        ofs << fmt::format("{:> 16.8f}  {:> 16.8f}\n", eigval_hc(i), fmt::join(eigvec_hc.col(i).transpose(), " "));
    ofs << "# C_ijkl written in 9x9 matrix. e11 e12 e13 e21 e22 e23 e31 e32 e33  (eV)\n";
    for (int i = 0; i < 9; ++i)
        ofs << fmt::format("{:> 16.8f}\n", fmt::join(cmatall.row(i), " "));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver3(cmatall, Eigen::ComputeEigenvectors);
    if (eigensolver3.info() != Eigen::Success)
        throw std::runtime_error("Failed to diagonalize Cartesian elastic tensor");
    auto eigval3 = eigensolver3.eigenvalues();
    Eigen::MatrixXd eigvec3 = eigensolver3.eigenvectors();
    ofs << "# C_ijkl EIGENSYSTEM\n";
    for (int i = 0; i < 9; ++i)
        ofs << fmt::format("{:> 16.8f}  {:> 16.8f}\n", eigval3(i), fmt::join(eigvec3.col(i).transpose(), " "));
    ofs << "# C_ij written in 6x6 matrix, e11 e22 e33 e23 e31 e12 (eV)\n";
    for (int i = 0; i < 6; ++i)
        ofs << fmt::format("{:> 16.8f}\n", fmt::join(Cmatorg.row(i), " "));
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, 6, 6>> eigensolver4(Cmatorg, Eigen::ComputeEigenvectors);
    if (eigensolver4.info() != Eigen::Success)
        throw std::runtime_error("Failed to diagonalize material stiffness");
    auto eigval4 = eigensolver4.eigenvalues();
    Eigen::MatrixXd eigvec4 = eigensolver4.eigenvectors();
    ofs << "# C_ij EIGENSYSTEM\n";
    for (int i = 0; i < 6; ++i)
        ofs << fmt::format("{:> 16.8f}  {:> 16.8f}\n", eigval4(i), fmt::join(eigvec4.col(i).transpose(), " "));

    ofs.close();
    if (!ofs)
        throw std::runtime_error("Failed to write eigensystem output: " + feigensys);
    return 0;
}
