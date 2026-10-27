// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "potentials/potential_wrapper.h"
#include "external/spglib_driver.h"
#include "tools/chem.h"
#include "tools/tools.h"
#include "tools/hessian_utils.h"
#include "analyse/compute_hessian.h"
#include "analyse/lowest_eigenmodes.h"
#include <stdexcept>

using namespace std;
int calc_fc(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    Atoms atom, supercell;
    std::vector<Atoms> atoms;
    string infile = "POSCAR";
    string informat = "";
    string indexfile = "index.map";
    string settingfile = "fc.ini";
    GPSettings settings;
    string inpot = "potential.mtp";
    string inpt = "mtp";
    vector<int> dim;
    Eigen::Matrix3d super_trans = Eigen::Matrix3d::Identity();
    GPIO iobj;
    bool symmreduce = true;
    double symprec = 1e-5;
    double freqconv = 15.633304237154925;
    double toeng = 0.06465415132978451;
    string hessian_method = "analytic";
    double hessian_nudge_pos = 0.001;
    double hessian_cutoff = 0.0;
    int lowest_modes = 0;
    double lowest_mode_tol = 1.0e-8;
    int lowest_mode_max_matvecs = 0;

    bool write_fc = true;
    bool check_symm = false;

    cmdopts.addop(indexfile, "index", "id", "index file")
           .addop(informat, "input-format", "if", "input structure format")
           .addop(write_fc, "write-force-constant", "fc", "write out FORCE_CONSTANT file or not, default true")
           .addop(check_symm, "check-symm", "", "check symmetry", false, 0, 10)
           .addop(symmreduce, "reduced", "rd", "symmetry reduced FORCE_CONSTANTS file, default is true")
           .addop(symprec, "symprec", "tol", "tolerance for symmetry finding")
           .addop(dim, "supercell", "dim", "dimension for supercell, left in row", false, -1, 5)
           .addop(settingfile, "settings", "", "setting file", false)
           .addop(inpot, "potfile", "pot", "potential file", true)
           .addop(inpt, "potential-type", "pt", "potential type", false)
           .addop(infile, "input", "in", "Input structure file", true)
           .addop(hessian_method, "hessian-method", "", "Hessian backend: analytic, auto, force_fd, energy_fd, or hybrid")
           .addop(hessian_nudge_pos, "hessian-nudge-pos", "", "Cartesian displacement for finite-difference Hessians")
           .addop(hessian_cutoff, "hessian-cutoff", "", "neighbor cutoff for finite-difference Hessians; non-positive uses the potential cutoff")
           .addop(lowest_modes, "lowest-modes", "", "matrix-free number of lowest internal Gamma modes (requires analytic HVP support)")
           .addop(lowest_mode_tol, "lowest-mode-tol", "", "relative residual tolerance for matrix-free lowest modes")
           .addop(lowest_mode_max_matvecs, "lowest-mode-max-matvecs", "", "maximum Hessian-vector products; non-positive uses max(400, 20*N_modes)")
           .add_request("potfile")
           .add_request("input")
           .attach_settings("pot:", "Potential settings",
               [](std::ostream &os) { Potentials::potential_wrapper().ListSettings(os, "pot:"); });
    cmdopts.set_helponexit(false);
    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        return 0;
    }
    if (GPIO::file_exist(settingfile))
    {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    if (!std::isfinite(hessian_nudge_pos) || hessian_nudge_pos <= 0.0
        || !std::isfinite(hessian_cutoff))
        throw std::invalid_argument("Invalid Hessian finite-difference option");
    if (lowest_modes < 0 || !std::isfinite(lowest_mode_tol) ||
        lowest_mode_tol <= 0.0)
        throw std::invalid_argument("Invalid matrix-free lowest-mode option");
    if (GPIO::file_exist(indexfile)) {
        iobj.readmap(indexfile);
    } else {
        throw std::runtime_error("Cannot open index file: " + indexfile);
    }

    if (dim.size() == 3) {
        for (int a = 0; a < 3; ++a) {
            super_trans(a, a) = dim[a];
        }
    } else if (dim.size() == 9) {
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                super_trans(a, b) = dim[3*a + b];
            }
        }
    } else if (!dim.empty()) {
        throw std::invalid_argument("--supercell requires exactly 3 or 9 integer values");
    }
    const double determinant = super_trans.determinant();
    const long long determinant_integer = std::llround(determinant);
    if (!dim.empty() && (determinant_integer <= 0 || std::abs(determinant - determinant_integer) > 1e-8))
        throw std::invalid_argument("--supercell transformation must have a positive integer determinant");

    std::map<int, double> mass;
    for (const auto &[type, name] : iobj.getmap()) {
        // index.map: element, radius, optional mass in amu.
        const auto values = iobj.atomic_vals.find(type);
        if (values != iobj.atomic_vals.end() && values->second.size() >= 2) {
            mass[type] = values->second[1];
        } else {
            const auto z = Chemicals::atomic_numbers.find(name);
            if (z == Chemicals::atomic_numbers.end())
                throw std::invalid_argument("Unknown element in index map: " + name);
            mass[type] = Chemicals::atomic_masses[z->second];
        }
        if (!std::isfinite(mass[type]) || mass[type] <= 0.0)
            throw std::invalid_argument("Atomic mass must be finite and positive for type "
                                        + std::to_string(type) + " (" + name + ")");
    }

    if (!iobj.read(atoms, infile, informat))
        throw std::runtime_error("Failed to read complete input: " + infile);
    if (atoms.empty())
        throw std::runtime_error("No structure was read from " + infile);
    atom = atoms[0];
    atom.wrap();

    supercell = atom;
    if (!dim.empty())
        supercell.createSuperL(super_trans);
    iobj.write_one(supercell, "supercell_created.xsf");

    int natoms_orig = atom.size();
    int natoms_supe = supercell.size();
    Potentials::potential_wrapper wrapper(inpot, inpt, settings.ExtractSubSettings("pot"));
    auto *pot = wrapper.pot;
    pot->omp_init();
    Potentials::HessianMatrix hessian;
    if (lowest_modes > 0) {
        if (hessian_method != "analytic" &&
            hessian_method != "auto")
            throw std::invalid_argument(
                "--lowest-modes requires an analytic Hessian-vector product");
        if (!pot->supports_analytical_hessian_vector_product())
            throw std::invalid_argument(
                "The selected potential does not support analytic Hessian-vector products");
        if (natoms_supe <= 1 || lowest_modes > 3 * natoms_supe - 3)
            throw std::invalid_argument(
                "--lowest-modes exceeds the number of internal Cartesian modes");
        if (check_symm)
            throw std::invalid_argument(
                "--lowest-modes is incompatible with full-matrix export or checks");
        if (write_fc)
            Warning("--lowest-modes is matrix-free; FORCE_CONSTANTS will not be written");

        std::vector<double> atom_masses(natoms_supe);
        for (int i = 0; i < natoms_supe; ++i) {
            const int type = supercell.chemmap_[i];
            const auto values = iobj.atomic_vals.find(type);
            double mass;
            if (values != iobj.atomic_vals.end() && values->second.size() >= 2) {
                mass = values->second[1];
            } else {
                const auto element = Chemicals::atomic_numbers.find(iobj.get_type(type));
                if (element == Chemicals::atomic_numbers.end())
                    throw std::invalid_argument("Unknown element for atom type " + std::to_string(type));
                mass = Chemicals::atomic_masses[element->second];
            }
            if (!std::isfinite(mass) || mass <= 0.0)
                throw std::invalid_argument("No positive mass for atom type " + std::to_string(type));
            atom_masses[i] = mass;
        }
        const Eigen::VectorXd sqrt_mass_coordinates =
            HessianUtils::sqrt_mass_coordinates(atom_masses);
        std::array<Eigen::VectorXd, 3> translations;
        double total_mass = 0.0;
        for (double value : atom_masses)
            total_mass += value;
        for (int axis = 0; axis < 3; ++axis) {
            translations[axis] = Eigen::VectorXd::Zero(3 * natoms_supe);
            for (int atom_index = 0; atom_index < natoms_supe; ++atom_index)
                translations[axis](3 * atom_index + axis) =
                    std::sqrt(atom_masses[atom_index] / total_mass);
        }
        auto project_translations = [&](Eigen::VectorXd &value) {
            for (const auto &translation : translations)
                value.noalias() -= translation * translation.dot(value);
        };

        Neighborhoods hvp_neighborhoods(supercell, pot->cutoff);
        auto apply_dynamical_matrix = [&](const Eigen::VectorXd &input,
                                          Eigen::VectorXd &output) {
            Eigen::VectorXd projected = input;
            project_translations(projected);
            const Eigen::VectorXd cartesian_direction =
                (projected.array() / sqrt_mass_coordinates.array()).matrix();
            Eigen::VectorXd cartesian_product;
            pot->applyHessianVectorProduct(
                supercell, hvp_neighborhoods, cartesian_direction,
                cartesian_product);
            output = (cartesian_product.array() /
                      sqrt_mass_coordinates.array()).matrix();
            project_translations(output);
        };

        MatrixFree::LowestEigenmodesSettings mode_settings;
        mode_settings.count = lowest_modes;
        mode_settings.tolerance = lowest_mode_tol;
        mode_settings.max_matvecs = lowest_mode_max_matvecs > 0
            ? lowest_mode_max_matvecs
            : std::max(400, 20 * lowest_modes);
        const auto modes = MatrixFree::solve_lowest_symmetric(
            3 * natoms_supe, apply_dynamical_matrix, mode_settings,
            project_translations);
        if (!modes.converged)
            throw std::runtime_error(fmt::format(
                "Matrix-free lowest-mode solver converged {}/{} modes after {} "
                "Hessian-vector products; maximum residual is {:.6e}",
                modes.converged_count, lowest_modes, modes.matvecs,
                modes.residual_norms.size() == 0
                    ? std::numeric_limits<double>::infinity()
                    : modes.residual_norms.maxCoeff()));

        ofstream ofseval("EIGENVALS");
        ofstream ofsevec("EIGENVECTORS");
        ofstream ofsesys("EIGENSYS");
        if (!ofseval || !ofsevec || !ofsesys)
            throw std::runtime_error("Cannot open matrix-free mode output files");
        fmt::print(ofseval,
            "# eigenvalue(eV*A^-2*a.m.u^-1) frequency(THz) energy(eV) frequency(cm^-1)\n");
        for (int mode = 0; mode < lowest_modes; ++mode) {
            const double eigenvalue = modes.eigenvalues(mode);
            const double sign = eigenvalue >= 0.0 ? 1.0 : -1.0;
            const double frequency = sign * freqconv * std::sqrt(std::abs(eigenvalue));
            fmt::print(ofseval,
                "{:>21.15f} {:>21.15f} {:>21.15f} {:>21.15f}\n",
                eigenvalue, frequency,
                sign * toeng * std::sqrt(std::abs(eigenvalue)),
                frequency * 33.356683);
            fmt::print(ofsevec, "{} {:>21.15f}\n", mode, eigenvalue);
            for (int atom_index = 0; atom_index < natoms_supe; ++atom_index)
                fmt::print(ofsevec,
                    "  {:>21.15f} {:>21.15f} {:>21.15f}\n",
                    modes.eigenvectors(3 * atom_index, mode),
                    modes.eigenvectors(3 * atom_index + 1, mode),
                    modes.eigenvectors(3 * atom_index + 2, mode));
            ofsevec << '\n';
        }
        ofsesys << modes.eigenvectors << std::endl;
        Message(fmt::format(
            "Wrote {} lowest internal Gamma modes using {} Hessian-vector products",
            lowest_modes, modes.matvecs));
        pot->getEFSOMP(supercell);
        iobj.write_one(supercell, "sup_debug_efs.xyz", "extxyz");
        return 0;
    }
    GPSettings hessian_settings;
    hessian_settings.Modify("method", hessian_method);
    hessian_settings.Modify("nudge_pos", std::to_string(hessian_nudge_pos));
    hessian_settings.Modify("cutoff", std::to_string(hessian_cutoff));
    ComputeHessian calculator(hessian_settings);
    calculator.symprec = symprec;
    calculator.compute_fc_blocks(supercell, hessian, pot);
    pot->getEFSOMP(supercell);
    iobj.write_one(supercell, "sup_debug_efs.xyz", "extxyz");

    // Check translational invariance through the tensor acoustic sum rule.
    if (check_symm)
    {
        const Eigen::MatrixXd residual = HessianUtils::acoustic_sum_rule_residual(
            HessianUtils::to_dense(hessian));
        fmt::print(std::cout, "Acoustic sum rule maximum residual: {:.10e} eV/A^2\n",
                   residual.cwiseAbs().maxCoeff());
        for (int row = 0; row < residual.rows(); ++row)
            fmt::print(std::cout, "{:5d}: {: .6e} {: .6e} {: .6e}\n", row,
                       residual(row, 0), residual(row, 1), residual(row, 2));
    }
    // output FORCE_CONSTANTS
    if (write_fc)
    {
        ofstream ofsfc("FORCE_CONSTANTS");
        if (!ofsfc)
            throw std::runtime_error("Cannot open FORCE_CONSTANTS for writing");
        if (natoms_orig == natoms_supe && symmreduce) {
            // 1. find primitive cell
            auto dataset = Symmetry3D::get_sym_dataset(atom, symprec);
            auto &mapping_to_primitive = dataset.mapping_to_primitive;
            std::set<int> id_primitives;
            std::set<int> primitives;
            for (int i = 0; i < mapping_to_primitive.size(); ++i) {
                if (id_primitives.count(mapping_to_primitive[i]) == 0) {
                    id_primitives.insert(mapping_to_primitive[i]);
                    primitives.insert(i);
                }
            }
            int nprimitive = primitives.size();
            fmt::print(ofsfc, " {} {}\n", nprimitive, natoms_orig);
            for (auto &i : primitives) {
                for (int j = 0; j < natoms_supe; ++j) {
                    fmt::print(ofsfc, "{} {}\n", i+1, j+1);
                    for (int a = 0; a < 3; ++a) {
                        fmt::print(ofsfc, " {:>21.15f} {:>21.15f} {:>21.15f}\n", hessian(i,j)(a,0), hessian(i,j)(a,1), hessian(i,j)(a,2));
                    }
                }
            }
        } else {
            if (symmreduce) {
                fmt::print(ofsfc, " {} {}\n", natoms_orig, natoms_supe);
                for (int i = 0; i < natoms_orig; ++i) {
                    for (int j = 0; j < natoms_supe; ++j) {
                        fmt::print(ofsfc, "{} {}\n", i+1, j+1);
                        for (int a = 0; a < 3; ++a) {
                            fmt::print(ofsfc, " {:>21.15f} {:>21.15f} {:>21.15f}\n", hessian(i,j)(a,0), hessian(i,j)(a,1), hessian(i,j)(a,2));
                        }
                    }
                }
            } else {
                fmt::print(ofsfc, " {} {}\n", natoms_supe, natoms_supe);
                for (int i = 0; i < natoms_supe; ++i) {
                    for (int j = 0; j < natoms_supe; ++j) {
                        fmt::print(ofsfc, "{} {}\n", i+1, j+1);
                        for (int a = 0; a < 3; ++a) {
                            fmt::print(ofsfc, " {:>21.15f} {:>21.15f} {:>21.15f}\n", hessian(i,j)(a,0), hessian(i,j)(a,1), hessian(i,j)(a,2));
                        }
                    }
                }
            }
        }
        ofsfc.close();
        if (!ofsfc)
            throw std::runtime_error("Failed to write FORCE_CONSTANTS");
    }
    return 0;
}
