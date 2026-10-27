// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"
#include "base/structures.h"
#include "potentials/potential.h"
#include <string>

class ComputeHessian : public ConnectSettings {
public:
    enum class HessianMethod {
        Auto,
        Analytic,
        ForceFD,
        EnergyFD,
        Hybrid
    };

    enum class HessianObjective {
        Energy,
        Enthalpy
    };

private:
    double nudgepos  = 0.001;
    double nudgecell = 0.001;
    double nudgefrac = 0.001;
    double cutoff = 0.0;
    std::string method_name = "auto";
    std::string objective_name = "energy";

    HessianMethod parse_method() const;
    HessianObjective parse_objective() const;
    HessianMethod resolve_atomic_method(const Potentials::AnyPotential *pot, const Atoms &atom) const;

public:
    bool request_fc = true;
    bool request_cell = false;
    bool cell_strain = false;       // 9 raw cell DOFs or 6 Voigt strain DOFs
    bool symm_strain = false;       // project the strain block using point symmetry
    bool fractional_pos = false;    // use fractional atomic coordinates as DOFs
    bool request_response_pos_cell = true;
    bool use_symmetry = true;       // apply symmetry projection where available
    bool engineering_shear = false; // use gamma_ij = 2 epsilon_ij
    double symprec = 1e-4;
    double pressure = 0.0;          // eV/A^3, used for the enthalpy objective
    HessianMethod method = HessianMethod::Auto;
    HessianObjective objective = HessianObjective::Energy;

    // Compute a full generalized-coordinate Hessian.
    void compute(Atoms &atom, Eigen::MatrixXd &hessian, Potentials::AnyPotential *pot);

    void compute_fc(Atoms &atom, Eigen::MatrixXd &hessian, Potentials::AnyPotential *pot);
    // Cartesian atom-major H*v. Auto uses analytical HVP when available,
    // otherwise two force evaluations with maximum per-atom motion nudge_pos.
    void compute_hvp(const Atoms &atom, const Eigen::VectorXd &direction,
                     Eigen::VectorXd &product, Potentials::AnyPotential *pot);

    // Block-storage counterpart used by force-constant applications. Analytic
    // backends fill blocks directly; numerical backends reuse the dense path.
    void compute_fc_blocks(Atoms &atom, Potentials::HessianMatrix &hessian,
                           Potentials::AnyPotential *pot);

    ComputeHessian() = default;
    ~ComputeHessian() = default;

    explicit ComputeHessian(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
        method = parse_method();
        objective = parse_objective();
    }

    void set_cutoff(double r) {
        cutoff = r;
    }

    double get_cutoff() const { return cutoff; }

    void set_method(const std::string &name) {
        method_name = name;
        method = parse_method();
    }

    void set_objective(const std::string &name) {
        objective_name = name;
        objective = parse_objective();
    }

    void InitSettings() {
        Connect(nudgepos, "nudge_pos");
        Connect(nudgecell, "nudge_cell");
        Connect(nudgefrac, "nudge_frac");
        Connect(cutoff, "cutoff");
        Connect(request_cell, "req_cell");
        Connect(request_fc, "req_fc");
        Connect(request_response_pos_cell, "req_fc_cell");
        Connect(cell_strain, "cell_strain");
        Connect(symm_strain, "symm_strain");
        Connect(fractional_pos, "frac_pos");
        Connect(use_symmetry, "use_symmetry");
        Connect(engineering_shear, "engineering_shear");
        Connect(symprec, "symprec");
        Connect(pressure, "pressure");
        Connect(method_name, "method");
        Connect(objective_name, "objective");
    }
};
