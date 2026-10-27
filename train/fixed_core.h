#pragma once
#include "hvp_fit.h"

namespace Training {
// Returned structures own their residual labels; reference labels are unchanged.
std::vector<Atoms> fixed_core_residuals(Potentials::AnyPotential &core,
    const std::vector<Atoms> &references, double hvp_step=1e-4, double hessian_step=1e-4);
void validate_fixed_core(Potentials::AnyPotential &residual, Potentials::AnyPotential &core);
void save_fixed_core_hybrid(Potentials::AnyPotential &residual, Potentials::AnyPotential &core,
    const std::string &path, const Atoms *reference=nullptr);
}
