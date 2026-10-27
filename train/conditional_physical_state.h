#pragma once
#include "conditional_linear_persistence.h"

namespace Training {
// Packed SOG coordinates use atan/tan. Restoring a probe must preserve the
// original physical bandwidths exactly, rather than round-trip that chart.
inline std::function<void()> conditional_bandwidth_restore(Potentials::AnyPotential &potential) {
    using namespace Potentials;
    if(typeid(potential)==typeid(PotMTP_LES)) {
        auto &p=static_cast<PotMTP_LES&>(potential);const auto widths=p.model_les.sog_log_bandwidths;
        return [&p,widths]{std::copy(widths.begin(),widths.end(),p.model_les.sog_log_bandwidths.begin());};
    }
#define IMRMLP_CONDITIONAL_WIDTH_RESTORE(Model) \
    if(typeid(potential)==typeid(PotMTP_M<Model>)) { \
        auto &p=static_cast<PotMTP_M<Model>&>(potential);const auto widths=p.model_les.sog_log_bandwidths; \
        return [&p,widths]{std::copy(widths.begin(),widths.end(),p.model_les.sog_log_bandwidths.begin());}; \
    }
    IMRMLP_CONDITIONAL_WIDTH_RESTORE(m_model_mtpli)
    IMRMLP_CONDITIONAL_WIDTH_RESTORE(m_model_mtpq2)
    IMRMLP_CONDITIONAL_WIDTH_RESTORE(m_model_mtpfs)
    IMRMLP_CONDITIONAL_WIDTH_RESTORE(m_model_nn1)
    IMRMLP_CONDITIONAL_WIDTH_RESTORE(m_model_nn2)
#undef IMRMLP_CONDITIONAL_WIDTH_RESTORE
    return []{};
}
inline std::function<void()> conditional_physical_restore(Potentials::AnyPotential &potential) {
    Eigen::VectorXd coefficients(potential.get_NL_coeffs_count());potential.nl_get_coeffs(coefficients.data());
    auto bandwidths=conditional_bandwidth_restore(potential);
    return [&potential,coefficients,bandwidths] {
        if(potential.get_NL_coeffs_count()!=coefficients.size())
            GPERROR("Potential parameter layout changed during a profile probe; reconstruct the objective");
        potential.nl_set_coeffs(coefficients.data());bandwidths();
    };
}
}
