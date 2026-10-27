#pragma once

#include "les.h"
#include "native_collectives.h"

namespace Potentials {

// All result arrays contain owned rows only, [0,nlocal). Energy and virial
// are rank-local contributions and must be reduced by the caller. Kernels
// never evaluate replicated full configurations or write ghost outputs.
LES::ChannelEvaluation native_les_distributed(
    const LES &, const Atoms &, const Eigen::MatrixXd &owned_latents, int nlocal,
    const NativeCollectives &, LES::Workspace *workspace = nullptr);

} // namespace Potentials
