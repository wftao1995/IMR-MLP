#pragma once

#include "runtime/mtp/gpu-direct/gpu_backend.h"
#include "native_long_range.h"

namespace Potentials {

// Same ownership and collective contract as native_long_range.h. Numerical
// long-range evaluation runs on the selected GPU; no CPU fallback is used.
LES::ChannelEvaluation native_les_distributed_gpu(
    const LES &, const Atoms &, const Eigen::MatrixXd &owned_latents, int nlocal,
    const NativeCollectives &, int device, LES::Workspace *workspace = nullptr);

} // namespace Potentials
