#pragma once
#include <string>

namespace Potentials {
// Keep the model text intact, strip MVS, and freeze native PolyB matrices.
// Re-exporting a deployed model preserves and validates its existing cache.
std::string export_standalone_radial(const std::string &text);
}
