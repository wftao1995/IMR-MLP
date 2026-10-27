#pragma once

#include <iosfwd>
#include <string>
#include <vector>

class Atoms;
class RelaxationDrvV2;

namespace AppsRelax {
// Validate selective-dynamics masks before modifying atoms. Configuration IDs
// are one-based; atom IDs are zero-based. Unlisted atoms remain fully movable.
void read_masks(std::istream &input, std::vector<Atoms> &atoms);

// Broadcast three binary flags to the atomic optimization coordinates used by
// FilterSelDyn. Validate conflicts before changing flags or masks; cell masks
// are preserved. Call after settings and ISIF have configured the driver.
void apply_position_mask(const std::string &text, std::vector<Atoms> &atoms,
                         RelaxationDrvV2 &driver);
}
