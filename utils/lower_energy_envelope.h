#pragma once

#include <cstddef>
#include <memory>
#include <vector>

// Lower envelope under convex mixing of the supplied coordinates. Only
// composition coordinates give this construction its usual phase-hull meaning.
class LowerEnergyEnvelope {
public:
    using Coordinates = std::vector<double>;
    LowerEnergyEnvelope(const std::vector<Coordinates> &coordinates,
                        const std::vector<double> &energies);
    ~LowerEnergyEnvelope();
    LowerEnergyEnvelope(const LowerEnergyEnvelope &) = delete;
    LowerEnergyEnvelope &operator=(const LowerEnergyEnvelope &) = delete;

    // Throws std::out_of_range outside the convex reference domain, including
    // outside its affine span. Energies and results have the caller's units.
    double energy_at(const Coordinates &coordinates);
    std::size_t affine_rank() const;
    std::size_t reference_size() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
