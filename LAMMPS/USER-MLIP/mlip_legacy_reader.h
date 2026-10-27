// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include <istream>
#include <stdexcept>
#include <string>

namespace MLIPLegacyReader {

// Read the header segment immediately after species_count. The two legacy
// LAMMPS adapters translate these exceptions through Error::all. Keep this
// parser independent of LAMMPS so the format contract is directly testable.
inline std::string read_radial_basis_type(std::istream &input) {
    auto next = [&]() {
        std::string token;
        if (!(input >> token))
            throw std::runtime_error("Unexpected end of MTP header before radial_basis_type");
        if (token == "yuk_coeffs" || token == "yuk3_coeffs")
            throw std::runtime_error("Yukawa coefficient sections (yuk_coeffs/yuk3_coeffs) have been removed and are unsupported");
        return token;
    };

    std::string token = next();
    if (token == "residual_cosplus")
        throw std::runtime_error("Error: this version of interface does not support residuals");
    if (token == "potential_tag") {
        std::string metadata;
        std::getline(input, metadata);
        token = next();
    }
    if (token != "radial_basis_type")
        throw std::runtime_error("Error reading .mtp file: expected radial_basis_type");
    char assignment = 0;
    std::string basis;
    if (!(input >> assignment >> basis) || assignment != '=')
        throw std::runtime_error("Error reading .mtp file: expected radial_basis_type = BASIS");
    return basis;
}

} // namespace MLIPLegacyReader
