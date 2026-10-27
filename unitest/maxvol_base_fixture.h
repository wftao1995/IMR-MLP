#pragma once
#include "potentials/potential_mtp.h"
#include <sstream>

// Small deterministic many-body MTP: includes scalar-density squared and a
// vector contraction, asymmetric pair coefficients and species-dependent radii.
inline void load_swap_fixture(Potentials::PotMTP &potential, int species = 3, bool split = true) {
    std::ostringstream out;
    out << "MTP\nversion = 1.1.0\npotential_name = MTP1m\nscaling = 1\n"
        << "species_count = " << species << "\npotential_tag =\n";
    if (split) {
        out << "radial_basis_split = 1\n";
        for (int i = 0; i < species; ++i)
            for (int j = 0; j < species; ++j)
                out << "pair " << i << ' ' << j << "\nradial_basis_type = RBChebyshev\n"
                    << "min_dist = 0.5\nmax_dist = " << 4.0 + 0.4 * (i + j)
                    << "\nradial_basis_size = 2\n";
    } else out << "radial_basis_type = RBChebyshev\nmin_dist = 0.5\nmax_dist = 5\nradial_basis_size = 2\n";
    out << "radial_funcs_count = 1\nradial_coeffs\n";
    for (int i = 0; i < species; ++i)
        for (int j = 0; j < species; ++j)
            out << i << '-' << j << "\n{" << 0.03 * (i + 1) / (j + 1) << ", " << 0.01 * (i - j + 1) << "}\n";
    out << "alpha_moments_count = 6\nalpha_index_basic_count = 4\n"
        << "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}\n"
        << "alpha_index_times_count = 4\n"
        << "alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}, {0, 0, 1, 5}}\n"
        << "alpha_scalar_moments = 3\nalpha_moment_mapping = {0, 4, 5}\nspecies_coeffs = {";
    for (int i = 0; i < species; ++i) out << (i ? ", " : "") << -0.2 * (i + 1);
    out << "}\nmoment_coeffs = {-0.15, 0.004, 0.005}\n";
    std::istringstream input(out.str());
    potential.Load(input);
}
