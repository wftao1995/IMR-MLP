// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "quick_mtp.h"
#include "fmt/format.h"
#include <sstream>
namespace Potentials {
PotMTP * quick_mtp_3b_s14_2f0ee25e(qmtp_config *p) {
    int species_count = 118;
    double rmin = 0;
    double rmax = 5;
    int rbsize = 4;
    int seed = 7538;
    if (p) {
        species_count = p->species_count;
        rmin = p->rmin;
        rmax = p->rmax;
        rbsize = p->rbsize;
        seed = p->seed;
    }
    std::string mtp_template = "MTP\nversion = 1.1.0\npotential_name = MTP1m\nscaling = 1.0\n" + 
                fmt::format("species_count = {}\n", species_count) +
                "potential_tag = \n"
                "radial_basis_type = RBChebyshev\n" +
                fmt::format("\t\tmin_dist = {}\n", rmin) +
                fmt::format("\t\tmax_dist = {}\n", rmax) +
                fmt::format("\t\tradial_basis_size = {}\n", rbsize) +
                "radial_funcs_count = 2\n"
                "alpha_moments_count = 52\n"
                "alpha_index_basic_count = 40\n"
                "alpha_index_basic = {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 1, 0, 0}, {0, 0, 0, 2}, {0, 0, 1, 1}, {0, 0, 2, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 2, 0, 0}, {1, 0, 0, 2}, {1, 0, 1, 1}, {1, 0, 2, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 2, 0, 0}, {0, 0, 0, 3}, {0, 0, 1, 2}, {0, 0, 2, 1}, {0, 0, 3, 0}, {0, 1, 0, 2}, {0, 1, 1, 1}, {0, 1, 2, 0}, {0, 2, 0, 1}, {0, 2, 1, 0}, {0, 3, 0, 0}, {1, 0, 0, 3}, {1, 0, 1, 2}, {1, 0, 2, 1}, {1, 0, 3, 0}, {1, 1, 0, 2}, {1, 1, 1, 1}, {1, 1, 2, 0}, {1, 2, 0, 1}, {1, 2, 1, 0}, {1, 3, 0, 0}}\n"
                "alpha_index_times_count = 60\n"
                "alpha_index_times = {{0, 0, 1, 40}, {2, 2, 1, 41}, {3, 3, 1, 41}, {4, 4, 1, 41}, {8, 8, 1, 42}, {9, 9, 2, 42}, {10, 10, 1, 42}, {11, 11, 2, 42}, {12, 12, 2, 42}, {13, 13, 1, 42}, {20, 20, 1, 43}, {21, 21, 3, 43}, {22, 22, 3, 43}, {23, 23, 1, 43}, {24, 24, 3, 43}, {25, 25, 6, 43}, {26, 26, 3, 43}, {27, 27, 3, 43}, {28, 28, 3, 43}, {29, 29, 1, 43}, {0, 1, 1, 44}, {2, 5, 1, 45}, {3, 6, 1, 45}, {4, 7, 1, 45}, {8, 14, 1, 46}, {9, 15, 2, 46}, {10, 16, 1, 46}, {11, 17, 2, 46}, {12, 18, 2, 46}, {13, 19, 1, 46}, {20, 30, 1, 47}, {21, 31, 3, 47}, {22, 32, 3, 47}, {23, 33, 1, 47}, {24, 34, 3, 47}, {25, 35, 6, 47}, {26, 36, 3, 47}, {27, 37, 3, 47}, {28, 38, 3, 47}, {29, 39, 1, 47}, {1, 1, 1, 48}, {5, 5, 1, 49}, {6, 6, 1, 49}, {7, 7, 1, 49}, {14, 14, 1, 50}, {15, 15, 2, 50}, {16, 16, 1, 50}, {17, 17, 2, 50}, {18, 18, 2, 50}, {19, 19, 1, 50}, {30, 30, 1, 51}, {31, 31, 3, 51}, {32, 32, 3, 51}, {33, 33, 1, 51}, {34, 34, 3, 51}, {35, 35, 6, 51}, {36, 36, 3, 51}, {37, 37, 3, 51}, {38, 38, 3, 51}, {39, 39, 1, 51}}\n"
                "alpha_scalar_moments = 14\n"
                "alpha_moment_mapping = {0, 1, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51}\n";
    std::stringstream iss(mtp_template);
    Potentials::PotMTP *pot = new Potentials::PotMTP();
    pot->Load(iss);
    pot->empty = false;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (int i = 0; i < pot->reg_coeff_size; ++i) {
        pot->regression_coeffs[i] = uniform(rng);
    }
    pot->Orthogonalize();
    pot->omp_init();
    return pot;
}

PotMTP* quick_mtp_3b_s11_f2861f4f(qmtp_config* p)
{
    int species_count = 118;
    double rmin = 0;
    double rmax = 5.2;
    int rbsize = 4;
    int seed = 9084;
    if (p) {
        species_count = p->species_count;
        rmin = p->rmin;
        rmax = p->rmax;
        rbsize = p->rbsize;
        seed = p->seed;
    }
    std::string mtp_template = "MTP\nversion = 1.1.0\npotential_name = MTP1m\nscaling = 1.0\n" + 
                fmt::format("species_count = {}\n", species_count) +
                "potential_tag = \n"
                "radial_basis_type = RBChebyshev\n" +
                fmt::format("\t\tmin_dist = {}\n", rmin) +
                fmt::format("\t\tmax_dist = {}\n", rmax) +
                fmt::format("\t\tradial_basis_size = {}\n", rbsize) +
                "radial_funcs_count = 2\n"
                "alpha_moments_count = 29\n"
                "alpha_index_basic_count = 20\n"
                "alpha_index_basic = {{0, 0, 0, 0}, {1, 0, 0, 0}, {0, 0, 0, 1}, {0, 0, 1, 0}, {0, 1, 0, 0}, {0, 0, 0, 2}, {0, 0, 1, 1}, {0, 0, 2, 0}, {0, 1, 0, 1}, {0, 1, 1, 0}, {0, 2, 0, 0}, {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 1, 0, 0}, {1, 0, 0, 2}, {1, 0, 1, 1}, {1, 0, 2, 0}, {1, 1, 0, 1}, {1, 1, 1, 0}, {1, 2, 0, 0}}\n"
                "alpha_index_times_count = 30\n"
                "alpha_index_times = {{0, 0, 1, 20}, {2, 2, 1, 21}, {3, 3, 1, 21}, {4, 4, 1, 21}, {5, 5, 1, 22}, {6, 6, 2, 22}, {7, 7, 1, 22}, {8, 8, 2, 22}, {9, 9, 2, 22}, {10, 10, 1, 22}, {0, 1, 1, 23}, {2, 11, 1, 24}, {3, 12, 1, 24}, {4, 13, 1, 24}, {5, 14, 1, 25}, {6, 15, 2, 25}, {7, 16, 1, 25}, {8, 17, 2, 25}, {9, 18, 2, 25}, {10, 19, 1, 25}, {1, 1, 1, 26}, {11, 11, 1, 27}, {12, 12, 1, 27}, {13, 13, 1, 27}, {14, 14, 1, 28}, {15, 15, 2, 28}, {16, 16, 1, 28}, {17, 17, 2, 28}, {18, 18, 2, 28}, {19, 19, 1, 28}}\n"
                "alpha_scalar_moments = 11\n"
                "alpha_moment_mapping = {0, 1, 20, 21, 22, 23, 24, 25, 26, 27, 28}\n";
    std::stringstream iss(mtp_template);
    Potentials::PotMTP *pot = new Potentials::PotMTP();
    pot->Load(iss);
    pot->empty = false;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> uniform(-1.0, 1.0);
    for (int i = 0; i < pot->reg_coeff_size; ++i) {
        pot->regression_coeffs[i] = uniform(rng);
    }
    pot->Orthogonalize();
    pot->omp_init();
    return pot;
}
}