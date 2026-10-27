// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_mindist.h"
#include "potential_settings.h"
#include "utils/gputils.h"
#include <cmath>
#include <iomanip>
#include <limits>

using namespace Potentials;

PotMDIST::PotMDIST(GPSettings &settings) {
    initialize(settings);
}

void PotMDIST::initialize(GPSettings &settings) {
    cutoff = 5.0;
    power = 2.0;
    factor = 1.0;
    InitSettings();
    ApplySettings(settings);
    cutoffsq = cutoff * cutoff;
    cutoffs.resize(1, 1);
    cutoffsqs.resize(1, 1);
    r_mins.resize(1, 1);
    cutoffs(0, 0) = cutoff;
    cutoffsqs(0, 0) = cutoffsq;
    r_mins(0, 0) = 0.0;
    nl_inited = true;
}

void PotMDIST::InitSettings() {
    Connect(cutoff, "cutoff", "Minimum-distance cutoff radius");
    Connect(factor, "factor", "Energy prefactor");
    Connect(power,  "power",  "Energy exponent (default 2)");
}

void PotMDIST::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        for (int jj = 0; jj < nbh.nbhcount_; ++jj) {
            double r  = nbh.distances_[jj];
            if (r >= cutoff) continue;

            double dr = r - cutoff;  // negative
            double E_raw   = factor * std::pow(dr, power);
            double dE_dr   = factor * power * std::pow(dr, power - 1.0);

            // 0.5: each pair is visited twice in symmetric neighbour lists
            atom.energy_ += 0.5 * E_raw;
            atom.site_energies_[ii] += 0.5 * E_raw;

            // Force on the center: +0.5 * dE/dr * neighbor direction.
            double f_over_r = 0.5 * dE_dr / r;
            Eigen::Vector3d fce = f_over_r * nbh.pos_[jj];
            atom.forces_[ii] += fce;
            atom.forces_[nbh.indices_[jj]] -= fce;

            // Virial stress
            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    atom.stress_(a, b) -= fce(a) * nbh.pos_[jj](b);
        }
    }
}

void PotMDIST::Load(std::istream &is) {
    auto settings = read_potential_settings(is);
    initialize(settings);
}

void PotMDIST::Save(std::ostream &os) {
    os << std::setprecision(std::numeric_limits<double>::max_digits10);
    os << "pot:pottype mindist\n";
    os << "pot:power " << power << "\n";
    os << "pot:factor " << factor << "\n";
    os << "pot:cutoff " << cutoff << "\n";
}
