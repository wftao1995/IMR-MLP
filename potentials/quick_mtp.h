// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "potential_mtp.h"

namespace Potentials {
struct qmtp_config {
    int species_count = 118;
    double rmin = 0;
    double rmax = 5;
    int rbsize = 4;
    int seed = 7538;
};
// only for some special use, i.e. find equiv atoms
PotMTP* quick_mtp_3b_s14_2f0ee25e(qmtp_config *p = nullptr);
// this is a super-quick 3b potential
PotMTP* quick_mtp_3b_s11_f2861f4f(qmtp_config *p = nullptr);
}
