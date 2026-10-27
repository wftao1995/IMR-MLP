// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_boo.h"
#include "base/anyneighborhoods.h"
#include "mathextra/angular_functions.h"
#include "tools/tools.h"
#include "base/neighborhoodKD.h"
#include <cmath>
#include <complex>
#include <string>

void ComputeBondOrder::compute(Atoms &atom)
{
    NeighborhoodKD nbhkd(&atom, r_cut);
    AnyNeighborhoods nbhs;
    nbhs.resize(atom.size());
    for (int i = 0; i < atom.size(); ++i) {
        if (use_nnn) {
            nbhs[i] = nbhkd.query(i, nnn);
        }
        else {
            nbhs[i] = nbhkd.query(i, r_cut);
            if (min_nn > 0 && nbhs[i].nbhcount_ < min_nn)
                nbhs[i] = nbhkd.query(i, min_nn);
            else if (max_nn > 0 && nbhs[i].nbhcount_ > max_nn)
                nbhs[i] = nbhkd.query(i, max_nn);
        }
    }
    compute(atom, nbhs);
}

void ComputeBondOrder::compute(Atoms &atom, const AnyNeighborhoods &nbhs)
{
    std::vector<double> Ql, Wl; // non-average, non-coarsed version
    Ql.resize(maxl_/2);
    Wl.resize(maxl_/2);
    // atom.local_attribs_v_["bopQ"].resize(Ql.size(), atom.size());
    // atom.local_attribs_v_["bopW"].resize(Wl.size(), atom.size());
    atom.alloc_lvd("bopQ", Ql.size());
    atom.alloc_lvd("bopW", Wl.size());
    std::vector<std::vector<std::complex<double>>> qlms(atom.size());
    int isolated = 0;
    for (int iatom = 0; iatom < atom.size(); ++iatom) {
        std::ranges::fill(Qlmc, std::complex<double>(0, 0));
        std::ranges::fill(Ql, 0);
        std::ranges::fill(Wl, 0);
        const Neighborhood &nbh = nbhs[iatom];
        if (nbh.nbhcount_ == 0) ++isolated;
        // calc sph
        for (int i = 0; i < nbh.nbhcount_; ++i) {
            csph.computeYCarSafe(nbh.pos_[i][0], nbh.pos_[i][1], nbh.pos_[i][2], nbh.distances_[i]);
            for (int l = 2; l <= maxl_; l+= 2) {
                Qlmc[get_idx(l, 0)] += csph.get(l, 0);
                for (int m = 1; m <= l; ++m) {
                    Qlmc[get_idx(l, m)] += csph.get(l, m);
                    Qlmc[get_idx(l, -m)] += csph.get(l, -m);
                }
            }
        }
        // calc avg 
        double factor = nbh.nbhcount_ > 0 ? 1.0 / nbh.nbhcount_ : 0.0;
        for (int l = 2; l <= maxl_; l += 2) {
            for (int m = -l; m <= l; ++m){
                Qlmc[get_idx(l, m)] *= factor;
            }
        }
        if (average) {
            qlms[iatom] = Qlmc;
        }
        // calc Ql, Wl
        for (int l = 2; l <= maxl_; l += 2) {
            Ql[l/2-1] += Qlmc[get_idx(l, 0)].real() * Qlmc[get_idx(l, 0)].real() +
                         Qlmc[get_idx(l, 0)].imag() * Qlmc[get_idx(l, 0)].imag();
            for (int m = 1; m <= l; ++m) {
                Ql[l/2-1] += Qlmc[get_idx(l, m)].real() * Qlmc[get_idx(l, m)].real() +
                             Qlmc[get_idx(l, m)].imag() * Qlmc[get_idx(l, m)].imag() +
                             Qlmc[get_idx(l, -m)].real() * Qlmc[get_idx(l, -m)].real() +
                             Qlmc[get_idx(l, -m)].imag() * Qlmc[get_idx(l, -m)].imag();
            }
            for (int m1 = -l; m1 <= l; ++m1) {
                for (int m2 = -l; m2 <= l; ++m2) {
                    int m3 = -m1 - m2;
                    if (m3 > l || m3 < -l)
                        continue;
                    double w3jm = AngularFuncs::Wigner3jIntCache(l, l, l, m1, m2, m3);
                    Wl[l/2-1] += w3jm * (Qlmc[get_idx(l, m1)] * Qlmc[get_idx(l, m2)] * Qlmc[get_idx(l, m3)]).real();
                }
            }
            // postprocess Ql, Wl
            Wl[l/2-1] = Ql[l/2-1] > 0 ? pow(Ql[l/2-1], -1.5) * Wl[l/2-1] : 0;
            Ql[l/2-1] = sqrt(4. * M_PI / (2. * l + 1.0) * Ql[l/2-1]); 
        }
        // store results
        atom.local_attribs_v_["bopQ"].col(iatom) = Eigen::Map<Eigen::VectorXd>(Ql.data(), Ql.size());
        atom.local_attribs_v_["bopW"].col(iatom) = Eigen::Map<Eigen::VectorXd>(Wl.data(), Wl.size());
    }
    if (average) {
        // atom.local_attribs_v_["bopQ_A"].resize(Ql.size(), atom.size());
        // atom.local_attribs_v_["bopW_A"].resize(Wl.size(), atom.size());
        atom.alloc_lvd("bopQ_A", Ql.size());
        atom.alloc_lvd("bopW_A", Wl.size());
        for (int iatom = 0; iatom < atom.size(); ++iatom) {
            std::vector<std::complex<double>> qlm(qlmsize, std::complex<double>(0, 0));
            std::vector<double> Qlavg(maxl_ / 2, 0);
            std::vector<double> Wlavg(maxl_ / 2, 0);
            const Neighborhood &nbh = nbhs[iatom];
            double factor = 1.0 / (nbh.nbhcount_ + 1.0);
            for (int l = 2; l <= maxl_; l += 2)
                for (int m = -l; m <= l; ++m)
                    qlm[get_idx(l, m)] = qlms[iatom][get_idx(l, m)] * factor;
            for (int inb = 0; inb < nbh.nbhcount_; ++inb) {
                for (int l = 2; l <= maxl_; l += 2) {
                    qlm[get_idx(l, 0)] += qlms[nbh.indices_[inb]][get_idx(l, 0)] * factor;
                    for (int m = 1; m <= l; ++m) {
                        qlm[get_idx(l, m)] += qlms[nbh.indices_[inb]][get_idx(l, m)] * factor;
                        qlm[get_idx(l, -m)] += qlms[nbh.indices_[inb]][get_idx(l, -m)] * factor;
                    }
                }
            }
            for (int l = 2; l <= maxl_; l += 2) {
                Qlavg[l/2-1] += qlm[get_idx(l, 0)].real() * qlm[get_idx(l, 0)].real() +
                                qlm[get_idx(l, 0)].imag() * qlm[get_idx(l, 0)].imag();
                for (int m = 1; m <= l; ++m) {
                    Qlavg[l/2-1] += qlm[get_idx(l, m)].real() * qlm[get_idx(l, m)].real() +
                                 qlm[get_idx(l, m)].imag() * qlm[get_idx(l, m)].imag() +
                                 qlm[get_idx(l, -m)].real() * qlm[get_idx(l, -m)].real() +
                                 qlm[get_idx(l, -m)].imag() * qlm[get_idx(l, -m)].imag();
                }
                for (int m1 = -l; m1 <= l; ++m1) {
                    for (int m2 = -l; m2 <= l; ++m2) {
                        int m3 = -m1 - m2;
                        if (m3 > l || m3 < -l)
                            continue;
                        double w3jm = AngularFuncs::Wigner3jIntCache(l, l, l, m1, m2, m3);
                        Wlavg[l/2-1] += w3jm * (qlm[get_idx(l, m1)] * qlm[get_idx(l, m2)] * qlm[get_idx(l, m3)]).real();
                    }
                }
                // postprocess Ql, Wl
                Wlavg[l/2-1] = Qlavg[l/2-1] > 0 ? pow(Qlavg[l/2-1], -1.5) * Wlavg[l/2-1] : 0;
                Qlavg[l/2-1] = sqrt(4. * M_PI / (2. * l + 1.0) * Qlavg[l/2-1]); 
            }
            atom.local_attribs_v_["bopQ_A"].col(iatom) = Eigen::Map<Eigen::VectorXd>(Qlavg.data(), Qlavg.size());
            atom.local_attribs_v_["bopW_A"].col(iatom) = Eigen::Map<Eigen::VectorXd>(Wlavg.data(), Wlavg.size());
        }
    }
    if (corase) {
        // atom.local_attribs_v_["bopQ_C"] = atom.local_attribs_v_["bopQ"];
        // atom.local_attribs_v_["bopW_C"] = atom.local_attribs_v_["bopW"];
        atom.alloc_lvd("bopQ_C", Ql.size());
        atom.alloc_lvd("bopW_C", Wl.size());
        atom.local_attribs_v_["bopQ_C"].setZero();
        atom.local_attribs_v_["bopW_C"].setZero();
        for (int iatom = 0; iatom < atom.size(); ++iatom) {
            double factor = nbhs[iatom].nbhcount_ > 0 ? 1.0 / nbhs[iatom].nbhcount_ : 0.0;
            for (int inb = 0; inb < nbhs[iatom].nbhcount_; ++inb) {
                atom.local_attribs_v_["bopQ_C"].col(iatom) += 
                        atom.local_attribs_v_["bopQ"].col(nbhs[iatom].indices_[inb]) * factor;
                atom.local_attribs_v_["bopW_C"].col(iatom) += 
                        atom.local_attribs_v_["bopW"].col(nbhs[iatom].indices_[inb]) * factor;
            }
        }
    }
    if (isolated > 0)
        Warning("Bond-order parameters: " + std::to_string(isolated) +
                " atoms have no neighbors; their Q/W values are set to zero");
}
