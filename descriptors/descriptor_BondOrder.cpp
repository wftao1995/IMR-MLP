// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "descriptor_BondOrder.h"
#include "tools/tools.h"
#include <limits>
#ifdef USE_OMP
#include <omp.h>
#endif

void DescriptorBondOrder::compute(const Neighborhood &nbh, std::vector<double> &Ql, std::vector<double> &Wl)
{
    // Wigner 3j coefficients couple complex spherical harmonics.
    compute_ref(nbh, Ql, Wl);
    // Preserve the real-harmonic coefficients exposed by get_qlm().
    for (int l = 0; l <= maxl_; ++l) {
        Qlm[get_idx(l, 0)] = Qlmc[get_idx(l, 0)].real();
        for (int m = 1; m <= l; ++m) {
            Qlm[get_idx(l, m)] = M_SQRT2 * Qlmc[get_idx(l, m)].real();
            Qlm[get_idx(l, -m)] = M_SQRT2 * Qlmc[get_idx(l, m)].imag();
        }
    }
}

void DescriptorBondOrder::compute_ref(const Neighborhood &nbh, std::vector<double> &Ql, std::vector<double> &Wl)
{
    Ql.resize(floor(maxl_/2.0) + 1);
    Wl.resize(floor(maxl_/2.0) + 1);
    for (int i = 0; i < Ql.size(); ++i)
    {
        Ql[i] = 0;
        Wl[i] = 0;
    }
    //Qlmc.resize(qmlsize());
    for (int i = 0; i < Qlmc.size(); ++i)
    {
        Qlmc[i] = std::complex<double>(0,0);
    }
    for (int i = 0; i < nbh.nbhcount_; ++i)
    {
        csph.computeYCarSafe(nbh.pos_[i][0], nbh.pos_[i][1], nbh.pos_[i][2], nbh.distances_[i]);
        for (int l = 0; l <= maxl_; ++l) {
            Qlmc[get_idx(l, 0)] += csph.get(l, 0);
        }
        for (int l = 0; l <= maxl_; ++l)
        {
            for (int m = 1; m <= l; ++m)
            {
                Qlmc[get_idx(l, m)] += csph.get(l, m);
                Qlmc[get_idx(l, -m)] += csph.get(l, -m);
            }
        }
    }
    if (nbh.nbhcount_ > 0) {
        for (int i = 0; i < Qlmc.size(); ++i)
        {
            Qlmc[i] /= nbh.nbhcount_;
        }
    }
    else {
        for (int i = 0; i < Qlmc.size(); ++i)
            Qlmc[i] = 0;
    }
    // compute Ql
    for (int l = 0; l <= maxl_; l += 2)
    {
        Ql[l/2] += (std::conj(Qlmc[get_idx(l, 0)]) * Qlmc[get_idx(l, 0)]).real();
        for (int m = 1; m <= l; ++m)
            Ql[l/2] += (std::conj(Qlmc[get_idx(l, m)]) * Qlmc[get_idx(l, m)] + std::conj(Qlmc[get_idx(l, -m)]) * Qlmc[get_idx(l, -m)]).real();
        // Ql[i] = sqrt(4*M_PI/(2*i + 1) * Ql[i]);
    }
    // compute Wl
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int l = 0; l <= maxl_; l += 2)
    {
        for (int m1 = -l; m1 <= l; ++m1)
        {
            for (int m2 = -l; m2 <= l; ++m2)
            {
                int m3 = -m1 - m2;
                if (m3 > l || m3 < -l)
                    continue;
                double w3jm = AngularFuncs::Wigner3jIntCache(l, l, l, m1, m2, m3);
                Wl[l/2] += w3jm * (Qlmc[get_idx(l, m1)] * Qlmc[get_idx(l, m2)] * Qlmc[get_idx(l, m3)]).real();
            }
        }
    }
    // normalize
    for (int l = 0; l <= maxl_; l += 2)
    {
        if (!normalize_w_outer)
            Wl[l/2] = Ql[l/2] > 0.0 ? pow(Ql[l/2], -3. / 2.) * Wl[l/2] : 0.0;
        Ql[l/2] = sqrt(4 * M_PI / (2 * l + 1) * Ql[l/2]); // order is correct, see paper carefully
    }
}

void DescriptorBondOrder::compute_ref_pairs(const Neighborhood &nbh, const double rcut, std::vector<double> &Ql, std::vector<double> &Wl)
{
    Ql.resize(floor(maxl_/2.0) + 1);
    Wl.resize(floor(maxl_/2.0) + 1);
    for (int i = 0; i < Ql.size(); ++i)
    {
        Ql[i] = 0;
        Wl[i] = 0;
    }
    //Qlmc.resize(qmlsize());
    for (int i = 0; i < Qlmc.size(); ++i)
    {
        Qlmc[i] = std::complex<double>(0,0);
    }
    std::size_t npairs = nbh.nbhcount_;
    // centered pairs
    for (int i = 0; i < nbh.nbhcount_; ++i)
    {
        csph.computeYCarSafe(nbh.pos_[i][0], nbh.pos_[i][1], nbh.pos_[i][2], nbh.distances_[i]);
        for (int l = 0; l <= maxl_; ++l) {
            Qlmc[get_idx(l, 0)] += csph.get(l, 0);
        }
        for (int l = 0; l <= maxl_; ++l)
        {
            for (int m = 1; m <= l; ++m)
            {
                Qlmc[get_idx(l, m)] += csph.get(l, m);
                Qlmc[get_idx(l, -m)] += csph.get(l, -m);
            }
        }
    }
    // other pairs
    for (int j = 0; j < nbh.nbhcount_ - 1; ++j) {
        for (int k = j + 1; k < nbh.nbhcount_; ++k) {
            Eigen::Vector3d dist = nbh.pos_[j] - nbh.pos_[k];
            double r = dist.norm();
            if (r < rcut) {
                ++npairs;
                csph.computeYCarSafe(dist[0], dist[1], dist[2], r);
                for (int l = 0; l <= maxl_; ++l) {
                    Qlmc[get_idx(l, 0)] += csph.get(l, 0);
                }
                for (int l = 0; l <= maxl_; ++l)
                {
                    for (int m = 1; m <= l; ++m)
                    {
                        Qlmc[get_idx(l, m)] += csph.get(l, m);
                        Qlmc[get_idx(l, -m)] += csph.get(l, -m);
                    }
                }
            }
        }
    }
    if (nbh.nbhcount_ > 0) {
        for (int i = 0; i < Qlmc.size(); ++i)
        {
            Qlmc[i] /= npairs;
        }
    }
    else {
        for (int i = 0; i < Qlmc.size(); ++i)
            Qlmc[i] = 0;
    }
    // compute Ql
    for (int l = 0; l <= maxl_; l += 2)
    {
        Ql[l/2] += (std::conj(Qlmc[get_idx(l, 0)]) * Qlmc[get_idx(l, 0)]).real();
        for (int m = 1; m <= l; ++m)
            Ql[l/2] += (std::conj(Qlmc[get_idx(l, m)]) * Qlmc[get_idx(l, m)] + std::conj(Qlmc[get_idx(l, -m)]) * Qlmc[get_idx(l, -m)]).real();
        // Ql[i] = sqrt(4*M_PI/(2*i + 1) * Ql[i]);
    }
    // compute Wl
#ifdef USE_OMP
#pragma omp parallel for
#endif
    for (int l = 0; l <= maxl_; l += 2)
    {
        for (int m1 = -l; m1 <= l; ++m1)
        {
            for (int m2 = -l; m2 <= l; ++m2)
            {
                int m3 = -m1 - m2;
                if (m3 > l || m3 < -l)
                    continue;
                double w3jm = AngularFuncs::Wigner3jIntCache(l, l, l, m1, m2, m3);
                Wl[l/2] += w3jm * (Qlmc[get_idx(l, m1)] * Qlmc[get_idx(l, m2)] * Qlmc[get_idx(l, m3)]).real();
            }
        }
    }
    // normalize
    for (int l = 0; l <= maxl_; l += 2)
    {
        if (!normalize_w_outer)
            Wl[l/2] = Ql[l/2] > 0.0 ? pow(Ql[l/2], -3. / 2.) * Wl[l/2] : 0.0;
        Ql[l/2] = sqrt(4 * M_PI / (2 * l + 1) * Ql[l/2]); // order is correct, see paper carefully
    }
}
void DescriptorBondOrder::compute(Neighborhood &nbh, Eigen::VectorXd &out)
{
    // Internal compute returns even-l values including l=0:
    //   Ql = [Q_0, Q_2, Q_4, Q_6, ...], Wl = [W_0, W_2, W_4, W_6, ...]
    // Q_0 and W_0 are always 1 (trivial invariants) — skip them.
    // Odd-l entries are always zero (not defined) — excluded.
    // Output layout: [Q_2, Q_4, ..., Q_maxl, W_2, W_4, ..., W_maxl]
    std::vector<double> Ql, Wl;
    compute(nbh, Ql, Wl);
    int n_feat = maxl_ / 2;          // number of even l >= 2
    out.resize(2 * n_feat);          // = maxl_
    for (int i = 1; i <= n_feat; ++i) {
        out[i - 1]            = Ql[i];  // Q_{2i}
        out[i - 1 + n_feat]   = Wl[i];  // W_{2i}
    }
}

void DescriptorBondOrder::compute(Neighborhood &nbh, Eigen::VectorXd &out,
                                   const ComputeOptions &opts)
{
    compute(nbh, out);
    if (!opts.compute_derivative || !opts.gradient) return;

    const int N = nbh.nbhcount_;
    const int sz = maxl_ / 2;
    auto &grad = opts.gradient->grad;
    grad.resize(2 * sz, N, 3);
    grad.setZero();
    if (N == 0) return;

    const double invN = 1.0 / N;
    std::vector<double> norm_sq(sz + 1, 0.0);
    std::vector<std::complex<double>> dW_dq(qlmsize, 0.0);
    for (int l = 2; l <= maxl_; l += 2) {
        for (int m = -l; m <= l; ++m)
            norm_sq[l/2] += std::norm(Qlmc[get_idx(l, m)]);
        for (int m1 = -l; m1 <= l; ++m1)
            for (int m2 = -l; m2 <= l; ++m2) {
                const int m3 = -m1 - m2;
                if (std::abs(m3) > l) continue;
                const int i1 = get_idx(l, m1), i2 = get_idx(l, m2), i3 = get_idx(l, m3);
                const double c = AngularFuncs::Wigner3jIntCache(l, l, l, m1, m2, m3);
                dW_dq[i1] += c * Qlmc[i2] * Qlmc[i3];
                dW_dq[i2] += c * Qlmc[i1] * Qlmc[i3];
                dW_dq[i3] += c * Qlmc[i1] * Qlmc[i2];
            }
    }

    for (int j = 0; j < N; ++j) {
        // This utility includes the finite Cartesian limits on both poles.
        csph.computedYCar(nbh.pos_[j][0], nbh.pos_[j][1], nbh.pos_[j][2], nbh.distances_[j]);
        const double *real_der[] = {csph.dYdx, csph.dYdy, csph.dYdz};
        const double *imag_der[] = {csph.dYidx, csph.dYidy, csph.dYidz};
        for (int l = 2; l <= maxl_; l += 2) {
            const int f = l/2 - 1;
            const double sq = norm_sq[l/2];
            for (int a = 0; a < 3; ++a) {
                double qdot = 0.0, wdot = 0.0;
                for (int m = -l; m <= l; ++m) {
                    const int idx = get_idx(l, m);
                    const std::complex<double> dq(real_der[a][idx], imag_der[a][idx]);
                    qdot += std::real(std::conj(Qlmc[idx]) * dq) * invN;
                    wdot += std::real(dW_dq[idx] * dq) * invN;
                }
                if (sq > 0) grad(f, j, a) = out[f] * qdot / sq;
                if (!normalize_w_outer)
                    wdot = sq > 0 ? wdot * std::pow(sq, -1.5) - 3 * out[f + sz] * qdot / sq : 0;
                grad(f + sz, j, a) = wdot;
            }
        }
    }
}
