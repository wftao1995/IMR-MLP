// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_DESCRIPTOR_BONDORDER_H
#define IMRMLP_DESCRIPTOR_BONDORDER_H

#include "descriptors.h"
#include "CutoffFuncs.h"
#include "mathextra/angular_functions.h"
#include "utils/gputils.h"

/**
 * @brief Steinhardt bond-order parameters Q_l and W_l (rotational invariants).
 * @ingroup descriptors
 *
 * Descriptor output layout: [Q_2, Q_4, ..., Q_maxl, W_2, W_4, ..., W_maxl]
 *   feature_size() = maxl_  (maxl_ must be even and at least 2)
 *
 * Excluded from output (no structural information):
 *   - Q_0, W_0: trivial invariants that are always 1 regardless of structure
 *   - Q_l, W_l for odd l: bond-order parameters are only defined for even l
 *
 * Internal compute() helpers return all even-l values including l=0;
 * the Descriptors interface methods strip trivial/undefined entries.
 *
 * @see Steinhardt, P. J., Nelson, D. R., & Ronchetti, M. (1983), Phys. Rev. B 28, 784.
 * **/

class DescriptorBondOrder : public Descriptors, public ConnectSettings
{
private:
    int maxl_ = 4;
    int qlmsize;
    std::vector<double> Qlm;
    std::vector<std::complex<double>> Qlmc;

    AngularFuncs::RealSphericalHarmoics rsph;
    AngularFuncs::ComplexSphericalHarmoics csph;
    DescriptorBondOrder(const DescriptorBondOrder &) = delete;
    DescriptorBondOrder &operator=(const DescriptorBondOrder &) = delete;

    bool normalize_w_inner = false;
    bool normalize_w_outer = false;

    inline int get_idx(const int &l, const int &m) const
    {
        return m + l * (l + 1);
    }

public:
    DescriptorBondOrder(int _maxl) : DescriptorBondOrder()
    {
        init(_maxl);
    };
    DescriptorBondOrder() { cutoff_ = 5.0; };

    std::unique_ptr<Descriptors> clone() const override {
        auto copy = std::make_unique<DescriptorBondOrder>(maxl_);
        copy->cutoff_ = cutoff_;
        copy->normalize_w_inner = normalize_w_inner;
        copy->normalize_w_outer = normalize_w_outer;
        copy->InitSettings();
        return copy;
    }
    ~DescriptorBondOrder(){};

    // ── GPSettings constructor (MSD pattern) ──
    void InitSettings()
    {
        Connect(cutoff_, "cutoff", "cutoff radius for neighborhood");
        Connect(maxl_, "maxl", "maximum angular momentum L (even, at least 2, default 4)");
        Connect(normalize_w_inner, "normalize_w_inner", "must be false; termwise normalization is not rotationally invariant");
        Connect(normalize_w_outer, "normalize_w_outer", "legacy flag: false gives normalized Wl, true gives raw Wl");
    }
    DescriptorBondOrder(const GPSettings &_settings) : DescriptorBondOrder()
    {
        InitSettings();
        ApplySettings(_settings);
        init(maxl_);
    }

    int lsize() const { return maxl_; };
    int qmlsize() const { return qlmsize; };
    void get_qlm(std::vector<double> &qlm)
    {
        qlm = Qlm;
    }
    void get_qlm(std::vector<std::complex<double>> &qlm)
    {
        qlm = Qlmc;
    }

    void init(int ml)
    {
        if (ml < 2 || ml % 2 != 0)
            GPERROR("BondOrder maxl must be even and at least 2");
        if (normalize_w_inner)
            GPERROR("BondOrder normalize_w_inner=true is not rotationally invariant; use false");
        maxl_ = ml;
        qlmsize = (maxl_ + 1) * (maxl_ + 1);
        Qlm.resize(qlmsize);
        Qlmc.resize(qlmsize); 
        rsph.init(maxl_, maxl_);
        csph.init(maxl_, maxl_);
    }

    void compute(const Neighborhood &nbh, std::vector<double> &Ql, std::vector<double> &Wl);
    void compute_ref(const Neighborhood &nbh, std::vector<double> &Ql, std::vector<double> &Wl);
    void compute_ref_pairs(const Neighborhood &nbh, const double rcut, std::vector<double> &Ql, std::vector<double> &Wl);

    // --- Descriptors interface ---
    void compute(Neighborhood &nbh, Eigen::VectorXd &out) override;
    void compute(Neighborhood &nbh, Eigen::VectorXd &out,
                 const ComputeOptions &opts) override;
    int feature_size() const override { return maxl_; }
    bool supports_weighted_site_hvp() const override { return true; }
    bool supports_weighted_site_hessian() const override { return true; }
    void compute_weighted_site_hvp(Neighborhood &, const Eigen::VectorXd &,
        const DescriptorSiteMatrix &, double &, DescriptorSiteMatrix &, DescriptorSiteMatrix &) override;
    void compute_weighted_site_hessian(Neighborhood &, const Eigen::VectorXd &,
        double &, DescriptorSiteMatrix &, Eigen::MatrixXd &) override;
};

#endif
