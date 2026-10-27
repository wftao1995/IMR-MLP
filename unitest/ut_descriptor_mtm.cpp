// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "descriptors/descriptor_MTM.h"
#include "descriptors/descriptors.h"
#include "maxvol/b_matrix_builder.h"
#include "potentials/quick_mtp.h"
#include "potentials/potential_mtp.h"
#include "base/structures.h"
#include <cmath>
#include <vector>

using Catch::Approx;

struct MTPHandle {
    Potentials::PotMTP *ptr;
    MTPHandle() : ptr(Potentials::quick_mtp_3b_s14_2f0ee25e()) {}
    ~MTPHandle() { delete ptr; }
    Potentials::PotMTP &get() { return *ptr; }
};

static Atoms make_cfg(double a, int nx, int ny, int nz, double noise = 0.0)
{
    Atoms cfg;
    cfg.cell_ = Eigen::Matrix3d::Identity() * a * nx;
    for (int i = 0; i < nx; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k) {
                double dx = noise * (std::rand() / (double)RAND_MAX - 0.5) * a;
                double dy = noise * (std::rand() / (double)RAND_MAX - 0.5) * a;
                double dz = noise * (std::rand() / (double)RAND_MAX - 0.5) * a;
                cfg.pos_.push_back({i * a + dx, j * a + dy, k * a + dz});
                cfg.chemmap_.push_back(0);
            }
    return cfg;
}

MTPHandle mtp;

TEST_CASE("DescriptorMTM construction", "[desc_mtm]")
{
    DescriptorMTM desc(mtp.get());
    REQUIRE(desc.feature_size() == mtp.get().get_alpha_scalar_moments());
    REQUIRE(desc.cutoff_ == Approx(mtp.get().get_cutoff()));
    REQUIRE(desc.feature_size() > 0);
}

TEST_CASE("DescriptorMTM compute matches CalcLocalDescriptors", "[desc_mtm]")
{
    DescriptorMTM desc(mtp.get());
    Atoms cfg = make_cfg(4.0, 2, 2, 2, 0.03);
    int natoms = cfg.size();
    int ncols = desc.feature_size();

    // Reference: PotMTP::CalcLocalDescriptors
    Eigen::MatrixXd ref;
    mtp.get().CalcLocalDescriptors(cfg, ref);

    // DescriptorMTM::compute (per-atom, via Neighborhoods)
    Neighborhoods nbhs(cfg, mtp.get().get_cutoff());
    for (int iat = 0; iat < natoms; ++iat) {
        Eigen::VectorXd out;
        desc.compute(nbhs[iat], out);

        REQUIRE(out.size() == ncols);
        for (int j = 0; j < ncols; ++j)
            REQUIRE(out[j] == Approx(ref(j, iat)).margin(1e-10));
    }
}

TEST_CASE("DescriptorMTM compute non-zero for valid structure", "[desc_mtm]")
{
    DescriptorMTM desc(mtp.get());
    Atoms cfg = make_cfg(4.0, 2, 2, 2);
    Neighborhoods nbhs(cfg, mtp.get().get_cutoff());

    Eigen::VectorXd out;
    desc.compute(nbhs[0], out);

    // All scalar moments should be finite
    for (int j = 0; j < out.size(); ++j)
        REQUIRE(std::isfinite(out[j]));

    // At least some moments should be non-zero
    double sum_abs = out.array().abs().sum();
    REQUIRE(sum_abs > 0.0);
}

TEST_CASE("DescriptorMTM with DescriptorBuilder via factory pattern", "[desc_mtm]")
{
    // Simulate what maxvol_select's make_builder does
    DescriptorMTM desc(mtp.get());
    DescriptorBuilder builder(desc);

    REQUIRE(builder.coeff_count() == desc.feature_size());
    REQUIRE(builder.cutoff() == Approx(mtp.get().get_cutoff()));

    Atoms cfg = make_cfg(4.0, 2, 2, 2);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    REQUIRE(nrows == cfg.size());  // one row per atom
    REQUIRE(B.rows() == cfg.size());
    REQUIRE(B.cols() == desc.feature_size());
}

TEST_CASE("DescriptorMTM clones compute concurrently without shared scratch races",
          "[desc_mtm][parallel]")
{
    constexpr int ncfg = 12;
    std::vector<Atoms> cfgs;
    cfgs.reserve(ncfg);
    for (int i = 0; i < ncfg; ++i)
        cfgs.push_back(make_cfg(3.8 + 0.03 * i, 2, 2, 2, 0.01 * (i + 1)));

    DescriptorMTM serial(mtp.get());
    std::vector<Eigen::VectorXd> expected(ncfg);
    for (int i = 0; i < ncfg; ++i) {
        Neighborhoods nbhs(cfgs[i], serial.cutoff_);
        serial.compute(nbhs[i % nbhs.size()], expected[i]);
    }

    std::vector<Eigen::VectorXd> actual(ncfg);
#ifdef USE_OMP
    #pragma omp parallel for schedule(dynamic, 1) num_threads(4)
#endif
    for (int i = 0; i < ncfg; ++i) {
        auto descriptor = serial.clone();
        Atoms local_cfg = cfgs[i];
        Neighborhoods nbhs(local_cfg, descriptor->cutoff_);
        descriptor->compute(nbhs[i % nbhs.size()], actual[i]);
    }

    for (int i = 0; i < ncfg; ++i) {
        REQUIRE(actual[i].size() == expected[i].size());
        REQUIRE(actual[i].isApprox(expected[i], 1e-12));
    }
}
