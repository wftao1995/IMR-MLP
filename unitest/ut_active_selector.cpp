// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "maxvol/active_selector.h"
#include "active_learn/al_potential.h"
#include "maxvol/b_matrix_builder.h"
#include "base/structures.h"
#include "potentials/quick_mtp.h"
#include "descriptors/descriptor_MTM.h"
#include "descriptors/descriptor_BondOrder.h"
#include <cmath>
#include <atomic>
#include <cstdio>
#include <fstream>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>
#ifdef USE_OMP
#include <omp.h>
#endif

using Catch::Approx;

struct OmpThreadRestore {
    int threads = 1;

    OmpThreadRestore()
    {
#ifdef USE_OMP
        threads = omp_get_max_threads();
#endif
    }

    ~OmpThreadRestore()
    {
#ifdef USE_OMP
        omp_set_num_threads(threads);
#endif
    }
};

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

struct SimpleRowBuilder : BMatrixBuilder {
    int coeff_count() const override { return 2; }
    double cutoff() const override { return 0.0; }
    int build_rows(Atoms &cfg,
                   Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                   double weight, double) override
    {
        const int cur = static_cast<int>(B.rows());
        B.conservativeResize(cur + 1, 2);
        B(cur, 0) = weight * std::stod(cfg.features_["x"]);
        B(cur, 1) = weight * std::stod(cfg.features_["y"]);
        return 1;
    }
    std::unique_ptr<BMatrixBuilder> clone() const override
    {
        return std::make_unique<SimpleRowBuilder>();
    }
};

struct PositionRowsBuilder : BMatrixBuilder {
    int coeff_count() const override { return 2; }
    double cutoff() const override { return 0.0; }
    int build_rows(Atoms &cfg,
                   Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                   double weight, double) override
    {
        const int cur = static_cast<int>(B.rows());
        const int nrows = static_cast<int>(cfg.pos_.size());
        B.conservativeResize(cur + nrows, 2);
        for (int i = 0; i < nrows; ++i) {
            B(cur + i, 0) = weight * cfg.pos_[i][0];
            B(cur + i, 1) = weight * cfg.pos_[i][1];
        }
        return nrows;
    }
    std::unique_ptr<BMatrixBuilder> clone() const override
    {
        return std::make_unique<PositionRowsBuilder>();
    }
};

struct NonCloneablePositionRowsBuilder : PositionRowsBuilder {
    std::unique_ptr<BMatrixBuilder> clone() const override
    {
        return nullptr;
    }
};

struct ThreeComponentRowBuilder : BMatrixBuilder {
    int coeff_count() const override { return 3; }
    double cutoff() const override { return 0.0; }
    int build_rows(Atoms &cfg,
                   Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B,
                   double weight, double) override
    {
        const int cur = static_cast<int>(B.rows());
        B.conservativeResize(cur + 1, 3);
        B.row(cur) << weight * std::stod(cfg.features_["x"]),
                      weight * std::stod(cfg.features_["y"]),
                      weight * std::stod(cfg.features_["z"]);
        return 1;
    }
    std::unique_ptr<BMatrixBuilder> clone() const override
    {
        return std::make_unique<ThreeComponentRowBuilder>();
    }
};

struct CountedRowBuilder : BMatrixBuilder {
    explicit CountedRowBuilder(std::shared_ptr<std::atomic<int>> calls = {})
        : calls(calls ? std::move(calls)
                      : std::make_shared<std::atomic<int>>(0)) {}

    int coeff_count() const override { return 2; }
    double cutoff() const override { return 0.0; }
    int row_count(const Atoms &) const override { return 1; }

    int build_rows(Atoms &cfg, RowMatrix &B,
                   double weight, double) override
    {
        const int current = static_cast<int>(B.rows());
        B.conservativeResize(current + 1, 2);
        fill(cfg, B.row(current), weight);
        return 1;
    }

    int build_rows_into(Atoms &cfg, Eigen::Ref<RowMatrix> out,
                        double weight, double,
                        std::vector<BRowRef> *refs = nullptr) override
    {
        if (out.rows() < 1 || out.cols() != 2)
            throw std::invalid_argument("CountedRowBuilder output shape");
        fill(cfg, out.row(0), weight);
        if (refs) {
            BRowRef ref;
            ref.local_row = 0;
            refs->push_back(ref);
        }
        return 1;
    }

    std::unique_ptr<BMatrixBuilder> clone() const override
    {
        return std::make_unique<CountedRowBuilder>(calls);
    }

    std::shared_ptr<std::atomic<int>> calls;

private:
    template <class Row>
    void fill(Atoms &cfg, Row &&row, double weight)
    {
        calls->fetch_add(1, std::memory_order_relaxed);
        row[0] = weight * std::stod(cfg.features_["x"]);
        row[1] = weight * std::stod(cfg.features_["y"]);
    }
};

static Atoms make_row_cfg(double x, double y)
{
    Atoms cfg;
    cfg.resize(1);
    cfg.features_["x"] = std::to_string(x);
    cfg.features_["y"] = std::to_string(y);
    return cfg;
}

static Atoms make_three_component_cfg(double x, double y, double z)
{
    Atoms cfg = make_row_cfg(x, y);
    cfg.features_["z"] = std::to_string(z);
    return cfg;
}

TEST_CASE("ActiveSelector MLIP initialization retains virtual rank slots",
          "[active_sel][mlip_init]")
{
    std::vector<Atoms> cfgs{
        make_three_component_cfg(2.0, 0.0, 0.0),
        make_three_component_cfg(0.0, 3.0, 0.0),
        make_three_component_cfg(1.0, 1.0, 0.0),
        make_three_component_cfg(-1.0, 2.0, 0.0)};

    ActiveSelector full(new ThreeComponentRowBuilder());
    full.init_mode = "mlip";
    full.init_value = 1e-6;
    full.select_mode = "full";
    full.threshold = 1.001;
    full.pow_scale = 0.0;
    full.add_candidates_parallel(cfgs);
    REQUIRE(full.select() == 2);
    REQUIRE(full.virtual_row_count() == 1);
    REQUIRE(full.converged());
    REQUIRE(full.max_remaining_grade() <= 1.001 + 1e-10);

    ActiveSelector stream(new ThreeComponentRowBuilder());
    stream.init_mode = "mlip";
    stream.init_value = 1e-6;
    stream.select_mode = "stream";
    stream.stream_mode = "exact";
    stream.stream_cache_mode = "memory";
    stream.stream_chunk_size = 1;
    stream.threshold = 1.001;
    stream.pow_scale = 0.0;
    REQUIRE(stream.select_streaming(cfgs) == 2);
    REQUIRE(stream.virtual_row_count() == 1);
    REQUIRE(stream.converged());
    REQUIRE(stream.max_remaining_grade() <= 1.001 + 1e-10);
    REQUIRE(stream.selected_indices() == full.selected_indices());
}

TEST_CASE("ActiveSelector reports capped MLIP selection",
          "[active_sel][mlip_init]")
{
    std::vector<Atoms> cfgs{
        make_three_component_cfg(2.0, 0.0, 0.0),
        make_three_component_cfg(0.0, 3.0, 0.0)};
    ActiveSelector selector(new ThreeComponentRowBuilder());
    selector.init_mode = "mlip";
    selector.init_value = 1e-6;
    selector.select_mode = "full";
    selector.max_swaps = 1;
    selector.pow_scale = 0.0;
    selector.add_candidates(cfgs);
    REQUIRE(selector.select() == 1);
    REQUIRE_FALSE(selector.converged());
    REQUIRE(selector.max_remaining_grade() > selector.threshold);
    REQUIRE(selector.termination_reason() == "max-swaps");
}

static Atoms make_position_rows(std::initializer_list<std::pair<double, double>> rows)
{
    Atoms cfg;
    cfg.cell_ = Eigen::Matrix3d::Identity();
    for (const auto &row : rows) {
        cfg.pos_.push_back({row.first, row.second, 0.0});
        cfg.chemmap_.push_back(0);
    }
    return cfg;
}

// Use DescriptorMTM + DescriptorBuilder for all tests
MTPHandle mtp;

TEST_CASE("ActiveSelector construction and add_candidate", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    ActiveSelector as(builder);
    as.weight = 1.0; as.pow_scale = 0.0;

    REQUIRE(as.num_candidates() == 0);

    Atoms cfg = make_cfg(4.0, 2, 2, 2);
    int nrows = as.add_candidate(cfg);
    REQUIRE(nrows == cfg.size());  // per-atom → natoms rows
    REQUIRE(as.num_candidates() == 1);
    REQUIRE(as.num_rows() == cfg.size());
}

TEST_CASE("ActiveSelector clear_candidates", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    ActiveSelector as(builder);
    as.weight = 1.0; as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 2));
    cfgs.push_back(make_cfg(4.2, 2, 2, 2));
    as.add_candidates(cfgs);
    REQUIRE(as.num_candidates() == 2);
    as.clear_candidates();
    REQUIRE(as.num_candidates() == 0);
}

TEST_CASE("ActiveSelector parallel add_candidates matches serial", "[active_sel]")
{
    DescriptorMTM desc_serial(mtp.get());
    auto *builder_serial = new DescriptorBuilder(desc_serial);
    ActiveSelector serial(builder_serial);
    serial.threshold = 1.001; serial.weight = 1.0; serial.pow_scale = 0.0;

    DescriptorMTM desc_parallel(mtp.get());
    auto *builder_parallel = new DescriptorBuilder(desc_parallel);
    ActiveSelector parallel(builder_parallel);
    parallel.threshold = 1.001; parallel.weight = 1.0; parallel.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(3.8, 2, 2, 1, 0.01));
    cfgs.push_back(make_cfg(4.0, 2, 2, 2, 0.02));
    cfgs.push_back(make_cfg(4.2, 3, 2, 1, 0.03));

    int nserial = serial.add_candidates(cfgs);
    int nparallel = parallel.add_candidates_parallel(cfgs);
    REQUIRE(nparallel == nserial);
    REQUIRE(parallel.num_candidates() == serial.num_candidates());
    REQUIRE(parallel.num_rows() == serial.num_rows());

    REQUIRE(parallel.select() == serial.select());
    REQUIRE(parallel.selected_indices() == serial.selected_indices());
}

TEST_CASE("ActiveSelector appends prebuilt candidates", "[active_sel]")
{
    DescriptorMTM desc_a(mtp.get());
    auto *builder_a = new DescriptorBuilder(desc_a);
    ActiveSelector a(builder_a);
    a.weight = 1.0; a.pow_scale = 0.0;

    DescriptorMTM desc_b(mtp.get());
    auto *builder_b = new DescriptorBuilder(desc_b);
    ActiveSelector b(builder_b);
    b.weight = 1.0; b.pow_scale = 0.0;

    std::vector<Atoms> first;
    first.push_back(make_cfg(4.0, 2, 2, 1));
    std::vector<Atoms> second;
    second.push_back(make_cfg(4.2, 2, 2, 2));
    second.push_back(make_cfg(4.4, 3, 2, 1));

    int rows_first = a.add_candidates_parallel(first);
    int rows_second = b.add_candidates_parallel(second);

    REQUIRE(b.append_built_candidates_from(a));
    REQUIRE(b.num_candidates() == 3);
    REQUIRE(b.num_rows() == rows_first + rows_second);
}

TEST_CASE("PotEFSBuilder factory clone matches original rows", "[efs_builder]")
{
    const std::string tmpfile = "/tmp/active_selector_potefs_clone.mtp";
    mtp.get().Save(tmpfile);

    GPSettings settings;
    auto builder = make_builder("pot_efs", tmpfile, "mtp", settings, "nbh",
                                1.0, 0.0, 0.0, 0.0, false);
    REQUIRE(builder);
    auto cloned = builder->clone();
    REQUIRE(cloned);
    REQUIRE(cloned->coeff_count() == builder->coeff_count());

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.01);
    Atoms cfg_ref = cfg;
    Atoms cfg_clone = cfg;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_ref, B_clone;
    int nref = builder->build_rows(cfg_ref, B_ref, 1.0, 0.0);
    int nclone = cloned->build_rows(cfg_clone, B_clone, 1.0, 0.0);

    REQUIRE(nclone == nref);
    REQUIRE(B_clone.rows() == B_ref.rows());
    REQUIRE(B_clone.cols() == B_ref.cols());
    for (int i = 0; i < B_ref.rows(); ++i)
        for (int j = 0; j < B_ref.cols(); ++j)
            REQUIRE(std::isfinite(B_clone(i, j)));

    std::remove(tmpfile.c_str());
}

TEST_CASE("ActiveSelector grade without selection is sentinel", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    ActiveSelector as(builder);
    Atoms cfg = make_cfg(4.0, 2, 2, 2);
    double g = as.grade(cfg);
    REQUIRE(g > 1e9);
}

TEST_CASE("ActiveSelector select and grade", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    ActiveSelector as(builder);
    as.threshold = 1.001; as.weight = 1.0; as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(3.8, 2, 2, 2, 0.05));
    cfgs.push_back(make_cfg(4.0, 2, 2, 2, 0.03));
    cfgs.push_back(make_cfg(4.2, 2, 2, 2, 0.04));

    as.add_candidates(cfgs);
    int nsel = as.select();
    REQUIRE(nsel >= 1);
    REQUIRE(static_cast<int>(as.get_selected().size()) == nsel);
    REQUIRE(as.log_determinant() > -1e30);

    Atoms new_cfg = make_cfg(5.0, 2, 2, 2, 0.1);
    double g = as.grade(new_cfg);
    REQUIRE(std::isfinite(g));
    REQUIRE(g >= 0.0);
}

TEST_CASE("ActiveSelector compact grading state matches full state", "[active_sel]")
{
    ActiveSelector full(new SimpleRowBuilder());
    full.threshold = 1.001; full.weight = 1.0; full.pow_scale = 0.0;

    std::vector<Atoms> train;
    train.push_back(make_row_cfg(1.0, 0.0));
    train.push_back(make_row_cfg(0.0, 1.0));
    full.add_candidates(train);
    REQUIRE(full.select() == 2);

    Atoms probe = make_row_cfg(0.25, 0.75);
    const double full_grade = full.grade(probe);
    REQUIRE(full_grade == Approx(0.75).margin(1e-12));

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> C(2, 2);
    C << 1.0, 0.0,
         0.0, 1.0;

    ActiveSelector compact(new SimpleRowBuilder());
    REQUIRE(compact.set_active_rows_for_grading(C, {0, 1}, &train));
    Atoms probe_copy = probe;
    const double compact_grade = compact.grade(probe_copy);
    REQUIRE(compact_grade == Approx(full_grade).margin(1e-12));
}

TEST_CASE("ActiveSelector streaming initialization keeps low-norm independent rows", "[active_sel]")
{
    ActiveSelector as(new SimpleRowBuilder());
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.select_mode = "stream";
    as.stream_mode = "exact";

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_row_cfg(100.0, 0.0));
    cfgs.push_back(make_row_cfg(90.0, 0.0));
    cfgs.push_back(make_row_cfg(0.0, 1.0));

    REQUIRE(as.select_streaming(cfgs) == 2);
    Atoms y_axis = make_row_cfg(0.0, 1.0);
    REQUIRE(as.grade(y_axis) == Approx(1.0).margin(1e-12));
}

TEST_CASE("ActiveSelector streaming initialization ignores a leading zero row",
          "[active_sel]")
{
    ActiveSelector as(new SimpleRowBuilder());
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.select_mode = "stream";
    as.stream_mode = "exact";

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_row_cfg(0.0, 0.0));
    cfgs.push_back(make_row_cfg(1.0, 0.0));
    cfgs.push_back(make_row_cfg(0.0, 1.0));

    REQUIRE(as.select_streaming(cfgs) == 2);
    Atoms y_axis = make_row_cfg(0.0, 1.0);
    REQUIRE(as.grade(y_axis) == Approx(1.0).margin(1e-12));
}

TEST_CASE("ActiveSelector cached stream builds each configuration once",
          "[active_sel][stream_cache]")
{
    auto calls = std::make_shared<std::atomic<int>>(0);
    ActiveSelector as(new CountedRowBuilder(calls));
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.max_swaps = 8;
    as.stream_mode = "exact";
    as.stream_cache_mode = "memory";
    as.stream_chunk_size = 2;

    std::vector<Atoms> cfgs{
        make_row_cfg(1.0, 0.0),
        make_row_cfg(0.0, 1.0),
        make_row_cfg(4.0, 0.1),
        make_row_cfg(0.1, 5.0),
        make_row_cfg(3.0, 3.0),
    };
    REQUIRE(as.select_streaming(cfgs) == 2);

    const auto &stats = as.stream_stats();
    REQUIRE(calls->load(std::memory_order_relaxed) ==
            static_cast<int>(cfgs.size()));
    REQUIRE(stats.builder_calls == cfgs.size());
    REQUIRE(stats.rows_built == cfgs.size());
    REQUIRE(stats.descriptor_passes == 1);
    REQUIRE(stats.score_passes >= 1);
    REQUIRE(stats.cache_backend == "memory");
}

TEST_CASE("ActiveSelector cached stream is numerically equivalent across threads",
          "[active_sel][stream_cache][openmp]")
{
    OmpThreadRestore restore_threads;
    std::vector<Atoms> cfgs{
        make_row_cfg(1.0, 0.0),
        make_row_cfg(0.0, 1.0),
        make_row_cfg(4.0, 0.1),
        make_row_cfg(0.1, 5.0),
        make_row_cfg(3.0, 3.0),
        make_row_cfg(-2.0, 4.0),
        make_row_cfg(5.0, -1.0),
        make_row_cfg(0.5, 0.25),
    };

#ifdef USE_OMP
    omp_set_num_threads(1);
#endif
    ActiveSelector serial(new SimpleRowBuilder());
    serial.threshold = 1.001;
    serial.weight = 1.0;
    serial.pow_scale = 0.0;
    serial.stream_mode = "exact";
    serial.stream_cache_mode = "memory";
    serial.stream_chunk_size = 1;
    const int serial_selected = serial.select_streaming(cfgs);

#ifdef USE_OMP
    omp_set_num_threads(4);
#endif
    ActiveSelector parallel(new SimpleRowBuilder());
    parallel.threshold = serial.threshold;
    parallel.weight = serial.weight;
    parallel.pow_scale = serial.pow_scale;
    parallel.stream_mode = serial.stream_mode;
    parallel.stream_cache_mode = serial.stream_cache_mode;
    parallel.stream_chunk_size = serial.stream_chunk_size;
    const int parallel_selected = parallel.select_streaming(cfgs);

    REQUIRE(parallel_selected == serial_selected);
    REQUIRE(parallel.log_determinant() ==
            Approx(serial.log_determinant()).epsilon(1e-9));
    REQUIRE(parallel.stream_stats().configured_threads >= 1);
    REQUIRE(parallel.stream_stats().build_seconds >= 0.0);
    REQUIRE(parallel.stream_stats().initialization_seconds >= 0.0);
    REQUIRE(parallel.stream_stats().score_seconds >= 0.0);
    REQUIRE(parallel.stream_stats().total_seconds >=
            parallel.stream_stats().score_seconds);
#ifdef USE_OMP
    REQUIRE(parallel.stream_stats().configured_threads == 4);
    REQUIRE(parallel.stream_stats().build_threads >= 2);
    REQUIRE(parallel.stream_stats().score_threads >= 2);
#endif

    for (auto cfg : cfgs)
        REQUIRE(parallel.grade(cfg) <=
                parallel.threshold + 1e-9);
}

TEST_CASE("ActiveSelector parallel fast shortlist matches serial volume",
          "[active_sel][stream_fast][openmp]")
{
    OmpThreadRestore restore_threads;
    std::vector<Atoms> cfgs{
        make_row_cfg(1.0, 0.0),
        make_row_cfg(0.0, 1.0),
        make_row_cfg(4.0, 0.1),
        make_row_cfg(0.1, 5.0),
        make_row_cfg(3.0, 3.0),
        make_row_cfg(-2.0, 4.0),
    };

    auto run = [&](int threads) {
#ifdef USE_OMP
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
        auto selector = std::make_unique<ActiveSelector>(
            new CountedRowBuilder());
        selector->threshold = 1.001;
        selector->weight = 1.0;
        selector->pow_scale = 0.0;
        selector->stream_mode = "fast";
        selector->stream_fast_passes = 8;
        selector->stream_fast_shortlist =
            static_cast<int>(cfgs.size());
        selector->stream_cache_mode = "memory";
        selector->stream_chunk_size = 1;
        selector->select_streaming(cfgs);
        return selector;
    };

    auto serial = run(1);
    auto parallel = run(4);
    REQUIRE(parallel->get_selected().size() ==
            serial->get_selected().size());
    REQUIRE(parallel->log_determinant() ==
            Approx(serial->log_determinant()).epsilon(1e-9));
}

TEST_CASE("ActiveSelector non-cloneable stream builder falls back to one thread",
          "[active_sel][stream_cache][openmp]")
{
    OmpThreadRestore restore_threads;
#ifdef USE_OMP
    omp_set_num_threads(4);
#endif
    ActiveSelector selector(new NonCloneablePositionRowsBuilder());
    selector.threshold = 1.001;
    selector.weight = 1.0;
    selector.pow_scale = 0.0;
    selector.stream_cache_mode = "memory";
    selector.stream_chunk_size = 1;

    std::vector<Atoms> cfgs{
        make_position_rows({{1.0, 0.0}}),
        make_position_rows({{0.0, 1.0}}),
        make_position_rows({{3.0, 0.5}}),
        make_position_rows({{0.5, 4.0}}),
    };
    REQUIRE(selector.select_streaming(cfgs) == 2);
    REQUIRE(selector.stream_stats().build_threads == 1);
}

TEST_CASE("ActiveSelector mmap cache preserves exact stream result",
          "[active_sel][stream_cache]")
{
    std::vector<Atoms> cfgs{
        make_row_cfg(1.0, 0.0),
        make_row_cfg(0.0, 1.0),
        make_row_cfg(4.0, 0.1),
        make_row_cfg(0.1, 5.0),
        make_row_cfg(3.0, 3.0),
    };

    ActiveSelector memory(new CountedRowBuilder());
    memory.threshold = 1.001;
    memory.weight = 1.0;
    memory.pow_scale = 0.0;
    memory.stream_cache_mode = "memory";
    REQUIRE(memory.select_streaming(cfgs) == 2);

    ActiveSelector mapped(new CountedRowBuilder());
    mapped.threshold = memory.threshold;
    mapped.weight = memory.weight;
    mapped.pow_scale = memory.pow_scale;
    mapped.stream_cache_mode = "mmap";
    mapped.stream_cache_dir = "/tmp";
    REQUIRE(mapped.select_streaming(cfgs) == 2);

    REQUIRE(mapped.selected_indices() == memory.selected_indices());
    REQUIRE(mapped.log_determinant() ==
            Approx(memory.log_determinant()).margin(1e-12));
    REQUIRE(mapped.stream_stats().resident_bytes == 0);
    REQUIRE(mapped.stream_stats().mapped_bytes > 0);
    REQUIRE(mapped.stream_stats().cache_backend == "mmap");
}

TEST_CASE("ActiveSelector fast full shortlist converges like exact stream",
          "[active_sel][stream_fast]")
{
    std::vector<Atoms> cfgs{
        make_row_cfg(1.0, 0.0),
        make_row_cfg(0.0, 1.0),
        make_row_cfg(4.0, 0.1),
        make_row_cfg(0.1, 5.0),
        make_row_cfg(3.0, 3.0),
    };

    ActiveSelector exact(new SimpleRowBuilder());
    exact.threshold = 1.001;
    exact.weight = 1.0;
    exact.pow_scale = 0.0;
    exact.stream_mode = "exact";
    exact.stream_cache_mode = "memory";
    REQUIRE(exact.select_streaming(cfgs) == 2);

    ActiveSelector fast(new SimpleRowBuilder());
    fast.threshold = exact.threshold;
    fast.weight = exact.weight;
    fast.pow_scale = exact.pow_scale;
    fast.stream_mode = "fast";
    fast.stream_fast_passes = 8;
    fast.stream_fast_shortlist = static_cast<int>(cfgs.size());
    fast.stream_cache_mode = "memory";
    REQUIRE(fast.select_streaming(cfgs) == 2);

    REQUIRE(fast.selected_indices() == exact.selected_indices());
    REQUIRE(fast.log_determinant() ==
            Approx(exact.log_determinant()).margin(1e-12));
}

TEST_CASE("ActiveSelector online upgrade processes configs appended after streaming",
          "[active_sel]")
{
    ActiveSelector as(new SimpleRowBuilder());
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.select_mode = "stream";

    std::vector<Atoms> initial;
    initial.push_back(make_row_cfg(1.0, 0.0));
    initial.push_back(make_row_cfg(0.0, 1.0));
    REQUIRE(as.select_streaming(initial) == 2);

    std::vector<Atoms> added;
    added.push_back(make_row_cfg(3.0, 0.0));
    as.add_candidates_unbuilt(added);
    as.select_mode = "online";
    REQUIRE(as.select_online() == 2);
    REQUIRE(as.online_capable());

    Atoms candidate = make_row_cfg(3.0, 0.0);
    REQUIRE(as.grade(candidate) == Approx(1.0).margin(1e-12));
}

TEST_CASE("ActiveSelector online grade sees directions outside a wide active span",
          "[active_sel]")
{
    ActiveSelector as(new ThreeComponentRowBuilder());
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.select_mode = "online";

    std::vector<Atoms> initial;
    initial.push_back(make_three_component_cfg(1.0, 0.0, 0.0));
    initial.push_back(make_three_component_cfg(0.0, 1.0, 0.0));
    as.add_candidates_unbuilt(initial);
    REQUIRE(as.select_online() == 2);
    REQUIRE(as.online_capable());

    Atoms orthogonal = make_three_component_cfg(0.0, 0.0, 2.0);
    REQUIRE(as.grade(orthogonal) == Approx(2.0).margin(1e-12));

    std::vector<Atoms> added{orthogonal};
    as.add_candidates_unbuilt(added);
    REQUIRE(as.select_online() == 3);
    REQUIRE(as.grade(orthogonal) == Approx(1.0).margin(1e-9));
}

TEST_CASE("ActiveSelector exact stream grade sees directions outside a wide active span",
          "[active_sel]")
{
    ActiveSelector as(new ThreeComponentRowBuilder());
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.select_mode = "stream";
    as.stream_mode = "exact";
    as.stream_chunk_size = 2;

    std::vector<Atoms> initial;
    initial.push_back(make_three_component_cfg(1.0, 0.0, 0.0));
    initial.push_back(make_three_component_cfg(0.0, 1.0, 0.0));
    REQUIRE(as.select_streaming(initial) == 2);

    Atoms orthogonal = make_three_component_cfg(0.0, 0.0, 2.0);
    REQUIRE(as.grade(orthogonal) == Approx(2.0).margin(1e-12));
}

TEST_CASE("ActiveSelector online max-swaps resumes an unfinished config",
          "[active_sel]")
{
    ActiveSelector as(new PositionRowsBuilder());
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.max_swaps = 1;
    as.select_mode = "online";

    std::vector<Atoms> initial;
    initial.push_back(make_position_rows({{1.0, 0.0}, {0.0, 1.0}}));
    as.add_candidates_unbuilt(initial);
    REQUIRE(as.select_online() == 1);

    std::vector<Atoms> added;
    added.push_back(make_position_rows({{2.0, 0.0}, {0.0, 3.0}}));
    as.add_candidates_unbuilt(added);
    REQUIRE(as.select_online() >= 1);
    REQUIRE(as.grade(added[0]) == Approx(2.0).margin(1e-12));

    REQUIRE(as.select_online() == 1);
    REQUIRE(as.grade(added[0]) == Approx(1.0).margin(1e-12));
}

TEST_CASE("ActiveSelector grade_all", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    ActiveSelector as(builder);
    as.threshold = 1.001; as.weight = 1.0; as.pow_scale = 0.0;

    std::vector<Atoms> train;
    train.push_back(make_cfg(4.0, 2, 2, 2));
    train.push_back(make_cfg(4.2, 2, 2, 2));
    as.add_candidates(train);
    as.select();

    std::vector<Atoms> test;
    test.push_back(make_cfg(5.0, 2, 2, 2, 0.1));
    test.push_back(make_cfg(3.5, 2, 2, 2, 0.1));
    std::vector<double> grades;
    as.grade_all(test, grades);
    REQUIRE(grades.size() == 2);
    REQUIRE(std::isfinite(grades[0]));
}

TEST_CASE("ActiveSelector chunked grade_all matches single-config grading", "[active_sel]")
{
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> C(2, 2);
    C << 1.0, 0.0,
         0.0, 1.0;

    ActiveSelector as(new PositionRowsBuilder());
    as.weight = 1.0;
    as.pow_scale = 0.0;
    as.stream_chunk_size = 2;
    REQUIRE(as.set_active_rows_for_grading(C, {0, 1}));

    std::vector<Atoms> probes;
    probes.push_back(make_position_rows({{0.25, 0.50}, {0.10, 0.80}}));
    probes.push_back(make_position_rows({{1.25, 0.10}}));
    probes.push_back(make_position_rows({{0.00, 0.33}, {0.44, 0.11}, {0.20, 0.90}}));

    std::vector<double> expected;
    for (auto cfg : probes)
        expected.push_back(as.grade(cfg));

    std::vector<double> grades;
    as.grade_all(probes, grades);
    REQUIRE(grades.size() == expected.size());
    for (size_t i = 0; i < grades.size(); ++i)
        REQUIRE(grades[i] == Approx(expected[i]).margin(1e-12));
}

TEST_CASE("ActiveSelector chunk size and non-cloneable fallback preserve grades", "[active_sel]")
{
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> C(2, 2);
    C << 1.0, 0.0,
         0.0, 1.0;

    std::vector<Atoms> probes;
    probes.push_back(make_position_rows({{0.25, 0.50}, {0.10, 0.80}}));
    probes.push_back(make_position_rows({{1.25, 0.10}}));
    probes.push_back(make_position_rows({{0.00, 0.33}, {0.44, 0.11}, {0.20, 0.90}}));

    ActiveSelector chunk_one(new PositionRowsBuilder());
    chunk_one.weight = 1.0;
    chunk_one.pow_scale = 0.0;
    chunk_one.stream_chunk_size = 1;
    REQUIRE(chunk_one.set_active_rows_for_grading(C, {0, 1}));
    std::vector<double> ref;
    chunk_one.grade_all(probes, ref);

    ActiveSelector chunk_many(new PositionRowsBuilder());
    chunk_many.weight = 1.0;
    chunk_many.pow_scale = 0.0;
    chunk_many.stream_chunk_size = 8;
    REQUIRE(chunk_many.set_active_rows_for_grading(C, {0, 1}));
    std::vector<double> many;
    chunk_many.grade_all(probes, many);

    ActiveSelector noncloneable(new NonCloneablePositionRowsBuilder());
    noncloneable.weight = 1.0;
    noncloneable.pow_scale = 0.0;
    noncloneable.stream_chunk_size = 2;
    REQUIRE(noncloneable.set_active_rows_for_grading(C, {0, 1}));
    std::vector<double> fallback;
    noncloneable.grade_all(probes, fallback);

    REQUIRE(many.size() == ref.size());
    REQUIRE(fallback.size() == ref.size());
    for (size_t i = 0; i < ref.size(); ++i) {
        REQUIRE(many[i] == Approx(ref[i]).margin(1e-12));
        REQUIRE(fallback[i] == Approx(ref[i]).margin(1e-12));
    }
}

// ── PotEFSBuilder tests ────────────────────────────────────────────────────

TEST_CASE("PotEFSBuilder construction and coeff_count", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    // linear_only defaults to true for performance — coeff_count() returns lin_coeff_size
    REQUIRE(builder.coeff_count() == mtp.get().get_lin_coeff_size());
    REQUIRE(builder.coeff_count() < mtp.get().get_NL_coeffs_count());
    REQUIRE(builder.cutoff() == Approx(mtp.get().get_cutoff()));

    // With linear_only=false, coeff_count() returns full reg_coeff_size
    builder.linear_only = false;
    REQUIRE(builder.coeff_count() == mtp.get().get_NL_coeffs_count());
}

TEST_CASE("PotEFSBuilder build_rows produces valid energy gradient", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 1.0;

    Atoms cfg = make_cfg(4.0, 2, 2, 2, 0.03);
    int ncols = builder.coeff_count();

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    // Energy-only mode: one row per structure
    REQUIRE(nrows == 1);
    REQUIRE(B.rows() == 1);
    REQUIRE(B.cols() == ncols);

    // All gradient entries should be finite
    for (int j = 0; j < ncols; ++j)
        REQUIRE(std::isfinite(B(0, j)));

    // At least some entries should be non-zero
    double sum_abs = B.row(0).array().abs().sum();
    REQUIRE(sum_abs > 0.0);
}

TEST_CASE("PotEFSBuilder with ActiveSelector integration", "[efs_builder]")
{
    auto *builder = new PotEFSBuilder(mtp.get());
    ActiveSelector as(builder);
    as.threshold = 1.001; as.weight = 1.0; as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(3.8, 2, 2, 2, 0.05));
    cfgs.push_back(make_cfg(4.0, 2, 2, 2, 0.03));
    cfgs.push_back(make_cfg(4.2, 2, 2, 2, 0.04));

    as.add_candidates(cfgs);
    int nsel = as.select();
    REQUIRE(nsel >= 1);
    REQUIRE(as.log_determinant() > -1e30);

    // Grading a new config
    Atoms new_cfg = make_cfg(5.0, 2, 2, 2, 0.1);
    double g = as.grade(new_cfg);
    REQUIRE(std::isfinite(g));
}

TEST_CASE("PotEFSBuilder site-energy mode produces per-atom rows", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 0.0;
    builder.w_site   = 1.0;

    Atoms cfg = make_cfg(4.0, 2, 2, 2, 0.03);
    int natoms = cfg.size();

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    REQUIRE(nrows == natoms);  // one row per atom
    REQUIRE(B.rows() == natoms);
    REQUIRE(B.cols() == builder.coeff_count());

    // Each row should be finite and non-zero
    for (int i = 0; i < nrows; ++i) {
        REQUIRE(B.row(i).array().isFinite().all());
        double nrm = B.row(i).norm();
        REQUIRE(nrm > 0.0);
    }
}

TEST_CASE("PotEFSBuilder site-energy rows apply global weight scaling", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 0.0;
    builder.w_site   = 1.0;

    Atoms cfg_ref = make_cfg(4.0, 2, 2, 1, 0.03);
    Atoms cfg_scaled = cfg_ref;
    const double weight = 2.0;
    const double pow_scale = 1.0;
    const double ratio = weight / static_cast<double>(cfg_ref.size());

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_ref, B_scaled;
    int nref = builder.build_rows(cfg_ref, B_ref, 1.0, 0.0);
    int nscaled = builder.build_rows(cfg_scaled, B_scaled, weight, pow_scale);

    REQUIRE(nscaled == nref);
    REQUIRE(B_scaled.rows() == B_ref.rows());
    REQUIRE(B_scaled.cols() == B_ref.cols());
    REQUIRE(B_ref.array().abs().sum() > 0.0);
    REQUIRE((B_scaled - ratio * B_ref).norm() == Approx(0.0).margin(1e-10));
}

TEST_CASE("PotEFSBuilder MLIP scaling and row metadata", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 1.0;
    builder.w_force = 1.0;
    builder.w_stress = 1.0;
    builder.w_site = 1.0;
    builder.row_scaling = RowScalingMode::MlipCompatible;

    Atoms cfg_ref = make_cfg(4.0, 2, 2, 1, 0.03);
    Atoms cfg_scaled = cfg_ref;
    const int natoms = cfg_ref.size();
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_ref, B_scaled;
    std::vector<BRowRef> refs;
    const int nref = builder.build_rows(cfg_ref, B_ref, 1.0, 0.0, &refs);
    const int nscaled = builder.build_rows(cfg_scaled, B_scaled, 2.0, 1.0);

    REQUIRE(nref == 1 + 3 * natoms + 9 + natoms);
    REQUIRE(nscaled == nref);
    REQUIRE(refs.size() == static_cast<size_t>(nref));
    for (int i = 0; i < nref; ++i) {
        const double ratio = (refs[i].kind == BRowKind::Force ||
                              refs[i].kind == BRowKind::Site)
                           ? 2.0 : 2.0 / natoms;
        REQUIRE((B_scaled.row(i) - ratio * B_ref.row(i)).norm()
                == Approx(0.0).margin(1e-9));
        REQUIRE(refs[i].local_row == i);
    }
    REQUIRE(refs.front().kind == BRowKind::Energy);
    REQUIRE(refs[1].kind == BRowKind::Force);
    REQUIRE(refs[1].atom_index == 0);
    REQUIRE(refs[1].component == 0);
    REQUIRE(refs[1 + 3 * natoms].kind == BRowKind::Stress);
    REQUIRE(refs[1 + 3 * natoms].component == 0);
    REQUIRE(refs.back().kind == BRowKind::Site);
    REQUIRE(refs.back().atom_index == natoms - 1);
}

TEST_CASE("PotEFSBuilder mixed energy + site mode", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 1.0;
    builder.w_site   = 1.0;

    Atoms cfg = make_cfg(4.0, 2, 2, 2, 0.03);
    int natoms = cfg.size();

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    // 1 energy row + natoms site rows
    REQUIRE(nrows == 1 + natoms);
    REQUIRE(B.rows() == 1 + natoms);
    REQUIRE(B.cols() == builder.coeff_count());

    for (int i = 0; i < B.rows(); ++i) {
        REQUIRE(B.row(i).array().isFinite().all());
    }
}

TEST_CASE("PotEFSBuilder force-only mode produces component rows", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 0.0;
    builder.w_site = 0.0;
    builder.w_force = 1.0;
    builder.w_stress = 0.0;

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.04);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    REQUIRE(nrows == 3 * cfg.size());
    REQUIRE(B.rows() == nrows);
    REQUIRE(B.cols() == builder.coeff_count());
    REQUIRE(B.array().isFinite().all());
    REQUIRE(B.array().abs().sum() > 0.0);
}

TEST_CASE("PotEFSBuilder stress-only mode produces tensor rows", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 0.0;
    builder.w_site = 0.0;
    builder.w_force = 0.0;
    builder.w_stress = 1.0;

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.04);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    REQUIRE(nrows == 9);
    REQUIRE(B.rows() == 9);
    REQUIRE(B.cols() == builder.coeff_count());
    REQUIRE(B.array().isFinite().all());
    REQUIRE(B.array().abs().sum() > 0.0);
}

TEST_CASE("PotEFSBuilder mixed E/F/S rows are all materialized", "[efs_builder]")
{
    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 1.0;
    builder.w_site = 0.0;
    builder.w_force = 1.0;
    builder.w_stress = 1.0;

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.04);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    REQUIRE(nrows == 1 + 3 * cfg.size() + 9);
    REQUIRE(B.rows() == nrows);
    REQUIRE(B.cols() == builder.coeff_count());
    REQUIRE(B.array().isFinite().all());
    REQUIRE(B.row(0).array().abs().sum() > 0.0);
    REQUIRE(B.bottomRows(nrows - 1).array().abs().sum() > 0.0);
}

TEST_CASE("DescriptorBuilder Cfg mode produces one row per structure", "[desc_builder]")
{
    DescriptorMTM desc_mtm(mtp.get());
    DescriptorBuilder builder(desc_mtm);
    builder.mode = DescriptorBuilder::Cfg;

    Atoms cfg = make_cfg(4.0, 2, 2, 2, 0.03);

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder.build_rows(cfg, B, 1.0, 0.0);

    REQUIRE(nrows == 1);
    REQUIRE(B.rows() == 1);
    REQUIRE(B.cols() == desc_mtm.feature_size());

    // Cfg row = sum of per-atom descriptors
    // Verify by comparing with per-atom Nbh mode sum
    DescriptorBuilder builder_nbh(desc_mtm);
    builder_nbh.mode = DescriptorBuilder::Nbh;
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_nbh;
    int n_nbh = builder_nbh.build_rows(cfg, B_nbh, 1.0, 0.0);
    REQUIRE(n_nbh == cfg.size());

    for (int j = 0; j < B.cols(); ++j) {
        double sum = 0.0;
        for (int i = 0; i < n_nbh; ++i)
            sum += B_nbh(i, j);
        REQUIRE(B(0, j) == Approx(sum).margin(1e-10));
    }
}

TEST_CASE("DescriptorBuilder Cfg mode integrated with ActiveSelector", "[desc_builder]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    builder->mode = DescriptorBuilder::Cfg;

    ActiveSelector as(builder);
    as.threshold = 1.001; as.weight = 1.0; as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(3.8, 2, 2, 2, 0.05));
    cfgs.push_back(make_cfg(4.0, 2, 2, 2, 0.03));
    cfgs.push_back(make_cfg(4.2, 2, 2, 2, 0.04));
    as.add_candidates(cfgs);

    int nsel = as.select();
    REQUIRE(nsel >= 1);
    REQUIRE(as.log_determinant() > -1e30);

    Atoms new_cfg = make_cfg(5.0, 2, 2, 2, 0.1);
    double g = as.grade(new_cfg);
    REQUIRE(std::isfinite(g));
    REQUIRE(g >= 0.0);
}

TEST_CASE("BMatrixBuilder load_config restores DescriptorBuilder from potfile override", "[desc_builder]")
{
    const std::string tmpfile = "/tmp/active_selector_desc_builder_config.mtp";
    mtp.get().Save(tmpfile);

    DescriptorMTM desc_mtm(mtp.get());
    DescriptorBuilder builder(desc_mtm);
    builder.mode = DescriptorBuilder::Cfg;
    builder.row_scaling = RowScalingMode::MlipCompatible;

    std::stringstream ss;
    builder.save_config(ss);
    REQUIRE_FALSE(BMatrixBuilder::load_config(ss));
    ss.clear();
    ss.seekg(0);
    auto loaded = BMatrixBuilder::load_config(ss, tmpfile);
    REQUIRE(loaded);
    REQUIRE(loaded->coeff_count() == builder.coeff_count());
    REQUIRE(loaded->row_scaling == RowScalingMode::MlipCompatible);

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.01);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = loaded->build_rows(cfg, B, 1.0, 0.0);
    REQUIRE(nrows == 1);
    REQUIRE(B.rows() == 1);
    REQUIRE(B.cols() == builder.coeff_count());
    REQUIRE(B.row(0).array().isFinite().all());

    std::remove(tmpfile.c_str());
}

TEST_CASE("make_builder owns MTM descriptor dependencies", "[desc_builder]")
{
    const std::string tmpfile = "/tmp/active_selector_make_builder_mtm.mtp";
    mtp.get().Save(tmpfile);

    GPSettings settings;
    auto builder = make_builder("mtm", tmpfile, "mtp", settings, "cfg",
                                1.0, 0.0, 0.0, 0.0);
    REQUIRE(builder);

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.01);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = builder->build_rows(cfg, B, 1.0, 0.0);
    REQUIRE(nrows == 1);
    REQUIRE(B.rows() == 1);
    REQUIRE(B.cols() == builder->coeff_count());
    REQUIRE(B.row(0).array().isFinite().all());

    auto cloned = builder->clone();
    REQUIRE(cloned);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_clone;
    int clone_rows = cloned->build_rows(cfg, B_clone, 1.0, 0.0);
    REQUIRE(clone_rows == nrows);
    REQUIRE(B_clone.rows() == B.rows());
    REQUIRE(B_clone.cols() == B.cols());
    REQUIRE(B_clone.row(0).array().isFinite().all());

    std::remove(tmpfile.c_str());
}

TEST_CASE("Descriptor builders reject unsupported types", "[desc_builder]")
{
    GPSettings settings;
    for (const std::string dtype : {"soap", "unknown_descriptor"}) {
        INFO(dtype);
        // An unsupported descriptor must fail before attempting to load a
        // potential for an unrelated fallback descriptor.
        REQUIRE_FALSE(make_builder(dtype, "missing.mtp", "mtp", settings,
                                   "cfg", 1.0, 0.0, 0.0, 0.0));

        DescriptorBondOrder desc(settings);
        DescriptorBuilder builder(desc);
        builder.set_config_metadata(dtype, "cfg", settings);
        std::stringstream state;
        builder.save_config(state);
        REQUIRE_FALSE(BMatrixBuilder::load_config(state, "missing.mtp"));
    }
}

TEST_CASE("BMatrixBuilder load_config rejects truncated DescriptorBuilder config", "[desc_builder]")
{
    const std::string tmpfile = "/tmp/active_selector_desc_builder_truncated_config.mtp";
    mtp.get().Save(tmpfile);

    DescriptorMTM desc_mtm(mtp.get());
    DescriptorBuilder builder(desc_mtm);
    builder.mode = DescriptorBuilder::Cfg;

    std::stringstream ss;
    builder.save_config(ss);
    std::string raw = ss.str();
    REQUIRE(raw.size() > 1);
    raw.pop_back();

    std::stringstream truncated(raw);
    REQUIRE_FALSE(BMatrixBuilder::load_config(truncated, tmpfile));

    std::remove(tmpfile.c_str());
}

TEST_CASE("BMatrixBuilder load_config restores non-MTM descriptor", "[desc_builder]")
{
    GPSettings settings;
    settings.Modify("bondorder:cutoff", "5.0");
    settings.Modify("bondorder:maxl", "4");
    DescriptorBondOrder bond(4);
    DescriptorBuilder builder(bond);
    builder.set_config_metadata("bondorder", "cfg", settings);

    std::stringstream ss;
    builder.save_config(ss);
    auto loaded = BMatrixBuilder::load_config(ss, "/tmp/nonexistent.mtp");
    REQUIRE(loaded);
    REQUIRE(loaded->coeff_count() == builder.coeff_count());

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.01);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = loaded->build_rows(cfg, B, 1.0, 0.0);
    REQUIRE(nrows == 1);
    REQUIRE(B.rows() == 1);
    REQUIRE(B.cols() == builder.coeff_count());
    REQUIRE(B.row(0).array().isFinite().all());
}

TEST_CASE("BMatrixBuilder load_config restores PotEFSBuilder weights", "[efs_builder]")
{
    const std::string tmpfile = "/tmp/active_selector_potefs_builder_config.mtp";
    mtp.get().Save(tmpfile);

    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 1.0;
    builder.w_site = 1.0;
    builder.linear_only = true;
    builder.row_scaling = RowScalingMode::MlipCompatible;

    std::stringstream ss;
    builder.save_config(ss);
    REQUIRE_FALSE(BMatrixBuilder::load_config(ss));
    ss.clear();
    ss.seekg(0);
    auto loaded = BMatrixBuilder::load_config(ss, tmpfile);
    REQUIRE(loaded);
    REQUIRE(loaded->coeff_count() == builder.coeff_count());
    REQUIRE(loaded->row_scaling == RowScalingMode::MlipCompatible);

    Atoms cfg = make_cfg(4.0, 2, 2, 1, 0.01);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
    int nrows = loaded->build_rows(cfg, B, 1.0, 0.0);
    REQUIRE(nrows == 1 + cfg.size());
    REQUIRE(B.rows() == 1 + cfg.size());
    REQUIRE(B.cols() == builder.coeff_count());
    for (int i = 0; i < B.rows(); ++i)
        REQUIRE(B.row(i).array().isFinite().all());

    std::remove(tmpfile.c_str());
}

TEST_CASE("BMatrixBuilder load_config rejects truncated PotEFSBuilder config", "[efs_builder]")
{
    const std::string tmpfile = "/tmp/active_selector_potefs_truncated_config.mtp";
    mtp.get().Save(tmpfile);

    PotEFSBuilder builder(mtp.get());
    builder.w_energy = 1.0;
    builder.w_site = 1.0;

    std::stringstream ss;
    builder.save_config(ss);
    std::string raw = ss.str();
    REQUIRE(raw.size() > 1);
    raw.pop_back();

    std::stringstream truncated(raw);
    REQUIRE_FALSE(BMatrixBuilder::load_config(truncated, tmpfile));

    std::remove(tmpfile.c_str());
}

TEST_CASE("ActiveSelector v7 state preserves selected config indices", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    ActiveSelector as(builder);
    as.threshold = 1.001; as.weight = 1.0; as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    cfgs.push_back(make_cfg(4.2, 2, 2, 2, 0.03));
    cfgs.push_back(make_cfg(4.4, 3, 2, 1, 0.04));

    as.add_candidates(cfgs);
    REQUIRE(as.select() >= 1);
    const auto expected = as.selected_indices();
    REQUIRE(!expected.empty());

    const std::string tmpfile = "/tmp/active_selector_v7_state.sel";
    as.save_state(tmpfile);

    ActiveSelector loaded;
    loaded.select_mode = "online";
    loaded.load_state(tmpfile);
    REQUIRE(loaded.num_candidates() == static_cast<int>(cfgs.size()));
    REQUIRE(loaded.selected_indices() == expected);
    REQUIRE(loaded.online_capable());

    std::remove(tmpfile.c_str());
}

TEST_CASE("ActiveSelector load_state rejects out-of-range selected rows", "[active_sel]")
{
    const std::string tmpfile = "/tmp/active_selector_bad_selected_row.sel";
    {
        std::ofstream ofs(tmpfile, std::ios::binary);
        const int magic = 0x41435456;
        const int version = 10;
        const int ncoeff = 2;
        const double threshold = 1.001;
        const double weight = 1.0;
        const double pow_scale = 0.0;
        const int nsel = 0;
        const int k = 1;
        const int m_full = 1;
        const int bad_selected_row = 2;

        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
        ofs.write(reinterpret_cast<const char*>(&ncoeff), sizeof(ncoeff));
        ofs.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
        ofs.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        ofs.write(reinterpret_cast<const char*>(&pow_scale), sizeof(pow_scale));
        const int scaling = 0, init_mode = 0, reason_size = 0;
        const double init_value = 1e-6, max_grade = 1.0;
        const char converged = 0;
        ofs.write(reinterpret_cast<const char*>(&scaling), sizeof(scaling));
        ofs.write(reinterpret_cast<const char*>(&init_mode), sizeof(init_mode));
        ofs.write(reinterpret_cast<const char*>(&init_value), sizeof(init_value));
        ofs.write(&converged, sizeof(converged));
        ofs.write(reinterpret_cast<const char*>(&max_grade), sizeof(max_grade));
        ofs.write(reinterpret_cast<const char*>(&reason_size), sizeof(reason_size));
        ofs.write(reinterpret_cast<const char*>(&nsel), sizeof(nsel));
        ofs.write(reinterpret_cast<const char*>(&k), sizeof(k));
        ofs.write(reinterpret_cast<const char*>(&m_full), sizeof(m_full));
        const int refs[6] = {0, 0, bad_selected_row, 0, -1, -1};
        ofs.write(reinterpret_cast<const char*>(refs), sizeof(refs));
    }

    ActiveSelector loaded;
    REQUIRE_THROWS(loaded.load_state(tmpfile));
    REQUIRE(loaded.num_candidates() == 0);
    REQUIRE(loaded.selected_indices().empty());

    std::remove(tmpfile.c_str());
}

TEST_CASE("ActiveSelector failed load_state preserves existing builder dimensions", "[active_sel]")
{
    const std::string tmpfile = "/tmp/active_selector_bad_ncoeff.sel";
    {
        std::ofstream ofs(tmpfile, std::ios::binary);
        const int magic = 0x41435456;
        const int version = 10;
        const int bad_ncoeff = 2;
        const double threshold = 1.001;
        const double weight = 1.0;
        const double pow_scale = 0.0;
        const int nsel = -1;

        ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
        ofs.write(reinterpret_cast<const char*>(&bad_ncoeff), sizeof(bad_ncoeff));
        ofs.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
        ofs.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
        ofs.write(reinterpret_cast<const char*>(&pow_scale), sizeof(pow_scale));
        const int scaling = 0, init_mode = 0, reason_size = 0;
        const double init_value = 1e-6, max_grade = 1.0;
        const char converged = 0;
        ofs.write(reinterpret_cast<const char*>(&scaling), sizeof(scaling));
        ofs.write(reinterpret_cast<const char*>(&init_mode), sizeof(init_mode));
        ofs.write(reinterpret_cast<const char*>(&init_value), sizeof(init_value));
        ofs.write(&converged, sizeof(converged));
        ofs.write(reinterpret_cast<const char*>(&max_grade), sizeof(max_grade));
        ofs.write(reinterpret_cast<const char*>(&reason_size), sizeof(reason_size));
        ofs.write(reinterpret_cast<const char*>(&nsel), sizeof(nsel));
    }

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    cfgs.push_back(make_cfg(4.2, 2, 2, 1, 0.03));
    cfgs.push_back(make_cfg(4.4, 2, 2, 1, 0.04));
    cfgs.push_back(make_cfg(4.6, 2, 2, 1, 0.05));

    DescriptorMTM desc_ref(mtp.get());
    auto *builder_ref = new DescriptorBuilder(desc_ref);
    builder_ref->mode = DescriptorBuilder::Cfg;
    ActiveSelector ref(builder_ref);
    ref.weight = 1.0;
    ref.pow_scale = 0.0;
    ref.add_candidates(cfgs);
    int expected = ref.select();

    DescriptorMTM desc_loaded(mtp.get());
    auto *builder_loaded = new DescriptorBuilder(desc_loaded);
    builder_loaded->mode = DescriptorBuilder::Cfg;
    ActiveSelector loaded(builder_loaded);
    loaded.weight = 1.0;
    loaded.pow_scale = 0.0;
    REQUIRE_THROWS(loaded.load_state(tmpfile));
    loaded.add_candidates(cfgs);
    int actual = loaded.select();

    REQUIRE(actual == expected);
    REQUIRE(loaded.selected_indices() == ref.selected_indices());

    std::remove(tmpfile.c_str());
}

TEST_CASE("ActiveSelector load_state rejects selected count mismatch", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    builder->mode = DescriptorBuilder::Cfg;

    ActiveSelector as(builder);
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    cfgs.push_back(make_cfg(4.2, 2, 2, 1, 0.03));
    cfgs.push_back(make_cfg(4.4, 2, 2, 1, 0.04));

    as.add_candidates(cfgs);
    REQUIRE(as.select() >= 1);

    const std::string tmpfile = "/tmp/active_selector_bad_nsel.sel";
    as.save_state(tmpfile);

    {
        std::fstream fs(tmpfile, std::ios::in | std::ios::out | std::ios::binary);
        REQUIRE(fs);
        const int bad_nsel = static_cast<int>(cfgs.size()) + 1;
        int integer = 0;
        double real = 0.0;
        char flag = 0;
        fs.read(reinterpret_cast<char*>(&integer), sizeof(int)); // magic
        fs.read(reinterpret_cast<char*>(&integer), sizeof(int)); // version
        fs.read(reinterpret_cast<char*>(&integer), sizeof(int)); // ncoeff
        fs.read(reinterpret_cast<char*>(&real), sizeof(double)); // threshold
        fs.read(reinterpret_cast<char*>(&real), sizeof(double)); // weight
        fs.read(reinterpret_cast<char*>(&real), sizeof(double)); // pow_scale
        fs.read(reinterpret_cast<char*>(&integer), sizeof(int)); // row scaling
        fs.read(reinterpret_cast<char*>(&integer), sizeof(int)); // init mode
        fs.read(reinterpret_cast<char*>(&real), sizeof(double)); // init value
        fs.read(&flag, 1);                                       // converged
        fs.read(reinterpret_cast<char*>(&real), sizeof(double)); // max grade
        int reason_size = 0;
        fs.read(reinterpret_cast<char*>(&reason_size), sizeof(int));
        REQUIRE(reason_size >= 0);
        fs.seekg(reason_size, std::ios::cur);
        const std::streampos nsel_offset = fs.tellg();
        fs.seekp(nsel_offset);
        fs.write(reinterpret_cast<const char*>(&bad_nsel), sizeof(bad_nsel));
        REQUIRE(fs);
    }

    ActiveSelector loaded;
    REQUIRE_THROWS(loaded.load_state(tmpfile));
    REQUIRE(loaded.num_candidates() == 0);
    REQUIRE(loaded.selected_indices().empty());

    std::remove(tmpfile.c_str());
}

TEST_CASE("ActiveSelector v7 state rebuild restores compute_grades input", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    builder->mode = DescriptorBuilder::Cfg;

    ActiveSelector as(builder);
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    cfgs.push_back(make_cfg(4.2, 2, 2, 1, 0.03));
    cfgs.push_back(make_cfg(4.4, 2, 2, 1, 0.04));
    cfgs.push_back(make_cfg(4.6, 2, 2, 1, 0.05));

    as.add_candidates(cfgs);
    REQUIRE(as.select() >= 1);
    std::vector<double> expected;
    as.compute_grades(expected);
    REQUIRE(expected.size() == cfgs.size());

    const std::string tmpfile = "/tmp/active_selector_v7_rebuild_grades.sel";
    as.save_state(tmpfile);

    ActiveSelector loaded;
    loaded.load_state(tmpfile);
    loaded.init_builder(mtp.get());
    loaded.rebuild_B_from_configs();

    std::vector<double> actual;
    loaded.compute_grades(actual);
    REQUIRE(actual.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        REQUIRE(std::isfinite(actual[i]));
        REQUIRE(actual[i] == Approx(expected[i]).margin(1e-8));
    }

    std::remove(tmpfile.c_str());
}

TEST_CASE("ActiveSelector v7 state restores PotEFSBuilder grading", "[efs_builder]")
{
    auto *builder = new PotEFSBuilder(mtp.get());
    builder->w_energy = 1.0;
    builder->w_site = 1.0;
    builder->linear_only = true;

    ActiveSelector as(builder);
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    cfgs.push_back(make_cfg(4.2, 2, 2, 1, 0.03));
    cfgs.push_back(make_cfg(4.4, 2, 2, 1, 0.04));

    as.add_candidates(cfgs);
    REQUIRE(as.select() >= 1);

    const std::string state_file = "/tmp/active_selector_potefs_v7_state.sel";
    as.save_state(state_file);

    ActiveSelector loaded;
    loaded.load_state(state_file);
    REQUIRE(loaded.num_candidates() == static_cast<int>(cfgs.size()));
    REQUIRE(loaded.init_mode == "mlip");
    REQUIRE(loaded.init_value == Approx(1e-6));
    REQUIRE(loaded.virtual_row_count() > 0);
    REQUIRE(loaded.converged() == as.converged());
    REQUIRE(loaded.max_remaining_grade()
            == Approx(as.max_remaining_grade()).margin(1e-12));
    loaded.init_builder(mtp.get());

    Atoms probe = make_cfg(4.6, 2, 2, 1, 0.05);
    double grade = loaded.grade(probe);
    REQUIRE(std::isfinite(grade));
    REQUIRE(grade >= 0.0);

    std::remove(state_file.c_str());
}

TEST_CASE("ActiveSelector v7 state restores non-MTM descriptor builder", "[desc_builder]")
{
    GPSettings settings;
    settings.Modify("bondorder:cutoff", "5.0");
    settings.Modify("bondorder:maxl", "4");

    auto desc = std::make_unique<DescriptorBondOrder>(
        settings.ExtractSubSettings("bondorder"));
    auto *builder = new DescriptorBuilder(std::move(desc));
    builder->set_config_metadata("bondorder", "cfg", settings);

    ActiveSelector as(builder);
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    cfgs.push_back(make_cfg(4.2, 2, 2, 1, 0.03));
    cfgs.push_back(make_cfg(4.4, 2, 2, 1, 0.04));

    as.add_candidates(cfgs);
    REQUIRE(as.select() >= 1);

    const std::string state_file = "/tmp/active_selector_bondorder_v7_state.sel";
    as.save_state(state_file);

    ActiveSelector loaded;
    loaded.load_state(state_file);
    REQUIRE(loaded.num_candidates() == static_cast<int>(cfgs.size()));
    loaded.init_builder(mtp.get());
    loaded.rebuild_B_from_configs();

    Atoms probe = make_cfg(4.6, 2, 2, 1, 0.05);
    double grade = loaded.grade(probe);
    REQUIRE(std::isfinite(grade));
    REQUIRE(grade >= 0.0);

    std::remove(state_file.c_str());
}

TEST_CASE("ActiveSelector dump_state reports config and local row counts", "[active_sel]")
{
    DescriptorMTM desc_mtm(mtp.get());
    auto *builder = new DescriptorBuilder(desc_mtm);
    builder->mode = DescriptorBuilder::Cfg;

    ActiveSelector as(builder);
    as.threshold = 1.001;
    as.weight = 1.0;
    as.pow_scale = 0.0;

    std::vector<Atoms> cfgs;
    cfgs.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    cfgs.push_back(make_cfg(4.2, 2, 2, 1, 0.03));
    cfgs.push_back(make_cfg(4.4, 2, 2, 1, 0.04));

    as.add_candidates(cfgs);
    REQUIRE(as.select() >= 1);

    const std::string state_file = "/tmp/active_selector_dump_v7_state.sel";
    as.save_state(state_file);

    std::ostringstream out;
    ActiveSelector::dump_state(state_file, out);
    std::string text = out.str();
    REQUIRE(text.find("train_cfgs:  3") != std::string::npos);
    REQUIRE(text.find("sel_cfgs:    ") != std::string::npos);
    REQUIRE(text.find("local_rows:  3") != std::string::npos);
    REQUIRE(text.find("rows/cfg:") != std::string::npos);

    std::remove(state_file.c_str());
}

TEST_CASE("ActiveSelector online state resumes with only appended configs", "[active_sel]")
{
    DescriptorMTM desc_a(mtp.get());
    auto *builder_a = new DescriptorBuilder(desc_a);
    builder_a->mode = DescriptorBuilder::Cfg;
    builder_a->row_scaling = RowScalingMode::MlipCompatible;

    ActiveSelector uninterrupted(builder_a);
    uninterrupted.select_mode = "online";
    uninterrupted.threshold = 1.001;
    uninterrupted.weight = 1.0;
    uninterrupted.pow_scale = 0.5;
    std::vector<Atoms> initial;
    initial.push_back(make_cfg(4.0, 2, 2, 1, 0.02));
    initial.push_back(make_cfg(4.2, 2, 2, 1, 0.03));
    initial.push_back(make_cfg(4.4, 2, 2, 1, 0.04));
    uninterrupted.add_candidates_unbuilt(initial);
    REQUIRE(uninterrupted.select_online() >= 1);
    REQUIRE(uninterrupted.online_capable());

    const std::string state_file = "/tmp/active_selector_online_v10.sel";
    uninterrupted.save_state(state_file);

    DescriptorMTM desc_b(mtp.get());
    auto *builder_b = new DescriptorBuilder(desc_b);
    builder_b->mode = DescriptorBuilder::Cfg;
    ActiveSelector resumed(builder_b);
    resumed.select_mode = "online";
    resumed.load_state(state_file);
    REQUIRE(resumed.online_capable());

    std::vector<Atoms> added;
    added.push_back(make_cfg(4.8, 2, 2, 1, 0.08));
    uninterrupted.add_candidates_unbuilt(added);
    resumed.add_candidates_unbuilt(added);
    REQUIRE(resumed.select_online() == uninterrupted.select_online());
    REQUIRE(resumed.num_candidates() == 4);
    REQUIRE(resumed.selected_row_refs().size() ==
            uninterrupted.selected_row_refs().size());
    for (const auto &ref : resumed.selected_row_refs())
        REQUIRE(ref.kind == BRowKind::Descriptor);

    Atoms probe = make_cfg(4.6, 2, 2, 1, 0.06);
    REQUIRE(resumed.grade(probe) ==
            Approx(uninterrupted.grade(probe)).margin(1e-10));

    std::ostringstream dump;
    ActiveSelector::dump_state(state_file, dump);
    REQUIRE(dump.str().find("version:     10") != std::string::npos);
    REQUIRE(dump.str().find("online:      yes") != std::string::npos);
    REQUIRE(dump.str().find("row_scaling: mlip") != std::string::npos);
    std::remove(state_file.c_str());
}

TEST_CASE("PotEFSBuilder site mode grading picks max over atoms", "[efs_builder]")
{
    auto *builder = new PotEFSBuilder(mtp.get());
    builder->w_energy = 0.0;
    builder->w_site   = 1.0;

    ActiveSelector as(builder);
    as.threshold = 1.001; as.weight = 1.0; as.pow_scale = 0.0;

    // Train with a few configs
    std::vector<Atoms> train;
    train.push_back(make_cfg(4.0, 2, 2, 2));
    train.push_back(make_cfg(4.2, 2, 2, 2));
    as.add_candidates(train);
    as.select();

    // Grade an extrapolative config — grade should be max over all atom rows
    Atoms new_cfg = make_cfg(5.0, 2, 2, 2, 0.1);
    double g = as.grade(new_cfg);
    REQUIRE(std::isfinite(g));
    REQUIRE(g >= 0.0);
}

TEST_CASE("Online AL rejects unsupported modes instead of silently disabling", "[active_sel][al]")
{
    GPSettings settings;
    REQUIRE_FALSE(AL::ALPotential::create(settings));
    settings.Modify("mode", "none");
    REQUIRE_FALSE(AL::ALPotential::create(settings));
    const std::string mode = GENERATE("committee", "unsupported-mode");
    settings.Modify("mode", mode);
    REQUIRE_THROWS_AS(AL::ALPotential::create(settings), std::invalid_argument);
}

TEST_CASE("Online MaxVol grading and pool collection survive a state roundtrip", "[active_sel][al]")
{
    DescriptorMTM descriptor(mtp.get());
    auto *builder = new DescriptorBuilder(descriptor);
    builder->mode = DescriptorBuilder::Cfg;
    ActiveSelector selector(builder);
    selector.threshold = 1.001;
    selector.weight = 1.0;
    selector.pow_scale = 0.0;
    std::vector<Atoms> candidates;
    for (int i = 0; i < 4; ++i)
        candidates.push_back(make_cfg(4.0 + 0.2 * i, 2, 2, 1, 0.02));
    selector.add_candidates(candidates);
    REQUIRE(selector.select() >= 1);
    const std::string state = "online_maxvol_roundtrip.sel";
    selector.save_state(state);

    GPSettings settings;
    settings.Modify("mode", "maxvol");
    settings.Modify("state_file", state);
    settings.Modify("threshold", "0");
    settings.Modify("threshold_break", "1e12");
    auto al = AL::ALPotential::create(settings);
    REQUIRE(al);
    REQUIRE(al->active());
    al->init_builder(mtp.get());
    auto candidate = candidates.front();
    const auto grade = al->evaluate(candidate).uncertainty;
    REQUIRE(std::isfinite(grade));
    REQUIRE(grade == Approx(selector.grade(candidate)).margin(1e-8));
    REQUIRE_FALSE(al->check(candidate));
    REQUIRE(al->pool_size() == 1);
    std::remove(state.c_str());
}

TEST_CASE("ActiveSelector full data reselection discards an old online cache",
          "[active_sel][cache]")
{
    ActiveSelector selector(new CountedRowBuilder());
    selector.init_mode = "data";
    selector.pow_scale = 0.0;
    std::vector<Atoms> initial{
        make_row_cfg(1.0, 0.0), make_row_cfg(0.0, 1.0),
        make_row_cfg(0.5, 0.5)};
    REQUIRE(selector.select_streaming(initial) == 2);
    REQUIRE(selector.online_capable());

    selector.select_mode = "full";
    selector.rebuild_B_from_configs();
    Atoms added = make_row_cfg(2.0, 0.0);
    selector.add_candidate(added);
    REQUIRE(selector.select() == 2);
    REQUIRE(selector.prepare_grading());
    REQUIRE_FALSE(selector.online_capable());
    REQUIRE(selector.converged());
    REQUIRE(selector.grade(initial[0]) == Approx(0.5).margin(1e-12));
    REQUIRE(selector.grade(added) == Approx(1.0).margin(1e-12));
}


TEST_CASE("ActiveSelector online updates only build appended configurations",
          "[active_sel][online][stream_cache]")
{
    auto calls = std::make_shared<std::atomic<int>>(0);
    ActiveSelector selector(new CountedRowBuilder(calls));
    selector.select_mode = "online";
    selector.pow_scale = 0.0;
    std::vector<Atoms> initial{
        make_row_cfg(1.0, 0.0), make_row_cfg(0.0, 1.0),
        make_row_cfg(1.0, 1.0)};
    selector.add_candidates_unbuilt(initial);
    REQUIRE(selector.select() == 2);
    REQUIRE(calls->load() == 3);
    REQUIRE(selector.stream_stats().resident_bytes == 0);
    REQUIRE(selector.stream_stats().mapped_bytes == 0);

    calls->store(0);
    std::vector<Atoms> appended{make_row_cfg(2.0, -2.0)};
    selector.add_candidates_unbuilt(appended);
    REQUIRE(selector.select_online() == 2);
    REQUIRE(calls->load() == 1);
    REQUIRE_FALSE(selector.converged());
    REQUIRE(std::isinf(selector.max_remaining_grade()));
    REQUIRE(selector.termination_reason() == "online-approximate");

    calls->store(0);
    REQUIRE(selector.select() == 2);
    REQUIRE(calls->load() == 0);
    REQUIRE(selector.termination_reason() == "online-approximate");

    // Explicit historical grading is still available when requested.
    std::vector<double> grades;
    selector.compute_grades(grades);
    REQUIRE(calls->load() == 4);
    REQUIRE(grades.size() == 4);
}


TEST_CASE("ActiveSelector scores one large configuration across workers",
          "[active_sel][stream_cache][openmp]")
{
    OmpThreadRestore restore_threads;
#ifdef USE_OMP
    omp_set_num_threads(4);
#endif
    Atoms cfg;
    cfg.resize(10000);
    for (auto &pos : cfg.pos_) pos << 0.5, 0.5, 0.0;
    cfg.pos_[0] << 1.0, 0.0, 0.0;
    cfg.pos_[1] << 0.0, 1.0, 0.0;
    std::vector<Atoms> pool{cfg};
    for (const std::string mode : {"exact", "fast"}) {
        ActiveSelector selector(new PositionRowsBuilder());
        selector.stream_mode = mode;
        selector.stream_cache_mode = "memory";
        selector.pow_scale = 0.0;
        REQUIRE(selector.select_streaming(pool) == 1);
        REQUIRE(selector.converged());
        REQUIRE(selector.max_remaining_grade() == Approx(1.0).margin(1e-12));
        // One pass finds no exchange; the final verification is also counted.
        REQUIRE(selector.stream_stats().score_passes == 2);
#ifdef USE_OMP
        REQUIRE(selector.stream_stats().score_threads > 1);
#endif
    }
}


TEST_CASE("ActiveSelector online initialization resumes after reaching its swap cap",
          "[active_sel][online][mlip_init]")
{
    auto calls = std::make_shared<std::atomic<int>>(0);
    ActiveSelector selector(new CountedRowBuilder(calls));
    selector.select_mode = "online";
    selector.init_mode = "mlip";
    selector.max_swaps = 1;
    selector.pow_scale = 0.0;
    std::vector<Atoms> pool{make_row_cfg(2.0, 0.0), make_row_cfg(0.0, 3.0)};
    selector.add_candidates_unbuilt(pool);
    REQUIRE(selector.select_online() == 1);
    REQUIRE(selector.termination_reason() == "max-swaps");
    REQUIRE(selector.virtual_row_count() == 1);
    REQUIRE(selector.select_online() == 2);
    REQUIRE(selector.virtual_row_count() == 0);
    REQUIRE(selector.grade(pool[0]) <= 1.001 + 1e-10);
    REQUIRE(selector.grade(pool[1]) <= 1.001 + 1e-10);
}


TEST_CASE("PotEFSBuilder bulk linear EFS matches component gradients",
          "[efs_builder][bulk]")
{
    Potentials::qmtp_config options;
    options.species_count = 2;
    std::unique_ptr<Potentials::PotMTP> potential(
        Potentials::quick_mtp_3b_s14_2f0ee25e(&options));
    Atoms cfg = make_cfg(3.0, 2, 2, 2, 0.07);
    for (int i = 0; i < cfg.size(); ++i) cfg.chemmap_[i] = i % 2;
    Neighborhoods nbhs(cfg, potential->cutoff);

    for (auto scaling : {RowScalingMode::Uniform, RowScalingMode::MlipCompatible}) {
        for (const auto &weights : {std::array{1.3, -0.4, 0.8},
                                    std::array{0.0, 1.0, 0.0},
                                    std::array{0.0, 0.0, 1.0}}) {
            PotEFSBuilder builder(*potential);
            builder.w_energy = weights[0];
            builder.w_force = weights[1];
            builder.w_stress = weights[2];
            builder.w_site = 0.7;
            builder.row_scaling = scaling;
            const int ncols = builder.coeff_count();
            BMatrixBuilder::RowMatrix actual = BMatrixBuilder::RowMatrix::Constant(2, ncols, 17.0);
            std::vector<BRowRef> refs;
            const double global_weight = 2.3, power = 0.5;
            const int count = builder.build_rows(cfg, actual, global_weight, power, &refs);
            REQUIRE(count == builder.row_count(cfg));
            REQUIRE(refs.size() == static_cast<std::size_t>(count));
            REQUIRE(actual.topRows(2).isConstant(17.0));
            const double cfg_scale = global_weight / std::pow(cfg.size(), power);
            const double local_scale = scaling == RowScalingMode::Uniform ? cfg_scale : global_weight;
            for (int row = 0; row < count; ++row) {
                const auto &ref = refs[row];
                Eigen::VectorXd expected = Eigen::VectorXd::Zero(ncols);
                Eigen::Matrix<double, -1, 3> dF = Eigen::Matrix<double, -1, 3>::Zero(cfg.size(), 3);
                Eigen::Matrix3d dS = Eigen::Matrix3d::Zero();
                double dE = 0.0, scale = cfg_scale;
                if (ref.kind == BRowKind::Energy) dE = builder.w_energy;
                if (ref.kind == BRowKind::Force) {
                    dF(ref.atom_index, ref.component) = builder.w_force;
                    scale = local_scale;
                }
                if (ref.kind == BRowKind::Stress)
                    dS(ref.component / 3, ref.component % 3) = builder.w_stress;
                if (ref.kind == BRowKind::Site) {
                    potential->AccumulateFPGardLonly(
                        nbhs[ref.atom_index], expected.data(), builder.w_site, nullptr);
                    scale = local_scale;
                } else {
                    potential->AccumulateEFSCombinationGardLonly(
                        cfg, nbhs, dE, dF, dS, expected.data());
                }
                CAPTURE(row, static_cast<int>(ref.kind), static_cast<int>(scaling));
                REQUIRE((actual.row(row + 2).transpose() - scale * expected).norm()
                        <= 1e-10 * std::max(1.0, expected.norm()));
                REQUIRE(ref.local_row == row);
                REQUIRE(ref.global_row == row + 2);
            }
        }
    }
}


TEST_CASE("make_builder rejects unknown descriptor types", "[desc_builder]")
{
    const std::string dtype = GENERATE("unknown", "sopa", "");
    const std::string potfile = GENERATE("unitest/test-data/potential.mtp", "");
    CAPTURE(dtype, potfile);

    GPSettings settings;
    BMatrixBuilderSettings builder_settings;
    builder_settings.dtype = dtype;
    builder_settings.potfile = potfile;
    builder_settings.pottype = "mtp";

    REQUIRE_FALSE(make_builder(builder_settings, settings));
    REQUIRE_FALSE(make_builder(dtype, potfile, "mtp", settings, "nbh",
                               1.0, 0.0, 0.0, 0.0));
}

TEST_CASE("MaxVol readers reject obsolete state versions", "[active_sel][state]") {
    const int version = GENERATE(7, 8, 9);
    const std::string path = "/tmp/imrmlp_obsolete_maxvol_state_" + std::to_string(version);
    {
        std::ofstream out(path, std::ios::binary);
        const int magic = 0x41435456;
        out.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
        out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    }
    ActiveSelector selector;
    REQUIRE_THROWS(selector.load_state(path));
    std::ostringstream summary;
    REQUIRE_THROWS(ActiveSelector::dump_state(path, summary));
    REQUIRE_THROWS(ActiveSelector::export_configs(path, path + ".xyz", false));
    REQUIRE(summary.str().empty());
    std::remove(path.c_str());
}

TEST_CASE("ActiveSelector grades a purely virtual active set", "[active_sel]")
{
    ActiveSelector selector(new PositionRowsBuilder());
    selector.select_mode = "online";
    selector.init_mode = "mlip";
    selector.init_value = 10.0;
    Atoms cfg;
    cfg.cell_ = Eigen::Matrix3d::Identity() * 10.0;
    cfg.pos_.push_back({2.0, 3.0, 0.0});
    cfg.chemmap_.push_back(0);
    std::vector<Atoms> candidates{cfg};
    selector.add_candidates_unbuilt(candidates);
    REQUIRE(selector.select_online() == 0);
    REQUIRE(selector.get_selected().empty());
    REQUIRE(selector.virtual_row_count() == 2);
    std::vector<double> grades;
    selector.grade_all(candidates, grades);
    REQUIRE(grades[0] == Approx(0.3));
    REQUIRE(selector.grade(cfg) == Approx(grades[0]));
}
