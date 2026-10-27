// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "maxvol/active_selector.h"
#include "potentials/model_bundle.h"
#include "potentials/gpm_codec.h"
#include "active_learn/gpm_al.h"
#include "io/gpio.h"
#include "potentials/potential_mtp.h"
#include "descriptors/descriptor_MTM.h"
#include "potentials/potential_wrapper.h"
#include <Eigen/Cholesky>
#include <Eigen/LU>
#include <fmt/format.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#ifdef USE_OMP
#include <omp.h>
#endif
#ifdef EIGEN_USE_MKL_ALL
#include <mkl_service.h>
#endif

namespace {

class MklLocalThreadGuard
{
public:
    explicit MklLocalThreadGuard(int threads)
    {
#ifdef EIGEN_USE_MKL_ALL
        previous_ = mkl_set_num_threads_local(std::max(1, threads));
#else
        (void)threads;
#endif
    }

    ~MklLocalThreadGuard()
    {
#ifdef EIGEN_USE_MKL_ALL
        mkl_set_num_threads_local(previous_);
#endif
    }

    MklLocalThreadGuard(const MklLocalThreadGuard &) = delete;
    MklLocalThreadGuard &operator=(const MklLocalThreadGuard &) = delete;

private:
#ifdef EIGEN_USE_MKL_ALL
    int previous_ = 1;
#endif
};

struct ActiveRowFactor
{
    double log_volume = std::numeric_limits<double>::lowest();
};

ActiveRowFactor factorize_active_rows(
    const Eigen::Ref<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>> &C)
{
    ActiveRowFactor out;
    const int rows = static_cast<int>(C.rows());
    const int cols = static_cast<int>(C.cols());
    if (rows == 0 || cols == 0 || rows > cols || !C.allFinite())
        return out;

    const Eigen::MatrixXd dense = C;
    if (rows == cols) {
        Eigen::PartialPivLU<Eigen::MatrixXd> lu(dense);
        const double rcond = lu.rcond();
        const double lu_tol = std::numeric_limits<double>::epsilon()
                            * static_cast<double>(rows);
        if (std::isfinite(rcond) && rcond > lu_tol) {
            out.log_volume = 0.0;
            for (int i = 0; i < rows; ++i)
                out.log_volume += std::log(std::abs(lu.matrixLU()(i, i)));
            return out;
        }
    }

    Eigen::MatrixXd gram = dense * dense.transpose();
    const double scale = gram.diagonal().cwiseAbs().maxCoeff();
    const double tol = std::numeric_limits<double>::epsilon()
                     * static_cast<double>(std::max(rows, cols))
                     * std::max(1.0, scale);
    Eigen::LDLT<Eigen::MatrixXd> ldlt(gram);
    bool exact = ldlt.info() == Eigen::Success && ldlt.vectorD().allFinite() &&
                 ldlt.vectorD().minCoeff() > tol;
    if (!exact) {
        gram.diagonal().array() += std::max(tol,
                                            1e-12 * std::max(1.0, scale));
        ldlt.compute(gram);
    }
    if (ldlt.info() != Eigen::Success || !ldlt.vectorD().allFinite() ||
        ldlt.vectorD().minCoeff() <= 0.0)
        return out;
    out.log_volume = 0.0;
    for (int i = 0; i < rows; ++i)
        out.log_volume += 0.5 * std::log(std::max(
            ldlt.vectorD()[i], std::numeric_limits<double>::min()));
    return out;
}

} // namespace

class StreamRowStore
{
public:
    struct Segment {
        int rows = 0;
        int cols = 0;
        std::vector<double> memory;
        double *mapped = nullptr;
        std::size_t mapped_bytes = 0;
        std::vector<BRowRef> refs;

        Segment() = default;
        Segment(const Segment &) = delete;
        Segment &operator=(const Segment &) = delete;

        Segment(Segment &&other) noexcept
            : rows(other.rows), cols(other.cols),
              memory(std::move(other.memory)), mapped(other.mapped),
              mapped_bytes(other.mapped_bytes), refs(std::move(other.refs))
        {
            other.mapped = nullptr;
            other.mapped_bytes = 0;
        }

        Segment &operator=(Segment &&other) noexcept
        {
            if (this == &other) return *this;
            release();
            rows = other.rows;
            cols = other.cols;
            memory = std::move(other.memory);
            mapped = other.mapped;
            mapped_bytes = other.mapped_bytes;
            refs = std::move(other.refs);
            other.mapped = nullptr;
            other.mapped_bytes = 0;
            return *this;
        }

        ~Segment() { release(); }

        double *data() { return mapped ? mapped : memory.data(); }
        const double *data() const { return mapped ? mapped : memory.data(); }
        bool is_mapped() const { return mapped != nullptr; }

        Eigen::Map<OnlineRowMatrix> matrix()
        {
            return Eigen::Map<OnlineRowMatrix>(data(), rows, cols);
        }

        Eigen::Map<const OnlineRowMatrix> matrix() const
        {
            return Eigen::Map<const OnlineRowMatrix>(data(), rows, cols);
        }

    private:
        void release()
        {
            if (mapped) {
                ::munmap(mapped, mapped_bytes);
                mapped = nullptr;
                mapped_bytes = 0;
            }
        }
    };

    StreamRowStore(std::string mode, std::size_t memory_limit,
                   std::string cache_dir, int cols)
        : mode_(std::move(mode)), memory_limit_(memory_limit),
          cache_dir_(std::move(cache_dir)), cols_(cols)
    {
        if (mode_ != "auto" && mode_ != "memory" && mode_ != "mmap")
            throw std::invalid_argument("invalid stream cache mode: " + mode_);
        if (cache_dir_.empty()) {
            if (const char *tmp = std::getenv("TMPDIR"); tmp && *tmp)
                cache_dir_ = tmp;
            else
                cache_dir_ = "/tmp";
        }
    }

    Segment &add_segment(int rows)
    {
        Segment segment;
        segment.rows = rows;
        segment.cols = cols_;
        segment.refs.resize(static_cast<std::size_t>(rows));
        const std::size_t bytes =
            static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols_)
            * sizeof(double);
        const bool map_segment =
            mode_ == "mmap" ||
            (mode_ == "auto" && resident_bytes_ + bytes > memory_limit_);

        if (mode_ == "memory" && resident_bytes_ + bytes > memory_limit_)
            throw std::runtime_error("stream memory cache limit exceeded");

        if (map_segment && bytes > 0) {
            std::filesystem::create_directories(cache_dir_);
            std::string pattern =
                (std::filesystem::path(cache_dir_) / "imrmlp-maxvol-XXXXXX").string();
            std::vector<char> name(pattern.begin(), pattern.end());
            name.push_back('\0');
            const int fd = ::mkstemp(name.data());
            if (fd < 0)
                throw std::runtime_error("failed to create stream mmap cache");
            const int alloc_status = ::posix_fallocate(fd, 0, bytes);
            if (alloc_status != 0) {
                ::close(fd);
                ::unlink(name.data());
                throw std::runtime_error("failed to allocate stream mmap cache");
            }
            void *mapping = ::mmap(
                nullptr, bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            ::unlink(name.data());
            ::close(fd);
            if (mapping == MAP_FAILED)
                throw std::runtime_error("failed to map stream row cache");
            segment.mapped = static_cast<double *>(mapping);
            segment.mapped_bytes = bytes;
            mapped_bytes_ += bytes;
        } else {
            segment.memory.resize(
                static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols_));
            resident_bytes_ += bytes;
        }

        segments_.push_back(std::move(segment));
        return segments_.back();
    }

    std::vector<Segment> &segments() { return segments_; }
    const std::vector<Segment> &segments() const { return segments_; }
    std::size_t resident_bytes() const { return resident_bytes_; }
    std::size_t mapped_bytes() const { return mapped_bytes_; }

    bool find_row(int global_row, Eigen::RowVectorXd &row, BRowRef *ref = nullptr) const
    {
        for (const auto &segment : segments_) {
            const auto it = std::lower_bound(
                segment.refs.begin(), segment.refs.end(), global_row,
                [](const BRowRef &lhs, int rhs) {
                    return lhs.global_row < rhs;
                });
            if (it == segment.refs.end() || it->global_row != global_row)
                continue;
            const int local = static_cast<int>(it - segment.refs.begin());
            row = segment.matrix().row(local);
            if (ref) *ref = *it;
            return true;
        }
        return false;
    }

private:
    std::string mode_;
    std::size_t memory_limit_ = 0;
    std::string cache_dir_;
    int cols_ = 0;
    std::size_t resident_bytes_ = 0;
    std::size_t mapped_bytes_ = 0;
    std::vector<Segment> segments_;
};

ActiveSelector::ActiveSelector()
{
    Connect(threshold, "threshold", "MaxVol selection threshold");
    Connect(weight, "weight", "B-matrix row weight");
    Connect(pow_scale, "pow-scale", "natoms scaling power");
    Connect(init_mode, "init-mode", "active initialization: auto | mlip | data");
    Connect(init_value, "init-value", "MLIP virtual identity scale");
    Connect(max_swaps, "max-swaps", "swap limit (0=unlimited)");
    Connect(verbose, "verbose", "verbose output");
    Connect(select_mode, "select-mode",
            "selection backend: stream | full | online (order-dependent approximate)");
    Connect(stream_mode, "stream-mode",
            "stream backend: exact (global rescans) | fast");
    Connect(stream_chunk_size, "stream-chunk-size", "streaming cfg chunk size");
    Connect(stream_fast_passes, "stream-fast-passes", "max passes in stream fast mode");
    Connect(stream_cache_mode, "stream-cache-mode",
            "stream cache: auto | memory | mmap | off");
    Connect(stream_memory_limit_mb, "stream-memory-limit-mb",
            "per-process resident stream cache limit in MiB");
    Connect(stream_cache_dir, "stream-cache-dir",
            "directory for temporary stream mmap files");
    Connect(stream_fast_shortlist, "stream-fast-shortlist",
            "fast-mode global shortlist size (0=auto)");
}

void ActiveSelector::RegisterConsole(CMDOption::CommandLineOptions &cmdopts)
{
    cmdopts.addop(threshold, "threshold", "", "MaxVol selection threshold")
           .addop(weight, "weight", "", "B-matrix row weight")
           .addop(pow_scale, "pow-scale", "", "natoms scaling power")
           .addop(init_mode, "init-mode", "",
                  "active initialization: auto | mlip | data")
           .addop(init_value, "init-value", "", "MLIP virtual identity scale")
           .addop(max_swaps, "max-swaps", "", "swap limit (0=unlimited)")
           .addop(select_mode, "select-mode", "",
                  "selection backend: stream | full | online (order-dependent approximate)")
           .addop(stream_mode, "stream-mode", "",
                  "stream backend: exact (global rescans) | fast")
           .addop(stream_chunk_size, "stream-chunk-size", "", "streaming cfg chunk size")
           .addop(stream_fast_passes, "stream-fast-passes", "", "max passes in stream fast mode")
           .addop(stream_cache_mode, "stream-cache-mode", "",
                  "stream cache: auto | memory | mmap | off")
           .addop(stream_memory_limit_mb, "stream-memory-limit-mb", "",
                  "per-process resident stream cache limit in MiB")
           .addop(stream_cache_dir, "stream-cache-dir", "",
                  "directory for temporary stream mmap files")
           .addop(stream_fast_shortlist, "stream-fast-shortlist", "",
                  "fast-mode global shortlist size (0=auto)")
           .addop(verbose, "verbose", "v", "verbose output", false, 0);
}

ActiveSelector::ActiveSelector(BMatrixBuilder *builder)
    : ActiveSelector()
{
    builder_ = builder;
    own_builder_ = true;
    n_coeff_ = builder_->coeff_count();
    mv_.verbose = verbose;
}

ActiveSelector::~ActiveSelector()
{
    if (own_builder_)
        delete builder_;
}

// ═══════════════════════════════════════════════════════════════════════════
// Candidate management
// ═══════════════════════════════════════════════════════════════════════════

int ActiveSelector::add_candidate(Atoms &cfg)
{
    if(select_mode=="full" && (!has_full_B_ || gpm_rebuild_full_) && !cfgs_.empty()) {rebuild_B_from_configs();gpm_rebuild_full_=false;}
    const bool compact=using_stream_state_;
    const int previous_rows=cfg_row_start_.back();
    invalidate_grading();
    cfg_row_start_.push_back(cfg_row_start_.back());
    cfgs_.push_back(cfg);
    is_new_.push_back(false);

    int nrows = builder_->build_rows(cfgs_.back(), B_, weight, pow_scale);
    cfg_row_start_.back() = previous_rows+nrows;
    has_full_B_ = !compact;
    using_stream_state_ = compact;
    return nrows;
}

int ActiveSelector::add_candidates(std::vector<Atoms> &cfgs)
{
    int total = 0;
    for (auto &c : cfgs) total += add_candidate(c);
    return total;
}

int ActiveSelector::add_candidates_parallel(std::vector<Atoms> &cfgs)
{
    if(using_stream_state_||gpm_rebuild_full_)return add_candidates(cfgs);
    invalidate_grading();
    if (cfgs.empty()) return 0;
    if (!builder_) return 0;

    auto trial = builder_->clone();
    if (!trial)
        return add_candidates(cfgs);
    trial.reset();

    struct RowBlock {
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
        int nrows = 0;
    };

    const size_t base = cfgs_.size();
    cfgs_.insert(cfgs_.end(), cfgs.begin(), cfgs.end());
    is_new_.insert(is_new_.end(), cfgs.size(), false);

    std::vector<RowBlock> blocks(cfgs.size());

    #pragma omp parallel
    {
        auto local_builder = std::unique_ptr<BMatrixBuilder>(builder_->clone());

        #pragma omp for schedule(dynamic, 16)
        for (size_t i = 0; i < cfgs.size(); ++i) {
            if (!local_builder) continue;
            blocks[i].nrows = local_builder->build_rows(
                cfgs_[base + i], blocks[i].B, weight, pow_scale);
        }
    }

    int total_rows = 0;
    for (const auto &b : blocks)
        total_rows += b.nrows;

    const int old_rows = static_cast<int>(B_.rows());
    B_.conservativeResize(old_rows + total_rows, n_coeff_);

    int row = old_rows;
    for (const auto &b : blocks) {
        if (b.nrows > 0) {
            B_.middleRows(row, b.nrows) = b.B.topRows(b.nrows);
            row += b.nrows;
        }
        cfg_row_start_.push_back(row);
    }

    has_full_B_ = true;
    using_stream_state_ = false;
    return total_rows;
}

int ActiveSelector::add_candidates_unbuilt(std::vector<Atoms> &cfgs)
{
    invalidate_grading();
    if (cfgs.empty()) return 0;
    cfgs_.insert(cfgs_.end(), cfgs.begin(), cfgs.end());
    is_new_.insert(is_new_.end(), cfgs.size(), false);
    has_full_B_ = false;
    return static_cast<int>(cfgs.size());
}

int ActiveSelector::add_candidates_mpi(std::vector<Atoms> &cfgs,
                                       MPI_Comm comm,
                                       int root)
{
    invalidate_grading();
    if (cfgs.empty()) return 0;
    if (!builder_) return 0;

    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size <= 1)
        return add_candidates_parallel(cfgs);

    const int ncfg = static_cast<int>(cfgs.size());
    const int ncols = n_coeff_;

    struct RowBlock {
        int cfg_index = -1;
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
        int nrows = 0;
    };

    std::vector<RowBlock> local_blocks;
    local_blocks.reserve((ncfg + size - 1) / size);

    for (int i = 0; i < ncfg; ++i) {
        if (i % size != rank) continue;
        RowBlock block;
        block.cfg_index = i;
        block.nrows = builder_->build_rows(cfgs[i], block.B, weight, pow_scale);
        local_blocks.push_back(std::move(block));
    }

    std::vector<int> local_row_counts(ncfg, 0);
    int local_rows = 0;
    for (const auto &block : local_blocks) {
        local_row_counts[block.cfg_index] = block.nrows;
        local_rows += block.nrows;
    }

    std::vector<double> local_data;
    local_data.reserve(static_cast<size_t>(local_rows) * ncols);
    for (const auto &block : local_blocks) {
        if (block.nrows <= 0) continue;
        const int nd = block.nrows * ncols;
        const double *ptr = block.B.data();
        local_data.insert(local_data.end(), ptr, ptr + nd);
    }

    std::vector<int> root_row_counts(rank == root ? ncfg : 0, 0);
    MPI_Reduce(local_row_counts.data(),
               rank == root ? root_row_counts.data() : nullptr,
               ncfg, MPI_INT, MPI_SUM, root, comm);

    int local_double_count = static_cast<int>(local_data.size());
    std::vector<int> recv_counts(rank == root ? size : 0, 0);
    MPI_Gather(&local_double_count, 1, MPI_INT,
               rank == root ? recv_counts.data() : nullptr, 1, MPI_INT,
               root, comm);

    std::vector<int> displs;
    std::vector<double> gathered_data;
    if (rank == root) {
        displs.resize(size, 0);
        int total_doubles = 0;
        for (int r = 0; r < size; ++r) {
            displs[r] = total_doubles;
            total_doubles += recv_counts[r];
        }
        gathered_data.resize(total_doubles);
    }

    MPI_Gatherv(local_data.empty() ? nullptr : local_data.data(),
                local_double_count, MPI_DOUBLE,
                rank == root ? gathered_data.data() : nullptr,
                rank == root ? recv_counts.data() : nullptr,
                rank == root ? displs.data() : nullptr,
                MPI_DOUBLE, root, comm);

    if (rank != root)
        return local_rows;

    const int total_rows = std::accumulate(root_row_counts.begin(),
                                           root_row_counts.end(), 0);
    const int old_rows = static_cast<int>(B_.rows());
    B_.conservativeResize(old_rows + total_rows, ncols);

    std::vector<int> rank_offsets(size, 0);
    int row = old_rows;
    cfgs_.insert(cfgs_.end(), cfgs.begin(), cfgs.end());
    is_new_.insert(is_new_.end(), cfgs.size(), false);

    for (int i = 0; i < ncfg; ++i) {
        const int nrows = root_row_counts[i];
        if (nrows > 0) {
            const int owner = i % size;
            const int offset = displs[owner] + rank_offsets[owner];
            Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>
                block(gathered_data.data() + offset, nrows, ncols);
            B_.middleRows(row, nrows) = block;
            rank_offsets[owner] += nrows * ncols;
            row += nrows;
        }
        cfg_row_start_.push_back(row);
    }

    has_full_B_ = true;
    using_stream_state_ = false;
    return total_rows;
}

bool ActiveSelector::append_built_candidates_from(const ActiveSelector &other)
{
    invalidate_grading();
    if (other.cfgs_.empty()) return true;
    if (other.B_.cols() != n_coeff_) return false;

    const int old_rows = static_cast<int>(B_.rows());
    const int add_rows = static_cast<int>(other.B_.rows());
    B_.conservativeResize(old_rows + add_rows, n_coeff_);
    if (add_rows > 0)
        B_.middleRows(old_rows, add_rows) = other.B_;

    cfgs_.insert(cfgs_.end(), other.cfgs_.begin(), other.cfgs_.end());
    is_new_.insert(is_new_.end(), other.is_new_.begin(), other.is_new_.end());

    for (size_t i = 1; i < other.cfg_row_start_.size(); ++i)
        cfg_row_start_.push_back(old_rows + other.cfg_row_start_[i]);

    has_full_B_ = true;
    using_stream_state_ = false;
    return true;
}

void ActiveSelector::clear_candidates()
{
    invalidate_grading();
    reset_selection_status();
    B_.resize(0, n_coeff_);
    cfgs_.clear();
    is_new_.clear();
    cfg_row_start_.clear();
    cfg_row_start_.push_back(0);
    selected_cfgs_.clear();
    selected_config_indices_.clear();
    selected_row_indices_.clear();
    selected_row_refs_.clear();
    stream_swap_history_.clear();
    B_full_.resize(0, n_coeff_);
    log_det_ = 0.0;
    using_stream_state_ = false;
    has_full_B_ = false;
    online_state_.clear();
    online_processed_cfgs_ = 0;
    stream_rows_.reset();
    stream_stats_ = {};
}

void ActiveSelector::rebuild_B_from_configs()
{
    invalidate_grading();
    release_stream_rows();
    B_.resize(0, n_coeff_);
    cfg_row_start_.clear();
    cfg_row_start_.push_back(0);

    auto remap_selected_rows = [&]() {
        if (selected_row_refs_.size() != selected_row_indices_.size())
            return;
        std::vector<int> remapped;
        remapped.reserve(selected_row_refs_.size());
        std::vector<char> used(static_cast<size_t>(B_full_.rows()), 0);
        for (const auto &ref : selected_row_refs_) {
            if (ref.cfg_index < 0 ||
                ref.cfg_index + 1 >= static_cast<int>(cfg_row_start_.size()))
                return;
            const int row = cfg_row_start_[ref.cfg_index] + ref.local_row;
            if (ref.local_row < 0 || row < cfg_row_start_[ref.cfg_index] ||
                row >= cfg_row_start_[ref.cfg_index + 1] ||
                used[static_cast<size_t>(row)])
                return;
            used[static_cast<size_t>(row)] = 1;
            remapped.push_back(row);
        }
        selected_row_indices_ = std::move(remapped);
        for (size_t i = 0; i < selected_row_refs_.size(); ++i)
            selected_row_refs_[i].global_row = selected_row_indices_[i];
    };

    if (cfgs_.empty()) {
        B_full_ = B_;
        has_full_B_ = true;
        return;
    }
    if (!builder_)
        return;

    auto trial = builder_->clone();
    if (!trial) {
        for (auto &cfg : cfgs_) {
            builder_->build_rows(cfg, B_, weight, pow_scale);
            cfg_row_start_.push_back(B_.rows());
        }
        B_full_ = B_;
        has_full_B_ = true;
        remap_selected_rows();
        return;
    }
    trial.reset();

    struct RowBlock {
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B;
        int nrows = 0;
    };

    std::vector<RowBlock> blocks(cfgs_.size());

    #pragma omp parallel
    {
        auto local_builder = std::unique_ptr<BMatrixBuilder>(builder_->clone());

        #pragma omp for schedule(dynamic, 16)
        for (size_t i = 0; i < cfgs_.size(); ++i) {
            if (!local_builder) continue;
            blocks[i].nrows = local_builder->build_rows(
                cfgs_[i], blocks[i].B, weight, pow_scale);
        }
    }

    int total_rows = 0;
    for (const auto &b : blocks)
        total_rows += b.nrows;

    B_.conservativeResize(total_rows, n_coeff_);
    int row = 0;
    for (const auto &b : blocks) {
        if (b.nrows > 0) {
            B_.middleRows(row, b.nrows) = b.B.topRows(b.nrows);
            row += b.nrows;
        }
        cfg_row_start_.push_back(row);
    }
    B_full_ = B_;
    has_full_B_ = true;
    using_stream_state_ = false;
    remap_selected_rows();
}

// ═══════════════════════════════════════════════════════════════════════════
// Selection
// ═══════════════════════════════════════════════════════════════════════════

bool ActiveSelector::uses_mlip_initialization() const
{
    if (init_mode == "mlip") return true;
    if (init_mode == "data") return false;
    if (init_mode != "auto")
        throw std::invalid_argument("init-mode must be auto, mlip, or data");
    return dynamic_cast<const PotEFSBuilder *>(builder_) != nullptr;
}

void ActiveSelector::reset_selection_status()
{
    converged_ = false;
    max_remaining_grade_ = std::numeric_limits<double>::infinity();
    termination_reason_ = "not-run";
}

void ActiveSelector::release_stream_rows()
{
    stream_rows_.reset();
    stream_stats_.resident_bytes = 0;
    stream_stats_.mapped_bytes = 0;
    stream_stats_.cache_backend = "off";
}

void ActiveSelector::set_selection_status(bool is_converged,
                                          double max_grade,
                                          std::string reason)
{
    converged_ = is_converged;
    max_remaining_grade_ = max_grade;
    termination_reason_ = std::move(reason);
}

int ActiveSelector::virtual_row_count() const
{
    int count = 0;
    for (const auto &ref : selected_row_refs_)
        if (ref.global_row < 0) ++count;
    return count;
}

int ActiveSelector::select(double thr) { threshold = thr; return select(); }

int ActiveSelector::select()
{
    if(gpm_rebuild_full_) {rebuild_B_from_configs();gpm_rebuild_full_=false;}
    invalidate_grading();
    if (select_mode == "online" && builder_ && !cfgs_.empty())
        return select_online_current();
    reset_selection_status();
    if (select_mode == "stream" && builder_ && !cfgs_.empty())
        return select_streaming_current();

    release_stream_rows();
    if (uses_mlip_initialization())
        return select_mlip_full_current();

    // Full/data selection prepares mv_'s cache below. An online factorization
    // from an earlier stream run or checkpoint no longer describes this set.
    online_state_.clear();
    online_processed_cfgs_ = 0;
    int nrows = static_cast<int>(B_.rows());
    int k = std::min(nrows, n_coeff_);
    if (k <= 0) return 0;

    mv_.tolerance = threshold - 1.0;
    mv_.max_swaps = max_swaps;
    mv_.verbose = verbose;

    std::vector<int> sel_rows = mv_.select_rows(B_, k);

    B_full_ = B_;
    has_full_B_ = true;
    using_stream_state_ = false;
    log_det_ = mv_.log_determinant();
    selected_row_indices_ = sel_rows;
    selected_row_refs_.clear();

    selected_config_indices_.clear();
    selected_cfgs_.clear();
    std::set<int> used;
    std::unordered_map<int, std::vector<BRowRef>> row_metadata;

    for (int r : sel_rows) {
        int ci = -1;
        for (int i = 0; i < static_cast<int>(cfg_row_start_.size()) - 1; ++i)
            if (r >= cfg_row_start_[i] && r < cfg_row_start_[i + 1])
                { ci = i; break; }
        if (ci >= 0 && !used.count(ci)) {
            used.insert(ci);
            selected_config_indices_.push_back(ci);
            selected_cfgs_.push_back(cfgs_[ci]);
        }
        if (ci >= 0) {
            BRowRef ref;
            ref.cfg_index = ci;
            ref.local_row = r - cfg_row_start_[ci];
            ref.global_row = r;
            if (builder_) {
                auto [it, inserted] = row_metadata.try_emplace(ci);
                if (inserted) {
                    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> ignored;
                    builder_->build_rows(cfgs_[ci], ignored, weight, pow_scale,
                                         &it->second);
                }
                if (ref.local_row >= 0 &&
                    ref.local_row < static_cast<int>(it->second.size())) {
                    ref.kind = it->second[ref.local_row].kind;
                    ref.atom_index = it->second[ref.local_row].atom_index;
                    ref.component = it->second[ref.local_row].component;
                }
            }
            selected_row_refs_.push_back(ref);
        }
    }
    std::vector<double> grades;
    compute_grades(grades);
    const double max_grade = grades.empty()
        ? 0.0 : *std::max_element(grades.begin(), grades.end());
    const bool ok = max_grade <= std::max(1.0, threshold) + 1e-8;
    set_selection_status(ok, max_grade,
                         ok ? "threshold" : "postcheck-failed");
    return static_cast<int>(selected_cfgs_.size());
}

int ActiveSelector::select_mlip_full_current()
{
    if (!builder_ || B_.rows() <= 0 || n_coeff_ <= 0) return 0;
    if (!std::isfinite(init_value) || init_value <= 0.0)
        throw std::invalid_argument("init-value must be finite and positive");

    std::vector<BRowRef> refs;
    refs.reserve(static_cast<std::size_t>(B_.rows()));
    for (int ci = 0; ci + 1 < static_cast<int>(cfg_row_start_.size()); ++ci) {
        const int count = cfg_row_start_[ci + 1] - cfg_row_start_[ci];
        std::vector<BRowRef> local_refs;
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> ignored;
        builder_->build_rows(cfgs_[ci], ignored, weight, pow_scale, &local_refs);
        if (static_cast<int>(local_refs.size()) < count)
            local_refs.resize(static_cast<std::size_t>(count));
        for (int lr = 0; lr < count; ++lr) {
            BRowRef ref = local_refs[static_cast<std::size_t>(lr)];
            ref.cfg_index = ci;
            ref.local_row = lr;
            ref.global_row = cfg_row_start_[ci] + lr;
            refs.push_back(ref);
        }
    }
    if (refs.size() != static_cast<std::size_t>(B_.rows())) return 0;

    online_state_.reset_identity(n_coeff_, init_value);
    if (!online_state_.valid()) return 0;
    const int swaps = online_state_.maximize_with_candidates(
        B_, refs, threshold, max_swaps);
    stream_swap_history_.clear();
    for (const auto &step : online_state_.last_swaps())
        stream_swap_history_.emplace_back(
            step.entering.global_row, step.outgoing.global_row, step.gain);

    const Eigen::VectorXd scores = online_state_.score_rows(B_);
    const double max_grade = scores.size() == 0 ? 0.0 : scores.maxCoeff();
    const bool ok = std::isfinite(max_grade) &&
        max_grade <= std::max(1.0, threshold) + 1e-8;
    const bool capped = max_swaps > 0 && swaps >= max_swaps && !ok;
    set_selection_status(ok, max_grade,
                         ok ? "threshold" :
                         (capped ? "max-swaps" : "postcheck-failed"));

    selected_row_refs_ = online_state_.selected_refs();
    if (!build_compact_selected_state(
            online_state_.active_rows(), selected_row_refs_))
        return 0;
    using_stream_state_ = true;
    online_processed_cfgs_ = ok ? cfgs_.size() : 0;
    return static_cast<int>(selected_cfgs_.size());
}

bool ActiveSelector::build_compact_selected_state(
    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &C,
    const std::vector<BRowRef> &refs)
{
    invalidate_grading();
    const int k = static_cast<int>(C.rows());
    if (k <= 0 || C.cols() != n_coeff_ || refs.size() != static_cast<size_t>(k))
        return false;

    B_full_ = C;
    B_.resize(0, n_coeff_);
    has_full_B_ = false;
    using_stream_state_ = true;

    selected_row_indices_.resize(k);
    std::iota(selected_row_indices_.begin(), selected_row_indices_.end(), 0);
    selected_row_refs_ = refs;

    selected_config_indices_.clear();
    selected_cfgs_.clear();
    std::set<int> used;
    for (const auto &ref : selected_row_refs_) {
        if (ref.cfg_index < 0 || ref.cfg_index >= static_cast<int>(cfgs_.size()))
            continue;
        if (!used.count(ref.cfg_index)) {
            used.insert(ref.cfg_index);
            selected_config_indices_.push_back(ref.cfg_index);
            selected_cfgs_.push_back(cfgs_[ref.cfg_index]);
        }
    }

    log_det_ = online_state_.valid()
             ? online_state_.log_determinant()
             : factorize_active_rows(B_full_).log_volume;
    return !selected_cfgs_.empty();
}

int ActiveSelector::select_streaming(std::vector<Atoms> &cfgs)
{
    clear_candidates();
    cfgs_ = cfgs;
    is_new_.assign(cfgs_.size(), false);
    return select_streaming_current();
}

int ActiveSelector::select_streaming()
{
    return select_streaming_current();
}

int ActiveSelector::select_streaming_mpi(MPI_Comm comm)
{
    return select_streaming_current(comm, true);
}

int ActiveSelector::select_online()
{
    return select_online_current();
}

int ActiveSelector::select_streaming_current()
{
    return select_streaming_current(MPI_COMM_WORLD, false);
}

int ActiveSelector::select_streaming_current(MPI_Comm comm, bool distributed)
{
    if (!distributed)
        comm = MPI_COMM_SELF;
    invalidate_grading();
    if (!builder_ || cfgs_.empty()) return 0;
    n_coeff_ = builder_->coeff_count();
    if (n_coeff_ <= 0) return 0;
    if (stream_memory_limit_mb <= 0)
        throw std::invalid_argument("stream-memory-limit-mb must be positive");
    if (stream_cache_mode != "auto" && stream_cache_mode != "memory" &&
        stream_cache_mode != "mmap" && stream_cache_mode != "off")
        throw std::invalid_argument(
            "stream-cache-mode must be auto, memory, mmap, or off");

    if (stream_cache_mode == "off") {
        stream_stats_ = {};
        stream_stats_.cache_backend = "off";
        if (uses_mlip_initialization()) {
            rebuild_B_from_configs();
            return select_mlip_full_current();
        }
        return select_streaming_legacy_current();
    }

    using Clock = std::chrono::steady_clock;
    struct Candidate {
        Eigen::RowVectorXd row;
        BRowRef ref;
        double norm2 = -1.0;
    };

    int rank = 0, size = 1;
    if (distributed) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &size);
    }

    stream_stats_ = {};
    stream_swap_history_.clear();
    selected_row_refs_.clear();
    selected_row_indices_.clear();
    selected_cfgs_.clear();
    selected_config_indices_.clear();
    B_.resize(0, n_coeff_);
    B_full_.resize(0, n_coeff_);
    has_full_B_ = false;
    using_stream_state_ = true;
    online_state_.clear();
    online_processed_cfgs_ = 0;

    const auto total_started = Clock::now();
    const auto build_started = total_started;
    const int ncfg = static_cast<int>(cfgs_.size());
    int thread_count = 1;
#ifdef USE_OMP
    thread_count = std::max(1, omp_get_max_threads());
#endif
    stream_stats_.configured_threads = thread_count;

    std::vector<std::unique_ptr<BMatrixBuilder>> thread_builders;
    thread_builders.reserve(static_cast<std::size_t>(thread_count));
    bool cloneable = thread_count > 1;
    if (cloneable) {
        for (int t = 0; t < thread_count; ++t) {
            auto clone = builder_->clone();
            if (!clone) {
                cloneable = false;
                thread_builders.clear();
                break;
            }
            thread_builders.push_back(std::move(clone));
        }
    }

    std::vector<int> local_counts(static_cast<std::size_t>(ncfg), 0);
    std::vector<int> row_counts(static_cast<std::size_t>(ncfg), 0);
    std::vector<OnlineRowMatrix> prebuilt(static_cast<std::size_t>(ncfg));
    std::vector<std::vector<BRowRef>> prebuilt_refs(
        static_cast<std::size_t>(ncfg));
    std::vector<char> was_prebuilt(static_cast<std::size_t>(ncfg), 0);
    std::vector<int> unknown_count_configs;

    for (int ci = 0; ci < ncfg; ++ci) {
        if (ci % size != rank) continue;
        const int count =
            builder_->row_count(cfgs_[static_cast<std::size_t>(ci)]);
        if (count < 0)
            unknown_count_configs.push_back(ci);
        else
            local_counts[static_cast<std::size_t>(ci)] = count;
    }

    std::vector<std::exception_ptr> prebuild_errors(
        unknown_count_configs.size());
    if (cloneable && !unknown_count_configs.empty()) {
        const int prebuild_threads = std::min(
            thread_count, static_cast<int>(unknown_count_configs.size()));
#ifdef USE_OMP
        #pragma omp parallel num_threads(prebuild_threads)
#endif
        {
            MklLocalThreadGuard mkl_threads(1);
            int tid = 0;
#ifdef USE_OMP
            tid = omp_get_thread_num();
            #pragma omp single
            stream_stats_.build_threads =
                std::max(stream_stats_.build_threads, omp_get_num_threads());
            #pragma omp for schedule(dynamic, 1)
#endif
            for (std::size_t ui = 0; ui < unknown_count_configs.size(); ++ui) {
                const int ci = unknown_count_configs[ui];
                try {
                    const int count = thread_builders[
                        static_cast<std::size_t>(tid)]->build_rows(
                            cfgs_[static_cast<std::size_t>(ci)],
                            prebuilt[static_cast<std::size_t>(ci)],
                            weight, pow_scale,
                            &prebuilt_refs[static_cast<std::size_t>(ci)]);
                    local_counts[static_cast<std::size_t>(ci)] = count;
                    was_prebuilt[static_cast<std::size_t>(ci)] = 1;
                } catch (...) {
                    prebuild_errors[ui] = std::current_exception();
                }
            }
        }
    } else {
        for (std::size_t ui = 0; ui < unknown_count_configs.size(); ++ui) {
            const int ci = unknown_count_configs[ui];
            try {
                const int count = builder_->build_rows(
                    cfgs_[static_cast<std::size_t>(ci)],
                    prebuilt[static_cast<std::size_t>(ci)],
                    weight, pow_scale,
                    &prebuilt_refs[static_cast<std::size_t>(ci)]);
                local_counts[static_cast<std::size_t>(ci)] = count;
                was_prebuilt[static_cast<std::size_t>(ci)] = 1;
            } catch (...) {
                prebuild_errors[ui] = std::current_exception();
            }
        }
    }
    std::exception_ptr prebuild_error;
    for (const auto &error : prebuild_errors)
        if (error) {
            prebuild_error = error;
            break;
        }
    int prebuild_ok = prebuild_error ? 0 : 1;
    int all_prebuilds_ok = 0;
    MPI_Allreduce(&prebuild_ok, &all_prebuilds_ok, 1,
                  MPI_INT, MPI_MIN, comm);
    if (!all_prebuilds_ok) {
        if (prebuild_error)
            std::rethrow_exception(prebuild_error);
        throw std::runtime_error(
            "stream row-count fallback failed on another MPI rank");
    }
    stream_stats_.builder_calls = unknown_count_configs.size();

    MPI_Allreduce(local_counts.data(), row_counts.data(), ncfg,
                  MPI_INT, MPI_SUM, comm);

    cfg_row_start_.clear();
    cfg_row_start_.reserve(static_cast<std::size_t>(ncfg + 1));
    cfg_row_start_.push_back(0);
    for (int count : row_counts) {
        if (count < 0)
            throw std::runtime_error("negative B-row count");
        if (cfg_row_start_.back() >
            std::numeric_limits<int>::max() - count)
            throw std::overflow_error("stream B-row count exceeds int range");
        cfg_row_start_.push_back(cfg_row_start_.back() + count);
    }
    if (cfg_row_start_.back() <= 0)
        return 0;

    const std::size_t memory_limit =
        static_cast<std::size_t>(stream_memory_limit_mb) * 1024u * 1024u;
    struct SegmentInfo {
        std::size_t store_index = 0;
        std::vector<std::pair<int, int>> configs; // cfg index, local row offset
    };
    std::vector<SegmentInfo> segment_infos;
    std::exception_ptr allocation_error;
    try {
        stream_rows_ = std::make_unique<StreamRowStore>(
            stream_cache_mode, memory_limit, stream_cache_dir, n_coeff_);
        const int chunk_cfgs = std::max(1, stream_chunk_size);
        for (int beg = 0; beg < ncfg; beg += chunk_cfgs) {
            const int end = std::min(ncfg, beg + chunk_cfgs);
            SegmentInfo info;
            int local_rows = 0;
            for (int ci = beg; ci < end; ++ci) {
                if (ci % size != rank) continue;
                info.configs.emplace_back(ci, local_rows);
                local_rows += row_counts[static_cast<std::size_t>(ci)];
            }
            if (local_rows == 0) continue;
            info.store_index = stream_rows_->segments().size();
            stream_rows_->add_segment(local_rows);
            segment_infos.push_back(std::move(info));
        }
    } catch (...) {
        allocation_error = std::current_exception();
        stream_rows_.reset();
        segment_infos.clear();
    }
    int allocation_ok = allocation_error ? 0 : 1;
    int all_allocations_ok = 0;
    MPI_Allreduce(&allocation_ok, &all_allocations_ok, 1,
                  MPI_INT, MPI_MIN, comm);
    if (!all_allocations_ok) {
        if (stream_cache_mode == "auto") {
            const int selected = select_streaming_legacy_current();
            stream_stats_.cache_backend = "off(auto-fallback)";
            return selected;
        }
        if (allocation_error)
            std::rethrow_exception(allocation_error);
        throw std::runtime_error(
            "stream cache allocation failed on another MPI rank");
    }

    struct BuildTask {
        std::size_t info_index = 0;
        int cfg_index = -1;
        int offset = 0;
    };
    std::vector<BuildTask> build_tasks;
    for (std::size_t si = 0; si < segment_infos.size(); ++si)
        for (const auto &[ci, offset] : segment_infos[si].configs)
            build_tasks.push_back(BuildTask{si, ci, offset});

    auto fill_config = [&](const BuildTask &task,
                           BMatrixBuilder &local_builder) {
        const auto &info = segment_infos[task.info_index];
        auto &segment = stream_rows_->segments()[info.store_index];
        auto matrix = segment.matrix();
        const int ci = task.cfg_index;
        const int offset = task.offset;
        const int count = row_counts[static_cast<std::size_t>(ci)];
        if (count <= 0) return;
        std::vector<BRowRef> local_refs;
        int built = 0;
        if (was_prebuilt[static_cast<std::size_t>(ci)]) {
            matrix.middleRows(offset, count) =
                prebuilt[static_cast<std::size_t>(ci)].topRows(count);
            local_refs = prebuilt_refs[static_cast<std::size_t>(ci)];
            built = count;
        } else {
            auto block = matrix.middleRows(offset, count);
            built = local_builder.build_rows_into(
                cfgs_[static_cast<std::size_t>(ci)], block,
                weight, pow_scale, &local_refs);
        }
        if (built != count)
            throw std::runtime_error(
                "BMatrixBuilder row_count/build_rows_into mismatch");
        if (static_cast<int>(local_refs.size()) < count)
            local_refs.resize(static_cast<std::size_t>(count));
        for (int lr = 0; lr < count; ++lr) {
            BRowRef ref = local_refs[static_cast<std::size_t>(lr)];
            ref.cfg_index = ci;
            ref.local_row = lr;
            ref.global_row =
                cfg_row_start_[static_cast<std::size_t>(ci)] + lr;
            segment.refs[static_cast<std::size_t>(offset + lr)] = ref;
        }
    };

    std::exception_ptr fill_error;
    if (cloneable && !build_tasks.empty()) {
        std::vector<std::exception_ptr> fill_errors(build_tasks.size());
        const int fill_threads =
            std::min(thread_count, static_cast<int>(build_tasks.size()));
#ifdef USE_OMP
        #pragma omp parallel num_threads(fill_threads)
#endif
        {
            MklLocalThreadGuard mkl_threads(1);
            int tid = 0;
#ifdef USE_OMP
            tid = omp_get_thread_num();
            #pragma omp single
            stream_stats_.build_threads =
                std::max(stream_stats_.build_threads, omp_get_num_threads());
            #pragma omp for schedule(dynamic, 1)
#endif
            for (std::size_t ti = 0; ti < build_tasks.size(); ++ti) {
                try {
                    fill_config(
                        build_tasks[ti],
                        *thread_builders[static_cast<std::size_t>(tid)]);
                } catch (...) {
                    fill_errors[ti] = std::current_exception();
                }
            }
        }
        for (const auto &error : fill_errors)
            if (error) {
                fill_error = error;
                break;
            }
    } else {
        try {
            for (const auto &task : build_tasks)
                fill_config(task, *builder_);
        } catch (...) {
            fill_error = std::current_exception();
        }
    }
    int fill_ok = fill_error ? 0 : 1;
    int all_fills_ok = 0;
    MPI_Allreduce(&fill_ok, &all_fills_ok, 1, MPI_INT, MPI_MIN, comm);
    if (!all_fills_ok) {
        if (fill_error)
            std::rethrow_exception(fill_error);
        throw std::runtime_error(
            "stream row-cache build failed on another MPI rank");
    }

    std::size_t built_calls = stream_stats_.builder_calls;
    std::size_t local_rows_built = 0;
    for (int ci = 0; ci < ncfg; ++ci) {
        if (ci % size != rank) continue;
        local_rows_built += static_cast<std::size_t>(
            row_counts[static_cast<std::size_t>(ci)]);
        if (!was_prebuilt[static_cast<std::size_t>(ci)])
            ++built_calls;
    }
    stream_stats_.builder_calls = built_calls;
    stream_stats_.rows_built = local_rows_built;
    stream_stats_.descriptor_passes = 1;
    stream_stats_.resident_bytes = stream_rows_->resident_bytes();
    stream_stats_.mapped_bytes = stream_rows_->mapped_bytes();
    stream_stats_.cache_backend =
        stream_stats_.mapped_bytes == 0 ? "memory" :
        (stream_stats_.resident_bytes == 0 ? "mmap" : "memory+mmap");
    stream_stats_.build_seconds =
        std::chrono::duration<double>(Clock::now() - build_started).count();

    const auto initialization_started = Clock::now();
    int local_usable = 0;
    const int segment_count =
        static_cast<int>(stream_rows_->segments().size());
    const int scan_threads = std::max(
        1, std::min(thread_count, std::max(1, segment_count)));
#ifdef USE_OMP
    #pragma omp parallel num_threads(scan_threads) reduction(+:local_usable)
#endif
    {
        MklLocalThreadGuard mkl_threads(1);
#ifdef USE_OMP
        #pragma omp for schedule(dynamic, 1)
#endif
        for (int si = 0; si < segment_count; ++si) {
            const auto &segment =
                stream_rows_->segments()[static_cast<std::size_t>(si)];
            const auto matrix = segment.matrix();
            for (int row = 0; row < matrix.rows(); ++row)
                if (matrix.row(row).allFinite())
                    ++local_usable;
        }
    }
    int usable_rows = 0;
    MPI_Allreduce(&local_usable, &usable_rows, 1, MPI_INT, MPI_SUM, comm);
    const int k = std::min(usable_rows, n_coeff_);
    if (k <= 0) return 0;

    auto residual_against = [](const auto &row,
                               const std::vector<Eigen::RowVectorXd> &qrows,
                               Eigen::RowVectorXd &residual) {
        residual = row;
        for (int pass = 0; pass < 2; ++pass)
            for (const auto &q : qrows)
                residual.noalias() -= residual.dot(q) * q;
    };
    auto is_independent = [&](const auto &row,
                              const std::vector<Eigen::RowVectorXd> &qrows,
                              Eigen::RowVectorXd &residual) {
        if (!row.allFinite()) return false;
        residual_against(row, qrows, residual);
        const double norm = residual.norm();
        const double tolerance =
            std::numeric_limits<double>::epsilon()
            * static_cast<double>(std::max(1, n_coeff_))
            * std::max(1.0, row.norm());
        return std::isfinite(norm) && norm > tolerance;
    };

    auto gather_best = [&](double local_gain, int local_global) {
        std::vector<double> gains(static_cast<std::size_t>(size));
        std::vector<int> globals(static_cast<std::size_t>(size));
        MPI_Allgather(&local_gain, 1, MPI_DOUBLE,
                      gains.data(), 1, MPI_DOUBLE, comm);
        MPI_Allgather(&local_global, 1, MPI_INT,
                      globals.data(), 1, MPI_INT, comm);
        double best_gain = -std::numeric_limits<double>::infinity();
        for (double gain : gains)
            if (std::isfinite(gain))
                best_gain = std::max(best_gain, gain);
        const double tie = 64.0 * std::numeric_limits<double>::epsilon()
                         * std::max(1.0, std::abs(best_gain));
        int best_global = -1;
        for (int r = 0; r < size; ++r) {
            if (globals[static_cast<std::size_t>(r)] < 0 ||
                gains[static_cast<std::size_t>(r)] < best_gain - tie)
                continue;
            if (best_global < 0 ||
                globals[static_cast<std::size_t>(r)] < best_global)
                best_global = globals[static_cast<std::size_t>(r)];
        }
        return std::pair<double, int>{best_gain, best_global};
    };

    auto broadcast_candidate = [&](int global_row) {
        Candidate candidate;
        candidate.row.resize(n_coeff_);
        const auto upper = std::upper_bound(
            cfg_row_start_.begin(), cfg_row_start_.end(), global_row);
        const int cfg_index =
            std::max(0, static_cast<int>(upper - cfg_row_start_.begin()) - 1);
        const int owner = distributed ? cfg_index % size : 0;
        if (rank == owner &&
            !stream_rows_->find_row(
                global_row, candidate.row, &candidate.ref))
            throw std::runtime_error("winning stream row not found on owner");
        MPI_Bcast(candidate.row.data(), n_coeff_, MPI_DOUBLE, owner, comm);
        int metadata[6] = {-1, -1, -1,
                           static_cast<int>(BRowKind::Unknown), -1, -1};
        if (rank == owner) {
            metadata[0] = candidate.ref.cfg_index;
            metadata[1] = candidate.ref.local_row;
            metadata[2] = candidate.ref.global_row;
            metadata[3] = static_cast<int>(candidate.ref.kind);
            metadata[4] = candidate.ref.atom_index;
            metadata[5] = candidate.ref.component;
        }
        MPI_Bcast(metadata, 6, MPI_INT, owner, comm);
        candidate.ref.cfg_index = metadata[0];
        candidate.ref.local_row = metadata[1];
        candidate.ref.global_row = metadata[2];
        candidate.ref.kind = static_cast<BRowKind>(metadata[3]);
        candidate.ref.atom_index = metadata[4];
        candidate.ref.component = metadata[5];
        candidate.norm2 = candidate.row.squaredNorm();
        return candidate;
    };

    std::vector<unsigned char> selected_mask(
        static_cast<std::size_t>(cfg_row_start_.back()), 0);
    OnlineRowMatrix C;
    std::vector<BRowRef> refs;
    if (uses_mlip_initialization()) {
        if (!std::isfinite(init_value) || init_value <= 0.0)
            throw std::invalid_argument("init-value must be finite and positive");
        C = init_value * OnlineRowMatrix::Identity(n_coeff_, n_coeff_);
        refs.assign(static_cast<std::size_t>(n_coeff_), BRowRef{});
    } else {
        std::vector<Candidate> initial;
        initial.reserve(static_cast<std::size_t>(k));
        std::vector<Eigen::RowVectorXd> qrows;
        qrows.reserve(static_cast<std::size_t>(k));
        std::vector<std::vector<char>> rejected;
        rejected.reserve(stream_rows_->segments().size());
        for (const auto &segment : stream_rows_->segments())
            rejected.emplace_back(static_cast<std::size_t>(segment.rows), 0);

        while (static_cast<int>(initial.size()) < k) {
            int local_global = -1;
            std::vector<int> thread_globals(
                static_cast<std::size_t>(scan_threads), -1);
            std::vector<std::exception_ptr> scan_errors(
                static_cast<std::size_t>(segment_count));
#ifdef USE_OMP
            #pragma omp parallel num_threads(scan_threads)
#endif
            {
                MklLocalThreadGuard mkl_threads(1);
                int tid = 0;
                Eigen::RowVectorXd residual(n_coeff_);
#ifdef USE_OMP
                tid = omp_get_thread_num();
                #pragma omp for schedule(dynamic, 1)
#endif
                for (int si = 0; si < segment_count; ++si) {
                    try {
                        int thread_global =
                            thread_globals[static_cast<std::size_t>(tid)];
                        const auto &segment =
                            stream_rows_->segments()[static_cast<std::size_t>(si)];
                        const auto matrix = segment.matrix();
                        for (int row = 0; row < segment.rows; ++row) {
                            if (rejected[static_cast<std::size_t>(si)]
                                        [static_cast<std::size_t>(row)])
                                continue;
                            const int global =
                                segment.refs[static_cast<std::size_t>(row)].global_row;
                            if (selected_mask[static_cast<std::size_t>(global)])
                                continue;
                            if (thread_global >= 0 && global >= thread_global)
                                continue;
                            if (is_independent(matrix.row(row), qrows, residual)) {
                                thread_global = global;
                            } else {
                                rejected[static_cast<std::size_t>(si)]
                                        [static_cast<std::size_t>(row)] = 1;
                            }
                        }
                        thread_globals[static_cast<std::size_t>(tid)] =
                            thread_global;
                    } catch (...) {
                        scan_errors[static_cast<std::size_t>(si)] =
                            std::current_exception();
                    }
                }
            }
            for (const auto &error : scan_errors)
                if (error) std::rethrow_exception(error);
            for (int global : thread_globals)
                if (global >= 0 &&
                    (local_global < 0 || global < local_global))
                    local_global = global;

            const auto [ignored_gain, global] = gather_best(
                local_global >= 0 ? 1.0 : -std::numeric_limits<double>::infinity(),
                local_global);
            (void)ignored_gain;
            if (global < 0) break;
            Candidate candidate = broadcast_candidate(global);
            Eigen::RowVectorXd residual(n_coeff_);
            if (!is_independent(candidate.row, qrows, residual))
                break;
            const double norm = residual.norm();
            qrows.push_back(residual / norm);
            selected_mask[static_cast<std::size_t>(global)] = 1;
            initial.push_back(std::move(candidate));
        }

        while (static_cast<int>(initial.size()) < k) {
            double local_norm2 = -std::numeric_limits<double>::infinity();
            int local_global = -1;
            std::vector<std::pair<double, int>> thread_best(
                static_cast<std::size_t>(scan_threads),
                {-std::numeric_limits<double>::infinity(), -1});
            std::vector<std::exception_ptr> scan_errors(
                static_cast<std::size_t>(segment_count));
#ifdef USE_OMP
            #pragma omp parallel num_threads(scan_threads)
#endif
            {
                MklLocalThreadGuard mkl_threads(1);
                int tid = 0;
#ifdef USE_OMP
                tid = omp_get_thread_num();
                #pragma omp for schedule(dynamic, 1)
#endif
                for (int si = 0; si < segment_count; ++si) {
                    try {
                        auto best = thread_best[static_cast<std::size_t>(tid)];
                        const auto &segment =
                            stream_rows_->segments()[static_cast<std::size_t>(si)];
                        const auto matrix = segment.matrix();
                        for (int row = 0; row < segment.rows; ++row) {
                            const int global =
                                segment.refs[static_cast<std::size_t>(row)].global_row;
                            if (selected_mask[static_cast<std::size_t>(global)] ||
                                !matrix.row(row).allFinite())
                                continue;
                            const double norm2 = matrix.row(row).squaredNorm();
                            if (norm2 > best.first ||
                                (norm2 == best.first &&
                                 (best.second < 0 || global < best.second)))
                                best = {norm2, global};
                        }
                        thread_best[static_cast<std::size_t>(tid)] = best;
                    } catch (...) {
                        scan_errors[static_cast<std::size_t>(si)] =
                            std::current_exception();
                    }
                }
            }
            for (const auto &error : scan_errors)
                if (error) std::rethrow_exception(error);
            for (const auto &[norm2, global] : thread_best)
                if (norm2 > local_norm2 ||
                    (norm2 == local_norm2 &&
                     global >= 0 &&
                     (local_global < 0 || global < local_global))) {
                    local_norm2 = norm2;
                    local_global = global;
                }

            const auto [best_norm2, global] =
                gather_best(local_norm2, local_global);
            if (global < 0 || !std::isfinite(best_norm2)) break;
            Candidate candidate = broadcast_candidate(global);
            selected_mask[static_cast<std::size_t>(global)] = 1;
            initial.push_back(std::move(candidate));
        }
        if (static_cast<int>(initial.size()) != k)
            return 0;

        std::sort(initial.begin(), initial.end(),
                  [](const Candidate &lhs, const Candidate &rhs) {
                      return lhs.ref.global_row < rhs.ref.global_row;
                  });
        C.resize(k, n_coeff_);
        refs.reserve(static_cast<std::size_t>(k));
        for (int i = 0; i < k; ++i) {
            C.row(i) = initial[static_cast<std::size_t>(i)].row;
            refs.push_back(initial[static_cast<std::size_t>(i)].ref);
        }
    }

    OnlineMaxVolState stream_state;
    if (!stream_state.initialize_from_rows(C, refs))
        return 0;
    stream_stats_.initialization_seconds =
        std::chrono::duration<double>(
            Clock::now() - initialization_started).count();
    log_det_ = stream_state.log_determinant();
    int swap_count = 0;
    const int pass_limit = stream_mode == "fast"
        ? std::max(1, stream_fast_passes)
        : std::max(1, mv_.max_iterations);
    const bool use_shortlist = stream_mode == "fast" ||
                               uses_mlip_initialization();
    const auto score_started = Clock::now();
    // Score by row blocks, independently of configuration/cache segments.
    // A single large force configuration can then use all workers, without
    // allocating one full segment-sized coefficient matrix per worker.
    struct ScoreTask {
        std::size_t segment;
        int first;
        int rows;
    };
    std::vector<ScoreTask> score_tasks;
    const int score_block_rows = stream_state.score_block_rows();
    for (std::size_t si = 0; si < stream_rows_->segments().size(); ++si) {
        const int rows = stream_rows_->segments()[si].rows;
        for (int first = 0; first < rows; first += score_block_rows)
            score_tasks.push_back({si, first, std::min(score_block_rows, rows - first)});
    }
    const int score_thread_count = std::max(
        1, std::min(thread_count, static_cast<int>(score_tasks.size())));
    std::vector<OnlineMaxVolState::ScoreWorkspace> score_workspaces(
        static_cast<std::size_t>(score_thread_count));

    auto better_candidate = [](double gain, int global,
                               double best_gain, int best_global) {
        if (global < 0 || !std::isfinite(gain))
            return false;
        if (best_global < 0 || !std::isfinite(best_gain))
            return true;
        const double tie =
            64.0 * std::numeric_limits<double>::epsilon()
            * std::max({1.0, std::abs(gain), std::abs(best_gain)});
        return gain > best_gain + tie ||
               (gain >= best_gain - tie && global < best_global);
    };

    auto record_swaps = [&](const std::vector<OnlineSwapStep> &steps) {
        for (const auto &step : steps) {
            stream_swap_history_.emplace_back(
                step.entering.global_row, step.outgoing.global_row, step.gain);
            if (verbose && rank == 0) {
                std::cout << "Stream MaxVol swap: row "
                          << step.outgoing.global_row << " out, row "
                          << step.entering.global_row << " in (gain "
                          << step.gain << ")\n";
            }
        }
    };

    for (int pass = 0; pass < pass_limit; ++pass) {
        std::fill(selected_mask.begin(), selected_mask.end(), 0);
        for (const auto &ref : stream_state.selected_refs())
            if (ref.global_row >= 0 &&
                ref.global_row < static_cast<int>(selected_mask.size()))
                selected_mask[static_cast<std::size_t>(ref.global_row)] = 1;

        if (!use_shortlist) {
            double local_gain = -std::numeric_limits<double>::infinity();
            int local_global = -1;
            std::vector<std::pair<double, int>> thread_best(
                static_cast<std::size_t>(score_thread_count),
                {-std::numeric_limits<double>::infinity(), -1});
            std::vector<std::exception_ptr> score_errors(
                score_tasks.size());
#ifdef USE_OMP
            #pragma omp parallel num_threads(score_thread_count)
#endif
            {
                MklLocalThreadGuard mkl_threads(1);
                int tid = 0;
#ifdef USE_OMP
                tid = omp_get_thread_num();
                #pragma omp single
                stream_stats_.score_threads =
                    std::max(stream_stats_.score_threads,
                             omp_get_num_threads());
                #pragma omp for schedule(dynamic, 1)
#endif
                for (std::size_t ti = 0; ti < score_tasks.size(); ++ti) {
                    try {
                        auto best =
                            thread_best[static_cast<std::size_t>(tid)];
                        const auto &task = score_tasks[ti];
                        const auto &segment =
                            stream_rows_->segments()[task.segment];
                        const auto cached = segment.matrix();
                        const auto matrix = cached.middleRows(task.first, task.rows);
                        const auto scores = stream_state.score_rows(
                            matrix,
                            score_workspaces[static_cast<std::size_t>(tid)]);
                        for (int row = 0; row < task.rows; ++row) {
                            const int global =
                                segment.refs[static_cast<std::size_t>(task.first + row)]
                                    .global_row;
                            if (selected_mask[static_cast<std::size_t>(global)])
                                continue;
                            const double gain = scores[row];
                            if (better_candidate(
                                    gain, global, best.first, best.second))
                                best = {gain, global};
                        }
                        thread_best[static_cast<std::size_t>(tid)] = best;
                    } catch (...) {
                        score_errors[ti] =
                            std::current_exception();
                    }
                }
            }
            for (const auto &error : score_errors)
                if (error) std::rethrow_exception(error);
            for (const auto &[gain, global] : thread_best)
                if (better_candidate(
                        gain, global, local_gain, local_global)) {
                    local_gain = gain;
                    local_global = global;
                }

            ++stream_stats_.score_passes;
            const auto [best_gain, best_global] =
                gather_best(local_gain, local_global);
            if (best_global < 0 || !std::isfinite(best_gain) ||
                best_gain <= std::max(1.0, threshold))
                break;

            Candidate best = broadcast_candidate(best_global);
            OnlineRowMatrix one(1, n_coeff_);
            one.row(0) = best.row;
            const int changed = stream_state.maximize_with_candidates(
                one, std::vector<BRowRef>{best.ref}, threshold, 1);
            if (changed != 1) break;
            record_swaps(stream_state.last_swaps());
            ++swap_count;
        } else {
            const int shortlist_limit = stream_fast_shortlist > 0
                ? stream_fast_shortlist
                : std::min(std::max(4 * k, 256), 4096);
            using HeapItem = std::pair<double, int>; // gain, -global row
            using MinHeap = std::priority_queue<
                HeapItem, std::vector<HeapItem>, std::greater<HeapItem>>;
            std::vector<MinHeap> thread_heaps(
                static_cast<std::size_t>(score_thread_count));
            std::vector<std::exception_ptr> score_errors(
                score_tasks.size());
#ifdef USE_OMP
            #pragma omp parallel num_threads(score_thread_count)
#endif
            {
                MklLocalThreadGuard mkl_threads(1);
                int tid = 0;
#ifdef USE_OMP
                tid = omp_get_thread_num();
                #pragma omp single
                stream_stats_.score_threads =
                    std::max(stream_stats_.score_threads,
                             omp_get_num_threads());
                #pragma omp for schedule(dynamic, 1)
#endif
                for (std::size_t ti = 0; ti < score_tasks.size(); ++ti) {
                    try {
                        auto &thread_heap =
                            thread_heaps[static_cast<std::size_t>(tid)];
                        const auto &task = score_tasks[ti];
                        const auto &segment =
                            stream_rows_->segments()[task.segment];
                        const auto cached = segment.matrix();
                        const auto matrix = cached.middleRows(task.first, task.rows);
                        const auto scores = stream_state.score_rows(
                            matrix,
                            score_workspaces[static_cast<std::size_t>(tid)]);
                        for (int row = 0; row < task.rows; ++row) {
                            const auto &ref =
                                segment.refs[static_cast<std::size_t>(task.first + row)];
                            if (selected_mask[
                                    static_cast<std::size_t>(ref.global_row)] ||
                                scores[row] <= std::max(1.0, threshold))
                                continue;
                            thread_heap.emplace(
                                scores[row], -ref.global_row);
                            if (static_cast<int>(thread_heap.size()) >
                                shortlist_limit)
                                thread_heap.pop();
                        }
                    } catch (...) {
                        score_errors[ti] =
                            std::current_exception();
                    }
                }
            }
            for (const auto &error : score_errors)
                if (error) std::rethrow_exception(error);

            MinHeap heap;
            for (auto &thread_heap : thread_heaps)
                while (!thread_heap.empty()) {
                    heap.push(thread_heap.top());
                    thread_heap.pop();
                    if (static_cast<int>(heap.size()) > shortlist_limit)
                        heap.pop();
                }
            ++stream_stats_.score_passes;

            std::vector<HeapItem> local_top;
            local_top.reserve(heap.size());
            while (!heap.empty()) {
                local_top.push_back(heap.top());
                heap.pop();
            }
            std::vector<int> counts(static_cast<std::size_t>(size), 0);
            int local_count = static_cast<int>(local_top.size());
            MPI_Allgather(&local_count, 1, MPI_INT,
                          counts.data(), 1, MPI_INT, comm);
            std::vector<int> displs(static_cast<std::size_t>(size), 0);
            int total = 0;
            for (int r = 0; r < size; ++r) {
                displs[static_cast<std::size_t>(r)] = total;
                total += counts[static_cast<std::size_t>(r)];
            }
            std::vector<double> local_gains(local_top.size());
            std::vector<int> local_globals(local_top.size());
            for (std::size_t i = 0; i < local_top.size(); ++i) {
                local_gains[i] = local_top[i].first;
                local_globals[i] = -local_top[i].second;
            }
            std::vector<double> all_gains(static_cast<std::size_t>(total));
            std::vector<int> all_globals(static_cast<std::size_t>(total));
            MPI_Allgatherv(local_gains.data(), local_count, MPI_DOUBLE,
                           all_gains.data(), counts.data(), displs.data(),
                           MPI_DOUBLE, comm);
            MPI_Allgatherv(local_globals.data(), local_count, MPI_INT,
                           all_globals.data(), counts.data(), displs.data(),
                           MPI_INT, comm);
            std::vector<std::pair<double, int>> global_top;
            global_top.reserve(static_cast<std::size_t>(total));
            for (int i = 0; i < total; ++i)
                global_top.emplace_back(
                    all_gains[static_cast<std::size_t>(i)],
                    all_globals[static_cast<std::size_t>(i)]);
            std::sort(global_top.begin(), global_top.end(),
                      [](const auto &lhs, const auto &rhs) {
                          if (lhs.first != rhs.first)
                              return lhs.first > rhs.first;
                          return lhs.second < rhs.second;
                      });
            std::unordered_set<int> shortlisted;
            std::vector<int> shortlist_rows;
            shortlist_rows.reserve(
                static_cast<std::size_t>(shortlist_limit));
            for (const auto &[gain, global] : global_top) {
                (void)gain;
                if (!shortlisted.insert(global).second) continue;
                shortlist_rows.push_back(global);
                if (static_cast<int>(shortlist_rows.size()) >= shortlist_limit)
                    break;
            }
            if (shortlist_rows.empty()) break;

            OnlineRowMatrix shortlist(
                static_cast<int>(shortlist_rows.size()), n_coeff_);
            std::vector<BRowRef> shortlist_refs;
            shortlist_refs.reserve(shortlist_rows.size());
            for (int i = 0; i < static_cast<int>(shortlist_rows.size()); ++i) {
                Candidate candidate =
                    broadcast_candidate(shortlist_rows[static_cast<std::size_t>(i)]);
                shortlist.row(i) = candidate.row;
                shortlist_refs.push_back(candidate.ref);
            }
            const int remaining =
                max_swaps > 0 ? std::max(0, max_swaps - swap_count) : 0;
            if (max_swaps > 0 && remaining == 0) break;
            const int changed = stream_state.maximize_with_candidates(
                shortlist, shortlist_refs, threshold, remaining);
            if (changed <= 0) break;
            record_swaps(stream_state.last_swaps());
            swap_count += changed;
        }

        log_det_ = stream_state.log_determinant();
        if (max_swaps > 0 && swap_count >= max_swaps)
            break;
    }

    stream_stats_.swaps = swap_count;
    refs = stream_state.selected_refs();
    C = stream_state.active_rows();
    online_state_.initialize_from_rows(C, refs);
    if (!build_compact_selected_state(C, refs))
        return 0;
    log_det_ = stream_state.log_determinant();
    double local_max_grade = 0.0;
    std::vector<std::exception_ptr> final_errors(score_tasks.size());
#ifdef USE_OMP
    #pragma omp parallel num_threads(score_thread_count) reduction(max:local_max_grade)
#endif
    {
        MklLocalThreadGuard mkl_threads(1);
        int tid = 0;
#ifdef USE_OMP
        tid = omp_get_thread_num();
        #pragma omp single
        stream_stats_.score_threads =
            std::max(stream_stats_.score_threads, omp_get_num_threads());
        #pragma omp for schedule(dynamic, 1)
#endif
        for (std::size_t ti = 0; ti < score_tasks.size(); ++ti) {
            try {
                const auto &task = score_tasks[ti];
                const auto &segment = stream_rows_->segments()[task.segment];
                const auto scores = online_state_.score_rows(
                    segment.matrix().middleRows(task.first, task.rows),
                    score_workspaces[static_cast<std::size_t>(tid)]);
                if (scores.size() > 0)
                    local_max_grade = std::max(local_max_grade, scores.maxCoeff());
            } catch (...) {
                final_errors[ti] = std::current_exception();
            }
        }
    }
    for (const auto &error : final_errors)
        if (error) std::rethrow_exception(error);
    ++stream_stats_.score_passes;
    double max_grade = local_max_grade;
    MPI_Allreduce(&local_max_grade, &max_grade, 1, MPI_DOUBLE, MPI_MAX, comm);
    stream_stats_.score_seconds =
        std::chrono::duration<double>(Clock::now() - score_started).count();
    const bool ok = std::isfinite(max_grade) &&
        max_grade <= std::max(1.0, threshold) + 1e-8;
    const bool capped = max_swaps > 0 && swap_count >= max_swaps && !ok;
    set_selection_status(ok, max_grade,
                         ok ? "threshold" :
                         (capped ? "max-swaps" :
                          (stream_mode == "fast" ? "fast-pass-limit" :
                           "postcheck-failed")));
    stream_stats_.total_seconds =
        std::chrono::duration<double>(Clock::now() - total_started).count();
    online_processed_cfgs_ = ok ? cfgs_.size() : 0;
    return static_cast<int>(selected_cfgs_.size());
}

int ActiveSelector::select_streaming_legacy_current()
{
    invalidate_grading();
    if (!builder_ || cfgs_.empty()) return 0;
    n_coeff_ = builder_->coeff_count();
    if (n_coeff_ <= 0) return 0;

    struct Candidate {
        Eigen::RowVectorXd row;
        BRowRef ref;
        double norm2 = -1.0;
    };

    auto update_norm_reservoir = [](std::vector<Candidate> &reservoir,
                                    const Candidate &cand,
                                    int limit) {
        if (limit <= 0) return;
        if (static_cast<int>(reservoir.size()) < limit) {
            reservoir.push_back(cand);
            return;
        }
        int worst = 0;
        for (int i = 1; i < static_cast<int>(reservoir.size()); ++i) {
            if (reservoir[i].norm2 < reservoir[worst].norm2)
                worst = i;
        }
        if (cand.norm2 > reservoir[worst].norm2)
            reservoir[worst] = cand;
    };

    auto linearly_independent = [&](const std::vector<Candidate> &basis,
                                    const Candidate &cand) {
        if (cand.norm2 <= 0.0 || !std::isfinite(cand.norm2))
            return false;
        if (basis.empty())
            return true;
        std::vector<Eigen::RowVectorXd> orthogonal_rows;
        orthogonal_rows.reserve(basis.size());
        for (const auto &base : basis) {
            Eigen::RowVectorXd residual = base.row;
            for (const auto &q : orthogonal_rows) {
                const double qn = q.squaredNorm();
                if (qn > 0.0) {
                    const double alpha = residual.dot(q) / qn;
                    residual -= alpha * q;
                }
            }
            const double base_tol = std::numeric_limits<double>::epsilon()
                                  * static_cast<double>(std::max(1, n_coeff_))
                                  * std::max(1.0, base.norm2);
            if (residual.squaredNorm() > base_tol)
                orthogonal_rows.push_back(residual);
        }
        Eigen::RowVectorXd residual = cand.row;
        for (const auto &q : orthogonal_rows) {
            const double qn = q.squaredNorm();
            if (qn > 0.0) {
                const double alpha = residual.dot(q) / qn;
                residual -= alpha * q;
            }
        }
        const double tol = std::numeric_limits<double>::epsilon()
                         * static_cast<double>(std::max(1, n_coeff_))
                         * std::max(1.0, cand.norm2);
        return residual.squaredNorm() > tol;
    };

    stream_swap_history_.clear();
    selected_row_refs_.clear();
    selected_row_indices_.clear();
    selected_cfgs_.clear();
    selected_config_indices_.clear();
    B_.resize(0, n_coeff_);
    B_full_.resize(0, n_coeff_);
    has_full_B_ = false;
    using_stream_state_ = true;
    online_state_.clear();
    online_processed_cfgs_ = 0;

    std::vector<Candidate> initial_basis;
    std::vector<Candidate> norm_reservoir;
    initial_basis.reserve(static_cast<size_t>(std::max(1, n_coeff_)));
    norm_reservoir.reserve(static_cast<size_t>(std::max(1, n_coeff_)));
    cfg_row_start_.clear();
    cfg_row_start_.push_back(0);

    int global_row = 0;
    int usable_rows = 0;
    for (size_t ci = 0; ci < cfgs_.size(); ++ci) {
        Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_cfg;
        std::vector<BRowRef> row_refs;
        const int nrows = builder_->build_rows(
            cfgs_[ci], B_cfg, weight, pow_scale, &row_refs);
        for (int lr = 0; lr < nrows; ++lr) {
            Candidate cand;
            cand.row = B_cfg.row(lr);
            if (lr < static_cast<int>(row_refs.size()))
                cand.ref = row_refs[lr];
            cand.ref.cfg_index = static_cast<int>(ci);
            cand.ref.local_row = lr;
            cand.ref.global_row = global_row;
            cand.norm2 = cand.row.squaredNorm();

            if (std::isfinite(cand.norm2)) {
                ++usable_rows;
                update_norm_reservoir(norm_reservoir, cand, n_coeff_);
                if (static_cast<int>(initial_basis.size()) < n_coeff_ &&
                    linearly_independent(initial_basis, cand))
                    initial_basis.push_back(cand);
            }
            ++global_row;
        }
        cfg_row_start_.push_back(global_row);
    }

    if (global_row <= 0 || usable_rows <= 0) return 0;

    const int k = std::min(usable_rows, n_coeff_);
    std::vector<Candidate> initial = initial_basis;
    if (static_cast<int>(initial.size()) < k) {
        std::unordered_set<int> used_globals;
        used_globals.reserve(static_cast<size_t>(k * 2 + 1));
        for (const auto &cand : initial)
            used_globals.insert(cand.ref.global_row);

        std::sort(norm_reservoir.begin(), norm_reservoir.end(),
                  [](const Candidate &a, const Candidate &b) {
                      if (a.norm2 != b.norm2) return a.norm2 > b.norm2;
                      return a.ref.global_row < b.ref.global_row;
                  });
        for (const auto &cand : norm_reservoir) {
            if (static_cast<int>(initial.size()) >= k) break;
            if (!used_globals.insert(cand.ref.global_row).second) continue;
            initial.push_back(cand);
        }
    }
    if (initial.empty()) return 0;

    std::sort(initial.begin(), initial.end(),
              [](const Candidate &a, const Candidate &b) {
                  return a.ref.global_row < b.ref.global_row;
              });

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> C(k, n_coeff_);
    std::vector<BRowRef> refs;
    refs.reserve(k);
    for (int i = 0; i < k; ++i) {
        C.row(i) = initial[i].row;
        refs.push_back(initial[i].ref);
    }

    const int pass_limit = (stream_mode == "fast")
        ? std::max(1, stream_fast_passes)
        : std::max(1, mv_.max_iterations);
    int swap_count = 0;

    // The same LU/LDLT state is used for cold-start streaming and persistent
    // online updates, so underdetermined states retain orthogonal-residual
    // information instead of relying on pseudo-inverse coefficients alone.
    OnlineMaxVolState stream_state;
    if (!stream_state.initialize_from_rows(C, refs))
        return 0;
    log_det_ = stream_state.log_determinant();

    // ── Pre-check: can the builder be cloned? ────────────────────────────
    bool can_clone = false;
    {
        auto probe = std::unique_ptr<BMatrixBuilder>(builder_->clone());
        can_clone = (probe != nullptr);
    }

    const std::size_t selection_chunk =
        static_cast<std::size_t>(std::max(1, stream_chunk_size));
    const std::size_t selection_chunks =
        (cfgs_.size() + selection_chunk - 1) / selection_chunk;

    for (int pass = 0; pass < pass_limit; ++pass) {
        std::unordered_set<int> selected_globals;
        selected_globals.reserve(refs.size() * 2 + 1);
        for (const auto &ref : refs)
            selected_globals.insert(ref.global_row);

        double best_gain = threshold;
        BRowRef best_ref;
        Eigen::RowVectorXd best_row;

        auto scan_chunk = [&](BMatrixBuilder &scan_builder,
                              std::size_t chunk,
                              double &local_best_gain,
                              BRowRef &local_best_ref,
                              Eigen::RowVectorXd &local_best_row) {
            const std::size_t beg = chunk * selection_chunk;
            const std::size_t end = std::min(beg + selection_chunk, cfgs_.size());
            OnlineRowMatrix B_chunk;
            std::vector<BRowRef> chunk_refs;
            for (std::size_t ci = beg; ci < end; ++ci) {
                const int row_start = static_cast<int>(B_chunk.rows());
                const int nrows = scan_builder.build_rows(
                    cfgs_[ci], B_chunk, weight, pow_scale, &chunk_refs);
                if (static_cast<int>(chunk_refs.size()) < row_start + nrows)
                    chunk_refs.resize(static_cast<size_t>(row_start + nrows));
                for (int lr = 0; lr < nrows; ++lr) {
                    BRowRef &ref = chunk_refs[static_cast<size_t>(row_start + lr)];
                    ref.cfg_index = static_cast<int>(ci);
                    ref.local_row = lr;
                    ref.global_row = cfg_row_start_[ci] + lr;
                }
            }
            if (B_chunk.rows() == 0) return;

            const Eigen::VectorXd scores = stream_state.score_rows(B_chunk);
            if (scores.size() != B_chunk.rows()) return;
            for (int row = 0; row < B_chunk.rows(); ++row) {
                const BRowRef &ref = chunk_refs[static_cast<size_t>(row)];
                if (selected_globals.count(ref.global_row)) continue;
                const double gain = scores[row];
                const bool better = gain > local_best_gain ||
                    (gain == local_best_gain && gain > threshold &&
                     (local_best_ref.global_row < 0 ||
                      ref.global_row < local_best_ref.global_row));
                if (better) {
                    local_best_gain = gain;
                    local_best_ref = ref;
                    local_best_row = B_chunk.row(row);
                }
            }
        };

        // ── A: Scan all configs (parallel if builder supports clone) ───
        if (can_clone) {
            #pragma omp parallel
            {
                auto local_builder = std::unique_ptr<BMatrixBuilder>(builder_->clone());

                double local_best_gain = threshold;
                BRowRef local_best_ref;
                Eigen::RowVectorXd local_best_row;

                #pragma omp for schedule(dynamic, 1)
                for (std::size_t chunk = 0; chunk < selection_chunks; ++chunk)
                    scan_chunk(*local_builder, chunk, local_best_gain,
                               local_best_ref, local_best_row);

                #pragma omp critical
                {
                    if (local_best_gain > best_gain ||
                        (local_best_gain == best_gain && local_best_gain > threshold &&
                         (best_ref.global_row < 0 ||
                          local_best_ref.global_row < best_ref.global_row))) {
                        best_gain = local_best_gain;
                        best_ref = local_best_ref;
                        best_row = local_best_row;
                    }
                }
            }
        } else {
            // Serial fallback for non-cloneable builders
            for (std::size_t chunk = 0; chunk < selection_chunks; ++chunk)
                scan_chunk(*builder_, chunk, best_gain, best_ref, best_row);
        }

        if (best_row.size() != n_coeff_) break;

        const std::vector<BRowRef> old_refs = stream_state.selected_refs();
        OnlineRowMatrix one_row(1, n_coeff_);
        one_row.row(0) = best_row;
        if (stream_state.maximize_with_candidates(
                one_row, std::vector<BRowRef>{best_ref}, threshold, 1) != 1)
            break;
        refs = stream_state.selected_refs();
        C = stream_state.active_rows();
        int best_slot = -1;
        for (int slot = 0; slot < k; ++slot) {
            if (refs[slot].global_row == best_ref.global_row &&
                old_refs[slot].global_row != refs[slot].global_row) {
                best_slot = slot;
                break;
            }
        }
        if (best_slot < 0) break;

        stream_swap_history_.emplace_back(
            best_ref.global_row, old_refs[best_slot].global_row, best_gain);
        if (verbose) {
            std::cout << "Stream MaxVol swap: row "
                      << old_refs[best_slot].global_row << " out, row "
                      << best_ref.global_row << " in (gain "
                      << best_gain << ")\n";
        }
        log_det_ = stream_state.log_determinant();

        if (max_swaps > 0 && ++swap_count >= max_swaps) break;
    }

    if (!build_compact_selected_state(C, refs))
        return 0;
    log_det_ = stream_state.log_determinant();
    return static_cast<int>(selected_cfgs_.size());
}

int ActiveSelector::select_online_current()
{
    invalidate_grading();
    if (!builder_ || cfgs_.empty()) return 0;
    n_coeff_ = builder_->coeff_count();
    if (n_coeff_ <= 0) return 0;

    // Initialize an online selector from an existing compact active matrix.
    if (!online_state_.valid() && B_full_.rows() > 0 &&
        B_full_.cols() == n_coeff_ &&
        selected_row_refs_.size() == static_cast<size_t>(B_full_.rows())) {
        online_state_.initialize_from_rows(B_full_, selected_row_refs_);
        if (online_processed_cfgs_ == 0) {
            const std::size_t historical_cfgs = cfg_row_start_.empty()
                ? 0 : cfg_row_start_.size() - 1;
            online_processed_cfgs_ = std::min(historical_cfgs, cfgs_.size());
        }
    }

    // A brand-new online selector uses the existing streaming backend once
    // to construct a well-conditioned initial active set.
    if (!online_state_.valid()) {
        if (select_streaming_current() <= 0)
            return 0;
        if (!online_state_.valid() &&
            !online_state_.initialize_from_rows(B_full_, selected_row_refs_))
            return 0;
        online_processed_cfgs_ = converged_ ? cfgs_.size() : 0;
        release_stream_rows();
        return static_cast<int>(selected_cfgs_.size());
    }

    // Online updates never consult the historical row cache. In particular,
    // do not retain the stream initialization's O(m*n) storage indefinitely.
    release_stream_rows();
    if (online_processed_cfgs_ >= cfgs_.size()) {
        if (std::isfinite(max_remaining_grade_)) {
            const bool ok = max_remaining_grade_ <= std::max(1.0, threshold) + 1e-8;
            set_selection_status(ok, max_remaining_grade_,
                                 ok ? "threshold" : "online-approximate");
        }
        return static_cast<int>(selected_cfgs_.size());
    }

    if (cfg_row_start_.empty())
        cfg_row_start_.push_back(0);
    while (cfg_row_start_.size() < online_processed_cfgs_ + 1)
        cfg_row_start_.push_back(cfg_row_start_.back());

    int remaining_swaps = max_swaps;
    std::size_t processed_after = online_processed_cfgs_;
    for (size_t ci = online_processed_cfgs_; ci < cfgs_.size(); ++ci) {
        OnlineRowMatrix B_cfg;
        std::vector<BRowRef> refs;
        const int nrows = builder_->build_rows(
            cfgs_[ci], B_cfg, weight, pow_scale, &refs);
        while (cfg_row_start_.size() < ci + 1)
            cfg_row_start_.push_back(cfg_row_start_.back());
        const int row_base = cfg_row_start_[ci];
        if (refs.size() != static_cast<size_t>(nrows))
            refs.resize(static_cast<size_t>(nrows));
        for (int lr = 0; lr < nrows; ++lr) {
            refs[lr].cfg_index = static_cast<int>(ci);
            refs[lr].local_row = lr;
            refs[lr].global_row = row_base + lr;
        }

        const int limit = max_swaps > 0 ? remaining_swaps : 0;
        const int swaps = online_state_.maximize_with_candidates(
            B_cfg, refs, threshold, limit);
        if (max_swaps > 0)
            remaining_swaps = std::max(0, remaining_swaps - swaps);
        if (cfg_row_start_.size() == ci + 1)
            cfg_row_start_.push_back(row_base + nrows);
        processed_after = ci + 1;
        if (max_swaps > 0 && remaining_swaps == 0) {
            // Revisit the current configuration on the next call: reaching
            // the limit does not prove that all of its rows are below the
            // threshold. Row counts for later configurations are still
            // recorded so the partially processed state remains persistable.
            processed_after = ci;
            for (++ci; ci < cfgs_.size(); ++ci) {
                if (cfg_row_start_.size() > ci + 1)
                    continue;
                int extra = builder_->row_count(cfgs_[ci]);
                if (extra < 0) {
                    OnlineRowMatrix ignored;
                    extra = builder_->build_rows(
                        cfgs_[ci], ignored, weight, pow_scale);
                }
                cfg_row_start_.push_back(cfg_row_start_.back() + extra);
            }
            break;
        }
    }
    online_processed_cfgs_ = processed_after;

    selected_row_refs_ = online_state_.selected_refs();
    if (!build_compact_selected_state(
            online_state_.active_rows(), selected_row_refs_))
        return 0;
    log_det_ = online_state_.log_determinant();
    // A later exchange can raise the grade of a rejected historical row.
    // Rebuilding all historical descriptors here defeats incremental updates;
    // leave the global grade unknown until the caller explicitly grades or
    // reselects the complete pool.
    set_selection_status(false, std::numeric_limits<double>::infinity(),
                         max_swaps > 0 && remaining_swaps == 0
                             ? "max-swaps" : "online-approximate");
    return static_cast<int>(selected_cfgs_.size());
}

// ═══════════════════════════════════════════════════════════════════════════
// Grading — private helper
// ═══════════════════════════════════════════════════════════════════════════

double ActiveSelector::grade_one(
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &B_cfg, int nrows)
{
    if (nrows <= 0) return 1e10;

    Eigen::VectorXd scores = online_state_.valid()
        ? online_state_.score_rows(B_cfg)
        : mv_.score_rows(B_cfg, selected_row_indices_);
    if (scores.size() != nrows) return 1e10;
    double grade = scores.maxCoeff();
    return std::isfinite(grade) ? grade : 1e10;
}

// ═══════════════════════════════════════════════════════════════════════════
// Grading — public API
// ═══════════════════════════════════════════════════════════════════════════

double ActiveSelector::grade(Atoms &cfg)
{
    if (B_full_.rows() == 0 || !builder_) return 1e10;

    // Build B rows for this config alone
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_cfg;
    int nrows = builder_->build_rows(cfg, B_cfg, weight, pow_scale);

    if (nrows <= 0) return 1e10;

    if (!prepare_grading()) return 1e10;

    return grade_one(B_cfg, nrows);
}

void ActiveSelector::score_candidate_chunk(std::vector<Atoms> &cfgs,
                                           std::size_t beg,
                                           std::size_t end,
                                           BMatrixBuilder &builder,
                                           std::vector<double> &grades)
{
    if (beg >= end) return;

    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> B_chunk;
    B_chunk.resize(0, n_coeff_);

    std::vector<int> row_ranges;
    row_ranges.reserve(end - beg + 1);
    row_ranges.push_back(0);

    for (std::size_t i = beg; i < end; ++i) {
        builder.build_rows(cfgs[i], B_chunk, weight, pow_scale);
        row_ranges.push_back(static_cast<int>(B_chunk.rows()));
    }

    if (B_chunk.rows() == 0) {
        for (std::size_t i = beg; i < end; ++i)
            grades[i] = 1e10;
        return;
    }

    Eigen::VectorXd row_scores = online_state_.valid()
        ? online_state_.score_rows(B_chunk)
        : mv_.score_rows(B_chunk, selected_row_indices_);
    if (row_scores.size() != B_chunk.rows()) {
        for (std::size_t i = beg; i < end; ++i)
            grades[i] = 1e10;
        return;
    }

    for (std::size_t i = beg; i < end; ++i) {
        const std::size_t local = i - beg;
        const int start = row_ranges[local];
        const int stop = row_ranges[local + 1];
        double grade = 1e10;
        if (stop > start) {
            grade = 0.0;
            for (int row = start; row < stop; ++row)
                grade = std::max(grade, row_scores[row]);
            if (!std::isfinite(grade))
                grade = 1e10;
        }
        grades[i] = grade;
        if (verbose) {
            #pragma omp critical
            std::cout << fmt::format("  [{}/{}] grade = {:.4f}\n",
                                     i + 1, cfgs.size(), grade)
                      << std::flush;
        }
    }
}

bool ActiveSelector::prepare_grading()
{
    if (selected_row_indices_.empty() || B_full_.rows() == 0)
        return false;
    if (grading_prepared_)
        return true;
    if (online_state_.valid()) {
        log_det_ = online_state_.log_determinant();
        grading_prepared_ = online_state_.active_size() == B_full_.rows() &&
                            online_state_.coeff_count() == B_full_.cols();
        return grading_prepared_;
    }
    mv_.warm_cache(selected_row_indices_, B_full_);
    log_det_ = factorize_active_rows(B_full_).log_volume;
    grading_prepared_ = true;
    return true;
}

bool ActiveSelector::set_active_rows_for_grading(
    const Eigen::Matrix<double, -1, -1, Eigen::RowMajor> &C,
    const std::vector<int> &selected_config_indices,
    const std::vector<Atoms> *all_cfgs)
{
    invalidate_grading();
    const int k = static_cast<int>(C.rows());
    const int ncols = static_cast<int>(C.cols());
    if (k <= 0 || ncols <= 0)
        return false;

    n_coeff_ = ncols;
    release_stream_rows();
    B_.resize(0, n_coeff_);
    B_full_ = C;
    has_full_B_ = false;
    using_stream_state_ = true;

    selected_row_indices_.resize(k);
    std::iota(selected_row_indices_.begin(), selected_row_indices_.end(), 0);
    selected_row_refs_.clear();
    selected_config_indices_ = selected_config_indices;

    selected_cfgs_.clear();
    if (all_cfgs) {
        for (int idx : selected_config_indices_) {
            if (idx >= 0 && idx < static_cast<int>(all_cfgs->size()))
                selected_cfgs_.push_back((*all_cfgs)[idx]);
        }
    }
    if (selected_cfgs_.empty())
        selected_cfgs_.push_back(Atoms{});

    cfgs_.clear();
    if (all_cfgs)
        cfgs_ = *all_cfgs;
    is_new_.assign(cfgs_.size(), false);
    cfg_row_start_.clear();
    cfg_row_start_.push_back(0);
    cfg_row_start_.push_back(k);
    online_state_.clear();
    if (select_mode == "online")
        online_state_.initialize_from_rows(B_full_, {});
    log_det_ = online_state_.valid()
             ? online_state_.log_determinant()
             : factorize_active_rows(B_full_).log_volume;

    return prepare_grading();
}

bool ActiveSelector::broadcast_grading_state(MPI_Comm comm,
                                             const std::vector<Atoms> *all_cfgs,
                                             int root)
{
    int rank = 0;
    MPI_Comm_rank(comm, &rank);

    int meta[4] = {0, 0, 0, 0};
    std::vector<int> cfg_indices;
    std::vector<double> c_data;

    if (rank == root) {
        const int k = static_cast<int>(selected_row_indices_.size());
        const int ncols = static_cast<int>(B_full_.cols());
        bool ok = (k > 0 && ncols > 0 &&
                   static_cast<int>(B_full_.rows()) > 0);
        if (ok) {
            cfg_indices = selected_config_indices_;
            Eigen::Matrix<double, -1, -1, Eigen::RowMajor> C(k, ncols);
            for (int i = 0; i < k; ++i) {
                const int ridx = selected_row_indices_[i];
                if (ridx < 0 || ridx >= static_cast<int>(B_full_.rows())) {
                    ok = false;
                    break;
                }
                C.row(i) = B_full_.row(ridx);
            }
            if (ok) {
                c_data.assign(C.data(), C.data() + C.size());
                meta[0] = 1;
                meta[1] = k;
                meta[2] = ncols;
                meta[3] = static_cast<int>(cfg_indices.size());
            }
        }
    }

    MPI_Bcast(meta, 4, MPI_INT, root, comm);
    if (!meta[0])
        return false;

    const int k = meta[1];
    const int ncols = meta[2];
    const int ncfgidx = meta[3];

    if (rank != root)
        cfg_indices.resize(ncfgidx);
    if (ncfgidx > 0)
        MPI_Bcast(cfg_indices.data(), ncfgidx, MPI_INT, root, comm);

    const int ndata = k * ncols;
    if (rank != root)
        c_data.resize(ndata);
    if (ndata > 0)
        MPI_Bcast(c_data.data(), ndata, MPI_DOUBLE, root, comm);

    if (rank == root)
        return prepare_grading();

    Eigen::Map<const Eigen::Matrix<double, -1, -1, Eigen::RowMajor>>
        C(c_data.data(), k, ncols);
    Eigen::Matrix<double, -1, -1, Eigen::RowMajor> C_copy = C;
    return set_active_rows_for_grading(C_copy, cfg_indices, all_cfgs);
}

void ActiveSelector::grade_all(std::vector<Atoms> &cfgs,
                                std::vector<double> &grades)
{
    grades.resize(cfgs.size());
    if (!builder_ || !prepare_grading()) {
        std::fill(grades.begin(), grades.end(), 1e10);
        return;
    }
    if (cfgs.empty())
        return;

    const std::size_t chunk_size =
        static_cast<std::size_t>(std::max(1, stream_chunk_size));
    const std::size_t nchunks = (cfgs.size() + chunk_size - 1) / chunk_size;

    // Try clone; fall back to two-phase if builder doesn't support it
    auto trial = builder_->clone();
    if (!trial) {
        // Non-cloneable builders are constructed serially, but each bounded
        // chunk is still scored as one matrix multiplication.
        for (std::size_t chunk = 0; chunk < nchunks; ++chunk) {
            const std::size_t beg = chunk * chunk_size;
            const std::size_t end = std::min(beg + chunk_size, cfgs.size());
            score_candidate_chunk(cfgs, beg, end, *builder_, grades);
        }
        return;
    }
    trial.reset();

    // Parallel chunked grading using cloned builders.
    #pragma omp parallel
    {
        auto local_builder = std::unique_ptr<BMatrixBuilder>(builder_->clone());

        #pragma omp for schedule(dynamic, 1)
        for (std::size_t chunk = 0; chunk < nchunks; ++chunk) {
            const std::size_t beg = chunk * chunk_size;
            const std::size_t end = std::min(beg + chunk_size, cfgs.size());
            if (local_builder)
                score_candidate_chunk(cfgs, beg, end, *local_builder, grades);
            else
                std::fill(grades.begin() + beg, grades.begin() + end, 1e10);
        }
    }
}

void ActiveSelector::compute_grades(std::vector<double> &grades)
{
    int nc = num_candidates();
    grades.clear();
    if (nc == 0 || selected_row_indices_.empty() || B_full_.rows() == 0) {
        grades.assign(nc, 0.0);
        return;
    }
    if (online_state_.valid()) {
        // A serial stream run retains every B row. Reuse it for --score-file
        // and explicit pool grading instead of evaluating descriptors again.
        // Distributed caches are only usable here if this rank owns all rows.
        std::size_t cached_rows = 0;
        if (stream_rows_)
            for (const auto &segment : stream_rows_->segments())
                cached_rows += static_cast<std::size_t>(segment.rows);
        if (stream_rows_ && cfg_row_start_.size() == cfgs_.size() + 1 &&
            cached_rows == static_cast<std::size_t>(cfg_row_start_.back())) {
            grades.assign(nc, 0.0);
            OnlineMaxVolState::ScoreWorkspace workspace;
            const int block_rows = online_state_.score_block_rows();
            for (const auto &segment : stream_rows_->segments()) {
                const auto matrix = segment.matrix();
                for (int first = 0; first < segment.rows; first += block_rows) {
                    const int count = std::min(block_rows, segment.rows - first);
                    const auto scores = online_state_.score_rows(
                        matrix.middleRows(first, count), workspace);
                    for (int row = 0; row < count; ++row) {
                        const int ci = segment.refs[static_cast<std::size_t>(first + row)].cfg_index;
                        const double score = std::isfinite(scores[row]) ? scores[row] : 1e10;
                        grades[ci] = std::max(grades[ci], score);
                    }
                }
            }
            for (int ci = 0; ci < nc; ++ci)
                if (cfg_row_start_[ci] == cfg_row_start_[ci + 1])
                    grades[ci] = 1e10;
            return;
        }
        if (!builder_) {
            grades.assign(nc, 1e10);
            return;
        }
        grade_all(cfgs_, grades);
        return;
    }
    if (has_full_B_ && B_full_.rows() == cfg_row_start_.back()) {
        // Match grade()/grade_all() semantics: selected rows are evaluated by
        // the same cached factorization instead of being overwritten with an
        // exact 1.0 by compute_scores(). This also keeps grades continuous
        // across a compact-state save, load, and full-B rebuild.
        mv_.warm_cache(selected_row_indices_, B_full_);
        const Eigen::VectorXd row_scores =
            mv_.score_rows(B_full_, selected_row_indices_);
        if (row_scores.size() != B_full_.rows()) {
            grades.assign(nc, 1e10);
            return;
        }
        grades.assign(nc, 0.0);
        for (int ci = 0; ci < nc; ++ci)
            for (int row = cfg_row_start_[ci]; row < cfg_row_start_[ci + 1]; ++row)
                grades[ci] = std::max(grades[ci], row_scores[row]);
        return;
    }
    if (!builder_) {
        grades.assign(nc, 1e10);
        return;
    }
    grade_all(cfgs_, grades);
}

// ═══════════════════════════════════════════════════════════════════════════
// Persistence  (v10 — virtual rows + convergence metadata)
// ═══════════════════════════════════════════════════════════════════════════

void ActiveSelector::save_state(const std::string &filename,
                                 const std::string &index_file,
                                 const std::string &pot_file,
                                 const std::string &pottype) const
{
    if(GPM::extension(filename)) {
        std::unique_ptr<Potentials::potential_wrapper> loaded;
        Potentials::AnyPotential *model=nullptr;
        if(auto *b=dynamic_cast<PotEFSBuilder*>(builder_))model=&b->potential();
        else if(!pot_file.empty()){loaded=std::make_unique<Potentials::potential_wrapper>(pot_file,pottype);model=loaded->pot;}
        GPM::require(model,"GPM selector export requires its target potential");
        GPM::Writer w;Potentials::GPMCodecAccess::encode(w,*model);
        w.manifest["species"]=w.manifest["models"].get<GPM::Object>().at("root").get<GPM::Object>().at("species");
        w.manifest["model_fingerprint"]=GPM::Value(Potentials::ModelBundle::fingerprint(*model));
        AL::GPMALAccess::encode_selector(w,*const_cast<ActiveSelector*>(this),*model);w.save(filename);return;
    }

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) GPERROR("Cannot open MaxVol state for writing: " + filename);

    const int magic = 0x41435456, version = 10;
    ofs.write(reinterpret_cast<const char*>(&magic), sizeof(magic));
    ofs.write(reinterpret_cast<const char*>(&version), sizeof(version));
    ofs.write(reinterpret_cast<const char*>(&n_coeff_), sizeof(n_coeff_));
    ofs.write(reinterpret_cast<const char*>(&threshold), sizeof(threshold));
    ofs.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    ofs.write(reinterpret_cast<const char*>(&pow_scale), sizeof(pow_scale));
    const int scaling = static_cast<int>(
        builder_ ? builder_->row_scaling : pending_builder_.row_scaling);
    ofs.write(reinterpret_cast<const char*>(&scaling), sizeof(scaling));
    const int persisted_init_mode = uses_mlip_initialization() ? 1 : 0;
    ofs.write(reinterpret_cast<const char*>(&persisted_init_mode), sizeof(int));
    ofs.write(reinterpret_cast<const char*>(&init_value), sizeof(init_value));
    const char persisted_converged = converged_ ? 1 : 0;
    ofs.write(&persisted_converged, 1);
    ofs.write(reinterpret_cast<const char*>(&max_remaining_grade_), sizeof(double));
    const int reason_size = static_cast<int>(termination_reason_.size());
    ofs.write(reinterpret_cast<const char*>(&reason_size), sizeof(int));
    ofs.write(termination_reason_.data(), reason_size);
    int nsel = static_cast<int>(selected_cfgs_.size());
    ofs.write(reinterpret_cast<const char*>(&nsel), sizeof(nsel));

    // ── Compact active C matrix: selected B rows only ────────────────
    int k = static_cast<int>(selected_row_indices_.size());
    int total_rows = cfg_row_start_.empty() ? 0 : cfg_row_start_.back();
    ofs.write(reinterpret_cast<const char*>(&k), sizeof(k));
    ofs.write(reinterpret_cast<const char*>(&total_rows), sizeof(total_rows));

    std::vector<BRowRef> refs = selected_row_refs_;
    if (static_cast<int>(refs.size()) != k) {
        refs.clear();
        refs.reserve(k);
        for (int i = 0; i < k; ++i) {
            BRowRef ref;
            int ridx = selected_row_indices_[i];
            ref.global_row = ridx;
            ref.cfg_index = -1;
            ref.local_row = -1;
            for (int ci = 0; ci < static_cast<int>(cfg_row_start_.size()) - 1; ++ci) {
                if (ridx >= cfg_row_start_[ci] && ridx < cfg_row_start_[ci + 1]) {
                    ref.cfg_index = ci;
                    ref.local_row = ridx - cfg_row_start_[ci];
                    break;
                }
            }
            refs.push_back(ref);
        }
    }
    for (int i = 0; i < k; ++i) {
        ofs.write(reinterpret_cast<const char*>(&refs[i].cfg_index), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&refs[i].local_row), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&refs[i].global_row), sizeof(int));
        const int kind = static_cast<int>(refs[i].kind);
        ofs.write(reinterpret_cast<const char*>(&kind), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&refs[i].atom_index), sizeof(int));
        ofs.write(reinterpret_cast<const char*>(&refs[i].component), sizeof(int));
    }

    for (int i = 0; i < k; ++i) {
        int ridx = selected_row_indices_[i];
        for (int j = 0; j < n_coeff_; ++j) {
            double v = 0.0;
            if (ridx >= 0 && ridx < static_cast<int>(B_full_.rows()))
                v = B_full_(ridx, j);
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
        }
    }

    // ── Training configs (EXTXYZ block with length prefix) ───────────
    {
        std::ostringstream oss;
        GPIO iobj;
        if (!index_file.empty()) iobj.readmap(index_file);
        std::vector<Atoms> cfgs_copy = cfgs_;
        if (!iobj.write(cfgs_copy, oss, "extxyz"))
            GPERROR("Failed to serialize MaxVol configurations: " + filename);
        std::string cfg_data = oss.str();
        int cfg_bytes = static_cast<int>(cfg_data.size());
        ofs.write(reinterpret_cast<const char*>(&cfg_bytes), sizeof(cfg_bytes));
        ofs.write(cfg_data.data(), cfg_bytes);
    }

    // ── Selected configs (EXTXYZ block) ──────────────────────────────
    {
        std::ostringstream oss;
        GPIO iobj;
        if (!index_file.empty()) iobj.readmap(index_file);
        std::vector<Atoms> sel_copy = selected_cfgs_;
        if (!iobj.write(sel_copy, oss, "extxyz"))
            GPERROR("Failed to serialize MaxVol configurations: " + filename);
        std::string sel_data = oss.str();
        int sel_bytes = static_cast<int>(sel_data.size());
        ofs.write(reinterpret_cast<const char*>(&sel_bytes), sizeof(sel_bytes));
        ofs.write(sel_data.data(), sel_bytes);
    }

    // ── Builder metadata (no pot paths — caller provides pot) ───────
    if (builder_) {
        int tag = 0x4255494C;  // "BUIL"
        ofs.write(reinterpret_cast<const char*>(&tag), sizeof(tag));
        builder_->save_config(ofs);
    }

    // ── cfg_row_start_ (v6+) ──────────────────────────────────────
    {
        int ncfg = static_cast<int>(cfg_row_start_.size());
        ofs.write(reinterpret_cast<const char*>(&ncfg), sizeof(ncfg));
        for (int v : cfg_row_start_)
            ofs.write(reinterpret_cast<const char*>(&v), sizeof(v));
    }

    const int processed = static_cast<int>(
        std::min(online_processed_cfgs_, cfgs_.size()));
    ofs.write(reinterpret_cast<const char*>(&processed), sizeof(processed));
    const char has_online = online_state_.valid() ? 1 : 0;
    ofs.write(&has_online, 1);
    if (has_online)
        online_state_.save(ofs);
    ofs.close();
    if (!ofs) GPERROR("Failed to write MaxVol state: " + filename);
}

void ActiveSelector::load_state(const std::string &filename,
                                 const std::string &index_file)
{
    auto clear_loaded_state = [&]() {
        cfgs_.clear();
        selected_cfgs_.clear();
        selected_config_indices_.clear();
        selected_row_indices_.clear();
        selected_row_refs_.clear();
        stream_swap_history_.clear();
        is_new_.clear();
        n_coeff_ = builder_ ? builder_->coeff_count() : 0;
        B_.resize(0, n_coeff_);
        B_full_.resize(0, n_coeff_);
        cfg_row_start_.clear();
        cfg_row_start_.push_back(0);
        pending_builder_ = BuilderParams{};
        log_det_ = 0.0;
        using_stream_state_ = false;
        has_full_B_ = false;
        grading_prepared_ = false;
        online_state_.clear();
        online_processed_cfgs_ = 0;
        stream_rows_.reset();
        stream_stats_ = {};
        reset_selection_status();
    };

    auto fail_load = [&]() {
        clear_loaded_state();
        GPERROR("Cannot load MaxVol state (missing, invalid or truncated): " + filename);
    };

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) { fail_load(); }

    int magic = 0, version = 0;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (!ifs || magic != 0x41435456 ||
        version != 10) {
        fail_load();
    }

    // Clear existing state before loading
    clear_loaded_state();

    ifs.read(reinterpret_cast<char*>(&n_coeff_), sizeof(n_coeff_));
    ifs.read(reinterpret_cast<char*>(&threshold), sizeof(threshold));
    ifs.read(reinterpret_cast<char*>(&weight), sizeof(weight));
    ifs.read(reinterpret_cast<char*>(&pow_scale), sizeof(pow_scale));
    int scaling = static_cast<int>(RowScalingMode::Uniform);
    ifs.read(reinterpret_cast<char*>(&scaling), sizeof(scaling));
    if (scaling < static_cast<int>(RowScalingMode::Uniform) ||
        scaling > static_cast<int>(RowScalingMode::MlipCompatible)) {
        fail_load();
    }
    pending_builder_.row_scaling = static_cast<RowScalingMode>(scaling);
    if (builder_)
        builder_->row_scaling = pending_builder_.row_scaling;
    {
        int persisted_init_mode = 0;
        char persisted_converged = 0;
        int reason_size = 0;
        ifs.read(reinterpret_cast<char*>(&persisted_init_mode), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&init_value), sizeof(init_value));
        ifs.read(&persisted_converged, 1);
        ifs.read(reinterpret_cast<char*>(&max_remaining_grade_), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&reason_size), sizeof(int));
        if (!ifs || (persisted_init_mode != 0 && persisted_init_mode != 1) ||
            !std::isfinite(init_value) || init_value <= 0.0 ||
            (persisted_converged != 0 && persisted_converged != 1) ||
            reason_size < 0 || reason_size > 1 << 20) {
            fail_load();
        }
        termination_reason_.resize(static_cast<std::size_t>(reason_size));
        ifs.read(termination_reason_.data(), reason_size);
        if (!ifs) { fail_load(); }
        init_mode = persisted_init_mode ? "mlip" : "data";
        converged_ = persisted_converged != 0;
    }
    int nsel;
    ifs.read(reinterpret_cast<char*>(&nsel), sizeof(nsel));
    if (!ifs || n_coeff_ <= 0 || nsel < 0) { fail_load(); }

    // ── Active row matrix ────────────────────────────────────────────
    // Compact active matrix and explicit row references.
    int k, total_rows;
    ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
    ifs.read(reinterpret_cast<char*>(&total_rows), sizeof(total_rows));
    if (!ifs || k < 0 || total_rows < 0) {
        fail_load();
    }
    selected_row_refs_.resize(k);

    {
        selected_row_indices_.resize(k);
        std::iota(selected_row_indices_.begin(), selected_row_indices_.end(), 0);
        for (int i = 0; i < k; ++i) {
            ifs.read(reinterpret_cast<char*>(&selected_row_refs_[i].cfg_index), sizeof(int));
            ifs.read(reinterpret_cast<char*>(&selected_row_refs_[i].local_row), sizeof(int));
            ifs.read(reinterpret_cast<char*>(&selected_row_refs_[i].global_row), sizeof(int));
            {
                int kind = static_cast<int>(BRowKind::Unknown);
                ifs.read(reinterpret_cast<char*>(&kind), sizeof(int));
                ifs.read(reinterpret_cast<char*>(&selected_row_refs_[i].atom_index), sizeof(int));
                ifs.read(reinterpret_cast<char*>(&selected_row_refs_[i].component), sizeof(int));
                if (kind < static_cast<int>(BRowKind::Energy) ||
                    kind > static_cast<int>(BRowKind::Unknown)) {
                    fail_load();
                }
                selected_row_refs_[i].kind = static_cast<BRowKind>(kind);
            }
            const int global = selected_row_refs_[i].global_row;
            const bool valid_virtual = global == -1;
            if (!ifs || (!valid_virtual &&
                         (global < 0 || global >= total_rows))) {
                fail_load();
            }
        }
        B_full_.resize(k, n_coeff_);
        for (int i = 0; i < k; ++i) {
            for (int j = 0; j < n_coeff_; ++j)
                ifs.read(reinterpret_cast<char*>(&B_full_(i, j)), sizeof(double));
        }
    }
    if (!ifs) { fail_load(); }

    // ── Training configs (EXTXYZ block) ──────────────────────────────
    {
        int cfg_bytes;
        ifs.read(reinterpret_cast<char*>(&cfg_bytes), sizeof(cfg_bytes));
        if (!ifs || cfg_bytes < 0) { fail_load(); }
        if (cfg_bytes > 0) {
            std::string cfg_data(cfg_bytes, '\0');
            ifs.read(cfg_data.data(), cfg_bytes);
            if (!ifs) { fail_load(); }
            std::istringstream iss(cfg_data);
            GPIO iobj;
            if (!index_file.empty()) iobj.readmap(index_file);
            if (!iobj.read(cfgs_, iss, "extxyz")) fail_load();
        }
    }

    // ── Selected configs (EXTXYZ block) ──────────────────────────────
    {
        int sel_bytes;
        ifs.read(reinterpret_cast<char*>(&sel_bytes), sizeof(sel_bytes));
        if (!ifs || sel_bytes < 0) { fail_load(); }
        if (sel_bytes > 0) {
            std::string sel_data(sel_bytes, '\0');
            ifs.read(sel_data.data(), sel_bytes);
            if (!ifs) { fail_load(); }
            std::istringstream iss(sel_data);
            GPIO iobj;
            if (!index_file.empty()) iobj.readmap(index_file);
            if (!iobj.read(selected_cfgs_, iss, "extxyz")) fail_load();
        }
        if (static_cast<int>(selected_cfgs_.size()) != nsel ||
            nsel > static_cast<int>(cfgs_.size())) {
            fail_load();
        }
    }

    // ── Builder metadata ───────────────────────────────────────────────
    int tag = 0;
    ifs.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    if (!ifs || tag != 0x4255494C) { fail_load(); }
    auto rd_str = [&]() -> std::string {
        int n = 0;
        ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!ifs || n < 0 || n > 1 << 20) { fail_load(); return {}; }
        std::string s(n, '\0');
        ifs.read(s.data(), n);
        return ifs ? s : std::string{};
    };
    auto rd_settings = [&]() -> GPSettings {
        int n = 0;
        ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
        if (!ifs || n < 0 || n > 100000) {
            ifs.setstate(std::ios::failbit);
            return {};
        }
        GPSettings settings;
        for (int i = 0; i < n; ++i) {
            std::string key = rd_str();
            std::string val = rd_str();
            if (!ifs || key.empty()) {
                ifs.setstate(std::ios::failbit);
                return {};
            }
            settings.settings.emplace(std::move(key), std::move(val));
        }
        return settings;
    };
    pending_builder_.dtype = rd_str();
    pending_builder_.mode  = rd_str();
    pending_builder_.valid = !pending_builder_.dtype.empty();
    if (!pending_builder_.valid) { fail_load(); }
    if (pending_builder_.dtype == "pot_efs" ||
        pending_builder_.dtype == "mtp_energy") {
        ifs.read(reinterpret_cast<char*>(&pending_builder_.w_energy), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&pending_builder_.w_site),   sizeof(double));
        ifs.read(reinterpret_cast<char*>(&pending_builder_.w_force),  sizeof(double));
        ifs.read(reinterpret_cast<char*>(&pending_builder_.w_stress), sizeof(double));
        char flags = 0; ifs.read(&flags, 1);
        pending_builder_.linear_only = (flags & 1) != 0;
    } else {
        pending_builder_.settings = rd_settings();
        if (!ifs) { fail_load(); }
    }

    // ── cfg_row_start_ ────────────────────────────────────────────────
    int ncfg = 0;
    ifs.read(reinterpret_cast<char*>(&ncfg), sizeof(ncfg));
    if (!ifs || ncfg != static_cast<int>(cfgs_.size()) + 1) {
        fail_load();
    }
    cfg_row_start_.resize(ncfg);
    for (int i = 0; i < ncfg; ++i)
        ifs.read(reinterpret_cast<char*>(&cfg_row_start_[i]), sizeof(int));
    if (!ifs || cfg_row_start_.empty() || cfg_row_start_.front() != 0 ||
        cfg_row_start_.back() != total_rows) {
        fail_load();
    }
    for (int i = 1; i < ncfg; ++i) {
        if (cfg_row_start_[i] < cfg_row_start_[i - 1]) {
            fail_load();
        }
    }


    online_processed_cfgs_ = cfgs_.size();
    {
        int processed = 0;
        char has_online = 0;
        ifs.read(reinterpret_cast<char*>(&processed), sizeof(processed));
        ifs.read(&has_online, 1);
        if (!ifs || processed < 0 ||
            processed > static_cast<int>(cfgs_.size()) ||
            (has_online != 0 && has_online != 1)) {
            fail_load();
        }
        online_processed_cfgs_ = static_cast<size_t>(processed);
        if (has_online && !online_state_.load(ifs)) {
            fail_load();
        }
        if (has_online &&
            (online_state_.coeff_count() != n_coeff_ ||
             online_state_.active_size() != k)) {
            fail_load();
        }
    }

    is_new_.assign(cfgs_.size(), false);

    // Rebuild selected_config_indices_ from the persisted row ranges.
    selected_config_indices_.clear();
    std::set<int> seen_configs;
    for (auto &ref : selected_row_refs_) {
        const int r = ref.global_row;
        for (int i = 0; i < static_cast<int>(cfg_row_start_.size()) - 1; ++i) {
            if (r >= cfg_row_start_[i] && r < cfg_row_start_[i + 1]) {
                ref.cfg_index = i;
                ref.local_row = r - cfg_row_start_[i];
                if (!seen_configs.count(i)) {
                    seen_configs.insert(i);
                    selected_config_indices_.push_back(i);
                }
                break;
            }
        }
    }
    if (static_cast<int>(selected_config_indices_.size()) != nsel) {
        fail_load();
    }
    // Full/stream selection can initialize online grading from its compact rows.
    gpm_legacy_unverified_ = true;
    if (!online_state_.valid() && select_mode == "online")
        online_state_.initialize_from_rows(B_full_, selected_row_refs_);
    has_full_B_ = false;
    using_stream_state_ = true;
    log_det_ = online_state_.valid()
             ? online_state_.log_determinant()
             : factorize_active_rows(B_full_).log_volume;
}

// ── Builder reconstruction (from stored params + user-provided potential) ──

void ActiveSelector::init_builder(Potentials::AnyPotential &pot)
{
    invalidate_grading();
    if (!pending_builder_.valid)
        GPERROR("Load a MaxVol state before reconstructing its builder");

    const auto &pb = pending_builder_;
    std::unique_ptr<BMatrixBuilder> rebuilt;
    if (pb.dtype == "mtm" || pb.dtype == "mtp" || pb.dtype == "mtp_scalar") {
        auto *mtp = dynamic_cast<Potentials::PotMTP *>(&pot);
        if (!mtp)
            GPERROR("MaxVol mtm state requires a PotMTP-derived model; use a pot_efs state for composite potentials");
        auto desc = std::make_unique<DescriptorMTM>(*mtp);
        auto db = std::make_unique<DescriptorBuilder>(std::move(desc));
        db->set_config_metadata(pb.dtype, pb.mode, pb.settings);
        rebuilt = std::move(db);
    } else if (pb.dtype == "pot_efs" || pb.dtype == "mtp_energy") {
        auto builder = std::make_unique<PotEFSBuilder>(pot);
        builder->w_energy = pb.w_energy;
        builder->w_site   = pb.w_site;
        builder->w_force  = pb.w_force;
        builder->w_stress = pb.w_stress;
        builder->linear_only = pb.linear_only;
        rebuilt = std::move(builder);
    } else {
        GPSettings settings = pb.settings;
        rebuilt = make_builder(pb.dtype, "", "", settings, pb.mode,
                               pb.w_energy, pb.w_site,
                               pb.w_force, pb.w_stress, true);
        if (!rebuilt)
            GPERROR("Cannot reconstruct MaxVol builder: " + pb.dtype);
    }
    rebuilt->row_scaling = pb.row_scaling;
    if (rebuilt->coeff_count() <= 0 || rebuilt->coeff_count() != n_coeff_)
        GPERROR("Model coefficient count does not match the MaxVol state; rebuild the state with matching potential settings");
    if (own_builder_) delete builder_;
    builder_ = rebuilt.release();
    own_builder_ = true;
}

// ── State file inspection / extraction ──────────────────────────────────────

void ActiveSelector::dump_state(const std::string &filename, std::ostream &out)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) GPERROR("Cannot open MaxVol state: " + filename);
    ifs.exceptions(std::ios::failbit | std::ios::badbit);

    int magic, version;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x41435456 ||
        version != 10) {
        GPERROR("Invalid MaxVol state header (expected v10): " + filename);
    }

    out << "version:     " << version << "\n";

    int nc; double thr, w, ps; int nsel;
    ifs.read(reinterpret_cast<char*>(&nc), sizeof(nc));
    ifs.read(reinterpret_cast<char*>(&thr), sizeof(thr));
    ifs.read(reinterpret_cast<char*>(&w), sizeof(w));
    ifs.read(reinterpret_cast<char*>(&ps), sizeof(ps));
    int scaling = static_cast<int>(RowScalingMode::Uniform);
    ifs.read(reinterpret_cast<char*>(&scaling), sizeof(scaling));
    int persisted_init_mode = 0;
    double persisted_init_value = 0.0;
    char persisted_converged = 0;
    double persisted_max_grade = std::numeric_limits<double>::infinity();
    std::string persisted_reason;
    {
        int reason_size = 0;
        ifs.read(reinterpret_cast<char*>(&persisted_init_mode), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&persisted_init_value), sizeof(double));
        ifs.read(&persisted_converged, 1);
        ifs.read(reinterpret_cast<char*>(&persisted_max_grade), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&reason_size), sizeof(int));
        if (reason_size < 0 || reason_size > 1 << 20)
            GPERROR("Invalid MaxVol state string size: " + filename);
        persisted_reason.resize(static_cast<std::size_t>(reason_size));
        ifs.read(persisted_reason.data(), reason_size);
    }
    ifs.read(reinterpret_cast<char*>(&nsel), sizeof(nsel));
    out << "n_coeffs:    " << nc << "\n";
    out << "threshold:   " << thr << "\n";
    out << "weight:      " << w << "\n";
    out << "pow_scale:   " << ps << "\n";
    out << "row_scaling: "
        << to_string(static_cast<RowScalingMode>(scaling)) << "\n";
    {
        out << "init_mode:   " << (persisted_init_mode ? "mlip" : "data") << "\n";
        out << "init_value:  " << persisted_init_value << "\n";
        out << "converged:   " << (persisted_converged ? "yes" : "no") << "\n";
        out << "max_grade:   " << persisted_max_grade << "\n";
        out << "termination: " << persisted_reason << "\n";
    }
    out << "n_selected:  " << nsel << "\n";

    int k, total_rows;
    ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
    ifs.read(reinterpret_cast<char*>(&total_rows), sizeof(total_rows));
    if (nc <= 0 || nsel < 0 || k < 0 || total_rows < 0)
        GPERROR("Invalid MaxVol state dimensions: " + filename);
    out << "sel_rows:    " << k << "\n";
    out << "total_rows:  " << total_rows << "\n";

    std::vector<int> kind_counts(static_cast<int>(BRowKind::Unknown) + 1, 0);
    {
        for (int i = 0; i < k; ++i) {
            int values[6] = {};
            ifs.read(reinterpret_cast<char *>(values), sizeof(values));
            if (values[3] >= 0 && values[3] < static_cast<int>(kind_counts.size()))
                ++kind_counts[values[3]];
        }
    }
    // Skip C matrix
    ifs.seekg(std::streamoff(k) * nc * sizeof(double), std::ios::cur);

    auto read_extxyz_block = [&](const char *label, int &bytes_out,
                                 std::vector<Atoms> &cfgs_out) {
        bytes_out = 0;
        ifs.read(reinterpret_cast<char*>(&bytes_out), sizeof(bytes_out));
        if (!ifs || bytes_out < 0) {
            GPERROR("Invalid MaxVol configuration block: " + filename);
        }
        std::string data(bytes_out, '\0');
        if (bytes_out > 0)
            ifs.read(data.data(), bytes_out);
        if (!ifs) {
            GPERROR("Truncated MaxVol configuration block: " + filename);
        }
        if (!data.empty()) {
            std::istringstream iss(data);
            GPIO iobj;
            if (!iobj.read(cfgs_out, iss, "extxyz"))
                GPERROR("Invalid MaxVol configurations: " + filename);
        }
        return true;
    };

    // Training configs
    int cfg_bytes;
    std::vector<Atoms> train_cfgs;
    if (!read_extxyz_block("train_cfgs", cfg_bytes, train_cfgs)) return;

    // Selected configs
    int sel_bytes;
    std::vector<Atoms> sel_cfgs;
    if (!read_extxyz_block("sel_cfgs", sel_bytes, sel_cfgs)) return;

    int tag = 0;
    ifs.read(reinterpret_cast<char*>(&tag), sizeof(tag));
    if (tag != 0x4255494C) GPERROR("Missing MaxVol builder metadata: " + filename);
    out << "builder:     " << ((ifs && tag == 0x4255494C) ? "embedded\n" : "missing\n");
    if (ifs && tag == 0x4255494C) {
        auto rd_str = [&]() -> std::string {
            int n = 0;
            ifs.read(reinterpret_cast<char*>(&n), sizeof(n));
            if (n < 0 || n > 1 << 20)
                GPERROR("Invalid MaxVol metadata string size: " + filename);
            std::string s(n, '\0');
            ifs.read(s.data(), n);
            return ifs ? s : std::string{};
        };
        auto skip_settings = [&]() {
            int ns = 0;
            ifs.read(reinterpret_cast<char*>(&ns), sizeof(ns));
            if (ns < 0 || ns > 100000)
                GPERROR("Invalid MaxVol settings count: " + filename);
            for (int i = 0; i < ns; ++i) {
                (void)rd_str();
                (void)rd_str();
                if (!ifs) return false;
            }
            return true;
        };

        std::string dtype = rd_str();
        std::string mode = rd_str();
        out << "builder_type:" << dtype << "\n";
        out << "builder_mode:" << mode << "\n";
        if (dtype == "pot_efs" || dtype == "mtp_energy") {
            double we = 0.0, ws = 0.0, wf = 0.0, wv = 0.0;
            char ac = 0;
            ifs.read(reinterpret_cast<char*>(&we), sizeof(we));
            ifs.read(reinterpret_cast<char*>(&ws), sizeof(ws));
            ifs.read(reinterpret_cast<char*>(&wf), sizeof(wf));
            ifs.read(reinterpret_cast<char*>(&wv), sizeof(wv));
            ifs.read(&ac, 1);
            out << "efs_weights: w_energy=" << we
                << " w_site=" << ws
                << " w_force=" << wf
                << " w_stress=" << wv << "\n";
            out << "linear_only:" << (ac ? " true\n" : " false\n");
        } else {
            std::streampos before = ifs.tellg();
            if (skip_settings()) {
                std::streampos after = ifs.tellg();
                ifs.clear();
                ifs.seekg(before);
                int ns = 0;
                ifs.read(reinterpret_cast<char*>(&ns), sizeof(ns));
                out << "settings:    " << ns << " entries\n";
                for (int i = 0; i < ns; ++i) {
                    std::string key = rd_str();
                    std::string val = rd_str();
                    out << "  " << key << " = " << val << "\n";
                }
                ifs.clear();
                ifs.seekg(after);
            }
        }
    }

    int ncfg_rows = 0;
    ifs.read(reinterpret_cast<char*>(&ncfg_rows), sizeof(ncfg_rows));
    if (ncfg_rows != static_cast<int>(train_cfgs.size()) + 1)
        GPERROR("Invalid MaxVol configuration row count: " + filename);
    std::vector<int> row_start;
    if (ifs && ncfg_rows > 0 && ncfg_rows < 100000000) {
        row_start.resize(ncfg_rows);
        for (int i = 0; i < ncfg_rows; ++i)
            ifs.read(reinterpret_cast<char*>(&row_start[i]), sizeof(int));
    }

    out << "train_cfgs:  " << train_cfgs.size()
        << " (" << cfg_bytes << " EXTXYZ bytes)\n";
    out << "sel_cfgs:    " << sel_cfgs.size()
        << " (" << sel_bytes << " EXTXYZ bytes)\n";
    out << "local_rows:  " << total_rows << "\n";
    if (row_start.size() == train_cfgs.size() + 1 && !row_start.empty()) {
        int min_rows = std::numeric_limits<int>::max();
        int max_rows = 0;
        for (size_t i = 1; i < row_start.size(); ++i) {
            int rows = row_start[i] - row_start[i - 1];
            min_rows = std::min(min_rows, rows);
            max_rows = std::max(max_rows, rows);
        }
        double avg_rows = train_cfgs.empty()
            ? 0.0
            : static_cast<double>(row_start.back()) / train_cfgs.size();
        out << "rows/cfg:    avg=" << avg_rows
            << " min=" << (train_cfgs.empty() ? 0 : min_rows)
            << " max=" << max_rows << "\n";
    }
    {
        int processed = 0;
        char has_online = 0;
        ifs.read(reinterpret_cast<char *>(&processed), sizeof(processed));
        ifs.read(&has_online, 1);
        if (processed < 0 || processed > static_cast<int>(train_cfgs.size()) ||
            (has_online != 0 && has_online != 1))
            GPERROR("Invalid MaxVol online metadata: " + filename);
        if (has_online) {
            OnlineMaxVolState online;
            if (!online.load(ifs)) GPERROR("Invalid MaxVol online state: " + filename);
        }
        out << "online:      " << (has_online ? "yes" : "no") << "\n";
        out << "processed:   " << processed << " configs\n";
        out << "row_kinds:  ";
        for (int kind = 0; kind < static_cast<int>(kind_counts.size()); ++kind) {
            if (kind_counts[kind] == 0) continue;
            out << " " << to_string(static_cast<BRowKind>(kind))
                << "=" << kind_counts[kind];
        }
        out << "\n";
    }
}

void ActiveSelector::export_configs(const std::string &filename,
                                     const std::string &output_file,
                                     bool selected_only,
                                     const std::string &index_file)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) GPERROR("Cannot open MaxVol state: " + filename);
    ifs.exceptions(std::ios::failbit | std::ios::badbit);

    int magic, version;
    ifs.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    ifs.read(reinterpret_cast<char*>(&version), sizeof(version));
    if (magic != 0x41435456 ||
        version != 10)
        GPERROR("Invalid MaxVol state header (expected v10): " + filename);

    // Skip header + C matrix
    int nc, nsel; double dummy;
    ifs.read(reinterpret_cast<char*>(&nc), sizeof(nc));
    ifs.read(reinterpret_cast<char*>(&dummy), sizeof(dummy)); // threshold
    ifs.read(reinterpret_cast<char*>(&dummy), sizeof(dummy)); // weight
    ifs.read(reinterpret_cast<char*>(&dummy), sizeof(dummy)); // pow_scale
    {
        int scaling = 0;
        ifs.read(reinterpret_cast<char*>(&scaling), sizeof(scaling));
    }
    {
        int init_mode_value = 0, reason_size = 0;
        char converged_value = 0;
        ifs.read(reinterpret_cast<char*>(&init_mode_value), sizeof(int));
        ifs.read(reinterpret_cast<char*>(&dummy), sizeof(double));
        ifs.read(&converged_value, 1);
        ifs.read(reinterpret_cast<char*>(&dummy), sizeof(double));
        ifs.read(reinterpret_cast<char*>(&reason_size), sizeof(int));
        if (reason_size < 0 || reason_size > 1 << 20)
            GPERROR("Invalid MaxVol state string size: " + filename);
        ifs.seekg(reason_size, std::ios::cur);
    }
    ifs.read(reinterpret_cast<char*>(&nsel), sizeof(nsel));
    int k, total_rows;
    ifs.read(reinterpret_cast<char*>(&k), sizeof(k));
    ifs.read(reinterpret_cast<char*>(&total_rows), sizeof(total_rows));
    if (nc <= 0 || nsel < 0 || k < 0 || total_rows < 0)
        GPERROR("Invalid MaxVol state dimensions: " + filename);
    constexpr int ref_ints = 6;
    ifs.seekg(std::streamoff(k) * ref_ints * sizeof(int), std::ios::cur);
    ifs.seekg(std::streamoff(k) * nc * sizeof(double), std::ios::cur);

    int cfg_bytes = 0;
    ifs.read(reinterpret_cast<char*>(&cfg_bytes), sizeof(cfg_bytes));
    if (cfg_bytes < 0) GPERROR("Invalid MaxVol configuration block size: " + filename);
    if (selected_only) {
        ifs.seekg(cfg_bytes, std::ios::cur);
        ifs.read(reinterpret_cast<char*>(&cfg_bytes), sizeof(cfg_bytes));
        if (cfg_bytes < 0) GPERROR("Invalid MaxVol configuration block size: " + filename);
    }
    // Raw export preserves the species names and original EXTXYZ bytes.
    std::string data(cfg_bytes, '\0');
    ifs.read(data.data(), cfg_bytes);
    std::ofstream ofs(output_file, std::ios::binary);
    if (!ofs) GPERROR("Cannot open configuration output: " + output_file);
    ofs.write(data.data(), cfg_bytes);
    ofs.close();
    if (!ofs) GPERROR("Failed to export configurations: " + output_file);
}
