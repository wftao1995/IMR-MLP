#include <catch2/catch_all.hpp>
#include "maxvol/active_selector.h"
#include "maxvol/mpi_batch.h"
#include <cmath>

#ifdef IMRMLP_MPI
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed = 123;
    session.configData().runOrder = Catch::TestRunOrder::Declared;
    int result = session.run(argc, argv);
    MPI_Finalize();
    return result;
}

namespace
{
void check(bool value, MPI_Comm comm = MPI_COMM_WORLD)
{
    int local = value, all = 0;
    MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, comm);
    REQUIRE(all == 1);
}

struct Rows : BMatrixBuilder
{
    int coeff_count() const override
    {
        return 3;
    }

    double cutoff() const override
    {
        return 0.;
    }

    int row_count(const Atoms &cfg) const override
    {
        return cfg.pos_.size();
    }

    std::unique_ptr<BMatrixBuilder> clone() const override
    {
        return std::make_unique<Rows>();
    }

    int build_rows(Atoms &cfg, RowMatrix &B, double w, double) override
    {
        const int first = B.rows();
        B.conservativeResize(first + cfg.pos_.size(), 3);
        for (int i = 0; i < int(cfg.pos_.size()); ++i)
            B.row(first + i) = w * cfg.pos_[i];
        return cfg.pos_.size();
    }
};

std::vector<Atoms> pool(int count, bool deficient)
{
    std::vector<Atoms> result(count);
    for (int i = 0; i < count; ++i)
    {
        result[i].cell_ = 10. * Eigen::Matrix3d::Identity();
        result[i].pos_.push_back({double(i + 1), std::sin(i + .2) * (i + 2),
                                  deficient ? 0. : std::cos(2. * i) * (i + 1)});
        result[i].chemmap_.push_back(0);
    }
    return result;
}

void setup(ActiveSelector &s, int cap)
{
    s.stream_mode = "batch";
    s.stream_cache_mode = "memory";
    s.stream_memory_limit_mb = 128;
    s.stream_fast_shortlist = 2;
    s.max_swaps = cap;
    s.pow_scale = 0.;
}
}

TEST_CASE("MPI batch canonical factors survive ridge updates and rank restoration", "[mpi][batch]")
{
    // Reverse rank order to exercise a caller communicator, not mpi.comm.
    MPI_Comm comm;
    MPI_Comm_split(MPI_COMM_WORLD, 0, mpi.size - mpi.rank, &comm);
    int rank = 0;
    MPI_Comm_rank(comm, &rank);
    OnlineMaxVolState state;
    state.enable_batch_updates();
    OnlineRowMatrix C = OnlineRowMatrix::Zero(3, 3);
    C(0, 0) = 10.;
    C(1, 1) = 1.;
    if (rank == 0)
        REQUIRE(state.initialize_from_rows(C));
    state.broadcast_runtime(comm);
    for (int step = 0; step < 130; ++step)
    {
        if (rank == 0)
        {
            OnlineRowMatrix one = state.active_rows().middleRows(step % 2, 1);
            one *= 1.002 / state.score_rows(one)[0];
            REQUIRE(state.maximize_with_candidates(one, {}, 1.001, 1) == 1);
        }
        state.broadcast_runtime(comm);
        CAPTURE(step);
        OnlineRowMatrix probes = OnlineRowMatrix::Identity(3, 3);
        auto actual = state.score_rows(probes);
        Eigen::VectorXd canonical = actual;
        MPI_Bcast(canonical.data(), 3, MPI_DOUBLE, 0, comm);
        check((actual - canonical).norm() == 0., comm);
        check(state.last_swaps().size() == 1, comm);
        OnlineMaxVolState rebuilt;
        check(rebuilt.initialize_from_rows(state.active_rows()), comm);
        auto reference = rebuilt.score_rows(probes);
        check(((actual - reference).array().abs() / reference.array().abs().max(1.)).maxCoeff() <
                  1e-4,
              comm);
    }
    INFO("updates=" << state.factor_stats().ridge_updates
                    << " rebuilds=" << state.factor_stats().rebuilds);
    // Some exchanges duplicate an existing direction and change lambda by >1%,
    // legitimately forcing an SVD. Every exchange is counted exactly once.
    check(state.factor_stats().ridge_updates > 0, comm);
    check(state.factor_stats().rebuilds > 1, comm);
    check(state.factor_stats().ridge_updates + state.factor_stats().rebuilds == 131, comm);
    if (rank == 0)
    {
        OnlineRowMatrix one(1, 3);
        one << 0., 0., 2.;
        REQUIRE(state.maximize_with_candidates(one, {}, 1.001, 1) == 1);
    }
    state.broadcast_runtime(comm);
    check(std::abs(state.active_rows().determinant()) > 0., comm);
    MPI_Comm_free(&comm);
}

TEST_CASE("MPI batch agrees with serial including empty ranks and capped final checks",
          "[mpi][batch]")
{
    for (bool deficient : {false, true})
        for (int count : {2, 19})
            for (int cap : {1, 200})
            {
                auto data = pool(count, deficient);
                ActiveSelector serial(new Rows()), parallel(new Rows());
                setup(serial, cap);
                setup(parallel, cap);
                serial.select_streaming(data);
                parallel.add_candidates_unbuilt(data);
                parallel.select_streaming_mpi(MPI_COMM_WORLD);
                check(parallel.selected_indices() == serial.selected_indices());
                check(parallel.converged() == serial.converged());
                check(parallel.termination_reason() == serial.termination_reason());
                check(std::abs(parallel.max_remaining_grade() - serial.max_remaining_grade()) <
                      1e-9);
                check(parallel.stream_stats().swaps == serial.stream_stats().swaps);
                std::vector<double> grades;
                parallel.grade_all(data, grades);
                check(std::abs(*std::max_element(grades.begin(), grades.end()) -
                               parallel.max_remaining_grade()) < 1e-9);
            }
}

TEST_CASE("MPI batch checks a purely virtual active set", "[mpi][batch]")
{
    auto data = pool(2, false);
    ActiveSelector selector(new Rows());
    setup(selector, 1);
    selector.init_mode = "mlip";
    selector.init_value = 1000.;
    selector.add_candidates_unbuilt(data);
    selector.select_streaming_mpi(MPI_COMM_WORLD);
    check(selector.converged());
    check(selector.virtual_row_count() == 3);
    check(selector.get_selected().empty());
    std::vector<double> grades;
    selector.grade_all(data, grades);
    check(std::abs(*std::max_element(grades.begin(), grades.end()) -
                   selector.max_remaining_grade()) < 1e-10);
}

TEST_CASE("MPI batch failures and inconsistent controls reach every rank", "[mpi][batch]")
{
    bool caught = false;
    try
    {
        MaxVolMPI::local(MPI_COMM_WORLD,
                         [&]
                         {
                             if (mpi.rank == mpi.size - 1)
                                 throw std::runtime_error("injected");
                         });
    }
    catch (const std::runtime_error &)
    {
        caught = true;
    }
    check(caught);
    if (mpi.size > 1)
    {
        ActiveSelector selector(new Rows());
        setup(selector, 1);
        if (mpi.rank)
            selector.stream_mode = "exact";
        auto data = pool(3, false);
        selector.add_candidates_unbuilt(data);
        caught = false;
        try
        {
            selector.select_streaming_mpi(MPI_COMM_WORLD);
        }
        catch (const std::exception &)
        {
            caught = true;
        }
        check(caught);
    }
}
#else
TEST_CASE("MPI batch contracts require an MPI build", "[mpi][batch]")
{
    SUCCEED();
}
#endif
