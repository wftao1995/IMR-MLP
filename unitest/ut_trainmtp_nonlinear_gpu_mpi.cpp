// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "fixtures/mtp_nonlinear_gpu.h"
#if defined(IMRMLP_MPI) && defined(IMRMLP_GPU)
int main(int argc, char **argv)
{
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed = 20261102;
    session.configData().runOrder = Catch::TestRunOrder::Declared;
    const int result = session.run(argc, argv);
    MPI_Finalize();
    return result;
}
namespace F = MTPNonlinearGpuFixture;

namespace
{
void all_require(bool condition)
{
    int local = condition, global = 0;
    MPI_Allreduce(&local, &global, 1, MPI_INT, MPI_MIN, mpi.comm);
    REQUIRE(global == 1);
}

struct Workspace
{
    std::unique_ptr<F::Workspace> owner;
    std::string path;

    Workspace()
    {
        Training::collective_try(
            [&]
            {
                if (mpi.rank == 0)
                {
                    owner = std::make_unique<F::Workspace>();
                    path = owner->directory.string();
                }
            });
        int size = path.size();
        MPI_Bcast(&size, 1, MPI_INT, 0, mpi.comm);
        path.resize(size);
        MPI_Bcast(path.data(), size, MPI_CHAR, 0, mpi.comm);
    }

    ~Workspace()
    {
        MPI_Barrier(mpi.comm);
    }
};
}

TEST_CASE("MPI trainmtp nonlinear GPU supports uneven and empty shards and collective retry",
          "[gpu][training][trainmtp-nonlinear][mpi]")
{
    if (!F::available())
        SKIP("No GPU");
    const int count = GENERATE(1, 5);
    const std::string method = GENERATE("lbfgs", "cg", "sgd");
    F::Capture capture;
    Workspace files;
    F::Fixture f(count);
    std::vector<Atoms> local;
    std::vector<Neighborhoods> neighbors;
    for (int i = 0; i < count; ++i)
        if (i % mpi.size == mpi.rank)
        {
            local.push_back(f.data[i]);
            neighbors.push_back(f.neighbors[i]);
        }
    auto s = F::settings();
    s.Modify("save_to", files.path + "/checkpoint.mtp");
    s.Modify("bfgs_rescale", "true");
    s.Modify("linear_fit_bfgs", "true");
    Trainer_MTP gpu(s);
    gpu.init(local, neighbors, &f.model);
    gpu.init_valid(local, neighbors);
    alglib::real_1d_array gg;
    gg.setlength(gpu.get_relaxed_coeffs_count());
    double gl = 0.;
    gpu.AccumulateGard(gl, gg);
    // Independent serial CPU objective on each rank, including full data.
    auto cpu_model = F::copy(f.model);
    s.Modify("device", "cpu");
    Trainer_MTP cpu(s);
    const auto world = mpi.comm;
    mpi.InitComm(MPI_COMM_SELF);
    cpu.init(f.data, f.neighbors, cpu_model.get());
    alglib::real_1d_array cg;
    cg.setlength(cpu.get_relaxed_coeffs_count());
    double cl = 0.;
    cpu.AccumulateGard(cl, cg);
    mpi.InitComm(world);
    all_require(std::abs(gl - cl) < 2e-9);
    bool close = true;
    for (int i = 0; i < gg.length(); ++i)
        close &= std::abs(gg[i] - cg[i]) < 2e-8 + 1e-8 * std::abs(cg[i]);
    all_require(close);
    const auto before = F::coefficients(f.model);
    const auto ridge = f.model.reg_L2_vector;
    Training::collective_try(
        [&]
        {
            if (mpi.rank == 0)
                std::filesystem::create_directory(files.path + "/checkpoint.mtp");
        });
    bool failed = false;
    try
    {
        F::optimize(gpu, method);
    }
    catch (const std::exception &)
    {
        failed = true;
    }
    all_require(failed);
    all_require((F::coefficients(f.model) - before).norm() == 0. && f.model.reg_L2_vector == ridge);
    Training::collective_try(
        [&]
        {
            if (mpi.rank == 0)
                std::filesystem::remove(files.path + "/checkpoint.mtp");
        });
    REQUIRE_NOTHROW(F::optimize(gpu, method));
    all_require(gpu.GetGpuNonlinearReport().gradient_evaluations > 0);
    auto changed = F::settings();
    changed.Modify("gpu_id", mpi.rank == 1 ? "99999" : "0");
    // Retain the same mathematical settings across ranks.
    changed.Modify("bfgs_rescale", "true");
    changed.Modify("linear_fit_bfgs", "true");
    changed.Modify("save_to", files.path + "/checkpoint.mtp");
    gpu.ApplySettings(changed);
    failed = false;
    try
    {
        F::optimize(gpu, method);
    }
    catch (const std::exception &)
    {
        failed = true;
    }
    all_require(failed);
    changed.Modify("gpu_id", "0");
    gpu.ApplySettings(changed);
    REQUIRE_NOTHROW(F::optimize(gpu, method));
    changed.Modify("device", mpi.rank == 1 ? "cpu" : F::G::compiled_backend());
    gpu.ApplySettings(changed);
    failed = false;
    try
    {
        gpu.AccumulateGard(gl, gg);
    }
    catch (const std::exception &)
    {
        failed = true;
    }
    all_require(failed);
    changed.Modify("device", F::G::compiled_backend());
    gpu.ApplySettings(changed);
    REQUIRE_NOTHROW(gpu.AccumulateGard(gl, gg));
}

TEST_CASE("MPI trainmtp initializes one shared random ordinary model before GPU training",
          "[gpu][training][trainmtp-nonlinear][mpi][initialization]")
{
    if (!F::available())
        SKIP("No GPU");
    F::Capture capture;
    Workspace files;
    F::Fixture f(1);
    f.model.set_empty_flag();
    std::vector<Atoms> local;
    std::vector<Neighborhoods> neighbors;
    if (mpi.rank == 0)
    {
        local = f.data;
        neighbors = f.neighbors;
    }
    auto s = F::settings();
    s.Modify("random_init", "true");
    s.Modify("force_randomlinear", "true");
    s.Modify("force_randomradial", "true");
    s.Modify("apply_disp", "true");
    s.Modify("save_to", files.path + "/random.mtp");
    Trainer_MTP gpu(s);
    REQUIRE_NOTHROW(gpu.init(local, neighbors, &f.model));
    auto values = F::coefficients(f.model), root = values;
    MPI_Bcast(root.data(), root.size(), MPI_DOUBLE, 0, mpi.comm);
    all_require((values - root).norm() == 0. && f.model.inited());
    REQUIRE_NOTHROW(gpu.TrainBFGS());
}
#endif
