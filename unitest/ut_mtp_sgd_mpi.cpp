#include <catch2/catch_all.hpp>
#ifdef IMRMLP_MPI
#include "train/train_mtp.h"
#include "train/collective_error.h"
#include <filesystem>
#include <algorithm>
#include <unistd.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed = 20261021;
    session.configData().runOrder = Catch::TestRunOrder::Declared;
    const int result = session.run(argc, argv);
    MPI_Finalize(); return result;
}
namespace {
void all_require(bool value) {
    int local = value, all = 0;
    MPI_Allreduce(&local, &all, 1, MPI_INT, MPI_MIN, mpi.comm);
    REQUIRE(all == 1);
}
template<class F> void self_reference(F &&run) {
    Training::collective_try([&] {
        struct RestoreWorld {~RestoreWorld() {mpi.InitComm(MPI_COMM_WORLD);}} restore;
        mpi.InitComm(MPI_COMM_SELF); run();
    });
}
struct Data {
    Potentials::PotMTP potential{"unitest/test-data/portable-mtp/les.mtp"};
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> neighbors;
    Data(int count, bool distributed, bool skip_middle, bool force_only = false) {
        potential.train_init(); potential.set_inited();
        for (int i = 0; i < count; ++i) {
            Atoms atom; atom.cell_ = 12. * Eigen::Matrix3d::Identity();
            atom.pos_ = {{2.,2.,2.},{4.1+.03*i,2.2,2.},{2.3,4.2,2.4},{5.,4.8,3.1}};
            atom.chemmap_ = {0,1,0,1}; atom.id = i;
            potential.getEFS(atom); atom.energy_ += .03 * (i + 1);
            if (force_only) {
                atom.have_energy = false;
                atom.forces_[0][0] += .03 * (i + 1);
            }
            const int owner = skip_middle ? (i < count-1 ? 0 : mpi.size-1) : i % mpi.size;
            if (!distributed || owner == mpi.rank) {
                atoms.push_back(atom); neighbors.emplace_back(atom, potential.cutoff);
            }
        }
    }
};
Eigen::VectorXd coefficients(Potentials::PotMTP &potential) {
    Eigen::VectorXd out(potential.get_NL_coeffs_count()); potential.nl_get_coeffs(out.data()); return out;
}
GPSettings controls(const std::string &path) {
    GPSettings settings;
    settings.Modify("energy_weight", "1"); settings.Modify("force_weight", ".2");
    settings.Modify("stress_weight", ".1"); settings.Modify("penalty_weight", ".01");
    settings.Modify("weight_L2", ".02"); settings.Modify("bfgs_rescale", "false");
    settings.Modify("epoch_limit", "1"); settings.Modify("batch_size", "100000");
    settings.Modify("learning_rate", ".00001"); settings.Modify("momentum", ".3");
    settings.Modify("save_every", "100"); settings.Modify("save_to", path);
    return settings;
}
}

TEST_CASE("MTP SGD matches whole-data updates with uneven and empty MPI shards", "[mpi][mtp][sgd]") {
    const int partition = GENERATE(0,1,2);
    const bool force_only = GENERATE(false, true);
    const int count = partition == 0 ? 1 : partition == 1 ? mpi.size + 1 : 4;
    Data whole(count, false, partition == 2, force_only), local(count, true, partition == 2, force_only);
    const auto stem = (std::filesystem::temp_directory_path() /
        ("atoms-agent-mtp-sgd-" + std::to_string(getpid()))).string();
    Eigen::VectorXd expected;
    self_reference([&] {
        Trainer_MTP trainer(controls(stem + ".self"));
        trainer.init(whole.atoms, whole.neighbors, &whole.potential);
        trainer.TrainSGD(); expected = coefficients(whole.potential);
        std::filesystem::remove(stem + ".self"); std::filesystem::remove(stem + ".self0");
    });
    Trainer_MTP trainer(controls(stem + ".world"));
    trainer.init(local.atoms, local.neighbors, &local.potential);
    trainer.TrainSGD();
    const auto actual = coefficients(local.potential);
    all_require((actual - expected).norm() < 1e-12 * std::max(1., expected.norm()));
    auto root = actual.eval(); MPI_Bcast(root.data(), root.size(), MPI_DOUBLE, 0, mpi.comm);
    all_require((actual - root).norm() == 0.);
    // Empty batches must not replay the previous gradient or momentum.
    trainer.SGDUpdate(0);
    all_require((coefficients(local.potential) - actual).norm() == 0.);
    bool reloaded = true;
    Training::collective_try([&] {
        if (mpi.rank == 0) {
            Potentials::PotMTP saved(stem + ".world");
            reloaded = (coefficients(saved) - actual).norm() < 1e-12 * std::max(1., actual.norm());
            std::filesystem::remove(stem + ".world"); std::filesystem::remove(stem + ".world0");
        }
    });
    all_require(reloaded);
}

TEST_CASE("Rank-local invalid MTP SGD controls fail collectively before a coefficient update", "[mpi][mtp][sgd]") {
    Data local(1, true, false);
    const auto before = coefficients(local.potential);
    auto settings = controls("unused-invalid-mpi-sgd.mtp");
    if (mpi.rank == mpi.size-1) {
        settings.Modify("lr_scheduler", "step"); settings.Modify("lr_decay_steps", "0");
    }
    Trainer_MTP trainer(settings); trainer.init(local.atoms, local.neighbors, &local.potential);
    std::string error;
    try {trainer.TrainSGD();} catch (const std::exception &e) {error = e.what();}
    all_require(error.find("positive decay steps") != std::string::npos);
    all_require((coefficients(local.potential) - before).norm() == 0.);
    // The failed validation leaves MPI usable for a valid zero-epoch call.
    GPSettings valid; valid.Modify("lr_scheduler", "none"); valid.Modify("epoch_limit", "0");
    trainer.ApplySettings(valid); trainer.TrainSGD();
    all_require((coefficients(local.potential) - before).norm() == 0.);
}
#endif
