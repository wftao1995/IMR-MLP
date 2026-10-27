#include <catch2/catch_all.hpp>
#include "train/train_nonlinear_potential.h"
#include "potentials/potential_mtp_LES.h"
#include "utils/comm.h"
#include <filesystem>
#include <fstream>
#include <limits>
#include <unistd.h>

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed = 20261018;
    session.configData().runOrder = Catch::TestRunOrder::Declared;
    const int result = session.run(argc, argv);
    MPI_Finalize();
    return result;
}
#endif

namespace {
namespace fs = std::filesystem;

// A one-parameter nonlinear energy makes optimizer states and failure timing
// observable without injecting test hooks into the production trainer.
class FailurePotential : public Potentials::AnyPotential {
public:
    enum class Fault { none, energy, gradient, nan_energy, nan_gradient, penalty, alglib, validation };
    double coefficient = 3.;
    Fault fault = Fault::none;
    int fail_rank = 0;
    int evaluations = 0;
    int fail_after = 0;
    int saves = 0;
    bool fail_save = false;
    int validation_evaluations = 0;
    int getter_failure_at_validation = 0;
    mutable bool pending_getter_failure = false;
    FailurePotential() { set_inited(); }
    bool failing() const { return mpi.rank == fail_rank && evaluations > fail_after; }
    int get_NL_coeffs_count() const override { return 1; }
    void nl_get_coeffs(double *x) const override {
        if (pending_getter_failure) {
            pending_getter_failure = false;
            throw std::runtime_error("injected validation coefficient read failure");
        }
        x[0] = coefficient;
    }
    void nl_set_coeffs(const double *x) override { coefficient = x[0]; }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        ++evaluations;
        if (a.id == 42) {
            ++validation_evaluations;
            if (mpi.rank == fail_rank && fault == Fault::validation)
                throw std::runtime_error("injected validation EFS failure");
            if (mpi.rank == fail_rank && getter_failure_at_validation > 0
                && validation_evaluations == getter_failure_at_validation)
                pending_getter_failure = true;
        }
        if (failing() && fault == Fault::energy) throw std::runtime_error("injected energy failure");
        if (failing() && fault == Fault::alglib) {
            alglib::real_1d_array point; point.setlength(1); point[0] = 1.;
            alglib::minlbfgsstate invalid;
            // A real ALGLIB exception, raised inside one rank's objective.
            alglib::minlbfgscreate(1, 0, point, invalid);
        }
        a.energy_ = failing() && fault == Fault::nan_energy
            ? std::numeric_limits<double>::quiet_NaN() : coefficient*coefficient;
        a.have_energy = true;
    }
    void AccumulateEFSCombinationGard(Atoms &, AnyNeighborhoods &, double dE, double *g) override {
        if (failing() && fault == Fault::gradient) throw std::runtime_error("injected gradient failure");
        g[0] += failing() && fault == Fault::nan_gradient
            ? std::numeric_limits<double>::infinity() : dE*2.*coefficient;
    }
    void AddPenaltyGard(double, double &, double *) override {
        if (mpi.rank == fail_rank && fault == Fault::penalty)
            throw std::runtime_error("injected penalty failure");
    }
    void Save(const std::string &path) override {
        ++saves;
        if (fail_save) throw std::runtime_error("injected save failure");
        std::ofstream out(path);
        out << std::setprecision(17) << coefficient << '\n';
        out.close();
        if (!out) throw std::runtime_error("cannot save test potential");
    }
};

struct TrainingCase {
    FailurePotential potential;
    std::vector<Atoms> data;
    std::vector<Neighborhoods> neighbors;
    fs::path directory;
    TrainingCase() {
        static int sequence = 0;
        int owner_pid = static_cast<int>(getpid());
        MPI_Bcast(&owner_pid, 1, MPI_INT, 0, mpi.comm);
        directory = fs::path("build/todo-audit-20261018/batch4/training-isolated") /
            ("case-" + std::to_string(owner_pid) + "-" + std::to_string(++sequence));
        if (mpi.rank == 0) fs::create_directories(directory);
        MPI_Barrier(mpi.comm);
        Atoms a;
        a.cell_ = Eigen::Matrix3d::Identity()*10.;
        a.pos_ = {{1.,1.,1.}};
        a.chemmap_ = {0};
        a.energy_ = 1.; a.have_energy = true; a.fitness = 1.;
        data.push_back(a);
        neighbors.emplace_back(data[0], potential.cutoff);
    }
    ~TrainingCase() {
        MPI_Barrier(mpi.comm);
        if (mpi.rank == 0) fs::remove_all(directory);
    }
    GPSettings settings() const {
        GPSettings s;
        s.Modify("energy_weight", "1"); s.Modify("force_weight", "0");
        s.Modify("stress_weight", "0"); s.Modify("penalty_weight", "0");
        s.Modify("bfgs_m", "1"); s.Modify("bfgs_maxiteration", "1");
        s.Modify("bfgs_grad_tol", "1e-20"); s.Modify("bfgs_conv_tol", "0");
        s.Modify("bfgs_step_tol", "0"); s.Modify("save_every", "1");
        s.Modify("save_to", (directory/"fitted.mtp").string());
        return s;
    }
    void preserve_file() const {
        if (mpi.rank == 0) {
            std::ofstream out(directory/"fitted.mtp"); out << "previous successful model\n";
        }
        MPI_Barrier(mpi.comm);
    }
    void check_preserved() const {
        std::ifstream in(directory/"fitted.mtp");
        REQUIRE(std::string(std::istreambuf_iterator<char>(in), {}) == "previous successful model\n");
    }
};
}

TEST_CASE("Nonlinear BFGS rejects invalid options before evaluating or saving", "[training][nonlinear-failure]") {
    const auto [key, value] = GENERATE(Catch::Generators::table<std::string,std::string>({{"bfgs_m","0"},
        {"bfgs_m","2"}, {"bfgs_maxiteration","-1"}, {"bfgs_grad_tol","nan"},
        {"bfgs_step_tol","inf"}, {"bfgs_conv_tol","-1"}, {"scaled_step_max","nan"},
        {"save_every","0"}, {"bfgs_iteration_offset","-1"}}));
    TrainingCase fixture;
    auto settings = fixture.settings();
    // In MPI this deliberately differs on one rank: rejection must be collective.
    if (mpi.rank == mpi.size-1) settings.Modify(key,value);
    TrainNonlinear trainer(settings);
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    fixture.preserve_file();
    REQUIRE_THROWS(trainer.TrainBFGS());
    REQUIRE(fixture.potential.evaluations == 0);
    REQUIRE(fixture.potential.saves == 0);
    REQUIRE(fixture.potential.coefficient == 3.);
    REQUIRE_FALSE(trainer.bfgs_converged());
    fixture.check_preserved();
}

TEST_CASE("Nonlinear BFGS collectively propagates evaluation failures and restores coefficients", "[training][nonlinear-failure]") {
    const auto fault = GENERATE(FailurePotential::Fault::energy, FailurePotential::Fault::gradient,
        FailurePotential::Fault::nan_energy, FailurePotential::Fault::nan_gradient,
        FailurePotential::Fault::penalty, FailurePotential::Fault::alglib);
    const int fail_after = GENERATE(0,2);
    TrainingCase fixture;
    TrainNonlinear trainer(fixture.settings());
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    fixture.potential.fault = fault;
    fixture.potential.fail_after = fail_after;
    fixture.potential.fail_rank = mpi.size-1;
    fixture.preserve_file();
    REQUIRE_THROWS(trainer.TrainBFGS());
    REQUIRE(fixture.potential.coefficient == 3.);
    REQUIRE(fixture.potential.saves == 0);
    REQUIRE(trainer.bfgs_termination_type == 0);
    fixture.check_preserved();
}

TEST_CASE("Nonlinear BFGS keeps finite budget-limited accepted points and checkpoints", "[training][nonlinear-failure]") {
    const bool empty_shard = GENERATE(false,true);
    TrainingCase fixture;
    if (empty_shard && mpi.rank != 0) { fixture.data.clear(); fixture.neighbors.clear(); }
    TrainNonlinear trainer(fixture.settings());
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    REQUIRE_NOTHROW(trainer.TrainBFGS());
    REQUIRE(trainer.bfgs_accepted_iter == 1);
    REQUIRE(trainer.bfgs_termination_type == 5);
    REQUIRE_FALSE(trainer.bfgs_converged());
    REQUIRE(std::isfinite(fixture.potential.coefficient));
    REQUIRE(std::abs(fixture.potential.coefficient*fixture.potential.coefficient-1.) < 8.);
    MPI_Barrier(mpi.comm);
    double final = 0., accepted = 0.;
    std::ifstream(fixture.directory/"fitted.mtp") >> final;
    std::ifstream(fixture.directory/"fitted.mtp.iter-000001.mtp") >> accepted;
    REQUIRE(final == fixture.potential.coefficient);
    REQUIRE(accepted == final);
}

TEST_CASE("Nonlinear BFGS reports convergence separately from exhausted budgets", "[training][nonlinear-failure]") {
    TrainingCase fixture;
    fixture.potential.coefficient = 1.;
    TrainNonlinear trainer(fixture.settings());
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    REQUIRE_NOTHROW(trainer.TrainBFGS());
    REQUIRE(trainer.bfgs_converged());
    REQUIRE(trainer.bfgs_accepted_iter == 0);
    REQUIRE(fixture.potential.coefficient == 1.);
}

TEST_CASE("Nonlinear BFGS propagates checkpoint and final-save errors to every rank", "[training][nonlinear-failure]") {
    const int save_every = GENERATE(1,100);
    TrainingCase fixture;
    auto settings = fixture.settings(); settings.Modify("save_every", std::to_string(save_every));
    TrainNonlinear trainer(settings);
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    fixture.potential.fail_save = true;
    fixture.preserve_file();
    REQUIRE_THROWS(trainer.TrainBFGS());
    fixture.check_preserved();
    if (save_every == 1) REQUIRE(fixture.potential.coefficient == 3.);
}

TEST_CASE("Nonlinear BFGS synchronizes root-only scaling errors", "[training][nonlinear-failure]") {
    TrainingCase fixture;
    auto settings = fixture.settings(); settings.Modify("parameter_scaling", "file");
    TrainNonlinear trainer(settings);
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    fixture.preserve_file();
    REQUIRE_THROWS_WITH(trainer.TrainBFGS(), Catch::Matchers::ContainsSubstring("scale_data"));
    REQUIRE(fixture.potential.evaluations == 0);
    fixture.check_preserved();
}

TEST_CASE("Nonlinear BFGS rejects uninitialized and nonfinite starting models", "[training][nonlinear-failure]") {
    TrainNonlinear uninitialized;
    REQUIRE_THROWS(uninitialized.TrainBFGS());
    REQUIRE_THROWS(uninitialized.TrainNonLinearBFGS(1));
    TrainingCase fixture;
    TrainNonlinear trainer(fixture.settings());
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    fixture.preserve_file();
    if (mpi.rank == mpi.size-1) fixture.potential.coefficient = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS(trainer.TrainBFGS());
    REQUIRE(fixture.potential.evaluations == 0);
    fixture.check_preserved();
}

TEST_CASE("Nonlinear BFGS failure retains accepted checkpoints and permits a fresh run", "[training][nonlinear-failure]") {
    TrainingCase fixture;
    auto settings = fixture.settings(); settings.Modify("bfgs_maxiteration", "3");
    TrainNonlinear trainer(settings);
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    fixture.potential.fault = FailurePotential::Fault::energy;
    fixture.potential.fail_after = 3; // fail on a probe after the first accepted step
    fixture.potential.fail_rank = mpi.size-1;
    fixture.preserve_file();
    REQUIRE_THROWS(trainer.TrainBFGS());
    REQUIRE(trainer.bfgs_accepted_iter == 1);
    REQUIRE(fixture.potential.coefficient == 3.);
    REQUIRE_FALSE(trainer.bfgs_converged());
    fixture.check_preserved();
    MPI_Barrier(mpi.comm);
    double accepted = 0.;
    REQUIRE(bool(std::ifstream(fixture.directory/"fitted.mtp.iter-000001.mtp") >> accepted));
    REQUIRE(std::isfinite(accepted));
    REQUIRE(std::abs(accepted*accepted-1.) < 8.);
    fixture.potential.fault = FailurePotential::Fault::none;
    REQUIRE_NOTHROW(trainer.TrainBFGS());
    REQUIRE(trainer.bfgs_accepted_iter == 3);
    REQUIRE(trainer.bfgs_termination_type == 5);
}

TEST_CASE("Nonlinear BFGS propagates SOG stage snapshot failures before the next collective", "[training][nonlinear-failure][sog]") {
    const auto [schedule, suffix] = GENERATE(Catch::Generators::table<std::string,std::string>({
        {"mtp_sog_from_scratch", ".sr.snapshot"},
        {"mtp_sog_from_scratch", ".sog.snapshot"},
        {"mtp_sog_resume", ".sog.snapshot"}}));
    class StagePotential : public Potentials::PotMTP_LES {
    public:
        std::string fail_suffix;
        int evaluations = 0;
        StagePotential() : PotMTP_LES("unitest/test-data/portable-mtp/les.mtp") {
            les_kernel = "sog"; sog_terms = 2; sog_channels = 1;
            train_init(); set_inited();
        }
        void getEFS(Atoms &a, AnyNeighborhoods &n) override {
            ++evaluations;
            PotMTP_LES::getEFS(a,n);
        }
        void Save(const std::string &path) override {
            if (path.ends_with(fail_suffix)) throw std::runtime_error("injected stage snapshot failure");
        }
    } potential;
    potential.fail_suffix = suffix;
    TrainingCase fixture;
    auto settings = fixture.settings();
    settings.Modify("training_schedule", schedule);
    settings.Modify("sr_stage_iterations", "0");
    settings.Modify("les_stage_iterations", "0");
    settings.Modify("joint_stage_iterations", "1");
    TrainNonlinear trainer(settings);
    trainer.init(fixture.data, fixture.neighbors, &potential);
    fixture.preserve_file();
    REQUIRE_THROWS_WITH(trainer.TrainBFGS(), Catch::Matchers::ContainsSubstring("stage snapshot failure"));
    REQUIRE(potential.evaluations == 0);
    fixture.check_preserved();
}

TEST_CASE("Nonlinear BFGS synchronizes validation failures with empty validation shards", "[training][nonlinear-failure][validation]") {
    const bool fail_in_getter = GENERATE(false,true);
    TrainingCase fixture;
    auto settings = fixture.settings(); settings.Modify("validation_interval", "99");
    TrainNonlinear trainer(settings);
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    auto valid = fixture.data;
    valid[0].id = 42;
    auto neighbors = fixture.neighbors;
    if (mpi.rank != mpi.size-1) { valid.clear(); neighbors.clear(); }
    trainer.init_valid(valid, neighbors);
    fixture.potential.fail_rank = mpi.size-1;
    if (fail_in_getter) fixture.potential.getter_failure_at_validation = 2;
    else fixture.potential.fault = FailurePotential::Fault::validation;
    fixture.preserve_file();
    REQUIRE_THROWS(trainer.TrainBFGS());
    REQUIRE(fixture.potential.coefficient == 3.);
    REQUIRE_FALSE(trainer.bfgs_converged());
    fixture.check_preserved();
    fixture.potential.getter_failure_at_validation = 0;
    fixture.potential.fault = FailurePotential::Fault::none;
    REQUIRE_NOTHROW(trainer.TrainBFGS());
    REQUIRE(trainer.bfgs_termination_type == 5);
}

TEST_CASE("Nonlinear BFGS preserves independent gradient objective and step stopping meanings", "[training][nonlinear-failure][tolerances]") {
    const auto [gradient, objective, step, reason, accepted] = GENERATE(
        Catch::Generators::table<double,double,double,int,int>({
            {100., 0., 0., 4, 0},
            {1e-30, 1., 0., 1, 1},
            {1e-30, 0., 2., 2, 1}
        }));
    TrainingCase fixture;
    auto settings = fixture.settings();
    settings.Modify("bfgs_maxiteration", "10");
    settings.Modify("bfgs_grad_tol", gradient > 1. ? "100" : "1e-30");
    settings.Modify("bfgs_conv_tol", std::to_string(objective));
    settings.Modify("bfgs_step_tol", std::to_string(step));
    // Loss (theta^2-1)^2 starts at theta=3: f=64, |g|=96.
    // The first normalized step gives theta=2: f=9, |g|=24.
    // Thus each case selects a distinct ALGLIB convergence criterion.
    TrainNonlinear trainer(settings);
    trainer.init(fixture.data, fixture.neighbors, &fixture.potential);
    REQUIRE_NOTHROW(trainer.TrainBFGS());
    REQUIRE(trainer.bfgs_converged());
    REQUIRE(trainer.bfgs_termination_type == reason);
    REQUIRE(trainer.bfgs_accepted_iter == accepted);
    REQUIRE(fixture.potential.coefficient == Catch::Approx(accepted ? 2. : 3.));
}
