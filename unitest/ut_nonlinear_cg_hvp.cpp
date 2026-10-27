#include <catch2/catch_all.hpp>
#include "nonlinear_cg_fixture.h"

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    MPI_Init(&argc,&argv); mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed=20261020; session.configData().runOrder=Catch::TestRunOrder::Declared;
    int result=session.run(argc,argv); MPI_Finalize(); return result;
}
#endif
using namespace CGFixture;
namespace {
class Quadratic : public Potential {
public:
    bool bad_pullback = false, bad_validation = false;
    std::vector<Eigen::RowVector3d> cached;
    Quadratic() { cutoff = 2.; species_count = 1; }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        ++evaluations;
        cached = a.pos_;
        a.resetEFS();
        for (int i = 0; i < a.size(); ++i) {
            a.energy_ += .5 * coefficients[0] * coefficients[0] * a.pos_[i].squaredNorm();
            a.forces_[i] = -coefficients[0] * coefficients[0] * a.pos_[i];
        }
        if (bad_validation && a.features_.count("fail_hvp")) a.forces_[0][0] = std::numeric_limits<double>::quiet_NaN();
    }
    void AccumulateEFSCombinationGard(Atoms &, AnyNeighborhoods &, double dE, double *g) override {
        for (const auto &r : cached) g[0] += dE * coefficients[0] * r.squaredNorm();
    }
    void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &n, double dE,
        Eigen::Matrix<double,-1,3> &dF, Eigen::Matrix3d &,
        Eigen::Matrix<double,-1,3,Eigen::RowMajor> &, double *g) override {
        if (bad_pullback) throw std::runtime_error("injected HVP pullback failure");
        AccumulateEFSCombinationGard(a,n,dE,g);
        for (int i = 0; i < a.size(); ++i) g[0] -= 2 * coefficients[0] * dF.row(i).dot(cached[i]);
    }
};
void hvp_data(Case &c) {
    for (auto &a : c.data) {
        a.have_energy = a.have_stress = false; a.forces_.clear(); a.fitness = 1.3;
        a.hvp_directions_.resize(3, 2);
        a.hvp_directions_ << 1., 0., .2, .7, 0., 1.;
        a.hvp_targets_ = 1.7 * a.hvp_directions_;
    }
}
GPSettings hvp_settings(Case &c) {
    auto s = c.settings();
    s.Modify("energy_weight", "0"); s.Modify("hvp_weight", ".3");
    s.Modify("bfgs_maxiteration", "100"); s.Modify("bfgs_grad_tol", "1e-7");
    return s;
}
}

TEST_CASE("CG HVP-only objective has independent finite-difference parameter gradient and converges", "[training][nonlinear-cg][cg-hvp]") {
    Case c; hvp_data(c); Quadratic p;
    TrainNonlinear trainer(hvp_settings(c)); trainer.init(c.data,c.neighbors,&p);
    double value; Eigen::VectorXd gradient(2);
    trainer.AccumulateGard(value,gradient);
    const double expected = .3 * 1.3 / 2 * std::pow(p.coefficients[0]*p.coefficients[0]-1.7,2) * c.data[0].hvp_directions_.squaredNorm();
    REQUIRE(value == Catch::Approx(expected).epsilon(1e-8));
    const auto initial = p.coefficients;
    for (int k = 0; k < 2; ++k) {
        p.coefficients = initial; p.coefficients[k] += 1e-5;
        double plus; trainer.AccumulateLoss(plus);
        p.coefficients = initial; p.coefficients[k] -= 1e-5;
        double minus; trainer.AccumulateLoss(minus);
        REQUIRE(gradient[k] == Catch::Approx((plus-minus)/2e-5).epsilon(1e-5).margin(1e-6));
    }
    p.coefficients = initial;
    auto validation = c.data;
    trainer.init_valid(validation,c.neighbors);
    REQUIRE_NOTHROW(trainer.TrainCG());
    REQUIRE(p.coefficients[0]*p.coefficients[0] == Catch::Approx(1.7).epsilon(1e-6));
    REQUIRE(p.coefficients[1] == initial[1]);
    REQUIRE(trainer.validation_evaluations >= 1);
    REQUIRE(trainer.emonitor_valid.hvp_rms < 1e-6);
    REQUIRE(trainer.emonitor_valid.nhvp == 18);
}

TEST_CASE("CG HVP rejects malformed or ineffective reference observations before evaluation", "[training][nonlinear-cg][cg-hvp]") {
    const int invalid = GENERATE(0,1,2,3);
    Case c; hvp_data(c); Quadratic p;
    TrainNonlinear trainer(hvp_settings(c)); trainer.init(c.data,c.neighbors,&p); c.preserve();
    if (invalid == 0) trainer.training_set_[0].hvp_targets_(0,0) = std::numeric_limits<double>::quiet_NaN();
    if (invalid == 1) trainer.training_set_[0].hvp_directions_.setZero();
    if (invalid == 2) for (auto &a : trainer.training_set_) a.clear_hvp();
    if (invalid == 3) for (auto &a : trainer.training_set_) a.fitness = 0.;
    REQUIRE_THROWS(trainer.TrainCG());
    REQUIRE(p.evaluations == 0);
    REQUIRE(c.saved() == "previous successful model\n");
}

TEST_CASE("CG HVP pullback and accepted-point validation failures restore coefficients and output", "[training][nonlinear-cg][cg-hvp]") {
    const bool validation_failure = GENERATE(false,true);
    Case c; hvp_data(c); Quadratic p;
    TrainNonlinear trainer(hvp_settings(c)); trainer.init(c.data,c.neighbors,&p);
    if (validation_failure) {
        auto valid = c.data; for (auto &a : valid) { a.id = 42; a.features_["fail_hvp"] = "1"; }
        trainer.init_valid(valid,c.neighbors); p.bad_validation = true;
    } else p.bad_pullback = true;
    auto initial = p.coefficients; c.preserve();
    REQUIRE_THROWS(trainer.TrainCG());
    REQUIRE(p.coefficients == initial);
    REQUIRE(trainer.cg_termination_type == 0);
    REQUIRE(c.saved() == "previous successful model\n");
}
