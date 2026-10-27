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
class Charged : public Potential {
public:
    bool bad_prediction = false, bad_pullback = false, bad_validation = false;
    bool supports_total_charge_training() const override { return true; }
    void getEFS(Atoms &a, AnyNeighborhoods &n) override {
        Potential::getEFS(a,n);
        a.charges_ = {coefficients[0]*coefficients[0]*a.pos_[0][0]+coefficients[1]};
        if (bad_prediction || (a.id == 42 && bad_validation))
            a.charges_[0] = std::numeric_limits<double>::quiet_NaN();
    }
    double predicted_total_charge(const Atoms &a) const override { return a.charges_[0]; }
    void AccumulateTotalChargeGard(Atoms &a, AnyNeighborhoods &, double weight, double *g) override {
        if (bad_pullback) throw std::runtime_error("injected charge pullback failure");
        g[0] += 2 * weight * coefficients[0] * a.pos_[0][0];
        g[1] += weight;
    }
};
void charge_data(Case &c) {
    for (auto &a : c.data) {
        a.have_energy = a.have_stress = false; a.forces_.clear(); a.fitness = 1.3;
        a.global_attribs_d_["total_charge"] = 1.7 * a.pos_[0][0] + .4;
    }
}
GPSettings charge_settings(Case &c) {
    auto s = c.settings(); s.Modify("energy_weight", "0"); s.Modify("charge_weight", ".3");
    s.Modify("bfgs_maxiteration", "100"); s.Modify("bfgs_grad_tol", "1e-8");
    return s;
}
}

TEST_CASE("CG nonzero total-charge-only loss has independent gradient and converges", "[training][nonlinear-cg][cg-charge]") {
    Case c; charge_data(c); Charged p;
    TrainNonlinear trainer(charge_settings(c)); trainer.init(c.data,c.neighbors,&p);
    double value; Eigen::VectorXd gradient(2);
    trainer.AccumulateGard(value,gradient);
    double expected = 0;
    for (const auto &a : c.data) {
        const double err = (p.coefficients[0]*p.coefficients[0]-1.7)*a.pos_[0][0]+p.coefficients[1]-.4;
        expected += .3 * 1.3 / c.data.size() * err*err;
    }
    REQUIRE(value == Catch::Approx(expected));
    const auto initial = p.coefficients;
    for (int k = 0; k < 2; ++k) {
        p.coefficients = initial; p.coefficients[k] += 1e-5;
        double plus; trainer.AccumulateLoss(plus);
        p.coefficients = initial; p.coefficients[k] -= 1e-5;
        double minus; trainer.AccumulateLoss(minus);
        REQUIRE(gradient[k] == Catch::Approx((plus-minus)/2e-5).epsilon(1e-6));
    }
    p.coefficients = initial;
    auto valid = c.data; for (auto &a : valid) a.id = 42;
    trainer.init_valid(valid,c.neighbors);
    REQUIRE_NOTHROW(trainer.TrainCG());
    REQUIRE(p.coefficients[0]*p.coefficients[0] == Catch::Approx(1.7).epsilon(1e-6));
    REQUIRE(p.coefficients[1] == Catch::Approx(.4).margin(1e-6));
    REQUIRE(trainer.qmonitor_valid.qrms < 1e-6);
    REQUIRE(trainer.validation_evaluations >= 1);
}

TEST_CASE("CG charge-only supervision honors frozen nonlinear parameter groups", "[training][nonlinear-cg][cg-charge]") {
    Case c; charge_data(c); Charged p;
    auto s = charge_settings(c); s.Modify("trainable_groups", "linear");
    TrainNonlinear trainer(s); trainer.init(c.data,c.neighbors,&p);
    const auto initial = p.coefficients;
    REQUIRE_NOTHROW(trainer.TrainCG());
    REQUIRE(p.coefficients[0] == initial[0]);
    REQUIRE(p.coefficients[1] != initial[1]);
}

TEST_CASE("CG charge errors reject before evaluation or restore failed objective models", "[training][nonlinear-cg][cg-charge]") {
    const int failure = GENERATE(0,1,2,3,4,5,6);
    Case c; charge_data(c); Charged p;
    TrainNonlinear trainer(charge_settings(c)); trainer.init(c.data,c.neighbors,&p);
    if (failure == 0) trainer.training_set_[0].global_attribs_d_["total_charge"] = std::numeric_limits<double>::quiet_NaN();
    if (failure == 1) { auto &a = trainer.training_set_[0]; a.global_attribs_d_.erase("total_charge"); a.features_["total_charge"] = "3 broken"; }
    if (failure == 2) for (auto &a : trainer.training_set_) a.fitness = 0;
    if (failure == 3) p.bad_prediction = true;
    if (failure == 4) p.bad_pullback = true;
    if (failure == 5) {
        auto valid = c.data; for (auto &a : valid) a.id = 42;
        trainer.init_valid(valid,c.neighbors); p.bad_validation = true;
    }
    if (failure == 6) {
        GPSettings invalid;
        invalid.Modify("charge_pow", "inf");
        trainer.ApplySettings(invalid);
    }
    const auto initial = p.coefficients; c.preserve();
    REQUIRE_THROWS(trainer.TrainCG());
    if (failure <= 2 || failure == 6) REQUIRE(p.evaluations == 0);
    REQUIRE(p.coefficients == initial);
    REQUIRE(trainer.cg_termination_type == 0);
    REQUIRE(c.saved() == "previous successful model\n");
}

TEST_CASE("CG combines nonzero HVP and total-charge supervision in one objective", "[training][nonlinear-cg][cg-charge][cg-hvp]") {
    class ChargedQuadratic : public Charged {
    public:
        void getEFS(Atoms &a, AnyNeighborhoods &n) override {
            Charged::getEFS(a,n);
            a.energy_ = .5 * coefficients[0]*coefficients[0] * a.pos_[0].squaredNorm();
            a.forces_[0] = -coefficients[0]*coefficients[0] * a.pos_[0];
        }
        void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &, double dE, double *g) override {
            g[0] += dE * coefficients[0] * a.pos_[0].squaredNorm();
        }
        void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &, double dE,
            Eigen::Matrix<double,-1,3> &dF, Eigen::Matrix3d &,
            Eigen::Matrix<double,-1,3,Eigen::RowMajor> &, double *g) override {
            g[0] += dE * coefficients[0] * a.pos_[0].squaredNorm()
                - 2 * coefficients[0] * dF.row(0).dot(a.pos_[0]);
        }
    } p;
    Case c; charge_data(c);
    for (auto &a : c.data) {
        a.hvp_directions_ = Eigen::MatrixXd::Identity(3,3);
        a.hvp_targets_ = 1.7 * a.hvp_directions_;
    }
    auto s = charge_settings(c); s.Modify("hvp_weight", ".3");
    TrainNonlinear trainer(s); trainer.init(c.data,c.neighbors,&p);
    REQUIRE_NOTHROW(trainer.TrainCG());
    REQUIRE(p.coefficients[0]*p.coefficients[0] == Catch::Approx(1.7).epsilon(1e-6));
    REQUIRE(p.coefficients[1] == Catch::Approx(.4).margin(1e-6));
    REQUIRE(trainer.emonitor.hvp_rms < 1e-6);
    REQUIRE(trainer.qmonitor.qrms < 1e-6);
}
