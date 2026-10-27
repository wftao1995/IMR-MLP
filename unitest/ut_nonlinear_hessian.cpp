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
Eigen::Matrix3d kernel() {
    Eigen::Matrix3d k;
    k << 2., .3, -.2, .3, 3., .4, -.2, .4, 1.;
    return k;
}
class Quadratic : public Potential {
public:
    bool bad_pullback = false, bad_validation = false;
    std::vector<Eigen::RowVector3d> cached;
    Quadratic() { cutoff = 2.; species_count = 1; }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        ++evaluations; cached = a.pos_; a.resetEFS();
        for (int i = 0; i < a.size(); ++i) {
            a.energy_ += .5 * coefficients[0]*coefficients[0] * a.pos_[i].dot(a.pos_[i]*kernel());
            a.forces_[i] = -coefficients[0]*coefficients[0] * a.pos_[i]*kernel();
        }
        if (bad_validation && a.features_.count("fail_fd"))
            a.forces_[0][0] = std::numeric_limits<double>::quiet_NaN();
    }
    void AccumulateEFSCombinationGard(Atoms &, AnyNeighborhoods &, double dE, double *g) override {
        for (const auto &r : cached) g[0] += dE * coefficients[0] * r.dot(r*kernel());
    }
    void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &n, double dE,
        Eigen::Matrix<double,-1,3> &dF, Eigen::Matrix3d &,
        Eigen::Matrix<double,-1,3,Eigen::RowMajor> &, double *g) override {
        if (bad_pullback) throw std::runtime_error("injected full Hessian pullback failure");
        AccumulateEFSCombinationGard(a,n,dE,g);
        for (int i = 0; i < a.size(); ++i)
            g[0] -= 2 * coefficients[0] * dF.row(i).dot(cached[i]*kernel());
    }
};
void hessian_data(Case &c, int count = 1) {
    c.neighbors.clear();
    for (auto &a : c.data) {
        if (count == 2) { a.pos_.push_back({7.,6.,5.}); a.chemmap_.push_back(0); }
        a.have_energy = a.have_stress = false; a.forces_.clear(); a.fitness = 1.3;
        a.hessian_ = Eigen::MatrixXd::Zero(3*count,3*count);
        for (int i = 0; i < count; ++i) a.hessian_.block<3,3>(3*i,3*i) = 1.7*kernel();
        a.have_hessian = true;
        c.neighbors.emplace_back(a,2.);
    }
}
GPSettings hessian_settings(Case &c) {
    auto s = c.settings(); s.Modify("energy_weight", "0"); s.Modify("hessian_weight", ".3");
    s.Modify("bfgs_maxiteration", "100"); s.Modify("bfgs_grad_tol", "1e-6");
    return s;
}
}

TEST_CASE("Full Hessian FD loss matches known offdiagonal Frobenius matrix and parameter gradient", "[training][full-hessian]") {
    const int count = GENERATE(1,2);
    Case c; hessian_data(c,count); Quadratic p;
    auto settings = hessian_settings(c); settings.Modify("hessian_pow", "1");
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&p);
    double loss; Eigen::VectorXd gradient(2); trainer.AccumulateGard(loss,gradient);
    // Sum of all nine K entries squared is14.58; offdiagonal terms count twice.
    const double diff = p.coefficients[0]*p.coefficients[0]-1.7;
    REQUIRE(loss == Catch::Approx(.3*1.3*diff*diff*14.58).epsilon(1e-8));
    const auto initial = p.coefficients;
    for (int k = 0; k < 2; ++k) {
        p.coefficients = initial; p.coefficients[k] += 1e-5;
        double plus; trainer.AccumulateLoss(plus);
        p.coefficients = initial; p.coefficients[k] -= 1e-5;
        double minus; trainer.AccumulateLoss(minus);
        REQUIRE(gradient[k] == Catch::Approx((plus-minus)/2e-5).epsilon(2e-5).margin(1e-6));
    }
    REQUIRE(trainer.emonitor.nhessian == 3 * 9 * count * count);
}

TEST_CASE("Full Hessian only BFGS and CG train actual offdiagonal curvature with accepted validation", "[training][full-hessian]") {
    const bool cg = GENERATE(false,true);
    Case c; hessian_data(c); Quadratic p;
    TrainNonlinear trainer(hessian_settings(c)); trainer.init(c.data,c.neighbors,&p);
    auto valid = c.data; trainer.init_valid(valid,c.neighbors);
    const auto initial = p.coefficients;
    if (cg) REQUIRE_NOTHROW(trainer.TrainCG());
    else REQUIRE_NOTHROW(trainer.TrainBFGS());
    REQUIRE(p.coefficients[0]*p.coefficients[0] == Catch::Approx(1.7).epsilon(1e-6));
    REQUIRE(p.coefficients[1] == initial[1]);
    REQUIRE(trainer.emonitor_valid.hrms < 1e-6);
    REQUIRE(trainer.emonitor_valid.nhessian == 27);
    REQUIRE(trainer.validation_evaluations >= 1);
}

TEST_CASE("Full Hessian labels and HVP probes contribute independent additive losses", "[training][full-hessian]") {
    Case c; hessian_data(c); Quadratic p;
    auto settings = hessian_settings(c); settings.Modify("hvp_weight", ".6");
    for (auto &a : c.data) {
        a.hvp_directions_ = Eigen::MatrixXd::Identity(3,3);
        a.hvp_targets_ = a.hessian_;
    }
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&p);
    double loss; Eigen::VectorXd gradient(2); trainer.AccumulateGard(loss,gradient);
    const double error = std::pow(p.coefficients[0]*p.coefficients[0]-1.7,2)*14.58;
    REQUIRE(loss == Catch::Approx(1.3*(.3+.6/3)*error).epsilon(1e-8));
    REQUIRE(trainer.emonitor.nhessian == 27);
    REQUIRE(trainer.emonitor.nhvp == 27);
}

TEST_CASE("Full Hessian CG label and callback failures preserve coefficients and prior model", "[training][full-hessian]") {
    const int failure = GENERATE(0,1,2,3,4);
    Case c; hessian_data(c); Quadratic p;
    TrainNonlinear trainer(hessian_settings(c)); trainer.init(c.data,c.neighbors,&p);
    if (failure == 0) trainer.training_set_[0].hessian_.resize(2,2);
    if (failure == 1) trainer.training_set_[0].hessian_(0,0) = std::numeric_limits<double>::quiet_NaN();
    if (failure == 2) for (auto &a : trainer.training_set_) a.have_hessian = false;
    if (failure == 3) p.bad_pullback = true;
    if (failure == 4) {
        auto valid = c.data; for (auto &a : valid) a.features_["fail_fd"] = "1";
        trainer.init_valid(valid,c.neighbors); p.bad_validation = true;
    }
    const auto initial = p.coefficients; c.preserve();
    REQUIRE_THROWS(trainer.TrainCG());
    if (failure < 3) REQUIRE(p.evaluations == 0);
    REQUIRE(p.coefficients == initial);
    REQUIRE(c.saved() == "previous successful model\n");
}

TEST_CASE("Full Hessian loss consumes nonsymmetric targets with shared EFS normalization", "[training][full-hessian]") {
    Case c; hessian_data(c); Quadratic p;
    for (auto &a : c.data) {
        a.hessian_(0,1) += .2; // Only one triangle is changed; no symmetrization.
        a.forces_ = {{1.,0.,0.}};
    }
    auto settings = hessian_settings(c);
    settings.Modify("force_wgt_d", "2");
    settings.Modify("force_wgt_f", "3");
    settings.Modify("force_wgt_p", "1");
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&p);
    double loss; Eigen::VectorXd gradient(2); trainer.AccumulateGard(loss,gradient);
    Eigen::Matrix3d residual = (p.coefficients[0]*p.coefficients[0]-1.7)*kernel();
    residual(0,1) -= .2;
    const double expected = .3 * 1.3 * (2./5.) * residual.squaredNorm();
    REQUIRE(loss == Catch::Approx(expected).epsilon(1e-8));
    const auto initial = p.coefficients;
    p.coefficients[0] += 1e-5; double plus; trainer.AccumulateLoss(plus);
    p.coefficients = initial;
    p.coefficients[0] -= 1e-5; double minus; trainer.AccumulateLoss(minus);
    p.coefficients = initial;
    REQUIRE(gradient[0] == Catch::Approx((plus-minus)/2e-5).epsilon(2e-5));
}
