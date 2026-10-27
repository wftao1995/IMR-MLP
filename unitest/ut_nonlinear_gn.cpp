#include <catch2/catch_all.hpp>
#include "nonlinear_cg_fixture.h"

using Catch::Approx;
namespace {
class ResidualModel : public CGFixture::Potential {
public:
    std::vector<Eigen::RowVector3d> cached;
    void train_init() override { reg_L1_vector.assign(2,0.); reg_L2_vector.assign(2,0.); }
    void getEFS(Atoms &a, AnyNeighborhoods &) override {
        cached = a.pos_; a.resetEFS();
        const double t = coefficients[0], u = coefficients[1];
        for (int i = 0; i < a.size(); ++i) {
            a.energy_ += .5*t*t*a.pos_[i].squaredNorm()+u*a.pos_[i][0];
            a.forces_[i] = -t*t*a.pos_[i].transpose()-Eigen::Vector3d(u,0.,0.);
        }
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
            a.stress_(i,j) = t*(i+1)+u*u*(j+1);
    }
    void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &, double dE, double *g) override {
        REQUIRE(cached == a.pos_);
        for (const auto &r : cached) { g[0] += dE*coefficients[0]*r.squaredNorm(); g[1] += dE*r[0]; }
    }
    void AccumulateEFSCombinationGard(Atoms &a, AnyNeighborhoods &n, double dE,
        Eigen::Matrix<double,-1,3> &force, Eigen::Matrix3d &stress,
        Eigen::Matrix<double,-1,3,Eigen::RowMajor> &, double *g) override {
        AccumulateEFSCombinationGard(a,n,dE,g);
        for (int i = 0; i < a.size(); ++i) {
            g[0] -= 2.*coefficients[0]*force.row(i).dot(cached[i]); g[1] -= force(i,0);
        }
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
            g[0] += stress(i,j)*(i+1); g[1] += stress(i,j)*2.*coefficients[1]*(j+1);
        }
    }
    bool supports_total_charge_training() const override { return true; }
    double predicted_total_charge(const Atoms &) const override { return coefficients.prod(); }
    void AccumulateTotalChargeGard(Atoms &a, AnyNeighborhoods &, double weight, double *g) override {
        REQUIRE(cached == a.pos_);
        g[0] += weight*coefficients[1]; g[1] += weight*coefficients[0];
    }
    void AddPenaltyGard(double weight, double &loss, double *g) override {
        // Deliberately nonuniform L2 mapping, plus a penalty which must not
        // appear in the L2-only metric diagonal.
        const double l2 = reg_L2_vector.empty() ? 0. : reg_L2_vector[0];
        loss += l2*coefficients[1]*coefficients[1]+weight*std::pow(coefficients[0],4);
        g[1] += 2.*l2*coefficients[1]; g[0] += 4.*weight*std::pow(coefficients[0],3);
    }
};
void prepare(CGFixture::Case &c) {
    for (int i = 0; i < c.data.size(); ++i) {
        auto &a = c.data[i]; a.fitness = .4+.2*i;
        a.pos_.push_back(Eigen::RowVector3d(1.4+.1*i,2.1,1.3)); a.chemmap_.push_back(0);
        a.forces_.push_back(Eigen::RowVector3d(.3,-.4,.2)); a.atomic_fitness = {.7,1.2};
        a.hvp_directions_.resize(6,2);
        a.hvp_directions_ << 1.,.3,.2,1.,.5,-.4,.1,-.7,-.6,.4,.8,.2;
        a.hvp_targets_ = a.hvp_directions_;
        a.hessian_ = Eigen::MatrixXd::Identity(6,6); a.have_hessian = true;
        c.neighbors[i] = Neighborhoods(a,c.potential.cutoff);
    }
}
GPSettings settings(const CGFixture::Case &c, const std::string &channels) {
    auto s = c.settings();
    const bool efs = channels == "efs-charge" || channels == "all";
    s.Modify("energy_weight",efs ? ".7" : "0");
    s.Modify("force_weight",efs ? ".3" : "0");
    s.Modify("stress_weight",efs ? ".2" : "0");
    s.Modify("charge_weight",efs ? ".4" : "0");
    s.Modify("hvp_weight",channels == "hvp" || channels == "all" ? ".6" : "0");
    s.Modify("hessian_weight",channels == "fullh" || channels == "all" ? ".5" : "0");
    s.Modify("energy_pow","1.2"); s.Modify("force_pow",".4"); s.Modify("stress_pow",".8");
    s.Modify("charge_pow",".5"); s.Modify("hvp_pow",".7"); s.Modify("hessian_pow",".9");
    s.Modify("force_wgt_f",".7"); s.Modify("gn_probes","8192");
    s.Modify("curvature_precondition","gn_diag");
    return s;
}
Eigen::VectorXd observations(CGFixture::Case &c, ResidualModel &p, const std::string &channels) {
    std::vector<double> values;
    const bool efs = channels == "efs-charge" || channels == "all";
    const double count = c.data.size()*mpi.size;
    for (int k = 0; k < c.data.size(); ++k) {
        const auto &a = c.data[k]; auto predicted = a; p.getEFS(predicted,c.neighbors[k]);
        double mean = 0.; for (const auto &f : a.forces_) mean += f.squaredNorm(); mean /= a.size();
        const double cc = .1/(.1+.7*mean), n = a.size();
        if (efs) {
            values.push_back(std::sqrt(.7*a.fitness*cc/std::pow(n,1.2)/count)*predicted.energy_);
            for (int i = 0; i < n; ++i) for (int j = 0; j < 3; ++j)
                values.push_back(std::sqrt(.3*a.atomic_fitness[i]*cc/std::pow(n,.4)/count)*predicted.forces_[i][j]);
            for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j)
                values.push_back(std::sqrt(.2*cc/std::pow(n,.8)/count)*predicted.stress_(i,j));
            values.push_back(std::sqrt(.4*a.fitness/std::pow(n,.5)/count)*p.predicted_total_charge(predicted));
        }
        if (channels == "hvp" || channels == "all") {
            const auto product = (p.coefficients[0]*p.coefficients[0]*a.hvp_directions_).eval();
            const double weight = std::sqrt(.6*a.fitness*cc/std::pow(n,.7)/a.hvp_directions_.cols()/count);
            for (int j = 0; j < product.cols(); ++j) for (int i = 0; i < product.rows(); ++i)
                values.push_back(weight*product(i,j));
        }
        if (channels == "fullh" || channels == "all") {
            const double weight = std::sqrt(.5*a.fitness*cc/std::pow(n,.9)/count);
            for (int j = 0; j < 3*n; ++j) for (int i = 0; i < 3*n; ++i)
                values.push_back(i == j ? weight*p.coefficients[0]*p.coefficients[0] : 0.);
        }
    }
    return Eigen::Map<const Eigen::VectorXd>(values.data(),values.size());
}
}

TEST_CASE("Weighted residual EFS charge HVP and full Hessian VJPs agree with independent parameter differences", "[training][gn-integration]") {
    const auto channels = GENERATE(std::string("efs-charge"),std::string("hvp"),std::string("fullh"),std::string("all"));
    CGFixture::Case c; prepare(c); ResidualModel p;
    TrainNonlinear trainer(settings(c,channels)); trainer.init(c.data,c.neighbors,&p);
    Eigen::VectorXd z(trainer.WeightedResidualCount());
    for (int i = 0; i < z.size(); ++i) z[i] = std::sin(.7*(i+1));
    Eigen::VectorXd gradient; trainer.AccumulateResidualVJP(z,gradient);
    const auto original = p.coefficients;
    for (const double step : {1e-4,2e-5}) for (int i = 0; i < 2; ++i) {
        CAPTURE(channels,step,i);
        p.coefficients = original; p.coefficients[i] += step;
        const double plus = observations(c,p,channels).dot(z);
        p.coefficients = original; p.coefficients[i] -= step;
        const double minus = observations(c,p,channels).dot(z);
        REQUIRE(gradient[i] == Approx((plus-minus)/(2.*step)).epsilon(2e-5).margin(1e-7));
    }
    p.coefficients = original;
    trainer.ConfigureTrainableGroups({"linear"}); trainer.AccumulateResidualVJP(z,gradient);
    REQUIRE(gradient[0] == 0.);
}

TEST_CASE("Production GN diagonal matches explicit residual differences and isolates model L2", "[training][gn-integration]") {
    CGFixture::Case c; prepare(c); ResidualModel p;
    auto s = settings(c,"all"); s.Modify("weight_L2",".37"); s.Modify("penalty_weight",".2");
    TrainNonlinear trainer(s); trainer.init(c.data,c.neighbors,&p);
    Eigen::Vector2d exact;
    const auto original = p.coefficients;
    for (int i = 0; i < 2; ++i) {
        p.coefficients = original; p.coefficients[i] += 1e-5; const auto plus = observations(c,p,"all");
        p.coefficients = original; p.coefficients[i] -= 1e-5; const auto minus = observations(c,p,"all");
        exact[i] = 2.*((plus-minus)/2e-5).squaredNorm()*mpi.size;
    }
    p.coefficients = original;
    exact[1] += .74; // Only the model's second coefficient has L2 curvature.
    const auto diagonal = trainer.EstimateGNDiagonal();
    REQUIRE(p.coefficients == original);
    REQUIRE(p.reg_L2_vector == (std::vector<double>{.37,.37}));
    for (int i = 0; i < 2; ++i) REQUIRE(diagonal[i] == Approx(exact[i]).epsilon(.07));
    trainer.ConfigureTrainableGroups({"linear"});
    const auto masked = trainer.EstimateGNDiagonal();
    REQUIRE(masked[0] == 0.);
    REQUIRE(masked[1] == Approx(exact[1]).epsilon(.07));
}

TEST_CASE("GN accepted-point refresh restarts optimizer history without resetting iteration budgets", "[training][gn-integration]") {
    const bool cg = GENERATE(false,true);
    CGFixture::Case c;
    auto s = c.settings(); s.Modify("curvature_precondition","gn_diag");
    s.Modify("gn_update","1"); s.Modify("gn_probes","16"); s.Modify("bfgs_maxiteration","3");
    TrainNonlinear trainer(s); trainer.init(c.data,c.neighbors,&c.potential);
    const auto before = c.potential.coefficients;
    if (cg) trainer.TrainCG(); else trainer.TrainBFGS();
    REQUIRE(trainer.bfgs_accepted_iter == 3);
    REQUIRE(trainer.gn_refreshes == 3);
    REQUIRE(c.potential.coefficients != before);
    REQUIRE((cg ? trainer.cg_termination_type == 5 : trainer.bfgs_termination_type == 5));
}

TEST_CASE("CG shares file and group coordinate scales while preserving frozen coefficients", "[training][gn-integration]") {
    const auto mode = GENERATE(std::string("file"),std::string("group_auto"));
    CGFixture::Case c;
    c.directory = std::filesystem::absolute(c.directory);
    struct RestoreDirectory {
        std::filesystem::path original = std::filesystem::current_path();
        ~RestoreDirectory() { std::filesystem::current_path(original); }
    } restore;
    std::filesystem::current_path(c.directory);
    auto s = c.settings(); s.Modify("parameter_scaling",mode); s.Modify("trainable_groups","radial");
    s.Modify("scale_floor",".01"); s.Modify("scale_ceiling","10"); s.Modify("bfgs_maxiteration","3");
    if (mpi.rank == 0 && mode == "file") std::ofstream("scale_data") << ".3\n2\n";
    MPI_Barrier(mpi.comm);
    TrainNonlinear trainer(s); trainer.init(c.data,c.neighbors,&c.potential);
    const double frozen = c.potential.coefficients[1];
    REQUIRE_NOTHROW(trainer.TrainCG());
    REQUIRE(c.potential.coefficients[1] == frozen);
}

TEST_CASE("GN settings and cotangent validation fail before nonfinite metrics escape", "[training][gn-integration]") {
    CGFixture::Case c;
    auto s = c.settings(); s.Modify("curvature_precondition","gn_diag"); s.Modify("gn_probes","0");
    TrainNonlinear invalid(s);
    REQUIRE_THROWS(invalid.init(c.data,c.neighbors,&c.potential));
    TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential);
    Eigen::VectorXd gradient, wrong = Eigen::VectorXd::Zero(trainer.WeightedResidualCount()+1);
    REQUIRE_THROWS(trainer.AccumulateResidualVJP(wrong,gradient));
    Eigen::VectorXd bad = Eigen::VectorXd::Zero(trainer.WeightedResidualCount());
    bad[0] = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(trainer.AccumulateResidualVJP(bad,gradient));
}
