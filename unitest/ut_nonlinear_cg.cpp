#include <catch2/catch_all.hpp>
#include "nonlinear_cg_fixture.h"

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    MPI_Init(&argc,&argv); mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;
    session.configData().rngSeed=20261019; session.configData().runOrder=Catch::TestRunOrder::Declared;
    int result=session.run(argc,argv); MPI_Finalize(); return result;
}
#endif
using namespace CGFixture;

TEST_CASE("CG honors direct budgets and converges on nonlinear EFS data", "[training][nonlinear-cg]") {
    const bool efs=GENERATE(false,true);
    Case c; auto settings=c.settings(); settings.Modify("bfgs_maxiteration","20");
    if(efs) { settings.Modify("force_weight","1"); settings.Modify("stress_weight",".2"); }
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&c.potential);
    double initial=c.energy_loss();
    REQUIRE_NOTHROW(trainer.TrainNonLinearCG(1));
    REQUIRE(trainer.bfgs_accepted_iter == 1); REQUIRE(trainer.cg_termination_type == 5);
    REQUIRE_FALSE(trainer.cg_converged()); REQUIRE(c.energy_loss() < initial);
    REQUIRE_NOTHROW(trainer.TrainCG()); REQUIRE(c.energy_loss() < initial*.01);
    REQUIRE(c.potential.coefficients.allFinite());
    Eigen::Vector2d saved; std::ifstream(c.directory/"fitted.mtp") >> saved[0] >> saved[1];
    REQUIRE(saved == c.potential.coefficients);
}

TEST_CASE("CG resets fresh repeated and previous BFGS state and preserves frozen groups", "[training][nonlinear-cg]") {
    const bool prior_bfgs=GENERATE(false,true);
    Case c; auto settings=c.settings(); settings.Modify("trainable_groups","linear");
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&c.potential);
    if(prior_bfgs) REQUIRE_NOTHROW(trainer.TrainBFGS());
    const double frozen=c.potential.coefficients[0];
    for(int repeat=0;repeat<2;++repeat) {
        REQUIRE_NOTHROW(trainer.TrainCG());
        REQUIRE(trainer.bfgs_accepted_iter <= 1);
        REQUIRE(c.potential.coefficients[0] == frozen);
    }
    GPSettings all; all.Modify("trainable_groups","radial linear"); trainer.ApplySettings(all);
    REQUIRE_NOTHROW(trainer.TrainCG());
    REQUIRE(c.potential.coefficients[0] != frozen);
}

TEST_CASE("CG tolerance names map to gradient function and step criteria", "[training][nonlinear-cg]") {
    const auto [key,code]=GENERATE(Catch::Generators::table<std::string,int>({
        {"bfgs_grad_tol",4},{"bfgs_conv_tol",1},{"bfgs_step_tol",2}}));
    Case c; auto settings=c.settings(); settings.Modify(key,"1e100"); settings.Modify("bfgs_maxiteration","20");
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&c.potential);
    REQUIRE_NOTHROW(trainer.TrainCG()); REQUIRE(trainer.cg_termination_type == code);
    REQUIRE(trainer.cg_converged());
    REQUIRE(trainer.bfgs_accepted_iter == (code==4 ? 0 : 1));
}

TEST_CASE("CG initial solution validates without accepted checkpoints", "[training][nonlinear-cg]") {
    Case c; c.potential.coefficients << 1.,1.;
    TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential);
    auto valid=c.data; for(auto &a:valid) a.id=42; trainer.init_valid(valid,c.neighbors);
    REQUIRE_NOTHROW(trainer.TrainCG()); REQUIRE(trainer.cg_converged());
    REQUIRE(trainer.bfgs_accepted_iter == 0); REQUIRE(trainer.validation_evaluations == 1);
    REQUIRE_FALSE(fs::exists(c.directory/"fitted.mtp.iter-000001.mtp"));
}

TEST_CASE("CG rejects unsupported and invalid settings at both public entry points", "[training][nonlinear-cg]") {
    const auto [key,value]=GENERATE(Catch::Generators::table<std::string,std::string>({
        {"device","cuda"},{"training_schedule","stages"},{"hvp_weight","-1"},{"charge_weight","-1"},
        {"linear_refit","always"},{"parameter_scaling","invalid_mode"},
        {"scaled_step_max","-1"},{"objective-convention","mean-moments"},{"bfgs_maxiteration","0"},
        {"bfgs_maxiteration","-1"},{"bfgs_grad_tol","nan"},{"bfgs_step_tol","-1"},
        {"bfgs_conv_tol","inf"},{"energy_weight","-1"},{"force_weight","nan"},
        {"stress_weight","inf"},{"save_every","0"},{"validation_interval","0"},
        {"bfgs_iteration_offset","-1"},{"weight_L1","-1"},{"penalty_weight","nan"}}));
    Case c; TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential);
    GPSettings invalid; invalid.Modify(key,value); trainer.ApplySettings(invalid); c.preserve();
    REQUIRE_THROWS(trainer.ValidateCGSettings()); REQUIRE_THROWS(trainer.TrainCG());
    // A valid direct budget supersedes only the configured iteration budget.
    if(key!="bfgs_maxiteration") REQUIRE_THROWS(trainer.TrainNonLinearCG(1));
    else REQUIRE_THROWS(trainer.TrainNonLinearCG(std::stoi(value)));
    REQUIRE(c.potential.evaluations == 0); REQUIRE(c.potential.saves == 0);
    REQUIRE(c.saved() == "previous successful model\n"); REQUIRE(trainer.cg_termination_type == 0);
}

TEST_CASE("CG rejects invalid data and missing effective supervision before evaluation", "[training][nonlinear-cg]") {
    const int invalid=GENERATE(0,1,2,3,4,5,6,7,8);
    Case c; TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential); c.preserve();
    if(invalid==0) trainer.training_set_[0].pos_[0][0]=std::numeric_limits<double>::quiet_NaN();
    if(invalid==1) trainer.training_set_[0].energy_=std::numeric_limits<double>::infinity();
    if(invalid==2) trainer.training_set_[0].forces_.resize(2,Eigen::Vector3d::Zero());
    if(invalid==3) trainer.training_set_[0].atomic_fitness={1.,1.};
    if(invalid==4) trainer.training_set_[0].chemmap_.clear();
    if(invalid==5) trainer.trainint_set_nbhs_.clear();
    if(invalid==6) for(auto &a:trainer.training_set_) a.fitness=0.;
    if(invalid==7) trainer.training_set_[0].fitness=-1.;
    if(invalid==8) trainer.training_set_[0].cell_.setZero();
    REQUIRE_THROWS(trainer.TrainCG()); REQUIRE(c.potential.evaluations == 0);
    REQUIRE(c.saved() == "previous successful model\n");
}

TEST_CASE("CG evaluation failures restore coefficients and permit retry", "[training][nonlinear-cg]") {
    const auto fault=GENERATE(Potential::Fault::energy,Potential::Fault::gradient,Potential::Fault::nan_energy,
        Potential::Fault::nan_gradient,Potential::Fault::penalty,Potential::Fault::nan_penalty,Potential::Fault::alglib);
    const int after=GENERATE(0,4);
    Case c; TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential);
    auto initial=c.potential.coefficients; c.potential.fault=fault; c.potential.fail_after=after;
    c.potential.fail_rank=mpi.size-1; c.preserve();
    REQUIRE_THROWS(trainer.TrainCG()); REQUIRE(c.potential.coefficients == initial);
    REQUIRE(trainer.cg_termination_type == 0); REQUIRE(c.saved() == "previous successful model\n");
    c.potential.fault=Potential::Fault::none;
    REQUIRE_NOTHROW(trainer.TrainCG()); REQUIRE(trainer.cg_termination_type != 0);
}

TEST_CASE("CG validation failures and nonfinite predictions never publish", "[training][nonlinear-cg]") {
    const auto fault=GENERATE(Potential::Fault::validation,Potential::Fault::nan_validation,Potential::Fault::validation_getter);
    Case c; TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential);
    auto valid=c.data; for(auto &a:valid) a.id=42; trainer.init_valid(valid,c.neighbors);
    c.potential.fault=fault; c.potential.fail_rank=mpi.size-1; c.preserve();
    auto initial=c.potential.coefficients;
    REQUIRE_THROWS(trainer.TrainCG()); REQUIRE(c.potential.coefficients == initial);
    REQUIRE(trainer.cg_termination_type == 0); REQUIRE(c.saved() == "previous successful model\n");
    c.potential.fault=Potential::Fault::none;
    REQUIRE_NOTHROW(trainer.TrainCG()); REQUIRE(trainer.validation_evaluations >= 1);
}

TEST_CASE("CG save transactions retain previous model and roll back the live model", "[training][nonlinear-cg]") {
    const int save_every=GENERATE(1,100);
    const int failure=GENERATE(0,1,2);
    Case c; auto settings=c.settings(); settings.Modify("save_every",std::to_string(save_every));
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&c.potential); c.preserve();
    c.potential.fail_save=failure==0; c.potential.empty_save=failure==1; c.potential.sidecar_save=failure==2;
    auto initial=c.potential.coefficients;
    REQUIRE_THROWS(trainer.TrainCG()); REQUIRE(c.potential.coefficients == initial);
    REQUIRE(trainer.cg_termination_type == 0); REQUIRE(c.saved() == "previous successful model\n");
    for(const auto &entry:fs::directory_iterator(c.directory)) REQUIRE_FALSE(entry.path().filename().string().starts_with(".imrmlp-cg-save-"));
    c.potential.fail_save=false; c.potential.empty_save=false; c.potential.sidecar_save=false;
    REQUIRE_NOTHROW(trainer.TrainCG());
}

TEST_CASE("CG external saves require unchanged finite results and protect special paths", "[training][nonlinear-cg]") {
    Case c; TrainNonlinear trainer(c.settings());
    REQUIRE_THROWS(trainer.SaveCGResult((c.directory/"absent.mtp").string()));
    REQUIRE_THROWS(trainer.TrainNonLinearCG(1));
    trainer.init(c.data,c.neighbors,&c.potential); REQUIRE_NOTHROW(trainer.TrainCG());
    const auto saved=c.saved();
    if(mpi.rank==0) {fs::create_symlink("fitted.mtp",c.directory/"link"); fs::create_symlink("missing",c.directory/"dangling");}
    MPI_Barrier(mpi.comm);
    REQUIRE_THROWS(trainer.SaveCGResult(c.directory.string()));
    REQUIRE_THROWS(trainer.SaveCGResult((c.directory/"link").string()));
    REQUIRE_THROWS(trainer.SaveCGResult((c.directory/"dangling").string()));
    REQUIRE(c.saved() == saved);
    if(mpi.rank==mpi.size-1) c.potential.coefficients[0] += .1;
    REQUIRE_THROWS(trainer.SaveCGResult((c.directory/"fitted.mtp").string()));
    REQUIRE(c.saved() == saved);
}

TEST_CASE("CG direct limit replaces configured budget and handles empty MPI shards", "[training][nonlinear-cg]") {
    Case c; auto settings=c.settings(); settings.Modify("bfgs_maxiteration","0");
    if(mpi.rank!=0) { c.data.clear(); c.neighbors.clear(); }
    TrainNonlinear trainer(settings); trainer.init(c.data,c.neighbors,&c.potential);
    REQUIRE_THROWS(trainer.ValidateCGSettings());
    REQUIRE_NOTHROW(trainer.TrainNonLinearCG(1));
    REQUIRE(trainer.bfgs_accepted_iter == 1); REQUIRE(trainer.cg_termination_type == 5);
    REQUIRE_NOTHROW(trainer.SaveCGResult((c.directory/"direct.mtp").string()));
}

TEST_CASE("CG preserves initial nonfinite models and rejects unknown groups", "[training][nonlinear-cg]") {
    const bool invalid_coefficients=GENERATE(false,true);
    Case c; TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential); c.preserve();
    if(invalid_coefficients) c.potential.coefficients[0]=std::numeric_limits<double>::infinity();
    else { GPSettings settings; settings.Modify("trainable_groups","absent"); trainer.ApplySettings(settings); }
    REQUIRE_THROWS(trainer.TrainCG()); REQUIRE(trainer.cg_termination_type == 0);
    REQUIRE(c.potential.evaluations == 0); REQUIRE(c.saved() == "previous successful model\n");
    if(invalid_coefficients) REQUIRE(std::isinf(c.potential.coefficients[0]));
}

TEST_CASE("CG publication cannot reuse completion after another model or optimizer", "[training][nonlinear-cg]") {
    const int change=GENERATE(0,1,2,3);
    Case c; TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential);
    REQUIRE_NOTHROW(trainer.TrainCG()); const auto saved=c.saved();
    Potential replacement;
    if(change==0) trainer.init(c.data,c.neighbors,&replacement);
    if(change==1) { Eigen::Vector2d point{2.,.2}; trainer.SetOptimizerCoeffs(point.data()); }
    if(change==2) REQUIRE_NOTHROW(trainer.TrainNonLinearBFGS(1));
    if(change==3) trainer.init(c.data,c.neighbors,&c.potential);
    REQUIRE(trainer.cg_termination_type == 0);
    REQUIRE_THROWS(trainer.SaveCGResult((c.directory/"fitted.mtp").string()));
    REQUIRE(c.saved() == saved);
    REQUIRE_NOTHROW(trainer.TrainCG());
    REQUIRE_NOTHROW(trainer.SaveCGResult((c.directory/"retried.mtp").string()));
}

TEST_CASE("CG rejects a potential whose coefficient dimension changed after initialization", "[training][nonlinear-cg]") {
    Case c; TrainNonlinear trainer(c.settings()); trainer.init(c.data,c.neighbors,&c.potential);
    REQUIRE_NOTHROW(trainer.TrainCG()); const auto saved=c.saved();
    if(mpi.rank==mpi.size-1) c.potential.coefficient_count=3;
    REQUIRE_THROWS(trainer.SaveCGResult((c.directory/"fitted.mtp").string()));
    REQUIRE_THROWS(trainer.TrainCG()); REQUIRE_THROWS(trainer.TrainNonLinearCG(1));
    REQUIRE(c.saved() == saved);
}
