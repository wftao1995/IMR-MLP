#include <catch2/catch_all.hpp>
#include "train/train_nonlinear_potential.h"
#include "unitest/les_curvature_fixture.h"
#include <filesystem>
#include <fstream>
#include <random>
#include <unistd.h>

using namespace Potentials;
namespace {
std::vector<double> stage_coefficients(const AnyPotential& potential) {
    std::vector<double> values(potential.get_NL_coeffs_count());potential.nl_get_coeffs(values.data());return values;
}
template<class Head> class StageMTPM : public PotMTP_M<Head> {
public:
    struct Snapshot {std::string path;std::vector<double> coefficients;};
    std::vector<Snapshot> snapshots;
    bool fail_warmup=false;
    StageMTPM() {
        LESCurvature::initialize(*this,true);
        std::fill_n(this->les_species,this->species_count,0.);
        std::fill_n(this->les_moment,this->alpha_scalar_moments,0.);
        this->set_inited();
    }
    void getEFS(Atoms& atom,AnyNeighborhoods&) override {
        const auto values=stage_coefficients(*this);atom.resetEFS();
        atom.energy_=std::accumulate(values.begin(),values.end(),0.)/values.size();
    }
    void AccumulateEFSCombinationGard(Atoms&,AnyNeighborhoods&,double de,double* out) override {
        for(int i=0;i<this->get_NL_coeffs_count();++i) out[i]+=de/this->get_NL_coeffs_count();
    }
    void AccumulateEFSCombinationGard(Atoms& atom,AnyNeighborhoods& n,double de,
        Eigen::Matrix<double,-1,3>&,Eigen::Matrix3d&,
        Eigen::Matrix<double,-1,3,Eigen::RowMajor>&,double* out) override {
        AccumulateEFSCombinationGard(atom,n,de,out);
    }
    void AddPenaltyGard(double,double&,double*) override {}
    void Save(const std::string& path) override {
        snapshots.push_back({path,stage_coefficients(*this)});
        if(fail_warmup && path.ends_with(".sog.snapshot")) {
            this->model_les.sog_weights[0]+=.03;this->model_les.sog_log_bandwidths[0]+=.01;
            this->lr_les=false;throw std::runtime_error("injected MTP-M SOG stage failure");
        }
        std::ofstream out(path);out<<"stage fixture\n";
    }
    const std::vector<double>& saved(const std::string& suffix) const {
        const auto found=std::find_if(snapshots.begin(),snapshots.end(),[&](const auto& x){return x.path.ends_with(suffix);});
        REQUIRE(found!=snapshots.end());return found->coefficients;
    }
};
struct StageData {
    std::filesystem::path directory=std::filesystem::temp_directory_path()/
        ("atoms-agent-mtpm-stages-"+std::to_string(getpid()));
    std::vector<Atoms> atoms;std::vector<Neighborhoods> neighbors;
    StageData(double cutoff) {
        std::filesystem::create_directories(directory);
        auto a=LESCurvature::geometry();a.energy_=2.;a.have_energy=true;
        atoms.push_back(a);neighbors.emplace_back(a,cutoff);
    }
    ~StageData() {std::error_code error;std::filesystem::remove_all(directory,error);}
    GPSettings settings(const std::string& schedule) const {
        GPSettings s;s.Modify("training_schedule",schedule);s.Modify("sr_stage_iterations","1");
        s.Modify("les_stage_iterations","1");s.Modify("joint_stage_iterations","1");
        s.Modify("energy_weight","1");s.Modify("force_weight","0");s.Modify("stress_weight","0");
        s.Modify("penalty_weight","0");s.Modify("parameter_scaling","none");s.Modify("save_every","100");
        s.Modify("bfgs_grad_tol","0");s.Modify("bfgs_conv_tol","0");s.Modify("bfgs_step_tol","0");
        s.Modify("les_zero_init_seed","731");s.Modify("save_to",(directory/"result.mtp").string());return s;
    }
};
}

TEMPLATE_TEST_CASE("Known MTP-M SOG heads share staged masks and fixed bandwidth warmup",
    "[training][nonlinear-schedule][mtpm][sog]",m_model_mtpli,m_model_mtpq2,m_model_mtpfs,m_model_nn1,m_model_nn2) {
    const std::string schedule=GENERATE("mtp_sog_from_scratch","mtp_sog_resume");
    StageMTPM<TestType> potential;StageData data(potential.cutoff);TrainNonlinear trainer(data.settings(schedule));
    trainer.init(data.atoms,data.neighbors,&potential);const auto initial=stage_coefficients(potential);
    trainer.TrainCG();REQUIRE(trainer.cg_termination_type>0);
    const auto& warm=potential.saved(".sog.snapshot");
    const auto& sr=schedule.ends_with("from_scratch") ? potential.saved(".sr.snapshot") : initial;
    for(const auto& group:potential.nl_parameter_groups()) for(int i=group.begin;i<group.end;++i) {
        if(group.name.starts_with("sr_")) REQUIRE(warm[i]==sr[i]);
        if(group.name=="sog_log_bandwidths") REQUIRE(warm[i]==Catch::Approx(initial[i]).margin(1e-12));
        if(schedule.ends_with("from_scratch") && (group.name=="les_species" || group.name=="les_moment"))
            REQUIRE(sr[i]==0.);
    }
    REQUIRE(std::filesystem::exists(data.directory/"result.mtp"));
}

TEMPLATE_TEST_CASE("MTP-M SOG stages seed only zero channels and roll back physical parameters",
    "[training][nonlinear-schedule][mtpm][rollback]",m_model_mtpli,m_model_mtpq2,m_model_mtpfs,m_model_nn1,m_model_nn2) {
    StageMTPM<TestType> potential;StageData data(potential.cutoff);auto settings=data.settings("mtp_sog_resume");
    settings.Modify("sr_stage_iterations","0");settings.Modify("les_stage_iterations","0");
    TrainNonlinear trainer(settings);trainer.init(data.atoms,data.neighbors,&potential);
    const auto initial=stage_coefficients(potential),weights=potential.model_les.sog_weights,widths=potential.model_les.sog_log_bandwidths;
    const auto before=trainer.CurrentSettings().settings;potential.fail_warmup=true;
    REQUIRE_THROWS_WITH(trainer.TrainCG(),Catch::Matchers::ContainsSubstring("injected MTP-M"));
    REQUIRE(potential.lr_les);REQUIRE(stage_coefficients(potential)==initial);
    REQUIRE(potential.model_les.sog_weights==weights);REQUIRE(potential.model_les.sog_log_bandwidths==widths);
    REQUIRE(trainer.CurrentSettings().settings==before);REQUIRE(trainer.cg_termination_type==0);
    std::mt19937 rng(731);std::uniform_real_distribution<double> draw(-.1,.1);
    std::vector<double> expected_moments(potential.alpha_scalar_moments),expected_species(potential.species_count);
    for(auto& value:expected_moments) value=draw(rng);for(auto& value:expected_species) value=draw(rng);
    const auto& warm=potential.saved(".sog.snapshot");
    for(const auto& group:potential.nl_parameter_groups()) {
        if(group.name=="les_species" || group.name=="les_moment") {
            const auto& expected=group.name=="les_species" ? expected_species : expected_moments;
            for(int i=0;i<int(expected.size());++i) REQUIRE(warm[group.begin+i]==expected[i]);
            for(int i=expected.size();i<group.end-group.begin;++i) REQUIRE(warm[group.begin+i]==initial[group.begin+i]);
        }
    }
}

TEST_CASE("MTP-M Coulomb LES schedules stay unsupported", "[training][nonlinear-schedule][mtpm]") {
    PotMTP_M<m_model_mtpq2> potential;LESCurvature::initialize(potential,false);
    StageData data(potential.cutoff);TrainNonlinear trainer(data.settings("mtp_les_resume"));
    trainer.init(data.atoms,data.neighbors,&potential);const auto before=stage_coefficients(potential);
    REQUIRE_THROWS_WITH(trainer.TrainCG(),Catch::Matchers::ContainsSubstring("Coulomb LES stages"));
    REQUIRE(stage_coefficients(potential)==before);
}
