#include <catch2/catch_all.hpp>
#include "train/train_mtp.h"
#include <filesystem>
#include <unistd.h>

namespace {
using namespace Potentials;
Atoms legacy_geometry() {
    Atoms a;a.cell_=14.*Eigen::Matrix3d::Identity();a.pbc_={false,false,false};
    a.pos_={{2.,2.,2.},{4.1,2.3,2.2},{2.4,4.3,2.5}};a.chemmap_={0,1,0};return a;
}
Eigen::VectorXd coefficients(PotMTP &p) {Eigen::VectorXd x(p.get_NL_coeffs_count());p.nl_get_coeffs(x.data());return x;}
struct Directory {
    std::filesystem::path original=std::filesystem::current_path(),path;
    Directory() {char name[]="/tmp/imrmlp_legacy_layout_XXXXXX";const auto *p=mkdtemp(name);
        if(!p)throw std::runtime_error("Cannot create legacy layout directory");path=p;std::filesystem::current_path(path);}
    ~Directory(){std::filesystem::current_path(original);std::filesystem::remove_all(path);}
};
struct LadderOwnership {
    Trainer_MTP &trainer;PotMTP *original;
    ~LadderOwnership(){if(trainer.p_ladder==original)std::swap(trainer.p_mtp,trainer.p_ladder);}
};
GPSettings settings() {GPSettings s;s.Modify("do_orthogonalization","false");s.Modify("bfgs_maxiteration","0");return s;}
}

TEST_CASE("Legacy trainer rejects appended parameter layouts before touching cached buffers", "[training][append][legacy-layout]") {
    const int frozen=GENERATE(0,1,2,3);PotMTP potential("unitest/test-data/portable-mtp/les.mtp");
    potential.set_element_map({{0,"Al"},{1,"Ni"}});auto atom=legacy_geometry();potential.getEFS(atom);
    std::vector<Atoms> atoms{atom};std::vector<Neighborhoods> neighbors;neighbors.emplace_back(atom,potential.cutoff);
    Directory directory;auto fit=settings();fit.Modify("weight_L1","1e-5");fit.Modify("weight_L2",".004");Trainer_MTP trainer(fit);
    trainer.fix_radial_coeffs=frozen==1;trainer.fix_linear_coeffs=frozen==2;trainer.fix_species_coeffs=frozen==3;
    trainer.init(atoms,neighbors,&potential);
    alglib::real_1d_array gradient;gradient.setlength(trainer.get_relaxed_coeffs_count());double value=0.;
    REQUIRE_NOTHROW(trainer.AccumulateGard(value,gradient));
    REQUIRE_NOTHROW(trainer.TrainNonLinearBFGS(1));
    REQUIRE(potential.reg_L1_vector.size()==size_t(potential.get_lin_coeff_size()));
    REQUIRE(potential.reg_L1_vector.front()==1e-5);
    REQUIRE(potential.reg_L2_vector.front()>0.);
    potential.append_species({0},{"Ge"});const auto appended=coefficients(potential);
    REQUIRE(potential.reg_L1_vector==std::vector<double>(potential.reg_coeff_size,0.));
    REQUIRE(potential.reg_L2_vector==std::vector<double>(potential.reg_coeff_size,0.));
    const auto stale=Catch::Matchers::ContainsSubstring("reinitialize the trainer after appending species");
    REQUIRE_THROWS_WITH(trainer.AccumulateGard(value,gradient),stale);
    REQUIRE_THROWS_WITH(trainer.AccumulateLoss(value),stale);
    REQUIRE_THROWS_WITH(trainer.AddLossGard(0,value),stale);
    REQUIRE_THROWS_WITH(trainer.AddLoss(0,value),stale);
    REQUIRE_THROWS_WITH(trainer.TrainLinear(),stale);
    REQUIRE_THROWS_WITH(trainer.TrainLinearAlg(),stale);
    REQUIRE_THROWS_WITH(trainer.TrainLinearFast(),stale);
    REQUIRE_THROWS_WITH(trainer.TrainBFGS(),stale);
    REQUIRE_THROWS_WITH(trainer.TrainNonLinearBFGS(0),stale);
    REQUIRE_THROWS_WITH(trainer.TrainCG(),stale);
    REQUIRE_THROWS_WITH(trainer.TrainSGD(),stale);
    REQUIRE_THROWS_WITH(trainer.save_current("stale.mtp"),stale);
    REQUIRE_FALSE(std::filesystem::exists("stale.mtp"));REQUIRE(coefficients(potential)==appended);
    trainer.init(atoms,neighbors,&potential);gradient.setlength(trainer.get_relaxed_coeffs_count());
    REQUIRE(potential.reg_L1_vector==std::vector<double>(potential.get_lin_coeff_size(),1e-5));
    REQUIRE(potential.reg_L2_vector==std::vector<double>(potential.get_lin_coeff_size(),.004));
    REQUIRE_NOTHROW(trainer.AccumulateGard(value,gradient));REQUIRE(std::isfinite(value));
}

TEST_CASE("Legacy ladder snapshots retain the training view across native serialization swaps", "[training][append][legacy-layout][ladder]") {
    const auto input=std::filesystem::absolute("unitest/test-data/portable-mtp/les.mtp");Directory directory;
    PotMTP potential(input.string());auto atom=legacy_geometry();potential.getEFS(atom);
    std::vector<Atoms> atoms{atom};std::vector<Neighborhoods> neighbors;neighbors.emplace_back(atom,potential.cutoff);
    Trainer_MTP trainer(settings());LadderOwnership ownership{trainer,&potential};
    trainer.do_ladder=true;trainer.ladder_cutoff=10;
    trainer.init(atoms,neighbors,&potential);
    double before=0.,after=0.;trainer.AccumulateLoss(before);
    REQUIRE_NOTHROW(trainer.save_current("ladder.mtp"));
    REQUIRE_NOTHROW(trainer.AccumulateLoss(after));REQUIRE(after==Catch::Approx(before).margin(1e-12));
    // The final swap intentionally exposes the complete output model; fresh
    // matrix assembly is supported, while cached training entry points are not.
    trainer.ladder_fin();REQUIRE(trainer.p_mtp==&potential);
    REQUIRE_THROWS_WITH(trainer.AccumulateLoss(after),Catch::Matchers::ContainsSubstring("reinitialize"));
    // Match the application lifecycle: the trainer now owns the temporary copy.
}

TEST_CASE("Legacy ladder detects append on the original model before copying a cached training view", "[training][append][legacy-layout][ladder]") {
    const auto input=std::filesystem::absolute("unitest/test-data/portable-mtp/les.mtp");Directory directory;
    PotMTP potential(input.string());potential.set_element_map({{0,"Al"},{1,"Ni"}});
    auto atom=legacy_geometry();potential.getEFS(atom);std::vector<Atoms> atoms{atom};
    std::vector<Neighborhoods> neighbors;neighbors.emplace_back(atom,potential.cutoff);
    Trainer_MTP trainer(settings());LadderOwnership ownership{trainer,&potential};
    trainer.do_ladder=true;trainer.ladder_cutoff=10;trainer.init(atoms,neighbors,&potential);
    potential.append_species({0},{"Ge"});const auto appended=coefficients(potential);double value=0.;
    REQUIRE_THROWS_WITH(trainer.AccumulateLoss(value),Catch::Matchers::ContainsSubstring("reinitialize"));
    REQUIRE_THROWS_WITH(trainer.save_current("stale-ladder.mtp"),Catch::Matchers::ContainsSubstring("reinitialize"));
    REQUIRE_FALSE(std::filesystem::exists("stale-ladder.mtp"));REQUIRE(coefficients(potential)==appended);
}
