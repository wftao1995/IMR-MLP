#include <catch2/catch_all.hpp>
#include "train/train_nonlinear_potential.h"

TEST_CASE("Generic nonlinear options reject explicitly unsupported fields atomically", "[training][nonlinear-options]") {
    const auto [key,value] = GENERATE(Catch::Generators::table<std::string,std::string>({
        {"weight_rad_w0","1e-8"},{"weight_rad_w1","0"},{"weight_rad_w2","0"},
        {"radial_reg_type","0"},{"rand_low","-0.1"},{"rand_up","0.1"},
        {"lfit_fastalgo","true"},{"popsize","20"}}));
    const bool empty = GENERATE(false,true);
    GPSettings invalid; invalid.Modify(key, empty ? "" : value); invalid.Modify("bfgs_m","1");
    REQUIRE_THROWS_WITH(TrainNonlinear(invalid),Catch::Matchers::ContainsSubstring(key));
    TrainNonlinear trainer;
    const auto before=trainer.CurrentSettings();
    REQUIRE_THROWS_WITH(trainer.ApplySettings(invalid),Catch::Matchers::ContainsSubstring(key));
    REQUIRE(trainer.CurrentSettings().settings == before.settings);
}

TEST_CASE("Unsupported generic bindings guard base settings calls and omit snapshots", "[training][nonlinear-options]") {
    const std::string key=GENERATE("weight_rad_w0","weight_rad_w1","weight_rad_w2","radial_reg_type",
                                   "rand_low","rand_up","lfit_fastalgo","popsize");
    TrainNonlinear trainer;
    REQUIRE(trainer.CurrentSettings().count(key) == 0);
    ConnectSettings &base=trainer;
    GPSettings invalid; invalid.Modify(key,"0");
    REQUIRE_THROWS_WITH(base.ApplySettings(invalid),Catch::Matchers::ContainsSubstring(key));
    REQUIRE_NOTHROW(trainer.ApplySettings(invalid,{key}));
    REQUIRE_NOTHROW(base.ApplySettings(invalid,{key}));
}

TEST_CASE("Generic nonlinear settings snapshots preserve supported regularization and defaults", "[training][nonlinear-options]") {
    GPSettings settings;
    settings.Modify("weight_L1","0.02"); settings.Modify("weight_L2","0.03"); settings.Modify("penalty_weight","0.04");
    TrainNonlinear trainer(settings);
    const auto snapshot=trainer.CurrentSettings();
    REQUIRE_NOTHROW(trainer.ApplySettings(snapshot));
    TrainNonlinear roundtrip(snapshot);
    REQUIRE(roundtrip.CurrentSettings().settings == snapshot.settings);
    REQUIRE(std::stod(snapshot["weight_L1"]) == Catch::Approx(.02));
    REQUIRE(std::stod(snapshot["weight_L2"]) == Catch::Approx(.03));
    REQUIRE(std::stod(snapshot["penalty_weight"]) == Catch::Approx(.04));
    GPSettings legacy; legacy.Modify("weight_reg","0.05");
    REQUIRE_NOTHROW(trainer.ApplySettings(legacy));
    REQUIRE(std::stod(trainer.CurrentSettings()["weight_L2"]) == Catch::Approx(.05));
}
