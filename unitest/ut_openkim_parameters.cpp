#include <catch2/catch_all.hpp>
#include "potentials/potential_openkim.h"
#include <cmath>

#ifdef USE_KIM
namespace {
bool model_available(const std::string &name) {
    KIM::Model *model = nullptr;
    int accepted = 0;
    const int error = KIM::Model::Create(KIM::NUMBERING::zeroBased, KIM::LENGTH_UNIT::A,
        KIM::ENERGY_UNIT::eV, KIM::CHARGE_UNIT::e, KIM::TEMPERATURE_UNIT::K,
        KIM::TIME_UNIT::ps, name, &accepted, &model);
    if (model) KIM::Model::Destroy(&model);
    return !error && accepted;
}
Atoms ar_pair(double distance) {
    Atoms a;
    a.pbc_ = {false, false, false};
    a.pos_ = {{0, 0, 0}, {distance, 0, 0}};
    a.chemmap_ = {0, 0};
    return a;
}
double lj(double distance, double epsilon, double sigma) {
    const double x = std::pow(sigma / distance, 6);
    return 4 * epsilon * (x * x - x);
}
}

TEST_CASE("OpenKIM parameter settings refresh values and neighbor cutoff", "[openkim][parameters]") {
    const std::string model = "ex_model_Ar_P_LJ";
    if (!model_available(model)) SKIP("Installed KIM example ex_model_Ar_P_LJ is unavailable");
    Potentials::PotKIM pot(model, {{0, "Ar"}});
    REQUIRE(std::stod(pot.CurrentSettings()["kim_parameter:epsilon"]) == Catch::Approx(.0104));
    GPSettings settings;
    settings.Modify("kim_parameter:epsilon", ".02");
    settings.Modify("kim_parameter:sigma", "3");
    settings.Modify("kim_parameter:cutoff", "6");
    pot.ApplyParameterSettings(settings);
    REQUIRE(pot.get_cutoff() == Catch::Approx(6));
    REQUIRE(pot.cutoff == Catch::Approx(6)); REQUIRE(pot.cutoffsq == Catch::Approx(36));
    REQUIRE(std::stod(pot.CurrentSettings()["kim_parameter:sigma"]) == Catch::Approx(3));
    auto a = ar_pair(4), b = ar_pair(5);
    pot.getEFS(a); pot.getEFS(b);
    // Energy differences eliminate the example model's constant cutoff shift.
    REQUIRE(a.energy_ - b.energy_ == Catch::Approx(lj(4, .02, 3) - lj(5, .02, 3)).margin(1e-12));
    const double x = std::pow(3. / 4, 6);
    const double derivative = 24 * .02 * (x - 2 * x * x) / 4;
    REQUIRE(a.forces_[0][0] == Catch::Approx(derivative).margin(1e-12));
    REQUIRE(a.forces_[1][0] == Catch::Approx(-derivative).margin(1e-12));
    auto outside = ar_pair(6.5); pot.getEFS(outside);
    REQUIRE(outside.energy_ == 0); REQUIRE(outside.forces_[0].isZero());
    GPSettings wider; wider.Modify("kim_parameter:cutoff", "7");
    pot.ApplyParameterSettings(wider); pot.getEFS(outside);
    REQUIRE(std::abs(outside.energy_) > 1e-8);

    for (const auto &[key, value] : std::vector<std::pair<std::string, std::string>>{
            {"epsilon", ".03 .04"}, {"epsilon", "nan"}, {"epsilon", ""}, {"unknown", "1"}}) {
        const auto old = pot.CurrentSettings();
        GPSettings bad; bad.Modify("kim_parameter:" + key, value);
        REQUIRE_THROWS(pot.ApplyParameterSettings(bad));
        REQUIRE(pot.CurrentSettings()["kim_parameter:epsilon"] == old["kim_parameter:epsilon"]);
        auto unchanged = ar_pair(4); pot.getEFS(unchanged);
        REQUIRE(unchanged.forces_[0][0] == Catch::Approx(derivative).margin(1e-12));
    }
}

TEST_CASE("OpenKIM integer parameter and array defaults use stable connections", "[openkim][parameters]") {
    const std::string model = "LennardJones612_UniversalShifted__MO_959249795837_003";
    if (!model_available(model)) SKIP("Installed universal LJ model is unavailable");
    Potentials::PotKIM pot(model, {{0, "Ar"}});
    REQUIRE(pot.CurrentSettings()["kim_parameter:shift"] == "1");
    const auto before = pot.CurrentSettings()["kim_parameter:epsilons"];
    auto a = ar_pair(4); pot.getEFS(a);
    const double shifted = a.energy_;
    GPSettings settings; settings.Modify("kim_parameter:shift", "0");
    pot.ApplyParameterSettings(settings); pot.getEFS(a);
    REQUIRE(pot.CurrentSettings()["kim_parameter:shift"] == "0");
    REQUIRE(pot.CurrentSettings()["kim_parameter:epsilons"] == before);
    REQUIRE(std::abs(a.energy_ - shifted) > 1e-10);
    GPSettings bad; bad.Modify("kim_parameter:shift", "0.5");
    REQUIRE_THROWS(pot.ApplyParameterSettings(bad));
    REQUIRE(pot.CurrentSettings()["kim_parameter:shift"] == "0");
    settings.Modify("kim_parameter:shift", "1");
    pot.ApplyParameterSettings(settings); pot.getEFS(a);
    REQUIRE(a.energy_ == Catch::Approx(shifted).margin(1e-12));
}
#else
TEST_CASE("OpenKIM parameter integration requires KIM", "[openkim][parameters]") {
    SKIP("Build with USE_KIM to test parameter integration");
}
#endif
