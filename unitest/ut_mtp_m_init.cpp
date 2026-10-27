#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_m.h"

#include <fstream>
#include <sstream>
#include <vector>

namespace {
std::string mtp_m_init_template()
{
    std::ifstream input("unitest/test-data/portable-mtp/les.mtp");
    REQUIRE(input.good());
    std::string source((std::istreambuf_iterator<char>(input)), {});
    const auto coefficients = source.find("species_coeffs");
    REQUIRE(coefficients != std::string::npos);
    source.resize(coefficients);
    return source;
}
}

TEMPLATE_TEST_CASE("MTP-M initialization preserves loaded defaults and explicit coefficients",
                   "[mtp_m][init_coeffs]", Potentials::m_model_mtpq2,
                   Potentials::m_model_nn1, Potentials::m_model_nn2)
{
    Potentials::PotMTP_M<TestType> potential;
    GPSettings settings;
    settings.Modify("n_input", "2");
    settings.Modify("n_neuron", "3");
    settings.Modify("n_neuron2", "2");
    potential.model.init(settings);
    const std::string fixture = mtp_m_init_template();
    std::istringstream defaults(fixture);
    potential.Load(defaults);
    REQUIRE_FALSE(potential.inited());
    REQUIRE_FALSE(static_cast<const Potentials::PotMTP&>(potential).inited());
    REQUIRE_FALSE(static_cast<Potentials::AnyPotential&>(potential).inited());
    for (int i = potential.radial_coeff_size; i < potential.reg_coeff_size; ++i)
        REQUIRE(potential.regression_coeffs[i] == 1e-3);

    SECTION("omitted coefficient sections keep Load defaults") {}
    SECTION("explicit coefficient sections keep every supplied value") {
        std::ostringstream body;
        body << fixture;
        const auto append = [&](const char* name, int count, double value) {
            body << name << " = {";
            for (int i = 0; i < count; ++i)
                body << (i ? ", " : "") << value * (i + 1);
            body << "}\n";
        };
        append("species_coeffs", potential.species_coeff_size, .125);
        append("moment_coeffs", potential.moment_coeff_size, -.0625);
        append("model_coeffs", potential.model_coeff_size, .03125);
        std::istringstream explicit_coefficients(body.str());
        potential.Load(explicit_coefficients);
        REQUIRE(potential.inited());
        REQUIRE(static_cast<const Potentials::PotMTP&>(potential).inited());
        REQUIRE(static_cast<Potentials::AnyPotential&>(potential).inited());
        for (int i = 0; i < potential.species_coeff_size; ++i)
            REQUIRE(potential.species_coeffs[i] == .125 * (i + 1));
        for (int i = 0; i < potential.moment_coeff_size; ++i)
            REQUIRE(potential.moment_coeffs[i] == -.0625 * (i + 1));
        for (int i = 0; i < potential.model_coeff_size; ++i)
            REQUIRE(potential.model_coeffs[i] == .03125 * (i + 1));
    }

    const std::vector<double> before(potential.regression_coeffs,
                                     potential.regression_coeffs + potential.reg_coeff_size);
    double* const allocation = potential.regression_coeffs;
    // Simulate a head awaiting reconnection, without changing owned coefficients.
    potential.model.coeffs = nullptr;
    if constexpr (std::is_same_v<TestType, Potentials::m_model_nn1>) {
        potential.model.w_input_map.reset();
        potential.model.w_hidden_map.reset();
        potential.model.b_hidden_map.reset();
    }
    potential.set_empty();
    Potentials::AnyPotential& api = potential;
    REQUIRE_FALSE(api.inited());
    REQUIRE_FALSE(static_cast<const Potentials::PotMTP&>(potential).inited());
    for (int repeat = 0; repeat < 2; ++repeat) {
        api.init_coeffs();
        REQUIRE(api.inited());
        REQUIRE(potential.inited());
        REQUIRE(static_cast<const Potentials::PotMTP&>(potential).inited());
        REQUIRE(potential.regression_coeffs == allocation);
        REQUIRE(potential.model.coeffs == potential.model_coeffs);
        REQUIRE(std::vector<double>(potential.regression_coeffs,
                    potential.regression_coeffs + potential.reg_coeff_size) == before);
        if constexpr (std::is_same_v<TestType, Potentials::m_model_nn1>) {
            REQUIRE(potential.model.w_input_map->data() == potential.model_coeffs);
            REQUIRE(potential.model.w_hidden_map->data() ==
                    potential.model_coeffs + potential.model.n_input * potential.model.n_neuron);
            REQUIRE(potential.model.b_hidden_map->data() ==
                    potential.model_coeffs + potential.model.n_input * potential.model.n_neuron
                    + potential.model.n_neuron);
        }
    }
    api.set_empty();
    REQUIRE_FALSE(api.inited());
    REQUIRE_FALSE(static_cast<const Potentials::PotMTP&>(potential).inited());
    api.set_inited();
    REQUIRE(api.inited());
    REQUIRE(static_cast<const Potentials::PotMTP&>(potential).inited());
    REQUIRE(potential.regression_coeffs==allocation);
    REQUIRE(std::vector<double>(potential.regression_coeffs,
        potential.regression_coeffs+potential.reg_coeff_size)==before);
}

TEST_CASE("MTP-M initialization rejects an unloaded model", "[mtp_m][init_coeffs]")
{
    Potentials::PotMTP_M<Potentials::m_model_mtpq2> potential;
    Potentials::AnyPotential& api = potential;
    REQUIRE_FALSE(static_cast<const Potentials::PotMTP&>(potential).inited());
    REQUIRE_THROWS_WITH(api.set_inited(), Catch::Matchers::ContainsSubstring("no loaded model"));
    REQUIRE_THROWS_WITH(api.init_coeffs(), Catch::Matchers::ContainsSubstring("no loaded model"));
    REQUIRE_FALSE(api.inited());
    REQUIRE(potential.regression_coeffs == nullptr);
    REQUIRE(potential.model.coeffs == nullptr);
}
