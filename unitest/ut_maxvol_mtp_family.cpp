#include <catch2/catch_all.hpp>
#include "maxvol/active_selector.h"
#include "maxvol_mtp_fixture.h"
#include <cstdlib>

namespace {
struct ModelFiles {
    std::filesystem::path dir;
    std::vector<MaxVolMTPCase> cases;
    ModelFiles() {
        auto pattern = (std::filesystem::temp_directory_path() / "maxvol-mtp-XXXXXX").string();
        const char *path = mkdtemp(pattern.data());
        if (!path) throw std::runtime_error("Cannot create model fixture directory");
        dir = path;
        cases = make_maxvol_mtp_models(dir);
    }
    ~ModelFiles() { std::filesystem::remove_all(dir); }
};
}

TEST_CASE("MaxVol MTP-family gradients include every model parameter", "[maxvol][mtp_family]")
{
    ModelFiles files;
    for (auto &test : files.cases) {
        CAPTURE(test.name);
        Potentials::potential_wrapper wrapper(test.settings);
        auto &pot = *wrapper.pot;
        REQUIRE(pot.species_count == 2);
        PotEFSBuilder builder(pot); // Default linear_only must not slice derived models.
        REQUIRE(builder.coeff_count() == pot.get_NL_coeffs_count());
        REQUIRE(builder.coeff_count() > 0);
        auto cfg = maxvol_mtp_atoms();
        BMatrixBuilder::RowMatrix rows;
        builder.build_rows(cfg, rows, 1.0, 0.0);
        REQUIRE(rows.rows() == 1);
        REQUIRE(rows.allFinite());
        std::vector<double> coeffs(pot.get_NL_coeffs_count());
        pot.nl_get_coeffs(coeffs.data());
        for (int i = 0; i < static_cast<int>(coeffs.size()); ++i) {
            auto plus = coeffs, minus = coeffs;
            plus[i] += 1e-6;
            minus[i] -= 1e-6;
            pot.nl_set_coeffs(plus.data());
            auto a = maxvol_mtp_atoms(); pot.getE(a);
            pot.nl_set_coeffs(minus.data());
            auto b = maxvol_mtp_atoms(); pot.getE(b);
            CAPTURE(i);
            REQUIRE(rows(0,i) == Catch::Approx((a.energy_ - b.energy_) / 2e-6).margin(2e-6));
        }
        pot.nl_set_coeffs(coeffs.data());

        builder.w_force = .1;
        builder.w_stress = .01;
        rows.resize(0,0);
        builder.build_rows(cfg, rows, 1.0, 0.0);
        REQUIRE(rows.rows() == 19);
        REQUIRE(rows.allFinite());
        // Force/virial gradients exercise the hmtp adapter and global electrostatics.
        for (int i : {0, int(coeffs.size()/2), int(coeffs.size()-1)}) {
            auto plus = coeffs, minus = coeffs;
            plus[i] += 1e-6; minus[i] -= 1e-6;
            pot.nl_set_coeffs(plus.data()); auto a = maxvol_mtp_atoms(); pot.getEFS(a);
            pot.nl_set_coeffs(minus.data()); auto b = maxvol_mtp_atoms(); pot.getEFS(b);
            REQUIRE(rows(1,i) == Catch::Approx(.1*(a.forces_[0][0]-b.forces_[0][0])/2e-6).margin(3e-6));
            REQUIRE(rows(10,i) == Catch::Approx(.01*(a.stress_(0,0)-b.stress_(0,0))/2e-6).margin(3e-6));
        }
        pot.nl_set_coeffs(coeffs.data());

        ActiveSelector selected(new PotEFSBuilder(pot));
        selected.select_mode = "full";
        selected.init_mode = "data";
        selected.add_candidate(cfg);
        REQUIRE(selected.select() > 0);
        const auto state = (files.dir / (test.name + ".sel")).string();
        selected.save_state(state);
        ActiveSelector restored;
        restored.load_state(state);
        restored.init_builder(pot);
        REQUIRE(restored.grade(cfg) == Catch::Approx(selected.grade(cfg)).margin(1e-12));

        GPSettings factory_settings;
        for (const auto &[key, value] : test.settings.settings)
            factory_settings.Modify("pot:" + key, value);
        if (dynamic_cast<Potentials::PotMTP *>(&pot)) {
            auto descriptor = make_builder("mtm", test.path, test.pottype, factory_settings,
                                           "nbh", 1., 0., 0., 0., false);
            REQUIRE(descriptor != nullptr);
            BMatrixBuilder::RowMatrix descriptor_rows;
            REQUIRE(descriptor->build_rows(cfg, descriptor_rows, 1., 0.) == cfg.size());
            REQUIRE(descriptor_rows.allFinite());
        } else {
            REQUIRE_THROWS(make_builder("mtm", test.path, test.pottype, factory_settings,
                                        "nbh", 1., 0., 0., 0., false));
        }

        if (test.pottype == "mtp_les" ||
            test.pottype == "hmtp" ||
            test.name == "mtp_m_les" || test.name == "mtp_m_sog") {
            builder.w_site = 1.;
            REQUIRE_THROWS(builder.build_rows(cfg, rows, 1., 0.));
        } else {
            builder.w_site = 1.;
            rows.resize(0, 0);
            builder.build_rows(cfg, rows, 1., 0.);
            REQUIRE(rows.rows() == 22);
            // Summed site-energy gradients reproduce the structure energy row.
            REQUIRE((rows.bottomRows(3).colwise().sum() - rows.row(0)).norm() < 1e-10);
        }
    }
}

TEST_CASE("MaxVol rejects incompatible restored MTP-family builders", "[maxvol][mtp_family]")
{
    ModelFiles files;
    Potentials::PotMTP plain((files.dir / "base.mtp").string());
    GPSettings settings;
    auto descriptor = make_builder("mtm", (files.dir / "base.mtp").string(), "mtp", settings,
                                   "nbh", 1., 0., 0., 0., false);
    ActiveSelector source(descriptor.release());
    source.select_mode = "full";
    source.init_mode = "data";
    auto cfg = maxvol_mtp_atoms();
    source.add_candidate(cfg);
    REQUIRE(source.select() > 0);
    const auto state = (files.dir / "descriptor.sel").string();
    source.save_state(state);
    ActiveSelector restored;
    restored.load_state(state);
    Potentials::potential_wrapper composite(files.cases.front().settings);
    REQUIRE_THROWS_WITH(restored.init_builder(*composite.pot),
                       Catch::Matchers::ContainsSubstring("pot_efs"));
    // A failed reconstruction must leave the state usable with the right model.
    REQUIRE_NOTHROW(restored.init_builder(plain));
    REQUIRE(restored.grade(cfg) == Catch::Approx(source.grade(cfg)).margin(1e-12));

    ActiveSelector efs(new PotEFSBuilder(plain));
    efs.select_mode = "full";
    efs.init_mode = "data";
    efs.add_candidate(cfg);
    REQUIRE(efs.select() > 0);
    efs.save_state(state);
    restored.load_state(state);
    REQUIRE_THROWS_WITH(restored.init_builder(*composite.pot),
                       Catch::Matchers::ContainsSubstring("coefficient count"));
    REQUIRE_NOTHROW(restored.init_builder(plain));
    REQUIRE(restored.grade(cfg) == Catch::Approx(efs.grade(cfg)).margin(1e-12));
}
