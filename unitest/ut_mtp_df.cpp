#include <catch2/catch_all.hpp>
#include "train/train_mtp.h"

TEST_CASE("DF constraints retain norm layout and optionally append pairwise orthogonality",
          "[training][mtp][df]") {
    Potentials::PotMTP potential("unitest/test-data/portable-mtp/les.mtp");
    REQUIRE(potential.rb_size == 8);
    REQUIRE(potential.radial_func_count == 3);
    REQUIRE(potential.species_count == 2);
    const bool orthogonal = GENERATE(false, true);
    GPSettings settings;
    if (orthogonal) settings.Modify("df_orthogonal_constraints", "true");
    Trainer_MTP trainer(settings);
    trainer.p_mtp = &potential;
    REQUIRE(trainer.df_orthogonal_constraints == orthogonal);
    REQUIRE(trainer.DFConstraintCount() == (orthogonal ? 24 : 12));
    alglib::real_1d_array bounds, x, values;
    trainer.DFConstraintBounds(bounds);
    REQUIRE(bounds.length() == trainer.DFConstraintCount());
    for (int i = 0; i < bounds.length(); ++i)
        REQUIRE(bounds[i] == (i < 12 ? 1.0 : 0.0));
    x.setlength(potential.radial_coeff_size);
    for (int i = 0; i < x.length(); ++i) x[i] = 0.0;
    for (int pair = 0; pair < 4; ++pair)
        for (int k = 0; k < 3; ++k) x[(pair * 3 + k) * 8 + k] = 1.0;
    values.setlength(trainer.DFConstraintCount() + 1);
    values[0] = 123.;
    trainer.EvaluateDFConstraints(x, values);
    REQUIRE(values[0] == 123.);
    for (int i = 0; i < bounds.length(); ++i) REQUIRE(values[i + 1] == bounds[i]);

    SECTION("norm rows use every ordered species pair and radial channel") {
        for (int pair = 0; pair < 4; ++pair)
            for (int k = 0; k < 3; ++k) x[(pair * 3 + k) * 8 + k] = pair * 3 + k + 1.;
        trainer.EvaluateDFConstraints(x, values);
        for (int i = 0; i < 12; ++i) REQUIRE(values[i + 1] == (i + 1.) * (i + 1.));
        for (int i = 13; i < values.length(); ++i) REQUIRE(values[i] == 0.);
    }
    SECTION("duplicate channels violate only their pair's appended dot product") {
        x[(2 * 3 + 1) * 8 + 1] = 0.;
        x[(2 * 3 + 1) * 8] = 1.;
        trainer.EvaluateDFConstraints(x, values);
        for (int i = 1; i <= 12; ++i) REQUIRE(values[i] == 1.);
        if (orthogonal)
            for (int i = 13; i < values.length(); ++i) REQUIRE(values[i] == (i == 19 ? 1. : 0.));
    }
    SECTION("constraint evaluation reads x without changing model coefficients") {
        const std::vector<double> before(potential.radial_coeffs,
                                        potential.radial_coeffs + potential.radial_coeff_size);
        trainer.EvaluateDFConstraints(x, values);
        REQUIRE(std::vector<double>(potential.radial_coeffs,
                potential.radial_coeffs + potential.radial_coeff_size) == before);
        for (int pair = 0; pair < 4; ++pair)
            for (int k = 0; k < 3; ++k) REQUIRE(x[(pair * 3 + k) * 8 + k] == 1.);
    }
    SECTION("incorrect callback sizes are rejected") {
        values.setlength(trainer.DFConstraintCount());
        REQUIRE_THROWS_WITH(trainer.EvaluateDFConstraints(x, values),
                            Catch::Matchers::ContainsSubstring("size mismatch"));
    }
}

TEST_CASE("DF inner refitting preserves radial proposals and restores projection settings",
          "[training][mtp][df]") {
    const bool fail = GENERATE(false, true);
    Potentials::PotMTP potential("unitest/test-data/portable-mtp/les.mtp");
    Atoms atom;
    atom.cell_ = 20. * Eigen::Matrix3d::Identity();
    atom.pos_ = {{2., 2., 2.}, {4.2, 2.3, 2.1}};
    atom.chemmap_ = {0, 1};
    potential.getEFS(atom);
    std::vector<Atoms> atoms{atom};
    std::vector<Neighborhoods> neighborhoods;
    neighborhoods.emplace_back(atom, potential.cutoff);
    GPSettings settings;
    settings.Modify("do_orthogonalization", "true");
    settings.Modify("weight_L2", "0.01");
    Trainer_MTP trainer(settings);
    trainer.init(atoms, neighborhoods, &potential);
    const std::vector<double> before(potential.radial_coeffs,
                                     potential.radial_coeffs + potential.radial_coeff_size);
    if (fail) {
        GPSettings invalid;
        invalid.Modify("linear_precondition", "invalid");
        trainer.ApplySettings(invalid);
        REQUIRE_THROWS_WITH(trainer.FitDFLinear(),
                            Catch::Matchers::ContainsSubstring("Unknown linear_precondition"));
        GPSettings valid;
        valid.Modify("linear_precondition", "none");
        trainer.ApplySettings(valid);
    } else {
        trainer.FitDFLinear();
    }
    REQUIRE(std::vector<double>(potential.radial_coeffs,
            potential.radial_coeffs + potential.radial_coeff_size) == before);
    trainer.TrainLinear();
    REQUIRE(std::vector<double>(potential.radial_coeffs,
            potential.radial_coeffs + potential.radial_coeff_size) != before);
}
