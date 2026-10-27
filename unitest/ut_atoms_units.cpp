#include <catch2/catch_all.hpp>
#include "io/atoms_units.h"

TEST_CASE("Typed conversions preserve legacy factors and reject wrong dimensions", "[units]") {
    using namespace GPUnits;
    REQUIRE(units_factor("Hartree") == Constants::ha2ev);
    REQUIRE(units_factor("degree") == Constants::deg2reg);
    for (const char *unit : {"eV", "Angstrom", "angstrom", "A", "eV/A", "picosecond"})
        REQUIRE(units_factor(unit) == 1.);
    REQUIRE(conversion_factor(Quantity::Length, "Bohr", "A") == Catch::Approx(.529177210903));
    REQUIRE(conversion_factor(Quantity::Energy, "Hartree", "eV") == Constants::ha2ev);
    REQUIRE(conversion_factor(Quantity::Energy, "kJ mol^-1", "eV") == Catch::Approx(Constants::kj2ev / 6.02214076e23));
    REQUIRE(conversion_factor(Quantity::Time, "fs", "ns") == Catch::Approx(1e-6));
    REQUIRE(conversion_factor(Quantity::Mass, "kg", "amu") == Catch::Approx(1. / 1.66053906660e-27));
    REQUIRE(conversion_factor(Quantity::Force, "Hartree/Bohr", "eV/A") ==
        Catch::Approx(Constants::ha2ev / .529177210903));
    REQUIRE(conversion_factor(Quantity::Velocity, "m/s", "A/ps") == Catch::Approx(.01));
    REQUIRE(conversion_factor(Quantity::Pressure, "GPa", "bar") == Catch::Approx(1e4));
    REQUIRE(conversion_factor(Quantity::Hessian, "Hartree/Bohr^2", "eV/A^2") ==
        Catch::Approx(Constants::ha2ev / (.529177210903 * .529177210903)));
    REQUIRE(conversion_factor(Quantity::Length, "nm", "Bohr") *
            conversion_factor(Quantity::Length, "Bohr", "nm") == Catch::Approx(1.));
    REQUIRE_THROWS_AS(conversion_factor(Quantity::Length, "eV", "A"), std::invalid_argument);
    REQUIRE_THROWS_AS(conversion_factor(Quantity::Pressure, "eV", "bar"), std::invalid_argument);
    REQUIRE_THROWS_AS(conversion_factor(Quantity::Virial, "GPa", "eV"), std::invalid_argument);
    REQUIRE_THROWS_AS(quantity_from_string("stress"), std::invalid_argument);
}

TEST_CASE("Explicit Atoms normalization scales only known physical fields", "[units][atoms]") {
    using namespace GPUnits;
    Atoms atoms;
    atoms.cell_ = Eigen::Matrix3d::Identity();
    atoms.pos_ = {Eigen::RowVector3d(1., 2., 3.)};
    atoms.chemmap_ = {0};
    atoms.energy_ = 2.; atoms.have_energy = true;
    atoms.fenergy_ = 3.; atoms.have_fenergy = true;
    atoms.stress_ = Eigen::Matrix3d::Identity() * 4.; atoms.have_stress = true;
    atoms.site_energies_ = {5.};
    atoms.forces_ = {Eigen::Vector3d::Ones()};
    atoms.velocities_ = {Eigen::Vector3d::Ones()};
    atoms.local_attribs_d_["masses"] = Eigen::VectorXd::Constant(1, 1.66053906660e-27);
    atoms.local_attribs_d_["energy"] = Eigen::VectorXd::Constant(1, 19.);
    atoms.global_attribs_d_["pressure"] = 7.;
    atoms.features_["units"] = "custom metadata, not interpreted";
    atoms.hessian_ = Eigen::MatrixXd::Identity(3,3); atoms.have_hessian = true;
    atoms.hvp_directions_ = Eigen::MatrixXd::Ones(3,1);
    atoms.hvp_targets_ = Eigen::MatrixXd::Ones(3,1);
    atoms.charges_ = {9.}; atoms.masks = {3};
    canonicalize_atoms(atoms, {"nm", "Hartree", "fs", "kg"});
    const double e = Constants::ha2ev;
    REQUIRE(atoms.cell_(0,0) == 10.);
    REQUIRE(atoms.pos_[0](2) == 30.);
    REQUIRE(atoms.energy_ == 2. * e);
    REQUIRE(atoms.fenergy_ == 3. * e);
    REQUIRE(atoms.site_energies_[0] == 5. * e);
    REQUIRE(atoms.stress_(0,0) == 4. * e);
    REQUIRE(atoms.forces_[0](0) == Catch::Approx(e / 10.));
    REQUIRE(atoms.velocities_[0](0) == Catch::Approx(1e4));
    REQUIRE(atoms.local_attribs_d_.at("masses")[0] == Catch::Approx(1.));
    REQUIRE(atoms.hessian_(0,0) == Catch::Approx(e / 100.));
    REQUIRE(atoms.hvp_targets_(0,0) == Catch::Approx(e / 100.));
    REQUIRE(atoms.hvp_directions_(0,0) == 1.);
    REQUIRE(atoms.local_attribs_d_.at("energy")[0] == 19.);
    REQUIRE(atoms.global_attribs_d_.at("pressure") == 7.);
    REQUIRE(atoms.features_.at("units") == "custom metadata, not interpreted");
    REQUIRE(atoms.charges_[0] == 9.);
    REQUIRE(atoms.masks[0] == 3);
}

TEST_CASE("Fractional normalization preserves coordinates and missing labels", "[units][atoms]") {
    using namespace GPUnits;
    Atoms atoms;
    atoms.cell_ = Eigen::Matrix3d::Identity();
    atoms.pos_ = {Eigen::RowVector3d(.25, .5, .75)}; atoms.cartesian = false;
    const auto original = atoms.pos_;
    REQUIRE_THROWS_AS(canonicalize_atoms(atoms, {"nm", "eV", "bogus", "amu"}), std::invalid_argument);
    REQUIRE(atoms.cell_.isIdentity());
    canonicalize_atoms(atoms, {"nm", "eV", "ps", "amu"});
    REQUIRE(atoms.pos_ == original);
    REQUIRE(atoms.cell_(0,0) == 10.);
    REQUIRE_FALSE(atoms.have_energy);
    REQUIRE_FALSE(atoms.have_fenergy);
    REQUIRE_FALSE(atoms.have_stress);
    REQUIRE_FALSE(atoms.have_hessian);
    REQUIRE(atoms.forces_.empty());
    REQUIRE(atoms.local_attribs_d_.empty());
    REQUIRE_FALSE(atoms.cartesian);
}

TEST_CASE("Pressure and virial require explicit volume and sign", "[units]") {
    using namespace GPUnits;
    const double factor = pressure_to_virial_factor("GPa", 20., PressureVirialSign::Same);
    REQUIRE(factor == Catch::Approx(20. * conversion_factor(Quantity::Pressure,"GPa","eV/A^3")));
    REQUIRE(pressure_to_virial_factor("GPa",20.,PressureVirialSign::Opposite) == -factor);
    REQUIRE_THROWS_AS(pressure_to_virial_factor("GPa",0.,PressureVirialSign::Same), std::invalid_argument);
    REQUIRE_THROWS_AS(pressure_to_virial_factor("eV",20.,PressureVirialSign::Same), std::invalid_argument);
    REQUIRE(named_source_units("atomic").energy == "Hartree");
    REQUIRE(named_source_units("real").time == "fs");
    REQUIRE_THROWS_AS(named_source_units("mystery"), std::invalid_argument);
}
