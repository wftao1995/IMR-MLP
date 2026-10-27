#include <catch2/catch_all.hpp>

#include "unitest/maxvol_mtp_fixture.h"
#include "potentials/potential_hybrid.h"
#include "potentials/potential_hybrid_scale.h"
#include <chrono>

using namespace Potentials;
using Catch::Approx;

namespace {
struct ModelDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("gpml_native_site_" + std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count()));
    ~ModelDirectory() { std::filesystem::remove_all(path); }
};
}

TEST_CASE("Native MTP family virtual site EFS reproduces complete evaluation",
          "[lammps][native][site-efs]")
{
    ModelDirectory directory;
    const auto models = make_maxvol_mtp_models(directory.path);
    for (const auto &entry : models) {
        if (entry.pottype == "mtp_les" ||
            entry.name == "mtp_m_sog" ||
            entry.name == "mtp_m_les") continue;
        INFO(entry.name);
        potential_wrapper wrapper(entry.settings);
        AnyPotential &potential = *wrapper.pot;
        REQUIRE(potential.supports_site_efs());
        for (bool periodic : {false, true}) {
            INFO(periodic);
            Atoms reference = maxvol_mtp_atoms();
            reference.cell_ << 5.7, 0., 0., 0.8, 6.2, 0., 0.3, 0.6, 6.5;
            reference.pbc_ = {periodic, periodic, periodic};
            reference.pos_[2] = {6.9, 2.3, 2.4};
            Neighborhoods neighborhoods(reference, potential.cutoff);
            potential.getEFS(reference, neighborhoods);
            double total = 0.0;
            Eigen::Matrix3d virial = Eigen::Matrix3d::Zero();
            SiteCoordinateMatrix forces = SiteCoordinateMatrix::Zero(reference.size(), 3);
            for (int i = 0; i < neighborhoods.size(); ++i) {
                auto &site = neighborhoods[i];
                double energy = 123.0;
                SiteCoordinateMatrix gradient;
                potential.CalcSiteEFS(site, energy, gradient);
                REQUIRE(gradient.rows() == site.nbhcount_);
                REQUIRE(energy == Approx(reference.site_energies_[i]).margin(2e-10));
                total += energy;
                for (int j = 0; j < site.nbhcount_; ++j) {
                    forces.row(i) += gradient.row(j);
                    forces.row(site.indices_[j]) -= gradient.row(j);
                    virial.noalias() -= gradient.row(j).transpose() * site.pos_[j];
                }
                if (site.nbhcount_) {
                    const double step = 1e-6;
                    Neighborhood plus = site, minus = site;
                    plus.pos_[0](1) += step;
                    minus.pos_[0](1) -= step;
                    plus.distances_[0] = plus.pos_[0].norm();
                    minus.distances_[0] = minus.pos_[0].norm();
                    double ep = 0., em = 0.;
                    potential.CalcSiteE(plus, ep);
                    potential.CalcSiteE(minus, em);
                    REQUIRE((ep - em) / (2. * step) ==
                        Approx(gradient(0, 1)).epsilon(2e-5).margin(2e-7));
                }
            }
            REQUIRE(total == Approx(reference.energy_).margin(2e-10));
            REQUIRE((virial - reference.stress_).norm() < 2e-9);
            for (int i = 0; i < reference.size(); ++i)
                REQUIRE((forces.row(i).transpose() - reference.forces_[i]).norm() < 2e-9);
        }
        Atoms isolated;
        isolated.cell_ = 20. * Eigen::Matrix3d::Identity();
        isolated.pbc_ = {false, false, false};
        isolated.pos_ = {{1., 2., 3.}};
        isolated.chemmap_ = {1};
        Neighborhoods empty_neighborhoods(isolated, potential.cutoff);
        potential.getEFS(isolated, empty_neighborhoods);
        double isolated_energy = 123.;
        SiteCoordinateMatrix empty_gradient = SiteCoordinateMatrix::Ones(5, 3);
        potential.CalcSiteEFS(empty_neighborhoods[0], isolated_energy, empty_gradient);
        REQUIRE(empty_gradient.rows() == 0);
        REQUIRE(isolated_energy == Approx(isolated.energy_).margin(2e-10));
    }
}

TEST_CASE("MTP-M long-range models reject virtual local site EFS",
          "[lammps][native][site-efs]")
{
    ModelDirectory directory;
    for (const auto &entry : make_maxvol_mtp_models(directory.path)) {
        if (entry.name != "mtp_m_sog" && entry.name != "mtp_m_les") continue;
        potential_wrapper wrapper(entry.settings);
        AnyPotential &potential = *wrapper.pot;
        REQUIRE_FALSE(potential.supports_site_efs());
        Atoms atoms = maxvol_mtp_atoms();
        Neighborhoods neighborhoods(atoms, potential.cutoff);
        double energy = 0.;
        SiteCoordinateMatrix gradient;
        REQUIRE_THROWS(potential.CalcSiteEFS(neighborhoods[0], energy, gradient));
    }
}

TEST_CASE("Native composite site EFS preserves fixed cores and mixing response",
          "[lammps][native][site-efs][hybrid]")
{
    ModelDirectory directory;
    std::filesystem::create_directories(directory.path);
    PotMTP first, second;
    load_swap_fixture(first, 2, false);
    load_swap_fixture(second, 2, true);
    const auto first_file = (directory.path / "first.mtp").string();
    const auto second_file = (directory.path / "second.mtp").string();
    first.Save(first_file);
    second.Save(second_file);
    for (int mode = 0; mode < 4; ++mode) {
        INFO(mode);
        GPSettings settings;
        settings.Modify("pottype", mode < 2 ? "hybrid" : "hybrid_scale");
        settings.Modify("npot", "2");
        settings.Modify("pot1:pottype", "mtp");
        settings.Modify("pot1:potfile", first_file);
        settings.Modify("pot1:train", "false");
        settings.Modify("pot2:pottype", mode == 1 ? "zbl" : "mtp");
        if (mode == 1) {
            settings.Modify("pot2:z", "2 3");
            settings.Modify("pot2:r0", "1.5");
            settings.Modify("pot2:r1", "3.8");
        } else {
            settings.Modify("pot2:potfile", second_file);
            settings.Modify("pot2:train", "false");
        }
        settings.Modify("feature", mode == 2 ? "min_distance" : "smooth_coordination");
        settings.Modify("feature_r0", "0.5");
        settings.Modify("feature_r1", "6.5");
        settings.Modify("boundary1", mode == 2 ? "1.5 3.5" : "0.2 1.8");
        potential_wrapper wrapper(settings);
        AnyPotential &potential = *wrapper.pot;
        REQUIRE(potential.supports_site_efs());
        if (mode < 2) {
            const auto &hybrid = dynamic_cast<const PotentialHybrid &>(potential);
            REQUIRE(hybrid.children().size() == 2);
        } else {
            const auto &hybrid = dynamic_cast<const PotentialHybridScale &>(potential);
            REQUIRE(hybrid.children().size() == 2);
            REQUIRE(hybrid.boundaries().size() == 1);
            REQUIRE(hybrid.child_cutoffs().size() == 2);
            REQUIRE(hybrid.child_types().size() == 2);
            REQUIRE(hybrid.feature_kind() == (mode == 2 ? PotentialHybridScale::FeatureKind::MinDistance
                : PotentialHybridScale::FeatureKind::SmoothCoordination));
        }
        Atoms atoms = maxvol_mtp_atoms();
        atoms.pbc_ = {false, false, false};
        atoms.pos_[2] = {6.9, 2.3, 2.4};
        Neighborhoods neighborhoods(atoms, potential.cutoff);
        potential.getEFS(atoms, neighborhoods);
        SiteCoordinateMatrix forces = SiteCoordinateMatrix::Zero(atoms.size(), 3);
        Eigen::Matrix3d virial = Eigen::Matrix3d::Zero();
        double sum = 0.;
        for (int i = 0; i < atoms.size(); ++i) {
            auto &site = neighborhoods[i];
            double energy = 17.;
            SiteCoordinateMatrix gradient;
            potential.CalcSiteEFS(site, energy, gradient);
            REQUIRE(energy == Approx(atoms.site_energies_[i]).margin(1e-10));
            sum += energy;
            for (int j = 0; j < site.nbhcount_; ++j) {
                forces.row(i) += gradient.row(j);
                forces.row(site.indices_[j]) -= gradient.row(j);
                virial.noalias() -= gradient.row(j).transpose() * site.pos_[j];
                const double step = 1e-6;
                auto plus = site, minus = site;
                plus.pos_[j](0) += step;
                minus.pos_[j](0) -= step;
                plus.distances_[j] = plus.pos_[j].norm();
                minus.distances_[j] = minus.pos_[j].norm();
                double ep = 0., em = 0.;
                potential.CalcSiteE(plus, ep);
                potential.CalcSiteE(minus, em);
                REQUIRE((ep - em) / (2. * step) == Approx(gradient(j, 0)).epsilon(2e-5).margin(2e-7));
            }
        }
        REQUIRE(sum == Approx(atoms.energy_).margin(1e-10));
        REQUIRE((virial - atoms.stress_).norm() < 2e-9);
        for (int i = 0; i < atoms.size(); ++i)
            REQUIRE((forces.row(i).transpose() - atoms.forces_[i]).norm() < 2e-9);
    }
}
