#include <catch2/catch_all.hpp>
#include "analyse/compute_fftfilter.h"
#include <cmath>

namespace {
Atoms frame(double f) {
    Atoms atom;
    atom.cell_ = 20. * Eigen::Matrix3d::Identity();
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{1 + .1 * f, 2, 3}, {4, 5 + .2 * f, 6}};
    atom.chemmap_ = {0, 0};
    atom.forces_ = {{.1 + f, .2, .3}, {-.4, .5 + 2 * f, -.6}};
    atom.velocities_ = {{1 + f, 2, 3}, {4, 5 + f, 6}};
    atom.site_energies_ = {2 + .4 * f, 4 + .6 * f};
    atom.have_energy = true; atom.energy_ = 6 + f;
    atom.have_fenergy = true; atom.fenergy_ = 5.8 + f;
    atom.have_stress = true; atom.stress_ = (1 + f) * Eigen::Matrix3d::Identity();
    atom.global_attribs_d_["temperature"] = 300 + 2 * f;
    atom.global_attribs_v_["global"] = Eigen::Vector2d(1 + f, 2 + 2 * f);
    atom.local_attribs_d_["local"] = Eigen::Vector2d(3 + f, 4 + 2 * f);
    atom.local_attribs_v_["vector"] = Eigen::MatrixXd::Constant(2, 2, 5 + f);
    return atom;
}
std::vector<Atoms> trajectory() { return {frame(0), frame(1), frame(2), frame(3)}; }
void require_same(const Atoms &actual, const Atoms &before, double tolerance = 1e-12) {
    REQUIRE(actual.cartesian == before.cartesian);
    REQUIRE(actual.pbc_ == before.pbc_);
    REQUIRE(actual.cell_.isApprox(before.cell_, tolerance));
    for (int i = 0; i < actual.size(); ++i)
        REQUIRE(actual.pos_[i].isApprox(before.pos_[i], tolerance));
    REQUIRE(actual.have_energy == before.have_energy);
    REQUIRE(actual.have_fenergy == before.have_fenergy);
    REQUIRE(actual.have_stress == before.have_stress);
    REQUIRE(actual.energy_ == Catch::Approx(before.energy_).margin(tolerance));
    REQUIRE(actual.fenergy_ == Catch::Approx(before.fenergy_).margin(tolerance));
    REQUIRE(actual.stress_.isApprox(before.stress_, tolerance));
    REQUIRE(actual.site_energies_.size() == before.site_energies_.size());
    for (int i = 0; i < actual.site_energies_.size(); ++i)
        REQUIRE(actual.site_energies_[i] == Catch::Approx(before.site_energies_[i]).margin(tolerance));
    REQUIRE(actual.forces_.size() == before.forces_.size());
    for (int i = 0; i < actual.forces_.size(); ++i) REQUIRE(actual.forces_[i].isApprox(before.forces_[i], tolerance));
    REQUIRE(actual.velocities_.size() == before.velocities_.size());
    for (int i = 0; i < actual.velocities_.size(); ++i) REQUIRE(actual.velocities_[i].isApprox(before.velocities_[i], tolerance));
    REQUIRE(actual.global_attribs_d_.size() == before.global_attribs_d_.size());
    for (const auto &[key, value] : before.global_attribs_d_)
        REQUIRE(actual.global_attribs_d_.at(key) == Catch::Approx(value).margin(tolerance));
    REQUIRE(actual.global_attribs_v_.size() == before.global_attribs_v_.size());
    for (const auto &[key, value] : before.global_attribs_v_)
        REQUIRE(actual.global_attribs_v_.at(key).isApprox(value, tolerance));
    REQUIRE(actual.local_attribs_d_.size() == before.local_attribs_d_.size());
    for (const auto &[key, value] : before.local_attribs_d_)
        REQUIRE(actual.local_attribs_d_.at(key).isApprox(value, tolerance));
    REQUIRE(actual.local_attribs_v_.size() == before.local_attribs_v_.size());
    for (const auto &[key, value] : before.local_attribs_v_)
        REQUIRE(actual.local_attribs_v_.at(key).isApprox(value, tolerance));
}
void smooth(ComputeSmoothFFT &smoother, std::vector<Atoms> &atoms, bool mirrored) {
    if (mirrored) smoother.compute2(atoms, 1, ComputeSmoothFFT::BLOCK);
    else smoother.compute(atoms, 1, ComputeSmoothFFT::BLOCK);
}
}

TEST_CASE("FFT first-frame enabled channels must be complete before any mutation", "[fft][channels]") {
    const auto mirrored = GENERATE(false, true);
    const std::string missing = GENERATE("forces", "partial forces", "site", "velocity", "energy", "free", "stress",
                                         "global double", "global vector", "short global", "local double", "short local",
                                         "local vector", "short vector rows", "short vector columns");
    auto atoms = trajectory();
    atoms.front().to_fractional();
    auto &last = atoms.back();
    if (missing == "forces") last.forces_.clear();
    if (missing == "partial forces") last.forces_.pop_back();
    if (missing == "site") last.site_energies_.clear();
    if (missing == "velocity") last.velocities_.clear();
    if (missing == "energy") last.have_energy = false;
    if (missing == "free") last.have_fenergy = false;
    if (missing == "stress") last.have_stress = false;
    if (missing == "global double") last.global_attribs_d_.erase("temperature");
    if (missing == "global vector") last.global_attribs_v_.erase("global");
    if (missing == "short global") last.global_attribs_v_.at("global").conservativeResize(1);
    if (missing == "local double") last.local_attribs_d_.erase("local");
    if (missing == "short local") last.local_attribs_d_.at("local").conservativeResize(1);
    if (missing == "local vector") last.local_attribs_v_.erase("vector");
    if (missing == "short vector rows") last.local_attribs_v_.at("vector").conservativeResize(1, 2);
    if (missing == "short vector columns") last.local_attribs_v_.at("vector").conservativeResize(2, 1);
    const auto before = atoms;
    ComputeSmoothFFT smoother;
    REQUIRE_THROWS_WITH(smooth(smoother, atoms, mirrored), Catch::Matchers::ContainsSubstring("incomplete"));
    for (int i = 0; i < atoms.size(); ++i) require_same(atoms[i], before[i], 0.);
}

TEST_CASE("Complete FFT physical and double metadata channels retain DC and original signals", "[fft][channels]") {
    const auto mirrored = GENERATE(false, true);
    const auto mean_only = GENERATE(false, true);
    auto atoms = trajectory();
    const auto before = atoms;
    ComputeSmoothFFT smoother;
    smoother.params.freq_cutoff = mean_only ? 0. : 1e9;
    smooth(smoother, atoms, mirrored);
    for (int i = 0; i < atoms.size(); ++i)
        require_same(atoms[i], mean_only ? frame(1.5) : before[i]);
}

TEST_CASE("FFT channels appearing only later retain their existing unfiltered behavior", "[fft][channels]") {
    const auto mirrored = GENERATE(false, true);
    auto atoms = trajectory();
    auto &first = atoms.front();
    first.forces_.clear(); first.velocities_.clear(); first.site_energies_.clear();
    first.have_energy = first.have_fenergy = first.have_stress = false;
    first.global_attribs_d_.clear(); first.global_attribs_v_.clear();
    first.local_attribs_d_.clear(); first.local_attribs_v_.clear();
    const auto before = atoms;
    ComputeSmoothFFT smoother;
    smoother.params.freq_cutoff = 1e9;
    smooth(smoother, atoms, mirrored);
    for (int i = 0; i < atoms.size(); ++i) require_same(atoms[i], before[i]);
}

TEST_CASE("FFT velocity reconstruction replaces an incomplete source velocity channel", "[fft][channels]") {
    auto control = trajectory(), incomplete = control;
    incomplete[1].velocities_.clear();
    incomplete[2].velocities_.resize(1);
    ComputeSmoothFFT smoother;
    smoother.rebuild_velo = true;
    smoother.params.freq_cutoff = 1e9;
    smoother.compute2(control, 1, ComputeSmoothFFT::BLOCK);
    smoother.compute2(incomplete, 1, ComputeSmoothFFT::BLOCK);
    for (int i = 0; i < incomplete.size(); ++i) {
        REQUIRE(incomplete[i].velocities_.size() == 2);
        for (int j = 0; j < 2; ++j) {
            REQUIRE(incomplete[i].velocities_[j].allFinite());
            REQUIRE(incomplete[i].velocities_[j].isApprox(control[i].velocities_[j], 1e-12));
        }
    }
}

TEST_CASE("FFT validates atom species order and PBC before mutation", "[fft][channels][identity]") {
    const bool mirrored = GENERATE(false, true);
    const std::string invalid = GENERATE("species", "species order", "short species", "PBC");
    auto atoms = trajectory();
    for (auto &atom : atoms) atom.chemmap_ = {0, 1};
    atoms.front().to_fractional();
    if (invalid == "species") atoms.back().chemmap_[1] = 2;
    if (invalid == "species order") std::swap(atoms.back().chemmap_[0], atoms.back().chemmap_[1]);
    if (invalid == "short species") atoms.back().chemmap_.pop_back();
    if (invalid == "PBC") atoms.back().pbc_ = {true, false, false};
    const auto before = atoms;
    ComputeSmoothFFT smoother;
    REQUIRE_THROWS_WITH(smooth(smoother, atoms, mirrored),
        Catch::Matchers::ContainsSubstring(invalid == "PBC" ? "same PBC" : "same species"));
    for (int i = 0; i < atoms.size(); ++i) {
        require_same(atoms[i], before[i], 0.);
        REQUIRE(atoms[i].chemmap_ == before[i].chemmap_);
    }
}
