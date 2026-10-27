#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp.h"
#include <set>

using namespace Potentials;

namespace {
Neighborhood environment() {
    Neighborhood n;
    n.centertype = 0;
    n.nbhcount_ = 4;
    n.chemmap_ = {0, 1, 0, 1};
    n.indices_ = {1, 2, 1, 0};
    n.pos_ = {{2.2, .3, .4}, {-.6, 2.5, .2}, {.5, -.4, 2.8}, {5.7, .1, .2}};
    for (const auto &r : n.pos_) n.distances_.push_back(r.norm());
    return n;
}

void compare_site(PotMTP &p, Neighborhood &n) {
    p.radial_local_cache = false;
    double reference = 0.;
    SiteCoordinateMatrix reference_gradient;
    p.CalcSiteEFS(n, reference, reference_gradient);
    p.radial_local_cache = true;
    double actual = 0.;
    SiteCoordinateMatrix gradient;
    p.CalcSiteEFS(n, actual, gradient);
    REQUIRE(actual == Catch::Approx(reference).margin(1e-12));
    REQUIRE((gradient - reference_gradient).norm() < 1e-11);

    std::set<std::pair<int, int>> pairs;
    for (int i = 0; i < p.alpha_index_basic_count; ++i)
        pairs.emplace(p.alpha_index_basic(i, 0), p.nu_arr_list[i]);
    int active = 0;
    for (int j = 0; j < n.nbhcount_; ++j)
        active += n.distances_[j] <= p.cutoffs(n.centertype, n.chemmap_[j]);
    REQUIRE(p.radial_cache.evaluations() == active * pairs.size());
    REQUIRE(p.radial_cache.evaluations() < active * p.alpha_index_basic_count);

    p.radial_local_cache = false;
    double energy_only = 0.;
    p.CalcSiteE(n, energy_only);
    p.radial_local_cache = true;
    actual = 0.;
    p.CalcSiteE(n, actual);
    REQUIRE(actual == Catch::Approx(energy_only).margin(1e-12));
    REQUIRE(p.radial_cache.evaluations() == active * pairs.size());
}
}

TEST_CASE("MTP radial normalization cache is local to geometry and parameters",
          "[mtp][radial-cache]") {
    PotMTP p("unitest/test-data/portable-mtp/les.mtp");
    REQUIRE_FALSE(p.radial_local_cache);
    auto n = environment();
    compare_site(p, n);
    const auto capacity = p.radial_cache.capacity();
    n.pos_[0] += Eigen::RowVector3d(.17, -.08, .04);
    n.distances_[0] = n.pos_[0].norm();
    p.radial_coeffs[0] += .23;
    compare_site(p, n);
    REQUIRE(p.radial_cache.capacity() == capacity);

    p.InitSettings();
    GPSettings settings;
    settings.Modify("radial_local_cache", "true");
    p.ApplySettings(settings);
    REQUIRE(p.radial_local_cache);
    PotMTP copy(p);
    REQUIRE(copy.radial_local_cache);
    compare_site(copy, n);
}

TEST_CASE("MTP radial normalization cache has separate OMP workspaces",
          "[mtp][radial-cache]") {
    PotMTP p("unitest/test-data/portable-mtp/les.mtp");
    p.radial_local_cache = true;
    p.omp_init();
    auto n = environment();
    p.buff_site_energy = 0.;
    p.CalcSiteEFS(n, p.buff_site_energy, p.buff_site_energy_ders);
    const auto expected = p.buff_site_energy;
    const auto gradient = p.buff_site_energy_ders;
    const auto serial_count = p.radial_cache.evaluations();
    auto &buffer = p.omp_buffers_list[0];
    buffer.buff_site_energy = 0.;
    p.CalcSiteEFSOMP(n, buffer);
    REQUIRE(buffer.buff_site_energy == Catch::Approx(expected).margin(1e-12));
    REQUIRE((buffer.buff_site_energy_ders - gradient).norm() < 1e-11);
    REQUIRE(buffer.radial_cache.evaluations() == serial_count);
    buffer.buff_site_energy = 0.;
    p.CalcSiteEOMP(n, buffer);
    REQUIRE(buffer.buff_site_energy == Catch::Approx(expected).margin(1e-12));
    REQUIRE(buffer.radial_cache.evaluations() == serial_count);
}
