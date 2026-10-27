#include <catch2/catch_all.hpp>
#include "utils/lower_energy_envelope.h"
#include <cmath>
#include <limits>
#include <stdexcept>

using Catch::Approx;

TEST_CASE("Lower energy envelope resolves binary and ternary mixtures", "[hull]") {
    LowerEnergyEnvelope binary({{1, 0}, {0, 1}, {.5, .5}, {.5, .5}}, {0, 0, -1, -.7});
    REQUIRE(binary.affine_rank() == 1);
    REQUIRE(binary.reference_size() == 3);
    REQUIRE(binary.energy_at({.5, .5}) == Approx(-1));
    REQUIRE(binary.energy_at({.75, .25}) == Approx(-.5));
    REQUIRE(binary.energy_at({1, 0}) == Approx(0).margin(1e-9));
    // Repeated RHS/objective changes must not alter the fixed reference set.
    REQUIRE(binary.energy_at({.25, .75}) == Approx(-.5));
    REQUIRE(binary.energy_at({.75, .25}) == Approx(-.5));

    LowerEnergyEnvelope ternary({{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, {1./3, 1./3, 1./3}},
                               {0, 0, 0, -.6});
    REQUIRE(ternary.affine_rank() == 2);
    REQUIRE(ternary.energy_at({1./3, 1./3, 1./3}) == Approx(-.6));
    REQUIRE(ternary.energy_at({.5, .25, .25}) == Approx(-.45));
}

TEST_CASE("Lower energy envelope handles unary and degenerate coordinates", "[hull]") {
    LowerEnergyEnvelope unary({{1}, {1}, {1}}, {-2, -1, 0});
    REQUIRE(unary.affine_rank() == 0);
    REQUIRE(unary.reference_size() == 1);
    REQUIRE(unary.energy_at({1}) == Approx(-2));
    REQUIRE_THROWS_AS(unary.energy_at({.9}), std::out_of_range);

    LowerEnergyEnvelope point({{4, -8, 2}}, {-3});
    REQUIRE(point.energy_at({4, -8, 2}) == Approx(-3));
    REQUIRE_THROWS_AS(point.energy_at({4, -8, 3}), std::out_of_range);

    LowerEnergyEnvelope line({{0, 1e10, 7}, {1e-10, 2e10, 7}, {2e-10, 3e10, 7}}, {0, -1, 0});
    REQUIRE(line.affine_rank() == 1);
    REQUIRE(line.energy_at({.5e-10, 1.5e10, 7}) == Approx(-.5));
    REQUIRE_THROWS_AS(line.energy_at({.5e-10, 1.6e10, 7}), std::out_of_range);
    REQUIRE_THROWS_AS(line.energy_at({3e-10, 4e10, 7}), std::out_of_range);

    LowerEnergyEnvelope no_coordinates({{}, {}}, {-2, -3});
    REQUIRE(no_coordinates.energy_at({}) == Approx(-3));
}

TEST_CASE("Lower energy envelope checks convex domain after affine reduction", "[hull]") {
    LowerEnergyEnvelope triangle({{0, 0, 1}, {1, 0, 1}, {0, 1, 1}}, {0, 1, 2});
    REQUIRE(triangle.energy_at({.2, .3, 1}) == Approx(.8));
    REQUIRE_THROWS_AS(triangle.energy_at({.8, .8, 1}), std::out_of_range);
    REQUIRE_THROWS_AS(triangle.energy_at({.2, .3, 2}), std::out_of_range);
    REQUIRE(triangle.energy_at({.3, .1, 1}) == Approx(.5));
    REQUIRE(triangle.energy_at({.2, .3, 1}) == Approx(.8));
}

TEST_CASE("Lower energy envelope is invariant under affine energy shifts and reference order", "[hull]") {
    LowerEnergyEnvelope original({{0}, {.5}, {1}}, {0, -1, 0});
    LowerEnergyEnvelope shifted({{1}, {0}, {.5}}, {13, 10, 10.5});
    for (double x : {0., .1, .5, .75, 1.})
        REQUIRE(shifted.energy_at({x}) == Approx(original.energy_at({x}) + 10 + 3*x));
}

TEST_CASE("Lower energy envelope preserves original reference endpoint energies exactly", "[hull]") {
    LowerEnergyEnvelope hull({{0}, {1}}, {-.6, .3});
    REQUIRE(hull.energy_at({0}) == -.6);
    REQUIRE(hull.energy_at({1}) == .3);
    REQUIRE(hull.energy_at({.5}) == Approx(-.15));
}

TEST_CASE("Lower energy envelope rejects invalid references and queries", "[hull]") {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(LowerEnergyEnvelope({}, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(LowerEnergyEnvelope({{0}}, {}), std::invalid_argument);
    REQUIRE_THROWS_AS(LowerEnergyEnvelope({{0}, {1, 2}}, {0, 1}), std::invalid_argument);
    REQUIRE_THROWS_AS(LowerEnergyEnvelope({{nan}}, {0}), std::invalid_argument);
    REQUIRE_THROWS_AS(LowerEnergyEnvelope({{0}}, {nan}), std::invalid_argument);
    LowerEnergyEnvelope hull({{0}, {1}}, {0, 1});
    REQUIRE_THROWS_AS(hull.energy_at({nan}), std::invalid_argument);
    REQUIRE_THROWS_AS(hull.energy_at({0, 1}), std::invalid_argument);
}
