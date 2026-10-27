// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/sgd_scheduler.h"
#include <cmath>
#include <memory>
#include <limits>

using Catch::Approx;
using namespace Training;

TEST_CASE("SGDScheduler factory creates correct types", "[sgd]")
{
    auto s1 = SGDScheduler::create("step", 0.01, 0.0, 100, 0.5, 10);
    REQUIRE(s1 != nullptr);
    REQUIRE(s1->get_lr() == Approx(0.01));

    auto s2 = SGDScheduler::create("cosine", 0.01, 0.001, 100, 0.0, 0);
    REQUIRE(s2 != nullptr);
    s2->on_step(0);
    REQUIRE(s2->get_lr() > 0.0);

    auto s3 = SGDScheduler::create("exponential", 0.01, 0.0, 100, 0.1, 0);
    REQUIRE(s3 != nullptr);
    REQUIRE(s3->get_lr() == Approx(0.01));

    auto s4 = SGDScheduler::create("none", 0.01, 0.0, 100, 0.0, 0);
    REQUIRE(s4 == nullptr);
}

TEST_CASE("StepDecayScheduler basic decay", "[sgd]")
{
    StepDecayScheduler s(1.0, 0.5, 5);

    REQUIRE(s.get_lr() == Approx(1.0));

    s.on_step(4);  // still in first decay period
    REQUIRE(s.get_lr() == Approx(1.0));

    s.on_step(5);  // first decay
    REQUIRE(s.get_lr() == Approx(0.5));

    s.on_step(9);  // still second period
    REQUIRE(s.get_lr() == Approx(0.5));

    s.on_step(10); // second decay
    REQUIRE(s.get_lr() == Approx(0.25));

    s.on_step(15); // third decay
    REQUIRE(s.get_lr() == Approx(0.125));
}

TEST_CASE("CosineDecayScheduler basic annealing", "[sgd]")
{
    CosineDecayScheduler s(1.0, 0.0, 100);

    // at step 0: progress=0, cos(0)=1 → lr = 0 + 0.5*1*2 = 1.0
    REQUIRE(s.get_lr() == Approx(0.0).margin(1e-10)); // not yet stepped

    s.on_step(0);
    REQUIRE(s.get_lr() == Approx(1.0).margin(1e-10));

    // at step 50 (halfway): progress=0.5, cos(π/2)=0 → lr = 0 + 0.5*1*1 = 0.5
    s.on_step(50);
    REQUIRE(s.get_lr() == Approx(0.5).margin(1e-10));

    // at step 100 (end): progress=1, cos(π)=-1 → lr = 0 + 0.5*1*0 = 0
    s.on_step(100);
    REQUIRE(s.get_lr() == Approx(0.0).margin(1e-10));
}

TEST_CASE("CosineDecayScheduler with non-zero final_lr", "[sgd]")
{
    CosineDecayScheduler s(0.01, 0.001, 200);

    s.on_step(0);
    REQUIRE(s.get_lr() == Approx(0.01).margin(1e-10));

    s.on_step(100); // halfway
    REQUIRE(s.get_lr() == Approx(0.0055).margin(1e-6));

    s.on_step(200);
    REQUIRE(s.get_lr() == Approx(0.001).margin(1e-10));
}

TEST_CASE("CosineDecayScheduler clamps at 1.0 progress", "[sgd]")
{
    CosineDecayScheduler s(1.0, 0.1, 10);

    s.on_step(10);
    REQUIRE(s.get_lr() == Approx(0.1).margin(1e-10));

    s.on_step(20); // past total_steps, progress clamped to 1.0
    REQUIRE(s.get_lr() == Approx(0.1).margin(1e-10));
}

TEST_CASE("ExponentialDecayScheduler basic decay", "[sgd]")
{
    ExponentialDecayScheduler s(1.0, 0.1);

    REQUIRE(s.get_lr() == Approx(1.0));

    s.on_step(0);
    REQUIRE(s.get_lr() == Approx(1.0).margin(1e-10));

    s.on_step(10);
    REQUIRE(s.get_lr() == Approx(std::exp(-1.0)).margin(1e-10));
}

TEST_CASE("ExponentialDecayScheduler large steps", "[sgd]")
{
    ExponentialDecayScheduler s(0.1, 0.5);

    s.on_step(0);
    REQUIRE(s.get_lr() == Approx(0.1));

    s.on_step(10);
    // lr = 0.1 * exp(-5.0) ≈ 0.00067
    REQUIRE(s.get_lr() > 0.0);
    REQUIRE(s.get_lr() < 0.01);
}

TEST_CASE("StepDecayScheduler with non-zero initial state", "[sgd]")
{
    // decay_steps=3, decay_rate=0.5
    StepDecayScheduler s(0.01, 0.5, 3);

    s.on_step(0);
    REQUIRE(s.get_lr() == Approx(0.01));

    s.on_step(2);
    REQUIRE(s.get_lr() == Approx(0.01));

    s.on_step(3);
    REQUIRE(s.get_lr() == Approx(0.005));

    s.on_step(6);
    REQUIRE(s.get_lr() == Approx(0.0025));

    s.on_step(9);
    REQUIRE(s.get_lr() == Approx(0.00125));
}

TEST_CASE("SGD schedulers reject invalid used parameters without rejecting unused controls", "[sgd]")
{
    const double nan = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(SGDScheduler::create("cosnie", .01, 0., 10, .5, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(StepDecayScheduler(.01, .5, 0), std::invalid_argument);
    REQUIRE_THROWS_AS(StepDecayScheduler(.01, .5, -1), std::invalid_argument);
    REQUIRE_THROWS_AS(StepDecayScheduler(.01, nan, 1), std::invalid_argument);
    REQUIRE_THROWS_AS(CosineDecayScheduler(.01, 0., 0), std::invalid_argument);
    REQUIRE_THROWS_AS(CosineDecayScheduler(.01, nan, 10), std::invalid_argument);
    REQUIRE_THROWS_AS(ExponentialDecayScheduler(.01, nan), std::invalid_argument);
    REQUIRE_THROWS_AS(SGDScheduler::create_with_warmup("step", .01, 0., 10, .5, 1, -1, 0.), std::invalid_argument);
    REQUIRE_THROWS_AS(SGDScheduler::create_with_warmup("step", .01, 0., 10, .5, 1, 1, nan), std::invalid_argument);
    REQUIRE_NOTHROW(SGDScheduler::create("exponential", .01, nan, 0, .5, 0));
    REQUIRE_NOTHROW(SGDScheduler::create("cosine", .01, 0., 10, nan, 0));
    REQUIRE_NOTHROW(SGDScheduler::create_with_warmup("step", .01, nan, 0, .5, 1, 0, nan));
    REQUIRE(SGDScheduler::create_with_warmup("none", .01, nan, 0, nan, 0, -1, nan) == nullptr);
}

TEST_CASE("SGD warmup retains its existing step offset and decay budget", "[sgd]")
{
    auto scheduler = SGDScheduler::create_with_warmup("cosine", 1., 0., 100, .5, 10, 20, 0.);
    scheduler->on_step(0); REQUIRE(scheduler->get_lr() == Approx(0.));
    scheduler->on_step(10); REQUIRE(scheduler->get_lr() == Approx(.5));
    scheduler->on_step(20); REQUIRE(scheduler->get_lr() == Approx(1.));
    scheduler->on_step(100); REQUIRE(scheduler->get_lr() == Approx(.5 * (1. + std::cos(.8 * M_PI))));
    scheduler->on_step(120); REQUIRE(scheduler->get_lr() == Approx(0.).margin(1e-14));
}
