// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "train/training_data.h"

#include <filesystem>

TEST_CASE("Shared training loader restores structure and atom fitness", "[training_data]") {
    const auto path = std::filesystem::temp_directory_path() / "imrmlp_training_data.bin";
    Atoms first;
    first.cell_ = 10 * Eigen::Matrix3d::Identity();
    first.pos_ = {{1, 0, 0}, {3, 0, 0}};
    first.chemmap_ = {0, 0};
    first.global_attribs_d_["fitness"] = .25;
    first.local_attribs_d_["atomic_fitness"] = Eigen::Vector2d(0.0, .75);
    auto second = first;
    second.global_attribs_d_.clear();
    GPIO io;
    std::vector<Atoms> frames{first, second};
    REQUIRE(io.write(frames, path.string(), "bin"));
    auto loaded = Training::load_training_data(io, path.string(), "bin");
    Training::restore_training_fitness(loaded);
    std::filesystem::remove(path);
    REQUIRE(loaded.size() == 2);
    REQUIRE(loaded[0].fitness == .25);
    REQUIRE(loaded[1].fitness == 1.0);
    REQUIRE(loaded[0].atomic_fitness == std::vector<double>{0.0, .75});
    auto neighborhoods = Training::training_neighborhoods(loaded, 3.0);
    REQUIRE(neighborhoods.size() == loaded.size());
    REQUIRE_THROWS(Training::load_training_data(io, path.string(), "bin"));
}

TEST_CASE("Shared training fitness rejects truncated atom weights", "[training_data]") {
    Atoms frame;
    frame.pos_ = {{1, 0, 0}, {3, 0, 0}};
    frame.chemmap_ = {0, 0};
    frame.local_attribs_d_["atomic_fitness"] = Eigen::VectorXd::Ones(1);
    std::vector<Atoms> frames{frame};
    REQUIRE_THROWS_AS(Training::restore_training_fitness(frames), std::invalid_argument);
}
