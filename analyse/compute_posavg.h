// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "utils/gputils.h"
#include "base/structures.h"

// Input preserves the coordinate representation of the first frame.
enum class AverageCoordinates { Input, Cartesian, Fractional };

struct PositionAverageOptions {
    AverageCoordinates coordinates = AverageCoordinates::Input;
    bool remove_drift = false; // Requires a fixed cell and per-atom masses.
    bool geometry_only = false;
    bool wrap_result = false;
    double relative_cell_tolerance = 1e-7;
    std::vector<double> masses;
};

class ComputePosavg : public ConnectSettings
{
private:
    PositionAverageOptions options_;
    AverageCoordinates coordinates_ = AverageCoordinates::Cartesian;
    Atoms reference_, previous_;
    std::vector<Eigen::RowVector3d> previous_fractional_, images_, unwrapped_;
    Eigen::Matrix3d cell_sum_ = Eigen::Matrix3d::Zero();
    std::vector<Eigen::RowVector3d> position_sum_;
    std::vector<double> site_energy_sum_;

public:
    int nframes = 0;
    Atoms atomacc;

    ComputePosavg() = default;
    explicit ComputePosavg(PositionAverageOptions options) : options_(std::move(options)) {}

    void init(const Atoms &atom);
    void accumulate(const Atoms &atom);
    void compute();
};
