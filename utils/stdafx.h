// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file stdafx.h
 * @brief Precompiled-header / convenience blanket include.
 *
 * Keep only what nearly every translation unit needs; heavy / rare headers
 * (FFTW3, Eigen/Tensor) are included explicitly where used so the PCH
 * stays lean.
 *
 * @see utils/eigen_core.h  — lightweight alternative when only Eigen + STL
 *                            is needed (no fmt, no OpenMP).
 */

#include "utils/eigen_core.h"

#include <chrono>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <ranges>
#include <random>

#include <omp.h>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <fmt/ostream.h>
