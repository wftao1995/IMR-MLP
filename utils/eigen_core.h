// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file eigen_core.h
 * @brief Core includes commonly needed: Eigen/Dense + minimal STL.
 *
 * This is the lightweight subset of stdafx.h for files that do NOT need
 * FFTW3, Eigen/Tensor, or OpenMP.  Include this instead of "stdafx.h"
 * when possible to reduce transitive dependency weight.
 */

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>
