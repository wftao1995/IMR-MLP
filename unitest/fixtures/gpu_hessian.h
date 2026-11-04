// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "mtp_linear_efs.h"
#include "analyse/compute_hessian.h"
#include "potentials/mtp_gpu_adapter.h"
#include "tools/hessian_utils.h"
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <string>

// Small, public synthetic models and geometries. No files or device work are
// created by this fixture; callers own all execution and temporary resources.
namespace GpuHessianFixture
{
inline constexpr const char *version = "gpu-hessian-v1";
// Frozen before device measurements. The propagated force gate below is only
// a diagnostic or a topology-boundary bound, never the smooth-curvature gate.
inline constexpr double curvature_atol = 2e-7, curvature_rtol = 2e-7;
inline constexpr double analytic_atol = 2e-6, analytic_rtol = 2e-6;
inline constexpr double force_atol = MTPLinearEFSFixture::force_atol;
inline constexpr double force_rtol = MTPLinearEFSFixture::rtol;

inline std::string number(double value)
{
    std::ostringstream out;
    out << std::setprecision(17) << value;
    return out.str();
}

inline GPSettings settings(const std::string &device = "cpu", double step = 1e-3, int atoms = 64,
                           int structures = 4, int memory = 64, int host = 16)
{
    GPSettings s;
    s.Modify("method", "force_fd");
    s.Modify("nudge_pos", number(step));
    s.Modify("req_fc", "true");
    s.Modify("req_cell", "false");
    s.Modify("use_symmetry", "false");
    s.Modify("device", device);
    s.Modify("gpu_id", "0");
    s.Modify("gpu_batch_atoms", std::to_string(atoms));
    s.Modify("gpu_batch_structures", std::to_string(structures));
    s.Modify("gpu_memory_mib", std::to_string(memory));
    s.Modify("host_memory_mib", std::to_string(host));
    return s;
}

inline void load(Potentials::PotMTP &model, int variant = 1)
{
    MTPLinearEFSFixture::load(model, variant);
    model.omp_init();
    model.set_inited();
}

inline Atoms cluster(int count = 5)
{
    Atoms a;
    a.cell_ = 20. * Eigen::Matrix3d::Identity();
    a.pbc_ = {false, false, false};
    a.pos_ = {{1.1, 1.3, 1.5}, {3.2, 1.6, 1.8}, {1.7, 3.8, 2.1}, {3.7, 3.3, 3.5}, {2.4, 2.2, 4.2}};
    a.chemmap_ = {0, 1, 0, 1, 0};
    a.pos_.resize(count);
    a.chemmap_.resize(count);
    return a;
}

inline Atoms periodic(int count = 4)
{
    Atoms a;
    a.pbc_ = {true, true, true};
    a.cell_ << 3.1, 0, 0, .2, 3.4, 0, .1, .3, 3.7;
    a.pos_ = {{.3, .4, .5}};
    a.chemmap_ = {0};
    if (count == 4)
    {
        a.pos_.push_back({1.5, 1.2, 1.8});
        a.chemmap_.push_back(1);
        for (int i = 0; i < 2; ++i)
        {
            a.pos_.push_back(a.pos_[i] + a.cell_.row(0));
            a.chemmap_.push_back(a.chemmap_[i]);
        }
        a.cell_.row(0) *= 2.;
    }
    return a;
}

inline void labels(Atoms &a)
{
    a.energy_ = 31.;
    a.have_energy = false;
    a.stress_ << 1., 2., 3., 4., 5., 6., 7., 8., 9.;
    a.have_stress = true;
    a.forces_.assign(a.size(), Eigen::Vector3d(.2, -.3, .7));
    a.site_energies_.assign(a.size(), -.4);
    a.fenergy_ = 17.;
    a.have_fenergy = true;
    a.charges_.assign(a.size(), .03);
    a.features_["preserved"] = "curvature input";
    a.fitness = .7;
    a.atomic_fitness.assign(a.size(), .2);
    // Deliberately small preexisting labels, not a quadratic allocation for a
    // resource-failure input. Curvature execution must not inspect or copy them
    // into every displaced structure.
    a.hessian_ = Eigen::MatrixXd::Constant(2, 2, .41);
    a.have_hessian = true;
    a.hvp_directions_ = Eigen::MatrixXd::Constant(3 * a.size(), 1, .13);
    a.hvp_targets_ = Eigen::MatrixXd::Constant(3 * a.size(), 1, -.23);
}

inline bool same_matrix(const Eigen::MatrixXd &a, const Eigen::MatrixXd &b)
{
    return a.rows() == b.rows() && a.cols() == b.cols() && (a.array() == b.array()).all();
}

inline bool unchanged(const Atoms &a, const Atoms &b)
{
    return a.energy_ == b.energy_ && a.have_energy == b.have_energy && a.stress_ == b.stress_ &&
           a.have_stress == b.have_stress && a.forces_ == b.forces_ &&
           a.site_energies_ == b.site_energies_ && a.fenergy_ == b.fenergy_ &&
           a.have_fenergy == b.have_fenergy && a.charges_ == b.charges_ && a.cell_ == b.cell_ &&
           a.pos_ == b.pos_ && a.chemmap_ == b.chemmap_ && a.pbc_ == b.pbc_ &&
           a.cartesian == b.cartesian && a.have_ghost == b.have_ghost && a.nlocal_ == b.nlocal_ &&
           a.nghost_ == b.nghost_ && a.map_ghost_ == b.map_ghost_ && a.features_ == b.features_ &&
           a.fitness == b.fitness && a.atomic_fitness == b.atomic_fitness &&
           a.have_hessian == b.have_hessian && same_matrix(a.hessian_, b.hessian_) &&
           same_matrix(a.hvp_directions_, b.hvp_directions_) &&
           same_matrix(a.hvp_targets_, b.hvp_targets_);
}

inline Eigen::VectorXd direction(int count, double phase = .0)
{
    Eigen::VectorXd v(3 * count);
    for (int i = 0; i < v.size(); ++i)
        v[i] = std::sin(.37 * (i + 1) + phase) + .2 * std::cos(.61 * (i + 2));
    return v;
}

inline Eigen::VectorXd translation(int count, int axis)
{
    Eigen::VectorXd v = Eigen::VectorXd::Zero(3 * count);
    for (int i = 0; i < count; ++i)
        v[3 * i + axis] = 1.;
    return v;
}

inline double maximum_atom_norm(const Eigen::VectorXd &direction)
{
    double value = 0.;
    for (int i = 0; i < direction.size() / 3; ++i)
        value = std::max(value, direction.segment<3>(3 * i).stableNorm());
    return value;
}

inline Eigen::VectorXd forces(Potentials::PotMTP &model, const Atoms &input, double cutoff = 0.)
{
    auto a = input;
    if (!a.cartesian)
        a.to_cartesian();
    a.resetEFS();
    Neighborhoods nbhs(a, cutoff > 1e-5 ? cutoff : model.cutoff);
    model.getEFS(a, nbhs);
    Eigen::VectorXd result(3 * a.size());
    for (int i = 0; i < a.size(); ++i)
        result.segment<3>(3 * i) = a.forces_[i];
    return result;
}

struct DirectionReference
{
    Eigen::VectorXd value;
    double delta = 0., force_scale = 0.;
};

inline DirectionReference force_direction(Potentials::PotMTP &model, const Atoms &input,
                                          const Eigen::VectorXd &v, double step, double cutoff = 0.)
{
    DirectionReference r;
    const double scale = maximum_atom_norm(v);
    if (scale == 0.)
    {
        r.value = Eigen::VectorXd::Zero(v.size());
        return r;
    }
    r.delta = step / scale;
    auto plus = input, minus = input;
    if (!plus.cartesian)
    {
        plus.to_cartesian();
        minus.to_cartesian();
    }
    for (int i = 0; i < input.size(); ++i)
    {
        const Eigen::RowVector3d dx = r.delta * v.segment<3>(3 * i).transpose();
        plus.pos_[i] += dx;
        minus.pos_[i] -= dx;
    }
    const auto fp = forces(model, plus, cutoff), fm = forces(model, minus, cutoff);
    r.force_scale = std::max(fp.cwiseAbs().maxCoeff(), fm.cwiseAbs().maxCoeff());
    r.value = -(fp - fm) / (2. * r.delta);
    return r;
}

struct DenseReference
{
    Eigen::MatrixXd value;
    double force_scale = 0.;
};

inline DenseReference force_dense(Potentials::PotMTP &model, const Atoms &input, double step,
                                  double cutoff = 0.)
{
    DenseReference r;
    const int n = 3 * input.size();
    r.value.resize(n, n);
    for (int j = 0; j < n; ++j)
    {
        const auto one = force_direction(model, input, Eigen::VectorXd::Unit(n, j), step, cutoff);
        r.value.col(j) = one.value;
        r.force_scale = std::max(r.force_scale, one.force_scale);
    }
    r.value = (.5 * (r.value + r.value.transpose())).eval();
    return r;
}

inline double fd_bound(double delta, double force_scale)
{
    return (force_atol + force_rtol * force_scale) / delta;
}

inline Eigen::MatrixXd analytic(Potentials::PotMTP &model, const Atoms &input, double cutoff = 0.)
{
    auto a = input;
    Neighborhoods nbhs(a, cutoff > 1e-5 ? cutoff : model.cutoff);
    Potentials::HessianMatrix blocks;
    model.getEFSHessian(a, nbhs, blocks);
    return HessianUtils::to_dense(blocks);
}
} // namespace GpuHessianFixture
