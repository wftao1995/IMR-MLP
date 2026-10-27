// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "constraints.h"
#include "utils/gputils.h"
#include <fmt/format.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace Constraints {
    namespace {
    void check_arguments(const double *x, int nx, double *g, int ng, double *f, double target, int expected) {
        if (!x || !g || !f || nx != expected || ng != expected || !std::isfinite(target))
            throw std::invalid_argument("Invalid lattice constraint callback arguments");
        for (int i = 0; i < nx; ++i)
            if (!std::isfinite(x[i])) throw std::invalid_argument("Non-finite lattice constraint coordinate");
    }
    void check_axis(const Atoms &atom, int axis) {
        if (axis < 0 || axis > 2) throw std::invalid_argument("Lattice constraint axis must be 0, 1, or 2");
        if (!atom.cell_.allFinite() || !std::isfinite(atom.cell_.row(axis).squaredNorm()) || atom.cell_.row(axis).squaredNorm() <= 0)
            throw std::invalid_argument("Lattice constraint requires a finite nonzero lattice vector");
    }
    void check_pair(const Atoms &atom, int a, int b) {
        check_axis(atom, a); check_axis(atom, b);
        if (a == b) throw std::invalid_argument("Lattice constraint axes must differ");
    }
    void check_target(const regulator &reg, bool angle = false) {
        if (!std::isfinite(reg.g0) || (angle ? std::abs(reg.g0) >= 1 : reg.g0 <= 0))
            throw std::invalid_argument("Invalid lattice constraint target");
    }
    }
    namespace {
    Eigen::RowVector3d atomic_position(const Atoms &atom, int id) {
        if (!atom.pos_[id].allFinite()) throw std::invalid_argument("Atomic constraint requires finite positions");
        return atom.cartesian ? atom.pos_[id] : (atom.pos_[id] * atom.cell_).eval();
    }
    Eigen::RowVector3i initial_image(const Atoms &source, int from, int to) {
        Atoms atom = source;
        const Eigen::RowVector3d delta = atomic_position(atom, to) - atomic_position(atom, from);
        const Eigen::RowVector3d translation = atom.get_rab(from, to) - delta;
        const int rank = int(atom.pbc_[0]) + int(atom.pbc_[1]) + int(atom.pbc_[2]);
        Eigen::RowVector3i image = Eigen::RowVector3i::Zero();
        if (!rank) return image;
        Eigen::MatrixXd basis(rank, 3);
        int row = 0;
        for (int a = 0; a < 3; ++a) if (atom.pbc_[a]) basis.row(row++) = atom.cell_.row(a);
        const Eigen::VectorXd shifts = (basis * basis.transpose()).fullPivLu().solve(basis * translation.transpose());
        row = 0;
        for (int a = 0; a < 3; ++a) if (atom.pbc_[a]) {
            const double value = std::round(shifts[row++]);
            if (!std::isfinite(value) || value < std::numeric_limits<int>::min() || value > std::numeric_limits<int>::max())
                throw std::invalid_argument("Atomic constraint image exceeds integer range");
            image[a] = static_cast<int>(value);
        }
        if (!(image.cast<double>() * atom.cell_).isApprox(translation, 1e-10) &&
            (image.cast<double>() * atom.cell_ - translation).norm() > 1e-10 * std::max(1.0, delta.norm()))
            throw std::invalid_argument("Atomic constraint image cannot be reconstructed");
        return image;
    }
    }

    AtomicGeometryEvaluation evaluate_atomic_geometry(const regulator &reg, const Atoms &atom) {
        const bool distance = reg.regulator_t == regulator::DISTANCE;
        if (!reg.atomic_geometry_bound || (!distance && reg.regulator_t != regulator::ANGLE))
            throw std::invalid_argument("Atomic geometry constraint has not been bound");
        const auto &ids = reg.integer_specs;
        if (ids.size() != (distance ? 2 : 3)) throw std::invalid_argument("Invalid atomic constraint index count");
        for (int id : ids) if (id < 0 || id >= atom.size())
            throw std::invalid_argument("Atomic constraint atom index out of bounds");
        for (const auto &image : reg.atomic_images) for (int a = 0; a < 3; ++a)
            if (!atom.pbc_[a] && image[a] != 0)
                throw std::invalid_argument("Atomic constraint image crosses a nonperiodic direction");
        const int center = distance ? ids[0] : ids[1];
        const int end = distance ? ids[1] : ids[0];
        const Eigen::RowVector3d u = atomic_position(atom, end) - atomic_position(atom, center)
            + reg.atomic_images[0].cast<double>() * atom.cell_;
        const double a2 = u.squaredNorm();
        if (!std::isfinite(a2) || a2 <= std::numeric_limits<double>::min())
            throw std::runtime_error("Atomic constraint has a zero or unrepresentable bond length");
        AtomicGeometryEvaluation out;
        out.cartesian_gradients.resize(distance ? 2 : 3);
        if (distance) {
            out.residual = a2 - reg.g0;
            out.physical_value = std::sqrt(a2);
            out.cartesian_gradients[0] = -2 * u;
            out.cartesian_gradients[1] = 2 * u;
            out.explicit_cell_gradient = reg.atomic_images[0].cast<double>().transpose() * (2 * u);
        } else {
            const Eigen::RowVector3d v = atomic_position(atom, ids[2]) - atomic_position(atom, center)
                + reg.atomic_images[1].cast<double>() * atom.cell_;
            const double b2 = v.squaredNorm();
            if (!std::isfinite(b2) || b2 <= std::numeric_limits<double>::min())
                throw std::runtime_error("Atomic angle constraint has a zero or unrepresentable arm");
            const double cosine = u.normalized().dot(v.normalized());
            if (!std::isfinite(cosine) || u.normalized().cross(v.normalized()).squaredNorm() <= 64 * std::numeric_limits<double>::epsilon() * std::numeric_limits<double>::epsilon())
                throw std::runtime_error("Atomic angle constraint has collinear arms");
            const Eigen::RowVector3d hu = v / std::sqrt(a2) / std::sqrt(b2) - cosine * u / a2;
            const Eigen::RowVector3d hv = u / std::sqrt(a2) / std::sqrt(b2) - cosine * v / b2;
            out.residual = cosine - reg.g0;
            out.physical_value = std::acos(std::clamp(cosine, -1.0, 1.0)) * 180 / std::acos(-1.0);
            out.cartesian_gradients[0] = hu;
            out.cartesian_gradients[1] = -hu - hv;
            out.cartesian_gradients[2] = hv;
            out.explicit_cell_gradient = reg.atomic_images[0].cast<double>().transpose() * hu
                + reg.atomic_images[1].cast<double>().transpose() * hv;
        }
        if (!std::isfinite(out.residual) || !std::isfinite(out.physical_value) || !out.explicit_cell_gradient.allFinite())
            throw std::runtime_error("Atomic constraint geometry is not finite");
        for (const auto &gradient : out.cartesian_gradients)
            if (!gradient.allFinite()) throw std::runtime_error("Atomic constraint gradient is not finite");
        return out;
    }

    void atomic_geometry_setup(regulator &requested, const Atoms &atom) {
        regulator reg = requested;
        const bool distance = reg.regulator_t == regulator::DISTANCE;
        if ((!distance && reg.regulator_t != regulator::ANGLE) || reg.integer_specs.size() != (distance ? 2 : 3))
            throw std::invalid_argument("Atomic distance/angle constraint requires two/three atom indices");
        for (size_t i = 0; i < reg.integer_specs.size(); ++i) {
            const int id = reg.integer_specs[i];
            if (id < 0 || id >= atom.size()) throw std::invalid_argument("Atomic constraint atom index out of bounds");
            for (size_t j = 0; j < i; ++j)
                if (id == reg.integer_specs[j]) throw std::invalid_argument("Atomic constraint atom indices must differ");
            (void)atomic_position(atom, id);
        }
        if (!atom.cell_.allFinite()) throw std::invalid_argument("Atomic constraint requires a finite cell");
        if (!reg.atomic_geometry_bound) {
            reg.atomic_images[0] = initial_image(atom, distance ? reg.integer_specs[0] : reg.integer_specs[1], distance ? reg.integer_specs[1] : reg.integer_specs[0]);
            if (!distance) reg.atomic_images[1] = initial_image(atom, reg.integer_specs[1], reg.integer_specs[2]);
            reg.atomic_geometry_bound = true;
            const auto initial = evaluate_atomic_geometry(reg, atom);
            if (!reg.to_target_g0) reg.double_target = initial.physical_value;
            const double target = reg.double_target;
            if (!std::isfinite(target) || target <= 0 || (!distance && target >= 180))
                throw std::invalid_argument("Atomic constraint target must be positive distance or angle strictly between 0 and 180 degrees");
            reg.g0 = distance ? target * target : std::cos(target * std::acos(-1.0) / 180);
            if (!std::isfinite(reg.g0) || (distance ? reg.g0 <= 0 : std::abs(reg.g0) >= 1))
                throw std::invalid_argument("Atomic constraint target is not representable");
        }
        (void)evaluate_atomic_geometry(reg, atom);
        requested = std::move(reg);
    }

    constexpr int l11 = 0, l12 = 1, l13 = 2, l21 = 3, l22 = 4, l23 = 5, l31 = 6, l32 = 7, l33 = 8;
    #define L11 cell[l11]
    #define L21 cell[l21]
    #define L31 cell[l31]
    #define L12 cell[l12]
    #define L22 cell[l22]
    #define L32 cell[l32]
    #define L13 cell[l13]
    #define L23 cell[l23]
    #define L33 cell[l33]
    // use fix_vol as possible if strict volume constraint is not required
    // This constraint is highly accurate, fix_vol only approximatly constraints the volume
    // However, this constraint is hard to converge.
    void constant_volume(double *cell, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom,
                         int *ids,  
                         void *pdata) {
        // Support the legacy atom-only call as well as an explicit raw9 cell.
        Eigen::Matrix3d lattice;
        if (nx == 0 && patom && g && f && ng == 9 && std::isfinite(g0)) lattice = patom->cell_;
        else {
            check_arguments(cell, nx, g, ng, f, g0, 9);
            for (int i = 0; i < 9; ++i) lattice(i / 3, i % 3) = cell[i];
        }
        const double vol = lattice.determinant();
        if (!lattice.allFinite() || !std::isfinite(vol) || vol <= 1e-14)
            throw std::invalid_argument("Volume constraint requires a right-handed nonsingular cell");
        *f = vol - g0;
        const Eigen::Matrix3d derivative = vol * lattice.inverse().transpose();
        for (int i = 0; i < 9; ++i) g[i] = derivative(i / 3, i % 3);
    }
    void constant_volume_setup(regulator &reg, Atoms &atom) {
        if (!atom.cell_.allFinite() || !std::isfinite(atom.get_volume()) || atom.get_volume() <= 1e-14)
            throw std::invalid_argument("Volume constraint requires a right-handed nonsingular cell");
        reg.inputs.resize(9);
        std::iota(reg.inputs.begin(), reg.inputs.end(), 0);
        reg.outputs = reg.inputs;
        reg.x.resize(9);
        reg.g.resize(9);
        if (!reg.to_target_g0) reg.g0 = atom.get_volume();
        check_target(reg);
        reg.compute = constant_volume;
    }
    void constant_vector(double *vec, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom,
                         int *ids,  
                         void *pdata) {
        check_arguments(vec, nx, g, ng, f, g0, 3);
        std::fill(g, g + ng, 0.0);
        double a2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
        *f = a2 - g0;
        for (int i = 0; i < 3; ++i)
            g[i] = 2*vec[i];
        Message(fmt::format("Lattice vector constraint: current a2 = {}, ref = {}, g = {}", a2, g0, *f));
    }
    void constant_vector_setup(regulator &reg, Atoms &atom, int l) {
        check_axis(atom, l);
        reg.x.resize(3);
        reg.g.resize(3);
        if (!reg.to_target_g0)
            reg.g0= atom.cell_.row(l).squaredNorm();
        check_target(reg);
        reg.compute = constant_vector;
        reg.inputs.resize(3);
        reg.inputs[0] = l*3 + 0;
        reg.inputs[1] = l*3 + 1;
        reg.inputs[2] = l*3 + 2;
        reg.outputs.resize(3);
        reg.outputs = reg.inputs;
    }
    // a,b | a,c | b,c, in order
    void constant_angle(double *cell, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom,
                         int *ids,  
                         void *pdata) {
        check_arguments(cell, nx, g, ng, f, g0, 6);
        std::fill(g, g + ng, 0.0);
        double adotb = L11 * L21 + L12 * L22 + L13 * L23;
        double a = sqrt(L11*L11 + L12*L12 + L13*L13);
        double b = sqrt(L21*L21 + L22*L22 + L23*L23);
        if (!std::isfinite(a) || !std::isfinite(b) || a <= 0 || b <= 0)
            throw std::invalid_argument("Angle constraint requires finite nonzero vectors");
        double ab = a*b;
        double a3b = a*a*a*b;
        double ab3 = a*b*b*b;
        double cosval = adotb / a / b;
        *f = cosval - g0;
        double fab  = 1./ab;
        double fa3b = 1./a3b;
        double fab3 = 1./ab3;
        g[l11] = L21*fab - adotb*L11*fa3b;
        g[l12] = L22*fab - adotb*L12*fa3b;
        g[l13] = L23*fab - adotb*L13*fa3b;
        g[l21] = L11*fab - adotb*L21*fab3;
        g[l22] = L12*fab - adotb*L22*fab3;
        g[l23] = L13*fab - adotb*L23*fab3;
        Message(fmt::format("Lattice angle constraint: current a2 = {}, ref = {}, g = {}", cosval, g0, *f));
    }
    void constant_angle_setup(regulator &reg, Atoms &atom, int l1, int l2) {
        check_pair(atom, l1, l2);
        reg.x.resize(6);
        reg.g.resize(6);
        reg.inputs.resize(6);
        reg.outputs.resize(6);
        reg.inputs[0] = l1 * 3 + 0;
        reg.inputs[1] = l1 * 3 + 1;
        reg.inputs[2] = l1 * 3 + 2;
        reg.inputs[3] = l2 * 3 + 0;
        reg.inputs[4] = l2 * 3 + 1;
        reg.inputs[5] = l2 * 3 + 2;
        reg.outputs = reg.inputs;
        if (!reg.to_target_g0)
            reg.g0 = atom.cell_.row(l1).dot(atom.cell_.row(l2))/ atom.cell_.row(l1).norm() / atom.cell_.row(l2).norm();
        check_target(reg, true);
        reg.compute = constant_angle;
    }
    // g = a2/b2 - a02/b02, might not be a good loss func
    void couple_vector(double *cell, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom,
                         int *ids,  
                         void *pdata) {
        check_arguments(cell, nx, g, ng, f, g0, 6);
        std::fill(g, g + ng, 0.0);
        double a2 = L11*L11 + L12*L12 + L13*L13;
        double b2 = L21*L21 + L22*L22 + L23*L23;
        if (!std::isfinite(a2) || !std::isfinite(b2) || b2 <= 0)
            throw std::invalid_argument("Ratio constraint requires finite vectors and a nonzero denominator");
        double b4 = b2 * b2;
        double invb2 = 1.0 / b2;
        double a2invb4 = a2 / b4;
        *f = a2 * invb2 - g0;
        g[l11] = 2*L11*invb2;
        g[l12] = 2*L12*invb2;
        g[l13] = 2*L13*invb2;
        g[l21] = -2*L21*a2invb4;
        g[l22] = -2*L22*a2invb4;
        g[l23] = -2*L23*a2invb4;
    }
    void couple_vector_setup(regulator &reg, Atoms &atom, int l1, int l2) {
        check_pair(atom, l1, l2);
        reg.x.resize(6);
        reg.g.resize(6);
        reg.inputs.resize(6);
        reg.outputs.resize(6);
        reg.inputs[0] = l1 * 3 + 0;
        reg.inputs[1] = l1 * 3 + 1;
        reg.inputs[2] = l1 * 3 + 2;
        reg.inputs[3] = l2 * 3 + 0;
        reg.inputs[4] = l2 * 3 + 1;
        reg.inputs[5] = l2 * 3 + 2;
        reg.outputs = reg.inputs;
        if (!reg.to_target_g0)  
            reg.g0 = atom.cell_.row(l1).squaredNorm() / atom.cell_.row(l2).squaredNorm();
        check_target(reg);
        reg.compute = couple_vector;
    }
    void mindist(double *x, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom,
                         int *ids,  
                         void *pdata) {
        const double min_dist2 = g0;
        const int nats = nx / 3;
        // std::vector<int> ai, aj;
        double currd2;
        Atoms geometry;
        geometry.cell_ = patom->cell_;
        geometry.pbc_ = patom->pbc_;
        geometry.cartesian = true;
        geometry.pos_.resize(nats);
        for (int i = 0; i < nats; ++i)
            geometry.pos_[i] = Eigen::RowVector3d(x[3*i], x[3*i+1], x[3*i+2]) * geometry.cell_;
        Eigen::RowVector3d cdiff;
        *f = 0;
        std::fill(g, g + ng, 0.0);
        double currmind2 = 1e19;
        double currmind = 0.0;
        for (int i = 0; i < nats; ++i) {
            for (int j = i+1; j < nats; ++j) {
                cdiff = geometry.get_rab(j, i);
                currd2 = cdiff.squaredNorm();
                if (currd2 < min_dist2) {
                    // ai.push_back(i);
                    // aj.push_back(j);
                    *f += currd2 - min_dist2;
                    for (int a = 0; a < 3; ++a) {
                        Eigen::RowVector3d dcar_di = patom->cell_.row(a);
                        double df_dxi = 2.0 * cdiff.dot(dcar_di);
                        g[3*i+a] +=  df_dxi;
                        g[3*j+a] += -df_dxi;
                    }
                }
                if (currd2 < currmind2) {
                    currmind2 = currd2;
                    currmind = sqrt(currmind2);
                }
            }
        }
        Message(fmt::format("Mindist: current mindist is {}, g = {}", currmind, *f));
    }
    void mindist_setup(regulator &reg, Atoms &atom, double min_dist, int xoffset) {
        reg.g0 = min_dist * min_dist;
        reg.inputs.resize(reg.integer_specs.size()*3);
        reg.outputs.resize(reg.inputs.size());
        reg.x.resize(reg.inputs.size());
        reg.g.resize(reg.outputs.size());
        for (int i = 0; i < reg.integer_specs.size(); ++i) {
            for (int a = 0; a < 3; ++a) {
                reg.inputs[3*i+a] = xoffset + 3 * reg.integer_specs[i] + a;
                reg.outputs[3*i+a] = reg.inputs[3*i+a];
            }
        }
        reg.compute = mindist;
    }
    // if car_mode = true
    void mindist_f(double *x, int nx, double *g, int ng, double *f, 
                         double g0,
                         Atoms *patom,
                         int *ids,  
                         void *pdata) {
        const double min_dist2 = g0;
        const int nats = nx / 3;
        // std::vector<int> ai, aj;
        double currd2;
        Atoms geometry;
        geometry.cell_ = patom->cell_;
        geometry.pbc_ = patom->pbc_;
        geometry.cartesian = true;
        geometry.pos_.resize(nats);
        for (int i = 0; i < nats; ++i)
            geometry.pos_[i] = Eigen::RowVector3d(x[3*i], x[3*i+1], x[3*i+2]);
        Eigen::RowVector3d cdiff;
        *f = 0;
        std::fill(g, g + ng, 0.0);
        double currmind2 = 1e19;
        double currmind = 0.0;
        for (int i = 0; i < nats; ++i) {
            for (int j = i+1; j < nats; ++j) {
                cdiff = geometry.get_rab(j, i);
                currd2 = cdiff.squaredNorm();
                if (currd2 < min_dist2) {
                    // ai.push_back(i);
                    // aj.push_back(j);
                    *f += currd2 - min_dist2;
                    for (int a = 0; a < 3; ++a) {
                        g[3*i+a] +=  2.0 * cdiff[a];
                        g[3*j+a] += -2.0 * cdiff[a];
                    }
                }
                if (currd2 < currmind2) {
                    currmind2 = currd2;
                    currmind = sqrt(currmind2);
                }
            }
        }
        Message(fmt::format("Mindist: current mindist is {}, g = {}", currmind, *f));
    }
    void mindist_f_setup(regulator &reg, Atoms &atom, double min_dist, int xoffset) {
        reg.g0 = min_dist * min_dist;
        reg.inputs.resize(reg.integer_specs.size()*3);
        reg.outputs.resize(reg.inputs.size());
        reg.x.resize(reg.inputs.size());
        reg.g.resize(reg.outputs.size());
        for (int i = 0; i < reg.integer_specs.size(); ++i) {
            for (int a = 0; a < 3; ++a) {
                reg.inputs[3*i+a] = xoffset + 3 * reg.integer_specs[i] + a;
                reg.outputs[3*i+a] = reg.inputs[3*i+a];
            }
        }
        reg.compute = mindist_f;
    }
}
