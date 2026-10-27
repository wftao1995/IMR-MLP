// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#ifndef IMRMLP_NEIGHBORHOODS_BASE_H
#define IMRMLP_NEIGHBORHOODS_BASE_H

#include "structures.h"
#include "tools/tools.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

// CRTP base for all Neighborhoods* classes.
// Extracts the shared extent() / reset() / init() pipeline.
// Derived classes only need to implement construct().

template <typename Derived, typename BaseClass>
class NeighborhoodsBase : public BaseClass
{
protected:
    std::vector<int> indices_orig;           // tmp
    std::vector<Eigen::RowVector3d> pos;      // tmp
#ifdef BOXED_NEBH
    std::vector<std::vector<std::vector<std::vector<int>>>> boxes; // tmp
    int boxle_[3];
#endif

    // Enumerate translations in an orthogonal frame. Each integer range is
    // conditioned on the outer indices, so skew cells do not require scanning
    // the Cartesian product of three independent inverse-cell bounds.
    void append_images(const auto &atom, double cutoff,
                       const Eigen::RowVector3d &lo, const Eigen::RowVector3d &hi)
    {
        if (atom.is_nonperiodic()) return;
        const int size = pos.size();
        std::array<int, 3> axes{0, 1, 2};
        std::sort(axes.begin(), axes.end(), [&](int a, int b) {
            return atom.cell_.row(a).squaredNorm() < atom.cell_.row(b).squaredNorm();
        });
        Eigen::Matrix3d basis = Eigen::Matrix3d::Zero();
        int dimensions = 0;
        for (int axis : axes)
            if (atom.pbc_[axis]) basis.col(dimensions++) = atom.cell_.row(axis).transpose();
        const Eigen::HouseholderQR<Eigen::Matrix3d> qr(basis);
        const Eigen::Matrix3d rotation = qr.householderQ();
        const Eigen::Matrix3d upper = qr.matrixQR().template triangularView<Eigen::Upper>();
        const Eigen::RowVector3d extent = hi - lo + Eigen::RowVector3d::Constant(cutoff);
        const Eigen::RowVector3d bound = extent * rotation.cwiseAbs();
        Eigen::Vector3d image = Eigen::Vector3d::Zero();
        auto enumerate = [&](auto &&self, int axis) -> void {
            if (axis >= 0) {
                image[axis] = 0;
                const double center = -upper.row(axis).dot(image) / upper(axis, axis);
                const double radius = bound[axis] / std::abs(upper(axis, axis));
                const double slack = 64 * std::numeric_limits<double>::epsilon() *
                    std::max(1.0, std::abs(center) + radius);
                const double first = std::ceil(center - radius - slack);
                const double last = std::floor(center + radius + slack);
                if (!std::isfinite(first) || !std::isfinite(last) ||
                    first <= std::numeric_limits<int>::min() || last >= std::numeric_limits<int>::max())
                    throw std::length_error("Neighbor periodic image index exceeds integer range");
                for (int i = static_cast<int>(first); i <= last; ++i) {
                    image[axis] = i;
                    self(self, axis - 1);
                }
                return;
            }
            if (image.isZero()) return;
            const Eigen::RowVector3d translation = (basis * image).transpose();
            for (int i = 0; i < size; ++i) {
                const Eigen::RowVector3d point = pos[i] + translation;
                if ((point.array() < lo.array() - cutoff).any() ||
                    (point.array() > hi.array() + cutoff).any()) continue;
#ifndef BOXED_NEBH
                int j = 0;
                while (j < size && (pos[j] - point).squaredNorm() > cutoff * cutoff) ++j;
                if (j == size) continue;
#endif
                if (pos.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
                    throw std::length_error("Neighbor image count exceeds integer range");
                pos.emplace_back(point);
                indices_orig.emplace_back(i);
            }
        };
        enumerate(enumerate, dimensions - 1);
    }

    void reset_storage()
    {
        pos.clear();
        indices_orig.clear();
#ifdef BOXED_NEBH
        // swap with empty temporary to force immediate deallocation;
        // avoids a GCC 16 libstdc++ use-after-free when clear()+dtor interact
        // on the deeply-nested vector<vector<vector<vector<int>>>>.
        decltype(boxes)().swap(boxes);
#endif
    }

    void load_positions(auto &atom)
    {
        pos = atom.pos_;
        if (!atom.is_fully_periodic()) return;
        const Eigen::Matrix3d inv_cell=atom.cell_.inverse();
        for (auto& position:pos) {
            Eigen::RowVector3d fractional=position*inv_cell;
            for (int d=0;d<3;++d) fractional(d)-=std::floor(fractional(d));
            position=fractional*atom.cell_;
        }
    }

#ifdef BOXED_NEBH
    // ---- BOXED extent: 27-box spatial hashing -------------------------------
    void extent_impl(auto &atom, double cutoff_bound)
    {
        load_positions(atom);
        const int size = pos.size();
        Eigen::RowVector3d min_pos = pos.front(), max_pos = pos.front();
        for (const auto &p : pos) {
            min_pos = min_pos.cwiseMin(p); max_pos = max_pos.cwiseMax(p);
        }
        // Nonzero box extents also cover planar and isolated structures.
        for (int d = 0; d < 3; ++d)
            if (max_pos[d] - min_pos[d] < cutoff_bound) max_pos[d] = min_pos[d] + cutoff_bound;

        append_images(atom, cutoff_bound, min_pos, max_pos);

        // boxes
        double xd1 = max_pos[0] - min_pos[0];
        double xd2 = max_pos[1] - min_pos[1];
        double xd3 = max_pos[2] - min_pos[2];
        for (int d = 0; d < 3; ++d) {
            double count = std::floor((max_pos[d] - min_pos[d]) / cutoff_bound) + 2;
            if (!std::isfinite(count) || count >= std::numeric_limits<int>::max())
                throw std::length_error("Neighbor spatial box count exceeds integer range");
            boxle_[d] = static_cast<int>(count);
        }
#ifndef NEIGHBOR_ASSUME_ALLRIGHT
        if (boxle_[0] == 2) { boxle_[0] = 3; }
        if (boxle_[1] == 2) { boxle_[1] = 3; }
        if (boxle_[2] == 2) { boxle_[2] = 3; }
#endif
        boxes.resize(boxle_[0]);
        for (int i = 0; i < boxle_[0]; ++i)
        {
            boxes[i].resize(boxle_[1]);
            for (int j = 0; j < boxle_[1]; ++j)
                boxes[i][j].resize(boxle_[2]);
        }

        double tp1, tp2, tp3;
        int i1, i2, i3;
        for (int i = 0; i < size; ++i)
        {
            tp1 = pos[i][0] - min_pos[0];
            tp2 = pos[i][1] - min_pos[1];
            tp3 = pos[i][2] - min_pos[2];
            i1 = ceil(tp1 * (boxle_[0] - 2) / xd1);
            i2 = ceil(tp2 * (boxle_[1] - 2) / xd2);
            i3 = ceil(tp3 * (boxle_[2] - 2) / xd3);
            if (i1 == 0) i1 = 1;
            if (i2 == 0) i2 = 1;
            if (i3 == 0) i3 = 1;
            if (i1 == boxle_[0] - 1) --i1;
            if (i2 == boxle_[1] - 1) --i2;
            if (i3 == boxle_[2] - 1) --i3;
            boxes[i1][i2][i3].emplace_back(i);
        }
        for (int i = size; i < pos.size(); ++i)
        {
            tp1 = pos[i][0] - min_pos[0];
            tp2 = pos[i][1] - min_pos[1];
            tp3 = pos[i][2] - min_pos[2];
            i1 = ceil(tp1 * (boxle_[0] - 2) / xd1);
            i2 = ceil(tp2 * (boxle_[1] - 2) / xd2);
            i3 = ceil(tp3 * (boxle_[2] - 2) / xd3);
            if (i1 < 0) i1 = 0;
            if (i2 < 0) i2 = 0;
            if (i3 < 0) i3 = 0;
            if (i1 >= boxle_[0]) i1 = boxle_[0] - 1;
            if (i2 >= boxle_[1]) i2 = boxle_[1] - 1;
            if (i3 >= boxle_[2]) i3 = boxle_[2] - 1;
            boxes[i1][i2][i3].emplace_back(i);
        }
    }
#else  // !BOXED_NEBH
    // ---- non-BOXED extent: O(N²) scan with fine validation ---------------
    void extent_impl(auto &atom, double cutoff_bound)
    {
        load_positions(atom);
        const int size = pos.size();

        Eigen::Vector3d max_pos = pos[0];
        Eigen::Vector3d min_pos = pos[0];
        for (int i = 0; i < size; ++i)
        {
            for (int j = 0; j < 3; ++j)
            {
                if (min_pos[j] > pos[i][j]) min_pos[j] = pos[i][j];
                if (max_pos[j] < pos[i][j]) max_pos[j] = pos[i][j];
            }
        }

        append_images(atom, cutoff_bound, min_pos.transpose(), max_pos.transpose());
    }
#endif // BOXED_NEBH

    // ---- init pipelines ---------------------------------------------------
    // Requires valid atom data and cutoffs. After an exception, reinitialize
    // before using the neighbor list again.
    void init_single(auto &atom, double cutoff)
    {
        reset_storage();
        this->nbhs.clear();
        if (atom.size() == 0) return;
        indices_orig.resize(atom.size());
        for (int i = 0; i < atom.size(); ++i) indices_orig[i] = i;
        extent_impl(atom, cutoff);
        static_cast<Derived *>(this)->construct(atom, cutoff);
        reset_storage();
    }

    void init_double(auto &atom, double cutoff_inner, double cutoff_outer)
    {
        reset_storage();
        this->nbhs.clear();
        if (atom.size() == 0) return;
        indices_orig.resize(atom.size());
        for (int i = 0; i < atom.size(); ++i) indices_orig[i] = i;
        extent_impl(atom, cutoff_outer);
        static_cast<Derived *>(this)->construct(atom, cutoff_inner, cutoff_outer);
        reset_storage();
    }
};

#endif // IMRMLP_NEIGHBORHOODS_BASE_H
