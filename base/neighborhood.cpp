// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/*******
 * The atom object should be wrapped before boxing, i won't do this here
 *
 * Method to calculate neighborhood atoms
 *
 * Written by wftao (wftao1995@gmail.com)
 *
 * TODO: check 177 - 182, --- use floor or ceil ---
 *
 * ******/

#include "neighborhood.h"
#include "tools/tools.h"
#include "tools/chem.h"
#include <type_traits>

#ifdef DEBUG_TIME
#include <chrono>
#endif

#ifndef BOXED_NEBH
void Neighborhoods::construct(const Atoms &atom, const double cutoff)
{
    const int size = atom.size();
    const double cutoffsq = cutoff * cutoff;

    if (!nbhs.empty())
    {
        nbhs.clear();
    }
    nbhs.resize(size);

    for (int i = 0; i < size; ++i)
    {
        for (int j = 0; j < pos.size(); ++j)
        {
            Eigen::RowVector3d rij = pos[j] - pos[i];
            double dijsq = rij(0) * rij(0) + rij(1) * rij(1) + rij(2) * rij(2);
            if ((dijsq < cutoffsq) && (dijsq > 1e-8) && (i <= indices_orig[j]))
            {
                // for (i > orig_atom_inds[j]) this connection should be already treated
                Neighborhood &nebi = nbhs[i]; //nbhs[i] and nbhs[orig[i]] are the same
                double dij = sqrt(dijsq);

                nebi.nbhcount_++;
                nebi.indices_.emplace_back(indices_orig[j]);
                nebi.pos_.emplace_back(rij);
                nebi.distances_.emplace_back(dij);
                nebi.chemmap_.emplace_back(atom.chemmap_[indices_orig[j]]);

                if (i != indices_orig[j])
                {
                    Neighborhood &nebj = nbhs[indices_orig[j]];
                    nebj.nbhcount_++;
                    nebj.indices_.emplace_back(i);
                    nebj.pos_.emplace_back(-rij);
                    nebj.distances_.emplace_back(dij);
                    nebj.chemmap_.emplace_back(atom.chemmap_[i]);
                }
            }
        }
        nbhs[i].centertype = atom.chemmap_[i];
#ifdef _ENABLE_FORCE_ONLY_PREDICTION_
        nbhs[i].force_ = atom.forces_[i];
#endif
    }
}
#else
// devide the lattice into boxes, and even sub boxes for extremely large latttices
// use cutoff as box length, eg. use ceil(latprarm/cutoff) as divide
// it's very effcient for very large systems compared to previous one.
// however, for small systems it's sightly ineffcient.
// consider 27 boxes for each box
void Neighborhoods::construct(const Atoms &atom, const double cutoff)
{
    const int size = atom.size();
    const double cutoffsq = cutoff * cutoff;

    if (!nbhs.empty())
    {
        nbhs.clear();
    }
    nbhs.resize(size);

    // consider boxes form 1 to boxle-2
    for (int i = 1; i < boxle_[0] - 1; ++i)
    {
        for (int j = 1; j < boxle_[1] - 1; ++j)
        {
            for (int k = 1; k < boxle_[2] - 1; ++k)
            {
                for (int s = 0; s < boxes[i][j][k].size(); s++)
                {
                    int indi = boxes[i][j][k][s];
                    if (indi >= size)
                        continue;
                    nbhs[indi].centertype = atom.chemmap_[indi];
                    for (int l1 = -1; l1 <= 1; l1++)
                    {
                        for (int l2 = -1; l2 <= 1; l2++)
                        {
                            for (int l3 = -1; l3 <= 1; l3++)
                            {
                                for (int t = 0; t < boxes[i + l1][j + l2][k + l3].size(); t++)
                                {
                                    int indj = boxes[i + l1][j + l2][k + l3][t];
                                    Eigen::RowVector3d rij = pos[indj] - pos[indi];
                                    const double dijsq = rij(0) * rij(0) + rij(1) * rij(1) + rij(2) * rij(2);
                                    if ((dijsq < cutoffsq) && (indi != indj) && (indi <= indices_orig[indj]))
                                    {
                                        Neighborhood &nebi = nbhs[indi];
                                        const double dij = sqrt(dijsq);

                                        nebi.nbhcount_++;
                                        nebi.indices_.emplace_back(indices_orig[indj]);
                                        nebi.pos_.emplace_back(rij);
                                        nebi.distances_.emplace_back(dij);
                                        nebi.chemmap_.emplace_back(atom.chemmap_[indices_orig[indj]]);
                                        if (indi != indices_orig[indj])
                                        {
                                            Neighborhood &nebj = nbhs[indices_orig[indj]];
                                            nebj.nbhcount_++;
                                            nebj.indices_.emplace_back(indi);
                                            nebj.pos_.emplace_back(-rij);
                                            nebj.distances_.emplace_back(dij);
                                            nebj.chemmap_.emplace_back(atom.chemmap_[indi]);
                                        }
                                    }
                                }
                            }
                        }
                    }
#ifdef _ENABLE_FORCE_ONLY_PREDICTION_
                    nbhs[indi].force_ = atom.forces_[indi];
#endif
                }
            }
        }
    }
}
#endif //BOXED_NEBH

void Neighborhoods::init(const Atoms &atom, const double cutoff)
{
    this->init_single(atom, cutoff);
}

Neighborhoods::Neighborhoods(const Atoms &atom, const double cutoff)
{
    init(atom, cutoff);
}

std::size_t Neighborhoods::owned_memory_bytes(bool copied) const
{
    std::size_t bytes = sizeof(Neighborhoods);
    const auto vector = [&](const auto &v) {
        using Element = typename std::decay_t<decltype(v)>::value_type;
        const std::size_t count = copied ? v.size() : v.capacity();
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(Element) ||
            count * sizeof(Element) > std::numeric_limits<std::size_t>::max() - bytes)
            throw std::length_error("Neighborhood owned storage size overflow");
        bytes += count * sizeof(Element);
    };
    vector(nbhs);
    for (const auto &site : nbhs) {
        vector(site.indices_); vector(site.chemmap_); vector(site.pos_); vector(site.distances_);
    }
    vector(indices_orig); vector(pos);
#ifdef BOXED_NEBH
    vector(boxes);
    for (const auto &x : boxes) {
        vector(x);
        for (const auto &y : x) {
            vector(y);
            for (const auto &z : y) vector(z);
        }
    }
#endif
    return bytes;
}

std::size_t Neighborhoods::construction_memory_bound(const Atoms &atom,
    double cutoff, std::size_t budget)
{
    // Follow the standard builder's wrapping, QR image ranges and pair filter,
    // but count before allocating any image, box or per-site array.
    if (!budget || !std::isfinite(cutoff) || cutoff <= 0 || !atom.cartesian ||
        atom.size() < 0 || atom.chemmap_.size() != std::size_t(atom.size()))
        throw std::invalid_argument("Invalid bounded neighborhood geometry, cutoff or budget");
    const std::size_t n = atom.size();
    if (!n) return 0;
    const double cutoffsq = cutoff * cutoff;
    if (!std::isfinite(cutoffsq) || cutoffsq <= 0)
        throw std::invalid_argument("Bounded neighborhood squared cutoff must be finite and positive");
    auto add = [](std::size_t a, std::size_t b) {
        if (b > std::numeric_limits<std::size_t>::max() - a)
            throw std::length_error("Bounded neighborhood byte count overflow");
        return a + b;
    };
    auto mul = [](std::size_t a, std::size_t b) {
        if (a && b > std::numeric_limits<std::size_t>::max() / a)
            throw std::length_error("Bounded neighborhood byte count overflow");
        return a * b;
    };
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Identity();
    if (atom.is_fully_periodic()) {
        inverse = atom.cell_.inverse();
        if (!inverse.allFinite()) throw std::invalid_argument("Invalid periodic cell in bounded neighborhood");
    }
    auto position = [&](std::size_t i) -> Eigen::RowVector3d {
        Eigen::RowVector3d p = atom.pos_[i];
        if (atom.is_fully_periodic()) {
            Eigen::RowVector3d fractional = p * inverse;
            for (int d = 0; d < 3; ++d) fractional[d] -= std::floor(fractional[d]);
            p = fractional * atom.cell_;
        }
        if (!p.allFinite()) throw std::invalid_argument("Nonfinite bounded neighborhood position");
        return p;
    };
    Eigen::RowVector3d lo = position(0), hi = lo;
    for (std::size_t i = 1; i < n; ++i) {
        const auto p = position(i); lo = lo.cwiseMin(p); hi = hi.cwiseMax(p);
    }
    std::size_t boxes_bytes = 0;
#ifdef BOXED_NEBH
    std::size_t counts[3];
    for (int d = 0; d < 3; ++d) {
        if (hi[d] - lo[d] < cutoff) hi[d] = lo[d] + cutoff;
        const double extent = hi[d] - lo[d];
        if (!std::isfinite(extent) || extent <= 0)
            throw std::invalid_argument("Bounded neighborhood padded box extent must be finite and positive");
        const double count = std::floor(extent / cutoff) + 2;
        if (!std::isfinite(count) || count >= std::numeric_limits<int>::max())
            throw std::length_error("Bounded neighborhood box index overflow");
        counts[d] = std::size_t(count);
#ifndef NEIGHBOR_ASSUME_ALLRIGHT
        counts[d] = std::max(std::size_t{3}, counts[d]);
#endif
    }
    boxes_bytes = mul(counts[0], sizeof(std::vector<std::vector<std::vector<int>>>));
    boxes_bytes = add(boxes_bytes, mul(mul(counts[0], counts[1]), sizeof(std::vector<std::vector<int>>)));
    boxes_bytes = add(boxes_bytes, mul(mul(mul(counts[0], counts[1]), counts[2]), sizeof(std::vector<int>)));
#endif
    std::size_t points = 0, edges = 0;
    auto bound = [&] {
        // Geometric vector growth may retain the old allocation during a
        // doubling. Three times the element count bounds that transient peak.
        std::size_t edge_capacity = edges;
#ifdef NBH_PREALLOC
        edge_capacity = add(edge_capacity, mul(n, PRE_ALLOC_NBH_SIZE));
#endif
        auto bytes = add(mul(n, sizeof(Neighborhood)), boxes_bytes);
        bytes = add(bytes, mul(mul(points, 3), sizeof(Eigen::RowVector3d) + sizeof(int)));
#ifdef BOXED_NEBH
        bytes = add(bytes, mul(mul(points, 3), sizeof(int)));
#endif
        bytes = add(bytes, mul(mul(edge_capacity, 3), sizeof(Eigen::RowVector3d) + sizeof(double) + 2 * sizeof(int)));
        if (bytes > budget) throw std::length_error("Bounded neighborhood construction exceeds the HVP host workspace budget");
        return bytes;
    };
    (void)bound(); // Reject enormous BOXED extents before any image scan.
    auto accept = [&](const Eigen::RowVector3d &p, std::size_t original, bool image) {
        points = add(points, 1);
        if (points >= std::size_t(std::numeric_limits<int>::max()))
            throw std::length_error("Bounded neighborhood image index overflow");
        (void)bound();
        for (std::size_t i = 0; i <= original; ++i) {
            const double distance = (p - position(i)).squaredNorm();
#ifdef BOXED_NEBH
            if (distance >= cutoffsq || (!image && i == original)) continue;
#else
            if (distance >= cutoffsq || distance <= 1e-8) continue;
#endif
            edges = add(edges, i == original ? 1 : 2);
            if (edges > std::size_t(std::numeric_limits<int>::max() / 3))
                throw std::length_error("Bounded neighborhood packed edge index overflow");
            (void)bound();
        }
    };
    for (std::size_t i = 0; i < n; ++i) accept(position(i), i, false);
    if (!atom.is_nonperiodic()) {
        std::array<int, 3> axes{0, 1, 2};
        std::sort(axes.begin(), axes.end(), [&](int a, int b) {
            return atom.cell_.row(a).squaredNorm() < atom.cell_.row(b).squaredNorm();
        });
        Eigen::Matrix3d basis = Eigen::Matrix3d::Zero();
        int dimensions = 0;
        for (int axis : axes) if (atom.pbc_[axis]) basis.col(dimensions++) = atom.cell_.row(axis).transpose();
        const Eigen::HouseholderQR<Eigen::Matrix3d> qr(basis);
        const Eigen::Matrix3d rotation = qr.householderQ();
        const Eigen::Matrix3d upper = qr.matrixQR().template triangularView<Eigen::Upper>();
        const Eigen::RowVector3d extent = hi - lo + Eigen::RowVector3d::Constant(cutoff);
        const Eigen::RowVector3d limits = extent * rotation.cwiseAbs();
        Eigen::Vector3d image = Eigen::Vector3d::Zero();
        auto enumerate = [&](auto &&self, int axis) -> void {
            if (axis >= 0) {
                image[axis] = 0;
                const double center = -upper.row(axis).dot(image) / upper(axis, axis);
                const double radius = limits[axis] / std::abs(upper(axis, axis));
                const double slack = 64 * std::numeric_limits<double>::epsilon() * std::max(1., std::abs(center) + radius);
                const double first = std::ceil(center - radius - slack), last = std::floor(center + radius + slack);
                if (!std::isfinite(first) || !std::isfinite(last) || first <= std::numeric_limits<int>::min() || last >= std::numeric_limits<int>::max())
                    throw std::length_error("Bounded neighborhood periodic image index overflow");
                for (int i = int(first); i <= last; ++i) { image[axis] = i; self(self, axis - 1); }
                return;
            }
            if (image.isZero()) return;
            const Eigen::RowVector3d translation = (basis * image).transpose();
            for (std::size_t i = 0; i < n; ++i) {
                const Eigen::RowVector3d p = position(i) + translation;
                if ((p.array() < lo.array() - cutoff).any() || (p.array() > hi.array() + cutoff).any()) continue;
#ifndef BOXED_NEBH
                std::size_t j = 0;
                while (j < n && (position(j) - p).squaredNorm() > cutoffsq) ++j;
                if (j == n) continue;
#endif
                accept(p, i, true);
            }
        };
        enumerate(enumerate, dimensions - 1);
    }
    return bound();
}
