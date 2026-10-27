#pragma once

#include "base/anyneighborhoods.h"
#include <algorithm>
#include <cmath>
#include <vector>

// Internal image identities in one root's Cartesian reference frame. Edges
// come exclusively from the supplied graph, including any prior filtering.
namespace CommonNeighborGeometry {
struct Image {
    int source;
    Eigen::RowVector3d position;
};

inline double matching_tolerance(const Eigen::RowVector3d &a,
                                 const Eigen::RowVector3d &b) {
    return 1e-8 + 1e-10 * std::max(a.norm(), b.norm());
}

inline bool same_image(const Image &a, const Image &b) {
    return a.source == b.source &&
        (a.position - b.position).norm() <= matching_tolerance(a.position, b.position);
}

inline void append_unique(std::vector<Image> &images, const Image &image) {
    if (std::none_of(images.begin(), images.end(), [&](const Image &other) {
            return same_image(other, image);
        })) images.push_back(image);
}

inline std::vector<Image> neighbors(const AnyNeighborhoods &nbhs, const Image &from) {
    std::vector<Image> result;
    const auto &site = nbhs[from.source];
    result.reserve(site.nbhcount_);
    for (int j = 0; j < site.nbhcount_; ++j)
        append_unique(result, {site.indices_[j], from.position + site.pos_[j]});
    return result;
}

inline std::vector<Image> candidates(const AnyNeighborhoods &nbhs, int root, bool unbond) {
    const Image origin{root, Eigen::RowVector3d::Zero()};
    const auto direct = neighbors(nbhs, origin);
    std::vector<Image> result;
    auto append = [&](const Image &image) {
        if (!same_image(image, origin)) append_unique(result, image);
    };
    for (const auto &image : direct) append(image);
    if (unbond)
        for (const auto &image : direct)
            for (const auto &next : neighbors(nbhs, image)) append(next);
    return result;
}

inline std::vector<Image> common(const AnyNeighborhoods &nbhs, int root, const Image &pair) {
    const Image origin{root, Eigen::RowVector3d::Zero()};
    const auto direct = neighbors(nbhs, origin), translated = neighbors(nbhs, pair);
    std::vector<Image> result;
    for (const auto &image : direct) {
        if (same_image(image, origin) || same_image(image, pair)) continue;
        if (std::any_of(translated.begin(), translated.end(), [&](const Image &other) {
                return same_image(image, other);
            })) append_unique(result, image);
    }
    return result;
}

inline bool bonded(const AnyNeighborhoods &nbhs, const Image &from, const Image &to) {
    const auto &site = nbhs[from.source];
    for (int j = 0; j < site.nbhcount_; ++j)
        if (same_image({site.indices_[j], from.position + site.pos_[j]}, to)) return true;
    return false;
}

inline bool owns_pair(int root, const Image &pair) {
    if (pair.source != root) return root < pair.source;
    const Eigen::RowVector3d zero = Eigen::RowVector3d::Zero();
    if (same_image({root, zero}, pair)) return false;
    const double tolerance = matching_tolerance(pair.position, zero);
    for (int axis = 0; axis < 3; ++axis)
        if (std::abs(pair.position[axis]) > tolerance) return pair.position[axis] > 0.0;
    // A vector just outside the norm tolerance can have all three components
    // below it; it still needs exactly one owner among the opposite images.
    for (int axis = 0; axis < 3; ++axis)
        if (pair.position[axis] != 0.0) return pair.position[axis] > 0.0;
    return false;
}
} // namespace CommonNeighborGeometry
