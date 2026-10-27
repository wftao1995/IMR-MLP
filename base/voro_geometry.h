// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "neighborhood_voro.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace voro_geometry
{
struct Face
{
    int neighbor;
    Eigen::RowVector3d normal, displacement;
    double offset, area;
};

// Keep the face plane and its periodic image together: a neighbor ID alone
// cannot distinguish opposite images of the same atom in a small cell.
inline std::vector<Face> faces(voro::voronoicell_neighbor &cell, const Atoms &atoms,
                               int center, const std::map<int, double> *radii)
{
    std::vector<int> neighbors, face_vertices;
    std::vector<double> normals, vertices, areas;
    cell.neighbors(neighbors);
    cell.normals(normals);
    cell.vertices(vertices);
    cell.face_vertices(face_vertices);
    cell.face_areas(areas);
    const Eigen::Matrix3d inverse = atoms.cell_.inverse();
    const double geometry_tolerance_squared = 1e-12 * std::max(1., atoms.cell_.squaredNorm());
    std::vector<Face> result;
    int vertex_offset = 0;
    for (int i = 0; i < neighbors.size(); ++i)
    {
        Face face;
        face.normal = Eigen::Map<const Eigen::RowVector3d>(&normals[3 * i]);
        const int vertex = face_vertices[vertex_offset + 1];
        face.offset = face.normal.dot(Eigen::Map<const Eigen::RowVector3d>(&vertices[3 * vertex]));
        face.area = areas[i];
        vertex_offset += face_vertices[vertex_offset] + 1;

        double best_error = std::numeric_limits<double>::infinity();
        // Voro++ labels initial orthogonal periodic faces with negative wall
        // IDs, and initial triclinic self-image faces with zero. Resolve those
        // against the actual lattice as well as the reported neighbor.
        for (int neighbor : {neighbors[i], center})
        {
            if (neighbor < 0 || (neighbor == center && neighbors[i] > 0 && neighbors[i] != center))
                continue;
            const double ri = radii ? radii->at(atoms.chemmap_[center]) : 0;
            const double rj = radii ? radii->at(atoms.chemmap_[neighbor]) : 0;
            // A radical plane is at (distance^2 + ri^2 - rj^2)/(2*distance).
            // Equal radii have only one valid root, 2*offset. Computing the
            // other (zero) root by subtraction can make it spuriously positive.
            const double radius_difference = (ri - rj) * (ri + rj);
            std::array<double, 2> distances{2 * face.offset, 0};
            if (ri != rj)
            {
                const double discriminant = std::fma(face.offset, face.offset, -radius_difference);
                const double roundoff = 8 * std::numeric_limits<double>::epsilon()
                    * std::max(face.offset * face.offset, std::abs(radius_difference));
                if (!std::isfinite(discriminant) || discriminant < -roundoff) continue;
                const double root = std::sqrt(std::max(0., discriminant));
                // Avoid cancellation in the smaller root; either positive root
                // may describe the actual neighbor, including for offset < 0.
                const double q = face.offset + std::copysign(root, face.offset);
                distances = {q, q != 0 ? radius_difference / q : 0};
            }
            for (double distance : distances)
            {
                if (!std::isfinite(distance) || distance <= 0) continue;
                const Eigen::RowVector3d expected = distance * face.normal;
                const Eigen::RowVector3d delta = atoms.pos_[neighbor] - atoms.pos_[center];
                const Eigen::RowVector3d image = ((expected - delta) * inverse).array().round();
                // The center in the zero lattice image cannot generate a face.
                // Keep all nonzero self images, which are essential in small cells.
                if (neighbor == center && image.isZero(0)) continue;
                const Eigen::RowVector3d displacement = delta + image * atoms.cell_;
                const double image_distance = displacement.norm();
                if (!std::isfinite(image_distance) || image_distance <= 0) continue;
                // Validate the original plane after lattice rounding; proximity
                // to a quadratic root alone does not establish a valid face.
                const double offset_error = .5 * (image_distance + radius_difference / image_distance) - face.offset;
                if (!std::isfinite(offset_error) || offset_error * offset_error > geometry_tolerance_squared) continue;
                const double error = (displacement - expected).squaredNorm();
                if (error < best_error)
                {
                    best_error = error;
                    face.neighbor = neighbor;
                    face.displacement = displacement;
                }
            }
        }
        if (!std::isfinite(best_error) || best_error > geometry_tolerance_squared)
            throw std::runtime_error("Cannot resolve the periodic image of a Voronoi face");
        result.push_back(face);
    }
    return result;
}

inline voro::voronoicell_neighbor *refine(voro::voronoicell_neighbor &original,
                                         voro::voronoicell_neighbor &refined,
                                         const std::vector<Face> &faces,
                                         double surftol, int numneigh,
                                         std::vector<int> &selected, bool &warned)
{
    selected.resize(faces.size());
    std::iota(selected.begin(), selected.end(), 0);
    if (std::abs(surftol) <= TOL) return &original;

    std::vector<int> ordered = selected;
    if (numneigh > 0)
        std::stable_sort(ordered.begin(), ordered.end(),
                         [&](int a, int b) { return faces[a].area > faces[b].area; });
    const double scale = surftol < 0 ? 1. / original.surface_area() : 1.;
    std::vector<int> retained;
    for (int i = 0; i < ordered.size(); ++i)
        if (i < numneigh || faces[ordered[i]].area * scale > std::abs(surftol))
            retained.push_back(ordered[i]);
    if (retained.size() == faces.size()) return &original;

    // Use the original planes, including radical offsets, without applying
    // another minimum-image wrap or radius rescaling.
    const double bound = 2 * std::sqrt(original.max_radius_squared());
    refined.init(-bound, bound, -bound, bound, -bound, bound);
    for (int i : retained)
    {
        const auto &face = faces[i];
        refined.nplane(face.normal[0], face.normal[1], face.normal[2], 2 * face.offset, i);
    }
    std::vector<int> refined_faces;
    refined.neighbors(refined_faces);
    if (std::any_of(refined_faces.begin(), refined_faces.end(), [](int i) { return i < 0; }))
    {
        if (!warned)
            std::cerr << "Warning: Voronoi surface filtering left bounding-box faces; "
                         "keeping the original cells where this occurs.\n";
        warned = true;
        return &original;
    }
    selected = std::move(refined_faces);
    return &refined;
}

inline Neighborhood neighborhood(const std::vector<Face> &faces, const std::vector<int> &selected,
                                  const Atoms &atoms, int center, const Eigen::Matrix3d &rotation)
{
    Neighborhood result;
    result.centertype = atoms.chemmap_[center];
    result.nbhcount_ = selected.size();
    for (int i : selected)
    {
        const auto &face = faces[i];
        result.indices_.push_back(face.neighbor);
        result.chemmap_.push_back(atoms.chemmap_[face.neighbor]);
        result.pos_.push_back(face.displacement * rotation);
        result.distances_.push_back(face.displacement.norm());
    }
    return result;
}
}
