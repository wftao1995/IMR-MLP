// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "intersitials.h"
#include "tools/tools.h"
#include "base/neighborhoodKD.h"
#include <algorithm>
#include <stdexcept>
void get_intersitial_sites(Atoms &atom,
                           std::vector<Eigen::RowVector3d> &intpos,
                           std::map<int, double> atomradii,
                           bool count_vec,
                           bool count_cedg,
                           bool count_cface,
                           double mindist,
                           double volprec,
                           double edgeprec,
                           double faceprec)
{
    intpos.clear();
    intpos.reserve(20);
    Atoms atomt = atom;
    atomt.mkLower();
    bool orthcell = atom.isorth();
    const double lx = atomt.cell_(0, 0);
    const double ly = atomt.cell_(1, 1);
    const double lz = atomt.cell_(2, 2);
    const double l = pow(double(atomt.size()) / (5.6 * lx * ly * lz), 1. / 3.);
    const int nx = floor(lx * l + 1);
    const int ny = floor(ly * l + 1);
    const int nz = floor(lz * l + 1);

    if (orthcell)
    {
        voro::container_poly con(0., lx, 0., ly, 0., lz, nx, ny, nz, true, true, true, 8);
        for (int i = 0; i < atomt.size(); i++)
        {
            con.put(i, atomt.pos_[i](0), atomt.pos_[i](1), atomt.pos_[i](2), atomradii[atomt.chemmap_[i]]);
        }
        voro::voronoicell ca;
        voro::c_loop_all cl(con);

        if (cl.start())
        {
            do
            {
                if (con.compute_cell(ca, cl))
                {
                    int id = cl.pid();
                    // if (cl.pid() != 42)
                    //     continue;
                    double volume = ca.volume();
                    if (volume < volprec)
                    {
                        continue;
                    }
                    int numface = ca.number_of_faces();
                    int numedge = ca.number_of_edges();
                    int numvert = ca.p;
                    int sitetotal = numface + numedge + numvert;
                    // vertices
                    if (count_vec)
                        for (int i = 0; i < numvert; ++i)
                        {
                            Eigen::RowVector3d pos;
                            pos(0) = ca.pts[3 * i] * 0.5;
                            pos(1) = ca.pts[3 * i + 1] * 0.5;
                            pos(2) = ca.pts[3 * i + 2] * 0.5;
                            pos += atomt.pos_[id];
                            intpos.emplace_back(pos);
                        }
                    // // center of edges
                    if (count_cedg)
                        for (int i = 0; i < numvert - 1; ++i)
                        {
                            for (int j = 0; j < ca.nu[i]; ++j)
                            {
                                int k = ca.ed[i][j];
                                if (k > i)
                                {
                                    Eigen::RowVector3d pos, dis;
                                    pos(0) = (ca.pts[3 * k] + ca.pts[3 * i]) * 0.25;
                                    pos(1) = (ca.pts[3 * k + 1] + ca.pts[3 * i + 1]) * 0.25;
                                    pos(2) = (ca.pts[3 * k + 2] + ca.pts[3 * i + 2]) * 0.25;
                                    pos += atomt.pos_[id];
                                    dis(0) = ca.pts[3 * k] - ca.pts[3 * i];
                                    dis(1) = ca.pts[3 * k + 1] - ca.pts[3 * i + 1];
                                    dis(2) = ca.pts[3 * k + 2] - ca.pts[3 * i + 2];
                                    dis *= 0.5;
                                    if (dis.norm() > edgeprec)
                                        intpos.push_back(pos);
                                }
                            }
                        }
                    // center of faces
                    if (count_cface)
                    {
                        std::vector<double> c;
                        std::vector<double> area;
                        std::vector<int> ford;
                        std::vector<int> fvec;
                        ca.face_orders(ford);
                        ca.face_vertices(fvec);
                        ca.face_areas(area);
                        int ct = 1;
                        for (int i = 0; i < numface; ++i)
                        {
                            if (area[i] > faceprec)
                            {
                                Eigen::RowVector3d pos = Eigen::RowVector3d::Zero();
                                int order = ford[i];
                                for (int j = 0; j < order; ++j)
                                {
                                    pos(0) += ca.pts[3 * fvec[ct] + 0] * 0.5;
                                    pos(1) += ca.pts[3 * fvec[ct] + 1] * 0.5;
                                    pos(2) += ca.pts[3 * fvec[ct] + 2] * 0.5;
                                    ++ct;
                                }
                                ++ct;
                                pos /= static_cast<double>(order);
                                pos += atomt.pos_[id];
                                intpos.push_back(pos);
                            }
                        }
                    }
                }
                else
                {
                    // GPWARNING
                }
            } while (cl.inc());
        }
        else
        {
            // GPERROR
        }
    }
    else
    {
        const double xy = atomt.cell_(1, 0);
        const double xz = atomt.cell_(2, 0);
        const double yz = atomt.cell_(2, 1);
        voro::container_periodic_poly con(lx, xy, ly, xz, yz, lz, nx, ny, nz, 8);
        for (int i = 0; i < atomt.size(); i++)
        {
            con.put(i, atomt.pos_[i](0), atomt.pos_[i](1), atomt.pos_[i](2), atomradii[atomt.chemmap_[i]]);
        }
        voro::voronoicell ca;
        voro::c_loop_all_periodic cl(con);

        if (cl.start())
        {
            do
            {
                if (con.compute_cell(ca, cl))
                {
                    int id = cl.pid();
                    _DEBUG_OUTPUT_(id);
                    // if (cl.pid() != 42)
                    //     continue;
                    double volume = ca.volume();
                    if (volume < volprec)
                    {
                        continue;
                    }
                    int numface = ca.number_of_faces();
                    int numedge = ca.number_of_edges();
                    int numvert = ca.p;
                    int sitetotal = numface + numedge + numvert;
                    _DEBUG_OUTPUT_(sitetotal);
                    // Vertices
                    if (count_vec)
                        for (int i = 0; i < numvert; ++i)
                        {
                            Eigen::RowVector3d pos;
                            pos(0) = ca.pts[3 * i] * 0.5;
                            pos(1) = ca.pts[3 * i + 1] * 0.5;
                            pos(2) = ca.pts[3 * i + 2] * 0.5;
                            pos += atomt.pos_[id];
                            intpos.emplace_back(pos);
                        }
                    // // center of edges
                    if (count_cedg)
                        for (int i = 0; i < numvert - 1; ++i)
                        {
                            for (int j = 0; j < ca.nu[i]; ++j)
                            {
                                int k = ca.ed[i][j];
                                if (k > i)
                                {
                                    Eigen::RowVector3d pos, dis;
                                    pos(0) = (ca.pts[3 * k] + ca.pts[3 * i]) * 0.25;
                                    pos(1) = (ca.pts[3 * k + 1] + ca.pts[3 * i + 1]) * 0.25;
                                    pos(2) = (ca.pts[3 * k + 2] + ca.pts[3 * i + 2]) * 0.25;
                                    pos += atomt.pos_[id];
                                    dis(0) = ca.pts[3 * k] - ca.pts[3 * i];
                                    dis(1) = ca.pts[3 * k + 1] - ca.pts[3 * i + 1];
                                    dis(2) = ca.pts[3 * k + 2] - ca.pts[3 * i + 2];
                                    dis *= 0.5;
                                    if (dis.norm() > edgeprec)
                                        intpos.push_back(pos);
                                }
                            }
                        }
                    // center of faces
                    if (count_cface)
                    {
                        std::vector<double> c;
                        std::vector<double> area;
                        std::vector<int> ford;
                        std::vector<int> fvec;
                        ca.face_orders(ford);
                        ca.face_vertices(fvec);
                        ca.face_areas(area);
                        int ct = 1;
                        for (int i = 0; i < numface; ++i)
                        {
                            if (area[i] > faceprec)
                            {
                                Eigen::RowVector3d pos = Eigen::RowVector3d::Zero();
                                int order = ford[i];
                                for (int j = 0; j < order; ++j)
                                {
                                    pos(0) += ca.pts[3 * fvec[ct] + 0] * 0.5;
                                    pos(1) += ca.pts[3 * fvec[ct] + 1] * 0.5;
                                    pos(2) += ca.pts[3 * fvec[ct] + 2] * 0.5;
                                    ++ct;
                                }
                                ++ct;
                                pos /= static_cast<double>(order);
                                pos += atomt.pos_[id];
                                intpos.push_back(pos);
                            }
                        }
                    }
                }
                else
                {
                    // GPWARNING
                }
            } while (cl.inc());
        }
        else
        {
            // GPERROR
            _DEBUG_OUTPUT_("GPERRER");
        }
    }
    Eigen::RowVector3d vec;
    auto invcell = atomt.cell_.inverse();
    for (int i = 0; i < intpos.size(); i++)
    {
        vec = intpos[i] * invcell;
        for (int j = 0; j < 3; ++j)
        {
            vec[j] -= floor(vec[j]);
            if (fabs(vec[j] - 1.00) < 1e-8)
                vec[j] = 0.0;
        }
        intpos[i] = vec * atomt.cell_;
    }
    std::sort(intpos.begin(), intpos.end(), [](Eigen::RowVector3d &a, Eigen::RowVector3d &b)
              {
                  if ((a(0) - b(0)) > 1e-5)
                      return true;
                  else if ((a(0) - b(0)) < -1e-5)
                      return false;
                  if ((a(1) - b(1)) > 1e-5)
                      return true;
                  else if ((a(1) - b(1)) < -1e-5)
                      return false;
                  if ((a(2) - b(2)) > 1e-5)
                      return true;
                  else if ((a(2) - b(2)) < -1e-5)
                      return false;
                  return false;
              });
    _DEBUG_OUTPUT_(intpos.size());
    intpos.erase(std::unique(intpos.begin(), intpos.end(), [](Eigen::RowVector3d &a, Eigen::RowVector3d &b)
                             {
                                 if ((a - b).norm() < 1e-5)
                                     return true;
                                 return false;
                             }),
                 intpos.end());
    _DEBUG_OUTPUT_(intpos.size());
    // Voronoi and deduplication use the lowered cell; return Cartesian sites
    // in the caller's frame via their unchanged fractional coordinates.
    for (auto &pos : intpos)
        pos = (pos * invcell * atom.cell_).eval();
    return;
}

/**
 * @brief Helper: extract Voronoi cell candidate sites (vertices, edge-centers,
 *        face-centers) from a voronoicell and add them to intpos.
 *
 * Mirrors the logic in the main get_intersitial_sites() loop body.
 */
static void extract_sites_from_cell(voro::voronoicell &ca, int id,
                                     const std::vector<Eigen::RowVector3d> &abs_pos,
                                     std::vector<Eigen::RowVector3d> &intpos,
                                     double volprec, double edgeprec, double faceprec,
                                     bool count_vec, bool count_cedg, bool count_cface,
                                     const Eigen::RowVector3d *box_min = nullptr,
                                     const Eigen::RowVector3d *box_max = nullptr)
{
    // ca.volume() is a non-const method in voro++, so we can't use const ref
    double volume = ca.volume();
    if (volume < volprec) return;

    int numvert = ca.p;
    const Eigen::RowVector3d &center = abs_pos[id];
    auto artificial_vertex = [&](int i) {
        if (!box_min) return false;
        const Eigen::RowVector3d vertex = center + Eigen::RowVector3d(ca.pts[3*i], ca.pts[3*i+1], ca.pts[3*i+2]) * .5;
        for (int d=0; d<3; ++d)
            if (vertex[d] <= (*box_min)[d]+1e-8 || vertex[d] >= (*box_max)[d]-1e-8) return true;
        return false;
    };

    // Vertices
    if (count_vec) {
        for (int i = 0; i < numvert; ++i) {
            if (artificial_vertex(i)) continue;
            Eigen::RowVector3d pos;
            pos(0) = ca.pts[3 * i] * 0.5;
            pos(1) = ca.pts[3 * i + 1] * 0.5;
            pos(2) = ca.pts[3 * i + 2] * 0.5;
            pos += center;
            intpos.emplace_back(pos);
        }
    }

    // Edge centers
    if (count_cedg) {
        for (int i = 0; i < numvert - 1; ++i) {
            for (int j = 0; j < ca.nu[i]; ++j) {
                int k = ca.ed[i][j];
                if (k > i && !artificial_vertex(i) && !artificial_vertex(k)) {
                    Eigen::RowVector3d pos, dis;
                    pos(0) = (ca.pts[3 * k] + ca.pts[3 * i]) * 0.25;
                    pos(1) = (ca.pts[3 * k + 1] + ca.pts[3 * i + 1]) * 0.25;
                    pos(2) = (ca.pts[3 * k + 2] + ca.pts[3 * i + 2]) * 0.25;
                    pos += center;
                    dis(0) = ca.pts[3 * k] - ca.pts[3 * i];
                    dis(1) = ca.pts[3 * k + 1] - ca.pts[3 * i + 1];
                    dis(2) = ca.pts[3 * k + 2] - ca.pts[3 * i + 2];
                    dis *= 0.5;
                    if (dis.norm() > edgeprec)
                        intpos.push_back(pos);
                }
            }
        }
    }

    // Face centers
    if (count_cface) {
        std::vector<int> ford;
        std::vector<int> fvec;
        std::vector<double> area;
        ca.face_orders(ford);
        ca.face_vertices(fvec);
        ca.face_areas(area);
        int ct = 1;
        int numface = ca.number_of_faces();
        for (int i = 0; i < numface; ++i) {
            const int start = ct;
            bool artificial = false;
            for (int j=0; j<ford[i]; ++j) artificial = artificial || artificial_vertex(fvec[start+j]);
            if (area[i] > faceprec && !artificial) {
                Eigen::RowVector3d pos = Eigen::RowVector3d::Zero();
                int order = ford[i];
                for (int j = 0; j < order; ++j) {
                    pos(0) += ca.pts[3 * fvec[ct] + 0] * 0.5;
                    pos(1) += ca.pts[3 * fvec[ct] + 1] * 0.5;
                    pos(2) += ca.pts[3 * fvec[ct] + 2] * 0.5;
                    ++ct;
                }
                ++ct;
                pos /= static_cast<double>(order);
                pos += center;
                intpos.push_back(pos);
            } else {
                ct += ford[i] + 1;
            }
        }
    }
}

namespace {
// Incremental triangular hull. Faces point away from a fixed interior point;
// coplanar surface triangles are harmless duplicate half-space constraints.
struct HullFace { int a,b,c; Eigen::RowVector3d normal; };
std::vector<HullFace> host_hull(const std::vector<Eigen::RowVector3d> &p, double eps)
{
    if (p.size()<4) throw std::invalid_argument("FFF interstitial search requires a three-dimensional atomic convex hull");
    int b=0,c=0,d=0;
    for(int i=1;i<p.size();++i) if((p[i]-p[0]).squaredNorm()>(p[b]-p[0]).squaredNorm()) b=i;
    for(int i=1;i<p.size();++i)
        if((p[b]-p[0]).cross(p[i]-p[0]).squaredNorm()>(p[b]-p[0]).cross(p[c]-p[0]).squaredNorm()) c=i;
    const Eigen::RowVector3d cross=(p[b]-p[0]).cross(p[c]-p[0]);
    if(cross.norm()<=eps*eps) throw std::invalid_argument("FFF interstitial search requires a three-dimensional atomic convex hull");
    const Eigen::RowVector3d normal=cross.normalized();
    for(int i=1;i<p.size();++i) if(std::abs(normal.dot(p[i]-p[0]))>std::abs(normal.dot(p[d]-p[0]))) d=i;
    if(std::abs(normal.dot(p[d]-p[0]))<=eps) throw std::invalid_argument("FFF interstitial search requires a three-dimensional atomic convex hull");
    const Eigen::RowVector3d inside=(p[0]+p[b]+p[c]+p[d])/4.;
    std::vector<HullFace> faces;
    auto add=[&](int i,int j,int k) {
        Eigen::RowVector3d n=(p[j]-p[i]).cross(p[k]-p[i]);
        if(n.norm()<=eps*eps) return;
        n.normalize();
        if(n.dot(inside-p[i])>0) { std::swap(j,k); n=-n; }
        faces.push_back({i,j,k,n});
    };
    add(0,b,c);add(0,d,b);add(0,c,d);add(b,d,c);
    for(int i=0;i<p.size();++i) {
        std::vector<HullFace> kept;
        std::vector<std::pair<int,int>> horizon;
        for(const auto &f:faces) {
            if(f.normal.dot(p[i]-p[f.a])<=eps) { kept.push_back(f);continue; }
            for(auto edge:{std::pair{f.a,f.b},std::pair{f.b,f.c},std::pair{f.c,f.a}}) {
                auto reverse=std::find(horizon.begin(),horizon.end(),std::pair{edge.second,edge.first});
                if(reverse==horizon.end()) horizon.push_back(edge);else horizon.erase(reverse);
            }
        }
        faces=std::move(kept);
        for(auto edge:horizon) add(edge.first,edge.second,i);
    }
    return faces;
}
}

void get_intersitial_sites_nonperiodic(Atoms &atom,
    std::vector<Eigen::RowVector3d> &intpos, std::map<int,double> atomradii,
    bool count_vec, bool count_cedg, bool count_cface, double mindist,
    double volprec, double edgeprec, double faceprec,
    const Eigen::RowVector3d *query, double search_radius)
{
    if(atom.pbc_[0] || atom.pbc_[1] || atom.pbc_[2])
        throw std::invalid_argument("Atomic convex-hull interstitial search requires F F F periodicity");
    Atoms geometry;geometry.pos_=atom.pos_;geometry.cell_=atom.cell_;geometry.cartesian=atom.cartesian;
    if(!geometry.cartesian) geometry.to_cartesian();
    const auto &positions=geometry.pos_;
    if(positions.empty()) throw std::invalid_argument("FFF interstitial search requires a three-dimensional atomic convex hull");
    Eigen::RowVector3d low=positions[0],high=low;
    for(const auto &p:positions) {
        if(!p.allFinite()) throw std::invalid_argument("FFF interstitial search requires finite positions");
        low=low.cwiseMin(p);high=high.cwiseMax(p);
    }
    const double eps=1e-9*std::max(1.,(high-low).norm());
    const auto hull=host_hull(positions,eps);
    std::vector<Eigen::RowVector3d> local;
    std::vector<int> ids;
    for(int i=0;i<positions.size();++i) if(!query || (positions[i]-*query).norm()<=search_radius) {
        local.push_back(positions[i]);ids.push_back(i);
    }
    intpos.clear();if(local.empty()) return;
    low=high=local[0];for(const auto &p:local) {low=low.cwiseMin(p);high=high.cwiseMax(p);}
    low.array()-=2.;high.array()+=2.;
    const int grid=std::max(1,int(std::cbrt(local.size()/5.6))+1);
    voro::container_poly con(low[0],high[0],low[1],high[1],low[2],high[2],grid,grid,grid,false,false,false,8);
    for(int i=0;i<local.size();++i) con.put(i,local[i][0],local[i][1],local[i][2],atomradii[atom.chemmap_[ids[i]]]);
    voro::voronoicell cell;voro::c_loop_all loop(con);
    if(loop.start()) do {
        if(con.compute_cell(cell,loop)) extract_sites_from_cell(cell,loop.pid(),local,intpos,
            volprec,edgeprec,faceprec,count_vec,count_cedg,count_cface,&low,&high);
    } while(loop.inc());
    std::vector<Eigen::RowVector3d> accepted;
    for(const auto &site:intpos) {
        bool keep=!query || (site-*query).norm()<=search_radius;
        for(const auto &face:hull) if(face.normal.dot(site-positions[face.a])>=-eps) {keep=false;break;}
        if(!keep) continue;
        for(const auto &host:positions) if((site-host).norm()<mindist) {keep=false;break;}
        if(!keep) continue;
        for(const auto &old:accepted) if((site-old).norm()<1e-5) {keep=false;break;}
        if(keep) accepted.push_back(site);
    }
    intpos=std::move(accepted);
}

/**
 * @brief Post-process interstitial sites: fold into unit cell, sort, deduplicate.
 */
static void postprocess_sites(std::vector<Eigen::RowVector3d> &intpos,
                               const Atoms &atom)
{
    if (intpos.empty()) return;
    Atoms wrapping;wrapping.cell_=atom.cell_;wrapping.pbc_=atom.pbc_;
    wrapping.pos_=std::move(intpos);wrapping.wrap();intpos=std::move(wrapping.pos_);

    std::sort(intpos.begin(), intpos.end(),
              [](const Eigen::RowVector3d &a, const Eigen::RowVector3d &b) {
                  if ((a(0) - b(0)) > 1e-5) return true;
                  if ((a(0) - b(0)) < -1e-5) return false;
                  if ((a(1) - b(1)) > 1e-5) return true;
                  if ((a(1) - b(1)) < -1e-5) return false;
                  if ((a(2) - b(2)) > 1e-5) return true;
                  if ((a(2) - b(2)) < -1e-5) return false;
                  return false;
              });

    intpos.erase(std::unique(intpos.begin(), intpos.end(),
                             [](const Eigen::RowVector3d &a, const Eigen::RowVector3d &b) {
                                 return (a - b).norm() < 1e-5;
                             }),
                 intpos.end());
}

// ========================================================================
// Local Voronoi: only build Voronoi for atoms near query_pos
// ========================================================================

void get_intersitial_sites_near(Atoms &atom,
                                const Eigen::RowVector3d &query_pos,
                                std::vector<Eigen::RowVector3d> &intpos,
                                std::map<int, double> atomradii,
                                double search_radius,
                                bool count_vec,
                                bool count_cedg,
                                bool count_cface,
                                double mindist,
                                double volprec,
                                double edgeprec,
                                double faceprec,
                                NeighborhoodKD *nkd_external)
{
    intpos.clear();
    intpos.reserve(20);

    // ---- Step 1: Build KD-tree with PBC ----
    // r_extent = search_radius, so periodic images within that range are included
    NeighborhoodKD nkd_local;
    NeighborhoodKD *nkd = nkd_external;
    if (nkd == nullptr) {
        nkd_local.init(&atom, search_radius);
        nkd = &nkd_local;
    }

    // ---- Step 2: Query nearby atoms within search_radius of query_pos ----
    double qpos[3] = {query_pos(0), query_pos(1), query_pos(2)};
    Neighborhood nbh = nkd->query(qpos, search_radius, 0, true);
    if (nbh.nbhcount_ == 0) {
        return;
    }

    // ---- Step 3: Build a local non-periodic voro++ container ----
    // Compute bounding box: query_pos ± (search_radius + margin)
    const double margin = 2.0;
    double x_min = query_pos(0) - search_radius - margin;
    double x_max = query_pos(0) + search_radius + margin;
    double y_min = query_pos(1) - search_radius - margin;
    double y_max = query_pos(1) + search_radius + margin;
    double z_min = query_pos(2) - search_radius - margin;
    double z_max = query_pos(2) + search_radius + margin;

    double lx = x_max - x_min;
    double ly = y_max - y_min;
    double lz = z_max - z_min;

    // Grid divisions for voro++ (same heuristic as the full-cell version)
    double l_grid = pow(double(nbh.nbhcount_) / (5.6 * lx * ly * lz), 1. / 3.);
    int nx = std::max(1, int(lx * l_grid + 1));
    int ny = std::max(1, int(ly * l_grid + 1));
    int nz = std::max(1, int(lz * l_grid + 1));

    // Non-periodic container_poly (we use radii for weighting)
    voro::container_poly con(x_min, x_max, y_min, y_max, z_min, z_max,
                             nx, ny, nz, false, false, false, 8);

    // Store absolute positions for later reference
    std::vector<Eigen::RowVector3d> abs_pos;
    abs_pos.reserve(nbh.nbhcount_);

    for (int i = 0; i < nbh.nbhcount_; ++i) {
        // Neighbor position relative to query_pos → absolute Cartesian
        Eigen::RowVector3d abs_p = query_pos + nbh.pos_[i];
        abs_pos.push_back(abs_p);

        int aid = nbh.indices_[i];
        double radius = atomradii[atom.chemmap_[aid]];
        con.put(i, abs_p(0), abs_p(1), abs_p(2), radius);
    }

    // ---- Step 4: Compute Voronoi cells for all atoms in the local container ----
    // We use the non-periodic loop (c_loop_all)
    voro::voronoicell ca;
    voro::c_loop_all cl(con);

    // Safety radius: only accept atoms whose Voronoi cell center is at least
    // this far from the container walls (to avoid truncated-cell artifacts)
    const double safe_margin = std::min(search_radius * 0.2, 1.0);
    double safe_x_min = x_min + safe_margin;
    double safe_x_max = x_max - safe_margin;
    double safe_y_min = y_min + safe_margin;
    double safe_y_max = y_max - safe_margin;
    double safe_z_min = z_min + safe_margin;
    double safe_z_max = z_max - safe_margin;

    const Eigen::RowVector3d box_min(x_min,y_min,z_min),box_max(x_max,y_max,z_max);
    if (cl.start()) {
        do {
            if (con.compute_cell(ca, cl)) {
                int id = cl.pid();

                // Skip atoms near the container boundary to avoid truncated-cell artifacts
                const Eigen::RowVector3d &ap = abs_pos[id];
                if (ap(0) < safe_x_min || ap(0) > safe_x_max ||
                    ap(1) < safe_y_min || ap(1) > safe_y_max ||
                    ap(2) < safe_z_min || ap(2) > safe_z_max) {
                    continue;
                }

                extract_sites_from_cell(ca, id, abs_pos, intpos,
                                        volprec, edgeprec, faceprec,
                                        count_vec, count_cedg, count_cface,&box_min,&box_max);
            }
        } while (cl.inc());
    }

    // ---- Step 5: Filter by minimum distance from host atoms ----
    if (!intpos.empty()) {
        Atoms metric;metric.cell_=atom.cell_;metric.pbc_=atom.pbc_;metric.pos_.resize(2);
        std::vector<Eigen::RowVector3d> filtered;
        for (auto &cp : intpos) {
            bool ok = true;
            if (mindist <= search_radius) {
                double p[3] = {cp(0), cp(1), cp(2)};
                Neighborhood near = nkd->queryu(p, mindist, 0, true);
                ok = (near.nbhcount_ == 0);
            } else {
                for (int ia = 0; ia < atom.size(); ++ia) {
                    metric.pos_[0]=cp;metric.pos_[1]=atom.pos_[ia];
                    double dist = metric.get_rab(0,1).norm();
                    if (dist < mindist) {
                        ok = false;
                        break;
                    }
                }
            }
            if (ok) filtered.push_back(cp);
        }
        intpos = std::move(filtered);
    }

    // ---- Step 6: Post-process (fold, sort, deduplicate) ----
    postprocess_sites(intpos, atom);

    return;
}
