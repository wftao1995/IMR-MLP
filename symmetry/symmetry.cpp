// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "symmetry.h"
#include "base/neighborhoodKD.h"
#include "tools/tools.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <set>
// TODO: trans_reduce, primi_reduce: deal with per-atom properties, e.g. forces.
// TODO: Use descriptor to check if symmetry operation may exist. 
//       i.e inverse symmetry can be checked with vector sum.
namespace Symmetry3D {
namespace {
std::vector<Eigen::RowVector3d> translation_candidates(Atoms& atom,const ReductionSearchOptions& options) {
    std::vector<Eigen::RowVector3d> result;
    auto append=[&](int anchor) {
        for(int j=0;j<atom.size();++j) {
            if(j==anchor)continue;
            if(options.enabled && options.prefilter && atom.chemmap_[anchor]!=atom.chemmap_[j])continue;
            result.push_back(atom.get_rab(anchor,j));
        }
    };
    if(options.enabled && options.rare_anchor) {
        const auto counts=atom.types_count_map();
        const auto rare=std::min_element(counts.begin(),counts.end(),[](const auto& a,const auto& b){return a.second<b.second;});
        int anchor=0;while(atom.chemmap_[anchor]!=rare->first)++anchor;
        if(anchor)append(anchor);
    }
    // Retain original-anchor candidates as a completeness backstop. Only exact
    // identical vectors may be cached; approximate bins cannot reject trials.
    append(0);
    if(options.enabled && options.order_candidates)
        std::stable_sort(result.begin(),result.end(),[](const auto& a,const auto& b){return a.squaredNorm()<b.squaredNorm();});
    return result;
}
template<size_t N> bool repeated_candidate(const std::array<double,N>& key,std::set<std::array<double,N>>& cache,
    const ReductionSearchOptions& options,ReductionSearchStats* stats) {
    if(!options.enabled || !options.cache_candidates)return false;
    if(cache.count(key)) {if(stats)++stats->cache_hits;return true;}
    if(cache.size()<options.cache_capacity)cache.insert(key);
    return false;
}
// Geometry kernels below consume Cartesian positions. Restore the caller's
// representation using the final cell, including after nested reductions.
struct ReductionCoordinateScope {
    Atoms& atoms;
    bool restore_fractional;

    explicit ReductionCoordinateScope(Atoms& value)
        : atoms(value), restore_fractional(!value.cartesian)
    {
        if (restore_fractional) atoms.to_cartesian();
    }

    ~ReductionCoordinateScope()
    {
        if (restore_fractional && atoms.cartesian) atoms.to_fractional();
    }
};

// Validate before wrapping positions or constructing a periodic neighborhood.
// Use the same inverse needed by the algorithms, without an absolute volume
// cutoff: small and left-handed, but representable, cells remain admissible.
void validate_reduction_input(const Atoms& atoms, double tol)
{
    if (atoms.pos_.empty())
        throw std::invalid_argument("Cell reduction requires a nonempty structure");
    if (atoms.pos_.size() != atoms.chemmap_.size())
        throw std::invalid_argument("Cell reduction requires matching position and species counts");
    if (!std::isfinite(tol) || tol <= 0 || !std::isfinite(5.0 * tol))
        throw std::invalid_argument("Cell reduction requires a positive finite tolerance with finite 5*tol");
    if (!atoms.cell_.allFinite())
        throw std::invalid_argument("Cell reduction requires a finite, nonsingular cell");
    const double determinant = atoms.cell_.determinant();
    if (!std::isfinite(determinant) || determinant == 0)
        throw std::invalid_argument("Cell reduction requires a finite, nonsingular cell");
    const Eigen::Matrix3d inverse = atoms.cell_.inverse();
    if (!inverse.allFinite())
        throw std::invalid_argument("Cell reduction requires a finite cell inverse");
    for (const auto& position : atoms.pos_) {
        if (!position.allFinite())
            throw std::invalid_argument("Cell reduction requires finite atom positions");
        if (atoms.cartesian && !atoms.is_nonperiodic() && !(position * inverse).allFinite())
            throw std::invalid_argument("Cell reduction requires finite fractional atom positions");
    }
}

void validate_primitive_periodicity(const Atoms& atoms)
{
    if (!atoms.pbc_[0] || !atoms.pbc_[1] || !atoms.pbc_[2])
        throw std::invalid_argument("3D primitive reduction requires full periodicity; use trans_reduce for mixed or nonperiodic structures");
}

void copy_reduction_properties(const Atoms& source, Atoms& reduced)
{
    reduced.id = source.id;
    for (const char* key : {"label", "config_type"}) {
        const auto found = source.features_.find(key);
        if (found != source.features_.end())
            reduced.features_.emplace(found->first, found->second);
    }
    // These frame labels are extensive; normalize at each successful reduction.
    const double scale = static_cast<double>(reduced.size()) / source.size();
    if (source.have_energy) {
        reduced.energy_ = source.energy_ * scale;
        reduced.have_energy = true;
    }
    if (source.have_fenergy) {
        reduced.fenergy_ = source.fenergy_ * scale;
        reduced.have_fenergy = true;
    }
    if (source.have_stress) {
        reduced.stress_ = source.stress_ * scale;
        reduced.have_stress = true;
    }
    const auto total_charge = source.global_attribs_d_.find("total_charge");
    if (total_charge != source.global_attribs_d_.end())
        reduced.global_attribs_d_.emplace("total_charge", total_charge->second * scale);
}
}

// TODO: use type with least atoms as initial reference atom to reduce search
/**
 * @brief Reduce translational multiplicity of an atomic cell along lattice directions.
 *
 * This function attempts to find and remove repeated translational copies of atoms
 * along the periodic cell row directions (0,1,2). If a valid periodic translation
 * vector is found that maps atoms onto chemically identical atoms, the function
 * collapses those translationally equivalent sites and shortens the corresponding
 * cell row by the integer multiplicity.
 *
 * Behavior / algorithm:
 * - Ensures atom positions are wrapped into the unit cell (atom.wrap()).
 * - For each periodic cell row direction dir in {0,1,2}:
 *   - Repeatedly search for candidate translation vectors from a reference atom (pos_[0])
 *     to other atoms in the cell (get_rab(0,j)).
 *   - For each candidate vector, test whether it is parallel to the chosen cell row and
 *     whether the ratio of the cell-row length to the candidate-vector length is (near) an integer.
 *     The integer test uses a hard threshold of 1e-3.
 *   - If an integer multiplicity nprd >= 2 is found, validate the translation symmetry
 *     using a NeighborhoodKD search (built with search cutoff 5*tol):
 *       - For every atom, verify that its translations by iprd * vec (iprd = 1..nprd-1)
 *         map to existing atoms within geometric tolerance tol and that the chemical
 *         types (chemmap_) match.
 *       - If this validation succeeds, construct a new Atoms object with cell row cdir
 *         divided by nprd and with one representative per translational orbit; positions
 *         are wrapped and the original atom object is replaced in-place.
 *   - Repeat attempts along the same dir until no further reduction is possible.
 *
 * Important details / thresholds:
 * - tol: geometric tolerance (distance) used for neighbor lookups and cross-product
 *        parallelism test (cross-product norm compared to tol).
 * - The integer multiplicity test for cellvec.norm()/vec.norm() uses a closeness
 *   threshold of 1e-3 to the nearest integer.
 * - NeighborhoodKD is used to map translated positions to atom indices via find_atom(..., tol).
 *
 * Parameters:
 * - atom (in/out): Atoms object to be examined and potentially reduced. The function
 *                  modifies atom.cell_, atom.pos_, atom.chemmap_ and atom.size().
 *                  atom.wrap() is called internally to keep positions in the cell.
 * - tol (in): geometric tolerance for comparisons and neighbor searches.
 *
 * Returns:
 * - An integer equal to the multiplicative reduction factor: original_size / new_size.
 *   If no reduction occurs, the function returns 1.
 *
 * Preconditions / assumptions:
 * - atom.cell_ has three row vectors; atom.pbc_ selects the periodic directions.
 * - atom.pos_, atom.chemmap_, and atom.size() are consistent (pos_ and chemmap_
 *   indexed by atom indices).
 * - NeighborhoodKD::find_atom expects coordinates in the same coordinate system as atom.pos_
 *   and uses tol as the matching radius.
 *
 * Side effects:
 * - Modifies the input Atoms object in-place when a valid reduction is found.
 * - Rebuilds atom.cell_ (shortens the specified row by the integer factor) and
 *   reduces the number of atoms to one representative per translational orbit.
 *
 * Complexity / performance:
 * - Builds a NeighborhoodKD for neighbor queries for each attempt; overall cost depends
 *   on the number of candidates and successful reductions. Using the KD structure
 *   reduces cost of nearest-atom queries, but worst-case behavior can be high if many
 *   candidate vectors are tested.
 *
 * Limitations:
 * - Only detects and collapses translations strictly parallel to a single cell row.
 * - Does not attempt reductions along combinations of cell rows or non-lattice-vector
 *   translation symmetries.
 * - The integer multiplicity tolerance (1e-3) and geometric tolerance tol may need
 *   adjustment for numerical noise or non-orthogonal cells.
 */
int trans_reduce(Atoms& atom, double tol) {return trans_reduce(atom,tol,ReductionSearchOptions{},nullptr);}

int trans_reduce(Atoms& atom, double tol,const ReductionSearchOptions& options,ReductionSearchStats* stats)
{
    validate_reduction_input(atom, tol);
    ReductionCoordinateScope coordinates(atom);
    int osize = atom.size();
    auto get_prd = [](const Eigen::RowVector3d& vec, const Atoms& atom, std::map<int, int> &types_count, NeighborhoodKD& nbhs, int& cdir, double tol) {
        cdir = -1;
        Eigen::RowVector3d cellvec;
        for (int idir = 0; idir < 3; ++idir) {
            if (!atom.pbc_[idir])
                continue;
            cellvec = atom.cell_.row(idir);
            if ((cellvec.cross(vec)).norm() < tol) {
                cdir = idir;
                break;
            }
        }
        if (cdir < 0)
            return 0;
        double nexpd = cellvec.norm() / vec.norm();
        int nexp = 0;
        if (fabs(nexpd - round(nexpd)) < 1e-6)
            nexp = round(nexpd);
        else
            return 0;
        if (nexp == 0) return 0;
        // check if atom is splitable by atoms count
        for (auto &it : types_count) {
            if (it.second % nexp != 0) return 0;
        }
        return nexp;
    };
    std::vector<int> mappings;
    auto check_isprd = [&mappings, &tol](Atoms& atom, NeighborhoodKD& nbhskdt, Eigen::RowVector3d& vec, int nprd, int cdir) {
        mappings.assign(atom.size(), -1);
        for (int iat = 0; iat < atom.size(); ++iat) {
            if (mappings[iat] >= 0)
                continue;
            int ivl = 0;
            for (int iprd = 1; iprd < nprd; ++iprd) {
                Eigen::RowVector3d pnew = atom.pos_[iat] + iprd * vec;
                atom.wrap(pnew);
                int nid = nbhskdt.find_atom(pnew.data(), tol);
                if (nid == -1)
                    return false;
                if (atom.chemmap_[iat] != atom.chemmap_[nid])
                    return false;
                mappings[nid] = iat;
                ++ivl;
            }
            if (ivl != nprd - 1)
                return false;
        }
        // build new cell
        Atoms atnew;
        atnew.resize(atom.size() / nprd);
        copy_reduction_properties(atom, atnew);
        atnew.cell_ = atom.cell_;
        atnew.pbc_ = atom.pbc_;
        atnew.cell_.row(cdir) = atom.cell_.row(cdir) / nprd;
        int ct = 0;
        for (int iat = 0; iat < atom.size(); ++iat) {
            if (mappings[iat] == -1) {
                atnew.pos_[ct] = atom.pos_[iat];
                atnew.chemmap_[ct] = atom.chemmap_[iat];
                ++ct;
            }
        }
        atom = atnew;
        atom.wrap();
        return true;
    };

    atom.wrap();
    int dir = -1;
    auto& posref = atom.pos_[0];
    int chemref = atom.chemmap_[0];
    // build kd-tree, todo: test rcut
    bool isprd = true;
    while (isprd) {
        isprd = false;
        NeighborhoodKD nbhskd(&atom, 5.0 * tol);
        auto types_count = atom.types_count_map();
        if(stats)++stats->geometry_stages;
        std::set<std::array<double,3>> cache;
        const auto candidates=options.enabled ? translation_candidates(atom,options) : std::vector<Eigen::RowVector3d>{};
        const size_t count=options.enabled ? candidates.size() : size_t(atom.size()-1);
        for(size_t candidate=0;candidate<count;++candidate) {
            auto posdiff=options.enabled ? candidates[candidate] : atom.get_rab(0,int(candidate)+1);
            if(stats)++stats->candidates;
            if(options.enabled && options.prefilter && posdiff.squaredNorm()==0.) {
                if(stats)++stats->necessary_rejections;continue;
            }
            if(options.enabled && options.cache_candidates) {
                const std::array<double,3> key={posdiff[0],posdiff[1],posdiff[2]};
                if(repeated_candidate(key,cache,options,stats))continue;
            }
            int n = get_prd(posdiff, atom, types_count, nbhskd, dir, tol);
            if (n < 2) {
                if(stats)++stats->necessary_rejections;continue;
            }
            if(stats)++stats->full_checks;
            isprd = check_isprd(atom, nbhskd, posdiff, n, dir);
            if (isprd) {
                break;
            } else {
                continue;
            }
        }
    }

    return osize / atom.size();
}

int trans_reduce_1d(Atoms& atom, int cdir, double tol) {return trans_reduce_1d(atom,cdir,tol,ReductionSearchOptions{},nullptr);}

int trans_reduce_1d(Atoms& atom, int cdir, double tol,const ReductionSearchOptions& options,ReductionSearchStats* stats)
{
    if (cdir < 0 || cdir > 2)
        throw std::invalid_argument("Cell reduction direction must be 0, 1 or 2");
    validate_reduction_input(atom, tol);
    if (!atom.pbc_[cdir])
        return 1;
    ReductionCoordinateScope coordinates(atom);
    int osize = atom.size();
    auto get_prd = [](const Eigen::RowVector3d& vec, const Atoms& atom, std::map<int, int> &types_count, NeighborhoodKD& nbhs, int cdir, double tol) {
        Eigen::RowVector3d cellvec = atom.cell_.row(cdir);
        if ((cellvec.cross(vec)).norm() > tol)
            return 0;
        double nexpd = cellvec.norm() / vec.norm();
        int nexp = 0;
        if (fabs(nexpd - round(nexpd)) < 1e-3)
            nexp = round(nexpd);
        else
            return 0;
        if (nexp == 0) return 0;
        // check if atom is splitable by atoms count
        for (auto &it : types_count) {
            if (it.second % nexp != 0) return 0;
        }
        return nexp;
    };
    std::vector<int> mappings;
    auto check_isprd = [&](Atoms& atom, NeighborhoodKD& nbhskdt, Eigen::RowVector3d& vec, int nprd, int cdir) {
        mappings.assign(atom.size(), -1);
        for (int iat = 0; iat < atom.size(); ++iat) {
            if (mappings[iat] >= 0)
                continue;
            int ivl = 0;
            for (int iprd = 1; iprd < nprd; ++iprd) {
                Eigen::RowVector3d pnew = atom.pos_[iat] + iprd * vec;
                atom.wrap(pnew);
                int nid = nbhskdt.find_atom(pnew.data(), tol);
                if (nid == -1)
                    return false;
                if (atom.chemmap_[iat] != atom.chemmap_[nid])
                    return false;
                mappings[nid] = iat;
                ++ivl;
            }
            if (ivl != nprd - 1)
                return false;
        }
        // build new cell
        Atoms atnew;
        atnew.resize(atom.size() / nprd);
        copy_reduction_properties(atom, atnew);
        atnew.cell_ = atom.cell_;
        atnew.pbc_ = atom.pbc_;
        atnew.cell_.row(cdir) = atom.cell_.row(cdir) / nprd;
        int ct = 0;
        for (int iat = 0; iat < atom.size(); ++iat) {
            if (mappings[iat] == -1) {
                atnew.pos_[ct] = atom.pos_[iat];
                atnew.chemmap_[ct] = atom.chemmap_[iat];
                ++ct;
            }
        }
        atom = atnew;
        atom.wrap();
        return true;
    };

    atom.wrap();
    // Eigen::RowVector3d cellvec = atom.cell_.row(0);
    auto& posref = atom.pos_[0];
    int chemref = atom.chemmap_[0];
    // build kd-tree, todo: test rcut
    bool isprd = true;
    while (isprd) {
        isprd = false;
        NeighborhoodKD nbhskd(&atom, 5.0 * tol);
        auto types_count = atom.types_count_map();
        if(stats)++stats->geometry_stages;
        std::set<std::array<double,3>> cache;
        const auto candidates=options.enabled ? translation_candidates(atom,options) : std::vector<Eigen::RowVector3d>{};
        const size_t count=options.enabled ? candidates.size() : size_t(atom.size()-1);
        for(size_t candidate=0;candidate<count;++candidate) {
            auto posdiff=options.enabled ? candidates[candidate] : atom.get_rab(0,int(candidate)+1);
            if(stats)++stats->candidates;
            if(options.enabled && options.prefilter && posdiff.squaredNorm()==0.) {
                if(stats)++stats->necessary_rejections;continue;
            }
            if(options.enabled && options.cache_candidates) {
                const std::array<double,3> key={posdiff[0],posdiff[1],posdiff[2]};
                if(repeated_candidate(key,cache,options,stats))continue;
            }
            int n = get_prd(posdiff, atom, types_count, nbhskd, cdir, tol);
            if (n < 2) {
                if(stats)++stats->necessary_rejections;continue;
            }
            if(stats)++stats->full_checks;
            isprd = check_isprd(atom, nbhskd, posdiff, n, cdir);
            if (isprd) {
                break;
            } else {
                continue;
            }
        }
    }

    return osize / atom.size();
}

int find_primitive(Atoms& atoms, double tol) {return find_primitive(atoms,tol,ReductionSearchOptions{},nullptr);}

int find_primitive(Atoms& atoms, double tol,const ReductionSearchOptions& options,ReductionSearchStats* stats)
{
    validate_reduction_input(atoms, tol);
    validate_primitive_periodicity(atoms);
    ReductionCoordinateScope coordinates(atoms);
    int n_ori = atoms.size();
    auto get_prd = [](const Eigen::Matrix3d& S, const Eigen::Matrix3d& P, Eigen::Matrix3i& L, std::map<int, int> &types_count, double tol) {
        // S = L * P, L must be integer in this case, since no rotation exist here
        Eigen::Matrix3d Ld = S * P.inverse();
        L = Ld.array().round().matrix().cast<int>();
        int nexpd = 0;
        if ((Ld - L.cast<double>()).cwiseAbs().maxCoeff() < tol) {
            nexpd = L.determinant();
        } else {
            return -1;
        }
        if (nexpd == 0) return -1;
        for (auto &it : types_count) {
            if (it.second % nexpd != 0) return -1;
        }
        return nexpd;
    };
    std::vector<int> mapping;
    std::vector<Eigen::RowVector3d> latpoints;
    auto check_prd = [&mapping, &latpoints](Atoms& atoms, NeighborhoodKD& nbhskd, const Eigen::Matrix3d& rcell, const Eigen::Matrix3i L, double tol) {
        mapping.assign(atoms.size(), -1);
        int xmin, xmax, ymin, ymax, zmin, zmax;
        Eigen::MatrixXi Lall; // all vertix 
        Lall.resize(8, 3);
        Lall.setZero();
        Lall.row(1) = L.row(0);
        Lall.row(2) = L.row(1);
        Lall.row(3) = L.row(2);
        Lall.row(4) = L.row(0) + L.row(1);
        Lall.row(5) = L.row(0) + L.row(2);
        Lall.row(6) = L.row(1) + L.row(2);
        Lall.row(7) = L.row(0) + L.row(1) + L.row(2);
        xmin = Lall.col(0).minCoeff();
        xmax = Lall.col(0).maxCoeff();
        ymin = Lall.col(1).minCoeff();
        ymax = Lall.col(1).maxCoeff();
        zmin = Lall.col(2).minCoeff();
        zmax = Lall.col(2).maxCoeff();
        Eigen::Matrix3d Ld = L.cast<double>();
        Eigen::Matrix3d rec = Ld.inverse(); // col is a*
        // build lattice points
        int detL = L.determinant();
        latpoints.clear();
        latpoints.reserve(detL);
        Eigen::Vector3d a = Ld.row(0).transpose();
        Eigen::Vector3d b = Ld.row(1).transpose();
        Eigen::Vector3d c = Ld.row(2).transpose();
        double aap = a.dot(rec.col(0));
        double bbp = b.dot(rec.col(1));
        double ccp = c.dot(rec.col(2));
        constexpr double ztol = 1e-8;
        for (int x = xmin; x <= xmax; ++x) {
            for (int y = ymin; y <= ymax; ++y) {
                for (int z = zmin; z <= zmax; ++z) {
                    if (x == 0 && y == 0 && z == 0)
                        continue;
                    Eigen::Vector3d p(x, y, z);
                    double xx = p.dot(rec.col(0)) / aap;
                    double yy = p.dot(rec.col(1)) / bbp;
                    double zz = p.dot(rec.col(2)) / ccp;
                    if (xx > -ztol && xx < 1 - ztol && yy > -ztol && yy < 1 - ztol && zz > -ztol && zz < 1 - ztol) {
                        latpoints.emplace_back(x, y, z);
                    }
                }
            }
        }
        if (latpoints.size() != detL - 1) {
            std::cerr << "ERROR: failed search lattice points, got " << latpoints.size() << " except " << detL - 1 << ", this could be a bug, please contact to the author: jtwang18s@imr.ac.cn" << std::endl;
            std::cerr << "Matrix L: \n" << L << std::endl;
            return false;
        }
        // search
        for (int iat = 0; iat < atoms.size(); ++iat) {
            if (mapping[iat] >= 0)
                continue;
            for (int iprd = 0; iprd < latpoints.size(); ++iprd) {
                Eigen::RowVector3d transvec = latpoints[iprd] * rcell;
                Eigen::RowVector3d pnew = atoms.pos_[iat] + transvec;
                atoms.wrap(pnew);
                int nid = nbhskd.find_atom(pnew.data(), tol);
                if (nid == -1)
                    return false;
                if (atoms.chemmap_[iat] != atoms.chemmap_[nid])
                    return false;
                mapping[nid] = iat;
            }
        }
        // reduce cell
        Atoms natoms;
        natoms.cell_ = rcell;
        natoms.resize(atoms.size() / detL);
        copy_reduction_properties(atoms, natoms);
        int ct = 0;
        for (int iat = 0; iat < atoms.size(); ++iat) {
            if (mapping[iat] == -1) {
                natoms.pos_[ct] = atoms.pos_[iat];
                natoms.chemmap_[ct] = atoms.chemmap_[iat];
                ++ct;
            }
        }
        natoms.wrap();
        atoms = natoms;
        return true;
    };
    // 1. find min type
    int min_type = 0;
    int min_count = std::numeric_limits<int>::max();
    auto types_count = atoms.types_count_map();
    std::vector<int> mintype_ids;
    for (auto& t : types_count) {
        if (t.second < min_count) {
            min_type = t.first;
            min_count = t.second;
        }
    }
    int id0 = 0;
    int sign = atoms.cell_.determinant() > 0 ? 1 : -1;
    // 2. try to build translation vectors
    bool isprd = true;
    while (isprd) {
        isprd = false;
        mintype_ids.clear();
        for (int i = 0; i < atoms.size(); ++i) {
            if (atoms.chemmap_[i] == min_type) {
                id0 = i;
                break;
            }
        }
        for (int i = 0; i < atoms.size(); ++i) {
            if (atoms.chemmap_[i] == min_type) {
                mintype_ids.push_back(i);
            }
        }
        if(stats)++stats->geometry_stages;
        std::set<std::array<double,9>> candidate_cache;
        if(options.enabled && options.order_candidates)
            std::stable_sort(mintype_ids.begin(),mintype_ids.end(),[&](int a,int b) {
                return atoms.get_rab(id0,a).squaredNorm()<atoms.get_rab(id0,b).squaredNorm();
            });
        // id0 - i - j - k
        Eigen::RowVector3d vx, vy, vz;
        Eigen::Matrix3d rcell;
        Eigen::Matrix3i matL;
        int nexp;
        atoms.wrap();
        NeighborhoodKD nbhskd(&atoms, 5.0 * tol);
        auto types_count = atoms.types_count_map();
        for (auto& i : mintype_ids) {
            for (auto& j : mintype_ids) {
                for (auto& k : mintype_ids) {
                    vx = atoms.get_rab(id0, i);
                    vy = atoms.get_rab(id0, j);
                    vz = atoms.get_rab(id0, k);
                    if (i == id0) {
                        vx = atoms.cell_.row(0);
                    } 
                    if (j == id0) {
                        vy = atoms.cell_.row(1);
                    } 
                    if (k == id0) {
                        vz = atoms.cell_.row(2);
                    } 
                    rcell.row(0) = vx;
                    rcell.row(1) = vy;
                    rcell.row(2) = vz;
                    int hand = rcell.determinant() > 0 ? 1 : -1;
                    if (hand != sign) {
                        vx *= -1.;
                        rcell.row(0) *= -1.;
                    }
                    if(stats)++stats->candidates;
                    // Exact rank failure is necessary; never use a tolerance
                    // or a descriptor threshold to discard a candidate cell.
                    if(options.enabled && options.prefilter && rcell.determinant()==0.) {
                        if(stats)++stats->necessary_rejections;continue;
                    }
                    if(options.enabled && options.cache_candidates) {
                        std::array<double,9> key;
                        for(int row=0;row<3;++row)for(int col=0;col<3;++col)key[3*row+col]=rcell(row,col);
                        if(repeated_candidate(key,candidate_cache,options,stats))continue;
                    }
                    // now check if {vx, vy, vz} can form a new cell
                    nexp = get_prd(atoms.cell_, rcell, matL, types_count, tol);
                    if (nexp <= 1) {
                        if(stats)++stats->necessary_rejections;continue;
                    }
                    if(stats)++stats->full_checks;
                    // check if rcell is reallay a new cell
                    isprd = check_prd(atoms, nbhskd, rcell, matL, tol);
                    if (isprd) {
                        goto lab_out;
                    } else {
                        continue;
                    }
                }
            }
        }
lab_out:
        ;
    }

    return n_ori / atoms.size();
}

// TODO: finish this
// 60 < all angles < 120 deg
int correct_cell(Atoms& atom, double tol)
{
    ReductionCoordinateScope coordinates(atom);
    bool proceed = true;
    while (proceed) {
        proceed = false;
        for (int i = 0; i < 3; i++) {
            Eigen::RowVector3d l_i = atom.cell_.row(i);
            // Vector3 l_i = Vector3(lattice[i][0], lattice[i][1], lattice[i][2]);
            for (int j = 0; j < 3; j++)
                if (i != j) {
                    Eigen::RowVector3d l_j = atom.cell_.row(j);
                    // Vector3 l_j = Vector3(lattice[j][0], lattice[j][1], lattice[j][2]);
                    double orig_l_j_norm2 = l_j.squaredNorm();
                    double new_l_j_norm2 = (l_j + l_i).squaredNorm();
                    for (int itr = 1; new_l_j_norm2 < orig_l_j_norm2; itr++) {
                        l_j += l_i;
                        new_l_j_norm2 = (l_j + l_i).squaredNorm();
                        proceed = true;
                    }
                    new_l_j_norm2 = (l_j - l_i).squaredNorm();
                    for (int itr = 1; new_l_j_norm2 < orig_l_j_norm2; itr++) {
                        l_j -= l_i;
                        new_l_j_norm2 = (l_j - l_i).squaredNorm();
                        proceed = true;
                    }
                    atom.cell_.row(j) = l_j;
                    // lattice[j][0] = l_j[0];
                    // lattice[j][1] = l_j[1];
                    // lattice[j][2] = l_j[2];
                }
        }
    }
    atom.wrap();
    return 0;
}

int all_reduce(Atoms& atoms, double tol) {return all_reduce(atoms,tol,ReductionSearchOptions{},nullptr);}

int all_reduce(Atoms& atoms, double tol,const ReductionSearchOptions& options,ReductionSearchStats* stats)
{
    validate_reduction_input(atoms, tol);
    validate_primitive_periodicity(atoms);
    ReductionCoordinateScope coordinates(atoms);
    int n_ori = atoms.size();
    auto cold = atoms.cell_;
    trans_reduce(atoms,tol,options,stats);
    find_primitive(atoms,tol,options,stats);
    correct_cell(atoms, tol);
    auto cnew = atoms.cell_;
    return n_ori / atoms.size();
}

int aligned_orientation(Atoms& atom, double tol)
{
    auto cellpars = atom.get_cell_parameters();
    double& a = cellpars[0];
    double& b = cellpars[1];
    double& c = cellpars[2];
    double& alpha = cellpars[3];
    double& beta = cellpars[4];
    double& gamma = cellpars[5];
    // 1.

    // CellType ctype;
    // // step 1. check cell type
    // bool eqab, eqac, eqbc, eqabc;
    // bool alpha90, beta90, gamma90;
    // bool eqalphabeta, eqbetagamma, eqgammaalpha, eqangles;
    // if (__is_close(a, b, tol)) {
    //     eqab = true;
    // }
    // if (__is_close(a, c, tol)) {
    //     eqac = true;
    // }
    // if (__is_close(b, c, tol)) {
    //     eqbc = true;
    // }
    // if (eqab && eqac) eqabc = true;
    // if (__is_close(alpha, M_PI_2, tol)) {
    //     alpha90 = true;
    // }
    // if (__is_close(beta, M_PI_2, tol)) {
    //     beta90 = true;
    // }
    // if (__is_close(gamma, M_PI_2, tol)) {
    //     gamma90 = true;
    // }

    // if (eqabc && alpha90 && beta90 && gamma90) {
    //     // 1. Cubic
    //     ctype = CUBI;
    //     // can swap any axis
    // } else if () {
    //     // 2. Hexagonal
    //     ;
    // }
    return 0;
}

BravsLatt get_bravs_type(Atoms& atom, double tol)
{
    niggli_reduce(atom, tol);
    std::vector<double> cellpars = atom.get_cell_parameters();
    double &l1 = cellpars[0];
    double &l2 = cellpars[1];
    double &l3 = cellpars[2];
    double &alpha = cellpars[3];
    double &beta = cellpars[4];
    double &gamma = cellpars[5];
    std::vector<BravsLatt> queue_brv;
    std::vector<CellType> queue_ctp;
    BravsLatt best = BravsLatt::cF;
    CellType celltype = CellType::CUBI;
    // I. compare cell parameters
    // if (__is_close()) {
    //     ;
    // }
    return best;
}
int find_primitive_2d(Atoms& atoms, int cdir1, int cdir2, double tol)
{
    if (cdir1 < 0 || cdir1 > 2 || cdir2 < 0 || cdir2 > 2 || cdir1 == cdir2)
        throw std::invalid_argument("2D primitive reduction requires two distinct cell directions");
    for (int axis = 0; axis < 3; ++axis)
        if (atoms.pbc_[axis] != (axis == cdir1 || axis == cdir2))
            throw std::invalid_argument("2D primitive reduction requires exactly the selected two periodic directions");
    if (atoms.size() == 0 || atoms.pos_.size() != atoms.chemmap_.size() ||
        !std::isfinite(tol) || tol <= 0 || !atoms.cell_.allFinite())
        throw std::invalid_argument("2D primitive reduction requires nonempty finite geometry and a positive tolerance");
    Eigen::Matrix<double, 2, 3> basis;
    basis.row(0) = atoms.cell_.row(cdir1);
    basis.row(1) = atoms.cell_.row(cdir2);
    const Eigen::Matrix2d gram = basis * basis.transpose();
    const auto decomposition = gram.fullPivLu();
    if (!gram.allFinite() || !decomposition.isInvertible())
        throw std::invalid_argument("2D primitive reduction requires independent periodic vectors");
    const Eigen::Matrix<double, 2, 3> dual = decomposition.solve(basis);
    if (!dual.allFinite() || (!atoms.cartesian && !atoms.cell_.fullPivLu().isInvertible()))
        throw std::invalid_argument("2D fractional input requires a finite full cell; singular slabs must store Cartesian positions");

    // Work on geometry only. A rejected/no-op search must not rewrite labels or
    // the caller's coordinate representation, including a missing open cell row.
    Atoms geometry;
    geometry.cell_ = atoms.cell_; geometry.pbc_ = atoms.pbc_;
    geometry.pos_ = atoms.pos_; geometry.chemmap_ = atoms.chemmap_;
    geometry.cartesian = atoms.cartesian;
    if (!geometry.cartesian) geometry.to_cartesian();
    for (const auto &position : geometry.pos_)
        if (!position.allFinite()) throw std::invalid_argument("2D primitive reduction requires finite positions");
    geometry.wrap();
    Atoms metric;
    metric.cell_ = geometry.cell_; metric.pbc_ = geometry.pbc_;
    metric.pos_.resize(2);
    auto same_position = [&](const Eigen::RowVector3d &a, const Eigen::RowVector3d &b) {
        metric.pos_[0] = a; metric.pos_[1] = b;
        return metric.get_rab(0, 1).norm() <= tol;
    };
    auto is_translation = [&](const Eigen::RowVector3d &translation) {
        std::vector<bool> used(geometry.size(), false);
        for (int i = 0; i < geometry.size(); ++i) {
            const Eigen::RowVector3d target = geometry.pos_[i] + translation;
            bool matched = false;
            for (int j = 0; j < geometry.size(); ++j) {
                if (!used[j] && geometry.chemmap_[i] == geometry.chemmap_[j] &&
                    same_position(target, geometry.pos_[j])) {
                    used[j] = true; matched = true; break;
                }
            }
            if (!matched) return false;
        }
        return true;
    };

    // Every translation sends an anchor to a same-species atom in the same
    // layer. Test that candidate against the WHOLE structure, not just a layer.
    const auto counts = geometry.types_count_map();
    const int least_type = std::min_element(counts.begin(), counts.end(),
        [](const auto &a, const auto &b) { return a.second < b.second; })->first;
    int anchor = 0;
    while (geometry.chemmap_[anchor] != least_type) ++anchor;
    std::vector<Eigen::RowVector2d> group{Eigen::RowVector2d::Zero()};
    for (int i = 0; i < geometry.size(); ++i) {
        if (geometry.chemmap_[i] != least_type) continue;
        const Eigen::RowVector3d displacement = geometry.pos_[i] - geometry.pos_[anchor];
        Eigen::RowVector2d fraction = displacement * dual.transpose();
        if ((displacement - fraction * basis).norm() > tol) continue;
        for (int axis = 0; axis < 2; ++axis) fraction[axis] -= std::floor(fraction[axis]);
        const Eigen::RowVector3d translation = fraction * basis;
        bool duplicate = false;
        for (const auto &existing : group)
            if (same_position(translation, existing * basis)) { duplicate = true; break; }
        if (!duplicate && is_translation(translation)) group.push_back(fraction);
    }
    const int multiplicity = group.size();
    if (multiplicity == 1 || geometry.size() % multiplicity != 0) return 1;

    // The finite translation group defines an index-m superlattice of Z^2.
    // Its canonical representatives plus the two original vectors contain a
    // primitive pair. Snap the inverse to an integer supercell matrix, then
    // revalidate its exact translations and all resulting atomic orbits.
    group.emplace_back(1., 0.); group.emplace_back(0., 1.);
    for (const auto &a : group) for (const auto &b : group) {
        Eigen::Matrix2d transform; transform.row(0) = a; transform.row(1) = b;
        if (transform.determinant() <= 0) continue;
        const Eigen::Matrix2d inverse = transform.inverse();
        if (!inverse.allFinite() || inverse.cwiseAbs().maxCoeff() > geometry.size() + 1.) continue;
        const Eigen::Matrix<long long, 2, 2> integer = inverse.array().round().cast<long long>();
        if (integer.determinant() != multiplicity) continue;
        const Eigen::Matrix<double, 2, 3> primitive = integer.cast<double>().inverse() * basis;
        if (!primitive.allFinite() || !is_translation(primitive.row(0)) || !is_translation(primitive.row(1))) continue;

        metric.cell_.row(cdir1) = primitive.row(0);
        metric.cell_.row(cdir2) = primitive.row(1);
        std::vector<int> representatives, orbit_sizes;
        for (int i = 0; i < geometry.size(); ++i) {
            int orbit = 0;
            for (; orbit < representatives.size(); ++orbit) {
                const int j = representatives[orbit];
                if (geometry.chemmap_[i] == geometry.chemmap_[j] &&
                    same_position(geometry.pos_[i], geometry.pos_[j])) break;
            }
            if (orbit == representatives.size()) { representatives.push_back(i); orbit_sizes.push_back(0); }
            ++orbit_sizes[orbit];
        }
        metric.cell_ = geometry.cell_;
        if (representatives.size() != geometry.size() / multiplicity ||
            std::any_of(orbit_sizes.begin(), orbit_sizes.end(),
                [&](int size) { return size != multiplicity; })) continue;

        Atoms reduced;
        reduced.cell_ = geometry.cell_; reduced.pbc_ = geometry.pbc_;
        reduced.cell_.row(cdir1) = primitive.row(0);
        reduced.cell_.row(cdir2) = primitive.row(1);
        for (int i : representatives) {
            reduced.pos_.push_back(geometry.pos_[i]); reduced.chemmap_.push_back(geometry.chemmap_[i]);
        }
        copy_reduction_properties(atoms, reduced);
        reduced.wrap();
        if (!atoms.cartesian) reduced.to_fractional();
        atoms = std::move(reduced);
        return multiplicity;
    }
    return 1;
}

}
