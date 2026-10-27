// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "io/gpio.h"
#include "utils/compressed_stream.h"
#include "symmetry/intersitials.h"
#include "tools/chem.h"
#include "base/neighborhoodKD.h"
#include <cmath>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <sstream>
#include <array>
#include <limits>
#include <cstdint>
#include <memory>
#include <chrono>

using namespace std;
using namespace Eigen;

// Covalent radii are now in Chemicals::covalent_radii[Z] (tools/chem.h)

/**
 * @brief Compute PBC-aware minimum distance between two Cartesian positions.
 */
// Keep only the crystal geometry and two reusable Cartesian positions. The
// shared Atoms metric chooses the nearest lattice image for skew cells too.
struct PeriodicMetric {
    Atoms pair;
    explicit PeriodicMetric(const Matrix3d &cell, bool periodic=true) {
        pair.cell_ = cell;
        pair.pbc_ = {periodic, periodic, periodic};
        pair.cartesian = true;
        pair.pos_.resize(2);
    }
    RowVector3d displacement(const RowVector3d &p1, const RowVector3d &p2) {
        pair.pos_[0] = p1;
        pair.pos_[1] = p2;
        return pair.get_rab(1, 0);
    }
};

static inline double pbc_dist(const RowVector3d &p1, const RowVector3d &p2,
                               PeriodicMetric &metric)
{
    return metric.displacement(p1, p2).norm();
}

/**
 * @brief Compute PBC-aware minimum distance from a point to any position in a list.
 */
static double min_pbc_dist(const RowVector3d &pt,
                            const vector<RowVector3d> &refs,
                            PeriodicMetric &metric)
{
    double dmin = 1e30;
    for (auto &r : refs) {
        double d = pbc_dist(pt, r, metric);
        if (d < dmin) dmin = d;
    }
    return dmin;
}

static inline RowVector3d wrap_frac(RowVector3d frac)
{
    for (int d = 0; d < 3; ++d) {
        frac(d) -= floor(frac(d));
        if (fabs(frac(d) - 1.0) < 1e-8) frac(d) = 0.0;
    }
    return frac;
}

static int64_t hash_wrapped_frac(const RowVector3d &pos, const Matrix3d &invcell,
                                 double scale = 1e5)
{
    RowVector3d frac = wrap_frac(pos * invcell);
    int64_t ix = llround(frac(0) * scale);
    int64_t iy = llround(frac(1) * scale);
    int64_t iz = llround(frac(2) * scale);
    const int64_t mask = (1LL << 21) - 1;
    return ((ix & mask) << 42) ^ ((iy & mask) << 21) ^ (iz & mask);
}

static int64_t bucket_key(int ix, int iy, int iz)
{
    return (int64_t(ix) << 42) ^ (int64_t(iy) << 21) ^ int64_t(iz);
}

struct PeriodicPointGrid {
    Atoms *atom = nullptr;
    Matrix3d invcell;
    mutable PeriodicMetric metric;
    double cutoff = 0.0;
    bool nonperiodic = false;
    std::array<int, 3> nbucket{1, 1, 1};
    std::array<int, 3> bucket_span{1, 1, 1};
    unordered_map<int64_t, vector<int>> buckets;

    PeriodicPointGrid(Atoms &atom_in, double cutoff_in)
        : atom(&atom_in), invcell(Matrix3d::Zero()), metric(atom_in.cell_), cutoff(cutoff_in)
    {
        nonperiodic = !atom->pbc_[0] && !atom->pbc_[1] && !atom->pbc_[2];
        metric.pair.pbc_ = atom->pbc_;
        if (nonperiodic) return;
        invcell=atom->cell_.inverse();
        for (int d = 0; d < 3; ++d)
            nbucket[d] = max(1, int(floor(atom->cell_.row(d).norm() / cutoff)));
        for (int d = 0; d < 3; ++d)
            bucket_span[d] = max(1, int(ceil(cutoff * invcell.col(d).norm() * nbucket[d])) + 1);

        buckets.reserve(atom->size() * 2);
        for (int i = 0; i < atom->size(); ++i) {
            auto b = bucket_of(atom->pos_[i]);
            buckets[bucket_key(b[0], b[1], b[2])].push_back(i);
        }
    }

    std::array<int, 3> bucket_of(const RowVector3d &pos) const
    {
        RowVector3d frac = wrap_frac(pos * invcell);
        std::array<int, 3> b;
        for (int d = 0; d < 3; ++d) {
            b[d] = int(floor(frac(d) * nbucket[d]));
            if (b[d] >= nbucket[d]) b[d] = 0;
        }
        return b;
    }

    double nearest_dist(const RowVector3d &pt) const
    {
        double dmin = numeric_limits<double>::max();
        if (nonperiodic) {
            for (const auto &p:atom->pos_) dmin=min(dmin,(pt-p).norm());
            return dmin;
        }
        auto b = bucket_of(pt);
        for (int dx = -bucket_span[0]; dx <= bucket_span[0]; ++dx) {
            int bx = (b[0] + dx) % nbucket[0];
            if (bx < 0) bx += nbucket[0];
            for (int dy = -bucket_span[1]; dy <= bucket_span[1]; ++dy) {
                int by = (b[1] + dy) % nbucket[1];
                if (by < 0) by += nbucket[1];
                for (int dz = -bucket_span[2]; dz <= bucket_span[2]; ++dz) {
                    int bz = (b[2] + dz) % nbucket[2];
                    if (bz < 0) bz += nbucket[2];
                    auto it = buckets.find(bucket_key(bx, by, bz));
                    if (it == buckets.end()) continue;
                    for (int ia : it->second) {
                        double d = pbc_dist(pt, atom->pos_[ia], metric);
                        if (d <= cutoff) dmin = min(dmin, d);
                    }
                }
            }
        }
        if (dmin < numeric_limits<double>::max()) return dmin;

        for (int ia = 0; ia < atom->size(); ++ia) {
            double d = pbc_dist(pt, atom->pos_[ia], metric);
            dmin = min(dmin, d);
        }
        return dmin;
    }

    bool has_within(const RowVector3d &pt, double radius) const
    {
        if (nonperiodic) {
            for (const auto &p:atom->pos_) if((pt-p).norm()<radius) return true;
            return false;
        }
        auto b = bucket_of(pt);
        for (int dx = -bucket_span[0]; dx <= bucket_span[0]; ++dx) {
            int bx = (b[0] + dx) % nbucket[0];
            if (bx < 0) bx += nbucket[0];
            for (int dy = -bucket_span[1]; dy <= bucket_span[1]; ++dy) {
                int by = (b[1] + dy) % nbucket[1];
                if (by < 0) by += nbucket[1];
                for (int dz = -bucket_span[2]; dz <= bucket_span[2]; ++dz) {
                    int bz = (b[2] + dz) % nbucket[2];
                    if (bz < 0) bz += nbucket[2];
                    auto it = buckets.find(bucket_key(bx, by, bz));
                    if (it == buckets.end()) continue;
                    for (int ia : it->second) {
                        if (pbc_dist(pt, atom->pos_[ia], metric) < radius)
                            return true;
                    }
                }
            }
        }
        return false;
    }
};

static vector<RowVector3d> filter_by_mindist(const vector<RowVector3d> &candidates,
                                             const PeriodicPointGrid &grid,
                                             double min_dist)
{
    vector<RowVector3d> filtered;
    filtered.reserve(candidates.size());
    for (const auto &cp : candidates) {
        if (!grid.has_within(cp, min_dist)) filtered.push_back(cp);
    }
    return filtered;
}

// Average a periodic cluster in the image around its first member. Wrapped
// Cartesian coordinates can lie on opposite sides of the simulation cell.
static vector<RowVector3d> cluster_centers(const vector<RowVector3d> &candidates,
                                           const vector<int> &labels,
                                           int count,
                                           const Matrix3d &cell,
                                           const Matrix3d &invcell, bool periodic=true)
{
    PeriodicMetric metric(cell,periodic);
    vector<RowVector3d> anchors(count), sums(count, RowVector3d::Zero());
    vector<int> sizes(count, 0);
    for (int i = 0; i < (int)candidates.size(); ++i) {
        const int label = labels[i];
        if (sizes[label] == 0) anchors[label] = candidates[i];
        sums[label] += anchors[label] + metric.displacement(candidates[i], anchors[label]);
        ++sizes[label];
    }
    vector<RowVector3d> centers;
    centers.reserve(count);
    for (int label = 0; label < count; ++label)
        if(periodic) centers.push_back(wrap_frac((sums[label] / sizes[label]) * invcell) * cell);
        else centers.push_back(sums[label] / sizes[label]);
    return centers;
}

static vector<RowVector3d> cluster_candidates_slow(const vector<RowVector3d> &candidates,
                                                   const Matrix3d &cell,
                                                   const Matrix3d &invcell,
                                                   double cluster_tol, bool periodic=true)
{
    PeriodicMetric metric(cell,periodic);
    vector<int> labels(candidates.size(), -1);
    int next_label = 0;

    for (int i = 0; i < (int)candidates.size(); ++i) {
        if (labels[i] >= 0) continue;
        labels[i] = next_label;
        for (int j = i + 1; j < (int)candidates.size(); ++j) {
            if (labels[j] >= 0) continue;
            double dist = pbc_dist(candidates[i], candidates[j], metric);
            if (dist < cluster_tol) labels[j] = next_label;
        }
        ++next_label;
    }

    return cluster_centers(candidates, labels, next_label, cell, invcell,periodic);
}

static vector<RowVector3d> cluster_candidates_fast(const vector<RowVector3d> &candidates,
                                                   const Matrix3d &cell,
                                                   const Matrix3d &invcell,
                                                   double cluster_tol)
{
    if (candidates.empty() || cluster_tol <= 0.0) return candidates;

    std::array<int, 3> nbucket;
    for (int d = 0; d < 3; ++d)
        nbucket[d] = max(1, int(floor(cell.row(d).norm() / cluster_tol)));

    std::array<int, 3> bucket_span;
    for (int d = 0; d < 3; ++d)
        bucket_span[d] = max(1, int(ceil(cluster_tol * invcell.col(d).norm() * nbucket[d])) + 1);

    unordered_map<int64_t, vector<int>> buckets;
    buckets.reserve(candidates.size() * 2);

    auto bucket_of = [&](const RowVector3d &pos) {
        RowVector3d frac = wrap_frac(pos * invcell);
        std::array<int, 3> b;
        for (int d = 0; d < 3; ++d) {
            b[d] = int(floor(frac(d) * nbucket[d]));
            if (b[d] >= nbucket[d]) b[d] = 0;
        }
        return b;
    };

    for (int i = 0; i < (int)candidates.size(); ++i) {
        auto b = bucket_of(candidates[i]);
        buckets[bucket_key(b[0], b[1], b[2])].push_back(i);
    }

    PeriodicMetric metric(cell);
    vector<int> labels(candidates.size(), -1);
    int next_label = 0;
    for (int i = 0; i < (int)candidates.size(); ++i) {
        if (labels[i] >= 0) continue;
        labels[i] = next_label;
        auto b = bucket_of(candidates[i]);
        for (int dx = -bucket_span[0]; dx <= bucket_span[0]; ++dx) {
            int bx = (b[0] + dx) % nbucket[0];
            if (bx < 0) bx += nbucket[0];
            for (int dy = -bucket_span[1]; dy <= bucket_span[1]; ++dy) {
                int by = (b[1] + dy) % nbucket[1];
                if (by < 0) by += nbucket[1];
                for (int dz = -bucket_span[2]; dz <= bucket_span[2]; ++dz) {
                    int bz = (b[2] + dz) % nbucket[2];
                    if (bz < 0) bz += nbucket[2];
                    auto it = buckets.find(bucket_key(bx, by, bz));
                    if (it == buckets.end()) continue;
                    for (int j : it->second) {
                        if (j <= i || labels[j] >= 0) continue;
                        double dist = pbc_dist(candidates[i], candidates[j], metric);
                        if (dist < cluster_tol) labels[j] = next_label;
                    }
                }
            }
        }
        ++next_label;
    }

    return cluster_centers(candidates, labels, next_label, cell, invcell);
}

/**
 * @brief Parse a comma-separated string into a vector of doubles.
 *        e.g. "0.5,0.25,0.75" -> {0.5, 0.25, 0.75}
 */
static vector<double> parse_comma_doubles(const string &s)
{
    vector<double> out;
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        out.push_back(stod(token));
    }
    return out;
}

/**
 * @brief Parse a comma-separated string of element symbols into a set of atomic numbers.
 *        e.g. "O,Fe" -> {8, 26}
 */
static set<int> parse_elements(const string &s)
{
    set<int> out;
    stringstream ss(s);
    string token;
    while (getline(ss, token, ',')) {
        // trim whitespace
        token.erase(0, token.find_first_not_of(" \t"));
        token.erase(token.find_last_not_of(" \t") + 1);
        if (token.empty()) continue;
        int z = Chemicals::atomic_numbers.count(token)
                    ? Chemicals::atomic_numbers.at(token)
                    : 0;
        if (z <= 0) {
            for (int i = 1; i < 118; ++i) {
                if (Chemicals::chemical_symbols[i] == token) { z = i; break; }
            }
        }
        if (z > 0) out.insert(z);
        else cerr << "Warning: unknown element \"" << token << "\" ignored." << endl;
    }
    return out;
}

/**
 * @brief Find interstitial sites in a crystal structure
 *
 * Uses Voronoi tessellation to locate candidate interstitial positions
 * (vertices, edge-centers, face-centers of Voronoi cells), then clusters
 * and filters them to find the best sites for inserting small atoms
 * such as H, C, N, O.
 *
 * Algorithm (full-cell mode):
 *   1. Voronoi tessellation with element-specific covalent radii
 *   2. Collect vertices, edge-centers, face-centers as candidates
 *   3. Filter by minimum distance from host atoms
 *   4. Cluster nearby candidates (merge duplicates from adjacent cells)
 *   5. Rank by cavity quality (minimum distance to nearest host atom)
 *   6. Select top-N sites or output all
 *
 * Algorithm (local mode, --local):
 *   1. Build KD-tree for fast neighbor search (with PBC)
 *   2. Find atoms near the query point within search_radius
 *   3. Build local Voronoi only with those nearby atoms
 *   4. Extract sites from well-formed Voronoi cells (excluding truncated ones)
 *   5. Filter, merge, deduplicate
 *   Use with --near-elem / --near-pos / --near-idx to specify query region.
 *   Near-filters auto-enable local mode.
 *
 * Usage:
 *   findint POSCAR
 *   findint POSCAR --insert=H --top=5 --min-dist=0.4
 *   findint POSCAR --insert=C --output-format=vasp -o POSCAR_CH
 *   findint POSCAR --local --near-elem=O                       # local Voronoi near O atoms
 *   findint POSCAR --near-elem=Fe,Cr --near-radius=2.0          # auto local, near Fe/Cr
 *   findint POSCAR --near-pos=0.5,0.5,0.5                      # auto local, near cell center
 *   findint POSCAR --near-idx=1,5,10 --near-radius=1.5          # auto local, near specific atoms
 */
int findint(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;

    // ==== Parameters ====
    string infile = "POSCAR";
    string informat = "";
    string outfile = "POSCAR_INT";
    string outformat = "";
    string insert_elem = "";        // element to insert (empty = list sites only)
    string indexfile = "";
    int top_n = -1;                 // keep top-N sites (-1 = all after filtering)
    double min_dist = 0.4;          // min distance from host (Angstrom)
    double cluster_tol = 0.8;       // cluster tolerance for merging (Angstrom)
    bool use_vertices = true;
    bool use_edges = false;         // edges give many redundant sites
    bool use_faces = false;         // faces are often same as vertices
    double volprec = 0.5;           // skip tiny Voronoi cells
    double edgeprec = 0.2;
    double faceprec = 0.2;
    bool write_sites_only = false;  // output only interstitial sites (no host)
    bool no_fast_cluster = false;   // debug fallback to the old O(M^2) clustering
    int verbosity = 0;

    // ---- Near-site filtering / local Voronoi parameters ----
    string near_elem = "";          // element symbol(s) to search near (comma-sep)
    string near_pos_frac = "";      // fractional position "x,y,z" to search near
    string near_pos_car = "";       // Cartesian position "x,y,z" to search near
    string near_idx_str = "";       // atom index/indices to search near (1-based, comma-sep)
    double near_radius = 3.0;       // search radius in Angstrom
    bool use_local = false;         // use local Voronoi (KD-tree accelerated)

    cmdopts.set_usage("findint <structure> [options]")
        .addop(infile, "input", "", "input structure file", true)
        .addop(informat, "input-format", "if", "input structure format")
        .addop(indexfile, "index-file", "id", "chemical species map (one symbol per type index); use --near-idx for atom selection")
        .addop(outfile, "output", "o", "output file (default: POSCAR_INT)")
        .addop(outformat, "output-format", "of", "output format")
        .addop(insert_elem, "insert", "", "element to insert at best sites (e.g. H, C, N, O)")
        .addop(top_n, "top", "n", "keep top-N sites by cavity size (-1 = all, default: -1)")
        .addop(min_dist, "min-dist", "md", "minimum distance from host atoms in Angstrom (default: 0.4)")
        .addop(cluster_tol, "cluster-tol", "ct", "cluster tolerance for merging duplicate sites (default: 0.8)")
        .addop(use_vertices, "vertices", "v", "use Voronoi vertices (default: yes)", false, 0)
        .addop(use_edges, "edges", "", "use Voronoi edge centers (default: no)", false, 0)
        .addop(use_faces, "faces", "", "use Voronoi face centers (default: no)", false, 0)
        .addop(volprec, "volprec", "vp", "minimum Voronoi cell volume (default: 0.5)")
        .addop(write_sites_only, "sites-only", "", "output interstitial sites only (no host atoms)", false, 0)
        .addop(no_fast_cluster, "no-fast-cluster", "", "use legacy O(N^2) candidate clustering", false, 0)
        .addop(verbosity, "verbose", "vb", "verbosity level")
        // Near-site filtering options
        .addop(near_elem, "near-elem", "ne", "search sites near element(s) (comma-sep, e.g. O or Fe,Cr)")
        .addop(near_pos_frac, "near-pos", "np", "search sites near fractional position (e.g. 0.5,0.5,0.5)")
        .addop(near_pos_car, "near-pos-car", "npc", "search sites near Cartesian position (e.g. 1.0,2.0,3.0)")
        .addop(near_idx_str, "near-idx", "ni", "search sites near atom index/indices (1-based, comma-sep)")
        .addop(near_radius, "near-radius", "nr", "radius for near-site search in Angstrom (default: 3.0)")
        .addop(use_local, "local", "lcl", "use local Voronoi (periodic crystals: KD images; FFF: Euclidean neighbors)", false, 0)
        .add_dummy("-")
        .phrase_options(argc, argv);

    // ==== Validate ====
    int insert_id = 1;
    if (!insert_elem.empty()) {
        const auto found = Chemicals::atomic_numbers.find(insert_elem);
        if (found == Chemicals::atomic_numbers.end() || found->second <= 0) {
            cerr << "Error: unknown insertion element: " << insert_elem << endl;
            return 128;
        }
        insert_id = found->second;
    }
    if (!GPIO::file_exist(infile)) {
        cerr << "Error: input file \"" << infile << "\" not found" << endl;
        return 128;
    }

    // ==== Read structure ====
    GPIO iobj;
    if (indexfile.empty()) {
        iobj.setmap();
    } else {
        if (!iobj.readmap(indexfile)) {
            cerr << "Error: failed to read chemical species map: " << indexfile << endl;
            return 1;
        }
        // readmap is a legacy permissive parser. This command needs chemical
        // symbols and fully parsed numeric properties, not an atom-index list.
        auto map_input = open_ifile(indexfile);
        if (!map_input) {
            cerr << "Error: failed to read chemical species map: " << indexfile << endl;
            return 1;
        }
        string line, name, property;
        set<string> seen;
        bool blank_tail = false;
        while (getline(*map_input, line)) {
            istringstream row(line);
            if (!(row >> name)) { blank_tail = true; continue; }
            const auto element = Chemicals::atomic_numbers.find(name);
            bool valid = !blank_tail && element != Chemicals::atomic_numbers.end() &&
                         element->second > 0 && seen.insert(name).second;
            while (valid && row >> property) {
                try {
                    size_t consumed = 0;
                    const double value = stod(property, &consumed);
                    valid = consumed == property.size() && isfinite(value);
                } catch (const exception &) { valid = false; }
            }
            if (!valid) {
                cerr << "Error: invalid chemical species map: " << indexfile << endl;
                return 1;
            }
        }
        if (map_input->bad() || seen.empty()) {
            cerr << "Error: failed to read chemical species map: " << indexfile << endl;
            return 1;
        }
    }
    Atoms atom;
    vector<Atoms> atoms_tmp;
    if (!iobj.read(atoms_tmp, infile, informat)) {
        cerr << "Error: failed to read input structure: " << infile << endl;
        return 1;
    }
    if (atoms_tmp.empty()) {
        cerr << "Error: no structures in input file" << endl;
        return 128;
    }
    atom = atoms_tmp[0];
    if (!indexfile.empty()) {
        // Voronoi radii, element filters and insertion IDs use atomic numbers.
        const auto input_names = iobj.getmap();
        for (int &type : atom.chemmap_) {
            const auto name = input_names.find(type);
            if (name == input_names.end()) {
                cerr << "Error: input type missing from chemical species map: " << type << endl;
                return 1;
            }
            type = Chemicals::atomic_numbers.at(name->second);
        }
        // The inserted element need not already occur in the input type map.
        iobj.setmap();
    }

    const bool nonperiodic=!atom.pbc_[0] && !atom.pbc_[1] && !atom.pbc_[2];
    if(!atom.cartesian) atom.to_cartesian();

    if (verbosity >= 0) {
        cout << "# Read structure: " << atom.size() << " atoms, "
             << atom.get_types_set().size() << " element types" << endl;
        auto cellp = atom.get_cell_parameters();
        cout << "# Cell: a=" << cellp[0] << " b=" << cellp[1] << " c=" << cellp[2]
             << " alpha=" << cellp[3] << " beta=" << cellp[4] << " gamma=" << cellp[5] << endl;
    }

    // ==== Build atomic radii map ====
    map<int, double> atomradii;
    set<int> types = atom.get_types_set();
    for (int t : types) {
        double r = Chemicals::covalent_radii[t];
        atomradii[t] = r;
        if (verbosity > 0) {
            cout << "#   " << Chemicals::chemical_symbols[t]
                 << " (Z=" << t << ") radius = " << r << " A" << endl;
        }
    }

    // ==== Determine strategy: local or full-cell Voronoi ====
    bool has_near_filter = (!near_elem.empty() ||
                            !near_pos_frac.empty() ||
                            !near_pos_car.empty() ||
                            !near_idx_str.empty());
    bool do_local = use_local || has_near_filter;
    double kd_radius = max(3.0, max(min_dist, cluster_tol) + 0.5);
    if (do_local) kd_radius = max(kd_radius, near_radius);
    unique_ptr<NeighborhoodKD> nkd;
    if (do_local && !nonperiodic) nkd = make_unique<NeighborhoodKD>(&atom, kd_radius);
    unique_ptr<PeriodicPointGrid> host_grid;

    // ==== Find candidate interstitial sites ====
    vector<RowVector3d> candidates;
    using Clock = chrono::steady_clock;

    if (do_local) {
        // ---- Local Voronoi: KD-tree accelerated, only nearby atoms ----
        // Build reference position list
        vector<RowVector3d> ref_pos;

        // near-elem
        if (!near_elem.empty()) {
            set<int> target_Z = parse_elements(near_elem);
            if (verbosity > 0) {
                cout << "# Near-filter: element(s) =";
                for (int z : target_Z) cout << " " << Chemicals::chemical_symbols[z];
                cout << endl;
            }
            for (int ia = 0; ia < atom.size(); ++ia) {
                if (target_Z.count(atom.chemmap_[ia]))
                    ref_pos.push_back(atom.pos_[ia]);
            }
            if (verbosity > 0)
                cout << "#   Found " << ref_pos.size() << " reference atoms"
                     << " of specified element(s)" << endl;
        }

        // near-idx
        if (!near_idx_str.empty()) {
            stringstream ss(near_idx_str);
            string token;
            while (getline(ss, token, ',')) {
                token.erase(0, token.find_first_not_of(" \t"));
                token.erase(token.find_last_not_of(" \t") + 1);
                if (token.empty()) continue;
                int idx = stoi(token) - 1;
                if (idx >= 0 && idx < atom.size())
                    ref_pos.push_back(atom.pos_[idx]);
                else
                    cerr << "Warning: atom index " << (idx+1)
                         << " out of range (1-" << atom.size() << "), ignored." << endl;
            }
            if (verbosity > 0)
                cout << "#   Added " << ref_pos.size() << " atoms by index" << endl;
        }

        // near-pos (fractional)
        if (!near_pos_frac.empty()) {
            vector<double> xyz = parse_comma_doubles(near_pos_frac);
            if (xyz.size() != 3) {
                cerr << "Error: --near-pos requires exactly 3 comma-separated values" << endl;
                return 128;
            }
            RowVector3d car_pos = RowVector3d(xyz[0], xyz[1], xyz[2]) * atom.cell_;
            ref_pos.push_back(car_pos);
            if (verbosity > 0)
                cout << "#   Added reference position (frac): "
                     << xyz[0] << " " << xyz[1] << " " << xyz[2] << endl;
        }

        // near-pos-car (Cartesian)
        if (!near_pos_car.empty()) {
            vector<double> xyz = parse_comma_doubles(near_pos_car);
            if (xyz.size() != 3) {
                cerr << "Error: --near-pos-car requires exactly 3 comma-separated values" << endl;
                return 128;
            }
            RowVector3d car_pos(xyz[0], xyz[1], xyz[2]);
            ref_pos.push_back(car_pos);
            if (verbosity > 0)
                cout << "#   Added reference position (car): "
                     << xyz[0] << " " << xyz[1] << " " << xyz[2] << endl;
        }

        // If no reference specified with --local alone, use cell center
        if (ref_pos.empty()) {
            RowVector3d center(0.5, 0.5, 0.5);
            if(nonperiodic) {
                center.setZero();for(const auto &p:atom.pos_) center+=p;
                center/=atom.size();ref_pos.push_back(center);
            } else ref_pos.push_back(center * atom.cell_);
            if (verbosity > 0)
                cout << "# Local Voronoi: using cell center as query point" << endl;
        }

        if (verbosity >= 0)
            cout << "# Local Voronoi: " << ref_pos.size()
                 << " query point(s), search radius = " << near_radius << " A" << endl;

        // Run local Voronoi for each query point, merge results
        Matrix3d invcell=Matrix3d::Zero();if(!nonperiodic) invcell=atom.cell_.inverse();
        unordered_set<int64_t> seen_hashes;

        for (size_t ir = 0; ir < ref_pos.size(); ++ir) {
            vector<RowVector3d> local_sites;
            if(nonperiodic) get_intersitial_sites_nonperiodic(atom,local_sites,atomradii,
                use_vertices,use_edges,use_faces,min_dist,volprec,edgeprec,faceprec,&ref_pos[ir],near_radius);
            else get_intersitial_sites_near(atom, ref_pos[ir], local_sites, atomradii,
                                       near_radius,
                                       use_vertices, use_edges, use_faces,
                                       min_dist, volprec, edgeprec, faceprec,
                                       nkd.get());

            for (auto &s : local_sites) {
                if(nonperiodic) {
                    bool duplicate=false;
                    for(const auto &old:candidates) if((s-old).norm()<1e-5) {duplicate=true;break;}
                    if(!duplicate) candidates.push_back(s);
                    continue;
                }
                auto h = hash_wrapped_frac(s, invcell);
                if (seen_hashes.count(h) == 0) {
                    seen_hashes.insert(h);
                    candidates.push_back(s);
                }
            }
            if (verbosity > 0)
                cout << "#   Query point " << (ir+1) << ": "
                     << local_sites.size() << " sites (" << candidates.size()
                     << " unique so far)" << endl;
        }

        if (verbosity >= 0)
            cout << "# Local Voronoi total: " << candidates.size() << " candidates" << endl;

    } else {
        // ---- Full-cell Voronoi (original approach) ----
        auto t_voro0 = Clock::now();
        if(nonperiodic) get_intersitial_sites_nonperiodic(atom,candidates,atomradii,
            use_vertices,use_edges,use_faces,min_dist,volprec,edgeprec,faceprec);
        else get_intersitial_sites(atom, candidates, atomradii,
                              use_vertices, use_edges, use_faces,
                              min_dist, volprec, edgeprec, faceprec);
        auto t_voro1 = Clock::now();

        if (verbosity > 0)
            cout << "# Voronoi candidates: " << candidates.size() << endl;
        if (verbosity > 0)
            cout << "# Timing: Voronoi = "
                 << chrono::duration<double>(t_voro1 - t_voro0).count()
                 << " s" << endl;

        if (candidates.empty()) {
            cout << "# No interstitial sites found." << endl;
            return 0;
        }

        // Filter: distance from host atoms
        auto t_filter0 = Clock::now();
        host_grid = make_unique<PeriodicPointGrid>(atom, kd_radius);
        candidates = filter_by_mindist(candidates, *host_grid, min_dist);
        auto t_filter1 = Clock::now();
        if (verbosity > 0)
            cout << "# After distance filter (> " << min_dist << " A): "
                 << candidates.size() << endl;
        if (verbosity > 0)
            cout << "# Timing: distance filter = "
                 << chrono::duration<double>(t_filter1 - t_filter0).count()
                 << " s" << endl;

        if (candidates.empty()) {
            cout << "# No sites pass the distance filter. Try lowering --min-dist." << endl;
            return 0;
        }

        // Clustering: merge near-duplicate sites
        {
            auto t_cluster0 = Clock::now();
            Matrix3d invcell=Matrix3d::Zero();if(!nonperiodic) invcell=atom.cell_.inverse();
            if (no_fast_cluster || nonperiodic)
                candidates = cluster_candidates_slow(candidates, atom.cell_, invcell, cluster_tol,!nonperiodic);
            else
                candidates = cluster_candidates_fast(candidates, atom.cell_, invcell, cluster_tol);
            auto t_cluster1 = Clock::now();
            if (verbosity > 0)
                cout << "# Timing: clustering = "
                     << chrono::duration<double>(t_cluster1 - t_cluster0).count()
                     << " s" << endl;
        }
        if (verbosity > 0)
            cout << "# After clustering (tol " << cluster_tol << " A): "
                 << candidates.size() << " unique sites" << endl;
    }

    if (candidates.empty()) {
        cout << "# No interstitial sites found." << endl;
        return 0;
    }
    if (!host_grid) host_grid = make_unique<PeriodicPointGrid>(atom, kd_radius);

    // ==== Rank by cavity quality (max min-distance to host atoms) ====
    {
        auto t_rank0 = Clock::now();
        Matrix3d invcell=Matrix3d::Zero();if(!nonperiodic) invcell=atom.cell_.inverse();
        vector<pair<double, RowVector3d>> ranked;
        for (auto &cp : candidates) {
            double max_gap = host_grid->nearest_dist(cp);
            ranked.push_back({max_gap, cp});
        }
        sort(ranked.begin(), ranked.end(),
             [](auto &a, auto &b) { return a.first > b.first; });

        candidates.clear();
        for (auto &r : ranked) candidates.push_back(r.second);
        auto t_rank1 = Clock::now();
        if (verbosity > 0)
            cout << "# Timing: ranking = "
                 << chrono::duration<double>(t_rank1 - t_rank0).count()
                 << " s" << endl;

        if (verbosity >= 0) {
            int show_n = min((int)ranked.size(), max(top_n, 20));
            cout << "# Ranked sites (top " << show_n << "):" << endl;
            cout << "#   Rank  Cavity(A)  Position (" << (nonperiodic?"Cartesian Angstrom":"fractional") << ")" << endl;
            for (int i = 0; i < show_n; ++i) {
                RowVector3d frac=ranked[i].second;
                if(!nonperiodic) frac=(frac*invcell).eval();
                for (int d = 0; !nonperiodic && d < 3; ++d) {
                    frac(d) -= floor(frac(d));
                    if (fabs(frac(d) - 1.0) < 1e-8) frac(d) = 0.0;
                }
                cout << "#   " << (i+1) << "     "
                     << ranked[i].first << "      "
                     << frac(0) << " " << frac(1) << " " << frac(2) << endl;
            }
        }
    }

    // ==== Select top-N ====
    if (top_n > 0 && top_n < candidates.size()) {
        candidates.resize(top_n);
    }

    // ==== Output ====
    // Inserting atoms creates a new structure. Host labels do not describe it,
    // and host-sized arrays cannot be indexed for the inserted atoms.
    Atoms out_atom;
    out_atom.cell_ = atom.cell_;
    out_atom.pbc_ = atom.pbc_;
    out_atom.cartesian = atom.cartesian;
    if (!write_sites_only) {
        out_atom.pos_ = atom.pos_;
        out_atom.chemmap_ = atom.chemmap_;
    }
    out_atom.pos_.insert(out_atom.pos_.end(), candidates.begin(), candidates.end());
    out_atom.chemmap_.insert(out_atom.chemmap_.end(), candidates.size(), insert_id);
    if (!iobj.write_one(out_atom, outfile, outformat)) {
        cerr << "Error: failed to write interstitial output: " << outfile << endl;
        return 1;
    }

    if (verbosity >= 0) {
        int nout = candidates.size();
        cout << "Output: " << outfile << "  (" << out_atom.size()
             << " atoms, " << nout << " interstitial";
        if (!insert_elem.empty()) cout << " " << insert_elem;
        cout << ")" << endl;
    }

    return 0;
}
