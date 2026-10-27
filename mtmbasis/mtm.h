// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once 

#include "tools/procstr.h"
#include "utils/gputils.h"
#include "mathextra/combinatorics.h"
#include <atomic>

namespace mtmbasis_detail {
// Internal diagnostic: counts actual canonicalization calls, never cache hits.
inline std::atomic<unsigned long long> tensor_solve_calls{0};
}

/**
 * @brief 
 * TODO: Given {mu, nu} pairs to constraint alpha_basic_index_count
 */

// using namespace std;
using namespace Eigen;

// #define EQU_TEST_USE_STDFORM

#ifndef _DEBUG_OUTPUT_
#define _DEBUG_OUTPUT_(a) std::cout << __FILE__ << " LINE: " << __LINE__ << " name " << #a << " " << a << std::endl;
#define _DEBUG_VECTOR_(a) std::cout << __FILE__ << " LINE: " << __LINE__ << " name " << #a << " " ; for (auto &it : a) {std::cout << it << " ";}; std::cout << std::endl;
#endif

struct alpha_index_type {
    VectorXi first;
    MatrixXi second;
    MatrixXi stdform;
    std::vector<std::vector<int>> canonical_orders;

    // Identity
    VectorXd exeigenval;
    // MatrixXd exeigvec;
    // MatrixXd ieigvec;
    // VectorXd exeigvecsum;
    int mu, nv;
    VectorXi musorted, nvsorted;
    VectorXi diag;

    int index_k_id; //  different for different permutation of equvalent tensor
    int tensor_id; // unique for each equvalent tensor group
    int basic_tensor_id; // unique for each tensor group, where first array is all zero. 

    // Flags
    bool solved = false;
    bool have_smeval = false;
    bool fast_test = false;
    // std::vector<std::vector<int>> smevalids;

    size_t size() const {
        return first.size();
    };
    // Build an exact canonical representative under simultaneous vertex
    // permutation.  Dimensions are limited to seven by mtmbasis, so the
    // exhaustive permutation search is small enough and avoids relying on
    // non-unique spectral or row-signature invariants.
    void GetStandard() {
        const int n = static_cast<int>(size());
        std::vector<int> permutation(n);
        std::iota(permutation.begin(), permutation.end(), 0);
        std::vector<int> best;
        bool have_best = false;
        canonical_orders.clear();
        do {
            std::vector<int> candidate;
            candidate.reserve(n + n * n);
            for (int i = 0; i < n; ++i)
                candidate.push_back(first(permutation[i]));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    candidate.push_back(second(permutation[i], permutation[j]));
            if (!have_best || candidate < best) {
                best = std::move(candidate);
                have_best = true;
                canonical_orders.assign(1, permutation);
            } else if (candidate == best) {
                canonical_orders.push_back(permutation);
            }
        } while (std::next_permutation(permutation.begin(), permutation.end()));

        stdform = MatrixXi::Zero(n + 1, n + 1);
        for (int i = 0; i < n; ++i) {
            stdform(0, i + 1) = best[i];
            stdform(i + 1, 0) = best[i];
        }
        int offset = n;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < n; ++j)
                stdform(i + 1, j + 1) = best[offset++];
    }

    void split_k(int k, alpha_index_type &bl, alpha_index_type &br) const {
        int n = size();
        bl.resize(k);
        br.resize(n - k);
        bl.first = first.head(k);
        br.first = first.tail(n - k);
        bl.second = second.block(0,0,k,k);
        br.second = second.block(k,k,n-k,n-k);
        for (int ii = 0; ii < k; ii++)
        {
            for (int jj = k; jj < n; jj++)
            {
                bl.second(ii, ii) += second(ii, jj);
            }
        }
        for (int ii = k; ii < n; ii++)
        {
            for (int jj = 0; jj < k; jj++)
            {
                br.second(ii - k, ii - k) += second(ii, jj);
            }
        }
    }
    // make sure idsleft is in ascending order!!!
    void split(std::vector<int> &idsleft, 
                alpha_index_type &bl, 
                alpha_index_type &br) const {
        bl.resize(idsleft.size());
        br.resize(size() - bl.size());
        alpha_index_type B;
        B.first = first;
        B.second = second;
        for (int i = 0; i < idsleft.size(); ++i) {
            if (i == idsleft[i]) continue;
            // 1. swap to i
            B.bswap(idsleft[i], i);
        }
        int k = idsleft.size();
        int n = size();
        bl.first = B.first.head(k);
        br.first = B.first.tail(n - k);
        bl.second = B.second.block(0, 0, k, k);
        br.second = B.second.block(k, k, n - k, n - k);
        for (int ii = 0; ii < k; ii++)
        {
            for (int jj = k; jj < n; jj++)
            {
                bl.second(ii, ii) += B.second(ii, jj);
            }
        }
        for (int ii = k; ii < n; ii++)
        {
            for (int jj = 0; jj < k; jj++)
            {
                br.second(ii - k, ii - k) += B.second(ii, jj);
            }
        }
    }

    void split_reorder(std::vector<int> &idsleft, 
                alpha_index_type &bl, 
                alpha_index_type &br) {
        bl.resize(idsleft.size());
        br.resize(size() - bl.size());
        for (int i = 0; i < idsleft.size(); ++i) {
            if (i == idsleft[i]) continue;
            // 1. swap to i
            bswap(idsleft[i], i);
        }
        int k = idsleft.size();
        int n = size();
        bl.first = first.head(k);
        br.first = first.tail(n - k);
        bl.second = second.block(0, 0, k, k);
        br.second = second.block(k, k, n - k, n - k);
        for (int ii = 0; ii < k; ii++)
        {
            for (int jj = k; jj < n; jj++)
            {
                bl.second(ii, ii) += second(ii, jj);
            }
        }
        for (int ii = k; ii < n; ii++)
        {
            for (int jj = 0; jj < k; jj++)
            {
                br.second(ii - k, ii - k) += second(ii, jj);
            }
        }
    }

    alpha_index_type(int n) {
        resize(n);
    }
    // firstly sort by nv value, then nv array, then mu value, then mu array
    bool operator<(const alpha_index_type &b) const {
        if (first.size() < b.first.size()) return true;
        if (first.size() > b.first.size()) return false;
        if (mu < b.mu) return true;
        if (mu > b.mu) return false;
        if (nv < b.nv) return true;
        if (nv > b.nv) return false;
        assert(solved);
        assert(b.solved);
        for (int i = 0; i < stdform.rows(); ++i) {
            for (int j = 0; j < stdform.cols(); ++j) {
                if (stdform(i, j) < b.stdform(i, j)) return true;
                if (stdform(i, j) > b.stdform(i, j)) return false;
            }
        }
        return false;
    }

    bool is_scalar() const {
        return second.trace() == 0;
    }

    alpha_index_type(){};
    
    void resize(int n) {
        first = VectorXi::Zero(n);
        second = MatrixXi::Zero(n, n);
        stdform.resize(0, 0);
        canonical_orders.clear();
        solved = false;
    }

    void bswap(int i, int j) {
        second.row(i).swap(second.row(j));
        second.col(i).swap(second.col(j));
        std::swap(first(i), first(j));
        if (solved) {
            for (auto &order : canonical_orders) {
                for (int &source : order) {
                    if (source == i)
                        source = j;
                    else if (source == j)
                        source = i;
                }
            }
        }
    }

    // sort mu, store block props
    void solve() {
        mtmbasis_detail::tensor_solve_calls.fetch_add(1, std::memory_order_relaxed);
        int n = first.size();
        musorted = first;
        nvsorted.resize(n);
        diag.resize(n);
        for (int i = 0; i < n; ++i) {
            nvsorted(i) = second.col(i).sum();
            diag(i) = second(i, i);
        }
        mu = musorted.sum();
        nv = nvsorted.sum();
        // bubble sort
        bool unsorted = true;
        while(unsorted) {
            unsorted = false;
            for (int i = 0; i < n-1; ++i) {
                if (musorted[i] > musorted[i+1]) {
                    unsorted = true;
                    std::swap(musorted[i], musorted[i+1]);
                    std::swap(nvsorted[i], nvsorted[i+1]);
                    std::swap(diag[i], diag[i+1]);
                } else if (musorted[i] == musorted[i+1]) {
                    if (nvsorted[i] > nvsorted[i+1]) {
                        unsorted = true;
                        std::swap(nvsorted[i], nvsorted[i+1]);
                        std::swap(diag[i], diag[i+1]);
                    }
                }
            }
        }
        GetStandard();
        solved = true;
    }

    bool is_same(const alpha_index_type &opt) const {
        return (first == opt.first) && (second == opt.second);
    }

    bool operator==(const alpha_index_type &opt) const {
        assert(solved);
        assert(opt.solved);
        // Check if B has been solved or not
        if (first.size() != opt.first.size()) return false;
        if (musorted != opt.musorted) return false;
        if (nvsorted != opt.nvsorted) return false;
        if (first.size() == 1) {
            if (first == opt.first && second == opt.second) return true;
            return false;
        }
        return stdform == opt.stdform;
    }

    friend std::ostream& operator<<(std::ostream &ofs, const alpha_index_type &opt) {
        // ofs << opt.first.transpose() << '\n';
        // for (int i = 0; i < opt.size(); ++i) {
        //     ofs << "  " << opt.second.row(i) << '\n';
        // }
        ofs << "Bid: " << opt.tensor_id << '\n';
        int s = opt.first.size();
        auto &b = opt;
        ofs << "mus: ";
        int dimt = b.first.size(), mut = 0, nvt = 0;
        for (int mi = 0; mi < dimt; ++mi) {
            mut += b.first(mi);
            ofs << b.first(mi) << " ";
        }
        ofs << '\n';
        ofs << "nvs: ";
        for (int mi = 0; mi < dimt; ++mi) {
            nvt += b.second.col(mi).sum();
            ofs << b.second.col(mi).sum() << " ";
        }
        ofs << '\n';
        ofs << "level: " << 0 << " dim " << dimt << " maxpoly " << nvt << '\n';

        ofs << b.first.transpose() << '\n';
        for (int j = 0; j < s; ++j) {
            ofs << "  ";
            for (int k = 0; k < s; ++k) {
                ofs << b.second(j, k) <<  "  ";
            }
            ofs << "\n";
        }
        ofs << "\n";
        return ofs;
    }
};

namespace mtmbasis_detail {

std::vector<alpha_index_type> enumerate_scalar_alpha_indices(
        const std::vector<int> &mus,
        const std::vector<int> &nvs);

std::vector<int> select_low_trace_decompositions(
        const std::vector<std::pair<int, int>> &candidates,
        int kselect);

void validate_alpha_topology(
        int alpha_index_basic_count,
        int alpha_moments_count,
        const std::vector<std::array<int, 4>> &alpha_index_times,
        const std::vector<int> &alpha_moment_mapping);

std::vector<double> normalized_selection_cdf(
        const std::vector<double> &fitness);

bool weighted_fitness_better(
        const std::pair<int, int> &candidate,
        const std::pair<int, int> &reference,
        double moment_weight,
        double rule_weight);

bool progress_report_due(
        int completed,
        int total,
        int max_reports = 10);

} // namespace mtmbasis_detail

// struct alpha_decomp_connection {
//     // index_k_id
//     int rootid;
//     int left, right;
// };

typedef std::array<int, 3> alpha_decomp_connection; // root, left, right
struct alpha_times_triple_hash {
    size_t operator()(const std::array<int, 3> &k) const {
        return std::hash<int>()(k[0]) * 1000003 ^
               std::hash<int>()(k[1]) * 97 ^ std::hash<int>()(k[2]);
    }
};

typedef std::unordered_map<std::array<int, 3>, int, alpha_times_triple_hash> alpha_times_index_flag_type;

struct alpha_decomp_tree {
    ;
};


// single B, collections of all possible decomp trees
class SingleTreeCollection {
public:
    std::vector<std::set<alpha_decomp_connection>> connections;
    int rootid;
};

struct non_ele_id_cache_key_hash {
    size_t operator()(const std::vector<int> &v) const {
        size_t h = 0x9e3779b97f4a7c15ULL;
        for (int x : v) {
            h ^= std::hash<int>()(x) + 0x9e3779b97f4a7c15ULL +
                 (h << 6) + (h >> 2);
        }
        return h;
    }
};

using non_ele_id_canonical_cache_type =
    std::unordered_map<std::vector<int>, std::vector<int>,
                       non_ele_id_cache_key_hash>;

struct non_ele_id_type {
    // tensorid beta1(x,y,z), beta2(x,y,z), ...
    int tid;
    std::vector<int> mu; // mu for each mt tensor comp
    std::vector<int> nv; // nv for each mt tensor comp
    std::vector<int> ind; // TODO: change to std::vector<std::array<int, 3>> ind;
    std::vector<int> canonical;
    non_ele_id_type(
            int id,
            const alpha_index_type &shape,
            const std::vector<int> &muind,
            const std::vector<int> &nvind,
            const std::vector<int> &vind,
            non_ele_id_canonical_cache_type *canonical_cache = nullptr)
    {
        tid = id;
        mu = muind;
        nv = nvind;
        ind = vind;
        assert(mu.size() *3 == ind.size());
        assert(mu.size() == nv.size());
        assert(mu.size() == shape.size());

        // GetStandard() already found every permutation that maps this local
        // tensor ordering to its canonical structure.  Only those orders can
        // minimize the combined structure/component key; enumerating all n!
        // permutations again for every component is redundant.
        const int n = static_cast<int>(shape.size());
        alpha_index_type solved_shape;
        const std::vector<std::vector<int>> *orders = &shape.canonical_orders;
        if (!shape.solved || orders->empty()) {
            solved_shape = shape;
            solved_shape.solve();
            orders = &solved_shape.canonical_orders;
        }
        // Memoize the canonical key, but only for shapes with a large
        // automorphism group.  The cache is owned by one PickTensors run and
        // the key includes the shape in its current vertex order, so tensor
        // ids reused by another pool or differently ordered shapes cannot
        // return a canonical component computed from incompatible orders.
        // For shapes with a small group the plain loop is cheaper than the
        // cache lookup, so it is skipped there.
        bool have_canonical = false;
        const bool use_cache = canonical_cache != nullptr && orders->size() > 8;
        if (use_cache) {
            std::vector<int> cache_key;
            cache_key.reserve(2 + n + n * n + mu.size() + nv.size() +
                              ind.size());
            cache_key.push_back(tid);
            cache_key.push_back(n);
            for (int i = 0; i < n; ++i)
                cache_key.push_back(shape.first(i));
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    cache_key.push_back(shape.second(i, j));
            cache_key.insert(cache_key.end(), mu.begin(), mu.end());
            cache_key.insert(cache_key.end(), nv.begin(), nv.end());
            cache_key.insert(cache_key.end(), ind.begin(), ind.end());
            auto cached = canonical_cache->find(cache_key);
            if (cached != canonical_cache->end()) {
                canonical = cached->second;
                return;
            }
            for (const auto &permutation : *orders) {
                std::vector<int> candidate;
                candidate.reserve(5 * n);
                for (int i = 0; i < n; ++i) {
                    const int source = permutation[i];
                    candidate.push_back(mu[source]);
                    candidate.push_back(nv[source]);
                    for (int component = 0; component < 3; ++component)
                        candidate.push_back(ind[3 * source + component]);
                }
                if (!have_canonical || candidate < canonical) {
                    canonical = std::move(candidate);
                    have_canonical = true;
                }
            }
            canonical_cache->emplace(std::move(cache_key), canonical);
            return;
        }
        for (const auto &permutation : *orders) {
            std::vector<int> candidate;
            candidate.reserve(5 * n);
            for (int i = 0; i < n; ++i) {
                const int source = permutation[i];
                candidate.push_back(mu[source]);
                candidate.push_back(nv[source]);
                for (int component = 0; component < 3; ++component)
                    candidate.push_back(ind[3 * source + component]);
            }
            if (!have_canonical || candidate < canonical) {
                canonical = std::move(candidate);
                have_canonical = true;
            }
        }
    }
    friend std::ostream &operator<<(std::ostream &os, const non_ele_id_type &opt) {
        os << fmt::format("{} | {} | {} | {}", opt.tid, fmt::join(opt.mu.begin(), opt.mu.end(), " "), fmt::join(opt.nv.begin(), opt.nv.end(), " "), fmt::join(opt.ind.begin(), opt.ind.end(), " "));
        return os;
    }
    bool operator<(const non_ele_id_type &opt) const
    {
        if (tid < opt.tid)
            return true;
        else if (tid > opt.tid)
            return false;
        return canonical < opt.canonical;
    }
    bool operator==(const non_ele_id_type &opt) const
    {
        return tid == opt.tid && canonical == opt.canonical;
    }
};

struct non_ele_id_hash {
    size_t operator()(const non_ele_id_type &k) const {
        size_t h = std::hash<int>()(k.tid);
        for (int x : k.canonical) {
            h ^= std::hash<int>()(x) + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
        }
        return h;
    }
};

struct looper {
    std::vector<int> lsize;
    std::vector<int> loop;
    looper(std::vector<int> &lpsize)
    {
        lsize = lpsize;
        loop.assign(lsize.size(), 0);
    }
    int size()
    {
        int ret = 1;
        for (int i = 0; i < lsize.size(); ++i)
            ret *= lsize[i];
        return ret;
    }
    void next()
    {
        loop[0] += 1;
        for (int i = 0; i < loop.size() - 1; ++i)
        {
            if (loop[i] >= lsize[i])
            {
                loop[i] = 0;
                loop[i + 1] += 1;
            }
        }
        if (loop.back() >= lsize.back())
            loop.assign(lsize.size(), 0);
    }
};

class PickTensors : public ConnectSettings {
public:
    int max_dim = 4, max_mu = 4, max_nv = 10, max_lev = 16;
    int max_mt_nv = 1500; // maxium nv of a moment tensor, can set equal to max_nv, in Shapeev's definition, this should not considered, so I give a large value here
    int max_basic_index_allowed = -1; // currently not used, to constraint {mu, nv} pair. One may give it directly
    int max_poly_allowed = 10000; // maxium polynomial order allowed, acturally sum of nv
    int kselect = 2; // select 
    bool apply_reuse_rule_first = false;
    bool apply_reuse_rule_second = false;
    double mu_wei = 2, nv_wei = 1, dim_wei = 2;
    // all tensors
    std::vector<alpha_index_type> huge_pool; // huge_pool contains tensors with different permutations
    std::vector<alpha_index_type> tensor_pool;
    // generated alphas, all scalar
    std::vector<alpha_index_type> alphas_gen;
    // connections
    // vector<alpha_decomp_connection> all_connections;
    std::map<int, std::set<alpha_decomp_connection>> all_connections;
    std::map<int, std::vector<alpha_decomp_connection>> all_connections_vec;
    std::map<alpha_decomp_connection, Eigen::MatrixXi> connection_contraction;
    std::set<int> alpha_gen_ids; // for generated scalar alphas, add id to here to check if B is in alphas_gen or not

    // 
    std::set<int> tensors_have_different_branchs_present;

    // for count
    std::map<alpha_decomp_connection, int> alpha_moments_count_add_count;
    std::map<alpha_decomp_connection, int> alpha_index_times_count_add_count;

    // id collection
    std::set<int> scalar_ids;
    std::set<int> mt_ids; // elementary tensors, moment tensors
    std::vector<int> scalar_ids_vec;
    std::vector<int> mt_ids_vec;
    // set<int> mt_ids_set;

    // simple hashing for fast compare
    // vector of different dims
    // 
    std::vector<std::map<int, std::set<int>>> hashmaparr;
    std::vector<std::map<int, std::set<int>>> hashmaparr_k;
    // return tensor id
    int add_to_tensor_pool(alpha_index_type &b);
    void add_to_connection_pool(alpha_index_type &b, alpha_index_type &bl, alpha_index_type &br) {
        alpha_decomp_connection conn;
        conn[0] = b.tensor_id;
        conn[1] = bl.tensor_id;
        conn[2] = br.tensor_id;
        if (conn[1] > conn[2]) {
            std::swap(conn[1], conn[2]);
        }
        all_connections[b.tensor_id].insert(conn);
    }
    bool is_in_tensor_pool(alpha_index_type &b);

    // vector<int> node_alpha_size;  // each dim where it starts and ends

    // for filter
    std::set<int> picked_already;
    std::set<int> need_decision;
    // map<int, alpha_decomp_connection> tensors_in_use_first;
    std::set<int> decided_decomp_tensors_first;

    // mapping alpha_basic_ids to tensor id, to fill mid_tid_mapping
    std::map<int, int> basic_mid_tid_mapping;

    // Per-generation memoization for symmetric component canonicalization.
    // It is cleared whenever a new tensor pool starts generation.
    non_ele_id_canonical_cache_type non_ele_id_canonical_cache;

    // Persistent generation state: the searched pot accumulates across
    // candidate evaluations.  Each call generates only the moments/rules of
    // connections not processed before; the candidate's pot is materialized
    // from these structures (filtered to the candidate's connections).
    std::unordered_map<non_ele_id_type, int, non_ele_id_hash> bi_id_map_persist;
    int next_moment_id_persist = 0;
    std::vector<std::array<int, 4>> alpha_index_times_persist;
    alpha_times_index_flag_type alpha_times_flag_persist;
    std::vector<int> alpha_moment_mapping_persist;
    std::map<int, int> mid_tid_mapping_persist; // moment id -> tensor id
    // moments of each root tensor, in creation order (global ids); the
    // candidate's moments are the union over its roots
    std::map<int, std::vector<int>> root_moment_ids;
    std::map<alpha_decomp_connection, std::pair<int, int>> conn_rule_range;
    // connection that first created each moment (to detect shared moments)
    std::map<int, alpha_decomp_connection> moment_creator;

    // for first loop
    std::set<int> tensors_must_appear_first;
    std::set<alpha_decomp_connection> conns_must_use_first;
    // results of first loop
    std::unordered_map<non_ele_id_type, int, non_ele_id_hash> bi_id_map_first;
    int alpha_moments_count_first;
    int alpha_index_times_count_first;
    std::vector<std::array<int, 4>> alpha_index_times_first;
    alpha_times_index_flag_type alpha_times_flag_first;
    std::vector<int> alpha_moment_mapping_first;

    // select
    int alpha_moments_count_min = INT_MAX;
    int alpha_index_times_count_min = INT_MAX;
    int cnt = 0;
    bool restart = false; // restart from initial guess
    bool rndinit = false; // random initial guess
    bool reassign = false; // reassign a solution when chopping trees
    bool chop_trees_first = true;
    bool chop_trees = true;
    int chop_start = 0; // only for opt mode 3, internal.

    // temper
    double T = 200;
    double w1 = 1.0;
    double w2 = 4.0;
    int maxiter = 1000000;

    int mode = 0; 

    // GA
    double k_fit = 0.002;
    double c_fit = 0.01;
    double frac_mutation_min = 0.2;
    double frac_mutation_max = 1.0;
    int max_mutation_sites = 4;
    double min_exchange_fraction = 0.1;
    double max_exchange_fraction = 0.5;
    int popsize = 10; // maintained pop size (num. survived)
    int childsize = 30; // current pop size
    int ga_seed = -1;
    int elite_count = 0;
    std::string initial_population = "legacy";
    int rstart = 0;
    int rstop = -1;
    std::string output_file = "potential.mtp";
    // temper
    double uacc_ratio = 0.5;
    
    std::vector<SingleTreeCollection> collections;
    std::vector<int> ga_upperlim;    

    std::map<int, std::vector<std::vector<int>>> mg1, mg2;


public:

    PickTensors(){ InitSettings(); };
    PickTensors(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
        max_dim = std::min(max_lev / 2, max_dim);
// PrintSettings();
    }

    void InitSettings() {
        Connect(add_new_scalar_to_pot, "add_new_scalar_to_pot");
        Connect(max_dim, "max_dim");
        Connect(max_mu, "max_mu");
        Connect(max_nv, "max_nv");
        Connect(max_lev, "max_lev");
        Connect(mu_wei, "mu_wei");
        Connect(nv_wei, "nv_wei");
        Connect(dim_wei, "dim_wei");
        Connect(max_poly_allowed, "max_poly");
        Connect(kselect, "kselect");
        Connect(T, "T_init");
        Connect(w1, "w1");
        Connect(w2, "w2");
        Connect(chop_trees_first, "chop_trees_first");
        Connect(chop_trees, "chop_trees");
        Connect(reassign, "reassign");
        Connect(uacc_ratio, "uacc");
        // ga
        Connect(k_fit, "kfit");
        Connect(c_fit, "cfit");
        Connect(frac_mutation_min, "frac_mutation_min");
        Connect(frac_mutation_max, "frac_mutation_max");
        Connect(max_mutation_sites, "max_mutation_sites");
        Connect(min_exchange_fraction, "min_ex_frac");
        Connect(max_exchange_fraction, "max_ex_frac");
        Connect(popsize, "popsize");
        Connect(childsize, "childsize");
        Connect(ga_seed, "seed");
        Connect(elite_count, "elite_count");
        Connect(initial_population, "initial_population");
        Connect(rstart, "rstart");
        Connect(rstop, "rstop");

        Connect(maxiter, "maxiter");
        Connect(restart, "restart");
        Connect(rndinit, "randominit");
        Connect(rndinit, "rndinit");
        Connect(mode, "mode");
        Connect(apply_reuse_rule_first, "filter_init");
        Connect(apply_reuse_rule_second, "filter_second");
    }

    void load_alpha_pool_from_file(const std::string &poolfile);
    void clean_pool(std::vector<alpha_index_type> &pool); // remove dups, usually process loaded pools
    void save_pool(const std::string &filename, std::vector<alpha_index_type> &pool);

    // generate alpha select pool
    void gen_pot_by_shapeev_level(std::string mtfile = "");
    void gen_alpha_pool_by_shapeev_level(std::string mtfile = "");
    void gen_pot_by_alphapool_GA();
    // generate alphas recursively
    void recursive_gen(
            std::vector<int> &ll,
            int lpos,
            int cmu,
            int cnv,
            std::vector<std::vector<int>> &results,
            std::vector<int> &mus,
            std::vector<int> &nus);
    // create connection graphs, find all possible tensors
    void create_hugepool_and_connections();
    void generate_trees();
    void create_collection(int scalar_id, SingleTreeCollection &collection, int branch);
    void create_collection(int tensorid, SingleTreeCollection &collection, std::vector<std::set<alpha_decomp_connection>> connection, int branch);
    std::pair<int, int> build_pot_from_tree(std::vector<int> &select, std::vector<SingleTreeCollection> &collections);
    void build_init();
    void reduce_tree();

    bool need_regen_first;
    bool add_new_scalar_to_pot = true; // add new scalar detected during generation to moments

    void GAsearch();

    void Allsearch();

    void regen_from_old();

    size_t alpha_index_basic_count = 0;
    size_t radial_funcs_count;
    std::vector<std::array<int, 4>> alpha_index_basic;
    std::map<std::array<int, 4>, int> alpha_index_basic_map;

    // size_t alpha_index_times_count = 0;
    // size_t alpha_moments_count;
    // vector<int> alpha_moment_mapping;


    void gen_alpha_basic_index();
    void gen_alpha_index_times(const std::vector<int> &tensors, const std::vector<alpha_decomp_connection> &conns,
                std::unordered_map<non_ele_id_type, int, non_ele_id_hash> &bi_id_map_this,
                int &alpha_moments_count_this,
                int &alpha_index_times_count_this,
                std::vector<std::array<int, 4>> &alpha_index_times_this,
                alpha_times_index_flag_type &alpha_times_flag_this,
                std::vector<int> &alpha_moment_mapping_this,
                std::set<int> &tensor_pool_used_this,
                std::map<int, int> &mid_tid_mapping);

    void iter_search(std::vector<int> &ss, 
                std::vector<int> &csel,
                std::vector<SingleTreeCollection> &cur_colls, int depth, int branch);
    
    // dump all info
    void dump_all();
    void dump_post();
    void save_potential(const std::string potname,
                    int alpha_moments_count_this,
                    int alpha_index_times_count_this,
                    std::vector<std::array<int, 4>> &alpha_index_times_this,
                    std::vector<int> &alpha_moment_mapping_this);

    alpha_index_type restore_b(const alpha_index_type &b,
                               const alpha_index_type &bl,
                               const alpha_index_type &br) const;
    bool restore_b_test(const alpha_index_type &b,
                        const alpha_index_type &bl,
                        const alpha_index_type &br) const;
};

/***
 * 1. generate all alpha index
 * 2. split all index iteratevely, generate all possible connections
 * 3. GA search
*/

inline double calc_level(int mu, int nv, int dim, double mu_wei, double nv_wei, double dim_wei) {
    return mu * mu_wei + nv * nv_wei + dim * dim_wei;
}
