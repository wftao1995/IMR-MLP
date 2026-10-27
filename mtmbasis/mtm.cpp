// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file mtm.cpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2024-03-26
 * @note 
 *  1. alpha_moment_mapping is NOT the same order as alphapool.dat or alphas_gen.dat
 * 
 * @copyright Copyright (c) 2024
 * 
 */

#include "mtm.h"
#include "tools/mtp_metadata.h"
#include "ga_population.h"
#include <fstream>
#include <bitset>
#include <stack>

namespace mtmbasis_detail {

std::vector<int> select_low_trace_decompositions(
        const std::vector<std::pair<int, int>> &candidates,
        int kselect) {
    if (kselect <= 0) {
        GPERROR("mtmbasis kselect must be positive");
    }
    if (candidates.empty()) return {};

    auto ordered = candidates;
    std::stable_sort(ordered.begin(), ordered.end(),
        [](const auto &left, const auto &right) {
            return left.second < right.second;
        });
    std::vector<int> distinct;
    for (const auto &[_, trace] : ordered) {
        if (distinct.empty() || distinct.back() != trace) distinct.push_back(trace);
    }
    const int threshold = distinct[std::min<int>(kselect, distinct.size()) - 1];
    std::vector<int> selected;
    for (const auto &[index, trace] : ordered) {
        if (trace <= threshold) selected.push_back(index);
    }
    return selected;
}

void validate_alpha_topology(
        int alpha_index_basic_count,
        int alpha_moments_count,
        const std::vector<std::array<int, 4>> &alpha_index_times,
        const std::vector<int> &alpha_moment_mapping) {
    if (alpha_index_basic_count < 0 ||
        alpha_moments_count < alpha_index_basic_count) {
        GPERROR("alpha_moments_count is smaller than alpha_index_basic_count");
    }

    std::vector<char> ready(alpha_moments_count, 0);
    for (int i = 0; i < alpha_index_basic_count; ++i) ready[i] = 1;
    int previous_target = alpha_index_basic_count - 1;
    for (const auto &rule : alpha_index_times) {
        for (int field : {0, 1, 3}) {
            if (rule[field] < 0 || rule[field] >= alpha_moments_count) {
                GPERROR("alpha_index_times contains an out-of-range moment id");
            }
        }
        if (rule[2] <= 0) {
            GPERROR("alpha_index_times contains a non-positive multiplication coefficient");
        }
        if (!ready[rule[0]] || !ready[rule[1]]) {
            GPERROR("alpha_index_times is not topologically ordered");
        }
        if (rule[3] < alpha_index_basic_count ||
            rule[0] >= rule[3] || rule[1] >= rule[3] ||
            rule[3] < previous_target) {
            GPERROR("alpha_index_times does not use increasing non-basic targets");
        }
        ready[rule[3]] = 1;
        previous_target = rule[3];
    }
    for (int moment_id = alpha_index_basic_count;
         moment_id < alpha_moments_count;
         ++moment_id) {
        if (!ready[moment_id]) {
            GPERROR("alpha_moments_count contains an undefined non-basic moment");
        }
    }
    for (int moment_id : alpha_moment_mapping) {
        if (moment_id < 0 || moment_id >= alpha_moments_count) {
            GPERROR("alpha_moment_mapping contains an out-of-range moment id");
        }
        if (!ready[moment_id]) {
            GPERROR("alpha_moment_mapping references an uncomputed moment");
        }
    }
}

std::vector<double> normalized_selection_cdf(
        const std::vector<double> &fitness) {
    std::vector<double> cdf(fitness.size() + 1, 0.0);
    if (fitness.empty()) return cdf;
    double sum = 0.0;
    for (double value : fitness) {
        if (!std::isfinite(value) || value < 0.0) {
            GPERROR("mtmbasis GA fitness must be finite and non-negative");
        }
        sum += value;
    }
    if (!std::isfinite(sum) || sum == 0.0) {
        for (size_t i = 0; i < fitness.size(); ++i)
            cdf[i + 1] = static_cast<double>(i + 1) / fitness.size();
        return cdf;
    }
    for (size_t i = 0; i < fitness.size(); ++i)
        cdf[i + 1] = cdf[i] + fitness[i] / sum;
    cdf.back() = 1.0;
    return cdf;
}

bool weighted_fitness_better(
        const std::pair<int, int> &candidate,
        const std::pair<int, int> &reference,
        double moment_weight,
        double rule_weight) {
    const double candidate_score =
        moment_weight * static_cast<double>(candidate.first) +
        rule_weight * static_cast<double>(candidate.second);
    const double reference_score =
        moment_weight * static_cast<double>(reference.first) +
        rule_weight * static_cast<double>(reference.second);
    if (candidate_score != reference_score)
        return candidate_score < reference_score;
    if (candidate.second != reference.second)
        return candidate.second < reference.second;
    return candidate.first < reference.first;
}

bool progress_report_due(int completed, int total, int max_reports) {
    if (completed <= 0 || total <= 0 || completed > total || max_reports <= 0)
        return false;
    if (completed == total) return true;
    const auto previous_bucket =
        static_cast<long long>(completed - 1) * max_reports / total;
    const auto current_bucket =
        static_cast<long long>(completed) * max_reports / total;
    return current_bucket > previous_bucket;
}

} // namespace mtmbasis_detail

/**
 * @brief 
 * 
 * @bug: if a tensor decomp in different ways, how to deal this? 
 */

bool debug_echo = false;

int PickTensors::add_to_tensor_pool(alpha_index_type &b) {
    int hash;
    int first = b.first.squaredNorm();
    int second = b.second.squaredNorm();
    // NOTE: int overflow for mu around 1500+ (only possible from a custom moments
    // file; default generation keeps mu <= max_mu = 4). Hash collisions are still
    // resolved by exact comparison in the bucket, so a wrong hash only slows lookup.
    hash = first * 1000 + second;
    int dim = b.size() - 1;
    bool match = false;
    // _DEBUG_OUTPUT_(dim);
    // _DEBUG_OUTPUT_(b);
    if (hashmaparr[dim].size() == 0) {
        match = false;
    } else if (hashmaparr[dim].count(hash) == 0) {
        match = false;
    } else {
        for (auto &id : hashmaparr[dim][hash]) {
            if (b == tensor_pool[id]) {
                match = true;
                b.tensor_id = id;
                break;
            }
        }
    }
    if (!match) {
        b.tensor_id = tensor_pool.size();
        tensor_pool.emplace_back(b);
        if (b.size() == 1) mt_ids.insert(b.tensor_id);
        if (b.is_scalar()) scalar_ids.insert(b.tensor_id);
        hashmaparr[dim][hash].insert(b.tensor_id);
    }
    // hashmaparr[dim][hash].insert()
    return b.tensor_id;
}

bool PickTensors::is_in_tensor_pool(alpha_index_type &b)
{
    int hash;
    int first = b.first.squaredNorm();
    int second = b.second.squaredNorm();
    hash = first * 1000 + second;
    int dim = b.size() - 1;
    for (auto &id : hashmaparr[dim][hash]) {
        if (b == tensor_pool[id]) {
            b.tensor_id = id;
            return true;
        }
    }
    return false;
}

void PickTensors::create_hugepool_and_connections() {
    if (kselect <= 0) {
        GPERROR("mtmbasis kselect must be positive");
    }
    hashmaparr.resize(max_dim);
    std::vector<int> decompo_flags;
    // step1. generate all permutations
    /**
     * Assume B has dim k, then:
     * We have C(k,1) + C(k,2) + C(k,3) + ... = 2^(k-1) ways
     * 
     * 
     * tid is the position in tensor_pool
    */
    int tid = 0;
    // 1d
    // for (int j = node_alpha_size[0]; j < node_alpha_size[1]; ++j) {
    //     alpha_index_type &B = alphas_gen[j];
    //     int id =  add_to_tensor_pool(B);
    //     alpha_gen_ids.insert(id);
    // }
    for (int j = 0; j < alphas_gen.size(); ++j) {
        alpha_index_type &B = alphas_gen[j];
        if (B.size() == 1) {
            int id = add_to_tensor_pool(B);
            alpha_gen_ids.insert(id);
        }
    }
    decompo_flags.assign(tensor_pool.size(), 1);
    // nd
    // discard high rank tensors?
    if (max_dim > 1) {
        for (int dim = 2; dim <= max_dim; ++dim) {
            // generate combos
            std::vector<std::vector<int>> g1;
            std::vector<std::vector<int>> g2;
            std::vector<int> g(dim);
            iota(g.begin(), g.end(), 0);
            int32_t bit = 0;
            int32_t bmax = 1 << (g.size() - 1);
            int ncombo = bmax - 1;
            g1.resize(bmax-1);
            g2.resize(bmax-1);
            for (int mask = 1; mask < bmax; ++mask) {
                for (int ii = 0; ii < g.size(); ++ii) {
                    if (mask & (1 << ii)) {
                        g1[mask-1].push_back(ii);
                    } else {
                        g2[mask-1].push_back(ii);
                    }
                }
            }
            mg1[dim] = g1;
            mg2[dim] = g2;

            std::vector<alpha_index_type> bls, brs;
            // std::vector<MatrixXi> bcontarct;
            std::vector<std::pair<int, int>> sumdim;
            std::vector<int> dimsel;
            // for (int j = node_alpha_size[dim-1]; j < node_alpha_size[dim]; ++j) {
            for (int j = 0; j < alphas_gen.size(); ++j) {
                alpha_index_type bl, br;
                auto &B = alphas_gen[j]; // do not use reference here, because after tensor_pool resized, reference will become invalid
                if (B.size() != dim) continue;
                // if rank 0 dim n tensor, use first only since all these tensors will present in scalar_moments
                // if (B.second.sum() == 0) {
                //     int n = B.size();
                //     int k = n / 2;
                //     B.split_k(k, bl, br);
                //     bl.solve();
                //     br.solve();
                //     add_to_tensor_pool(bl);
                //     add_to_tensor_pool(br);
                //     add_to_connection_pool(B, bl, br);
                //     continue;
                // }
                int alpha_id = add_to_tensor_pool(B);
                if (alpha_gen_ids.count(alpha_id)) continue;
                alpha_gen_ids.insert(alpha_id);
                bls.clear();
                brs.clear();
                sumdim.clear();
                for (int s = 0; s < g1.size(); ++s) {
                    B.split(g1[s], bl, br);
                    // if (alpha_id == 23830) {
                    //     _DEBUG_std::vector_(g1[s])
                    //     _DEBUG_OUTPUT_(B);
                    //     _DEBUG_OUTPUT_(bl);
                    //     _DEBUG_OUTPUT_(br);
                    // }
                    bls.push_back(bl);
                    brs.push_back(br);
                    sumdim.push_back(std::make_pair(s, bl.second.trace() + br.second.trace()));
                }
                dimsel = mtmbasis_detail::select_low_trace_decompositions(
                    sumdim, kselect);
                for (int i = 0; i < dimsel.size(); ++i) {
                    alpha_index_type &Bl = bls[dimsel[i]];
                    alpha_index_type &Br = brs[dimsel[i]];
                    Bl.solve();
                    Br.solve();
                    // _DEBUG_OUTPUT_(B);
                    // _DEBUG_OUTPUT_(Bl);
                    // _DEBUG_OUTPUT_(Br);
                    add_to_tensor_pool(Bl);
                    add_to_tensor_pool(Br);
                    add_to_connection_pool(B, Bl, Br);
                    // update flags
                    decompo_flags.resize(tensor_pool.size(), 0);
                    decompo_flags[B.tensor_id] = 1;
                    if (all_connections.count(Bl.tensor_id)) decompo_flags[Bl.tensor_id] = 1;
                    if (all_connections.count(Br.tensor_id)) decompo_flags[Br.tensor_id] = 1;
                    if (Bl.size() == 1) decompo_flags[Bl.tensor_id] = 1;
                    if (Br.size() == 1) decompo_flags[Br.tensor_id] = 1;
                }
            }
        }
    }
    // Now all alphas_gen has been added, then find all connections
    while (! all_of(decompo_flags.begin(), decompo_flags.end(), [](const int &a) {
                    return a == 1;})) {
        std::vector<alpha_index_type> bls, brs;
        std::vector<std::pair<int, int>> sumdim;
        std::vector<int> dimsel;
        for (int i = 0; i < decompo_flags.size(); ++i) {
            if (decompo_flags[i] == 1) continue;

            alpha_index_type bl, br;
            auto B = tensor_pool[i];
            int cdim = B.size();
            if (cdim == 1) {
                std::cerr << "This should not be exectuted, check the code" << std::endl;
                _DEBUG_OUTPUT_(B);
                decompo_flags[i] = 1;
                continue;
            }
            auto &g1 = mg1[cdim];
            auto &g2 = mg2[cdim];
            bls.clear();
            brs.clear();
            sumdim.clear();

            for (int s = 0; s < g1.size(); ++s) {
                B.split(g1[s], bl, br);
                bls.push_back(bl);
                brs.push_back(br);
                sumdim.push_back(std::make_pair(s, bl.second.trace() + br.second.trace()));
            }
            dimsel = mtmbasis_detail::select_low_trace_decompositions(
                sumdim, kselect);
            for (int ii = 0; ii < dimsel.size(); ++ii) {
                alpha_index_type &Bl = bls[dimsel[ii]];
                alpha_index_type &Br = brs[dimsel[ii]];
                Bl.solve();
                Br.solve();
                add_to_tensor_pool(Bl);
                add_to_tensor_pool(Br);
                add_to_connection_pool(B, Bl, Br);
                // update flags
                decompo_flags.resize(tensor_pool.size(), 0);
                decompo_flags[B.tensor_id] = 1;
                if (all_connections.count(Bl.tensor_id)) decompo_flags[Bl.tensor_id] = 1;
                if (all_connections.count(Br.tensor_id)) decompo_flags[Br.tensor_id] = 1;
                if (Bl.size() == 1) decompo_flags[Bl.tensor_id] = 1;
                if (Br.size() == 1) decompo_flags[Br.tensor_id] = 1;
            }
        }
    }
    // all connections has been established.
    for (auto &it : all_connections) {
        all_connections_vec[it.first].resize(it.second.size());
        copy(it.second.begin(), it.second.end(), all_connections_vec[it.first].begin());
    }
    // convert to std::vector
    scalar_ids_vec.resize(scalar_ids.size());
    copy(scalar_ids.begin(), scalar_ids.end(), scalar_ids_vec.begin());
    mt_ids_vec.resize(mt_ids.size());
    copy(mt_ids.begin(), mt_ids.end(), mt_ids_vec.begin());
}

void PickTensors::dump_all()
{
    // dump alphas_gen
    {
        std::string fnm = "alphas_gen.log";
        std::ofstream ofs(fnm);
        for (int i = 0; i < alphas_gen.size(); ++i) {
            ofs << "Tensor id: " << alphas_gen[i].tensor_id << '\n';
            ofs << alphas_gen[i] << '\n';
        }
    }
    {
        std::string fnm = "alphas_gen_ids.log";
        std::ofstream ofs(fnm);
        for (auto &it : alpha_gen_ids) {
            ofs << it << '\n';
        }
        ofs.close();
    }
    // dump scalar alphas
    {
        std::string fnm = "alphas_scalar.log";
        std::ofstream ofs(fnm);
        for (auto &i : scalar_ids) {
            ofs << "Tensor id: " << tensor_pool[i].tensor_id << '\n';
            ofs << tensor_pool[i] << '\n';
        }
    }
    // dump tensor_pool
    {
        std::string fnm = "tensor_pool.log";
        std::ofstream ofs(fnm);
        for (auto &B : tensor_pool) {
            ofs << "Tensor id: " << B.tensor_id << '\n';
            ofs << B << '\n';
        }
    }
    // dump connection graph
    {
        std::string fnm = "connection_graph.log";
        std::ofstream ofs(fnm);
        for (auto &conn : all_connections) {
            ofs << "Tensor id: " << conn.first << '\n';
            ofs << "Graphs: " << '\n';
            for (auto &hh : conn.second) {
                ofs << "  " << hh[0] << " " << hh[1] << " " << hh[2] << '\n';
            }
            ofs << '\n';
        }
    }
    // dump graph collections
    {
        std::string fnm = "collection_graphs.log";
        std::ofstream ofs(fnm);
        for (int i = 0; i < collections.size(); ++i) {
            ofs << "Tensor id: " << scalar_ids_vec[i] << '\n';
            ofs << "Collections size: " << collections[i].connections.size() << '\n';
            for (int j = 0; j < collections[i].connections.size(); ++j) {
                ofs << " Collection " << j << " :\n";
                for (auto &it : collections[i].connections[j]) {
                    ofs << "  " << it[0] << " " << it[1] << " " << it[2] << '\n';
                }
            }
            ofs << '\n';
        }
    }
}

void PickTensors::dump_post() {
    // dump connections that must present
    {
        std::string fnm = "basic_connections.log";
        std::ofstream ofs(fnm);
        for (auto &it : conns_must_use_first) {
            ofs << " " << it[0] << " " << it[1] << " " << it[2] << '\n';
        }
    }
    // dump graph collections
    {
        std::string fnm = "collection_graphs_reduced.log";
        std::ofstream ofs(fnm);
        for (int i = 0; i < collections.size(); ++i) {
            if (collections[i].connections.size() < 2) continue;
            ofs << "Tensor id: " << scalar_ids_vec[i] << '\n';
            ofs << "Collections size: " << collections[i].connections.size() << '\n';
            for (int j = 0; j < collections[i].connections.size(); ++j) {
                ofs << " Collection " << j << " :\n";
                for (auto &it : collections[i].connections[j]) {
                    ofs << "  " << it[0] << " " << it[1] << " " << it[2] << '\n';
                }
            }
            ofs << '\n';
        }
    }
    // basic potential
    // {
    //     string fnm = "basic_pot.mtp";
    //     save_potential(fnm, 
    //                     alpha_moments_count_first, 
    //                     alpha_index_times_count_first,
    //                     alpha_index_times_first,
    //                     alpha_moment_mapping_first);
    // }
}

void PickTensors::save_potential(const std::string potname,
                    int alpha_moments_count_this,
                    int alpha_index_times_count_this,
                    std::vector<std::array<int, 4>> &alpha_index_times_this,
                    std::vector<int> &alpha_moment_mapping_this) {
    if (alpha_index_times_count_this != (int)alpha_index_times_this.size()) {
        GPERROR("alpha_index_times_count does not match alpha_index_times size");
    }
    std::ofstream ofs(potname + mpi.fnm_ending);
    if (!ofs.is_open()) {
        GPERROR((std::string) "Cannot open mtmbasis output file: " + potname + mpi.fnm_ending);
    }
    ofs << "MTP\n";
    ofs << "version = 1.1.0\n";
    ofs << "potential_name = MTP1m\n";
    ofs << "species_count = 1\n";
    ofs << "potential_tag = " << MTPMetadata::Value(MTPMetadata::Tag{}.output(
        {{"type", MTPMetadata::Value("MTP")}, {"model", MTPMetadata::Value("linear")}}, 1)).serialize() << '\n';
    ofs << "radial_basis_type = RBChebyshev\n";
    ofs << "\tmin_dist = 2\n";
    ofs << "\tmax_dist = 5\n";
    ofs << "\tradial_basis_size = 8\n";
    ofs << "\tradial_funcs_count = " << radial_funcs_count << "\n";
    ofs << "alpha_moments_count = " << alpha_moments_count_this << "\n";
    ofs << "alpha_index_basic_count = " << alpha_index_basic_count << "\n";
    ofs << "alpha_index_basic = {";
    for (int i = 0; i < alpha_index_basic_count; ++i)
    {
        ofs << "{" << alpha_index_basic[i][0] << ", "
            << alpha_index_basic[i][1] << ", "
            << alpha_index_basic[i][2] << ", "
            << alpha_index_basic[i][3] << "}";
        if (i != (alpha_index_basic_count - 1))
            ofs << ", ";
    }
    ofs << "}\n";
    ofs << "alpha_index_times_count = " << alpha_index_times_count_this << "\n";
    ofs << "alpha_index_times = {";
    for (int i = 0; i < alpha_index_times_count_this; ++i)
    {
        ofs << "{" << alpha_index_times_this[i][0] << ", "
            << alpha_index_times_this[i][1] << ", "
            << alpha_index_times_this[i][2] << ", "
            << alpha_index_times_this[i][3] << "}";
        if (i != (alpha_index_times_count_this - 1))
            ofs << ", ";
    }
    ofs << "}\n";
    std::sort(alpha_moment_mapping_this.begin(), alpha_moment_mapping_this.end());
    alpha_moment_mapping_this.erase(unique(alpha_moment_mapping_this.begin(), alpha_moment_mapping_this.end()), alpha_moment_mapping_this.end());
    ofs << "alpha_scalar_moments = " << alpha_moment_mapping_this.size() << "\n";
    ofs << "alpha_moment_mapping = {";
    for (int i = 0; i < alpha_moment_mapping_this.size(); ++i)
    {
        ofs << alpha_moment_mapping_this[i];
        if (i != alpha_moment_mapping_this.size() - 1)
            ofs << ", ";
    }
    ofs << "}\n";
}


alpha_index_type PickTensors::restore_b(
        const alpha_index_type &b,
        const alpha_index_type &bl,
        const alpha_index_type &br) const {
    if (bl.size() + br.size() != b.size()) {
        GPERROR("Cannot restore mtmbasis parent: child dimensions do not match");
    }
    if (restore_b_test(b, bl, br)) return b;

    const int n = static_cast<int>(b.size());
    std::vector<int> permutation(n);
    std::iota(permutation.begin(), permutation.end(), 0);
    do {
        alpha_index_type candidate(n);
        candidate.tensor_id = b.tensor_id;
        for (int i = 0; i < n; ++i) {
            candidate.first(i) = b.first(permutation[i]);
            for (int j = 0; j < n; ++j)
                candidate.second(i, j) = b.second(permutation[i], permutation[j]);
        }
        if (restore_b_test(candidate, bl, br)) {
            candidate.solve();
            return candidate;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));

    GPERROR("Unable to restore an exact parent ordering for mtmbasis decomposition");
}

bool PickTensors::restore_b_test(
        const alpha_index_type &b,
        const alpha_index_type &bl,
        const alpha_index_type &br) const {
    const int k = static_cast<int>(bl.size());
    const int n = static_cast<int>(b.size());
    if (k + static_cast<int>(br.size()) != n) return false;
    if (b.first.head(k) != bl.first || b.first.tail(n - k) != br.first) return false;

    MatrixXi left = b.second.block(0, 0, k, k);
    MatrixXi right = b.second.block(k, k, n - k, n - k);
    for (int i = 0; i < k; ++i)
        for (int j = k; j < n; ++j)
            left(i, i) += b.second(i, j);
    for (int i = k; i < n; ++i)
        for (int j = 0; j < k; ++j)
            right(i - k, i - k) += b.second(i, j);
    return left == bl.second && right == br.second;
}

void PickTensors::generate_trees() {
    // Now, build tree for each scalar
    /**
     * @brief 
     * picked_already: tensors that MUST appear
     * need_decision: this tensor MUST appear, however it has multiple childs
     * @todo: clean this code, and imporve this, i.e. non-scalar ones
     */
    if (apply_reuse_rule_first) {
        // step1. scalar ones
        for (auto &it : scalar_ids) {
            // scalars must appear
            picked_already.insert(it);
            auto &b = tensor_pool[it];
            if (b.size() == 1) {
                ;
            } else if (all_connections_vec[it].size() == 1) {
                // if scalar has only one solution, it must appear
                picked_already.insert(all_connections_vec[it][0][1]);
                picked_already.insert(all_connections_vec[it][0][2]);
            } else if (all_connections_vec[it].size() > 1) {
                need_decision.insert(it);
            }
        }
        // step2. loop until all definite ones selected
        bool not_ready = true;
        // while (not_ready) {
        //     not_ready = false;
        //     // non-scalar one dervied from scalar ones. If one of them have only one solution, it must appear
        //     for (auto &it : picked_already) {
        //         if (all_connections_vec[it].size() == 1) {
        //             auto id1 = all_connections_vec[it][0][1];
        //             auto id2 = all_connections_vec[it][0][2];
        //             if (picked_already.count(id1) == 0) {
        //                 not_ready = true;
        //                 picked_already.insert(id1);
        //             }
        //             if (picked_already.count(id2) == 0) {
        //                 not_ready = true;
        //                 picked_already.insert(id2);
        //             }
        //         } 
        //     }
        // }

        // step3. check reuse, non-scalar?
        not_ready = true;
        while(not_ready) {
            not_ready = false;

            for (auto &it : picked_already) {
                if (all_connections_vec[it].size() == 1) {
                    auto id1 = all_connections_vec[it][0][1];
                    auto id2 = all_connections_vec[it][0][2];
                    if (picked_already.count(id1) == 0) {
                        not_ready = true;
                        picked_already.insert(id1);
                        if (all_connections_vec[id1].size() > 1) {
                            need_decision.insert(id1);
                        }
                    }
                    if (picked_already.count(id2) == 0) {
                        not_ready = true;
                        picked_already.insert(id2);
                        if (all_connections_vec[id2].size() > 1) {
                            need_decision.insert(id2);
                        }
                    }
                } 
            }

            for (auto it = need_decision.begin(); it != need_decision.end();) {
                auto &b = tensor_pool[*it];
                bool erase = false;
                std::vector<alpha_decomp_connection> conns_bothreuse;
                std::vector<alpha_decomp_connection> left_reuse, right_reuse, non_reuse;
                for (auto &conn : all_connections_vec[*it]) {
                    const auto &id1 = conn[1];
                    const auto &id2 = conn[2];
                    // may have multiple ones
                    // if have multile one, 
                    if (picked_already.count(id1) && picked_already.count(id2)) {
                        conns_bothreuse.push_back(conn);
                    } else if (picked_already.count(id1)) {
                        left_reuse.push_back(conn);
                    } else if (picked_already.count(id2)) {
                        right_reuse.push_back(conn);
                    } else {
                        non_reuse.push_back(conn);
                    }
                }
                // check if erase, we need rules about reuse here
                /**
                 * 1. Both reuse
                 * 2. One reuse
                 * 3. None reuse
                 * Rule1. If have one both reuse, use this one
                 * Rule2. If multiple both reuse,
                 * 
                 * @todo: mathematically best one?
                */
                if (conns_bothreuse.size() == 1) {
                    not_ready = true;
                    erase = true;
                    picked_already.insert(*it);
                    all_connections_vec[*it].clear();
                    all_connections_vec[*it].push_back(conns_bothreuse[0]);
                    all_connections[*it].clear();
                    all_connections[*it].insert(conns_bothreuse[0]);
                } 
                else if (conns_bothreuse.size() > 1) {
                    // erase non-reuse ones
                    // not_ready = true;
                    all_connections_vec[*it].clear();
                    all_connections_vec[*it] = conns_bothreuse;
                    all_connections[*it].clear();
                    for (auto &cc : all_connections_vec[*it]) {
                        all_connections[*it].insert(cc);
                    }
                } 
                // @todo: if only one left?
                else if ((right_reuse.size() > 0 || left_reuse.size() > 0) && apply_reuse_rule_second) {
                    // erase non-reuse ones
                    // not_ready = true;
                    all_connections_vec[*it].clear();
                    all_connections[*it].clear();
                    all_connections_vec[*it].insert(all_connections_vec[*it].end(), right_reuse.begin(), right_reuse.end());
                    all_connections_vec[*it].insert(all_connections_vec[*it].end(), left_reuse.begin(), left_reuse.end());
                    for (auto &cc : all_connections_vec[*it]) {
                        all_connections[*it].insert(cc);
                    }
                }
                else 
                {
                    ;
                }
                // erase if needed
                if (erase) {
                    it = need_decision.erase(it);
                    // _DEBUG_OUTPUT_("ERASE");
                } else {
                    ++it;
                }
            }
        }

    }

    // create tree collection for each scalar 
    collections.resize(scalar_ids_vec.size());
    for (int i = 0; i < collections.size(); ++i) {
        collections[i].connections.resize(1);
        collections[i].rootid = scalar_ids_vec[i];
        create_collection(scalar_ids_vec[i], collections[i], 1);
    }
    // refine collection
    bool needrefine = true;
    while (needrefine) {
        needrefine = false;
        for (int i = 0; i < collections.size(); ++i) {
            for (int j = 0; j < collections[i].connections.size(); ++j) {
                // loop over tree, find if a tree is not ended with moment tensor
                auto cs = collections[i].connections[j];
                // std::set<int> processed;
                std::set<int> correct;
                std::set<int> root;
                std::set<int> needcorrect;
                for (auto &c : cs) {
                    int tid = c[0];
                    // int lid = c[1];
                    // int rid = c[2];
                    root.insert(tid);
                }
                for (auto &c : cs) {
                    int lid = c[1];
                    int rid = c[2];
                    if (root.count(lid) == 0 && mt_ids.count(lid) == 0) {
                        needrefine = true;
                        if (all_connections_vec[lid].size() == 1) {
                            collections[i].connections[j].insert(all_connections_vec[lid][0]);
                        } else {
                            SingleTreeCollection newcollection;
                            newcollection.connections.resize(1);
                            newcollection.connections[0] = collections[i].connections[j];
                            create_collection(lid, newcollection, 1);
                            int nadd = newcollection.connections.size();
                            collections[i].connections[j] = newcollection.connections[0];
                            for (int ic = 1; ic < nadd; ++ic) {
                                collections[i].connections.push_back(newcollection.connections[ic]);
                            }
                        }
                    }
                    if (root.count(rid) == 0 && mt_ids.count(rid) == 0) {
                        needrefine = true;
                        if (all_connections_vec[rid].size() == 1) {
                            collections[i].connections[j].insert(all_connections_vec[rid][0]);
                        } else {
                            SingleTreeCollection newcollection;
                            newcollection.connections.resize(1);
                            newcollection.connections[0] = collections[i].connections[j];
                            create_collection(rid, newcollection, 1);
                            int nadd = newcollection.connections.size();
                            collections[i].connections[j] = newcollection.connections[0];
                            for (int ic = 1; ic < nadd; ++ic) {
                                collections[i].connections.push_back(newcollection.connections[ic]);
                            }
                        }
                    }
                }
            }
        }
    }
}

// create collection of trees for a single scalar
// branch start from 1
// Each branch keeps one decomposition per tensor; build_pot_from_tree checks
// for conflicting decompositions when combining branches.
void PickTensors::create_collection(int tensorid, SingleTreeCollection &collection, int branch) {
    if (mt_ids.count(tensorid)) {
        return;
    }
    // check if tensor has been added or not, make sure only one tensor decomp routine exist
    auto &cc_this = collection.connections[branch-1];
    for (auto &t : cc_this) {
        if (t[0] == tensorid) return;
    }
    auto &conns = all_connections_vec[tensorid];
    // case1: only one solution
    if (conns.size() == 1) {
        collection.connections[branch-1].insert(conns[0]);
        create_collection(conns[0][1], collection, branch);
        create_collection(conns[0][2], collection, branch);
    }
    // case2: multiple ones
    else if (conns.size() > 1) {
        int brold = collection.connections.size();
        collection.connections.resize(collection.connections.size() + conns.size() - 1);
        // create new branchs
        for (int i = 0; i < conns.size() - 1; ++i) {
            collection.connections[brold + i] = collection.connections[branch - 1];
        }
        // first add on current branch
        collection.connections[branch-1].insert(conns[0]);
        create_collection(conns[0][1], collection, branch);
        create_collection(conns[0][2], collection, branch);
        for (int i = 1; i < conns.size(); ++i) {
            auto &conn = conns[i];
            collection.connections[brold + i - 1].insert(conn);
            create_collection(conn[1], collection, brold + i);
            create_collection(conn[2], collection, brold + i);
        }
    } 
    else {
        GPERROR("No decomposition connections available for a non-MT tensor in mtmbasis");
    }
}

void PickTensors::GAsearch() {
    if (mode < 0 || mode > 3) GPERROR("mtmbasis mode must be between 0 and 3");
    if (!std::isfinite(w1) || !std::isfinite(w2) || w1 < 0.0 || w2 < 0.0 ||
        (w1 == 0.0 && w2 == 0.0)) {
        GPERROR("mtmbasis search weights must be finite, non-negative, and not both zero");
    }
    if (maxiter < 0) GPERROR("mtmbasis maxiter must be non-negative");
    if(ga_seed < -1) GPERROR("mtmbasis seed must be -1 or nonnegative");
    if(elite_count<0 || elite_count>popsize) GPERROR("mtmbasis elite_count must lie between zero and popsize");
    if(initial_population!="legacy" && initial_population!="diverse") GPERROR("Unknown mtmbasis initial_population");
    if(restart && (elite_count>0 || initial_population!="legacy"))
        GPERROR("GA elite/diverse policies require a fresh run; restart format is unchanged");
    if (mode == 0 && T <= 0.0) GPERROR("mtmbasis T_init must be positive");
    if (mode == 2) {
        if (popsize <= 0 || childsize <= 0 || max_mutation_sites <= 0) {
            GPERROR("mtmbasis GA population and mutation sizes must be positive");
        }
        if (frac_mutation_min < 0.0 || frac_mutation_max > 1.0 ||
            frac_mutation_min > frac_mutation_max ||
            min_exchange_fraction < 0.0 || max_exchange_fraction > 1.0 ||
            min_exchange_fraction > max_exchange_fraction) {
            GPERROR("mtmbasis GA fraction bounds must be ordered within [0, 1]");
        }
    }
    // std::set up upper limit for GA search
    int nsp = 0;
    ga_upperlim.assign(collections.size(), 0);
    double totalcases = 0;
    if (collections.empty()) {
        GPERROR("mtmbasis search has no tensor collections");
    }
    if (rstop < 0) rstop = static_cast<int>(ga_upperlim.size()) - 1;
    if (rstart < 0 || rstop < rstart ||
        rstop >= static_cast<int>(ga_upperlim.size())) {
        GPERROR("mtmbasis rstart/rstop must select a valid inclusive gene range");
    }
    for (int i = 0; i < collections.size(); ++i) {
        ga_upperlim[i] = collections[i].connections.size();
        if (ga_upperlim[i] > 1 && i >= rstart && i <= rstop) {
            nsp++;
            totalcases += log10(ga_upperlim[i]);
        }
    }

    const uint32_t actual_seed=ga_seed<0 ? std::random_device{}() : static_cast<uint32_t>(ga_seed);
    std::mt19937 mtd(actual_seed);
    if(ga_seed>=0) mtd=mtmbasis_detail::ga_random_stream(actual_seed,mpi.rank);
    Message(fmt::format("MTP basis random seed={}, rank={}, deterministic={}",actual_seed,mpi.rank,ga_seed>=0));
    // 1. pick a initial guess, build initial pop
    std::vector<int> initial_guess(ga_upperlim.size(), 0);
    // now ga_upperlim is id itself. 0 means 1 solution
    for (int i = 0; i < ga_upperlim.size(); ++i) {
        if (ga_upperlim[i] == 1) {
            ga_upperlim[i] = -1;
            initial_guess[i] = -1;
        }
        else {
            --ga_upperlim[i];
            initial_guess[i] = 0;
        }
    }
    // random initial guess
    if (rndinit)
    {
        for (int i = 0; i < ga_upperlim.size(); ++i) {
            if (ga_upperlim[i] == -1) {
                initial_guess[i] = -1;
            } else {
                auto rng = std::uniform_int_distribution<int>(0, ga_upperlim[i]);
                initial_guess[i] = rng(mtd);
            }
        }
    }
    // reasonalbe init
    /**
     * @brief 
     * 1. dim min
     * 2. equal spacing
     * 3. try reuse
     */
    else {
        std::map<int, int> idkmin;
        // loop to get kmin
        for (int i = 0; i < collections.size(); ++i) {
            int ncol = collections[i].connections.size();
            int maxdepth = tensor_pool[collections[i].rootid].size();
            int rootid = collections[i].rootid;
            double kmin = 1e99;
            std::vector<int> kminids;
            for (int j = 0; j < ncol; ++j) {
                for (auto &c : collections[i].connections[j]) {
                    int mid = c[0];
                    int lid = c[1];
                    int rid = c[2];
                    int dimsum = tensor_pool[lid].diag.sum();
                    if (idkmin.count(mid)) {
                        if (dimsum < idkmin[mid]) {
                            idkmin[mid] = dimsum;
                        }
                    } else {
                        idkmin[mid] = dimsum;
                    }
                }
            }
        }
        // give a guess
        std::set<int> selected_ones;
        for (int i = 0; i < collections.size(); ++i) {
            int ncol = collections[i].connections.size();
            int maxdepth = tensor_pool[collections[i].rootid].size();
            int rootid = collections[i].rootid;
            int iselect = 0;
            double fitness_c_max = 0; // +1 if equal spacing, +1 if one reuse
            for (int j = 0; j < ncol; ++j) {
                bool choose_this = true;
                double fit = 0;
                for (auto &c : collections[i].connections[j]) {
                    int mid = c[0];
                    int lid = c[1];
                    int rid = c[2];
                    int dimsum = tensor_pool[lid].diag.sum();
                    if (dimsum != idkmin[mid]) {
                        choose_this = false;
                        break;
                    }
                    if (tensor_pool[lid].size() == tensor_pool[rid].size()) fit += 1;
                    if (selected_ones.count(lid)) fit += 1;
                    if (selected_ones.count(rid)) fit += 1;
                }
                if (!choose_this) continue;
                if (fit > fitness_c_max) {
                    fitness_c_max = fit;
                    iselect = j;
                }
                // if (choose_this) {
                //     initial_guess[i] = j;
                //     break;
                // }
            }
            initial_guess[i] = iselect;
            for (auto &c : collections[i].connections[iselect]) {
                int mid = c[0];
                int lid = c[1];
                int rid = c[2];
                selected_ones.insert(mid);
                selected_ones.insert(lid);
                selected_ones.insert(rid);
            }
        }
    }

    if (restart) {
        std::ifstream ifs("initial_guess");
        if (!ifs.is_open()) {
            GPERROR((std::string) "mtmbasis restart requested but 'initial_guess' file does not exist");
        }
        for (int i = 0; i < ga_upperlim.size(); ++i) {
            if (!(ifs >> initial_guess[i])) {
                GPERROR((std::string) "mtmbasis restart file 'initial_guess' is truncated (expected " +
                    std::to_string(ga_upperlim.size()) + " entries)");
            }
        }
        // now check if exist last collection info and rebuild new initial_guess, todo
        // string poolfile = "initial_tensorpool.dat";
        // string collfile = "initial_collection.dat";
        // ifstream ifsp(poolfile);
        // ifstream ifsc(collfile);
        // if (ifsp.is_open() && ifsc.is_open()) {
        //     ;
        // }
    }
    int pop_size = 10;
    int candisize = 100;

    double cross = 0.9;
    double mutation = 0.1;
    int itermax = 1000000;

    std::vector<std::pair<std::vector<int>, std::pair<int, int>>> pops;
    std::vector<std::pair<std::vector<int>, std::pair<int, int>>> candidates;
    std::pair<int, int> fitness_best; // moments_count, times_count

    {
        fitness_best = build_pot_from_tree(initial_guess, collections);
    }
    std::vector<int> target_pos;
    for (int i = 0; i < ga_upperlim.size(); ++i) {
        if (i < rstart) continue;
        if (i > rstop) continue;
        if (ga_upperlim[i] > 0) target_pos.push_back(i);
    }
    static constexpr std::array<const char *, 4> mode_names = {
        "tempering", "random/local", "genetic", "coordinate descent"};
    if (mode == 3) {
        Message(fmt::format(
            "MTP basis search started: mode={} ({}), variable branches={}, "
            "log10(search space)={:.2f}",
            mode, mode_names[mode], nsp, totalcases));
    } else {
        Message(fmt::format(
            "MTP basis search started: mode={} ({}), variable branches={}, "
            "iterations={}, log10(search space)={:.2f}",
            mode, mode_names[mode], nsp, maxiter, totalcases));
    }
    auto best_score = [&]() {
        return w1 * static_cast<double>(alpha_moments_count_min) +
               w2 * static_cast<double>(alpha_index_times_count_min);
    };
    auto report_iteration_progress = [&](int completed) {
        if (!mtmbasis_detail::progress_report_due(completed, maxiter)) return;
        const int percent = static_cast<int>(
            static_cast<long long>(completed) * 100 / maxiter);
        Message(fmt::format(
            "MTP basis search progress: {}% ({}/{}), best moments={}, "
            "rules={}, score={:.6g}",
            percent, completed, maxiter, alpha_moments_count_min,
            alpha_index_times_count_min, best_score()));
    };
    // if only one solution, no need to search
    if (target_pos.size() == 0) {
        Message(fmt::format(
            "MTP basis search finished: no variable branches; moments={}, "
            "rules={}, score={:.6g}",
            alpha_moments_count_min, alpha_index_times_count_min,
            best_score()));
        return;
    }
    auto pickpos = std::uniform_int_distribution<int>(0, target_pos.size()-1);
    auto lambda_pick_1 = [&](const std::pair<int, int> &current) {
        return mtmbasis_detail::weighted_fitness_better(
            current, fitness_best, w1, w2);
    };
    auto fitnessfun = [&](double fval, double fbestval) {
        return 1.0 - 1.0 / (1.0 + exp(-k_fit * (fval - fbestval) + c_fit));
    };

    // temper search
    if (mode == 0)
    {
        double Tinit = T;
        int iter = 0;
        double delta_T = Tinit / (maxiter + 1);
        auto fitness_last = fitness_best;
        // double w1 = 1.0;
        // double w2 = 6.0;
        auto dicegen = std::uniform_real_distribution<double>(0, 1);
        for (int iter = 0; iter < maxiter; ++iter) {
            double k = 1.0 / (Tinit - delta_T * iter);
            bool accept = false;
            int pick = target_pos[pickpos(mtd)];
            int so = std::uniform_int_distribution<int>(0, ga_upperlim[pick])(mtd);
            auto bck = initial_guess[pick];
            initial_guess[pick] = so;
            std::pair<int, int> fitness_current;
            fitness_current = build_pot_from_tree(initial_guess, collections);
            double delta = (fitness_current.first - fitness_last.first) * w1 + (fitness_current.second - fitness_last.second) * w2;
            if (delta < 0) accept = true;
            else if (delta < 0.5) { // if unchanged, accept 50%
                double dice = dicegen(mtd);
                if (dice < uacc_ratio) accept = true;
            }
            else {
                double frac = exp(-k * delta);
                double dice = dicegen(mtd);
                if (frac > dice) {
                    accept = true;
                }
            }
            if (accept) {
                fitness_last = fitness_current;
            } else {
                initial_guess[pick] = bck;
            }
            report_iteration_progress(iter + 1);
        }
    }
    // random search
    else if (mode == 1) {
        std::vector<int> s(ga_upperlim.size(), 0);
        iota(s.begin(), s.end(), 0);
        for (int iter = 0; iter < maxiter; ++iter) {
            // choose a guess randomly
            for (int i = 0; i < ga_upperlim.size(); ++i) {
                if (ga_upperlim[i] == -1) {
                    initial_guess[i] = -1;
                } else {
                    auto rng = std::uniform_int_distribution<int>(0, ga_upperlim[i]);
                    initial_guess[i] = rng(mtd);
                }
            }
            // optimize it
            shuffle(s.begin(), s.end(), mtd);
            for (int k = 0; k < 2; ++k)
            for (auto picked : s) {
                // for an initial guess, pick a random position, find its best, then another
                int best = initial_guess[picked];
                for (int i = 0; i <= ga_upperlim[picked]; ++i) {
                    initial_guess[picked] = i;
                    auto fitness = build_pot_from_tree(initial_guess, collections);
                    if (lambda_pick_1(fitness)) {
                        best = i;
                        fitness_best = fitness;
                    }
                }
                initial_guess[picked] = best;
            }
            report_iteration_progress(iter + 1);
        }
    }
    // GA search; elite retention and diverse initialization are explicit opt-ins.
    else if (mode == 2) {
        // for heriate
        std::vector<int> her_array(ga_upperlim.size());
        iota(her_array.begin(), her_array.end(), 0);

        std::vector<std::vector<int>> populate(popsize);
        std::vector<std::vector<int>> childs(childsize);
        for (int i = 0; i < populate.size(); ++i) {
            populate[i].resize(initial_guess.size());
        }
        for (int i = 0; i < childs.size(); ++i) {
            childs[i].resize(initial_guess.size());
        }
        populate[0] = initial_guess;
        std::vector<double> fitness_child(childsize), select_prob(childsize+1), ga_dice(childsize);
        std::vector<double> fitness_pops(popsize);
        std::vector<mtmbasis_detail::GAIndividual> initial_scored;
        if(initial_population=="diverse") {
            auto initial=mtmbasis_detail::diverse_ga_population(
                mtmbasis_detail::GAIndividual{initial_guess,fitness_best},ga_upperlim,target_pos,popsize,max_mutation_sites,mtd,
                [&](std::vector<int>& genes){return build_pot_from_tree(genes,collections);});
            initial_scored=std::move(initial.members);
            for(int i=0;i<popsize;++i)populate[i]=initial_scored[i].genes;
            Message(fmt::format("GA diverse initialization: unique={}, requested={}, evaluated={}",initial.unique_count,popsize,initial.evaluations));
        } else {
            for (int i = 1; i < populate.size(); ++i) {
                populate[i] = initial_guess;
                int pick = target_pos[pickpos(mtd)];
                int so = std::uniform_int_distribution<int>(0, ga_upperlim[pick])(mtd);
                populate[i][pick] = so;
            }
            if(elite_count>0) {
                initial_scored.push_back({initial_guess,fitness_best});
                for(int i=1;i<popsize;++i) {
                    const auto fitness=build_pot_from_tree(populate[i],collections);
                    initial_scored.push_back({populate[i],fitness});
                }
            }
        }
        std::vector<mtmbasis_detail::GAIndividual> elites;
        if(elite_count>0) mtmbasis_detail::update_ga_elites(elites,initial_scored,elite_count,w1,w2);
        // now move on search
        auto frac_mutation_gen = std::uniform_real_distribution<double>(frac_mutation_min, frac_mutation_max);
        auto mutation_nsites_gen = std::uniform_int_distribution<int>(1, max_mutation_sites);
        auto mutation_select = std::uniform_int_distribution<int>(0, popsize - 1);
        auto dice_gen = std::uniform_real_distribution<double>(0, 1);
        int exchange_min = min_exchange_fraction * her_array.size();
        int exchange_max = max_exchange_fraction * her_array.size();
        auto ncross_gen = std::uniform_int_distribution<int>(exchange_min, exchange_max);
        double fbestval = fitness_best.first * w1 + fitness_best.second * w2;
        for(const auto &individual:initial_scored)fbestval=std::min(fbestval,w1*individual.fitness.first+w2*individual.fitness.second);
        for (int iter = 0; iter < maxiter; ++iter) {
            // 1. Generate childs
            double frac_mutation = frac_mutation_gen(mtd);
            int nmutation = floor(frac_mutation * childsize);
            int nheriate = childsize - nmutation;
            // mutation
            int pt = 0;
            for (int nmu = 0; nmu < nmutation; ++nmu) {
                int nsites = mutation_nsites_gen(mtd);
                int iselect = mutation_select(mtd);
                childs[pt] = populate[iselect];
                for (int ll = 0; ll < nsites; ++ll) {
                    int pickedpos = target_pos[pickpos(mtd)];
                    childs[pt][pickedpos] = std::uniform_int_distribution<int>(0, ga_upperlim[pickedpos])(mtd);
                }
                ++pt;
            }
            // cross-over
            for (int ncro = 0; ncro < nheriate; ++ncro) {
                int parent1 = mutation_select(mtd);
                int parent2 = mutation_select(mtd);
                childs[pt] = populate[parent1];
                // do cross-over
                shuffle(her_array.begin(), her_array.end(), mtd);
                int ncross = ncross_gen(mtd);
                for (int ii = 0; ii < ncross; ++ii) {
                    childs[pt][her_array[ii]] = populate[parent2][her_array[ii]];
                }
                ++pt;
            }
            // 2. Calcualte fitness
            double fbest_candicate = fbestval;
            std::vector<mtmbasis_detail::GAIndividual> child_scored;
            for (int i = 0; i < childsize; ++i) {
                auto fmtr = build_pot_from_tree(childs[i], collections);
                if(elite_count>0) child_scored.push_back({childs[i],fmtr});
                double fval = w1 * fmtr.first + w2 * fmtr.second;
                if (fval < fbest_candicate) {
                    fbest_candicate = fval;
                }
                double fitness_val = fitnessfun(fval, fbestval);
                fitness_child[i] = fitness_val;
                select_prob[i+1] = fitness_val;
            }
            fbestval = fbest_candicate;
            // 3. select population
            select_prob = mtmbasis_detail::normalized_selection_cdf(fitness_child);
            if(elite_count>0) mtmbasis_detail::update_ga_elites(elites,child_scored,elite_count,w1,w2);
            for (int i = 0; i < popsize; ++i) {
                if(i<int(elites.size())) {populate[i]=elites[i].genes;continue;}
                double diceval = dice_gen(mtd);
                auto selected = std::upper_bound(
                    select_prob.begin() + 1, select_prob.end(), diceval);
                int child = static_cast<int>(
                    std::distance(select_prob.begin() + 1, selected));
                if (child >= childsize) child = childsize - 1;
                populate[i] = childs[child];
            }
            // next iteration
            report_iteration_progress(iter + 1);
        }
    }
    // one way opt
    else if (mode == 3) {
        std::vector<int> s(ga_upperlim.size());
        iota(s.begin(), s.end(), 0);
        if (rndinit)
            shuffle(s.begin(), s.end(), mtd);
        bool improved;
        int sweep = 0;
        do {
            improved = false;
            ++sweep;
            for (int ii = rstart; ii <= rstop; ++ii) {
                int i = s[ii];
                if (ga_upperlim[i] < 0) continue;
                reassign = true;
                auto best_fitness = fitness_best;
                auto bestguess = initial_guess;
                for (int j = 0; j <= ga_upperlim[i]; ++j) {
                    if (j == initial_guess[i]) continue;
                    auto guess = initial_guess;
                    guess[i] = j;
                    auto fitness = build_pot_from_tree(guess, collections);
                    if (mtmbasis_detail::weighted_fitness_better(
                            fitness, best_fitness, w1, w2)) {
                        best_fitness = fitness;
                        bestguess = guess;
                    }
                }
                if (mtmbasis_detail::weighted_fitness_better(
                        best_fitness, fitness_best, w1, w2)) {
                    initial_guess = std::move(bestguess);
                    fitness_best = best_fitness;
                    improved = true;
                }
            }
            if (sweep == 1 || sweep % 10 == 0 || !improved) {
                Message(fmt::format(
                    "MTP basis search progress: sweep {}, best moments={}, "
                    "rules={}, score={:.6g}{}",
                    sweep, alpha_moments_count_min,
                    alpha_index_times_count_min, best_score(),
                    improved ? "" : " (converged)"));
            }
        } while (improved);
    }

    Message(fmt::format(
        "MTP basis search finished: moments={}, rules={}, score={:.6g}",
        alpha_moments_count_min, alpha_index_times_count_min, best_score()));

}

void PickTensors::Allsearch() {
    int slen = 0;
    std::vector<int> sloc, supm;
    for (int i = 0; i < ga_upperlim.size(); ++i) {
        if (ga_upperlim[i] > 0) {
            ++slen;
            sloc.push_back(i);
            supm.push_back(ga_upperlim[i]);
        }
    }
    reverse(sloc.begin(), sloc.end());
    reverse(supm.begin(), supm.end());
    /**
     * @brief Algo
     * 
     * supm will change during optimization
     * Loop back will break stack size during search 
     * 
     */
    // how to deal this??
    
}
// NOTE: alphapool.dat should be the SAME!!!
void PickTensors::regen_from_old() {
    // 1. read old tensorpool tensor_pool_old.log
    std::ifstream ifs("tensor_pool_old.log");
    std::string buffer;
    std::vector<std::string> vstr;
    std::vector<alpha_index_type> tensor_pool_old;
    while(!ifs.eof()) {
        getline(ifs, buffer);
        // search Bid, read a block
        if (buffer.find("Bid") != std::string::npos) {
            str_split(buffer, vstr);
            int bid = stoi(vstr[1]);
            // read mus
            getline(ifs, buffer);
            str_split(buffer, vstr);
            int d = vstr.size() - 1;
            alpha_index_type B;
            B.resize(d);
            for (int i = 0; i < d; ++i) {
                B.first(i) = stoi(vstr[i+1]);
            }
            // nu
            getline(ifs, buffer);
            // level
            getline(ifs, buffer);
            // mu
            getline(ifs, buffer);
            // matrix
            for (int i = 0; i < d; ++i) {
                for (int j = 0; j < d; ++j) {
                    ifs >> B.second(i, j);
                }
            }
            ifs.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            // solve B
            B.solve();
            B.tensor_id = bid;
            tensor_pool_old.push_back(B);
        }
    }
    ifs.close();
    // 2. read collections_graphs_old.log
    std::ifstream ifs2("collection_graphs_old.log");
    std::vector<SingleTreeCollection> collections_old(alphas_gen.size());
    int ct = 0;
    while(!ifs2.eof()) {
        getline(ifs2, buffer);
        if (buffer.find("Tensor") != std::string::npos) {
            str_split(buffer, vstr);
            int tid = stoi(vstr[2]);
            // collection size
            getline(ifs2, buffer);
            str_split(buffer, vstr);
            int dsize = stoi(vstr[2]);
            collections_old[ct].rootid = tid;
            collections_old[ct].connections.resize(dsize);
            // read each collection
            std::vector<std::string> buffers;
            getline(ifs2, buffer);
            // read unitl empty line
            while (!buffer.empty()) {
                buffers.push_back(buffer);
                getline(ifs2, buffer);
            }
            _DEBUG_VECTOR_(buffers);
            // process
            if (buffers.size() == 2) {
                ;// do nothing
            } else {
                std::vector<int> pts;
                int ct2 = 0;
                pts.resize(dsize + 1, 0);
                for (int i = 0; i < buffers.size(); ++i) {
                    if (buffers[i].find("Collection") != std::string::npos) {
                        pts[ct2] = i;
                        ++ct2;
                    }
                }
                pts[dsize] = buffers.size();
                _DEBUG_VECTOR_(pts);
                _DEBUG_OUTPUT_("S3");
                for (int i = 0; i < dsize; ++i) {
                    int pt1 = pts[i];
                    int pt2 = pts[i+1];
                    _DEBUG_OUTPUT_(pt1);
                    _DEBUG_OUTPUT_(pt2);
                    for (int j = pt1+1; j < pt2; ++j) {
                        _DEBUG_OUTPUT_(j);
                        _DEBUG_OUTPUT_(buffers.size());
                        _DEBUG_OUTPUT_(buffers[j]);
                        alpha_decomp_connection tt;
                        str_split(buffers[j], vstr);
                        tt[0] = stoi(vstr[0]);
                        tt[1] = stoi(vstr[1]);
                        tt[2] = stoi(vstr[2]);
                        _DEBUG_OUTPUT_(ct);
                        _DEBUG_OUTPUT_(i);
                        collections_old[ct].connections[i].insert(tt);
                    }
                }
                _DEBUG_OUTPUT_("S5");
            }
            ct++;
        }
    }
    // 3. read initial_guess_old
    _DEBUG_OUTPUT_("S3");
    std::ifstream ifs3("initial_guess_old");
    std::vector<int> initial_guess_old(alphas_gen.size());
    for (int i = 0; i < initial_guess_old.size(); ++i) {
        ifs3 >> initial_guess_old[i];
        if (initial_guess_old[i] < 0) initial_guess_old[i] = 0;
    }
    // 4. build mapping between tensor_pool and tensor_pool_old
    _DEBUG_OUTPUT_("S4");
    std::map<int, int> mapping_old_new;
    for (int i = 0; i < tensor_pool_old.size(); ++i) {
        auto &bold = tensor_pool_old[i];
        for (int j = 0; j < tensor_pool.size(); ++j) {
            if (bold == tensor_pool[j]) {
                mapping_old_new[bold.tensor_id] = tensor_pool[j].tensor_id;
                break;
            }
        }
    }
    // 5. rebuild initial_guess
    _DEBUG_OUTPUT_("S5");
    std::vector<int> initial_guess(initial_guess_old.size(), 0);
    for (int i = 0; i < initial_guess.size(); ++i) {
        _DEBUG_OUTPUT_(i);
        if (collections[i].connections.size() == 1 && collections_old[i].connections.size() == 1) {
            continue;
        }
        _DEBUG_OUTPUT_(collections[i].connections.size());
        _DEBUG_OUTPUT_(collections_old[i].connections.size());
        _DEBUG_OUTPUT_(initial_guess_old[i]);
        auto &cold = collections_old[i].connections[initial_guess_old[i]];
        auto &csnew = collections[i].connections;
        _DEBUG_OUTPUT_("RR");
        std::set<alpha_decomp_connection> crebuild;
        for (auto &c : cold) {
            _DEBUG_OUTPUT_("1");
            alpha_decomp_connection cc;
            cc[0] = mapping_old_new[c[0]];
            _DEBUG_OUTPUT_("2");
            cc[1] = mapping_old_new[c[1]];
            _DEBUG_OUTPUT_("3");
            cc[2] = mapping_old_new[c[2]];
            _DEBUG_OUTPUT_("4");
            crebuild.insert(cc);
        }
        _DEBUG_OUTPUT_("MATCH");
        // match
        bool correct = false;
        for (int j = 0; j < csnew.size(); ++j) {
            auto &cnew = csnew[j];
            correct = true;
            for (auto &c : crebuild) {
                if (!cnew.count(c)) {
                    correct = false;
                    break;
                }
            }
            if (correct) {
                initial_guess[i] = j;
                break;
            }
        }
    }
    // 6. write output
    _DEBUG_OUTPUT_("S6");
    std::ofstream ofs("initial_guess");
    for (int i = 0; i < initial_guess.size(); ++i) {
        ofs << initial_guess[i] << " ";
    }
    ofs.close();
}

void PickTensors::clean_pool(std::vector<alpha_index_type> &pool) {
    _DEBUG_OUTPUT_("SOLVING POOL");
    #pragma omp parallel for
    for (auto &t : pool) {
        if (!t.solved)
            t.solve();
    }
    _DEBUG_OUTPUT_("REMOVE DUPS");
    for (int i = 0; i < pool.size(); ++i) {
        if (i%1000==0) _DEBUG_OUTPUT_(i);
        auto &g0 = pool[i];
        for (auto it = pool.begin() + i + 1; it != pool.end(); ) {
            if (g0 == *it) {
                it = pool.erase(it);
            } else {
                ++it;
            }
        }
    }
}

void PickTensors::save_pool(const std::string &filename, std::vector<alpha_index_type> &pool) {
    std::ofstream ofs(filename);
    if (!ofs.is_open()) {
        GPERROR((std::string) "Cannot open mtmbasis alpha pool output: " + filename);
    }
    for (int i = 0; i < pool.size(); ++i) {
        ofs << "Bid: " << i << '\n';
        auto &b = pool[i];
        // auto &wid = selected_ones[i];
        int s = b.first.size();
        
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
        ofs << "level: " << calc_level(mut, nvt, dimt, mu_wei, nv_wei, dim_wei) << " dim " << dimt << " maxpoly " << nvt << '\n';


        ofs << b.first.transpose() << '\n';
        for (int j = 0; j < s; ++j) {
            ofs << "  ";
            for (int k = 0; k < s; ++k) {
                ofs << b.second(j, k) <<  "  ";
            }
            ofs << "\n";
        }
        ofs << "\n";
    }
    ofs.close();
}

/**
 * @brief 
 * 
 * @param supm 
 * @param colls 
 * @param depth   current depth
 * @param branch  last branch
 */
void PickTensors::iter_search(std::vector<int> &supm, 
            std::vector<int> &csel,
            std::vector<SingleTreeCollection> &colls, 
            int depth, int branch) {
    // 1. check if we have reached the last one
    if (depth == supm.size() - 1) {
        // loop over branches
        for (int i = 0; i < supm[depth]; ++i) {
            csel[depth] = i;
            build_pot_from_tree(csel, colls);
        }
        return;
    }
    
    // 2. check if current depth has single branch
    if (supm[depth] == 1) {
        // find next fork
        ;
    }
    // 3. fork
    ;

}

void PickTensors::build_init() {
    need_regen_first = false;
    // first threat single collection
    bool notready = true;
    while (notready) {
        notready = false;
        for (int i = 0; i < collections.size(); ++i) {
            int tid = collections[i].rootid;
            // if (tensor_pool[tid].second.sum() == 0 && 
            //         tensor_pool[tid].size() > 1 && 
            //         collections[i].connections.size() > 1) {
            //     collections[i].connections.resize(1);
            // }
            if (collections[i].connections.size() > 1) continue;
            // non-exist
            if (collections[i].connections.size() == 0) {
                _DEBUG_OUTPUT_("WARNING: NON_EXIST CASE HAPPENED");
                tensors_must_appear_first.insert(tid);
                continue;
            }
            // scalar mt
            if (collections[i].connections[0].size() == 0) {
                tensors_must_appear_first.insert(tid);
            }
            // scalar tensors with only one decomp
            for (auto &it : collections[i].connections[0]) {
                auto existing = decided_decomp_tensors_first.find(it[0]);
                if (existing != decided_decomp_tensors_first.end()) {
                    auto conn_it = std::find_if(conns_must_use_first.begin(), conns_must_use_first.end(),
                        [&](const alpha_decomp_connection &conn) { return conn[0] == it[0]; });
                    if (conn_it != conns_must_use_first.end() && *conn_it != it) {
                        GPERROR("Conflicting mandatory decompositions in mtmbasis tree initialization");
                    }
                }
                tensors_must_appear_first.insert(it[0]);
                tensors_must_appear_first.insert(it[1]);
                tensors_must_appear_first.insert(it[2]);
                conns_must_use_first.insert(it);
                // tensors_in_use_first[it[0]] = it;
                decided_decomp_tensors_first.insert(it[0]);
            }
        }
    }
    // generate moments and times for these tensors and connections
    // _DEBUG_std::vector_(tensors_must_appear_first)
    alpha_index_times_count_first = 0;
    alpha_moments_count_first = alpha_index_basic_count;
    alpha_index_times_first.clear();
    alpha_moment_mapping_first.clear();
    alpha_times_flag_first.clear();
    bi_id_map_first.clear();
    // gen_alpha_index_times(tensors_must_appear_first, conns_must_use_first, 
    //         bi_id_map_first, 
    //         alpha_moments_count_first, 
    //         alpha_index_times_count_first, 
    //         alpha_index_times_first,
    //         alpha_times_flag_first,
    //         alpha_moment_mapping_first);
    // reduce collections, remove tensors already in use in tree
    if (chop_trees_first)
    for (auto &i : collections) {
        int csize = i.connections.size();
        for (auto &j : i.connections) {
            for (auto it = j.begin(); it != j.end(); ) {
                if (decided_decomp_tensors_first.count((*it)[0])) {
                    // _DEBUG_OUTPUT_("erse1");
                    it = j.erase(it);
                } else {
                    ++it;
                }
            }
            // remove isolated ones who can not connected to root
            // std::set<alpha_decomp_connection> flags;
            std::set<int> ccs;
            ccs.insert(i.rootid);
            for (auto &k : j) {
                if (k[0] == i.rootid) {
                    ccs.insert(k[1]);
                    ccs.insert(k[2]);
                    break;
                }
            }
            bool nf = true;
            while(nf) {
                nf = false;
                for (auto &k : j) {
                    if (ccs.count(k[0]) && (ccs.count(k[1]) == 0 || ccs.count(k[2]) == 0)) {
                        nf = true;
                        ccs.insert(k[1]);
                        ccs.insert(k[2]);
                        // flags.insert(k);
                    } 
                }
            }
            for (auto it = j.begin(); it != j.end(); ) {
                if (!ccs.count((*it)[0])) {
                    // _DEBUG_OUTPUT_("ORPHAN1");
                    // _DEBUG_std::vector_(*it)
                    it = j.erase(it);
                } else {
                    ++it;
                }
            }
        }
        // merge same branches
        sort(i.connections.begin(), i.connections.end());
        i.connections.erase(unique(i.connections.begin(), i.connections.end()), i.connections.end());
        if (i.connections.size() != csize) {
            need_regen_first = true;
        }
    }

    // at this time, we may encounter cases that a tensor may present in different decomp ways
    // now we need k_id instead of t_id
    /** CODE HERE **/
    // huge_pool = tensor_pool;
    // for (int i = 0; i < huge_pool.size(); ++i) {
    //     huge_pool[i].index_k_id = tensor_pool[i].tensor_id;
    // }
    // hashmaparr_k = hashmaparr;
    // convert collections from tid to kid
    /** CODE HERE **/
}

// element in select will be -1 if they have been already processed
std::pair<int, int> PickTensors::build_pot_from_tree(std::vector<int> &select, 
        std::vector<SingleTreeCollection> &collections_this) {
    // 1. pick one by one, and reduce trees step by step
    // _DEBUG_OUTPUT_(select.size());
    // Per-candidate tree state as flat arrays instead of std::sets: the
    // search loop rebuilds this state for every candidate, and the set-based
    // version cost ~10k heap allocations per call.
    std::vector<char> decided_flags(tensor_pool.size(), 0);
    std::vector<char> tensor_used_flags(tensor_pool.size(), 0);
    std::vector<char> root_seen(tensor_pool.size(), 0);
    std::vector<alpha_decomp_connection> root_conn(tensor_pool.size());
    std::vector<alpha_decomp_connection> conns_in_use_curr;
    std::vector<int> tensors_in_use;
    auto add_conn = [&](const alpha_decomp_connection &conn) {
        if (root_seen[conn[0]]) {
            if (root_conn[conn[0]] != conn) {
                GPERROR("Conflicting decompositions selected for one tensor in mtmbasis");
            }
            return;
        }
        root_seen[conn[0]] = 1;
        root_conn[conn[0]] = conn;
        conns_in_use_curr.push_back(conn);
    };
    for (const auto &conn : conns_must_use_first) add_conn(conn);
    for (int t : tensors_must_appear_first) {
        if (!tensor_used_flags[t]) {
            tensor_used_flags[t] = 1;
            tensors_in_use.push_back(t);
        }
    }
    for (int t : decided_decomp_tensors_first) decided_flags[t] = 1;

    std::map<int, alpha_decomp_connection> reassign_map;

    // Branch overlay: this function must not mutate the shared `collections`
    // (callers used to deep-copy the whole collection graph per evaluation,
    // which dominated the search loop cost).  Only the chosen branch of each
    // collection is ever rewritten, so keep those rewrites in a small
    // per-call map keyed by (collection id, branch id) and read through it.
    // The value is exactly what the former in-place write would have left.
    std::map<std::pair<int, int>, std::set<alpha_decomp_connection>> branch_overlay;
    auto branch_content = [&](int collection_id,
                              int branch_id) -> const std::set<alpha_decomp_connection> & {
        auto it = branch_overlay.find({collection_id, branch_id});
        if (it != branch_overlay.end()) {
            return it->second;
        }
        return collections_this[collection_id].connections[branch_id];
    };

    // Scratch buffers reused across all prunes of one call: prune is the
    // per-iteration hot loop (~n_collections prunes per candidate), so avoid
    // per-prune allocations.  prune_visited uses a generation counter instead
    // of resetting, so early returns cannot leave stale state.
    std::vector<int> prune_visited(tensor_pool.size(), 0);
    int prune_gen = 0;
    std::vector<int> prune_stack;
    std::vector<int> prune_roots;
    std::vector<alpha_decomp_connection> prune_conns;

    auto prune_candidate_tree =
        [&](const std::set<alpha_decomp_connection> &source,
            int rootid,
            std::set<alpha_decomp_connection> &out) -> bool {
        out.clear();
        if (mt_ids.count(rootid) || decided_flags[rootid]) {
            return true;
        }
        // source is sorted by (root, left, right), so same-root connections
        // are contiguous: scan once and reject a root with two distinct conns.
        ++prune_gen;
        prune_roots.clear();
        prune_conns.clear();
        for (const auto &conn : source) {
            if (decided_flags[conn[0]]) {
                continue;
            }
            if (!prune_roots.empty() && prune_roots.back() == conn[0]) {
                if (prune_conns.back() != conn) {
                    return false;
                }
            } else {
                prune_roots.push_back(conn[0]);
                prune_conns.push_back(conn);
            }
        }
        if (prune_roots.empty()) {
            return false;
        }
        // BFS reachability from rootid over the root->(left,right) edges.
        prune_stack.clear();
        prune_stack.push_back(rootid);
        while (!prune_stack.empty()) {
            const int tid = prune_stack.back();
            prune_stack.pop_back();
            if (prune_visited[tid] == prune_gen) {
                continue;
            }
            prune_visited[tid] = prune_gen;
            if (mt_ids.count(tid) || decided_flags[tid]) {
                continue;
            }
            int idx = -1;
            for (int i = 0; i < (int)prune_roots.size(); ++i) {
                if (prune_roots[i] == tid) {
                    idx = i;
                    break;
                }
            }
            if (idx < 0) {
                return false;
            }
            prune_stack.push_back(prune_conns[idx][1]);
            prune_stack.push_back(prune_conns[idx][2]);
        }
        for (int i = 0; i < (int)prune_roots.size(); ++i) {
            if (prune_visited[prune_roots[i]] == prune_gen) {
                out.insert(prune_conns[i]);
            }
        }
        return true;
    };

    auto choose_unique_branch =
        [&](int collection_id, int preferred, std::set<alpha_decomp_connection> &out) -> int {
        const auto &branches = collections_this[collection_id].connections;
        if (branches.empty()) {
            return -1;
        }
        int rootid = collections_this[collection_id].rootid;
        if (preferred < 0 || preferred >= (int)branches.size()) {
            preferred = 0;
        }
        if (prune_candidate_tree(branch_content(collection_id, preferred),
                                 rootid, out)) {
            return preferred;
        }
        for (int branch_id = 0; branch_id < (int)branches.size(); ++branch_id) {
            if (branch_id == preferred) {
                continue;
            }
            if (prune_candidate_tree(branch_content(collection_id, branch_id),
                                     rootid, out)) {
                return branch_id;
            }
        }
        return -1;
    };
    
    for (int i = 0; i < select.size(); ++i) {
        // _DEBUG_OUTPUT_(collections_this[i].rootid);
        int ibranch = select[i];
        if (ibranch < 0) continue;
        // if ibranch larger than collection size, reduced it. This should not happened.
        if (collections_this[i].connections.empty()) {
            select[i] = -1;
            continue;
        }
        if (collections_this[i].connections.size() <= ibranch) {
            ibranch = collections_this[i].connections.size() - 1;
            // TODO: use this or not?
            select[i] = ibranch;
        }
        std::set<alpha_decomp_connection> conns_in_use;
        ibranch = choose_unique_branch(i, ibranch, conns_in_use);
        if (ibranch < 0) {
            GPERROR("Unable to select a unique decomposition tree in mtmbasis");
        }
        select[i] = ibranch;
        if (conns_in_use.empty()) {
            select[i] = -1;
            continue;
        }
        branch_overlay[{i, ibranch}] = conns_in_use;
        // std::set<int> tensors_in_use;
        for (auto &it : conns_in_use) {
            add_conn(it);
            decided_flags[it[0]] = 1;
            for (int t : {it[0], it[1], it[2]}) {
                if (!tensor_used_flags[t]) {
                    tensor_used_flags[t] = 1;
                    tensors_in_use.push_back(t);
                }
            }
            if (reassign) {
                reassign_map[it[0]] = it;
            }
        }
        // _DEBUG_OUTPUT_("GEN1");
        // dump_all();
        // gen_alpha_index_times(tensors_in_use, conns_in_use, bi_id_map, 
        //             alpha_moments_count, 
        //             alpha_index_times_count, 
        //             alpha_index_times,
        //             alpha_times_flag,
        //             alpha_moment_mapping);
        // _DEBUG_OUTPUT_("REDUCE");
        /**
         * @brief reduce the tree
         * TODO: adjust select array
         */
        // reduce tree and adjust select array, only need to chop jselect
        if (chop_trees && i >= chop_start)
        for (int j = i+1; j < std::min((int)select.size(), i+2); ++j) {
            int rootj = collections_this[j].rootid;
            int jselect = select[j];
            if (jselect < 0) jselect = 0;
            if (collections_this[j].connections.empty()) {
                select[j] = -1;
                continue;
            }
            // overlay write instead of in-place mutation of the shared tree.
            // prune is monotone and idempotent, so when it succeeds here
            // collection j re-derives the identical pruned tree from the
            // original branch during its own choose; only the failure
            // (invalidation) case needs to be recorded to preserve the
            // fallback-branch behavior.
            if (jselect >= (int)collections_this[j].connections.size()) {
                continue;
            }
            std::set<alpha_decomp_connection> pruned_tree;
            if (!prune_candidate_tree(
                    branch_content(j, jselect), rootj, pruned_tree)) {
                branch_overlay[{j, jselect}].clear();
            }
        }
        // this will loop entire collections
        // for (int j = i+1; j < min((int)select.size(), i+2); ++j) {
        //     int rootj = collections_this[j].rootid;

        //     for (auto &connections : collections_this[j].connections) {
        //         for (auto it = connections.begin(); it != connections.end(); ) {
        //             if (decided_decomp_tensors.count((*it)[0])) {
        //                 it = connections.erase(it);
        //             } else {
        //                 ++it;
        //             }
        //         }
        //         // remove isolated ones who can not connected to root
        //         // todo: do this in previous step
        //         // std::set<alpha_decomp_connection> flags;
        //         std::set<int> ccs;
        //         ccs.insert(rootj);
        //         for (auto &k : connections) {
        //             if (k[0] == rootj) {
        //                 ccs.insert(k[1]);
        //                 ccs.insert(k[2]);
        //                 break;
        //             }
        //         }
        //         bool nf = true;
        //         while(nf) {
        //             nf = false;
        //             for (auto &k : connections) {
        //                 if (ccs.count(k[0]) && (ccs.count(k[1]) == 0 || ccs.count(k[2]) == 0)) {
        //                     nf = true;
        //                     ccs.insert(k[1]);
        //                     ccs.insert(k[2]);
        //                     // flags.insert(k);
        //                 } 
        //             }
        //         }
        //         for (auto it = connections.begin(); it != connections.end(); ) {
        //             if (ccs.count((*it)[0]) == 0) {
        //                 // _DEBUG_OUTPUT_("ORPHAN");
        //                 // _DEBUG_std::vector_(*it)
        //                 it = connections.erase(it);
        //             } else {
        //                 ++it;
        //             }
        //         }
        //     }
        //     // node: do not use this to keep consistency
        //     // int ccc = collections_this[j].connections.size();
        //     // stable_sort(collections_this[j].connections.begin(), collections_this[j].connections.end());
        //     // collections_this[j].connections.erase(unique(collections_this[j].connections.begin(), 
        //     //         collections_this[j].connections.end()), 
        //     //         collections_this[j].connections.end());
        //     // if (collections_this[j].connections.size() != ccc) {
        //     //     _DEBUG_OUTPUT_("GOTCLT");
        //     // }
        // }
        //
    }
    std::set<int> tensorpool_used_this;

    auto candidate_would_save = [&](int moments_count, int rules_count) {
        return mtmbasis_detail::weighted_fitness_better(
            {moments_count, rules_count},
            {alpha_moments_count_min, alpha_index_times_count_min},
            w1, w2);
    };

    // Incremental generation: connections processed in earlier evaluations
    // are skipped inside gen_alpha_index_times (their moments and rules are
    // already in the persistent state); only new connections generate.
    auto t_gen0 = std::chrono::steady_clock::now();
    int global_moments_count = next_moment_id_persist;
    int global_rules_count = static_cast<int>(alpha_index_times_persist.size());
    gen_alpha_index_times(tensors_in_use, conns_in_use_curr, bi_id_map_persist,
        global_moments_count,
        global_rules_count,
        alpha_index_times_persist,
        alpha_times_flag_persist,
        alpha_moment_mapping_persist,
        tensorpool_used_this,
        mid_tid_mapping_persist);
    next_moment_id_persist = global_moments_count;

    // Candidate counts from the per-root moment lists and per-connection rule
    // ranges.  Moments are shared across decompositions of the same tensor,
    // so the union over a root's connections equals the moments of its chosen
    // connection, and the per-connection rule ranges are exact.  This is the
    // O(roots) hot path; the full pot materialization below runs only for
    // winners.
    int alpha_moments_count = static_cast<int>(alpha_index_basic_count);
    int alpha_index_times_count = 0;
    for (const auto &conn : conns_in_use_curr) {
        auto ids = root_moment_ids.find(conn[0]);
        auto rit = conn_rule_range.find(conn);
        if (ids == root_moment_ids.end() || rit == conn_rule_range.end()) {
            GPERROR("Missing generation state for a connection in mtmbasis generation");
        }
        alpha_moments_count += static_cast<int>(ids->second.size());
        alpha_index_times_count += rit->second.second - rit->second.first;
    }

    // refresh the per-connection count bookkeeping used by conns_exp.log
    for (const auto &conn : conns_in_use_curr) {
        auto ids = root_moment_ids.find(conn[0]);
        auto rit = conn_rule_range.find(conn);
        if (ids != root_moment_ids.end()) {
            alpha_moments_count_add_count[conn] =
                static_cast<int>(ids->second.size());
        }
        if (rit != conn_rule_range.end()) {
            alpha_index_times_count_add_count[conn] =
                rit->second.second - rit->second.first;
        }
    }

    bool save = candidate_would_save(
        alpha_moments_count, alpha_index_times_count);
    // Preserve the checkpoint contract: every strict global improvement is
    // written exactly once, with cnt identifying that improvement.
    if (save) {
        alpha_moments_count_min = alpha_moments_count;
        alpha_index_times_count_min = alpha_index_times_count;
    }

    // Full pot materialization (only for winners): the candidate's moments
    // and rules, remapped to contiguous ids.
    std::vector<char> is_moment;
    std::vector<int> remap;
    std::vector<std::array<int, 4>> alpha_index_times;
    std::vector<int> alpha_moment_mapping;
    std::vector<int> alpha_moment_mapping_global;
    if (save) {
    std::vector<char> root_flag(tensor_pool.size(), 0);
    for (const auto &conn : conns_in_use_curr) {
        root_flag[conn[0]] = 1;
    }
    is_moment.assign(next_moment_id_persist, 0);
    for (int m = 0; m < static_cast<int>(alpha_index_basic_count); ++m) {
        is_moment[m] = 1;
    }
    std::vector<std::pair<int, int>> rule_ranges;
    for (const auto &conn : conns_in_use_curr) {
        auto ids = root_moment_ids.find(conn[0]);
        if (ids == root_moment_ids.end()) {
            GPERROR("Missing moment ids for a root tensor in mtmbasis generation");
        }
        for (int m : ids->second) {
            is_moment[m] = 1;
        }
        auto rit = conn_rule_range.find(conn);
        if (rit == conn_rule_range.end()) {
            GPERROR("Missing rule range for a connection in mtmbasis generation");
        }
        rule_ranges.push_back(rit->second);
    }
    std::vector<std::array<int, 4>> raw_rules;
    for (const auto &[rs, re] : rule_ranges) {
        for (int i = rs; i < re; ++i) {
            raw_rules.push_back(alpha_index_times_persist[i]);
        }
    }
    // Topological order (Kahn) over the rule dependency graph.  A shared
    // moment keeps the id assigned when it was first created, while the rule
    // of a later decomposition may reference components with higher global
    // ids, so the global id order is not topological.
    std::map<int, int> rule_of_target;
    for (int i = 0; i < static_cast<int>(raw_rules.size()); ++i) {
        rule_of_target[raw_rules[i][3]] = i;
    }
    std::vector<std::vector<int>> users(raw_rules.size());
    std::vector<int> indeg(raw_rules.size(), 0);
    for (int i = 0; i < static_cast<int>(raw_rules.size()); ++i) {
        for (int component : {raw_rules[i][0], raw_rules[i][1]}) {
            if (component >= static_cast<int>(alpha_index_basic_count)) {
                auto it = rule_of_target.find(component);
                if (it == rule_of_target.end()) {
                    GPERROR("Materialized rule references an outside moment in mtmbasis generation");
                }
                users[it->second].push_back(i);
                ++indeg[i];
            }
        }
    }
    std::vector<int> kahn;
    std::vector<int> queue;
    for (int i = 0; i < static_cast<int>(raw_rules.size()); ++i) {
        if (indeg[i] == 0) queue.push_back(i);
    }
    for (size_t q = 0; q < queue.size(); ++q) {
        const int i = queue[q];
        kahn.push_back(i);
        for (int dependent : users[i]) {
            if (--indeg[dependent] == 0) queue.push_back(dependent);
        }
    }
    if (static_cast<int>(kahn.size()) != static_cast<int>(raw_rules.size())) {
        GPERROR("Cycle in the mtmbasis rule dependency graph");
    }
    // Local ids assigned in topological emission order, so components always
    // precede their targets and the rule list is target-ordered.
    remap.assign(next_moment_id_persist, -1);
    for (int m = 0; m < static_cast<int>(alpha_index_basic_count); ++m) {
        remap[m] = m;
    }
    int local_id = static_cast<int>(alpha_index_basic_count);
    for (int i : kahn) {
        const auto &r = raw_rules[i];
        if (remap[r[0]] < 0 || remap[r[1]] < 0) {
            GPERROR("Materialized rule references an outside component in mtmbasis generation");
        }
        if (remap[r[3]] < 0) remap[r[3]] = local_id++;
        alpha_index_times.push_back(
            {remap[r[0]], remap[r[1]], r[2], remap[r[3]]});
    }
    if (local_id != alpha_moments_count) {
        GPERROR("Materialized moment count does not match rule targets in mtmbasis generation");
    }
    // A moment carries several rules (one per contraction partition), so the
    // emission order is not target-sorted; stable-sort by the local target to
    // restore the non-decreasing target order required by the topology
    // validation (equal targets keep the topological emission order).
    std::stable_sort(alpha_index_times.begin(), alpha_index_times.end(),
        [](const std::array<int, 4> &left, const std::array<int, 4> &right) {
            return left[3] < right[3];
        });
    std::vector<std::pair<int, int>> mapping_pairs;
    for (int m : alpha_moment_mapping_persist) {
        if (is_moment[m] && remap[m] >= 0) {
            mapping_pairs.push_back({remap[m], m});
        }
    }
    std::sort(mapping_pairs.begin(), mapping_pairs.end());
    mapping_pairs.erase(
        std::unique(mapping_pairs.begin(), mapping_pairs.end()),
        mapping_pairs.end());
    for (const auto &[local, global] : mapping_pairs) {
        alpha_moment_mapping.push_back(local);
        alpha_moment_mapping_global.push_back(global);
    }
    mtmbasis_detail::validate_alpha_topology(
        static_cast<int>(alpha_index_basic_count),
        alpha_moments_count,
        alpha_index_times,
        alpha_moment_mapping);
    } // end of save-gated materialization

    // refresh the per-connection count bookkeeping used by conns_exp.log
    for (const auto &conn : conns_in_use_curr) {
        auto ids = root_moment_ids.find(conn[0]);
        auto rit = conn_rule_range.find(conn);
        if (ids != root_moment_ids.end()) {
            alpha_moments_count_add_count[conn] =
                static_cast<int>(ids->second.size());
        }
        if (rit != conn_rule_range.end()) {
            alpha_index_times_count_add_count[conn] =
                rit->second.second - rit->second.first;
        }
    }

    if (save && reassign) {
        // compare collections and conns_in_use_curr && reassign_map
        for (int i = 0; i < select.size(); ++i) {
            int iselect = select[i];
            if (iselect < 0) continue;
            int iselect_true = 0;
            bool is_this = false;
            for (int j = 0; j < collections[i].connections.size(); ++j) {
                auto &connj = collections[i].connections[j];
                is_this = true;
                for (auto &c : connj) {
                    if (reassign_map[c[0]] != c) {
                        is_this = false;
                        break;
                    }
                }
                if (is_this) {
                    iselect_true = j;
                    break;
                }
            }
            if (iselect != iselect_true) {
                select[i] = iselect_true;
            }
        }
    }

    if (save) {
        save_potential(output_file + "_" + std::to_string(cnt), alpha_moments_count, alpha_index_times_count, alpha_index_times, alpha_moment_mapping);
        // output tensor in use
        // {
        //     std::string fnm = "tensorinuse_" + std::to_string(cnt) + ".log" + mpi.fnm_ending;
        //     std::ofstream ofs(fnm);
        //     for (auto &t : tensors_in_use) {
        //         ofs << t << '\n';
        //     }
        //     ofs.close();
        // }
        // output connection graphs used
        {
            std::string fnm = "collection_graphs_" + std::to_string(cnt) + ".log" + mpi.fnm_ending;
            std::ofstream ofs(fnm);
            for (int i = 0; i < select.size(); ++i) {
                int ibranch = select[i];
                if (ibranch < 0) ibranch = 0;
                if (collections_this[i].connections.size() <= ibranch) {
                    ibranch = collections_this[i].connections.size() - 1;
                }
                ofs << "Tensor id: " << scalar_ids_vec[i] << '\n';
                for (auto &it : branch_content(i, ibranch)) {
                    ofs << "  " << it[0] << " " << it[1] << " " << it[2] << '\n';
                }
                ofs << '\n';
            }
            ofs.close();
        } 
        // output tensor pool used
        {
            std::string fnm = "tensorpool_" + std::to_string(cnt) + ".log" + mpi.fnm_ending;
            std::ofstream ofs(fnm);
            for (auto &it : tensorpool_used_this) {
                ofs << "Tensor id: " << it << "\n";
                ofs << tensor_pool[it] << '\n';
            }
            ofs.close();
        }
        // output select info, for restart
        {
            std::string fnm = "select_" + std::to_string(cnt) + ".log" + mpi.fnm_ending;
            std::ofstream ofs(fnm);
            for (auto &it : select) {
                ofs << it << " ";
            }
            ofs.close();
        }
        // output mid tid mapping (winner moments only, local pot ids)
        {
            std::string fnm = "mid_tid_mapping_" + std::to_string(cnt) + ".log" + mpi.fnm_ending;
            std::ofstream ofs(fnm);
            for (auto &it : mid_tid_mapping_persist) {
                if (is_moment[it.first]) {
                    ofs << remap[it.first] << " " << it.second << '\n';
                }
            }
            ofs.close();
        }
        // output scalar id - tensor id mapping
        {
            std::string fnm = "scalar_id_tensor_id_mapping_" + std::to_string(cnt) + ".log" + mpi.fnm_ending;
            std::ofstream ofs(fnm);
            for (int i = 0; i < alpha_moment_mapping_global.size(); ++i) {
                int tid = mid_tid_mapping_persist[alpha_moment_mapping_global[i]];
                ofs << i << " " << tid << '\n';
            }
            ofs.close();
        }
        // output scalar pool
        {
            std::string fnm = "scalar_pool_" + std::to_string(cnt) + ".dat" + mpi.fnm_ending;
            std::ofstream ofs(fnm);
            for (int m : alpha_moment_mapping_global) {
                int tid = mid_tid_mapping_persist[m];
                ofs << tensor_pool[tid];
            }
            ofs.close();
        }
        // connections
        {
            std::string fnm = "conns_use_curr_" + std::to_string(cnt) + ".dat" + mpi.fnm_ending;
            std::ofstream ofs(fnm);
            for (auto &i : conns_in_use_curr) {
                ofs << i[0] << " " << i[1] << " " << i[2] << '\n';
            }
            ofs.close();
        }
        ++cnt;
    }

    return std::make_pair(alpha_moments_count, alpha_index_times_count);
    
}

void PickTensors::gen_pot_by_shapeev_level(std::string mtfile) {
    Message("Begin");
    gen_alpha_pool_by_shapeev_level(mtfile);
    Message("alphas generated");
    gen_pot_by_alphapool_GA();
}

void PickTensors::gen_pot_by_alphapool_GA() {
    Message("Building MTP contraction search space");
    create_hugepool_and_connections();
    generate_trees();
    // generate alpha_basic_index, this is valid for all selection
    gen_alpha_basic_index();
    non_ele_id_canonical_cache.clear();
    // seed the persistent generation state (accumulates across the search)
    bi_id_map_persist.clear();
    next_moment_id_persist = static_cast<int>(alpha_index_basic_count);
    alpha_index_times_persist.clear();
    alpha_times_flag_persist.clear();
    alpha_moment_mapping_persist.clear();
    root_moment_ids.clear();
    conn_rule_range.clear();
    moment_creator.clear();
    mid_tid_mapping_persist = basic_mid_tid_mapping;
    // dump_all();
    need_regen_first = true;
    while (need_regen_first) {
        build_init();
    }
    dump_all();
    Message(fmt::format(
        "MTP contraction search space ready: {} tensors, {} collections",
        tensor_pool.size(), collections.size()));
    // reduce further, now search branches, remove branches which can be determined by others
    // i.e for a scalar tensor 1936, 
    // 1936 -> 0 373
    // 373 -> 2439 2680 | 373 -> 2449 2566
    // in this case, 373 must selected, and if 373 is a scalar tensor, this branch can be choped


    // convert from tid to kid ???
    
    // now go search, currently kid is not allowed
    GAsearch();
    
    dump_post();
}
