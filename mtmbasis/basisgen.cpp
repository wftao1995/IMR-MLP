// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "solved_tensor_transport.h"
#include "mtm.h"
#include <fstream>
#include <bitset>
#include <iterator>
#include <sstream>

namespace mtmbasis_detail {
namespace {

void enumerate_edges(
        const std::vector<std::pair<int, int>> &edges,
        int edge_id,
        std::vector<int> &remaining,
        MatrixXi &matrix,
        const VectorXi &mus,
        std::vector<alpha_index_type> &result) {
    if (edge_id == static_cast<int>(edges.size())) {
        if (std::any_of(remaining.begin(), remaining.end(),
                        [](int value) { return value != 0; })) {
            return;
        }
        alpha_index_type alpha(static_cast<int>(remaining.size()));
        alpha.first = mus;
        alpha.second = matrix;
        alpha.solve();
        if (std::none_of(result.begin(), result.end(),
                         [&](const alpha_index_type &other) {
                             return alpha == other;
                         })) {
            result.push_back(std::move(alpha));
        }
        return;
    }

    const auto [left, right] = edges[edge_id];
    const bool last_edge_for_left =
        edge_id + 1 == static_cast<int>(edges.size()) ||
        edges[edge_id + 1].first != left;
    const int lower = last_edge_for_left ? remaining[left] : 0;
    const int upper = std::min(remaining[left], remaining[right]);
    if (lower > upper) return;

    for (int value = lower; value <= upper; ++value) {
        remaining[left] -= value;
        remaining[right] -= value;
        matrix(left, right) = value;
        matrix(right, left) = value;

        int total_remaining = 0;
        int maximum_remaining = 0;
        for (int degree : remaining) {
            total_remaining += degree;
            maximum_remaining = std::max(maximum_remaining, degree);
        }
        if (total_remaining % 2 == 0 &&
            maximum_remaining <= total_remaining - maximum_remaining) {
            enumerate_edges(edges, edge_id + 1, remaining, matrix, mus, result);
        }

        matrix(left, right) = 0;
        matrix(right, left) = 0;
        remaining[left] += value;
        remaining[right] += value;
    }
}

} // namespace

std::vector<alpha_index_type> enumerate_scalar_alpha_indices(
        const std::vector<int> &mus,
        const std::vector<int> &nvs) {
    if (mus.size() != nvs.size() || mus.empty() || mus.size() > 7) {
        GPERROR("mtmbasis scalar enumeration requires matching dimensions from 1 to 7");
    }
    if (std::any_of(mus.begin(), mus.end(), [](int value) { return value < 0; }) ||
        std::any_of(nvs.begin(), nvs.end(), [](int value) { return value < 0; })) {
        GPERROR("mtmbasis scalar enumeration requires non-negative mu and nu");
    }

    const int dimension = static_cast<int>(mus.size());
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < dimension; ++i)
        for (int j = i + 1; j < dimension; ++j)
            edges.emplace_back(i, j);

    VectorXi mu_vector(dimension);
    for (int i = 0; i < dimension; ++i) mu_vector(i) = mus[i];
    std::vector<int> remaining = nvs;
    MatrixXi matrix = MatrixXi::Zero(dimension, dimension);
    std::vector<alpha_index_type> result;
    enumerate_edges(edges, 0, remaining, matrix, mu_vector, result);
    std::sort(result.begin(), result.end());
    return result;
}

} // namespace mtmbasis_detail

void PickTensors::gen_alpha_pool_by_shapeev_level(std::string mtfile)
{

    // node_alpha_size.assign(max_dim + 2, 0);
    /**
     * Buffers need to be bcasted:
     *  alphas_gen
     *  node_alpha_size
    */
    if (mpi.rank == 0)
    {
        // 1. gen mtm tensors, all possible M_{\mu, \nu}
        std::vector<int> mus, nvs;
        if (mtfile.empty()) {
            for (int i = 0; i <= max_mu; ++i) {
                for (int j = 0; j <= max_nv; ++j) {
                    mus.push_back(i);
                    nvs.push_back(j);
                }
            }
        } else {
            // file:
            // line1: count
            // line2+: mu, nu, mu, nu, ...
            std::ifstream ifs(mtfile);
            if (!ifs.is_open()) {
                GPERROR((std::string) "Could not open mtmbasis moments file: " + mtfile);
            }
            int count;
            if (!(ifs >> count) || count <= 0) {
                GPERROR((std::string) "Invalid moment count in mtmbasis moments file: " + mtfile);
            }
            mus.resize(count);
            nvs.resize(count);
            std::set<std::pair<int, int>> seen;
            for (int i = 0; i < count; ++i) {
                if (!(ifs >> mus[i] >> nvs[i])) {
                    GPERROR((std::string) "Truncated moment pair list in mtmbasis moments file: " + mtfile);
                }
                if (mus[i] < 0 || nvs[i] < 0) {
                    GPERROR((std::string) "Negative mu or nu in mtmbasis moments file: " + mtfile);
                }
                if (!seen.emplace(mus[i], nvs[i]).second) {
                    GPERROR((std::string) "Duplicate mu, nu pair in mtmbasis moments file: " + mtfile);
                }
            }
            std::string extra;
            if (ifs >> extra) {
                GPERROR((std::string) "Extra data after moment pairs in mtmbasis moments file: " + mtfile);
            }
            ifs.close();
        }
        if (max_dim < 1 || max_dim > 7) {
            GPERROR("mtmbasis max_dim must be between 1 and 7");
        }
        if (max_basic_index_allowed > 0) {
            // pick out some (mu nv) pair
            int bsize = 0;
            for (int i = 0; i < mus.size(); ++i) {
                bsize += (nvs[i] + 1) * (nvs[i] + 2) / 2;
            }
            // sort by bsize and delete maxium ones with higher mu and nv
            if (bsize > max_basic_index_allowed) {
                ;
            }
        }
        // 2. gen mtm tensors combo ristrited by level
        // 2.1 two body, 1d, must be scalar
        std::vector<int> count(max_dim+2,0);
        std::vector<int> nodestart(max_dim+2,0);
        std::vector<std::vector<int>> combo;
        nodestart[1] = 0;
        for (int i = 0; i < mus.size(); ++i) {
            if (nvs[i] == 0 && calc_level(mus[i], nvs[i], 1, mu_wei, nv_wei, dim_wei) <= max_lev) {
                combo.push_back(std::vector<int>({i}));
                count[1]++;
            }
            else continue;
        }
        // 2.2 three body, 2d, must have equal nv
        int cmu1, cnv1, cmu2, cnv2;
        if (max_dim > 1) {
            nodestart[2] = nodestart[1] + count[1];
            for (int i1 = 0; i1 < mus.size(); ++i1) {
                cmu1 = mus[i1];
                cnv1 = nvs[i1];
                if (calc_level(cmu1, cnv1, 2, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                for (int i2 = i1; i2 < mus.size(); ++i2) {
                    if (nvs[i1] != nvs[i2]) continue;
                    cmu2 = cmu1 + mus[i2];
                    cnv2 = cnv1 + nvs[i2];
                    if (cnv2 > max_poly_allowed) continue;
                    if (calc_level(cmu2, cnv2, 2, mu_wei, nv_wei, dim_wei) <= max_lev) {           
                        combo.push_back(std::vector<int>{i1,i2});
                        count[2]++;
                    }
                    else continue;
                }
            }
        }
        // 2.3 four body, 3d, v1+v2-v3, v1-v2+v3, -v1+v2+v3 must be even intengers
        int cmu3, cnv3;
        if (max_dim > 2) {
            nodestart[3] = nodestart[2] + count[2];
            for (int i1 = 0; i1 < mus.size(); ++i1) {
                cmu1 = mus[i1];
                cnv1 = nvs[i1];
                if (calc_level(cmu1, cnv1, 3, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                for (int i2 = i1; i2 < mus.size(); ++i2) {
                    cmu2 = cmu1 + mus[i2];
                    cnv2 = cnv1 + nvs[i2];
                    if (calc_level(cmu2, cnv2, 3, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                    for (int i3 = i2; i3 < mus.size(); ++i3) {
                        int t1 = nvs[i1] + nvs[i2] - nvs[i3];
                        int t2 = nvs[i1] - nvs[i2] + nvs[i3];
                        int t3 =-nvs[i1] + nvs[i2] + nvs[i3];
                        if (t1<0 || t1%2!=0 || t2<0 || t2%2!=0 || t3<0 || t3%2!=0) continue;
                        cmu3 = cmu2 + mus[i3];
                        cnv3 = cnv2 + nvs[i3];
                        if (cnv3 > max_poly_allowed) continue;
                        if (calc_level(cmu3, cnv3, 3, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                        else {
                            combo.push_back(std::vector<int>{i1,i2,i3});
                            count[3]++;
                        }
                    }
                }
            }
        }
        // 2.4 five body, 4d
        int cmu4, cnv4;
        if (max_dim > 3) {
            nodestart[4] = nodestart[3] + count[3];
            for (int i1 = 0; i1 < mus.size(); ++i1) {
                cmu1 = mus[i1];
                cnv1 = nvs[i1];
                if (calc_level(cmu1, cnv1, 4, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                for (int i2 = i1; i2 < mus.size(); ++i2) {
                    cmu2 = cmu1 + mus[i2];
                    cnv2 = cnv1 + nvs[i2];
                    if (calc_level(cmu2, cnv2, 4, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                    for (int i3 = i2; i3 < mus.size(); ++i3) {
                        cmu3 = cmu2 + mus[i3];
                        cnv3 = cnv2 + nvs[i3];
                        if (calc_level(cmu3, cnv3, 4, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                        for (int i4 = i3; i4 < mus.size(); ++i4) {
                            cmu4 = cmu3 + mus[i4];
                            cnv4 = cnv3 + nvs[i4];
                            if (cnv4 > max_poly_allowed) continue;
                            if (calc_level(cmu4, cnv4, 4, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                            else {
                                combo.push_back(std::vector<int>{i1,i2,i3,i4});
                                // if (mus[i1] == 0 && mus[i2] == 0 && mus[i3] == 0 & mus[i4] == 0)
                                count[4]++;
                            }
                        }
                    }
                }
            }
        }
        // 2.5 six body, 5d
        int cmu5, cnv5;
        if (max_dim > 4) {
            nodestart[5] = nodestart[4] + count[4];
            for (int i1 = 0; i1 < mus.size(); ++i1) {
                cmu1 = mus[i1];
                cnv1 = nvs[i1];
                if (calc_level(cmu1, cnv1, 5, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                for (int i2 = i1; i2 < mus.size(); ++i2) {
                    cmu2 = cmu1 + mus[i2];
                    cnv2 = cnv1 + nvs[i2];
                    if (calc_level(cmu2, cnv2, 5, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                    for (int i3 = i2; i3 < mus.size(); ++i3) {
                        cmu3 = cmu2 + mus[i3];
                        cnv3 = cnv2 + nvs[i3];
                        if (calc_level(cmu3, cnv3, 5, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                        for (int i4 = i3; i4 < mus.size(); ++i4) {
                            cmu4 = cmu3 + mus[i4];
                            cnv4 = cnv3 + nvs[i4];
                            if (calc_level(cmu4, cnv4, 5, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                            for (int i5 = i4; i5 < mus.size(); ++i5) {
                                cmu5 = cmu4 + mus[i5];
                                cnv5 = cnv4 + nvs[i5];
                                if (cnv5 > max_poly_allowed) continue;
                                if (calc_level(cmu5, cnv5, 5, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                                else {
                                    combo.push_back(std::vector<int>{i1,i2,i3,i4,i5});
                                    count[5]++;
                                }
                            }
                        }
                    }
                }
            }
        }
        // 2.6 seven body, 6d
        int cmu6, cnv6;
        if (max_dim > 5) {
            nodestart[6] = nodestart[5] + count[5];
            for (int i1 = 0; i1 < mus.size(); ++i1) {
                cmu1 = mus[i1];
                cnv1 = nvs[i1];
                if (calc_level(cmu1, cnv1, 6, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                for (int i2 = i1; i2 < mus.size(); ++i2) {
                    cmu2 = cmu1 + mus[i2];
                    cnv2 = cnv1 + nvs[i2];
                    if (calc_level(cmu2, cnv2, 6, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                    for (int i3 = i2; i3 < mus.size(); ++i3) {
                        cmu3 = cmu2 + mus[i3];
                        cnv3 = cnv2 + nvs[i3];
                        if (calc_level(cmu3, cnv3, 6, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                        for (int i4 = i3; i4 < mus.size(); ++i4) {
                            cmu4 = cmu3 + mus[i4];
                            cnv4 = cnv3 + nvs[i4];
                            if (calc_level(cmu4, cnv4, 6, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                            for (int i5 = i4; i5 < mus.size(); ++i5) {
                                cmu5 = cmu4 + mus[i5];
                                cnv5 = cnv4 + nvs[i5];
                                if (calc_level(cmu5, cnv5, 6, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                                for (int i6 = i5; i6 < mus.size(); ++i6) {
                                    cmu6 = cmu5 + mus[i6];
                                    cnv6 = cnv5 + nvs[i6];
                                    if (cnv6 > max_poly_allowed) continue;
                                    if (calc_level(cmu6, cnv6, 6, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                                    else {
                                        count[6]++;
                                        combo.push_back(std::vector<int>{i1,i2,i3,i4,i5,i6});
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // 2.7 eight body, 7d
        int cmu7, cnv7;
        if (max_dim > 6) {
            nodestart[7] = nodestart[6] + count[6];
            for (int i1 = 0; i1 < mus.size(); ++i1) {
                cmu1 = mus[i1];
                cnv1 = nvs[i1];
                if (calc_level(cmu1, cnv1, 7, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                for (int i2 = i1; i2 < mus.size(); ++i2) {
                    cmu2 = cmu1 + mus[i2];
                    cnv2 = cnv1 + nvs[i2];
                    if (calc_level(cmu2, cnv2, 7, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                    for (int i3 = i2; i3 < mus.size(); ++i3) {
                        cmu3 = cmu2 + mus[i3];
                        cnv3 = cnv2 + nvs[i3];
                        if (calc_level(cmu3, cnv3, 7, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                        for (int i4 = i3; i4 < mus.size(); ++i4) {
                            cmu4 = cmu3 + mus[i4];
                            cnv4 = cnv3 + nvs[i4];
                            if (calc_level(cmu4, cnv4, 7, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                            for (int i5 = i4; i5 < mus.size(); ++i5) {
                                cmu5 = cmu4 + mus[i5];
                                cnv5 = cnv4 + nvs[i5];
                                if (calc_level(cmu5, cnv5, 7, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                                for (int i6 = i5; i6 < mus.size(); ++i6) {
                                    cmu6 = cmu5 + mus[i6];
                                    cnv6 = cnv5 + nvs[i6];
                                    if (calc_level(cmu6, cnv6, 7, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                                    for (int i7 = i6; i7 < mus.size(); ++i7) {
                                        cmu7 = cmu6 + mus[i7];
                                        cnv7 = cnv6 + nvs[i7];
                                        if (cnv7 > max_poly_allowed) continue;
                                        if (calc_level(cmu7, cnv7, 7, mu_wei, nv_wei, dim_wei) > max_lev) continue;
                                        else {
                                            count[7]++;
                                            combo.push_back(std::vector<int>{i1,i2,i3,i4,i5,i6,i7});
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // manybody
        // NOTE: unreachable while the max_dim <= 7 check above errors out; kept in
        // case > 7-body tensors are ever supported again.
        if (max_dim > 7) {
            for (int cdim = 7; cdim <= max_dim; ++cdim) {
                nodestart[cdim] = nodestart[cdim - 1] + count[cdim - 1];
                std::vector<int> ll(cdim, 0);
                recursive_gen(ll, 0, 0, 0, combo, mus, nvs);
            }
        }

        nodestart[max_dim+1] = nodestart[max_dim] + count[max_dim];

        // Generate every labeled contraction multigraph in a single,
        // dimension-independent order, then remove only exact permutation
        // equivalents.  This replaces the former dimension-specific formulae.
        for (const auto &ids : combo) {
            std::vector<int> combo_mus(ids.size());
            std::vector<int> combo_nvs(ids.size());
            for (int i = 0; i < static_cast<int>(ids.size()); ++i) {
                combo_mus[i] = mus[ids[i]];
                combo_nvs[i] = nvs[ids[i]];
            }
            auto generated = mtmbasis_detail::enumerate_scalar_alpha_indices(
                combo_mus, combo_nvs);
            alphas_gen.insert(alphas_gen.end(),
                              std::make_move_iterator(generated.begin()),
                              std::make_move_iterator(generated.end()));
        }


        // #pragma omp parallel for 
        // for (int i = 0; i < alphas_gen.size(); ++i) {
        //     alphas_gen[i].solve();
        // }

        std::stable_sort(alphas_gen.begin(), alphas_gen.end());
        std::ofstream ofs("alphapool.dat");
        for (int i = 0; i < alphas_gen.size(); ++i) {
            ofs << "Bid: " << i << '\n';
            auto &b = alphas_gen[i];
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
    MPI_Barrier(MPI_COMM_WORLD);
    Message("alpha generated, boradcasting");
    mtmbasis_detail::broadcast_solved_tensors(alphas_gen);
    Message("alpha distributed on each core");
    MPI_Barrier(MPI_COMM_WORLD);
}

void PickTensors::load_alpha_pool_from_file(const std::string &file) {
    std::ifstream ifs(file);
    if (!ifs.is_open()) {
        GPERROR((std::string) "Could not open mtmbasis alpha pool: " + file);
    }

    alphas_gen.clear();
    int maxdim = 1;
    int line_number = 0;
    int block_number = 0;

    auto read_line = [&](std::string &line) -> bool {
        if (!std::getline(ifs, line)) return false;
        ++line_number;
        return true;
    };
    auto parse_values = [&](const std::string &line,
                            const std::string &prefix,
                            int expected,
                            int block) {
        if (!line.starts_with(prefix)) {
            GPERROR(fmt::format(
                "{}:{}: expected '{}' in alpha pool block {}",
                file, line_number, prefix, block));
        }
        std::istringstream input(line.substr(prefix.size()));
        std::vector<int> values;
        int value = 0;
        while (input >> value) values.push_back(value);
        if (!input.eof() || (expected >= 0 && values.size() != static_cast<size_t>(expected))) {
            GPERROR(fmt::format(
                "{}:{}: invalid integer list in alpha pool block {}",
                file, line_number, block));
        }
        return values;
    };

    std::string line;
    while (read_line(line)) {
        if (line.find_first_not_of(" \t\r") == std::string::npos) continue;
        ++block_number;
        if (!line.starts_with("Bid:")) {
            GPERROR(fmt::format(
                "{}:{}: expected 'Bid:' at start of alpha pool block {}",
                file, line_number, block_number));
        }

        if (!read_line(line)) {
            GPERROR(fmt::format("{}: truncated alpha pool block {}", file, block_number));
        }
        auto mus = parse_values(line, "mus:", -1, block_number);
        const int dimension = static_cast<int>(mus.size());
        if (dimension < 1 || dimension > 7 ||
            std::any_of(mus.begin(), mus.end(), [](int value) { return value < 0; })) {
            GPERROR(fmt::format(
                "{}:{}: invalid dimension or mu in alpha pool block {}",
                file, line_number, block_number));
        }

        if (!read_line(line)) {
            GPERROR(fmt::format("{}: truncated alpha pool block {}", file, block_number));
        }
        auto nvs = parse_values(line, "nvs:", dimension, block_number);
        if (std::any_of(nvs.begin(), nvs.end(), [](int value) { return value < 0; })) {
            GPERROR(fmt::format(
                "{}:{}: negative nu in alpha pool block {}",
                file, line_number, block_number));
        }

        if (!read_line(line) || !line.starts_with("level:")) {
            GPERROR(fmt::format(
                "{}:{}: missing level line in alpha pool block {}",
                file, line_number, block_number));
        }
        if (!read_line(line)) {
            GPERROR(fmt::format("{}: truncated alpha pool block {}", file, block_number));
        }
        auto repeated_mus = parse_values(line, "", dimension, block_number);
        if (repeated_mus != mus) {
            GPERROR(fmt::format(
                "{}:{}: repeated mu list does not match in alpha pool block {}",
                file, line_number, block_number));
        }

        alpha_index_type alpha(dimension);
        for (int i = 0; i < dimension; ++i) alpha.first(i) = mus[i];
        for (int i = 0; i < dimension; ++i) {
            if (!read_line(line)) {
                GPERROR(fmt::format("{}: truncated matrix in alpha pool block {}",
                                    file, block_number));
            }
            auto row = parse_values(line, "", dimension, block_number);
            for (int j = 0; j < dimension; ++j) alpha.second(i, j) = row[j];
        }
        if ((alpha.second.array() < 0).any() ||
            alpha.second != alpha.second.transpose() ||
            alpha.second.trace() != 0) {
            GPERROR(fmt::format(
                "{}:{}: alpha matrix must be non-negative, symmetric, and scalar in block {}",
                file, line_number, block_number));
        }
        for (int i = 0; i < dimension; ++i) {
            if (alpha.second.col(i).sum() != nvs[i]) {
                GPERROR(fmt::format(
                    "{}:{}: nu does not match matrix degree in alpha pool block {}",
                    file, line_number, block_number));
            }
        }
        alpha.solve();
        alphas_gen.push_back(std::move(alpha));
        maxdim = std::max(maxdim, dimension);
    }
    if (alphas_gen.empty()) {
        GPERROR((std::string) "No alpha blocks found in mtmbasis alpha pool: " + file);
    }
    max_dim = maxdim;
    // assign node alpha size, assume alphas are sorted by dim
    // node_alpha_size.assign(max_dim + 2, 0);
    // for (int i = 0; i < alphas_gen.size(); ++i) {
    //     int dim = alphas_gen[i].first.size();
    //     node_alpha_size[dim]++;
    // }
    // for (int i = 1; i < node_alpha_size.size(); ++i) {
    //     node_alpha_size[i] += node_alpha_size[i-1];
    // }
}

void PickTensors::recursive_gen(std::vector<int> &ll,
                                int lpos, int cmu, int cnv,
                                std::vector<std::vector<int>> &results,
                                std::vector<int> &mus,
                                std::vector<int> &nus)
{
    // NOTE: use `continue` (not `return`) in the level filters so that combos are
    // not silently skipped when mus/nus from a custom moments file are not sorted.
    if (lpos == ll.size() - 1) {
        for (int i = ll[lpos-1]; i < mus.size(); ++i) {
            ll[lpos] = i;
            int ncmu = cmu + mus[i];
            int ncnv = cnv + nus[i];
            if (ncnv > max_poly_allowed) continue;
            if (calc_level(ncmu, ncnv, ll.size(), mu_wei, nv_wei, dim_wei) > max_lev) continue;
            results.push_back(ll);
        }
        return;
    }
    else if (lpos == 0){
        for (int i = 0; i < mus.size(); ++i) {
            ll[lpos] = i;
            int ncmu = cmu + mus[i];
            int ncnv = cnv + nus[i];
            if (calc_level(ncmu, ncnv, ll.size(), mu_wei, nv_wei, dim_wei) > max_lev) continue;
            recursive_gen(ll, lpos+1, ncmu, ncnv, results, mus, nus);
        }
    }
    else {
        for (int i = ll[lpos-1]; i < mus.size(); ++i) {
            ll[lpos] = i;
            int ncmu = cmu + mus[i];
            int ncnv = cnv + nus[i];
            if (calc_level(ncmu, ncnv, ll.size(), mu_wei, nv_wei, dim_wei) > max_lev) continue;
            recursive_gen(ll, lpos+1, ncmu, ncnv, results, mus, nus);
        }
    }
}

void PickTensors::gen_alpha_basic_index() {
    alpha_index_basic.clear();
    alpha_index_basic_map.clear();
    basic_mid_tid_mapping.clear();
    alpha_index_basic_count = 0;
    radial_funcs_count = 0;
    for (auto &i : mt_ids)
    {
        int mu = tensor_pool[i].first(0);
        int nv = tensor_pool[i].second(0, 0);
        int size = (nv + 1) * (nv + 2) / 2;
        alpha_index_basic_count += size;
        radial_funcs_count = mu > radial_funcs_count ? mu : radial_funcs_count;
    }
    ++radial_funcs_count;
    alpha_index_basic.reserve(alpha_index_basic_count);
    for (auto &i : mt_ids)
    {
        int mu = tensor_pool[i].first(0);
        int nv = tensor_pool[i].second(0, 0);
        for (int l1 = 0; l1 <= nv; ++l1)
        {
            for (int l2 = 0; l2 <= nv - l1; ++l2)
            {
                int l3 = nv - l1 - l2;
                alpha_index_basic.push_back(std::array<int, 4>{mu, l1, l2, l3});
                alpha_index_basic_map[alpha_index_basic.back()] = alpha_index_basic.size() - 1;
                basic_mid_tid_mapping[alpha_index_basic.size() - 1] = tensor_pool[i].tensor_id;
            }
        }
        // put mt tensors into huge pool so that k_id is same as t_id
        ;
    }
}

#define DEBUG_SCALAR(a) std::cout << "SCALAR_ADDED: " << a.tensor_id << std::endl;

void PickTensors::gen_alpha_index_times(const std::vector<int> &tensors, const std::vector<alpha_decomp_connection> &conns,
                std::unordered_map<non_ele_id_type, int, non_ele_id_hash> &bi_id_map_this,
                int &alpha_moments_count_this,
                int &alpha_index_times_count_this,
                std::vector<std::array<int, 4>> &alpha_index_times_this,
                alpha_times_index_flag_type &alpha_times_flag_this,
                std::vector<int> &alpha_moment_mapping_this,
                std::set<int> &tensorpool_used_this,
                std::map<int, int> &mid_tid_mapping) {
    // _DEBUG_OUTPUT_("CALL");
    std::set<int> flags;
    const std::vector<alpha_decomp_connection> &conns_vec = conns;
    // flag mt tensors
    for (auto &it : tensors) {
        if (tensor_pool[it].size() == 1) {
            flags.insert(it);
            tensorpool_used_this.insert(it);
            if (tensor_pool[it].is_scalar()) {
                std::array<int, 4> tmpind;
                tmpind[0] = tensor_pool[it].first(0);
                tmpind[1] = 0;
                tmpind[2] = 0;
                tmpind[3] = 0;
                // DEBUG_SCALAR(tensor_pool[it])
                // _DEBUG_OUTPUT_(alpha_index_basic_map[tmpind]);
                if (alpha_index_basic_map.count(tmpind) == 0) {
                    GPERROR("Missing scalar basic component in mtmbasis generation");
                }
                mid_tid_mapping[alpha_index_basic_map[tmpind]] = it;
                alpha_moment_mapping_this.push_back(alpha_index_basic_map[tmpind]);
            }
        }
    }
    // Single pass ordered by root dimension: every child tensor has a
    // strictly smaller dimension than its parent, so the children's
    // connections are always ready before their parents' in this order.
    std::vector<alpha_decomp_connection> ordered_conns(conns_vec);
    std::stable_sort(ordered_conns.begin(), ordered_conns.end(),
        [&](const alpha_decomp_connection &left,
            const alpha_decomp_connection &right) {
            return tensor_pool[left[0]].size() < tensor_pool[right[0]].size();
        });
    for (const auto &conn : ordered_conns) {
        int tid = conn[0];
        int lid = conn[1];
        int rid = conn[2];
        tensorpool_used_this.insert(tid);
        tensorpool_used_this.insert(lid);
        tensorpool_used_this.insert(rid);
        if (flags.count(tid)) continue;
        // Skip connections already processed in an earlier evaluation:
        // their moments and rules live in the persistent state, so
        // nothing new can be generated for them.
        if (conn_rule_range.count(conn)) {
            flags.insert(tid);
            continue;
        }
        flags.insert(tid);
        const int rules_before = alpha_index_times_count_this;
        std::set<int> ruled_this_conn;
            auto b = tensor_pool[tid];
            auto &bl = tensor_pool[lid];
            auto &br = tensor_pool[rid];
            // case1. two mt
            if (mt_ids.count(conn[1]) && mt_ids.count(conn[2])) {
                // 1. build correct form of b
                if ((b.second.row(0).sum() != bl.second(0, 0)) || 
                     b.first(0) != bl.first(0)) {
                    b.bswap(0, 1);
                }
                // go on
                int contract = b.second(0, 1);
                int mul = bl.first(0);
                int nvl = bl.second(0, 0);
                int lnvl = nvl - contract; // rank left after contract
                int mur = br.first(0);
                int nvr = br.second(0, 0);
                int lnvr = nvr - contract;
                // target_dim
                std::array<int, 4> index_basic;
                for (int l1 = 0; l1 <= lnvl; ++l1) {
                    for (int l2 = 0; l2 <= lnvl - l1; ++l2) {
                        int l3 = lnvl - l1 - l2;
                        for (int p1 = 0; p1 <= lnvr; ++p1) {
                            for (int p2 = 0; p2 <= lnvr - p1; ++p2) {
                                int p3 = lnvr - p1 - p2;
                                std::vector<int> biid(6);
                                std::vector<int> bimu(2);
                                std::vector<int> binv(2);
                                binv[0] = lnvl;
                                binv[1] = lnvr;
                                bimu[0] = mul;
                                bimu[1] = mur;
                                biid[0] = l1;
                                biid[1] = l2;
                                biid[2] = l3;
                                biid[3] = p1;
                                biid[4] = p2;
                                biid[5] = p3;
                                non_ele_id_type bi_id(
                                    tid, b, bimu, binv, biid,
                                    &non_ele_id_canonical_cache);
                                auto [moment_it, moment_inserted] =
                                    bi_id_map_this.emplace(std::move(bi_id),
                                                           alpha_moments_count_this);
                                const int moment_id =
                                    moment_inserted ? alpha_moments_count_this++
                                                    : moment_it->second;
                                if (moment_inserted)
                                {
                                    root_moment_ids[tid].push_back(moment_id);
                                    moment_creator[moment_id] = conn;
                                }
                                // Rules are generated when the moment is new,
                                // or when it was created by a different
                                // decomposition of the same tensor (shared
                                // moments need the current connection's
                                // construction).  Within one connection only
                                // the first occurrence generates rules.
                                if ((moment_inserted ||
                                     moment_creator[moment_id] != conn) &&
                                    ruled_this_conn.insert(moment_id).second)
                                {
                                    // @NEED: non-basic index
                                    for (int c1 = 0; c1 <= contract; ++c1)
                                    {
                                        for (int c2 = 0; c2 <= contract - c1; ++c2)
                                        {
                                            int c3 = contract - c1 - c2;
                                            // NOTE: multinomial coefficient. Fits in int for
                                            // contract <= 12 (guaranteed by the default max_nv = 10);
                                            // a custom moments file with nv >= 13 would overflow here.
                                            int times = Combinatorics::factorial_int(contract) /
                                                        Combinatorics::factorial_int(c1) /
                                                        Combinatorics::factorial_int(c2) /
                                                        Combinatorics::factorial_int(c3); // how many {c1, c2, c3} -> {\beta}
                                            std::array<int, 3> tmpind;
                                            std::array<int, 4> basic_id1, basic_id2;
                                            basic_id1[0] = mul;
                                            basic_id1[1] = c1 + l1;
                                            basic_id1[2] = c2 + l2;
                                            basic_id1[3] = c3 + l3;
                                            basic_id2[0] = mur;
                                            basic_id2[1] = c1 + p1;
                                            basic_id2[2] = c2 + p2;
                                            basic_id2[3] = c3 + p3;
                                            auto basic1 = alpha_index_basic_map.find(basic_id1);
                                            auto basic2 = alpha_index_basic_map.find(basic_id2);
                                            if (basic1 == alpha_index_basic_map.end() ||
                                                basic2 == alpha_index_basic_map.end()) {
                                                GPERROR("Missing basic component in MT-MT mtmbasis generation");
                                            }
                                            tmpind[0] = basic1->second;
                                            tmpind[1] = basic2->second;
                                            // tmpind[2] = times;
                                            // tmpind[3] = alpha_moments_count_this - 1;
                                            tmpind[2] = moment_id;
                                            // add moment id to moment_tensor mapping 
                                            // mid_tid_mapping[tmpind[3]] = tid;
                                            mid_tid_mapping[tmpind[2]] = tid;
                                                    auto [rule_it, rule_inserted] =
                                                alpha_times_flag_this.emplace(
                                                    tmpind, alpha_index_times_this.size());
                                            if (rule_inserted)
                                            {
                                                alpha_index_times_count_this += 1;
                                                alpha_index_times_this.push_back({tmpind[0], tmpind[1], times, tmpind[2]});
                                            } else {
                                                /**
                                                 * @brief This case will happen
                                                 * i.e
                                                 * a = 
                                                 * 0 ( 1 3 )
                                                 * 0   3 1  
                                                 * and 
                                                 * b = 0 (4)
                                                 * a -> b + b
                                                 * Then {013,013} will appear twice, so does {022,022}
                                                 */
                                                int id = rule_it->second;
                                                alpha_index_times_this[id][2] += times;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                // if scalar, add to scalar map
                if (b.is_scalar()) {
                    if (alpha_gen_ids.count(b.tensor_id) || add_new_scalar_to_pot) {
                        alpha_moment_mapping_this.push_back(alpha_index_times_this.back().back());
                    } 
                    // DEBUG_SCALAR(b)
                    // _DEBUG_OUTPUT_(alpha_index_times_this.back().back());
                }
            }
            // case2. one mt, left
            else if (mt_ids.count(conn[1])) {
                /**
                 * Both left and right now is already correct, just correct form of b
                 * and extract contraction indices matrix from restored b
                 */
                // Eigen::MatrixXi contract = it.second;
                Eigen::MatrixXi contract;
                b = restore_b(b, bl, br);
                contract = b.second.block(0, bl.size(), bl.size(), br.size());

                int mul = bl.first(0);
                int nvl = bl.second(0, 0);
                std::vector<int> mu(1 + br.size());
                std::vector<int> nv(mu.size());
                for (int i = 0; i < b.size(); ++i)
                {
                    mu[i] = b.first(i);
                    nv[i] = b.second(i, i);
                }
                int tmulti = 1;
                int lnvl = nvl - contract.sum();
                int sizer = br.size();
                for (int i = 0; i < sizer; ++i)
                {
                    tmulti *= Combinatorics::factorial_int(contract(0, i));
                }
                std::vector<std::vector<std::array<int, 3>>> contractinds(sizer); // combine them to get real contraction inds
                std::vector<std::array<int, 3>> leftinds;
                std::vector<std::vector<std::array<int, 3>>> rightinds(sizer);
                std::vector<int> mur(sizer), nvr(sizer), lnvr(sizer);
                int lnvrsum = 0;
                for (int i = 0; i < sizer; ++i)
                {
                    mur[i] = br.first[i];
                    nvr[i] = br.second(i, i);
                    lnvr[i] = nvr[i] - contract(0, i);
                    lnvrsum += lnvr[i];
                }
                // build all contract inds
                for (int i = 0; i < sizer; ++i)
                {
                    std::array<int, 3> tmpids; // sum of elements equals to nvi
                    int nvthis = contract(0, i);
                    int nele = (nvthis + 1) * (nvthis + 2) / 2;
                    contractinds[i].reserve(nele);
                    for (int c1 = 0; c1 <= nvthis; ++c1)
                    {
                        for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                        {
                            int c3 = nvthis - c1 - c2;
                            tmpids[0] = c1;
                            tmpids[1] = c2;
                            tmpids[2] = c3;
                            contractinds[i].push_back(tmpids);
                        }
                    }
                }
                // build non-equvalent indices of left
                std::array<int, 3> tmpids;
                int nvthis = lnvl;
                int nele = (nvthis + 1) * (nvthis + 2) / 2;
                leftinds.reserve(nele);
                for (int c1 = 0; c1 <= nvthis; ++c1)
                {
                    for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                    {
                        int c3 = nvthis - c1 - c2;
                        tmpids[0] = c1;
                        tmpids[1] = c2;
                        tmpids[2] = c3;
                        leftinds.push_back(tmpids);
                    }
                }
                // build non-equvenlent indices of right
                for (int i = 0; i < sizer; ++i)
                {
                    std::array<int, 3> tmpids; // sum of elements equals to nvi
                    int nvthis = lnvr[i];
                    int nele = (nvthis + 1) * (nvthis + 2) / 2;
                    rightinds[i].reserve(nele);
                    for (int c1 = 0; c1 <= nvthis; ++c1)
                    {
                        for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                        {
                            int c3 = nvthis - c1 - c2;
                            tmpids[0] = c1;
                            tmpids[1] = c2;
                            tmpids[2] = c3;
                            rightinds[i].push_back(tmpids);
                        }
                    }
                }
                // do contraction
                /* combine left and right inds, then loop over contraciton inds */
                for (int i = 0; i < leftinds.size(); ++i)
                {
                    std::vector<int> tmplate_ind(3 + 3 * sizer);   // id, betas
                    std::array<int, 4> tmplate_left_ind;           // residual left
                    std::vector<int> tmplate_right_ind(3 * sizer); // redidual right
                    std::array<int, 4> tmp_cleft;
                    std::vector<int> tmp_cright(3 * sizer);
                    // tmp_cright[0] = idr;
                    tmp_cleft[0] = bl.first(0);
                    // tmplate_right_ind[0] = idr;
                    // tmplate_ind[0] = idm;
                    tmplate_ind[0] = leftinds[i][0];
                    tmplate_ind[1] = leftinds[i][1];
                    tmplate_ind[2] = leftinds[i][2];
                    tmplate_left_ind[0] = bl.first(0);
                    tmplate_left_ind[1] = leftinds[i][0];
                    tmplate_left_ind[2] = leftinds[i][1];
                    tmplate_left_ind[3] = leftinds[i][2];
                    // use looper to loop right inds
                    std::vector<int> rlooperlim(sizer);
                    for (int ll = 0; ll < sizer; ++ll)
                    {
                        rlooperlim[ll] = rightinds[ll].size();
                    }
                    looper loop_r_ind(rlooperlim);
                    // std::array<int, 4> tmp_all_ind_times;
                    std::array<int, 3> tmp_all_ind_times;
                    for (int j = 0; j < loop_r_ind.size(); ++j)
                    {
                        int tshift = 3;
                        for (int ll = 0; ll < rightinds.size(); ++ll)
                        {
                            tmplate_ind[tshift] = rightinds[ll][loop_r_ind.loop[ll]][0];
                            tmplate_ind[tshift + 1] = rightinds[ll][loop_r_ind.loop[ll]][1];
                            tmplate_ind[tshift + 2] = rightinds[ll][loop_r_ind.loop[ll]][2];
                            tmplate_right_ind[tshift - 3] = rightinds[ll][loop_r_ind.loop[ll]][0];
                            tmplate_right_ind[tshift - 2] = rightinds[ll][loop_r_ind.loop[ll]][1];
                            tmplate_right_ind[tshift - 1] = rightinds[ll][loop_r_ind.loop[ll]][2];
                            tshift += 3;
                        }

                        non_ele_id_type bi_id(
                            tid, b, mu, nv, tmplate_ind,
                            &non_ele_id_canonical_cache);
                        auto [moment_it, moment_inserted] =
                            bi_id_map_this.emplace(std::move(bi_id),
                                                   alpha_moments_count_this);
                        const int moment_id =
                            moment_inserted ? alpha_moments_count_this++
                                            : moment_it->second;
                        if (moment_inserted)
                        {
                            root_moment_ids[tid].push_back(moment_id);
                            moment_creator[moment_id] = conn;
                        }
                        // Rules are generated when the moment is new, or when
                        // it was created by a different decomposition (shared
                        // moments, see case 1 note); within one connection
                        // only the first occurrence generates rules.
                        if ((moment_inserted ||
                             moment_creator[moment_id] != conn) &&
                            ruled_this_conn.insert(moment_id).second)
                        {
                            // now loop over contraction inds
                            /* combine each block contraction inds */
                            std::vector<int> loopc_limit(contractinds.size());
                            for (int ll = 0; ll < contractinds.size(); ++ll)
                            {
                                loopc_limit[ll] = contractinds[ll].size();
                            }
                            // tmp_all_ind_times[3] = alpha_moments_count_this - 1;
                            tmp_all_ind_times[2] = moment_id;
                            // add mapping
                            mid_tid_mapping[moment_id] = tid;
                            looper loop_c_ind(loopc_limit);
                            for (int lcr = 0; lcr < loop_c_ind.size(); ++lcr)
                            {
                                int times = tmulti;
                                std::vector<int> tmplate_c_inds(3 * sizer);
                                int ct = 0;
                                tmp_cleft = tmplate_left_ind;
                                tmp_cright = tmplate_right_ind;
                                for (int ll = 0; ll < contractinds.size(); ++ll)
                                {
                                    int v1, v2, v3;
                                    v1 = contractinds[ll][loop_c_ind.loop[ll]][0];
                                    v2 = contractinds[ll][loop_c_ind.loop[ll]][1];
                                    v3 = contractinds[ll][loop_c_ind.loop[ll]][2];
                                    tmp_cright[ct] += v1;
                                    ++ct;
                                    tmp_cright[ct] += v2;
                                    ++ct;
                                    tmp_cright[ct] += v3;
                                    ++ct;
                                    tmp_cleft[1] += v1;
                                    tmp_cleft[2] += v2;
                                    tmp_cleft[3] += v3;
                                    times /= Combinatorics::factorial_int(v1) *
                                             Combinatorics::factorial_int(v2) *
                                             Combinatorics::factorial_int(v3);
                                }
                                non_ele_id_type tmp_id_cright(rid, br, mur, nvr, tmp_cright);
                                auto right_component = bi_id_map_this.find(tmp_id_cright);
                                if (right_component == bi_id_map_this.end())
                                {
                                    GPERROR("Missing right non-MT component in mtmbasis generation");
                                }
                                auto left_component = alpha_index_basic_map.find(tmp_cleft);
                                if (left_component == alpha_index_basic_map.end())
                                {
                                    GPERROR("Missing left basic component in mtmbasis generation");
                                }
                                tmp_all_ind_times[0] = left_component->second;
                                tmp_all_ind_times[1] = right_component->second;
                                // tmp_all_ind_times[2] = times;
                                auto [rule_it, rule_inserted] =
                                    alpha_times_flag_this.emplace(
                                        tmp_all_ind_times, alpha_index_times_this.size());
                                if (rule_inserted)
                                {
                                    alpha_index_times_count_this += 1;
                                    // alpha_index_times_this.push_back(tmp_all_ind_times);
                                    alpha_index_times_this.push_back({tmp_all_ind_times[0], tmp_all_ind_times[1], times, tmp_all_ind_times[2]});
                                }else {
                                    int id = rule_it->second;
                                    alpha_index_times_this[id][2] += times;
                                }
                                loop_c_ind.next();
                            }
                        }
                        loop_r_ind.next();
                    }
                }
                // if scalar
                if (b.is_scalar()) {
                    // DEBUG_SCALAR(b)
                    // _DEBUG_OUTPUT_(alpha_index_times_this.back().back());
                    if (alpha_gen_ids.count(b.tensor_id) || add_new_scalar_to_pot) {
                        alpha_moment_mapping_this.push_back(alpha_index_times_this.back().back());
                    } 
                }
            }
            // case3. one mt, right
            else if (mt_ids.count(conn[2])) {
                Eigen::MatrixXi contract;
                b = restore_b(b, bl, br);
                contract = b.second.block(0, bl.size(), bl.size(), br.size());

                std::vector<int> mu(1 + bl.size());
                std::vector<int> nv(mu.size());
                for (int i = 0; i < b.size(); ++i)
                {
                    mu[i] = b.first(i);
                    nv[i] = b.second(i, i);
                }
                int mur = br.first[0];
                int nvr = br.second(0, 0);
                int tmulti = 1;
                int lnvr = nvr - contract.sum();
                int sizel = bl.size();
                for (int i = 0; i < sizel; ++i)
                {
                    tmulti *= Combinatorics::factorial_int(contract(i, 0));
                }
                std::vector<std::vector<std::array<int, 3>>> contractinds(sizel); // combine them to get real contraction inds
                std::vector<std::array<int, 3>> rightinds;
                std::vector<std::vector<std::array<int, 3>>> leftinds(sizel);
                std::vector<int> mul(sizel), nvl(sizel), lnvl(sizel);
                int lnvlsum = 0;
                for (int i = 0; i < sizel; ++i)
                {
                    mul[i] = bl.first[i];
                    nvl[i] = bl.second(i, i);
                    lnvl[i] = nvl[i] - contract(i, 0);
                    lnvlsum += lnvl[i];
                }
                // build all contract inds
                for (int i = 0; i < sizel; ++i)
                {
                    std::array<int, 3> tmpids; // sum of elements equals to nvi
                    int nvthis = contract(i, 0);
                    int nele = (nvthis + 1) * (nvthis + 2) / 2;
                    contractinds[i].reserve(nele);
                    for (int c1 = 0; c1 <= nvthis; ++c1)
                    {
                        for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                        {
                            int c3 = nvthis - c1 - c2;
                            tmpids[0] = c1;
                            tmpids[1] = c2;
                            tmpids[2] = c3;
                            contractinds[i].push_back(tmpids);
                        }
                    }
                }
                // build non-equvalent indices of right
                {
                    std::array<int, 3> tmpids;
                    int nvthis = lnvr;
                    int nele = (nvthis + 1) * (nvthis + 2) / 2;
                    rightinds.reserve(nele);
                    for (int c1 = 0; c1 <= nvthis; ++c1)
                    {
                        for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                        {
                            int c3 = nvthis - c1 - c2;
                            tmpids[0] = c1;
                            tmpids[1] = c2;
                            tmpids[2] = c3;
                            rightinds.push_back(tmpids);
                        }
                    }
                }
                // build non-equvenlent indices of left
                for (int i = 0; i < sizel; ++i)
                {
                    std::array<int, 3> tmpids; // sum of elements equals to nvi
                    int nvthis = lnvl[i];
                    int nele = (nvthis + 1) * (nvthis + 2) / 2;
                    leftinds[i].reserve(nele);
                    for (int c1 = 0; c1 <= nvthis; ++c1)
                    {
                        for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                        {
                            int c3 = nvthis - c1 - c2;
                            tmpids[0] = c1;
                            tmpids[1] = c2;
                            tmpids[2] = c3;
                            leftinds[i].push_back(tmpids);
                        }
                    }
                }
                /* combine left and right inds, then loop over contraciton inds */
                for (int i = 0; i < rightinds.size(); ++i)
                {
                    std::vector<int> target_ind_ref(3 + 3 * sizel);
                    std::vector<int> left_ind_ref(3 * sizel);
                    std::array<int, 4> right_ind_ref;
                    //left_ind_ref[0] = idl;
                    right_ind_ref[0] = br.first(0);
                    right_ind_ref[1] = rightinds[i][0];
                    right_ind_ref[2] = rightinds[i][1];
                    right_ind_ref[3] = rightinds[i][2];
                    // target_ind_ref[0] = idm;
                    target_ind_ref[0 + 3 * sizel] = rightinds[i][0];
                    target_ind_ref[1 + 3 * sizel] = rightinds[i][1];
                    target_ind_ref[2 + 3 * sizel] = rightinds[i][2];
                    // loop over left indices
                    std::vector<int> llooperlim(sizel);
                    for (int ll = 0; ll < sizel; ++ll)
                        llooperlim[ll] = leftinds[ll].size();
                    looper loop_l_ind(llooperlim);
                    for (int j = 0; j < loop_l_ind.size(); ++j)
                    {
                        int tshift = 0;
                        for (int ll = 0; ll < leftinds.size(); ++ll)
                        {
                            for (int kk = 0; kk < 3; ++kk)
                            {
                                target_ind_ref[tshift] = leftinds[ll][loop_l_ind.loop[ll]][kk];
                                left_ind_ref[tshift] = target_ind_ref[tshift];
                                ++tshift;
                            }
                        }
                        non_ele_id_type bi_id(
                            tid, b, mu, nv, target_ind_ref,
                            &non_ele_id_canonical_cache);
                        auto [moment_it, moment_inserted] =
                            bi_id_map_this.emplace(std::move(bi_id),
                                                   alpha_moments_count_this);
                        const int moment_id =
                            moment_inserted ? alpha_moments_count_this++
                                            : moment_it->second;
                        if (moment_inserted)
                        {
                            root_moment_ids[tid].push_back(moment_id);
                            moment_creator[moment_id] = conn;
                        }
                        // Rules are generated when the moment is new, or when
                        // it was created by a different decomposition (shared
                        // moments, see case 1 note); within one connection
                        // only the first occurrence generates rules.
                        if ((moment_inserted ||
                             moment_creator[moment_id] != conn) &&
                            ruled_this_conn.insert(moment_id).second)
                        {
                            std::array<int, 3> times_ind;
                            // times_ind[3] = alpha_moments_count_this - 1;
                            times_ind[2] = moment_id;
                            // add mapping
                            mid_tid_mapping[moment_id] = tid;
                            // now loop over contraction inds
                            std::vector<int> loopc_limit(contractinds.size());
                            for (int ll = 0; ll < contractinds.size(); ++ll)
                                loopc_limit[ll] = contractinds[ll].size();
                            looper loop_c_ind(loopc_limit);
                            for (int ll = 0; ll < loop_c_ind.size(); ++ll)
                            {
                                int times = tmulti;
                                std::vector<int> tmplate_c_inds(3 * sizel);
                                int ct = 0;
                                auto leftind = left_ind_ref;
                                auto rightind = right_ind_ref;
                                for (int ll = 0; ll < contractinds.size(); ++ll)
                                {
                                    int v1, v2, v3;
                                    v1 = contractinds[ll][loop_c_ind.loop[ll]][0];
                                    v2 = contractinds[ll][loop_c_ind.loop[ll]][1];
                                    v3 = contractinds[ll][loop_c_ind.loop[ll]][2];
                                    leftind[ct] += v1;
                                    ++ct;
                                    leftind[ct] += v2;
                                    ++ct;
                                    leftind[ct] += v3;
                                    ++ct;
                                    rightind[1] += v1;
                                    rightind[2] += v2;
                                    rightind[3] += v3;
                                    times /= Combinatorics::factorial_int(v1) *
                                             Combinatorics::factorial_int(v2) *
                                             Combinatorics::factorial_int(v3);
                                }
                                non_ele_id_type tmp_id_cleft(lid, bl, mul, nvl, leftind);
                                auto right_component = alpha_index_basic_map.find(rightind);
                                if (right_component == alpha_index_basic_map.end())
                                {
                                    GPERROR("Missing right basic component in mtmbasis generation");
                                }
                                auto left_component = bi_id_map_this.find(tmp_id_cleft);
                                if (left_component == bi_id_map_this.end())
                                {
                                    GPERROR("Missing left non-MT component in mtmbasis generation");
                                }
                                times_ind[0] = right_component->second;
                                times_ind[1] = left_component->second;
                                // times_ind[2] = times;
                                auto [rule_it, rule_inserted] =
                                    alpha_times_flag_this.emplace(
                                        times_ind, alpha_index_times_this.size());
                                if (rule_inserted) {
                                    alpha_index_times_count_this += 1;
                                    alpha_index_times_this.push_back({times_ind[0], times_ind[1], times, times_ind[2]});
                                } else {
                                    int id = rule_it->second;
                                    alpha_index_times_this[id][2] += times;
                                }
                                loop_c_ind.next();
                            }
                        }
                        loop_l_ind.next();
                    }
                }
                // if scalar
                if (b.is_scalar()) {
                    // DEBUG_SCALAR(b)
                    // _DEBUG_OUTPUT_(alpha_index_times_this.back().back());
                    if (alpha_gen_ids.count(b.tensor_id) || add_new_scalar_to_pot) {
                        alpha_moment_mapping_this.push_back(alpha_index_times_this.back().back());
                    } 
                }
            }
            // case 4: no mt
            else
            {
                Eigen::MatrixXi contract;
                b = restore_b(b, bl, br);
                contract = b.second.block(0, bl.size(), bl.size(), br.size());

                int sizel = bl.size();
                int sizer = br.size();
                int tmult = 1;
                int nleft = contract.rows(); // == sizel
                int nright = contract.cols(); // == sizer

                /**
                 * @brief params
                 * let t -> a b
                 *       a'  | m
                 * t ->  -------
                 *       m^T | b' 
                 * 
                 * mul:  mu of a
                 * mur:  mu of b
                 * nvl:  nv of a (dia only)
                 * nvr:  nv of b (dia only)
                 * mu:   mu of t
                 * nv:   nv of t
                 * lnvl: nv of a'
                 * lnvr: nv of b'
                 * nleft: num of contractions taken on left tensor
                 * nright: as above
                 * left_neq_inds: for each contraction, inequvalent indices of tensor comps
                 */

                std::vector<int> mul(nleft, 0), mur(nright, 0), nvl(nleft, 0), nvr(nright, 0), mu(nleft + nright, 0), nv(nleft + nright, 0);
                std::vector<int> lnvl(nleft, 0), lnvr(nright, 0);

                std::vector<std::vector<std::array<int, 3>>> left_neq_inds(nleft);
                std::vector<std::vector<std::array<int, 3>>> right_neq_inds(nright);

                for (int i = 0; i < b.size(); ++i)
                {
                    mu[i] = b.first(i);
                    nv[i] = b.second(i, i);
                }

                for (int i = 0; i < nleft; ++i)
                {
                    nvl[i] = bl.second(i, i);
                    mul[i] = bl.first[i];
                    lnvl[i] = nvl[i] - contract.row(i).sum();
                }
                for (int i = 0; i < nright; ++i)
                {
                    nvr[i] = br.second(i, i);
                    mur[i] = br.first[i];
                    lnvr[i] = nvr[i] - contract.col(i).sum();
                }
                // multiplier for times
                for (int i = 0; i < sizel; ++i)
                {
                    for (int j = 0; j < sizer; ++j)
                        tmult *= Combinatorics::factorial_int(contract(i, j));
                }
                // build left inds
                for (int i = 0; i < left_neq_inds.size(); ++i)
                {
                    std::array<int, 3> tmpids; // sum of elements equals to nvi
                    int nvthis = lnvl[i];
                    int nele = (nvthis + 1) * (nvthis + 2) / 2;
                    left_neq_inds[i].reserve(nele);
                    for (int c1 = 0; c1 <= nvthis; ++c1)
                    {
                        for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                        {
                            int c3 = nvthis - c1 - c2;
                            tmpids[0] = c1;
                            tmpids[1] = c2;
                            tmpids[2] = c3;
                            left_neq_inds[i].push_back(tmpids);
                        }
                    }
                }
                // build right inds
                for (int i = 0; i < right_neq_inds.size(); ++i)
                {
                    std::array<int, 3> tmpids; // sum of elements equals to nvi
                    int nvthis = lnvr[i];
                    int nele = (nvthis + 1) * (nvthis + 2) / 2;
                    right_neq_inds[i].reserve(nele);
                    for (int c1 = 0; c1 <= nvthis; ++c1)
                    {
                        for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                        {
                            int c3 = nvthis - c1 - c2;
                            tmpids[0] = c1;
                            tmpids[1] = c2;
                            tmpids[2] = c3;
                            right_neq_inds[i].push_back(tmpids);
                        }
                    }
                }
                // build contraction inds
                std::vector<std::vector<std::array<int, 3>>> contraction_inds;
                contraction_inds.resize(nleft * nright);
                for (int i = 0; i < nleft; ++i)
                {
                    for (int j = 0; j < nright; ++j)
                    {
                        std::array<int, 3> tmpids;
                        int nvthis = contract(i, j);
                        int nele = (nvthis + 1) * (nvthis + 2) / 2;
                        contraction_inds[i * nright + j].reserve(nele);
                        for (int c1 = 0; c1 <= nvthis; ++c1)
                        {
                            for (int c2 = 0; c2 <= nvthis - c1; ++c2)
                            {
                                int c3 = nvthis - c1 - c2;
                                tmpids[0] = c1;
                                tmpids[1] = c2;
                                tmpids[2] = c3;
                                contraction_inds[i * nright + j].push_back(tmpids);
                            }
                        }
                    }
                }
                // do contraction
                // step1: build non-equal target indies by combining two residual indices of left and right
                std::vector<int> l_looperlim(left_neq_inds.size()), r_looperlim(right_neq_inds.size());
                std::vector<int> target_inds_ref(nleft * 3 + nright * 3, 0);
                std::vector<int> left_ind_ref(nleft * 3), right_ind_ref(nright * 3);
                // target_inds_ref[0] = idm;
                // left_ind_ref[0] = idl;
                // right_ind_ref[0] = idr;
                for (int i = 0; i < left_neq_inds.size(); ++i)
                    l_looperlim[i] = left_neq_inds[i].size();
                for (int i = 0; i < right_neq_inds.size(); ++i)
                    r_looperlim[i] = right_neq_inds[i].size();
                // firstly loop over left inds
                looper loop_over_left_inds(l_looperlim);
                for (int i = 0; i < loop_over_left_inds.size(); ++i)
                {
                    auto current_inds = target_inds_ref;
                    int lshift = 0;
                    for (int ll = 0; ll < nleft; ++ll)
                    {
                        for (int ss = 0; ss < 3; ++ss)
                        {
                            current_inds[lshift] = left_neq_inds[ll][loop_over_left_inds.loop[ll]][ss];
                            left_ind_ref[lshift] = current_inds[lshift];
                            lshift++;
                        }
                    }
                    // then loop over right inds inside loop in left inds
                    looper loop_over_right_inds(r_looperlim);
                    for (int j = 0; j < loop_over_right_inds.size(); ++j)
                    {
                        int rshift = lshift;
                        for (int ll = 0; ll < nright; ++ll)
                        {
                            for (int ss = 0; ss < 3; ++ss)
                            {
                                current_inds[rshift] = right_neq_inds[ll][loop_over_right_inds.loop[ll]][ss];
                                right_ind_ref[rshift - lshift] = current_inds[rshift];
                                rshift++;
                            }
                        }
                        non_ele_id_type bi_id(
                            tid, b, mu, nv, current_inds,
                            &non_ele_id_canonical_cache);
                        auto [moment_it, moment_inserted] =
                            bi_id_map_this.emplace(std::move(bi_id),
                                                   alpha_moments_count_this);
                        const int moment_id =
                            moment_inserted ? alpha_moments_count_this++
                                            : moment_it->second;
                        if (moment_inserted)
                        {
                            root_moment_ids[tid].push_back(moment_id);
                            moment_creator[moment_id] = conn;
                        }
                        // Rules are generated when the moment is new, or when
                        // it was created by a different decomposition (shared
                        // moments, see case 1 note); within one connection
                        // only the first occurrence generates rules.
                        if ((moment_inserted ||
                             moment_creator[moment_id] != conn) &&
                            ruled_this_conn.insert(moment_id).second)
                        {
                            std::vector<int> curr_contraction_inds(3 * contract.size());
                            std::vector<int> c_loop_limit(contract.size(), 0);
                            for (int ll = 0; ll < nleft; ++ll)
                            {
                                for (int kk = 0; kk < nright; ++kk)
                                    c_loop_limit[ll * nright + kk] = contraction_inds[ll * nright + kk].size();
                            }
                            // loop over contractions
                            looper c_loop(c_loop_limit);
                            for (int ii = 0; ii < c_loop.size(); ++ii)
                            {
                                int ct = 0;
                                int times = tmult;
                                std::array<int, 3> times_ind; // left right times result
                                for (int ll = 0; ll < contraction_inds.size(); ++ll)
                                {
                                    for (int kk = 0; kk < 3; ++kk)
                                    {
                                        curr_contraction_inds[ct] = contraction_inds[ll][c_loop.loop[ll]][kk];
                                        times /= Combinatorics::factorial_int(curr_contraction_inds[ct]);
                                        ++ct;
                                    }
                                }
                                // times_ind[2] = times;
                                // times_ind[3] = alpha_moments_count_this - 1;
                                times_ind[2] = moment_id;
                                // add mapping
                                mid_tid_mapping[moment_id] = tid;
                                // build left
                                auto left_ind = left_ind_ref;
                                ct = 0;
                                for (int ll = 0; ll < nleft; ++ll)
                                {
                                    for (int ss = 0; ss < nright; ++ss)
                                    {
                                        for (int kk = 0; kk < 3; ++kk)
                                        {
                                            left_ind[ll * 3 + kk] += curr_contraction_inds[ct];
                                            ++ct;
                                        }
                                    }
                                }
                                non_ele_id_type leftid(lid, bl, mul, nvl, left_ind);
                                auto left_component = bi_id_map_this.find(leftid);
                                if (left_component == bi_id_map_this.end())
                                {
                                    GPERROR("Missing left non-MT component in mtmbasis generation");
                                }
                                times_ind[0] = left_component->second;
                                // build right
                                auto right_ind = right_ind_ref;
                                ct = 0;
                                for (int ll = 0; ll < nright; ++ll)
                                {
                                    for (int ss = 0; ss < nleft; ++ss)
                                    {
                                        for (int kk = 0; kk < 3; ++kk)
                                        {
                                            right_ind[ll * 3 + kk] += curr_contraction_inds[ss * nright * 3 + ll * 3 + kk];
                                            ++ct;
                                        }
                                    }
                                }
                                non_ele_id_type rightid(rid, br, mur, nvr, right_ind);
                                auto right_component = bi_id_map_this.find(rightid);
                                if (right_component == bi_id_map_this.end())
                                {
                                    GPERROR("Missing right non-MT component in mtmbasis generation");
                                }
                                times_ind[1] = right_component->second;
                                auto [rule_it, rule_inserted] =
                                    alpha_times_flag_this.emplace(
                                        times_ind, alpha_index_times_this.size());
                                if (rule_inserted) {
                                    alpha_index_times_count_this += 1;
                                    alpha_index_times_this.push_back({times_ind[0], times_ind[1], times, times_ind[2]});
                                } else {
                                    int id = rule_it->second;
                                    alpha_index_times_this[id][2] += times;
                                    /**
                                     * Ignore this !!!
                                     * Tensor a
                                     * mu: 0 0 0 0 
                                     * alpha: 
                                     *     0 1 0 2
                                     *     1 0 2 0
                                     *     0 2 0 4
                                     *     2 0 4 0
                                     * Tensor b
                                     * mu: 0 0 0 0
                                     * alpha:
                                     *     0 1 1 1
                                     *     1 0 1 1
                                     *     1 1 0 4
                                     *     1 1 4 0
                                     * and a -> c + d, b -> c + d
                                     * Tensor c
                                     * mu: 0 0
                                     * alpha:
                                     *     2 1 
                                     *     1 2
                                     * Tensor d
                                     * mu: 0 0
                                     * alpha:
                                     *     2 4
                                     *     4 2
                                     * a and b decomposition only differs in contraction matrix,(in this case times rule will be rused : modified: not true)
                                     */
                                    /**
                                     * This problem raise from here:
                                     * i.e 
                                     * c =
                                     * 0   0 0 1 1    
                                     * 0   0 0 1 1     
                                     * 0   1 1 0 0 
                                     * 0   1 1 0 0   
                                     * to 
                                     * a = 
                                     * 0   2 0
                                     * 0   0 2
                                     * and 
                                     * b = 
                                     * 0   2 0
                                     * 0   0 2
                                     * Thus:
                                     * c = a_{ijkl} b_{ikjl}
                                     * For
                                     * a_{0,1,1,0} * b_{0,1,1,0} and a_{1,0,0,1} * b_{1,0,0,1} 
                                     * both result in
                                     * {0, 1, 1, 0, 1, 1}
                                     * dups raise from here.
                                     * Solution:
                                     * for {id1, id2, c, idthis}, check if exist {id1, id2, c', idthis}, and 
                                     * modify existed one as 
                                     * {id1, id2, c + c', idthis}
                                     */
                                }
                                c_loop.next();
                            }
                        }
                        loop_over_right_inds.next();
                    }
                    loop_over_left_inds.next();
                }
                // if scalar
                if (b.is_scalar()) {
                    // DEBUG_SCALAR(b)
                    // _DEBUG_OUTPUT_(alpha_index_times_this.back().back());
                    if (alpha_gen_ids.count(b.tensor_id) || add_new_scalar_to_pot) {
                        alpha_moment_mapping_this.push_back(alpha_index_times_this.back().back());
                    } 
                }
            }
        if ((int)alpha_index_times_this.size() == rules_before) {
            GPERROR("A connection produced no rules in mtmbasis generation");
        }
        conn_rule_range[conn] = {rules_before, (int)alpha_index_times_this.size()};
    }
}
