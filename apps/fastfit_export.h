// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "fastfit_cost.h"
#include "train/efs_objective.h"
#include <filesystem>
#include <fstream>

namespace func_fastfit::cost {
struct Metrics { std::array<double, 4> train{}, valid{}; };

// Called on rank 0 after every candidate has been serialized, reloaded and
// checked against the full training/validation data. Publish a fresh directory
// only after all its files have been written; never overwrite an existing run.
inline void export_bundle(const std::string& directory, const std::string& parent_bytes,
        const Training::EFSObjective& objective, const EFSFitWeights& weights,
        const std::vector<Entry>& entries, const std::vector<std::unique_ptr<PotMTP>>& models,
        const std::vector<Measurement>& measurements, const std::vector<Metrics>& metrics,
        const Workload& workload, double limit, int mode, int best, int seed) {
    namespace fs = std::filesystem;
    if (entries.size()!=models.size() || entries.size()!=measurements.size() || entries.size()!=metrics.size())
        throw std::runtime_error("Incomplete Pareto export inputs");
    const fs::path target(directory), staging(directory + ".tmp");
    if (fs::exists(target) || !fs::create_directory(staging))
        throw std::runtime_error("Pareto export requires a fresh directory and .tmp path");
    try {
        auto write = [&](const fs::path& name, auto&& action) {
            std::ofstream out(staging/name, std::ios::binary);
            if (!out) throw std::runtime_error("Cannot open Pareto export file: " + name.string());
            out << std::setprecision(17);
            action(out); out.close();
            if (!out) throw std::runtime_error("Cannot write Pareto export file: " + name.string());
        };
        auto eligible=entries;
        if (mode==5) eligible[0].valid_loss=INFINITY;
        std::vector<double> estimated, measured;
        for (std::size_t i=0; i<entries.size(); ++i) {
            estimated.push_back(entries[i].relative);
            measured.push_back(measurements[i].total.median);
        }
        const auto proxy=pareto_front(eligible,estimated,limit);
        const auto actual=pareto_front(eligible,measured,limit);
        write("parent.mtp",[&](auto& out) {out << parent_bytes;});
        write("objective.tsv",[&](auto& out) {objective.write(out,weights);});
        write("rounds.tsv",[&](auto& out) {
            out << "candidate\tround\tkernel_ns_atom\ttotal_ns_atom\n";
            for (std::size_t i=0; i<entries.size(); ++i) {
                const auto& m=measurements[i];
                if (m.kernel_rounds.size()!=m.total_rounds.size() || m.total_rounds.size()<3)
                    throw std::runtime_error("Missing raw Pareto timing rounds");
                for (std::size_t round=0; round<m.total_rounds.size(); ++round)
                    out << i << '\t' << round << '\t' << m.kernel_rounds[round] << '\t' << m.total_rounds[round] << '\n';
            }
        });
        for (std::size_t i=0; i<models.size(); ++i)
            write("candidate-"+std::to_string(i)+".mtp",[&](auto& out) {models[i]->Save(out);});
        write("manifest.tsv",[&](auto& out) {
            out << "# schema\tfastfit-pareto-v1\n# scope\tverified measured shortlist, not a global Pareto front\n"
                << "# cost_metrics\tefs-ops-v1 estimate and measured total median; no significance claim\n"
                << "# parent\tparent.mtp is an exact copy of the input model\n"
                << "# max_valid_loss\t" << limit << "\n# seed\t" << seed
                << "\n# backend\tPotMTP::getEFSOMP; total includes neighborhoods\n"
                << "# omp_threads\t" << omp_num_threads << "\n# sampled_global_validation_ids\t";
            for (int id:workload.global_ids) out << id << ',';
            out << "\n# active_neighbors_per_atom\t" << workload.active
                << "\n# stored_neighbors_per_atom\t" << workload.stored << '\n';
            out << "index\tfile\tbaseline\tselected\tfeasible\tproxy_pareto\tmeasured_pareto\ttrain_mean_loss\tvalid_mean_loss"
                << "\tcost_relative\ttotal_ns_atom\ttotal_q25\ttotal_q75\tscalars\tbasics\tproducts\tchannels"
                << "\ttrain_E_RMSE\ttrain_F_vector_RMSE\ttrain_V_builtin_RMSE"
                << "\tvalid_E_RMSE\tvalid_F_vector_RMSE\tvalid_V_builtin_RMSE\toriginal_scalar_ids\n";
            for (std::size_t i=0; i<entries.size(); ++i) {
                const auto& e=entries[i]; const auto& t=measurements[i].total;
                out << i << "\tcandidate-" << i << ".mtp\t" << (i==0) << '\t' << (int(i)==best)
                    << '\t' << feasible(e.valid_loss,limit)
                    << '\t' << (std::find(proxy.begin(),proxy.end(),i)!=proxy.end())
                    << '\t' << (std::find(actual.begin(),actual.end(),i)!=actual.end())
                    << '\t' << e.train_loss << '\t' << e.valid_loss << '\t' << e.relative
                    << '\t' << t.median << '\t' << t.low << '\t' << t.high
                    << '\t' << e.stats.scalars << '\t' << e.stats.basics << '\t' << e.stats.products << '\t' << e.stats.channels;
                for (int k=0;k<3;++k) out << '\t' << metrics[i].train[k];
                for (int k=0;k<3;++k) out << '\t' << metrics[i].valid[k];
                out << '\t'; for (int id:e.ids) out << id << ','; out << '\n';
            }
        });
        fs::rename(staging,target);
    } catch (...) {
        std::error_code ignored; fs::remove_all(staging,ignored);
        throw;
    }
}
} // namespace func_fastfit::cost
