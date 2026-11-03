// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "fixtures/mtp_linear_efs.h"
#include "potentials/mtp_gpu_adapter.h"
#include "train/linear_efs_observations.h"
#include <algorithm>
#include <numeric>

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); mpi.InitComm(MPI_COMM_SELF);
    const int result = Catch::Session().run(argc, argv);
    MPI_Finalize(); return result;
}
#endif

TEST_CASE("Linear EFS options default to a valid CPU backend", "[training][linear-efs]") {
    Training::LinearEFSOptions options;
    REQUIRE(options.device == "cpu");
    REQUIRE(options.gpu_batch_atoms > 0);
    REQUIRE(options.gpu_batch_structures > 0);
    REQUIRE_NOTHROW(options.validate());
}

#ifdef IMRMLP_GPU
TEST_CASE("GPU linear observation rows contract to CPU energy forces and full virial",
          "[gpu][training][linear-efs][observations]") {
    int devices = 0; std::string error;
    const bool available = MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(
        &devices, &error) && devices > 0;
    if (std::getenv("IMRMLP_REQUIRE_GPU")) { INFO(error); REQUIRE(available); }
    if (!available) SKIP("No GPU: " << error);

    const int variant = GENERATE(1,2,3,4);
    const bool masked = GENERATE(false,true);
    CAPTURE(variant, masked);
    Potentials::PotMTP model;
    MTPLinearEFSFixture::load(model,variant);
    model.train_init(); model.set_inited();
    auto atoms = MTPLinearEFSFixture::frames(model.species_count);
    std::vector<Neighborhoods> neighbors;
    auto reference = atoms;
    for (size_t k=0; k<atoms.size(); ++k) {
        neighbors.emplace_back(atoms[k],model.cutoff);
        reference[k].resetEFS(); model.getEFS(reference[k],neighbors.back());
    }
    std::vector<std::array<bool,3>> requested(atoms.size(),{true,true,true});
    if (masked) for (size_t k=0; k<atoms.size(); ++k)
        requested[k] = {k%3!=1,k%3!=2,k%3!=0};

    // Preserve the complete model contraction while checking reordered and
    // repeated native columns. Split the repeated coefficient equally.
    std::vector<int> columns(model.lin_coeff_size);
    std::iota(columns.begin(),columns.end(),0);
    std::reverse(columns.begin(),columns.end());
    columns.push_back(columns.front());
    std::vector<std::array<std::vector<int>,3>> seen(atoms.size());
    for (size_t k=0; k<atoms.size(); ++k) {
        seen[k][0].assign(1,0);
        seen[k][1].assign(3*atoms[k].size(),0);
        seen[k][2].assign(9,0);
    }
    Training::LinearEFSOptions options;
    options.device = MLIPCore::GpuDirect::compiled_backend();
    options.gpu_id = 0;
    options.gpu_batch_atoms = 32;
    options.gpu_batch_structures = 2;
    options.gpu_memory_bytes = size_t{8} << 20;
    options.gpu_workspace_bytes = size_t{1} << 20;
    Training::LinearEFSStatistics stats;
    Training::visit_gpu_linear_efs(model,atoms,requested,columns,options,stats,{},
        [&](const Training::LinearEFSObservationBlock &block) {
            REQUIRE(block.native_columns.size() == columns.size());
            REQUIRE(std::equal(columns.begin(),columns.end(),block.native_columns.begin()));
            REQUIRE(block.values.size() == block.rows.size()*columns.size());
            for (size_t r=0; r<block.rows.size(); ++r) {
                const auto &row = block.rows[r];
                REQUIRE(row.configuration < atoms.size());
                REQUIRE(row.channel >= 0);
                REQUIRE(row.channel < 3);
                REQUIRE(requested[row.configuration][row.channel]);
                const auto &cpu = reference[row.configuration];
                int index = 0; double expected = 0.;
                if (row.channel == 0) {
                    REQUIRE(row.atom == -1); REQUIRE(row.component == -1);
                    expected = cpu.energy_;
                } else if (row.channel == 1) {
                    REQUIRE(row.atom >= 0); REQUIRE(row.atom < cpu.size());
                    REQUIRE(row.component >= 0); REQUIRE(row.component < 3);
                    index = 3*row.atom+row.component;
                    expected = cpu.forces_[row.atom][row.component];
                } else {
                    REQUIRE(row.atom == -1);
                    REQUIRE(row.component >= 0); REQUIRE(row.component < 9);
                    index = row.component;
                    expected = cpu.stress_(index/3,index%3);
                }
                REQUIRE(++seen[row.configuration][row.channel][index] == 1);
                const auto values = block.values.subspan(r*columns.size(),columns.size());
                double actual = 0.;
                for (size_t c=0; c<columns.size(); ++c) {
                    const auto repeats = std::count(columns.begin(),columns.end(),columns[c]);
                    actual += values[c]*model.get_linear_coeff(columns[c])/repeats;
                }
                REQUIRE(actual == Catch::Approx(expected).margin(2e-8+2e-9*std::abs(expected)));
            }
        },&neighbors);
    // Exact coverage catches omitted rows as well as wrong row identities.
    for (size_t k=0; k<atoms.size(); ++k) for (int channel=0; channel<3; ++channel)
        for (int count : seen[k][channel]) REQUIRE(count == int(requested[k][channel]));
    REQUIRE(stats.batches > 1);
    REQUIRE(stats.observation_blocks > 0);
    REQUIRE(stats.tracked_device_bytes > 0);
    REQUIRE(stats.observation_workspace_bytes <= options.gpu_workspace_bytes);
}
#endif
