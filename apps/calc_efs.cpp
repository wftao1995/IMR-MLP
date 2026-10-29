// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "runtime/mtp/gpu-direct/gpu_backend.h"
#include <exception>
#include "potentials/potential_wrapper.h"
#include "io/gpio.h"

#ifdef IMRMLP_GPU
#include "potentials/mtp_gpu_adapter.h"
#endif

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <numeric>

using namespace std;
using namespace Eigen;

#ifdef USE_OMP
#include <omp.h>
#endif

#ifdef IMRMLP_GPU
namespace {

size_t estimate_gpu_inference_bytes(int active_moment_slots,
                                     size_t atoms, size_t neighbors,
                                     bool energy_only)
{
    const long double moment_arrays = energy_only ? 1.0L : 2.0L;
    const long double moment_bytes = moment_arrays * atoms *
        active_moment_slots * sizeof(double);
    const long double atom_bytes = atoms * (
        3.0L * sizeof(int) + // centers, offsets, types
        (energy_only ? 4.0L : 13.0L) * sizeof(double)); // x/E or x/F/E/virial
    const long double neighbor_bytes = neighbors *
        (sizeof(int) + 3.0L * sizeof(double));
    const long double pipeline_input_bytes = atoms *
        (3.0L * sizeof(int) + 3.0L * sizeof(double)) + neighbor_bytes;
    const long double estimate =
        (moment_bytes + atom_bytes + neighbor_bytes + pipeline_input_bytes +
         2.0L * 1024.0L * 1024.0L) * 1.10L;
    if (estimate >= std::numeric_limits<size_t>::max()) {
        return std::numeric_limits<size_t>::max();
    }
    return static_cast<size_t>(std::ceil(estimate));
}

} // namespace
#endif

int calc_efs(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    GPSettings settings;
    std::vector<Atoms> atoms;
    string settingfile = "input.ini";
    bool show_progress = false;
    string informat = "";
    string infile = "POSCAR";
    string outfile = "output-efs.cfg";
    string outformat = "";
    string indexfile = "index.map";
    string pfile = "potential.mtp";
    string ptype = "mtp";
    string device = "cpu";
    int gpu_id = -1;
    bool no_site_energy = false;
    bool energy_only = false;
    int gpu_batch_atoms = 0;
    int gpu_batch_structures = 0;
    double gpu_memory_fraction = 0.75;
    bool gpu_reuse_topology = false;
    // cmdopts.addop(settingfile, "setting", "", "setting file", true)
    cmdopts.addop(infile, "input", "in", "Input structure file path", true)
           .addop(outfile, "output", "out", "Output structure file path", true)
           .addop(informat, "input-format", "if", "Input file format (e.g., POSCAR, xyz, cif)")
           .addop(outformat, "output-format", "of", "Output file format (e.g., POSCAR, xyz, cif)")
           .addop(indexfile, "index", "id", "Index file mapping atom types to indices")
           .addop(pfile, "potfile", "pot", "Potential parameter file path (alternative: use 'pot:' tag in settings)")
           .addop(ptype, "pottype", "pt", "Potential type identifier (alternative: use 'pot:' tag in settings)")
           .addop(device, "device", "", "Evaluation device: cpu | cuda | hip (dcu)")
           .addop(gpu_id, "gpu-id", "", "GPU device id; -1 uses the current GPU device")
           .addop(gpu_batch_atoms, "gpu-batch-atoms", "", "Maximum atoms per GPU chunk; zero selects automatically")
           .addop(gpu_batch_structures, "gpu-batch-structures", "", "Maximum structures per GPU chunk; zero selects automatically")
           .addop(gpu_memory_fraction, "gpu-memory-fraction", "", "Fraction of currently free GPU memory available to a chunk")
           .addop(gpu_reuse_topology, "gpu-reuse-topology", "", "Reuse exactly compatible consecutive single-structure trajectory topologies", false, 0)
           .addop(no_site_energy, "no-site-energy", "ns", "Exclude site energy from output", false, 0)
           .addop(energy_only, "energy-only", "eno", "Output only energy information (excludes forces, stress, site energies)", false, 0)
           .addop(settingfile, "settings", "", "Main configuration settings file")
           .addop(show_progress, "show-progress", "sp", "Report completed structure counts during calculation", false, 0)
           .attach_settings("pot:", "Potential settings",
               [](std::ostream &os) { Potentials::potential_wrapper().ListSettings(os, "pot:"); })
           .add_request("input")
           .add_request("output")
           .set_helponexit(false)
           .phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        exit(0);
    }
    if (cmdopts.was_provided("settings") || GPIO::file_exist(settingfile)) {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
        apply_settings_to_cmdopts(cmdopts, settings);
    }
    std::transform(device.begin(), device.end(), device.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    device = MLIPCore::GpuDirect::canonical_device(device);
    if (!MLIPCore::GpuDirect::valid_device(device)) {
        GPERROR("--device must be cpu, cuda, hip, or dcu");
    }
    if (!MLIPCore::GpuDirect::device_supported(device)) {
        GPERROR("--device=" + device + " is unavailable in this " +
                MLIPCore::GpuDirect::compiled_backend() + " build");
    }
    if (gpu_batch_atoms < 0 || gpu_batch_structures < 0) {
        GPERROR("GPU batch limits must be zero (automatic) or positive");
    }
    if (!(gpu_memory_fraction > 0.0 && gpu_memory_fraction <= 1.0)) {
        GPERROR("--gpu-memory-fraction must be in (0, 1]");
    }
    if (gpu_reuse_topology && gpu_batch_structures > 1) {
        GPERROR("--gpu-reuse-topology requires --gpu-batch-structures=1 or zero");
    }

    if (GPIO::file_exist(indexfile)) {
        iobj.readmap(indexfile); // should be with atom radii
    } else {
        cerr << "ERROR: indexfile not found\n";
    }
    // read atoms
    Message("Reading trajectory file " + infile);
    if (!iobj.read(atoms, infile, informat))
        GPERROR("Failed to read structures from file: " + infile);
    if (atoms.empty()) {
        GPERROR("no structures read from file: " + infile);
    }

    size_t ndone = 0;
    const size_t report_every = 1 + (atoms.size() - 1) / 100;

#ifdef IMRMLP_GPU
    if (device != "cpu") {
        Message("GPU backend: " + device);
        Potentials::potential_wrapper wrapper(
            pfile, ptype, settings.ExtractSubSettings("pot"));
        auto *mtp = dynamic_cast<Potentials::PotMTP *>(wrapper.pot);
        if (mtp == nullptr) {
            GPERROR("GPU calc_efs requires a supported PotMTP or PotMTP-M potential");
        }

        std::vector<Neighborhoods> neighborhoods;
        neighborhoods.reserve(atoms.size());
        std::vector<size_t> neighbor_counts(atoms.size(), 0);
        for (Atoms &structure : atoms) {
            structure.wrap();
            neighborhoods.emplace_back(structure, mtp->cutoff);
            const size_t structure_index = neighborhoods.size() - 1;
            for (int i = 0; i < neighborhoods.back().size(); ++i) {
                neighbor_counts[structure_index] += static_cast<size_t>(
                    neighborhoods.back()[i].nbhcount_);
            }
        }

        std::string gpu_error;
        Potentials::MTPGpuEvaluator evaluator;
        Potentials::MTPGpuBatch batch;
        if (!evaluator.initialize(*mtp, gpu_id, &gpu_error)) {
            GPERROR("Unable to initialize GPU MTP evaluation: " + gpu_error);
        }
        const auto potential_stats = evaluator.context().potential_stats();
        Message("GPU MTP moment workspace: declared_slots=" +
                std::to_string(potential_stats.declared_moment_slots) +
                ", active_slots=" +
                std::to_string(potential_stats.active_moment_slots) +
                ", block_groups_forward=" +
                std::to_string(potential_stats.forward_block_groups) +
                ", block_groups_reverse=" +
                std::to_string(potential_stats.reverse_block_groups));

        size_t free_memory = 0;
        size_t total_memory = 0;
        if (!evaluator.context().memory_info(
                &free_memory, &total_memory, &gpu_error)) {
            GPERROR("Unable to query GPU memory for calc_efs: " + gpu_error);
        }
        const size_t memory_budget = static_cast<size_t>(
            static_cast<long double>(free_memory) * gpu_memory_fraction);
        const int structure_limit = gpu_reuse_topology ? 1 : gpu_batch_structures;
        std::vector<std::vector<int>> chunks;
        std::vector<int> current;
        size_t current_atoms = 0;
        size_t current_neighbors = 0, current_reciprocal = 0;
        const auto exceeds_limit = [&](size_t atom_count,
                                       size_t neighbor_count,
                                       size_t structure_count, size_t reciprocal) {
            return (gpu_batch_atoms > 0 &&
                    atom_count > static_cast<size_t>(gpu_batch_atoms)) ||
                   (structure_limit > 0 &&
                    structure_count > static_cast<size_t>(structure_limit)) ||
                   static_cast<long double>(estimate_gpu_inference_bytes(
                       potential_stats.active_moment_slots,
                       atom_count, neighbor_count, energy_only)) + reciprocal >
                       memory_budget;
        };
        for (size_t i = 0; i < atoms.size(); ++i) {
            const size_t reciprocal = evaluator.reciprocal_bytes(atoms[i]);
            const size_t structure_atoms = atoms[i].size();
            const size_t structure_neighbors = neighbor_counts[i];
            if (!current.empty() && exceeds_limit(
                    current_atoms + structure_atoms,
                    current_neighbors + structure_neighbors,
                    current.size() + 1, reciprocal > std::numeric_limits<size_t>::max()-current_reciprocal
                        ? std::numeric_limits<size_t>::max() : current_reciprocal + reciprocal)) {
                chunks.push_back(std::move(current));
                current.clear();
                current_atoms = 0;
                current_neighbors = 0; current_reciprocal = 0;
            }
            if (exceeds_limit(structure_atoms, structure_neighbors, 1, reciprocal)) {
                GPERROR("GPU calc_efs structure " + std::to_string(i) +
                    " cannot fit one chunk: atoms=" +
                    std::to_string(structure_atoms) +
                    ", directed_neighbors=" +
                    std::to_string(structure_neighbors) +
                    ", estimated_bytes=" +
                    std::to_string(estimate_gpu_inference_bytes(
                        potential_stats.active_moment_slots,
                        structure_atoms, structure_neighbors, energy_only) + reciprocal) +
                    ", budget_bytes=" + std::to_string(memory_budget));
            }
            current.push_back(static_cast<int>(i));
            current_atoms += structure_atoms;
            current_neighbors += structure_neighbors; current_reciprocal += reciprocal;
        }
        if (!current.empty()) chunks.push_back(std::move(current));

        size_t total_atoms = 0;
        size_t total_neighbors = 0;
        size_t reused_topologies = 0;
        const auto stats_before = evaluator.context().runtime_stats();
        for (const auto &chunk : chunks) {
            bool reuse_resident_topology = false;
            if (gpu_reuse_topology && chunk.size() == 1) {
                std::string compatibility_error;
                reuse_resident_topology = batch.refresh_compatible(
                    atoms[chunk[0]], neighborhoods[chunk[0]],
                    &compatibility_error);
            }
            if (!reuse_resident_topology &&
                !batch.build(atoms, neighborhoods, chunk, &gpu_error)) {
                GPERROR("Unable to build GPU MTP evaluation chunk: " +
                        gpu_error);
            }
            if (energy_only) {
                if (!batch.evaluate_energy(evaluator, atoms, chunk,
                                           reuse_resident_topology,
                                           &gpu_error)) {
                    GPERROR("GPU MTP energy evaluation failed: " +
                            gpu_error);
                }
            } else if (!batch.evaluate(evaluator, atoms, chunk,
                                       reuse_resident_topology, &gpu_error)) {
                GPERROR("GPU MTP evaluation failed: " + gpu_error);
            }
            reused_topologies += reuse_resident_topology ? 1 : 0;
            total_atoms += static_cast<size_t>(batch.atom_count());
            total_neighbors += static_cast<size_t>(batch.neighbor_count());
            if (show_progress) {
                const size_t previous = ndone;
                ndone += chunk.size();
                if (ndone / report_every != previous / report_every || ndone == atoms.size())
                    Message(fmt::format("Calculated EFS {}/{}", ndone, atoms.size()));
            }
        }
        Message("GPU MTP evaluated " + std::to_string(atoms.size()) +
                " structures, " + std::to_string(total_atoms) + " atoms, and " +
                std::to_string(total_neighbors) + " directed neighbors in " +
                std::to_string(chunks.size()) + " structure-preserving chunk(s); " +
                "budget=" + std::to_string(memory_budget / (1024 * 1024)) +
                " MiB, compatible_topology_reuses=" +
                std::to_string(reused_topologies) +
                ", async_input_uploads=" +
                std::to_string(
                    evaluator.context().runtime_stats().pipelined_input_uploads -
                    stats_before.pipelined_input_uploads));
    } else
#endif
    {
    // Validate the potential before the parallel region: each thread builds
    // its own wrapper below (getEFS is not thread-safe on a shared pot), but a
    // failed load must throw here — outside the OpenMP region, where
    // exceptions cannot escape and would terminate the process.
    {
        Potentials::potential_wrapper wrapper(pfile, ptype, settings.ExtractSubSettings("pot"));
    }
    std::exception_ptr evaluation_error;
#ifdef USE_OMP
        if (atoms.size() < omp_num_threads)
            omp_num_threads = atoms.size();
        omp_set_num_threads(omp_num_threads);
#pragma omp parallel
#endif
        {
            Potentials::AnyPotential *pot;
            Potentials::potential_wrapper wrapper(pfile, ptype, settings.ExtractSubSettings("pot"));
            pot = wrapper.pot;
            #ifdef USE_OMP
            #pragma omp for
            #endif
            for (int i = 0; i < atoms.size(); i++) {
                try {
                    atoms[i].wrap();
                    if (energy_only)
                        pot->getE(atoms[i]);
                    else
                        pot->getEFS(atoms[i]);
                } catch (...) {
                    #pragma omp critical
                    {
                        if (!evaluation_error) evaluation_error = std::current_exception();
                    }
                }
                if (show_progress) {
                    #pragma omp critical
                    {
                        ++ndone;
                        if (ndone % report_every == 0 || ndone == atoms.size())
                            Message(fmt::format("Calculated EFS {}/{}", ndone, atoms.size()));
                    }
                }
            }
        }
        if (evaluation_error) {
            std::rethrow_exception(evaluation_error);
        }
    }

    if (energy_only) {
        for (Atoms &structure : atoms) {
            structure.forces_.clear();
            structure.site_energies_.clear();
            structure.stress_.setZero();
            structure.have_stress = false;
        }
    } else if (no_site_energy) {
        for (int i = 0; i < atoms.size(); ++i) {
            atoms[i].site_energies_.clear();
        }
    }

    const bool written = iobj.write(atoms, outfile, outformat);
    if (!written) return 1;
    Message("Complete!");
    return 0;
}
