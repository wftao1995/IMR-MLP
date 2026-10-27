// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "gpu-direct/gpu_mpi.h"
#include "pair_MLIP_gpu_direct.h"
#include "mlip_model_mpi.h"
#include "mlip_restart.h"

#include <cstring>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "mpi.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "update.h"
#include "utils.h"

using namespace LAMMPS_NS;

namespace {

bool is_option_key(const char *arg)
{
    return std::strcmp(arg, "device") == 0 ||
           std::strcmp(arg, "precision") == 0 ||
           std::strcmp(arg, "force_kernel") == 0 ||
           std::strcmp(arg, "deterministic") == 0 ||
           std::strcmp(arg, "basic_tile") == 0 ||
           std::strcmp(arg, "neighbor_lanes") == 0 ||
           std::strcmp(arg, "basic_lanes") == 0 ||
           std::strcmp(arg, "propagate_threads") == 0 ||
           std::strcmp(arg, "block_neigh") == 0 ||
           std::strcmp(arg, "block_basic") == 0 ||
           std::strcmp(arg, "block_times") == 0 ||
           std::strcmp(arg, "gpu_id") == 0 ||
           std::strcmp(arg, "centers_per_chunk") == 0 ||
           std::strcmp(arg, "memory_budget_mib") == 0 ||
           std::strcmp(arg, "chunk_size") == 0 ||
           std::strcmp(arg, "chunck_size") == 0;
}

} // namespace

PairMLIPGpuDirect::PairMLIPGpuDirect(LAMMPS *lmp) : Pair(lmp)
{
    restartinfo = 1;
    manybody_flag = 1;
    single_enable = 0;
    allocated = 0;
    centroidstressflag = CENTROID_NOTAVAIL;
}

PairMLIPGpuDirect::~PairMLIPGpuDirect()
{
    if (copymode) {
        return;
    }
    if (allocated) {
        deallocate();
    }
    destroy_core();
}

void PairMLIPGpuDirect::destroy_core()
{
    if (p_cpu_buffers != nullptr) {
        MLIPCore::destroy_mlip_buffer(p_cpu_buffers);
        p_cpu_buffers = nullptr;
    }
    if (p_mlip != nullptr) {
        MLIPCore::destroy_mlip(p_mlip);
        p_mlip = nullptr;
    }
    delete p_gpu;
    p_gpu = nullptr;
    gpu_initialized = false;
    gpu_stats_logged = false;
}

void PairMLIPGpuDirect::settings(int narg, char **arg)
{
    try {
        if (metadata_arguments_.defer(narg, arg, narg > 0 && is_option_key(arg[0]))) return;
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
    if (narg == 0) {
        error->all(FLERR, "Incorrect pair_style mlip/gpudirect arguments");
    }
    if (std::strlen(arg[0]) >= sizeof(MLIPpot_filename)) {
        error->all(FLERR, "MLIP settings file name too long");
    }

    destroy_core();
    gpu_precision = MLIPCore::GpuDirect::MTPGpuPrecision::FP64;
    force_kernel = MLIPCore::GpuDirect::MTPGpuForceKernel::Auto;
    deterministic_reduction = false;
    centers_per_chunk = 0;
    memory_budget_bytes = 0;

    std::string model_text;
    int iarg = 1;
    while (iarg < narg && !is_option_key(arg[iarg])) ++iarg;
    try {
        if (pending_restart_model_text_.empty())
            model_text = MLIPCore::read_model_text_mpi(world, arg[0]);
        else {
            model_text = std::move(pending_restart_model_text_);
            pending_restart_model_text_.clear();
        }
        std::istringstream header_input(model_text);
        const auto header = MTPMetadata::read_header(header_input);
        spmap.assign(static_cast<std::size_t>(iarg), -1);
        metadata_arguments_.load(narg, arg, iarg, spmap.data(), spmap.size(), false, true, &header);
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }

    for (; iarg < narg; ++iarg) {
        if (iarg + 1 >= narg) {
            error->all(FLERR, "Missing value for mlip/gpudirect option");
        }
        const char *key = arg[iarg++];
        if (std::strcmp(key, "device") == 0) {
            std::string_view requested = arg[iarg];
            if (requested == "0") requested = "cpu";
            if (requested == "1") requested = MLIPCore::GpuDirect::MTPGpuDirect::backend_name();
            requested = MLIPCore::GpuDirect::canonical_device(requested);
            if (!MLIPCore::GpuDirect::valid_device(requested)) {
                error->all(FLERR, "mlip/gpudirect device must be cpu, cuda, hip, or dcu");
            }
            if (requested != "cpu" && requested != MLIPCore::GpuDirect::MTPGpuDirect::backend_name()) {
                error->all(FLERR, "mlip/gpudirect device does not match the compiled backend");
            }
            use_gpu = requested != "cpu";
        } else if (std::strcmp(key, "precision") == 0) {
            if (!MLIPCore::GpuDirect::parse_precision(
                    arg[iarg], &gpu_precision)) {
                error->all(FLERR,
                    "mlip/gpudirect precision must be fp64, mixed, or mixed-aggressive");
            }
        } else if (std::strcmp(key, "force_kernel") == 0) {
            if (!MLIPCore::GpuDirect::parse_force_kernel(
                    arg[iarg], &force_kernel)) {
                error->all(FLERR,
                    "mlip/gpudirect force_kernel must be auto, basic-tile, or center-neighbors");
            }
        } else if (std::strcmp(key, "deterministic") == 0) {
            if (std::strcmp(arg[iarg], "yes") == 0 ||
                std::strcmp(arg[iarg], "true") == 0 ||
                std::strcmp(arg[iarg], "1") == 0) {
                deterministic_reduction = true;
            } else if (std::strcmp(arg[iarg], "no") == 0 ||
                       std::strcmp(arg[iarg], "false") == 0 ||
                       std::strcmp(arg[iarg], "0") == 0) {
                deterministic_reduction = false;
            } else {
                error->all(FLERR,
                    "mlip/gpudirect deterministic must be yes or no");
            }
        } else if (std::strcmp(key, "gpu_id") == 0) {
            if (std::strcmp(arg[iarg], "auto") == 0) {
                gpu_id = -1;
            } else {
                gpu_id = utils::inumeric(FLERR, arg[iarg], false, lmp);
                if (gpu_id < 0) {
                    error->all(FLERR, "mlip/gpudirect gpu_id must be auto or non-negative");
                }
            }
        } else if (std::strcmp(key, "basic_tile") == 0) {
            basic_tile = utils::inumeric(FLERR, arg[iarg], false, lmp);
        } else if (std::strcmp(key, "neighbor_lanes") == 0 ||
                   std::strcmp(key, "block_neigh") == 0) {
            neighbor_lanes = utils::inumeric(FLERR, arg[iarg], false, lmp);
        } else if (std::strcmp(key, "basic_lanes") == 0 ||
                   std::strcmp(key, "block_basic") == 0) {
            basic_lanes = utils::inumeric(FLERR, arg[iarg], false, lmp);
        } else if (std::strcmp(key, "propagate_threads") == 0 ||
                   std::strcmp(key, "block_times") == 0) {
            propagate_threads = utils::inumeric(FLERR, arg[iarg], false, lmp);
        } else if (std::strcmp(key, "centers_per_chunk") == 0 ||
                   std::strcmp(key, "chunk_size") == 0 ||
                   std::strcmp(key, "chunck_size") == 0) {
            centers_per_chunk = utils::inumeric(FLERR, arg[iarg], false, lmp);
            if (centers_per_chunk < 0)
                error->all(FLERR, "mlip/gpudirect centers_per_chunk must be nonnegative");
        } else if (std::strcmp(key, "memory_budget_mib") == 0) {
            try {
                const std::string value(arg[iarg]);
                if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
                    throw std::invalid_argument("invalid budget");
                const auto mib = std::stoull(value);
                if (mib > (std::numeric_limits<std::size_t>::max() >> 20))
                    throw std::out_of_range("budget overflow");
                memory_budget_bytes = static_cast<std::size_t>(mib) << 20;
            } catch (const std::exception &) {
                error->all(FLERR, "mlip/gpudirect memory_budget_mib must be a nonnegative integer that fits size_t");
            }
        } else {
            error->all(FLERR, "Unknown mlip/gpudirect option");
        }
    }

    if (basic_tile <= 0 || neighbor_lanes <= 0 ||
        basic_lanes <= 0 || propagate_threads <= 0) {
        error->all(FLERR, "mlip/gpudirect tuning parameters must be positive");
    }
    if ((propagate_threads & (propagate_threads - 1)) != 0) {
        error->all(FLERR, "mlip/gpudirect propagate_threads must be a power of two");
    }
    if (!use_gpu && (centers_per_chunk || memory_budget_bytes))
        error->all(FLERR, "mlip/gpudirect chunking and GPU memory budgets require a GPU device");
    if (!use_gpu &&
        gpu_precision != MLIPCore::GpuDirect::MTPGpuPrecision::FP64) {
        error->all(FLERR,
            "mlip/gpudirect reduced precision requires a GPU device");
    }
    if (!use_gpu && deterministic_reduction) {
        error->all(FLERR,
            "mlip/gpudirect deterministic yes requires a GPU device");
    }
    if (!use_gpu && force_kernel !=
            MLIPCore::GpuDirect::MTPGpuForceKernel::Auto) {
        error->all(FLERR,
            "mlip/gpudirect force_kernel tuning requires a GPU device");
    }
    if (deterministic_reduction && force_kernel ==
            MLIPCore::GpuDirect::MTPGpuForceKernel::CenterNeighbors) {
        error->all(FLERR,
            "mlip/gpudirect center-neighbors is unavailable with deterministic yes");
    }

    std::strcpy(MLIPpot_filename, arg[0]);
    try {
        MLIPCore::collective_try(world, [&] {
            p_mlip = MLIPCore::setup_from_text(model_text, spmap.data(), &cutoff, spmap.size());
            if (!use_gpu) p_cpu_buffers = MLIPCore::create_mlip_buffer(p_mlip);
        });
    } catch (const std::exception &e) { error->all(FLERR, e.what()); }
    no_virial_fdotr_compute = use_gpu ? 1 : 0;
    if (!use_gpu && p_mlip->site_model != MLIPCore::mlip::MODEL_LINEAR) {
        error->all(FLERR,
            "mlip/gpudirect nonlinear MTP-M models require a GPU device");
    }
    if (use_gpu) {
        std::string gpu_error;
        try {
            MLIPCore::collective_try(world, [&] {
                p_gpu = new MLIPCore::GpuDirect::MTPGpuDirect();
                int device_count = 0;
                if (!MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(
                        &device_count, &gpu_error) || device_count <= 0)
                    throw std::runtime_error(gpu_error.empty() ?
                        "No GPU device is visible to mlip/gpudirect" : gpu_error);
            });
        } catch (const std::exception &e) { error->all(FLERR, e.what()); }
        try {
            if (!MLIPCore::GpuDirect::select_mpi_device(
                    world, gpu_id, &selected_gpu_id, &gpu_error))
                throw std::runtime_error(gpu_error);
        } catch (const std::exception &e) { error->one(FLERR, e.what()); }
        try {
            MLIPCore::collective_try(world, [&] {
                if (!p_gpu->set_device(selected_gpu_id, &gpu_error))
                    throw std::runtime_error(gpu_error);
                MLIPCore::GpuDirect::MTPGpuInitializationOptions options;
                options.precision = gpu_precision;
                options.memory_budget_bytes = memory_budget_bytes;
                if (!p_gpu->initialize_from_mlip(*p_mlip, options, &gpu_error))
                    throw std::runtime_error(gpu_error);
            });
        } catch (const std::exception &e) { error->all(FLERR, e.what()); }
        gpu_initialized = true;
        if (comm->me == 0) {
            const auto stats = p_gpu->potential_stats();
            std::ostringstream message;
            message << "mlip/gpudirect " << MLIPCore::GpuDirect::MTPGpuDirect::backend_name() << " device " << selected_gpu_id
                    << ", precision "
                    << MLIPCore::GpuDirect::precision_name(stats.precision)
                    << ", moment element bytes " << stats.moment_element_bytes
                    << ", adjoint element bytes " << stats.adjoint_element_bytes
                    << ", resident potential bytes " << stats.potential_bytes
                    << ", deterministic reduction "
                    << (deterministic_reduction ? "yes" : "no")
                    << ", force kernel "
                    << MLIPCore::GpuDirect::force_kernel_name(force_kernel)
                    << "\n";
            utils::logmesg(lmp, message.str());
        }
    }
    restart_model_text_ = std::move(model_text);
    restart_arguments_.assign(arg, arg + narg);
    metadata_arguments_.loaded = true;
}

void PairMLIPGpuDirect::write_restart(FILE *file)
{
    if (comm->me != 0) return;
    try {
        MLIPCore::RestartState state;
        state.filename = MLIPpot_filename;
        state.model_text = restart_model_text_;
        state.arguments = restart_arguments_;
        MLIPCore::write_restart_state(file, state);
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
}

void PairMLIPGpuDirect::read_restart(FILE *file)
{
    try {
        auto state = MLIPCore::read_restart_state(file, world);
        pending_restart_model_text_ = std::move(state.model_text);
        std::vector<char *> arguments;
        for (auto &argument : state.arguments) arguments.push_back(argument.data());
        metadata_arguments_ = MTPMetadata::PairArguments{};
        settings(static_cast<int>(arguments.size()), arguments.data());
        if (!allocated) allocate();
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
}

void PairMLIPGpuDirect::coeff(int narg, char **arg)
{
    try {
        auto arguments = metadata_arguments_.coefficients(narg, arg, atom->ntypes);
        if (!arguments.empty()) {
            std::vector<char *> pointers;
            for (auto &value : arguments) pointers.push_back(value.data());
            settings(static_cast<int>(pointers.size()), pointers.data());
        }
        metadata_arguments_.validate(atom->ntypes);
    } catch (const std::exception &e) { error->one(FLERR, e.what()); }
    if (!allocated) {
        allocate();
    }
}

void PairMLIPGpuDirect::init_style()
{
    try { metadata_arguments_.validate(atom->ntypes); }
    catch (const std::exception &e) { error->all(FLERR, e.what()); }
    if (force->newton_pair != 1) {
        error->all(FLERR, "Pair style mlip/gpudirect requires Newton pair on");
    }
    if (p_mlip == nullptr) {
        error->all(FLERR, "Pair style mlip/gpudirect potential is not initialized");
    }
    if (std::strcmp(update->unit_style, "metal") != 0) {
        error->all(FLERR, "Pair style mlip/gpudirect requires units metal");
    }
    if (static_cast<std::size_t>(atom->ntypes) >= spmap.size())
        error->all(FLERR, "mlip/gpudirect is missing atom type mappings");
    if (p_mlip->site_model != MLIPCore::mlip::MODEL_LINEAR) {
        if (domain->dimension != 3)
            error->all(FLERR, "MTP_M requires dimension 3");
        // Neighbor::exclude is finalized after pair initialization. Inspect
        // the configured lists so first-run exclusions are rejected as well.
        if (neighbor->nex_type || neighbor->nex_group || neighbor->nex_mol)
            error->all(FLERR, "MTP_M requires complete environments; neighbor exclusions are unsupported");
        if (neighbor->includegroup)
            error->all(FLERR, "MTP_M requires all owned centers; neigh_modify include is unsupported");
        if (atom->molecular) for (int k = 1; k <= 3; ++k)
            if (force->special_lj[k] != 1.0 || force->special_coul[k] != 1.0)
                error->all(FLERR, "MTP_M molecular systems require special_bonds lj/coul 1 1 1");
    }
    for (int type = 1; type <= atom->ntypes; ++type) {
        if (spmap[type] < 0 || spmap[type] >= p_mlip->species_count) {
            error->all(FLERR, "mlip/gpudirect is missing a valid species mapping");
        }
    }
    if (!allocated) {
        allocate();
    }

    cutoffsq = cutoff * cutoff;
    const int n = atom->ntypes;
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            cutsq[i][j] = cutoffsq;
        }
    }

    inited = true;
    neighbor->add_request(this, NeighConst::REQ_FULL);
}

void PairMLIPGpuDirect::allocate()
{
    const int n = atom->ntypes;
    memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
    memory->create(setflag, n + 1, n + 1, "pair:setflag");
    for (int i = 1; i <= n; ++i) {
        for (int j = 1; j <= n; ++j) {
            setflag[i][j] = 1;
        }
    }
    allocated = 1;
}

void PairMLIPGpuDirect::deallocate()
{
    if (allocated) {
        memory->destroy(cutsq);
        memory->destroy(setflag);
    }
    allocated = 0;
}

void PairMLIPGpuDirect::compute(int eflag, int vflag)
try
{
    if (eflag || vflag) {
        ev_setup(eflag, vflag);
    } else {
        eflag = vflag = eflag_global = eflag_atom = vflag_global = vflag_atom = vflag_fdotr = 0;
    }

    double en = 0.0;
    double direct_virial[6] = {};
    double *p_site_en = nullptr;
    double **p_site_virial = nullptr;
    if (eflag_atom) {
        p_site_en = &eatom[0];
    }
    if (vflag_atom) {
        p_site_virial = vatom;
    }

    if (!use_gpu) {
        MLIPCore::compute(p_mlip, p_cpu_buffers,
                          list->ilist,
                          list->numneigh,
                          list->inum,
                          atom->type,
                          atom->x,
                          list->firstneigh,
                          &en,
                          p_site_en,
                          p_site_virial,
                          atom->f);
    } else {
        if (!gpu_initialized || p_gpu == nullptr) {
            error->one(FLERR, "Pair style mlip/gpudirect GPU context is not initialized");
        }
        MLIPCore::GpuDirect::MTPGpuComputeRequest req;
        req.ilist = list->ilist;
        req.numneigh = list->numneigh;
        req.firstneigh = list->firstneigh;
        req.inum = list->inum;
        req.nlocal = atom->nlocal;
        req.ntotal = atom->nlocal + atom->nghost;
        req.type = atom->type;
        req.x = atom->x;
        req.force = atom->f;
        req.total_energy = &en;
        req.site_energy = p_site_en;
        req.site_virial = p_site_virial;
        req.global_virial = vflag_global ? direct_virial : nullptr;
        req.inference_only = true;
        req.centers_per_chunk = centers_per_chunk;
        req.memory_budget_bytes = memory_budget_bytes;
        req.neighbor_index_mask = NEIGHMASK;
        req.accumulate_force = true;
        req.reuse_topology = neighbor->ago != 0;
        req.compute_global_energy = eflag_global != 0;
        req.compute_site_energy = eflag_atom != 0;
        req.deterministic_reduction = deterministic_reduction;
        req.force_kernel = force_kernel;
        req.basic_tile = basic_tile;
        req.neighbor_lanes = neighbor_lanes;
        req.basic_lanes = basic_lanes;
        req.propagate_threads = propagate_threads;

        std::string gpu_error;
        if (!p_gpu->compute_lammps(req, &gpu_error)) {
            error->one(FLERR, gpu_error.c_str());
        }
        if (!gpu_stats_logged) {
            gpu_stats_logged = true;
            if (comm->me == 0) {
                const auto stats = p_gpu->memory_stats();
                const auto runtime = p_gpu->runtime_stats();
                std::ostringstream message;
                message << "mlip/gpudirect allocated GPU bytes: total "
                        << stats.tracked_device_bytes << ", moment "
                        << stats.moment_workspace_bytes << ", adjoint "
                        << stats.adjoint_workspace_bytes << ", input "
                        << stats.input_bytes << ", output "
                        << stats.output_bytes << ", topology "
                        << stats.topology_bytes << ", neural cache "
                        << stats.neural_cache_bytes << ", conversion "
                        << stats.conversion_buffer_bytes
                        << "; uploads: topology "
                        << runtime.topology_uploads << "/"
                        << runtime.topology_upload_bytes << " bytes, types "
                        << runtime.type_uploads << "/"
                        << runtime.type_upload_bytes << " bytes, coordinates "
                        << runtime.coordinate_uploads << "/"
                        << runtime.coordinate_upload_bytes
                        << " bytes; force launches: basic-tile "
                        << runtime.basic_tile_force_launches
                        << ", center-neighbors "
                        << runtime.center_neighbors_force_launches
                        << "; centers/chunk " << runtime.last_centers_per_chunk
                        << ", chunks " << runtime.last_chunk_count
                        << ", output bytes " << runtime.output_download_bytes << "\n";
                utils::logmesg(lmp, message.str());
            }
        }
    }

    if (eflag_global) {
        eng_vdwl += en;
    }
    if (use_gpu && vflag_global)
        for (int k = 0; k < 6; ++k) virial[k] += direct_virial[k];
    if (vflag_fdotr) {
        virial_fdotr_compute();
    }
}
catch (const std::exception &e)
{
    // Force evaluation is rank-local; a barrier here can strand the other ranks
    // in reverse communication after a device or allocation failure.
    error->one(FLERR, e.what());
}

double PairMLIPGpuDirect::init_one(int, int)
{
    return cutoff;
}
