// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(mlip/gpudirect, PairMLIPGpuDirect)

#else

#ifndef LMP_PAIR_MLIP_GPU_DIRECT_H
#define LMP_PAIR_MLIP_GPU_DIRECT_H

#include "mlip_pair_metadata.h"
#include "pair.h"
#include "mlip.h"
#include "gpu-direct/mtp_gpu_direct.h"

namespace LAMMPS_NS
{

    class PairMLIPGpuDirect : public Pair
    {
    public:
        explicit PairMLIPGpuDirect(class LAMMPS *);
        ~PairMLIPGpuDirect() override;

        void compute(int, int) override;
        void init_style() override;
        double init_one(int, int) override;
        void settings(int, char **) override;
        void coeff(int, char **) override;
        void write_restart(FILE *) override;
        void read_restart(FILE *) override;

    protected:
        MTPMetadata::PairArguments metadata_arguments_;
        std::vector<std::string> restart_arguments_;
        std::string restart_model_text_, pending_restart_model_text_;
        void allocate();
        void deallocate();
        void destroy_core();

        bool inited = false;
        bool gpu_initialized = false;
        double cutoff = 0.0;
        double cutoffsq = 0.0;
        char MLIPpot_filename[1000] = {};

        std::vector<int> spmap;
        bool use_gpu = true;
        MLIPCore::GpuDirect::MTPGpuPrecision gpu_precision =
            MLIPCore::GpuDirect::MTPGpuPrecision::FP64;
        MLIPCore::GpuDirect::MTPGpuForceKernel force_kernel =
            MLIPCore::GpuDirect::MTPGpuForceKernel::Auto;
        bool deterministic_reduction = false;
        int gpu_id = -1;
        int selected_gpu_id = -1;
        bool gpu_stats_logged = false;
        int basic_tile = 128;
        int neighbor_lanes = 8;
        int basic_lanes = 16;
        int propagate_threads = 256;
        int centers_per_chunk = 0;
        std::size_t memory_budget_bytes = 0;

        MLIPCore::mlip *p_mlip = nullptr;
        MLIPCore::mlip_buffers *p_cpu_buffers = nullptr;
        MLIPCore::GpuDirect::MTPGpuDirect *p_gpu = nullptr;
    };

}

#endif
#endif
