// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "potential.h"

#ifdef USE_KIM
#include <kim-api/KIM_SimulatorHeaders.hpp>

namespace Potentials {
    class PotKIM : public AnyPotential {
    friend struct GPMCodecAccess;
    private:
        std::string kim_modelname;
        struct ModelParameter {
            int index, extent;
            std::vector<int> integers;
            std::vector<double> doubles;
        };
        std::map<std::string, ModelParameter> model_parameters;
        void refresh_neighbor_metadata();
        void read_parameter_values();

        KIM::Model *kim_Model;
        KIM::ComputeArguments *computeArguments;
        KIM::LengthUnit lengthUnit;
        KIM::EnergyUnit energyUnit;
        KIM::ChargeUnit chargeUnit;
        KIM::TemperatureUnit temperatureUnit;
        KIM::TimeUnit timeUnit;

        KIM::SupportStatus kim_support_pe = KIM::SUPPORT_STATUS::notSupported;
        KIM::SupportStatus kim_support_pv = KIM::SUPPORT_STATUS::notSupported;

        std::vector<int> model_code;
        double model_cutoff = 5.0;
        int n_neigh_lists = 0;
        std::vector<double> neigh_cutoffs;
        std::vector<int> modelWillNotRequestNeighborsOfNoncontributing;

        // Neighbor list cache for KIM callback (flat arrays for O(1) lookup)
        std::vector<int> neigh_list_flat;
        std::vector<int> neigh_offsets;

        // Compute buffers (re-allocated on atom count change)
        std::vector<double> coords_flat;
        std::vector<double> forces_flat;
        std::vector<double> virial_flat;
        double energy_val;
        int n_atoms;

        // buffers
        struct omp_buffer_type {
            int natoms;
            std::vector<int> types;
            std::vector<int> particle_contrib;
            std::vector<double> coords;
            std::vector<double> forces;
            std::vector<double> atomic_prop;
        };
        std::vector<omp_buffer_type> omp_buffers;

        // Static callback for KIM GetNeighborList
        static int GetNeighborListCallback(void *dataObject,
                                           int numberOfNeighborLists,
                                           double const *cutoffs,
                                           int neighborListIndex,
                                           int particleNumber,
                                           int *numberOfNeighbors,
                                           int const **neighborsOfParticle);

    public:
        PotKIM(const std::string &modelname, const std::map<int, std::string> &typemap);
        ~PotKIM();

        // Apply complete model arrays, then refresh model-derived metadata once.
        void ApplyParameterSettings(const GPSettings &settings);

        /// Auto-detect species supported by a KIM model (used when index.map is missing)
        static std::map<int, std::string> QueryModelSpecies(const std::string &modelname);

        void omp_init() {
            omp_buffers.resize(omp_num_threads);
            omp_inited = true;
        }

        double get_cutoff() const { return model_cutoff; }

        void getEFS(Atoms &atom);
        void getEFS(Atoms &atom, AnyNeighborhoods &nbhs) override { getEFS(atom); }
    };
}

#endif
