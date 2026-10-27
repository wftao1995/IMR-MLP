// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

// KIM Model Driver for MTP (Moment Tensor Potential)
// Implements the KIM API v2 model-side interface.
// The .mtp parameter file is read at model creation time.

#include <kim-api/KIM_ModelDriverHeaders.hpp>
#include "mtp_kim_core.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

extern "C" int MTPDriver_ComputeArgumentsCreate(
    KIM::ModelCompute const *, KIM::ModelComputeArgumentsCreate *);
extern "C" int MTPDriver_Compute(
    KIM::ModelCompute const *, KIM::ModelComputeArguments const *);
extern "C" int MTPDriver_ComputeArgumentsDestroy(
    KIM::ModelCompute const *, KIM::ModelComputeArgumentsDestroy *);
extern "C" int MTPDriver_Destroy(KIM::ModelDestroy *);

// ── Model-specific data structure ──────────────────────────────────────
struct MTPModel {
    mtp_kim::MTPCore core;
    double cutoff = 0.0;
    double lengthToInternal = 1.0;
    double energyFromInternal = 1.0;
    double forceFromInternal = 1.0;
};

namespace {

void log(KIM::ModelDriverCreate const *create, KIM::LogVerbosity verbosity,
         const std::string &message, int line)
{
    create->LogEntry(verbosity, message, line, __FILE__);
}

bool has_suffix(const std::string &text, const std::string &suffix)
{
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool read_species(const std::string &path, std::vector<std::string> &species,
                  std::string &error)
{
    std::ifstream input(path);
    if (!input) {
        error = "cannot open species file " + path;
        return false;
    }
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::istringstream tokens(line);
        std::string name;
        while (tokens >> name) species.push_back(name);
    }
    return true;
}

} // namespace

// =========================================================================
// Create — called when a new model instance is created from parameters
// =========================================================================
extern "C" int
MTPDriver_Create(KIM::ModelDriverCreate * const modelDriverCreate,
                  KIM::LengthUnit const requestedLengthUnit,
                  KIM::EnergyUnit const requestedEnergyUnit,
                  KIM::ChargeUnit const requestedChargeUnit,
                  KIM::TemperatureUnit const requestedTemperatureUnit,
                  KIM::TimeUnit const requestedTimeUnit)
{
    auto *model = new MTPModel();
    if (!model) return true;

    // ── Get parameter file ────────────────────────────────────────────
    std::string const *paramDir;
    modelDriverCreate->GetParameterFileDirectoryName(&paramDir);
    int numParamFiles;
    modelDriverCreate->GetNumberOfParameterFiles(&numParamFiles);

    std::string mtpPath;
    std::string speciesPath;
    int mtpFileCount = 0;
    int speciesFileCount = 0;
    for (int i = 0; i < numParamFiles; ++i) {
        std::string const *basename;
        modelDriverCreate->GetParameterFileBasename(i, &basename);
        std::string name = *basename;
        if (has_suffix(name, ".mtp")) {
            mtpPath = *paramDir + "/" + name;
            ++mtpFileCount;
        } else if (has_suffix(name, ".species")) {
            speciesPath = *paramDir + "/" + name;
            ++speciesFileCount;
        }
    }
    if (mtpFileCount != 1) {
        log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
            "Error: exactly one .mtp parameter file is required.", __LINE__);
        delete model;
        return true;
    }

    // ── Load MTP model ───────────────────────────────────────────────
    if (!model->core.load(mtpPath)) {
        log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
            "Error: failed to load MTP model from " + mtpPath, __LINE__);
        delete model;
        return true;
    }

    std::vector<std::string> species;
    std::string speciesError;
    if (speciesFileCount != 1 ||
        !read_species(speciesPath, species, speciesError) ||
        static_cast<int>(species.size()) != model->core.speciesCount()) {
        const std::string detail = speciesFileCount != 1
            ? "exactly one .species parameter file is required"
            : (!speciesError.empty()
                ? speciesError
                : ".species entry count does not match species_count");
        log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
            "Error: " + detail +
            ". The file must list chemical symbols in MTP species-index order.",
            __LINE__);
        delete model;
        return true;
    }
    for (std::size_t i = 0; i < species.size(); ++i) {
        if (std::find(species.begin(), species.begin() + i, species[i]) !=
            species.begin() + i) {
            log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
                "Error: duplicate species name '" + species[i] + "'.",
                __LINE__);
            delete model;
            return true;
        }
    }

    // The potential file is stored in Angstrom/eV. Keep the core in those
    // units and convert KIM inputs/outputs at its boundary.
    double lengthFromInternal = 0.0;
    if (KIM::ModelDriverCreate::ConvertUnit(
            KIM::LENGTH_UNIT::A, KIM::ENERGY_UNIT::eV,
            KIM::CHARGE_UNIT::unused, KIM::TEMPERATURE_UNIT::unused,
            KIM::TIME_UNIT::unused,
            requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
            requestedTemperatureUnit, requestedTimeUnit,
            1.0, 0.0, 0.0, 0.0, 0.0, &lengthFromInternal) ||
        KIM::ModelDriverCreate::ConvertUnit(
            KIM::LENGTH_UNIT::A, KIM::ENERGY_UNIT::eV,
            KIM::CHARGE_UNIT::unused, KIM::TEMPERATURE_UNIT::unused,
            KIM::TIME_UNIT::unused,
            requestedLengthUnit, requestedEnergyUnit, requestedChargeUnit,
            requestedTemperatureUnit, requestedTimeUnit,
            0.0, 1.0, 0.0, 0.0, 0.0, &model->energyFromInternal) ||
        !std::isfinite(lengthFromInternal) || lengthFromInternal <= 0.0 ||
        !std::isfinite(model->energyFromInternal) ||
        model->energyFromInternal <= 0.0) {
        log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
            "Error: cannot convert Angstrom/eV MTP units to the requested units.",
            __LINE__);
        delete model;
        return true;
    }
    model->lengthToInternal = 1.0 / lengthFromInternal;
    model->forceFromInternal =
        model->energyFromInternal * model->lengthToInternal;
    model->cutoff = model->core.max_cutoff * lengthFromInternal;

    if (modelDriverCreate->SetUnits(
        requestedLengthUnit, requestedEnergyUnit,
        KIM::CHARGE_UNIT::unused,
        KIM::TEMPERATURE_UNIT::unused,
        KIM::TIME_UNIT::unused)) {
        log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
            "Error: KIM API rejected MTP units.", __LINE__);
        delete model;
        return true;
    }

    // ── Set influence distance ────────────────────────────────────────
    modelDriverCreate->SetInfluenceDistancePointer(&model->cutoff);

    // ── Set neighbor list pointers ────────────────────────────────────
    static const int noNonContributing = 1;
    modelDriverCreate->SetNeighborListPointers(
        1, &model->cutoff, &noNonContributing);

    // ── Register species ──────────────────────────────────────────────
    // MTP species are integer indices 0..N-1; the companion file supplies
    // their chemical names in exactly that order.
    for (int i = 0; i < model->core.speciesCount(); ++i) {
        const KIM::SpeciesName speciesName(species[i]);
        if (!speciesName.Known() || modelDriverCreate->SetSpeciesCode(speciesName, i)) {
            log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
                "Error: invalid KIM species name '" + species[i] + "'.", __LINE__);
            delete model;
            return true;
        }
    }

    // ── Set numbering ─────────────────────────────────────────────────
    if (modelDriverCreate->SetModelNumbering(KIM::NUMBERING::zeroBased)) {
        log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
            "Error: KIM API rejected zero-based numbering.", __LINE__);
        delete model;
        return true;
    }

    // ── Register routine pointers ─────────────────────────────────────
    int registrationError = 0;
    registrationError |= modelDriverCreate->SetRoutinePointer(
        KIM::MODEL_ROUTINE_NAME::ComputeArgumentsCreate,
        KIM::LANGUAGE_NAME::cpp, true,
        reinterpret_cast<KIM::Function *>(MTPDriver_ComputeArgumentsCreate));
    registrationError |= modelDriverCreate->SetRoutinePointer(
        KIM::MODEL_ROUTINE_NAME::Compute,
        KIM::LANGUAGE_NAME::cpp, true,
        reinterpret_cast<KIM::Function *>(MTPDriver_Compute));
    registrationError |= modelDriverCreate->SetRoutinePointer(
        KIM::MODEL_ROUTINE_NAME::ComputeArgumentsDestroy,
        KIM::LANGUAGE_NAME::cpp, true,
        reinterpret_cast<KIM::Function *>(MTPDriver_ComputeArgumentsDestroy));
    registrationError |= modelDriverCreate->SetRoutinePointer(
        KIM::MODEL_ROUTINE_NAME::Destroy,
        KIM::LANGUAGE_NAME::cpp, true,
        reinterpret_cast<KIM::Function *>(MTPDriver_Destroy));
    if (registrationError) {
        log(modelDriverCreate, KIM::LOG_VERBOSITY::fatal,
            "Error: failed to register one or more MTP model routines.", __LINE__);
        delete model;
        return true;
    }

    // ── Store model buffer ────────────────────────────────────────────
    modelDriverCreate->SetModelBufferPointer(model);

    log(modelDriverCreate, KIM::LOG_VERBOSITY::information,
        "MTP model loaded: " + std::to_string(model->core.speciesCount()) +
        " species, cutoff=" + std::to_string(model->cutoff) +
        ", " + std::to_string(model->core.alphaBasicCount()) + " basic moments",
        __LINE__);

    return false;
}

// =========================================================================
// ComputeArgumentsCreate — declare argument/callback support
// =========================================================================
extern "C" int
MTPDriver_ComputeArgumentsCreate(
    KIM::ModelCompute const *,
    KIM::ModelComputeArgumentsCreate * const argsCreate)
{
    int error = 0;
    // Energy, forces, virial are all optional
    error |= argsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
        KIM::SUPPORT_STATUS::optional);
    error |= argsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialForces,
        KIM::SUPPORT_STATUS::optional);
    error |= argsCreate->SetArgumentSupportStatus(
        KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
        KIM::SUPPORT_STATUS::optional);

    // GetNeighborList is required (standard for short-range potentials)
    error |= argsCreate->SetCallbackSupportStatus(
        KIM::COMPUTE_CALLBACK_NAME::GetNeighborList,
        KIM::SUPPORT_STATUS::requiredByAPI);
    error |= argsCreate->SetCallbackSupportStatus(
        KIM::COMPUTE_CALLBACK_NAME::ProcessDEDrTerm,
        KIM::SUPPORT_STATUS::notSupported);
    error |= argsCreate->SetCallbackSupportStatus(
        KIM::COMPUTE_CALLBACK_NAME::ProcessD2EDr2Term,
        KIM::SUPPORT_STATUS::notSupported);

    return error != 0;
}

// =========================================================================
// Compute — the main EFS calculation
// =========================================================================
extern "C" int
MTPDriver_Compute(
    KIM::ModelCompute const *modelCompute,
    KIM::ModelComputeArguments const *args)
{
    // ── Get model ──────────────────────────────────────────────────────
    MTPModel *model = nullptr;
    modelCompute->GetModelBufferPointer((void **)&model);
    if (!model) return true;
    auto &core = model->core;
    const double lengthToInternal = model->lengthToInternal;

    // ── Get argument pointers ─────────────────────────────────────────
    int const *numberOfParticlesPointer = nullptr;
    args->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles,
        &numberOfParticlesPointer);

    int const *speciesCodes = nullptr;
    args->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes,
        &speciesCodes);

    int const *contributing = nullptr;
    args->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::particleContributing,
        &contributing);

    double const *coordinates = nullptr;
    args->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::coordinates,
        &coordinates);

    double *energy = nullptr;
    args->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialEnergy,
        &energy);

    double *forces = nullptr;
    args->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialForces,
        &forces);

    double *virial = nullptr;
    args->GetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialVirial,
        &virial);

    if (!numberOfParticlesPointer || !speciesCodes || !coordinates) return true;
    const int numberOfParticles = *numberOfParticlesPointer;
    if (numberOfParticles < 0) return true;
    for (int i = 0; i < numberOfParticles; ++i)
        if (speciesCodes[i] < 0 || speciesCodes[i] >= core.speciesCount())
            return true;

    // ── Zero outputs ──────────────────────────────────────────────────
    double energySum = 0.0;
    if (forces)
        for (int i = 0; i < numberOfParticles * 3; ++i) forces[i] = 0.0;
    if (virial)
        for (int i = 0; i < 6; ++i) virial[i] = 0.0;

    // ── Per-atom buffers (reused across atoms) ────────────────────────
    std::vector<double> relPosX, relPosY, relPosZ;
    std::vector<double> dists;
    std::vector<int> jtypes;
    std::vector<int> neighborAtoms;
    std::vector<double> ders;
    std::vector<mtp_kim::Vec3> relPos;
    mtp_kim::MTPCore::ThreadBuffer buffer;
    buffer.ensure(core.rb_size, core.max_mu, core.max_alpha_index_basic,
                  core.alpha_moments_count, 0);

    // ── Main loop over atoms ──────────────────────────────────────────
    for (int i = 0; i < numberOfParticles; ++i) {
        if (contributing && !contributing[i]) continue;
        int itype = speciesCodes[i];

        // Get neighbor list from simulator via KIM callback
        int numNeighbors = 0;
        int const *neighborIndices = nullptr;
        if (args->GetNeighborList(0, i, &numNeighbors, &neighborIndices))
            return true;
        if (numNeighbors < 0 || (numNeighbors > 0 && !neighborIndices))
            return true;

        // Build relative position arrays (filter by cutoff)
        relPosX.clear(); relPosY.clear(); relPosZ.clear();
        dists.clear(); jtypes.clear(); neighborAtoms.clear();

        for (int jj = 0; jj < numNeighbors; ++jj) {
            int j = neighborIndices[jj];
            if (j < 0 || j >= numberOfParticles) return true;
            if (j == i) continue;
            int jtype = speciesCodes[j];
            double dx = (coordinates[3*j] - coordinates[3*i]) * lengthToInternal;
            double dy = (coordinates[3*j+1] - coordinates[3*i+1]) * lengthToInternal;
            double dz = (coordinates[3*j+2] - coordinates[3*i+2]) * lengthToInternal;
            double r = std::sqrt(dx*dx + dy*dy + dz*dz);

            int pairIdx = itype * core.speciesCount() + jtype;
            if (r >= core.cutoffs[pairIdx] || r <= core.r_mins[pairIdx]) continue;

            relPosX.push_back(dx); relPosY.push_back(dy); relPosZ.push_back(dz);
            dists.push_back(r);
            jtypes.push_back(jtype);
            neighborAtoms.push_back(j);
        }

        int nValid = (int)dists.size();
        if (nValid == 0) {
            energySum += core.species_coeffs[itype] * model->energyFromInternal;
            continue;
        }

        // Build Vec3 array
        relPos.resize(nValid);
        for (int jj = 0; jj < nValid; ++jj)
            relPos[jj] = {relPosX[jj], relPosY[jj], relPosZ[jj]};

        // Compute site energy and derivatives
        ders.assign(nValid * 3, 0.0);
        double siteE = 0.0;
        core.calcSiteEFS(itype, nValid, relPos.data(), dists.data(),
                         jtypes.data(), siteE, ders.data(), buffer);
        energySum += siteE * model->energyFromInternal;

        // Accumulate forces (Newton's 3rd law: f_i += dE/dx_j, f_j -= dE/dx_j)
        if (forces || virial) {
            for (int jj = 0; jj < nValid; ++jj) {
                int j = neighborAtoms[jj];
                const double fx = ders[jj*3 + 0] * model->forceFromInternal;
                const double fy = ders[jj*3 + 1] * model->forceFromInternal;
                const double fz = ders[jj*3 + 2] * model->forceFromInternal;
                if (forces) {
                    forces[3*i + 0] += fx;
                    forces[3*i + 1] += fy;
                    forces[3*i + 2] += fz;
                    forces[3*j + 0] -= fx;
                    forces[3*j + 1] -= fy;
                    forces[3*j + 2] -= fz;
                }

                if (virial) {
                    const double scale = model->energyFromInternal;
                    virial[0] += scale * ders[jj*3 + 0] * relPos[jj].x;
                    virial[1] += scale * ders[jj*3 + 1] * relPos[jj].y;
                    virial[2] += scale * ders[jj*3 + 2] * relPos[jj].z;
                    virial[3] += scale * ders[jj*3 + 1] * relPos[jj].z;
                    virial[4] += scale * ders[jj*3 + 2] * relPos[jj].x;
                    virial[5] += scale * ders[jj*3 + 0] * relPos[jj].y;
                }
            }
        }
    }

    if (energy) *energy = energySum;
    return false;
}

// =========================================================================
// ComputeArgumentsDestroy
// =========================================================================
extern "C" int
MTPDriver_ComputeArgumentsDestroy(
    KIM::ModelCompute const *,
    KIM::ModelComputeArgumentsDestroy *)
{
    return false;
}

// =========================================================================
// Destroy
// =========================================================================
extern "C" int
MTPDriver_Destroy(KIM::ModelDestroy * const modelDestroy)
{
    MTPModel *model = nullptr;
    modelDestroy->GetModelBufferPointer((void **)&model);
    delete model;
    return false;
}
