// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_openkim.h"
#include <iomanip>
#include <cmath>

#ifdef USE_KIM
#include <kim-api/KIM_SupportedExtensions.hpp>

// ── Helper: KIM species name string → enum ──────────────────────────────
static KIM::SpeciesName strToSpeciesName(const std::string &s) {
    int n; KIM::SPECIES_NAME::GetNumberOfSpeciesNames(&n);
    for (int i = 0; i < n; ++i) {
        KIM::SpeciesName sn;
        KIM::SPECIES_NAME::GetSpeciesName(i, &sn);
        if (sn.ToString() == s) return sn;
    }
    // If not found as a standard name, try to create as a custom species string.
    // KIM API v2 supports arbitrary species via the SpeciesName(string) constructor.
    return KIM::SpeciesName(s);
}

// ── QueryModelSpecies: auto-detect supported species ─────────────────────
std::map<int, std::string> Potentials::PotKIM::QueryModelSpecies(const std::string &modelname) {
    std::map<int, std::string> result;
    int unitsAccepted;
    KIM::Model *tmpModel = nullptr;
    int err = KIM::Model::Create(
        KIM::NUMBERING::zeroBased,
        KIM::LENGTH_UNIT::A,
        KIM::ENERGY_UNIT::eV,
        KIM::CHARGE_UNIT::e,
        KIM::TEMPERATURE_UNIT::K,
        KIM::TIME_UNIT::ps,
        modelname,
        &unitsAccepted,
        &tmpModel);
    if (err || !tmpModel) {
        std::cerr << "QueryModelSpecies: unable to create temporary model \"" << modelname << "\"" << std::endl;
        return result;
    }

    int nSpecies;
    KIM::SPECIES_NAME::GetNumberOfSpeciesNames(&nSpecies);
    int nextId = 0;
    for (int i = 0; i < nSpecies; ++i) {
        KIM::SpeciesName sn;
        KIM::SPECIES_NAME::GetSpeciesName(i, &sn);
        int supported, code;
        tmpModel->GetSpeciesSupportAndCode(sn, &supported, &code);
        if (supported) {
            result[nextId] = sn.ToString();
            ++nextId;
        }
    }
    KIM::Model::Destroy(&tmpModel);
    return result;
}

// ── Constructor ─────────────────────────────────────────────────────────
Potentials::PotKIM::PotKIM(const std::string &modelname, const std::map<int, std::string> &typemap)
    : kim_modelname(modelname), kim_Model(nullptr), computeArguments(nullptr),
      n_atoms(0), energy_val(0.0)
{
    int requested_units_accepted;
    int error = KIM::Model::Create(
        KIM::NUMBERING::zeroBased,
        KIM::LENGTH_UNIT::A,
        KIM::ENERGY_UNIT::eV,
        KIM::CHARGE_UNIT::e,
        KIM::TEMPERATURE_UNIT::K,
        KIM::TIME_UNIT::ps,
        modelname,
        &requested_units_accepted,
        &kim_Model);
    if (error || !kim_Model) {
        GPERROR("Can't create KIM Model \"" + modelname + "\"");
    }
    if (!requested_units_accepted) {
        std::cerr << "Warning: KIM model \"" << modelname
                  << "\" does not accept requested units (A, eV). "
                  << "Will attempt to use model's native units." << std::endl;
    }

    // Check that we know about all required routines
    int numberOfModelRoutineNames;
    KIM::MODEL_ROUTINE_NAME::GetNumberOfModelRoutineNames(&numberOfModelRoutineNames);
    for (int i = 0; i < numberOfModelRoutineNames; ++i) {
        KIM::ModelRoutineName modelRoutineName;
        int err = KIM::MODEL_ROUTINE_NAME::GetModelRoutineName(i, &modelRoutineName);
        if (err) { GPERROR("Unable to get ModelRoutineName."); }
        int present, required;
        err = kim_Model->IsRoutinePresent(modelRoutineName, &present, &required);
        if (err) { GPERROR("Unable to get routine present/required."); }

        std::cout << "Model routine name \"" << modelRoutineName.ToString()
                  << "\" has present = " << present
                  << " and required = " << required << "." << std::endl;

        if ((present == true) && (required == true)) {
            using namespace KIM::MODEL_ROUTINE_NAME;
            if (!((modelRoutineName == Create) || (modelRoutineName == ComputeArgumentsCreate)
                  || (modelRoutineName == Compute) || (modelRoutineName == Refresh)
                  || (modelRoutineName == ComputeArgumentsDestroy) || (modelRoutineName == Destroy))) {
                GPERROR("Unknown Routine \"" + modelRoutineName.ToString() + "\" is required by model.");
            }
        }
    }

    kim_Model->GetUnits(&lengthUnit, &energyUnit, &chargeUnit, &temperatureUnit, &timeUnit);
    std::cout << "LengthUnit is \"" << lengthUnit.ToString() << "\"" << std::endl
              << "EnergyUnit is \"" << energyUnit.ToString() << "\"" << std::endl
              << "ChargeUnit is \"" << chargeUnit.ToString() << "\"" << std::endl
              << "TemperatureUnit is \"" << temperatureUnit.ToString() << "\"" << std::endl
              << "TimeUnit is \"" << timeUnit.ToString() << "\"" << std::endl;

    refresh_neighbor_metadata();

    // Check species — auto-detect if typemap is empty
    std::map<int, std::string> effective_typemap = typemap;
    if (effective_typemap.empty()) {
        std::cout << "No species map provided, auto-detecting from KIM model..." << std::endl;
        int nSpecies;
        KIM::SPECIES_NAME::GetNumberOfSpeciesNames(&nSpecies);
        int nextId = 0;
        for (int i = 0; i < nSpecies; ++i) {
            KIM::SpeciesName sn;
            KIM::SPECIES_NAME::GetSpeciesName(i, &sn);
            int supported, code;
            kim_Model->GetSpeciesSupportAndCode(sn, &supported, &code);
            // Only include species that are both supported AND have a standard chemical symbol
            if (supported) {
                std::string sym = sn.ToString();
                // Filter out non-standard user species names like "user01", "user02", etc.
                // and the special "electron" species
                if (sym == "electron" || sym.rfind("user", 0) == 0) continue;
                effective_typemap[nextId] = sym;
                std::cout << "  Auto-mapped id " << nextId << " -> " << sym << std::endl;
                ++nextId;
            }
        }
        if (effective_typemap.empty()) {
            GPERROR("Failed to auto-detect any species from KIM model \"" + modelname
                    + "\". Please provide an index.map file.");
        }
    }
    gpm_elements=effective_typemap;species_count=effective_typemap.size();
    model_code.resize(effective_typemap.size());
    for (auto &it : effective_typemap) {
        int speciesIsSupported;
        KIM::SpeciesName sn = strToSpeciesName(it.second);
        error = kim_Model->GetSpeciesSupportAndCode(sn, &speciesIsSupported, &model_code[it.first]);
        if ((error) || (!speciesIsSupported)) {
            GPERROR("Species " + it.second + " not supported by KIM model \"" + modelname + "\"");
        }
        std::cout << "Species \"" << it.second << "\" -> KIM code " << model_code[it.first] << std::endl;
    }

    error = kim_Model->ComputeArgumentsCreate(&computeArguments);
    if (error) {
        GPERROR("Unable to create a ComputeArguments object.");
    }

    // Check compute arguments
    int numberOfComputeArgumentNames;
    KIM::COMPUTE_ARGUMENT_NAME::GetNumberOfComputeArgumentNames(&numberOfComputeArgumentNames);
    for (int i = 0; i < numberOfComputeArgumentNames; ++i) {
        KIM::ComputeArgumentName computeArgumentName;
        KIM::SupportStatus supportStatus;
        KIM::COMPUTE_ARGUMENT_NAME::GetComputeArgumentName(i, &computeArgumentName);
        KIM::DataType dataType;
        KIM::COMPUTE_ARGUMENT_NAME::GetComputeArgumentDataType(computeArgumentName, &dataType);
        error = computeArguments->GetArgumentSupportStatus(computeArgumentName, &supportStatus);
        if (error) GPERROR("unable to get ComputeArgument SupportStatus");

        std::cout << "ComputeArgument Name \"" << computeArgumentName.ToString() << "\""
                  << " is of type \"" << dataType.ToString() << "\""
                  << " and has supportStatus \"" << supportStatus.ToString() << "\"" << std::endl;

        // Can only handle energy and force as a required arg
        if (supportStatus == KIM::SUPPORT_STATUS::required) {
            if ((computeArgumentName != KIM::COMPUTE_ARGUMENT_NAME::partialEnergy)
                && (computeArgumentName != KIM::COMPUTE_ARGUMENT_NAME::partialForces)
                && (computeArgumentName != KIM::COMPUTE_ARGUMENT_NAME::partialVirial)) {
                GPERROR("unsupported required ComputeArgument: " + computeArgumentName.ToString());
            }
        }

        // Must have energy and forces
        if ((computeArgumentName == KIM::COMPUTE_ARGUMENT_NAME::partialEnergy)
            || (computeArgumentName == KIM::COMPUTE_ARGUMENT_NAME::partialForces)) {
            if (!((supportStatus == KIM::SUPPORT_STATUS::required)
                  || (supportStatus == KIM::SUPPORT_STATUS::optional))) {
                GPERROR("energy or forces not available from KIM model");
            }
        }

        if (computeArgumentName == KIM::COMPUTE_ARGUMENT_NAME::partialEnergy) {
            kim_support_pe = supportStatus;
        }
        if (computeArgumentName == KIM::COMPUTE_ARGUMENT_NAME::partialVirial) {
            kim_support_pv = supportStatus;
        }
    }

    // Check compute callbacks
    int numberOfComputeCallbackNames;
    KIM::COMPUTE_CALLBACK_NAME::GetNumberOfComputeCallbackNames(&numberOfComputeCallbackNames);
    for (int i = 0; i < numberOfComputeCallbackNames; ++i) {
        KIM::ComputeCallbackName computeCallbackName;
        KIM::COMPUTE_CALLBACK_NAME::GetComputeCallbackName(i, &computeCallbackName);
        KIM::SupportStatus supportStatus;
        computeArguments->GetCallbackSupportStatus(computeCallbackName, &supportStatus);

        std::cout << "ComputeCallback Name \"" << computeCallbackName.ToString() << "\""
                  << " has supportStatus \"" << supportStatus.ToString() << "\"" << std::endl;

        // Cannot handle any "required" callbacks other than GetNeighborList
        if (supportStatus == KIM::SUPPORT_STATUS::required) {
            if (computeCallbackName != KIM::COMPUTE_CALLBACK_NAME::GetNeighborList) {
                GPERROR("unsupported required ComputeCallback: " + computeCallbackName.ToString());
            }
        }
    }

    int numberOfParameters;
    kim_Model->GetNumberOfParameters(&numberOfParameters);
    for (int i = 0; i < numberOfParameters; ++i) {
        KIM::DataType type;
        std::string const *name, *description;
        int extent;
        if (kim_Model->GetParameterMetadata(i, &type, &extent, &name, &description) || extent <= 0)
            GPERROR("KIM: invalid parameter metadata");
        ModelParameter value;
        value.index = i;
        value.extent = extent;
        if (type == KIM::DATA_TYPE::Integer) value.integers.resize(extent);
        else if (type == KIM::DATA_TYPE::Double) value.doubles.resize(extent);
        else GPERROR("KIM: unsupported parameter data type for " + *name);
        auto [entry, inserted] = model_parameters.emplace(*name, std::move(value));
        if (!inserted) GPERROR("KIM: duplicate parameter name " + *name);
        const auto key = "kim_parameter:" + *name;
        if (!entry->second.integers.empty()) Connect(entry->second.integers, key, *description, extent);
        else Connect(entry->second.doubles, key, *description, extent);
    }
    read_parameter_values();

    // Check supported extensions, if any
    int present, requiredExt;
    error = kim_Model->IsRoutinePresent(KIM::MODEL_ROUTINE_NAME::Extension, &present, &requiredExt);
    if (error) {
        GPERROR("Unable to get Extension present/required.");
    }
    if (present) {
        KIM::SupportedExtensions supportedExtensions;
        error = kim_Model->Extension(KIM_SUPPORTED_EXTENSIONS_ID, &supportedExtensions);
        if (error) {
            GPERROR("Error returned from KIM::Model::Extension().");
        }
        std::cout << "Model Supports " << supportedExtensions.numberOfSupportedExtensions
                  << " Extensions:" << std::endl;
        for (int i = 0; i < supportedExtensions.numberOfSupportedExtensions; ++i) {
            std::cout << " supportedExtensionID[" << std::setw(2) << i << "] = \""
                      << supportedExtensions.supportedExtensionID[i] << "\" "
                      << "which has required = "
                      << supportedExtensions.supportedExtensionRequired[i] << "." << std::endl;
        }
    }

    // Initialize omp buffers
    omp_buffers.resize(omp_num_threads);
}

// Refresh invalidates the model's internal pointers; always copy them again.
void Potentials::PotKIM::refresh_neighbor_metadata() {
    kim_Model->GetInfluenceDistance(&model_cutoff);
    double const *cuts = nullptr;
    int const *requests = nullptr;
    kim_Model->GetNeighborListPointers(&n_neigh_lists, &cuts, &requests);
    if (!std::isfinite(model_cutoff) || model_cutoff < 0 || n_neigh_lists < 0 ||
        (n_neigh_lists > 0 && (!cuts || !requests)))
        GPERROR("KIM: invalid influence distance or neighbor-list metadata");
    neigh_cutoffs.clear();
    modelWillNotRequestNeighborsOfNoncontributing.clear();
    if (n_neigh_lists > 0) {
        neigh_cutoffs.assign(cuts, cuts + n_neigh_lists);
        modelWillNotRequestNeighborsOfNoncontributing.assign(requests, requests + n_neigh_lists);
        for (double value : neigh_cutoffs) {
            if (!std::isfinite(value) || value < 0) GPERROR("KIM: invalid neighbor-list cutoff");
            model_cutoff = std::max(model_cutoff, value);
        }
    }
    cutoff = model_cutoff;
    cutoffsq = cutoff * cutoff;
}

void Potentials::PotKIM::read_parameter_values() {
    auto current = model_parameters;
    for (auto &[name, value] : current) {
        for (int j = 0; j < value.extent; ++j) {
            const int error = value.integers.empty()
                ? kim_Model->GetParameter(value.index, j, &value.doubles[j])
                : kim_Model->GetParameter(value.index, j, &value.integers[j]);
            if (error) GPERROR("KIM: unable to read parameter " + name);
        }
    }
    for (auto &[name, value] : current) {
        model_parameters.at(name).integers = std::move(value.integers);
        model_parameters.at(name).doubles = std::move(value.doubles);
    }
}

void Potentials::PotKIM::ApplyParameterSettings(const GPSettings &settings) {
    bool changed = false;
    for (const auto &[key, text] : settings.settings) {
        if (!key.starts_with("kim_parameter:")) continue;
        const auto name = key.substr(std::string("kim_parameter:").size());
        if (!model_parameters.count(name)) GPERROR("KIM: unknown model parameter " + name);
        if (text.empty()) GPERROR("KIM: parameter array cannot be empty: " + name);
        changed = true;
    }
    if (!changed) return; // Parameter-free models must not call ClearThenRefresh.
    auto proposed = model_parameters;
    ConnectSettings parser;
    for (auto &[name, value] : proposed) {
        if (!value.integers.empty()) parser.Connect(value.integers, "kim_parameter:" + name, "", value.extent);
        else parser.Connect(value.doubles, "kim_parameter:" + name, "", value.extent);
    }
    parser.ApplySettings(settings); // Validate the complete request before touching KIM.
    for (const auto &[name, value] : proposed)
        for (double number : value.doubles)
            if (!std::isfinite(number)) GPERROR("KIM: nonfinite parameter " + name);
    auto write_values = [&](const auto &values) {
        for (const auto &[name, value] : values)
            for (int j = 0; j < value.extent; ++j) {
                const int error = value.integers.empty()
                    ? kim_Model->SetParameter(value.index, j, value.doubles[j])
                    : kim_Model->SetParameter(value.index, j, value.integers[j]);
                if (error) GPERROR("KIM: unable to set parameter " + name);
            }
    };
    try {
        write_values(proposed);
        if (kim_Model->ClearThenRefresh()) GPERROR("KIM: parameter refresh failed");
        refresh_neighbor_metadata();
        read_parameter_values();
    } catch (...) {
        // Keep connected storage stable: restore KIM from the untouched cache.
        write_values(model_parameters);
        if (kim_Model->ClearThenRefresh()) GPERROR("KIM: parameter rollback refresh failed");
        refresh_neighbor_metadata();
        throw;
    }
}

// ── Destructor ───────────────────────────────────────────────────────────
Potentials::PotKIM::~PotKIM() {
    if (computeArguments) {
        kim_Model->ComputeArgumentsDestroy(&computeArguments);
        computeArguments = nullptr;
    }
    if (kim_Model) {
        KIM::Model::Destroy(&kim_Model);
        kim_Model = nullptr;
    }
}

// ── GetNeighborListCallback ──────────────────────────────────────────────
int Potentials::PotKIM::GetNeighborListCallback(void *dataObject,
                                                 int /*numberOfNeighborLists*/,
                                                 double const * /*cutoffs*/,
                                                 int neighborListIndex,
                                                 int particleNumber,
                                                 int *numberOfNeighbors,
                                                 int const **neighborsOfParticle) {
    auto *self = static_cast<PotKIM *>(dataObject);
    // neigh_offsets is sized (n_atoms * n_neigh_lists + 1) with prefix-sum offsets
    int flatIdx = particleNumber * self->n_neigh_lists + neighborListIndex;
    *numberOfNeighbors = self->neigh_offsets[flatIdx + 1] - self->neigh_offsets[flatIdx];
    *neighborsOfParticle = self->neigh_list_flat.data() + self->neigh_offsets[flatIdx];
    return false; // false = no error in KIM API convention
}

// ── getEFS ───────────────────────────────────────────────────────────────
void Potentials::PotKIM::getEFS(Atoms &atom) {
    const int nat = atom.size();

    // Reset or accumulate
    if (!add_efs) atom.resetEFS();
    else atom.setEFSflag();

    // Resize buffers if atom count changed
    if (nat != n_atoms) {
        n_atoms = nat;
        coords_flat.resize(3 * nat);
        forces_flat.resize(3 * nat);
        virial_flat.assign(6, 0.0);
    }

    // ── 1. Fill input buffers ─────────────────────────────────────────
    std::vector<int> speciesCodes(nat);
    std::vector<int> particleContributing(nat, 1);
    for (int i = 0; i < nat; ++i) {
        speciesCodes[i] = model_code[atom.chemmap_[i]];
        coords_flat[3 * i + 0] = atom.pos_[i].x();
        coords_flat[3 * i + 1] = atom.pos_[i].y();
        coords_flat[3 * i + 2] = atom.pos_[i].z();
    }

    // ── 2. Build neighbor list ────────────────────────────────────────
    // Use our own neighborhood construction (independent of AnyNeighborhoods parameter)
    Neighborhoods nbhs(atom, model_cutoff);

    // Flatten neighborhoods into KIM-compatible flat arrays
    // neigh_offsets[flat_idx] = start in neigh_list_flat
    // flat_idx = particleNumber * n_neigh_lists + neighborListIndex
    if (n_neigh_lists > 0) {
        neigh_offsets.assign(nat * n_neigh_lists + 1, 0);

        // First pass: count neighbors per (particle, list)
        for (int p = 0; p < nat; ++p) {
            auto &nbh = nbhs[p];
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                int neighAtom = nbh.indices_[j];
                if (neighAtom == p) continue; // exclude self
                double dist = nbh.distances_[j];
                for (int l = 0; l < n_neigh_lists; ++l) {
                    if (dist <= neigh_cutoffs[l]) {
                        neigh_offsets[p * n_neigh_lists + l]++;
                    }
                }
            }
        }

        // Prefix sum to get offsets
        int total = 0;
        for (int i = 0; i < nat * n_neigh_lists; ++i) {
            int cnt = neigh_offsets[i];
            neigh_offsets[i] = total;
            total += cnt;
        }
        neigh_offsets[nat * n_neigh_lists] = total;
        neigh_list_flat.resize(total);

        // Second pass: fill neighbor indices
        std::vector<int> cursor = neigh_offsets; // copy current offsets as write cursors
        for (int p = 0; p < nat; ++p) {
            auto &nbh = nbhs[p];
            for (int j = 0; j < nbh.nbhcount_; ++j) {
                int neighAtom = nbh.indices_[j];
                if (neighAtom == p) continue;
                double dist = nbh.distances_[j];
                for (int l = 0; l < n_neigh_lists; ++l) {
                    if (dist <= neigh_cutoffs[l]) {
                        int flatIdx = p * n_neigh_lists + l;
                        neigh_list_flat[cursor[flatIdx]++] = neighAtom;
                    }
                }
            }
        }
    }

    // ── 3. Set compute argument pointers ──────────────────────────────
    int nat_local = nat; // non-const for SetArgumentPointer
    energy_val = 0.0;
    std::fill(forces_flat.begin(), forces_flat.end(), 0.0);
    std::fill(virial_flat.begin(), virial_flat.end(), 0.0);

    int err = 0;
    err = err || computeArguments->SetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::numberOfParticles, &nat_local);
    err = err || computeArguments->SetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::particleSpeciesCodes, speciesCodes.data());
    err = err || computeArguments->SetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::particleContributing, particleContributing.data());
    err = err || computeArguments->SetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::coordinates, coords_flat.data());
    err = err || computeArguments->SetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialEnergy, &energy_val);
    err = err || computeArguments->SetArgumentPointer(
        KIM::COMPUTE_ARGUMENT_NAME::partialForces, forces_flat.data());

    // Set virial pointer if model supports it
    if (kim_support_pv == KIM::SUPPORT_STATUS::required
        || kim_support_pv == KIM::SUPPORT_STATUS::optional) {
        err = err || computeArguments->SetArgumentPointer(
            KIM::COMPUTE_ARGUMENT_NAME::partialVirial, virial_flat.data());
    }

    if (err) {
        GPERROR("KIM: failed to set compute argument pointers");
    }

    // ── 4. Set neighbor list callback ──────────────────────────────────
    err = computeArguments->SetCallbackPointer(
        KIM::COMPUTE_CALLBACK_NAME::GetNeighborList,
        KIM::LANGUAGE_NAME::cpp,
        reinterpret_cast<KIM::Function *>(GetNeighborListCallback),
        this);
    if (err) {
        GPERROR("KIM: failed to set GetNeighborList callback");
    }

    // ── 5. Verify all required arguments/callbacks are present ────────
    int ready = 0;
    computeArguments->AreAllRequiredArgumentsAndCallbacksPresent(&ready);
    if (!ready) {
        GPERROR("KIM: not all required arguments/callbacks are present");
    }

    // ── 6. Compute ────────────────────────────────────────────────────
    err = kim_Model->Compute(computeArguments);
    if (err) {
        GPERROR("KIM: Compute() failed for model \"" + kim_modelname + "\"");
    }

    // ── 7. Extract results ────────────────────────────────────────────
    atom.energy_ += energy_val;
    atom.have_energy = true;

    for (int i = 0; i < nat; ++i) {
        atom.forces_[i] += Eigen::Vector3d(
            forces_flat[3 * i], forces_flat[3 * i + 1], forces_flat[3 * i + 2]);
    }

    // KIM partialVirial is +dE/d(strain), ordered [xx, yy, zz, yz, zx, xy].
    // IMRMLP stores W = -dE/d(strain), in energy units (no volume division).
    if (kim_support_pv == KIM::SUPPORT_STATUS::required
        || kim_support_pv == KIM::SUPPORT_STATUS::optional) {
        // Expand KIM's symmetric tensor and convert its sign convention.
        Eigen::Matrix3d virial_tensor;
        virial_tensor << virial_flat[0], virial_flat[5], virial_flat[4],
                         virial_flat[5], virial_flat[1], virial_flat[3],
                         virial_flat[4], virial_flat[3], virial_flat[2];
        atom.stress_ -= virial_tensor;
        atom.have_stress = true;
    } else {
        // Fallback: model does not provide partialVirial.
        // Cannot compute correct stress from total forces for many-body potentials.
        // Leave stress at zero.
        atom.have_stress = false;
    }

    // Site energies: not available from basic KIM MO interface, set to zero
    atom.site_energies_.resize(nat, 0.0);
}

#endif
