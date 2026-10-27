#pragma once

#include "potential.h"
#include "native_collectives.h"
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Potentials {

struct NativeInferenceOptions {
    std::string pottype;
    std::string settings_file;
    std::string device = "cpu";
    int device_id = -1;
};

struct NativeInferenceRequest {
    bool forces = true;
    bool energy = true;
    bool site_energy = true;
    bool virial = true;
    bool site_virial = true;
    bool charges = true;
    bool latents = true;
};

// Borrowed CSR storage for callers that already own packed neighborhoods.
// Centers are the owned atoms [0, owned_count), in that order; neighbors
// address owned+ghost atom arrays. Geometry is refreshed on every call.
// Positions and types must agree with the accompanying Atoms. Vectors may
// be null only when positions[neighbor]-positions[center] is the intended
// image displacement (as with explicit ghosts). Null distances use its norm.
// A nonzero owner/revision pair promises unchanged centers, offsets,
// neighbors and types. Change revision for any topology/type change and
// owner when switching providers; zero disables topology caching. Neither
// token permits caching positions, displacements or computed predictions.
struct NativePackedNeighborhoodView {
    int owned_count = 0, total_count = 0, edge_count = 0;
    const int *centers = nullptr, *offsets = nullptr, *neighbors = nullptr, *types = nullptr;
    const double *positions = nullptr, *vectors = nullptr, *distances = nullptr;
    std::uintptr_t topology_owner = 0;
    std::uint64_t topology_revision = 0;
};

// Ordered skin candidates, with implicit owned centers [0, owned_count).
// Coordinates and species come from Atoms; explicit periodic images must be
// represented by ghost atoms. The evaluator filters the physical cutoff and
// owns the prepared geometry/CSR for the complete call. No caller-supplied
// revision or validation certificate can skip validation on this entry.
struct NativeCandidateNeighborhoodView {
    int owned_count = 0, total_count = 0, candidate_count = 0;
    const int *offsets = nullptr, *neighbors = nullptr;
};

struct NativeInferenceResult {
    double energy = 0.;
    SiteCoordinateMatrix forces;
    Eigen::Matrix3d virial = Eigen::Matrix3d::Zero();
    Eigen::VectorXd site_energy;
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> site_virial;
    Eigen::VectorXd charges;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> latents;
};

struct NativeInferenceStatistics {
    int owned_atoms=0,ghost_atoms=0;
    std::size_t neighbor_edges=0,model_device_bytes=0;
    std::uint64_t model_evaluations=0,force_kernel_launches=0;
    // Internal builder generation; zero for other entry points or after failure.
    std::uint64_t candidate_topology_revision=0;
};

// Owns the original trained model and all inference workspaces. Local calls
// use one neighborhood for each owned atom in [0, neighborhoods.size()).
// Neighbor indices address the supplied owned+ghost Atoms arrays. Outputs
// include ghost force/virial contributions; the caller performs reverse comm.
class NativeInference {
public:
    NativeInference(const std::string &model_file, const NativeInferenceOptions &options);
    ~NativeInference();
    NativeInference(const NativeInference &) = delete;
    NativeInference &operator=(const NativeInference &) = delete;

    int species_count() const;
    double cutoff() const;
    std::vector<std::string> elements() const;
    bool supports_local() const;
    bool has_charges() const;
    int latent_channels() const;
    std::string description() const;
    NativeInferenceStatistics statistics() const;
    AnyPotential &potential();

    NativeInferenceResult evaluate(Atoms &atoms, const NativeInferenceRequest &request = {});
    NativeInferenceResult evaluate_local(Atoms &atoms, AnyNeighborhoods &neighborhoods,
                                         const NativeInferenceRequest &request = {});
    NativeInferenceResult evaluate_distributed(Atoms &atoms, AnyNeighborhoods &neighborhoods,
        const NativeCollectives &collectives, const NativeInferenceRequest &request = {});
    NativeInferenceResult evaluate_distributed_packed(Atoms &atoms,
        const NativePackedNeighborhoodView &neighborhoods,
        const NativeCollectives &collectives, const NativeInferenceRequest &request = {});
    NativeInferenceResult evaluate_distributed_candidates(const Atoms &atoms,
        const NativeCandidateNeighborhoodView &candidates,
        const NativeCollectives &collectives, const NativeInferenceRequest &request = {});
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Potentials
