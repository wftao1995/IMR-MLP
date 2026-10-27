#pragma once
#include <memory>

class Atoms;
class AnyNeighborhoods;
namespace Potentials {
class AnyPotential;
class PotMTP;
class LES;
struct NativeCollectives;
struct NativeInferenceResult;
struct NativeInferenceRequest;
struct NativePackedNeighborhoodView;
struct NativeInferenceStatistics;

class NativeGpuInference {
public:
    NativeGpuInference(const AnyPotential &,int device,const PotMTP *latent_model,
        const LES *reciprocal,const double *species,const double *moments,
        int channels,int parity);
    ~NativeGpuInference();
    NativeInferenceResult evaluate(Atoms &,AnyNeighborhoods &,const NativeCollectives &,const NativeInferenceRequest &);
    NativeInferenceResult evaluate_packed(Atoms &,const NativePackedNeighborhoodView &,const NativeCollectives &,const NativeInferenceRequest &);
    NativeInferenceStatistics statistics() const;
private:
    friend class NativeInference;
    // Only NativeInference's owned candidate builder may use this path.
    // It is consumed within the same call, after collective validation.
    NativeInferenceResult evaluate_prepared(Atoms &,const NativePackedNeighborhoodView &,
        const NativeCollectives &,const NativeInferenceRequest &);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
