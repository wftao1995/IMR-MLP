#pragma once

#if __has_include("../mlip.h")
#include "../mlip.h"
#else
#include "mlip.h"
#endif
#include <array>
#include <cstddef>
#include <memory>
#include <vector>

namespace MLIPCore::Cpu {

enum class DerivativeMode { Auto, Cached, Recompute };
DerivativeMode derivative_mode(const char *name);
// Detect the extended coefficient section without narrowing the legacy opt
// reader's supported radial families. The full reader performs validation.
bool has_model_coefficients(const char *filename);

struct ModelDeleter {
    void operator()(mlip *model) const { if (model) destroy_mlip(model); }
};
using Model = std::unique_ptr<mlip, ModelDeleter>;

// One workspace per concurrently executing center/thread. reserve() is called
// outside a threaded region; evaluate_site() never grows an allocation.
class Workspace {
public:
    explicit Workspace(const mlip &model);
    Workspace(const Workspace &) = delete;
    Workspace &operator=(const Workspace &) = delete;
    void reserve(const mlip &model, std::size_t neighbors, DerivativeMode mode);
    std::size_t capacity() const { return neighbors_.size(); }
    std::size_t allocated_bytes() const;
    DerivativeMode selected_mode() const { return mode_; }
    int edge_count() const { return edge_count_; }
    int neighbor(int edge) const { return neighbors_[edge]; }
    const double *displacement(int edge) const { return displacements_[edge].data(); }
    const double *gradient(int edge) const { return gradients_[edge].data(); }

private:
    friend class Executor;
    struct BufferDeleter {
        void operator()(mlip_buffers *buffer) const { if (buffer) destroy_mlip_buffer(buffer); }
    };
    std::unique_ptr<mlip_buffers, BufferDeleter> buffer_;
    std::vector<int> neighbors_;
    std::vector<std::array<double, 3>> displacements_, gradients_;
    std::vector<double> jacobian_, q_, dq_, hidden1_, hidden2_, derivative1_;
    DerivativeMode mode_ = DerivativeMode::Recompute;
    int edge_count_ = 0;
    std::size_t fixed_bytes_ = 0;
};

// The model must outlive its executor/workspaces. Geometry and mappings are
// borrowed. Neighbors may include the two LAMMPS special-bond index bits;
// coordinates already contain the intended periodic ghost images.
class Executor {
public:
    explicit Executor(const mlip &model);
    double evaluate_site(int center, int neighbor_count, const int *neighbors,
                         const int *types, const double *const *positions,
                         const int *type_to_species, bool energy, bool derivatives,
                         Workspace &workspace) const;
    const mlip &model() const { return model_; }

private:
    struct ProductGroup { int begin, end, output; bool assign; };
    const mlip &model_;
    std::vector<ProductGroup> groups_;
    std::vector<int> scalar_ids_;
    std::vector<double> projection_, linear_species_;
    bool linear_ = false;
    bool accepts(int pair, double squared_distance) const;
    double radial(int pair, const double *r, mlip_buffers &buffer) const;
    void basic(const double *r, double inverse_distance, mlip_buffers &buffer, double *jacobian,
               double *gradient, bool accumulate) const;
    double head(Workspace &workspace, bool energy, bool derivatives) const;
};

} // namespace MLIPCore::Cpu
