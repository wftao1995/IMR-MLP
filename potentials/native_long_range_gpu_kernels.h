#pragma once

#include <cstddef>
#include <memory>

namespace Potentials {

// POD-only boundary: the device translation unit does not include Eigen or
// the trainer's C++23 model classes. All arrays passed here reside on host.
class NativeLongRangeGpu {
public:
    explicit NativeLongRangeGpu(int device);
    ~NativeLongRangeGpu();
    NativeLongRangeGpu(const NativeLongRangeGpu &) = delete;
    NativeLongRangeGpu &operator=(const NativeLongRangeGpu &) = delete;

    void atoms(int owned, int total, const double *positions, const double *charges);
    void charges(int total, const double *charges);
    // Every wave is kx,ky,kz,h,h',reserved; h' differentiates k^2.
    void waves(int count, const double *waves);
    void structure(double *real_imag);
    void reciprocal(const double *global_real_imag, double prefactor);
    void les_self(double coefficient);
    void response(double *phi);
    void outputs(double *energy, double *force, double *virial);
private:
    struct State;
    std::unique_ptr<State> state_;
};

} // namespace Potentials
