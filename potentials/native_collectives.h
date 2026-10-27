#pragma once
#include <functional>
#include <cstddef>
#include <stdexcept>
#include <string>

namespace Potentials {
// LAMMPS owns MPI and halo topology; core inference never uses the trainer's
// global communicator. Empty callbacks denote a one-rank complete structure.
struct NativeCollectives {
    int rank = 0;
    int size = 1;
    std::function<void(double *, std::size_t)> sum;
    void allreduce(double *values, std::size_t count) const { if (sum) sum(values,count); }
    // Synchronize local validation/kernel failures before the next collective.
    // Operations passed here must themselves contain no MPI collective.
    template<class Operation> void stage(Operation &&operation) const {
        std::string message;
        try { operation(); }
        catch (const std::exception &error) { message=error.what(); }
        catch (...) { message="unknown native inference failure"; }
        double failed=message.empty()?0.:1.;
        allreduce(&failed,1);
        if(failed) throw std::runtime_error(message.empty()
            ? "mlip/native: inference failed on another MPI rank" : message);
    }
};
}
