#pragma once
#include "potentials/potential.h"
#include <exception>

// Exactly one OpenMP level: independent images, or the potential's internal
// parallel evaluator. MPI calls belong to the caller, outside this function.
inline void evaluate_neb_images(Potentials::AnyPotential &potential,
                                std::vector<Atoms> &images, int begin, int end,
                                const std::string &mode, int rank = 0, int ranks = 1) {
    if (mode != "images" && mode != "atoms")
        throw std::invalid_argument("NEB: omp_mode must be images or atoms");
    std::vector<std::exception_ptr> errors(end-begin);
    auto evaluate = [&](int i) {
        const auto positions = images[i].pos_;
        try {
            images[i].wrap();
            if (mode == "atoms") potential.getEFSOMP(images[i]);
            else potential.getEFSOMPthread(images[i]);
            if (images[i].forces_.size() != images[i].size())
                throw std::runtime_error("NEB calculator returned an invalid force array");
        } catch (...) { errors[i-begin] = std::current_exception(); }
        images[i].pos_ = positions;
    };
    if (mode == "images") {
        #pragma omp parallel for schedule(static)
        for (int i = begin+rank; i < end; i += ranks) evaluate(i);
    } else {
        for (int i = begin+rank; i < end; i += ranks) evaluate(i);
    }
    for (const auto &error : errors) if (error) std::rethrow_exception(error);
}
