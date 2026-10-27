#pragma once

#include "base/structures.h"
#include "io/gpio.h"
#include "tools/ti_harmonic_reference.h"

#include <string>
#include <vector>

namespace TIReferenceApp {

struct OutputOptions {
    std::string report_file = "TI_REFERENCE.dat";
    double temperature_kelvin = 300.0;
    int ensemble_size = 0;
    std::string ensemble_file = "TI_ENSEMBLE.xyz";
    std::string ensemble_format = "extxyz";
    int seed = 5489;
};

std::vector<TIHarmonicReference::SymmetryOperation> symmetry_operations(
    Atoms &supercell, double symprec);

double maximum_force(const Atoms &atoms);

void write_outputs(const TIHarmonicReference::ReferenceResult &reference,
                   const TIHarmonicReference::Options &reference_options,
                   const OutputOptions &output_options,
                   Atoms &supercell,
                   GPIO &writer,
                   double maximum_reference_force,
                   int symmetry_operation_count);

} // namespace TIReferenceApp
