#include "apps/ti_reference_app.h"

#include "external/spglib_driver.h"
#include "tools/tools.h"
#include "utils/compressed_stream.h"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace TIReferenceApp {

std::vector<TIHarmonicReference::SymmetryOperation> symmetry_operations(
    Atoms &supercell, double symprec)
{
    auto dataset = Symmetry3D::get_sym_dataset(supercell, symprec, true);
    std::vector<TIHarmonicReference::SymmetryOperation> operations;
    operations.reserve(dataset.n_operations);
    const Eigen::Matrix3d inverse_cell = supercell.cell_.inverse();
    for (int iop = 0; iop < dataset.n_operations; ++iop) {
        TIHarmonicReference::SymmetryOperation operation;
        operation.atom_mapping.resize(supercell.size());
        for (int iat = 0; iat < supercell.size(); ++iat)
            operation.atom_mapping[iat] = dataset.symop_mapping(iop, iat);
        operation.cartesian_rotation = inverse_cell * dataset.symm_ops[iop].first * supercell.cell_;
        operations.push_back(std::move(operation));
    }
    return operations;
}

double maximum_force(const Atoms &atoms)
{
    double maximum = 0.0;
    for (const auto &force : atoms.forces_) maximum = std::max(maximum, force.norm());
    return maximum;
}

void write_outputs(const TIHarmonicReference::ReferenceResult &reference,
                   const TIHarmonicReference::Options &reference_options,
                   const OutputOptions &output_options,
                   Atoms &supercell,
                   GPIO &writer,
                   double maximum_reference_force,
                   int symmetry_operation_count)
{
    std::ofstream report(output_options.report_file);
    if (!report) throw std::runtime_error("Cannot open TI reference report: " + output_options.report_file);
    auto finish_report = [&] {
        report.close();
        if (!report) throw std::runtime_error("Failed to write TI reference report: " + output_options.report_file);
    };
    const auto &diagnostics = reference.diagnostics;
    report << "# ti_harmonic_reference\n";
    report << "policy " << TIHarmonicReference::policy_name(reference_options.stabilization) << '\n';
    report << "root_representation " << TIHarmonicReference::root_name(reference_options.root_representation) << '\n';
    report << "unstable_frequency_thz " << reference_options.unstable_frequency_thz << '\n';
    report << "positive_definite_floor_thz " << reference_options.positive_definite_floor_thz << '\n';
    report << "temperature_kelvin " << output_options.temperature_kelvin << '\n';
    report << "supercell_atoms " << supercell.size() << '\n';
    report << "internal_degrees_of_freedom " << reference.internal_dynamical_matrix.rows() << '\n';
    report << "space_group_operations " << symmetry_operation_count << '\n';
    report << "maximum_reference_force_ev_per_a " << maximum_reference_force << '\n';
    report << "corrected_modes " << diagnostics.corrected_modes << '\n';
    report << "transpose_error_before " << diagnostics.transpose_error_before << '\n';
    report << "space_group_error_before " << diagnostics.space_group_error_before << '\n';
    report << "space_group_error_after " << diagnostics.space_group_error_after << '\n';
    report << "asr_error_before " << diagnostics.acoustic_sum_rule_error_before << '\n';
    report << "asr_error_after " << diagnostics.acoustic_sum_rule_error_after << '\n';
    report << "relative_hessian_change " << diagnostics.relative_hessian_change << '\n';
    report << "condition_number_internal " << diagnostics.condition_number_internal << '\n';
    report << "log_determinant_internal " << diagnostics.log_determinant_internal << '\n';
    report << "optimizer_iterations " << diagnostics.optimizer_iterations << '\n';
    if (std::isfinite(diagnostics.force_rmse)) report << "force_rmse " << diagnostics.force_rmse << '\n';
    if (std::isfinite(diagnostics.delta_energy_mean)) {
        report << "delta_energy_mean " << diagnostics.delta_energy_mean << '\n';
        report << "delta_energy_stddev " << diagnostics.delta_energy_stddev << '\n';
    }
    report << "# mode eigenvalue_before frequency_before_thz eigenvalue_after frequency_after_thz\n";
    for (int i = 0; i < diagnostics.eigenvalues_after.size(); ++i)
        report << i << ' ' << diagnostics.eigenvalues_before(i) << ' '
               << TIHarmonicReference::signed_frequency_thz(diagnostics.eigenvalues_before(i)) << ' '
               << diagnostics.eigenvalues_after(i) << ' '
               << TIHarmonicReference::signed_frequency_thz(diagnostics.eigenvalues_after(i)) << '\n';

    if (output_options.ensemble_size == 0) {
        finish_report();
        return;
    }
    TIHarmonicReference::EnsembleOptions ensemble_options;
    ensemble_options.temperature_kelvin = output_options.temperature_kelvin;
    ensemble_options.size = output_options.ensemble_size;
    ensemble_options.seed = static_cast<std::uint32_t>(output_options.seed);
    const auto ensemble = TIHarmonicReference::sample_classical_ensemble(reference, ensemble_options);
    auto output = open_ofile(output_options.ensemble_file);
    if (!output) throw std::runtime_error("Cannot open TI ensemble output: " + output_options.ensemble_file);
    if (!writer.seq_write_init(*output, output_options.ensemble_format, output_options.ensemble_file))
        throw std::runtime_error("Cannot initialize TI ensemble writer: " + output_options.ensemble_file);

    double reference_minimum_distance = std::numeric_limits<double>::infinity();
    for (int i = 0; i < supercell.size(); ++i)
        for (int j = i + 1; j < supercell.size(); ++j)
            reference_minimum_distance = std::min(reference_minimum_distance, supercell.get_rab(i, j).norm());
    double sum_squared_displacement = 0.0;
    double maximum_displacement = 0.0;
    double ensemble_minimum_distance = std::numeric_limits<double>::infinity();
    for (int isample = 0; isample < static_cast<int>(ensemble.size()); ++isample) {
        Atoms generated = supercell;
        // This dataset defines harmonic total energies and forces only.
        // Reference stress, site partition and free energy are not recomputed.
        generated.have_stress = false;
        generated.site_energies_.clear();
        generated.have_fenergy = false;
        generated.forces_.resize(supercell.size());
        for (int iat = 0; iat < supercell.size(); ++iat) {
            const Eigen::Vector3d displacement = ensemble[isample].displacement_cartesian.segment<3>(3 * iat);
            generated.pos_[iat] += displacement.transpose();
            generated.forces_[iat] = ensemble[isample].reference_force_cartesian.segment<3>(3 * iat);
            sum_squared_displacement += displacement.squaredNorm();
            maximum_displacement = std::max(maximum_displacement, displacement.norm());
        }
        generated.wrap();
        for (int i = 0; i < generated.size(); ++i)
            for (int j = i + 1; j < generated.size(); ++j)
                ensemble_minimum_distance = std::min(ensemble_minimum_distance, generated.get_rab(i, j).norm());
        generated.energy_ = supercell.energy_ + ensemble[isample].harmonic_energy;
        generated.setEflag();
        generated.global_attribs_d_["ti_temperature_K"] = output_options.temperature_kelvin;
        generated.global_attribs_d_["ti_reference_energy0"] = supercell.energy_;
        generated.global_attribs_d_["ti_harmonic_delta_energy"] = ensemble[isample].harmonic_energy;
        generated.global_attribs_i_["ti_sample"] = isample;
        generated.global_attribs_i_["ti_seed"] = output_options.seed;
        generated.global_attribs_i_["ti_reference"] = 1;
        if (!writer.seq_write(*output, generated))
            throw std::runtime_error("Failed to write TI ensemble output: " + output_options.ensemble_file);
    }
    if (!close_ofile(output))
        throw std::runtime_error("Failed to finish TI ensemble output: " + output_options.ensemble_file);
    const double rms_displacement = std::sqrt(sum_squared_displacement
        / std::max(1, output_options.ensemble_size * supercell.size()));
    report << "ensemble_size " << ensemble.size() << '\n';
    report << "ensemble_seed " << output_options.seed << '\n';
    report << "ensemble_rms_displacement_a " << rms_displacement << '\n';
    report << "ensemble_maximum_displacement_a " << maximum_displacement << '\n';
    report << "reference_minimum_distance_a " << reference_minimum_distance << '\n';
    report << "ensemble_minimum_distance_a " << ensemble_minimum_distance << '\n';
    if (maximum_displacement > 0.5)
        Warning("TI ensemble contains displacements larger than 0.5 A; the exact Gaussian samples were retained");
    if (std::isfinite(reference_minimum_distance)
        && ensemble_minimum_distance < 0.7 * reference_minimum_distance)
        Warning("TI ensemble contains a pair distance below 70% of the reference minimum; the exact Gaussian samples were retained");
    finish_report();
}

} // namespace TIReferenceApp
