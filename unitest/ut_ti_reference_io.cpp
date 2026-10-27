#include <catch2/catch_all.hpp>

#include "tools/hessian_io.h"
#include "tools/ti_harmonic_reference.h"
#include "tools/ti_reference_io.h"

#include <fstream>

using Catch::Approx;

TEST_CASE("TI reference IO writes standard and versioned force constants",
          "[ti-reference][io]")
{
    Atoms atoms;
    atoms.cell_ << 3.0, 0.0, 0.0,
                   0.5, 2.8, 0.0,
                   0.2, 0.3, 4.0;
    atoms.pos_ = {Eigen::RowVector3d(0.1, 0.2, 0.3),
                  Eigen::RowVector3d(1.1, 1.2, 1.3)};
    atoms.chemmap_ = {0, 1};
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(6, 6);
    Eigen::Matrix3d coupling;
    coupling << -1.0, 0.2, 0.1,
                 0.2, -2.0, 0.3,
                 0.1, 0.3, -3.0;
    hessian.block<3, 3>(0, 3) = coupling;
    hessian.block<3, 3>(3, 0) = coupling.transpose();
    hessian.block<3, 3>(0, 0) = -coupling;
    hessian.block<3, 3>(3, 3) = -coupling;

    const std::string force_file = "/tmp/ut_ti_reference_FORCE_CONSTANTS";
    const std::string lammps_file = "/tmp/ut_ti_reference_lammps";
    TIReferenceIO::write_force_constants(force_file, hessian);
    REQUIRE(HessianIO::read_force_constants(force_file, 2).isApprox(hessian, 1e-13));
    const Atoms rotated = TIReferenceIO::write_lammps_reference(
        lammps_file, atoms, hessian, -12.5);
    REQUIRE(rotated.cell_(0, 1) == Approx(0.0).margin(1e-14));
    REQUIRE(rotated.cell_(0, 2) == Approx(0.0).margin(1e-14));
    REQUIRE(rotated.cell_(1, 2) == Approx(0.0).margin(1e-14));
    REQUIRE(rotated.cell_.determinant() == Approx(atoms.cell_.determinant()).epsilon(1e-12));
    std::ifstream input(lammps_file);
    std::string magic;
    int version = 0;
    input >> magic >> version;
    REQUIRE(magic == "TI_HARMONIC_REFERENCE");
    REQUIRE(version == 1);
}

TEST_CASE("Sparse TI reference truncation preserves ASR and positive modes",
          "[ti-reference][io][sparse]")
{
    Atoms atoms;
    atoms.cell_ = 10.0 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {Eigen::RowVector3d(0.0, 0.0, 0.0),
                  Eigen::RowVector3d(1.0, 0.0, 0.0),
                  Eigen::RowVector3d(4.0, 0.0, 0.0)};
    atoms.chemmap_ = {0, 0, 0};
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(9, 9);
    auto add_pair = [&](int i, int j, double strength) {
        const Eigen::Matrix3d block = -strength * Eigen::Matrix3d::Identity();
        hessian.block<3, 3>(3 * i, 3 * j) += block;
        hessian.block<3, 3>(3 * j, 3 * i) += block;
        hessian.block<3, 3>(3 * i, 3 * i) -= block;
        hessian.block<3, 3>(3 * j, 3 * j) -= block;
    };
    add_pair(0, 1, 2.0);
    add_pair(1, 2, 1.0);
    add_pair(0, 2, 1e-5);

    TIReferenceIO::SparseOptions options;
    options.block_threshold = 1e-4;
    options.minimum_frequency_thz = 0.5;
    const auto sparse = TIReferenceIO::truncate_force_constants(
        atoms, hessian, {1.0, 1.0, 1.0}, options);
    REQUIRE(sparse.diagnostics.retained_offdiagonal_blocks == 2);
    REQUIRE(sparse.diagnostics.retained_fraction == Approx(2.0 / 3.0));
    REQUIRE(sparse.hessian.block<3, 3>(0, 6).norm() == Approx(0.0).margin(1e-14));
    REQUIRE(sparse.diagnostics.acoustic_sum_rule_error == Approx(0.0).margin(1e-12));
    REQUIRE(sparse.diagnostics.minimum_frequency_after_thz >= 0.5 - 1e-8);
    const Eigen::VectorXd exported_values = TIReferenceIO::internal_mode_eigenvalues(
        sparse.hessian, {1.0, 1.0, 1.0});
    REQUIRE(TIHarmonicReference::signed_frequency_thz(exported_values.minCoeff())
            == Approx(sparse.diagnostics.minimum_frequency_after_thz).epsilon(1e-12));
}

TEST_CASE("Sparse TI reference can stabilize an indefinite retained graph",
          "[ti-reference][io][sparse]")
{
    Atoms atoms;
    atoms.cell_ = 8.0 * Eigen::Matrix3d::Identity();
    atoms.pos_ = {Eigen::RowVector3d(0.0, 0.0, 0.0),
                  Eigen::RowVector3d(1.0, 0.0, 0.0)};
    atoms.chemmap_ = {0, 0};
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(6, 6);
    hessian.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();
    hessian.block<3, 3>(3, 0) = Eigen::Matrix3d::Identity();
    hessian.block<3, 3>(0, 0) = -Eigen::Matrix3d::Identity();
    hessian.block<3, 3>(3, 3) = -Eigen::Matrix3d::Identity();
    TIReferenceIO::SparseOptions options;
    options.minimum_frequency_thz = 0.25;
    const auto sparse = TIReferenceIO::truncate_force_constants(
        atoms, hessian, {1.0, 1.0}, options);
    REQUIRE(sparse.diagnostics.minimum_frequency_before_thz < 0.0);
    REQUIRE(sparse.diagnostics.isotropic_stabilizer > 0.0);
    REQUIRE(sparse.diagnostics.minimum_frequency_after_thz >= 0.25 - 1e-8);
    REQUIRE(sparse.diagnostics.acoustic_sum_rule_error == Approx(0.0).margin(1e-12));
}
