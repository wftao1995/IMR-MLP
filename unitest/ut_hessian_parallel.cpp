// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp.h"
#include "external/spglib_driver.h"
#include "utils/cmdopts.h"
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <set>
#ifdef USE_OMP
#include <omp.h>
#endif

using Potentials::HessianMatrix;
using Potentials::PotMTP;
using Catch::Approx;

namespace {
struct ThreadSettings {
    int preference = omp_num_threads;
#ifdef USE_OMP
    int requested = omp_get_max_threads();
    int dynamic = omp_get_dynamic();
    int levels = omp_get_max_active_levels();
#endif
    ~ThreadSettings() {
        omp_num_threads = preference;
#ifdef USE_OMP
        omp_set_num_threads(requested);
        omp_set_dynamic(dynamic);
        omp_set_max_active_levels(levels);
#endif
    }
};

struct ModelFixture {
    std::filesystem::path directory, filename;
    ModelFixture() {
        auto pattern = (std::filesystem::temp_directory_path() / "imrmlp-hessian-parallel-XXXXXX").string();
        const char *created = mkdtemp(pattern.data());
        if (!created) throw std::runtime_error("Cannot create Hessian fixture directory");
        directory = created;
        filename = directory / "model.mtp";
        // Exact tracked synthetic model: examples/mtp-gpu-alni/p0-real-data/mixedprec-poly5.mtp.
        // Angular moments and two species give nonzero mixed Hessian blocks.
        std::ofstream output(filename);
        output << R"mtp(MTP
version = 1.1.0
potential_name = MTP1m
scaling = 1
species_count = 2
potential_tag =
radial_basis_type = RBChebyshevPoly5
  min_dist = 0.4
  max_dist = 4.5
  radial_basis_size = 2
  rb_b = -0.7
radial_funcs_count = 1
radial_coeffs
  0-0
    {0.70, -0.20}
  0-1
    {0.45, 0.10}
  1-0
    {-0.35, 0.25}
  1-1
    {0.80, -0.15}
alpha_moments_count = 5
alpha_index_basic_count = 4
alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}
alpha_index_times_count = 3
alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}}
alpha_scalar_moments = 2
alpha_moment_mapping = {0, 4}
species_coeffs = {0.3, -0.1}
moment_coeffs = {1.2, -0.4}
)mtp";
        output.close();
        if (!output) {
            std::error_code error;
            std::filesystem::remove_all(directory, error);
            throw std::runtime_error("Cannot write Hessian fixture");
        }
    }
    ~ModelFixture() {
        std::error_code error;
        std::filesystem::remove_all(directory, error);
    }
};

Atoms geometry(int count) {
    Atoms atom;
    atom.cell_ = 20 * Eigen::Matrix3d::Identity();
    // The symmetry API requires a fully periodic structure. The 20-Angstrom
    // box isolates the small clusters; assert_isolated_fixture checks that
    // no additional periodic neighbor is introduced by this representation.
    atom.pbc_ = {true, true, true};
    atom.pos_ = {{1.1,1.3,1.5}, {3.2,1.6,1.8}, {1.7,3.8,2.1},
                 {3.7,3.3,3.5}, {2.4,2.2,4.2}};
    atom.chemmap_ = {0,1,0,1,0};
    atom.pos_.resize(count); atom.chemmap_.resize(count);
    if (count == 1 || count == 4) {
        atom.pbc_ = {true, true, true};
        atom.cell_ << 3.1,0,0, .2,3.4,0, .1,.3,3.7;
        atom.pos_ = {{.3,.4,.5}};
        atom.chemmap_ = {0};
        if (count == 4) {
            // Two primitive basis atoms repeated exactly along lattice row 0.
            // Short rows 1 and 2 keep same-index periodic images in the list.
            atom.pos_.push_back({1.5,1.2,1.8}); atom.chemmap_.push_back(1);
            for (int i = 0; i < 2; ++i) {
                atom.pos_.push_back(atom.pos_[i] + atom.cell_.row(0));
                atom.chemmap_.push_back(atom.chemmap_[i]);
            }
            atom.cell_.row(0) *= 2;
        }
    }
    return atom;
}

void assert_isolated_fixture(const Atoms &atom, double cutoff) {
    Neighborhoods nbhs(atom,cutoff);
    for (int i = 0; i < atom.size(); ++i) {
        int expected_count = 0;
        for (int j = 0; j < atom.size(); ++j)
            if (i != j && (atom.pos_[j]-atom.pos_[i]).norm() < cutoff) ++expected_count;
        REQUIRE(nbhs[i].nbhcount_ == expected_count);
        std::set<int> seen;
        for (int k = 0; k < nbhs[i].nbhcount_; ++k) {
            const int j = nbhs[i].indices_[k];
            REQUIRE(j >= 0); REQUIRE(j < atom.size()); REQUIRE(j != i);
            REQUIRE(seen.insert(j).second);
            // Count alone would not detect replacing a direct pair by an
            // image. Check each relative vector against the unshifted pair.
            REQUIRE(nbhs[i].pos_[k].isApprox(atom.pos_[j]-atom.pos_[i],1e-13));
        }
    }
}

Eigen::MatrixXd dense(const HessianMatrix &hessian) {
    Eigen::MatrixXd result(3*hessian.size, 3*hessian.size);
    for (int i = 0; i < hessian.size; ++i)
        for (int j = 0; j < hessian.size; ++j)
            result.block<3,3>(3*i,3*j) = hessian(i,j);
    return result;
}

Eigen::VectorXd forces(const Atoms &atom) {
    Eigen::VectorXd result(3*atom.size());
    for (int i = 0; i < atom.size(); ++i) result.segment<3>(3*i) = atom.forces_[i];
    return result;
}

Eigen::MatrixXd directions(int count) {
    Eigen::MatrixXd result(3*count,2);
    for (int i = 0; i < result.rows(); ++i) {
        result(i,0) = std::sin(.37*(i+1));
        result(i,1) = std::cos(.61*(i+1)) + .2*std::sin(.17*(i+2));
    }
    for (int j = 0; j < result.cols(); ++j) result.col(j).normalize();
    return result;
}

void close(const Eigen::MatrixXd &actual, const Eigen::MatrixXd &expected, double tolerance = 2e-10) {
    REQUIRE(actual.rows() == expected.rows());
    REQUIRE(actual.cols() == expected.cols());
    REQUIRE(actual.allFinite()); REQUIRE(expected.allFinite());
    Eigen::Index row, column;
    const double error = (actual-expected).cwiseAbs().maxCoeff(&row,&column);
    const double scale = std::max(1.,expected.cwiseAbs().maxCoeff());
    INFO("worst component " << row << "," << column << ": " << actual(row,column)
         << " versus " << expected(row,column) << "; scaled error " << error/scale);
    REQUIRE(error <= tolerance*scale);
}

void close_efs(const Atoms &actual, const Atoms &expected) {
    REQUIRE(std::isfinite(actual.energy_));
    REQUIRE(actual.energy_ == Approx(expected.energy_).epsilon(2e-10).margin(2e-10));
    REQUIRE(actual.forces_.size() == static_cast<size_t>(expected.size()));
    close(forces(actual),forces(expected));
    close(actual.stress_,expected.stress_);
    REQUIRE(actual.site_energies_.size() == static_cast<size_t>(expected.size()));
    for (int i = 0; i < expected.size(); ++i)
        REQUIRE(actual.site_energies_[i] == Approx(expected.site_energies_[i]).epsilon(2e-10).margin(2e-10));
    REQUIRE(actual.energy_ == Approx(std::accumulate(actual.site_energies_.begin(),
        actual.site_energies_.end(),0.)).epsilon(2e-10).margin(2e-10));
}

struct Reference {
    Atoms efs;
    Eigen::MatrixXd hessian;
};

Reference reference(PotMTP &pot, const Atoms &input) {
    if (input.size() == 2 || input.size() == 5) assert_isolated_fixture(input,pot.cutoff);
    auto atom = input;
    HessianMatrix blocks;
    pot.getEFSHessian(atom,blocks);
    auto ordinary = input;
    pot.getEFS(ordinary);
    close_efs(atom,ordinary); // independent EFS path, not only parallel/serial agreement
    auto hessian = dense(blocks);
    REQUIRE(std::abs(atom.energy_) > 1e-6);
    REQUIRE(atom.stress_.norm() > 1e-6);
    if (input.size() > 1) {
        REQUIRE(forces(atom).norm() > 1e-6);
        REQUIRE(hessian.norm() > 1e-6);
    }
    close(hessian,hessian.transpose());
    for (int axis = 0; axis < 3; ++axis) {
        Eigen::VectorXd translation = Eigen::VectorXd::Zero(hessian.rows());
        for (int i = 0; i < input.size(); ++i) translation[3*i+axis] = 1;
        close(hessian*translation,Eigen::VectorXd::Zero(hessian.rows()),
              2e-10*std::max(1.,hessian.norm()));
    }
    // One periodic atom translates with all its images: nonzero E/W, zero H.
    if (input.size() == 1) close(hessian,Eigen::Matrix3d::Zero());
    return {atom,hessian};
}

void assert_periodic_fixture(Atoms atom, double cutoff) {
    Neighborhoods nbhs(atom,cutoff);
    int self_images = 0;
    for (int i = 0; i < atom.size(); ++i)
        for (int j = 0; j < nbhs[i].nbhcount_; ++j)
            if (nbhs[i].indices_[j] == i) ++self_images;
    REQUIRE(self_images > 0);
    if (atom.size() == 4) {
        const auto dataset = Symmetry3D::get_sym_dataset(atom,1e-7);
        REQUIRE(dataset.mapping_to_primitive.size() == 4);
        REQUIRE(std::set<int>(dataset.mapping_to_primitive.begin(),dataset.mapping_to_primitive.end()).size() == 2);
    }
}

struct Outputs {
    Atoms regular, symmetric;
    HessianMatrix regular_hessian, symmetric_hessian;
    Eigen::MatrixXd products;
    Eigen::VectorXd scratch;
};

void calculate(PotMTP &pot, const Atoms &input, Outputs &out) {
    out.regular = input; out.symmetric = input;
    // Nonzero sentinels ensure overwrite semantics do not depend on fresh data.
    for (auto *atom : {&out.regular,&out.symmetric}) {
        atom->resetEFS(); atom->energy_ = 17; atom->stress_.setConstant(11);
        for (auto &force : atom->forces_) force.setConstant(7);
        for (auto &energy : atom->site_energies_) energy = 13;
    }
    Neighborhoods nbhs(out.regular,pot.cutoff);
    pot.getEFSHessianOMP(out.regular,nbhs,out.regular_hessian);
    pot.getEFSHessianSymmOMP(out.symmetric,out.symmetric_hessian,1e-7);
    const auto probes = directions(input.size());
    out.products.resize(3*input.size(),probes.cols());
    // Exercise both neighborhood and convenience HVP interfaces. Reuse output
    // scratch across probes and across larger/smaller systems.
    pot.applyHessianVectorProduct(out.regular,nbhs,probes.col(0),out.scratch);
    out.products.col(0) = out.scratch;
    pot.applyHessianVectorProduct(out.regular,probes.col(1),out.scratch);
    out.products.col(1) = out.scratch;
}

void compare(const Outputs &out, const Reference &expected) {
    close_efs(out.regular,expected.efs); close_efs(out.symmetric,expected.efs);
    close(dense(out.regular_hessian),expected.hessian);
    close(dense(out.symmetric_hessian),expected.hessian);
    close(out.products,expected.hessian*directions(expected.efs.size()));
}

Eigen::VectorXd force_difference(PotMTP &pot, const Atoms &input,
                                const Eigen::VectorXd &direction, double step) {
    auto plus = input, minus = input;
    for (int i = 0; i < input.size(); ++i) {
        plus.pos_[i] += step*direction.segment<3>(3*i).transpose();
        minus.pos_[i] -= step*direction.segment<3>(3*i).transpose();
    }
    pot.getEFS(plus); pot.getEFS(minus);
    return -(forces(plus)-forces(minus))/(2*step);
}
} // namespace

TEST_CASE("MTP parallel dense Hessians and HVPs preserve full outputs across sizes and teams", "[hessian][hvp][openmp]") {
    ThreadSettings restore;
    ModelFixture fixture;
    set_omp_num_threads(1);
    PotMTP pot(fixture.filename.string());
    std::vector<Reference> expected;
    for (int count : {5,2,1,4}) {
        auto atom = geometry(count);
        if (count == 1 || count == 4) assert_periodic_fixture(atom,pot.cutoff);
        expected.push_back(reference(pot,atom));
    }
    Outputs out;
    out.scratch = Eigen::VectorXd::Constant(31,19);
    for (int threads : {1,2,4,1}) {
        set_omp_num_threads(threads);
        for (int index : {0,1,2,3,1,0}) {
            CAPTURE(threads,expected[index].efs.size());
            calculate(pot,geometry(expected[index].efs.size()),out);
            compare(out,expected[index]);
        }
    }
}

TEST_CASE("MTP dense and directional curvature agrees with independent force differences", "[hessian][hvp][numerics]") {
    ThreadSettings restore;
    ModelFixture fixture;
    set_omp_num_threads(1);
    PotMTP pot(fixture.filename.string());
    for (int count : {5,4}) {
        CAPTURE(count);
        const auto atom = geometry(count);
        // Check a shell beyond the cutoff too: a finite-difference displacement
        // must not change topology by bringing an omitted pair into range.
        Neighborhoods shell(atom,pot.cutoff+.05);
        for (int i = 0; i < count; ++i)
            for (double distance : shell[i].distances_) REQUIRE(std::abs(distance-pot.cutoff) > .01);
        const auto expected = reference(pot,atom);
        Eigen::MatrixXd finite(3*count,3*count);
        for (int j = 0; j < finite.cols(); ++j) {
            const Eigen::VectorXd unit = Eigen::VectorXd::Unit(3*count,j);
            finite.col(j) = force_difference(pot,atom,unit,1e-5);
        }
        close(expected.hessian,finite,2e-7);
        const auto probes = directions(count);
        for (int j = 0; j < probes.cols(); ++j) {
            const Eigen::VectorXd exact = expected.hessian*probes.col(j);
            const double coarse_error = (force_difference(pot,atom,probes.col(j),2e-3)-exact).norm();
            const double fine_error = (force_difference(pot,atom,probes.col(j),1e-3)-exact).norm();
            CAPTURE(j,coarse_error,fine_error);
            REQUIRE(fine_error <= .35*coarse_error+2e-9*std::max(1.,exact.norm()));
            REQUIRE(fine_error <= 2e-6*std::max(1.,exact.norm()));
        }
    }
}

TEST_CASE("MTP curvature reductions handle dynamic and serialized nested teams", "[hessian][hvp][openmp]") {
#ifndef USE_OMP
    SKIP("Dynamic and nested team regression requires OpenMP");
#else
    ThreadSettings restore;
    ModelFixture fixture;
    set_omp_num_threads(4);
    omp_set_dynamic(0);
    PotMTP pot(fixture.filename.string());
    Outputs out;
    calculate(pot,geometry(5),out); // populate all four scratch slots first
    for (int count : {2,4}) {
        CAPTURE(count);
        const auto atom = geometry(count);
        const auto expected = reference(pot,atom);
        omp_set_dynamic(1);
        int dynamic_team = 0;
#pragma omp parallel num_threads(4)
        {
#pragma omp single
            dynamic_team = omp_get_num_threads();
        }
        CAPTURE(dynamic_team);
        REQUIRE(dynamic_team >= 1); REQUIRE(dynamic_team <= 4);
        calculate(pot,atom,out);
        compare(out,expected);
        omp_set_dynamic(0);
        omp_set_max_active_levels(1);
        int nested_team = 0;
#pragma omp parallel num_threads(2)
        {
#pragma omp single
            {
                omp_set_num_threads(4);
#pragma omp parallel num_threads(4)
                {
#pragma omp single
                    nested_team = omp_get_num_threads();
                }
                calculate(pot,atom,out);
            }
        }
        REQUIRE(nested_team == 1);
        compare(out,expected);
    }
#endif
}
