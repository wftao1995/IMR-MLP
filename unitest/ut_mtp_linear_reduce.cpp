#include <catch2/catch_all.hpp>
#include "train/mtp_linear_reduce.h"
#include "train/train_mtp.h"

#include <limits>

namespace {
template<class Matrix, class Vector>
void packed_reduction_contract(bool upper, bool root) {
    Matrix local, result;
    Vector rhs, reduced_rhs;
    if constexpr (std::is_same_v<Matrix, alglib::real_2d_array>) {
        local.setlength(3, 3);
        result.setlength(root ? 3 : 1, root ? 3 : 1);
        rhs.setlength(3);
        reduced_rhs.setlength(root ? 3 : 1);
    } else {
        local.resize(3, 3);
        result.resize(root ? 3 : 1, root ? 3 : 1);
        rhs.resize(3);
        reduced_rhs.resize(root ? 3 : 1);
    }
    const Eigen::Matrix3d h = (Eigen::Matrix3d() << 4., 1., 2., 1., 6., 3., 2., 3., 10.).finished();
    for (int i = 0; i < 3; ++i) {
        rhs[i] = i + 1.;
        for (int j = 0; j < 3; ++j)
            local(i, j) = (upper ? j >= i : j <= i) ? h(i, j)
                              : std::numeric_limits<double>::quiet_NaN();
    }
    result(0, 0) = -19.;
    reduced_rhs[0] = -23.;
    std::vector<double> send, receive;
    const std::vector<double> expected = upper
        ? std::vector<double>{1., 2., 3., 4., 1., 2., 6., 3., 10.}
        : std::vector<double>{1., 2., 3., 4., 1., 6., 2., 3., 10.};
    int calls = 0;
    double *storage = nullptr;
    for (int repeat = 0; repeat < 2; ++repeat) {
        Training::reduce_mtp_linear_system(local, rhs, result, reduced_rhs, 3,
            upper, root, send, receive, [&](double *input, double *output, int count) {
                ++calls;
                REQUIRE(count == 3 + 3 * 4 / 2);
                REQUIRE(count < 3 + 3 * 3); // Full Gram reduction sent 12 doubles.
                REQUIRE(std::vector<double>(input, input + count) == expected);
                if (root)
                    for (int i = 0; i < count; ++i) output[i] = 2. * input[i];
            });
        REQUIRE(calls == repeat + 1); // One collective per fit, including the RHS.
        if (repeat == 0) storage = send.data();
        REQUIRE(send.data() == storage);
    }
    if (!root) {
        REQUIRE(result(0, 0) == -19.);
        REQUIRE(reduced_rhs[0] == -23.);
        REQUIRE(receive.empty());
        return;
    }
    Eigen::Matrix3d sum;
    Eigen::Vector3d sum_rhs;
    for (int i = 0; i < 3; ++i) {
        sum_rhs[i] = reduced_rhs[i];
        for (int j = 0; j < 3; ++j)
            sum(i, j) = (upper ? j >= i : j <= i) ? result(i, j) : result(j, i);
    }
    REQUIRE((sum - 2. * h).norm() == 0.);
    REQUIRE((sum_rhs - 2. * Eigen::Vector3d(1., 2., 3.)).norm() == 0.);
    const Eigen::Vector3d solution = sum.ldlt().solve(sum_rhs);
    REQUIRE((solution - h.ldlt().solve(Eigen::Vector3d(1., 2., 3.))).norm() < 1e-14);
}
}

TEST_CASE("Legacy MTP normal reductions send one triangle and reuse buffers",
          "[training][mtp][mpi-volume]") {
    const bool root = GENERATE(false, true);
    SECTION("ALGLIB upper triangle") {
        packed_reduction_contract<alglib::real_2d_array, alglib::real_1d_array>(true, root);
    }
    SECTION("Eigen LDLT lower triangle") {
        packed_reduction_contract<Eigen::MatrixXd, Eigen::VectorXd>(false, root);
    }
}

TEST_CASE("Legacy MTP normal solvers preserve fitted predictions across assembly paths",
          "[training][mtp][linear-equivalence]") {
    const bool energy_only = GENERATE(false, true);
    const bool fixed_species = GENERATE(false, true);
    const bool precondition = GENERATE(false, true);
    Potentials::PotMTP fast("unitest/test-data/portable-mtp/les.mtp");
    Potentials::PotMTP slow("unitest/test-data/portable-mtp/les.mtp");
    std::vector<Atoms> atoms;
    std::vector<Neighborhoods> neighborhoods;
    for (int i = 0; i < 4; ++i) {
        Atoms atom;
        atom.cell_ = 20. * Eigen::Matrix3d::Identity();
        atom.pos_ = {{2., 2., 2.}, {4.1 + .15 * i, 2.3, 2.1},
                     {3.3, 4.4 - .1 * i, 3.1}};
        atom.chemmap_ = {0, 1, i % 2};
        fast.getEFS(atom);
        atoms.push_back(atom);
        neighborhoods.emplace_back(atom, fast.cutoff);
    }
    for (Potentials::PotMTP *potential : {&fast, &slow})
        for (int i = 0; i < potential->lin_coeff_size; ++i)
            potential->linear_coeffs[i] += .02 * (i + 1);
    GPSettings settings;
    settings.Modify("do_orthogonalization", "false");
    settings.Modify("fix_species_coeffs", fixed_species ? "true" : "false");
    settings.Modify("linear_precondition", precondition ? "diagonal" : "none");
    settings.Modify("weight_L2", "0.01");
    Trainer_MTP fast_trainer(settings), slow_trainer(settings);
    fast_trainer.fast_lfit = true;
    slow_trainer.fast_lfit = false;
    fast_trainer.linear_fit_eng_only = slow_trainer.linear_fit_eng_only = energy_only;
    fast_trainer.init(atoms, neighborhoods, &fast);
    slow_trainer.init(atoms, neighborhoods, &slow);
    fast_trainer.TrainLinear();
    slow_trainer.TrainLinear();
    for (int i = 0; i < fast.lin_coeff_size; ++i)
        REQUIRE(fast.linear_coeffs[i] == Catch::Approx(slow.linear_coeffs[i]).margin(1e-9));
    for (auto atom : atoms) {
        Atoms other = atom;
        fast.getEFS(atom);
        slow.getEFS(other);
        REQUIRE(atom.energy_ == Catch::Approx(other.energy_).margin(1e-9));
        for (int i = 0; i < atom.size(); ++i)
            REQUIRE((atom.forces_[i] - other.forces_[i]).norm() < 1e-9);
        REQUIRE((atom.stress_ - other.stress_).norm() < 1e-9);
    }
}
