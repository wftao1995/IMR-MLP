#include <catch2/catch_all.hpp>

#include "potentials/les.h"

using namespace Potentials;
using Catch::Approx;

namespace {
Eigen::Matrix<double, 1, 6> six(const Eigen::Matrix3d &value) {
    Eigen::Matrix<double, 1, 6> result;
    result << value(0,0), value(1,1), value(2,2), value(0,1), value(0,2), value(1,2);
    return result;
}
Atoms cell() {
    Atoms atoms;
    atoms.cell_ << 8.2, 0., 0., 1.2, 8.9, 0., -.5, .8, 9.4;
    atoms.pos_ = {{1.3, 2.1, 1.7}, {4.1, 2.7, 2.4}, {6.4, 5.3, 5.6}};
    atoms.chemmap_ = {0, 1, 0};
    return atoms;
}
}

TEST_CASE("LES and SOG per-atom virials differentiate per-atom energies",
          "[les][sog][site-virial]")
{
    const int channels = GENERATE(0, 1, 3);
    const bool zero_mode = GENERATE(false, true);
    const bool remove_self = GENERATE(false, true);
    const bool recurrence = GENERATE(false, true);
    INFO(channels << ' ' << zero_mode << ' ' << remove_self << ' ' << recurrence);
    LES model;
    model.kernel = channels ? LES::Kernel::SOG : LES::Kernel::Coulomb;
    model.sog_channels = std::max(1, channels);
    model.sog_include_zero = zero_mode;
    model.remove_self_interaction = remove_self;
    model.use_recurrence = recurrence;
    model.dl = 2.2;
    model.taper_fraction = .2;
    for (int c = 0; c < channels; ++c) {
        model.sog_weights.insert(model.sog_weights.end(), {2.3 + c, -.6 - .2*c});
        model.sog_log_bandwidths.insert(model.sog_log_bandwidths.end(), {-.6 + .1*c, .2});
    }
    const Atoms atoms = cell();
    Eigen::MatrixXd latents(atoms.size(), std::max(1, channels));
    for (int c = 0; c < latents.cols(); ++c)
        latents.col(c) << .6 + .13*c, -.4 + .07*c, .25 - .03*c;
    LES::Workspace workspace;
    const auto reference = model.evaluate_channels(atoms, latents, workspace);
    REQUIRE(reference.per_atom_virial.rows() == atoms.size());
    REQUIRE((reference.per_atom_virial.colwise().sum() - six(reference.virial)).norm() < 2e-12);
    const auto legacy = model.evaluate_channels(atoms, latents);
    REQUIRE((legacy.per_atom_virial - reference.per_atom_virial).norm() < 2e-12);
    Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> channel_sum =
        Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor>::Zero(atoms.size(), 6);
    for (int c = 0; c < latents.cols(); ++c) {
        const auto value = model.evaluate(atoms, latents.col(c), true, workspace, c);
        REQUIRE((value.per_atom_virial.colwise().sum() - six(value.virial)).norm() < 2e-12);
        channel_sum += value.per_atom_virial;
        LES::ChannelScope scope(model, c);
        const auto legacy_single = model.evaluate(atoms, latents.col(c), true);
        REQUIRE((legacy_single.per_atom_virial - value.per_atom_virial).norm() < 2e-12);
    }
    REQUIRE((channel_sum - reference.per_atom_virial).norm() < 2e-12);
    const int first[] = {0, 1, 2, 0, 0, 1};
    const int second[] = {0, 1, 2, 1, 2, 2};
    constexpr double step = 1e-6;
    for (int component = 0; component < 6; ++component) {
        Eigen::Matrix3d strain = Eigen::Matrix3d::Zero();
        const int a = first[component], b = second[component];
        strain(a,b) = a == b ? 1. : .5;
        if (a != b) strain(b,a) = .5;
        Atoms plus = atoms, minus = atoms;
        plus.cell_ = atoms.cell_ * (Eigen::Matrix3d::Identity() + step * strain);
        minus.cell_ = atoms.cell_ * (Eigen::Matrix3d::Identity() - step * strain);
        for (int i = 0; i < atoms.size(); ++i) {
            plus.pos_[i] = atoms.pos_[i] * (Eigen::Matrix3d::Identity() + step * strain);
            minus.pos_[i] = atoms.pos_[i] * (Eigen::Matrix3d::Identity() - step * strain);
        }
        const auto p = model.evaluate_channels(plus, latents, workspace);
        const auto m = model.evaluate_channels(minus, latents, workspace);
        for (int i = 0; i < atoms.size(); ++i)
            REQUIRE(-(p.per_atom_energy(i) - m.per_atom_energy(i)) / (2. * step) ==
                Approx(reference.per_atom_virial(i, component)).epsilon(2e-6).margin(2e-7));
    }
    const auto zeros = model.evaluate_channels(atoms, Eigen::MatrixXd::Zero(latents.rows(), latents.cols()), workspace);
    REQUIRE(zeros.per_atom_virial.rows() == atoms.size());
    REQUIRE(zeros.per_atom_virial.isZero());
}
