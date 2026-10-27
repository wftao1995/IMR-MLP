// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include "train/train_nonlinear_potential.h"
#include <catch2/catch_all.hpp>
#include <fstream>
#include <sstream>
using namespace Potentials;
using Catch::Approx;
#ifdef IMRMLP_MPI
// Catch2's stock main does not initialize the communicator used by training.
int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);
    mpi.InitComm(MPI_COMM_WORLD);
    const int result = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return result;
}
#endif
namespace {
Atoms atoms()
{
    Atoms a;
    a.cell_ << 10., 0, 0, 1.2, 10.5, 0, -.7, .6, 11.;
    a.pos_ = { { .2, 2., 2. }, { 9.2, 2.2, 2. }, { 2.3, 4.2, 2.4 }, { 5., 4.8, 3.1 } };
    a.chemmap_ = { 0, 1, 0, 1 };
    return a;
}
void initialize(PotMTP_LES& p, int channels = 3)
{
    p.les_kernel = "sog";
    p.sog_channels = channels;
    p.sog_terms = 2;
    p.train_init();
    for (int c = 0; c < channels; ++c) {
        p.les_species_coeffs[2 * c] = .23 + .11 * c;
        p.les_species_coeffs[2 * c + 1] = -.17 + .07 * c;
        for (int a = 0; a < p.get_alpha_scalar_moments(); ++a)
            p.les_moment_coeffs[c * p.get_alpha_scalar_moments() + a]
                = a < 3 ? .12 * std::sin(a + c + 1.) : 0.;
        p.model_les.sog_weights[2 * c] = 4.2 + c;
        p.model_les.sog_weights[2 * c + 1] = -1.3 - .2 * c;
        p.model_les.sog_log_bandwidths[2 * c] = -.4 + .1 * c;
        p.model_les.sog_log_bandwidths[2 * c + 1] = .2 - .1 * c;
    }
}
template <class Potential> void gradients(Potential& p)
{
    auto a = atoms();
    Neighborhoods nbhs(a, p.cutoff);
    Eigen::Matrix<double, -1, 3> fw(4, 3);
    fw << .2, -.1, .3, -.5, .7, .1, .3, .2, -.4, .1, .2, .4;
    Eigen::Matrix3d sw;
    sw << .1, .2, -.1, .3, -.2, .5, -.1, .2, .7;
    const bool energy_only = GENERATE(false, true);
    std::vector<double> x(p.get_NL_coeffs_count()), g(x.size(), 0.);
    p.nl_get_coeffs(x.data());
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> unused;
    if (energy_only) {
        fw.setZero();
        sw.setZero();
        p.AccumulateEFSCombinationGard(a, nbhs, .37, g.data());
    } else
        p.AccumulateEFSCombinationGard(a, nbhs, .37, fw, sw, unused, g.data());
    auto loss = [&]() {
        auto b = atoms();
        p.getEFS(b);
        double l = .37 * b.energy_ + (sw.array() * b.stress_.array()).sum();
        for (int i = 0; i < 4; ++i)
            l += fw.row(i).dot(b.forces_[i]);
        return l;
    };
    for (size_t i = 0; i < x.size(); ++i) {
        CAPTURE(i, energy_only);
        auto plus = x, minus = x;
        plus[i] += 1e-6;
        minus[i] -= 1e-6;
        p.nl_set_coeffs(plus.data());
        double ep = loss();
        p.nl_set_coeffs(minus.data());
        double em = loss();
        REQUIRE(g[i] == Approx((ep - em) / 2e-6).margin(4e-6));
    }
    p.nl_set_coeffs(x.data());
}
} // namespace
TEST_CASE(
    "Multichannel SOG sums independent EFS and differentiates every parameter", "[sog][channels]")
{
    PotMTP_LES p("tests/case-mtp-les/les.mtp");
    initialize(p);
    auto a = atoms();
    p.getEFS(a);
    REQUIRE(a.charges_.empty());
    REQUIRE(p.LES_all_latents.cols() == 3);
    PotMTP sr("tests/case-mtp-les/les.mtp");
    auto baseline = atoms();
    sr.getEFS(baseline);
    const auto kernel = p.model_les.evaluate_channels(a, p.LES_all_latents);
    REQUIRE(kernel.energy == Approx(a.energy_ - baseline.energy_).margin(1e-11));
    REQUIRE((kernel.dEdq - p.model_les.apply_charge_hessian_channels(a, p.LES_all_latents)).norm()
        < 1e-11);
    REQUIRE(kernel.per_atom_energy.sum() == Approx(kernel.energy).margin(1e-11));
    REQUIRE_THROWS_AS(p.model_les.evaluate_channels(a, Eigen::MatrixXd::Zero(4, 2)), GPException);
    double e = -2 * baseline.energy_;
    Eigen::Matrix3d w = -2 * baseline.stress_;
    std::vector<Eigen::Vector3d> f = baseline.forces_;
    for (auto& v : f)
        v *= -2;
    for (int c = 0; c < 3; ++c) {
        PotMTP_LES one("tests/case-mtp-les/les.mtp");
        initialize(one, 1);
        std::copy_n(p.les_species_coeffs + 2 * c, 2, one.les_species_coeffs);
        std::copy_n(p.les_moment_coeffs + c * p.get_alpha_scalar_moments(),
            p.get_alpha_scalar_moments(), one.les_moment_coeffs);
        for (int m = 0; m < 2; ++m) {
            one.model_les.sog_weights[m] = p.model_les.sog_weights[2 * c + m];
            one.model_les.sog_log_bandwidths[m] = p.model_les.sog_log_bandwidths[2 * c + m];
        }
        auto b = atoms();
        one.getEFS(b);
        e += b.energy_;
        w += b.stress_;
        for (int i = 0; i < 4; ++i)
            f[i] += b.forces_[i];
        REQUIRE((one.LES_all_qs - p.LES_all_latents.col(c)).norm() < 1e-12);
    }
    REQUIRE(a.energy_ == Approx(e).margin(1e-11));
    REQUIRE((a.stress_ - w).norm() < 1e-10);
    for (int i = 0; i < 4; ++i)
        REQUIRE((a.forces_[i] - f[i]).norm() < 1e-10);
    gradients(p);
}
TEST_CASE("SOG v4 roundtrips, validates dimensions, and initializes zero "
          "channels independently",
    "[sog][channels][load]")
{
    GPSettings settings;
    settings.Modify("les_kernel", "sog");
    settings.Modify("sog_channels", "3");
    settings.Modify("sog_terms", "2");
    PotMTP_LES configured("tests/case-mtp-les/les.mtp", settings);
    configured.train_init();
    REQUIRE(configured.channels() == 3);
    REQUIRE(configured.model_les.sog_weights.size() == 6);
    REQUIRE(configured.initialize_zero_les_coeffs({ }, { }, .1, 42));
    PotMTP_LES p("tests/case-mtp-les/les.mtp");
    initialize(p);
    std::fill_n(p.les_species_coeffs + 2, 2, 0.);
    std::fill_n(
        p.les_moment_coeffs + p.get_alpha_scalar_moments(), p.get_alpha_scalar_moments(), 0.);
    const double first = p.les_species_coeffs[0], last = p.les_species_coeffs[4];
    REQUIRE(p.initialize_zero_les_coeffs({ }, { }, .1, 42));
    REQUIRE(p.les_species_coeffs[0] == first);
    REQUIRE(p.les_species_coeffs[4] == last);
    REQUIRE_FALSE(p.initialize_zero_les_coeffs({ }, { }, .1, 42));
    std::ostringstream saved;
    p.Save(saved);
    REQUIRE(saved.str().find("les_version = 4") != std::string::npos);
    PotMTP_LES q;
    std::istringstream in(saved.str());
    q.Load(in);
    q.train_init();
    std::vector<double> x(p.get_NL_coeffs_count()), y(q.get_NL_coeffs_count());
    p.nl_get_coeffs(x.data());
    q.nl_get_coeffs(y.data());
    REQUIRE(x.size() == y.size());
    for (size_t i = 0; i < x.size(); ++i)
        REQUIRE(x[i] == Approx(y[i]).margin(1e-12));
    auto a = atoms(), b = atoms();
    p.getEFS(a);
    q.getEFS(b);
    REQUIRE(a.energy_ == Approx(b.energy_).margin(1e-12));
    for (const auto& change : { std::pair { "les_version = 4", "les_version = 3" },
             std::pair { "sog_channels = 3", "sog_channels = 2" },
             std::pair { "sog_channels = 3", "sog_channels = 0" } }) {
        auto text = saved.str();
        text.replace(text.find(change.first), std::string(change.first).size(), change.second);
        PotMTP_LES invalid;
        std::istringstream bad(text);
        REQUIRE_THROWS_AS(invalid.Load(bad), GPException);
    }
    PotMTP_LES one("tests/case-mtp-les/les.mtp");
    initialize(one, 1);
    std::ostringstream old;
    one.Save(old);
    std::istringstream reload(old.str());
    q.Load(reload);
    REQUIRE(q.sog_channels == 1);
}
TEST_CASE(
    "MTP-M multichannel SOG supports full and kernel-only gradients", "[sog][channels][mtp_m]")
{
    using M = PotMTP_M<m_model_mtpq2>;
    PotMTP_LES reference("tests/case-mtp-les/les.mtp");
    initialize(reference, 2);
    std::ifstream input("tests/case-mtp-les/les.mtp");
    std::string text((std::istreambuf_iterator<char>(input)), { });
    text.resize(text.find("species_coeffs"));
    std::ostringstream source;
    source << text << std::setprecision(17) << "les_species_coeffs = {";
    for (int i = 0; i < 4; ++i)
        source << reference.les_species_coeffs[i] << ", ";
    source << "}\nles_moment_coeffs = {";
    for (int i = 0; i < 2 * reference.get_alpha_scalar_moments(); ++i)
        source << reference.les_moment_coeffs[i] << ", ";
    source << "}\n";
    SOGModel model;
    model.channels = 2;
    model.weights = reference.model_les.sog_weights;
    model.log_bandwidths = reference.model_les.sog_log_bandwidths;
    model.write(source);
    M p;
    std::istringstream in(source.str());
    p.Load(in);
    p.train_init();
    p.model_coeffs[0] = .4;
    p.model_coeffs[1] = .7;
    p.model_coeffs[2] = .2;
    p.model_coeffs[3] = .15;
    p.model_coeffs[4] = .1;
    const bool frozen = GENERATE(false, true);
    if (frozen) {
        p.req_radical = false;
        p.req_species = false;
        p.req_moment = false;
        p.req_model = false;
        p.req_les = false;
        p.train_init();
    }
    gradients(p);
    std::ostringstream out;
    p.Save(out);
    M loaded;
    std::istringstream reload(out.str());
    loaded.Load(reload);
    loaded.train_init();
    auto a = atoms(), b = atoms();
    p.getEFS(a);
    loaded.getEFS(b);
    REQUIRE(a.energy_ == Approx(b.energy_).margin(1e-11));
    REQUIRE(p.LES_all_latents.cols() == 2);
    model.channels = 1;
    REQUIRE_THROWS_AS(p.set_sog_definition(model), GPException);
}

TEST_CASE("Multichannel kernel fitting reduces loss with frozen heads", "[sog][channels][training]")
{
    PotMTP_LES p("tests/case-mtp-les/les.mtp");
    initialize(p);
    std::vector<Atoms> data;
    for (int i = 0; i < 4; ++i) {
        auto a = atoms();
        a.pos_[2](0) += .17 * i;
        a.id = i;
        p.getEFS(a);
        data.push_back(a);
    }
    for (auto& w : p.model_les.sog_weights)
        w *= .6;
    std::vector<Neighborhoods> nbhs;
    for (auto& a : data)
        nbhs.emplace_back(a, p.cutoff);
    GPSettings settings;
    settings.Modify("energy_weight", "1");
    settings.Modify("force_weight", "1");
    settings.Modify("stress_weight", ".1");
    settings.Modify("charge_weight", "0");
    settings.Modify("penalty_weight", "0");
    settings.Modify("parameter_scaling", "none");
    settings.Modify("bfgs_grad_tol", "1e-14");
    settings.Modify("bfgs_step_tol", "0");
    settings.Modify("bfgs_conv_tol", "0");
    TrainNonlinear trainer(settings);
    trainer.init(data, nbhs, &p);
    trainer.ConfigureTrainableGroups({ "sog_weights" });
    std::vector<double> initial(p.get_NL_coeffs_count()), fitted(initial.size()),
        penalty(initial.size(), 0.);
    p.nl_get_coeffs(initial.data());
    double floor = 0.;
    p.AddPenaltyGard(0., floor, penalty.data());
    Eigen::VectorXd gradient(p.get_NL_coeffs_count());
    double before = 0., after = 0.;
    trainer.AccumulateGard(before, gradient);
    trainer.TrainNonLinearBFGS(30);
    trainer.AccumulateGard(after, gradient);
    p.nl_get_coeffs(fitted.data());
    REQUIRE(before - floor > 1e-10);
    REQUIRE(after - floor < (before - floor) * .01);
    for (const auto& group : p.nl_parameter_groups())
        if (group.name != "sog_weights")
            for (int i = group.begin; i < group.end; ++i)
                REQUIRE(initial[i] == Approx(fitted[i]).margin(1e-12));
}
