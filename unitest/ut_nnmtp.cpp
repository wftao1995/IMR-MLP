// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "NNBase/nn1hidden.h"
#include "NNBase/nn2hidden.h"
#include "apps/apps.h"
#include "base/neighborhood.h"
#include "io/gpio.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_wrapper.h"
#include "utils/gputils.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

using Catch::Approx;

static GPSettings nnmtp_settings(const std::string &mtp,
                                 const std::string &nn,
                                 const std::string &sc,
                                 int nnlay = 1)
{
    GPSettings settings;
    settings.Modify("mtpcoeff", mtp);
    settings.Modify("nncoeff", nn);
    settings.Modify("spcoeff", sc);
    settings.Modify("nnlay", std::to_string(nnlay));
    settings.Modify("nnhidden", "4");
    settings.Modify("nnhidden2", "3");
    return settings;
}

static std::vector<char *> argv_from(std::vector<std::string> &args)
{
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &arg : args)
        argv.push_back(arg.data());
    return argv;
}

static void remove_prefix_outputs(const std::string &prefix)
{
    std::filesystem::remove(prefix + ".mtp");
    std::filesystem::remove(prefix + ".nn");
    std::filesystem::remove(prefix + ".sc");
    std::filesystem::remove(prefix + ".stat");
}

TEST_CASE("PotMTPNN1 regularizes NN coefficients at the correct offset", "[nnmtp][regularization]")
{
    Potentials::PotMTPNN1 pot(nnmtp_settings("tests/test-nnmtp/_Trained_Potential_.mtp",
                                             "tests/test-nnmtp/_Trained_Potential_.nn",
                                             "tests/test-nnmtp/_Trained_Potential_.sc"));
    pot.train_init();
    std::fill(pot.reg_L1_vector.begin(), pot.reg_L1_vector.end(), 0.0);
    std::fill(pot.reg_L2_vector.begin(), pot.reg_L2_vector.end(), 1.0);

    const int ncoeff = pot.get_NL_coeffs_count();
    std::vector<double> grad(ncoeff, 0.0);
    double loss = 0.0;
    pot.AddPenaltyGard(0.0, loss, grad.data());

    const int nn_offset = pot.p_mtp->radial_coeff_size + pot.p_mtp->species_count;
    double before_norm = 0.0;
    double nn_norm = 0.0;
    for (int i = 0; i < nn_offset; ++i)
        before_norm += std::abs(grad[i]);
    for (int i = nn_offset; i < nn_offset + pot.nn_coeff_size; ++i)
        nn_norm += std::abs(grad[i]);

    REQUIRE(loss > 0.0);
    REQUIRE(before_norm == Approx(0.0).margin(1e-14));
    REQUIRE(nn_norm > 0.0);
}

TEST_CASE("PotMTPNN1 Load rejects missing species coeffs", "[nnmtp][load]")
{
    const std::string prefix = "/tmp/ut_nnmtp_missing_sc";
    remove_prefix_outputs(prefix);
    std::filesystem::copy_file("tests/test-nnmtp/_Trained_Potential_.mtp",
                               prefix + ".mtp",
                               std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file("tests/test-nnmtp/_Trained_Potential_.nn",
                               prefix + ".nn",
                               std::filesystem::copy_options::overwrite_existing);

    REQUIRE_THROWS_AS(Potentials::PotMTPNN1(prefix), GPException);
    remove_prefix_outputs(prefix);
}

TEST_CASE("PotMTPNN1 saves two-layer NN coefficients to .nn", "[nnmtp][nn2]")
{
    Potentials::PotMTPNN1 pot(nnmtp_settings("tests/test-nnmtp/_Trained_Potential_.mtp",
                                             "/tmp/ut_nnmtp_missing_input.nn",
                                             "/tmp/ut_nnmtp_missing_input.sc",
                                             2));
    pot.random_init();

    const std::string prefix = "/tmp/ut_nnmtp_two_layer";
    remove_prefix_outputs(prefix);
    pot.Save(prefix);

    IMRMLPNN::NN2lay loaded(prefix + ".nn");
    REQUIRE(loaded.n_input == pot.p_mtp->alpha_scalar_moments + 1);
    REQUIRE(loaded.n_neuron1 == 4);
    REQUIRE(loaded.n_neuron2 == 3);

    remove_prefix_outputs(prefix);
}

TEST_CASE("nnmtp command writes descriptor normalization with per-site statistics", "[nnmtp][normalize]")
{
    const std::string prefix = "/tmp/ut_nnmtp_template";
    remove_prefix_outputs(prefix);

    std::vector<std::string> args = {
        "nnmtp",
        "--potfile=tests/test-nnmtp/_Trained_Potential_.mtp",
        "--data=tests/test-nnmtp/sub.cfg",
        "--input-format=mlpcfg",
        "--index=tests/test-nnmtp/index.map",
        "--output=" + prefix,
        "--initmtp=false",
        "--normalize=true",
        "--nneuron=4",
    };
    std::vector<char *> argv = argv_from(args);
    REQUIRE(nnmtp(static_cast<int>(argv.size()), argv.data()) == 0);

    GPIO io;
    io.readmap("tests/test-nnmtp/index.map");
    std::vector<Atoms> atoms;
    io.read(atoms, "tests/test-nnmtp/sub.cfg", "mlpcfg");

    Potentials::PotMTP pot("tests/test-nnmtp/_Trained_Potential_.mtp");
    std::vector<double> desc;
    std::vector<double> sum(pot.alpha_scalar_moments, 0.0);
    std::vector<double> sqsum(pot.alpha_scalar_moments, 0.0);
    size_t site_count = 0;

    for (auto &atom : atoms) {
        Neighborhoods nbhs(atom, pot.cutoff);
        for (int iat = 0; iat < atom.size(); ++iat) {
            pot.compute_descriptors(nbhs[iat], desc);
            for (int i = 0; i < pot.alpha_scalar_moments; ++i)
                sum[i] += desc[i + 1];
            ++site_count;
        }
    }
    REQUIRE(site_count > 0);
    for (double &v : sum)
        v /= static_cast<double>(site_count);

    for (auto &atom : atoms) {
        Neighborhoods nbhs(atom, pot.cutoff);
        for (int iat = 0; iat < atom.size(); ++iat) {
            pot.compute_descriptors(nbhs[iat], desc);
            for (int i = 0; i < pot.alpha_scalar_moments; ++i) {
                double delta = desc[i + 1] - sum[i];
                sqsum[i] += delta * delta;
            }
        }
    }
    for (double &v : sqsum)
        v = std::sqrt(v / static_cast<double>(site_count));

    IMRMLPNN::NN1lay nn(prefix + ".nn");
    REQUIRE(nn.n_input == pot.alpha_scalar_moments + 1);
    for (int i = 0; i < std::min(8, pot.alpha_scalar_moments); ++i) {
        double scale = sqsum[i] > std::numeric_limits<double>::epsilon() ? sqsum[i] : 1.0;
        REQUIRE(nn.norm_mean(i + 1) == Approx(sum[i]).margin(1e-10));
        REQUIRE(nn.norm_irstd(i + 1) == Approx(1.0 / scale).epsilon(1e-10));
    }

    remove_prefix_outputs(prefix);
}

static Atoms nnmtp_dimer() {
    Atoms atom;
    atom.cell_ = Eigen::Matrix3d::Identity() * 20;
    atom.pbc_ = {false, false, false};
    atom.pos_ = {{0, 0, 0}, {2, 0.2, 0}};
    atom.chemmap_ = {0, 1};
    return atom;
}

TEST_CASE("NNMTP factory reload preserves both network architectures", "[nnmtp][io]") {
    const int layers = GENERATE(1, 2);
    const bool legacy = GENERATE(false, true);
    Potentials::PotMTPNN1 original(nnmtp_settings("tests/test-nnmtp/_Trained_Potential_.mtp",
                                                "unused.nn", "unused.sc", layers));
    std::vector<double> coefficients(original.nn_coeff_size, 0.03);
    if (layers == 1) {
        original.p_nn->set_coeffs(coefficients.data());
        original.p_nn->acfunc = legacy ? IMRMLPNN::TANH : IMRMLPNN::SIGMOID;
    } else {
        original.p_nn2->set_coeffs(coefficients.data());
        original.p_nn2->acfunc = legacy ? IMRMLPNN::TANH : IMRMLPNN::SIGMOID;
    }
    original.species_coeffs = {-1.25, 0.7};
    original.Save("nn-roundtrip");
    GPSettings settings;
    if (legacy) {
        // Strip the versioned header to exercise the original dimension/array layout.
        std::ifstream input("nn-roundtrip.nn", std::ios::binary);
        input.seekg(4 * sizeof(std::int32_t));
        const std::string payload{std::istreambuf_iterator<char>(input), {}};
        input.close();
        std::ofstream output("nn-roundtrip.nn", std::ios::binary);
        output.write(payload.data(), payload.size());
        settings.Modify("nnlay", std::to_string(layers));
    }
    Potentials::potential_wrapper restored("nn-roundtrip", "mtpnn1", settings);
    const auto *nn = dynamic_cast<Potentials::PotMTPNN1 *>(restored.pot);
    REQUIRE(nn != nullptr);
    REQUIRE(nn->nnlay == layers);
    REQUIRE(nn->get_NL_coeffs_count() == original.get_NL_coeffs_count());
    auto expected = nnmtp_dimer();
    Neighborhoods nbhs(expected, original.cutoff);
    original.getEFS(expected, nbhs);
    for (bool omp : {false, true}) {
        auto atom = nnmtp_dimer();
        if (omp) restored.pot->getEFSOMP(atom, nbhs);
        else restored.pot->getEFS(atom, nbhs);
        REQUIRE(atom.energy_ == Approx(expected.energy_).margin(1e-10));
        REQUIRE(atom.stress_.isApprox(expected.stress_, 1e-9));
        for (int i = 0; i < atom.size(); ++i)
            REQUIRE(atom.forces_[i].isApprox(expected.forces_[i], 1e-9));
    }
}

TEST_CASE("NNMTP rejects descriptor dimension mismatches at both loading entry points", "[nnmtp][io]") {
    const int layers = GENERATE(1, 2);
    if (layers == 1) {
        IMRMLPNN::NN1lay wrong(1, 2, 1);
        wrong.random_init();
        wrong.dump_coeffs("mismatch.nn");
    } else {
        IMRMLPNN::NN2lay wrong(1, 2, 2, 1);
        wrong.dump_coeffs("mismatch.nn");
    }
    auto settings = nnmtp_settings("tests/test-nnmtp/_Trained_Potential_.mtp",
                                   "mismatch.nn", "tests/test-nnmtp/_Trained_Potential_.sc", layers);
    REQUIRE_THROWS_WITH(Potentials::PotMTPNN1(settings), Catch::Matchers::ContainsSubstring("input dimension"));
    std::filesystem::copy_file("tests/test-nnmtp/_Trained_Potential_.mtp", "mismatch.mtp",
                              std::filesystem::copy_options::overwrite_existing);
    std::filesystem::copy_file("tests/test-nnmtp/_Trained_Potential_.sc", "mismatch.sc",
                              std::filesystem::copy_options::overwrite_existing);
    REQUIRE_THROWS_WITH(Potentials::PotMTPNN1(std::string("mismatch")), Catch::Matchers::ContainsSubstring("input dimension"));
}

TEST_CASE("NNMTP energy-only paths preserve forces and stress", "[nnmtp]") {
    const bool omp = GENERATE(false, true);
    const bool add = GENERATE(false, true);
    Potentials::PotMTPNN1 pot(std::string("tests/test-nnmtp/_Trained_Potential_"));
    auto atom = nnmtp_dimer();
    Neighborhoods nbhs(atom, pot.cutoff);
    pot.getEFS(atom, nbhs);
    const double expected = atom.energy_;
    const auto forces = atom.forces_;
    const auto stress = atom.stress_;
    pot.add_efs = add;
    if (omp) pot.getEOMP(atom, nbhs);
    else pot.getE(atom, nbhs);
    REQUIRE(atom.energy_ == Approx((add ? 2 : 1) * expected));
    REQUIRE(atom.forces_ == forces);
    REQUIRE(atom.stress_ == stress);
    REQUIRE_FALSE(atom.have_stress);
}

TEST_CASE("NNMTP save reports species coefficient output failures", "[nnmtp][io]") {
    Potentials::PotMTPNN1 pot(std::string("tests/test-nnmtp/_Trained_Potential_"));
    std::filesystem::create_directory("blocked-nn.sc");
    REQUIRE_THROWS_WITH(pot.Save("blocked-nn"), Catch::Matchers::ContainsSubstring("blocked-nn.sc"));
    std::filesystem::remove("blocked-nn.sc");
    if (!std::filesystem::exists("/dev/full")) SKIP("/dev/full unavailable");
    std::filesystem::create_symlink("/dev/full", "blocked-nn.sc");
    REQUIRE_THROWS_WITH(pot.Save("blocked-nn"), Catch::Matchers::ContainsSubstring("blocked-nn.sc"));
    std::filesystem::remove("blocked-nn.sc");
}

namespace {
struct NNCommandWorkspace {
    std::filesystem::path dir;
    std::ostringstream warnings;
    std::ostream *previous_debug = nullptr;
    NNCommandWorkspace() {
        char name[] = "/tmp/imrmlp_nn_command_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("Cannot create NNMTP test directory");
        dir = created;
        previous_debug = SetStreamForDebug(&warnings);
    }
    ~NNCommandWorkspace() {
        CMDOption::global_cmdopts.reset();
        SetStreamForDebug(previous_debug);
        std::filesystem::remove_all(dir);
    }
    std::string prefix() const { return (dir / "out").string(); }
    std::string data() const { return (dir / "train.cfg").string(); }
    int run(std::vector<std::string> extra = {}) {
        CMDOption::global_cmdopts.reset();
        std::vector<std::string> args = {
            "nnmtp", "--potfile=tests/test-nnmtp/_Trained_Potential_.mtp",
            "--index=tests/test-nnmtp/index.map", "--input-format=mlpcfg",
            "--data=" + data(), "--output=" + prefix(), "--initmtp=false", "--nneuron=4"
        };
        args.insert(args.end(), extra.begin(), extra.end());
        auto argv = argv_from(args);
        return nnmtp(argv.size(), argv.data());
    }
    void frame(int type, bool labeled, double energy = 0, bool append = true) {
        std::ofstream out(data(), append ? std::ios::app : std::ios::out);
        out << "BEGIN_CFG\nSize\n1\nSupercell\n20 0 0\n0 20 0\n0 0 20\n"
               "AtomData: id type cartes_x cartes_y cartes_z\n1 " << type << " 0 0 0\n";
        if (labeled) out << "Energy\n" << energy << "\n";
        out << "END_CFG\n\n";
    }
};
}

TEST_CASE("NNMTP skips missing energy labels while retaining normalization samples", "[nnmtp][command]") {
    NNCommandWorkspace work;
    work.frame(0, true, -2);
    work.frame(1, true, -4);
    REQUIRE(work.run() == 0);
    IMRMLPNN::NN1lay labeled(work.prefix() + ".nn");
    // Both species were sampled once. An unlabeled type-0 dimer must
    // change normalization, while leaving the fitted energy offsets alone.
    {
        std::ofstream out(work.data(), std::ios::app);
        out << "BEGIN_CFG\nSize\n2\nSupercell\n20 0 0\n0 20 0\n0 0 20\n"
               "AtomData: id type cartes_x cartes_y cartes_z\n1 0 0 0 0\n2 0 2.5 0 0\nEND_CFG\n\n";
    }
    REQUIRE(work.run() == 0);
    std::ifstream sc(work.prefix() + ".sc");
    double first, second;
    REQUIRE(static_cast<bool>(sc >> first >> second));
    REQUIRE(first == Approx(-2));
    REQUIRE(second == Approx(-4));
    REQUIRE(work.warnings.str().find("skipped 1 structures without energy labels") != std::string::npos);
    IMRMLPNN::NN1lay mixed(work.prefix() + ".nn");
    REQUIRE(!mixed.norm_mean.isApprox(labeled.norm_mean));
    REQUIRE(mixed.norm_irstd.allFinite());
}

TEST_CASE("NNMTP supports templates with no energy labels explicitly", "[nnmtp][command]") {
    NNCommandWorkspace work;
    const int layers = GENERATE(1, 2);
    work.frame(0, false);
    work.frame(1, false);
    REQUIRE(work.run({"--nnlay=" + std::to_string(layers)}) == 0);
    REQUIRE(work.warnings.str().find("no energy labels") != std::string::npos);
    std::ifstream sc(work.prefix() + ".sc");
    double first, second;
    REQUIRE(static_cast<bool>(sc >> first >> second));
    REQUIRE(first == 0);
    REQUIRE(second == 0);
    Potentials::PotMTPNN1 model(work.prefix());
    REQUIRE(model.nnlay == layers);
    if (layers == 1) REQUIRE(model.p_nn->norm_irstd.allFinite());
    else REQUIRE(model.p_nn2->norm_irstd.allFinite());
}

TEST_CASE("NNMTP rejects missing empty and unreadable training data", "[nnmtp][command]") {
    NNCommandWorkspace work;
    SECTION("missing") {}
    SECTION("empty") { std::ofstream out(work.data()); }
    SECTION("malformed after valid structure") {
        work.frame(0, true, -2);
        std::ofstream out(work.data(), std::ios::app);
        out << "BEGIN_CFG\nSize\n1\nSupercell\n20 0 0\n0 20 0\n0 0 20\n"
               "AtomData: id type cartes_x cartes_y cartes_z\n1 0 bad 0 0\nEND_CFG\n";
    }
    REQUIRE_THROWS(work.run({"--normalize=false"}));
    REQUIRE_FALSE(std::filesystem::exists(work.prefix() + ".nn"));
}

TEST_CASE("NNMTP validates model dimensions and normalization extent", "[nnmtp][command]") {
    NNCommandWorkspace work;
    work.frame(0, true, -2);
    const auto options = GENERATE(std::vector<std::string>{"--nneuron=0"},
        std::vector<std::string>{"--nneuron=-1"},
        std::vector<std::string>{"--nnlay=2", "--nneuron2=0"},
        std::vector<std::string>{"--nnlay=2", "--nneuron2=-1"},
        std::vector<std::string>{"--nnlay=3"},
        std::vector<std::string>{"--extent=0"},
        std::vector<std::string>{"--extent=-1"},
        std::vector<std::string>{"--extent=nan"},
        std::vector<std::string>{"--extent=inf"});
    REQUIRE_THROWS(work.run(options));
    REQUIRE_FALSE(std::filesystem::exists(work.prefix() + ".nn"));
}

TEST_CASE("NNMTP ignores dimensions of unused hidden layers", "[nnmtp][command]") {
    NNCommandWorkspace work;
    work.frame(0, true, -2);
    REQUIRE(work.run({"--nnlay=1", "--nneuron2=-1"}) == 0);
    REQUIRE_NOTHROW(IMRMLPNN::NN1lay(work.prefix() + ".nn"));
}

TEST_CASE("NNMTP reports statistics and species output failures", "[nnmtp][command]") {
    NNCommandWorkspace work;
    work.frame(0, true, -2);
    const std::string suffix = GENERATE(".sc", ".stat");
    const bool full = GENERATE(false, true);
    const auto path = work.prefix() + suffix;
    if (full) std::filesystem::create_symlink("/dev/full", path);
    else std::filesystem::create_directory(path);
    REQUIRE_THROWS_WITH(work.run(), Catch::Matchers::ContainsSubstring(path));
}
