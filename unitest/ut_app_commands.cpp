// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include "potentials/potential_mtp_m.h"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <iomanip>
#include <sstream>
#include <tuple>

using Catch::Approx;

namespace {
struct Workspace {
    std::filesystem::path original_dir = std::filesystem::current_path();
    std::string mtp = std::filesystem::absolute("tests/test-nnmtp/_Trained_Potential_.mtp").string();
    std::filesystem::path path;
    std::ostringstream warnings;
    std::ostream *old_debug;

    Workspace() {
        char name[] = "/tmp/imrmlp_app_commands_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("Cannot create command test directory");
        path = dir;
        std::filesystem::current_path(path);
        std::ofstream("index.map") << "Al\nNi\n";
        old_debug = SetStreamForDebug(&warnings);
    }
    ~Workspace() {
        SetStreamForDebug(old_debug);
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(original_dir);
        std::filesystem::remove_all(path);
    }
    int run(std::vector<std::string> args) {
        CMDOption::global_cmdopts.reset();
        RNG.seed(1729);
        std::vector<char *> argv;
        for (auto &arg : args) argv.push_back(arg.data());
        return command_maps.at(args[0]).first(argv.size(), argv.data());
    }
    void make_template(bool fs = false) {
        std::ifstream in(mtp);
        REQUIRE(in.good());
        std::ofstream out("template.mtp");
        std::string line;
        while (std::getline(in, line)) {
            if (line.starts_with("species_coeffs")) break;
            out << line << '\n';
        }
        out << (fs ? "species_coeffs = {0, 0, 0, 0, 0, 0}\n" : "species_coeffs = {0, 0, 0, 0}\n");
        while (std::getline(in, line)) {
            if (line.starts_with("moment_coeffs")) break;
        }
        REQUIRE(line.starts_with("moment_coeffs"));
        if (fs) {
            auto vals = line.substr(line.find('{') + 1);
            vals = vals.substr(0, vals.find('}'));
            out << "moment_coeffs = {" << vals << ", " << vals << "}\nmodel_coeffs = {0.7}\n";
        } else {
            out << line << "\nmodel_coeffs = {}\n";
        }
        std::ofstream("init.ini") << "pot:potfile template.mtp\npot:extmodel "
            << (fs ? "mtp_m_fs" : "mtp_m_li")
            << "\npotfile out.mtp\ndataset train.cfg\nformat mlpcfg\nindex index.map\n";
    }
    std::vector<std::string> efs_args() {
        return {"calc_efs", "--input=train.cfg", "--output=out.cfg",
                "--input-format=mlpcfg", "--output-format=mlpcfg", "--index=index.map",
                "--pottype=mtp", "--potfile=" + mtp};
    }
};

Atoms dimer(int type = 0, double distance = 2.5, bool labeled = true) {
    Atoms at;
    at.cell_ = 20 * Eigen::Matrix3d::Identity();
    at.pos_ = {{0, 0, 0}, {distance, 0, 0}};
    at.chemmap_ = {type, type};
    at.have_energy = labeled;
    at.energy_ = type == 0 ? -4 : -8;
    return at;
}

void write_frames(std::vector<Atoms> frames) {
    GPIO io;
    REQUIRE(io.write(frames, "train.cfg", "mlpcfg"));
}

std::vector<Atoms> read_frames(const std::string &path, const std::string &format) {
    GPIO io;
    io.readmap("index.map");
    std::vector<Atoms> frames;
    REQUIRE(io.read(frames, path, format));
    return frames;
}

struct Coefficients {
    std::vector<double> radial, species, moment, model;
};

Coefficients load_coefficients(const std::string &file, bool fs = false) {
    GPSettings settings;
    settings.Modify("potfile", file);
    settings.Modify("extmodel", fs ? "mtp_m_fs" : "mtp_m_li");
    std::unique_ptr<Potentials::AnyPotential> pot(Potentials::create_mtp_m_potential(settings));
    auto *mtp = dynamic_cast<Potentials::PotMTP *>(pot.get());
    REQUIRE(mtp != nullptr);
    int nr, ns, nm, nx;
    double *r, *s, *m, *x;
    mtp->get_n_coeffs(nr, ns, nm, nx);
    mtp->get_pointers(r, s, m, x);
    Coefficients c;
    c.radial.assign(r, r + nr);
    c.species.assign(s, s + ns);
    c.moment.assign(m, m + nm);
    if (nx > 0) c.model.assign(x, x + nx);
    return c;
}
}

TEST_CASE("initmtpm preserves existing coefficients when randomization is disabled", "[apps][initmtpm]") {
    Workspace work;
    work.make_template(true);
    write_frames({dimer(), dimer(1)});
    auto original = load_coefficients("template.mtp", true);
    REQUIRE(work.run({"initmtpm", "--settings=init.ini", "--randomize=false"}) == 0);
    auto kept = load_coefficients("out.mtp", true);
    REQUIRE_THAT(kept.radial, Catch::Matchers::Approx(original.radial));
    REQUIRE_THAT(kept.moment, Catch::Matchers::Approx(original.moment));
    REQUIRE_THAT(kept.model, Catch::Matchers::Approx(original.model));
    REQUIRE(kept.species[0] == Approx(-2));
    REQUIRE(kept.species[1] == Approx(-4));

    REQUIRE(work.run({"initmtpm", "--settings=init.ini", "--randomize=true"}) == 0);
    auto randomized = load_coefficients("out.mtp", true);
    REQUIRE(randomized.radial != kept.radial);
    REQUIRE(randomized.moment != kept.moment);
    REQUIRE(randomized.model != kept.model);
}

TEST_CASE("initmtpm skips missing energy labels while fitting species offsets", "[apps][initmtpm]") {
    const bool all_unlabeled = GENERATE(false, true);
    Workspace work;
    work.make_template();
    if (all_unlabeled) write_frames({dimer(0, 2.5, false), dimer(1, 2.5, false)});
    else write_frames({dimer(), dimer(1), dimer(0, 3, false)});
    REQUIRE(work.run({"initmtpm", "--settings=init.ini"}) == 0);
    const auto coefficients = load_coefficients("out.mtp");
    REQUIRE(coefficients.species[0] == Approx(all_unlabeled ? 0 : -2));
    REQUIRE(coefficients.species[1] == Approx(all_unlabeled ? 0 : -4));
    REQUIRE(work.warnings.str().find("without energy labels") != std::string::npos);
    if (all_unlabeled)
        REQUIRE(work.warnings.str().find("offsets are initialized to zero") != std::string::npos);
}

TEST_CASE("initmtpm rejects missing empty and truncated training datasets", "[apps][initmtpm]") {
    const std::string kind = GENERATE("missing", "empty", "truncated");
    Workspace work;
    work.make_template();
    if (kind == "empty") std::ofstream("train.cfg");
    if (kind == "truncated") {
        write_frames({dimer()});
        std::ofstream("train.cfg", std::ios::app) << "BEGIN_CFG\nSize\n";
    }
    REQUIRE_THROWS_AS(work.run({"initmtpm", "--settings=init.ini"}), GPException);
    REQUIRE_FALSE(std::filesystem::exists("out.mtp"));
}

TEST_CASE("calc_efs rejects a truncated trajectory after valid frames", "[apps][calc_efs]") {
    Workspace work;
    write_frames({dimer()});
    std::ofstream("train.cfg", std::ios::app) << "BEGIN_CFG\nSize\n";
    REQUIRE_THROWS_AS(work.run(work.efs_args()), GPException);
    REQUIRE_FALSE(std::filesystem::exists("out.cfg"));
}

TEST_CASE("calc_efs requires explicitly requested settings and allows absent defaults", "[apps][calc_efs]") {
    const bool explicit_settings = GENERATE(false, true);
    Workspace work;
    write_frames({dimer(), dimer(1)});
    auto args = work.efs_args();
    if (explicit_settings) {
        args.push_back("--settings=missing.ini");
        REQUIRE_THROWS_AS(work.run(args), GPException);
        REQUIRE_FALSE(std::filesystem::exists("out.cfg"));
    } else {
        REQUIRE(work.run(args) == 0);
        auto frames = read_frames("out.cfg", "mlpcfg");
        REQUIRE(frames.size() == 2);
        REQUIRE(frames[0].have_energy);
        REQUIRE(frames[0].forces_.size() == 2);
    }
}

TEST_CASE("FPS command respects both count and distance limits", "[apps][fps]") {
    auto [count, threshold, expected] = GENERATE(
        std::make_tuple(3, "1e20", 1), std::make_tuple(0, "1e20", 1),
        std::make_tuple(2, "1e-20", 2), std::make_tuple(2, "0", 2));
    Workspace work;
    write_frames({dimer(0, 2), dimer(0, 2.5), dimer(0, 3)});
    REQUIRE(work.run({"diverse_select", "--pool=train.cfg", "--format=mlpcfg",
                     "--output=out.cfg", "--index=index.map", "--potfile=" + work.mtp,
                     "--first-idx=0", "--n-select=" + std::to_string(count),
                     "--threshold-distance=" + std::string(threshold)}) == 0);
    REQUIRE(read_frames("out.cfg", "mlpcfg").size() == expected);
}

TEST_CASE("Fused energy components keep every moment and sum to the site energy", "[apps][descriptors]") {
    Workspace work;
    auto at = dimer();
    at.chemmap_[1] = 1;
    write_frames({at});
    REQUIRE(work.run({"calc_descriptors", "--input=train.cfg", "--input-format=mlpcfg",
        "--potfile=" + work.mtp, "--output=desc.bin", "--output-format=bin",
        "--local-energy-components=true", "--fuse-species-info=true"}) == 0);
    auto frames = read_frames("desc.bin", "bin");
    Potentials::PotMTP pot(work.mtp);
    pot.getEFS(at);
    const auto &local = frames[0].local_attribs_v_.at("desc");
    REQUIRE(local.rows() == pot.get_alpha_scalar_moments() + 1);
    REQUIRE(local.cols() == at.size());
    for (int i = 0; i < at.size(); ++i)
        REQUIRE(local.col(i).sum() == Approx(at.site_energies_[i]).margin(1e-12));
    REQUIRE(frames[0].global_attribs_v_.at("desc").sum() == Approx(at.energy_ / at.size()));
}

TEST_CASE("Discarding local descriptors preserves global results for every branch", "[apps][descriptors]") {
    const std::string type = GENERATE("mtp", "mtm", "bondorder");
    Workspace work;
    auto at = dimer();
    at.local_attribs_v_["desc"] = Eigen::MatrixXd::Ones(3, at.size());
    GPIO io;
    std::vector<Atoms> frames{at};
    REQUIRE(io.write(frames, "in.bin", "bin"));
    std::ofstream("desc.ini") << "bondorder:cutoff 5\n";
    std::vector<std::string> args = {"calc_descriptors", "--input=in.bin", "--input-format=bin",
        "--dtype=" + type, "--settings=desc.ini", "--potfile=" + work.mtp,
        "--output=desc.bin", "--output-format=bin"};
    REQUIRE(work.run(args) == 0);
    auto global = read_frames("desc.bin", "bin")[0].global_attribs_v_.at("desc");
    args.push_back("--discard-local-components=true");
    REQUIRE(work.run(args) == 0);
    auto result = read_frames("desc.bin", "bin")[0];
    REQUIRE(result.local_attribs_v_.count("desc") == 0);
    REQUIRE(result.global_attribs_v_.at("desc").isApprox(global));
}

TEST_CASE("Descriptor cutoff overrides configure the descriptor before initialization", "[apps][descriptors]") {
    const std::string type = GENERATE("bondorder", "behler");
    CAPTURE(type);
    Workspace work;
    auto at = dimer();
    at.pos_.push_back({0, 2.5, 0});
    at.chemmap_.push_back(0);
    write_frames({at});
    auto config = [&](int cutoff) {
        std::ofstream out("desc.ini");
        out << type << ":cutoff " << cutoff << '\n';
        out << "behler:dispcount 1\n";
    };
    std::vector<std::string> args = {"calc_descriptors", "--input=train.cfg", "--input-format=mlpcfg",
        "--dtype=" + type, "--settings=desc.ini", "--output=desc.bin", "--output-format=bin"};
    config(5);
    REQUIRE(work.run(args) == 0);
    auto large = read_frames("desc.bin", "bin")[0].global_attribs_v_.at("desc");
    auto override_args = args;
    override_args.push_back("--cutoff=2");
    REQUIRE(work.run(override_args) == 0);
    auto overridden = read_frames("desc.bin", "bin")[0].global_attribs_v_.at("desc");
    config(2);
    REQUIRE(work.run(args) == 0);
    auto small = read_frames("desc.bin", "bin")[0].global_attribs_v_.at("desc");
    REQUIRE(small.allFinite());
    REQUIRE(overridden.isApprox(small));
    REQUIRE((large - small).norm() > 1e-10);
}

TEST_CASE("Descriptor cutoff rejects invalid values and model overrides", "[apps][descriptors]") {
    auto [type, cutoff] = GENERATE(std::make_pair("bondorder", "0"),
        std::make_pair("bondorder", "-1"), std::make_pair("bondorder", "inf"),
        std::make_pair("mtp", "1"), std::make_pair("mtm", "1"));
    Workspace work;
    write_frames({dimer()});
    REQUIRE_THROWS(work.run({"calc_descriptors", "--input=train.cfg", "--input-format=mlpcfg",
        "--dtype=" + std::string(type), "--cutoff=" + std::string(cutoff),
        "--potfile=" + work.mtp, "--output=desc.xyz"}));
    REQUIRE_FALSE(std::filesystem::exists("desc.xyz"));
}

TEST_CASE("EOS invalid equilibrium and interpolation requests preserve output files", "[apps][eos]") {
    const std::string kind = GENERATE("no_minimum", "negative_root", "one_point", "zero_points");
    Workspace work;
    {
        std::ofstream curve("curve.dat");
        curve << std::setprecision(17);
        for (int v = 8; v <= 16; ++v) {
            double x = std::pow(v, -2.0 / 3.0);
            double e = (x - .2) * (x - .2);
            if (kind == "no_minimum") e = x + x*x*x;
            if (kind == "negative_root") e = x + x*x + .1*x*x*x;
            curve << v << ' ' << e << '\n';
        }
    }
    std::ofstream("fit.dat") << "previous fit\n";
    std::ofstream("veph.dat") << "previous interpolation\n";
    auto args = std::vector<std::string>{"calc_eos", "--fit-only", "--output=curve.dat",
        "--output-eos=fit.dat", "--out-veph=veph.dat"};
    if (kind == "one_point") args.push_back("--ninterp=1");
    if (kind == "zero_points") args.push_back("--ninterp=0");
    REQUIRE_THROWS_AS(work.run(args), GPException);
    std::string line;
    REQUIRE(std::getline(std::ifstream("fit.dat"), line));
    REQUIRE(line == "previous fit");
    REQUIRE(std::getline(std::ifstream("veph.dat"), line));
    REQUIRE(line == "previous interpolation");
}

TEST_CASE("EOS keeps valid quadratic and cubic equilibrium parameters finite", "[apps][eos]") {
    const bool quadratic = GENERATE(false, true);
    Workspace work;
    {
        std::ofstream curve("curve.dat");
        curve << std::setprecision(17);
        for (int v = 8; v <= 16; ++v) {
            const double x = std::pow(v, -2.0 / 3.0);
            const double e = quadratic ? (x - .2)*(x - .2) : .004 - .3*x*x + x*x*x;
            curve << v << ' ' << e << '\n';
        }
    }
    REQUIRE(work.run({"calc_eos", "--fit-only", "--output=curve.dat",
        "--output-eos=fit.dat", "--out-veph=veph.dat", "--ninterp=2"}) == 0);
    std::ifstream fit("fit.dat");
    std::string line;
    std::map<std::string, double> parameters;
    while (std::getline(fit, line)) {
        auto equal = line.find('=');
        if (equal == std::string::npos || line.starts_with("E  ")) continue;
        if (line.starts_with("V0") || line.starts_with("B0") || line.starts_with("E0")) {
            double value = std::stod(line.substr(equal + 1));
            REQUIRE(std::isfinite(value));
            parameters[line.substr(0, 2)] = value;
            if (line.starts_with("B0'")) REQUIRE(value == Approx(quadratic ? 4 : 16.0 / 3).margin(1e-6));
        }
    }
    REQUIRE(parameters.at("V0") == Approx(std::pow(.2, -1.5)).margin(1e-6));
    REQUIRE(parameters.at("E0") == Approx(0).margin(1e-6));
    const double expected_bulk = 8.0 / 9.0 * (quadratic ? 1 : .3) * std::pow(.2, 3.5);
    REQUIRE(parameters.at("B0") == Approx(expected_bulk).margin(1e-6));
    std::ifstream interpolated("veph.dat");
    std::getline(interpolated, line);
    int rows = 0;
    double v, e, p, h;
    while (interpolated >> v >> e >> p >> h) {
        REQUIRE(std::isfinite(e)); REQUIRE(std::isfinite(p)); REQUIRE(std::isfinite(h));
        ++rows;
    }
    REQUIRE(rows == 2);
}

TEST_CASE("descriptor commands reject invalid settings before writing output", "[apps][descriptors]") {
    const std::string config = GENERATE(
        "behler:dispcount 2\nbehler:dispselect 1\n", "behler:dispcount 6\n",
        "bondorder:maxl 3\n", "bondorder:maxl 0\n");
    Workspace work;
    write_frames({dimer()});
    std::ofstream("desc.ini") << "behler:cutoff 5\n" << config;
    const std::string type = config.substr(0, config.find(':'));
    REQUIRE_THROWS(work.run({"calc_descriptors", "--input=train.cfg", "--input-format=mlpcfg",
                            "--dtype=" + type, "--settings=desc.ini", "--output=out.xyz"}));
    REQUIRE_FALSE(std::filesystem::exists("out.xyz"));
}



TEST_CASE("Descriptor settings file is optional only when not explicitly requested", "[apps][descriptors]") {
    Workspace work;
    write_frames({dimer()});
    std::vector<std::string> args = {"calc_descriptors", "--input=train.cfg", "--input-format=mlpcfg",
        "--dtype=bondorder", "--output=desc.bin", "--output-format=bin"};
    SECTION("explicit missing file fails before writing output") {
        args.push_back("--settings=missing.ini");
        REQUIRE_THROWS_WITH(work.run(args), Catch::Matchers::ContainsSubstring("Cannot open file: missing.ini"));
        REQUIRE_FALSE(std::filesystem::exists("desc.bin"));
    }
    SECTION("absent default file uses descriptor defaults") {
        REQUIRE(work.run(args) == 0);
        REQUIRE(read_frames("desc.bin", "bin")[0].local_attribs_v_.at("desc").rows() == 4);
    }
    SECTION("existing default file is loaded") {
        std::ofstream("input.inp") << "bondorder:maxl 6\n";
        REQUIRE(work.run(args) == 0);
        REQUIRE(read_frames("desc.bin", "bin")[0].local_attribs_v_.at("desc").rows() == 6);
    }
}


TEST_CASE("Findint preserves the cubic cavity geometry in both output modes", "[apps][findint][io]") {
    Workspace work;
    std::ofstream("host.xyz") << "1\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3\nH 0 0 0\n";
    const bool sites_only = GENERATE(false, true);
    std::vector<std::string> arguments{"findint", "--input=host.xyz", "--input-format=extxyz",
        "--output=sites.xyz", "--output-format=extxyz", "--top=1", "--insert=H"};
    if (sites_only) arguments.push_back("--sites-only");
    REQUIRE(work.run(arguments) == 0);
    GPIO io;
    Atoms result;
    REQUIRE(io.read_one(result, "sites.xyz", "extxyz"));
    REQUIRE(result.size() == (sites_only ? 1 : 2));
    REQUIRE(result.cell_.isApprox(4 * Eigen::Matrix3d::Identity()));
    if (!sites_only) REQUIRE(result.pos_[0].isZero());
    Eigen::RowVector3d fraction = result.pos_.back() * result.cell_.inverse();
    fraction.array() -= fraction.array().floor();
    REQUIRE(fraction.isApprox(Eigen::RowVector3d(.5, .5, .5)));
}

TEST_CASE("Findint rejects an incomplete input suffix before searching or writing", "[apps][findint][io]") {
    Workspace work;
    std::ofstream("host.xyz") << "1\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3\nH 0 0 0\n"
                             << "2\nProperties=species:S:1:pos:R:3\nH 0 0 0\n";
    std::ofstream("sites.xyz") << "existing output\n";
    REQUIRE(work.run({"findint", "--input=host.xyz", "--input-format=extxyz",
                      "--output=sites.xyz", "--output-format=extxyz", "--top=1"}) == 1);
    std::ifstream output("sites.xyz");
    REQUIRE(std::string(std::istreambuf_iterator<char>(output), {}) == "existing output\n");
}

TEST_CASE("Findint reports write failures in sites-only and appended output modes", "[apps][findint][io]") {
    Workspace work;
    std::ofstream("host.xyz") << "1\nLattice=\"4 0 0 0 4 0 0 0 4\" Properties=species:S:1:pos:R:3\nH 0 0 0\n";
    const bool sites_only = GENERATE(false, true);
    const std::string destination = sites_only ? "/dev/full" : "missing/sites.xyz";
    if (sites_only && !std::filesystem::exists(destination)) SKIP("/dev/full is unavailable");
    std::vector<std::string> arguments{"findint", "--input=host.xyz", "--input-format=extxyz",
        "--output=" + destination, "--output-format=extxyz", "--top=1"};
    if (sites_only) arguments.push_back("--sites-only");
    REQUIRE(work.run(arguments) == 1);
}
