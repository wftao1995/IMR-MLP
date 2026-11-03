// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "fixtures/mtp_linear_efs.h"
#include "train/train_mtp.h"
#include "train/training_data.h"
#include "train/conditional_linear.h"
#include "potentials/mtp_gpu_adapter.h"
#include "tools/picojson.h"
#include "io/gpio.h"
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    MPI_Init(&argc, &argv); mpi.InitComm(MPI_COMM_SELF);
    const int result = Catch::Session().run(argc, argv);
    MPI_Finalize(); return result;
}
#endif

namespace {
namespace fs = std::filesystem;
using Potentials::PotMTP;
struct Workspace {
    fs::path original = fs::current_path(), directory;
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    Workspace() {
        char name[] = "/tmp/gpml_gpu_trainmtp_cli_XXXXXX";
        const auto path = mkdtemp(name);
        if (!path) throw std::runtime_error("Cannot create CLI fixture");
        directory = path; fs::current_path(directory);
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        std::error_code error; fs::current_path(original, error); fs::remove_all(directory, error);
    }
};
struct Capture {
    std::ostringstream stream;
    std::streambuf *buffer = std::cout.rdbuf(stream.rdbuf());
    std::ostream *output = SetStreamForOutput(&stream);
    ~Capture() { SetStreamForOutput(output); std::cout.rdbuf(buffer); }
};
int invoke(std::vector<std::string> args) {
    CMDOption::global_cmdopts = CMDOption::CommandLineOptions();
    std::vector<char *> argv; for (auto &arg : args) argv.push_back(arg.data());
    return trainmtp(argv.size(), argv.data());
}
void settings(const std::string &backend, bool fast, const std::string &extra = "") {
    std::ofstream out("fit.ini");
    out << "fit:device " << backend << "\nfit:gpu_id 0\nfit:gpu_workspace_mib 1\n"
        << "fit:lfit_fastalgo " << (fast ? "true" : "false")
        << "\nfit:weight_L2 .01\nfit:do_orthogonalization false\n"
        << "fit:energy_weight 1\nfit:force_weight .05\nfit:stress_weight .001\n" << extra;
}
std::vector<std::string> arguments() {
    return {"trainmtp", "--linear-fit", "--potfile=input.mtp", "--trainset=train.bin",
        "--trainset-format=bin", "--index=index.map", "--settings=fit.ini"};
}
std::string contents(const fs::path &path) {
    std::ifstream in(path); return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}
void same_predictions(PotMTP &a, PotMTP &b, const std::vector<Atoms> &data) {
    for (auto reference : data) {
        auto other = reference; a.getEFS(reference); b.getEFS(other);
        auto av = Training::ConditionalLinearAdapter::efs_values(reference);
        auto bv = Training::ConditionalLinearAdapter::efs_values(other);
        REQUIRE(av.allFinite()); REQUIRE(bv.allFinite());
        REQUIRE((av-bv).lpNorm<Eigen::Infinity>() <= 2e-7 + 1e-8*bv.lpNorm<Eigen::Infinity>());
    }
}
}

TEST_CASE("trainmtp rejects explicit GPU unsupported actions before side effects", "[training][trainmtp-cli]") {
    Workspace work;
    const auto compiled = MLIPCore::GpuDirect::compiled_backend();
    for (const std::string extra : {"--do-df", "--do-lbfgs", "--do-matrix-a", "--calc-grad", "--check-grad"}) {
        settings(compiled == "cpu" ? "cuda" : compiled, true, "fit:debug true\n");
        auto args = arguments(); args.push_back(extra);
        REQUIRE_THROWS(invoke(args));
        REQUIRE_FALSE(fs::exists("DEBUG_0")); REQUIRE_FALSE(fs::exists("_Trained_Potential_.mtp"));
        REQUIRE(std::distance(fs::directory_iterator("."), fs::directory_iterator{}) == 1);
    }
}

#ifdef IMRMLP_GPU
TEST_CASE("trainmtp GPU commands publish verified models and recover from publication failure", "[gpu][training][trainmtp-cli]") {
    int devices = 0; std::string error;
    if (!MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&devices, &error) || !devices) {
        if (std::getenv("IMRMLP_REQUIRE_GPU")) FAIL("Required GPU unavailable: " << error);
        SKIP("No GPU: " << error);
    }
    const bool fast = GENERATE(false, true);
    PotMTP source; MTPLinearEFSFixture::load(source,4); source.train_init(); source.set_inited();
    std::vector<Atoms> data;
    for (int k = 0; k < 7; ++k) {
        Atoms a; a.cell_ = 12.*Eigen::Matrix3d::Identity();
        a.pos_ = {{2.,2.,2.}, {4.1+.1*k,2.2,2.}, {2.3,4.2,2.4}}; a.chemmap_ = {0,1,k%2};
        a.fitness = .4+.1*k; a.atomic_fitness = {0.,.5,2.}; source.getEFS(a);
        a.global_attribs_d_["fitness"] = a.fitness;
        a.local_attribs_d_["atomic_fitness"] = Eigen::Map<Eigen::VectorXd>(a.atomic_fitness.data(), a.size());
        data.push_back(a);
    }
    for (int i = 0; i < source.lin_coeff_size; ++i) source.linear_coeffs[i] += .02*std::sin(i+1);
    Workspace work; source.Save("input.mtp");
    GPIO io; io.setmap(); REQUIRE(io.write(data, "train.bin", "bin"));
    { std::ofstream out("index.map"); auto map = io.getmap(); out << map.at(0) << '\n' << map.at(1) << '\n'; }
    settings("cpu", fast); REQUIRE(invoke(arguments()) == 0);
    fs::rename("_Trained_Potential_.mtp", "cpu.mtp");
    settings(MLIPCore::GpuDirect::compiled_backend(), fast);
    // A directory at the final path must survive a failed native publication.
    fs::create_directory("_Trained_Potential_.mtp");
    std::ofstream("_Trained_Potential_.mtp/keep") << "keep\n";
    REQUIRE_THROWS(invoke(arguments())); REQUIRE(contents("_Trained_Potential_.mtp/keep") == "keep\n");
    fs::remove_all("_Trained_Potential_.mtp");
    std::string log;
    { Capture capture; REQUIRE(invoke(arguments()) == 0); log = capture.stream.str(); }
    const std::string marker = "trainmtp GPU linear report ";
    const auto start = log.find(marker); REQUIRE(start != std::string::npos);
    const auto begin = start+marker.size();
    picojson::value report;
    REQUIRE(picojson::parse(report, log.substr(begin, log.find('\n', begin)-begin)).empty());
    REQUIRE(report.get("verified").get<bool>());
    REQUIRE(report.get("observation_backend").get<std::string>() == MLIPCore::GpuDirect::compiled_backend());
    REQUIRE(report.get("acceptance_backend").get<std::string>() == "cpu/direct");
    REQUIRE(report.get("solve_backend").get<std::string>() == (fast ? "cpu/fast/eigen_ldlt" : "cpu/ordinary/alglib"));
    REQUIRE(report.get("objective_after").get<double>() <= report.get("objective_before").get<double>()+1e-10);
    REQUIRE(report.get("observations").get("blocks").get<double>() > 0.);
    PotMTP cpu("cpu.mtp"), gpu("_Trained_Potential_.mtp"); same_predictions(gpu, cpu, data);
    const auto published = contents("_Trained_Potential_.mtp");
    settings(MLIPCore::GpuDirect::compiled_backend(), fast, "fit:weight_L1 .1\n");
    REQUIRE_THROWS(invoke(arguments())); REQUIRE(contents("_Trained_Potential_.mtp") == published);
    for (const auto &entry : fs::directory_iterator("."))
        REQUIRE_FALSE(entry.path().filename().string().starts_with(".gpml-affine-save-"));
}
#endif
