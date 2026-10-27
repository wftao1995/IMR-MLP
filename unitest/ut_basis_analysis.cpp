// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "basis_analysis/BasisAnalyzer.h"
#include "basis_analysis/strategies/ThresholdPruning.h"
#include "potentials/potential_mtp.h"
#include <filesystem>
#include <fstream>

using namespace BasisAnalysis;
namespace {
struct ReportWorkspace {
    std::filesystem::path dir;
    ReportWorkspace() {
        char name[] = "/tmp/imrmlp_basis_report_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("Cannot create report test directory");
        dir = created;
    }
    ~ReportWorkspace() { std::filesystem::remove_all(dir); }
    std::string prefix() const { return (dir / "report").string(); }
    std::string text(const std::string &suffix) const {
        std::ifstream in(prefix() + suffix);
        return {std::istreambuf_iterator<char>(in), {}};
    }
};
Atoms dimer(double distance = 2.5) {
    Atoms a;
    a.cell_ = Eigen::Matrix3d::Identity() * 20;
    a.pbc_ = {false, false, false};
    a.pos_ = {{0, 0, 0}, {distance, 0, 0}};
    a.chemmap_ = {0, 1};
    return a;
}
}

TEST_CASE("Empty basis analysis supports both phased and full selection", "[basis_analysis]") {
    Potentials::PotMTP pot("tests/test-nnmtp/_Trained_Potential_.mtp");
    const bool zero_limit = GENERATE(false, true);
    BasisAnalyzer analyzer(pot, zero_limit ? 0 : 50000);
    const std::vector<Atoms> frames = zero_limit ? std::vector<Atoms>{dimer()} : std::vector<Atoms>{};
    auto result = analyzer.analyze(frames);
    REQUIRE(result->nsamples == 0);
    REQUIRE(result->nfeatures == pot.alpha_scalar_moments);
    REQUIRE(result->pearson.rows() == result->nfeatures);
    REQUIRE(result->pearson.cols() == result->nfeatures);
    REQUIRE(result->highly_correlated_pairs().empty());
    REQUIRE(analyzer.last_result() != nullptr);
    REQUIRE(analyzer.last_result()->nsamples == 0);
    auto phased = analyzer.apply_strategy(*result, std::make_unique<ThresholdPruning>());
    REQUIRE(phased.first.size() == pot.alpha_scalar_moments);
    REQUIRE(phased.second.empty());
    REQUIRE(analyzer.run(frames, std::make_unique<ThresholdPruning>()) == phased);
    ReportWorkspace report;
    REQUIRE_NOTHROW(analyzer.report().write_all(report.prefix()));
    REQUIRE(report.text("_summary.txt").find("Sampled atoms:             0") != std::string::npos);
}

TEST_CASE("New analysis refreshes cached data and clears previous strategy results", "[basis_analysis]") {
    Potentials::PotMTP pot("tests/test-nnmtp/_Trained_Potential_.mtp");
    BasisAnalyzer analyzer(pot);
    ReportWorkspace report;
    analyzer.run({dimer()}, std::make_unique<ThresholdPruning>(), false);
    analyzer.report().write_all(report.prefix());
    REQUIRE_FALSE(report.text("_selection.dat").empty());
    REQUIRE(analyzer.last_result()->nsamples == 2);

    const bool empty = GENERATE(false, true);
    auto result = analyzer.analyze(empty ? std::vector<Atoms>{} : std::vector<Atoms>{dimer(), dimer(3)});
    REQUIRE(result->nsamples == (empty ? 0 : 4));
    REQUIRE(analyzer.last_result()->nsamples == result->nsamples);
    analyzer.report().write_all(report.prefix());
    REQUIRE(report.text("_selection.dat").empty());
    REQUIRE(report.text("_summary.txt").find("## Strategy:") == std::string::npos);
    REQUIRE(report.text("_summary.txt").find("Sampled atoms:             " + std::to_string(result->nsamples)) != std::string::npos);
}

TEST_CASE("Basis selection rejects incompatible analysis results", "[basis_analysis]") {
    Potentials::PotMTP pot("tests/test-nnmtp/_Trained_Potential_.mtp");
    BasisAnalyzer analyzer(pot);
    auto result = analyzer.analyze({});
    REQUIRE_THROWS_AS(analyzer.apply_strategy(*result, nullptr), std::invalid_argument);
    result->pearson.resize(0, 0);
    REQUIRE_THROWS_AS(analyzer.apply_strategy(*result, std::make_unique<ThresholdPruning>()), std::invalid_argument);
}

TEST_CASE("Diagnostic reports propagate open and write failures for every output", "[basis_analysis][io]") {
    ReportWorkspace work;
    Eigen::MatrixXd data(2, 4);
    data << 1, 2, 3, 4, 1, 4, 2, 3;
    DiagnosticReport report;
    report.set_analysis(CorrelationAnalyzer(data).compute_all());
    report.add_strategy_result("test", {0}, {1});
    const std::string suffix = GENERATE("_summary.txt", "_matrix.dat", "_vif.dat",
                                       "_eigenvalues.dat", "_pairs.dat", "_selection.dat");
    const bool full = GENERATE(false, true);
    const auto path = work.prefix() + suffix;
    if (full) std::filesystem::create_symlink("/dev/full", path);
    else std::filesystem::create_directory(path);
    REQUIRE_THROWS_WITH(report.write_all(work.prefix()), Catch::Matchers::ContainsSubstring(path));
}
