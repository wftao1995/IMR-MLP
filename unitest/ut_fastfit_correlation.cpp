// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "basis_analysis/CorrelationAnalyzer.h"
#include "basis_analysis/strategies/ThresholdPruning.h"
#include "basis_analysis/strategies/IterativeGreedyVIF.h"
#include <random>

using namespace BasisAnalysis;
using namespace Eigen;

TEST_CASE("VIF pruning handles exact multivariate dependence", "[fastfit][correlation]") {
    MatrixXd d(3,4); d << 1,-1,1,-1, 1,1,-1,-1, 2,0,0,-2;
    const auto c=CorrelationAnalyzer(d).compute_all();
    REQUIRE(c.rank()==2);
    for (int i=0;i<3;++i) REQUIRE(std::isinf(c.vif[i]));
    REQUIRE(ThresholdPruning(.95,10).select(c,3).first.size()==2);
    REQUIRE(IterativeGreedyVIF(10).select(c,3).first.size()==2);
    MatrixXd singular=MatrixXd::Constant(121,121,-1./120.);
    singular.diagonal().setOnes();
    auto vif=CorrelationAnalyzer::compute_vif(singular);
    for (double value:vif) REQUIRE(std::isinf(value));
}

TEST_CASE("VIF feature limit and variance checks are enforced", "[fastfit][correlation]") {
    CorrelationResult c; c.pearson=MatrixXd::Identity(5,5);
    REQUIRE(IterativeGreedyVIF(10,2).select(c,5).first.size()==2);
    c.pearson=MatrixXd::Constant(3,3,.999); c.pearson.diagonal().setOnes();
    REQUIRE(IterativeGreedyVIF(2,2).select(c,3).first.size()==1);
    MatrixXd constant=MatrixXd::Constant(3,5,7.);
    c=CorrelationAnalyzer(constant).compute_all();
    REQUIRE(c.rank()==0);
    REQUIRE(c.pearson.isZero());
    REQUIRE(IterativeGreedyVIF(10).select(c,3).first.empty());
    REQUIRE(ThresholdPruning(.95,10).select(c,3).first.empty());
    MatrixXd single=MatrixXd::Ones(2,1);
    REQUIRE(CorrelationAnalyzer(single).compute_all().rank()==0);
    MatrixXd empty(0,0);
    REQUIRE(CorrelationAnalyzer(empty).compute_all().vif.size()==0);
}

TEST_CASE("VIF inverse updates agree with repeated spectral selection", "[fastfit][correlation]") {
    std::mt19937 rng(1729);
    std::normal_distribution<double> normal;
    MatrixXd data(55,100);
    for (auto &value:data.reshaped()) value=normal(rng);
    const auto c=CorrelationAnalyzer(data).compute_all();
    std::set<int> expected;
    for (int i=0;i<55;++i) expected.insert(i);
    while (expected.size()>7) {
        std::vector<int> ids(expected.begin(),expected.end());
        MatrixXd sub(ids.size(),ids.size());
        for (int i=0;i<ids.size();++i) for (int j=0;j<ids.size();++j)
            sub(i,j)=c.pearson(ids[i],ids[j]);
        Index worst;
        CorrelationAnalyzer::compute_vif(sub).maxCoeff(&worst);
        expected.erase(ids[worst]);
    }
    REQUIRE(IterativeGreedyVIF(100,7).select(c,55).first==expected);
}
