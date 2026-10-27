#include <catch2/catch_all.hpp>
#include "apps/convstr_scalar_expression.h"

namespace {
Atoms scalar_frame() {
    Atoms frame;
    frame.id = 7;
    frame.cell_ = 2 * Eigen::Matrix3d::Identity();
    frame.pos_ = {Eigen::RowVector3d::Zero(), Eigen::RowVector3d::Ones()};
    frame.chemmap_ = {0, 0};
    frame.energy_ = 12; frame.have_energy = true;
    frame.fenergy_ = 10; frame.have_fenergy = true;
    frame.fitness = .5;
    frame.global_attribs_d_["offset"] = 1.25;
    frame.global_attribs_i_["count"] = 3;
    frame.features_["text"] = "42";
    return frame;
}
}

TEST_CASE("convstr scalar expressions respect arithmetic precedence and functions", "[convstr][scalar-expression]") {
    const Atoms frame = scalar_frame();
    for (const auto &[source, expected] : std::vector<std::pair<std::string, double>>{
             {"answer=2+3*4", 14}, {"answer=(2+3)*4", 20},
             {"answer=-2^2", -4}, {"answer=2^-2", .25},
             {"answer=2^3^2", 512}, {"answer= .5e+2 + 1E-2", 50.01},
             {"answer=sqrt(9)+exp(0)+log(exp(1))+sin(0)+cos(0)", 6},
             {"answer=-(energy-fenergy)/size+offset+count+fitness+id+volume", 18.75},
             {"answer=energy^(1+1)", 144}}) {
        INFO(source);
        REQUIRE(ConvstrLabels::Expression(source).evaluate(frame) == Catch::Approx(expected));
    }
}

TEST_CASE("convstr scalar expression grammar rejects unsafe or ambiguous assignments", "[convstr][scalar-expression]") {
    for (const auto &source : {"energy=1", "id=2", "total_charge=2", "forces=0", "hessian=0",
                              "=1", "1bad=1", "answer=", "answer=1 2", "answer=tan(0)",
                              "answer=sin(1,2)", "answer=2^size", "answer=1e999", "answer=1;2"}) {
        INFO(source);
        REQUIRE_THROWS_AS(ConvstrLabels::Expression{source}, std::invalid_argument);
    }
    REQUIRE_THROWS_AS(ConvstrLabels::compile({"a=1", "a=2"}), std::invalid_argument);
    REQUIRE_THROWS_AS(ConvstrLabels::compile({"a=1"}, {"a"}), std::invalid_argument);
}

TEST_CASE("convstr scalar failures name frame and expression without changing metadata", "[convstr][scalar-expression]") {
    auto frame = scalar_frame();
    for (const auto &source : {"answer=missing", "answer=text", "answer=1/0", "answer=sqrt(-1)",
                              "answer=log(0)", "answer=exp(1000)", "answer=(-1)^.5"}) {
        INFO(source);
        try {
            ConvstrLabels::Expression{source}.evaluate(frame);
            FAIL("Expected scalar evaluation error");
        } catch (const std::invalid_argument &error) {
            REQUIRE(std::string(error.what()).find("frame 7") != std::string::npos);
            REQUIRE(std::string(error.what()).find(source) != std::string::npos);
        }
    }
    frame.have_energy = false;
    frame.global_attribs_d_["energy"] = 42;
    REQUIRE_THROWS_AS(ConvstrLabels::Expression{"answer=energy"}.evaluate(frame), std::invalid_argument);
    frame.global_attribs_d_["offset"] = std::numeric_limits<double>::infinity();
    REQUIRE_THROWS_AS(ConvstrLabels::Expression{"answer=offset"}.evaluate(frame), std::invalid_argument);
    REQUIRE_THROWS_AS(ConvstrLabels::assign(frame, ConvstrLabels::compile({"count=2", "answer=missing"})),
                      std::invalid_argument);
    REQUIRE(frame.global_attribs_i_.at("count") == 3);
    REQUIRE_FALSE(frame.global_attribs_d_.count("count"));
}

TEST_CASE("convstr metadata overwrites read one original snapshot and keep numeric types", "[convstr][scalar-expression]") {
    auto frame = scalar_frame();
    frame.features_["count"] = "stale";
    const auto expressions = ConvstrLabels::compile({"count=count+1", "offset=count+offset", "text=energy/size"});
    ConvstrLabels::assign(frame, expressions);
    REQUIRE(frame.global_attribs_d_.at("count") == 4);
    REQUIRE(frame.global_attribs_d_.at("offset") == 4.25);
    REQUIRE(frame.global_attribs_d_.at("text") == 6);
    REQUIRE_FALSE(frame.global_attribs_i_.count("count"));
    REQUIRE_FALSE(frame.features_.count("count"));
    REQUIRE_FALSE(frame.features_.count("text"));
    // New outputs cannot be consumed as if a dependency graph had been evaluated.
    REQUIRE_THROWS_AS(ConvstrLabels::assign(frame, ConvstrLabels::compile({"a=1", "b=a+1"})),
                      std::invalid_argument);
    REQUIRE_FALSE(frame.global_attribs_d_.count("a"));
}

TEST_CASE("convstr CFG metadata bridge matches the scalar double and rejects array targets", "[convstr][scalar-expression]") {
    auto frame = scalar_frame();
    ConvstrLabels::assign(frame, ConvstrLabels::compile({"offset=1/3", "count=count+1"}), true);
    REQUIRE(std::stod(frame.features_.at("offset")) == frame.global_attribs_d_.at("offset"));
    REQUIRE(frame.features_.at("count") == "4");
    frame.global_attribs_v_["vector"] = Eigen::VectorXd::Ones(2);
    REQUIRE_THROWS_AS(ConvstrLabels::assign(frame, ConvstrLabels::compile({"vector=1"})),
                      std::invalid_argument);
    REQUIRE_FALSE(frame.global_attribs_d_.count("vector"));
}
