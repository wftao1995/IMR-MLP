// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <sstream>

using Catch::Approx;

TEST_CASE("GPIO extxyz round-trips Hessian global tag", "[gpio][hessian]")
{
    Atoms input;
    input.cell_ = Eigen::Matrix3d::Identity() * 8.0;
    input.pos_ = {{0.0, 0.0, 0.0}, {1.0, 0.2, 0.3}};
    input.chemmap_ = {0, 0};
    input.hessian_.resize(6, 6);
    for (int i = 0; i < input.hessian_.rows(); ++i)
        for (int j = 0; j < input.hessian_.cols(); ++j)
            input.hessian_(i, j) = 0.1 * i + 0.01 * j;
    input.have_hessian = true;

    std::stringstream buffer;
    GPIO writer;
    REQUIRE(writer.write_one(input, buffer, "extxyz"));

    Atoms output;
    GPIO reader;
    REQUIRE(reader.read_one(output, buffer, "extxyz"));

    REQUIRE(output.has_hessian());
    REQUIRE(output.hessian_.rows() == 6);
    REQUIRE(output.hessian_.cols() == 6);
    for (int i = 0; i < output.hessian_.rows(); ++i)
        for (int j = 0; j < output.hessian_.cols(); ++j)
            REQUIRE(output.hessian_(i, j) == Approx(input.hessian_(i, j)));
}

TEST_CASE("GPIO extxyz rejects Hessian tag with wrong length", "[gpio][hessian]")
{
    std::stringstream input;
    input << "1\n";
    input << "Lattice=\"5 0 0 0 5 0 0 0 5\" hessian=\"1 2 3\" Properties=species:S:1:pos:R:3\n";
    input << "H 0 0 0\n";

    Atoms atom;
    GPIO reader;
    REQUIRE_THROWS(reader.read_one(atom, input, "extxyz"));
}

TEST_CASE("GPIO extxyz preserves three-axis PBC flags", "[gpio][pbc]")
{
    Atoms input;
    input.cell_=5.0*Eigen::Matrix3d::Identity();
    input.pos_={{0.0,0.0,0.0}};
    input.chemmap_={0};
    input.pbc_={{true,false,true}};
    std::stringstream buffer;
    GPIO writer;
    REQUIRE(writer.write_one(input,buffer,"extxyz"));
    REQUIRE(buffer.str().find("pbc=\"T F T\"")!=std::string::npos);
    Atoms output;
    GPIO reader;
    REQUIRE(reader.read_one(output,buffer,"extxyz"));
    REQUIRE(output.pbc_==input.pbc_);
}
