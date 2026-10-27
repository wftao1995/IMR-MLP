// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "tools/hessian_io.h"

#include <fstream>

using Catch::Approx;

TEST_CASE("HessianIO reads dense phonopy FORCE_CONSTANTS", "[hessian_io]")
{
    const std::string path = "/tmp/imrmlp_force_constants_dense_test";
    {
        std::ofstream out(path);
        out << "2\n";
        out << "1 1\n";
        out << "1 2 3\n";
        out << "4 5 6\n";
        out << "7 8 9\n";
        out << "1 2\n";
        out << "10 11 12\n";
        out << "13 14 15\n";
        out << "16 17 18\n";
        out << "2 1\n";
        out << "19 20 21\n";
        out << "22 23 24\n";
        out << "25 26 27\n";
        out << "2 2\n";
        out << "28 29 30\n";
        out << "31 32 33\n";
        out << "34 35 36\n";
    }

    auto hessian = HessianIO::read_force_constants(path, 2);
    REQUIRE(hessian.rows() == 6);
    REQUIRE(hessian.cols() == 6);
    REQUIRE(hessian(0, 0) == Approx(1.0));
    REQUIRE(hessian(2, 5) == Approx(18.0));
    REQUIRE(hessian(3, 0) == Approx(19.0));
    REQUIRE(hessian(5, 5) == Approx(36.0));
}

TEST_CASE("HessianIO reads vasprun hessian varray", "[hessian_io]")
{
    const std::string path = "/tmp/imrmlp_vasprun_hessian_test.xml";
    {
        std::ofstream out(path);
        out << "<modeling>\n";
        out << "<varray name=\"hessian\">\n";
        for (int i = 0; i < 6; ++i) {
            out << "<v>";
            for (int j = 0; j < 6; ++j)
                out << ' ' << (10 * i + j);
            out << " </v>\n";
        }
        out << "</varray>\n";
        out << "</modeling>\n";
    }

    auto hessian = HessianIO::read_vasprun_hessian(path, 2);
    REQUIRE(hessian.rows() == 6);
    REQUIRE(hessian.cols() == 6);
    REQUIRE(hessian(0, 0) == Approx(0.0));
    REQUIRE(hessian(2, 4) == Approx(24.0));
    REQUIRE(hessian(5, 5) == Approx(55.0));
}

TEST_CASE("HessianIO rejects duplicate dense FORCE_CONSTANTS blocks", "[hessian_io]")
{
    const std::string path = "/tmp/imrmlp_force_constants_duplicate_test";
    {
        std::ofstream out(path);
        out << "2\n";
        for (int block = 0; block < 4; ++block) {
            out << "1 1\n";
            out << "1 0 0\n0 1 0\n0 0 1\n";
        }
    }
    REQUIRE_THROWS_WITH(HessianIO::read_force_constants(path, 2),
                        Catch::Matchers::ContainsSubstring("Duplicate"));
}
