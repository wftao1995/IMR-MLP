#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "tools/chem.h"
#include <iterator>
#include <sstream>

TEST_CASE("Default GPIO mapping includes every element and the dummy", "[gpio][mapping]") {
    GPIO io;
    const auto forward = io.getmap();
    const auto reverse = io.getrevmap();
    REQUIRE(forward.size() == std::size(Chemicals::chemical_symbols));
    REQUIRE(reverse.size() == forward.size());
    for (int number = 0; number < static_cast<int>(std::size(Chemicals::chemical_symbols)); ++number) {
        CAPTURE(number);
        REQUIRE(forward.at(number) == Chemicals::chemical_symbols[number]);
        REQUIRE(reverse.at(Chemicals::chemical_symbols[number]) == number);
    }
    REQUIRE(forward.at(0) == "X");
    REQUIRE(forward.at(1) == "H");
    REQUIRE(forward.at(117) == "Ts");
    REQUIRE(forward.at(118) == "Og");
    io.setmap();
    REQUIRE(io.getmap() == forward);
    REQUIRE(io.getrevmap() == reverse);

    io.setmap({"Og", "H"});
    REQUIRE(io.getmap() == std::map<int, std::string>{{0, "Og"}, {1, "H"}});
    REQUIRE(io.getrevmap() == std::map<std::string, int>{{"Og", 0}, {"H", 1}});
    io.setmap({});
    REQUIRE(io.getmap() == forward);
    REQUIRE(io.getrevmap() == reverse);
}

TEST_CASE("Default upper element survives geometry format round trips", "[gpio][mapping]") {
    const auto format = GENERATE("xsf", "cp2kxyz", "extxyz");
    GPIO io;
    io.lossless = true;
    Atoms original;
    original.resize(2);
    original.chemmap_ = {117, 118};
    original.cell_ = 10 * Eigen::Matrix3d::Identity();
    original.pos_ = {{1, 2, 3}, {4, 5, 6}};
    for (const bool symbols : {false, true}) {
        CAPTURE(format, symbols);
        io.write_symbol = symbols;
        std::ostringstream encoded;
        REQUIRE(io.write_one(original, encoded, format));
        Atoms decoded;
        std::istringstream input(encoded.str());
        REQUIRE(io.read_one(decoded, input, format));
        REQUIRE(decoded.chemmap_ == original.chemmap_);
        REQUIRE(decoded.pos_ == original.pos_);
    }
}
