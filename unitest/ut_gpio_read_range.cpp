// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "io/gpio.h"

#include <climits>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef IMRMLP_NETCDF
#include <netcdf>
#endif

namespace {
struct RangeWorkspace {
    std::filesystem::path path;
    RangeWorkspace() {
        char name[] = "/tmp/imrmlp_gpio_range_XXXXXX";
        const char *created = mkdtemp(name);
        if (!created) throw std::runtime_error("mkdtemp failed");
        path = created;
    }
    ~RangeWorkspace() { std::filesystem::remove_all(path); }
};

struct Selection {
    int start, end, stride;
    std::vector<int> expected;
};

const std::vector<Selection> valid_ranges = {
    {0, 0, 1, {0}},
    {1, 1, 1, {1}},
    {3, 3, 1, {3}},
    {1, 3, 2, {1, 3}},
    {0, 2, 2, {0, 2}},
    {0, 3, 2, {0, 2}}, // Inclusive end need not lie on the stride.
    {0, INT_MAX, INT_MAX, {0}},
    {1, INT_MAX, INT_MAX, {1}},
    {0, INT_MAX - 1, 1, {0, 1, 2, 3}},
    {4, INT_MAX, 1, {}},
    {INT_MAX - 1, INT_MAX, 1, {}},
    {INT_MAX, INT_MAX, INT_MAX, {}},
};

const std::vector<Selection> invalid_ranges = {
    {-1, INT_MAX, 1, {}},
    {INT_MIN, INT_MAX, 1, {}},
    {0, -1, 1, {}},
    {0, INT_MIN, 1, {}},
    {2, 1, 1, {}},
    {INT_MAX, INT_MAX - 1, 1, {}},
    {0, INT_MAX, 0, {}},
    {0, INT_MAX, -1, {}},
    {0, INT_MAX, INT_MIN, {}},
};

void set_selection(GPIO &io, const Selection &selection) {
    io.read_startfrom = selection.start;
    io.read_endto = selection.end;
    io.read_irate = selection.stride;
}

Eigen::RowVector3d position(int frame, int atom) {
    return {10.0 * frame + atom + .125, .25, .375};
}

Eigen::Vector3d force(int frame, int atom) {
    return {.5 * (frame + 1), -.25 * (atom + 1), .125 * (frame + atom + 1)};
}

Atoms sentinel() {
    Atoms atom;
    atom.id = 123;
    atom.pos_ = {{-7, -8, -9}};
    atom.chemmap_ = {0};
    atom.forces_ = {{3, 4, 5}};
    return atom;
}

void check_unchanged(const std::vector<Atoms> &frames) {
    REQUIRE(frames.size() == 1);
    const Atoms expected = sentinel();
    CHECK(frames[0].id == expected.id);
    CHECK(frames[0].chemmap_ == expected.chemmap_);
    REQUIRE(frames[0].pos_.size() == 1);
    CHECK((frames[0].pos_[0] - expected.pos_[0]).norm() == 0);
    REQUIRE(frames[0].has_forces());
    REQUIRE(frames[0].forces_.size() == 1);
    CHECK((frames[0].forces_[0] - expected.forces_[0]).norm() == 0);
}

void check_selected(const std::vector<Atoms> &frames,
                    const std::vector<int> &expected, bool with_forces) {
    REQUIRE(frames.size() == expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
        const int frame = expected[i];
        CAPTURE(frame, with_forces);
        const auto &atom = frames[i];
        CHECK(atom.id == frame);
        REQUIRE(atom.size() == 2);
        REQUIRE(atom.has_energy());
        CHECK(atom.energy_ == -10.0 + frame);
        REQUIRE(atom.has_forces() == with_forces);
        if (with_forces) REQUIRE(atom.forces_.size() == 2);
        else CHECK(atom.forces_.empty());
        for (int j = 0; j < 2; ++j) {
            CHECK((atom.pos_[j] - position(frame, j)).norm() == 0);
            if (with_forces) CHECK((atom.forces_[j] - force(frame, j)).norm() == 0);
        }
    }
}

// Text is authored independently of GPIO's writers, so selection/label checks
// cannot pass merely because the reader and writer share a mistake.
std::string extxyz_fixture(bool with_forces) {
    std::ostringstream text;
    for (int frame = 0; frame < 4; ++frame) {
        text << "2\nLattice=\"6 0 0 0 7 0 0 0 8\" Properties=species:S:1:pos:R:3";
        if (with_forces) text << ":forces:R:3";
        text << " energy=" << -10 + frame << " pbc=\"T T T\"\n";
        for (int atom = 0; atom < 2; ++atom) {
            text << (atom == 0 ? "H" : "He") << ' ' << position(frame, atom);
            if (with_forces) text << ' ' << force(frame, atom).transpose();
            text << '\n';
        }
    }
    return text.str();
}

#ifdef IMRMLP_NETCDF
// Independent NetCDF API fixture; no GPIO writer or file slicing is used.
void netcdf_fixture(const std::string &path, int count, bool with_forces) {
    using namespace netCDF;
    NcFile file(path, NcFile::newFile, NcFile::classic64);
    file.putAtt("Conventions", "AMBER");
    const auto frame = file.addDim("frame", count);
    const auto atom = file.addDim("atom", 2);
    const auto spatial = file.addDim("spatial", 3);
    auto coordinates = file.addVar("coordinates", ncDouble, {frame, atom, spatial});
    coordinates.putAtt("units", "angstrom");
    auto energy = file.addVar("energy", ncDouble, {frame});
    energy.putAtt("units", "eV");
    auto type = file.addVar("type", ncInt, {frame, atom});
    std::vector<double> positions(6 * count), forces(6 * count), energies(count);
    std::vector<int> types(2 * count);
    for (int f = 0; f < count; ++f) {
        energies[f] = -10.0 + f;
        for (int a = 0; a < 2; ++a) {
            types[2 * f + a] = a + 1;
            for (int k = 0; k < 3; ++k) {
                positions[6 * f + 3 * a + k] = position(f, a)[k];
                forces[6 * f + 3 * a + k] = force(f, a)[k];
            }
        }
    }
    coordinates.putVar(positions.data());
    energy.putVar(energies.data());
    type.putVar(types.data());
    if (with_forces) {
        auto variable = file.addVar("forces", ncDouble, {frame, atom, spatial});
        variable.putAtt("units", "eV/angstrom");
        variable.putVar(forces.data());
    }
}
#endif
}

TEST_CASE("GPIO stream frame selection uses inclusive bounds and preserves labels",
          "[gpio][read-range][regression]") {
    const bool with_forces = GENERATE(false, true);
    for (const auto &selection : valid_ranges) {
        CAPTURE(selection.start, selection.end, selection.stride);
        GPIO io;
        set_selection(io, selection);
        std::istringstream input(extxyz_fixture(with_forces));
        std::vector<Atoms> output;
        REQUIRE(io.read(output, input, "extxyz"));
        check_selected(output, selection.expected, with_forces);
    }
}

TEST_CASE("GPIO rejects invalid frame ranges before consuming stream or changing output",
          "[gpio][read-range][regression]") {
    const bool format_enum = GENERATE(false, true);
    for (const auto &selection : invalid_ranges) {
        CAPTURE(selection.start, selection.end, selection.stride, format_enum);
        GPIO io;
        set_selection(io, selection);
        std::istringstream input(extxyz_fixture(true));
        const auto initial_position = input.tellg();
        std::vector<Atoms> output = {sentinel()};
        const bool status = format_enum ? io.read(output, input, GPIO::EXTXYZ)
                                        : io.read(output, input, "extxyz");
        CHECK_FALSE(status);
        CHECK(input.tellg() == initial_position);
        check_unchanged(output);
        // Rejecting an invalid range must not poison a later valid read.
        set_selection(io, {0, INT_MAX, 1, {}});
        output.clear();
        REQUIRE(io.read(output, input, "extxyz"));
        check_selected(output, {0, 1, 2, 3}, true);
    }
}

TEST_CASE("GPIO file read rejects invalid ranges without appending frames",
          "[gpio][read-range][regression]") {
    RangeWorkspace work;
    const auto path = (work.path / "frames.extxyz").string();
    { std::ofstream output(path); output << extxyz_fixture(true); }
    for (const auto &selection : invalid_ranges) {
        CAPTURE(selection.start, selection.end, selection.stride);
        GPIO io;
        set_selection(io, selection);
        std::vector<Atoms> output = {sentinel()};
        CHECK_FALSE(io.read(output, path, "extxyz"));
        check_unchanged(output);
    }
}

#ifdef IMRMLP_NETCDF
TEST_CASE("NetCDF default and explicit INT_MAX upper bounds preserve two frame force labels",
          "[gpio][netcdf][read-range][a14][regression]") {
    RangeWorkspace work;
    const bool with_forces = GENERATE(false, true);
    const bool explicit_upper = GENERATE(false, true);
    const auto path = (work.path / "two.nc").string();
    netcdf_fixture(path, 2, with_forces);
    GPIO io;
    CHECK(io.read_startfrom == 0);
    CHECK(io.read_irate == 1);
    CHECK(io.read_endto == INT_MAX);
    if (explicit_upper) io.read_endto = INT_MAX;
    std::vector<Atoms> output;
    REQUIRE(io.read(output, path, "netcdf"));
    check_selected(output, {0, 1}, with_forces);
}

TEST_CASE("NetCDF frame selection handles inclusive and int sentinel boundaries",
          "[gpio][netcdf][read-range][a14][regression]") {
    RangeWorkspace work;
    const bool with_forces = GENERATE(false, true);
    const auto path = (work.path / "four.nc").string();
    netcdf_fixture(path, 4, with_forces);
    for (const auto &selection : valid_ranges) {
        CAPTURE(selection.start, selection.end, selection.stride);
        GPIO io;
        set_selection(io, selection);
        std::vector<Atoms> output;
        REQUIRE(io.read(output, path, "netcdf"));
        check_selected(output, selection.expected, with_forces);
    }
}

TEST_CASE("NetCDF invalid ranges leave output and continuation state unchanged",
          "[gpio][netcdf][read-range][a14][regression]") {
    RangeWorkspace work;
    const auto path = (work.path / "two.nc").string();
    netcdf_fixture(path, 2, true);
    for (const auto &selection : invalid_ranges) {
        CAPTURE(selection.start, selection.end, selection.stride);
        GPIO io;
        io.reset_status_every_read = false;
        std::vector<Atoms> first;
        REQUIRE(io.read(first, path, "netcdf"));
        check_selected(first, {0, 1}, true);
        set_selection(io, selection);
        std::vector<Atoms> output = {sentinel()};
        CHECK_FALSE(io.read(output, path, "netcdf"));
        check_unchanged(output);
        set_selection(io, {0, INT_MAX, 1, {}});
        output.clear();
        REQUIRE(io.read(output, path, "netcdf"));
        REQUIRE(output.size() == 2);
        CHECK(output[0].id == 2);
        CHECK(output[1].id == 3);
        REQUIRE(output[0].forces_.size() == 2);
        REQUIRE(output[1].forces_.size() == 2);
        CHECK((output[0].forces_[0] - force(0, 0)).norm() == 0);
        CHECK((output[1].forces_[0] - force(1, 0)).norm() == 0);
    }
}
#endif
