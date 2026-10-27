// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>

#include "io/gpio.h"

#include <sstream>

using Catch::Approx;

namespace {

// VASP writes these section headers with a fixed layout; the reader matches
// them by exact prefix, so the fixtures reproduce the real byte sequences.
constexpr std::string_view kIterHeader = "--------------------------------------- Iteration";
constexpr std::string_view kAbortLine = "------------------------ aborting loop because EDIFF is reached";
constexpr std::string_view kTotenLine = "  free  energy   TOTEN  =";
constexpr std::string_view kPositionHeader = " POSITION                                       TOTAL-FORCE (eV/Angst)";

// One converged ionic step: electronic iterations, the EDIFF abort line, the
// VOLUME/BASIS block, the POSITION block and the TOTEN energy line.
// With `abort = false` the EDIFF line is omitted and `tail` is emitted in its
// place, which lets tests exercise the NELM-limit fallback paths.
static void append_step(std::ostringstream &os, int niterations, int natoms,
                        bool abort = true, std::string_view tail = "")
{
    for (int i = 1; i <= niterations; ++i) {
        os << kIterHeader << "      " << i << "(   1)  ---------------------------------------\n";
        os << " DAV:  -0.1234567E+03  -0.1234567E+03  -0.1234567E+03  -0.1234567E+03\n";
    }
    if (abort) {
        os << kAbortLine << " ----------------------------------------\n";
    } else if (!tail.empty()) {
        os << tail << "\n";
    }
    os << " VOLUME and BASIS-vectors are now :\n"
       << " -----------------------------------------------------------------------------\n"
       << "  energy-cutoff  :   400.00000\n"
       << "  volume of cell :     104.0361\n"
       << "       direct lattice vectors        reciprocal lattice vectors\n"
       << "      0.86602540  0.50000000  0.00000000      1.15470054  0.00000000  0.00000000\n"
       << "      0.00000000  1.00000000  0.00000000      0.00000000  1.00000000  0.00000000\n"
       << "      0.00000000  0.00000000  1.00000000      0.00000000  0.00000000  1.00000000\n";
    os << kPositionHeader << "\n"
       << " -----------------------------------------------------------------------------------\n";
    for (int i = 0; i < natoms; ++i) {
        os << "       " << (i + 1) << ".00000000  " << (i + 2) << ".00000000  " << (i + 3)
           << ".00000000       0.00000000       0.00000000       0.00000000\n";
    }
    os << kTotenLine << "      " << (-239.0 - niterations) << " eV\n";
}

// Minimal VASP OUTCAR: header with species info + INCAR attributes + steps.
static std::string make_outcar(std::string version_line, int nelm, int nsteps,
                               int niterations, bool abort = true,
                               std::string_view last_tail = "")
{
    std::ostringstream os;
    os << version_line << "\n"
       << "  INCAR:    SYSTEM = test\n"
       << "  POTCAR:    PAW_PBE Fe_sv 06Sep2000\n"
       << "  POTCAR:    PAW_PBE O 06Sep2000\n"
       << "   VRHFIN =Fe: s2p6d6\n"
       << "  ions per type =              2   3\n"
       << "  LCOMPAT=      F    compatible to vasp.4.4\n"
       << "  NELM   =    " << nelm << ";   NELMIN=  2; NELMDL= -5     # of ELM steps\n"
       << "  EDIFF  = 0.1E-05   stopping-criterion for ELM\n"
       << "  EDIFFG = -.1E+00   stopping-criterion for IOM\n"
       << "  NSW    =    100    number of steps for IOM\n"
       << "  IBRION =      3    ionic relax: 0-MD 1-quasi-New 2-CG\n"
       << "  ISIF   =      2    stress and relaxation\n";
    for (int s = 0; s < nsteps; ++s) {
        const bool last = (s == nsteps - 1);
        append_step(os, niterations, /*natoms=*/5, abort, last ? last_tail : "");
    }
    return os.str();
}

static std::string make_mlff_outcar()
{
    auto header = make_outcar(" vasp.6.4.3", 60, 0, 0);
    header.insert(header.find("  POTCAR:"), "  MLFF enabled\n");
    std::ostringstream os;
    os << header;
    for (int step = 0; step < 2; ++step) {
        const int side = step + 2;
        os << " VOLUME and BASIS-vectors are now :\n"
           << " -----------------------------------------------------------------------------\n"
           << "  energy-cutoff  :   400.00000\n"
           << "  volume of cell : " << side * side * side << "\n"
           << "       direct lattice vectors        reciprocal lattice vectors\n"
           << "      " << side << " 0 0 0 0 0\n"
           << "      0 " << side << " 0 0 0 0\n"
           << "      0 0 " << side << " 0 0 0\n"
           << kPositionHeader << "\n"
           << " -----------------------------------------------------------------------------------\n";
        for (int i = 0; i < 5; ++i)
            os << "  " << 10 * step + i + 1 << " 2 3  .1 .2 .3\n";
        os << kTotenLine << " " << -10 - step << " eV\n";
    }
    os << " General timing and accounting informations for this job:\n";
    return os.str();
}

static void check_fe2o3(const Atoms &atom)
{
    // Fe_sv -> Fe (26), O -> 8, ions per type = 2 3 -> Fe Fe O O O
    REQUIRE(atom.chemmap_ == std::vector<int>{26, 26, 8, 8, 8});
    REQUIRE(atom.pos_.size() == 5);
    REQUIRE(atom.cell_(0, 0) == Approx(0.86602540));
    REQUIRE(atom.cell_(1, 1) == Approx(1.0));
    REQUIRE(atom.pos_[0](0) == Approx(1.0));
    REQUIRE(atom.pos_[4](2) == Approx(7.0));
}

} // namespace

TEST_CASE("GPIO reads multi-element OUTCAR with all ion counts", "[gpio][outcar]")
{
    // vasp 5 header, 5 electronic iterations (< NELM), converged step
    std::istringstream input(make_outcar(" vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Dec 14 2021 08:20:39) complex", 60, 1, 5));
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "outcar"));
    check_fe2o3(atom);
    REQUIRE(atom.energy_ == Approx(-239.0 - 5));
    REQUIRE(atom.have_energy);
}

TEST_CASE("GPIO reads vasp 6 OUTCAR header", "[gpio][outcar]")
{
    std::istringstream input(make_outcar(" vasp.6.3.2 27Jun22 (build Jan 06 2023 19:30:47) complex", 60, 1, 4));
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "outcar"));
    check_fe2o3(atom);
}

TEST_CASE("GPIO reads the six OUTCAR stress components as a symmetric tensor", "[gpio][outcar]")
{
    auto text = make_outcar(" vasp.5.4.4", 60, 1, 4);
    const auto cell_begin = text.find(" VOLUME and BASIS-vectors");
    const auto cell_end = text.find(kPositionHeader, cell_begin);
    REQUIRE(cell_begin != std::string::npos);
    REQUIRE(cell_end != std::string::npos);
    text.replace(cell_begin, cell_end - cell_begin,
        "  FORCE on cell =-STRESS in cart. coord. units (eV):\n"
        "  Direction    XX YY ZZ XY YZ ZX\n"
        "  in kB        1 2 3 4 5 6\n"
        " VOLUME and BASIS-vectors are now :\n"
        " -----------------------------------------------------------------------------\n"
        "  energy-cutoff  :   400.00000\n"
        "  volume of cell :     24.0\n"
        "       direct lattice vectors        reciprocal lattice vectors\n"
        "      2 0 0  0.5 0 0\n"
        "      0 3 0  0 0.333333333333 0\n"
        "      0 0 4  0 0 0.25\n");

    // VASP's XX YY ZZ XY YZ ZX order, times 24 A^3 and the
    // independent kbar-to-eV/A^3 factor, gives this full virial tensor.
    const double factor = 24.0 * 0.000624150912588;
    const double expected[3][3] = {{1, 4, 6}, {4, 2, 5}, {6, 5, 3}};
    auto check_stress = [&](const Atoms &atom) {
        REQUIRE(atom.have_stress);
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                REQUIRE(atom.stress_(i, j) == Approx(expected[i][j] * factor));
        REQUIRE(atom.get_volume() == Approx(24.0));
        REQUIRE(atom.size() == 5);
        REQUIRE(atom.energy_ == Approx(-243.0));
    };

    SECTION("read_one") {
        std::istringstream input(text);
        GPIO io;
        Atoms atom;
        REQUIRE(io.read_one(atom, input, "outcar"));
        check_stress(atom);
    }
    SECTION("bulk read") {
        std::istringstream input(text);
        GPIO io;
        std::vector<Atoms> atoms;
        REQUIRE(io.read(atoms, input, "outcar"));
        REQUIRE(atoms.size() == 1);
        check_stress(atoms[0]);
    }
}

TEST_CASE("GPIO accepts OUTCAR steps converging exactly on the NELM-th iteration", "[gpio][outcar]")
{
    // step 2 takes exactly NELM=10 electronic iterations; VASP still prints
    // the EDIFF-abort line right after the last one, so the frame is converged
    std::istringstream input(make_outcar(" vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Dec 14 2021 08:20:39) complex", 10, 2, 10));
    std::vector<Atoms> atoms;
    GPIO io;
    REQUIRE(io.read(atoms, input, "outcar"));
    REQUIRE(atoms.size() == 2);
    check_fe2o3(atoms[1]);
}

TEST_CASE("GPIO accepts OUTCAR NELM-limited steps converged by energy change", "[gpio][outcar]")
{
    // exactly NELM iterations and no EDIFF-abort line, but the last iteration's
    // energy change is below EDIFF -> still converged via the fallback scan
    std::string tail = " total energy-change (2. order) :0.1234567E-07  (0.1234567E-07)";
    std::istringstream input(make_outcar(" vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Dec 14 2021 08:20:39) complex", 10, 1, 10, false, tail));
    Atoms atom;
    GPIO io;
    REQUIRE(io.read_one(atom, input, "outcar"));
    check_fe2o3(atom);
}

TEST_CASE("GPIO rejects OUTCAR steps that hit NELM without converging", "[gpio][outcar]")
{
    // exactly NELM iterations, no EDIFF-abort line and a large energy change
    // (positive, so the energy-change fallback cannot mark it as converged)
    std::string tail = " total energy-change (2. order) :0.1234567E+03  (0.1234567E+03)";
    std::istringstream input(make_outcar(" vasp.5.4.4.18Apr17-6-g9f103f2a35 (build Dec 14 2021 08:20:39) complex", 10, 1, 10, false, tail));
    Atoms atom;
    GPIO io;
    REQUIRE_FALSE(io.read_one(atom, input, "outcar"));
}

TEST_CASE("GPIO rejects OUTCARs with an unsupported version line", "[gpio][outcar][errors]")
{
    SECTION("no vasp version line at all") {
        std::istringstream input("SOMETHING ELSE\nnot a vasp file\n");
        Atoms atom;
        GPIO io;
        REQUIRE_FALSE(io.read_one(atom, input, "outcar"));
    }
    SECTION("version line too short to contain a version digit") {
        std::istringstream input(" vasp\n");
        Atoms atom;
        GPIO io;
        REQUIRE_FALSE(io.read_one(atom, input, "outcar"));
    }
}

TEST_CASE("GPIO advances between MLFF OUTCAR frames and reaches EOF", "[gpio][outcar]")
{
    std::istringstream input(make_mlff_outcar());
    GPIO io;
    // A third-frame limit bounds even a regression that repeatedly accepts
    // the cached TOTEN line. A correct reader reaches EOF after two frames.
    io.read_endto = 2;
    std::vector<Atoms> atoms;
    REQUIRE(io.read(atoms, input, "outcar"));
    REQUIRE(atoms.size() == 2);
    REQUIRE(input.eof());
    for (int step = 0; step < 2; ++step) {
        const auto &atom = atoms[step];
        REQUIRE(atom.chemmap_ == std::vector<int>{26, 26, 8, 8, 8});
        REQUIRE(atom.forces_.size() == 5);
        REQUIRE(atom.have_energy);
        REQUIRE(atom.energy_ == Approx(-10 - step));
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
                REQUIRE(atom.cell_(i, j) == Approx(i == j ? step + 2 : 0));
        for (int i = 0; i < 5; ++i) {
            REQUIRE(atom.pos_[i](0) == Approx(10 * step + i + 1));
            REQUIRE(atom.pos_[i](1) == Approx(2));
            REQUIRE(atom.pos_[i](2) == Approx(3));
            for (int j = 0; j < 3; ++j)
                REQUIRE(atom.forces_[i](j) == Approx(.1 * (j + 1)));
        }
    }
}

TEST_CASE("GPIO resets MLFF mode when reading another OUTCAR header", "[gpio][outcar]")
{
    GPIO io;
    Atoms atom;
    std::istringstream mlff(make_mlff_outcar());
    REQUIRE(io.read_one(atom, mlff, "outcar"));
    REQUIRE(atom.size() == 5);
    REQUIRE(atom.energy_ == Approx(-10));

    std::istringstream ordinary(make_outcar(" vasp.5.4.4", 10, 1, 10, false,
        " total energy-change (2. order) :0.1234567E+03  (0.1234567E+03)"));
    REQUIRE_FALSE(io.read_one(atom, ordinary, "outcar"));
}

TEST_CASE("GPIO streams ordinary and MLFF OUTCAR steps with independent frames", "[gpio][outcar][stream]")
{
    std::string text;
    bool mlff = false;
    bool statistics_tail = true;
    SECTION("VASP 5") { text = make_outcar(" vasp.5.4.4", 60, 2, 4); }
    SECTION("VASP 6") { text = make_outcar(" vasp.6.3.2", 60, 2, 4); }
    SECTION("existing MLFF layout") { text = make_mlff_outcar(); mlff = true; }
    SECTION("last energy without a newline") {
        text = make_outcar(" vasp.5.4.4", 60, 2, 4);
        text.pop_back();
        statistics_tail = false;
    }
    // Also accept a statistics tail whose last line has no newline.
    if (statistics_tail) text += " General timing and accounting informations for this job:";
    std::istringstream input(text);
    GPIO io;
    REQUIRE(io.seq_read_init(input, "outcar"));
    Atoms first, second;
    REQUIRE(io.seq_read(input, first));
    REQUIRE(io.seq_read(input, second));
    REQUIRE(first.id == 0);
    REQUIRE(second.id == 1);
    REQUIRE(first.size() == 5);
    REQUIRE(second.size() == 5);
    REQUIRE(first.forces_.size() == 5);
    REQUIRE(second.forces_.size() == 5);
    if (mlff) {
        REQUIRE(first.energy_ == Approx(-10));
        REQUIRE(second.energy_ == Approx(-11));
        REQUIRE(first.cell_(0, 0) == Approx(2));
        REQUIRE(second.cell_(0, 0) == Approx(3));
        REQUIRE(first.pos_[0](0) == Approx(1));
        REQUIRE(second.pos_[0](0) == Approx(11));
    } else {
        check_fe2o3(first);
        check_fe2o3(second);
        REQUIRE(first.energy_ == Approx(-243));
        REQUIRE(second.energy_ == Approx(-243));
    }
    const double saved = first.pos_[0](0);
    second.pos_[0](0) += 100;
    REQUIRE(first.pos_[0](0) == Approx(saved));
    REQUIRE_FALSE(io.seq_read(input, second));
    REQUIRE(second.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, second));
}

TEST_CASE("GPIO rejects incomplete started OUTCAR steps without consuming a later step", "[gpio][outcar][stream]")
{
    std::ostringstream step_out;
    append_step(step_out, 4, 5);
    const auto complete = step_out.str();
    std::string broken = complete;
    SECTION("electronic iteration only") { broken = std::string(kIterHeader) + " 1(1)\n"; }
    SECTION("missing cell") {
        const auto start = broken.find(" VOLUME and BASIS-vectors");
        broken.erase(start, broken.find(kPositionHeader) - start);
    }
    SECTION("incomplete cell row") {
        const auto start = broken.find("      0.86602540");
        broken.replace(start, broken.find('\n', start) - start, "      1 2");
    }
    SECTION("missing positions and forces") {
        const auto start = broken.find(kPositionHeader);
        broken.erase(start, broken.find(kTotenLine) - start);
    }
    SECTION("missing force column") {
        const auto start = broken.find("       1.00000000");
        broken.replace(start, broken.find('\n', start) - start, " 1 2 3 0 0");
    }
    SECTION("truncated force rows") { broken.resize(broken.find("       3.00000000")); }
    SECTION("missing energy value") {
        const auto start = broken.find(kTotenLine);
        broken.resize(start);
        broken += std::string(kTotenLine) + " invalid eV\n";
    }
    SECTION("missing energy before next step") {
        broken.resize(broken.find(kTotenLine));
        broken += complete;
    }
    const auto text = make_outcar(" vasp.5.4.4", 60, 1, 4) + broken;
    GPIO io;
    std::istringstream input(text);
    REQUIRE(io.seq_read_init(input, "outcar"));
    Atoms atom;
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.energy_ == Approx(-243));
    REQUIRE_THROWS_WITH(io.seq_read(input, atom), Catch::Matchers::ContainsSubstring("OUTCAR frame 1"));
    REQUIRE(atom.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, atom));

    GPIO bulk_io;
    std::istringstream bulk_input(text);
    std::vector<Atoms> atoms;
    REQUIRE_FALSE(bulk_io.read(atoms, bulk_input, "outcar"));
    REQUIRE(atoms.size() == 1);
    GPIO one_io;
    std::istringstream one_input(text);
    REQUIRE(one_io.read_one(atom, one_input, "outcar"));
    REQUIRE(atom.energy_ == Approx(-243));
}

TEST_CASE("GPIO OUTCAR streaming preserves free-energy and sigma-zero selection", "[gpio][outcar][stream]")
{
    struct RestoreEnergySelection {
        bool previous = GPIO::vasp_use_free_eng;
        ~RestoreEnergySelection() { GPIO::vasp_use_free_eng = previous; }
    } restore;
    auto text = make_outcar(" vasp.5.4.4", 60, 1, 4);
    text += "\n  energy  without entropy= -242.5  energy(sigma->0) = -242.25\n";
    SECTION("default free energy") { GPIO::vasp_use_free_eng = true; }
    SECTION("sigma zero") { GPIO::vasp_use_free_eng = false; }
    const double expected = GPIO::vasp_use_free_eng ? -243 : -242.25;
    std::istringstream input(text);
    GPIO io;
    Atoms atom;
    REQUIRE(io.seq_read_init(input, "outcar"));
    REQUIRE(io.seq_read(input, atom));
    REQUIRE(atom.energy_ == Approx(expected));
    REQUIRE_FALSE(io.seq_read(input, atom));
}

TEST_CASE("GPIO OUTCAR sigma-zero selection requires the selected energy line", "[gpio][outcar][stream]")
{
    struct RestoreEnergySelection {
        bool previous = GPIO::vasp_use_free_eng;
        ~RestoreEnergySelection() { GPIO::vasp_use_free_eng = previous; }
    } restore;
    GPIO::vasp_use_free_eng = false;
    auto text = make_outcar(" vasp.5.4.4", 60, 1, 4);
    SECTION("truncated sigma line") { text += "\n  energy without entropy= -242.5\n"; }
    SECTION("next ionic step is not a sigma energy") { text += make_outcar(" vasp.5.4.4", 60, 1, 4); }
    std::istringstream input(text);
    GPIO io;
    Atoms atom;
    REQUIRE(io.seq_read_init(input, "outcar"));
    REQUIRE_THROWS_WITH(io.seq_read(input, atom), Catch::Matchers::ContainsSubstring("OUTCAR frame 0"));
    REQUIRE(atom.size() == 0);
}
