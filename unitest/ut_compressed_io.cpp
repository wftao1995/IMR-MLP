// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "utils/compressed_stream.h"
#include "utils/gputils.h"

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>

using Catch::Approx;
// Note: unitest/main_init.cpp (linked into every test binary via tests.mk)
// pre-applies sync_with_stdio(false) so GPIO's constructor does not orphan
// Catch2's reporter stream.

// ── helpers ─────────────────────────────────────────────────────────────────

static std::string read_all(const std::string &path)
{
    auto is = open_ifile(path);
    REQUIRE(is != nullptr);
    return std::string((std::istreambuf_iterator<char>(*is)), std::istreambuf_iterator<char>());
}

static std::string raw_file_content(const std::string &path)
{
    std::ifstream ifs(path, std::ios::binary);
    REQUIRE(ifs.is_open());
    return std::string((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
}

static Atoms make_compressed_io_atom()
{
    Atoms atom;
    atom.cell_ << 4.0, 0.0, 0.0,
                  0.0, 5.0, 0.0,
                  0.0, 0.0, 6.0;
    atom.pos_ = {{0.1, 0.2, 0.3}, {1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}};
    atom.chemmap_ = {0, 1, 0};
    atom.pbc_ = {{true, true, true}};
    atom.energy_ = -12.5;
    atom.have_energy = true;
    return atom;
}

// ── stream layer ────────────────────────────────────────────────────────────

TEST_CASE("open_ofile returns null when its parent is not a directory", "[compressed][errors]")
{
    const std::string parent = "cs_not_a_directory";
    std::ofstream(parent) << "file";
    for (const char *suffix : {".txt", ".gz", ".zst"})
        REQUIRE(open_ofile(parent + "/out" + suffix) == nullptr);
    std::remove(parent.c_str());
}

// Codecs actually compiled in (nullptr sentinel keeps the array non-empty
// and terminates the loop when all codecs are disabled).
static const char *compiled_codecs[] = {
#ifdef USE_GZIP
    ".gz",
#endif
    nullptr,
};

TEST_CASE("open_ofile/open_ifile codec roundtrip", "[compressed]")
{
    for (const char *ext : compiled_codecs) {
        if (ext == nullptr) break;
        std::string path = std::string("cs_stream_t") + ext;
        {
            auto os = open_ofile(path);
            REQUIRE(os != nullptr);
            *os << "hello compressed world\nline two\n";
            REQUIRE(close_ofile(os));
        }
        REQUIRE(read_all(path) == "hello compressed world\nline two\n");
        std::remove(path.c_str());
    }
}

#ifdef USE_GZIP
TEST_CASE("open_ifile detects compression by magic bytes without extension", "[compressed]")
{
    const std::string path = "cs_magic.dat";  // no .gz extension
    {
        auto os = open_ofile("cs_magic_src.gz");
        *os << "magic detection content\n";
        REQUIRE(close_ofile(os));
    }
    // copy the gzip bytes to the extension-less name
    {
        std::string content = raw_file_content("cs_magic_src.gz");
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(content.data(), (std::streamsize)content.size());
    }
    REQUIRE(probe_compression(path) == CompressionKind::Gzip);
    REQUIRE(read_all(path) == "magic detection content\n");
    std::remove(path.c_str());
    std::remove("cs_magic_src.gz");
}
#endif  // USE_GZIP

TEST_CASE("plain files still read through open_ifile", "[compressed]")
{
    const std::string path = "cs_plain.txt";
    {
        std::ofstream ofs(path);
        ofs << "plain text\n";
    }
    REQUIRE(read_all(path) == "plain text\n");
    std::remove(path.c_str());
}

// ── GPIO integration ────────────────────────────────────────────────────────

TEST_CASE("GPIO extxyz write/read roundtrip through compiled-in codecs", "[compressed][gpio]")
{
    Atoms input = make_compressed_io_atom();
    for (const char *ext : compiled_codecs) {
        if (ext == nullptr) break;
        std::string path = std::string("cs_xyz_t") + ext;
        GPIO writer;
        REQUIRE(writer.write_one(input, path, "extxyz"));
        // validate the produced file is really compressed
        std::string raw = raw_file_content(path);
        REQUIRE(((unsigned char)raw[0] == 0x1f && (unsigned char)raw[1] == 0x8b));

        std::vector<Atoms> atoms;
        GPIO reader;
        REQUIRE(reader.read(atoms, path, ""));  // auto-detect from magic/extension
        REQUIRE(atoms.size() == 1);
        REQUIRE(atoms[0].chemmap_ == input.chemmap_);
        REQUIRE(atoms[0].pos_[0].isApprox(input.pos_[0]));
        REQUIRE(atoms[0].pos_[2].isApprox(input.pos_[2]));
        REQUIRE(atoms[0].energy_ == Approx(input.energy_));
        std::remove(path.c_str());
    }
}

#ifdef USE_GZIP
TEST_CASE("GPIO read auto-detects compressed POSCAR.gz", "[compressed][gpio]")
{
    // POSCAR writes species grouped in chemmap order (Al Al Si), so build the
    // input already grouped for an exact roundtrip.
    Atoms input = make_compressed_io_atom();
    input.chemmap_ = {0, 0, 1};
    input.pos_ = {{0.1, 0.2, 0.3}, {2.1, 2.2, 2.3}, {1.1, 1.2, 1.3}};
    const std::string path = "cs_POSCAR.gz";
    GPIO writer;
    REQUIRE(writer.write_one(input, path, "poscar"));

    Atoms output;
    GPIO reader;
    REQUIRE(reader.read_one(output, path, ""));
    REQUIRE(output.size() == input.size());
    REQUIRE(output.chemmap_ == input.chemmap_);
    for (int i = 0; i < input.size(); ++i)
        REQUIRE(output.pos_[i].isApprox(input.pos_[i]));
    std::remove(path.c_str());
}
#endif  // USE_GZIP

TEST_CASE("GPIO binary MLIPBIN roundtrip through compiled-in codecs", "[compressed][gpio][binary]")
{
    Atoms input = make_compressed_io_atom();
    input.forces_ = {{0.01, 0.02, 0.03}, {0.11, 0.12, 0.13}, {0.21, 0.22, 0.23}};
    input.have_stress = false;
    for (const char *ext : compiled_codecs) {
        if (ext == nullptr) break;
        std::string path = std::string("cs_bin_t") + ext;
        GPIO writer;
        REQUIRE(writer.write_one(input, path, "bin"));

        Atoms output;
        GPIO reader;
        REQUIRE(reader.read_one(output, path, "bin"));
        REQUIRE(output.size() == input.size());
        REQUIRE(output.cell_.isApprox(input.cell_));
        for (int i = 0; i < input.size(); ++i) {
            REQUIRE(output.pos_[i].isApprox(input.pos_[i]));
            REQUIRE(output.forces_[i].isApprox(input.forces_[i]));
        }
        std::remove(path.c_str());
    }
}

#ifdef USE_GZIP
TEST_CASE("GPIO multi-frame compressed trajectory", "[compressed][gpio]")
{
    std::string path = "cs_multi.xyz.gz";
    std::vector<Atoms> input;
    for (int i = 0; i < 3; ++i) {
        Atoms atom = make_compressed_io_atom();
        atom.pos_[0][0] += 0.1 * i;
        atom.energy_ += 0.5 * i;
        input.push_back(atom);
    }
    GPIO writer;
    REQUIRE(writer.write(input, path, "extxyz"));
    std::vector<Atoms> atoms;
    GPIO reader;
    REQUIRE(reader.read(atoms, path, ""));
    REQUIRE(atoms.size() == 3);
    for (int i = 0; i < 3; ++i) {
        REQUIRE(atoms[i].energy_ == Approx(-12.5 + 0.5 * i));
        REQUIRE(atoms[i].pos_[0][0] == Approx(0.1 + 0.1 * i));
    }
    std::remove(path.c_str());
}
#endif  // USE_GZIP

#ifdef USE_GZIP
TEST_CASE("compressed multi-member append is readable", "[compressed][gpio]")
{
    const std::string path = "cs_append.xyz.gz";
    // two separate gzip members written by two separate open_ofile calls
    {
        auto os = open_ofile(path);
        REQUIRE(os != nullptr);
        *os << "2\nLattice=\"5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0\" Properties=species:S:1:pos:R:3 energy=1.0\n"
               "Si 0.0 0.0 0.0\nSi 1.0 1.0 1.0\n";
        REQUIRE(close_ofile(os));
    }
    {
        auto os = open_ofile(path, /*append=*/true);
        REQUIRE(os != nullptr);
        *os << "2\nLattice=\"5.0 0.0 0.0 0.0 5.0 0.0 0.0 0.0 5.0\" Properties=species:S:1:pos:R:3 energy=2.0\n"
               "Si 0.0 0.0 0.0\nSi 1.0 1.0 1.0\n";
        REQUIRE(close_ofile(os));
    }
    std::vector<Atoms> atoms;
    GPIO reader;
    REQUIRE(reader.read(atoms, path, "extxyz"));
    REQUIRE(atoms.size() == 2);
    REQUIRE(atoms[0].energy_ == Approx(1.0));
    REQUIRE(atoms[1].energy_ == Approx(2.0));
    std::remove(path.c_str());
}
#endif  // USE_GZIP

TEST_CASE("guess_format strips compression extensions", "[compressed][gpio]")
{
    REQUIRE(GPIO::guess_format("structure.xyz.gz") == GPIO::EXTXYZ);
    REQUIRE(GPIO::guess_format("structure.xyz.zst") == GPIO::EXTXYZ);
    REQUIRE(GPIO::guess_format("POSCAR.gz") == GPIO::POSCAR);
    REQUIRE(GPIO::guess_format("vasprun.xml.gz") == GPIO::VASPRUN);
    REQUIRE(GPIO::guess_format("data.bin.gz") == GPIO::MLIPBIN);
    REQUIRE(GPIO::guess_format("traj.lammpstrj.gz") == GPIO::LMPTRJ);
    REQUIRE(GPIO::guess_format("plain.xyz") == GPIO::EXTXYZ);
}

// ── settings / map files ────────────────────────────────────────────────────

#ifdef USE_GZIP
TEST_CASE("GPSettings loads from compressed .ini.gz", "[compressed][settings]")
{
    const std::string path = "cs_settings.ini.gz";
    {
        auto os = open_ofile(path);
        REQUIRE(os != nullptr);
        // GPSettings expects "key value" pairs (value = everything after the
        // first space); comments start with '#'.
        *os << "[pot]\n"
               "pottype mtp\n"
               "potfile potential.mtp\n"
               "[lbfgspp]\n"
               "maxiter 100\n";
        REQUIRE(close_ofile(os));
    }
    GPSettings settings;
    settings.Load(path);
    REQUIRE(settings.count("pot:pottype") == 1);
    REQUIRE(settings["pot:pottype"] == "mtp");
    REQUIRE(settings["pot:potfile"] == "potential.mtp");
    REQUIRE(settings["lbfgspp:maxiter"] == "100");
    std::remove(path.c_str());
}
#endif  // USE_GZIP

#ifdef USE_GZIP
TEST_CASE("GPIO readmap works from compressed index.map.gz", "[compressed][gpio]")
{
    const std::string path = "cs_index.map.gz";
    {
        auto os = open_ofile(path);
        REQUIRE(os != nullptr);
        *os << "Si 2.22 14\n"
               "Ge 2.30 32\n";
        REQUIRE(close_ofile(os));
    }
    GPIO iobj;
    REQUIRE(iobj.readmap(path));
    REQUIRE(iobj.getrevmap().count("Si") == 1);
    REQUIRE(iobj.getrevmap()["Si"] == 0);
    REQUIRE(iobj.getrevmap()["Ge"] == 1);
    REQUIRE(iobj.getmap()[0] == "Si");
    std::remove(path.c_str());
}
#endif  // USE_GZIP

// ── error handling ──────────────────────────────────────────────────────────

#ifdef USE_GZIP
TEST_CASE("corrupt gzip data fails gracefully", "[compressed][gpio]")
{
    const std::string path = "cs_bad.gz";
    {
        const char bad_gzip[] = "\x1f\x8b\x08\x00garbage-not-gzip";
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(bad_gzip, sizeof(bad_gzip) - 1);
    }
    // stream layer throws
    REQUIRE_THROWS_AS(open_ifile(path), std::runtime_error);
    // GPIO layer converts to a clean false
    std::vector<Atoms> atoms;
    GPIO reader;
    REQUIRE_FALSE(reader.read(atoms, path, ""));
    std::remove(path.c_str());
}
#endif  // USE_GZIP


// ── disabled-codec diagnostics (only when the codec is NOT compiled in) ─────

#ifndef USE_GZIP
TEST_CASE("disabled gzip reports a clear error", "[compressed]")
{
    const std::string path = "cs_disabled.gz";
    {
        const char gzip_magic[] = "\x1f\x8b\x08\x00";
        std::ofstream ofs(path, std::ios::binary);
        ofs.write(gzip_magic, sizeof(gzip_magic) - 1);
    }
    REQUIRE(probe_compression(path) == CompressionKind::Gzip);  // recognition stays on
    REQUIRE(open_ifile(path) == nullptr);                       // decoding rejected
    REQUIRE(open_ofile(path) == nullptr);                       // never plain-write .gz
    std::remove(path.c_str());
}
#endif  // !USE_GZIP

TEST_CASE("unsupported compression is rejected without touching output", "[compressed][gpio]")
{
    const std::string path = GENERATE("cs_rejected.xyz.zst", "cs_rejected.xyz.ZST");
    const bool append = GENERATE(false, true);
    const std::string preserved = "preserve this file\n";
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs << preserved;
    }
    REQUIRE(open_ifile(path) == nullptr);
    REQUIRE(open_ofile(path, append) == nullptr);
    REQUIRE(raw_file_content(path) == preserved);
    GPIO io;
    io.append = append;
    Atoms atom = make_compressed_io_atom();
    std::vector<Atoms> atoms = {atom};
    REQUIRE_FALSE(io.write(atoms, path, "extxyz"));
    REQUIRE_FALSE(io.write_one(atom, path, "extxyz"));
    REQUIRE_FALSE(io.read(atoms, path, "extxyz"));
    REQUIRE_FALSE(io.read_one(atom, path, "extxyz"));
    REQUIRE(raw_file_content(path) == preserved);
    std::remove(path.c_str());
    REQUIRE(open_ofile(path) == nullptr);
    REQUIRE_FALSE(GPIO::file_exist(path));
}

TEST_CASE("Zstandard magic is rejected without a compression suffix", "[compressed][gpio]")
{
    const std::string path = "cs_rejected.xyz";
    const std::string content("\x28\xb5\x2f\xfd", 4);
    {
        std::ofstream ofs(path, std::ios::binary);
        ofs << content;
    }
    REQUIRE(probe_compression(path) == CompressionKind::Unsupported);
    REQUIRE(open_ifile(path) == nullptr);
    GPIO io;
    std::vector<Atoms> atoms;
    Atoms atom;
    REQUIRE_FALSE(io.read(atoms, path, "extxyz"));
    REQUIRE_FALSE(io.read_one(atom, path, "extxyz"));
    REQUIRE(raw_file_content(path) == content);
    std::remove(path.c_str());
}

#ifdef USE_GZIP
TEST_CASE("Gzip members crossing input buffer boundaries are read completely", "[compressed][gzip]")
{
    const int boundary = GENERATE(65535, 65536, 65537);
    const std::string path = "cs_boundary.gz";
    {
        auto os = open_ofile(path);
        *os << "first member\n";
        REQUIRE(close_ofile(os));
    }
    auto member = raw_file_content(path);
    REQUIRE(member.size() > 10);
    REQUIRE((member[3] & 4) == 0);
    // A legal optional gzip extra field pads the first member to the boundary.
    const size_t extra_size = boundary - member.size() - 2;
    REQUIRE(extra_size <= 65535);
    member[3] |= 4;
    std::string extra;
    extra.push_back(static_cast<char>(extra_size & 255));
    extra.push_back(static_cast<char>(extra_size >> 8));
    extra.append(extra_size, '\0');
    member.insert(10, extra);
    REQUIRE(member.size() == boundary);
    {
        std::ofstream os(path, std::ios::binary);
        os.write(member.data(), member.size());
    }
    REQUIRE(read_all(path) == "first member\n");
    {
        auto os = open_ofile(path, true);
        *os << "second member\n";
        REQUIRE(close_ofile(os));
    }
    REQUIRE(read_all(path) == "first member\nsecond member\n");
    auto complete = raw_file_content(path);
    SECTION("truncated second member") { complete.resize(complete.size() - 4); }
    SECTION("corrupt second member") { complete[boundary] = 'x'; }
    {
        std::ofstream os(path, std::ios::binary);
        os.write(complete.data(), complete.size());
    }
    REQUIRE_THROWS_AS(open_ifile(path), std::runtime_error);
    std::remove(path.c_str());
}

#endif
