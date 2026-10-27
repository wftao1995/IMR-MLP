#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include "utils/compressed_stream.h"
#include "utils/cmdopts.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>

int relaxstrv2(int argc, char *argv[]);

namespace {
// N2P2 is read-only; use its native Bohr/Hartree records directly.
const std::string n2p2_first =
    "begin\n"
    "lattice 20 1 .5\nlattice 0 21 .7\nlattice 0 0 22\n"
    "atom 4 4 4 H 0 0 0 0 0\natom 6.5 4.5 4.2 H 0 0 0 0 0\n"
    "energy 0\ncharge 0\nend\n";
const std::string n2p2_second =
    "begin\n"
    "lattice 22 1 .5\nlattice 0 23 .7\nlattice 0 0 24\n"
    "atom 4 4 4 H 0 0 0 0 0\natom 6.7 4.5 4.2 H 0 0 0 0 0\n"
    "atom 14 14 14 H 0 0 0 0 0\nenergy 0\ncharge 0\nend\n";
struct Directory {
    std::filesystem::path previous = std::filesystem::current_path();
    std::filesystem::path path;
    Directory() {
        char pattern[] = "/tmp/imrmlp_relax_stream_XXXXXX";
        const char *directory = mkdtemp(pattern);
        if (!directory) throw std::runtime_error("Cannot create stream test directory");
        path = directory;
        std::filesystem::current_path(path);
        std::ofstream("index.map") << "H 0\n";
    }
    ~Directory() {
        CMDOption::global_cmdopts.reset();
        std::filesystem::current_path(previous);
        std::filesystem::remove_all(path);
    }
};
std::vector<Atoms> frames() {
    std::vector<Atoms> atoms(3);
    for (int i = 0; i < 3; ++i) {
        auto &a = atoms[i];
        a.id = i;
        a.cell_ << 10 + i, 1, .5, 0, 11 + i, .7, 0, 0, 12 + i;
        a.pos_ = {{2, 2, 2}, {3.4 + .1*i, 2.3, 2.2}};
        if (i == 1) a.pos_.push_back({7, 7, 7});
        a.chemmap_.assign(a.size(), 0);
    }
    return atoms;
}
void write_frames(const std::string &path = "input.cfg", const std::string &format = "mlpcfg") {
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    auto atoms = frames();
    REQUIRE(io.write(atoms, path, format));
}
void settings(const std::string &extra = "") {
    std::istringstream text(
        "input input.cfg\ninput-format mlpcfg\noutput output.xyz\noutput-format extxyz\n"
        "pot:pottype lj612\npot:ntypes 1\npot:0-0 1 1\npot:cut:0-0 4\n"
        "isif 2\nrelax:gtol 1e-5\nrelax:maxiteration 300\nrelax:write_traj False\n" + extra);
    std::map<std::string, std::string> values;
    for (std::string line; std::getline(text, line); ) {
        const auto split = line.find(' ');
        if (split != std::string::npos) values[line.substr(0, split)] = line.substr(split + 1);
    }
    std::ofstream output("relax.ini");
    for (const auto &[key, value] : values) output << key << ' ' << value << '\n';
}
int run(std::vector<std::string> extra = {}) {
    std::vector<std::string> args{"relax", "relax.ini", "--num_threads", "1"};
    args.insert(args.end(), extra.begin(), extra.end());
    std::vector<char *> argv;
    for (auto &arg : args) argv.push_back(arg.data());
    CMDOption::global_cmdopts.reset();
    return relaxstrv2(argv.size(), argv.data());
}
std::vector<Atoms> read_output(const std::string &path = "output.xyz", const std::string &format = "extxyz") {
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    std::vector<Atoms> atoms;
    REQUIRE(io.read(atoms, path, format));
    return atoms;
}
void same_results(const std::vector<Atoms> &a, const std::vector<Atoms> &b) {
    REQUIRE(a.size() == b.size());
    for (size_t k = 0; k < a.size(); ++k) {
        CAPTURE(k);
        REQUIRE(a[k].size() == b[k].size());
        REQUIRE(a[k].cell_.isApprox(b[k].cell_, 1e-9));
        REQUIRE(a[k].energy_ == Catch::Approx(b[k].energy_).margin(1e-8));
        REQUIRE(a[k].stress_.isApprox(b[k].stress_, 1e-8));
        REQUIRE(a[k].forces_.size() == b[k].forces_.size());
        for (int j = 0; j < a[k].size(); ++j) {
            REQUIRE(a[k].pos_[j].isApprox(b[k].pos_[j], 1e-9));
            REQUIRE((a[k].forces_[j] - b[k].forces_[j]).norm() < 1e-8);
        }
    }
}
}

TEST_CASE("V2 sequential relaxation matches batch for changing atom counts and cells", "[relax_stream][cli]") {
    Directory directory;
    const std::string format = GENERATE("mlpcfg", "xdatcar6", "extxyz");
    const bool masked = GENERATE(false, true);
    write_frames("input.cfg", format);
    settings(masked ? "position-mask 1 0 0\n" : "");
    REQUIRE(run({"--input-format", format}) == 0);
    const auto batch = read_output();
    REQUIRE(run({"--input-format", format, "--seq-read", "--seq-write=False"}) == 0);
    same_results(batch, read_output());
    if (masked) {
        const auto initial = frames();
        for (size_t i = 0; i < batch.size(); ++i)
            for (int j = 0; j < batch[i].size(); ++j) {
                REQUIRE(batch[i].pos_[j](1) == Catch::Approx(initial[i].pos_[j](1)).margin(1e-8));
                REQUIRE(batch[i].pos_[j](2) == Catch::Approx(initial[i].pos_[j](2)).margin(1e-8));
            }
    }
}

TEST_CASE("V2 sequential relaxation reads all supported inputs and writes compressed streams", "[relax_stream][compressed]") {
    Directory directory;
    std::vector<std::string> suffixes{""};
#ifdef USE_GZIP
    suffixes.push_back(".gz");
#endif
#ifdef USE_ZSTD
    suffixes.push_back(".zst");
#endif
    const std::string input_format = GENERATE("mlpcfg", "xdatcar6", "extxyz");
    for (const auto &suffix : suffixes) {
        for (const std::string format : {"mlpcfg", "extxyz"}) {
            CAPTURE(suffix, format);
            const std::string input = "input.cfg" + suffix;
            const std::string output = (format == "mlpcfg" ? "output.cfg" : "output.xyz") + suffix;
            write_frames(input, input_format);
            settings("position-mask 0 0 0\n");
            REQUIRE(run({"--seq-read", "--input", input, "--input-format", input_format, "--output", output, "--output-format", format}) == 0);
            const auto actual = read_output(output, format);
            const auto initial = frames();
            REQUIRE(actual.size() == initial.size());
            for (size_t i = 0; i < actual.size(); ++i) {
                REQUIRE(actual[i].size() == initial[i].size());
                for (int j = 0; j < actual[i].size(); ++j)
                    REQUIRE(actual[i].pos_[j].isApprox(initial[i].pos_[j], 1e-8));
            }
        }
    }
}

TEST_CASE("V2 sequential relaxation distinguishes a broken frame from EOF and preserves completed output", "[relax_stream][errors]") {
    Directory directory;
    const std::string format = GENERATE("mlpcfg", "xdatcar6", "extxyz");
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    auto atom = frames()[0];
    auto os = open_ofile("input.cfg");
    REQUIRE(io.seq_write_init(*os, format));
    REQUIRE(io.seq_write(*os, atom));
    *os << (format == "mlpcfg" ? "BEGIN_CFG\nSize\n2\nSupercell\n1 0\n" : format == "xdatcar6" ? "bad frame\n1.0\n1 0\n" : "2\nProperties=species:S:1:pos:R:3\nH 0 0 bad\n");
    REQUIRE(close_ofile(os));
    settings("position-mask 0 0 0\n");
    REQUIRE(run({"--seq-read", "--input-format", format}) == 1);
    const auto result = read_output();
    REQUIRE(result.size() == 1);
    REQUIRE(result[0].pos_[0].isApprox(atom.pos_[0], 1e-8));
    std::filesystem::remove("output.xyz");
    REQUIRE(run({"--input-format", format}) == 1);
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
}

TEST_CASE("V2 sequential relaxation rejects unsupported modes before opening output", "[relax_stream][errors]") {
    Directory directory;
    write_frames();
    const std::string extra = GENERATE("input-format poscar\n", "output-format poscar\n",
        "seldyn masks.dat\n", "filter masks.dat\n",
        "al:mode committee\n", "relax:selective_dynamics True\nrelax:seldyn_file masks.dat\n");
    std::ofstream("masks.dat") << "6 1 1 1 1 1 1\n";
    settings(extra);
    REQUIRE_THROWS_WITH(run({"--seq-read"}), Catch::Matchers::ContainsSubstring("seq-read"));
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
}

TEST_CASE("V2 sequential relaxation protects input inode including links", "[relax_stream][errors]") {
    Directory directory;
    write_frames();
    settings();
    const std::string kind = GENERATE("same", "hard", "symbolic");
    std::string output = "input.cfg";
    if (kind == "hard") { output = "linked.cfg"; std::filesystem::create_hard_link("input.cfg", output); }
    if (kind == "symbolic") { output = "linked.cfg"; std::filesystem::create_symlink("input.cfg", output); }
    const auto size = std::filesystem::file_size("input.cfg");
    REQUIRE_THROWS_WITH(run({"--seq-read", "--output", output}), Catch::Matchers::ContainsSubstring("same"));
    REQUIRE(std::filesystem::file_size("input.cfg") == size);
}

TEST_CASE("V2 sequential relaxation reports empty input and output failure", "[relax_stream][errors]") {
    Directory directory;
    settings();
    std::ofstream("input.cfg") << "\n";
    REQUIRE(run({"--seq-read"}) == 1);
    REQUIRE_FALSE(std::filesystem::exists("output.xyz"));
    write_frames();
    REQUIRE(run({"--seq-read", "--output", "/dev/full"}) == 1);
}

TEST_CASE("V2 sequential relaxation returns nonconvergence and retains final structures", "[relax_stream][cli]") {
    Directory directory;
    write_frames();
    settings("relax:gtol 1e-30\nrelax:maxiteration 1\n");
    REQUIRE(run({"--seq-read"}) == 2);
    REQUIRE(read_output().size() == 3);
}

TEST_CASE("V2 sequential relaxation accepts format inference and does not accumulate frames", "[relax_stream][cli]") {
    Directory directory;
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    auto os = open_ofile("input.cfg");
    REQUIRE(io.seq_write_init(*os, "mlpcfg"));
    auto atom = frames()[0];
    for (int i = 0; i < 40; ++i) {
        atom.pos_[0](2) = 2 + .001*i;
        REQUIRE(io.seq_write(*os, atom));
    }
    REQUIRE(close_ofile(os));
    settings("position-mask 0 0 0\n");
    REQUIRE(run({"--seq-read", "--input-format", "", "--output-format", ""}) == 0);
    const auto result = read_output();
    REQUIRE(result.size() == 40);
    for (int i = 0; i < 40; ++i)
        REQUIRE(result[i].pos_[0](2) == Catch::Approx(2 + .001*i).margin(1e-8));
}

TEST_CASE("V2 sequential and batch relaxation share default cell masks", "[relax_stream][cli][cell]") {
    Directory directory;
    write_frames();
    const bool strain = GENERATE(false, true);
    settings(std::string("isif 3\nrelax:selective_dynamics True\nrelax:cell_strain ") +
             (strain ? "True\n" : "False\n") + "relax:gtol 1e-30\nrelax:maxiteration 2\n");
    const int batch_status = run();
    REQUIRE(batch_status != 1);
    const auto batch = read_output();
    REQUIRE(run({"--seq-read"}) == batch_status);
    same_results(batch, read_output());
    REQUIRE_FALSE(batch[0].cell_.isApprox(frames()[0].cell_, 1e-10));
}

TEST_CASE("V2 sequential EXTXYZ infers compressed suffix and preserves no-newline final frame", "[relax_stream][extxyz]") {
    Directory directory;
    std::vector<std::string> suffixes{""};
#ifdef USE_GZIP
    suffixes.push_back(".gz");
#endif
#ifdef USE_ZSTD
    suffixes.push_back(".zst");
#endif
    for (const auto &suffix : suffixes) {
        const std::string filename = "input.xyz" + suffix;
        auto output = open_ofile(filename);
        *output << "2\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 pbc=\"T T T\"\n"
                   "H 2 2 2\nH 3.4 2.3 2.2";
        REQUIRE(close_ofile(output));
        settings("position-mask 0 0 0\n");
        REQUIRE(run({"--seq-read", "--input", filename, "--input-format", ""}) == 0);
        REQUIRE(read_output().size() == 1);
        REQUIRE(read_output()[0].pos_[1].isApprox(Eigen::RowVector3d(3.4, 2.3, 2.2), 1e-8));
    }
}

TEST_CASE("V2 sequential N2P2 relaxation matches batch for different atom counts and cells", "[relax_stream][n2p2]") {
    Directory directory;
    std::ofstream("input.n2p2") << n2p2_first << n2p2_second;
    settings("input input.n2p2\ninput-format n2p2\n");
    const auto initial = read_output("input.n2p2", "n2p2");
    REQUIRE(run() == 0);
    const auto batch = read_output();
    REQUIRE(batch.size() == 2);
    REQUIRE(batch[0].size() == 2);
    REQUIRE(batch[1].size() == 3);
    REQUIRE_FALSE(batch[0].cell_.isApprox(batch[1].cell_));
    for (size_t i = 0; i < batch.size(); ++i) {
        REQUIRE(batch[i].cell_.isApprox(initial[i].cell_, 1e-9));
        REQUIRE((batch[i].pos_[0] - initial[i].pos_[0]).norm() > 1e-4);
    }
    REQUIRE(run({"--seq-read"}) == 0);
    same_results(batch, read_output());
}

TEST_CASE("V2 sequential N2P2 infers compressed input format", "[relax_stream][n2p2][compressed]") {
    Directory directory;
    std::string filename = "input.n2p2";
#ifdef USE_GZIP
    filename += ".gz";
#elif defined(USE_ZSTD)
    filename += ".zst";
#else
    SKIP("Compression support not enabled");
#endif
    auto output = open_ofile(filename);
    *output << n2p2_first << n2p2_second;
    REQUIRE(close_ofile(output));
    settings("position-mask 0 0 0\n");
    REQUIRE(run({"--seq-read", "--input", filename, "--input-format", ""}) == 0);
    const auto actual = read_output();
    const auto initial = read_output(filename, "n2p2");
    REQUIRE(actual.size() == 2);
    for (size_t i = 0; i < actual.size(); ++i) {
        REQUIRE(actual[i].size() == initial[i].size());
        REQUIRE(actual[i].cell_.isApprox(initial[i].cell_, 1e-9));
        for (int j = 0; j < actual[i].size(); ++j)
            REQUIRE(actual[i].pos_[j].isApprox(initial[i].pos_[j], 1e-9));
    }
}

TEST_CASE("V2 sequential N2P2 keeps completed output after a truncated second frame", "[relax_stream][n2p2][errors]") {
    Directory directory;
    std::ofstream("input.n2p2") << n2p2_first << n2p2_second.substr(0, n2p2_second.size() - 4);
    settings("input input.n2p2\ninput-format n2p2\nposition-mask 0 0 0\n");
    GPIO io;
    REQUIRE(io.readmap("index.map"));
    std::istringstream first(n2p2_first);
    Atoms initial;
    REQUIRE(io.read_one(initial, first, "n2p2"));
    REQUIRE(run({"--seq-read"}) == 1);
    const auto actual = read_output();
    REQUIRE(actual.size() == 1);
    REQUIRE(actual[0].size() == initial.size());
    REQUIRE(actual[0].cell_.isApprox(initial.cell_, 1e-9));
    for (int j = 0; j < initial.size(); ++j)
        REQUIRE(actual[0].pos_[j].isApprox(initial.pos_[j], 1e-9));
}
