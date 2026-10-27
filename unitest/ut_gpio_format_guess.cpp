#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <filesystem>

TEST_CASE("Format inference ignores directory names for unknown basenames", "[gpio][format]")
{
    const std::string directory = GENERATE("vasprun-analysis", "binary-training", "xsf-results", "OUTCAR-cache");
    const std::filesystem::path relative = std::filesystem::path(directory) / "sample.unknown";
    REQUIRE(GPIO::guess_format("sample.unknown") == GPIO::UNKNOWN);
    REQUIRE(GPIO::guess_format(relative.string()) == GPIO::UNKNOWN);
    REQUIRE(GPIO::guess_format(std::filesystem::absolute(relative).string()) == GPIO::UNKNOWN);
    REQUIRE(GPIO::guess_format((relative.string() + ".gz")) == GPIO::UNKNOWN);
    REQUIRE(GPIO::guess_format((relative.string() + ".zst")) == GPIO::UNKNOWN);
}

TEST_CASE("Format inference retains conventional basenames and compression suffixes", "[gpio][format]")
{
    const auto [basename, expected] = GENERATE(
        std::pair{"POSCAR_001", GPIO::POSCAR},
        std::pair{"CONTCAR", GPIO::POSCAR},
        std::pair{"XDATCAR6", GPIO::XDATCAR6},
        std::pair{"XDATCAR", GPIO::XDATCAR},
        std::pair{"OUTCAR", GPIO::OUTCAR},
        std::pair{"vasprun.xml", GPIO::VASPRUN},
        std::pair{"vasprun", GPIO::VASPRUN},
        std::pair{"trajectory.cfgbin", GPIO::MLIPBIN});
    const std::string compression = GENERATE("", ".gz", ".zst");
    const std::filesystem::path relative = std::filesystem::path("binary-vasprun-xsf-archive") /
        (std::string(basename) + compression);
    REQUIRE(GPIO::guess_format(relative.string()) == expected);
    REQUIRE(GPIO::guess_format(std::filesystem::absolute(relative).string()) == expected);
}

TEST_CASE("Known suffixes retain priority over filename and directory heuristics", "[gpio][format]")
{
    const std::string compression = GENERATE("", ".gz", ".zst");
    REQUIRE(GPIO::guess_format("vasprun-cache/POSCAR_result.xyz" + compression) == GPIO::EXTXYZ);
    REQUIRE(GPIO::guess_format("binary-training/OUTCAR_result.cfg" + compression) == GPIO::MLPCFG);
    REQUIRE(GPIO::guess_format("xsf-cache/vasprun_result.bin" + compression) == GPIO::MLIPBIN);
}
