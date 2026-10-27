#include <catch2/catch_all.hpp>
#include "io/gpio.h"
#include <sstream>
#include <streambuf>

namespace {
const std::string prefix = "ANIMSTEPS 2\nCRYSTAL\nPRIMVEC\n4 0 0\n0 5 0\n0 0 6\nPRIMCOORD 1\n1 1\nH .25 .5 .75 .1 -.2 .3\n";
struct OnePass : std::streambuf {
    std::string text;
    size_t position = 0, fail_at;
    int seeks = 0;
    explicit OnePass(std::string data, size_t failure) : text(std::move(data)), fail_at(failure) {}
    int_type underflow() override {
        if (position == fail_at) throw std::runtime_error("unread XSF suffix accessed");
        return position == text.size() ? traits_type::eof() : traits_type::to_int_type(text[position]);
    }
    int_type uflow() override {
        const auto value = underflow();
        if (!traits_type::eq_int_type(value, traits_type::eof())) ++position;
        return value;
    }
    pos_type seekoff(off_type, std::ios_base::seekdir, std::ios_base::openmode) override { ++seeks; return pos_type(off_type(-1)); }
    pos_type seekpos(pos_type, std::ios_base::openmode) override { ++seeks; return pos_type(off_type(-1)); }
};
}

TEST_CASE("XSF native sequence yields counted frames lazily on nonseekable input", "[gpio][xsf][stream]") {
    OnePass data(prefix + "PRIMCOORD 2\n1 1\nH 1 2 3\n", prefix.size());
    std::istream input(&data);
    GPIO io;
    REQUIRE(io.seq_read_init(input, "xsf")); REQUIRE(data.position == 0);
    Atoms a;
    REQUIRE(io.seq_read(input, a)); REQUIRE(data.position == prefix.size());
    REQUIRE(a.pos_[0].isApprox(Eigen::RowVector3d(.25, .5, .75)));
    REQUIRE(a.forces_[0][0] == Catch::Approx(2.7211396641308));
    const Atoms owned = a;
    REQUIRE_THROWS(io.seq_read(input, a)); REQUIRE(a.size() == 0);
    REQUIRE_FALSE(io.seq_read(input, a)); REQUIRE(data.seeks == 0);
    REQUIRE(owned.forces_[0][1] == Catch::Approx(-5.4422793282616));
}

TEST_CASE("XSF pending molecular frame and cell state are value copied and reset", "[gpio][xsf][stream]") {
    const std::string text = "ANIMSTEPS 2\nATOMS 1\nH 0 0 0\nATOMS 2\nH 1 2 3\n";
    std::istringstream input(text);
    GPIO io;
    REQUIRE(io.seq_read_init(input, "axsf"));
    Atoms first;
    REQUIRE(io.seq_read(input, first));
    GPIO copied = io;
    const auto position = static_cast<size_t>(input.tellg());
    std::istringstream remaining(text.substr(position));
    Atoms a, b;
    REQUIRE(io.seq_read(input, a)); REQUIRE(copied.seq_read(remaining, b));
    REQUIRE(a.id == 1); REQUIRE(b.id == 1); REQUIRE(a.pos_ == b.pos_);
    a.pos_[0].setZero(); REQUIRE(b.pos_[0][2] == 3); REQUIRE(first.pos_[0].isZero());
    REQUIRE_FALSE(io.seq_read(input, a)); REQUIRE_FALSE(copied.seq_read(remaining, b));
    copied.clean_state();
    std::istringstream restarted(prefix + "PRIMCOORD 2\n1 1\nH 2 3 4\n");
    REQUIRE(copied.seq_read_init(restarted, "xsf"));
    REQUIRE(copied.seq_read(restarted, b)); REQUIRE(b.id == 0); REQUIRE(b.is_fully_periodic());
    REQUIRE(b.cell_(1, 1) == 5);
}

TEST_CASE("XSF sequence reports declared missing frames and malformed late frames", "[gpio][xsf][stream]") {
    for (const auto &suffix : {std::string(""), std::string("PRIMCOORD 3\n1 1\nH 1 2 3\n"),
                              std::string("PRIMCOORD 2\n2 1\nH 1 2 3\n"),
                              std::string("PRIMCOORD 2\n1 1\nH 1 2\n")}) {
        CAPTURE(suffix);
        std::istringstream input(prefix + suffix);
        GPIO io; Atoms a;
        REQUIRE(io.seq_read_init(input, "xsf")); REQUIRE(io.seq_read(input, a));
        REQUIRE_THROWS_WITH(io.seq_read(input, a), Catch::Matchers::ContainsSubstring("XSF invalid or truncated frame 1"));
        REQUIRE(a.size() == 0); REQUIRE_FALSE(io.seq_read(input, a));
    }
}
