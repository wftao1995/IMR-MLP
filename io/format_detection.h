#pragma once
#include <cstddef>
#include <istream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace GPIOFormat {
inline constexpr std::size_t MAX_PREFIX = 65536;
// A unique reliable signature sets format and its sole candidate. Unknown has
// no candidates; conflicting evidence has multiple candidates and no format.
struct Detection {
    std::string format;
    std::vector<std::string> candidates;
};
Detection detect_prefix(std::string_view prefix);
struct ProbedInput {
    std::unique_ptr<std::istream> input;
    Detection detection;
};
// Consumes at most limit decompressed bytes and returns a non-seeking stream
// replaying the complete prefix before the original remainder. limit: 1..64KiB.
ProbedInput probe(std::unique_ptr<std::istream> input, std::size_t limit = MAX_PREFIX);
// Names/extensions never supply evidence. Opening uses progressive decoding.
Detection detect_file(const std::string &path, std::size_t limit = MAX_PREFIX);
}
