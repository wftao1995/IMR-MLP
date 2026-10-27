#include "format_detection.h"
#include "utils/compressed_stream.h"
#include <algorithm>
#include <charconv>
#include <cmath>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace GPIOFormat {
namespace {
std::string_view trim(std::string_view text) {
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.front()))) text.remove_prefix(1);
    while (!text.empty() && std::isspace(static_cast<unsigned char>(text.back()))) text.remove_suffix(1);
    return text;
}
std::vector<std::string_view> lines_of(std::string_view text) {
    std::vector<std::string_view> result;
    while (!text.empty()) {
        const auto newline = text.find('\n');
        if (newline == std::string_view::npos) { result.push_back(trim(text)); break; }
        result.push_back(trim(text.substr(0, newline)));
        text.remove_prefix(newline + 1);
    }
    return result;
}
std::vector<std::string> tokens_of(std::string_view line) {
    std::istringstream input{std::string(line)};
    std::vector<std::string> result;
    for (std::string token; input >> token;) result.push_back(std::move(token));
    return result;
}
bool finite_number(const std::string &token) {
    // iostream accepts the input dialect used by the text codecs, while an
    // exact tail check prevents a numeric prefix from counting as evidence.
    std::istringstream input(token);
    double value;
    return bool(input >> value) && input.peek() == std::char_traits<char>::eof() && std::isfinite(value);
}
bool species(const std::string &token) {
    return !token.empty() && token.size() <= 3 &&
           std::isupper(static_cast<unsigned char>(token.front())) &&
           std::all_of(token.begin() + 1, token.end(), [](unsigned char c) { return std::islower(c); });
}
bool xyz_atom(std::string_view line) {
    const auto tokens = tokens_of(line);
    return tokens.size() >= 4 && species(tokens[0]) && finite_number(tokens[1]) &&
           finite_number(tokens[2]) && finite_number(tokens[3]);
}
bool n2p2_atom(std::string_view line) {
    const auto tokens = tokens_of(line);
    if (tokens.size() != 10 || tokens[0] != "atom" || !species(tokens[4])) return false;
    for (std::size_t i = 1; i < tokens.size(); ++i)
        if (i != 4 && !finite_number(tokens[i])) return false;
    return true;
}
bool modeling_root(std::string_view text) {
    text = trim(text);
    // XML signature must be the root, beyond optional declarations/comments.
    while (text.starts_with("<?") || text.starts_with("<!--")) {
        const bool comment = text.starts_with("<!--");
        const auto terminator = comment ? "-->" : "?>";
        const auto end = text.find(terminator);
        if (end == std::string_view::npos) return false;
        text = trim(text.substr(end + (comment ? 3 : 2)));
    }
    return text.starts_with("<modeling") && text.size() > 9 &&
           (text[9] == '>' || std::isspace(static_cast<unsigned char>(text[9])));
}
void validate_limit(std::size_t limit) {
    if (limit == 0 || limit > MAX_PREFIX)
        throw std::invalid_argument("Detection prefix limit must be between 1 and 65536 bytes");
}
class ReplayBuffer final : public std::streambuf {
public:
    ReplayBuffer(std::unique_ptr<std::istream> input, std::string prefix)
        : input_(std::move(input)), prefix_(std::move(prefix)) {
        if (!prefix_.empty()) setg(prefix_.data(), prefix_.data(), prefix_.data() + prefix_.size());
    }
protected:
    int_type underflow() override {
        if (gptr() && gptr() < egptr()) return traits_type::to_int_type(*gptr());
        if (input_->eof()) return traits_type::eof();
        try { input_->read(buffer_, sizeof(buffer_)); }
        catch (const std::ios_base::failure &) {
            if (input_->bad() || !input_->eof()) throw;
        }
        const auto count = input_->gcount();
        if (input_->bad() || (input_->fail() && !input_->eof()))
            throw std::runtime_error("Detection replay input failure");
        if (count == 0) return traits_type::eof();
        setg(buffer_, buffer_, buffer_ + count);
        return traits_type::to_int_type(*gptr());
    }
private:
    std::unique_ptr<std::istream> input_;
    std::string prefix_;
    char buffer_[8192];
};
}

Detection detect_prefix(std::string_view prefix) {
    if (prefix.size() > MAX_PREFIX) prefix = prefix.substr(0, MAX_PREFIX);
    Detection result;
    const auto add = [&](const char *format) {
        if (std::find(result.candidates.begin(), result.candidates.end(), format) == result.candidates.end())
            result.candidates.emplace_back(format);
    };
    // Binary magic is at the physical start, without text whitespace handling.
    if (prefix.starts_with("GPMLBIN2")) { result.format = "bin"; result.candidates = {"bin"}; return result; }
    if (prefix.find('\0') != std::string_view::npos) return result;
    if (prefix.starts_with("\xef\xbb\xbf")) prefix.remove_prefix(3);
    const auto lines = lines_of(prefix);
    std::size_t first = 0;
    while (first < lines.size() && lines[first].empty()) ++first;
    if (first == lines.size()) return result;
    if (lines[first] == "BEGIN_CFG") add("cfg");
    if (modeling_root(prefix)) add("vasprun");

    // Grid-only XSF/BXSF belongs to its separate field interface.
    bool grid = false, xsf = false;
    for (const auto line : lines) {
        const auto tokens = tokens_of(line);
        if (tokens.empty() || tokens.front().starts_with("#")) continue;
        const auto &keyword = tokens.front();
        if (keyword.find("DATAGRID") != std::string::npos || keyword.find("BANDGRID") != std::string::npos)
            grid = true;
        if ((keyword == "PRIMCOORD" || keyword == "ATOMS") && tokens.size() <= 2) xsf = true;
    }
    if (xsf && !grid) add("xsf");
    if (lines[first] == "begin")
        for (std::size_t i = first + 1; i < lines.size(); ++i)
            if (n2p2_atom(lines[i])) { add("n2p2"); break; }

    std::size_t count = 0;
    const auto count_line = lines[first];
    const auto parsed = std::from_chars(count_line.data(), count_line.data() + count_line.size(), count);
    if (parsed.ec == std::errc{} && parsed.ptr == count_line.data() + count_line.size() &&
        count > 0 && first + 2 < lines.size()) {
        // A CP2K trajectory's distinctive comment needs its explicit codec.
        const auto comment = lines[first + 1];
        const bool cp2k = comment.find("i =") != std::string_view::npos &&
            (comment.find("time =") != std::string_view::npos || comment.find("E =") != std::string_view::npos);
        if (!cp2k && xyz_atom(lines[first + 2])) add("extxyz");
    }
    if (result.candidates.size() == 1) result.format = result.candidates.front();
    return result;
}

ProbedInput probe(std::unique_ptr<std::istream> input, std::size_t limit) {
    validate_limit(limit);
    if (!input || !input->good()) throw std::invalid_argument("Detection requires a readable input stream");
    std::string prefix(limit, '\0');
    try { input->read(prefix.data(), static_cast<std::streamsize>(limit)); }
    catch (const std::ios_base::failure &) {
        if (input->bad() || !input->eof()) throw;
    }
    prefix.resize(static_cast<std::size_t>(input->gcount()));
    if (input->bad() || (input->fail() && !input->eof()))
        throw std::runtime_error("Detection prefix input failure");
    auto detection = detect_prefix(prefix);
    input->clear();
    // Replay reads should report progressive codec failures as exceptions.
    auto replay = std::make_unique<OwnedIStream>(std::make_unique<ReplayBuffer>(std::move(input), std::move(prefix)));
    replay->exceptions(std::ios::badbit);
    return {std::move(replay), std::move(detection)};
}

Detection detect_file(const std::string &path, std::size_t limit) {
    validate_limit(limit);
    auto input = open_seq_ifile(path);
    if (!input) throw std::runtime_error("Cannot open file for format detection: " + path);
    return probe(std::move(input), limit).detection;
}
}
