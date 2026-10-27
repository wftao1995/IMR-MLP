#include "xsf_fields.h"
#include "utils/compressed_stream.h"

#include <charconv>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace {
[[noreturn]] void invalid(const std::string& message) {
    throw std::runtime_error("XSF DATAGRID: " + message);
}
bool starts(const std::string& text, const std::string& prefix) {
    return text.starts_with(prefix);
}
void validate_identifier(const std::string& name) {
    if (name.empty() || name.front() == '#' || name.find_first_of(" \t\r\n\v\f") != std::string::npos)
        invalid("identifier must be a nonempty single word, not a comment");
}
std::string trim(const std::string& text) {
    auto first = text.find_first_not_of(" \t\r\n\v\f");
    if (first == std::string::npos) return {};
    auto last = text.find_last_not_of(" \t\r\n\v\f");
    return text.substr(first, last - first + 1);
}
// A line-aware tokenizer preserves embedded # in identifiers, accepts arbitrary
// numeric whitespace, and recognizes only whole-line comments.
class Tokens {
public:
    explicit Tokens(std::istream& input) : input_(input) { line_.imbue(std::locale::classic()); }
    bool next(std::string& value) {
        while (!(line_ >> value)) {
            std::string text;
            if (!std::getline(input_, text)) {
                if (input_.bad() || (!input_.eof() && input_.fail())) invalid("input read failed");
                return false;
            }
            text = trim(text);
            line_.clear();
            line_.str(text.empty() || text.front() == '#' ? "" : text);
        }
        return true;
    }
    std::string required() {
        std::string value;
        if (!next(value)) invalid("truncated section");
        return value;
    }
    void end_line() {
        std::string extra;
        if (line_ >> extra) invalid("unexpected token after section marker: " + extra);
    }
private:
    std::istream& input_;
    std::istringstream line_;
};
std::size_t positive_size(const std::string& value) {
    std::size_t n = 0;
    auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), n);
    if (error != std::errc{} || end != value.data() + value.size() || n == 0)
        invalid("shape must contain positive integers");
    return n;
}
double number(const std::string& value) {
    std::istringstream text(value);
    text.imbue(std::locale::classic());
    double x;
    char extra;
    if (!(text >> x) || (text >> extra) || !std::isfinite(x)) invalid("expected finite number, got " + value);
    return x;
}
std::size_t c_index(const XsfGridLayout& layout, std::size_t i, std::size_t j, std::size_t k) {
    return layout.dimension() == 2 ? i * layout.shape[1] + j : (i * layout.shape[1] + j) * layout.shape[2] + k;
}
XsfDataGridBlock block(Tokens& tokens, std::size_t dimension) {
    XsfDataGridBlock result;
    result.identifier = tokens.required();
    validate_identifier(result.identifier);
    tokens.end_line();
    const auto suffix = std::to_string(dimension) + "D";
    const auto begin = "BEGIN_DATAGRID_" + suffix + "_";
    std::string token;
    while ((token = tokens.required()) != "END_BLOCK_DATAGRID_" + suffix) {
        if (!starts(token, begin)) invalid("expected " + begin + "identifier, got " + token);
        XsfScalarField field;
        field.identifier = token.substr(begin.size());
        validate_identifier(field.identifier);
        tokens.end_line();
        XsfGridLayout layout;
        for (std::size_t a = 0; a < dimension; ++a) layout.shape.push_back(positive_size(tokens.required()));
        // Check size before reading or allocating any values.
        const auto value_count = layout.size();
        for (auto& x : layout.origin) x = number(tokens.required());
        layout.spanning_vectors.resize(dimension);
        for (auto& v : layout.spanning_vectors) for (auto& x : v) x = number(tokens.required());
        layout.validate();
        if (result.fields.empty()) result.layout = layout;
        else if (layout != result.layout) invalid("all fields in a block must share the same layout");
        // Grow only as values arrive: malformed gigantic shapes fail at EOF
        // without eagerly allocating a huge tensor.
        const auto nx = layout.shape[0], ny = layout.shape[1];
        const auto nz = dimension == 3 ? layout.shape[2] : 1;
        std::vector<double> file_values;
        for (std::size_t n = 0; n < value_count; ++n) file_values.push_back(number(tokens.required()));
        if (tokens.required() != "END_DATAGRID_" + suffix) invalid("wrong DATAGRID closing marker or extra values");
        tokens.end_line();
        field.values.resize(file_values.size());
        std::size_t n = 0;
        for (std::size_t k = 0; k < nz; ++k)
            for (std::size_t j = 0; j < ny; ++j)
                for (std::size_t i = 0; i < nx; ++i)
                    field.values[c_index(layout, i, j, k)] = file_values[n++];
        result.fields.push_back(std::move(field));
    }
    tokens.end_line();
    result.validate();
    return result;
}
void validate_blocks(const std::vector<XsfDataGridBlock>& blocks) {
    if (blocks.empty()) invalid("at least one data block is required");
    for (const auto& b : blocks) b.validate();
}
}

std::size_t XsfGridLayout::size() const {
    if (shape.size() != 2 && shape.size() != 3) invalid("dimension must be 2 or 3");
    std::size_t n = 1;
    for (auto x : shape) {
        if (!x || x > std::numeric_limits<std::size_t>::max() / n) invalid("invalid or overflowing shape");
        n *= x;
    }
    if (n > std::vector<double>().max_size()) invalid("shape exceeds value storage limit");
    return n;
}
void XsfGridLayout::validate() const {
    size();
    if (spanning_vectors.size() != dimension()) invalid("one spanning vector is required per axis");
    for (auto x : origin) if (!std::isfinite(x)) invalid("origin must be finite");
    for (const auto& v : spanning_vectors) for (auto x : v) if (!std::isfinite(x)) invalid("spanning vectors must be finite");
}
std::array<double, 3> XsfGridLayout::coordinate(const std::vector<std::size_t>& indices) const {
    validate();
    if (indices.size() != dimension()) invalid("coordinate index dimension mismatch");
    auto point = origin;
    for (std::size_t a = 0; a < dimension(); ++a) {
        if (indices[a] >= shape[a]) invalid("coordinate index out of range");
        double fraction = shape[a] == 1 ? 0. : static_cast<double>(indices[a]) / (shape[a] - 1);
        for (std::size_t d = 0; d < 3; ++d) point[d] += fraction * spanning_vectors[a][d];
    }
    return point;
}
void XsfDataGridBlock::validate() const {
    validate_identifier(this->identifier);
    layout.validate();
    if (fields.empty()) invalid("block must contain at least one scalar field");
    for (const auto& field : fields) {
        ::validate_identifier(field.identifier);
        if (field.values.size() != layout.size()) invalid("value count does not match shape");
        for (auto x : field.values) if (!std::isfinite(x)) invalid("field values must be finite");
    }
}
std::vector<XsfDataGridBlock> read_xsf_fields(std::istream& input) {
    std::vector<XsfDataGridBlock> result;
    // Geometry is intentionally skipped as lines, not parsed as Atoms. Only a
    // static prefix is accepted; animation cannot be silently reduced to grids.
    std::string line;
    bool geometry_prefix = false;
    while (std::getline(input, line)) {
        line = trim(line);
        if (line.empty() || line.front() == '#') continue;
        std::istringstream header(line);
        std::string key, extra;
        header >> key;
        if (key == "ANIMSTEPS") invalid("AXSF time-dependent fields are unsupported");
        if (key == "ATOMS" || key == "PRIMCOORD" || key == "CONVCOORD" || key == "PRIMVEC" || key == "CONVVEC") {
            if (header >> extra) invalid("indexed geometry/AXSF fields are unsupported");
            geometry_prefix = true;
        }
        if (key == "BEGIN_BLOCK_DATAGRID_2D" || key == "BEGIN_BLOCK_DATAGRID_3D") {
            if (header >> extra) invalid("unexpected token after block marker");
            Tokens tokens(input);
            result.push_back(block(tokens, key == "BEGIN_BLOCK_DATAGRID_2D" ? 2 : 3));
        } else if (key.find("DATAGRID") != std::string::npos || key.find("BANDGRID") != std::string::npos || key == "BEGIN_INFO") {
            invalid("unsupported or misplaced section: " + key);
        } else if (!result.empty()) invalid("only data blocks may follow the first block");
        else if (key == "MOLECULE" || key == "POLYMER" || key == "SLAB" || key == "CRYSTAL") {
            if (header >> extra) invalid("unexpected token after geometry keyword");
            geometry_prefix = true;
        } else if (!geometry_prefix) invalid("expected static geometry or a DATAGRID block");
    }
    if (input.bad() || (!input.eof() && input.fail())) invalid("input read failed");
    if (result.empty()) invalid("no DATAGRID blocks found");
    return result;
}
std::vector<XsfDataGridBlock> read_xsf_fields(const std::string& path) {
    auto input = open_seq_ifile(path);
    if (!input) invalid("cannot open input " + path);
    return read_xsf_fields(*input);
}
void write_xsf_fields(std::ostream& output, const std::vector<XsfDataGridBlock>& blocks) {
    validate_blocks(blocks);
    const auto old_precision = output.precision();
    const auto old_flags = output.flags();
    const auto old_locale = output.getloc();
    output.imbue(std::locale::classic());
    output << std::defaultfloat << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (const auto& b : blocks) {
        const auto dimension = b.layout.dimension();
        const auto suffix = std::to_string(dimension) + "D";
        output << "BEGIN_BLOCK_DATAGRID_" << suffix << '\n' << b.identifier << '\n';
        for (const auto& f : b.fields) {
            output << "BEGIN_DATAGRID_" << suffix << '_' << f.identifier << '\n';
            for (auto n : b.layout.shape) output << n << ' ';
            output << '\n';
            for (auto x : b.layout.origin) output << x << ' ';
            output << '\n';
            for (const auto& v : b.layout.spanning_vectors) {
                for (auto x : v) output << x << ' ';
                output << '\n';
            }
            std::size_t count = 0;
            const auto nz = dimension == 3 ? b.layout.shape[2] : 1;
            for (std::size_t k = 0; k < nz; ++k)
                for (std::size_t j = 0; j < b.layout.shape[1]; ++j)
                    for (std::size_t i = 0; i < b.layout.shape[0]; ++i) {
                        output << f.values[c_index(b.layout, i, j, k)];
                        output << (++count % 6 == 0 ? '\n' : ' ');
                    }
            if (count % 6) output << '\n';
            output << "END_DATAGRID_" << suffix << '\n';
        }
        output << "END_BLOCK_DATAGRID_" << suffix << '\n';
    }
    output.precision(old_precision);
    output.flags(old_flags);
    output.imbue(old_locale);
    if (!output) invalid("output write failed");
}
void write_xsf_fields(const std::string& path, const std::vector<XsfDataGridBlock>& blocks) {
    validate_blocks(blocks); // Do not truncate an existing path for invalid objects.
    auto output = open_ofile(path);
    if (!output) invalid("cannot open output " + path);
    write_xsf_fields(*output, blocks);
    if (!close_ofile(output)) invalid("output close failed: " + path);
}
