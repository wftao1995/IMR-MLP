// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpio.h"
#include "hvp_io.h"
#include "gpio_binary.h"
#include "gpio_tokenizer.h"
#include "tools/chem.h"
#include "tools/procstr.h"
#include "tools/tools.h"
#include <cctype>
#include <cmath>
#include <limits>
#include <set>
#include <type_traits>
#include <sstream>
#include <stdexcept>

namespace {

// Checked parsing shared only by XSF, GRO, N2P2 and CP2KXYZ. Keep other codecs'
// compatibility rules independent of this restricted text subset.
bool structure_text_line(std::istream &input, std::string &line) {
    line.clear();
    try {
        return static_cast<bool>(std::getline(input, line));
    } catch (const std::ios_base::failure &) {
        if (input.bad() || !input.eof()) throw;
        return !line.empty(); // a complete last line need not end in a newline
    }
}

template<class T> bool structure_text_number(std::string_view token, T &value) {
    if (token.empty()) return false;
    if (token.front() == '+') {
        token.remove_prefix(1);
        if (token.empty() || token.front() == '+' || token.front() == '-') return false;
    }
    const auto result = fast_float::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) return false;
    if constexpr (std::is_floating_point_v<T>) return std::isfinite(value);
    return true;
}

bool structure_text_vector(GPIO_LineTokenizer<> &tokens, Eigen::RowVector3d &value,
                           double scale = 1) {
    for (int i = 0; i < 3; ++i) {
        if (!structure_text_number(tokens.next_token(), value[i])) return false;
        value[i] *= scale;
        if (!std::isfinite(value[i])) return false;
    }
    return true;
}

// strip CIF quotes from a token: 'Fe' / "Fe" -> Fe
std::string_view strip_cif_quotes(std::string_view tok)
{
    if (tok.size() >= 2 &&
        ((tok.front() == '\'' && tok.back() == '\'') ||
         (tok.front() == '"' && tok.back() == '"'))) {
        return tok.substr(1, tok.size() - 2);
    }
    return tok;
}

// resolve a CIF element token (possibly quoted, 'Fe2+', 'fe', 'D', ...) to
// the internal species index, or -1 when it cannot be resolved
int cif_element_index(std::string_view tok, const std::map<std::string, int> &mapnameid)
{
    auto exact = [&](std::string_view s) {
        auto it = mapnameid.find(std::string(s));
        return it == mapnameid.end() ? -1 : it->second;
    };
    int id = exact(tok);
    if (id >= 0) return id;
    // strip trailing charge/digit suffixes: Fe2+ -> Fe, Fe1 -> Fe
    std::string t(tok);
    while (!t.empty() && (t.back() == '+' || t.back() == '-' ||
                          std::isdigit(static_cast<unsigned char>(t.back())))) {
        t.pop_back();
    }
    id = exact(t);
    if (id >= 0) return id;
    // labels can carry digits in the middle too: Fe1a -> Fe
    auto pos = t.find_first_of("0123456789");
    if (pos != std::string::npos) {
        id = exact(std::string_view(t).substr(0, pos));
        if (id >= 0) return id;
    }
    // deuterium / tritium are not in the periodic table
    if (t == "D" || t == "T") return exact("H");
    // case-insensitive fallback (tolerate sloppy tools)
    for (auto &kv : mapnameid) {
        if (kv.first.size() != t.size()) continue;
        bool match = true;
        for (size_t i = 0; i < t.size(); ++i) {
            if (std::tolower(static_cast<unsigned char>(t[i])) !=
                std::tolower(static_cast<unsigned char>(kv.first[i]))) {
                match = false;
                break;
            }
        }
        if (match) return kv.second;
    }
    return -1;
}

// strip '#' comments (outside quotes) from a CIF line
void strip_cif_comment(std::string &line)
{
    bool in_q = false;
    char qc = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        char ch = line[i];
        if (in_q) {
            if (ch == qc) in_q = false;
        } else if (ch == '\'' || ch == '"') {
            in_q = true;
            qc = ch;
        } else if (ch == '#') {
            line.resize(i);
            break;
        }
    }
}

} // namespace

const char *exact_mol_name(const char *inp)
{
    for (int i = 0; inp[i] != '\0'; ++i)
    {
        if (inp[i] < '0' || inp[i] > '9')
        {
            return inp + i;
        }
    }
    return inp;
}
bool GPIO::read_gro_data(std::istream &is, Atoms &output)
{
    return load_next_from_gro(is, output);
}

bool GPIO::load_next_from_gro(std::istream &is, Atoms &output, bool *clean_eof)
{
    output.clear();
    if (clean_eof) *clean_eof = false;
    if (absolute_index_read) {
        std::cerr << "Error: GRO resolves element tokens; absolute_index_read is unsupported\n";
        return false;
    }
    if (is.bad() || is.fail()) return false;
    auto at_clean_eof = [&]() {
        return is.eof() && !is.bad() && !(is.rdstate() & std::ios::failbit);
    };
    // A blank title is still a started frame, not an ignorable EOF suffix.
    try {
        if (is.eof() || is.peek() == std::char_traits<char>::eof()) {
            if (clean_eof) *clean_eof = at_clean_eof();
            return false;
        }
    } catch (const std::ios_base::failure &) {
        if (!at_clean_eof()) throw;
        if (clean_eof) *clean_eof = true;
        return false;
    }
    Atoms atom;
    atom.clear();
    std::string line;
    if (!structure_text_line(is, line) || !structure_text_line(is, line)) return false;
    GPIO_LineTokenizer counts(line);
    int count;
    if (!structure_text_number(counts.next_token(), count) || count <= 0 || count > gpio_binary::MAX_ATOMS ||
        !counts.next_token().empty()) return false;
    std::vector<Eigen::Vector3d> velocities;
    std::string prev_residue_number, prev_residue_name, prevmolname;
    int molid = -1, molsize = 0, moltype = 0;
    for (int i = 0; i < count; ++i) {
        if (!structure_text_line(is, line)) return false;
        auto field = [&](size_t offset, size_t width) {
            std::string_view value(line.data() + offset, width);
            const auto first = value.find_first_not_of(" \t");
            if (first == std::string_view::npos) return std::string_view{};
            const auto last = value.find_last_not_of(" \t");
            return value.substr(first, last - first + 1);
        };
        std::string residue_number, residue_name, species;
        Eigen::RowVector3d position;
        int residue_id = 0, serial = 0;
        const bool fixed = line.size() >= 20 &&
            structure_text_number(field(0, 5), residue_id) && residue_id >= 0 &&
            !field(5, 5).empty() && field(5, 5).find_first_of(" \t") == std::string_view::npos &&
            !field(10, 5).empty() && field(10, 5).find_first_of(" \t") == std::string_view::npos &&
            structure_text_number(field(15, 5), serial) && serial >= 0;
        if (fixed) {
            // Standard GRO: i5, a5, a5, i5, 3f8.3. Once metadata matches,
            // malformed fixed coordinates must not fall back to token parsing.
            residue_number = std::to_string(residue_id);
            residue_name = field(5, 5);
            species = field(10, 5);
            if (line.size() < 44) return false;
            for (int j = 0; j < 3; ++j) {
                if (!structure_text_number(field(20 + 8 * j, 8), position[j])) return false;
                position[j] *= 10;
                if (!std::isfinite(position[j])) return false;
            }
            // Only complete standard 3f8.4 velocity columns are recognized.
            // Other legacy suffixes remain geometry-only, without partial labels.
            if (line.size() >= 68 && line.find_first_not_of(" \t\r", 68) == std::string::npos) {
                Eigen::Vector3d velocity;
                bool complete = true;
                for (int j = 0; j < 3; ++j) {
                    if (!structure_text_number(field(44 + 8 * j, 8), velocity[j]) ||
                        !std::isfinite(velocity[j] * 10)) {
                        complete = false;
                        break;
                    }
                    velocity[j] *= 10; // nm/ps -> Angstrom/ps
                }
                if (complete) velocities.push_back(velocity);
            }
        } else {
            GPIO_LineTokenizer tokens(line);
            residue_name = tokens.next_token();
            species = tokens.next_token();
            if (residue_name.empty() || species.empty() ||
                !structure_text_number(tokens.next_token(), serial) || serial < 0 ||
                !structure_text_vector(tokens, position, 10)) return false;
        }
        const auto type = mapnameid.find(species);
        if (type == mapnameid.end()) return false;
        // Atom names resolve exactly through the existing map; no element
        // inference is applied.
        const std::string molecule(exact_mol_name(residue_name.c_str()));
        if (residue_number == prev_residue_number && residue_name == prev_residue_name) {
            ++molsize;
        } else {
            if (molsize > 0) atom.molsize_.push_back(molsize);
            ++molid;
            molsize = 1;
            prev_residue_number = residue_number;
            prev_residue_name = residue_name;
        }
        if (molecule != prevmolname) {
            atom.mol_types_.push_back(moltype++);
            prevmolname = molecule;
        }
        atom.molid_.push_back(molid);
        atom.molnum_ = molid + 1;
        atom.chemmap_.push_back(type->second);
        atom.pos_.push_back(position);
    }
    atom.molsize_.push_back(molsize);
    if (!structure_text_line(is, line)) return false;
    GPIO_LineTokenizer tokens(line);
    std::array<double, 9> box;
    int fields = 0;
    for (auto token = tokens.next_token(); !token.empty(); token = tokens.next_token()) {
        if (fields == 9 || !structure_text_number(token, box[fields])) return false;
        box[fields] *= 10;
        if (!std::isfinite(box[fields])) return false;
        ++fields;
    }
    if (fields != 3 && fields != 9) return false;
    atom.cell_ = Eigen::Matrix3d::Zero();
    atom.cell_(0, 0) = box[0];
    atom.cell_(1, 1) = box[1];
    atom.cell_(2, 2) = box[2];
    if (fields == 9) {
        atom.cell_(0, 1) = box[3];
        atom.cell_(0, 2) = box[4];
        atom.cell_(1, 0) = box[5];
        atom.cell_(1, 2) = box[6];
        atom.cell_(2, 0) = box[7];
        atom.cell_(2, 1) = box[8];
    }
    if (velocities.size() == atom.pos_.size()) atom.velocities_ = std::move(velocities);
    output = std::move(atom);
    return true;
}

/***
 * A functional, tolerant cif reader.
 *
 * Handles the common parts of the CIF grammar:
 *  - multiple data_ blocks (each read as one structure)
 *  - quoted values ('Fe', "5.0"), '#' comments and ';' text blocks
 *  - cell parameters _cell_length_{a,b,c} / _cell_angle_{alpha,beta,gamma}
 *    as plain tags, or inside a loop_ section (Materials Studio style);
 *    missing angles default to 90 degrees
 *  - atom loops with _atom_site_fract_{x,y,z} (preferred) or
 *    _atom_site_Cartn_{x,y,z}, element from _atom_site_type_symbol
 *    (preferred) or _atom_site_label, optional _atom_site_occupancy and
 *    _atom_site_U_iso_or_equiv
 *  - partial occupancy is stored in local_attribs_d_["occupancy"] and
 *    reported on stderr; unresolved element symbols are skipped with a
 *    warning (D/T -> H, Fe2+ -> Fe, case-insensitive fallback)
 *  - symmetry loops are ignored (no space-group expansion)
 * ***/
bool GPIO::read_cif(std::istream &is, Atoms &atom)
{
    // cell parameters; angles in degrees, default to 90
    double a = 0, b = 0, c = 0;
    double alpha = 90.0, beta = 90.0, gamma = 90.0;

    enum class LoopState { None, Header, Data };
    LoopState loop = LoopState::None;
    bool cell_loop = false;   // current loop carries cell parameters (MS style)
    bool atom_loop = false;   // current loop is an atom-site loop
    bool frac = false;        // _atom_site_fract_* columns present
    bool cart = false;        // _atom_site_Cartn_* columns present
    bool have_type_symbol = false;
    bool have_label = false;
    bool have_occ = false;
    bool have_u_iso = false;
    bool cart_coords = false;
    bool cart_decided = false;
    bool atoms_parsed = false;
    int loop_col = 0;   // next column index in the current loop header
    std::map<std::string, int> readpos;   // atom-site column of interest
    std::map<std::string, int> cellpos;   // cell-parameter column (loop style)

    // per-block atom data (fractional coordinates until the cell is known)
    std::vector<int> types;
    std::vector<Eigen::RowVector3d> fracpos;
    std::vector<double> occs, uisos;
    bool any_partial_occ = false;
    bool any_u_iso = false;

    auto read_noncomment_line = [&]() -> bool {
        while (std::getline(is, line_buffer)) {
            strip_cif_comment(line_buffer);
            size_t p = line_buffer.find_first_not_of(" \t\r");
            if (p == std::string::npos) continue;             // blank
            if (line_buffer[p] == '#') continue;              // comment
            if (line_buffer[p] == ';') {                      // ';' text block
                while (std::getline(is, line_buffer)) {
                    size_t q = line_buffer.find_first_not_of(" \t\r");
                    if (q != std::string::npos && line_buffer[q] == ';') break;
                }
                continue;
            }
            return true;
        }
        return false;
    };

    auto parse_double = [](std::string_view s, double &val) -> bool {
        std::string_view v = strip_cif_quotes(s);
        if (v == "?" || v == "." || v.empty()) return false;
        auto res = fast_float::from_chars(v.data(), v.data() + v.size(), val);
        return res.ec == std::errc{};
    };

    while (read_noncomment_line()) {
        GPIO_LineTokenizer toks{line_buffer};
        auto tok = toks.next_token();
        if (tok.empty()) continue;

        if (tok.starts_with("data_")) {
            // this block is finished; a later call handles the next block
            if (atoms_parsed) break;
            loop = LoopState::None;
            continue;
        }

        if (tok == "loop_") {
            loop = LoopState::Header;
            loop_col = 0;
            cell_loop = false;
            atom_loop = false;
            frac = false;
            cart = false;
            have_type_symbol = false;
            have_label = false;
            have_occ = false;
            have_u_iso = false;
            readpos.clear();
            cellpos.clear();
            continue;
        }

        if (tok[0] == '_') {
            if (loop == LoopState::Header) {
                // loop header: one or more tags on this line
                do {
                    const int col = loop_col++;
                    if (tok == "_atom_site_fract_x") {
                        frac = true;
                        atom_loop = true;
                        readpos["x"] = col;
                    } else if (tok == "_atom_site_fract_y") {
                        frac = true;
                        atom_loop = true;
                        readpos["y"] = col;
                    } else if (tok == "_atom_site_fract_z") {
                        frac = true;
                        atom_loop = true;
                        readpos["z"] = col;
                    } else if (tok == "_atom_site_Cartn_x") {
                        cart = true;
                        atom_loop = true;
                        if (!frac) readpos["x"] = col;
                    } else if (tok == "_atom_site_Cartn_y") {
                        cart = true;
                        atom_loop = true;
                        if (!frac) readpos["y"] = col;
                    } else if (tok == "_atom_site_Cartn_z") {
                        cart = true;
                        atom_loop = true;
                        if (!frac) readpos["z"] = col;
                    } else if (tok == "_atom_site_type_symbol") {
                        readpos["type"] = col;
                        have_type_symbol = true;
                    } else if (tok == "_atom_site_label") {
                        readpos["label"] = col;
                        have_label = true;
                    } else if (tok == "_atom_site_occupancy") {
                        readpos["occ"] = col;
                        have_occ = true;
                    } else if (tok == "_atom_site_U_iso_or_equiv") {
                        readpos["u_iso"] = col;
                        have_u_iso = true;
                    } else if (tok == "_cell_length_a") {
                        cellpos["a"] = col;
                        cell_loop = true;
                    } else if (tok == "_cell_length_b") {
                        cellpos["b"] = col;
                        cell_loop = true;
                    } else if (tok == "_cell_length_c") {
                        cellpos["c"] = col;
                        cell_loop = true;
                    } else if (tok == "_cell_angle_alpha") {
                        cellpos["alpha"] = col;
                        cell_loop = true;
                    } else if (tok == "_cell_angle_beta") {
                        cellpos["beta"] = col;
                        cell_loop = true;
                    } else if (tok == "_cell_angle_gamma") {
                        cellpos["gamma"] = col;
                        cell_loop = true;
                    }
                    // other loop tags (symmetry ops, ...) are ignored
                    tok = toks.next_token();
                } while (!tok.empty());
                continue;
            }
            // key-value tag outside a loop, or ending loop data
            loop = LoopState::None;
            double v = 0;
            if (tok == "_cell_length_a") {
                if (parse_double(toks.next_token(), v)) a = v;
            } else if (tok == "_cell_length_b") {
                if (parse_double(toks.next_token(), v)) b = v;
            } else if (tok == "_cell_length_c") {
                if (parse_double(toks.next_token(), v)) c = v;
            } else if (tok == "_cell_angle_alpha") {
                if (parse_double(toks.next_token(), v)) alpha = v;
            } else if (tok == "_cell_angle_beta") {
                if (parse_double(toks.next_token(), v)) beta = v;
            } else if (tok == "_cell_angle_gamma") {
                if (parse_double(toks.next_token(), v)) gamma = v;
            }
            continue;
        }

        // data row of the current loop (or stray data outside a loop)
        if (loop == LoopState::Header) {
            loop = LoopState::Data;
            if (atom_loop && !cart_decided) {
                cart_coords = cart && !frac;
                cart_decided = true;
            }
        }
        if (loop != LoopState::Data) continue;

        if (atom_loop) {
            std::vector<std::string> splstr;
            str_split(line_buffer, splstr);
            const int ncols = static_cast<int>(splstr.size());
            auto col_ok = [&](const char *k) {
                auto it = readpos.find(k);
                return it != readpos.end() && it->second < ncols;
            };
            if (!(col_ok("x") && col_ok("y") && col_ok("z"))) continue;
            std::string symbol_tok;
            if (have_type_symbol && col_ok("type")) {
                symbol_tok = splstr[readpos["type"]];
            } else if (have_label && col_ok("label")) {
                symbol_tok = splstr[readpos["label"]];
            }
            int type = cif_element_index(strip_cif_quotes(symbol_tok), mapnameid);
            if (type < 0) {
                std::cerr << "Warning: cif: unknown element \"" << symbol_tok
                          << "\", skipping atom" << std::endl;
                continue;
            }
            double px, py, pz;
            try {
                px = std::stod(std::string(strip_cif_quotes(splstr[readpos["x"]])));
                py = std::stod(std::string(strip_cif_quotes(splstr[readpos["y"]])));
                pz = std::stod(std::string(strip_cif_quotes(splstr[readpos["z"]])));
            } catch (...) {
                std::cerr << "Warning: cif: invalid coordinate, skipping atom" << std::endl;
                continue;
            }
            types.push_back(type);
            fracpos.emplace_back(px, py, pz);
            atoms_parsed = true;
            double occ = 1.0;
            if (have_occ && col_ok("occ")) {
                if (parse_double(splstr[readpos["occ"]], occ)) {
                    if (std::abs(occ - 1.0) > 1e-8) {
                        any_partial_occ = true;
                        std::cerr << "Warning: partial occupation " << occ
                                  << " stored in 'occupancy', not expanded" << std::endl;
                    }
                }
            }
            occs.push_back(occ);
            double u = 0.0;
            if (have_u_iso && col_ok("u_iso")) {
                if (parse_double(splstr[readpos["u_iso"]], u)) {
                    any_u_iso = true;
                }
            }
            uisos.push_back(u);
        } else if (cell_loop) {
            // cell parameters given as loop data rows (Materials Studio)
            std::vector<std::string> splstr;
            str_split(line_buffer, splstr);
            const int ncols = static_cast<int>(splstr.size());
            auto col_ok = [&](const char *k) {
                auto it = cellpos.find(k);
                return it != cellpos.end() && it->second < ncols;
            };
            double v = 0;
            if (col_ok("a") && parse_double(splstr[cellpos["a"]], v)) a = v;
            if (col_ok("b") && parse_double(splstr[cellpos["b"]], v)) b = v;
            if (col_ok("c") && parse_double(splstr[cellpos["c"]], v)) c = v;
            if (col_ok("alpha") && parse_double(splstr[cellpos["alpha"]], v)) alpha = v;
            if (col_ok("beta") && parse_double(splstr[cellpos["beta"]], v)) beta = v;
            if (col_ok("gamma") && parse_double(splstr[cellpos["gamma"]], v)) gamma = v;
        }
        // symmetry and other loops: rows are ignored
    }

    // build the cell; angles are in degrees, default 90
    if (a <= 0 || b <= 0 || c <= 0) {
        std::cerr << "Error: cif: invalid or missing cell parameters" << std::endl;
        return false;
    }
    const double ar = alpha * M_PI / 180.0;
    const double br = beta * M_PI / 180.0;
    const double gr = gamma * M_PI / 180.0;
    const double sg = sin(gr);
    if (std::abs(sg) < 1e-12) {
        std::cerr << "Error: cif: degenerate cell" << std::endl;
        return false;
    }
    atom.cell_ << a, 0, 0,
                  b * cos(gr), b * sg, 0,
                  c * cos(br), c * (cos(ar) - cos(br) * cos(gr)) / sg, 0;
    double cz_sq = c * c - atom.cell_(2, 0) * atom.cell_(2, 0) - atom.cell_(2, 1) * atom.cell_(2, 1);
    atom.cell_(2, 2) = std::sqrt(std::max(0.0, cz_sq));

    atom.chemmap_.reserve(types.size());
    atom.pos_.reserve(types.size());
    for (size_t i = 0; i < types.size(); ++i) {
        atom.chemmap_.push_back(types[i]);
        if (cart_coords)
            atom.pos_.push_back(fracpos[i]);
        else
            atom.pos_.push_back(fracpos[i] * atom.cell_);
    }
    atom.pbc_ = {true, true, true};
    if (any_partial_occ) {
        auto &o = atom.local_attribs_d_["occupancy"];
        o.resize(occs.size());
        for (size_t i = 0; i < occs.size(); ++i) o[i] = occs[i];
    }
    if (any_u_iso) {
        auto &u = atom.local_attribs_d_["u_iso"];
        u.resize(uisos.size());
        for (size_t i = 0; i < uisos.size(); ++i) u[i] = uisos[i];
    }
    return atom.chemmap_.size() > 0;
}

bool GPIO::read_xsf(std::istream &is, Atoms &output) {
    xsf_read_ = {};
    return load_next_from_xsf(is, output);
}

bool GPIO::load_next_from_xsf(std::istream &is, Atoms &output, bool *clean_eof) {
    output.clear();
    if (clean_eof) *clean_eof = false;
    if (absolute_index_read) {
        std::cerr << "Error: XSF resolves element tokens; absolute_index_read is unsupported\n";
        return false;
    }
    auto &state = xsf_read_;
    if (is.bad() || (is.fail() && !state.at_eof)) return false;
    std::string line;
    auto content_line = [&]() {
        if (!state.pending.empty()) {
            line = std::move(state.pending);
            state.pending.clear();
            return true;
        }
        if (state.at_eof) return false;
        while (structure_text_line(is, line)) {
            line.resize(line.find('#') == std::string::npos ? line.size() : line.find('#'));
            GPIO_LineTokenizer tokens(line);
            if (!tokens.next_token().empty()) return true;
        }
        state.at_eof = is.eof() && !is.bad();
        return false;
    };
    auto section = [&](std::string_view name, bool indexed) {
        GPIO_LineTokenizer tokens(line);
        if (tokens.next_token() != name) return false;
        const auto index = tokens.next_token();
        int value;
        if (indexed) {
            if (!structure_text_number(index, value) || value != state.next_step) return false;
        } else if (!index.empty()) return false;
        return tokens.next_token().empty();
    };
    auto cell = [&](Eigen::Matrix3d &value) {
        for (int row = 0; row < 3; ++row) {
            if (!content_line()) return false;
            GPIO_LineTokenizer tokens(line);
            Eigen::RowVector3d vector;
            if (!structure_text_vector(tokens, vector) || !tokens.next_token().empty()) return false;
            value.row(row) = vector;
        }
        return true;
    };
    if (!state.initialized) {
        if (!content_line()) return false;
        GPIO_LineTokenizer first(line);
        if (first.next_token() == "ANIMSTEPS") {
            if (!structure_text_number(first.next_token(), state.expected_steps) || state.expected_steps <= 0 ||
                !first.next_token().empty() || !content_line()) return false;
            state.animated = true;
        }
        GPIO_LineTokenizer geometry(line);
        const auto kind = geometry.next_token();
        if (kind == "ATOMS") {
            state.molecular = true;
            state.pbc = {false, false, false};
            state.pending = line;
        } else {
            if (!geometry.next_token().empty()) return false;
            if (kind == "CRYSTAL") state.pbc = {true, true, true};
            else if (kind == "SLAB") state.pbc = {true, true, false};
            else if (kind == "POLYMER") state.pbc = {true, false, false};
            else return false;
        }
        state.initialized = true;
    }
    if (state.next_step > state.expected_steps) {
        if (content_line() || !state.at_eof) return false;
        if (clean_eof) *clean_eof = true;
        return false;
    }
    Atoms atom;
    atom.clear();
    atom.pbc_ = state.pbc;
    bool force_columns = false;
    auto atom_row = [&]() {
        if (atom.size() >= gpio_binary::MAX_ATOMS) return false;
        GPIO_LineTokenizer tokens(line);
        const auto species = tokens.next_token();
        if (species.empty()) return false;
        std::string symbol(species);
        if (std::isdigit(static_cast<unsigned char>(species.front())) ||
            species.front() == '+' || species.front() == '-') {
            int number;
            if (!structure_text_number(species, number) || number < 0 ||
                number >= static_cast<int>(std::size(Chemicals::chemical_symbols))) return false;
            symbol = Chemicals::chemical_symbols[number];
        }
        const auto type = mapnameid.find(symbol);
        if (type == mapnameid.end()) return false;
        Eigen::RowVector3d position, force;
        if (!structure_text_vector(tokens, position)) return false;
        const auto first_force = tokens.next_token();
        const bool have_forces = !first_force.empty();
        if (!atom.pos_.empty() && have_forces != force_columns) return false;
        if (have_forces) {
            if (!structure_text_number(first_force, force[0]) ||
                !structure_text_number(tokens.next_token(), force[1]) ||
                !structure_text_number(tokens.next_token(), force[2]) || !tokens.next_token().empty()) return false;
            force *= HARTREE_TO_EV; // XSF is Hartree/Angstrom.
            if (!force.allFinite()) return false;
        }
        atom.chemmap_.push_back(type->second);
        atom.pos_.push_back(position);
        if (have_forces) atom.forces_.push_back(force);
        force_columns = have_forces;
        return true;
    };
    if (state.molecular) {
        if (!content_line() || !section("ATOMS", state.animated)) return false;
        while (content_line()) {
            GPIO_LineTokenizer tokens(line);
            if (state.animated && tokens.next_token() == "ATOMS") {
                state.pending = line; // The next frame validates its own index.
                break;
            }
            if (!atom_row()) return false;
        }
        if (state.pending.empty() && !state.at_eof) return false;
    } else {
        if (!content_line()) return false;
        if (state.next_step == 1 || state.variable_cell) {
            if (state.next_step == 1) {
                GPIO_LineTokenizer vectors(line);
                if (vectors.next_token() != "PRIMVEC") return false;
                state.variable_cell = state.animated && !vectors.next_token().empty();
            }
            if (!section("PRIMVEC", state.variable_cell) || !cell(state.cell) || !content_line()) return false;
            GPIO_LineTokenizer possible_conventional(line);
            if (possible_conventional.next_token() == "CONVVEC") {
                Eigen::Matrix3d conventional;
                if (!section("CONVVEC", state.variable_cell) || !cell(conventional) || !content_line()) return false;
            }
        }
        if (!section("PRIMCOORD", state.animated) || !content_line()) return false;
        GPIO_LineTokenizer counts(line);
        int count, coordinate_sets;
        if (!structure_text_number(counts.next_token(), count) || count < 0 || count > gpio_binary::MAX_ATOMS ||
            !structure_text_number(counts.next_token(), coordinate_sets) || coordinate_sets != 1 ||
            !counts.next_token().empty()) return false;
        if (state.animated && state.expected_atoms >= 0 && count != state.expected_atoms) return false;
        for (int i = 0; i < count; ++i)
            if (!content_line() || !atom_row()) return false;
        atom.cell_ = state.cell;
    }
    if (state.animated) {
        if (state.expected_atoms >= 0 && atom.size() != state.expected_atoms) return false;
        state.expected_atoms = atom.size();
    }
    ++state.next_step;
    output = std::move(atom);
    return true;
}

// Standard periodic or molecular N2P2 input: Bohr / Hartree / Hartree per Bohr.
bool GPIO::load_next_from_n2p2_data(std::istream &is, Atoms &output, bool *clean_eof)
{
    output.clear();
    if (clean_eof) *clean_eof = false;
    if (absolute_index_read) {
        std::cerr << "Error: N2P2 resolves element tokens; absolute_index_read is unsupported\n";
        return false;
    }
    Atoms atom;
    atom.clear();
    std::string line;
    bool begun = false, metadata = false, have_charge = false;
    int lattice_rows = 0;
    while (true) {
        if (is.bad() || is.fail()) return false;
        auto at_clean_eof = [&]() {
            return is.eof() && !is.bad() && !(is.rdstate() & std::ios::failbit);
        };
        // Probe before getline so empty/blank inter-frame tails are clean EOF,
        // while an I/O failure or EOF after begin can never complete a frame.
        try {
            if (is.eof() || is.peek() == std::char_traits<char>::eof()) {
                if (clean_eof) *clean_eof = !begun && at_clean_eof();
                return false;
            }
        } catch (const std::ios_base::failure &) {
            if (!at_clean_eof()) throw;
            if (clean_eof) *clean_eof = !begun;
            return false;
        }
        if (!structure_text_line(is, line)) return false;
        GPIO_LineTokenizer tokens(line);
        const auto tag = tokens.next_token();
        if (tag.empty()) continue;
        if (!begun) {
            if (tag != "begin" || !tokens.next_token().empty() || ncfgs_ == INT_MAX) return false;
            begun = true;
            continue;
        }
        if (tag == "comment") continue;
        if (tag == "lattice") {
            if (lattice_rows == 3 || metadata || !atom.pos_.empty()) return false;
            Eigen::RowVector3d row;
            if (!structure_text_vector(tokens, row, BOHR_TO_A) || !tokens.next_token().empty()) return false;
            atom.cell_.row(lattice_rows++) = row;
        } else if (tag == "atom") {
            if ((lattice_rows != 0 && lattice_rows != 3) || metadata || atom.size() >= gpio_binary::MAX_ATOMS) return false;
            Eigen::RowVector3d position, force;
            if (!structure_text_vector(tokens, position, BOHR_TO_A)) return false;
            const auto type = mapnameid.find(std::string(tokens.next_token()));
            if (type == mapnameid.end()) return false;
            double charge, site_energy;
            if (!structure_text_number(tokens.next_token(), charge) ||
                !structure_text_number(tokens.next_token(), site_energy) ||
                !std::isfinite(site_energy * HARTREE_TO_EV) ||
                !structure_text_vector(tokens, force, HARTREE_TO_EV * A_TO_BOHR) ||
                !tokens.next_token().empty()) return false;
            atom.pos_.push_back(position);
            atom.forces_.push_back(force);
            atom.chemmap_.push_back(type->second);
            atom.charges_.push_back(charge);
            atom.site_energies_.push_back(site_energy * HARTREE_TO_EV);
        } else if (tag == "energy" || tag == "charge") {
            if ((lattice_rows != 0 && lattice_rows != 3) || (tag == "energy" ? atom.have_energy : have_charge)) return false;
            double value;
            if (!structure_text_number(tokens.next_token(), value) || !tokens.next_token().empty()) return false;
            metadata = true;
            if (tag == "energy") {
                value *= HARTREE_TO_EV;
                if (!std::isfinite(value)) return false;
                atom.energy_ = value;
                atom.have_energy = true;
            } else {
                have_charge = true;
                atom.global_attribs_d_["total_charge"] = value;
            }
        } else if (tag == "end") {
            if ((lattice_rows != 0 && lattice_rows != 3) || !tokens.next_token().empty()) return false;
            atom.pbc_ = {lattice_rows == 3, lattice_rows == 3, lattice_rows == 3};
            output = std::move(atom);
            return true;
        } else {
            return false;
        }
    }
}
bool GPIO::load_next_from_cp2kxyz(std::istream &is, Atoms &output, bool *clean_eof) {
    output.clear();
    if (clean_eof) *clean_eof = false;
    if (absolute_index_read) {
        std::cerr << "Error: CP2KXYZ resolves element tokens; absolute_index_read is unsupported\n";
        return false;
    }
    if (is.bad() || is.fail()) return false;
    auto at_clean_eof = [&]() {
        return is.eof() && !is.bad() && !(is.rdstate() & std::ios::failbit);
    };
    // Only a healthy frame boundary is EOF. Once the count line has started,
    // any bad/missing count, comment or atom is a malformed frame.
    try {
        if (is.eof() || is.peek() == std::char_traits<char>::eof()) {
            if (clean_eof) *clean_eof = at_clean_eof();
            return false;
        }
    } catch (const std::ios_base::failure &) {
        if (!at_clean_eof()) throw;
        if (clean_eof) *clean_eof = true;
        return false;
    }
    if (ncfgs_ == INT_MAX) return false;
    Atoms atom;
    atom.clear();
    atom.pbc_ = {false, false, false}; // Geometry-only XYZ supplies no periodic cell.
    std::string line;
    if (!structure_text_line(is, line)) return false;
    GPIO_LineTokenizer counts(line);
    int count;
    if (!structure_text_number(counts.next_token(), count) || count < 0 ||
        count > gpio_binary::MAX_ATOMS || !counts.next_token().empty()) return false;
    if (!structure_text_line(is, line)) return false; // comment may be an empty line
    for (int i = 0; i < count; ++i) {
        if (!structure_text_line(is, line)) return false;
        GPIO_LineTokenizer tokens(line);
        const auto type = mapnameid.find(std::string(tokens.next_token()));
        if (type == mapnameid.end()) return false;
        Eigen::RowVector3d position;
        if (!structure_text_vector(tokens, position) || !tokens.next_token().empty()) return false;
        atom.chemmap_.push_back(type->second);
        atom.pos_.push_back(position);
    }
    output = std::move(atom);
    return true;
}
namespace {

// These helpers are private to EXTXYZ; other text formats retain their contracts.
bool extxyz_space(char c) { return std::isspace(static_cast<unsigned char>(c)); }

std::string_view extxyz_trim(std::string_view text) {
    while (!text.empty() && extxyz_space(text.front())) text.remove_prefix(1);
    while (!text.empty() && extxyz_space(text.back())) text.remove_suffix(1);
    return text;
}

bool extxyz_line(std::istream &input, std::string &line) {
    line.clear();
    try {
        return static_cast<bool>(std::getline(input, line));
    } catch (const std::ios_base::failure &) {
        // A caller may enable failbit/eofbit exceptions. A last, unterminated
        // but nonempty line is valid; an I/O/decoder failure must propagate.
        if (input.bad() || !input.eof()) throw;
        return !line.empty();
    }
}

std::string extxyz_text(std::string_view value, char quote) {
    std::string text;
    text.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        // Decode only quoted delimiters and backslashes; keep other literal
        // backslashes, including paths, unchanged.
        if (quote && value[i] == '\\' && i + 1 < value.size() &&
            (value[i + 1] == quote || value[i + 1] == '\\')) ++i;
        text += value[i];
    }
    return text;
}

template<class T> bool extxyz_number(std::string_view token, T &value) {
    token = extxyz_trim(token);
    if (token.empty()) return false;
    if (token.front() == '+') {
        token.remove_prefix(1);
        if (token.empty() || token.front() == '-' || token.front() == '+') return false;
    }
    const auto result = fast_float::from_chars(token.data(), token.data() + token.size(), value);
    if (result.ec != std::errc{} || result.ptr != token.data() + token.size()) return false;
    if constexpr (std::is_floating_point_v<T>) return std::isfinite(value);
    return true;
}

// Atom/property tokens may be quoted strings. Numeric tokens still have to be
// complete numbers after their optional quotes have been removed.
bool extxyz_tokens(std::string_view text, std::vector<std::string_view> &tokens) {
    tokens.clear();
    while (!(text = extxyz_trim(text)).empty()) {
        if (text.front() == '"' || text.front() == '\'') {
            const char quote = text.front();
            size_t end = 1;
            for (; end < text.size(); ++end) {
                if (text[end] == '\\' && end + 1 < text.size()) { ++end; continue; }
                if (text[end] == quote) break;
            }
            if (end == text.size()) return false;
            tokens.push_back(text.substr(1, end - 1));
            text.remove_prefix(end + 1);
            if (!text.empty() && !extxyz_space(text.front())) return false;
        } else {
            size_t end = 0;
            while (end < text.size() && !extxyz_space(text[end])) ++end;
            tokens.push_back(text.substr(0, end));
            text.remove_prefix(end);
        }
    }
    return true;
}

bool extxyz_numbers(std::string_view text, std::vector<double> &values) {
    std::vector<std::string_view> tokens;
    if (!extxyz_tokens(text, tokens)) return false;
    values.clear();
    for (auto token : tokens) {
        double value;
        if (!extxyz_number(token, value)) return false;
        values.push_back(value);
    }
    return true;
}

struct ExtXYZProperty {
    std::string name;
    char type;
    int width;
};

bool extxyz_properties(std::string_view text, std::vector<ExtXYZProperty> &properties, size_t &columns) {
    properties.clear();
    columns = 0;
    std::set<std::string> names;
    while (!text.empty()) {
        const size_t first = text.find(':');
        if (first == 0 || first == std::string_view::npos) return false;
        std::string name(text.substr(0, first));
        if (name == "charge") name = "charges";
        if (name.find_first_of(" \t\r\n") != std::string::npos || !names.insert(name).second) return false;
        text.remove_prefix(first + 1);
        if (text.size() < 3 || text[1] != ':') return false;
        const char type = text.front();
        if (type != 'R' && type != 'I' && type != 'S') return false;
        text.remove_prefix(2);
        const size_t end = text.find(':');
        int width = 0;
        if (!extxyz_number(text.substr(0, end), width) || width <= 0 ||
            width > gpio_binary::MAX_VECTOR_SIZE || columns > size_t(gpio_binary::MAX_VECTOR_SIZE - width)) return false;
        if (name == "species") {
            if (type != 'S' || width != 1) return false;
        } else if (name == "pos" || name == "force" || name == "forces" || name == "velocities") {
            if (type != 'R' || width != 3) return false;
        } else if (name == "site_en" || name == "energies" || name == "charges") {
            if (type != 'R' || width != 1) return false;
        } else if (name == "eff_charges") {
            if (type != 'R' || width != 9) return false;
        }
        columns += width;
        properties.push_back({std::move(name), type, width});
        if (end == std::string_view::npos) break;
        text.remove_prefix(end + 1);
        if (text.empty()) return false;
    }
    return names.count("species") && names.count("pos") &&
           !(names.count("force") && names.count("forces")) &&
           !(names.count("site_en") && names.count("energies"));
}

} // namespace

bool GPIO::load_next_from_extxyz(std::istream &is, Atoms &atom, bool *clean_eof)
{
    if (clean_eof) *clean_eof = false;
    std::string line;
    do {
        if (!extxyz_line(is, line)) {
            if (clean_eof) *clean_eof = is.eof() && !is.bad();
            return false;
        }
    } while (extxyz_trim(line).empty());
    int natoms = 0;
    if (!extxyz_number(line, natoms) || natoms < 0 || natoms > gpio_binary::MAX_ATOMS) return false;
    if (!extxyz_line(is, line)) return false;

    // Do not expose a partly parsed frame, even when high-order labels throw.
    Atoms frame;
    frame.resize(natoms);
    bool have_lattice = false, have_pbc = false, have_properties = false, stress_is_density = false;
    bool have_canonical_fenergy = false;
    std::vector<ExtXYZProperty> properties;
    size_t columns = 0;
    std::vector<double> values;
    std::string_view header(line);
    while (!(header = extxyz_trim(header)).empty()) {
        size_t key_end = 0;
        while (key_end < header.size() && header[key_end] != '=' && !extxyz_space(header[key_end])) ++key_end;
        if (!key_end) return false;
        const std::string key(header.substr(0, key_end));
        header.remove_prefix(key_end);
        header = extxyz_trim(header);
        // Free comment words were historically ignored, including the trailing
        // words in a malformed duplicate `pbc=T T T` after a valid pbc value.
        if (header.empty() || header.front() != '=') continue;
        header.remove_prefix(1);
        header = extxyz_trim(header);
        if (header.empty()) return false;
        std::string_view value;
        char value_quote = '\0';
        if (header.front() == '"' || header.front() == '\'') {
            const char quote = header.front();
            value_quote = quote;
            size_t end = 1;
            for (; end < header.size(); ++end) {
                if (header[end] == '\\' && end + 1 < header.size()) { ++end; continue; }
                if (header[end] == quote) break;
            }
            if (end == header.size()) {
                if (HVPIO::is_key(key)) throw std::invalid_argument("Unterminated HVP tag");
                return false;
            }
            value = header.substr(1, end - 1);
            header.remove_prefix(end + 1);
            if (!header.empty() && !extxyz_space(header.front())) return false;
        } else {
            size_t end = 0;
            while (end < header.size() && !extxyz_space(header[end])) ++end;
            value = header.substr(0, end);
            header.remove_prefix(end);
        }

        if (key == "config_type" || key == "label") {
            frame.features_[key] = extxyz_text(value, value_quote);
        } else if (key == "energy" || key == "fenergy") {
            double number;
            if (!extxyz_number(value, number)) return false;
            if (key == "energy") { frame.energy_ = number; frame.have_energy = true; }
            else {
                frame.fenergy_ = number;
                frame.have_fenergy = true;
                have_canonical_fenergy = true;
            }
        } else if (key == "free_energy") {
            double number;
            if (extxyz_number(value, number)) {
                if (!have_canonical_fenergy) {
                    frame.fenergy_ = number;
                    frame.have_fenergy = true;
                }
            } else {
                frame.features_[key] = extxyz_text(value, value_quote);
            }
        } else if (HVPIO::is_key(key)) {
            HVPIO::read(frame, key, value);
        } else if (key == "hessian") {
            const size_t expected = size_t(9) * natoms * natoms;
            if (!extxyz_numbers(value, values) || values.size() != expected)
                throw std::invalid_argument("Invalid EXTXYZ hessian: expected " + std::to_string(expected) + " finite numbers");
            frame.hessian_.resize(3 * natoms, 3 * natoms);
            for (int i = 0; i < 3 * natoms; ++i)
                for (int j = 0; j < 3 * natoms; ++j)
                    frame.hessian_(i, j) = values[size_t(i) * 3 * natoms + j];
            frame.have_hessian = true;
        } else if (key == "Lattice" || key == "lattice") {
            if (!extxyz_numbers(value, values) || values.size() != 9) return false;
            for (int i = 0; i < 9; ++i) frame.cell_(i / 3, i % 3) = values[i];
            have_lattice = true;
        } else if (key == "Properties") {
            if (have_properties || !extxyz_properties(value, properties, columns)) return false;
            have_properties = true;
        } else if (key == "pbc") {
            std::array<bool, 3> pbc{};
            std::vector<std::string_view> tokens;
            bool valid = extxyz_tokens(value, tokens);
            if (valid && tokens.size() == 1 && tokens[0].size() == 3) {
                const auto compact = tokens[0];
                tokens = {compact.substr(0, 1), compact.substr(1, 1), compact.substr(2, 1)};
            }
            valid = valid && tokens.size() == 3;
            if (valid) {
                for (int i = 0; i < 3; ++i) {
                    std::string flag(tokens[i]);
                    for (char &c : flag) c = std::tolower(static_cast<unsigned char>(c));
                    if (flag == "t" || flag == "true") pbc[i] = true;
                    else if (flag == "f" || flag == "false") pbc[i] = false;
                    else valid = false;
                }
            }
            if (valid) { frame.pbc_ = pbc; have_pbc = true; }
            else if (!have_pbc) return false;
        } else if (key == "virial" || key == "stress") {
            if (!extxyz_numbers(value, values) || (values.size() != 9 && values.size() != 6)) return false;
            if (values.size() == 9) {
                for (int i = 0; i < 9; ++i) frame.stress_(i / 3, i % 3) = values[i];
            } else {
                frame.stress_ << values[0], values[3], values[4],
                                 values[3], values[1], values[5],
                                 values[4], values[5], values[2];
            }
            stress_is_density = key == "stress";
            frame.have_stress = true;
        } else if (read_all_keys || gtags_.count(key)) {
            double number;
            if (extxyz_number(value, number)) {
                frame.global_attribs_d_[key] = number;
            } else {
                // Untyped metadata may contain dates, identifiers or other text.
                frame.features_[key] = extxyz_text(value, value_quote);
            }
        }
    }
    if (!have_properties) return false;
    frame.validate_hvp();

    for (const auto &property : properties) {
        if (natoms && property.width > gpio_binary::MAX_VECTOR_SIZE / natoms) return false;
        const auto &name = property.name;
        if (name == "forces" || name == "force") frame.forces_.resize(natoms);
        else if (name == "velocities") frame.velocities_.resize(natoms);
        else if (name == "site_en" || name == "energies") frame.site_energies_.resize(natoms);
        else if (name == "charges") frame.charges_.resize(natoms);
        else if (name == "eff_charges") frame.born_eff_chgs_.resize(natoms);
        else if (name == "magmoms" && property.type == 'R' && property.width == 1)
            frame.magmom_scalar_.resize(natoms);
        else if (name == "magmoms" && property.type == 'R' && property.width == 3) {
            frame.magmoms_.resize(natoms);
            frame.collinear_ = false;
        } else if (name != "species" && name != "pos") {
            if (property.type == 'R') {
                if (property.width == 1) frame.alloc_ld(name);
                else frame.alloc_lvd(name, property.width);
            } else if (property.type == 'I') {
                if (property.width == 1) frame.alloc_li(name);
                else frame.alloc_lvi(name, property.width);
            }
            // Atoms has no per-atom string arrays; keep the established S
            // behavior: consume and validate their columns without storing them.
        }
    }
    Eigen::RowVector3d lo = Eigen::RowVector3d::Constant(std::numeric_limits<double>::infinity());
    Eigen::RowVector3d hi = -lo;
    std::vector<std::string_view> tokens;
    for (int i = 0; i < natoms; ++i) {
        if (!extxyz_line(is, line) || !extxyz_tokens(line, tokens) || tokens.size() != columns) return false;
        size_t column = 0;
        for (const auto &property : properties) {
            const auto &name = property.name;
            if (name == "species") {
                auto species = mapnameid.find(std::string(tokens[column]));
                if (species == mapnameid.end()) return false;
                frame.chemmap_[i] = species->second;
            }
            for (int j = 0; j < property.width; ++j) {
                const auto token = tokens[column++];
                if (property.type == 'S') continue;
                if (property.type == 'I') {
                    int number;
                    if (!extxyz_number(token, number)) return false;
                    if (property.width == 1) frame.local_attribs_i_[name](i) = number;
                    else frame.local_attribs_vi_[name](j, i) = number;
                } else {
                    double number;
                    if (!extxyz_number(token, number)) return false;
                    if (name == "pos") frame.pos_[i][j] = number;
                    else if (name == "forces" || name == "force") frame.forces_[i][j] = number;
                    else if (name == "velocities") frame.velocities_[i][j] = number;
                    else if (name == "site_en" || name == "energies") frame.site_energies_[i] = number;
                    else if (name == "charges") frame.charges_[i] = number;
                    else if (name == "eff_charges") frame.born_eff_chgs_[i](j / 3, j % 3) = number;
                    else if (name == "magmoms" && property.width == 1) frame.magmom_scalar_[i] = number;
                    else if (name == "magmoms" && property.width == 3) frame.magmoms_[i][j] = number;
                    else if (property.width == 1) frame.local_attribs_d_[name](i) = number;
                    else frame.local_attribs_v_[name](j, i) = number;
                }
            }
        }
        lo = lo.cwiseMin(frame.pos_[i]);
        hi = hi.cwiseMax(frame.pos_[i]);
    }
    if (!have_lattice) {
        if (!std::isfinite(r_extent) || r_extent < 0) return false;
        frame.cell_.setZero();
        for (int i = 0; i < 3; ++i) frame.cell_(i, i) = (natoms ? hi[i] - lo[i] : 0.0) + 2 * r_extent;
        if (!frame.cell_.allFinite()) return false;
    }
    if (stress_is_density) {
        frame.stress_ *= -frame.get_volume();
        if (!frame.stress_.allFinite()) return false;
    }
    atom = std::move(frame);
    return true;
}

bool GPIO::read_extented_xyz_sequence(std::istream &is, Atoms &atom) {
    return load_next_from_extxyz(is, atom);
}
