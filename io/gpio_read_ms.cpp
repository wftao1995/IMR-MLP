// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpio.h"
#include "gpio_tokenizer.h"
#include "tools/chem.h"
#include "tools/procstr.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <tinyxml2.h>

namespace {

// resolve a Materials Studio element token (possibly 'ca', 'Fe2+', 'D', ...)
// to the internal species index, or -1 when it cannot be resolved
int ms_element_index(std::string_view tok, const std::map<std::string, int> &mapnameid)
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

// lower-triangular cell from a,b,c (Angstrom) and angles in degrees
Eigen::Matrix3d cell_from_params(double a, double b, double c,
                                 double alpha, double beta, double gamma)
{
    const double ar = alpha * M_PI / 180.0;
    const double br = beta * M_PI / 180.0;
    const double gr = gamma * M_PI / 180.0;
    const double sg = sin(gr);
    if (a <= 0 || b <= 0 || c <= 0 || std::abs(sg) < 1e-12)
        return Eigen::Matrix3d::Zero();
    Eigen::Matrix3d cell;
    cell << a, 0, 0,
            b * cos(gr), b * sg, 0,
            c * cos(br), c * (cos(ar) - cos(br) * cos(gr)) / sg, 0;
    double cz_sq = c * c - cell(2, 0) * cell(2, 0) - cell(2, 1) * cell(2, 1);
    cell(2, 2) = std::sqrt(std::max(0.0, cz_sq));
    return cell;
}

// tokenize an MSI record like "(A D A3 (1.0 0.0 0.0))" into bare tokens,
// keeping quoted strings intact and dropping the parentheses:
// -> {"A", "D", "A3", "1.0", "0.0", "0.0"}
void tokenize_record(std::string_view rec, std::vector<std::string> &toks)
{
    toks.clear();
    size_t i = 0;
    const size_t n = rec.size();
    while (i < n) {
        while (i < n && std::isspace(static_cast<unsigned char>(rec[i]))) ++i;
        if (i >= n) break;
        if (rec[i] == '(' || rec[i] == ')') {
            ++i;
            continue;
        }
        const size_t start = i;
        if (rec[i] == '"' || rec[i] == '\'') {
            const char q = rec[i];
            ++i;
            while (i < n && rec[i] != q) ++i;
            ++i;
            toks.emplace_back(rec.substr(start, i - start));
        } else {
            while (i < n && !std::isspace(static_cast<unsigned char>(rec[i])) &&
                   rec[i] != '(' && rec[i] != ')') {
                ++i;
            }
            toks.emplace_back(rec.substr(start, i - start));
        }
    }
    for (auto &t : toks) {
        if (t.size() >= 2 && (t.front() == '"' || t.front() == '\'') &&
            t.back() == t.front()) {
            t = t.substr(1, t.size() - 2);
        }
    }
}

// parse a comma/space separated vector attribute: "9.37,-0.74,-0.69" -> (9.37, -0.74, -0.69)
bool parse_vector3(const char *text, Eigen::RowVector3d &v)
{
    if (!text) return false;
    std::string s(text);
    for (auto &c : s) {
        if (c == ',') c = ' ';
    }
    GPIO_LineTokenizer toks{s};
    double x, y, z;
    if (!toks.next_numeric(x) || !toks.next_numeric(y) || !toks.next_numeric(z)) return false;
    v << x, y, z;
    return true;
}

struct XsdAtom {
    int id = 0;
    int type = -1;
    Eigen::RowVector3d pos = Eigen::RowVector3d::Zero();
    bool cart = false;         // coordinates are Cartesian (vs fractional)
    bool have_charge = false;
    double charge = 0.0;
};

} // namespace

/***
 * Readers for the three Materials Studio structure formats:
 * - CAR: Biosym archive text format
 * - MSI: Cerius2 DataModel text format
 * - XSD: Materials Studio XML structure format
 * ***/

/***
 * Read a Materials Studio / Biosym CAR file.
 *
 * Layout (as written by Materials Studio, see also LAMMPS' msi2lmp):
 *
 *   !BIOSYM archive 3
 *   PBC=ON
 *   <comment line>
 *   !DATE ...
 *   PBC   9.42 18.84 6.88 90.0 90.0 90.0 (P1)
 *   O1    -1.62 8.10 1.78 XXXX 1 o O -0.587
 *   ... (one line per atom; 9 columns:
 *        label x y z coordflag mol potential element charge)
 *
 * Molecule sections are separated by "end" lines; all atoms of the file
 * are merged into one structure. Coordinates are Cartesian in Angstrom.
 * The cell line may be absent (non-periodic systems).
 * ***/
bool GPIO::read_car(std::istream &is, Atoms &atom)
{
    std::string line;
    if (!std::getline(is, line)) return false;
    size_t p = line.find_first_not_of(" \t\r");
    if (p == std::string::npos || line.compare(p, 8, "!BIOSYM ") != 0) {
        std::cerr << "Error: not a Biosym CAR file (missing !BIOSYM header)" << std::endl;
        return false;
    }

    bool periodic = false;
    double a = 0, b = 0, c = 0, alpha = 90.0, beta = 90.0, gamma = 90.0;
    bool have_cell = false;
    bool have_charge = false;
    while (std::getline(is, line)) {
        GPIO_LineTokenizer toks{line};
        auto tok = toks.next_token();
        if (tok.empty()) continue;
        if (tok[0] == '!' || tok[0] == '#') continue;
        if (tok == "end") continue;
        if (tok.starts_with("PBC=")) {
            periodic = tok.size() >= 6 && tok[4] == 'O' && tok[5] == 'N';
            continue;
        }
        if (tok == "PBC") {
            // cell line: PBC a b c alpha beta gamma [spacegroup]
            double v[6];
            bool ok = true;
            for (int i = 0; i < 6; ++i)
                ok = toks.next_numeric(v[i]) && ok;
            if (ok) {
                a = v[0];
                b = v[1];
                c = v[2];
                alpha = v[3];
                beta = v[4];
                gamma = v[5];
                have_cell = true;
            }
            continue;
        }
        // atom line
        std::vector<std::string> splstr;
        str_split(line, splstr);
        const int ncols = static_cast<int>(splstr.size());
        if (ncols < 5) continue;
        // skip comment-like lines (e.g. "Materials Studio Generated CAR File")
        double dummy;
        GPIO_LineTokenizer ctoks{line};
        ctoks.next_token(); // label
        if (!ctoks.next_numeric(dummy)) continue;
        ctoks.next_numeric(dummy);
        ctoks.next_numeric(dummy);
        int type = -1;
        if (ncols >= 8) {
            type = ms_element_index(splstr[7], mapnameid);
        }
        if (type < 0) {
            // fallback: element from the atom label (e.g. O1 -> O)
            std::string_view lab = splstr[0];
            while (!lab.empty() && std::isdigit(static_cast<unsigned char>(lab.back())))
                lab.remove_suffix(1);
            type = ms_element_index(lab, mapnameid);
        }
        if (type < 0) {
            std::cerr << "Warning: car: unknown element \"" << splstr[0]
                      << "\", skipping atom" << std::endl;
            continue;
        }
        double x, y, z;
        GPIO_LineTokenizer atoks{line};
        atoks.next_token(); // label
        if (!atoks.next_numeric(x) || !atoks.next_numeric(y) || !atoks.next_numeric(z)) {
            std::cerr << "Warning: car: invalid coordinates, skipping atom" << std::endl;
            continue;
        }
        atom.chemmap_.push_back(type);
        atom.pos_.emplace_back(x, y, z);
        atom.charges_.push_back(0.0);
        if (ncols >= 9) {
            have_charge = true;
            double q = 0;
            auto res = fast_float::from_chars(splstr[8].data(), splstr[8].data() + splstr[8].size(), q);
            if (res.ec == std::errc{})
                atom.charges_.back() = q;
        }
    }

    if (!have_charge)
        atom.charges_.clear();
    if (have_cell)
        atom.cell_ = cell_from_params(a, b, c, alpha, beta, gamma);
    if (!periodic && !have_cell)
        atom.pbc_ = {false, false, false};
    else
        atom.pbc_ = {true, true, true};
    return atom.chemmap_.size() > 0;
}

/***
 * Read a Materials Studio / Cerius2 MSI DataModel file.
 *
 * The format is a Lisp-like text model, one structure per "(N Model ...)"
 * block; a file may hold several models (read() returns one structure per
 * model). Relevant records:
 *
 *   (A I PeriodicType 100)
 *   (A D A3 (x y z))  (A D B3 (x y z))  (A D C3 (x y z))   lattice vectors
 *   (A C SpaceGroup "63 5")
 *   (1 Atom
 *     (A C ACL "6 C")        atomic number + element symbol
 *     (A D XYZ (x y z))      coordinates (Angstrom, Cartesian)
 *     (A D Charge 0.5)       optional charge
 *   )
 *   (1 Bond (A I Atom1 i) (A I Atom2 j) (A I Type 1))
 *
 * Unknown records are ignored. Following OpenBabel's reader, atom
 * coordinates are taken as Cartesian in Angstrom.
 * ***/
bool GPIO::read_msi(std::istream &is, Atoms &atom)
{
    // per-model state
    Eigen::Matrix3d cell = Eigen::Matrix3d::Zero();
    std::vector<int> atomids;              // MSI atom ID -> order index
    std::vector<int> acl_num;              // atomic number from the ACL record
    std::vector<std::string> acl_sym;      // element symbol from the ACL record
    std::vector<Eigen::RowVector3d> pos;
    std::vector<double> charges;
    bool have_charges = false;
    std::vector<std::array<int, 2>> bonds;
    int bond_a = -1, bond_b = -1;

    bool in_model = false;
    int depth = 0;                         // running paren depth across lines
    int atom_index = -1;                   // index of the atom record being filled

    auto parse_d = [](const std::string &tok, double &v) -> bool {
        auto res = fast_float::from_chars(tok.data(), tok.data() + tok.size(), v);
        return res.ec == std::errc{};
    };
    auto parse_i = [](const std::string &tok, int &v) -> bool {
        auto res = fast_float::from_chars(tok.data(), tok.data() + tok.size(), v);
        return res.ec == std::errc{};
    };

    auto flush_bond = [&]() {
        if (bond_a < 0 && bond_b < 0) return;
        auto ia = std::find(atomids.begin(), atomids.end(), bond_a);
        auto ib = std::find(atomids.begin(), atomids.end(), bond_b);
        if (bond_a >= 0 && bond_b >= 0 && ia != atomids.end() && ib != atomids.end()) {
            bonds.push_back({static_cast<int>(ia - atomids.begin()),
                             static_cast<int>(ib - atomids.begin())});
        } else {
            std::cerr << "Warning: msi: bond " << bond_a << "-" << bond_b
                      << " references a missing atom, skipping bond" << std::endl;
        }
        bond_a = bond_b = -1;
    };

    auto finalize = [&]() {
        atom.chemmap_.reserve(pos.size());
        atom.pos_.reserve(pos.size());
        std::vector<int> retained_indices(pos.size(), -1);
        for (size_t i = 0; i < pos.size(); ++i) {
            int type = -1;
            if (!acl_sym[i].empty())
                type = ms_element_index(acl_sym[i], mapnameid);
            if (type < 0 && acl_num[i] > 0 && acl_num[i] < 119)
                type = ms_element_index(Chemicals::chemical_symbols[acl_num[i]], mapnameid);
            if (type < 0) {
                std::cerr << "Warning: msi: unknown element \"" << acl_sym[i]
                          << "\", skipping atom" << std::endl;
                continue;
            }
            retained_indices[i] = static_cast<int>(atom.pos_.size());
            atom.chemmap_.push_back(type);
            atom.pos_.push_back(pos[i]);
        }
        atom.cell_ = cell;
        if (have_charges && charges.size() == pos.size()) {
            atom.charges_ = charges;
            if (atom.chemmap_.size() != pos.size()) atom.charges_.clear();
        }
        for (const auto &bond : bonds) {
            const int a = retained_indices[bond[0]], b = retained_indices[bond[1]];
            if (a < 0 || b < 0) {
                std::cerr << "Warning: msi: bond " << atomids[bond[0]] << "-"
                          << atomids[bond[1]]
                          << " references a skipped atom, skipping bond" << std::endl;
                continue;
            }
            atom.bonds_.push_back({a, b});
        }
        return atom.chemmap_.size() > 0;
    };

    std::string line;
    while (std::getline(is, line)) {
        size_t p = line.find_first_not_of(" \t\r");
        if (p == std::string::npos) continue;
        if (line[p] == '!' || line[p] == '#') continue;

        // scan the line, tracking paren depth across lines; records that open
        // and close on this line are processed immediately, container openers
        // (Model / Atom / Bond) are handled by their first line
        const size_t n = line.size();
        size_t i = 0;
        int rec_begin = -1;    // '(' index of the outermost record on this line
        int rec_open_depth = -1;
        while (i < n) {
            const char c = line[i];
            if (c == '"' || c == '\'') {
                const char q = c;
                ++i;
                while (i < n && line[i] != q) ++i;
                ++i;
                continue;
            }
            if (c == '(') {
                if (rec_begin < 0) {
                    rec_begin = static_cast<int>(i);
                    rec_open_depth = depth;
                }
                ++depth;
                ++i;
                continue;
            }
            if (c == ')') {
                --depth;
                if (rec_begin >= 0 && depth == rec_open_depth) {
                    // complete record on this line
                    std::string_view rec(line.data() + rec_begin, i - rec_begin + 1);
                    rec_begin = -1;
                    std::vector<std::string> toks;
                    tokenize_record(rec, toks);
                    if (toks.size() >= 3 && toks[2] == "A3" && toks.size() >= 6) {
                        double x, y, z;
                        if (parse_d(toks[3], x) && parse_d(toks[4], y) && parse_d(toks[5], z))
                            cell.row(0) << x, y, z;
                    } else if (toks.size() >= 3 && toks[2] == "B3" && toks.size() >= 6) {
                        double x, y, z;
                        if (parse_d(toks[3], x) && parse_d(toks[4], y) && parse_d(toks[5], z))
                            cell.row(1) << x, y, z;
                    } else if (toks.size() >= 3 && toks[2] == "C3" && toks.size() >= 6) {
                        double x, y, z;
                        if (parse_d(toks[3], x) && parse_d(toks[4], y) && parse_d(toks[5], z))
                            cell.row(2) << x, y, z;
                    } else if (toks.size() >= 3 && toks[2] == "XYZ" && atom_index >= 0 &&
                               toks.size() >= 6) {
                        double x, y, z;
                        if (parse_d(toks[3], x) && parse_d(toks[4], y) && parse_d(toks[5], z))
                            pos[atom_index] << x, y, z;
                    } else if (toks.size() >= 3 && toks[2] == "ACL" && atom_index >= 0 &&
                               toks.size() >= 4) {
                        // ACL "6 C" -> atomic number + element symbol
                        std::vector<std::string> sub;
                        str_split(toks[3], sub);
                        if (!sub.empty()) {
                            int num = 0;
                            auto res = fast_float::from_chars(sub[0].data(),
                                                              sub[0].data() + sub[0].size(), num);
                            acl_num[atom_index] = (res.ec == std::errc{}) ? num : 0;
                            if (sub.size() >= 2) acl_sym[atom_index] = sub[1];
                            else if (acl_num[atom_index] > 0 && acl_num[atom_index] < 119)
                                acl_sym[atom_index] = Chemicals::chemical_symbols[acl_num[atom_index]];
                        }
                    } else if (toks.size() >= 3 && (toks[2] == "Charge" || toks[2] == "Q") &&
                               atom_index >= 0 && toks.size() >= 4) {
                        double q = 0;
                        if (parse_d(toks[3], q)) {
                            charges[atom_index] = q;
                            have_charges = true;
                        }
                    } else if (toks.size() >= 3 && toks[2] == "Atom1" && toks.size() >= 4) {
                        int v = 0;
                        if (parse_i(toks[3], v)) bond_a = v;
                    } else if (toks.size() >= 3 && toks[2] == "Atom2" && toks.size() >= 4) {
                        int v = 0;
                        if (parse_i(toks[3], v)) bond_b = v;
                    }
                    // all other records (PeriodicType, Name, Type, ...) are ignored
                }
                ++i;
                continue;
            }
            ++i;
        }

        // container openers: a record whose outer parens did not close on this line
        if (rec_begin >= 0) {
            std::string_view rec(line.data() + rec_begin, n - rec_begin);
            std::vector<std::string> toks;
            tokenize_record(rec, toks);
            rec_begin = -1;
            if (toks.size() >= 2 && toks[1] == "Model") {
                // "(N Model" at top level: start of a structure
                in_model = true;
            } else if (in_model && toks.size() >= 2 && toks[1] == "Atom") {
                flush_bond();
                atom_index = static_cast<int>(pos.size());
                int id = 0;
                if (!toks.empty()) {
                    auto res = fast_float::from_chars(toks[0].data(),
                                                      toks[0].data() + toks[0].size(), id);
                    if (res.ec != std::errc{}) id = atom_index + 1;
                }
                atomids.push_back(id);
                acl_num.push_back(0);
                acl_sym.emplace_back();
                pos.emplace_back(0.0, 0.0, 0.0);
                charges.push_back(0.0);
            } else if (in_model && toks.size() >= 2 && toks[1] == "Bond") {
                flush_bond();
            }
        }

        // model end: a ')' that closed the top-level record on this line
        if (in_model && depth == 0) {
            flush_bond();
            if (atomids.empty()) {
                // model without atoms — keep scanning for the next model
                in_model = false;
                continue;
            }
            return finalize();
        }
    }

    if (in_model) {
        flush_bond();
        return finalize();
    }
    return false;
}

/***
 * Read a Materials Studio XML structure (.xsd) file.
 *
 * The layout varies between exports; all of the following are handled by
 * walking the whole document tree:
 *
 *   <XSD Version="18.1">
 *     <AtomisticTreeRoot ...>...</AtomisticTreeRoot>
 *     <SpaceGroup Name="R-3M" ITNumber="166"
 *       AVector="9.37,-0.74,-0.69" BVector="..." CVector="..."
 *       Operators="..."/>
 *     <Atom3d ID="8" Name="Si1" XYZ="0.104,0.334,0.875" Components="Si"/>
 *     <Bond ID="10" Connects="8,4"/>
 *   </XSD>
 *
 * - the cell comes from the first SpaceGroup carrying AVector/BVector/CVector
 * - atoms are all Atom3d elements with Components + coordinates; XYZ is
 *   fractional when a cell is present, CartesianXYZ is always Cartesian
 *   (non-periodic systems keep XYZ Cartesian)
 * - Bond elements map to bonds_ via the Atom3d ID attributes
 * - charges are read from the optional Charge attribute
 * ***/
bool GPIO::read_xsd(std::istream &is, Atoms &atom)
{
    using namespace tinyxml2;
    std::string buf((std::istreambuf_iterator<char>(is)), std::istreambuf_iterator<char>());
    XMLDocument doc;
    if (doc.Parse(buf.data(), buf.size()) != XML_SUCCESS) {
        std::cerr << "Error: xsd: XML parse failed" << std::endl;
        return false;
    }
    XMLElement *root = doc.RootElement();
    if (!root) {
        std::cerr << "Error: xsd: empty document" << std::endl;
        return false;
    }

    Eigen::Matrix3d cell = Eigen::Matrix3d::Zero();
    bool have_cell = false;
    std::vector<XsdAtom> atoms;
    std::vector<std::pair<int, int>> bonds;   // Atom3d IDs

    // recursive walk over the document tree; the cell must be found before
    // atoms are parsed (XYZ is fractional only when a cell is present)
    std::function<void(XMLElement *)> find_cell = [&](XMLElement *el) {
        for (XMLElement *child = el->FirstChildElement(); child; child = child->NextSiblingElement()) {
            if (have_cell) return;
            const char *name = child->Name();
            if (name && strcmp(name, "SpaceGroup") == 0) {
                Eigen::RowVector3d a, b, c;
                if (parse_vector3(child->Attribute("AVector"), a) &&
                    parse_vector3(child->Attribute("BVector"), b) &&
                    parse_vector3(child->Attribute("CVector"), c)) {
                    cell.row(0) = a;
                    cell.row(1) = b;
                    cell.row(2) = c;
                    have_cell = true;
                    return;
                }
            }
            find_cell(child);
        }
    };
    find_cell(root);

    std::function<void(XMLElement *)> collect = [&](XMLElement *el) {
        for (XMLElement *child = el->FirstChildElement(); child; child = child->NextSiblingElement()) {
            const char *name = child->Name();
            if (!name) continue;
            if (strcmp(name, "Atom3d") == 0) {
                const char *comp = child->Attribute("Components");
                const char *xyz = child->Attribute("XYZ");
                const char *cxyz = child->Attribute("CartesianXYZ");
                if (!comp || (!xyz && !cxyz)) continue;
                XsdAtom at;
                if (const char *idattr = child->Attribute("ID")) {
                    auto res = fast_float::from_chars(idattr, idattr + std::strlen(idattr), at.id);
                    if (res.ec != std::errc{}) at.id = 0;
                }
                at.type = ms_element_index(comp, mapnameid);
                if (at.type < 0) {
                    std::cerr << "Warning: xsd: unknown element \"" << comp
                              << "\", skipping atom" << std::endl;
                    continue;
                }
                if (cxyz && parse_vector3(cxyz, at.pos)) {
                    at.cart = true;
                } else if (xyz && parse_vector3(xyz, at.pos)) {
                    at.cart = !have_cell;
                } else {
                    continue;
                }
                if (const char *chg = child->Attribute("Charge")) {
                    auto res = fast_float::from_chars(chg, chg + std::strlen(chg), at.charge);
                    if (res.ec == std::errc{}) at.have_charge = true;
                }
                atoms.push_back(at);
            } else if (strcmp(name, "Bond") == 0) {
                const char *conn = child->Attribute("Connects");
                if (!conn) continue;
                int a = 0, b = 0;
                auto res = fast_float::from_chars(conn, conn + std::strlen(conn), a);
                if (res.ec != std::errc{}) continue;
                const char *comma = std::strchr(conn, ',');
                if (!comma) continue;
                auto res2 = fast_float::from_chars(comma + 1, comma + std::strlen(comma) + 1, b);
                if (res2.ec != std::errc{}) continue;
                bonds.emplace_back(a, b);
            }
            collect(child);
        }
    };
    collect(root);

    if (atoms.empty()) {
        std::cerr << "Error: xsd: no atoms found" << std::endl;
        return false;
    }

    atom.cell_ = cell;
    if (have_cell)
        atom.pbc_ = {true, true, true};
    else
        atom.pbc_ = {false, false, false};
    std::vector<int> ids;
    bool have_charges = false;
    atom.chemmap_.reserve(atoms.size());
    atom.pos_.reserve(atoms.size());
    for (auto &at : atoms) {
        atom.chemmap_.push_back(at.type);
        if (at.cart)
            atom.pos_.push_back(at.pos);
        else
            atom.pos_.push_back(at.pos * cell);
        ids.push_back(at.id);
        if (at.have_charge) have_charges = true;
    }
    if (have_charges) {
        atom.charges_.resize(atoms.size());
        for (size_t i = 0; i < atoms.size(); ++i)
            atom.charges_[i] = atoms[i].charge;
    }
    for (auto &bd : bonds) {
        auto ia = std::find(ids.begin(), ids.end(), bd.first);
        auto ib = std::find(ids.begin(), ids.end(), bd.second);
        if (ia != ids.end() && ib != ids.end())
            atom.bonds_.push_back({static_cast<int>(ia - ids.begin()),
                                   static_cast<int>(ib - ids.begin())});
    }
    return true;
}
