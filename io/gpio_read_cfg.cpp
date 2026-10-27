// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpio.h"
#include "hvp_io.h"
#include "io/hessian_io.h"
#include "gpio_binary.h"
#include "gpio_tokenizer.h"
#include "gpio_lmp.h"
#include "tools/chem.h"
#include "tools/procstr.h"
#include "tools/tools.h"
#include "utils/comm.h"
#include <cmath>
#include <limits>

namespace {

bool parse_atom_id(double value, int natoms, std::vector<bool> &seen, int &id)
{
    if (!std::isfinite(value) || std::trunc(value) != value || value < 1.0 || value > natoms)
        return false;
    id = static_cast<int>(value) - 1;
    if (seen[id]) return false;
    seen[id] = true;
    return true;
}

bool parse_number(std::string_view token, double &value)
{
    const auto result = fast_float::from_chars(token.data(), token.data() + token.size(), value);
    return result.ec == std::errc{} && result.ptr == token.data() + token.size();
}

} // namespace

bool GPIO::load_next_from_mlpcfg_bin(std::istream &is, Atoms &output, bool *clean_eof) {
    using namespace gpio_binary;
    output.clear();
    if (clean_eof) *clean_eof = false;
    // Only a probe at a healthy frame boundary may report normal EOF. In
    // particular, an earlier failed read must not hide a transport failure.
    if (is.bad() || is.fail()) return false;
    if (is.eof()) {
        if (clean_eof) *clean_eof = true;
        return false;
    }
    try {
        if (is.peek() == std::char_traits<char>::eof()) {
            if (clean_eof) *clean_eof = is.eof() && !is.bad() && !(is.rdstate() & std::ios::failbit);
            return false;
        }
    } catch (const std::ios_base::failure &) {
        // A caller may enable eofbit exceptions even for a normal EOF probe.
        if (is.eof() && !is.bad() && !(is.rdstate() & std::ios::failbit)) {
            if (clean_eof) *clean_eof = true;
            return false;
        }
        throw;
    }
    if (ncfgs_ == INT_MAX)
        throw std::runtime_error("MLIPBIN frame count exceeds the supported range");
    // Commit only complete records; false returns and decoding exceptions
    // leave the public output empty rather than exposing a partial frame.
    Atoms atom;
    atom.clear();
    bool has_pos = false;
    bool has_lattice = false;
    bool has_types = false;

    int natoms = 0;
    std::uint32_t format_version = 0;
    if (!read_header(is, natoms, &format_version)) {
        std::cerr << "Invalid binary mlpcfg header" << std::endl;
        return false;
    }
    atom.resize(natoms);

    while (true) {
        Tag tag = Tag::End;
        if (!read_tag(is, tag)) return false;

        if (tag == Tag::Cell) {
            if (!read_span(is, atom.cell_.data(), 9)) return false;
            has_lattice = true;
        } else if (tag == Tag::Positions) {
            if (!read_span(is, atom.pos_.data(), atom.pos_.size())) return false;
            has_pos = true;
        } else if (tag == Tag::Types) {
            if (!read_span(is, atom.chemmap_.data(), atom.chemmap_.size())) return false;
            has_types = true;
        } else if (tag == Tag::Forces) {
            atom.forces_.resize(natoms);
            if (!read_span(is, atom.forces_.data(), atom.forces_.size())) return false;
        } else if (tag == Tag::FreeEnergy) {
            if (!read_value(is, atom.fenergy_)) return false;
            atom.have_fenergy = true;
        } else if (tag == Tag::SiteEnergies) {
            atom.site_energies_.resize(natoms);
            if (!read_span(is, atom.site_energies_.data(), atom.site_energies_.size())) return false;
        } else if (tag == Tag::MagmomVector) {
            atom.magmoms_.resize(natoms);
            if (!read_span(is, atom.magmoms_.data(), atom.magmoms_.size())) return false;
        } else if (tag == Tag::MagmomScalar) {
            atom.magmom_scalar_.resize(natoms);
            if (!read_span(is, atom.magmom_scalar_.data(), atom.magmom_scalar_.size())) return false;
        } else if (tag == Tag::Charges) {
            atom.charges_.resize(natoms);
            if (!read_span(is, atom.charges_.data(), atom.charges_.size())) return false;
        } else if (tag == Tag::Velocities) {
            atom.velocities_.resize(natoms);
            if (!read_span(is, atom.velocities_.data(), atom.velocities_.size())) return false;
        } else if (tag == Tag::Ids) {
            atom.ids_.resize(natoms);
            if (!read_span(is, atom.ids_.data(), atom.ids_.size())) return false;
        } else if (tag == Tag::PBC) {
            std::array<std::uint8_t,3> flags{};
            if (!read_span(is,flags.data(),flags.size())) return false;
            for (int d=0;d<3;++d) {
                if (flags[d]>1) return false;
                atom.pbc_[d]=flags[d]!=0;
            }
        } else if (tag == Tag::Energy) {
            if (!read_value(is, atom.energy_)) return false;
            atom.have_energy = true;
        } else if (tag == Tag::Stress) {
            if (!read_span(is, atom.stress_.data(), 9)) return false;
            atom.have_stress = true;
        } else if (tag == Tag::Feature) {
            std::string name, value;
            if (!read_named_string(is, name, value)) return false;
            if (HVPIO::is_key(name)) HVPIO::read(atom, name, value);
            else if (name=="hessian" && HessianIO::read_feature(atom,value)) {}
            else atom.features_[name] = value;
        } else if (tag == Tag::End) {
            break;
        } else if (tag == Tag::LocalDouble) {
            std::string name;
            if (!read_string(is, name)) return false;
            auto &values = atom.local_attribs_d_[name];
            values.resize(natoms);
            if (!read_span(is, values.data(), values.size())) return false;
        } else if (tag == Tag::LocalDoubleVector) {
            std::string name;
            std::int32_t rows = 0;
            if (!read_string(is, name) || !read_value(is, rows)) return false;
            if (rows < 0 || rows > MAX_VECTOR_SIZE ||
                (natoms > 0 && rows > MAX_VECTOR_SIZE / natoms)) return false;
            auto &values = atom.local_attribs_v_[name];
            values.resize(rows, natoms);
            if (!read_span(is, values.data(), values.size())) return false;
        } else if (tag == Tag::GlobalDouble) {
            std::string name;
            double value = 0.0;
            if (!read_string(is, name) || !read_value(is, value)) return false;
            atom.global_attribs_d_[name] = value;
        } else if (tag == Tag::GlobalDoubleVector) {
            std::string name;
            if (!read_string(is, name)) return false;
            if (!read_vector(is, atom.global_attribs_v_[name])) return false;
        } else if (tag == Tag::GlobalInt) {
            std::string name;
            int value = 0;
            if (!read_string(is, name) || !read_value(is, value)) return false;
            atom.global_attribs_i_[name] = value;
        } else if (tag == Tag::GlobalIntVector) {
            std::string name;
            if (!read_string(is, name)) return false;
            if (!read_vector(is, atom.global_attribs_vi_[name])) return false;
        } else if (tag == Tag::LocalInt) {
            std::string name;
            if (!read_string(is, name)) return false;
            auto &values = atom.local_attribs_i_[name];
            values.resize(natoms);
            if (!read_span(is, values.data(), values.size())) return false;
        } else if (tag == Tag::LocalIntVector) {
            std::string name;
            std::int32_t rows = 0;
            if (!read_string(is, name) || !read_value(is, rows)) return false;
            if (rows < 0 || rows > MAX_VECTOR_SIZE ||
                (natoms > 0 && rows > MAX_VECTOR_SIZE / natoms)) return false;
            auto &values = atom.local_attribs_vi_[name];
            values.resize(rows, natoms);
            if (!read_span(is, values.data(), values.size())) return false;
        } else {
            std::cerr << "Unsupported binary mlpcfg tag " << static_cast<char>(tag) << std::endl;
            return false;
        }
    }

    if (!has_types)
        atom.chemmap_.resize(natoms, 0);
    has_types = true;

    if (!(has_lattice && has_pos)) {
        std::cerr << "Error: file contains no info about structures" << std::endl;
        return false;
    }
    // Version 2 had no PBC field and historically represented periodic cells.
    if (format_version==2) atom.pbc_={{true,true,true}};
    atom.validate_hvp();
    if (absolute_index_read) {
        for (int type : atom.chemmap_)
            if (type < 0 || static_cast<std::size_t>(type) >= std::size(Chemicals::chemical_symbols))
                return false;
    }
    output = std::move(atom);
    return true;
}
// add pbc flag in Atoms class
bool GPIO::load_next_from_mlpcfg(std::istream &is, Atoms &atom, bool *clean_eof) {
    // atom.resetEFS();
    atom.clear();
    if (clean_eof) *clean_eof = false;
    // 1. seek BEGIN_CFG line
    bool found_begin = false;
    while (std::getline(is, line_buffer)) {
        if (line_buffer.find("BEGIN_CFG") != std::string::npos) {
            found_begin = true;
            break;
        }
    }
    if (!found_begin) {
        if (clean_eof) *clean_eof = is.eof() && !is.bad();
        return false;
    }
    // 2. Size
    if (!std::getline(is, line_buffer) || !std::getline(is, line_buffer)) return false;
    int natoms = 0;
    {
        GPIO_LineTokenizer tokenizer{line_buffer};
        if (!tokenizer.next_numeric(natoms) || natoms < 0 || natoms > gpio_binary::MAX_ATOMS)
            return false;
    }
    atom.resize(natoms);
    // 3. Supercell
    bool have_cell = false;
    if (!std::getline(is, line_buffer)) return false;
    if (line_buffer.find("Supercell") != std::string::npos) {
        for (int a = 0; a < 3; ++a) {
            if (!std::getline(is, line_buffer)) return false;
            GPIO_LineTokenizer tokenizer{line_buffer};
            if (!tokenizer.next_numeric(atom.cell_(a, 0)) ||
                !tokenizer.next_numeric(atom.cell_(a, 1)) ||
                !tokenizer.next_numeric(atom.cell_(a, 2)) ||
                !atom.cell_.row(a).allFinite()) return false;
        }
    } else {
        goto lb_cfgreadtext_data;
    }
    if (!std::getline(is, line_buffer)) return false;
lb_cfgreadtext_data:
    // 4. AtomData:
    // CFG frames may have different optional columns (for example forces).
    // Parse every AtomData header, including when reusing GPIO across files.
    {
        GPIO_LineTokenizer tokenizer{line_buffer};
        auto tag = tokenizer.next_token();
        if (tag != "AtomData:") return false;
        if (!lmp_parse_tags(tokenizer, LMP_TAG_SITEEN | LMP_TAG_MAGMOM | LMP_TAG_CARTES_ALIAS))
            return false;
        lmp_tag_phrased = true;
    }
    if (lmp_tag_flags.have_force()) {
        atom.forces_.resize(natoms);
    }
    if (lmp_tag_flags.have_velo()) {
        atom.velocities_.resize(natoms);
    }
    if (lmp_tag_flags.has_siteen) {
        atom.site_energies_.resize(natoms);
    }
    if (lmp_tag_flags.has_q) {
        atom.charges_.resize(natoms);
    }
    if (lmp_tag_flags.has_mx || lmp_tag_flags.has_my || lmp_tag_flags.has_mz) {
        atom.magmoms_.resize(natoms);
    }
    for (auto &it : lmp_tagmap) {
        atom.local_attribs_d_[it.first].resize(natoms);
    }
    lmp_readbuffer.resize(lmp_tag_flags.nvals);
    std::vector<bool> seen(static_cast<size_t>(natoms), false);
    for (int i = 0; i < natoms; ++i) {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        // assume less than 2^53 atoms
        int id = i;
        for (int l = 0; l < lmp_tag_flags.nvals; ++l) {
            if (!toks.next_numeric(lmp_readbuffer[l]) || !std::isfinite(lmp_readbuffer[l])) return false;
        }
        // id type x y z fx fy fz vx vy vz ix iy iz mol q
        // 0  1    2 3 4 5  6  7  8  9  10 11 12 13 14  15
        if (lmp_tag_flags.has_id) {
            if (!parse_atom_id(lmp_readbuffer[lmp_common_tagid[0]], natoms, seen, id)) return false;
        } else {
            seen[id] = true;
        }
        if (lmp_tag_flags.has_type) {
            const double type = lmp_readbuffer[lmp_common_tagid[1]];
            if (type < 0 || type > std::numeric_limits<int>::max() || std::trunc(type) != type)
                return false;
            atom.chemmap_[id] = static_cast<int>(type);
        } else
            atom.chemmap_[id] = 0;
        // assume must have x y z
        atom.pos_[id](0) = lmp_readbuffer[lmp_common_tagid[2]];
        atom.pos_[id](1) = lmp_readbuffer[lmp_common_tagid[3]];
        atom.pos_[id](2) = lmp_readbuffer[lmp_common_tagid[4]];
        // forces
        if (lmp_tag_flags.has_fx) {
            atom.forces_[id](0) = lmp_readbuffer[lmp_common_tagid[5]];
        } 
        if (lmp_tag_flags.has_fy) {
            atom.forces_[id](1) = lmp_readbuffer[lmp_common_tagid[6]];
        } 
        if (lmp_tag_flags.has_fz) {
            atom.forces_[id](2) = lmp_readbuffer[lmp_common_tagid[7]];
        } 
        // velocities
        if (lmp_tag_flags.has_vx) {
            atom.velocities_[id](0) = lmp_readbuffer[lmp_common_tagid[8]];
        } 
        if (lmp_tag_flags.has_vy) {
            atom.velocities_[id](1) = lmp_readbuffer[lmp_common_tagid[9]];
        } 
        if (lmp_tag_flags.has_vz) {
            atom.velocities_[id](2) = lmp_readbuffer[lmp_common_tagid[10]];
        } 
        // charge
        if (lmp_tag_flags.has_q) {
            atom.charges_[id] = lmp_readbuffer[lmp_common_tagid[15]];
        }
        // site_en
        if (lmp_tag_flags.has_siteen) {
            atom.site_energies_[id] = lmp_readbuffer[lmp_common_tagid[20]];
        }
        // magmom
        if (lmp_tag_flags.has_mx) {
            atom.magmoms_[id](0) = lmp_readbuffer[lmp_common_tagid[16]];
        }
        if (lmp_tag_flags.has_my) {
            atom.magmoms_[id](1) = lmp_readbuffer[lmp_common_tagid[17]];
        }
        if (lmp_tag_flags.has_mz) {
            atom.magmoms_[id](2) = lmp_readbuffer[lmp_common_tagid[18]];
        }
        // other mapped scalar tags
        for (auto &kv : lmp_tagmap) {
            const auto &key = kv.first;
            int pos = kv.second;
            atom.local_attribs_d_[key][id] = lmp_readbuffer[pos];
        }
    }
    
    bool found_end = false;
    while (std::getline(is, line_buffer)) {
        GPIO_LineTokenizer tokenizer{line_buffer};
        auto tok = tokenizer.next_token();
        if (tok == "END_CFG") {
            found_end = true;
            break;
        }
        if (tok == "Energy" || tok == "energy") {
            if (!std::getline(is, line_buffer)) return false;
            GPIO_LineTokenizer tokenizer{line_buffer};
            if (!tokenizer.next_numeric(atom.energy_) || !std::isfinite(atom.energy_)) return false;
            atom.have_energy = true;
        } else if (tok == "PlusStress:" || tok == "plusstress:" || tok == "Stress:" || tok == "stress:") {
            if (!std::getline(is, line_buffer)) return false;
            GPIO_LineTokenizer tokenizer{line_buffer};
            if (!tokenizer.next_numeric(atom.stress_(0, 0)) ||
                !tokenizer.next_numeric(atom.stress_(1, 1)) ||
                !tokenizer.next_numeric(atom.stress_(2, 2)) ||
                !tokenizer.next_numeric(atom.stress_(1, 2)) ||
                !tokenizer.next_numeric(atom.stress_(0, 2)) ||
                !tokenizer.next_numeric(atom.stress_(0, 1)) || !atom.stress_.allFinite()) return false;
            atom.stress_(2, 1) = atom.stress_(1, 2);
            atom.stress_(2, 0) = atom.stress_(0, 2);
            atom.stress_(1, 0) = atom.stress_(0, 1);
            atom.have_stress = true;
        } else if (tok == "Feature" || tok == "feature") {
            auto tag = tokenizer.next_token();
            auto val = tokenizer.next_token();
            if (tag.empty() || val.empty()) return false;
            const std::string_view remainder(val.data(),
                line_buffer.data() + line_buffer.size() - val.data());
            if (HVPIO::is_key(tag)) {
                HVPIO::read(atom, tag, remainder);
            } else if (tag=="hessian" && HessianIO::read_feature(atom,remainder)) {
                // Complete finite matrices promote; arbitrary legacy text stays metadata.
            } else if (tag == "COLLINEAR") {
                if (val == "1" || val[0] == 't' || val[0] == 'T') {
                    atom.collinear_ = true;
                }
            } else if (tag == "Fitness") {
                if (!parse_number(val, atom.fitness)) return false;
                atom.global_attribs_d_["fitness"] = atom.fitness;
            } else if (tag == "MV_grade") {
                if (!parse_number(val, atom.mvgrade_)) return false;
            } else if (tag == "fenergy") {
                double fenergy;
                if (parse_number(val, fenergy) && std::isfinite(fenergy) &&
                    tokenizer.next_token().empty()) {
                    atom.fenergy_ = fenergy;
                    atom.have_fenergy = true;
                } else {
                    atom.features_[std::string(tag)] = std::string(remainder);
                }
            } else if (tag == "total_charge") {
                double total_charge;
                // A legacy Feature may contain arbitrary text. Promote only
                // a complete finite scalar, allowing trailing whitespace.
                if (parse_number(val, total_charge) && std::isfinite(total_charge) &&
                    tokenizer.next_token().empty())
                    atom.global_attribs_d_["total_charge"] = total_charge;
                else
                    atom.features_[std::string(tag)] = std::string(remainder);
            } else {
                atom.features_[std::string(tag)] = std::string(remainder);
            }
        }
    }
    if (found_end) atom.validate_hvp();
    return found_end;
}

bool GPIO::read_mlpcfg_bin_sequence(std::istream &is, Atoms &atom) {
    return load_next_from_mlpcfg_bin(is, atom);
}

bool GPIO::read_mlpcfg_sequence(std::istream &is, Atoms &atom) {
    return load_next_from_mlpcfg(is, atom);
}
