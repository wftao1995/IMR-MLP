// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpio.h"
#include "gpio_binary.h"
#include "gpio_tokenizer.h"
#include "gpio_lmp.h"
#include "tools/chem.h"
#include "tools/procstr.h"
#include "tools/tools.h"
#include "tools/constants.h"

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

} // namespace
/****
 * Read lammps traj files
 *
 * TODO: read_specific to specify types, ie. read xv, fx, yus by set ["xv",
 *              "fx", "yus"]
 *
 * Currently, the only supported format is :
 * ITEM: TIMESTEP
 * timestep
 * ITEM: NUMBER OF ATOMS
 * natoms
 * ITEM: BOX BOUNDS xy xz yz pp pp pp
 * 3 lines
 * ITEM: ATOMS id type xs ys zs
 *
 * for positions, 
 * xs, ys, zs: scaled positions
 * x, y, z: unscaled coordinates
 * xu, yu, zu, unscaled and unwrapped
 * xsu, ysu, zsu: scaled and unwrapped
 * 
 * read tags: {x,y,z}{,s,u,su}, f{x,y,z}, v{x,y,z}, 
 *      i{x,y,z}(when present with x{,s}, will use this to unwrap positions)
 * 
 * TODO: hdf5
 * TODO: q, radius, diameter, element and mass
 * TODO: same element may have different element tag and vice versa
 *****/

// TODO: add a read_header, and read one frame first, then move pointer back to file start
bool GPIO::load_next_from_lmptrj(std::istream& is, Atoms& atom)
{
    // load next
    atom.clear();
    atom.cell_(0, 1) = 0;
    atom.cell_(0, 2) = 0;
    atom.cell_(1, 2) = 0;
    double tilt[3];
    int natoms = 0;
    int nval;
    // read until ITEM: TIMESTEP
    while (std::getline(is, line_buffer)) {
        if (line_buffer == "ITEM: TIMESTEP") {
            break;
        }
    }
    if (is.eof()) return false;
    // timestep value
    std::getline(is, line_buffer);
    // ITEM: NUMBER OF ATOMS
    std::getline(is, line_buffer);
    if (line_buffer != "ITEM: NUMBER OF ATOMS") return false;
    std::getline(is, line_buffer);
    auto [ptr, ec] = std::from_chars(line_buffer.data(), line_buffer.data() + line_buffer.size(), natoms);
    if (ec != std::errc() || ptr != line_buffer.data() + line_buffer.size() ||
        natoms < 0 || natoms > gpio_binary::MAX_ATOMS) return false;
    atom.pos_.resize(natoms);
    atom.chemmap_.resize(natoms);
    // box
    std::getline(is, line_buffer);
    if (line_buffer.substr(0, 9) == "ITEM: BOX") {
        bool trinic = false;
        for (int i = 0; i < 3; ++i) {
            std::getline(is, line_buffer);
            double x1, x2, x3;
            GPIO_LineTokenizer toks{line_buffer};
            if (!toks.next_numeric(x1) || !toks.next_numeric(x2)) return false;
            auto tok = toks.next_token();
            if (tok.empty()) {
                x3 = 0;
            } else {
                const auto result = fast_float::from_chars(tok.data(), tok.data() + tok.size(), x3);
                if (result.ec != std::errc{} || result.ptr != tok.data() + tok.size()) return false;
                trinic = true;
            }
            atom.cell_(i, i) = x2 - x1;
            tilt[i] = x3;
        }
        atom.cell_(1, 0) = tilt[0];
        atom.cell_(2, 0) = tilt[1];
        atom.cell_(2, 1) = tilt[2];
        if (trinic) {
            atom.cell_(0, 0) -= fabs(tilt[0]) + fabs(tilt[1]);
            atom.cell_(1, 1) -= fabs(tilt[2]);
        }
    }
    // ITEM: ATOMS tags
    std::getline(is, line_buffer);
    if (line_buffer.substr(0, 11) != "ITEM: ATOMS") return false;
    if (!lmp_tag_phrased || lmp_phrase_tag_always)
    {
        GPIO_LineTokenizer tok{line_buffer};
        tok.next_token(); // ITEM:
        tok.next_token(); // ATOMS
        if (!lmp_parse_tags(tok, LMP_TAG_SCALED_COORDS | LMP_TAG_IMAGE_FLAGS |
                                LMP_TAG_MOL | LMP_TAG_SITEEN | LMP_TAG_BRACKET_VEC |
                                LMP_TAG_GUARDED_COORDS | LMP_TAG_INIT_VECMAPS)) return false;
        lmp_tag_phrased = true;
    }

    if (lmp_tag_flags.have_force()) {
        atom.forces_.resize(natoms);
    }
    if (lmp_tag_flags.have_velo()) {
        atom.velocities_.resize(natoms);
    }
    if (lmp_tag_flags.has_mol) {
        atom.molid_.resize(natoms);
    }
    if (lmp_tag_flags.has_siteen) {
        atom.site_energies_.resize(natoms);
    }
    if (lmp_tag_flags.has_q) {
        atom.charges_.resize(natoms);
    }
    for (auto &it : lmp_tagmap) {
        atom.local_attribs_d_[it.first].resize(natoms);
    }
    for (auto &it : lmp_tagvecsize) {
        atom.alloc_lvd(it.first, it.second);
    }
    lmp_readbuffer.resize(lmp_tag_flags.nvals);
    std::vector<bool> seen(static_cast<size_t>(natoms), false);
    for (int i = 0; i < natoms; ++i) {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        // assume less than 2^53 atoms
        int id = i;
        for (int l = 0; l < lmp_tag_flags.nvals; ++l) {
            if (!toks.next_numeric(lmp_readbuffer[l])) return false;
        }
        // id type x y z fx fy fz vx vy vz ix iy iz mol q
        // 0  1    2 3 4 5  6  7  8  9  10 11 12 13 14  15
        if (lmp_tag_flags.has_id) {
            if (!parse_atom_id(lmp_readbuffer[lmp_common_tagid[0]], natoms, seen, id)) return false;
        } else {
            seen[id] = true;
        }
        if (lmp_tag_flags.has_type)
            atom.chemmap_[id] = lmp_readbuffer[lmp_common_tagid[1]] - 1;
        else 
            atom.chemmap_[id] = 0;
        // assume must have x y z
        atom.pos_[id](0) = lmp_readbuffer[lmp_common_tagid[2]];
        atom.pos_[id](1) = lmp_readbuffer[lmp_common_tagid[3]];
        atom.pos_[id](2) = lmp_readbuffer[lmp_common_tagid[4]];
        // image flags / unwrap indices
        int ix = 0, iy = 0, iz = 0;
        if (lmp_tag_flags.has_ix) {
            ix = lmp_readbuffer[lmp_common_tagid[11]];
        }
        if (lmp_tag_flags.has_iy) {
            iy = lmp_readbuffer[lmp_common_tagid[12]];
        }
        if (lmp_tag_flags.has_iz) {
            iz = lmp_readbuffer[lmp_common_tagid[13]];
        }
        // determine if coordinates are scaled (xs/xsu etc.), assume x, y, z are in same format
        if (lmp_tag_flags.wix) {
            atom.pos_[id](0) += ix;
            atom.pos_[id](1) += iy;
            atom.pos_[id](2) += iz;
            atom.pos_[id] *= atom.cell_;
        } else {
            atom.pos_[id] += ix * atom.cell_.row(0) + iy * atom.cell_.row(1) + iz * atom.cell_.row(2);
        }
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
        // mol id
        if (lmp_tag_flags.has_mol) {
            atom.molid_[id] = lmp_readbuffer[lmp_common_tagid[14]];
        }
        // site_en
        if (lmp_tag_flags.has_siteen) {
            atom.site_energies_[id] = lmp_readbuffer[lmp_common_tagid[20]];
        }
        // charge
        if (lmp_tag_flags.has_q) {
            atom.charges_[id] = lmp_readbuffer[lmp_common_tagid[15]];
        }
        // other mapped scalar tags
        for (auto &kv : lmp_tagmap) {
            const auto &key = kv.first;
            int pos = kv.second;
            atom.local_attribs_d_[key][id] = lmp_readbuffer[pos];
        }
        // other mapped vector tags
        for (auto &kv : lmp_tagvecmap) {
            for (int ll = 0; ll < lmp_tagvecsize[kv.first]; ++ll) {
                int pos = kv.second[ll];
                if (pos >= 0)
                    atom.local_attribs_v_[kv.first](ll, id) = lmp_readbuffer[pos];
                else 
                    atom.local_attribs_v_[kv.first](ll, id) = 0;
            }
        }
    }
    return true;
}

bool GPIO::read_lammps_trj_sequence(std::istream &is, Atoms &atom) {
    if (is.eof()) {
        return false;
    }
    return load_next_from_lmptrj(is, atom);
}

bool GPIO::load_next_from_lmpyaml(std::istream &is, Atoms &atom) {
    if (is.eof()) return false;
    atom.clear();
    atom.have_fenergy = false;
    atom.have_energy = false;
    atom.have_stress = false;
    constexpr int press_state_all_true = 0x3f;
    int press_flag = 0;
    atom.cell_.setZero();
    double tilt[3];
    int natoms = 0;
    int nval;

    if (!std::getline(is, line_buffer)) return false;
    // first line should always be '---'
    if (line_buffer != "---") return false;
    // creator, skip this line
    if (!std::getline(is, line_buffer)) return false;
    // timestep 
    if (!std::getline(is, line_buffer)) return false;
    // natoms 
    if (!std::getline(is, line_buffer)) return false;
    {
        GPIO_LineTokenizer<WS_LmpYaml> toks{line_buffer};
        auto t1 = toks.next_token();
        if (t1 != "natoms:") return false;
        if (!toks.next_numeric(natoms) || natoms < 0 || natoms > gpio_binary::MAX_ATOMS)
            return false;
    }
    atom.pos_.resize(natoms);
    atom.chemmap_.resize(natoms);

    // boundary: TODO non-pbc
    if (!std::getline(is, line_buffer)) return false;
    // thermo, may not exist 
    if (!std::getline(is, line_buffer)) return false;
    if (line_buffer == "thermo:") {
        // keywords
        std::unordered_map<std::string, std::vector<double>> ketvals;
        if (!std::getline(is, line_buffer)) return false; // keywords
        if (!std::getline(is, line_buffer2)) return false; // data        
        GPIO_LineTokenizer<WS_LmpYaml> toks{line_buffer};
        GPIO_LineTokenizer<WS_LmpYaml> toks2{line_buffer2};
        auto tag = toks.next_token(); // -
        auto val = toks2.next_token(); // -
        tag = toks.next_token(); // keywords:
        val = toks2.next_token(); // data:
        // phrase
        tag = toks.next_token();
        double tmpdval;
        if (!toks2.next_numeric(tmpdval)) return false;
        while (!tag.empty()) {
            if (tag == "Enthalpy") {
                atom.have_fenergy = true;
                atom.fenergy_ = tmpdval;
            } else if (tag == "PotEng") {
                atom.have_energy = true;
                atom.energy_ = tmpdval;
            } else if (!tag.empty() && tag.back() == ']') {
                const size_t bracket_open = tag.rfind('[');
                if (bracket_open == std::string_view::npos || bracket_open == 0) return false;
                const auto number_str = tag.substr(bracket_open + 1, tag.size() - bracket_open - 2);
                int number = 0;
                const auto result = std::from_chars(number_str.data(), number_str.data() + number_str.size(), number);
                if (result.ec != std::errc{} || result.ptr != number_str.data() + number_str.size() || number <= 0)
                    return false;
                const std::string name(tag.substr(0, bracket_open));
                if (static_cast<int>(ketvals[name].size()) < number)
                    ketvals[name].resize(number);
                ketvals[name][number - 1] = tmpdval;
            } else {
                atom.global_attribs_d_[std::string(tag)] = tmpdval;
            }
            tag = toks.next_token();
            if (!tag.empty() && !toks2.next_numeric(tmpdval)) return false;
        }
        // push ketvals to atom
        for (auto &it : ketvals) {
            atom.global_attribs_v_[it.first] = Eigen::Map<const Eigen::VectorXd>(it.second.data(), it.second.size());
        }
    } else {
        goto lb_lmp_yaml_box;
    }
    // box
    if (!std::getline(is, line_buffer)) return false;
    if (line_buffer != "box:") {
        std::cerr << "Error: except box: keyword" << std::endl;
        return false;
    }
lb_lmp_yaml_box:
    // xlo xhi
    if (!std::getline(is, line_buffer)) return false;
    {
        GPIO_LineTokenizer<WS_LmpYaml> toks{line_buffer};
        double xlo, xhi;
        auto tok = toks.next_token(); // -
        if (!toks.next_numeric(xlo) || !toks.next_numeric(xhi)) return false;
        atom.cell_(0, 0) = xhi - xlo;
    }
    // ylo yhi
    if (!std::getline(is, line_buffer)) return false;
    {
        GPIO_LineTokenizer<WS_LmpYaml> toks{line_buffer};
        double xlo, xhi;
        auto tok = toks.next_token(); // -
        if (!toks.next_numeric(xlo) || !toks.next_numeric(xhi)) return false;
        atom.cell_(1, 1) = xhi - xlo;
    }
    // zlo zhi
    if (!std::getline(is, line_buffer)) return false;
    {
        GPIO_LineTokenizer<WS_LmpYaml> toks{line_buffer};
        double xlo, xhi;
        auto tok = toks.next_token(); // -
        if (!toks.next_numeric(xlo) || !toks.next_numeric(xhi)) return false;
        atom.cell_(2, 2) = xhi - xlo;
    }
    // check if there exists xy xz yz
    if (!std::getline(is, line_buffer)) return false;
    {
        GPIO_LineTokenizer<WS_LmpYaml> toks{line_buffer};
        auto tok = toks.next_token();
        if (tok == "keywords:") {
            goto lb_lmp_yaml_keywords;
        }
        if (!toks.next_numeric(tilt[0]) || !toks.next_numeric(tilt[1]) ||
            !toks.next_numeric(tilt[2])) return false;
        atom.cell_(1, 0) = tilt[0];
        atom.cell_(2, 0) = tilt[1];
        atom.cell_(2, 1) = tilt[2];
        atom.cell_(0, 0) -= fabs(tilt[0]) + fabs(tilt[1]);
        atom.cell_(1, 1) -= fabs(tilt[2]);
    }
    // keywords
    if (!std::getline(is, line_buffer)) return false;
lb_lmp_yaml_keywords:
    if (!lmp_tag_phrased || lmp_phrase_tag_always)
    {
        GPIO_LineTokenizer<WS_LmpYaml> tok{line_buffer};
        tok.next_token(); // keywords
        if (!lmp_parse_tags(tok, LMP_TAG_SCALED_COORDS | LMP_TAG_IMAGE_FLAGS |
                                LMP_TAG_MOL | LMP_TAG_BRACKET_VEC |
                                LMP_TAG_GUARDED_COORDS | LMP_TAG_INIT_VECMAPS)) return false;
        lmp_tag_phrased = true;
    }
    // data line
    if (!std::getline(is, line_buffer)) return false;
    if (lmp_tag_flags.have_force()) {
        atom.forces_.resize(natoms);
    }
    if (lmp_tag_flags.have_velo()) {
        atom.velocities_.resize(natoms);
    }
    if (lmp_tag_flags.has_mol) {
        atom.molid_.resize(natoms);
    }
    if (lmp_tag_flags.has_q) {
        atom.charges_.resize(natoms);
    }
    for (auto &it : lmp_tagmap) {
        atom.local_attribs_d_[it.first].resize(natoms);
    }
    for (auto &it : lmp_tagvecsize) {
        atom.alloc_lvd(it.first, it.second);
    }
    lmp_readbuffer.resize(lmp_tag_flags.nvals);
    std::vector<bool> seen(static_cast<size_t>(natoms), false);
    for (int i = 0; i < natoms; ++i) {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer<WS_LmpYaml> toks{line_buffer};
        toks.next_token(); // skip first -
        // assume less than 2^53 atoms
        int id = i;
        for (int l = 0; l < lmp_tag_flags.nvals; ++l) {
            if (!toks.next_numeric(lmp_readbuffer[l])) return false;
        }
        // id type x y z fx fy fz vx vy vz ix iy iz mol q
        // 0  1    2 3 4 5  6  7  8  9  10 11 12 13 14  15
        if (lmp_tag_flags.has_id) {
            if (!parse_atom_id(lmp_readbuffer[lmp_common_tagid[0]], natoms, seen, id)) return false;
        } else {
            seen[id] = true;
        }
        if (lmp_tag_flags.has_type)
            atom.chemmap_[id] = lmp_readbuffer[lmp_common_tagid[1]] - 1;
        else 
            atom.chemmap_[id] = 0;
        // assume must have x y z
        atom.pos_[id](0) = lmp_readbuffer[lmp_common_tagid[2]];
        atom.pos_[id](1) = lmp_readbuffer[lmp_common_tagid[3]];
        atom.pos_[id](2) = lmp_readbuffer[lmp_common_tagid[4]];
        // image flags / unwrap indices
        int ix = 0, iy = 0, iz = 0;
        if (lmp_tag_flags.has_ix) {
            ix = lmp_readbuffer[lmp_common_tagid[11]];
        }
        if (lmp_tag_flags.has_iy) {
            iy = lmp_readbuffer[lmp_common_tagid[12]];
        }
        if (lmp_tag_flags.has_iz) {
            iz = lmp_readbuffer[lmp_common_tagid[13]];
        }
        // determine if coordinates are scaled (xs/xsu etc.), assume x, y, z are in same format
        if (lmp_tag_flags.wix) {
            atom.pos_[id](0) += ix;
            atom.pos_[id](1) += iy;
            atom.pos_[id](2) += iz;
            atom.pos_[id] *= atom.cell_;
        } else {
            atom.pos_[id] += ix * atom.cell_.row(0) + iy * atom.cell_.row(1) + iz * atom.cell_.row(2);
        }
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
        // mol id
        if (lmp_tag_flags.has_mol) {
            atom.molid_[id] = lmp_readbuffer[lmp_common_tagid[14]];
        }
        // charge
        if (lmp_tag_flags.has_q) {
            atom.charges_[id] = lmp_readbuffer[lmp_common_tagid[15]];
        }
        // other mapped scalar tags
        for (auto &kv : lmp_tagmap) {
            const auto &key = kv.first;
            int pos = kv.second;
            atom.local_attribs_d_[key][id] = lmp_readbuffer[pos];
        }
        // other mapped vector tags
        for (auto &kv : lmp_tagvecmap) {
            for (int ll = 0; ll < lmp_tagvecsize[kv.first]; ++ll) {
                int pos = kv.second[ll];
                if (pos >= 0)
                    atom.local_attribs_v_[kv.first](ll, id) = lmp_readbuffer[pos];
                else 
                    atom.local_attribs_v_[kv.first](ll, id) = 0;
            }
        }
    }
    // LAST --- LINE
    std::getline(is, line_buffer);
    return true;
}

bool GPIO::read_lammps_yaml_sequence(std::istream &is, Atoms &atom) {
    if (is.eof()) {
        return false;
    }
    return load_next_from_lmpyaml(is, atom);
}
// this only read basic data file, not for charged/full/spin/... etc
bool GPIO::read_lammps_dat(std::istream &is, Atoms &atom) { 
    const auto is_empty_line = [](const std::string& line) noexcept -> bool {
        for (char c : line) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                continue;
            }
            return (c == '#');
        }
        return true;
    };
    if (!std::getline(is, line_buffer)) return false;
    while (is_empty_line(line_buffer)) {
        if (!std::getline(is, line_buffer)) return false;
    }
    // GPIO_Tokenizer tokenizer;
    int natoms = 0, ntypes;
    // run untill reach "Atoms # [style]" line
    while (!is.eof()) {
        GPIO_LineTokenizer tokenizer{line_buffer};
        if (line_buffer.find(" atoms") != std::string::npos) {
            tokenizer.next_numeric_unsafe(natoms);
            atom.resize(natoms);
        // } else if (line_buffer.find(" atom types") != std::string::npos) {
        //     tokenizer.read_next_int(ntypes);
        } else if (line_buffer.find("xlo") != std::string::npos) {
            double xlo, xhi;
            tokenizer.next_numeric_unsafe(xlo);
            tokenizer.next_numeric_unsafe(xhi);
            atom.cell_(0, 0) = xhi - xlo;
        } else if (line_buffer.find("ylo") != std::string::npos) {
            double xlo, xhi;
            tokenizer.next_numeric_unsafe(xlo);
            tokenizer.next_numeric_unsafe(xhi);
            atom.cell_(1, 1) = xhi - xlo;
        } else if (line_buffer.find("zlo") != std::string::npos) {
            double xlo, xhi;
            tokenizer.next_numeric_unsafe(xlo);
            tokenizer.next_numeric_unsafe(xhi);
            atom.cell_(2, 2) = xhi - xlo;
        } else if (line_buffer.find("xy") != std::string::npos) {
            double xy, xz, yz;
            tokenizer.next_numeric_unsafe(xy);
            tokenizer.next_numeric_unsafe(xz);
            tokenizer.next_numeric_unsafe(yz);
            // Data files store true restricted-triclinic box limits.  Only
            // dump files use tilt-expanded bounding limits.
            atom.cell_(1, 0) = xy;
            atom.cell_(2, 0) = xz;
            atom.cell_(2, 1) = yz;
        } else if (line_buffer.find("Atoms") != std::string::npos) {
            break;
        }
        if (!std::getline(is, line_buffer)) return false;
    }
    if (natoms == 0) {
        std::cerr << "No atoms info found, please check the input source or check the code" << std::endl;
        return false;
    }
    std::string style = "";
    {
        GPIO_LineTokenizer tokenizer{line_buffer};
        tokenizer.next_token();
        tokenizer.next_token();
        style = std::string(tokenizer.next_token());
    }
    // skip empty lines
    if (!std::getline(is, line_buffer)) return false;
    while (is_empty_line(line_buffer)) {
        if (!std::getline(is, line_buffer)) return false;
    }
    // read atoms
    if (style == "atomic" || style.empty()) {
        // first line will process manually since we can not make sure stream support seek
        {
            GPIO_LineTokenizer tokenizer{line_buffer};
            int id, type;
            tokenizer.next_numeric_unsafe(id);
            tokenizer.next_numeric_unsafe(type);
            id -= 1;
            atom.chemmap_[id] = type - 1;
            tokenizer.next_numeric_unsafe(atom.pos_[id](0));
            tokenizer.next_numeric_unsafe(atom.pos_[id](1));
            tokenizer.next_numeric_unsafe(atom.pos_[id](2));
        }
        for (int i = 1; i < natoms; ++i) {
            if (!std::getline(is, line_buffer)) return false;
            GPIO_LineTokenizer tokenizer{line_buffer};
            int id, type;
            tokenizer.next_numeric_unsafe(id);
            tokenizer.next_numeric_unsafe(type);
            id -= 1;
            atom.chemmap_[id] = type - 1;
            tokenizer.next_numeric_unsafe(atom.pos_[id](0));
            tokenizer.next_numeric_unsafe(atom.pos_[id](1));
            tokenizer.next_numeric_unsafe(atom.pos_[id](2));
        }
    } else {
        std::cerr << "ERROR: style " << style << " is currently not supported" << std::endl;
        return false;
    }
    return true;
}
