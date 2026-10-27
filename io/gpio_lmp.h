// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

/**
 * @file gpio_lmp.h
 * @brief Template implementation of GPIO::lmp_parse_tags.
 *
 * Extracted from gpio.h — only gpio_read_lmp.cpp and gpio_read_cfg.cpp
 * need to include this, reducing recompilation of 60+ unrelated TUs.
 *
 * Must be included AFTER gpio.h (uses GPIO private members).
 */

// Template implementation: unified LAMMPS/CFG tag parsing.
// Must be in header so both gpio_read_lmp.cpp and gpio_read_cfg.cpp can instantiate.
template<typename Tok>
bool GPIO::lmp_parse_tags(Tok &tok, unsigned flags) {
    lmp_tagmap.clear();
    lmp_common_tagid.fill(-1);
    lmp_tag_flags.reset();
    if (flags & LMP_TAG_INIT_VECMAPS) {
        lmp_tagvecmap.clear();
        lmp_tagvecsize.clear();
    }
    auto tag = tok.next_token();
    int itag = 0;
    while (!tag.empty()) {
        // id, type, q — always present
        if (tag == "id") {
            lmp_tag_flags.has_id = true;
            lmp_common_tagid[0] = itag;
            ++lmp_tag_flags.nvals;
        } else if (tag == "type") {
            lmp_tag_flags.has_type = true;
            lmp_common_tagid[1] = itag;
            ++lmp_tag_flags.nvals;
        } else if (tag == "q") {
            lmp_tag_flags.has_q = true;
            lmp_common_tagid[15] = itag;
            ++lmp_tag_flags.nvals;
        }
        // unscaled coordinates: x/xu or x/cartes_x
        else if (tag == "x" || ((flags & LMP_TAG_CARTES_ALIAS) ? tag == "cartes_x" : tag == "xu")) {
            if (!(flags & LMP_TAG_GUARDED_COORDS) || !lmp_tag_flags.has_x) {
                lmp_common_tagid[2] = itag;
                lmp_tag_flags.has_x = true;
                ++lmp_tag_flags.nvals;
            }
        } else if (tag == "y" || ((flags & LMP_TAG_CARTES_ALIAS) ? tag == "cartes_y" : tag == "yu")) {
            if (!(flags & LMP_TAG_GUARDED_COORDS) || !lmp_tag_flags.has_y) {
                lmp_common_tagid[3] = itag;
                lmp_tag_flags.has_y = true;
                ++lmp_tag_flags.nvals;
            }
        } else if (tag == "z" || ((flags & LMP_TAG_CARTES_ALIAS) ? tag == "cartes_z" : tag == "zu")) {
            if (!(flags & LMP_TAG_GUARDED_COORDS) || !lmp_tag_flags.has_z) {
                lmp_common_tagid[4] = itag;
                lmp_tag_flags.has_z = true;
                ++lmp_tag_flags.nvals;
            }
        }
        // scaled coordinates (LAMMPS only)
        else if ((flags & LMP_TAG_SCALED_COORDS) && (tag == "xs" || tag == "xsu")) {
            if (!lmp_tag_flags.has_x) {
                lmp_common_tagid[2] = itag;
                lmp_tag_flags.has_x = true;
                lmp_tag_flags.wix = true;
                ++lmp_tag_flags.nvals;
            }
        } else if ((flags & LMP_TAG_SCALED_COORDS) && (tag == "ys" || tag == "ysu")) {
            if (!lmp_tag_flags.has_y) {
                lmp_common_tagid[3] = itag;
                lmp_tag_flags.has_y = true;
                lmp_tag_flags.wiy = true;
                ++lmp_tag_flags.nvals;
            }
        } else if ((flags & LMP_TAG_SCALED_COORDS) && (tag == "zs" || tag == "zsu")) {
            if (!lmp_tag_flags.has_z) {
                lmp_common_tagid[4] = itag;
                lmp_tag_flags.has_z = true;
                lmp_tag_flags.wiz = true;
                ++lmp_tag_flags.nvals;
            }
        }
        // forces — always present
        else if (tag == "fx") {
            lmp_tag_flags.has_fx = true;
            lmp_common_tagid[5] = itag;
            ++lmp_tag_flags.nvals;
        } else if (tag == "fy") {
            lmp_tag_flags.has_fy = true;
            lmp_common_tagid[6] = itag;
            ++lmp_tag_flags.nvals;
        } else if (tag == "fz") {
            lmp_tag_flags.has_fz = true;
            lmp_common_tagid[7] = itag;
            ++lmp_tag_flags.nvals;
        }
        // velocities — always present
        else if (tag == "vx") {
            lmp_tag_flags.has_vx = true;
            lmp_common_tagid[8] = itag;
            ++lmp_tag_flags.nvals;
        } else if (tag == "vy") {
            lmp_tag_flags.has_vy = true;
            lmp_common_tagid[9] = itag;
            ++lmp_tag_flags.nvals;
        } else if (tag == "vz") {
            lmp_tag_flags.has_vz = true;
            lmp_common_tagid[10] = itag;
            ++lmp_tag_flags.nvals;
        }
        // image flags (LAMMPS only)
        else if ((flags & LMP_TAG_IMAGE_FLAGS) && tag == "ix") {
            lmp_tag_flags.has_ix = true;
            lmp_common_tagid[11] = itag;
            ++lmp_tag_flags.nvals;
        } else if ((flags & LMP_TAG_IMAGE_FLAGS) && tag == "iy") {
            lmp_tag_flags.has_iy = true;
            lmp_common_tagid[12] = itag;
            ++lmp_tag_flags.nvals;
        } else if ((flags & LMP_TAG_IMAGE_FLAGS) && tag == "iz") {
            lmp_tag_flags.has_iz = true;
            lmp_common_tagid[13] = itag;
            ++lmp_tag_flags.nvals;
        }
        // mol (LAMMPS only)
        else if ((flags & LMP_TAG_MOL) && tag == "mol") {
            lmp_tag_flags.has_mol = true;
            lmp_common_tagid[14] = itag;
            ++lmp_tag_flags.nvals;
        }
        // site_en (LMPTRJ, MLPCFG — not LMPYAML)
        else if ((flags & LMP_TAG_SITEEN) && tag == "site_en") {
            lmp_tag_flags.has_siteen = true;
            lmp_common_tagid[20] = itag;
            ++lmp_tag_flags.nvals;
        }
        // magmom (MLPCFG only)
        else if ((flags & LMP_TAG_MAGMOM) && tag == "magmom_x") {
            lmp_tag_flags.has_mx = true;
            lmp_common_tagid[16] = itag;
            ++lmp_tag_flags.nvals;
        } else if ((flags & LMP_TAG_MAGMOM) && tag == "magmom_y") {
            lmp_tag_flags.has_my = true;
            lmp_common_tagid[17] = itag;
            ++lmp_tag_flags.nvals;
        } else if ((flags & LMP_TAG_MAGMOM) && tag == "magmom_z") {
            lmp_tag_flags.has_mz = true;
            lmp_common_tagid[18] = itag;
            ++lmp_tag_flags.nvals;
        }
        // bracket vectors: v_name[N] (LAMMPS only)
        else if ((flags & LMP_TAG_BRACKET_VEC) && !tag.empty() && tag.back() == ']') {
            const size_t bracket_open = tag.rfind('[');
            if (bracket_open == std::string_view::npos || bracket_open == 0) return false;
            const std::string_view number_str = tag.substr(bracket_open + 1, tag.size() - bracket_open - 2);
            int number = 0;
            const auto result = std::from_chars(number_str.data(), number_str.data() + number_str.size(), number);
            if (result.ec != std::errc{} || result.ptr != number_str.data() + number_str.size() || number <= 0)
                return false;

            const std::string name(tag.substr(0, bracket_open));
            auto &columns = lmp_tagvecmap[name];
            if (static_cast<int>(columns.size()) < number)
                columns.resize(number, -1);
            lmp_tagvecsize[name] = std::max(lmp_tagvecsize[name], static_cast<size_t>(number));
            columns[number - 1] = itag;
            ++lmp_tag_flags.nvals;
        }
        // fallback
        else {
            lmp_tagmap[std::string(tag)] = itag;
            ++lmp_tag_flags.nvals;
        }
        tag = tok.next_token();
        ++itag;
    }
    lmp_tag_flags.nvals = itag;
    return lmp_tag_flags.has_x && lmp_tag_flags.has_y && lmp_tag_flags.has_z;
}
