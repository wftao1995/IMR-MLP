// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "gpio.h"
#include "gpio_tokenizer.h"
#include "utils/compressed_stream.h"
#include "tools/chem.h"
#include "tools/procstr.h"
#include "tools/tools.h"
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include "tools/constants.h"

#ifdef USEXML 
#include <tinyxml2.h>
#endif

// get vasp version, chemical info and INCAR settings

/**
 * @brief Only used when get species info, remove all charters after '_' including itself.
 * 
 * @param str Input char buffer
 * @return char* Output, same as input str.
 * @note  In OUTCAR, read POTCAR settings, like Fe_sv
 */

/**
 * @brief Read an integer attribute from OUTCAR in the format of NAME = VALUE
 * 
 * @param fp Input file pointer
 * @param attribute Attribute name
 * @param att Attribute value
 * @return true Success
 * @return false Failed
 */
bool GPIO::read_vasp_outcar_attrib(std::istream &is, std::string attribute, int &att)
{
    do
    {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        auto tok = toks.next_token();
        if (tok == attribute) {
            if (toks.until_next_numeric(att))
                return true;
            else 
                return false;
        }
    } while (!is.eof());

    return false;
}

bool GPIO::read_vasp_outcar_attrib(std::istream &is, std::string attribute, double &att)
{
    do
    {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        auto tok = toks.next_token();
        if (tok == attribute) {
            if (toks.until_next_numeric(att))
                return true;
            else 
                return false;
        }
    } while (!is.eof());

    return false;
}

bool GPIO::read_vasp_outcar_header(std::istream &is)
{
    is_vasp_ml = false;
    // first line, with version info, e.g. " vasp.5.4.4..." or " vasp.6.4.3..."
    do {
        if (!std::getline(is, line_buffer)) return false;
    } while (line_buffer.compare(0, 5, " vasp") != 0);

    // guard against short/malformed version lines (out-of-range operator[] is UB)
    if (line_buffer.size() > 6 && line_buffer[6] == '5')
        vasp_ver = 5;
    else if (line_buffer.size() > 6 && line_buffer[6] == '6')
        vasp_ver = 6;
    else {
        std::cerr << "Unsupported vasp version\n";
        return false;
    }

    while (!is.eof())
    {
        if (!std::getline(is, line_buffer)) return false;
        if (line_buffer.find("INCAR:") != std::string::npos) {
            break;
        }
    }

    // if vasp version 6, check MLFF is enabled or not
    while (!is.eof())
    {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        auto tok = toks.next_token();
        if (tok.empty()) continue;
        if (tok == "POTCAR:") {
            break;
        } else if (tok.find("MLFF") != std::string::npos) {
            is_vasp_ml = true;
        }
    }
    // now we on POTCAR:
    chemtypescount = 1;
    chem.clear();
    // POTCAR species may carry a valence suffix (e.g. "Fe_sv", "K_pv", "O_d"):
    // strip everything from '_' onward before mapping to atomic numbers
    auto strip_suffix = [](std::string_view tok) {
        auto us = tok.find('_');
        return us == std::string_view::npos ? tok : tok.substr(0, us);
    };
    {
        GPIO_LineTokenizer toks{line_buffer};
        toks.next_token();
        toks.next_token();
        auto tok = strip_suffix(toks.next_token());
        chem.push_back(mapnameid[std::string(tok)]);
    }
    while (!is.eof()) {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        auto tok = toks.next_token();
        if (tok != "POTCAR:") break;
        toks.next_token();
        tok = strip_suffix(toks.next_token());
        int cid = mapnameid[std::string(tok)];
        if (cid == chem[0]) break;
        chem.push_back(cid);
        chemtypescount += 1;
    }

    chemcounts.clear();
    total_atomsred = 0;
    while (!is.eof())
    {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        auto tok = toks.next_token();
        if (tok == "ions") {
            // "ions per type   =   n1 n2 ..." — read every count on the line
            int tmpc;
            while (toks.until_next_numeric(tmpc)) {
                chemcounts.push_back(tmpc);
                total_atomsred += tmpc;
            }
            break;
        }
    }
    chemmap.clear();
    for (int i = 0; i < chemtypescount && i < static_cast<int>(chemcounts.size()); ++i)
        for (int j = 0; j < chemcounts[i]; ++j)
            chemmap.push_back(chem[i]);
    read_vasp_outcar_attrib(is, "NELM", nelm_);
    read_vasp_outcar_attrib(is, "EDIFF", ediff_);
    read_vasp_outcar_attrib(is, "EDIFFG", ediffg_);
    read_vasp_outcar_attrib(is, "NSW", nsw_);
    read_vasp_outcar_attrib(is, "IBRION", ibrion_);
    read_vasp_outcar_attrib(is, "ISIF", isif_);
    return true;
}

bool GPIO::load_next_from_vasp_outcar(std::istream &is, Atoms &atom, bool *clean_eof)
{
    atom.clear();
    line_buffer.clear();
    if (clean_eof) *clean_eof = false;
    bool started = false;
    bool have_cell = false;
    bool have_positions = false;
    auto next_line = [&]() {
        if (!std::getline(is, line_buffer)) {
            if (clean_eof) *clean_eof = !started && is.eof() && !is.bad();
            return false;
        }
        if (line_buffer.starts_with("--------------------------------------- Iteration") ||
            line_buffer.starts_with("------------------------ aborting loop because EDIFF is reached") ||
            line_buffer.starts_with("  FORCE on cell =-STRESS") ||
            line_buffer.starts_with(" VOLUME and BASIS-vectors") ||
            line_buffer.starts_with(" POSITION                                       TOTAL-FORCE (eV/Angst)") ||
            line_buffer.starts_with("  free  energy   TOTEN  ="))
            started = true;
        return true;
    };
    //--------------------------------------- Iteration
    int iterid = 0; 
    bool converge = false;
    if (is_vasp_ml)
        converge = true;
    else
    {
        do
        {
            if (!next_line()) return false;
            if (line_buffer.size() >= 49 && line_buffer.compare(0, 49, "--------------------------------------- Iteration") == 0)
                ++iterid;
            else if (line_buffer.size() >= 63 && line_buffer.compare(0, 63, "------------------------ aborting loop because EDIFF is reached") == 0) 
            {
                converge = true;
                break;
            }
            if (iterid == nelm_)
                break;
        } while (true);

        double ediffsteps = 0;
        if (!converge)
        {
            // find total energy change
            while (!is.eof())
            {
                if (!next_line()) return false;
                // a step can reach convergence exactly on its NELM-th electronic
                // iteration: VASP still prints the "aborting loop because EDIFF
                // is reached" line right after the last iteration, so accept it
                // (previously any step with exactly NELM iterations was dropped)
                if (line_buffer.size() >= 63 &&
                    line_buffer.compare(0, 63, "------------------------ aborting loop because EDIFF is reached") == 0) {
                    converge = true;
                    break;
                }
                // stop scanning at the step's TOTEN line or at the next
                // electronic iteration (VASP would continue to it only when the
                // step failed to converge)
                if (line_buffer.substr(0, 25) == "  free  energy   TOTEN  =")
                    break;
                if (line_buffer.size() >= 49 &&
                    line_buffer.compare(0, 49, "--------------------------------------- Iteration") == 0)
                    break;
                // skip the last iteration's remaining output (eigenvalues,
                // density matrices, ...) until its energy-change line
                if (line_buffer.substr(0, 20) == " total energy-change") {
                    GPIO_LineTokenizer toks{line_buffer};
                    auto tok = toks.next_token(); // total
                    tok = toks.next_token(); // energy-change
                    tok = toks.next_token(); // (2.
                    tok = toks.next_token(); // order)
                    toks.until_next_numeric(ediffsteps);
                    if (ediffsteps < ediff_) {
                        converge = true;
                        break;
                    }
                }
            }
        }
    }
    // here we use sigma->0
    while (line_buffer.substr(0, 25) != "  free  energy   TOTEN  =")
    {
        if (!next_line()) return false;
        if (line_buffer.starts_with("--------------------------------------- Iteration")) return false;

        if (read_mag && line_buffer.substr(0, 18) == " magnetization (x)") {
            if (!next_line() || !next_line() || !next_line()) return false;
            atom.magmoms_.resize(total_atomsred);
            for (int i = 0; i < total_atomsred; ++i)
            {
                if (!next_line()) return false;
                GPIO_LineTokenizer toks{line_buffer};
                toks.next_token(); // id
                toks.next_token(); // s
                toks.next_token(); // p
                toks.next_token(); // d
                auto tok = toks.next_token(); // tot or f
                if (tok == "f") tok = toks.next_token();
                fast_float::from_chars(tok.data(), tok.data() + tok.size(), atom.magmoms_[i](0));
                atom.magmoms_[i](1) = 0;
                atom.magmoms_[i](2) = 0;
            }
        }
        else if (line_buffer.substr(0, 24) == "  FORCE on cell =-STRESS")
        {
            if (!next_line()) return false;
            // TODO: use in kB to get more digits
            while (line_buffer.substr(0, 7) != "  in kB")
            // while (line_buffer.substr(0, 7) != "  Total")
                if (!next_line()) return false;
            GPIO_LineTokenizer toks{line_buffer};
            auto tok = toks.next_token();
            tok = toks.next_token(); // kB
            for (int i = 0; i < 3; ++i)
            {
                toks.next_numeric(atom.stress_(i, i));
            }
            toks.next_numeric(atom.stress_(1, 0));
            toks.next_numeric(atom.stress_(2, 1));
            toks.next_numeric(atom.stress_(2, 0));
            atom.stress_(0, 1) = atom.stress_(1, 0);
            atom.stress_(0, 2) = atom.stress_(2, 0);
            atom.stress_(1, 2) = atom.stress_(2, 1);
            atom.have_stress = true;
            // convert from kB to eV
        }
        else if (line_buffer.substr(0, 25) == " VOLUME and BASIS-vectors")
        {
            if (have_cell || have_positions) return false;
            if (!next_line()) return false; //-------
            if (!next_line()) return false; // energy-cutoff
            if (!next_line()) return false; // volume of cell
            if (!next_line()) return false; // direct/reciprocal lattice vectors
            for (int i = 0; i < 3; ++i)
            {
                if (!next_line()) return false;
                GPIO_LineTokenizer toks{line_buffer};
                for (int j = 0; j < 3; ++j) {
                    if (!toks.next_numeric(atom.cell_(i, j))) return false;
                }
            }
            have_cell = true;
            if (atom.have_stress) {
                // convert kB to eV
                atom.stress_ = atom.stress_ * kbar_to_GPa * atom.get_volume() * GPa_to_eVA3;
            }
        }
        else if (line_buffer.substr(0, 70) == " POSITION                                       TOTAL-FORCE (eV/Angst)")
        {
            if (have_positions || total_atomsred <= 0) return false;
            atom.pos_.resize(total_atomsred);
            atom.forces_.resize(total_atomsred);
            atom.chemmap_.resize(total_atomsred);
            int tmpct = 0;
            for (int i = 0; i < chemtypescount && i < static_cast<int>(chemcounts.size()); ++i)
            {
                for (int j = 0; j < chemcounts[i]; ++j)
                {
                    atom.chemmap_[tmpct] = chem[i];
                    ++tmpct;
                }
            }
            if (!next_line()) return false; // -----
            for (int i = 0; i < total_atomsred; ++i) {
                if (!next_line()) return false;
                GPIO_LineTokenizer toks{line_buffer};
                for (int j = 0; j < 3; ++j) {
                    if (!toks.next_numeric(atom.pos_[i](j))) return false;
                }
                for (int j = 0; j < 3; ++j) {
                    if (!toks.next_numeric(atom.forces_[i](j))) return false;
                }
            }
            have_positions = true;
        }
    }
    if (!have_cell || !have_positions) return false;
    if (is_vasp_ml) {
        // read the first numerical value
        GPIO_LineTokenizer toks{line_buffer};
        atom.have_energy = toks.until_next_numeric(atom.energy_);
    } else {
        GPIO_LineTokenizer free_energy_tokens{line_buffer};
        atom.have_fenergy = free_energy_tokens.until_next_numeric(atom.fenergy_);
        if (!atom.have_fenergy) return false;
        if (read_free_energy()) {
            atom.energy_ = atom.fenergy_;
            atom.have_energy = true;
        } else {
            if (!next_line() || !next_line()) return false;
            constexpr std::string_view sigma_label = "energy(sigma->0)";
            const auto sigma_position = line_buffer.find(sigma_label);
            if (sigma_position == std::string::npos) return false;
            GPIO_LineTokenizer toks{std::string_view(line_buffer).substr(sigma_position + sigma_label.size())};
            atom.have_energy = toks.until_next_numeric(atom.energy_);
        }
    }
    if (!atom.have_energy)
        return false;
    if (converge)
        return true;
    return false;
}

#ifdef USEXML
void readvector(Eigen::Vector3d &pos, tinyxml2::XMLElement *iterdata)
{
    std::string buffer;
    buffer = iterdata->GetText();
    std::istringstream iss;
    iss.str(buffer);
    double tmpd;
    for (int vj = 0; vj < 3; vj++)
    {
        iss >> tmpd;
        pos(vj) = tmpd;
    }
}

void readvector(Eigen::RowVector3d &posn, tinyxml2::XMLElement *iterdata)
{
    std::string buffer;
    Eigen::Vector3d pos;
    buffer = iterdata->GetText();
    std::istringstream iss;
    iss.str(buffer);
    double tmpd;
    for (int vj = 0; vj < 3; vj++)
    {
        iss >> tmpd;
        pos(vj) = tmpd;
    }
    posn = pos.transpose();
}
// used for driver_vasp only
// BUG: need to convert to indexed chemmap

bool GPIO::read_vasp_xml(const char *filename, Atoms &atom)
{
    using namespace tinyxml2;
    XMLDocument doc;
    atom.clear();
    XMLError errxml = XML_ERROR_FILE_READ_ERROR;
    if (auto is = open_ifile(filename)) {
        std::string xmlbuf((std::istreambuf_iterator<char>(*is)), std::istreambuf_iterator<char>());
        errxml = doc.Parse(xmlbuf.data(), xmlbuf.size());
    }
    if (XML_SUCCESS == errxml)
    {
        // find root
        XMLElement *xmlroot = doc.RootElement();
        XMLElement *elmdata = nullptr;
        XMLElement *attribdata = nullptr;
        XMLElement *subattribdata = nullptr;
        XMLElement *thirdsubatt = nullptr;
        int totalatom = 0;
        int chemtypes = 0;
        std::vector<int> chem;
        int tmp;
        std::vector<int> chemcount;
        /* #####-----@ATTENTION!!@-----#####
         * See G. Kresse, J. Furthmuller/ Computational Materials Science 6 (1996)
         * 15-50 , we should use e_0_energy here. and
         * https://www.vasp.at/forum/viewtopic.php?f=3&t=16942 , here is a bug in
         * vasp5.4.4 and previous versions of vasp in vasprun.xml this bug is std::fixed
         * in vasp6, so for vasp5.4.4 and earlier versions, e_wo_entrp should be
         * used, for calculations using vasp6 and after, e_0_energy should be used.
         * and reference energies should be subtracted.
         * ######-----@ATTENTION!!@-----#####
         */
        attribdata = xmlroot->FirstChildElement("generator")->FirstChildElement();
        attribdata = attribdata->NextSiblingElement();
        const char *version = attribdata->GetText();
        bool vasp6 = false;
        if (version[0] > '5')
        {
            vasp6 = true;
        }
        attribdata = NULL;
        // main loop
        for (elmdata = xmlroot->FirstChildElement(); elmdata != NULL;
             elmdata = elmdata->NextSiblingElement())
        { // query all nodes in root
            // chemical info
            if (strcmp(elmdata->Name(), "atominfo") == 0)
            {
                attribdata = elmdata->FirstChildElement("atoms");
                attribdata->QueryIntText(&totalatom);
                attribdata = elmdata->FirstChildElement("types");
                attribdata->QueryIntText(&chemtypes);
                chemcount.reserve(chemtypes);
                attribdata = elmdata->LastChildElement("array");
                subattribdata = attribdata->FirstChildElement("set");
                for (XMLElement *subsubattribdata = subattribdata->FirstChildElement();
                     subsubattribdata != NULL;
                     subsubattribdata = subsubattribdata->NextSiblingElement())
                {
                    // many rc here
                    thirdsubatt = subsubattribdata->FirstChildElement();
                    thirdsubatt->QueryIntText(&tmp);
                    chemcount.push_back(tmp);
                    thirdsubatt = thirdsubatt->NextSiblingElement();
                    const char *ctmp = thirdsubatt->GetText();
                    std::string tmps(ctmp);
                    trim_all(tmps);
                    auto us_pos = tmps.find('_');
                    if (us_pos != std::string::npos)
                        tmps = tmps.substr(0, us_pos);
                    chem.push_back(Chemicals::atomic_numbers.at(tmps));
                }
                atom.pos_.reserve(totalatom);
                atom.chemmap_.reserve(totalatom);
            }
            // read loops
            else if (strcmp(elmdata->Name(), "calculation") == 0)
            {
                Eigen::RowVector3d pos;
                Eigen::Matrix3d cell;
                int tmpc = 0; // template
                // crystal
                attribdata = elmdata->FirstChildElement("structure")
                                 ->FirstChildElement("crystal")
                                 ->FirstChildElement();
                for (XMLElement *iterdata = attribdata->FirstChildElement();
                     iterdata != NULL; iterdata = iterdata->NextSiblingElement())
                {
                    readvector(pos, iterdata);
                    cell.row(tmpc) = pos;
                    tmpc++;
                }
                atom.cell_ = cell;
                // pos
                attribdata = elmdata->FirstChildElement("structure")
                                 ->FirstChildElement("varray");
                for (XMLElement *iterdata = attribdata->FirstChildElement();
                     iterdata != NULL; iterdata = iterdata->NextSiblingElement())
                {
                    readvector(pos, iterdata);
                    atom.pos_.push_back(pos * cell);
                }
                // forces
                Eigen::Vector3d force;
                attribdata = elmdata->FirstChildElement("varray");
                for (XMLElement *iterdata = attribdata->FirstChildElement();
                     iterdata != NULL; iterdata = iterdata->NextSiblingElement())
                {
                    readvector(force, iterdata);
                    atom.forces_.push_back(force);
                }
                // stress, may not have, in kbar and convert to ev/A3
                attribdata = attribdata->NextSiblingElement("varray");
                if (attribdata != NULL && strcmp(attribdata->Attribute("name"), "stress") == 0)
                {
                    int cnt = 0;
                    Eigen::Matrix3d stress;
                    for (XMLElement *iterdata = attribdata->FirstChildElement();
                         iterdata != NULL; iterdata = iterdata->NextSiblingElement())
                    {
                        readvector(pos, iterdata);
#ifdef VASP_STRESS_USE_PR
                        stress.row(cnt) = pos * atom.cell_.determinant() / 1602.1766208;
#else
                        stress.row(cnt) = pos;
#endif
                        cnt++;
                    }
                    atom.stress_ = stress;
                    atom.have_stress = true;
                }
                // energy
                double energy;
                attribdata = elmdata->FirstChildElement("energy");
                // get energy
                if (vasp_use_free_eng)
                {
                    thirdsubatt = attribdata->FirstChildElement();
                }
                else
                {
                    // before vasp6 there is a bug in writting vasprun.xml
                    if (vasp6)
                    {
                        thirdsubatt = attribdata->FirstChildElement()
                                          ->NextSiblingElement()
                                          ->NextSiblingElement();
                    }
                    else
                    {
                        thirdsubatt = attribdata->FirstChildElement()->NextSiblingElement();
                    }
                }
                thirdsubatt->QueryDoubleText(&energy);
                atom.energy_ = energy;
                atom.have_energy = true;
                // chemmap
                int ct = 0;
                for (int i = 0; i < chemtypes; ++i)
                {
                    for (int j = 0; j < chemcount[i]; ++j)
                    {
                        atom.chemmap_[ct] = chem[i];
                        ++ct;
                    }
                }
            }
            else
            {
                continue;
            }
        }
    }
    //@TODO: finish
    else
    {
        std::cout << "Error while reading vasprun.xml" << std::endl;
        return false;
    }
    // #ifndef IMRMLP_USE_ABSOLUTE_INDEX
    //     atom.to_indexed_chemmap();
    // #endif
    return true;
}

bool GPIO::read_vasp_xml(std::vector<Atoms> &atoms, std::string filename)
{
    using namespace tinyxml2;
    auto fail = [](const char *message) {
        std::cerr << "Error reading vasprun.xml: " << message << '\n';
        return false;
    };
    XMLDocument doc;
    if (auto is = open_ifile(filename)) {
        const std::string xmlbuf((std::istreambuf_iterator<char>(*is)), std::istreambuf_iterator<char>());
        if (is->bad() || doc.Parse(xmlbuf.data(), xmlbuf.size()) != XML_SUCCESS)
            return fail("invalid XML document");
    } else return fail("cannot open input");
    auto *root = doc.RootElement();
    if (!root || std::strcmp(root->Name(), "modeling") != 0)
        return fail("missing modeling root");
    auto named = [](XMLElement *parent, const char *tag, const char *name) -> XMLElement * {
        if (parent)
            for (auto *item = parent->FirstChildElement(tag); item; item = item->NextSiblingElement(tag)) {
                const char *key = item->Attribute("name");
                if (key && std::strcmp(key, name) == 0) return item;
            }
        return nullptr;
    };
    auto number = [](XMLElement *item, auto &value) {
        if (!item || !item->GetText()) return false;
        GPIO_LineTokenizer tokens(item->GetText());
        return tokens.next_numeric(value) && tokens.next_token().empty() && std::isfinite(value);
    };
    auto vector = [](XMLElement *item, Eigen::RowVector3d &value) {
        if (!item || !item->GetText()) return false;
        GPIO_LineTokenizer tokens(item->GetText());
        for (int axis = 0; axis < 3; ++axis)
            if (!tokens.next_numeric(value[axis]) || !std::isfinite(value[axis])) return false;
        return tokens.next_token().empty();
    };
    auto rows = [&](XMLElement *array, int count, auto &&consume) {
        if (!array) return false;
        int row = 0;
        for (auto *item = array->FirstChildElement(); item; item = item->NextSiblingElement()) {
            Eigen::RowVector3d value;
            if (row >= count || std::strcmp(item->Name(), "v") != 0 || !vector(item, value)) return false;
            consume(row++, value);
        }
        return row == count;
    };
    auto *version_item = named(root->FirstChildElement("generator"), "i", "version");
    if (!version_item || !version_item->GetText()) return fail("missing VASP version");
    std::string version(version_item->GetText());
    trim_all(version);
    const auto end_major = version.find_first_not_of("0123456789");
    const std::string major_text = version.substr(0, end_major);
    int major = 0;
    GPIO_LineTokenizer version_tokens(major_text);
    if (!version_tokens.next_numeric(major) || major <= 0 ||
        (end_major != std::string::npos && version[end_major] != '.'))
        return fail("invalid VASP version");
    const bool vasp6 = major >= 6;

    // Optional run settings do not reuse uninitialized or previous-file values.
    ibrion_ = nsw_ = isif_ = nelm_ = 0;
    auto *incar = root->FirstChildElement("incar");
    for (const auto &[key, output] : {std::pair{"IBRION", &ibrion_}, {"NSW", &nsw_}, {"ISIF", &isif_}})
        if (auto *item = named(incar, "i", key); item && !number(item, *output))
            return fail("invalid INCAR integer");
    auto *electronic = named(root->FirstChildElement("parameters"), "separator", "electronic");
    auto *convergence = named(electronic, "separator", "electronic convergence");
    if (auto *item = named(convergence, "i", "NELM"); item && (!number(item, nelm_) || nelm_ <= 0))
        return fail("invalid NELM");

    auto *atominfo = root->FirstChildElement("atominfo");
    int totalatom = 0, ntypes = 0;
    if (!atominfo || !number(atominfo->FirstChildElement("atoms"), totalatom) || totalatom <= 0 ||
        !number(atominfo->FirstChildElement("types"), ntypes) || ntypes <= 0 || ntypes > totalatom)
        return fail("invalid atom counts");
    auto *types = named(atominfo, "array", "atomtypes");
    auto *set = types ? types->FirstChildElement("set") : nullptr;
    if (!set) return fail("missing atomtypes array");
    std::vector<int> frame_types, parsed_chem, parsed_counts;
    int count_sum = 0;
    for (auto *item = set->FirstChildElement(); item; item = item->NextSiblingElement()) {
        auto *count_item = item->FirstChildElement("c");
        auto *symbol_item = count_item ? count_item->NextSiblingElement("c") : nullptr;
        int count = 0;
        if (std::strcmp(item->Name(), "rc") != 0 || !number(count_item, count) || count <= 0 ||
            count > totalatom - count_sum || !symbol_item || !symbol_item->GetText())
            return fail("invalid atomtypes row");
        std::string symbol(symbol_item->GetText());
        trim_all(symbol);
        if (auto underscore = symbol.find('_'); underscore != std::string::npos) symbol.resize(underscore);
        const auto type = mapnameid.find(symbol);
        if (type == mapnameid.end()) return fail("unmapped atom species");
        parsed_chem.push_back(type->second);
        parsed_counts.push_back(count);
        count_sum += count;
        // Expand only after the row/count checks, and never past totalatom.
        frame_types.insert(frame_types.end(), count, type->second);
    }
    if (count_sum != totalatom || parsed_chem.size() != static_cast<size_t>(ntypes))
        return fail("atomtypes counts do not match atoms/types");
    chem = std::move(parsed_chem);
    chemcounts = std::move(parsed_counts);
    chemtypescount = ntypes;
    total_atomsred = totalatom;

    auto structure = [&](XMLElement *source, Atoms &frame) {
        auto *crystal = source ? source->FirstChildElement("crystal") : nullptr;
        if (!rows(named(crystal, "varray", "basis"), 3,
                  [&](int row, const auto &value) { frame.cell_.row(row) = value; })) return false;
        if (!rows(named(source, "varray", "positions"), totalatom,
                  [&](int, const auto &value) { frame.pos_.push_back(value * frame.cell_); })) return false;
        for (const auto &position : frame.pos_) if (!position.allFinite()) return false;
        frame.chemmap_ = frame_types;
        return true;
    };
    for (auto *element = root->FirstChildElement(); element; element = element->NextSiblingElement()) {
#ifdef XML_READ_INI
        const char *structure_name = element->Attribute("name");
        if (std::strcmp(element->Name(), "structure") == 0 && structure_name &&
            std::strcmp(structure_name, "initialpos") == 0) {
            Atoms frame;
            if (!structure(element, frame)) return fail("invalid initial structure");
            frame.typeinfo_ = &typeinfo;
            atoms.emplace_back(std::move(frame));
        }
#endif
        if (std::strcmp(element->Name(), "calculation") != 0) continue;
        int scsteps = 0;
        for (auto *step = element->FirstChildElement("scstep"); step; step = step->NextSiblingElement("scstep"))
            ++scsteps;
        if (nelm_ > 0 && scsteps == nelm_) {
            std::cout << "Warning: non-converged calculations detected\n";
            if (drop) continue;
        }
        Atoms frame;
        auto *source = element->FirstChildElement("structure");
        if (!structure(source, frame)) return fail("invalid calculation structure");
        double volume = 0;
        if (!number(named(source->FirstChildElement("crystal"), "i", "volume"), volume) || volume <= 0)
            return fail("invalid cell volume");
        if (!rows(named(element, "varray", "forces"), totalatom,
                  [&](int, const auto &value) { frame.forces_.push_back(value.transpose()); }))
            return fail("invalid force array");
        if (auto *stress = named(element, "varray", "stress")) {
            if (!rows(stress, 3, [&](int row, const auto &value) { frame.stress_.row(row) = value; }))
                return fail("invalid stress array");
#ifdef VASP_STRESS_USE_PR
            frame.stress_ *= volume / 1602.1766208;
#endif
            if (!frame.stress_.allFinite()) return fail("invalid converted stress");
            frame.have_stress = true;
        }
        auto *energy = element->FirstChildElement("energy");
        if (auto *free = named(energy, "i", "e_fr_energy")) {
            if (!number(free, frame.fenergy_)) return fail("invalid free energy");
            frame.have_fenergy = true;
        }
        // Preserve the historical pre-VASP6 entropy workaround and independent F label.
        const char *selected = read_free_energy() ? "e_fr_energy" : (vasp6 ? "e_0_energy" : "e_wo_entrp");
        if (!number(named(energy, "i", selected), frame.energy_)) return fail("missing or invalid selected energy");
        frame.have_energy = true;
        if (readornot()) {
            frame.id = ncfgs_;
            discard_properties(frame);
            atoms.emplace_back(std::move(frame));
        }
        ++ncfgs_;
    }
    return true;
}
#endif

bool GPIO::read_vasp_xdatcar_header(std::istream &is, bool comment_read, bool *clean_eof)
{
    // Parse into local values so a malformed header cannot reuse or partially
    // replace the preceding frame's cell or species information.
    if (clean_eof) *clean_eof = false;
    if (!comment_read && !std::getline(is, line_buffer)) {
        if (clean_eof) *clean_eof = is.eof() && !is.bad();
        return false;
    }
    bool blank_comment = false;
    while (GPIO_LineTokenizer{line_buffer}.next_token().empty()) {
        blank_comment = true;
        if (!std::getline(is, line_buffer)) {
            if (clean_eof) *clean_eof = is.eof() && !is.bad();
            return false;
        }
    }
    const std::string comment = line_buffer;
    if (!std::getline(is, line_buffer)) return false;
    std::string first_cell_line;
    if (blank_comment) {
        // A blank POSCAR comment is legal. Distinguish its scale+first cell
        // row from blank separators followed by a nonempty comment+scale.
        GPIO_LineTokenizer possible_scale{comment}, possible_row{line_buffer};
        double candidate_scale, entry;
        if (possible_scale.next_numeric(candidate_scale) && possible_scale.next_token().empty() &&
            possible_row.next_numeric(entry) && possible_row.next_numeric(entry) &&
            possible_row.next_numeric(entry) && possible_row.next_token().empty()) {
            first_cell_line = line_buffer;
            line_buffer = comment;
        }
    }
    double scale;
    GPIO_LineTokenizer scale_tokens{line_buffer};
    if (!scale_tokens.next_numeric(scale) || !std::isfinite(scale) || scale == 0 ||
        !scale_tokens.next_token().empty()) return false;
    Eigen::Matrix3d cell;
    for (int i = 0; i < 3; ++i) {
        if (i == 0 && !first_cell_line.empty()) line_buffer = first_cell_line;
        else if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        for (int j = 0; j < 3; ++j)
            if (!toks.next_numeric(cell(i, j)) || !std::isfinite(cell(i, j))) return false;
        if (!toks.next_token().empty()) return false;
    }
    const double volume = std::abs(cell.determinant());
    if (!std::isfinite(volume) || volume == 0) return false;
    // Like POSCAR, a negative scale specifies the desired cell volume.
    cell *= scale > 0 ? scale : std::cbrt(-scale / volume);
    if (!cell.allFinite()) return false;

    std::vector<int> species, counts, mapping;
    if (!std::getline(is, line_buffer)) return false;
    GPIO_LineTokenizer species_tokens{line_buffer};
    for (auto name = species_tokens.next_token(); !name.empty();
         name = species_tokens.next_token()) {
        const auto found = mapnameid.find(std::string(name));
        if (found == mapnameid.end()) return false;
        species.push_back(found->second);
    }
    if (species.empty() || !std::getline(is, line_buffer)) return false;
    GPIO_LineTokenizer count_tokens{line_buffer};
    int total = 0;
    for (size_t i = 0; i < species.size(); ++i) {
        int count;
        if (!count_tokens.next_numeric(count) || count < 0 ||
            count > std::numeric_limits<int>::max() - total) return false;
        counts.push_back(count);
        total += count;
    }
    if (total == 0 || !count_tokens.next_token().empty()) return false;
    for (size_t i = 0; i < species.size(); ++i)
        mapping.insert(mapping.end(), counts[i], species[i]);

    cell_ = cell;
    chem = std::move(species);
    chemcounts = std::move(counts);
    chemmap = std::move(mapping);
    total_atomsred = total;
    return true;
}

bool GPIO::load_next_from_xdatcar(std::istream &is, Atoms &atom, bool *clean_eof)
{
    if (clean_eof) *clean_eof = false;
    bool repeated_header = false;
    int configuration_id;
    while (std::getline(is, line_buffer)) {
        GPIO_LineTokenizer marker{line_buffer};
        const auto first = marker.next_token();
        if (first.empty()) continue;
        if (first == "Direct") {
            auto rest = std::string_view(line_buffer).substr(marker.pos);
            const auto equal = rest.find('=');
            if (equal == std::string_view::npos) return false;
            GPIO_LineTokenizer label{rest.substr(0, equal)};
            GPIO_LineTokenizer number{rest.substr(equal + 1)};
            if (label.next_token() != "configuration" || !label.next_token().empty() ||
                !number.next_numeric(configuration_id) || configuration_id < 0 ||
                !number.next_token().empty()) return false;
            break;
        }
        // Older GPIO writers emitted the first header twice. Accept a fully
        // valid duplicate, never arbitrary skipped lines or a different cell.
        const Eigen::Matrix3d previous_cell = cell_;
        const auto previous_mapping = chemmap;
        const auto previous_counts = chemcounts;
        if (!read_vasp_xdatcar_header(is, true) || !cell_.isApprox(previous_cell, 1e-12) ||
            chemmap != previous_mapping || chemcounts != previous_counts) return false;
        repeated_header = true;
    }
    if (!is) {
        if (clean_eof) *clean_eof = is.eof() && !is.bad() && !repeated_header;
        return false;
    }

    Atoms frame;
    frame.resize(total_atomsred);
    frame.id = configuration_id;
    frame.cell_ = cell_;
    for (int i = 0; i < total_atomsred; ++i) {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        Eigen::RowVector3d position;
        for (int j = 0; j < 3; ++j)
            if (!toks.next_numeric(position(j)) || !std::isfinite(position(j))) return false;
        if (!toks.next_token().empty()) return false;
        frame.pos_[i] = position * cell_;
        if (!frame.pos_[i].allFinite()) return false;
    }
    frame.chemmap_ = chemmap;
    atom = std::move(frame);
    return true;
}

bool GPIO::load_next_from_xdatcar6(std::istream& is, Atoms& atom, bool *clean_eof)
{
    if (!read_vasp_xdatcar_header(is, false, clean_eof)) return false;
    // Once a header starts, a missing configuration or coordinates is invalid.
    return load_next_from_xdatcar(is, atom);
}

bool GPIO::read_vasp_xdatcar_sequence(std::istream &is, Atoms &atom)
{
    return load_next_from_xdatcar(is, atom);
}

bool GPIO::read_vasp_poscar(std::istream &is, Atoms &atom)
{
    // comment line
    int natoms = 0;
    if (!std::getline(is, line_buffer)) return false;
    double scale;
    if (!std::getline(is, line_buffer)) return false;
    GPIO_LineTokenizer{line_buffer}.until_next_numeric(scale);
    for (int i = 0; i < 3; ++i) {
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        for (int j = 0; j < 3; ++j) {
            toks.next_numeric_unsafe(atom.cell_(i, j));
        }
    }
    if (scale > 0)
        atom.cell_ *= scale;
    else {
        double volcell = fabs(atom.cell_.determinant());
        double voltarget = -scale;
        double ss = pow(voltarget / volcell, 1./3.);
        atom.cell_ *= ss;
    }
    // get type info
    chem.clear();
    chemcounts.clear();
    // TODO: first check if ver 5 or before 5
    if (std::getline(is, line_buffer)) {
        GPIO_LineTokenizer toks{line_buffer};
        auto tok = toks.next_token();
        while (!tok.empty()) {
            chem.push_back(mapnameid[std::string(tok)]);
            tok = toks.next_token();
        }
    } else return false;
    if (std::getline(is, line_buffer)) {
        GPIO_LineTokenizer toks{line_buffer};
        chemcounts.resize(chem.size());
        for (int i = 0; i < chem.size(); ++i) {
            toks.next_numeric(chemcounts[i]);
            natoms += chemcounts[i];
        }
    } else return false;
    atom.resize(natoms);
    int ct = 0;
    for (int i = 0; i < chem.size(); ++i) {
        for (int j = 0; j < chemcounts[i]; ++j) {
            atom.chemmap_[ct] = chem[i];
            ++ct;
        }
    }
    bool is_car_mode = true;
    bool seld = false;
    if (!std::getline(is, line_buffer)) return false; // Selective dynamics or D/C, currently not s is not supported
    auto tok = GPIO_LineTokenizer{line_buffer}.next_token();
    char flag = toupper(tok[0]);
    if (flag == 'S') {
        seld = true;
        if (!std::getline(is, line_buffer)) return false;
        auto tok2 = GPIO_LineTokenizer{line_buffer}.next_token();
        flag = toupper(tok2[0]);
    } 
    if (flag == 'C' || flag == 'K') {
        is_car_mode = true;
    } else if (flag == 'D' || flag == 'F') {
        is_car_mode = false;
    } else {
        std::cerr << "ERROR: unknown POSCAR format" << std::endl;
        return false;
    }
    // read atoms
    if (seld) {
        atom.sel_dynpos_.assign(natoms, Eigen::RowVector3d::Zero());
    }
    Eigen::RowVector3d tmppos;
    for (int i = 0; i < natoms; ++i) { 
        if (!std::getline(is, line_buffer)) return false;
        GPIO_LineTokenizer toks{line_buffer};
        // positions
        for (int a = 0; a < 3; ++a) {
            toks.next_numeric_unsafe(atom.pos_[i](a));
        }
        // selective dyanmics info
        if (seld) {
            for (int a = 0; a < 3; ++a) {
                tok = toks.next_token();
                if (tok.empty()) {
                    std::cerr << "ERROR: no selective dynamics info given" << std::endl;
                    return false;
                }
                if (toupper(tok[0]) == 'T') {
                    atom.sel_dynpos_[i](a) = 1;
                } else if (toupper(tok[0]) == 'F') {
                    atom.sel_dynpos_[i](a) = 0;
                } else {
                    std::cerr << "Unknown selective dynamics info " << tok << std::endl;
                    return false;
                }
            }
        }
    }
    if (!is_car_mode) {
        atom.cartesian = false;
        atom.to_cartesian();
    }
    return true;
}

bool GPIO::load_next_from_vasp_ABN(std::istream &is, Atoms &atom)
{
    int numtypes = 0;
    int natoms = 0;
    
    while (true) {
        if (!std::getline(is, line_buffer)) break;
        std::string_view line = skipLeadingSpaces(line_buffer);
        if (line.substr(0, 6) == "System") {
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            auto lab = GPIO_LineTokenizer{line_buffer}.next_token();
            atom.features_["label"] = std::string(lab);
        } else if (line.substr(0, 24) == "The number of atom types") {
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            GPIO_LineTokenizer{line_buffer}.until_next_numeric(numtypes);
            chem.resize(numtypes);
            chemcounts.clear();
            chemcounts.resize(numtypes, 0);
        } else if (line.substr(0, 19) == "The number of atoms") {
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            GPIO_LineTokenizer{line_buffer}.until_next_numeric(natoms);
            chemmap.resize(natoms);
        } else if (line.substr(0, 10) == "Atom types") {
            std::getline(is, line_buffer);
            int ct = 0;
            for (int i = 0; i < numtypes; ++i)
            {
                std::getline(is, line_buffer);
                GPIO_LineTokenizer toks{line_buffer};
                auto tok = toks.next_token();
                toks.next_numeric(chemcounts[i]);
                chem[i] = mapnameid[std::string(tok)];
                for (int j = 0; j < chemcounts[i]; ++j) {
                    chemmap[ct] = chem[i];
                    ++ct;
                }
            }
            atom.chemmap_ = chemmap;
        } else if (line.substr(0, 17) == "Primitive lattice") {
            std::getline(is, line_buffer);
            for (int i = 0; i < 3; ++i) {
                std::getline(is, line_buffer);
                GPIO_LineTokenizer toks{line_buffer};
                for (int j = 0; j < 3; ++j)
                    toks.next_numeric(atom.cell_(i, j));
            }
        } else if (line.substr(0, 16) == "Atomic positions") {
            std::getline(is, line_buffer);
            for (int i = 0; i < natoms; ++i) {
                std::getline(is, line_buffer);
                GPIO_LineTokenizer toks{line_buffer};
                atom.pos_.resize(natoms);
                for (int j = 0; j < 3; ++j)
                    toks.next_numeric(atom.pos_[i](j));
            }
        } else if (line.substr(0, 12) == "Total energy") {
            atom.have_energy = true;
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            GPIO_LineTokenizer{line_buffer}.until_next_numeric(atom.energy_);
        } else if (line.substr(0, 6) == "Forces") {
            std::getline(is, line_buffer);
            atom.forces_.resize(natoms);
            for (int i = 0; i < natoms; ++i) {
                std::getline(is, line_buffer);
                GPIO_LineTokenizer toks{line_buffer};
                for (int j = 0; j < 3; ++j)
                    toks.next_numeric(atom.forces_[i](j));
            }
        } else if (line.substr(0, 6) == "Stress") {
            atom.have_stress = true;
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            GPIO_LineTokenizer toks{line_buffer};
            double val;
            for (int i = 0; i < 3; ++i) {
                toks.next_numeric(val);
                atom.stress_(i, i) = val * atom.get_volume() / 1602.1766208;
            }
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            std::getline(is, line_buffer);
            toks.line = line_buffer;
            toks.pos = 0;
            toks.next_numeric(val);
            atom.stress_(0, 1) = val * atom.get_volume() / 1602.1766208;
            atom.stress_(1, 0) = atom.stress_(0, 1);
            toks.next_numeric(val);
            atom.stress_(1, 2) = val * atom.get_volume() / 1602.1766208;
            atom.stress_(2, 1) = atom.stress_(1, 2);
            toks.next_numeric(val);
            atom.stress_(0, 2) = val * atom.get_volume() / 1602.1766208;
            atom.stress_(2, 0) = atom.stress_(0, 2);
        } else if (line.substr(0, 13) == "Configuration") break;
    }
    if (atom.pos_.size() == atom.chemmap_.size())
        return true;
    return false;
}

bool GPIO::read_vasp_abn_header(std::istream &is) {
    while (true) {
        if (!std::getline(is, line_buffer)) return false;
        std::string_view line = skipLeadingSpaces(line_buffer);
        if (line.substr(0, 13) == "Configuration") break;
    }
    return true;
}
