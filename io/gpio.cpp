#include "io/hessian_io.h"
#include "gpio.h"
#include "gpio_tokenizer.h"
#include "utils/compressed_stream.h"
#include "tools/chem.h"

#include <stdlib.h>
#include <stdio.h>
#include <sstream>
#include <stdexcept>
#include <iterator>
#include <filesystem>

#define BUFFER_SIZE1 200
#define MAX_PROP_ACCEPT 10

namespace fs = std::filesystem;

std::string_view skipLeadingSpaces(const std::string& input) {
    size_t start = 0;
    while (start < input.size() && std::isspace(static_cast<unsigned char>(input[start]))) {
        ++start;
    }
    return std::string_view(input.data() + start, input.size() - start);
}

std::map<std::string, GPIO::supported_format> GPIO::format_dict = {
    {"xml", GPIO::VASPRUN},
    {"vasprun", GPIO::VASPRUN},
    {"vasprun.xml", GPIO::VASPRUN},
    {"bin", GPIO::MLIPBIN},
    {"cfg", GPIO::MLPCFG},
    {"mlpcfg", GPIO::MLPCFG},
    {"xsf", GPIO::XSF},
    {"axsf", GPIO::XSF},
    {"xyz", GPIO::EXTXYZ},
    {"extxyz", GPIO::EXTXYZ},
    {"cif", GPIO::CIF},
    {"CIF", GPIO::CIF},
    {"CFG", GPIO::CFG}, // TODO: cfg format
    {"vasp", GPIO::POSCAR},
    {"poscar", GPIO::POSCAR},
    {"POSCAR", GPIO::POSCAR},
    {"bin", GPIO::MLIPBIN}, 
    {"xdatcar", GPIO::XDATCAR},
    {"XDATCAR", GPIO::XDATCAR},
    {"xdatcar6", GPIO::XDATCAR6},
    {"XDATCAR6", GPIO::XDATCAR6},
    {"ab", GPIO::VASPABN},
    {"lammpstrj", GPIO::LMPTRJ},
    {"lammpstraj", GPIO::LMPTRJ},
    {"lmptrj", GPIO::LMPTRJ},
    {"lmp", GPIO::LMPDAT},
    {"lmpdat", GPIO::LMPDAT},
    {"yaml", GPIO::LMPYAML},
    {"lat", GPIO::LMPDAT},
    {"poscars", GPIO::POSCARPACKAGE},
    // {"poutcar", GPIO::POUTCAR},
    // {"POUTCAR", GPIO::POUTCAR},
    {"OUTCAR", GPIO::OUTCAR},
    {"outcar", GPIO::OUTCAR},
    {"gro", GPIO::GRODATA},
    {"n2p2", GPIO::N2P2DAT},
    {"cp2kxyz", GPIO::CP2KXYZ},
    {"ML_AB", GPIO::VASPABN}, 
    {"ML_ANB", GPIO::VASPABN},
    {"VASPABN", GPIO::VASPABN},
    {"vaspabn", GPIO::VASPABN},
    {"vaspab", GPIO::VASPABN},
    {"ml_ab", GPIO::VASPABN},
    {"ml_abn", GPIO::VASPABN},
    {"vtk", GPIO::VTK},
    {"dlpoly", GPIO::DLPOLY4},
    {"DLPOLY", GPIO::DLPOLY4},
    {"msi", GPIO::MSI},
    {"xsd", GPIO::XSD},
    {"car", GPIO::CAR},
};

bool GPIO::vasp_use_free_eng = true;
bool GPIO::overwrite = false;
bool GPIO::backupold = false;
// TODO: convert to token reader
bool GPIO::readmap(std::string filename) {
    mapidname.clear();
    mapnameid.clear();
    atomradii.clear();
    atomic_vals.clear();
    auto ifs = open_ifile(filename);
    if (!ifs) {
        std::cerr << "Error: cannot open map file " << filename << std::endl;
        return false;
    }
    int ct = 0;
    std::string line, name, value;
    while (std::getline(*ifs, line)) {
        std::istringstream row(line);
        if (!(row >> name)) break;
        mapnameid[name] = ct;
        mapidname[ct] = name;
        for (int column = 1; column < MAX_PROP_ACCEPT && row >> value; ++column) {
            const double number = std::atof(value.c_str());
            if (column == 1) atomradii[ct] = number;
            atomic_vals[ct].push_back(number);
        }
        ++ct;
    }
    return ct > 0 && !ifs->bad();
}

void GPIO::setoutrate(int irate, int start, int end) {
    outrate = irate;
    out_startfrom = start;
    out_endto = end;
}

bool GPIO::validate_read_range() const {
    if (read_startfrom < 0 || read_endto < read_startfrom || read_irate <= 0) {
        std::cerr << "Error: invalid frame selection: require 0 <= read_startfrom <= read_endto"
                  << " and read_irate > 0 (got " << read_startfrom << ", "
                  << read_endto << ", " << read_irate << ")\n";
        return false;
    }
    return true;
}

bool GPIO::readornot() {
    // Compare inclusive bounds directly: read_endto + 1 overflows for INT_MAX.
    return ncfgs_ >= read_startfrom && ncfgs_ <= read_endto &&
           (ncfgs_ - read_startfrom) % read_irate == 0;
}

bool GPIO::writeornot() {
    return write_icfg >= out_startfrom && (write_icfg - out_startfrom) % outrate == 0 && write_icfg <= out_endto;
}

void GPIO::setmap(void) {
    for (int i = 0; i < static_cast<int>(std::size(Chemicals::chemical_symbols)); ++i) {
        mapidname[i] = Chemicals::chemical_symbols[i];
        mapnameid[Chemicals::chemical_symbols[i]] = i;
    }
}

void GPIO::setmap(const std::vector<std::string> &elements) {
    mapidname.clear();
    mapnameid.clear();
    atomradii.clear();
    atomic_vals.clear();
    if (elements.empty()) {
        setmap();
        return;
    }
    for (size_t i = 0; i < elements.size(); ++i) {
        mapidname[i] = elements[i];
        mapnameid[elements[i]] = i;
    }
}

void GPIO::setmap(const std::map<int, std::string> &elements) {
    mapidname = elements;
    mapnameid.clear();
    atomradii.clear();
    atomic_vals.clear();
    for (const auto &[index, name] : elements) mapnameid[name] = index;
}

GPIO::GPIO() : GPIO(true) {}

GPIO::GPIO(bool bind_console) : outdirect(false), isdummytype(false),
               outrate(1), out_startfrom(0), out_endto(INT_MAX),
               append(false), force_tilt(false), drop(false), cfg_have_efs(false) {
#ifndef IMRMLP_DEBUG
#ifndef SYNC
    if (bind_console) {
        std::ios_base::sync_with_stdio(false);
        std::cin.tie(NULL);
    }
#endif
#endif
    setmap();

    const auto setting = [&]<class T>(T &value, const std::string &name,
                        const std::string &alias, const std::string &help, int nargs, int level) {
        if (bind_console) ConnectWithConsole(CMDOption::global_cmdopts,value,name,alias,help,nargs,level);
        else Connect(value,name,help,nargs,level);
    };
    setting(absolute_index_read, "absolute_index_read", "abr", "weather use atomic numbers when reading", 1, 100);
    setting(absolute_index_write, "absolute_index_write", "abw", "weather use atomic numbers when writing instead of 1.2.3..", 1, 100);
    setting(lossless, "lossless", "", "output text file in lossless format, not all supported", 1, 10);
    setting(outdirect, "output_in_direct", "", "output structure file in direct mode if format support it", 0, 10);
    setting(write_virtual, "xyz_write_virtual", "", "in extxyz file, output virtual", 1, 10);
    setting(write_stress, "xyz_write_stress", "", "in extxyz file, output stress", 1, 10);
    setting(write_mass, "lmp_write_mass", "", "in lammps data file, output mass", 1, 10);
    // ConnectWithConsole(CMDOption::global_cmdopts, reset_status_every_read, "reset_read_status", "", "reset read status every time when read() is called", 0, 10);
    setting(iwrite_offset, "output_index_offset", "", "set output index offset, usually 0 or 1", 1, 10);
    setting(lmp_phrase_tag_always, "lmp_phrase_tag_always", "", "rephrase tag each time when reading a frame", 1, 10);
}

std::unique_ptr<GPIO> GPIO::new_session_state() const {
    auto worker = std::make_unique<GPIO>(false);
    // Copy configuration explicitly: ConnectSettings' bindings must continue to
    // point at the worker, and parser/header/counter state must start empty.
    worker->ApplySettings(CurrentSettings());
    worker->mapnameid=mapnameid; worker->mapidname=mapidname;
    worker->atomradii=atomradii; worker->atomic_vals=atomic_vals;
    worker->drop=drop; worker->packatoms=packatoms; worker->r_extent=r_extent;
    worker->discard_props=discard_props; worker->force_write_all_to_one=force_write_all_to_one;
    worker->reset_status_every_write=reset_status_every_write;
    worker->reset_status_every_read=reset_status_every_read;
    worker->suffix_id_use_internal=suffix_id_use_internal; worker->fname_separator=fname_separator;
    worker->header_line=header_line; worker->cfg_have_efs=cfg_have_efs;
    worker->cfg_have_mag=cfg_have_mag; worker->read_mag=read_mag; worker->write_empty=write_empty;
    worker->colinear_mag=colinear_mag; worker->read_startfrom=read_startfrom;
    worker->read_endto=read_endto; worker->read_irate=read_irate;
    worker->outrate=outrate; worker->out_startfrom=out_startfrom; worker->out_endto=out_endto;
    worker->isdummytype=isdummytype; worker->append=append;
    worker->force_tilt=force_tilt;
    worker->write_symbol=write_symbol;
    worker->read_all_keys=read_all_keys; worker->gtags_=gtags_; worker->tags_=tags_;
    worker->session_free_energy_=read_free_energy();
    return worker;
}

void GPIO::build_remap(Atoms &atom) {
    remapid.clear();

    chemmap = atom.chemmap_;
    total_atomsred = atom.pos_.size();
    std::map<int, int> chem_map; //type:count
    for (int i = 0; i < total_atomsred; ++i) {
        chem_map[chemmap[i]]++;
    }
    int chemtypescount = chem_map.size();
    chemcounts.resize(chem_map.size());
    int ct = 0;
    chem.resize(chemtypescount);
    for (std::map<int, int>::iterator it = chem_map.begin(); it != chem_map.end(); ++it) {
        chem[ct] = it->first;
        chemcounts[ct] = it->second;
        ++ct;
    }

    for (int i = 0; i < chemtypescount; ++i) {
        for (int j = 0; j < atom.size(); ++j) {
            if (atom.chemmap_[j] == chem[i]) {
                remapid.push_back(j);
            }
        }
    }
}

/**
 * @brief Convert format string into internal format code
 * 
 * @param input format string
 * @return GPIO::supported_format format code
 */
GPIO::supported_format GPIO::format2enum(const std::string &input) {
    auto it = format_dict.find(input);
    return (it != format_dict.end()) ? it->second : UNKNOWN;
}

GPIO::supported_format GPIO::guess_format(const std::string &instr) {
    supported_format iformat;
    // Drop a compression extension (.gz/.zst) so that e.g. "dump.gz" is
    // detected from its underlying name; the read path handles the
    // decompression itself.
    std::string name = strip_compression_suffix(instr);
    std::string suffix = fs::path(name).extension().string();
    suffix.erase(0, 1);
    // 1. suff exact match
    if (const auto entry=format_dict.find(suffix); !suffix.empty() && entry!=format_dict.end()) {
        iformat = entry->second;
        if (iformat != UNKNOWN) return iformat;
    }
    // 2. suff inexact match TODO: finish this
    // 3. filename match, TODO: refine
    const std::string filename = fs::path(name).filename().string();
#define GPIO_REGSTR(a, b) \
    if (filename.find(a) != std::string::npos) \
        return b
    // if not found, use substring match (ordered: more specific first)
    GPIO_REGSTR("POSCAR", POSCAR);
    GPIO_REGSTR("poscar", POSCAR);
    GPIO_REGSTR("CONTCAR", POSCAR);
    GPIO_REGSTR("contcar", POSCAR);
    GPIO_REGSTR("XDATCAR6", XDATCAR6);
    GPIO_REGSTR("xdatcar6", XDATCAR6);
    GPIO_REGSTR("XDATCAR", XDATCAR);
    GPIO_REGSTR("xdatcar", XDATCAR);
    GPIO_REGSTR("OUTCAR", OUTCAR);
    GPIO_REGSTR("outcar", OUTCAR);
    // GPIO_REGSTR("POUTCAR", POUTCAR);
    // GPIO_REGSTR("poutcar", POUTCAR);
    GPIO_REGSTR("ML_ABN", VASPABN);
    GPIO_REGSTR("ml_abn", VASPABN);
    GPIO_REGSTR("ML_AB", VASPABN);
    GPIO_REGSTR("ml_ab", VASPABN);
    GPIO_REGSTR("vasprun", VASPRUN);
    GPIO_REGSTR("vasp", POSCAR);
    GPIO_REGSTR("gro", GRODATA);
    GPIO_REGSTR("xyz", EXTXYZ);
    GPIO_REGSTR("vtk", VTK);

    GPIO_REGSTR("cfgbin", MLIPBIN);
    GPIO_REGSTR("bincfg", MLIPBIN);
    GPIO_REGSTR("lammpstrj", LMPTRJ);
    GPIO_REGSTR("lmptrj", LMPTRJ);
    GPIO_REGSTR("yaml", LMPYAML);
    GPIO_REGSTR("lmpdat", LMPDAT);
    GPIO_REGSTR("cif", CIF);
    GPIO_REGSTR("cfg", MLPCFG);
    GPIO_REGSTR("xsf", XSF);
    GPIO_REGSTR("bin", MLIPBIN);
    GPIO_REGSTR("msi", MSI);
    GPIO_REGSTR("xsd", XSD);
    GPIO_REGSTR("car", CAR);
#undef GPIO_REGSTR
    // 4. content match, TODO: 
    
    std::cerr << "Warning: cannot determine format for " << instr << std::endl;
    return UNKNOWN;
}

bool GPIO::read(std::vector<Atoms> &atoms, std::string filename, std::string format) {
    if (!validate_read_range()) return false;
    if (reset_status_every_read) ncfgs_ = 0;
    if (ncfgs_ > read_endto) {
        const auto range_format = format.empty() ? guess_format(filename) : format2enum(format);
        if (range_format == MLIPBIN || range_format == CP2KXYZ || range_format == N2P2DAT || range_format == XSF) return true;
    }
    if (!file_exist(filename))
        return false;
    supported_format iformat;
    if (format.empty()) {
        iformat = guess_format(filename);
    } else {
        iformat = format2enum(format);
    }
    bool status = true;
    std::ifstream ifs;
    if (iformat == VASPRUN) {
        // TODO: as above
        #ifdef USEXML
        status = read_vasp_xml(atoms, filename);
        #else 
        std::cerr << "Error: XML support is not enabled, recompile with -DUSEXML to enable it" << std::endl;
        status = false;
        #endif 
        return status;
    }
    // for other format
    try {
        if (probe_compression(filename) != CompressionKind::None ||
            compression_of_extension(filename) != CompressionKind::None) {
            auto is = (iformat == MLIPBIN || iformat == XSF) ? open_seq_ifile(filename) : open_ifile(filename);
            if (!is) {
                std::cerr << "Error: cannot read " << filename << std::endl;
                return false;
            }
            return read(atoms, *is, iformat);
        }
        MMapStreamBuf buf(filename.c_str());
        std::istream is(&buf);
        return read(atoms, is, iformat);
    } catch (const std::runtime_error &error) {
        std::cerr << "Error: cannot read " << filename << ": " << error.what() << std::endl;
        return false;
    }
}

bool GPIO::read(std::vector<Atoms> &atoms, std::istream& is, std::string format)
{
    supported_format iformat = format2enum(format);
    if (iformat == UNKNOWN) {
        std::cerr << "ERROR: format " << format << " is not supported in read()" << std::endl;
        return false;
    }
    return read(atoms, is, iformat);
}

bool GPIO::read_header_dispatch(std::istream &is, supported_format iformat) {
    switch (iformat) {
    case XSF: xsf_read_ = {}; return true;
    case OUTCAR:  return read_vasp_outcar_header(is);
    case XDATCAR: return read_vasp_xdatcar_header(is);
    case VASPABN: return read_vasp_abn_header(is);
    default:      return true;
    }
}

bool GPIO::read_atom_dispatch(std::istream &is, supported_format iformat, Atoms &atom, bool &doneflag) {
    switch (iformat) {
    case POSCAR:
        doneflag = true;
        return read_vasp_poscar(is, atom);
    case POSCARPACKAGE:
        return read_vasp_poscar(is, atom);
    case OUTCAR:
        return load_next_from_vasp_outcar(is, atom, &doneflag);
    case XDATCAR:
        return load_next_from_xdatcar(is, atom, &doneflag);
    case XDATCAR6:
        return load_next_from_xdatcar6(is, atom, &doneflag);
    case VASPABN:
        return load_next_from_vasp_ABN(is, atom);
    case MLPCFG:
        return load_next_from_mlpcfg(is, atom, &doneflag);
    case MLIPBIN:
        return load_next_from_mlpcfg_bin(is, atom, &doneflag);
    case LMPTRJ:
        return load_next_from_lmptrj(is, atom);
    case LMPYAML:
        return load_next_from_lmpyaml(is, atom);
    case LMPDAT:
        doneflag = true;
        return read_lammps_dat(is, atom);
    case CIF:
        doneflag = false;
        return read_cif(is, atom);
    case GRODATA:
        doneflag = true;
        return read_gro_data(is, atom);
    case XSF: {
        const bool status = load_next_from_xsf(is, atom, &doneflag);
        if (status && !xsf_read_.animated) doneflag = true; // Legacy single geometry.
        return status;
    }
    case MSI:
        doneflag = false;
        return read_msi(is, atom);
    case XSD:
        doneflag = true;
        return read_xsd(is, atom);
    case CAR:
        doneflag = true;
        return read_car(is, atom);
    case N2P2DAT:
        return load_next_from_n2p2_data(is, atom, &doneflag);
    case EXTXYZ:
        return load_next_from_extxyz(is, atom, &doneflag);
    case CP2KXYZ:
        return load_next_from_cp2kxyz(is, atom, &doneflag);
    default:
        std::cerr << "Error: format " << iformat << " not supported" << std::endl;
        return false;
    }
}

void GPIO::discard_properties(Atoms &atom) const {
    for (auto &it : discard_props) {
        #define ERASEIT(a) if (a.count(it)) { a.erase(it); }
        ERASEIT(atom.features_)
        ERASEIT(atom.local_attribs_d_)
        ERASEIT(atom.local_attribs_i_)
        ERASEIT(atom.local_attribs_v_)
        ERASEIT(atom.local_attribs_vi_)
        ERASEIT(atom.global_attribs_d_)
        ERASEIT(atom.global_attribs_i_)
        ERASEIT(atom.global_attribs_v_)
        ERASEIT(atom.global_attribs_vi_)
        #undef ERASEIT
        if (it == "forces") atom.forces_.clear();
        else if (it == "virial") atom.have_stress = false;
        else if (it == "site_en") atom.site_energies_.clear();
        else if (it == "energy") atom.have_energy = false;
        else if (it == "fenergy" || it == "free_energy") atom.have_fenergy = false;
    }
}

bool GPIO::read(std::vector<Atoms> &atoms, std::istream& is, supported_format iformat)
{
    if (!validate_read_range()) return false;
    if (iformat == UNKNOWN) return false;
    if ((iformat == MLIPBIN && binary_read_limit_reached()) ||
        ((iformat == CP2KXYZ || iformat == N2P2DAT || iformat == XSF) && ncfgs_ > read_endto)) return true;
    if (!read_header_dispatch(is, iformat)) return false;
    headers_read = true;
    Atoms atom;
    bool doneflag = false;
    bool status = true;
    bool read_frame = false;
    while (!doneflag) {
        // These codecs distinguish I/O failures from a clean inter-frame EOF.
        if (iformat != EXTXYZ && iformat != MLIPBIN && iformat != XSF &&
            iformat != GRODATA && iformat != CP2KXYZ && iformat != N2P2DAT &&
            is.peek() == std::char_traits<char>::eof()) break;
        atom.clear();
        status = read_atom_dispatch(is, iformat, atom, doneflag);
        if (!status) break;
        read_frame = true;
        if (readornot()) {
            atom.id = ncfgs_;
            discard_properties(atom);
            if (absolute_index_read) {
                for (int i = 0; i < atom.size(); ++i)
                    atom.chemmap_[i] = mapnameid[Chemicals::chemical_symbols[atom.chemmap_[i]]];
            }
            atoms.push_back(atom);
            if (iformat != MLIPBIN && iformat != CP2KXYZ && iformat != N2P2DAT && iformat != XSF && ncfgs_ >= read_endto) return true;
        }
        if (status) ncfgs_++;
        if ((iformat == MLIPBIN && binary_read_limit_reached()) ||
            ((iformat == CP2KXYZ || iformat == N2P2DAT || iformat == XSF) && ncfgs_ > read_endto)) return true;
    }
    if (iformat == XDATCAR6 && !read_frame) return false;
    if (!status && (iformat == MLPCFG || iformat == EXTXYZ || iformat == MLIPBIN)) return doneflag;
    if (!status && (iformat == XDATCAR || iformat == XDATCAR6 || iformat == OUTCAR)) return doneflag;
    if (!status && iformat == GRODATA) return false;
    if (!status && (iformat == CP2KXYZ || iformat == N2P2DAT || iformat == XSF)) return doneflag;
    if (!status && is.eof() && atom.size() == 0 && iformat != MLIPBIN)
        status = true;
    return status;
}

// TODO: make thread_safe, currently not thread safe due to use of headers_read and ncfgs_
bool GPIO::read_one(Atoms &atom, std::string filename, std::string format) {
    supported_format iformat;
    if (format.empty()) {
        iformat = guess_format(filename);
    } else {
        iformat = format2enum(format);
    }
    if (iformat == XSF || iformat == GRODATA || iformat == CP2KXYZ || iformat == N2P2DAT) atom.clear();
    if (!file_exist(filename)) return false;
    try {

        if (probe_compression(filename) != CompressionKind::None ||
            compression_of_extension(filename) != CompressionKind::None) {
            auto is = (iformat == MLIPBIN || iformat == XSF) ? open_seq_ifile(filename) : open_ifile(filename);
            if (!is) {
                std::cerr << "Error: cannot read " << filename << std::endl;
                return false;
            }
            return read_one(atom, *is, iformat);
        }
        MMapStreamBuf buf(filename.c_str());
        std::istream is(&buf);
        return read_one(atom, is, iformat);
    } catch (const std::runtime_error &error) {
        std::cerr << "Error: cannot read " << filename << ": " << error.what() << std::endl;
        return false;
    }
}

bool GPIO::read_one(Atoms &atom, std::istream & is, std::string format)
{
    supported_format iformat = format2enum(format);
    if (iformat == UNKNOWN) {
        std::cerr << "ERROR: format " << format << " is not supported in read_one()" << std::endl;
        return false;
    }
    return read_one(atom, is, iformat);
}

bool GPIO::read_one(Atoms &atom, std::istream& is, supported_format iformat)
{

    if (!read_header_dispatch(is, iformat)) return false;
    headers_read = true;
    // EXTXYZ commits only complete frames, preserving the caller on failure.
    if (iformat != EXTXYZ) atom.clear();
    bool dummy_done = false;
    bool status = read_atom_dispatch(is, iformat, atom, dummy_done);
    if (status) {
        atom.id = ncfgs_;
        ncfgs_++;
        if (absolute_index_read) {
            for (int i = 0; i < atom.size(); ++i)
                atom.chemmap_[i] = mapnameid[Chemicals::chemical_symbols[atom.chemmap_[i]]];
        }
    }
    return status;
}

bool GPIO::seq_read_init(std::istream &is, std::string format) {
    seq_readheader = false;
    xdatcar6_header_pending = false;
    xsf_read_ = {};
    // seqinformat = format;
    seq_iformat = format2enum(format);
    ncfgs_ = 0;
    switch (seq_iformat) {
    case OUTCAR:
        seq_readheader = read_vasp_outcar_header(is);
        if (!seq_readheader)
            throw std::runtime_error("Invalid or truncated OUTCAR header");
        return true;
    case LMPTRJ:
        seq_readheader = true;
        return true;
    case LMPYAML:
        seq_readheader = true;
        return true;
    case XDATCAR:
    case XDATCAR6:
        seq_readheader = read_vasp_xdatcar_header(is);
        xdatcar6_header_pending = seq_readheader && seq_iformat == XDATCAR6;
        if (!seq_readheader && seq_iformat == XDATCAR6)
            throw std::runtime_error("Invalid or truncated XDATCAR6 header");
        return seq_readheader;
    case MLPCFG:
        seq_readheader = true;
        return true;
    case MLIPBIN:
        seq_readheader = true;
        return true;
    case EXTXYZ:
    case N2P2DAT:
    case CP2KXYZ:
    case XSF:
    case GRODATA:
        seq_readheader = true;
        return true;
    default:
        std::cerr << "Error: unknown format (when reading, seq) " << format << std::endl;
        return false;
    }
}

bool GPIO::write(std::vector<Atoms> &atoms, std::string filename, std::string format) {
    if (reset_status_every_write) write_icfg = 0;
    supported_format oformat;
    if (format.empty()) {
        oformat = guess_format(filename);
        std::cout << "Output format guessed to be " << oformat << std::endl;
    } else {
        oformat = format2enum(format);
    }
    if (oformat == UNKNOWN) {
        std::cerr << "Error: cannot determine output format for " << filename << std::endl;
        return false;
    }
    if (atoms.empty()) {

        return true;
    }
    if (oformat == XSF && (format == "axsf" || strip_compression_suffix(filename).ends_with(".axsf"))) {
        if (append) {
            std::cerr << "Error: AXSF append is unsupported; write a complete animation\n";
            return false;
        }
        if (outrate <= 0) return false;
        std::vector<const Atoms *> selected;
        for (const auto &frame : atoms) {
            if (writeornot()) selected.push_back(&frame);
            ++write_icfg;
        }
        if (selected.empty()) return true;
        if (!validate_xsf_animation(selected)) return false; // Before touching the target.
        auto output = open_ofile(filename, false);
        if (!output) return false;
        const bool ok = write_xsf_animation(*output, selected);
        const bool closed = close_ofile(output);
        return ok && closed;
    }
    bool ssoutput = false;
    if (oformat == POSCAR || oformat == LMPDAT || oformat == LMPDATFULL || 
           oformat == VTK || oformat == XSF || 
           oformat == CIF || oformat == DLPOLY4) {
        if (force_write_all_to_one) ssoutput = false;
        else ssoutput = true;
    }
    if (!ssoutput) {
        auto ofs = open_ofile(filename, append, oformat == MLIPBIN);
        if (!ofs) {
            std::cerr << "Error: cannot open output file " << filename << std::endl;
            return false;
        }

        if (oformat == XDATCAR && (!append || !headers_write))
            xdatcar_output_header_ = false;
        headers_write = true;
        // read other formats
        for (int i = 0; i < atoms.size(); ++i) {
            if (writeornot()) {
                if (!write_atom_dispatch(*ofs, oformat, atoms[i])) {
                    std::cerr << "Error: failed writing " << filename << std::endl;
                    return false;
                }
            }
            ++write_icfg;
        }
        if (!close_ofile(ofs)) {
            std::cerr << "Error: failed closing output file " << filename << std::endl;
            return false;
        }
    } else {
        for (int i = 0; i < atoms.size(); ++i) {
            if (writeornot()) {
                int id = write_icfg;
                if (suffix_id_use_internal) id = atoms[i].id;
                id += iwrite_offset;
                // keep a compression extension at the end: out.xyz.gz -> out.xyz_0.gz
                std::string fnm;
                if (compression_of_extension(filename) != CompressionKind::None) {
                    std::string ext = filename.substr(filename.rfind('.'));
                    fnm = strip_compression_suffix(filename) + fname_separator + std::to_string(id) + ext;
                } else {
                    fnm = filename + fname_separator + std::to_string(id);
                }
                auto ofs = open_ofile(fnm);
                if (!ofs) {
                    std::cerr << "Error: cannot open output file " << fnm << std::endl;
                    return false;
                }
                if (!write_atom_dispatch(*ofs, oformat, atoms[i])) {
                    std::cerr << "Error: failed writing " << fnm << std::endl;
                    return false;
                }
                if (!close_ofile(ofs)) {
                    std::cerr << "Error: failed closing output file " << fnm << std::endl;
                    return false;
                }
            }
            ++write_icfg;
        }
    }
    return true;
}

bool GPIO::write(std::vector<Atoms>& atoms, std::ostream& os, std::string format)
{
    if (format == "axsf") {
        if (reset_status_every_write) write_icfg = 0;
        if (append || outrate <= 0) return false;
        std::vector<const Atoms *> selected;
        for (const auto &frame : atoms) {
            if (writeornot()) selected.push_back(&frame);
            ++write_icfg;
        }
        return write_xsf_animation(os, selected);
    }
    supported_format oformat = format2enum(format);
    return write(atoms, os, oformat);
}

bool GPIO::write(std::vector<Atoms>& atoms, std::ostream& os, supported_format oformat)
{
    if (reset_status_every_write) write_icfg = 0;
    if (oformat == UNKNOWN) {
        std::cerr << "Error: unknown output format" << std::endl;
        return false;
    }
    if (atoms.empty()) return true;
    if (oformat == XDATCAR && !headers_write) xdatcar_output_header_ = false;
    headers_write = true;

    for (int i = 0; i < atoms.size(); ++i) {
        if (writeornot()) {
            if (!write_atom_dispatch(os, oformat, atoms[i])) return false;
        }
        ++write_icfg;
    }
    return static_cast<bool>(os);
}

bool GPIO::write_one(Atoms &atom, std::string filename, std::string format) {
    supported_format oformat;
    if (format.empty()) {
        oformat = guess_format(filename);
        std::cout << "Output format guessed to be " << oformat << std::endl;
    } else {
        oformat = format2enum(format);
    }
    if (oformat == UNKNOWN) {
        std::cerr << "Error: cannot determine output format for " << filename << std::endl;
        return false;
    }
    auto ofs = open_ofile(filename, append, oformat == MLIPBIN);
    if (!ofs) {
        std::cerr << "Error: cannot open output file " << filename << std::endl;
        return false;
    }
    bool ret = write_one(atom, *ofs, oformat) && static_cast<bool>(*ofs);
    return close_ofile(ofs) && ret;
}

bool GPIO::write_one(Atoms& atom, std::ostream& os, std::string format)
{
    supported_format oformat = format2enum(format);
    if (oformat == UNKNOWN) {
        std::cerr << "Error: unknown output format " << format << std::endl;
        return false;
    }
    return write_one(atom, os, oformat);
}

bool GPIO::write_atom_dispatch(std::ostream &os, supported_format oformat, Atoms &atom) {
    if (oformat==MLPCFG || oformat==MLIPBIN) HessianIO::validate(atom);
    atom.validate_hvp();
    if (atom.has_hvp() && oformat != EXTXYZ && oformat != MLPCFG && oformat != MLIPBIN)
        throw std::invalid_argument("HVP labels require extxyz, cfg or mlip binary output");
    for (const auto &key : {"hvp_directions", "hvp_targets"})
        if (atom.features_.count(key) || atom.global_attribs_v_.count(key) ||
            atom.global_attribs_d_.count(key) || atom.global_attribs_i_.count(key) ||
            atom.global_attribs_vi_.count(key))
            throw std::invalid_argument("HVP tag names are reserved for paired HVP data");
    std::vector<int> original_chemmap;
    if (absolute_index_write) {
        original_chemmap = atom.chemmap_;
        for (int i = 0; i < atom.size(); ++i)
            atom.chemmap_[i] = Chemicals::atomic_numbers[mapidname[atom.chemmap_[i]]];
    }

    bool ok = true;
    switch (oformat) {
    case MLPCFG:    write_next_mlpcfg(os, atom);       break;
    case MLIPBIN:   write_next_mlpcfg_bin(os, atom);   break;
    case LMPTRJ:    write_next_lammps_trj(os, atom);   break;
    case XDATCAR:   ok = write_vasp_xdatcar_sequence(os, atom); break;
    case XDATCAR6:  write_next_xdatcar6(os, atom);     break;
    case EXTXYZ:    write_next_extxyz(os, atom);       break;
    case VASPABN:   write_next_vasp_ABN(os, atom);     break;
    case CP2KXYZ:   write_next_cp2kxyz(os, atom);      break;
    case N2P2DAT:   ok = write_n2p2(os, atom);          break;
    case POSCAR:    ok = write_vasp_poscar(os, atom);        break;
    case XSF:       ok = write_xsf(os, atom);               break;
    case LMPDAT:    ok = write_lammps_dat(os, atom);        break;
    case LMPDATFULL:ok = write_lammps_dat_full(os, atom);   break;
    case VTK:       ok = write_vtk(os, atom);               break;
    case CIF:       ok = write_cif(os, atom);               break;
    case DLPOLY4:   ok = write_dlpoly4(os, atom);           break;
    case UNKNOWN:
    default:
        std::cerr << "Error: UNKNOWN format (when writing) " << oformat << std::endl;
        ok = false;
        break;
    }

    if (absolute_index_write)
        atom.chemmap_ = std::move(original_chemmap);
    return ok && static_cast<bool>(os);
}

bool GPIO::write_one(Atoms& atom, std::ostream& os, supported_format oformat)
{
    if (oformat == XDATCAR) {
        xdatcar_output_header_ = false;
    }
    return write_atom_dispatch(os, oformat, atom);
}

bool GPIO::seq_read(std::istream &is, Atoms &atom) {
    atom.clear();
    if (!seq_readheader) {
        return false;
    }
    bool status = false;
    switch (seq_iformat)
    {
    case OUTCAR: {
        bool clean_eof = false;
        try {
            status = load_next_from_vasp_outcar(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error("Invalid, unconverged or truncated OUTCAR frame " + std::to_string(ncfgs_));
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw;
        }
        if (!status) seq_readheader = false;
        break;
    }
    case XDATCAR: {
        bool clean_eof = false;
        try {
            status = load_next_from_xdatcar(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error("Invalid or truncated XDATCAR frame " + std::to_string(ncfgs_));
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw;
        }
        if (!status) seq_readheader = false;
        break;
    }
    case XDATCAR6: {
        bool clean_eof = false;
        if (xdatcar6_header_pending) {
            xdatcar6_header_pending = false;
            // seq_read_init already consumed this frame's header. EOF here
            // is a truncated first frame, not the end of a trajectory.
            status = load_next_from_xdatcar(is, atom);
        } else {
            status = load_next_from_xdatcar6(is, atom, &clean_eof);
        }
        if (!status) {
            seq_readheader = false;
            if (!clean_eof)
                throw std::runtime_error("Invalid or truncated XDATCAR6 frame " + std::to_string(ncfgs_));
        }
        break;
    }
    case MLPCFG: {
        bool clean_eof = false;
        try {
            status = load_next_from_mlpcfg(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error("Invalid or truncated MLPCFG frame " + std::to_string(ncfgs_));
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw;
        }
        if (!status) seq_readheader = false;
        break;
    }
    case XSF: {
        bool clean_eof = false;
        try {
            status = load_next_from_xsf(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error("XSF invalid or truncated frame " + std::to_string(ncfgs_));
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw;
        }
        if (!status) seq_readheader = false;
        break;
    }
    case GRODATA: {
        bool clean_eof = false;
        try {
            status = load_next_from_gro(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error("GRO invalid or truncated frame " + std::to_string(ncfgs_));
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw;
        }
        if (!status) seq_readheader = false;
        break;
    }
    case N2P2DAT:
    case CP2KXYZ: {
        bool clean_eof = false;
        try {
            status = seq_iformat == N2P2DAT
                ? load_next_from_n2p2_data(is, atom, &clean_eof)
                : load_next_from_cp2kxyz(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error(std::string(seq_iformat == N2P2DAT ? "N2P2" : "CP2KXYZ") +
                                         " invalid or truncated frame " + std::to_string(ncfgs_));
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw;
        }
        if (!status) seq_readheader = false;
        break;
    }
    case MLIPBIN: {
        bool clean_eof = false;
        try {
            status = load_next_from_mlpcfg_bin(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error("invalid, truncated or unreadable frame");
        } catch (const std::exception &error) {
            atom.clear();
            seq_readheader = false;
            throw std::runtime_error("MLIPBIN frame " + std::to_string(ncfgs_) + ": " + error.what());
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw std::runtime_error("MLIPBIN frame " + std::to_string(ncfgs_) + ": input failure");
        }
        if (!status) seq_readheader = false;
        break;
    }
    case LMPTRJ:
        status = read_lammps_trj_sequence(is, atom);
        break;
    case LMPYAML:
        status = read_lammps_yaml_sequence(is, atom);
        break;
    case EXTXYZ: {
        bool clean_eof = false;
        try {
            status = load_next_from_extxyz(is, atom, &clean_eof);
            if (!status && !clean_eof)
                throw std::runtime_error("invalid or truncated frame");
        } catch (const std::bad_alloc &) {
            atom.clear();
            seq_readheader = false;
            throw;
        } catch (const std::exception &error) {
            atom.clear();
            seq_readheader = false;
            throw std::runtime_error("EXTXYZ frame " + std::to_string(ncfgs_) + ": " + error.what());
        } catch (...) {
            atom.clear();
            seq_readheader = false;
            throw;
        }
        if (!status) seq_readheader = false;
        break;
    }
    default:
        return false;
        break;
    }
    if (status) {
        if (seq_iformat == EXTXYZ || seq_iformat == MLPCFG ||
            seq_iformat == MLIPBIN || seq_iformat == OUTCAR ||
            seq_iformat == N2P2DAT || seq_iformat == XSF)
            discard_properties(atom);
        atom.id = ncfgs_;
        ncfgs_++;
        if (absolute_index_read) {
            for (int i = 0; i < atom.size(); ++i) {
                atom.chemmap_[i] = mapnameid[Chemicals::chemical_symbols[atom.chemmap_[i]]];
            }
        }
    }
    return status;
}

// The caller is responsible for opening the output stream (e.g. via
// open_ofile, which transparently compresses .gz output files);
// outfile is used only to guess the format when format is empty.
bool GPIO::seq_write_init(std::ostream &ofs, std::string format, std::string outfile) {
    ncfgs_ = 0;
    if (format == "")
        seq_oformat = guess_format(outfile);
    else
        seq_oformat = format2enum(format);
    if (seq_oformat == UNKNOWN) {
        std::cerr << "Error: unknown sequential output format " << format << std::endl;
        return false;
    }

    seq_writeheader = false;
    xdatcar_output_header_ = false;

    switch (seq_oformat)
    {
    case MLPCFG:
    case MLIPBIN:
    case LMPTRJ:
    case EXTXYZ:
    case CP2KXYZ:
    case N2P2DAT:
    case XDATCAR6:
        seq_writeheader = true;
        return true;
    case XDATCAR:
        seq_writeheader = false;
        return true;
    default:
        return false;
    }
}

bool GPIO::seq_write(std::ostream &ofs, Atoms &atom) {
    switch (seq_oformat)
    {
    case MLPCFG:
        return write_mlpcfg_sequence(ofs, atom);
    case MLIPBIN:
        return write_mlpcfg_bin_sequence(ofs, atom);
    case XDATCAR:
        return write_vasp_xdatcar_sequence(ofs, atom);
    case XDATCAR6:
        return write_vasp_xdatcar6_sequence(ofs, atom);
    case EXTXYZ:
        return write_extxyz_sequence(ofs, atom);
    case CP2KXYZ:
        return write_atom_dispatch(ofs, CP2KXYZ, atom);
    case N2P2DAT:
        return write_atom_dispatch(ofs, N2P2DAT, atom);
    case LMPTRJ:
        write_next_lammps_trj(ofs, atom);
        return static_cast<bool>(ofs);
    
    default:
        return false;
    }
}

void GPIO::clean_state() {
    ncfgs_ = 0;
    headers_read = false;
    headers_write = false;
    seq_readheader = false;
    xdatcar6_header_pending = false;
    xsf_read_ = {};
    seq_writeheader = false;
    xdatcar_output_header_ = false;
}

void GPIO::InitSettings()
{
    Connect(r_extent, "npbc_extent");
}
