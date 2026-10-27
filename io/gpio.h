// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/**
 * @file gpio.h
 * @brief General-Purpose I/O driver for reading and writing atomic structure files.
 *
 * Supports a wide range of formats: VASP (POSCAR, OUTCAR, XDATCAR, vasprun.xml),
 * LAMMPS (data, trajectory, YAML dump), MLIP's mlpcfg (text and binary),
 * extended XYZ, CIF, XSF, CP2K XYZ, GROMACS, DLPOLY, VTK,
 * and Boost.Serialization (XML/text/binary).
 *
 * Both single-shot (read_one/write_one) and sequential (seq_read/seq_write)
 * modes are available. The GPIO class inherits ConnectSettings for ini-style
 * configuration.
 *
 * @note Units: eV and Angstrom.
 * @bug MPI problems with atom IDs exist.
 * @todo MPI read/write.
 * Explicit unit conversion and isolated stream sessions are available in
 * atoms_units.h and gpio_session.h. Legacy GPIO objects retain mutable state.
 */

#pragma once

#define INDEXFILE "index.map"
#define LINE_BUFFER_SIZE 2000
#define SPLIT_BUFFER_SIZE 200

#include "base/structures.h"
#include "utils/gputils.h"
#include <memory>
#include <optional>
#include <initializer_list>
#include <cstdint>
#include <sys/stat.h>
// for reading vasprun.xml
#define VASP_ENERGY_USE_FR
//#define VASP_ENERGY_USE_SIG0
#define VASP_STRESS_USE_PR // stress * volume (pressure), not stress itself

std::string_view skipLeadingSpaces(const std::string& input);

/**
 * @brief General-purpose I/O driver for atomic structures (the standard reader/writer).
 * @ingroup io
 *
 * Supports reading and writing atoms in numerous formats. Key public methods:
 * - read() / read_one(): Load structures from file or stream
 * - write() / write_one(): Save structures to file or stream
 * - seq_read_init() / seq_read(): Sequential reading for large trajectories
 * - seq_write_init() / seq_write(): Sequential writing
 * - guess_format() / format2enum(): Format detection from filename or string
 *
 * Both text and binary formats are supported internally. The class tracks
 * chemical species via an integer-to-string map (see setmap, getmap).
 *
 * @see Atoms, ConnectSettings
 */
class GPIO : public ConnectSettings
{
private:
    // Per-parser state; concurrent streams use separate GPIO sessions.
    Eigen::Matrix3d cell_;
    std::vector<int> chem;
    std::vector<int> chemmap;
    int chemtypescount;
    int total_atomsred;
    std::vector<int> chemcounts;
    int curr_id = 0;
    std::string tmp_string = "LATTICE";

    // buffer
    char linebuffer[LINE_BUFFER_SIZE];
    char *splitbuffer[SPLIT_BUFFER_SIZE];
    std::vector<double> buffer_double_v_;
    std::vector<int> buffer_int_v_;
    std::string line_buffer, line_buffer2;

    // tools
    std::vector<int> remapid; // for vasp output, ie POSCAR
    bool seq_readheader = false;
    bool xdatcar6_header_pending = false;
    struct XSFReadState {
        bool initialized = false, animated = false, molecular = false;
        bool variable_cell = false, at_eof = false;
        int expected_steps = 1, expected_atoms = -1;
        std::int64_t next_step = 1;
        std::array<bool, 3> pbc{{false, false, false}};
        Eigen::Matrix3d cell = Eigen::Matrix3d::Zero();
        std::string pending;
    } xsf_read_;

    bool seq_writeheader = false;
    // Fixed XDATCAR output state must survive reads on this same GPIO object.
    bool xdatcar_output_header_ = false;
    Eigen::Matrix3d xdatcar_output_cell_ = Eigen::Matrix3d::Zero();
    std::vector<int> xdatcar_output_species_, xdatcar_output_counts_;
    bool headers_read = false;
    bool headers_write = false;
    int ncfgs_ = 0;
    int write_icfg = 0;
    bool validate_read_range() const;
    std::optional<bool> session_free_energy_;
    std::string informat;
    std::string outformat;
    std::map<std::string, int> mapnameid;
    std::map<int, std::string> mapidname;

    // for lammps
    bool lmp_phrase_tag_always = false;
    bool lmp_tag_phrased = false;
    // id type x y z fx fy fz vx vy vz ix iy iz mol q  mx my mz grad site_en
    // 0  1    2 3 4 5  6  7  8  9  10 11 12 13 14  15 16 17 18 19   20    
    std::array<int, 32> lmp_common_tagid; 
    std::unordered_map<std::string, size_t> lmp_tagmap, lmp_tagvecsize;
    std::unordered_map<std::string, std::vector<int>> lmp_tagvecmap; // vector value temp
    std::vector<int> lmp_typeinfo; // for lammps and extxyz, each col type info, 0 - int, 1 - float, -1 - string
    std::vector<double> lmp_readbuffer;
    struct {
        bool has_id = false, has_type = false, has_coord = false, has_x = false, has_y = false, has_z = false;
        bool has_vx = false, has_vy = false, has_vz = false;
        bool has_mol = false;
        bool has_fx = false, has_fy = false, has_fz = false;
        bool has_ix = false, has_iy = false, has_iz = false;
        bool wix = false, wiy = false, wiz = false;
        bool has_q = false, has_mx = false, has_my = false, has_mz = false;
        bool has_siteen = false;
        int nvals = 0;
        void reset() {
            has_id = false; has_type = false; has_coord = false; has_x = false; has_y = false; has_z = false;
            has_vx = false; has_vy = false; has_vz = false;
            has_mol = false;
            has_fx = false; has_fy = false; has_fz = false;
            has_ix = false; has_iy = false; has_iz = false;
            wix = false; wiy = false; wiz = false; 
            has_q = false;
            has_mx = false; has_my = false; has_mz = false;
            has_siteen = false;
            nvals = 0;
        }
        bool have_force() {return has_fx || has_fy || has_fz;}
        bool have_velo() {return has_vx || has_vy || has_vz;}
    } lmp_tag_flags;
    // Flags for lmp_parse_tags() to control which tag groups to process
    enum LmpTagFlags : unsigned {
        LMP_TAG_SCALED_COORDS  = 1 << 0,  // xs/xsu/ys/ysu/zs/zsu
        LMP_TAG_IMAGE_FLAGS    = 1 << 1,  // ix/iy/iz
        LMP_TAG_MOL            = 1 << 2,  // mol
        LMP_TAG_SITEEN         = 1 << 3,  // site_en
        LMP_TAG_MAGMOM         = 1 << 4,  // magmom_x/y/z
        LMP_TAG_BRACKET_VEC    = 1 << 5,  // v_name[1] style bracket vectors
        LMP_TAG_GUARDED_COORDS = 1 << 6,  // Use !has_x guard for coordinate assignment
        LMP_TAG_CARTES_ALIAS   = 1 << 7,  // Use cartes_x/y/z alias (vs xu/yu/zu)
        LMP_TAG_INIT_VECMAPS   = 1 << 8,  // Also clear lmp_tagvecmap/size
    };
    template<typename Tok>
    bool lmp_parse_tags(Tok &tok, unsigned flags);

    int nelm_, nsw_, isif_, ibrion_; // for vasp
    bool is_vasp_ml = false;
    double ediff_, ediffg_;
    int vasp_ver = 5;
    bool read_vasp_outcar_header(std::istream &is);
    bool read_vasp_outcar_attrib(std::istream &is, std::string attrib, int &att);
    bool read_vasp_outcar_attrib(std::istream &is, std::string attribute, double &att);
    bool load_next_from_lmptrj(std::istream &is, Atoms &atom);
    bool load_next_from_lmpyaml(std::istream &is, Atoms &atom);
    // bool load_next_from_lmpyaml(std::istream &is, Atoms &atom);
    bool load_next_from_mlpcfg(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_mlpcfg_bin(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_vasp_outcar(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_xdatcar(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_xdatcar6(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_n2p2_data(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_xsf(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_extxyz(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_cp2kxyz(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);
    bool load_next_from_vasp_ABN(std::istream &is, Atoms &atom);

    void write_next_mlpcfg(std::ostream &os, Atoms &atom);
    void write_next_mlpcfg_bin(std::ostream &os, Atoms &atom);
    void write_next_lammps_trj(std::ostream &os, Atoms &atom);
    void write_next_xdatcar6(std::ostream &os, Atoms &atom);
    void write_next_vasp_ABN(std::ostream &os, Atoms &atom);
    void write_next_extxyz(std::ostream &os, Atoms &atom);
    void write_next_cp2kxyz(std::ostream &os, Atoms &atom);

protected:
    void discard_properties(Atoms &atom) const;
    bool read_mlpcfg_sequence(std::istream &is, Atoms &atom);
    bool read_mlpcfg_bin_sequence(std::istream &is, Atoms &atom);
    bool read_lammps_trj_sequence(std::istream &is, Atoms &atom);
    bool read_lammps_yaml_sequence(std::istream &is, Atoms &atom);
#ifdef USEXML
    bool read_vasp_xml(std::vector<Atoms> &atoms, std::string filename);
#endif
    bool read_vasp_outcar_sequence(std::istream &is);
    bool read_vasp_xdatcar_header(std::istream &is, bool comment_read = false, bool *clean_eof = nullptr);
    bool read_vasp_abn_header(std::istream &is);
    bool read_vasp_xdatcar_sequence(std::istream &is, Atoms &atom);
    bool read_extented_xyz_sequence(std::istream &is, Atoms &atom);
    bool read_gro_data(std::istream &is, Atoms &atom);
    bool load_next_from_gro(std::istream &is, Atoms &atom, bool *clean_eof = nullptr);

    bool write_lammps_trj_sequence(std::ostream &os);
    bool write_vasp_xdatcar6_sequence(std::ostream &os, Atoms &atom);
    void write_vasp_xdatcar_header(std::ostream &os, Atoms &atom);
    bool write_vasp_xdatcar_sequence(std::ostream &os, Atoms &atom);
    bool write_vasp_xdatcar_positions(std::ostream &os, Atoms &atom,
                                     const Eigen::Matrix3d &cell);
    bool write_mlpcfg_sequence(std::ostream &os, Atoms &atom);
    bool write_mlpcfg_bin_sequence(std::ostream &os, Atoms &atom);
    bool write_vtk(std::ostream &os, Atoms &atom);
    bool write_vesta(std::ostream &os, Atoms &atom);
    bool write_dlpoly4(std::ostream &os, Atoms &atom);
    bool write_extxyz_sequence(std::ostream &os, Atoms &atom);

    bool read_vasp_poscar(std::istream &is, Atoms &atom);
    bool read_cif(std::istream &is, Atoms &atom);
    bool read_lammps_dat(std::istream &is, Atoms &atom);
    bool read_xsf(std::istream &is, Atoms &atom);
    bool read_msi(std::istream &is, Atoms &atom);
    bool read_car(std::istream &is, Atoms &atom);
    bool read_xsd(std::istream &is, Atoms &atom);

    bool write_lammps_dat(std::ostream &os, Atoms &atom);
    bool write_lammps_dat_full(std::ostream &os, Atoms &atom);
    bool write_vasp_poscar(std::ostream &os, Atoms &atom);
    bool write_xsf(std::ostream &os, Atoms &atom);
    bool validate_xsf_animation(const std::vector<const Atoms *> &frames) const;
    bool write_xsf_animation(std::ostream &os, const std::vector<const Atoms *> &frames) const;
    bool write_n2p2(std::ostream &os, Atoms &atom);
    bool write_cif(std::ostream &os, Atoms &atom);

    bool readornot();
    bool writeornot();

public:
    /// @brief Supported file formats enumeration
    enum supported_format {
        UNKNOWN,         ///< Unknown/undetected format
        MLPCFG,          ///< MLIP's text cfg format
        CFG,             ///< Generic cfg format
        EXTXYZ,          ///< Extended XYZ format (per-atom properties)
        SIMPLEXYZ,       ///< Simple XYZ format
        XSF,             ///< XCrySDen's XSF format
        CIF,             ///< Crystallographic Information File
        OUTCAR,          ///< VASP OUTCAR (v5.x supported; v6.x limited)
        POSCAR,          ///< VASP POSCAR
        VASPRUN,         ///< VASP vasprun.xml (v6.3.x)
        MLIPBIN,         ///< MLIP binary cfg format
        XDATCAR,         ///< VASP XDATCAR (fixed cell trajectory)
        XDATCAR6,        ///< VASP XDATCAR v6 (variable cell trajectory)
        N2P2DAT,         ///< n2p2 training dataset format
        VASPABN,         ///< VASP ML_ABN configuration format (v6.2.x)
        LMPTRJ,          ///< LAMMPS dump trajectory
        LMPDAT,          ///< LAMMPS data file
        LMPYAML,         ///< LAMMPS YAML dump
        POSCARPACKAGE,   ///< Multiple POSCARs in one file
        GRODATA,         ///< GROMACS data file
        CP2KXYZ = 21,    ///< CP2K XYZ format (preserve existing format IDs)
        LMPDATFULL,      ///< LAMMPS data file (full format with all fields)
        VTK,             ///< VTK format (ParaView)
        DLPOLY4,         ///< DLPOLY4 structure file
        MSI,             ///< Materials Studio / Cerius2 MSI DataModel format
        XSD,             ///< Materials Studio XML structure format
        CAR,             ///< Materials Studio / Biosym CAR archive format
    };
    supported_format seq_iformat;
    supported_format seq_oformat;

    bool read_header_dispatch(std::istream &is, supported_format iformat);
    bool read_atom_dispatch(std::istream &is, supported_format iformat, Atoms &atom, bool &doneflag);
    bool write_atom_dispatch(std::ostream &os, supported_format oformat, Atoms &atom);

    bool drop; //drop non-converged structures
    bool lossless = true;
    bool write_virtual = true;
    bool write_stress = false; // ASE's convension
    bool write_mass = false; // write mass in lammps data file, default false

    static bool overwrite;
    static bool backupold;
    bool packatoms = true; // pack atom types in POSCAR

    double r_extent = 5.0; // for case if cell is missing, add a large cell

    std::vector<std::string> discard_props;

    // see if file exists, overwrite or backup it 
    // static std::ostream &create_ofs(std::ostream &os, std::string fnm);

    bool force_write_all_to_one = false; // force write all images to a single file, for format like POSCAR
    bool reset_status_every_write = true;
    bool reset_status_every_read = true;
    bool suffix_id_use_internal = false; // atom.id instead of write_icfg
    // bool standalong_format_piped_out = true; // set to false so that POSCAR will not write as POSCAR_0 when ncfg is exactly one
    std::string fname_separator = ""; // for POSCAR multiple output, POSCAR_0, POSCAR0 POSCAR-0, etc
    std::string header_line = "Generated by GPIO"; // for output file support headers, write headers info
    bool cfg_have_efs;
    bool cfg_have_mag = false;
    bool read_mag = false;
    bool write_empty = false;
    bool colinear_mag = true;
    // Bulk read selection uses zero-based, inclusive bounds and a positive stride.
    // INT_MAX is a valid upper bound; a start beyond the input yields no frames.
    int read_startfrom = 0;
    int read_endto = INT_MAX;
    int read_irate = 1;
    /// Whether a bulk MLIPBIN read has consumed its inclusive end bound.
    bool binary_read_limit_reached() const { return ncfgs_ > read_endto; }
    int outrate = 1;
    int out_startfrom = 0;  //output
    int out_endto = INT_MAX;      //output, include endto
    int iwrite_offset = 0; // output index offset, usually use 0 or 1
    bool outdirect = false; // weather output use direct coordinate
    bool isdummytype;
    bool append;
    bool force_tilt;
    bool in_seq_write_state = false;
    bool in_seq_read_state = false;
    static bool vasp_use_free_eng;
    std::map<int, double> atomradii;
    std::map<int, std::vector<double>> atomic_vals;
    static std::map<std::string, supported_format> format_dict;
    GPIO();
    /// Session workers register local settings only, without changing CLI bindings.
    explicit GPIO(bool bind_console);
    /// Fresh parser with independent bindings and a snapshot of this configuration.
    std::unique_ptr<GPIO> new_session_state() const;
    bool read_free_energy() const { return session_free_energy_.value_or(vasp_use_free_eng); }
    void set_read_free_energy(bool value) { session_free_energy_ = value; }

    /// @name Single-shot I/O
    /// @{

    /// @brief Read all structures from file into a vector (auto-detects format from extension)
    bool read(std::vector<Atoms> &atoms, std::string filename, std::string format = "");
    /// @brief Read all structures from stream with format string
    bool read(std::vector<Atoms> &atoms, std::istream &is, std::string format);
    /// @brief Read all structures from stream with format enum
    bool read(std::vector<Atoms> &atoms, std::istream &is, supported_format format);
    /// @brief Read a single structure from file
    bool read_one(Atoms &atom, std::string filename, std::string format = "");
    /// @brief Read a single structure from stream with format string
    bool read_one(Atoms &atom, std::istream &is, std::string format);
    /// @brief Read a single structure from stream with format enum
    bool read_one(Atoms &atom, std::istream &is, supported_format format);
    /// @brief Reset internal reader/writer state
    void clean_state();
    /// @brief Write all structures to file
    bool write(std::vector<Atoms> &atoms, std::string filename, std::string format = "");
    /// @brief Write all structures to stream with format string
    bool write(std::vector<Atoms> &atoms, std::ostream &os, std::string format);
    /// @brief Write all structures to stream with format enum
    bool write(std::vector<Atoms> &atoms, std::ostream &os, supported_format oformat);
    /// @brief Write a single structure to file (for POSCAR, CIF, VTK, etc.)
    bool write_one(Atoms &atom, std::string filename, std::string format = "");
    /// @brief Write a single structure to stream with format string
    bool write_one(Atoms &atom, std::ostream &os, std::string format);
    /// @brief Write a single structure to stream with format enum
    bool write_one(Atoms &atom, std::ostream &os, supported_format oformat);
    /// @}

    /// @name Sequential I/O (for large trajectories)
    /// @{

    /// @brief Initialize sequential reading from stream
    bool seq_read_init(std::istream &is, std::string format);
    /// @brief Read next structure sequentially
    bool seq_read(std::istream &is, Atoms &atom);
    /// @brief Initialize sequential writing to stream.
    ///        The caller must open the stream first (e.g. via open_ofile,
    ///        which transparently compresses .gz output files);
    ///        outfile is used only to guess the format when format is empty.
    bool seq_write_init(std::ostream &ofs, std::string format, std::string outfile = "");
    /// @brief Write next structure sequentially
    bool seq_write(std::ostream &os, Atoms &atom);
    /// @}
#ifdef USEXML
    static bool read_vasp_xml(const char *filename, Atoms &atom);
#endif
    /// @brief Set output rate: every N structures, from start to end
    void setoutrate(int rate = 1, int start = 0, int end = INT_MAX);
    /// @brief Build internal chemistry map from current data
    void setmap(void);
    /// @brief Set element names in species-index order (empty restores atomic numbers).
    void setmap(const std::vector<std::string> &elements);
    void setmap(std::initializer_list<std::string> elements) { setmap(std::vector<std::string>(elements)); }
    /// @brief Restore exact species IDs, including nonconsecutive IDs.
    void setmap(const std::map<int, std::string> &elements);

    /// @brief Read chemistry map from file
    bool readmap(std::string mapfile);
    /// @brief Guess file format from filename extension
    static supported_format guess_format(const std::string &input);
    /// @brief Convert format name string to enum
    static supported_format format2enum(const std::string &input);

    /// @brief Get chemistry map (integer ID -> element name)
    inline std::map<int, std::string> getmap(void) const { return mapidname; };
    /// @brief Get reverse chemistry map (element name -> integer ID)
    inline std::map<std::string, int> getrevmap(void) const { return mapnameid; };
    static inline bool file_exist(const std::string &name)
    {
        struct stat fbuf;
        return (stat(name.c_str(), &fbuf) == 0);
    }
    std::vector<int> &report_chem(std::vector<int> &ret)
    {
        ret.clear();
        for (auto &it : mapidname)
            ret.push_back(it.first);
        return ret;
    }
    std::string get_type(int t)
    {
        return mapidname[t];
    }

    bool absolute_index_read = false; // write atomic numbers instead of 0,1,2,...
    bool absolute_index_write = false;
    bool write_symbol = true; // for format like xsf, xyz, write symbol or atomic number

    std::set<std::string> tags_; // used in lammps and extented xyz format, per atom
    std::set<std::string> gtags_; // global attrib tags for extented xyz format
    bool read_all_keys = true;

    void set_header(const std::string &line) { header_line = line; }

    void build_remap(Atoms &atom);

    // todo: complete this
    void InitSettings();
    void InitBySettings(GPSettings settings) {
        InitSettings();
        ApplySettings(settings);
    }
    GPIO(GPSettings settings) : GPIO() {
        InitBySettings(settings);
    }
};
