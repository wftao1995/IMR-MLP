// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mindriver/relax_new.h"
#include "mindriver/relax_constraints_io.h"
#include "mindriver/al_modifier.h"
#include "active_learn/al_potential.h"
#include "potentials/potential_wrapper.h"
#include "io/gpio.h"
#include "utils/compressed_stream.h"
#include "tools/procstr.h"
#include "tools/tools.h"
#include "utils/gputils.h"
#include "help_utils.h"
#include "relax_masks.h"
#include <cmath>
#include <filesystem>
#include <stdexcept>
using namespace std;

void AppsRelax::read_masks(std::istream &input, std::vector<Atoms> &atoms) {
    size_t line = 0;
    auto fail = [&](const std::string &message) {
        throw std::invalid_argument("Invalid seldyn mask at line " + std::to_string(line) + ": " + message);
    };
    std::vector<std::string> pending;
    auto next_row = [&]() {
        if (!pending.empty()) {
            auto values = std::move(pending);
            pending.clear();
            return values;
        }
        std::string text;
        std::vector<std::string> values;
        while (std::getline(input, text)) {
            ++line;
            trim_comment(text, "#!");
            trim_all(text);
            if (text.empty()) continue;
            str_split(text, values);
            return values;
        }
        return values;
    };
    auto read_row = [&](size_t width) {
        auto values = next_row();
        if (values.empty()) fail("unexpected end of file");
        if (values.size() != width) fail("expected " + std::to_string(width) + " fields");
        return values;
    };
    auto integer = [&](const std::string &value) {
        size_t used = 0;
        int result = 0;
        try { result = std::stoi(value, &used); }
        catch (const std::exception &) { fail("invalid integer: " + value); }
        if (used != value.size()) fail("invalid integer: " + value);
        return result;
    };
    auto mask_value = [&](const std::string &value) {
        size_t used = 0;
        double result = 0;
        try { result = std::stod(value, &used); }
        catch (const std::exception &) { fail("invalid mask value: " + value); }
        if (used != value.size() || !std::isfinite(result)) fail("mask values must be finite numbers");
        return result;
    };

    const int count = integer(read_row(1)[0]);
    if (count < 0 || static_cast<size_t>(count) != atoms.size()) fail("configuration count does not match input structures");
    std::vector<bool> seen(atoms.size(), false);
    std::vector<Eigen::Matrix3d> cell_masks(atoms.size());
    std::vector<std::vector<Eigen::RowVector3d>> position_masks(atoms.size());
    for (int record = 0; record < count; ++record) {
        const auto header = read_row(2);
        const int configuration = integer(header[0]);
        const int atom_count = integer(header[1]);
        if (configuration < 1 || static_cast<size_t>(configuration) > atoms.size()) fail("configuration ID is out of range (one-based)");
        const size_t index = configuration - 1;
        if (seen[index]) fail("duplicate configuration ID");
        if (atom_count < 0 || atom_count != atoms[index].size()) fail("atom count does not match structure");
        seen[index] = true;
        for (int row = 0; row < 3; ++row) {
            const auto values = read_row(3);
            for (int k = 0; k < 3; ++k) cell_masks[index](row, k) = mask_value(values[k]);
        }
        position_masks[index].assign(atom_count, Eigen::RowVector3d::Ones());
        std::vector<bool> atoms_seen(atom_count, false);
        for (auto values = next_row(); !values.empty(); values = next_row()) {
            if (values.size() == 2) {
                pending = std::move(values);
                break;
            }
            if (values.size() != 4) fail("expected atom ID and three mask components");
            const int atom_id = integer(values[0]);
            if (atom_id < 0 || atom_id >= atom_count) fail("atom ID is out of range (zero-based)");
            if (atoms_seen[atom_id]) fail("duplicate atom ID");
            atoms_seen[atom_id] = true;
            for (int k = 1; k < 4; ++k) position_masks[index][atom_id](k - 1) = mask_value(values[k]);
        }
    }
    if (!pending.empty()) fail("unexpected trailing configuration");
    std::string text;
    while (std::getline(input, text)) {
        ++line;
        trim_comment(text, "#!");
        trim_all(text);
        if (!text.empty()) fail("unexpected trailing record");
    }
    if (input.bad() || (input.fail() && !input.eof())) fail("failed to read file");
    for (size_t i = 0; i < atoms.size(); ++i) {
        atoms[i].sel_dyncell_ = cell_masks[i];
        atoms[i].sel_dynpos_ = std::move(position_masks[i]);
    }
}

void AppsRelax::apply_position_mask(const std::string &text, std::vector<Atoms> &atoms,
                                    RelaxationDrvV2 &driver) {
    std::vector<std::string> values;
    str_split(text, values);
    if (values.size() != 3)
        throw std::invalid_argument("position-mask requires exactly three values (0=fixed, 1=relaxed)");
    Eigen::RowVector3d mask;
    for (int i = 0; i < 3; ++i) {
        size_t used = 0;
        double value;
        try { value = std::stod(values[i], &used); }
        catch (const std::exception &) {
            throw std::invalid_argument("position-mask values must be 0 or 1");
        }
        if (used != values[i].size() || !std::isfinite(value) || (value != 0 && value != 1))
            throw std::invalid_argument("position-mask values must be 0 or 1");
        mask(i) = value;
    }
    if (!driver.flags.relax_atom)
        throw std::invalid_argument("position-mask requires atomic relaxation; incompatible with ISIF 5/6/7 or relax:relax_atom=False");
    if (driver.flags.keep_symm)
        throw std::invalid_argument("position-mask is incompatible with relax:keep_symm");
    if (driver.flags.vibffilter || driver.flags.constraint_vib_modes ||
        driver.flags.linear_constraint || driver.flags.linear_transform)
        throw std::invalid_argument("position-mask is incompatible with VIB/VIB2/LC/LF coordinate modifiers");
    if (!driver.control.seldyn_file.empty() && GPIO::file_exist(driver.control.seldyn_file))
        throw std::invalid_argument("position-mask is incompatible with the optimization-coordinate mask file: " + driver.control.seldyn_file +
                                    "; remove that file or set relax:seldyn_file to a nonexistent path (e.g. NONE)");

    for (auto &atom : atoms)
        atom.sel_dynpos_.assign(atom.size(), mask);
    driver.flags.selective_dynamics = true;
}

class Relaxstr2Params : public ConnectSettings {
public:
    // common input/output
    string inputfile = "POSCAR";
    string inputformat = "";
    string indexfile = "index.map";
    string outputfile = "output-relaxed.cfg";
    string outputformat = "";
    bool seq_write = false;
    bool seq_read = false;
    // potential related
    // string potfile = "";
    // string pottype = "";
    bool create_input = false; // create sample input file

    // control
    string file_seldyn = "";
    std::string file_filter, file_constraints, gpm;
    //               FixCell   FixPos   FixShape   FixVol
    // isif = 2:        T        F         F          F      relax atom positions only
    // isif = 3:        F        F         F          F      full relax
    // isif = 13:       F        F         F          F      full relax, but fix angle
    // isif = 4:        F        F         F          T      fix cell volume
    // isif = 5:        F        T         F          T      fix cell volume and positions
    // isif = 6:        F        T         F          F      fix atom positions
    // isif = 7:        F        T         T          F      fix cell shape and atom positions
    // isif = 8:        F        F         T          F      fix cell shape
    int isif = 0;
    std::string gcm = ""; // global cell mask

    bool write_force = true;
    bool write_stress = true;
    bool write_ateng = true;
    bool discard_all = false;

    // quick reference
    double gtol = -1;
    double stepmax = -1;
    bool selective_dynamics = false;
    std::string strain_mask = "";
    double pressure = 0;
    bool dprecond = false;

    string potfile = "";
    string pottype = "";

    vector<string> tags;
    vector<string> vals;

public:
    Relaxstr2Params() {
        auto &cmdopts = CMDOption::global_cmdopts;
        ConnectWithConsole(cmdopts, inputfile, "input", "in", "Input structure file to be relaxed");
        ConnectWithConsole(cmdopts, inputformat, "input-format", "if", "Input structure file format");
        ConnectWithConsole(cmdopts, outputfile, "output", "out", "Output relaxed structure file");
        ConnectWithConsole(cmdopts, outputformat, "output-format", "of", "Output structure file format");
        ConnectWithConsole(cmdopts, indexfile, "index", "id", "Index file for types mapping");
        ConnectWithConsole(cmdopts, isif, "isif", "isif", "Determines which degrees of freedom are optimized during relaxation (VASP-compatible, support 2-8)");
        ConnectWithConsole(cmdopts, seq_write, "seq-write", "", "write one by one", 0, 2);
        ConnectWithConsole(cmdopts, gcm, "cell-mask", "", "Finite cell selection flags (positive=movable, nonpositive=fixed); 9 row-major values for lattice/nonsymmetric strain or 6 values (xx yy zz yz xz xy) for symmetric strain. Validated after ISIF/tags; requires cell relaxation without fix_shape");
        ConnectWithConsole(cmdopts, file_seldyn, "seldyn", "", "selective dynamics info, for formats other than POSCAR");
        ConnectWithConsole(cmdopts, file_filter, "filter", "", "Full optimizer-coordinate mask file: n followed by exactly n binary 0/1 flags (0=fixed); n must match each frame. Enables selective dynamics and overrides relax:seldyn_file. Cell coordinates come first (1 scale, 6 symmetric-strain components in xx xy xz yy yz zz order, or 9 row-major lattice/strain components), then atomic coordinates in the same space as position-mask. Incompatible with position-mask, seldyn, embedded frozen-atom flags, cell-mask, strain_mask/lattice_mask, keep_symm and VIB/VIB2/LC/LF.");
        ConnectWithConsole(cmdopts, gpm, "position-mask", "", "Three global 0/1 position flags, e.g. '1 0 1' (0=fixed, 1=relaxed). Overwrite atomic seldyn flags in every frame and enable selective dynamics; preserve cell flags. Components follow the optimizer coordinates: current Cartesian when relax:cartesian=True; fractional when False and relax:atomic_force_recast=False; reference-cell Cartesian (fractional * reference cell) otherwise. Incompatible with keep_symm, ISIF 5/6/7, relax_atom=False, VIB/VIB2/LC/LF and an active seldyn_file.");
        ConnectWithConsole(cmdopts, file_constraints, "file-constraints", "", "JSON V1 geometry equalities; V2 adds physical bounds, walls, formulas, dihedrals and selected initial symmetry operations. driver=sqp|nlc uses modern SQP. CPU, single MPI rank, batch or seq-read. Omitted targets bind each frame's initial geometry. Atomic constraints support ordinary position masks/frozen atoms and supported cell coordinate modes; lattice constraints retain raw9 and mask restrictions. See the relax reference for schema and residual units", 1, 10);
        ConnectWithConsole(cmdopts, seq_read, "seq-read", "", "Relax and write one frame at a time. Input: mlpcfg/XDATCAR6/extxyz/n2p2; output: mlpcfg/extxyz, including .gz/.zst. Supports file-constraints and preserves completed output on a later frame failure. Implies sequential output; incompatible with external seldyn/filter and active learning collection", 0, 2);
        // hidden
        ConnectWithConsole(cmdopts, write_force, "write-force", "", "Write force info into output file");
        ConnectWithConsole(cmdopts, write_stress, "write-stress", "", "Write stress info into output file");
        ConnectWithConsole(cmdopts, write_ateng, "write-ateng", "", "Write atomic energy info into output file");
        ConnectWithConsole(cmdopts, discard_all, "discard-all", "", "Discard all extra info when writing output file");
        ConnectWithConsole(cmdopts, create_input, "create-input", "", "Create example simple input file", false, 0);

        ConnectWithConsole(cmdopts, potfile, "potfile", "pot", "Potential file");
        ConnectWithConsole(cmdopts, pottype, "pottype", "pt", "Potential type");

        ConnectWithConsole(cmdopts, tags, "tags", "tags", "Add relax: tags, do not need prepend relax: prefix, must in pair with --vals", -1);
        ConnectWithConsole(cmdopts, vals, "vals", "vals", "Add relax: tags value, must in pair with --tags", -1);

        // quick reference
        ConnectWithConsole(cmdopts, gtol, "gtol", "gtol", "Gradient convergence threshold.");
        ConnectWithConsole(cmdopts, selective_dynamics, "selective_dynamics", "sd", "Enables selective dynamics");
        ConnectWithConsole(cmdopts, strain_mask, "strain_mask", "", "6/9-component mask tensor (0=fixed, 1=relaxed) for strain components during cell optimization");
        ConnectWithConsole(cmdopts, stepmax, "stepmax", "", "Maximum allowed step size per degree of freedom in cell/atom updates");
        ConnectWithConsole(cmdopts, pressure, "pressure", "p", "External hydrostatic pressure (GPa). Positive values = compressive loading");
        ConnectWithConsole(cmdopts, dprecond, "dprecond", "", "Diagonal preconditioning intensity for thin/flat systems", 1, 10);
        for (const auto &name : {"gtol", "selective_dynamics", "strain_mask", "stepmax", "pressure", "dprecond"})
            cmdopts.map_setting(name, std::string("relax:") + name);
    }

    void InitBySettings(GPSettings &settings) {
        ApplySettings(settings);
    }
};

#define INPUT_EXAMPLE \
"# This example input file demonstrates the most commonly used options.\n\
input                    POSCAR          # Read initial structure from this file, auto detect or assign format\n\
output                   CONTCAR         # Output relaxed structure in this file, auto detect or assign format\n\
# input-format           POSCAR          # Input file format, will be guessed if not present\n\
# output-format          POSCAR          # Output file format, will be gussed if not present\n\n\
pot:pottype              mtp             # Potential type\n\
pot:potfile              potential.mtp   # Potential file\n\
index                    index.map       # Type mapping file\n\n\
isif                     3               # Same as VASP, support 2 - 8, will overwrite other settings like relax_cell, relax_atom, fix_vol, fix_shape, etc\n\
relax:pressure           0               # External hydrostatic pressure (GPa). Positive values means compressed\n\
relax:stepmax            0.2             # Maximum allowed step size per degree of freedom in each iteration\n\
relax:driver             lbfgspp         # options: lbfgspp, fire, gd, cg, lbfgs\n\
relax:cartesian          False           # Set to True to use Cartesian coordinates during relaxation, only use this in fix cell relaxation\n\
relax:maxiteration    1000            # Max iteration steps\n\
relax:selective_dynamics False           # Selective dynamics, read info in POSCAR or an external definition file\n\
# position-mask          1 0 1           # Global 0=fixed/1=relaxed flags; auto-enable selective dynamics, preserve cell flags\n\
# Default mask components are reference-cell Cartesian; cartesian=True uses current Cartesian; cartesian=False plus atomic_force_recast=False uses fractional\n\
# filter                 seldyn_filter   # Full optimizer mask: n then exactly n flags, each 0 or 1\n\
relax:gtol               1e-3            # Gradient tolerance\n\
relax:index              index.map       # Type index mapping for some format\n\
relax:write_traj         False           # Set to True to enable relax trajectory output\n\
relax:trajfile           reltrj.xyz      # Trajectory file\n\
relax:nblock             5               # Write log/trajectory every this many steps\n\
relax:apply_disp         False           # Set to True to apply an initial displacement\n\
relax:disp               0.05            # Magnitute of displacement\n\
# relax:external_stress  1 2 1 0 0 0     # xx yy zz yz xz xy, in unit GPa\n\
# relax:strain_mask      1 0 0 0 0 0     # xx yy zz yz xz xy, 1 - relax, 0 - fix\n\
# relax:couple           0               # 0 - no, 1 - ab, 2 - bc, 3 - ca, 4 - abc\n\
"

int relaxstrv2(int argc, char *argv[]) {
    auto &cmdopts = CMDOption::global_cmdopts;
    std::string settingfile = "";
    GPSettings settings;
    GPIO iobj;
    Relaxstr2Params params;
    cmdopts.addop(settingfile, "settings", "", "Settings file", true)
        .attach_settings("relax:", "Relaxation driver (v2)",
            [](std::ostream &os) { RelaxationDrvV2().ListSettings(os, "relax:"); })
        .set_helponexit(false);
    cmdopts.phrase_options(argc, argv);
    if (cmdopts.displayhelp) {
        cmdopts.showhelp(std::cout);
        return 0;
    }
    if (params.create_input) {
        ofstream ofs("relax_simple_example.ini");
        if (!ofs) {
            cerr << "Error: cannot open sample input file relax_simple_example.ini" << endl;
            return 1;
        }
        ofs << INPUT_EXAMPLE << endl;
        ofs.close();
        if (!ofs) {
            cerr << "Error: cannot write sample input file relax_simple_example.ini" << endl;
            return 1;
        }
        cout << "Sample input file created: relax_simple_example.ini" << endl;
        return 0;
    }
    
    if (GPIO::file_exist(settingfile)) {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
        params.InitBySettings(settings);
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    cmdopts.apply_overrides(settings);
    // Merge relax tags before mask preparation reads the resulting settings.
    if (params.tags.size() > 0) {
        if (params.tags.size() != params.vals.size()) {
            cerr << "ERROR: tags and vals must in pair" << endl;
            return 1;
        }
        for (int i = 0; i < params.tags.size(); ++i) {
            settings.Modify("relax:" + params.tags[i], params.vals[i]);
        }
    }
    const bool has_position_mask = cmdopts.was_provided("position-mask") || settings.count("position-mask");
    const bool has_filter = cmdopts.was_provided("filter") || settings.count("filter");
    if (params.seq_read) {
        if (!params.file_seldyn.empty() || has_filter)
            throw std::invalid_argument("seq-read is incompatible with external seldyn and filter");
        if (!settings["al:mode"].empty() && settings["al:mode"] != "none")
            throw std::invalid_argument("seq-read does not support active learning collection");
        const auto input_format = params.inputformat.empty() ? GPIO::guess_format(params.inputfile) : GPIO::format2enum(params.inputformat);
        const auto output_format = params.outputformat.empty() ? GPIO::guess_format(params.outputfile) : GPIO::format2enum(params.outputformat);
        if (input_format != GPIO::MLPCFG && input_format != GPIO::XDATCAR6 &&
            input_format != GPIO::EXTXYZ && input_format != GPIO::N2P2DAT)
            throw std::invalid_argument("seq-read input must be mlpcfg, XDATCAR6, extxyz or n2p2");
        if (output_format != GPIO::MLPCFG && output_format != GPIO::EXTXYZ)
            throw std::invalid_argument("seq-read output must be mlpcfg or extxyz");
        params.inputformat = input_format == GPIO::MLPCFG ? "mlpcfg" : input_format == GPIO::XDATCAR6 ? "xdatcar6" :
            input_format == GPIO::N2P2DAT ? "n2p2" : "extxyz";
        params.outputformat = output_format == GPIO::MLPCFG ? "mlpcfg" : "extxyz";
        // Input and result collections must both remain bounded by one frame.
        params.seq_write = true;
    }
    Constraints::ConstraintDeclarations constraint_declarations;
    bool has_lattice_constraints = false;
    if (!params.file_constraints.empty()) {
        if (mpi.size > 1)
            throw std::invalid_argument("file-constraints supports only a single MPI rank");
#if defined(IMRMLP_CUDA) || defined(IMRMLP_HIP) || defined(IMRMLP_GPU)
        throw std::invalid_argument("file-constraints requires a CPU build");
#endif
        // Syntax and schema errors precede structure IO and output creation.
        constraint_declarations = Constraints::read_constraints(params.file_constraints);
        has_lattice_constraints = std::any_of(constraint_declarations.begin(), constraint_declarations.end(),
            [](const auto &item) { return item.version == 1 && item.kind != Constraints::ConstraintKind::Distance &&
                                         item.kind != Constraints::ConstraintKind::AtomicAngle; });
        if (has_lattice_constraints &&
            (has_position_mask || has_filter || !params.gcm.empty() || !params.file_seldyn.empty()))
            throw std::invalid_argument("file-constraints is incompatible with position-mask, filter, cell-mask and seldyn");
    }
    if (has_filter) {
        if (has_position_mask || !params.file_seldyn.empty() || !params.gcm.empty())
            throw std::invalid_argument("filter is incompatible with position-mask, seldyn and cell-mask");
        std::ifstream input(params.file_filter);
        if (params.file_filter.empty() || !input)
            throw std::invalid_argument("Cannot open filter mask file: " + params.file_filter);
    }

    iobj.force_write_all_to_one = true;
    std::vector<Atoms> atoms;
    std::unique_ptr<std::istream> ifseq;
    Potentials::AnyPotential *pot;

    if (!GPIO::file_exist(params.inputfile)) {
        std::cerr << "Error: input structure file " << params.inputfile << " not found" << std::endl;
        return 1;
    }

    if (GPIO::file_exist(params.indexfile))
        iobj.readmap(params.indexfile);
    else 
        iobj.setmap();
    
    if (params.seq_read) {
        std::error_code error;
        const bool same = std::filesystem::equivalent(params.inputfile, params.outputfile, error);
        if (same)
            throw std::invalid_argument("seq-read input and output refer to the same file");
        try {
            ifseq = open_seq_ifile(params.inputfile);
            if (!ifseq || !iobj.seq_read_init(*ifseq, params.inputformat)) {
                std::cerr << "Error: cannot initialize sequential input " << params.inputfile << '\n';
                return 1;
            }
            atoms.resize(1);
            if (!iobj.seq_read(*ifseq, atoms[0])) atoms.clear();
        } catch (const std::exception &e) {
            std::cerr << "Failed reading structure 0: " << e.what() << '\n';
            return 1;
        }
    } else if (!iobj.read(atoms, params.inputfile, params.inputformat)) {
        std::cerr << "Error: failed to read complete input " << params.inputfile << '\n';
        return 1;
    }
    if (atoms.size() == 0) {
        std::cerr << "Error: no structures loaded" << std::endl;
        return 1;
    }
    /**
    ignore # and ! lines
    seldyn mask: 
        line 1: n // int, n configurations

        line 2: i m // int, one-based configuration ID, total atoms in that configuration
        line 3-5: 3x3 matrix
        line 6.. : id x y z, zero-based atom ID; omitted atoms remain movable
        next configuration    
    **/
    const bool initialize_seldyn_defaults = !has_filter && !has_position_mask &&
        params.file_seldyn.empty() && str_to_bool(settings["relax:selective_dynamics"]);
    if (!has_filter && (str_to_bool(settings["relax:selective_dynamics"]) || has_position_mask)) {
        if (!params.file_seldyn.empty()) {
            std::ifstream ifs(params.file_seldyn);
            if (!ifs) throw std::invalid_argument("Cannot open seldyn mask file: " + params.file_seldyn);
            AppsRelax::read_masks(ifs, atoms);
        }
        else if (!has_position_mask) {
            for (int i = 0; i < atoms.size(); ++i) {
                if (atoms[i].sel_dynpos_.size() == 0)  { 
                    std::cerr << "WARNING: NO SELDYNINFO" << std::endl;
                    atoms[i].sel_dynpos_.resize(atoms[i].size(), Eigen::RowVector3d::Ones());
                    // exit(0);
                }
                atoms[i].sel_dyncell_.setOnes();
            }
        }
    }
    // ISIF
    std::vector<std::vector<regulator>> frame_constraints;
    std::vector<regulator> current_frame_constraints;
    if (params.isif > 1 && !settingfile.empty()) {
        Warning("Set ISIF will overwrite relax: settings in config file, and may conflict with constraints if they are provided");
    }
    if (!params.gcm.empty()) {
        settings.Modify("relax:lattice_mask", params.gcm);
        settings.Modify("relax:strain_mask", params.gcm);
    }
    RelaxationDrvV2 relaxdrv(settings.ExtractSubSettings("relax"));
    Constraints::configure_symmetry(constraint_declarations, relaxdrv);
    relaxdrv.set_isif(params.isif);
    // Explicit global masks override driver mask settings after --tags and ISIF.
    if (!params.gcm.empty()) relaxdrv.configure_cell_mask(params.gcm);
    const Eigen::Matrix3d gcmask = relaxdrv.cellmask;
    if (!params.gcm.empty()) {
        for (auto &frame : atoms) {
            frame.sel_dyncell_ = gcmask;
            if (frame.sel_dynpos_.empty())
                frame.sel_dynpos_.assign(frame.size(), Eigen::RowVector3d::Ones());
        }
    }
    if (has_filter) {
        if (!relaxdrv.control.lattice_mask_str.empty())
            throw std::invalid_argument("filter is incompatible with strain_mask/lattice_mask");
        if (relaxdrv.flags.keep_symm || relaxdrv.flags.vibffilter || relaxdrv.flags.constraint_vib_modes ||
            relaxdrv.flags.linear_constraint || relaxdrv.flags.linear_transform)
            throw std::invalid_argument("filter is incompatible with keep_symm and VIB/VIB2/LC/LF modifiers");
        for (const auto &frame : atoms)
            for (const auto &mask : frame.sel_dynpos_)
                if (!(mask.array() == 1).all())
                    throw std::invalid_argument("filter is incompatible with embedded frozen-atom selective dynamics flags");
        // Apply after settings, --tags and ISIF, so this explicit request wins.
        relaxdrv.control.seldyn_file = params.file_filter;
        relaxdrv.flags.selective_dynamics = true;
    }
    relaxdrv.validate_options();
    if (!relaxdrv.control.precond_file.empty() || !relaxdrv.control.hessian_precond_file.empty())
        for (const auto &frame : atoms)
            relaxdrv.check_preconditioner(frame);
    if (params.seq_read && (relaxdrv.flags.selective_dynamics || has_position_mask) &&
        !relaxdrv.control.seldyn_file.empty() && GPIO::file_exist(relaxdrv.control.seldyn_file))
        throw std::invalid_argument("seq-read is incompatible with an active optimization-coordinate mask file (relax:seldyn_file)");
    if (has_position_mask)
        AppsRelax::apply_position_mask(params.gpm, atoms, relaxdrv);
    relaxdrv.control.indexfile = params.indexfile;
    const auto prepare_frame_constraints = [&](const Atoms &frame, size_t i) {
        try {
            if (frame.size() == 0 || frame.chemmap_.size() != frame.pos_.size())
                throw std::invalid_argument("invalid initial geometry or species count");
            for (size_t atom_id = 0; atom_id < frame.pos_.size(); ++atom_id)
                if (!frame.pos_[atom_id].allFinite() || frame.chemmap_[atom_id] < 0)
                    throw std::invalid_argument("invalid initial coordinates or species index");
            if (!frame.sel_dynpos_.empty() && frame.sel_dynpos_.size() != frame.pos_.size())
                throw std::invalid_argument("embedded selective-dynamics flags have the wrong size");
            for (const auto &mask : frame.sel_dynpos_)
                if (has_lattice_constraints && !(mask.array() == 1).all())
                    throw std::invalid_argument("file-constraints is incompatible with embedded frozen-atom flags");
            // Bind omitted targets and initial atomic images before displacement.
            auto requested = Constraints::bind_constraints(constraint_declarations, frame);
            return relaxdrv.prepare_constraints(frame, requested);
        } catch (const std::exception &error) {
            throw std::invalid_argument("file-constraints frame " + std::to_string(i) + ": " + error.what());
        }
    };
    if (!constraint_declarations.empty()) {
        if (!relaxdrv.constrained_solver() && std::any_of(constraint_declarations.begin(),constraint_declarations.end(),
            [](const auto &d){return d.kind!=Constraints::ConstraintKind::Symmetry;}))
            relaxdrv.flags.with_constraint_aul = true;
        // Preflight the batch, or the first sequential frame, before output access.
        if (params.seq_read) {
            current_frame_constraints = prepare_frame_constraints(atoms[0], 0);
        } else {
            frame_constraints.reserve(atoms.size());
            for (size_t i = 0; i < atoms.size(); ++i)
                frame_constraints.push_back(prepare_frame_constraints(atoms[i], i));
        }
    }
    
    // Potentials::potential_wrapper wrapper(params.potfile, params.pottype, settings.ExtractSubSettings("pot"));
    Potentials::potential_wrapper wrapper(params.potfile, params.pottype, settings.ExtractSubSettings("pot"));
    pot = wrapper.pot;
    const auto validate_constraint_species = [&](const Atoms &frame, size_t i) {
        for (size_t j = 0; j < frame.chemmap_.size(); ++j) {
            const int species = frame.chemmap_[j];
            if (species < 0 || species >= pot->species_count)
                throw std::invalid_argument("file-constraints frame " + std::to_string(i) +
                    " atom " + std::to_string(j) + ": species index " + std::to_string(species) +
                    " is outside potential range [0, " + std::to_string(pot->species_count) + ")");
        }
    };
    if (!constraint_declarations.empty()) {
        // Species bounds depend on the loaded potential. Check all loaded frames
        // before evaluation, AL setup, or opening sequential output.
        if (pot->species_count <= 0)
            throw std::invalid_argument("file-constraints requires a potential with a positive species_count");
        for (size_t i = 0; i < atoms.size(); ++i)
            validate_constraint_species(atoms[i], i);
    }
    pot->omp_init();
    // std::vector<regulator> empty_constraints;
    // ── Active learning setup (optional) ─────────────────────────────
    auto al = AL::ALPotential::create(settings.ExtractSubSettings("al"));
    ALModifier al_mod(al.get());
    if (al && al->active()) {
        al->init_builder(*pot);
        relaxdrv.al_modifier_ = &al_mod;
    }

    // Reading and writing have independent counters, headers and chemistry state.
    GPIO writer;
    if (GPIO::file_exist(params.indexfile)) writer.readmap(params.indexfile);
    else writer.setmap();
    std::unique_ptr<std::ostream> ofseq;
    if (params.seq_write) {
        ofseq = open_ofile(params.outputfile);
        if (!ofseq) {
            std::cerr << "Error: cannot open output file " << params.outputfile << std::endl;
            return 1;
        }
        if (!writer.seq_write_init(*ofseq, params.outputformat, params.outputfile)) return 1;
    }
    int exit_status = 0;
    bool fatal = false;
    size_t output_count = 0;
    auto relax_one = [&](Atoms &at, size_t i, std::vector<regulator> *requested) {
        Message(fmt::format("Conf {}", i));
        try {
            relaxdrv.init(pot, at, requested);
            relaxdrv.run();
            for (size_t j = 0; j < relaxdrv.constraints.size(); ++j) {
                const auto &constraint = relaxdrv.constraints[j];
                Message(fmt::format("Constraint {}: residual={:.12g}, tolerance={:.12g}",
                                    j, constraint.f, constraint.tol));
                if (constraint.regulator_t == regulator::DISTANCE || constraint.regulator_t == regulator::ANGLE) {
                    const auto value = Constraints::evaluate_atomic_geometry(constraint, relaxdrv.atom);
                    Message(fmt::format("Constraint {}: value={:.12g}, target={:.12g}, deviation={:.12g} {}",
                        j, value.physical_value, constraint.double_target,
                        value.physical_value - constraint.double_target,
                        constraint.regulator_t == regulator::DISTANCE ? "Angstrom" : "degrees"));
                }
            }
            if (!relaxdrv.state_vars.converged) {
                if (exit_status == 0) exit_status = 2;
                Message(fmt::format("Structure {} did not converge: {}", i, relaxdrv.stop_reason));
            }
        } catch (const std::bad_alloc &e) {
            std::cerr << "Failed structure " << i << ": " << e.what() << '\n';
            exit_status = 1;
            fatal = true;
            return;
        } catch (const std::exception &e) {
            std::cerr << "Failed structure " << i << ": " << e.what() << '\n';
            exit_status = 1;
            return;
        } catch (const alglib::ap_error &e) {
            std::cerr << "Failed structure " << i << ": " << e.msg << '\n';
            exit_status = 1;
            return;
        }
        if (!params.write_force) relaxdrv.atom.forces_.clear();
        if (!params.write_stress) relaxdrv.atom.have_stress = false;
        if (!params.write_ateng) relaxdrv.atom.site_energies_.clear();
        if (params.discard_all) {
            relaxdrv.atom.local_attribs_d_.clear();
            relaxdrv.atom.local_attribs_v_.clear();
            relaxdrv.atom.global_attribs_d_.clear();
            relaxdrv.atom.global_attribs_v_.clear();
        }
        if (params.seq_write) {
            try {
                if (!writer.seq_write(*ofseq, relaxdrv.atom))
                    throw std::runtime_error("failed to write output frame");
                if (params.seq_read) ofseq->flush();
                if (!*ofseq) throw std::runtime_error("failed to flush output frame");
            } catch (const std::exception &e) {
                std::cerr << "Failed writing structure " << i << ": " << e.what() << '\n';
                exit_status = 1;
                fatal = true;
            }
        } else {
            atoms[output_count++] = relaxdrv.atom;
        }
    };
    if (params.seq_read) {
        for (size_t i = 0; ; ++i) {
            relax_one(atoms[0], i, current_frame_constraints.empty() ? nullptr : &current_frame_constraints);
            if (fatal) break;
            try {
                // seq_read clears its destination before loading the next frame.
                if (!iobj.seq_read(*ifseq, atoms[0])) {
                    if (ifseq->bad()) throw std::runtime_error("input stream read failed");
                    break;
                }
                if (initialize_seldyn_defaults) {
                    atoms[0].sel_dyncell_.setOnes();
                    if (atoms[0].sel_dynpos_.empty())
                        atoms[0].sel_dynpos_.assign(atoms[0].size(), Eigen::RowVector3d::Ones());
                }
                if (!params.gcm.empty()) atoms[0].sel_dyncell_ = gcmask;
                if (has_position_mask) AppsRelax::apply_position_mask(params.gpm, atoms, relaxdrv);
                if (!constraint_declarations.empty()) {
                    current_frame_constraints = prepare_frame_constraints(atoms[0], i + 1);
                    validate_constraint_species(atoms[0], i + 1);
                }
            } catch (const std::exception &e) {
                std::cerr << "Failed reading/preparing structure " << i + 1 << ": " << e.what() << '\n';
                exit_status = 1;
                break;
            }
        }
    } else {
        for (size_t i = 0; i < atoms.size() && !fatal; ++i)
            relax_one(atoms[i], i, frame_constraints.empty() ? nullptr : &frame_constraints[i]);
    }
    if (!params.seq_write) {
        atoms.resize(output_count);
        if (!iobj.write(atoms, params.outputfile, params.outputformat)) return 1;
    } else if (!close_ofile(ofseq)) {
        std::cerr << "Error: failed to close output " << params.outputfile << '\n';
        return 1;
    }

    // Active learning — sparsify pool and export (pass existing iobj)
    if (al) al->finalize(&iobj);

    return exit_status;
}
