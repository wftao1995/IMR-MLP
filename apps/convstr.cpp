// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

/******************************************
 * Written by wftao (wftao1995@gmail.com)
 * 
 * Supported input format: 
 *  mlpcfg : vector, A.Shapeev's mlip format
 *  lammpstrj : vector, lammps trajectory
 *  lammpsdat : single
 *  vasprun.xml : vector
 *  POSCAR : single
 *  XDATCAR : vector
 * 
 * Supported output format:
 *  lammpsdat : single, lammps data file
 *  mlpcfg : vector
 *  POSCAR : single
 *  XDATCAR : vector
 * 
 * 
 * **************************************/
#include "io/gpio.h"
#include "apps/convstr_scalar_expression.h"
#include "io/format_detection.h"
#include "io/atoms_units.h"
#include "utils/compressed_stream.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#define USAGESTR "Example: \n\
convstr --input-format lammpstrj --output-format POSCAR --index-file index.map input.lammpstrj output.vasp \n\
convstr -if poscar -of cif -id index.map input.vasp output.cif \n\
Supported formats: \
\tXDATCAR, lammps trajectory text file (lammpstrj), vasprun.xml, mlip's configuration file (mlpcfg), \n\
\txsf, cif (limited support), vasp structure file (POSCAR), \n\
\tbinary file (bin), extended xyz (extxyz), lammps data of structure (lmpdat, limited support) \n\
\tDL_POLY4 structure file (dlpoly), gromacs data file (gro, limited support), n2p2 data file (n2p2)\n\
\tVASP OUTCAR (OUTCAR), VASP MLFF (VASPABN), vtk (limited support), cp2kxyz\n\
All indices supplied in command-line options are start from 0 unless use -b false option\n\
A example index.map file:\n\
[begin]\nAl\nNi\nRe\n[end]\n"


static std::vector<std::string> split_csv(const std::string &s)
{
    std::vector<std::string> result;
    size_t pos = 0, found;
    while ((found = s.find(',', pos)) != std::string::npos) {
        if (found > pos) result.push_back(s.substr(pos, found - pos));
        pos = found + 1;
    }
    if (pos < s.size()) result.push_back(s.substr(pos));
    return result;
}

static std::vector<bool> parse_sort_order(const std::string &order_str, size_t nkeys)
{
    if (order_str.empty()) return std::vector<bool>(nkeys, true);
    auto orders = split_csv(order_str);
    std::vector<bool> ascending;
    for (const auto &o : orders) {
        if (o == "asc")
            ascending.push_back(true);
        else if (o == "desc")
            ascending.push_back(false);
        else {
            std::cerr << "Error: invalid --sort-order \"" << o
                      << "\", expected asc or desc" << std::endl;
            exit(128);
        }
    }
    if (ascending.size() == 1 && nkeys > 1)
        ascending.assign(nkeys, ascending[0]);
    else if (ascending.size() != nkeys) {
        std::cerr << "Error: number of --sort-order values (" << ascending.size()
                  << ") does not match number of --sort-by keys (" << nkeys << ")"
                  << std::endl;
        exit(128);
    }
    return ascending;
}

static std::function<double(const Atoms &)> resolve_sort_key(const std::string &key)
{
    // Built-in keys
    if (key == "energy") {
        return [](const Atoms &a) -> double {
            if (!a.has_energy()) {
                std::cerr << "Error: sort key \"energy\" requested but structure "
                          << a.id << " has no energy data" << std::endl;
                exit(128);
            }
            return a.energy_;
        };
    }
    if (key == "fenergy") {
        return [](const Atoms &a) -> double {
            if (!a.has_fenergy()) {
                std::cerr << "Error: sort key \"fenergy\" requested but structure "
                          << a.id << " has no free energy data" << std::endl;
                exit(128);
            }
            return a.fenergy_;
        };
    }
    if (key == "fitness")  return [](const Atoms &a) -> double { return a.fitness; };
    if (key == "volume")   return [](const Atoms &a) -> double { return a.get_volume(); };
    if (key == "size")     return [](const Atoms &a) -> double { return static_cast<double>(a.size()); };
    if (key == "id")       return [](const Atoms &a) -> double { return static_cast<double>(a.id); };

    // Global attributes: double first, then int
    return [key](const Atoms &a) -> double {
        {
            auto it = a.global_attribs_d_.find(key);
            if (it != a.global_attribs_d_.end()) return it->second;
        }
        {
            auto it = a.global_attribs_i_.find(key);
            if (it != a.global_attribs_i_.end()) return static_cast<double>(it->second);
        }
        std::cerr << "Error: sort key \"" << key << "\" not found in structure " << a.id
                  << ".\n  Built-in keys: energy, fenergy, fitness, volume, size, id"
                  << "\n  Available double attributes:";
        for (const auto &kv : a.global_attribs_d_)
            std::cerr << " " << kv.first;
        std::cerr << "\n  Available integer attributes:";
        for (const auto &kv : a.global_attribs_i_)
            std::cerr << " " << kv.first;
        std::cerr << std::endl;
        exit(128);
        return 0.0;
    };
}


int convstr(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::vector<std::string> inputfiles;
    std::string outputfile = "";
    std::vector<std::string> addlabel;
    std::vector<std::string> labelvalue;
    std::vector<std::string> label_expressions;
    std::string indexfile = "index.map";
    std::ifstream ifs;
    std::ofstream ofs;
    std::string informat;
    bool detect_format = false;
    std::string input_unit_system;
    std::optional<GPUnits::SourceUnits> source_units;
    std::string outformat;
    // std::string chgcarfl = "CHGCAR";
    GPIO iobj;
    Atoms atom;
    std::vector<Atoms> atoms;
    bool vecinput = true;
    bool vecoutput = true;
    // int select = -1; // from zero, phrase input from 1
    std::vector<int> select;
    int outrate = 0;
    int startfrom = 0; // from zero, phrase input from 1
    int end = INT_MAX; // from zero, phrase input from 1, include end
    int outstartfrom = 0;
    int outend = INT_MAX;
    int loadrate = 0;  // when loadrate is one, use keepone mode
    std::string outsuffix = "";
    bool overwrite = false;
    bool appendto = false;
    bool has_mag = false;
    bool wrap = false;
    bool non_colinear_mag = false;
    // bool compute_mag_from_chg = false;
    bool binaryin = false;
    bool binaryout = false;
    // bool outidreset = false;
    bool lastonly = false;
    bool reverse = false;
    bool bstart = true; // when prasing cmd options, id start from 0 instead of 1
    bool random_shuffle = false;
    std::string sort_by_str;
    std::string sort_order_str;
    bool apply_disp = false;
    int rand_type = 0; // 0: gaussian, 1: uniform
    double disp_mag = 0.05;
    iobj.suffix_id_use_internal = true;
    iobj.reset_status_every_write = false;
    iobj.reset_status_every_read = false;

    cmdopts.addop(informat, "input-format", "if", "Input file format")
           .addop(detect_format, "detect-format", "", "Detect input content when input-format is omitted", false, 0)
           .addop(input_unit_system, "input-unit-system", "", "Explicit EXTXYZ source units: canonical, metal, atomic, or real")
           .addop(outformat, "output-format", "of", "Output file format")
           .addop(indexfile, "index", "id", "Index file mapping atom types to indices")
           .addop(wrap, "wrap", "w", "Wrap atomic coordinates to periodic boundaries")
           .addop(iobj.suffix_id_use_internal, "resetid", "rid", "Reset atom type IDs to sequential integers (0, 1, 2, ...)", false, 0)
           .addop(outsuffix, "outsuffix", "suff", "Suffix appended to output filenames")
           .addop(select, "select", "s", "Select specific image indices (multiple values allowed)", false, -1)
           .addop(loadrate, "loadrate", "lr", "Read images to memory every this many structures (1 = sequential reading)")
           .addop(outrate, "outrate", "or", "Write images to file every this many structures (1 = sequential writing)")
           .addop(startfrom, "start", "st", "Starting image index for reading")
           .addop(outstartfrom, "ostart", "ost", "Starting image index for writing from memory")
           .addop(bstart, "bstart", "b", "When parsing options, id start from zero")
           .addop(end, "end", "ed", "Ending image index for reading")
           .addop(outend, "oend", "oed", "Ending image index for writing from memory")
           .addop(overwrite, "overwrite", "ov", "Overwrite existing file", false, 0)
           .addop(appendto, "append", "a", "Append to existing file", false, 0)
           .addop(reverse, "reverse", "r", "Reverse the order of images", false, 0)
           .addop(random_shuffle, "shuffle", "", "Randomly shuffle image order", false, 0)
           .addop(sort_by_str, "sort-by", "", "Comma-separated sort keys: energy, fenergy, fitness, volume, size, id, or global attributes")
           .addop(sort_order_str, "sort-order", "", "Sort order per key: asc (default) or desc, single value or comma-separated")
           .addop(iobj.r_extent, "rextent", "When reading non-PBC structures, extending cell size to this value")
           .addop(iobj.drop, "drop", "d", "When reading OUTCAR/vasprun.xml, try to discard calculations if iteration == NSW, may not work sometimes")
           .addop(iobj.write_symbol, "write-symbol", "", "Write atomic symbols instead of numeric IDs (format-dependent)", false, 0, 5)
           .addop(iobj.discard_props, "discard", "", "Discard specific properties during file reading", false, -1)
           .addop(iobj.packatoms, "pack-atoms", "", "Consolidate atom types in POSCAR output (default: true)")
           .addop(has_mag, "magnetic", "mag", "Read magnetic moments from input files", false, 0, 3)
           .addop(non_colinear_mag, "magnetic-non-colinear", "mncl", "Assume non-linear magnetic", false, 0, 3)
           .addop(addlabel, "label", "lb", "Add custom labels to images (multiple values allowed)", false, -1)
           .addop(labelvalue, "value", "lv", "Label values corresponding to --label option", false, -1)
           .addop(label_expressions, "label-expr", "", "Scalar metadata assignments, e.g. energy_per_atom=energy/size", false, -1)
           .addop(lastonly, "last", "last", "Output only the final image from trajectory", false, 0)
           .addop(iobj.force_tilt, "force-tilt", "", "Force tilt cell format in LAMMPS data even for orthogonal cells", false, 0)
           .addop(iobj.force_write_all_to_one, "force-one", "", "Write all structures to a single file (e.g., POSCAR format)", false, 0)
           .addop(iobj.fname_separator, "separator", "sep", "Separator between filename and ID in output")
           .addop(apply_disp, "apply-disp", "ad", "Apply displacement to images", false, 0)
           .addop(rand_type, "rand-type", "rt", "Random displacement type (0: gaussian, 1: uniform)", false)
           .addop(disp_mag, "disp-mag", "dm", "Displacement magnitude", false)
           .addop(inputfiles, "inputfiles", "in", "Input structure files (multiple files supported)", true, -1)
           .addop(outputfile, "outputfile", "out", "Output file", true)
           .add_request("inputfiles")
           .add_request("outputfile")
           .check()
           .set_usage(USAGESTR);
    cmdopts.modify_level("absolute_index_read", 0);
    cmdopts.modify_level("absolute_index_write", 0);
    cmdopts.modify_level("lossless", 0);
    cmdopts.modify_level("output_in_direct", 0);
    cmdopts.modify_level("output_index_offset", 0);
    cmdopts.phrase_options(argc, argv);
    if (!bstart) {
        for (int i = 0; i < select.size(); ++i) {
            select[i]--;
        }
        startfrom--;
        end--;
        outend--;
        outstartfrom--;
    }
    if (!input_unit_system.empty()) {
        try { source_units = GPUnits::named_source_units(input_unit_system); }
        catch (const std::exception &error) {
            std::cerr << "Error: " << error.what() << '\n';
            return 1;
        }
    }
    for (auto it = inputfiles.begin(); it != inputfiles.end();) {
        if (!iobj.file_exist(*it)) {
            if (detect_format && informat.empty()) {
                std::cerr << "Error: cannot open input file " << *it << '\n';
                return 1;
            }
            const auto input_format = informat.empty() ? GPIO::guess_format(inputfiles.front())
                                                       : GPIO::format2enum(informat);
            if (input_format == GPIO::MLIPBIN || input_format == GPIO::XSF ||
                input_format == GPIO::GRODATA || input_format == GPIO::CP2KXYZ ||
                input_format == GPIO::N2P2DAT || input_format == GPIO::OUTCAR) {
                std::cerr << "Error: cannot open "
                          << (input_format == GPIO::MLIPBIN ? "MLIPBIN " : "")
                          << "input file " << *it << std::endl;
                return 1;
            }
            std::cerr << "Warning: File " << *it << " not exist, skipping this file" << std::endl;
            it = inputfiles.erase(it);
        } else
            ++it;
    }
    if (inputfiles.size() == 0) {
        std::cerr << "ERROR: No files to be processed" << std::endl;
        exit(128);
    }
    // Reject unsupported compressed input before opening any output.
    for (const auto &file : inputfiles) {
        if (compression_of_extension(file) == CompressionKind::Unsupported ||
            probe_compression(file) == CompressionKind::Unsupported) {
            std::cerr << "ERROR: unsupported compression for " << file << std::endl;
            return 1;
        }
    }
    if (compression_of_extension(outputfile) == CompressionKind::Unsupported) {
        std::cerr << "ERROR: unsupported compression for " << outputfile << std::endl;
        return 1;
    }
    // if format not present, guess it
    GPIO::supported_format iinformat, ioutformat;
    if (informat.empty() && detect_format) {
        try {
            for (const auto &filename : inputfiles) {
                const auto detection = GPIOFormat::detect_file(filename);
                if (detection.format.empty()) {
                    std::cerr << "Error: " << (detection.candidates.empty() ? "cannot detect" : "ambiguous")
                              << " input content in " << filename << "; specify --input-format\n";
                    return 1;
                }
                if (informat.empty()) informat = detection.format;
                else if (GPIO::format2enum(informat) != GPIO::format2enum(detection.format)) {
                    std::cerr << "Error: detected input formats differ; convert each format separately\n";
                    return 1;
                }
            }
        } catch (const std::exception &error) {
            std::cerr << "Error: " << error.what() << '\n';
            return 1;
        }
    }
    if (informat.empty()) {
        iinformat = GPIO::guess_format(inputfiles[0]);
    } else {
        iinformat = GPIO::format2enum(informat);
    }
    if (outformat.empty()) {
        ioutformat = GPIO::guess_format(outputfile);
    } else {
        ioutformat = GPIO::format2enum(outformat);
    }
    if (source_units && iinformat != GPIO::EXTXYZ) {
        std::cerr << "Error: --input-unit-system is supported only for EXTXYZ input\n";
        return 1;
    }
    if (source_units && informat.empty()) {
        for (const auto &filename : inputfiles)
            if (GPIO::guess_format(filename) != GPIO::EXTXYZ) {
                std::cerr << "Error: --input-unit-system requires consistent EXTXYZ inputs; specify --input-format\n";
                return 1;
            }
        informat = "extxyz";
    }
    const bool checked_text_input = iinformat == GPIO::VASPRUN ||
                                    iinformat == GPIO::OUTCAR ||
                                    iinformat == GPIO::XDATCAR ||
                                    iinformat == GPIO::XDATCAR6 ||
                                    iinformat == GPIO::XSF ||
                                    iinformat == GPIO::GRODATA ||
                                    iinformat == GPIO::CP2KXYZ ||
                                    iinformat == GPIO::N2P2DAT;
    const bool bounded_seq_input = iinformat == GPIO::XSF ||
                                   iinformat == GPIO::OUTCAR || iinformat == GPIO::MLIPBIN ||
                                   iinformat == GPIO::N2P2DAT || iinformat == GPIO::CP2KXYZ || iinformat == GPIO::GRODATA;
    const std::string seq_informat = iinformat == GPIO::XSF ? "xsf" :
                                    iinformat == GPIO::OUTCAR ? "outcar" :
                                    iinformat == GPIO::MLIPBIN ? "bin" :
                                    iinformat == GPIO::N2P2DAT ? "n2p2" :
                                    iinformat == GPIO::CP2KXYZ ? "cp2kxyz" :
                                    iinformat == GPIO::GRODATA ? "gro" :
                                    iinformat == GPIO::XDATCAR ? "xdatcar" :
                                    iinformat == GPIO::XDATCAR6 ? "xdatcar6" : informat;
    // reset default values against input value.
    if (iinformat == GPIO::POSCAR || iinformat == GPIO::CIF ||
        iinformat == GPIO::LMPDAT ||
        iinformat == GPIO::XSF)
        vecinput = false;
    if (iinformat == GPIO::XSF) vecinput = true; // Single or ANIMSTEPS input.
    if (ioutformat == GPIO::POSCAR || ioutformat == GPIO::CIF ||
        ioutformat == GPIO::LMPDAT ||
        ioutformat == GPIO::XSF || ioutformat == GPIO::VTK)
        vecoutput = false;
    const bool animation_output = ioutformat == GPIO::XSF &&
        (outformat == "axsf" || strip_compression_suffix(outputfile).ends_with(".axsf"));
    if (animation_output) vecoutput = true; // Known-vector animation output.
    if (outsuffix == "") {
        if (outformat == "vtk")
            outsuffix = ".vtk";
        // else if (outformat == "cif")
        //     outsuffix = ".cif";
        // else if (outformat == "xsf")
        //     outsuffix = ".xsf";
    } else {
        outsuffix = "." + outsuffix;
    }
    if (iinformat == GPIO::MLIPBIN)
        binaryin = true;
    if (ioutformat == GPIO::MLIPBIN)
        binaryout = true;
    if (!GPIO::file_exist(indexfile))
    {
        std::cerr << "Warning: no index file found" << std::endl;
        iobj.setmap();
    }
    else if (!iobj.readmap(indexfile))
    {
        std::cerr << "Warning: failed to read index file, use atom numbers instead" << std::endl;
        iobj.setmap();
    }
    if (overwrite && appendto)
    {
        std::cerr << "Conflict settings: append and overwrite" << std::endl;
        exit(128);
    }
    if (iobj.file_exist(outputfile) && (!overwrite && !appendto))
    {
        std::cerr << "Error: outputfile " << outputfile << " exist, please use --overwrite or --append" << std::endl;
        std::exit(128);
    }
    if (!vecoutput && appendto)
    {
        std::cerr << "Error: can't apply append mode to format " << outformat << " use --overwrite instead" << std::endl;
        std::exit(128);
    }
    if (labelvalue.size() != addlabel.size()) {
        std::cerr << "Error: incorrect feature label and value settings" << std::endl;
        std::exit(128);
    }
    const auto expressions = ConvstrLabels::compile(label_expressions, addlabel);
    iobj.append = appendto;
    if (startfrom < 0)
        startfrom = 0;
    if (outstartfrom < 0)
        outstartfrom = 0;
    if (end < startfrom)
        end = startfrom + 1;
    if (outend < outstartfrom) 
        outend = outstartfrom + 1;
    // if (outrate < 1)
    //     outrate = 1;
    if (loadrate > 0 && iinformat == GPIO::VASPRUN)
    {
        std::cerr << "Error: vasprun.xml does not support sequential reading, fall back into full reading" << std::endl;
        loadrate = 0;
    }
    if (has_mag)
        iobj.read_mag = has_mag;
    iobj.colinear_mag = !non_colinear_mag;
    iobj.read_startfrom = startfrom;
    iobj.read_endto = end;
    // iobj.outrate = outrate;
    if (outrate < 1) {
        iobj.outrate = 1;
    } else {
        iobj.outrate = outrate;
    }
    if (loadrate > 0)
        iobj.read_irate = loadrate;
    iobj.out_startfrom = outstartfrom;
    iobj.out_endto = outend;
    int smax = 0;
    if (select.size() > 0) {
        std::ranges::sort(select);
        select.erase(std::unique(select.begin(), select.end()), select.end());
        for (int i = 0; i < select.size(); ++i) {
            if (select[i] < 0) {
                std::cerr << "ERROR: select should not be negative, if you want to select configurations from backward, use -r" << std::endl;
                exit(128);
            }
            smax = smax > select[i] ? smax : select[i];
        }
        if (reverse) smax = INT_MAX;
        iobj.read_endto = smax;
    }

    // Build sort criteria: extractor functions + ascending flags
    std::vector<std::function<double(const Atoms &)>> sort_extractors;
    std::vector<bool> sort_ascending;
    if (!sort_by_str.empty()) {
        auto sort_keys = split_csv(sort_by_str);
        sort_ascending = parse_sort_order(sort_order_str, sort_keys.size());
        sort_extractors.reserve(sort_keys.size());
        for (const auto &k : sort_keys)
            sort_extractors.push_back(resolve_sort_key(k));
    }

    auto sort_compare = [&](const Atoms &a, const Atoms &b) -> bool {
        for (size_t i = 0; i < sort_extractors.size(); ++i) {
            double va = sort_extractors[i](a);
            double vb = sort_extractors[i](b);
            // Treat NaN as less than all finite values
            bool na = std::isnan(va), nb = std::isnan(vb);
            if (na && !nb) return bool(sort_ascending[i]);
            if (!na && nb) return bool(!sort_ascending[i]);
            if (na && nb) continue;
            if (va < vb) return bool(sort_ascending[i]);
            if (va > vb) return bool(!sort_ascending[i]);
        }
        return false;
    };

    auto sort_traj = [&](std::vector<Atoms> &traj) {
        if (!sort_extractors.empty())
            std::stable_sort(traj.begin(), traj.end(), sort_compare);
    };

    auto displace_frame = [&](Atoms &frame) {
        if (!apply_disp) return;
        // Keep distributions local to each frame, including the normal cache.
        std::uniform_real_distribution<double> disp_dist(-disp_mag, disp_mag);
        std::normal_distribution<double> norm_dist(0.0, disp_mag);
        if (rand_type == 0) {
            for (int i = 0; i < frame.size(); ++i) {
                for (int a = 0; a < 3; ++a)
                    frame.pos_[i][a] += norm_dist(RNG.mtd);
            }
        } else {
            for (int i = 0; i < frame.size(); ++i) {
                for (int a = 0; a < 3; ++a)
                    frame.pos_[i][a] += disp_dist(RNG.mtd);
            }
        }
    };

    // Every path prepares physical data once, before geometry transforms or sorting.
    auto prepare_frame = [&](Atoms &frame) {
        if (source_units) GPUnits::canonicalize_atoms(frame, *source_units);
        ConvstrLabels::assign(frame, expressions, ioutformat == GPIO::MLPCFG);
        for (std::size_t i = 0; i < addlabel.size(); ++i)
            frame.features_[addlabel[i]] = labelvalue[i];
        if (wrap) frame.wrap();
    };

    auto transform_traj = [&](std::vector<Atoms> &traj) {
        if (traj.size() == 0) {
            std::cerr << "Error: no structures found" << std::endl;
            return;
        }
        for (auto &frame : traj) prepare_frame(frame);
        if (reverse) 
            std::ranges::reverse(traj);
        if (random_shuffle) {
            std::ranges::shuffle(traj, RNG.mtd);
        }
        for (auto &frame : traj) displace_frame(frame);
    };

    // reading
    // 1. seq i/o
    // 2. seq i only
    bool seqin = false, seqout = false;
    if (loadrate == 1) seqin = true;
    if (outrate == 1) seqout = true;
    if (seqout && animation_output) {
        std::cerr << "Error: AXSF output requires known-count vector mode (--outrate 0)\n";
        return 1;
    }
    if (seqin && seqout) {
        if (reverse) {
            std::cerr << "Error: --reverse cannot be used with sequential I/O" << std::endl;
            exit(128);
        }
        if (random_shuffle) {
            std::cerr << "Error: --shuffle cannot be used with sequential I/O" << std::endl;
            exit(128);
        }
        if (lastonly) {
            std::cerr << "Error: --last cannot be used with sequential I/O" << std::endl;
            exit(128);
        }
        if (!sort_by_str.empty()) {
            std::cerr << "Error: --sort-by cannot be used with sequential I/O" << std::endl;
            exit(128);
        }
        auto ofs = open_ofile(outputfile, iobj.append);
        if (!ofs) {
            std::cerr << "Error: cannot open output file " << outputfile << std::endl;
            return 1;
        }
        if (!iobj.seq_write_init(*ofs, outformat, outputfile)) return 1;
        int global_idx = 0;
        for (int ifi = 0; ifi < inputfiles.size(); ++ifi) {
            if (bounded_seq_input &&
                (global_idx > end || (!select.empty() && global_idx > smax))) break;
            auto ifs = bounded_seq_input ? open_seq_ifile(inputfiles[ifi])
                                         : open_ifile(inputfiles[ifi]);
            if (!ifs) {
                std::cerr << "Error: cannot open input file " << inputfiles[ifi] << std::endl;
                if (iinformat == GPIO::MLIPBIN || checked_text_input) return 1;
                continue;
            }
            if (!iobj.seq_read_init(*ifs, seq_informat)) {
                std::cerr << "Error: cannot initialize reading file " << inputfiles[ifi] << std::endl;
                if (iinformat == GPIO::MLIPBIN || checked_text_input) return 1;
                continue;
            }
            for (; iinformat == GPIO::XSF || !ifs->eof(); ++global_idx) {
                // Stop before parsing a frame outside the requested range.
                if (bounded_seq_input &&
                    (global_idx > end || (!select.empty() && global_idx > smax))) break;
                Atoms atom;
                if (!iobj.seq_read(*ifs, atom)) break;
                if (global_idx < startfrom) continue;
                if (global_idx > end) break;  
                if (!select.empty()) {
                    if (std::find(select.begin(), select.end(), global_idx) == select.end())
                        continue;
                }
                atom.id = global_idx;  
                prepare_frame(atom);
                displace_frame(atom);
                if (!iobj.seq_write(*ofs, atom)) return 1;
            }
        }
        if (!close_ofile(ofs)) return 1;
    } else if (seqin) {
        int istr = 0;
        for (int ifi = 0; ifi < inputfiles.size(); ++ifi) {
            if (bounded_seq_input &&
                (istr > end || (!select.empty() && istr > smax))) break;
            auto ifs = bounded_seq_input ? open_seq_ifile(inputfiles[ifi])
                                         : open_ifile(inputfiles[ifi]);
            if (!ifs) {
                std::cerr << "Error: cannot open input file " << inputfiles[ifi] << std::endl;
                if (iinformat == GPIO::MLIPBIN || checked_text_input) return 1;
                continue;
            }
            if (vecinput) {
                if (iobj.seq_read_init(*ifs, seq_informat)) {
                    for (; iinformat == GPIO::XSF || !ifs->eof(); ++istr)
                    {
                        if (bounded_seq_input && istr > end) break;
                        if (select.size() > 0 && istr > smax) break;
                        if (!iobj.seq_read(*ifs, atom)) break;
                        if (istr >= startfrom && istr <= end && (istr - startfrom) % loadrate == 0) {
                            atom.id = istr;
                            atoms.emplace_back(atom);
                        }
                    }
                } else if (iinformat == GPIO::MLIPBIN || checked_text_input) {
                    std::cerr << "Error: cannot initialize input " << inputfiles[ifi] << std::endl;
                    return 1;
                }
            } else {
                if (!iobj.read_one(atom, inputfiles[ifi], informat)) {
                    std::cerr << "ERROR WHEN READING FILE " << inputfiles[ifi] << std::endl;
                    if (checked_text_input) return 1;
                    continue;
                }
                atom.id = ifi;
                atoms.emplace_back(atom);
                if (select.size() > 0 && ifi > smax) break;
            }
        }
        if (atoms.empty()) {
            std::cerr << "Error: no structures were read" << std::endl;
            return 1;
        }
        transform_traj(atoms);
        if (lastonly) {
            std::vector<Atoms> atnew;
            atnew.push_back(atoms.back());
            sort_traj(atnew);
            if (!iobj.write(atnew, outputfile, outformat)) return 1;
        } else {
            if (select.size() > 0) {
                std::vector<Atoms> atnew;
                for (int i = 0; i < select.size(); ++i) {
                    if (select[i] >= 0 && select[i] < (int)atoms.size())
                        atnew.push_back(atoms[select[i]]);
                }
                if (!atnew.empty()) {
                    sort_traj(atnew);
                    if (!iobj.write(atnew, outputfile, outformat)) return 1;
                }
            } else {
                sort_traj(atoms);
                if (!iobj.write(atoms, outputfile, outformat)) return 1;
            }
        }
    } else if (seqout) {
        // This is not meaningful at all
        for (int i = 0; i < inputfiles.size(); ++i) {
            bool state = iobj.read(atoms, inputfiles[i], informat);
            if (!state) {
                std::cerr << "ERROR WHEN READING FILE " << inputfiles[i] << std::endl;
                if (iinformat == GPIO::XDATCAR6 ||
                    iinformat == GPIO::MLIPBIN || checked_text_input) return 1;
            }
            if (iinformat == GPIO::MLIPBIN && iobj.binary_read_limit_reached()) break;
        }
        if (atoms.empty()) {
            std::cerr << "Error: no structures were read" << std::endl;
            return 1;
        }
        for (auto &frame : atoms) prepare_frame(frame);
        if (reverse)
            std::ranges::reverse(atoms);
        if (random_shuffle) {
            std::random_device rd;
            std::mt19937 mtd(rd());
            std::ranges::shuffle(atoms, mtd);
        }
        for (auto &frame : atoms) displace_frame(frame);
        if (lastonly) {
            std::vector<Atoms> atnew{atoms.back()};
            sort_traj(atnew);
            if (!iobj.write(atnew, outputfile, outformat)) return 1;
            return 0;
        }
        std::unique_ptr<std::ostream> ofs;
        if (vecoutput) {
            ofs = open_ofile(outputfile, iobj.append);
            if (!ofs) {
                std::cerr << "Error: cannot open output file " << outputfile << std::endl;
                return 1;
            }
            if (!iobj.seq_write_init(*ofs, outformat, outputfile)) return 1;
        }
        if (select.size() > 0) {
            std::vector<Atoms> atnew;
            for (int i = 0; i < select.size(); ++i) {
                if (select[i] >= 0 && select[i] < (int)atoms.size())
                    atnew.push_back(atoms[select[i]]);
            }
            if (atnew.empty()) {
                if (ofs && !close_ofile(ofs)) return 1;
                return 0;
            }
            sort_traj(atnew);
            if (vecoutput) {
                for (int i = 0; i < atnew.size(); ++i) {
                    if (!iobj.seq_write(*ofs, atnew[i])) return 1;
                }
            } else {
                for (int i = 0; i < atnew.size(); ++i) {
                    int id = i;
                    if (iobj.suffix_id_use_internal) id = atnew[i].id;
                    std::string fnm = outputfile + iobj.fname_separator + std::to_string(id);
                    if (!iobj.write_one(atnew[i], fnm)) return 1;
                }
            }
        } else {
            sort_traj(atoms);
            if (vecoutput) {
                for (int i = 0; i < atoms.size(); ++i) {
                    if (!iobj.seq_write(*ofs, atoms[i])) return 1;
                }
            } else {
                for (int i = 0; i < atoms.size(); ++i) {
                    std::string fnm = outputfile + iobj.fname_separator + std::to_string(atoms[i].id);
                    if (!iobj.write_one(atoms[i], fnm)) return 1;
                }
            }
        }
        if (ofs && !close_ofile(ofs)) return 1;
    } else {
        for (int i = 0; i < inputfiles.size(); ++i) {
            bool state = iobj.read(atoms, inputfiles[i], informat);
            if (!state) {
                std::cerr << "ERROR WHEN READING FILE " << inputfiles[i] << std::endl;
                if (iinformat == GPIO::XDATCAR6 ||
                    iinformat == GPIO::MLIPBIN || checked_text_input) return 1;
            }
            if (iinformat == GPIO::MLIPBIN && iobj.binary_read_limit_reached()) break;
        }
        if (atoms.empty()) {
            std::cerr << "Error: no structures were read" << std::endl;
            return 1;
        }
        transform_traj(atoms);
        if (lastonly) {
            std::vector<Atoms> atnew;
            atnew.push_back(atoms.back());
            sort_traj(atnew);
            if (!iobj.write(atnew, outputfile, outformat)) return 1;
        } else {
            if (select.size() > 0) {
                std::vector<Atoms> atnew;
                for (auto &i : select) {
                    if (i >= 0 && i < (int)atoms.size())
                        atnew.push_back(atoms[i]);
                }
                if (!atnew.empty()) {
                    sort_traj(atnew);
                    if (!iobj.write(atnew, outputfile, outformat)) return 1;
                }
            } else {
                sort_traj(atoms);
                if (vecoutput || atoms.size() > 1) {
                    if (!iobj.write(atoms, outputfile, outformat)) return 1;
                } else {
                    if (!iobj.write_one(atoms[0], outputfile, outformat)) return 1;
                }
            }
        }
    }
    return 0;
}

#undef USAGESTR
