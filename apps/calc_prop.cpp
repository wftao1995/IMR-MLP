// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "analyse/compute_bondangle.h"
#include "analyse/compute_sro.h"
#include "analyse/compute_ha.h"
#include "analyse/compute_voro.h"
#include "analyse/compute_rdf.h"
#include "analyse/compute_msd.h"
#include "analyse/compute_boo.h"
#include "analyse/compute_vacf.h"
#include "analyse/compute_posavg.h"
#include "analyse/compute_cna.h"
#include "analyse/compute_wcsro.h"
#include "utils/gputils.h"
#include "tools/chem.h"
#include "io/gpio.h"
#include "tools/tools.h"
#include "base/neighborhood_shell.h"
#include "base/neighborhood.h"
#include "base/neighborhood_voro_radi.h"
#include "base/neighborhoodKD.h"
#include "help_utils.h"
#include <array>
#include <sstream>
#include <fmt/format.h>
#include <memory>
#include <set>
#include <filesystem>

// TODO: for spice-spice interaction, how about just use neighbors of each species pair
// i.e. use Neighborhoods.get_pair(type1, type2)
// index.map : type radii mass
// TODO: migrate to new cmdopt driver

#ifdef USE_OMP
#include <omp.h>
#endif

#define USAGESTR "calc_prop [options] input.inp\n Calculate bond-angle, dihedral angle, ha-index, voro-index, sro and bond-order parameters\n"

using namespace std;
using namespace Eigen;

namespace {
std::filesystem::path resolved_output_path(const std::string &name) {
    auto path = std::filesystem::absolute(name);
    std::set<std::filesystem::path> visited;
    // Resolve the final symlink even when it points at an output that has not
    // been created yet; weakly_canonical alone retains a dangling link's name.
    for (;;) {
        path = (std::filesystem::weakly_canonical(path.parent_path()) / path.filename()).lexically_normal();
        if (!std::filesystem::is_symlink(path)) return path;
        if (!visited.insert(path).second) GPERROR("Output path contains a symbolic-link cycle");
        const auto target = std::filesystem::read_symlink(path);
        path = target.is_absolute() ? target : path.parent_path() / target;
    }
}
}

int calc_prop(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    GPSettings settings;

    bool c_sro = false;
    bool c_bondangle = false;
    bool c_dihedral = false;
    bool c_bop = false;
    bool voro_neighbor = false;
    bool c_ha = false;
    bool c_voro = false;
    bool c_rdf = false;
    bool c_msd = false;
    bool c_vacf = false;
    bool c_avgpos = false;
    bool avgpos_remove_drift = false;
    bool avgpos_fractional = false;
    bool c_cna = false;
    bool c_wcsro = false;
    bool output_all_indices = false;
    bool use_pair_cutoff = false;
    bool rdf_rho_all = false;
    bool no_rdf_pbc = false;

    int framestart = 0;
    int frameend = INT_MAX;

    string informat = "";
    string indexfile = "index.map";
    string settingfile = "";
    string trjfile = "input.cfg";
    string outfile_sro = "output_sro.txt";
    string outfile_cna = "output_cna.txt";
    string outfile_neig = "output_neighbor.txt";
    string outfile_ha_prefix = "out_total";
    string outfile_pcf_total = "gr_total.txt";
    string outfile_pcf_partial_prefix = "gr_partial";
    string outfile_xyz = "";

    double neig_cutoff = 5.0;
    double neig_cutoff_inner = -1.0; // negative means no
    int ntypes = -1;
    double angle_low = 0;
    double angle_high = M_PI;
    int voro_num_print = 10;
    double rdf_rlimit = 15.0;
    double **pair_cutoff;
    bool rdf_kd = false;

    int nskip = 1;

    Atoms atom;
    std::vector<Atoms> atoms;

    cmdopts.addop(neig_cutoff, "cutoff", "c", "Neighborhood cutoff radius")
        .addop(framestart, "framestart", "s", "Starting frame index for analysis")
        .addop(frameend, "frameend", "e", "Ending frame index for analysis")
        .addop(nskip, "skip-every", "skip", "Number of frames to skip between analyses")
        .addop(use_pair_cutoff, "pair-cutoff", "pc", "Enable type-dependent pairwise cutoff for neighbor lists", false, 0)
        .addop(rdf_rho_all, "rdf-rho-all", "", "Calculate density from single structure or average over all frames", false, 0)
        .addop(neig_cutoff_inner, "cutoff-inner", "ci", "Inner cutoff radius for neighbor shell exclusion")
        .addop(rdf_kd, "rdf-kd", "rkd", "Use KD-tree algorithm for RDF neighbor search", false, 0)
        .addop(no_rdf_pbc, "rdf-no-pbc", "rnp", "Disable periodic boundary conditions in KD-tree based RDF calculation", false, 0)
        .addop(output_all_indices, "output-all-indices", "", "Output all ha and voro indices", false, 0)
        .addop(c_bondangle, "bondangle", "b", "Calculate bond angle distributions", false, 0)
        .addop(c_dihedral, "dihedral", "d", "Calculate dihedral angle distributions", false, 0)
        .addop(c_sro, "sro", "sro", "Compute short-range order (SRO) parameters", false, 0)
        .addop(c_rdf, "rdf", "r", "Compute radial distribution/pair correlation function (RDF/PCF)", false, 0)
        .addop(rdf_rlimit, "ref-rlimit", "rl", "Maximum radius for pair correlation function calculation")
        .addop(c_ha, "ha-index", "ha", "Compute Honeycutt-Andersen (HA) indices (uses Voronoi neighbors if combined with --voro-index)", false, 0)
        .addop(c_voro, "voro-index", "voro", "Compute Voronoi indices (used as neighbor basis for HA indices when combined)", false, 0)
        .addop(c_bop, "bop", "bop", "Compute bond-order parameters for local structure analysis", false, 0)
        .addop(c_cna, "c-cna", "cna", "Compute Common Neighbor Analysis (CNA) indices", false, 0)
        .addop(c_wcsro, "c-wcsro", "wcsro", "Compute Warren-Cowley SRO (binary, off-lattice)", false, 0)
        .addop(c_avgpos, "avgpos", "caps", "Compute average atomic positions over trajectory", false, 0)
        .addop(avgpos_remove_drift, "avgpos-remove-drift", "", "Remove COM drift when averaging fixed-cell trajectories", false, 0)
        .addop(avgpos_fractional, "avgpos-fractional", "", "Average fractional positions and cell", false, 0)
        .addop(c_msd, "msd", "msd", "Compute mean squared displacement (MSD)", false, 0)
        .addop(c_vacf, "vacf", "vacf", "Compute velocity autocorrelation function (VACF)", false, 0)
        .addop(outfile_xyz, "output-xyz", "o", "Output annotated trajectory in EXTXYZ format")
        .addop(indexfile, "index", "id", "Index file mapping atom types to indices")
        .addop(trjfile, "input", "in", "Input trajectory file")
        .addop(informat, "input-format", "if", "Trajectory file format (e.g., lammps, xyz)")
        .addop(outfile_sro, "output-sro", "os", "Output file for SRO analysis results")
        .addop(outfile_cna, "output-cna", "", "CNA counts by selected frame, atom and index (also adds cna_<index> columns to --output-xyz)")
        .addop(outfile_neig, "output-neighbor", "on", "Output file for neighbor information")
        .addop(ntypes, "num-types", "nt", "Number of atom types (-1 for automatic detection)")
        .addop(voro_num_print, "voro-num-print", "vn", "Number of Voronoi indices to output")
        .addop(voro_neighbor, "voro-neighbor", "vb", "Use Voronoi neighbors for analysis", false, 0)
        .addop(settingfile, "settings", "", "Input configuration settings file", true, 1)
        // Register settings file options for each analysis module
        .attach_settings("sro:", "SRO analysis",
            [](std::ostream &os) { ComputeSRO sro; sro.InitSettings(); sro.ListSettings(os, "sro:"); })
        .attach_settings("ha_index:", "HA indices",
            [](std::ostream &os) { ComputeHA ha; ha.InitSettings(); ha.ListSettings(os, "ha_index:"); })
        .attach_settings("voro_index:", "Voronoi indices",
            [](std::ostream &os) { ComputeVoro voro; voro.InitSettings(); voro.ListSettings(os, "voro_index:"); })
        .attach_settings("bondangle:", "Bond angle analysis",
            [](std::ostream &os) {
                ComputeAngles ba; ba.InitSettings();
                ba.ListSettings(os, "bondangle:");
                std::string triplets, dihedrals;
                ConnectSettings selection;
                selection.Connect(triplets, "triplets", "optional semicolon-separated element triples, e.g. O H O; H O H");
                selection.Connect(dihedrals, "dihedrals", "optional semicolon-separated element quadruples, e.g. H O O H");
                selection.ListSettings(os, "bondangle:");
            })
        .attach_settings("bop:", "Bond order parameters",
            [](std::ostream &os) { ComputeBondOrder(GPSettings{}).ListSettings(os, "bop:"); })
        .attach_settings("rdf:", "RDF/PCF",
            [](std::ostream &os) {
                ComputeRDF rdf(0.1, 15.0); rdf.InitSettings();
                rdf.ListSettings(os, "rdf:");
            })
        .attach_settings("msd:", "MSD analysis",
            [](std::ostream &os) {
                GPSettings empty; ComputeMSD msd(empty); msd.ListSettings(os, "msd:");
            })
        .attach_settings("vacf:", "VACF analysis",
            [](std::ostream &os) {
                GPSettings empty; ComputeVACF vacf(empty); vacf.ListSettings(os, "vacf:");
            })
        .attach_settings("wcsro:", "Warren-Cowley SRO",
            [](std::ostream &os) { ComputeWCSRO wcsro; wcsro.InitSettings(); wcsro.ListSettings(os, "wcsro:"); })
        .attach_settings("cna:", "CNA analysis",
            [](std::ostream &os) {
                ComputeCNA cna; cna.InitSettings();
                cna.ListSettings(os, "cna:");
            })
        .set_helponexit(false)
        .phrase_options(argc, argv);
    if (cmdopts.displayhelp)
    {
        cmdopts.showhelp(std::cout);
        std::cout << std::endl << USAGESTR << std::endl;
        exit(0);
    }
    if (GPIO::file_exist(settingfile))
    {
        Message("Loading settings");
        settings.Load(settingfile);
    }
    apply_settings_to_cmdopts(cmdopts, settings);
    if ((c_cna || c_wcsro) && (!std::isfinite(neig_cutoff) || neig_cutoff <= 0))
        GPERROR("CNA/WCSRO requires a finite positive cutoff");
    const auto same_output = [](const std::string &a, const std::string &b) {
        if (a.empty() || b.empty()) return false;
        const auto left = resolved_output_path(a), right = resolved_output_path(b);
        std::error_code error;
        return left == right || std::filesystem::equivalent(left, right, error);
    };
    if (c_cna && (outfile_cna.empty() || same_output(outfile_cna, outfile_xyz) ||
                  ((c_sro || c_wcsro) && same_output(outfile_cna, outfile_sro)) ||
                  (c_sro && same_output(outfile_cna, outfile_neig))))
        GPERROR("CNA output must have a nonempty path distinct from the trajectory and SRO output");
    if ((c_sro || c_wcsro) && (outfile_sro.empty() || same_output(outfile_sro, outfile_xyz) ||
                              (c_sro && same_output(outfile_sro, outfile_neig))))
        GPERROR("SRO output must have a nonempty path distinct from the trajectory and neighbor output");
    if (GPIO::file_exist(indexfile))
        iobj.readmap(indexfile); // should be with atom radii
    else
        cerr << "ERROR: indexfile not found\n";
    if (voro_neighbor)
        c_voro = true;
    // read atoms
    Message("Reading trajectory file " + trjfile);
    if (framestart < 0) framestart = 0;
    if (frameend <= 0) frameend = INT_MAX;
    iobj.read_irate = nskip;
    iobj.read_startfrom = framestart;
    iobj.read_endto = frameend;
    if (!iobj.read(atoms, trjfile, informat) || atoms.empty()) {
        std::cerr << "ERROR: no structures loaded from " << trjfile << std::endl;
        return 1;
    }
    std::vector<int> wcsro_types;
    std::map<int, int> wcsro_type_map;
    if (c_wcsro) {
        std::set<int> present;
        for (const auto &configuration : atoms)
            present.insert(configuration.chemmap_.begin(), configuration.chemmap_.end());
        if (present.size() != 2)
            GPERROR("WCSRO requires exactly two species in the selected trajectory");
        wcsro_types.assign(present.begin(), present.end());
        for (int i = 0; i < 2; ++i) wcsro_type_map[wcsro_types[i]] = i;
    }

    Message("Configurations loaded, initializing computation engines");

    std::vector<int> typechem;
    iobj.report_chem(typechem);

    Message("Start computing");
    if (neig_cutoff_inner < 0)
        neig_cutoff_inner = 0;

    ComputeSRO *compute_sro;
    ComputeAngles *compute_ba;
    ComputeHA *compute_ha;
    ComputeVoro *compute_voro;
    ComputeRDF *compute_rdf;
    std::unique_ptr<ComputeCNA> compute_cna;
    std::unique_ptr<ComputeWCSRO> compute_wcsro;

    std::vector<double> mass;
    mass.resize(typechem.size());
    for (int i = 0; i < typechem.size(); ++i)
    {
        if (iobj.atomic_vals[i].size() >= 2)
        {
            mass[i] = iobj.atomic_vals[i][1];
            std::cout << "Mass of type " << i << " is " << mass[i] << std::endl;
        }
        else 
        {
            mass[i] = Chemicals::atomic_masses[Chemicals::atomic_numbers[iobj.get_type(i)]];
        }
    }

    if (c_sro)
    {
        compute_sro = new ComputeSRO(settings.ExtractSubSettings("sro"));
    }
    if (c_bondangle || c_dihedral)
    {
        const auto angle_settings = settings.ExtractSubSettings("bondangle");
        compute_ba = new ComputeAngles(angle_settings);
        const auto element_types = iobj.getrevmap();
        auto parse_sequences = [&](const std::string &text, int width, auto add) {
            if (text.empty()) return;
            std::istringstream list(text);
            std::string sequence;
            while (std::getline(list, sequence, ';')) {
                std::istringstream row(sequence);
                std::array<int, 4> types{};
                for (int i = 0; i < width; ++i) {
                    std::string element;
                    if (!(row >> element) || !element_types.contains(element))
                        GPERROR("Angle sequence requires " + std::to_string(width) + " mapped element symbols: " + sequence);
                    types[i] = element_types.at(element);
                }
                std::string extra;
                if (row >> extra)
                    GPERROR("Angle sequence requires " + std::to_string(width) + " mapped element symbols: " + sequence);
                add(types);
            }
        };
        std::set<bondangle_type_index> triples;
        std::set<dihedral_type_index> quadruples;
        parse_sequences(angle_settings["triplets"], 3, [&](const auto &t) { triples.emplace(t[0], t[1], t[2]); });
        parse_sequences(angle_settings["dihedrals"], 4, [&](const auto &t) { quadruples.emplace(t[0], t[1], t[2], t[3]); });
        compute_ba->set_bondangle_types(triples);
        compute_ba->set_dihedral_types(quadruples);
    }
    if (c_ha)
    {
        compute_ha = new ComputeHA(settings.ExtractSubSettings("ha_index"));
    }
    if (c_voro)
    {
        compute_voro = new ComputeVoro(settings.ExtractSubSettings("voro_index"), iobj.atomradii);
    }
    if (c_rdf)
    {
        compute_rdf = new ComputeRDF(settings.ExtractSubSettings("rdf"), rdf_rlimit);
    }
    if (c_cna)
    {
        compute_cna = std::make_unique<ComputeCNA>(settings.ExtractSubSettings("cna"));
    }
    if (c_wcsro)
    {
        compute_wcsro = std::make_unique<ComputeWCSRO>(settings.ExtractSubSettings("wcsro"));
    }
    if (ntypes == -1)
    {
        ntypes = typechem.size();
    }
    if (c_bondangle || c_dihedral)
        compute_ba->ntypes = ntypes;
    if (c_rdf)
        compute_rdf->ntypes = ntypes;

    std::vector<std::vector<pair_srotype>> results_sro(c_sro ? atoms.size() : 0);
    std::vector<std::map<sro_counter, size_t>> results_counter(c_sro ? atoms.size() : 0);
    ha_index_image hacountall;
    VoroInfo vorocountall;
    rdf_count_type rdf_count_all;
    rdf_pcount_type rdf_pcount_all;
    bondangle_count ba_count_all;
    dihedral_count di_count_all;
    bondangle_type ba_res_all;
    dihedral_type di_res_all;
    unsigned long rdf_total = 0;
    std::vector<unsigned long> rdf_type_count(ntypes, 0);
    // Only RDF requires a normalization volume. Other analyses may use cell-free structures.
    double num_rho = 0.0;
    double total_vol = 0;
    std::vector<double> pnum_rho(ntypes);
    std::vector<int> atsize(atoms.size());
    auto atomtypes = atoms[0].types_count_map();
    auto spmap = iobj.getmap();
    auto idmap = iobj.getrevmap();
    if (c_rdf) {
        auto valid_volume = [](const Atoms &frame) {
            const double volume = frame.get_volume();
            if (!std::isfinite(volume) || volume <= 0.0)
                GPERROR("RDF normalization requires a finite positive cell volume; open-system RDF normalization is not defined");
            return volume;
        };
        if (rdf_rho_all) {
            for (const auto &frame : atoms) total_vol += valid_volume(frame);
            if (!std::isfinite(total_vol) || total_vol <= 0.0)
                GPERROR("RDF normalization requires a finite positive total volume");
        } else {
            const double volume = valid_volume(atoms[0]);
            num_rho = static_cast<double>(atoms[0].size()) / volume;
            for (int i = 0; i < ntypes; ++i) pnum_rho[i] = atomtypes[i] / volume;
        }
    }

    if (c_bondangle)
    {
        compute_ba->initbret(ba_res_all);
    }
    if (c_dihedral)
    {
        compute_ba->initdret(di_res_all);
    }
    // pair-cutoff
    if (use_pair_cutoff && ntypes > 0)
    {
        pair_cutoff = new double*[ntypes];
        for (int i = 0; i < ntypes; ++i)
        {
            pair_cutoff[i] = new double[ntypes]{0};
        }
        auto pairs = settings.ExtractSubSettings("pair_cutoff");
        for (auto it : pairs.settings)
        {
            auto idx = it.first;
            auto idn = idx.rfind('-');      
            auto sp1 = idx.substr(0, idn);
            auto sp2 = idx.substr(idn+1, idx.length());
            double cutoffij = stod(it.second);
            pair_cutoff[idmap[sp1]][idmap[sp2]] = cutoffij;
            pair_cutoff[idmap[sp2]][idmap[sp1]] = cutoffij;
            if (cutoffij > neig_cutoff)
            {
                neig_cutoff = cutoffij;
                Warning("adjust global cutoff to " + to_string(cutoffij));
            }
        }
    }

    #ifdef USE_OMP
    omp_set_num_threads(omp_num_threads);
    #endif 

    // if we process large configuration, put it into main loop will comsume large amount of memory
    // and we need to get cutoff raidus form rdf computations
    if (c_rdf)
    {
        Message("Computing RDF, this may take times");
        compute_rdf->initret(rdf_count_all);
        compute_rdf->initret(rdf_pcount_all);
        #ifdef USE_OMP
        #pragma omp parallel
        {
            rdf_count_type rdf_count_private;
            rdf_pcount_type rdf_pcount_private;
            std::vector<unsigned long> rdf_type_count_private;
            compute_rdf->initret(rdf_count_private);
            compute_rdf->initret(rdf_pcount_private);
            rdf_type_count_private.resize(ntypes, 0);
            unsigned long rdf_total_private = 0;
            #pragma omp for
        #endif
            for (int i = 0; i < atoms.size(); ++i)
            {
                if (!(rdf_kd && no_rdf_pbc)) atoms[i].wrap();
                atsize[i] = atoms[i].size();
                if (rdf_kd && no_rdf_pbc)
                {
                    NeighborhoodKD nbhs(&atoms[i]);
                    #ifdef USE_OMP
                    compute_rdf->count_pcf(nbhs, rdf_count_private, rdf_pcount_private, rdf_total_private, rdf_type_count_private);
                    #else 
                    compute_rdf->count_pcf(nbhs, rdf_count_all, rdf_pcount_all, rdf_total, rdf_type_count);
                    #endif
                }
                else if (rdf_kd)
                {
                    NeighborhoodKD nbhs(&atoms[i], rdf_rlimit);
                    #ifdef USE_OMP
                    compute_rdf->count_pcf(nbhs, rdf_count_private, rdf_pcount_private, rdf_total_private, rdf_type_count_private);
                    #else 
                    compute_rdf->count_pcf(nbhs, rdf_count_all, rdf_pcount_all, rdf_total, rdf_type_count);
                    #endif
                }
                else 
                {
                    Neighborhoods nbhs(atoms[i], rdf_rlimit);
                    #ifdef USE_OMP
                    compute_rdf->count_pcf(nbhs, rdf_count_private, rdf_pcount_private, rdf_total_private, rdf_type_count_private);
                    #else 
                    compute_rdf->count_pcf(nbhs, rdf_count_all, rdf_pcount_all, rdf_total, rdf_type_count);
                    #endif
                }
                
            }
        #ifdef USE_OMP
            #pragma omp critical
            add2t1(rdf_pcount_all, rdf_pcount_private);
            #pragma omp critical
            add2t1(rdf_count_all, rdf_count_private);
            #pragma omp critical
            add2t1(rdf_type_count, rdf_type_count_private);
            #pragma omp critical
            rdf_total += rdf_total_private;
        }
        #endif

        // process rdf
        if (rdf_rho_all)
        {
            num_rho = (double)rdf_total / total_vol;
            for (int k = 0; k < ntypes; ++k)
            {
                pnum_rho[k] = (double)rdf_type_count[k] / total_vol;
            }
        }
        // total pcf
        rdf_result_type tpcf;
        compute_rdf->compute_pcf(rdf_count_all, rdf_total, num_rho, tpcf);
        if (neig_cutoff < 0)
        {
            neig_cutoff = compute_rdf->get_rdfcut(tpcf);
        }
        ofstream ofs_rdf_t(outfile_pcf_total);
        ofs_rdf_t << "# r g(r) n(r) \n";
        double accu1 = 0;
        double rdf_total_inv = rdf_total > 0 ? 1.0 / rdf_total : 0.0;
        for (int i = 0; i < compute_rdf->r_dists.size(); ++i)
        {
            accu1 += (double)rdf_count_all[i] * rdf_total_inv;
            ofs_rdf_t << compute_rdf->r_dists[i] << " " << tpcf[i] << " " << accu1 << '\n';
        }
        ofs_rdf_t.close();
        if (!ofs_rdf_t) GPERROR("Failed writing RDF output file: " + outfile_pcf_total);
        for (int i = 0; i < ntypes; ++i)
        {
            for (int j = 0; j < ntypes; ++j)
            {
                compute_rdf->compute_pcf(i, j, rdf_type_count[i], pnum_rho[j], rdf_pcount_all, tpcf);
                if (use_pair_cutoff)
                {
                    if (pair_cutoff[i][j] < 0)
                    {
                        pair_cutoff[i][j] = compute_rdf->get_rdfcut(tpcf);
                        pair_cutoff[j][i] = pair_cutoff[i][j];
                    }
                }
                accu1 = 0.0;
                const std::string partial_path = outfile_pcf_partial_prefix + "_" + spmap[i] + "-" + spmap[j];
                ofstream ofs(partial_path);
                ofs << "# r g(r) n(r) " + spmap[i] + spmap[j] + " \n";
                rdf_partial_type pind(i, j);
                double type_count_inv = rdf_type_count[i] > 0 ? 1.0 / rdf_type_count[i] : 0.0;
                const double pair_weight = i == j ? 1.0 : 0.5;
                for (int k = 0; k < compute_rdf->r_dists.size(); ++k)
                {
                    accu1 += pair_weight * (double)rdf_pcount_all[pind][k] * type_count_inv;
                    ofs << compute_rdf->r_dists[k] << " " << tpcf[k] << " " << accu1 << '\n';
                }
                ofs.close();
                if (!ofs) GPERROR("Failed writing RDF output file: " + partial_path);
            }
        }
        Message("RDF computation complete");
    }

    if (c_sro || c_bop || c_bondangle || c_dihedral || c_ha || c_voro) {
    #ifdef USE_OMP
    #pragma omp parallel
    {
        ha_index_image hacountall_private;
        VoroInfo vorocountall_private;
        bondangle_count ba_conut_private;
        dihedral_count di_count_private;
        bondangle_type ba_res_private;
        dihedral_type di_res_private;
        if (c_bondangle)
        {
            compute_ba->initbret(ba_res_private);
        }
        if (c_dihedral)
        {
            compute_ba->initdret(di_res_private);
        }
        ComputeBondOrder compute_bop(settings.ExtractSubSettings("bop")); // to make parallel or not
    #else 
        ComputeBondOrder compute_bop(settings.ExtractSubSettings("bop")); // to make parallel or not
    #endif
        if (settings["bop:cutoff"].empty())
            compute_bop.set_rcut(neig_cutoff);
    #ifdef USE_OMP
        #pragma omp for nowait schedule(static)
    #endif
        for (int i = 0; i < atoms.size(); i++)
        {
            VoroInfo voroinfo_tmp;
            NeighborhoodsVoro *pnei_voro = nullptr;
            AnyNeighborhoods *pnbhs = nullptr;
            atoms[i].wrap();
            if (voro_neighbor)
            {
                voroinfo_tmp = compute_voro->compute(atoms[i], &pnei_voro);
                pnbhs = pnei_voro;
            }
            else if (c_sro || c_bondangle || c_dihedral || c_ha)
            {
                pnbhs = new NeighborhoodsShell(atoms[i], neig_cutoff_inner, neig_cutoff);
            }
            if (c_bondangle)
            {
                bondangle_count count;
                bondangle_type res;
                if (use_pair_cutoff)
                    res = compute_ba->compute(*pnbhs, count, pair_cutoff);
                else 
                    res = compute_ba->compute(*pnbhs, count);
                #ifdef USE_OMP
                add2t1(ba_conut_private, count);
                add2t1(ba_res_private, res);
                #else
                add2t1(ba_count_all, count);
                add2t1(ba_res_all, res);
                #endif
            }
            if (c_dihedral)
            {
                dihedral_count count;
                dihedral_type res;
                if (use_pair_cutoff)
                    res = compute_ba->compute(*pnbhs, count, pair_cutoff);
                else 
                    res = compute_ba->compute(*pnbhs, count);
                #ifdef USE_OMP
                add2t1(di_count_private, count);
                add2t1(di_res_private, res);
                #else
                add2t1(di_count_all, count);
                add2t1(di_res_all, res);
                #endif
            }
            if (c_sro)
            {
                results_sro[i] = compute_sro->compute(atoms[i], *pnbhs, &results_counter[i]);
            }
            if (c_ha)
            {
                ha_index_image hainfo_tmp = use_pair_cutoff ? compute_ha->compute(*pnbhs, pair_cutoff)
                                                          : compute_ha->compute(*pnbhs);
                #ifdef USE_OMP
                hacountall_private += hainfo_tmp;
                #else
                hacountall += hainfo_tmp;
                #endif
            }
            if (c_voro)
            {
                if(!voro_neighbor)
                {
                    voroinfo_tmp = compute_voro->compute(atoms[i]);
                }
                #ifdef USE_OMP
                vorocountall_private += voroinfo_tmp;
                #else
                vorocountall += voroinfo_tmp;
                #endif
            }
            if (c_bop)
            {
                if (voro_neighbor)
                    compute_bop.compute(atoms[i], *pnbhs);
                else
                    compute_bop.compute(atoms[i]);
            }

            if (pnbhs)
            {
                delete pnbhs;
                pnbhs = nullptr;
            }
            #pragma omp critical
            cout << "Configuration #" << i << " done" << endl;
        }
    #ifdef USE_OMP
        #pragma omp critical
        hacountall += hacountall_private;
        #pragma omp critical
        vorocountall += vorocountall_private;
        #pragma omp critical
        add2t1(ba_count_all, ba_conut_private);
        #pragma omp critical
        add2t1(ba_res_all, ba_res_private);
        #pragma omp critical
        add2t1(di_res_all, di_res_private);
        #pragma omp critical
        add2t1(di_count_all, di_count_private);
    }
    #endif
    }
    Message("Writing outputs");
    if (c_sro)
    {

        Message("Writing outputs");
        ofstream ofs_sro(outfile_sro);
        ofstream ofs_neig(outfile_neig);
        if (!ofs_sro || !ofs_neig) GPERROR("Cannot open SRO or neighbor output");
        ofs_sro << "# SRO parameter of configurations: \n";
        ofs_sro << "ID   ";
        for (int i = 0; i < typechem.size(); ++i)
        {
            for (int j = i; j < typechem.size(); ++j)
            {
                ofs_sro << spmap[typechem[i]] + "-" + spmap[typechem[j]] << "    ";
            }
        }
        ofs_sro << endl;
        ofs_neig << "# Neighborhood info of configurations: \n";
        for (int i = 0; i < typechem.size(); ++i)
        {
            ofs_neig << spmap[typechem[i]] << "   ";
        }
        ofs_neig << endl;
        ofs_sro << setprecision(15);
        ofs_sro << fixed;
        for (int i = 0; i < atoms.size(); ++i)
        {
            if (results_sro[i].empty())
            {
                Warning("No SRO result for configuration " + to_string(i + 1) + ": no neighbor shells");
                ofs_sro << "Configuration " << i + 1 << "  unavailable (no neighbor shells)\n";
                ofs_neig << "Configuration #" << i + 1 << "\n";
                continue;
            }
            ofs_sro << "Configuration " << to_string(i + 1) << "  ";
            int prv_nn = results_sro[i][0].order + 1;
            ofs_sro << " " << to_string(prv_nn) << "nn: ";
            for (int k = 0; k < results_sro[i].size(); ++k)
            {
                int curr_nn = results_sro[i][k].order + 1;
                if (curr_nn != prv_nn)
                {
                    prv_nn = curr_nn;
                    ofs_sro << "  " << to_string(prv_nn) << "nn: ";
                }
                ofs_sro << setw(20) << results_sro[i][k].sro;
            }
            ofs_sro << '\n';

            ofs_neig << "Configuration #" << to_string(i + 1) << '\n';
            for (auto &it : results_counter[i])
            {
                ofs_neig << to_string(it.first.order + 1) << "nn: "
                         << spmap[it.first.typea] << "  " << spmap[it.first.typeb] << "  " << to_string(it.second) << '\n';
            }
        }

        ofs_sro.close();
        ofs_neig.close();
        if (!ofs_sro || !ofs_neig) GPERROR("Failed to write SRO or neighbor output");
        delete compute_sro;
    }
    if (c_bondangle || c_dihedral)
    {
        if (c_bondangle)
        {
            string outfilename_prefix = "analyse_bondangles";
            for (const auto &[bindex, bins] : ba_res_all) {
                string filename = outfilename_prefix + spmap[bindex.chem[0]] + spmap[bindex.chem[1]] + spmap[bindex.chem[2]] + ".txt";
                ofstream ofs(filename);
                const auto combination_count = ba_count_all[bindex];
                for (int t = 0; t < compute_ba->angles.size(); ++t) {
                    const double percentage = combination_count == 0 ? 0.0
                        : 100.0 * bins[t] / double(combination_count);
                    ofs << compute_ba->angles[t] * 180.0 / M_PI << "  " << percentage << '\n';
                }
                ofs.close();
                if (!ofs) GPERROR("Failed writing bond-angle output file: " + filename);
            }
        }
        if (c_dihedral)
        {
            string outfilename_prefix = "analyse_dihedralangles";
            for (const auto &[dindex, bins] : di_res_all) {
                string filename = outfilename_prefix + spmap[dindex.chem[0]] + spmap[dindex.chem[1]] + spmap[dindex.chem[2]] + spmap[dindex.chem[3]] + ".txt";
                ofstream ofs(filename);
                const auto combination_count = di_count_all[dindex];
                for (int t = 0; t < compute_ba->angles.size(); ++t) {
                    const double percentage = combination_count == 0 ? 0.0
                        : 100.0 * bins[t] / double(combination_count);
                    ofs << compute_ba->angles[t] * 180.0 / M_PI << "  " << percentage << '\n';
                }
                ofs.close();
                if (!ofs) GPERROR("Failed writing dihedral output file: " + filename);
            }
        }
        delete compute_ba;
    }
    
    if (c_ha)
    {
        ofstream ofs;
        int bflag = compute_ha->bflag();
        // total
        const string total_filename = outfile_ha_prefix + (voro_neighbor ? "_voro_neighbor.txt" : ".txt");
        ofs.open(total_filename, fstream::out);
        ofs << "Total analyse: \n";
        ofs << "Indices" << setw(15) << "Counts" << setw(15) << "Percentage" << '\n';
        ofs << "-----------------------------------" << '\n';
        if (compute_ha->has_selection())
            hacountall.outselected(ofs);
        else if (output_all_indices)
            hacountall.outall(ofs);
        else
            hacountall.outallcomm(ofs, bflag);
        ofs.close();
        if (!ofs) GPERROR("Failed writing HA output file: " + total_filename);
        // partial
        for (int i = 0; i < typechem.size(); ++i)
        {
            for (int j = i; j < typechem.size(); ++j)
            {
                string filename = "Bond_" + spmap[typechem[i]] + "_" + spmap[typechem[j]];
                if (voro_neighbor)
                    filename += "_voro.txt";
                else
                    filename += ".txt";
                ofs.open(filename);
                ofs << "Partial analyse for bond type " + spmap[typechem[i]] + "-" + spmap[typechem[j]] + ": \n"
                    << "Indices" << setw(15) << "Counts" << setw(15) << "Percentage" << '\n'
                    << "-----------------------------------" << '\n';
                if (compute_ha->has_selection())
                    hacountall.outpartselected(ofs, typechem[i], typechem[j]);
                else if (output_all_indices)
                    hacountall.outpart(ofs, typechem[i], typechem[j]);
                else
                    hacountall.outpartcomm(ofs, typechem[i], typechem[j], bflag);
                ofs.close();
                if (!ofs) GPERROR("Failed writing HA output file: " + filename);
            }
        }
        delete compute_ha;
    }

    if (c_voro)
    {
        ofstream ofs;
        //total
        ofs.open("voro_out_total.txt", fstream::out);
        ofs << "Total voro index:" << endl;
        ofs << "Indices" << setw(25) << "Counts" << setw(15) << "Percentage" << endl;
        ofs << "---------------------------------------------" << endl;
        if (output_all_indices)
            vorocountall.outall(ofs, compute_voro->get_num_edges());
        else
            vorocountall.outallcomm(ofs, compute_voro->get_num_edges(), voro_num_print);
        ofs.close();
        //partial
        for (int i = 0; i < typechem.size(); ++i)
        {
            ofs.open("Voro_" + spmap[typechem[i]] + ".txt", fstream::out);
            ofs << "Partial analyse for " + spmap[typechem[i]] + ": \n";
            ofs << "Indices" << setw(25) << "Counts" << setw(15) << "Percentage" << endl;
            ofs << "---------------------------------------------" << endl;
            if (output_all_indices)
                vorocountall.outpart(ofs,typechem[i], compute_voro->get_num_edges());
            else
                vorocountall.outpartcomm(ofs, typechem[i], compute_voro->get_num_edges(), voro_num_print);
            ofs.close();
        }
        delete compute_voro;
    }

    if (c_rdf)
    {
        delete compute_rdf;
    }

    if (use_pair_cutoff)
    {
        for (int i = 0; i < ntypes; ++i)
        {
            delete[] pair_cutoff[i];
        }
        delete[] pair_cutoff;
    }

    // Enumerate all enabled derived paths before either analysis publishes files.
    {
        std::vector<std::string> paths;
        if (c_vacf) {
            ComputeVACF planner(settings.ExtractSubSettings("vacf"));
            auto planned = planner.planned_output_paths(atoms.front());
            paths.insert(paths.end(), planned.begin(), planned.end());
        }
        if (c_msd) {
            ComputeMSD planner(settings.ExtractSubSettings("msd"));
            auto planned = planner.planned_output_paths(atoms.front());
            paths.insert(paths.end(), planned.begin(), planned.end());
        }
        for (size_t i = 0; i < paths.size(); ++i)
            for (size_t j = 0; j < i; ++j)
                if (same_output(paths[i], paths[j]))
                    GPERROR("MSD/VACF output collision: " + paths[i] + " and " + paths[j]);
    }

    if (c_avgpos) {
        Message("Computing POS AVG in serial mode");
        PositionAverageOptions options;
        options.geometry_only = true;
        options.coordinates = avgpos_fractional ? AverageCoordinates::Fractional : AverageCoordinates::Input;
        options.remove_drift = avgpos_remove_drift;
        if (avgpos_remove_drift)
            for (int type : atoms.front().chemmap_) options.masses.push_back(mass.at(type));
        ComputePosavg compute_posavg(std::move(options));
        compute_posavg.init(atoms[0]);
        for (int i = 0; i < atoms.size(); ++i) {
            compute_posavg.accumulate(atoms[i]);
        }
        compute_posavg.compute();
        if (!iobj.write_one(compute_posavg.atomacc, "COMPUTE_POSAVG_RESULT.xyz", "extxyz"))
            GPERROR("Failed to write average positions: COMPUTE_POSAVG_RESULT.xyz");
    }

    if (c_vacf)
    {
        Message("Computing VACF in serial mode");
        GPSettings vacf_settings = settings.ExtractSubSettings("vacf");
        const double raw_timestep = vacf_settings["timestep"].empty()
                                      ? 1.0 : std::stod(vacf_settings["timestep"]);
        vacf_settings.Modify("timestep", fmt::format("{:.17g}", raw_timestep * nskip));
        ComputeVACF compute_vacf(vacf_settings);
        compute_vacf.atom_masses = mass;
        compute_vacf.compute(atoms);
        compute_vacf.write_results();
    }

    if (c_msd)
    {
        Message("Computing MSD in serial mode");
        GPSettings msd_settings = settings.ExtractSubSettings("msd");
        const double raw_timestep = msd_settings["timestep"].empty()
                                      ? 1.0 : std::stod(msd_settings["timestep"]);
        msd_settings.Modify("timestep", fmt::format("{:.17g}", raw_timestep * nskip));
        ComputeMSD compute_msd(msd_settings);
        compute_msd.atom_masses = mass;
        compute_msd.compute(atoms);
        compute_msd.write_results();
    }


    // CNA and WCSRO analysis — serial compute
    if (c_cna && compute_cna) {
        Message("Computing CNA indices (serial)");
        std::vector<std::vector<cna_atom>> results(atoms.size());
        std::vector<std::vector<cna_atom_sp>> species_results(atoms.size());
        std::set<cna_index> indices;
        std::set<std::string> species_columns;
        const auto species_column = [](const cna_index_sp &index) {
            return "cna_sp_" + std::to_string(index.chem[0]) + "_" +
                std::to_string(index.chem[1]) + "_" + std::to_string(index.index) +
                "_" + std::to_string(index.index_type);
        };
        for (size_t i = 0; i < atoms.size(); ++i) {
            Neighborhoods nbhs(atoms[i], neig_cutoff);
            results[i] = compute_cna->compute(nbhs);
            for (const auto &site : results[i])
                for (const auto &[index, count] : site) indices.insert(index);
            if (compute_cna->species_information()) {
                species_results[i] = compute_cna->compute_sp(nbhs);
                for (const auto &site : species_results[i])
                    for (const auto &[index, count] : site)
                        species_columns.insert(species_column(index));
            }
        }
        ofstream output(outfile_cna);
        if (!output) GPERROR("Cannot open CNA output: " + outfile_cna);
        output << "# CNA counts; frame and atom are zero-based selected indices\n"
               << "# neighbor_type=-1 and species_index=-1 denote species-aggregated counts\n"
               << "# frame atom central_type neighbor_type cna_index species_index count\n";
        for (size_t frame = 0; frame < atoms.size(); ++frame) {
            auto &configuration = atoms[frame];
            const auto old_cna = [](const auto &entry) { return entry.first.starts_with("cna_"); };
            std::erase_if(configuration.local_attribs_i_, old_cna);
            std::erase_if(configuration.local_attribs_d_, old_cna);
            std::erase_if(configuration.local_attribs_v_, old_cna);
            std::erase_if(configuration.local_attribs_vi_, old_cna);
            configuration.local_attribs_i_["cna_total"] = Eigen::VectorXi::Zero(configuration.size());
            for (int index : indices)
                configuration.local_attribs_i_["cna_" + std::to_string(index)] = Eigen::VectorXi::Zero(configuration.size());
            for (const auto &column : species_columns)
                configuration.local_attribs_i_[column] = Eigen::VectorXi::Zero(configuration.size());
            for (int atom_id = 0; atom_id < configuration.size(); ++atom_id) {
                const auto &site = results[frame][atom_id];
                const int type = configuration.chemmap_[atom_id];
                if (site.empty()) output << frame << ' ' << atom_id << ' ' << type << " -1 0 -1 0\n";
                for (const auto &[index, count] : site) {
                    configuration.local_attribs_i_["cna_" + std::to_string(index)][atom_id] = count;
                    configuration.local_attribs_i_["cna_total"][atom_id] += count;
                    output << frame << ' ' << atom_id << ' ' << type << " -1 " << index << " -1 " << count << '\n';
                }
                if (compute_cna->species_information())
                    for (const auto &[index, count] : species_results[frame][atom_id]) {
                        configuration.local_attribs_i_[species_column(index)][atom_id] = count;
                        output << frame << ' ' << atom_id << ' ' << type << ' ' << index.chem[1]
                               << ' ' << index.index << ' ' << index.index_type << ' ' << count << '\n';
                    }
            }
        }
        output.close();
        if (!output) GPERROR("Failed to write CNA output: " + outfile_cna);
    }
    if (c_wcsro && compute_wcsro) {
        Message("Computing WCSRO (serial)");
        const double inner = compute_wcsro->inner_radius(neig_cutoff_inner);
        const double outer = compute_wcsro->outer_radius(neig_cutoff);
        if (!std::isfinite(inner) || !std::isfinite(outer) || inner < 0 || outer <= inner)
            GPERROR("WCSRO radii must satisfy finite 0 <= inner < outer");
        std::vector<size_t> counter(4, 0), type_counter(2, 0);
        size_t total_atoms = 0, total_pairs = 0;
        for (int i = 0; i < (int)atoms.size(); ++i) {
            Atoms mapped = atoms[i];
            for (int &type : mapped.chemmap_) type = wcsro_type_map.at(type);
            NeighborhoodsShell nbhs(mapped, inner, outer);
            compute_wcsro->compute(nbhs, counter, type_counter, total_atoms, total_pairs);
        }
        std::vector<std::vector<double>> sro_results;
        compute_wcsro->postprocess(counter, type_counter, total_atoms, total_pairs, sro_results);
        ofstream output(outfile_sro, c_sro ? std::ios::app : std::ios::out);
        if (!output) GPERROR("Cannot open WCSRO output: " + outfile_sro);
        output << std::setprecision(17)
               << "# WCSRO pooled over " << atoms.size() << " selected frames; directed conditional probabilities\n"
               << "# types " << wcsro_types[0] << '=' << spmap[wcsro_types[0]] << ' '
               << wcsro_types[1] << '=' << spmap[wcsro_types[1]] << '\n'
               << "# type_i type_j concentration_i concentration_j directed_pairs sro\n";
        for (int i = 0; i < 2; ++i)
            for (int j = 0; j < 2; ++j)
                output << wcsro_types[i] << ' ' << wcsro_types[j] << ' '
                       << static_cast<double>(type_counter[i]) / total_atoms << ' '
                       << static_cast<double>(type_counter[j]) / total_atoms << ' '
                       << static_cast<double>(counter[i+j]) / (i == j ? 1 : 2) << ' '
                       << sro_results[i][j] << '\n';
        output.close();
        if (!output) GPERROR("Failed to write WCSRO output: " + outfile_sro);
    }

    if (c_bop || !outfile_xyz.empty()) {
        if (!iobj.write(atoms, outfile_xyz, "extxyz"))
            GPERROR("Failed to write trajectory: " + outfile_xyz);
    }

    Message("Complete!");
    return 0;
}

#undef USAGESTR
