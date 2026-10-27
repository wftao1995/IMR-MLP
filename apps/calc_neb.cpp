// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mindriver/neb.h"
#include <iomanip>
#include <optional>
#include "mindriver/lbfgs.h"
#include "mindriver/fire.h"
#include "mindriver/sgd.h"
#include "mindriver/cg.h"
#include "active_learn/al_potential.h"
#include "tools/chem.h"
#include "tools/constants.h"
#include "deplibs/LBFGSpp/include/LBFGS.h"
#include "potentials/potential_wrapper.h"
#include "libalglib/interpolation.h"
#include "libalglib/ap.h"
#include "help_utils.h"
#include "neb_parallel.h"

using namespace std;


#ifdef IMRMLP_MPI
#include <mpi.h>
#endif

class NEBAPPParam : public ConnectSettings {
public:
    vector<string> inputstrfile;
    string informat = "";
    int nimgs = 0;
    int interp_npoints = 100;

    LBFGSpp::LBFGSParam<double> lbfgsparam;

    string indexfile = "index.map";
    string mepout = "mep.xyz";
    string mepformat = "extxyz";
    string summary = "neb.dat";
    string profile, reference_cell, path_pbc;
    bool rebegin = false;
    double cellweight = 100;

    // string potfile = "potential.mtp";
    // string pottype = "mtp";

    string driver = "lbfgs"; // lbfgs or fire
    int max_total_iter = 1000000; // total iteration budget (for variable-image restarts)

    // lbgs params
    int lbfgs_m = 15;

    void InitSettings() {
        CMDOption::CommandLineOptions &cmdopts = CMDOption::global_cmdopts;
        ConnectWithConsole(cmdopts, informat, "input-format", "if", "Input structure format");
        ConnectWithConsole(cmdopts, nimgs, "nimgs", "", "Number of images, including initial and final images");
        ConnectWithConsole(cmdopts, indexfile, "index", "id", "Index file mapping atom types to indices");
        ConnectWithConsole(cmdopts, interp_npoints, "ninterp", "", "Number of interpolation points for MEP curve");
        ConnectWithConsole(cmdopts, cellweight, "cellweight", "cwt", "Deprecated; use neb:cellweight (top-level value overrides it)", 1, 10);

        ConnectWithConsole(cmdopts, mepout, "mepout", "", "MEP trajectory output file");
        ConnectWithConsole(cmdopts, mepformat, "mepformat", "", "MEP trajectory format (extxyz in default)");
        ConnectWithConsole(cmdopts, path_pbc, "neb:path_pbc", "", "Explicit PBC bits (e.g. 110) for a fresh complete path imported from POSCAR; empty preserves input flags");
        ConnectWithConsole(cmdopts, reference_cell, "neb:reference_cell", "", "Reference cell file for a complete fresh VC path; preserves its frame, disables MIC/IDPP; incompatible with restart");
        ConnectWithConsole(cmdopts, profile, "neb:profile", "", "Optional MPI+OMP timing JSON filename; empty disables (default)");
        ConnectWithConsole(cmdopts, summary, "summary", "", "Output MEP summary");
        // opt settings
        ConnectWithConsole(cmdopts, lbfgsparam.m, "lbfgs_m");
        ConnectWithConsole(cmdopts, lbfgsparam.epsgmax, "gtol", "gtol", "Complete active NEB force tolerance; overrides neb:gtol");
        ConnectWithConsole(cmdopts, lbfgsparam.initial_step, "initial_step", "", "Initial step size", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.max_step, "max_step", "", "maximum step size allowed", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.max_step_move, "max_move", "", "maximum step size allowed");
        ConnectWithConsole(cmdopts, lbfgsparam.s_curv_coeff, "s_curv_coeff", "", "", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.damping, "damping", "", "mixing parameter in lbfgs driver", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.epsilon, "epsilon", "", "accuracy of force, norm of force vector", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.epsilon_rel, "epsilon_rel", "", "", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.past, "past", "", "", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.delta, "delta", "", "", 1, 2);
        ConnectWithConsole(cmdopts, lbfgsparam.first_initial_step, "first_step", "", "first step size allowed");
        ConnectWithConsole(cmdopts, lbfgsparam.max_iterations, "maxiteration", "maxit", "iteration limit");
        ConnectWithConsole(cmdopts, max_total_iter, "max_total_iter", "", "total iteration budget across all image-adjust restarts");

        ConnectWithConsole(cmdopts, driver, "driver", "", "Optimization driver, valid options are lbfgs, sgd, cg, fire. FIRE is the most stable one");
        ConnectWithConsole(cmdopts, rebegin, "next_step", "r", "restart calculation by reading nebimage_xx_final.vasp", 0);
    }

    NEBAPPParam(){
        InitSettings();
    };

    void InitBySettings(GPSettings &settings) {
        // InitSettings();
        ApplySettings(settings);
        if (nimgs < 3) {
            throw std::runtime_error("nimgs must be at least 3 (including endpoints)");
        }
        if (interp_npoints < 2) throw std::runtime_error("ninterp must be at least 2");
        // read images, images id should start form 0
        GPSettings sub = settings.ExtractSubSettings("image");
        for (int i = 0; i < nimgs; ++i) {
            std::string fnm = sub[to_string(i)];
            inputstrfile.push_back(fnm);
        }
        if (rebegin) {
            inputstrfile.resize(nimgs);
            for (int i = 0; i < nimgs; ++i) {
                inputstrfile[i] = "nebimage_" + std::to_string(i) + "_final.vasp";
            }
        }
        // make sure essential structrues are not empty
        if (inputstrfile.empty()) {
            throw std::runtime_error("No input structure files specified");
        }
        if (inputstrfile[0].empty()) {
            throw std::runtime_error("First input structure file path is empty");
        }
        if (inputstrfile.back().empty()) {
            throw std::runtime_error("Last input structure file path is empty");
        }
// PrintSettings();
    }
};

static int calc_neb(int argc, char *argv[], NEBParallel &parallel) {
    CMDOption::CommandLineOptions &cmdopts = CMDOption::global_cmdopts;
    GPSettings settings;
    string settingfile;
    GPIO iobj;
    iobj.packatoms = false;
    double pressure = 0;
    bool create_sample_input = false;
    NEBAPPParam params;
    cmdopts.addop(settingfile, "settings", "", "Input setting file.", true)
           .addop(create_sample_input, "create-input", "", "Create a basic input file example for neb calculation", false, 0)
           .attach_settings("neb:", "NEB driver",
               [](std::ostream &os) { NEBDriver().ListSettings(os, "neb:"); })
           .attach_settings("lbfgs:", "L-BFGS optimizer",
               [](std::ostream &os) { SimpleLBFGS().ListSettings(os, "lbfgs:"); })
           .attach_settings("fire:", "FIRE optimizer",
               [](std::ostream &os) { SimpleFireF().ListSettings(os, "fire:"); })
           .attach_settings("cg:", "Conjugate gradient optimizer",
               [](std::ostream &os) { SimpleCGF(GPSettings{}).ListSettings(os, "cg:"); })
           .attach_settings("sgd:", "Gradient descent optimizer",
               [](std::ostream &os) { SimpleSGD(GPSettings{}).ListSettings(os, "sgd:"); })
           .set_helponexit(false)
           .phrase_options(argc, argv);
    if (cmdopts.displayhelp)
    {
        std::cout << "Usage: " + std::string(argv[0]) << " input.ini" << std::endl;
        std::cout << "Nudged elastic band calculation\n";
        cmdopts.showhelp(std::cout);
        std::cout << "\nParallel execution: MPI ranks distribute images; only rank 0 optimizes and writes output.\n"
                     "neb:omp_mode images (default): OpenMP over each rank's images.\n"
                     "neb:omp_mode atoms: images serial per rank, OpenMP inside the potential (if supported).\n"
                     "--num_threads N sets threads per rank; avoid nested BLAS/OpenMP oversubscription.\n"
                     "All ranks need the same potential files. External calculators must allocate their own resources.\n"
                     "MPI active learning is currently unsupported. Exit codes: 0 converged, 2 unconverged, 1 error.\n"
                     "Top-level gtol overrides neb:gtol; relax: settings do not apply.\n"
                     "Fixed-cell tolerance is eV/Angstrom; VC-NEB uses a generalized coordinate metric.\n"
                     "Symmetry: optional report, smooth seeded perturbation, or common-subgroup constraints.\n"
                     "Constraints require full PBC; VC requires symm_strain. No atom remapping or path reversal detection.\n"
                     "perturb_amplitude is a Cartesian Angstrom bound; keep_symm is incompatible with perturbation/masks.\n"
                     "New constraints/perturbations require a fresh calculation, not --next_step.\n"
                     "Optional neb:profile FILE records MPI/OMP stage timing; empty disables.\n"
                     "Optional neb:symmetry_release DIR runs bounded independent release trials after saving the main result.\n"
                     "Release trials do not certify a saddle or global minimum path; inspect DIR/report.json.\n"
                     "Restart retains its existing path/phase recovery behavior and resets optimizer history.\n";
        return 0;
    }
    if (create_sample_input) {
        ofstream ofs("neb_simple_example.ini");
        if (!ofs) GPERROR("Cannot open NEB example output: neb_simple_example.ini");
        ofs << fmt::format("{:20} {:20} {}\n", "informat", "POSCAR", "# Input structure file format");
        ofs << fmt::format("{:20} {:20} {}\n", "index", "index.map", "# Species index");
        ofs << fmt::format("{:20} {:20} {}\n", "gtol", "1e-2", "# Force tolerance, this should be adjusted according to lattice parameter for VCNEB");
        ofs << fmt::format("{:20} {:20} {}\n", "max_move", "0.02", "# Maximum step size");
        ofs << fmt::format("{:20} {:20} {}\n", "first_step", "0.001", "# First step size, should be small");
        ofs << fmt::format("{:20} {:20} {}\n", "mepout", "mep.xyz", "# Output file containing MEP trajectory");
        ofs << fmt::format("{:20} {:20} {}\n", "summary", "neb.dat", "# Summary output");
        ofs << fmt::format("{:20} {:20} {}\n", "driver", "fire", "# Optimization driver, fire (stable but slow) or lbfgs (fast but unstable)");
        ofs << fmt::format("{:20} {:20} {}\n", "maxiteration", "2000", "# Iteration limit");
        ofs << endl;
        ofs << fmt::format("{:20} {:20} {}\n", "pot:pottype", "eamalloy", "# Potential type, use shellfile for external calculator");
        ofs << fmt::format("{:20} {:20} {}\n", "pot:potfile", "Ni.eam.alloy", "# Potential filename");
        ofs << endl;
        ofs << fmt::format("{:20} {:20} {}\n", "nimgs", "5", "# Total number of images, including initial and final");
        ofs << fmt::format("{:20} {:20} {}\n", "num_threads", "8", "# Number of threads to use, can be any value, better no larger than physical cores");
        ofs << fmt::format("{:20} {:20} {}\n", "image:0", "POSCAR_INI", "# Images, image:0 to image:nimgs-1, initial and final image must present, missing imges will be generated by linear interpolation");
        ofs << fmt::format("{:20} {:20} {}\n", "image:2", "POSCAR_MID", "# Images, image:0 to image:nimgs-1, initial and final image must present, missing imges will be generated by linear interpolation");
        ofs << fmt::format("{:20} {:20} {}\n", "image:4", "POSCAR_FIN", "# Images, image:0 to image:nimgs-1, initial and final image must present, missing imges will be generated by linear interpolation");
        ofs << endl;
        ofs << fmt::format("{:20} {:20} {}\n", "neb:vcneb", "false", "# Enable VCNEB");
        ofs << fmt::format("{:20} {:20} {}\n", "neb:spring", "5", "# Spring constant");
        ofs << fmt::format("{:20} {:20} {}\n", "neb:climb", "true", "# Enable CI-NEB");
        ofs << fmt::format("{:20} {:20} {}\n", "neb:valley", "false", "# Enable valley-image refinement");
        ofs << fmt::format("{:20} {:20} {}\n", "neb:pin", "false", "# Pin local maximum when climb=True, local minimum when valley=True, only set this true for refining results");
        ofs << fmt::format("{:20} {:20} {}\n", "neb:check_mic", "false", "# Note in some special case this may not work");
        ofs << fmt::format("{:20} {:20} {}\n", "neb:write_traj", "false", "# Write trajectory or not, may be very large");
        ofs << fmt::format("{:20} {:20} {}\n", "neb:pressure", "0", "# External pressure for VCNEB, in unit GPa");
        ofs.close();
        if (!ofs) GPERROR("Failed to write NEB example: neb_simple_example.ini");
        std::cout << "Sample input file created: neb_simple_example.ini" << std::endl;
        std::cout << "You can modify the file to suit your needs." << std::endl;
        return 0;
    }
    if (GPIO::file_exist(settingfile))
    {
        Message("Loading settings");
        settings.Load(settingfile);
        cmdopts.apply_overrides(settings);
        // settings.Print();
        params.InitBySettings(settings);
        apply_settings_to_cmdopts(cmdopts, settings);
    } else {
        std::cerr << "Error: setting file " << settingfile << " not found" << std::endl;
        return 128;
    }

    std::map<std::string,std::string> restart_metadata;
    if (params.rebegin) {
        std::ifstream previous(params.summary);
        std::string line;
        while (std::getline(previous, line)) {
            std::istringstream input(line);
            std::string hash, key, value;
            if (input >> hash >> key >> value; hash == "#") restart_metadata[key]=value;
        }
        if(!settings.count("neb:solver_policy")) {
            const auto policy=restart_metadata.count("solver_policy")?restart_metadata["solver_policy"]:"legacy";
            settings.Modify("neb:solver_policy",policy);
            Message("NEB restart numerical policy: "+policy);
            if(!restart_metadata.count("solver_policy"))Warning("Restart has no solver policy; using legacy");
        }
    }
    std::array<double,4> stages{};
    auto stamp=params.profile.empty()?NEBReporting::Clock::time_point{}:NEBReporting::Clock::now();
    NEBDriver nebobj(settings.ExtractSubSettings("neb"));
    // Keep neb:cellweight when present; the legacy top-level spelling remains
    // an explicit override for backwards compatibility.
    if (settings.count("cellweight") || cmdopts.was_provided("cellweight")) {
        Warning("Top-level cellweight is deprecated; use neb:cellweight. The top-level value takes precedence.");
        nebobj.cell_weight = params.cellweight;
    }
    pressure = nebobj.pressure;
    Potentials::potential_wrapper wrapper(settings.ExtractSubSettings("pot"));
    nebobj.pot = wrapper.pot;
    parallel.init(settings.ExtractSubSettings("pot"), *wrapper.pot, nebobj.omp_mode, !params.profile.empty());
    Message(fmt::format("NEB evaluation: {} MPI rank(s), {} OpenMP thread(s) per rank, mode={}",
                        mpi.size, get_omp_max_threads(), nebobj.omp_mode));
    nebobj.efs_evaluator = [&](std::vector<Atoms> &images,int begin,int end) { parallel(images,begin,end); };

    // ── On-the-fly active learning (optional) ───────────────────────
    auto al = AL::ALPotential::create(settings.ExtractSubSettings("al"));
    if (al && al->active()) {
        if (mpi.size>1) throw std::invalid_argument("NEB: active learning currently requires one MPI rank");
        al->init_builder(*nebobj.pot);
        nebobj.al_pot_ = al.get();
    }

    // init images
    if (!GPIO::file_exist(params.indexfile)) {
        throw std::runtime_error("Index file not found: " + params.indexfile);
    }
    iobj.readmap(params.indexfile);
    std::optional<std::array<bool,3>> restart_pbc;
    if (params.rebegin) {
        std::ifstream previous(params.summary);
        std::string line;
        while (std::getline(previous,line)) {
            std::istringstream input(line);
            std::string hash,key,value;
            if (input>>hash>>key>>value; hash=="#" && key=="pbc" && value.size()==3 &&
                value.find_first_not_of("01")==std::string::npos)
                restart_pbc=std::array<bool,3>{value[0]=='1',value[1]=='1',value[2]=='1'};
        }
        if (!restart_pbc && GPIO::file_exist(settings["image:0"])) {
            std::vector<Atoms> original;
            if(iobj.read(original,settings["image:0"],params.informat) && !original.empty())
                restart_pbc=original[0].pbc_;
        }
    }
    vector<Atoms> atoms(params.nimgs);
    // read confs if present
    for (int i = 0; i < params.nimgs; ++i) {
        if (!params.inputstrfile[i].empty()) {
            if (!GPIO::file_exist(params.inputstrfile[i])) {
                throw std::runtime_error("Input structure file not found: " + params.inputstrfile[i]);
            }
            std::vector<Atoms> ats;
            if (!iobj.read(ats, params.inputstrfile[i], params.rebegin ? "POSCAR" : params.informat)) {
                std::cerr << "Failed to read NEB input structure: " << params.inputstrfile[i] << '\n';
                return 1;
            }
            if (ats.empty())
                throw std::runtime_error("No structures read from: " + params.inputstrfile[i]);
            atoms[i] = ats[0];
            if (restart_pbc) atoms[i].pbc_=*restart_pbc;
        }
    }
    if(!params.path_pbc.empty()) {
        if(params.rebegin || params.path_pbc.size()!=3 || params.path_pbc.find_first_not_of("01")!=std::string::npos)
            throw std::invalid_argument("neb:path_pbc requires three binary digits and a fresh calculation");
        for(auto &a:atoms)a.pbc_={params.path_pbc[0]=='1',params.path_pbc[1]=='1',params.path_pbc[2]=='1'};
    }
    // Explicit geometry import is a fresh calculation, not optimizer recovery.
    if(!params.reference_cell.empty()) {
        if(params.rebegin || !nebobj.vcneb)throw std::invalid_argument("neb:reference_cell requires a fresh VC path");
        std::ifstream input(params.reference_cell);Eigen::Matrix3d reference;
        for(int i=0;i<3;++i)for(int j=0;j<3;++j)
            if(!(input>>reference(i,j)))throw std::invalid_argument("Cannot read neb:reference_cell");
        if(nebobj.keep_symm || nebobj.perturb_amplitude>0)throw std::invalid_argument("Explicit reference import requires an already prepared unconstrained path");
        nebobj.prepare_complete_path(atoms,reference);
        nebobj.preserve_complete_path=true;
    }
    // check if restart, read refcell
    if (params.rebegin && nebobj.vcneb) {
        std::ifstream cref("neb_reference_cell.data");
        if (cref.is_open()) {
            Eigen::Matrix3d refc;
            for (int i = 0; i < 3; ++i) {
                for (int j = 0; j < 3; ++j) {
                    if(!(cref >> refc(i,j)))throw std::runtime_error("Cannot read NEB restart reference cell");
                }
            }
            nebobj.refcell = refc;
            nebobj.ivrefcell.setZero(); // validated and inverted by init
            nebobj.iref = -3;
            cref.close();
        } else {
            if(nebobj.solver_policy=="stable-v1")throw std::runtime_error("VCNEB restart requires neb_reference_cell.data");
            Warning("Reference cell file neb_reference_cell.data not found, cannot restart vcneb properly!");
        }
    }
    // Set up neb object, interpolate
    Message("Init path");
    // On restart, skip MIC — images are already optimized and consistent.
    // Re-applying MIC shifts atoms by cell vectors, changing the tangent
    // direction and making the initial gradient differ from the previous run.
    if (params.rebegin) {
        nebobj.check_mic = false;
    }
    if(params.rebegin && (nebobj.keep_symm || nebobj.perturb_amplitude>0 || !nebobj.symmetry_operations.empty()))
        throw std::invalid_argument("NEB: new symmetry constraints/perturbations are not supported by --next_step; start a new path calculation");
    nebobj.init(atoms);
    nebobj.iobj = &iobj;
    // write images
    for (int i = 0; i < nebobj.images.size(); ++i) {
        const string filename = "nebimage_" + to_string(i) + ".vasp";
        if (!iobj.write_one(nebobj.images[i], filename, "POSCAR"))
            GPERROR("Failed to write NEB image: " + filename);
    }

    // Every public entry point uses the same NEB acceptance controller.
    SimpleFireF fireopt(settings);
    SimpleSGD sgdopt(settings);
    SimpleCGF cgopt(settings);
    if (settings.count("gtol")) nebobj.neb_ftol = params.lbfgsparam.epsgmax;
    NEBDriver::RunOptions options;
    options.driver = params.driver;
    options.optimizer_settings = settings;
    if(settings.count("neb:experimental_acceleration"))
        options.experimental_acceleration=settings["neb:experimental_acceleration"];
    if(settings.count("neb:diagnostic_gmres_max_directions"))
        options.max_krylov_directions=std::stoi(settings["neb:diagnostic_gmres_max_directions"]);
    if(options.max_krylov_directions>160)Message("NEB: extended Krylov diagnostic, maximum directions="+std::to_string(options.max_krylov_directions));
    if(params.rebegin && restart_metadata["solver_policy"]==nebobj.solver_policy) {
        if(restart_metadata.count("accepted_steps"))options.accepted_step_offset=std::stoi(restart_metadata["accepted_steps"]);
        if(nebobj.pin && restart_metadata["pinned_stage"]=="true") {
            auto read_indices=[&](const std::string &key,std::set<int> &indices) {
                std::istringstream input(restart_metadata[key]);std::string part;
                while(std::getline(input,part,','))if(!part.empty() && part!="none") {
                    int index=std::stoi(part);
                    if(index<1 || index>=nebobj.images.size()-1)throw std::runtime_error("Invalid pinned image in NEB restart");
                    indices.insert(index);
                }
            };
            read_indices("climbing_images",nebobj.iclimbs);read_indices("valley_images",nebobj.ivallys);
            nebobj.found_state_site=true;
            if(restart_metadata.count("protected_image"))nebobj.iclimb_=std::stoi(restart_metadata["protected_image"]);
        }
    }
    const int driver_limit = params.driver == "lbfgs" ? params.lbfgsparam.max_iterations
        : params.driver == "fire" ? fireopt.max_iter
        : params.driver == "cg" ? cgopt.max_iter : sgdopt.max_iter;
    options.max_iterations = std::min(params.max_total_iter,
        driver_limit > 0 ? driver_limit : (nebobj.solver_policy == "stable-v1" ? 1000 : params.max_total_iter));
    options.max_move = params.driver == "lbfgs" ? params.lbfgsparam.max_step_move
        : params.driver == "fire" ? fireopt.max_step
        : params.driver == "cg" ? cgopt.max_step : sgdopt.max_step;
    if (nebobj.solver_policy == "stable-v1" && !settings.count("max_move")) options.max_move = .1;
    if (nebobj.diagnostics_file.empty()) nebobj.diagnostics_file = params.summary + ".iterations.csv";
    if(parallel.profiling()) {stages[0]=NEBReporting::seconds(stamp);stamp=NEBReporting::Clock::now();}
    nebobj.run(options);
    if(parallel.profiling()) {stages[1]=NEBReporting::seconds(stamp);stamp=NEBReporting::Clock::now();}
    Message(fmt::format("NEB stopped after {} steps: {} (max NEB force = {:.6g})",
                        nebobj.optimizer_iterations, nebobj.stop_reason_string(), nebobj.max_neb_force));
    // Output results
    if (!iobj.write(nebobj.images, params.mepout, params.mepformat))
        GPERROR("Failed to write NEB path: " + params.mepout);
    // final images in POSCAR format
    for (int i = 0; i < nebobj.images.size(); ++i) {
        const string filename = "nebimage_" + to_string(i) + "_final.vasp";
        if (!iobj.write_one(nebobj.images[i], filename, "POSCAR"))
            GPERROR("Failed to write NEB final image: " + filename);
    }
    // reference cell
    if (nebobj.vcneb) {
        ofstream cref("neb_reference_cell.data");
        if (!cref) GPERROR("Cannot open NEB reference cell output: neb_reference_cell.data");
        cref << std::setprecision(17) << nebobj.refcell << std::endl;
        cref.close();
        if (!cref) GPERROR("Failed to write NEB reference cell: neb_reference_cell.data");
    }
    // calcualte NEB results, i.e. fit neb results
    alglib::real_1d_array ae_x, ae_y, ae_g;
    ae_x.setlength(nebobj.images.size());
    ae_y.setlength(ae_x.length());
    ae_g.setlength(ae_x.length());
    ae_x[0] = 0;
    ae_y[0] = 0;
    ae_g[0] = 0;
    int nimg = nebobj.images.size();
    double d = 0;
    double g = 0;
    ofstream ofs(params.summary);
    if (!ofs) GPERROR("Cannot open NEB summary: " + params.summary);
    fmt::print(ofs, "# status {}\n", nebobj.stop_reason_string());
    fmt::print(ofs, "# pbc {}{}{}\n", int(nebobj.images.front().pbc_[0]),
               int(nebobj.images.front().pbc_[1]),int(nebobj.images.front().pbc_[2]));
    fmt::print(ofs, "# solver_policy {}\n# climb_delay_units {}\n", nebobj.solver_policy,
               nebobj.solver_policy == "stable-v1" ? "accepted_steps" : "force_evaluations");
    fmt::print(ofs,"# max_gmres_directions {}\n# numerical_variant {}\n",options.max_krylov_directions,
               options.max_krylov_directions>160?"extended_krylov_diagnostic":"standard");
    fmt::print(ofs,"# experimental_acceleration {}\n",options.experimental_acceleration);
    fmt::print(ofs, "# solver_phase {}\n# optimizer_iteration_units {}\n# force_evaluation_units band\n",
               nebobj.climbing_active ? "climbing" : (nebobj.vally && !nebobj.ivallys.empty() ? "valley" : "ordinary"),
               nebobj.solver_policy == "stable-v1" ? "attempts" : "legacy_optimizer_iterations");
    if(nebobj.solver_policy=="stable-v1")
        fmt::print(ofs, "# max_atomic_move_A {:.17g}\n# max_cell_strain {:.17g}\n", options.max_move, nebobj.max_cell_strain);
    else fmt::print(ofs,"# legacy_max_move {:.17g}\n",options.max_move);
    fmt::print(ofs, "# accepted_steps {}\n# rejected_steps {}\n# image_force_evaluations {}\n",
               nebobj.accepted_steps, nebobj.rejected_steps, nebobj.image_force_evaluations);
    fmt::print(ofs,"# initial_accepted_steps {}\n# pinned_stage {}\n# protected_image {}\n",options.accepted_step_offset,nebobj.found_state_site,nebobj.iclimb_);
    auto indices=[](const std::set<int> &values) {
        std::string text;for(int value:values){if(!text.empty())text+=',';text+=std::to_string(value);}return text.empty()?std::string("none"):text;
    };
    fmt::print(ofs,"# climbing_images {}\n# valley_images {}\n",indices(nebobj.iclimbs),indices(nebobj.ivallys));
    fmt::print(ofs, "# last_atomic_move_A {:.17g}\n# last_cell_change {:.17g}\n# last_fallback {}\n",
               nebobj.last_atomic_move, nebobj.last_cell_change, nebobj.last_fallback);
    std::ofstream cells(params.summary + ".cells.csv");
    if (!cells) GPERROR("Cannot open NEB cell diagnostics");
    nebobj.write_diagnostics(cells);
    cells.close();
    if (!cells) GPERROR("Failed to write NEB cell diagnostics");
    if (nebobj.vcneb) {
        fmt::print(ofs, "# vcneb true\n# pressure_GPa {:.17g}\n# cell_weight {:.17g}\n", nebobj.pressure, nebobj.cell_weight);
        fmt::print(ofs, "# symm_strain {}\n# recast_atomic_force {}\n# apply_scal {}\n# reference_image {}\n", nebobj.symm_strain, nebobj.recast_atomic_force, nebobj.apply_scal, nebobj.iref);
        fmt::print(ofs, "# joint_metric vcneb-frobenius-v1\n");
        fmt::print(ofs, "# restore_F_cholesky {}\n", nebobj.restore_F_cholesky);
        for (int j = 0; j < nebobj.vcoffset; ++j) fmt::print(ofs, "# cell_scale {} {:.17g}\n", j, nebobj.scalval[j]);
        for (int i = 0; i < static_cast<int>(nebobj.images.size()); ++i) {
            const double volume = nebobj.images[i].get_volume();
            const double pv = nebobj.pressure / eVA3_to_GPa * volume;
            fmt::print(ofs, "# image_energy {} {:.17g} {:.17g} {:.17g} {:.17g}\n", i, nebobj.images[i].energy_-pv, nebobj.images[i].energy_, pv, volume);
        }
    }
    fmt::print(ofs, "# converged {}\n", nebobj.converged ? "true" : "false");
    fmt::print(ofs, "# optimizer_iterations {}\n", nebobj.optimizer_iterations);
    fmt::print(ofs, "# force_evaluations {}\n", nebobj.force_evaluations);
    fmt::print(ofs, "# max_neb_force {:.12g}\n", nebobj.max_neb_force);
    fmt::print(ofs, "# max_perp_force {:.12g}\n", nebobj.max_perp_force);
    fmt::print(ofs, "# max_spring_force {:.12g}\n", nebobj.max_spring_force);
    fmt::print(ofs, "# climbing_image {}\n", nebobj.climbing_image);
    fmt::print(ofs, "# {:<10} {:<15} {:<15} {:<15} {:<15} {:<15}\n", "image", "IRC", "enthalpy", "ent_diff", "g_proj", "volume");
    fmt::print(ofs, " {:<10d} {:< 15f} {:< 15f} {:< 15f} {:< 15f} {:< 15f}\n", 
                    0, 
                    d, 
                    nebobj.images[0].energy_, 
                    0.,
                    g,
                    nebobj.images[0].get_volume());
    double eng0 = nebobj.images[0].energy_;
    for (int i = 1; i < nebobj.images.size()-1; ++i) {
        d += nebobj.distance(nebobj.images[i-1], nebobj.images[i], 0.);
        ae_x[i] = d;
        ae_y[i] = nebobj.images[i].energy_ - eng0;
        g = nebobj.tagent_g(i-1, i, i+1);
        ae_g[i] = g;
        fmt::print(ofs, " {:<10d} {:< 15f} {:< 15f} {:< 15f} {:< 15f} {:< 15f}\n", 
                    i, 
                    d, 
                    nebobj.images[i].energy_, nebobj.images[i].energy_ - eng0,
                    g,
                    nebobj.images[i].get_volume());
    }
    d += nebobj.distance(nebobj.images[nimg - 2], nebobj.images.back(), 0.);
    ae_x[nimg-1] = d;
    ae_y[nimg-1] = nebobj.images.back().energy_ - eng0,
    ae_g[nimg-1] = 0;
    fmt::print(ofs, " {:<10d} {:< 15f} {:< 15f} {:< 15f} {:< 15f} {:< 15f}\n", 
                    nebobj.images.size() - 1, 
                    d, 
                    nebobj.images.back().energy_, nebobj.images.back().energy_ - eng0,
                    0.,
                    nebobj.images.back().get_volume());
    ofs.close();
    if (!ofs) GPERROR("Failed to write NEB summary: " + params.summary);

    // build a spline fitting
    Message("Generating spline");
    ofs.open("neb_hermite_spline.dat");
    if (!ofs) GPERROR("Cannot open NEB spline output: neb_hermite_spline.dat");
    fmt::print(ofs, "# IRC ENG\n");
    try {
        alglib::spline1dinterpolant ae_interp;
        alglib::spline1dbuildhermite(ae_x, ae_y, ae_g, ae_interp);
        double delta = d / (params.interp_npoints - 1.);
        for (int i = 0; i < params.interp_npoints; ++i) {
            double c_x = delta * i;
            fmt::print(ofs, "{:< 10.6f} {:< 10.6f}\n", c_x,
                    alglib::spline1dcalc(ae_interp, c_x));
        }
    } catch (alglib::ap_error &e) {
        std::cerr << "Warning: Hermite spline build failed: " << e.msg
                  << ". Skipping spline output.\n";
        fmt::print(ofs, "# Spline build failed: {}\n", e.msg);
    }
    ofs.close();
    if (!ofs) GPERROR("Failed to write NEB spline: neb_hermite_spline.dat");

    // Active learning — sparsify pool and export
    if (al) al->finalize(&iobj);

    if(parallel.profiling()) {stages[2]=NEBReporting::seconds(stamp);stamp=NEBReporting::Clock::now();}
    if(!nebobj.symmetry_release.empty()) {
        parallel.begin_release();
        nebobj.check_symmetry_release(options);
    }
    if(parallel.profiling()) {
        stages[3]=NEBReporting::seconds(stamp);
        parallel.write_profile(params.profile,stages);
    }
    return nebobj.converged ? 0 : 2;
}

int calc_neb(int argc, char *argv[]) {
    // The generic parser exits directly for --version. Handle it collectively
    // before workers enter the request loop.
    for (int i=1;i<argc;++i) if (std::string(argv[i])=="--version" || std::string(argv[i])=="-ver") {
        if (mpi.rank==0) CMDOption::global_cmdopts.showversion(std::cout);
        return 0;
    }
    NEBParallel parallel;
    if (mpi.rank!=0) return parallel.workers();
    int status=1;
    try { status=calc_neb(argc,argv,parallel); }
    catch (const std::exception &error) { std::cerr<<error.what()<<'\n'; }
    catch (...) { std::cerr<<"Unknown NEB error\n"; }
    parallel.finish(status);
    return status;
}
