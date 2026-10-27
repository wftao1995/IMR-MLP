// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "apps/fastfit_errors.h"
#include "apps/fastfit_cost.h"
#include "apps/fastfit_cache.h"
#include "apps/fastfit_utils.h"
#include "train/linear_efs_system.h"
#include "train/train_mtp.h"
#include "train/train_nonlinear_potential.h"
#include "io/gpio.h"
#include "utils/gputils_eigen.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <map>

using Catch::Approx;
using Potentials::PotMTP;
using namespace Eigen;
namespace {
struct Workspace {
    std::filesystem::path original = std::filesystem::current_path(), path;
    std::string model = std::filesystem::absolute("tests/case-mtp-les/les.mtp").string();
    CMDOption::CommandLineOptions saved = CMDOption::global_cmdopts;
    int threads = omp_num_threads;
    Workspace() {
        char name[] = "/tmp/imrmlp_fastfit_XXXXXX";
        const char *dir = mkdtemp(name);
        if (!dir) throw std::runtime_error("Cannot create fastfit test directory");
        path = dir;
        std::filesystem::current_path(path);
        omp_num_threads = 1;
    }
    ~Workspace() {
        CMDOption::global_cmdopts = saved;
        omp_num_threads = threads;
        SetStreamForOutput(&std::cout);
        std::filesystem::current_path(original);
        std::filesystem::remove_all(path);
    }
};
Atoms frame(PotMTP &p, int k = 0) {
    Atoms a;
    a.cell_ = 12. * Matrix3d::Identity();
    a.pos_ = {{2.,2.,2.}, {4.1+.13*std::sin(k),2.2,2.},
              {2.3,4.2+.21*std::cos(k),2.4}, {5.,4.8,3.1}};
    a.chemmap_ = {0,1,0,1};
    if (k % 2) { a.pos_.pop_back(); a.chemmap_.pop_back(); }
    p.getEFS(a);
    return a;
}
void equal_efs(const Atoms &a, const Atoms &b) {
    REQUIRE(a.energy_ == Approx(b.energy_).epsilon(1e-11).margin(1e-12));
    REQUIRE((a.stress_-b.stress_).norm() < 1e-10);
    for (int i=0;i<a.size();++i) REQUIRE((a.forces_[i]-b.forces_[i]).norm() < 1e-10);
}
std::string contents(const std::string &name) {
    std::ifstream file(name);
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
int run(std::vector<std::string> args) {
    CMDOption::global_cmdopts = CMDOption::CommandLineOptions();
    std::vector<char*> argv;
    for (auto &arg:args) argv.push_back(arg.data());
    return fastfit(argv.size(), argv.data());
}
std::string captured_run(const std::vector<std::string> &args) {
    std::ostringstream output;
    auto *previous = std::cout.rdbuf(output.rdbuf());
    try {
        const int result = run(args);
        std::cout.rdbuf(previous);
        REQUIRE(result == 0);
    } catch (...) {
        std::cout.rdbuf(previous);
        throw;
    }
    return output.str();
}
struct ProfileRow {
    double root, mean, maximum, calls_root, calls_sum;
};
std::map<std::string, ProfileRow> read_profile(const std::string &filename) {
    std::istringstream input(contents(filename));
    std::string line;
    std::map<std::string, ProfileRow> rows;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#' || line.starts_with("stage\t")) continue;
        std::istringstream fields(line);
        std::string stage;
        ProfileRow row{};
        REQUIRE(bool(fields >> stage >> row.root >> row.mean >> row.maximum >> row.calls_root >> row.calls_sum));
        REQUIRE(row.root >= 0);
        REQUIRE(row.root == row.mean);
        REQUIRE(row.root == row.maximum);
        REQUIRE(row.calls_root == row.calls_sum);
        rows.emplace(stage, row);
    }
    REQUIRE(rows.size() == 16);
    double exclusive = 0;
    for (const auto &[stage, row] : rows) if (stage != "total") exclusive += row.root;
    REQUIRE(exclusive == Approx(rows.at("total").root).epsilon(1e-8).margin(1e-10));
    REQUIRE(rows.at("input").root > 0);
    REQUIRE(rows.at("communication").root == 0);
    REQUIRE(rows.at("communication").calls_root == 0);
    REQUIRE(contents(filename).find("# status\tok\n") != std::string::npos);
#if defined(__linux__) || defined(__APPLE__)
    const std::string marker = "# rss_peak_rank0_mib\t";
    const auto report = contents(filename);
    const auto pos = report.find(marker);
    REQUIRE(pos != std::string::npos);
    REQUIRE(std::stod(report.substr(pos + marker.size())) > 0);
#endif
    return rows;
}
}

TEST_CASE("MTP copies own their radial bases and survive repeated searches", "[fastfit][mtp]") {
    Workspace work;
    PotMTP original(work.model);
    auto expected = frame(original);
    auto *basis = original.p_radical_basis;
    for (int i=0;i<3;++i) {
        PotMTP copy(original);
        REQUIRE(original.p_radical_basis == basis);
        REQUIRE(copy.p_radical_basis != basis);
        equal_efs(frame(copy), expected);
        copy.radial_coeffs[0] += 10;
        equal_efs(frame(original), expected);
    }
    equal_efs(frame(original), expected);
}

TEST_CASE("MTP compaction preserves EFS and radial channel ordering", "[fastfit][mtp]") {
    Workspace work;
    const std::vector<std::set<int>> keep_sets{{2}, {0,2}, {7,12,28}, {}};
    for (const auto &keep : keep_sets) {
        PotMTP p(work.model);
        std::set<int> removed;
        for (int i=0;i<p.alpha_scalar_moments;++i) if (!keep.count(i)) removed.insert(i);
        p.remove_scalar_by_id(removed);
        auto expected = frame(p);
        p.omp_init(); // compaction must invalidate already allocated buffers too
        p.remove_orphans();
        equal_efs(frame(p), expected);
        p.Save("reduced.mtp");
        PotMTP reloaded("reduced.mtp");
        REQUIRE(reloaded.alpha_scalar_moments == keep.size());
        equal_efs(frame(reloaded), expected);
        auto parallel = expected;
        p.getEFSOMPthread(parallel);
        equal_efs(parallel, expected);
        p.remove_orphans();
        equal_efs(frame(p), expected);
    }
}

TEST_CASE("Cached EFS loss agrees with direct weighted predictions", "[fastfit][training]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<Atoms> data;
    for (int k=0;k<8;++k) {
        auto a = frame(p,k);
        a.energy_ += .02*(k+1);
        a.stress_(0,1) += .04; // asymmetric labels still use all nine components
        a.forces_[0][1] += .03;
        a.fitness = .4+k;
        a.atomic_fitness.assign(a.size(), .5+k);
        a.atomic_fitness[0] = 0;
        if (k==1) a.have_energy = false;
        if (k==2) a.forces_.clear();
        if (k==3) a.have_stress = false;
        if (k==4) a.fitness = 0;
        data.push_back(a);
    }
    EFSFitWeights w;
    w.energy=2.3; w.force=.17; w.stress=.031;
    w.energy_power=1.3; w.force_power=.2; w.stress_power=.7;
    w.norm_f=.8; w.norm_d=.3; w.norm_p=1.4;
    auto unit = w; unit.energy=unit.force=unit.stress=1;
    Training::LinearEFSSystem system(p,data,unit);
    Training::LinearEFSSystem fixed_species(p,data,unit,p.species_count);
    Training::LinearEFSSystem square_root(p,data,unit,0,true);
    square_root.reduce();
    const VectorXd anchor = Map<VectorXd>(p.linear_coeffs,p.lin_coeff_size);
    for (int trial=0;trial<5;++trial) {
        VectorXd x=anchor;
        for (int i=p.species_count;i<x.size();++i) x[i]+=.1*trial*std::sin(i+1);
        p.set_linear_coeff(x.data());
        func_fastfit::ErrorsReport report(w);
        for (auto &ref:data) { auto pred=ref; p.getEFS(pred); report.add(pred,ref); }
        REQUIRE(system.loss(x,w.energy,w.force,w.stress) == Approx(report.loss).epsilon(1e-10).margin(1e-12));
        REQUIRE(square_root.loss(x,w.energy,w.force,w.stress) == Approx(report.loss).epsilon(1e-10).margin(1e-12));
        REQUIRE(fixed_species.loss(x.tail(x.size()-p.species_count),w.energy,w.force,w.stress)
                == Approx(report.loss).epsilon(1e-10).margin(1e-12));
    }
    const MatrixXd gram=system.matrix(w.energy,w.force,w.stress);
    REQUIRE(gram.isApprox(gram.transpose(),1e-14));
    // Independently verify the normal-equation derivative against direct EFS.
    const VectorXd x=Map<VectorXd>(p.linear_coeffs,p.lin_coeff_size);
    const VectorXd gradient=2*(gram*x-system.rhs(w.energy,w.force,w.stress));
    for (int i : {0,2,4}) {
        auto direct = [&](double step) {
            VectorXd perturbed=x; perturbed[i]+=step; p.set_linear_coeff(perturbed.data());
            func_fastfit::ErrorsReport report(w);
            for (auto &ref:data) { auto pred=ref; p.getEFS(pred); report.add(pred,ref); }
            return report.loss;
        };
        REQUIRE((direct(1e-5)-direct(-1e-5))/2e-5 == Approx(gradient[i]).epsilon(1e-6).margin(1e-9));
    }
}

TEST_CASE("ATA export is unweighted symmetric and repeatable in every path", "[fastfit][training]") {
    Workspace work;
    const bool fast = GENERATE(false,true);
    const bool energy_only = GENERATE(false,true);
    PotMTP p(work.model); p.train_init(); p.set_inited();
    std::vector<Atoms> data{frame(p),frame(p,1)};
    data[0].atomic_fitness={0,1,.5,2}; data[1].fitness=0;
    std::vector<Neighborhoods> nbhs;
    for (auto &a:data) nbhs.emplace_back(a,p.cutoff);
    GPSettings settings;
    settings.Modify("skip_preinit","true"); settings.Modify("do_orthogonalization","false");
    settings.Modify("lfit_fastalgo",fast?"true":"false");
    settings.Modify("lfit_eng_only",energy_only?"true":"false");
    settings.Modify("energy_weight","2.3"); settings.Modify("force_weight",".17");
    settings.Modify("stress_weight",".031");
    Trainer_MTP trainer(settings); trainer.init(data,nbhs,&p);
    EFSFitWeights unit; unit.energy=1; unit.force=unit.stress=energy_only?0:1;
    Training::LinearEFSSystem expected(p,data,unit);
    for (int repeat=0;repeat<2;++repeat) {
        trainer.get_ATA();
        int c=0;
        for (const auto *suffix: {"E","F","S"}) {
            MatrixXd a; VectorXd b;
            readMatrixFromBinFile(std::string("MATATA_")+suffix,a);
            readVectorFromBinFile(std::string("VECATA_")+suffix,b);
            REQUIRE(a.isApprox(expected.channels[c].gram,1e-12));
            REQUIRE(a.isApprox(a.transpose(),1e-14));
            const VectorXd rhs=expected.channels[c].residual_rhs+expected.channels[c].gram*expected.anchor;
            REQUIRE(b.isApprox(rhs,1e-12));
            if (c==0 || !energy_only) REQUIRE(a.norm()>0);
            ++c;
        }
    }
}

TEST_CASE("Fastfit CLI fitting and search modes work without exported matrices", "[fastfit][apps]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<Atoms> data{frame(p),frame(p,1),frame(p,2)};
    GPIO io; io.setmap(); REQUIRE(io.write(data,"train.cfg","mlpcfg"));
    p.Save("potential.mtp");
    std::ofstream("fastfit.ini") << "mode 0\nweight-l2 0\n";
    // CLI overrides both the mode and the singular-system regularizer in config.
    REQUIRE(run({"fastfit","fastfit.ini","-m","1","--weight-energy","0","--weight-stress","0","--weight-l2",".1"})==0);
    REQUIRE(std::filesystem::exists("__Trained__.mtp"));
    for (int mode : {3,5,6}) {
        func_fastfit::detail::WriteScalarIds("selected_ones.dat",{2});
        std::set<int> removed;
        for (int i=0;i<p.alpha_scalar_moments;++i) if(i!=2) removed.insert(i);
        func_fastfit::detail::WriteScalarIds("removed.dat",removed);
        REQUIRE(run({"fastfit","fastfit.ini","-m",std::to_string(mode),"--weight-l2",".1",
                     "--search-steps","4","--population","4","--generations","2","--seed","17"})==0);
    }
}

TEST_CASE("Mean-moments fastfit and nonlinear training share the objective and L2", "[fastfit][training][objective]") {
    Workspace work;
    PotMTP parent(work.model); parent.set_inited();
    std::vector<Atoms> data;
    for (int k=0;k<8;++k) {
        auto a=frame(parent,k); a.energy_+=.015*std::cos(k); a.forces_[0][0]+=.02;
        a.stress_(0,1)+=.03; a.stress_(1,0)+=.03; data.push_back(a);
    }
    GPIO io; io.setmap(); REQUIRE(io.write(data,"train.cfg","mlpcfg"));
    // Both paths must consume the same serialized labels (CFG stores a symmetric virial).
    data.clear(); REQUIRE(io.read(data,"train.cfg","mlpcfg"));
    parent.Save("potential.mtp");
    std::ofstream("fit.ini") << "objective-convention mean-moments\nweight-l2 .01\n"
        << "weight-energy 1\nweight-force .1\nweight-stress .001\nmethod svd\n";
    REQUIRE(run({"fastfit","fit.ini","-m","1","--objective-report","objective.tsv"})==0);
    REQUIRE(contents("objective.tsv").find("l2_scope\tmoments\n")!=std::string::npos);
    PotMTP fitted("__Trained__.mtp"); fitted.set_inited();
    std::vector<Neighborhoods> nbhs;
    for (auto& a:data) nbhs.emplace_back(a,fitted.cutoff);
    GPSettings settings;
    settings.Modify("objective-convention","mean-moments");
    settings.Modify("weight_L2",".01"); settings.Modify("penalty_weight","0");
    settings.Modify("energy_weight","1"); settings.Modify("force_weight",".1"); settings.Modify("stress_weight",".001");
    TrainNonlinear trainer(settings); trainer.init(data,nbhs,&fitted);
    double f; VectorXd g(fitted.get_NL_coeffs_count()); trainer.AccumulateGard(f,g);
    func_fastfit::ErrorsReport errors(1.,.1,.001);
    for (auto& ref:data) {auto a=ref;fitted.getEFS(a);errors.add(a,ref);}
    double penalty=.01*Map<VectorXd>(fitted.moment_coeffs,fitted.moment_coeff_size).squaredNorm();
    REQUIRE(f==Approx(errors.loss/data.size()+penalty).epsilon(1e-9).margin(1e-10));
    REQUIRE(g.tail(fitted.lin_coeff_size).lpNorm<Infinity>()<1e-8);
    // Species bias has no L2 gradient, including in an energy-free objective.
    settings.Modify("energy_weight","0");
    TrainNonlinear force_only(settings);force_only.init(data,nbhs,&fitted);force_only.AccumulateGard(f,g);
    REQUIRE(g.segment(fitted.radial_coeff_size,fitted.species_count).norm()<1e-14);
    // A sum-referenced lambda is converted using the actual global frame count.
    REQUIRE(run({"fastfit","fit.ini","-m","1","--l2-reference","sum","--weight-l2",".08"})==0);
    PotMTP converted("__Trained__.mtp");
    for (auto ref:data) {auto a=ref,b=ref;fitted.getEFS(a);converted.getEFS(b);equal_efs(a,b);}
    // Duplicating data leaves the mean objective and its optimum unchanged.
    auto doubled=data;doubled.insert(doubled.end(),data.begin(),data.end());
    REQUIRE(io.write(doubled,"train.cfg","mlpcfg"));
    REQUIRE(run({"fastfit","fit.ini","-m","1"})==0);
    PotMTP duplicate("__Trained__.mtp");
    for (auto ref:data) {auto a=ref,b=ref;fitted.getEFS(a);duplicate.getEFS(b);equal_efs(a,b);}
    settings.Modify("l2-reference","sum");settings.Modify("weight_L2",".08");settings.Modify("energy_weight","1");
    TrainNonlinear from_sum(settings);from_sum.init(data,nbhs,&fitted);from_sum.AccumulateGard(f,g);
    REQUIRE(f==Approx(errors.loss/data.size()+penalty).epsilon(1e-9).margin(1e-10));
    settings.Modify("penalty_weight","1e-6");
    TrainNonlinear extra_penalty(settings);REQUIRE_THROWS(extra_penalty.init(data,nbhs,&fitted));
    REQUIRE_THROWS(run({"fastfit","fit.ini","-m","1","--objective-convention","typo"}));
}

TEST_CASE("Fastfit profiling preserves results and accounts for every mode", "[fastfit][profile]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<Atoms> data{frame(p), frame(p,1), frame(p,2)};
    GPIO io; io.setmap(); REQUIRE(io.write(data,"train.cfg","mlpcfg"));
    REQUIRE(io.write(data,"valid.cfg","mlpcfg"));
    p.Save("potential.mtp");
    std::ofstream("fastfit.ini") << "mode 0\nweight-l2 .1\nprofile false\n";
    const std::vector<std::vector<std::string>> artifacts{
        {"EIGVEC", "EIGVAL", "EIGENSYS"}, {"__Trained__.mtp"},
        {"__REDUCED__.mtp"}, {}, {"selected_ones.dat", "removed.dat"},
        {"__TEMPERED__.mtp"}, {"__NSGA2__.mtp", "selected_ones_nsga2.dat"},
        {"__DECORR__.mtp", "selected_ones.dat", "removed.dat"}
    };
    auto reset_selection = [&] {
        func_fastfit::detail::WriteScalarIds("selected_ones.dat", {2});
        std::set<int> removed;
        for (int i=0; i<p.alpha_scalar_moments; ++i) if (i!=2) removed.insert(i);
        func_fastfit::detail::WriteScalarIds("removed.dat", removed);
    };
    for (int mode=0; mode<=7; ++mode) {
        INFO("mode=" << mode);
        std::vector<std::string> args{"fastfit", "fastfit.ini", "-m", std::to_string(mode),
            "--search-steps", "4", "--population", "4", "--generations", "2", "--seed", "17",
            "--corr-strategy", "2", "--corr-max-features", "2", "--corr-save-reduced"};
        reset_selection();
        const auto baseline = captured_run(args);
        REQUIRE(baseline.find("# fastfit_profile_version") == std::string::npos);
        std::vector<std::string> expected;
        for (const auto &file : artifacts[mode]) expected.push_back(contents(file));
        reset_selection();
        args.insert(args.end(), {"--profile-output", "profile.tsv"}); // implies --profile
        const auto profiled = captured_run(args);
        REQUIRE(profiled.starts_with(baseline));
        REQUIRE(profiled.substr(baseline.size()) == contents("profile.tsv"));
        for (size_t i=0; i<expected.size(); ++i) REQUIRE(contents(artifacts[mode][i]) == expected[i]);
        const auto rows = read_profile("profile.tsv");
        REQUIRE(rows.at("assembly").root > 0);
        if (mode != 3) REQUIRE(rows.at("output").root > 0);
        if (mode==1 || mode==2 || mode==3 || mode==5 || mode==6 || mode==7)
            REQUIRE(rows.at("solve").root > 0);
        if (mode==5 || mode==6) {
            REQUIRE(rows.at("cached_score").root > 0);
            REQUIRE(rows.at("final_efs").root > 0);
        }
        if (mode==7) {
            REQUIRE(rows.at("descriptors").root > 0);
            REQUIRE(rows.at("correlation").root > 0);
        }
    }
    // The flag overrides profile=false; descriptor-only mode does no EFS assembly.
    const auto flag_output = captured_run({"fastfit", "fastfit.ini", "-m", "7", "--profile"});
    REQUIRE(flag_output.find("# fastfit_profile_version") != std::string::npos);
    REQUIRE(flag_output.find("assembly\t0\t0\t0\t0\t0\n") != std::string::npos);
    REQUIRE(flag_output.find("direct_efs\t0\t0\t0\t0\t0\n") != std::string::npos);
}

TEST_CASE("NSGA-II search controls preserve defaults and reproduce sparse searches", "[fastfit][apps]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<Atoms> data{frame(p), frame(p,1), frame(p,2)};
    GPIO io; io.setmap(); REQUIRE(io.write(data,"train.cfg","mlpcfg"));
    REQUIRE(io.write(data,"valid.cfg","mlpcfg"));
    p.Save("potential.mtp");
    std::ofstream("fastfit.ini") << "weight-l2 .1\n";
    std::vector<std::string> args{"fastfit","fastfit.ini","-m","6",
        "--population","8","--generations","3","--seed","17"};
    captured_run(args);
    const auto defaults = contents("__NSGA2__.mtp");
    const auto default_ids = contents("selected_ones_nsga2.dat");
    auto explicit_defaults = args;
    explicit_defaults.insert(explicit_defaults.end(), {"--mutation-rate",".05","--population-init","half","--group-mutation-rate","0"});
    captured_run(explicit_defaults);
    REQUIRE(contents("__NSGA2__.mtp") == defaults);
    REQUIRE(contents("selected_ones_nsga2.dat") == default_ids);
    args.insert(args.end(), {"--mutation-rate","-1","--population-init","stratified"});
    const auto log = captured_run(args);
    REQUIRE(log.find("expected_flips=1") != std::string::npos);
    const auto sparse = contents("__NSGA2__.mtp");
    const auto sparse_ids = contents("selected_ones_nsga2.dat");
    captured_run(args);
    REQUIRE(contents("__NSGA2__.mtp") == sparse);
    REQUIRE(contents("selected_ones_nsga2.dat") == sparse_ids);
    PotMTP result("__NSGA2__.mtp");
    REQUIRE(result.alpha_scalar_moments > 0);
    REQUIRE(result.alpha_scalar_moments <= p.alpha_scalar_moments);
    auto dependency_args = explicit_defaults;
    dependency_args[dependency_args.size()-3] = "dependency";
    dependency_args.back() = "1";
    const auto dependency_log = captured_run(dependency_args);
    REQUIRE(dependency_log.find("dependency_groups=") != std::string::npos);
    const auto dependency_model = contents("__NSGA2__.mtp");
    const auto dependency_ids = contents("selected_ones_nsga2.dat");
    captured_run(dependency_args);
    REQUIRE(contents("__NSGA2__.mtp") == dependency_model);
    REQUIRE(contents("selected_ones_nsga2.dat") == dependency_ids);
    auto explicit_basic = dependency_args;
    explicit_basic.insert(explicit_basic.end(), {"--group-mutation-kind", "basic"});
    captured_run(explicit_basic);
    REQUIRE(contents("__NSGA2__.mtp") == dependency_model);
    REQUIRE(contents("selected_ones_nsga2.dat") == dependency_ids);
    for (const auto &kind : {"radial", "mixed"}) {
        auto group_args = dependency_args;
        group_args.insert(group_args.end(), {"--group-mutation-kind", kind});
        const auto group_log = captured_run(group_args);
        REQUIRE(group_log.find(std::string("group_mutation_kind=") + kind) != std::string::npos);
        const auto model = contents("__NSGA2__.mtp"), ids = contents("selected_ones_nsga2.dat");
        captured_run(group_args);
        REQUIRE(contents("__NSGA2__.mtp") == model);
        REQUIRE(contents("selected_ones_nsga2.dat") == ids);
        PotMTP loaded("__NSGA2__.mtp");
        REQUIRE(loaded.alpha_scalar_moments > 0);
        Atoms probe = frame(p); loaded.getEFS(probe);
        REQUIRE(std::isfinite(probe.energy_));
    }
    for (const auto &rate : {"-0.1", "1.1", "nan", "inf"})
        REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","6","--mutation-rate",rate}));
    for (const auto &rate : {"-1", "1.1", "nan", "inf"})
        REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","6","--group-mutation-rate",rate}));
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","6","--population-init","unknown"}));
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","6","--group-mutation-kind","unknown"}));
}

TEST_CASE("Radial dependency groups eliminate complete channels", "[fastfit][cost]") {
    Workspace work;
    PotMTP p(work.model);
    const auto groups = func_fastfit::cost::radial_dependency_groups(p);
    REQUIRE_FALSE(groups.empty());
    REQUIRE(std::is_sorted(groups.begin(), groups.end()));
    REQUIRE(std::adjacent_find(groups.begin(), groups.end()) == groups.end());
    std::vector<int> all(p.alpha_scalar_moments); std::iota(all.begin(), all.end(), 0);
    const auto full = p.analyze_basis(all);
    for (const auto &group : groups) {
        std::vector<int> keep;
        std::set_difference(all.begin(), all.end(), group.begin(), group.end(), std::back_inserter(keep));
        REQUIRE_FALSE(keep.empty());
        const auto reduced = p.analyze_basis(keep);
        std::set<int> channels_before, channels_after;
        for (int b = 0; b < p.alpha_index_basic_count; ++b) {
            if (full.keep[b]) channels_before.insert(p.alpha_index_basic(b, 0));
            if (reduced.keep[b]) channels_after.insert(p.alpha_index_basic(b, 0));
        }
        REQUIRE(channels_after.size() < channels_before.size());
        REQUIRE(reduced.stats.efs_work(8, 10, p.rb_size) < full.stats.efs_work(8, 10, p.rb_size));
        // Every removed output uses at least one of the eliminated channels.
        for (int scalar : group) {
            const auto closure = p.analyze_basis({scalar});
            bool uses_removed = false;
            for (int b = 0; b < p.alpha_index_basic_count; ++b)
                uses_removed |= closure.keep[b] && !channels_after.contains(p.alpha_index_basic(b, 0));
            REQUIRE(uses_removed);
        }
        PotMTP compact(p);
        std::set<int> removed(group.begin(), group.end());
        compact.remove_scalar_by_id(removed);
        compact.remove_orphans();
        compact.Save("channel-compact.mtp");
        PotMTP loaded("channel-compact.mtp");
        REQUIRE(loaded.radial_func_count == int(channels_after.size()));
    }
}

TEST_CASE("Batch EFS assembly cache checks inputs and preserves fits", "[fastfit][training]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<Atoms> data{frame(p),frame(p,1),frame(p,2)};
    GPIO io; io.setmap(); REQUIRE(io.write(data,"train.cfg","mlpcfg"));
    REQUIRE(io.write(data,"valid.cfg","mlpcfg")); p.Save("potential.mtp");
    std::ofstream("fastfit.ini") << "weight-l2 .1\n";
    EFSFitWeights weights;
    const auto original_key = func_fastfit::assembly_key(contents("potential.mtp"),data,weights,0,1);
    auto changed = [&](auto mutate) {
        auto altered=data; mutate(altered[0]);
        REQUIRE(func_fastfit::assembly_key(contents("potential.mtp"),altered,weights,0,1) != original_key);
    };
    changed([](Atoms &a){a.pos_[0][0]+=1e-9;});
    changed([](Atoms &a){a.cell_(0,0)+=1e-9;});
    changed([](Atoms &a){a.pbc_[0]=false;});
    changed([](Atoms &a){a.chemmap_[0]=1;});
    changed([](Atoms &a){a.energy_+=1e-9;});
    changed([](Atoms &a){a.have_energy=false;});
    changed([](Atoms &a){a.forces_[0][1]+=1e-9;});
    changed([](Atoms &a){a.stress_(0,1)+=1e-9;});
    changed([](Atoms &a){a.have_stress=false;});
    changed([](Atoms &a){a.fitness=.2;});
    changed([](Atoms &a){a.atomic_fitness.assign(a.size(),.5);});
    auto normalization=weights; normalization.norm_f=.4;
    REQUIRE(func_fastfit::assembly_key(contents("potential.mtp"),data,normalization,0,1) != original_key);
    REQUIRE(func_fastfit::assembly_key(contents("potential.mtp")+"\n",data,weights,0,1) != original_key);
    REQUIRE(func_fastfit::assembly_key(contents("potential.mtp"),data,weights,0,2) != original_key);
    auto ghosts=data; ghosts[0].have_ghost=true;
    REQUIRE_THROWS(func_fastfit::assembly_key("model",ghosts,weights,0,1));

    func_fastfit::AssemblyCache cache;
    std::vector<std::string> args{"fastfit","fastfit.ini","-m","6","--population","4","--generations","2","--seed","17"};
    auto cached = [&](bool only=false) {
        CMDOption::global_cmdopts=CMDOption::CommandLineOptions();
        std::vector<char*> argv; for(auto &arg:args) argv.push_back(arg.data());
        REQUIRE(func_fastfit::fastfit_cached(argv.size(),argv.data(),cache,only)==0);
    };
    cached(true); REQUIRE_FALSE(std::filesystem::exists("__NSGA2__.mtp"));
    cached(); const auto first=contents("__NSGA2__.mtp");
    func_fastfit::BenchmarkObserver observer;
    int systems_seen=0, candidates_seen=0;
    observer.systems=[&](const auto&,auto training,auto validation,const auto &H,const auto &b,int nt,int nv) {
        REQUIRE(training); REQUIRE(validation); REQUIRE(nt==3); REQUIRE(nv==3);
        REQUIRE(H.rows()==b.size()); ++systems_seen;
    };
    observer.candidate=[&](const auto &ids,double train,double valid,bool) {
        REQUIRE_FALSE(ids.empty()); REQUIRE(std::isfinite(train)); REQUIRE(std::isfinite(valid));
        ++candidates_seen;
    };
    CMDOption::global_cmdopts=CMDOption::CommandLineOptions();
    std::vector<char*> observed_argv;for(auto &arg:args) observed_argv.push_back(arg.data());
    REQUIRE(func_fastfit::fastfit_observed(observed_argv.size(),observed_argv.data(),cache,observer)==0);
    REQUIRE(systems_seen==1); REQUIRE(candidates_seen>=12);
    REQUIRE(contents("__NSGA2__.mtp")==first);
    observer.candidate=[](const auto&,double,double,bool) { throw std::runtime_error("observer stop"); };
    CMDOption::global_cmdopts=CMDOption::CommandLineOptions();
    REQUIRE_THROWS_WITH(func_fastfit::fastfit_observed(observed_argv.size(),observed_argv.data(),cache,observer),Catch::Matchers::ContainsSubstring("observer stop"));
    cached(); REQUIRE(contents("__NSGA2__.mtp")==first);
    captured_run(args); REQUIRE(contents("__NSGA2__.mtp")==first);
    // Changed labels at the same path must rebuild the systems.
    data[0].energy_+=.4; REQUIRE(io.write(data,"train.cfg","mlpcfg"));
    cached(); const auto modified=contents("__NSGA2__.mtp");
    REQUIRE(modified!=first);
    captured_run(args); REQUIRE(contents("__NSGA2__.mtp")==modified);
    args.insert(args.end(),{"--force-norm-f",".4"});
    cached(); const auto normalized=contents("__NSGA2__.mtp");
    captured_run(args); REQUIRE(contents("__NSGA2__.mtp")==normalized);
    args.insert(args.end(),{"--weight-force",".8","--weight-l2",".2"});
    cached(); const auto reweighted=contents("__NSGA2__.mtp");
    captured_run(args); REQUIRE(contents("__NSGA2__.mtp")==reweighted);
}

TEST_CASE("Fastfit profiling reports failures without swallowing them", "[fastfit][profile]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<Atoms> data{frame(p)};
    GPIO io; io.setmap(); REQUIRE(io.write(data,"train.cfg","mlpcfg"));
    p.Save("potential.mtp");
    std::ofstream("fastfit.ini") << "weight-l2 .1\nprofile true\n";
    SECTION("root-only eigenfile read failure") {
        REQUIRE_THROWS(run({"fastfit", "fastfit.ini", "-m", "4", "--profile-output", "failed.tsv"}));
        REQUIRE(contents("failed.tsv").find("# status\tfailed\n") != std::string::npos);
    }
    SECTION("unwritable report") {
        REQUIRE_THROWS_WITH(run({"fastfit", "fastfit.ini", "-m", "1", "--profile-output", "missing/profile.tsv"}),
                            "Cannot write fastfit profile: missing/profile.tsv");
    }
}

TEST_CASE("Binary matrices handle layout and reject malformed payloads", "[fastfit][io]") {
    Workspace work;
    Matrix<double,-1,-1,RowMajor> row(2,3); row<<1,2,3,4,5,6;
    MatrixXd col=row, read;
    writeMatrixToBinFile("row.bin",row); readMatrixFromBinFile("row.bin",read);
    REQUIRE(read==col);
    writeMatrixToBinFile("col.bin",col); readMatrixFromBinFile("col.bin",row);
    REQUIRE(row==col);
    auto bytes=contents("row.bin");
    bytes.pop_back();
    std::ofstream("bad.bin",std::ios::binary).write(bytes.data(),bytes.size());
    REQUIRE_THROWS(readMatrixFromBinFile("bad.bin",read));
    REQUIRE(read==col);
    bytes[8]=2;
    std::ofstream("bad.bin",std::ios::binary).write(bytes.data(),bytes.size());
    REQUIRE_THROWS(readMatrixFromBinFile("bad.bin",read));
    int negative=-1;
    std::ofstream("bad-vector.bin",std::ios::binary).write(reinterpret_cast<char*>(&negative),sizeof(negative));
    VectorXd vector;
    REQUIRE_THROWS(readVectorFromBinFile("bad-vector.bin",vector));
}

TEST_CASE("Basis cost follows the union of retained computation paths", "[fastfit][cost]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<std::vector<int>> subsets{{}, {0}, {2}, {0,2}, {7,12,28}};
    std::vector<int> all(p.alpha_scalar_moments); std::iota(all.begin(),all.end(),0);
    subsets.push_back(all);
    for (const auto &ids : subsets) {
        const auto analysis = p.analyze_basis(ids);
        // Independent fixed-point dependency closure (no reverse-order assumption).
        std::set<int> needed;
        for (int id : ids) needed.insert(p.alpha_moment_mapping[id]);
        if (ids.empty()) needed.insert(0);
        bool changed = true;
        while (changed) {
            changed = false;
            for (int i=0; i<p.alpha_index_times_count; ++i) {
                if (!needed.count(p.alpha_index_times(i,3))) continue;
                changed |= needed.insert(p.alpha_index_times(i,0)).second;
                changed |= needed.insert(p.alpha_index_times(i,1)).second;
            }
        }
        int products = 0;
        for (int i=0; i<p.alpha_index_times_count; ++i) if (needed.count(p.alpha_index_times(i,3))) ++products;
        REQUIRE(analysis.stats.moments == needed.size());
        REQUIRE(analysis.stats.products == products);
        PotMTP reduced(p);
        std::set<int> removed;
        for (int i : all) if (!std::binary_search(ids.begin(),ids.end(),i)) removed.insert(i);
        reduced.remove_scalar_by_id(removed); reduced.remove_orphans();
        reduced.Save("compact.mtp"); PotMTP loaded("compact.mtp");
        std::vector<int> compact_ids(ids.size()); std::iota(compact_ids.begin(),compact_ids.end(),0);
        REQUIRE(loaded.analyze_basis(compact_ids).stats == analysis.stats);
        REQUIRE(analysis.stats.radial_bytes == loaded.radial_coeff_size*sizeof(double));
        REQUIRE(analysis.stats.efs_work(8,10,p.rb_size) <= p.analyze_basis(all).stats.efs_work(8,10,p.rb_size));
    }
    const auto one = p.analyze_basis({0}).stats, other = p.analyze_basis({28}).stats;
    REQUIRE(one.scalars == other.scalars);
    REQUIRE(one.efs_work(8,10,p.rb_size) != other.efs_work(8,10,p.rb_size));
    bool shared = false, multiple_rules = false;
    for (int i=0; i<p.alpha_scalar_moments; ++i) for (int j=i+1; j<p.alpha_scalar_moments; ++j) {
        auto a=p.analyze_basis({i}), b=p.analyze_basis({j}), both=p.analyze_basis({i,j});
        if (both.stats.moments < a.stats.moments+b.stats.moments) shared = true;
        std::map<int,int> targets;
        for (int k=0; k<p.alpha_index_times_count; ++k)
            if (both.keep[p.alpha_index_times(k,3)]) if (++targets[p.alpha_index_times(k,3)]>1) multiple_rules=true;
    }
    REQUIRE(shared); REQUIRE(multiple_rules);
    REQUIRE_THROWS(p.analyze_basis({2,0})); REQUIRE_THROWS(p.analyze_basis({0,0}));
    REQUIRE_THROWS(p.analyze_basis({-1})); REQUIRE_THROWS(p.analyze_basis({p.alpha_scalar_moments}));
}

TEST_CASE("Dependency groups remove complete basic-moment branches", "[fastfit][cost]") {
    Workspace work;
    PotMTP p(work.model);
    const auto groups = func_fastfit::cost::dependency_groups(p);
    REQUIRE_FALSE(groups.empty());
    REQUIRE(std::is_sorted(groups.begin(), groups.end()));
    REQUIRE(std::adjacent_find(groups.begin(), groups.end()) == groups.end());
    std::vector<int> all(p.alpha_scalar_moments); std::iota(all.begin(), all.end(), 0);
    const auto full = p.analyze_basis(all);
    for (const auto &group : groups) {
        REQUIRE_FALSE(group.empty());
        std::vector<int> keep;
        std::set_difference(all.begin(), all.end(), group.begin(), group.end(), std::back_inserter(keep));
        REQUIRE_FALSE(keep.empty());
        const auto reduced = p.analyze_basis(keep);
        REQUIRE(reduced.stats.basics < full.stats.basics);
        REQUIRE(reduced.stats.efs_work(8,10,p.rb_size) < full.stats.efs_work(8,10,p.rb_size));
        // Every scalar in a group really uses a common removed basic moment.
        bool common = false;
        for (int b=0; b<p.alpha_index_basic_count; ++b) {
            if (!full.keep[b] || reduced.keep[b]) continue;
            bool used_by_all = true;
            for (int scalar : group) used_by_all &= p.analyze_basis({scalar}).keep[b];
            common |= used_by_all;
        }
        REQUIRE(common);
    }
    std::mt19937 rng(17), repeat_rng(17);
    std::vector<bool> chrom(all.size(), true), repeat = chrom;
    for (int i=0; i<100; ++i) {
        auto before = chrom;
        const bool changed = func_fastfit::cost::drop_dependency_group(chrom, groups, rng);
        REQUIRE(changed == func_fastfit::cost::drop_dependency_group(repeat, groups, repeat_rng));
        REQUIRE(chrom == repeat);
        REQUIRE(std::count(chrom.begin(), chrom.end(), true) > 0);
        if (changed) REQUIRE(std::count(chrom.begin(), chrom.end(), true) < std::count(before.begin(), before.end(), true));
        for (int j=0; j<chrom.size(); ++j) REQUIRE_FALSE((chrom[j] && !before[j]));
    }
    std::vector<bool> single(all.size(), false); single[0] = true;
    REQUIRE_FALSE(func_fastfit::cost::drop_dependency_group(single, groups, rng));
    REQUIRE_FALSE(func_fastfit::cost::drop_dependency_group(chrom, {}, rng));
}

TEST_CASE("Cost constraints archives and measured selection have deterministic semantics", "[fastfit][cost]") {
    using namespace func_fastfit::cost;
    REQUIRE(feasible(0,0)); REQUIRE_FALSE(feasible(1e-6,0));
    REQUIRE(order_loss(.01) == order_loss(.01+1e-18));
    REQUIRE_FALSE(dominates(.01, .5, .01+1e-18, .5, .02));
    REQUIRE(std::isfinite(sa_score(1.,.1,0.)));
    REQUIRE(sa_score(.01,1.,.02) < sa_score(.03,.01,.02));
    REQUIRE(dominates(.01,1.,.03,.01,.02));
    REQUIRE_FALSE(dominates(.03,.01,.01,1.,.02));
    REQUIRE(dominates(.03,.9,.04,.1,.02));
    REQUIRE_FALSE(dominates(.01,.5,.005,.6,.02));
    Archive archive(4,.02);
    for (int i=0; i<20; ++i) archive.add({{i},0,.001*i,1.-.04*i,{}});
    REQUIRE(archive.entries().size()<=4);
    const auto shortlisted=archive.shortlist(4);
    REQUIRE(shortlisted.size()==4);
    REQUIRE(std::any_of(shortlisted.begin(),shortlisted.end(),[](const Entry &e){return e.ids==std::vector<int>{0};}));
    REQUIRE(std::any_of(shortlisted.begin(),shortlisted.end(),[](const Entry &e){return e.ids==std::vector<int>{19};}));
    const auto count=archive.entries().size(); archive.add(archive.entries().front()); REQUIRE(archive.entries().size()==count);
    std::vector<Entry> entries{{{0},0,.001,1,{}},{{1},0,.002,.7,{}},{{2},0,.1,.3,{}}};
    std::vector<Measurement> times(3);
    times[0].total={110,105,115}; times[1].total={80,75,85}; times[2].total={20,19,21};
    REQUIRE(choose_measured(entries,times,.02)==1); // fastest is infeasible
    times[0].total={83,80,90};
    REQUIRE(choose_measured(entries,times,.02)==0); // overlap: prefer lower error
    REQUIRE_THROWS(choose_measured(entries,times,0));
    auto summary=summarize({3,1,5,2,4}); REQUIRE(summary.median==3); REQUIRE(summary.low==2); REQUIRE(summary.high==4);
    REQUIRE_THROWS(summarize({1,2}));
}

TEST_CASE("Pareto archive and exported front retain tradeoffs, endpoints and deterministic ties", "[fastfit][cost]") {
    using namespace func_fastfit::cost;
    Archive archive(8,.1,true);
    std::vector<Entry> entries{{{0},0,.01,1,{}},{{1},0,.02,.7,{}},{{2},0,.03,.8,{}},
                               {{3},0,.04,.4,{}},{{4},0,.2,.1,{}},{{5},0,.02,.7,{}}};
    for (auto e:entries) archive.add(e);
    REQUIRE(archive.entries().size()==3);
    auto selected=archive.shortlist(2);
    REQUIRE(selected.size()==2);
    REQUIRE(selected[0].ids==std::vector<int>{3});
    REQUIRE(selected[1].ids==std::vector<int>{0});
    REQUIRE(pareto_front(entries,{1,.7,.8,.4,.1,.7},.1)==std::vector<std::size_t>{3,1,0});
    // Measured front can differ from the proxy front. Infeasible fast entries never qualify.
    REQUIRE(pareto_front(entries,{100,70,50,60,10,70},.1)==std::vector<std::size_t>{2,1,0});
    REQUIRE_THROWS(pareto_front(entries,{},.1));
    auto mean=Training::EFSObjective::resolve("mean-moments","native",.1,10,true);
    REQUIRE(mean.native_l2==Approx(1.)); REQUIRE(mean.mean_l2==Approx(.1));
    auto sum=Training::EFSObjective::resolve("mean-moments","sum",.1,10,false);
    REQUIRE(sum.native_l2==Approx(.01)); REQUIRE(sum.mean_l2==Approx(.01));
    REQUIRE(Training::EFSObjective::resolve("legacy","native",.1,10,true).native_l2==.1);
    REQUIRE_THROWS(Training::EFSObjective::resolve("legacy","mean",.1,10,true));
    REQUIRE_THROWS(Training::EFSObjective::resolve("mean-moments","mean",.1,0,true));
    REQUIRE_THROWS(Training::EFSObjective::resolve("mean-moments","mean",INFINITY,10,true));
}

TEST_CASE("Cost-aware fastfit verifies and measures constrained models", "[fastfit][cost][apps]") {
    Workspace work;
    PotMTP p(work.model);
    std::vector<Atoms> data{frame(p),frame(p,1),frame(p,2)};
    GPIO io; io.setmap(); REQUIRE(io.write(data,"train.cfg","mlpcfg")); REQUIRE(io.write(data,"valid.cfg","mlpcfg"));
    p.Save("potential.mtp");
    std::ofstream("fastfit.ini") << "weight-l2 .1\nselection-objective error-cost\nmax-valid-loss 1000\n"
        << "cost-benchmark-topk 2\ncost-benchmark-configs 2\ncost-archive-size 8\n"
        << "cost-benchmark-rounds 3\ncost-benchmark-seconds .0001\nsearch-steps 4\npopulation 4\ngenerations 2\n";
    for (int mode : {5,6}) {
        func_fastfit::detail::WriteScalarIds("selected_ones.dat",{2});
        std::set<int> removed; for(int i=0;i<p.alpha_scalar_moments;++i) if(i!=2) removed.insert(i);
        func_fastfit::detail::WriteScalarIds("removed.dat",removed);
        std::vector<std::string> args{"fastfit","fastfit.ini","-m",std::to_string(mode),"--profile-output","cost-profile.tsv"};
        if (mode==6) args.insert(args.end(), {"--population-init","dependency","--group-mutation-rate","1"});
        const std::string bundle="pareto-"+std::to_string(mode);
        args.insert(args.end(), {"--cost-export-dir",bundle,"--cost-archive-rule","pareto"});
        REQUIRE(run(args)==0);
        const auto manifest=contents(bundle+"/manifest.tsv");
        REQUIRE(manifest.find("proxy_pareto\tmeasured_pareto")!=std::string::npos);
        REQUIRE(contents(bundle+"/parent.mtp")==contents("potential.mtp"));
        REQUIRE(contents(bundle+"/rounds.tsv").find("kernel_ns_atom\ttotal_ns_atom")!=std::string::npos);
        std::istringstream rows_in(manifest); std::string row;
        int exported=0;
        while (std::getline(rows_in,row)) {
            if (row.empty() || row[0]=='#' || row.starts_with("index\t")) continue;
            std::istringstream fields(row);std::string value;std::vector<std::string> values;
            while(std::getline(fields,value,'\t')) values.push_back(value);
            REQUIRE(values.size()==24);
            PotMTP saved(bundle+"/"+values[1]);
            REQUIRE(saved.alpha_scalar_moments>0);
            func_fastfit::ErrorsReport saved_errors(1.,.1,.001);
            for(auto& ref:data) {auto pred=ref;saved.getEFS(pred);saved_errors.add(pred,ref);}
            REQUIRE(saved_errors.loss/data.size()==Approx(std::stod(values[8])).margin(1e-10));
            REQUIRE(saved.alpha_scalar_moments==std::stoi(values[13])); ++exported;
        }
        REQUIRE(exported>=1); REQUIRE(exported<=3);
        REQUIRE_THROWS(run(args)); // rerunning must preserve the complete existing bundle
        REQUIRE(contents(bundle+"/manifest.tsv")==manifest);
        REQUIRE(contents("fastfit-cost.tsv").find("# status\tok\n") != std::string::npos);
        REQUIRE(contents("fastfit-cost.tsv").find("\tNA\t") == std::string::npos);
        PotMTP best(mode==5 ? "__TEMPERED__.mtp" : "__NSGA2__.mtp");
        if(mode==5) REQUIRE(best.alpha_scalar_moments==1);
        func_fastfit::ErrorsReport errors(1.,.1,.001);
        for(auto &ref:data) {auto pred=ref;best.getEFS(pred);errors.add(pred,ref);}
        REQUIRE(errors.loss/data.size() <= 1000.);
        const auto rows=read_profile("cost-profile.tsv");
        REQUIRE(rows.at("cost_analysis").root>0); REQUIRE(rows.at("cost_benchmark").root>0);
    }
    const auto saved=contents("__TEMPERED__.mtp");
    // Labels inconsistent across identical geometries make a zero ceiling impossible.
    data.push_back(data.front()); data.back().energy_+=1.;
    REQUIRE(io.write(data,"valid.cfg","mlpcfg"));
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","5","--max-valid-loss","0"}));
    REQUIRE(contents("fastfit-cost.tsv").find("# status\tno_feasible_candidate\n") != std::string::npos);
    REQUIRE(contents("__TEMPERED__.mtp")==saved);
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","5","--cost-report","missing/report.tsv"}));
    REQUIRE(contents("__TEMPERED__.mtp")==saved);
    std::filesystem::rename("__TEMPERED__.mtp", "saved.mtp");
    std::filesystem::create_directory("__TEMPERED__.mtp");
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","5"}));
    REQUIRE(contents("fastfit-cost.tsv").find("# status\toutput_failed\n") != std::string::npos);
    REQUIRE(contents("saved.mtp")==saved);
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","5","--max-valid-loss","-1"}));
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","5","--cost-benchmark-rounds","2"}));
    std::filesystem::remove("valid.cfg");
    REQUIRE_THROWS(run({"fastfit","fastfit.ini","-m","6"}));
}
