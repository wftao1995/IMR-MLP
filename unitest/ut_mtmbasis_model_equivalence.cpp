#include <catch2/catch_all.hpp>
#ifndef IMRMLP_MPI
#include "mtmbasis/mtm.h"
#include "potentials/potential_mtp.h"
#include "apps/apps.h"
#include <filesystem>
#include <fstream>
#include <climits>
#include <unistd.h>
namespace {
struct BasisDirectory {
    std::filesystem::path original=std::filesystem::current_path(),directory;
    CMDOption::CommandLineOptions saved=CMDOption::global_cmdopts;
    BasisDirectory() {
        char pattern[]="/tmp/imrmlp-basis-model-XXXXXX";const auto path=mkdtemp(pattern);
        if(!path)throw std::runtime_error("Cannot create basis test directory");
        directory=path;std::filesystem::current_path(directory);
    }
    ~BasisDirectory(){CMDOption::global_cmdopts=saved;std::filesystem::current_path(original);std::filesystem::remove_all(directory);}
    int command(std::vector<std::string> args) {
        CMDOption::global_cmdopts.reset();std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
        return command_maps.at("mtmbasis").first(argv.size(),argv.data());
    }
};
struct ModelValues {Atoms atom;std::vector<std::vector<double>> basis;};
ModelValues evaluate(const std::string& file) {
    Potentials::PotMTP potential(file);potential.train_init();potential.set_inited();
    for(int i=0;i<potential.radial_coeff_size;++i)potential.radial_coeffs[i]=.02*std::cos(.7*i);
    for(int i=0;i<potential.lin_coeff_size;++i)potential.linear_coeffs[i]=1.;
    for(int i=0;i<potential.species_count;++i)potential.species_coeffs[i]=0.;
    ModelValues values;auto& a=values.atom;a.cell_=14.*Eigen::Matrix3d::Identity();
    a.pbc_={false,false,false};a.pos_={{2.,2.,2.},{4.4,2.3,2.1},{2.2,4.6,2.4},{4.5,4.8,3.1}};a.chemmap_={0,0,0,0};
    potential.getEFS(a);Neighborhoods neighborhoods(a,potential.cutoff);
    for(int i=0;i<a.size();++i) {
        potential.CalcSiteBasis(neighborhoods[i]);std::vector<double> basis;
        for(int j=0;j<potential.alpha_scalar_moments;++j)basis.push_back(potential.moment_vals[potential.alpha_moment_mapping[j]]);
        std::sort(basis.begin(),basis.end());values.basis.push_back(std::move(basis));
    }
    return values;
}
void same_observables(const ModelValues& a,const ModelValues& b) {
    REQUIRE(b.basis.size()==a.basis.size());
    for(size_t i=0;i<a.basis.size();++i) {
        REQUIRE(b.basis[i].size()==a.basis[i].size());
        for(size_t j=0;j<a.basis[i].size();++j)REQUIRE(b.basis[i][j]==Catch::Approx(a.basis[i][j]).margin(2e-12));
    }
    REQUIRE(b.atom.energy_==Catch::Approx(a.atom.energy_).margin(2e-12));
    REQUIRE((b.atom.stress_-a.atom.stress_).norm()<2e-11);
    for(int i=0;i<a.atom.size();++i)REQUIRE((b.atom.forces_[i]-a.atom.forces_[i]).norm()<2e-11);
}
}
TEST_CASE("Alternative repaired contraction choices load with identical scalar basis and EFS", "[mtmbasis][ga][model]") {
    BasisDirectory work;PickTensors pick;
    pick.alphas_gen=mtmbasis_detail::enumerate_scalar_alpha_indices({0,1,2,3},{1,1,1,1});
    pick.mode=2;pick.maxiter=0;pick.ga_seed=27;pick.elite_count=2;pick.initial_population="diverse";
    pick.output_file="basis.mtp";pick.gen_pot_by_alphapool_GA();REQUIRE(pick.cnt>0);
    std::vector<int> genes(pick.ga_upperlim.size());
    for(size_t i=0;i<genes.size();++i)genes[i]=pick.ga_upperlim[i]<0 ? -1:0;
    int alternatives=0;
    for(size_t i=0;i<genes.size() && alternatives<8;++i)for(int allele=0;allele<=pick.ga_upperlim[i] && alternatives<8;++allele) {
        if(pick.ga_upperlim[i]<1)continue;
        auto candidate=genes;candidate[i]=allele;
        // Emit this legal repaired candidate independently of its search score.
        pick.alpha_moments_count_min=INT_MAX;pick.alpha_index_times_count_min=INT_MAX;
        pick.build_pot_from_tree(candidate,pick.collections);++alternatives;
    }
    REQUIRE(alternatives>=2);
    const auto reference=evaluate("basis.mtp_0");
    for(int i=1;i<pick.cnt;++i)same_observables(reference,evaluate("basis.mtp_"+std::to_string(i)));
}
TEST_CASE("mtmbasis CLI accepts seeded elite diverse settings and publishes loadable basis", "[mtmbasis][ga][cli]") {
    BasisDirectory work;PickTensors source;
    auto pool=mtmbasis_detail::enumerate_scalar_alpha_indices({0,1,2,3},{1,1,1,1});source.save_pool("pool.dat",pool);
    std::ofstream("settings.ini")<<"mode 2\nseed 42\nelite_count 2\ninitial_population diverse\n"
        <<"maxiter 2\npopsize 4\nchildsize 6\n";
    REQUIRE(work.command({"mtmbasis","--settings=settings.ini","--alphapool=pool.dat","--potfile=cli.mtp"})==0);
    REQUIRE(std::filesystem::exists("cli.mtp_0"));
    const auto result=evaluate("cli.mtp_0");REQUIRE(std::isfinite(result.atom.energy_));
}
#else
TEST_CASE("MTP basis model CLI integration uses the serial CLI harness", "[mtmbasis][ga][model]") {
    SKIP("Serial model/CLI tests run in the release gate; distributed basis is checked by ut_mtmbasis_transport_mpi");
}
#endif
