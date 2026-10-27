#include <catch2/catch_all.hpp>
#include "mtmbasis/ga_population.h"
#include "mtmbasis/solved_tensor_transport.h"

TEST_CASE("GA explicit seeds define repeatable rank streams","[mtmbasis][ga]") {
    auto a=mtmbasis_detail::ga_random_stream(128,0),b=mtmbasis_detail::ga_random_stream(128,0);
    auto c=mtmbasis_detail::ga_random_stream(128,1);bool different=false;
    for(int i=0;i<50;++i) {const auto x=a();REQUIRE(x==b());different|=x!=c();}
    REQUIRE(different);
    PickTensors defaults;REQUIRE(defaults.ga_seed==-1);REQUIRE(defaults.elite_count==0);
    REQUIRE(defaults.initial_population=="legacy");
    GPSettings settings;settings.Modify("seed","42");settings.Modify("elite_count","2");
    settings.Modify("initial_population","diverse");PickTensors explicit_settings(settings);
    REQUIRE(explicit_settings.ga_seed==42);REQUIRE(explicit_settings.elite_count==2);
    REQUIRE(explicit_settings.initial_population=="diverse");
}
TEST_CASE("GA elites preserve best repaired distinct genotypes","[mtmbasis][ga]") {
    using namespace mtmbasis_detail;
    std::vector<GAIndividual> elites;
    update_ga_elites(elites,{{{1,0},{4,4}},{{1,0},{4,4}},{{0,1},{4,4}},{{2,2},{9,9}}},2,1,1);
    REQUIRE(elites.size()==2);REQUIRE(elites[0].genes==std::vector<int>{0,1});
    const auto best=elites[0];
    update_ga_elites(elites,{{{3,3},{10,10}},{{4,4},{11,11}}},2,1,1);
    REQUIRE(elites[0].genes==best.genes);REQUIRE(elites[0].fitness==best.fitness);
    update_ga_elites(elites,{{{5,5},{1,2}}},2,1,1);
    REQUIRE(elites[0].genes==std::vector<int>{5,5});REQUIRE(elites[1].genes==best.genes);
}
TEST_CASE("GA diverse initialization repairs deduplicates and terminates","[mtmbasis][ga]") {
    using namespace mtmbasis_detail;
    auto run=[](uint32_t seed) {
        auto random=ga_random_stream(seed,0);
        return diverse_ga_population(GAIndividual{{0,0,7},{0,0}},{1,1,0},{0,1},8,2,random,
            [](std::vector<int>& genes) {return std::pair{genes[0]+genes[1],0};});
    };
    const auto a=run(7),b=run(7);REQUIRE(a.unique_count==4);REQUIRE(a.members.size()==8);
    bool multigene=false;
    for(size_t i=0;i<a.members.size();++i) {
        REQUIRE(a.members[i].genes==b.members[i].genes);REQUIRE(a.members[i].genes[2]==7);
        multigene|=a.members[i].genes[0]==1 && a.members[i].genes[1]==1;
    }
    REQUIRE(multigene);
    auto random=ga_random_stream(8,0);size_t calls=0;
    const auto collapsed=diverse_ga_population(GAIndividual{{0,0},{1,2}},{100,100},{0,1},5,2,random,
        [&](std::vector<int>& genes) {++calls;genes={0,0};return std::pair{1,2};});
    REQUIRE(collapsed.unique_count==1);REQUIRE(collapsed.members.size()==5);
    REQUIRE(calls==collapsed.evaluations);REQUIRE(calls<=5*80);
}
TEST_CASE("Solved tensor transport preserves canonicalization without solving","[mtmbasis][transport]") {
    auto source=mtmbasis_detail::enumerate_scalar_alpha_indices({0,1,2,3},{1,1,1,1});
    const auto count=mtmbasis_detail::tensor_solve_calls.load();
    const auto packet=mtmbasis_detail::pack_solved_tensors(source);
    auto restored=mtmbasis_detail::unpack_solved_tensors(packet);
    REQUIRE(mtmbasis_detail::tensor_solve_calls.load()==count);
    REQUIRE(restored.size()==source.size());
    for(size_t i=0;i<source.size();++i) {
        REQUIRE(restored[i]==source[i]);REQUIRE(restored[i].first==source[i].first);
        REQUIRE(restored[i].second==source[i].second);REQUIRE(restored[i].stdform==source[i].stdform);
        REQUIRE(restored[i].canonical_orders==source[i].canonical_orders);
        REQUIRE(restored[i].diag==source[i].diag);REQUIRE(restored[i].mu==source[i].mu);
        REQUIRE(restored[i].nv==source[i].nv);REQUIRE(restored[i].musorted==source[i].musorted);
        REQUIRE(restored[i].nvsorted==source[i].nvsorted);REQUIRE(restored[i].solved);
    }
    auto truncated=packet;truncated.pop_back();REQUIRE_THROWS(mtmbasis_detail::unpack_solved_tensors(truncated));
    auto trailing=packet;trailing.push_back(0);REQUIRE_THROWS(mtmbasis_detail::unpack_solved_tensors(trailing));
    source[0].solved=false;REQUIRE_THROWS(mtmbasis_detail::pack_solved_tensors(source));
}

#ifndef IMRMLP_MPI
#include <filesystem>
#include <fstream>
#include <unistd.h>
TEST_CASE("Seeded diverse elite GA reproduces actual generated basis files", "[mtmbasis][ga][integration]") {
    const auto pool=mtmbasis_detail::enumerate_scalar_alpha_indices({0,1,2,3},{1,1,1,1});
    const auto original=std::filesystem::current_path();
    char pattern[]="/tmp/imrmlp-mtmbasis-ga-XXXXXX";const char* path=mkdtemp(pattern);REQUIRE(path!=nullptr);
    const auto directory=std::filesystem::path(path);
    struct Cleanup {std::filesystem::path original,directory;~Cleanup(){std::filesystem::current_path(original);std::filesystem::remove_all(directory);}} cleanup{original,directory};
    auto run=[&](std::string name) {
        std::filesystem::create_directory(directory/name);std::filesystem::current_path(directory/name);
        PickTensors pick;pick.alphas_gen=pool;pick.mode=2;pick.maxiter=3;pick.ga_seed=519;
        pick.popsize=5;pick.childsize=8;pick.elite_count=2;pick.initial_population="diverse";
        pick.output_file="basis.mtp";pick.gen_pot_by_alphapool_GA();REQUIRE(pick.cnt>0);
        std::vector<std::string> result;
        for(int i=0;i<pick.cnt;++i) {
            std::ifstream in("basis.mtp_"+std::to_string(i));REQUIRE(in.good());
            result.emplace_back(std::istreambuf_iterator<char>(in),std::istreambuf_iterator<char>());
        }
        return result;
    };
    const auto first=run("first"),second=run("second");REQUIRE(first==second);
}
#endif
