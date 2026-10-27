#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "io/gpio.h"
#include <filesystem>
#include <fstream>
using Catch::Approx;
int trans_reduce(int,char**);
int primi_reduce(int,char**);
namespace {
struct SearchCLI {
    std::filesystem::path previous=std::filesystem::current_path(),directory;
    SearchCLI() {
        char name[]="/tmp/imrmlp_search_cli_XXXXXX";const char* value=mkdtemp(name);
        if(!value)throw std::runtime_error("Cannot create search CLI test directory");
        directory=value;std::filesystem::current_path(directory);std::ofstream("index.map")<<"Al\nNi\n";
    }
    ~SearchCLI(){CMDOption::global_cmdopts.reset();std::filesystem::current_path(previous);std::filesystem::remove_all(directory);}
    int run(std::vector<std::string> args) {
        CMDOption::global_cmdopts.reset();std::vector<char*> argv;for(auto& arg:args)argv.push_back(arg.data());
        const auto command=args.front()=="trans_reduce" ? &trans_reduce : &primi_reduce;
        return command(argv.size(),argv.data());
    }
};
Atoms input(bool slab) {
    Atoms atom;atom.cell_=Eigen::Matrix3d::Identity();atom.cell_.diagonal()<<6,3.7,4.1;
    atom.pos_={{.2,.3,.4},{1.1,1.3,1.7},{2.2,.9,2.6},{3.2,.3,.4},{4.1,1.3,1.7},{5.2,.9,2.6}};
    atom.chemmap_={0,0,1,0,0,1};if(slab)atom.pbc_={true,true,false};return atom;
}
}
TEST_CASE("Reduction CLI opt-in search preserves outputs and reports scope only when requested", "[apps][symmetry][search]") {
    SearchCLI workspace;GPIO io;io.readmap("index.map");
    for(bool slab:{false,true}) {
        std::vector<Atoms> frames{input(slab)};
        REQUIRE(io.write(frames,"input.xyz","extxyz"));
        for(const std::string command:{"trans_reduce","primi_reduce"})for(bool two_stage:{false,true}) {
            std::vector<Atoms> outputs[2];
            for(int optimized=0;optimized<2;++optimized) {
                const std::string output=optimized ? "on.xyz" : "off.xyz";
                std::vector<std::string> args{command,"--input=input.xyz","--input-format=extxyz","--output="+output,"--output-format=extxyz","--index=index.map","--overwrite"};
                if(command=="primi_reduce")args.push_back(two_stage ? "--two-stage=true" : "--two-stage=false");
                if(optimized)args.push_back("--optimize-search");
                REQUIRE(workspace.run(args)==0);
                REQUIRE(CMDOption::global_cmdopts.was_provided("optimize-search")==bool(optimized));
                REQUIRE(io.read(outputs[optimized],output,"extxyz"));REQUIRE(outputs[optimized].size()==1);
            }
            const auto& a=outputs[0][0];const auto& b=outputs[1][0];
            REQUIRE(a.size()==b.size());REQUIRE(a.pbc_==b.pbc_);REQUIRE(a.types_count_map()==b.types_count_map());
            REQUIRE(a.cell_.isApprox(b.cell_,1e-10));
            for(int i=0;i<a.size();++i)REQUIRE(a.pos_[i].isApprox(b.pos_[i],1e-10));
        }
    }
}
