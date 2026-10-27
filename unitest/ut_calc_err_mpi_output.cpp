#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "utils/comm.h"
#include "utils/cmdopts.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided);
    mpi.InitComm(MPI_COMM_WORLD);
    const int status = Catch::Session().run(argc,argv);
    MPI_Finalize(); return status;
}
#endif

TEST_CASE("calc_err only root publishes replicated MPI statistics", "[calc-err][mpi][output]") {
    namespace fs=std::filesystem;
    int owner=static_cast<int>(getpid());
    MPI_Bcast(&owner,1,MPI_INT,0,mpi.comm);
    const auto directory=fs::temp_directory_path()/("imrmlp-calc-err-output-"+std::to_string(owner));
    if (mpi.rank==0) {
        fs::create_directories(directory);
        std::ofstream(directory/"index.map") << "H 0\n";
        std::ofstream(directory/"ref.xyz") << "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 energy=1\nH 1 1 1\n";
        std::ofstream(directory/"pred.xyz") << "1\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 energy=3\nH 1 1 1\n";
    }
    MPI_Barrier(mpi.comm);
    const auto report=directory/("summary-"+std::to_string(mpi.rank));
    const auto energy=directory/("energy-"+std::to_string(mpi.rank));
    auto saved=CMDOption::global_cmdopts;
    CMDOption::global_cmdopts=CMDOption::CommandLineOptions{};
    CMDOption::global_cmdopts.set_helponexit(false);
    std::vector<std::string> args{"calc_err","--input",(directory/"ref.xyz").string(),
        "--input-predict",(directory/"pred.xyz").string(),"--input-format","extxyz",
        "--index",(directory/"index.map").string(),"--output-summary",report.string(),
        "--output-energy-diff",energy.string()};
    std::vector<char*> argv; for(auto &s:args) argv.push_back(s.data());
    CHECK(calc_err(argv.size(),argv.data())==0);
    CMDOption::global_cmdopts=saved;
    CHECK(fs::exists(report)==(mpi.rank==0));
    CHECK(fs::exists(energy)==(mpi.rank==0));
    if (mpi.rank==0) {
        std::ifstream input(energy); std::string header; std::getline(input,header);
        double reference=0,predicted=0; input>>reference>>predicted;
        CHECK(reference==1.); CHECK(predicted==3.);
    }
    MPI_Barrier(mpi.comm);
    if(mpi.rank==0) fs::remove_all(directory);
}
