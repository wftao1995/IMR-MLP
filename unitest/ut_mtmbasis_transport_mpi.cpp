#include <catch2/catch_all.hpp>
#ifdef IMRMLP_MPI
#include "mtmbasis/solved_tensor_transport.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);mpi.InitComm(MPI_COMM_WORLD);
    Catch::Session session;session.configData().runOrder=Catch::TestRunOrder::Declared;
    const int result=session.run(argc,argv);MPI_Finalize();return result;
}
TEST_CASE("Solved tensor MPI distribution avoids worker solves and preserves generated basis", "[mtmbasis][transport][mpi]") {
    std::vector<alpha_index_type> tensors;
    const auto before=mtmbasis_detail::tensor_solve_calls.load();
    if(mpi.rank==0)tensors=mtmbasis_detail::enumerate_scalar_alpha_indices({0,1,2,3},{1,1,1,1});
    const auto after_root=mtmbasis_detail::tensor_solve_calls.load();
    mtmbasis_detail::broadcast_solved_tensors(tensors);
    const auto after_broadcast=mtmbasis_detail::tensor_solve_calls.load();
    REQUIRE(after_broadcast==after_root);REQUIRE(tensors.size()==3);
    const int local_solves=int(after_broadcast-before);int total_solves=0;
    MPI_Allreduce((void*)&local_solves,&total_solves,1,MPI_INT,MPI_SUM,mpi.comm);
    INFO("Actual root generation solves="<<total_solves<<", actual worker distribution solves=0, ranks="<<mpi.size);
    REQUIRE(total_solves==3);REQUIRE(local_solves==(mpi.rank==0 ? 3:0));
    auto independent=tensors;
    for(auto& tensor:independent)tensor.solve();
    REQUIRE(mtmbasis_detail::pack_solved_tensors(independent)==mtmbasis_detail::pack_solved_tensors(tensors));
    int root_pid=getpid();MPI_Bcast(&root_pid,1,MPI_INT,0,mpi.comm);
    const auto original=std::filesystem::current_path();
    const auto dir=std::filesystem::path("/tmp")/("imrmlp-mtmbasis-mpi-"+std::to_string(root_pid)+"-"+std::to_string(mpi.rank));
    std::filesystem::create_directories(dir);
    struct Cleanup {std::filesystem::path original,dir;~Cleanup(){std::filesystem::current_path(original);std::filesystem::remove_all(dir);}} cleanup{original,dir};
    auto generate=[&](const std::string& name,const std::vector<alpha_index_type>& pool) {
        std::filesystem::create_directory(dir/name);std::filesystem::current_path(dir/name);
        PickTensors pick;pick.alphas_gen=pool;pick.mode=2;pick.maxiter=2;pick.ga_seed=817;
        pick.popsize=4;pick.childsize=6;pick.elite_count=2;pick.initial_population="diverse";
        pick.output_file="basis.mtp";pick.gen_pot_by_alphapool_GA();
        REQUIRE(pick.cnt>0);
        std::vector<std::string> outputs;
        for(int i=0;i<pick.cnt;++i) {
            std::ifstream file("basis.mtp_"+std::to_string(i)+mpi.fnm_ending);REQUIRE(file.good());
            outputs.emplace_back(std::istreambuf_iterator<char>(file),std::istreambuf_iterator<char>());
        }
        return outputs;
    };
    const auto distributed=generate("distributed",tensors);
    const auto independently_solved=generate("independent",independent);
    REQUIRE(distributed==independently_solved);
}
#else
TEST_CASE("Solved tensor MPI transport requires an MPI build", "[mtmbasis][transport][mpi]") {
    SKIP("Run ut_mtmbasis_transport_mpi.test with BUILD_MODE=mpi and mpiexec -n 2 or -n 3");
}
#endif
