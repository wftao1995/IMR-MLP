#include <catch2/catch_all.hpp>
#include "apps/apps.h"
#include "potentials/potential_wrapper.h"
#include "potentials/potential_lj.h"
#include "potentials/potential_mtp.h"
#include "potentials/potential_hybrid.h"
#include "potentials/potential_hybrid_scale.h"
#include "utils/comm.h"
#include "utils/cmdopts.h"
#include <filesystem>
#include <fstream>
#include <unistd.h>

#ifdef IMRMLP_MPI
int main(int argc, char **argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    mpi.InitComm(MPI_COMM_WORLD);
    const int status = Catch::Session().run(argc, argv);
    MPI_Finalize();
    return status;
}
#endif

TEST_CASE("frame-local EFS capability is concrete, not inherited", "[calc-err][frames][factory]") {
    struct UnreviewedMTP : Potentials::PotMTP {};
    Potentials::PotMTP native;
    UnreviewedMTP derived;
    CHECK(Potentials::supports_frame_local_efs(native));
    CHECK_FALSE(Potentials::supports_frame_local_efs(derived));
    GPSettings settings;
    Potentials::LJ612 lj(settings);
    CHECK(Potentials::supports_frame_local_efs(lj));
}

TEST_CASE("frame-local EFS checks every composite child recursively", "[calc-err][frames][factory]") {
    using namespace Potentials;
    GPSettings hybrid_settings;
    hybrid_settings.Modify("npot", "1");
    hybrid_settings.Modify("pot1:pottype", "zbl");
    hybrid_settings.Modify("pot1:z", "13 28");
    PotentialHybrid hybrid(hybrid_settings);
    CHECK(supports_frame_local_efs(hybrid));

    namespace fs = std::filesystem;
    fs::path fixture = "unitest/test-data/portable-mtp/plain.mtp";
    if (!fs::exists(fixture)) fixture = "test-data/portable-mtp/plain.mtp";
    REQUIRE(fs::exists(fixture));
    GPSettings scale_settings;
    scale_settings.Modify("npot", "1");
    scale_settings.Modify("feature_r1", "4");
    scale_settings.Modify("pot1:pottype", "mtp");
    scale_settings.Modify("pot1:potfile", fixture.string());
    auto scale = std::make_unique<PotentialHybridScale>(scale_settings);
    CHECK(supports_frame_local_efs(*scale));
    hybrid.ppots[0] = std::move(scale);
    CHECK(supports_frame_local_efs(hybrid));

    // EAM is a valid scale child but has not been approved for frame-local EFS.
    GPSettings unapproved_settings;
    unapproved_settings.Modify("npot", "1");
    unapproved_settings.Modify("feature_r1", "4");
    unapproved_settings.Modify("pot1:pottype", "eampoly");
    unapproved_settings.Modify("pot1:nelems", "1");
    unapproved_settings.Modify("pot1:npoly_rho", "2");
    unapproved_settings.Modify("pot1:npoly_f", "1");
    unapproved_settings.Modify("pot1:npoly_pair", "2");
    unapproved_settings.Modify("pot1:rmax_pair_0_0", "6");
    unapproved_settings.Modify("pot1:rmax_rho_0_0", "6");
    unapproved_settings.Modify("pot1:coeffs", "0 0 0 0 0 1");
    auto unapproved_scale = std::make_unique<PotentialHybridScale>(unapproved_settings);
    CHECK_FALSE(supports_frame_local_efs(*unapproved_scale));
    hybrid.ppots[0] = std::move(unapproved_scale);
    CHECK_FALSE(supports_frame_local_efs(hybrid));

    struct UnreviewedMTP : PotMTP {};
    hybrid.ppots[0] = std::make_unique<UnreviewedMTP>();
    CHECK_FALSE(supports_frame_local_efs(hybrid));
}

TEST_CASE("calc_err frames preserves comparison output and empty shards", "[calc-err][frames][mpi]") {
    namespace fs = std::filesystem;
    int owner = getpid(); MPI_Bcast(&owner,1,MPI_INT,0,mpi.comm);
    const auto dir = fs::temp_directory_path()/("imrmlp-frames-"+std::to_string(owner));
    if (mpi.rank == 0) {
        fs::create_directories(dir);
        std::ofstream(dir/"index.map") << "H 0\n";
        std::ofstream ref(dir/"ref.xyz"), pred(dir/"pred.xyz");
        for (int frame=0;frame<2;++frame) {
            for (auto *stream : {&ref,&pred}) {
                *stream << frame+1 << "\nLattice=\"10 0 0 0 10 0 0 0 10\" Properties=species:S:1:pos:R:3 energy="
                        << frame+(stream==&pred ? 2:0) << "\n";
                for (int atom=0;atom<=frame;++atom) *stream << "H " << atom+1 << " 1 1\n";
            }
        }
    }
    MPI_Barrier(mpi.comm);
    auto saved = CMDOption::global_cmdopts;
    CMDOption::global_cmdopts = CMDOption::CommandLineOptions{};
    CMDOption::global_cmdopts.set_helponexit(false);
    const auto output=dir/("energy-"+std::to_string(mpi.rank));
    std::vector<std::string> args{"calc_err","--mpi-mode","frames","--input",(dir/"ref.xyz").string(),
        "--input-predict",(dir/"pred.xyz").string(),"--input-format","extxyz","--index",(dir/"index.map").string(),
        "--output-energy-diff",output.string(),"--percentile","0.75"};
    std::vector<char*> argv; for(auto &arg:args) argv.push_back(arg.data());
    CHECK(calc_err(argv.size(),argv.data())==0);
    CMDOption::global_cmdopts = saved;
    CHECK(fs::exists(output)==(mpi.rank==0));
    if(mpi.rank==0) {
        std::ifstream stream(output); std::string header; std::getline(stream,header);
        double ref,pred; stream>>ref>>pred; CHECK(ref==0); CHECK(pred==2);
        stream>>ref>>pred; CHECK(ref==1); CHECK(pred==3);
    }
    MPI_Barrier(mpi.comm);
    if(mpi.rank==0) fs::remove_all(dir);
}
