#include <catch2/catch_all.hpp>

#include "potentials/native_long_range.h"
#include "potentials/native_long_range_gpu.h"
#ifdef IMRMLP_GPU
#include "runtime/mtp/gpu-direct/mtp_gpu_direct.h"
#endif
#include <barrier>
#include <thread>
#include <numeric>
#include <limits>

using namespace Potentials;
using Catch::Approx;

namespace {
int gpu_devices() {
#ifdef IMRMLP_GPU
    int count=0;std::string error;
    const bool available=MLIPCore::GpuDirect::MTPGpuDirect::available_device_count(&count,&error);
    INFO(error);REQUIRE(available);REQUIRE(count>0);
    return count;
#else
    return 0;
#endif
}

Atoms geometry(bool periodic) {
    Atoms atoms;
    atoms.cell_ << 6.2, 0., 0., 1.1, 6.8, 0., -.5, .7, 7.3;
    atoms.pos_ = {{.2, 1.4, 2.1}, {5.8, 1.8, 2.3}, {2.7, 4.1, 4.9}, {4.3, 5.3, 1.5}};
    atoms.chemmap_ = {0, 1, 0, 1};
    atoms.pbc_ = {periodic, periodic, periodic};
    return atoms;
}

struct RankData {
    Atoms atoms;
    Eigen::VectorXd charges;
    Eigen::MatrixXd latents;
    AnyNeighborhoods neighbors;
    std::vector<int> owned;
};

std::vector<RankData> partition(const Atoms &atoms, const Eigen::VectorXd &charges,
                              int ranks, double cutoff) {
    const Neighborhoods full(atoms,cutoff);
    std::vector<RankData> data(ranks);
    for (int rank=0;rank<ranks;++rank) {
        auto &local=data[rank];
        local.owned = ranks==1 ? std::vector<int>{0,1,2,3} :
            rank==0 ? std::vector<int>{0,3} : rank==1 ? std::vector<int>{1} :
            rank==2 ? std::vector<int>{2} : std::vector<int>{};
        std::vector<int> order=local.owned;
        for(int i=0;i<atoms.size();++i)
            if(std::find(order.begin(),order.end(),i)==order.end()) order.push_back(i);
        std::vector<int> inverse(order.size());
        local.atoms.cell_=atoms.cell_;local.atoms.pbc_=atoms.pbc_;
        local.charges.resize(order.size());
        for(int i=0;i<static_cast<int>(order.size());++i) {
            inverse[order[i]]=i;
            local.atoms.pos_.push_back(atoms.pos_[order[i]]);
            local.atoms.chemmap_.push_back(atoms.chemmap_[order[i]]);
            local.charges[i]=charges[order[i]];
        }
        local.neighbors.resize(local.owned.size());
        local.latents.resize(local.owned.size(),2);
        for(int i=0;i<static_cast<int>(local.owned.size());++i) {
            local.neighbors[i]=full[local.owned[i]];
            for(auto &j:local.neighbors[i].indices_) j=inverse[j];
            local.latents(i,0)=charges[local.owned[i]];
            local.latents(i,1)=.13+.7*charges[local.owned[i]];
        }
    }
    return data;
}

// A blocking collective simulator exercises the actual rank-local algorithm,
// including asymmetric counts and an empty rank.
struct Collectives {
    int size;
    std::barrier<> barrier;
    std::vector<double *> values;
    std::vector<std::size_t> counts;
    explicit Collectives(int n):size(n),barrier(n),values(n),counts(n) {}
    NativeCollectives rank(int rank) {
        NativeCollectives calls;
        calls.rank=rank;calls.size=size;
        calls.sum=[&,rank](double *data,std::size_t count) {
            values[rank]=data;counts[rank]=count;barrier.arrive_and_wait();
            if(rank==0) {
                for(int r=1;r<size;++r) if(counts[r]!=count) std::terminate();
                for(std::size_t i=0;i<count;++i) {
                    double total=0.;for(int r=0;r<size;++r) total+=values[r][i];
                    for(int r=0;r<size;++r) values[r][i]=total;
                }
            }
            barrier.arrive_and_wait();
        };
        return calls;
    }
};

template<class Result,class Evaluate>
std::pair<std::vector<Result>,std::vector<std::string>> run_ranks(int ranks,Evaluate evaluate) {
    Collectives communication(ranks);
    std::vector<Result> results(ranks);
    std::vector<std::string> errors(ranks);
    std::vector<std::thread> workers;
    for(int rank=0;rank<ranks;++rank) workers.emplace_back([&,rank] {
        try { results[rank]=evaluate(rank,communication.rank(rank)); }
        catch(const std::exception &error) { errors[rank]=error.what(); }
    });
    for(auto &worker:workers) worker.join();
    return {std::move(results),std::move(errors)};
}

template<class Result,class Reference>
void compare(const std::vector<Result> &results,const std::vector<RankData> &data,
             const Reference &reference) {
    double energy=0.;Eigen::Matrix3d virial=Eigen::Matrix3d::Zero();
    for(int rank=0;rank<static_cast<int>(results.size());++rank) {
        const auto &actual=results[rank];const auto &owned=data[rank].owned;
        REQUIRE(actual.forces.rows()==static_cast<int>(owned.size()));
        REQUIRE(actual.per_atom_virial.rows()==static_cast<int>(owned.size()));
        energy+=actual.energy;virial+=actual.virial;
        for(int i=0;i<static_cast<int>(owned.size());++i) {
            const int source=owned[i];
            REQUIRE(actual.per_atom_energy[i]==Approx(reference.per_atom_energy[source]).margin(2e-11));
            REQUIRE((actual.forces.row(i)-reference.forces.row(source)).norm()<2e-11);
            REQUIRE((actual.per_atom_virial.row(i)-reference.per_atom_virial.row(source)).norm()<2e-11);
            REQUIRE((actual.dEdq.row(i)-reference.dEdq.row(source)).norm()<2e-11);
        }
    }
    REQUIRE(energy==Approx(reference.energy).margin(2e-11));
    REQUIRE((virial-reference.virial).norm()<2e-11);
}
}

TEST_CASE("Distributed LES channels preserve zero and self conventions",
          "[lammps][native][distributed][les][sog]") {
#ifdef IMRMLP_GPU
    const int devices=gpu_devices();const bool gpu=GENERATE(false,true);
#endif
    const bool sog=GENERATE(false,true);
    const bool zero=GENERATE(false,true);
    const bool self=GENERATE(false,true);
    const int ranks=GENERATE(1,4);
    const auto atoms=geometry(true);
    Eigen::VectorXd charges(4);charges<<.6,-.4,.25,-.1;
    LES model;
    if(sog) {
        model.kernel=LES::Kernel::SOG;model.sog_channels=2;
        model.sog_weights={2.3,-.6,1.9,-.3};model.sog_log_bandwidths={-.6,.2,-.2,.3};
    }
    model.sog_include_zero=zero;model.remove_self_interaction=self;
    Eigen::MatrixXd q(atoms.size(),sog?2:1);q.col(0)=charges;
    if(sog) q.col(1)=(.13+.7*charges.array()).matrix();
    const auto reference=model.evaluate_channels(atoms,q);
    auto data=partition(atoms,charges,ranks,4.);
    const auto [results,errors]=run_ranks<LES::ChannelEvaluation>(ranks,
        [&](int rank,const NativeCollectives &calls) {
            const auto &local=data[rank];
#ifdef IMRMLP_GPU
            if(gpu) return native_les_distributed_gpu(model,local.atoms,
                Eigen::MatrixXd(local.latents.leftCols(sog?2:1)),local.owned.size(),calls,rank%devices);
#endif
            return native_les_distributed(model,local.atoms,
                Eigen::MatrixXd(local.latents.leftCols(sog?2:1)),local.owned.size(),calls);
        });
    for(const auto &error:errors) { INFO(error);REQUIRE(error.empty()); }
    compare(results,data,reference);
}

TEST_CASE("Distributed long-range input failures reach every rank",
          "[lammps][native][distributed][failure]") {
#ifdef IMRMLP_GPU
    const int devices=gpu_devices();const bool gpu=GENERATE(false,true);
#endif
    const auto atoms=geometry(true);
    Eigen::VectorXd charges=Eigen::VectorXd::Ones(atoms.size());
    auto data=partition(atoms,charges,4,4.);
    data[1].latents(0,0)=std::numeric_limits<double>::quiet_NaN();
    LES model;
    const auto [results,errors]=run_ranks<LES::ChannelEvaluation>(4,
        [&](int rank,const NativeCollectives &calls) {
            const auto &local=data[rank];
#ifdef IMRMLP_GPU
            if(gpu) return native_les_distributed_gpu(model,local.atoms,
                Eigen::MatrixXd(local.latents.leftCols(1)),local.owned.size(),calls,rank%devices);
#endif
            return native_les_distributed(model,local.atoms,
                Eigen::MatrixXd(local.latents.leftCols(1)),local.owned.size(),calls);
        });
    for(const auto &error:errors) REQUIRE_FALSE(error.empty());
}
