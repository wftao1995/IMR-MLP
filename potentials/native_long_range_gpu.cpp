#include "native_long_range_gpu.h"
#include "native_long_range_gpu_kernels.h"

#ifdef IMRMLP_GPU
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace Potentials {
namespace {

void validate(const NativeCollectives &collectives,int nlocal,const Atoms &atoms) {
    if(collectives.size<1 || collectives.rank<0 || collectives.rank>=collectives.size)
        throw std::runtime_error("Invalid native GPU communicator layout");
    if(collectives.size>1 && !collectives.sum)
        throw std::runtime_error("Distributed native GPU inference requires a sum collective");
    if(nlocal<0 || nlocal>atoms.size())
        throw std::runtime_error("Invalid native GPU owned atom count");
    for(int i=0;i<nlocal;++i) if(!atoms.pos_[i].allFinite())
        throw std::runtime_error("Native GPU positions are not finite");
}

std::vector<double> positions(const Atoms &atoms,int nlocal) {
    std::vector<double> result(std::size_t(nlocal)*3);
    for(int i=0;i<nlocal;++i) for(int d=0;d<3;++d) result[3*i+d]=atoms.pos_[i](d);
    return result;
}

void reduce_blocks(const NativeCollectives &collectives,std::vector<double> &values) {
    constexpr std::size_t block=131072;
    for(std::size_t i=0;i<values.size();i+=block)
        collectives.allreduce(values.data()+i,std::min(block,values.size()-i));
}

template<class Result> void initialize(Result &result,int nlocal) {
    result.forces.setZero(nlocal,3);result.per_atom_energy.setZero(nlocal);
    result.per_atom_virial.setZero(nlocal,6);
}

template<class Result> void finish(NativeLongRangeGpu &gpu,Result &result) {
    gpu.outputs(result.per_atom_energy.data(),result.forces.data(),result.per_atom_virial.data());
    result.energy=result.per_atom_energy.sum();
    const Eigen::Matrix<double,1,6> w=result.per_atom_virial.colwise().sum();
    result.virial << w(0),w(3),w(4),w(3),w(1),w(5),w(4),w(5),w(2);
}

} // namespace

LES::ChannelEvaluation native_les_distributed_gpu(
    const LES &model,const Atoms &atoms,const Eigen::MatrixXd &latents,int nlocal,
    const NativeCollectives &collectives,int device,LES::Workspace *provided_workspace) {
    LES::ChannelEvaluation result;
    LES::Workspace local_workspace;
    auto &workspace=provided_workspace ? *provided_workspace : local_workspace;
    const int channels=model.kernel==LES::Kernel::SOG ? model.sog_channels : 1;
    std::unique_ptr<NativeLongRangeGpu> gpu;
    collectives.stage([&] {
        validate(collectives,nlocal,atoms);LES::validate_periodic_geometry(atoms);
        if(channels<=0 || latents.rows()!=nlocal || latents.cols()!=channels || !latents.allFinite())
            throw std::runtime_error("Invalid owned GPU LES latent matrix");
        initialize(result,nlocal);result.dEdq.setZero(nlocal,channels);
        gpu=std::make_unique<NativeLongRangeGpu>(device);
        const auto x=positions(atoms,nlocal);gpu->atoms(nlocal,nlocal,x.data(),latents.col(0).data());
    });
    for(int channel=0;channel<channels;++channel) {
        std::vector<double> structure;
        collectives.stage([&] {
            model.generate_k_vectors(atoms,workspace.kvecs,true,workspace,channel);
            if(workspace.kvecs.size()>std::numeric_limits<int>::max())
                throw std::runtime_error("GPU LES reciprocal grid exceeds the index limit");
            std::vector<double> waves(workspace.kvecs.size()*6,0.);
            for(std::size_t k=0;k<workspace.kvecs.size();++k) {
                for(int d=0;d<3;++d) waves[6*k+d]=workspace.kvecs[k](d);
                waves[6*k+3]=workspace.k_kernel[k];waves[6*k+4]=workspace.k_kernel_derivative[k];
            }
            gpu->charges(nlocal,latents.col(channel).data());
            gpu->waves(workspace.kvecs.size(),waves.data());
            structure.resize(2*workspace.kvecs.size());gpu->structure(structure.data());
        });
        reduce_blocks(collectives,structure);
        collectives.stage([&] {
            gpu->reciprocal(structure.data(),.5/atoms.get_volume());
            if(model.remove_self_interaction) gpu->les_self(model.self_coefficient(channel));
            gpu->response(result.dEdq.col(channel).data());
        });
    }
    collectives.stage([&] { finish(*gpu,result); });
    return result;
}

} // namespace Potentials
#endif
