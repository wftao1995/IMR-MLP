#include "native_long_range.h"
#include "native_profile.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Potentials {
namespace {

void validate_collectives(const NativeCollectives &collectives, int nlocal, int total) {
    if (collectives.size < 1 || collectives.rank < 0 || collectives.rank >= collectives.size)
        throw std::runtime_error("Invalid native inference communicator layout");
    if (collectives.size > 1 && !collectives.sum)
        throw std::runtime_error("Distributed native inference requires a sum collective");
    if (nlocal < 0 || nlocal > total)
        throw std::runtime_error("Invalid native owned atom count");
}

void reduce_blocks(const NativeCollectives &collectives, std::vector<double> &values) {
    constexpr std::size_t block = 131072;
    for (std::size_t begin = 0; begin < values.size(); begin += block)
        collectives.allreduce(values.data() + begin, std::min(block, values.size() - begin));
}

void add_mode(Eigen::Matrix<double, Eigen::Dynamic, 6, Eigen::RowMajor> &output,
              int i, double energy, double derivative, const Eigen::Vector3d &k) {
    output(i,0) += energy + derivative*k(0)*k(0);
    output(i,1) += energy + derivative*k(1)*k(1);
    output(i,2) += energy + derivative*k(2)*k(2);
    output(i,3) += derivative*k(0)*k(1);
    output(i,4) += derivative*k(0)*k(2);
    output(i,5) += derivative*k(1)*k(2);
}

template<class Result> void summarize(Result &result) {
    result.energy = result.per_atom_energy.sum();
    const Eigen::Matrix<double, 1, 6> w = result.per_atom_virial.colwise().sum();
    result.virial << w(0), w(3), w(4), w(3), w(1), w(5), w(4), w(5), w(2);
}

} // namespace

LES::ChannelEvaluation native_les_distributed(
    const LES &model, const Atoms &atoms, const Eigen::MatrixXd &latents, int nlocal,
    const NativeCollectives &collectives, LES::Workspace *provided_workspace) {
    const int channels=model.kernel==LES::Kernel::SOG ? model.sog_channels : 1;
    LES::ChannelEvaluation result;
    LES::Workspace local_workspace;
    auto &workspace=provided_workspace ? *provided_workspace : local_workspace;
    collectives.stage([&] {
        validate_collectives(collectives,nlocal,atoms.size());
        LES::validate_periodic_geometry(atoms);
        if (latents.rows()!=nlocal || latents.cols()!=channels || !latents.allFinite())
            throw std::runtime_error("Invalid owned LES latent matrix");
        result.forces.setZero(nlocal,3);result.dEdq.setZero(nlocal,channels);
        result.per_atom_energy.setZero(nlocal);result.per_atom_virial.setZero(nlocal,6);
    });
    const double prefactor=.5/atoms.get_volume();
    for(int channel=0;channel<channels;++channel) {
        std::vector<double> structure;
        collectives.stage([&] {
            model.generate_k_vectors(atoms,workspace.kvecs,true,workspace,channel);
            structure.assign(2*workspace.kvecs.size(),0.);
            workspace.cosine.resize(nlocal);workspace.sine.resize(nlocal);
            std::array<std::array<double,3>,3> reciprocal;
            const auto basis=atoms.get_recip();
            for(int a=0;a<3;++a) for(int d=0;d<3;++d) reciprocal[a][d]=basis(a,d);
            workspace.phases.prepare(nlocal,workspace.phase_bounds_,reciprocal,
                [&](int i,int d) { return atoms.pos_[i](d); },model.use_recurrence);
            for(std::size_t k=0;k<workspace.kvecs.size();++k) {
                const auto &vector=workspace.kvecs[k];
                workspace.phases.evaluate(workspace.k_indices_[k],{vector(0),vector(1),vector(2)},
                    workspace.cosine.data(),workspace.sine.data());
                for(int i=0;i<nlocal;++i) {
                    structure[2*k]+=latents(i,channel)*workspace.cosine[i];
                    structure[2*k+1]+=latents(i,channel)*workspace.sine[i];
                }
            }
        });
        reduce_blocks(collectives,structure);
        collectives.stage([&] {
            for(std::size_t mode=0;mode<workspace.kvecs.size();++mode) {
                const auto &k=workspace.kvecs[mode];
                const double h=workspace.k_kernel[mode], hp=workspace.k_kernel_derivative[mode];
                const double sr=structure[2*mode],si=structure[2*mode+1];
                workspace.phases.evaluate(workspace.k_indices_[mode],{k(0),k(1),k(2)},
                    workspace.cosine.data(),workspace.sine.data());
                for(int i=0;i<nlocal;++i) {
                    const double c=workspace.cosine[i],s=workspace.sine[i];
                    const double re=c*sr+s*si,im=s*sr-c*si,q=latents(i,channel);
                    const double energy=prefactor*h*q*re;
                    result.per_atom_energy[i]+=energy;
                    result.dEdq(i,channel)+=2.*prefactor*h*re;
                    result.forces.row(i)+=(2.*prefactor*h*q*im*k).transpose();
                    add_mode(result.per_atom_virial,i,energy,2.*prefactor*hp*q*re,k);
                }
            }
            if(model.remove_self_interaction) {
                const double self=model.self_coefficient(channel);
                for(int i=0;i<nlocal;++i) {
                    const double q=latents(i,channel);
                    result.per_atom_energy[i]-=.5*self*q*q;
                    result.dEdq(i,channel)-=self*q;
                }
            }
        });
    }
    summarize(result);
    return result;
}

} // namespace Potentials
