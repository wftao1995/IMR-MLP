// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once
#include "mtp_linear_efs.h"
#include "potentials/mtp_gpu_adapter.h"
#include "train/train_mtp.h"
#include "train/collective_error.h"
#include <filesystem>
#include <unistd.h>

namespace MTPNonlinearGpuFixture {
using Potentials::PotMTP;
namespace G = MLIPCore::GpuDirect;
struct Capture {
    std::ostringstream stream;
    std::ostream *previous = SetStreamForOutput(&stream);
    ~Capture() { SetStreamForOutput(previous); }
};
inline GPSettings settings(bool gpu = true) {
    GPSettings s;
    for (const auto &[k,v] : std::vector<std::pair<std::string,std::string>>{
        {"device",gpu ? G::compiled_backend() : "cpu"}, {"gpu_id","0"},
        {"gpu_batch_atoms","8"}, {"gpu_batch_structures","2"},
        {"gpu_memory_mib","8"}, {"gpu_workspace_mib","1"},
        {"energy_weight","2.3"}, {"force_weight",".17"}, {"stress_weight",".031"},
        {"energy_pow","1.3"}, {"force_pow",".2"}, {"stress_pow",".7"},
        {"force_wgt_d",".3"}, {"force_wgt_f",".8"}, {"force_wgt_p","1.4"},
        {"weight_L2",".013"}, {"penalty_weight",".003"},
        {"skip_preinit","true"}, {"bfgs_maxiteration","2"},
        {"bfgs_grad_tol","0"}, {"bfgs_step_tol","0"}, {"bfgs_conv_tol","0"},
        {"do_orthogonalization","false"}, {"rescale_mode","fixed"},
        {"epoch_limit","1"}, {"batch_size","3"}, {"learning_rate",".00001"},
        {"save_every","10000"}}) s.Modify(k,v);
    return s;
}
struct Fixture {
    PotMTP model;
    std::vector<Atoms> data;
    std::vector<Neighborhoods> neighbors;
    Fixture(int count = 5, int variant = 1) {
        MTPLinearEFSFixture::load(model, variant); model.train_init(); model.set_inited();
        for (int k=0;k<count;++k) {
            Atoms a; a.id=k; a.cell_=12.*Eigen::Matrix3d::Identity();
            a.pos_={{2.,2.,2.},{4.1+.03*k,2.2,2.},{2.3,4.2+.02*k,2.4},{5.,4.8,3.1+.01*k}};
            a.chemmap_={0,model.species_count-1,0,model.species_count-1};
            a.fitness=.4+.09*k; a.atomic_fitness={.0,.8,1.2,.6};
            model.getEFS(a); a.energy_+=.03*std::cos(k+.3); a.forces_[1][0]+=.011;
            a.stress_(0,1)+=.017; a.stress_(1,0)-=.009;
            if (k==1) a.have_energy=false;
            if (k==2) a.forces_.clear();
            if (k==3) a.have_stress=false;
            if (k==4) a.fitness=0.;
            data.push_back(a); neighbors.emplace_back(a,model.cutoff);
        }
        model.radial_coeffs[0]+=.001; model.moment_coeffs[0]+=.02;
    }
};
inline double scale(const PotMTP &model) {
    return model.radial_basis_split ? model.p_radical_basis_pair.front()->scaling : model.p_radical_basis->scaling;
}
inline void rescale(PotMTP &model, double value) {
    if (model.radial_basis_split) {
        const double ratio=value/scale(model);
        for (auto *basis : model.p_radical_basis_pair) basis->scaling*=ratio;
    } else model.p_radical_basis->scaling=value;
}
inline Eigen::VectorXd coefficients(const PotMTP &model) {
    Eigen::VectorXd x(model.get_coeff_size()); model.get_coeffs(x); return x;
}
inline std::unique_ptr<PotMTP> copy(PotMTP &model) {
    std::ostringstream os; model.Save(os); auto p=std::make_unique<PotMTP>();
    std::istringstream is(os.str()); p->Load(is); p->train_init(); p->set_inited(); return p;
}
inline bool available() {
#ifdef IMRMLP_GPU
    int count=0; std::string error;
    bool ok=G::MTPGpuDirect::available_device_count(&count,&error) && count>0;
    if (std::getenv("IMRMLP_REQUIRE_GPU") && !ok) throw std::runtime_error("Required GPU unavailable: "+error);
    return ok;
#else
    return false;
#endif
}
struct Workspace {
    std::filesystem::path directory;
    Workspace() {
        const auto root=std::filesystem::absolute("build"); std::filesystem::create_directories(root);
        std::string name=(root/"trainmtp-nonlinear-XXXXXX").string();
        std::vector<char> buffer(name.begin(),name.end()); buffer.push_back(0);
        if (!mkdtemp(buffer.data())) throw std::runtime_error("Cannot create test directory");
        directory=buffer.data();
    }
    ~Workspace() { std::error_code e; std::filesystem::remove_all(directory,e); }
};
inline void optimize(Trainer_MTP &t, const std::string &method) {
    if (method=="lbfgs") t.TrainBFGS();
    else if (method=="cg") t.TrainCG();
    else t.TrainSGD();
}
}
