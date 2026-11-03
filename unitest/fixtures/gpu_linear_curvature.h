// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#pragma once

#include "gpu_analytic_hvp.h"
#include "train/train_mtp.h"
#include <memory>
#include <numeric>

// Shared small CPU references; never starts a device or changes acceptance
// gates in response to a GPU result.
namespace GpuLinearCurvatureFixture {
namespace F=GpuHessianFixture;
namespace A=GpuAnalyticHvpFixture;
using Potentials::PotMTP;
inline constexpr const char *version="gpu-linear-curvature-v1";
inline constexpr double cpu_tolerance=2e-10, gpu_atol=2e-8, gpu_rtol=2e-8;

inline std::unique_ptr<PotMTP> copy(PotMTP &source) {
    std::ostringstream output;source.Save(output);
    auto result=std::make_unique<PotMTP>();std::istringstream input(output.str());result->Load(input);
    result->train_init();result->set_inited();
    result->reg_L1_vector=source.reg_L1_vector;result->reg_L2_vector=source.reg_L2_vector;return result;
}
inline Eigen::VectorXd coefficients(PotMTP &model) {
    Eigen::VectorXd result(model.get_NL_coeffs_count());model.nl_get_coeffs(result.data());return result;
}
inline Eigen::MatrixXd components(PotMTP &model,Atoms input,const Eigen::VectorXd &v) {
    Neighborhoods nbhs(input,model.cutoff);Eigen::MatrixXd result;
    model.CalcHessianVectorCmpt(input,nbhs,v,result);return result;
}
inline Eigen::MatrixXd force_components(PotMTP &model,Atoms input) {
    Neighborhoods nbhs(input,model.cutoff);std::vector<double> energy;
    Eigen::Tensor<double,3,Eigen::RowMajor> forces,virial;
    model.CalcEFSCmpt_R(input,nbhs,energy,forces,virial);
    Eigen::MatrixXd result(3*input.size(),model.lin_coeff_size);
    for(int i=0;i<input.size();++i)for(int d=0;d<3;++d)for(int j=0;j<model.lin_coeff_size;++j)
        result(3*i+d,j)=forces(i,d,j);
    return result;
}
inline Eigen::MatrixXd force_difference(PotMTP &model,const Atoms &input,const Eigen::VectorXd &v,double h) {
    const double delta=h/F::maximum_atom_norm(v);auto plus=input,minus=input;
    for(int i=0;i<input.size();++i) {
        const Eigen::RowVector3d dx=delta*v.segment<3>(3*i).transpose();plus.pos_[i]+=dx;minus.pos_[i]-=dx;
    }
    return -(force_components(model,plus)-force_components(model,minus))/(2*delta);
}
inline Eigen::MatrixXd dense_components(PotMTP &model,Atoms input,const Eigen::VectorXd &v) {
    // Deliberately test-only and bounded to N<=5 by every caller.
    Neighborhoods nbhs(input,model.cutoff);std::vector<Potentials::HessianMatrix> dense;
    Eigen::VectorXd saved; model.get_coeffs(saved);
    Eigen::MatrixXd result(v.size(), model.lin_coeff_size);
    struct Restore { PotMTP &model; Eigen::VectorXd values; ~Restore() {model.set_coeffs(values);} } restore{model,saved};
    for (int j = 0; j < model.lin_coeff_size; ++j) {
        std::fill(model.linear_coeffs, model.linear_coeffs + model.lin_coeff_size, 0.);
        model.linear_coeffs[j] = 1.;
        Potentials::HessianMatrix hessian;
        auto copy = input; model.getEFSHessianOMP(copy, nbhs, hessian);
        result.col(j).setZero();
        for (int r = 0; r < v.size(); ++r) for (int c = 0; c < v.size(); ++c)
            result(r,j) += hessian(r/3,r%3,c/3,c%3)*v[c];
    }
    return result;
}
inline std::vector<int> columns(const PotMTP &model) {
    return {model.lin_coeff_size-1,0,model.species_count,model.lin_coeff_size-1,model.species_count-1};
}
struct Options {
    bool fast=false,fixed_species=true,diagonal=false;
    double energy=2.3,force=.17,virial=.031,hessian=.023,hvp=.37;
    double energy_power=1.3,force_power=.2,virial_power=.7,hessian_power=.8,hvp_power=1.2;
    double norm_d=.3,norm_f=.8,norm_p=1.4;
    GPSettings settings(bool gpu=false) const {
        GPSettings s;
        for(const auto &[key,value]:std::vector<std::pair<std::string,std::string>>{
            {"device",gpu?"cuda":"cpu"},{"gpu_id","0"},{"gpu_memory_mib","64"},
            {"gpu_workspace_mib","1"},{"gpu_curvature_host_mib","16"},
            {"gpu_batch_atoms","10"},{"gpu_batch_structures","2"},
            {"weight_L1","0"},{"weight_L2",".013"},{"penalty_weight","0"},
            {"weight_rad_w0","0"},{"weight_rad_w1","0"},{"weight_rad_w2","0"},
            {"rescale_mode","fixed"},{"random_init","false"},{"force_randomlinear","false"},
            {"force_randomradial","false"},{"do_orthogonalization","false"}})s.Modify(key,value);
        s.Modify("lfit_fastalgo",fast?"true":"false");s.Modify("fix_species_coeffs",fixed_species?"true":"false");
        s.Modify("linear_precondition",diagonal?"diagonal":"none");
        for(const auto &[key,value]:std::vector<std::pair<std::string,double>>{
            {"energy_weight",energy},{"force_weight",force},{"stress_weight",virial},
            {"hessian_weight",hessian},{"hvp_weight",hvp},{"energy_pow",energy_power},
            {"force_pow",force_power},{"stress_pow",virial_power},{"hessian_pow",hessian_power},
            {"hvp_pow",hvp_power},{"force_wgt_d",norm_d},{"force_wgt_f",norm_f},{"force_wgt_p",norm_p}})
            s.Modify(key,F::number(value));
        return s;
    }
};
struct Sample {
    PotMTP truth;
    std::vector<Atoms> data;
    std::vector<Neighborhoods> neighbors;
    explicit Sample(int kind=2,int variant=1,int count=6) {
        F::load(truth,variant);truth.train_init();truth.set_inited();
        for(int k=0;k<count;++k) {
            auto a=F::cluster(k%2?5:4);a.pos_[1][0]+=.04*std::sin(k+.3);
            if(truth.species_count==1)std::fill(a.chemmap_.begin(),a.chemmap_.end(),0);
            a.id=k;a.fitness=.4+.09*k;a.atomic_fitness.assign(a.size(),.6+.04*k);a.atomic_fitness[0]=0.;
            Neighborhoods nbhs(a,truth.cutoff);truth.getEFS(a,nbhs);
            a.energy_+=.013*std::cos(.6*k);a.forces_[1][0]+=.011;
            a.stress_(0,1)+=.017;a.stress_(1,0)-=.009;
            a.hessian_=F::analytic(truth,a);a.have_hessian=true;
            for(int r=0;r<a.hessian_.rows();++r)for(int c=0;c<a.hessian_.cols();++c)
                a.hessian_(r,c)+=r>c?.7+.01*r:.0007*std::sin(r+2*c+.3*k);
            a.hvp_directions_.resize(3*a.size(),1+k%2);a.hvp_targets_.resize(3*a.size(),1+k%2);
            for(int j=0;j<a.hvp_directions_.cols();++j) {
                a.hvp_directions_.col(j)=F::direction(a.size(),.2*j+.03*k);
                Eigen::VectorXd value;truth.applyHessianVectorProduct(a,nbhs,a.hvp_directions_.col(j),value);
                for(int r=0;r<value.size();++r)value[r]+=.001*std::sin(r+j+.7*k);
                a.hvp_targets_.col(j)=value;
            }
            if(kind==0)a.clear_hvp();
            if(kind==1){a.have_hessian=false;a.hessian_.resize(0,0);}
            if(kind!=2){a.have_energy=false;a.forces_.clear();a.have_stress=false;}
            if(k==1)a.have_energy=false;if(k==2)a.forces_.clear();if(k==3)a.have_stress=false;
            if(k==4)a.fitness=0.;if(k==5)a.clear_hvp();
            data.push_back(a);neighbors.emplace_back(data.back(),truth.cutoff);
        }
    }
    std::unique_ptr<PotMTP> initial() {
        auto model=copy(truth);
        for(int i=model->species_count;i<model->lin_coeff_size;++i)model->linear_coeffs[i]+=.06*std::sin(i+1.);
        return model;
    }
};
inline double normalization(const Atoms &a,const Options &o) {
    if(o.norm_f<=0. || (o.fast&&!a.has_forces()))return 1.;
    double force2=0.;if(a.has_forces())for(const auto &v:a.forces_)force2+=v.squaredNorm();
    return o.norm_d/(o.norm_d+std::pow(o.norm_f*force2/a.size(),o.norm_p));
}
inline double data_loss(PotMTP &model,const std::vector<Atoms> &data,
                        const std::vector<Neighborhoods> &neighbors,const Options &o) {
    double result=0.;
    for(size_t k=0;k<data.size();++k) {
        const auto &a=data[k];auto predicted=a;auto nbhs=neighbors[k];model.getEFS(predicted,nbhs);
        const double cc=normalization(a,o);
        if(a.has_energy())result+=cc*o.energy*a.fitness/std::pow(a.size(),o.energy_power)*std::pow(predicted.energy_-a.energy_,2);
        if(a.has_forces())for(size_t i=0;i<a.size();++i)
            result+=cc*o.force/std::pow(a.size(),o.force_power)*(a.atomic_fitness.empty()?1.:a.atomic_fitness[i])*(predicted.forces_[i]-a.forces_[i]).squaredNorm();
        if(a.has_stresses())result+=cc*o.virial/std::pow(a.size(),o.virial_power)*(predicted.stress_-a.stress_).squaredNorm();
        // Independent physical CPU H*v, never generated GPU basis rows or the
        // nonlinear trainer's finite-step monitor. No dense Identity matrix.
        if(a.has_hessian()&&o.hessian>0.)for(int j=0;j<3*a.size();++j) {
            Eigen::VectorXd v=Eigen::VectorXd::Zero(3*a.size()),value;v[j]=1.;
            model.applyHessianVectorProduct(predicted,nbhs,v,value);
            for(int r=0;r<=j;++r)result+=cc*o.hessian*a.fitness/std::pow(a.size(),o.hessian_power)*std::pow(value[r]-a.hessian_(r,j),2);
        }
        if(a.has_hvp()&&o.hvp>0.)for(int j=0;j<a.hvp_directions_.cols();++j) {
            Eigen::VectorXd value;model.applyHessianVectorProduct(predicted,nbhs,a.hvp_directions_.col(j),value);
            result+=cc*o.hvp*a.fitness/std::pow(a.size(),o.hvp_power)/a.hvp_directions_.cols()*(value-a.hvp_targets_.col(j)).squaredNorm();
        }
    }
    return result/data.size();
}
inline double objective(PotMTP &model,const std::vector<Atoms> &data,
                        const std::vector<Neighborhoods> &neighbors,const Options &o) {
    double value=data_loss(model,data,neighbors,o);
    for(int j=o.fixed_species?model.species_count:0;j<model.lin_coeff_size;++j)
        value+=model.reg_L2_vector.at(j)*model.linear_coeffs[j]*model.linear_coeffs[j];
    return value;
}
} // namespace GpuLinearCurvatureFixture
