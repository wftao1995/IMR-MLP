// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "train_nonlinear_potential.h"
#include "potentials/potential_mtp.h"
#include "varpro.h"
#include "conditional_physical_state.h"
#include "collective_error.h"
#include "utils/comm.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <optional>
#include <sstream>

namespace {
// JSON strings must also escape control characters in user-supplied paths.
std::string quote(const std::string& value) {
    std::ostringstream out; out << '"';
    for (unsigned char c:value) {
        if (c=='"' || c=='\\') out << '\\' << c;
        else if (c<32) out << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c);
        else out << c;
    }
    out << '"'; return out.str();
}
struct Metrics {
    std::optional<double> energy, force, stress;
};
Metrics metrics(const std::vector<Atoms>& data,const std::vector<Atoms>& prediction) {
    double e=0.,f=0.,s=0.; double ne=0,nf=0,ns=0;
    Training::collective_try([&] {
    if(data.size()!=prediction.size()) GPERROR("VP prediction count mismatch");
    for (size_t k=0;k<data.size();++k) {
        const auto& a=data[k]; const auto& b=prediction[k];
        if (!a.size()) continue;
        if (a.has_energy()) {e+=std::pow((b.energy_-a.energy_)/a.size(),2);++ne;}
        if (a.has_forces()) for (int i=0;i<a.size();++i) {
            f+=(b.forces_[i]-a.forces_[i]).squaredNorm();nf+=3;
        }
        if (a.has_stresses()) {
            const double volume=std::abs(a.cell_.determinant());
            if (!(volume>0.)) GPERROR("VP stress metrics require nonzero cell volume");
            // Full-tensor component RMSE: invariant under a common rotation.
            s+=(b.stress_-a.stress_).squaredNorm()*std::pow(160.21766208/volume,2);
            ns+=9;
        }
    }
    });
#ifdef IMRMLP_MPI
    double local[]={e,f,s,ne,nf,ns},global[6];
    MPI_Allreduce(local,global,6,MPI_DOUBLE,MPI_SUM,mpi.comm);
    e=global[0];f=global[1];s=global[2];ne=global[3];nf=global[4];ns=global[5];
#endif
    Metrics m;
    if(ne) m.energy=std::sqrt(e/ne);
    if(nf) m.force=std::sqrt(f/nf);
    if(ns) m.stress=std::sqrt(s/ns);
    for (const auto& v:{m.energy,m.force,m.stress})
        if (v && !std::isfinite(*v)) GPERROR("Nonfinite VP EFS metrics");
    return m;
}
void number(std::ostream& out,const std::optional<double>& v) {
    if (v) out << *v; else out << "null";
}
void write_metrics(std::ostream& out,const Metrics& m) {
    out << "{\"energy_eV_atom\":";number(out,m.energy);
    out << ",\"force_eV_A\":";number(out,m.force);
    out << ",\"stress_GPa\":";number(out,m.stress);out << '}';
}
std::optional<double> score(const Metrics& m,const std::string& name) {
    if(name=="energy") return m.energy;
    if(name=="force") return m.force;
    return m.stress;
}
bool within(const std::optional<double>& v,double cap) {return cap<0. || (v && *v<=cap);}
struct Callbacks {
    std::function<void(const alglib::real_1d_array&,double&,alglib::real_1d_array&)> gradient;
    std::function<void(const alglib::real_1d_array&)> accepted;
    static void grad(const alglib::real_1d_array& x,double& f,alglib::real_1d_array& g,void* ptr) {
        static_cast<Callbacks*>(ptr)->gradient(x,f,g);
    }
    static void report(const alglib::real_1d_array& x,double,void* ptr) {
        static_cast<Callbacks*>(ptr)->accepted(x);
    }
};
struct FlagGuard {
    bool& value; bool saved;
    FlagGuard(bool& v):value(v),saved(v) {value=true;}
    ~FlagGuard() {value=saved;}
};
}

void TrainNonlinear::TrainVarPro() {
    RequireCurrentParameterLayout();
    Training::collective_try([&] {
    if (training_device!="cpu") GPERROR("VP requires CPU observations");
    if(weight_charge!=0. && !p_pot->supports_total_charge_training())
        GPERROR("VP requires training without a charge loss unless the model supports analytic total-charge observations");
    if (training_schedule!="single" || !trainable_groups.empty()
        || parameter_scaling!="none" || bfgs_set_scale || scaled_step_max!=0.
        || curvature_precondition!="none")
        GPERROR("VP requires schedule=single, all parameter groups, no parameter/step scaling and curvature_precondition=none");
    if (maxiter<0 || bfgs_m<=0 || bfgs_iteration_offset<0 || save_every_it<=0 || validation_interval<=0)
        GPERROR("Invalid VP iteration, history, checkpoint or validation interval setting");
    for (double v:{bfgs_grad_tol,bfgs_conv_tol,bfgs_step_tol})
        if (!std::isfinite(v) || v<0.) GPERROR("VP stopping tolerances must be finite and nonnegative");
    for (double v:{validation_max_energy,validation_max_force,validation_max_stress})
        if (!std::isfinite(v)) GPERROR("VP validation caps must be finite");
    if (validation_metric!="energy" && validation_metric!="force" && validation_metric!="stress")
        GPERROR("VP validation_metric must be energy, force or stress");
    if(mpi.rank==0) {
    if (varpro_report.empty() || std::filesystem::exists(varpro_report))
        GPERROR("VP requires a new nonempty fit:varpro_report path");
    const std::vector<std::string> checkpoints={potfilesave,potfilesave+".best-training.mtp",
        potfilesave+".best-data.mtp",potfilesave+".best-validation.mtp"};
    for (const auto& path:checkpoints)
        for(const auto &file:Training::conditional_linear_files(*p_pot,path))
            if (std::filesystem::exists(file)) GPERROR("VP requires a fresh fit:save_to prefix: "+file);
    const auto report_path=std::filesystem::absolute(varpro_report).lexically_normal();
    for(const auto& path:checkpoints)
        for(const auto &file:Training::conditional_linear_files(*p_pot,path))
            if(report_path==std::filesystem::absolute(file).lexically_normal()
                || report_path==std::filesystem::absolute(file+".tmp").lexically_normal())
                GPERROR("VP report and model paths must differ");
    }
    });
    auto failure_restore=Training::conditional_physical_restore(*p_pot);
    try {
    FlagGuard defer(defer_validation_);
    FlagGuard collective_errors(collective_evaluation_errors_);
    // Random initialization may differ by rank; use one physical model.
    Eigen::VectorXd starting(n_relax_coeffs);p_pot->nl_get_coeffs(starting.data());
    MPI_Bcast(starting.data(),n_relax_coeffs,MPI_DOUBLE,0,mpi.comm);
    p_pot->nl_set_coeffs(starting.data());
    std::unique_ptr<MTPProfiledObjective> profile;
    Training::collective_try([&] {
        auto observation_settings=linear_refit_settings;
        observation_settings.hvp=hvp_fit;observation_settings.hessian=hessian_fit;
        if(weight_charge>0.) {
            observation_settings.charge_weight=[&](const Atoms &atom){return charge_weight(atom);};
            observation_settings.charge_target=[&](const Atoms &atom){return total_charge_target(atom);};
        }
        profile=std::make_unique<MTPProfiledObjective>(*p_pot,training_set_,trainint_set_nbhs_,efs_weights(),
            weight_penatly,observation_settings,
            [&](double& f,Eigen::VectorXd& g){AccumulateGard(f,g);},varpro_linear_gradient_tolerance,varpro_linear_refinement_steps);
    });
    auto& objective=*profile;
    Eigen::VectorXd radial_scales=Eigen::VectorXd::Ones(objective.radial_size());
    Training::collective_try([&] {
        if(mpi.rank!=0 || varpro_radial_scale_file.empty()) return;
        std::ifstream in(varpro_radial_scale_file);
        if(!in) GPERROR("Cannot open VP radial scale file: "+varpro_radial_scale_file);
        for(int i=0;i<radial_scales.size();++i)
            if(!(in>>radial_scales[i]) || !std::isfinite(radial_scales[i]) || radial_scales[i]<=0.)
                GPERROR("VP radial scale file requires exactly one finite positive value per radial coefficient");
        std::string extra;
        if(in>>extra) GPERROR("Extra values in VP radial scale file");
    });
    MPI_Bcast(radial_scales.data(),radial_scales.size(),MPI_DOUBLE,0,mpi.comm);
    VarProRadialScaling coordinates(radial_scales);
    // Check overflow/underflow-sensitive input before creating output files.
    coordinates.optimizer(objective.outer_coefficients(starting));
    // Missing labels must not win selection by an artificial zero RMSE.
    if (valid_set_size_all) {
        const auto labels=metrics(valid_set_,valid_set_);
        if (!score(labels,validation_metric)
            || (validation_max_energy>=0. && !labels.energy)
            || (validation_max_force>=0. && !labels.force)
            || (validation_max_stress>=0. && !labels.stress))
            GPERROR("VP validation selection requires labels for its metric and enabled caps");
    } else if (validation_max_energy>=0. || validation_max_force>=0. || validation_max_stress>=0.) {
        GPERROR("VP validation caps require a validation set");
    }
    for(double v:{weight_eng,weight_force,weight_stress,weight_eng_power,weight_force_power,
                  weight_stress_power,weight_force_norm_d,weight_force_norm_f,weight_force_norm_p,
                  weight_penatly,reg_L1,reg_L2})
        if(!std::isfinite(v)) GPERROR("VP requires finite loss weights and powers");
    std::ofstream out;
    auto& p=*p_pot;
    Training::collective_try([&] {
    if(mpi.rank!=0) return;
    out.open(varpro_report);
    if (!out) GPERROR("Cannot open VP report: "+varpro_report);
    out << std::setprecision(17);
    out << "{\"event\":\"settings\",\"schema_version\":3,\"optimizer\":\"varpro_lbfgs\","
        << "\"stress_metric\":\"full_tensor_rmse_9\","
        << "\"mpi_processes\":" << mpi.size << ",\"restart\":\"coefficients_only_history_reset\",\"iteration_offset\":" << bfgs_iteration_offset
        << ",\"save_to\":" << quote(potfilesave) << ",\"validation_metric\":" << quote(validation_metric)
        << ",\"validation_interval\":" << validation_interval
        << ",\"validation_caps\":{\"energy_eV_atom\":" << validation_max_energy
        << ",\"force_eV_A\":" << validation_max_force << ",\"stress_GPa\":" << validation_max_stress << '}'
        << ",\"training_configurations\":" << train_set_size_all << ",\"validation_configurations\":" << valid_set_size_all
        << ",\"weights\":{\"energy\":" << weight_eng << ",\"force\":" << weight_force
        << ",\"virial\":" << weight_stress << ",\"energy_power\":" << weight_eng_power
        << ",\"force_power\":" << weight_force_power << ",\"virial_power\":" << weight_stress_power
        << ",\"force_norm_d\":" << weight_force_norm_d << ",\"force_norm_f\":" << weight_force_norm_f
        << ",\"hvp\":" << hvp_fit.weight << ",\"hvp_power\":" << hvp_fit.power << ",\"hvp_fd_step\":" << hvp_fit.step
        << ",\"hessian\":" << hessian_fit.weight << ",\"hessian_power\":" << hessian_fit.power << ",\"hessian_fd_step\":" << hessian_fit.step
        << ",\"force_norm_p\":" << weight_force_norm_p
        << ",\"radial_penalty\":" << weight_penatly
        << ",\"moment_L1\":" << reg_L1 << ",\"moment_L2\":" << reg_L2 << '}'
        << ",\"inner\":{\"solver\":\"qr_svd\",\"rtol\":" << linear_refit_settings.rtol
        << ",\"atol\":" << linear_refit_settings.atol << ",\"acceptance_tolerance\":" << linear_refit_settings.acceptance_tolerance
        << ",\"gradient_tolerance\":" << varpro_linear_gradient_tolerance << ",\"refinement_steps\":" << varpro_linear_refinement_steps << '}'
        << ",\"outer\":{\"max_iterations\":" << maxiter << ",\"history\":" << bfgs_m
        << ",\"epsg\":" << bfgs_grad_tol << ",\"epsf\":" << bfgs_conv_tol << ",\"epsx\":" << bfgs_step_tol << '}'
        << ",\"radial_scaling\":{\"file\":" << quote(varpro_radial_scale_file)
        << ",\"mapping\":\"physical_r=scale*z\",\"stopping_coordinates\":\"z\",\"scales\":[";
    for(int i=0;i<radial_scales.size();++i) {if(i) out << ',';out << radial_scales[i];}
    out << "]},\"outer_parameter_indices\":[";
    for(size_t i=0;i<objective.outer_indices().size();++i) {if(i) out<<',';out<<objective.outer_indices()[i];}
    out << "],\"inner_parameter_indices\":[";
    for(size_t i=0;i<objective.inner_indices().size();++i) {if(i) out<<',';out<<objective.inner_indices()[i];}
    out << "]}\n";
    out.flush(); if (!out) GPERROR("Failed writing VP settings");
    });

    Eigen::VectorXd accepted(n_relax_coeffs);p.nl_get_coeffs(accepted.data());
    VarProEvaluation current;
    Metrics training;
    double energy_loss=0., force_loss=0., stress_loss=0., hvp_loss=0., hessian_loss=0., charge_loss=0.;
    size_t evaluations=0;
    int accepted_steps=0, best_train_iteration=-1, best_data_iteration=-1, best_valid_iteration=-1;
    double best_train=std::numeric_limits<double>::infinity(),best_data=best_train,best_valid=best_train;
    auto evaluate=[&](const Eigen::VectorXd& radial) {
        ++evaluations;
        try { current=objective.evaluate(radial); }
        catch(const std::exception& error) {
            if(mpi.rank==0) {
                out << "{\"event\":\"evaluation_failed\",\"evaluation\":" << evaluations
                    << ",\"message\":" << quote(error.what()) << "}\n";
                out.flush();
            }
            throw;
        }
        training=metrics(training_set_,efs_set_);
        energy_loss=emonitor.loss_e;force_loss=emonitor.loss_f;stress_loss=emonitor.loss_s;
        hvp_loss=emonitor.loss_hvp;hessian_loss=emonitor.loss_h;charge_loss=qmonitor.loss_q;
        Training::collective_try([&] {
            if(mpi.rank!=0 || current.linear_gradient_history.size()==1) return;
            out << "{\"event\":\"linear_refinement\",\"evaluation\":" << evaluations
                << ",\"linear_gradient_history\":[";
            for(size_t i=0;i<current.linear_gradient_history.size();++i) {
                if(i) out << ',';
                out << current.linear_gradient_history[i];
            }
            out << "]}\n";out.flush();
            if(!out) GPERROR("Failed writing VP refinement report");
        });
    };
    enum Command { Evaluate=1, Accepted=2, Final=3, Finish=4, Abort=5 };
    auto dispatch=[&](int command) {MPI_Bcast(&command,1,MPI_INT,0,mpi.comm);};
    auto ensure=[&](const double* optimizer_point) {
        const Eigen::VectorXd r=coordinates.physical(Eigen::Map<const Eigen::VectorXd>(optimizer_point,objective.radial_size()));
        if (current.coefficients.size()==0 || !(objective.outer_coefficients(current.coefficients).array()==r.array()).all()) {
            dispatch(Evaluate);
            Eigen::VectorXd trial=r;
            MPI_Bcast(trial.data(),trial.size(),MPI_DOUBLE,0,mpi.comm);
            evaluate(trial);
        }
    };
    auto save=[&](const std::string& path) {
        const Atoms *sample=nullptr;
        for(const auto &atom:training_set_) if(atom.size()) {sample=&atom;break;}
        Training::publish_conditional_linear(p,path,sample);
    };
    auto record=[&](const char* event,bool force_validation) {
        const double data_loss=energy_loss+force_loss+stress_loss+hvp_loss+hessian_loss+charge_loss;
        Training::collective_try([&] {
            if(mpi.rank==0 && std::abs(current.objective-data_loss-current.radial_penalty-current.linear_penalty)
                >1e-9*std::max(1.,std::abs(current.objective))) GPERROR("VP loss decomposition mismatch");
        });
        accepted=current.coefficients;
        p.nl_set_coeffs(accepted.data());
        failure_restore=Training::conditional_physical_restore(p);
        const int iteration=bfgs_iteration_offset+accepted_steps;
        const bool validate=valid_set_size_all && (force_validation || iteration%validation_interval==0);
        Metrics validation;
        if(validate) {
            EvaluateValidation();
            validation=metrics(valid_set_,efs_valid_set_);
        }
        Training::collective_try([&] {
        if(mpi.rank!=0) return;
        if(current.objective<best_train) {
            save(potfilesave+".best-training.mtp");
            best_train=current.objective;best_train_iteration=iteration;
        }
        if(data_loss<best_data) {
            save(potfilesave+".best-data.mtp");
            best_data=data_loss;best_data_iteration=iteration;
        }
        const bool eligible=validate && within(validation.energy,validation_max_energy)
            && within(validation.force,validation_max_force) && within(validation.stress,validation_max_stress);
        if(eligible && *score(validation,validation_metric)<best_valid) {
            save(potfilesave+".best-validation.mtp");
            best_valid=*score(validation,validation_metric);best_valid_iteration=iteration;
        }
        // Update the restart model only at accepted points (including initial projection).
        save(potfilesave);
        if(accepted_steps>0 && iteration%save_every_it==0)
            save(fmt::format("{}.iter-{:06d}.mtp",potfilesave,iteration));
        out << "{\"event\":" << quote(event) << ",\"iteration\":" << iteration
            << ",\"evaluations\":" << evaluations << ",\"loss\":{\"total\":" << current.objective
            << ",\"data\":" << data_loss << ",\"energy\":" << energy_loss
            << ",\"force\":" << force_loss << ",\"virial\":" << stress_loss
            << ",\"hvp\":" << hvp_loss << ",\"hessian\":" << hessian_loss << ",\"charge\":" << charge_loss
            << ",\"radial_penalty\":" << current.radial_penalty << ",\"linear_penalty\":" << current.linear_penalty << '}'
            << ",\"radial_gradient_inf\":" << (current.gradient.size() ? current.gradient.lpNorm<Eigen::Infinity>() : 0.)
            << ",\"optimizer_gradient_inf\":" << (current.gradient.size() ? coordinates.gradient(current.gradient).lpNorm<Eigen::Infinity>() : 0.)
            << ",\"linear_gradient_inf\":" << current.linear_gradient_inf
            << ",\"linear_refinements\":" << current.linear_gradient_history.size()-1
            << ",\"linear_gradient_initial_inf\":" << current.linear_gradient_history.front()
            << ",\"inner_optimality\":" << current.linear.optimality
            << ",\"inner_rank\":" << current.linear.solve_rank
            << ",\"inner_seconds\":{\"assembly\":" << current.linear.assembly_seconds
            << ",\"solve\":" << current.linear.solve_seconds
            << ",\"verification\":" << current.linear.verification_seconds << '}'
            << ",\"radial_norms\":[";
        for(int k=0;k<current.radial_norms.size();++k) {if(k) out << ',';out << current.radial_norms[k];}
        out << "],\"train\":";write_metrics(out,training);
        out << ",\"validation\":";
        if(validate) write_metrics(out,validation);else out << "null";
        out << ",\"validation_eligible\":" << (eligible ? "true" : "false")
            << ",\"best_training_iteration\":" << best_train_iteration
            << ",\"best_data_iteration\":" << best_data_iteration
            << ",\"best_validation_iteration\":" << best_valid_iteration << "}\n";
        out.flush();if(!out) GPERROR("Failed writing VP report");
        Message(fmt::format("VP {} {}: total={:.12g} data={:.12g} radial_penalty={:.6g} linear_penalty={:.6g}",
            event,iteration,current.objective,data_loss,current.radial_penalty,current.linear_penalty));
        });
    };
    try {
        evaluate(objective.outer_coefficients(accepted));
        record("initial",true);
    } catch(...) {
        p.nl_set_coeffs(accepted.data());
        if(mpi.rank==0) {out << "{\"event\":\"failed\",\"accepted_steps\":0}\n";out.flush();}
        throw;
    }
    // Only root runs L-BFGS. Workers execute the same collective phases, then
    // return to this command loop even after a synchronized failure so root
    // can send Abort without leaving a peer blocked in an optimizer callback.
    if(mpi.rank!=0) {
        std::exception_ptr failure;
        while(true) {
            int command=0;MPI_Bcast(&command,1,MPI_INT,0,mpi.comm);
            if(command==Finish) break;
            if(command==Abort) {
                p.nl_set_coeffs(accepted.data());
                if(failure) std::rethrow_exception(failure);
                GPERROR("VP optimizer or output failed on root rank");
            }
            try {
                if(command==Evaluate) {
                    Eigen::VectorXd trial(objective.radial_size());
                    MPI_Bcast(trial.data(),trial.size(),MPI_DOUBLE,0,mpi.comm);
                    evaluate(trial);
                } else if(command==Accepted) {++accepted_steps;record("accepted",false);}
                else if(command==Final) record("final",true);
                else GPERROR("Unknown VP worker command");
            } catch(...) {failure=std::current_exception();}
        }
        bfgs_accepted_iter=accepted_steps;p.nl_get_coeffs(x.data());
        return;
    }
    try {
        int termination=objective.radial_size()==0 ? 4 : 0;
        if(maxiter>0 && objective.radial_size()>0) {
            alglib::real_1d_array radial;
            const Eigen::VectorXd z=coordinates.optimizer(objective.outer_coefficients(accepted));
            radial.setcontent(objective.radial_size(),z.data());
            alglib::minlbfgscreate(objective.radial_size(),std::min(bfgs_m,objective.radial_size()),radial,bfgsstate);
            alglib::minlbfgssetcond(bfgsstate,bfgs_grad_tol,bfgs_conv_tol,bfgs_step_tol,maxiter);
            alglib::minlbfgssetxrep(bfgsstate,true);
            bool initial_report=true;
            Callbacks callbacks;
            callbacks.gradient=[&](const alglib::real_1d_array& r,double& f,alglib::real_1d_array& g) {
                ensure(r.getcontent());
                f=current.objective;
                const Eigen::VectorXd transformed=coordinates.gradient(current.gradient);
                for(int i=0;i<objective.radial_size();++i) g[i]=transformed[i];
            };
            callbacks.accepted=[&](const alglib::real_1d_array& r) {
                ensure(r.getcontent());
                if(initial_report) initial_report=false;
                else {dispatch(Accepted);++accepted_steps;record("accepted",false);}
            };
            alglib::minlbfgsoptimize(bfgsstate,Callbacks::grad,Callbacks::report,&callbacks);
            alglib::minlbfgsreport report;
            alglib::minlbfgsresults(bfgsstate,radial,report);
            // The last gradient request can be a rejected line-search trial.
            termination=report.terminationtype;
            if(termination<0) GPERROR("VP L-BFGS failed; last accepted model retained");
            ensure(radial.getcontent());
        }
        dispatch(Final);record("final",true);
        out << "{\"event\":\"termination\",\"type\":" << termination
            << ",\"accepted_steps\":" << accepted_steps << ",\"evaluations\":" << evaluations
            << ",\"converged\":" << ((termination==1 || termination==2 || termination==4) ? "true" : "false")
            << ",\"best_training_iteration\":" << best_train_iteration
            << ",\"best_data_iteration\":" << best_data_iteration
            << ",\"best_validation_iteration\":" << best_valid_iteration << "}\n";
        out.flush();if(!out) GPERROR("Failed writing VP termination report");
        if(termination<0) GPERROR("VP L-BFGS failed; last accepted model retained (see report)");
        Message(fmt::format("VP finished: accepted={}, evaluations={}, termination_type={}",accepted_steps,evaluations,termination));
        dispatch(Finish);
    } catch (...) {
        dispatch(Abort);
        p.nl_set_coeffs(accepted.data());
        out << "{\"event\":\"failed\",\"accepted_steps\":" << accepted_steps << "}\n";out.flush();
        throw;
    }
    bfgs_accepted_iter=accepted_steps;
    p.nl_get_coeffs(x.data());
    } catch(...) {
        failure_restore();throw;
    }
}
