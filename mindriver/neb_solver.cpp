#include "neb.h"
#include "fire.h"
#include "sgd.h"
#include "cg.h"
#include "neb_preconditioner.h"
#include <algorithm>
#include <deque>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>

namespace {
using Vector = Eigen::VectorXd;
using Matrix = Eigen::MatrixXd;

struct History {
    std::deque<Vector> s, y;
    int capacity = 15;
    void clear() { s.clear(); y.clear(); }
    void update(const Vector &step, const Vector &change) {
        if (step.dot(change) <= 1e-12 * step.stableNorm() * change.stableNorm()) return;
        s.push_back(step); y.push_back(change);
        if (s.size() > capacity) { s.pop_front(); y.pop_front(); }
    }
    Vector direction(const Vector &gradient) const {
        Vector q = gradient;
        std::vector<double> alpha(s.size());
        for (int j = int(s.size())-1; j >= 0; --j) {
            alpha[j] = s[j].dot(q) / s[j].dot(y[j]);
            q -= alpha[j] * y[j];
        }
        if (!s.empty()) q *= std::clamp(s.back().dot(y.back()) / y.back().squaredNorm(), 1e-8, 10.);
        for (int j = 0; j < s.size(); ++j)
            q += s[j] * (alpha[j] - y[j].dot(q) / s[j].dot(y[j]));
        return -q;
    }
};

// Multisecant inverse action for a nonconservative residual. Only committed
// steps enter this history; the common controller validates every proposal.
struct ResidualHistory {
    std::deque<Vector> s,y;
    void clear() {s.clear();y.clear();}
    void update(const Vector &step,const Vector &change) {
        if(!step.allFinite() || !change.allFinite() || step.norm()<1e-14 || change.norm()<1e-14)return;
        s.push_back(step);y.push_back(change);
        if(s.size()>10) {s.pop_front();y.pop_front();}
    }
    Vector direction(const Vector &residual) const {
        Matrix steps(residual.size(),s.size()),changes(residual.size(),s.size());
        for(int i=0;i<s.size();++i) {
            // Column normalization makes rank regularization independent of
            // the physical cap and the length of an accepted backtrack.
            const double norm=y[i].norm();steps.col(i)=s[i]/norm;changes.col(i)=y[i]/norm;
        }
        Eigen::JacobiSVD<Matrix> svd(changes,Eigen::ComputeThinU|Eigen::ComputeThinV);
        const Vector singular=svd.singularValues();
        Vector weights=svd.matrixU().transpose()*residual;
        for(int i=0;i<weights.size();++i)
            weights[i]*=singular[i]/(singular[i]*singular[i]+1e-6*singular[0]*singular[0]);
        const Vector coeff=svd.matrixV()*weights;
        const double beta=std::clamp(std::abs(s.back().dot(y.back()))/y.back().squaredNorm(),1e-4,1.);
        return -beta*residual-(steps-beta*changes)*coeff;
    }
    void inverse_factors(Matrix &left,Matrix &right,double &beta) const {
        Matrix steps(s.front().size(),s.size()),changes(s.front().size(),s.size());
        for(int i=0;i<s.size();++i) {
            const double norm=y[i].norm();steps.col(i)=s[i]/norm;changes.col(i)=y[i]/norm;
        }
        Eigen::JacobiSVD<Matrix> svd(changes,Eigen::ComputeThinU|Eigen::ComputeThinV);
        Vector weights=svd.singularValues();const double regularizer=1e-6*weights[0]*weights[0];
        for(int i=0;i<weights.size();++i)weights[i]/=weights[i]*weights[i]+regularizer;
        beta=std::clamp(std::abs(s.back().dot(y.back()))/y.back().squaredNorm(),1e-4,1.);
        left=steps-beta*changes;
        right=svd.matrixV()*weights.asDiagonal()*svd.matrixU().transpose();
    }
};

struct BandState {
    std::vector<Atoms> images;
    Eigen::Matrix<double,-1,-1,Eigen::RowMajor> x, g;
    std::vector<double> springs;
    double objective, perpendicular, spring, residual, saddle, equilibrium_length;
    int climbing_image;
    std::set<int> climbs,valleys;
    bool active,found;
    int protected_image;
    explicit BandState(const NEBDriver &d): images(d.images), x(d.x), g(d.g), springs(d.spconsts),
        objective(d.f), perpendicular(d.max_perp_force), spring(d.max_spring_force),
        residual(d.max_neb_force), saddle(d.saddle_g_proj), equilibrium_length(d.Leq), climbing_image(d.climbing_image),
        climbs(d.iclimbs),valleys(d.ivallys),active(d.climbing_active),found(d.found_state_site),protected_image(d.iclimb_) {}
    void restore(NEBDriver &d) const {
        d.images=images; d.x=x; d.g=g; d.spconsts=springs; d.f=objective;
        d.max_perp_force=perpendicular; d.max_spring_force=spring;
        d.max_neb_force=residual; d.saddle_g_proj=saddle;d.Leq=equilibrium_length; d.climbing_image=climbing_image;
        d.iclimbs=climbs;d.ivallys=valleys;d.climbing_active=active;
        d.found_state_site=found;d.iclimb_=protected_image;
    }
};

// Relative deformation restricted to periodic rows, including singular slabs.
double cell_move(const Atoms &old, const Atoms &next, bool subspace) {
    const double invalid=std::numeric_limits<double>::infinity();
    Matrix deformation;
    if (!subspace) {
        deformation=old.cell_.inverse()*next.cell_-Eigen::Matrix3d::Identity();
    } else {
        std::vector<int> axes;
        for(int i=0;i<3;++i) if(old.pbc_[i]) axes.push_back(i);
        if(axes.empty()) return 0;
        Matrix rows(axes.size(),3), delta(axes.size(),3);
        for(int i=0;i<axes.size();++i) { rows.row(i)=old.cell_.row(axes[i]); delta.row(i)=next.cell_.row(axes[i])-rows.row(i); }
        Eigen::LLT<Matrix> factor(rows*rows.transpose());
        if(factor.info()!=Eigen::Success)return invalid;
        deformation=rows.transpose()*factor.solve(delta);
    }
    if(!deformation.allFinite())return invalid;
    const Eigen::JacobiSVD<Matrix> spectrum(deformation);
    return spectrum.info()==Eigen::Success && spectrum.singularValues().allFinite()
        ? spectrum.singularValues().maxCoeff():invalid;
}
}

int NEBDriver::run(const std::string &driver, int max_iterations, double max_move) {
    RunOptions options;
    options.driver=driver; options.max_iterations=max_iterations; options.max_move=max_move;
    return run(options);
}

int NEBDriver::run(const RunOptions &options) {
    if(options.experimental_acceleration!="none" && options.experimental_acceleration!="secant" &&
       options.experimental_acceleration!="anderson" && options.experimental_acceleration!="combined" &&
       options.experimental_acceleration!="multisecant" && options.experimental_acceleration!="guarded" &&
       options.experimental_acceleration!="adaptive" && options.experimental_acceleration!="cell-block" &&
       options.experimental_acceleration!="graph" && options.experimental_acceleration!="colored" &&
       options.experimental_acceleration!="retained")
        throw std::invalid_argument("NEB: experimental_acceleration must be none, secant, anderson, combined, multisecant, guarded, adaptive, cell-block, graph, colored, or retained");
    if(solver_policy=="legacy" && options.experimental_acceleration!="none")
        throw std::invalid_argument("NEB: experimental acceleration requires stable-v1");
    if(options.max_iterations<=0 || options.accepted_step_offset<0 || options.max_krylov_directions<1 || options.max_krylov_directions>640 || !(options.max_move>0) || !std::isfinite(options.max_move))
        throw std::invalid_argument("NEB: positive finite move and iteration budget required");
    if(options.driver!="fire" && options.driver!="lbfgs" && options.driver!="cg" && options.driver!="sgd" && options.driver!="gd")
        throw std::invalid_argument("NEB: unknown optimizer '"+options.driver+"'");
    validate_images();
    const int result=solver_policy=="legacy" ? run_legacy(options) : run_stable(options);
    record_symmetry("final",true);
    return result;
}

bool NEBDriver::limit_step(const Vector &base, Vector &step, double max_move,
                           double &atomic_move, double &cell_change) {
    project_symmetry_step(step);
    const auto original_images=images;
    const auto original_x=x;
    auto restore=[&](){images=original_images; x=original_x;};
    try {
        X2LatPos_interf(base);
        const auto base_images=images;
        double factor=1.;
        for(int trial=0;trial<48;++trial) {
            try {
                X2LatPos_interf(base+factor*step);
                atomic_move=cell_change=0;
                for(int i=1;i<images.size()-1;++i) {
                    if(vcneb) cell_change=std::max(cell_change,cell_move(base_images[i],images[i],subspace_active));
                    for(int a=0;a<images[i].size();++a)
                        atomic_move=std::max(atomic_move,(images[i].pos_[a]-base_images[i].pos_[a]).stableNorm());
                }
                if(!std::isfinite(atomic_move) || !std::isfinite(cell_change)) { factor*=.5; continue; }
                const double ratio=std::min(max_move/std::max(atomic_move,1e-300),max_cell_strain/std::max(cell_change,1e-300));
                if(ratio>=1.-1e-13) { step*=factor; restore(); return true; }
                factor*=(vcneb?.99:1.)*ratio;
            } catch(const NEBInvalidGeometry &) { factor*=.5; }
        }
        restore();return false;
    } catch(...) {restore();throw;}
}

void NEBDriver::write_accepted_frame() {
    if(!iobj || !write_traj || (optimizer_iterations!=0 && accepted_steps%nblock))return;
    if(!trjofs.is_open()) {
        trjofs.open(trajfile);
        if(!trjofs || !iobj->seq_write_init(trjofs,trjformat,trajfile))
            throw std::runtime_error("Cannot initialize NEB trajectory: "+trajfile);
    }
    for(auto &image:images) if(!iobj->seq_write(trjofs,image))
        throw std::runtime_error("Cannot write NEB trajectory: "+trajfile);
    trjofs.flush();
    if(!trjofs)throw std::runtime_error("Cannot flush NEB trajectory: "+trajfile);
}

int NEBDriver::run_stable(const RunOptions &options) {
    stable_evaluation_=true; freeze_phase_=false; reuse_efs_=false;
    struct ResetFlags {
        bool &stable,&freeze,&reuse;
        ~ResetFlags(){stable=freeze=reuse=false;}
    } reset_flags{stable_evaluation_,freeze_phase_,reuse_efs_};
    converged=false; stop_reason=StopReason::not_run;
    optimizer_iterations=accepted_steps=rejected_steps=force_evaluations=iter=0;
    accepted_steps=options.accepted_step_offset;
    last_atomic_move=last_cell_change=0;last_fallback="none";
    const long long evaluation_limit=max_force_evaluations?max_force_evaluations:10LL*options.max_iterations+1;
    SimpleFireF fire(options.optimizer_settings);
    SimpleSGD sgd(options.optimizer_settings);
    SimpleCGF cg(options.optimizer_settings);
    double sgd_step=sgd.max_step_init;
    if(options.driver=="cg" && (!(cg.finite_step>0) || !std::isfinite(cg.finite_step)))
        throw std::invalid_argument("NEB: CG finite_step must be positive and finite");
    if((options.driver=="sgd" || options.driver=="gd") &&
       (!(sgd_step>0) || !std::isfinite(sgd_step) || !(sgd.step_upscale>0) || !(sgd.step_downscale>0)))
        throw std::invalid_argument("NEB: SGD step parameters must be positive and finite");
    if(!(fire.ts_init>0) || !(fire.ts_max>0) || !(fire.ts_min>0))
        throw std::invalid_argument("NEB: FIRE timesteps must be positive");
    History history;
    ResidualHistory residual_history;
    // FIRE's useful momentum must get the first chance after model reuse.
    // Its recycled GMRES model retains the baseline update rule; residual
    // history is only a recovery proposal when the native step is rejected.
    const bool cell_block=options.experimental_acceleration=="cell-block";
    const bool graph_model=options.experimental_acceleration=="graph";
    const bool colored_model=options.experimental_acceleration=="colored";
    const bool retained_model=options.experimental_acceleration=="retained";
    const bool adaptive=options.experimental_acceleration=="adaptive" || cell_block || graph_model || colored_model || retained_model;
    const bool guarded=options.experimental_acceleration=="guarded" || adaptive;
    const bool preserve_fire=guarded && options.driver=="fire";
    const bool rescue_history=adaptive && preserve_fire;
    const bool use_secant=options.experimental_acceleration=="secant" || options.experimental_acceleration=="combined" || (guarded && !preserve_fire);
    const bool use_multisecant=options.experimental_acceleration=="multisecant";
    const bool use_anderson=options.experimental_acceleration=="anderson" || options.experimental_acceleration=="combined" || use_multisecant || guarded;
    std::deque<Vector> outer_steps;
    if(options.optimizer_settings.count("lbfgs_m")) history.capacity=std::stoi(options.optimizer_settings["lbfgs_m"]);
    if(history.capacity<=0)throw std::invalid_argument("NEB: lbfgs_m must be positive");
    Vector native,scale,z,gradient,velocity,cg_previous,cg_direction,inverse_mass,previous_step;
    auto initialize_vectors=[&](){
        get_x(native);scale.resize(native.size());
        for(int i=0;i<scale.size();++i)scale[i]=scalval[i%scalval.size()];
        z=native.cwiseProduct(scale);velocity=Vector::Zero(z.size());
        previous_step=Vector::Zero(z.size());inverse_mass=Vector::Ones(z.size());
        if(options.driver=="fire" && fire.assignmass) {
            const auto masses=fire_masses();
            for(int j=0;j<inverse_mass.size();++j)inverse_mass[j]=scale[j]*scale[j]/masses[j];
        }
        cg_previous.resize(0);cg_direction.resize(0);history.clear();
        outer_steps.clear();
        residual_history.clear();
    };
    initialize_vectors();
    double timestep=fire.ts_init,alpha=fire.alpha_start;
    int positive_steps=0,slow_steps=0;
    Matrix recycled_basis,recycled_directions,recycled_h;
    Matrix recycled_gram_inverse;
    bool spatial_model_needed=false;
    Matrix retained_directions;
    Eigen::JacobiSVD<Matrix> recycled_svd;
    int recycled_uses=0;
    auto clear_model=[&](){recycled_basis.resize(0,0);recycled_directions.resize(0,0);recycled_h.resize(0,0);recycled_uses=0;};
    std::deque<double> merits;
    std::ofstream trace,model_trace,trial_trace,cell_trace,graph_trace,color_trace;
    if(!diagnostics_file.empty()) {
        trace.open(diagnostics_file);
        if(!trace)throw std::runtime_error("Cannot open NEB diagnostics: "+diagnostics_file);
        trace<<"accepted_steps,optimizer_iterations,band_evaluations,image_evaluations,rejected_steps,max_neb_force,residual_squared,atomic_move_A,cell_change,fallback\n"<<std::setprecision(17);
        model_trace.open(diagnostics_file+".krylov.csv");
        if(!model_trace)throw std::runtime_error("Cannot open NEB model diagnostics");
        model_trace<<"accepted_steps,band_evaluations,directions,largest_singular,smallest_singular,rhs_norm,linear_residual,predicted_drop\n"<<std::setprecision(17);
        trial_trace.open(diagnostics_file+".trials.csv");
        if(!trial_trace)throw std::runtime_error("Cannot open NEB trial diagnostics");
        trial_trace<<"accepted_steps,band_evaluations,kind,valid,accepted,cap,fraction,base_merit,merit,predicted_drop,reduction_ratio\n"<<std::setprecision(17);
        if(cell_block) {
            cell_trace.open(diagnostics_file+".cell-model.csv");
            if(!cell_trace)throw std::runtime_error("Cannot open NEB cell model diagnostics");
            cell_trace<<"accepted_steps,band_evaluations,probe_evaluations,cell_dofs,largest_singular,smallest_singular,valid\n"<<std::setprecision(17);
        }
        if(graph_model) {
            graph_trace.open(diagnostics_file+".graph.csv");
            if(!graph_trace)throw std::runtime_error("Cannot open NEB graph model diagnostics");
            graph_trace<<"accepted_steps,band_evaluations,valid_image_models,total_internal_images\n";
        }
        if(colored_model) {
            color_trace.open(diagnostics_file+".colors.csv");
            if(!color_trace)throw std::runtime_error("Cannot open NEB colored model diagnostics");
            color_trace<<"accepted_steps,band_evaluations,independent_products,color\n";
        }
    }
    auto record=[&](){
        Message(fmt::format("NEB {} accepted={} attempts={} EFS={} residual={:.9g} move={:.6g} cell={:.6g} fallback={}",
                    options.driver,accepted_steps,optimizer_iterations,force_evaluations,max_neb_force,last_atomic_move,last_cell_change,last_fallback));
        if(trace.is_open()) {
            trace<<accepted_steps<<','<<optimizer_iterations<<','<<force_evaluations<<','<<image_force_evaluations<<','<<rejected_steps<<','<<max_neb_force<<','<<gradient.squaredNorm()<<','<<last_atomic_move<<','<<last_cell_change<<','<<last_fallback<<'\n';
            trace.flush();if(!trace)throw std::runtime_error("Failed to write NEB diagnostics");
        }
        write_accepted_frame();
    };
    const std::vector<Atoms> *trusted_images=nullptr;
    double trusted_move_limit=options.max_move;
    auto evaluate=[&](const Vector &point,Vector &result)->bool {
        if(force_evaluations>=evaluation_limit) {stop_reason=StopReason::max_evaluations;return false;}
        try {
            native=point.cwiseQuotient(scale);
            X2LatPos_interf(native);
            if(trusted_images) {
                for(int i=1;i<images.size()-1;++i) {
                    if(vcneb && cell_move((*trusted_images)[i],images[i],subspace_active)>max_cell_strain*(1+1e-12))
                        {last_fallback="cell_step_rejected";return false;}
                    for(int a=0;a<images[i].size();++a)
                        if((images[i].pos_[a]-(*trusted_images)[i].pos_[a]).stableNorm()>trusted_move_limit*(1+1e-12))
                            {last_fallback="atomic_step_rejected";return false;}
                }
            }
            Vector raw;
            LatPos2X(native,raw);
            result=raw.cwiseQuotient(scale);
            return result.allFinite() && std::isfinite(f) && std::isfinite(max_neb_force);
        } catch(const NEBInvalidGeometry &) {last_fallback="invalid_geometry";return false;}
    };
    if(!evaluate(z,gradient)) {stop_reason=StopReason::numerical_failure;return 2;}
    double progress_anchor=gradient.squaredNorm();
    record();
    BandState checkpoint(*this);
    try {
      while(optimizer_iterations<options.max_iterations && !al_abort_) {
        if(check_convergence())break;
        const Vector base=z, base_gradient=gradient;
        // A frozen accepted-state secant map may still help the inner linear
        // solve after an outer proposal is rejected. Never update it with a
        // trial; the actual residual/model ratio safeguards its use.
        ResidualHistory linear_history;
        if(use_multisecant || rescue_history)linear_history=residual_history;
        const double move_limit=options.driver=="fire" && accepted_steps<fire.first_delay
            ? std::min(options.max_move,fire.first_step):options.max_move;
        trusted_images=&checkpoint.images;trusted_move_limit=move_limit;
        if(!(move_limit>0) || !std::isfinite(move_limit))throw std::invalid_argument("NEB: FIRE first_step must be positive");
        const double base_merit=gradient.squaredNorm();
        ++optimizer_iterations;
        if(force_evaluations>=evaluation_limit) {stop_reason=StopReason::max_evaluations;break;}
        // A stationary pre-climb band is an accepted zero-displacement step.
        // It advances only the explicitly requested delay, never probe clocks.
        bool accepted=base_merit==0;
        double accepted_atomic=0,accepted_cell=0;
        const double reference=merits.empty()?base_merit:std::max(base_merit,*std::max_element(merits.begin(),merits.end()));
        const double roundoff=64*std::numeric_limits<double>::epsilon()*std::max(base_merit,1e-300);
        freeze_phase_=true;
        Vector proposal;
        bool fire_backward=false,fire_accelerated=false,used_krylov=false;
        bool used_anderson=false;
        if(options.driver=="fire") {
            Vector driving=gradient;
            if(fire.freeze_small_gd)for(int j=0;j<driving.size();++j)
                if(std::abs(driving[j])<fire.freeze_gd_tol)driving[j]=0;
            if(fire.assignmass && fire.massactall)driving=driving.cwiseProduct(inverse_mass);
            bool backward=false;
            if(velocity.squaredNorm()>0) {
                if(-driving.dot(velocity)>0) {
                    const double vn=velocity.norm();velocity*=1-alpha;
                    if(driving.norm()>0)velocity-=alpha*driving*(vn/driving.norm());
                    if(!fire.initaildeyal || positive_steps>fire.ndelay) {
                        timestep=std::min(fire.ts_max,timestep*fire.ts_grow);alpha*=fire.alpha_reduce;
                    }
                    ++positive_steps;
                } else {
                    velocity.setZero();alpha=fire.alpha_start;positive_steps=0;
                    timestep=std::max(std::min(fire.ts_min,fire.ts_init),timestep*fire.ts_reduce);
                    backward=fire.half_step_back && previous_step.size()==z.size() && previous_step.squaredNorm()>0;
                }
            }
            if(fire.assignmass && !fire.massactall)driving=driving.cwiseProduct(inverse_mass);
            velocity-=timestep*driving;proposal=timestep*velocity;
            if(backward) {proposal=-fire.backward_ratio*previous_step;velocity.setZero();fire_backward=true;}
        } else if(options.driver=="lbfgs") {
            proposal=history.direction(gradient);
            const char *key=accepted_steps==0?"first_step":"initial_step";
            if(options.optimizer_settings.count(key)) {
                const double step=std::stod(options.optimizer_settings[key]);
                if(step>0)proposal*=step;
            }
            if(options.optimizer_settings.count("damping")) {
                double damping=std::stod(options.optimizer_settings["damping"]);
                if(damping>0 && damping<1)proposal*=damping;
            }
        } else if(options.driver=="cg") {
            double beta=0;
            if(cg_previous.size()==gradient.size() && cg_previous.squaredNorm()>0)
                beta=std::max(0.,gradient.dot(gradient-cg_previous)/cg_previous.squaredNorm());
            proposal=-gradient;
            if(cg_direction.size()==gradient.size())proposal+=beta*cg_direction;
            if(proposal.dot(gradient)>=0)proposal=-gradient;
            if(proposal.stableNorm()>0) {
                const Vector unit=proposal.normalized();
                Vector probe=(cg.finite_step*unit).cwiseQuotient(scale),probed;
                double atom=0,cell=0;
                if(limit_step(base.cwiseQuotient(scale),probe,move_limit,atom,cell)) {
                    probe=probe.cwiseProduct(scale);
                    if(probe.stableNorm()>0 && evaluate(base+probe,probed)) {
                        const double curvature=unit.dot(probed-base_gradient)/probe.stableNorm();
                        if(curvature>0 && std::isfinite(curvature))proposal=(-base_gradient.dot(unit)/curvature)*unit;
                    }
                }
                checkpoint.restore(*this);
            }
        } else {
            const double normalization=sgd.gonly?1.:1./std::max(gradient.cwiseAbs().maxCoeff(),1e-300);
            proposal=-sgd_step*normalization*gradient;
        }
        if(!proposal.allFinite())proposal=-gradient;

        auto try_direction=[&](const Vector &direction,double upper,double linear=0,double quadratic=0)->bool {
            const std::string direction_kind=last_fallback;
            Vector physical=direction.cwiseQuotient(scale);
            double atom=0,cell=0;
            if(!limit_step(base.cwiseQuotient(scale),physical,move_limit,atom,cell)) {
                ++rejected_steps;checkpoint.restore(*this);return false;
            }
            const Vector bounded=physical.cwiseProduct(scale);
            const double cap=direction.squaredNorm()>0?bounded.norm()/direction.norm():0;
            double fraction=1;
            for(int trial=0;trial<8 && !al_abort_;++trial,fraction*=.5) {
                Vector candidate=base+fraction*bounded,result;
                if((candidate-base).stableNorm()==0)break;
                bool valid=evaluate(candidate,result);
                if(al_abort_ || stop_reason==StopReason::max_evaluations) {checkpoint.restore(*this);return false;}
                const double merit=valid?result.squaredNorm():std::numeric_limits<double>::infinity();
                bool progress=merit<upper-roundoff;
                double predicted=0,ratio=0;
                if(linear>0) {
                    const double step=fraction*cap;
                    predicted=step*(linear-step*quadratic);
                    if(predicted>0)ratio=(base_merit-merit)/predicted;
                    progress=predicted>roundoff && merit<upper-roundoff && base_merit-merit>=.1*predicted;
                }
                if(trial_trace.is_open()) {
                    trial_trace<<accepted_steps<<','<<force_evaluations<<','<<direction_kind<<','<<valid<<','<<(valid&&progress)<<','<<cap<<','<<fraction<<','<<base_merit<<','<<merit<<','<<predicted<<','<<ratio<<'\n';
                    trial_trace.flush();if(!trial_trace)throw std::runtime_error("Failed to write NEB trial diagnostics");
                }
                // Meeting the force threshold never bypasses residual/model acceptance.
                if(valid && progress) {
                    z=candidate;gradient=result;
                    // Measure the actual accepted decoded step, not a linear estimate.
                    accepted_atomic=accepted_cell=0;
                    for(int i=1;i<images.size()-1;++i) {
                        if(vcneb)accepted_cell=std::max(accepted_cell,cell_move(checkpoint.images[i],images[i],subspace_active));
                        for(int a=0;a<images[i].size();++a)
                            accepted_atomic=std::max(accepted_atomic,(images[i].pos_[a]-checkpoint.images[i].pos_[a]).norm());
                    }
                    return true;
                }
                ++rejected_steps;checkpoint.restore(*this);
                if(valid)last_fallback="residual_rejected";
            }
            return false;
        };
        // Reuse a still predictive Krylov model. Rebuilding the same 160
        // products after every small capped correction wastes most EFS calls.
        // Acceptance still uses the true residual and the actual/predicted
        // reduction ratio, so stale models are discarded immediately.
        auto bounded_model=[&](const Matrix &vectors,const Matrix &hk,const Vector &rhs,
                               Vector &correction,double &linear,double &quadratic,
                               const Eigen::JacobiSVD<Matrix> *prepared=nullptr) {
            const double curvature=hk.squaredNorm()/double(hk.cols());
            Eigen::JacobiSVD<Matrix> computed;
            if(!prepared) {computed.compute(hk,Eigen::ComputeThinU|Eigen::ComputeThinV);prepared=&computed;}
            const auto &svd=*prepared;
            const Vector singular=svd.singularValues();
            const Vector projected=svd.matrixU().transpose()*rhs;
            double best_drop=0,least_residual=rhs.norm();
            for(double damping:{0.,1e-8,1e-6,1e-4,.01,1.,100.,1e4}) {
                Vector filtered=Vector::Zero(singular.size());
                for(int j=0;j<singular.size();++j) if(singular[j]>1e-8*singular[0])
                    filtered[j]=projected[j]*singular[j]/(singular[j]*singular[j]+damping*curvature);
                const Vector coeff=svd.matrixV()*filtered;
                if(!coeff.allFinite())continue;
                const Vector candidate=vectors*coeff;
                Vector bounded=candidate.cwiseQuotient(scale);
                double atom=0,cell=0;
                if(!limit_step(base.cwiseQuotient(scale),bounded,move_limit,atom,cell))continue;
                const double cap=candidate.norm()>0?bounded.cwiseProduct(scale).norm()/candidate.norm():0;
                const Vector model=hk*coeff;
                if(damping==0)least_residual=(model-rhs).norm();
                const double lin=2*rhs.dot(model),quad=model.squaredNorm();
                const double drop=cap*(lin-cap*quad);
                if(std::isfinite(drop) && drop>best_drop) {
                    best_drop=drop;correction=candidate;linear=lin;quadratic=quad;
                }
            }
            if(model_trace.is_open()) {
                model_trace<<accepted_steps<<','<<force_evaluations<<','<<hk.cols()<<','<<singular[0]<<','<<singular[singular.size()-1]<<','<<rhs.norm()<<','<<least_residual<<','<<best_drop<<'\n';
                model_trace.flush();if(!model_trace)throw std::runtime_error("Failed to write NEB model diagnostics");
            }
            return std::pair<double,double>{best_drop,least_residual};
        };
        if(!accepted && recycled_h.cols()>0) {
            Vector correction=Vector::Zero(z.size());double linear=0,quadratic=0;
            const Vector rhs=-recycled_basis.transpose()*base_gradient;
            bounded_model(recycled_directions,recycled_h,rhs,correction,linear,quadratic,&recycled_svd);
            last_fallback="gmres_reused";
            accepted=linear>0 && try_direction(correction,base_merit,linear,quadratic);
            used_krylov=accepted;
            if(!accepted || ++recycled_uses>=32)clear_model();
        }
        bool native_attempted=false;
        if(!accepted && preserve_fire && slow_steps<8) {
            last_fallback="none";
            accepted=try_direction(proposal,reference);
            native_attempted=true;
            fire_accelerated=accepted && !fire_backward;
        }
        if(!accepted && slow_steps<8) {
            if(use_anderson && !rescue_history && residual_history.s.size()>=2) {
                last_fallback="anderson";
                const Vector accelerated=residual_history.direction(base_gradient);
                if(accelerated.allFinite())accepted=try_direction(accelerated,reference);
                used_anderson=accepted;
                if(!accepted)residual_history.clear();
            }
        }
        if(!accepted && !native_attempted && slow_steps<8) {
            last_fallback="none";
            accepted=try_direction(proposal,reference);
            fire_accelerated=accepted && options.driver=="fire" && !fire_backward;
        }
        if(!accepted && !al_abort_ && stop_reason!=StopReason::max_evaluations) {
            history.clear();velocity.setZero();cg_previous.resize(0);cg_direction.resize(0);
            residual_history.clear();
            positive_steps=0;timestep=std::max(std::min(fire.ts_min,fire.ts_init),timestep*.5);alpha=fire.alpha_start;
            last_fallback="force_flow";
            accepted=try_direction(-.1*base_gradient,base_merit);
        }
        if((!accepted || slow_steps>=8) && !al_abort_ && stop_reason!=StopReason::max_evaluations && base_merit>0) {
            // Retain a valid force-flow improvement if a local solve cannot
            // improve it; neither probes nor a failed solve replace it.
            const Vector fallback_z=z,fallback_gradient=gradient;
            const bool fallback_accepted=accepted;
            const double fallback_atom=accepted_atomic,fallback_cell=accepted_cell;
            BandState fallback_state(*this);
            checkpoint.restore(*this);z=base;gradient=base_gradient;
            const double beta=std::sqrt(base_merit);
            const int nk=std::min<int>(options.max_krylov_directions,z.size());
            const int augmentation=nk>40?std::min<int>(6,outer_steps.size()):0;
            const Matrix retained=retained_directions;
            Matrix next_retained;
            // A band residual couples only adjacent images when the discrete
            // phase is frozen and springs are fixed. Three colors separate
            // those supports, so 3*cell_dofs probes recover the cell columns
            // of the band Jacobian. They are charged to the same EFS budget.
            // This right preconditioner changes search directions, not the
            // residual metric, projection, degrees of freedom or acceptance.
            Matrix cell_products,cell_inverse;
            const int internal=images.size()-2, stride=scale.size()/internal;
            const int nc=internal*vcoffset;
            if(cell_block && vcneb && !sel_dyn && !variable_spring && nc>0) {
                const long long before=force_evaluations;
                bool valid=true;cell_products=Matrix::Zero(z.size(),nc);
                for(int component=0;component<vcoffset && valid;++component)for(int color=0;color<std::min(3,internal) && valid;++color) {
                    Vector column=Vector::Zero(z.size());
                    for(int i=color;i<internal;i+=3)column[i*stride+component]=1;
                    double delta=1e-5/column.norm(),atom=0,cell=0;
                    Vector probe=(delta*column).cwiseQuotient(scale),response;
                    valid=limit_step(base.cwiseQuotient(scale),probe,move_limit,atom,cell);
                    if(valid)delta=probe.cwiseProduct(scale).norm()/column.norm();
                    valid=valid && delta>0 && (base+delta*column-base).norm()>0;
                    if(valid) {
                        valid=false;
                        for(int trial=0;trial<8 && force_evaluations+1<evaluation_limit && !al_abort_;++trial,delta*=.5)
                            if(evaluate(base+delta*column,response)) {valid=true;break;}
                    }
                    if(valid && !al_abort_) {
                        response=(response-base_gradient)/delta;
                        for(int i=color;i<internal;i+=3)for(int j=std::max(0,i-1);j<=std::min(internal-1,i+1);++j)
                            cell_products.block(j*stride,i*vcoffset+component,stride,1)=response.segment(j*stride,stride);
                    } else valid=false;
                    checkpoint.restore(*this);
                }
                double largest=0,smallest=0;
                if(valid && cell_products.allFinite()) {
                    Matrix block(nc,nc);
                    for(int i=0;i<internal;++i)block.middleRows(i*vcoffset,vcoffset)=cell_products.middleRows(i*stride,vcoffset);
                    Eigen::JacobiSVD<Matrix> svd(block,Eigen::ComputeFullU|Eigen::ComputeFullV);
                    valid=svd.info()==Eigen::Success && svd.singularValues().allFinite();
                    if(valid) {
                        Vector weights=svd.singularValues();largest=weights[0];smallest=weights[weights.size()-1];
                        const double floor=std::max(1e-6*largest,1e-8);
                        for(int i=0;i<weights.size();++i)weights[i]=1/std::max(weights[i],floor);
                        cell_inverse=svd.matrixV()*weights.asDiagonal()*svd.matrixU().transpose();
                        valid=cell_inverse.allFinite();
                    }
                } else valid=false;
                if(!valid) {cell_products.resize(0,0);cell_inverse.resize(0,0);}
                if(cell_trace.is_open()) {
                    cell_trace<<accepted_steps<<','<<force_evaluations<<','<<force_evaluations-before<<','<<nc<<','<<largest<<','<<smallest<<','<<valid<<'\n';
                    cell_trace.flush();if(!cell_trace)throw std::runtime_error("Failed to write NEB cell model diagnostics");
                }
            }
            std::vector<std::unique_ptr<NEBNumerics::AtomicGraph>> atomic_graphs;
            if(graph_model && spatial_model_needed && !sel_dyn && (!vcneb || recast_atomic_force)) {
                // Frozen sparse factors require geometry only, with no EFS
                // probes. Delayed activation preserves productive native and
                // low-dimensional Newton sequences on the easier paths.
                for(int i=1;i<images.size()-1;++i) {
                    auto graph=std::make_unique<NEBNumerics::AtomicGraph>();
                    graph->build(checkpoint.images[i],pot->cutoff);
                    atomic_graphs.push_back(std::move(graph));
                }
                if(graph_trace.is_open()) {
                    int valid_models=0;for(const auto &graph:atomic_graphs)valid_models+=graph->valid();
                    graph_trace<<accepted_steps<<','<<force_evaluations<<','<<valid_models<<','<<internal<<'\n';
                    graph_trace.flush();if(!graph_trace)throw std::runtime_error("Failed to write NEB graph model diagnostics");
                }
            }
            Matrix inverse_left,inverse_right;double inverse_scale=1;
            if(use_multisecant && linear_history.s.size()>=3)
                linear_history.inverse_factors(inverse_left,inverse_right,inverse_scale);
            Matrix basis=Matrix::Zero(z.size(),nk+1),directions=Matrix::Zero(z.size(),nk),h=Matrix::Zero(nk+1,nk);
            Vector target=Vector::Zero(nk+1);target[0]=beta;
            basis.col(0)=-base_gradient/beta;
            Vector correction=Vector::Zero(z.size());
            double best_drop=0,best_linear=beta,linear=0,quadratic=0;
            Matrix selected_basis,selected_directions,selected_h;
            bool corrected=false;
            bool poorly_resolved=false;
            std::unique_ptr<BandState> verified_state;
            Vector verified_z,verified_gradient;
            Matrix verified_basis,verified_directions,verified_h;
            double verified_atomic=0,verified_cell=0;
            last_fallback="gmres";
            const bool use_colored=colored_model && spatial_model_needed && !variable_spring;
            struct Product {Vector direction,value;};
            std::deque<Product> products;
            std::vector<Matrix> local_bases;
            std::vector<int> local_sizes(internal,0);
            std::vector<bool> local_active(internal,true);
            int next_color=0;
            if(use_colored)for(int i=0;i<internal;++i) {
                local_bases.push_back(Matrix::Zero(stride,nk+1));
                const Vector residual=base_gradient.segment(i*stride,stride);
                if(residual.norm()>0)local_bases.back().col(0)=-residual.normalized();
                else local_active[i]=false;
            }
            for(int k=0;k<nk && !al_abort_ &&
                (force_evaluations+1<evaluation_limit || (use_colored && !products.empty() && force_evaluations<evaluation_limit));++k) {
                Vector direction,jv;
                if(use_colored) {
                    if(products.empty()) {
                        std::vector<int> selected;
                        int color=0;
                        for(int scan=0;scan<3 && selected.empty();++scan) {
                            color=next_color;next_color=(next_color+1)%3;
                            for(int i=color;i<internal && selected.size()<nk-k;i+=3)
                                if(local_active[i])selected.push_back(i);
                        }
                        if(selected.empty())break;
                        Vector grouped=Vector::Zero(z.size());
                        for(int i:selected)grouped.segment(i*stride,stride)=local_bases[i].col(local_sizes[i]);
                        double delta=1e-5/grouped.norm(),atom=0,cell=0;
                        Vector probe=(delta*grouped).cwiseQuotient(scale),response;
                        bool valid=limit_step(base.cwiseQuotient(scale),probe,move_limit,atom,cell);
                        if(valid)delta=probe.cwiseProduct(scale).norm()/grouped.norm();
                        valid=valid && delta>0 && (base+delta*grouped-base).norm()>0;
                        if(valid) {
                            valid=false;
                            for(int trial=0;trial<8 && force_evaluations+1<evaluation_limit && !al_abort_;++trial,delta*=.5)
                                if(evaluate(base+delta*grouped,response)) {valid=true;break;}
                        }
                        checkpoint.restore(*this);
                        if(!valid || al_abort_)break;
                        response=(response-base_gradient)/delta;
                        // Three colors have disjoint residual supports. Recover
                        // each independent product from its three image rows;
                        // a probe is not a committed path state.
                        for(int i:selected) {
                            Vector column=Vector::Zero(z.size()),value=Vector::Zero(z.size());
                            const int q=local_sizes[i];column.segment(i*stride,stride)=local_bases[i].col(q);
                            for(int j=std::max(0,i-1);j<=std::min(internal-1,i+1);++j)
                                value.segment(j*stride,stride)=response.segment(j*stride,stride);
                            Vector diagonal=value.segment(i*stride,stride);
                            const double original=diagonal.norm();
                            for(int pass=0;pass<2;++pass)for(int j=0;j<=q;++j)
                                diagonal-=local_bases[i].col(j).dot(diagonal)*local_bases[i].col(j);
                            const double remaining=diagonal.norm();
                            if(remaining>1e-8*std::max(original,1e-300))local_bases[i].col(q+1)=diagonal/remaining;
                            else local_active[i]=false;
                            ++local_sizes[i];
                            // A diagonal block may break down before the band
                            // does. A neighbor's response can supply its next
                            // independent direction without another probe.
                            for(int j=std::max(0,i-1);j<=std::min(internal-1,i+1);++j)if(j!=i && !local_active[j]) {
                                Vector coupled=value.segment(j*stride,stride);
                                const double initial=coupled.norm();
                                for(int pass=0;pass<2;++pass)for(int qj=0;qj<local_sizes[j];++qj)
                                    coupled-=local_bases[j].col(qj).dot(coupled)*local_bases[j].col(qj);
                                if(coupled.norm()>1e-8*std::max(initial,1e-300)) {
                                    local_bases[j].col(local_sizes[j])=coupled.normalized();local_active[j]=true;
                                }
                            }
                            products.push_back({std::move(column),std::move(value)});
                        }
                        if(color_trace.is_open()) {
                            color_trace<<accepted_steps<<','<<force_evaluations<<','<<selected.size()<<','<<color<<'\n';
                            color_trace.flush();if(!color_trace)throw std::runtime_error("Failed to write NEB colored model diagnostics");
                        }
                    }
                    direction=std::move(products.front().direction);jv=std::move(products.front().value);products.pop_front();
                } else {
                    // LGMRES-style augmentation carries a few verified correction
                    // directions across restarts. Re-evaluate their products at
                    // this trusted point; no stale EFS or Jacobian is assumed.
                    if(k>0 && k<=retained.cols())direction=retained.col(k-1);
                    else direction=k>=nk-augmentation?outer_steps[k-(nk-augmentation)]:Vector(basis.col(k));
                    if(k<nk-augmentation && inverse_left.cols()>0)
                        direction=inverse_scale*direction+inverse_left*(inverse_right*direction);
                    if(k<nk-augmentation && cell_inverse.cols()>0) {
                        Vector cells(nc);
                        for(int i=0;i<internal;++i)cells.segment(i*vcoffset,vcoffset)=direction.segment(i*stride,vcoffset);
                        cells=(cell_inverse*cells).eval();
                        direction-=cell_products*cells;
                        for(int i=0;i<internal;++i)direction.segment(i*stride,vcoffset)=cells.segment(i*vcoffset,vcoffset);
                    }
                    if(k<nk-augmentation && !atomic_graphs.empty()) {
                        for(int i=0;i<internal;++i)if(atomic_graphs[i]->valid()) {
                            Matrix rhs(images[i+1].size(),3);
                            for(int a=0;a<rhs.rows();++a)rhs.row(a)=direction.segment<3>(i*stride+vcoffset+3*a).transpose();
                            const Matrix solved=atomic_graphs[i]->solve(rhs);
                            if(solved.allFinite())for(int a=0;a<rhs.rows();++a)
                                direction.segment<3>(i*stride+vcoffset+3*a)=solved.row(a).transpose();
                        }
                    }
                    if(!direction.allFinite() || direction.stableNorm()<1e-20)direction=basis.col(k);
                    // Scaling a search column must not change the damped local
                    // solve's effective penalty or numerical rank cutoff.
                    direction/=direction.stableNorm();
                    const double direction_norm=direction.stableNorm();
                    Vector probed;double delta=1e-5/direction_norm;bool valid=false;
                    // Probes obey the same decoded geometry limits as physical
                    // candidates. Use the reduced increment in the quotient too.
                    Vector probe_step=(delta*direction).cwiseQuotient(scale);
                    double probe_atom=0,probe_cell=0;
                    if(!limit_step(base.cwiseQuotient(scale),probe_step,move_limit,probe_atom,probe_cell))break;
                    delta=probe_step.cwiseProduct(scale).stableNorm()/direction_norm;
                    if(z.size()<=128) {
                        probe_step=(-delta*direction).cwiseQuotient(scale);
                        if(!limit_step(base.cwiseQuotient(scale),probe_step,move_limit,probe_atom,probe_cell))break;
                        delta=std::min(delta,probe_step.cwiseProduct(scale).stableNorm()/direction_norm);
                    }
                    if(delta==0 || (base+delta*direction-base).stableNorm()==0)break;
                    for(int trial=0;trial<8;++trial,delta*=.5) {
                        if(evaluate(base+delta*direction,probed)) {valid=true;break;}
                        if(stop_reason==StopReason::max_evaluations || al_abort_)break;
                    }
                    if(!valid || al_abort_)break;
                    jv=(probed-base_gradient)/delta;
                    // Small paths often have exact translation/gauge null modes.
                    // Central probes avoid manufacturing rank from directional
                    // truncation error in those degenerate Krylov spaces.
                    if(z.size()<=128) {
                        if(force_evaluations+1>=evaluation_limit)break;
                        Vector opposite;
                        if(evaluate(base-delta*direction,opposite))jv=(probed-opposite)/(2*delta);
                        if(al_abort_)break;
                    }
                }
                directions.col(k)=direction;
                const double original_norm=jv.stableNorm();
                for(int pass=0;pass<2;++pass) for(int j=0;j<=k;++j) {
                    double dot=basis.col(j).dot(jv);h(j,k)+=dot;jv-=dot*basis.col(j);
                }
                const double norm=jv.stableNorm();h(k+1,k)=norm;
                const bool breakdown=norm<1e-8*std::max(original_norm,1e-300);
                if(std::isfinite(norm) && norm>0 && !(use_colored && breakdown))basis.col(k+1)=jv/norm;
                else if(use_colored && breakdown)h(k+1,k)=0;
                const bool colored_exhausted=use_colored && products.empty() &&
                    std::none_of(local_active.begin(),local_active.end(),[](bool active){return active;});
                const bool checkpoint_solve=k==19 || k==39 || k==79 || k==159 || k==319 || k==nk-1 || breakdown || colored_exhausted || (use_colored && products.empty() && force_evaluations+1>=evaluation_limit);
                if(checkpoint_solve) {
                    const Matrix hk=h.topLeftCorner(k+2,k+1);
                    // A near-null mode can make an exact Newton correction
                    // enormous. Select by predicted decrease AFTER the physical
                    // cap, using inexpensive damped solves in this same Krylov
                    // space. This is a trust-region model, not a new force field.
                    Vector candidate=Vector::Zero(z.size());double lin=0,quad=0;
                    auto [drop,least_residual]=bounded_model(directions.leftCols(k+1),hk,target.head(k+2),candidate,lin,quad);
                    best_linear=std::min(best_linear,least_residual);
                    if(k>=159) {
                        poorly_resolved=least_residual>.9*beta;
                        if(poorly_resolved)spatial_model_needed=true;
                        if(retained_model && poorly_resolved)
                            next_retained=NEBNumerics::soft_directions(directions.leftCols(k+1),hk,20);
                    }
                    if(drop>best_drop) {
                        best_drop=drop;correction=candidate;linear=lin;quadratic=quad;
                        selected_basis=basis.leftCols(k+2);selected_directions=directions.leftCols(k+1);selected_h=hk;
                    }
                    // Test at 20, 40, 80, 160 directions in turn. A physical
                    // move cap may preclude a large fractional reduction;
                    // a small verified improvement is still worth accepting.
                    checkpoint.restore(*this);
                    double upper=fallback_accepted?std::min(base_merit,fallback_gradient.squaredNorm()):base_merit;
                    if(verified_state)upper=std::min(upper,verified_gradient.squaredNorm());
                    if(lin>0 && try_direction(candidate,upper,lin,quad)) {
                        verified_state=std::make_unique<BandState>(*this);
                        verified_z=z;verified_gradient=gradient;
                        verified_atomic=accepted_atomic;verified_cell=accepted_cell;
                        verified_basis=basis.leftCols(k+2);verified_directions=directions.leftCols(k+1);verified_h=hk;
                        best_drop=drop;
                        // This only controls the cost of model expansion.
                        // Even an arbitrarily small verified improvement is
                        // retained if larger spaces fail or exhaust the budget.
                        if(least_residual<=.5*beta || base_merit-gradient.squaredNorm()>=.02*base_merit || (breakdown && !use_colored) || k==nk-1)break;
                    }
                    checkpoint.restore(*this);z=base;gradient=base_gradient;
                    if(al_abort_ || stop_reason==StopReason::max_evaluations)break;
                    // A well-solved local model remains useful when its step
                    // is necessarily capped; do not spend another 140 EFS
                    // calls demanding a large decrease impossible at this cap.
                    // Only a failed full expansion or happy breakdown ends
                    // an unsuccessful solve; prediction alone is insufficient.
                }
                if(!std::isfinite(norm) || (breakdown && !use_colored))break;
            }
            // A derivative probe needs a later candidate evaluation. Reserve
            // that call and report the budget boundary rather than a false stall.
            if(force_evaluations+1>=evaluation_limit)stop_reason=StopReason::max_evaluations;
            if(verified_state) {
                verified_state->restore(*this);z=verified_z;gradient=verified_gradient;
                accepted_atomic=verified_atomic;accepted_cell=verified_cell;
                selected_basis=std::move(verified_basis);selected_directions=std::move(verified_directions);selected_h=std::move(verified_h);
                corrected=true;
            } else checkpoint.restore(*this);
            Message(fmt::format("NEB Krylov directions={} predicted_drop={:.9g} linear_residual={:.9g} corrected={} force_flow={}",
                        selected_h.cols(),best_drop,best_linear,corrected,fallback_accepted));
            if(corrected) {
                recycled_basis=std::move(selected_basis);recycled_directions=std::move(selected_directions);recycled_h=std::move(selected_h);recycled_uses=0;
                recycled_svd.compute(recycled_h,Eigen::ComputeThinU|Eigen::ComputeThinV);
                if(use_secant) {
                    const Matrix gram=recycled_directions.transpose()*recycled_directions;
                    Eigen::JacobiSVD<Matrix> inverse(gram,Eigen::ComputeThinU|Eigen::ComputeThinV);
                    inverse.setThreshold(1e-10);
                    recycled_gram_inverse=inverse.solve(Matrix::Identity(gram.rows(),gram.cols()));
                }
            }
            else clear_model();
            accepted=corrected || fallback_accepted;
            if(accepted && next_retained.cols()>0)retained_directions=std::move(next_retained);
            used_krylov=used_krylov || corrected;
            if(!corrected && fallback_accepted) {
                fallback_state.restore(*this);z=fallback_z;gradient=fallback_gradient;
                accepted_atomic=fallback_atom;accepted_cell=fallback_cell;
                last_fallback="gmres_failed_force_flow";
            }
            history.clear();velocity.setZero();slow_steps=0;
            // A cheap history proposal must beat the best verified candidate,
            // not merely an older nonmonotone reference. Do not interrupt a
            // FIRE/Newton sequence whose residual model already solves well.
            if(rescue_history && poorly_resolved && linear_history.s.size()>=2 &&
               !al_abort_ && stop_reason!=StopReason::max_evaluations) {
                BandState best_state(*this);
                const Vector best_z=z,best_gradient=gradient;
                const double upper=accepted?std::min(base_merit,gradient.squaredNorm()):base_merit;
                const std::string prior_fallback=last_fallback;
                checkpoint.restore(*this);z=base;gradient=base_gradient;
                last_fallback="anderson_rescue";
                const Vector accelerated=linear_history.direction(base_gradient);
                if(accelerated.allFinite() && try_direction(accelerated,upper)) {
                    accepted=true;used_anderson=true;used_krylov=false;fire_accelerated=false;clear_model();
                } else {
                    best_state.restore(*this);z=best_z;gradient=best_gradient;last_fallback=prior_fallback;
                }
            }
        }
        if(al_abort_) {checkpoint.restore(*this);z=base;gradient=base_gradient;break;}
        if(!accepted) {
            checkpoint.restore(*this);z=base;gradient=base_gradient;
            if(stop_reason!=StopReason::max_evaluations)stop_reason=StopReason::stalled;
            break;
        }
        ++accepted_steps;
        last_atomic_move=accepted_atomic;last_cell_change=accepted_cell;
        if(fire_accelerated) velocity=(z-base)/timestep;
        previous_step=z-base;
        cg_previous=base_gradient;cg_direction=z-base;
        history.update(z-base,gradient-base_gradient);
        if(use_anderson)residual_history.update(z-base,gradient-base_gradient);
        if(used_anderson)velocity.setZero();
        if(use_secant && used_krylov && recycled_h.cols()>0 && recycled_uses<16) {
            const Vector step=z-base;
            if(step.squaredNorm()>1e-24) {
                const Vector projections=recycled_directions.transpose()*step;
                const Vector coeff=recycled_gram_inverse*projections;
                // Correct J*P with the observed secant, including components
                // outside the old residual basis. No additional EFS is used.
                Vector defect=gradient-base_gradient-recycled_basis*(recycled_h*coeff);
                Vector components=Vector::Zero(recycled_basis.cols());
                for(int pass=0;pass<2;++pass) {
                    const Vector part=recycled_basis.transpose()*defect;
                    components+=part;defect-=recycled_basis*part;
                }
                const Eigen::RowVectorXd row=projections.transpose()/step.squaredNorm();
                recycled_h+=components*row;
                if(defect.norm()>1e-10*(gradient-base_gradient).norm()) {
                    const int n=recycled_basis.cols();const double norm=defect.norm();
                    recycled_basis.conservativeResize(Eigen::NoChange,n+1);recycled_basis.col(n)=defect/norm;
                    recycled_h.conservativeResize(n+1,Eigen::NoChange);recycled_h.row(n)=norm*row;
                }
                recycled_svd.compute(recycled_h,Eigen::ComputeThinU|Eigen::ComputeThinV);
            }
        }
        if((options.driver=="sgd" || options.driver=="gd") && !sgd.gonly) {
            const double change=gradient.squaredNorm()<base_merit?sgd.step_upscale:sgd.step_downscale;
            sgd_step=std::clamp(sgd_step*change,std::max(1e-20,sgd.min_step),1e20);
        }
        if(used_krylov && (z-base).stableNorm()>1e-20) {
            Vector direction=(z-base).normalized();
            bool distinct=true;for(const auto &old:outer_steps)if(std::abs(old.dot(direction))>.9999)distinct=false;
            if(distinct) {outer_steps.push_back(std::move(direction));if(outer_steps.size()>6)outer_steps.pop_front();}
        }
        // Alternating high/low residuals must not reset stagnation detection
        // on every descending half-cycle of the nonmonotone window.
        if(gradient.squaredNorm()<.99*progress_anchor) {
            progress_anchor=gradient.squaredNorm();slow_steps=0;
        } else ++slow_steps;
        merits.push_back(gradient.squaredNorm());if(merits.size()>5)merits.pop_front();
        const auto previous_climbs=iclimbs,previous_valleys=ivallys;
        const bool previous_active=climbing_active;
        freeze_phase_=false;reuse_efs_=true;
        native=z.cwiseQuotient(scale);
        Vector raw;LatPos2X(native,raw);gradient=raw.cwiseQuotient(scale);
        reuse_efs_=false;
        if(iclimbs!=previous_climbs || ivallys!=previous_valleys || climbing_active!=previous_active) {
            spatial_model_needed=false;
            retained_directions.resize(0,0);
            clear_model();
            history.clear();velocity.setZero();cg_previous.resize(0);cg_direction.resize(0);
            outer_steps.clear();
            residual_history.clear();
            merits.clear();slow_steps=0;
            progress_anchor=gradient.squaredNorm();
        }
        checkpoint=BandState(*this);
        record();
        if(variable_image && accepted_steps%image_adjust_interval==0) {
            check_image_spacing();
            if(needs_image_adjust) {
                spatial_model_needed=false;
                retained_directions.resize(0,0);
                clear_model();
                trusted_images=nullptr;
                rebuild_path();needs_image_adjust=false;initialize_vectors();merits.clear();slow_steps=0;
                if(!evaluate(z,gradient))break;
                checkpoint=BandState(*this);
                progress_anchor=gradient.squaredNorm();
            }
        }
      }
    } catch(...) {checkpoint.restore(*this);throw;}
    checkpoint.restore(*this);
    if(al_abort_)stop_reason=StopReason::active_learning_abort;
    else if(check_convergence()) {
        if(climb && climbing_image<0)stop_reason=StopReason::no_internal_barrier;
        else {stop_reason=StopReason::converged;converged=true;}
    } else if(stop_reason==StopReason::not_run)stop_reason=StopReason::max_iterations;
    if(trjofs.is_open()) {trjofs.close();if(!trjofs)throw std::runtime_error("Failed to close NEB trajectory");}
    return converged?0:2;
}

void NEBDriver::write_diagnostics(std::ostream &out) const {
    out<<"image,volume,tilt_bx,tilt_cx,tilt_cy,stretch_min,stretch_max,eq_stretch_min,eq_stretch_max";
    for(int i=0;i<9;++i)out<<",U"<<i/3<<i%3;
    for(int i=0;i<9;++i)out<<",equivalent_A"<<i/3<<i%3;
    out<<'\n'<<std::setprecision(17);
    for(int i=0;i<images.size();++i) {
        const auto &a=images[i];Eigen::Matrix3d reduced=a.cell_;
        Eigen::Matrix<long long,3,3> transform=Eigen::Matrix<long long,3,3>::Identity();
        // Integer row operations only; this never changes the optimization path.
        for(int row=1;row<3;++row) if(a.pbc_[row]) for(int col=row-1;col>=0;--col) if(a.pbc_[col]) {
            double denom=reduced.row(col).squaredNorm();
            double projection=denom>0?(reduced.row(row)-refcell.row(row)).dot(reduced.row(col))/denom:0;
            if(std::isfinite(projection) && std::abs(projection)<1e6) {
                long long n=std::llround(projection);
                reduced.row(row)-=double(n)*reduced.row(col);transform.row(row)-=n*transform.row(col);
            }
        }
        Vector raw=Vector::Ones(1),eq=raw;
        std::vector<int> axes;
        for(int j=0;j<3;++j)if(a.pbc_[j])axes.push_back(j);
        if(vcneb && !subspace_active)axes={0,1,2};
        if(!axes.empty()) {
            Matrix reference(axes.size(),3),current(axes.size(),3),equivalent(axes.size(),3);
            for(int j=0;j<axes.size();++j) {
                reference.row(j)=refcell.row(axes[j]);current.row(j)=a.cell_.row(axes[j]);equivalent.row(j)=reduced.row(axes[j]);
            }
            Eigen::LLT<Matrix> factor(reference*reference.transpose());
            if(factor.info()==Eigen::Success) {
                Matrix mapped=factor.matrixL().solve(current),mapped_eq=factor.matrixL().solve(equivalent);
                raw=Eigen::JacobiSVD<Matrix>(mapped).singularValues();
                eq=Eigen::JacobiSVD<Matrix>(mapped_eq).singularValues();
            }
        }
        out<<i<<','<<a.get_volume()<<','<<a.cell_(1,0)<<','<<a.cell_(2,0)<<','<<a.cell_(2,1)<<','<<raw.minCoeff()<<','<<raw.maxCoeff()<<','<<eq.minCoeff()<<','<<eq.maxCoeff();
        for(int r=0;r<3;++r)for(int c=0;c<3;++c)out<<','<<transform(r,c);
        for(int r=0;r<3;++r)for(int c=0;c<3;++c)out<<','<<reduced(r,c);
        out<<'\n';
    }
}
