#include "train_nonlinear_potential.h"
#include "collective_error.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include <random>

namespace {
struct LESStage {
    Potentials::LES* kernel=nullptr;
    bool* active=nullptr;
    std::vector<std::string> short_range;
    std::function<void(const std::vector<Atoms>&,const std::vector<double>&,double,int)> initialize;
    bool sog() const {return kernel && active && *active && kernel->kernel==Potentials::LES::Kernel::SOG;}
};

// Preserve the ordinary SOG initialization contract: nonzero channels stay
// untouched, and each zero channel draws moments before species constants.
void initialize_sog_heads(double* species,double* moments,int ns,int nm,int channels,double range,int seed) {
    if(!(range>0.) || !std::isfinite(range) || seed<0) GPERROR("Invalid SOG initialization range or seed");
    std::mt19937 rng(static_cast<unsigned int>(seed));
    std::uniform_real_distribution<double> draw(-range,range);
    for(int c=0;c<channels;++c) {
        bool zero=true;
        for(int i=0;i<ns;++i) zero=zero && std::abs(species[c*ns+i])<=1e-14;
        for(int a=0;a<nm;++a) zero=zero && std::abs(moments[c*nm+a])<=1e-14;
        if(!zero) continue;
        for(int a=0;a<nm;++a) moments[c*nm+a]=draw(rng);
        for(int i=0;i<ns;++i) species[c*ns+i]=draw(rng);
    }
#ifdef IMRMLP_MPI
    MPI_Bcast(species,channels*ns,MPI_DOUBLE,0,mpi.comm);
    MPI_Bcast(moments,channels*nm,MPI_DOUBLE,0,mpi.comm);
#endif
}

LESStage les_stage(Potentials::AnyPotential& potential) {
    LESStage result;
    if(auto* p=dynamic_cast<Potentials::PotMTP_LES*>(&potential)) {
        result.kernel=&p->model_les;result.active=&p->lr_les;result.short_range={"sr_radial","sr_linear"};
        result.initialize=[p](const auto& atoms,const auto& targets,double range,int seed) {
            p->initialize_zero_les_coeffs(atoms,targets,range,seed);
        };
        return result;
    }
    const auto select=[&]<class Head>() {
        auto* p=dynamic_cast<Potentials::PotMTP_M<Head>*>(&potential);
        if(!p) return;
        result.kernel=&p->model_les;result.active=&p->lr_les;
        for(const auto& g:p->nl_parameter_groups())
            if(g.name=="sr_radial" || g.name=="sr_species" || g.name=="sr_moment" || g.name=="sr_model")
                result.short_range.push_back(g.name);
        result.initialize=[p](const auto&,const auto&,double range,int seed) {
            if(!p->uses_sog() || !p->les_species || !p->les_moment)
                GPERROR("MTP-M stages require an initialized active SOG head");
            initialize_sog_heads(p->les_species,p->les_moment,p->species_count,
                p->alpha_scalar_moments,p->les_channels(),range,seed);
        };
    };
    select.template operator()<Potentials::m_model_mtpli>();
    select.template operator()<Potentials::m_model_mtpq2>();
    select.template operator()<Potentials::m_model_mtpfs>();
    select.template operator()<Potentials::m_model_nn1>();
    select.template operator()<Potentials::m_model_nn2>();
    return result;
}
} // namespace


void TrainNonlinear::TrainSingleStage(const std::string &optimizer, int iterations) {
    RequireCurrentParameterLayout();
    Training::collective_try([&] {
        ConfigureTrainableGroups(trainable_groups);
        p_pot->nl_get_coeffs(x.data());
        update_vector.setZero();
        update_vector_last.setZero();
        sgd_scheduler.reset();
        InitPenaltyGard();
    });
    if (optimizer == "bfgs") TrainNonLinearBFGS(iterations);
    else if (optimizer == "cg") TrainNonLinearCG(iterations);
    else if (optimizer == "sgd") TrainNonLinearSGD(iterations);
    else GPERROR("Unknown scheduled optimizer: " + optimizer);
}

void TrainNonlinear::TrainScheduled(const std::string &optimizer) {
    RequireCurrentParameterLayout();
    Training::collective_try([&] {
        if (schedule_active_) GPERROR("Recursive nonlinear training schedule");
        if (optimizer != "bfgs" && optimizer != "cg" && optimizer != "sgd" && optimizer != "shlbfgs")
            GPERROR("Unknown scheduled optimizer: " + optimizer);
        if (!p_pot || n_relax_coeffs <= 0 || p_pot->get_NL_coeffs_count() != n_relax_coeffs)
            GPERROR("Training schedule requires an initialized potential with unchanged coefficient count");
        if ((optimizer == "sgd" || optimizer == "shlbfgs") && curvature_precondition != "none")
            GPERROR("curvature_precondition requires ordinary L-BFGS or CG");
        if (training_schedule != "single") {
            if (linear_refit_settings.mode != "off")
                GPERROR("Staged training requires linear_refit=off");
            if (training_device != "cpu") {
                const bool sog=training_schedule=="mtp_sog_from_scratch" || training_schedule=="mtp_sog_resume";
                const auto model=les_stage(*p_pot);
                if (!sog || !model.sog() || weight_charge>0.
                    || hvp_fit.weight>0. || hessian_fit.weight>0.)
                    GPERROR("GPU stages require active MTP-SOG and E/F/S losses; Coulomb LES and charge/curvature losses require CPU");
            }
            for (int count : {sr_stage_iterations, les_stage_iterations, joint_stage_iterations})
                if (count < 0) GPERROR("Training stage budgets must be nonnegative");
        }
    });

    const auto schedule = training_schedule;
    const auto groups = trainable_groups;
    const auto mask = trainable_mask;
    const auto frozen = frozen_coeffs;
    const auto destination = potfilesave;
    const double charge_weight_before = weight_charge;
    const bool was_initialized = p_pot->inited();
    Eigen::VectorXd initial(n_relax_coeffs);
    auto *les = dynamic_cast<Potentials::PotMTP_LES *>(p_pot);
    auto staged_les=les_stage(*p_pot);
    const bool les_active_before = staged_les.active && *staged_les.active;
    // Optimizer bandwidth coordinates use tan/atan. Preserve physical SOG values
    // too, so a failed run restores them exactly without a round-trip change.
    const auto sog_weights_before = staged_les.kernel ? staged_les.kernel->sog_weights : std::vector<double>{};
    const auto sog_widths_before = staged_les.kernel ? staged_les.kernel->sog_log_bandwidths : std::vector<double>{};
    Training::collective_try([&] { p_pot->nl_get_coeffs(initial.data()); });
    auto restore_settings = [&] {
        training_schedule = schedule;
        trainable_groups = groups;
        potfilesave = destination;
        weight_charge = charge_weight_before;
        schedule_active_ = false;
    };
    schedule_active_ = true;
    // Single-stage optimizer validators retain their established contract.
    training_schedule = "single";
    InvalidateCGResult();
    auto stage = [&](const std::string &label, const std::vector<std::string> &selected,
                     int budget, const std::string &path, bool single = false) {
        trainable_groups = selected;
        potfilesave = path;
        Training::collective_try([&] { ConfigureTrainableGroups(selected); });
        Message("Training stage: " + label + ", optimizer=" + optimizer
            + ", budget=" + std::to_string(budget) + (optimizer == "sgd" ? " epochs" : " iterations"));
        if (single || budget > 0) TrainSingleStage(optimizer, budget);
        if (optimizer == "cg" && (single || budget > 0)) SaveCGResult(path);
        else Training::collective_try([&] { save_current(path); });
    };
    try {
        const auto execute = [&] {
            if (schedule == "single") {
                stage("single", groups, optimizer == "sgd" ? maxepoches : maxiter, destination, true);
                return;
            }
            const bool sog_schedule = schedule == "mtp_sog_from_scratch" || schedule == "mtp_sog_resume";
            const bool les_schedule = schedule == "mtp_les_from_scratch" || schedule == "mtp_les_resume";
            Training::collective_try([&] {
                if (!sog_schedule && !les_schedule) GPERROR("Unknown fit:training_schedule: " + schedule);
                if (!staged_les.kernel) GPERROR("LES stages require ordinary MTP-LES or a supported MTP-M head");
                if (sog_schedule && !staged_les.sog()) GPERROR("SOG schedule requires an initialized active SOG kernel");
                if (les_schedule && !les) GPERROR("Coulomb LES stages require ordinary PotMTP_LES");
                if (les_schedule && staged_les.sog()) GPERROR("SOG requires an mtp_sog training schedule");
                if (sog_schedule && weight_charge > 0.)
                    GPERROR("Generic SOG latent variables do not support physical charge targets");
                if (les_schedule && weight_charge <= 0.)
                    GPERROR("MTP+LES staged training requires fit:charge_weight > 0");
            });
            const bool scratch = schedule.ends_with("from_scratch");
            if (sog_schedule || scratch) {
                Training::collective_try([&] {
                    std::vector<double> targets;
                    if (!sog_schedule)
                        for (const auto &atom : training_set_) targets.push_back(total_charge_target(atom));
                    staged_les.initialize(training_set_, targets,
                        sog_schedule ? sog_zero_init_range : les_zero_init_range, les_zero_init_seed);
                });
            }
            if (scratch) {
                Eigen::VectorXd initialized(n_relax_coeffs), sr(n_relax_coeffs);
                Training::collective_try([&] {
                    p_pot->nl_get_coeffs(initialized.data());
                    sr = initialized;
                    for (const auto &group : p_pot->nl_parameter_groups())
                        if (group.name == "les_species" || group.name == "les_moment")
                            sr.segment(group.begin, group.end - group.begin).setZero();
                    p_pot->nl_set_coeffs(sr.data());
                });
                weight_charge = 0.;
                stage("short range", staged_les.short_range, sr_stage_iterations,
                      destination + ".sr.snapshot");
                Training::collective_try([&] {
                    p_pot->nl_get_coeffs(sr.data());
                    for (const auto &group : p_pot->nl_parameter_groups())
                        if (group.name == "les_species" || group.name == "les_moment")
                            sr.segment(group.begin, group.end - group.begin) =
                                initialized.segment(group.begin, group.end - group.begin);
                    p_pot->nl_set_coeffs(sr.data());
                });
            }
            weight_charge = sog_schedule ? 0. : charge_weight_before * charge_warmup_factor;
            stage(sog_schedule ? "fixed bandwidth warm-up" : "LES warm-up",
                  sog_schedule ? std::vector<std::string>{"les_species", "les_moment", "sog_weights"}
                               : std::vector<std::string>{"les_species", "les_moment"},
                  les_stage_iterations, destination + (sog_schedule ? ".sog.snapshot" : ".les.snapshot"));
            weight_charge = charge_weight_before;
            auto joint_groups=staged_les.short_range;
            joint_groups.insert(joint_groups.end(),{"les_species","les_moment"});
            if(sog_schedule) joint_groups.insert(joint_groups.end(),{"sog_weights","sog_log_bandwidths"});
            stage("joint optimization",joint_groups,
                  joint_stage_iterations > 0 ? joint_stage_iterations : maxiter, destination);
        };
        execute();
        restore_settings();
        Training::collective_try([&] { ConfigureTrainableGroups(groups); });
    } catch (...) {
        restore_settings();
        Training::collective_try([&] {
            if (staged_les.active) *staged_les.active = les_active_before;
            p_pot->nl_set_coeffs(initial.data());
            if (staged_les.kernel) {
                staged_les.kernel->sog_weights = sog_weights_before;
                staged_les.kernel->sog_log_bandwidths = sog_widths_before;
            }
            if (was_initialized) p_pot->set_inited(); else p_pot->set_empty();
            x = initial;
            trainable_mask = mask;
            frozen_coeffs = frozen;
            validation_current_ = false;
            validation_coefficients_.clear();
            InvalidateCGResult();
        });
        throw;
    }
}
