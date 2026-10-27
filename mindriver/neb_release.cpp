#include "neb.h"
#include "neb_perturbation.h"
#include "neb_symmetry.h"
#include "neb_reporting.h"
#include "neb_evaluate.h"
#include <filesystem>
#include <random>
#include <numbers>
#include <limits>

namespace {
struct NoReleaseDisplacement : std::runtime_error { using std::runtime_error::runtime_error; };
double barrier(const NEBDriver &driver) {
    double maximum=-std::numeric_limits<double>::infinity();
    for(const auto &a:driver.images) maximum=std::max(maximum,a.energy_);
    return maximum-driver.images.front().energy_;
}
void save_path(const NEBDriver &driver, const std::filesystem::path &folder, GPIO *io) {
    // POSCAR needs no external element registry; retain a caller's registry when available.
    GPIO local; if(!io) io=&local;
    for(size_t i=0;i<driver.images.size();++i) {
        auto image=driver.images[i];
        if(!io->write_one(image,(folder/("image-"+std::to_string(i)+".vasp")).string(),"POSCAR"))
            throw std::runtime_error("Cannot write symmetry release path");
    }
}
}

void NEBDriver::check_symmetry_release(const RunOptions &options) {
    using namespace NEBReporting;
    namespace fs=std::filesystem;
    symmetry_release_status="disabled";
    if(symmetry_release.empty()) return;
    symmetry_release_status="failed";
    std::ofstream report;
    try {
        if(symmetry_release_trials<0 || symmetry_release_trials>1000 || symmetry_release_seed<0 ||
           symmetry_release_seed>std::numeric_limits<int>::max()-symmetry_release_trials ||
           symmetry_release_steps<1 || symmetry_release_evaluations<1 ||
           !std::isfinite(symmetry_release_amplitude) || symmetry_release_amplitude<0 ||
           !std::isfinite(symmetry_release_strain) || symmetry_release_strain<0 ||
           !std::isfinite(symmetry_release_energy_tol) || symmetry_release_energy_tol<0)
            throw std::invalid_argument("Invalid symmetry release budgets, seed or amplitudes");
        const fs::path folder(symmetry_release);
        if(!fs::create_directory(folder)) throw std::runtime_error("Symmetry release directory already exists");
        report.open(folder/"report.json");
        if(!report) throw std::runtime_error("Cannot open symmetry release report");
        report<<std::setprecision(17)<<"{\n\"schema_version\":1,\n\"trial_solver_policy\":\"stable-v1\",\n\"energy_quantity\":\""<<(vcneb?"enthalpy":"energy")
              <<"\",\n\"barrier_unit\":\"eV_total\",\n\"baseline_barrier\":";
        number(report,barrier(*this));
        report<<",\n\"baseline_residual\":";number(report,max_neb_force);
        report<<",\n\"energy_tolerance\":"<<symmetry_release_energy_tol
              <<",\n\"iteration_budget\":"<<symmetry_release_steps<<",\n\"band_evaluation_budget\":"<<symmetry_release_evaluations
              <<",\n\"atomic_amplitude_A\":"<<symmetry_release_amplitude<<",\n\"cell_deformation_bound\":"<<symmetry_release_strain;
        std::string skip;
        if(!keep_symm || !symmetry_) skip="requires keep_symm and an initialized common subgroup";
        else if(!converged) skip="main calculation did not converge";
        else if(al_pot_) skip="online active learning is unsupported";
        else if(!external_stress.isZero() || !external_stress_str.empty()) skip="external stress is unsupported";
        else if(symmetry_->atomic_dofs()==npx && symmetry_->cell_dofs()==vcoffset) skip="no excluded symmetry modes";
        if(!skip.empty()) {
            symmetry_release_status="skipped";
            report<<",\n\"status\":\"skipped\",\n\"reason\":";string(report,skip);report<<",\n\"trials\":[]\n}\n";
        } else {
            save_path(*this,folder,iobj);
            bool lower=false, failed=false, incomplete=false;
            report<<",\n\"trials\":[\n";
            for(int index=0;index<=symmetry_release_trials;++index) {
                if(index) report<<",\n";
                const auto start=Clock::now();
                const fs::path trial_folder=folder/("trial-"+std::to_string(index));
                report<<"{\"index\":"<<index<<",\"seed\":";
                if(index) report<<symmetry_release_seed+index-1;else report<<"null";
                NEBDriver trial;
                long long attempted_images=0;
                try {
                    fs::create_directory(trial_folder);
                    trial.ApplySettings(CurrentSettings());
                    trial.debug_level=0; // Do not reopen the main driver's global debug stream.
                    trial.symmetry_release.clear();trial.symmetry_diagnostics.clear();trial.diagnostics_file.clear();
                    trial.keep_symm=false;trial.symmetry_operations.clear();trial.perturb_amplitude=0;
                    trial.check_mic=false;trial.init_method="linear";trial.variable_image=false;
                    trial.write_traj=false;trial.pot=pot;
                    trial.efs_evaluator=[&](std::vector<Atoms> &batch,int begin,int end) {
                        attempted_images+=end-begin;
                        if(this->efs_evaluator) this->efs_evaluator(batch,begin,end);
                        else evaluate_neb_images(*pot,batch,begin,end,omp_mode);
                    };
                    trial.iref=-3;trial.refcell=refcell;trial.ciival=ciival;
                    trial.max_force_evaluations=symmetry_release_evaluations;
                    trial.solver_policy="stable-v1"; // The bounded diagnostic controller, independent of the main policy.
                    // A converged climbing path must not repeat the initial delay.
                    trial.climb_delay=climbing_active?0:climb_delay;
                    auto path=images;trial.init(path);
                    // Preserve the labelled endpoint data and the exact main coordinate frame.
                    trial.x=x;trial.images=images;
                    double actual_atom=0,actual_cell=0;
                    if(index) {
                        std::tie(actual_atom,actual_cell)=NEBPathSeeds::release(trial,*symmetry_,
                            symmetry_release_seed+index-1,symmetry_release_amplitude,symmetry_release_strain);
                        if(actual_atom<1e-14 && actual_cell<1e-14) {
                            throw NoReleaseDisplacement("no nontranslation perturbation in requested modes");
                        }
                    }
                    save_path(trial,trial_folder,iobj);
                    auto run_options=options;
                    run_options.max_iterations=symmetry_release_steps;run_options.accepted_step_offset=0;
                    trial.run(run_options);
                    const auto final_folder=trial_folder/"final";fs::create_directory(final_folder);
                    save_path(trial,final_folder,iobj);
                    const double change=barrier(trial)-barrier(*this);
                    lower|=trial.converged && change < -symmetry_release_energy_tol;
                    const bool numerical=trial.stop_reason==StopReason::numerical_failure;
                    failed|=numerical;incomplete|=!trial.converged && !numerical;
                    report<<",\"status\":";string(report,trial.stop_reason_string());
                    report<<",\"converged\":"<<(trial.converged?"true":"false")<<",\"barrier\":";number(report,barrier(trial));
                    report<<",\"barrier_change\":";number(report,change);
                    report<<",\"max_neb_force\":";number(report,trial.max_neb_force);
                    report<<",\"atomic_displacement_A\":"<<actual_atom<<",\"cell_deformation\":"<<actual_cell;
                } catch(const NoReleaseDisplacement &error) {
                    incomplete=true;report<<",\"status\":\"skipped\",\"reason\":";string(report,error.what());
                } catch(const std::exception &error) {
                    failed=true;report<<",\"status\":\"failed\",\"reason\":";string(report,error.what());
                } catch(...) {
                    failed=true;report<<",\"status\":\"failed\",\"reason\":\"unknown calculator exception\"";
                }
                report<<",\"force_evaluations\":"<<trial.force_evaluations<<",\"image_force_evaluations\":"<<trial.image_force_evaluations
                      <<",\"attempted_image_evaluations\":"<<attempted_images<<",\"optimizer_iterations\":"<<trial.optimizer_iterations
                      <<",\"wall_seconds\":"<<seconds(start)<<"}";report.flush();
            }
            symmetry_release_status=lower?"lower_barrier_found":failed?"failed":incomplete?"inconclusive":"no_lower_barrier_observed";
            report<<"\n],\n\"status\":";string(report,symmetry_release_status);
            report<<",\n\"has_failed_trials\":"<<(failed?"true":"false")<<",\n\"has_incomplete_trials\":"<<(incomplete?"true":"false")
                  <<",\n\"limitation\":\"Finite local release trials; neither saddle certification nor global path minimality\"\n}\n";
        }
        report.close();if(!report) throw std::runtime_error("Failed to write symmetry release report");
        Message("NEB symmetry release: "+symmetry_release_status+" ("+symmetry_release+")");
    } catch(const std::exception &error) {
        symmetry_release_status="failed";
        Warning("NEB symmetry release failed; main result preserved: "+std::string(error.what()));
    } catch(...) {
        symmetry_release_status="failed";
        Warning("NEB symmetry release failed; main result preserved: unknown exception");
    }
}
