// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potentials/potential_wrapper.h"
#include "io/gpio.h"
#include "tools/constants.h"
#include "utils/comm.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>


using namespace std;

#ifdef USE_OMP
#include <omp.h>
#endif

namespace {
// Called on the main thread at every phase boundary, including empty shards.
void calc_err_check(const std::string &local) {
    int source = local.empty() ? mpi.size : mpi.rank, first = mpi.size;
    MPI_Allreduce(&source, &first, 1, MPI_INT, MPI_MIN, mpi.comm);
    if (first == mpi.size) return;
    int length = mpi.rank == first ? static_cast<int>(local.size()) : 0;
    MPI_Bcast(&length, 1, MPI_INT, first, mpi.comm);
    std::vector<int> message(length);
    if (mpi.rank == first)
        std::copy(local.begin(), local.end(), message.begin());
    MPI_Bcast(message.data(), length, MPI_INT, first, mpi.comm);
    throw GPException("calc_err failure on rank " + std::to_string(first) + ": " +
                      std::string(message.begin(), message.end()));
}
template<class F> void calc_err_phase(const char *phase, F &&operation) {
    std::string error;
    try { operation(); }
    catch (const std::exception &e) { error = std::string(phase) + ": " + e.what(); }
    catch (...) { error = std::string(phase) + ": unknown exception"; }
    calc_err_check(error);
}

// Only E/F/full virial and presence flags travel. The original replicated
// geometry supplies atom counts and root retains its original frame order.
void gather_frame_predictions(std::vector<Atoms> &atoms) {
    constexpr size_t capacity = 65536;
    std::vector<double> send, receive;
    std::vector<int> counts, displacements;
    std::vector<size_t> offsets;
    std::vector<std::array<bool,3>> flags;
    calc_err_phase("gather preparation", [&] {
        send.resize(capacity); receive.resize(capacity);
        counts.resize(mpi.size); displacements.resize(mpi.size);
        offsets.push_back(0);
        for (const auto &atom : atoms) {
            const size_t length = 13 + size_t(atom.size()) * 3;
            if (offsets.back() > std::numeric_limits<size_t>::max()-length)
                GPERROR("prediction payload too large");
            offsets.push_back(offsets.back()+length);
            flags.push_back({atom.has_energy(), atom.has_forces(), atom.has_stresses()});
        }
        if (mpi.rank == 0)
            for (auto &atom : atoms) atom.forces_.resize(atom.size());
    });
    for (size_t offset = 0; offset < offsets.back();) {
        const size_t length = std::min(capacity, offsets.back()-offset);
        for (int rank = 0; rank < mpi.size; ++rank) {
            const size_t begin = std::max(offset, offsets[atoms.size()*rank/mpi.size]);
            const size_t end = std::min(offset+length, offsets[atoms.size()*(rank+1)/mpi.size]);
            counts[rank] = end > begin ? end-begin : 0;
            displacements[rank] = end > begin ? begin-offset : 0;
        }
        const size_t start = offset+displacements[mpi.rank];
        size_t frame = std::upper_bound(offsets.begin(), offsets.end(), start)-offsets.begin()-1;
        for (int k = 0; k < counts[mpi.rank]; ++k) {
            while (start+k >= offsets[frame+1]) ++frame;
            const size_t field = start+k-offsets[frame];
            const auto &atom = atoms[frame];
            if (field < 3) send[k] = flags[frame][field];
            else if (field == 3) send[k] = flags[frame][0] ? atom.energy_ : 0;
            else if (field < 13) send[k] = flags[frame][2] ? atom.stress_((field-4)/3,(field-4)%3) : 0;
            else send[k] = flags[frame][1] ? atom.forces_[(field-13)/3][(field-13)%3] : 0;
        }
        MPI_Gatherv(send.data(), counts[mpi.rank], MPI_DOUBLE, receive.data(),
                    counts.data(), displacements.data(), MPI_DOUBLE, 0, mpi.comm);
        if (mpi.rank == 0) {
            frame = std::upper_bound(offsets.begin(), offsets.end(), offset)-offsets.begin()-1;
            for (size_t k = 0; k < length; ++k) {
                while (offset+k >= offsets[frame+1]) ++frame;
                const size_t field = offset+k-offsets[frame];
                auto &atom = atoms[frame];
                if (field < 3) flags[frame][field] = receive[k] != 0;
                else if (field == 3) atom.energy_ = receive[k];
                else if (field < 13) atom.stress_((field-4)/3,(field-4)%3) = receive[k];
                else atom.forces_[(field-13)/3][(field-13)%3] = receive[k];
            }
        }
        offset += length;
    }
    if (mpi.rank == 0)
        for (size_t frame = 0; frame < atoms.size(); ++frame) {
            atoms[frame].have_energy = flags[frame][0];
            atoms[frame].have_stress = flags[frame][2];
            if (!flags[frame][1]) atoms[frame].forces_.clear();
        }
}
} // namespace

struct ErrorAccumulator {
    size_t natoms = 0;
    size_t nenergy = 0;
    size_t nstress = 0;

    // energy
    double mean_eng_error_accu = 0; //
    double rms_eng_error_accu = 0; //
    double max_eng_error = 0; //

    // energy/at
    double mean_engat_error_accu = 0; //
    double rms_engat_error_accu = 0; //
    double max_engat_error = 0;     //

    // force, err =  nrom(\vec{f1} - \vec{f2})
    double mean_force_error_accu = 0; //
    double rms_force_error_accu = 0; //
    double mean_force_accu = 0; //
    double rms_force_accu = 0; //
    double max_force = 0; //
    double max_force_error = 0; //

    // force cmpt
    double mean_force_cmpt_error_accu = 0;
    double mean_force_cmpt_accu = 0;
    double max_force_cmpt = 0;
    double max_force_cmpt_error = 0;

    // stress
    double mean_stress_error_accu = 0; //
    double rms_stress_error_accu = 0; //
    double max_stress_error = 0; //
    double max_stress = 0; // 
    double mean_stress_accu = 0; // 
    double rms_stress_accu = 0; // 

    // stress cmpt
    double mean_stress_cmpt_accu = 0;
    double rms_stress_cmpt_accu = 0;
    double mean_stress_cmpt_error_accu = 0;
    double rms_stress_cmpt_error_accu = 0;

    // virial
    double mean_virial_accu = 0; //
    double rms_virial_accu = 0; //
    double max_virial_error = 0; // 
    double max_virial = 0; // 
    double mean_virial_error_accu = 0; //
    double rms_virial_error_accu = 0; //

    // virial cmpt
    double mean_virial_cmpt_accu = 0; //
    double rms_virial_cmpt_accu = 0; //
    double mean_virial_cmpt_error_accu = 0; //
    double rms_virial_cmpt_error_accu = 0; //

    // individual errors for percentile computation
    std::vector<double> eng_errors_;
    std::vector<double> engat_errors_;
    std::vector<double> force_errors_;
    std::vector<double> force_cmpt_errors_;
    std::vector<double> stress_errors_;
    std::vector<double> virial_errors_;

    void accumulate(Atoms &ref, Atoms &efs) {
        if (ref.size() == 0 || ref.size() != efs.size())
            GPERROR("calc_err: paired structures must have matching, nonzero atom counts");
        if (ref.chemmap_ != efs.chemmap_)
            GPERROR("calc_err: paired structures must have matching species in atom order");
        if ((ref.has_forces() && ref.forces_.size() != size_t(ref.size())) ||
            (efs.has_forces() && efs.forces_.size() != size_t(efs.size())))
            GPERROR("calc_err: invalid force array length");
        for (const auto *frame : {&ref, &efs}) {
            if ((frame->has_energy() && !std::isfinite(frame->energy_)) ||
                (frame->has_stresses() && !frame->stress_.allFinite()))
                GPERROR("calc_err: energy and virial labels must be finite");
            if (frame->has_forces())
                for (const auto &force : frame->forces_)
                    if (!force.allFinite())
                        GPERROR("calc_err: force labels must be finite");
        }
        // energy
        if (ref.has_energy() && efs.has_energy()) {
            ++nenergy;
            double engdiff = fabs(ref.energy_ - efs.energy_);
            mean_eng_error_accu += engdiff;
            rms_eng_error_accu += engdiff * engdiff;
            max_eng_error = std::max(max_eng_error, engdiff);
            double engatdiff = engdiff / ref.size();
            mean_engat_error_accu += engatdiff;
            rms_engat_error_accu += engatdiff * engatdiff;
            max_engat_error = std::max(engatdiff, max_engat_error);
            eng_errors_.push_back(engdiff);
            engat_errors_.push_back(engatdiff);
        }
        // stress and virial
        if (ref.has_stresses() && efs.has_stresses()) {
            const double volume = std::abs(ref.get_volume());
            if (!std::isfinite(volume) || volume == 0)
                GPERROR("calc_err: stress conversion requires a finite nonzero cell volume");
            ++nstress;
            Eigen::Matrix3d stressdiff = ref.stress_ - efs.stress_;
            stressdiff = stressdiff.cwiseAbs();
            double tt = 1.0 / volume * eVA3_to_GPa;
            Eigen::Matrix3d vref = ref.stress_ * tt;
            Eigen::Matrix3d vefs = efs.stress_ * tt;
            Eigen::Matrix3d vdiff = vref - vefs;
            vdiff = vdiff.cwiseAbs();
            double cserr = stressdiff.norm();
            double cmaxstress = ref.stress_.cwiseAbs().maxCoeff();
            double cverr = vdiff.norm();
            double cvmax = vref.cwiseAbs().maxCoeff();
            max_stress_error = std::max(max_stress_error, stressdiff.cwiseAbs().maxCoeff());
            max_stress = std::max(max_stress, cmaxstress);
            max_virial_error = std::max(max_virial_error, vdiff.cwiseAbs().maxCoeff());
            max_virial = std::max(max_virial, cvmax);
            mean_stress_error_accu += cserr;
            rms_stress_error_accu += stressdiff.squaredNorm();
            mean_stress_accu += ref.stress_.norm();
            rms_stress_accu += ref.stress_.squaredNorm();
            mean_virial_error_accu += cverr;
            rms_virial_error_accu += vdiff.squaredNorm();
            mean_virial_accu += vref.norm();
            rms_virial_accu += vref.squaredNorm();
            // Component statistics use all nine entries of the full tensor.
            mean_stress_cmpt_accu += ref.stress_.cwiseAbs().sum();
            mean_stress_cmpt_error_accu += stressdiff.cwiseAbs().sum();
            rms_stress_cmpt_accu += ref.stress_.squaredNorm();
            rms_stress_cmpt_error_accu += stressdiff.squaredNorm();

            mean_virial_cmpt_accu += vref.cwiseAbs().sum();
            mean_virial_cmpt_error_accu += vdiff.cwiseAbs().sum();
            rms_virial_cmpt_accu += vref.squaredNorm();
            rms_virial_cmpt_error_accu += vdiff.squaredNorm();

            stress_errors_.push_back(cserr);
            virial_errors_.push_back(cverr);
        }
        if (ref.has_forces() && efs.has_forces()) {
            natoms += ref.size();
            for (int i = 0; i < ref.size(); ++i) {
                Eigen::Vector3d fdiff = ref.forces_[i] - efs.forces_[i];
                double fdiffnorm = fdiff.norm();
                mean_force_error_accu += fdiffnorm;
                rms_force_error_accu += fdiff.squaredNorm();
                mean_force_accu += ref.forces_[i].norm();
                rms_force_accu += ref.forces_[i].squaredNorm();
                max_force_error = std::max(max_force_error, fdiffnorm);
                max_force = std::max(max_force, ref.forces_[i].norm());
                max_force_cmpt = std::max({max_force_cmpt, 
                        fabs(ref.forces_[i](0)),
                        fabs(ref.forces_[i](1)),
                        fabs(ref.forces_[i](2))});
                max_force_cmpt_error = std::max({max_force_cmpt_error, 
                        fabs(fdiff(0)),
                        fabs(fdiff(1)),
                        fabs(fdiff(2))});
                mean_force_cmpt_error_accu += fdiff.cwiseAbs().sum();
                mean_force_cmpt_accu += ref.forces_[i].cwiseAbs().sum();
                force_errors_.push_back(fdiffnorm);
                force_cmpt_errors_.push_back(fabs(fdiff(0)));
                force_cmpt_errors_.push_back(fabs(fdiff(1)));
                force_cmpt_errors_.push_back(fabs(fdiff(2)));
            }
        }
    }

    double get_max_energy_error() {
        if (nenergy == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_eng_error;
    }

    double get_avg_energy_error() {
        if (nenergy == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_eng_error_accu / nenergy;
    }

    double get_rms_energy_error() {
        if (nenergy == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_eng_error_accu / nenergy);
    }

    double get_max_energyat_error() {
        if (nenergy == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_engat_error;
    }

    double get_avg_energyat_error() {
        if (nenergy == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_engat_error_accu / nenergy;
    }

    double get_rms_energyat_error() {
        if (nenergy == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_engat_error_accu / nenergy);
    }

    double get_max_force_error() {
        if (natoms == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_force_error;
    }

    double get_avg_force_error() {
        if (natoms == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_force_error_accu / natoms;
    }

    double get_rms_force_error() {
        if (natoms == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_force_error_accu / natoms);
    }

    double get_max_ferr_d_max_f() {
        if (natoms == 0 || max_force == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_force_error / max_force;
    }

    double get_avg_ferr_d_avg_f() {
        if (natoms == 0 || mean_force_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_force_error_accu / mean_force_accu;
    }

    double get_rms_ferr_d_rms_f() {
        if (natoms == 0 || rms_force_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_force_error_accu / rms_force_accu);
    }

    double get_max_force_cmpt_error() {
        if (natoms == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_force_cmpt_error;
    }

    double get_avg_force_cmpt_error() {
        if (natoms == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_force_cmpt_error_accu / natoms / 3.0;
    }

    double get_rms_force_cmpt_error() {
        if (natoms == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_force_error_accu / natoms / 3.0);
    }

    double get_max_fcmpterr_d_max_f() {
        if (natoms == 0 || max_force_cmpt == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_force_cmpt_error / max_force_cmpt;
    }

    double get_avg_fcmpterr_d_avg_f() {
        if (natoms == 0 || mean_force_cmpt_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_force_cmpt_error_accu / mean_force_cmpt_accu;
    }

    double get_rms_fcmpterr_d_rms_f() {
        if (natoms == 0 || rms_force_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_force_error_accu / rms_force_accu);
    }

    double get_max_stress_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_stress_error;
    }

    double get_avg_stress_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_stress_error_accu / nstress;
    }

    double get_rms_stress_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_stress_error_accu / nstress);
    }

    double get_max_serr_d_max_s() {
        if (nstress == 0 || max_stress == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_stress_error / max_stress;
    }

    double get_rms_serr_d_rms_s() {
        if (nstress == 0 || rms_stress_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_stress_error_accu / rms_stress_accu);
    }

    double get_avg_serr_d_avg_s() {
        if (nstress == 0 || mean_stress_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_stress_error_accu / mean_stress_accu;
    }

    double get_avg_stress_cmpt_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_stress_cmpt_error_accu / nstress / 9.0;
    }

    double get_rms_stress_cmpt_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_stress_cmpt_error_accu / nstress / 9.0);
    }

    double get_rms_scmpt_err_d_rms_s() {
        if (nstress == 0 || rms_stress_cmpt_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_stress_cmpt_error_accu / rms_stress_cmpt_accu);
    }

    double get_avg_scmpt_err_d_avg_s() {
        if (nstress == 0 || mean_stress_cmpt_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_stress_cmpt_error_accu / mean_stress_cmpt_accu;
    }

    double get_max_virial_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_virial_error;
    }

    double get_avg_virial_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_virial_error_accu / nstress;
    }

    double get_rms_virial_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_virial_error_accu / nstress);
    }

    double get_max_verr_d_max_v() {
        if (nstress == 0 || max_virial == 0) return std::numeric_limits<double>::quiet_NaN();
        return max_virial_error / max_virial;
    }

    double get_rms_verr_d_rms_v() {
        if (nstress == 0 || rms_virial_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_virial_error_accu / rms_virial_accu);
    }

    double get_avg_verr_d_avg_v() {
        if (nstress == 0 || mean_virial_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_virial_error_accu / mean_virial_accu;
    }

    double get_avg_virial_cmpt_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_virial_cmpt_error_accu / nstress / 9.0;
    }

    double get_rms_virial_cmpt_error() {
        if (nstress == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_virial_cmpt_error_accu / nstress / 9.0);
    }

    double get_rms_vcmpt_err_d_rms_v() {
        if (nstress == 0 || rms_virial_cmpt_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return sqrt(rms_virial_cmpt_error_accu / rms_virial_cmpt_accu);
    }

    double get_avg_vcmpt_err_d_avg_v() {
        if (nstress == 0 || mean_virial_cmpt_accu == 0) return std::numeric_limits<double>::quiet_NaN();
        return mean_virial_cmpt_error_accu / mean_virial_cmpt_accu;
    }

    static double compute_percentile(std::vector<double>& data, double p) {
        if (data.empty()) return std::numeric_limits<double>::quiet_NaN();
        std::sort(data.begin(), data.end());
        double idx = p * (data.size() - 1);
        size_t lo = static_cast<size_t>(idx);
        size_t hi = std::min(lo + 1, data.size() - 1);
        double frac = idx - lo;
        return data[lo] * (1.0 - frac) + data[hi] * frac;
    }

    double get_percentile_energy_error(double p) { return compute_percentile(eng_errors_, p); }
    double get_percentile_energyat_error(double p) { return compute_percentile(engat_errors_, p); }
    double get_percentile_force_error(double p) { return compute_percentile(force_errors_, p); }
    double get_percentile_force_cmpt_error(double p) { return compute_percentile(force_cmpt_errors_, p); }
    double get_percentile_stress_error(double p) { return compute_percentile(stress_errors_, p); }
    double get_percentile_virial_error(double p) { return compute_percentile(virial_errors_, p); }
};

int calc_err(int argc, char *argv[])
{
    auto &cmdopts = CMDOption::global_cmdopts;
    GPIO iobj;
    std::vector<Atoms> atoms;
    string indexfile = "index.map"; // can use with formation energy, first is dft, second is mlp
    string pfile = "";
    string ptype = "mtp";
    string inprd = "";
    string mpi_mode = "replicated";
    string infile = "valid.cfg";
    string informat = "";
    string output_energy_diff;
    string output_energy_at_diff;
    string output_formation_energy_diff;
    string output_force_diff;
    string output_stress_diff;
    string output_virial_diff;
    string output_summary;
    string settingfile;
    GPSettings settings;
    bool show_progress = false;
    double percentile = 0.0;

    cmdopts.addop(indexfile, "index", "id", "Index file mapping atom types to indices")
           .addop(pfile, "potfile", "pot", "Potential file")
           .addop(ptype, "pottype", "pt", "Potential type")
           .addop(infile, "input", "in", "Input structure file")
           .addop(informat, "input-format", "if", "Input file format")
           .addop(mpi_mode, "mpi-mode", "", "MPI evaluation mode: replicated or frames")
           .addop(inprd, "input-predict", "inprd", "Input file containing potential-predicted structures for comparison")
           .addop(output_energy_diff, "output-energy-diff", "oe", "Output file for total energy differences")
           .addop(output_energy_at_diff, "output-energy-at-diff", "oeat", "Output file for per-atom energy differences")
           .addop(output_force_diff, "output-force-diff", "ofce", "Output file for atomic force vector differences")
           .addop(output_stress_diff, "output-stress-diff", "ofst", "Output file for stress tensor differences (6 components)")
           .addop(output_virial_diff, "output-virial-diff", "ofvr", "Output file for virial tensor differences (6 components)")
           .addop(output_summary, "output-summary", "out", "Summary output file (default: stdout if unspecified)")
           .addop(show_progress, "show-progress", "sp", "Report completed structure counts during calculation", false, 0)
           .addop(percentile, "percentile", "q", "Percentile ratio for reporting (e.g., 0.95 for Q95; 0 to disable)")
           .addop(settingfile, "settings", "", "Main configuration settings file", true);
    calc_err_phase("options", [&] {
    cmdopts.phrase_options(argc, argv);
    if (GPIO::file_exist(settingfile))
    {
        Message("Loading settings");
        settings.Load(settingfile);
        settings.Print();
        apply_settings_to_cmdopts(cmdopts, settings);
    }
    if (!std::isfinite(percentile) || percentile < 0 || percentile > 1)
        GPERROR("calc_err: percentile must be finite and between 0 and 1");
    if (mpi_mode != "replicated" && mpi_mode != "frames")
        GPERROR("calc_err: mpi-mode must be replicated or frames");
    });
    calc_err_phase("input", [&] {
    if (GPIO::file_exist(indexfile))
    {
        iobj.readmap(indexfile); // should be with atom radii
    }
    else
    {
        cerr << "ERROR: indexfile not found\n";
    }
    // read atoms
    Message("Reading trajectory file " + infile);
    if (!iobj.read(atoms, infile, informat) || atoms.empty())
        GPERROR("calc_err: failed to read nonempty reference dataset: " + infile);

    });

    std::vector<Atoms> ref;
    calc_err_phase("input preparation", [&] { ref = atoms; });
    size_t progress_done = 0;
    const auto report_progress = [&](size_t total) {
        ++progress_done;
        const size_t interval = 1 + (total - 1) / 100;
        if (progress_done % interval == 0 || progress_done == total)
            Message("Calculating EFS " + std::to_string(progress_done) + "/" +
                std::to_string(total) + " (" + std::to_string(100 * progress_done / total) + "%)");
    };
    if (!inprd.empty()) {
        calc_err_phase("prediction input", [&] {
            if (mpi.rank == 0 || mpi_mode == "replicated") {
                atoms.clear();
                if (!iobj.read(atoms, inprd, informat) || atoms.empty())
                    GPERROR("calc_err: failed to read nonempty prediction dataset: " + inprd);
            }
        });
    } else if (mpi_mode == "frames") {
        const size_t begin = atoms.size() * mpi.rank / mpi.size;
        const size_t end = atoms.size() * (mpi.rank + 1) / mpi.size;
        int workers = 1;
#ifdef USE_OMP
        workers = std::max(1, std::min<int>(omp_num_threads, end-begin));
#ifdef IMRMLP_MPI
        int provided = 0; MPI_Query_thread(&provided);
        if (provided < MPI_THREAD_FUNNELED) workers = 1;
#endif
#endif
        std::vector<std::unique_ptr<Potentials::potential_wrapper>> models;
        std::vector<std::string> failures;
        calc_err_phase("model preparation", [&] {
            failures.resize(atoms.size());
            for (int worker = 0; worker < workers; ++worker) {
                auto model = std::make_unique<Potentials::potential_wrapper>(pfile, ptype, settings.ExtractSubSettings("pot"));
                if (!Potentials::supports_frame_local_efs(*model->pot))
                    GPERROR("calc_err: potential does not support frame-local EFS evaluation; use --mpi-mode replicated");
                models.push_back(std::move(model));
            }
        });
        size_t completed = 0;
#ifdef USE_OMP
#pragma omp parallel for num_threads(workers) schedule(static) reduction(+:completed)
#endif
        for (size_t frame = begin; frame < end; ++frame) {
            try {
                auto &atom = atoms[frame];
                if (atom.size() == 0 || !atom.cell_.allFinite())
                    GPERROR("invalid geometry: nonempty structure and finite cell required");
                if (atom.pbc_[0] && atom.pbc_[1] && atom.pbc_[2] &&
                    (!std::isfinite(atom.get_volume()) || atom.get_volume() == 0))
                    GPERROR("invalid geometry: fully periodic cell must have finite nonzero volume");
                for (const auto &position : atom.pos_)
                    if (!position.allFinite()) GPERROR("invalid geometry: nonfinite position");
                atom.wrap();
                int worker = 0;
#ifdef USE_OMP
                worker = omp_get_thread_num();
#endif
                models[worker]->pot->getEFS(atom);
                ErrorAccumulator check; check.accumulate(ref[frame], atom);
                ++completed;
                if (show_progress && mpi.rank == 0) {
#ifdef USE_OMP
#pragma omp critical(calc_err_progress)
#endif
                    { report_progress(end - begin); }
                }
            } catch (const std::exception &e) {
                failures[frame] = "frame " + std::to_string(frame) + ": " + e.what();
            } catch (...) { failures[frame] = "frame " + std::to_string(frame) + ": unknown exception"; }
        }
        std::string failure;
        for (size_t frame = begin; frame < end; ++frame)
            if (!failures[frame].empty()) { failure = "evaluation: " + failures[frame]; break; }
        calc_err_check(failure);
        double local_count = completed, total_count = 0;
        MPI_Reduce(&local_count, &total_count, 1, MPI_DOUBLE, MPI_SUM, 0, mpi.comm);
        if (mpi.rank == 0)
            Message("calc_err frame evaluations: " + std::to_string(static_cast<size_t>(total_count)));
        gather_frame_predictions(atoms);
    } else {
#ifdef USE_OMP
        if (atoms.size() < omp_num_threads) omp_num_threads = atoms.size();
        omp_set_num_threads(omp_num_threads);
#pragma omp parallel
#endif
        {
            Potentials::potential_wrapper wrapper(pfile, ptype, settings.ExtractSubSettings("pot"));
#ifdef USE_OMP
#pragma omp for
#endif
            for (size_t i = 0; i < atoms.size(); ++i) {
                atoms[i].wrap();
                wrapper.pot->getEFS(atoms[i]);
                if (show_progress && mpi.rank == 0) {
#ifdef USE_OMP
#pragma omp critical(calc_err_progress)
#endif
                    { report_progress(atoms.size()); }
                }
            }
        }
    }
    ErrorAccumulator errors;
    calc_err_phase("statistics", [&] {
        if (mpi.rank != 0) return;
        if (atoms.size() != ref.size())
            GPERROR("calc_err: reference and prediction frame counts differ");
        for (size_t i = 0; i < atoms.size(); ++i) errors.accumulate(ref[i], atoms[i]);
    });
    calc_err_phase("output", [&] {
    if (mpi.rank != 0) return;

    auto open_output = [](const string &path) {
        ofstream output(path);
        if (!output)
            GPERROR("calc_err: failed to open output file: " + path);
        return output;
    };
    auto finish_output = [](ofstream &output, const string &path) {
        output.close();
        if (!output)
            GPERROR("calc_err: failed to write output file: " + path);
    };
    ofstream *ofs_summ = nullptr;
    unique_ptr<ScopedStreamRedirect> summary_redirect;
    if (!output_summary.empty())
    {
        auto output = make_unique<ofstream>(open_output(output_summary));
        ofs_summ = output.get();
        summary_redirect = make_unique<ScopedStreamRedirect>(global_ostream, output.release());
    }

    auto metric = [](double value) {
        return std::isfinite(value) ? to_string(value) : std::string("N/A");
    };
    auto print_pct = [&](const std::string& label, double val) {
        if (percentile > 0)
            Message("\tP" + to_string(int(percentile * 100)) + "    absolute difference = " + metric(val));
    };

    Message("\nEnergy (eV): ");
    Message("\tErrors checked for " + to_string(errors.nenergy) + " configurations");
    Message("\tMaximal absolute difference = " + metric(errors.get_max_energy_error()));
    Message("\tAverage absolute difference = " + metric(errors.get_avg_energy_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_energy_error()));
    print_pct("energy", errors.get_percentile_energy_error(percentile));
    Message("\nEnergy per atom (eV):");
    Message("\tErrors checked for " + to_string(errors.nenergy) + " configurations");
    Message("\tMaximal absolute difference = " + metric(errors.get_max_energyat_error()));
    Message("\tAverage absolute difference = " + metric(errors.get_avg_energyat_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_energyat_error()));
    print_pct("energy/atom", errors.get_percentile_energyat_error(percentile));
    Message("\nForces (eV/A):");
    Message("\tErrors checked for " + to_string(errors.natoms) + " atoms");
    Message("\tMaximal absolute difference = " + metric(errors.get_max_force_error()));
    Message("\tAverage absolute difference = " + metric(errors.get_avg_force_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_force_error()));
    print_pct("force", errors.get_percentile_force_error(percentile));
    Message("\tMax(ForceDiff) / Max(Force) = " + metric(errors.get_max_ferr_d_max_f()));
    Message("\tRMS(ForceDiff) / RMS(FORCE) = " + metric(errors.get_rms_ferr_d_rms_f()));
    Message("\tMEAN(ForceDiff)/MEAN(FORCE) = " + metric(errors.get_avg_ferr_d_avg_f()));
    Message("\nForces components (eV/A):");
    Message("\tErrors checked for " + to_string(errors.natoms) + " atoms");
    Message("\tMaximal absolute difference = " + metric(errors.get_max_force_cmpt_error()));
    Message("\tAverage absolute difference = " + metric(errors.get_avg_force_cmpt_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_force_cmpt_error()));
    print_pct("force cmpt", errors.get_percentile_force_cmpt_error(percentile));
    Message("\tMax(ForceDiff) / Max(Force) = " + metric(errors.get_max_fcmpterr_d_max_f()));
    Message("\tRMS(ForceDiff) / RMS(FORCE) = " + metric(errors.get_rms_fcmpterr_d_rms_f()));
    Message("\tMEAN(ForceDiff)/MEAN(FORCE) = " + metric(errors.get_avg_fcmpterr_d_avg_f()));
    Message("\nStress (eV):");
    Message("\tErrors checked for " + to_string(errors.nstress) + " configurations");
    Message("\tMax     absolute difference = " + metric(errors.get_max_stress_error()));
    Message("\tAverage absolute difference = " + metric(errors.get_avg_stress_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_stress_error()));
    print_pct("stress", errors.get_percentile_stress_error(percentile));
    Message("\tMax(StressDiff)/Max(Stress) = " + metric(errors.get_max_serr_d_max_s()));
    Message("\tRMS(StressDiff)/RMS(Stress) = " + metric(errors.get_rms_serr_d_rms_s()));
    Message("\tMEAN(StressDif)/MEAN(Stres) = " + metric(errors.get_avg_serr_d_avg_s()));
    Message("\nStress components (eV):");
    Message("\tComponent statistics use all 9 tensor entries");
    Message("\tErrors checked for " + to_string(errors.nstress) + " configurations");
    Message("\tAverage absolute difference = " + metric(errors.get_avg_stress_cmpt_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_stress_cmpt_error()));
    Message("\tRMS(StressDiff)/RMS(Stress) = " + metric(errors.get_rms_scmpt_err_d_rms_s()));
    Message("\tMEAN(StressDif)/MEAN(Stres) = " + metric(errors.get_avg_scmpt_err_d_avg_s()));
    Message("\nVirial stresses (GPa):");
    Message("\tErrors checked for " + to_string(errors.nstress) + " configurations");
    Message("\tMax     absolute difference = " + metric(errors.get_max_virial_error()));
    Message("\tAverage absolute difference = " + metric(errors.get_avg_virial_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_virial_error()));
    print_pct("virial", errors.get_percentile_virial_error(percentile));
    Message("\tMax(VirialDiff)/Max(Virial) = " + metric(errors.get_max_verr_d_max_v()));
    Message("\tRMS(VirialDiff)/RMS(Virial) = " + metric(errors.get_rms_verr_d_rms_v()));
    Message("\tMEAN(VirialDif)/MEAN(Viria) = " + metric(errors.get_avg_verr_d_avg_v()));
    Message("\nVirial components (GPa):");
    Message("\tComponent statistics use all 9 tensor entries");
    Message("\tErrors checked for " + to_string(errors.nstress) + " configurations");
    Message("\tAverage absolute difference = " + metric(errors.get_avg_virial_cmpt_error()));
    Message("\tRMS     absolute difference = " + metric(errors.get_rms_virial_cmpt_error()));
    Message("\tRMS(VirialDiff)/RMS(Virial) = " + metric(errors.get_rms_vcmpt_err_d_rms_v()));
    Message("\tMEAN(VirialDif)/MEAN(Viria) = " + metric(errors.get_avg_vcmpt_err_d_avg_v()));

    if (ofs_summ)
        finish_output(*ofs_summ, output_summary);
    summary_redirect.reset();

    // MLP-DFT energy
    if (!output_energy_diff.empty())
    {
        auto ofs = open_output(output_energy_diff);
        ofs << "# dft-total mlp-total\n";
        ofs << setprecision(15) << fixed;
        for (int i = 0; i < atoms.size(); ++i)
        {
            if (!ref[i].has_energy() || !atoms[i].has_energy()) continue;
            ofs << ref[i].energy_ << "  " << atoms[i].energy_ << "\n";
        }
        finish_output(ofs, output_energy_diff);
    }
    if (!output_energy_at_diff.empty())
    {
        auto ofs = open_output(output_energy_at_diff);
        ofs << "#dft-pat mlp-pat\n";
        ofs << setprecision(15) << fixed;
        for (int i = 0; i < atoms.size(); ++i)
        {
            if (!ref[i].has_energy() || !atoms[i].has_energy()) continue;
            ofs << ref[i].energy_ / ref[i].size() << " " << atoms[i].energy_ / atoms[i].size() << "\n";
        }
        finish_output(ofs, output_energy_at_diff);
    }
    // MLP-DFT formation energy
    // MLP-DFT force
    if (!output_force_diff.empty())
    {
        auto ofs = open_output(output_force_diff);
        auto ofscmpt = open_output(output_force_diff + "cmpt");
        ofscmpt << "# dft mlp\n";
        ofs << "# dft mlp\n";
        ofs << setprecision(15) << fixed;
        for (int i = 0; i < atoms.size(); ++i)
        {
            if (!ref[i].has_forces() || !atoms[i].has_forces()) continue;
            for (int j = 0; j < atoms[i].size(); ++j)
            {
                // int sign = ref[i].forces_[j][2] < 0 ? -1 : 1;
                // ofs << (double)sign * ref[i].forces_[j].norm() << "  " << (double)sign * atoms[i].forces_[j].norm() << "\n";
                ofs << ref[i].forces_[j].norm() << "  " << atoms[i].forces_[j].norm() << "\n";
                for (int a = 0; a < 3; ++a) {
                    ofscmpt << ref[i].forces_[j](a) << " " << atoms[i].forces_[j](a) << "\n";
                }
            }
        }
        finish_output(ofs, output_force_diff);
        finish_output(ofscmpt, output_force_diff + "cmpt");
    }
    if (!output_stress_diff.empty()) {
        auto ofs = open_output(output_stress_diff);
        ofs << "# dft mlp\n";
        ofs << setprecision(15) << fixed;
        for (int i = 0; i < atoms.size(); ++i) {
            if (!ref[i].has_stresses() || !atoms[i].has_stresses()) continue;
            for (int j = 0; j < 3; ++j) {
                for (int k = j; k < 3; ++k) {
                    ofs << ref[i].stress_(j, k) << " " << atoms[i].stress_(j, k) << "\n";
                }
            }
        }
        finish_output(ofs, output_stress_diff);
    }
    if (!output_virial_diff.empty()) {
        auto ofs = open_output(output_virial_diff);
        ofs << "# dft mlp\n";
        ofs << setprecision(15) << fixed;
        
        for (int i = 0; i < atoms.size(); ++i) {
            if (!ref[i].has_stresses() || !atoms[i].has_stresses()) continue;
            double tt = 1.0 / std::abs(ref[i].get_volume()) * eVA3_to_GPa;
            Eigen::Matrix3d mref = ref[i].stress_ * tt;
            Eigen::Matrix3d mefs = atoms[i].stress_ * tt;
            for (int j = 0; j < 3; ++j) {
                for (int k = j; k < 3; ++k) {
                    ofs << mref(j, k) << " " << mefs(j, k) << '\n';
                }
            }
        }
        finish_output(ofs, output_virial_diff);
    }
    Message("Complete!");
    });
    return 0;
}
