// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_msd.h"
#include "tools/chem.h"
#include "tools/tools.h"
#include "tools/constants.h"
#include "trajectory_transform.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
Eigen::Matrix3d msd_coordinate_map(const Atoms &frame)
{
    // All-NPBC trajectories return before preparation. Preserve the original
    // full-rank coordinates when an NPBC frame occurs in a mixed-PBC trajectory.
    if (frame.is_nonperiodic() && frame.cell_.fullPivLu().isInvertible())
        return frame.cell_.inverse();
    return trajectory_periodic_coordinate_map(frame);
}

void validate_msd_trajectory(const std::vector<Atoms> &atoms)
{
    if (atoms.empty())
        throw std::invalid_argument("MSD requires at least one trajectory frame");
    if (atoms[0].size() == 0)
        throw std::invalid_argument("MSD requires at least one atom");

    const int natoms = atoms[0].size();
    for (int i = 0; i < (int)atoms.size(); ++i) {
        if (atoms[i].size() != natoms)
            throw std::invalid_argument("MSD requires a constant atom count across frames");
        if ((int)atoms[i].chemmap_.size() != natoms)
            throw std::invalid_argument("MSD requires chemmap size to match atom count");
        if ((int)atoms[i].pos_.size() != natoms)
            throw std::invalid_argument("MSD requires position size to match atom count");
        if (atoms[i].chemmap_ != atoms[0].chemmap_)
            throw std::invalid_argument("MSD requires stable atom types and ordering across frames");
        if (!atoms[i].cell_.allFinite() ||
            (!atoms[i].cartesian && std::abs(atoms[i].cell_.determinant()) <= 1e-14))
            throw std::invalid_argument("MSD requires a finite, non-singular cell in every frame");
        if (atoms[i].cartesian)
            (void)trajectory_periodic_coordinate_map(atoms[i]);
        for (const auto &pos : atoms[i].pos_)
            if (!pos.allFinite())
                throw std::invalid_argument("MSD positions must be finite");
    }
}

bool finite_nonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

void write_fit_metadata(std::ofstream &ofs, const ComputeMSD::DiffusionFitResult &fit,
                        double timestep)
{
    if (!fit.fitted) return;
    ofs << "# fit_start " << fit.begin * timestep
        << " fit_end " << (fit.end - 1) * timestep
        << " slope " << fit.slope << " intercept " << fit.intercept
        << " slope_stderr " << fit.slope_stderr << '\n';
    ofs << "# D_PBC " << fit.D_PBC
        << " D_correction " << fit.D_correction
        << " D_infinite " << fit.D_infinite << " [A^2/ps]\n";
}
}

// ── Helper: normalize weights for a subset of atoms ──────────────────
void ComputeMSD::normalize_weights(std::vector<double> &wgt, double &total,
                                    const std::vector<double> &atomic_weight,
                                    const std::vector<int> &indices) {
    total = 0;
    for (int idx : indices) {
        if (idx < 0 || idx >= (int)atomic_weight.size())
            throw std::out_of_range("MSD weight index is out of range");
        if (!finite_nonnegative(atomic_weight[idx]))
            throw std::runtime_error("MSD weights must be finite and non-negative");
        total += atomic_weight[idx];
    }
    if (indices.empty())
        throw std::runtime_error("MSD selection must contain at least one atom");
    if (!std::isfinite(total) || total <= 0.0)
        throw std::runtime_error("MSD weights must sum to a positive value");
    if (total > 0)
        for (int idx : indices) wgt[idx] = atomic_weight[idx] / total;
}

void ComputeMSD::reset_run_state()
{
    nsteps_total = nframes_total = natoms = group_size = track_size = 0;
    chemmap.clear(); tbc.clear(); track_filters.clear(); track_inds.clear();
    types_tbc.clear(); types_tbccount.clear(); group_idxs.clear();
    cm.clear(); vm.clear(); positions.clear(); time.clear();
    msd_x.clear(); msd_y.clear(); msd_z.clear();
    msd_total.clear(); msd_total_d.clear(); msd_total_s.clear();
    diffusion_total_inst.clear(); diffusion_type_inst.clear();
    diffusion_track_inst.clear(); diffusion_group_inst.clear();
    msd_total_x.clear(); msd_total_y.clear(); msd_total_z.clear();
    msd_total_x_d.clear(); msd_total_y_d.clear(); msd_total_z_d.clear();
    msd_total_x_s.clear(); msd_total_y_s.clear(); msd_total_z_s.clear();
    msd_type.clear(); msd_type_d.clear(); msd_type_s.clear();
    msd_type_x.clear(); msd_type_y.clear(); msd_type_z.clear();
    msd_type_x_d.clear(); msd_type_y_d.clear(); msd_type_z_d.clear();
    msd_type_x_s.clear(); msd_type_y_s.clear(); msd_type_z_s.clear();
    msd_track.clear(); msd_track_d.clear(); msd_track_s.clear();
    msd_group.clear(); msd_group_d.clear(); msd_group_s.clear();
    msd_group_x.clear(); msd_group_y.clear(); msd_group_z.clear();
    msd_group_x_d.clear(); msd_group_y_d.clear(); msd_group_z_d.clear();
    msd_group_x_s.clear(); msd_group_y_s.clear(); msd_group_z_s.clear();
    msd_x_s.clear(); msd_y_s.clear(); msd_z_s.clear();
    msd_x_d.clear(); msd_y_d.clear(); msd_z_d.clear();
    k_total = b_total = std::numeric_limits<double>::quiet_NaN();
    k_type.clear(); b_type.clear(); k_track.clear(); b_track.clear();
    k_group.clear(); b_group.clear();
    diffusion_total = {}; diffusion_type.clear(); diffusion_track.clear(); diffusion_group.clear();
}

void ComputeMSD::validate_settings_and_input(const std::vector<Atoms> &atoms) const
{
    validate_msd_trajectory(atoms);
    if (!std::isfinite(ts) || ts <= 0.0)
        throw std::invalid_argument("MSD timestep must be finite and positive");
    if (max_length == 0 || max_length < -1)
        throw std::invalid_argument("MSD max_length must be -1 or positive");
    if (drift_cm && drift_vm)
        throw std::invalid_argument("MSD drift_cm and drift_vm are mutually exclusive");
    if (fsc_mode < 0 || fsc_mode > 2)
        throw std::invalid_argument("MSD fsc_mode must be 0, 1, or 2");
    if (fsc_mode > 0) {
        for (const auto &frame : atoms)
            if (!frame.is_fully_periodic())
                throw std::invalid_argument("MSD finite-size correction requires fully periodic boundaries in every frame");
        if (!do_lfit)
            throw std::invalid_argument("MSD finite-size correction requires do_lfit=true");
        if (!std::isfinite(temperature) || temperature <= 0.0 ||
            !std::isfinite(viscosity) || viscosity <= 0.0 ||
            !std::isfinite(fsc_xi) || fsc_xi <= 0.0)
            throw std::invalid_argument("MSD finite-size correction parameters must be finite and positive");
    }
    if (drift_cm || drift_vm) {
        if (atom_masses.empty())
            throw std::invalid_argument("MSD drift correction requires atom_masses");
        for (int type : atoms[0].chemmap_) {
            if (type < 0 || type >= (int)atom_masses.size() ||
                !std::isfinite(atom_masses[type]) || atom_masses[type] <= 0.0)
                throw std::invalid_argument("MSD atom_masses must cover all types with finite positive masses");
        }
    }
    if (drift_vm) {
        for (const auto &frame : atoms) {
            if ((int)frame.velocities_.size() != frame.size())
                throw std::invalid_argument("MSD drift_vm requires velocities in every frame");
            for (const auto &v : frame.velocities_)
                if (!v.allFinite())
                    throw std::invalid_argument("MSD velocities must be finite");
        }
    }
}

void ComputeMSD::prepare_positions(const std::vector<Atoms> &atoms)
{
    positions.assign(nframes_total, std::vector<Eigen::RowVector3d>(natoms));
    if (std::all_of(atoms.begin(), atoms.end(),
        [](const Atoms &frame) { return frame.is_nonperiodic(); })) {
        for (int frame = 0; frame < nframes_total; ++frame)
            for (int i = 0; i < natoms; ++i) {
                positions[frame][i] = atoms[frame].pos_[i];
                if (!atoms[frame].cartesian)
                    positions[frame][i] *= atoms[frame].cell_;
            }
        return;
    }
    std::vector<Eigen::RowVector3d> images(natoms, Eigen::RowVector3d::Zero());
    std::vector<Eigen::RowVector3d> previous_fractional(natoms);
    const Eigen::Matrix3d inv0 = msd_coordinate_map(atoms[0]);
    for (int i = 0; i < natoms; ++i) {
        // Keep the source representation; MSD works with Cartesian scratch.
        if (atoms[0].cartesian) {
            positions[0][i] = atoms[0].pos_[i];
            previous_fractional[i] = atoms[0].pos_[i] * inv0;
        } else {
            positions[0][i] = atoms[0].pos_[i] * atoms[0].cell_;
            previous_fractional[i] = atoms[0].pos_[i];
        }
    }
    for (int frame = 1; frame < nframes_total; ++frame) {
        const Eigen::Matrix3d invcell = msd_coordinate_map(atoms[frame]);
        for (int i = 0; i < natoms; ++i) {
            Eigen::RowVector3d fractional = atoms[frame].pos_[i];
            if (atoms[frame].cartesian) fractional *= invcell;
            const Eigen::RowVector3d delta = fractional - previous_fractional[i];
            for (int a = 0; a < 3; ++a)
                if (atoms[frame].pbc_[a])
                    images[i][a] -= std::round(delta[a]);
            if (atoms[frame].cartesian)
                positions[frame][i] = atoms[frame].pos_[i] + images[i] * atoms[frame].cell_;
            else
                positions[frame][i] = (fractional + images[i]) * atoms[frame].cell_;
            previous_fractional[i] = fractional;
        }
    }
}

void ComputeMSD::apply_drift(const std::vector<Atoms> &atoms)
{
    if (!drift_cm && !drift_vm) return;
    std::vector<Eigen::RowVector3d> correction(nframes_total, Eigen::RowVector3d::Zero());
    if (drift_cm) {
        cm.assign(nframes_total, Eigen::RowVector3d::Zero());
        double total_mass = 0.0;
        for (int i = 0; i < natoms; ++i) total_mass += atom_masses[chemmap[i]];
        for (int frame = 0; frame < nframes_total; ++frame) {
            for (int i = 0; i < natoms; ++i)
                cm[frame] += atom_masses[chemmap[i]] * positions[frame][i];
            cm[frame] /= total_mass;
            correction[frame] = cm[frame];
        }
    } else {
        vm.resize(nframes_total);
        for (int frame = 0; frame < nframes_total; ++frame)
            vm[frame] = ::get_center_of_mass_v(atoms[frame], atom_masses);
        for (int frame = 1; frame < nframes_total; ++frame)
            correction[frame] = correction[frame - 1] + 0.5 * ts * (vm[frame - 1] + vm[frame]);
    }
    for (int frame = 0; frame < nframes_total; ++frame)
        for (int i = 0; i < natoms; ++i)
            positions[frame][i] -= correction[frame];
}

// ── Allocate output buffers ───────────────────────────────────────────
void ComputeMSD::allocate_buffers(const std::vector<Atoms> &atoms) {
    nframes_total = (int)atoms.size();
    nsteps_total = max_length < 0 ? nframes_total : std::min(max_length, nframes_total);

    time.resize(nsteps_total);
    for (int i = 0; i < nsteps_total; ++i) time[i] = ts * i;

    if (compute_total) {
        msd_total.assign(nsteps_total, 0);
        if (proj) msd_total_x = msd_total_y = msd_total_z = msd_total;
    }
    if (compute_partial) {
        int ntypes = (int)types_tbc.size();
        types_tbccount = atoms[0].types_count_map();
        if (ntypes == 0) {
            for (auto &it : types_tbccount) types_tbc.push_back(it.first);
            ntypes = (int)types_tbc.size();
        }
        msd_type.resize(ntypes);
        for (auto &v : msd_type) v.assign(nsteps_total, 0);
        if (proj) { msd_type_x = msd_type_y = msd_type_z = msd_type; }
    }
    if (compute_track) {
        if (track_inds.empty()) {
            track_inds.resize(natoms);
            std::iota(track_inds.begin(), track_inds.end(), 0);
        }
        track_size = (int)track_inds.size();
        track_filters.assign(natoms, 0);
        for (int it : track_inds) track_filters[it] = 1;
        msd_track.resize(track_size);
        for (auto &v : msd_track) v.assign(nsteps_total, 0);
    }
    if (compute_groups && !group_idxs.empty()) {
        msd_group.resize(group_size);
        for (auto &v : msd_group) v.assign(nsteps_total, 0);
        if (proj) { msd_group_x = msd_group_y = msd_group_z = msd_group; }
    } else {
        compute_groups = false;
    }

    msd_x.resize(natoms);
    if (proj) { msd_y.resize(natoms); msd_z.resize(natoms); }
    for (int i = 0; i < natoms; ++i) {
        msd_x[i].assign(nsteps_total, 0);
        if (proj) {
            msd_y[i].assign(nsteps_total, 0);
            msd_z[i].assign(nsteps_total, 0);
        }
    }
}

// ── Build to-be-computed atom list ────────────────────────────────────
void ComputeMSD::build_tbc_list() {
    if (compute_total) {
        tbc.resize(natoms);
        std::iota(tbc.begin(), tbc.end(), 0);
        return;
    }
    tbc.clear();
    std::set<int> tset(types_tbc.begin(), types_tbc.end());
    for (int i = 0; i < natoms; ++i) {
        if (compute_partial && tset.count(chemmap[i]))   { tbc.push_back(i); continue; }
        if (compute_track && track_filters[i] == 1)      { tbc.push_back(i); continue; }
        if (compute_groups) {
            for (int gid = 0; gid < group_size; ++gid)
                if (group_idxs[gid].count(i)) { tbc.push_back(i); break; }
        }
    }
}

// ── Dispatch to selected algorithm ────────────────────────────────────
void ComputeMSD::dispatch_algorithm(const std::vector<Atoms> &atoms) {
    if (!fft || !tac_avg) {
        if (!tac_avg) compute_msd_nac(atoms);
        else          compute_msd_ac(atoms);
    } else {
        compute_msd_ac_fft(atoms);
    }
}

// ═══════════════════════════════════════════════════════════════════════
void ComputeMSD::compute(const std::vector<Atoms> &atoms)
{
    reset_run_state();
    validate_settings_and_input(atoms);
    prepare_output_selection(atoms[0]);
    nframes_total = (int)atoms.size();
    atoms_volume = atoms[0].get_volume();
    read_weight();
    if (atomic_weight.empty() || using_default_weights) {
        atomic_weight.assign(atoms[0].size(), 1.0);
        using_default_weights = true;
    }
    if (atomic_weight.size() != (size_t)natoms)
        throw std::invalid_argument("MSD weight count must match atom count");
    prepare_positions(atoms);
    apply_drift(atoms);

    // Allocate
    allocate_buffers(atoms);
    if (do_lfit) {
        if (!std::isfinite(lfit_timestart) || !std::isfinite(lfit_timeend) ||
            lfit_timestart < 0.0 || lfit_timeend <= lfit_timestart ||
            lfit_timeend > time.back())
            throw std::invalid_argument("MSD fitting requires an explicit valid fit_start/fit_end window");
        lfit_start = (int)std::ceil(lfit_timestart / ts);
        lfit_end = (int)std::floor(lfit_timeend / ts) + 1;
        if (lfit_end - lfit_start < 2)
            throw std::invalid_argument("MSD fitting window must contain at least two samples");
    }
    build_tbc_list();

    // Compute
    dispatch_algorithm(atoms);
    postprocessing();
    apply_finite_size_correction();
    if (compute_deri || smooth) compute_der();
}

void ComputeMSD::compute_msd_ac(const std::vector<Atoms> &atoms)
{
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 1; m < nsteps_total; ++m)
    {
        double pref = 1.0 / (nframes_total - m);
        double tmpx, tmpy, tmpz;
        for (int k = 0; k < nframes_total - m; ++k)
        {
            // msd at time m
            for (auto &idx : tbc)
            {
                tmpx = positions[k + m][idx][0] - positions[k][idx][0];
                tmpy = positions[k + m][idx][1] - positions[k][idx][1];
                tmpz = positions[k + m][idx][2] - positions[k][idx][2];
                if (proj) {
                    msd_x[idx][m] += pref * tmpx * tmpx;
                    msd_y[idx][m] += pref * tmpy * tmpy;
                    msd_z[idx][m] += pref * tmpz * tmpz;
                } else {
                    msd_x[idx][m] += pref * (tmpx * tmpx + tmpy * tmpy + tmpz * tmpz);
                }
            }
        }
    }
}

void ComputeMSD::compute_msd_nac(const std::vector<Atoms> &atoms)
{

#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 1; m < nsteps_total; ++m)
    {
        double tmpx, tmpy, tmpz;
        for (auto &idx : tbc)
        {
            tmpx = positions[m][idx][0] - positions[0][idx][0];
            tmpy = positions[m][idx][1] - positions[0][idx][1];
            tmpz = positions[m][idx][2] - positions[0][idx][2];
            if (proj) {
                msd_x[idx][m] = tmpx * tmpx;
                msd_y[idx][m] = tmpy * tmpy;
                msd_z[idx][m] = tmpz * tmpz;
            } else {
                msd_x[idx][m] = tmpx * tmpx + tmpy * tmpy + tmpz * tmpz;
            }
        }
    }
}
// using alglib's corrr1d function
void ComputeMSD::compute_msd_ac_fft(const std::vector<Atoms> &atoms)
{
    (void)atoms;
    alglib::ae_int_t SigS = nframes_total;
#ifdef USE_OMP
#pragma omp parallel
#endif
    {
        std::vector<double> dsqx, dsqy, dsqz;
        double sumsqx, sumsqy, sumsqz;
        alglib::real_1d_array sigx, sigy, sigz;
        alglib::real_1d_array corrx, corry, corrz;
        sigx.setlength(nframes_total);
        sigy.setlength(nframes_total);
        sigz.setlength(nframes_total);
        corrx.setlength(2 * SigS); // zero-padding
        corry.setlength(2 * SigS);
        corrz.setlength(2 * SigS);
        dsqx.resize(nframes_total, 0);
        dsqy.resize(nframes_total, 0);
        dsqz.resize(nframes_total, 0);

#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
        for (auto &idx : tbc)
        {
            std::ranges::fill(dsqx, 0);
            std::ranges::fill(dsqy, 0);
            std::ranges::fill(dsqz, 0);
            sumsqx = 0;
            sumsqy = 0;
            sumsqz = 0;

            // compute dsq
            const Eigen::RowVector3d origin = positions[0][idx];
            for (int k = 0; k < nframes_total; ++k)
            {
                const Eigen::RowVector3d shifted = positions[k][idx] - origin;
                sigx[k] = shifted[0];
                sigy[k] = shifted[1];
                sigz[k] = shifted[2];
                dsqx[k] = sigx[k] * sigx[k];
                dsqy[k] = sigy[k] * sigy[k];
                dsqz[k] = sigz[k] * sigz[k];
                sumsqx += 2 * dsqx[k];
                sumsqy += 2 * dsqy[k];
                sumsqz += 2 * dsqz[k];
            }
            alglib::corrr1dbuf(sigx, SigS, sigx, SigS, corrx);
            alglib::corrr1dbuf(sigy, SigS, sigy, SigS, corry);
            alglib::corrr1dbuf(sigz, SigS, sigz, SigS, corrz);
            // compute msd
            for (int m = 1; m < nsteps_total; ++m)
            {
                double pref = 1.0 / (nframes_total - m);
                sumsqx -= dsqx[m - 1] + dsqx[nframes_total - m];
                sumsqy -= dsqy[m - 1] + dsqy[nframes_total - m];
                sumsqz -= dsqz[m - 1] + dsqz[nframes_total - m];
                const double raw_x = pref * (sumsqx - 2 * corrx[m]);
                const double raw_y = pref * (sumsqy - 2 * corry[m]);
                const double raw_z = pref * (sumsqz - 2 * corrz[m]);
                const double scale = pref * (std::abs(sumsqx) + 2 * std::abs(corrx[m]) +
                                             std::abs(sumsqy) + 2 * std::abs(corry[m]) +
                                             std::abs(sumsqz) + 2 * std::abs(corrz[m]));
                const double tolerance = 128.0 * std::numeric_limits<double>::epsilon() *
                                         std::max(1.0, scale);
                auto clean = [tolerance](double value) {
                    return value < 0.0 && value >= -tolerance ? 0.0 : value;
                };
                const double clean_x = clean(raw_x);
                const double clean_y = clean(raw_y);
                const double clean_z = clean(raw_z);
                if (proj) {
                    msd_x[idx][m] = clean_x;
                    msd_y[idx][m] = clean_y;
                    msd_z[idx][m] = clean_z;
                } else {
                    msd_x[idx][m] = clean_x + clean_y + clean_z;
                }
            }
        }
    }
    for (int idx : tbc)
        for (int m = 0; m < nsteps_total; ++m)
            if (msd_x[idx][m] < 0.0 ||
                (proj && (msd_y[idx][m] < 0.0 || msd_z[idx][m] < 0.0)))
                throw std::runtime_error("MSD FFT produced a significant negative value");
}

void ComputeMSD::postprocessing()
{
    if (compute_total)
    {
        std::vector<int> all_atoms(natoms);
        std::iota(all_atoms.begin(), all_atoms.end(), 0);
        std::vector<double> wgt(natoms, 0);
        double wgt_total;
        normalize_weights(wgt, wgt_total, atomic_weight, all_atoms);
        for (int m = 0; m < nsteps_total; ++m)
            for (int i = 0; i < natoms; ++i)
            {
                const double atom_msd = proj
                    ? msd_x[i][m] + msd_y[i][m] + msd_z[i][m]
                    : msd_x[i][m];
                msd_total[m] += atom_msd * wgt[i];
                if (proj)
                {
                    msd_total_x[m] += msd_x[i][m] * wgt[i];
                    msd_total_y[m] += msd_y[i][m] * wgt[i];
                    msd_total_z[m] += msd_z[i][m] * wgt[i];
                }
            }
        if (do_lfit) {
            lfit(time, msd_total, k_total, b_total, &diffusion_total);
        }
    }
    if (compute_partial)
    {
        int tid = 0;
        if (do_lfit) {
            k_type.resize(types_tbc.size());
            b_type.resize(types_tbc.size());
            diffusion_type.resize(types_tbc.size());
        }
        for (auto &itype : types_tbc)
        {
            std::vector<int> type_indices;
            for (int i = 0; i < natoms; ++i)
                if (chemmap[i] == itype) type_indices.push_back(i);
            std::vector<double> wgt(natoms, 0);
            double wgt_total;
            normalize_weights(wgt, wgt_total, atomic_weight, type_indices);
            for (int m = 0; m < nsteps_total; ++m)
                for (int i = 0; i < natoms; ++i)
                {
                    if (chemmap[i] == itype)
                    {
                        const double atom_msd = proj
                            ? msd_x[i][m] + msd_y[i][m] + msd_z[i][m]
                            : msd_x[i][m];
                        msd_type[tid][m] += atom_msd * wgt[i];
                        if (proj)
                        {
                            msd_type_x[tid][m] += msd_x[i][m] * wgt[i];
                            msd_type_y[tid][m] += msd_y[i][m] * wgt[i];
                            msd_type_z[tid][m] += msd_z[i][m] * wgt[i];
                        }
                    }
                }
            if (do_lfit) {
                lfit(time, msd_type[tid], k_type[tid], b_type[tid], &diffusion_type[tid]);
            }
            ++tid;
        }
    }
    if (compute_track)
    {
        if (do_lfit) {
            k_track.resize(track_inds.size());
            b_track.resize(track_inds.size());
            diffusion_track.resize(track_inds.size());
        }
        for (int i = 0; i < track_inds.size(); ++i)
        {
            int idx = track_inds[i];
            for (int m = 0; m < nsteps_total; ++m)
                msd_track[i][m] = proj
                    ? msd_x[idx][m] + msd_y[idx][m] + msd_z[idx][m]
                    : msd_x[idx][m];
            if (do_lfit) {
                lfit(time, msd_track[i], k_track[i], b_track[i], &diffusion_track[i]);
            }
        }
    }
    if (compute_groups)
    {
        if (do_lfit) {
            k_group.resize(group_size);
            b_group.resize(group_size);
            diffusion_group.resize(group_size);
        }
        for (int gid = 0; gid < group_size; ++gid)
        {
            std::vector<int> gindices(group_idxs[gid].begin(), group_idxs[gid].end());
            std::vector<double> wgt(natoms, 0);
            double wgt_total;
            normalize_weights(wgt, wgt_total, atomic_weight, gindices);
            for (auto &idx : group_idxs[gid])
            {
                for (int m = 0; m < nsteps_total; ++m)
                {
                    const double atom_msd = proj
                        ? msd_x[idx][m] + msd_y[idx][m] + msd_z[idx][m]
                        : msd_x[idx][m];
                    msd_group[gid][m] += atom_msd * wgt[idx];
                    if (proj)
                    {
                        msd_group_x[gid][m] +=
                            msd_x[idx][m] * wgt[idx];
                        msd_group_y[gid][m] +=
                            msd_y[idx][m] * wgt[idx];
                        msd_group_z[gid][m] +=
                            msd_z[idx][m] * wgt[idx];
                    }
                }
            }
            if (do_lfit) {
                lfit(time, msd_group[gid], k_group[gid], b_group[gid], &diffusion_group[gid]);
            }
        }
    }
}

void ComputeMSD::apply_finite_size_correction()
{
    if (!do_lfit || !compute_total || !diffusion_total.fitted) return;
    diffusion_total.D_PBC = diffusion_total.slope / 6.0;
    diffusion_total.D_infinite = diffusion_total.D_PBC;
    if (fsc_mode == 0) return;

    if (!std::isfinite(atoms_volume) || atoms_volume <= 0.0)
        throw std::invalid_argument("MSD finite-size correction requires a positive cell volume");
    constexpr double mpa_s_to_ev_ps_a3 = 6.241509074e-3;
    const double eta = viscosity * mpa_s_to_ev_ps_a3;
    const double L = std::cbrt(atoms_volume);
    diffusion_total.D_correction = Boltzmann * temperature * fsc_xi /
                                   (6.0 * M_PI * eta * L);
    diffusion_total.D_infinite = diffusion_total.D_PBC + diffusion_total.D_correction;
    Message(fmt::format("MSD FSC: D_PBC={:.6g} D_corr={:.6g} D_inf={:.6g} A^2/ps",
                        diffusion_total.D_PBC, diffusion_total.D_correction,
                        diffusion_total.D_infinite));
}

void ComputeMSD::compute_spline(alglib::real_1d_array &x, std::vector<double> &y, int &fit_M, double &lam, std::vector<double> &s, std::vector<double> &d)
{
    alglib::spline1dfitreport rep;
    alglib::spline1dinterpolant spline;
    alglib::real_1d_array yval;
    yval.setcontent(y.size(), y.data());
    alglib::spline1dfit(x, yval, fit_M, lam, spline, rep);
    s.resize(y.size());
    d.resize(y.size());
    double trash;
    for (int i = 0; i < y.size(); ++i)
        alglib::spline1ddiff(spline, x[i], s[i], d[i], trash);
}

void ComputeMSD::compute_spline(alglib::real_1d_array &x, std::vector<double> &y, int &fit_M, double &lam, std::vector<double> &s)
{
    alglib::spline1dfitreport rep;
    alglib::spline1dinterpolant spline;
    alglib::real_1d_array yval;
    yval.setcontent(y.size(), y.data());
    alglib::spline1dfit(x, yval, fit_M, lam, spline, rep);
    s.resize(y.size());
    double trash;
    for (int i = 0; i < y.size(); ++i)
        s[i] = alglib::spline1dcalc(spline, x[i]);
}

void ComputeMSD::compute_der()
{
    if (time.size() < 4)
        throw std::invalid_argument("MSD smoothing/derivative requires at least four lag samples");
    if (fit_M < 4)
    {
        fit_M = 4;
    }
    alglib::real_1d_array tval;
    tval.setcontent(time.size(), time.data());
    if (compute_total)
    {
        if (compute_deri)
        {
            compute_spline(tval, msd_total, fit_M, fit_regfactor, msd_total_s, msd_total_d);
            if (proj)
            {
                compute_spline(tval, msd_total_x, fit_M, fit_regfactor, msd_total_x_s, msd_total_x_d);
                compute_spline(tval, msd_total_y, fit_M, fit_regfactor, msd_total_y_s, msd_total_y_d);
                compute_spline(tval, msd_total_z, fit_M, fit_regfactor, msd_total_z_s, msd_total_z_d);
            }
        }
        else
        {
            compute_spline(tval, msd_total, fit_M, fit_regfactor, msd_total_s);
            if (proj)
            {
                compute_spline(tval, msd_total_x, fit_M, fit_regfactor, msd_total_x_s);
                compute_spline(tval, msd_total_y, fit_M, fit_regfactor, msd_total_y_s);
                compute_spline(tval, msd_total_z, fit_M, fit_regfactor, msd_total_z_s);
            }
        }
    }
    if (compute_partial)
    {
        int tid = 0;
        msd_type_s.resize(msd_type.size());
        msd_type_d.resize(msd_type.size());
        if (proj)
        {
            msd_type_x_s.resize(msd_type.size());
            msd_type_y_s.resize(msd_type.size());
            msd_type_z_s.resize(msd_type.size());
            msd_type_x_d.resize(msd_type.size());
            msd_type_y_d.resize(msd_type.size());
            msd_type_z_d.resize(msd_type.size());
        }
        for (auto &itype : types_tbc)
        {
            if (compute_deri)
            {
                compute_spline(tval, msd_type[tid], fit_M, fit_regfactor, msd_type_s[tid], msd_type_d[tid]);
                if (proj)
                {
                    compute_spline(tval, msd_type_x[tid], fit_M, fit_regfactor, msd_type_x_s[tid], msd_type_x_d[tid]);
                    compute_spline(tval, msd_type_y[tid], fit_M, fit_regfactor, msd_type_y_s[tid], msd_type_y_d[tid]);
                    compute_spline(tval, msd_type_z[tid], fit_M, fit_regfactor, msd_type_z_s[tid], msd_type_z_d[tid]);
                }
            }
            else
            {
                compute_spline(tval, msd_type[tid], fit_M, fit_regfactor, msd_type_s[tid]);
                if (proj)
                {
                    compute_spline(tval, msd_type_x[tid], fit_M, fit_regfactor, msd_type_x_s[tid]);
                    compute_spline(tval, msd_type_y[tid], fit_M, fit_regfactor, msd_type_y_s[tid]);
                    compute_spline(tval, msd_type_z[tid], fit_M, fit_regfactor, msd_type_z_s[tid]);
                }
            }
            ++tid;
        }
    }
    if (compute_groups)
    {
        msd_group_s.resize(msd_group.size());
        msd_group_d.resize(msd_group.size());
        if (proj)
        {
            msd_group_x_s.resize(msd_group.size());
            msd_group_y_s.resize(msd_group.size());
            msd_group_z_s.resize(msd_group.size());
            msd_group_x_d.resize(msd_group.size());
            msd_group_y_d.resize(msd_group.size());
            msd_group_z_d.resize(msd_group.size());
        }
        for (int gid = 0; gid < group_size; ++gid)
        {
            if (compute_deri)
            {
                compute_spline(tval, msd_group[gid], fit_M, fit_regfactor, msd_group_s[gid], msd_group_d[gid]);
                if (proj)
                {
                    compute_spline(tval, msd_group_x[gid], fit_M, fit_regfactor, msd_group_x_s[gid], msd_group_x_d[gid]);
                    compute_spline(tval, msd_group_y[gid], fit_M, fit_regfactor, msd_group_y_s[gid], msd_group_y_d[gid]);
                    compute_spline(tval, msd_group_z[gid], fit_M, fit_regfactor, msd_group_z_s[gid], msd_group_z_d[gid]);
                }
            }
            else
            {
                compute_spline(tval, msd_group[gid], fit_M, fit_regfactor, msd_group_s[gid]);
                if (proj)
                {
                    compute_spline(tval, msd_group_x[gid], fit_M, fit_regfactor, msd_group_x_s[gid]);
                    compute_spline(tval, msd_group_y[gid], fit_M, fit_regfactor, msd_group_y_s[gid]);
                    compute_spline(tval, msd_group_z[gid], fit_M, fit_regfactor, msd_group_z_s[gid]);
                }
            }
        }
    }
    if (compute_track)
    {
        msd_track_s.resize(msd_track.size());
        msd_track_d.resize(msd_track.size());
        if (proj)
        {
            msd_x_s.resize(msd_x.size());
            msd_x_d.resize(msd_x.size());
            msd_y_s.resize(msd_y.size());
            msd_y_d.resize(msd_y.size());
            msd_z_s.resize(msd_z.size());
            msd_z_d.resize(msd_z.size());
        }
#ifdef USE_OMP
#pragma omp parallel for
#endif
        for (int i = 0; i < track_inds.size(); ++i)
        {
            int idx = track_inds[i];
            if (compute_deri)
            {
                compute_spline(tval, msd_track[i], fit_M, fit_regfactor, msd_track_s[i], msd_track_d[i]);
                if (proj)
                {
                    compute_spline(tval, msd_x[idx], fit_M, fit_regfactor, msd_x_s[idx], msd_x_d[idx]);
                    compute_spline(tval, msd_y[idx], fit_M, fit_regfactor, msd_y_s[idx], msd_y_d[idx]);
                    compute_spline(tval, msd_z[idx], fit_M, fit_regfactor, msd_z_s[idx], msd_z_d[idx]);
                }
            }
            else
            {
                compute_spline(tval, msd_track[i], fit_M, fit_regfactor, msd_track_s[i]);
                if (proj)
                {
                    compute_spline(tval, msd_x[idx], fit_M, fit_regfactor, msd_x_s[idx]);
                    compute_spline(tval, msd_y[idx], fit_M, fit_regfactor, msd_y_s[idx]);
                    compute_spline(tval, msd_z[idx], fit_M, fit_regfactor, msd_z_s[idx]);
                }
            }
        }
    }
    if (compute_deri) {
        auto to_diffusion = [](const std::vector<double> &derivative) {
            std::vector<double> result(derivative.size());
            std::transform(derivative.begin(), derivative.end(), result.begin(),
                           [](double value) { return value / 6.0; });
            return result;
        };
        if (compute_total) diffusion_total_inst = to_diffusion(msd_total_d);
        if (compute_partial) {
            diffusion_type_inst.resize(msd_type_d.size());
            for (size_t i = 0; i < msd_type_d.size(); ++i)
                diffusion_type_inst[i] = to_diffusion(msd_type_d[i]);
        }
        if (compute_groups) {
            diffusion_group_inst.resize(msd_group_d.size());
            for (size_t i = 0; i < msd_group_d.size(); ++i)
                diffusion_group_inst[i] = to_diffusion(msd_group_d[i]);
        }
        if (compute_track) {
            diffusion_track_inst.resize(msd_track_d.size());
            for (size_t i = 0; i < msd_track_d.size(); ++i)
                diffusion_track_inst[i] = to_diffusion(msd_track_d[i]);
        }
    }
}

void ComputeMSD::write_header(std::ofstream &ofs)
{
    ofs << std::setprecision(15);
    ofs << "# time[ps] msd[A^2] ";
    if (smooth || compute_deri)
        ofs << " smsd ";
    if (compute_deri)
        ofs << " dmsd[A^2/ps] ";
    if (proj)
    {
        ofs << " msdx msdy msdz ";
        if (smooth || compute_deri)
            ofs << " smsdx smsdy smsdz ";
        if (compute_deri)
            ofs << " dmsdx dmsdy dmsdz ";
    }
    if (compute_deri)
        ofs << " D_inst[A^2/ps] ";
    ofs << '\n';
}

void ComputeMSD::prepare_output_selection(const Atoms &first)
{
    natoms = first.size();
    chemmap = first.chemmap_;
    types_tbc.clear();
    if (compute_partial)
        for (const auto &[type, count] : first.types_count_map()) types_tbc.push_back(type);
    track_inds.clear(); group_idxs.clear(); group_size = 0;
    if (compute_track) read_filter();
    if (compute_groups) read_group();
}

std::string ComputeMSD::output_path(const std::string &category, int id) const
{
    if (category == "total") return f_total + ".dat";
    const auto &prefix = category == "type" ? f_type : category == "group" ? f_group : f_track;
    return prefix + "_" + std::to_string(id) + ".dat";
}

std::vector<std::string> ComputeMSD::planned_output_paths(const Atoms &first)
{
    prepare_output_selection(first);
    std::vector<std::string> paths;
    if (compute_total) paths.push_back(output_path("total"));
    if (compute_partial) for (int id : types_tbc) paths.push_back(output_path("type", id));
    if (compute_groups) for (int id = 0; id < group_size; ++id) paths.push_back(output_path("group", id));
    if (compute_track) for (int id : track_inds) paths.push_back(output_path("track", id));
    return paths;
}

void ComputeMSD::write_results()
{
    if (compute_total)
    {
        std::ofstream ofs(output_path("total"));
        if (!ofs) throw std::runtime_error("Cannot open MSD output file: " + f_total + ".dat");
        write_header(ofs);
        ofs << std::setprecision(15);
        if (do_lfit) write_fit_metadata(ofs, diffusion_total, ts);
        for (int i = 0; i < nsteps_total; ++i)
        {
            ofs << time[i] << std::setw(25) << msd_total[i];
            if (smooth || compute_deri)
                ofs << std::setw(25) << msd_total_s[i];
            if (compute_deri)
                ofs << std::setw(25) << msd_total_d[i];
            if (proj)
            {
                ofs << std::setw(25) << msd_total_x[i]
                    << std::setw(25) << msd_total_y[i]
                    << std::setw(25) << msd_total_z[i];
                if (smooth || compute_deri)
                {
                    ofs << std::setw(25) << msd_total_x_s[i]
                        << std::setw(25) << msd_total_y_s[i]
                        << std::setw(25) << msd_total_z_s[i];
                }
                if (compute_deri)
                {
                    ofs << std::setw(25) << msd_total_x_d[i]
                        << std::setw(25) << msd_total_y_d[i]
                        << std::setw(25) << msd_total_z_d[i];
                }
            }
            if (compute_deri)
                ofs << std::setw(25) << diffusion_total_inst[i];
            ofs << '\n';
        }
        ofs.close();
        if (!ofs) throw std::runtime_error("Failed writing MSD output file: " + f_total + ".dat");
    }
    if (compute_partial)
    {
        int idtype = 0;
        for (auto &tid : types_tbc)
        {
            std::ofstream ofs(output_path("type", tid));
            if (!ofs) throw std::runtime_error("Cannot open MSD type output file");
            write_header(ofs);
            ofs << std::setprecision(15);
            if (do_lfit) write_fit_metadata(ofs, diffusion_type[idtype], ts);
            for (int i = 0; i < nsteps_total; ++i)
            {
                ofs << time[i] << std::setw(25) << msd_type[idtype][i];
                if (smooth || compute_deri)
                    ofs << std::setw(25) << msd_type_s[idtype][i];
                if (compute_deri)
                    ofs << std::setw(25) << msd_type_d[idtype][i];
                if (proj)
                {
                    ofs << std::setw(25) << msd_type_x[idtype][i]
                        << std::setw(25) << msd_type_y[idtype][i]
                        << std::setw(25) << msd_type_z[idtype][i];
                    if (smooth || compute_deri)
                    {
                        ofs << std::setw(25) << msd_type_x_s[idtype][i]
                            << std::setw(25) << msd_type_y_s[idtype][i]
                            << std::setw(25) << msd_type_z_s[idtype][i];
                    }
                    if (compute_deri)
                    {
                        ofs << std::setw(25) << msd_type_x_d[idtype][i]
                            << std::setw(25) << msd_type_y_d[idtype][i]
                            << std::setw(25) << msd_type_z_d[idtype][i];
                    }
                }
                if (compute_deri)
                    ofs << std::setw(25) << diffusion_type_inst[idtype][i];
                ofs << '\n';
            }
            ofs.close();
            if (!ofs) throw std::runtime_error("Failed writing MSD type output file");
            ++idtype;
        }
    }
    if (compute_groups)
    {
        for (int gid = 0; gid < group_size; ++gid)
        {
            std::ofstream ofs(output_path("group", gid));
            if (!ofs) throw std::runtime_error("Cannot open MSD group output file");
            write_header(ofs);
            ofs << std::setprecision(15);
            if (do_lfit) write_fit_metadata(ofs, diffusion_group[gid], ts);
            for (int i = 0; i < nsteps_total; ++i)
            {
                ofs << time[i] << std::setw(25) << msd_group[gid][i];
                if (smooth || compute_deri)
                    ofs << std::setw(25) << msd_group_s[gid][i];
                if (compute_deri)
                    ofs << std::setw(25) << msd_group_d[gid][i];
                if (proj)
                {
                    ofs << std::setw(25) << msd_group_x[gid][i]
                        << std::setw(25) << msd_group_y[gid][i]
                        << std::setw(25) << msd_group_z[gid][i];
                    if (smooth || compute_deri)
                    {
                        ofs << std::setw(25) << msd_group_x_s[gid][i]
                            << std::setw(25) << msd_group_y_s[gid][i]
                            << std::setw(25) << msd_group_z_s[gid][i];
                    }
                    if (compute_deri)
                    {
                        ofs << std::setw(25) << msd_group_x_d[gid][i]
                            << std::setw(25) << msd_group_y_d[gid][i]
                            << std::setw(25) << msd_group_z_d[gid][i];
                    }
                }
                if (compute_deri)
                    ofs << std::setw(25) << diffusion_group_inst[gid][i];
                ofs << '\n';
            }
            ofs.close();
            if (!ofs) throw std::runtime_error("Failed writing MSD group output file");
        }
    }
    if (compute_track)
    {
        for (int ict = 0; ict < track_inds.size(); ++ict)
        {
            int idx = track_inds[ict];
            std::ofstream ofs(output_path("track", idx));
            if (!ofs) throw std::runtime_error("Cannot open MSD track output file");
            write_header(ofs);
            ofs << std::setprecision(15);
            if (do_lfit) write_fit_metadata(ofs, diffusion_track[ict], ts);
            for (int i = 0; i < nsteps_total; ++i)
            {
                ofs << time[i] << std::setw(25) << msd_track[ict][i];
                if (smooth || compute_deri)
                    ofs << std::setw(25) << msd_track_s[ict][i];
                if (compute_deri)
                    ofs << std::setw(25) << msd_track_d[ict][i];
                if (proj)
                {
                    ofs << std::setw(25) << msd_x[idx][i]
                        << std::setw(25) << msd_y[idx][i]
                        << std::setw(25) << msd_z[idx][i];
                    if (smooth || compute_deri)
                    {
                        ofs << std::setw(25) << msd_x_s[idx][i]
                            << std::setw(25) << msd_y_s[idx][i]
                            << std::setw(25) << msd_z_s[idx][i];
                    }
                    if (compute_deri)
                    {
                        ofs << std::setw(25) << msd_x_d[idx][i]
                            << std::setw(25) << msd_y_d[idx][i]
                            << std::setw(25) << msd_z_d[idx][i];
                    }
                }
                if (compute_deri)
                    ofs << std::setw(25) << diffusion_track_inst[ict][i];
                ofs << '\n';
            }
            ofs.close();
            if (!ofs) throw std::runtime_error("Failed writing MSD track output file");
        }
    }
}

void ComputeMSD::read_group()
{
    /**
     * [ngroups]
     * \[ [name] [gid] [size] \]
     * [... ids ...]
     * just like gromacs
     * */
    std::ifstream ifs(f_group_id);
    char tmpch;
    std::string word;
    if (!ifs)
        throw std::runtime_error("Cannot open MSD group file: " + f_group_id);
    if (!(ifs >> group_size) || group_size <= 0 || group_size > 1000000)
        throw std::runtime_error("Invalid group count in MSD group file: " + f_group_id);
    group_idxs.assign(group_size, {});
    std::vector<bool> seen(group_size, false);
    for (int i = 0; i < group_size; ++i) {
        int gidsize, gid;
        char close;
        if (!(ifs >> tmpch >> word >> gid >> gidsize >> close) || tmpch != '[' || close != ']' ||
            gid < 0 || gid >= group_size || gidsize <= 0 || gidsize > natoms || seen[gid])
            throw std::runtime_error("Invalid group header in MSD group file: " + f_group_id);
        seen[gid] = true;
        for (int j = 0; j < gidsize; ++j) {
            int idx;
            if (!(ifs >> idx) || idx < 0 || idx >= natoms)
                throw std::runtime_error("Invalid atom ID in MSD group file: " + f_group_id);
            if (!group_idxs[gid].insert(idx).second)
                throw std::runtime_error("Duplicate atom ID in MSD group file: " + f_group_id);
        }
    }
}

void ComputeMSD::read_filter()
{
    /**
     * [size]
     * [... ids ...]
     * */
    std::ifstream ifs(f_track_id);
    if (!ifs)
        throw std::runtime_error("Cannot open MSD track file: " + f_track_id);
    int size;
    if (!(ifs >> size) || size <= 0 || size > natoms)
        throw std::runtime_error("Invalid selection size in MSD track file: " + f_track_id);
    track_inds.resize(size);
    std::set<int> seen;
    for (int i = 0; i < size; ++i) {
        if (!(ifs >> track_inds[i]) || track_inds[i] < 0 || track_inds[i] >= natoms)
            throw std::runtime_error("Invalid atom ID in MSD track file: " + f_track_id);
        if (!seen.insert(track_inds[i]).second)
            throw std::runtime_error("Duplicate atom ID in MSD track file: " + f_track_id);
    }
}

void ComputeMSD::read_weight()
{
    /**
     * [size]
     * [... wgts ...]
     * */
    if (f_weight_id.empty()) return;
    std::ifstream ifs(f_weight_id);
    if (!ifs)
        throw std::runtime_error("Cannot open MSD weight file: " + f_weight_id);
    int size;
    if (!(ifs >> size) || size != natoms)
        throw std::runtime_error("MSD weight count must match atom count: " + f_weight_id);
    atomic_weight.resize(size);
    using_default_weights = false;
    for (int i = 0; i < size; ++i)
        if (!(ifs >> atomic_weight[i]) || !finite_nonnegative(atomic_weight[i]))
            throw std::runtime_error("MSD weights must be finite and non-negative: " + f_weight_id);
}

void ComputeMSD::lfit(const std::vector<double> &x, const std::vector<double> &y,
                      double &k, double &b, DiffusionFitResult *result) {
    if (x.size() != y.size() || lfit_start < 0 || lfit_end > (int)x.size() ||
        lfit_end - lfit_start < 2)
        throw std::invalid_argument("Invalid MSD linear-fit data or bounds");
    const int n = lfit_end - lfit_start;
    Eigen::MatrixXd A(n, 2);
    for (int i = 0; i < n; ++i) {
        A(i, 0) = x[i + lfit_start];
        A(i, 1) = 1;
    }
    Eigen::VectorXd vb(n);
    for (int i = 0; i < n; ++i) vb(i) = y[i + lfit_start];
    const Eigen::VectorXd res = A.colPivHouseholderQr().solve(vb);
    k = res(0);
    b = res(1);
    if (!std::isfinite(k) || !std::isfinite(b))
        throw std::runtime_error("MSD linear fit produced a non-finite result");
    if (result) {
        result->fitted = true;
        result->begin = lfit_start;
        result->end = lfit_end;
        result->slope = k;
        result->intercept = b;
        result->D_PBC = k / 6.0;
        result->D_infinite = result->D_PBC;
        if (n > 2) {
            const double xmean = std::accumulate(x.begin() + lfit_start, x.begin() + lfit_end, 0.0) / n;
            double sxx = 0.0;
            for (int i = lfit_start; i < lfit_end; ++i) {
                const double dx = x[i] - xmean;
                sxx += dx * dx;
            }
            const double rss = (A * res - vb).squaredNorm();
            result->slope_stderr = std::sqrt((rss / (n - 2)) / sxx);
        }
    }
}
