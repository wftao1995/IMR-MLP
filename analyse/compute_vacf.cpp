// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_vacf.h"
#include "libalglib/fasttransforms.h"
#include "tools/tools.h"
#include <fftw3.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace {
void validate_vacf_trajectory(const std::vector<Atoms> &atoms, int ndim)
{
    if (atoms.empty())
        throw std::invalid_argument("VACF requires at least one trajectory frame");
    if (atoms[0].size() == 0)
        throw std::invalid_argument("VACF requires at least one atom");
    if (ndim < 1 || ndim > 3)
        throw std::invalid_argument("VACF ndim must be between 1 and 3");

    const int natoms = atoms[0].size();
    for (int i = 0; i < (int)atoms.size(); ++i) {
        if (atoms[i].size() != natoms)
            throw std::invalid_argument("VACF requires a constant atom count across frames");
        if ((int)atoms[i].chemmap_.size() != natoms)
            throw std::invalid_argument("VACF requires chemmap size to match atom count");
        if (atoms[i].chemmap_ != atoms[0].chemmap_)
            throw std::invalid_argument("VACF requires stable atom types and ordering across frames");
    }
}

Eigen::RowVector3d minimum_image_displacement(const Atoms &frame,
                                               const Eigen::RowVector3d &delta)
{
    Atoms pair;
    pair.cell_ = frame.cell_;
    pair.pbc_ = frame.pbc_;
    pair.cartesian = true;
    pair.pos_ = {Eigen::RowVector3d::Zero(), delta};
    pair.chemmap_ = {0, 0};
    return pair.get_rab(0, 1);
}

Eigen::RowVector3d cartesian_position(const Atoms &frame, int iatom)
{
    if (frame.cartesian) return frame.pos_[iatom];
    return frame.pos_[iatom] * frame.cell_;
}

void normalize_series(std::vector<std::array<double, 4>> &series)
{
    if (series.empty()) return;
    for (int a = 0; a < 4; ++a) {
        const double c0 = series[0][a];
        if (!std::isfinite(c0))
            throw std::runtime_error("VACF contains a non-finite zero-lag value");
        if (std::abs(c0) <= 1e-30) {
            for (auto &row : series) row[a] = 0.0;
            continue;
        }
        for (auto &row : series) row[a] /= c0;
    }
}

// Fit an AR(p) model to a velocity series via Yule-Walker equations and
// return the analytic VACF gamma(m), m = 0..nlag-1, from the AR coefficients:
//   gamma(m) = r(m) for m < p (sample autocovariances, reproduced exactly
//              by the Yule-Walker fit)
//   gamma(m) = sum_{k=1..p} a_k * gamma(m-k) for m >= p
//
// Order selection: p = ar_order when > 0, otherwise the order minimizing
// BIC = N*ln(sigma2_p) + p*ln(N) over p = 1..pmax is chosen. Levinson-Durbin
// recursion provides the coefficients for every order incrementally in one
// pass; recursion stops early if a reflection coefficient reaches 1 (the
// series is no longer stationary at that order).
static int fit_ar(const std::vector<double> &v, int pmax, int fixed_order,
                  std::vector<double> &a, std::vector<double> &r)
{
    const int n = (int)v.size();
    a.assign(pmax, 0.0);
    r.assign(pmax + 1, 0.0);
    if (n < 3 || pmax < 1)
        return 0;

    // biased autocovariance estimates r(k), k = 0..pmax
    for (int k = 0; k <= pmax; ++k) {
        for (int t = k; t < n; ++t)
            r[k] += v[t] * v[t - k];
        r[k] /= n;
    }
    if (r[0] <= 1e-30)
        return 0;

    // Levinson-Durbin: a(k) holds the order-k coefficients (indexed 0..k-1)
    std::vector<double> ak(pmax, 0.0), prev(pmax, 0.0);
    int best_p = 1;
    double best_bic = 1e300;
    double sigma2 = r[0];
    for (int p = 1; p <= pmax; ++p) {
        if (2 * p + 1 > n)
            break;
        double num = r[p];
        for (int k = 1; k < p; ++k)
            num -= ak[k - 1] * r[p - k];
        const double refl = num / sigma2; // reflection (partial autocorr) coeff
        if (std::abs(refl) >= 1.0)
            break;
        ak[p - 1] = refl;
        for (int k = 1; k < p; ++k)
            ak[k - 1] -= refl * prev[p - 1 - k];
        prev = ak;
        sigma2 *= (1.0 - refl * refl);
        if (sigma2 <= 1e-30)
            break;

        if (fixed_order > 0 && p == fixed_order) {
            std::copy(ak.begin(), ak.begin() + p, a.begin());
            return p;
        }
        if (fixed_order == 0) {
            const double bic = n * std::log(sigma2) + p * std::log((double)n);
            if (bic < best_bic) {
                best_bic = bic;
                best_p = p;
                std::copy(ak.begin(), ak.begin() + p, a.begin());
            }
        }
    }
    if (fixed_order > 0)
        return -1; // recursion stopped before the requested order
    if (best_bic == 1e300)
        return 0; // no order evaluated: degenerate series
    return best_p;
}

static void ar_vacf(const std::vector<double> &r, const std::vector<double> &a, int p,
                    int nlag, std::vector<double> &gamma)
{
    gamma.resize(nlag);
    // the Yule-Walker fit reproduces the sample autocovariances r(0..p-1)
    for (int m = 0; m < p && m < nlag; ++m)
        gamma[m] = r[m];
    for (int m = p; m < nlag; ++m) {
        double g = 0.0;
        for (int k = 1; k <= p; ++k)
            g += a[k - 1] * gamma[m - k];
        gamma[m] = g;
    }
}

static void vacf_from_ar(const std::vector<double> &v, int ar_order, int ar_pmax, int nlag,
                         std::vector<double> &gamma)
{
    gamma.resize(nlag);
    std::vector<double> a, r;
    const int pmax = (ar_order > 0)
                         ? ar_order
                         : std::min(ar_pmax, (static_cast<int>(v.size()) - 1) / 2);
    int p = fit_ar(v, pmax, ar_order, a, r);
    if (p < 0)
        throw std::runtime_error("VACF could not fit the requested fixed AR order");
    if (p < 1) {
        std::fill(gamma.begin(), gamma.end(), 0.0);
        return;
    }
    ar_vacf(r, a, p, nlag, gamma);
}
}

void ComputeVACF::reset_run_state()
{
    types_tbc.clear();
    group_idxs.clear();
    group_size = 0;
    track_inds.clear();
    track_filters.clear();
    track_size = 0;
    projection_dir.clear();
    nprojection = 0;
    tbc.clear();
    derived_velocities.clear();
    velocity_drift.clear();
    using_derived_velocities = false;
    vacf_array.clear();
    vacf_isotropic = {};
    vacf_projection.clear();
    time.clear();
    frquency.clear();
}

Eigen::Vector3d ComputeVACF::velocity_at(const std::vector<Atoms> &atoms,
                                         int iframe, int iatom) const
{
    Eigen::Vector3d v = using_derived_velocities
        ? derived_velocities[iframe - run_start][iatom]
        : atoms[iframe].velocities_[iatom];
    if (!velocity_drift.empty())
        v -= velocity_drift[iframe - run_start];
    return v;
}

void ComputeVACF::prepare_velocities(const std::vector<Atoms> &atoms,
                                     int requested_start, int requested_stop)
{
    bool any_velocity = false;
    bool all_velocity = true;
    for (const auto &frame : atoms) {
        const bool full = (int)frame.velocities_.size() == natoms;
        const bool empty = frame.velocities_.empty();
        if (!full && !empty)
            throw std::invalid_argument("VACF velocity size must match atom count in every frame");
        any_velocity = any_velocity || full;
        all_velocity = all_velocity && full;
    }
    if (any_velocity && !all_velocity)
        throw std::invalid_argument("VACF requires velocities in all frames or in none");

    if (all_velocity) {
        run_start = requested_start;
        nsteps_total = requested_stop - requested_start;
        return;
    }

    if (atoms.size() < 3)
        throw std::invalid_argument("VACF requires velocities or at least three frames for central differences");
    for (const auto &frame : atoms)
        if ((int)frame.pos_.size() != natoms)
            throw std::invalid_argument("VACF position size must match atom count when estimating velocities");

    run_start = std::max(requested_start, 1);
    const int run_stop = std::min(requested_stop, (int)atoms.size() - 1);
    nsteps_total = run_stop - run_start;
    if (nsteps_total <= 0)
        throw std::invalid_argument("VACF selected range has no central-difference velocity samples");

    using_derived_velocities = true;
    derived_velocities.assign(nsteps_total,
                              std::vector<Eigen::Vector3d>(natoms, Eigen::Vector3d::Zero()));
    for (int iframe = run_start; iframe < run_stop; ++iframe) {
        for (int iatom = 0; iatom < natoms; ++iatom) {
            const auto dprev = minimum_image_displacement(
                atoms[iframe], cartesian_position(atoms[iframe], iatom)
                    - cartesian_position(atoms[iframe - 1], iatom));
            const auto dnext = minimum_image_displacement(
                atoms[iframe + 1], cartesian_position(atoms[iframe + 1], iatom)
                    - cartesian_position(atoms[iframe], iatom));
            derived_velocities[iframe - run_start][iatom] =
                ((dprev + dnext) / (2.0 * ts)).transpose();
        }
    }
}

void ComputeVACF::prepare_weights()
{
    auto validate_masses = [&]() {
        for (int type : chemmap) {
            if (type < 0 || type >= (int)atom_masses.size() ||
                !std::isfinite(atom_masses[type]) || atom_masses[type] <= 0.0)
                throw std::invalid_argument("VACF requires a positive mass for every chemical type");
        }
    };

    if (mass_weighted) {
        if (!f_weight_id.empty())
            throw std::invalid_argument("VACF mass_weighted and weightfile cannot be used together");
        validate_masses();
        atomic_weight.resize(natoms);
        for (int i = 0; i < natoms; ++i) atomic_weight[i] = atom_masses[chemmap[i]];
    } else if (atomic_weight.size() != (size_t)natoms) {
        if (!f_weight_id.empty())
            throw std::invalid_argument("VACF weight file size must match atom count");
        if (!atomic_weight.empty())
            Warning("Warning: VACF weighting reset to uniform");
        atomic_weight.assign(natoms, 1.0);
    }

    for (double weight : atomic_weight)
        if (!std::isfinite(weight) || weight < 0.0)
            throw std::invalid_argument("VACF atom weights must be finite and non-negative");

    if (remove_drift) validate_masses();
}

void ComputeVACF::compute(const std::vector<Atoms> &atoms)
{
    reset_run_state();
    validate_vacf_trajectory(atoms, ndim);
    const int requested_start = std::max(nstart, 0);
    const int requested_stop = (nstop < 0 || nstop > (int)atoms.size())
        ? (int)atoms.size() : nstop;
    if (requested_stop <= requested_start)
        throw std::invalid_argument("VACF end_to must be greater than start_from");
    if (ts <= 0.0)
        throw std::invalid_argument("VACF timestep must be positive");
    if (lal_frac <= 0.0 || lal_frac > 1.0)
        throw std::invalid_argument("VACF lal_frac must be in (0, 1]");
    if (ismear < 0 || ismear > 5)
        throw std::invalid_argument("VACF ismear must be between 0 and 5");
    if (ismear != 0 && dtau <= 0.0)
        throw std::invalid_argument("VACF dtau must be positive when smearing is enabled");
    if (freq_step <= 0.0)
        throw std::invalid_argument("VACF dfreq must be positive");
    if (fmax < 0.0)
        throw std::invalid_argument("VACF fmax must be non-negative");
    if (fft_zero_padding < 0)
        throw std::invalid_argument("VACF zero_padding must be non-negative");
    if (proj_only && !proj)
        throw std::invalid_argument("VACF noiso requires projection=true");
    if (!compute_total && !compute_partial && !compute_track && !compute_groups)
        throw std::invalid_argument("VACF requires at least one total, types, track, or groups output");

    natoms = atoms[0].size();
    chemmap = atoms[0].chemmap_;
    prepare_velocities(atoms, requested_start, requested_stop);
    prepare_weights();

    if (remove_drift) {
        velocity_drift.assign(nsteps_total, Eigen::Vector3d::Zero());
        double total_mass = 0.0;
        for (int i = 0; i < natoms; ++i) total_mass += atom_masses[chemmap[i]];
        for (int iframe = run_start; iframe < run_start + nsteps_total; ++iframe) {
            Eigen::Vector3d comv = Eigen::Vector3d::Zero();
            for (int i = 0; i < natoms; ++i) {
                const Eigen::Vector3d raw = using_derived_velocities
                    ? derived_velocities[iframe - run_start][i]
                    : atoms[iframe].velocities_[i];
                comv += atom_masses[chemmap[i]] * raw;
            }
            velocity_drift[iframe - run_start] = comv / total_mass;
        }
    }
    if (nsteps_total <= 0)
        throw std::invalid_argument("VACF selected frame range is empty");
    if (algo == 3) {
        if (ar_order < 0)
            throw std::invalid_argument("VACF ar_order must be non-negative");
        if (ar_pmax < 1)
            throw std::invalid_argument("VACF ar_pmax must be positive");
        if (ar_order > 0 && 2LL * ar_order + 1 > nsteps_total)
            throw std::invalid_argument("VACF trajectory is too short for the requested AR order");
    }
    noutput = std::max(1, (int)std::round(nsteps_total * lal_frac));
    time.resize(nsteps_total);
    for (int i = 0; i < nsteps_total; ++i)
        time[i] = ts * i;

    prepare_output_selection(atoms[0]);

    if (compute_vdos) {
        const double required = std::ceil(1.0 / (2.0 * ts * freq_step)) + 1.0;
        if (required > (double)std::numeric_limits<int>::max())
            throw std::invalid_argument("VACF requested frequency grid is too large");
        ndos_total = std::max({2, noutput + fft_zero_padding, (int)required});
        dv = 1.0 / double(2.0 * ts * (ndos_total - 1));
        const double nyquist = 1.0 / (2.0 * ts);
        const double output_max = std::min(fmax, nyquist);
        if (fmax > nyquist)
            Warning(fmt::format("VACF fmax={} THz exceeds Nyquist={} THz; output is truncated", fmax, nyquist));
        ndos_output = std::min(ndos_total, (int)std::floor(output_max / dv) + 1);
        frquency.resize(ndos_output);
        for (int i = 0; i < ndos_output; ++i) frquency[i] = i * dv;
    }

    // init memory
    if (proj)
        vacf_projection.resize(nprojection);

    // init tbc
    if (compute_total)
    {
        tbc.resize(natoms);
        std::iota(tbc.begin(), tbc.end(), 0);
    }
    else
    {
        tbc.clear();
        std::set<int> tset;
        for (auto &it : types_tbc)
            tset.insert(it);
        for (int i = 0; i < natoms; ++i)
        {
            if (compute_partial)
            {
                if (tset.count(chemmap[i]) > 0)
                {
                    tbc.push_back(i);
                    continue;
                }
            }
            if (compute_track)
            {
                if (track_filters[i] == 1)
                {
                    tbc.push_back(i);
                    continue;
                }
            }
            if (compute_groups)
            {
                for (int gid = 0; gid < group_size; ++gid)
                {
                    if (group_idxs[gid].count(i) > 0)
                    {
                        tbc.push_back(i);
                        break;
                    }
                }
            }
        }
    }

    // compute
    if (!proj_only)
    {
        if (algo == 1)
        {
            compute_vacf_ac_fft(atoms);
        }
        else if (algo == 2)
        {
            compute_vacf_ac(atoms);
        }
        else if (algo == 0)
        {
            compute_vacf_nac(atoms);
        }
        else if (algo == 3)
        {
            compute_vacf_ac_ar(atoms);
        }
        else
        {
            throw std::invalid_argument("Unknown VACF algorithm");
        }
        // for (int i = 0; i < natoms; ++i)
        //     for (int j = 0; j < nsteps_total; ++j)
        //         vacf_array[i][j] *= 0.33333333333333333333;
        postprocessing(vacf_isotropic);
    }
    if (proj)
    {
        for (int i = 0; i < nprojection; ++i)
        {
            std::vector<std::vector<double>> projected(nsteps_total,
                                                       std::vector<double>(natoms));
            for (int iframe = 0; iframe < nsteps_total; ++iframe)
                for (int iatom = 0; iatom < natoms; ++iatom)
                    projected[iframe][iatom] = projection_dir[i].dot(
                        velocity_at(atoms, iframe + run_start, iatom));
            if (algo == 1)
            {
                compute_vacf_ac_fft(projected);
            }
            else if (algo == 2)
            {
                compute_vacf_ac(projected);
            }
            else if (algo == 0)
            {
                compute_vacf_nac(projected);
            }
            else if (algo == 3)
            {
                compute_vacf_ac_ar(projected);
            }
            else
            {
                throw std::invalid_argument("Unknown VACF algorithm");
            }
            postprocessing(vacf_projection[i]);
        }
    }
}

void ComputeVACF::compute_vacf_ac(std::vector<std::vector<double>> &vs)
{
    clear_arr();
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < nsteps_total; ++m)
    {
        double pref = 1.0 / (nsteps_total - m);
        for (int k = 0; k < nsteps_total - m; ++k)
        {
            // vacf at time m
            for (auto &idx : tbc)
            {
                vacf_array[idx][m][0] +=
                    vs[k + m][idx] * vs[k][idx] * pref;
            }
        }
        for (int idx : tbc)
            vacf_array[idx][m][3] = vacf_array[idx][m][0];
    }
}

void ComputeVACF::compute_vacf_ac(const std::vector<Atoms> &atoms)
{
    clear_arr();
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < nsteps_total; ++m)
    {
        double pref = 1.0 / (nsteps_total - m);
        for (int k = 0; k < nsteps_total - m; ++k)
        {
            // vacf at time m
            for (auto &idx : tbc)
            {
                // vacf_array[idx][m] +=
                //     v(k+m).dot(v(k)) * pref;
                const auto va = velocity_at(atoms, k + m + run_start, idx);
                const auto vb = velocity_at(atoms, k + run_start, idx);
                for (int a = 0; a < ndim; ++a)
                    vacf_array[idx][m][a] += pref * va[a] * vb[a];
            }
        }
        for (auto &idx : tbc)
            vacf_array[idx][m][3] = (vacf_array[idx][m][0] + vacf_array[idx][m][1] + vacf_array[idx][m][2]) / (double)ndim;
    }
}

void ComputeVACF::compute_vacf_nac(std::vector<std::vector<double>> &v_s)
{
    clear_arr();
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < nsteps_total; ++m)
    {
        for (auto &idx : tbc)
        {
            vacf_array[idx][m][0] +=
                v_s[m][idx] * v_s[0][idx];
            vacf_array[idx][m][3] = vacf_array[idx][m][0];
        }
    }
}

void ComputeVACF::compute_vacf_nac(const std::vector<Atoms> &atoms)
{
    clear_arr();
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int m = 0; m < nsteps_total; ++m)
    {
        for (auto &idx : tbc)
        {
            // vacf_array[idx][m] +=
            //     atoms[m + nstart].velocities_[idx].dot(atoms[nstart].velocities_[idx]);
            const auto vm = velocity_at(atoms, m + run_start, idx);
            const auto v0 = velocity_at(atoms, run_start, idx);
            for (int a = 0; a < ndim; ++a)
                vacf_array[idx][m][a] += 
                    vm[a] * v0[a];
            vacf_array[idx][m][3] += (vacf_array[idx][m][0] + vacf_array[idx][m][1] + vacf_array[idx][m][2]) / (double)ndim;
        }
    }
}

// using alglib's corrr1d function
void ComputeVACF::compute_vacf_ac_fft(const std::vector<Atoms> &atoms)
{
    clear_arr();
    alglib::ae_int_t SigS = nsteps_total;
#ifdef USE_OMP
#pragma omp parallel
#endif
    {
        alglib::real_1d_array sigx, sigy, sigz;
        alglib::real_1d_array corrx, corry, corrz;
        sigx.setlength(nsteps_total);
        if (ndim > 1) sigy.setlength(nsteps_total);
        if (ndim > 2) sigz.setlength(nsteps_total);

#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
        for (auto &idx : tbc)
        {
            // compute sab
            for (int i = 0; i < nsteps_total; ++i)
            {
                const auto v = velocity_at(atoms, i + run_start, idx);
                sigx[i] = v[0];
                if (ndim > 1) sigy[i] = v[1];
                if (ndim > 2) sigz[i] = v[2];
            }
            alglib::corrr1d(sigx, SigS, sigx, SigS, corrx);
            if (ndim > 1) alglib::corrr1d(sigy, SigS, sigy, SigS, corry);
            if (ndim > 2) alglib::corrr1d(sigz, SigS, sigz, SigS, corrz);
            // compute vacf
            for (int m = 0; m < nsteps_total; ++m)
            {
                double pref = 1.0 / (nsteps_total - m);
                vacf_array[idx][m][0] = pref * corrx[m];
                if (ndim > 1) vacf_array[idx][m][1] = pref * corry[m];
                if (ndim > 2) vacf_array[idx][m][2] = pref * corrz[m];
                vacf_array[idx][m][3] = (vacf_array[idx][m][0] + vacf_array[idx][m][1] + vacf_array[idx][m][2]) / (double)ndim;
            }
        }
    }
}

void ComputeVACF::compute_vacf_ac_fft(std::vector<std::vector<double>> &vs)
{
    clear_arr();
    alglib::ae_int_t SigS = nsteps_total;
#ifdef USE_OMP
#pragma omp parallel
#endif
    {
        alglib::real_1d_array sig;
        alglib::real_1d_array corr;
        sig.setlength(nsteps_total);

#ifdef USE_OMP
#pragma omp for schedule(static)
#endif
        for (auto &idx : tbc)
        {
            // compute sab
            for (int i = 0; i < nsteps_total; ++i)
            {
                sig[i] = vs[i][idx];
            }
            alglib::corrr1d(sig, SigS, sig, SigS, corr);
            // compute vacf
            for (int m = 0; m < nsteps_total; ++m)
            {
                double pref = 1.0 / (nsteps_total - m);
                vacf_array[idx][m][0] = pref * corr[m];
                vacf_array[idx][m][3] = vacf_array[idx][m][0];
            }
        }
    }
}

void ComputeVACF::compute_vacf_ac_ar(std::vector<std::vector<double>> &vs)
{
    clear_arr();
    std::exception_ptr fit_error;
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int tidx = 0; tidx < (int)tbc.size(); ++tidx)
    {
        try {
            const int idx = tbc[tidx];
            std::vector<double> v(nsteps_total), gamma;
            for (int i = 0; i < nsteps_total; ++i)
                v[i] = vs[i][idx];
            vacf_from_ar(v, ar_order, ar_pmax, nsteps_total, gamma);
            for (int m = 0; m < nsteps_total; ++m)
            {
                vacf_array[idx][m][0] = gamma[m];
                vacf_array[idx][m][3] = gamma[m];
            }
        } catch (...) {
#ifdef USE_OMP
#pragma omp critical(vacf_ar_error)
#endif
            {
                if (!fit_error)
                    fit_error = std::current_exception();
            }
        }
    }
    if (fit_error)
        std::rethrow_exception(fit_error);
}

void ComputeVACF::compute_vacf_ac_ar(const std::vector<Atoms> &atoms)
{
    clear_arr();
    std::exception_ptr fit_error;
#ifdef USE_OMP
#pragma omp parallel for schedule(static)
#endif
    for (int tidx = 0; tidx < (int)tbc.size(); ++tidx)
    {
        try {
            const int idx = tbc[tidx];
            std::vector<double> v(nsteps_total), gamma;
            for (int a = 0; a < ndim; ++a)
            {
                for (int i = 0; i < nsteps_total; ++i)
                    v[i] = velocity_at(atoms, i + run_start, idx)[a];
                vacf_from_ar(v, ar_order, ar_pmax, nsteps_total, gamma);
                for (int m = 0; m < nsteps_total; ++m)
                    vacf_array[idx][m][a] = gamma[m];
            }
            for (int m = 0; m < nsteps_total; ++m)
            {
                double iso = 0.0;
                for (int a = 0; a < ndim; ++a)
                    iso += vacf_array[idx][m][a];
                vacf_array[idx][m][3] = iso / (double)ndim;
            }
        } catch (...) {
#ifdef USE_OMP
#pragma omp critical(vacf_ar_error)
#endif
            {
                if (!fit_error)
                    fit_error = std::current_exception();
            }
        }
    }
    if (fit_error)
        std::rethrow_exception(fit_error);
}

void ComputeVACF::postprocessing(vacf_result &res)
{
    if (compute_total)
    {
        std::vector<double> wgt = atomic_weight;
        double wgt_total = 0;
        for (int i = 0; i < atomic_weight.size(); ++i)
            wgt_total += atomic_weight[i];
        if (wgt_total <= 0.0)
            throw std::runtime_error("VACF total weights must sum to a positive value");
        for (int i = 0; i < atomic_weight.size(); ++i)
            wgt[i] /= wgt_total;
        res.total.resize(nsteps_total);
        std::fill(res.total.begin(), res.total.end(), std::array<double, 4>{0., 0., 0., 0.});
        for (int m = 0; m < nsteps_total; ++m)
            for (int i = 0; i < natoms; ++i)
                for (int a = 0; a < 4; ++a)
                    res.total[m][a] += vacf_array[i][m][a] * wgt[i];
        if (normalize) normalize_series(res.total);
        if (compute_vdos) {
            compute_vdosfft(res.total, res.vdos_total);
        }
    }
    if (compute_partial)
    {
        int tid = 0;
        res.type.resize(types_tbc.size());
        if (compute_vdos) res.vdos_type.resize(types_tbc.size());
        for (auto &itype : types_tbc)
        {
            std::vector<double> wgt(natoms, 0);
            double wgt_total = 0;
            for (int i = 0; i < natoms; ++i)
            {
                if (chemmap[i] == itype)
                {
                    wgt[i] = atomic_weight[i];
                    wgt_total += wgt[i];
                }
            }
            for (int i = 0; i < wgt.size(); ++i)
            {
                if (wgt_total <= 0.0)
                    throw std::runtime_error("VACF type weights must sum to a positive value");
                wgt[i] /= wgt_total;
            }
            res.type[tid].resize(nsteps_total);
            std::fill(res.type[tid].begin(), res.type[tid].end(), std::array<double, 4>{0., 0., 0., 0.});
            for (int m = 0; m < nsteps_total; ++m)
                for (int i = 0; i < natoms; ++i)
                    if (chemmap[i] == itype)
                        for (int a = 0; a < 4; ++a)
                            res.type[tid][m][a] += vacf_array[i][m][a] * wgt[i];
            if (normalize) normalize_series(res.type[tid]);
            if (compute_vdos) {
                compute_vdosfft(res.type[tid], res.vdos_type[tid]);
            }
            ++tid;
        }
    }
    if (compute_track)
    {
        res.track.resize(track_inds.size());
        if (compute_vdos) res.vdos_track.resize(track_inds.size());
        for (int i = 0; i < track_inds.size(); ++i)
        {
            int idx = track_inds[i];
            res.track[i].resize(nsteps_total);
            for (int m = 0; m < nsteps_total; ++m)
                res.track[i][m] = vacf_array[idx][m];
            if (normalize) normalize_series(res.track[i]);
            if (compute_vdos) {
                compute_vdosfft(res.track[i], res.vdos_track[i]);
            }
        }
    }
    if (compute_groups)
    {
        res.group.resize(group_size);
        if (compute_vdos) res.vdos_group.resize(group_size);
        for (int gid = 0; gid < group_size; ++gid)
        {
            std::vector<double> wgt(natoms, 0);
            double wgt_total = 0;
            for (auto &gidx : group_idxs[gid])
            {
                wgt[gidx] = atomic_weight[gidx];
                wgt_total += wgt[gidx];
            }
            for (int i = 0; i < wgt.size(); ++i)
            {
                if (wgt_total <= 0.0)
                    throw std::runtime_error("VACF group weights must sum to a positive value");
                wgt[i] /= wgt_total;
            }
            res.group[gid].resize(nsteps_total);
            std::fill(res.group[gid].begin(), res.group[gid].end(), std::array<double, 4>{0., 0., 0., 0.});
            for (auto &idx : group_idxs[gid])
                for (int m = 0; m < nsteps_total; ++m)
                    for (int a = 0; a < 4; ++a)
                        res.group[gid][m][a] += vacf_array[idx][m][a] * wgt[idx];
            if (normalize) normalize_series(res.group[gid]);
            if (compute_vdos) {
                compute_vdosfft(res.group[gid], res.vdos_group[gid]);
            }
        }
    }
}
// Time-domain smearing/damping window for VACF before FFT.
// Applies a smooth cutoff around tau0 with width dtau to suppress
// long-time noise in the VACF, which otherwise pollutes the VDOS.
//
// ismear options:
//   0 — no smearing (identity)
//   1 — Fermi-Dirac: 1/(1+exp(dx))          smooth step
//   2 — Gaussian:     0.5*(1-erf(dx))       smooth step (Gaussian CDF)
//   3 — Power-4:      1/(1+dx^4) for dx>0   asymmetric, zero below tau0
//   4 — Cubic-Fermi:  1/(1+exp(dx^3))       sharper transition
//   5 — Cubic-Gauss:  0.5*(1-erf(dx^3))     sharper Gaussian
//
// where dx = (t - tau0) / dtau, t = istep * ts
double ComputeVACF::smearing(int istep)
{
    if (ismear == 0) return 1.0;
    double x = istep * ts;
    double dx = (x - tau0) / dtau;

    switch (ismear) {
    case 1:
        if (dx >= 0.0) {
            const double e = std::exp(-dx);
            return e / (1.0 + e);
        }
        return 1.0 / (1.0 + std::exp(dx));
    case 2: return 0.5 * std::erfc(dx);
    case 3:
        if (dx > 0.0) {
            double x2 = dx * dx;
            return 1.0 / (1.0 + x2 * x2);
        }
        return 1.0;
    case 4: {
        const double dx3 = dx * dx * dx;
        if (dx3 >= 0.0) {
            const double e = std::exp(-dx3);
            return e / (1.0 + e);
        }
        return 1.0 / (1.0 + std::exp(dx3));
    }
    case 5: return 0.5 * std::erfc(dx * dx * dx);
    default: return 1.0;
    }
}

// compute_int() and compute_spline() were superseded by direct FFT-based VDOS
// (compute_vdosfft). The spline-integration path is no longer maintained.

void ComputeVACF::compute_vdosfft(std::vector<std::array<double, 4>> &in, std::vector<std::array<double, 4>> &out) {
    std::vector<double> fftw_in, fftw_out;
    // v1 v2 v3 .. vn 0 0 .. 0 vn-1 vn-2 .. v2 : dt (ps), length = 2 * nlag - 2
    // f1 f2 f3 .. fm : dv (THz)
    // m = n + n_padding
    int nlag = noutput;
    if ((int)in.size() < nlag || ndos_total < 2)
        throw std::invalid_argument("VACF VDOS input or transform length is invalid");
    out.assign(ndos_total, std::array<double, 4>{0., 0., 0., 0.});
    fftw_in.assign(ndos_total, 0.0);
    fftw_out.assign(ndos_total, 0.0);
    fftw_plan r2r = fftw_plan_r2r_1d(ndos_total, fftw_in.data(), fftw_out.data(), FFTW_REDFT00, FFTW_ESTIMATE);
    if (!r2r)
        throw std::runtime_error("VACF failed to create FFTW VDOS plan");
    for (int a = 0; a < 4; ++a) {
        if (std::abs(in[0][a]) <= 1e-30) continue;
        for (int i = 0; i < nlag; ++i) fftw_in[i] = in[i][a] * smearing(i);
        for (int i = nlag; i < ndos_total; ++i) fftw_in[i] = 0.;
        fftw_execute(r2r);
        for (int i = 0; i < ndos_total; ++i) out[i][a] = ts * fftw_out[i];
    }
    fftw_destroy_plan(r2r);

    if (normalize) {
        for (int a = 0; a < 4; ++a) {
            if (std::abs(in[0][a]) <= 1e-30) continue;
            double integral = 0.5 * (out[0][a] + out[ndos_total - 1][a]);
            for (int i = 1; i < ndos_total - 1; ++i) integral += out[i][a];
            integral *= dv;
            if (!std::isfinite(integral) || std::abs(integral) <= 1e-30)
                throw std::runtime_error("VACF cannot normalize a zero or non-finite VDOS integral");
            for (int i = 0; i < ndos_total; ++i) out[i][a] /= integral;
        }
    }
}

void ComputeVACF::write_header(std::ofstream &ofs, bool projected)
{
    if (projected)
        ofs << "# time(ps) vacf_projection unused_y unused_z vacf_projection_iso\n";
    else
        ofs << "# time(ps) vacf_x vacf_y vacf_z vacf_iso\n";
}

void ComputeVACF::write_header_dos(std::ofstream &ofs)
{
    ofs << "# freq(THz) dos_x dos_y dos_z dos_iso\n";
}

void ComputeVACF::prepare_output_selection(const Atoms &first)
{
    natoms = first.size();
    chemmap = first.chemmap_;
    const std::set<int> types(chemmap.begin(), chemmap.end());
    types_tbc.assign(types.begin(), types.end());
    track_inds.clear(); group_idxs.clear(); group_size = 0;
    projection_dir.clear(); nprojection = 0;
    // init by file
    if (compute_track)
    {
        read_filter();
        if (track_inds.size() == 0)
        {
            track_inds.resize(natoms);
            std::iota(track_inds.begin(), track_inds.end(), 0);
        }
        track_size = (int)track_inds.size();
        track_filters.assign(natoms, 0);
        for (int idx : track_inds) {
            if (idx < 0 || idx >= natoms)
                throw std::out_of_range("VACF track index out of range");
            track_filters[idx] = 1;
        }
    }
    if (compute_groups)
    {
        read_group();
        for (const auto &group : group_idxs)
            for (int idx : group)
                if (idx < 0 || idx >= natoms)
                    throw std::out_of_range("VACF group index out of range");
    }
    if (proj)
    {
        read_proj();
    }
}

std::string ComputeVACF::output_prefix(int projection) const
{
    return projection == 0 ? f_header : f_header + "_proj_" + std::to_string(projection);
}

std::string ComputeVACF::output_path(const std::string &prefix, const std::string &category, int id, bool vdos)
{
    return prefix + (vdos ? "_vdos_" : "_") + category +
        (category == "total" ? "" : "_" + std::to_string(id)) + ".dat";
}

std::vector<std::string> ComputeVACF::planned_output_paths(const Atoms &first)
{
    if (proj_only && !proj)
        throw std::invalid_argument("VACF noiso requires projection=true");
    prepare_output_selection(first);
    std::vector<std::string> paths;
    auto append = [&](const std::string &prefix) {
        auto add = [&](const std::string &category, int id = 0) {
            paths.push_back(output_path(prefix, category, id));
            if (compute_vdos) paths.push_back(output_path(prefix, category, id, true));
        };
        if (compute_total) add("total");
        if (compute_partial) for (int id : types_tbc) add("type", id);
        if (compute_groups) for (int id = 0; id < group_size; ++id) add("group", id);
        if (compute_track) for (int id : track_inds) add("track", id);
    };
    if (!proj_only) append(output_prefix());
    if (proj) for (int i = 0; i < nprojection; ++i) append(output_prefix(i + 1));
    return paths;
}

void ComputeVACF::write_results(vacf_result &res, const std::string &fnm_header)
{
    const bool projected = fnm_header.find("_proj_") != std::string::npos;
    auto open_output = [](std::ofstream &ofs, const std::string &path) {
        ofs.open(path);
        if (!ofs)
            throw std::runtime_error("VACF cannot open output file: " + path);
    };
    auto finish_output = [](std::ofstream &ofs, const std::string &path) {
        ofs.close();
        if (!ofs)
            throw std::runtime_error("VACF failed while writing output file: " + path);
    };
    if (res.total.size() > 0)
    {
        const std::string path = output_path(fnm_header, "total", 0);
        std::ofstream ofs;
        open_output(ofs, path);
        write_header(ofs, projected);
        for (int i = 0; i < noutput; ++i)
        {
            fmt::print(ofs, "{} {}\n", time[i], fmt::join(res.total[i], " "));
        }
        finish_output(ofs, path);
        if (compute_vdos) {
            const std::string vdos_path = output_path(fnm_header, "total", 0, true);
            open_output(ofs, vdos_path);
            write_header_dos(ofs);
            for (int i = 0; i < ndos_output; ++i) {
                fmt::print(ofs, "{} {}\n", i * dv, fmt::join(res.vdos_total[i], " "));
            }
            finish_output(ofs, vdos_path);
        }
    }
    if (res.type.size() > 0)
    {
        for (int idtype = 0; idtype < (int)types_tbc.size(); ++idtype)
        {
            int tid = types_tbc[idtype];
            const std::string path = output_path(fnm_header, "type", tid);
            std::ofstream ofs;
            open_output(ofs, path);
            write_header(ofs, projected);
            for (int i = 0; i < noutput; ++i)
            {
                fmt::print(ofs, "{} {}\n", time[i], fmt::join(res.type[idtype][i], " "));
            }
            finish_output(ofs, path);
            if (compute_vdos) {
                const std::string vdos_path = output_path(fnm_header, "type", tid, true);
                open_output(ofs, vdos_path);
                write_header_dos(ofs);
                for (int i = 0; i < ndos_output; ++i) {
                    fmt::print(ofs, "{} {}\n", i * dv, fmt::join(res.vdos_type[idtype][i], " "));
                }
                finish_output(ofs, vdos_path);
            }
        }
    }
    if (res.group.size() > 0)
    {
        for (int gid = 0; gid < group_size; ++gid)
        {
            const std::string path = output_path(fnm_header, "group", gid);
            std::ofstream ofs;
            open_output(ofs, path);
            write_header(ofs, projected);
            for (int i = 0; i < noutput; ++i)
            {
                fmt::print(ofs, "{} {}\n", time[i], fmt::join(res.group[gid][i], " "));
            }
            finish_output(ofs, path);
            if (compute_vdos) {
                const std::string vdos_path = output_path(fnm_header, "group", gid, true);
                open_output(ofs, vdos_path);
                write_header_dos(ofs);
                for (int i = 0; i < ndos_output; ++i) {
                    fmt::print(ofs, "{} {}\n", i * dv, fmt::join(res.vdos_group[gid][i], " "));
                }
                finish_output(ofs, vdos_path);
            }
        }
    }
    if (res.track.size() > 0)
    {
        for (int ict = 0; ict < track_inds.size(); ++ict)
        {
            int idx = track_inds[ict];
            const std::string path = output_path(fnm_header, "track", idx);
            std::ofstream ofs;
            open_output(ofs, path);
            write_header(ofs, projected);
            for (int i = 0; i < noutput; ++i)
            {
                fmt::print(ofs, "{} {}\n", time[i], fmt::join(res.track[ict][i], " "));
            }
            finish_output(ofs, path);
            if (compute_vdos) {
                const std::string vdos_path = output_path(fnm_header, "track", idx, true);
                open_output(ofs, vdos_path);
                write_header_dos(ofs);
                for (int i = 0; i < ndos_output; ++i) {
                    fmt::print(ofs, "{} {}\n", i * dv, fmt::join(res.vdos_track[ict][i], " "));
                }
                finish_output(ofs, vdos_path);
            }
        }
    }
}

void ComputeVACF::write_results()
{
    if (!proj_only)
        write_results(vacf_isotropic, output_prefix());
    if (proj)
        for (int i = 0; i < nprojection; ++i)
            write_results(vacf_projection[i], output_prefix(i + 1));
}

void ComputeVACF::read_group()
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
        throw std::runtime_error("VACF cannot open group file: " + f_group_id);
    if (!(ifs >> group_size) || group_size <= 0)
        throw std::runtime_error("VACF group file must contain a positive group count");
    group_idxs.assign(group_size, {});
    for (int i = 0; i < group_size; ++i)
    {
        int gidsize, gid;
        if (!(ifs >> tmpch >> word >> gid >> gidsize >> tmpch) ||
            gid < 0 || gid >= group_size || gidsize <= 0)
            throw std::runtime_error("VACF group file has an invalid group header");
        for (int j = 0; j < gidsize; ++j)
        {
            int idx;
            if (!(ifs >> idx))
                throw std::runtime_error("VACF group file is truncated");
            group_idxs[gid].insert(idx);
        }
    }
}

void ComputeVACF::read_filter()
{
    /**
     * [size]
     * [... ids ...]
     * */
    std::ifstream ifs(f_track_id);
    if (!ifs)
        throw std::runtime_error("VACF cannot open track file: " + f_track_id);
    int size;
    if (!(ifs >> size) || size <= 0)
        throw std::runtime_error("VACF track file must contain a positive atom count");
    std::set<int> unique;
    for (int i = 0; i < size; ++i) {
        int idx;
        if (!(ifs >> idx))
            throw std::runtime_error("VACF track file is truncated");
        if (!unique.insert(idx).second)
            throw std::runtime_error("VACF track file contains duplicate atom IDs");
        track_inds.push_back(idx);
    }
}

void ComputeVACF::read_weight()
{
    /**
     * [size]
     * [... ids ...]
     * */
    if (f_weight_id.empty()) return;
    std::ifstream ifs(f_weight_id);
    if (!ifs)
        throw std::runtime_error("VACF cannot open weight file: " + f_weight_id);
    int size;
    if (!(ifs >> size) || size <= 0)
        throw std::runtime_error("VACF weight file must contain a positive weight count");
    atomic_weight.resize(size);
    for (int i = 0; i < size; ++i)
        if (!(ifs >> atomic_weight[i]) || !std::isfinite(atomic_weight[i]) || atomic_weight[i] < 0.0)
            throw std::runtime_error("VACF weight file contains an invalid weight");
}

void ComputeVACF::read_proj()
{
    /**
     * [size]
     * a1 a2 a3
     * b1 b2 b3
     * */
    std::ifstream ifs(f_proj_dir);
    if (!ifs)
        throw std::runtime_error("VACF cannot open projection file: " + f_proj_dir);
    int size = 0;
    if (!(ifs >> size) || size <= 0)
        throw std::runtime_error("VACF projection file must contain a positive direction count");
    nprojection = size;
    projection_dir.resize(size);
    for (int i = 0; i < size; ++i)
    {
        if (!(ifs >> projection_dir[i][0] >> projection_dir[i][1] >> projection_dir[i][2]))
            throw std::runtime_error("VACF projection file is truncated");
        const double norm = projection_dir[i].norm();
        if (!std::isfinite(norm) || norm <= 1e-14)
            throw std::runtime_error("VACF projection direction must be finite and non-zero");
        projection_dir[i] /= norm;
    }
}
