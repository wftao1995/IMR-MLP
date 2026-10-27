// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "compute_fftfilter.h"
#include "libalglib/fasttransforms.h"
#include "libalglib/statistics.h"
#include "trajectory_transform.h"
#include "fft_resample.h"
#include <algorithm>
#include <exception>
#include <limits>
#include <stdexcept>
#include <string>

namespace {
void validate_fft_input(const std::vector<Atoms> &atoms, double dt, double freq_cutoff, bool rebuild_velo, bool rebuild_force)
{
    if (atoms.size() < 2 || atoms.front().size() == 0)
        GPERROR("FFT smoothing requires at least two nonempty frames");
    if (!std::isfinite(dt) || dt <= 0 || !std::isfinite(dt * atoms.size()))
        GPERROR("FFT smoothing requires a finite positive timestep and duration");
    if (!std::isfinite(freq_cutoff) || freq_cutoff < 0)
        GPERROR("FFT smoothing requires a finite nonnegative frequency cutoff");
    for (const auto &frame : atoms)
        if (frame.size() != atoms.front().size())
            GPERROR("FFT smoothing requires the same atom count in every frame");

    const auto &first = atoms.front();
    const auto natoms = first.pos_.size();
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const auto &frame = atoms[i];
        if (frame.chemmap_.size() != natoms || frame.chemmap_ != first.chemmap_)
            throw std::invalid_argument("FFT smoothing requires the same species in atom order in every frame (frame " +
                std::to_string(i) + ")");
        if (frame.pbc_ != first.pbc_)
            throw std::invalid_argument("FFT smoothing requires the same PBC flags in every frame (frame " +
                std::to_string(i) + ")");
    }
    if (rebuild_force) {
        const auto masses = first.local_attribs_d_.find("masses");
        if (masses == first.local_attribs_d_.end() || masses->second.size() != natoms ||
            !masses->second.allFinite() || (masses->second.array() <= 0).any())
            throw std::invalid_argument("Force reconstruction requires explicit positive per-atom masses in amu");
        for (const auto &frame : atoms) {
            const auto current = frame.local_attribs_d_.find("masses");
            if (current == frame.local_attribs_d_.end() || current->second.size() != natoms ||
                (current->second.array() != masses->second.array()).any())
                throw std::invalid_argument("Force reconstruction requires the same per-atom masses in every frame");
        }
    }
    for (std::size_t i = 0; i < atoms.size(); ++i) {
        const auto &frame = atoms[i];
        const auto require_channel = [&](bool complete, const std::string &name) {
            if (!complete)
                throw std::invalid_argument("FFT smoothing channel " + name +
                    " is incomplete at frame " + std::to_string(i));
        };
        if (first.has_forces() && !rebuild_force) require_channel(frame.forces_.size() == natoms, "forces");
        if (first.has_site_energies()) require_channel(frame.site_energies_.size() == natoms, "site energies");
        if (first.has_velo() && !rebuild_velo) require_channel(frame.velocities_.size() == natoms, "velocities");
        if (first.has_energy()) require_channel(frame.has_energy(), "energy");
        if (first.has_fenergy()) require_channel(frame.has_fenergy(), "free energy");
        if (first.has_stresses()) require_channel(frame.has_stresses(), "stress");
        for (const auto &[key, value] : first.global_attribs_d_)
            require_channel(frame.global_attribs_d_.count(key), "global double " + key);
        for (const auto &[key, value] : first.global_attribs_v_) {
            const auto entry = frame.global_attribs_v_.find(key);
            require_channel(entry != frame.global_attribs_v_.end() && entry->second.size() == value.size(),
                            "global vector " + key);
        }
        for (const auto &[key, value] : first.local_attribs_d_) {
            const auto entry = frame.local_attribs_d_.find(key);
            require_channel(entry != frame.local_attribs_d_.end() && entry->second.size() == natoms,
                            "local double " + key);
        }
        for (const auto &[key, value] : first.local_attribs_v_) {
            const auto entry = frame.local_attribs_v_.find(key);
            require_channel(entry != frame.local_attribs_v_.end() && entry->second.rows() == value.rows() &&
                            entry->second.cols() == natoms, "local vector " + key);
        }
    }
}

bool use_cartesian_fft_positions(const std::vector<Atoms> &atoms)
{
    const bool nonperiodic = std::all_of(atoms.begin(), atoms.end(),
        [](const Atoms &frame) { return frame.is_nonperiodic(); });
    if (nonperiodic)
        return std::any_of(atoms.begin(), atoms.end(),
            [](const Atoms &frame) { return !frame.cell_.fullPivLu().isInvertible(); });

    const bool singular_partial = std::any_of(atoms.begin(), atoms.end(),
        [](const Atoms &frame) {
            return !frame.is_nonperiodic() && !frame.is_fully_periodic()
                && !frame.cell_.fullPivLu().isInvertible();
        });
    if (!singular_partial) return false;
    const auto &first = atoms.front();
    for (const auto &frame : atoms) {
        if ((frame.cell_.array() != first.cell_.array()).any())
            throw std::invalid_argument("FFT smoothing with a variable singular partial cell is not supported");
        if (!frame.cartesian)
            throw std::invalid_argument("FFT smoothing with a singular partial cell requires Cartesian positions");
    }
    // A common linear filter commutes with a fixed cell transformation.
    // Keep Cartesian components so the open-direction residual is retained.
    return true;
}
}
/**
 * @brief 
 * 
 * Unwraps positions and smooths both positions and cell components.
 * Velocity reconstruction remains unfinished in this legacy compute() path;
 * compute2() implements it for the command-line trajectory smoother.
 * 
 * @param atoms 
 * @param dt 
 * @param freq_cutoff 
 */
// length should be even for fast speed
void ComputeSmoothFFT::compute(std::vector<Atoms> &atoms, double dt, int wtype)
{
    if (rebuild_velo || rebuild_force)
        throw std::invalid_argument("Derivative reconstruction is not implemented in ComputeSmoothFFT::compute; use compute2");
    validate_fft_input(atoms, dt, params.freq_cutoff, rebuild_velo, rebuild_force);
    int nframe = atoms.size();
    if (nframe % 2 != 0) {
        Warning("Odd frame count, may be slow");
    }
    int natoms = atoms[0].size();
    double T = nframe * dt;
    // double base_freq = 1.0 / T;
    int ncutoff = std::min(params.freq_cutoff * T, double(std::numeric_limits<int>::max()));
    param_type param_curr = params;
    param_curr.dt = dt;
    param_curr.ncutoff = ncutoff;

    // deltaq not needed in compute() — FFT low-pass filter uses
    // direct frequency cutoff, not a calibrated frequency axis
    _DEBUG_OUTPUT_(ncutoff);

    const bool cartesian_positions = use_cartesian_fft_positions(atoms);
    for (auto &frame : atoms)
        if (!frame.cartesian) frame.to_cartesian();
    const bool nonperiodic = std::all_of(atoms.begin(), atoms.end(),
        [](const Atoms &frame) { return frame.is_nonperiodic(); });
    if (!nonperiodic) unwarp(atoms);

    bool have_force = atoms[0].has_forces(); //
    bool have_stress = atoms[0].has_stresses(); //
    bool have_velo = atoms[0].has_velo(); //
    bool have_energy = atoms[0].has_energy(); //
    bool have_siteen = atoms[0].has_site_energies(); //
    bool have_fenergy = atoms[0].has_fenergy(); //

    #define COMPUTE_FFT_BLOCK(param) \
        for (int j = 0; j < nframe; ++j) { \
            buffer_data[j] = atoms[j].param; \
        } \
        alglib::samplemoments(buffer_data, mean, variance, skewness, kurtosis); \
        if (variance < 1e-3) variance = 1; \
        for (int l = 0; l < nframe; ++l) { \
            buffer_data[l] = (buffer_data[l] - mean) / variance; \
        } \
        alglib::fftr1d(buffer_data, buffer_fft); \
        applywindow1(buffer_fft, wtype, param_curr); \
        alglib::fftr1dinv(buffer_fft, buffer_data); \
        for (int j = 0; j < nframe; ++j) { \
            atoms[j].param = buffer_data[j] * variance + mean; \
        }

// atomic props
#pragma omp parallel
{
    // create buffers
    alglib::real_1d_array buffer_data;
    alglib::complex_1d_array buffer_fft;
    buffer_data.setlength(nframe);
    buffer_fft.setlength(nframe);
    double mean, variance, skewness, kurtosis;

    #pragma omp for
    for (int i = 0; i < nframe; ++i)
        if (!cartesian_positions) atoms[i].to_fractional();

    // smooth properties
    #pragma omp for
    for (int i = 0; i < natoms; ++i) {
        // position
        for (int k = 0; k < 3; ++k) {
            COMPUTE_FFT_BLOCK(pos_[i](k))
        }
        // force
        if (have_force)
        for (int k = 0; k < 3; ++k) {
            COMPUTE_FFT_BLOCK(forces_[i](k))
        }
        // velocity
        if (have_velo && !rebuild_velo)
        for (int k = 0; k < 3; ++k) {
            COMPUTE_FFT_BLOCK(velocities_[i](k))
        }
        // siteen
        if (have_siteen) {
            COMPUTE_FFT_BLOCK(site_energies_[i])
        }
        // local_d
        for (auto &it : atoms[0].local_attribs_d_) {
            COMPUTE_FFT_BLOCK(local_attribs_d_.at(it.first)(i))
        }
        // local_v
        for (auto &it : atoms[0].local_attribs_v_) {
            auto name = it.first;
            for (int k = 0; k < it.second.rows(); ++k) {
                COMPUTE_FFT_BLOCK(local_attribs_v_.at(name)(k,i))
            }
        }
    }
}
// global vector/matrix props
#pragma omp parallel
{
    // create buffers
    alglib::real_1d_array buffer_data;
    alglib::complex_1d_array buffer_fft;
    buffer_data.setlength(nframe);
    buffer_fft.setlength(nframe);
    double mean, variance, skewness, kurtosis;

    // cell
    #pragma omp for
    for (int i = 0; i < 9; ++i) {
        int s1 = floor(i / 3);
        int s2 = i % 3;
        COMPUTE_FFT_BLOCK(cell_(s1, s2))
    }
    #pragma omp for
    for (int i = 0; i < nframe; ++i)
        if (!cartesian_positions) atoms[i].to_cartesian();

    // stress
    if (have_stress) {
        #pragma omp for
        for (int i = 0; i < 9; ++i) {
            int s1 = floor(i / 3);
            int s2 = i % 3;
            COMPUTE_FFT_BLOCK(stress_(s1, s2))
        }
    }
    // global vector
    for (auto &it : atoms[0].global_attribs_v_) {
        #pragma omp for
        for (int ss = 0; ss < it.second.size(); ++ss) {
            COMPUTE_FFT_BLOCK(global_attribs_v_.at(it.first)(ss))
        }
    }
}
    // global scalar props
{
    // create buffers
    alglib::real_1d_array buffer_data;
    alglib::complex_1d_array buffer_fft;
    buffer_data.setlength(nframe);
    buffer_fft.setlength(nframe);
    double mean, variance, skewness, kurtosis;
    // energy
    if (have_energy) {
        COMPUTE_FFT_BLOCK(energy_)
    }
    // fenergy
    if (have_fenergy) {
        COMPUTE_FFT_BLOCK(fenergy_)
    }
    // global scalar
    for (auto &it : atoms[0].global_attribs_d_) {
        COMPUTE_FFT_BLOCK(global_attribs_d_.at(it.first))
    }
}
    #undef COMPUTE_FFT_BLOCK
}
// Duplicate the sequence for mirror-symmetric FFT.
// When even_seq=false, the first element (DC) is not duplicated, avoiding
// double-weighting the origin. When even_seq=true, all points are mirrored.
void ComputeSmoothFFT::compute2_base(std::vector<Atoms> &atoms, double dt, int wtype,bool reconstruct)
{
    validate_fft_input(atoms, dt, params.freq_cutoff, rebuild_velo, rebuild_force);
    int nframe = atoms.size();
    // int buffer_len = 2*nframe - 1;
    int buffer_len = even_seq? 2*nframe : 2*nframe - 1;
    // if (nframe % 2 != 0) {
    //     Warning("Odd frame count, may be slow");
    // }
    int natoms = atoms[0].size();
    double T = (nframe - 1) * dt;
    int ncutoff = std::min(params.freq_cutoff * T, double(std::numeric_limits<int>::max()));
    param_type param_curr = params;
    param_curr.dt = dt;
    param_curr.ncutoff = ncutoff;

    _DEBUG_OUTPUT_(ncutoff);

    const bool cartesian_positions = use_cartesian_fft_positions(atoms);
    for (auto &frame : atoms)
        if (!frame.cartesian) frame.to_cartesian();
    const bool nonperiodic = std::all_of(atoms.begin(), atoms.end(),
        [](const Atoms &frame) { return frame.is_nonperiodic(); });
    if (!nonperiodic) unwarp(atoms);

    bool have_force = atoms[0].has_forces(); //
    bool have_stress = atoms[0].has_stresses(); //
    bool have_velo = atoms[0].has_velo(); //
    bool have_energy = atoms[0].has_energy(); //
    bool have_siteen = atoms[0].has_site_energies(); //
    bool have_fenergy = atoms[0].has_fenergy(); //

    if (rebuild_velo)
        for (auto &frame : atoms) frame.velocities_.resize(natoms);

    if (rebuild_force)
        for (auto &frame : atoms) frame.forces_.resize(natoms);

    // Exceptions must leave OpenMP regions on the calling thread.
    std::exception_ptr fft_error;
    auto run_fft = [&](auto &&operation) {
        try {
            operation();
        } catch (...) {
            #pragma omp critical(imrmlp_fft_error)
            {
                if (!fft_error) fft_error = std::current_exception();
            }
        }
    };

    #define COMPUTE_FFT_BLOCK(param) \
        run_fft([&]() { \
        for (int j = 0; j < nframe; ++j) { \
            buffer_data[j] = atoms[j].param; \
        } \
        alglib::samplemoments(buffer_data, nframe, mean, variance, skewness, kurtosis); \
        if (variance < 1e-3) variance = 1; \
        for (int l = 0; l < nframe; ++l) { \
            buffer_data[l] = (buffer_data[l] - mean) / variance; \
        } \
        fill_data(buffer_data, nframe); \
        alglib::fftr1d(buffer_data, buffer_fft); \
        applywindow(buffer_fft, wtype, param_c); \
        alglib::fftr1dinv(buffer_fft, buffer_data); \
        for (int j = 0; j < nframe; ++j) { \
            atoms[j].param = buffer_data[j] * variance + mean; \
        } \
        });
// global vector/matrix props
#pragma omp parallel
{
    // create buffers
    alglib::real_1d_array buffer_data;
    alglib::complex_1d_array buffer_fft;
    buffer_data.setlength(buffer_len);
    buffer_fft.setlength(buffer_len);
    double mean, variance, skewness, kurtosis;
    param_type param_c = param_curr;
    #pragma omp for
    for (int i = 0; i < nframe; ++i)
        if (!cartesian_positions) atoms[i].to_fractional();
    // cell
    #pragma omp for
    for (int i = 0; i < 9; ++i) {
        int s1 = floor(i / 3);
        int s2 = i % 3;
        COMPUTE_FFT_BLOCK(cell_(s1, s2))
    }
    // stress
    if (have_stress) {
        #pragma omp for
        for (int i = 0; i < 9; ++i) {
            int s1 = floor(i / 3);
            int s2 = i % 3;
            COMPUTE_FFT_BLOCK(stress_(s1, s2))
        }
    }
    // global vector
    for (auto &it : atoms[0].global_attribs_v_) {
        #pragma omp for
        for (int ss = 0; ss < it.second.size(); ++ss) {
            COMPUTE_FFT_BLOCK(global_attribs_v_.at(it.first)(ss))
        }
    }
}
    if (fft_error) std::rethrow_exception(fft_error);
// atomic props
#pragma omp parallel
{
    // create buffers
    alglib::real_1d_array buffer_data;
    alglib::complex_1d_array buffer_fft;
    buffer_data.setlength(buffer_len);
    buffer_fft.setlength(buffer_len);
    double mean, variance, skewness, kurtosis;
    param_type param_c = param_curr;

    // #pragma omp for
    // for (int i = 0; i < nframe; ++i)
    //     atoms[i].to_fractional();

    // smooth properties
    #pragma omp for
    for (int i = 0; i < natoms; ++i) {
        // position
        for (int k = 0; k < 3; ++k) {
            COMPUTE_FFT_BLOCK(pos_[i](k))
        }
        // force
        if (have_force && !rebuild_force)
        for (int k = 0; k < 3; ++k) {
            COMPUTE_FFT_BLOCK(forces_[i](k))
        }
        // velocity
        if (have_velo && !rebuild_velo)
        for (int k = 0; k < 3; ++k) {
            COMPUTE_FFT_BLOCK(velocities_[i](k))
        }
        // siteen
        if (have_siteen) {
            COMPUTE_FFT_BLOCK(site_energies_[i])
        }
        // local_d
        for (auto &it : atoms[0].local_attribs_d_) {
            COMPUTE_FFT_BLOCK(local_attribs_d_.at(it.first)(i))
        }
        // local_v
        for (auto &it : atoms[0].local_attribs_v_) {
            auto name = it.first;
            for (int k = 0; k < it.second.rows(); ++k) {
                COMPUTE_FFT_BLOCK(local_attribs_v_.at(name)(k,i))
            }
        }
    }

    #pragma omp for
    for (int i = 0; i < nframe; ++i)
        if (!cartesian_positions) atoms[i].to_cartesian();
}

    if (fft_error) std::rethrow_exception(fft_error);

    if (reconstruct && (rebuild_velo || rebuild_force)) {
        // Differentiate the smoothed Cartesian trajectory, including cell motion.
        // The FFT period is the length of the mirrored sequence times dt.
        const double deltaq = 2 * M_PI / (buffer_len * dt);
        #pragma omp parallel for
        for (int i = 0; i < natoms; ++i) {
            run_fft([&]() {
                alglib::real_1d_array data;
                alglib::complex_1d_array spectrum;
                data.setlength(buffer_len);
                for (int k = 0; k < 3; ++k) {
                    for (int j = 0; j < nframe; ++j) data[j] = atoms[j].pos_[i](k);
                    fill_data(data, nframe);
                    alglib::fftr1d(data, spectrum);
                    if (rebuild_force) {
                        auto acceleration = spectrum;
                        for (int q = 0; q < acceleration.length(); ++q) {
                            const int frequency = q <= buffer_len / 2 ? q : q - buffer_len;
                            const double omega = frequency * deltaq;
                            acceleration[q] *= -omega * omega;
                        }
                        alglib::fftr1dinv(acceleration, data);
                        // CODATA 2018 amu in kg; A/fs² -> m/s²; eV in J.
                        constexpr double mass_acceleration_to_force = 1.66053906660e-27 * 1e10 / 1.602176634e-19;
                        const double mass = atoms.front().local_attribs_d_.at("masses")[i];
                        for (int j = 0; j < nframe; ++j)
                            atoms[j].forces_[i](k) = mass_acceleration_to_force * mass * data[j];
                    }
                    if (rebuild_velo) {
                        for (int q = 0; q < spectrum.length(); ++q) {
                            int frequency = q <= buffer_len / 2 ? q : q - buffer_len;
                            if (buffer_len % 2 == 0 && q == buffer_len / 2) frequency = 0;
                            spectrum[q] *= alglib::complex(0, frequency * deltaq);
                        }
                        alglib::fftr1dinv(spectrum, data);
                        for (int j = 0; j < nframe; ++j) atoms[j].velocities_[i](k) = data[j];
                    }
                }
            });
        }
        if (fft_error) std::rethrow_exception(fft_error);
    }

    // global scalar props
{
    // create buffers
    alglib::real_1d_array buffer_data;
    alglib::complex_1d_array buffer_fft;
    buffer_data.setlength(buffer_len);
    buffer_fft.setlength(buffer_len);
    double mean, variance, skewness, kurtosis;
    param_type param_c = param_curr;
    // energy
    if (have_energy) {
        COMPUTE_FFT_BLOCK(energy_)
    }
    // fenergy
    if (have_fenergy) {
        COMPUTE_FFT_BLOCK(fenergy_)
    }
    // global scalar
    for (auto &it : atoms[0].global_attribs_d_) {
        COMPUTE_FFT_BLOCK(global_attribs_d_.at(it.first))
    }
}
    if (fft_error) std::rethrow_exception(fft_error);
    #undef COMPUTE_FFT_BLOCK
}

namespace {
bool fft_unsupported_calculation(const std::string& key) {
    return key=="hessian" || key=="hvp_directions" || key=="hvp_targets" ||
        key=="charges" || key=="charge" || key=="magmom" || key=="magmoms" ||
        key=="eff_charges" || key=="born_effective_charges";
}
void clear_fft_unsupported_calculations(Atoms& atom) {
    atom.have_hessian=false;atom.hessian_.resize(0,0);atom.clear_hvp();
    atom.charges_.clear();atom.magmom_scalar_.clear();atom.magmoms_.clear();
    atom.have_eff_chg=false;atom.born_eff_chgs_.clear();
    const auto erase=[](auto& attributes) {
        for(auto it=attributes.begin();it!=attributes.end();) {
            if(fft_unsupported_calculation(it->first)) it=attributes.erase(it);else ++it;
        }
    };
    erase(atom.features_);erase(atom.global_attribs_d_);erase(atom.global_attribs_v_);
    erase(atom.local_attribs_d_);erase(atom.local_attribs_v_);
}
}

void ComputeSmoothFFT::compute2(std::vector<Atoms>& atoms,double dt,int wtype) {
    if(interpolation_factor<1) throw std::invalid_argument("FFT interpolation_factor must be a positive integer");
    if(interpolation_factor==1) {compute2_base(atoms,dt,wtype,true);return;}
    if(atoms.size()>size_t(std::numeric_limits<int>::max())) throw std::length_error("Too many FFT frames");
    const int n=static_cast<int>(atoms.size()),q=interpolation_factor;
    const int count=FFTResample::output_size(n,q,even_seq);
    validate_fft_input(atoms,dt,params.freq_cutoff,rebuild_velo,rebuild_force);
    if(!(dt/q>0.) || !std::isfinite(dt/q)) throw std::invalid_argument("FFT interpolated timestep must be finite and positive");
    // Mass is atom identity metadata, not a time-dependent Fourier channel.
    const auto masses=atoms.front().local_attribs_d_.find("masses");
    if(masses!=atoms.front().local_attribs_d_.end()) {
        if(!masses->second.allFinite() || (masses->second.array()<=0).any())
            throw std::invalid_argument("FFT interpolation requires positive fixed masses when supplied");
        for(const auto& frame:atoms) if((frame.local_attribs_d_.at("masses").array()!=masses->second.array()).any())
            throw std::invalid_argument("FFT interpolation requires the same per-atom masses in every frame");
    }
    // Preserve the input on bad interpolation geometry or any worker failure.
    auto smoothed=atoms;
    compute2_base(smoothed,dt,wtype,false);
    const auto& first=atoms.front();
    std::vector<Atoms> output;output.reserve(count);
    for(int m=0;m<count;++m) {
        output.push_back(smoothed[std::min(m/q,n-1)]);
        auto& frame=output.back();
        clear_fft_unsupported_calculations(frame);
        if(!first.has_forces() && !rebuild_force) frame.forces_.clear();
        if(!first.has_velo() && !rebuild_velo) frame.velocities_.clear();
        if(!first.has_energy()) frame.have_energy=false;
        if(!first.has_fenergy()) frame.have_fenergy=false;
        if(!first.has_stresses()) frame.have_stress=false;
        if(!first.has_site_energies()) frame.site_energies_.clear();
    }
    const auto resample=[&](auto read,auto write,int derivative=0) {
        std::vector<double> samples(n);for(int j=0;j<n;++j) samples[j]=read(smoothed[j]);
        const auto values=FFTResample::interpolate(samples,q,even_seq,dt,derivative);
        for(int m=0;m<count;++m) write(output[m],values[m]);
    };
    for(int a=0;a<3;++a) for(int b=0;b<3;++b) {
        resample([&](const Atoms& f){return f.cell_(a,b);},[&](Atoms& f,double v){f.cell_(a,b)=v;});
        if(first.has_stresses()) resample([&](const Atoms& f){return f.stress_(a,b);},[&](Atoms& f,double v){f.stress_(a,b)=v;});
    }
    for(int i=0;i<first.size();++i) {
        for(int a=0;a<3;++a) {
            resample([&](const Atoms& f){return f.pos_[i][a];},[&](Atoms& f,double v){f.pos_[i][a]=v;});
            if(rebuild_velo) resample([&](const Atoms& f){return f.pos_[i][a];},[&](Atoms& f,double v){f.velocities_[i][a]=v;},1);
            else if(first.has_velo()) resample([&](const Atoms& f){return f.velocities_[i][a];},[&](Atoms& f,double v){f.velocities_[i][a]=v;});
            if(rebuild_force) {
                constexpr double conversion=1.66053906660e-27*1e10/1.602176634e-19;
                const double mass=first.local_attribs_d_.at("masses")[i];
                resample([&](const Atoms& f){return f.pos_[i][a];},[&](Atoms& f,double v){f.forces_[i][a]=conversion*mass*v;},2);
            } else if(first.has_forces()) resample([&](const Atoms& f){return f.forces_[i][a];},[&](Atoms& f,double v){f.forces_[i][a]=v;});
        }
        if(first.has_site_energies()) resample([&](const Atoms& f){return f.site_energies_[i];},[&](Atoms& f,double v){f.site_energies_[i]=v;});
    }
    if(first.has_energy()) resample([](const Atoms& f){return f.energy_;},[](Atoms& f,double v){f.energy_=v;});
    if(first.has_fenergy()) resample([](const Atoms& f){return f.fenergy_;},[](Atoms& f,double v){f.fenergy_=v;});
    for(const auto& [key,value]:first.global_attribs_d_) if(!fft_unsupported_calculation(key))
        resample([&](const Atoms& f){return f.global_attribs_d_.at(key);},[&](Atoms& f,double v){f.global_attribs_d_.at(key)=v;});
    for(const auto& [key,value]:first.global_attribs_v_) if(!fft_unsupported_calculation(key)) for(int k=0;k<value.size();++k)
        resample([&](const Atoms& f){return f.global_attribs_v_.at(key)[k];},[&](Atoms& f,double v){f.global_attribs_v_.at(key)[k]=v;});
    for(const auto& [key,value]:first.local_attribs_d_) if(!fft_unsupported_calculation(key) && key!="masses") for(int i=0;i<value.size();++i)
        resample([&](const Atoms& f){return f.local_attribs_d_.at(key)[i];},[&](Atoms& f,double v){f.local_attribs_d_.at(key)[i]=v;});
    for(const auto& [key,value]:first.local_attribs_v_) if(!fft_unsupported_calculation(key)) for(int k=0;k<value.rows();++k) for(int i=0;i<value.cols();++i)
        resample([&](const Atoms& f){return f.local_attribs_v_.at(key)(k,i);},[&](Atoms& f,double v){f.local_attribs_v_.at(key)(k,i)=v;});
    const bool require_invertible=!first.is_nonperiodic() && first.cell_.fullPivLu().isInvertible();
    const bool negative=std::signbit(first.cell_.determinant());
    for(int m=0;m<count;++m) {
        auto& frame=output[m];
        if(!frame.cell_.allFinite() || (require_invertible && (!frame.cell_.fullPivLu().isInvertible()
            || std::signbit(frame.cell_.determinant())!=negative)))
            throw std::invalid_argument("FFT interpolation produced a singular or inverted periodic cell");
        if(masses!=first.local_attribs_d_.end()) frame.local_attribs_d_["masses"]=masses->second;
        else frame.local_attribs_d_.erase("masses");
        // This generated name has a single canonical numeric representation.
        // In particular, a previous CFG output may carry a Feature alias.
        frame.features_.erase("fft_elapsed_fs");frame.global_attribs_i_.erase("fft_elapsed_fs");
        frame.global_attribs_vi_.erase("fft_elapsed_fs");frame.global_attribs_v_.erase("fft_elapsed_fs");
        frame.id=m;frame.global_attribs_d_["fft_elapsed_fs"]=m*(dt/q);frame.cartesian=true;
    }
    atoms.swap(output);
}

/**
 * @brief 
 * 
 * @param arr 
 * @param wtype 
 * 
 * 1. Generate window function
 */
void ComputeSmoothFFT::applywindow(alglib::complex_1d_array &arr, int wtype, param_type &param) {
    int alength = arr.length();
    if (param.ncutoff == 0) {
        // A cutoff below the first frequency bin keeps only the mean (DC).
        for (int i = 1; i < alength; ++i) arr[i] = 0;
        return;
    }
    int half = floor(alength / 2.0 + 0.5);
    double halfd = alength / 2.0 + 0.5;
    static thread_local std::vector<double> funcval;
    // #pragma omp threadprivate(funcval)
    funcval.resize(alength);

    double x = 0;
    double dx = 1.0 / (alength - 1);

    switch (wtype) {
    case BLOCK:
        if (param.ncutoff >= alength) return;
        funcval.assign(alength, 0);
        //param.ncutoff = floor(param.freq_cutoff * param.dt * half);
        for (int i = std::max(0, half - param.ncutoff + 1);
             i < std::min(alength, half + param.ncutoff); ++i) {
            funcval[i] = 1;
        }
        break;
    case SUPERGAUSSIAN:
        // param.ncutoff = floor(pow( 2.0 * log2(2.0), 0.5 /  param.m ) * param.sigma);
        param.sigma = param.ncutoff / ( pow(2.0 * log(2), 0.5 / param.m) );
        for (int i = 0; i < alength; ++i) {
            x = i * dx;
            funcval[i] = exp(-0.5 * pow((i - halfd) / param.sigma, 2.0 * param.m));
        }
        break;
    }
    // roll funcval
    std::rotate(funcval.begin(), funcval.begin() + half, funcval.end());
    funcval[0] = 1; // A low-pass window must preserve the mean (DC).
    // multiply
    for (int i = 0; i < alength; ++i) {
        arr[i] *= funcval[i];
    }
}

void ComputeSmoothFFT::applywindow1(alglib::complex_1d_array &arr, int wtype, param_type &param) {
    int alength = arr.length();

    switch (wtype) {
    case BLOCK:
        for (int i = std::max(1, param.ncutoff); i < alength; ++i) {
            arr[i] = 0;
        }
        break;
    case SUPERGAUSSIAN:
        // param.ncutoff = floor(pow( 2.0 * log2(2.0), 0.5 /  param.m ) * param.sigma);
        param.sigma = param.ncutoff / ( pow(2.0 * log(2), 0.5 / param.m) );
        for (int i = 1; i < alength; ++i) {
            arr[i] *= exp(-0.5 * pow(i / param.sigma, 2.0 * param.m));
        }
        break;
    }
}

void ComputeSmoothFFT::fill_data(alglib::real_1d_array &arr, int len) {
    if (even_seq) {
        for (int i = 0; i < len; ++i) {
            arr[2*len - i - 1] = arr[i];
        }
    } else {
        for (int i = 1; i < len; ++i) {
            arr[2*len - i - 1] = arr[i];
        }
    }
}
