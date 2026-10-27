// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "les.h"
#include <set>
#include <sstream>
#include "potential_mtp.h"

#include <algorithm>
#include <complex>
#include <limits>

namespace Potentials {

namespace {
constexpr double two_pi = 2.0 * M_PI;

inline bool positive_hemisphere(int n1, int n2, int n3)
{
    if (n1 != 0) return n1 > 0;
    if (n2 != 0) return n2 > 0;
    return n3 > 0;
}
} // namespace

const char* LES::cutoff_mode_name(CutoffMode mode)
{
    return mode == CutoffMode::Smooth ? "smooth" : "reference_hard";
}

LES::CutoffMode LES::parse_cutoff_mode(const std::string& value)
{
    if (value == "smooth") return CutoffMode::Smooth;
    if (value == "reference_hard" || value == "hard")
        return CutoffMode::ReferenceHard;
    GPERROR("Unknown LES cutoff mode '" + value
        + "' (expected smooth or reference_hard)");
    return CutoffMode::Smooth;
}

void LES::reject_obsolete_parameter(const std::string& name)
{
    if (name == "lr_k_cutoff" || name == "les_output_scaling"
        || name == "les_stress_fd_nudge" || name == "enable_indirect_force"
        || name == "remove_net_charge")
        GPERROR("Unsupported LES field: " + name);
}

void LES::reject_obsolete_settings(const GPSettings& settings)
{
    for (const auto& [name, value] : settings.settings)
        reject_obsolete_parameter(name);
}

void LES::load_coulomb_definition(std::istream& in)
{
    LES definition;
    std::set<std::string> seen;
    std::string field;
    int version=0;
    while (in>>field) {
        reject_obsolete_parameter(field);
        if (field=="lr_sigma") field="les_sigma";
        if ((field.rfind("les_",0)!=0 && field!="remove_self_interaction")
            || field=="les_species_coeffs" || field=="les_moment_coeffs") continue;
        if (!seen.insert(field).second) GPERROR("Duplicate LES field: "+field);
        char eq;
        std::string value;
        if (!(in>>eq>>value) || eq!='=') GPERROR("Malformed LES field: "+field);
        const auto scalar=[&](auto& target) {
            std::istringstream input(value);
            if (!(input>>target) || !input.eof()) GPERROR("Invalid LES field value: "+field);
        };
        if (field=="les_version") scalar(version);
        else if (field=="les_kernel") {
            if (value!="coulomb") GPERROR("Expected Coulomb LES kernel");
        } else if (field=="les_sigma") scalar(definition.lr_sigma);
        else if (field=="les_dl") scalar(definition.dl);
        else if (field=="les_taper_fraction") scalar(definition.taper_fraction);
        else if (field=="les_cutoff_mode") definition.cutoff_mode=parse_cutoff_mode(value);
        else if (field=="remove_self_interaction") scalar(definition.remove_self_interaction);
        else if (field=="les_charge_treatment") {
            if (value!="uniform_background") GPERROR("LES v2 requires uniform_background charge treatment");
        } else if (field=="les_response_parity") {
            int parity;scalar(parity);
            if (parity!=1) GPERROR("MTP-M supports les_response_parity=1 only");
        } else GPERROR("Unknown Coulomb LES field: "+field);
    }
    if (!seen.count("les_version") || version!=format_version)
        GPERROR("MTP-M Coulomb metadata requires les_version=2");
    definition.validate();
    if (!std::isfinite(definition.taper_fraction)) GPERROR("LES taper_fraction must be finite");
    definition.use_recurrence=use_recurrence;
    definition.use_cache=use_cache;
    *this=std::move(definition);
}

void LES::save_coulomb_definition(std::ostream& out) const
{
    if (kernel!=Kernel::Coulomb) GPERROR("Expected Coulomb LES kernel");
    validate();
    if (!std::isfinite(taper_fraction)) GPERROR("LES taper_fraction must be finite");
    out<<"les_version = "<<format_version<<"\nles_kernel = coulomb\n"
       <<"les_sigma = "<<lr_sigma<<"\nles_dl = "<<dl<<'\n'
       <<"les_cutoff_mode = "<<cutoff_mode_name(cutoff_mode)<<'\n'
       <<"les_taper_fraction = "<<taper_fraction<<'\n'
       <<"les_charge_treatment = uniform_background\n"
       <<"remove_self_interaction = "<<remove_self_interaction<<'\n';
}

void LES::validate(int channel) const
{
    if (kernel == Kernel::Coulomb && (!(lr_sigma > 0.0) || !std::isfinite(lr_sigma)))
        GPERROR("LES les_sigma must be finite and positive");
    if (kernel == Kernel::SOG) {
        if (sog_channels<1 || sog_channels>16 || channel<0 || channel>=sog_channels
            || sog_weights.empty() || sog_weights.size()%sog_channels || channel_terms()>256
            || sog_weights.size() != sog_log_bandwidths.size())
            GPERROR("SOG requires matching nonempty weights and log bandwidths");
        for (size_t m=0; m<sog_weights.size(); ++m) {
            const double u=sog_log_bandwidths[m];
            if (!std::isfinite(sog_weights[m]) || !std::isfinite(u)
                || u < -30.0 || u > 30.0)
                GPERROR("SOG requires finite weights and log bandwidths in [-30,30]");
        }
    }
    if (!(dl > 0.0) || !std::isfinite(dl))
        GPERROR("LES les_dl must be finite and positive");
    if (cutoff_mode == CutoffMode::Smooth
        && (!(taper_fraction > 0.0) || taper_fraction >= 1.0
            || !std::isfinite(taper_fraction)))
        GPERROR("LES taper_fraction must lie in (0,1)");
}

void LES::validate_periodic_geometry(const Atoms& atom)
{
    if (!atom.is_fully_periodic())
        GPERROR("LES requires periodic boundaries in all three lattice directions");
    if (!atom.cell_.allFinite())
        GPERROR("LES cell contains non-finite values");
    const double volume=atom.get_volume();
    if (!(volume>1e-12) || !std::isfinite(volume))
        GPERROR("LES requires a finite, right-handed, non-singular cell");
    for (const auto& position:atom.pos_)
        if (!position.allFinite())
            GPERROR("LES atomic positions contain non-finite values");
}

double LES::cutoff_weight(double k, double& derivative_wrt_k2) const
{
    derivative_wrt_k2 = 0.0;
    const double kmax = two_pi / dl;
    if (cutoff_mode == CutoffMode::ReferenceHard) return 1.0;
    if (k >= kmax) return 0.0;

    const double kon = kmax * (1.0 - taper_fraction);
    if (k <= kon) return 1.0;

    const double width = kmax - kon;
    const double x = (k - kon) / width;
    const double x2 = x * x;
    const double x3 = x2 * x;
    const double x4 = x3 * x;
    const double x5 = x4 * x;
    const double w = 1.0 - 10.0 * x3 + 15.0 * x4 - 6.0 * x5;
    const double dwdx = -30.0 * x2 + 60.0 * x3 - 30.0 * x4;
    derivative_wrt_k2 = dwdx / (width * 2.0 * k);
    return w;
}

void LES::kernel_value(double k2, double& value, double& derivative, const Workspace& workspace, int channel) const
{
    if (kernel == Kernel::Coulomb) {
        value = 4.0 * M_PI * coulomb_constant
            * std::exp(-0.5 * lr_sigma * lr_sigma * k2) / k2;
        derivative = value * (-0.5 * lr_sigma * lr_sigma - 1.0 / k2);
        return;
    }
    value = derivative = 0.0;
    for (int m=channel*channel_terms(); m<(channel+1)*channel_terms(); ++m) {
        const double inv_s2=workspace.inverse_widths_[m];
        const double term=sog_weights[m]*std::exp(-k2*inv_s2);
        value += term;
        derivative -= inv_s2*term;
    }
}

double LES::self_coefficient(int channel) const
{
    if (kernel == Kernel::Coulomb)
        return 2.0*coulomb_constant/(lr_sigma*std::sqrt(2.0*M_PI));
    double value=0.0;
    for (int m=channel*channel_terms(); m<(channel+1)*channel_terms(); ++m)
        value += sog_weights[m]*std::exp(3.0*sog_log_bandwidths[m])
            /(8.0*std::pow(M_PI,1.5));
    return value;
}

void LES::generate_k_vectors(const Atoms& atom, KVectorList& out, bool update_kernel, Workspace& workspace, int channel) const
{
    validate(channel);
    validate_periodic_geometry(atom);
    const double volume = atom.get_volume();
    if (!(volume > 1e-12) || !std::isfinite(volume))
        GPERROR("LES requires a finite, right-handed periodic cell");

    const bool zero = kernel == Kernel::SOG && sog_include_zero;
    const bool new_grid = !use_cache || !workspace.grid_valid_ || workspace.cached_cell_ != atom.cell_
        || workspace.cached_dl_ != dl || workspace.cached_taper_ != taper_fraction || workspace.cached_cutoff_ != cutoff_mode
        || workspace.cached_zero_ != zero;
    if (new_grid) {
        workspace.cached_grid_.clear();
        workspace.k_indices_.clear();
        workspace.grid_factors_.clear();
        workspace.grid_taper_.clear();
        workspace.grid_taper_derivative_.clear();
        workspace.phase_bounds_ = { 0, 0, 0 };
        const auto append = [&](const Eigen::Vector3d& k, int n1, int n2, int n3, double factor,
                                double taper, double derivative) {
            workspace.cached_grid_.push_back(k);
            workspace.k_indices_.push_back({ n1, n2, n3 });
            workspace.grid_factors_.push_back(factor);
            workspace.grid_taper_.push_back(taper);
            workspace.grid_taper_derivative_.push_back(derivative);
            for (int a = 0; a < 3; ++a)
                workspace.phase_bounds_[a] = std::max(workspace.phase_bounds_[a], std::abs(workspace.k_indices_.back()[a]));
        };
        if (zero)
            append(Eigen::Vector3d::Zero(), 0, 0, 0, 1., 1., 0.);
        const Eigen::Matrix3d reciprocal = atom.get_recip();
        const double kmax = two_pi / dl;
        const auto bound = [&](int a) {
            double x = atom.cell_.row(a).norm() / dl;
            return std::max(
                1, static_cast<int>(cutoff_mode == CutoffMode::ReferenceHard ? x : std::ceil(x)));
        };
        const int nmax1 = bound(0), nmax2 = bound(1), nmax3 = bound(2);
        for (int n1 = -nmax1; n1 <= nmax1; ++n1)
            for (int n2 = -nmax2; n2 <= nmax2; ++n2)
                for (int n3 = -nmax3; n3 <= nmax3; ++n3) {
                    if (!positive_hemisphere(n1, n2, n3))
                        continue;
                    Eigen::Vector3d k
                        = (n1 * reciprocal.row(0) + n2 * reciprocal.row(1) + n3 * reciprocal.row(2))
                              .transpose();
                    const double kmag = k.norm();
                    if (cutoff_mode == CutoffMode::ReferenceHard ? kmag > kmax : kmag >= kmax)
                        continue;
                    double derivative;
                    const double taper = cutoff_weight(kmag, derivative);
                    if (taper != 0.)
                        append(k, n1, n2, n3, 2., taper, derivative);
                }
        workspace.cached_cell_ = atom.cell_;
        workspace.cached_dl_ = dl;
        workspace.cached_taper_ = taper_fraction;
        workspace.cached_cutoff_ = cutoff_mode;
        workspace.cached_zero_ = zero;
        workspace.grid_valid_ = true;
    }
    if (new_grid) workspace.kernel_valid_ = false;
    if (!update_kernel) {
        out = workspace.cached_grid_;
        workspace.kvec_factors = workspace.grid_factors_;
        return;
    }
    if (new_grid || workspace.cached_channel_!=channel || workspace.cached_channels_!=sog_channels || !workspace.kernel_valid_ || workspace.cached_kernel_ != kernel || workspace.cached_sigma_ != lr_sigma
        || workspace.cached_weights_ != sog_weights || workspace.cached_widths_ != sog_log_bandwidths) {
        workspace.inverse_widths_.resize(sog_log_bandwidths.size());
        for (size_t m = 0; m < workspace.inverse_widths_.size(); ++m)
            workspace.inverse_widths_[m] = std::exp(-2 * sog_log_bandwidths[m]);
        workspace.cached_values_.resize(workspace.cached_grid_.size());
        workspace.cached_derivatives_.resize(workspace.cached_grid_.size());
        for (size_t i = 0; i < workspace.cached_grid_.size(); ++i) {
            double g, gp;
            kernel_value(workspace.cached_grid_[i].squaredNorm(), g, gp,workspace,channel);
            workspace.cached_values_[i] = workspace.grid_factors_[i] * workspace.grid_taper_[i] * g;
            workspace.cached_derivatives_[i]
                = workspace.grid_factors_[i] * (workspace.grid_taper_[i] * gp + workspace.grid_taper_derivative_[i] * g);
        }
        workspace.cached_channel_=channel;workspace.cached_channels_=sog_channels;
        workspace.cached_kernel_ = kernel;
        workspace.cached_sigma_ = lr_sigma;
        workspace.cached_weights_ = sog_weights;
        workspace.cached_widths_ = sog_log_bandwidths;
        workspace.kernel_valid_ = true;
    }
    out = workspace.cached_grid_;
    workspace.kvec_factors = workspace.grid_factors_;
    workspace.k_kernel = workspace.cached_values_;
    workspace.k_kernel_derivative = workspace.cached_derivatives_;
}

const ReciprocalPhases& LES::make_phases(const Atoms& atom, Workspace& workspace) const
{
    auto& phases=workspace.phases;
    std::array<std::array<double, 3>, 3> reciprocal;
    const auto b = atom.get_recip();
    for (int a = 0; a < 3; ++a)
        for (int d = 0; d < 3; ++d)
            reciprocal[a][d] = b(a, d);
    phases.prepare(
        atom.size(), workspace.phase_bounds_, reciprocal, [&](int i, int d) { return atom.pos_[i](d); },
        use_recurrence);
    return phases;
}

LES::Evaluation LES::evaluate(const Atoms& atom, const Eigen::VectorXd& q,
                              bool need_charge_hessian, Workspace& workspace, int channel) const
{
    const int n = atom.size();
    if (q.size() != n)
        GPERROR("LES charge count does not match atom count");
    if (!q.allFinite())
        GPERROR("LES charges contain non-finite values");
    generate_k_vectors(atom,workspace.kvecs,true,workspace,channel);

    const double volume = atom.get_volume();
    const double energy_prefactor = 0.5 / volume;
    const double gradient_prefactor = 2.0 * energy_prefactor;

    Evaluation result;
    result.forces.setZero(n, 3);
    result.per_atom_virial.setZero(n, 6);
    result.dEdq.setZero(n);
    result.per_atom_energy.setZero(n);
    if (need_charge_hessian) result.charge_hessian.setZero(n, n);

    auto& c=workspace.cosine;auto& s=workspace.sine;c.resize(n);s.resize(n);
    const auto& phases=make_phases(atom,workspace);
    for (std::size_t ik = 0; ik < workspace.kvecs.size(); ++ik) {
        const Eigen::Vector3d& k = workspace.kvecs[ik];
        phases.evaluate(workspace.k_indices_[ik],{k(0),k(1),k(2)},c.data(),s.data());
        double sr = 0.0;
        double si = 0.0;
        for (int i = 0; i < n; ++i) {
            sr += q(i) * c[i];
            si += q(i) * s[i];
        }
        const double sk2 = sr * sr + si * si;
        const double h = workspace.k_kernel[ik];
        const double hp = workspace.k_kernel_derivative[ik];
        const double mode_energy = energy_prefactor * h * sk2;
        result.energy += mode_energy;
        result.virial += mode_energy * Eigen::Matrix3d::Identity()
            + (2.0 * energy_prefactor * hp * sk2) * (k * k.transpose());

        for (int i = 0; i < n; ++i) {
            const double re = c[i] * sr + s[i] * si;
            const double im = s[i] * sr - c[i] * si;
            result.dEdq(i) += gradient_prefactor * h * re;
            result.forces.row(i) +=
                (gradient_prefactor * h * q(i) * im * k).transpose();
            const double atom_energy = energy_prefactor * h * q(i) * re;
            const double atom_kernel_derivative = 2.0 * energy_prefactor * hp * q(i) * re;
            result.per_atom_energy(i) += atom_energy;
            result.per_atom_virial(i,0) += atom_energy + atom_kernel_derivative * k(0) * k(0);
            result.per_atom_virial(i,1) += atom_energy + atom_kernel_derivative * k(1) * k(1);
            result.per_atom_virial(i,2) += atom_energy + atom_kernel_derivative * k(2) * k(2);
            result.per_atom_virial(i,3) += atom_kernel_derivative * k(0) * k(1);
            result.per_atom_virial(i,4) += atom_kernel_derivative * k(0) * k(2);
            result.per_atom_virial(i,5) += atom_kernel_derivative * k(1) * k(2);
        }
        if (need_charge_hessian) {
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < n; ++j)
                    result.charge_hessian(i, j) += gradient_prefactor * h
                        * (c[i] * c[j] + s[i] * s[j]);
        }
    }

    if (remove_self_interaction) {
        const double self_prefactor = 0.5 * self_coefficient(channel);
        for (int i = 0; i < n; ++i) {
            const double self = -self_prefactor * q(i) * q(i);
            result.energy += self;
            result.per_atom_energy(i) += self;
            result.dEdq(i) -= 2.0 * self_prefactor * q(i);
            if (need_charge_hessian)
                result.charge_hessian(i, i) -= 2.0 * self_prefactor;
        }
    }
    return result;
}

Eigen::VectorXd LES::apply_charge_hessian(const Atoms& atom, const Eigen::VectorXd& v, Workspace& workspace, int channel) const
{
    if (v.size() != atom.size() || !v.allFinite())
        GPERROR("Invalid LES charge Hessian vector");
    generate_k_vectors(atom,workspace.kvecs,true,workspace,channel);
    const int n = atom.size();
    const double inverse_volume = 1. / atom.get_volume();
    Eigen::VectorXd out = Eigen::VectorXd::Zero(n);
    auto& c=workspace.cosine;auto& s=workspace.sine;c.resize(n);s.resize(n);
    const auto& phases=make_phases(atom,workspace);
    for (size_t ik = 0; ik < workspace.kvecs.size(); ++ik) {
        const auto& k = workspace.kvecs[ik];
        phases.evaluate(workspace.k_indices_[ik], { k(0), k(1), k(2) }, c.data(), s.data());
        double re = 0., im = 0.;
        for (int i = 0; i < n; ++i) {
            re += v(i) * c[i];
            im += v(i) * s[i];
        }
        const double factor = inverse_volume * workspace.k_kernel[ik];
        for (int i = 0; i < n; ++i)
            out(i) += factor * (c[i] * re + s[i] * im);
    }
    if (remove_self_interaction)
        out -= self_coefficient(channel) * v;
    return out;
}

LES::KernelComponents LES::kernel_components(const Atoms& atom,const Eigen::MatrixXd& q, Workspace& workspace) const
{
    if(kernel!=Kernel::SOG || q.rows()!=atom.size() || q.cols()!=sog_channels || !q.allFinite())
        GPERROR("Invalid SOG kernel component inputs");
    generate_k_vectors(atom,workspace.kvecs,true,workspace,0);
    const int n=atom.size(), terms=channel_terms(), count=sog_weights.size();
    KernelComponents out;
    out.efs=Eigen::MatrixXd::Zero(1+3*n+9,count);
    out.dEdq=Eigen::MatrixXd::Zero(n,count);
    const double iv=1./atom.get_volume();
    auto& cosine=workspace.cosine;auto& sine=workspace.sine;cosine.resize(n);sine.resize(n);
    const auto& phases=make_phases(atom,workspace);
    Eigen::VectorXd re(n),im(n);
    for(size_t ik=0;ik<workspace.kvecs.size();++ik) {
        const auto& k=workspace.kvecs[ik];const double x=k.squaredNorm();
        phases.evaluate(workspace.k_indices_[ik],{k[0],k[1],k[2]},cosine.data(),sine.data());
        double tp;const double taper=cutoff_weight(std::sqrt(x),tp);
        for(int c=0;c<sog_channels;++c) {
            const double sr=q.col(c).dot(Eigen::Map<Eigen::VectorXd>(cosine.data(),n));
            const double si=q.col(c).dot(Eigen::Map<Eigen::VectorXd>(sine.data(),n));
            const double sk2=sr*sr+si*si;
            for(int i=0;i<n;++i) {re[i]=cosine[i]*sr+sine[i]*si;im[i]=sine[i]*sr-cosine[i]*si;}
            for(int m=c*terms;m<(c+1)*terms;++m) {
                const double inv_s2=workspace.inverse_widths_[m];
                const double gaussian=std::exp(-x*inv_s2)*workspace.kvec_factors[ik];
                const double h=gaussian*taper, hp=gaussian*(tp-inv_s2*taper);
                const double energy=.5*iv*h*sk2;
                out.efs(0,m)+=energy;
                for(int i=0;i<n;++i) {
                    out.dEdq(i,m)+=iv*h*re[i];
                    for(int d=0;d<3;++d) out.efs(1+3*i+d,m)+=iv*h*q(i,c)*im[i]*k[d];
                }
                for(int i=0;i<3;++i) for(int j=0;j<3;++j)
                    out.efs(1+3*n+3*i+j,m)+=(i==j ? energy : 0.)+iv*hp*sk2*k[i]*k[j];
            }
        }
    }
    if(remove_self_interaction) for(int c=0;c<sog_channels;++c)
        for(int m=c*terms;m<(c+1)*terms;++m) {
            const double coefficient=std::exp(3.*sog_log_bandwidths[m])/(8.*std::pow(M_PI,1.5));
            out.efs(0,m)-=.5*coefficient*q.col(c).squaredNorm();
            out.dEdq.col(m)-=coefficient*q.col(c);
        }
    return out;
}

LES::ChannelEvaluation LES::evaluate_channels(const Atoms& atom,const Eigen::MatrixXd& q, Workspace& workspace) const
{
    validate(0);
    const int count=kernel==Kernel::SOG ? sog_channels : 1;
    if(q.rows()!=atom.size() || q.cols()!=count || !q.allFinite())GPERROR("Invalid LES channel matrix");
    ChannelEvaluation out;out.forces.setZero(atom.size(),3);out.dEdq.resize(atom.size(),count);
    out.per_atom_energy.setZero(atom.size());
    out.per_atom_virial.setZero(atom.size(),6);
    for(int c=0;c<count;++c){
        const auto value=evaluate(atom,q.col(c),workspace,c);
        out.energy+=value.energy;out.forces+=value.forces;out.virial+=value.virial;
        out.per_atom_energy+=value.per_atom_energy;out.dEdq.col(c)=value.dEdq;
        out.per_atom_virial+=value.per_atom_virial;
    }
    return out;
}

Eigen::MatrixXd LES::apply_charge_hessian_channels(const Atoms& atom,const Eigen::MatrixXd& v, Workspace& workspace) const
{
    validate(0);
    const int count=kernel==Kernel::SOG ? sog_channels : 1;
    if(v.rows()!=atom.size() || v.cols()!=count || !v.allFinite())GPERROR("Invalid LES channel Hessian matrix");
    Eigen::MatrixXd out(atom.size(),count);
    for(int c=0;c<count;++c){out.col(c)=apply_charge_hessian(atom,v.col(c),workspace,c);}
    return out;
}

void LES::add_latent_ewald_efs(Atoms& atom, AnyNeighborhoods&,
                               Eigen::VectorXd& q_latent,
                               Eigen::VectorXd* dEdq_out, Workspace& workspace, int channel) const
{
    Evaluation result = evaluate(atom,q_latent,false,workspace,channel);
    if (static_cast<int>(atom.forces_.size()) < atom.size())
        atom.forces_.assign(atom.size(), Eigen::Vector3d::Zero());
    atom.energy_ += result.energy;
    atom.stress_ += result.virial;
    for (int i = 0; i < atom.size(); ++i)
        atom.forces_[i] += result.forces.row(i).transpose();
    if (dEdq_out) *dEdq_out = std::move(result.dEdq);
}

void LES::compute_les_dLdq_from_forces(const Atoms& atom,
    const Eigen::VectorXd& q,
    const Eigen::Matrix<double, -1, 3>& dLdF,
    const Eigen::Matrix3d& dLdS,
    Eigen::VectorXd& out, Workspace& workspace, int channel) const
{
    const int n = atom.size();
    out.setZero(n);
    if (q.size() != n || dLdF.rows() != n)
        GPERROR("LES force VJP dimensions do not match atom count");
    if (!q.allFinite() || !dLdF.allFinite() || !dLdS.allFinite())
        GPERROR("LES force VJP inputs contain non-finite values");
    generate_k_vectors(atom,workspace.kvecs,true,workspace,channel);
    const double volume = atom.get_volume();
    const double ep = 0.5 / volume;
    const double gp = 2.0 * ep;
    auto& c=workspace.cosine;auto& s=workspace.sine;auto& wk=workspace.weighted_k;
    c.resize(n);s.resize(n);wk.resize(n);
    const auto& phases=make_phases(atom,workspace);

    for (std::size_t ik = 0; ik < workspace.kvecs.size(); ++ik) {
        const auto& k = workspace.kvecs[ik];
        phases.evaluate(workspace.k_indices_[ik],{k(0),k(1),k(2)},c.data(),s.data());
        double sr = 0.0, si = 0.0;
        double weighted_sin = 0.0, weighted_cos = 0.0;
        for (int i = 0; i < n; ++i) {
            wk[i] = dLdF.row(i).dot(k.transpose());
            sr += q(i) * c[i];
            si += q(i) * s[i];
            weighted_sin += q(i) * wk[i] * s[i];
            weighted_cos += q(i) * wk[i] * c[i];
        }
        const double h = workspace.k_kernel[ik];
        const double hp = workspace.k_kernel_derivative[ik];
        const Eigen::Matrix3d stress_factor =
            h * Eigen::Matrix3d::Identity() + 2.0 * hp * (k * k.transpose());
        const double stress_weight = (dLdS.array() * stress_factor.array()).sum();

        for (int i = 0; i < n; ++i) {
            const double re = c[i] * sr + s[i] * si;
            const double im = s[i] * sr - c[i] * si;
            out(i) += gp * h * (wk[i] * im
                + c[i] * weighted_sin - s[i] * weighted_cos);
            out(i) += gp * stress_weight * re;
        }
    }
}

void LES::accumulate_kernel_vjp(const Atoms& atom, const Eigen::VectorXd& q,
    double dLdE, const Eigen::Matrix<double,-1,3>& dLdF,
    const Eigen::Matrix3d& dLdW, const Eigen::VectorXd& response,
    double* gradient, Workspace& workspace, int channel) const
{
    if (kernel != Kernel::SOG) return;
    const int n=atom.size(), terms=sog_weights.size();
    if (q.size()!=n || dLdF.rows()!=n || response.size()!=n
        || !q.allFinite() || !dLdF.allFinite() || !response.allFinite()
        || !dLdW.allFinite() || !std::isfinite(dLdE))
        GPERROR("Invalid SOG kernel VJP inputs");
    generate_k_vectors(atom,workspace.kvecs,true,workspace,channel);
    const double inv_volume=1.0/atom.get_volume();
    auto& c=workspace.cosine;auto& s=workspace.sine;c.resize(n);s.resize(n);
    const auto& phases=make_phases(atom,workspace);
    for (size_t ik=0; ik<workspace.kvecs.size(); ++ik) {
        const auto& k=workspace.kvecs[ik];
        const double x=k.squaredNorm();
        phases.evaluate(workspace.k_indices_[ik],{k(0),k(1),k(2)},c.data(),s.data());
        double sr=0.0,si=0.0;
        for (int i=0; i<n; ++i) {
            sr+=q(i)*c[i]; si+=q(i)*s[i];
        }
        const double sk2=sr*sr+si*si;
        double value_weight=0.5*(dLdE+dLdW.trace())*sk2;
        for (int i=0; i<n; ++i) {
            const double re=c[i]*sr+s[i]*si, im=s[i]*sr-c[i]*si;
            value_weight+=q(i)*im*dLdF.row(i).dot(k.transpose())+response(i)*re;
        }
        value_weight*=inv_volume*workspace.kvec_factors[ik];
        const double derivative_weight=sk2*k.dot(dLdW*k)
            *inv_volume*workspace.kvec_factors[ik];
        double tp;
        const double t=cutoff_weight(std::sqrt(x),tp);
        for (int m=channel*channel_terms(); m<(channel+1)*channel_terms(); ++m) {
            const double inv_s2=workspace.inverse_widths_[m];
            const double a=std::exp(-x*inv_s2);
            gradient[m]+=a*(value_weight*t+derivative_weight*(tp-t*inv_s2));
            gradient[terms+m]+=2.0*sog_weights[m]*a*inv_s2
                *(value_weight*x*t+derivative_weight*(x*tp+(1.0-x*inv_s2)*t));
        }
    }
    if (remove_self_interaction) {
        const double weight=-0.5*dLdE*q.squaredNorm()-response.dot(q);
        for (int m=channel*channel_terms(); m<(channel+1)*channel_terms(); ++m) {
            const double c=std::exp(3.0*sog_log_bandwidths[m])/(8.0*std::pow(M_PI,1.5));
            gradient[m]+=weight*c;
            gradient[terms+m]+=weight*3.0*sog_weights[m]*c;
        }
    }
}

void LES::compute_les_force_stress_grad(const Atoms& atom,
    const Eigen::VectorXd& q, const Eigen::MatrixXd& descriptors,
    const std::vector<int>& atom_types, int species_count,
    int alpha_scalar_moments,
    const Eigen::Matrix<double, -1, 3>& dLdF,
    const Eigen::Matrix3d& dLdS, double, double* out, Workspace& workspace, int channel) const
{
    Eigen::VectorXd dLdq;
    compute_les_dLdq_from_forces(atom,q,dLdF,dLdS,dLdq,workspace,channel);
    for (int i = 0; i < atom.size(); ++i)
        out[atom_types[i]] += dLdq(i);
    for (int a = 0; a < alpha_scalar_moments; ++a)
        out[species_count + a] += dLdq.dot(descriptors.col(a));
}

Eigen::Vector3cd LES::compute_periodic_dipole_fractional(
    const Atoms& atom, const Eigen::VectorXd& q)
{
    validate_periodic_geometry(atom);
    if (q.size() != atom.size())
        GPERROR("LES polarization charge count mismatch");
    if (!q.allFinite())
        GPERROR("LES polarization charges contain non-finite values");
    const Eigen::Matrix3d inv_cell = atom.cell_.inverse();
    Eigen::Vector3cd dipole = Eigen::Vector3cd::Zero();
    for (int alpha = 0; alpha < 3; ++alpha) {
        std::complex<double> structure = 0.0;
        for (int i = 0; i < atom.size(); ++i) {
            const Eigen::RowVector3d fractional = atom.pos_[i] * inv_cell;
            structure += q(i) * std::exp(std::complex<double>(
                0.0, two_pi * fractional(alpha)));
        }
        dipole(alpha) = structure / std::complex<double>(0.0, two_pi);
    }
    return dipole;
}

void LES::compute_bec(const Atoms& atom, const Eigen::VectorXd& q_latent,
                      std::vector<Eigen::Matrix3d>& bec_out,
                      bool remove_mean, double epsilon_factor) const
{
    if (kernel == Kernel::SOG)
        GPERROR("Generic SOG latent variables do not define physical Born charges");
    if (!(epsilon_factor > 0.0) || !std::isfinite(epsilon_factor))
        GPERROR("LES BEC epsilon_factor must be finite and positive");
    const int n = atom.size();
    if (q_latent.size() != n)
        GPERROR("LES BEC charge count does not match atom count");
    validate_periodic_geometry(atom);
    if (!q_latent.allFinite())
        GPERROR("LES BEC charges contain non-finite values");
    Eigen::VectorXd q = q_latent;
    if (remove_mean && n > 0) q.array() -= q.mean();
    const double scale = std::sqrt(epsilon_factor);
    bec_out.assign(n, Eigen::Matrix3d::Zero());

    const Eigen::Matrix3d inv_cell = atom.cell_.inverse();
    for (int i = 0; i < n; ++i) {
        const Eigen::RowVector3d fractional = atom.pos_[i] * inv_cell;
        for (int alpha = 0; alpha < 3; ++alpha) {
            const std::complex<double> phase = std::exp(std::complex<double>(
                0.0, two_pi * fractional(alpha)));
            for (int gamma = 0; gamma < 3; ++gamma) {
                // d p_frac(alpha) / d r_gamma = q exp(i2pi s_alpha)
                //                                      * (H^-1)_{gamma,alpha}
                const std::complex<double> dp = q(i) * phase * inv_cell(gamma, alpha);
                // Periodic BEC requires removal of the displaced atom's
                // phase before taking the real part.  For fixed q this makes
                // the direct response exactly q_i I at every position.
                const double dephased_dp = (std::conj(phase) * dp).real();
                for (int cart = 0; cart < 3; ++cart)
                    bec_out[i](cart, gamma) += scale
                        * dephased_dp * atom.cell_(alpha, cart);
            }
        }
    }
}

} // namespace Potentials

namespace Potentials {
LES::CartesianResponse LES::evaluate_cartesian_directional(const Atoms& atom,
    const Eigen::VectorXd& q, const Eigen::VectorXd& q_directional,
    const SiteCoordinateMatrix& velocity, Workspace& workspace, int channel) const
{
    const int n = atom.size();
    if (q.size() != n || q_directional.size() != n || velocity.rows() != n ||
        !q.allFinite() || !q_directional.allFinite() || !velocity.allFinite())
        GPERROR("Invalid LES Cartesian directional inputs");
    generate_k_vectors(atom,workspace.kvecs,true,workspace,channel);
    CartesianResponse response;
    response.force_directional = SiteCoordinateMatrix::Zero(n,3);
    response.dEdq = Eigen::VectorXd::Zero(n);
    response.dEdq_directional = Eigen::VectorXd::Zero(n);
    const double inverse_volume = 1./atom.get_volume();
    const auto& phases=make_phases(atom,workspace);
    auto& c=workspace.cosine;auto& s=workspace.sine;
    auto& cd=workspace.cosine_directional;auto& sd=workspace.sine_directional;
    c.resize(n);s.resize(n);cd.resize(n);sd.resize(n);
    for (size_t ik = 0; ik < workspace.kvecs.size(); ++ik) {
        const auto& k = workspace.kvecs[ik];
        phases.evaluate(workspace.k_indices_[ik],{k(0),k(1),k(2)},c.data(),s.data());
        double sr = 0.,si = 0.,srd = 0.,sid = 0.;
        for (int i = 0; i < n; ++i) {
            const double phase_directional = velocity.row(i).dot(k);
            cd[i] = -s[i]*phase_directional;
            sd[i] = c[i]*phase_directional;
            sr += q[i]*c[i];
            si += q[i]*s[i];
            srd += q_directional[i]*c[i]+q[i]*cd[i];
            sid += q_directional[i]*s[i]+q[i]*sd[i];
        }
        const double factor = inverse_volume*workspace.k_kernel[ik];
        for (int i = 0; i < n; ++i) {
            const double re = c[i]*sr+s[i]*si;
            const double red = cd[i]*sr+c[i]*srd+sd[i]*si+s[i]*sid;
            const double im = s[i]*sr-c[i]*si;
            const double imd = sd[i]*sr+s[i]*srd-cd[i]*si-c[i]*sid;
            response.dEdq[i] += factor*re;
            response.dEdq_directional[i] += factor*red;
            response.force_directional.row(i) +=
                (factor*(q_directional[i]*im+q[i]*imd)*k).transpose();
        }
    }
    if (remove_self_interaction) {
        const double self = self_coefficient(channel);
        response.dEdq -= self*q;
        response.dEdq_directional -= self*q_directional;
    }
    return response;
}

namespace {
struct LatentFC2Head {
    const double* species;
    const double* moments;
    int count;
    void propagate(const std::vector<double>& basis, double& charge,
        Eigen::VectorXd& gradient, Eigen::MatrixXd& hessian, int& type, int&) const {
        charge = species[type];
        gradient.setZero(basis.size());
        for (int a = 0; a < count; ++a) {
            charge += moments[a]*basis[a+1];
            gradient[a+1] = moments[a];
        }
        hessian.resize(0,0);
    }
};
}

Eigen::VectorXd latent_cartesian_hvp(PotMTP& basis, LES& engine,
    const Atoms& atom, AnyNeighborhoods& neighborhoods,
    const Eigen::VectorXd& direction, const double* species,
    const double* moments, int channels) {
    LES::Workspace les_workspace;
    if (direction.size() != 3*atom.size() || !direction.allFinite() ||
        neighborhoods.size() != atom.size() || !species || !moments || channels < 1)
        GPERROR("Invalid latent Cartesian Hessian-vector inputs");
    const int n = atom.size();
    const Eigen::Map<const SiteCoordinateMatrix> velocity(direction.data(),n,3);
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    Eigen::VectorXd q(n),qd(n);
    SiteCoordinateMatrix relative,gradient,head_product;
    PotMTP::ompbuffers unused;
    HessianMatrix unused_hessian;
    auto evaluate_head = [&](Neighborhood& neighborhood, LatentFC2Head& head,
                             int center, double& charge) {
        relative.resize(neighborhood.nbhcount_,3);
        for (int j = 0; j < neighborhood.nbhcount_; ++j)
            relative.row(j) = velocity.row(neighborhood.indices_[j])-velocity.row(center);
        basis.CalcSiteHessImpl<false,LatentFC2Head,0>(neighborhood,unused,charge,
            gradient,unused_hessian,head,0,&relative,&head_product);
    };
    for (int channel = 0; channel < channels; ++channel) {

        LatentFC2Head head{species+channel*basis.species_count,
            moments+channel*basis.alpha_scalar_moments,basis.alpha_scalar_moments};
        for (int i = 0; i < n; ++i) {
            evaluate_head(neighborhoods[i],head,i,q[i]);
            qd[i] = (gradient.array()*relative.array()).sum();
        }
        const auto response = engine.evaluate_cartesian_directional(atom,q,qd,velocity,les_workspace,channel);
        for (int i = 0; i < n; ++i) {
            result.segment<3>(3*i) -= response.force_directional.row(i).transpose();
            double charge;
            evaluate_head(neighborhoods[i],head,i,charge);
            // J^T*dot(lambda) + sum(lambda_i H_qi v). Together with the
            // fixed-q force differential this retains every mixed chain term.
            const SiteCoordinateMatrix local = response.dEdq_directional[i]*gradient
                                             +response.dEdq[i]*head_product;
            const auto& neighborhood = neighborhoods[i];
            for (int j = 0; j < neighborhood.nbhcount_; ++j) {
                result.segment<3>(3*neighborhood.indices_[j]) += local.row(j).transpose();
                result.segment<3>(3*i) -= local.row(j).transpose();
            }
        }
    }
    return result;
}

} // namespace Potentials

namespace Potentials {
LES::NativeResponse LES::evaluate_native_directional(const Atoms& atom,
    const Eigen::VectorXd& q, const Eigen::VectorXd& qd,
    const SiteCoordinateMatrix& velocity, const Eigen::Matrix3d& dc, Workspace& workspace, int channel) const
{
    const int n=atom.size();
    if(q.size()!=n || qd.size()!=n || velocity.rows()!=n || !q.allFinite() || !qd.allFinite()
        || !velocity.allFinite() || !dc.allFinite()) GPERROR("Invalid LES native directional inputs");
    if(cutoff_mode!=CutoffMode::Smooth && !dc.isZero())
        GPERROR("LES native cell curvature requires smooth reciprocal cutoff");
    generate_k_vectors(atom,workspace.kvecs,true,workspace,channel);
    const Eigen::Matrix3d inverse=atom.cell_.inverse();
    const double trace=(inverse*dc).trace(),alpha=1./atom.get_volume();
    NativeResponse out;
    out.forces=SiteCoordinateMatrix::Zero(n,3);
    out.force_directional=SiteCoordinateMatrix::Zero(n,3);
    out.dEdq=Eigen::VectorXd::Zero(n);out.dEdq_directional=Eigen::VectorXd::Zero(n);
    auto& c=workspace.cosine;auto& s=workspace.sine;
    auto& cd=workspace.cosine_directional;auto& sd=workspace.sine_directional;
    c.resize(n);s.resize(n);cd.resize(n);sd.resize(n);
    const auto& phases=make_phases(atom,workspace);
    for(size_t ik=0;ik<workspace.kvecs.size();++ik) {
        const auto& k=workspace.kvecs[ik];
        const Eigen::Vector3d kd=-inverse*dc*k;
        const double t=k.squaredNorm(),td=2*k.dot(kd),kmag=std::sqrt(t);
        double tp;const double taper=cutoff_weight(kmag,tp);
        double tpp=0.;
        const double kmax=two_pi/dl,kon=kmax*(1.-taper_fraction),width=kmax-kon;
        if(cutoff_mode==CutoffMode::Smooth && kmag>kon && kmag<kmax) {
            const double x=(kmag-kon)/width;
            const double first=-30*x*x+60*x*x*x-30*x*x*x*x;
            const double second=-60*x+180*x*x-120*x*x*x;
            tpp=second/(4*width*width*t)-first/(4*width*t*kmag);
        }
        double g,gp;kernel_value(t,g,gp,workspace,channel);
        double gpp=0.;
        if(kernel==Kernel::Coulomb) {
            const double a=.5*lr_sigma*lr_sigma;
            gpp=g*((a+1./t)*(a+1./t)+1./(t*t));
        } else for(int m=channel*channel_terms();m<(channel+1)*channel_terms();++m) {
            const double inv=workspace.inverse_widths_[m];
            gpp+=inv*inv*sog_weights[m]*std::exp(-t*inv);
        }
        const double h=workspace.k_kernel[ik],hp=workspace.k_kernel_derivative[ik];
        const double hpp=workspace.kvec_factors[ik]*(tpp*g+2*tp*gp+taper*gpp);
        const double hd=hp*td,hpd=hpp*td;
        phases.evaluate(workspace.k_indices_[ik],{k[0],k[1],k[2]},c.data(),s.data());
        double sr=0.,si=0.,srd=0.,sid=0.;
        for(int i=0;i<n;++i) {
            const double phase=velocity.row(i).dot(k)+atom.pos_[i].dot(kd);
            cd[i]=-s[i]*phase;sd[i]=c[i]*phase;
            sr+=q[i]*c[i];si+=q[i]*s[i];
            srd+=qd[i]*c[i]+q[i]*cd[i];sid+=qd[i]*s[i]+q[i]*sd[i];
        }
        const double sk2=sr*sr+si*si,sk2d=2*(sr*srd+si*sid);
        const double energy=.5*alpha*h*sk2;
        const double ed=.5*alpha*((hd-h*trace)*sk2+h*sk2d);
        out.virial+=energy*Eigen::Matrix3d::Identity()+alpha*hp*sk2*(k*k.transpose());
        out.virial_directional+=ed*Eigen::Matrix3d::Identity()
            +alpha*((hpd-hp*trace)*sk2+hp*sk2d)*(k*k.transpose())
            +alpha*hp*sk2*(kd*k.transpose()+k*kd.transpose());
        for(int i=0;i<n;++i) {
            const double re=c[i]*sr+s[i]*si,im=s[i]*sr-c[i]*si;
            const double red=cd[i]*sr+c[i]*srd+sd[i]*si+s[i]*sid;
            const double imd=sd[i]*sr+s[i]*srd-cd[i]*si-c[i]*sid;
            out.dEdq[i]+=alpha*h*re;
            out.dEdq_directional[i]+=alpha*((hd-h*trace)*re+h*red);
            out.forces.row(i)+=(alpha*h*q[i]*im*k).transpose();
            out.force_directional.row(i)+=(alpha*(((hd-h*trace)*q[i]*im
                +h*(qd[i]*im+q[i]*imd))*k+h*q[i]*im*kd)).transpose();
        }
    }
    if(remove_self_interaction) {
        const double self=self_coefficient(channel);
        out.dEdq-=self*q;out.dEdq_directional-=self*qd;
    }
    return out;
}

void assemble_cartesian_hessian(AnyPotential& potential,Atoms& atom,
    AnyNeighborhoods& neighborhoods,HessianMatrix& hessian) {
    LatentCurvatureCallbacks callbacks;
    callbacks.whole_hvp=[&](const Eigen::VectorXd& direction,Eigen::VectorXd& product) {
        potential.applyHessianVectorProduct(atom,neighborhoods,direction,product);
    };
    callbacks.finish_efs=[&] {potential.getEFS(atom,neighborhoods);};
    assemble_cartesian_hessian(potential,atom,neighborhoods,hessian,callbacks);
}
void assemble_cartesian_hessian(AnyPotential& potential,Atoms& atom,
    AnyNeighborhoods& neighborhoods,HessianMatrix& hessian,const LatentCurvatureCallbacks& callbacks) {
    if(!callbacks.whole_hvp || !callbacks.finish_efs) GPERROR("Incomplete caller-owned Hessian callbacks");
    if(!potential.supports_analytical_hessian()) GPERROR("Unsupported analytical atomic Hessian");
    HessianMatrix result;result.resize(atom.size());
    Eigen::VectorXd direction=Eigen::VectorXd::Zero(3*atom.size()),product;
    for(int j=0;j<atom.size();++j) for(int b=0;b<3;++b) {
        direction[3*j+b]=1.;callbacks.whole_hvp(direction,product);
        direction[3*j+b]=0.;
        for(int i=0;i<atom.size();++i) result(i,j).col(b)=product.segment<3>(3*i);
    }
    callbacks.finish_efs();hessian=std::move(result);
}
void assemble_requested_hessian(AnyPotential& potential,Atoms& atom,AnyNeighborhoods& neighborhoods,
    const std::vector<std::pair<int,int>>& indices,HessianSparse& hessian) {
    if(!potential.supports_analytical_hessian()) GPERROR("Unsupported analytical requested Hessian");
    HessianSparse result;result.size=atom.size();
    std::map<int,std::vector<int>> columns;
    for(auto [i,j]:indices) {
        if(i<0 || j<0 || i>=atom.size() || j>=atom.size()) GPERROR("Sparse Hessian atom index is out of range");
        if(i>j) std::swap(i,j);
        result(i,j).setZero();columns[j].push_back(i);
    }
    Eigen::VectorXd direction=Eigen::VectorXd::Zero(3*atom.size()),product;
    for(const auto& [j,rows]:columns) for(int b=0;b<3;++b) {
        direction[3*j+b]=1.;potential.applyHessianVectorProduct(atom,neighborhoods,direction,product);
        direction[3*j+b]=0.;
        for(int i:rows) result(i,j).col(b)=product.segment<3>(3*i);
    }
    potential.getEFS(atom,neighborhoods);hessian=std::move(result);
}
void assemble_cartesian_diagonal(AnyPotential& potential,Atoms& atom,
    AnyNeighborhoods& neighborhoods,std::vector<double>& diagonal) {
    if(!potential.supports_analytical_hessian()) GPERROR("Unsupported analytical Hessian diagonal");
    std::vector<double> result(3*atom.size());
    Eigen::VectorXd direction=Eigen::VectorXd::Zero(result.size()),product;
    for(int j=0;j<direction.size();++j) {
        direction[j]=1.;potential.applyHessianVectorProduct(atom,neighborhoods,direction,product);
        direction[j]=0.;result[j]=product[j];
    }
    potential.getEFS(atom,neighborhoods);diagonal=std::move(result);
}

static void assemble_latent_native_hessian_impl(PotMTP& basis,AnyPotential& potential,LES& engine,
    Atoms& atom,AnyNeighborhoods& neighborhoods,const double* species,const double* moments,
    int channels,bool long_range,const LocalCartesianHVP& short_range,
    HessianMatrix& hessian,Eigen::MatrixXd& cell,Eigen::MatrixXd& mixed,
    const LatentCurvatureCallbacks* callbacks,LES::Workspace* external_workspace) {
    LES::Workspace local_workspace;
    auto& les_workspace=external_workspace ? *external_workspace : local_workspace;
    if(callbacks && (!callbacks->finish_efs || (long_range && !callbacks->charge_hvp)))
        GPERROR("Incomplete caller-owned native Hessian callbacks");
    if(!potential.supports_analytical_cell_hessian()) GPERROR("Unsupported analytical native cell Hessian");
    if(neighborhoods.size()!=atom.size() || !atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible())
        GPERROR("Native cell Hessian requires matching neighborhoods and finite invertible cell");
    if(long_range) {
        LES::validate_periodic_geometry(atom);
        if(engine.cutoff_mode!=LES::CutoffMode::Smooth) GPERROR("LES native cell curvature requires smooth reciprocal cutoff");
    }
    const int n=atom.size();const Eigen::Matrix3d inverse=atom.cell_.inverse();
    auto action=[&](const Eigen::Matrix3d& dc,const Eigen::VectorXd& direction,
                    Eigen::VectorXd& u,Eigen::Matrix3d& c) {
        const Eigen::Map<const SiteCoordinateMatrix> velocity_u(direction.data(),n,3);
        u=Eigen::VectorXd::Zero(3*n);c.setZero();
        SiteCoordinateMatrix relative,gradient,product;
        auto local_velocity=[&](Neighborhood& neighborhood,int center) {
            relative.resize(neighborhood.nbhcount_,3);
            for(int j=0;j<neighborhood.nbhcount_;++j)
                relative.row(j)=neighborhood.pos_[j]*inverse*dc
                    +velocity_u.row(neighborhood.indices_[j])-velocity_u.row(center);
        };
        auto scatter=[&](const Neighborhood& neighborhood,int center,const SiteCoordinateMatrix& value) {
            for(int j=0;j<neighborhood.nbhcount_;++j) {
                u.segment<3>(3*neighborhood.indices_[j])+=value.row(j).transpose();
                u.segment<3>(3*center)-=value.row(j).transpose();
                const Eigen::RowVector3d fractional=neighborhood.pos_[j]*inverse;
                c+=fractional.transpose()*value.row(j);
            }
        };
        for(int i=0;i<n;++i) {
            local_velocity(neighborhoods[i],i);short_range(neighborhoods[i],relative,product);
            scatter(neighborhoods[i],i,product);
        }
        if(!long_range) return;
        SiteCoordinateMatrix velocity(n,3);
        for(int i=0;i<n;++i) velocity.row(i)=atom.pos_[i]*inverse*dc+velocity_u.row(i);
        Eigen::VectorXd q(n),qd(n);
        PotMTP::ompbuffers unused;HessianMatrix unused_hessian;
        auto head_value=[&](Neighborhood& neighborhood,int center,LatentFC2Head& head,int channel,double& charge) {
            local_velocity(neighborhood,center);
            if(callbacks) callbacks->charge_hvp(neighborhood,channel,relative,charge,gradient,product);
            else basis.CalcSiteHessImpl<false,LatentFC2Head,0>(neighborhood,unused,charge,gradient,
                unused_hessian,head,0,&relative,&product);
        };
        for(int channel=0;channel<channels;++channel) {

            LatentFC2Head head{species+channel*basis.species_count,
                moments+channel*basis.alpha_scalar_moments,basis.alpha_scalar_moments};
            for(int i=0;i<n;++i) {
                head_value(neighborhoods[i],i,head,channel,q[i]);qd[i]=(gradient.array()*relative.array()).sum();
            }
            const auto response=engine.evaluate_native_directional(atom,q,qd,velocity,dc,les_workspace,channel);
            // Native residual u is independent of C. Its contribution to the
            // cell chart derivative is essential even when the base u is zero.
            c+=inverse.transpose()*dc.transpose()*inverse.transpose()*response.virial.transpose()
               -inverse.transpose()*response.virial_directional.transpose();
            for(int i=0;i<n;++i) {
                u.segment<3>(3*i)-=response.force_directional.row(i).transpose();
                const Eigen::RowVector3d residual_fractional=velocity_u.row(i)*inverse;
                c+=residual_fractional.transpose()*response.forces.row(i);
                double charge;head_value(neighborhoods[i],i,head,channel,charge);
                const SiteCoordinateMatrix local=response.dEdq_directional[i]*gradient+response.dEdq[i]*product;
                scatter(neighborhoods[i],i,local);
            }
        }
    };
    HessianMatrix atomic;atomic.resize(n);
    Eigen::MatrixXd cc=Eigen::MatrixXd::Zero(9,9),xc=Eigen::MatrixXd::Zero(9,3*n);
    Eigen::VectorXd direction=Eigen::VectorXd::Zero(3*n),up;Eigen::Matrix3d cp,dc=Eigen::Matrix3d::Zero();
    for(int column=0;column<9;++column) {
        dc(column/3,column%3)=1.;action(dc,direction,up,cp);dc.setZero();
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) cc(3*a+b,column)=cp(a,b);
    }
    for(int j=0;j<n;++j) for(int b=0;b<3;++b) {
        direction[3*j+b]=1.;action(dc,direction,up,cp);direction[3*j+b]=0.;
        for(int i=0;i<n;++i) atomic(i,j).col(b)=up.segment<3>(3*i);
        for(int a=0;a<3;++a) for(int k=0;k<3;++k) xc(3*a+k,3*j+b)=cp(a,k);
    }
    if(callbacks) callbacks->finish_efs();else potential.getEFS(atom,neighborhoods);
    hessian=std::move(atomic);cell=std::move(cc);mixed=std::move(xc);
}
void assemble_latent_native_hessian(PotMTP& basis,AnyPotential& potential,LES& engine,
    Atoms& atom,AnyNeighborhoods& neighborhoods,const double* species,const double* moments,
    int channels,bool long_range,const LocalCartesianHVP& short_range,
    HessianMatrix& hessian,Eigen::MatrixXd& cell,Eigen::MatrixXd& mixed) {
    assemble_latent_native_hessian_impl(basis,potential,engine,atom,neighborhoods,species,moments,
        channels,long_range,short_range,hessian,cell,mixed,nullptr,nullptr);
}
void assemble_latent_native_hessian(PotMTP& basis,AnyPotential& potential,LES& engine,
    Atoms& atom,AnyNeighborhoods& neighborhoods,const double* species,const double* moments,
    int channels,bool long_range,const LocalCartesianHVP& short_range,
    HessianMatrix& hessian,Eigen::MatrixXd& cell,Eigen::MatrixXd& mixed,
    const LatentCurvatureCallbacks& callbacks,LES::Workspace& workspace) {
    assemble_latent_native_hessian_impl(basis,potential,engine,atom,neighborhoods,species,moments,
        channels,long_range,short_range,hessian,cell,mixed,&callbacks,&workspace);
}

} // namespace Potentials

namespace Potentials {
void LES::publish_workspace()
{
    publish_legacy_diagnostics(default_workspace_);
}
void LES::publish_legacy_diagnostics(const Workspace& source)
{
    kvecs=source.kvecs;
    kvec_factors=source.kvec_factors;
    k_kernel=source.k_kernel;
    k_kernel_derivative=source.k_kernel_derivative;
    if(&source!=&default_workspace_) {
        default_workspace_.grid_taper_=source.grid_taper_;
        default_workspace_.grid_taper_derivative_=source.grid_taper_derivative_;
        // The default cache may describe another cell/channel. Its cached grid
        // must be rebuilt before using the newly published taper arrays.
        default_workspace_.grid_valid_=false;
        default_workspace_.kernel_valid_=false;
    }
}
void LES::validate() const {validate(sog_channel);}
void LES::generate_k_vectors(const Atoms& atom,KVectorList& out,bool update_kernel)
{
    generate_k_vectors(atom,out,update_kernel,default_workspace_,sog_channel);
    // Preserve the historical external-output form: only out receives the grid.
    kvec_factors=default_workspace_.kvec_factors;
    if(update_kernel) {k_kernel=default_workspace_.k_kernel;
        k_kernel_derivative=default_workspace_.k_kernel_derivative;}
}
LES::Evaluation LES::evaluate(const Atoms& atom,const Eigen::VectorXd& q,
    Workspace& workspace,int channel) const
{return evaluate(atom,q,false,workspace,channel);}
LES::Evaluation LES::evaluate(const Atoms& atom,const Eigen::VectorXd& q,bool need_charge_hessian)
{
    auto out=evaluate(atom,q,need_charge_hessian,default_workspace_,sog_channel);
    publish_workspace();return out;
}
LES::ChannelEvaluation LES::evaluate_channels(const Atoms& atom,const Eigen::MatrixXd& q)
{
    validate();
    auto out=evaluate_channels(atom,q,default_workspace_);publish_workspace();return out;
}
LES::KernelComponents LES::kernel_components(const Atoms& atom,const Eigen::MatrixXd& q)
{
    validate();
    auto out=kernel_components(atom,q,default_workspace_);
    generate_k_vectors(atom,default_workspace_.kvecs,true,default_workspace_,sog_channel);
    publish_workspace();return out;
}
Eigen::VectorXd LES::apply_charge_hessian(const Atoms& atom,const Eigen::VectorXd& v)
{
    auto out=apply_charge_hessian(atom,v,default_workspace_,sog_channel);publish_workspace();return out;
}
Eigen::MatrixXd LES::apply_charge_hessian_channels(const Atoms& atom,const Eigen::MatrixXd& v)
{
    validate();
    auto out=apply_charge_hessian_channels(atom,v,default_workspace_);publish_workspace();return out;
}
LES::CartesianResponse LES::evaluate_cartesian_directional(const Atoms& atom,
    const Eigen::VectorXd& q,const Eigen::VectorXd& qd,const SiteCoordinateMatrix& v)
{
    auto out=evaluate_cartesian_directional(atom,q,qd,v,default_workspace_,sog_channel);
    publish_workspace();return out;
}
LES::NativeResponse LES::evaluate_native_directional(const Atoms& atom,
    const Eigen::VectorXd& q,const Eigen::VectorXd& qd,const SiteCoordinateMatrix& v,
    const Eigen::Matrix3d& dc)
{
    auto out=evaluate_native_directional(atom,q,qd,v,dc,default_workspace_,sog_channel);
    publish_workspace();return out;
}

void LES::add_latent_ewald_efs(Atoms& atom,AnyNeighborhoods& neighborhoods,
    Eigen::VectorXd& q,Eigen::VectorXd* out)
{
    add_latent_ewald_efs(atom,neighborhoods,q,out,default_workspace_,sog_channel);publish_workspace();
}
void LES::compute_les_dLdq_from_forces(const Atoms& atom,const Eigen::VectorXd& q,
    const Eigen::Matrix<double,-1,3>& df,const Eigen::Matrix3d& ds,Eigen::VectorXd& out)
{
    compute_les_dLdq_from_forces(atom,q,df,ds,out,default_workspace_,sog_channel);publish_workspace();
}
void LES::accumulate_kernel_vjp(const Atoms& atom,const Eigen::VectorXd& q,double de,
    const Eigen::Matrix<double,-1,3>& df,const Eigen::Matrix3d& dw,
    const Eigen::VectorXd& response,double* gradient)
{
    accumulate_kernel_vjp(atom,q,de,df,dw,response,gradient,default_workspace_,sog_channel);
    publish_workspace();
}

} // namespace Potentials

namespace Potentials {
void LES::compute_les_force_stress_grad(const Atoms& atom,const Eigen::VectorXd& q,
    const Eigen::MatrixXd& descriptors,const std::vector<int>& types,int species,
    int moments,const Eigen::Matrix<double,-1,3>& df,const Eigen::Matrix3d& ds,double* gradient) {
    compute_les_force_stress_grad(atom,q,descriptors,types,species,moments,df,ds,1.,
        gradient,default_workspace_,sog_channel);
    publish_workspace();
}
}
