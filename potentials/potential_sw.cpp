// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_sw.h"
#include "base/neighborhood.h"
#include "utils/compressed_stream.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <random>
#include <iomanip>

using namespace Potentials;

SW::SW(const std::string &filename,
       const std::map<std::string, int> &typemap) {
    auto ifs = open_ifile(filename);
    if (!ifs || !*ifs)
        throw std::runtime_error("SW: cannot open " + filename);
    load_stream(*ifs, typemap);
}

void SW::load_stream(std::istream &is, const std::map<std::string, int> &typemap) {
    std::vector<std::string> lines;
    auto types = typemap;
    std::string line;
    while (getline(is, line)) {
        if (auto comment = line.find('#'); comment != std::string::npos) line.erase(comment);
        std::istringstream row(line);
        std::string ei, ej, ek;
        if (!(row >> ei)) continue;
        if (!(row >> ej >> ek)) throw std::runtime_error("SW: incomplete parameter row");
        if (typemap.empty()) {
            for (const auto &name : {ei, ej, ek}) {
                if (name != "2body" && !types.contains(name)) {
                    const int id = types.size();
                    types.emplace(name, id);
                }
            }
        }
        lines.push_back(line);
    }
    if (is.bad() || lines.empty()) throw std::runtime_error("SW: missing or unreadable parameters");
    nelements = static_cast<int>(types.size());
    element_names_.assign(nelements, "");
    for (const auto &[name, id] : types) {
        if (id < 0 || id >= nelements || !element_names_[id].empty())
            throw std::runtime_error("SW: invalid species indices in type map");
        element_names_[id] = name;
    }
    species_count = nelements;

    pair_params.resize(nelements,
        std::vector<PairParam>(nelements));
    trip_params.resize(nelements,
        std::vector<std::vector<TripletParam>>(nelements,
            std::vector<TripletParam>(nelements)));

    double global_cutoff = 0;
    for (const auto &line : lines) {
        std::istringstream iss(line);
        std::string elem_i, elem_j, elem_k;
        iss >> elem_i >> elem_j >> elem_k;
        if (elem_i.empty()) continue;

        int it = types.at(elem_i);
        int jt = types.at(elem_j);

        if (elem_k == "2body") {
            auto &p = pair_params[it][jt];
            if (!(iss >> p.epsilon >> p.sigma >> p.A >> p.B >> p.p >> p.q
                >> p.a >> p.cutoff)) throw std::runtime_error("SW: incomplete pair parameters");
            pair_params[jt][it] = p;
            if (p.cutoff > global_cutoff) global_cutoff = p.cutoff;
        } else {
            // 3-body: element1 element2 element3 ...
            int kt = types.at(elem_k);
            auto &t = trip_params[it][jt][kt];
            if (!(iss >> t.epsilon >> t.sigma >> t.a
                >> t.lambda >> t.gamma >> t.cos_theta0 >> t.cutoff))
                throw std::runtime_error("SW: incomplete triplet parameters");
            // Keep the center fixed; exchange only the two neighbors.
            trip_params[it][kt][jt] = t;
            if (t.cutoff > global_cutoff) global_cutoff = t.cutoff;
        }
    }

    cutoff = global_cutoff;
    cutoffsq = cutoff * cutoff;
    cutoffs.resize(nelements, nelements);
    cutoffsqs.resize(nelements, nelements);
    r_mins.resize(nelements, nelements);
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            cutoffs(i, j) = global_cutoff;
            cutoffsqs(i, j) = global_cutoff * global_cutoff;
            r_mins(i, j) = 0.0;
        }
    }
    nl_inited = true;
}

void SW::Load(std::istream &is) {
    SW loaded;
    loaded.load_stream(is, {});
    *this = std::move(loaded);
}

void SW::Save(std::ostream &os) {
    os << std::setprecision(17) << "# SW potential\n";
    // Save pair parameters
    for (int i = 0; i < nelements; ++i) {
        for (int j = i; j < nelements; ++j) {
            const auto &p = pair_params[i][j];
            os << element_names_[i] << " " << element_names_[j] << " 2body "
               << p.epsilon << " " << p.sigma << " " << p.A << " " << p.B << " "
               << p.p << " " << p.q << " " << p.a << " " << p.cutoff << "\n";
        }
    }
    // Save triplet parameters
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            for (int k = j; k < nelements; ++k) {
                const auto &t = trip_params[i][j][k];
                if (t.epsilon == 0 && t.sigma == 0 && t.a == 0
                    && t.lambda == 0 && t.gamma == 0 && t.cos_theta0 == 0
                    && t.cutoff == 0) continue;
                os << element_names_[i] << " " << element_names_[j] << " " << element_names_[k]
                   << " " << t.epsilon << " " << t.sigma << " " << t.a << " "
                   << t.lambda << " " << t.gamma << " " << t.cos_theta0 << " " << t.cutoff << "\n";
            }
        }
    }
}

void SW::Save(const std::string &fnm) {
    if (save_gpm_if_requested(*this, fnm)) return;
    AnyPotential::Save(fnm);
}

void SW::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        int itype = nbh.centertype;
        int ncount = nbh.nbhcount_;

        // 2-body contributions
        for (int jj = 0; jj < ncount; ++jj) {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            double r = nbh.distances_[jj];
            const auto &p = pair_params[itype][jtype];
            if (r >= p.cutoff || p.epsilon == 0) continue;

            double sig_r = p.sigma / r;
            double term = p.B * std::pow(sig_r, p.p) - std::pow(sig_r, p.q);
            double exp_arg = p.sigma / (r - p.a * p.sigma);
            double exp_term = std::exp(exp_arg);
            double val = p.epsilon * p.A * term * exp_term;

            // Derivative
            double dterm_dr = -p.B * p.p * std::pow(sig_r, p.p - 1) * sig_r / r
                            + p.q * std::pow(sig_r, p.q - 1) * sig_r / r;
            double dexp_dr = -exp_term * exp_arg / (r - p.a * p.sigma);
            double deriv = p.epsilon * p.A * (dterm_dr * exp_term + term * dexp_dr);

            atom.energy_ += 0.5 * val;
            atom.site_energies_[ii] += 0.5 * val;

            double fpair = -0.5 * deriv / r;
            atom.forces_[ii](0) -= nbh.pos_[jj](0) * fpair;
            atom.forces_[ii](1) -= nbh.pos_[jj](1) * fpair;
            atom.forces_[ii](2) -= nbh.pos_[jj](2) * fpair;
            atom.forces_[jind](0) += nbh.pos_[jj](0) * fpair;
            atom.forces_[jind](1) += nbh.pos_[jj](1) * fpair;
            atom.forces_[jind](2) += nbh.pos_[jj](2) * fpair;

            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    atom.stress_(a, b) += nbh.pos_[jj](a) * fpair * nbh.pos_[jj](b);
        }

        // 3-body contributions
        for (int jj = 0; jj < ncount; ++jj) {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            double rij = nbh.distances_[jj];

            for (int kk = jj + 1; kk < ncount; ++kk) {
                int ktype = nbh.chemmap_[kk];
                int kind = nbh.indices_[kk];
                double rik = nbh.distances_[kk];

                const auto &t = trip_params[itype][jtype][ktype];
                if (t.epsilon == 0 || rij >= t.cutoff || rik >= t.cutoff) continue;

                // cos(theta) and theta
                double cos_theta = nbh.pos_[jj].dot(nbh.pos_[kk]) / (rij * rik);
                if (cos_theta > 1.0) cos_theta = 1.0;
                if (cos_theta < -1.0) cos_theta = -1.0;
                double dcos = cos_theta - t.cos_theta0;

                // Common exponential factor
                double exp_rj = std::exp(t.gamma * t.sigma / (rij - t.a * t.sigma));
                double exp_rk = std::exp(t.gamma * t.sigma / (rik - t.a * t.sigma));
                double exp_prod = exp_rj * exp_rk;

                double val = t.epsilon * t.lambda * exp_prod * dcos * dcos;

                double third = 1.0 / 3.0;
                atom.energy_ += third * val;
                atom.site_energies_[ii] += third * val;

                // Exponential derivatives
                double dexp_rj = -exp_rj * t.gamma * t.sigma
                                 / ((rij - t.a * t.sigma) * (rij - t.a * t.sigma));
                double dexp_rk = -exp_rk * t.gamma * t.sigma
                                 / ((rik - t.a * t.sigma) * (rik - t.a * t.sigma));

                double dV_drij = t.epsilon * t.lambda * dcos * dcos
                                 * dexp_rj * exp_rk;
                double dV_drik = t.epsilon * t.lambda * dcos * dcos
                                 * exp_rj * dexp_rk;
                double dV_dcosth = t.epsilon * t.lambda * exp_prod * 2.0 * dcos;

                // Radial forces
                double fp_rij = dV_drij;
                double fp_rik = dV_drik;

                // Angular forces: d(cos_theta)/d(pos) derivatives
                double inv_rij = 1.0 / rij;
                double inv_rik = 1.0 / rik;
                Eigen::Vector3d r_ij_hat = nbh.pos_[jj] * inv_rij;
                Eigen::Vector3d r_ik_hat = nbh.pos_[kk] * inv_rik;

                Eigen::Vector3d dcth_drj = (r_ik_hat - cos_theta * r_ij_hat) * inv_rij;
                Eigen::Vector3d dcth_drk = (r_ij_hat - cos_theta * r_ik_hat) * inv_rik;

                Eigen::Vector3d fj = -r_ij_hat * fp_rij - dcth_drj * dV_dcosth;
                Eigen::Vector3d fk = -r_ik_hat * fp_rik - dcth_drk * dV_dcosth;
                Eigen::Vector3d fi = -(fj + fk);

                fj *= third; fk *= third; fi *= third;

                atom.forces_[ii] += fi;
                atom.forces_[jind] += fj;
                atom.forces_[kind] += fk;

                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        atom.stress_(a, b) += fj(a) * nbh.pos_[jj](b)
                                            + fk(a) * nbh.pos_[kk](b);
            }
        }
    }
}

// Differentiate the existing pair/triplet position gradients directly along
// the probe. Only the O(N) result and fixed-size local vectors are stored.
void SW::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
                                   const Eigen::VectorXd &direction,
                                   Eigen::VectorXd &product) {
    if (direction.size() != 3*atom.size() || !direction.allFinite())
        GPERROR("Hessian-vector direction must be finite with size 3*N");
    if (nbhs.size() != atom.size())
        GPERROR("Hessian-vector neighborhoods have the wrong size");
    // Separate accumulation permits direction and product to alias.
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &n = nbhs[i];
        for (int j = 0; j < n.nbhcount_; ++j) {
            const int source = n.indices_[j];
            const double r = n.distances_[j];
            const auto &p = pair_params[n.centertype][n.chemmap_[j]];
            if (r >= p.cutoff || p.epsilon == 0) continue;
            const Eigen::Vector3d u = n.pos_[j].transpose()/r;
            const Eigen::Vector3d w = direction.segment<3>(3*source)-direction.segment<3>(3*i);
            const double alpha = u.dot(w);
            const double repulsive = p.B*std::pow(p.sigma/r, p.p);
            const double attractive = std::pow(p.sigma/r, p.q);
            const double term = repulsive-attractive;
            const double first_term = (-p.p*repulsive+p.q*attractive)/r;
            const double second_term = (p.p*(p.p+1)*repulsive-p.q*(p.q+1)*attractive)/(r*r);
            const double denominator = r-p.a*p.sigma;
            const double log_first = -p.sigma/(denominator*denominator);
            const double log_second = 2*p.sigma/(denominator*denominator*denominator);
            const double factor = .5*p.epsilon*p.A*std::exp(p.sigma/denominator);
            const double first = factor*(first_term+term*log_first);
            const double second = factor*(second_term+2*first_term*log_first
                +term*(log_first*log_first+log_second));
            const Eigen::Vector3d contribution = second*alpha*u+(first/r)*(w-alpha*u);
            result.segment<3>(3*source) += contribution;
            result.segment<3>(3*i) -= contribution;
        }
        for (int j = 0; j < n.nbhcount_; ++j) {
            const double r = n.distances_[j];
            for (int k = j+1; k < n.nbhcount_; ++k) {
                const double s = n.distances_[k];
                const auto &t = trip_params[n.centertype][n.chemmap_[j]][n.chemmap_[k]];
                if (t.epsilon == 0 || r >= t.cutoff || s >= t.cutoff) continue;
                const Eigen::Vector3d u = n.pos_[j].transpose()/r;
                const Eigen::Vector3d v = n.pos_[k].transpose()/s;
                const Eigen::Vector3d wj = direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i);
                const Eigen::Vector3d wk = direction.segment<3>(3*n.indices_[k])-direction.segment<3>(3*i);
                const double alpha = u.dot(wj), beta = v.dot(wk);
                const Eigen::Vector3d du = (wj-alpha*u)/r, dv = (wk-beta*v)/s;
                // Preserve the same cosine clamp as the EFS kernel. There is
                // no acos/small-sine branch, including collinear geometries.
                double cosine = n.pos_[j].dot(n.pos_[k])/(r*s);
                if (cosine > 1.) cosine = 1.;
                if (cosine < -1.) cosine = -1.;
                const double delta_cosine = du.dot(v)+u.dot(dv);
                const double angular = cosine-t.cos_theta0;
                const Eigen::Vector3d gj = (v-cosine*u)/r, gk = (u-cosine*v)/s;
                const Eigen::Vector3d dgj = (dv-delta_cosine*u-cosine*du-alpha*gj)/r;
                const Eigen::Vector3d dgk = (du-delta_cosine*v-cosine*dv-beta*gk)/s;
                const double dr = r-t.a*t.sigma, ds = s-t.a*t.sigma;
                const double scale = t.gamma*t.sigma;
                const double lr = -scale/(dr*dr), ls = -scale/(ds*ds);
                const double dlr = 2*scale*alpha/(dr*dr*dr);
                const double dls = 2*scale*beta/(ds*ds*ds);
                const double factor = (t.epsilon*t.lambda/3.)*std::exp(scale/dr)*std::exp(scale/ds);
                const double delta_log = lr*alpha+ls*beta;
                const double first_r = factor*lr*angular*angular;
                const double first_s = factor*ls*angular*angular;
                const double first_cosine = 2*factor*angular;
                const double delta_first_r = factor*((delta_log*lr+dlr)*angular*angular
                    +2*lr*angular*delta_cosine);
                const double delta_first_s = factor*((delta_log*ls+dls)*angular*angular
                    +2*ls*angular*delta_cosine);
                const double delta_first_cosine = 2*factor*(delta_log*angular+delta_cosine);
                const Eigen::Vector3d contribution_j = delta_first_r*u+first_r*du
                    +delta_first_cosine*gj+first_cosine*dgj;
                const Eigen::Vector3d contribution_k = delta_first_s*v+first_s*dv
                    +delta_first_cosine*gk+first_cosine*dgk;
                result.segment<3>(3*n.indices_[j]) += contribution_j;
                result.segment<3>(3*n.indices_[k]) += contribution_k;
                result.segment<3>(3*i) -= contribution_j+contribution_k;
            }
        }
    }
    product = std::move(result);
}

// ═════════════════════════════════════════════════════════════════════════
//  Training / fitting interface
// ═════════════════════════════════════════════════════════════════════════

void SW::build_param_map() {
    param_map_.clear();

    // Pair parameters: unique pairs (i <= j)
    for (int i = 0; i < nelements; ++i) {
        for (int j = i; j < nelements; ++j) {
            // 8 pair fields
            param_map_.push_back({Field::PAIR_EPSILON, i, j, 0, false});
            param_map_.push_back({Field::PAIR_SIGMA, i, j, 0, false});
            param_map_.push_back({Field::PAIR_A, i, j, 0, false});
            param_map_.push_back({Field::PAIR_B, i, j, 0, false});
            param_map_.push_back({Field::PAIR_P, i, j, 0, false});
            param_map_.push_back({Field::PAIR_Q, i, j, 0, false});
            param_map_.push_back({Field::PAIR_A_PARAM, i, j, 0, false});
            param_map_.push_back({Field::PAIR_CUTOFF, i, j, 0, false});
        }
    }

    // Triplet parameters: each center i, with unordered neighbors j <= k.
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            for (int k = j; k < nelements; ++k) {
                // 7 triplet fields
                param_map_.push_back({Field::TRIP_EPSILON, i, j, k, true});
                param_map_.push_back({Field::TRIP_SIGMA, i, j, k, true});
                param_map_.push_back({Field::TRIP_A, i, j, k, true});
                param_map_.push_back({Field::TRIP_LAMBDA, i, j, k, true});
                param_map_.push_back({Field::TRIP_GAMMA, i, j, k, true});
                param_map_.push_back({Field::TRIP_COS_THETA0, i, j, k, true});
                param_map_.push_back({Field::TRIP_CUTOFF, i, j, k, true});
            }
        }
    }

    coeffs_.resize(param_map_.size());
    // Initialize coeffs_ from current parameter values
    for (size_t idx = 0; idx < param_map_.size(); ++idx)
        coeffs_[idx] = get_single_param(idx);
}

void SW::unpack_coeffs() {
    for (size_t idx = 0; idx < param_map_.size(); ++idx)
        set_single_param(idx, coeffs_[idx]);
}

double SW::get_single_param(int idx) const {
    const auto &e = param_map_[idx];
    if (e.is_triplet) {
        const auto &t = trip_params[e.i][e.j][e.k];
        switch (e.field) {
            case Field::TRIP_EPSILON:    return t.epsilon;
            case Field::TRIP_SIGMA:      return t.sigma;
            case Field::TRIP_A:          return t.a;
            case Field::TRIP_LAMBDA:     return t.lambda;
            case Field::TRIP_GAMMA:      return t.gamma;
            case Field::TRIP_COS_THETA0: return t.cos_theta0;
            case Field::TRIP_CUTOFF:     return t.cutoff;
            default: return 0;
        }
    } else {
        const auto &p = pair_params[e.i][e.j];
        switch (e.field) {
            case Field::PAIR_EPSILON:  return p.epsilon;
            case Field::PAIR_SIGMA:    return p.sigma;
            case Field::PAIR_A:        return p.A;
            case Field::PAIR_B:        return p.B;
            case Field::PAIR_P:        return p.p;
            case Field::PAIR_Q:        return p.q;
            case Field::PAIR_A_PARAM:  return p.a;
            case Field::PAIR_CUTOFF:   return p.cutoff;
            default: return 0;
        }
    }
}

void SW::set_single_param(int idx, double val) {
    const auto &e = param_map_[idx];
    if (e.is_triplet) {
        auto set_field = [&](TripletParam &t) {
            switch (e.field) {
                case Field::TRIP_EPSILON:    t.epsilon = val; break;
                case Field::TRIP_SIGMA:      t.sigma = val; break;
                case Field::TRIP_A:          t.a = val; break;
                case Field::TRIP_LAMBDA:     t.lambda = val; break;
                case Field::TRIP_GAMMA:      t.gamma = val; break;
                case Field::TRIP_COS_THETA0: t.cos_theta0 = val; break;
                case Field::TRIP_CUTOFF:     t.cutoff = val; break;
                default: break;
            }
        };
        set_field(trip_params[e.i][e.j][e.k]);
        // Keep the center fixed when updating the symmetric entry.
        if (e.j != e.k)
            set_field(trip_params[e.i][e.k][e.j]);
    } else {
        auto set_field = [&](PairParam &p) {
            switch (e.field) {
                case Field::PAIR_EPSILON:  p.epsilon = val; break;
                case Field::PAIR_SIGMA:    p.sigma = val; break;
                case Field::PAIR_A:        p.A = val; break;
                case Field::PAIR_B:        p.B = val; break;
                case Field::PAIR_P:        p.p = val; break;
                case Field::PAIR_Q:        p.q = val; break;
                case Field::PAIR_A_PARAM:  p.a = val; break;
                case Field::PAIR_CUTOFF:   p.cutoff = val; break;
                default: break;
            }
        };
        set_field(pair_params[e.i][e.j]);
        if (e.i != e.j)
            set_field(pair_params[e.j][e.i]);
    }
    coeffs_[idx] = val;
}

int SW::get_NL_coeffs_count() const {
    return static_cast<int>(coeffs_.size());
}

void SW::nl_get_coeffs(double *x) const {
    for (size_t i = 0; i < coeffs_.size(); ++i)
        x[i] = coeffs_[i];
}

void SW::nl_set_coeffs(const double *x) {
    for (size_t i = 0; i < coeffs_.size(); ++i)
        coeffs_[i] = x[i];
    unpack_coeffs();

    // Update global cutoff from per-pair/triplet cutoffs
    double global_cutoff = 0;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            if (pair_params[i][j].cutoff > global_cutoff)
                global_cutoff = pair_params[i][j].cutoff;
            for (int k = 0; k < nelements; ++k) {
                if (trip_params[i][j][k].cutoff > global_cutoff)
                    global_cutoff = trip_params[i][j][k].cutoff;
            }
        }
    }
    cutoff = global_cutoff;
    cutoffsq = cutoff * cutoff;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            cutoffs(i, j) = global_cutoff;
            cutoffsqs(i, j) = global_cutoff * global_cutoff;
        }
    }
}

void SW::train_init() {
    if (param_map_.empty())
        build_param_map();
    reg_L1_vector.resize(coeffs_.size(), 0);
    reg_L2_vector.resize(coeffs_.size(), 0);
    nl_inited = true;
}

void SW::random_init() {
    if (param_map_.empty())
        build_param_map();

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (size_t idx = 0; idx < param_map_.size(); ++idx) {
        const auto &e = param_map_[idx];
        // Use physically reasonable random ranges per field type
        double val = 0;
        if (e.is_triplet) {
            switch (e.field) {
                case Field::TRIP_EPSILON:    val = std::abs(dis(gen)) * 5.0; break;
                case Field::TRIP_SIGMA:      val = std::abs(dis(gen)) * 2.0 + 0.5; break;
                case Field::TRIP_A:          val = std::abs(dis(gen)) * 2.0 + 0.5; break;
                case Field::TRIP_LAMBDA:     val = std::abs(dis(gen)) * 50.0; break;
                case Field::TRIP_GAMMA:      val = std::abs(dis(gen)) * 2.0 + 0.5; break;
                case Field::TRIP_COS_THETA0: val = dis(gen); break;  // [-1, 1]
                case Field::TRIP_CUTOFF:     val = std::abs(dis(gen)) * 2.0 + 2.0; break;
                default: break;
            }
        } else {
            switch (e.field) {
                case Field::PAIR_EPSILON:  val = std::abs(dis(gen)) * 5.0; break;
                case Field::PAIR_SIGMA:    val = std::abs(dis(gen)) * 2.0 + 0.5; break;
                case Field::PAIR_A:        val = std::abs(dis(gen)) * 50.0; break;
                case Field::PAIR_B:        val = std::abs(dis(gen)) * 2.0 + 0.5; break;
                case Field::PAIR_P:        val = std::abs(dis(gen)) * 10.0 + 1.0; break;
                case Field::PAIR_Q:        val = std::abs(dis(gen)) * 5.0 + 0.5; break;
                case Field::PAIR_A_PARAM:  val = std::abs(dis(gen)) * 2.0 + 0.5; break;
                case Field::PAIR_CUTOFF:   val = std::abs(dis(gen)) * 2.0 + 2.0; break;
                default: break;
            }
        }
        coeffs_[idx] = val;
    }
    unpack_coeffs();

    // Update global cutoff
    double global_cutoff = 0;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            if (pair_params[i][j].cutoff > global_cutoff)
                global_cutoff = pair_params[i][j].cutoff;
            for (int k = 0; k < nelements; ++k) {
                if (trip_params[i][j][k].cutoff > global_cutoff)
                    global_cutoff = trip_params[i][j][k].cutoff;
            }
        }
    }
    cutoff = global_cutoff;
    cutoffsq = cutoff * cutoff;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            cutoffs(i, j) = global_cutoff;
            cutoffsqs(i, j) = global_cutoff * global_cutoff;
        }
    }
    nl_inited = true;
}

// ═════════════════════════════════════════════════════════════════════════
//  Numerical gradient via central finite differences
// ═════════════════════════════════════════════════════════════════════════

void SW::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs,
    double dLdE,
    Eigen::Matrix<double, -1, 3> &dLdF,
    Eigen::Matrix3d &dLdS,
    Eigen::Matrix<double, -1, 3, Eigen::RowMajor> &/*wgt_buf*/,
    double *out_gard_accumulator)
{
    numerical_gradient_efs(atom, nbhs, dLdE, dLdF, dLdS,
        out_gard_accumulator, static_cast<int>(coeffs_.size()),
        [this](int idx, double delta) {
            set_single_param(idx, coeffs_[idx] + delta);
        });
}

// Energy-only overload (no force/stress terms)
void SW::AccumulateEFSCombinationGard(
    Atoms &atom, AnyNeighborhoods &nbhs,
    double dLdE,
    double *out_gard_accumulator)
{
    numerical_gradient_eonly(atom, nbhs, dLdE,
        out_gard_accumulator, static_cast<int>(coeffs_.size()),
        [this](int idx, double delta) {
            set_single_param(idx, coeffs_[idx] + delta);
        });
}

Eigen::Matrix3d SW::pair_curvature(const Neighborhood &n, int j) const {
    const double r = n.distances_[j];
    const auto &p = pair_params[n.centertype][n.chemmap_[j]];
    const Eigen::Vector3d u = n.pos_[j].transpose()/r;
    const double repulsive = p.B*std::pow(p.sigma/r, p.p);
    const double attractive = std::pow(p.sigma/r, p.q);
    const double term = repulsive-attractive;
    const double first_term = (-p.p*repulsive+p.q*attractive)/r;
    const double second_term = (p.p*(p.p+1)*repulsive-p.q*(p.q+1)*attractive)/(r*r);
    const double denominator = r-p.a*p.sigma;
    const double log_first = -p.sigma/(denominator*denominator);
    const double log_second = 2*p.sigma/(denominator*denominator*denominator);
    const double factor = .5*p.epsilon*p.A*std::exp(p.sigma/denominator);
    const double first = factor*(first_term+term*log_first);
    const double second = factor*(second_term+2*first_term*log_first
        +term*(log_first*log_first+log_second));
    return second*u*u.transpose()+(first/r)*(Eigen::Matrix3d::Identity()-u*u.transpose());
}

Eigen::Matrix<double,6,6> SW::triplet_curvature(const Neighborhood &n, int j, int k) const {
    const double r = n.distances_[j], s = n.distances_[k];
    const auto &t = trip_params[n.centertype][n.chemmap_[j]][n.chemmap_[k]];
    const Eigen::Vector3d u = n.pos_[j].transpose()/r, v = n.pos_[k].transpose()/s;
    double cosine = n.pos_[j].dot(n.pos_[k])/(r*s);
    if (cosine > 1.) cosine = 1.;
    if (cosine < -1.) cosine = -1.;
    const double angular = cosine-t.cos_theta0;
    const Eigen::Vector3d gj = (v-cosine*u)/r, gk = (u-cosine*v)/s;
    const double dr = r-t.a*t.sigma, ds = s-t.a*t.sigma;
    const double scale = t.gamma*t.sigma;
    const double lr = -scale/(dr*dr), ls = -scale/(ds*ds);
    const double factor = (t.epsilon*t.lambda/3.)*std::exp(scale/dr)*std::exp(scale/ds);
    const double first_r = factor*lr*angular*angular;
    const double first_s = factor*ls*angular*angular;
    const double first_cosine = 2*factor*angular;
    Eigen::Matrix<double,6,6> local;
    // The same analytic gradient differential as the native HVP, evaluated
    // on six local bond directions. No perturbed coordinates or global probes.
    for (int column = 0; column < 6; ++column) {
        Eigen::Vector3d wj = Eigen::Vector3d::Zero(), wk = Eigen::Vector3d::Zero();
        if (column < 3) wj[column] = 1.; else wk[column-3] = 1.;
        const double alpha = u.dot(wj), beta = v.dot(wk);
        const Eigen::Vector3d du = (wj-alpha*u)/r, dv = (wk-beta*v)/s;
        const double delta_cosine = du.dot(v)+u.dot(dv);
        const Eigen::Vector3d dgj = (dv-delta_cosine*u-cosine*du-alpha*gj)/r;
        const Eigen::Vector3d dgk = (du-delta_cosine*v-cosine*dv-beta*gk)/s;
        const double dlr = 2*scale*alpha/(dr*dr*dr);
        const double dls = 2*scale*beta/(ds*ds*ds);
        const double delta_log = lr*alpha+ls*beta;
        const double delta_first_r = factor*((delta_log*lr+dlr)*angular*angular
            +2*lr*angular*delta_cosine);
        const double delta_first_s = factor*((delta_log*ls+dls)*angular*angular
            +2*ls*angular*delta_cosine);
        const double delta_first_cosine = 2*factor*(delta_log*angular+delta_cosine);
        local.block<3,1>(0,column) = delta_first_r*u+first_r*du
            +delta_first_cosine*gj+first_cosine*dgj;
        local.block<3,1>(3,column) = delta_first_s*v+first_s*dv
            +delta_first_cosine*gk+first_cosine*dgk;
    }
    return local;
}

void SW::assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
    Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess) {
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    if (cellhess && (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible()))
        GPERROR("Analytical SW cell Hessian requires a finite invertible cell");
    getEFS(atom, nbhs);
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    if (cellhess) {
        inverse = atom.cell_.inverse();
        *cellhess = Eigen::MatrixXd::Zero(9,9);
        *lrhess = Eigen::MatrixXd::Zero(9,3*atom.size());
    }
    for (int i = 0; i < nbhs.size(); ++i) {
        const auto &n = nbhs[i];
        for (int j = 0; j < n.nbhcount_; ++j) {
            const auto &p = pair_params[n.centertype][n.chemmap_[j]];
            if (n.distances_[j] >= p.cutoff || p.epsilon == 0) continue;
            const int source = n.indices_[j];
            const auto local = pair_curvature(n,j);
            hessian(i,i) += local;
            hessian(source,source) += local;
            hessian(i,source) -= local;
            hessian(source,i) -= local;
            if (cellhess) {
                Eigen::Matrix<double,3,9> affine = Eigen::Matrix<double,3,9>::Zero();
                const Eigen::RowVector3d fractional = n.pos_[j]*inverse;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) affine(b,3*a+b) = fractional(a);
                *cellhess += affine.transpose()*local*affine;
                const Eigen::Matrix<double,9,3> mixed = affine.transpose()*local;
                lrhess->middleCols(3*source,3) += mixed;
                lrhess->middleCols(3*i,3) -= mixed;
            }
        }
        for (int j = 0; j < n.nbhcount_; ++j)
            for (int k = j+1; k < n.nbhcount_; ++k) {
                const auto &t = trip_params[n.centertype][n.chemmap_[j]][n.chemmap_[k]];
                if (t.epsilon == 0 || n.distances_[j] >= t.cutoff || n.distances_[k] >= t.cutoff) continue;
                const auto local = triplet_curvature(n,j,k);
                const int sources[2] = {n.indices_[j],n.indices_[k]};
                // Relative bonds are x_source-x_i; the incidence scatter also
                // handles repeated sources and periodic self images.
                for (int a = 0; a < 2; ++a)
                    for (int b = 0; b < 2; ++b) {
                        const Eigen::Matrix3d block = local.block<3,3>(3*a,3*b);
                        hessian(sources[a],sources[b]) += block;
                        hessian(i,sources[b]) -= block;
                        hessian(sources[a],i) -= block;
                        hessian(i,i) += block;
                    }
                if (cellhess) {
                    Eigen::Matrix<double,6,9> affine = Eigen::Matrix<double,6,9>::Zero();
                    const int bonds[2] = {j,k};
                    for (int bond = 0; bond < 2; ++bond) {
                        const Eigen::RowVector3d fractional = n.pos_[bonds[bond]]*inverse;
                        for (int a = 0; a < 3; ++a)
                            for (int b = 0; b < 3; ++b) affine(3*bond+b,3*a+b) = fractional(a);
                    }
                    *cellhess += affine.transpose()*local*affine;
                    const Eigen::Matrix<double,9,6> mixed = affine.transpose()*local;
                    for (int bond = 0; bond < 2; ++bond) {
                        lrhess->middleCols(3*sources[bond],3) += mixed.middleCols<3>(3*bond);
                        lrhess->middleCols(3*i,3) -= mixed.middleCols<3>(3*bond);
                    }
                }
            }
    }
}

void SW::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    assemble_hessian(atom,nbhs,hessian,nullptr,nullptr);
}
void SW::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessian(atom,nbhs,hessian);
}
void SW::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    getEFSHessian(atom,nbhs,hessian);
}
void SW::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double) {
    getEFSHessian(atom,hessian);
}
void SW::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    assemble_hessian(atom,nbhs,hessian,&cellhess,&lrhess);
}
void SW::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessianCell(atom,nbhs,hessian,cellhess,lrhess);
}
void SW::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCell(atom,nbhs,hessian,cellhess,lrhess);
}
void SW::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    getEFSHessianCell(atom,hessian,cellhess,lrhess);
}
void SW::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    getEFSHessianCell(atom,hessian,cellhess,lrhess);
}
