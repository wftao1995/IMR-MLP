// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_tersoff.h"
#include "base/neighborhood.h"
#include "utils/compressed_stream.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <map>
#include <random>
#include <iomanip>
#include <algorithm>
#include <array>

using namespace Potentials;

// ── Common math helpers ──────────────────────────────────────────────

static inline double fc(double r, double R, double D) {
    if (r < R - D) return 1.0;
    if (r > R + D) return 0.0;
    return 0.5 * (1.0 - std::sin(M_PI_2 * (r - R) / D));
}

static inline double fc_deriv(double r, double R, double D) {
    if (r <= R - D || r >= R + D) return 0.0;
    return -0.5 * M_PI_2 / D * std::cos(M_PI_2 * (r - R) / D);
}

static inline double g_theta(double costh, double c, double d, double h) {
    double denom = d * d + (h - costh) * (h - costh);
    return 1.0 + c * c / (d * d) - c * c / denom;
}

static inline double g_theta_deriv(double costh, double c, double d, double h) {
    double denom = d * d + (h - costh) * (h - costh);
    return -2.0 * c * c * (h - costh) / (denom * denom);
}

// ── Parameter parsing helpers ────────────────────────────────────────

/// Parse key=value pairs from a string into a map.
static std::map<std::string, double> parse_key_values(const std::string &s) {
    std::map<std::string, double> kv;
    std::istringstream iss(s);
    std::string token;
    while (iss >> token) {
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        kv[token.substr(0, eq)] = std::stod(token.substr(eq + 1));
    }
    return kv;
}

/// Apply key=value map to a Param struct. Returns true if any keys matched.
static bool apply_kv_to_param(const std::map<std::string, double> &kv,
                              Tersoff::Param &p) {
    bool matched = false;
    for (const auto &[k, v] : kv) {
        if (k == "A")           { p.A = v; matched = true; }
        else if (k == "B")      { p.B = v; matched = true; }
        else if (k == "lambda1"){ p.lambda1 = v; matched = true; }
        else if (k == "lambda2"){ p.lambda2 = v; matched = true; }
        else if (k == "lambda3"){ p.lambda3 = v; matched = true; }
        else if (k == "m")      { p.m = v; matched = true; }
        else if (k == "n")      { p.n = v; matched = true; }
        else if (k == "beta")   { p.beta = v; matched = true; }
        else if (k == "R")      { p.R = v; matched = true; }
        else if (k == "D")      { p.D = v; matched = true; }
        else if (k == "c")      { p.c = v; matched = true; }
        else if (k == "d")      { p.d = v; matched = true; }
        else if (k == "h")      { p.h = v; matched = true; }
        else if (k == "D0")     { p.D0 = v; matched = true; }
        else if (k == "S")      { p.S = v; matched = true; }
        else if (k == "r0")     { p.r0 = v; matched = true; }
        else if (k == "gamma")  { p.gamma = v; matched = true; }
    }
    return matched;
}

/// Parse positional Tersoff parameters from a stream (old format).
static void parse_positional(std::istringstream &iss, Tersoff::Param &p) {
    iss >> p.A >> p.B >> p.lambda1 >> p.lambda2
        >> p.R >> p.D >> p.beta >> p.n
        >> p.c >> p.d >> p.h >> p.lambda3;
    if (!iss.eof()) iss >> p.m;
}

/// Check whether a token string looks like a key=value pair.
static inline bool is_key_value(const std::string &token) {
    return token.find('=') != std::string::npos;
}

// ── Constructor ──────────────────────────────────────────────────────

Tersoff::Tersoff(const std::string &filename,
                 const std::map<std::string, int> &typemap) {
    auto ifs = open_ifile(filename);
    if (!ifs || !*ifs)
        throw std::runtime_error("Tersoff: cannot open " + filename);
    load_stream(*ifs, typemap);
}

void Tersoff::load_stream(std::istream &is, const std::map<std::string, int> &typemap) {
    // Parse the string-valued mode separately from numeric pair parameters.
    std::vector<std::string> lines;
    std::string line;
    while (getline(is, line)) {
        if (auto comment = line.find('#'); comment != std::string::npos) line.erase(comment);
        std::istringstream row(line);
        std::string first;
        if (!(row >> first)) continue;
        if (first == "mode" || first.starts_with("mode=")) {
            const auto eq = line.find('=');
            if (eq == std::string::npos) throw std::runtime_error("Tersoff: expected mode=tersoff or mode=brenner");
            std::istringstream value(line.substr(eq + 1));
            std::string mode_name;
            value >> mode_name;
            if (mode_name == "brenner") mode_ = Mode::Brenner;
            else if (mode_name == "tersoff") mode_ = Mode::Tersoff;
            else throw std::runtime_error("Tersoff: unknown mode " + mode_name);
            continue;
        }
        lines.push_back(line);
    }
    if (is.bad() || lines.empty()) throw std::runtime_error("Tersoff: missing or unreadable parameters");

    // Detect old Brenner format: first content line is a single integer
    for (const auto &line : lines) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string first_token;
        iss >> first_token;
        if (first_token.find('=') != std::string::npos) break; // new format
        // Check if it's a lone integer (old Brenner species count)
        bool is_int = true;
        for (char c : first_token) {
            if (!std::isdigit(c) && c != '-') { is_int = false; break; }
        }
        if (is_int) {
            // Check if second token exists (element name in Tersoff format)
            // or is absent (old Brenner format with just species count)
            std::string second;
            iss >> second;
            if (second.empty()) {
                throw std::runtime_error(
                    "Tersoff: old tersoffb 3-line format is no longer supported. "
                    "Use key=value format with 'mode = brenner' and element names. "
                    "Example:\n  mode = brenner\n  Fe Fe D0=0.0013 S=1.78 r0=2.58 "
                    "beta=1.53 R=2.75 D=0.05 gamma=1.0 c=0.0 d=1.0 h=0.0"
                );
            }
        }
        break; // Only check first non-comment line
    }

    auto types = typemap;
    if (types.empty()) {
        for (const auto &line : lines) {
            std::istringstream row(line);
            std::string ei, ej;
            if (!(row >> ei >> ej)) throw std::runtime_error("Tersoff: incomplete pair row");
            for (const auto &name : {ei, ej}) {
                if (!types.contains(name)) {
                    const int id = types.size();
                    types.emplace(name, id);
                }
            }
        }
    }
    nelements = static_cast<int>(types.size());
    element_names_.assign(nelements, "");
    for (const auto &[name, id] : types) {
        if (id < 0 || id >= nelements || !element_names_[id].empty())
            throw std::runtime_error("Tersoff: invalid species indices in type map");
        element_names_[id] = name;
    }
    species_count = nelements;
    params.resize(nelements, std::vector<Param>(nelements));

    // Process pair definition lines.
    for (const auto &line : lines) {
        if (line.empty() || line[0] == '#') continue;
        // Skip global-setting lines (no element names)
        if (line.find('=') != std::string::npos) {
            // Check if this is a pure key=value line (no element name prefix)
            std::istringstream iss(line);
            std::string first;
            iss >> first;
            if (first.find('=') != std::string::npos) {
                continue; // global setting, skip
            }
        }

        std::istringstream iss(line);
        std::string ei, ej;
        iss >> ei >> ej;
        if (ei.empty() || ej.empty()) continue;

        int it = types.at(ei);
        int jt = types.at(ej);
        auto &p = params[it][jt];

        // Peek next token to detect format
        std::string next;
        iss >> next;
        if (next.empty()) continue;

        if (is_key_value(next)) {
            // New key=value format
            // Put back the peeked token
            std::string rest = next;
            std::string more;
            while (iss >> more) rest += " " + more;
            auto kv = parse_key_values(rest);
            apply_kv_to_param(kv, p);
        } else {
            // Old positional format: put back and parse positionally
            // Reconstruct: ei ej next [remaining tokens...]
            std::string remaining = next;
            std::string more;
            while (iss >> more) remaining += " " + more;
            std::istringstream pss(line);
            std::string dummy_ei, dummy_ej;
            pss >> dummy_ei >> dummy_ej;
            parse_positional(pss, p);
        }

        params[jt][it] = p;

        double rcut = p.R + p.D;
        if (rcut > cut_global) cut_global = rcut;
    }

    cutoff = cut_global;
    cutoffsq = cutoff * cutoff;
    cutoffs.resize(nelements, nelements);
    cutoffsqs.resize(nelements, nelements);
    r_mins.resize(nelements, nelements);
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            cutoffs(i, j) = cut_global;
            cutoffsqs(i, j) = cut_global * cut_global;
            r_mins(i, j) = 0.0;
        }
    }
    nl_inited = true;
}

// ── Load / Save ──────────────────────────────────────────────────────

void Tersoff::Load(std::istream &is) {
    Tersoff loaded;
    loaded.load_stream(is, {});
    *this = std::move(loaded);
}

void Tersoff::Save(std::ostream &os) {
    os << std::setprecision(17) << "# Tersoff potential\n";
    os << "mode=" << (mode_ == Mode::Brenner ? "brenner" : "tersoff") << "\n";
    // Save per-pair parameters
    for (int i = 0; i < nelements; ++i) {
        for (int j = i; j < nelements; ++j) {
            const auto &p = params[i][j];
            os << element_names_[i] << " " << element_names_[j]
               << " A=" << p.A << " B=" << p.B
               << " lambda1=" << p.lambda1 << " lambda2=" << p.lambda2
               << " lambda3=" << p.lambda3 << " m=" << p.m << " n=" << p.n
               << " beta=" << p.beta
               << " R=" << p.R << " D=" << p.D
               << " c=" << p.c << " d=" << p.d << " h=" << p.h
               << " D0=" << p.D0 << " S=" << p.S
               << " r0=" << p.r0 << " gamma=" << p.gamma << "\n";
        }
    }
}

void Tersoff::Save(const std::string &fnm) {
    if (save_gpm_if_requested(*this, fnm)) return;
    AnyPotential::Save(fnm);
}

// ── getEFS ───────────────────────────────────────────────────────────

void Tersoff::getEFS(Atoms &atom, AnyNeighborhoods &nbhs) {
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();

    for (int ii = 0; ii < nbhs.size(); ++ii) {
        auto &nbh = nbhs[ii];
        int itype = nbh.centertype;
        int ncount = nbh.nbhcount_;

        for (int jj = 0; jj < ncount; ++jj) {
            int jtype = nbh.chemmap_[jj];
            int jind = nbh.indices_[jj];
            double rij = nbh.distances_[jj];
            const auto &p = params[itype][jtype];
            if (rij >= p.R + p.D) continue;

            double fcij = fc(rij, p.R, p.D);
            if (fcij <= 0.0) continue;

            // ══════════════════════════════════════════════════════════
            // BRANCH 1: Pair repulsive / attractive terms
            // ══════════════════════════════════════════════════════════
            double VR, VA, dVR, dVA;
            if (mode_ == Mode::Brenner) {
                double sqrt2S  = std::sqrt(2.0 * p.S);
                double sqrt2oS = std::sqrt(2.0 / p.S);
                double pref    = p.D0 / (p.S - 1.0);
                double dr      = rij - p.r0;
                VR  = pref * std::exp(-sqrt2S  * p.beta * dr);
                VA  = pref * p.S * std::exp(-sqrt2oS * p.beta * dr);
                dVR = -sqrt2S  * p.beta * VR;
                dVA = -sqrt2oS * p.beta * VA;
            } else {
                double exp_l1r = std::exp(-p.lambda1 * rij);
                double exp_l2r = std::exp(-p.lambda2 * rij);
                VR  = p.A * exp_l1r;
                VA  = p.B * exp_l2r;
                dVR = -p.lambda1 * VR;
                dVA = -p.lambda2 * VA;
            }

            // ══════════════════════════════════════════════════════════
            // Zeta summation for bond order
            // ══════════════════════════════════════════════════════════
            double zeta = 0.0;
            for (int kk = 0; kk < ncount; ++kk) {
                if (kk == jj) continue;
                int ktype = nbh.chemmap_[kk];
                double rik = nbh.distances_[kk];
                const auto &pk = params[itype][ktype];
                if (rik >= pk.R + pk.D) continue;

                double fcik = fc(rik, pk.R, pk.D);
                if (fcik <= 0.0) continue;

                double cos_theta = nbh.pos_[jj].dot(nbh.pos_[kk]) / (rij * rik);
                if (cos_theta > 1.0) cos_theta = 1.0;
                if (cos_theta < -1.0) cos_theta = -1.0;

                // BRANCH 2a: angular function lookup
                double g_val;
                if (mode_ == Mode::Brenner) {
                    g_val = pk.gamma * g_theta(cos_theta, pk.c, pk.d, pk.h);
                } else {
                    g_val = g_theta(cos_theta, p.c, p.d, p.h);
                }

                // BRANCH 2b: zeta exponent
                if (mode_ == Mode::Brenner) {
                    zeta += fcik * g_val;
                } else {
                    double dr = rij - rik;
                    zeta += fcik * g_val * std::exp(p.lambda3 * p.lambda3 * p.lambda3
                                                     * dr * dr * dr);
                }
            }

            // ══════════════════════════════════════════════════════════
            // BRANCH 3: Bond order
            // ══════════════════════════════════════════════════════════
            double bij;
            double dbij_dzeta = 0;
            if (mode_ == Mode::Brenner) {
                if (zeta > 0) {
                    double base = 1.0 + zeta;
                    bij = std::pow(base, -0.5);
                    dbij_dzeta = -0.5 * std::pow(base, -1.5);
                } else {
                    bij = 1.0;
                }
            } else {
                if (zeta > 0 && p.n > 0 && p.beta > 0) {
                    double beta_n = std::pow(p.beta, p.n);
                    double zeta_n = std::pow(zeta, p.n);
                    double base = 1.0 + beta_n * zeta_n;
                    double exp_n = -1.0 / (2.0 * p.n);
                    bij = std::pow(base, exp_n);
                    dbij_dzeta = bij * exp_n * beta_n * p.n * std::pow(zeta, p.n - 1) / base;
                } else {
                    bij = 1.0;
                }
            }

            // ══════════════════════════════════════════════════════════
            // Energy and 2-body forces
            // ══════════════════════════════════════════════════════════
            double eij = fcij * (VR - bij * VA) * 0.5;
            atom.energy_ += eij;
            atom.site_energies_[ii] += eij;

            double dfcij = fc_deriv(rij, p.R, p.D);
            double fpair = dfcij * (VR - bij * VA) + fcij * (dVR - bij * dVA);
            fpair *= -0.5 / rij;

            atom.forces_[ii](0) -= nbh.pos_[jj](0) * fpair;
            atom.forces_[ii](1) -= nbh.pos_[jj](1) * fpair;
            atom.forces_[ii](2) -= nbh.pos_[jj](2) * fpair;
            atom.forces_[jind](0) += nbh.pos_[jj](0) * fpair;
            atom.forces_[jind](1) += nbh.pos_[jj](1) * fpair;
            atom.forces_[jind](2) += nbh.pos_[jj](2) * fpair;

            for (int a = 0; a < 3; ++a)
                for (int b = 0; b < 3; ++b)
                    atom.stress_(a, b) += nbh.pos_[jj](a) * fpair * nbh.pos_[jj](b);

            // ══════════════════════════════════════════════════════════
            // 3-body forces from bond-order derivative
            // ══════════════════════════════════════════════════════════
            if (dbij_dzeta == 0) continue;

            for (int kk = 0; kk < ncount; ++kk) {
                if (kk == jj) continue;
                int ktype = nbh.chemmap_[kk];
                int kind = nbh.indices_[kk];
                double rik = nbh.distances_[kk];
                const auto &pk = params[itype][ktype];
                if (rik >= pk.R + pk.D) continue;

                double fcik = fc(rik, pk.R, pk.D);
                if (fcik <= 0.0) continue;

                double cos_theta = nbh.pos_[jj].dot(nbh.pos_[kk]) / (rij * rik);
                if (cos_theta > 1.0) cos_theta = 1.0;
                if (cos_theta < -1.0) cos_theta = -1.0;

                // BRANCH 4: Angular function + zeta exponent for 3-body derivatives
                double gv, dg_dcosth;
                double exp_m = 1.0;
                double dexp_drij = 0, dexp_drik = 0;

                if (mode_ == Mode::Brenner) {
                    gv = pk.gamma * g_theta(cos_theta, pk.c, pk.d, pk.h);
                    dg_dcosth = pk.gamma * g_theta_deriv(cos_theta, pk.c, pk.d, pk.h);
                    // dexp stays zero — no distance-difference term in Brenner
                } else {
                    gv = g_theta(cos_theta, p.c, p.d, p.h);
                    dg_dcosth = g_theta_deriv(cos_theta, p.c, p.d, p.h);
                    double dr = rij - rik;
                    exp_m = std::exp(p.lambda3 * p.lambda3 * p.lambda3 * dr * dr * dr);
                    dexp_drij = exp_m * 3.0 * p.lambda3 * p.lambda3 * p.lambda3 * dr * dr;
                    dexp_drik = -dexp_drij;
                }

                // Force contribution is -dE/dzeta times the zeta gradient.
                double coeff = 0.5 * fcij * VA * dbij_dzeta;

                Eigen::Vector3d r_ij = nbh.pos_[jj];
                Eigen::Vector3d r_ik = nbh.pos_[kk];
                double inv_rij = 1.0 / rij;
                double inv_rik = 1.0 / rik;
                Eigen::Vector3d r_ij_hat = r_ij * inv_rij;
                Eigen::Vector3d r_ik_hat = r_ik * inv_rik;

                Eigen::Vector3d dz_drj = fcik * gv * dexp_drij * r_ij_hat
                    + fcik * exp_m * dg_dcosth * (r_ik_hat - cos_theta * r_ij_hat) * inv_rij;

                Eigen::Vector3d dz_drk = gv * (fcik * dexp_drik + fc_deriv(rik, pk.R, pk.D) * exp_m) * r_ik_hat
                    + fcik * exp_m * dg_dcosth * (r_ij_hat - cos_theta * r_ik_hat) * inv_rik;

                Eigen::Vector3d fj_3b = coeff * dz_drj;
                Eigen::Vector3d fk_3b = coeff * dz_drk;
                Eigen::Vector3d fi_3b = -(fj_3b + fk_3b);

                atom.forces_[ii]   += fi_3b;
                atom.forces_[jind] += fj_3b;
                atom.forces_[kind] += fk_3b;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b)
                        atom.stress_(a, b) += fj_3b(a) * r_ij(b) + fk_3b(a) * r_ik(b);
            }
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════
//  Training / fitting interface
// ═════════════════════════════════════════════════════════════════════════

bool Tersoff::is_field_active(Field f) const {
    switch (f) {
        // Common fields: always active
        case Field::R: case Field::D:
        case Field::c: case Field::d: case Field::h:
            return true;
        // Tersoff-only fields
        case Field::A: case Field::B:
        case Field::lambda1: case Field::lambda2:
        case Field::lambda3: case Field::m: case Field::n:
            return mode_ == Mode::Tersoff;
        // Brenner-only fields
        case Field::D0: case Field::S:
        case Field::r0: case Field::gamma:
            return mode_ == Mode::Brenner;
        // beta is common but used differently per mode
        case Field::beta:
            return true;
    }
    return false;
}

void Tersoff::build_param_map() {
    param_map_.clear();

    // All fields in enum order (excluding mode-specific that don't apply)
    const std::vector<Field> all_fields = {
        Field::R, Field::D, Field::c, Field::d, Field::h,
        Field::A, Field::B, Field::lambda1, Field::lambda2,
        Field::lambda3, Field::m, Field::n, Field::beta,
        Field::D0, Field::S, Field::r0, Field::gamma
    };

    // Unique pairs (i <= j)
    for (int i = 0; i < nelements; ++i) {
        for (int j = i; j < nelements; ++j) {
            for (auto f : all_fields) {
                if (is_field_active(f))
                    param_map_.push_back({f, i, j});
            }
        }
    }

    coeffs_.resize(param_map_.size());
    for (size_t idx = 0; idx < param_map_.size(); ++idx)
        coeffs_[idx] = get_single_param(idx);
}

void Tersoff::unpack_coeffs() {
    for (size_t idx = 0; idx < param_map_.size(); ++idx)
        set_single_param(idx, coeffs_[idx]);
}

double Tersoff::get_single_param(int idx) const {
    const auto &e = param_map_[idx];
    const auto &p = params[e.i][e.j];
    switch (e.field) {
        case Field::R:       return p.R;
        case Field::D:       return p.D;
        case Field::c:       return p.c;
        case Field::d:       return p.d;
        case Field::h:       return p.h;
        case Field::A:       return p.A;
        case Field::B:       return p.B;
        case Field::lambda1: return p.lambda1;
        case Field::lambda2: return p.lambda2;
        case Field::lambda3: return p.lambda3;
        case Field::m:       return p.m;
        case Field::n:       return p.n;
        case Field::beta:    return p.beta;
        case Field::D0:      return p.D0;
        case Field::S:       return p.S;
        case Field::r0:      return p.r0;
        case Field::gamma:   return p.gamma;
    }
    return 0;
}

void Tersoff::set_single_param(int idx, double val) {
    const auto &e = param_map_[idx];
    auto &p = params[e.i][e.j];
    auto &ps = params[e.j][e.i]; // symmetric copy
    switch (e.field) {
        case Field::R:       p.R = val; ps.R = val; break;
        case Field::D:       p.D = val; ps.D = val; break;
        case Field::c:       p.c = val; ps.c = val; break;
        case Field::d:       p.d = val; ps.d = val; break;
        case Field::h:       p.h = val; ps.h = val; break;
        case Field::A:       p.A = val; ps.A = val; break;
        case Field::B:       p.B = val; ps.B = val; break;
        case Field::lambda1: p.lambda1 = val; ps.lambda1 = val; break;
        case Field::lambda2: p.lambda2 = val; ps.lambda2 = val; break;
        case Field::lambda3: p.lambda3 = val; ps.lambda3 = val; break;
        case Field::m:       p.m = val; ps.m = val; break;
        case Field::n:       p.n = val; ps.n = val; break;
        case Field::beta:    p.beta = val; ps.beta = val; break;
        case Field::D0:      p.D0 = val; ps.D0 = val; break;
        case Field::S:       p.S = val; ps.S = val; break;
        case Field::r0:      p.r0 = val; ps.r0 = val; break;
        case Field::gamma:   p.gamma = val; ps.gamma = val; break;
    }
    coeffs_[idx] = val;
}

int Tersoff::get_NL_coeffs_count() const {
    return static_cast<int>(coeffs_.size());
}

void Tersoff::nl_get_coeffs(double *x) const {
    for (size_t i = 0; i < coeffs_.size(); ++i)
        x[i] = coeffs_[i];
}

void Tersoff::nl_set_coeffs(const double *x) {
    for (size_t i = 0; i < coeffs_.size(); ++i)
        coeffs_[i] = x[i];
    unpack_coeffs();

    // Update global cutoff from per-pair R+D
    cut_global = 0;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            double rcut = params[i][j].R + params[i][j].D;
            if (rcut > cut_global) cut_global = rcut;
        }
    }
    cutoff = cut_global;
    cutoffsq = cutoff * cutoff;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            cutoffs(i, j) = cut_global;
            cutoffsqs(i, j) = cut_global * cut_global;
        }
    }
}

void Tersoff::train_init() {
    if (param_map_.empty())
        build_param_map();
    reg_L1_vector.resize(coeffs_.size(), 0);
    reg_L2_vector.resize(coeffs_.size(), 0);
    nl_inited = true;
}

void Tersoff::random_init() {
    if (param_map_.empty())
        build_param_map();

    std::mt19937 gen(std::random_device{}());
    std::uniform_real_distribution<double> dis(-1.0, 1.0);

    for (size_t idx = 0; idx < param_map_.size(); ++idx) {
        const auto &e = param_map_[idx];
        double val = 0;
        switch (e.field) {
            case Field::R:       val = 2.0 + std::abs(dis(gen)) * 3.0; break;
            case Field::D:       val = 0.05 + std::abs(dis(gen)) * 0.5; break;
            case Field::c:       val = std::abs(dis(gen)) * 10.0; break;
            case Field::d:       val = 0.5 + std::abs(dis(gen)) * 5.0; break;
            case Field::h:       val = dis(gen); break;
            case Field::A:       val = std::abs(dis(gen)) * 5000.0 + 100.0; break;
            case Field::B:       val = std::abs(dis(gen)) * 500.0 + 10.0; break;
            case Field::lambda1: val = std::abs(dis(gen)) * 5.0 + 1.0; break;
            case Field::lambda2: val = std::abs(dis(gen)) * 5.0 + 0.5; break;
            case Field::lambda3: val = std::abs(dis(gen)) * 3.0 + 0.5; break;
            case Field::m:       val = std::abs(dis(gen)) * 5.0 + 1.0; break;
            case Field::n:       val = std::abs(dis(gen)) * 3.0 + 0.5; break;
            case Field::beta:    val = std::abs(dis(gen)) * 5.0 + 0.5; break;
            case Field::D0:      val = std::abs(dis(gen)) * 10.0 + 0.1; break;
            case Field::S:       val = std::abs(dis(gen)) * 3.0 + 1.0; break;
            case Field::r0:      val = std::abs(dis(gen)) * 2.0 + 1.5; break;
            case Field::gamma:   val = std::abs(dis(gen)) * 2.0 + 0.1; break;
        }
        coeffs_[idx] = val;
    }
    unpack_coeffs();

    // Update global cutoff
    cut_global = 0;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            double rcut = params[i][j].R + params[i][j].D;
            if (rcut > cut_global) cut_global = rcut;
        }
    }
    cutoff = cut_global;
    cutoffsq = cutoff * cutoff;
    for (int i = 0; i < nelements; ++i) {
        for (int j = 0; j < nelements; ++j) {
            cutoffs(i, j) = cut_global;
            cutoffsqs(i, j) = cut_global * cut_global;
        }
    }
    nl_inited = true;
}

// ═════════════════════════════════════════════════════════════════════════
//  Numerical gradient via central finite differences
// ═════════════════════════════════════════════════════════════════════════

void Tersoff::AccumulateEFSCombinationGard(
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

// Energy-only overload
void Tersoff::AccumulateEFSCombinationGard(
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

namespace {
namespace TersoffDirectional {
// One directional tangent, used to differentiate the existing site gradient.
// Its storage is independent of the number of atoms and neighbors.
struct Scalar {
    double value, tangent;
    Scalar(double v = 0., double d = 0.) : value(v), tangent(d) {}
};
Scalar operator+(Scalar a, Scalar b) { return {a.value+b.value,a.tangent+b.tangent}; }
Scalar operator-(Scalar a, Scalar b) { return {a.value-b.value,a.tangent-b.tangent}; }
Scalar operator-(Scalar a) { return {-a.value,-a.tangent}; }
Scalar operator*(Scalar a, Scalar b) { return {a.value*b.value,a.tangent*b.value+a.value*b.tangent}; }
Scalar operator/(Scalar a, Scalar b) {
    return {a.value/b.value,(a.tangent*b.value-a.value*b.tangent)/(b.value*b.value)};
}
Scalar exp(Scalar a) { const double v = std::exp(a.value); return {v,v*a.tangent}; }
Scalar pow(Scalar a, double exponent) {
    return {std::pow(a.value,exponent),exponent*std::pow(a.value,exponent-1)*a.tangent};
}
Scalar sin(Scalar a) { return {std::sin(a.value),std::cos(a.value)*a.tangent}; }
Scalar cos(Scalar a) { return {std::cos(a.value),-std::sin(a.value)*a.tangent}; }
using Vector = std::array<Scalar,3>;
Vector operator+(const Vector& a, const Vector& b) { return {a[0]+b[0],a[1]+b[1],a[2]+b[2]}; }
Vector operator-(const Vector& a, const Vector& b) { return {a[0]-b[0],a[1]-b[1],a[2]-b[2]}; }
Vector operator*(Scalar s, const Vector& v) { return {s*v[0],s*v[1],s*v[2]}; }
Vector operator/(const Vector& v, Scalar s) { return {v[0]/s,v[1]/s,v[2]/s}; }
Scalar dot(const Vector& a, const Vector& b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }
Scalar cutoff_value(Scalar r, const Tersoff::Param& p) {
    if (r.value < p.R-p.D) return 1.;
    if (r.value > p.R+p.D) return 0.;
    return .5*(1.-sin(M_PI_2*(r-p.R)/p.D));
}
Scalar cutoff_derivative(Scalar r, const Tersoff::Param& p) {
    if (r.value <= p.R-p.D || r.value >= p.R+p.D) return 0.;
    return -.5*M_PI_2/p.D*cos(M_PI_2*(r-p.R)/p.D);
}
Scalar angular(Scalar cosine, const Tersoff::Param& p) {
    const Scalar delta = p.h-cosine;
    return 1.+p.c*p.c/(p.d*p.d)-p.c*p.c/(p.d*p.d+delta*delta);
}
Scalar angular_derivative(Scalar cosine, const Tersoff::Param& p) {
    const Scalar delta = p.h-cosine, denominator = p.d*p.d+delta*delta;
    return -2.*p.c*p.c*delta/(denominator*denominator);
}
Scalar clamped_cosine(const Vector& j, const Vector& k, Scalar r, Scalar s) {
    Scalar cosine = dot(j,k)/(r*s);
    // Match the scalar EFS clamp while retaining the analytic tangent at
    // roundoff-sized collinear overshoots (the angular gradient is zero there).
    cosine.value = std::clamp(cosine.value,-1.,1.);
    return cosine;
}
}
}

bool Tersoff::supports_site_hessian() const {
    return typeid(*this) == typeid(Tersoff) && nelements > 0 &&
        (mode_ == Mode::Tersoff || mode_ == Mode::Brenner);
}
bool Tersoff::supports_site_hvp() const { return supports_site_hessian(); }
bool Tersoff::supports_analytical_hessian() const { return supports_site_hessian(); }
bool Tersoff::supports_analytical_hessian_vector_product() const { return supports_site_hvp(); }
bool Tersoff::supports_analytical_cell_hessian() const { return supports_site_hessian(); }

void Tersoff::site_directional(Neighborhood &n, const SiteCoordinateMatrix &direction,
    double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix &product) {
    using namespace TersoffDirectional;
    std::vector<Vector> bonds(n.nbhcount_), gradients(n.nbhcount_);
    std::vector<Scalar> radii(n.nbhcount_);
    for (int j = 0; j < n.nbhcount_; ++j) {
        for (int a = 0; a < 3; ++a) bonds[j][a] = {n.pos_[j][a],direction(j,a)};
        radii[j] = {n.distances_[j],n.pos_[j].dot(direction.row(j))/n.distances_[j]};
    }
    Scalar total;
    for (int j = 0; j < n.nbhcount_; ++j) {
        const auto &p = params[n.centertype][n.chemmap_[j]];
        const Scalar r = radii[j];
        if (r.value >= p.R+p.D) continue;
        const Scalar fcj = cutoff_value(r,p);
        if (fcj.value <= 0.) continue;
        Scalar vr,va,dvr,dva;
        if (mode_ == Mode::Brenner) {
            const double kr = std::sqrt(2.*p.S)*p.beta, ka = std::sqrt(2./p.S)*p.beta;
            const double prefactor = p.D0/(p.S-1.);
            vr = prefactor*exp(-kr*(r-p.r0));
            va = prefactor*p.S*exp(-ka*(r-p.r0));
            dvr = -kr*vr;
            dva = -ka*va;
        } else {
            vr = p.A*exp(-p.lambda1*r);
            va = p.B*exp(-p.lambda2*r);
            dvr = -p.lambda1*vr;
            dva = -p.lambda2*va;
        }
        Scalar zeta;
        for (int k = 0; k < n.nbhcount_; ++k) {
            if (k == j) continue;
            const auto &pk = params[n.centertype][n.chemmap_[k]];
            const Scalar s = radii[k];
            if (s.value >= pk.R+pk.D) continue;
            const Scalar fck = cutoff_value(s,pk);
            if (fck.value <= 0.) continue;
            const Scalar cosine = clamped_cosine(bonds[j],bonds[k],r,s);
            if (mode_ == Mode::Brenner) zeta = zeta+fck*pk.gamma*angular(cosine,pk);
            else {
                const Scalar dr = r-s;
                zeta = zeta+fck*angular(cosine,p)*exp(p.lambda3*p.lambda3*p.lambda3*dr*dr*dr);
            }
        }
        Scalar bond = 1., dbond = 0.;
        if (mode_ == Mode::Brenner) {
            if (zeta.value > 0.) {
                bond = pow(1.+zeta,-.5);
                dbond = -.5*pow(1.+zeta,-1.5);
            }
        } else if (zeta.value > 0. && p.n > 0. && p.beta > 0.) {
            const double beta_n = std::pow(p.beta,p.n), exponent = -1./(2.*p.n);
            const Scalar base = 1.+beta_n*pow(zeta,p.n);
            bond = pow(base,exponent);
            dbond = bond*exponent*beta_n*p.n*pow(zeta,p.n-1.)/base;
        }
        total = total+.5*fcj*(vr-bond*va);
        const Vector uj = bonds[j]/r;
        const Scalar radial = .5*(cutoff_derivative(r,p)*(vr-bond*va)+fcj*(dvr-bond*dva));
        gradients[j] = gradients[j]+radial*uj;
        // dbond carries b''(zeta)*dot(zeta), so different-neighbor zeta
        // gradients remain coupled through the full bond-order differential.
        if (dbond.value == 0. && dbond.tangent == 0.) continue;
        const Scalar coefficient = .5*fcj*va*dbond;
        for (int k = 0; k < n.nbhcount_; ++k) {
            if (k == j) continue;
            const auto &pk = params[n.centertype][n.chemmap_[k]];
            const Scalar s = radii[k];
            if (s.value >= pk.R+pk.D) continue;
            const Scalar fck = cutoff_value(s,pk);
            if (fck.value <= 0.) continue;
            const Vector uk = bonds[k]/s;
            const Scalar cosine = clamped_cosine(bonds[j],bonds[k],r,s);
            Scalar g,dg,exponential = 1., dexpr = 0., dexps = 0.;
            if (mode_ == Mode::Brenner) {
                g = pk.gamma*angular(cosine,pk);
                dg = pk.gamma*angular_derivative(cosine,pk);
            } else {
                g = angular(cosine,p);
                dg = angular_derivative(cosine,p);
                const Scalar dr = r-s;
                const double lambda_cube = p.lambda3*p.lambda3*p.lambda3;
                exponential = exp(lambda_cube*dr*dr*dr);
                dexpr = exponential*3.*lambda_cube*dr*dr;
                dexps = -dexpr;
            }
            const Vector dzj = (fck*g*dexpr)*uj+(fck*exponential*dg/r)*(uk-cosine*uj);
            const Vector dzk = (g*(fck*dexps+cutoff_derivative(s,pk)*exponential))*uk
                             +(fck*exponential*dg/s)*(uj-cosine*uk);
            gradients[j] = gradients[j]-coefficient*dzj;
            gradients[k] = gradients[k]-coefficient*dzk;
        }
    }
    energy = total.value;
    gradient.resize(n.nbhcount_,3);
    product.resize(n.nbhcount_,3);
    for (int j = 0; j < n.nbhcount_; ++j)
        for (int a = 0; a < 3; ++a) {
            gradient(j,a) = gradients[j][a].value;
            product(j,a) = gradients[j][a].tangent;
        }
}

void Tersoff::apply_site_hvp(Neighborhood &nbh, const SiteCoordinateMatrix &direction,
    double &energy, SiteCoordinateMatrix &gradient, SiteCoordinateMatrix &product) {
    if (!supports_site_hvp()) GPERROR("Unsupported Tersoff/Brenner site HVP");
    if (direction.rows() != nbh.nbhcount_ || !direction.allFinite())
        GPERROR("Invalid Tersoff/Brenner site HVP direction");
    const SiteCoordinateMatrix input = direction;
    site_directional(nbh,input,energy,gradient,product);
}

void Tersoff::evaluate_site_hessian(Neighborhood &nbh, double &energy,
    SiteCoordinateMatrix &gradient, HessianMatrix &hessian) {
    if (!supports_site_hessian()) GPERROR("Unsupported Tersoff/Brenner site Hessian");
    SiteCoordinateMatrix direction = SiteCoordinateMatrix::Zero(nbh.nbhcount_,3), product;
    hessian.resize(nbh.nbhcount_);
    if (nbh.nbhcount_ == 0) site_directional(nbh,direction,energy,gradient,product);
    for (int k = 0; k < nbh.nbhcount_; ++k)
        for (int b = 0; b < 3; ++b) {
            direction.setZero();
            direction(k,b) = 1.;
            site_directional(nbh,direction,energy,gradient,product);
            for (int j = 0; j < nbh.nbhcount_; ++j)
                for (int a = 0; a < 3; ++a) hessian(j,a,k,b) = product(j,a);
        }
}

void Tersoff::applyHessianVectorProduct(Atoms &atom, AnyNeighborhoods &nbhs,
    const Eigen::VectorXd &direction, Eigen::VectorXd &product) {
    if (!supports_analytical_hessian_vector_product()) GPERROR("Unsupported Tersoff/Brenner HVP");
    if (direction.size() != 3*atom.size() || !direction.allFinite())
        GPERROR("Hessian-vector direction must be finite with size 3*N");
    if (nbhs.size() != atom.size()) GPERROR("Hessian-vector neighborhoods have the wrong size");
    Eigen::VectorXd result = Eigen::VectorXd::Zero(direction.size());
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &n = nbhs[i];
        SiteCoordinateMatrix relative(n.nbhcount_,3), gradient, local;
        for (int j = 0; j < n.nbhcount_; ++j)
            relative.row(j) = (direction.segment<3>(3*n.indices_[j])-direction.segment<3>(3*i)).transpose();
        double energy;
        apply_site_hvp(n,relative,energy,gradient,local);
        for (int j = 0; j < n.nbhcount_; ++j) {
            result.segment<3>(3*n.indices_[j]) += local.row(j).transpose();
            result.segment<3>(3*i) -= local.row(j).transpose();
        }
    }
    product = std::move(result);
}

void Tersoff::assemble_hessian(Atoms &atom, AnyNeighborhoods &nbhs,
    HessianMatrix &hessian, Eigen::MatrixXd *cellhess, Eigen::MatrixXd *lrhess) {
    if (!supports_analytical_hessian()) GPERROR("Unsupported Tersoff/Brenner Hessian");
    if (nbhs.size() != atom.size()) GPERROR("Hessian neighborhoods have the wrong size");
    if (cellhess && (!atom.cell_.allFinite() || !atom.cell_.fullPivLu().isInvertible()))
        GPERROR("Analytical Tersoff/Brenner cell Hessian requires a finite invertible cell");
    if (!add_efs) atom.resetEFS(); else atom.setEFSflag();
    hessian.resize(atom.size());
    Eigen::Matrix3d inverse = Eigen::Matrix3d::Zero();
    if (cellhess) {
        inverse = atom.cell_.inverse();
        *cellhess = Eigen::MatrixXd::Zero(9,9);
        *lrhess = Eigen::MatrixXd::Zero(9,3*atom.size());
    }
    for (int i = 0; i < nbhs.size(); ++i) {
        auto &n = nbhs[i];
        double energy;
        SiteCoordinateMatrix gradient;
        HessianMatrix local;
        evaluate_site_hessian(n,energy,gradient,local);
        atom.energy_ += energy;
        atom.site_energies_[i] += energy;
        for (int j = 0; j < n.nbhcount_; ++j) {
            atom.forces_[i] += gradient.row(j).transpose();
            atom.forces_[n.indices_[j]] -= gradient.row(j).transpose();
            atom.stress_ -= gradient.row(j).transpose()*n.pos_[j];
        }
        for (int j = 0; j < n.nbhcount_; ++j)
            for (int k = 0; k < n.nbhcount_; ++k) {
                const auto block = local(j,k);
                const int sj = n.indices_[j], sk = n.indices_[k];
                hessian(sj,sk) += block;
                hessian(i,sk) -= block;
                hessian(sj,i) -= block;
                hessian(i,i) += block;
                if (cellhess) {
                    Eigen::Matrix<double,3,9> aj = Eigen::Matrix<double,3,9>::Zero();
                    Eigen::Matrix<double,3,9> ak = Eigen::Matrix<double,3,9>::Zero();
                    const Eigen::RowVector3d fj = n.pos_[j]*inverse, fk = n.pos_[k]*inverse;
                    for (int a = 0; a < 3; ++a)
                        for (int b = 0; b < 3; ++b) {
                            aj(b,3*a+b) = fj(a);
                            ak(b,3*a+b) = fk(a);
                        }
                    *cellhess += aj.transpose()*block*ak;
                    const Eigen::Matrix<double,9,3> mixed = aj.transpose()*block;
                    lrhess->middleCols(3*sk,3) += mixed;
                    lrhess->middleCols(3*i,3) -= mixed;
                }
            }
    }
}

void Tersoff::getEFSHessian(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    assemble_hessian(atom,nbhs,hessian,nullptr,nullptr);
}
void Tersoff::getEFSHessian(Atoms &atom, HessianMatrix &hessian) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessian(atom,nbhs,hessian);
}
void Tersoff::getEFSHessianOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian) {
    getEFSHessian(atom,nbhs,hessian);
}
void Tersoff::getEFSHessianSymm(Atoms &atom, HessianMatrix &hessian, double) {
    getEFSHessian(atom,hessian);
}
void Tersoff::getEFSHessianCell(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    assemble_hessian(atom,nbhs,hessian,&cellhess,&lrhess);
}
void Tersoff::getEFSHessianCell(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    Neighborhoods nbhs(atom,cutoff);
    getEFSHessianCell(atom,nbhs,hessian,cellhess,lrhess);
}
void Tersoff::getEFSHessianCellOMP(Atoms &atom, AnyNeighborhoods &nbhs, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess) {
    getEFSHessianCell(atom,nbhs,hessian,cellhess,lrhess);
}
void Tersoff::getEFSHessianCellSymm(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    getEFSHessianCell(atom,hessian,cellhess,lrhess);
}
void Tersoff::getEFSHessianCellSymmOMP(Atoms &atom, HessianMatrix &hessian,
    Eigen::MatrixXd &cellhess, Eigen::MatrixXd &lrhess, double) {
    getEFSHessianCell(atom,hessian,cellhess,lrhess);
}
