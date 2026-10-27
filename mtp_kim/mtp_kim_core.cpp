// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "mtp_kim_core.h"
#include <cstdlib>
#include <cstring>

namespace mtp_kim {

// ── Helper: skip delimiters { } , = and whitespace ──────────────────
static void skipDelimiters(std::istream &ifs) {
    char c;
    while (ifs.get(c)) {
        if (c != '{' && c != '}' && c != ',' && c != ' ' && c != '=' && c != '\r' && c != '\n') {
            ifs.unget(); return;
        }
    }
}

// ── Helper: parse braced list {v1, v2, ...} ──────────────────────────
static void readBracedList(std::istream &ifs, std::vector<double> &out) {
    char c;
    if (!(ifs >> c) || c != '{')
        throw std::runtime_error("expected '{'");
    std::string buf;
    auto append = [&out](const std::string &text) {
        if (text.empty()) throw std::runtime_error("empty list entry");
        std::size_t parsed = 0;
        const double value = std::stod(text, &parsed);
        if (parsed != text.size() || !std::isfinite(value))
            throw std::runtime_error("invalid list entry");
        out.push_back(value);
    };
    while (ifs >> c && c != '}') {
        if (c == ',') { append(buf); buf.clear(); }
        else buf += c;
    }
    if (!ifs || c != '}') throw std::runtime_error("unterminated list");
    append(buf);
}

// ── Destructor ─────────────────────────────────────────────────────────
MTPCore::~MTPCore() {
    delete p_radial_basis;
    for (auto *b : p_radial_basis_pair) delete b;
}

// ── Radial basis evaluation helper ────────────────────────────────────
void MTPCore::evalRadial(int itype, int jtype, double r,
                           double *vals_out, double *ders_out) const {
    if (radial_basis_split) {
        auto *basis = p_radial_basis_pair[itype * species_count + jtype];
        basis->evaluate(r, vals_out, ders_out);
    } else {
        p_radial_basis->evaluate(r, vals_out, ders_out);
    }
}

// ── Load .mtp file ────────────────────────────────────────────────────
bool MTPCore::load(const std::string &filepath) {
    std::ifstream ifs(filepath);
    if (!ifs.is_open()) return false;
    return load(ifs);
}

bool MTPCore::load(std::istream &ifs) {
    std::string line, tmpstr;
    char tmpline[4096];

    delete p_radial_basis;
    p_radial_basis = nullptr;
    for (auto *basis : p_radial_basis_pair) delete basis;
    p_radial_basis_pair.clear();
    radial_basis_split = false;
    m_bufferPool.clear();
    m_nextBuffer = 0;
    m_maxBufCount = 0;

    // ── Magic ─────────────────────────────────────────────────────────
    ifs.getline(tmpline, sizeof(tmpline));
    if (std::string(tmpline).find("MTP") != 0) {
        std::cerr << "Error: not an MTP file" << std::endl;
        return false;
    }

    // ── Version ───────────────────────────────────────────────────────
    ifs.getline(tmpline, sizeof(tmpline));
    std::string verLine(tmpline);
    if (verLine.find("version = 1.1.0") == std::string::npos) {
        std::cerr << "Error: MTP file must have version 1.1.0" << std::endl;
        return false;
    }

    // ── Optional: potential_name ──────────────────────────────────────
    ifs >> tmpstr;
    if (tmpstr == "potential_name") {
        ifs.ignore(10, '='); ifs >> tmpstr; // name
        ifs >> tmpstr; // next token
    }

    // ── Optional: scaling ─────────────────────────────────────────────
    double scaling = 0.01;
    if (tmpstr == "scaling") {
        ifs.ignore(10, '='); ifs >> scaling; ifs >> tmpstr;
    }
    if (!ifs || !std::isfinite(scaling)) {
        std::cerr << "Error: invalid scaling" << std::endl;
        return false;
    }

    // ── species_count ─────────────────────────────────────────────────
    if (tmpstr != "species_count") {
        std::cerr << "Error: expected species_count, got " << tmpstr << std::endl;
        return false;
    }
    ifs.ignore(10, '='); ifs >> species_count;
    if (!ifs || species_count <= 0) {
        std::cerr << "Error: species_count must be positive" << std::endl;
        return false;
    }
    ifs >> tmpstr;

    // ── Optional: potential_tag ───────────────────────────────────────
    if (tmpstr == "potential_tag") {
        ifs.ignore(10, '='); // skip " = "
        getline(ifs, line);  // consume the tag value (or empty)
        ifs >> tmpstr;
    }

    // ── Optional: radial_split ────────────────────────────────────────
    if (tmpstr == "radial_split" || tmpstr == "radial_basis_split") {
        ifs.ignore(10, '='); int split; ifs >> split;
        radial_basis_split = (split != 0);
        if (!radial_basis_split) ifs >> tmpstr; // next token
    }

    // ── Radial basis ──────────────────────────────────────────────────
    if (radial_basis_split) {
        p_radial_basis_pair.resize(species_count * species_count, nullptr);
        cutoffs.resize(species_count * species_count);
        r_mins.resize(species_count * species_count);
        max_cutoff = 0.0;
        std::vector<bool> pair_seen(species_count * species_count, false);
        int common_basis_size = 0;
        for (int i = 0; i < species_count * species_count; ++i) {
            int ci, cj;
            if (!(ifs >> tmpstr) || tmpstr != "pair" || !(ifs >> ci >> cj) ||
                ci < 0 || ci >= species_count || cj < 0 || cj >= species_count) {
                std::cerr << "Error: invalid split radial pair definition" << std::endl;
                return false;
            }
            const int pair_index = ci * species_count + cj;
            if (pair_seen[pair_index]) {
                std::cerr << "Error: duplicate split radial pair "
                          << ci << ' ' << cj << std::endl;
                return false;
            }
            pair_seen[pair_index] = true;
            std::string equals;
            std::string bname;
            if (!(ifs >> tmpstr >> equals >> bname) ||
                tmpstr != "radial_basis_type" || equals != "=") {
                std::cerr << "Error: expected radial_basis_type for pair "
                          << ci << ' ' << cj << std::endl;
                return false;
            }
            try {
                auto basis = RadialBasis::load(bname, ifs, scaling);
                if (common_basis_size == 0)
                    common_basis_size = basis->basis_size;
                else if (basis->basis_size != common_basis_size) {
                    std::cerr << "Error: split radial bases must have the same "
                                 "radial_basis_size" << std::endl;
                    return false;
                }
                const double cut = basis->max_dist;
                max_cutoff = std::max(max_cutoff, cut);
                cutoffs[pair_index] = cut;
                r_mins[pair_index] = basis->min_dist;
                p_radial_basis_pair[pair_index] = basis.release();
            } catch (const std::exception &error) {
                std::cerr << "Error loading " << bname << " for pair "
                          << ci << ' ' << cj << ": " << error.what() << std::endl;
                return false;
            }
        }
        rb_size = common_basis_size;
    } else {
        if (tmpstr != "radial_basis_type") {
            std::cerr << "Error: expected radial_basis_type, got " << tmpstr << std::endl;
            return false;
        }
        ifs.ignore(10, '='); std::string bname; ifs >> bname;
        try {
            auto basis = RadialBasis::load(bname, ifs, scaling);
            max_cutoff = basis->max_dist;
            rb_size = basis->basis_size;
            p_radial_basis = basis.release();
        } catch (const std::exception &error) {
            std::cerr << "Error loading " << bname << ": "
                      << error.what() << std::endl;
            return false;
        }
        cutoffs.assign(species_count * species_count, max_cutoff);
        r_mins.assign(species_count * species_count, p_radial_basis->min_dist);
    }

    // ── radial_funcs_count ────────────────────────────────────────────
    ifs >> tmpstr;
    if (tmpstr != "radial_funcs_count") {
        std::cerr << "Error: expected radial_funcs_count, got " << tmpstr << std::endl;
        return false;
    }
    ifs.ignore(10, '='); ifs >> radial_func_count;
    if (!ifs || radial_func_count <= 0) {
        std::cerr << "Error: radial_funcs_count must be positive" << std::endl;
        return false;
    }

    // ── radial_coeffs ─────────────────────────────────────────────────
    int pairs_count = species_count * species_count;
    radial_coeffs.assign(pairs_count * radial_func_count * rb_size, 0.0);

    ifs >> tmpstr;
    if (tmpstr == "radial_coeffs") {
        std::vector<bool> coefficient_pair_seen(pairs_count, false);
        for (int sp = 0; sp < pairs_count; ++sp) {
            int ts1, ts2;
            char foo;
            if (!(ifs >> ts1 >> foo >> ts2) || foo != '-' ||
                ts1 < 0 || ts1 >= species_count ||
                ts2 < 0 || ts2 >= species_count) {
                std::cerr << "Error: invalid radial coefficient pair" << std::endl;
                return false;
            }
            const int coefficient_pair = ts1 * species_count + ts2;
            if (coefficient_pair_seen[coefficient_pair]) {
                std::cerr << "Error: duplicate radial coefficient pair "
                          << ts1 << '-' << ts2 << std::endl;
                return false;
            }
            coefficient_pair_seen[coefficient_pair] = true;

            for (int mu = 0; mu < radial_func_count; ++mu) {
                std::vector<double> braced;
                try {
                    readBracedList(ifs, braced);
                } catch (const std::exception &) {
                    std::cerr << "Error: invalid radial coefficients for pair "
                              << ts1 << '-' << ts2 << std::endl;
                    return false;
                }
                if ((int)braced.size() != rb_size) {
                    std::cerr << "Error: radial coeff size mismatch for pair "
                              << ts1 << "-" << ts2 << " mu=" << mu << std::endl;
                    return false;
                }
                int base = (ts1 * species_count + ts2) * radial_func_count * rb_size + mu * rb_size;
                for (int xi = 0; xi < rb_size; ++xi) {
                    radial_coeffs[base + xi] = braced[xi];
                }
            }
        }
        ifs >> tmpstr; // next token
    }

    // ── alpha_moments_count ───────────────────────────────────────────
    if (tmpstr != "alpha_moments_count") {
        std::cerr << "Error: expected alpha_moments_count, got " << tmpstr << std::endl;
        return false;
    }
    ifs.ignore(10, '='); ifs >> alpha_moments_count;
    if (!ifs || alpha_moments_count <= 0) {
        std::cerr << "Error: alpha_moments_count must be positive" << std::endl;
        return false;
    }
    ifs >> tmpstr;

    // ── alpha_index_basic_count ───────────────────────────────────────
    if (tmpstr != "alpha_index_basic_count") {
        std::cerr << "Error: expected alpha_index_basic_count" << std::endl;
        return false;
    }
    ifs.ignore(10, '='); ifs >> alpha_index_basic_count;
    if (!ifs || alpha_index_basic_count <= 0 ||
        alpha_index_basic_count > alpha_moments_count) {
        std::cerr << "Error: invalid alpha_index_basic_count" << std::endl;
        return false;
    }
    ifs >> tmpstr;

    // ── alpha_index_basic ─────────────────────────────────────────────
    if (tmpstr != "alpha_index_basic") {
        std::cerr << "Error: expected alpha_index_basic" << std::endl;
        return false;
    }
    // Skip " = {{" or " = " (depends on format variant)
    { char c; while (ifs.get(c) && (c == ' ' || c == '=' || c == '{' || c == '\r')); ifs.unget(); }
    alpha_index_basic.resize(alpha_index_basic_count * 4);
    nu_arr.resize(alpha_index_basic_count);
    max_mu = 0;
    max_alpha_index_basic = 0;
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        int mu, b1, b2, b3;
        // Skip delimiters { , } with flexible parsing
        char c;
        while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ' || c == '\r' || c == '\n')) {}
        ifs.unget();
        if (!(ifs >> mu)) {
            std::cerr << "Error: invalid alpha_index_basic" << std::endl;
            return false;
        }
        while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ')) {}
        ifs.unget();
        if (!(ifs >> b1)) {
            std::cerr << "Error: invalid alpha_index_basic" << std::endl;
            return false;
        }
        while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ')) {}
        ifs.unget();
        if (!(ifs >> b2)) {
            std::cerr << "Error: invalid alpha_index_basic" << std::endl;
            return false;
        }
        while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ')) {}
        ifs.unget();
        if (!(ifs >> b3) || mu < 0 || mu >= radial_func_count ||
            b1 < 0 || b2 < 0 || b3 < 0) {
            std::cerr << "Error: invalid alpha_index_basic" << std::endl;
            return false;
        }
        alpha_index_basic[i * 4 + 0] = mu;
        alpha_index_basic[i * 4 + 1] = b1;
        alpha_index_basic[i * 4 + 2] = b2;
        alpha_index_basic[i * 4 + 3] = b3;
        if (mu > max_mu) max_mu = mu;
        int nu = b1 + b2 + b3;
        nu_arr[i] = nu;
        if (nu > max_alpha_index_basic) max_alpha_index_basic = nu;
    }
    max_mu++; // convert max index to count
    max_alpha_index_basic++; // convert max index to count
    if (max_mu != radial_func_count) {
        std::cerr << "Error: alpha_index_basic does not use exactly "
                  << "radial_funcs_count radial functions" << std::endl;
        return false;
    }

    skipDelimiters(ifs);
    ifs >> tmpstr;

    // ── alpha_index_times_count ───────────────────────────────────────
    if (tmpstr != "alpha_index_times_count") {
        std::cerr << "Error: expected alpha_index_times_count, got " << tmpstr << std::endl;
        return false;
    }
    ifs.ignore(10, '='); ifs >> alpha_index_times_count;
    if (!ifs || alpha_index_times_count < 0) {
        std::cerr << "Error: invalid alpha_index_times_count" << std::endl;
        return false;
    }
    skipDelimiters(ifs);
    ifs >> tmpstr;

    // ── alpha_index_times ─────────────────────────────────────────────
    if (tmpstr != "alpha_index_times") {
        std::cerr << "Error: expected alpha_index_times" << std::endl;
        return false;
    }
    // Skip " = {{" or similar
    { char c; while (ifs.get(c) && (c == ' ' || c == '=' || c == '{' || c == '\r')); ifs.unget(); }
    if (alpha_index_times_count > 0) {
        alpha_index_times.resize(alpha_index_times_count * 4);
        for (int i = 0; i < alpha_index_times_count; ++i) {
            int id1, id2, times, bid;
            char c;
            while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ' || c == '\r' || c == '\n')) {}
            ifs.unget();
            if (!(ifs >> id1)) {
                std::cerr << "Error: invalid alpha_index_times" << std::endl;
                return false;
            }
            while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ')) {}
            ifs.unget();
            if (!(ifs >> id2)) {
                std::cerr << "Error: invalid alpha_index_times" << std::endl;
                return false;
            }
            while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ')) {}
            ifs.unget();
            if (!(ifs >> times)) {
                std::cerr << "Error: invalid alpha_index_times" << std::endl;
                return false;
            }
            while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ')) {}
            ifs.unget();
            if (!(ifs >> bid) || id1 < 0 || id1 >= alpha_moments_count ||
                id2 < 0 || id2 >= alpha_moments_count ||
                bid < 0 || bid >= alpha_moments_count) {
                std::cerr << "Error: invalid alpha_index_times" << std::endl;
                return false;
            }
            alpha_index_times[i * 4 + 0] = id1;
            alpha_index_times[i * 4 + 1] = id2;
            alpha_index_times[i * 4 + 2] = times;
            alpha_index_times[i * 4 + 3] = bid;
        }
    }
    skipDelimiters(ifs);
    ifs >> tmpstr;

    // ── alpha_scalar_moments ──────────────────────────────────────────
    if (tmpstr != "alpha_scalar_moments") {
        std::cerr << "Error: expected alpha_scalar_moments, got " << tmpstr << std::endl;
        return false;
    }
    ifs.ignore(10, '='); ifs >> alpha_scalar_moments;
    if (!ifs || alpha_scalar_moments <= 0) {
        std::cerr << "Error: alpha_scalar_moments must be positive" << std::endl;
        return false;
    }
    ifs >> tmpstr;

    // ── alpha_moment_mapping ──────────────────────────────────────────
    if (tmpstr != "alpha_moment_mapping") {
        std::cerr << "Error: expected alpha_moment_mapping" << std::endl;
        return false;
    }
    // Skip " = {{" or similar
    { char c; while (ifs.get(c) && (c == ' ' || c == '=' || c == '{')); ifs.unget(); }
    alpha_moment_mapping.resize(alpha_scalar_moments);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        char c;
        while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ' || c == '\r' || c == '\n')) {}
        ifs.unget();
        if (!(ifs >> alpha_moment_mapping[i]) || alpha_moment_mapping[i] < 0 ||
            alpha_moment_mapping[i] >= alpha_moments_count) {
            std::cerr << "Error: invalid alpha_moment_mapping" << std::endl;
            return false;
        }
    }
    skipDelimiters(ifs);
    ifs >> tmpstr;

    // ── species_coeffs (optional) ─────────────────────────────────────
    species_coeffs.assign(species_count, 0.0);
    if (tmpstr == "species_coeffs") {
        // Handle " = {val1, val2, ...}" format
        char c;
        while (ifs.get(c) && (c == ' ' || c == '=' || c == '{' || c == '\r' || c == '\n')) {}
        ifs.unget();
        for (int i = 0; i < species_count; ++i) {
            while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ' || c == '\r' || c == '\n')) {}
            ifs.unget();
            if (!(ifs >> species_coeffs[i]) || !std::isfinite(species_coeffs[i])) {
                std::cerr << "Error: invalid species_coeffs" << std::endl;
                return false;
            }
        }
        skipDelimiters(ifs);
        ifs >> tmpstr;
    }

    // ── moment_coeffs (optional) ──────────────────────────────────────
    moment_coeffs.assign(alpha_scalar_moments, 0.0);
    if (tmpstr == "moment_coeffs") {
        // Handle " = {val1, val2, ...}" format
        char c;
        while (ifs.get(c) && (c == ' ' || c == '=' || c == '{' || c == '\r' || c == '\n')) {}
        ifs.unget();
        for (int i = 0; i < alpha_scalar_moments; ++i) {
            while (ifs.get(c) && (c == '{' || c == '}' || c == ',' || c == ' ' || c == '\r' || c == '\n')) {}
            ifs.unget();
            if (!(ifs >> moment_coeffs[i]) || !std::isfinite(moment_coeffs[i])) {
                std::cerr << "Error: invalid moment_coeffs" << std::endl;
                return false;
            }
        }
        skipDelimiters(ifs);
        ifs >> tmpstr;
    }

    // ── Initialize thread buffer pool ─────────────────────────────────
    m_maxBufCount = 1; // Will be increased if needed by acquireBuffer

    return true;
}

// ── Buffer pool management ────────────────────────────────────────────

void MTPCore::ensureBufferPool(int bufCount) const {
    if ((int)m_bufferPool.size() >= bufCount) return;
    m_bufferPool.resize(bufCount);
    int maxJacSize = alpha_index_basic_count * 200 * 3; // default for ~200 neighbors
    for (auto &buf : m_bufferPool) {
        buf.ensure(rb_size, max_mu, max_alpha_index_basic,
                   alpha_moments_count, maxJacSize);
    }
}

MTPCore::ThreadBuffer &MTPCore::acquireBuffer() const {
    if (m_bufferPool.empty()) ensureBufferPool(m_maxBufCount);
    // Round-robin: safe for single-threaded and OMP use
    int idx = m_nextBuffer;
    m_nextBuffer = (m_nextBuffer + 1) % (int)m_bufferPool.size();
    // Ensure jacobian is sized for the current call (resized inside calcSiteEFS)
    return m_bufferPool[idx];
}

// ── calcSiteEFS (pool version) ────────────────────────────────────────

void MTPCore::calcSiteEFS(int centertype,
                            int nNeighbors,
                            const Vec3 *relPos,
                            const double *dists,
                            const int *jtypes,
                            double &site_energy,
                            double *site_energy_ders) const
{
    auto &buf = acquireBuffer();
    calcSiteEFS(centertype, nNeighbors, relPos, dists, jtypes,
                site_energy, site_energy_ders, buf);
}

// ── calcSiteEFS (thread-safe buffer version) ──────────────────────────

void MTPCore::calcSiteEFS(int centertype,
                            int nNeighbors,
                            const Vec3 *relPos,
                            const double *dists,
                            const int *jtypes,
                            double &site_energy,
                            double *site_energy_ders,
                            ThreadBuffer &buf) const
{
    // Zero output
    site_energy = 0.0;
    for (int j = 0; j < nNeighbors * 3; ++j) site_energy_ders[j] = 0.0;

    // Allocate or resize Jacobian buffer
    int jacSize = alpha_index_basic_count * nNeighbors * 3;
    if ((int)buf.moment_jacobian.size() < jacSize)
        buf.moment_jacobian.assign(jacSize, 0.0);
    else
        std::fill(buf.moment_jacobian.begin(), buf.moment_jacobian.begin() + jacSize, 0.0);

    // Zero reusable buffers
    std::fill(buf.moment_vals.begin(), buf.moment_vals.end(), 0.0);

    const int itype = centertype;

    // ── Loop over neighbors ────────────────────────────────────────────
    for (int jj = 0; jj < nNeighbors; ++jj) {
        const double rij = dists[jj];
        const int jtype = jtypes[jj];
        int pairIdx = itype * species_count + jtype;
        if (rij >= cutoffs[pairIdx] || rij <= r_mins[pairIdx]) continue;

        // Evaluate radial basis
        std::fill(buf.rb_vals.begin(), buf.rb_vals.end(), 0.0);
        std::fill(buf.rb_ders.begin(), buf.rb_ders.end(), 0.0);
        evalRadial(itype, jtype, rij, buf.rb_vals.data(), buf.rb_ders.data());

        const double rijinv = 1.0 / rij;
        const double rx = relPos[jj].x, ry = relPos[jj].y, rz = relPos[jj].z;

        // Precompute distance powers and coordinate powers
        // Initialize: buf.dist_powers[0] = 1/r^0 = 1; buf.coords_powers[0*3+a] = pos^0 = 1
        buf.dist_powers[0] = 1.0;
        buf.coords_powers[0 * 3 + 0] = 1.0;
        buf.coords_powers[0 * 3 + 1] = 1.0;
        buf.coords_powers[0 * 3 + 2] = 1.0;
        for (int k = 1; k < max_alpha_index_basic; ++k) {
            buf.dist_powers[k] = buf.dist_powers[k - 1] * rijinv;
            // buf.coords_powers: offset by 1 (buf.coords_powers[(k)*3+a] = pos^a ^ (k-1))
            // buf.coords_powers[1*3+a] = rx, buf.coords_powers[2*3+a] = rx^2, ...
            buf.coords_powers[k * 3 + 0] = buf.coords_powers[(k - 1) * 3 + 0] * rx;
            buf.coords_powers[k * 3 + 1] = buf.coords_powers[(k - 1) * 3 + 1] * ry;
            buf.coords_powers[k * 3 + 2] = buf.coords_powers[(k - 1) * 3 + 2] * rz;
        }

        // Compute buf.val_mu[mu] = sum_xi radial_coeffs[idx] * buf.rb_vals[xi]
        for (int mu = 0; mu < max_mu; ++mu) {
            buf.val_mu[mu] = 0.0;
            buf.der_mu[mu] = 0.0;
            int idxoff = pairIdx * radial_func_count * rb_size + mu * rb_size;
            for (int xi = 0; xi < rb_size; ++xi) {
                buf.val_mu[mu] += radial_coeffs[idxoff + xi] * buf.rb_vals[xi];
                buf.der_mu[mu] += radial_coeffs[idxoff + xi] * buf.rb_ders[xi];
            }
        }

        // ── Basic moments ──────────────────────────────────────────────
        for (int i = 0; i < alpha_index_basic_count; ++i) {
            int mu = alpha_index_basic[i * 4 + 0];
            int nu_x = alpha_index_basic[i * 4 + 1];
            int nu_y = alpha_index_basic[i * 4 + 2];
            int nu_z = alpha_index_basic[i * 4 + 3];
            int k = nu_arr[i];  // nu_x + nu_y + nu_z

            double val = buf.val_mu[mu];
            double der = buf.der_mu[mu];

            val *= buf.dist_powers[k];
            der = der * buf.dist_powers[k] - k * val * rijinv;

            // Coordinate powers: buf.coords_powers[k*3+a] = pos[a]^k
            double powX = buf.coords_powers[nu_x * 3 + 0];
            double powY = buf.coords_powers[nu_y * 3 + 1];
            double powZ = buf.coords_powers[nu_z * 3 + 2];

            double mult0 = powX * powY * powZ;
            buf.moment_vals[i] += val * mult0;

            // Jacobian: d(b_i) / d(x_{j,a})
            double mult1 = der / rij;
            buf.moment_jacobian[i * nNeighbors * 3 + jj * 3 + 0] += mult1 * mult0 * rx;
            buf.moment_jacobian[i * nNeighbors * 3 + jj * 3 + 1] += mult1 * mult0 * ry;
            buf.moment_jacobian[i * nNeighbors * 3 + jj * 3 + 2] += mult1 * mult0 * rz;

            // Polynomial derivative terms: d(x^n)/dx = n * x^{n-1}
            if (nu_x > 0) {
                double powXm1 = buf.coords_powers[(nu_x - 1) * 3 + 0];
                buf.moment_jacobian[i * nNeighbors * 3 + jj * 3 + 0] +=
                    nu_x * val * powXm1 * powY * powZ;
            }
            if (nu_y > 0) {
                double powYm1 = buf.coords_powers[(nu_y - 1) * 3 + 1];
                buf.moment_jacobian[i * nNeighbors * 3 + jj * 3 + 1] +=
                    nu_y * val * powX * powYm1 * powZ;
            }
            if (nu_z > 0) {
                double powZm1 = buf.coords_powers[(nu_z - 1) * 3 + 2];
                buf.moment_jacobian[i * nNeighbors * 3 + jj * 3 + 2] +=
                    nu_z * val * powX * powY * powZm1;
            }
        }
    }

    // ── Non-elementary moments (multiplication rules) ──────────────────
    for (int i = 0; i < alpha_index_times_count; ++i) {
        int id1 = alpha_index_times[i * 4 + 0];
        int id2 = alpha_index_times[i * 4 + 1];
        int times = alpha_index_times[i * 4 + 2];
        int bid  = alpha_index_times[i * 4 + 3];
        buf.moment_vals[bid] += times * buf.moment_vals[id1] * buf.moment_vals[id2];
    }

    // ── Site energy from linear coefficients ──────────────────────────
    site_energy += species_coeffs[centertype];
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        site_energy += moment_coeffs[i] * buf.moment_vals[alpha_moment_mapping[i]];
    }

    // ── Backpropagation ───────────────────────────────────────────────
    // Step 1: dE/dB_i for scalar moments
    std::fill(buf.moment_ders.begin(),
              buf.moment_ders.end(), 0.0);
    for (int i = 0; i < alpha_scalar_moments; ++i) {
        buf.moment_ders[alpha_moment_mapping[i]] = moment_coeffs[i];
    }

    // Step 2: Backprop through multiplication rules (reverse order)
    for (int i = alpha_index_times_count - 1; i >= 0; --i) {
        int id1 = alpha_index_times[i * 4 + 0];
        int id2 = alpha_index_times[i * 4 + 1];
        int times = alpha_index_times[i * 4 + 2];
        int bid  = alpha_index_times[i * 4 + 3];

        buf.moment_ders[id1] +=
            buf.moment_ders[bid] * times * buf.moment_vals[id2];
        buf.moment_ders[id2] +=
            buf.moment_ders[bid] * times * buf.moment_vals[id1];
    }

    // Step 3: Multiply by Jacobian to get site_energy_ders
    for (int i = 0; i < alpha_index_basic_count; ++i) {
        double dEdB = buf.moment_ders[i];
        if (std::abs(dEdB) < 1e-40) continue;
        for (int j = 0; j < nNeighbors; ++j) {
            for (int a = 0; a < 3; ++a) {
                site_energy_ders[j * 3 + a] +=
                    dEdB * buf.moment_jacobian[i * nNeighbors * 3 + j * 3 + a];
            }
        }
    }
}

// ── computeEFS — total system energy/forces/virial ─────────────────────

void MTPCore::computeEFS(int nAtoms,
                           const int *speciesCodes,
                           const int *contributing,
                           const double *coordinates,
                           double &energy,
                           double *forces,
                           double *virial)
{
    energy = 0.0;
    if (nAtoms < 0 || (nAtoms > 0 && (!speciesCodes || !coordinates)))
        return;
    if (forces) std::fill(forces, forces + nAtoms * 3, 0.0);
    if (virial) std::fill(virial, virial + 6, 0.0);
    for (int i = 0; i < nAtoms; ++i)
        if (speciesCodes[i] < 0 || speciesCodes[i] >= species_count)
            return;

    // Build neighbor lists using simple O(N^2) — caller should provide better
    // This is the fallback; KIM models use the GetNeighborList callback instead
    std::vector<Vec3> pos(nAtoms);
    for (int i = 0; i < nAtoms; ++i)
        pos[i] = {coordinates[3*i], coordinates[3*i+1], coordinates[3*i+2]};

    // Temporary per-atom buffers
    std::vector<Vec3> relPos;
    std::vector<double> dists;
    std::vector<int> jtypes;
    std::vector<int> neighbor_atoms;
    std::vector<double> ders;
    ThreadBuffer buffer;
    buffer.ensure(rb_size, max_mu, max_alpha_index_basic,
                  alpha_moments_count, 0);

    for (int i = 0; i < nAtoms; ++i) {
        if (contributing && !contributing[i]) continue;
        int itype = speciesCodes[i];

        // Find neighbors
        relPos.clear(); dists.clear(); jtypes.clear(); neighbor_atoms.clear();
        for (int j = 0; j < nAtoms; ++j) {
            if (i == j) continue;
            Vec3 dr = {pos[j].x - pos[i].x, pos[j].y - pos[i].y, pos[j].z - pos[i].z};
            double r = len(dr);
            int jtype = speciesCodes[j];
            if (r < cutoffs[itype * species_count + jtype] && r > r_mins[itype * species_count + jtype]) {
                relPos.push_back(dr);
                dists.push_back(r);
                jtypes.push_back(jtype);
                neighbor_atoms.push_back(j);
            }
        }

        int nNbh = (int)relPos.size();
        if (nNbh == 0) { energy += species_coeffs[itype]; continue; }

        ders.assign(nNbh * 3, 0.0);
        double siteE = 0.0;
        calcSiteEFS(itype, nNbh, relPos.data(), dists.data(), jtypes.data(),
                    siteE, ders.data(), buffer);
        energy += siteE;

        // Accumulate forces
        if (forces || virial) {
            for (int jj = 0; jj < nNbh; ++jj) {
                int j = neighbor_atoms[jj];
                if (forces) {
                    forces[3*i + 0] += ders[jj*3 + 0];
                    forces[3*i + 1] += ders[jj*3 + 1];
                    forces[3*i + 2] += ders[jj*3 + 2];
                    forces[3*j + 0] -= ders[jj*3 + 0];
                    forces[3*j + 1] -= ders[jj*3 + 1];
                    forces[3*j + 2] -= ders[jj*3 + 2];
                }

                // Virial
                if (virial) {
                    virial[0] += ders[jj*3 + 0] * relPos[jj].x; // 11
                    virial[1] += ders[jj*3 + 1] * relPos[jj].y; // 22
                    virial[2] += ders[jj*3 + 2] * relPos[jj].z; // 33
                    virial[3] += ders[jj*3 + 1] * relPos[jj].z; // 23
                    virial[4] += ders[jj*3 + 2] * relPos[jj].x; // 31
                    virial[5] += ders[jj*3 + 0] * relPos[jj].y; // 12
                }
            }
        }
    }
}

// ── Binning for fast neighbor list construction ────────────────────────

void MTPCore::buildBins(int nAtoms, const double *coords, double binSize) const {
    if (nAtoms == 0) return;

    // Find bounding box
    m_bboxMin[0] = m_bboxMin[1] = m_bboxMin[2] = 1e100;
    m_bboxMax[0] = m_bboxMax[1] = m_bboxMax[2] = -1e100;
    for (int i = 0; i < nAtoms; ++i) {
        for (int a = 0; a < 3; ++a) {
            double v = coords[3*i + a];
            if (v < m_bboxMin[a]) m_bboxMin[a] = v;
            if (v > m_bboxMax[a]) m_bboxMax[a] = v;
        }
    }
    // Pad to ensure atoms exactly at max still fit
    double eps = 1e-10;
    for (int a = 0; a < 3; ++a) {
        m_bboxMin[a] -= eps;
        m_bboxMax[a] += eps + binSize;
    }

    m_binSize = binSize;
    m_binNx = std::max(1, (int)((m_bboxMax[0] - m_bboxMin[0]) / binSize));
    m_binNy = std::max(1, (int)((m_bboxMax[1] - m_bboxMin[1]) / binSize));
    m_binNz = std::max(1, (int)((m_bboxMax[2] - m_bboxMin[2]) / binSize));
    int totalBins = m_binNx * m_binNy * m_binNz;

    m_binHead.assign(totalBins, -1);
    m_binNext.resize(nAtoms);
    m_binCells.resize(nAtoms * 3);

    for (int i = 0; i < nAtoms; ++i) {
        int ix = (int)((coords[3*i+0] - m_bboxMin[0]) / binSize);
        int iy = (int)((coords[3*i+1] - m_bboxMin[1]) / binSize);
        int iz = (int)((coords[3*i+2] - m_bboxMin[2]) / binSize);
        ix = std::max(0, std::min(ix, m_binNx-1));
        iy = std::max(0, std::min(iy, m_binNy-1));
        iz = std::max(0, std::min(iz, m_binNz-1));
        int binIdx = (iz * m_binNy + iy) * m_binNx + ix;
        m_binCells[3*i+0] = ix;
        m_binCells[3*i+1] = iy;
        m_binCells[3*i+2] = iz;
        m_binNext[i] = m_binHead[binIdx];
        m_binHead[binIdx] = i;
    }
}

// ── KIM GetNeighborList callback ───────────────────────────────────────

int MTPCore::getNbhCallback(void *data, int /*nLists*/,
                              const double *cutoffs_,
                              int /*listIdx*/, int particleNum,
                              int *nNeigh, int const **neighbors) {
    auto *ctx = static_cast<NbhCallback *>(data);
    double cutoff = cutoffs_ ? cutoffs_[0] : ctx->core->max_cutoff;

    double px = ctx->coords[3 * particleNum];
    double py = ctx->coords[3 * particleNum + 1];
    double pz = ctx->coords[3 * particleNum + 2];
    int itype = ctx->speciesCodes[particleNum];
    int sp = ctx->core->speciesCount();

    ctx->resultBuf.clear();
    for (int j = 0; j < ctx->nAtoms; ++j) {
        if (j == particleNum) continue;
        double dx = ctx->coords[3*j] - px;
        double dy = ctx->coords[3*j+1] - py;
        double dz = ctx->coords[3*j+2] - pz;
        double r2 = dx*dx + dy*dy + dz*dz;
        double r = std::sqrt(r2);
        int jtype = ctx->speciesCodes[j];
        if (r < ctx->core->cutoffs[itype * sp + jtype] && r < cutoff)
            ctx->resultBuf.push_back(j);
    }

    *nNeigh = (int)ctx->resultBuf.size();
    *neighbors = ctx->resultBuf.data();
    return false; // no error
}

} // namespace mtp_kim
