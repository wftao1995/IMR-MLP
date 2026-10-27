// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include <catch2/catch_all.hpp>
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtp_m.h"
#include "utils/cmdopts.h"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

using namespace Potentials;
using Catch::Approx;
using QuadraticMTP = PotMTP_M<m_model_mtpq2>;

namespace {
struct OneThread {
    int previous = get_omp_max_threads();
    int previous_global = omp_num_threads;
    OneThread() { omp_num_threads = 1; global_omp_init(); }
    ~OneThread() { set_omp_num_threads(previous); omp_num_threads = previous_global; }
};

std::string source(int species = 2, bool split = false, int basis_size = 2,
                   const std::string &basis = "RBChebyshev", int scalars = 3) {
    std::ostringstream out;
    out << "MTP\nversion = 1.1.0\nscaling = 1\nspecies_count = " << species << "\n";
    const auto radial = [&] {
        out << "radial_basis_type = " << basis << "\nmin_dist = 0.5\nmax_dist = 4.5\nradial_basis_size = "
            << basis_size << "\n";
        if (basis == "RBChebyshevPoly5") out << "rb_b = 0.5\n";
    };
    if (split) {
        out << "radial_basis_split = 1\n";
        for (int i = 0; i < species; ++i) for (int j = 0; j < species; ++j) {
            out << "pair " << i << ' ' << j << "\n";
            radial();
        }
    } else radial();
    out << "radial_funcs_count = 1\nradial_coeffs\n";
    for (int i = 0; i < species; ++i) for (int j = 0; j < species; ++j) {
        out << i << '-' << j << "\n{";
        for (int k = 0; k < basis_size; ++k)
            out << (k ? ", " : "") << .02 * (i + 1) / (j + k + 1);
        out << "}\n";
    }
    out << "alpha_moments_count = 6\nalpha_index_basic_count = 4\n"
        << "alpha_index_basic = {{0, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}}\n"
        << "alpha_index_times_count = 4\n"
        << "alpha_index_times = {{1, 1, 1, 4}, {2, 2, 1, 4}, {3, 3, 1, 4}, {0, 0, 1, 5}}\n"
        << "alpha_scalar_moments = " << scalars << "\nalpha_moment_mapping = {0, 4"
        << (scalars == 3 ? ", 5" : "") << "}\nspecies_coeffs = {";
    for (int i = 0; i < species; ++i) out << (i ? ", " : "") << -.2 * (i + 1);
    out << "}\nmoment_coeffs = {-0.15, 0.004" << (scalars == 3 ? ", 0.005" : "") << "}\n";
    return out.str();
}

void load(PotMTP &p, const std::string &text) {
    std::istringstream in(text);
    p.Load(in);
}

Atoms atoms(int species) {
    Atoms a;
    a.cell_ << 10., 0, 0, .3, 10.5, 0, .2, -.1, 11.;
    a.pos_ = {{2., 2., 2.}, {4.1, 2.2, 2.}, {2.3, 4.2, 2.4}, {5., 4.8, 3.1}};
    for (int i = 0; i < a.size(); ++i) a.chemmap_.push_back(i % species);
    return a;
}

void same_efs(PotMTP &p, PotMTP &reference, bool omp = false) {
    auto a = atoms(reference.species_count), b = a;
    reference.getEFS(b);
    if (omp) p.getEFSOMP(a); else p.getEFS(a);
    REQUIRE(std::isfinite(a.energy_));
    REQUIRE(a.energy_ == Approx(b.energy_).margin(1e-11));
    REQUIRE(a.stress_.allFinite());
    REQUIRE((a.stress_ - b.stress_).cwiseAbs().maxCoeff() < 1e-10);
    REQUIRE(a.forces_.size() == b.forces_.size());
    for (int i = 0; i < a.size(); ++i) {
        REQUIRE(a.forces_[i].allFinite());
        REQUIRE((a.forces_[i] - b.forces_[i]).cwiseAbs().maxCoeff() < 1e-10);
    }
}

void unloaded(PotMTP &p) {
    REQUIRE(p.species_count == 0);
    REQUIRE(p.reg_coeff_size == 0);
    REQUIRE(p.radial_coeff_size == 0);
    REQUIRE(p.alpha_scalar_moments == 0);
    REQUIRE(p.rb_size == 0);
    REQUIRE(p.regression_coeffs == nullptr);
    REQUIRE(p.radial_coeffs == nullptr);
    REQUIRE(p.species_coeffs == nullptr);
    REQUIRE(p.moment_coeffs == nullptr);
    REQUIRE(p.linear_coeffs == nullptr);
    REQUIRE(p.nu_arr_list == nullptr);
    REQUIRE(p.alpha_moment_mapping == nullptr);
    REQUIRE(p.p_radical_basis == nullptr);
    REQUIRE(p.p_radical_basis_pair.empty());
    REQUIRE(p.omp_buffers_list.empty());
    REQUIRE_FALSE(p.omp_inited);
    REQUIRE_FALSE(p.nl_inited);
    REQUIRE(p.empty);
    auto a = atoms(1);
    REQUIRE_THROWS_WITH(p.getEFS(a), Catch::Matchers::ContainsSubstring("no loaded model"));
    REQUIRE_THROWS_WITH(p.getEFSOMP(a), Catch::Matchers::ContainsSubstring("no loaded model"));
    std::ostringstream saved;
    REQUIRE_THROWS(p.Save(saved));
}

std::string replace_once(std::string text, const std::string &old, const std::string &replacement) {
    const auto at = text.find(old);
    REQUIRE(at != std::string::npos);
    text.replace(at, old.size(), replacement);
    return text;
}

std::string les_source(bool sog, int channels = 1) {
    PotMTP_LES p;
    load(p, source(2, sog, sog ? 3 : 2));
    if (sog) { p.les_kernel = "sog"; p.sog_terms = 2; p.sog_channels = channels; }
    p.dl = 3.;
    p.train_init();
    for (int i = 0; i < p.latent_species_count(); ++i) p.les_species_coeffs[i] = .03 * (i + 1);
    for (int i = 0; i < p.latent_moment_count(); ++i) p.les_moment_coeffs[i] = .004 * (i + 1);
    std::ostringstream out;
    p.Save(out);
    return out.str();
}

std::string quadratic_source(int species, bool split, bool sog = false) {
    auto text = source(species, split, split ? 3 : 2);
    text.resize(text.find("species_coeffs"));
    QuadraticMTP p;
    if (sog) {
        SOGModel definition;
        definition.channels = 2;
        definition.dl = 3.;
        definition.weights = {1., .2, .5, -.1};
        definition.log_bandwidths = {-.4, .2, -.3, .1};
        p.set_sog_definition(definition);
        text += "les_species_coeffs = {";
        for (int i = 0; i < 2 * species; ++i) text += (i ? ", " : "") + std::to_string(.03 * (i + 1));
        text += "}\nles_moment_coeffs = {0.01, 0.02, 0.03, 0.02, 0.01, 0.03}\n";
    }
    load(p, text);
    for (int i = 0; i < p.model_coeff_size; ++i) p.model_coeffs[i] = .1 * (i + 1);
    std::ostringstream out;
    p.Save(out);
    return out.str();
}

struct File {
    std::filesystem::path path;
    explicit File(const std::string &text) {
        static int serial = 0;
        path = std::filesystem::temp_directory_path() /
            ("imrmlp-load-ownership-" + std::to_string(getpid()) + "-" + std::to_string(serial++) + ".mtp");
        std::ofstream out(path);
        out << text;
        if (!out) throw std::runtime_error("Cannot write MTP ownership test fixture");
    }
    ~File() { std::error_code error; std::filesystem::remove(path, error); }
};
}

TEST_CASE("MTP repeated loads replace normal and split bases with new dimensions", "[mtp][load][ownership]") {
    OneThread thread;
    const std::vector<std::string> texts{source(), source(3, true, 3, "RBChebyshevPoly5", 2), source(1)};
    PotMTP p;
    for (int round = 0; round < 3; ++round) for (const auto &text : texts) {
        PotMTP reference;
        load(reference, text);
        load(p, text);
        REQUIRE(p.get_coeff_size() == reference.get_coeff_size());
        REQUIRE(p.rb_size == reference.rb_size);
        REQUIRE(p.radial_basis_split == reference.radial_basis_split);
        REQUIRE(p.p_radical_basis_pair.size() == reference.p_radical_basis_pair.size());
        REQUIRE((p.p_radical_basis == nullptr) == p.radial_basis_split);
        same_efs(p, reference);
        same_efs(p, reference, true);
    }
}

TEST_CASE("MTP failed loads release every allocation and permit a subsequent valid load", "[mtp][load][ownership]") {
    OneThread thread;
    const auto good = source(2, true, 3);
    const std::vector<std::string> bad{
        "", "MTP\n", good.substr(0, good.find("alpha_moments_count")),
        good.substr(0, good.find("alpha_moment_mapping")),
        replace_once(good, "pair 0 1", "pair 0 0"),
        replace_once(good, "pair 0 1", "pair 9 0"),
        replace_once(good, "species_coeffs = {-0.2, -0.4}", "species_coeffs = {broken}"),
        replace_once(good, "moment_coeffs = {-0.15, 0.004, 0.005}", "moment_coeffs = {broken}")};
    PotMTP reference, p;
    load(reference, good);
    for (const auto &text : bad) {
        load(p, good);
        p.omp_init();
        REQUIRE_THROWS(load(p, text));
        unloaded(p);
        load(p, good);
        same_efs(p, reference);
    }
    File invalid(bad[2]);
    REQUIRE_THROWS(PotMTP(invalid.path.string()));
    REQUIRE_THROWS(p.Load(invalid.path.string() + ".missing"));
    unloaded(p);
}

TEST_CASE("MTP loading and destruction never free borrowed coefficient storage", "[mtp][load][ownership]") {
    OneThread thread;
    const auto good = source();
    for (int action = 0; action < 3; ++action) for (bool detached_regression : {false, true}) {
        std::vector<double> external;
        {
            PotMTP p;
            load(p, good);
            external.assign(p.regression_coeffs, p.regression_coeffs + p.reg_coeff_size);
            delete[] p.regression_coeffs;
            p.regression_coeffs = external.data();
            p.update_layout();
            p.coeffs_handled_external = true;
            // Hybrid potentials may expose only the individual coefficient slices.
            if (detached_regression) p.regression_coeffs = nullptr;
            const auto before = external;
            if (action == 0) {
                load(p, source(3, true, 3));
                REQUIRE_FALSE(p.coeffs_handled_external);
                REQUIRE(p.regression_coeffs != external.data());
            } else if (action == 1) {
                REQUIRE_THROWS(load(p, good.substr(0, good.find("alpha_moments_count"))));
                unloaded(p);
            }
            REQUIRE(external == before);
        }
        REQUIRE_FALSE(external.empty());
        external.front() = 17.; // ASan diagnoses an invalid free/use-after-free here.
        REQUIRE(external.front() == 17.);
    }
}

TEST_CASE("LES and SOG reloads clean core resources and rejected latent heads", "[mtp][les][sog][load][ownership]") {
    OneThread thread;
    const auto coulomb = les_source(false), sog = les_source(true, 2);
    const std::vector<std::string> texts{coulomb, source(3, true, 3), sog, source(), coulomb};
    PotMTP_LES p;
    for (const auto &text : texts) {
        PotMTP_LES reference;
        load(reference, text);
        load(p, text);
        REQUIRE(p.lr_les == reference.lr_les);
        REQUIRE(p.channels() == reference.channels());
        same_efs(p, reference);
    }
    const auto malformed = replace_once(sog, "les_moment_coeffs = {", "les_moment_coeffs = {broken,");
    REQUIRE_THROWS(load(p, malformed));
    unloaded(p);
    REQUIRE(p.les_species_coeffs == nullptr);
    REQUIRE(p.les_moment_coeffs == nullptr);
    REQUIRE_FALSE(p.lr_les);
    REQUIRE(p.get_NL_coeffs_count() == 0);
    load(p, coulomb);
    PotMTP_LES reference;
    load(reference, coulomb);
    same_efs(p, reference);
    File file(malformed);
    REQUIRE_THROWS(PotMTP_LES(file.path.string()));
    // File-open failures in the inherited filename overload must also clear
    // the derived head, even though its stream parser is never entered.
    REQUIRE_THROWS(static_cast<PotMTP &>(p).Load(file.path.string() + ".missing"));
    unloaded(p);
    REQUIRE(p.les_species_coeffs == nullptr);
    REQUIRE(p.les_moment_coeffs == nullptr);
    REQUIRE(p.get_NL_coeffs_count() == 0);
}

TEST_CASE("MTP M reloads clean radial staging and reject both metadata conflicts", "[mtp][mtp_m][load][ownership]") {
    OneThread thread;
    const auto good = quadratic_source(2, false);
    QuadraticMTP p, reference;
    load(reference, good);
    for (const auto &text : {good, quadratic_source(3, true), quadratic_source(2, true, true), good}) {
        QuadraticMTP fresh;
        load(fresh, text);
        load(p, text);
        REQUIRE(p.model.coeffs == p.model_coeffs);
        same_efs(p, fresh);
    }
    const std::vector<std::pair<std::string, std::string>> bad{
        {replace_once(replace_once(good, "\"q\":2", "\"q\":3"),
                      "# mtp_descriptor_count = 2", "# mtp_descriptor_count = 3"),
         "MTP metadata: analytic head requires q=2"},
        {replace_once(good, "# mtp_descriptor_count = 2", "# mtp_descriptor_count = 3"),
         "MTP metadata: conflicting JSON/footer metadata for mtp_descriptor_count"},
        {replace_once(good, "species_coeffs = {", "species_coeffs = {broken,"), ""}};
    for (const auto &[text, message] : bad) {
        load(p, good);
        if (message.empty()) REQUIRE_THROWS(load(p, text));
        else REQUIRE_THROWS_WITH(load(p, text), message);
        unloaded(p);
        REQUIRE(p.model_coeffs == nullptr);
        REQUIRE(p.model.coeffs == nullptr);
        REQUIRE(p.les_species == nullptr);
        REQUIRE(p.les_moment == nullptr);
        REQUIRE(p.les_coeffs == nullptr);
        REQUIRE(p.get_NL_coeffs_count() == 0);
        load(p, good);
        same_efs(p, reference);
        // Also exercise failure while the enclosing constructor is unwinding.
        File file(text);
        if (message.empty()) REQUIRE_THROWS(QuadraticMTP(file.path.string()));
        else REQUIRE_THROWS_WITH(QuadraticMTP(file.path.string()), message);
    }
}

TEST_CASE("MTP load cleanup preserves valid coefficient templates and zero-basic models", "[mtp][load][ownership]") {
    OneThread thread;
    auto graph = source();
    graph.resize(graph.find("species_coeffs"));
    PotMTP p;
    load(p, graph);
    REQUIRE(p.empty);
    REQUIRE(p.regression_coeffs != nullptr);
    auto a = atoms(2);
    REQUIRE_NOTHROW(p.getEFS(a));
    REQUIRE(std::isfinite(a.energy_));
    File file(graph);
    GPSettings settings;
    PotMTP_LES les(file.path.string(), settings);
    REQUIRE(les.les_species_coeffs != nullptr);
    REQUIRE_NOTHROW(les.getEFS(a));
    settings.Modify("enable_indirect_force", "false");
    REQUIRE_THROWS(PotMTP_LES(file.path.string(), settings));

    const std::string zero =
        "MTP\nversion = 1.1.0\nspecies_count = 1\nradial_basis_type = RBChebyshev\n"
        "min_dist = 0.5\nmax_dist = 4.5\nradial_basis_size = 2\nradial_funcs_count = 0\n"
        "alpha_moments_count = 0\nalpha_index_basic_count = 0\nalpha_index_basic = {}\n"
        "alpha_index_times_count = 0\nalpha_index_times = {}\nalpha_scalar_moments = 0\n"
        "alpha_moment_mapping = {}\nspecies_coeffs = {-0.25}\nmoment_coeffs = {}\n";
    for (int round = 0; round < 2; ++round) {
        load(p, zero);
        REQUIRE(p.radial_coeff_size == 0);
        REQUIRE(p.alpha_index_basic_count == 0);
        REQUIRE(p.reg_coeff_size == 1);
        auto constant = atoms(1);
        p.getEFS(constant);
        REQUIRE(constant.energy_ == Approx(-.25 * constant.size()));
        REQUIRE(constant.stress_.norm() == 0);
        for (const auto &force : constant.forces_) REQUIRE(force.norm() == 0);
        std::ostringstream saved;
        p.Save(saved);
        REQUIRE(saved.str().find("alpha_moment_mapping = {}\n") != std::string::npos);
        PotMTP restored;
        load(restored, saved.str());
        same_efs(restored, p);
        same_efs(restored, p, true);
        PotMTP copied(p);
        same_efs(copied, p);
        same_efs(copied, p, true);
        load(p, graph);
    }
}
