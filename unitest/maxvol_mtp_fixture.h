#pragma once

#include "maxvol_base_fixture.h"
#include "potentials/potential_hybmtp.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_wrapper.h"
#include "potentials/sog_model.h"
#include <cmath>
#include <filesystem>
#include <fstream>

struct MaxVolMTPCase {
    std::string name, pottype, path;
    GPSettings settings;
};

// Small deterministic models, generated at test time without private datasets.
inline std::vector<MaxVolMTPCase> make_maxvol_mtp_models(const std::filesystem::path &dir)
{
    using namespace Potentials;
    std::filesystem::create_directories(dir);
    const auto base = (dir / "base.mtp").string();
    PotMTP reference;
    load_swap_fixture(reference, 2, false);
    reference.Save(base);
    std::ostringstream serialized;
    reference.Save(serialized);
    auto header = serialized.str();
    header.resize(header.find("species_coeffs"));
    const auto template_file = (dir / "template.mtp").string();
    std::ofstream(template_file) << header;
    std::vector<MaxVolMTPCase> cases;
    const auto add = [&](std::string name, std::string type, GPSettings settings = {}) {
        MaxVolMTPCase c{name, type, (dir / name).string(), settings};
        c.settings.Modify("pottype", type);
        c.settings.Modify("potfile", c.path);
        cases.push_back(c);
        return c.path;
    };
    for (int layers : {1, 2}) {
        GPSettings settings;
        settings.Modify("mtpcoeff", base);
        settings.Modify("nncoeff", (dir / "missing.nn").string());
        settings.Modify("spcoeff", (dir / "missing.sc").string());
        settings.Modify("nnlay", std::to_string(layers));
        settings.Modify("nnhidden", "3");
        settings.Modify("nnhidden2", "2");
        PotMTPNN1 model(settings);
        model.train_init();
        std::vector<double> coefficients(model.get_NL_coeffs_count());
        for (size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = .04 + .01 * std::cos(i);
        model.nl_set_coeffs(coefficients.data());
        model.Save(add("nn" + std::to_string(layers), "mtpnn1"));
    }
    for (const auto *extmodel : {"mtp_m_fs", "mtp_m_q2", "mtp_m_li", "mtp_m_n1", "mtp_m_n2"}) {
        GPSettings settings;
        settings.Modify("extmodel", extmodel);
        settings.Modify("n_input", "2");
        settings.Modify("n_neuron", "3");
        settings.Modify("n_neuron2", "2");
        settings.Modify("potfile", template_file);
        settings.Modify("pottype", "mtp_m");
        potential_wrapper model(settings);
        model.pot->train_init();
        std::vector<double> coefficients(model.pot->get_NL_coeffs_count());
        for (size_t i = 0; i < coefficients.size(); ++i) coefficients[i] = .06 + .01 * std::cos(i);
        model.pot->nl_set_coeffs(coefficients.data());
        model.pot->Save(add(extmodel, "mtp_m", settings));
    }
    for (bool sog : {false, true}) {
        PotMTP_LES model(base);
        model.lr_les = true;
        model.les_species_coeffs = new double[2]{.2, -.15};
        model.les_moment_coeffs = new double[3]{.05, -.02, .01};
        if (sog) {
            model.les_kernel = "sog";
            model.sog_terms = 2;
        }
        model.train_init();
        if (sog) {
            model.model_les.sog_weights = {1.2, -.3};
            model.model_les.sog_log_bandwidths = {-.4, .2};
        }
        const auto path = add(sog ? "sog" : "les", "mtp_les");
        std::ofstream out(path);
        model.Save(out);
    }
    for (bool sog : {false, true}) {
        const auto file = dir / "longrange-template.mtp";
        std::ofstream source(file);
        source << header << "les_species_coeffs = {0.2, -0.15}\n"
               << "les_moment_coeffs = {0.05, -0.02, 0.01}\n";
        if (sog) {
            SOGModel definition;
            definition.weights = {1.2, -.3};
            definition.log_bandwidths = {-.4, .2};
            definition.write(source);
        }
        source.close();
        GPSettings settings;
        settings.Modify("extmodel", "mtp_m_q2");
        settings.Modify("lr_les", "true");
        settings.Modify("pottype", "mtp_m");
        settings.Modify("potfile", file.string());
        potential_wrapper model(settings);
        model.pot->Save(add(sog ? "mtp_m_sog" : "mtp_m_les", "mtp_m", settings));
    }
    for (int count : {2, 3}) {
        PotMTP second, third;
        load_swap_fixture(second, 2, true);
        load_swap_fixture(third, 2, false);
        PotMTPHyb3 model(&reference, &second, count == 3 ? &third : nullptr);
        GPSettings settings;
        settings.Modify("potlevel", std::to_string(count));
        model.Save(add("hmtp" + std::to_string(count), "hmtp", settings));
    }
    return cases;
}

inline Atoms maxvol_mtp_atoms()
{
    Atoms cfg;
    cfg.cell_ = 12.0 * Eigen::Matrix3d::Identity();
    cfg.pos_ = {{2., 2., 2.}, {4.4, 2.6, 2.5}, {7.1, 2.8, 2.9}};
    cfg.chemmap_ = {1, 0, 1};
    return cfg;
}
