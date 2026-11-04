// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_mtp_m.h"

namespace Potentials
{

template <class T> PotMTP_M<T>::PotMTP_M(PotMTP_M<T> &source) : PotMTP(source)
{
    InitSettings();
    ApplySettings(source.CurrentSettings());
    n_props = source.n_props;
    model_coeff_size = source.model_coeff_size;
    req_radical = source.req_radical;
    req_species = source.req_species;
    req_moment = source.req_moment;
    req_model = source.req_model;
    req_les = source.req_les;
    req_sog = source.req_sog;
    lr_les = source.lr_les;
    model_les = source.model_les;
    if constexpr (std::is_same_v<T, m_model_nn1>)
    {
        model.n_input = source.model.n_input;
        model.n_neuron = source.model.n_neuron;
        model.acfunc_enum = source.model.acfunc_enum;
        GPSettings settings;
        model.init(settings);
    }
    else
        model = source.model;
    ext_dUdQ.resize(n_props);
    ext_dUdt.resize(model_coeff_size);
    ext_d2UdQdQ.resize(n_props, n_props);
    ext_d2UdQdt.resize(n_props, model_coeff_size);
    init_coeffs();
    empty = source.empty;
    nl_inited = source.nl_inited;
}

template PotMTP_M<m_model_mtpfs>::PotMTP_M(PotMTP_M<m_model_mtpfs> &);
template PotMTP_M<m_model_mtpq2>::PotMTP_M(PotMTP_M<m_model_mtpq2> &);
template PotMTP_M<m_model_mtpli>::PotMTP_M(PotMTP_M<m_model_mtpli> &);
template PotMTP_M<m_model_nn1>::PotMTP_M(PotMTP_M<m_model_nn1> &);
template PotMTP_M<m_model_nn2>::PotMTP_M(PotMTP_M<m_model_nn2> &);

#define EXTMODEL_TYPES                                                                             \
    X(mtp_m_fs, m_model_mtpfs)                                                                     \
    X(mtp_m_q2, m_model_mtpq2)                                                                     \
    X(mtp_m_li, m_model_mtpli)                                                                     \
    X(mtp_m_n1, m_model_nn1)                                                                       \
    X(mtp_m_n2, m_model_nn2)                                                                       \
    X(mtp_m_nn2, m_model_nn2)

// #define CREATE_POT_CASE(T, spl, safe)                           \
//     case ((spl) | ((safe) << 1)): \
//         return new PotMTP_M<T, spl, safe>(potfile, settings);

// #define CREATE_POT_CASE(T)                           \
//     return new PotMTP_M<T>(potfile, settings);

// #define GENERATE_SWITCH_CASES(T)      \
//     CREATE_POT_CASE(T, 1, 0) \
//     CREATE_POT_CASE(T, 0, 1) \
//     CREATE_POT_CASE(T, 1, 1) \
//     CREATE_POT_CASE(T, 0, 0)

AnyPotential *create_mtp_m_potential(GPSettings &settings)
{
    std::string potfile = settings["potfile"];
    std::string extmodel = settings["extmodel"];
    auto stream = open_ifile(potfile);
    if (!stream)
        GPERROR("Cannot read potential metadata");
    const auto header = MTPMetadata::read_header(*stream);
    if (header.tag.text("type") == "MTP_M")
    {
        auto declared = MTPMetadata::canonical_model(header.tag.text("model"));
        if (!extmodel.empty() && MTPMetadata::canonical_model(extmodel) != declared)
            GPERROR("External model setting conflicts with potential_tag");
        extmodel = declared;
        settings.Modify("extmodel", declared);
    }

    int req_safe = 0;
    int spl_cutoff = 0;
    if (settings.count("req_safe") > 0)
        req_safe = stoi(settings["req_safe"]);
    if (settings.count("spl_cutoff") > 0)
        spl_cutoff = stoi(settings["spl_cutoff"]);

    // int case_key = spl_cutoff | (req_safe << 1) | (req_mom << 2) | (req_mod << 3) | (req_safe << 4);
    int case_key = spl_cutoff | (req_safe << 1);

    // #define X(model_name, model_sig)                                       \
    //     if (extmodel == #model_name) {                                     \
    //         switch (case_key) {                                            \
    //             GENERATE_SWITCH_CASES(model_sig)                           \
    //         default:                                                       \
    //             throw std::runtime_error("Invalid parameter combination"); \
    //         }                                                              \
    //     }

#define X(model_name, model_sig)                                                                   \
    if (extmodel == #model_name)                                                                   \
    {                                                                                              \
        return new PotMTP_M<model_sig>(potfile, settings);                                         \
    }

    EXTMODEL_TYPES

#undef X

    throw std::runtime_error("Unknown extmodel type: " + extmodel);
}

#undef CREATE_POT_CASE

};
