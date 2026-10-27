// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_mtp_m.h"

namespace Potentials {

#define EXTMODEL_TYPES         \
    X(mtp_m_fs, m_model_mtpfs) \
    X(mtp_m_q2, m_model_mtpq2) \
    X(mtp_m_li, m_model_mtpli) \
    X(mtp_m_n1, m_model_nn1) \
    X(mtp_m_n2, m_model_nn2) \
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

AnyPotential* create_mtp_m_potential(GPSettings& settings)
{
    std::string potfile = settings["potfile"];
    std::string extmodel = settings["extmodel"];
    auto stream=open_ifile(potfile);if(!stream)GPERROR("Cannot read potential metadata");
    const auto header=MTPMetadata::read_header(*stream);
    if(header.tag.text("type")=="MTP_M") {
        auto declared=MTPMetadata::canonical_model(header.tag.text("model"));
        if(!extmodel.empty()&&MTPMetadata::canonical_model(extmodel)!=declared)GPERROR("External model setting conflicts with potential_tag");
        extmodel=declared;settings.Modify("extmodel",declared);
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

#define X(model_name, model_sig)                                       \
    if (extmodel == #model_name) {                                     \
        return new PotMTP_M<model_sig>(potfile, settings);            \
    }

    EXTMODEL_TYPES

#undef X

    throw std::runtime_error("Unknown extmodel type: " + extmodel);
}

#undef CREATE_POT_CASE

};
