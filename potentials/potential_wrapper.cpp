// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#include "potential_wrapper.h"
#include "model_bundle.h"
#include "tools/gpm_mtp.h"

// ── All concrete potential types (moved from header to break recompilation) ──
#include "potentials/potential_mtp.h"
#include "potentials/potential_eam_zhou_2004.h"
#include "potentials/potential_shell.h"
#include "potentials/potential_eamalloy_spline.h"
#include "potentials/potential_eamfs_spline.h"
#include "potentials/ace_lmp.h"
#include "potentials/potential_hybmtp.h"
#ifdef USE_KIM
#include "potentials/potential_openkim.h"
#endif
#include "potentials/potential_hybrid.h"
#include "potentials/potential_hybrid_scale.h"
#include "potentials/potential_mindist.h"
#include "potentials/potential_lj.h"
#include "potentials/potential_mtpnn1.h"
#include "potentials/potential_mtp_m.h"
#include "potentials/potential_eam_poly.h"
#include "potentials/potential_mtp_LES.h"
#include "potentials/potential_descriptor_linear.h"
#include "potentials/potential_zbl.h"
#include "potentials/potential_pair_table.h"
#include "potentials/potential_threebody_table.h"
#include "potentials/potential_sw.h"
#include "potentials/potential_tersoff.h"
#include "potentials/potential_settings.h"

#include <filesystem>
#include <fstream>
#include <cassert>
#include <typeinfo>
#include "io/gpio.h"
#include "tools/procstr.h"
#include "tools/chem.h"
#include "utils/compressed_stream.h"

namespace fs = std::filesystem;

namespace Potentials {

bool supports_frame_local_efs(const AnyPotential &pot) {
    const auto &type = typeid(pot);
    if (type == typeid(PotentialHybrid)) {
        const auto &children = static_cast<const PotentialHybrid &>(pot).ppots;
        return !children.empty() && std::all_of(children.begin(), children.end(),
            [](const auto &child) { return child && supports_frame_local_efs(*child); });
    }
    if (type == typeid(PotentialHybridScale)) {
        const auto &children = static_cast<const PotentialHybridScale &>(pot).ppots_;
        return !children.empty() && std::all_of(children.begin(), children.end(),
            [](const auto &child) { return child && supports_frame_local_efs(*child); });
    }
    return type == typeid(PotMTP) || type == typeid(PotMTPNN1)
        || type == typeid(PotMTP_M<m_model_mtpli>)
        || type == typeid(PotMTP_M<m_model_mtpq2>)
        || type == typeid(PotMTP_M<m_model_mtpfs>)
        || type == typeid(PotMTP_M<m_model_nn1>)
        || type == typeid(PotMTP_M<m_model_nn2>)
        || type == typeid(PotMTP_LES)
        || type == typeid(PotentialDescriptorLinear) || type == typeid(LJ612)
        || type == typeid(PotentialZBL) || type == typeid(SW)
        || type == typeid(Tersoff) || type == typeid(PairTable)
        || type == typeid(ThreebodyTable) || type == typeid(PotMDIST);
}

void potential_wrapper::read_map(std::string file,
                                 std::map<int, std::string> &mi2s,
                                 std::map<std::string, int> &ms2i) {
    auto ifs = open_ifile(file);
    if (!ifs || !*ifs)
        throw std::runtime_error("Cannot open species map: " + file);
    int id = 0;
    std::string buffer;
    while (getline(*ifs, buffer)) {
        std::vector<std::string> splbuf;
        str_split(buffer, splbuf);
        if (splbuf.size() >= 1) {
            mi2s[id] = splbuf[0];
            ms2i[splbuf[0]] = id;
            ++id;
        }
    }
    if (ifs->bad()) throw std::runtime_error("Failed to read species map: " + file);
}

void potential_wrapper::init(GPSettings &settings) {
    InitSettings();
    ApplySettings(settings);
    if (GPM::is_file(potfile) || GPM::extension(potfile)) {
        auto bundle = ModelBundle::load(potfile);
        const auto actual = ModelBundle::type(bundle->potential());
        auto requested = settings["pottype"];
        GPM::require(requested.empty() || requested == "auto" || requested == actual,
                     "explicit potential type conflicts with model");
        auto effective=bundle->potential().CurrentSettings();
        const auto &manifest=bundle->manifest();const auto &node=GPM::object(GPM::at(GPM::object(GPM::at(manifest,"models")),GPM::string(GPM::at(manifest,"root_model"))));
        if(node.count("configuration"))for(const auto &[key,value]:GPM::object(node.at("configuration"))) {
            if(value.is<std::string>())effective.Modify(key,value.get<std::string>());
            else if(value.is<GPM::Object>()&&value.get<GPM::Object>().count("block")){auto numbers=GPM::values(*bundle->source(),value);std::ostringstream text;text<<std::setprecision(17);for(double n:numbers)text<<n<<' ';effective.Modify(key,GPM::trim(text.str()));}
        }
        if(auto *mtp=dynamic_cast<PotMTP*>(&bundle->potential())){
            auto meta=mtp->potential_tag.output(mtp->model_metadata(),mtp->species_count);
            if(actual=="mtp_m")effective.Modify("extmodel",GPM::string(GPM::at(meta,"model")));
            effective.Modify("lr_les",meta.count("long_range")?"true":"false");
        }
        auto normalized=[](const std::string &key,std::string value){
            if(key=="extmodel")value=MTPMetadata::canonical_model(value);
            if(key=="activation_function")for(char &c:value)c=std::toupper(static_cast<unsigned char>(c));
            if(value=="true")value="1";if(value=="false")value="0";
            std::istringstream input(value);std::vector<double> nums;double n;while(input>>n)nums.push_back(n);
            if(!nums.empty()&&input.eof()){std::ostringstream out;out<<std::setprecision(17);for(auto x:nums)out<<x<<' ';return out.str();}
            return GPM::trim(value);
        };
        for(const auto &[key,value]:settings.settings) {
            if(value.empty()||key=="potfile"||key=="pottype"||key=="typemap"||key=="potlevel")continue;
            GPM::require(effective.count(key)&&normalized(key,value)==normalized(key,effective[key]),"explicit model configuration conflicts with GPM: "+key);
        }
        if (settings.count("typemap") && !settings["typemap"].empty()) {
            std::map<int,std::string> mapping; std::map<std::string,int> reverse;
            read_map(settings["typemap"],mapping,reverse);
            auto &model = bundle->potential();
            GPM::require(model.species_count==0||mapping.size()==size_t(model.species_count),"element mapping dimensions conflict with model");
            for(const auto &[id,name]:model.gpm_elements)
                GPM::require(mapping.count(id)&&mapping.at(id)==name,"element mapping conflicts with model");
            model.set_element_map(mapping); model.gpm_elements = mapping;
        }
        pottype = actual; pot = bundle->release_potential().release();
        return;
    }
    if (pottype == "auto" || pottype.empty()) pottype = "mtp";
    std::map<std::string, int> mappingtype2id;
    std::map<int, std::string> mappingid2type;
    if (GPIO::file_exist(typemap)) {
        read_map(typemap, mappingid2type, mappingtype2id);
    } else {
        std::cerr << "Warning: index file not found" << std::endl;
    }
    if (pottype == "acelmp") {
        #ifdef WITH_ACELMP
        if(mappingid2type.empty())throw std::runtime_error("ACE requires an explicit ordered element map");
        std::vector<std::string> elements;for(const auto &[id,name]:mappingid2type)elements.push_back(name);
        pot = new PotentialACELmp(potfile,elements);
        #else
        GPERROR("WITH_ACELMP not defined");
        #endif
    } else if (pottype == "mtp") {
        assert(!potfile.empty());
        pot = new PotMTP(potfile);
    } else if (pottype == "mtpnn1") {
        if (potfile.empty())
            pot = new PotMTPNN1(settings);
        else
            pot = new PotMTPNN1(potfile, settings);
    } else if (pottype == "mtp_m") {
        assert(!potfile.empty());
        pot = create_mtp_m_potential(settings);
    } else if (pottype == "hmtp") {
        assert(!potfile.empty());
        pot = new PotMTPHyb3(potfile, plev);
    } else if (pottype == "shellfile") {
        pot = new PotShellCalc(settings.ExtractSubSettings("shellfile"));
    } else if (pottype == "eamalloy") {
        pot = new EAMAlloySpline(potfile, mappingtype2id);
    } else if (pottype == "eamfs") {
        pot = new EAMFsSpline(potfile, mappingtype2id);
    } else if (pottype == "openkim") {
        #ifdef USE_KIM
        std::string kmodel = settings["kim_model"];
        if (kmodel.empty() && !potfile.empty()) {
            kmodel = potfile;
        }
        if (kmodel.empty()) {
            GPERROR("openkim: kim_model (or potfile) must specify the KIM model name");
        }

        if (mappingid2type.empty()) {
            std::cout << "index.map not found, PotKIM will auto-detect species" << std::endl;
        } else {
            std::cout << "Read " << mappingid2type.size()
                      << " species from index.map" << std::endl;
        }

        auto kim = std::make_unique<PotKIM>(kmodel, mappingid2type);
        kim->ApplyParameterSettings(settings);
        pot = kim.release();
        #else
        GPERROR("USE_KIM not defined. Rebuild with KIM support enabled.");
        #endif
    } else if (pottype == "hybrid") {
        pot = new PotentialHybrid(settings);
    } else if (pottype == "hybrid_scale") {
        pot = new PotentialHybridScale(settings);
    } else if (pottype == "mindist" || pottype == "lj612" || pottype == "zbl") {
        GPSettings parameters = settings;
        if (!potfile.empty()) {
            auto stream = open_ifile(potfile);
            if (!stream) GPERROR("Cannot open potential file: " + potfile);
            parameters = read_potential_settings(*stream);
            // Explicit settings override parameters read from the file.
            for (const auto &[key, value] : settings.settings)
                if (!value.empty()) parameters.Modify(key, value);
        }
        if (pottype == "mindist") pot = new PotMDIST(parameters);
        else if (pottype == "lj612") pot = new LJ612(parameters);
        else pot = new PotentialZBL(parameters);
    } else if (pottype == "eampoly") {
        if (potfile.empty())
            pot = new PotentialEAMPoly(settings, mappingtype2id);
        else
            pot = new PotentialEAMPoly(potfile, mappingtype2id);
    } else if (pottype == "mtp_les") {
        pot = new PotMTP_LES(potfile, settings);
    } else if (pottype == "descriptor_linear") {
        pot = potfile.empty()
            ? new PotentialDescriptorLinear(settings.ExtractSubSettings("descriptor_linear"))
            : new PotentialDescriptorLinear(potfile);
    } else if (pottype == "pair_table") {
        pot = new PairTable(potfile, mappingtype2id);
    } else if (pottype == "threebody_table") {
        pot = new ThreebodyTable(potfile, mappingtype2id);
    } else if (pottype == "sw") {
        pot = new SW(potfile, mappingtype2id);
    } else if (pottype == "tersoff") {
        pot = new Tersoff(potfile, mappingtype2id);
    } else {
        std::cerr << "FORMAT IS " << pottype << std::endl;
        GPERROR("Unknown format");
    }
    // An IO map may describe only the species present in the input structure.
    // Attach identities only when it is a complete declaration of the model.
    if(!mappingid2type.empty() && (pot->species_count==0 || static_cast<int>(mappingid2type.size())==pot->species_count))
        { pot->set_element_map(mappingid2type); pot->gpm_elements = mappingid2type; }
    // post process, enable openmp
    pot->omp_init();
}

} // namespace Potentials
