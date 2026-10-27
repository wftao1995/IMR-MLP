// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT
#include <catch2/catch_all.hpp>
#include "tools/mtp_metadata.h"
#include "LAMMPS/USER-MLIP/mlip_pair_metadata.h"
#include "potentials/potential_mtp_m.h"
#include <filesystem>
#include <fstream>
#include <sstream>
#include <unistd.h>

TEST_CASE("MTP tag parses bounded metadata and rejects ambiguous input", "[mtp][metadata]") {
    using namespace MTPMetadata;
    auto tag=parse_tag("original note | imrmlp_meta_v1:{\"model_type\":\"mtp\",\"elements\":[\"Al\",\"Ni\"],\"custom\":{\"note\":\"\\u91d1\\u5c5e\"}}",2);
    REQUIRE(tag.legacy=="original note");REQUIRE(tag.elements()==std::vector<std::string>{"Al","Ni"});
    REQUIRE(parse_tag(Value(tag.output({{"type",Value("MTP")},{"model",Value("linear")}},2)).serialize(),2).fields.count("custom")==1);
    for(const auto &bad:{"imrmlp_meta_v2:{}","imrmlp_meta_v1:{\"elements\":[\"Xx\"]}","imrmlp_meta_v1:{\"elements\":[\"Al\"]}","imrmlp_meta_v1:{\"model_type\":\"mtp\",\"model_type\":\"mtp_m\"}","imrmlp_meta_v1:{\"q\":{\"count\":2,\"names\":[\"q0\",\"q0\"]}}"})REQUIRE_THROWS(parse_tag(bad,2));
    REQUIRE_THROWS(parse_tag(std::string(max_bytes+1,'x')));
}
TEST_CASE("MTP element mappings retain legacy indices and resolve names in model order", "[mtp][metadata][lammps]") {
    using namespace MTPMetadata;Header h;h.species_count=2;h.tag=parse_tag("imrmlp_meta_v1:{\"model_type\":\"mtp\",\"elements\":[\"Al\",\"Ni\"]}",2);
    REQUIRE(mapping({"Ni","Al"},h,2,20)==std::vector<int>{1,0});
    REQUIRE(mapping({"1","0"},h,2,20)==std::vector<int>{1,0});
    REQUIRE(mapping({"Ni","Ni"},h,2,20)==std::vector<int>{1,1});
    for(const auto &args:std::vector<std::vector<std::string>>{{"Ni"},{"Ni","Fe"},{"0","Ni"},{"2","0"},{"-1","0"},{"0.5","1"}})REQUIRE_THROWS(mapping(args,h,2,20));
    h.tag={};REQUIRE(mapping({"0","1"},h,2,20)==std::vector<int>{0,1});REQUIRE_THROWS(mapping({"Al","Ni"},h,2,20));
}
TEST_CASE("MTP save and species permutation preserve chemistry and user metadata", "[mtp][metadata][roundtrip]") {
    Potentials::PotMTP pot("unitest/test-data/pot4b.mtp");pot.set_element_symbols({"Al","Ni"});
    pot.potential_tag.legacy="fixture note";pot.potential_tag.fields["custom"]=MTPMetadata::Value("keep me");
    std::ostringstream out;pot.Save(out);Potentials::PotMTP restored;std::istringstream in(out.str());restored.Load(in);
    REQUIRE(restored.element_symbols()==std::vector<std::string>{"Al","Ni"});REQUIRE(restored.potential_tag.legacy=="fixture note");
    REQUIRE(restored.potential_tag.text("custom")=="keep me");
    REQUIRE(restored.reg_coeff_size==pot.reg_coeff_size);for(int i=0;i<pot.reg_coeff_size;++i)REQUIRE(restored.regression_coeffs[i]==pot.regression_coeffs[i]);
    restored.adjust_species({1,0});REQUIRE(restored.element_symbols()==std::vector<std::string>{"Ni","Al"});
    REQUIRE_THROWS(restored.set_element_symbols({"Al","Ni"}));REQUIRE_THROWS(restored.adjust_species({0,1,2}));
}
TEST_CASE("MTP-M tag reconstructs NN layout without external settings", "[mtp][metadata][mtp_m]") {
    const auto dir=std::filesystem::temp_directory_path()/("imrmlp-metadata-"+std::to_string(getpid()));std::filesystem::create_directories(dir);
    const auto basis=(dir/"basis.mtp").string(),fitted=(dir/"fitted.mtp").string();
    {std::ifstream src("unitest/test-data/pot4b.mtp");std::ofstream dst(basis);std::string line;while(std::getline(src,line)){if(line.rfind("species_coeffs",0)==0)break;dst<<line<<'\n';}}
    GPSettings settings;settings.Modify("n_input","3");settings.Modify("n_neuron","4");settings.Modify("n_neuron2","2");settings.Modify("activation_function","TANH");
    Potentials::PotMTP_M<Potentials::m_model_nn2> pot(basis,settings);pot.set_element_symbols({"Al","Ni"});pot.Save(fitted);
    auto h=MTPMetadata::read_header(fitted);REQUIRE(h.tag.text("type")=="MTP_M");
    GPSettings automatic;automatic.Modify("potfile",fitted);std::unique_ptr<Potentials::AnyPotential> loaded(Potentials::create_mtp_m_potential(automatic));
    auto *model=dynamic_cast<Potentials::PotMTP_M<Potentials::m_model_nn2>*>(loaded.get());REQUIRE(model!=nullptr);REQUIRE(model->n_props==3);REQUIRE(model->model.n_neuron==4);REQUIRE(model->model.n_neuron2==2);
    REQUIRE(model->element_symbols()==std::vector<std::string>{"Al","Ni"});
    automatic.Modify("n_input","2");REQUIRE_THROWS(Potentials::create_mtp_m_potential(automatic));
    automatic.Modify("n_input","3");automatic.Modify("n_neuron1","2");REQUIRE_THROWS(Potentials::create_mtp_m_potential(automatic));
    GPSettings fresh;fresh.Modify("potfile",fitted);
    {std::ifstream input(fitted);std::string text((std::istreambuf_iterator<char>(input)),{});const std::string key="# mtp_hidden1 = 4";auto pos=text.find(key);REQUIRE(pos!=std::string::npos);text.replace(pos,key.size(),"# mtp_hidden1 = 5");std::ofstream output(fitted);output<<text;}
    REQUIRE_THROWS(Potentials::create_mtp_m_potential(fresh));
    std::filesystem::remove_all(dir);
}

namespace {
using namespace MTPMetadata;
const std::string tag_json=R"({"schema":"gpml.mtp.metadata","version":1,"type":"MTP","model":"linear","elements":["Al","Ni"]})";
}
TEST_CASE("MTP JSON metadata validates types and preserves extensions", "[mtp][metadata]") {
    auto tag = Tag::parse(tag_json, 2);
    REQUIRE(tag.elements() == std::vector<std::string>{"Al", "Ni"});
    REQUIRE(Tag::parse("", 2).elements().empty());
    REQUIRE(Tag::parse("= old free text", 2).legacy == "old free text");
    REQUIRE(Tag::parse(R"({"other":"format"})", 2).legacy == R"({"other":"format"})");
    REQUIRE_THROWS(Tag::parse(tag_json, 3));
    for (const auto &suffix : {" garbage", ","}) REQUIRE_THROWS(Tag::parse(tag_json + suffix, 2));
    auto invalid = tag_json;
    invalid.pop_back();
    invalid += R"(,"version":1})";
    REQUIRE_THROWS(Tag::parse(invalid, 2));
    REQUIRE_THROWS(Tag::parse(tag_json.substr(0, tag_json.size() - 1), 2));
    for (const auto &bad : {Value("1"), Value(1.5), Value(0.0), Value(2.0)}) {
        auto fields = tag.fields;
        fields["version"] = bad;
        REQUIRE_THROWS(Tag::parse(Value(fields).serialize(), 2));
    }
    tag.fields["extension"] = Value(Object{{"label", Value("literal imrmlp_meta_v1: quote\" newline\n Unicode \xe4\xb8\xad")},
        {"enabled", Value(true)}, {"array", Value(Array{Value(), Value(1.25)})}});
    auto saved = Value(tag.output({{"type", Value("MTP")}, {"model", Value("linear")}}, 2)).serialize();
    REQUIRE(saved.find('\n') == std::string::npos);
    REQUIRE(Tag::parse(saved, 2).fields.at("extension") == tag.fields.at("extension"));
    REQUIRE_THROWS(tag.set_elements({"Ni", "Al"}, 2));
    REQUIRE_THROWS(Tag{}.set_elements({"Al", "Al"}, 2));
    REQUIRE_THROWS(Tag{}.set_elements({"Al"}, 2));
}

TEST_CASE("LAMMPS element mapping is ordered and validates all types", "[mtp][metadata][lammps]") {
    Header header{2, Tag::parse(tag_json, 2)};
    REQUIRE(map_types(header, {"Ni", "Al"}) == std::vector<int>{1, 0});
    REQUIRE(map_types(header, {"Ni", "Ni", "Al"}) == std::vector<int>{1, 1, 0});
    REQUIRE(map_types(header, {"1", "0"}) == std::vector<int>{1, 0});
    for (const auto &bad : {std::vector<std::string>{"Ni", "Fe"}, {"0", "Ni"}, {"2", "0"}, {"NULL", "Al"}, {"-1", "0"}})
        REQUIRE_THROWS(map_types(header, bad));
    REQUIRE_THROWS(map_types(Header{2, {}}, {"Ni", "Al"}));
    REQUIRE(map_types(Header{2, {}}, {"1", "0"}) == std::vector<int>{1, 0});
    PairArguments command;
    command.loaded = true;
    command.mapping_count = 2;
    REQUIRE_NOTHROW(command.validate(2));
    REQUIRE_THROWS(command.validate(3));
}
