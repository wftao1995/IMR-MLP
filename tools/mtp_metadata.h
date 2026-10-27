// Copyright (c) 2026 atoms-agent and IMR-MLP contributors
// SPDX-License-Identifier: MIT
#ifndef IMRMLP_MTP_METADATA_H
#define IMRMLP_MTP_METADATA_H
#include "gpm_container.h"

// Shared by GPML and the standalone LAMMPS readers (C++17, no linked dependencies).
// picojson.h is vendored from kazuho/picojson v1.3.0; its BSD license is in the file.
#include "picojson.h"
#include <algorithm>
#include <cmath>
#include <cctype>
#include <fstream>
#include <limits>
#include <set>
#include <sstream>

namespace MTPMetadata {
inline constexpr size_t max_bytes = 65536;
inline bool element(const std::string &s);
using Object = picojson::object;
using Array = picojson::array;
using Value = picojson::value;

inline std::string trim(const std::string &s) {
    const auto first = s.find_first_not_of(" \t\r\n");
    return first == std::string::npos ? "" : s.substr(first, s.find_last_not_of(" \t\r\n") - first + 1);
}
inline void require(bool ok, const std::string &message) {
    if (!ok) throw std::runtime_error("MTP metadata: " + message);
}
inline int integer(const Value &v, const std::string &key, int minimum = 0) {
    require(v.is<double>(), key + " must be an integer");
    const double n = v.get<double>();
    require(std::isfinite(n) && n >= minimum && n <= std::numeric_limits<int>::max()
            && std::floor(n) == n, key + " is outside the integer range");
    return static_cast<int>(n);
}
inline std::string string(const Value &v, const std::string &key) {
    require(v.is<std::string>() && !v.get<std::string>().empty(), key + " must be a nonempty string");
    return v.get<std::string>();
}
inline std::string uppercase(std::string name) {
    std::transform(name.begin(), name.end(), name.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    return name;
}
inline std::string canonical_model(std::string name) {
    if (name == "mtp_m_nn2") name = "mtp_m_n2";
    return name;
}

// Reject duplicate object keys and excessively nested input, including extensions.
class ParseContext : public picojson::default_parse_context {
    Value *out;
    int depth;
public:
    ParseContext(Value *value, int level = 0) : picojson::default_parse_context(value), out(value), depth(level) {}
    template<class Iter> bool parse_object_item(picojson::input<Iter> &in, const std::string &key) {
        auto &object = out->get<Object>();
        if (depth >= 64 || object.count(key)) return false;
        ParseContext child(&object[key], depth + 1);
        return picojson::_parse(child, in);
    }
    template<class Iter> bool parse_array_item(picojson::input<Iter> &in, size_t) {
        if (depth >= 64) return false;
        auto &array = out->get<Array>();
        array.emplace_back();
        ParseContext child(&array.back(), depth + 1);
        return picojson::_parse(child, in);
    }
};

struct Tag {
    Object fields;
    std::string legacy;

    bool structured() const { return fields.count("schema") != 0; }
    std::string text(const std::string &key) const {
        auto it = fields.find(key);
        return it == fields.end() ? "" : string(it->second, key);
    }
    std::vector<std::string> elements() const {
        std::vector<std::string> result;
        auto it = fields.find("elements");
        if (it == fields.end()) return result;
        require(it->second.is<Array>(), "elements must be an array");
        for (const auto &v : it->second.get<Array>()) result.push_back(string(v, "element"));
        return result;
    }
    void validate(int species_count) const {
        if (!structured()) return;
        require(text("schema") == "gpml.mtp.metadata", "unknown schema");
        auto version = fields.find("version");
        require(version != fields.end() && integer(version->second, "version", 1) == 1,
                "unsupported or missing metadata version");
        require(fields.count("type") && fields.count("model"), "type and model are required");
        require(text("type") == "MTP" || text("type") == "MTP_M" || text("type") == "MTP_LES", "unknown model type");
        const auto model = canonical_model(text("model"));
        require(model == "linear" || model == "mtp_m_li" || model == "mtp_m_q2" || model == "mtp_m_fs" || model == "mtp_m_n1" || model == "mtp_m_n2", "unknown model");
        require(text("type") == "MTP_M" || model == "linear", "ordinary MTP cannot declare an external head");
        if (fields.count("elements")) {
            const auto names = elements();
            require(static_cast<int>(names.size()) == species_count, "elements count does not match species_count");
            std::set<std::string> unique;
            for (const auto &name : names) {
                require(element(name), "unknown element symbol");
                require(!name.empty() && name.size() <= 3 && name[0] >= 'A' && name[0] <= 'Z'
                        && std::all_of(name.begin() + 1, name.end(), [](char c) { return c >= 'a' && c <= 'z'; }),
                        "invalid element symbol '" + name + "'");
                require(unique.insert(name).second, "duplicate element '" + name + "'");
            }
        }
        for (const auto &key : {"q", "model_coeffs_count"}) {
            auto it = fields.find(key);
            if (it != fields.end()) integer(it->second, key, std::string(key) == "q" ? 1 : 0);
        }
        if (text("type") == "MTP_M") {
            require(fields.count("q"), "MTP_M requires q");
            const int q = integer(fields.at("q"),"q",1);
            require(model != "linear", "MTP_M requires an external model");
            require(model != "mtp_m_li" || q == 1,"linear head requires q=1");
            require((model != "mtp_m_q2" && model != "mtp_m_fs") || q == 2,"analytic head requires q=2");
        } else require(!fields.count("q") && !fields.count("nn"),"ordinary MTP cannot declare an external head");
        auto nn = fields.find("nn");
        if (nn != fields.end()) {
            require(nn->second.is<Object>(), "nn must be an object");
            const auto &v = nn->second;
            string(v.get("activation"), "nn.activation");
            require(v.get("hidden_layers").is<Array>(), "nn.hidden_layers must be an array");
            const auto &layers = v.get("hidden_layers").get<Array>();
            const size_t expected = model=="mtp_m_n1" ? 1 : model=="mtp_m_n2" ? 2 : 0;
            require(layers.size()==expected && expected>0, "nn.hidden_layers conflicts with model");
            const auto activation=uppercase(string(v.get("activation"),"nn.activation"));
            require(activation=="TANH" || activation=="RELU" || activation=="SIGMOID" || activation=="LINEAR", "unknown activation");
            for (const auto &layer : layers) integer(layer, "nn.hidden_layers", 1);
        }
        auto lr = fields.find("long_range");
        if (lr != fields.end()) {
            require(lr->second.is<Object>(), "long_range must be an object");
            string(lr->second.get("kernel"), "long_range.kernel");
            integer(lr->second.get("channels"), "long_range.channels", 1);
        }
    }
    static Tag parse(std::string source, int species_count) {
        Tag tag;
        require(source.size() <= max_bytes, "tag exceeds 64 KiB");
        source = trim(source);
        if (!source.empty() && source.front() == '=') source = trim(source.substr(1));
        tag.legacy = source;
        const auto old_position = !source.empty() && source.front()=='{' ? std::string::npos : source.find("imrmlp_meta_v");
        if (old_position != std::string::npos) {
            require(source.compare(old_position, 15, "imrmlp_meta_v1:") == 0, "unsupported IMR-MLP metadata version");
            require(!old_position || (old_position >= 3 && source.substr(old_position-3,3) == " | "), "invalid namespace position");
            const auto legacy = old_position ? source.substr(0,old_position-3) : "";
            Value old; std::string error; ParseContext context(&old);
            const auto begin = source.begin() + old_position + 15;
            const auto end = picojson::_parse(context, begin, source.end(), &error);
            require(error.empty() && trim(std::string(end,source.end())).empty() && old.is<Object>(), "invalid IMR-MLP tag JSON");
            auto fields = old.get<Object>();
            const auto type = fields.count("model_type") ? string(fields.at("model_type"),"model_type") : "";
            require(type.empty() || type == "mtp" || type == "MTP" || type == "mtp_m" || type == "MTP_M", "unknown model_type");
            const auto layout = fields.count("layout") ? string(fields.at("layout"),"layout") : "";
            require(layout.empty() || layout == "mtp_v1" || layout == "mtp_m_v1", "unsupported layout");
            const bool multi = type == "mtp_m" || type == "MTP_M" || layout == "mtp_m_v1" || fields.count("external_model") || fields.count("q");
            require(type.empty() || layout.empty() || (multi ? layout == "mtp_m_v1" : layout == "mtp_v1"), "layout conflicts with model_type");
            require(type != "mtp" && type != "MTP" || !multi, "ordinary MTP cannot declare an external head");
            fields["schema"] = Value("gpml.mtp.metadata"); fields["version"] = Value(1.0);
            fields["type"] = Value(multi ? "MTP_M" : "MTP"); fields["model"] = Value("linear");
            fields.erase("model_type"); fields.erase("layout");
            if (fields.count("q")) {
                const auto q = fields.at("q"); require(q.is<Object>() && q.contains("count"), "q.count is required");
                integer(q.get("count"), "q.count", 1);
                if (q.contains("names")) {
                    require(q.get("names").is<Array>() && q.get("names").get<Array>().size() == size_t(integer(q.get("count"),"q.count",1)), "q names differ from count");
                    std::set<std::string> names;
                    for (const auto &v : q.get("names").get<Array>()) require(names.insert(string(v,"q name")).second,"duplicate q name");
                }
                fields["imrmlp_q"] = q; fields["q"] = q.get("count");
            }
            if (fields.count("external_model")) {
                const auto ext = fields.at("external_model"); require(ext.is<Object>() && ext.contains("type"), "external_model.type is required");
                fields["model"] = ext.get("type");
                if (ext.contains("hidden_layers") || ext.contains("activation")) {
                    fields["nn"] = Value(Object{{"hidden_layers", ext.get("hidden_layers")}, {"activation", ext.get("activation")}});
                }
                fields["imrmlp_external_model"] = ext; fields.erase("external_model");
            }
            if (!legacy.empty()) fields["legacy_tag"] = Value(legacy);
            source = Value(fields).serialize();
        }
        if (source.empty() || source.front() != '{') return tag;
        Value parsed;
        std::string error;
        ParseContext context(&parsed);
        const auto end = picojson::_parse(context, source.begin(), source.end(), &error);
        if (!error.empty() || !trim(std::string(end, source.end())).empty()) {
            require(source.find("gpml.mtp.metadata") == std::string::npos, "invalid potential_tag JSON: " + error);
            return tag;
        }
        if (!parsed.is<Object>() || !parsed.contains("schema") ||
            !parsed.get("schema").is<std::string>() || parsed.get("schema").get<std::string>() != "gpml.mtp.metadata") return tag;
        tag.fields = parsed.get<Object>();
        tag.legacy = tag.fields.count("legacy_tag") ? string(tag.fields.at("legacy_tag"),"legacy_tag") : "";
        tag.validate(species_count);
        tag.fields["model"] = Value(canonical_model(tag.text("model")));
        auto nn = tag.fields.find("nn");
        if (nn != tag.fields.end()) {
            auto &network = nn->second.get<Object>();
            network["activation"] = Value(uppercase(string(network.at("activation"), "activation")));
        }
        return tag;
    }
    void set_elements(const std::vector<std::string> &names, int species_count) {
        if (names.empty()) return;
        const auto previous = elements();
        require(previous.empty() || previous == names, "element order conflicts with the supplied index map");
        Array array;
        for (const auto &name : names) array.emplace_back(name);
        auto candidate = *this;
        candidate.fields["elements"] = Value(array);
        // Validate even when attaching elements to a legacy file.
        candidate.fields["schema"] = Value("gpml.mtp.metadata");
        candidate.fields["version"] = Value(1.0);
        if (!candidate.fields.count("type")) candidate.fields["type"] = Value("MTP");
        if (!candidate.fields.count("model")) candidate.fields["model"] = Value("linear");
        candidate.validate(species_count);
        *this = std::move(candidate);
    }
    Object output(const Object &model, int species_count) const {
        Object result = fields;
        // Regenerate model-derived fields; retain application extensions.
        for (const auto &key : {"type", "model", "q", "nn", "model_coeffs_count", "long_range"}) result.erase(key);
        for (const auto &item : model) result[item.first] = item.second;
        result["schema"] = Value("gpml.mtp.metadata");
        result["version"] = Value(1.0);
        if (!legacy.empty()) result["legacy_tag"] = Value(legacy);
        Tag check; check.fields = result; check.validate(species_count);
        require(Value(result).serialize().size() <= max_bytes, "tag exceeds 64 KiB");
        return result;
    }
};

// Legacy footer keys remain supported by standalone GPU readers.
inline std::map<std::string, std::string> legacy_model_fields(const Tag &tag) {
    std::map<std::string, std::string> fields;
    if (!tag.structured()) return fields;
    fields["mtp_model"] = canonical_model(tag.text("model"));
    for (const auto &pair : {std::make_pair("q", "mtp_descriptor_count"),
                             std::make_pair("model_coeffs_count", "mtp_model_coeffs_count")}) {
        auto it = tag.fields.find(pair.first);
        if (it != tag.fields.end()) fields[pair.second] = std::to_string(integer(it->second, pair.first));
    }
    auto nn = tag.fields.find("nn");
    if (nn != tag.fields.end()) {
        fields["mtp_activation"] = uppercase(string(nn->second.get("activation"), "activation"));
        const auto &layers = nn->second.get("hidden_layers").get<Array>();
        require(layers.size() <= 2, "MTP_M supports one or two hidden layers");
        fields["mtp_hidden1"] = std::to_string(integer(layers[0], "hidden1", 1));
        fields["mtp_hidden2"] = layers.size() == 2 ? std::to_string(integer(layers[1], "hidden2", 1)) : "0";
    }
    return fields;
}
inline std::string normalize_legacy(const std::string &key, std::string value) {
    if (key == "mtp_model") {
        value = canonical_model(value);
        if (value == "mtp") value = "linear";
    }
    if (key == "mtp_activation") value = uppercase(value);
    return value;
}
inline void merge_legacy(std::map<std::string, std::string> &fields,
                         const std::string &key, std::string value) {
    value = normalize_legacy(key, value);
    auto previous = fields.find(key);
    require(previous == fields.end() || previous->second == value, "conflicting JSON/footer metadata for " + key);
    fields[key] = value;
}

struct Header { int species_count = 0; Tag tag; };
inline Header read_header(std::istream &input) {
    Header header;
    std::string line;
    require(static_cast<bool>(std::getline(input, line)) && trim(line) == "MTP", "expected an MTP file");
    while (std::getline(input, line)) {
        const auto equals = line.find('=');
        const auto key = trim(line.substr(0, equals));
        if (key == "species_count") {
            std::istringstream value(line.substr(equals + 1));
            require(static_cast<bool>(value >> header.species_count) && header.species_count > 0, "invalid species_count");
        } else if (key == "potential_tag") {
            require(header.species_count > 0, "potential_tag precedes species_count");
            header.tag = Tag::parse(line.substr(equals + 1), header.species_count);
            return header;
        } else if (key == "radial_basis_type" || key == "radial_basis_split" || key == "radial_split") break;
    }
    require(header.species_count > 0, "missing species_count");
    return header;
}
inline Header read_header(const std::string &path) {
    if(GPM::is_file(path)) {
        GPM::Reader reader(path);const auto &node=GPM::object(GPM::at(GPM::object(GPM::at(reader.manifest,"models")),GPM::string(GPM::at(reader.manifest,"root_model"))));
        const auto &config=GPM::object(GPM::at(node,"configuration"));
        int species=GPM::integer(GPM::at(config,"species_count"));
        return Header{species,Tag::parse(GPM::at(config,"metadata").serialize(),species)};
    }

    std::ifstream input(path);
    require(input.good(), "cannot open '" + path + "'");
    return read_header(input);
}
inline std::vector<int> map_types(const Header &header, const std::vector<std::string> &names) {
    require(!names.empty(), "missing atom-type mapping");
    const auto elements = header.tag.elements();
    std::vector<int> result;
    int mode = -1;
    for (const auto &name : names) {
        const bool numeric = !name.empty() && std::all_of(name.begin(), name.end(), [](char c) { return c >= '0' && c <= '9'; });
        require(mode == -1 || mode == static_cast<int>(numeric), "cannot mix element names and numeric species IDs");
        mode = numeric;
        int id = -1;
        if (numeric) {
            try { id = std::stoi(name); } catch (...) { require(false, "invalid numeric species ID '" + name + "'"); }
        } else {
            require(name != "NULL", "NULL mappings are not supported");
            require(!elements.empty(), "element mapping requires elements in potential_tag; use numeric species IDs for a legacy file");
            const auto it = std::find(elements.begin(), elements.end(), name);
            require(it != elements.end(), "unknown element '" + name + "'");
            id = static_cast<int>(it - elements.begin());
        }
        require(id >= 0 && id < header.species_count, "species ID is outside the model range");
        result.push_back(id);
    }
    return result;
}
inline void require_model(const Tag &tag, bool allow_long_range = false, bool allow_nonlinear = false) {
    if (!tag.structured()) return;
    require(allow_nonlinear || (tag.text("model") == "linear" && tag.text("type") != "MTP_M"), "this pair style does not support the model '" + tag.text("model") + "'");
    require(allow_long_range || !tag.fields.count("long_range"), "this pair style does not support long-range terms");
}
inline bool element(const std::string &s) {
    static const std::string symbols=" H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og ";
    return !s.empty()&&s.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz")==std::string::npos&&symbols.find(' '+s+' ')!=std::string::npos;
}
inline Tag parse_tag(const std::string &raw,int species_count=0) { return Tag::parse(raw,species_count); }
inline std::vector<int> mapping(const std::vector<std::string> &arguments,const Header &h,int atom_types,int capacity) {
    require(static_cast<int>(arguments.size()) == atom_types && atom_types < capacity, "one species mapping is required per LAMMPS atom type (within adapter capacity)");
    require_model(h.tag); return map_types(h,arguments);
}
} // namespace MTPMetadata

#endif // IMRMLP_MTP_METADATA_H
