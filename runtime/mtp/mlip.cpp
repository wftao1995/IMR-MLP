// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: MIT

#if __has_include("gpu-direct/radial_gpu_math.h")
#include "gpu-direct/radial_gpu_math.h"
#else
#include "radial_gpu_math.h"
#endif
#include "mlip.h"
#include "radial_basis.h"
#include "radial_cache.h"
#if __has_include("mtp_metadata.h")
#include "mtp_metadata.h"
#include "gpm_mtp.h"
#else
#include "../../tools/mtp_metadata.h"
#include "../../tools/gpm_mtp.h"
#endif
#include <cstring>
#include <cmath>

#include <algorithm>
#include <cctype>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <memory>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>

namespace {

std::string trim_copy(const std::string &source)
{
    const auto first = source.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const auto last = source.find_last_not_of(" \t\r\n");
    return source.substr(first, last - first + 1);
}

bool metadata_assignment(const std::string &line, std::string *key,
                         std::string *value)
{
    std::string content = trim_copy(line);
    if (!content.empty() && content.front() == '#') {
        content = trim_copy(content.substr(1));
    }
    if (content.rfind("mtp_", 0) != 0) return false;
    const auto equals = content.find('=');
    if (equals == std::string::npos) return false;
    *key = trim_copy(content.substr(0, equals));
    *value = trim_copy(content.substr(equals + 1));
    return true;
}

[[noreturn]] void metadata_error(const std::string &message)
{
    throw std::runtime_error("Invalid MTP metadata: " + message);
}

void read_mtp_gpu_metadata(const std::string &text, MLIPCore::mlip *potential)
{
    std::istringstream input(text);

    std::map<std::string,std::string> fields;
    try {
        const auto header = MTPMetadata::read_header(input);
        input.clear(); input.seekg(0);
        MTPMetadata::require_model(header.tag, false, true);
        fields = MTPMetadata::legacy_model_fields(header.tag);
    } catch (const std::exception &e) { metadata_error(e.what()); }
    bool has_model_coeffs = false;
    std::string line;
    while (std::getline(input, line)) {
        const std::string stripped = trim_copy(line);
        // Production IMRMLP potentials may append a large active-learning MVS
        // payload. Model metadata belongs to the text potential prefix, so
        // never scan or retain the binary payload in the standalone reader.
        if (stripped.rfind("#MVS_v1.1", 0) == 0) break;
        if (stripped.rfind("les_", 0) == 0 || stripped.rfind("sog_", 0) == 0 ||
            stripped.rfind("lr_les", 0) == 0 || stripped.rfind("long_range", 0) == 0) {
            metadata_error("long-range payload is unsupported by the short-range runtime");
        }
        if (stripped.rfind("model_coeffs", 0) == 0) {
            has_model_coeffs = true;
        }

        std::string key;
        std::string value;
        if (!metadata_assignment(line, &key, &value)) continue;
        try {MTPMetadata::merge_legacy(fields,key,value);}
        catch(const std::exception &e) {metadata_error(e.what());}
    }
    const auto integer = [](const std::string &value) {
        std::size_t consumed = 0;
        const int result = std::stoi(value, &consumed);
        if (consumed != value.size()) throw std::invalid_argument("trailing integer characters");
        return result;
    };
    for (const auto &field : fields) {
        const auto &key = field.first;
        auto value = field.second;
        try {
            if (key == "mtp_model") {
                if (value == "linear" || value == "mtp") {
                    potential->site_model = MLIPCore::mlip::MODEL_LINEAR;
                } else if (value == "mtp_m_li") {
                    potential->site_model = MLIPCore::mlip::MODEL_MTPM_LI;
                } else if (value == "mtp_m_q2") {
                    potential->site_model = MLIPCore::mlip::MODEL_MTPM_Q2;
                } else if (value == "mtp_m_fs") {
                    potential->site_model = MLIPCore::mlip::MODEL_MTPM_FS;
                } else if (value == "mtp_m_n1") {
                    potential->site_model = MLIPCore::mlip::MODEL_MTPM_NN1;
                } else if (value == "mtp_m_n2" || value == "mtp_m_nn2") {
                    potential->site_model = MLIPCore::mlip::MODEL_MTPM_NN2;
                } else {
                    metadata_error("unknown mtp_model '" + value + "'");
                }
            } else if (key == "mtp_descriptor_count") {
                potential->descriptor_count = integer(value);
            } else if (key == "mtp_model_coeffs_count") {
                potential->model_coeffs_count = integer(value);
            } else if (key == "mtp_hidden1") {
                potential->model_hidden1 = integer(value);
            } else if (key == "mtp_hidden2") {
                potential->model_hidden2 = integer(value);
            } else if (key == "mtp_activation") {
                std::transform(value.begin(), value.end(), value.begin(),
                    [](unsigned char c) {
                        return static_cast<char>(std::toupper(c));
                    });
                if (value == "TANH") {
                    potential->model_activation = MLIPCore::mlip::ACTIVATION_TANH;
                } else if (value == "RELU") {
                    potential->model_activation = MLIPCore::mlip::ACTIVATION_RELU;
                } else if (value == "SIGMOID") {
                    potential->model_activation = MLIPCore::mlip::ACTIVATION_SIGMOID;
                } else if (value == "LINEAR") {
                    potential->model_activation = MLIPCore::mlip::ACTIVATION_LINEAR;
                } else {
                    metadata_error("unknown mtp_activation '" + value + "'");
                }
            }
        } catch (const std::exception &) {
            metadata_error("invalid value for " + key + ": '" + value + "'");
        }
    }

    if (potential->descriptor_count < 0 || potential->model_coeffs_count < 0 ||
        potential->model_hidden1 < 0 || potential->model_hidden2 < 0) {
        metadata_error("model dimensions and coefficient counts must be nonnegative");
    }
    for (const int dimension : {potential->descriptor_count, potential->model_coeffs_count,
                              potential->model_hidden1, potential->model_hidden2}) {
        if (static_cast<std::size_t>(dimension) > text.size())
            metadata_error("model dimension exceeds the serialized model size");
    }
    const bool neural_model = potential->site_model == MLIPCore::mlip::MODEL_MTPM_NN1 ||
        potential->site_model == MLIPCore::mlip::MODEL_MTPM_NN2;
    if (!neural_model && (potential->model_hidden1 != 0 || potential->model_hidden2 != 0))
        metadata_error("non-neural models require zero hidden dimensions");
    if (potential->site_model == MLIPCore::mlip::MODEL_MTPM_NN1 && potential->model_hidden2 != 0)
        metadata_error("NN1 requires zero hidden2");
    if (potential->site_model == MLIPCore::mlip::MODEL_LINEAR) {
        if (has_model_coeffs) {
            metadata_error(
                "model_coeffs is present but mtp_model metadata is missing");
        }
        return;
    }
    if (!has_model_coeffs) {
        metadata_error("nonlinear mtp_model requires a model_coeffs section");
    }
    if (potential->site_model == MLIPCore::mlip::MODEL_MTPM_LI) {
        if (potential->descriptor_count == 0) potential->descriptor_count = 1;
        if (potential->descriptor_count != 1 || potential->model_coeffs_count != 0)
            metadata_error("linear MTP-M requires one descriptor and no head coefficients");
    } else if (potential->site_model == MLIPCore::mlip::MODEL_MTPM_Q2 ||
        potential->site_model == MLIPCore::mlip::MODEL_MTPM_FS) {
        if (potential->descriptor_count == 0) {
            potential->descriptor_count = 2;
        } else if (potential->descriptor_count != 2) {
            metadata_error("analytic q2/fs models require descriptor_count 2");
        }
        const int expected = potential->site_model == MLIPCore::mlip::MODEL_MTPM_Q2
            ? 5 : 1;
        if (potential->model_coeffs_count == 0) {
            potential->model_coeffs_count = expected;
        } else if (potential->model_coeffs_count != expected) {
            metadata_error("analytic model coefficient count does not match model type");
        }
    } else {
        if (potential->descriptor_count <= 0 || potential->model_hidden1 <= 0) {
            metadata_error("neural model requires positive descriptor_count and hidden1");
        }
        if (potential->site_model == MLIPCore::mlip::MODEL_MTPM_NN2 &&
            potential->model_hidden2 <= 0) {
            metadata_error("NN2 model requires positive hidden2");
        }
        const long long expected = potential->site_model == MLIPCore::mlip::MODEL_MTPM_NN1
            ? static_cast<long long>(potential->model_hidden1) * (static_cast<long long>(potential->descriptor_count) + 2)
            : static_cast<long long>(potential->descriptor_count) * potential->model_hidden1 +
                  potential->model_hidden1 +
                  static_cast<long long>(potential->model_hidden1) * potential->model_hidden2 +
                  2LL * potential->model_hidden2;
        if (expected > std::numeric_limits<int>::max())
            metadata_error("neural coefficient count overflows the model index type");
        if (potential->site_model == MLIPCore::mlip::MODEL_MTPM_NN2 &&
            potential->model_activation == MLIPCore::mlip::ACTIVATION_LINEAR)
            metadata_error("NN2 does not support LINEAR activation");
        if (potential->model_coeffs_count == 0) {
            potential->model_coeffs_count = expected;
        } else if (potential->model_coeffs_count != expected) {
            metadata_error("neural model coefficient count does not match dimensions");
        }
    }
}

[[noreturn]] void radial_parse_error(const std::string &message)
{
    throw std::runtime_error("Error reading radial basis: " + message);
}

template <typename Value>
void read_assignment(std::istream &input, const char *expected, Value *value)
{
    std::string key;
    std::string equals;
    if (!(input >> key >> equals >> *value) || key != expected || equals != "=") {
        radial_parse_error(std::string("expected '") + expected + " = ...'");
    }
}

template <typename Value>
void read_assignment_value(std::istream &input, const char *key, Value *value)
{
    std::string equals;
    if (!(input >> equals >> *value) || equals != "=") {
        radial_parse_error(std::string("expected '") + key + " = ...'");
    }
}

using MLIPCore::Radial::ParsedRadialBasis;
using MLIPCore::Radial::read_radial_basis;

} // namespace

// #define DEBUG(a) std::cout << "DEBUG: " << __FILE__ << " " << __LINE__ << " " << #a << " " << a << std::endl;

namespace MLIPCore
{
    /**
     * @brief setup potential
     *
     * @param mlip_fnm : input, potential filename
     * @param sp : input, species mapping
     * @param cutoff : output, cutoff radius
     * @return mlip coeffs
     */
    std::string read_model_text(const char *filename)
    {
        if (!filename) throw std::runtime_error("Missing MTP potential filename");
        if(GPM::is_file(filename)) {
            GPM::Reader reader(filename);const auto &node=GPM::object(GPM::at(GPM::object(GPM::at(reader.manifest,"models")),GPM::string(GPM::at(reader.manifest,"root_model"))));
            auto type=GPM::string(GPM::at(node,"type"));GPM::require(type=="mtp"||type=="mtp_m","standalone does not support this GPM model type: "+type);
            GPM::require(GPM::integer(GPM::at(node,"version"))==1,"unsupported model version");
            if(node.count("required_features"))GPM::require(GPM::array(node.at("required_features")).empty(),"unsupported required model capability");
            const auto &units=GPM::object(GPM::at(reader.manifest,"units"));
            GPM::require(GPM::string(GPM::at(units,"length"))=="angstrom"&&GPM::string(GPM::at(units,"energy"))=="eV"&&GPM::string(GPM::at(units,"charge"))=="e"&&GPM::string(GPM::at(units,"stress"))=="virial_eV","unsupported GPM units");
            const auto &metadata=GPM::object(GPM::at(GPM::object(GPM::at(node,"configuration")),"metadata"));
            GPM::require(!metadata.count("long_range"),"standalone does not support long-range GPM models");
            return GPM::mtp_text(reader,node,true);
        }
        std::ifstream input(filename);
        if (!input) throw std::runtime_error("Cannot open MTP potential: " + std::string(filename));
        std::string text, line;
        while (std::getline(input, line)) {
            if (trim_copy(line).rfind("#MVS_v1.1", 0) == 0) break;
            text += line;
            text += '\n';
        }
        if (input.bad()) throw std::runtime_error("Cannot read MTP potential: " + std::string(filename));
        return text;
    }

    mlip *setup(const char *filename, const int *sp, double *cutoff,
                std::size_t mapping_count)
    {
        return setup_from_text(read_model_text(filename), sp, cutoff, mapping_count);
    }

    mlip *setup_from_text(const std::string &text, const int *sp, double *cutoff,
                         std::size_t mapping_count)
    {
        if (!sp || !cutoff || mapping_count == 0)
            throw std::runtime_error("MTP reader requires a mapping and cutoff output");
        std::istringstream ifs(text);
        const auto file_bytes = text.size();
        std::unique_ptr<mlip, void(*)(mlip*)> owner(new mlip{}, destroy_mlip);
        auto *p_mlip = owner.get();
        const auto fail = [](const std::string &message) {
            throw std::runtime_error("Invalid MTP potential: " + message);
        };
        const auto finite = [&](double value, const char *field) {
            if (!std::isfinite(value)) fail(std::string(field) + " must be finite");
        };
        const auto count = [&](int value, const char *field, bool positive = false) {
            if (value < (positive ? 1 : 0) ||
                (static_cast<unsigned long long>(value) > file_bytes + 1))
                fail(std::string(field) + " has an invalid count");
            return value;
        };
        const auto product = [&](int a, int b, const char *field) {
            const long long result = static_cast<long long>(a) * b;
            if (result < 0 || result > std::numeric_limits<int>::max())
                fail(std::string(field) + " exceeds the model index range");
            return static_cast<int>(result);
        };
        const auto sum = [&](int a, int b, const char *field) {
            if (a < 0 || b < 0 || a > std::numeric_limits<int>::max() - b)
                fail(std::string(field) + " exceeds the model index range");
            return a + b;
        };
        const auto punctuation = [&](char expected) {
            char actual = 0;
            if (!(ifs >> actual) || actual != expected)
                fail(std::string("expected '") + expected + "'");
        };
        const auto array = [&](auto *values, int size, const char *field) {
            punctuation('{');
            for (int i = 0; i < size; ++i) {
                if (!(ifs >> values[i])) fail(std::string("cannot read ") + field);
                if constexpr (std::is_floating_point_v<std::remove_pointer_t<decltype(values)>>)
                    finite(values[i], field);
                if (i + 1 < size) punctuation(',');
            }
            punctuation('}');
        };
        const auto array_assignment = [&](const char *field, auto *values, int size) {
            std::string key, equals;
            if (!(ifs >> key >> equals) || key != field || equals != "=")
                fail(std::string("expected array ") + field);
            array(values, size, field);
        };
        std::string line;
        if (!std::getline(ifs, line) || trim_copy(line) != "MTP") fail("expected MTP header");
        if (!std::getline(ifs, line) || trim_copy(line) != "version = 1.1.0")
            fail("only version 1.1.0 is supported");
        read_mtp_gpu_metadata(text, p_mlip);
        std::string token;
        if (!(ifs >> token)) fail("missing species_count");
        if (token == "potential_name") {
            std::getline(ifs, line);
            if (!(ifs >> token)) fail("missing species_count");
        }
        if (token == "scaling") {
            read_assignment_value(ifs, "scaling", &p_mlip->scaling);
            finite(p_mlip->scaling, "scaling");
            if (!(ifs >> token)) fail("missing species_count");
        }
        if (token != "species_count") fail("missing species_count");
        read_assignment_value(ifs, "species_count", &p_mlip->species_count);
        count(p_mlip->species_count, "species_count", true);
        p_mlip->type_mapping.assign(sp, sp + mapping_count);
        std::fill_n(p_mlip->spmap, mlip_g_s, -1);
        for (std::size_t i = 0; i < mapping_count; ++i) {
            if (sp[i] < -1 || sp[i] >= p_mlip->species_count) fail("invalid atom type mapping");
            if (i < mlip_g_s) p_mlip->spmap[i] = sp[i];
        }
        if (!(ifs >> token)) fail("missing radial basis");
        if (token == "potential_tag") {
            std::getline(ifs, line);
            if (!(ifs >> token)) fail("missing radial basis");
        }
        bool split = false;
        if (token == "radial_basis_split") {
            int value = 0;
            read_assignment_value(ifs, "radial_basis_split", &value);
            if (value != 1) fail("radial_basis_split must equal 1");
            split = true;
            if (!(ifs >> token)) fail("missing split pair block");
        }
        const int pairs = product(p_mlip->species_count, p_mlip->species_count, "species pairs");
        count(pairs, "species pairs", true);
        std::vector<ParsedRadialBasis> bases(split ? pairs : 1);
        std::vector<bool> seen(split ? pairs : 1, false);
        for (int block = 0; block < (split ? pairs : 1); ++block) {
            int pair = 0;
            if (split) {
                int center = -1, neighbor = -1;
                if (token != "pair" || !(ifs >> center >> neighbor) ||
                    center < 0 || center >= p_mlip->species_count ||
                    neighbor < 0 || neighbor >= p_mlip->species_count)
                    fail("invalid ordered radial species pair");
                pair = center * p_mlip->species_count + neighbor;
                if (seen[pair]) fail("duplicate ordered radial species pair");
                if (!(ifs >> token)) fail("missing pair radial basis");
            }
            seen[pair] = true;
            if (token != "radial_basis_type") fail("missing radial_basis_type");
            bases[pair] = read_radial_basis(ifs, p_mlip->scaling);
            auto &basis = bases[pair].basis;
            count(basis.size, "radial_basis_size", true);
            finite(basis.min_dist, "min_dist"); finite(basis.max_dist, "max_dist");
            finite(basis.cutoff_parameter, "cutoff parameter");
            finite(basis.transform_parameter, "transform parameter");
            finite(basis.transform_scale_parameter, "transform scale");
            if (basis.min_dist < 0 || basis.max_dist <= 0) fail("invalid radial interval");
            token = bases[pair].next_token;
        }
        std::vector<mlip::radial_basis_t *> radial_pointers;
        for (auto &b : bases) radial_pointers.push_back(&b.basis);
        Radial::finish_bases(text,radial_pointers,split,p_mlip->species_count);
        const auto &first = bases.front();
        p_mlip->rb = first.basis.rb;
        p_mlip->rb_size = first.basis.size;
        p_mlip->min_dist = first.basis.min_dist;
        p_mlip->max_dist = first.basis.max_dist;
        p_mlip->cutoff = first.basis.max_dist;
        for (const auto &parsed : bases) {
            if (parsed.basis.size != p_mlip->rb_size) fail("ordered pair radial sizes differ");
            p_mlip->cutoff = std::max(p_mlip->cutoff, parsed.basis.max_dist);
        }
        p_mlip->cutoffsq = p_mlip->cutoff * p_mlip->cutoff;
        p_mlip->mult = 2.0 / (p_mlip->max_dist - p_mlip->min_dist);
        p_mlip->plus_r = p_mlip->min_dist + p_mlip->max_dist;
        p_mlip->sub_r = p_mlip->max_dist - p_mlip->min_dist;
        p_mlip->irpi = M_PI / p_mlip->max_dist;
        p_mlip->p0 = 1.0 - p_mlip->min_dist / p_mlip->max_dist;
        p_mlip->pp = 1.0 / p_mlip->p0 / p_mlip->max_dist;
        switch (p_mlip->rb) {
        case mlip::RB_CHEBVHAR: p_mlip->rbcalc = &RB_CalcDer; break;
        case mlip::RB_CHEBVCOS: p_mlip->rbcalc = &RB_CalcDerCos; break;
        case mlip::RB_CHEBYPOLY5: p_mlip->rbcalc = &RB_CalcDerPoly5; break;
        default: p_mlip->rbcalc = nullptr; break;
        }
        if (split || p_mlip->rb == mlip::RB_CHEBYCONFIGURABLE ||
            p_mlip->rb == mlip::RB_CHEBYPOLY4 || Radial::is_poly(first.basis) || first.has_poly5_parameter) {
            p_mlip->pair_radial_basis_count = pairs;
            p_mlip->pair_radial_bases = new mlip::radial_basis_t[pairs];
            for (int pair = 0; pair < pairs; ++pair)
                p_mlip->pair_radial_bases[pair] = bases[split ? pair : 0].basis;
        }
        if (token != "radial_funcs_count") fail("missing radial_funcs_count");
        read_assignment_value(ifs, "radial_funcs_count", &p_mlip->radial_func_count);
        count(p_mlip->radial_func_count, "radial_funcs_count", true);
        const int radial_size = product(product(pairs, p_mlip->radial_func_count,
                                                "radial coefficients"),
                                        p_mlip->rb_size, "radial coefficients");
        count(radial_size, "radial coefficients");
        std::vector<double> radial(radial_size);
        if (!(ifs >> token) || token != "radial_coeffs") fail("missing radial_coeffs");
        for (int pair = 0; pair < pairs; ++pair) {
            int center = -1, neighbor = -1; char separator = 0;
            if (!(ifs >> center >> separator >> neighbor) || separator != '-' ||
                center != pair / p_mlip->species_count || neighbor != pair % p_mlip->species_count)
                fail("radial coefficient species pair order is invalid");
            for (int mu = 0; mu < p_mlip->radial_func_count; ++mu)
                array(radial.data() + (pair * p_mlip->radial_func_count + mu) * p_mlip->rb_size,
                      p_mlip->rb_size, "radial_coeffs");
        }
        p_mlip->radial_coeffs_count = radial_size;
        read_assignment(ifs, "alpha_moments_count", &p_mlip->alpha_moments_count);
        read_assignment(ifs, "alpha_index_basic_count", &p_mlip->alpha_index_basic_count);
        if (p_mlip->alpha_moments_count <= 0) fail("alpha_moments_count must be positive");
        count(p_mlip->alpha_index_basic_count, "alpha_index_basic_count", true);
        if (p_mlip->alpha_index_basic_count > p_mlip->alpha_moments_count)
            fail("basic moment count exceeds total moment slots");
        std::string equals;
        if (!(ifs >> token >> equals) || token != "alpha_index_basic" || equals != "=")
            fail("missing alpha_index_basic");
        p_mlip->alpha_index_basic = new alpha_basic_t[p_mlip->alpha_index_basic_count];
        p_mlip->nu_arr_list = new int[p_mlip->alpha_index_basic_count];
        p_mlip->max_mu = 0; p_mlip->max_alpha_index_basic = 0;
        punctuation('{');
        for (int i = 0; i < p_mlip->alpha_index_basic_count; ++i) {
            int values[4]; array(values, 4, "alpha_index_basic");
            if (values[0] < 0 || values[0] >= p_mlip->radial_func_count ||
                values[1] < 0 || values[2] < 0 || values[3] < 0)
                fail("invalid basic moment index");
            const int nu = sum(sum(values[1], values[2], "angular degree"), values[3], "angular degree");
            count(nu, "angular degree");
            p_mlip->alpha_index_basic[i] = {values[0], values[1], values[2], values[3]};
            p_mlip->nu_arr_list[i] = nu;
            p_mlip->max_mu = std::max(p_mlip->max_mu, values[0]);
            p_mlip->max_alpha_index_basic = std::max(p_mlip->max_alpha_index_basic, nu);
            if (i + 1 < p_mlip->alpha_index_basic_count) punctuation(',');
        }
        punctuation('}');
        read_assignment(ifs, "alpha_index_times_count", &p_mlip->alpha_index_times_count);
        count(p_mlip->alpha_index_times_count, "alpha_index_times_count");
        if (!(ifs >> token >> equals) || token != "alpha_index_times" || equals != "=")
            fail("missing alpha_index_times");
        p_mlip->alpha_index_times = new alpha_times_t[p_mlip->alpha_index_times_count];
        std::unordered_set<int> defined, consumed;
        for (int i = 0; i < p_mlip->alpha_index_basic_count; ++i) defined.insert(i);
        punctuation('{');
        for (int i = 0; i < p_mlip->alpha_index_times_count; ++i) {
            int v[4]; array(v, 4, "alpha_index_times");
            if (v[0] < 0 || v[0] >= p_mlip->alpha_moments_count ||
                v[1] < 0 || v[1] >= p_mlip->alpha_moments_count ||
                v[3] < p_mlip->alpha_index_basic_count || v[3] >= p_mlip->alpha_moments_count ||
                !defined.count(v[0]) || !defined.count(v[1]) || v[3] == v[0] || v[3] == v[1] || consumed.count(v[3]))
                fail("moment graph has an undefined input, cycle or late accumulation");
            p_mlip->alpha_index_times[i] = {v[0], v[1], v[2], v[3]};
            defined.insert(v[3]); consumed.insert(v[0]); consumed.insert(v[1]);
            if (i + 1 < p_mlip->alpha_index_times_count) punctuation(',');
        }
        punctuation('}');
        read_assignment(ifs, "alpha_scalar_moments", &p_mlip->alpha_scalar_moments);
        count(p_mlip->alpha_scalar_moments, "alpha_scalar_moments");
        p_mlip->alpha_moment_mapping = new int[p_mlip->alpha_scalar_moments];
        array_assignment("alpha_moment_mapping", p_mlip->alpha_moment_mapping, p_mlip->alpha_scalar_moments);
        for (int i = 0; i < p_mlip->alpha_scalar_moments; ++i) {
            const int node = p_mlip->alpha_moment_mapping[i];
            if (node < 0 || node >= p_mlip->alpha_moments_count || !defined.count(node))
                fail("scalar maps to an undefined moment");
        }
        const bool nonlinear = p_mlip->site_model != mlip::MODEL_LINEAR;
        const int descriptors = nonlinear ? p_mlip->descriptor_count : 1;
        const int species_coeffs_count = product(p_mlip->species_count,
            nonlinear ? sum(descriptors, 1, "descriptor species coefficients") : 1,
            "species coefficients");
        p_mlip->moment_coeffs_count = product(p_mlip->alpha_scalar_moments, descriptors, "moment coefficients");
        p_mlip->descriptor_species_coeffs_count = nonlinear ? species_coeffs_count - p_mlip->species_count : 0;
        p_mlip->descriptor_moment_coeffs_count = nonlinear ? p_mlip->moment_coeffs_count : 0;
        p_mlip->linear_coeffs_count = sum(sum(species_coeffs_count, p_mlip->moment_coeffs_count,
            "non-radial coefficients"), p_mlip->model_coeffs_count, "non-radial coefficients");
        p_mlip->coeffs_count = sum(radial_size, p_mlip->linear_coeffs_count, "coefficients");
        count(p_mlip->coeffs_count, "coefficients");
        p_mlip->regression_coeffs = new double[p_mlip->coeffs_count];
        std::copy(radial.begin(), radial.end(), p_mlip->regression_coeffs);
        p_mlip->radial_coeffs = p_mlip->regression_coeffs;
        p_mlip->species_coeffs = p_mlip->radial_coeffs + radial_size;
        p_mlip->moment_coeffs = p_mlip->species_coeffs + species_coeffs_count;
        if (nonlinear) {
            p_mlip->descriptor_species_coeffs = p_mlip->species_coeffs + p_mlip->species_count;
            p_mlip->descriptor_moment_coeffs = p_mlip->moment_coeffs;
            p_mlip->model_coeffs = p_mlip->moment_coeffs + p_mlip->moment_coeffs_count;
        }
        array_assignment("species_coeffs", p_mlip->species_coeffs, species_coeffs_count);
        array_assignment("moment_coeffs", p_mlip->moment_coeffs, p_mlip->moment_coeffs_count);
        if (nonlinear) array_assignment("model_coeffs", p_mlip->model_coeffs, p_mlip->model_coeffs_count);
        if (ifs >> token) {
            if (token == "max_level") {
                read_assignment_value(ifs, "max_level", &p_mlip->max_level);
                count(p_mlip->max_level, "max_level");
                const int levels = sum(p_mlip->max_level, 1, "levels");
                p_mlip->lev_scalar_nodes = new int[levels];
                p_mlip->lev_scalar_count = new int[levels];
                p_mlip->lev_scalar_accum = new int[levels]{};
                p_mlip->lev_times_count = new int[levels];
                p_mlip->lev_times_nodes = new int[levels]{};
                array_assignment("lev_scalar_nodes", p_mlip->lev_scalar_nodes, levels);
                array_assignment("lev_scalar_count", p_mlip->lev_scalar_count, levels);
                array_assignment("lev_times_count", p_mlip->lev_times_count, levels);
                for (int i = 0; i < levels; ++i) {
                    if (p_mlip->lev_scalar_nodes[i] < 0 ||
                        p_mlip->lev_scalar_nodes[i] >= p_mlip->alpha_moments_count)
                        fail("level scalar node exceeds the declared moment range");
                    count(p_mlip->lev_scalar_count[i], "lev_scalar_count");
                    count(p_mlip->lev_times_count[i], "lev_times_count");
                    if (i) p_mlip->lev_scalar_accum[i] = sum(p_mlip->lev_scalar_accum[i-1],
                        p_mlip->lev_scalar_count[i-1], "level scalar prefix");
                    if (i > 1) p_mlip->lev_times_nodes[i] = sum(p_mlip->lev_times_nodes[i-1],
                        p_mlip->lev_times_count[i-1], "level product prefix");
                }
                p_mlip->have_order_info = true;
                if (!(ifs >> token)) token.clear();
            }
            while (!token.empty()) {
                if (token.rfind("#MVS_v1.1", 0) == 0) break;
                if (token.front() != '#') fail("unsupported trailing section: " + token);
                std::getline(ifs, line);
                if (!(ifs >> token)) break;
            }
        }
        p_mlip->max_alpha_index_basic = sum(p_mlip->max_alpha_index_basic, 1, "angular workspace");
        p_mlip->max_mu = sum(p_mlip->max_mu, 1, "radial workspace");
        *cutoff = p_mlip->cutoff;
        return owner.release();
    }

    mlip_buffers *create_mlip_buffer(const mlip *p_mlip) {
        std::unique_ptr<mlip_buffers, void(*)(mlip_buffers*)> owner(
            new mlip_buffers{}, destroy_mlip_buffer);
        mlip_buffers *p_buffer = owner.get();
        p_buffer->val_mu = new double[p_mlip->max_mu];
        p_buffer->der_mu = new double[p_mlip->max_mu];
        p_buffer->moment_vals = new double[p_mlip->alpha_moments_count];
        p_buffer->site_energy_ders_wrt_moments_ = new double[p_mlip->alpha_moments_count];
        p_buffer->rb_vals = new double[p_mlip->rb_size];
        p_buffer->rb_ders = new double[p_mlip->rb_size];
        p_buffer->dist_powers = new double[p_mlip->max_alpha_index_basic];
        p_buffer->dist_powers[0] = 1;
        // // coord_powers is offset by 3 to avoid if-statement in main loop
        p_buffer->coords_powers = new coordpow_t[p_mlip->max_alpha_index_basic+1];
        p_buffer->coords_powers[0].x = 0;
        p_buffer->coords_powers[0].y = 0;
        p_buffer->coords_powers[0].z = 0;
        p_buffer->coords_powers[1].x = 1;
        p_buffer->coords_powers[1].y = 1;
        p_buffer->coords_powers[1].z = 1;
        return owner.release();
    }

    void destroy_mlip_buffer(mlip_buffers *p_buffer) {
        delete[] p_buffer->val_mu;
        delete[] p_buffer->der_mu;
        delete[] p_buffer->moment_vals;
        delete[] p_buffer->site_energy_ders_wrt_moments_;
        delete[] p_buffer->rb_vals;
        delete[] p_buffer->rb_ders;
        delete[] p_buffer->dist_powers;
        delete[] p_buffer->coords_powers;
        if (p_buffer->moment_jacobian) delete[] p_buffer->moment_jacobian;
        if (p_buffer->jlist_true) delete[] p_buffer->jlist_true;
        if (p_buffer->buff_site_energy_ders) delete[] p_buffer->buff_site_energy_ders;
        delete p_buffer;
        p_buffer = nullptr;
    }

    void destroy_mlip(mlip *p_mlip) {
        if (p_mlip->regression_coeffs) delete[] p_mlip->regression_coeffs;
        if (p_mlip->pair_radial_bases) delete[] p_mlip->pair_radial_bases;
        if (p_mlip->alpha_index_basic) delete[] p_mlip->alpha_index_basic;
        if (p_mlip->alpha_index_times) delete[] p_mlip->alpha_index_times;
        if (p_mlip->nu_arr_list) delete[] p_mlip->nu_arr_list;
        if (p_mlip->alpha_moment_mapping) delete[] p_mlip->alpha_moment_mapping;
        if (p_mlip->lev_scalar_count) delete[] p_mlip->lev_scalar_count;
        if (p_mlip->lev_scalar_nodes) delete[] p_mlip->lev_scalar_nodes;
        if (p_mlip->lev_scalar_accum) delete[] p_mlip->lev_scalar_accum;
        if (p_mlip->lev_times_count) delete[] p_mlip->lev_times_count;
        if (p_mlip->lev_times_nodes) delete[] p_mlip->lev_times_nodes;
        delete p_mlip;
    }

    void extent_mlip_buffer(const mlip *p_mlip, mlip_buffers *p_buffer, int jnum_new) {
        double *mom_new = new double[p_mlip->alpha_index_basic_count*jnum_new*3];
        if (p_buffer->moment_jacobian)
            delete[] p_buffer->moment_jacobian;
        p_buffer->moment_jacobian = mom_new;
        int *mom_list = new int[jnum_new];
        if (p_buffer->jlist_true) 
            delete[] p_buffer->jlist_true;
        p_buffer->jlist_true = mom_list;
        double *mom_der = new double[jnum_new * 3];
        if (p_buffer->buff_site_energy_ders) 
            delete[] p_buffer->buff_site_energy_ders;
        p_buffer->buff_site_energy_ders = mom_der;
    }

    /**
     * @brief compute interactions
     * 
     * @param ilist 
     * @param numneigh 
     * @param inum 
     * @param type 
     * @param x 
     * @param firstneigh 
     * @param en 
     * @param p_site_en 
     * @param p_site_virial 
     * @param f 
     */
    void compute(const mlip *p_mlip,
                 mlip_buffers *p_buffer,
                 const int *ilist,
                 const int *numneigh,
                 const int inum,
                 const int *type,
                 double **x,
                 int **firstneigh,
                 double *en,
                 double *p_site_en,
                 double **p_site_virial,
                 double **f)
    {
        const auto *spmap = p_mlip->mapping_data();
        auto &jnum_old = p_buffer->jnum_old;
        auto &moment_jacobian = p_buffer->moment_jacobian;
        auto &alpha_index_basic_count = p_mlip->alpha_index_basic_count;
        auto &moment_vals = p_buffer->moment_vals;
        auto &buff_site_energy_ders = p_buffer->buff_site_energy_ders;
        auto &alpha_moments_count = p_mlip->alpha_moments_count;
        auto &cutoffsq = p_mlip->cutoffsq;
        auto &jlist_true = p_buffer->jlist_true;
        auto &max_alpha_index_basic = p_mlip->max_alpha_index_basic;
        auto &dist_powers = p_buffer->dist_powers;
        auto &coords_powers = p_buffer->coords_powers;
        auto &radial_coeffs = p_mlip->radial_coeffs;
        auto &max_mu = p_mlip->max_mu;
        auto &val_mu = p_buffer->val_mu;
        auto &der_mu = p_buffer->der_mu;
        auto &species_count = p_mlip->species_count;
        auto &radial_func_count = p_mlip->radial_func_count;
        auto &rb_size = p_mlip->rb_size;
        auto &rb_vals = p_buffer->rb_vals;
        auto &rb_ders = p_buffer->rb_ders;
        auto &alpha_index_basic = p_mlip->alpha_index_basic;
        auto &nu_arr_list = p_mlip->nu_arr_list;
        auto &alpha_index_times_count = p_mlip->alpha_index_times_count;
        auto &species_coeffs = p_mlip->species_coeffs;
        auto &alpha_scalar_moments = p_mlip->alpha_scalar_moments;
        auto &alpha_index_times = p_mlip->alpha_index_times;
        auto &moment_coeffs = p_mlip->moment_coeffs;
        auto &alpha_moment_mapping = p_mlip->alpha_moment_mapping;
        auto &site_energy_ders_wrt_moments_ = p_buffer->site_energy_ders_wrt_moments_;

        for (int ii = 0; ii < inum; ++ii) {
            const int ati = ilist[ii];
            const int jnum = numneigh[ati];
            const int type_central = spmap[type[ati]];
            if (type_central < 0) continue;
            const double ivec[3] = {x[ati][0], x[ati][1], x[ati][2]};
            double buff_site_energy = 0;
            // allocate memory
            if (jnum_old < jnum) {
                extent_mlip_buffer(p_mlip, p_buffer, jnum);
            }
            memset(moment_jacobian, 0, alpha_index_basic_count*3*jnum*sizeof(double));
            memset(buff_site_energy_ders, 0, jnum*3*sizeof(double));
            memset(moment_vals, 0, alpha_moments_count*sizeof(double));

            int jnum_true = 0;
            for (int jj = 0; jj < jnum; ++jj) {
                int atj = firstneigh[ati][jj] & mlip_neighbor_index_mask;
                const double *jvec = x[atj];
                const int jtype = spmap[type[atj]];
                if (jtype < 0) continue;
                const double rvec[3] = {jvec[0] - ivec[0], jvec[1] - ivec[1], jvec[2] - ivec[2]};
                const double rijsq = rvec[0] * rvec[0] + rvec[1] * rvec[1] + rvec[2] * rvec[2];
                const int pair = type_central * species_count + jtype;
                const mlip::radial_basis_t *pair_basis =
                    p_mlip->pair_radial_basis_count != 0
                        ? p_mlip->pair_radial_bases + pair : nullptr;
                const double pair_cutoffsq = pair_basis != nullptr
                    ? pair_basis->max_dist * pair_basis->max_dist : cutoffsq;
                const bool below_pair_min = pair_basis != nullptr &&
                    pair_basis->min_dist > 0.0 &&
                    (pair_basis->rb == mlip::RB_CHEBYPOLY4 || pair_basis->rb == mlip::RB_CHEBYPOLY5 ||
                     pair_basis->rb == mlip::RB_CHEBYCONFIGURABLE) &&
                    rijsq < pair_basis->min_dist * pair_basis->min_dist;
                if (rijsq > 0.0 && rijsq < pair_cutoffsq && !below_pair_min) {
                    const double rij = sqrt(rijsq);
                    const double rijinv = 1.0 / rij;
                    jlist_true[jnum_true] = atj;
                    // begin-RB
                    if (pair_basis != nullptr) {
                        RB_CalcDerPair(p_mlip, pair_basis, p_buffer, rij);
                    } else {
                        p_mlip->rbcalc(p_mlip, p_buffer, rij);
                    }
                    // end-RB
                    for (int k = 1; k < max_alpha_index_basic; ++k) {
                        dist_powers[k] = dist_powers[k - 1] * rijinv;
                        coords_powers[k+1] = {
                            coords_powers[k].x * rvec[0],
                            coords_powers[k].y * rvec[1],
                            coords_powers[k].z * rvec[2]
                        };
                    }
                    // memset(&val_mu[0], 0, max_mu * sizeof(val_mu[0]));
                    // memset(&der_mu[0], 0, max_mu * sizeof(der_mu[0]));
                    for (int mu = 0; mu < max_mu; ++mu) {
                        val_mu[mu] = 0;
                        der_mu[mu] = 0;
                        const int idxoffset = (type_central * species_count + jtype) * radial_func_count * rb_size + mu * rb_size;
                        #pragma ivdep
                        for (int xi = 0; xi < rb_size; ++xi) {
                            val_mu[mu] += radial_coeffs[idxoffset + xi] * rb_vals[xi];
                        }
                        #pragma ivdep
                        for (int xi = 0; xi < rb_size; ++xi) {
                            der_mu[mu] += radial_coeffs[idxoffset + xi] * rb_ders[xi];
                        }
                    }
                    // inner loop
                    for (int i = 0; i < alpha_index_basic_count; ++i) {
                        const int mu = alpha_index_basic[i].mu;
                        double val = val_mu[mu], der = der_mu[mu];

                        const int k  = nu_arr_list[i];
                        const double rpowk = dist_powers[k];
                        val *= rpowk;
                        der = der * rpowk - k * val * rijinv;
                        
                        const double pow0 = coords_powers[alpha_index_basic[i].b1 + 1].x;
                        const double pow1 = coords_powers[alpha_index_basic[i].b2 + 1].y;
                        const double pow2 = coords_powers[alpha_index_basic[i].b3 + 1].z;
                        double mult0 = pow0 * pow1 * pow2;
                        
                        moment_vals[i] += val * mult0;
                        mult0 *= der * rijinv;
                        const int idxoffset = i * jnum * 3 + 3 * jnum_true;
                        #pragma ivdep
                        for (int a = 0; a < 3; ++a) {
                            moment_jacobian[idxoffset + a] += mult0 * rvec[a];
                        }

                        moment_jacobian[idxoffset + 0] += 
                            alpha_index_basic[i].b1 * val * 
                            coords_powers[alpha_index_basic[i].b1].x *
                            pow1 * pow2;
                        moment_jacobian[idxoffset + 1] += 
                            alpha_index_basic[i].b2 * val * 
                            coords_powers[alpha_index_basic[i].b2].y *
                            pow0 * pow2;
                        moment_jacobian[idxoffset + 2] += 
                            alpha_index_basic[i].b3 * val * 
                            coords_powers[alpha_index_basic[i].b3].z *
                            pow0 * pow1;
                    }
                    ++jnum_true;
                }
            }
            // non-elementary b_i
            // TODO: compute each moments seperately, GPU
            // for (int ilev = 1; ilev <= max_level; ++ilev) {
            //     const int idxoffset = lev_times_nodes[ilev];
            //     for (int i = 0; i < lev_times_count[ilev]; ++i) {
            //         const alpha_times_t &idtime = alpha_index_times[idxoffset + i];
            //         double val0 = moment_vals[idtime.l1];
            //         double val1 = moment_vals[idtime.l2];
            //         moment_vals[idtime.id] += idtime.c * val0 * val1;
            //     }
            // }
            for (int i = 0; i < alpha_index_times_count; ++i) {
                const double val0 = moment_vals[alpha_index_times[i].l1];
                const double val1 = moment_vals[alpha_index_times[i].l2];
                moment_vals[alpha_index_times[i].id] += 
                        alpha_index_times[i].c * val0 * val1;
            }
            // BP-1
            buff_site_energy += species_coeffs[type_central];
            memset(&site_energy_ders_wrt_moments_[0], 0, alpha_moments_count * sizeof(double));
            // for (int ilev = 0; ilev <= max_level; ++ilev) {
            //     const int idxoffset = lev_scalar_nodes[ilev];
            //     const int idxaccu = lev_scalar_accum[ilev];
            //     #pragma ivdep
            //     for (int i = 0; i < lev_scalar_count[ilev]; ++i) {
            //         buff_site_energy += 
            //             moment_coeffs[idxaccu + i] * moment_vals[idxoffset + i];
            //     }
            //     #pragma ivdep 
            //     for (int i = 0; i < lev_scalar_count[ilev]; ++i) {
            //         site_energy_ders_wrt_moments_[idxoffset + i] = moment_coeffs[idxaccu + i];
            //     }
            // }
            #pragma ivdep
            for (int i = 0; i < alpha_scalar_moments; i++) {
                buff_site_energy += moment_coeffs[i] * moment_vals[alpha_moment_mapping[i]];
            }
            // TODO: sort moments by level and is_scalar, then this can be parallized and much more effective
            #pragma ivdep 
            for (int i = 0; i < alpha_scalar_moments; ++i) {
                site_energy_ders_wrt_moments_[alpha_moment_mapping[i]] = moment_coeffs[i];
            }
            // BP-2
            // for (int ilev = max_level; ilev >= 1; --ilev) {
            //     const int idxoffset = lev_times_nodes[ilev];
            //     for (int i = 0; i < lev_times_count[ilev]; ++i) {
            //         const alpha_times_t &idtime = alpha_index_times[idxoffset + i];
            //         double val0 = moment_vals[idtime.l1];
            //         double val1 = moment_vals[idtime.l2];
            //         double mulc = idtime.c;

            //         site_energy_ders_wrt_moments_[idtime.l2] += 
            //             site_energy_ders_wrt_moments_[idtime.id] * mulc * val0;
            //         site_energy_ders_wrt_moments_[idtime.l1] += 
            //             site_energy_ders_wrt_moments_[idtime.id] * mulc * val1;
            //     }
            // }
            for (int i = alpha_index_times_count - 1; i >= 0; i--)
            {
                const double val0 = moment_vals[alpha_index_times[i].l1];
                const double val1 = moment_vals[alpha_index_times[i].l2];
                const double mulc = alpha_index_times[i].c;

                site_energy_ders_wrt_moments_[alpha_index_times[i].l2] +=
                    site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val0;
                site_energy_ders_wrt_moments_[alpha_index_times[i].l1] +=
                    site_energy_ders_wrt_moments_[alpha_index_times[i].id] * mulc * val1;
            }
            // BP-3
            for (int i = 0; i < alpha_index_basic_count; i++) {
                const int idxoffset1 = i * jnum * 3;
                for (int j = 0; j < jnum_true; j++) {
                    const int idxoffset2 = j * 3;
                    #pragma ivdep
                    for (int a = 0; a < 3; a++) {
                        buff_site_energy_ders[idxoffset2 + a] += site_energy_ders_wrt_moments_[i] * moment_jacobian[idxoffset1 + idxoffset2 + a];
                    }
                }
            }

            // extract energy, force, virual, etc
            *en += buff_site_energy;
            if (p_site_en != nullptr)
            {
                p_site_en[ati] = buff_site_energy;
            }
            for (int i = 0; i < jnum_true; i++)
            {
                const int idxoffset = i * 3;
                for (int a = 0; a < 3; ++a) {
                    f[ati][a] += buff_site_energy_ders[idxoffset + a];
                }
                for (int a = 0; a < 3; ++a) {
                    f[jlist_true[i]][a] -= buff_site_energy_ders[idxoffset + a];
                }
            }
            if (p_site_virial != nullptr)
            {
                for (int i = 0; i < jnum_true; i++)
                {
                    int atj = jlist_true[i];
                    double jvec[3] = {x[atj][0] - x[ati][0], x[atj][1] - x[ati][1], x[atj][2] - x[ati][2]};
                    p_site_virial[ati][0] -= 0.5 * buff_site_energy_ders[i * 3 + 0] * jvec[0];
                    p_site_virial[ati][1] -= 0.5 * buff_site_energy_ders[i * 3 + 1] * jvec[1];
                    p_site_virial[ati][2] -= 0.5 * buff_site_energy_ders[i * 3 + 2] * jvec[2];
                    p_site_virial[ati][3] -= 0.25 * (buff_site_energy_ders[i * 3 + 1] * jvec[0] +
                                                    buff_site_energy_ders[i * 3 + 0] * jvec[1]);
                    p_site_virial[ati][4] -= 0.25 * (buff_site_energy_ders[i * 3 + 2] * jvec[0] +
                                                    buff_site_energy_ders[i * 3 + 0] * jvec[2]);
                    p_site_virial[ati][5] -= 0.25 * (buff_site_energy_ders[i * 3 + 2] * jvec[1] +
                                                    buff_site_energy_ders[i * 3 + 1] * jvec[2]);

                    p_site_virial[atj][0] -= 0.5 * buff_site_energy_ders[i * 3 + 0] * jvec[0];
                    p_site_virial[atj][1] -= 0.5 * buff_site_energy_ders[i * 3 + 1] * jvec[1];
                    p_site_virial[atj][2] -= 0.5 * buff_site_energy_ders[i * 3 + 2] * jvec[2];
                    p_site_virial[atj][3] -= 0.25 * (buff_site_energy_ders[i * 3 + 1] * jvec[0] +
                                                    buff_site_energy_ders[i * 3 + 0] * jvec[1]);
                    p_site_virial[atj][4] -= 0.25 * (buff_site_energy_ders[i * 3 + 2] * jvec[0] +
                                                    buff_site_energy_ders[i * 3 + 0] * jvec[2]);
                    p_site_virial[atj][5] -= 0.25 * (buff_site_energy_ders[i * 3 + 2] * jvec[1] +
                                                    buff_site_energy_ders[i * 3 + 1] * jvec[2]);
                }
            }

        }
    }

    void RB_CalcDerPair(const mlip *p_mlip, const mlip::radial_basis_t *basis,
                       mlip_buffers *p_buffers, const double &r)
    {
        Radial::evaluate(*basis,r,p_buffers->rb_vals,p_buffers->rb_ders);
    }

    void RB_CalcDer(const mlip *p_mlip, mlip_buffers *p_buffers, const double &r)
    {
        // if (rb_type == 1)
        double ksi = (2 * r - p_mlip->plus_r) / p_mlip->sub_r;
        double rm = r - p_mlip->max_dist;
        p_buffers->rb_vals[0] = p_mlip->scaling * rm * rm;

        p_buffers->rb_ders[0] = p_mlip->scaling * 2 * rm;
        if (p_mlip->rb_size == 1) return;
        p_buffers->rb_vals[1] = p_mlip->scaling * ksi * rm * rm;
        p_buffers->rb_ders[1] = p_mlip->scaling * (p_mlip->mult * rm * rm + 2 * ksi * rm);

        for (int i = 2; i < p_mlip->rb_size; i++)
        {
            p_buffers->rb_vals[i] = 2 * ksi * p_buffers->rb_vals[i - 1] - p_buffers->rb_vals[i - 2];
            p_buffers->rb_ders[i] = 2 * (p_mlip->mult * p_buffers->rb_vals[i - 1] + ksi * p_buffers->rb_ders[i - 1]) - p_buffers->rb_ders[i - 2];
        }
    }

    void RB_CalcDerCos(const mlip *p_mlip, mlip_buffers *p_buffers, const double &r)
    {
        // if (rb_type == 1)
        double ksi = (2 * r - p_mlip->plus_r) / p_mlip->sub_r;
        double rc = 0.5 * (1.0 + cos(r * p_mlip->irpi ));
        double drc = -0.5 * p_mlip->irpi * sin(r * p_mlip->irpi);
        p_buffers->rb_vals[0] = p_mlip->scaling * rc;

        p_buffers->rb_ders[0] = p_mlip->scaling * drc;
        if (p_mlip->rb_size == 1) return;
        p_buffers->rb_vals[1] = p_mlip->scaling * ksi * rc;
        p_buffers->rb_ders[1] = p_mlip->scaling * (p_mlip->mult * rc + ksi * drc);

        for (int i = 2; i < p_mlip->rb_size; i++)
        {
            p_buffers->rb_vals[i] = 2 * ksi * p_buffers->rb_vals[i - 1] - p_buffers->rb_vals[i - 2];
            p_buffers->rb_ders[i] = 2 * (p_mlip->mult * p_buffers->rb_vals[i - 1] + ksi * p_buffers->rb_ders[i - 1]) - p_buffers->rb_ders[i - 2];
        }
    }

    void RB_CalcDerPoly5(const mlip *p_mlip, mlip_buffers *p_buffers, const double &r) {
        double ksi = (2 * r - p_mlip->plus_r) / p_mlip->sub_r;
        double p = 1.0 - r / p_mlip->cutoff;
        double q = p / p_mlip->p0;
        double q2 = q * q;
        double q3 = q * q2;
        double q4 = q * q3;
        double q5 = q * q4;

        double fc = 6.0 * q5 - 15.0 * q4 + 10.0 * q3;
        double dfc = -(30.0 * q4 - 60.0 * q3 + 30.0 * q2) * p_mlip->pp;

        p_buffers->rb_vals[0] = p_mlip->scaling * fc;
        p_buffers->rb_ders[0] = p_mlip->scaling * dfc;
        if (p_mlip->rb_size == 1) return;
        p_buffers->rb_vals[1] = p_mlip->scaling * ksi * fc;
        p_buffers->rb_ders[1] = p_mlip->scaling * (p_mlip->mult * fc + ksi * dfc);
        for (int i = 2; i < p_mlip->rb_size; i++)
        {
            p_buffers->rb_vals[i] = 2 * ksi * p_buffers->rb_vals[i - 1] - p_buffers->rb_vals[i - 2];
            p_buffers->rb_ders[i] = 2 * (p_mlip->mult * p_buffers->rb_vals[i - 1] + ksi * p_buffers->rb_ders[i - 1]) - p_buffers->rb_ders[i - 2];
        }
    }
    /**
     * @brief For CUDA
     * 1. split atoms into chuncks
     * 2. for each chunck start a kernel
     * 3. Each atom is assigned to a `Thread Block`
     * 4. moment_jacs : block i for center atom, thread j for each j atom
     * 5. moment_vals : block i for center atom, 
     *      
     */

};
