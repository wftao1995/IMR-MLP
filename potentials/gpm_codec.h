#pragma once
#include "gpm_state.h"
#include "potential.h"
#include <memory>

namespace Potentials {
struct GPMCodecAccess {
    struct Codec {
        unsigned version;
        std::string (*encode)(GPM::Writer &, AnyPotential &, const std::string &);
        std::unique_ptr<AnyPotential> (*decode)(const GPM::Reader &, const std::string &, std::set<std::string> &);
    };
    static const std::map<std::string,Codec> &registry();
    static std::string encode(GPM::Writer &, AnyPotential &, const std::string &id="root");
    static std::unique_ptr<AnyPotential> decode(const GPM::Reader &, const std::string &id="root");
    static std::string type(const AnyPotential &);
private:
    static std::string encode_builtin(GPM::Writer &, AnyPotential &, const std::string &);
    static std::unique_ptr<AnyPotential> decode_builtin(const GPM::Reader &, const std::string &,std::set<std::string>&);
    static std::unique_ptr<AnyPotential> decode_node(const GPM::Reader &, const std::string &,std::set<std::string>&);
    template<class T> static void fields(GPM::State &, T &);
};
}
