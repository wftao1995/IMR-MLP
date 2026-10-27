#pragma once
#include "tools/gpm_container.h"
#include <memory>
class ActiveSelector;
class GPSettings;
namespace Potentials { class AnyPotential; }
namespace AL {
class ALPotential;
struct GPMALAccess {
    static void encode(GPM::Writer &,ALPotential &,Potentials::AnyPotential &);
    static void encode_selector(GPM::Writer &,ActiveSelector &,Potentials::AnyPotential &);
    static std::unique_ptr<ALPotential> decode(const GPM::Reader &,Potentials::AnyPotential &,const GPSettings &);
private:
    static GPM::Object selector(GPM::Writer &,ActiveSelector &,Potentials::AnyPotential &);
    static std::unique_ptr<ActiveSelector> selector(const GPM::Reader &,const GPM::Object &,Potentials::AnyPotential &);
};
}
