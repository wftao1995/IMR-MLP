#pragma once
#include "tools/gpm_container.h"
#include <memory>
#include <string>

class GPSettings;
class ActiveSelector;
namespace AL { class ALPotential; }
namespace Potentials {
class AnyPotential;
class ModelBundle : public std::enable_shared_from_this<ModelBundle> {
public:
    enum class Profile { Preserve, Inference, Resume };
    static std::shared_ptr<ModelBundle> load(const std::string &path);
    static std::shared_ptr<ModelBundle> from_settings(GPSettings settings);
    static GPM::Object inspect(const std::string &path);
    static void validate(const std::string &path);
    static void save_potential(AnyPotential &potential,const std::string &path,
                               Profile profile=Profile::Preserve);
    static std::string type(const AnyPotential &potential);
    static std::string fingerprint(AnyPotential &potential);
    ~ModelBundle();
    AnyPotential &potential() const;
    std::unique_ptr<AnyPotential> release_potential();
    void save(const std::string &path,Profile profile=Profile::Preserve);
    std::unique_ptr<AL::ALPotential> create_al(const GPSettings &overrides);
    void attach_al(AL::ALPotential &al);
    void attach_selector(ActiveSelector &selector);
    const GPM::Object &manifest() const {return pending_ ? pending_->manifest : source_->manifest;}
    std::shared_ptr<GPM::Reader> source() const {return source_;}
private:
    std::shared_ptr<GPM::Reader> source_;
    std::unique_ptr<AnyPotential> potential_;
    std::unique_ptr<GPM::Writer> pending_;
    ModelBundle()=default;
};
bool save_gpm_if_requested(AnyPotential &potential,const std::string &path);
} // namespace Potentials
