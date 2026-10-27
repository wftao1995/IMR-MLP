#include "lammpsplugin.h"
#include "version.h"
#include "pair_mlip_native.h"
#include "compute_mlip_native.h"

using namespace LAMMPS_NS;
namespace {
Pair *create_pair(LAMMPS *lmp) { return new PairMLIPNative(lmp); }
Compute *create_charge(LAMMPS *lmp,int n,char **v) { return new ComputeMLIPNativeCharge(lmp,n,v); }
Compute *create_latent(LAMMPS *lmp,int n,char **v) { return new ComputeMLIPNativeLatent(lmp,n,v); }
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "pair";
  plugin.name = "mlip/native";
  plugin.info = "atoms-agent native MTP-family inference";
  plugin.author = "atoms-agent contributors";
  plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&create_pair);
  plugin.handle = handle;
  register_plugin(&plugin,lmp);
  plugin.style = "compute";
  plugin.name = "mlip/native/charge";
  plugin.info = "atoms-agent native physical-response charges";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(&create_charge);
  register_plugin(&plugin,lmp);
  plugin.name = "mlip/native/latent";
  plugin.info = "atoms-agent native latent variables";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(&create_latent);
  register_plugin(&plugin,lmp);
}
