#include "lammpsplugin.h"
#include "version.h"
#include "pair_MLIP_opt.h"
#ifdef MTP_STANDALONE_OPENMP
#include "pair_MLIP_omp.h"
#endif
#ifdef MTP_STANDALONE_GPU
#include "pair_MLIP_gpu_direct.h"
#endif

using namespace LAMMPS_NS;
namespace {
Pair *create_opt(LAMMPS *lmp) { return new PairMLIPOPT(lmp); }
#ifdef MTP_STANDALONE_OPENMP
Pair *create_omp(LAMMPS *lmp) { return new PairMLIPOMP(lmp); }
#endif
#ifdef MTP_STANDALONE_GPU
Pair *create_gpu(LAMMPS *lmp) { return new PairMLIPGpuDirect(lmp); }
#endif
}
extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
    const auto register_plugin = reinterpret_cast<lammpsplugin_regfunc>(regfunc);
    lammpsplugin_t plugin{};
    plugin.version = LAMMPS_VERSION;
    plugin.style = "pair";
    plugin.info = "Independent MTP-M FP64 inference";
    plugin.author = "atoms-agent contributors";
    plugin.handle = handle;
    plugin.name = "mlip/opt";
    plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&create_opt);
    register_plugin(&plugin, lmp);
#ifdef MTP_STANDALONE_OPENMP
    plugin.name = "mlip/opt/omp";
    plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&create_omp);
    register_plugin(&plugin, lmp);
#endif
#ifdef MTP_STANDALONE_GPU
    plugin.name = "mlip/gpudirect";
    plugin.creator.v1 = reinterpret_cast<lammpsplugin_factory1 *>(&create_gpu);
    register_plugin(&plugin, lmp);
#endif
}
