#include "register.h"
#include "lammps.h"
#include "force.h"
#include "pair_MLIP_opt.h"
#ifdef MTP_STANDALONE_OPENMP
#include "pair_MLIP_omp.h"
#endif
#ifdef MTP_STANDALONE_GPU
#include "pair_MLIP_gpu_direct.h"
#endif

void register_mtp_standalone(LAMMPS_NS::LAMMPS *lmp)
{
    using namespace LAMMPS_NS;
    (*lmp->force->pair_map)["mlip/opt"] = [](LAMMPS *p) -> Pair * { return new PairMLIPOPT(p); };
#ifdef MTP_STANDALONE_OPENMP
    (*lmp->force->pair_map)["mlip/opt/omp"] = [](LAMMPS *p) -> Pair * { return new PairMLIPOMP(p); };
#endif
#ifdef MTP_STANDALONE_GPU
    (*lmp->force->pair_map)["mlip/gpudirect"] = [](LAMMPS *p) -> Pair * { return new PairMLIPGpuDirect(p); };
#endif
}
