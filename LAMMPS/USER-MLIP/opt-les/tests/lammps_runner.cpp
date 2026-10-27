// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

// Test executable using an existing LAMMPS library and these pair/compute sources.
#include "lammps.h"
#include "input.h"
#include "force.h"
#include "modify.h"
#include "pair_MLIP_opt_les.h"
#include "compute_mlip_les_charge.h"
#ifdef IMRMLP_TEST_ALL_CPU
#include "pair_MLIP.h"
#include "pair_MLIP_opt.h"
#include "pair_MLIP_omp.h"
#include "pair_MLIP_spline.h"
#include "pair_mlip_transport.h"
#include "pair_mlip_transport_omp.h"
#include "pair_mlip_transport_spline.h"
#include "compute_mlip_heat_flux.h"
#include "compute_mlip_heat_current.h"
#endif
#include <mpi.h>
using namespace LAMMPS_NS;
int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    auto* lmp=new LAMMPS(argc,argv,MPI_COMM_WORLD);
#ifdef IMRMLP_TEST_ALL_CPU
    (*lmp->force->pair_map)["mlip"]=[](LAMMPS* p)->Pair*{return new PairMLIP(p);};
    (*lmp->force->pair_map)["mlip/opt"]=[](LAMMPS* p)->Pair*{return new PairMLIPOPT(p);};
    (*lmp->force->pair_map)["mlip/opt/omp"]=[](LAMMPS* p)->Pair*{return new PairMLIPOMP(p);};
    (*lmp->force->pair_map)["mlip/spline"]=[](LAMMPS* p)->Pair*{return new PairMLIPSpline(p);};
    (*lmp->force->pair_map)["mlip/transport"]=[](LAMMPS* p)->Pair*{return new PairMLIPTransport(p);};
    (*lmp->force->pair_map)["mlip/transport/omp"]=[](LAMMPS* p)->Pair*{return new PairMLIPTransportOMP(p);};
    (*lmp->force->pair_map)["mlip/transport/spline"]=[](LAMMPS* p)->Pair*{return new PairMLIPTransportSpline(p);};
    (*lmp->modify->compute_map)["mlip/heat/flux"]=[](LAMMPS* p,int n,char** a)->Compute*{return new ComputeMLIPHeatFlux(p,n,a);};
    (*lmp->modify->compute_map)["mlip/heat/current"]=[](LAMMPS* p,int n,char** a)->Compute*{return new ComputeMLIPHeatCurrent(p,n,a);};
#endif
    (*lmp->force->pair_map)["mlip/opt-les"]=[](LAMMPS* p)->Pair*{return new PairMLIPOPTLES(p);};
    (*lmp->modify->compute_map)["mlip/les/charge"]=[](LAMMPS* p,int n,char** a)->Compute*{return new ComputeMLIPLESCharge(p,n,a);};
    (*lmp->modify->compute_map)["mlip/sog/latent"]=(*lmp->modify->compute_map)["mlip/les/charge"];
    lmp->input->file();
    delete lmp;
    MPI_Finalize();
}
