// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "compute_mlip_les_charge.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "pair.h"
#include "update.h"
#include <cstring>

using namespace LAMMPS_NS;

ComputeMLIPLESCharge::ComputeMLIPLESCharge(LAMMPS *lmp, int narg, char **arg)
  : Compute(lmp,narg,arg)
{
  if (narg != 3) error->all(FLERR,"Illegal compute mlip/les/charge command");
  latent_ = std::strcmp(arg[2],"mlip/sog/latent")==0;
  peratom_flag = 1;
  size_peratom_cols = 0;
  if(latent_)init();
}

ComputeMLIPLESCharge::~ComputeMLIPLESCharge()
{
  memory->destroy(vector_atom);
  memory->destroy(array_atom);
}

void ComputeMLIPLESCharge::init()
{
  pair_ = force->pair_match("mlip/opt-les",1);
  if (!pair_)
    error->all(FLERR,"Compute mlip/les/charge requires pair_style mlip/opt-les");
  if(latent_){
    int dim=0;const auto* count=static_cast<int*>(pair_->extract("sog_channels",dim));
    const int columns=count && *count>1 ? *count : 0;
    if(layout_initialized_ && columns!=size_peratom_cols)
      error->all(FLERR,"Recreate SOG latent compute after changing channel count");
    size_peratom_cols=columns;layout_initialized_=true;
  }
}

void ComputeMLIPLESCharge::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (atom->nmax > nmax_) {
    memory->destroy(vector_atom);
    nmax_ = atom->nmax;
    if(size_peratom_cols) {memory->destroy(array_atom);memory->create(array_atom,nmax_,size_peratom_cols,"mlip/sog/latent:array_atom");}
    else memory->create(vector_atom,nmax_,"mlip/les/charge:vector_atom");
  }
  int dim = 0;
  if(size_peratom_cols){
    auto** values=static_cast<double**>(pair_->extract("sog_latents",dim));
    if(!values || dim!=2)error->all(FLERR,"SOG latent channels unavailable");
    for(int i=0;i<atom->nlocal;++i)for(int c=0;c<size_peratom_cols;++c)
      array_atom[i][c]=(atom->mask[i]&groupbit)?values[i][c]:0.;
    return;
  }
  auto *charge = static_cast<double *>(pair_->extract(latent_ ? "les_latent" : "les_charge",dim));
  if (!charge || dim != 1)
    error->all(FLERR,"Requested LES charge or SOG latent output is unavailable");
  for (int i=0; i<atom->nlocal; ++i)
    vector_atom[i] = (atom->mask[i] & groupbit) ? charge[i] : 0.0;
}

double ComputeMLIPLESCharge::memory_usage()
{
  return static_cast<double>(nmax_)*sizeof(double)*(size_peratom_cols?size_peratom_cols:1);
}
