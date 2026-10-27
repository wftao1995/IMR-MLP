#include "compute_mlip_native.h"
#include "atom.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "pair.h"
#include "update.h"

using namespace LAMMPS_NS;

ComputeMLIPNativeObservable::ComputeMLIPNativeObservable(
  LAMMPS *lmp, int narg, char **arg, bool latent)
  : Compute(lmp,narg,arg), latent_(latent)
{
  if (narg != 3) error->all(FLERR,"Illegal mlip/native observable compute command");
  peratom_flag = 1;
  size_peratom_cols = 0;
  // The shape must already be known when dump/variable consumers are created.
  pair_ = force->pair_match("mlip/native",1);
  if (!pair_) error->all(FLERR,"Define pair_style mlip/native and pair_coeff before its observable compute");
  int dim = 0;
  auto *capability = static_cast<int *>(pair_->extract(
    latent_ ? "native_latent_channels" : "native_has_charges",dim));
  if (!capability || *capability <= 0)
    error->all(FLERR,latent_ ? "This mlip/native model has no latent observable" :
      "This mlip/native model has no physical charge observable (SOG latent is dimensionless)");
  channels_ = latent_ ? *capability : 1;
  size_peratom_cols = channels_ > 1 ? channels_ : 0;
}

ComputeMLIPNativeObservable::~ComputeMLIPNativeObservable()
{
  memory->destroy(vector_atom);
  memory->destroy(array_atom);
}

void ComputeMLIPNativeObservable::init()
{
  auto *current = force->pair_match("mlip/native",1);
  if (!current || current != pair_)
    error->all(FLERR,"Recreate mlip/native observable computes after replacing the pair style");
  int dim = 0;
  auto *capability = static_cast<int *>(pair_->extract(
    latent_ ? "native_latent_channels" : "native_has_charges",dim));
  if (!capability || *capability != (latent_ ? channels_ : 1))
    error->all(FLERR,"mlip/native observable shape changed; recreate the compute and its consumers");
}

void ComputeMLIPNativeObservable::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  if (atom->nmax > nmax_) {
    nmax_ = atom->nmax;
    if (channels_ == 1) {
      memory->destroy(vector_atom);
      memory->create(vector_atom,nmax_,"mlip/native:observable");
    } else {
      memory->destroy(array_atom);
      memory->create(array_atom,nmax_,channels_,"mlip/native:observable");
    }
  }
  int dim = 0;
  auto *values = static_cast<double *>(pair_->extract(
    latent_ ? "native_latent" : "native_charge",dim));
  if (!values && atom->nlocal)
    error->all(FLERR,"mlip/native observable is unavailable before pair evaluation");
  for (int i = 0; i < atom->nlocal; ++i) {
    for (int c = 0; c < channels_; ++c) {
      const double value = (atom->mask[i] & groupbit) ? values[i*channels_+c] : 0.;
      if (channels_ == 1) vector_atom[i] = value;
      else array_atom[i][c] = value;
    }
  }
}

double ComputeMLIPNativeObservable::memory_usage()
{
  return static_cast<double>(nmax_) * channels_ * sizeof(double);
}
