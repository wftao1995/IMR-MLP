// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "compute_mlip_heat_current.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "group.h"
#include "memory.h"
#include "pair.h"
#include "update.h"
#include "utils.h"

#include <cstring>

using namespace LAMMPS_NS;

ComputeMLIPHeatCurrent::ComputeMLIPHeatCurrent(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
  if (narg != 4) error->all(FLERR, "Illegal compute mlip/heat/current command");
  right_group_name_ = utils::strdup(arg[3]);
  right_group_ = group->find(right_group_name_);
  if (right_group_ < 0)
    error->all(FLERR, "Compute mlip/heat/current group {} does not exist",
               right_group_name_);
  right_groupbit_ = group->bitmask[right_group_];

  scalar_flag = 1;
  extscalar = 1;
  peratom_flag = 1;
  size_peratom_cols = 3;
}

ComputeMLIPHeatCurrent::~ComputeMLIPHeatCurrent()
{
  delete[] right_group_name_;
  memory->destroy(interface_force_);
}

void ComputeMLIPHeatCurrent::validate_groups()
{
  if (igroup == right_group_)
    error->all(FLERR, "Compute mlip/heat/current requires two distinct groups");
  if (group->dynamic[igroup] || group->dynamic[right_group_])
    error->all(FLERR, "Compute mlip/heat/current requires static groups");

  bigint local[3] = {0, 0, 0};
  for (int i = 0; i < atom->nlocal; ++i) {
    const bool left = atom->mask[i] & groupbit;
    const bool right = atom->mask[i] & right_groupbit_;
    local[0] += left;
    local[1] += right;
    local[2] += (left == right);
  }
  bigint global[3] = {0, 0, 0};
  MPI_Allreduce(local, global, 3, MPI_LMP_BIGINT, MPI_SUM, world);
  if (global[0] == 0 || global[1] == 0)
    error->all(FLERR, "Compute mlip/heat/current groups must both be nonempty");
  if (global[2] != 0)
    error->all(FLERR, "Compute mlip/heat/current groups must be disjoint and cover all atoms");
}

void ComputeMLIPHeatCurrent::init()
{
  right_group_ = group->find(right_group_name_);
  if (right_group_ < 0)
    error->all(FLERR, "Compute mlip/heat/current group {} does not exist",
               right_group_name_);
  right_groupbit_ = group->bitmask[right_group_];
  validate_groups();

  provider_ = dynamic_cast<MLIPTransportProvider *>(force->pair);
  if (!provider_)
    error->all(FLERR, "Compute mlip/heat/current requires a non-hybrid mlip/transport pair style");
  if (channel_ < 0 || channel_ >= provider_->transport_interface_count())
    error->all(FLERR, "MTP interface-current channel was not registered during pair initialization");
}

double ComputeMLIPHeatCurrent::compute_scalar()
{
  invoked_scalar = update->ntimestep;
  provider_->ensure_transport_reverse(update->ntimestep);
  const double *force = provider_->transport_interface_force(channel_);
  double local = 0.0;
  for (int i = 0; i < atom->nlocal; ++i) {
    const double *g = force + static_cast<size_t>(i) * 3;
    local += g[0] * atom->v[i][0] + g[1] * atom->v[i][1] +
             g[2] * atom->v[i][2];
  }
  MPI_Allreduce(&local, &scalar, 1, MPI_DOUBLE, MPI_SUM, world);
  return scalar;
}

void ComputeMLIPHeatCurrent::compute_peratom()
{
  invoked_peratom = update->ntimestep;
  provider_->ensure_transport_reverse(update->ntimestep);
  if (atom->nmax > nmax_) {
    memory->destroy(interface_force_);
    nmax_ = atom->nmax;
    memory->create(interface_force_, nmax_, 3, "mlip/heat/current:force");
    array_atom = interface_force_;
  }
  const double *source = provider_->transport_interface_force(channel_);
  for (int i = 0; i < atom->nlocal; ++i)
    for (int k = 0; k < 3; ++k)
      interface_force_[i][k] = source[static_cast<size_t>(i) * 3 + k];
}

double ComputeMLIPHeatCurrent::memory_usage()
{
  return static_cast<double>(nmax_) * 3 * sizeof(double);
}
