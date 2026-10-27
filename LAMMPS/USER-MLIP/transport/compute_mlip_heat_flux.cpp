// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "compute_mlip_heat_flux.h"

#include "atom.h"
#include "error.h"
#include "force.h"
#include "pair.h"
#include "update.h"

using namespace LAMMPS_NS;

ComputeMLIPHeatFlux::ComputeMLIPHeatFlux(LAMMPS *lmp, int narg, char **arg) :
    Compute(lmp, narg, arg)
{
  if (narg != 3) error->all(FLERR, "Illegal compute mlip/heat/flux command");
  if (igroup != 0)
    error->all(FLERR, "Compute mlip/heat/flux requires group all");
  vector_flag = 1;
  size_vector = 6;
  extvector = 1;
  vector = new double[size_vector]();
}

ComputeMLIPHeatFlux::~ComputeMLIPHeatFlux()
{
  delete[] vector;
}

void ComputeMLIPHeatFlux::init()
{
  provider_ = dynamic_cast<MLIPTransportProvider *>(force->pair);
  if (!provider_)
    error->all(FLERR, "Compute mlip/heat/flux requires a non-hybrid mlip/transport pair style");
  if (!provider_->transport_flux_enabled())
    error->all(FLERR, "MTP heat-flux channel was not enabled during pair initialization");
  if (force->bond || force->angle || force->dihedral || force->improper || force->kspace)
    error->all(FLERR, "Compute mlip/heat/flux only supports systems whose potential energy is provided entirely by pair_style mlip/transport");
}

void ComputeMLIPHeatFlux::compute_vector()
{
  invoked_vector = update->ntimestep;
  provider_->ensure_transport_reverse(update->ntimestep);

  const double *site_energy = provider_->transport_site_energy();
  const double *heat_virial = provider_->transport_heat_virial();
  double **velocity = atom->v;
  double *mass = atom->mass;
  double *rmass = atom->rmass;
  int *type = atom->type;
  const double mvv2e = force->mvv2e;

  double local[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  for (int i = 0; i < atom->nlocal; ++i) {
    const double atom_mass = rmass ? rmass[i] : mass[type[i]];
    const double kinetic = 0.5 * mvv2e * atom_mass *
        (velocity[i][0] * velocity[i][0] +
         velocity[i][1] * velocity[i][1] +
         velocity[i][2] * velocity[i][2]);
    const double energy = kinetic + site_energy[i];
    const double convective[3] = {energy * velocity[i][0],
                                  energy * velocity[i][1],
                                  energy * velocity[i][2]};
    const double *b = heat_virial + static_cast<size_t>(i) * 9;
    const double virial[3] = {
        b[0] * velocity[i][0] + b[3] * velocity[i][1] + b[4] * velocity[i][2],
        b[6] * velocity[i][0] + b[1] * velocity[i][1] + b[5] * velocity[i][2],
        b[7] * velocity[i][0] + b[8] * velocity[i][1] + b[2] * velocity[i][2]};
    for (int k = 0; k < 3; ++k) {
      local[k] += convective[k] + virial[k];
      local[k + 3] += convective[k];
    }
  }
  MPI_Allreduce(local, vector, 6, MPI_DOUBLE, MPI_SUM, world);
}
