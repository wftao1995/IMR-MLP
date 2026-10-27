// Copyright (c) 2026 atoms-agent and IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only

#include "lammpsplugin.h"
#include "version.h"
#include "fix_mlip_maxvol.h"

using namespace LAMMPS_NS;

namespace {
Fix *create_maxvol_fix(LAMMPS *lmp, int argc, char **argv)
{
  return new FixMLIPMaxVol(lmp,argc,argv);
}
}

extern "C" void lammpsplugin_init(void *lmp, void *handle, void *regfunc)
{
  lammpsplugin_t plugin{};
  plugin.version = LAMMPS_VERSION;
  plugin.style = "fix";
  plugin.name = "mlip/maxvol";
  plugin.info = "IMR-MLP native MaxVol sampling for MTP-family potentials";
  plugin.author = "IMR-MLP contributors";
  plugin.creator.v2 = reinterpret_cast<lammpsplugin_factory2 *>(&create_maxvol_fix);
  plugin.handle = handle;
  reinterpret_cast<lammpsplugin_regfunc>(regfunc)(&plugin,lmp);
}
