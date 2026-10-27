// Copyright (c) 2026 atoms-agent and IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only

#include "fix_mlip_maxvol.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "update.h"
#include "utils.h"

#include "io/gpio.h"
#include "maxvol/active_selector.h"
#include "potentials/potential_wrapper.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

using namespace LAMMPS_NS;

FixMLIPMaxVol::FixMLIPMaxVol(LAMMPS *lmp, int narg, char **arg) : Fix(lmp,narg,arg)
{
  if (atom->ntypes <= 0 || narg < 7 + atom->ntypes)
    error->all(FLERR,
      "Usage: fix ID all mlip/maxvol NEVERY MODEL STATE INDEX TYPE_TO_SPECIES... "
      "[pottype TYPE] [settings FILE] "
      "[save FILE] [threshold-save VALUE] [threshold-break VALUE|none]");
  if (igroup != 0)
    error->all(FLERR,"Fix mlip/maxvol requires group all (whole configurations)");
  nevery = utils::inumeric(FLERR,arg[3],false,lmp);
  if (nevery <= 0) error->all(FLERR,"Fix mlip/maxvol NEVERY must be positive");
  model_file_ = arg[4];
  state_file_ = arg[5];
  index_file_ = arg[6];
  for (int i = 0; i < atom->ntypes; ++i)
    species_map_.push_back(utils::inumeric(FLERR,arg[7+i],false,lmp));
  for (int i = 7 + atom->ntypes; i < narg; i += 2) {
    if (i + 1 == narg) error->all(FLERR,"Missing fix mlip/maxvol keyword value");
    if (std::strcmp(arg[i],"save") == 0) {
      save_file_ = arg[i+1];
    } else if (std::strcmp(arg[i],"pottype") == 0) {
      potential_type_ = arg[i+1];
    } else if (std::strcmp(arg[i],"settings") == 0) {
      settings_file_ = arg[i+1];
    } else if (std::strcmp(arg[i],"threshold-save") == 0) {
      threshold_save_ = utils::numeric(FLERR,arg[i+1],false,lmp);
    } else if (std::strcmp(arg[i],"threshold-break") == 0) {
      threshold_break_ = std::strcmp(arg[i+1],"none") == 0
        ? std::numeric_limits<double>::infinity()
        : utils::numeric(FLERR,arg[i+1],false,lmp);
    } else {
      error->all(FLERR,"Unknown fix mlip/maxvol keyword: " + std::string(arg[i]));
    }
  }
  if (!std::isfinite(threshold_save_) || threshold_save_ < 0.0 ||
      std::isnan(threshold_break_) || threshold_break_ < threshold_save_)
    error->all(FLERR,"Fix mlip/maxvol requires 0 <= threshold-save <= threshold-break");

  scalar_flag = vector_flag = 1;
  size_vector = 2;
  global_freq = 1;  // Most recent sample is available also between sampling steps.
  extscalar = extvector = 0;
}

FixMLIPMaxVol::~FixMLIPMaxVol() = default;

int FixMLIPMaxVol::setmask()
{
  return FixConst::END_OF_STEP;
}

// All ranks call this after root-only work, before the next collective.
void FixMLIPMaxVol::root_error(std::string &message)
{
  int length = static_cast<int>(message.size());
  MPI_Bcast(&length,1,MPI_INT,0,world);
  message.resize(length);
  if (length) {
    MPI_Bcast(message.data(),length,MPI_CHAR,0,world);
    error->all(FLERR,"Fix mlip/maxvol: " + message);
  }
}

void FixMLIPMaxVol::init()
{
  if (std::strcmp(update->unit_style,"metal") != 0)
    error->all(FLERR,"Fix mlip/maxvol requires units metal");
  if (domain->dimension != 3)
    error->all(FLERR,"Fix mlip/maxvol requires dimension 3");
  if (!atom->tag_enable)
    error->all(FLERR,"Fix mlip/maxvol requires atom IDs");
  if (update->whichflag != 1)
    error->all(FLERR,"Fix mlip/maxvol supports run, not minimize");
  if (atom->ntypes != static_cast<int>(species_map_.size()))
    error->all(FLERR,"Fix mlip/maxvol atom type count changed; redefine the fix");
  if (ready_) return;

  std::string message;
  if (comm->me == 0) {
    try {
      io_ = std::make_unique<GPIO>();
      if (!io_->readmap(index_file_))
        throw std::runtime_error("cannot read index map: " + index_file_);
      GPSettings settings;
      if (!settings_file_.empty()) settings.Load(settings_file_);
      auto potential_settings = settings.ExtractSubSettings("pot");
      if (potential_type_.empty()) potential_type_ = potential_settings["pottype"];
      if (potential_type_.empty()) potential_type_ = settings["builder:pottype"];
      if (potential_type_.empty()) potential_type_ = "mtp";
      const std::vector<std::string> supported = {
        "mtp", "mtpnn1", "mtp_m", "mtp_les", "hmtp"};
      if (std::find(supported.begin(),supported.end(),potential_type_) == supported.end())
        throw std::runtime_error("unsupported MTP potential type: " + potential_type_);
      potential_settings.Modify("pottype",potential_type_);
      potential_settings.Modify("potfile",model_file_);
      potential_settings.Modify("typemap",index_file_);
      Potentials::potential_wrapper wrapper(potential_settings);
      model_.reset(wrapper.pot);
      wrapper.pot = nullptr;
      for (int species : species_map_)
        if (species < 0 || species >= model_->species_count ||
            !io_->getmap().count(species))
          throw std::runtime_error("type mapping is outside the model/index species range");
      selector_ = std::make_unique<ActiveSelector>();
      selector_->load_state(state_file_,index_file_);
      selector_->init_builder(*model_);
      if (!selector_->prepare_grading())
        throw std::runtime_error("MaxVol state has no usable active rows");
      // Append across independent jobs/restarts. Never truncate existing samples.
      output_.open(save_file_,std::ios::app);
      if (!output_) throw std::runtime_error("cannot open sample output: " + save_file_);
      if (!io_->seq_write_init(output_,"extxyz"))
        throw std::runtime_error("cannot initialize extxyz output");
    } catch (const std::exception &exception) {
      message = exception.what();
    }
  }
  root_error(message);
  ready_ = true;
}

void FixMLIPMaxVol::setup(int)
{
  // Refresh even after same-timestep geometry changes between run commands;
  // the setup evaluation is observational and never writes a duplicate frame.
  sample(false);
}

void FixMLIPMaxVol::end_of_step()
{
  sample(true);
}

void FixMLIPMaxVol::sample(bool scheduled)
{
  const int nlocal = atom->nlocal;
  if (atom->natoms <= 0 || atom->natoms > std::numeric_limits<int>::max() / 3)
    error->all(FLERR,"Fix mlip/maxvol requires 1 <= atom count <= INT_MAX/3");
  std::vector<int> counts, offsets, xyz_counts, xyz_offsets;
  std::vector<double> positions;
  std::vector<int> types;
  std::vector<tagint> tags;
  if (comm->me == 0) counts.resize(comm->nprocs);
  MPI_Gather(&nlocal,1,MPI_INT,counts.data(),1,MPI_INT,0,world);
  std::string message;
  int natoms = 0;
  if (comm->me == 0) {
    try {
      offsets.resize(comm->nprocs);
      xyz_counts.resize(comm->nprocs);
      xyz_offsets.resize(comm->nprocs);
      for (int rank = 0; rank < comm->nprocs; ++rank) {
        if (counts[rank] < 0 || counts[rank] > std::numeric_limits<int>::max()/3 - natoms)
          throw std::runtime_error("MPI gather size overflow");
        offsets[rank] = natoms;
        xyz_counts[rank] = 3 * counts[rank];
        xyz_offsets[rank] = 3 * natoms;
        natoms += counts[rank];
      }
      if (natoms != atom->natoms)
        throw std::runtime_error("global atom count disagrees with owned atoms");
      positions.resize(3 * natoms);
      types.resize(natoms);
      tags.resize(natoms);
    } catch (const std::exception &exception) {
      message = exception.what();
    }
  }
  root_error(message);
  MPI_Gatherv(nlocal ? atom->x[0] : nullptr,3*nlocal,MPI_DOUBLE,
              positions.data(),xyz_counts.data(),xyz_offsets.data(),MPI_DOUBLE,0,world);
  MPI_Gatherv(atom->type,nlocal,MPI_INT,types.data(),counts.data(),offsets.data(),MPI_INT,0,world);
  MPI_Gatherv(atom->tag,nlocal,MPI_LMP_TAGINT,tags.data(),counts.data(),offsets.data(),
              MPI_LMP_TAGINT,0,world);

  if (comm->me == 0) {
    try {
      std::vector<int> order(natoms);
      std::iota(order.begin(),order.end(),0);
      std::sort(order.begin(),order.end(),[&](int a, int b) { return tags[a] < tags[b]; });
      Atoms cfg;
      cfg.cell_ << domain->h[0], 0.0, 0.0,
                   domain->h[5], domain->h[1], 0.0,
                   domain->h[4], domain->h[3], domain->h[2];
      cfg.pbc_ = {{bool(domain->xperiodic),bool(domain->yperiodic),bool(domain->zperiodic)}};
      cfg.pos_.resize(natoms);
      cfg.chemmap_.resize(natoms);
      for (int i = 0; i < natoms; ++i) {
        const int j = order[i];
        if (tags[j] <= 0 || (i && tags[j] == tags[order[i-1]]))
          throw std::runtime_error("atom IDs must be positive and unique");
        if (types[j] < 1 || types[j] > static_cast<int>(species_map_.size()))
          throw std::runtime_error("invalid atom type");
        cfg.chemmap_[i] = species_map_[types[j]-1];
        for (int d = 0; d < 3; ++d) {
          cfg.pos_[i][d] = positions[3*j+d] - domain->boxlo[d];
          if (!std::isfinite(cfg.pos_[i][d]))
            throw std::runtime_error("non-finite atom position");
        }
      }
      grade_ = selector_->grade(cfg);
      if (!std::isfinite(grade_) || grade_ < 0.0)
        throw std::runtime_error("invalid extrapolation grade");
      if ((scheduled && grade_ > threshold_save_) || grade_ > threshold_break_) {
        // Builder evaluation may set predicted EFS; samples are unlabeled.
        cfg.clearEFS();
        cfg.charges_.clear();
        std::ostringstream grade_text;
        grade_text << std::setprecision(17) << grade_;
        cfg.features_["MV_grade"] = grade_text.str();
        cfg.features_["timestep"] = std::to_string(update->ntimestep);
        if (!io_->seq_write(output_,cfg)) throw std::runtime_error("cannot write sample");
        output_.flush();
        if (!output_) throw std::runtime_error("cannot flush sample output: " + save_file_);
        ++saved_;
      }
    } catch (const std::exception &exception) {
      message = exception.what();
    }
  }
  root_error(message);
  MPI_Bcast(&grade_,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&saved_,1,MPI_LMP_BIGINT,0,world);
  last_sample_ = update->ntimestep;
  if (grade_ > threshold_break_)
    error->all(FLERR,"Fix mlip/maxvol threshold-break exceeded; triggering configuration saved");
}

double FixMLIPMaxVol::compute_scalar() { return grade_; }

double FixMLIPMaxVol::compute_vector(int index)
{
  return index == 0 ? static_cast<double>(saved_) : static_cast<double>(last_sample_);
}
