#include "pair_harmonic_reference.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace LAMMPS_NS;

PairHarmonicReference::PairHarmonicReference(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 0;
  manybody_flag = 1;
  single_enable = 0;
  one_coeff = 1;
}

PairHarmonicReference::~PairHarmonicReference()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairHarmonicReference::allocate()
{
  const int n = atom->ntypes;
  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      setflag[i][j] = 0;
      cutsq[i][j] = 0.0;
    }
  allocated = 1;
}

void PairHarmonicReference::settings(int narg, char **arg)
{
  if (narg < 1 || narg > 2)
    error->all(FLERR, "Illegal pair_style harmonic/reference command");
  requested_storage_ = StorageMode::AUTO;
  if (narg == 2) {
    const std::string mode(arg[1]);
    if (mode == "auto") requested_storage_ = StorageMode::AUTO;
    else if (mode == "replicated") requested_storage_ = StorageMode::REPLICATED;
    else if (mode == "distributed") requested_storage_ = StorageMode::DISTRIBUTED;
    else error->all(FLERR, "Unknown harmonic/reference storage mode");
  }
  load_reference(arg[0]);
}

void PairHarmonicReference::coeff(int narg, char **arg)
{
  if (narg != 2 || std::string(arg[0]) != "*" || std::string(arg[1]) != "*")
    error->all(FLERR, "Incorrect args for pair coefficients");
  if (!allocated) allocate();
  for (int i = 1; i <= atom->ntypes; ++i)
    for (int j = i; j <= atom->ntypes; ++j) setflag[i][j] = 1;
}

void PairHarmonicReference::load_reference(const std::string &filename)
{
  std::ifstream input(filename);
  if (!input) error->all(FLERR, "Cannot open harmonic/reference file");
  std::string token;
  int version = 0;
  input >> token >> version;
  if (token != "TI_HARMONIC_REFERENCE" || version != 1)
    error->all(FLERR, "Unsupported harmonic/reference file header");
  input >> token;
  if (token != "units") error->all(FLERR, "Missing units in harmonic/reference file");
  input >> token;
  if (token != "metal") error->all(FLERR, "harmonic/reference currently requires metal units");
  input >> token >> reference_atoms_;
  if (token != "natoms" || reference_atoms_ <= 0)
    error->all(FLERR, "Invalid natoms in harmonic/reference file");
  constexpr double auto_distributed_bytes = 256.0 * 1024.0 * 1024.0;
  const double replicated_block_bytes = 9.0 * sizeof(double)
      * static_cast<double>(reference_atoms_) * static_cast<double>(reference_atoms_);
  distributed_ = requested_storage_ == StorageMode::DISTRIBUTED
      || (requested_storage_ == StorageMode::AUTO && comm->nprocs > 1
          && replicated_block_bytes > auto_distributed_bytes);
  local_row_begin_ = distributed_ ? reference_atoms_ * comm->me / comm->nprocs : 0;
  local_row_end_ = distributed_ ? reference_atoms_ * (comm->me + 1) / comm->nprocs
                                : reference_atoms_;
  input >> token >> energy0_;
  if (token != "energy0" || !std::isfinite(energy0_))
    error->all(FLERR, "Invalid energy0 in harmonic/reference file");
  input >> token;
  if (token != "cell") error->all(FLERR, "Missing cell in harmonic/reference file");
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j) input >> reference_cell_[3 * i + j];
  input >> token;
  const double determinant = reference_cell_[0]
      * (reference_cell_[4] * reference_cell_[8] - reference_cell_[5] * reference_cell_[7])
      - reference_cell_[1]
      * (reference_cell_[3] * reference_cell_[8] - reference_cell_[5] * reference_cell_[6])
      + reference_cell_[2]
      * (reference_cell_[3] * reference_cell_[7] - reference_cell_[4] * reference_cell_[6]);
  if (!input || token != "atoms" || determinant <= 0.0)
    error->all(FLERR, "Invalid cell or atoms section in harmonic/reference file");

  reference_positions_.assign(3 * reference_atoms_, 0.0);
  reference_types_.assign(reference_atoms_, 0);
  std::vector<bool> seen(reference_atoms_, false);
  for (int line = 0; line < reference_atoms_; ++line) {
    int tag = 0;
    int type = 0;
    input >> tag >> type;
    if (!input || tag <= 0 || tag > reference_atoms_ || type <= 0 || seen[tag - 1])
      error->all(FLERR, "Invalid atom entry in harmonic/reference file");
    seen[tag - 1] = true;
    reference_types_[tag - 1] = type;
    for (int a = 0; a < 3; ++a) input >> reference_positions_[3 * (tag - 1) + a];
  }

  int block_count = 0;
  input >> token >> block_count;
  if (!input || token != "blocks" || block_count < 0)
    error->all(FLERR, "Invalid blocks section in harmonic/reference file");
  using RowEntry = std::pair<int, Block>;
  std::vector<std::vector<RowEntry>> rows(reference_atoms_);
  int previous_i = -1;
  int previous_j = -1;
  for (int block_id = 0; block_id < block_count; ++block_id) {
    int i = 0;
    int j = 0;
    input >> i >> j;
    --i;
    --j;
    if (!input || i < 0 || j < i || j >= reference_atoms_
        || i < previous_i || (i == previous_i && j <= previous_j))
      error->all(FLERR, "Invalid, duplicate, or unordered FC block in harmonic/reference file");
    previous_i = i;
    previous_j = j;
    Block block{};
    for (double &value : block) input >> value;
    if (!input || !std::all_of(block.begin(), block.end(),
                               [](double value) { return std::isfinite(value); }))
      error->all(FLERR, "Invalid FC value in harmonic/reference file");
    if (!distributed_ || (i >= local_row_begin_ && i < local_row_end_))
      rows[i].emplace_back(j, block);
    if (i != j) {
      Block transpose{};
      for (int a = 0; a < 3; ++a)
        for (int b = 0; b < 3; ++b) transpose[3 * a + b] = block[3 * b + a];
      if (!distributed_ || (j >= local_row_begin_ && j < local_row_end_))
        rows[j].emplace_back(i, transpose);
    }
  }
  input >> token;
  if (!input || token != "end") error->all(FLERR, "Missing end marker in harmonic/reference file");
  input >> token;
  if (input) error->all(FLERR, "Unexpected trailing data in harmonic/reference file");
  row_offsets_.assign(reference_atoms_ + 1, 0);
  columns_.clear();
  blocks_.clear();
  for (int i = 0; i < reference_atoms_; ++i) {
    std::sort(rows[i].begin(), rows[i].end(),
              [](const RowEntry &lhs, const RowEntry &rhs) { return lhs.first < rhs.first; });
    row_offsets_[i + 1] = row_offsets_[i] + rows[i].size();
    for (auto &[column, block] : rows[i]) {
      columns_.push_back(column);
      blocks_.push_back(std::move(block));
    }
  }
  displacements_.assign(3 * reference_atoms_, 0.0);
  if (distributed_) {
    responses_local_.assign(3 * (local_row_end_ - local_row_begin_), 0.0);
    responses_.assign(3 * reference_atoms_, 0.0);
    response_counts_.resize(comm->nprocs);
    response_displacements_.resize(comm->nprocs);
    for (int rank = 0; rank < comm->nprocs; ++rank) {
      const int begin = reference_atoms_ * rank / comm->nprocs;
      const int end = reference_atoms_ * (rank + 1) / comm->nprocs;
      response_counts_[rank] = 3 * (end - begin);
      response_displacements_[rank] = 3 * begin;
    }
  } else {
    responses_local_.clear();
    responses_.clear();
    response_counts_.clear();
    response_displacements_.clear();
  }
  loaded_ = true;
}

void PairHarmonicReference::validate_system()
{
  if (!loaded_) error->all(FLERR, "pair_style harmonic/reference has no reference file");
  if (!atom->tag_enable) error->all(FLERR, "pair_style harmonic/reference requires atom IDs");
  if (atom->natoms != reference_atoms_)
    error->all(FLERR, "LAMMPS atom count differs from harmonic/reference file");
  if (!domain->xperiodic || !domain->yperiodic || !domain->zperiodic)
    error->all(FLERR, "pair_style harmonic/reference requires periodic boundaries");
  const std::array<double, 9> cell{
      domain->xprd, 0.0, 0.0,
      domain->xy, domain->yprd, 0.0,
      domain->xz, domain->yz, domain->zprd};
  double difference2 = 0.0;
  double scale2 = 0.0;
  for (int i = 0; i < 9; ++i) {
    difference2 += (cell[i] - reference_cell_[i]) * (cell[i] - reference_cell_[i]);
    scale2 += reference_cell_[i] * reference_cell_[i];
  }
  if (std::sqrt(difference2) > 1e-8 * std::max(1.0, std::sqrt(scale2)))
    error->all(FLERR, "LAMMPS box differs from harmonic/reference cell");
  for (int i = 0; i < atom->nlocal; ++i) {
    const tagint tag = atom->tag[i];
    if (tag <= 0 || tag > reference_atoms_ || atom->type[i] != reference_types_[tag - 1])
      error->all(FLERR, "LAMMPS atom ID/type differs from harmonic/reference file");
  }
  reference_masses_.resize(reference_atoms_);
  total_mass_ = 0.0;
  for (int id = 0; id < reference_atoms_; ++id) {
    const double mass = atom->mass[reference_types_[id]];
    if (!std::isfinite(mass) || mass <= 0.0)
      error->all(FLERR, "pair_style harmonic/reference requires positive atom masses");
    reference_masses_[id] = mass;
    total_mass_ += mass;
  }
}

void PairHarmonicReference::init_style()
{
  if (!allocated) allocate();
  validate_system();
}

double PairHarmonicReference::init_one(int, int)
{
  return 0.0;
}

void PairHarmonicReference::evaluate_row(int i, double *response) const
{
  response[0] = response[1] = response[2] = 0.0;
  for (int entry = row_offsets_[i]; entry < row_offsets_[i + 1]; ++entry) {
    const auto &block = blocks_[entry];
    const double *u = displacements_.data() + 3 * columns_[entry];
    response[0] += block[0] * u[0] + block[1] * u[1] + block[2] * u[2];
    response[1] += block[3] * u[0] + block[4] * u[1] + block[5] * u[2];
    response[2] += block[6] * u[0] + block[7] * u[1] + block[8] * u[2];
  }
}

void PairHarmonicReference::compute(int eflag, int vflag)
{
  if (eflag || vflag) ev_setup(eflag, vflag);
  else evflag = vflag_fdotr = eflag_global = eflag_atom = 0;

  std::fill(displacements_.begin(), displacements_.end(), 0.0);
#pragma omp parallel for schedule(static)
  for (int i = 0; i < atom->nlocal; ++i) {
    const int id = atom->tag[i] - 1;
    double closest[3];
    domain->closest_image(reference_positions_.data() + 3 * id, atom->x[i], closest);
    for (int a = 0; a < 3; ++a)
      displacements_[3 * id + a] = closest[a] - reference_positions_[3 * id + a];
  }
  MPI_Allreduce(MPI_IN_PLACE, displacements_.data(), 3 * reference_atoms_,
                MPI_DOUBLE, MPI_SUM, world);

  double center0 = 0.0;
  double center1 = 0.0;
  double center2 = 0.0;
#pragma omp parallel for reduction(+:center0,center1,center2) schedule(static)
  for (int id = 0; id < reference_atoms_; ++id) {
    const double mass = reference_masses_[id];
    center0 += mass * displacements_[3 * id];
    center1 += mass * displacements_[3 * id + 1];
    center2 += mass * displacements_[3 * id + 2];
  }
  center0 /= total_mass_;
  center1 /= total_mass_;
  center2 /= total_mass_;
#pragma omp parallel for schedule(static)
  for (int id = 0; id < reference_atoms_; ++id) {
    displacements_[3 * id] -= center0;
    displacements_[3 * id + 1] -= center1;
    displacements_[3 * id + 2] -= center2;
  }

  if (distributed_) {
#pragma omp parallel for schedule(static)
    for (int i = local_row_begin_; i < local_row_end_; ++i)
      evaluate_row(i, responses_local_.data() + 3 * (i - local_row_begin_));
    MPI_Allgatherv(responses_local_.data(), static_cast<int>(responses_local_.size()), MPI_DOUBLE,
                   responses_.data(), response_counts_.data(), response_displacements_.data(),
                   MPI_DOUBLE, world);

    if (eflag_global) {
      double local_energy = comm->me == 0 ? energy0_ : 0.0;
#pragma omp parallel for reduction(+:local_energy) schedule(static)
      for (int i = local_row_begin_; i < local_row_end_; ++i)
        for (int a = 0; a < 3; ++a)
          local_energy += 0.5 * displacements_[3 * i + a]
                        * responses_local_[3 * (i - local_row_begin_) + a];
      eng_vdwl += local_energy;
    }
#pragma omp parallel for schedule(static)
    for (int local = 0; local < atom->nlocal; ++local) {
      const int i = atom->tag[local] - 1;
      double harmonic_energy = 0.0;
      for (int a = 0; a < 3; ++a) {
        if (eflag_atom)
          harmonic_energy += 0.5 * displacements_[3 * i + a] * responses_[3 * i + a];
        atom->f[local][a] -= responses_[3 * i + a];
      }
      if (eflag_atom)
        eatom[local] += energy0_ / static_cast<double>(reference_atoms_)
                      + harmonic_energy;
    }
  } else {
    double local_energy = eflag_global
        ? energy0_ * static_cast<double>(atom->nlocal)
            / static_cast<double>(reference_atoms_)
        : 0.0;
#pragma omp parallel for reduction(+:local_energy) schedule(static)
    for (int local = 0; local < atom->nlocal; ++local) {
      const int i = atom->tag[local] - 1;
      double response[3];
      evaluate_row(i, response);
      double harmonic_energy = 0.0;
      for (int a = 0; a < 3; ++a) {
        if (eflag_global || eflag_atom)
          harmonic_energy += 0.5 * displacements_[3 * i + a] * response[a];
        atom->f[local][a] -= response[a];
      }
      if (eflag_global) local_energy += harmonic_energy;
      if (eflag_atom)
        eatom[local] += energy0_ / static_cast<double>(reference_atoms_)
                      + harmonic_energy;
    }
    if (eflag_global) eng_vdwl += local_energy;
  }
  if (vflag_fdotr) virial_fdotr_compute();
}

double PairHarmonicReference::memory_usage()
{
  double bytes = displacements_.capacity() * sizeof(double);
  bytes += responses_local_.capacity() * sizeof(double);
  bytes += responses_.capacity() * sizeof(double);
  bytes += response_counts_.capacity() * sizeof(int);
  bytes += response_displacements_.capacity() * sizeof(int);
  bytes += reference_positions_.capacity() * sizeof(double);
  bytes += reference_types_.capacity() * sizeof(int);
  bytes += reference_masses_.capacity() * sizeof(double);
  bytes += row_offsets_.capacity() * sizeof(int);
  bytes += columns_.capacity() * sizeof(int);
  bytes += blocks_.capacity() * sizeof(Block);
  return bytes;
}
