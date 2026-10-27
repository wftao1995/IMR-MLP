// Copyright (c) 2026 IMR-MLP contributors
// SPDX-License-Identifier: GPL-2.0-only
// Adapted for the IMR-MLP public source layout, 2026-09-22.

#include "pair_mlip_transport_omp.h"

#include "atom.h"
#include "comm.h"
#include "neigh_list.h"
#include "suffix.h"
#include "update.h"
#if __has_include("fix_omp.h")
#include "fix_omp.h"
#include "omp_compat.h"
#include "thr_data.h"
#else
#include "OPENMP/fix_omp.h"
#include "OPENMP/omp_compat.h"
#include "OPENMP/thr_data.h"
#endif

#include <algorithm>
#include <vector>

using namespace LAMMPS_NS;

PairMLIPTransportOMP::PairMLIPTransportOMP(LAMMPS *lmp) :
    PairMLIPTransport(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
}

PairMLIPTransportOMP::~PairMLIPTransportOMP()
{
  if (!copymode) destroy_thread_scratch();
}

void PairMLIPTransportOMP::destroy_thread_scratch()
{
  for (auto &scratch : thread_scratch_) scratch.destroy(memory);
  thread_scratch_.clear();
}

void PairMLIPTransportOMP::allocate_thread_scratch()
{
  if (static_cast<int>(thread_scratch_.size()) == comm->nthreads) return;
  destroy_thread_scratch();
  thread_scratch_.resize(comm->nthreads);
  for (auto &scratch : thread_scratch_)
    scratch.allocate(max_mu, alpha_moments_count, rb_size,
                     max_alpha_index_basic, alpha_index_basic_count, memory);
}

void PairMLIPTransportOMP::init_style()
{
  PairMLIPTransport::init_style();
  allocate_thread_scratch();
}

void PairMLIPTransportOMP::compute(int eflag, int vflag)
{
  ev_init(eflag, vflag);

  const int inum = list->inum;
  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  allocate_thread_scratch();
  prepare_transport_storage(nall);

  int max_neighbors = 0;
  for (int ii = 0; ii < inum; ++ii)
    max_neighbors = std::max(max_neighbors, list->numneigh[list->ilist[ii]]);
  for (auto &scratch : thread_scratch_) {
    scratch.grow_neighbors(max_neighbors, alpha_index_basic_count, memory);
    scratch.reset_transport(nall, flux_enabled_,
                            static_cast<int>(interfaces_.size()));
  }
  std::vector<double> thread_energy(nthreads, 0.0);

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag,inum,nall,nthreads,thread_energy)
#endif
  {
    int ifrom, ito, tid;
    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    double *thread_eatom = eflag_atom ? eatom + static_cast<size_t>(tid) * nall : nullptr;
    double **thread_vatom = vflag_atom ? vatom + static_cast<size_t>(tid) * nall : nullptr;
    compute_range(ifrom, ito, thread_scratch_[tid], thr->get_f(),
                  thread_eatom, thread_vatom, thread_energy[tid]);

    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  }

  if (eflag_global)
    for (double value : thread_energy) eng_vdwl += value;
  std::vector<Scratch *> sources;
  sources.reserve(thread_scratch_.size());
  for (auto &scratch : thread_scratch_) sources.push_back(&scratch);
  reduce_transport_scratch(sources);

  transport_step_ = update->ntimestep;
  reverse_step_ = -1;
}
