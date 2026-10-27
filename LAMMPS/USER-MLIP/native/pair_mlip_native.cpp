#include "pair_mlip_native.h"
#include "native_neighbor_order.h"

#include "atom.h"
#include "comm.h"
#include "domain.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "utils.h"
#include "potentials/native_inference.h"
#include "potentials/native_profile.h"
#include "tools/mtp_metadata.h"
#include "runtime/mtp/gpu-direct/gpu_backend.h"
#ifdef IMRMLP_GPU
#include "runtime/mtp/gpu-direct/gpu_mpi.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <numeric>
#include <stdexcept>

using namespace LAMMPS_NS;
using Potentials::NativeInferenceRequest;
using Potentials::NativeInferenceResult;

namespace {
std::array<double,6> symmetric_virial(const Eigen::Matrix3d &w)
{
  return {w(0,0),w(1,1),w(2,2),.5*(w(0,1)+w(1,0)),
    .5*(w(0,2)+w(2,0)),.5*(w(1,2)+w(2,1))};
}

}

// The LAMMPS skin list is reusable; the subset inside the model cutoff is not.
// Keep sorted skin candidates and refresh their geometry/filter on every call.
// In particular, a pair can cross the physical cutoff without reneighboring.
struct PairMLIPNative::Workspace {
  Atoms configuration;
  AnyNeighborhoods neighborhoods;
  std::vector<tagint> tags;
  std::vector<int> atom_types, candidate_offsets, candidates;
  std::vector<NativeNeighborOrder::ImageGroup> image_groups;
  std::array<double,6> cell{};
  std::array<int,3> periodic{};
  NeighList *source = nullptr;
  bigint last_build = -1;
  int owned = -1, total = -1;
  std::uint64_t topology_revision = 0;
  bool valid = false;
#ifdef IMRMLP_NATIVE_HOST_PROFILE
  std::uint64_t profile_call = 0;
  std::size_t profile_edges = 0;
  bool profile_rebuilt = false;
#endif
};

PairMLIPNative::PairMLIPNative(LAMMPS *lmp) : Pair(lmp)
{
  restartinfo = 1;
  one_coeff = 1;
  manybody_flag = 1;
  single_enable = 0;
  no_virial_fdotr_compute = 1;
  centroidstressflag = CENTROID_NOTAVAIL;
}

PairMLIPNative::~PairMLIPNative()
{
  if (copymode) return;
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);
  }
}

void PairMLIPNative::collective_error(const std::string &local_message)
{
  IMRMLP_NATIVE_PROFILE_SCOPE("pair.collective_error");
  int source = local_message.empty() ? comm->nprocs : comm->me;
  int first = comm->nprocs;
  MPI_Allreduce(&source,&first,1,MPI_INT,MPI_MIN,world);
  if (first == comm->nprocs) return;
  std::string message = comm->me == first ? local_message : std::string{};
  int length = static_cast<int>(message.size());
  MPI_Bcast(&length,1,MPI_INT,first,world);
  message.resize(length);
  MPI_Bcast(message.data(),length,MPI_CHAR,first,world);
  error->all(FLERR,"mlip/native rank " + std::to_string(first) + ": " + message);
}

void PairMLIPNative::settings(int narg, char **arg)
{
  if (ready_) error->all(FLERR,"Create a new mlip/native pair style before changing its model or options");
  pottype_.clear(); settings_file_.clear(); device_ = "cpu";
  requested_gpu_ = -1;
  if (narg % 2) error->all(FLERR,"mlip/native options require keyword/value pairs");
  for (int i = 0; i < narg; i += 2) {
    const std::string key = arg[i], value = arg[i+1];
    if (key == "device") {
      device_ = std::string(MLIPCore::GpuDirect::canonical_device(value));
      if (!MLIPCore::GpuDirect::valid_device(device_))
        error->all(FLERR,"mlip/native device must be cpu, cuda, hip, or dcu");
    } else if (key == "precision") {
      if (value != "fp64") error->all(FLERR,"mlip/native supports precision fp64 only");
    } else if (key == "pottype") pottype_ = value;
    else if (key == "settings") settings_file_ = std::filesystem::absolute(value).lexically_normal().string();
    else if (key == "gpu_id") {
      requested_gpu_ = value == "auto" ? -1 : utils::inumeric(FLERR,arg[i+1],false,lmp);
      if (requested_gpu_ < -1 || (requested_gpu_ == -1 && value != "auto"))
        error->all(FLERR,"mlip/native gpu_id must be auto or a nonnegative integer");
    } else error->all(FLERR,"Unknown mlip/native option: " + key);
  }
  if (pottype_.empty()) pottype_ = "auto";
  if (device_ == "cpu" && requested_gpu_ >= 0)
    error->all(FLERR,"mlip/native gpu_id requires a GPU device");
}

void PairMLIPNative::load_model()
{
  selected_gpu_ = -1;
  if (device_ != "cpu") {
#ifdef IMRMLP_GPU
    if (device_ != MLIPCore::GpuDirect::compiled_backend())
      error->all(FLERR,"mlip/native requested GPU backend does not match the linked IMR-MLP library");
    std::string message;
    if (!MLIPCore::GpuDirect::select_mpi_device(world,requested_gpu_,&selected_gpu_,&message))
      error->all(FLERR,message);
#else
    error->all(FLERR,"mlip/native GPU requested but this IMR-MLP build has no GPU backend; CPU fallback is disabled");
#endif
  }
  std::string message;
  try {
    Potentials::NativeInferenceOptions options;
    options.pottype = pottype_; options.settings_file = settings_file_;
    options.device = device_; options.device_id = selected_gpu_;
    model_ = std::make_unique<Potentials::NativeInference>(model_file_,options);
    workspace_ = std::make_unique<Workspace>();
    cutoff_ = model_->cutoff();
    local_ = model_->supports_local();
    channels_ = model_->latent_channels();
    has_charges_ = model_->has_charges() ? 1 : 0;
    if (!std::isfinite(cutoff_) || cutoff_ <= 0. || model_->species_count() <= 0 || channels_ < 0)
      throw std::runtime_error("invalid model dimensions or cutoff");
  } catch (const std::exception &e) { message = e.what(); }
  collective_error(message);
  std::array<int,4> contract{model_->species_count(),static_cast<int>(local_),channels_,has_charges_};
  auto lowest = contract, highest = contract;
  MPI_Allreduce(MPI_IN_PLACE,lowest.data(),4,MPI_INT,MPI_MIN,world);
  MPI_Allreduce(MPI_IN_PLACE,highest.data(),4,MPI_INT,MPI_MAX,world);
  if (lowest != highest) error->all(FLERR,"mlip/native model capabilities differ between MPI ranks");
  double cutoff_low = cutoff_, cutoff_high = cutoff_;
  MPI_Allreduce(MPI_IN_PLACE,&cutoff_low,1,MPI_DOUBLE,MPI_MIN,world);
  MPI_Allreduce(MPI_IN_PLACE,&cutoff_high,1,MPI_DOUBLE,MPI_MAX,world);
  collective_error(cutoff_low != cutoff_high ? "model cutoff differs between MPI ranks" : "");
  ready_ = true; evaluated_ = false;
  if (comm->me == 0) {
    const auto description = model_->description();
    const char *execution = local_ ? "owned-center domain decomposition" : "owned-center distributed long range";
    // IMR-MLP and LAMMPS may vendor different fmt ABI versions. Do not instantiate
    // LAMMPS's fmt-templated log helper from a IMR-MLP precompiled-header context.
    if (screen) fprintf(screen,"mlip/native: %s device=%s precision=fp64 execution=%s ranks=%d\n",
      description.c_str(),device_.c_str(),execution,comm->nprocs);
    if (logfile) fprintf(logfile,"mlip/native: %s device=%s precision=fp64 execution=%s ranks=%d\n",
      description.c_str(),device_.c_str(),execution,comm->nprocs);
  }
}

void PairMLIPNative::coeff(int narg, char **arg)
{
  if (ready_) error->all(FLERR,"mlip/native model is already loaded; create a new pair style to reload");
  if (atom->ntypes <= 0 || narg != 3 + atom->ntypes ||
      std::strcmp(arg[0],"*") || std::strcmp(arg[1],"*"))
    error->all(FLERR,"Use pair_coeff * * MODEL ELEMENT_OR_INDEX... with one mapping per atom type");
  model_file_ = std::filesystem::absolute(arg[2]).lexically_normal().string();
  load_model();
  std::string message;
  try {
    MTPMetadata::Header header;
    header.species_count = model_->species_count();
    auto elements = model_->elements();
    if (!elements.empty()) header.tag.set_elements(elements,header.species_count);
    std::vector<std::string> names(arg+3,arg+narg);
    species_map_ = MTPMetadata::map_types(header,names);
  } catch (const std::exception &e) { message = e.what(); }
  collective_error(message);
  if (!allocated) allocate();
}

void PairMLIPNative::allocate()
{
  const int n = atom->ntypes;
  memory->create(setflag,n+1,n+1,"mlip/native:setflag");
  memory->create(cutsq,n+1,n+1,"mlip/native:cutsq");
  for (int i = 1; i <= n; ++i)
    for (int j = 1; j <= n; ++j) {
      setflag[i][j] = 1;
      cutsq[i][j] = cutoff_*cutoff_;
    }
  allocated = 1;
}

void PairMLIPNative::init_style()
{
  if (!ready_ || species_map_.size() != static_cast<std::size_t>(atom->ntypes))
    error->all(FLERR,"mlip/native requires a loaded model and a complete species mapping");
  if (std::strcmp(update->unit_style,"metal") || domain->dimension != 3)
    error->all(FLERR,"mlip/native requires units metal and dimension 3");
  if (!force->newton_pair) error->all(FLERR,"mlip/native requires newton pair on");
  if (!atom->tag_enable) error->all(FLERR,"mlip/native requires atom IDs");
  neighbor->add_request(this,NeighConst::REQ_FULL);
}

double PairMLIPNative::init_one(int, int) { return cutoff_; }

void PairMLIPNative::set_geometry(Atoms &configuration) const
{
  configuration.cell_ << domain->h[0],0.,0.,domain->h[5],domain->h[1],0.,
    domain->h[4],domain->h[3],domain->h[2];
  configuration.pbc_ = {static_cast<bool>(domain->xperiodic),
    static_cast<bool>(domain->yperiodic),static_cast<bool>(domain->zperiodic)};
}

void PairMLIPNative::compute(int eflag, int vflag)
{
#ifdef IMRMLP_NATIVE_HOST_PROFILE
  Potentials::NativeHostProfileRecord profile;
  Potentials::NativeHostProfileFrame profile_frame(profile);
#endif
  {
  IMRMLP_NATIVE_PROFILE_SCOPE("pair.compute");
  ev_init(eflag,vflag);
  NativeInferenceRequest request;
  request.energy = eflag_global != 0;
  request.site_energy = eflag_atom != 0;
  request.virial = vflag_global != 0;
  request.site_virial = vflag_atom != 0;
  compute_local(request);
  if (!evaluated_) {
    Potentials::NativeInferenceStatistics statistics;
    std::string message;
    try { statistics = model_->statistics(); }
    catch (const std::exception &e) { message = e.what(); }
    collective_error(message);
    const std::array<int,3> local{atom->nlocal,atom->nghost,selected_gpu_};
    const std::array<unsigned long long,4> counters{statistics.neighbor_edges,
      statistics.model_evaluations,statistics.force_kernel_launches,statistics.model_device_bytes};
    std::vector<int> placement(3*comm->nprocs);
    std::vector<unsigned long long> all_counters(4*comm->nprocs);
    MPI_Allgather(local.data(),3,MPI_INT,placement.data(),3,MPI_INT,world);
    MPI_Allgather(counters.data(),4,MPI_UNSIGNED_LONG_LONG,all_counters.data(),4,MPI_UNSIGNED_LONG_LONG,world);
    if (comm->me == 0) for (int rank = 0; rank < comm->nprocs; ++rank) {
      for (FILE *output : {screen,logfile}) if (output)
        fprintf(output,"mlip/native rank=%d owned_atoms=%d ghost_atoms=%d gpu_id=%d "
          "neighbor_edges=%llu model_evaluations=%llu force_kernel_launches=%llu model_device_bytes=%llu\n",
          rank,placement[3*rank],placement[3*rank+1],placement[3*rank+2],
          all_counters[4*rank],all_counters[4*rank+1],all_counters[4*rank+2],all_counters[4*rank+3]);
    }
  }
  evaluated_ = true;
  }
#ifdef IMRMLP_NATIVE_HOST_PROFILE
  const auto &work = *workspace_;
  Potentials::NativeHostProfileMetadata metadata;
  metadata.rank = comm->me; metadata.owned = atom->nlocal;
  metadata.total = atom->nlocal+atom->nghost; metadata.step = update->ntimestep;
  metadata.call = workspace_->profile_call++;
  metadata.eflag = eflag; metadata.vflag = vflag;
  metadata.candidates = work.candidates.size(); metadata.active_edges = work.profile_edges;
  metadata.image_groups = work.image_groups.size(); metadata.topology_revision = work.topology_revision;
  metadata.rebuilt = work.profile_rebuilt;
  Potentials::write_native_host_profile(profile,metadata);
#endif
}

void PairMLIPNative::compute_local(const NativeInferenceRequest &request)
{
  const int nlocal = atom->nlocal, ntotal = nlocal + atom->nghost;
  NativeInferenceResult result;
  std::string message;
  try {
    auto &work = *workspace_;
    auto &configuration = work.configuration;
    const bool packed = device_ != "cpu";
    std::array<double,6> cell;
    std::copy_n(domain->h,6,cell.begin());
    const std::array<int,3> periodic{domain->xperiodic,domain->yperiodic,domain->zperiodic};
    bool rebuild = !work.valid || neighbor->ago == 0 || work.source != list ||
      work.last_build != neighbor->lastcall || work.owned != nlocal || work.total != ntotal ||
      work.cell != cell || work.periodic != periodic;
    {
    IMRMLP_NATIVE_PROFILE_SCOPE("pair.geometry");
    set_geometry(configuration);
    configuration.nlocal_ = nlocal; configuration.nghost_ = atom->nghost;
    configuration.have_ghost = atom->nghost != 0;
    configuration.pos_.resize(ntotal); configuration.chemmap_.resize(ntotal);
    work.tags.resize(ntotal); work.atom_types.resize(ntotal);
    for (int i = 0; i < ntotal; ++i) {
      if (atom->type[i] < 1 || atom->type[i] > atom->ntypes)
        throw std::runtime_error("atom type outside the species mapping");
      if (work.tags[i] != atom->tag[i] || work.atom_types[i] != atom->type[i]) {
        rebuild = true;
      }
      work.tags[i] = atom->tag[i]; work.atom_types[i] = atom->type[i];
      for (int k = 0; k < 3; ++k) {
        configuration.pos_[i][k] = atom->x[i][k] - domain->boxlo[k];
        // Reject before the image-order comparator sees non-finite values.
        if (packed && !std::isfinite(configuration.pos_[i][k]))
          throw std::runtime_error("native atom position must be finite");
      }
      configuration.chemmap_[i] = species_map_[atom->type[i]-1];
    }
    }
    if (list->inum != nlocal) throw std::runtime_error("full neighbor list must include every owned center");
#ifdef IMRMLP_NATIVE_HOST_PROFILE
    work.profile_rebuilt = rebuild;
#endif
    {
    IMRMLP_NATIVE_PROFILE_SCOPE("pair.candidates");
    if (rebuild) {
      IMRMLP_NATIVE_PROFILE_SCOPE("pair.candidates_rebuild");
      work.candidate_offsets.assign(nlocal+1,0);
      for (int ii = 0; ii < list->inum; ++ii) {
        const int i = list->ilist[ii];
        if (i < 0 || i >= nlocal || work.candidate_offsets[i+1])
          throw std::runtime_error("invalid or duplicate owned neighbor-list center");
        // Mark even an empty center as present during validation.
        work.candidate_offsets[i+1] = 1;
      }
      for (int i = 0; i < nlocal; ++i) {
        if (list->numneigh[i] < 0 || list->numneigh[i] > std::numeric_limits<int>::max()-work.candidate_offsets[i])
          throw std::runtime_error("native neighbor list exceeds the supported index range");
        work.candidate_offsets[i+1] = work.candidate_offsets[i]+list->numneigh[i];
      }
      work.candidates.resize(work.candidate_offsets.back());
      work.image_groups.clear();
      for (int i = 0; i < nlocal; ++i) {
        const auto begin = work.candidates.begin()+work.candidate_offsets[i];
        const auto end = work.candidates.begin()+work.candidate_offsets[i+1];
        for (int jj = 0; jj < list->numneigh[i]; ++jj) {
          const int j = list->firstneigh[i][jj] & NEIGHMASK;
          if (j < 0 || j >= ntotal) throw std::runtime_error("invalid neighbor atom index");
          begin[jj] = j;
        }
        std::sort(begin,end,[&](int a,int b) {
          return NativeNeighborOrder::less(a,b,atom->tag,atom->x);
        });
        NativeNeighborOrder::append_image_groups(work.candidates,work.candidate_offsets[i],
          work.candidate_offsets[i+1],atom->tag,work.image_groups);
      }
      work.source = list; work.last_build = neighbor->lastcall;
      work.cell = cell; work.periodic = periodic;
    } else {
      IMRMLP_NATIVE_PROFILE_SCOPE("pair.candidates_image_order");
      NativeNeighborOrder::refresh_images(work.candidates,work.image_groups,atom->x);
    }
    }
    std::size_t edge = 0;
    {
    IMRMLP_NATIVE_PROFILE_SCOPE("pair.filter_pack");
    if (!packed) {
      work.neighborhoods.resize(nlocal);
      for (int i = 0; i < nlocal; ++i) {
        auto &nbh = work.neighborhoods[i];
        nbh.centertype = configuration.chemmap_[i];
        nbh.indices_.clear(); nbh.chemmap_.clear(); nbh.pos_.clear(); nbh.distances_.clear();
        for (int jj = work.candidate_offsets[i]; jj < work.candidate_offsets[i+1]; ++jj) {
          const int j = work.candidates[jj];
          const Eigen::RowVector3d displacement = configuration.pos_[j]-configuration.pos_[i];
          const double distance = displacement.norm();
          if (!(distance < cutoff_)) continue;
          if (!(distance > 0.) || !std::isfinite(distance))
            throw std::runtime_error("native neighbor distance must be finite and positive");
          nbh.indices_.push_back(j); nbh.chemmap_.push_back(configuration.chemmap_[j]);
          nbh.pos_.push_back(displacement); nbh.distances_.push_back(distance);
          ++edge;
        }
        nbh.nbhcount_ = static_cast<int>(nbh.indices_.size());
      }
    }
    }
#ifdef IMRMLP_NATIVE_HOST_PROFILE
    work.profile_edges = edge;
#endif
    work.owned = nlocal; work.total = ntotal; work.valid = true;
  } catch (const std::exception &e) { message = e.what(); }
  // No rank may enter model collectives before all local geometry is valid.
  {
    IMRMLP_NATIVE_PROFILE_SCOPE("pair.geometry_sync");
    collective_error(message);
  }
  message.clear();
  try {
    auto &work = *workspace_;
    {
    IMRMLP_NATIVE_PROFILE_SCOPE("pair.inference");
    if (device_ == "cpu")
      result = model_->evaluate_distributed(work.configuration,work.neighborhoods,collectives(),request);
    else {
      Potentials::NativeCandidateNeighborhoodView view;
      view.owned_count = nlocal; view.total_count = ntotal;
      view.candidate_count = work.candidates.size();
      view.offsets = work.candidate_offsets.data(); view.neighbors = work.candidates.data();
      result = model_->evaluate_distributed_candidates(work.configuration,view,collectives(),request);
#ifdef IMRMLP_NATIVE_HOST_PROFILE
      const auto statistics = model_->statistics();
      work.profile_edges = statistics.neighbor_edges;
      work.topology_revision = statistics.candidate_topology_revision;
#endif
    }
    }
    IMRMLP_NATIVE_PROFILE_SCOPE("pair.result_validate");
    if (result.forces.rows() != ntotal ||
        (request.site_virial && result.site_virial.rows() != ntotal) ||
        (request.site_energy && result.site_energy.size() < nlocal) ||
        (has_charges_ && result.charges.size() < nlocal) ||
        (channels_ && (result.latents.rows() < nlocal || result.latents.cols() != channels_)))
      throw std::runtime_error("native local evaluator returned inconsistent output dimensions");
  } catch (const std::exception &e) { message = e.what(); }
  {
    IMRMLP_NATIVE_PROFILE_SCOPE("pair.result_sync");
    collective_error(message);
  }
  accumulate(result,ntotal,true);
}

void PairMLIPNative::accumulate(const NativeInferenceResult &result,int force_count,bool global)
{
  IMRMLP_NATIVE_PROFILE_SCOPE("pair.accumulate");
  for (int i = 0; i < force_count; ++i) {
    for (int k = 0; k < 3; ++k) atom->f[i][k] += result.forces(i,k);
    if (vflag_atom) for (int k = 0; k < 6; ++k) vatom[i][k] += result.site_virial(i,k);
  }
  if (eflag_atom) for (int i = 0; i < atom->nlocal; ++i) eatom[i] += result.site_energy(i);
  if (global && eflag_global) eng_vdwl += result.energy;
  if (global && vflag_global) {
    const auto tensor = symmetric_virial(result.virial);
    for (int k = 0; k < 6; ++k) virial[k] += tensor[k];
  }
  if (has_charges_) {
    charges_.resize(atom->nmax);
    for (int i = 0; i < atom->nlocal; ++i) charges_[i] = result.charges(i);
  }
  if (channels_) {
    latents_.resize(static_cast<std::size_t>(atom->nmax)*channels_);
    for (int i = 0; i < atom->nlocal; ++i)
      for (int c = 0; c < channels_; ++c) latents_[i*channels_+c] = result.latents(i,c);
  }
}

void PairMLIPNative::write_string(FILE *file,const std::string &value)
{
  const int length = static_cast<int>(value.size());
  fwrite(&length,sizeof(int),1,file);
  if (length) fwrite(value.data(),sizeof(char),length,file);
}

void PairMLIPNative::read_string(FILE *file,std::string &value)
{
  int length = 0;
  if (comm->me == 0) utils::sfread(FLERR,&length,sizeof(int),1,file,nullptr,error);
  MPI_Bcast(&length,1,MPI_INT,0,world);
  if (length < 0 || length > 1048576) error->all(FLERR,"Invalid mlip/native restart string length");
  value.resize(length);
  if (comm->me == 0 && length) utils::sfread(FLERR,value.data(),sizeof(char),length,file,nullptr,error);
  MPI_Bcast(value.data(),length,MPI_CHAR,0,world);
}

void PairMLIPNative::write_restart(FILE *file) { write_restart_settings(file); }
void PairMLIPNative::read_restart(FILE *file)
{
  read_restart_settings(file);
  if (!allocated) allocate();
}
void PairMLIPNative::write_restart_settings(FILE *file)
{
  const int version = 1, count = static_cast<int>(species_map_.size());
  // Keep version 1 restart layout for models without a total-charge constraint.
  const int charge_mode = 0;
  const double total_charge = 0.;
  fwrite(&version,sizeof(int),1,file);
  write_string(file,model_file_); write_string(file,settings_file_);
  write_string(file,pottype_); write_string(file,device_);
  fwrite(&requested_gpu_,sizeof(int),1,file);
  fwrite(&count,sizeof(int),1,file);
  fwrite(species_map_.data(),sizeof(int),count,file);
  fwrite(&charge_mode,sizeof(int),1,file);
  fwrite(&total_charge,sizeof(double),1,file);
}
void PairMLIPNative::read_restart_settings(FILE *file)
{
  int version = 0, count = 0, charge_mode = 0;
  double total_charge = 0.;
  if (comm->me == 0) utils::sfread(FLERR,&version,sizeof(int),1,file,nullptr,error);
  MPI_Bcast(&version,1,MPI_INT,0,world);
  if (version != 1) error->all(FLERR,"Unsupported mlip/native restart version");
  read_string(file,model_file_); read_string(file,settings_file_);
  read_string(file,pottype_); read_string(file,device_);
  if (comm->me == 0) {
    utils::sfread(FLERR,&requested_gpu_,sizeof(int),1,file,nullptr,error);
    utils::sfread(FLERR,&count,sizeof(int),1,file,nullptr,error);
  }
  MPI_Bcast(&requested_gpu_,1,MPI_INT,0,world);
  MPI_Bcast(&count,1,MPI_INT,0,world);
  if (count != atom->ntypes) error->all(FLERR,"mlip/native restart species map does not match atom types");
  species_map_.resize(count);
  if (comm->me == 0) {
    utils::sfread(FLERR,species_map_.data(),sizeof(int),count,file,nullptr,error);
    utils::sfread(FLERR,&charge_mode,sizeof(int),1,file,nullptr,error);
    utils::sfread(FLERR,&total_charge,sizeof(double),1,file,nullptr,error);
  }
  MPI_Bcast(species_map_.data(),count,MPI_INT,0,world);
  MPI_Bcast(&charge_mode,1,MPI_INT,0,world);
  MPI_Bcast(&total_charge,1,MPI_DOUBLE,0,world);
  if (charge_mode != 0 || !std::isfinite(total_charge) || requested_gpu_ < -1 ||
      pottype_.empty() || model_file_.empty() || !MLIPCore::GpuDirect::valid_device(device_))
    error->all(FLERR,"Invalid mlip/native restart settings");
  load_model();
  for (int species : species_map_)
    if (species < 0 || species >= model_->species_count())
      error->all(FLERR,"mlip/native restart species mapping is outside the model range");
}

void *PairMLIPNative::extract(const char *name,int &dim)
{
  dim = 0;
  if (!std::strcmp(name,"native_has_charges")) return &has_charges_;
  if (!std::strcmp(name,"native_latent_channels")) return &channels_;
  if (!std::strcmp(name,"native_charge")) { dim = 1; return evaluated_ ? charges_.data() : nullptr; }
  if (!std::strcmp(name,"native_latent")) { dim = 2; return evaluated_ ? latents_.data() : nullptr; }
  return nullptr;
}

double PairMLIPNative::memory_usage()
{
  double bytes = static_cast<double>(charges_.capacity()+latents_.capacity())*sizeof(double)
    + static_cast<double>(species_map_.capacity())*sizeof(int);
  if (workspace_) {
    const auto &work = *workspace_;
    bytes += static_cast<double>(work.tags.capacity())*sizeof(tagint);
    bytes += static_cast<double>(work.image_groups.capacity())*sizeof(NativeNeighborOrder::ImageGroup);
    bytes += static_cast<double>(work.atom_types.capacity()+work.candidate_offsets.capacity()+
      work.candidates.capacity()+work.configuration.chemmap_.capacity())*sizeof(int);
    bytes += static_cast<double>(work.configuration.pos_.capacity())*sizeof(Eigen::RowVector3d);
    for (int i = 0; i < work.neighborhoods.size(); ++i) {
      const auto &nbh = work.neighborhoods[i];
      bytes += sizeof(Neighborhood)+static_cast<double>(nbh.indices_.capacity()+nbh.chemmap_.capacity())*sizeof(int)
        +static_cast<double>(nbh.pos_.capacity())*sizeof(Eigen::RowVector3d)
        +static_cast<double>(nbh.distances_.capacity())*sizeof(double);
    }
  }
  return bytes;
}

Potentials::NativeCollectives PairMLIPNative::collectives()
{
  Potentials::NativeCollectives callbacks;
  callbacks.rank = comm->me; callbacks.size = comm->nprocs;
  callbacks.sum = [this](double *values,std::size_t count) {
    IMRMLP_NATIVE_PROFILE_SCOPE("mpi.sum");
    while (count) {
      const int block = static_cast<int>(std::min<std::size_t>(count,std::numeric_limits<int>::max()));
      MPI_Allreduce(MPI_IN_PLACE,values,block,MPI_DOUBLE,MPI_SUM,world);
      values += block; count -= block;
    }
  };
  return callbacks;
}
