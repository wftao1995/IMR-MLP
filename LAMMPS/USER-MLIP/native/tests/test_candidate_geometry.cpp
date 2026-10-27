#include "../native_neighbor_geometry.h"
#include "potentials/native_inference.h"
#include "unitest/gpm_fixture.h"
#include <mpi.h>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>

namespace {
#if defined(__GNUC__) && !defined(__clang__)
#define NOINLINE __attribute__((noinline,noipa))
#elif defined(__clang__)
#define NOINLINE __attribute__((noinline))
#else
#define NOINLINE
#endif
NOINLINE double pair_distance(const Eigen::RowVector3d &a,const Eigen::RowVector3d &b)
{
  return LAMMPS_NS::NativeNeighborGeometry::distance(a,b);
}
NOINLINE double packed_squared(const Eigen::RowVector3d &a,const Eigen::RowVector3d &b)
{
  double squared=0.;
  for(int d=0;d<3;++d){const double delta=b[d]-a[d];squared+=delta*delta;}
  return squared;
}
}
int main(int argc,char **argv)
{
  MPI_Init(&argc,&argv);
  int status=0;
  try {
    const auto path=std::filesystem::temp_directory_path()/
      ("native-candidate-geometry-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".mtp");
    struct Cleanup{std::filesystem::path file;~Cleanup(){std::filesystem::remove(file);}} cleanup{path};
    Potentials::PotMTP model;load_swap_fixture(model,2,false);model.Save(path.string());
    Potentials::NativeInferenceOptions options;options.pottype="mtp";
    Potentials::NativeInference inference(path.string(),options);
    Atoms atoms;atoms.pos_.resize(2);atoms.chemmap_={0,1};atoms.cell_=32.*Eigen::Matrix3d::Identity();
    const int offsets[]={0,1},neighbors[]={1};
    const Potentials::NativeCandidateNeighborhoodView view{1,2,1,offsets,neighbors};
    Potentials::NativeInferenceRequest request;
    request.energy=request.forces=request.site_energy=request.virial=request.site_virial=request.charges=request.latents=false;
    std::size_t cases=0,decisions=0;
    const auto check=[&](const Eigen::RowVector3d &a,const Eigen::RowVector3d &b) {
      const double distance=pair_distance(a,b),squared=packed_squared(a,b);
      atoms.pos_[0]=a;atoms.pos_[1]=b;
      std::array<double,3> cutoffs{std::nextafter(distance,0.),distance,
        std::nextafter(distance,std::numeric_limits<double>::infinity())};
      if(!a.allFinite() || !b.allFinite() || !std::isfinite(distance))cutoffs={1.,5.,std::numeric_limits<double>::max()};
      for(const double cutoff:cutoffs) {
        if(!(cutoff>0.) || !std::isfinite(cutoff))continue;
        inference.potential().cutoff=cutoff;
        const bool active=distance<cutoff;
        const bool bad=!a.allFinite() || !b.allFinite() ||
          (active && (!(distance>0.) || !std::isfinite(distance) || !(squared>0.) || !std::isfinite(squared)));
        bool rejected=false;
        try {inference.evaluate_distributed_candidates(atoms,view,{},request);}
        catch(const std::exception &){rejected=true;}
        if(rejected!=bad || (!bad && inference.statistics().neighbor_edges!=static_cast<std::size_t>(active)))
          throw std::runtime_error("candidate core differs from Pair cutoff or public packed error classification");
        ++decisions;
      }
      ++cases;
    };
    const Eigen::RowVector3d zero(0.,0.,0.);
    check(zero,{2.4515835088113644,3.7189776864989215,0.});
    for(double value:{0.,-0.,std::numeric_limits<double>::denorm_min(),1e-300,1e-160,1e-100,1.,1e100,1e160,1e300,
        std::numeric_limits<double>::max(),std::numeric_limits<double>::infinity(),-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
      check(zero,{value,0.,0.});check(zero,{value,value,value});check({-value,0.,value},{value,value,-value});
    }
    std::mt19937_64 random(20261024);
    for(int i=0;i<1000000;++i) {
      Eigen::RowVector3d a,b;
      for(int k=0;k<3;++k) {
        const double fraction=static_cast<double>(random()>>11)*0x1p-53-.5;
        const int exponent=static_cast<int>(random()%2098)-1074;
        a[k]=std::ldexp(fraction,exponent);
        b[k]=i%2?std::nextafter(a[k],0.):std::ldexp(static_cast<double>(random()>>11)*0x1p-53-.5,exponent);
      }
      check(a,b);
    }
    std::cout<<"Candidate geometry across executable/core boundary: "<<cases<<" vectors, "<<decisions
      <<" exact cutoff/error classifications passed\n";
  } catch(const std::exception &e){std::cerr<<e.what()<<'\n';status=1;}
  MPI_Finalize();return status;
}
