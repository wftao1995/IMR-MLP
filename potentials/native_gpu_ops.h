#pragma once
#include <vector>

namespace Potentials::NativeGpu {
// Host staging views. Numerical operations below execute on the selected
// device; buffers contain owned centers and only their required ghosts.
struct Neighborhood {
    int nlocal=0,ntotal=0,edges=0;
    const int *offsets=nullptr,*neighbors=nullptr,*types=nullptr;
    const double *vectors=nullptr,*distances=nullptr;
};
struct Output {
    double *forces=nullptr,*energy=nullptr,*virial=nullptr;
};
void add(int device,int count,Output destination,Output source,bool include_energy=true);
void zbl(int device,Neighborhood,const std::vector<double> &atomic_numbers,
         double r0,double r1,const double *weights,Output);
struct Scale {
    // Child-major weights; derivative vectors follow the CSR edge order.
    std::vector<double> weights,derivatives,feature_gradient;
};
Scale scale(int device,Neighborhood,bool nearest,double r0,double r1,
            const std::vector<double> &boundaries);
void scale_response(int device,Neighborhood,int children,const Scale &,
                    const std::vector<double> &child_energies,Output);

}
