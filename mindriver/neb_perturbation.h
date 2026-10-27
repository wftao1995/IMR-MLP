#pragma once
#include "neb.h"
#include <cstdint>

class NEBPathSymmetry;
struct NEBSeedOptions {
    std::string mode="sine1";
    std::uint64_t seed=0;
    int sign=1;
    double amplitude=.01, strain=.001, radius=0;
    int center=-1;
    std::vector<int> atoms;
};
struct NEBSeedResult {
    std::vector<Atoms> images;
    std::vector<int> selected_atoms;
    std::string status="generated", reason, subspace;
    double atomic_displacement=0,cell_deformation=0;
};
// Shared geometry-only helpers. Legacy entrypoints retain their exact RNG and
// scaling conventions; the search entrypoint never mutates its source driver.
class NEBPathSeeds {
public:
    static void initial(NEBDriver &,const NEBPathSymmetry *);
    static std::pair<double,double> release(NEBDriver &,const NEBPathSymmetry &,int,double,double);
    static NEBSeedResult generate(const NEBDriver &,const NEBSeedOptions &);
};
